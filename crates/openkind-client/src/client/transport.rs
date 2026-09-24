//! HTTP transport execution, retry loop, and header management.

use std::time::{Duration, Instant};

use reqwest::header::CONTENT_TYPE;
use reqwest::Method;
use serde::de::DeserializeOwned;

use super::core::Client;
use super::options::{
    RequestOptions, CLOUDFLARE_RUN_PATH, HEALTH_PATH, MODELS_PATH, RETRY_COUNT_HEADER,
    SYSTEM_ONE_PATH,
};
use crate::error::{parse_retry_after, ApiError, Error, REQUEST_ID_HEADER};

/// Maximum number of response-body bytes buffered by the client (8 MiB).
///
/// The limit applies to successful and error responses, including chunked
/// responses without a `Content-Length` header.
pub const MAX_RESPONSE_BODY_SIZE: usize = 8 * 1024 * 1024;

/// Immutable description of one logical request, shared by every retry
/// attempt (only the retry count and headers differ between attempts).
pub(crate) struct RequestDesc<'a> {
    pub(crate) method: Method,
    pub(crate) url: &'a reqwest::Url,
    pub(crate) endpoint: &'static str,
    pub(crate) body: Option<Vec<u8>>,
    pub(crate) timeout: Duration,
}

impl Client {
    /// Send one logical request (possibly several HTTP attempts) and decode
    /// the 2xx body as `T`.
    pub(crate) async fn send_json<T: DeserializeOwned>(
        &self,
        method: Method,
        url: &reqwest::Url,
        path: &'static str,
        body: Option<Vec<u8>>,
        opts: &RequestOptions,
    ) -> Result<T, Error> {
        let policy = opts.retry.as_ref().unwrap_or(&self.inner.retry);
        let endpoint: &'static str = match path {
            SYSTEM_ONE_PATH => "POST /v1/systemone",
            CLOUDFLARE_RUN_PATH => "POST /ai/run",
            MODELS_PATH => "GET /v1/models",
            HEALTH_PATH => "GET /health",
            _ => "GET /",
        };
        let desc = RequestDesc {
            method,
            url,
            endpoint,
            body,
            timeout: opts.timeout.unwrap_or(self.inner.timeout),
        };
        let start = Instant::now();

        let mut retries: u32 = 0;
        loop {
            match self.attempt_once::<T>(&desc, opts, retries).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    if retries >= policy.max_retries || !policy.is_retryable(&error) {
                        return Err(error);
                    }
                    let delay = policy.delay_for(&error, retries);
                    if let Some(budget) = policy.total_timeout {
                        // Mirrors tenacity's stop_before_delay: don't start a
                        // retry whose wait would exceed the remaining budget.
                        if start.elapsed() + delay >= budget {
                            return Err(error);
                        }
                    }
                    retries += 1;
                    tracing::debug!(
                        endpoint = %desc.endpoint,
                        retry = retries,
                        ?delay,
                        error = %error,
                        "retrying request"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn attempt_once<T: DeserializeOwned>(
        &self,
        desc: &RequestDesc<'_>,
        opts: &RequestOptions,
        retries: u32,
    ) -> Result<T, Error> {
        // Per-call extras replace same-name defaults (HeaderMap::insert
        // semantics), matching the Python SDK's header merge. The Python SDK
        // re-sets the protocol headers after merging user input, so they
        // cannot be overridden per call; the retry-count header is client-
        // managed outright, and content-type belongs to the JSON body.
        let mut headers = self.inner.base_headers.clone();
        if let Some(extra) = &opts.extra_headers {
            for (name, value) in extra {
                if is_user_overridable(name.as_str(), desc.body.is_some()) {
                    headers.insert(name.clone(), value.clone());
                }
            }
        }

        // Direct HTTP/1.1 transport for plain-http targets. A redirect
        // response falls through to the reqwest path below so the whole
        // chain is followed exactly as reqwest would.
        if let Some(pool) = self.inner.fast_h1.as_ref() {
            let target = request_target(desc.url);
            let wire = super::http1::encode_request(
                desc.method.as_str(),
                &target,
                pool.authority(),
                &headers,
                desc.body.as_deref(),
                (retries > 0).then_some(retries),
            );
            match super::http1::exchange(pool, &wire, MAX_RESPONSE_BODY_SIZE, desc.timeout).await {
                Ok(Ok(response)) => {
                    let retry_after = if (200..300).contains(&response.status) {
                        None
                    } else {
                        crate::error::parse_retry_after_with(|name| response.header(name))
                    };
                    return finish_response(
                        response.status,
                        response.header(REQUEST_ID_HEADER).map(str::to_owned),
                        retry_after,
                        response.body,
                        desc,
                    );
                }
                Ok(Err(super::http1::FastError::Redirect)) => {}
                Ok(Err(error)) => {
                    return Err(super::http1::classify_fast_error(
                        error,
                        desc.timeout,
                        MAX_RESPONSE_BODY_SIZE,
                    ))
                }
                Err(timeout_error) => return Err(timeout_error),
            }
        }

        let mut request = self
            .inner
            .http
            .request(desc.method.clone(), desc.url.clone())
            .headers(headers)
            .timeout(desc.timeout);
        if retries > 0 {
            // Lets the server observe (and metrics count) client retries.
            request = request.header(RETRY_COUNT_HEADER, retries.to_string());
        }
        if let Some(body) = &desc.body {
            // Byte-identical to `.json(body)`: the JSON was serialized once
            // by the caller and is stamped with the same content type.
            request = request
                .header(CONTENT_TYPE, "application/json")
                .body(body.clone());
        }

        let mut response = request
            .send()
            .await
            .map_err(|e| classify_transport_error(e, desc.timeout))?;
        let status = response.status();
        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let retry_after = if status.is_success() {
            None
        } else {
            parse_retry_after(response.headers())
        };
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BODY_SIZE as u64)
        {
            return Err(Error::ResponseTooLarge {
                limit: MAX_RESPONSE_BODY_SIZE,
            });
        }

        // Do not trust Content-Length as the enforcement mechanism: it may be
        // absent (for example, for chunked bodies) or incorrect. Count bytes
        // while consuming the stream and reject before extending the buffer.
        // When present, it still sizes the buffer so the body is not
        // reallocated while growing.
        let mut bytes = Vec::with_capacity(
            response
                .content_length()
                .map(|length| (length as usize).min(MAX_RESPONSE_BODY_SIZE))
                .unwrap_or(0),
        );
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| classify_transport_error(e, desc.timeout))?
        {
            if chunk.len() > MAX_RESPONSE_BODY_SIZE - bytes.len() {
                return Err(Error::ResponseTooLarge {
                    limit: MAX_RESPONSE_BODY_SIZE,
                });
            }
            bytes.extend_from_slice(&chunk);
        }

        if !status.is_success() {
            return Err(Error::Api(Box::new(ApiError::from_response(
                status.as_u16(),
                request_id,
                retry_after,
                &bytes,
                desc.endpoint.to_string(),
            ))));
        }

        finish_response(status.as_u16(), request_id, retry_after, bytes, desc)
    }
}

/// Shared response tail: non-2xx maps to [`ApiError`] (with retry-after and
/// request-id extracted by the caller), 2xx decodes as `T`.
fn finish_response<T: DeserializeOwned>(
    status: u16,
    request_id: Option<String>,
    retry_after: Option<std::time::Duration>,
    bytes: Vec<u8>,
    desc: &RequestDesc<'_>,
) -> Result<T, Error> {
    if !(200..300).contains(&status) {
        return Err(Error::Api(Box::new(ApiError::from_response(
            status,
            request_id,
            retry_after,
            &bytes,
            desc.endpoint.to_string(),
        ))));
    }
    serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
        status,
        body_excerpt: String::from_utf8_lossy(&bytes).chars().take(200).collect(),
        source,
    })
}

/// Request target (path plus optional query) for the HTTP/1.1 request line.
fn request_target(url: &reqwest::Url) -> String {
    match url.query() {
        Some(query) => format!("{}?{}", url.path(), query),
        None => url.path().to_owned(),
    }
}

/// Headers user code may not override (the Python SDK's protected set):
/// SDK identification/protocol headers always win, the retry-count header is
/// client-managed, and content-type belongs to the JSON body.
pub(crate) fn is_user_overridable(name: &str, has_body: bool) -> bool {
    match name {
        "authorization"
        | "accept"
        | "user-agent"
        | "x-typesafe-sdk"
        | "x-typesafe-runtime"
        | "x-typesafe-retry-count" => false,
        "content-type" => !has_body,
        _ => true,
    }
}

pub(crate) fn classify_transport_error(error: reqwest::Error, timeout: Duration) -> Error {
    if error.is_timeout() {
        Error::Timeout { timeout }
    } else {
        Error::Connection(Box::new(error))
    }
}
