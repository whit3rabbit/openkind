//! HTTP transport execution, retry loop, and header management.

use std::time::{Duration, Instant};

use reqwest::Method;
use serde::de::DeserializeOwned;

use super::core::Client;
use super::options::{RequestOptions, RETRY_COUNT_HEADER};
use crate::error::{parse_retry_after, ApiError, Error, REQUEST_ID_HEADER};

/// Immutable description of one logical request, shared by every retry
/// attempt (only the retry count and headers differ between attempts).
pub(crate) struct RequestDesc<'a> {
    pub(crate) method: Method,
    pub(crate) url: String,
    pub(crate) endpoint: String,
    pub(crate) body: Option<&'a serde_json::Value>,
    pub(crate) timeout: Duration,
}

impl Client {
    /// Send one logical request (possibly several HTTP attempts) and decode
    /// the 2xx body as `T`.
    pub(crate) async fn send_json<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<&serde_json::Value>,
        opts: &RequestOptions,
    ) -> Result<T, Error> {
        let policy = opts.retry.as_ref().unwrap_or(&self.inner.retry);
        let endpoint = format!("{method} {path}");
        let desc = RequestDesc {
            method,
            url: format!("{}{}", self.inner.base_url, path),
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
        let mut request = self
            .inner
            .http
            .request(desc.method.clone(), &desc.url)
            .headers(headers)
            .timeout(desc.timeout);
        if retries > 0 {
            // Lets the server observe (and metrics count) client retries.
            request = request.header(RETRY_COUNT_HEADER, retries.to_string());
        }
        if let Some(body) = desc.body {
            request = request.json(body);
        }

        let response = request
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
        let bytes = response
            .bytes()
            .await
            .map_err(|e| classify_transport_error(e, desc.timeout))?;

        if !status.is_success() {
            return Err(Error::Api(Box::new(ApiError::from_response(
                status.as_u16(),
                request_id,
                retry_after,
                &bytes,
                desc.endpoint.clone(),
            ))));
        }

        serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
            status: status.as_u16(),
            body_excerpt: String::from_utf8_lossy(&bytes).chars().take(200).collect(),
            source,
        })
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
        Error::Connection(error)
    }
}
