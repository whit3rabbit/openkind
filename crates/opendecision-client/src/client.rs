//! The async [`Client`] for the opendecision / TypeSafe SystemOne HTTP API.
//!
//! - **Transports**: `POST /v1/systemone` (canonical) and `GET /v1/models`;
//!   `GET /health` for liveness. Wire-compatible with `https://api.typesafe.ai`,
//!   so the same client targets either the hosted service or a local
//!   `opendecisiond` daemon by swapping `base_url`.
//! - **Retries**: automatic, per [`RetryPolicy`] — 429/529 carry
//!   `Retry-After` and are honored; transient 5xx, connection, and timeout
//!   errors back off exponentially.
//! - **Errors**: typed per [`crate::error`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use opendecision_core::{ModelsResponse, Question, State, SystemRequest, SystemResponse};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION};
use reqwest::Method;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{parse_retry_after, ApiError, Error, REQUEST_ID_HEADER};
use crate::retry::RetryPolicy;

/// Default API base URL (the TypeSafe hosted service). Override with
/// `base_url` for a local `opendecisiond` daemon.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";

/// Default model alias when none is set explicitly or via the environment.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Default per-attempt HTTP timeout (matches the Python SDK's 10s default).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) const SYSTEM_ONE_PATH: &str = "/v1/systemone";
pub(crate) const MODELS_PATH: &str = "/v1/models";
pub(crate) const HEALTH_PATH: &str = "/health";

pub(crate) const SDK_NAME: &str = "opendecision-client";

const USER_AGENT_HEADER: &str = "user-agent";
const SDK_HEADER: &str = "x-typesafe-sdk";
const RUNTIME_HEADER: &str = "x-typesafe-runtime";
/// Sent on retry attempts (value = retry number), mirroring the Python
/// SDK's `X-TypeSafe-Retry-Count` so servers can observe client retries.
const RETRY_COUNT_HEADER: &str = "x-typesafe-retry-count";

/// Response of `GET /health` — liveness probe, no authentication required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Health {
    /// Service status string (currently `"ok"`).
    pub status: String,
}

/// Conversion into a Jev [`State`], letting [`Client::system_one`] accept
/// plain strings, structured objects, arrays, or a ready-made `State` —
/// mirroring the Python SDK's `str | dict | list` state parameter.
///
/// (A blanket `From<&str> for State` cannot be provided because both types
/// are foreign to this crate.)
pub trait IntoState {
    /// Convert into the wire [`State`] type.
    fn into_state(self) -> State;
}

impl IntoState for State {
    fn into_state(self) -> State {
        self
    }
}

impl IntoState for &str {
    fn into_state(self) -> State {
        State::Text(self.to_owned())
    }
}

impl IntoState for String {
    fn into_state(self) -> State {
        State::Text(self)
    }
}

impl IntoState for &String {
    fn into_state(self) -> State {
        State::Text(self.clone())
    }
}

impl IntoState for serde_json::Map<String, serde_json::Value> {
    fn into_state(self) -> State {
        State::Object(self)
    }
}

impl IntoState for Vec<serde_json::Value> {
    fn into_state(self) -> State {
        State::Array(self)
    }
}

/// Per-call overrides applied on top of the client's defaults.
///
/// Passed to the `*_with` method variants:
///
/// ```
/// use std::time::Duration;
/// use opendecision_client::{Client, RequestOptions, RetryPolicy};
///
/// # async fn demo(client: &Client, request: opendecision_client::SystemRequest) {
/// let opts = RequestOptions::new()
///     .timeout(Duration::from_secs(2))
///     .retry(RetryPolicy::new().max_retries(0));
/// let _ = client.evaluate_with(request, &opts).await;
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub(crate) timeout: Option<Duration>,
    pub(crate) retry: Option<RetryPolicy>,
    pub(crate) extra_headers: Option<HeaderMap>,
}

impl RequestOptions {
    /// An empty set of overrides (all client defaults apply).
    pub fn new() -> Self {
        Self::default()
    }

    /// Per-attempt timeout, overriding the client default.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Retry policy for this call, overriding the client default.
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Extra header sent with this request only.
    pub fn header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.extra_headers
            .get_or_insert_with(HeaderMap::new)
            .insert(name, value);
        self
    }
}

/// Build an async client for the SystemOne API.
///
/// Settings resolve in priority order: explicit builder value →
/// `OPENDECISION_*` environment variable → `TYPESAFE_*` environment variable
/// (for drop-in parity with the Python SDK) → SDK default. The API key is
/// required; everything else has a default.
///
/// # Examples
/// ```
/// use opendecision_client::Client;
///
/// # fn demo() -> Result<(), opendecision_client::Error> {
/// // Hosted TypeSafe API (default base URL).
/// let client = Client::builder().api_key("sk-...").build()?;
///
/// // Local daemon.
/// let local = Client::builder()
///     .api_key("dev-key")
///     .base_url("http://127.0.0.1:8080")
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct ClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<String>,
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
    retry: Option<RetryPolicy>,
    default_headers: HeaderMap,
    http_client: Option<reqwest::Client>,
}

impl ClientBuilder {
    /// A builder with every setting unset (environment/defaults apply).
    pub fn new() -> Self {
        Self::default()
    }

    /// Bearer API key. Falls back to `OPENDECISION_API_KEY`, then
    /// `TYPESAFE_API_KEY`, when unset.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// API base URL, e.g. `http://127.0.0.1:8080`. Falls back to
    /// `OPENDECISION_BASE_URL`, then `TYPESAFE_BASE_URL`, then
    /// [`DEFAULT_BASE_URL`].
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Model alias used by [`Client::system_one`] when the request does not
    /// name one. Falls back to `OPENDECISION_DEFAULT_MODEL`, then
    /// `TYPESAFE_DEFAULT_MODEL`, then [`DEFAULT_MODEL`].
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Per-attempt request timeout. Default 10s.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// TCP connect timeout. Unset by default.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Retry behavior. Defaults to [`RetryPolicy::default`].
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// A header sent with every request.
    pub fn header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.default_headers.insert(name, value);
        self
    }

    /// Supply a preconfigured `reqwest::Client` (connection pooling, proxy,
    /// TLS settings). Default headers and timeouts are still applied
    /// per-request by this SDK.
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Validate settings and construct the [`Client`].
    pub fn build(self) -> Result<Client, Error> {
        let api_key = resolve_setting(self.api_key, "OPENDECISION_API_KEY", "TYPESAFE_API_KEY")
            .ok_or_else(|| {
                Error::Config(
                    "no API key provided; pass ClientBuilder::api_key or set \
                     OPENDECISION_API_KEY (or TYPESAFE_API_KEY)"
                        .into(),
                )
            })?;
        if api_key.chars().any(|c| c.is_control()) {
            return Err(Error::Config("api_key contains control characters".into()));
        }

        let base_url = resolve_setting(self.base_url, "OPENDECISION_BASE_URL", "TYPESAFE_BASE_URL")
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        let base_url = base_url.trim_end_matches('/').to_owned();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(Error::Config(format!(
                "base_url must start with http:// or https://, got `{base_url}`"
            )));
        }

        let default_model = resolve_setting(
            self.default_model,
            "OPENDECISION_DEFAULT_MODEL",
            "TYPESAFE_DEFAULT_MODEL",
        )
        .unwrap_or_else(|| DEFAULT_MODEL.to_owned());

        let timeout = self.timeout.unwrap_or(DEFAULT_TIMEOUT);
        if timeout.is_zero() {
            return Err(Error::Config("timeout must be greater than zero".into()));
        }

        let retry = self.retry.unwrap_or_default();
        retry.validate()?;

        let http = match self.http_client {
            Some(client) => client,
            None => {
                let mut builder = reqwest::Client::builder();
                if let Some(connect_timeout) = self.connect_timeout {
                    builder = builder.connect_timeout(connect_timeout);
                }
                builder
                    .build()
                    .map_err(|e| Error::Config(format!("failed to build HTTP client: {e}")))?
            }
        };

        // Caller defaults first — but the SDK-managed headers below always
        // win on collision, mirroring the Python SDK's protected headers.
        // The retry-count header is stripped entirely: it is client-managed.
        let mut base_headers = HeaderMap::with_capacity(8);
        for (name, value) in self.default_headers {
            if let Some(name) = name {
                if name.as_str() == RETRY_COUNT_HEADER {
                    continue;
                }
                base_headers.insert(name, value);
            }
        }
        let sdk_version = env!("CARGO_PKG_VERSION");
        base_headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|_| Error::Config("api_key contains invalid header characters".into()))?,
        );
        base_headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        base_headers.insert(
            USER_AGENT_HEADER,
            HeaderValue::from_str(&format!("{SDK_NAME}/{sdk_version}"))
                .map_err(|_| Error::Config("user agent construction failed".into()))?,
        );
        base_headers.insert(
            SDK_HEADER,
            HeaderValue::from_str(&format!("{SDK_NAME}/{sdk_version}"))
                .map_err(|_| Error::Config("sdk header construction failed".into()))?,
        );
        base_headers.insert(
            RUNTIME_HEADER,
            HeaderValue::from_str(&format!(
                "rust ({}; {})",
                std::env::consts::OS,
                std::env::consts::ARCH
            ))
            .map_err(|_| Error::Config("runtime header construction failed".into()))?,
        );

        Ok(Client {
            inner: Arc::new(ClientInner {
                http,
                base_url,
                default_model,
                timeout,
                retry,
                base_headers,
            }),
        })
    }
}

/// Resolve: explicit value → primary env var → fallback env var, ignoring
/// empty/whitespace environment values (matches the Python SDK).
fn resolve_setting(explicit: Option<String>, primary: &str, fallback: &str) -> Option<String> {
    resolve_lookup(explicit, non_empty_env, primary, fallback)
}

/// Pure precedence core, split out so unit tests can inject lookups without
/// touching process-global environment state.
fn resolve_lookup(
    explicit: Option<String>,
    lookup: impl Fn(&str) -> Option<String>,
    primary: &str,
    fallback: &str,
) -> Option<String> {
    explicit
        .or_else(|| lookup(primary))
        .or_else(|| lookup(fallback))
}

fn non_empty_env(name: &str) -> Option<String> {
    clean_env_value(std::env::var(name).ok()?)
}

/// Trim and drop empty/whitespace environment values, matching the Python
/// SDK (`os.environ.get(env, "").strip() or default`).
fn clean_env_value(raw: String) -> Option<String> {
    let value = raw.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

struct ClientInner {
    http: reqwest::Client,
    base_url: String,
    default_model: String,
    timeout: Duration,
    retry: RetryPolicy,
    base_headers: HeaderMap,
}

/// An async client for the SystemOne HTTP API.
///
/// Cheap to clone; clones share one connection pool. All methods are safe to
/// call concurrently from many tasks.
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.inner.base_url)
            .field("default_model", &self.inner.default_model)
            .field("timeout", &self.inner.timeout)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// `Client::builder().api_key(key).build()?` with all other defaults.
    pub fn new(api_key: impl Into<String>) -> Result<Self, Error> {
        Self::builder().api_key(api_key).build()
    }

    /// Start building a customized client.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// The resolved base URL (no trailing slash).
    pub fn base_url(&self) -> &str {
        &self.inner.base_url
    }

    /// The resolved default model alias.
    pub fn default_model(&self) -> &str {
        &self.inner.default_model
    }

    /// Evaluate a [`SystemRequest`], `POST`ing it to `/v1/systemone`.
    ///
    /// Retries and rate-limit backoff follow the client's [`RetryPolicy`];
    /// see [`evaluate_with`](Self::evaluate_with) for per-call overrides.
    pub async fn evaluate(&self, request: SystemRequest) -> Result<SystemResponse, Error> {
        self.evaluate_with(request, &RequestOptions::new()).await
    }

    /// [`evaluate`](Self::evaluate) with per-call [`RequestOptions`].
    pub async fn evaluate_with(
        &self,
        request: SystemRequest,
        opts: &RequestOptions,
    ) -> Result<SystemResponse, Error> {
        let body = serde_json::to_value(&request)
            .map_err(|e| Error::Config(format!("request could not be serialized: {e}")))?;
        self.send_json(Method::POST, SYSTEM_ONE_PATH, Some(&body), opts)
            .await
    }

    /// Evaluate ad-hoc questions against a state, mirroring the Python SDK's
    /// `client.system_one(...)`. The client's default model is used.
    ///
    /// # Examples
    /// ```
    /// use opendecision_client::{question, Client};
    ///
    /// # async fn demo(client: &Client) -> Result<(), opendecision_client::Error> {
    /// let response = client
    ///     .system_one(
    ///         "I was charged twice. Please help.",
    ///         [
    ///             ("billing", question::noul("Is this about billing?")),
    ///             ("tone", question::choice("What is the tone?", [("calm", None), ("angry", None)])),
    ///         ],
    ///     )
    ///     .await?;
    /// assert!(response.answers.contains_key("billing"));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn system_one<S, I, K>(&self, state: S, questions: I) -> Result<SystemResponse, Error>
    where
        S: IntoState,
        I: IntoIterator<Item = (K, Question)>,
        K: Into<String>,
    {
        self.system_one_with(state, questions, &RequestOptions::new())
            .await
    }

    /// [`system_one`](Self::system_one) with per-call [`RequestOptions`].
    pub async fn system_one_with<S, I, K>(
        &self,
        state: S,
        questions: I,
        opts: &RequestOptions,
    ) -> Result<SystemResponse, Error>
    where
        S: IntoState,
        I: IntoIterator<Item = (K, Question)>,
        K: Into<String>,
    {
        let request = SystemRequest {
            state: state.into_state(),
            model: self.inner.default_model.clone(),
            questions: questions
                .into_iter()
                .map(|(id, question)| (id.into(), question))
                .collect(),
        };
        self.evaluate_with(request, opts).await
    }

    /// `GET /v1/models` — list model aliases registered on the server.
    pub async fn list_models(&self) -> Result<ModelsResponse, Error> {
        self.list_models_with(&RequestOptions::new()).await
    }

    /// [`list_models`](Self::list_models) with per-call [`RequestOptions`].
    pub async fn list_models_with(&self, opts: &RequestOptions) -> Result<ModelsResponse, Error> {
        self.send_json(Method::GET, MODELS_PATH, None, opts).await
    }

    /// `GET /health` — liveness probe. Unauthenticated, so it works even
    /// with a wrong API key.
    pub async fn health(&self) -> Result<Health, Error> {
        self.health_with(&RequestOptions::new()).await
    }

    /// [`health`](Self::health) with per-call [`RequestOptions`].
    pub async fn health_with(&self, opts: &RequestOptions) -> Result<Health, Error> {
        self.send_json(Method::GET, HEALTH_PATH, None, opts).await
    }

    /// Send one logical request (possibly several HTTP attempts) and decode
    /// the 2xx body as `T`.
    async fn send_json<T: DeserializeOwned>(
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

/// Immutable description of one logical request, shared by every retry
/// attempt (only the retry count and headers differ between attempts).
struct RequestDesc<'a> {
    method: Method,
    url: String,
    endpoint: String,
    body: Option<&'a serde_json::Value>,
    timeout: Duration,
}

/// Headers user code may not override (the Python SDK's protected set):
/// SDK identification/protocol headers always win, the retry-count header is
/// client-managed, and content-type belongs to the JSON body.
fn is_user_overridable(name: &str, has_body: bool) -> bool {
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

fn classify_transport_error(error: reqwest::Error, timeout: Duration) -> Error {
    if error.is_timeout() {
        Error::Timeout { timeout }
    } else {
        Error::Connection(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of the Python SDK's `test_resolution` precedence matrix, using an
    /// injected lookup so no process-global environment state is touched.
    #[test]
    fn resolve_lookup_precedence() {
        let primary = "OPENDECISION_API_KEY";
        let fallback = "TYPESAFE_API_KEY";
        let env = |wanted: &'static str, value: &'static str| {
            move |name: &str| (name == wanted).then(|| value.to_owned())
        };

        // Explicit wins over both env vars.
        assert_eq!(
            resolve_lookup(
                Some("explicit".into()),
                env(primary, "p"),
                primary,
                fallback
            ),
            Some("explicit".into())
        );
        // Primary env beats fallback env.
        assert_eq!(
            resolve_lookup(None, env(primary, "p"), primary, fallback),
            Some("p".into())
        );
        // Fallback env is used when primary is absent (TYPESAFE_* parity).
        assert_eq!(
            resolve_lookup(None, env(fallback, "f"), primary, fallback),
            Some("f".into())
        );
        // Default applies only when nothing resolves (None via |_: Option<_>|).
        assert_eq!(resolve_lookup(None, |_| None, primary, fallback), None);
        // Empty/whitespace env values are ignored, like the Python SDK
        // (`os.environ.get(env, "").strip() or default`); the lookup layer
        // (`non_empty_env`) applies that filter before resolution.
        assert_eq!(clean_env_value("   ".into()), None);
        assert_eq!(clean_env_value("".into()), None);
        assert_eq!(
            clean_env_value("  real-value  ".into()),
            Some("real-value".into())
        );
    }

    #[test]
    fn client_debug_does_not_leak_api_key() {
        let client = Client::new("super-secret-key").unwrap();
        let rendered = format!("{client:?}");
        assert!(!rendered.contains("super-secret-key"), "{rendered}");
    }

    #[test]
    fn base_url_trailing_slashes_trimmed() {
        let client = Client::builder()
            .api_key("k")
            .base_url("http://example.test/prefix///")
            .build()
            .unwrap();
        assert_eq!(client.base_url(), "http://example.test/prefix");
    }

    #[test]
    fn invalid_base_url_rejected_at_build() {
        let err = Client::builder()
            .api_key("k")
            .base_url("ftp://example.test")
            .build()
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err:?}");
    }

    #[test]
    fn zero_timeout_rejected_at_build() {
        let err = Client::builder()
            .api_key("k")
            .base_url("http://example.test")
            .timeout(Duration::ZERO)
            .build()
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err:?}");
    }
}
