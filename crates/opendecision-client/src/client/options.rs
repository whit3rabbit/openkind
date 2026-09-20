//! Request configuration options, default constants, and state conversion helpers for the client.

use std::time::Duration;

use opendecision_core::State;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

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

pub(crate) const USER_AGENT_HEADER: &str = "user-agent";
pub(crate) const SDK_HEADER: &str = "x-typesafe-sdk";
pub(crate) const RUNTIME_HEADER: &str = "x-typesafe-runtime";
/// Sent on retry attempts (value = retry number), mirroring the Python
/// SDK's `X-TypeSafe-Retry-Count` so servers can observe client retries.
pub(crate) const RETRY_COUNT_HEADER: &str = "x-typesafe-retry-count";

/// Response of `GET /health` — liveness probe, no authentication required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Health {
    /// Service status string (currently `"ok"`).
    pub status: String,
}

/// Conversion into a Jev [`State`], letting [`crate::Client::system_one`] accept
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
