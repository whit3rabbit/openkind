//! Optional proxy-cache hook for the evaluation handlers.
//!
//! When the daemon runs with `--proxy-cache-upstream`, it builds a
//! [`SystemProxy`] implementation and stores it on [`AppState`]. The HTTP
//! handler consults the hook before dispatching to the local registry, so
//! proxied model aliases can be answered from the distilling cache or
//! forwarded to the upstream Jev API with the caller's own credentials.
//! The trait lives here so `openkind-api` never depends on the server or
//! backends crates; the daemon supplies the implementation.

use async_trait::async_trait;
use openkind_core::{SystemRequest, SystemResponse};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::models::ModelsResponse;

/// Where a proxied answer came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProxySource {
    /// Served locally by the distilling cache.
    Local,
    /// Forwarded to the upstream Jev-compatible API.
    Upstream,
}

impl ProxySource {
    /// Stable wire token for the `x-openkind-cache` response header.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProxySource::Local => "local",
            ProxySource::Upstream => "upstream",
        }
    }
}

/// The outcome of a proxied evaluation.
#[derive(Debug)]
pub struct ProxyOutcome {
    /// The wire response (locally built or relayed from upstream).
    pub response: SystemResponse,
    /// Where the answer came from.
    pub source: ProxySource,
    /// Optional per-question routing detail (JSON object: question id →
    /// student version or forward reason).
    pub detail: Option<serde_json::Value>,
}

/// The proxy-cache hook the daemon installs.
#[async_trait]
pub trait SystemProxy: Send + Sync {
    /// Whether this hook wants to handle a request (e.g. its `model` is a
    /// proxied alias). Cheap and synchronous.
    fn wants(&self, request: &SystemRequest) -> bool;

    /// Whether the proxy forwards caller credentials to the upstream Jev API.
    ///
    /// When `false` (for example, when a fixed upstream key is configured on
    /// the daemon), the HTTP handler does not extract the inbound `Authorization`
    /// bearer token or supply it to [`Self::evaluate`], preventing local daemon
    /// bearer credentials from crossing the proxy boundary.
    fn forwards_caller_credentials(&self) -> bool {
        true
    }

    /// Handle the request: answer locally when the cache is confident,
    /// otherwise forward upstream with the caller's credentials.
    async fn evaluate(
        &self,
        request: SystemRequest,
        caller_key: Option<String>,
    ) -> Result<ProxyOutcome, ApiError>;

    /// Upstream `/v1/models` listing when the proxy should answer model
    /// discovery transparently; `None` falls back to the local registry.
    async fn models(&self) -> Option<ModelsResponse>;
}
