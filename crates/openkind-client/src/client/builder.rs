//! Client builder and environment resolution logic.

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION};

use super::core::{Client, ClientInner};
use super::options::{
    DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT, RUNTIME_HEADER, SDK_HEADER, SDK_NAME,
    USER_AGENT_HEADER,
};
use crate::error::Error;
use crate::retry::RetryPolicy;

/// Build an async client for the SystemOne API.
///
/// Settings resolve in priority order: explicit builder value →
/// `OPENKIND_*` environment variable → `TYPESAFE_*` environment variable
/// (for drop-in parity with the Python SDK) → SDK default. The API key is
/// required; everything else has a default. Cloudflare mode selects its own
/// account URL and `typesafe/jev` model unless overridden explicitly.
///
/// # Examples
/// ```
/// use openkind_client::Client;
///
/// # fn demo() -> Result<(), openkind_client::Error> {
/// // Hosted TypeSafe API (default base URL).
/// let client = Client::builder().api_key("sk-...").build()?;
///
/// // Local daemon.
/// let local = Client::builder()
///     .api_key("dev-key")
///     .base_url("http://127.0.0.1:18080")
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct ClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    cloudflare_account: Option<String>,
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

    /// Bearer API key. Falls back to `OPENKIND_API_KEY`, then
    /// `TYPESAFE_API_KEY`, when unset.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// API base URL, e.g. `http://127.0.0.1:18080`. Falls back to
    /// `OPENKIND_BASE_URL`, then `TYPESAFE_BASE_URL`, then
    /// [`DEFAULT_BASE_URL`].
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Use Cloudflare Workers AI Jev through the account's `/ai/run` API.
    /// The API token is supplied with [`api_key`](Self::api_key). An explicit
    /// [`base_url`](Self::base_url) may override the Cloudflare host for a proxy
    /// or test server. Cloudflare mode ignores base URL and model env defaults.
    pub fn cloudflare_account(mut self, account_id: impl Into<String>) -> Self {
        self.cloudflare_account = Some(account_id.into());
        self
    }

    /// Model alias used by [`Client::system_one`] when the request does not
    /// name one. Falls back to `OPENKIND_DEFAULT_MODEL`, then
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
        let api_key = resolve_api_key(self.api_key, non_empty_env).map_err(Error::Config)?;

        let cloudflare_account = self.cloudflare_account;
        if let Some(id) = &cloudflare_account {
            if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
                return Err(Error::Config(
                    "cloudflare account id must be alphanumeric".into(),
                ));
            }
        }
        let base_url = match (&self.base_url, &cloudflare_account) {
            (Some(url), _) => url.clone(),
            (None, Some(id)) => format!("https://api.cloudflare.com/client/v4/accounts/{id}"),
            (None, None) => resolve_setting(None, "OPENKIND_BASE_URL", "TYPESAFE_BASE_URL")
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
        };
        let base_url = base_url.trim_end_matches('/').to_owned();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(Error::Config(format!(
                "base_url must start with http:// or https://, got `{base_url}`"
            )));
        }
        let parsed_base =
            reqwest::Url::parse(&base_url).map_err(|_| Error::Config("invalid base URL".into()))?;
        if parsed_base.query().is_some()
            || parsed_base.fragment().is_some()
            || !parsed_base.username().is_empty()
            || parsed_base.password().is_some()
        {
            return Err(Error::Config(
                "base_url must not contain a query, fragment, or credentials".into(),
            ));
        }

        let default_model = if cloudflare_account.is_some() {
            self.default_model
                .unwrap_or_else(|| "typesafe/jev".to_owned())
        } else {
            resolve_setting(
                self.default_model,
                "OPENKIND_DEFAULT_MODEL",
                "TYPESAFE_DEFAULT_MODEL",
            )
            .unwrap_or_else(|| DEFAULT_MODEL.to_owned())
        };

        let timeout = self.timeout.unwrap_or(DEFAULT_TIMEOUT);
        if timeout.is_zero() {
            return Err(Error::Config("timeout must be greater than zero".into()));
        }

        let retry = self.retry.unwrap_or_default();
        retry.validate()?;

        let owns_http_client = self.http_client.is_none();
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
                if !super::transport::is_user_overridable(name.as_str(), false) {
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

        // The direct HTTP/1.1 transport applies only when this SDK owns the
        // HTTP client, the base URL is plain http, and no proxy is
        // configured — otherwise reqwest handles everything, exactly as
        // before.
        let fast_h1 = match (
            owns_http_client,
            parsed_base,
            super::http1::proxy_env_present(),
        ) {
            (true, url, false) if url.scheme() == "http" => {
                match (url.host_str(), url.port_or_known_default()) {
                    (Some(host), Some(port)) => {
                        Some(super::http1::H1Pool::new(host, port, self.connect_timeout))
                    }
                    _ => None,
                }
            }
            _ => None,
        };

        Ok(Client {
            inner: Arc::new(ClientInner {
                http,
                urls: super::core::EndpointUrls::new(&base_url, cloudflare_account.is_some())?,
                base_url,
                is_cloudflare: cloudflare_account.is_some(),
                default_model,
                timeout,
                retry,
                base_headers,
                fast_h1,
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
pub(crate) fn resolve_lookup(
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

/// Resolve and validate the API key like the Python SDK's
/// `resolve_and_validate_api_key`: an explicit key wins even when invalid
/// (trimmed, never falling back to the environment), environment values are
/// trimmed and must be non-empty, and the survivor must be printable ASCII
/// without whitespace.
pub(crate) fn resolve_api_key(
    explicit: Option<String>,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String, String> {
    let resolved = match explicit {
        Some(value) => Some(value.trim().to_owned()),
        None => lookup("OPENKIND_API_KEY").or_else(|| lookup("TYPESAFE_API_KEY")),
    }
    .filter(|key| !key.is_empty());
    match resolved {
        Some(key) if key.chars().all(|c| matches!(c, '!'..='~')) => Ok(key),
        Some(_) => {
            Err("API key must contain only printable ASCII characters without whitespace".into())
        }
        None => Err("no API key provided; pass ClientBuilder::api_key or set \
             OPENKIND_API_KEY (or TYPESAFE_API_KEY)"
            .into()),
    }
}

/// Trim and drop empty/whitespace environment values, matching the Python
/// SDK (`os.environ.get(env, "").strip() or default`).
pub(crate) fn clean_env_value(raw: String) -> Option<String> {
    let value = raw.trim();
    (!value.is_empty()).then(|| value.to_owned())
}
