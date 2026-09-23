//! Core `Client` struct and public endpoint methods.

use std::sync::Arc;
use std::time::Duration;

use openkind_core::{ModelsResponse, Question, SystemRequest, SystemResponse};
use reqwest::header::HeaderMap;
use reqwest::Method;

use super::builder::ClientBuilder;
use super::options::{
    Health, IntoState, RequestOptions, HEALTH_PATH, MODELS_PATH, SYSTEM_ONE_PATH,
};
use crate::error::Error;
use crate::retry::RetryPolicy;

/// The endpoints this SDK talks to, with per-endpoint request URLs resolved
/// once at client construction instead of formatted and parsed per call.
#[derive(Clone)]
pub(crate) struct EndpointUrls {
    /// `POST` target for evaluation requests.
    pub(crate) system_one: reqwest::Url,
    /// `GET` target for model discovery.
    pub(crate) models: reqwest::Url,
    /// `GET` target for liveness probes.
    pub(crate) health: reqwest::Url,
}

impl EndpointUrls {
    pub(crate) fn new(base_url: &str) -> Result<Self, Error> {
        let join = |path: &str| {
            reqwest::Url::parse(&format!("{base_url}{path}"))
                .map_err(|e| Error::Config(format!("invalid base URL `{base_url}`: {e}")))
        };
        Ok(Self {
            system_one: join(SYSTEM_ONE_PATH)?,
            models: join(MODELS_PATH)?,
            health: join(HEALTH_PATH)?,
        })
    }
}

pub(crate) struct ClientInner {
    pub(crate) http: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) urls: EndpointUrls,
    pub(crate) default_model: String,
    pub(crate) timeout: Duration,
    pub(crate) retry: RetryPolicy,
    pub(crate) base_headers: HeaderMap,
    /// Direct HTTP/1.1 pool used for plain-`http` base URLs. `None` keeps
    /// every request on the reqwest transport (https base URLs, caller
    /// supplied HTTP clients, or proxy environments).
    pub(crate) fast_h1: Option<tokio::sync::Mutex<super::http1::H1Pool>>,
}

/// An async client for the SystemOne HTTP API.
///
/// Cheap to clone; clones share one connection pool. All methods are safe to
/// call concurrently from many tasks.
#[derive(Clone)]
pub struct Client {
    pub(crate) inner: Arc<ClientInner>,
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
        // Serialize once here; the transport sends the bytes directly so the
        // request is not serialized a second time inside the HTTP client.
        let body = serde_json::to_vec(&request)
            .map_err(|e| Error::Config(format!("request could not be serialized: {e}")))?;
        self.send_json(
            Method::POST,
            &self.inner.urls.system_one,
            SYSTEM_ONE_PATH,
            Some(body),
            opts,
        )
        .await
    }

    /// Evaluate ad-hoc questions against a state, mirroring the Python SDK's
    /// `client.system_one(...)`. The client's default model is used.
    ///
    /// # Examples
    /// ```
    /// use openkind_client::{question, Client};
    ///
    /// # async fn demo(client: &Client) -> Result<(), openkind_client::Error> {
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
        self.send_json(
            Method::GET,
            &self.inner.urls.models,
            MODELS_PATH,
            None,
            opts,
        )
        .await
    }

    /// `GET /health` — liveness probe. Unauthenticated, so it works even
    /// with a wrong API key.
    pub async fn health(&self) -> Result<Health, Error> {
        self.health_with(&RequestOptions::new()).await
    }

    /// [`health`](Self::health) with per-call [`RequestOptions`].
    pub async fn health_with(&self, opts: &RequestOptions) -> Result<Health, Error> {
        self.send_json(
            Method::GET,
            &self.inner.urls.health,
            HEALTH_PATH,
            None,
            opts,
        )
        .await
    }
}
