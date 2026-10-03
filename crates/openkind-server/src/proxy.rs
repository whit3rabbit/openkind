//! The proxy-cache service: a distilling cache in front of a remote Jev API.
//!
//! Requests for proxied model aliases are grouped by (instructions, criteria)
//! into tasks. Each request is either answered entirely locally — when every
//! choice question passes its task's calibrated confidence policy and OOD
//! gate — or forwarded upstream with the caller's own bearer key, with the
//! teacher's answers recorded as training rows. Noul and score questions are
//! always forwarded; a request with any forward-only question is forwarded
//! wholesale (all-or-nothing), and locally answerable items become
//! co-deferred training rows.
//!
//! Transparency rules (matching the reference proxy design):
//! - A caller key is only trusted after the upstream accepted a request with
//!   it; until then every request forwards.
//! - Forward-only requests are never embedded.
//! - Any proxy-internal failure fails open to the upstream.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use async_trait::async_trait;
use openkind_api::{ApiError, ProxyOutcome, ProxySource, SystemProxy};
use openkind_backends::proxy_cache::engine::{
    RouteDecision, TaskEngine, TaskStatus, TeacherAnswer, TickOutcome,
};
use openkind_backends::proxy_cache::manager::ProxyCacheManager;
use openkind_backends::proxy_cache::task::{peakedness, Channel, RoutingReason, TaskSpec};
use openkind_backends::proxy_cache::text::state_text;
use openkind_backends::proxy_cache::ProxyCacheError;
use openkind_client::Client;
use openkind_core::{
    validate_response_for_request, Answer, ChoiceAnswer, ModelInfo, Question, SystemRequest,
    SystemResponse, Usage,
};
use openkind_engine::{DecisionEngine, EngineError, EngineResult};
use sha2::{Digest, Sha256};

/// One routed group: task identity, its question ids, the routing decision,
/// and the engine handle (absent when the task is not admitted yet).
type RoutedGroup = (
    TaskSpec,
    Vec<String>,
    RouteDecision,
    Option<Arc<std::sync::Mutex<TaskEngine>>>,
);

/// Grouping cap: requests with more distinct choice specs forward wholesale.
const MAX_QUESTIONS: usize = 32;
/// Detail header cap: above this many questions the header collapses to a count.
const MAX_DETAIL_QUESTIONS: usize = 64;

/// Whether a routing decision may be returned without consulting the teacher.
///
/// Audit decisions intentionally carry the student's would-be prediction for
/// drift scoring, so the presence of `local` alone is not sufficient.
fn is_locally_servable(decision: &RouteDecision) -> bool {
    decision.local.is_some()
        && decision.channel == Channel::Student
        && decision.reason == RoutingReason::Confident
}

fn decision_for_teacher(decision: &RouteDecision) -> RouteDecision {
    let mut decision = decision.clone();
    if is_locally_servable(&decision) {
        decision.channel = Channel::CoDeferred;
    }
    decision
}

/// Server-side proxy-cache configuration (parsed from CLI flags).
#[derive(Debug, Clone)]
pub struct ProxyCacheServiceConfig {
    /// Upstream base URL (e.g. `https://api.typesafe.ai`).
    pub upstream: String,
    /// Optional upstream bearer key used instead of the caller's key.
    pub upstream_key: Option<String>,
    /// Per-attempt upstream timeout.
    pub upstream_timeout_ms: u64,
    /// Model aliases the proxy intercepts.
    pub proxied_models: Vec<String>,
}

/// Verified caller keys: a key is accepted after the upstream answered a
/// request made with it, and expires after the TTL. Only salted hashes are
/// stored; raw keys never touch disk or logs.
pub struct KeyRegistry {
    verified: Mutex<HashMap<String, Instant>>,
    ttl: Duration,
}

impl KeyRegistry {
    fn new(ttl: Duration) -> Self {
        Self {
            verified: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    fn verified(&self, hash: &str) -> bool {
        let mut guard = match self.verified.lock() {
            Ok(guard) => guard,
            Err(_) => return false,
        };
        match guard.get(hash) {
            Some(accepted_at) if accepted_at.elapsed() <= self.ttl => true,
            Some(_) => {
                guard.remove(hash);
                false
            }
            None => false,
        }
    }

    fn accept(&self, hash: &str) {
        if let Ok(mut guard) = self.verified.lock() {
            guard.insert(hash.to_owned(), Instant::now());
        }
    }

    fn revoke(&self, hash: &str) {
        if let Ok(mut guard) = self.verified.lock() {
            guard.remove(hash);
        }
    }
}

/// The proxy-cache service implementing the API crate's [`SystemProxy`] hook.
pub struct ProxyService {
    manager: Arc<ProxyCacheManager>,
    upstream: String,
    upstream_key: Option<String>,
    upstream_timeout: Duration,
    proxied_models: HashSet<String>,
    keys: KeyRegistry,
    clients: Mutex<HashMap<String, Client>>,
}

impl ProxyService {
    /// Build the service on top of a ready manager.
    pub fn new(manager: Arc<ProxyCacheManager>, config: ProxyCacheServiceConfig) -> Self {
        Self {
            manager,
            upstream: config.upstream,
            upstream_key: config.upstream_key,
            upstream_timeout: Duration::from_millis(config.upstream_timeout_ms.max(1)),
            proxied_models: config.proxied_models.into_iter().collect(),
            keys: KeyRegistry::new(Duration::from_secs(3600)),
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// The underlying manager (status endpoints, tests).
    #[allow(dead_code)]
    pub fn manager(&self) -> &Arc<ProxyCacheManager> {
        &self.manager
    }

    fn key_hash(&self, key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.manager.salt());
        hasher.update(key.as_bytes());
        let digest = hasher.finalize();
        let mut out = String::with_capacity(32);
        for byte in &digest[..16] {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// A client bound to one bearer key (or the configured upstream key).
    fn client_for(&self, caller_key: Option<&str>) -> Result<Client, ApiError> {
        let effective_key = caller_key.or(self.upstream_key.as_deref());
        let Some(key) = effective_key else {
            return Err(ApiError::Unauthorized);
        };
        let hash = self.key_hash(key);
        if let Ok(guard) = self.clients.lock() {
            if let Some(client) = guard.get(&hash) {
                return Ok(client.clone());
            }
        }
        let client = Client::builder()
            .api_key(key)
            .base_url(&self.upstream)
            .timeout(self.upstream_timeout)
            .build()
            .map_err(|error| ApiError::Internal(format!("upstream client: {error}")))?;
        if let Ok(mut guard) = self.clients.lock() {
            if guard.len() >= 128 {
                guard.clear();
            }
            guard.insert(hash, client.clone());
        }
        Ok(client)
    }

    /// Forward a request verbatim (fail-open path: no routing, no recording).
    async fn plain_forward(
        &self,
        request: SystemRequest,
        caller_key: Option<&str>,
        reason: &'static str,
    ) -> Result<ProxyOutcome, ApiError> {
        let client = self.client_for(caller_key)?;
        let response = client
            .evaluate(request)
            .await
            .map_err(|error| map_upstream_error(error, self.upstream_key.is_some()))?;
        let detail = serde_json::json!({ "request": reason });
        Ok(ProxyOutcome {
            response,
            source: ProxySource::Upstream,
            detail: Some(detail),
        })
    }

    async fn evaluate_inner(
        &self,
        request: SystemRequest,
        caller_key: Option<String>,
    ) -> Result<ProxyOutcome, ApiError> {
        let (groups, others) = group_questions(&request);

        // Not understood: no choice questions, or too many distinct specs.
        if groups.is_empty() {
            return self
                .plain_forward(request, caller_key.as_deref(), "no_choice")
                .await;
        }
        if groups.len() > MAX_QUESTIONS || others > MAX_QUESTIONS {
            return self
                .plain_forward(request, caller_key.as_deref(), "too_many_questions")
                .await;
        }

        let caller_hash = caller_key.as_deref().map(|key| self.key_hash(key));
        let key_verified = caller_hash
            .as_deref()
            .map(|hash| self.keys.verified(hash))
            .unwrap_or(false);

        if !key_verified {
            // Forward first; only a parsed 2xx answer makes the key trusted.
            let client = self.client_for(caller_key.as_deref())?;
            let response = client.evaluate(request.clone()).await.map_err(|error| {
                if let Some(hash) = &caller_hash {
                    if matches!(error.status(), Some(401) | Some(403)) {
                        self.keys.revoke(hash);
                    }
                }
                map_upstream_error(error, self.upstream_key.is_some())
            })?;
            if let Some(hash) = &caller_hash {
                self.keys.accept(hash);
            }
            // Record the choice answers as unverified-key deferred rows so
            // bootstrap traffic is not lost; embedding happens here because
            // no routing decision was needed to serve the request.
            self.record_deferred_rows(&request, &groups, &response, "key_unverified")
                .await;
            let detail = detail_map(
                groups.iter().flat_map(|(_, ids)| ids.iter()),
                "key_unverified",
            );
            return Ok(ProxyOutcome {
                response,
                source: ProxySource::Upstream,
                detail: Some(detail),
            });
        }

        // Route: one embedding per request (the state text), shared by all
        // groups. Forward-only requests never reach this line.
        let text = state_text(&request.state);
        let state_type = match &request.state {
            openkind_core::State::Text(_) => "text",
            _ => "json",
        };
        let embeddings = self
            .manager
            .embedder()
            .encode(std::slice::from_ref(&text))
            .map_err(|error| ApiError::Internal(format!("proxy encoder: {error}")))?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::Internal("proxy encoder returned no embedding".into()))?;

        // Resolve engines and route each group.
        let mut routed: Vec<RoutedGroup> = Vec::new();
        let mut any_routed = false;
        let mut all_local = others == 0;
        {
            for (spec, ids) in &groups {
                let engine = match self.manager.route_context(spec, &request.model) {
                    Ok(Some(engine)) => Some(engine),
                    Ok(None) => None, // not admitted yet
                    Err(error) => {
                        tracing::warn!("proxy-cache task resolution failed: {error}");
                        None
                    }
                };
                let decision = match &engine {
                    Some(engine) => {
                        let mut guard = engine
                            .lock()
                            .map_err(|_| ApiError::Internal("proxy task mutex poisoned".into()))?;
                        guard.begin_request();
                        guard.route(&embedding)
                    }
                    None => RouteDecision {
                        reason: RoutingReason::Bootstrap,
                        channel: Channel::Bootstrap,
                        local: None,
                        student_version: None,
                    },
                };
                if is_locally_servable(&decision) {
                    any_routed = true;
                } else {
                    all_local = false;
                }
                routed.push((spec.clone(), ids.clone(), decision, engine));
            }
        }

        // All-or-nothing: answer locally only when every choice question can
        // be answered and there are no other (noul/score) questions.
        if all_local && any_routed {
            let response = self.build_local_response(&request, &routed)?;
            // Record locally served rows.
            for (_spec, _ids, decision, engine) in &routed {
                if let Some(engine) = engine {
                    if let Ok(mut guard) = engine.lock() {
                        let _ = guard.observe_student_served(
                            decision,
                            &embedding,
                            &text,
                            state_type,
                            &self.manager.embedder().id(),
                        );
                    }
                }
            }
            let detail = self.detail_for_local(&routed);
            return Ok(ProxyOutcome {
                response,
                source: ProxySource::Local,
                detail: Some(detail),
            });
        }

        // Forward wholesale; record teacher answers for every routed group.
        let started = Instant::now();
        let client = self.client_for(caller_key.as_deref())?;
        let response = client.evaluate(request.clone()).await.map_err(|error| {
            if let Some(hash) = &caller_hash {
                if matches!(error.status(), Some(401) | Some(403)) {
                    self.keys.revoke(hash);
                }
            }
            map_upstream_error(error, self.upstream_key.is_some())
        })?;
        if let Some(hash) = &caller_hash {
            self.keys.accept(hash);
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

        for (spec, ids, decision, engine) in &routed {
            let Some(engine) = engine else {
                // route_context counted this request toward admission.
                continue;
            };
            // A locally answerable item forwarded with its request becomes a
            // co-deferred training row (teacher-labelled).
            let decision = decision_for_teacher(decision);
            let Some(mut answer) = teacher_answer_for(&response, ids, &spec.classes()) else {
                continue;
            };
            answer.latency_ms = latency_ms;
            let events = {
                let mut guard = engine
                    .lock()
                    .map_err(|_| ApiError::Internal("proxy task mutex poisoned".into()))?;
                guard
                    .observe_teacher(
                        &decision,
                        &embedding,
                        &text,
                        state_type,
                        &self.manager.embedder().id(),
                        &answer,
                    )
                    .map_err(|error| {
                        // Recording is best-effort: never fail the caller's
                        // request because the cache could not persist a row.
                        tracing::warn!("proxy-cache record failed: {error}");
                    })
                    .unwrap_or_default()
            };
            for (kind, _) in events {
                tracing::info!(event = %kind, "proxy-cache");
            }
            let tick = {
                let mut guard = engine
                    .lock()
                    .map_err(|_| ApiError::Internal("proxy task mutex poisoned".into()))?;
                guard.tick().unwrap_or_default()
            };
            let key = spec.task_key(&self.manager.config().default_tenant, &request.model);
            self.run_tick_outcome(&key, tick);
        }

        let reason = if any_routed {
            "co_deferred"
        } else {
            "unroutable"
        };
        let detail = detail_map(routed.iter().flat_map(|(_, ids, _, _)| ids.iter()), reason);
        Ok(ProxyOutcome {
            response,
            source: ProxySource::Upstream,
            detail: Some(detail),
        })
    }

    fn run_tick_outcome(&self, key: &str, outcome: TickOutcome) {
        for (kind, _) in outcome.events {
            tracing::info!(event = %kind, "proxy-cache");
        }
        if outcome.train_requested {
            self.manager.request_training(key);
        }
    }

    async fn record_deferred_rows(
        &self,
        request: &SystemRequest,
        groups: &[(TaskSpec, Vec<String>)],
        response: &SystemResponse,
        reason: &'static str,
    ) {
        let text = state_text(&request.state);
        let state_type = match &request.state {
            openkind_core::State::Text(_) => "text",
            _ => "json",
        };
        let Ok(embeddings) = self.manager.embedder().encode(std::slice::from_ref(&text)) else {
            return; // fail open: recording is best-effort
        };
        let Some(embedding) = embeddings.into_iter().next() else {
            return;
        };
        for (spec, ids) in groups {
            let Some(engine) = self
                .manager
                .route_context(spec, &request.model)
                .ok()
                .flatten()
            else {
                continue;
            };
            let Some(answer) = teacher_answer_for(response, ids, &spec.classes()) else {
                continue;
            };
            let decision = RouteDecision {
                reason: RoutingReason::Bootstrap,
                channel: Channel::CoDeferred,
                local: None,
                student_version: None,
            };
            if let Ok(mut guard) = engine.lock() {
                guard.begin_request();
                let _ = guard.observe_teacher(
                    &decision,
                    &embedding,
                    &text,
                    state_type,
                    &self.manager.embedder().id(),
                    &answer,
                );
            }
            let _ = reason;
        }
    }

    fn build_local_response(
        &self,
        request: &SystemRequest,
        routed: &[RoutedGroup],
    ) -> Result<SystemResponse, ApiError> {
        // The local response reports the lineage's resolved teacher model so
        // callers see the same `model` they would get from the upstream.
        let model = routed
            .iter()
            .filter_map(|(_, _, _, engine)| engine.as_ref())
            .filter_map(|engine| engine.lock().ok().map(|guard| guard.status()))
            .find_map(|status: TaskStatus| status.teacher_model)
            .unwrap_or_else(|| request.model.clone());

        let mut answers: std::collections::HashMap<String, Answer, openkind_core::WireHashState> =
            std::collections::HashMap::with_hasher(openkind_core::WireHashState::default());
        for (spec, ids, decision, engine) in routed {
            let Some(local) = &decision.local else {
                continue;
            };
            let floor = engine
                .as_ref()
                .and_then(|engine| engine.lock().ok())
                .map(|guard| guard.status().confidence_floor)
                .unwrap_or(None);
            let confidence = peakedness(&local.probabilities).max(floor.unwrap_or(0.0));
            let mut probabilities = std::collections::HashMap::new();
            for (class, probability) in spec.classes().iter().zip(&local.probabilities) {
                probabilities.insert(class.clone(), *probability);
            }
            for id in ids {
                answers.insert(
                    id.clone(),
                    Answer::Choice(ChoiceAnswer {
                        choice: local.label.clone(),
                        probabilities: probabilities.clone(),
                        confidence,
                    }),
                );
            }
        }
        let response = SystemResponse {
            model,
            answers,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        };
        // The distilling cache must uphold the same wire contract as the
        // engines it front-runs.
        validate_response_for_request(&response, request).map_err(|error| {
            ApiError::Internal(format!("local answer violates contract: {error}"))
        })?;
        Ok(response)
    }

    fn detail_for_local(&self, routed: &[RoutedGroup]) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        let mut count = 0usize;
        for (_, ids, decision, _) in routed {
            for id in ids {
                count += 1;
                if count <= MAX_DETAIL_QUESTIONS {
                    let value = decision
                        .student_version
                        .clone()
                        .unwrap_or_else(|| decision.reason.as_str().to_owned());
                    map.insert(id.clone(), serde_json::json!(value));
                }
            }
        }
        serde_json::Value::Object(map)
    }
}

/// Group a request's choice questions by (instructions, criteria). Duplicate
/// specs share one routing; the returned order is deterministic.
fn group_questions(request: &SystemRequest) -> (Vec<(TaskSpec, Vec<String>)>, usize) {
    let mut groups: BTreeMap<String, (TaskSpec, Vec<String>)> = BTreeMap::new();
    let mut others = 0usize;
    for (id, question) in request.questions.iter() {
        match question {
            Question::Choice(choice) => {
                let criteria: BTreeMap<String, Option<String>> = choice
                    .criteria
                    .iter()
                    .map(|(name, description)| (name.clone(), description.clone()))
                    .collect();
                let spec = TaskSpec {
                    // The fingerprint canonicalizes the value; store it raw.
                    instructions: choice.instructions.clone(),
                    criteria,
                };
                let key = format!(
                    "{}\u{0}{}",
                    serde_json::to_string(&spec.instructions).unwrap_or_default(),
                    serde_json::to_string(&spec.criteria).unwrap_or_default(),
                );
                let entry = groups.entry(key).or_insert_with(|| (spec, Vec::new()));
                entry.1.push(id.clone());
            }
            _ => others += 1,
        }
    }
    (groups.into_values().collect(), others)
}

/// Extract the teacher answer for one group from an upstream response: the
/// first choice answer among the group's question ids, mapped onto the
/// task's class order.
fn teacher_answer_for(
    response: &SystemResponse,
    ids: &[String],
    classes: &[String],
) -> Option<TeacherAnswer> {
    for id in ids {
        let Some(answer) = response.answers.get(id) else {
            continue;
        };
        let Answer::Choice(choice) = answer else {
            continue;
        };
        let ChoiceAnswer {
            choice,
            probabilities,
            confidence,
            ..
        } = choice;
        let aligned: Vec<f64> = classes
            .iter()
            .map(|class| probabilities.get(class).copied().unwrap_or(0.0))
            .collect();
        return Some(TeacherAnswer {
            label: choice.clone(),
            probabilities: aligned,
            confidence: *confidence,
            model: response.model.clone(),
            latency_ms: 0.0,
        });
    }
    None
}

fn detail_map<'a>(ids: impl Iterator<Item = &'a String>, reason: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    let mut count = 0usize;
    for id in ids {
        count += 1;
        if count <= MAX_DETAIL_QUESTIONS {
            map.insert(id.clone(), serde_json::json!(reason));
        }
    }
    serde_json::Value::Object(map)
}

/// Map an upstream client failure onto the local error surface.
fn map_upstream_error(error: openkind_client::Error, has_upstream_key: bool) -> ApiError {
    if !has_upstream_key && matches!(error, openkind_client::Error::Config(_)) {
        return ApiError::Unauthorized;
    }
    match error.status() {
        Some(401) | Some(403) => ApiError::Unauthorized,
        Some(422) => ApiError::InvalidBody("upstream rejected the request".into()),
        Some(429) => ApiError::RateLimited {
            retry_after_ms: error
                .retry_after()
                .map(|duration| duration.as_millis() as u64)
                .unwrap_or(1_000),
        },
        Some(status) => ApiError::BadGateway(format!("upstream returned status {status}")),
        None => match &error {
            openkind_client::Error::Timeout { .. } => {
                ApiError::BadGateway("upstream timed out".into())
            }
            other => ApiError::BadGateway(format!("upstream unreachable: {other}")),
        },
    }
}

#[async_trait]
impl SystemProxy for ProxyService {
    fn wants(&self, request: &SystemRequest) -> bool {
        self.proxied_models.contains(&request.model)
    }

    async fn evaluate(
        &self,
        request: SystemRequest,
        caller_key: Option<String>,
    ) -> Result<ProxyOutcome, ApiError> {
        match self
            .evaluate_inner(request.clone(), caller_key.clone())
            .await
        {
            Ok(outcome) => Ok(outcome),
            // Fail open: any proxy-internal error forwards to the upstream.
            Err(ApiError::Internal(_)) => {
                tracing::warn!("proxy-cache internal failure, forwarding upstream");
                self.plain_forward(request, caller_key.as_deref(), "internal_error")
                    .await
            }
            Err(error) => Err(error),
        }
    }

    async fn models(&self) -> Option<openkind_api::ModelsResponse> {
        let client = self.client_for(self.upstream_key.as_deref()).ok()?;
        client.list_models().await.ok()
    }
}

/// A forward-only decision engine for the gRPC surface: proxy mode keeps
/// gRPC working by relaying evaluations upstream with the configured
/// upstream key (caller-key passthrough over gRPC is not supported).
pub struct ProxyForwardEngine {
    service: Arc<ProxyService>,
    alias: String,
}

impl ProxyForwardEngine {
    /// Register this engine for one proxied alias.
    pub fn new(service: Arc<ProxyService>, alias: impl Into<String>) -> Self {
        Self {
            service,
            alias: alias.into(),
        }
    }
}

#[async_trait]
impl DecisionEngine for ProxyForwardEngine {
    fn backend_id(&self) -> &str {
        "proxy-cache/upstream-forward"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: self.alias.clone(),
            description:
                "Distilling proxy-cache alias; gRPC evaluations forward to the upstream Jev API"
                    .into(),
            release_date: String::new(),
        }
    }

    async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
        let outcome =
            self.service
                .evaluate(req, None)
                .await
                .map_err(|error| EngineError::Backend {
                    backend: "proxy-cache".into(),
                    message: error.to_string(),
                })?;
        Ok(outcome.response)
    }
}

const _: fn() = || {
    // Compile-time contract checks.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ProxyService>();
};

/// Default proxy-cache state directory (sibling of the model store).
pub fn default_proxy_cache_dir() -> Result<PathBuf, ProxyCacheError> {
    if let Some(path) = std::env::var_os("OPENKIND_PROXY_CACHE_DATA_DIR") {
        if path.is_empty() {
            return Err(ProxyCacheError::Contract(
                "OPENKIND_PROXY_CACHE_DATA_DIR is empty".into(),
            ));
        }
        return Ok(PathBuf::from(path));
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME").ok_or_else(|| {
            ProxyCacheError::Contract(
                "HOME is unavailable; set OPENKIND_PROXY_CACHE_DATA_DIR".into(),
            )
        })?;
        Ok(PathBuf::from(home).join("Library/Application Support/openkind/proxy-cache"))
    }
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("APPDATA")
            .or_else(|| {
                std::env::var_os("USERPROFILE")
                    .map(|home| PathBuf::from(home).join("AppData/Roaming").into_os_string())
            })
            .ok_or_else(|| {
                ProxyCacheError::Contract(
                    "APPDATA is unavailable; set OPENKIND_PROXY_CACHE_DATA_DIR".into(),
                )
            })?;
        Ok(PathBuf::from(base).join("openkind/proxy-cache"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
            })
            .ok_or_else(|| {
                ProxyCacheError::Contract(
                    "HOME is unavailable; set OPENKIND_PROXY_CACHE_DATA_DIR".into(),
                )
            })?;
        Ok(base.join("openkind/proxy-cache"))
    }
}

/// Resolve the proxy-cache encoder from its `openkind pull` name.
///
/// `hash` builds the dependency-free embedder. Anything else is a model
/// store name: an installed profile is used in place; otherwise the daemon
/// attempts one explicit pull and fails closed with a message telling the
/// operator to download the model (regular or MLX profile) first.
pub async fn resolve_encoder(
    encoder_name: &str,
    backend: crate::args::ProxyCacheEncoderBackendArg,
    cuda_device: usize,
    models_dir: Option<&std::path::Path>,
) -> anyhow::Result<(
    Arc<dyn openkind_backends::proxy_cache::TextEmbedder>,
    Option<openkind_model_store::InstalledModel>,
)> {
    let _ = cuda_device;
    use openkind_backends::proxy_cache::{
        bert_encoder::{BertEmbedder, BertEmbedderArtifacts},
        HashEmbedder, TextEmbedder,
    };

    if encoder_name.eq_ignore_ascii_case("hash") {
        let embedder = HashEmbedder::new(512, 0, true)
            .map_err(|error| anyhow::anyhow!("proxy cache encoder `hash`: {error}"))?;
        return Ok((Arc::new(embedder), None));
    }

    // Downloads never happen during serving startup (model-store
    // invariant): the operator runs `openkind pull <name>` explicitly, and
    // a missing installation fails closed with that instruction.
    let dir = match models_dir {
        Some(dir) => dir.to_path_buf(),
        None => openkind_model_store::default_models_dir()
            .map_err(|error| anyhow::anyhow!("resolve models dir: {error}"))?,
    };
    let store = openkind_model_store::ModelStore::new(dir)?;
    let installed = store.acquire_serving(encoder_name).map_err(|error| {
        anyhow::anyhow!(
            "proxy cache encoder `{}` is not installed ({}). \
             Download the model first with `openkind pull {}` (regular or MLX profile), \
             then restart openkindd.",
            encoder_name,
            error,
            encoder_name
        )
    })?;

    let manifest = &installed.manifest;
    if manifest.loader_id != "encoder-embedding" {
        return Err(anyhow::anyhow!(
            "profile `{}` (loader `{}`) cannot serve as a proxy-cache encoder; \
             use `hash` or an `encoder-embedding` profile",
            encoder_name,
            manifest.loader_id
        ));
    }
    let artifacts = BertEmbedderArtifacts::from_model_root(&installed.root, "encoder-embedding");
    let embedder: Arc<dyn TextEmbedder> = crate::backend::load(
        backend,
        &installed.root,
        crate::args::DeviceOrdinals {
            cuda: cuda_device,
            rocm: 0,
        },
        |backend| {
            let embedder: Arc<dyn TextEmbedder> = match backend {
                crate::args::ProxyCacheEncoderBackendArg::Auto => {
                    unreachable!("auto resolves before loading")
                }
                crate::args::ProxyCacheEncoderBackendArg::Cpu => {
                    Arc::new(BertEmbedder::load(&artifacts).context("load encoder `proxy-cache`")?)
                }
                #[cfg(feature = "cuda")]
                crate::args::ProxyCacheEncoderBackendArg::Cuda => Arc::new(
                    BertEmbedder::load_with_device(
                        &artifacts,
                        openkind_backends::device::FamilyExecution::Cuda {
                            device_id: cuda_device,
                        }
                        .candle_device()
                        .context("open CUDA device")?,
                    )
                    .context("load encoder `proxy-cache`")?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                crate::args::ProxyCacheEncoderBackendArg::MlxFp32 => Arc::new(
                    openkind_backends::proxy_cache::mlx_bert_encoder::MlxBertEmbedder::load(
                        &artifacts,
                    )
                    .context("load MLX encoder `proxy-cache`")?,
                ),
            };
            Ok(embedder)
        },
    )?;
    Ok((embedder, Some(installed)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_backends::proxy_cache::engine::LocalPrediction;

    fn decision(reason: RoutingReason, channel: Channel) -> RouteDecision {
        RouteDecision {
            reason,
            channel,
            local: Some(LocalPrediction {
                label: "alpha".into(),
                probabilities: vec![0.9, 0.1],
                routing_confidence: 0.9,
                ood: 0.0,
            }),
            student_version: Some("student-v1".into()),
        }
    }

    #[test]
    fn only_confident_student_decisions_are_locally_servable() {
        assert!(is_locally_servable(&decision(
            RoutingReason::Confident,
            Channel::Student
        )));
        assert!(!is_locally_servable(&decision(
            RoutingReason::Audit,
            Channel::Audit
        )));
        assert!(!is_locally_servable(&RouteDecision {
            reason: RoutingReason::Confident,
            channel: Channel::Student,
            local: None,
            student_version: Some("student-v1".into()),
        }));
    }

    #[test]
    fn forwarding_preserves_the_audit_channel() {
        let audit = decision(RoutingReason::Audit, Channel::Audit);
        assert_eq!(decision_for_teacher(&audit).channel, Channel::Audit);

        let student = decision(RoutingReason::Confident, Channel::Student);
        assert_eq!(decision_for_teacher(&student).channel, Channel::CoDeferred);
    }
}
