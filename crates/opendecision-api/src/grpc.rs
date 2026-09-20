//! gRPC layer — tonic 0.14.
//!
//! Same engine surface as the HTTP layer. The Python SDK and other
//! latency-sensitive clients should use this instead of HTTP/JSON.

use std::collections::HashMap;

use opendecision_core::{Answer, Question, State, SystemRequest};
use opendecision_engine::{dispatch, EngineRegistry};
use opendecision_proto::opendecision as pb;
use tonic::{Request, Response, Status};

use crate::middleware::AuthConfig;
use crate::AppState;

/// gRPC service implementation of the `opendecision.SystemOne` service contract.
pub struct SystemOneService {
    /// Shared application state containing the model registry.
    pub state: AppState,
    /// Authentication configuration.
    pub auth: AuthConfig,
}

impl SystemOneService {
    /// Construct a new `SystemOneService` backed by the specified engine registry and default auth.
    pub fn new(registry: EngineRegistry) -> Self {
        Self::with_auth(registry, AuthConfig::default())
    }

    /// Construct a new `SystemOneService` backed by the specified engine registry and auth configuration.
    pub fn with_auth(registry: EngineRegistry, auth: AuthConfig) -> Self {
        Self {
            state: AppState::new(registry),
            auth,
        }
    }
}

type RpcResult<T> = Result<Response<T>, Status>;

#[tonic::async_trait]
impl pb::system_one_server::SystemOne for SystemOneService {
    async fn evaluate(
        &self,
        request: Request<pb::SystemOneRequest>,
    ) -> RpcResult<pb::SystemOneResponse> {
        let req_id = match request
            .metadata()
            .get("x-typesafe-request-id")
            .and_then(|m| m.to_str().ok())
        {
            Some(id) if crate::middleware::is_safe_request_id(id) => id.to_string(),
            _ => uuid::Uuid::new_v4().to_string(),
        };

        // Authenticate request if an API key is configured.
        if self.auth.is_required() && !check_grpc_auth(request.metadata(), &self.auth) {
            let mut status = Status::unauthenticated("missing or invalid API key");
            if let Ok(meta_val) = req_id.parse() {
                status
                    .metadata_mut()
                    .insert("x-typesafe-request-id", meta_val);
            }
            return Err(status);
        }

        let pb_req = request.into_inner();

        // Convert protobuf → core.
        let state = pb_state_to_core(pb_req.state.as_ref())?;
        let questions = pb_questions_to_core(pb_req.questions)?;
        let req = SystemRequest {
            state,
            model: pb_req.model,
            questions,
        };

        let resp = match dispatch(req, &self.state.registry).await {
            Ok(r) => r,
            Err(e) => {
                let mut status = status_from_engine(e);
                if let Ok(meta_val) = req_id.parse() {
                    status
                        .metadata_mut()
                        .insert("x-typesafe-request-id", meta_val);
                }
                return Err(status);
            }
        };

        let mut response = Response::new(core_to_pb_response(resp));
        // Stamp x-typesafe-request-id on every gRPC response. The Python
        // SDK exposes `response.raw_http_response` for HTTP, but for gRPC
        // request_id is a metadata header — emit it here for parity.
        response.metadata_mut().append(
            "x-typesafe-request-id",
            req_id
                .parse()
                .unwrap_or_else(|_| tonic::metadata::MetadataValue::from_static("invalid")),
        );
        Ok(response)
    }
}

fn check_grpc_auth(metadata: &tonic::metadata::MetadataMap, auth: &AuthConfig) -> bool {
    let expected = match auth.expected.as_deref() {
        Some(exp) => exp,
        None => return true,
    };

    let supplied = metadata
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            s.strip_prefix("Bearer ")
                .or_else(|| s.strip_prefix("bearer "))
        })
        .or_else(|| metadata.get("x-api-key").and_then(|v| v.to_str().ok()));

    match supplied {
        Some(token) => crate::middleware::secure_token_eq(token, expected),
        None => false,
    }
}

// ---------- conversions ----------

fn pb_state_to_core(pb: Option<&pb::State>) -> Result<State, Status> {
    let pb = pb.ok_or_else(|| Status::invalid_argument("state is required"))?;
    match &pb.value {
        Some(pb::state::Value::Text(s)) => Ok(State::Text(s.clone())),
        Some(pb::state::Value::Structured(s)) => {
            // `bytes json` carries a JSON document. We re-parse into Value
            // so we can normalize to Object/Array.
            let v: serde_json::Value = serde_json::from_slice(&s.json)
                .map_err(|e| Status::invalid_argument(e.to_string()))?;
            json_value_to_state(v)
        }
        None => Err(Status::invalid_argument("state must be set")),
    }
}

fn json_value_to_state(v: serde_json::Value) -> Result<State, Status> {
    match v {
        serde_json::Value::String(s) => Ok(State::Text(s)),
        serde_json::Value::Object(m) => Ok(State::Object(m)),
        serde_json::Value::Array(a) => Ok(State::Array(a)),
        _ => Err(Status::invalid_argument(
            "structured state must be a JSON object, array, or string",
        )),
    }
}

fn pb_questions_to_core(
    pb: HashMap<String, pb::Question>,
) -> Result<HashMap<String, Question>, Status> {
    if pb.len() > opendecision_core::MAX_QUESTIONS_PER_REQUEST {
        return Err(Status::invalid_argument(format!(
            "request exceeds maximum question count limit (got {}, max {})",
            pb.len(),
            opendecision_core::MAX_QUESTIONS_PER_REQUEST
        )));
    }
    let cap = pb.len().min(opendecision_core::MAX_QUESTIONS_PER_REQUEST);
    let mut out = HashMap::with_capacity(cap);
    for (id, q) in pb {
        let kind = q
            .kind
            .ok_or_else(|| Status::invalid_argument("question has no kind"))?;
        let core = match kind {
            pb::question::Kind::Noul(n) => {
                let instr = parse_json(&n.instructions_json)?;
                let criteria = n.criteria.map(|c| opendecision_core::NoulCriteria {
                    r#true: c.is_true,
                    r#false: c.is_false,
                });
                Question::Noul(opendecision_core::NoulQuestion {
                    instructions: instr,
                    criteria,
                })
            }
            pb::question::Kind::Choice(c) => {
                if c.criteria.len() > opendecision_core::MAX_CRITERIA_OPTIONS {
                    return Err(Status::invalid_argument(format!(
                        "choice question `{id}` exceeds maximum criteria options limit (got {}, max {})",
                        c.criteria.len(),
                        opendecision_core::MAX_CRITERIA_OPTIONS
                    )));
                }
                let instr = parse_json(&c.instructions_json)?;
                let criteria = c
                    .criteria
                    .into_iter()
                    .map(|(k, v)| (k, if v.is_empty() { None } else { Some(v) }))
                    .collect();
                Question::Choice(opendecision_core::ChoiceQuestion {
                    instructions: instr,
                    criteria,
                })
            }
            pb::question::Kind::Score(s) => {
                if s.criteria.len() > opendecision_core::MAX_CRITERIA_OPTIONS {
                    return Err(Status::invalid_argument(format!(
                        "score question `{id}` exceeds maximum criteria options limit (got {}, max {})",
                        s.criteria.len(),
                        opendecision_core::MAX_CRITERIA_OPTIONS
                    )));
                }
                let instr = parse_json(&s.instructions_json)?;
                Question::Score(opendecision_core::ScoreQuestion {
                    instructions: instr,
                    criteria: s.criteria,
                })
            }
        };
        out.insert(id, core);
    }
    Ok(out)
}

fn parse_json(bytes: &[u8]) -> Result<serde_json::Value, Status> {
    if bytes.is_empty() {
        return Ok(serde_json::Value::Null);
    }
    serde_json::from_slice(bytes).map_err(|e| Status::invalid_argument(e.to_string()))
}

fn core_to_pb_response(resp: opendecision_core::SystemResponse) -> pb::SystemOneResponse {
    let answers = resp
        .answers
        .into_iter()
        .map(|(id, ans)| {
            let pb_ans = match ans {
                Answer::Noul(n) => pb::Answer {
                    kind: Some(pb::answer::Kind::Noul(pb::NoulAnswer { noul: n.noul })),
                },
                Answer::Choice(c) => pb::Answer {
                    kind: Some(pb::answer::Kind::Choice(pb::ChoiceAnswer {
                        choice: c.choice,
                        probabilities: c.probabilities,
                        confidence: c.confidence,
                    })),
                },
                Answer::Score(s) => pb::Answer {
                    kind: Some(pb::answer::Kind::Score(pb::ScoreAnswer {
                        score: s.score,
                        legend: s.legend,
                        probabilities: s.probabilities,
                        confidence: s.confidence,
                    })),
                },
            };
            (id, pb_ans)
        })
        .collect();
    pb::SystemOneResponse {
        model: resp.model,
        answers,
        usage: Some(pb::Usage {
            input_tokens: resp.usage.input_tokens,
            output_tokens: resp.usage.output_tokens,
        }),
    }
}

fn status_from_engine(e: opendecision_engine::EngineError) -> Status {
    use opendecision_engine::EngineError::*;
    match e {
        Invalid(_) => Status::invalid_argument(e.to_string()),
        UnknownModel(_) => Status::not_found(e.to_string()),
        Unsupported { .. } => Status::invalid_argument(e.to_string()),
        Overloaded { .. } => Status::unavailable(e.to_string()),
        Backend { .. } => Status::internal(e.to_string()),
    }
}

/// Build a tonic ServerBuilder pre-configured with the SystemOne service and default auth.
pub fn server(
    registry: EngineRegistry,
) -> pb::system_one_server::SystemOneServer<SystemOneService> {
    server_with_auth(registry, AuthConfig::default())
}

/// Build a tonic ServerBuilder pre-configured with the SystemOne service and explicit auth configuration.
pub fn server_with_auth(
    registry: EngineRegistry,
    auth: AuthConfig,
) -> pb::system_one_server::SystemOneServer<SystemOneService> {
    pb::system_one_server::SystemOneServer::new(SystemOneService::with_auth(registry, auth))
        .max_decoding_message_size(16 * 1024 * 1024)
        .max_encoding_message_size(16 * 1024 * 1024)
}

/// Convenience: full tonic service map with default auth.
pub fn service(
    registry: EngineRegistry,
) -> pb::system_one_server::SystemOneServer<SystemOneService> {
    server(registry)
}

/// Convenience: full tonic service map with explicit auth configuration.
pub fn service_with_auth(
    registry: EngineRegistry,
    auth: AuthConfig,
) -> pb::system_one_server::SystemOneServer<SystemOneService> {
    server_with_auth(registry, auth)
}
