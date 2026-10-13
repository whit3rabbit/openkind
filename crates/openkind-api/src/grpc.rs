//! gRPC layer — tonic 0.14.
//!
//! Same engine surface as the HTTP layer. The Python SDK and other
//! latency-sensitive clients should use this instead of HTTP/JSON.

use std::collections::HashMap;

use openkind_core::{Answer, Question, State, SystemRequest};
use openkind_engine::{dispatch, EngineRegistry};
use openkind_proto::openkind as pb;
use tonic::{Request, Response, Status};

use crate::error::{classify_engine, retry_after_secs};
use crate::middleware::{AuthConfig, RequestLimits, REQUEST_ID_HEADER};
use crate::ApiError;
use crate::AppState;

/// gRPC service implementation of the `openkind.SystemOne` service contract.
pub struct SystemOneService {
    /// Shared application state containing the model registry.
    pub state: AppState,
    /// Authentication configuration.
    pub auth: AuthConfig,
    /// Transport budgets, optionally shared with the HTTP listener.
    pub limits: RequestLimits,
}

impl SystemOneService {
    /// Construct a new `SystemOneService` backed by the specified engine registry and default auth.
    pub fn new(registry: EngineRegistry) -> Self {
        Self::with_auth(registry, AuthConfig::default())
    }

    /// Construct a new `SystemOneService` backed by the specified engine registry and auth configuration.
    pub fn with_auth(registry: EngineRegistry, auth: AuthConfig) -> Self {
        Self::with_auth_and_limits(registry, auth, RequestLimits::default())
    }

    /// Construct the service with explicit, potentially shared transport budgets.
    pub fn with_auth_and_limits(
        registry: EngineRegistry,
        auth: AuthConfig,
        limits: RequestLimits,
    ) -> Self {
        Self {
            state: AppState::new(registry),
            auth,
            limits,
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
            .get(REQUEST_ID_HEADER.as_str())
            .and_then(|m| m.to_str().ok())
        {
            Some(id) if crate::middleware::is_safe_request_id(id) => id.to_string(),
            _ => uuid::Uuid::new_v4().to_string(),
        };

        // Authenticate request if an API key is configured.
        if self.auth.is_required() && !check_grpc_auth(request.metadata(), &self.auth) {
            metrics::counter!("openkind_auth_failures_total", "transport" => "grpc").increment(1);
            if let Some(peer) = request.remote_addr() {
                if let Err(ms) = self.limits.failed_auth.check(peer.ip()) {
                    return Err(status_with_request_id(
                        rate_limited_status("authentication rate limited", ms),
                        &req_id,
                    ));
                }
            }
            return Err(status_with_request_id(unauthenticated_status(), &req_id));
        }

        if let Some(peer) = request.remote_addr() {
            if let Err(ms) = self.limits.evaluation.check(peer.ip()) {
                return Err(status_with_request_id(
                    rate_limited_status("rate limited", ms),
                    &req_id,
                ));
            }
        }

        let pb_req = request.into_inner();

        // Convert protobuf → core.
        let state = pb_state_to_core(pb_req.state.as_ref())
            .map_err(|status| status_with_request_id(status, &req_id))?;
        let questions = pb_questions_to_core(pb_req.questions)
            .map_err(|status| status_with_request_id(status, &req_id))?;
        let req = SystemRequest {
            state,
            model: pb_req.model,
            questions,
        };

        let resp = dispatch(req, &self.state.registry)
            .await
            .map_err(|error| status_with_request_id(status_from_engine(error), &req_id))?;

        let mut response = Response::new(core_to_pb_response(resp));
        // Stamp x-typesafe-request-id on every gRPC response. The Python
        // SDK exposes `response.raw_http_response` for HTTP, but for gRPC
        // request_id is a metadata header — emit it here for parity.
        response.metadata_mut().append(
            request_id_key(),
            req_id
                .parse()
                .unwrap_or_else(|_| tonic::metadata::MetadataValue::from_static("invalid")),
        );
        Ok(response)
    }
}

fn status_with_retry(mut status: Status, ms: u64) -> Status {
    if let Ok(value) = ms.to_string().parse() {
        status.metadata_mut().insert("retry-after-ms", value);
    }
    if let Ok(value) = retry_after_secs(ms).to_string().parse() {
        status.metadata_mut().insert("retry-after", value);
    }
    status
}

/// Transport-level throttle (evaluation or failed-auth budget). Code and
/// retry pairing come from the shared classification; the message is a fixed
/// literal chosen by the caller.
fn rate_limited_status(message: &'static str, ms: u64) -> Status {
    let c = ApiError::RateLimited { retry_after_ms: ms }.classify();
    status_with_retry(Status::new(c.grpc, message), ms)
}

fn unauthenticated_status() -> Status {
    Status::new(
        ApiError::Unauthorized.classify().grpc,
        "missing or invalid API key",
    )
}

/// Metadata key for the request id, derived from the HTTP header constant so
/// the two transports cannot disagree on the name.
fn request_id_key() -> tonic::metadata::MetadataKey<tonic::metadata::Ascii> {
    tonic::metadata::MetadataKey::from_bytes(REQUEST_ID_HEADER.as_str().as_bytes())
        .expect("request id header is a valid metadata key")
}

fn status_with_request_id(mut status: Status, request_id: &str) -> Status {
    // Conversion failures happen before dispatch, but need the same
    // correlation metadata as authentication and engine failures.
    if let Ok(value) = request_id.parse() {
        status.metadata_mut().insert(request_id_key(), value);
    }
    status
}

fn check_grpc_auth(metadata: &tonic::metadata::MetadataMap, auth: &AuthConfig) -> bool {
    if !auth.is_required() {
        return true;
    }

    let supplied = metadata
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            let (scheme, token) = s.split_once(' ')?;
            scheme.eq_ignore_ascii_case("Bearer").then_some(token)
        })
        .or_else(|| metadata.get("x-api-key").and_then(|v| v.to_str().ok()));

    match supplied {
        Some(token) => auth.token_matches(token),
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
) -> Result<HashMap<String, Question, openkind_core::WireHashState>, Status> {
    if pb.len() > openkind_core::MAX_QUESTIONS_PER_REQUEST {
        return Err(Status::invalid_argument(format!(
            "request exceeds maximum question count limit (got {}, max {})",
            pb.len(),
            openkind_core::MAX_QUESTIONS_PER_REQUEST
        )));
    }
    let cap = pb.len().min(openkind_core::MAX_QUESTIONS_PER_REQUEST);
    let mut out: HashMap<String, Question, openkind_core::WireHashState> =
        HashMap::with_capacity_and_hasher(cap, Default::default());
    for (id, q) in pb {
        let kind = q
            .kind
            .ok_or_else(|| Status::invalid_argument("question has no kind"))?;
        let core = match kind {
            pb::question::Kind::Noul(n) => {
                let instr = parse_json(&n.instructions_json)?;
                let criteria = n.criteria.map(|c| openkind_core::NoulCriteria {
                    r#true: c.is_true,
                    r#false: c.is_false,
                });
                Question::Noul(openkind_core::NoulQuestion {
                    instructions: instr,
                    criteria,
                })
            }
            pb::question::Kind::Choice(c) => {
                if c.criteria.len() > openkind_core::MAX_CRITERIA_OPTIONS {
                    return Err(Status::invalid_argument(format!(
                        "choice question `{id}` exceeds maximum criteria options limit (got {}, max {})",
                        c.criteria.len(),
                        openkind_core::MAX_CRITERIA_OPTIONS
                    )));
                }
                let instr = parse_json(&c.instructions_json)?;
                let criteria = c
                    .criteria
                    .into_iter()
                    .map(|(k, v)| (k, if v.is_empty() { None } else { Some(v) }))
                    .collect();
                Question::Choice(openkind_core::ChoiceQuestion {
                    instructions: instr,
                    criteria,
                })
            }
            pb::question::Kind::Score(s) => {
                if s.criteria.len() > openkind_core::MAX_CRITERIA_OPTIONS {
                    return Err(Status::invalid_argument(format!(
                        "score question `{id}` exceeds maximum criteria options limit (got {}, max {})",
                        s.criteria.len(),
                        openkind_core::MAX_CRITERIA_OPTIONS
                    )));
                }
                let instr = parse_json(&s.instructions_json)?;
                Question::Score(openkind_core::ScoreQuestion {
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

fn core_to_pb_response(resp: openkind_core::SystemResponse) -> pb::SystemOneResponse {
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

fn status_from_engine(e: openkind_engine::EngineError) -> Status {
    let c = classify_engine(&e);
    let status = Status::new(c.grpc, e.to_string());
    match c.retry_after_ms {
        Some(ms) => status_with_retry(status, ms),
        None => status,
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

/// Build a gRPC service using transport budgets shared with the HTTP router.
pub fn service_with_auth_and_limits(
    registry: EngineRegistry,
    auth: AuthConfig,
    limits: RequestLimits,
) -> pb::system_one_server::SystemOneServer<SystemOneService> {
    pb::system_one_server::SystemOneServer::new(SystemOneService::with_auth_and_limits(
        registry, auth, limits,
    ))
    .max_decoding_message_size(16 * 1024 * 1024)
    .max_encoding_message_size(16 * 1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiError;
    use axum::{http::StatusCode, response::IntoResponse};
    use openkind_core::ValidationError;
    use openkind_engine::EngineError;
    use tonic::Code;

    struct Row {
        name: &'static str,
        make: fn() -> EngineError,
        http: u16,
        code: &'static str,
        grpc: Code,
        retry_ms: Option<u64>,
    }

    fn rows() -> Vec<Row> {
        vec![
            Row {
                name: "invalid",
                make: || EngineError::Invalid(ValidationError::NoQuestions),
                http: 422,
                code: "invalid_body",
                grpc: Code::InvalidArgument,
                retry_ms: None,
            },
            Row {
                name: "unknown_model",
                make: || EngineError::UnknownModel("m".into()),
                http: 404,
                code: "unknown_model",
                grpc: Code::NotFound,
                retry_ms: None,
            },
            Row {
                name: "unsupported",
                make: || EngineError::Unsupported {
                    backend: "b".into(),
                    message: "x".into(),
                },
                http: 422,
                code: "invalid_body",
                grpc: Code::InvalidArgument,
                retry_ms: None,
            },
            Row {
                name: "overloaded",
                make: || EngineError::Overloaded {
                    backend: "b".into(),
                    retry_after_ms: 1500,
                },
                http: 529,
                code: "overloaded",
                grpc: Code::Unavailable,
                retry_ms: Some(1500),
            },
            Row {
                name: "deadline",
                make: || EngineError::DeadlineExceeded {
                    backend: "b".into(),
                    timeout_ms: 10,
                },
                http: 504,
                code: "deadline_exceeded",
                grpc: Code::DeadlineExceeded,
                retry_ms: None,
            },
            Row {
                name: "backend",
                make: || EngineError::Backend {
                    backend: "b".into(),
                    message: "m".into(),
                },
                http: 500,
                code: "backend_error",
                grpc: Code::Internal,
                retry_ms: None,
            },
            Row {
                name: "backend_validation",
                make: || EngineError::BackendValidation {
                    backend: "b".into(),
                    source: ValidationError::MissingAnswer("q".into()),
                },
                http: 500,
                code: "backend_error",
                grpc: Code::Internal,
                retry_ms: None,
            },
        ]
    }

    async fn http_parts(e: EngineError) -> (StatusCode, axum::http::HeaderMap, serde_json::Value) {
        use http_body_util::BodyExt;
        let resp = ApiError::Engine(e).into_response();
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, headers, serde_json::from_slice(&bytes).unwrap())
    }

    /// One table drives both transports so a new error category cannot be
    /// mapped on one side only.
    #[tokio::test]
    async fn engine_errors_map_symmetrically_over_http_and_grpc() {
        for row in rows() {
            let (status, headers, body) = http_parts((row.make)()).await;
            assert_eq!(status.as_u16(), row.http, "{} http status", row.name);
            assert_eq!(body["error"]["code"], row.code, "{} http code", row.name);

            let grpc_err = (row.make)();
            let grpc_msg = grpc_err.to_string();
            let grpc = status_from_engine(grpc_err);
            assert_eq!(grpc.code(), row.grpc, "{} grpc code", row.name);
            assert_eq!(grpc.message(), grpc_msg, "{} grpc message", row.name);

            match row.retry_ms {
                Some(ms) => {
                    let secs = ms.div_ceil(1000).to_string();
                    assert_eq!(headers["retry-after-ms"], ms.to_string().as_str());
                    assert_eq!(headers["retry-after"], secs.as_str());
                    assert_eq!(
                        grpc.metadata().get("retry-after-ms").unwrap(),
                        ms.to_string().as_str()
                    );
                    assert_eq!(grpc.metadata().get("retry-after").unwrap(), secs.as_str());
                }
                None => {
                    assert!(headers.get("retry-after-ms").is_none(), "{}", row.name);
                    assert!(headers.get("retry-after").is_none(), "{}", row.name);
                    assert!(
                        grpc.metadata().get("retry-after-ms").is_none(),
                        "{}",
                        row.name
                    );
                    assert!(grpc.metadata().get("retry-after").is_none(), "{}", row.name);
                }
            }
        }
    }

    #[test]
    fn transport_level_statuses_keep_codes_and_retry_metadata() {
        let unauth = unauthenticated_status();
        assert_eq!(unauth.code(), Code::Unauthenticated);
        assert_eq!(unauth.message(), "missing or invalid API key");

        let limited = rate_limited_status("rate limited", 2001);
        assert_eq!(limited.code(), Code::ResourceExhausted);
        assert_eq!(limited.message(), "rate limited");
        assert_eq!(limited.metadata().get("retry-after-ms").unwrap(), "2001");
        assert_eq!(limited.metadata().get("retry-after").unwrap(), "3");
    }
}
