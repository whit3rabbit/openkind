//! gRPC layer — tonic 0.14.
//!
//! Same engine surface as the HTTP layer. The Python SDK and other
//! latency-sensitive clients should use this instead of HTTP/JSON.

use std::collections::HashMap;

use openpick_core::{Answer, Question, State, SystemRequest};
use openpick_engine::{dispatch, EngineRegistry};
use openpick_proto::openpick as pb;
use tonic::{Request, Response, Status};

use crate::AppState;

pub struct SystemOneService {
    pub state: AppState,
}

impl SystemOneService {
    pub fn new(registry: EngineRegistry) -> Self {
        Self {
            state: AppState::new(registry),
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
        let pb_req = request.into_inner();

        // Convert protobuf → core.
        let state = pb_state_to_core(pb_req.state.as_ref())?;
        let questions = pb_questions_to_core(pb_req.questions)?;
        let req = SystemRequest {
            state,
            model: pb_req.model,
            questions,
        };

        let resp = dispatch(req, &self.state.registry)
            .await
            .map_err(status_from_engine)?;

        let mut response = Response::new(core_to_pb_response(resp));
        // Stamp x-typesafe-request-id on every gRPC response. The Python
        // SDK exposes `response.raw_http_response` for HTTP, but for gRPC
        // request_id is a metadata header — emit it here for parity.
        let req_id = uuid::Uuid::new_v4().to_string();
        response.metadata_mut().append(
            "x-typesafe-request-id",
            req_id.parse().unwrap_or_else(|_| {
                tonic::metadata::MetadataValue::from_static("invalid")
            }),
        );
        Ok(response)
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
            let v: serde_json::Value =
                serde_json::from_slice(&s.json).map_err(|e| Status::invalid_argument(e.to_string()))?;
            Ok(json_value_to_state(v))
        }
        None => Err(Status::invalid_argument("state must be set")),
    }
}

fn json_value_to_state(v: serde_json::Value) -> State {
    match v {
        serde_json::Value::String(s) => State::Text(s),
        serde_json::Value::Object(m) => State::Object(m),
        other => State::Array(vec![other]),
    }
}

fn pb_questions_to_core(
    pb: HashMap<String, pb::Question>,
) -> Result<HashMap<String, Question>, Status> {
    let mut out = HashMap::with_capacity(pb.len());
    for (id, q) in pb {
        let kind = q.kind.ok_or_else(|| Status::invalid_argument("question has no kind"))?;
        let core = match kind {
            pb::question::Kind::Noul(n) => {
                let instr = parse_json(&n.instructions_json)?;
                let criteria = n
                    .criteria
                    .map(|c| openpick_core::NoulCriteria {
                        r#true: c.is_true,
                        r#false: c.is_false,
                    });
                Question::Noul(openpick_core::NoulQuestion {
                    instructions: instr,
                    criteria,
                })
            }
            pb::question::Kind::Choice(c) => {
                let instr = parse_json(&c.instructions_json)?;
                let criteria = c
                    .criteria
                    .into_iter()
                    .map(|(k, v)| (k, if v.is_empty() { None } else { Some(v) }))
                    .collect();
                Question::Choice(openpick_core::ChoiceQuestion {
                    instructions: instr,
                    criteria,
                })
            }
            pb::question::Kind::Score(s) => {
                let instr = parse_json(&s.instructions_json)?;
                Question::Score(openpick_core::ScoreQuestion {
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

fn core_to_pb_response(resp: openpick_core::SystemResponse) -> pb::SystemOneResponse {
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

fn status_from_engine(e: openpick_engine::EngineError) -> Status {
    use openpick_engine::EngineError::*;
    match e {
        Invalid(_) => Status::invalid_argument(e.to_string()),
        UnknownModel(_) => Status::not_found(e.to_string()),
        Backend { .. } => Status::internal(e.to_string()),
    }
}

/// Build a tonic ServerBuilder pre-configured with the SystemOne service.
/// The HTTP/2 server is bound and run by `openpick-server`.
pub fn server(
    registry: EngineRegistry,
) -> pb::system_one_server::SystemOneServer<SystemOneService> {
    pb::system_one_server::SystemOneServer::new(SystemOneService::new(registry))
}

/// Convenience: full tonic service map.
pub fn service(
    registry: EngineRegistry,
) -> pb::system_one_server::SystemOneServer<SystemOneService> {
    server(registry)
}