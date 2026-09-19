//! `opendecision-proto`: Protocol Buffers and gRPC wire contracts for `opendecision`.
//!
//! # Overview
//! This crate contains the compiled Protobuf bindings for the `opendecision` service,
//! generated from `proto/proto/opendecision.proto` at build time via `tonic-prost-build`.
//!
//! # Invariants & Wire Parity
//! - **64-bit Floating Point**: All probabilities, scores, and confidence metrics use
//!   double precision (`double`), guaranteeing wire parity with `f64` in `opendecision-core`.
//! - **Raw JSON Polymorphism**: Fields that accept polymorphic types in the JSON wire
//!   format (`state` and question `instructions`) serialize arbitrary JSON structures as raw UTF-8 bytes.
//! - **Noul Answer Wire Contract**: Boolean probability (`Noul`) answers contain only `double noul = 1`
//!   and never carry a `confidence` field, adhering strictly to the Jev specification.

#![allow(clippy::all)]

/// Generated Protobuf message types and gRPC client/server stubs for the `opendecision` package.
#[allow(missing_docs)]
pub mod opendecision {
    tonic::include_proto!("opendecision");
}

#[cfg(test)]
mod tests {
    use super::opendecision::*;
    use prost::Message;
    use std::collections::HashMap;

    #[test]
    fn test_system_one_request_protobuf_roundtrip() {
        let mut questions = HashMap::new();
        questions.insert(
            "urgent".into(),
            Question {
                kind: Some(question::Kind::Noul(NoulQuestion {
                    instructions_json: bytes::Bytes::from(r#""Is it urgent?""#),
                    criteria: Some(NoulCriteria {
                        is_true: "Yes".into(),
                        is_false: "No".into(),
                    }),
                })),
            },
        );
        let mut choice_criteria = HashMap::new();
        choice_criteria.insert("billing".into(), "Billing issues".into());
        questions.insert(
            "team".into(),
            Question {
                kind: Some(question::Kind::Choice(ChoiceQuestion {
                    instructions_json: bytes::Bytes::from(r#""Which team?""#),
                    criteria: choice_criteria,
                })),
            },
        );
        questions.insert(
            "severity".into(),
            Question {
                kind: Some(question::Kind::Score(ScoreQuestion {
                    instructions_json: bytes::Bytes::from(r#""How severe?""#),
                    criteria: vec!["Low".into(), "High".into()],
                })),
            },
        );

        let req = SystemOneRequest {
            state: Some(State {
                value: Some(state::Value::Structured(Structured {
                    json: bytes::Bytes::from(r#"{"ticket_id": 123}"#),
                })),
            }),
            model: "jev-latest".into(),
            questions,
        };

        let mut buf = Vec::new();
        req.encode(&mut buf).expect("encode succeeds");
        let decoded = SystemOneRequest::decode(&buf[..]).expect("decode succeeds");
        assert_eq!(req, decoded);
    }

    #[test]
    fn test_system_one_response_protobuf_roundtrip() {
        let mut answers = HashMap::new();

        // Noul
        answers.insert(
            "urgent".into(),
            Answer {
                kind: Some(answer::Kind::Noul(NoulAnswer { noul: 0.92 })),
            },
        );

        // Choice
        let mut choice_probs = HashMap::new();
        choice_probs.insert("billing".into(), 0.85);
        choice_probs.insert("tech".into(), 0.15);
        answers.insert(
            "team".into(),
            Answer {
                kind: Some(answer::Kind::Choice(ChoiceAnswer {
                    choice: "billing".into(),
                    probabilities: choice_probs,
                    confidence: 0.82,
                })),
            },
        );

        // Score
        let mut legend = HashMap::new();
        legend.insert("0".into(), "Low".into());
        legend.insert("1".into(), "High".into());
        let mut score_probs = HashMap::new();
        score_probs.insert("0".into(), 0.3);
        score_probs.insert("1".into(), 0.7);
        answers.insert(
            "severity".into(),
            Answer {
                kind: Some(answer::Kind::Score(ScoreAnswer {
                    score: 0.7,
                    legend,
                    probabilities: score_probs,
                    confidence: 0.75,
                })),
            },
        );

        let resp = SystemOneResponse {
            model: "mock".into(),
            answers,
            usage: Some(Usage {
                input_tokens: 128,
                output_tokens: 6,
            }),
        };

        let mut buf = Vec::new();
        resp.encode(&mut buf).expect("encode succeeds");
        let decoded = SystemOneResponse::decode(&buf[..]).expect("decode succeeds");
        assert_eq!(resp, decoded);
    }
}
