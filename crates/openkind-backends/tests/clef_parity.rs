//! Offline parity and failure-mode tests for the pinned `clef` family.
//!
//! Unit-level contract checks always run. The golden replays run only when
//! the pinned checkpoints are present locally (point `OPENKIND_CLEF_FLASH_MODEL_ROOT`,
//! `OPENKIND_CLEF_FLASH_GGUF_MODEL_ROOT`, or `OPENKIND_CLEF_27B_GGUF_MODEL_ROOT`
//! at the model root); tests never download model artifacts.
//!
//! The three fixture requests are the `gen_clef_fixture` cases: a joint
//! multi-question record (Noul + Choice + Score — the head's designed
//! operating mode), a stated-verbatim Choice, and a text-passthrough Noul.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::clef::{profile_by_loader_id, ClefEngine, PROFILES};
use openkind_backends::families::support::{derive_profile_id, FamilyLimits};
use openkind_core::SystemResponse;
use openkind_engine::DecisionEngine;
use serde::Deserialize;
use serde_json::json;

/// Cross-execution agreement: quantized profiles reproduce the BF16 oracle
/// within this probability tolerance.
const PROBABILITY_TOLERANCE: f64 = 0.02;

fn fixture_dir(profile_id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/clef_{profile_id}"))
}

fn model_root(env: &str) -> Option<PathBuf> {
    std::env::var_os(env)
        .map(PathBuf::from)
        .filter(|root| root.is_dir())
}

fn limits() -> FamilyLimits {
    FamilyLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 0,
        retry_after_ms: 100,
        evaluation_timeout: None,
    }
}

/// The fixture requests, in generator order.
fn fixture_requests() -> Vec<serde_json::Value> {
    vec![
        json!({
            "model": "clef-fixture",
            "state": {
                "ticket": "A-771",
                "priority": "high",
                "category": "billing",
                "refund_approved": true,
                "amount_usd": 412.5
            },
            "questions": {
                "billing_refund": {
                    "type": "noul",
                    "instructions": "Was the refund approved?"
                },
                "category": {
                    "type": "choice",
                    "instructions": "Which category does the ticket belong to?",
                    "criteria": {
                        "billing": "Charges, refunds, invoices",
                        "security": "Account takeover, phishing",
                        "performance": "Latency and outages"
                    }
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate operational severity.",
                    "criteria": ["low", "moderate", "high", "critical"]
                }
            }
        }),
        json!({
            "model": "clef-fixture",
            "state": "Incident log: the failing component is the scheduler. All other components healthy.",
            "questions": {
                "failing_component": {
                    "type": "choice",
                    "instructions": "Which component is failing?",
                    "criteria": {
                        "api_gateway": "Front door",
                        "scheduler": "Job scheduler",
                        "storage": "Object storage"
                    }
                }
            }
        }),
        json!({
            "model": "clef-fixture",
            "state": "The customer explicitly declined the renewal offer on 2026-09-14.",
            "questions": {
                "renewal": {
                    "type": "noul",
                    "instructions": "Did the customer decline the renewal?"
                }
            }
        }),
    ]
}

#[derive(Debug, Deserialize)]
struct Golden {
    cases: Vec<SystemResponse>,
}

fn profile_contract_checks() {
    for profile in PROFILES {
        assert_eq!(
            profile.profile_id,
            derive_profile_id("clef", profile.backbone_id, profile.backbone_revision),
            "profile {} id derivation",
            profile.loader_id,
        );
    }
    assert!(profile_by_loader_id("clef-flash").is_some());
    assert!(profile_by_loader_id("clef-flash-gguf").is_some());
    assert!(profile_by_loader_id("clef-27b-gguf").is_some());
    assert!(profile_by_loader_id("clef-flash-mlx-4bit").is_none());
}

fn compare_answers(golden: &SystemResponse, actual: &SystemResponse) {
    assert_eq!(
        golden.answers.len(),
        actual.answers.len(),
        "answer count mismatch"
    );
    for (id, golden_answer) in &golden.answers {
        let actual_answer = actual
            .answers
            .get(id)
            .unwrap_or_else(|| panic!("missing answer {id}"));
        let golden_probabilities = answer_probabilities(golden_answer);
        let actual_probabilities = answer_probabilities(actual_answer);
        assert_eq!(
            golden_probabilities.len(),
            actual_probabilities.len(),
            "answer {id} option count"
        );
        for (index, (expected, found)) in golden_probabilities
            .iter()
            .zip(&actual_probabilities)
            .enumerate()
        {
            let difference = (expected - found).abs();
            assert!(
                difference <= PROBABILITY_TOLERANCE,
                "answer {id} option {index} drifted by {difference:.5} (expected {expected:.4}, found {found:.4})"
            );
        }
    }
}

/// Extract the comparable probability vector of one wire answer in its
/// serialized option order.
fn answer_probabilities(answer: &openkind_core::Answer) -> Vec<f64> {
    match answer {
        openkind_core::Answer::Noul(noul) => vec![noul.noul],
        openkind_core::Answer::Choice(choice) => {
            let mut keys: Vec<&String> = choice.probabilities.keys().collect();
            keys.sort();
            keys.iter().map(|key| choice.probabilities[*key]).collect()
        }
        openkind_core::Answer::Score(score) => {
            let mut keys: Vec<&String> = score.probabilities.keys().collect();
            keys.sort();
            keys.iter().map(|key| score.probabilities[*key]).collect()
        }
    }
}

fn replay(profile_loader_id: &str, env_var: &str) {
    let Some(root) = model_root(env_var) else {
        return;
    };
    let profile = profile_by_loader_id(profile_loader_id).expect("profile");
    let engine = ClefEngine::load(&root, profile, limits()).expect("engine load");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let golden_path = fixture_dir(profile.profile_id).join("golden.json");
    let golden: Golden =
        serde_json::from_slice(&fs::read(&golden_path).expect("golden file")).expect("golden");
    assert_eq!(golden.cases.len(), fixture_requests().len());
    for (request_value, expected) in fixture_requests().iter().zip(&golden.cases) {
        let request: openkind_core::SystemRequest =
            serde_json::from_value(request_value.clone()).expect("request");
        let actual = runtime
            .block_on(engine.evaluate(request))
            .expect("evaluate");
        compare_answers(expected, &actual);
    }
}

#[test]
fn clef_profile_contract() {
    profile_contract_checks();
}

#[test]
fn clef_flash_golden_replay() {
    replay("clef-flash", "OPENKIND_CLEF_FLASH_MODEL_ROOT");
}

#[test]
fn clef_flash_gguf_golden_replay() {
    replay("clef-flash-gguf", "OPENKIND_CLEF_FLASH_GGUF_MODEL_ROOT");
}

#[test]
fn clef_27b_gguf_golden_replay() {
    replay("clef-27b-gguf", "OPENKIND_CLEF_27B_GGUF_MODEL_ROOT");
}
