//! Offline parity and failure-mode tests for the pinned `sieve-4b` profile
//! (the `sieve` family's causal LoRA-merged code-readout engine).
//!
//! Unit-level contract checks always run. The full golden replay runs only
//! when the pinned artifacts are present locally (point
//! `OPENKIND_SIEVE_MODEL_ROOT` at the installation root); tests never
//! download model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::sieve::{SieveEngine, SieveEngineConfig, FAMILY_SLUG, SIEVE_4B};
use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/sieve_8b97b56524b55c41fbf4";
const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_SIEVE_MODEL_ROOT")
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

#[derive(Debug, Deserialize)]
struct Golden {
    cases: Vec<GoldenCase>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    name: String,
    request: openkind_core::SystemRequest,
    response: GoldenResponse,
}

#[derive(Debug, Deserialize)]
struct GoldenResponse {
    answers: std::collections::BTreeMap<String, serde_json::Value>,
    usage: GoldenUsage,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoldenUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn assert_answer_matches(label: &str, actual: &serde_json::Value, expected: &serde_json::Value) {
    assert_eq!(
        actual["type"], expected["type"],
        "{label}: answer type drifted"
    );
    match expected["type"].as_str().expect("tagged answer") {
        "noul" => {
            let actual = actual["noul"].as_f64().expect("noul");
            let expected = expected["noul"].as_f64().expect("noul");
            assert!(
                (actual - expected).abs() <= PROBABILITY_TOLERANCE,
                "{label}: noul {actual} vs {expected}"
            );
        }
        "choice" => {
            assert_eq!(
                actual["choice"], expected["choice"],
                "{label}: selected option drifted"
            );
            let probabilities = expected["probabilities"]
                .as_object()
                .expect("probabilities");
            for (key, expected_probability) in probabilities {
                let actual_probability = actual["probabilities"][key]
                    .as_f64()
                    .unwrap_or_else(|| panic!("{label}: missing probability for {key}"));
                let expected_probability = expected_probability.as_f64().expect("probability");
                assert!(
                    (actual_probability - expected_probability).abs() <= PROBABILITY_TOLERANCE,
                    "{label}: {key} {actual_probability} vs {expected_probability}"
                );
            }
        }
        "score" => {
            let actual_score = actual["score"].as_f64().expect("score");
            let expected_score = expected["score"].as_f64().expect("score");
            assert!(
                (actual_score - expected_score).abs() <= PROBABILITY_TOLERANCE,
                "{label}: score {actual_score} vs {expected_score}"
            );
            let probabilities = expected["probabilities"]
                .as_object()
                .expect("probabilities");
            for (key, expected_probability) in probabilities {
                let actual_probability = actual["probabilities"][key]
                    .as_f64()
                    .unwrap_or_else(|| panic!("{label}: missing probability for level {key}"));
                let expected_probability = expected_probability.as_f64().expect("probability");
                assert!(
                    (actual_probability - expected_probability).abs() <= PROBABILITY_TOLERANCE,
                    "{label}: level {key} {actual_probability} vs {expected_probability}"
                );
            }
        }
        other => panic!("{label}: unexpected answer type {other}"),
    }
}

#[test]
fn profile_constants_are_self_consistent() {
    assert_eq!(
        SIEVE_4B.profile_id,
        openkind_backends::families::support::derive_profile_id(
            FAMILY_SLUG,
            SIEVE_4B.backbone_id,
            SIEVE_4B.backbone_revision,
        )
    );
    assert_eq!(SIEVE_4B.backbone_id, "sthanika-ai/Sieve-4B");
    assert_eq!(
        SIEVE_4B.backbone_revision,
        "ea7cd1edee29590a82ed9fbfcb644227a8eed5f1"
    );
    // The base is the causal Qwen3.5-4B hybrid (the pinned qwen3.5 dims).
    const {
        assert!(SIEVE_4B.geometry.causal);
        assert!(SIEVE_4B.geometry.hidden_size == 2_560);
        assert!(SIEVE_4B.lora_rank == 32);
        assert!(SIEVE_4B.lora_scale == 2.0);
    }
}

#[test]
fn renderer_derives_the_reference_code_table() {
    let bytes = fs::read(fixture_dir().join("tokenizer-vocab-sample.json"))
        .expect("vendored tokenizer sample is checked in");
    let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).expect("sample parses");
    let derived = openkind_backends::families::sieve::derive_sieve_answer_codes(&tokenizer)
        .expect("sample tokenizer provides the code table");
    assert_eq!(derived.0.len(), 255);
    assert_eq!(derived.0[0], "A");
    assert_eq!(derived.0[26], "AA");
    for (code, id) in derived.0.iter().zip(derived.1.iter()) {
        let decoded = tokenizer.decode(&[*id], false).expect("decode");
        assert_eq!(&decoded, code, "code {code} must round-trip");
    }
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!("openkind-sieve-missing-{}", std::process::id()));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match SieveEngine::load(SieveEngineConfig {
        model_root: empty.clone(),
        limits: limits(),
        profile: &SIEVE_4B,
    }) {
        Err(error) => error,
        Ok(_) => panic!("missing artifacts must fail closed"),
    };
    assert!(
        error.to_string().contains("missing") || error.to_string().contains("read"),
        "unexpected error: {error}"
    );
    let _ = fs::remove_dir_all(&empty);
}

#[test]
fn golden_replay_matches_the_pinned_checkpoint() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_SIEVE_MODEL_ROOT is not set");
        return;
    };
    let golden: Golden = serde_json::from_slice(
        &fs::read(fixture_dir().join("golden.json")).expect("read golden fixture"),
    )
    .expect("decode golden fixture");
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let engine = SieveEngine::load(SieveEngineConfig {
        model_root: root,
        limits: limits(),
        profile: &SIEVE_4B,
    })
    .expect("load pinned engine");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    for case in &golden.cases {
        let response = runtime
            .block_on(engine.evaluate(case.request.clone()))
            .unwrap_or_else(|error| panic!("case {}: {error}", case.name));
        for (question_id, expected) in &case.response.answers {
            let actual = response
                .answers
                .get(question_id)
                .unwrap_or_else(|| panic!("case {}: missing answer {question_id}", case.name));
            let actual = serde_json::to_value(actual).expect("serialize answer");
            assert_answer_matches(&case.name, &actual, expected);
        }
        assert_eq!(
            response.usage.output_tokens, 0,
            "{}: decision engines never generate output tokens",
            case.name
        );
        assert_eq!(
            response.usage.input_tokens, case.response.usage.input_tokens,
            "{}: usage drifted",
            case.name
        );
    }
}
