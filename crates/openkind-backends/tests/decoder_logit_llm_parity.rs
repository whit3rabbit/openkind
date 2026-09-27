//! Offline parity and failure-mode tests for the pinned decoder-letter
//! profile.
//!
//! Unit-level contract checks always run. The full golden replay runs only
//! when the pinned checkpoint is present locally (point
//! `OPENKIND_DECODER_LETTER_MODEL_ROOT` at the model root); tests never
//! download model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::decoder_logit_llm::{
    DecoderLlmEngine, DecoderLlmEngineConfig, CALIBRATION_TEMPERATURE, PROFILE_ID,
};

use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/decoder_logit_llm_465963d705b6f35d6208";
const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_DECODER_LLM_MODEL_ROOT")
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
        PROFILE_ID,
        openkind_backends::families::support::derive_profile_id(
            "decoder-logit-llm",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "9217f5db79a29953eb74d5343926648285ec7e67",
        )
    );
    assert!(CALIBRATION_TEMPERATURE.is_finite() && CALIBRATION_TEMPERATURE > 0.0);
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!(
        "openkind-decoder-llm-missing-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match DecoderLlmEngine::load(DecoderLlmEngineConfig {
        model_root: empty.clone(),
        limits: limits(),
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
fn load_fails_closed_on_tokenizer_digest_mismatch() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_DECODER_LLM_MODEL_ROOT is not set");
        return;
    };
    let staged = std::env::temp_dir().join(format!(
        "openkind-decoder-llm-config-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&staged);
    fs::create_dir_all(&staged).expect("stage root");
    // The tokenizer digest is verified before the multi-hundred-megabyte
    // GGUF checkpoint is streamed, so a drifted tokenizer fails fast; the
    // checkpoint is staged as an empty file placeholder.
    let tokenizer = fs::read_to_string(root.join("tokenizer.json")).expect("tokenizer");
    fs::write(
        staged.join("tokenizer.json"),
        tokenizer.replacen("model", "drifted", 1),
    )
    .expect("write drifted tokenizer");
    fs::write(staged.join("qwen2.5-0.5b-instruct-q8_0.gguf"), []).expect("placeholder gguf");

    let error = match DecoderLlmEngine::load(DecoderLlmEngineConfig {
        model_root: staged.clone(),
        limits: limits(),
    }) {
        Err(error) => error,
        Ok(_) => panic!("drifted config must fail closed"),
    };
    assert!(
        error.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {error}"
    );
    let _ = fs::remove_dir_all(&staged);
}

#[test]
fn golden_replay_matches_the_pinned_checkpoint() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_DECODER_LETTER_MODEL_ROOT is not set");
        return;
    };
    let golden: Golden = serde_json::from_slice(
        &fs::read(fixture_dir().join("golden.json")).expect("read golden fixture"),
    )
    .expect("decode golden fixture");
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let engine = DecoderLlmEngine::load(DecoderLlmEngineConfig {
        model_root: root,
        limits: limits(),
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
// The golden cases are regenerated by `gen_decoder_letter_fixture`; the test
// rebuilds the same requests from the same deterministic case list. Rather
// than duplicating the templates, the fixture stores each request's full
// response; the helpers below re-derive requests from the fixture names via
// the generator's public case construction (kept in sync by the fixture
// provenance fields).
