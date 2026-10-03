//! Offline parity and failure-mode tests for the pinned `kev` profile.
//!
//! Unit-level contract checks always run. The full golden replay runs only
//! when the pinned artifacts are present locally (point
//! `OPENKIND_KEV_MODEL_ROOT` and `OPENKIND_KEV_BASE_ROOT` at the roots);
//! tests never download model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::kev::{
    KevEngine, KevEngineConfig, BASE_MODEL_ID, BASE_MODEL_REVISION, CALIBRATION_TEMPERATURE,
    PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/kev_39d88c11faeb4ac165fa";
const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_KEV_MODEL_ROOT")
        .map(PathBuf::from)
        .filter(|root| root.is_dir())
}

fn base_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_KEV_BASE_ROOT")
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
            "kev",
            "jaredpalmer/kev-0.6b",
            "dece6dba8d43f0f7ded45e9f5b9df12474d90843",
        )
    );
    assert!(CALIBRATION_TEMPERATURE.is_finite() && CALIBRATION_TEMPERATURE > 0.0);
    assert_eq!(BASE_MODEL_ID, "Qwen/Qwen3-0.6B-Base");
    assert_eq!(BASE_MODEL_REVISION.len(), 40);
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!("openkind-kev-missing-{}", std::process::id()));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match KevEngine::load(KevEngineConfig {
        model_root: empty.clone(),
        base_root: empty.clone(),
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
fn golden_replay_matches_the_pinned_checkpoint() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_KEV_MODEL_ROOT is not set");
        return;
    };
    let Some(base) = base_root() else {
        eprintln!("skipping: OPENKIND_KEV_BASE_ROOT is not set");
        return;
    };
    let golden: Golden = serde_json::from_slice(
        &fs::read(fixture_dir().join("golden.json")).expect("read golden fixture"),
    )
    .expect("decode golden fixture");
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let engine = KevEngine::load(KevEngineConfig {
        model_root: root,
        base_root: base,
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

/// Accelerated ONNX selections fail closed on a family whose dual-backbone
/// readout has no ONNX export; checkpoint-gated because artifact
/// verification runs before the execution selection.
#[test]
fn accelerated_executions_fail_closed_without_an_export() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_KEV_MODEL_ROOT is not set");
        return;
    };
    let Some(base) = base_root() else {
        eprintln!("skipping: OPENKIND_KEV_BASE_ROOT is not set");
        return;
    };

    #[cfg(feature = "onnx")]
    {
        let error = match KevEngine::load_with_execution(
            KevEngineConfig {
                model_root: root.clone(),
                base_root: base.clone(),
                limits: limits(),
            },
            openkind_backends::device::FamilyExecution::Onnx { device_id: None },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx execution must fail closed for the kev family"),
        };
        assert!(
            error.to_string().contains("execution backend unavailable"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "onnx-rocm")]
    {
        let error = match KevEngine::load_with_execution(
            KevEngineConfig {
                model_root: root.clone(),
                base_root: base.clone(),
                limits: limits(),
            },
            openkind_backends::device::FamilyExecution::OnnxRocm { device_id: 0 },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx-rocm execution must fail closed for the kev family"),
        };
        assert!(
            error.to_string().contains("execution backend unavailable"),
            "unexpected error: {error}"
        );
    }

    #[cfg(not(any(feature = "onnx", feature = "onnx-rocm")))]
    let _ = (root, base);
}

#[test]
fn load_fails_closed_on_artifact_digest_mismatch() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_KEV_MODEL_ROOT is not set");
        return;
    };
    let Some(base) = base_root() else {
        eprintln!("skipping: OPENKIND_KEV_BASE_ROOT is not set");
        return;
    };
    let staged_model =
        std::env::temp_dir().join(format!("openkind-kev-model-{}", std::process::id()));
    let staged_base =
        std::env::temp_dir().join(format!("openkind-kev-base-{}", std::process::id()));
    let _ = fs::remove_dir_all(&staged_model);
    let _ = fs::remove_dir_all(&staged_base);
    fs::create_dir_all(&staged_model).expect("stage model root");
    fs::create_dir_all(&staged_base).expect("stage base root");
    // Stage every cheap artifact verbatim; the base checkpoint is replaced
    // with a placeholder because the drifted tokenizer fails before the
    // base checkpoint digest ever streams.
    for file in [
        "adapter_model.safetensors",
        "tokenizer.json",
        "head.safetensors",
    ] {
        fs::copy(root.join(file), staged_model.join(file)).expect("stage model artifact");
    }
    fs::copy(base.join("config.json"), staged_base.join("config.json")).expect("stage config");
    fs::write(staged_base.join("model.safetensors"), b"placeholder").expect("stage checkpoint");

    let mut tokenizer = fs::read(staged_model.join("tokenizer.json")).expect("read tokenizer");
    tokenizer.push(b' ');
    fs::write(staged_model.join("tokenizer.json"), tokenizer).expect("write drifted tokenizer");

    let error = match KevEngine::load(KevEngineConfig {
        model_root: staged_model.clone(),
        base_root: staged_base.clone(),
        limits: limits(),
    }) {
        Err(error) => error,
        Ok(_) => panic!("drifted tokenizer must fail closed"),
    };
    assert!(
        error.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {error}"
    );
    let _ = fs::remove_dir_all(&staged_model);
    let _ = fs::remove_dir_all(&staged_base);
}
