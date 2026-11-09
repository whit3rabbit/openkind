//! Offline parity and failure-mode tests for the pinned von profile.
//!
//! Unit-level contract checks always run. The golden replays run only when a
//! pinned checkpoint is present locally (point `OPENKIND_VON_MODEL_ROOT` at
//! the model root); tests never download model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::support::FamilyLimits;
use openkind_backends::families::von::{VonEngine, VonEngineConfig, PROFILE_ID, VON};
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/von_69219703407bd39cca0c")
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_VON_MODEL_ROOT")
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
    profile_id: String,
    execution_arithmetic: String,
    probability_tolerance: f64,
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
fn golden_fixture_identity_matches_the_compiled_profile() {
    let golden: Golden =
        serde_json::from_slice(&fs::read(fixture_dir().join("golden.json")).expect("read fixture"))
            .expect("decode fixture");
    assert_eq!(golden.profile_id, PROFILE_ID);
    assert_eq!(golden.profile_id, VON.profile_id);
    assert_eq!(golden.execution_arithmetic, "candle-cpu-fp32-von");
    assert!(golden.probability_tolerance <= PROBABILITY_TOLERANCE);
    assert!(!golden.cases.is_empty());
}

#[test]
fn golden_replay_matches_the_reference() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_VON_MODEL_ROOT is not set");
        return;
    };
    let golden: Golden =
        serde_json::from_slice(&fs::read(fixture_dir().join("golden.json")).expect("read fixture"))
            .expect("decode fixture");
    let engine = VonEngine::load(VonEngineConfig {
        profile: &VON,
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
            assert_answer_matches(&format!("{}:{question_id}", case.name), &actual, expected);
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

#[test]
fn rejected_requests_fail_closed() {
    // Loading without the pinned checkpoint artifacts must fail; requests
    // that never touch the checkpoint therefore fail closed too.
    let bogus = VonEngine::load(VonEngineConfig {
        profile: &VON,
        model_root: PathBuf::from("/nonexistent/von/root"),
        limits: limits(),
    });
    assert!(bogus.is_err(), "missing artifacts must fail the load");
    let _ = fixture_dir;
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!("openkind-von-missing-{}", std::process::id()));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match VonEngine::load(VonEngineConfig {
        profile: &VON,
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
fn load_fails_closed_on_config_digest_mismatch() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_VON_MODEL_ROOT is not set");
        return;
    };
    let staged = std::env::temp_dir().join(format!("openkind-von-config-{}", std::process::id()));
    let _ = fs::remove_dir_all(&staged);
    fs::create_dir_all(staged.join("checkpoint")).expect("stage root");
    // Copy only the cheap artifacts; the checkpoint is deliberately absent
    // so a drifted config fails before any large file is streamed.
    for file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "marker_calibration.json",
    ] {
        fs::copy(
            root.join("checkpoint").join(file),
            staged.join("checkpoint").join(file),
        )
        .expect("stage artifact");
    }
    let mut config: serde_json::Value = serde_json::from_slice(
        &fs::read(staged.join("checkpoint/config.json")).expect("read config"),
    )
    .expect("decode config");
    config["hidden_size"] = serde_json::json!(127);
    fs::write(staged.join("checkpoint/config.json"), config.to_string())
        .expect("write drifted config");

    let error = match VonEngine::load(VonEngineConfig {
        profile: &VON,
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

/// MLX/Metal parity gates for the pinned profile (feature `mlx`, macOS
/// arm64). The candle CPU golden replay is the correctness oracle: the MLX
/// engine must reproduce the same committed fixture answers with unchanged
/// selections and bounded probability drift, per the workspace MLX parity
/// gates.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_replay {
    use super::*;

    /// Absolute drift budget on calibrated probabilities (the workspace MLX
    /// gate for FP32 backend parity).
    const MLX_PROBABILITY_TOLERANCE: f64 = 0.005;

    struct MlxParity {
        max_probability_error: f64,
        selection_flips: usize,
    }

    fn measure_answer(
        label: &str,
        actual: &serde_json::Value,
        expected: &serde_json::Value,
        parity: &mut MlxParity,
    ) {
        if actual["type"] != expected["type"] {
            panic!("{label}: answer type drifted on the MLX backend");
        }
        match expected["type"].as_str().expect("tagged answer") {
            "noul" => {
                let actual = actual["noul"].as_f64().expect("noul");
                let expected = expected["noul"].as_f64().expect("noul");
                parity.max_probability_error =
                    parity.max_probability_error.max((actual - expected).abs());
            }
            "choice" => {
                let actual_selection = actual["choice"].as_str().expect("choice selection");
                let expected_selection = expected["choice"].as_str().expect("choice selection");
                if actual_selection != expected_selection {
                    parity.selection_flips += 1;
                    eprintln!(
                        "{label}: MLX selection flip {actual_selection} vs {expected_selection}"
                    );
                }
                for (key, expected_probability) in expected["probabilities"]
                    .as_object()
                    .expect("probabilities")
                {
                    let actual_probability = actual["probabilities"][key]
                        .as_f64()
                        .unwrap_or_else(|| panic!("{label}: missing MLX probability {key}"));
                    parity.max_probability_error = parity
                        .max_probability_error
                        .max((actual_probability - expected_probability.as_f64().unwrap()).abs());
                }
            }
            "score" => {
                let actual_score = actual["score"].as_f64().expect("score");
                let expected_score = expected["score"].as_f64().expect("score");
                parity.max_probability_error = parity
                    .max_probability_error
                    .max((actual_score - expected_score).abs());
                for (key, expected_probability) in expected["probabilities"]
                    .as_object()
                    .expect("probabilities")
                {
                    let actual_probability = actual["probabilities"][key]
                        .as_f64()
                        .unwrap_or_else(|| panic!("{label}: missing MLX level {key}"));
                    parity.max_probability_error = parity
                        .max_probability_error
                        .max((actual_probability - expected_probability.as_f64().unwrap()).abs());
                }
            }
            other => panic!("{label}: unexpected answer type {other}"),
        }
    }

    #[test]
    fn mlx_golden_replay_matches_the_pinned_checkpoint() {
        let Some(root) = model_root() else {
            eprintln!("skipping: OPENKIND_VON_MODEL_ROOT is not set");
            return;
        };
        let golden: Golden = serde_json::from_slice(
            &fs::read(fixture_dir().join("golden.json")).expect("read fixture"),
        )
        .expect("decode fixture");
        assert!(!golden.cases.is_empty(), "golden fixture has cases");

        let engine = VonEngine::load_mlx(VonEngineConfig {
            profile: &VON,
            model_root: root,
            limits: limits(),
        })
        .expect("load pinned MLX engine");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");

        let mut parity = MlxParity {
            max_probability_error: 0.0,
            selection_flips: 0,
        };
        let mut answers = 0_usize;
        for case in &golden.cases {
            let response = runtime
                .block_on(engine.evaluate(case.request.clone()))
                .unwrap_or_else(|error| panic!("mlx case {}: {error}", case.name));
            for (question_id, expected) in &case.response.answers {
                let actual = response.answers.get(question_id).unwrap_or_else(|| {
                    panic!("mlx case {}: missing answer {question_id}", case.name)
                });
                let actual = serde_json::to_value(actual).expect("serialize answer");
                measure_answer(
                    &format!("mlx {}: {question_id}", case.name),
                    &actual,
                    expected,
                    &mut parity,
                );
                answers += 1;
            }
            assert_eq!(
                response.usage.input_tokens, case.response.usage.input_tokens,
                "mlx {}: usage drifted",
                case.name
            );
        }
        assert_eq!(
            parity.selection_flips, 0,
            "MLX backend changed {} selections",
            parity.selection_flips
        );
        assert!(
            parity.max_probability_error <= MLX_PROBABILITY_TOLERANCE,
            "MLX probability drift {} exceeds {MLX_PROBABILITY_TOLERANCE}",
            parity.max_probability_error
        );
        eprintln!(
            "mlx von parity: {answers} answers, max |Δp| = {:.3e}, selection flips = {}",
            parity.max_probability_error, parity.selection_flips
        );
    }
}
