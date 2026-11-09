//! Offline parity and failure-mode tests for the pinned `drex-v15` profile (Drex v1.5).
//!
//! Unit-level contract checks always run. The golden replay runs only when
//! the pinned checkpoint is present locally (point
//! `OPENKIND_DREX_V15_MODEL_ROOT` at the model root); tests never download
//! model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::drex::{
    DrexEngine, DrexEngineConfig, BACKBONE_ID, BACKBONE_REVISION, CALIBRATION_TEMPERATURE,
    EXECUTION_ARITHMETIC_ID, FAMILY_SLUG, PROFILE_ID, TEACHER_USE_PERMITTED,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/drex_2d495e07ea056d6c4317";
const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_DREX_V15_MODEL_ROOT")
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
    backbone: GoldenBackbone,
    execution_arithmetic: String,
    calibration_temperature: f64,
    cases: Vec<GoldenCase>,
}

#[derive(Debug, Deserialize)]
struct GoldenBackbone {
    id: String,
    revision: String,
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
struct GoldenUsage {
    input_tokens: u64,
}

fn golden() -> Golden {
    serde_json::from_slice(&fs::read(fixture_dir().join("golden.json")).expect("read golden"))
        .expect("decode golden fixture")
}

/// Compare every probability of one answer, plus the selected option.
fn assert_answer_matches(label: &str, actual: &serde_json::Value, expected: &serde_json::Value) {
    assert_eq!(
        actual["type"], expected["type"],
        "{label}: answer type drifted"
    );
    let close = |field: &str, actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= PROBABILITY_TOLERANCE,
            "{label}: {field} {actual} vs {expected}"
        );
    };
    match expected["type"].as_str().expect("tagged answer") {
        "noul" => close(
            "noul",
            actual["noul"].as_f64().expect("noul"),
            expected["noul"].as_f64().expect("noul"),
        ),
        kind @ ("choice" | "score") => {
            if kind == "choice" {
                assert_eq!(
                    actual["choice"], expected["choice"],
                    "{label}: choice drifted"
                );
            } else {
                close(
                    "score",
                    actual["score"].as_f64().expect("score"),
                    expected["score"].as_f64().expect("score"),
                );
            }
            for (key, probability) in expected["probabilities"]
                .as_object()
                .expect("probabilities")
            {
                close(
                    key,
                    actual["probabilities"][key]
                        .as_f64()
                        .unwrap_or_else(|| panic!("{label}: missing probability {key}")),
                    probability.as_f64().expect("probability"),
                );
            }
        }
        other => panic!("{label}: unexpected answer type {other}"),
    }
}

#[test]
fn golden_metadata_matches_the_pinned_profile() {
    let golden = golden();
    assert_eq!(golden.profile_id, PROFILE_ID);
    assert_eq!(golden.backbone.id, BACKBONE_ID);
    assert_eq!(golden.backbone.revision, BACKBONE_REVISION);
    assert_eq!(golden.execution_arithmetic, EXECUTION_ARITHMETIC_ID);
    assert_eq!(golden.calibration_temperature, CALIBRATION_TEMPERATURE);
    assert!(!golden.cases.is_empty());
    assert_eq!(
        PROFILE_ID,
        openkind_backends::families::support::derive_profile_id(
            FAMILY_SLUG,
            BACKBONE_ID,
            BACKBONE_REVISION
        )
    );
}

#[test]
fn teacher_use_stays_refused() {
    // License policy: Drex output must never train another model.
    let permitted = TEACHER_USE_PERMITTED;
    assert!(!permitted);
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!("openkind-drex-missing-{}", std::process::id()));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match DrexEngine::load(DrexEngineConfig {
        model_root: empty.clone(),
        limits: limits(),
    }) {
        Err(error) => error,
        Ok(_) => panic!("missing artifacts must fail closed"),
    };
    let message = error.to_string();
    assert!(
        message.contains("config.json") || message.contains("missing") || message.contains("read"),
        "unexpected error: {message}"
    );
    let _ = fs::remove_dir_all(&empty);
}

#[test]
fn golden_replay_matches_the_pinned_checkpoint() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_DREX_V15_MODEL_ROOT is not set");
        return;
    };
    let golden = golden();
    let engine = DrexEngine::load(DrexEngineConfig {
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
        assert_eq!(
            response.answers.len(),
            case.response.answers.len(),
            "{}: answer count drifted",
            case.name
        );
        for (question_id, expected) in &case.response.answers {
            let actual = response
                .answers
                .get(question_id)
                .unwrap_or_else(|| panic!("case {}: missing answer {question_id}", case.name));
            let actual = serde_json::to_value(actual).expect("serialize answer");
            assert_answer_matches(&format!("{}/{question_id}", case.name), &actual, expected);
        }
        assert_eq!(
            response.usage.output_tokens, 0,
            "{}: no generation",
            case.name
        );
        assert_eq!(
            u64::from(response.usage.input_tokens),
            case.response.usage.input_tokens,
            "{}: usage drifted",
            case.name
        );
    }
}

/// MLX/Metal parity gates for the pinned profile (feature `mlx`, macOS
/// arm64). The candle CPU golden replay is the correctness oracle: the MLX
/// engine must reproduce the same committed fixture answers with unchanged
/// selections and bounded probability drift, per the workspace MLX parity
/// gates.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_replay {
    use super::*;

    use openkind_backends::families::drex::{DrexMlxEngine, DrexMlxEngineConfig};

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
            eprintln!("skipping: OPENKIND_DREX_V15_MODEL_ROOT is not set");
            return;
        };
        let golden = golden();
        assert!(!golden.cases.is_empty(), "golden fixture has cases");

        let engine = DrexMlxEngine::load(DrexMlxEngineConfig {
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
                u64::from(response.usage.input_tokens),
                case.response.usage.input_tokens,
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
            "mlx drex-v15 parity: {answers} answers, max |Δp| = {:.3e}, selection flips = {}",
            parity.max_probability_error, parity.selection_flips
        );
    }
}
