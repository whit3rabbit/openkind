//! Offline failure-mode tests for the pinned `gemma4-decision`
//! (`winnow-e4b`) profile.
//!
//! Unit-level contract checks always run. Checkpoint-dependent checks run
//! only when the pinned GGUF is present locally (point
//! `OPENKIND_GEMMA4_MODEL_ROOT` at the model root); tests never download
//! model artifacts.

use std::fs;
use std::path::PathBuf;

use openkind_backends::families::gemma4::{
    Gemma4DecisionEngine, Gemma4EngineConfig, BACKBONE_ID, PROFILE_ID,
};
use openkind_backends::families::support::{derive_profile_id, FamilyLimits};

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_GEMMA4_MODEL_ROOT")
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

#[test]
fn profile_constants_are_self_consistent() {
    assert_eq!(
        PROFILE_ID,
        derive_profile_id(
            "gemma4-decision",
            BACKBONE_ID,
            "1b257e8fa80b270a62338362a8b35e37f7890273"
        )
    );
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty =
        std::env::temp_dir().join(format!("openkind-gemma4-missing-{}", std::process::id()));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match Gemma4DecisionEngine::load(Gemma4EngineConfig {
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

/// Accelerated selections must fail closed on a family with no ONNX export,
/// after (not instead of) artifact verification, matching the sibling
/// fail-closed families' error kind.
#[test]
fn onnx_executions_fail_closed_without_an_export() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_GEMMA4_MODEL_ROOT is not set");
        return;
    };
    let config = Gemma4EngineConfig {
        model_root: root,
        limits: limits(),
    };

    #[cfg(feature = "onnx")]
    {
        let error = match Gemma4DecisionEngine::load_with_execution(
            config.clone(),
            openkind_backends::device::FamilyExecution::Onnx { device_id: None },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx execution must fail closed for gemma4-decision"),
        };
        assert!(
            matches!(
                error,
                openkind_backends::families::gemma4::Gemma4Error::Family(
                    openkind_backends::families::support::FamilyError::ExecutionUnavailable(_)
                )
            ),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "onnx-rocm")]
    {
        let error = match Gemma4DecisionEngine::load_with_execution(
            config,
            openkind_backends::device::FamilyExecution::OnnxRocm { device_id: 0 },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx-rocm execution must fail closed for gemma4-decision"),
        };
        assert!(
            matches!(
                error,
                openkind_backends::families::gemma4::Gemma4Error::Family(
                    openkind_backends::families::support::FamilyError::ExecutionUnavailable(_)
                )
            ),
            "unexpected error: {error}"
        );
    }

    #[cfg(not(any(feature = "onnx", feature = "onnx-rocm")))]
    {
        // Without the ONNX features the accelerated variants cannot be
        // constructed; assert the CPU selection still resolves a device.
        let _ = config;
        assert!(!openkind_backends::device::execution_support().onnx);
    }
}

/// MLX/Metal parity gates for the pinned profile (feature `mlx`, macOS
/// arm64). The profile has no committed golden fixture: the candle CPU q8_0
/// engine loaded from the same local checkpoint is the correctness oracle,
/// and the MLX engine must reproduce its answers with unchanged selections
/// and bounded probability drift (the workspace MLX parity budget). The
/// MLX execution is fully FP32-dequantized — a declared arithmetic
/// difference from the CPU oracle's f16-block-scaled quantized kernels,
/// named by the MLX backend id — so exact equality is not expected.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_replay {
    use super::*;

    use openkind_backends::families::gemma4::Gemma4MlxEngine;
    use openkind_core::SystemRequest;
    use openkind_engine::DecisionEngine;

    /// Absolute drift budget on calibrated probabilities (the workspace MLX
    /// gate for FP32-dequantized backend parity).
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

    /// Ground-truth decision requests of the kind the profile's operator
    /// smoke run exercises: obvious argmax cases across all three wire
    /// primitives over the same structured record.
    fn replay_requests() -> Vec<(&'static str, serde_json::Value)> {
        let state = serde_json::json!({
            "record": {
                "order_id": "A-2231",
                "destination": "Reykjavik, Iceland",
                "carrier": "Northern Air",
                "departures": ["09:30", "14:05"],
                "cancelled_departures": ["09:30"],
                "active_departure": "14:05",
            }
        });
        vec![
            (
                "noul_true",
                serde_json::json!({
                    "model": "winnow-e4b",
                    "state": state,
                    "questions": {
                        "q0": {
                            "type": "noul",
                            "instructions": "Does the record still list an active departure?"
                        }
                    }
                }),
            ),
            (
                "choice_destination",
                serde_json::json!({
                    "model": "winnow-e4b",
                    "state": state,
                    "questions": {
                        "q0": {
                            "type": "choice",
                            "instructions": "Which city is the order shipping to?",
                            "criteria": {
                                "lisbon": "The order ships to Lisbon, Portugal.",
                                "reykjavik": "The order ships to Reykjavik, Iceland."
                            }
                        }
                    }
                }),
            ),
            (
                "score_cancelled_count",
                serde_json::json!({
                    "model": "winnow-e4b",
                    "state": state,
                    "questions": {
                        "q0": {
                            "type": "score",
                            "instructions": "How many departures were cancelled?",
                            "criteria": [
                                "no departure was cancelled",
                                "exactly one departure was cancelled",
                                "two or more departures were cancelled"
                            ]
                        }
                    }
                }),
            ),
        ]
    }

    #[test]
    fn mlx_replay_matches_the_cpu_oracle() {
        let Some(root) = model_root() else {
            eprintln!("skipping: OPENKIND_GEMMA4_MODEL_ROOT is not set");
            return;
        };
        let cpu = Gemma4DecisionEngine::load(Gemma4EngineConfig {
            model_root: root.clone(),
            limits: limits(),
        })
        .expect("load the CPU oracle engine");
        let mlx = Gemma4MlxEngine::load_mlx(Gemma4EngineConfig {
            model_root: root,
            limits: limits(),
        })
        .expect("load the MLX engine");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");

        let mut parity = MlxParity {
            max_probability_error: 0.0,
            selection_flips: 0,
        };
        let mut answers = 0_usize;
        for (name, request_value) in replay_requests() {
            let request: SystemRequest =
                serde_json::from_value(request_value).expect("decode request");
            let expected = runtime
                .block_on(cpu.evaluate(request.clone()))
                .unwrap_or_else(|error| panic!("cpu {name}: {error}"));
            let actual = runtime
                .block_on(mlx.evaluate(request))
                .unwrap_or_else(|error| panic!("mlx {name}: {error}"));
            for (question_id, expected_answer) in &expected.answers {
                let actual_answer = actual
                    .answers
                    .get(question_id)
                    .unwrap_or_else(|| panic!("mlx {name}: missing answer {question_id}"));
                let actual = serde_json::to_value(actual_answer).expect("serialize MLX answer");
                let expected = serde_json::to_value(expected_answer).expect("serialize CPU answer");
                measure_answer(
                    &format!("mlx {name}: {question_id}"),
                    &actual,
                    &expected,
                    &mut parity,
                );
                answers += 1;
            }
            assert_eq!(
                actual.usage.input_tokens, expected.usage.input_tokens,
                "mlx {name}: usage drifted"
            );
        }
        assert!(answers > 0, "the replay must exercise at least one answer");
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
            "mlx gemma4-decision parity: {answers} answers, max |Δp| = {:.3e}, selection flips = {}",
            parity.max_probability_error, parity.selection_flips
        );
    }
}
