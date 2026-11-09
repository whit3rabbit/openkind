//! Offline parity and failure-mode tests for the pinned `strands-decider-2b`
//! profile.
//!
//! Unit-level contract checks always run. The full golden replay runs only
//! when the pinned artifacts are present locally (point
//! `OPENKIND_STRANDS_DECIDER_MODEL_ROOT` and
//! `OPENKIND_STRANDS_DECIDER_BASE_ROOT` at the roots); tests never download
//! model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::strands_decider::{
    StrandsDeciderEngine, StrandsDeciderEngineConfig, BACKBONE_REVISION, BASE_MODEL_ID,
    BASE_MODEL_REVISION, HOBSON_V21, PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/strands_decider_6a02bb0d1c6b25cae74b";
const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_STRANDS_DECIDER_MODEL_ROOT")
        .map(PathBuf::from)
        .filter(|root| root.is_dir())
}

fn base_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_STRANDS_DECIDER_BASE_ROOT")
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
            "strands-decider",
            "StrandsAgents/strands-decider-2B-hobson-v19",
            "bb282d786bc251fd4e3068de3ada9ddbb38127cd",
        )
    );
    assert_eq!(
        HOBSON_V21.profile_id,
        openkind_backends::families::support::derive_profile_id(
            "strands-decider",
            HOBSON_V21.backbone_id,
            HOBSON_V21.backbone_revision,
        )
    );
    assert_eq!(BASE_MODEL_ID, "Qwen/Qwen3.5-2B-Base");
    assert_eq!(BACKBONE_REVISION.len(), 40);
    assert_eq!(HOBSON_V21.backbone_revision.len(), 40);
    assert_eq!(BASE_MODEL_REVISION.len(), 40);
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!(
        "openkind-strands-decider-missing-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match StrandsDeciderEngine::load(StrandsDeciderEngineConfig {
        model_root: empty.clone(),
        base_root: empty.clone(),
        limits: limits(),
        profile: None,
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
        eprintln!("skipping: OPENKIND_STRANDS_DECIDER_MODEL_ROOT is not set");
        return;
    };
    let Some(base) = base_root() else {
        eprintln!("skipping: OPENKIND_STRANDS_DECIDER_BASE_ROOT is not set");
        return;
    };
    let golden: Golden = serde_json::from_slice(
        &fs::read(fixture_dir().join("golden.json")).expect("read golden fixture"),
    )
    .expect("decode golden fixture");
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let engine = StrandsDeciderEngine::load(StrandsDeciderEngineConfig {
        model_root: root,
        base_root: base,
        limits: limits(),
        profile: None,
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

/// Accelerated ONNX selections fail closed on a family whose pointer-head
/// readout has no ONNX export; checkpoint-gated because artifact
/// verification runs before the execution selection.
#[test]
fn accelerated_executions_fail_closed_without_an_export() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_STRANDS_DECIDER_MODEL_ROOT is not set");
        return;
    };
    let Some(base) = base_root() else {
        eprintln!("skipping: OPENKIND_STRANDS_DECIDER_BASE_ROOT is not set");
        return;
    };

    #[cfg(feature = "onnx")]
    {
        let error = match StrandsDeciderEngine::load_with_execution(
            StrandsDeciderEngineConfig {
                profile: None,
                model_root: root.clone(),
                base_root: base.clone(),
                limits: limits(),
            },
            openkind_backends::device::FamilyExecution::Onnx { device_id: None },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx execution must fail closed for the strands-decider family"),
        };
        assert!(
            error.to_string().contains("execution backend unavailable"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "onnx-rocm")]
    {
        let error = match StrandsDeciderEngine::load_with_execution(
            StrandsDeciderEngineConfig {
                profile: None,
                model_root: root.clone(),
                base_root: base.clone(),
                limits: limits(),
            },
            openkind_backends::device::FamilyExecution::OnnxRocm { device_id: 0 },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx-rocm execution must fail closed for the strands-decider family"),
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
        eprintln!("skipping: OPENKIND_STRANDS_DECIDER_MODEL_ROOT is not set");
        return;
    };
    let Some(base) = base_root() else {
        eprintln!("skipping: OPENKIND_STRANDS_DECIDER_BASE_ROOT is not set");
        return;
    };
    let staged_model =
        std::env::temp_dir().join(format!("openkind-strands-model-{}", std::process::id()));
    let staged_base =
        std::env::temp_dir().join(format!("openkind-strands-base-{}", std::process::id()));
    let _ = fs::remove_dir_all(&staged_model);
    let _ = fs::remove_dir_all(&staged_base);
    fs::create_dir_all(&staged_model).expect("stage model root");
    fs::create_dir_all(&staged_base).expect("stage base root");
    // Stage every cheap artifact verbatim; the base checkpoint is replaced
    // with a placeholder because the drifted config fails before the base
    // checkpoint digest ever streams.
    for file in [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.json",
        "head.safetensors",
        "hobson_config.json",
    ] {
        fs::copy(root.join(file), staged_model.join(file)).expect("stage model artifact");
    }
    fs::copy(base.join("config.json"), staged_base.join("config.json")).expect("stage config");
    fs::write(staged_base.join("model.safetensors"), b"placeholder").expect("stage checkpoint");

    let mut config = fs::read(staged_model.join("hobson_config.json")).expect("read config");
    config.push(b' ');
    fs::write(staged_model.join("hobson_config.json"), config).expect("write drifted config");

    let error = match StrandsDeciderEngine::load(StrandsDeciderEngineConfig {
        model_root: staged_model.clone(),
        base_root: staged_base.clone(),
        limits: limits(),
        profile: None,
    }) {
        Err(error) => error,
        Ok(_) => panic!("drifted hobson config must fail closed"),
    };
    assert!(
        error.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {error}"
    );
    let _ = fs::remove_dir_all(&staged_model);
    let _ = fs::remove_dir_all(&staged_base);
}

/// MLX/Metal parity gates for the pinned profiles (feature `mlx`, macOS
/// arm64). The candle CPU golden replay is the correctness oracle: the MLX
/// engine must reproduce the same committed fixture answers with unchanged
/// selections and bounded probability drift, per the workspace MLX parity
/// gates.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx_replay {
    use super::*;

    use openkind_backends::families::strands_decider::{StrandsMlxEngine, StrandsMlxEngineConfig};

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
            eprintln!("skipping: OPENKIND_STRANDS_DECIDER_MODEL_ROOT is not set");
            return;
        };
        let Some(base) = base_root() else {
            eprintln!("skipping: OPENKIND_STRANDS_DECIDER_BASE_ROOT is not set");
            return;
        };
        let golden: Golden = serde_json::from_slice(
            &fs::read(fixture_dir().join("golden.json")).expect("read golden fixture"),
        )
        .expect("decode golden fixture");
        assert!(!golden.cases.is_empty(), "golden fixture has cases");

        let engine = StrandsMlxEngine::load(StrandsMlxEngineConfig {
            model_root: root,
            base_root: base,
            limits: limits(),
            profile: None,
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
            "mlx strands-decider-2b parity: {answers} answers, max |Δp| = {:.3e}, selection \
             flips = {}",
            parity.max_probability_error, parity.selection_flips
        );
    }
}
