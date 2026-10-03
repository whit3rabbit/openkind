//! Offline parity tests for the pinned winnow profile.
//!
//! The adapter is vendored in-repo; the base checkpoint is env-gated. The
//! golden replay asserts the routing distribution over pinned states within
//! the frozen probability tolerance, plus behavioral routing through the
//! composite engine.

use std::path::{Path, PathBuf};

use openkind_backends::families::support::FamilyLimits;
use openkind_backends::families::winnow::{WinnowEngine, WinnowEngineConfig};
use openkind_engine::{DecisionEngine, MockEngine};
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/winnow_4dff8c5b03cfbf680db6";
const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_WINNOW_MODEL_ROOT")
        .map(PathBuf::from)
        .filter(|root| root.is_dir())
}

fn adapter_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/winnow_adapter/adapters.safetensors")
}

fn config(model_root: PathBuf) -> WinnowEngineConfig {
    WinnowEngineConfig {
        model_root,
        adapter_path: adapter_path(),
        limits: FamilyLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 0,
            retry_after_ms: 100,
            evaluation_timeout: None,
        },
    }
}

fn siblings() -> Vec<(String, std::sync::Arc<dyn DecisionEngine>)> {
    vec![
        (
            "english-sibling".to_owned(),
            std::sync::Arc::new(MockEngine::new()),
        ),
        (
            "multilingual-sibling".to_owned(),
            std::sync::Arc::new(MockEngine::new()),
        ),
    ]
}

#[derive(Debug, Deserialize)]
struct Golden {
    cases: Vec<GoldenCase>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    name: String,
    expected_label: String,
    state: String,
    probabilities: Vec<f64>,
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty =
        std::env::temp_dir().join(format!("openkind-winnow-missing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&empty);
    std::fs::create_dir_all(&empty).expect("create empty root");
    let error = match WinnowEngine::load(config(empty.clone()), siblings()) {
        Err(error) => error,
        Ok(_) => panic!("missing artifacts must fail closed"),
    };
    assert!(
        error.to_string().contains("missing") || error.to_string().contains("read"),
        "unexpected error: {error}"
    );
    let _ = std::fs::remove_dir_all(&empty);
}

#[test]
fn load_fails_closed_on_adapter_digest_mismatch() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_WINNOW_MODEL_ROOT is not set");
        return;
    };
    let staged_adapter = std::env::temp_dir().join(format!(
        "openkind-winnow-adapter-{}.safetensors",
        std::process::id()
    ));
    std::fs::copy(adapter_path(), &staged_adapter).expect("stage adapter");
    let mut bytes = std::fs::read(&staged_adapter).expect("read adapter");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&staged_adapter, bytes).expect("write drifted adapter");

    let error = match WinnowEngine::load(
        WinnowEngineConfig {
            model_root: root,
            adapter_path: staged_adapter.clone(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 100,
                evaluation_timeout: None,
            },
        },
        siblings(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("drifted adapter must fail closed"),
    };
    assert!(
        error.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {error}"
    );
    let _ = std::fs::remove_dir_all(&staged_adapter);
}

#[tokio::test]
async fn golden_routing_matches_the_pinned_adapter() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_WINNOW_MODEL_ROOT is not set");
        return;
    };
    let golden: Golden = serde_json::from_slice(
        &std::fs::read(fixture_dir().join("golden.json")).expect("read golden fixture"),
    )
    .expect("decode golden fixture");
    let engine = WinnowEngine::load(config(root), siblings()).expect("load pinned engine");

    for case in &golden.cases {
        let probabilities = engine
            .debug_route_probabilities(&case.state)
            .unwrap_or_else(|error| panic!("case {}: {error}", case.name));
        for (index, expected) in case.probabilities.iter().enumerate() {
            assert!(
                (probabilities[index] - expected).abs() <= PROBABILITY_TOLERANCE,
                "{}: label {index} {} vs {}",
                case.name,
                probabilities[index],
                expected
            );
        }
        let routed = if probabilities[0] >= probabilities[1] {
            "A"
        } else {
            "B"
        };
        let expected = match case.expected_label.as_str() {
            "english" => "A",
            _ => "B",
        };
        assert_eq!(routed, expected, "{}: routed label drifted", case.name);
    }
}

#[tokio::test]
async fn composite_engine_delegates_to_the_routed_sibling() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_WINNOW_MODEL_ROOT is not set");
        return;
    };
    let engine = WinnowEngine::load(config(root), siblings()).expect("load pinned engine");
    let request: openkind_core::SystemRequest = serde_json::from_value(serde_json::json!({
        "state": "Сводка инцидента: затронутый актив — сервер приложений.",
        "model": "winnow-router",
        "questions": {
            "q0": { "type": "noul", "instructions": "Does the record state the fact?" }
        }
    }))
    .expect("request");
    let response = engine.evaluate(request).await.expect("evaluate");
    assert!(
        response.answers.contains_key("q0"),
        "routed sibling answered"
    );
}

/// Accelerated ONNX selections fail closed on a family whose LoRA-merged
/// decoder has no ONNX export; checkpoint-gated because artifact
/// verification runs before the execution selection.
#[test]
fn accelerated_executions_fail_closed_without_an_export() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_WINNOW_MODEL_ROOT is not set");
        return;
    };

    #[cfg(feature = "onnx")]
    {
        let error = match WinnowEngine::load_with_execution(
            config(root.clone()),
            siblings(),
            openkind_backends::device::FamilyExecution::Onnx { device_id: None },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx execution must fail closed for the winnow family"),
        };
        assert!(
            error.to_string().contains("execution backend unavailable"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "onnx-rocm")]
    {
        let error = match WinnowEngine::load_with_execution(
            config(root.clone()),
            siblings(),
            openkind_backends::device::FamilyExecution::OnnxRocm { device_id: 0 },
        ) {
            Err(error) => error,
            Ok(_) => panic!("onnx-rocm execution must fail closed for the winnow family"),
        };
        assert!(
            error.to_string().contains("execution backend unavailable"),
            "unexpected error: {error}"
        );
    }

    #[cfg(not(any(feature = "onnx", feature = "onnx-rocm")))]
    let _ = root;
}

/// The router binds exactly two siblings; the check runs before artifact
/// verification so it is testable offline.
#[test]
fn load_rejects_sibling_counts_other_than_two_before_verification() {
    let empty =
        std::env::temp_dir().join(format!("openkind-winnow-siblings-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&empty);
    std::fs::create_dir_all(&empty).expect("create empty root");

    for count in [1_usize, 3] {
        let mut group: Vec<(String, std::sync::Arc<dyn DecisionEngine>)> = siblings();
        while group.len() < count {
            group.push((
                format!("extra-sibling-{count}"),
                std::sync::Arc::new(MockEngine::new()),
            ));
        }
        group.truncate(count);

        let error = match WinnowEngine::load(config(empty.clone()), group) {
            Err(error) => error,
            Ok(_) => panic!("{count} siblings must fail closed"),
        };
        assert!(
            error.to_string().contains("exactly two labels"),
            "unexpected error for {count} siblings: {error}"
        );
    }
    let _ = std::fs::remove_dir_all(&empty);
}
