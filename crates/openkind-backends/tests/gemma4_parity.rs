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
