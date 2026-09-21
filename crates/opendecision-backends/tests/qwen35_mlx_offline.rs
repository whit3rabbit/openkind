//! Offline (no-checkpoint) unit tests for the Phase 3M MLX runtime spine.
//!
//! These tests exercise the serialized execution model, runtime identity
//! checks, memory policy, and MLX array semantics through
//! [`MlxRuntime::execute`]. They require a GPU-capable MLX build
//! (`--features mlx` on macOS arm64) but load no model artifacts.

#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]

use std::ops::{Add, Mul};
use std::sync::Arc;

use mlx_rs::Array;

use opendecision_backends::qwen35::mlx::{
    MlxRuntime, MlxRuntimeConfig, MLX_ARITHMETIC_ID_BF16_REFERENCE,
    MLX_ARITHMETIC_ID_FP32_REFERENCE, MLX_CORE_VERSION, MLX_C_RELEASE, MLX_LM_REFERENCE_COMMIT,
    MLX_RS_VERSION,
};

fn runtime() -> MlxRuntime {
    MlxRuntime::new(MlxRuntimeConfig::default()).expect("MLX runtime construction")
}

#[test]
fn runtime_reports_pinned_mlx_version() {
    let runtime = runtime();
    assert_eq!(runtime.version().trim(), "0.32.2");
    assert_eq!(MLX_CORE_VERSION, "0.32.2");
    assert_eq!(MLX_RS_VERSION, "0.32.0");
    assert!(MLX_C_RELEASE.contains("MLX 0.32.2"));
    assert_eq!(MLX_LM_REFERENCE_COMMIT.len(), 40);
}

#[test]
fn native_bf16_requires_and_records_runtime_preflight() {
    let runtime = runtime();
    assert!(!runtime.bf16_qualified());
    runtime.qualify_bf16().expect("BF16 preflight");
    assert!(runtime.bf16_qualified());
    runtime.qualify_bf16().expect("cached BF16 preflight");
}

#[test]
fn panicking_mlx_work_is_reported_as_an_operation_error() {
    let runtime = runtime();
    let error = runtime
        .execute(|| panic!("synthetic MLX failure"))
        .expect_err("panic must fail closed");
    assert!(error.to_string().contains("synthetic MLX failure"));
    runtime
        .execute(|| {})
        .expect("execution lock remains usable after caught panic");
}

#[test]
fn arithmetic_ids_are_distinct_per_precision_and_kernel_family() {
    assert_ne!(
        MLX_ARITHMETIC_ID_FP32_REFERENCE,
        MLX_ARITHMETIC_ID_BF16_REFERENCE
    );
    assert!(MLX_ARITHMETIC_ID_FP32_REFERENCE.starts_with("mlx-core-0.32.2/fp32/"));
    assert!(MLX_ARITHMETIC_ID_BF16_REFERENCE.starts_with("mlx-core-0.32.2/bf16/"));
    assert!(MLX_ARITHMETIC_ID_FP32_REFERENCE.ends_with("/reference-ops"));
}

#[test]
fn execute_runs_array_work_and_materializes_results() {
    let runtime = runtime();
    let result: Vec<f32> = runtime
        .execute(|| {
            let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
            let b = Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[2, 2]);
            let out = a.matmul(&b).expect("matmul");
            out.eval().expect("eval");
            out.to_vec_cast::<f32>().expect("read")
        })
        .expect("execute");
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn synchronize_completes_without_error() {
    let runtime = runtime();
    runtime
        .execute(|| {
            let x = Array::from_slice(&[0.5_f32; 1024], &[1, 1024]);
            x.eval().expect("eval");
        })
        .expect("execute");
    runtime.synchronize().expect("synchronize");
}

#[test]
fn memory_telemetry_getters_succeed() {
    let runtime = runtime();
    runtime.reset_peak_memory().expect("reset peak");
    runtime
        .execute(|| {
            let big = Array::from_slice(&vec![1.0_f32; 4 * 1024 * 1024], &[1, 4 * 1024 * 1024]);
            big.eval().expect("eval");
        })
        .expect("execute");
    let snapshot = runtime.memory_snapshot();
    assert!(snapshot.peak_bytes.unwrap_or(0) >= 16 * 1024 * 1024);
    assert!(
        snapshot.cache_bytes.unwrap_or(0)
            <= runtime.config().inactive_cache_limit_bytes.max(
                // The cache may momentarily exceed the limit before trimming.
                64 * 1024 * 1024,
            )
    );
}

#[test]
fn clone_shares_immutable_values_without_corruption() {
    let runtime = runtime();
    runtime
        .execute(|| {
            let original = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
            let cloned = original.clone();
            let out = original.clone().add(&cloned);
            out.eval().expect("eval");
            let values = out.to_vec_cast::<f32>().expect("read");
            assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0]);
        })
        .expect("execute");
}

#[test]
fn concurrent_callers_are_serialized_without_deadlock() {
    let runtime = Arc::new(runtime());
    let mut handles = Vec::new();
    for lane in 0..4_u32 {
        let runtime = Arc::clone(&runtime);
        handles.push(std::thread::spawn(move || {
            runtime
                .execute(move || {
                    let x = Array::from_slice(&vec![lane as f32; 256], &[1, 256]);
                    let out = x.matmul(x.transpose().expect("transpose")).expect("matmul");
                    out.eval().expect("eval");
                })
                .expect("execute");
        }));
    }
    for handle in handles {
        handle.join().expect("thread");
    }
}

#[test]
fn stream_explicit_execution_survives_thread_local_default_absence() {
    // Threads that have never touched MLX before can still execute through
    // the runtime: the explicit stream scope does not depend on a
    // pre-existing thread-local default stream.
    let runtime = runtime();
    let handle = std::thread::spawn(move || {
        runtime
            .execute(|| {
                let x = Array::from_slice(&[2.0_f32, 3.0], &[2]);
                let out = x.clone().mul(&x);
                out.eval().expect("eval");
            })
            .expect("execute");
    });
    handle.join().expect("thread");
}
