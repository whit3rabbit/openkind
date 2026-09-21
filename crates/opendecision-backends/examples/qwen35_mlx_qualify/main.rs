//! Phase 3M.0 — MLX/Metal runtime qualification gate.
//!
//! Proves the linked MLX runtime computes the primitive operations this
//! backend depends on *before* any 4B-parameter model work starts:
//!
//! - FP32: gather (`take_axis`), matmul, fast RMSNorm, RoPE, depthwise
//!   `conv1d` (native MLX conv with `groups`), cumsum, reshape/concatenate,
//!   stream synchronization, plus the mlx-c issue #115 combined pipeline.
//! - BF16: gather, fast RMSNorm, matmul, and the exact issue #115
//!   reproduction (embedding `[131072, 4096]` filled with 0.5, RMS weight
//!   all 1.0, `q_proj` all 0.25; the correct first matmul element is ~1024 —
//!   a runtime built with a broken toolchain yields 512).
//!
//! Gate semantics (frozen; no relaxation):
//!
//! - FP32 failure  => the runtime is invalid: exit code 1.
//! - BF16 failure  => the `mlx-native-bf16` candidate profile is BLOCKED for
//!   this runtime/toolchain: exit code 3 (FP32 spine is unaffected).
//! - All pass      => exit code 0.
//!
//! `--formal` additionally writes a runtime provenance JSON covering the
//! pinned crate/MLX versions, linked static-archive and `mlx.metallib`
//! SHA-256 hashes, Xcode/Metal/macOS toolchain identity, and the git
//! commit. Formal parity evidence must attach that file; the same mlx-c
//! sources have produced different BF16 results under different Xcode
//! builds (ml-explore/mlx-c#115), so toolchain identity is part of the
//! numerical function.
//!
//! This example requires `--features mlx` on macOS arm64 and the `SDKROOT`
//! environment set to the macOS SDK for bindgen. It is offline: no model
//! artifacts are loaded.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod bf16;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod fp32;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod gate;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod helpers;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::process::ExitCode;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("qualify: fatal error: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!(
        "qwen35_mlx_qualify requires --features mlx on macOS arm64 \
         (build with SDKROOT=$(xcrun --show-sdk-path))"
    );
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn run() -> Result<ExitCode, String> {
    let formal = std::env::args().any(|arg| arg == "--formal");
    if formal {
        gate::require_clean_worktree()?;
    }
    let report = gate::run(formal)?;
    println!();
    if !report.fp32_pass {
        println!(
            "RESULT: FP32 RUNTIME INVALID — {} failure(s): {:?}",
            report.failures.len(),
            report.failures
        );
        return Ok(ExitCode::from(1));
    }
    if !report.bf16_pass {
        println!(
            "RESULT: FP32 QUALIFIED, BF16 BLOCKED — the mlx-native-bf16 candidate \
             profile must not run model workloads on this runtime/toolchain \
             (ml-explore/mlx-c#115). Failures: {:?}",
            report.failures
        );
        return Ok(ExitCode::from(3));
    }
    println!(
        "RESULT: QUALIFIED (fp32 + bf16), MLX {}",
        report.runtime_version
    );
    Ok(ExitCode::SUCCESS)
}
