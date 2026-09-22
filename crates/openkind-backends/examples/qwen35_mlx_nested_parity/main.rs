//! Phase 3M.4 — MLX sequential nested `state → question → candidate`
//! continuation parity against the frozen Phase 3B fixtures.
//!
//! For every fixture case the shared state is prefilled once, each question
//! runs through an immutable [`BranchableState`] fork, and each candidate
//! forks its question state. Declared gates:
//!
//! - head probability parity under the frozen `0.005` tolerance with zero
//!   argmax and zero policy changes against `PROBABILITY_REFERENCE.json`;
//! - continuation positions match the exported token fixtures exactly
//!   (`root → question → candidate`);
//! - root immutability: the retained root's strict content fingerprint is
//!   unchanged by all downstream work and equals an independent fresh
//!   prefill of the same tokens;
//! - fork isolation: sibling candidate forks do not disturb each other
//!   (reversed-order replay of the same suffix reproduces the same feature);
//! - cached-versus-full: the nested candidate feature matches an
//!   independent MLX full-sequence prefill within the Phase 3B `1e-4`
//!   self-consistency guard (the CPU oracle's cached path is *exactly*
//!   equal; MLX reports the achieved bound);
//! - logical root tensor bytes equal the architecture-derived contract. FP32
//!   therefore matches the exported Phase 3B `root_cache_bytes` value
//!   (`59,899,904` for the 98-token root); native BF16 uses two-byte state
//!   elements and is checked against the corresponding half-width contract.
//!
//! Hidden-vector deltas are localization diagnostics, not tolerances. FP32
//! gate failure is fatal (exit 1); BF16 runs the same gates as the
//! candidate profile and exit code 4 records a failed candidate gate.
//!
//! Usage: `qwen35_mlx_nested_parity [--precision fp32|bf16] [--formal]
//! <checkpoint-root> <phase3b-reference-root> <head-bundle-root>` with
//! `--features mlx` on macOS arm64 (`SDKROOT` set). Not part of
//! `cargo test`; never loads the Candle model in this process.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod fixtures;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod stage;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::error::Error;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::fs;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::path::PathBuf;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::process::ExitCode;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use self::fixtures::{git_commit, require_clean_worktree};
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use self::stage::nested_stage;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("qwen35_mlx_nested_parity: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!(
        "qwen35_mlx_nested_parity requires --features mlx on macOS arm64 \
         (build with SDKROOT=$(xcrun --show-sdk-path))"
    );
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn run() -> Result<ExitCode, Box<dyn Error>> {
    use openkind_backends::qwen35::mlx::{
        MlxPrecision, MlxQwen35Backbone, MlxRuntime, MlxRuntimeConfig,
    };

    let mut arguments = std::env::args_os().skip(1);
    let mut precision = "fp32".to_owned();
    let mut formal = false;
    let mut positional = Vec::new();
    while let Some(argument) = arguments.next() {
        match argument.to_string_lossy().into_owned().as_str() {
            "--precision" => {
                precision = arguments
                    .next()
                    .ok_or("--precision requires a value")?
                    .to_string_lossy()
                    .into_owned();
            }
            "--formal" => formal = true,
            other => positional.push(PathBuf::from(other)),
        }
    }
    if !["fp32", "bf16"].contains(&precision.as_str()) {
        return Err(format!("unknown precision `{precision}`").into());
    }
    if positional.len() != 3 {
        return Err(
            "usage: qwen35_mlx_nested_parity [--precision fp32|bf16] [--formal] \
             <checkpoint-root> <phase3b-reference-root> <head-bundle-root>"
                .into(),
        );
    }
    if formal {
        require_clean_worktree()?;
    }
    let runtime = std::sync::Arc::new(
        MlxRuntime::new(MlxRuntimeConfig::default())
            .map_err(|error| format!("runtime init failed: {error}"))?,
    );
    let precision = if precision == "bf16" {
        MlxPrecision::NativeBf16
    } else {
        MlxPrecision::Fp32
    };
    if precision == MlxPrecision::NativeBf16 {
        runtime
            .qualify_bf16()
            .map_err(|error| format!("native BF16 runtime qualification failed: {error}"))?;
    }
    let backbone = MlxQwen35Backbone::load(&positional[0], runtime, precision)
        .map_err(|error| format!("MLX backbone load failed: {error}"))?;
    println!("backend: {}", backbone.arithmetic_id());

    let report = nested_stage(&backbone, &positional[1], &positional[2])?;
    println!("{}", serde_json::to_string_pretty(&report.json)?);
    if formal {
        let path = format!(
            "qwen35_mlx_nested_parity_{}_{}.json",
            precision.as_str(),
            git_commit().unwrap_or_else(|| "dirty".into())
        );
        fs::write(&path, serde_json::to_string_pretty(&report.json)?)
            .map_err(|error| format!("failed to write {path}: {error}"))?;
        println!("formal report written: {path}");
    }
    if !report.gate_passed {
        return Ok(ExitCode::from(4));
    }
    Ok(ExitCode::SUCCESS)
}
