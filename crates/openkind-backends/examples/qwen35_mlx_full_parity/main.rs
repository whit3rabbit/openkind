//! Phase 3M.2–3M.4 — MLX full-sequence parity gate against the frozen
//! Phase 3B bundle.
//!
//! Staged comparison keeps first-divergent-layer debugging focused. The
//! verified weight load is complete for every stage, while the embedding-only
//! stage skips decoder execution:
//!
//! - `--stage embedding` (default `all`): prefill the trace input and
//!   compare the final-token embedding row against the golden `embedding`
//!   trace vector. The embedding is read host-side with the oracle's exact
//!   widening, so this gate is **bit-exact**.
//! - `--stage layer0`: compare layer_00's final-token output against the
//!   golden trace (localization diagnostic; sanity-bounded, not parity).
//! - `--stage full`: all 10 Phase 3B candidates through the complete
//!   32-layer backbone and the fitted head under the **frozen** gates —
//!   probability ≤ 0.005, zero argmax changes, zero policy changes — plus
//!   the full 34-stage trace as diagnostics. FP32 failure is fatal
//!   (exit 1). BF16 runs the same gates as the `mlx-native-bf16` candidate
//!   profile: its result is reported and exit code 4 marks a failed
//!   candidate gate (an outcome to record, not an infrastructure error).
//!
//! Usage:
//!
//! ```text
//! cargo run -p openkind-backends --features mlx --release \
//!     --example qwen35_mlx_full_parity -- \
//!     [--stage embedding|layer0|full|all] [--precision fp32|bf16] [--formal] \
//!     <checkpoint-root> <phase3b-reference-root> <head-bundle-root>
//! ```
//!
//! This example requires `--features mlx` on macOS arm64 with `SDKROOT`
//! set. It never loads the Candle model: every comparison target is a
//! Phase 3B saved vector.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod fixtures;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod full;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod trace;

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
use self::full::full_stage;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use self::trace::trace_stage;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("qwen35_mlx_full_parity: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!(
        "qwen35_mlx_full_parity requires --features mlx on macOS arm64 \
         (build with SDKROOT=$(xcrun --show-sdk-path))"
    );
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn run() -> Result<ExitCode, Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let mut stage = "all".to_owned();
    let mut precision = "fp32".to_owned();
    let mut formal = false;
    let mut positional = Vec::new();
    while let Some(argument) = arguments.next() {
        let argument = argument.to_string_lossy().into_owned();
        match argument.as_str() {
            "--stage" => {
                stage = arguments
                    .next()
                    .ok_or("--stage requires a value")?
                    .to_string_lossy()
                    .into_owned();
            }
            "--precision" => {
                precision = arguments
                    .next()
                    .ok_or("--precision requires a value")?
                    .to_string_lossy()
                    .into_owned();
            }
            "--formal" => formal = true,
            _ => positional.push(PathBuf::from(argument)),
        }
    }
    if !["embedding", "layer0", "full", "all"].contains(&stage.as_str()) {
        return Err(format!("unknown stage `{stage}`").into());
    }
    if !["fp32", "bf16"].contains(&precision.as_str()) {
        return Err(format!("unknown precision `{precision}`").into());
    }
    if positional.len() != 3 {
        return Err(
            "usage: qwen35_mlx_full_parity [--stage embedding|layer0|full|all] \
             [--precision fp32|bf16] [--formal] <checkpoint-root> \
             <phase3b-reference-root> <head-bundle-root>"
                .into(),
        );
    }
    if formal {
        require_clean_worktree()?;
    }
    let checkpoint_root = positional[0].clone();
    let reference_root = positional[1].clone();
    let head_bundle_root = positional[2].clone();

    let runtime = {
        use openkind_backends::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};
        std::sync::Arc::new(
            MlxRuntime::new(MlxRuntimeConfig::default())
                .map_err(|error| format!("runtime init failed: {error}"))?,
        )
    };
    let precision = match precision.as_str() {
        "bf16" => openkind_backends::qwen35::mlx::MlxPrecision::NativeBf16,
        _ => openkind_backends::qwen35::mlx::MlxPrecision::Fp32,
    };
    if precision == openkind_backends::qwen35::mlx::MlxPrecision::NativeBf16 {
        runtime
            .qualify_bf16()
            .map_err(|error| format!("native BF16 runtime qualification failed: {error}"))?;
    }
    let backbone = openkind_backends::qwen35::mlx::MlxQwen35Backbone::load(
        &checkpoint_root,
        runtime.clone(),
        precision,
    )
    .map_err(|error| format!("MLX backbone load failed: {error}"))?;
    println!(
        "backend: {} (MLX runtime {}), weight load {:.1}s, peak MLX {:?} peak RSS {:?}",
        backbone.arithmetic_id(),
        runtime.version(),
        backbone.load_report.load_seconds,
        backbone.load_report.peak_active_mlx_bytes,
        backbone.load_report.peak_process_bytes,
    );

    let reference = openkind_backends::qwen35::BackboneReference::load(&reference_root)?;

    if ["embedding", "layer0", "all"].contains(&stage.as_str()) {
        let trace = trace_stage(&backbone, &reference, stage == "embedding")?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "stage": "trace",
                "embedding_exact": trace.embedding_exact,
                "layer0_max_abs": trace.layer0_max_abs,
                "stages": trace.summary,
            }))?
        );
        if stage == "embedding" && !trace.embedding_exact {
            return Err("embedding gate failed: MLX embedding row is not bit-exact".into());
        }
        if stage == "layer0" && (!trace.embedding_exact || !trace.layer0_max_abs.is_finite()) {
            return Err("layer0 gate failed".into());
        }
        if stage == "layer0" && trace.layer0_max_abs > 1e-2 {
            return Err(format!(
                "layer0 sanity bound exceeded: max_abs {:.6} > 1e-2",
                trace.layer0_max_abs
            )
            .into());
        }
    }

    if ["full", "all"].contains(&stage.as_str()) {
        let report = full_stage(&backbone, &reference, &reference_root, &head_bundle_root)?;
        let gate_passed = report.gate_passed;
        println!("{}", serde_json::to_string_pretty(&report.json)?);
        if formal {
            let path = format!(
                "qwen35_mlx_full_parity_{}_{}.json",
                match precision {
                    openkind_backends::qwen35::mlx::MlxPrecision::Fp32 => "fp32",
                    openkind_backends::qwen35::mlx::MlxPrecision::NativeBf16 => "bf16",
                },
                git_commit().unwrap_or_else(|| "dirty".into())
            );
            fs::write(&path, serde_json::to_string_pretty(&report.json)?)
                .map_err(|error| format!("failed to write {path}: {error}"))?;
            println!("formal report written: {path}");
        }
        if !gate_passed {
            return Ok(ExitCode::from(4));
        }
    }
    Ok(ExitCode::SUCCESS)
}
