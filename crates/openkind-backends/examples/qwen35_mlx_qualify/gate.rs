//! Qualification gate driver and report generation.

use openkind_backends::qwen35::mlx::{
    MlxRuntime, MlxRuntimeConfig, MLX_C_RELEASE, MLX_LM_REFERENCE_COMMIT, MLX_RS_VERSION,
};

use super::bf16::bf16_section;
use super::fp32::fp32_section;
use super::helpers::capture_provenance;

/// Qualification outcome report recording pass/fail status and failures.
pub struct GateReport {
    /// Whether FP32 reference arithmetic passed all qualification checks.
    pub fp32_pass: bool,
    /// Whether BF16 arithmetic passed all qualification checks.
    pub bf16_pass: bool,
    /// List of failure diagnostics, if any.
    pub failures: Vec<String>,
    /// Runtime version string reported by the linked MLX library.
    pub runtime_version: String,
}

/// Runs the full qualification test suite against the linked MLX runtime.
pub fn run(formal: bool) -> Result<GateReport, String> {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default())
        .map_err(|error| format!("runtime initialization failed: {error}"))?;
    println!(
        "runtime: MLX {} (mlx-rs {}, mlx-c {})",
        runtime.version(),
        MLX_RS_VERSION,
        MLX_C_RELEASE
    );
    println!("reference authority: mlx-lm commit {MLX_LM_REFERENCE_COMMIT} (not executed)");
    println!(
        "inactive cache limit: {} MiB",
        runtime.config().inactive_cache_limit_bytes / (1024 * 1024)
    );

    let mut failures = Vec::new();
    let fp32_pass = fp32_section(&runtime, &mut failures);
    let bf16_pass = bf16_section(&runtime, &mut failures);

    runtime.synchronize().map_err(|e| e.to_string())?;
    let snapshot = runtime.memory_snapshot();
    println!(
        "memory: active={:?} cache={:?} peak={:?}",
        snapshot.active_bytes, snapshot.cache_bytes, snapshot.peak_bytes
    );

    if formal {
        let provenance = capture_provenance(&runtime, snapshot)?;
        let path = "qwen35_mlx_runtime_provenance.json";
        std::fs::write(path, serde_json::to_string_pretty(&provenance).unwrap())
            .map_err(|error| format!("failed to write {path}: {error}"))?;
        println!("provenance written: {path}");
    }

    Ok(GateReport {
        fp32_pass,
        bf16_pass,
        failures,
        runtime_version: runtime.version().to_owned(),
    })
}

/// Verifies that the git working tree is clean before formal qualification.
pub fn require_clean_worktree() -> Result<(), String> {
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .map_err(|error| format!("formal evidence could not inspect git worktree: {error}"))?;
    if !output.status.success() {
        return Err("formal evidence could not inspect git worktree state".to_owned());
    }
    if !output.stdout.is_empty() {
        return Err(
            "formal runtime evidence requires a clean worktree; rerun after committing the implementation".to_owned(),
        );
    }
    Ok(())
}

/// Flatten `Result<Result<T, String>, MlxError>` into `Result<T, String>`.
pub(crate) fn flatten<T>(
    result: Result<Result<T, String>, openkind_backends::qwen35::mlx::MlxError>,
) -> Result<T, String> {
    result
        .map_err(|error| error.to_string())
        .and_then(|inner| inner)
}
