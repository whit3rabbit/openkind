//! Helper functions and provenance capture for MLX qualification.

use mlx_rs::Array;
use opendecision_backends::qwen35::mlx::{
    MlxMemorySnapshot, MlxRuntime, MLX_CORE_VERSION, MLX_C_RELEASE, MLX_LM_REFERENCE_COMMIT,
    MLX_RS_VERSION,
};

pub(crate) fn to_f32(array: &Array) -> Vec<f32> {
    array.eval().unwrap_or_else(|e| panic!("eval failed: {e}"));
    array
        .to_vec_cast::<f32>()
        .unwrap_or_else(|error| panic!("failed to read array to host: {error}"))
}

pub(crate) fn close(got: &[f32], host: &[f64], rel_tol: f64) -> bool {
    got.len() == host.len()
        && got.iter().zip(host).all(|(g, h)| {
            let diff = (*g as f64 - h).abs();
            diff <= rel_tol * h.abs().max(1.0)
        })
}

pub(crate) fn host_matmul_f64(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Vec<f64> {
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    assert_eq!(k, k2);
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for idx in 0..k {
                acc += a[i * k + idx] * b[idx * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

pub(crate) fn host_rms_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

pub(crate) fn capture_provenance(
    runtime: &MlxRuntime,
    snapshot: MlxMemorySnapshot,
) -> Result<serde_json::Value, String> {
    let command = |program: &str, args: &[&str]| -> String {
        std::process::Command::new(program)
            .args(args)
            .output()
            .map(|out| {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_owned();
                if text.is_empty() {
                    String::from_utf8_lossy(&out.stderr).trim().to_owned()
                } else {
                    text
                }
            })
            .unwrap_or_else(|error| format!("<unavailable: {error}>"))
    };

    let mut archives = Vec::new();
    let target = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
    for profile in ["debug", "release"] {
        let builds = target.join(profile).join("build");
        let Ok(entries) = std::fs::read_dir(&builds) else {
            continue;
        };
        for entry in entries.flatten() {
            let lib = entry.path().join("out/build/lib");
            for name in ["libmlx.a", "libmlxc.a", "libgguflib.a"] {
                let path = lib.join(name);
                if path.is_file() {
                    let hash = sha256_file(&path)?;
                    archives.push(serde_json::json!({
                        "path": path.to_string_lossy(),
                        "sha256": hash,
                    }));
                }
            }
        }
    }

    let mut metallibs = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        let lib_dir = std::path::Path::new(&home).join(".mlx/lib");
        if let Ok(keys) = std::fs::read_dir(&lib_dir) {
            for key in keys.flatten() {
                let path = key.path().join("mlx.metallib");
                if path.is_file() {
                    metallibs.push(serde_json::json!({
                        "path": path.to_string_lossy(),
                        "key": key.file_name().to_string_lossy(),
                        "sha256": sha256_file(&path)?,
                    }));
                }
            }
        }
    }

    Ok(serde_json::json!({
        "schema": "opendecision-mlx-runtime-provenance/v1",
        "mlx_rs_version": MLX_RS_VERSION,
        "mlx_c_release": MLX_C_RELEASE,
        "mlx_core_version_pinned": MLX_CORE_VERSION,
        "mlx_core_version_linked": runtime.version(),
        "mlx_lm_reference_commit": MLX_LM_REFERENCE_COMMIT,
        "xcode_build": command("xcodebuild", &["-version"]),
        "metal_toolchain": command("xcrun", &["-sdk", "macosx", "metal", "--version"]),
        "macos_product_version": command("sw_vers", &["-productVersion"]),
        "macos_build_version": command("sw_vers", &["-buildVersion"]),
        "rustc": command("rustc", &["--version"]),
        "git_commit": command("git", &["rev-parse", "HEAD"]),
        "inactive_cache_limit_bytes": runtime.config().inactive_cache_limit_bytes,
        "memory_snapshot": {
            "active_bytes": snapshot.active_bytes,
            "cache_bytes": snapshot.cache_bytes,
            "peak_bytes": snapshot.peak_bytes,
        },
        "static_archives": archives,
        "metallibs": metallibs,
    }))
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}
