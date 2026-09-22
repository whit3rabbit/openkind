//! BF16 qualification section.

use half::bf16;
use mlx_rs::fast;
use mlx_rs::Array;
use openkind_backends::qwen35::mlx::MlxRuntime;

use super::gate::flatten;
use super::helpers::{close, to_f32};

const BF16_GATE_TOL: f64 = 1.0;

pub(crate) fn bf16_section(runtime: &MlxRuntime, failures: &mut Vec<String>) -> bool {
    let mut pass = true;
    macro_rules! check {
        ($name:literal, $ok:expr, $detail:expr) => {
            if $ok {
                println!("  [PASS] bf16/{}", $name);
            } else {
                let detail: String = $detail;
                println!("  [FAIL] bf16/{}: {}", $name, detail);
                failures.push(format!("bf16/{}: {}", $name, detail));
                pass = false;
            }
        };
    }

    // ---- gather ----
    let result = flatten(runtime.execute(|| {
        let embedding = Array::from_slice(&vec![bf16::from_f32(0.5); 512 * 64], &[512, 64]);
        let rows = Array::from_slice(&[1u32, 511], &[2]);
        embedding.take_axis(&rows, 0).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let got = to_f32(&out);
            let host = vec![0.5_f64; 128];
            check!(
                "take_axis",
                close(&got, &host, 1e-6),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("take_axis", false, error),
    }

    // ---- fast RMSNorm on constants: 0.5 / sqrt(0.25) = 1.0 ----
    let result = flatten(runtime.execute(|| {
        let x = Array::from_slice(&vec![bf16::from_f32(0.5); 4096], &[1, 4096]);
        let weight = Array::from_slice(&vec![bf16::from_f32(1.0); 4096], &[4096]);
        fast::rms_norm(&x, Some(&weight), 1e-5).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let got = to_f32(&out);
            check!(
                "fast_rms_norm",
                got.iter().all(|v| (v - 1.0).abs() <= 1e-2),
                format!("first={}", got.first().copied().unwrap_or(f32::NAN))
            );
        }
        Err(error) => check!("fast_rms_norm", false, error),
    }

    // ---- matmul of ones by 0.25 over 4096: 1024 ----
    let result = flatten(runtime.execute(|| {
        let ones = Array::from_slice(&vec![bf16::from_f32(1.0); 4096], &[1, 4096]);
        let q = Array::from_slice(&vec![bf16::from_f32(0.25); 4096 * 4096], &[4096, 4096]);
        ones.matmul(&q).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let first = to_f32(&out).first().copied().unwrap_or(f32::NAN);
            check!(
                "matmul",
                (first as f64 - 1024.0).abs() <= BF16_GATE_TOL,
                format!("first={first:.8} (expected ~1024)")
            );
        }
        Err(error) => check!("matmul", false, error),
    }

    // ---- issue #115 exact reproduction (bf16) ----
    let result = flatten(runtime.execute(|| -> Result<f32, String> {
        let embedding =
            Array::from_slice(&vec![bf16::from_f32(0.5); 131_072 * 4096], &[131_072, 4096]);
        let rows = Array::from_slice(&[0u32, 1, 2, 3], &[4]);
        let gathered = embedding.take_axis(&rows, 0).map_err(|e| e.to_string())?;
        drop(embedding);
        let weight = Array::from_slice(&vec![bf16::from_f32(1.0); 4096], &[4096]);
        let normed = fast::rms_norm(&gathered, Some(&weight), 1e-5).map_err(|e| e.to_string())?;
        let q = Array::from_slice(&vec![bf16::from_f32(0.25); 4096 * 4096], &[4096, 4096]);
        let out = normed.matmul(&q).map_err(|e| e.to_string())?;
        let first = out
            .take_axis(Array::from_slice(&[0u32], &[1]), 0)
            .and_then(|row| row.take_axis(Array::from_slice(&[0u32], &[1]), 0))
            .map_err(|e| e.to_string())?;
        Ok(to_f32(&first)[0])
    }));
    match result {
        Ok(first) => check!(
            "issue115_pipeline",
            (first as f64 - 1024.0).abs() <= BF16_GATE_TOL,
            format!(
                "first={first:.8} (expected ~1024, broken-toolchain value 512; \
                 ml-explore/mlx-c#115)"
            )
        ),
        Err(error) => check!("issue115_pipeline", false, error),
    }

    pass
}
