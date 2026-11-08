use std::ops::Mul;
use std::time::Instant;

use mlx_rs::{fast, Array};

use super::fused_gated_norm_silu;
use crate::qwen35::mlx::layers::differential_tests::host_reference::seeded;
use crate::qwen35::mlx::layers::ops::op;
use crate::qwen35::mlx::layers::{HEAD_DIM, RMS_EPSILON, VALUE_HEADS, VALUE_SIZE};
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

fn reference_gated_norm(
    y: &Array,
    z: &Array,
    weight: &Array,
    eps: f32,
    rows: usize,
) -> Result<Array, crate::qwen35::mlx::layers::MlxError> {
    let row_count = rows as i32;
    let y_3d = y
        .reshape(&[row_count, VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("test y reshape"))?;
    let z_3d = z
        .reshape(&[row_count, VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("test z reshape"))?;
    let normed = fast::rms_norm(&y_3d, Some(weight), eps).map_err(op("test rms norm"))?;
    let silu_z = mlx_rs::nn::silu(&z_3d).map_err(op("test silu"))?;
    normed
        .mul(&silu_z)
        .reshape(&[row_count, VALUE_SIZE as i32])
        .map_err(op("test out reshape"))
}

#[test]
fn fused_gated_norm_matches_reference_mlx_ops() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
    runtime
        .execute(|| {
            for &rows in &[1, 4, 12, 48, 192] {
                let y_vec = seeded(&[rows * VALUE_SIZE], 101);
                let z_vec = seeded(&[rows * VALUE_SIZE], 202);
                let w_vec = seeded(&[HEAD_DIM], 303);

                let y = Array::from_slice(&y_vec, &[rows as i32, VALUE_HEADS as i32, HEAD_DIM as i32]);
                let z = Array::from_slice(&z_vec, &[rows as i32, VALUE_SIZE as i32]);
                let weight = Array::from_slice(&w_vec, &[HEAD_DIM as i32]);

                let expected = reference_gated_norm(&y, &z, &weight, RMS_EPSILON, rows).unwrap();
                let actual = fused_gated_norm_silu(&y, &z, &weight, RMS_EPSILON).unwrap();

                expected.eval().unwrap();
                actual.eval().unwrap();

                let exp_slice: &[f32] = expected.as_slice();
                let act_slice: &[f32] = actual.as_slice();

                assert_eq!(exp_slice.len(), act_slice.len());
                let mut max_abs = 0.0f32;
                let mut max_rel = 0.0f32;
                for (i, (&e, &a)) in exp_slice.iter().zip(act_slice.iter()).enumerate() {
                    let diff = (e - a).abs();
                    if diff > max_abs {
                        max_abs = diff;
                    }
                    let denom = e.abs().max(a.abs()).max(1e-6);
                    let rel = diff / denom;
                    if rel > max_rel {
                        max_rel = rel;
                    }
                    assert!(
                        rel < 1e-4,
                        "mismatch at index {i} for rows {rows}: expected {e}, got {a}, diff {diff}, rel {rel}"
                    );
                }
                eprintln!(
                    "[test] rows={rows:3}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
                );
            }
        })
        .unwrap();
}

#[test]
#[ignore = "timing diagnostic, not a correctness gate"]
fn print_reference_versus_fused_gated_norm_timing() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
    runtime
        .execute(|| {
            for &rows in &[12, 192, 1792] {
                let y_vec = seeded(&[rows * VALUE_SIZE], 111);
                let z_vec = seeded(&[rows * VALUE_SIZE], 222);
                let w_vec = seeded(&[HEAD_DIM], 333);

                let y = Array::from_slice(&y_vec, &[rows as i32, VALUE_HEADS as i32, HEAD_DIM as i32]);
                let z = Array::from_slice(&z_vec, &[rows as i32, VALUE_SIZE as i32]);
                let weight = Array::from_slice(&w_vec, &[HEAD_DIM as i32]);

                // Warmup
                for _ in 0..5 {
                    let ref_out = reference_gated_norm(&y, &z, &weight, RMS_EPSILON, rows).unwrap();
                    ref_out.eval().unwrap();
                    let fused_out = fused_gated_norm_silu(&y, &z, &weight, RMS_EPSILON).unwrap();
                    fused_out.eval().unwrap();
                }

                let iters = 50;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let out = reference_gated_norm(&y, &z, &weight, RMS_EPSILON, rows).unwrap();
                    out.eval().unwrap();
                }
                let ref_elapsed = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

                let t1 = Instant::now();
                for _ in 0..iters {
                    let out = fused_gated_norm_silu(&y, &z, &weight, RMS_EPSILON).unwrap();
                    out.eval().unwrap();
                }
                let fused_elapsed = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

                eprintln!(
                    "rows {rows:4}: reference {ref_elapsed:.3} ms, fused {fused_elapsed:.3} ms, speedup {:.2}x",
                    ref_elapsed / fused_elapsed
                );
            }
        })
        .unwrap();
}
