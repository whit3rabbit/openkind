use std::ops::Mul;
use std::time::Instant;

use mlx_rs::Array;

use super::fused_qk_norm_split_expand;
use crate::qwen35::mlx::layers::differential_tests::host_reference::seeded;
use crate::qwen35::mlx::layers::ops::{op, recip_sqrt_eps, repeat_heads_rows, scalar_like};
use crate::qwen35::mlx::layers::{
    HEAD_DIM, KEY_HEADS, KEY_SIZE, QKV_SIZE, RMS_EPSILON, VALUE_HEADS,
};
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

fn reference_qk_norm(
    qkv: &Array,
    rows: usize,
) -> Result<(Array, Array, Array), crate::qwen35::mlx::layers::MlxError> {
    let row_count = rows as i32;
    let query_scale = (HEAD_DIM as f32).sqrt().recip();
    let parts = qkv
        .split_at_indices(&[KEY_SIZE as i32, (2 * KEY_SIZE) as i32], 1)
        .map_err(op("test qkv split"))?;
    let query = parts[0]
        .clone()
        .reshape(&[row_count, KEY_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("test q reshape"))?;
    let key = parts[1]
        .clone()
        .reshape(&[row_count, KEY_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("test k reshape"))?;
    let value = parts[2]
        .clone()
        .reshape(&[row_count, VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("test v reshape"))?;
    let q_sq = query
        .clone()
        .square()
        .map_err(op("test q square"))?
        .sum_axis(2, true)
        .map_err(op("test q sum"))?;
    let k_sq = key
        .clone()
        .square()
        .map_err(op("test k square"))?
        .sum_axis(2, true)
        .map_err(op("test k sum"))?;
    let query_norm = recip_sqrt_eps(&q_sq)?.mul(scalar_like(&q_sq, query_scale));
    let key_norm = recip_sqrt_eps(&k_sq)?;
    let query_hat = repeat_heads_rows(&query)?.mul(&repeat_heads_rows(&query_norm)?);
    let key_hat = repeat_heads_rows(&key)?.mul(&repeat_heads_rows(&key_norm)?);
    Ok((query_hat, key_hat, value))
}

#[test]
fn fused_qk_norm_matches_reference_hoisted_pipeline() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
    runtime
        .execute(|| {
            for &rows in &[1, 4, 12, 48, 192] {
                let qkv_vec = seeded(&[rows * QKV_SIZE], 555);
                let qkv = Array::from_slice(&qkv_vec, &[rows as i32, QKV_SIZE as i32]);

                let (exp_q, exp_k, exp_v) = reference_qk_norm(&qkv, rows).unwrap();
                let exp_q = exp_q.contiguous().unwrap();
                let exp_k = exp_k.contiguous().unwrap();
                let exp_v = exp_v.contiguous().unwrap();
                let (act_q, act_k, act_v) =
                    fused_qk_norm_split_expand(&qkv, RMS_EPSILON).unwrap();

                exp_q.eval().unwrap();
                exp_k.eval().unwrap();
                exp_v.eval().unwrap();
                act_q.eval().unwrap();
                act_k.eval().unwrap();
                act_v.eval().unwrap();

                // Value is exact copy
                assert_eq!(exp_v.as_slice::<f32>(), act_v.as_slice::<f32>());

                for (name, exp, act) in [("query", exp_q, act_q), ("key", exp_k, act_k)] {
                    let exp_slice: &[f32] = exp.as_slice();
                    let act_slice: &[f32] = act.as_slice();
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
                            "{name} mismatch at index {i} for rows {rows}: expected {e}, got {a}, diff {diff}, rel {rel}"
                        );
                    }
                    eprintln!(
                        "[test] rows={rows:3} {name}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
                    );
                }
            }
        })
        .unwrap();
}

#[test]
#[ignore = "timing diagnostic, not a correctness gate"]
fn print_reference_versus_fused_qk_norm_timing() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
    runtime
        .execute(|| {
            for &rows in &[12, 192, 1792] {
                let qkv_vec = seeded(&[rows * QKV_SIZE], 666);
                let qkv = Array::from_slice(&qkv_vec, &[rows as i32, QKV_SIZE as i32]);

                // Warmup
                for _ in 0..5 {
                    let (q, k, v) = reference_qk_norm(&qkv, rows).unwrap();
                    q.eval().unwrap();
                    k.eval().unwrap();
                    v.eval().unwrap();
                    let (fq, fk, fv) = fused_qk_norm_split_expand(&qkv, RMS_EPSILON).unwrap();
                    fq.eval().unwrap();
                    fk.eval().unwrap();
                    fv.eval().unwrap();
                }

                let iters = 50;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let (q, k, v) = reference_qk_norm(&qkv, rows).unwrap();
                    q.eval().unwrap();
                    k.eval().unwrap();
                    v.eval().unwrap();
                }
                let ref_elapsed = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

                let t1 = Instant::now();
                for _ in 0..iters {
                    let (fq, fk, fv) = fused_qk_norm_split_expand(&qkv, RMS_EPSILON).unwrap();
                    fq.eval().unwrap();
                    fk.eval().unwrap();
                    fv.eval().unwrap();
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
