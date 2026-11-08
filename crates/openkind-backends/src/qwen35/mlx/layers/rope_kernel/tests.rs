use std::time::Instant;

use mlx_rs::Array;

use super::fused_apply_rotary;
use crate::qwen35::mlx::layers::differential_tests::host_reference::seeded;
use crate::qwen35::mlx::layers::full_attention::apply_rotary as reference_apply_rotary;
use crate::qwen35::mlx::layers::{ATTENTION_HEADS, ATTENTION_HEAD_DIM, KV_HEADS};
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

fn max_relative_difference(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (&x, &y) in a.iter().zip(b) {
        let abs = (x - y).abs();
        max_abs = max_abs.max(abs);
        let denom = x.abs().max(y.abs());
        if denom > 1e-6 {
            max_rel = max_rel.max(abs / denom);
        }
    }
    (max_abs, max_rel)
}

#[test]
fn fused_rope_matches_reference_implementation() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| {
            for rows in [1, 4, 12, 48, 192] {
                for heads in [KV_HEADS, ATTENTION_HEADS] {
                    let seed = 42000 + rows as u64 * 100 + heads as u64;
                    let values_host = seeded(&[rows * heads * ATTENTION_HEAD_DIM], seed);
                    let values = Array::from_slice(
                        &values_host,
                        &[rows as i32, heads as i32, ATTENTION_HEAD_DIM as i32],
                    );
                    let position_start = 17;

                    let ref_out = reference_apply_rotary(&values, rows, position_start, heads)
                        .expect("reference rope");
                    let fused_out = fused_apply_rotary(&values, rows, position_start, heads)
                        .expect("fused rope");

                    ref_out.eval().expect("eval ref");
                    fused_out.eval().expect("eval fused");

                    let ref_slice = ref_out.to_vec_cast::<f32>().expect("host read ref");
                    let fused_slice = fused_out.to_vec_cast::<f32>().expect("host read fused");

                    let (max_abs, max_rel) = max_relative_difference(&ref_slice, &fused_slice);
                    eprintln!(
                        "[test] rows={rows:3} heads={heads:2}: max_abs={max_abs:e}, max_rel={max_rel:e}"
                    );
                    assert!(
                        max_abs <= 2e-5,
                        "fused RoPE drifted: max_abs={max_abs:e}, max_rel={max_rel:e}"
                    );
                }
            }
        })
        .expect("execute");
}

#[test]
#[ignore = "timing diagnostic, not a correctness gate"]
fn print_reference_versus_fused_rope_timing() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| {
            for rows in [12, 192, 1792] {
                let heads = ATTENTION_HEADS;
                let seed = 77000 + rows as u64;
                let values_host = seeded(&[rows * heads * ATTENTION_HEAD_DIM], seed);
                let values = Array::from_slice(
                    &values_host,
                    &[rows as i32, heads as i32, ATTENTION_HEAD_DIM as i32],
                );
                let position_start = 0;

                // Warmup
                for _ in 0..5 {
                    let r = reference_apply_rotary(&values, rows, position_start, heads).unwrap();
                    r.eval().unwrap();
                    let f = fused_apply_rotary(&values, rows, position_start, heads).unwrap();
                    f.eval().unwrap();
                }

                const ITERS: usize = 20;
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    let r = reference_apply_rotary(&values, rows, position_start, heads).unwrap();
                    r.eval().unwrap();
                }
                let ref_time = t0.elapsed().as_secs_f64() * 1000.0 / (ITERS as f64);

                let t1 = Instant::now();
                for _ in 0..ITERS {
                    let f = fused_apply_rotary(&values, rows, position_start, heads).unwrap();
                    f.eval().unwrap();
                }
                let fused_time = t1.elapsed().as_secs_f64() * 1000.0 / (ITERS as f64);

                eprintln!(
                    "rows {rows:4} heads {heads:2}: reference {ref_time:.3} ms, fused {fused_time:.3} ms, speedup {:.2}x",
                    ref_time / fused_time
                );
            }
        })
        .expect("execute");
}
