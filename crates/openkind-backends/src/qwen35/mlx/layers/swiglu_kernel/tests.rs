use std::ops::Mul;
use std::time::Instant;

use mlx_rs::Array;

use super::fused_swiglu;
use crate::qwen35::mlx::layers::differential_tests::host_reference::seeded;
use crate::qwen35::mlx::layers::ops::op;
use crate::qwen35::mlx::layers::INTERMEDIATE_SIZE;
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

fn reference_swiglu(
    gate: &Array,
    up: &Array,
) -> Result<Array, crate::qwen35::mlx::layers::MlxError> {
    Ok(mlx_rs::nn::silu(gate).map_err(op("test ref silu"))?.mul(up))
}

#[test]
fn fused_swiglu_matches_reference_implementation() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| {
            for rows in [1, 4, 12, 48, 192] {
                let count = rows * INTERMEDIATE_SIZE;
                let gate_host = seeded(&[count], 33000 + rows as u64);
                let up_host = seeded(&[count], 44000 + rows as u64);

                let gate = Array::from_slice(&gate_host, &[rows as i32, INTERMEDIATE_SIZE as i32]);
                let up = Array::from_slice(&up_host, &[rows as i32, INTERMEDIATE_SIZE as i32]);

                let ref_out = reference_swiglu(&gate, &up).expect("reference swiglu");
                let fused_out = fused_swiglu(&gate, &up).expect("fused swiglu");

                ref_out.eval().expect("eval ref");
                fused_out.eval().expect("eval fused");

                let ref_slice = ref_out.to_vec_cast::<f32>().expect("read ref");
                let fused_slice = fused_out.to_vec_cast::<f32>().expect("read fused");

                let mut max_abs = 0.0_f32;
                let mut max_rel = 0.0_f32;
                for (&r, &f) in ref_slice.iter().zip(&fused_slice) {
                    let abs = (r - f).abs();
                    max_abs = max_abs.max(abs);
                    let denom = r.abs().max(f.abs());
                    if denom > 1e-6 {
                        max_rel = max_rel.max(abs / denom);
                    }
                }

                eprintln!("[test] rows={rows:3}: max_abs={max_abs:e}, max_rel={max_rel:e}");
                assert!(
                    max_abs < 1e-5,
                    "fused SwiGLU drifted: max_abs={max_abs:e}, max_rel={max_rel:e}"
                );
            }
        })
        .expect("execute");
}

#[test]
#[ignore = "timing diagnostic, not a correctness gate"]
fn print_reference_versus_fused_swiglu_timing() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| {
            for rows in [12, 192, 1792] {
                let count = rows * INTERMEDIATE_SIZE;
                let gate_host = seeded(&[count], 55000 + rows as u64);
                let up_host = seeded(&[count], 66000 + rows as u64);

                let gate = Array::from_slice(&gate_host, &[rows as i32, INTERMEDIATE_SIZE as i32]);
                let up = Array::from_slice(&up_host, &[rows as i32, INTERMEDIATE_SIZE as i32]);

                // Warmup
                for _ in 0..5 {
                    let r = reference_swiglu(&gate, &up).unwrap();
                    r.eval().unwrap();
                    let f = fused_swiglu(&gate, &up).unwrap();
                    f.eval().unwrap();
                }

                const ITERS: usize = 20;
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    let r = reference_swiglu(&gate, &up).unwrap();
                    r.eval().unwrap();
                }
                let ref_time = t0.elapsed().as_secs_f64() * 1000.0 / (ITERS as f64);

                let t1 = Instant::now();
                for _ in 0..ITERS {
                    let f = fused_swiglu(&gate, &up).unwrap();
                    f.eval().unwrap();
                }
                let fused_time = t1.elapsed().as_secs_f64() * 1000.0 / (ITERS as f64);

                eprintln!(
                    "rows {rows:4}: reference {ref_time:.3} ms, fused {fused_time:.3} ms, speedup {:.2}x",
                    ref_time / fused_time
                );
            }
        })
        .expect("execute");
}
