use std::ops::Mul;

use mlx_rs::ops::{concatenate, stack};
use mlx_rs::Array;

use super::fused_causal_conv_silu;
use crate::qwen35::mlx::layers::differential_tests::host_reference::seeded;
use crate::qwen35::mlx::layers::ops::zeros;
use crate::qwen35::mlx::layers::{CONV_KERNEL, QKV_SIZE};
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

fn reference_hoisted_conv(
    raw_qkv: &Array,
    conv: &Array,
    conv1d: &Array,
    rows: usize,
) -> (Array, Array) {
    let row_count = rows as i32;
    let extended = concatenate(&[&conv.clone().transpose().unwrap(), raw_qkv], 0).unwrap();
    let taps = (0..CONV_KERNEL)
        .map(|tap| {
            let indices: Vec<u32> = (tap + 1..tap + 1 + rows).map(|row| row as u32).collect();
            extended
                .take_axis(Array::from_slice(&indices, &[row_count]), 0)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let windows = stack(&taps, 2).unwrap();
    let summed = windows.mul(conv1d).sum_axis(2, false).unwrap();
    let qkv = mlx_rs::nn::silu(&summed).unwrap();

    let tail: Vec<u32> = (rows..rows + CONV_KERNEL).map(|row| row as u32).collect();
    let next_conv = extended
        .take_axis(Array::from_slice(&tail, &[CONV_KERNEL as i32]), 0)
        .unwrap()
        .transpose()
        .unwrap()
        .contiguous()
        .unwrap();

    (qkv, next_conv)
}

fn host(array: &Array) -> Vec<f32> {
    array.eval().expect("eval");
    array.to_vec_cast::<f32>().expect("host read")
}

fn assert_bits_equal(left: &[f32], right: &[f32], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: length");
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}[{index}]: fused {a} vs ref {b}"
        );
    }
}

fn relative_deviation(actual: &[f32], expected: &[f32]) -> f32 {
    let scale = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
        / scale
}

#[test]
fn fused_causal_conv_matches_reference_hoisted_exactly() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
    runtime
        .execute(|| {
            let conv_w = Array::from_slice(
                &seeded(&[QKV_SIZE * CONV_KERNEL], 42),
                &[QKV_SIZE as i32, CONV_KERNEL as i32],
            );

            for rows in [1, 2, 7, 12, 48, 192] {
                let raw_qkv = Array::from_slice(
                    &seeded(&[rows * QKV_SIZE], 100 + rows as u64),
                    &[rows as i32, QKV_SIZE as i32],
                );
                let conv = Array::from_slice(
                    &seeded(&[QKV_SIZE * CONV_KERNEL], 200 + rows as u64),
                    &[QKV_SIZE as i32, CONV_KERNEL as i32],
                );

                let (ref_out, ref_conv) = reference_hoisted_conv(&raw_qkv, &conv, &conv_w, rows);
                let (fused_out, fused_conv) =
                    fused_causal_conv_silu(&raw_qkv, &conv, &conv_w).unwrap();

                let fused_host = host(&fused_out);
                let ref_host = host(&ref_out);
                let dev = relative_deviation(&fused_host, &ref_host);
                println!("rows={rows} conv output max relative deviation: {dev:e}");
                assert!(
                    dev <= 1e-6,
                    "conv output rows={rows}: relative deviation {dev}"
                );

                // Carried state is untouched by arithmetic rounding; it must match bit for bit.
                assert_bits_equal(
                    &host(&fused_conv),
                    &host(&ref_conv),
                    &format!("next_conv rows={rows}"),
                );

                // Fresh replay must produce bit-identical results.
                let (replay_out, replay_conv) =
                    fused_causal_conv_silu(&raw_qkv, &conv, &conv_w).unwrap();
                assert_bits_equal(
                    &host(&fused_out),
                    &host(&replay_out),
                    &format!("replay output rows={rows}"),
                );
                assert_bits_equal(
                    &host(&fused_conv),
                    &host(&replay_conv),
                    &format!("replay conv rows={rows}"),
                );
            }
        })
        .unwrap();
}

#[test]
fn fused_causal_conv_continuation_is_bit_identical_to_full_sequence() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
    runtime
        .execute(|| {
            let conv_w = Array::from_slice(
                &seeded(&[QKV_SIZE * CONV_KERNEL], 55),
                &[QKV_SIZE as i32, CONV_KERNEL as i32],
            );
            let init_conv = zeros(&[QKV_SIZE as i32, CONV_KERNEL as i32]);

            let full_rows = 16;
            let split = 9;
            let raw_full_vec = seeded(&[full_rows * QKV_SIZE], 777);
            let raw_full = Array::from_slice(&raw_full_vec, &[full_rows as i32, QKV_SIZE as i32]);

            // One-pass full
            let (full_out, full_final_conv) =
                fused_causal_conv_silu(&raw_full, &init_conv, &conv_w).unwrap();

            // Segment 1
            let raw_seg1 = Array::from_slice(
                &raw_full_vec[..split * QKV_SIZE],
                &[split as i32, QKV_SIZE as i32],
            );
            let (seg1_out, seg1_conv) =
                fused_causal_conv_silu(&raw_seg1, &init_conv, &conv_w).unwrap();

            // Segment 2 starting from seg1_conv
            let raw_seg2 = Array::from_slice(
                &raw_full_vec[split * QKV_SIZE..],
                &[(full_rows - split) as i32, QKV_SIZE as i32],
            );
            let (seg2_out, seg2_conv) =
                fused_causal_conv_silu(&raw_seg2, &seg1_conv, &conv_w).unwrap();

            let mut stitched = host(&seg1_out);
            stitched.extend(host(&seg2_out));

            assert_bits_equal(
                &stitched,
                &host(&full_out),
                "stitched segments vs one-pass full output",
            );
            assert_bits_equal(
                &host(&seg2_conv),
                &host(&full_final_conv),
                "final conv state continuation vs one-pass full",
            );
        })
        .unwrap();
}

#[test]
#[ignore = "timing diagnostic, not a correctness gate"]
fn print_reference_versus_fused_conv_timing() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
    runtime
        .execute(|| {
            let conv_w = Array::from_slice(
                &seeded(&[QKV_SIZE * CONV_KERNEL], 42),
                &[QKV_SIZE as i32, CONV_KERNEL as i32],
            );
            println!("\n=== Fused Causal Conv vs Reference Multi-Op Timing ===");
            println!(
                "{:>6} | {:>14} | {:>14} | {:>8}",
                "Rows", "Ref Multi-Op", "Fused Metal", "Speedup"
            );
            println!("{:-<6}-+-{:-<14}-+-{:-<14}-+-{:-<8}", "", "", "", "");

            for rows in [12, 192, 1792] {
                let raw_qkv = Array::from_slice(
                    &seeded(&[rows * QKV_SIZE], 100 + rows as u64),
                    &[rows as i32, QKV_SIZE as i32],
                );
                let conv = Array::from_slice(
                    &seeded(&[QKV_SIZE * CONV_KERNEL], 200 + rows as u64),
                    &[QKV_SIZE as i32, CONV_KERNEL as i32],
                );

                // Warmup
                let (r1, r2) = reference_hoisted_conv(&raw_qkv, &conv, &conv_w, rows);
                r1.eval().unwrap();
                r2.eval().unwrap();
                let (f1, f2) = fused_causal_conv_silu(&raw_qkv, &conv, &conv_w).unwrap();
                f1.eval().unwrap();
                f2.eval().unwrap();

                // Timed Reference
                let iters = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let (o, c) = reference_hoisted_conv(&raw_qkv, &conv, &conv_w, rows);
                    o.eval().unwrap();
                    c.eval().unwrap();
                }
                let ref_time = t0.elapsed() / iters as u32;

                // Timed Fused
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let (o, c) = fused_causal_conv_silu(&raw_qkv, &conv, &conv_w).unwrap();
                    o.eval().unwrap();
                    c.eval().unwrap();
                }
                let fused_time = t0.elapsed() / iters as u32;

                let speedup = ref_time.as_secs_f64() / fused_time.as_secs_f64();
                println!(
                    "{:>6} | {:>11.3} ms | {:>11.3} ms | {:>7.2}x",
                    rows,
                    ref_time.as_secs_f64() * 1000.0,
                    fused_time.as_secs_f64() * 1000.0,
                    speedup
                );
            }
        })
        .unwrap();
}
