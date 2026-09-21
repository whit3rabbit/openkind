//! FP32 qualification section.

use mlx_rs::fast;
use mlx_rs::ops::{concatenate, conv1d, cumsum, reshape};
use mlx_rs::Array;
use opendecision_backends::qwen35::mlx::MlxRuntime;

use super::gate::flatten;
use super::helpers::{close, host_matmul_f64, host_rms_norm, to_f32};

const FP32_REL_TOL: f64 = 1e-5;

pub(crate) fn fp32_section(runtime: &MlxRuntime, failures: &mut Vec<String>) -> bool {
    let mut pass = true;
    macro_rules! check {
        ($name:literal, $ok:expr, $detail:expr) => {
            if $ok {
                println!("  [PASS] fp32/{}", $name);
            } else {
                let detail: String = $detail;
                println!("  [FAIL] fp32/{}: {}", $name, detail);
                failures.push(format!("fp32/{}: {}", $name, detail));
                pass = false;
            }
        };
    }

    // ---- gather (take_axis) ----
    let result = flatten(runtime.execute(|| {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
        let array = Array::from_slice(&data, &[8, 4]);
        let indices = Array::from_slice(&[1u32, 3, 7], &[3]);
        array.take_axis(&indices, 0).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let host: Vec<f64> = [1usize, 3, 7]
                .iter()
                .flat_map(|r| {
                    ((r * 4)..(r * 4 + 4))
                        .map(|i| i as f64 * 0.25)
                        .collect::<Vec<_>>()
                })
                .collect();
            let got = to_f32(&out);
            check!(
                "take_axis",
                close(&got, &host, FP32_REL_TOL),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("take_axis", false, error),
    }

    // ---- matmul ----
    let a_data = (0..64)
        .map(|i| (i % 16) as f64 * 0.125 - 1.0)
        .collect::<Vec<_>>();
    let b_data = (0..64)
        .map(|i| ((i * 7) % 13) as f64 * 0.0625)
        .collect::<Vec<_>>();
    let result = flatten(runtime.execute(|| {
        let a = Array::from_slice(
            &(0..64)
                .map(|i| (i % 16) as f32 * 0.125 - 1.0)
                .collect::<Vec<_>>(),
            &[4, 16],
        );
        let b = Array::from_slice(
            &(0..64)
                .map(|i| ((i * 7) % 13) as f32 * 0.0625)
                .collect::<Vec<_>>(),
            &[16, 4],
        );
        a.matmul(&b).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let host = host_matmul_f64(&a_data, &[4, 16], &b_data, &[16, 4]);
            let got = to_f32(&out);
            check!(
                "matmul",
                close(&got, &host, FP32_REL_TOL),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("matmul", false, error),
    }

    // ---- fast RMSNorm (x / sqrt(mean(x^2)+eps) * w) ----
    let input_f64: Vec<f64> = (0..64)
        .map(|i| ((i * 31) % 17) as f64 * 0.1 - 0.8)
        .collect();
    let result = flatten(runtime.execute(|| {
        let data: Vec<f32> = (0..64)
            .map(|i| ((i * 31) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let x = Array::from_slice(&data, &[1, 64]);
        let weight = Array::from_slice(&vec![1.0_f32; 64], &[64]);
        fast::rms_norm(&x, Some(&weight), 1e-5).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let host = host_rms_norm(&input_f64, 1e-5);
            let got = to_f32(&out);
            check!(
                "fast_rms_norm",
                close(&got, &host, 1e-4),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("fast_rms_norm", false, error),
    }

    // ---- fast RoPE: dims=2, freqs=[1.0] => row p rotated by p radians ----
    let result = flatten(runtime.execute(|| {
        let pairs: Vec<f32> = vec![
            1.0, 0.25, -0.5, 0.75, 0.125, -0.375, 0.625, 0.875, -0.25, 0.5,
        ];
        let x = Array::from_slice(&pairs, &[1, 5, 2]);
        let freqs = Array::from_slice(&[1.0_f32], &[1]);
        fast::rope(&x, 2, false, None, 1.0, 0, Some(&freqs)).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let pairs = [
                (1.0_f64, 0.25_f64),
                (-0.5, 0.75),
                (0.125, -0.375),
                (0.625, 0.875),
                (-0.25, 0.5),
            ];
            let mut host = Vec::new();
            for (position, (a, b)) in pairs.iter().enumerate() {
                let angle = position as f64;
                let (s, c) = angle.sin_cos();
                host.push(a * c - b * s);
                host.push(a * s + b * c);
            }
            let got = to_f32(&out);
            check!(
                "fast_rope",
                close(&got, &host, 1e-4),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("fast_rope", false, error),
    }

    // ---- depthwise causal conv1d (native MLX conv, groups=C) ----
    // MLX conv1d layout: input (N, L, C), weight (C_out, K, C_in/groups).
    // MLX conv1d computes TRUE convolution (kernel reversed), unlike the
    // cross-correlation convention of torch/candle: flip the kernel along
    // K to obtain the causal cross-correlation semantics Qwen uses.
    const C: usize = 4;
    const L: usize = 8;
    const K: usize = 3;
    let result = flatten(runtime.execute(|| {
        let mut input = vec![0.0_f32; (L + K - 1) * C]; // left-padded with zeros
        for t in 0..L {
            for c in 0..C {
                input[((K - 1 + t) * C) + c] = (((t * 5 + c * 3) % 11) as f32) * 0.1 - 0.5;
            }
        }
        let mut weight = [0.0_f32; C * K];
        for c in 0..C {
            for k in 0..K {
                weight[c * K + k] = (((c + k * 2) % 5) as f32) * 0.25 - 0.3;
            }
        }
        let mut flipped = [0.0_f32; C * K];
        for c in 0..C {
            for k in 0..K {
                flipped[c * K + k] = weight[c * K + (K - 1 - k)];
            }
        }
        let x = Array::from_slice(&input, &[1, (L + K - 1) as i32, C as i32]);
        let w = Array::from_slice(&flipped, &[C as i32, K as i32, 1]);
        conv1d(&x, &w, None, None, None, C as i32).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let weight = |c: usize, k: usize| -> f64 { (((c + k * 2) % 5) as f64) * 0.25 - 0.3 };
            let input = |t: i64, c: usize| -> f64 {
                if t < 0 {
                    0.0
                } else {
                    (((t as usize * 5 + c * 3) % 11) as f64) * 0.1 - 0.5
                }
            };
            let mut host = Vec::new();
            for t in 0..L {
                for c in 0..C {
                    let mut acc = 0.0;
                    for k in 0..K {
                        acc += weight(c, k) * input(t as i64 - k as i64, c);
                    }
                    host.push(acc);
                }
            }
            let got = to_f32(&out);
            check!(
                "conv1d_groups",
                close(&got, &host, 1e-4),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("conv1d_groups", false, error),
    }

    // ---- cumsum (inclusive) ----
    let result = flatten(runtime.execute(|| {
        let data: Vec<f32> = (0..16).map(|i| ((i * 7) % 5) as f32 * 0.25 - 0.5).collect();
        let x = Array::from_slice(&data, &[16]);
        cumsum(&x, None, None, None).map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let data: Vec<f64> = (0..16).map(|i| ((i * 7) % 5) as f64 * 0.25 - 0.5).collect();
            let host = data
                .iter()
                .scan(0.0, |acc, v| {
                    *acc += v;
                    Some(*acc)
                })
                .collect::<Vec<_>>();
            let got = to_f32(&out);
            check!(
                "cumsum",
                close(&got, &host, 1e-4),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("cumsum", false, error),
    }

    // ---- reshape / concatenate / gather round trip ----
    let result = flatten(runtime.execute(|| {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Array::from_slice(&data, &[2, 3, 4]);
        let flat = reshape(&x, &[6, 4])?;
        let doubled = concatenate(&[&flat, &flat], 0)?;
        let back = doubled.take_axis(Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[6]), 0)?;
        back.contiguous().map_err(|e| e.to_string())
    }));
    match result {
        Ok(out) => {
            let host: Vec<f64> = (0..24).map(|i| i as f64).collect();
            let got = to_f32(&out);
            check!(
                "reshape_concatenate",
                close(&got, &host, 1e-6),
                format!("{got:?} vs {host:?}")
            );
        }
        Err(error) => check!("reshape_concatenate", false, error),
    }

    // ---- issue #115 combined pipeline, FP32 ----
    let result = flatten(runtime.execute(|| -> Result<f32, String> {
        let embedding = Array::from_slice(&vec![0.5_f32; 131_072 * 4096], &[131_072, 4096]);
        let rows = Array::from_slice(&[7u32, 100, 4095, 131_071], &[4]);
        let gathered = embedding.take_axis(&rows, 0).map_err(|e| e.to_string())?;
        drop(embedding);
        let weight = Array::from_slice(&vec![1.0_f32; 4096], &[4096]);
        let normed = fast::rms_norm(&gathered, Some(&weight), 1e-5).map_err(|e| e.to_string())?;
        let q = Array::from_slice(&vec![0.25_f32; 4096 * 4096], &[4096, 4096]);
        let out = normed.matmul(&q).map_err(|e| e.to_string())?;
        let first = out
            .take_axis(Array::from_slice(&[0u32], &[1]), 0)
            .and_then(|row| row.take_axis(Array::from_slice(&[0u32], &[1]), 0))
            .map_err(|e| e.to_string())?;
        Ok(to_f32(&first)[0])
    }));
    match result {
        Ok(first) => check!(
            "issue115_pipeline_fp32",
            (first as f64 - 1024.0).abs() <= 0.05,
            format!("first={first:.8} (expected ~1024)")
        ),
        Err(error) => check!("issue115_pipeline_fp32", false, error),
    }

    // ---- synchronize ----
    match runtime.synchronize() {
        Ok(()) => check!("synchronize", true, String::new()),
        Err(error) => check!("synchronize", false, error.to_string()),
    }

    pass
}
