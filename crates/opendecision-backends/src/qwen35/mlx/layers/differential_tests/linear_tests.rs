use std::ops::Mul;

use mlx_rs::Array;

use super::super::linear_attention::gated_delta_step;
use super::super::ops::{linear, op, rms_norm, shift_window, zeros};
use super::super::*;
use super::host_reference::*;
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

#[test]
fn synthetic_stage_bisect() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let tensors = synthetic_tensors();
    let layer = MlxDecoderLayer::load(&tensors, 0, MlxPrecision::Fp32).expect("layer");
    let w = host_weights();
    let hidden_values = seeded(&[ROWS * HIDDEN_SIZE], 99);

    let read = |array: &Array| -> Vec<f32> {
        array.eval().expect("eval");
        array.to_vec_cast::<f32>().expect("read")
    };
    let delta = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    };

    // Stage 1: input layernorm.
    let host_normed = host_rms_norm(&hidden_values, &w.input_layernorm);
    let mlx_normed = runtime
        .execute(|| {
            let hidden = Array::from_slice(&hidden_values, &[ROWS as i32, HIDDEN_SIZE as i32]);
            rms_norm(&hidden, &layer.input_layernorm).expect("norm")
        })
        .expect("execute");
    println!(
        "stage normed:      max_abs {:.6}",
        delta(&read(&mlx_normed), &host_normed)
    );

    // Stage 2: qkv projection.
    let host_qkv = host_matmul(&host_normed, ROWS, &w.in_proj_qkv, QKV_SIZE);
    let mlx_qkv = runtime
        .execute(|| {
            let TokenMixer::Linear(mixer) = &layer.mixer else {
                panic!("linear")
            };
            linear(&mlx_normed, &mixer.in_proj_qkv).expect("qkv")
        })
        .expect("execute");
    let qkv_delta = delta(&read(&mlx_qkv), &host_qkv);
    println!("stage qkv:         max_abs {qkv_delta:.6}");
    assert!(qkv_delta.is_finite() && qkv_delta <= 1e-2);

    // Stage 3: one recurrence step with a fixed synthetic convolved row.
    let mut qkv_row = seeded(&[QKV_SIZE], 71);
    // Apply the same silu the layer would apply post-conv.
    for value in &mut qkv_row {
        *value = silu(*value);
    }
    let z_row = seeded(&[VALUE_SIZE], 72);
    let beta_row = seeded(&[VALUE_HEADS], 73);
    let decay_row = seeded(&[VALUE_HEADS], 74);
    let host_step = host_recurrence_step(&qkv_row, &z_row, &beta_row, &decay_row, &w);
    let mlx_step = runtime
        .execute(|| -> Vec<f32> {
            let TokenMixer::Linear(mixer) = &layer.mixer else {
                panic!("linear")
            };
            let mut recurrent = zeros(&[VALUE_HEADS as i32, HEAD_DIM as i32, HEAD_DIM as i32]);
            let out = gated_delta_step(
                &Array::from_slice(&qkv_row, &[QKV_SIZE as i32]),
                &Array::from_slice(&z_row, &[VALUE_SIZE as i32]),
                &Array::from_slice(&beta_row, &[VALUE_HEADS as i32]),
                &Array::from_slice(&decay_row, &[VALUE_HEADS as i32]),
                &mixer.dt_bias,
                &mixer.a_log,
                &mixer.delta_norm,
                &mut recurrent,
            )
            .expect("step");
            read(&out)
        })
        .expect("execute");
    let recurrence_delta = delta(&mlx_step, &host_step.0);
    println!("stage recurrence:  max_abs {recurrence_delta:.6}");
    assert!(recurrence_delta.is_finite() && recurrence_delta <= 1e-2);

    // Stage 4: the complete linear mixer (conv loop + recurrence +
    // out_proj) over all rows.
    let host_mixer = host_linear_mixer(&host_normed, &w);
    let TokenMixer::Linear(mixer) = &layer.mixer else {
        panic!("linear")
    };
    let mlx_mixer = runtime
        .execute(|| mixer.forward(&mlx_normed, ROWS, None).expect("mixer").0)
        .expect("execute");
    let mixer_delta = delta(&read(&mlx_mixer), &host_mixer);
    println!("stage mixer:        max_abs {mixer_delta:.6}");
    assert!(mixer_delta.is_finite() && mixer_delta <= 1e-2);

    // Stage 5: one-row mixer (rows = 1) to isolate cross-row state.
    let one_normed = &host_normed[..HIDDEN_SIZE];
    let host_one = host_linear_mixer_one(one_normed, &w);
    let mlx_one = runtime
        .execute(|| {
            let input = Array::from_slice(one_normed, &[1, HIDDEN_SIZE as i32]);
            let TokenMixer::Linear(mixer) = &layer.mixer else {
                panic!("linear")
            };
            let (out, _) = mixer.forward(&input, 1, None).expect("mixer one");
            out
        })
        .expect("execute");
    let mixer_one_delta = delta(&read(&mlx_one), &host_one);
    println!("stage mixer_one:    max_abs {mixer_one_delta:.6}");
    assert!(mixer_one_delta.is_finite() && mixer_one_delta <= 1e-2);

    // Stage 6: onset scan — mixer forward for growing row counts.
    for probe_rows in [2_usize, 3, 4, 8] {
        let host_probe = host_linear_mixer(&host_normed[..probe_rows * HIDDEN_SIZE], &w);
        let mlx_probe = runtime
            .execute(|| {
                let input = Array::from_slice(
                    &host_normed[..probe_rows * HIDDEN_SIZE],
                    &[probe_rows as i32, HIDDEN_SIZE as i32],
                );
                let TokenMixer::Linear(mixer) = &layer.mixer else {
                    panic!("linear")
                };
                let (out, _) = mixer
                    .forward(&input, probe_rows, None)
                    .expect("mixer probe");
                out
            })
            .expect("execute");
        let probe_delta = delta(&read(&mlx_probe), &host_probe);
        println!("stage mixer_rows={probe_rows}: max_abs {probe_delta:.6}");
        assert!(probe_delta.is_finite() && probe_delta <= 1e-2);
    }

    // Stage 7: chained recurrence across two fixed inputs (no conv).
    let qkv_a = seeded(&[QKV_SIZE], 81);
    let qkv_b = seeded(&[QKV_SIZE], 82);
    for value in [&qkv_a, &qkv_b] {
        let _ = value;
    }
    let z_a = seeded(&[VALUE_SIZE], 83);
    let z_b = seeded(&[VALUE_SIZE], 84);
    let beta_a = seeded(&[VALUE_HEADS], 85);
    let beta_b = seeded(&[VALUE_HEADS], 86);
    let decay_a = seeded(&[VALUE_HEADS], 87);
    let decay_b = seeded(&[VALUE_HEADS], 88);
    let mut qkv_a_s = qkv_a.clone();
    for v in &mut qkv_a_s {
        *v = silu(*v);
    }
    let mut qkv_b_s = qkv_b.clone();
    for v in &mut qkv_b_s {
        *v = silu(*v);
    }
    let (_host_step1, host_state1) = host_recurrence_step_raw(
        &qkv_a_s,
        &z_a,
        &beta_a,
        &decay_a,
        &w.dt_bias,
        &w.a_log,
        &w.delta_norm,
        &vec![0.0; VALUE_HEADS * HEAD_DIM * HEAD_DIM],
    );
    let (host_step2, _host_state2) = host_recurrence_step_raw(
        &qkv_b_s,
        &z_b,
        &beta_b,
        &decay_b,
        &w.dt_bias,
        &w.a_log,
        &w.delta_norm,
        &host_state1,
    );
    let mlx_chain = runtime
        .execute(|| -> Vec<f32> {
            let TokenMixer::Linear(mixer) = &layer.mixer else {
                panic!("linear")
            };
            let mut recurrent = zeros(&[VALUE_HEADS as i32, HEAD_DIM as i32, HEAD_DIM as i32]);
            let _ = gated_delta_step(
                &Array::from_slice(&qkv_a_s, &[QKV_SIZE as i32]),
                &Array::from_slice(&z_a, &[VALUE_SIZE as i32]),
                &Array::from_slice(&beta_a, &[VALUE_HEADS as i32]),
                &Array::from_slice(&decay_a, &[VALUE_HEADS as i32]),
                &mixer.dt_bias,
                &mixer.a_log,
                &mixer.delta_norm,
                &mut recurrent,
            )
            .expect("step a");
            let step_b = gated_delta_step(
                &Array::from_slice(&qkv_b_s, &[QKV_SIZE as i32]),
                &Array::from_slice(&z_b, &[VALUE_SIZE as i32]),
                &Array::from_slice(&beta_b, &[VALUE_HEADS as i32]),
                &Array::from_slice(&decay_b, &[VALUE_HEADS as i32]),
                &mixer.dt_bias,
                &mixer.a_log,
                &mixer.delta_norm,
                &mut recurrent,
            )
            .expect("step b");
            let flat = step_b;
            flat.eval().expect("eval");
            flat.to_vec_cast::<f32>().expect("read")
        })
        .expect("execute");
    println!(
        "stage recurrence_two: max_abs {:.6}",
        delta(&mlx_chain, &host_step2)
    );

    // Stage 8: conv window chain across two rows extracted from a lazy
    // projected array (mirrors the mixer loop wiring exactly).
    let raw_values = seeded(&[2 * QKV_SIZE], 91);
    let host_conv = {
        let mut window = vec![0.0_f32; QKV_SIZE * CONV_KERNEL];
        let mut outputs = Vec::new();
        for row in 0..2 {
            for c in 0..QKV_SIZE {
                for tap in 0..CONV_KERNEL - 1 {
                    window[c * CONV_KERNEL + tap] = window[c * CONV_KERNEL + tap + 1];
                }
                window[c * CONV_KERNEL + (CONV_KERNEL - 1)] = raw_values[row * QKV_SIZE + c];
            }
            let mut row_out = vec![0.0_f32; QKV_SIZE];
            for c in 0..QKV_SIZE {
                let mut acc = 0.0_f32;
                for tap in 0..CONV_KERNEL {
                    acc += window[c * CONV_KERNEL + tap] * w.conv1d[c * CONV_KERNEL + tap];
                }
                row_out[c] = silu(acc);
            }
            outputs.extend(row_out);
        }
        outputs
    };
    let mlx_conv = runtime
        .execute(|| -> Vec<f32> {
            let TokenMixer::Linear(mixer) = &layer.mixer else {
                panic!("linear")
            };
            let mut conv = zeros(&[QKV_SIZE as i32, CONV_KERNEL as i32]);
            let mut rows_out = Vec::new();
            for row in 0..2 {
                let raw_row = Array::from_slice(
                    &raw_values[row * QKV_SIZE..(row + 1) * QKV_SIZE],
                    &[QKV_SIZE as i32],
                );
                let window = shift_window(&conv, &raw_row).expect("window");
                let summed = window
                    .clone()
                    .mul(&mixer.conv1d)
                    .sum_axis(1, false)
                    .expect("sum");
                let silued = mlx_rs::nn::silu(&summed).expect("silu");
                silued.eval().expect("eval");
                rows_out.extend(silued.to_vec_cast::<f32>().expect("read"));
                conv = window;
            }
            rows_out
        })
        .expect("execute");
    println!(
        "stage conv_two:     max_abs {:.6}",
        delta(&mlx_conv, &host_conv)
    );

    // rows=2 value dump vs host and vs rows=8 first rows.
    let host2 = host_linear_mixer(&host_normed[..2 * HIDDEN_SIZE], &w);
    let mlx2full = runtime
        .execute(|| {
            let input =
                Array::from_slice(&host_normed[..2 * HIDDEN_SIZE], &[2, HIDDEN_SIZE as i32]);
            let TokenMixer::Linear(mixer) = &layer.mixer else {
                panic!("linear")
            };
            let (out, _) = mixer.forward(&input, 2, None).expect("mixer 2");
            out
        })
        .expect("execute");
    let mlx2full = read(&mlx2full);
    let mlx8full = runtime
        .execute(|| {
            let input = Array::from_slice(&host_normed, &[ROWS as i32, HIDDEN_SIZE as i32]);
            let TokenMixer::Linear(mixer) = &layer.mixer else {
                panic!("linear")
            };
            let (out, _) = mixer.forward(&input, ROWS, None).expect("mixer 8");
            out
        })
        .expect("execute");
    let mlx8full = read(&mlx8full);
    println!("  host2[0..4] {:?}", &host2[0..4]);
    println!("  mlx2 [0..4] {:?}", &mlx2full[0..4]);
    println!("  mlx8 [0..4] {:?}", &mlx8full[0..4]);
    println!("  host2[2560..2564] {:?}", &host2[2560..2564]);
    println!("  mlx2 [2560..2564] {:?}", &mlx2full[2560..2564]);
    println!("  mlx8 [2560..2564] {:?}", &mlx8full[2560..2564]);
    println!("  host row0[0..4] {:?}", &host_conv[0..4]);
    println!("  mlx  row0[0..4] {:?}", &mlx_conv[0..4]);
    println!("  host row1[0..4] {:?}", &host_conv[QKV_SIZE..QKV_SIZE + 4]);
    println!("  mlx  row1[0..4] {:?}", &mlx_conv[QKV_SIZE..QKV_SIZE + 4]);
    println!("  raw   row0[0..4] {:?}", &raw_values[0..4]);
    println!(
        "  raw   row1[0..4] {:?}",
        &raw_values[QKV_SIZE..QKV_SIZE + 4]
    );
    println!("  w     c0[0..4]  {:?}", &w.conv1d[0..4]);
}

#[test]
fn synthetic_linear_layer_matches_host_reference() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let tensors = synthetic_tensors();
    let layer = MlxDecoderLayer::load(&tensors, 0, MlxPrecision::Fp32).expect("layer");
    let hidden_values = seeded(&[ROWS * HIDDEN_SIZE], 99);
    let hidden = runtime
        .execute(|| Array::from_slice(&hidden_values, &[ROWS as i32, HIDDEN_SIZE as i32]))
        .expect("execute");
    let (mlx_out, _state) = layer.forward(&hidden, ROWS, 0, None).expect("forward");
    let mlx_host = mlx_out
        .eval()
        .ok()
        .and_then(|()| mlx_out.to_vec_cast::<f32>().ok())
        .expect("read");
    let host_out = host_layer(&hidden_values, &host_weights());
    let max_abs = mlx_host
        .iter()
        .zip(&host_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let host_scale = host_out
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    assert!(
        max_abs <= 5e-3 * host_scale,
        "synthetic layer_00 diverged from host reference: max_abs {max_abs:.6} on scale {host_scale:.3}"
    );
}

#[test]
fn gated_delta_softplus_is_finite_at_extremes() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| -> Result<(), MlxError> {
            let qkv = Array::from_slice(&vec![0.0_f32; QKV_SIZE], &[QKV_SIZE as i32]);
            let z = Array::from_slice(&vec![0.0_f32; VALUE_SIZE], &[VALUE_SIZE as i32]);
            let beta = Array::from_slice(&[0.0_f32; VALUE_HEADS], &[VALUE_HEADS as i32]);
            let decay_values: Vec<f32> = (0..VALUE_HEADS)
                .map(|index| if index % 2 == 0 { -100.0 } else { 100.0 })
                .collect();
            let decay = Array::from_slice(&decay_values, &[VALUE_HEADS as i32]);
            let dt_bias = Array::from_slice(&[0.0_f32; VALUE_HEADS], &[VALUE_HEADS as i32]);
            let a_log = Array::from_slice(&[0.0_f32; VALUE_HEADS], &[VALUE_HEADS as i32]);
            let norm = Array::from_slice(&vec![1.0_f32; HEAD_DIM], &[HEAD_DIM as i32]);
            let mut recurrent = zeros(&[VALUE_HEADS as i32, HEAD_DIM as i32, HEAD_DIM as i32]);
            let output = gated_delta_step(
                &qkv,
                &z,
                &beta,
                &decay,
                &dt_bias,
                &a_log,
                &norm,
                &mut recurrent,
            )?;
            output.eval().map_err(op("extreme softplus eval"))?;
            let values = output
                .to_vec_cast::<f32>()
                .map_err(op("extreme softplus read"))?;
            assert!(values.iter().all(|value| value.is_finite()));
            Ok(())
        })
        .expect("execute")
        .expect("extreme softplus");
}
