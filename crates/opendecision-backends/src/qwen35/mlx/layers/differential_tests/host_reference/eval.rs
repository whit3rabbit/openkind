use crate::qwen35::mlx::layers::*;

use super::ops::*;
use super::weights::*;

pub(crate) fn host_layer(hidden: &[f32], w: &HostLinearWeights) -> Vec<f32> {
    let normed = host_rms_norm(hidden, &w.input_layernorm);
    let raw_qkv = host_matmul(&normed, ROWS, &w.in_proj_qkv, QKV_SIZE);
    let z = host_matmul(&normed, ROWS, &w.in_proj_z, VALUE_SIZE);
    let beta_all = host_matmul(&normed, ROWS, &w.in_proj_b, VALUE_HEADS);
    let decay_all = host_matmul(&normed, ROWS, &w.in_proj_a, VALUE_HEADS);

    // Causal conv window `[C, K]`, oldest tap first, zero-initialized.
    let mut window = vec![0.0_f32; QKV_SIZE * CONV_KERNEL];
    let mut recurrent = vec![0.0_f32; VALUE_HEADS * HEAD_DIM * HEAD_DIM];
    let mut mixed = vec![0.0_f32; ROWS * VALUE_SIZE];
    let query_scale = (HEAD_DIM as f32).sqrt().recip();
    for row in 0..ROWS {
        for c in 0..QKV_SIZE {
            for tap in 0..CONV_KERNEL - 1 {
                window[c * CONV_KERNEL + tap] = window[c * CONV_KERNEL + tap + 1];
            }
            window[c * CONV_KERNEL + (CONV_KERNEL - 1)] = raw_qkv[row * QKV_SIZE + c];
        }
        let mut qkv_row = vec![0.0_f32; QKV_SIZE];
        for c in 0..QKV_SIZE {
            let mut acc = 0.0_f32;
            for tap in 0..CONV_KERNEL {
                acc += window[c * CONV_KERNEL + tap] * w.conv1d[c * CONV_KERNEL + tap];
            }
            qkv_row[c] = silu(acc);
        }

        for value_head in 0..VALUE_HEADS {
            let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
            let query_offset = key_head * HEAD_DIM;
            let key_offset = KEY_SIZE + key_head * HEAD_DIM;
            let value_offset = KEY_SIZE * 2 + value_head * HEAD_DIM;
            let state_offset = value_head * HEAD_DIM * HEAD_DIM;
            let state_head = &mut recurrent[state_offset..state_offset + HEAD_DIM * HEAD_DIM];

            let query_norm = (qkv_row[query_offset..query_offset + HEAD_DIM]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                + RMS_EPSILON)
                .sqrt()
                .recip()
                * query_scale;
            let key_norm = (qkv_row[key_offset..key_offset + HEAD_DIM]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                + RMS_EPSILON)
                .sqrt()
                .recip();
            let beta = sigmoid(beta_all[row * VALUE_HEADS + value_head]);
            let decay = (-w.a_log[value_head].exp()
                * softplus(decay_all[row * VALUE_HEADS + value_head] + w.dt_bias[value_head]))
            .exp();
            for state in state_head.iter_mut() {
                *state *= decay;
            }

            let mut delta = [0.0_f32; HEAD_DIM];
            for value_index in 0..HEAD_DIM {
                let mut memory = 0.0_f32;
                for key_index in 0..HEAD_DIM {
                    memory += state_head[key_index * HEAD_DIM + value_index]
                        * qkv_row[key_offset + key_index]
                        * key_norm;
                }
                delta[value_index] = (qkv_row[value_offset + value_index] - memory) * beta;
            }
            for key_index in 0..HEAD_DIM {
                let key = qkv_row[key_offset + key_index] * key_norm;
                for value_index in 0..HEAD_DIM {
                    state_head[key_index * HEAD_DIM + value_index] += key * delta[value_index];
                }
            }

            let output_offset = row * VALUE_SIZE + value_head * HEAD_DIM;
            for value_index in 0..HEAD_DIM {
                let mut value = 0.0_f32;
                for key_index in 0..HEAD_DIM {
                    value += state_head[key_index * HEAD_DIM + value_index]
                        * qkv_row[query_offset + key_index]
                        * query_norm;
                }
                mixed[output_offset + value_index] = value;
            }
            let variance = mixed[output_offset..output_offset + HEAD_DIM]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                / HEAD_DIM as f32;
            let norm_scale = (variance + RMS_EPSILON).sqrt().recip();
            for value_index in 0..HEAD_DIM {
                let gate = silu(z[row * VALUE_SIZE + value_head * HEAD_DIM + value_index]);
                mixed[output_offset + value_index] *= norm_scale * w.delta_norm[value_index] * gate;
            }
        }
    }

    let attention = host_matmul(&mixed, ROWS, &w.out_proj, HIDDEN_SIZE);
    let mut hidden_out: Vec<f32> = hidden.iter().zip(&attention).map(|(h, a)| h + a).collect();
    let mlp_input = host_rms_norm(&hidden_out, &w.post_attention_layernorm);
    let gate = host_matmul(&mlp_input, ROWS, &w.gate_proj, 9216);
    let up = host_matmul(&mlp_input, ROWS, &w.up_proj, 9216);
    let activated: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    let mlp = host_matmul(&activated, ROWS, &w.down_proj, HIDDEN_SIZE);
    for (h, m) in hidden_out.iter_mut().zip(mlp) {
        *h += m;
    }
    hidden_out
}

pub(crate) fn host_linear_mixer_one(normed: &[f32], w: &HostLinearWeights) -> Vec<f32> {
    let raw_qkv = host_matmul(normed, 1, &w.in_proj_qkv, QKV_SIZE);
    let z = host_matmul(normed, 1, &w.in_proj_z, VALUE_SIZE);
    let beta_all = host_matmul(normed, 1, &w.in_proj_b, VALUE_HEADS);
    let decay_all = host_matmul(normed, 1, &w.in_proj_a, VALUE_HEADS);
    let window = raw_qkv.clone();
    let mut qkv_row = vec![0.0_f32; QKV_SIZE];
    for c in 0..QKV_SIZE {
        let acc = window[c] * w.conv1d[c * CONV_KERNEL + (CONV_KERNEL - 1)];
        qkv_row[c] = silu(acc);
    }
    let (step, _state) = host_recurrence_step_raw(
        &qkv_row,
        &z,
        &beta_all,
        &decay_all,
        &w.dt_bias,
        &w.a_log,
        &w.delta_norm,
        &vec![0.0; VALUE_HEADS * HEAD_DIM * HEAD_DIM],
    );
    host_matmul(&step, 1, &w.out_proj, HIDDEN_SIZE)
}

pub(crate) fn host_recurrence_step(
    qkv_row: &[f32],
    z_row: &[f32],
    beta_row: &[f32],
    decay_row: &[f32],
    w: &HostLinearWeights,
) -> (Vec<f32>, Vec<f32>) {
    let mut recurrent = vec![0.0_f32; VALUE_HEADS * HEAD_DIM * HEAD_DIM];
    let mut mixed = vec![0.0_f32; VALUE_SIZE];
    let query_scale = (HEAD_DIM as f32).sqrt().recip();
    for value_head in 0..VALUE_HEADS {
        let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
        let query_offset = key_head * HEAD_DIM;
        let key_offset = KEY_SIZE + key_head * HEAD_DIM;
        let value_offset = KEY_SIZE * 2 + value_head * HEAD_DIM;
        let state_offset = value_head * HEAD_DIM * HEAD_DIM;
        let state_head = &mut recurrent[state_offset..state_offset + HEAD_DIM * HEAD_DIM];

        let query_norm = (qkv_row[query_offset..query_offset + HEAD_DIM]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            + RMS_EPSILON)
            .sqrt()
            .recip()
            * query_scale;
        let key_norm = (qkv_row[key_offset..key_offset + HEAD_DIM]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            + RMS_EPSILON)
            .sqrt()
            .recip();
        let beta = sigmoid(beta_row[value_head]);
        let decay = (-w.a_log[value_head].exp()
            * softplus(decay_row[value_head] + w.dt_bias[value_head]))
        .exp();
        for state in state_head.iter_mut() {
            *state *= decay;
        }
        let mut delta = [0.0_f32; HEAD_DIM];
        for value_index in 0..HEAD_DIM {
            let mut memory = 0.0_f32;
            for key_index in 0..HEAD_DIM {
                memory += state_head[key_index * HEAD_DIM + value_index]
                    * qkv_row[key_offset + key_index]
                    * key_norm;
            }
            delta[value_index] = (qkv_row[value_offset + value_index] - memory) * beta;
        }
        for key_index in 0..HEAD_DIM {
            let key = qkv_row[key_offset + key_index] * key_norm;
            for value_index in 0..HEAD_DIM {
                state_head[key_index * HEAD_DIM + value_index] += key * delta[value_index];
            }
        }
        let output_offset = value_head * HEAD_DIM;
        for value_index in 0..HEAD_DIM {
            let mut value = 0.0_f32;
            for key_index in 0..HEAD_DIM {
                value += state_head[key_index * HEAD_DIM + value_index]
                    * qkv_row[query_offset + key_index]
                    * query_norm;
            }
            mixed[output_offset + value_index] = value;
        }
        let variance = mixed[output_offset..output_offset + HEAD_DIM]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / HEAD_DIM as f32;
        let norm_scale = (variance + RMS_EPSILON).sqrt().recip();
        for value_index in 0..HEAD_DIM {
            let gate = silu(z_row[value_head * HEAD_DIM + value_index]);
            mixed[output_offset + value_index] *= norm_scale * w.delta_norm[value_index] * gate;
        }
    }
    (mixed, recurrent)
}

pub(crate) fn host_linear_mixer(normed: &[f32], w: &HostLinearWeights) -> Vec<f32> {
    let rows = normed.len() / HIDDEN_SIZE;
    let raw_qkv = host_matmul(normed, rows, &w.in_proj_qkv, QKV_SIZE);
    let z = host_matmul(normed, rows, &w.in_proj_z, VALUE_SIZE);
    let beta_all = host_matmul(normed, rows, &w.in_proj_b, VALUE_HEADS);
    let decay_all = host_matmul(normed, rows, &w.in_proj_a, VALUE_HEADS);
    let mut window = vec![0.0_f32; QKV_SIZE * CONV_KERNEL];
    let mut recurrent = vec![0.0_f32; VALUE_HEADS * HEAD_DIM * HEAD_DIM];
    let mut mixed = vec![0.0_f32; rows * VALUE_SIZE];
    for row in 0..rows {
        for c in 0..QKV_SIZE {
            for tap in 0..CONV_KERNEL - 1 {
                window[c * CONV_KERNEL + tap] = window[c * CONV_KERNEL + tap + 1];
            }
            window[c * CONV_KERNEL + (CONV_KERNEL - 1)] = raw_qkv[row * QKV_SIZE + c];
        }
        let mut qkv_row = vec![0.0_f32; QKV_SIZE];
        for c in 0..QKV_SIZE {
            let mut acc = 0.0_f32;
            for tap in 0..CONV_KERNEL {
                acc += window[c * CONV_KERNEL + tap] * w.conv1d[c * CONV_KERNEL + tap];
            }
            qkv_row[c] = silu(acc);
        }
        let (step, state) = host_recurrence_step_raw(
            &qkv_row,
            &z[row * VALUE_SIZE..(row + 1) * VALUE_SIZE],
            &beta_all[row * VALUE_HEADS..(row + 1) * VALUE_HEADS],
            &decay_all[row * VALUE_HEADS..(row + 1) * VALUE_HEADS],
            &w.dt_bias,
            &w.a_log,
            &w.delta_norm,
            &recurrent,
        );
        mixed[row * VALUE_SIZE..(row + 1) * VALUE_SIZE].copy_from_slice(&step);
        recurrent = state;
    }
    host_matmul(&mixed, rows, &w.out_proj, HIDDEN_SIZE)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn host_recurrence_step_raw(
    qkv_row: &[f32],
    z_row: &[f32],
    beta_row: &[f32],
    decay_row: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    delta_norm: &[f32],
    initial: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut recurrent = initial.to_vec();
    let mut mixed = vec![0.0_f32; VALUE_SIZE];
    let query_scale = (HEAD_DIM as f32).sqrt().recip();
    for value_head in 0..VALUE_HEADS {
        let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
        let query_offset = key_head * HEAD_DIM;
        let key_offset = KEY_SIZE + key_head * HEAD_DIM;
        let value_offset = KEY_SIZE * 2 + value_head * HEAD_DIM;
        let state_offset = value_head * HEAD_DIM * HEAD_DIM;
        let state_head = &mut recurrent[state_offset..state_offset + HEAD_DIM * HEAD_DIM];
        let query_norm = (qkv_row[query_offset..query_offset + HEAD_DIM]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            + RMS_EPSILON)
            .sqrt()
            .recip()
            * query_scale;
        let key_norm = (qkv_row[key_offset..key_offset + HEAD_DIM]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            + RMS_EPSILON)
            .sqrt()
            .recip();
        let beta = sigmoid(beta_row[value_head]);
        let decay = (-a_log[value_head].exp()
            * softplus(decay_row[value_head] + dt_bias[value_head]))
        .exp();
        for state in state_head.iter_mut() {
            *state *= decay;
        }
        let mut delta = [0.0_f32; HEAD_DIM];
        for value_index in 0..HEAD_DIM {
            let mut memory = 0.0_f32;
            for key_index in 0..HEAD_DIM {
                memory += state_head[key_index * HEAD_DIM + value_index]
                    * qkv_row[key_offset + key_index]
                    * key_norm;
            }
            delta[value_index] = (qkv_row[value_offset + value_index] - memory) * beta;
        }
        for key_index in 0..HEAD_DIM {
            let key = qkv_row[key_offset + key_index] * key_norm;
            for value_index in 0..HEAD_DIM {
                state_head[key_index * HEAD_DIM + value_index] += key * delta[value_index];
            }
        }
        let output_offset = value_head * HEAD_DIM;
        for value_index in 0..HEAD_DIM {
            let mut value = 0.0_f32;
            for key_index in 0..HEAD_DIM {
                value += state_head[key_index * HEAD_DIM + value_index]
                    * qkv_row[query_offset + key_index]
                    * query_norm;
            }
            mixed[output_offset + value_index] = value;
        }
        let variance = mixed[output_offset..output_offset + HEAD_DIM]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / HEAD_DIM as f32;
        let norm_scale = (variance + RMS_EPSILON).sqrt().recip();
        for value_index in 0..HEAD_DIM {
            let gate = silu(z_row[value_head * HEAD_DIM + value_index]);
            mixed[output_offset + value_index] *= norm_scale * delta_norm[value_index] * gate;
        }
    }
    (mixed, recurrent)
}
