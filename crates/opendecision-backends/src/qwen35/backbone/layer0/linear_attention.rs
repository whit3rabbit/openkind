//! Linear attention and DeltaNet recurrence for Qwen 3.5 decoder blocks.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::qwen35::Qwen35Error;

use super::{
    linear, tensor_values, vector, LayerState, CONV_KERNEL, HEAD_DIM, HIDDEN_SIZE, KEY_HEADS,
    KEY_SIZE, QKV_SIZE, RMS_EPSILON, VALUE_HEADS, VALUE_SIZE,
};

#[derive(Debug)]
pub(crate) struct LinearAttention {
    in_proj_qkv: Tensor,
    in_proj_z: Tensor,
    in_proj_b: Tensor,
    in_proj_a: Tensor,
    conv1d: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    delta_norm: Vec<f32>,
    out_proj: Tensor,
}

impl LinearAttention {
    pub(crate) fn load(variables: &VarBuilder<'_>) -> Result<Self, Qwen35Error> {
        Ok(Self {
            in_proj_qkv: variables
                .get((QKV_SIZE, HIDDEN_SIZE), "linear_attn.in_proj_qkv.weight")?,
            in_proj_z: variables.get((VALUE_SIZE, HIDDEN_SIZE), "linear_attn.in_proj_z.weight")?,
            in_proj_b: variables.get((VALUE_HEADS, HIDDEN_SIZE), "linear_attn.in_proj_b.weight")?,
            in_proj_a: variables.get((VALUE_HEADS, HIDDEN_SIZE), "linear_attn.in_proj_a.weight")?,
            conv1d: tensor_values(
                &variables.get((QKV_SIZE, 1, CONV_KERNEL), "linear_attn.conv1d.weight")?,
            )?,
            dt_bias: vector(variables, VALUE_HEADS, "linear_attn.dt_bias")?,
            a_log: vector(variables, VALUE_HEADS, "linear_attn.A_log")?,
            delta_norm: vector(variables, HEAD_DIM, "linear_attn.norm.weight")?,
            out_proj: variables.get((HIDDEN_SIZE, VALUE_SIZE), "linear_attn.out_proj.weight")?,
        })
    }

    pub(crate) fn forward_with_state(
        &self,
        normalized: &[f32],
        token_count: usize,
        device: &Device,
        previous_state: Option<(&[f32], &[f32])>,
    ) -> Result<(Vec<f32>, LayerState), Qwen35Error> {
        let raw_qkv = linear(
            normalized,
            token_count,
            HIDDEN_SIZE,
            &self.in_proj_qkv,
            QKV_SIZE,
            device,
        )?;
        let (previous_conv, previous_recurrent) = previous_state.unzip();
        let (qkv, conv) = causal_depthwise_conv_silu_with_state(
            &raw_qkv,
            token_count,
            QKV_SIZE,
            &self.conv1d,
            CONV_KERNEL,
            previous_conv,
        );
        let z = linear(
            normalized,
            token_count,
            HIDDEN_SIZE,
            &self.in_proj_z,
            VALUE_SIZE,
            device,
        )?;
        let beta = linear(
            normalized,
            token_count,
            HIDDEN_SIZE,
            &self.in_proj_b,
            VALUE_HEADS,
            device,
        )?;
        let decay = linear(
            normalized,
            token_count,
            HIDDEN_SIZE,
            &self.in_proj_a,
            VALUE_HEADS,
            device,
        )?;
        let (mixed, recurrent) = gated_delta_recurrent_with_state(
            &qkv,
            &z,
            &beta,
            &decay,
            &self.dt_bias,
            &self.a_log,
            &self.delta_norm,
            token_count,
            previous_recurrent,
        );
        let output = linear(
            &mixed,
            token_count,
            VALUE_SIZE,
            &self.out_proj,
            HIDDEN_SIZE,
            device,
        )?;
        Ok((output, LayerState::Linear { conv, recurrent }))
    }
}

#[cfg(test)]
pub(crate) fn causal_depthwise_conv_silu(
    input: &[f32],
    rows: usize,
    channels: usize,
    weight: &[f32],
    kernel: usize,
) -> Vec<f32> {
    causal_depthwise_conv_silu_with_state(input, rows, channels, weight, kernel, None).0
}

pub(crate) fn causal_depthwise_conv_silu_with_state(
    input: &[f32],
    rows: usize,
    channels: usize,
    weight: &[f32],
    kernel: usize,
    previous_state: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(input.len(), rows * channels);
    debug_assert_eq!(weight.len(), channels * kernel);
    if let Some(previous) = previous_state {
        debug_assert_eq!(previous.len(), channels * kernel);
    }
    let preceding_rows = previous_state.map_or(0, |_| kernel);
    let mut combined = Vec::with_capacity((preceding_rows + rows) * channels);
    if let Some(previous) = previous_state {
        combined.extend_from_slice(previous);
    }
    combined.extend_from_slice(input);
    let mut output = vec![0.0_f32; input.len()];
    let padding = kernel - 1;
    for row in 0..rows {
        let combined_row = preceding_rows + row;
        for channel in 0..channels {
            let mut value = 0.0_f32;
            for tap in 0..kernel {
                if combined_row + tap >= padding {
                    let source_row = combined_row + tap - padding;
                    value +=
                        combined[source_row * channels + channel] * weight[channel * kernel + tap];
                }
            }
            output[row * channels + channel] = silu(value);
        }
    }
    let state_start = combined.len().saturating_sub(channels * kernel);
    let mut state = vec![0.0_f32; channels * kernel];
    let available = &combined[state_start..];
    let destination = state.len() - available.len();
    state[destination..].copy_from_slice(available);
    (output, state)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_recurrent_with_state(
    qkv: &[f32],
    z: &[f32],
    beta_projection: &[f32],
    decay_projection: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    norm_weight: &[f32],
    rows: usize,
    initial_state: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(qkv.len(), rows * QKV_SIZE);
    debug_assert_eq!(z.len(), rows * VALUE_SIZE);
    debug_assert_eq!(beta_projection.len(), rows * VALUE_HEADS);
    debug_assert_eq!(decay_projection.len(), rows * VALUE_HEADS);
    let mut state = initial_state.map_or_else(
        || vec![0.0_f32; VALUE_HEADS * HEAD_DIM * HEAD_DIM],
        <[f32]>::to_vec,
    );
    debug_assert_eq!(state.len(), VALUE_HEADS * HEAD_DIM * HEAD_DIM);
    let mut output = vec![0.0_f32; rows * VALUE_SIZE];
    let query_scale = (HEAD_DIM as f32).sqrt().recip();

    for row in 0..rows {
        let qkv_row = row * QKV_SIZE;
        for value_head in 0..VALUE_HEADS {
            let key_head = value_head / (VALUE_HEADS / KEY_HEADS);
            let query_offset = qkv_row + key_head * HEAD_DIM;
            let key_offset = qkv_row + KEY_SIZE + key_head * HEAD_DIM;
            let value_offset = qkv_row + KEY_SIZE * 2 + value_head * HEAD_DIM;
            let state_offset = value_head * HEAD_DIM * HEAD_DIM;
            let state_head = &mut state[state_offset..state_offset + HEAD_DIM * HEAD_DIM];

            let query_norm = (qkv[query_offset..query_offset + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + RMS_EPSILON)
                .sqrt()
                .recip()
                * query_scale;
            let key_norm = (qkv[key_offset..key_offset + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + RMS_EPSILON)
                .sqrt()
                .recip();
            let beta = sigmoid(beta_projection[row * VALUE_HEADS + value_head]);
            let decay = (-a_log[value_head].exp()
                * softplus(decay_projection[row * VALUE_HEADS + value_head] + dt_bias[value_head]))
            .exp();
            for state in state_head.iter_mut() {
                *state *= decay;
            }

            let mut delta = [0.0_f32; HEAD_DIM];
            for value_index in 0..HEAD_DIM {
                let mut memory = 0.0_f32;
                for key_index in 0..HEAD_DIM {
                    let key = qkv[key_offset + key_index] * key_norm;
                    memory += state_head[key_index * HEAD_DIM + value_index] * key;
                }
                delta[value_index] = (qkv[value_offset + value_index] - memory) * beta;
            }
            for key_index in 0..HEAD_DIM {
                let key = qkv[key_offset + key_index] * key_norm;
                let state_row = &mut state_head[key_index * HEAD_DIM..(key_index + 1) * HEAD_DIM];
                for value_index in 0..HEAD_DIM {
                    state_row[value_index] += key * delta[value_index];
                }
            }

            let output_offset = row * VALUE_SIZE + value_head * HEAD_DIM;
            for value_index in 0..HEAD_DIM {
                let mut value = 0.0_f32;
                for key_index in 0..HEAD_DIM {
                    let query = qkv[query_offset + key_index] * query_norm;
                    value += state_head[key_index * HEAD_DIM + value_index] * query;
                }
                output[output_offset + value_index] = value;
            }
            let variance = output[output_offset..output_offset + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / HEAD_DIM as f32;
            let norm_scale = (variance + RMS_EPSILON).sqrt().recip();
            for value_index in 0..HEAD_DIM {
                let gate = silu(z[output_offset + value_index]);
                output[output_offset + value_index] *= norm_scale * norm_weight[value_index] * gate;
            }
        }
    }
    (output, state)
}

pub(crate) fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(crate) fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

pub(crate) fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}
