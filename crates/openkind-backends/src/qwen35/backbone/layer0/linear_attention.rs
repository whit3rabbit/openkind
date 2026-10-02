//! Linear attention and DeltaNet recurrence for Qwen 3.5 decoder blocks.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::qwen35::backbone::geometry::Qwen35Geometry;
use crate::qwen35::Qwen35Error;

use super::{linear, tensor_values, vector, LayerState, RMS_EPSILON};

#[derive(Debug)]
pub(crate) struct LinearAttention {
    geometry: Qwen35Geometry,
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
    pub(crate) fn load(
        variables: &VarBuilder<'_>,
        geometry: Qwen35Geometry,
    ) -> Result<Self, Qwen35Error> {
        Ok(Self {
            geometry,
            in_proj_qkv: variables.get(
                (geometry.qkv_size(), geometry.hidden_size),
                "linear_attn.in_proj_qkv.weight",
            )?,
            in_proj_z: variables.get(
                (geometry.value_size(), geometry.hidden_size),
                "linear_attn.in_proj_z.weight",
            )?,
            in_proj_b: variables.get(
                (geometry.value_heads, geometry.hidden_size),
                "linear_attn.in_proj_b.weight",
            )?,
            in_proj_a: variables.get(
                (geometry.value_heads, geometry.hidden_size),
                "linear_attn.in_proj_a.weight",
            )?,
            conv1d: tensor_values(&variables.get(
                (geometry.qkv_size(), 1, geometry.conv_kernel),
                "linear_attn.conv1d.weight",
            )?)?,
            dt_bias: vector(variables, geometry.value_heads, "linear_attn.dt_bias")?,
            a_log: vector(variables, geometry.value_heads, "linear_attn.A_log")?,
            delta_norm: vector(variables, geometry.head_dim, "linear_attn.norm.weight")?,
            out_proj: variables.get(
                (geometry.hidden_size, geometry.value_size()),
                "linear_attn.out_proj.weight",
            )?,
        })
    }

    pub(crate) fn forward_with_state(
        &self,
        normalized: &[f32],
        token_count: usize,
        device: &Device,
        previous_state: Option<(&[f32], &[f32])>,
    ) -> Result<(Vec<f32>, LayerState), Qwen35Error> {
        let geometry = self.geometry;
        let raw_qkv = linear(
            normalized,
            token_count,
            geometry.hidden_size,
            &self.in_proj_qkv,
            geometry.qkv_size(),
            device,
        )?;
        let (previous_conv, previous_recurrent) = previous_state.unzip();
        let (qkv, conv) = causal_depthwise_conv_silu_with_state(
            &raw_qkv,
            token_count,
            geometry.qkv_size(),
            &self.conv1d,
            geometry.conv_kernel,
            previous_conv,
        );
        let z = linear(
            normalized,
            token_count,
            geometry.hidden_size,
            &self.in_proj_z,
            geometry.value_size(),
            device,
        )?;
        let beta = linear(
            normalized,
            token_count,
            geometry.hidden_size,
            &self.in_proj_b,
            geometry.value_heads,
            device,
        )?;
        let decay = linear(
            normalized,
            token_count,
            geometry.hidden_size,
            &self.in_proj_a,
            geometry.value_heads,
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
            geometry,
            previous_recurrent,
        );
        let output = linear(
            &mixed,
            token_count,
            geometry.value_size(),
            &self.out_proj,
            geometry.hidden_size,
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
    geometry: Qwen35Geometry,
    initial_state: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let value_heads = geometry.value_heads;
    let head_dim = geometry.head_dim;
    let key_heads = geometry.key_heads;
    let qkv_size = geometry.qkv_size();
    let value_size = geometry.value_size();
    let key_size = geometry.key_size();
    debug_assert_eq!(qkv.len(), rows * qkv_size);
    debug_assert_eq!(z.len(), rows * value_size);
    debug_assert_eq!(beta_projection.len(), rows * value_heads);
    debug_assert_eq!(decay_projection.len(), rows * value_heads);
    let mut state = initial_state.map_or_else(
        || vec![0.0_f32; value_heads * head_dim * head_dim],
        <[f32]>::to_vec,
    );
    debug_assert_eq!(state.len(), value_heads * head_dim * head_dim);
    let mut output = vec![0.0_f32; rows * value_size];
    let query_scale = (head_dim as f32).sqrt().recip();

    for row in 0..rows {
        let qkv_row = row * qkv_size;
        for value_head in 0..value_heads {
            let key_head = value_head / (value_heads / key_heads);
            let query_offset = qkv_row + key_head * head_dim;
            let key_offset = qkv_row + key_size + key_head * head_dim;
            let value_offset = qkv_row + key_size * 2 + value_head * head_dim;
            let state_offset = value_head * head_dim * head_dim;
            let state_head = &mut state[state_offset..state_offset + head_dim * head_dim];

            let query_norm = (qkv[query_offset..query_offset + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + RMS_EPSILON)
                .sqrt()
                .recip()
                * query_scale;
            let key_norm = (qkv[key_offset..key_offset + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + RMS_EPSILON)
                .sqrt()
                .recip();
            let beta = sigmoid(beta_projection[row * value_heads + value_head]);
            let decay = (-a_log[value_head].exp()
                * softplus(decay_projection[row * value_heads + value_head] + dt_bias[value_head]))
            .exp();
            for state in state_head.iter_mut() {
                *state *= decay;
            }

            let mut delta = vec![0.0_f32; head_dim];
            for value_index in 0..head_dim {
                let mut memory = 0.0_f32;
                for key_index in 0..head_dim {
                    let key = qkv[key_offset + key_index] * key_norm;
                    memory += state_head[key_index * head_dim + value_index] * key;
                }
                delta[value_index] = (qkv[value_offset + value_index] - memory) * beta;
            }
            for key_index in 0..head_dim {
                let key = qkv[key_offset + key_index] * key_norm;
                let state_row = &mut state_head[key_index * head_dim..(key_index + 1) * head_dim];
                for value_index in 0..head_dim {
                    state_row[value_index] += key * delta[value_index];
                }
            }

            let output_offset = row * value_size + value_head * head_dim;
            for value_index in 0..head_dim {
                let mut value = 0.0_f32;
                for key_index in 0..head_dim {
                    let query = qkv[query_offset + key_index] * query_norm;
                    value += state_head[key_index * head_dim + value_index] * query;
                }
                output[output_offset + value_index] = value;
            }
            let variance = output[output_offset..output_offset + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / head_dim as f32;
            let norm_scale = (variance + RMS_EPSILON).sqrt().recip();
            for value_index in 0..head_dim {
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
