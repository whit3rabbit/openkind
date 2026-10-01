//! Full attention and grouped-query attention for Qwen 3.5 decoder blocks.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::qwen35::Qwen35Error;
use crate::qwen35::backbone::geometry::Qwen35Geometry;

use super::linear_attention::sigmoid;
use super::{linear, vector, LayerState};

#[derive(Debug)]
pub(crate) struct FullAttention {
    geometry: Qwen35Geometry,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out_proj: Tensor,
}

impl FullAttention {
    pub(crate) fn load(
        variables: &VarBuilder<'_>,
        geometry: Qwen35Geometry,
    ) -> Result<Self, Qwen35Error> {
        Ok(Self {
            geometry,
            q_proj: variables.get(
                (geometry.attention_size() * 2, geometry.hidden_size),
                "self_attn.q_proj.weight",
            )?,
            k_proj: variables.get(
                (geometry.kv_size(), geometry.hidden_size),
                "self_attn.k_proj.weight",
            )?,
            v_proj: variables.get(
                (geometry.kv_size(), geometry.hidden_size),
                "self_attn.v_proj.weight",
            )?,
            q_norm: vector(
                variables,
                geometry.attention_head_dim,
                "self_attn.q_norm.weight",
            )?,
            k_norm: vector(
                variables,
                geometry.attention_head_dim,
                "self_attn.k_norm.weight",
            )?,
            out_proj: variables.get(
                (geometry.hidden_size, geometry.attention_size()),
                "self_attn.o_proj.weight",
            )?,
        })
    }

    pub(crate) fn forward_with_state(
        &self,
        normalized: &[f32],
        token_count: usize,
        position_start: usize,
        device: &Device,
        previous_state: Option<(&[f32], &[f32])>,
    ) -> Result<(Vec<f32>, LayerState), Qwen35Error> {
        let geometry = self.geometry;
        let projected_q = linear(
            normalized,
            token_count,
            geometry.hidden_size,
            &self.q_proj,
            geometry.attention_size() * 2,
            device,
        )?;
        let projected_k = linear(
            normalized,
            token_count,
            geometry.hidden_size,
            &self.k_proj,
            geometry.kv_size(),
            device,
        )?;
        let current_values = linear(
            normalized,
            token_count,
            geometry.hidden_size,
            &self.v_proj,
            geometry.kv_size(),
            device,
        )?;
        let (mut queries, gates) = split_query_gate(&projected_q, token_count, geometry);
        let mut keys = projected_k;
        rms_norm_heads(&mut queries, geometry.attention_heads, &self.q_norm);
        rms_norm_heads(&mut keys, geometry.kv_heads, &self.k_norm);
        apply_rotary(
            &mut queries,
            geometry.attention_heads,
            token_count,
            position_start,
            geometry,
        );
        apply_rotary(
            &mut keys,
            geometry.kv_heads,
            token_count,
            position_start,
            geometry,
        );
        let (previous_keys, previous_values) = previous_state.unzip();
        let kv_size = geometry.kv_size();
        let past_tokens = previous_keys.map_or(0, |values| values.len() / kv_size);
        if past_tokens != position_start {
            return Err(Qwen35Error::InvalidInput(format!(
                "attention cache has {past_tokens} positions, continuation starts at {position_start}"
            )));
        }
        let mut all_keys = previous_keys.map_or_else(Vec::new, <[f32]>::to_vec);
        all_keys.extend_from_slice(&keys);
        let mut all_values = previous_values.map_or_else(Vec::new, <[f32]>::to_vec);
        all_values.extend_from_slice(&current_values);
        let mut mixed = causal_grouped_query_attention(
            &queries,
            &all_keys,
            &all_values,
            token_count,
            past_tokens,
            geometry,
        );
        for (value, gate) in mixed.iter_mut().zip(gates) {
            *value *= sigmoid(gate);
        }
        let output = linear(
            &mixed,
            token_count,
            geometry.attention_size(),
            &self.out_proj,
            geometry.hidden_size,
            device,
        )?;
        Ok((
            output,
            LayerState::Full {
                keys: all_keys,
                values: all_values,
            },
        ))
    }
}

pub(crate) fn split_query_gate(
    projected: &[f32],
    rows: usize,
    geometry: Qwen35Geometry,
) -> (Vec<f32>, Vec<f32>) {
    let attention_heads = geometry.attention_heads;
    let attention_head_dim = geometry.attention_head_dim;
    let attention_size = geometry.attention_size();
    debug_assert_eq!(projected.len(), rows * attention_size * 2);
    let mut query = Vec::with_capacity(rows * attention_size);
    let mut gate = Vec::with_capacity(rows * attention_size);
    for row in 0..rows {
        let row_start = row * attention_size * 2;
        for head in 0..attention_heads {
            let head_start = row_start + head * attention_head_dim * 2;
            query.extend_from_slice(&projected[head_start..head_start + attention_head_dim]);
            gate.extend_from_slice(
                &projected[head_start + attention_head_dim
                    ..head_start + attention_head_dim * 2],
            );
        }
    }
    (query, gate)
}

pub(crate) fn rms_norm_heads(values: &mut [f32], heads: usize, weight: &[f32]) {
    let head_width = weight.len();
    debug_assert_eq!(values.len() % (heads * head_width), 0);
    for head in values.chunks_exact_mut(head_width) {
        let variance = head.iter().map(|value| value * value).sum::<f32>() / head_width as f32;
        let scale = (variance + 1e-6).sqrt().recip();
        for (value, weight) in head.iter_mut().zip(weight) {
            *value *= scale * (1.0 + weight);
        }
    }
}

pub(crate) fn apply_rotary(
    values: &mut [f32],
    heads: usize,
    rows: usize,
    position_start: usize,
    geometry: Qwen35Geometry,
) {
    let rotary_dim = geometry.rotary_dim();
    let attention_head_dim = geometry.attention_head_dim;
    let rope_theta = Qwen35Geometry::ROPE_THETA;
    debug_assert_eq!(values.len(), rows * heads * attention_head_dim);
    for row in 0..rows {
        for head in 0..heads {
            let offset = (row * heads + head) * attention_head_dim;
            let rotary = &mut values[offset..offset + rotary_dim];
            let original: Vec<f32> = rotary.to_vec();
            for index in 0..rotary_dim / 2 {
                let frequency = rope_theta.powf(-((2 * index) as f32) / rotary_dim as f32);
                let angle = (position_start + row) as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                rotary[index] = original[index] * cos - original[index + rotary_dim / 2] * sin;
                rotary[index + rotary_dim / 2] =
                    original[index + rotary_dim / 2] * cos + original[index] * sin;
            }
        }
    }
}

pub(crate) fn causal_grouped_query_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    past_rows: usize,
    geometry: Qwen35Geometry,
) -> Vec<f32> {
    let attention_heads = geometry.attention_heads;
    let kv_heads = geometry.kv_heads;
    let attention_head_dim = geometry.attention_head_dim;
    let attention_size = geometry.attention_size();
    let kv_size = geometry.kv_size();
    debug_assert_eq!(queries.len(), rows * attention_size);
    debug_assert_eq!(keys.len(), (past_rows + rows) * kv_size);
    debug_assert_eq!(values.len(), (past_rows + rows) * kv_size);
    let mut output = vec![0.0_f32; rows * attention_size];
    let scale = (attention_head_dim as f32).sqrt().recip();
    for row in 0..rows {
        for query_head in 0..attention_heads {
            let kv_head = query_head / (attention_heads / kv_heads);
            let query_offset = (row * attention_heads + query_head) * attention_head_dim;
            let visible_rows = past_rows + row + 1;
            let mut scores = Vec::with_capacity(visible_rows);
            let mut maximum = f32::NEG_INFINITY;
            for key_row in 0..visible_rows {
                let key_offset = (key_row * kv_heads + kv_head) * attention_head_dim;
                let score = queries[query_offset..query_offset + attention_head_dim]
                    .iter()
                    .zip(&keys[key_offset..key_offset + attention_head_dim])
                    .map(|(query, key)| query * key)
                    .sum::<f32>()
                    * scale;
                maximum = maximum.max(score);
                scores.push(score);
            }
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            let output_offset = (row * attention_heads + query_head) * attention_head_dim;
            for (key_row, score) in scores.into_iter().enumerate() {
                let probability = score / denominator;
                let value_offset = (key_row * kv_heads + kv_head) * attention_head_dim;
                for column in 0..attention_head_dim {
                    output[output_offset + column] += probability * values[value_offset + column];
                }
            }
        }
    }
    output
}
