//! Full attention and grouped-query attention for Qwen 3.5 decoder blocks.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::qwen35::Qwen35Error;

use super::linear_attention::sigmoid;
use super::{
    linear, vector, LayerState, ATTENTION_HEADS, ATTENTION_HEAD_DIM, ATTENTION_SIZE, HIDDEN_SIZE,
    KV_HEADS, KV_SIZE, RMS_EPSILON, ROPE_THETA, ROTARY_DIM,
};

#[derive(Debug)]
pub(crate) struct FullAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out_proj: Tensor,
}

impl FullAttention {
    pub(crate) fn load(variables: &VarBuilder<'_>) -> Result<Self, Qwen35Error> {
        Ok(Self {
            q_proj: variables.get((ATTENTION_SIZE * 2, HIDDEN_SIZE), "self_attn.q_proj.weight")?,
            k_proj: variables.get((KV_SIZE, HIDDEN_SIZE), "self_attn.k_proj.weight")?,
            v_proj: variables.get((KV_SIZE, HIDDEN_SIZE), "self_attn.v_proj.weight")?,
            q_norm: vector(variables, ATTENTION_HEAD_DIM, "self_attn.q_norm.weight")?,
            k_norm: vector(variables, ATTENTION_HEAD_DIM, "self_attn.k_norm.weight")?,
            out_proj: variables.get((HIDDEN_SIZE, ATTENTION_SIZE), "self_attn.o_proj.weight")?,
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
        let projected_q = linear(
            normalized,
            token_count,
            HIDDEN_SIZE,
            &self.q_proj,
            ATTENTION_SIZE * 2,
            device,
        )?;
        let projected_k = linear(
            normalized,
            token_count,
            HIDDEN_SIZE,
            &self.k_proj,
            KV_SIZE,
            device,
        )?;
        let current_values = linear(
            normalized,
            token_count,
            HIDDEN_SIZE,
            &self.v_proj,
            KV_SIZE,
            device,
        )?;
        let (mut queries, gates) = split_query_gate(&projected_q, token_count);
        let mut keys = projected_k;
        rms_norm_heads(&mut queries, ATTENTION_HEADS, &self.q_norm);
        rms_norm_heads(&mut keys, KV_HEADS, &self.k_norm);
        apply_rotary(&mut queries, ATTENTION_HEADS, token_count, position_start);
        apply_rotary(&mut keys, KV_HEADS, token_count, position_start);
        let (previous_keys, previous_values) = previous_state.unzip();
        let past_tokens = previous_keys.map_or(0, |values| values.len() / KV_SIZE);
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
        );
        for (value, gate) in mixed.iter_mut().zip(gates) {
            *value *= sigmoid(gate);
        }
        let output = linear(
            &mixed,
            token_count,
            ATTENTION_SIZE,
            &self.out_proj,
            HIDDEN_SIZE,
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

fn split_query_gate(projected: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(projected.len(), rows * ATTENTION_SIZE * 2);
    let mut query = Vec::with_capacity(rows * ATTENTION_SIZE);
    let mut gate = Vec::with_capacity(rows * ATTENTION_SIZE);
    for row in 0..rows {
        let row_start = row * ATTENTION_SIZE * 2;
        for head in 0..ATTENTION_HEADS {
            let head_start = row_start + head * ATTENTION_HEAD_DIM * 2;
            query.extend_from_slice(&projected[head_start..head_start + ATTENTION_HEAD_DIM]);
            gate.extend_from_slice(
                &projected[head_start + ATTENTION_HEAD_DIM..head_start + ATTENTION_HEAD_DIM * 2],
            );
        }
    }
    (query, gate)
}

fn rms_norm_heads(values: &mut [f32], heads: usize, weight: &[f32]) {
    debug_assert_eq!(values.len() % (heads * ATTENTION_HEAD_DIM), 0);
    debug_assert_eq!(weight.len(), ATTENTION_HEAD_DIM);
    let head_width = weight.len();
    for head in values.chunks_exact_mut(head_width) {
        let variance =
            head.iter().map(|value| value * value).sum::<f32>() / ATTENTION_HEAD_DIM as f32;
        let scale = (variance + RMS_EPSILON).sqrt().recip();
        for (value, weight) in head.iter_mut().zip(weight) {
            *value *= scale * (1.0 + weight);
        }
    }
}

pub(crate) fn apply_rotary(values: &mut [f32], heads: usize, rows: usize, position_start: usize) {
    debug_assert_eq!(values.len(), rows * heads * ATTENTION_HEAD_DIM);
    for row in 0..rows {
        for head in 0..heads {
            let offset = (row * heads + head) * ATTENTION_HEAD_DIM;
            let rotary = &mut values[offset..offset + ROTARY_DIM];
            let original = <[f32; ROTARY_DIM]>::try_from(&*rotary).expect("fixed rotary width");
            for index in 0..ROTARY_DIM / 2 {
                let frequency = ROPE_THETA.powf(-((2 * index) as f32) / ROTARY_DIM as f32);
                let angle = (position_start + row) as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                rotary[index] = original[index] * cos - original[index + ROTARY_DIM / 2] * sin;
                rotary[index + ROTARY_DIM / 2] =
                    original[index + ROTARY_DIM / 2] * cos + original[index] * sin;
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
) -> Vec<f32> {
    debug_assert_eq!(queries.len(), rows * ATTENTION_SIZE);
    debug_assert_eq!(keys.len(), (past_rows + rows) * KV_SIZE);
    debug_assert_eq!(values.len(), (past_rows + rows) * KV_SIZE);
    let mut output = vec![0.0_f32; rows * ATTENTION_SIZE];
    let scale = (ATTENTION_HEAD_DIM as f32).sqrt().recip();
    for row in 0..rows {
        for query_head in 0..ATTENTION_HEADS {
            let kv_head = query_head / (ATTENTION_HEADS / KV_HEADS);
            let query_offset = (row * ATTENTION_HEADS + query_head) * ATTENTION_HEAD_DIM;
            let visible_rows = past_rows + row + 1;
            let mut scores = Vec::with_capacity(visible_rows);
            let mut maximum = f32::NEG_INFINITY;
            for key_row in 0..visible_rows {
                let key_offset = (key_row * KV_HEADS + kv_head) * ATTENTION_HEAD_DIM;
                let score = queries[query_offset..query_offset + ATTENTION_HEAD_DIM]
                    .iter()
                    .zip(&keys[key_offset..key_offset + ATTENTION_HEAD_DIM])
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
            let output_offset = (row * ATTENTION_HEADS + query_head) * ATTENTION_HEAD_DIM;
            for (key_row, score) in scores.into_iter().enumerate() {
                let probability = score / denominator;
                let value_offset = (key_row * KV_HEADS + kv_head) * ATTENTION_HEAD_DIM;
                for column in 0..ATTENTION_HEAD_DIM {
                    output[output_offset + column] += probability * values[value_offset + column];
                }
            }
        }
    }
    output
}
