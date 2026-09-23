use std::ops::{Add, Div, Mul, Sub};

use mlx_rs::fast;
use mlx_rs::ops::{broadcast_to, concatenate};
use mlx_rs::{Array, Dtype};

use super::ops::{batch_row_of, linear, op, scalar_like, stack_batch_time};
use super::{
    MlxError, MlxFullState, MlxLayerState, ATTENTION_HEADS, ATTENTION_HEAD_DIM, ATTENTION_SIZE,
    KV_HEADS, RMS_EPSILON, ROPE_THETA, ROTARY_DIM,
};

pub(crate) struct FullAttention {
    pub(crate) q_proj: Array,
    pub(crate) k_proj: Array,
    pub(crate) v_proj: Array,
    pub(crate) q_norm: Array,
    pub(crate) k_norm: Array,
    pub(crate) o_proj: Array,
}

impl FullAttention {
    pub(crate) fn forward(
        &self,
        normalized: &Array,
        rows: usize,
        position_start: usize,
        previous: Option<&MlxFullState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let projected_q = linear(normalized, &self.q_proj)?;
        let projected_k = linear(normalized, &self.k_proj)?;
        let projected_v = linear(normalized, &self.v_proj)?;

        // Split the interleaved per-head [q | gate] halves.
        let reshaped = projected_q
            .reshape(&[
                rows as i32,
                ATTENTION_HEADS as i32,
                2,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("qproj reshape"))?;
        let halves = reshaped.split_at_indices(&[1], 2).map_err(op("q split"))?;
        let mut queries = halves[0]
            .clone()
            .reshape(&[
                rows as i32,
                ATTENTION_HEADS as i32,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("q view"))?;
        let gates = halves[1]
            .clone()
            .reshape(&[
                rows as i32,
                ATTENTION_HEADS as i32,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("gate view"))?;

        let mut keys = projected_k
            .reshape(&[rows as i32, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32])
            .map_err(op("k reshape"))?;
        let values = projected_v
            .reshape(&[rows as i32, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32])
            .map_err(op("v reshape"))?;

        // Per-head offset-RMSNorm (weights folded at load) on q and k.
        queries = norm_heads(&queries, &self.q_norm)?;
        keys = norm_heads(&keys, &self.k_norm)?;

        // Partial rotary embedding over the first ROTARY_DIM dims, NeoX
        // pairing, absolute positions position_start + row.
        queries = apply_rotary(&queries, rows, position_start, ATTENTION_HEADS)?;
        keys = apply_rotary(&keys, rows, position_start, KV_HEADS)?;

        let (all_keys, all_values) = match previous {
            Some(state) => (
                concatenate(&[&state.keys, &keys], 0).map_err(op("keys concat"))?,
                concatenate(&[&state.values, &values], 0).map_err(op("values concat"))?,
            ),
            None => (keys, values),
        };

        let mut mixed =
            grouped_query_attention(&queries, &all_keys, &all_values, rows, position_start)?;
        mixed = mixed
            .reshape(&[
                rows as i32,
                ATTENTION_HEADS as i32,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("mixed view"))?
            .mul(&mlx_rs::nn::sigmoid(&gates).map_err(op("gate sigmoid"))?);
        mixed = mixed
            .reshape(&[rows as i32, ATTENTION_SIZE as i32])
            .map_err(op("mixed flatten"))?;
        let output = linear(&mixed, &self.o_proj)?;
        Ok((
            output,
            MlxLayerState::Full(MlxFullState {
                keys: all_keys,
                values: all_values,
            }),
        ))
    }

    /// Causal multi-lane attention over right-padded suffixes. The cache is
    /// rectangular during the forward, then the model adapter truncates each
    /// lane back to its true length before returning state.
    pub(crate) fn forward_batched(
        &self,
        normalized: &Array,
        rows: usize,
        position_start: usize,
        lengths: &[usize],
        previous: &MlxFullState,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let shape = normalized.shape();
        if shape.len() != 3 || shape[1] as usize != rows {
            return Err(MlxError::InvalidState(format!(
                "invalid batched attention input shape {shape:?}"
            )));
        }
        let lanes = shape[0] as usize;
        if lengths.len() != lanes || lengths.iter().any(|length| *length == 0 || *length > rows) {
            return Err(MlxError::InvalidState(
                "invalid true lengths for batched attention".to_owned(),
            ));
        }
        let expected_prefix = [
            lanes as i32,
            position_start as i32,
            KV_HEADS as i32,
            ATTENTION_HEAD_DIM as i32,
        ];
        if previous.keys.shape() != expected_prefix || previous.values.shape() != expected_prefix {
            return Err(MlxError::InvalidState(
                "invalid batched attention cache shape".to_owned(),
            ));
        }

        let projected_q = linear(normalized, &self.q_proj)?;
        let projected_k = linear(normalized, &self.k_proj)?;
        let projected_v = linear(normalized, &self.v_proj)?;
        let reshaped = projected_q
            .reshape(&[
                lanes as i32,
                rows as i32,
                ATTENTION_HEADS as i32,
                2,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("batched qproj reshape"))?;
        let halves = reshaped
            .split_at_indices(&[1], 3)
            .map_err(op("batched q split"))?;
        let queries = halves[0]
            .clone()
            .reshape(&[
                lanes as i32,
                rows as i32,
                ATTENTION_HEADS as i32,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("batched q view"))?;
        let gates = halves[1]
            .clone()
            .reshape(&[
                lanes as i32,
                rows as i32,
                ATTENTION_HEADS as i32,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("batched gate view"))?;
        let keys = projected_k
            .reshape(&[
                lanes as i32,
                rows as i32,
                (KV_HEADS * ATTENTION_HEAD_DIM) as i32,
            ])
            .map_err(op("batched k reshape"))?;
        let values = projected_v
            .reshape(&[
                lanes as i32,
                rows as i32,
                (KV_HEADS * ATTENTION_HEAD_DIM) as i32,
            ])
            .map_err(op("batched v reshape"))?;

        let mut all_keys = previous.keys.clone();
        let mut all_values = previous.values.clone();
        let mut mixed_rows = Vec::with_capacity(rows);
        for row in 0..rows {
            let query = batch_row_of(
                &queries
                    .reshape(&[
                        lanes as i32,
                        rows as i32,
                        (ATTENTION_HEADS * ATTENTION_HEAD_DIM) as i32,
                    ])
                    .map_err(op("batched q flatten"))?,
                row,
            )?
            .reshape(&[
                lanes as i32,
                ATTENTION_HEADS as i32,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("batched q token reshape"))?;
            let gate = batch_row_of(
                &gates
                    .reshape(&[
                        lanes as i32,
                        rows as i32,
                        (ATTENTION_HEADS * ATTENTION_HEAD_DIM) as i32,
                    ])
                    .map_err(op("batched gate flatten"))?,
                row,
            )?
            .reshape(&[
                lanes as i32,
                ATTENTION_HEADS as i32,
                ATTENTION_HEAD_DIM as i32,
            ])
            .map_err(op("batched gate token reshape"))?;
            let mut query = norm_heads(&query, &self.q_norm)?;
            let mut key = batch_row_of(&keys, row)?
                .reshape(&[lanes as i32, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32])
                .map_err(op("batched k token reshape"))?;
            let value = batch_row_of(&values, row)?
                .reshape(&[lanes as i32, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32])
                .map_err(op("batched v token reshape"))?;
            key = norm_heads(&key, &self.k_norm)?;
            query = apply_rotary_batch_position(&query, position_start + row, ATTENTION_HEADS)?;
            key = apply_rotary_batch_position(&key, position_start + row, KV_HEADS)?;

            let key = key
                .reshape(&[lanes as i32, 1, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32])
                .map_err(op("batched cache key reshape"))?;
            let value = value
                .reshape(&[lanes as i32, 1, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32])
                .map_err(op("batched cache value reshape"))?;
            all_keys = concatenate(&[&all_keys, &key], 1).map_err(op("batched keys concat"))?;
            all_values =
                concatenate(&[&all_values, &value], 1).map_err(op("batched values concat"))?;

            let mut mixed = grouped_query_attention_batched(
                &query,
                &all_keys,
                &all_values,
                lengths,
                position_start,
                row,
            )?;
            mixed = mixed.mul(&mlx_rs::nn::sigmoid(&gate).map_err(op("batched gate sigmoid"))?);
            mixed = mixed
                .reshape(&[lanes as i32, ATTENTION_SIZE as i32])
                .map_err(op("batched mixed flatten"))?;
            mixed_rows.push(linear(&mixed, &self.o_proj)?);
        }
        Ok((
            stack_batch_time(&mixed_rows)?,
            MlxLayerState::Full(MlxFullState {
                keys: all_keys,
                values: all_values,
            }),
        ))
    }
}

fn grouped_query_attention_batched(
    queries: &Array,
    keys: &Array,
    values: &Array,
    lengths: &[usize],
    position_start: usize,
    row: usize,
) -> Result<Array, MlxError> {
    let lanes = queries.shape()[0];
    let total = keys.shape()[1];
    let repeat = ATTENTION_HEADS / KV_HEADS;
    let expand = |kv: &Array| -> Result<Array, MlxError> {
        let reshaped = kv
            .clone()
            .reshape(&[lanes, total, KV_HEADS as i32, 1, ATTENTION_HEAD_DIM as i32])
            .map_err(op("batched kv reshape"))?;
        broadcast_to(
            &reshaped,
            &[
                lanes,
                total,
                KV_HEADS as i32,
                repeat as i32,
                ATTENTION_HEAD_DIM as i32,
            ],
        )
        .map_err(op("batched kv broadcast"))?
        .reshape(&[
            lanes,
            total,
            ATTENTION_HEADS as i32,
            ATTENTION_HEAD_DIM as i32,
        ])
        .map_err(op("batched kv flatten"))
    };
    let k = expand(keys)?
        .transpose_axes(&[0, 2, 3, 1])
        .map_err(op("batched k layout"))?;
    let v = expand(values)?
        .transpose_axes(&[0, 2, 1, 3])
        .map_err(op("batched v layout"))?;
    let q = queries
        .clone()
        .reshape(&[lanes, ATTENTION_HEADS as i32, 1, ATTENTION_HEAD_DIM as i32])
        .map_err(op("batched q layout"))?;
    let mut scores = q
        .matmul(&k)
        .map_err(op("batched scores matmul"))?
        .mul(scalar_like(
            queries,
            (ATTENTION_HEAD_DIM as f32).sqrt().recip(),
        ));
    let total = total as usize;
    let mask = batched_key_padding_mask(lengths, position_start, row, total)?;
    let mask = Array::from_slice(&mask, &[lanes, 1, 1, total as i32]);
    scores = scores.add(&mask);
    let maximum = scores
        .max_axis(-1, true)
        .map_err(op("batched softmax max"))?;
    let exp = scores
        .sub(&maximum)
        .exp()
        .map_err(op("batched softmax exp"))?;
    let denominator = exp.sum_axis(-1, true).map_err(op("batched softmax sum"))?;
    let probabilities = exp.div(&denominator);
    probabilities
        .matmul(&v)
        .map_err(op("batched mix matmul"))?
        .reshape(&[lanes, ATTENTION_HEADS as i32, ATTENTION_HEAD_DIM as i32])
        .map_err(op("batched mixed reshape"))
}

fn batched_key_padding_mask(
    lengths: &[usize],
    position_start: usize,
    row: usize,
    total_keys: usize,
) -> Result<Vec<f32>, MlxError> {
    if lengths.is_empty() || lengths.contains(&0) {
        return Err(MlxError::InvalidState(
            "attention padding mask requires non-empty lane lengths".to_owned(),
        ));
    }
    let mut mask = Vec::with_capacity(lengths.len().saturating_mul(total_keys));
    for length in lengths {
        let visible = (position_start + row + 1).min(position_start + length);
        for column in 0..total_keys {
            mask.push(if column < visible {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Ok(mask)
}

fn apply_rotary_batch_position(
    values: &Array,
    position: usize,
    heads: usize,
) -> Result<Array, MlxError> {
    let lanes = values.shape()[0] as usize;
    let rot = ROTARY_DIM as i32;
    let half = (ROTARY_DIM / 2) as i32;
    let reshaped = values
        .clone()
        .reshape(&[lanes as i32, heads as i32, 4, rot])
        .map_err(op("batched rope reshape"))?;
    let parts = reshaped
        .split_at_indices(&[1], 2)
        .map_err(op("batched rope split"))?;
    let first = parts[0]
        .clone()
        .reshape(&[lanes as i32, heads as i32, rot])
        .map_err(op("batched rope first"))?;
    let rest = parts[1]
        .clone()
        .reshape(&[lanes as i32, heads as i32, 3 * rot])
        .map_err(op("batched rope rest"))?;
    let pairs = first
        .reshape(&[lanes as i32, heads as i32, 2, half])
        .map_err(op("batched rope pairs"))?;
    let pair_split = pairs
        .split_at_indices(&[1], 2)
        .map_err(op("batched rope pair split"))?;
    let x_lo = pair_split[0]
        .clone()
        .reshape(&[lanes as i32, heads as i32, half])
        .map_err(op("batched rope lo"))?;
    let x_hi = pair_split[1]
        .clone()
        .reshape(&[lanes as i32, heads as i32, half])
        .map_err(op("batched rope hi"))?;
    let mut cos = Vec::with_capacity(ROTARY_DIM / 2);
    let mut sin = Vec::with_capacity(ROTARY_DIM / 2);
    for index in 0..ROTARY_DIM / 2 {
        let frequency = ROPE_THETA.powf(-((2 * index) as f32) / ROTARY_DIM as f32);
        let (s, c) = (position as f32 * frequency).sin_cos();
        cos.push(c);
        sin.push(s);
    }
    let shape = &[1, 1, half];
    let (cos, sin) = if values.dtype() == Dtype::Bfloat16 {
        let cast = |values: &[f32]| {
            values
                .iter()
                .map(|value| half::bf16::from_f32(*value))
                .collect::<Vec<_>>()
        };
        (
            Array::from_slice(&cast(&cos), shape),
            Array::from_slice(&cast(&sin), shape),
        )
    } else {
        (
            Array::from_slice(&cos, shape),
            Array::from_slice(&sin, shape),
        )
    };
    let lo_rot = x_lo.clone().mul(&cos).sub(&x_hi.clone().mul(&sin));
    let hi_rot = x_hi.mul(&cos).add(&x_lo.mul(&sin));
    let rotated = concatenate(&[&lo_rot, &hi_rot], -1).map_err(op("batched rope join"))?;
    concatenate(&[&rotated, &rest], -1).map_err(op("batched rope concat"))
}

/// Causal grouped-query attention with explicit max-subtracted softmax,
/// matching the oracle's numerics. q `[rows, H, D]`; k/v `[total, KV, D]`.
pub(crate) fn grouped_query_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    rows: usize,
    past_rows: usize,
) -> Result<Array, MlxError> {
    let total_rows = past_rows + rows;
    let repeat = ATTENTION_HEADS / KV_HEADS;
    let k16 = expand_kv_heads(keys, repeat)?;
    let v16 = expand_kv_heads(values, repeat)?;

    // [H, rows, D] @ [H, D, total] -> [H, rows, total]
    let q_t = queries
        .clone()
        .transpose_axes(&[1, 0, 2])
        .map_err(op("q heads first"))?;
    let k_t = k16.transpose_axes(&[1, 2, 0]).map_err(op("k layout"))?;
    let v_t = v16.transpose_axes(&[1, 0, 2]).map_err(op("v layout"))?;
    let scale = (ATTENTION_HEAD_DIM as f32).sqrt().recip();
    let scores = q_t
        .matmul(&k_t)
        .map_err(op("scores matmul"))?
        .mul(scalar_like(queries, scale));

    // Causal + prefix mask: query row i (absolute past_rows + i) attends to
    // keys 0..=past_rows + i.
    let mut mask = vec![f32::NEG_INFINITY; rows * total_rows];
    for row in 0..rows {
        let visible = (past_rows + row + 1).min(total_rows);
        for column in 0..visible {
            mask[row * total_rows + column] = 0.0;
        }
    }
    let mask = if scores.dtype() == Dtype::Bfloat16 {
        let mask = mask
            .iter()
            .map(|value| half::bf16::from_f32(*value))
            .collect::<Vec<_>>();
        Array::from_slice(&mask, &[rows as i32, total_rows as i32])
    } else {
        Array::from_slice(&mask, &[rows as i32, total_rows as i32])
    }
    .reshape(&[1, rows as i32, total_rows as i32])
    .map_err(op("mask reshape"))?;
    let scores = scores.add(&mask);

    let maximum = scores.max_axis(-1, true).map_err(op("softmax max"))?;
    let exp = scores.sub(&maximum).exp().map_err(op("softmax exp"))?;
    let denominator = exp.sum_axis(-1, true).map_err(op("softmax sum"))?;
    let probabilities = exp.div(&denominator);

    let mixed = probabilities.matmul(&v_t).map_err(op("mix matmul"))?;
    mixed
        .transpose_axes(&[1, 0, 2])
        .map_err(op("mixed back"))?
        .reshape(&[rows as i32, ATTENTION_SIZE as i32])
        .map_err(op("mixed reshape"))
}

/// Expand `[total, KV, D]` to `[total, H, D]` by repeating each kv head
/// `repeat` times (query head h uses kv head h / repeat).
pub(crate) fn expand_kv_heads(kv: &Array, repeat: usize) -> Result<Array, MlxError> {
    let total = kv.shape()[0];
    let reshaped = kv
        .clone()
        .reshape(&[total, KV_HEADS as i32, 1, ATTENTION_HEAD_DIM as i32])
        .map_err(op("kv reshape"))?;
    broadcast_to(
        &reshaped,
        &[
            total,
            KV_HEADS as i32,
            repeat as i32,
            ATTENTION_HEAD_DIM as i32,
        ],
    )
    .map_err(op("kv broadcast"))?
    .reshape(&[total, ATTENTION_HEADS as i32, ATTENTION_HEAD_DIM as i32])
    .map_err(op("kv flatten"))
}

/// Partial rotary embedding over the first `ROTARY_DIM` dims of each head.
///
/// Mirrors the oracle exactly: NeoX pairing `(i, i + ROTARY_DIM/2)`,
/// frequencies `theta^(-2i/ROTARY_DIM)` computed on the host in FP32, and
/// rotation by `(position_start + row) * frequency`. Only the first 64 of
/// the 256 head dims rotate. `fast::rope` is deliberately not used here:
/// its position-axis convention and internal frequency math would add an
/// avoidable numerical variable to the first parity bring-up.
pub(crate) fn apply_rotary(
    values: &Array,
    rows: usize,
    position_start: usize,
    heads: usize,
) -> Result<Array, MlxError> {
    let rot = ROTARY_DIM as i32;
    let half = (ROTARY_DIM / 2) as i32;

    // Split each head's dims into [first ROTARY_DIM | remaining 192]: the
    // 256 dims are four 64-wide chunks; chunk 0 is the rotary window.
    let reshaped = values
        .clone()
        .reshape(&[rows as i32, heads as i32, 4, rot])
        .map_err(op("rope reshape"))?;
    let parts = reshaped
        .split_at_indices(&[1], 2)
        .map_err(op("rope split"))?;
    let first = parts[0]
        .clone()
        .reshape(&[rows as i32, heads as i32, rot])
        .map_err(op("rope first"))?;
    let rest = parts[1]
        .clone()
        .reshape(&[rows as i32, heads as i32, 3 * rot])
        .map_err(op("rope rest"))?;

    // NeoX pair split: (lo, hi) halves of the rotary window.
    let pairs = first
        .reshape(&[rows as i32, heads as i32, 2, half])
        .map_err(op("rope pairs"))?;
    let pair_split = pairs
        .split_at_indices(&[1], 2)
        .map_err(op("rope pair split"))?;
    let x_lo = pair_split[0]
        .clone()
        .reshape(&[rows as i32, heads as i32, half])
        .map_err(op("rope lo"))?;
    let x_hi = pair_split[1]
        .clone()
        .reshape(&[rows as i32, heads as i32, half])
        .map_err(op("rope hi"))?;

    // cos/sin tables `[rows, 1, half]` built on the host in FP32.
    let mut cos = Vec::with_capacity(rows * (ROTARY_DIM / 2));
    let mut sin = Vec::with_capacity(rows * (ROTARY_DIM / 2));
    for row in 0..rows {
        for index in 0..ROTARY_DIM / 2 {
            let frequency = ROPE_THETA.powf(-((2 * index) as f32) / ROTARY_DIM as f32);
            let angle = (position_start + row) as f32 * frequency;
            let (s, c) = angle.sin_cos();
            cos.push(c);
            sin.push(s);
        }
    }
    let shape = &[rows as i32, 1, half];
    let (cos, sin) = if values.dtype() == Dtype::Bfloat16 {
        let to_bf16 = |values: &[f32]| -> Vec<half::bf16> {
            values.iter().map(|v| half::bf16::from_f32(*v)).collect()
        };
        (
            Array::from_slice(&to_bf16(&cos), shape),
            Array::from_slice(&to_bf16(&sin), shape),
        )
    } else {
        (
            Array::from_slice(&cos, shape),
            Array::from_slice(&sin, shape),
        )
    };

    let lo_rot = x_lo.clone().mul(&cos).sub(&x_hi.clone().mul(&sin));
    let hi_rot = x_hi.mul(&cos).add(&x_lo.mul(&sin));
    let rotated = concatenate(&[&lo_rot, &hi_rot], -1).map_err(op("rope join"))?;
    concatenate(&[&rotated, &rest], -1).map_err(op("rope concat"))
}

/// Per-head RMSNorm with folded `(1 + w)` weights over the last axis.
pub(crate) fn norm_heads(values: &Array, weight: &Array) -> Result<Array, MlxError> {
    let dims = values.shape().to_vec();
    let rows: i32 = dims.iter().take(dims.len() - 1).product();
    let width = *dims.last().expect("non-empty shape");
    let flat = values
        .clone()
        .reshape(&[rows, width])
        .map_err(op("norm reshape"))?;
    let normed = fast::rms_norm(&flat, Some(weight), RMS_EPSILON).map_err(op("head norm"))?;
    normed.reshape(&dims).map_err(op("norm restore"))
}

#[cfg(test)]
mod vectorized_tests {
    use super::batched_key_padding_mask;

    #[test]
    fn attention_mask_excludes_right_padding_for_shorter_lanes() {
        let mask = batched_key_padding_mask(&[2, 4], 3, 2, 6).unwrap();
        assert_eq!(&mask[..6], &[0.0, 0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY]);
        assert_eq!(&mask[6..], &[0.0; 6]);
    }
}
