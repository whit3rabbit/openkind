//! Full attention and grouped-query attention for Qwen 3.5 decoder blocks.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::qwen35::backbone::geometry::Qwen35Geometry;
use crate::qwen35::Qwen35Error;

use super::linear_attention::{available_workers, sigmoid};
use super::{linear, tensor_bytes, vector, LayerState};

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
    /// Bytes of weight storage this layer holds.
    pub(crate) fn resident_bytes(&self) -> usize {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj]
            .iter()
            .map(|tensor| tensor_bytes(tensor))
            .sum::<usize>()
            + 4 * (self.q_norm.len() + self.k_norm.len())
    }

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

    /// Attention forward over `token_count` new rows after an optional
    /// cached prefix, with an optional cooperative `check`.
    ///
    /// With a check, the quadratic attention kernel also runs it every
    /// [`CONTROL_CHECK_ROWS`] query rows, so a deadline or disconnect is
    /// observed within one layer. Without one, the unchecked kernel runs
    /// exactly as before.
    pub(crate) fn forward_with_state_and_check<E>(
        &self,
        normalized: &[f32],
        token_count: usize,
        position_start: usize,
        device: &Device,
        previous_state: Option<(&[f32], &[f32])>,
        check: Option<&mut dyn FnMut() -> Result<(), E>>,
    ) -> Result<(Vec<f32>, LayerState), E>
    where
        E: From<Qwen35Error>,
    {
        let geometry = self.geometry;
        // A noncausal cache is never valid: the cached prefix rows attended
        // only to the prefix, but bidirectional attention requires them to
        // see the suffix too, so continuing from them silently computes a
        // different model. Every noncausal caller runs full sequences.
        if !geometry.causal && previous_state.is_some() {
            return Err(Qwen35Error::InvalidInput(
                "noncausal full attention cannot continue from cached keys and values".to_owned(),
            )
            .into());
        }
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
            ))
            .into());
        }
        // Size each buffer once: `to_vec` followed by `extend_from_slice`
        // would copy the cached prefix again when the vector grows.
        let mut all_keys = Vec::with_capacity(past_tokens * kv_size + keys.len());
        all_keys.extend_from_slice(previous_keys.unwrap_or_default());
        all_keys.extend_from_slice(&keys);
        let mut all_values = Vec::with_capacity(past_tokens * kv_size + current_values.len());
        all_values.extend_from_slice(previous_values.unwrap_or_default());
        all_values.extend_from_slice(&current_values);
        let mut mixed = match check {
            None => causal_grouped_query_attention(
                &queries,
                &all_keys,
                &all_values,
                token_count,
                past_tokens,
                geometry,
            ),
            Some(check) => causal_grouped_query_attention_checked(
                &queries,
                &all_keys,
                &all_values,
                token_count,
                past_tokens,
                geometry,
                check,
            )?,
        };
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
                &projected[head_start + attention_head_dim..head_start + attention_head_dim * 2],
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
    let half = rotary_dim / 2;
    let attention_head_dim = geometry.attention_head_dim;
    let rope_theta = Qwen35Geometry::ROPE_THETA;
    debug_assert_eq!(values.len(), rows * heads * attention_head_dim);
    // The angle depends on the row and frequency index only, never on the head,
    // so build each (cos, sin) pair once and reuse it across every head.
    let frequencies: Vec<f32> = (0..half)
        .map(|index| rope_theta.powf(-((2 * index) as f32) / rotary_dim as f32))
        .collect();
    let mut trig = Vec::with_capacity(half);
    for row in 0..rows {
        trig.clear();
        trig.extend(frequencies.iter().map(|&frequency| {
            let angle = (position_start + row) as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            (cos, sin)
        }));
        for head in 0..heads {
            let offset = (row * heads + head) * attention_head_dim;
            let rotary = &mut values[offset..offset + rotary_dim];
            for (index, &(cos, sin)) in trig.iter().enumerate() {
                let first = rotary[index];
                let second = rotary[index + half];
                rotary[index] = first * cos - second * sin;
                rotary[index + half] = second * cos + first * sin;
            }
        }
    }
}

/// Query-row interval between cooperative control checks inside
/// [`causal_grouped_query_attention_checked`]. The check costs one atomic
/// load and one clock read; 64 rows of causal attention dwarf both.
const CONTROL_CHECK_ROWS: usize = 64;

pub(crate) fn causal_grouped_query_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    past_rows: usize,
    geometry: Qwen35Geometry,
) -> Vec<f32> {
    causal_grouped_query_attention_checked(queries, keys, values, rows, past_rows, geometry, || {
        Ok::<(), std::convert::Infallible>(())
    })
    .expect("control check without a failure mode cannot fail")
}

/// Causal grouped-query attention with a cooperative control check every
/// [`CONTROL_CHECK_ROWS`] query rows.
///
/// The scalar kernel is quadratic in the caller-selected row count, so a
/// deadline or disconnect must be observable inside the loop, not only
/// around the whole call: a remote caller otherwise occupies the execution
/// slot for the full cost of one attention layer however long that takes.
pub(crate) fn causal_grouped_query_attention_checked<E>(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    past_rows: usize,
    geometry: Qwen35Geometry,
    mut check: impl FnMut() -> Result<(), E>,
) -> Result<Vec<f32>, E> {
    let attention_heads = geometry.attention_heads;
    let attention_size = geometry.attention_size();
    let kv_size = geometry.kv_size();
    debug_assert_eq!(queries.len(), rows * attention_size);
    debug_assert_eq!(keys.len(), (past_rows + rows) * kv_size);
    debug_assert_eq!(values.len(), (past_rows + rows) * kv_size);
    check()?;
    let inputs = AttentionInputs {
        queries,
        keys,
        values,
        rows,
        past_rows,
        geometry,
    };
    let mut output = vec![0.0_f32; rows * attention_size];
    // Query rows only read the shared keys and values, so rows within one
    // control-check block run on separate threads. The check itself stays on
    // the calling thread at every block boundary, exactly as before.
    for (block_index, block) in output
        .chunks_mut(CONTROL_CHECK_ROWS * attention_size)
        .enumerate()
    {
        check()?;
        let first_row = block_index * CONTROL_CHECK_ROWS;
        let block_rows = block.len() / attention_size;
        let key_visits = block_rows * attention_heads * (past_rows + first_row + block_rows / 2);
        let workers = if key_visits < PARALLEL_KEY_VISITS {
            1
        } else {
            available_workers().min(block_rows).max(1)
        };
        if workers <= 1 {
            for (offset, output_row) in block.chunks_exact_mut(attention_size).enumerate() {
                attend_row(&inputs, first_row + offset, output_row);
            }
        } else {
            let rows_per_worker = block_rows.div_ceil(workers);
            std::thread::scope(|scope| {
                for (chunk_index, chunk) in block
                    .chunks_mut(rows_per_worker * attention_size)
                    .enumerate()
                {
                    let inputs = &inputs;
                    scope.spawn(move || {
                        let chunk_first = first_row + chunk_index * rows_per_worker;
                        for (offset, output_row) in
                            chunk.chunks_exact_mut(attention_size).enumerate()
                        {
                            attend_row(inputs, chunk_first + offset, output_row);
                        }
                    });
                }
            });
        }
    }
    Ok(output)
}

/// Query-row blocks smaller than this many key visits stay on one thread.
const PARALLEL_KEY_VISITS: usize = 4_096;

struct AttentionInputs<'a> {
    queries: &'a [f32],
    keys: &'a [f32],
    values: &'a [f32],
    rows: usize,
    past_rows: usize,
    geometry: Qwen35Geometry,
}

/// Attention output of one query row across every head, accumulated into the
/// zeroed `output_row`.
fn attend_row(inputs: &AttentionInputs<'_>, row: usize, output_row: &mut [f32]) {
    let AttentionInputs {
        queries,
        keys,
        values,
        rows,
        past_rows,
        geometry,
    } = *inputs;
    let attention_heads = geometry.attention_heads;
    let kv_heads = geometry.kv_heads;
    let attention_head_dim = geometry.attention_head_dim;
    let scale = (attention_head_dim as f32).sqrt().recip();
    // Causal checkpoints let a query row see the keys up to and including its
    // own position; noncausal checkpoints (the d3 family's
    // `noncausal_full_attention` mode) lift the mask so every position sees
    // the whole prompt. The rotary positions are absolute either way, and the
    // recurrent DeltaNet layers stay causal by construction.
    let visible_rows = if geometry.causal {
        past_rows + row + 1
    } else {
        past_rows + rows
    };
    let mut scores = Vec::with_capacity(visible_rows);
    for query_head in 0..attention_heads {
        let kv_head = query_head / (attention_heads / kv_heads);
        let query_offset = (row * attention_heads + query_head) * attention_head_dim;
        scores.clear();
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
        let head_output =
            &mut output_row[query_head * attention_head_dim..(query_head + 1) * attention_head_dim];
        for (key_row, &score) in scores.iter().enumerate() {
            let probability = score / denominator;
            let value_offset = (key_row * kv_heads + kv_head) * attention_head_dim;
            let value_row = &values[value_offset..value_offset + attention_head_dim];
            for (out, &value) in head_output.iter_mut().zip(value_row) {
                *out += probability * value;
            }
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;

    #[allow(clippy::needless_range_loop)]
    fn apply_rotary_reference(
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

    fn values(count: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..count)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// The single-threaded kernel this module used before query rows were
    /// distributed across threads.
    fn reference(
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
        let mut output = vec![0.0_f32; rows * attention_size];
        let scale = (attention_head_dim as f32).sqrt().recip();
        for row in 0..rows {
            for query_head in 0..attention_heads {
                let kv_head = query_head / (attention_heads / kv_heads);
                let query_offset = (row * attention_heads + query_head) * attention_head_dim;
                let visible_rows = if geometry.causal {
                    past_rows + row + 1
                } else {
                    past_rows + rows
                };
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
                        output[output_offset + column] +=
                            probability * values[value_offset + column];
                    }
                }
            }
        }
        output
    }

    #[test]
    fn threaded_attention_matches_the_reference_bit_for_bit() {
        // The causal mask and the d3 family's noncausal mask both reach the
        // threaded path.
        for causal in [true, false] {
            let geometry = Qwen35Geometry {
                attention_heads: 4,
                kv_heads: 2,
                attention_head_dim: 8,
                causal,
                ..Qwen35Geometry::PINNED
            };
            // 70 rows span two control blocks; past rows exercise continuation.
            // The first block's key-visit count is above the threading threshold.
            for (rows, past_rows) in [(70, 0), (70, 11), (5, 3)] {
                let queries = values(rows * geometry.attention_size(), 1);
                let keys = values((past_rows + rows) * geometry.kv_size(), 2);
                let value_rows = values((past_rows + rows) * geometry.kv_size(), 3);
                let actual = causal_grouped_query_attention(
                    &queries,
                    &keys,
                    &value_rows,
                    rows,
                    past_rows,
                    geometry,
                );
                let expected = reference(&queries, &keys, &value_rows, rows, past_rows, geometry);
                assert_eq!(actual.len(), expected.len());
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "causal={causal} rows={rows} past={past_rows} [{index}]"
                    );
                }
            }
        }
    }

    fn tiny_layer(geometry: Qwen35Geometry) -> FullAttention {
        let tensor = |rows: usize, columns: usize, seed: u32| {
            Tensor::from_vec(values(rows * columns, seed), (rows, columns), &Device::Cpu)
                .expect("synthetic projection")
        };
        FullAttention {
            geometry,
            q_proj: tensor(geometry.attention_size() * 2, geometry.hidden_size, 31),
            k_proj: tensor(geometry.kv_size(), geometry.hidden_size, 32),
            v_proj: tensor(geometry.kv_size(), geometry.hidden_size, 33),
            q_norm: values(geometry.attention_head_dim, 34),
            k_norm: values(geometry.attention_head_dim, 35),
            out_proj: tensor(geometry.hidden_size, geometry.attention_size(), 36),
        }
    }

    #[test]
    fn continuing_from_a_cached_prefix_matches_one_pass_bit_for_bit() {
        let geometry = Qwen35Geometry {
            hidden_size: 16,
            attention_heads: 4,
            kv_heads: 2,
            attention_head_dim: 8,
            ..Qwen35Geometry::PINNED
        };
        let layer = tiny_layer(geometry);
        let rows = 7;
        let prefix_rows = 4;
        let input = values(rows * geometry.hidden_size, 41);
        let run = |slice: &[f32], count: usize, start: usize, cache: Option<(&[f32], &[f32])>| {
            layer
                .forward_with_state_and_check::<Qwen35Error>(
                    slice,
                    count,
                    start,
                    &Device::Cpu,
                    cache,
                    None,
                )
                .expect("attention forward")
        };
        let (full_output, full_state) = run(&input, rows, 0, None);
        let split = prefix_rows * geometry.hidden_size;
        let (prefix_output, prefix_state) = run(&input[..split], prefix_rows, 0, None);
        let LayerState::Full { keys, values } = &prefix_state else {
            panic!("full attention returns a key/value cache");
        };
        let before = (keys.clone(), values.clone());
        let (suffix_output, suffix_state) = run(
            &input[split..],
            rows - prefix_rows,
            prefix_rows,
            Some((keys, values)),
        );

        let joined: Vec<f32> = prefix_output.into_iter().chain(suffix_output).collect();
        assert_eq!(joined.len(), full_output.len());
        for (index, (a, e)) in joined.iter().zip(&full_output).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "output[{index}]");
        }
        // The extended cache is the whole prefix followed by the new rows.
        let (
            LayerState::Full {
                keys: a_keys,
                values: a_values,
            },
            LayerState::Full {
                keys: e_keys,
                values: e_values,
            },
        ) = (&suffix_state, &full_state)
        else {
            panic!("full attention returns a key/value cache");
        };
        assert_eq!(a_keys.len(), rows * geometry.kv_size());
        for (a, e) in a_keys
            .iter()
            .zip(e_keys)
            .chain(a_values.iter().zip(e_values))
        {
            assert_eq!(a.to_bits(), e.to_bits());
        }
        // The borrowed prefix cache was not touched.
        let LayerState::Full { keys, values } = &prefix_state else {
            unreachable!()
        };
        assert_eq!((keys, values), (&before.0, &before.1));
    }

    #[test]
    fn rotary_table_matches_the_per_head_reference_bit_for_bit() {
        let geometry = Qwen35Geometry {
            attention_heads: 4,
            kv_heads: 2,
            attention_head_dim: 16,
            ..Qwen35Geometry::PINNED
        };
        for heads in [4, 2] {
            for (rows, start) in [(7, 0), (5, 31)] {
                let input = values(rows * heads * geometry.attention_head_dim, 21);
                let mut actual = input.clone();
                let mut expected = input;
                apply_rotary(&mut actual, heads, rows, start, geometry);
                apply_rotary_reference(&mut expected, heads, rows, start, geometry);
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "heads={heads} rows={rows} [{index}]"
                    );
                }
            }
        }
    }
}
