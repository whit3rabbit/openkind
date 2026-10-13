//! Linear attention and DeltaNet recurrence for Qwen 3.5 decoder blocks.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::qwen35::backbone::geometry::Qwen35Geometry;
use crate::qwen35::Qwen35Error;

use super::{linear, tensor_bytes, tensor_values, vector, LayerState, RMS_EPSILON};

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
    /// Bytes of weight storage this layer holds.
    pub(crate) fn resident_bytes(&self) -> usize {
        let tensors = [
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_b,
            &self.in_proj_a,
            &self.out_proj,
        ];
        tensors
            .iter()
            .map(|tensor| tensor_bytes(tensor))
            .sum::<usize>()
            + 4 * (self.conv1d.len()
                + self.dt_bias.len()
                + self.a_log.len()
                + self.delta_norm.len())
    }

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
    // The checkpoint stores taps per channel; transposing to tap-major lets the
    // inner loop run along contiguous channels. Each output element still sums
    // its taps in increasing tap order from 0.0, so values are bit-identical.
    let mut tap_major = vec![0.0_f32; weight.len()];
    for (channel, taps) in weight.chunks_exact(kernel).enumerate() {
        for (tap, &value) in taps.iter().enumerate() {
            tap_major[tap * channels + channel] = value;
        }
    }
    let mut output = vec![0.0_f32; input.len()];
    let padding = kernel - 1;
    for (row, output_row) in output.chunks_exact_mut(channels).enumerate() {
        let combined_row = preceding_rows + row;
        for tap in 0..kernel {
            if combined_row + tap < padding {
                continue;
            }
            let source_row = combined_row + tap - padding;
            let source = &combined[source_row * channels..(source_row + 1) * channels];
            let taps = &tap_major[tap * channels..(tap + 1) * channels];
            for ((value, &source), &tap_weight) in output_row.iter_mut().zip(source).zip(taps) {
                *value += source * tap_weight;
            }
        }
        for value in output_row.iter_mut() {
            *value = silu(*value);
        }
    }
    let state_start = combined.len().saturating_sub(channels * kernel);
    let mut state = vec![0.0_f32; channels * kernel];
    let available = &combined[state_start..];
    let destination = state.len() - available.len();
    state[destination..].copy_from_slice(available);
    (output, state)
}

/// Read-only inputs shared by every head of one recurrence call.
struct RecurrenceInputs<'a> {
    qkv: &'a [f32],
    z: &'a [f32],
    beta_projection: &'a [f32],
    decay_projection: &'a [f32],
    dt_bias: &'a [f32],
    a_log: &'a [f32],
    norm_weight: &'a [f32],
    rows: usize,
    geometry: Qwen35Geometry,
}

/// Below this many head-rows the thread hand-off costs more than it saves.
const PARALLEL_HEAD_ROWS: usize = 64;

/// Operator ceiling on kernel threads per request; 0 means no ceiling.
static KERNEL_THREAD_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Cap the threads one request's CPU kernels may use (0 lifts the cap).
///
/// The kernels parallelize within a request, so a daemon that runs several
/// evaluations at once should divide the cores among them to avoid
/// oversubscription. Results do not depend on the thread count.
pub fn set_kernel_thread_limit(limit: usize) {
    KERNEL_THREAD_LIMIT.store(limit, std::sync::atomic::Ordering::Relaxed);
}

/// Threads one request's kernels may use: the hardware count, read once,
/// bounded by the operator ceiling.
pub(crate) fn available_workers() -> usize {
    static HARDWARE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let hardware = *HARDWARE.get_or_init(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    match KERNEL_THREAD_LIMIT.load(std::sync::atomic::Ordering::Relaxed) {
        0 => hardware,
        limit => limit.min(hardware),
    }
}

/// Worker count for the per-head recurrence. Heads never read each other's
/// state, so splitting them across threads cannot change any value.
fn recurrence_workers(value_heads: usize, rows: usize) -> usize {
    if rows * value_heads < PARALLEL_HEAD_ROWS {
        return 1;
    }
    available_workers().min(value_heads).max(1)
}

/// Gated-DeltaNet recurrence over `rows` tokens.
///
/// Arithmetic is the exact per-element sequence of
/// `gated_delta_recurrent_reference`: every reduction still accumulates over
/// the key index in increasing order, so outputs are bit-identical. What
/// changes is the memory walk and the scheduling: the state is traversed along
/// its contiguous value axis (the old column-strided reductions could not
/// vectorize), the four passes over each head's state are fused into two, and
/// the independent value heads run on separate threads.
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
    let qkv_size = geometry.qkv_size();
    let value_size = geometry.value_size();
    debug_assert_eq!(qkv.len(), rows * qkv_size);
    debug_assert_eq!(z.len(), rows * value_size);
    debug_assert_eq!(beta_projection.len(), rows * value_heads);
    debug_assert_eq!(decay_projection.len(), rows * value_heads);
    let mut state = initial_state.map_or_else(
        || vec![0.0_f32; value_heads * head_dim * head_dim],
        <[f32]>::to_vec,
    );
    debug_assert_eq!(state.len(), value_heads * head_dim * head_dim);
    let inputs = RecurrenceInputs {
        qkv,
        z,
        beta_projection,
        decay_projection,
        dt_bias,
        a_log,
        norm_weight,
        rows,
        geometry,
    };

    // Each head writes its rows into its own [row][dim] block, scattered into
    // the interleaved output once every head is done.
    let head_state_len = head_dim * head_dim;
    let head_output_len = rows * head_dim;
    let mut head_outputs = vec![0.0_f32; value_heads * head_output_len];
    let workers = recurrence_workers(value_heads, rows);
    if workers <= 1 {
        for (value_head, (state_head, head_output)) in state
            .chunks_mut(head_state_len)
            .zip(head_outputs.chunks_mut(head_output_len))
            .enumerate()
        {
            recur_head(&inputs, value_head, state_head, head_output);
        }
    } else {
        let heads_per_worker = value_heads.div_ceil(workers);
        std::thread::scope(|scope| {
            let state_blocks = state.chunks_mut(heads_per_worker * head_state_len);
            let output_blocks = head_outputs.chunks_mut(heads_per_worker * head_output_len);
            for (block, (state_block, output_block)) in state_blocks.zip(output_blocks).enumerate()
            {
                let inputs = &inputs;
                scope.spawn(move || {
                    for (offset, (state_head, head_output)) in state_block
                        .chunks_mut(head_state_len)
                        .zip(output_block.chunks_mut(head_output_len))
                        .enumerate()
                    {
                        recur_head(
                            inputs,
                            block * heads_per_worker + offset,
                            state_head,
                            head_output,
                        );
                    }
                });
            }
        });
    }

    let mut output = vec![0.0_f32; rows * value_size];
    for (value_head, head_output) in head_outputs.chunks(head_output_len).enumerate() {
        for (row, values) in head_output.chunks(head_dim).enumerate() {
            let start = row * value_size + value_head * head_dim;
            output[start..start + head_dim].copy_from_slice(values);
        }
    }
    (output, state)
}

/// One value head across all rows: updates `state_head` in place and writes
/// the normalized, gated outputs to `head_output` (`rows * head_dim` values).
fn recur_head(
    inputs: &RecurrenceInputs<'_>,
    value_head: usize,
    state_head: &mut [f32],
    head_output: &mut [f32],
) {
    let RecurrenceInputs {
        qkv,
        z,
        beta_projection,
        decay_projection,
        dt_bias,
        a_log,
        norm_weight,
        rows,
        geometry,
    } = *inputs;
    let value_heads = geometry.value_heads;
    let head_dim = geometry.head_dim;
    let key_heads = geometry.key_heads;
    let qkv_size = geometry.qkv_size();
    let value_size = geometry.value_size();
    let key_size = geometry.key_size();
    let query_scale = (head_dim as f32).sqrt().recip();
    let key_head = value_head / (value_heads / key_heads);

    let mut key = vec![0.0_f32; head_dim];
    let mut query = vec![0.0_f32; head_dim];
    // Holds the readout of the decayed state, then the delta computed from it.
    let mut delta = vec![0.0_f32; head_dim];

    for row in 0..rows {
        let qkv_row = row * qkv_size;
        let query_offset = qkv_row + key_head * head_dim;
        let key_offset = qkv_row + key_size + key_head * head_dim;
        let value_offset = qkv_row + key_size * 2 + value_head * head_dim;

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
        for (slot, value) in key.iter_mut().zip(&qkv[key_offset..key_offset + head_dim]) {
            *slot = value * key_norm;
        }
        for (slot, value) in query
            .iter_mut()
            .zip(&qkv[query_offset..query_offset + head_dim])
        {
            *slot = value * query_norm;
        }
        let beta = sigmoid(beta_projection[row * value_heads + value_head]);
        let decay = (-a_log[value_head].exp()
            * softplus(decay_projection[row * value_heads + value_head] + dt_bias[value_head]))
        .exp();

        // Pass one: decay the state and read it out along the key axis.
        delta.fill(0.0);
        for (state_row, &key_value) in state_head.chunks_exact_mut(head_dim).zip(&key) {
            for (state, memory) in state_row.iter_mut().zip(delta.iter_mut()) {
                *state *= decay;
                *memory += *state * key_value;
            }
        }
        for (memory, value) in delta
            .iter_mut()
            .zip(&qkv[value_offset..value_offset + head_dim])
        {
            *memory = (value - *memory) * beta;
        }

        // Pass two: rank-one update, fused with the query readout.
        let output = &mut head_output[row * head_dim..(row + 1) * head_dim];
        output.fill(0.0);
        for ((state_row, &key_value), &query_value) in
            state_head.chunks_exact_mut(head_dim).zip(&key).zip(&query)
        {
            for ((state, out), &delta_value) in
                state_row.iter_mut().zip(output.iter_mut()).zip(&delta)
            {
                *state += key_value * delta_value;
                *out += *state * query_value;
            }
        }

        let variance = output.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let norm_scale = (variance + RMS_EPSILON).sqrt().recip();
        let z_offset = row * value_size + value_head * head_dim;
        for (value_index, out) in output.iter_mut().enumerate() {
            let gate = silu(z[z_offset + value_index]);
            *out *= norm_scale * norm_weight[value_index] * gate;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn gated_delta_recurrent_reference(
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

#[cfg(test)]
mod recurrence_tests {
    use super::*;

    /// Deterministic pseudo-random values in roughly [-1, 1).
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

    fn assert_bit_identical(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "{what}[{index}]: {a} vs {e}");
        }
    }

    fn check(rows: usize, with_state: bool) {
        // Grouped value heads (2 per key head) with a head width that is not a
        // power of two exercises the key-head mapping and the chunk splitting.
        let geometry = Qwen35Geometry {
            value_heads: 6,
            key_heads: 3,
            head_dim: 12,
            ..Qwen35Geometry::PINNED
        };
        let value_heads = geometry.value_heads;
        let qkv = values(rows * geometry.qkv_size(), 1);
        let z = values(rows * geometry.value_size(), 2);
        let beta = values(rows * value_heads, 3);
        let decay = values(rows * value_heads, 4);
        let dt_bias = values(value_heads, 5);
        let a_log = values(value_heads, 6);
        let norm = values(geometry.head_dim, 7);
        let initial =
            with_state.then(|| values(value_heads * geometry.head_dim * geometry.head_dim, 8));
        let (out, state) = gated_delta_recurrent_with_state(
            &qkv,
            &z,
            &beta,
            &decay,
            &dt_bias,
            &a_log,
            &norm,
            rows,
            geometry,
            initial.as_deref(),
        );
        let (expected_out, expected_state) = gated_delta_recurrent_reference(
            &qkv,
            &z,
            &beta,
            &decay,
            &dt_bias,
            &a_log,
            &norm,
            rows,
            geometry,
            initial.as_deref(),
        );
        assert_bit_identical(&out, &expected_out, "output");
        assert_bit_identical(&state, &expected_state, "state");
    }

    /// The channel-strided convolution this module used before it was
    /// restructured for contiguous access.
    fn conv_reference(
        input: &[f32],
        rows: usize,
        channels: usize,
        weight: &[f32],
        kernel: usize,
        previous: Option<&[f32]>,
    ) -> Vec<f32> {
        let preceding_rows = previous.map_or(0, |_| kernel);
        let mut combined = Vec::new();
        if let Some(previous) = previous {
            combined.extend_from_slice(previous);
        }
        combined.extend_from_slice(input);
        let padding = kernel - 1;
        let mut output = vec![0.0_f32; input.len()];
        for row in 0..rows {
            let combined_row = preceding_rows + row;
            for channel in 0..channels {
                let mut value = 0.0_f32;
                for tap in 0..kernel {
                    if combined_row + tap >= padding {
                        let source_row = combined_row + tap - padding;
                        value += combined[source_row * channels + channel]
                            * weight[channel * kernel + tap];
                    }
                }
                output[row * channels + channel] = silu(value);
            }
        }
        output
    }

    #[test]
    fn convolution_matches_the_strided_reference_bit_for_bit() {
        let (rows, channels, kernel) = (9, 37, 4);
        let input = values(rows * channels, 11);
        let weight = values(channels * kernel, 12);
        let previous = values(channels * kernel, 13);
        for previous in [None, Some(previous.as_slice())] {
            let (output, _) = causal_depthwise_conv_silu_with_state(
                &input, rows, channels, &weight, kernel, previous,
            );
            let expected = conv_reference(&input, rows, channels, &weight, kernel, previous);
            assert_bit_identical(&output, &expected, "convolution");
        }
    }

    #[test]
    fn recurrence_matches_the_reference_bit_for_bit_serially() {
        // 6 heads x 3 rows is below the threading threshold.
        check(3, false);
        check(3, true);
    }

    #[test]
    fn recurrence_matches_the_reference_bit_for_bit_across_threads() {
        // 6 heads x 17 rows crosses the threshold, so heads run on workers.
        assert!(recurrence_workers(6, 17) > 1 || std::thread::available_parallelism().is_err());
        check(17, false);
        check(17, true);
    }
}
