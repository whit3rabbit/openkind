use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::super::Qwen35Error;
use super::embedding::{verify_decoder_shard, Qwen35Embedding};

const HIDDEN_SIZE: usize = 2_560;
const INTERMEDIATE_SIZE: usize = 9_216;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const KEY_SIZE: usize = KEY_HEADS * HEAD_DIM;
const VALUE_SIZE: usize = VALUE_HEADS * HEAD_DIM;
const QKV_SIZE: usize = KEY_SIZE * 2 + VALUE_SIZE;
const CONV_KERNEL: usize = 4;
const RMS_EPSILON: f32 = 1e-6;
const ATTENTION_HEADS: usize = 16;
const KV_HEADS: usize = 4;
const ATTENTION_HEAD_DIM: usize = 256;
const ATTENTION_SIZE: usize = ATTENTION_HEADS * ATTENTION_HEAD_DIM;
const KV_SIZE: usize = KV_HEADS * ATTENTION_HEAD_DIM;
const ROTARY_DIM: usize = 64;
const ROPE_THETA: f32 = 10_000_000.0;

/// FP32 output from the first Qwen3.5 decoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0Output {
    token_count: usize,
    values: Vec<f32>,
}

impl Layer0Output {
    /// Number of sequence positions in the output.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    /// Row-major `[token_count, 2560]` decoder output.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Final-token vector compared with `diagnostic.layer_00`.
    #[must_use]
    pub fn last_token(&self) -> &[f32] {
        let start = (self.token_count - 1) * HIDDEN_SIZE;
        &self.values[start..]
    }
}

/// Native FP32 implementation of Qwen3.5 decoder `layer_00`.
///
/// Large projections use Candle's CPU tensor backend, with Accelerate BLAS on
/// macOS. The DeltaNet recurrence and causal depthwise convolution are explicit
/// host FP32 implementations of the frozen Transformers 5.17.0 equations.
#[derive(Debug)]
pub struct Qwen35Layer0 {
    embedding: Qwen35Embedding,
    layer: DecoderLayer,
}

#[derive(Debug)]
pub(super) struct DecoderLayer {
    input_layernorm: Vec<f32>,
    mixer: TokenMixer,
    post_attention_layernorm: Vec<f32>,
    mlp_gate_proj: Tensor,
    mlp_up_proj: Tensor,
    mlp_down_proj: Tensor,
    device: Device,
}

#[derive(Debug)]
enum TokenMixer {
    Linear(LinearAttention),
    Full(FullAttention),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum LayerState {
    Linear { conv: Vec<f32>, recurrent: Vec<f32> },
    Full { keys: Vec<f32>, values: Vec<f32> },
}

impl LayerState {
    pub(super) fn byte_len(&self) -> usize {
        let values = match self {
            Self::Linear { conv, recurrent } => conv.len() + recurrent.len(),
            Self::Full { keys, values } => keys.len() + values.len(),
        };
        values * std::mem::size_of::<f32>()
    }
}

#[derive(Debug)]
struct LinearAttention {
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

#[derive(Debug)]
struct FullAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out_proj: Tensor,
}

impl Qwen35Layer0 {
    /// Load exact layer-zero weights from the pinned checkpoint.
    ///
    /// Both checkpoint shards are size- and digest-verified before this method
    /// maps them. Qwen3.5 stores the layer-zero MLP in the first shard and its
    /// norms plus DeltaNet weights in the second shard.
    pub fn load(checkpoint_root: impl AsRef<Path>) -> Result<Self, Qwen35Error> {
        let checkpoint_root = checkpoint_root.as_ref();
        let embedding = Qwen35Embedding::load(checkpoint_root)?;
        let decoder_shard = verify_decoder_shard(checkpoint_root)?;
        let embedding_shard = embedding.verified_shard_path();
        let device = Device::Cpu;
        // SAFETY: both paths name read-only shards whose exact byte lengths and
        // SHA-256 digests were validated immediately above. The mapping stays
        // owned by the tensors returned through this VarBuilder.
        let variables = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[embedding_shard, decoder_shard.as_path()],
                DType::F32,
                &device,
            )?
        }
        .pp("model")
        .pp("language_model")
        .pp("layers");

        Ok(Self {
            embedding,
            layer: DecoderLayer::load(&variables, 0, &device)?,
        })
    }

    /// Execute the first full-sequence decoder block in FP32.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Layer0Output, Qwen35Error> {
        let embedding = self.embedding.embed(input_ids)?;
        let token_count = embedding.token_count();
        let hidden = self.layer.forward(embedding.values(), token_count, 0)?;
        Ok(Layer0Output {
            token_count,
            values: hidden,
        })
    }
}

impl DecoderLayer {
    pub(super) fn load(
        variables: &VarBuilder<'_>,
        layer_index: usize,
        device: &Device,
    ) -> Result<Self, Qwen35Error> {
        let variables = variables.pp(layer_index);
        let mixer = if layer_index % 4 == 3 {
            TokenMixer::Full(FullAttention {
                q_proj: variables
                    .get((ATTENTION_SIZE * 2, HIDDEN_SIZE), "self_attn.q_proj.weight")?,
                k_proj: variables.get((KV_SIZE, HIDDEN_SIZE), "self_attn.k_proj.weight")?,
                v_proj: variables.get((KV_SIZE, HIDDEN_SIZE), "self_attn.v_proj.weight")?,
                q_norm: vector(&variables, ATTENTION_HEAD_DIM, "self_attn.q_norm.weight")?,
                k_norm: vector(&variables, ATTENTION_HEAD_DIM, "self_attn.k_norm.weight")?,
                out_proj: variables
                    .get((HIDDEN_SIZE, ATTENTION_SIZE), "self_attn.o_proj.weight")?,
            })
        } else {
            TokenMixer::Linear(LinearAttention {
                in_proj_qkv: variables
                    .get((QKV_SIZE, HIDDEN_SIZE), "linear_attn.in_proj_qkv.weight")?,
                in_proj_z: variables
                    .get((VALUE_SIZE, HIDDEN_SIZE), "linear_attn.in_proj_z.weight")?,
                in_proj_b: variables
                    .get((VALUE_HEADS, HIDDEN_SIZE), "linear_attn.in_proj_b.weight")?,
                in_proj_a: variables
                    .get((VALUE_HEADS, HIDDEN_SIZE), "linear_attn.in_proj_a.weight")?,
                conv1d: tensor_values(
                    &variables.get((QKV_SIZE, 1, CONV_KERNEL), "linear_attn.conv1d.weight")?,
                )?,
                dt_bias: vector(&variables, VALUE_HEADS, "linear_attn.dt_bias")?,
                a_log: vector(&variables, VALUE_HEADS, "linear_attn.A_log")?,
                delta_norm: vector(&variables, HEAD_DIM, "linear_attn.norm.weight")?,
                out_proj: variables
                    .get((HIDDEN_SIZE, VALUE_SIZE), "linear_attn.out_proj.weight")?,
            })
        };
        Ok(Self {
            input_layernorm: vector(&variables, HIDDEN_SIZE, "input_layernorm.weight")?,
            mixer,
            post_attention_layernorm: vector(
                &variables,
                HIDDEN_SIZE,
                "post_attention_layernorm.weight",
            )?,
            mlp_gate_proj: variables
                .get((INTERMEDIATE_SIZE, HIDDEN_SIZE), "mlp.gate_proj.weight")?,
            mlp_up_proj: variables.get((INTERMEDIATE_SIZE, HIDDEN_SIZE), "mlp.up_proj.weight")?,
            mlp_down_proj: variables
                .get((HIDDEN_SIZE, INTERMEDIATE_SIZE), "mlp.down_proj.weight")?,
            device: device.clone(),
        })
    }

    pub(super) fn forward(
        &self,
        residual: &[f32],
        token_count: usize,
        layer_index: usize,
    ) -> Result<Vec<f32>, Qwen35Error> {
        self.forward_with_state(residual, token_count, layer_index, 0, None)
            .map(|(hidden, _)| hidden)
    }

    pub(super) fn forward_with_state(
        &self,
        residual: &[f32],
        token_count: usize,
        layer_index: usize,
        position_start: usize,
        previous_state: Option<&LayerState>,
    ) -> Result<(Vec<f32>, LayerState), Qwen35Error> {
        let normalized =
            rms_norm_zero_centered(residual, token_count, HIDDEN_SIZE, &self.input_layernorm);
        let (attention, state) = match (&self.mixer, previous_state) {
            (TokenMixer::Linear(mixer), None) => {
                mixer.forward_with_state(&normalized, token_count, &self.device, None)?
            }
            (TokenMixer::Linear(mixer), Some(LayerState::Linear { conv, recurrent })) => mixer
                .forward_with_state(
                    &normalized,
                    token_count,
                    &self.device,
                    Some((conv, recurrent)),
                )?,
            (TokenMixer::Full(mixer), None) => mixer.forward_with_state(
                &normalized,
                token_count,
                position_start,
                &self.device,
                None,
            )?,
            (TokenMixer::Full(mixer), Some(LayerState::Full { keys, values })) => mixer
                .forward_with_state(
                    &normalized,
                    token_count,
                    position_start,
                    &self.device,
                    Some((keys, values)),
                )?,
            _ => {
                return Err(Qwen35Error::InvalidInput(format!(
                    "cache kind does not match layer_{layer_index:02}"
                )))
            }
        };
        let mut hidden = add(residual, &attention);
        let mlp_input = rms_norm_zero_centered(
            &hidden,
            token_count,
            HIDDEN_SIZE,
            &self.post_attention_layernorm,
        );
        let mut gate = linear(
            &mlp_input,
            token_count,
            HIDDEN_SIZE,
            &self.mlp_gate_proj,
            INTERMEDIATE_SIZE,
            &self.device,
        )?;
        let up = linear(
            &mlp_input,
            token_count,
            HIDDEN_SIZE,
            &self.mlp_up_proj,
            INTERMEDIATE_SIZE,
            &self.device,
        )?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = silu(*gate) * up;
        }
        let mlp = linear(
            &gate,
            token_count,
            INTERMEDIATE_SIZE,
            &self.mlp_down_proj,
            HIDDEN_SIZE,
            &self.device,
        )?;
        for (hidden, mlp) in hidden.iter_mut().zip(mlp) {
            *hidden += mlp;
        }
        ensure_finite(&hidden, &format!("layer_{layer_index:02}"))?;
        Ok((hidden, state))
    }
}

impl LinearAttention {
    fn forward_with_state(
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

impl FullAttention {
    fn forward_with_state(
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

fn ensure_finite(values: &[f32], stage: &str) -> Result<(), Qwen35Error> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Qwen35Error::Numerical(format!(
            "{stage} output element {index} is not finite: {value}"
        )));
    }
    Ok(())
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

fn apply_rotary(values: &mut [f32], heads: usize, rows: usize, position_start: usize) {
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

fn causal_grouped_query_attention(
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

fn vector(variables: &VarBuilder<'_>, size: usize, name: &str) -> Result<Vec<f32>, Qwen35Error> {
    tensor_values(&variables.get(size, name)?)
}

fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>, Qwen35Error> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

fn linear(
    input: &[f32],
    rows: usize,
    input_width: usize,
    weight: &Tensor,
    output_width: usize,
    device: &Device,
) -> Result<Vec<f32>, Qwen35Error> {
    if input.len() != rows * input_width {
        return Err(Qwen35Error::InvalidInput(format!(
            "linear input expected {} values, found {}",
            rows * input_width,
            input.len()
        )));
    }
    let input = Tensor::from_vec(input.to_vec(), (rows, input_width), device)?;
    let output = input.matmul(&weight.t()?)?;
    let dimensions = output.dims2()?;
    if dimensions != (rows, output_width) {
        return Err(Qwen35Error::InvalidTensor {
            name: "linear output".to_owned(),
            message: format!("expected [{rows}, {output_width}], found {dimensions:?}"),
        });
    }
    tensor_values(&output)
}

pub(super) fn rms_norm_zero_centered(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * width);
    debug_assert_eq!(weight.len(), width);
    let mut output = vec![0.0_f32; input.len()];
    for row in 0..rows {
        let start = row * width;
        let values = &input[start..start + width];
        let variance = values.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let scale = (variance + RMS_EPSILON).sqrt().recip();
        for column in 0..width {
            output[start + column] = values[column] * scale * (1.0 + weight[column]);
        }
    }
    output
}

#[cfg(test)]
fn causal_depthwise_conv_silu(
    input: &[f32],
    rows: usize,
    channels: usize,
    weight: &[f32],
    kernel: usize,
) -> Vec<f32> {
    causal_depthwise_conv_silu_with_state(input, rows, channels, weight, kernel, None).0
}

fn causal_depthwise_conv_silu_with_state(
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
fn gated_delta_recurrent_with_state(
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

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_depthwise_convolution_uses_left_padding_and_silu() {
        let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        let weight = [1.0_f32, 2.0, 1.0, -1.0];
        let output = causal_depthwise_conv_silu(&input, 3, 2, &weight, 2);
        let expected_raw = [2.0_f32, -10.0, 5.0, -10.0, 8.0, -10.0];
        for (actual, expected) in output.iter().zip(expected_raw) {
            assert!((actual - silu(expected)).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_centered_rms_norm_applies_one_plus_weight() {
        let output = rms_norm_zero_centered(&[3.0, 4.0], 1, 2, &[0.0, 1.0]);
        let scale = ((25.0_f32 / 2.0) + RMS_EPSILON).sqrt().recip();
        assert!((output[0] - 3.0 * scale).abs() < 1e-6);
        assert!((output[1] - 8.0 * scale).abs() < 1e-6);
    }

    #[test]
    fn activations_are_finite_at_extremes() {
        for value in [-100.0_f32, -1.0, 0.0, 1.0, 100.0] {
            assert!(sigmoid(value).is_finite());
            assert!(silu(value).is_finite());
            assert!(softplus(value).is_finite());
        }
    }

    #[test]
    fn cached_convolution_matches_one_pass_execution() {
        let channels = 2;
        let kernel = 4;
        let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0];
        let weight = [1.0_f32, 0.5, -0.25, 2.0, -1.0, 0.25, 0.5, 1.0];
        let (full, _) =
            causal_depthwise_conv_silu_with_state(&input, 5, channels, &weight, kernel, None);
        let (prefix, state) =
            causal_depthwise_conv_silu_with_state(&input[..6], 3, channels, &weight, kernel, None);
        let (suffix, _) = causal_depthwise_conv_silu_with_state(
            &input[6..],
            2,
            channels,
            &weight,
            kernel,
            Some(&state),
        );
        assert_eq!([prefix, suffix].concat(), full);
    }

    #[test]
    fn rotary_positions_are_continuation_stable() {
        let mut full = vec![0.25_f32; 3 * ATTENTION_HEAD_DIM];
        let mut prefix = full[..2 * ATTENTION_HEAD_DIM].to_vec();
        let mut suffix = full[2 * ATTENTION_HEAD_DIM..].to_vec();
        apply_rotary(&mut full, 1, 3, 0);
        apply_rotary(&mut prefix, 1, 2, 0);
        apply_rotary(&mut suffix, 1, 1, 2);
        assert_eq!([prefix, suffix].concat(), full);
    }

    #[test]
    fn cached_causal_attention_matches_one_pass_execution() {
        let queries = vec![0.125_f32; 2 * ATTENTION_SIZE];
        let keys = vec![0.25_f32; 2 * KV_SIZE];
        let values: Vec<_> = (0..2 * KV_SIZE)
            .map(|index| index as f32 / 10_000.0)
            .collect();
        let full = causal_grouped_query_attention(&queries, &keys, &values, 2, 0);
        let prefix = causal_grouped_query_attention(
            &queries[..ATTENTION_SIZE],
            &keys[..KV_SIZE],
            &values[..KV_SIZE],
            1,
            0,
        );
        let suffix =
            causal_grouped_query_attention(&queries[ATTENTION_SIZE..], &keys, &values, 1, 1);
        assert_eq!([prefix, suffix].concat(), full);
    }

    #[test]
    fn cached_delta_recurrence_matches_one_pass_execution() {
        let rows = 3;
        let qkv = vec![0.01_f32; rows * QKV_SIZE];
        let z = vec![0.2_f32; rows * VALUE_SIZE];
        let beta = vec![0.1_f32; rows * VALUE_HEADS];
        let decay = vec![0.05_f32; rows * VALUE_HEADS];
        let dt_bias = vec![0.0_f32; VALUE_HEADS];
        let a_log = vec![0.0_f32; VALUE_HEADS];
        let norm = vec![1.0_f32; HEAD_DIM];
        let (full, full_state) = gated_delta_recurrent_with_state(
            &qkv, &z, &beta, &decay, &dt_bias, &a_log, &norm, rows, None,
        );
        let (prefix, prefix_state) = gated_delta_recurrent_with_state(
            &qkv[..2 * QKV_SIZE],
            &z[..2 * VALUE_SIZE],
            &beta[..2 * VALUE_HEADS],
            &decay[..2 * VALUE_HEADS],
            &dt_bias,
            &a_log,
            &norm,
            2,
            None,
        );
        let (suffix, suffix_state) = gated_delta_recurrent_with_state(
            &qkv[2 * QKV_SIZE..],
            &z[2 * VALUE_SIZE..],
            &beta[2 * VALUE_HEADS..],
            &decay[2 * VALUE_HEADS..],
            &dt_bias,
            &a_log,
            &norm,
            1,
            Some(&prefix_state),
        );
        assert_eq!([prefix, suffix].concat(), full);
        assert_eq!(suffix_state, full_state);
    }
}
