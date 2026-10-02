use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::super::Qwen35Error;
use super::embedding::{verify_decoder_shard, Qwen35Embedding};
use super::geometry::Qwen35Geometry;

mod full_attention;
mod linear_attention;

pub(crate) use full_attention::*;
pub(crate) use linear_attention::*;

/// Geometry of the pinned backbone the module constants mirror. Layer
/// structs carry [`Qwen35Geometry`] values so the Clef family can reuse the
/// same kernels at its own frozen widths.
pub(super) const PINNED: Qwen35Geometry = Qwen35Geometry::PINNED;
pub(super) const HIDDEN_SIZE: usize = PINNED.hidden_size;
// State-shape constants are shared with the branch-state implementation so
// storage accounting and fixtures stay tied to the executing architecture.
pub(super) const VALUE_HEADS: usize = PINNED.value_heads;
pub(super) const HEAD_DIM: usize = PINNED.head_dim;
// The remaining aliases survive for the kernel-equivalence tests and the
// branch-state storage accounting; the executing kernels take geometry
// values directly.
#[allow(dead_code)]
pub(super) const KEY_SIZE: usize = PINNED.key_size();
#[allow(dead_code)]
pub(super) const VALUE_SIZE: usize = PINNED.value_size();
pub(super) const QKV_SIZE: usize = PINNED.qkv_size();
pub(super) const CONV_KERNEL: usize = PINNED.conv_kernel;
pub(super) const RMS_EPSILON: f32 = PINNED.rms_epsilon;
#[allow(dead_code)]
pub(super) const ATTENTION_HEADS: usize = PINNED.attention_heads;
#[allow(dead_code)]
pub(super) const KV_HEADS: usize = PINNED.kv_heads;
#[allow(dead_code)]
pub(super) const ATTENTION_HEAD_DIM: usize = PINNED.attention_head_dim;
#[allow(dead_code)]
pub(super) const ATTENTION_SIZE: usize = PINNED.attention_size();
pub(super) const KV_SIZE: usize = PINNED.kv_size();
#[allow(dead_code)]
pub(super) const ROTARY_DIM: usize = PINNED.rotary_dim();
#[allow(dead_code)]
pub(super) const ROPE_THETA: f32 = Qwen35Geometry::ROPE_THETA;

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
    geometry: Qwen35Geometry,
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
            layer: DecoderLayer::load(&variables, 0, &device, PINNED)?,
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
        geometry: Qwen35Geometry,
    ) -> Result<Self, Qwen35Error> {
        let variables = variables.pp(layer_index);
        let mixer = if geometry.is_full_attention(layer_index) {
            TokenMixer::Full(FullAttention::load(&variables, geometry)?)
        } else {
            TokenMixer::Linear(LinearAttention::load(&variables, geometry)?)
        };
        Ok(Self {
            geometry,
            input_layernorm: vector(&variables, geometry.hidden_size, "input_layernorm.weight")?,
            mixer,
            post_attention_layernorm: vector(
                &variables,
                geometry.hidden_size,
                "post_attention_layernorm.weight",
            )?,
            mlp_gate_proj: variables.get(
                (geometry.intermediate_size, geometry.hidden_size),
                "mlp.gate_proj.weight",
            )?,
            mlp_up_proj: variables.get(
                (geometry.intermediate_size, geometry.hidden_size),
                "mlp.up_proj.weight",
            )?,
            mlp_down_proj: variables.get(
                (geometry.hidden_size, geometry.intermediate_size),
                "mlp.down_proj.weight",
            )?,
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
        let hidden_size = self.geometry.hidden_size;
        let normalized =
            rms_norm_zero_centered(residual, token_count, hidden_size, &self.input_layernorm);
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
            hidden_size,
            &self.post_attention_layernorm,
        );
        let mut gate = linear(
            &mlp_input,
            token_count,
            hidden_size,
            &self.mlp_gate_proj,
            self.geometry.intermediate_size,
            &self.device,
        )?;
        let up = linear(
            &mlp_input,
            token_count,
            hidden_size,
            &self.mlp_up_proj,
            self.geometry.intermediate_size,
            &self.device,
        )?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = silu(*gate) * up;
        }
        let mlp = linear(
            &gate,
            token_count,
            self.geometry.intermediate_size,
            &self.mlp_down_proj,
            hidden_size,
            &self.device,
        )?;
        for (hidden, mlp) in hidden.iter_mut().zip(mlp) {
            *hidden += mlp;
        }
        ensure_finite(&hidden, &format!("layer_{layer_index:02}"))?;
        Ok((hidden, state))
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

pub(crate) fn vector(
    variables: &VarBuilder<'_>,
    size: usize,
    name: &str,
) -> Result<Vec<f32>, Qwen35Error> {
    tensor_values(&variables.get(size, name)?)
}

pub(crate) fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>, Qwen35Error> {
    Ok(tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?)
}

pub(crate) fn linear(
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
    // BF16-stored checkpoints keep their mmap view; the upcast happens per
    // use so only one layer's weights are ever widened in memory. F32
    // tensors pass through `to_dtype` as the same refcounted handle.
    let weight = weight.to_dtype(DType::F32)?;
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

pub(crate) fn rms_norm_zero_centered(
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

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect()
}
