//! Candle quantized-runner execution for the Clef GGUF profiles.
//!
//! The bartowski GGUF checkpoints carry the Qwen3.5 hybrid backbone in
//! llama.cpp `qwen3.5` tensor layout: fused `attn_qkv`, `attn_gate`, and
//! `ssm_*` tensors for the gated-DeltaNet layers, split q/k/v tensors for
//! the full-attention layers, and K-quantized MLPs. Projections execute
//! through candle's `QMatMul` kernels over the on-disk packing; small
//! SSM/conv/norm tensors are dequantized to F32 at load and keep the exact
//! vector math of the CPU oracle. The joint head and the untied output
//! embedding come from the official BF16 artifacts.
//!
//! The forward equations — per-token DeltaNet recurrence, causal conv,
//! grouped-query attention, and rotary embedding — reuse the
//! parity-verified kernels from [`crate::qwen35::backbone`] so the
//! quantized path cannot drift from the CPU oracle in its arithmetic.

use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::gguf_file as gguf;
use candle_core::quantized::{QMatMul, QTensor};
use candle_nn::Module as _;
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::families::support::{read_json, verify_digest, FamilyControl, FamilyError};
use crate::qwen35::{
    apply_rotary, causal_depthwise_conv_silu_with_state, causal_grouped_query_attention,
    gated_delta_recurrent_with_state, rms_norm_heads, rms_norm_zero_centered, split_query_gate,
    Qwen35Geometry,
};

use super::head::{JointSchemaHead, LexicalLookup};
use super::renderer::EncodedRecord;
use super::{ClefProfile, JOINT_HEAD_CONFIG};

#[derive(Debug, Deserialize)]
struct PinnedTextConfig {
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct PinnedConfig {
    text_config: PinnedTextConfig,
}

/// One projection weight: K-quantized through `QMatMul`, or a plain F32
/// tensor for the small SSM projections llama.cpp keeps unquantized.
enum Weight {
    Quantized(QMatMul),
    Dense(Tensor),
}

impl Weight {
    #[allow(clippy::trait_duplication_in_bounds)]
    fn forward(&self, input: &Tensor) -> Result<Tensor, FamilyError> {
        Ok(match self {
            Self::Quantized(matmul) => matmul.forward(input)?,
            Self::Dense(tensor) => {
                if tensor.dtype() != input.dtype() {
                    input.matmul(&tensor.to_dtype(input.dtype())?.t()?)?
                } else {
                    input.matmul(&tensor.t()?)?
                }
            }
        })
    }
}

/// Flatten one tensor to F32 row-major values.
fn values(tensor: &Tensor) -> Result<Vec<f32>, FamilyError> {
    Ok(tensor.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
}

/// Loaded GGUF quantized Clef model.
pub struct GgufModel {
    embedding: Tensor,
    lm_head: Tensor,
    final_norm: Vec<f32>,
    layers: Vec<DecoderLayer>,
    head: JointSchemaHead,
    geometry: Qwen35Geometry,
    hidden_size: usize,
    device: Device,
}

enum Mixer {
    Linear(LinearAttention),
    Full(FullAttention),
}

struct DecoderLayer {
    input_layernorm: Vec<f32>,
    mixer: Mixer,
    post_attention_layernorm: Vec<f32>,
    mlp_gate: Weight,
    mlp_up: Weight,
    mlp_down: Weight,
}

struct LinearAttention {
    qkv: Weight,
    z: Weight,
    beta: Weight,
    decay: Weight,
    out: Weight,
    /// Channels-major causal conv taps `[qkv_size, kernel]`.
    conv1d: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    delta_norm: Vec<f32>,
}

struct FullAttention {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl GgufModel {
    /// Load and verify the pinned GGUF profile and official joint head.
    pub fn load(
        model_root: impl AsRef<Path>,
        profile: &'static ClefProfile,
    ) -> Result<Self, FamilyError> {
        let model_root = model_root.as_ref();
        for (name, digest, bytes) in profile.checkpoint_shards {
            let path = model_root.join(name);
            let length = std::fs::metadata(&path)
                .map_err(|source| FamilyError::Io {
                    path: path.clone(),
                    source,
                })?
                .len();
            if length != *bytes {
                return Err(FamilyError::ContractMismatch {
                    field: "checkpoint_size",
                    expected: format!("{name} is {bytes} bytes"),
                    actual: format!("{length} bytes"),
                });
            }
            verify_digest(&path, digest)?;
        }
        let (head_name, head_digest, head_bytes) = profile.joint_head;
        let head_path = model_root.join(head_name);
        let head_length = std::fs::metadata(&head_path)
            .map_err(|source| FamilyError::Io {
                path: head_path.clone(),
                source,
            })?
            .len();
        if head_length != head_bytes {
            return Err(FamilyError::ContractMismatch {
                field: "joint_head_size",
                expected: format!("{head_name} is {head_bytes} bytes"),
                actual: format!("{head_length} bytes"),
            });
        }
        verify_digest(&head_path, head_digest)?;
        verify_digest(
            &model_root.join(profile.tokenizer_path),
            profile.tokenizer_json_sha256,
        )?;
        verify_digest(
            &model_root.join(profile.config_path),
            profile.config_json_sha256,
        )?;
        let config: PinnedConfig = read_json(&model_root.join(profile.config_path))?;
        let vocab = config.text_config.vocab_size;
        let geometry = profile.geometry;
        let device = Device::Cpu;

        let gguf_path = model_root.join(profile.checkpoint_shards[0].0);
        let mut reader = std::fs::File::open(&gguf_path).map_err(|source| FamilyError::Io {
            path: gguf_path.clone(),
            source,
        })?;
        let content = gguf::Content::read(&mut reader).map_err(FamilyError::Candle)?;
        let reader = std::cell::RefCell::new(reader);
        let qtensor = |name: &str| -> Result<Arc<QTensor>, FamilyError> {
            Ok(Arc::new(
                content
                    .tensor(&mut *reader.borrow_mut(), name, &Device::Cpu)
                    .map_err(FamilyError::Candle)?,
            ))
        };
        let _ = &device;

        // Input embedding and untied output embedding, dequantized to BF16
        // so row gathers stay cheap without widening the whole table.
        let embedding = dequantized_embedding(&qtensor, "token_embd.weight", vocab, &geometry)?;
        let lm_head = dequantized_embedding(&qtensor, "output.weight", vocab, &geometry)?;

        let block_count = content
            .metadata
            .get("qwen35.block_count")
            .and_then(|value| match value {
                gguf::Value::U32(value) => Some(*value as usize),
                gguf::Value::U64(value) => usize::try_from(*value).ok(),
                _ => None,
            })
            .unwrap_or(geometry.layer_count);
        if block_count != geometry.layer_count {
            return Err(FamilyError::ContractMismatch {
                field: "block_count",
                expected: geometry.layer_count.to_string(),
                actual: block_count.to_string(),
            });
        }

        let mut layers = Vec::with_capacity(block_count);
        for index in 0..block_count {
            layers.push(load_layer(&qtensor, index, geometry)?);
        }
        let final_norm = {
            let tensor = qtensor("output_norm.weight")?;
            let tensor = tensor.dequantize(&device)?;
            if tensor.dims() != [geometry.hidden_size] {
                return Err(FamilyError::ContractMismatch {
                    field: "output_norm_shape",
                    expected: format!("[{}]", geometry.hidden_size),
                    actual: format!("{:?}", tensor.dims()),
                });
            }
            values(&tensor)?
        };
        let head = JointSchemaHead::load(&JOINT_HEAD_CONFIG, &head_path, &device)?;
        Ok(Self {
            embedding,
            lm_head,
            final_norm,
            layers,
            head,
            geometry,
            hidden_size: geometry.hidden_size,
            device,
        })
    }

    fn embed(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let ids = Tensor::from_vec(input_ids.to_vec(), input_ids.len(), &self.device)?;
        let rows = self
            .embedding
            .to_dtype(DType::F32)?
            .index_select(&ids, 0)?
            .to_vec2::<f32>()?;
        Ok(rows.into_iter().flatten().collect())
    }

    /// Run the full hybrid forward, returning final-norm hidden states.
    fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        if input_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "clef input must contain at least one token".into(),
            ));
        }
        let token_count = input_ids.len();
        let mut hidden = self.embed(input_ids)?;
        for layer in &self.layers {
            let normalized = rms_norm_zero_centered(
                &hidden,
                token_count,
                self.hidden_size,
                &layer.input_layernorm,
            );
            let attention = match &layer.mixer {
                Mixer::Linear(mixer) => {
                    mixer.forward(&normalized, token_count, self.geometry, &self.device)?
                }
                Mixer::Full(mixer) => {
                    mixer.forward(&normalized, token_count, self.geometry, &self.device)?
                }
            };
            let mut mixed = hidden;
            for (hidden_value, attention_value) in mixed.iter_mut().zip(&attention) {
                *hidden_value += attention_value;
            }
            let mlp_input = rms_norm_zero_centered(
                &mixed,
                token_count,
                self.hidden_size,
                &layer.post_attention_layernorm,
            );
            let mlp_input =
                Tensor::from_vec(mlp_input, (token_count, self.hidden_size), &self.device)?;
            let gate = candle_nn::ops::silu(&layer.mlp_gate.forward(&mlp_input)?)?;
            let up = layer.mlp_up.forward(&mlp_input)?;
            let activated = (gate * up)?;
            let mlp = layer.mlp_down.forward(&activated)?;
            let mlp = mlp.to_vec2::<f32>()?;
            for (hidden_row, mlp_row) in mixed.chunks_exact_mut(self.hidden_size).zip(&mlp) {
                for (value, mlp_value) in hidden_row.iter_mut().zip(mlp_row) {
                    *value += mlp_value;
                }
            }
            hidden = mixed;
        }
        Ok(rms_norm_zero_centered(
            &hidden,
            token_count,
            self.hidden_size,
            &self.final_norm,
        ))
    }
}

fn dequantized_embedding(
    qtensor: &dyn Fn(&str) -> Result<Arc<QTensor>, FamilyError>,
    name: &str,
    vocab: usize,
    geometry: &Qwen35Geometry,
) -> Result<Tensor, FamilyError> {
    let device = Device::Cpu;
    let rows = qtensor(name)?.dequantize(&device)?.to_dtype(DType::BF16)?;
    let dims = rows.dims().to_vec();
    if dims != [vocab, geometry.hidden_size] {
        return Err(FamilyError::ContractMismatch {
            field: "embedding_shape",
            expected: format!("[{vocab}, {}]", geometry.hidden_size),
            actual: format!("{dims:?}"),
        });
    }
    Ok(rows)
}

fn load_layer(
    qtensor: &dyn Fn(&str) -> Result<Arc<QTensor>, FamilyError>,
    index: usize,
    geometry: Qwen35Geometry,
) -> Result<DecoderLayer, FamilyError> {
    let prefix = format!("blk.{index}");
    let vector = |name: &str, width: usize| -> Result<Vec<f32>, FamilyError> {
        let tensor = qtensor(name)?.dequantize(&Device::Cpu)?;
        if tensor.dims() != [width] {
            return Err(FamilyError::ContractMismatch {
                field: "gguf_vector_shape",
                expected: format!("[{width}]"),
                actual: format!("{:?}", tensor.dims()),
            });
        }
        values(&tensor)
    };
    let matrix = |name: &str| -> Result<Weight, FamilyError> {
        Ok(Weight::Quantized(QMatMul::from_arc(qtensor(name)?)?))
    };
    let mixer = if geometry.is_full_attention(index) {
        Mixer::Full(FullAttention {
            q: matrix(&format!("{prefix}.attn_q.weight"))?,
            k: matrix(&format!("{prefix}.attn_k.weight"))?,
            v: matrix(&format!("{prefix}.attn_v.weight"))?,
            out: matrix(&format!("{prefix}.attn_output.weight"))?,
            q_norm: vector(
                &format!("{prefix}.attn_q_norm.weight"),
                geometry.attention_head_dim,
            )?,
            k_norm: vector(
                &format!("{prefix}.attn_k_norm.weight"),
                geometry.attention_head_dim,
            )?,
        })
    } else {
        Mixer::Linear(LinearAttention {
            qkv: matrix(&format!("{prefix}.attn_qkv.weight"))?,
            z: matrix(&format!("{prefix}.attn_gate.weight"))?,
            beta: matrix(&format!("{prefix}.ssm_beta.weight"))?,
            decay: matrix(&format!("{prefix}.ssm_alpha.weight"))?,
            out: matrix(&format!("{prefix}.ssm_out.weight"))?,
            // GGUF stores the conv kernel as ne=[kernel, channels]; candle
            // reverses that to [channels, kernel], matching the oracle's
            // channels-major layout.
            conv1d: {
                let tensor = qtensor(&format!("{prefix}.ssm_conv1d.weight"))?
                    .dequantize(&Device::Cpu)?;
                let dims = tensor.dims().to_vec();
                if dims != [geometry.qkv_size(), geometry.conv_kernel] {
                    return Err(FamilyError::ContractMismatch {
                        field: "ssm_conv1d_shape",
                        expected: format!("[{}, {}]", geometry.qkv_size(), geometry.conv_kernel),
                        actual: format!("{dims:?}"),
                    });
                }
                values(&tensor)?
            },
            dt_bias: vector(&format!("{prefix}.ssm_dt.bias"), geometry.value_heads)?,
            a_log: vector(&format!("{prefix}.ssm_a"), geometry.value_heads)?,
            delta_norm: vector(&format!("{prefix}.ssm_norm.weight"), geometry.head_dim)?,
        })
    };
    Ok(DecoderLayer {
        input_layernorm: vector(&format!("{prefix}.attn_norm.weight"), geometry.hidden_size)?,
        mixer,
        post_attention_layernorm: vector(
            &format!("{prefix}.post_attention_norm.weight"),
            geometry.hidden_size,
        )?,
        mlp_gate: matrix(&format!("{prefix}.ffn_gate.weight"))?,
        mlp_up: matrix(&format!("{prefix}.ffn_up.weight"))?,
        mlp_down: matrix(&format!("{prefix}.ffn_down.weight"))?,
    })
}

impl LinearAttention {
    fn forward(
        &self,
        normalized: &[f32],
        token_count: usize,
        geometry: Qwen35Geometry,
        device: &Device,
    ) -> Result<Vec<f32>, FamilyError> {
        let flatten = |tensor: Tensor| -> Result<Vec<f32>, FamilyError> {
            Ok(tensor.to_vec2::<f32>()?.into_iter().flatten().collect())
        };
        let normalized_tensor =
            Tensor::from_vec(normalized.to_vec(), (token_count, geometry.hidden_size), device)?;
        let raw_qkv = flatten(self.qkv.forward(&normalized_tensor)?)?;
        let (qkv, _conv_state) = causal_depthwise_conv_silu_with_state(
            &raw_qkv,
            token_count,
            geometry.qkv_size(),
            &self.conv1d,
            geometry.conv_kernel,
            None,
        );
        let z = flatten(self.z.forward(&normalized_tensor)?)?;
        let beta = flatten(self.beta.forward(&normalized_tensor)?)?;
        let decay = flatten(self.decay.forward(&normalized_tensor)?)?;
        let (mixed, _recurrent) = gated_delta_recurrent_with_state(
            &qkv,
            &z,
            &beta,
            &decay,
            &self.dt_bias,
            &self.a_log,
            &self.delta_norm,
            token_count,
            geometry,
            None,
        );
        let mixed = Tensor::from_vec(mixed, (token_count, geometry.value_size()), device)?;
        Ok(flatten(self.out.forward(&mixed)?)?)
    }
}

impl FullAttention {
    fn forward(
        &self,
        normalized: &[f32],
        token_count: usize,
        geometry: Qwen35Geometry,
        device: &Device,
    ) -> Result<Vec<f32>, FamilyError> {
        let flatten = |tensor: Tensor| -> Result<Vec<f32>, FamilyError> {
            Ok(tensor.to_vec2::<f32>()?.into_iter().flatten().collect())
        };
        let normalized_tensor =
            Tensor::from_vec(normalized.to_vec(), (token_count, geometry.hidden_size), device)?;
        let projected_q = flatten(self.q.forward(&normalized_tensor)?)?;
        let projected_k = flatten(self.k.forward(&normalized_tensor)?)?;
        let projected_v = flatten(self.v.forward(&normalized_tensor)?)?;
        let (mut queries, gates) = split_query_gate(&projected_q, token_count, geometry);
        let mut keys = projected_k;
        rms_norm_heads(&mut queries, geometry.attention_heads, &self.q_norm);
        rms_norm_heads(&mut keys, geometry.kv_heads, &self.k_norm);
        apply_rotary(
            &mut queries,
            geometry.attention_heads,
            token_count,
            0,
            geometry,
        );
        apply_rotary(&mut keys, geometry.kv_heads, token_count, 0, geometry);
        let mut mixed = causal_grouped_query_attention(
            &queries,
            &keys,
            &projected_v,
            token_count,
            0,
            geometry,
        );
        for (value, gate) in mixed.iter_mut().zip(gates) {
            *value *= sigmoid(gate);
        }
        let mixed = Tensor::from_vec(mixed, (token_count, geometry.attention_size()), device)?;
        Ok(flatten(self.out.forward(&mixed)?)?)
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

impl super::model::ClefExecutionModel for GgufModel {
    fn evaluate_record(
        &self,
        encoded: &EncodedRecord,
        control: &FamilyControl,
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        control.check()?;
        let hidden = self.forward_hidden(&encoded.input_ids)?;
        let token_count = encoded.input_ids.len();
        if hidden.len() != token_count * self.hidden_size {
            return Err(FamilyError::InvalidInput(
                "gguf backbone output width does not match the pinned geometry".into(),
            ));
        }
        control.check()?;
        let lexical = GgufLexical {
            lm_head: &self.lm_head,
            hidden_size: self.hidden_size,
            device: &self.device,
        };
        let result = self
            .head
            .forward(&hidden, token_count, encoded, &lexical, &self.device)?;
        control.check()?;
        Ok(result)
    }
}

struct GgufLexical<'a> {
    lm_head: &'a Tensor,
    hidden_size: usize,
    device: &'a Device,
}

impl LexicalLookup for GgufLexical<'_> {
    fn rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = Tensor::from_vec(token_ids.to_vec(), token_ids.len(), self.device)?;
        let rows = self
            .lm_head
            .to_dtype(DType::F32)?
            .index_select(&ids, 0)?
            .to_vec2::<f32>()?;
        Ok(rows.into_iter().flatten().collect())
    }
}
