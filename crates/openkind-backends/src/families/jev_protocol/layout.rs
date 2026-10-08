//! Checkpoint contract of the JEV-27B-VL 8-bit MLX profile: the parsed root
//! `config.json` and the exact tensor names and shapes the loader expects.
//!
//! Nothing here needs MLX, so the contract is checked offline against the
//! pinned checkpoint's real tensor headers on every host.

use serde::Deserialize;

use crate::families::support::FamilyError;
use crate::qwen35::Qwen35Geometry;

use super::{DecisionConfig, Protocol};

/// Affine quantization parameters shared by every quantized tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantParams {
    /// Elements per scale/bias group.
    pub group_size: i32,
    /// Bits per packed element.
    pub bits: i32,
}

/// Tensor-name prefix of the decoder in the converted checkpoint.
pub const DECODER_PREFIX: &str = "language_model.model";
/// Tensor name of the unquantized output embedding.
pub const LM_HEAD: &str = "language_model.lm_head.weight";
/// Tensors under this prefix belong to the vision tower and are not loaded.
pub const VISION_PREFIX: &str = "vision_tower.";

#[derive(Deserialize)]
struct RawQuantization {
    group_size: i32,
    bits: i32,
    mode: String,
}

#[derive(Deserialize)]
struct RawRope {
    rope_theta: f64,
    partial_rotary_factor: f64,
}

#[derive(Deserialize)]
struct RawText {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    full_attention_interval: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    attn_output_gate: bool,
    tie_word_embeddings: bool,
    rope_parameters: RawRope,
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    quantization: RawQuantization,
    text_config: RawText,
    decision_config: serde_json::Value,
}

/// The validated contents of the pinned root `config.json`.
#[derive(Debug, Clone)]
pub struct JevConfig {
    /// Text decoder geometry.
    pub geometry: Qwen35Geometry,
    /// Quantization of every quantized tensor.
    pub quant: QuantParams,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// The JEV readout configuration.
    pub decision: DecisionConfig,
}

impl JevConfig {
    /// Parse and validate the root `config.json` bytes. Every field the
    /// forward depends on must equal the pinned 27B Qwen3.5 geometry; a
    /// different checkpoint fails closed instead of running wrong equations.
    pub fn parse(bytes: &[u8]) -> Result<Self, FamilyError> {
        let raw: RawConfig = serde_json::from_slice(bytes).map_err(|error| {
            FamilyError::InvalidInput(format!("config.json does not parse: {error}"))
        })?;
        let text = &raw.text_config;
        let expected = Qwen35Geometry::CLEF;
        let geometry = Qwen35Geometry {
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            layer_count: text.num_hidden_layers,
            full_attention_interval: text.full_attention_interval,
            key_heads: text.linear_num_key_heads,
            value_heads: text.linear_num_value_heads,
            head_dim: text.linear_value_head_dim,
            conv_kernel: text.linear_conv_kernel_dim,
            attention_heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            attention_head_dim: text.head_dim,
            rms_epsilon: text.rms_norm_eps as f32,
        };
        let supported = raw.model_type == "jev"
            && geometry == expected
            && text.linear_key_head_dim == text.linear_value_head_dim
            && text.attn_output_gate
            && !text.tie_word_embeddings
            && text.rope_parameters.rope_theta == f64::from(Qwen35Geometry::ROPE_THETA)
            && text.rope_parameters.partial_rotary_factor == 0.25
            && raw.quantization.mode == "affine"
            && raw.quantization.bits == 8
            && raw.quantization.group_size == 64;
        if !supported {
            return Err(FamilyError::ContractMismatch {
                field: "config.json",
                expected: "jev, affine 8-bit group 64, untied, gated, Qwen3.5 27B geometry".into(),
                actual: format!(
                    "{} {:?} bits {} group {}",
                    raw.model_type, geometry, raw.quantization.bits, raw.quantization.group_size
                ),
            });
        }
        let decision = DecisionConfig::from_json(
            Protocol::Jev,
            &serde_json::to_vec(&raw.decision_config).map_err(|error| {
                FamilyError::InvalidInput(format!("decision_config does not serialize: {error}"))
            })?,
        )?;
        Ok(Self {
            geometry,
            quant: QuantParams {
                group_size: raw.quantization.group_size,
                bits: raw.quantization.bits,
            },
            vocab_size: text.vocab_size,
            decision,
        })
    }
}

/// How a tensor is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Storage {
    /// Affine triple; `weight` is `[rows, cols * bits / 32]` `U32`,
    /// `scales` and `biases` are `[rows, cols / group_size]`.
    Quantized {
        /// Output rows.
        rows: usize,
        /// Logical input columns.
        cols: usize,
    },
    /// A single dense tensor of this shape.
    Dense(Vec<usize>),
}

/// One tensor the loader requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    /// Name of the `weight` (or only) tensor.
    pub name: String,
    /// Storage layout.
    pub storage: Storage,
}

/// Every tensor the text decoder needs, in a stable order.
pub fn expected_tensors(geometry: &Qwen35Geometry, vocab_size: usize) -> Vec<ExpectedTensor> {
    let g = geometry;
    let mut tensors = Vec::new();
    let mut quantized = |name: String, rows: usize, cols: usize| {
        tensors.push(ExpectedTensor {
            name,
            storage: Storage::Quantized { rows, cols },
        });
    };
    quantized(
        format!("{DECODER_PREFIX}.embed_tokens.weight"),
        vocab_size,
        g.hidden_size,
    );
    for layer in 0..g.layer_count {
        let p = format!("{DECODER_PREFIX}.layers.{layer}");
        if g.is_full_attention(layer) {
            quantized(
                format!("{p}.self_attn.q_proj.weight"),
                g.attention_size() * 2,
                g.hidden_size,
            );
            quantized(
                format!("{p}.self_attn.k_proj.weight"),
                g.kv_size(),
                g.hidden_size,
            );
            quantized(
                format!("{p}.self_attn.v_proj.weight"),
                g.kv_size(),
                g.hidden_size,
            );
            quantized(
                format!("{p}.self_attn.o_proj.weight"),
                g.hidden_size,
                g.attention_size(),
            );
        } else {
            quantized(
                format!("{p}.linear_attn.in_proj_qkv.weight"),
                g.qkv_size(),
                g.hidden_size,
            );
            quantized(
                format!("{p}.linear_attn.in_proj_z.weight"),
                g.value_size(),
                g.hidden_size,
            );
            quantized(
                format!("{p}.linear_attn.in_proj_b.weight"),
                g.value_heads,
                g.hidden_size,
            );
            quantized(
                format!("{p}.linear_attn.in_proj_a.weight"),
                g.value_heads,
                g.hidden_size,
            );
            quantized(
                format!("{p}.linear_attn.out_proj.weight"),
                g.hidden_size,
                g.value_size(),
            );
        }
        quantized(
            format!("{p}.mlp.gate_proj.weight"),
            g.intermediate_size,
            g.hidden_size,
        );
        quantized(
            format!("{p}.mlp.up_proj.weight"),
            g.intermediate_size,
            g.hidden_size,
        );
        quantized(
            format!("{p}.mlp.down_proj.weight"),
            g.hidden_size,
            g.intermediate_size,
        );
    }
    let mut dense = |name: String, shape: Vec<usize>| {
        tensors.push(ExpectedTensor {
            name,
            storage: Storage::Dense(shape),
        });
    };
    for layer in 0..g.layer_count {
        let p = format!("{DECODER_PREFIX}.layers.{layer}");
        dense(format!("{p}.input_layernorm.weight"), vec![g.hidden_size]);
        dense(
            format!("{p}.post_attention_layernorm.weight"),
            vec![g.hidden_size],
        );
        if g.is_full_attention(layer) {
            dense(
                format!("{p}.self_attn.q_norm.weight"),
                vec![g.attention_head_dim],
            );
            dense(
                format!("{p}.self_attn.k_norm.weight"),
                vec![g.attention_head_dim],
            );
        } else {
            dense(
                format!("{p}.linear_attn.conv1d.weight"),
                vec![g.qkv_size(), g.conv_kernel, 1],
            );
            dense(format!("{p}.linear_attn.A_log"), vec![g.value_heads]);
            dense(format!("{p}.linear_attn.dt_bias"), vec![g.value_heads]);
            dense(format!("{p}.linear_attn.norm.weight"), vec![g.head_dim]);
        }
    }
    dense(format!("{DECODER_PREFIX}.norm.weight"), vec![g.hidden_size]);
    dense(LM_HEAD.to_owned(), vec![vocab_size, g.hidden_size]);
    tensors
}
