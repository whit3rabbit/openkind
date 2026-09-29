//! MLX/Metal execution backend for the pinned BERT embedding checkpoint
//! (feature `mlx`, macOS arm64).
//!
//! Mirrors the candle CPU arithmetic in [`super::bert_encoder`] op for op:
//! FP32 compute over the FP32-stored shard, classic post-LN BERT encoder,
//! CLS pooling, and L2-normalized outputs, so both backends produce the
//! same embedding for the same text (the candle CPU path remains the
//! correctness oracle). The encoder `id` pins the backend and the
//! checkpoint content hash.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::io::{Read, Seek};
use std::sync::Arc;

use mlx_rs::ops;
use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::families::support::FamilyError;
use crate::qwen35::mlx::{MlxError, MlxRuntime, MlxRuntimeConfig};

use super::bert_encoder::{BertEmbedderArtifacts, MAX_LENGTH};
use super::encoder::{l2_normalize, TextEmbedder};
use super::ProxyCacheError;

impl From<MlxError> for ProxyCacheError {
    fn from(error: MlxError) -> Self {
        ProxyCacheError::Encoder(error.to_string())
    }
}

impl From<FamilyError> for ProxyCacheError {
    fn from(error: FamilyError) -> Self {
        ProxyCacheError::Encoder(error.to_string())
    }
}

/// Map an operation name onto an [`MlxError::Operation`] constructor.
fn op<E: std::fmt::Display>(name: &'static str) -> impl Fn(E) -> MlxError {
    move |error| MlxError::Operation {
        operation: name,
        message: error.to_string(),
    }
}

/// One-element FP32 array used as a broadcast scalar.
fn scalar(value: f32) -> Array {
    Array::from_slice(&[value], &[1])
}

/// Biased LayerNorm over the last axis (`nn.LayerNorm` semantics).
fn layer_norm(input: &Array, weight: &Array, bias: &Array, eps: f64) -> Result<Array, MlxError> {
    let centered = input
        .subtract(&input.mean_axis(-1, true).map_err(op("norm mean"))?)
        .map_err(op("norm center"))?;
    let variance = centered
        .square()
        .map_err(op("norm square"))?
        .mean_axis(-1, true)
        .map_err(op("norm variance"))?;
    let normalized = centered
        .divide(
            &variance
                .add(scalar(eps as f32))
                .map_err(op("norm eps"))?
                .sqrt()
                .map_err(op("norm sqrt"))?,
        )
        .map_err(op("norm normalize"))?;
    normalized
        .multiply(weight)
        .map_err(op("norm scale"))?
        .add(bias)
        .map_err(op("norm bias"))
}

/// Exact erf-based GELU, matching candle's `Activation::Gelu`.
fn gelu(input: &Array) -> Result<Array, MlxError> {
    let half = input.multiply(scalar(0.5)).map_err(op("gelu scale"))?;
    let erf_arg = input
        .multiply(scalar(std::f32::consts::FRAC_1_SQRT_2))
        .map_err(op("gelu arg"))?;
    let erf = ops::erf(&erf_arg).map_err(op("gelu erf"))?;
    half.multiply(&erf.add(scalar(1.0)).map_err(op("gelu one"))?)
        .map_err(op("gelu product"))
}

/// Pinned BERT geometry, checked against `config.json` at load.
#[derive(Debug, Clone)]
pub struct BertMlxConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden width (embedding dim).
    pub hidden_size: usize,
    /// Encoder layer count.
    pub num_hidden_layers: usize,
    /// Attention heads.
    pub num_attention_heads: usize,
    /// Feed-forward width.
    pub intermediate_size: usize,
    /// Position budget.
    pub max_position_embeddings: usize,
    /// Token-type embedding rows.
    pub type_vocab_size: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// Padding token id.
    pub pad_token_id: u32,
}

impl BertMlxConfig {
    /// Decode and contract-check the model's `config.json`.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, MlxError> {
        let field = |name: &str| -> Result<i64, MlxError> {
            value
                .get(name)
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| MlxError::InvalidState(format!("bert config is missing {name}")))
        };
        Ok(Self {
            vocab_size: field("vocab_size")? as usize,
            hidden_size: field("hidden_size")? as usize,
            num_hidden_layers: field("num_hidden_layers")? as usize,
            num_attention_heads: field("num_attention_heads")? as usize,
            intermediate_size: field("intermediate_size")? as usize,
            max_position_embeddings: field("max_position_embeddings")? as usize,
            type_vocab_size: field("type_vocab_size")? as usize,
            layer_norm_eps: value
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-12),
            pad_token_id: field("pad_token_id")? as u32,
        })
    }
}

/// Raw safetensors tensor metadata.
#[derive(Debug, serde::Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Sequential reader over the FP32 shard.
struct Shard {
    file: std::fs::File,
    data_base: u64,
    tensors: std::collections::BTreeMap<String, RawTensor>,
}

impl Shard {
    fn open(path: &std::path::Path) -> Result<Self, MlxError> {
        let mut file = std::fs::File::open(path).map_err(|error| {
            MlxError::InvalidState(format!(
                "open pinned embedding checkpoint {}: {error}",
                path.display()
            ))
        })?;
        let mut length_bytes = [0_u8; 8];
        file.read_exact(&mut length_bytes)
            .map_err(|error| MlxError::InvalidState(format!("read safetensors header: {error}")))?;
        let header_len = u64::from_le_bytes(length_bytes);
        let mut header = vec![0_u8; header_len as usize];
        file.read_exact(&mut header)
            .map_err(|error| MlxError::InvalidState(format!("read safetensors header: {error}")))?;
        let mut tensors: std::collections::BTreeMap<String, RawTensor> =
            serde_json::from_slice(&header).map_err(|error| {
                MlxError::InvalidState(format!("decode safetensors header: {error}"))
            })?;
        tensors.remove("__metadata__");
        Ok(Self {
            file,
            data_base: 8 + header_len,
            tensors,
        })
    }

    /// Read one F32 tensor with the pinned shape; missing, mistyped, or
    /// reshaped tensors fail closed.
    fn f32_tensor(&mut self, name: &str, expected: &[usize]) -> Result<Array, MlxError> {
        let raw = self.tensors.get(name).ok_or_else(|| {
            MlxError::InvalidState(format!(
                "pinned embedding checkpoint is missing tensor `{name}`"
            ))
        })?;
        if raw.dtype != "F32" {
            return Err(MlxError::InvalidState(format!(
                "tensor `{name}` stores {} but the pinned embedding shard is FP32",
                raw.dtype
            )));
        }
        if raw.shape != expected {
            return Err(MlxError::InvalidState(format!(
                "tensor `{name}` has shape {:?} but the pinned contract requires {:?}",
                raw.shape, expected
            )));
        }
        let byte_len = (raw.data_offsets[1] - raw.data_offsets[0]) as usize;
        let mut bytes = vec![0_u8; byte_len];
        self.file
            .seek(std::io::SeekFrom::Start(
                self.data_base + raw.data_offsets[0],
            ))
            .map_err(|error| MlxError::InvalidState(format!("seek `{name}`: {error}")))?;
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| MlxError::InvalidState(format!("read `{name}`: {error}")))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let shape: Vec<i32> = expected.iter().map(|&dim| dim as i32).collect();
        Ok(Array::from_slice(&values, &shape))
    }

    /// Read one F32 matrix and materialize its transpose so every forward
    /// matmul consumes `(input, output)` weights directly.
    fn transposed_f32_tensor(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Array, MlxError> {
        let weight = self.f32_tensor(name, &[rows, cols])?;
        let transposed = weight
            .transpose_axes(&[1, 0])
            .map_err(op("weight transpose"))?;
        transposed.eval().map_err(op("weight transpose eval"))?;
        Ok(transposed)
    }
}

/// One post-LN BERT encoder layer.
struct BertLayer {
    query: (Array, Array),
    key: (Array, Array),
    value: (Array, Array),
    attn_out: (Array, Array),
    attn_norm: (Array, Array),
    intermediate: (Array, Array),
    output: (Array, Array),
    output_norm: (Array, Array),
}

impl BertLayer {
    fn load(shard: &mut Shard, config: &BertMlxConfig, index: usize) -> Result<Self, MlxError> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let prefix = format!("encoder.layer.{index}");
        let mut linear = |suffix: &str, out: usize| -> Result<(Array, Array), MlxError> {
            Ok((
                shard.transposed_f32_tensor(&format!("{prefix}.{suffix}.weight"), out, hidden)?,
                shard.f32_tensor(&format!("{prefix}.{suffix}.bias"), &[out])?,
            ))
        };
        Ok(Self {
            query: linear("attention.self.query", hidden)?,
            key: linear("attention.self.key", hidden)?,
            value: linear("attention.self.value", hidden)?,
            attn_out: linear("attention.output.dense", hidden)?,
            attn_norm: (
                shard.f32_tensor(
                    &format!("{prefix}.attention.output.LayerNorm.weight"),
                    &[hidden],
                )?,
                shard.f32_tensor(
                    &format!("{prefix}.attention.output.LayerNorm.bias"),
                    &[hidden],
                )?,
            ),
            intermediate: (
                shard.transposed_f32_tensor(
                    &format!("{prefix}.intermediate.dense.weight"),
                    intermediate,
                    hidden,
                )?,
                shard.f32_tensor(
                    &format!("{prefix}.intermediate.dense.bias"),
                    &[intermediate],
                )?,
            ),
            output: (
                shard.transposed_f32_tensor(
                    &format!("{prefix}.output.dense.weight"),
                    hidden,
                    intermediate,
                )?,
                shard.f32_tensor(&format!("{prefix}.output.dense.bias"), &[hidden])?,
            ),
            output_norm: (
                shard.f32_tensor(&format!("{prefix}.output.LayerNorm.weight"), &[hidden])?,
                shard.f32_tensor(&format!("{prefix}.output.LayerNorm.bias"), &[hidden])?,
            ),
        })
    }

    fn linear_forward(
        input: &Array,
        weight: &Array,
        bias: &Array,
        name: &'static str,
    ) -> Result<Array, MlxError> {
        input
            .matmul(weight)
            .map_err(op(name))?
            .add(bias)
            .map_err(op(format!("{name} bias").leak()))
    }

    fn forward(
        &self,
        xs: &Array,
        mask: &Array,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Array, MlxError> {
        let shape = xs.shape();
        let seq = shape[0];
        let hidden = shape[1];
        let project =
            |(weight, bias): &(Array, Array), name: &'static str| -> Result<Array, MlxError> {
                Self::linear_forward(xs, weight, bias, name)
            };
        let q = project(&self.query, "attn q")?;
        let k = project(&self.key, "attn k")?;
        let v = project(&self.value, "attn v")?;
        let head_shape = [seq, num_heads as i32, head_dim as i32];
        let reshape_heads = |x: Array, name: &'static str| -> Result<Array, MlxError> {
            x.reshape(&head_shape)
                .map_err(op(name))?
                .transpose_axes(&[1, 0, 2])
                .map_err(op("attn heads transpose"))
        };
        let q = reshape_heads(q, "attn q reshape")?;
        let k = reshape_heads(k, "attn k reshape")?;
        let v = reshape_heads(v, "attn v reshape")?;

        let scale = (head_dim as f64).sqrt() as f32;
        let scores = q
            .matmul(&k.transpose_axes(&[0, 2, 1]).map_err(op("attn k^T"))?)
            .map_err(op("attn scores"))?
            .divide(scalar(scale))
            .map_err(op("attn scale"))?
            .add(mask)
            .map_err(op("attn mask add"))?;
        let weights = ops::softmax_axis(&scores, -1, None).map_err(op("attn softmax"))?;
        let attended = weights
            .matmul(&v)
            .map_err(op("attn value"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("attn transpose"))?
            .reshape(&[seq, hidden])
            .map_err(op("attn merge heads"))?;

        // Post-LN: dense → residual → LayerNorm.
        let attn = Self::linear_forward(&attended, &self.attn_out.0, &self.attn_out.1, "attn out")?;
        let xs = xs.add(&attn).map_err(op("attn residual"))?;
        let xs = layer_norm(&xs, &self.attn_norm.0, &self.attn_norm.1, 1e-12)?;

        let mid = Self::linear_forward(
            &xs,
            &self.intermediate.0,
            &self.intermediate.1,
            "intermediate",
        )?;
        let mid = gelu(&mid)?;
        let out = Self::linear_forward(&mid, &self.output.0, &self.output.1, "output dense")?;
        let xs = xs.add(&out).map_err(op("output residual"))?;
        layer_norm(&xs, &self.output_norm.0, &self.output_norm.1, 1e-12)
    }
}

/// The loaded MLX BERT encoder.
pub struct MlxBertModel {
    config: BertMlxConfig,
    /// Embedding tables stay on the Rust side: the gather is a plain copy
    /// and indexing MLX arrays per token would be far slower.
    word_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    token_type_embeddings: Vec<f32>,
    embedding_norm: (Array, Array),
    layers: Vec<BertLayer>,
}

// SAFETY: every array evaluation in this model's methods runs under the
// process-wide `MlxRuntime` execution mutex (see the qwen35 `mlx` module
// docs); outside the lock, array handles are immutable refcounted values
// whose handle-only operations are safe concurrently. This is the same
// soundness argument as `unsafe impl Sync for LayaMlxModel`, and it lets
// the embedder serve concurrent requests from the blocking pool.
unsafe impl Send for MlxBertModel {}
unsafe impl Sync for MlxBertModel {}

impl MlxBertModel {
    /// Load and verify the pinned checkpoint (in place; nothing is copied).
    pub fn load(artifacts: &BertEmbedderArtifacts) -> Result<Self, MlxError> {
        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&artifacts.config_json).map_err(|error| {
                MlxError::InvalidState(format!("read {}: {error}", artifacts.config_json.display()))
            })?,
        )
        .map_err(|error| MlxError::InvalidState(format!("decode config.json: {error}")))?;
        let config = BertMlxConfig::from_json(&config_json)?;
        let mut shard = Shard::open(&artifacts.checkpoint)?;
        let hidden = config.hidden_size;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(BertLayer::load(&mut shard, &config, index)?);
        }
        let word_embeddings = shard.f32_tensor(
            "embeddings.word_embeddings.weight",
            &[config.vocab_size, hidden],
        )?;
        let position_embeddings = shard.f32_tensor(
            "embeddings.position_embeddings.weight",
            &[config.max_position_embeddings, hidden],
        )?;
        let token_type_embeddings = shard.f32_tensor(
            "embeddings.token_type_embeddings.weight",
            &[config.type_vocab_size, hidden],
        )?;
        let word_vec: Vec<f32> = word_embeddings.as_slice::<f32>().to_vec();
        let position_vec: Vec<f32> = position_embeddings.as_slice::<f32>().to_vec();
        let type_vec: Vec<f32> = token_type_embeddings.as_slice::<f32>().to_vec();
        Ok(Self {
            word_embeddings: word_vec,
            position_embeddings: position_vec,
            token_type_embeddings: type_vec,
            embedding_norm: (
                shard.f32_tensor("embeddings.LayerNorm.weight", &[hidden])?,
                shard.f32_tensor("embeddings.LayerNorm.bias", &[hidden])?,
            ),
            layers,
            config,
        })
    }

    /// Embedding lookup + LayerNorm for one `(seq,)` token row.
    fn embed(&self, token_ids: &[u32], type_ids: &[u32]) -> Result<Array, MlxError> {
        let seq = token_ids.len();
        let hidden = self.config.hidden_size;
        let mut embedded = Vec::with_capacity(seq * hidden);
        for (position, (&token, &type_id)) in token_ids.iter().zip(type_ids).enumerate() {
            let token_row = token as usize * hidden;
            let type_row = type_id as usize * hidden;
            let position_row = position * hidden;
            for column in 0..hidden {
                embedded.push(
                    self.word_embeddings[token_row + column]
                        + self.position_embeddings[position_row + column]
                        + self.token_type_embeddings[type_row + column],
                );
            }
        }
        let shape = [seq as i32, hidden as i32];
        let summed = Array::from_slice(&embedded, &shape);
        layer_norm(
            &summed,
            &self.embedding_norm.0,
            &self.embedding_norm.1,
            1e-12,
        )
    }

    /// Forward one row to final hidden states, returning the CLS position.
    fn forward_cls(&self, token_ids: &[u32], type_ids: &[u32]) -> Result<Vec<f32>, MlxError> {
        let seq = token_ids.len();
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.hidden_size / num_heads;
        let mut xs = self.embed(token_ids, type_ids)?;

        // Additive attention mask: 0 on real tokens, -inf on padding.
        let mut mask_values = vec![0.0_f32; seq];
        for (index, &token) in token_ids.iter().enumerate() {
            if token == self.config.pad_token_id {
                mask_values[index] = f32::NEG_INFINITY;
            }
        }
        let mask = Array::from_slice(&mask_values, &[seq as i32, 1]);

        for layer in &self.layers {
            xs = layer.forward(&xs, &mask, num_heads, head_dim)?;
        }
        xs.eval().map_err(op("final eval"))?;
        let hidden = self.config.hidden_size;
        let values = xs
            .try_as_slice::<f32>()
            .map_err(|error| op("read hidden states")(error.to_string()))?;
        Ok(values[..hidden].to_vec())
    }
}

/// MLX-backed BERT sentence embedder.
pub struct MlxBertEmbedder {
    runtime: Arc<MlxRuntime>,
    model: MlxBertModel,
    tokenizer: Tokenizer,
    id: String,
    dim: usize,
}

impl MlxBertEmbedder {
    /// Construct the runtime and load the verified checkpoint onto it.
    pub fn load(artifacts: &BertEmbedderArtifacts) -> Result<Self, ProxyCacheError> {
        let content_hash = artifacts.verify()?;
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let model = runtime.execute(|| MlxBertModel::load(artifacts))??;

        let mut tokenizer = Tokenizer::from_file(&artifacts.tokenizer_json).map_err(|error| {
            ProxyCacheError::Encoder(format!(
                "load tokenizer {}: {error}",
                artifacts.tokenizer_json.display()
            ))
        })?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_LENGTH.min(model.config.max_position_embeddings),
                ..Default::default()
            }))
            .map_err(|error| ProxyCacheError::Encoder(format!("set truncation: {error}")))?;

        let dim = model.config.hidden_size;
        let id = format!(
            "mlx-bert:{}:cls:{}:{content_hash}",
            artifacts.repository, MAX_LENGTH
        );
        Ok(Self {
            runtime,
            model,
            tokenizer,
            id,
            dim,
        })
    }
}

impl TextEmbedder for MlxBertEmbedder {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProxyCacheError> {
        let mut encodings = Vec::with_capacity(texts.len());
        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|error| ProxyCacheError::Encoder(format!("tokenize: {error}")))?;
            encodings.push(encoding);
        }
        // One row per MLX execution; batching across rows on the GPU is a
        // future optimization and must not change the arithmetic.
        let model = &self.model;
        let rows = self.runtime.execute(move || {
            let mut rows = Vec::with_capacity(encodings.len());
            for encoding in &encodings {
                let token_ids: Vec<u32> = encoding.get_ids().to_vec();
                let type_ids: Vec<u32> = encoding.get_type_ids().to_vec();
                let mut cls = model.forward_cls(&token_ids, &type_ids)?;
                l2_normalize(&mut cls);
                rows.push(cls);
            }
            Ok::<Vec<Vec<f32>>, MlxError>(rows)
        })??;
        Ok(rows)
    }

    fn backend_id(&self) -> &str {
        "encoder-embedding/mlx-fp32"
    }
}
