//! Shared MLX execution for dense Qwen decoder backends (feature `mlx`,
//! macOS arm64).
//!
//! Dense Qwen3 (per-head q/k RMSNorm before RoPE) and Qwen2 (no q/k norm)
//! checkpoints share one forward here, parameterized by the pinned config:
//! plain RMSNorm, grouped-query attention with HF `repeat_kv` interleaved
//! semantics, full-dimension NeoX rotary embedding at the profile's theta,
//! causal masking, SiLU-gated MLP, and a tied-embedding readout whose rows
//! stay host-resident (the decoder tensors widen BF16 to FP32 exactly, the
//! same values the candle oracles load). An optional PEFT LoRA adapter
//! merges host-side exactly like the CPU merges (`base + (B @ A) · scale`).
//!
//! Every MLX operation — model load and per-request forward — runs through
//! the process-wide [`MlxRuntime`] execution lock, per the workspace MLX
//! discipline.

use std::collections::BTreeMap;
use std::ops::{Div, Sub};
use std::path::PathBuf;
use std::sync::Arc;

use mlx_rs::Array;

use crate::families::qwen35_survey_mlx::{mlx_error, survey_array, visit_safetensors};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::runtime::MlxRuntime;
use crate::qwen35::mlx::weights::{load_lora_factors_for_dense, MlxSurveyLora};
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding};

/// One dense Qwen decoder's frozen geometry and flags.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseQwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Attention head width. Qwen3 pins this explicitly (`128`); Qwen2
    /// derives it from `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Qwen3 applies per-head q/k RMSNorm before RoPE; Qwen2 does not.
    pub qk_norm: bool,
    /// Maximum sequence length the RoPE tables are built for.
    pub max_sequence_tokens: usize,
}

/// A loaded dense Qwen model on MLX: FP32 arrays over the `model.layers.*`
/// decoder tensors plus the host-resident tied embedding.
pub(crate) struct DenseQwenMlxModel {
    /// Decoder tensors keyed by their suffix under the `model.` namespace
    /// (`layers.N.<rest>`, `norm.weight`).
    tensors: BTreeMap<String, Array>,
    embedding: Qwen35Embedding,
    config: DenseQwenConfig,
    /// Precomputed RoPE tables: `cos`/`sin` at `(max_sequence_tokens,
    /// head_dim / 2)`.
    rope_cos: Array,
    rope_sin: Array,
    runtime: Arc<MlxRuntime>,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared qwen35 MLX backbone).
unsafe impl Sync for DenseQwenMlxModel {}
unsafe impl Send for DenseQwenMlxModel {}

/// What one digest-verified dense Qwen load consumes.
pub(crate) struct DenseQwenLoadSpec<'a> {
    /// Digest-verified checkpoint shards, in load order.
    pub shards: &'a [PathBuf],
    /// Shard carrying the tied embedding table.
    pub embedding_shard: &'a PathBuf,
    /// Optional PEFT adapter: path, rank, and merge scale (`alpha / r`).
    /// Keys map onto the `model.layers.*` namespace.
    pub adapter: Option<(&'a PathBuf, usize, f64)>,
    pub config: DenseQwenConfig,
}

impl DenseQwenMlxModel {
    /// Load the verified shards into FP32 MLX arrays, merging the optional
    /// adapter exactly like the CPU oracle's merge. Must run inside
    /// [`MlxRuntime::execute`].
    pub(crate) fn load(
        spec: DenseQwenLoadSpec<'_>,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let config = spec.config;
        let factors = match spec.adapter {
            Some((path, rank, scale)) => {
                let lora = MlxSurveyLora {
                    adapter_path: path.clone(),
                    rank,
                    scale,
                };
                Some(load_lora_factors_for_dense(&lora)?)
            }
            None => None,
        };
        let keep = |name: &str| name.starts_with("model.layers.") || name == "model.norm.weight";
        let mut tensors = BTreeMap::new();
        for shard in spec.shards {
            visit_safetensors(shard, keep, |name, mut values, shape| {
                if let Some((_, _, scale)) = spec.adapter {
                    if let Some((a, b)) = factors.as_ref().and_then(|factors| factors.get(&name)) {
                        crate::qwen35::mlx::weights::apply_lora_delta_to_values(
                            &name,
                            &mut values,
                            a,
                            b,
                            scale,
                        )?;
                    }
                }
                let suffix = name.trim_start_matches("model.");
                let array = survey_array(&values, shape);
                array.eval().map_err(mlx_error("weight materialization"))?;
                if tensors.insert(suffix.to_owned(), array).is_some() {
                    return Err(FamilyError::InvalidInput(format!(
                        "duplicate tensor `{name}` across shards"
                    )));
                }
                Ok(())
            })?;
        }
        if tensors.is_empty() {
            return Err(FamilyError::InvalidInput(
                "checkpoint shards carry no `model.layers` decoder tensors".to_owned(),
            ));
        }
        // Every adapter target must have contributed a merged tensor.
        if let Some(factors) = &factors {
            for name in factors.keys() {
                let Some(suffix) = name.strip_prefix("model.") else {
                    return Err(FamilyError::InvalidInput(format!(
                        "adapter targets `{name}`, outside the checkpoint namespace"
                    )));
                };
                if !tensors.contains_key(suffix) {
                    return Err(FamilyError::InvalidInput(format!(
                        "adapter targets `{name}`, which the checkpoint layout does not load"
                    )));
                }
            }
        }
        let embedding = {
            let layout = EmbeddingLayout::read_named(
                spec.embedding_shard,
                "model.embed_tokens.weight",
                config.vocab_size,
                config.hidden_size,
            )
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
            Qwen35Embedding::from_layout(spec.embedding_shard.clone(), layout)
        };
        let (rope_cos, rope_sin) = rope_tables(&config);
        let model = Self {
            tensors,
            embedding,
            config,
            rope_cos,
            rope_sin,
            runtime: Arc::clone(runtime),
        };
        Ok(model)
    }

    /// The process-wide runtime this model evaluates on. Forwards must run
    /// inside `self.runtime().execute(...)`.
    pub(crate) fn runtime(&self) -> &Arc<MlxRuntime> {
        &self.runtime
    }

    /// Host-resident tied-embedding rows for `token_ids`, widened to FP32
    /// exactly like the CPU readouts.
    pub(crate) fn embed_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let rows = self
            .embedding
            .embed(token_ids)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        Ok(rows.values().to_vec())
    }

    /// Causal full-sequence forward, returning the final-normed hidden
    /// states as host FP32 row-major `[token_count, hidden]` values.
    ///
    /// Must run inside [`MlxRuntime::execute`].
    pub(crate) fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let config = self.config;
        let hidden_size = config.hidden_size;
        let token_count = input_ids.len();
        if token_count == 0 {
            return Err(FamilyError::InvalidInput(
                "dense Qwen prompts carry at least one token".to_owned(),
            ));
        }
        let embedded = self
            .embedding
            .embed(input_ids)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        let mut hidden =
            Array::from_slice(embedded.values(), &[token_count as i32, hidden_size as i32]);
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            let normalized = rms_norm(
                &hidden,
                self.get(&format!("{prefix}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            let attention = self.attention(&prefix, &normalized, token_count)?;
            let mixed = hidden.clone() + attention;
            let post = rms_norm(
                &mixed,
                self.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            let gate = self.project(&format!("{prefix}.mlp.gate_proj.weight"), &post)?;
            let up = self.project(&format!("{prefix}.mlp.up_proj.weight"), &post)?;
            let activated = if gate.dtype() == mlx_rs::Dtype::Float32
                && crate::qwen35::mlx::layers::fused_swiglu_enabled()
            {
                crate::qwen35::mlx::layers::fused_swiglu(&gate, &up)
                    .map_err(mlx_error("mlp swiglu"))?
            } else {
                mlx_rs::nn::silu(&gate).map_err(mlx_error("mlp silu"))? * up
            };
            let down = self.project(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            hidden = mixed + down;
            // Materialize per layer so no forward graph spans more than one
            // block.
            hidden.eval().map_err(mlx_error("layer materialization"))?;
        }
        let final_hidden = rms_norm(&hidden, self.get("norm.weight")?, config.rms_norm_eps)?;
        let flattened = final_hidden
            .reshape(&[(token_count * hidden_size) as i32])
            .map_err(mlx_error("final flatten"))?;
        flattened.eval().map_err(mlx_error("host eval"))?;
        flattened
            .to_vec_cast::<f32>()
            .map_err(|error| FamilyError::Mlx(format!("host read: {error}")))
    }

    fn get(&self, name: &str) -> Result<&Array, FamilyError> {
        self.tensors.get(name).ok_or_else(|| {
            FamilyError::InvalidInput(format!("tensor `{name}` missing from checkpoint"))
        })
    }

    /// `x @ w.T` over a dense FP32 projection.
    fn project(&self, name: &str, x: &Array) -> Result<Array, FamilyError> {
        let transposed = self.get(name)?.transpose().map_err(mlx_error("weight t"))?;
        x.matmul(&transposed).map_err(mlx_error("linear matmul"))
    }

    /// Causal grouped-query attention with per-head q/k RMSNorm (Qwen3) and
    /// full-dimension NeoX rotary embedding at absolute positions.
    fn attention(
        &self,
        prefix: &str,
        normalized: &Array,
        token_count: usize,
    ) -> Result<Array, FamilyError> {
        let config = self.config;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let [projected_q, keys, values] =
            qkv_projections(normalized, prefix, !config.qk_norm, |name| self.get(name))?;
        let query = projected_q
            .reshape(&[token_count as i32, heads as i32, head_dim as i32])
            .map_err(mlx_error("q reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("q heads"))?;
        let keys = keys
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("k reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("k heads"))?;
        let values = values
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("v reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("v heads"))?;
        if config.qk_norm {
            // Qwen3: RMSNorm over the head dimension, per head, before RoPE.
            let query = norm_heads(
                &query,
                self.get(&format!("{prefix}.self_attn.q_norm.weight"))?,
                config.rms_norm_eps,
            )?;
            let keys = norm_heads(
                &keys,
                self.get(&format!("{prefix}.self_attn.k_norm.weight"))?,
                config.rms_norm_eps,
            )?;
            return self.attention_scores(
                prefix,
                query,
                keys,
                values,
                heads,
                kv_heads,
                head_dim,
                token_count,
            );
        }
        self.attention_scores(
            prefix,
            query,
            keys,
            values,
            heads,
            kv_heads,
            head_dim,
            token_count,
        )
    }

    /// RoPE, GQA expansion, causal softmax, and the output projection.
    #[allow(clippy::too_many_arguments)]
    fn attention_scores(
        &self,
        prefix: &str,
        query: Array,
        keys: Array,
        values: Array,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        token_count: usize,
    ) -> Result<Array, FamilyError> {
        let query = apply_rotary(&query, &self.rope_cos, &self.rope_sin)?;
        let keys = apply_rotary(&keys, &self.rope_cos, &self.rope_sin)?;
        // HF `repeat_kv` interleave: query head `i` pairs with KV head
        // `i / group`.
        let group = heads / kv_heads;
        let keys_r = mlx_rs::ops::stack(&(0..group).map(|_| keys.clone()).collect::<Vec<_>>(), 0)
            .map_err(mlx_error("k repeat"))?
            .transpose_axes(&[1, 0, 2, 3])
            .map_err(mlx_error("k repeat transpose"))?
            .reshape(&[heads as i32, token_count as i32, head_dim as i32])
            .map_err(mlx_error("k repeat reshape"))?;
        let values_r =
            mlx_rs::ops::stack(&(0..group).map(|_| values.clone()).collect::<Vec<_>>(), 0)
                .map_err(mlx_error("v repeat"))?
                .transpose_axes(&[1, 0, 2, 3])
                .map_err(mlx_error("v repeat transpose"))?
                .reshape(&[heads as i32, token_count as i32, head_dim as i32])
                .map_err(mlx_error("v repeat reshape"))?;

        let mixed = causal_attention(&query, &keys_r, &values_r, head_dim)?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("mix heads"))?
            .reshape(&[token_count as i32, (heads * head_dim) as i32])
            .map_err(mlx_error("mix flatten"))?;
        self.project(&format!("{prefix}.self_attn.o_proj.weight"), &mixed)
    }
}

/// Plain RMSNorm over the last axis (`x · w · rsqrt(mean(x²) + eps)`), the
/// Qwen norm without the Qwen3.5 offset fold.
/// Qwen2 has learned Q/K/V biases; Qwen3 applies head normalization instead.
pub(crate) fn qkv_projections<'a>(
    input: &Array,
    prefix: &str,
    biased: bool,
    get: impl Fn(&str) -> Result<&'a Array, FamilyError>,
) -> Result<[Array; 3], FamilyError> {
    let project = |component| {
        let stem = format!("{prefix}.self_attn.{component}_proj");
        let weight = get(&format!("{stem}.weight"))?;
        let output = input
            .matmul(
                &weight
                    .transpose()
                    .map_err(mlx_error("QKV weight transpose"))?,
            )
            .map_err(mlx_error("QKV matmul"))?;
        if biased {
            output
                .add(get(&format!("{stem}.bias"))?)
                .map_err(mlx_error("QKV bias"))
        } else {
            Ok(output)
        }
    };
    Ok([project("q")?, project("k")?, project("v")?])
}

fn rms_norm(x: &Array, weight: &Array, eps: f64) -> Result<Array, FamilyError> {
    mlx_rs::fast::rms_norm(x, Some(weight), eps as f32).map_err(mlx_error("rms norm"))
}

/// Per-head RMSNorm over the last axis of `(heads, seq, head_dim)` values.
fn norm_heads(values: &Array, weight: &Array, eps: f64) -> Result<Array, FamilyError> {
    let dims = values.shape().to_vec();
    let rows: i32 = dims.iter().take(dims.len() - 1).product();
    let width = *dims.last().expect("non-empty shape");
    let flat = values
        .clone()
        .reshape(&[rows, width])
        .map_err(mlx_error("norm reshape"))?;
    mlx_rs::fast::rms_norm(&flat, Some(weight), eps as f32)
        .map_err(mlx_error("head norm"))?
        .reshape(&dims)
        .map_err(mlx_error("norm restore"))
}

/// Host FP32 RoPE tables at `(max_sequence_tokens, head_dim / 2)`, the same
/// math as the candle oracles' `RotaryEmbedding`.
fn rope_tables(config: &DenseQwenConfig) -> (Array, Array) {
    let half = config.head_dim / 2;
    let mut cos_values = Vec::with_capacity(config.max_sequence_tokens * half);
    let mut sin_values = Vec::with_capacity(config.max_sequence_tokens * half);
    for position in 0..config.max_sequence_tokens {
        for index in 0..half {
            let exponent = (2 * index) as f64 / config.head_dim as f64;
            let inv_freq = 1.0 / config.rope_theta.powf(exponent);
            let angle = position as f32 * inv_freq as f32;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }
    let shape = [config.max_sequence_tokens as i32, half as i32];
    let cos = Array::from_slice(&cos_values, &shape);
    let sin = Array::from_slice(&sin_values, &shape);
    cos.eval().expect("rope cos materializes");
    sin.eval().expect("rope sin materializes");
    (cos, sin)
}

/// Full-dimension NeoX rotary embedding over `(heads, seq, head_dim)`
/// values: split each head's width into halves, rotate the halves against
/// each other at the absolute positions' angles.
fn apply_rotary(values: &Array, rope_cos: &Array, rope_sin: &Array) -> Result<Array, FamilyError> {
    let dims = values.shape();
    let seq = dims[1];
    let indices: Vec<u32> = (0..seq as u32).collect();
    let idx = Array::from_slice(&indices, &[seq]);
    let cos = rope_cos
        .take_axis(&idx, 0)
        .map_err(mlx_error("rope cos slice"))?;
    let sin = rope_sin
        .take_axis(&idx, 0)
        .map_err(mlx_error("rope sin slice"))?;
    let head_dim = *dims.last().expect("head dim") as usize;
    let half = (head_dim / 2) as i32;
    let split = values.split_equal(2, -1).map_err(mlx_error("rope split"))?;
    let (x1, x2) = (&split[0], &split[1]);
    let broadcast_shape = [dims[0], seq, half];
    let cos = cos
        .reshape(&[seq, half])
        .map_err(mlx_error("rope cos reshape"))?;
    let cos = mlx_rs::ops::broadcast_to(&cos, &broadcast_shape)
        .map_err(mlx_error("rope cos broadcast"))?;
    let sin = sin
        .reshape(&[seq, half])
        .map_err(mlx_error("rope sin reshape"))?;
    let sin = mlx_rs::ops::broadcast_to(&sin, &broadcast_shape)
        .map_err(mlx_error("rope sin broadcast"))?;
    let out1 = x1
        .multiply(&cos)
        .map_err(mlx_error("rope out1 mul"))?
        .subtract(&x2.multiply(&sin).map_err(mlx_error("rope out2 mul"))?)
        .map_err(mlx_error("rope out1"))?;
    let out2 = x2
        .multiply(&cos)
        .map_err(mlx_error("rope out2 mul"))?
        .add(&x1.multiply(&sin).map_err(mlx_error("rope out1 mul"))?)
        .map_err(mlx_error("rope out2"))?;
    mlx_rs::ops::concatenate(&[out1, out2], -1).map_err(mlx_error("rope concat"))
}

/// Head-first causal attention with one score block live at a time.
/// Call under the MLX runtime lock.
pub(crate) fn causal_attention(
    query: &Array,
    keys: &Array,
    values: &Array,
    head_dim: usize,
) -> Result<Array, FamilyError> {
    use mlx_rs::ops::indexing::TryIndexOp as _;

    const CHUNK: i32 = 256;
    let seq = query.shape()[1];
    let scale = Array::from_f32((head_dim as f32).sqrt());
    let keys = keys.transpose_axes(&[0, 2, 1]).map_err(mlx_error("k t"))?;
    // Keep only one score block in the lazy graph. At the 16K Drex limit
    // a dense 16-head score tensor alone is 16 GiB.
    let mut blocks = Vec::new();
    for start in (0..seq).step_by(CHUNK as usize) {
        let end = (start + CHUNK).min(seq);
        let q = query
            .try_index((.., start..end, ..))
            .map_err(mlx_error("q block"))?;
        let k = keys
            .try_index((.., .., ..end))
            .map_err(mlx_error("k prefix"))?;
        let v = values
            .try_index((.., ..end, ..))
            .map_err(mlx_error("v prefix"))?;
        let mut mask = vec![f32::NEG_INFINITY; ((end - start) * end) as usize];
        for (offset, row) in mask.chunks_exact_mut(end as usize).enumerate() {
            row[..start as usize + offset + 1].fill(0.0);
        }
        let mask = Array::from_slice(&mask, &[1, end - start, end]);
        let scores = q
            .matmul(&k)
            .map_err(mlx_error("scores matmul"))?
            .divide(&scale)
            .map_err(mlx_error("scores scale"))?
            + mask;
        let maximum = scores
            .max_axis(-1, true)
            .map_err(mlx_error("softmax max"))?;
        let exp = scores
            .sub(&maximum)
            .exp()
            .map_err(mlx_error("softmax exp"))?;
        let denominator = exp.sum_axis(-1, true).map_err(mlx_error("softmax sum"))?;
        let mixed = exp
            .div(&denominator)
            .matmul(&v)
            .map_err(mlx_error("mix matmul"))?;
        mixed.eval().map_err(mlx_error("mix block eval"))?;
        blocks.push(mixed);
    }
    mlx_rs::ops::concatenate(&blocks, 1).map_err(mlx_error("mix concat"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_attention_keeps_the_full_prefix_across_chunks() {
        let runtime = MlxRuntime::new(crate::qwen35::mlx::MlxRuntimeConfig::default()).unwrap();
        runtime
            .execute(|| {
                let seq = 300;
                let qk = Array::from_slice(&vec![0.0_f32; seq], &[1, seq as i32, 1]);
                let values = (0..seq).map(|index| index as f32).collect::<Vec<_>>();
                let v = Array::from_slice(&values, &[1, seq as i32, 1]);
                let out = causal_attention(&qk, &qk, &v, 1).unwrap();
                let actual = out.to_vec_cast::<f32>().unwrap();
                assert_eq!(actual.len(), seq);
                for (index, actual) in actual.into_iter().enumerate() {
                    // Zero logits give a uniform distribution on the visible prefix.
                    assert!((actual - index as f32 / 2.0).abs() < 0.001, "row {index}");
                }
            })
            .unwrap();
    }

    #[test]
    fn qwen2_adds_each_attention_bias_and_qwen3_omits_them() {
        let runtime = MlxRuntime::new(crate::qwen35::mlx::MlxRuntimeConfig::default()).unwrap();
        runtime
            .execute(|| {
                let weight = Array::from_slice(&[0.0_f32, 0.0], &[1, 2]);
                let q = Array::from_slice(&[1.0_f32], &[1]);
                let k = Array::from_slice(&[2.0_f32], &[1]);
                let v = Array::from_slice(&[3.0_f32], &[1]);
                let input = Array::from_slice(&[4.0_f32, 5.0, 6.0, 7.0], &[2, 2]);
                let get = |name: &str| match name.rsplit_once('.').unwrap() {
                    (_, "weight") => Ok(&weight),
                    ("layers.0.self_attn.q_proj", "bias") => Ok(&q),
                    ("layers.0.self_attn.k_proj", "bias") => Ok(&k),
                    ("layers.0.self_attn.v_proj", "bias") => Ok(&v),
                    _ => panic!("unexpected tensor: {name}"),
                };
                for (biased, expected) in [(true, [1.0, 2.0, 3.0]), (false, [0.0; 3])] {
                    let outputs = qkv_projections(&input, "layers.0", biased, get).unwrap();
                    for (output, value) in outputs.into_iter().zip(expected) {
                        assert_eq!(output.to_vec_cast::<f32>().unwrap(), [value; 2]);
                    }
                }
            })
            .unwrap();
    }
}
