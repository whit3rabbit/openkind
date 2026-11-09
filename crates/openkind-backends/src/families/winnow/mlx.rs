//! MLX/Metal execution of the pinned winnow router checkpoint (feature
//! `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified base checkpoint and LoRA adapter merge host-side
//! exactly like the CPU oracle (`super::model`), then feed a dense Qwen2
//! FP32 forward mirroring the shared [`DenseQwenMlxModel`] body at the
//! pinned Qwen2 geometry (plain RMSNorm, grouped-query attention with HF
//! `repeat_kv` interleave, full-dimension NeoX rotary embedding at theta
//! 1e6, causal masking, SiLU-gated MLP; no q/k norm). The readout mirrors
//! the CPU letter readout: because the pinned `tie_word_embeddings`
//! checkpoint has no separate LM head, the full-vocab logits row the CPU
//! oracle indexes is exactly `hidden · embedding.T`, so the per-letter dots
//! reproduce the indexed values. Nothing is sampled and no continuation
//! state exists.
//!
//! This module deliberately duplicates the shared dense body's forward
//! instead of reusing it: the pinned adapter pairs lowercase `lora_a`
//! (in, rank) / `lora_b` (rank, out) factors and transposes the delta into
//! the `(out, in)` Linear layout, while [`DenseQwenLoadSpec`]'s adapter
//! hook consumes the PEFT convention (`lora_A` (rank, in), `lora_B`
//! (out, rank), no transpose). The merge therefore integrates at tensor
//! read here, and the shared loader's shard-only constructor cannot serve
//! pre-merged values without staging intermediate files (forbidden by the
//! no-staging invariant).
//!
//! Every MLX operation — model load and per-pass forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::collections::BTreeMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use mlx_rs::Array;

use crate::families::qwen35_survey_mlx::{mlx_error, survey_array, visit_safetensors};
use crate::families::qwen_dense_mlx::{causal_attention, qkv_projections, DenseQwenConfig};
use crate::families::support::{FamilyControl, FamilyError};
use crate::qwen35::mlx::MlxRuntime;
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding};

use super::model::VerifiedArtifacts;
use super::{LORA_RANK, LORA_SCALE};

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// base checkpoint plus LoRA delta merged host-side, widened to FP32 arrays
/// computed on the Metal GPU.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-winnow";

/// Backend id reported by the MLX engine serving this profile.
pub const MLX_BACKEND_ID: &str = "winnow/mlx-fp32";

/// The pinned winnow model on MLX: merged decoder tensors plus the
/// host-resident tied embedding.
pub struct WinnowMlxModel {
    /// Decoder tensors keyed by their suffix under the `model.` namespace
    /// (`layers.N.<rest>`, `norm.weight`), with the LoRA delta merged in.
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
unsafe impl Sync for WinnowMlxModel {}
unsafe impl Send for WinnowMlxModel {}

/// The pinned dense Qwen2 geometry of the winnow base checkpoint, derived
/// from the same pinned config the CPU engine enforces.
fn pinned_dense_config(config: &candle_transformers::models::qwen2::Config) -> DenseQwenConfig {
    DenseQwenConfig {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        head_dim: config.hidden_size / config.num_attention_heads,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        num_hidden_layers: config.num_hidden_layers,
        intermediate_size: config.intermediate_size,
        rope_theta: config.rope_theta,
        rms_norm_eps: config.rms_norm_eps,
        // Qwen2 has no per-head q/k RMSNorm.
        qk_norm: false,
        max_sequence_tokens: super::MAX_SEQUENCE_TOKENS,
    }
}

/// Read the adapter's lowercase factor pairs from the digest-verified
/// adapter file: one `(a, b)` pair per target projection stem
/// (`model.layers.N.<proj>`), where `a` is `(in, rank)` and `b` is
/// `(rank, out)`.
fn adapter_factors(
    adapter: &std::path::Path,
) -> Result<BTreeMap<String, (Tensor, Tensor)>, FamilyError> {
    // SAFETY: the family verified the immutable size and SHA-256 of the
    // read-only adapter file before this mapper opens it.
    let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(adapter)? };
    let mut pairs: BTreeMap<String, (Option<Tensor>, Option<Tensor>)> = BTreeMap::new();
    for name in safetensors_names(adapter)? {
        let tensor = mmap.load(&name, &Device::Cpu)?.to_dtype(DType::F32)?;
        let (stem, kind) = name
            .rsplit_once('.')
            .ok_or_else(|| FamilyError::InvalidInput(format!("bad adapter key {name}")))?;
        let entry = pairs.entry(stem.to_owned()).or_default();
        match kind {
            "lora_a" => entry.0 = Some(tensor),
            "lora_b" => entry.1 = Some(tensor),
            other => {
                return Err(FamilyError::InvalidInput(format!(
                    "unexpected adapter tensor kind `{other}` in {name}"
                )));
            }
        }
    }
    let mut factors = BTreeMap::new();
    for (stem, (a, b)) in pairs {
        let (Some(a), Some(b)) = (a, b) else {
            return Err(FamilyError::InvalidInput(format!(
                "adapter projection `{stem}` is missing lora_a or lora_b"
            )));
        };
        factors.insert(stem, (a, b));
    }
    Ok(factors)
}

/// Read the tensor names from a safetensors header without loading data
/// (the same header walk the CPU merge uses for the adapter).
fn safetensors_names(path: &std::path::Path) -> Result<Vec<String>, FamilyError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let length = u64::from_le_bytes(length_bytes) as usize;
    let mut header_bytes = vec![0u8; length];
    file.read_exact(&mut header_bytes)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|error| FamilyError::Json {
            path: path.to_path_buf(),
            source: error,
        })?;
    let mut names: Vec<String> = header
        .as_object()
        .ok_or_else(|| FamilyError::InvalidInput("safetensors header is not an object".into()))?
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect();
    names.sort();
    Ok(names)
}

impl WinnowMlxModel {
    /// Load the digest-verified base checkpoint, merge the verified adapter
    /// exactly like the CPU oracle, and build the FP32 MLX arrays. Must run
    /// inside [`MlxRuntime::execute`].
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        config: &candle_transformers::models::qwen2::Config,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let dense = pinned_dense_config(config);
        let factors = adapter_factors(&artifacts.adapter)?;
        let keep = |name: &str| name.starts_with("model.layers.") || name == "model.norm.weight";
        let mut tensors: BTreeMap<String, Array> = BTreeMap::new();
        visit_safetensors(&artifacts.checkpoint, keep, |name, mut values, shape| {
            let suffix = name.trim_start_matches("model.");
            if let Some((a, b)) = name
                .strip_suffix(".weight")
                .and_then(|stem| factors.get(stem))
            {
                merge_delta(&name, &mut values, shape, a, b)?;
            }
            let array = survey_array(&values, shape);
            array.eval().map_err(mlx_error("weight materialization"))?;
            if tensors.insert(suffix.to_owned(), array).is_some() {
                return Err(FamilyError::InvalidInput(format!(
                    "duplicate tensor `{name}` across shards"
                )));
            }
            Ok(())
        })?;
        if tensors.is_empty() {
            return Err(FamilyError::InvalidInput(
                "checkpoint shards carry no `model.layers` decoder tensors".to_owned(),
            ));
        }
        // Every adapter target must have contributed a merged tensor: a
        // factor pair whose base tensor is missing would otherwise silently
        // load the unadapted model.
        for stem in factors.keys() {
            let Some(suffix) = stem.strip_prefix("model.") else {
                return Err(FamilyError::InvalidInput(format!(
                    "adapter targets `{stem}`, outside the checkpoint namespace"
                )));
            };
            if !tensors.contains_key(&format!("{suffix}.weight")) {
                return Err(FamilyError::ContractMismatch {
                    field: "adapter.target",
                    expected: format!("base tensor `{stem}.weight` present"),
                    actual: "missing".to_owned(),
                });
            }
        }
        let embedding = {
            let layout = EmbeddingLayout::read_named(
                &artifacts.checkpoint,
                "model.embed_tokens.weight",
                dense.vocab_size,
                dense.hidden_size,
            )
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
            Qwen35Embedding::from_layout(artifacts.checkpoint.clone(), layout)
        };
        // The MLP width the forward executes must be the pinned config's,
        // so a drifted base checkpoint fails closed instead of silently
        // changing routing answers.
        let gate = tensors
            .get("layers.0.mlp.gate_proj.weight")
            .ok_or_else(|| {
                FamilyError::InvalidInput("tensor `layers.0.mlp.gate_proj.weight` missing".into())
            })?;
        let gate_shape = gate.shape();
        if gate_shape.len() != 2 || gate_shape[1] != dense.intermediate_size as i32 {
            return Err(FamilyError::contract(
                "winnow.base_geometry",
                format!("MLP intermediate width {}", dense.intermediate_size),
                format!("{gate_shape:?}"),
            ));
        }
        let (rope_cos, rope_sin) = rope_tables(&dense);
        let model = Self {
            tensors,
            embedding,
            config: dense,
            rope_cos,
            rope_sin,
            runtime: Arc::clone(runtime),
        };
        // Materialize every retained array so no unevaluated lazy graph
        // crosses the loading thread's boundary.
        Ok(model)
    }

    /// Forward the routing prompt once and return the next-token logits at
    /// the letter slots. Nothing is sampled and no continuation exists.
    ///
    /// The forward is one uninterruptible MLX pass, so cancellation and the
    /// deadline are observed on either side of it, exactly like the CPU
    /// model lock discipline.
    pub(super) fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        control.check()?;
        let hidden = self.runtime.execute(|| self.forward_hidden(prompt_ids))??;
        control.check()?;
        let hidden_size = self.config.hidden_size;
        let token_count = prompt_ids.len();
        if hidden.len() != token_count * hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "backbone output width does not match the pinned geometry: {} values for \
                 {token_count} tokens",
                hidden.len()
            )));
        }
        // The letter ids resolve through the renderer's completion-token
        // contract; the tied-embedding dots fail closed on out-of-range ids
        // via the embedding reader.
        let rows = self.embed_rows(letter_ids)?;
        if rows.len() != letter_ids.len() * hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "letter embedding rows have unexpected geometry: {} values for {} letters",
                rows.len(),
                letter_ids.len()
            )));
        }
        // Last prompt position: the routing slot.
        let slot = &hidden[(token_count - 1) * hidden_size..];
        let logits: Vec<f64> = rows
            .chunks_exact(hidden_size)
            .map(|row| {
                row.iter()
                    .zip(slot)
                    .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                    .sum::<f64>()
            })
            .collect();
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "winnow readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }

    /// Host-resident tied-embedding rows for `token_ids`, widened to FP32
    /// exactly like the CPU readouts.
    fn embed_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
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
    fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let config = self.config;
        let hidden_size = config.hidden_size;
        let token_count = input_ids.len();
        if token_count == 0 {
            return Err(FamilyError::InvalidInput(
                "winnow prompts carry at least one token".to_owned(),
            ));
        }
        let embedded = self.embed_rows(input_ids)?;
        let mut hidden = Array::from_slice(&embedded, &[token_count as i32, hidden_size as i32]);
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
            let activated = mlx_rs::nn::silu(&gate).map_err(mlx_error("mlp silu"))? * up;
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

    /// Causal grouped-query attention with full-dimension NeoX rotary
    /// embedding at absolute positions.
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
            qkv_projections(normalized, prefix, true, |name| self.get(name))?;
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

/// Merge one adapter delta into the base weight values (host FP32,
/// elementwise), the CPU merge exactly: `delta = (a @ b) · scale`
/// transposed from `(in, out)` into the `(out, in)` Linear layout.
fn merge_delta(
    name: &str,
    values: &mut [f32],
    shape: &[usize],
    a: &Tensor,
    b: &Tensor,
) -> Result<(), FamilyError> {
    if a.dim(1)? != LORA_RANK || b.dim(0)? != LORA_RANK {
        return Err(FamilyError::contract(
            "adapter.rank",
            format!("rank {LORA_RANK} for `{name}`"),
            format!("a {:?}, b {:?}", a.shape(), b.shape()),
        ));
    }
    // lora_a (in, rank) @ lora_b (rank, out) = (in, out); the Linear layout
    // is (out, in), so the delta is transposed.
    let delta = (a.matmul(b)? * LORA_SCALE)?.t()?;
    let delta = delta
        .flatten_all()
        .and_then(|delta| delta.to_vec1::<f32>())
        .map_err(FamilyError::Candle)?;
    if delta.len() != values.len() || shape.len() != 2 {
        return Err(FamilyError::contract(
            "adapter.delta.shape",
            format!("{shape:?}"),
            format!("delta has {} values", delta.len()),
        ));
    }
    for (value, delta) in values.iter_mut().zip(&delta) {
        *value += delta;
    }
    Ok(())
}

/// Plain RMSNorm over the last axis (`x · w · rsqrt(mean(x²) + eps)`), the
/// Qwen norm without the Qwen3.5 offset fold.
fn rms_norm(x: &Array, weight: &Array, eps: f64) -> Result<Array, FamilyError> {
    mlx_rs::fast::rms_norm(x, Some(weight), eps as f32).map_err(mlx_error("rms norm"))
}

/// Host FP32 RoPE tables at `(max_sequence_tokens, head_dim / 2)`, the same
/// math as the candle oracle's `RotaryEmbedding`.
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
    let cos = mlx_rs::ops::broadcast_to(
        &cos.reshape(&[seq, half])
            .map_err(mlx_error("rope cos reshape"))?,
        &broadcast_shape,
    )
    .map_err(mlx_error("rope cos broadcast"))?;
    let sin = mlx_rs::ops::broadcast_to(
        &sin.reshape(&[seq, half])
            .map_err(mlx_error("rope sin reshape"))?,
        &broadcast_shape,
    )
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
