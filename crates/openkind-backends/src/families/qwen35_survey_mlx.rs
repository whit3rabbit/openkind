//! Shared geometry-parameterized **causal** MLX execution for surveyed
//! Qwen3.5 decision backends (feature `mlx`, macOS arm64).
//!
//! The parity-verified CPU oracle ([`crate::qwen35::backbone::layer0`])
//! executes hybrid decoder blocks at whatever geometry its profile pins;
//! the shared causal MLX backbone hard-codes the pinned 4B dimensions.
//! Surveyed families whose checkpoints move the same hybrid architecture to
//! another width (strands-decider-2b, drex-v1.5's 9B) execute here: the
//! layer math mirrors the oracle at the profile's frozen geometry, as
//! causal full prefill over one row — the stateless decision families never
//! retain continuation state, so the shared backbone's state machinery is
//! not needed.
//!
//! The fused Metal kernels integrate at the geometries they were verified
//! at (fused SwiGLU everywhere FP32, the SIMD DeltaNet recurrence for the
//! 32-value-head width, fused partial RoPE for the 256-wide attention
//! heads); every other path runs the reference array ops.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! the process-wide [`MlxRuntime`] execution lock, per the workspace MLX
//! discipline.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mlx_rs::ops::indexing::TryIndexOp as _;
use mlx_rs::Array;

use crate::families::qwen_dense_mlx::causal_attention;
use crate::families::support::{FamilyControl, FamilyError};
use crate::qwen35::mlx::runtime::MlxRuntime;
use crate::qwen35::mlx::weights::MlxSurveyLora;
use crate::qwen35::mlx::{MlxBackboneOutput, MlxQwen35Backbone};
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding, Qwen35Geometry};

/// RMS epsilon shared by every Qwen3.5 geometry. A future geometry with a
/// different epsilon must fail the load instead of silently computing the
/// wrong norms.
pub(crate) const RMS_EPSILON: f32 = 1e-6;

/// Evaluate one independent row with bounded attention scratch. Continuation
/// carries the row's causal prefix, including KV, convolution, and DeltaNet
/// state, only until this readout finishes.
pub(crate) fn prefill_row(
    backbone: &MlxQwen35Backbone,
    ids: &[u32],
    control: &FamilyControl,
) -> Result<MlxBackboneOutput, FamilyError> {
    // The shared backbone's attention scores have one row per query and
    // one column per prefix token. Limit the number of simultaneous queries.
    let mut blocks = ids.chunks(256);
    let first = blocks
        .next()
        .ok_or_else(|| FamilyError::InvalidInput("prompt token ids are empty".to_owned()))?;
    control.check()?;
    let (mut output, mut state) = backbone.prefill(first)?;
    for block in blocks {
        control.check()?;
        (output, state) = backbone.continue_from(&state, block)?;
    }
    Ok(output)
}

/// Map an error onto [`FamilyError::Mlx`] with the operation's name.
pub(crate) fn mlx_error<E: std::fmt::Display>(
    operation: &'static str,
) -> impl Fn(E) -> FamilyError {
    move |error| FamilyError::Mlx(format!("{operation}: {error}"))
}

/// A loaded surveyed Qwen3.5 model on MLX: FP32 arrays over the
/// `model.language_model.*` tensors plus the host-resident tied embedding.
pub(crate) struct SurveyMlxModel {
    /// Decoder tensors keyed by their suffix under the
    /// `model.language_model.` namespace (`layers.N.<rest>`, `norm.weight`).
    tensors: BTreeMap<String, Array>,
    embedding: Qwen35Embedding,
    geometry: Qwen35Geometry,
    runtime: Arc<MlxRuntime>,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared qwen35 MLX backbone).
unsafe impl Sync for SurveyMlxModel {}
unsafe impl Send for SurveyMlxModel {}

/// What one digest-verified survey load consumes.
pub(crate) struct SurveyLoadSpec<'a> {
    /// Digest-verified base shards, in load order.
    pub shards: &'a [PathBuf],
    /// Shard carrying the tied embedding table.
    pub embedding_shard: &'a Path,
    /// Name of the embedding tensor inside `embedding_shard`.
    pub embedding_tensor: &'a str,
    /// Vocab size the embedding table must have.
    pub vocab_size: usize,
    /// Frozen geometry of the checkpoint.
    pub geometry: Qwen35Geometry,
    /// Optional PEFT LoRA adapter merged into the decoder tensors as they
    /// stream in. Its factor keys must target the
    /// `model.language_model.layers.*` namespace.
    pub adapter: Option<&'a MlxSurveyLora>,
}

impl SurveyMlxModel {
    /// Load the verified shards into FP32 MLX arrays, merging the optional
    /// adapter exactly like the CPU oracle's merge. Must run inside
    /// [`MlxRuntime::execute`] (the engine's load holds it).
    pub(crate) fn load(
        spec: SurveyLoadSpec<'_>,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        if spec.geometry.rms_epsilon != RMS_EPSILON {
            return Err(FamilyError::InvalidInput(format!(
                "survey MLX execution pins RMS epsilon {RMS_EPSILON}, the profile pins {}",
                spec.geometry.rms_epsilon
            )));
        }
        if !spec.geometry.has_uniform_causal_layout() {
            return Err(FamilyError::InvalidInput(
                "survey MLX execution implements the uniform causal mixer table only".to_owned(),
            ));
        }
        let factors = match spec.adapter {
            Some(adapter) => Some(crate::qwen35::mlx::weights::load_lora_factors_for_survey(
                adapter,
            )?),
            None => None,
        };
        let mut tensors = BTreeMap::new();
        let keep = |name: &str| {
            name.starts_with("model.language_model.layers.")
                || name == "model.language_model.norm.weight"
        };
        for shard in spec.shards {
            visit_safetensors(shard, keep, |name, mut values, shape| {
                if let Some(adapter) = spec.adapter {
                    if let Some((a, b)) = factors.as_ref().and_then(|factors| factors.get(&name)) {
                        crate::qwen35::mlx::weights::apply_lora_delta_to_values(
                            &name,
                            &mut values,
                            a,
                            b,
                            adapter.scale,
                        )?;
                    }
                }
                let suffix = name.trim_start_matches("model.language_model.");
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
                "checkpoint shards carry no `model.language_model` decoder tensors".to_owned(),
            ));
        }
        // Every adapter target must have contributed a merged tensor: a
        // factor pair whose base tensor is missing would otherwise silently
        // load the unadapted model.
        if let Some(factors) = &factors {
            for name in factors.keys() {
                let Some(suffix) = name.strip_prefix("model.language_model.") else {
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
                spec.embedding_tensor,
                spec.vocab_size,
                spec.geometry.hidden_size,
            )
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
            Qwen35Embedding::from_layout(spec.embedding_shard.to_path_buf(), layout)
        };
        let model = Self {
            tensors,
            embedding,
            geometry: spec.geometry,
            runtime: Arc::clone(runtime),
        };
        Ok(model)
    }

    /// The process-wide runtime this model evaluates on. Forwards must run
    /// inside `self.runtime().execute(...)` — the forward itself does not
    /// take the lock (the lock is not reentrant).
    pub(crate) fn runtime(&self) -> &Arc<MlxRuntime> {
        &self.runtime
    }

    /// Causal full-sequence forward, returning the final-normed hidden
    /// states as host FP32 row-major `[token_count, hidden]` values.
    ///
    /// Must run inside [`MlxRuntime::execute`].
    pub(crate) fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let geometry = self.geometry;
        let hidden_size = geometry.hidden_size;
        let token_count = input_ids.len();
        if token_count == 0 {
            return Err(FamilyError::InvalidInput(
                "survey prompts carry at least one token".to_owned(),
            ));
        }
        let embedded = self
            .embedding
            .embed(input_ids)
            .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
        debug_assert_eq!(embedded.values().len(), token_count * hidden_size);
        let mut hidden =
            Array::from_slice(embedded.values(), &[token_count as i32, hidden_size as i32]);
        for layer in 0..geometry.layer_count {
            let prefix = format!("layers.{layer}");
            let normalized = rms_norm(
                &hidden,
                self.get(&format!("{prefix}.input_layernorm.weight"))?,
            )?;
            let attention = if geometry.is_full_attention(layer) {
                self.full_attention(&prefix, &normalized, token_count)?
            } else {
                self.linear_attention(&prefix, &normalized, token_count)?
            };
            let mixed = hidden.clone() + attention;
            let post = rms_norm(
                &mixed,
                self.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
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
            // block (the DeltaNet token loop is the quadratic-in-tokens part).
            hidden.eval().map_err(mlx_error("layer materialization"))?;
        }
        let final_hidden = rms_norm(&hidden, self.get("norm.weight")?)?;
        let flattened = final_hidden
            .reshape(&[(token_count * hidden_size) as i32])
            .map_err(mlx_error("final flatten"))?;
        let widened = flattened
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(mlx_error("host widen"))?;
        widened.eval().map_err(mlx_error("host eval"))?;
        widened
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

    /// **Causal** grouped-query attention: position `t` attends to positions
    /// `..=t` (the standard Qwen3.5 serving geometry, unlike d3's lifted
    /// noncausal mask). q/k carry per-head offset RMSNorm, the first quarter
    /// of each head rotates (NeoX pairing at theta 1e7, absolute positions),
    /// and the fused query/gate rows gate the mixed heads.
    fn full_attention(
        &self,
        prefix: &str,
        normalized: &Array,
        token_count: usize,
    ) -> Result<Array, FamilyError> {
        let geometry = self.geometry;
        let heads = geometry.attention_heads;
        let kv_heads = geometry.kv_heads;
        let head_dim = geometry.attention_head_dim;
        let rotary_dim = geometry.rotary_dim();
        let projected_q = self.project(&format!("{prefix}.self_attn.q_proj.weight"), normalized)?;
        let keys = self.project(&format!("{prefix}.self_attn.k_proj.weight"), normalized)?;
        let values = self.project(&format!("{prefix}.self_attn.v_proj.weight"), normalized)?;
        // Split the interleaved per-head [query | gate] halves.
        let q_reshaped = projected_q
            .reshape(&[token_count as i32, heads as i32, 2, head_dim as i32])
            .map_err(mlx_error("q reshape"))?;
        let halves = q_reshaped
            .split_at_indices(&[1], 2)
            .map_err(mlx_error("q split"))?;
        let query = halves[0]
            .clone()
            .reshape(&[token_count as i32, heads as i32, head_dim as i32])
            .map_err(mlx_error("q view"))?;
        let gates = halves[1]
            .clone()
            .reshape(&[token_count as i32, heads as i32, head_dim as i32])
            .map_err(mlx_error("gate view"))?;
        let keys = keys
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("k reshape"))?;
        let values = values
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("v reshape"))?;

        // Per-head offset-RMSNorm on q and k (weights folded at load).
        let query = norm_heads(
            &query,
            &folded(self.get(&format!("{prefix}.self_attn.q_norm.weight"))?)?,
        )?;
        let keys = norm_heads(
            &keys,
            &folded(self.get(&format!("{prefix}.self_attn.k_norm.weight"))?)?,
        )?;

        let query = apply_rotary(&query, token_count, rotary_dim)?;
        let keys = apply_rotary(&keys, token_count, rotary_dim)?;

        // Heads-first layout for the attention matmuls.
        let query = query
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("q heads"))?;
        let keys = keys
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("k heads"))?;
        let values = values
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("v heads"))?;

        // Expand the grouped kv heads (query head h uses kv head h/group).
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
            .reshape(&[token_count as i32, geometry.attention_size() as i32])
            .map_err(mlx_error("mix flatten"))?;
        let gated = mixed
            * mlx_rs::nn::sigmoid(&gates)
                .map_err(mlx_error("gate sigmoid"))?
                .reshape(&[token_count as i32, geometry.attention_size() as i32])
                .map_err(mlx_error("gate flatten"))?;
        self.project(&format!("{prefix}.self_attn.o_proj.weight"), &gated)
    }

    /// Causal gated-DeltaNet layer: one forward pass over the whole row
    /// with the per-token recurrence starting from a zeroed convolution
    /// window and recurrent state (every surveyed row is an independent
    /// full sequence; nothing is retained across rows).
    fn linear_attention(
        &self,
        prefix: &str,
        normalized: &Array,
        token_count: usize,
    ) -> Result<Array, FamilyError> {
        let geometry = self.geometry;
        let value_heads = geometry.value_heads;
        let key_heads = geometry.key_heads;
        let head_dim = geometry.head_dim;
        let qkv_size = geometry.qkv_size();
        let value_size = geometry.value_size();
        let key_size = geometry.key_size();
        let kernel = geometry.conv_kernel;

        let raw_qkv = self.project(
            &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
            normalized,
        )?;
        let z_all = self.project(
            &format!("{prefix}.linear_attn.in_proj_z.weight"),
            normalized,
        )?;
        let beta_all = self.project(
            &format!("{prefix}.linear_attn.in_proj_b.weight"),
            normalized,
        )?;
        let decay_all = self.project(
            &format!("{prefix}.linear_attn.in_proj_a.weight"),
            normalized,
        )?;
        // The convolution taps arrive [channels, 1, kernel] (or the
        // singleton-axis variant) and collapse to [channels, kernel].
        let conv1d = {
            let array = self.get(&format!("{prefix}.linear_attn.conv1d.weight"))?;
            let shape = array.shape();
            if shape != [qkv_size as i32, 1, kernel as i32]
                && shape != [qkv_size as i32, kernel as i32, 1]
            {
                return Err(FamilyError::InvalidInput(format!(
                    "conv1d tensor has unexpected shape {shape:?}"
                )));
            }
            array
                .reshape(&[qkv_size as i32, kernel as i32])
                .map_err(mlx_error("conv reshape"))?
        };
        let host_vector = |name: &str, width: usize| -> Result<Array, FamilyError> {
            let array = self.get(name)?;
            let widened = array
                .as_dtype(mlx_rs::Dtype::Float32)
                .map_err(mlx_error("vector widen"))?;
            widened.eval().map_err(mlx_error("vector eval"))?;
            let values = widened
                .to_vec_cast::<f32>()
                .map_err(|error| FamilyError::Mlx(format!("vector read: {error}")))?;
            if values.len() != width {
                return Err(FamilyError::InvalidInput(format!(
                    "tensor `{name}` has {} values, expected {width}",
                    values.len()
                )));
            }
            Ok(Array::from_slice(&values, &[width as i32]))
        };
        let dt_bias = host_vector(&format!("{prefix}.linear_attn.dt_bias"), value_heads)?;
        let a_log = host_vector(&format!("{prefix}.linear_attn.A_log"), value_heads)?;
        let delta_norm = host_vector(&format!("{prefix}.linear_attn.norm.weight"), head_dim)?;

        // The fused SIMD recurrence was verified at the 32-value-head width;
        // other geometries run the reference loop.
        if value_heads == 32
            && key_heads == 16
            && head_dim == 128
            && kernel == 4
            && raw_qkv.dtype() == mlx_rs::Dtype::Float32
            && matches!(
                crate::qwen35::mlx::MlxGatedDeltaKernel::default(),
                crate::qwen35::mlx::MlxGatedDeltaKernel::MetalSimd
            )
        {
            let zeros_conv = vec![0_f32; qkv_size * kernel];
            let conv = Array::from_slice(&zeros_conv, &[qkv_size as i32, kernel as i32]);
            let zeros_recurrent = vec![0_f32; value_heads * head_dim * head_dim];
            let recurrent = Array::from_slice(
                &zeros_recurrent,
                &[value_heads as i32, head_dim as i32, head_dim as i32],
            );
            let (mixed, _state) = crate::qwen35::mlx::layers::forward_hoisted_gated_delta(
                token_count,
                &raw_qkv,
                &z_all,
                &beta_all,
                &decay_all,
                &conv1d,
                &dt_bias,
                &a_log,
                &delta_norm,
                &conv,
                recurrent,
                crate::qwen35::mlx::MlxGatedDeltaKernel::MetalSimd,
            )
            .map_err(mlx_error("hoisted gated delta"))?;
            return self.project(&format!("{prefix}.linear_attn.out_proj.weight"), &mixed);
        }

        let zeros_conv = vec![0_f32; qkv_size * kernel];
        let mut conv = Array::from_slice(&zeros_conv, &[qkv_size as i32, kernel as i32]);
        let zeros_recurrent = vec![0_f32; value_heads * head_dim * head_dim];
        let mut recurrent = Array::from_slice(
            &zeros_recurrent,
            &[value_heads as i32, head_dim as i32, head_dim as i32],
        );
        // Value head v reads key head v / (value_heads / key_heads).
        let groups = value_heads / key_heads;
        let key_index: Vec<u32> = (0..value_heads)
            .map(|head| (head / groups) as u32)
            .collect();
        let key_index = Array::from_slice(&key_index, &[value_heads as i32]);
        let query_scale = (head_dim as f32).sqrt().recip();
        const RECURRENCE_CHUNK: usize = 32;
        let mut blocks = Vec::with_capacity(token_count.div_ceil(RECURRENCE_CHUNK));
        let mut mixed_rows = Vec::with_capacity(RECURRENCE_CHUNK);
        for row in 0..token_count {
            let index = Array::from_slice(&[row as u32], &[1]);
            let raw_row = take_row(&raw_qkv, &index, qkv_size)?
                .reshape(&[qkv_size as i32, 1])
                .map_err(mlx_error("qkv row col"))?;
            // Causal depthwise conv window: drop the oldest tap, append the
            // current raw row, then sum taps with silu.
            let keep = kernel - 1;
            let window_indices: Vec<u32> = (1..kernel as u32).collect();
            let shifted = conv
                .take_axis(Array::from_slice(&window_indices, &[keep as i32]), 1)
                .map_err(mlx_error("conv shift"))?;
            let window = mlx_rs::ops::concatenate(&[shifted, raw_row], 1)
                .map_err(mlx_error("conv window"))?;
            let summed = (window.clone() * conv1d.clone())
                .sum_axis(1, false)
                .map_err(mlx_error("conv sum"))?;
            let qkv_row = mlx_rs::nn::silu(&summed).map_err(mlx_error("conv silu"))?;
            // Split the fused q/k/v row into per-head blocks.
            let queries = qkv_row
                .try_index((0..key_size as i32,))
                .map_err(mlx_error("q slice"))?
                .reshape(&[key_heads as i32, head_dim as i32])
                .map_err(mlx_error("q heads"))?;
            let keys = qkv_row
                .try_index((key_size as i32..(key_size * 2) as i32,))
                .map_err(mlx_error("k slice"))?
                .reshape(&[key_heads as i32, head_dim as i32])
                .map_err(mlx_error("k heads"))?;
            let values = qkv_row
                .try_index(((key_size * 2) as i32..qkv_size as i32,))
                .map_err(mlx_error("v slice"))?
                .reshape(&[value_heads as i32, head_dim as i32])
                .map_err(mlx_error("v heads"))?;
            // Match the CPU oracle: q uses L2 normalization plus 1/sqrt(d),
            // k without the scale.
            let normalize = |heads: &Array, multiplier: f32| -> Result<Array, FamilyError> {
                let scale = (heads.clone() * heads.clone())
                    .sum_axis(-1, Some(true))
                    .map_err(mlx_error("qk sum"))?
                    + Array::from_f32(RMS_EPSILON);
                let scale = scale
                    .sqrt()
                    .map_err(mlx_error("qk sqrt"))?
                    .reciprocal()
                    .map_err(mlx_error("qk recip"))?;
                Ok(heads * scale * Array::from_f32(multiplier))
            };
            let queries = normalize(&queries, query_scale)?;
            let keys = normalize(&keys, 1.0)?;
            let queries = queries
                .take_axis(&key_index, 0)
                .map_err(mlx_error("q expand"))?;
            let keys = keys
                .take_axis(&key_index, 0)
                .map_err(mlx_error("k expand"))?;
            // Decay and beta gates: decay = exp(-exp(A_log) * softplus(dt)).
            let beta_row = take_row(&beta_all, &index, value_heads)?;
            let decay_row = take_row(&decay_all, &index, value_heads)?;
            let z_row = take_row(&z_all, &index, value_size)?
                .reshape(&[value_heads as i32, head_dim as i32])
                .map_err(mlx_error("z heads"))?;
            let beta = mlx_rs::ops::sigmoid(&beta_row).map_err(mlx_error("beta sigmoid"))?;
            let decay_argument = mlx_rs::nn::softplus(&(decay_row + dt_bias.clone()))
                .map_err(mlx_error("dt softplus"))?;
            let decay = mlx_rs::ops::exp(
                &(-mlx_rs::ops::exp(&a_log).map_err(mlx_error("a exp"))? * decay_argument),
            )
            .map_err(mlx_error("decay exp"))?
            .reshape(&[value_heads as i32, 1, 1])
            .map_err(mlx_error("decay shape"))?;
            recurrent = recurrent.clone() * decay;
            // The state layout is S[key, value]: the readout contracts the
            // KEY axis, so memory = k^T @ S per head.
            let keys_col = keys
                .reshape(&[value_heads as i32, head_dim as i32, 1])
                .map_err(mlx_error("k col"))?;
            let memory = keys_col
                .transpose_axes(&[0, 2, 1])
                .map_err(mlx_error("k row"))?
                .matmul(&recurrent)
                .map_err(mlx_error("memory matmul"))?
                .transpose_axes(&[0, 2, 1])
                .map_err(mlx_error("memory shape"))?;
            let beta_col = beta
                .reshape(&[value_heads as i32, 1, 1])
                .map_err(mlx_error("beta shape"))?;
            let values_col = values
                .reshape(&[value_heads as i32, head_dim as i32, 1])
                .map_err(mlx_error("v col"))?;
            let delta = (values_col - memory) * beta_col;
            // S[v] += k[v] outer delta[v].
            recurrent += keys_col
                .matmul(
                    &delta
                        .transpose_axes(&[0, 2, 1])
                        .map_err(mlx_error("delta t"))?,
                )
                .map_err(mlx_error("outer matmul"))?;
            // out[v] = q^T @ S per head, then offset RMSNorm * silu(z).
            let out = queries
                .reshape(&[value_heads as i32, 1, head_dim as i32])
                .map_err(mlx_error("q row"))?
                .matmul(&recurrent)
                .map_err(mlx_error("out matmul"))?
                .reshape(&[value_heads as i32, head_dim as i32])
                .map_err(mlx_error("out reshape"))?;
            let out_scale = (out.clone() * out.clone())
                .mean_axis(-1, Some(true))
                .map_err(mlx_error("out mean"))?
                + Array::from_f32(RMS_EPSILON);
            let out_scale = out_scale
                .sqrt()
                .map_err(mlx_error("out sqrt"))?
                .reciprocal()
                .map_err(mlx_error("out recip"))?;
            let gated = (out * out_scale)
                * delta_norm.clone()
                * mlx_rs::nn::silu(&z_row).map_err(mlx_error("z silu"))?;
            mixed_rows.push(
                gated
                    .reshape(&[value_size as i32])
                    .map_err(mlx_error("mixed row reshape"))?,
            );
            conv = window;
            if mixed_rows.len() == RECURRENCE_CHUNK || row + 1 == token_count {
                // Cut the lazy recurrence graph while retaining its terminal
                // state and every output row for the final projection.
                let block = mlx_rs::ops::stack(&mixed_rows, 0).map_err(mlx_error("mixed stack"))?;
                block.eval().map_err(mlx_error("mixed block eval"))?;
                recurrent
                    .eval()
                    .map_err(mlx_error("recurrent state eval"))?;
                conv.eval().map_err(mlx_error("conv state eval"))?;
                mixed_rows.clear();
                blocks.push(block);
            }
        }
        let stacked = mlx_rs::ops::concatenate(&blocks, 0).map_err(mlx_error("mixed concat"))?;
        self.project(&format!("{prefix}.linear_attn.out_proj.weight"), &stacked)
    }
}

/// Take one row of `[rows, width]` as a flat `[width]` array.
pub(crate) fn take_row(array: &Array, index: &Array, width: usize) -> Result<Array, FamilyError> {
    array
        .take_axis(index, 0)
        .map_err(mlx_error("row take"))?
        .reshape(&[width as i32])
        .map_err(mlx_error("row reshape"))
}

/// Offset RMSNorm over the last axis, folding the raw `(1 + w)` weight the
/// way the CPU oracle computes it at runtime.
pub(crate) fn rms_norm(x: &Array, raw_weight: &Array) -> Result<Array, FamilyError> {
    let variance = (x.clone() * x.clone())
        .mean_axis(-1, Some(true))
        .map_err(mlx_error("rms mean"))?;
    let scale = (variance + Array::from_f32(RMS_EPSILON))
        .sqrt()
        .map_err(mlx_error("rms sqrt"))?
        .reciprocal()
        .map_err(mlx_error("rms recip"))?;
    let folded = raw_weight.clone() + Array::from_f32(1.0);
    Ok((x * scale) * folded)
}

/// Fold a raw norm weight into the `(1 + w)` form the oracle computes at
/// runtime (host FP32, once per forward step's use site).
pub(crate) fn folded(weight: &Array) -> Result<Array, FamilyError> {
    let widened = weight
        .as_dtype(mlx_rs::Dtype::Float32)
        .map_err(mlx_error("norm widen"))?;
    widened.eval().map_err(mlx_error("norm eval"))?;
    let values = widened
        .to_vec_cast::<f32>()
        .map_err(|error| FamilyError::Mlx(format!("norm read: {error}")))?;
    let folded: Vec<f32> = values.iter().map(|value| 1.0 + value).collect();
    Ok(Array::from_slice(&folded, &[values.len() as i32]))
}

/// Per-head offset-RMSNorm over the last axis with a folded weight.
pub(crate) fn norm_heads(values: &Array, folded_weight: &Array) -> Result<Array, FamilyError> {
    let dims = values.shape().to_vec();
    let rows: i32 = dims.iter().take(dims.len() - 1).product();
    let width = *dims.last().expect("non-empty shape");
    let flat = values
        .clone()
        .reshape(&[rows, width])
        .map_err(mlx_error("norm reshape"))?;
    mlx_rs::fast::rms_norm(&flat, Some(folded_weight), RMS_EPSILON)
        .map_err(mlx_error("head norm"))?
        .reshape(&dims)
        .map_err(mlx_error("norm restore"))
}

/// Partial rotary embedding over the first `rotary_dim` dims of each head:
/// NeoX pairing at theta 1e7, host FP32 trig over absolute positions 0..T.
pub(crate) fn apply_rotary(
    values: &Array,
    token_count: usize,
    rotary_dim: usize,
) -> Result<Array, FamilyError> {
    let dims = values.shape();
    let heads = dims[1];
    let head_dim = *dims.last().expect("head dim") as usize;
    if rotary_dim == 64
        && head_dim == 256
        && values.dtype() == mlx_rs::Dtype::Float32
        && crate::qwen35::mlx::layers::fused_rope_enabled()
    {
        return crate::qwen35::mlx::layers::fused_apply_rotary(
            values,
            token_count,
            0,
            heads as usize,
        )
        .map_err(mlx_error("fused rope"));
    }
    let half = (rotary_dim / 2) as i32;
    let mut cos_rows = Vec::with_capacity(token_count);
    let mut sin_rows = Vec::with_capacity(token_count);
    for token in 0..token_count {
        let mut cos_row = Vec::with_capacity(rotary_dim / 2);
        let mut sin_row = Vec::with_capacity(rotary_dim / 2);
        for index in 0..rotary_dim / 2 {
            let frequency = f64::powf(10_000_000.0, -((2 * index) as f64) / rotary_dim as f64);
            let (sin, cos) = ((token as f64) * frequency).sin_cos();
            cos_row.push(cos as f32);
            sin_row.push(sin as f32);
        }
        cos_rows.push(Array::from_slice(&cos_row, &[1, half]));
        sin_rows.push(Array::from_slice(&sin_row, &[1, half]));
    }
    let cos = mlx_rs::ops::concatenate(&cos_rows, 0)
        .map_err(mlx_error("rope cos"))?
        .reshape(&[token_count as i32, 1, half])
        .map_err(mlx_error("rope cos reshape"))?;
    let sin = mlx_rs::ops::concatenate(&sin_rows, 0)
        .map_err(mlx_error("rope sin"))?
        .reshape(&[token_count as i32, 1, half])
        .map_err(mlx_error("rope sin reshape"))?;
    // Split each head's dims into [first rotary_dim | remaining head_dim -
    // rotary_dim]; the rotary window's two halves rotate against each other
    // and the tail passes through untouched.
    let head_dim = *dims.last().expect("head dim") as usize;
    assert_eq!(
        head_dim % rotary_dim,
        0,
        "the rotary window tiles the head dims"
    );
    let chunks = (head_dim / rotary_dim) as i32;
    let reshaped = values
        .clone()
        .reshape(&[dims[0], heads, chunks, rotary_dim as i32])
        .map_err(mlx_error("rope reshape"))?;
    let parts = reshaped
        .split_at_indices(&[1], 2)
        .map_err(mlx_error("rope split"))?;
    let first = parts[0]
        .clone()
        .reshape(&[dims[0], heads, rotary_dim as i32])
        .map_err(mlx_error("rope first"))?;
    let tail = parts[1]
        .clone()
        .reshape(&[
            dims[0],
            heads,
            chunks * rotary_dim as i32 - rotary_dim as i32,
        ])
        .map_err(mlx_error("rope tail"))?;
    let pairs = first
        .reshape(&[dims[0], heads, 2, half])
        .map_err(mlx_error("rope pairs"))?;
    let pair_split = pairs
        .split_at_indices(&[1], 2)
        .map_err(mlx_error("rope pair split"))?;
    let lo = pair_split[0]
        .clone()
        .reshape(&[dims[0], heads, half])
        .map_err(mlx_error("rope lo"))?;
    let hi = pair_split[1]
        .clone()
        .reshape(&[dims[0], heads, half])
        .map_err(mlx_error("rope hi"))?;
    let rotated_lo = lo.clone() * cos.clone() - hi.clone() * sin.clone();
    let rotated_hi = hi * cos + lo * sin;
    let rotated =
        mlx_rs::ops::concatenate(&[rotated_lo, rotated_hi], -1).map_err(mlx_error("rope join"))?;
    mlx_rs::ops::concatenate(&[rotated, tail], -1).map_err(mlx_error("rope concat"))
}

/// One text-side checkpoint tensor: FP32 host values plus its shape.
pub(crate) struct SurveyTensor {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
    pub array: Array,
}

/// Build an MLX array from a shape validated by `visit_safetensors`.
pub(crate) fn survey_array(values: &[f32], shape: &[usize]) -> Array {
    let shape: Vec<i32> = shape.iter().map(|&dim| dim as i32).collect();
    Array::from_slice(values, &shape)
}

/// Read small heads or encoders into host values and MLX arrays.
pub(crate) fn read_safetensors_filtered(
    path: &Path,
    keep: impl Fn(&str) -> bool,
) -> Result<BTreeMap<String, SurveyTensor>, FamilyError> {
    let mut weights = BTreeMap::new();
    visit_safetensors(path, keep, |name, values, shape| {
        let array = survey_array(&values, shape);
        weights.insert(
            name,
            SurveyTensor {
                values,
                shape: shape.to_vec(),
                array,
            },
        );
        Ok(())
    })?;
    Ok(weights)
}

/// Widen one selected tensor at a time from a verified, mapped shard.
/// The visitor consumes it before the next tensor is decoded, so it never
/// needs a second FP32 copy of the whole decoder beside the MLX arrays.
pub(crate) fn visit_safetensors(
    path: &Path,
    keep: impl Fn(&str) -> bool,
    mut visit: impl FnMut(String, Vec<f32>, &[usize]) -> Result<(), FamilyError>,
) -> Result<(), FamilyError> {
    use candle_core::{safetensors::MmapedSafetensors, DType};

    // SAFETY: callers verify the size and digest of this immutable shard
    // in its read-only source directory before invoking this loader.
    let mapped = unsafe { MmapedSafetensors::new(path) }?;
    let mut views = mapped.tensors();
    views.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (name, view) in views {
        if !keep(&name) {
            continue;
        }
        let shape = view.shape();
        if shape.iter().any(|&dim| i32::try_from(dim).is_err()) {
            return Err(FamilyError::InvalidInput(format!(
                "tensor `{name}` has a dimension that does not fit MLX"
            )));
        }
        let values = match DType::try_from(view.dtype())? {
            DType::BF16 => view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
                .collect(),
            DType::F32 => view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect(),
            other => {
                return Err(FamilyError::InvalidInput(format!(
                    "tensor `{name}` has unsupported checkpoint dtype {other:?}"
                )));
            }
        };
        visit(name, values, shape)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn streams_only_selected_weights_and_widens_bf16_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.safetensors");
        let tensors = std::collections::HashMap::from([
            (
                "model.language_model.layers.0.weight".to_owned(),
                Tensor::new(&[half::bf16::from_f32(1.5)], &Device::Cpu).unwrap(),
            ),
            (
                "vision.position_ids".to_owned(),
                Tensor::new(&[1_i64], &Device::Cpu).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let mut visited = Vec::new();
        visit_safetensors(
            &path,
            |name| name.starts_with("model.language_model.layers."),
            |name, values, shape| {
                assert_eq!(shape, &[1]);
                assert_eq!(values, [1.5]);
                visited.push(name);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(visited, ["model.language_model.layers.0.weight"]);
    }
}
