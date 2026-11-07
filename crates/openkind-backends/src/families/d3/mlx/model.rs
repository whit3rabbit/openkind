//! Pinned d3 checkpoint loading and noncausal forward execution on MLX.
//!
//! The same digest-verified artifacts as the CPU oracle stream into MLX
//! arrays: checkpoint tensors under the text-only `language_model.` tensor
//! namespace widen BF16 to FP32 exactly, the tied embedding table stays
//! host-resident (evaluation gathers only the prompt's rows), and the
//! separate float32 code readout never leaves the host. `d3-flash` instead
//! quantizes every layer projection to affine 4-bit (group 64) at load —
//! its FP32 weights exceed the reference host's wired limit — so its
//! projections run through `quantized_matmul` while every norm, gate, and
//! the readout stay FP32.
//!
//! The layer math mirrors the parity-verified CPU oracle
//! ([`crate::qwen35::backbone::layer0`]) at the profile's frozen geometry
//! instead of the pinned-geometry constants of the shared causal MLX
//! backbone: full-attention layers run the **noncausal** grouped-query
//! kernel (the mask is lifted; every position sees the whole prompt; rotary
//! positions stay absolute), the gated-DeltaNet layers run the causal
//! one-token recurrence from a zeroed state, and the per-layer mixer table
//! follows the profile's `full_attention_layers` so the pruned d3-edge
//! subset lands on the right layers.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! the process-wide [`MlxRuntime`] execution lock, per the workspace MLX
//! discipline.

use std::collections::BTreeMap;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::ops::{Div, Sub};
use std::path::Path;
use std::sync::Arc;

use mlx_rs::ops::indexing::TryIndexOp as _;
use mlx_rs::Array;

use crate::families::support::FamilyError;
use crate::qwen35::mlx::runtime::MlxRuntime;
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding, Qwen35Geometry};

use super::super::model::{read_rows, VerifiedArtifacts, EMBEDDING_TENSOR};
use super::super::{D3Profile, CODE_TABLE_WIDTH, VOCAB_SIZE};

/// RMS epsilon shared by every pinned d3 geometry. A future geometry with a
/// different epsilon must fail the load instead of silently computing the
/// wrong norms.
const RMS_EPSILON: f32 = 1e-6;

/// Affine quantization parameters of the `d3-flash` at-load packing.
pub(super) const GROUP_SIZE: i32 = 64;
pub(super) const BITS: i32 = 4;

/// One loaded tensor: affine-quantized, or a dense FP32 array.
pub(super) enum MlxWeight {
    Quantized {
        weight: Array,
        scales: Array,
        biases: Array,
    },
    Dense(Array),
}

fn mlx_error(operation: &'static str) -> impl Fn(mlx_rs::error::Exception) -> FamilyError {
    move |error| FamilyError::Mlx(format!("{operation}: {error}"))
}

/// Loaded pinned d3 model on MLX.
pub(super) struct D3MlxModel {
    tensors: BTreeMap<String, MlxWeight>,
    embedding: Qwen35Embedding,
    /// Row-major `[CODE_TABLE_WIDTH, hidden]` float32 readout rows.
    readout: Vec<f32>,
    geometry: Qwen35Geometry,
    runtime: Arc<MlxRuntime>,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared qwen35 MLX backbone).
unsafe impl Sync for D3MlxModel {}
unsafe impl Send for D3MlxModel {}

impl D3MlxModel {
    /// Load the verified checkpoint into MLX arrays.
    ///
    /// `quantize` selects the affine-4 `d3-flash` packing; every other
    /// profile loads dense FP32. The caller verified every artifact digest
    /// and holds the runtime lock for the whole load.
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        profile: &D3Profile,
        runtime: &Arc<MlxRuntime>,
        quantize: bool,
    ) -> Result<Self, FamilyError> {
        let geometry: Qwen35Geometry = profile.geometry;
        if geometry.rms_epsilon != RMS_EPSILON {
            return Err(FamilyError::InvalidInput(format!(
                "d3 MLX execution pins RMS epsilon {RMS_EPSILON}, the profile pins {}",
                geometry.rms_epsilon
            )));
        }
        let hidden_size = geometry.hidden_size;
        let mut tensors = BTreeMap::new();
        for shard in &artifacts.shards {
            for (name, entry) in read_safetensors(shard)? {
                let kept = name.starts_with("language_model.layers.")
                    || name == "language_model.norm.weight";
                if !kept {
                    // The tied embedding stays host-resident, and the unused
                    // vision tower and language-model head never load.
                    continue;
                }
                if tensors.insert(name.clone(), entry).is_some() {
                    return Err(FamilyError::InvalidInput(format!(
                        "duplicate tensor `{name}` across shards"
                    )));
                }
            }
        }
        if tensors.is_empty() {
            return Err(FamilyError::InvalidInput(
                "checkpoint shard carries no `language_model` tensors".to_owned(),
            ));
        }
        let mut model = Self {
            tensors,
            // The host-resident tied embedding over the verified shard: the
            // gather reads only the prompt's rows, exactly like the CPU path.
            embedding: {
                let shard = artifacts
                    .shards
                    .first()
                    .ok_or_else(|| FamilyError::InvalidInput("profile pins no shard".into()))?;
                let layout =
                    EmbeddingLayout::read_named(shard, EMBEDDING_TENSOR, VOCAB_SIZE, hidden_size)
                        .map_err(|error| FamilyError::InvalidInput(error.to_string()))?;
                Qwen35Embedding::from_layout(shard.clone(), layout)
            },
            readout: read_rows(&artifacts.readout, CODE_TABLE_WIDTH, hidden_size)?,
            geometry,
            runtime: Arc::clone(runtime),
        };
        if quantize {
            model.quantize_layer_projections()?;
        }
        for weight in model.tensors.values() {
            materialize_weight(weight)?;
        }
        Ok(model)
    }

    /// Affine-quantize every 2-D layer projection to 4-bit group 64.
    ///
    /// Used only by `d3-flash`, whose FP32 weights exceed the reference
    /// host's wired limit. Norms, gates, the convolution, and the host-side
    /// embedding and readout stay FP32.
    fn quantize_layer_projections(&mut self) -> Result<(), FamilyError> {
        let names: Vec<String> = self
            .tensors
            .keys()
            .filter(|name| {
                name.starts_with("language_model.layers.")
                    && (name.contains("self_attn.")
                        || name.contains("linear_attn.in_proj_")
                        || name.contains("linear_attn.out_proj")
                        || name.contains("mlp."))
                    && name.ends_with(".weight")
            })
            .cloned()
            .collect();
        for name in names {
            let Some(MlxWeight::Dense(array)) = self.tensors.remove(&name) else {
                continue;
            };
            let linear = mlx_rs::nn::Linear {
                weight: mlx_rs::module::Param::new(array),
                bias: mlx_rs::module::Param::new(None),
            };
            let quantized = mlx_rs::nn::QuantizedLinear::try_from_linear(linear, GROUP_SIZE, BITS)
                .map_err(mlx_error("affine quantize"))?;
            self.tensors.insert(
                name,
                MlxWeight::Quantized {
                    weight: quantized.inner.weight.value.clone(),
                    scales: quantized.scales.value.clone(),
                    biases: quantized.biases.value.clone(),
                },
            );
        }
        Ok(())
    }

    /// Code logits of the first `count` readout rows at the final prompt
    /// position. The readout matvec stays host-side in FP64, exactly as on
    /// the CPU reference path.
    pub(super) fn code_logits(
        &self,
        prompt_ids: &[u32],
        count: usize,
        control: &crate::families::support::FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        if count == 0 || count > CODE_TABLE_WIDTH {
            return Err(FamilyError::InvalidInput(format!(
                "code readout serves 1..={CODE_TABLE_WIDTH} codes, found {count}"
            )));
        }
        control.check()?;
        let hidden = self.runtime.execute(|| self.forward_hidden(prompt_ids))??;
        let final_token = &hidden[hidden.len() - self.geometry.hidden_size..];
        let mut logits = Vec::with_capacity(count);
        for row in self
            .readout
            .chunks_exact(self.geometry.hidden_size)
            .take(count)
        {
            let logit: f64 = row
                .iter()
                .zip(final_token)
                .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                .sum();
            if !logit.is_finite() {
                return Err(FamilyError::Numerical(
                    "code readout produced a non-finite logit".to_owned(),
                ));
            }
            logits.push(logit);
        }
        Ok(logits)
    }

    /// Full-sequence noncausal forward, returning the final-normed hidden
    /// states as host FP32 row-major `[token_count, hidden]` values.
    fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let geometry = self.geometry;
        let hidden_size = geometry.hidden_size;
        let token_count = input_ids.len();
        if token_count == 0 {
            return Err(FamilyError::InvalidInput(
                "d3 prompts carry at least one token".to_owned(),
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
            let prefix = format!("language_model.layers.{layer}");
            let normalized = rms_norm(
                &hidden,
                self.dense(&format!("{prefix}.input_layernorm.weight"))?,
            )?;
            let attention = if geometry.is_full_attention(layer) {
                self.full_attention(&prefix, &normalized, token_count)?
            } else {
                self.linear_attention(&prefix, &normalized, token_count)?
            };
            let mixed = hidden.clone() + attention;
            let post = rms_norm(
                &mixed,
                self.dense(&format!("{prefix}.post_attention_layernorm.weight"))?,
            )?;
            let gate = self.project(&format!("{prefix}.mlp.gate_proj.weight"), &post)?;
            let up = self.project(&format!("{prefix}.mlp.up_proj.weight"), &post)?;
            let activated = mlx_rs::nn::silu(&gate).map_err(mlx_error("mlp silu"))? * up;
            let down = self.project(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            hidden = mixed + down;
            // Materialize per layer so no forward graph spans more than one
            // block (the DeltaNet token loop is the quadratic-in-tokens part).
            hidden.eval().map_err(mlx_error("layer materialization"))?;
        }
        let final_hidden = rms_norm(&hidden, self.dense("language_model.norm.weight")?)?;
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

    fn get(&self, name: &str) -> Result<&MlxWeight, FamilyError> {
        self.tensors.get(name).ok_or_else(|| {
            FamilyError::InvalidInput(format!("tensor `{name}` missing from checkpoint"))
        })
    }

    /// A dense (never quantized) tensor: norms, gates, and the convolution.
    fn dense(&self, name: &str) -> Result<&Array, FamilyError> {
        match self.get(name)? {
            MlxWeight::Dense(tensor) => Ok(tensor),
            MlxWeight::Quantized { .. } => Err(FamilyError::InvalidInput(format!(
                "tensor `{name}` unexpectedly quantized"
            ))),
        }
    }

    /// `x @ w.T` through the quantized or dense kernel.
    fn project(&self, name: &str, x: &Array) -> Result<Array, FamilyError> {
        match self.get(name)? {
            MlxWeight::Quantized {
                weight,
                scales,
                biases,
            } => mlx_rs::ops::quantized_matmul(
                x,
                weight,
                scales,
                Some(biases),
                true,
                GROUP_SIZE,
                BITS,
            )
            .map_err(mlx_error("quantized matmul")),
            MlxWeight::Dense(tensor) => {
                let transposed = tensor.transpose().map_err(mlx_error("weight t"))?;
                x.matmul(&transposed).map_err(mlx_error("linear matmul"))
            }
        }
    }

    /// **Noncausal** grouped-query attention: every position attends to the
    /// whole prompt (the `noncausal_full_attention` mode lifts the causal
    /// mask; the rotary positions stay absolute). q/k carry per-head offset
    /// RMSNorm, the first quarter of each head rotates (NeoX pairing at
    /// theta 1e7), and the fused query/gate rows gate the mixed heads.
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
            &folded(self.dense(&format!("{prefix}.self_attn.q_norm.weight"))?)?,
        )?;
        let keys = norm_heads(
            &keys,
            &folded(self.dense(&format!("{prefix}.self_attn.k_norm.weight"))?)?,
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

        // Expand the grouped kv heads (query head h uses kv head h/group),
        // then one unmasked softmax over the whole prompt.
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

        let scale = (head_dim as f32).sqrt().recip();
        let scores = query
            .matmul(
                &keys_r
                    .transpose_axes(&[0, 2, 1])
                    .map_err(mlx_error("k t"))?,
            )
            .map_err(mlx_error("scores matmul"))?
            * Array::from_f32(scale);
        // Explicit max-subtracted softmax, matching the oracle's numerics.
        let maximum = scores
            .max_axis(-1, true)
            .map_err(mlx_error("softmax max"))?;
        let exponentiated = scores
            .sub(&maximum)
            .exp()
            .map_err(mlx_error("softmax exp"))?;
        let denominator = exponentiated
            .sum_axis(-1, true)
            .map_err(mlx_error("softmax sum"))?;
        let probabilities = exponentiated.div(&denominator);
        let mixed = probabilities
            .matmul(&values_r)
            .map_err(mlx_error("mix matmul"))?
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
    /// window and recurrent state (every d3 row is an independent full
    /// sequence; nothing is retained across rows).
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
            let array = self.dense(&format!("{prefix}.linear_attn.conv1d.weight"))?;
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
            let array = self.dense(name)?;
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
        let mut mixed_rows: Vec<Array> = Vec::with_capacity(token_count);
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
        }
        let stacked = mlx_rs::ops::stack(&mixed_rows, 0).map_err(mlx_error("mixed stack"))?;
        self.project(&format!("{prefix}.linear_attn.out_proj.weight"), &stacked)
    }
}

/// Take one row of `[rows, width]` as a flat `[width]` array.
fn take_row(array: &Array, index: &Array, width: usize) -> Result<Array, FamilyError> {
    array
        .take_axis(index, 0)
        .map_err(mlx_error("row take"))?
        .reshape(&[width as i32])
        .map_err(mlx_error("row reshape"))
}

/// Offset RMSNorm over the last axis, folding the raw `(1 + w)` weight the
/// way the CPU oracle computes it at runtime.
fn rms_norm(x: &Array, raw_weight: &Array) -> Result<Array, FamilyError> {
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
/// runtime (host FP32, once at load).
fn folded(weight: &Array) -> Result<Array, FamilyError> {
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
fn norm_heads(values: &Array, folded_weight: &Array) -> Result<Array, FamilyError> {
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
fn apply_rotary(
    values: &Array,
    token_count: usize,
    rotary_dim: usize,
) -> Result<Array, FamilyError> {
    let dims = values.shape();
    let heads = dims[1];
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

/// Read the text-side tensors of one digest-verified safetensors shard into
/// FP32 MLX arrays. BF16 widens exactly; the tied embedding, the unused
/// vision tower, the language-model head, and `__metadata__` entries stay
/// unread.
fn read_safetensors(path: &Path) -> Result<BTreeMap<String, MlxWeight>, FamilyError> {
    use std::fs::File;

    let mut file = File::open(path).map_err(|source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
        .map_err(|_| FamilyError::InvalidInput("safetensors header too large".into()))?;
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return Err(FamilyError::InvalidInput(format!(
            "invalid safetensors header length {header_len}"
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    #[derive(serde::Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [u64; 2],
    }
    let entries: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|error| FamilyError::Json {
            path: path.to_path_buf(),
            source: error,
        })?;
    let data_start = 8_u64 + header_len as u64;
    let mut weights = BTreeMap::new();
    for (name, value) in &entries {
        if name.starts_with("__metadata__") {
            continue;
        }
        let entry: Entry =
            serde_json::from_value(value.clone()).map_err(|error| FamilyError::Json {
                path: path.join(name),
                source: error,
            })?;
        let shape: Vec<i32> = entry.shape.iter().map(|dim| *dim as i32).collect();
        file.seek(SeekFrom::Start(data_start + entry.data_offsets[0]))
            .map_err(|source| FamilyError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let len = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
        let mut bytes = vec![0_u8; len];
        file.read_exact(&mut bytes)
            .map_err(|source| FamilyError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let array = match entry.dtype.as_str() {
            "BF16" => {
                let values: Vec<f32> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| f32::from(half::bf16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            "F32" => {
                let values: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            other => {
                return Err(FamilyError::InvalidInput(format!(
                    "tensor `{name}` has unsupported checkpoint dtype {other}"
                )));
            }
        };
        weights.insert(name.clone(), MlxWeight::Dense(array));
    }
    Ok(weights)
}

/// Evaluate every retained array so no unevaluated lazy graph crosses the
/// loading thread's boundary.
fn materialize_weight(weight: &MlxWeight) -> Result<(), FamilyError> {
    let eval = |label: &'static str, array: &Array| -> Result<(), FamilyError> {
        array
            .eval()
            .map_err(|error| FamilyError::Mlx(format!("{label}: {error}")))
    };
    match weight {
        MlxWeight::Dense(array) => eval("weight materialization", array),
        MlxWeight::Quantized {
            weight,
            scales,
            biases,
        } => {
            eval("packed weight materialization", weight)?;
            eval("scales materialization", scales)?;
            eval("biases materialization", biases)
        }
    }
}
