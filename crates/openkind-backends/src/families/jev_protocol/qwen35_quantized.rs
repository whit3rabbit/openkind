//! Affine-quantized Qwen3.5 hybrid backbone on MLX (feature `mlx`, macOS
//! arm64) for the JEV decision model.
//!
//! The checkpoint stores every projection as an MLX affine triple (a packed
//! `U32` `weight` with `scales` and `biases`), the token embedding the same
//! way, the untied `lm_head` dense, and the vision tower unquantized. This
//! module runs the text decoder over those triples with
//! `mlx_rs::quantized_matmul`; it never dequantizes a projection matrix.
//!
//! It implements the same equations as the Candle CPU oracle in
//! [`crate::qwen35::backbone`] (offset-RMSNorm, causal depthwise convolution
//! with SiLU, gated DeltaNet with FP32 recurrent state, grouped-query
//! attention with a sigmoid output gate, partial NeoX rotary at theta 1e7,
//! SiLU-gated MLP) at any [`Qwen35Geometry`], and unlike the shared 4B MLX
//! layers it is not tied to the pinned 2,560-wide profile. Activations run in
//! FP32 over BF16 scales. All sequence-wide work is batched across tokens;
//! only the DeltaNet recurrence advances one token at a time.
//!
//! Callers must hold the process-wide MLX execution lock (see
//! `qwen35::mlx::runtime::MlxRuntime::execute`) around every method here.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_rs::ops::indexing::TryIndexOp as _;
use mlx_rs::{Array, Dtype};

use crate::families::support::FamilyError;
use crate::qwen35::Qwen35Geometry;

use super::{expected_tensors, QuantParams, Storage};

/// Tokens advanced between evaluations of the DeltaNet recurrence. Bounds the
/// retained lazy graph and the per-step recurrent states it would pin.
const RECURRENCE_CHUNK: usize = 32;
/// Additive mask value for future positions.
const MASKED: f32 = -1.0e9;

/// How norm weights are stored relative to the effective multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormConvention {
    /// Stored `w` multiplies as `(1 + w)` (original Hugging Face layout).
    Offset,
    /// Stored value already includes the `+1` (mlx-vlm converted layout).
    Folded,
}

/// One loaded tensor.
#[derive(Clone)]
pub enum Weight {
    /// MLX affine triple.
    Quantized {
        /// Packed `U32` elements.
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Per-group biases.
        biases: Array,
    },
    /// Ordinary dense tensor.
    Dense(Array),
}

fn mlx_error(operation: &'static str) -> impl Fn(mlx_rs::error::Exception) -> FamilyError {
    move |error| FamilyError::Mlx(format!("{operation}: {error}"))
}

fn scalar(value: f32) -> Array {
    Array::from_slice(&[value], &[])
}

fn f32_array(array: &Array, operation: &'static str) -> Result<Array, FamilyError> {
    array.as_dtype(Dtype::Float32).map_err(mlx_error(operation))
}

fn index_array(indices: &[u32]) -> Array {
    Array::from_slice(indices, &[indices.len() as i32])
}

/// Loaded quantized Qwen3.5 text backbone.
pub struct QuantizedQwen35 {
    tensors: BTreeMap<String, Weight>,
    geometry: Qwen35Geometry,
    quant: QuantParams,
    norms: NormConvention,
    /// Tensor-name prefix of the decoder, for example `language_model.model`.
    prefix: String,
    /// Full tensor name of the output embedding.
    lm_head: String,
}

// SAFETY: MLX array handles are immutable refcounted values; every
// evaluation happens under the caller-held process-wide MLX execution lock.
unsafe impl Sync for QuantizedQwen35 {}
unsafe impl Send for QuantizedQwen35 {}

impl QuantizedQwen35 {
    /// Wrap already-loaded tensors.
    pub fn new(
        tensors: BTreeMap<String, Weight>,
        geometry: Qwen35Geometry,
        quant: QuantParams,
        norms: NormConvention,
        prefix: &str,
        lm_head: &str,
    ) -> Self {
        Self {
            tensors,
            geometry,
            quant,
            norms,
            prefix: prefix.to_owned(),
            lm_head: lm_head.to_owned(),
        }
    }

    /// Geometry this backbone executes.
    pub fn geometry(&self) -> Qwen35Geometry {
        self.geometry
    }

    /// Check every required tensor for presence, storage kind, and shape
    /// against the layout contract of [`expected_tensors`].
    pub fn validate_layout(&self, vocab_size: usize) -> Result<(), FamilyError> {
        let packing = (32 / self.quant.bits) as usize;
        let group = self.quant.group_size as usize;
        let mismatch =
            |name: &str, expected: String, actual: String| FamilyError::ContractMismatch {
                field: "tensor_layout",
                expected: format!("{name}: {expected}"),
                actual,
            };
        let shape_of =
            |array: &Array| -> Vec<usize> { array.shape().iter().map(|d| *d as usize).collect() };
        for tensor in expected_tensors(&self.geometry, vocab_size) {
            match (&tensor.storage, self.get(&tensor.name)?) {
                (
                    Storage::Quantized { rows, cols },
                    Weight::Quantized {
                        weight,
                        scales,
                        biases,
                    },
                ) => {
                    let base = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
                    for (label, actual, expected) in [
                        ("weight", shape_of(weight), vec![*rows, cols / packing]),
                        ("scales", shape_of(scales), vec![*rows, cols / group]),
                        ("biases", shape_of(biases), vec![*rows, cols / group]),
                    ] {
                        if actual != expected {
                            return Err(mismatch(
                                &format!("{base}.{label}"),
                                format!("{expected:?}"),
                                format!("{actual:?}"),
                            ));
                        }
                    }
                }
                (Storage::Dense(shape), Weight::Dense(array)) => {
                    if &shape_of(array) != shape {
                        return Err(mismatch(
                            &tensor.name,
                            format!("{shape:?}"),
                            format!("{:?}", shape_of(array)),
                        ));
                    }
                }
                (storage, _) => {
                    return Err(mismatch(
                        &tensor.name,
                        format!("{storage:?}"),
                        "different storage kind".into(),
                    ))
                }
            }
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Result<&Weight, FamilyError> {
        self.tensors.get(name).ok_or_else(|| {
            FamilyError::InvalidInput(format!("tensor `{name}` missing from checkpoint"))
        })
    }

    fn dense(&self, name: &str) -> Result<&Array, FamilyError> {
        match self.get(name)? {
            Weight::Dense(array) => Ok(array),
            Weight::Quantized { .. } => Err(FamilyError::InvalidInput(format!(
                "tensor `{name}` is quantized but must be dense"
            ))),
        }
    }

    /// `x @ w.T` for `x` of shape `[rows, in]`, in FP32.
    fn project(&self, name: &str, x: &Array) -> Result<Array, FamilyError> {
        let output = match self.get(name)? {
            Weight::Quantized {
                weight,
                scales,
                biases,
            } => mlx_rs::ops::quantized_matmul(
                x,
                weight,
                scales,
                biases,
                true,
                self.quant.group_size,
                self.quant.bits,
            )
            .map_err(mlx_error("quantized matmul"))?,
            Weight::Dense(weight) => {
                let weight = f32_array(weight, "dense weight widen")?
                    .transpose()
                    .map_err(mlx_error("dense weight transpose"))?;
                x.matmul(&weight).map_err(mlx_error("dense matmul"))?
            }
        };
        f32_array(&output, "projection widen")
    }

    /// Effective RMSNorm multiplier as an FP32 vector of `width`.
    fn norm_weight(&self, name: &str, width: usize) -> Result<Array, FamilyError> {
        let tensor = self.dense(name)?;
        if tensor.shape() != [width as i32] {
            return Err(FamilyError::ContractMismatch {
                field: "norm_shape",
                expected: format!("{name}: [{width}]"),
                actual: format!("{:?}", tensor.shape()),
            });
        }
        let tensor = f32_array(tensor, "norm widen")?;
        Ok(match self.norms {
            NormConvention::Offset => tensor + scalar(1.0),
            NormConvention::Folded => tensor,
        })
    }

    /// Raw FP32 vector (no offset), for DeltaNet parameters.
    fn raw_vector(&self, name: &str, width: usize) -> Result<Array, FamilyError> {
        let tensor = self.dense(name)?;
        if tensor.shape() != [width as i32] {
            return Err(FamilyError::ContractMismatch {
                field: "vector_shape",
                expected: format!("{name}: [{width}]"),
                actual: format!("{:?}", tensor.shape()),
            });
        }
        f32_array(tensor, "vector widen")
    }

    fn rms_norm(&self, x: &Array, weight: &Array) -> Result<Array, FamilyError> {
        let variance = (x * x)
            .mean_axis(-1, Some(true))
            .map_err(mlx_error("rms mean"))?;
        let scale = (variance + scalar(self.geometry.rms_epsilon))
            .sqrt()
            .map_err(mlx_error("rms sqrt"))?
            .reciprocal()
            .map_err(mlx_error("rms recip"))?;
        Ok((x * scale) * weight)
    }

    fn layer_name(&self, layer: usize) -> String {
        format!("{}.layers.{layer}", self.prefix)
    }

    /// Embedding rows for `ids` as FP32 `[ids.len(), hidden]`.
    fn embed(&self, ids: &[u32]) -> Result<Array, FamilyError> {
        let name = format!("{}.embed_tokens.weight", self.prefix);
        self.rows(&name, ids)
    }

    /// Rows of a dense or quantized `[rows, hidden]` table, as FP32.
    fn rows(&self, name: &str, ids: &[u32]) -> Result<Array, FamilyError> {
        let ids = index_array(ids);
        let rows = match self.get(name)? {
            Weight::Quantized {
                weight,
                scales,
                biases,
            } => {
                let take = |tensor: &Array, label: &'static str| {
                    tensor.take_axis(&ids, 0).map_err(mlx_error(label))
                };
                mlx_rs::ops::dequantize(
                    &take(weight, "rows weight")?,
                    &take(scales, "rows scales")?,
                    &take(biases, "rows biases")?,
                    self.quant.group_size,
                    self.quant.bits,
                )
                .map_err(mlx_error("rows dequantize"))?
            }
            Weight::Dense(table) => table.take_axis(&ids, 0).map_err(mlx_error("rows take"))?,
        };
        f32_array(&rows, "rows widen")
    }

    /// Run every decoder layer and the final norm, returning the final-norm
    /// hidden state of the last token as host FP32 `[hidden]`.
    pub fn forward_last_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        if input_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "backbone input must contain at least one token".into(),
            ));
        }
        let geometry = self.geometry;
        let tokens = input_ids.len();
        let mut hidden = self.embed(input_ids)?;
        hidden.eval().map_err(mlx_error("embed eval"))?;
        for layer in 0..geometry.layer_count {
            let prefix = self.layer_name(layer);
            let normalized = self.rms_norm(
                &hidden,
                &self.norm_weight(
                    &format!("{prefix}.input_layernorm.weight"),
                    geometry.hidden_size,
                )?,
            )?;
            let mixer = if geometry.is_full_attention(layer) {
                self.full_attention(&prefix, &normalized, tokens)?
            } else {
                self.linear_attention(&prefix, &normalized, tokens)?
            };
            let mixed = &hidden + mixer;
            let post = self.rms_norm(
                &mixed,
                &self.norm_weight(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    geometry.hidden_size,
                )?,
            )?;
            let gate = self.project(&format!("{prefix}.mlp.gate_proj.weight"), &post)?;
            let up = self.project(&format!("{prefix}.mlp.up_proj.weight"), &post)?;
            let activated = mlx_rs::nn::silu(&gate).map_err(mlx_error("mlp silu"))? * up;
            let down = self.project(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            hidden = mixed + down;
            hidden.eval().map_err(mlx_error("layer eval"))?;
        }
        let last = hidden
            .take_axis(index_array(&[(tokens - 1) as u32]), 0)
            .map_err(mlx_error("last row"))?;
        let normalized = self.rms_norm(
            &last,
            &self.norm_weight(
                &format!("{}.norm.weight", self.prefix),
                geometry.hidden_size,
            )?,
        )?;
        host_f32(
            &normalized
                .reshape(&[-1])
                .map_err(mlx_error("last flatten"))?,
        )
    }

    /// Logits of the `token_ids` output rows against a final-norm `hidden`
    /// vector: `lm_head[token_ids] . hidden`, in FP64.
    pub fn logits_for_tokens(
        &self,
        hidden: &[f32],
        token_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        if hidden.len() != self.geometry.hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "hidden vector has {} elements, expected {}",
                hidden.len(),
                self.geometry.hidden_size
            )));
        }
        let rows = self.rows(&self.lm_head, token_ids)?;
        let column = Array::from_slice(hidden, &[hidden.len() as i32, 1]);
        let logits = rows
            .matmul(&column)
            .map_err(mlx_error("lm_head matmul"))?
            .reshape(&[-1])
            .map_err(mlx_error("lm_head flatten"))?;
        Ok(host_f32(&logits)?.into_iter().map(f64::from).collect())
    }

    fn linear_attention(
        &self,
        prefix: &str,
        normalized: &Array,
        tokens: usize,
    ) -> Result<Array, FamilyError> {
        let geometry = self.geometry;
        let (key_heads, value_heads, head_dim) =
            (geometry.key_heads, geometry.value_heads, geometry.head_dim);
        let (qkv_size, key_size, value_size) = (
            geometry.qkv_size(),
            geometry.key_size(),
            geometry.value_size(),
        );
        let kernel = geometry.conv_kernel;
        let name = |suffix: &str| format!("{prefix}.linear_attn.{suffix}");

        let raw_qkv = self.project(&name("in_proj_qkv.weight"), normalized)?;
        let z = self.project(&name("in_proj_z.weight"), normalized)?;
        let beta_raw = self.project(&name("in_proj_b.weight"), normalized)?;
        let decay_raw = self.project(&name("in_proj_a.weight"), normalized)?;

        // Causal depthwise convolution over the whole sequence: zero-pad the
        // front by `kernel - 1` rows, then tap `k` multiplies the row `k`
        // positions into the padded window.
        let conv_weight = f32_array(self.dense(&name("conv1d.weight"))?, "conv widen")?
            .reshape(&[qkv_size as i32, kernel as i32])
            .map_err(mlx_error("conv reshape"))?;
        let padding = Array::zeros::<f32>(&[(kernel - 1) as i32, qkv_size as i32])
            .map_err(mlx_error("conv padding"))?;
        let padded =
            mlx_rs::ops::concatenate(&[padding, raw_qkv], 0).map_err(mlx_error("conv concat"))?;
        let mut convolved: Option<Array> = None;
        for tap in 0..kernel {
            let window = padded
                .try_index((tap as i32..(tap + tokens) as i32, ..))
                .map_err(mlx_error("conv window"))?;
            let weight = conv_weight
                .take_axis(index_array(&[tap as u32]), 1)
                .map_err(mlx_error("conv tap"))?
                .reshape(&[qkv_size as i32])
                .map_err(mlx_error("conv tap reshape"))?;
            let term = window * weight;
            convolved = Some(match convolved {
                Some(sum) => sum + term,
                None => term,
            });
        }
        let qkv = mlx_rs::nn::silu(convolved.expect("kernel is at least one tap"))
            .map_err(mlx_error("conv silu"))?;

        let heads_of = |start: usize, end: usize, heads: usize| -> Result<Array, FamilyError> {
            qkv.try_index((.., start as i32..end as i32))
                .map_err(mlx_error("qkv slice"))?
                .reshape(&[tokens as i32, heads as i32, head_dim as i32])
                .map_err(mlx_error("qkv heads"))
        };
        let queries = heads_of(0, key_size, key_heads)?;
        let keys = heads_of(key_size, key_size * 2, key_heads)?;
        let values = heads_of(key_size * 2, qkv_size, value_heads)?;

        // L2 normalization (epsilon inside the root, as the oracle does); the
        // query additionally carries 1/sqrt(head_dim).
        let l2 = |heads: &Array, multiplier: f32| -> Result<Array, FamilyError> {
            let norm = ((heads * heads)
                .sum_axis(-1, Some(true))
                .map_err(mlx_error("l2 sum"))?
                + scalar(geometry.rms_epsilon))
            .sqrt()
            .map_err(mlx_error("l2 sqrt"))?
            .reciprocal()
            .map_err(mlx_error("l2 recip"))?;
            Ok(heads * norm * scalar(multiplier))
        };
        let queries = l2(&queries, (head_dim as f32).sqrt().recip())?;
        let keys = l2(&keys, 1.0)?;

        // Value head `v` reads key head `v / (value_heads / key_heads)`.
        let group = value_heads / key_heads;
        let key_index: Vec<u32> = (0..value_heads).map(|head| (head / group) as u32).collect();
        let key_index = index_array(&key_index);
        let expand = |heads: &Array| -> Result<Array, FamilyError> {
            heads
                .take_axis(&key_index, 1)
                .map_err(mlx_error("head expand"))
        };
        let queries = expand(&queries)?;
        let keys = expand(&keys)?;

        let beta = mlx_rs::ops::sigmoid(&beta_raw).map_err(mlx_error("beta sigmoid"))?;
        let dt_bias = self.raw_vector(&name("dt_bias"), value_heads)?;
        let a_log = self.raw_vector(&name("A_log"), value_heads)?;
        let softplus =
            mlx_rs::nn::softplus(&(decay_raw + dt_bias)).map_err(mlx_error("softplus"))?;
        let decay = mlx_rs::ops::exp(
            &(-mlx_rs::ops::exp(&a_log).map_err(mlx_error("a_log exp"))? * softplus),
        )
        .map_err(mlx_error("decay exp"))?;

        // Row/column views per token for the recurrence matmuls.
        let tokens_i = tokens as i32;
        let (vh, hd) = (value_heads as i32, head_dim as i32);
        let key_rows = keys
            .reshape(&[tokens_i, vh, 1, hd])
            .map_err(mlx_error("k rows"))?;
        let key_cols = keys
            .reshape(&[tokens_i, vh, hd, 1])
            .map_err(mlx_error("k cols"))?;
        let query_rows = queries
            .reshape(&[tokens_i, vh, 1, hd])
            .map_err(mlx_error("q rows"))?;
        let value_rows = values
            .reshape(&[tokens_i, vh, 1, hd])
            .map_err(mlx_error("v rows"))?;
        let beta_scalars = beta
            .reshape(&[tokens_i, vh, 1, 1])
            .map_err(mlx_error("beta shape"))?;
        let decay_scalars = decay
            .reshape(&[tokens_i, vh, 1, 1])
            .map_err(mlx_error("decay shape"))?;
        let split = |array: &Array| -> Result<Vec<Array>, FamilyError> {
            mlx_rs::ops::split_equal(array, tokens_i, 0).map_err(mlx_error("token split"))
        };
        let (key_rows, key_cols, query_rows, value_rows, beta_scalars, decay_scalars) = (
            split(&key_rows)?,
            split(&key_cols)?,
            split(&query_rows)?,
            split(&value_rows)?,
            split(&beta_scalars)?,
            split(&decay_scalars)?,
        );

        // State layout S[head, key, value]; memory = k^T S, S += k (x) delta.
        let mut state = Array::zeros::<f32>(&[vh, hd, hd]).map_err(mlx_error("state init"))?;
        let mut chunks: Vec<Array> = Vec::new();
        let mut pending: Vec<Array> = Vec::with_capacity(RECURRENCE_CHUNK);
        for token in 0..tokens {
            let rows_shape = [vh, 1, hd];
            let squeeze = |array: &Array| -> Result<Array, FamilyError> {
                array
                    .reshape(&rows_shape)
                    .map_err(mlx_error("token squeeze"))
            };
            let key_row = squeeze(&key_rows[token])?;
            let key_col = key_cols[token]
                .reshape(&[vh, hd, 1])
                .map_err(mlx_error("token key col"))?;
            let query_row = squeeze(&query_rows[token])?;
            let value_row = squeeze(&value_rows[token])?;
            let beta_t = beta_scalars[token]
                .reshape(&[vh, 1, 1])
                .map_err(mlx_error("token beta"))?;
            let decay_t = decay_scalars[token]
                .reshape(&[vh, 1, 1])
                .map_err(mlx_error("token decay"))?;
            state = &state * decay_t;
            let memory = key_row.matmul(&state).map_err(mlx_error("memory matmul"))?;
            let delta = (value_row - memory) * beta_t;
            state = &state + key_col.matmul(&delta).map_err(mlx_error("update matmul"))?;
            pending.push(
                query_row
                    .matmul(&state)
                    .map_err(mlx_error("readout matmul"))?
                    .reshape(&[1, vh, hd])
                    .map_err(mlx_error("readout shape"))?,
            );
            if pending.len() == RECURRENCE_CHUNK || token + 1 == tokens {
                let chunk =
                    mlx_rs::ops::concatenate(&pending, 0).map_err(mlx_error("chunk concat"))?;
                chunk.eval().map_err(mlx_error("chunk eval"))?;
                state.eval().map_err(mlx_error("state eval"))?;
                chunks.push(chunk);
                pending.clear();
            }
        }
        // [tokens, value_heads, head_dim]
        let recurrent =
            mlx_rs::ops::concatenate(&chunks, 0).map_err(mlx_error("recurrent concat"))?;

        // Gated RMSNorm (raw weight) then SiLU(z).
        let norm_weight = self.raw_vector(&name("norm.weight"), head_dim)?;
        let scale = ((&recurrent * &recurrent)
            .mean_axis(-1, Some(true))
            .map_err(mlx_error("delta mean"))?
            + scalar(geometry.rms_epsilon))
        .sqrt()
        .map_err(mlx_error("delta sqrt"))?
        .reciprocal()
        .map_err(mlx_error("delta recip"))?;
        let gate = mlx_rs::nn::silu(
            &z.reshape(&[tokens_i, vh, hd])
                .map_err(mlx_error("z heads"))?,
        )
        .map_err(mlx_error("z silu"))?;
        let mixed = ((recurrent * scale) * norm_weight * gate)
            .reshape(&[tokens_i, value_size as i32])
            .map_err(mlx_error("mixed flatten"))?;
        self.project(&name("out_proj.weight"), &mixed)
    }

    fn full_attention(
        &self,
        prefix: &str,
        normalized: &Array,
        tokens: usize,
    ) -> Result<Array, FamilyError> {
        let geometry = self.geometry;
        let (heads, kv_heads, head_dim) = (
            geometry.attention_heads,
            geometry.kv_heads,
            geometry.attention_head_dim,
        );
        let rotary_dim = geometry.rotary_dim();
        let name = |suffix: &str| format!("{prefix}.self_attn.{suffix}");
        let (tokens_i, heads_i, kv_i, dim_i) = (
            tokens as i32,
            heads as i32,
            kv_heads as i32,
            head_dim as i32,
        );

        let projected_q = self
            .project(&name("q_proj.weight"), normalized)?
            .reshape(&[tokens_i, heads_i, 2, dim_i])
            .map_err(mlx_error("q reshape"))?;
        let keys = self.project(&name("k_proj.weight"), normalized)?;
        let values = self.project(&name("v_proj.weight"), normalized)?;

        // Per-head query and sigmoid-gate rows are interleaved in q_proj.
        let part = |index: u32| -> Result<Array, FamilyError> {
            projected_q
                .take_axis(index_array(&[index]), 2)
                .map_err(mlx_error("q part"))?
                .reshape(&[tokens_i, heads_i, dim_i])
                .map_err(mlx_error("q part reshape"))
        };
        let queries = part(0)?;
        let gates = part(1)?;
        let keys = keys
            .reshape(&[tokens_i, kv_i, dim_i])
            .map_err(mlx_error("k reshape"))?;
        let values = values
            .reshape(&[tokens_i, kv_i, dim_i])
            .map_err(mlx_error("v reshape"))?;

        let q_norm = self.norm_weight(&name("q_norm.weight"), head_dim)?;
        let k_norm = self.norm_weight(&name("k_norm.weight"), head_dim)?;
        let queries = self.rms_norm(&queries, &q_norm)?;
        let keys = self.rms_norm(&keys, &k_norm)?;

        // Partial NeoX rotary over the first `rotary_dim` entries. cos/sin
        // are `[tokens, 1, rotary/2]` and broadcast over heads.
        let half = rotary_dim / 2;
        let mut cos_table = Vec::with_capacity(tokens * half);
        let mut sin_table = Vec::with_capacity(tokens * half);
        for position in 0..tokens {
            for index in 0..half {
                let frequency = f64::from(Qwen35Geometry::ROPE_THETA)
                    .powf(-((2 * index) as f64) / rotary_dim as f64);
                let (sin, cos) = (position as f64 * frequency).sin_cos();
                cos_table.push(cos as f32);
                sin_table.push(sin as f32);
            }
        }
        let table_shape = [tokens_i, 1, half as i32];
        let cos = Array::from_slice(&cos_table, &table_shape);
        let sin = Array::from_slice(&sin_table, &table_shape);
        let rotate = |x: &Array| -> Result<Array, FamilyError> {
            let first = x
                .try_index((.., .., 0..half as i32))
                .map_err(mlx_error("rope first"))?;
            let second = x
                .try_index((.., .., half as i32..rotary_dim as i32))
                .map_err(mlx_error("rope second"))?;
            let tail = x
                .try_index((.., .., rotary_dim as i32..dim_i))
                .map_err(mlx_error("rope tail"))?;
            let rotated_first = &first * &cos - &second * &sin;
            let rotated_second = &second * &cos + &first * &sin;
            mlx_rs::ops::concatenate(&[rotated_first, rotated_second, tail], 2)
                .map_err(mlx_error("rope concat"))
        };
        let queries = rotate(&queries)?;
        let keys = rotate(&keys)?;

        // Grouped-query attention: query head `h` reads KV head `h / group`.
        let group = heads / kv_heads;
        let kv_index: Vec<u32> = (0..heads).map(|head| (head / group) as u32).collect();
        let kv_index = index_array(&kv_index);
        let expand = |tensor: &Array| -> Result<Array, FamilyError> {
            tensor
                .take_axis(&kv_index, 1)
                .map_err(mlx_error("kv expand"))?
                .transpose_axes(&[1, 0, 2])
                .map_err(mlx_error("kv heads first"))
        };
        let keys = expand(&keys)?; // [heads, tokens, dim]
        let values = expand(&values)?;
        let queries = queries
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("q heads first"))?;

        let mut mask = vec![0.0_f32; tokens * tokens];
        for row in 0..tokens {
            for column in row + 1..tokens {
                mask[row * tokens + column] = MASKED;
            }
        }
        let mask = Array::from_slice(&mask, &[tokens_i, tokens_i]);
        let scores = queries
            .matmul(
                &keys
                    .transpose_axes(&[0, 2, 1])
                    .map_err(mlx_error("k transpose"))?,
            )
            .map_err(mlx_error("scores"))?
            * scalar((head_dim as f32).sqrt().recip())
            + mask;
        let probabilities =
            mlx_rs::ops::softmax_axis(&scores, -1, false).map_err(mlx_error("softmax"))?;
        let mixed = probabilities
            .matmul(&values)
            .map_err(mlx_error("mix"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("mix tokens first"))?;
        let gated = mixed * mlx_rs::ops::sigmoid(&gates).map_err(mlx_error("gate sigmoid"))?;
        let flat = gated
            .reshape(&[tokens_i, geometry.attention_size() as i32])
            .map_err(mlx_error("gate flatten"))?;
        self.project(&name("o_proj.weight"), &flat)
    }
}

fn host_f32(array: &Array) -> Result<Vec<f32>, FamilyError> {
    let widened = f32_array(array, "host widen")?;
    widened.eval().map_err(mlx_error("host eval"))?;
    widened
        .try_as_slice::<f32>()
        .map(<[f32]>::to_vec)
        .map_err(|error| FamilyError::Mlx(format!("host read: {error}")))
}

/// Read every tensor in one safetensors file, grouping `X.weight`,
/// `X.scales`, `X.biases` into [`Weight::Quantized`] and skipping tensors
/// whose name starts with any of `skip_prefixes`.
///
/// Shards are read in place one tensor at a time; nothing is staged on disk.
pub fn read_safetensors(
    path: &Path,
    skip_prefixes: &[&str],
    into: &mut BTreeMap<String, Weight>,
) -> Result<(), FamilyError> {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let io = |source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = std::fs::File::open(path).map_err(io)?;
    let mut length = [0_u8; 8];
    file.read_exact(&mut length).map_err(io)?;
    let header_len = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| FamilyError::InvalidInput("safetensors header too large".into()))?;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Err(FamilyError::InvalidInput(format!(
            "invalid safetensors header length {header_len}"
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header).map_err(io)?;

    #[derive(serde::Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [u64; 2],
    }
    let entries: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|source| FamilyError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let entry = |name: &str| -> Result<Entry, FamilyError> {
        let value = entries.get(name).ok_or_else(|| {
            FamilyError::InvalidInput(format!("tensor `{name}` missing from `{}`", path.display()))
        })?;
        serde_json::from_value(value.clone()).map_err(|source| FamilyError::Json {
            path: path.to_path_buf(),
            source,
        })
    };
    let data_start = 8_u64 + header_len as u64;
    let mut read_array = |entry: &Entry| -> Result<Array, FamilyError> {
        file.seek(SeekFrom::Start(data_start + entry.data_offsets[0]))
            .map_err(io)?;
        let mut bytes = vec![0_u8; (entry.data_offsets[1] - entry.data_offsets[0]) as usize];
        file.read_exact(&mut bytes).map_err(io)?;
        let shape: Vec<i32> = entry.shape.iter().map(|dim| *dim as i32).collect();
        let array = match entry.dtype.as_str() {
            "U32" => Array::from_slice(
                &bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>(),
                &shape,
            ),
            "F32" => Array::from_slice(
                &bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>(),
                &shape,
            ),
            "BF16" => Array::from_slice(
                &bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                    .collect::<Vec<_>>(),
                &shape,
            ),
            "F16" => Array::from_slice(
                &bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect::<Vec<_>>(),
                &shape,
            ),
            other => {
                return Err(FamilyError::InvalidInput(format!(
                    "unsupported safetensors dtype `{other}`"
                )))
            }
        };
        array.eval().map_err(mlx_error("tensor materialization"))?;
        Ok(array)
    };

    for name in entries.keys() {
        if name == "__metadata__"
            || skip_prefixes.iter().any(|prefix| name.starts_with(prefix))
            || name.ends_with(".scales")
            || name.ends_with(".biases")
        {
            continue;
        }
        // MLX names the siblings of `X.weight` as `X.scales` and `X.biases`.
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let weight = if entries.contains_key(&format!("{base}.scales")) {
            let biases = format!("{base}.biases");
            if !entries.contains_key(&biases) {
                return Err(FamilyError::InvalidInput(format!(
                    "quantized tensor `{name}` has scales but no biases"
                )));
            }
            Weight::Quantized {
                weight: read_array(&entry(name)?)?,
                scales: read_array(&entry(&format!("{base}.scales"))?)?,
                biases: read_array(&entry(&biases)?)?,
            }
        } else {
            Weight::Dense(read_array(&entry(name)?)?)
        };
        if into.insert(name.clone(), weight).is_some() {
            return Err(FamilyError::InvalidInput(format!(
                "duplicate tensor `{name}` across shards"
            )));
        }
    }
    Ok(())
}
