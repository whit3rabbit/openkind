//! MLX array implementation of the pinned laya encoder and typed head.
//!
//! Mirrors the candle CPU arithmetic in `families::modernbert` and
//! `families::laya::arch` op for op (FP32 compute over FP16-stored shards
//! upcast at load): weight-only embedding/final norms, layer 0 without
//! `attn_norm`, fused bias-free `Wqkv`, NeoX rotate-half RoPE with the
//! per-layer-type theta, symmetric sliding-window band on non-global layers,
//! gated GELU MLP, and the pre-norm bias-carrying head layers with the
//! marker scorer. The candle CPU path remains the correctness oracle.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use mlx_rs::ops;
use mlx_rs::Array;
use serde::Deserialize;

use crate::families::laya::LayaProfile;
use crate::qwen35::mlx::MlxError;

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

/// Weight-only LayerNorm: `(x - mean) / sqrt(var + eps) * weight`, matching
/// `candle_nn::LayerNorm` with no bias over the last axis.
fn layer_norm_weight_only(input: &Array, weight: &Array, eps: f64) -> Result<Array, MlxError> {
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
    normalized.multiply(weight).map_err(op("norm scale"))
}

/// Biased LayerNorm used by the head (`nn.LayerNorm` semantics).
fn layer_norm_biased(
    input: &Array,
    weight: &Array,
    bias: &Array,
    eps: f64,
) -> Result<Array, MlxError> {
    layer_norm_weight_only(input, weight, eps)?
        .add(bias)
        .map_err(op("head norm bias"))
}

/// Exact erf-based GELU, matching candle's `Activation::Gelu`:
/// `0.5 * x * (1 + erf(x / sqrt(2)))`.
fn gelu(input: &Array) -> Result<Array, MlxError> {
    let half = input.multiply(scalar(0.5)).map_err(op("gelu scale"))?;
    let erf_arg = input
        .multiply(scalar(std::f32::consts::FRAC_1_SQRT_2))
        .map_err(op("gelu arg"))?;
    let erf = ops::erf(&erf_arg).map_err(op("gelu erf"))?;
    half.multiply(&erf.add(scalar(1.0)).map_err(op("gelu one"))?)
        .map_err(op("gelu product"))
}

/// Raw safetensors tensor metadata.
#[derive(Debug, Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Sequential reader over the pinned single-file FP16 shard.
struct Shard {
    file: std::fs::File,
    data_base: u64,
    tensors: BTreeMap<String, RawTensor>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, MlxError> {
        let mut file = std::fs::File::open(path).map_err(|error| {
            MlxError::InvalidState(format!(
                "open pinned laya checkpoint {}: {error}",
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
        let mut tensors: BTreeMap<String, RawTensor> =
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

    /// Read one FP16 tensor, upcast it to FP32, and return the array with
    /// the pinned shape. Missing, mistyped, or reshaped tensors fail closed.
    fn f32_tensor(&mut self, name: &str, expected: &[usize]) -> Result<Array, MlxError> {
        let raw = self.tensors.get(name).ok_or_else(|| {
            MlxError::InvalidState(format!("pinned laya checkpoint is missing tensor `{name}`"))
        })?;
        if raw.dtype != "F16" {
            return Err(MlxError::InvalidState(format!(
                "tensor `{name}` stores {} but the pinned laya shard is FP16",
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
            .seek(SeekFrom::Start(self.data_base + raw.data_offsets[0]))
            .map_err(|error| MlxError::InvalidState(format!("seek `{name}`: {error}")))?;
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| MlxError::InvalidState(format!("read `{name}`: {error}")))?;
        let mut values = Vec::with_capacity(byte_len / 2);
        for pair in bytes.as_chunks::<2>().0 {
            values.push(f32::from(half::f16::from_le_bytes([pair[0], pair[1]])));
        }
        let shape: Vec<i32> = expected.iter().map(|&dim| dim as i32).collect();
        Ok(Array::from_slice(&values, &shape))
    }

    /// Read one FP16 matrix and materialize its FP32 transpose so every
    /// forward matmul consumes `(input, output)` weights directly.
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

/// NeoX rotate-half RoPE tables for one layer type: `cos`/`sin` shaped
/// `(max_seq, head_dim)` with the frequency halves duplicated, computed with
/// the same FP32 math as the candle reference tables.
struct RotaryTables {
    cos: Array,
    sin: Array,
}

impl RotaryTables {
    fn new(head_dim: usize, theta: f64, max_seq: usize) -> Result<Self, MlxError> {
        let half = head_dim / 2;
        let mut doubled = vec![0_f32; max_seq * head_dim];
        for position in 0..max_seq {
            for k in 0..half {
                let inv_freq = (1_f64 / theta.powf(2.0 * k as f64 / head_dim as f64)) as f32;
                let frequency = position as f32 * inv_freq;
                doubled[position * head_dim + k] = frequency;
                doubled[position * head_dim + half + k] = frequency;
            }
        }
        let shape = &[max_seq as i32, head_dim as i32];
        let frequencies = Array::from_slice(&doubled, shape);
        let cos = ops::cos(&frequencies).map_err(op("rope cos"))?;
        let sin = ops::sin(&frequencies).map_err(op("rope sin"))?;
        cos.eval().map_err(op("rope cos eval"))?;
        sin.eval().map_err(op("rope sin eval"))?;
        Ok(Self { cos, sin })
    }

    /// Apply rotate-half RoPE to `(heads, seq, head_dim)` queries or keys,
    /// slicing the tables to the sequence length and broadcasting over heads.
    fn apply(&self, x: &Array, seq: usize) -> Result<Array, MlxError> {
        let indices: Vec<u32> = (0..seq as u32).collect();
        let idx = Array::from_slice(&indices, &[seq as i32]);
        let cos = self.cos.take_axis(&idx, 0).map_err(op("rope cos slice"))?;
        let sin = self.sin.take_axis(&idx, 0).map_err(op("rope sin slice"))?;
        let split = x.split_equal(2, -1).map_err(op("rope split"))?;
        let (x1, x2) = (&split[0], &split[1]);
        let rotated = ops::concatenate(
            &[&x2.clone().negative().map_err(op("rope negate"))?, x1],
            -1,
        )
        .map_err(op("rope concat"))?;
        x.multiply(&cos)
            .map_err(op("rope cos mul"))?
            .add(&rotated.multiply(&sin).map_err(op("rope sin mul"))?)
            .map_err(op("rope add"))
    }
}

/// Bidirectional attention over one `(seq, hidden)` row with a fused QKV
/// projection. `rotary` is `Some` on encoder layers (NeoX RoPE) and `None`
/// on head layers (no positional encoding). `mask` is `Some` on non-global
/// encoder layers: the symmetric sliding-window band, zero inside
/// `|i - j| <= half_window` and negative infinity outside.
fn attention(
    xs: &Array,
    wqkv: &Array,
    wo: &Array,
    num_heads: usize,
    head_dim: usize,
    rotary: Option<&RotaryTables>,
    mask: Option<&Array>,
) -> Result<Array, MlxError> {
    let shape = xs.shape();
    let seq = shape[0];
    let hidden = shape[1];
    let qkv = xs.matmul(wqkv).map_err(op("attn qkv"))?;
    let qkv = qkv
        .reshape(&[seq, 3, num_heads as i32, head_dim as i32])
        .map_err(op("attn qkv reshape"))?;
    let parts = qkv.split_equal(3, 1).map_err(op("attn qkv split"))?;
    let head_shape = [seq, num_heads as i32, head_dim as i32];
    let mut q = parts[0]
        .reshape(&head_shape)
        .map_err(op("attn q reshape"))?
        .transpose_axes(&[1, 0, 2])
        .map_err(op("attn q transpose"))?;
    let mut k = parts[1]
        .reshape(&head_shape)
        .map_err(op("attn k reshape"))?
        .transpose_axes(&[1, 0, 2])
        .map_err(op("attn k transpose"))?;
    let v = parts[2]
        .reshape(&head_shape)
        .map_err(op("attn v reshape"))?
        .transpose_axes(&[1, 0, 2])
        .map_err(op("attn v transpose"))?;
    if let Some(rotary) = rotary {
        let seq = seq as usize;
        q = rotary.apply(&q, seq)?;
        k = rotary.apply(&k, seq)?;
    }
    let scale = (head_dim as f64).sqrt() as f32;
    let mut scores = q
        .matmul(&k.transpose_axes(&[0, 2, 1]).map_err(op("attn k^T"))?)
        .map_err(op("attn scores"))?;
    scores = scores.divide(scalar(scale)).map_err(op("attn scale"))?;
    if let Some(mask) = mask {
        scores = scores.add(mask).map_err(op("window add"))?;
    }
    let weights = ops::softmax_axis(&scores, -1, None).map_err(op("attn softmax"))?;
    let attended = weights.matmul(&v).map_err(op("attn value"))?;
    let attended = attended
        .transpose_axes(&[1, 0, 2])
        .map_err(op("attn transpose"))?
        .reshape(&[seq, hidden])
        .map_err(op("attn merge heads"))?;
    attended.matmul(wo).map_err(op("attn wo"))
}

/// One ModernBERT encoder layer in MLX. `attn_norm` is `None` on layer 0.
struct EncoderLayer {
    attn_norm: Option<Array>,
    wqkv: Array,
    wo: Array,
    mlp_norm: Array,
    wi: Array,
    wo_mlp: Array,
    is_global: bool,
}

impl EncoderLayer {
    fn load(shard: &mut Shard, profile: &LayaProfile, index: usize) -> Result<Self, MlxError> {
        let hidden = profile.encoder.hidden_size;
        let intermediate = profile.encoder.intermediate_size;
        let prefix = format!("encoder.layers.{index}");
        Ok(Self {
            attn_norm: if index == 0 {
                None
            } else {
                Some(shard.f32_tensor(&format!("{prefix}.attn_norm.weight"), &[hidden])?)
            },
            wqkv: shard.transposed_f32_tensor(
                &format!("{prefix}.attn.Wqkv.weight"),
                3 * hidden,
                hidden,
            )?,
            wo: shard.transposed_f32_tensor(&format!("{prefix}.attn.Wo.weight"), hidden, hidden)?,
            mlp_norm: shard.f32_tensor(&format!("{prefix}.mlp_norm.weight"), &[hidden])?,
            wi: shard.transposed_f32_tensor(
                &format!("{prefix}.mlp.Wi.weight"),
                2 * intermediate,
                hidden,
            )?,
            wo_mlp: shard.transposed_f32_tensor(
                &format!("{prefix}.mlp.Wo.weight"),
                hidden,
                intermediate,
            )?,
            is_global: index.is_multiple_of(3),
        })
    }

    fn forward(
        &self,
        xs: &Array,
        rotary_global: &RotaryTables,
        rotary_local: &RotaryTables,
        mask: Option<&Array>,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Array, MlxError> {
        let normed = match &self.attn_norm {
            Some(norm) => layer_norm_weight_only(xs, norm, 1e-5)?,
            None => xs.clone(),
        };
        let rotary = if self.is_global {
            rotary_global
        } else {
            rotary_local
        };
        let layer_mask = if self.is_global { None } else { mask };
        let attended = attention(
            &normed,
            &self.wqkv,
            &self.wo,
            num_heads,
            head_dim,
            Some(rotary),
            layer_mask,
        )?;
        let xs = xs.add(&attended).map_err(op("encoder residual"))?;
        let normed = layer_norm_weight_only(&xs, &self.mlp_norm, 1e-5)?;
        let projected = normed.matmul(&self.wi).map_err(op("mlp wi"))?;
        let halves = projected.split_equal(2, -1).map_err(op("mlp split"))?;
        let gated = gelu(&halves[0])?
            .multiply(&halves[1])
            .map_err(op("mlp gate"))?;
        let out = gated.matmul(&self.wo_mlp).map_err(op("mlp wo"))?;
        xs.add(&out).map_err(op("encoder mlp residual"))
    }
}

/// One pre-norm head encoder layer with bias-carrying projections
/// (`nn.TransformerEncoderLayer` semantics from the reference head).
struct HeadLayer {
    norm1_weight: Array,
    norm1_bias: Array,
    in_proj: Array,
    in_proj_bias: Array,
    out_proj: Array,
    out_proj_bias: Array,
    norm2_weight: Array,
    norm2_bias: Array,
    fc1: Array,
    fc1_bias: Array,
    fc2: Array,
    fc2_bias: Array,
}

impl HeadLayer {
    fn load(shard: &mut Shard, hidden: usize, prefix: &str) -> Result<Self, MlxError> {
        Ok(Self {
            norm1_weight: shard.f32_tensor(&format!("{prefix}.norm1.weight"), &[hidden])?,
            norm1_bias: shard.f32_tensor(&format!("{prefix}.norm1.bias"), &[hidden])?,
            in_proj: shard.transposed_f32_tensor(
                &format!("{prefix}.self_attn.in_proj_weight"),
                3 * hidden,
                hidden,
            )?,
            in_proj_bias: shard
                .f32_tensor(&format!("{prefix}.self_attn.in_proj_bias"), &[3 * hidden])?,
            out_proj: shard.transposed_f32_tensor(
                &format!("{prefix}.self_attn.out_proj.weight"),
                hidden,
                hidden,
            )?,
            out_proj_bias: shard
                .f32_tensor(&format!("{prefix}.self_attn.out_proj.bias"), &[hidden])?,
            norm2_weight: shard.f32_tensor(&format!("{prefix}.norm2.weight"), &[hidden])?,
            norm2_bias: shard.f32_tensor(&format!("{prefix}.norm2.bias"), &[hidden])?,
            fc1: shard.transposed_f32_tensor(
                &format!("{prefix}.linear1.weight"),
                4 * hidden,
                hidden,
            )?,
            fc1_bias: shard.f32_tensor(&format!("{prefix}.linear1.bias"), &[4 * hidden])?,
            fc2: shard.transposed_f32_tensor(
                &format!("{prefix}.linear2.weight"),
                hidden,
                4 * hidden,
            )?,
            fc2_bias: shard.f32_tensor(&format!("{prefix}.linear2.bias"), &[hidden])?,
        })
    }

    fn forward(&self, xs: &Array, num_heads: usize, head_dim: usize) -> Result<Array, MlxError> {
        let normed = layer_norm_biased(xs, &self.norm1_weight, &self.norm1_bias, 1e-5)?;
        let qkv = normed
            .matmul(&self.in_proj)
            .map_err(op("head qkv"))?
            .add(&self.in_proj_bias)
            .map_err(op("head qkv bias"))?;
        let seq = xs.shape()[0];
        let hidden = xs.shape()[1];
        let qkv = qkv
            .reshape(&[seq, 3, num_heads as i32, head_dim as i32])
            .map_err(op("head qkv reshape"))?;
        let parts = qkv.split_equal(3, 1).map_err(op("head qkv split"))?;
        let head_shape = [seq, num_heads as i32, head_dim as i32];
        let q = parts[0]
            .reshape(&head_shape)
            .map_err(op("head q reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("head q transpose"))?;
        let k = parts[1]
            .reshape(&head_shape)
            .map_err(op("head k reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("head k transpose"))?;
        let v = parts[2]
            .reshape(&head_shape)
            .map_err(op("head v reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("head v transpose"))?;
        let scale = (head_dim as f64).sqrt() as f32;
        let scores = q
            .matmul(&k.transpose_axes(&[0, 2, 1]).map_err(op("head k^T"))?)
            .map_err(op("head scores"))?
            .divide(scalar(scale))
            .map_err(op("head scale"))?;
        let weights = ops::softmax_axis(&scores, -1, None).map_err(op("head softmax"))?;
        let attended = weights
            .matmul(&v)
            .map_err(op("head value"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("head transpose"))?
            .reshape(&[seq, hidden])
            .map_err(op("head merge heads"))?;
        let attended = attended
            .matmul(&self.out_proj)
            .map_err(op("head out proj"))?
            .add(&self.out_proj_bias)
            .map_err(op("head out bias"))?;
        let xs = xs.add(&attended).map_err(op("head residual"))?;
        let normed = layer_norm_biased(&xs, &self.norm2_weight, &self.norm2_bias, 1e-5)?;
        let projected = normed
            .matmul(&self.fc1)
            .map_err(op("head fc1"))?
            .add(&self.fc1_bias)
            .map_err(op("head fc1 bias"))?;
        let relu = ops::maximum(&projected, scalar(0.0)).map_err(op("head relu"))?;
        let projected = relu
            .matmul(&self.fc2)
            .map_err(op("head fc2"))?
            .add(&self.fc2_bias)
            .map_err(op("head fc2 bias"))?;
        xs.add(&projected).map_err(op("head ffn residual"))
    }
}

/// The pinned laya model in MLX arrays: ModernBERT body plus typed head.
pub(super) struct LayaMlxModel {
    tok_embeddings: Array,
    embedding_norm: Array,
    layers: Vec<EncoderLayer>,
    final_norm: Array,
    rotary_global: RotaryTables,
    rotary_local: RotaryTables,
    /// Sliding-window band masks keyed by sequence length, built on demand.
    window_masks: std::sync::Mutex<std::collections::HashMap<usize, Array>>,
    head_type_emb: Array,
    head_layers: Vec<HeadLayer>,
    scorer_norm_weight: Array,
    scorer_norm_bias: Array,
    scorer_fc1: Array,
    scorer_fc1_bias: Array,
    scorer_fc2: Array,
    scorer_fc2_bias: Array,
    num_heads: usize,
    head_dim: usize,
    head_num_heads: usize,
    head_head_dim: usize,
    half_window: usize,
}

// SAFETY: every array evaluation in this model's methods runs under the
// process-wide `MlxRuntime` execution mutex (see the qwen35 `mlx` module
// docs); outside the lock, array handles are immutable refcounted values
// whose handle-only operations (clone, shape, dtype) are safe concurrently.
// The window-mask cache is additionally guarded by its own mutex. This is
// the same soundness argument as `unsafe impl Sync for MlxQwen35Backbone`,
// and it is what lets the bounded engine share one loaded model across the
// blocking-pool threads.
unsafe impl Send for LayaMlxModel {}
unsafe impl Sync for LayaMlxModel {}

impl LayaMlxModel {
    /// Load the digest-verified checkpoint into FP32 MLX arrays. Must run
    /// inside [`MlxRuntime::execute`]; every required tensor is checked
    /// against the pinned shape and any missing tensor fails closed.
    pub(super) fn load(profile: &LayaProfile, checkpoint: &Path) -> Result<Self, MlxError> {
        let mut shard = Shard::open(checkpoint)?;
        let hidden = profile.encoder.hidden_size;
        let vocab = profile.encoder.vocab_size;
        let num_heads = profile.encoder.num_attention_heads;
        let head_dim = hidden / num_heads;
        let head_num_heads = (hidden / 64).max(1);
        let head_head_dim = hidden / head_num_heads;
        let layers = (0..profile.encoder.num_hidden_layers)
            .map(|index| EncoderLayer::load(&mut shard, profile, index))
            .collect::<Result<Vec<_>, _>>()?;
        let head_layers = (0..2)
            .map(|index| HeadLayer::load(&mut shard, hidden, &format!("head.layers.{index}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            tok_embeddings: shard
                .f32_tensor("encoder.embeddings.tok_embeddings.weight", &[vocab, hidden])?,
            embedding_norm: shard.f32_tensor("encoder.embeddings.norm.weight", &[hidden])?,
            layers,
            final_norm: shard.f32_tensor("encoder.final_norm.weight", &[hidden])?,
            rotary_global: RotaryTables::new(
                head_dim,
                profile.encoder.global_rope_theta,
                profile.max_sequence_tokens,
            )?,
            rotary_local: RotaryTables::new(
                head_dim,
                profile.encoder.local_rope_theta,
                profile.max_sequence_tokens,
            )?,
            window_masks: std::sync::Mutex::new(std::collections::HashMap::new()),
            head_type_emb: shard.f32_tensor("type_emb.weight", &[3, hidden])?,
            head_layers,
            scorer_norm_weight: shard.f32_tensor("scorer.0.weight", &[hidden])?,
            scorer_norm_bias: shard.f32_tensor("scorer.0.bias", &[hidden])?,
            scorer_fc1: shard.transposed_f32_tensor("scorer.1.weight", hidden, hidden)?,
            scorer_fc1_bias: shard.f32_tensor("scorer.1.bias", &[hidden])?,
            scorer_fc2: shard.transposed_f32_tensor("scorer.3.weight", 1, hidden)?,
            scorer_fc2_bias: shard.f32_tensor("scorer.3.bias", &[1])?,
            num_heads,
            head_dim,
            head_num_heads,
            head_head_dim,
            half_window: 128 / 2,
        })
    }

    /// The sliding-window band mask for `seq` positions, built on CPU with
    /// the same layout as the candle reference and cached per length.
    fn window_mask(&self, seq: usize) -> Result<Array, MlxError> {
        let mut cache = self
            .window_masks
            .lock()
            .map_err(|_| MlxError::InvalidState("window mask cache poisoned".to_owned()))?;
        if let Some(mask) = cache.get(&seq) {
            return Ok(mask.clone());
        }
        let mut values = Vec::with_capacity(seq * seq);
        for i in 0..seq {
            for j in 0..seq {
                values.push(if i.abs_diff(j) <= self.half_window {
                    0.0
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        let mask = Array::from_slice(&values, &[seq as i32, seq as i32]);
        cache.insert(seq, mask.clone());
        Ok(mask)
    }

    /// One forward pass over the rendered sequence: encoder, type embedding,
    /// head layers, and one logit per marker in marker order.
    pub(super) fn option_logits(
        &self,
        token_ids: &[u32],
        markers: &[usize],
        qtype: usize,
    ) -> Result<Vec<f64>, MlxError> {
        let seq = token_ids.len();
        let ids = Array::from_slice(token_ids, &[seq as i32]);
        let mut xs = self
            .tok_embeddings
            .take_axis(&ids, 0)
            .map_err(op("embed tokens"))?;
        xs = layer_norm_weight_only(&xs, &self.embedding_norm, 1e-5)?;
        let mask = self.window_mask(seq)?;
        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                &self.rotary_global,
                &self.rotary_local,
                Some(&mask),
                self.num_heads,
                self.head_dim,
            )?;
        }
        xs = layer_norm_weight_only(&xs, &self.final_norm, 1e-5)?;

        let type_row = self
            .head_type_emb
            .take_axis(Array::from_slice(&[qtype as u32], &[1]), 0)
            .map_err(op("type row"))?;
        xs = xs.add(&type_row).map_err(op("type add"))?;
        for layer in &self.head_layers {
            xs = layer.forward(&xs, self.head_num_heads, self.head_head_dim)?;
        }

        let marker_ids: Vec<u32> = markers.iter().map(|&position| position as u32).collect();
        let gathered = xs
            .take_axis(
                Array::from_slice(&marker_ids, &[marker_ids.len() as i32]),
                0,
            )
            .map_err(op("marker gather"))?;
        let normed = layer_norm_biased(
            &gathered,
            &self.scorer_norm_weight,
            &self.scorer_norm_bias,
            1e-5,
        )?;
        let scored = normed
            .matmul(&self.scorer_fc1)
            .map_err(op("scorer fc1"))?
            .add(&self.scorer_fc1_bias)
            .map_err(op("scorer fc1 bias"))?;
        let scored = gelu(&scored)?;
        let logits = scored
            .matmul(&self.scorer_fc2)
            .map_err(op("scorer fc2"))?
            .add(&self.scorer_fc2_bias)
            .map_err(op("scorer fc2 bias"))?
            .reshape(&[marker_ids.len() as i32])
            .map_err(op("scorer squeeze"))?;
        logits.eval().map_err(op("logits eval"))?;
        Ok(logits
            .to_vec_cast::<f32>()
            .map_err(op("logits read"))?
            .into_iter()
            .map(f64::from)
            .collect())
    }
}
