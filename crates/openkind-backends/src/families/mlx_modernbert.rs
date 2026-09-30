//! Shared MLX array implementation of the ModernBERT encoder body.
//!
//! Mirrors the candle CPU arithmetic in [`crate::families::modernbert`] op
//! for op (FP32 compute over pinned shards, upcast at load): weight-only
//! embedding/final norms, layer 0 without `attn_norm`, fused bias-free
//! `Wqkv`, NeoX rotate-half RoPE with the per-layer-type theta, symmetric
//! sliding-window band on non-global layers, and gated GELU MLP. The candle
//! CPU path remains the correctness oracle.
//!
//! Every family whose backbone is a ModernBERT-shaped encoder loads this
//! body behind its own family-specific head, so the shared numerical path
//! cannot drift between them.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use mlx_rs::ops;
use mlx_rs::Array;
use serde::Deserialize;

use crate::qwen35::mlx::MlxError;

/// Map an operation name onto an [`MlxError::Operation`] constructor.
pub(crate) fn op<E: std::fmt::Display>(name: &'static str) -> impl Fn(E) -> MlxError {
    move |error| MlxError::Operation {
        operation: name,
        message: error.to_string(),
    }
}

/// One-element FP32 array used as a broadcast scalar.
pub(crate) fn scalar(value: f32) -> Array {
    Array::from_slice(&[value], &[1])
}

/// Weight-only LayerNorm: `(x - mean) / sqrt(var + eps) * weight`, matching
/// `candle_nn::LayerNorm` with no bias over the last axis.
pub(crate) fn layer_norm_weight_only(
    input: &Array,
    weight: &Array,
    eps: f64,
) -> Result<Array, MlxError> {
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

/// Biased LayerNorm used by family heads (`nn.LayerNorm` semantics):
/// weight-only normalization followed by an affine bias add.
pub(crate) fn layer_norm_biased(
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
pub(crate) fn gelu(input: &Array) -> Result<Array, MlxError> {
    let half = input.multiply(scalar(0.5)).map_err(op("gelu scale"))?;
    let erf_arg = input
        .multiply(scalar(std::f32::consts::FRAC_1_SQRT_2))
        .map_err(op("gelu arg"))?;
    let erf = ops::erf(&erf_arg).map_err(op("gelu erf"))?;
    half.multiply(&erf.add(scalar(1.0)).map_err(op("gelu one"))?)
        .map_err(op("gelu product"))
}

/// Storage dtype the pinned shard must carry. The reader fails closed on any
/// other dtype so a re-converted checkpoint cannot silently change the
/// arithmetic of a family's execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PinnedDtype {
    /// Half-precision checkpoint (the pinned laya shards).
    F16,
    /// Full-precision checkpoint (the pinned GLiClass shard).
    F32,
}

impl PinnedDtype {
    fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }
}

/// Raw safetensors tensor metadata.
#[derive(Debug, Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Sequential reader over a pinned single-file safetensors shard.
pub(crate) struct Shard {
    file: std::fs::File,
    data_base: u64,
    tensors: BTreeMap<String, RawTensor>,
    dtype: PinnedDtype,
}

impl Shard {
    /// Open the shard and decode its header without reading any tensor.
    pub(crate) fn open(path: &Path, dtype: PinnedDtype) -> Result<Self, MlxError> {
        let mut file = std::fs::File::open(path).map_err(|error| {
            MlxError::InvalidState(format!(
                "open pinned checkpoint {}: {error}",
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
        // Metadata is a free-form string map, not tensor metadata: strip it
        // before the strict per-tensor decode.
        let mut raw_header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|error| {
            MlxError::InvalidState(format!("decode safetensors header: {error}"))
        })?;
        raw_header.remove("__metadata__");
        let mut tensors = BTreeMap::new();
        for (name, value) in raw_header {
            let tensor: RawTensor = serde_json::from_value(value).map_err(|error| {
                MlxError::InvalidState(format!("decode safetensors tensor `{name}`: {error}"))
            })?;
            tensors.insert(name, tensor);
        }
        Ok(Self {
            file,
            data_base: 8 + header_len,
            tensors,
            dtype,
        })
    }

    /// Read one tensor in the pinned dtype, upcast it to FP32, and return the
    /// array with the pinned shape. Missing, mistyped, or reshaped tensors
    /// fail closed.
    pub(crate) fn f32_tensor(&mut self, name: &str, expected: &[usize]) -> Result<Array, MlxError> {
        let raw = self.tensors.get(name).ok_or_else(|| {
            MlxError::InvalidState(format!("pinned checkpoint is missing tensor `{name}`"))
        })?;
        let stored = raw.dtype.as_str();
        if stored != self.dtype.as_str() {
            return Err(MlxError::InvalidState(format!(
                "tensor `{name}` stores {stored} but the pinned checkpoint is {}",
                self.dtype.as_str()
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
        let (shape, values): (Vec<i32>, Vec<f32>) = match self.dtype {
            PinnedDtype::F32 => {
                let values = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect();
                (expected.iter().map(|&dim| dim as i32).collect(), values)
            }
            PinnedDtype::F16 => {
                let values = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| f32::from(half::f16::from_le_bytes(*pair)))
                    .collect();
                (expected.iter().map(|&dim| dim as i32).collect(), values)
            }
        };
        Ok(Array::from_slice(&values, &shape))
    }

    /// Read one matrix and materialize its FP32 transpose so every forward
    /// matmul consumes `(input, output)` weights directly.
    pub(crate) fn transposed_f32_tensor(
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

/// Configuration subset the MLX body needs, pinned by each family from its
/// checkpoint contract (mirror of [`crate::families::modernbert`]
/// `ModernBertConfig`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModernBertMlxConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) intermediate_size: usize,
    /// Local (sliding) attention window; the band half-width is half of it.
    pub(crate) local_attention: usize,
    /// Every layer whose index divides this runs full attention.
    pub(crate) global_attn_every_n_layers: usize,
    pub(crate) global_rope_theta: f64,
    pub(crate) local_rope_theta: f64,
    pub(crate) norm_eps: f64,
    /// Maximum sequence length the rope tables are built for.
    pub(crate) max_sequence_tokens: usize,
}

/// NeoX rotate-half RoPE tables for one layer type: `cos`/`sin` shaped
/// `(max_seq, head_dim)` with the frequency halves duplicated, computed with
/// the same FP32 math as the candle reference tables.
pub(crate) struct RotaryTables {
    cos: Array,
    sin: Array,
}

impl RotaryTables {
    pub(crate) fn new(head_dim: usize, theta: f64, max_seq: usize) -> Result<Self, MlxError> {
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
    pub(crate) fn apply(&self, x: &Array, seq: usize) -> Result<Array, MlxError> {
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
pub(crate) fn attention(
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
pub(crate) struct EncoderLayer {
    attn_norm: Option<Array>,
    wqkv: Array,
    wo: Array,
    mlp_norm: Array,
    wi: Array,
    wo_mlp: Array,
    is_global: bool,
    norm_eps: f64,
}

impl EncoderLayer {
    fn load(
        shard: &mut Shard,
        config: &ModernBertMlxConfig,
        prefix: &str,
        index: usize,
    ) -> Result<Self, MlxError> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let layer_prefix = format!("{prefix}.layers.{index}");
        Ok(Self {
            attn_norm: if index == 0 {
                None
            } else {
                Some(shard.f32_tensor(&format!("{layer_prefix}.attn_norm.weight"), &[hidden])?)
            },
            wqkv: shard.transposed_f32_tensor(
                &format!("{layer_prefix}.attn.Wqkv.weight"),
                3 * hidden,
                hidden,
            )?,
            wo: shard.transposed_f32_tensor(
                &format!("{layer_prefix}.attn.Wo.weight"),
                hidden,
                hidden,
            )?,
            mlp_norm: shard.f32_tensor(&format!("{layer_prefix}.mlp_norm.weight"), &[hidden])?,
            wi: shard.transposed_f32_tensor(
                &format!("{layer_prefix}.mlp.Wi.weight"),
                2 * intermediate,
                hidden,
            )?,
            wo_mlp: shard.transposed_f32_tensor(
                &format!("{layer_prefix}.mlp.Wo.weight"),
                hidden,
                intermediate,
            )?,
            is_global: index.is_multiple_of(config.global_attn_every_n_layers),
            norm_eps: config.norm_eps,
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
            Some(norm) => layer_norm_weight_only(xs, norm, self.norm_eps)?,
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
        let normed = layer_norm_weight_only(&xs, &self.mlp_norm, self.norm_eps)?;
        let projected = normed.matmul(&self.wi).map_err(op("mlp wi"))?;
        let halves = projected.split_equal(2, -1).map_err(op("mlp split"))?;
        let gated = gelu(&halves[0])?
            .multiply(&halves[1])
            .map_err(op("mlp gate"))?;
        let out = gated.matmul(&self.wo_mlp).map_err(op("mlp wo"))?;
        xs.add(&out).map_err(op("encoder mlp residual"))
    }
}

/// The ModernBERT encoder body in MLX arrays: embeddings, layers, final norm.
pub(crate) struct ModernBertMlx {
    tok_embeddings: Array,
    embedding_norm: Array,
    layers: Vec<EncoderLayer>,
    final_norm: Array,
    rotary_global: RotaryTables,
    rotary_local: RotaryTables,
    /// Sliding-window band masks keyed by sequence length, built on demand.
    window_masks: std::sync::Mutex<std::collections::HashMap<usize, Array>>,
    num_heads: usize,
    head_dim: usize,
    half_window: usize,
    norm_eps: f64,
}

// SAFETY: every array evaluation in this body's methods runs under the
// process-wide `MlxRuntime` execution mutex (see the qwen35 `mlx` module
// docs); outside the lock, array handles are immutable refcounted values
// whose handle-only operations (clone, shape, dtype) are safe concurrently.
// The window-mask cache is additionally guarded by its own mutex. This is
// the same soundness argument as `unsafe impl Sync for MlxQwen35Backbone`,
// and it is what lets the bounded engine share one loaded model across the
// blocking-pool threads.
unsafe impl Send for ModernBertMlx {}
unsafe impl Sync for ModernBertMlx {}

impl ModernBertMlx {
    /// Load the body from a digest-verified shard. Must run inside
    /// [`MlxRuntime::execute`]; every required tensor is checked against the
    /// pinned shape and any missing tensor fails closed. `prefix` is the
    /// checkpoint's ModernBERT namespace (for example `encoder` or
    /// `model.encoder_model`).
    pub(crate) fn load(
        shard: &mut Shard,
        config: &ModernBertMlxConfig,
        prefix: &str,
    ) -> Result<Self, MlxError> {
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let layers = (0..config.num_hidden_layers)
            .map(|index| EncoderLayer::load(shard, config, prefix, index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            tok_embeddings: shard.f32_tensor(
                &format!("{prefix}.embeddings.tok_embeddings.weight"),
                &[config.vocab_size, hidden],
            )?,
            embedding_norm: shard
                .f32_tensor(&format!("{prefix}.embeddings.norm.weight"), &[hidden])?,
            layers,
            final_norm: shard.f32_tensor(&format!("{prefix}.final_norm.weight"), &[hidden])?,
            rotary_global: RotaryTables::new(
                head_dim,
                config.global_rope_theta,
                config.max_sequence_tokens,
            )?,
            rotary_local: RotaryTables::new(
                head_dim,
                config.local_rope_theta,
                config.max_sequence_tokens,
            )?,
            window_masks: std::sync::Mutex::new(std::collections::HashMap::new()),
            num_heads,
            head_dim,
            half_window: config.local_attention / 2,
            norm_eps: config.norm_eps,
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

    /// One full-sequence forward: embedding, all encoder layers, and final
    /// norm, returning the `(seq, hidden)` final-norm hidden states.
    pub(crate) fn forward_hidden(&self, token_ids: &[u32]) -> Result<Array, MlxError> {
        let seq = token_ids.len();
        let ids = Array::from_slice(token_ids, &[seq as i32]);
        let mut xs = self
            .tok_embeddings
            .take_axis(&ids, 0)
            .map_err(op("embed tokens"))?;
        xs = layer_norm_weight_only(&xs, &self.embedding_norm, self.norm_eps)?;
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
        layer_norm_weight_only(&xs, &self.final_norm, self.norm_eps)
    }
}
