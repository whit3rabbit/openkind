//! Gemma 4 text decoder — the openkind quantized reference port.
//!
//! Stateless full-sequence forward for decision readouts: no KV cache is
//! retained between questions, so every evaluation is a fresh decode from
//! position 0, numerically identical to the reference runtime's cold-prefix
//! path. Compute weights execute through candle's `QMatMul` kernels over the
//! checkpoint's own q8_0 blocks — the binding the `decoder-logit-llm` family
//! established — and the two giant embedding tables stream their rows from
//! the verified GGUF on demand, so the loaded footprint stays near the
//! on-disk artifact size instead of a dequantized copy.
//!
//! Attention is banded, never dense: queries attend in fixed-size blocks over
//! only the keys the mask would admit (the causal prefix for full-attention
//! layers, a `window + block` band for sliding layers), so no `seq x seq`
//! score or mask tensor is ever materialized and per-forward scratch grows
//! linearly with prompt length. Forwards re-check the request control between
//! layers so cancellation and deadlines release the execution slot during a
//! long pass, and a load-time scratch budget caps the accepted context length
//! before any tensor is allocated.
//!
//! The implementation follows the two agreeing references for Gemma 4
//! (mistral.rs `vision_models/gemma4` and llama.cpp `src/models/gemma4.cpp`):
//!
//! - RMSNorm applies the stored weight directly — no `+1` shift (the
//!   llama.cpp `Gemma4Model` converter pins `norm_shift = 0`, unlike
//!   Gemma 1-3).
//! - Attention logits are not scaled by `1/sqrt(head_dim)`: both references
//!   set the softmax scale to `1.0` (`f_attention_scale = 1.0` /
//!   `softmax_scale: 1.0`); the learned Q/K norms carry the scale.
//! - Values receive a pure RMS normalization without a learned weight.
//! - Sliding layers use full-dimension RoPE with base `1e4`; full-attention
//!   layers use proportional partial RoPE with base `1e6` and a `0.25`
//!   rotary factor.
//! - The E-series per-layer embeddings (PLE) inject a per-layer input after
//!   the feed-forward block: a shared `(vocab, layers x 256)` embedding
//!   table and a model-level projection are combined with fixed scalars,
//!   then each layer gates, multiplies, projects, and post-norms its slice.
//! - Layers configured as KV-shared (the trailing `shared_kv_layers`) reuse
//!   the K/V of their same-type donor layer computed earlier in the same
//!   forward; their own K/V projections, when present in the checkpoint,
//!   are dead weights in the reference runtimes and are not loaded here.

use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Activation;

use super::gguf::Gemma4Checkpoint;
use crate::families::support::{FamilyControl, FamilyError};

/// Query-block size of the banded attention. Every score tensor the forward
/// materializes is bounded by `(heads, CHUNK, CHUNK + sliding_window)` on
/// sliding layers and `(heads, CHUNK, seq)` on full-attention layers, so
/// attention scratch grows linearly with sequence length instead of
/// quadratically and an accepted prompt can never allocate a
/// `heads x seq x seq` tensor.
const ATTENTION_QUERY_CHUNK: usize = 256;

/// Pinned Gemma 4 E4B text geometry.
///
/// Every field is a constant of the checkpoint family this module loads;
/// the values mirror the GGUF metadata the loader enforces, so the config
/// carries no runtime degrees of freedom.
#[derive(Debug, Clone)]
pub struct Gemma4TextConfig {
    /// Hidden size (`gemma4.embedding_length`).
    pub hidden_size: usize,
    /// FFN intermediate size (`gemma4.feed_forward_length`).
    pub intermediate_size: usize,
    /// Vocabulary size (rows of the tied embedding).
    pub vocab_size: usize,
    /// Number of decoder layers (`gemma4.block_count`).
    pub num_hidden_layers: usize,
    /// Query heads (`gemma4.attention.head_count`).
    pub num_attention_heads: usize,
    /// Key/value heads for both layer types (`gemma4.attention.head_count_kv`).
    pub num_key_value_heads: usize,
    /// Head dimension of sliding-window layers (`...key_length_swa`).
    pub head_dim: usize,
    /// Head dimension of full-attention layers (`...key_length`).
    pub global_head_dim: usize,
    /// Sliding-window span (`gemma4.attention.sliding_window`).
    pub sliding_window: usize,
    /// RMSNorm epsilon (`gemma4.attention.layer_norm_rms_epsilon`).
    pub rms_norm_eps: f64,
    /// RoPE base of full-attention layers (`gemma4.rope.freq_base`).
    pub rope_theta: f64,
    /// RoPE base of sliding layers (`gemma4.rope.freq_base_swa`).
    pub rope_local_base_freq: f64,
    /// Partial rotary factor of full-attention layers (Gemma 4 E4B: `0.25`).
    pub partial_rotary_factor: f64,
    /// Final logit softcapping (`gemma4.final_logit_softcapping`).
    pub final_logit_softcapping: Option<f64>,
    /// Per-layer embedding width (`gemma4.embedding_length_per_layer_input`).
    pub per_layer_input_dim: usize,
    /// Trailing layers that reuse donor K/V (`gemma4.attention.shared_kv_layers`).
    pub shared_kv_layers: usize,
    /// Activation of the FFN and the PLE gate (`gelu_pytorch_tanh`).
    pub activation: Activation,
}

impl Gemma4TextConfig {
    /// The pinned Winnow-E4B geometry.
    pub fn winnow_e4b() -> Self {
        Self {
            hidden_size: 2_560,
            intermediate_size: 10_240,
            vocab_size: 262_144,
            num_hidden_layers: 42,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            global_head_dim: 512,
            sliding_window: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.25,
            final_logit_softcapping: Some(30.0),
            per_layer_input_dim: 256,
            shared_kv_layers: 18,
            activation: Activation::GeluPytorchTanh,
        }
    }

    /// First layer index that reuses donor K/V.
    pub fn first_shared_layer(&self) -> usize {
        self.num_hidden_layers.saturating_sub(self.shared_kv_layers)
    }

    /// `true` when `layer` is a sliding-window layer. The pinned E4B pattern
    /// is five sliding layers followed by one full-attention layer,
    /// repeating.
    pub fn is_sliding(&self, layer: usize) -> bool {
        debug_assert!(layer < self.num_hidden_layers);
        layer % 6 != 5
    }

    /// Donor layer whose K/V `layer` reuses, for KV-shared layers. The donor
    /// is the last layer of the same attention type before the shared range
    /// — sliding layers donate to sliding layers, full to full, so the head
    /// dimensions line up. This matches `llama_hparams::has_kv`'s
    /// `n_layer_kv_from_start - (is_swa ? 2 : 1)` rule and mistral.rs's
    /// same-type donor search for the pinned geometry.
    pub fn kv_donor(&self, layer: usize) -> Option<usize> {
        let first_shared = self.first_shared_layer();
        if self.shared_kv_layers == 0 || layer < first_shared {
            return None;
        }
        Some(if self.is_sliding(layer) {
            first_shared - 2
        } else {
            first_shared - 1
        })
    }

    /// Conservative over-estimate, in bytes, of the transient scratch one
    /// full-sequence forward allocates for `seq` tokens on top of the mapped
    /// checkpoint: the widest layer's banded attention blocks and expanded
    /// K/V, the FFN intermediates that are live together, the shared
    /// per-layer-input buffers, and the donor K/V retained to the end of the
    /// pass. Every term is linear in `seq` — attention is banded, so a
    /// quadratic term here (or in the execution path) is a regression.
    pub fn estimated_peak_scratch_bytes(&self, seq: usize) -> u64 {
        let bytes = |values: u64| values.saturating_mul(4);
        let seq = seq as u64;
        let heads = self.num_attention_heads as u64;
        let max_head_dim = self.head_dim.max(self.global_head_dim) as u64;
        // Attention is executed one layer at a time; budget the widest layer:
        // expanded Q/K/V, the block outputs, the largest (full-attention)
        // score block plus its softmax copy, and the hidden residual streams
        // crossing the layer.
        let attention = bytes(heads * seq * max_head_dim * 6)
            + bytes(heads * ATTENTION_QUERY_CHUNK as u64 * seq)
            + bytes(3 * seq * self.hidden_size as u64);
        // gate, gelu(gate), up, and their product are live together when the
        // product is formed.
        let ffn = bytes(4 * seq * self.intermediate_size as u64);
        // The shared (seq, layers, ple_dim) per-layer-input buffer plus the
        // contiguous per-layer slice retained for the whole forward.
        let ple = bytes(2 * seq * self.num_hidden_layers as u64 * self.per_layer_input_dim as u64);
        // One sliding and one full-attention donor's K/V, both counted at the
        // larger head dim, retained until the last shared layer.
        let donors = bytes(2 * self.num_key_value_heads as u64 * seq * max_head_dim * 2);
        // Token embeddings and the hidden stream crossing a layer boundary.
        let stream = bytes(2 * seq * self.hidden_size as u64);
        attention.max(ffn) + ple + donors + stream
    }

    /// Largest accepted sequence length whose estimated peak scratch stays
    /// within `budget_bytes`, capped at the frozen family maximum. Returns a
    /// length below the viability floor only when the budget cannot fit even
    /// a minimal prompt, which loaders must treat as fail-closed.
    pub fn max_sequence_tokens_for_scratch_budget(&self, budget_bytes: u64) -> usize {
        let ceiling = crate::families::gemma4::MAX_SEQUENCE_TOKENS;
        let mut low = 0usize;
        let mut high = ceiling;
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if self.estimated_peak_scratch_bytes(mid) <= budget_bytes {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        low
    }
}

/// RMSNorm applying the stored weight directly (Gemma 4 dropped the
/// historical Gemma `weight + 1` shift).
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let internal = match dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let dim = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal)?;
        let mean_sq = (x.sqr()?.sum_keepdim(D::Minus1)? / dim as f64)?;
        let x_normed = x.broadcast_div(&(mean_sq + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// Pure RMS normalization without a learned weight, applied to values.
fn rms_norm_unweighted(v: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = v.dtype();
    let v = v.to_dtype(DType::F32)?;
    let mean_sq = (v.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    v.broadcast_div(&mean_sq)?.to_dtype(dtype)
}

/// Rotary tables shared by every layer of one kind.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, head_dim: usize, base: f64, max_seq: usize, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim / 2)
            .map(|i| 1f64 / base.powf((2 * i) as f64 / head_dim as f64))
            .collect();
        Self::from_inv_freq(dtype, inv_freq, max_seq, dev)
    }

    fn from_inv_freq(
        dtype: DType,
        inv_freq: Vec<f64>,
        max_seq: usize,
        dev: &Device,
    ) -> Result<Self> {
        let len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, len), dev)?.to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, x: &Tensor, seq_offset: usize) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let cos = self.cos.narrow(0, seq_offset, seq)?;
        let sin = self.sin.narrow(0, seq_offset, seq)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}

/// Proportional partial rotary embedding for full-attention layers: only the
/// first `partial_rotary_factor * head_dim / 2` pair frequencies rotate; the
/// remaining pairs carry identity rotation (cos = 1, sin = 0).
#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    inner: RotaryEmbedding,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        base: f64,
        partial_rotary_factor: f64,
        max_seq: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rotated_pairs = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..rotated_pairs {
            inv_freq.push(1f64 / base.powf((2 * i) as f64 / head_dim as f64));
        }
        inv_freq.extend(std::iter::repeat_n(0f64, half - rotated_pairs));
        Ok(Self {
            inner: RotaryEmbedding::from_inv_freq(dtype, inv_freq, max_seq, dev)?,
        })
    }

    fn apply(&self, x: &Tensor, seq_offset: usize) -> Result<Tensor> {
        self.inner.apply(x, seq_offset)
    }
}

/// Gated FFN of one decoder layer over quantized weights.
#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    activation: Activation,
}

impl Mlp {
    fn new(cfg: &Gemma4TextConfig, ckpt: &Gemma4Checkpoint, layer: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: QMatMul::from_arc(
                ckpt.q_tensor(&format!("layers.{layer}.mlp.gate_proj.weight"))?,
            )?,
            up_proj: QMatMul::from_arc(
                ckpt.q_tensor(&format!("layers.{layer}.mlp.up_proj.weight"))?,
            )?,
            down_proj: QMatMul::from_arc(
                ckpt.q_tensor(&format!("layers.{layer}.mlp.down_proj.weight"))?,
            )?,
            activation: cfg.activation,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.activation)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

/// Causal self-attention with the Gemma 4 hybrid geometry. KV-shared layers
/// carry only the query path and attend over their donor's K/V.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: Option<candle_nn::Linear>,
    v_proj: Option<candle_nn::Linear>,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    rotary_local: Arc<RotaryEmbedding>,
    rotary_global: Arc<ProportionalRotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_sliding: bool,
    sliding_window: usize,
    kv_donor: Option<usize>,
    rms_norm_eps: f64,
}

impl Attention {
    fn new(
        cfg: &Gemma4TextConfig,
        layer: usize,
        rotary_local: Arc<RotaryEmbedding>,
        rotary_global: Arc<ProportionalRotaryEmbedding>,
        ckpt: &Gemma4Checkpoint,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding(layer);
        let head_dim = if is_sliding {
            cfg.head_dim
        } else {
            cfg.global_head_dim
        };
        let kv_donor = cfg.kv_donor(layer);
        let q_proj = ckpt.f32_linear(&format!("layers.{layer}.self_attn.q_proj.weight"))?;
        let (k_proj, v_proj, k_norm) = if kv_donor.is_some() {
            (None, None, None)
        } else {
            (
                Some(ckpt.f32_linear(&format!("layers.{layer}.self_attn.k_proj.weight"))?),
                Some(ckpt.f32_linear(&format!("layers.{layer}.self_attn.v_proj.weight"))?),
                Some(RmsNorm::new(
                    ckpt.f32_tensor(&format!("layers.{layer}.self_attn.k_norm.weight"))?,
                    cfg.rms_norm_eps,
                )),
            )
        };
        let o_proj = ckpt.f32_linear(&format!("layers.{layer}.self_attn.o_proj.weight"))?;
        let q_norm = RmsNorm::new(
            ckpt.f32_tensor(&format!("layers.{layer}.self_attn.q_norm.weight"))?,
            cfg.rms_norm_eps,
        );
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_local,
            rotary_global,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            is_sliding,
            sliding_window: cfg.sliding_window,
            kv_donor,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    /// Attend over this layer's own K/V (projected from the normed hidden,
    /// exactly like the query path — HF and llama.cpp both feed the
    /// input-layernorm output to every projection) or the donor's K/V. The
    /// softmax scale is the Gemma 4 constant `1.0`: the learned Q/K norms
    /// carry the magnitude and both reference runtimes pin the scale to one.
    ///
    /// The score tensor is computed per query block over only the keys the
    /// dense mask would admit — never as one `heads x seq x seq` tensor — so
    /// a maximum-length prompt costs linear scratch, not a 2 GiB score plus
    /// dense masks. Returns the attention output and, for layers with their
    /// own projections, the post-rope K/V for later KV-shared consumers.
    fn forward(
        &self,
        xs: &Tensor,
        donor_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b, seq, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let q = if self.is_sliding {
            self.rotary_local.apply(&q, 0)?
        } else {
            self.rotary_global.apply(&q, 0)?
        };
        let window = self.is_sliding.then_some(self.sliding_window);
        let rep = self.num_heads / self.num_kv_heads;
        let (attn, own_kv) = match (&self.k_proj, &self.v_proj, donor_kv) {
            (Some(k_proj), Some(v_proj), _) => {
                let k = k_proj
                    .forward(xs)?
                    .reshape((b, seq, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let v = v_proj
                    .forward(xs)?
                    .reshape((b, seq, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let k = self
                    .k_norm
                    .as_ref()
                    .expect("k norm on non-shared layers")
                    .forward(&k)?;
                let v = rms_norm_unweighted(&v, self.rms_norm_eps)?;
                let k = if self.is_sliding {
                    self.rotary_local.apply(&k, 0)?
                } else {
                    self.rotary_global.apply(&k, 0)?
                };
                let attn =
                    chunked_attention(&q, &repeat_kv(&k, rep)?, &repeat_kv(&v, rep)?, window)?;
                (attn, Some((k, v)))
            }
            (None, None, Some((dk, dv))) => (
                chunked_attention(&q, &repeat_kv(dk, rep)?, &repeat_kv(dv, rep)?, window)?,
                None,
            ),
            _ => candle_core::bail!("attention needs own projections or donor K/V"),
        };
        let out = attn
            .transpose(1, 2)?
            .reshape((b, seq, ()))?
            .apply(&self.o_proj)?;
        Ok((out, own_kv))
    }
}

/// Grouped-query expansion of K/V to query heads. Takes the source by
/// reference so KV-shared layers expand their donor's K/V without cloning
/// the full per-sequence tensors.
fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (b, kv_heads, seq, head_dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, seq, head_dim))?
        .reshape((b, kv_heads * n_rep, seq, head_dim))
}

/// Key span `[key_start, key_end)` that the query block
/// `[query_start, query_end)` can attend to: the trailing causal range,
/// narrowed to the last `window` positions on sliding layers.
fn block_key_span(
    query_start: usize,
    query_end: usize,
    sliding_window: Option<usize>,
) -> (usize, usize) {
    match sliding_window {
        Some(window) => (query_start.saturating_sub(window), query_end),
        None => (0, query_end),
    }
}

/// Compact block-local attention mask, shaped `(1, 1, query_len, key_len)`.
/// Entry is `0` where the query may attend the key and `-inf` where the
/// dense mask would have blocked it (causal `key <= query`, plus
/// `key + window >= query` on sliding layers). Every row keeps at least one
/// `0` (the query's own position always lies inside the span), so the
/// softmax never sees an all-masked row.
fn attention_block_mask(
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let mut data = Vec::with_capacity(query_len * key_len);
    for qi in 0..query_len {
        let query = query_start + qi;
        for ki in 0..key_len {
            let key = key_start + ki;
            let allowed = key <= query && sliding_window.is_none_or(|window| key + window >= query);
            data.push(if allowed { 0f32 } else { f32::NEG_INFINITY });
        }
    }
    Tensor::from_vec(data, (query_len, key_len), dev)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(dtype)
}

/// Banded attention over query blocks: each block of queries multiplies only
/// the keys it can attend to and applies a compact block-local mask before
/// the softmax. Numerically this is the same attention as the dense masked
/// form — masked-out keys contribute `exp(-inf) = 0` to the softmax, so
/// dropping them changes only the matmul blocking, not the math.
fn chunked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let seq = q.dim(2)?;
    let mut blocks = Vec::with_capacity(seq.div_ceil(ATTENTION_QUERY_CHUNK));
    for query_start in (0..seq).step_by(ATTENTION_QUERY_CHUNK) {
        let query_end = (query_start + ATTENTION_QUERY_CHUNK).min(seq);
        let (key_start, key_end) = block_key_span(query_start, query_end, sliding_window);
        let key_len = key_end - key_start;
        let q_block = q.narrow(2, query_start, query_end - query_start)?;
        let k_block = k.narrow(2, key_start, key_len)?;
        let v_block = v.narrow(2, key_start, key_len)?;
        let scores = q_block.matmul(&k_block.transpose(2, 3)?)?;
        let mask = attention_block_mask(
            query_start,
            query_end - query_start,
            key_start,
            key_len,
            sliding_window,
            scores.dtype(),
            scores.device(),
        )?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        blocks.push(weights.matmul(&v_block)?);
    }
    Tensor::cat(&blocks, 2)
}

/// One Gemma 4 decoder layer: attention, FFN, and the per-layer-embedding
/// injection, each wrapped in Gemma's pre/post RMSNorm pairs.
#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    per_layer_input_gate: Option<QMatMul>,
    per_layer_projection: Option<QMatMul>,
    post_per_layer_input_norm: Option<RmsNorm>,
    layer_output_scale: Option<Tensor>,
    activation: Activation,
}

impl DecoderLayer {
    fn new(
        cfg: &Gemma4TextConfig,
        layer: usize,
        rotary_local: Arc<RotaryEmbedding>,
        rotary_global: Arc<ProportionalRotaryEmbedding>,
        ckpt: &Gemma4Checkpoint,
    ) -> Result<Self> {
        let eps = cfg.rms_norm_eps;
        let ple = cfg.per_layer_input_dim;
        let norm = |name: String| ckpt.f32_tensor(&name).map(|w| RmsNorm::new(w, eps));
        let self_attn = Attention::new(cfg, layer, rotary_local, rotary_global, ckpt)?;
        let mlp = Mlp::new(cfg, ckpt, layer)?;
        let input_layernorm = norm(format!("layers.{layer}.input_layernorm.weight"))?;
        let post_attention_layernorm =
            norm(format!("layers.{layer}.post_attention_layernorm.weight"))?;
        let pre_feedforward_layernorm =
            norm(format!("layers.{layer}.pre_feedforward_layernorm.weight"))?;
        let post_feedforward_layernorm =
            norm(format!("layers.{layer}.post_feedforward_layernorm.weight"))?;
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if ple > 0 {
            (
                Some(QMatMul::from_arc(ckpt.q_tensor(&format!(
                    "layers.{layer}.per_layer_input_gate.weight"
                ))?)?),
                Some(QMatMul::from_arc(ckpt.q_tensor(&format!(
                    "layers.{layer}.per_layer_projection.weight"
                ))?)?),
                Some(norm(format!(
                    "layers.{layer}.post_per_layer_input_norm.weight"
                ))?),
            )
        } else {
            (None, None, None)
        };
        let layer_output_scale = ckpt
            .f32_tensor(&format!("layers.{layer}.layer_output_scale.weight"))
            .ok();
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_output_scale,
            activation: cfg.activation,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        donor_kv: Option<&(Tensor, Tensor)>,
        per_layer_input: Option<&Tensor>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let (xs, own_kv) = self.self_attn.forward(&normed, donor_kv)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (&xs + residual)?;
        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        let mut xs = (xs + residual)?;
        if let (Some(gate), Some(projection), Some(norm), Some(pli)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let gated = gate.forward(&xs)?.apply(&self.activation)?;
            let gated = (gated * pli)?;
            let projected = projection.forward(&gated)?;
            let projected = norm.forward(&projected)?;
            xs = (projected + xs)?;
        }
        let out = match &self.layer_output_scale {
            Some(scale) => xs.broadcast_mul(scale)?,
            None => xs,
        };
        Ok((out, own_kv))
    }
}

/// Gemma 4 text model executing the pinned q8_0 checkpoint.
#[derive(Clone)]
pub struct Gemma4TextModel {
    config: Gemma4TextConfig,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    embed_rows: Arc<super::gguf::GgufRowTable>,
    ple_rows: Arc<super::gguf::GgufRowTable>,
    per_layer_model_projection: Option<QMatMul>,
    per_layer_projection_norm: Option<RmsNorm>,
    lm_head: QMatMul,
    final_logit_softcapping: Option<f64>,
    hidden_size: usize,
    per_layer_input_dim: usize,
    /// Whether any later KV-shared layer consumes this layer's K/V. Layers
    /// whose K/V has no consumer drop it at the end of their forward instead
    /// of retaining every layer's K/V to the end of the pass.
    retains_donor_kv: Vec<bool>,
    /// Cap on the estimated per-forward scratch; `forward_with_check` fails
    /// closed before allocating anything when a sequence's estimate exceeds
    /// it. `None` (tests, bring-up) skips the estimate gate.
    scratch_budget: Option<u64>,
}

impl Gemma4TextModel {
    /// Build the model from the staged checkpoint on its loading device.
    pub fn new(cfg: &Gemma4TextConfig, ckpt: &Gemma4Checkpoint) -> Result<Self> {
        let max_seq = crate::families::gemma4::MAX_SEQUENCE_TOKENS;
        let dev = &ckpt.token_embd_rows.device;
        let dtype = DType::F32;
        let rotary_local = Arc::new(RotaryEmbedding::new(
            dtype,
            cfg.head_dim,
            cfg.rope_local_base_freq,
            max_seq,
            dev,
        )?);
        let rotary_global = Arc::new(ProportionalRotaryEmbedding::new(
            dtype,
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor,
            max_seq,
            dev,
        )?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                layer,
                rotary_local.clone(),
                rotary_global.clone(),
                ckpt,
            )?);
        }
        let norm = RmsNorm::new(ckpt.f32_tensor("norm.weight")?, cfg.rms_norm_eps);
        // Fail-closed geometry checks: the checkpoint must match the pinned
        // config before any weight executes.
        let embed_shape = ckpt
            .q_tensor("embed_tokens.weight")?
            .shape()
            .dims()
            .to_vec();
        if embed_shape != [cfg.vocab_size, cfg.hidden_size] {
            candle_core::bail!(
                "token embedding shape {embed_shape:?} does not match the pinned \
                 (vocab, hidden) = ({}, {})",
                cfg.vocab_size,
                cfg.hidden_size
            );
        }
        let mlp_shape = ckpt
            .q_tensor("layers.0.mlp.gate_proj.weight")?
            .shape()
            .dims()
            .to_vec();
        if mlp_shape != [cfg.intermediate_size, cfg.hidden_size] {
            candle_core::bail!(
                "FFN gate shape {mlp_shape:?} does not match the pinned \
                 (intermediate, hidden) = ({}, {})",
                cfg.intermediate_size,
                cfg.hidden_size
            );
        }
        let ple = cfg.per_layer_input_dim;
        let (per_layer_model_projection, per_layer_projection_norm) = if ple > 0 {
            (
                Some(QMatMul::from_arc(
                    ckpt.q_tensor("per_layer_model_projection.weight")?,
                )?),
                Some(RmsNorm::new(
                    ckpt.f32_tensor("per_layer_projection_norm.weight")?,
                    cfg.rms_norm_eps,
                )),
            )
        } else {
            (None, None)
        };
        let retains_donor_kv = (0..cfg.num_hidden_layers)
            .map(|layer| {
                (layer + 1..cfg.num_hidden_layers)
                    .any(|consumer| cfg.kv_donor(consumer) == Some(layer))
            })
            .collect();
        Ok(Self {
            config: cfg.clone(),
            layers,
            norm,
            embed_rows: Arc::new(ckpt.token_embd_rows.clone()),
            ple_rows: Arc::new(ckpt.ple_rows.clone()),
            per_layer_model_projection,
            per_layer_projection_norm,
            lm_head: ckpt.lm_head.clone(),
            final_logit_softcapping: cfg.final_logit_softcapping,
            hidden_size: cfg.hidden_size,
            per_layer_input_dim: ple,
            retains_donor_kv,
            scratch_budget: None,
        })
    }

    /// Input embeddings: token rows scaled by `sqrt(hidden_size)`, shaped
    /// `(1, seq, hidden)`.
    fn embed_tokens(&self, input_ids: &[u32]) -> Result<Tensor> {
        let rows = self.embed_rows.rows(input_ids)?;
        rows.reshape((1, input_ids.len(), ()))?
            .affine((self.hidden_size as f64).sqrt(), 0f64)
    }

    /// Compute the shared per-layer inputs:
    /// `(rms_norm(per_layer_model_projection(xs) / sqrt(hidden)) +
    /// ple_embedding(ids) * sqrt(ple_dim)) / sqrt(2)`, split per layer.
    fn compute_per_layer_inputs(
        &self,
        input_ids: &[u32],
        xs: &Tensor,
    ) -> Result<Option<Vec<Tensor>>> {
        if self.per_layer_input_dim == 0 {
            return Ok(None);
        }
        let ple_proj = self
            .per_layer_model_projection
            .as_ref()
            .expect("PLE projection on PLE models");
        let ple_norm = self
            .per_layer_projection_norm
            .as_ref()
            .expect("PLE norm on PLE models");
        let seq = input_ids.len();
        let layer_count = self.layers.len();
        let ple_dim = self.per_layer_input_dim;

        let embedded = self
            .ple_rows
            .rows(input_ids)?
            .reshape((1, seq, layer_count, ple_dim))?
            .affine((ple_dim as f64).sqrt(), 0f64)?;
        let projected = ple_proj
            .forward(xs)?
            .affine((self.hidden_size as f64).powf(-0.5), 0f64)?
            .reshape((1, seq, layer_count, ple_dim))?;
        let projected = ple_norm.forward(&projected)?;
        let combined = (projected + embedded)?.affine(std::f64::consts::FRAC_1_SQRT_2, 0f64)?;
        // (1, seq, layers, ple_dim) -> one (1, seq, ple_dim) slice per layer.
        let mut per_layer = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            per_layer.push(combined.narrow(2, layer, 1)?.squeeze(2)?.contiguous()?);
        }
        Ok(Some(per_layer))
    }

    /// Cap the per-forward scratch estimate enforced by
    /// [`Self::forward_with_check`]. Engines set this at load time from the
    /// configured budget.
    pub fn set_scratch_budget(&mut self, budget_bytes: u64) {
        self.scratch_budget = Some(budget_bytes);
    }

    /// Full-sequence forward returning the final-position logits (after
    /// softcapping when configured) as `(1, vocab)`. Uncancelled bring-up
    /// entry point (chat probes); production paths use
    /// [`Self::forward_with_check`].
    #[allow(dead_code)]
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor> {
        self.forward_with_check(input_ids, || Ok::<(), candle_core::Error>(()))
    }

    /// [`Self::forward`] with a cooperative check invoked before the pass,
    /// between every decoder layer, and before the LM head, so cancellation
    /// and deadlines release the execution slot during a long forward instead
    /// of only between questions. The check's error type must also absorb the
    /// backbone's candle errors.
    pub fn forward_with_check<E>(
        &self,
        input_ids: &[u32],
        mut check: impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<Tensor, E>
    where
        E: From<candle_core::Error>,
    {
        check()?;
        let seq = input_ids.len();
        if seq == 0 || seq > crate::families::gemma4::MAX_SEQUENCE_TOKENS {
            return Err(candle_core::Error::Msg(format!(
                "sequence length {seq} outside (0, {}]",
                crate::families::gemma4::MAX_SEQUENCE_TOKENS
            ))
            .into());
        }
        if let Some(budget) = self.scratch_budget {
            let estimate = self.config.estimated_peak_scratch_bytes(seq);
            if estimate > budget {
                return Err(candle_core::Error::Msg(format!(
                    "estimated peak scratch of a {seq}-token forward is {estimate} bytes, \
                     over the configured {budget}-byte budget; refusing to allocate \
                     (lower the prompt or raise OPENKIND_GEMMA4_SCRATCH_BUDGET_MB)"
                ))
                .into());
            }
        }
        let xs = self.embed_tokens(input_ids).map_err(E::from)?;
        let per_layer_inputs = self
            .compute_per_layer_inputs(input_ids, &xs)
            .map_err(E::from)?;

        // Post-rope K/V of the layers that donate to KV-shared layers. Only
        // layers a later consumer actually reads retain their K/V; the pinned
        // geometry's donors sit before every consumer, so the entries are
        // filled in by the time a shared layer reads them.
        let mut donors: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let mut xs = xs;
        let no_scale = std::env::var_os("OPENKIND_GEMMA4_NO_SCALE").is_some();
        let debug = std::env::var_os("OPENKIND_GEMMA4_DEBUG").is_some();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            check()?;
            let donor_kv = layer.self_attn.kv_donor.map(|donor| {
                donors[donor]
                    .clone()
                    .expect("donor K/V computed before its consumer")
            });
            let pli = per_layer_inputs.as_ref().map(|inputs| &inputs[layer_idx]);
            let (out, own_kv) = layer
                .forward(&xs, donor_kv.as_ref(), pli)
                .map_err(E::from)?;
            if own_kv.is_some() && self.retains_donor_kv[layer_idx] {
                donors[layer_idx] = own_kv;
            }
            xs = out;
            if no_scale {
                xs = xs.affine(1.0, 0.0).map_err(E::from)?;
            }
            if debug {
                let flat = xs
                    .flatten_all()
                    .map_err(E::from)?
                    .to_vec1::<f32>()
                    .map_err(E::from)?;
                let max_abs = flat.iter().fold(0f32, |acc, v| acc.max(v.abs()));
                let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
                eprintln!("layer {layer_idx}: max_abs={max_abs:.4} non_finite={non_finite}");
            }
        }
        check()?;
        let last = xs
            .narrow(1, seq - 1, 1)
            .map_err(E::from)?
            .squeeze(1)
            .map_err(E::from)?;
        let logits = self
            .lm_head
            .forward(&self.norm.forward(&last).map_err(E::from)?)
            .map_err(E::from)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(logits
                .affine(1f64 / sc, 0f64)
                .map_err(E::from)?
                .tanh()
                .map_err(E::from)?
                .affine(sc, 0f64)
                .map_err(E::from)?),
        }
    }

    /// Forward the full prompt once and return the final-position logits
    /// restricted to `letter_ids` (after softcapping). The answer slot is
    /// the last prompt position; nothing is sampled and no continuation
    /// tokens exist.
    #[allow(dead_code)]
    pub fn letter_logits(&self, prompt_ids: &[u32], letter_ids: &[u32]) -> Result<Vec<f64>> {
        if prompt_ids.is_empty() {
            candle_core::bail!("prompt token ids are empty");
        }
        if letter_ids.is_empty() {
            candle_core::bail!("letter token ids are empty");
        }
        let logits = self.forward(prompt_ids)?.squeeze(0)?.to_vec1::<f32>()?;
        select_letter_logits(&logits, letter_ids)
    }

    /// [`Self::letter_logits`] running the forward under the request
    /// control: the pass aborts with the control's error when the caller
    /// disconnects or the queue-inclusive deadline elapses between layers.
    pub fn letter_logits_controlled(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> std::result::Result<Vec<f64>, FamilyError> {
        if prompt_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        if letter_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "letter token ids are empty".to_owned(),
            ));
        }
        let logits = self
            .forward_with_check(prompt_ids, || control.check())?
            .squeeze(0)
            .map_err(FamilyError::from)?
            .to_vec1::<f32>()
            .map_err(FamilyError::from)?;
        select_letter_logits(&logits, letter_ids).map_err(FamilyError::from)
    }
}

/// Restrict a final-position logit row to the letter tokens, in candidate
/// order.
fn select_letter_logits(logits: &[f32], letter_ids: &[u32]) -> Result<Vec<f64>> {
    letter_ids
        .iter()
        .map(|token| {
            usize::try_from(*token)
                .ok()
                .and_then(|index| logits.get(index).copied())
                .map(f64::from)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "letter token id {token} outside the model vocabulary"
                    ))
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4b_layer_pattern_matches_the_pinned_geometry() {
        let cfg = Gemma4TextConfig::winnow_e4b();
        let full_layers: Vec<usize> = (0..42).filter(|i| !cfg.is_sliding(*i)).collect();
        assert_eq!(full_layers, [5, 11, 17, 23, 29, 35, 41]);
        assert_eq!(cfg.first_shared_layer(), 24);
    }

    #[test]
    fn kv_donors_follow_the_reference_rule() {
        let cfg = Gemma4TextConfig::winnow_e4b();
        // Sliding consumers read the last sliding donor (22); full consumers
        // read the last full donor (23) — llama.cpp's `from_start - (swa ?
        // 2 : 1)` and mistral.rs's same-type search agree on this geometry.
        assert_eq!(cfg.kv_donor(0), None);
        assert_eq!(cfg.kv_donor(23), None);
        assert_eq!(cfg.kv_donor(24), Some(22));
        assert_eq!(cfg.kv_donor(28), Some(22));
        assert_eq!(cfg.kv_donor(29), Some(23));
        assert_eq!(cfg.kv_donor(41), Some(23));
    }

    #[test]
    fn sliding_mask_blocks_keys_beyond_the_window() {
        let window = 4usize;
        let seq = 8usize;
        // Mirrors the mask predicate under test: `i < j || j + window < i`.
        let allowed = |i: usize, j: usize| !(i < j || (j + window) < i);
        // A query always sees itself and the previous `window` keys.
        for i in 0..seq {
            assert!(allowed(i, i));
            assert!(allowed(i, i.saturating_sub(window)));
            if i > window {
                assert!(!allowed(i, i - window - 1));
            }
        }
    }

    /// Dense masked attention — the pre-banding implementation's math — as
    /// the offline reference for [`chunked_attention`].
    fn dense_attention_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let seq = q.dim(2)?;
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let mut dense = Vec::with_capacity(seq * seq);
        for i in 0..seq {
            for j in 0..seq {
                let allowed = j <= i && sliding_window.is_none_or(|window| j + window >= i);
                dense.push(if allowed { 0f32 } else { f32::NEG_INFINITY });
            }
        }
        let mask = Tensor::from_vec(dense, (seq, seq), q.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        weights.matmul(v)
    }

    #[test]
    fn chunked_attention_matches_dense_masked_attention() {
        let dev = Device::Cpu;
        // Windows on both sides of the query chunk plus the full-attention
        // case, with a sequence long enough to span several query blocks and
        // a short one that fits inside a single (partial) block.
        for (seq, window) in [
            (600, Some(64)),
            (600, Some(512)),
            (600, None),
            (37, Some(4)),
        ] {
            let shape = (1usize, 2usize, seq, 8usize);
            let q = Tensor::randn(0f32, 1.0, shape, &dev).expect("q");
            let k = Tensor::randn(0f32, 1.0, shape, &dev).expect("k");
            let v = Tensor::randn(0f32, 1.0, shape, &dev).expect("v");
            let dense = dense_attention_reference(&q, &k, &v, window).expect("dense");
            let banded = chunked_attention(&q, &k, &v, window).expect("banded");
            let a = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = banded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(a.len(), b.len());
            let max_diff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff < 1e-5,
                "seq={seq} window={window:?}: banded attention diverged from the dense reference by {max_diff:e}"
            );
        }
    }

    #[test]
    fn block_key_span_bounds_the_attention_band() {
        for start in (0..600).step_by(64) {
            let end = (start + ATTENTION_QUERY_CHUNK).min(600);
            let (key_start, key_end) = block_key_span(start, end, Some(64));
            assert_eq!(key_start, start.saturating_sub(64));
            assert!(key_end - key_start <= ATTENTION_QUERY_CHUNK + 64);
            let (key_start, key_end) = block_key_span(start, end, None);
            assert_eq!((key_start, key_end), (0, end));
        }
    }

    #[test]
    fn block_masks_match_the_dense_predicate_and_keep_a_diagonal() {
        // A middle block of a sliding layer: query range [256, 512), key
        // range [192, 512) — causal corner plus windowed head, the general
        // case. The first and final blocks are covered by the equivalence
        // test above.
        let mask = attention_block_mask(256, 256, 192, 320, Some(64), DType::F32, &Device::Cpu)
            .expect("mask");
        let rows = mask
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(rows.len(), 256);
        assert_eq!(rows[0].len(), 320);
        for (qi, row) in rows.iter().enumerate() {
            let query = 256 + qi;
            for (ki, &value) in row.iter().enumerate() {
                let key = 192 + ki;
                let allowed = key <= query && key + 64 >= query;
                assert_eq!(value == 0f32, allowed, "query {query} key {key}");
                assert_eq!(value == f32::NEG_INFINITY, !allowed);
            }
            assert!(
                row.contains(&0f32),
                "a mask row must always keep at least the query's own position"
            );
        }
    }

    #[test]
    fn peak_scratch_estimate_is_linear_and_derives_the_context_cap() {
        let cfg = Gemma4TextConfig::winnow_e4b();
        let half = cfg.estimated_peak_scratch_bytes(4_096);
        let full = cfg.estimated_peak_scratch_bytes(8_192);
        assert_eq!(
            full,
            half * 2,
            "the scratch estimate must stay linear; a quadratic term is a dense-attention regression"
        );
        assert!(full > 0 && full < 4 << 30);
        assert!(cfg.estimated_peak_scratch_bytes(64) < half);
        assert_eq!(
            cfg.max_sequence_tokens_for_scratch_budget(u64::MAX),
            crate::families::gemma4::MAX_SEQUENCE_TOKENS
        );
        assert_eq!(cfg.max_sequence_tokens_for_scratch_budget(0), 0);
        let capped = cfg.max_sequence_tokens_for_scratch_budget(full - 1);
        assert!(capped < crate::families::gemma4::MAX_SEQUENCE_TOKENS);
        assert_eq!(
            cfg.max_sequence_tokens_for_scratch_budget(full),
            crate::families::gemma4::MAX_SEQUENCE_TOKENS
        );
    }

    #[test]
    fn only_consumed_donors_retain_their_kv() {
        let cfg = Gemma4TextConfig::winnow_e4b();
        let retains: Vec<bool> = (0..cfg.num_hidden_layers)
            .map(|layer| {
                (layer + 1..cfg.num_hidden_layers)
                    .any(|consumer| cfg.kv_donor(consumer) == Some(layer))
            })
            .collect();
        // Exactly the two donor layers of the pinned geometry retain K/V;
        // the other own-projection layers drop theirs at the end of the
        // forward instead of holding every layer's K/V to the last layer.
        let donors: Vec<usize> = retains
            .iter()
            .enumerate()
            .filter_map(|(layer, &keeps)| keeps.then_some(layer))
            .collect();
        assert_eq!(donors, [22, 23]);
    }
}
