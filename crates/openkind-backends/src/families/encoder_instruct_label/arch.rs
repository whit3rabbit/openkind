//! Minimal ModernBERT architecture for the candle CPU path.
//!
//! Hand-implemented for the pinned GLiClass uni-encoder checkpoint; candle
//! 0.8.0 ships no ModernBERT. The forward implements exactly the semantics
//! verified against the HuggingFace reference on short and
//! beyond-window inputs:
//!
//! - word embeddings plus a weight-only LayerNorm; positions come from RoPE
//!   only (the checkpoint stores no absolute position tensor);
//! - pre-norm residual layers where layer 0 has no `attn_norm` (the
//!   embedding norm feeds attention directly);
//! - fused bias-free `Wqkv`, NeoX-style rotate-half RoPE applied to queries
//!   and keys, fp32 softmax;
//! - global attention on every layer whose index divides
//!   `global_attn_every_n_layers` (0, 3, 6, …), a symmetric sliding-window
//!   band of half-width `local_attention / 2` on the others;
//! - gated GELU MLP (`Wi` produces 2×`intermediate_size`, chunked into
//!   input and gate), and a weight-only final norm.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear_no_bias, Activation, LayerNorm, Linear, Module, VarBuilder};

/// Pinned ModernBERT configuration (subset used by the forward path).
#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    /// Local (sliding) attention window; the band half-width is half of it.
    pub local_attention: usize,
    /// Every layer whose index divides this runs full attention.
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    pub norm_eps: f64,
    /// Maximum sequence length the rope tables are built for.
    pub max_sequence_tokens: usize,
}

/// RoPE tables for one layer type: `cos`/`sin` shaped `(seq, head_dim)` with
/// the frequency halves duplicated, matching the NeoX rotate-half layout.
struct RotaryTables {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryTables {
    fn new(
        dtype: DType,
        head_dim: usize,
        theta: f64,
        max_seq: usize,
        dev: &Device,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| (1f64 / theta.powf(2.0 * k as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let positions = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        let doubled = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            cos: doubled.cos()?.to_dtype(dtype)?,
            sin: doubled.sin()?.to_dtype(dtype)?,
        })
    }

    /// Apply NeoX rotate-half RoPE to `(1, heads, seq, head_dim)` queries or
    /// keys, broadcasting over heads.
    fn apply(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&(x2.neg()?), &x1], D::Minus1)?;
        let cos = self.cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(1)?;
        let sin = self.sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(1)?;
        let scaled = x.broadcast_mul(&cos)?;
        let rotated_scaled = rotated.broadcast_mul(&sin)?;
        scaled.broadcast_add(&rotated_scaled)
    }
}

/// Weight-only LayerNorm from a VarBuilder that carries no bias tensor.
fn layer_norm_no_bias(hidden: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get((hidden,), "weight")?;
    Ok(LayerNorm::new_no_bias(weight, eps))
}

/// Symmetric sliding-window attention bias: zero inside the band
/// `|i - j| <= half_window`, negative infinity outside.
fn sliding_window_bias(seq_len: usize, half_window: usize, dev: &Device) -> Result<Tensor> {
    let mut bias = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            let distance = i.abs_diff(j);
            bias.push(if distance <= half_window {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::from_slice(&bias, (seq_len, seq_len), dev)
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    rotary: RotaryTables,
    num_heads: usize,
    head_dim: usize,
    half_window: Option<usize>,
}

impl Attention {
    fn load(cfg: &ModernBertConfig, is_global: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let wqkv = linear_no_bias(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("Wqkv"))?;
        let wo = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("Wo"))?;
        let theta = if is_global {
            cfg.global_rope_theta
        } else {
            cfg.local_rope_theta
        };
        Ok(Self {
            wqkv,
            wo,
            rotary: RotaryTables::new(
                DType::F32,
                head_dim,
                theta,
                cfg.max_sequence_tokens,
                vb.device(),
            )?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            half_window: if is_global {
                None
            } else {
                Some(cfg.local_attention / 2)
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = xs.dims3()?;
        let qkv = self.wqkv.forward(xs)?;
        // (batch, seq, 3, heads, head_dim); the fused output rows split as
        // [q | k | v] along the second-to-last axis.
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = self.rotary.apply(&q.to_dtype(DType::F32)?, seq_len)?;
        let k = self.rotary.apply(&k.to_dtype(DType::F32)?, seq_len)?;
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / scale)?;
        if let Some(half_window) = self.half_window {
            let bias = sliding_window_bias(seq_len, half_window, xs.device())?;
            scores = scores.broadcast_add(&bias)?;
        }
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn = weights.matmul(&v.contiguous()?)?;
        let attn = attn.transpose(1, 2)?.reshape((batch, seq_len, hidden))?;
        self.wo.forward(&attn)
    }
}

struct Mlp {
    wi: Linear,
    wo: Linear,
    intermediate: usize,
}

impl Mlp {
    fn load(cfg: &ModernBertConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wi: linear_no_bias(cfg.hidden_size, cfg.intermediate_size * 2, vb.pp("Wi"))?,
            wo: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("Wo"))?,
            intermediate: cfg.intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let projected = self.wi.forward(xs)?;
        let input = projected.narrow(D::Minus1, 0, self.intermediate)?;
        let gate = projected.narrow(D::Minus1, self.intermediate, self.intermediate)?;
        let gated = input.apply(&Activation::Gelu)?.broadcast_mul(&gate)?;
        self.wo.forward(&gated)
    }
}

struct Layer {
    /// `None` on layer 0: the embedding norm output feeds attention directly.
    attn_norm: Option<LayerNorm>,
    attention: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &ModernBertConfig, index: usize, vb: VarBuilder) -> Result<Self> {
        let is_global = index.is_multiple_of(cfg.global_attn_every_n_layers);
        Ok(Self {
            attn_norm: if index == 0 {
                None
            } else {
                Some(layer_norm_no_bias(
                    cfg.hidden_size,
                    cfg.norm_eps,
                    vb.pp("attn_norm"),
                )?)
            },
            attention: Attention::load(cfg, is_global, vb.pp("attn"))?,
            mlp_norm: layer_norm_no_bias(cfg.hidden_size, cfg.norm_eps, vb.pp("mlp_norm"))?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(norm) => norm.forward(xs)?,
            None => xs.clone(),
        };
        let attended = self.attention.forward(&normed)?;
        let xs = xs.broadcast_add(&attended)?;
        let normed = self.mlp_norm.forward(&xs)?;
        let projected = self.mlp.forward(&normed)?;
        xs.broadcast_add(&projected)
    }
}

/// The pinned ModernBERT encoder body: embeddings, layers, final norm.
pub struct ModernBertModel {
    tok_embeddings: candle_nn::Embedding,
    embedding_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
}

impl ModernBertModel {
    pub fn load(cfg: &ModernBertConfig, vb: VarBuilder) -> Result<Self> {
        let tok_embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let embedding_norm =
            layer_norm_no_bias(cfg.hidden_size, cfg.norm_eps, vb.pp("embeddings.norm"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| Layer::load(cfg, index, vb.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = layer_norm_no_bias(cfg.hidden_size, cfg.norm_eps, vb.pp("final_norm"))?;
        Ok(Self {
            tok_embeddings,
            embedding_norm,
            layers,
            final_norm,
        })
    }

    /// Forward one unpadded sequence and return every final-norm hidden
    /// state as `(seq, hidden_size)`.
    pub fn forward(&self, token_ids: &[u32], device: &Device) -> Result<Tensor> {
        let input = Tensor::new(token_ids, device)?.unsqueeze(0)?;
        let mut xs = self
            .embedding_norm
            .forward(&self.tok_embeddings.forward(&input)?)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.final_norm.forward(&xs)?.squeeze(0)
    }
}

/// Build the configuration from the pinned checkpoint values.
pub fn config_from_pinned(max_sequence_tokens: usize) -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 50_370,
        hidden_size: 768,
        num_attention_heads: 12,
        num_hidden_layers: 22,
        intermediate_size: 1_152,
        local_attention: 128,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
        norm_eps: 1e-5,
        max_sequence_tokens,
    }
}
