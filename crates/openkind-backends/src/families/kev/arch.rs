//! Qwen3 dense backbone for the kev row form: batched causal rows with
//! explicit position ids.
//!
//! Reuses the hand-rolled Qwen3 implementation the qwen3guard family
//! validated against a PyTorch reference (per-head q/k RMSNorm before RoPE,
//! interleaved GQA repeat, fp32 softmax). Two differences serve this
//! family's row form: forward takes a `[B, L]` batch of independent causal
//! rows with one row's positions per batch entry (padded rows are right
//! padded with position 0; a real token never attends a pad because pads
//! sit after it under the causal mask), and every final-norm hidden state
//! is returned so the pointer head can read out the decide and option
//! boundary positions.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Activation, Linear, Module, RmsNorm, VarBuilder};

/// Pinned Qwen3 configuration (subset used by the forward path).
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Maximum sequence length the rope tables are built for.
    pub max_sequence_tokens: usize,
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.max_sequence_tokens as u32, dev)?
            .to_dtype(dtype)?
            .reshape((cfg.max_sequence_tokens, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
}

impl Attention {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;
        // Qwen3 applies RMSNorm over the head dimension, per head, before
        // RoPE — hence the per-head dimension passed to RmsNorm.
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary: RotaryEmbedding::new(DType::F32, cfg, vb.device())?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let cos = self.rotary.cos.i((..seq_len, ..))?;
        let sin = self.rotary.sin.i((..seq_len, ..))?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?.to_dtype(DType::F32)?, &cos, &sin)?
            .to_dtype(xs.dtype())?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?.to_dtype(DType::F32)?, &cos, &sin)?
            .to_dtype(xs.dtype())?;
        // Grouped-query attention: repeat each KV head `group` times
        // consecutively (HF `repeat_kv` interleave semantics), so query head
        // `i` pairs with KV head `i / group`.
        let group = self.num_heads / self.num_kv_heads;
        let repeat_kv = |t: &Tensor| -> Result<Tensor> {
            if group == 1 {
                return Ok(t.clone());
            }
            let (b, kv_heads, seq, hd) = t.dims4()?;
            t.unsqueeze(2)?
                .expand((b, kv_heads, group, seq, hd))?
                .contiguous()?
                .reshape((b, kv_heads * group, seq, hd))
        };
        let k = repeat_kv(&k)?;
        let v = repeat_kv(&v)?;
        // Causal mask over the row length. Rows are independent causal rows
        // that are right padded, so a real token never attends a pad (pads
        // sit strictly after every real token of its row).
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / scale)?;
        let scores = if seq_len > 1 {
            let mask: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), xs.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn = weights.matmul(&v.contiguous()?)?;
        let attn = attn.transpose(1, 2)?.reshape((batch, seq_len, ()))?;
        let out = self.o_proj.forward(&attn)?;
        Ok(out)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&Activation::Silu)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

struct Layer {
    input_norm: RmsNorm,
    attention: Attention,
    post_norm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attention: Attention::load(cfg, vb.pp("self_attn"))?,
            post_norm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_norm.forward(xs)?;
        let attn = self.attention.forward(&xs)?;
        let xs = (attn + residual)?;
        let residual = &xs;
        let xs = self.post_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

/// The pinned Qwen3 backbone body: embeddings, layers, final RMSNorm.
pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

impl Qwen3Model {
    pub fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| Layer::load(cfg, vb.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward one batch of independent causal rows.
    ///
    /// `rows` holds one `(token_ids, row_len)` per row: the ids buffer is
    /// the padded `[B, L]` row-major buffer and `row_len` is each row's real
    /// length (`row_len <= L`). Positions are `0..row_len` per row; pads
    /// take position 0 and are never read. Returns the final-norm hidden
    /// states as `[B, L, hidden]`.
    pub fn forward_rows(
        &self,
        rows: &[u32],
        row_lens: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let batch = row_lens.len();
        let seq_len = rows.len() / batch;
        let input = Tensor::from_slice(rows, (batch, seq_len), device)?;
        let mut xs = self.embed_tokens.forward(&input)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }
}

/// Build the configuration from the pinned base checkpoint values.
pub fn config_from_pinned(max_sequence_tokens: usize) -> Qwen3Config {
    Qwen3Config {
        vocab_size: 151_936,
        hidden_size: 1024,
        head_dim: 128,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        num_hidden_layers: 28,
        intermediate_size: 3072,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        max_sequence_tokens,
    }
}
