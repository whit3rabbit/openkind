//! Laya typed-decision head for the candle CPU path.
//!
//! Port of the reference `DecisionModel` head over the ModernBERT encoder:
//! a type embedding added to every position, two pre-norm
//! `nn.TransformerEncoderLayer`s over the full token sequence, and a scorer
//! applied at the gathered marker positions (`LayerNorm → Linear → GELU →
//! Linear(1)`), yielding one logit per option span. The reference's
//! `act_head` is deliberately not loaded; see the family module docs.

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, Activation, Embedding, LayerNorm, LayerNormConfig, Linear, Module,
    VarBuilder,
};

/// `nn.LayerNorm` semantics: subtract mean, divide by std, affine, eps 1e-5.
const LN_CONFIG: LayerNormConfig = LayerNormConfig {
    eps: 1e-5,
    remove_mean: true,
    affine: true,
};

/// Head hyperparameters fixed by the pinned `rl_agent_config.json`.
pub(crate) struct LayaHeadConfig {
    /// Encoder hidden size (head width).
    pub(crate) hidden_size: usize,
    /// Pre-norm encoder layers over the sequence (pinned: 2).
    pub(crate) head_layers: usize,
}

/// One pre-norm `nn.TransformerEncoderLayer` (ReLU FFN, packed-QKV
/// self-attention with bias, LayerNorm eps 1e-5 with bias).
struct HeadEncoderLayer {
    norm1: LayerNorm,
    attn_in_weight: Tensor,
    attn_in_bias: Tensor,
    attn_out: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl HeadEncoderLayer {
    fn load(hidden: usize, vb: VarBuilder) -> Result<Self> {
        let num_heads = (hidden / 64).max(1);
        Ok(Self {
            norm1: layer_norm(hidden, LN_CONFIG, vb.pp("norm1"))?,
            attn_in_weight: vb.get((3 * hidden, hidden), "self_attn.in_proj_weight")?,
            attn_in_bias: vb.get((3 * hidden,), "self_attn.in_proj_bias")?,
            attn_out: linear(hidden, hidden, vb.pp("self_attn.out_proj"))?,
            norm2: layer_norm(hidden, LN_CONFIG, vb.pp("norm2"))?,
            fc1: linear(hidden, 4 * hidden, vb.pp("linear1"))?,
            fc2: linear(4 * hidden, hidden, vb.pp("linear2"))?,
            num_heads,
            head_dim: hidden / num_heads,
        })
    }

    /// Self-attention over `(batch, seq, hidden)`; identical query, key, and
    /// value use the packed projection as a single matmul.
    fn self_attn(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = xs.dims3()?;
        let qkv = xs
            .broadcast_matmul(&self.attn_in_weight.t()?)?
            .broadcast_add(&self.attn_in_bias)?;
        let shape = (batch, seq_len, 3, self.num_heads, self.head_dim);
        let qkv = qkv.to_dtype(DType::F32)?.reshape(shape)?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn = weights.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.reshape((batch, seq_len, hidden))?;
        self.attn_out.forward(&attn)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attended = self.self_attn(&self.norm1.forward(xs)?)?;
        let xs = xs.broadcast_add(&attended)?;
        let projected = self.fc2.forward(
            &self
                .fc1
                .forward(&self.norm2.forward(&xs)?)?
                .apply(&Activation::Relu)?,
        )?;
        xs.broadcast_add(&projected)
    }
}

/// The typed decision head: type embedding, sequence layers, marker scorer.
pub(crate) struct LayaDecisionHead {
    type_emb: Embedding,
    layers: Vec<HeadEncoderLayer>,
    scorer_norm: LayerNorm,
    scorer_fc1: Linear,
    scorer_fc2: Linear,
    hidden_size: usize,
}

impl LayaDecisionHead {
    pub(crate) fn load(cfg: &LayaHeadConfig, vb: VarBuilder) -> Result<Self> {
        let type_emb = Embedding::new(
            vb.get((3, cfg.hidden_size), "type_emb.weight")?,
            cfg.hidden_size,
        );
        let layers = (0..cfg.head_layers)
            .map(|index| {
                HeadEncoderLayer::load(cfg.hidden_size, vb.pp(format!("head.layers.{index}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let scorer_norm = layer_norm(cfg.hidden_size, LN_CONFIG, vb.pp("scorer.0"))?;
        let scorer_fc1 = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("scorer.1"))?;
        let scorer_fc2 = linear(cfg.hidden_size, 1, vb.pp("scorer.3"))?;
        Ok(Self {
            type_emb,
            layers,
            scorer_norm,
            scorer_fc1,
            scorer_fc2,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Add the type-embedding row for `qtype` to every position of the
    /// `(seq, hidden)` encoder output, returning `(1, seq, hidden)` ready
    /// for the sequence layers.
    pub(crate) fn typed(
        &self,
        hidden_states: &Tensor,
        qtype: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let index = Tensor::from_slice(&[qtype as u32], (1,), device)?.unsqueeze(0)?;
        let row = self.type_emb.forward(&index)?.to_dtype(DType::F32)?; // (1, 1, hidden)
        let typed = hidden_states
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .broadcast_add(&row)?;
        Ok(typed)
    }

    /// Run the sequence layers over `(1, seq, hidden)` and score the marker
    /// positions, returning one logit per marker in order.
    pub(crate) fn marker_logits(&self, typed: &Tensor, markers: &[usize]) -> Result<Vec<f64>> {
        let mut xs = typed.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        let positions = Tensor::from_slice(
            &(markers
                .iter()
                .map(|&position| position as u32)
                .collect::<Vec<u32>>()),
            markers.len(),
            typed.device(),
        )?;
        let gathered = xs.squeeze(0)?.index_select(&positions, 0)?; // (k, hidden)
        let _ = self.hidden_size;
        let scored = self.scorer_fc2.forward(
            &self
                .scorer_fc1
                .forward(&self.scorer_norm.forward(&gathered)?)?
                .apply(&Activation::Gelu)?,
        )?;
        let logits = scored.squeeze(D::Minus1)?.to_vec1::<f32>()?;
        Ok(logits.into_iter().map(f64::from).collect())
    }
}
