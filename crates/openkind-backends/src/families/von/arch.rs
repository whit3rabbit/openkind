//! Von option-marker scorer head for the candle CPU path.
//!
//! Port of the reference `OptionMarkerScorer`: `LayerNorm → Linear(H, H/2)
//! → GELU → LayerNorm(H/2) → Linear(H/2, 1)` over each gathered marker
//! hidden state, yielding one scalar logit per option. The reference's
//! dropout is training-only and evaluates to identity.

use candle_core::{Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, Activation, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder,
};

/// `torch.nn.LayerNorm` default semantics: eps 1e-5, affine with bias.
const LN_CONFIG: LayerNormConfig = LayerNormConfig {
    eps: 1e-5,
    remove_mean: true,
    affine: true,
};

/// Head width derived from the encoder hidden size (`hidden / 2`).
pub(crate) struct VonScorerConfig {
    pub(crate) hidden_size: usize,
}

/// The calibrated MLP scorer over marker representations.
pub(crate) struct VonScorer {
    input_norm: LayerNorm,
    dense: Linear,
    norm: LayerNorm,
    out_proj: Linear,
}

impl VonScorer {
    pub(crate) fn load(cfg: &VonScorerConfig, vb: VarBuilder) -> Result<Self> {
        let intermediate = cfg.hidden_size / 2;
        Ok(Self {
            input_norm: layer_norm(cfg.hidden_size, LN_CONFIG, vb.pp("input_norm"))?,
            dense: linear(cfg.hidden_size, intermediate, vb.pp("dense"))?,
            norm: layer_norm(intermediate, LN_CONFIG, vb.pp("norm"))?,
            out_proj: linear(intermediate, 1, vb.pp("out_proj"))?,
        })
    }

    /// Score `(options, hidden)` marker representations into one logit per
    /// option, in marker order.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Vec<f64>> {
        let normed = self.input_norm.forward(x)?;
        let projected = self.dense.forward(&normed)?;
        let activated = self.norm.forward(&projected.apply(&Activation::Gelu)?)?;
        let scored = self.out_proj.forward(&activated)?;
        Ok(scored
            .squeeze(D::Minus1)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    }
}
