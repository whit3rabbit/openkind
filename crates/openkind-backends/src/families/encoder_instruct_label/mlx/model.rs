//! MLX array implementation of the pinned GLiClass uni-encoder.
//!
//! The ModernBERT body mirrors the candle CPU arithmetic in
//! [`crate::families::modernbert`] op for op through the shared
//! [`ModernBertMlx`](crate::families::mlx_modernbert::ModernBertMlx)
//! implementation; this module adds the GLiClass projector pair and
//! dot-product scorer mirrored from `super::model`. FP32-stored shards load
//! directly. The candle CPU path remains the correctness oracle.

use mlx_rs::Array;

use crate::families::encoder_instruct_label::{CLASS_TOKEN_ID, MAX_SEQUENCE_TOKENS};
use crate::families::mlx_modernbert::{
    gelu, op, ModernBertMlx, ModernBertMlxConfig, PinnedDtype, Shard,
};
use crate::qwen35::mlx::MlxError;

/// Build the ModernBERT-base configuration from the pinned checkpoint values,
/// mirroring the CPU loader's contract.
fn body_config() -> ModernBertMlxConfig {
    ModernBertMlxConfig {
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
        max_sequence_tokens: MAX_SEQUENCE_TOKENS,
    }
}

/// Two bias `Linear` layers with a GELU activation between them.
struct Projector {
    linear_1: Array,
    linear_1_bias: Array,
    linear_2: Array,
    linear_2_bias: Array,
}

impl Projector {
    fn load(shard: &mut Shard, prefix: &str, hidden: usize) -> Result<Self, MlxError> {
        Ok(Self {
            linear_1: shard.transposed_f32_tensor(
                &format!("{prefix}.linear_1.weight"),
                hidden,
                hidden,
            )?,
            linear_1_bias: shard.f32_tensor(&format!("{prefix}.linear_1.bias"), &[hidden])?,
            linear_2: shard.transposed_f32_tensor(
                &format!("{prefix}.linear_2.weight"),
                hidden,
                hidden,
            )?,
            linear_2_bias: shard.f32_tensor(&format!("{prefix}.linear_2.bias"), &[hidden])?,
        })
    }

    fn forward(&self, xs: &Array) -> Result<Array, MlxError> {
        let projected = xs
            .matmul(&self.linear_1)
            .map_err(op("projector linear_1"))?
            .add(&self.linear_1_bias)
            .map_err(op("projector linear_1 bias"))?;
        let activated = gelu(&projected)?;
        activated
            .matmul(&self.linear_2)
            .map_err(op("projector linear_2"))?
            .add(&self.linear_2_bias)
            .map_err(op("projector linear_2 bias"))
    }
}

/// The pinned label-marker model in MLX arrays: ModernBERT body plus the
/// GLiClass projector pair and dot-product scorer.
pub(super) struct GliclassMlxModel {
    body: ModernBertMlx,
    text_projector: Projector,
    classes_projector: Projector,
}

// SAFETY: every array evaluation in this model's methods runs under the
// process-wide `MlxRuntime` execution mutex (see the qwen35 `mlx` module
// docs); outside the lock, array handles are immutable refcounted values
// whose handle-only operations (clone, shape, dtype) are safe concurrently.
// This is the same soundness argument as `unsafe impl Sync for
// MlxQwen35Backbone`, and it is what lets the bounded engine share one
// loaded model across the blocking-pool threads.
unsafe impl Send for GliclassMlxModel {}
unsafe impl Sync for GliclassMlxModel {}

impl GliclassMlxModel {
    /// Load the digest-verified checkpoint into FP32 MLX arrays. Must run
    /// inside [`MlxRuntime::execute`]; every required tensor is checked
    /// against the pinned shape and any missing tensor fails closed.
    pub(super) fn load(checkpoint: &std::path::Path) -> Result<Self, MlxError> {
        let mut shard = Shard::open(checkpoint, PinnedDtype::F32)?;
        let body = ModernBertMlx::load(&mut shard, &body_config(), "model.encoder_model")?;
        Ok(Self {
            text_projector: Projector::load(&mut shard, "model.text_projector", 768)?,
            classes_projector: Projector::load(&mut shard, "model.classes_projector", 768)?,
            body,
        })
    }

    /// One forward pass over the rendered marker+state sequence, returning
    /// one raw dot-product logit per marker position in marker order.
    ///
    /// The text representation is the hidden state at position 0 (the `[CLS]`
    /// the tokenizer template prepends); class representations are the hidden
    /// states at the `<<LABEL>>` marker positions. The checkpoint's
    /// `logit_scale` is unused: the pinned config sets `normalize_features:
    /// false`, so the reference never applies it.
    pub(super) fn marker_logits(
        &self,
        token_ids: &[u32],
        marker_positions: &[usize],
    ) -> Result<Vec<f64>, MlxError> {
        let hidden = self.body.forward_hidden(token_ids)?;
        let text_hidden = hidden
            .take_axis(Array::from_slice(&[0_u32], &[1]), 0)
            .map_err(op("text row gather"))?;
        let text_projected = self.text_projector.forward(&text_hidden)?;
        let marker_ids: Vec<u32> = marker_positions
            .iter()
            .map(|&position| position as u32)
            .collect();
        let class_hidden = hidden
            .take_axis(
                Array::from_slice(&marker_ids, &[marker_ids.len() as i32]),
                0,
            )
            .map_err(op("marker gather"))?;
        let classes_projected = self.classes_projector.forward(&class_hidden)?;
        // Dot-product scorer: one logit per marker, summed over the hidden
        // axis in FP32 like the candle `broadcast_mul` + `sum` readout.
        let scores = classes_projected
            .multiply(&text_projected)
            .map_err(op("scorer product"))?;
        let logits = scores
            .sum_axis(-1, false)
            .map_err(op("scorer sum"))?
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

/// Marker positions in the encoded sequence: every `<<LABEL>>` token.
pub(super) fn marker_positions(token_ids: &[u32]) -> Vec<usize> {
    token_ids
        .iter()
        .enumerate()
        .filter(|&(_, &token)| token == CLASS_TOKEN_ID)
        .map(|(position, _)| position)
        .collect()
}
