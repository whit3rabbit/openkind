//! MLX array implementation of the pinned laya typed head over the shared
//! ModernBERT encoder body.
//!
//! The body mirrors the candle CPU arithmetic in
//! [`crate::families::modernbert`] op for op through the shared
//! [`ModernBertMlx`](crate::families::mlx_modernbert::ModernBertMlx)
//! implementation; this module owns the laya-specific typed head (type
//! embedding, pre-norm bias-carrying head layers, marker scorer) mirrored
//! from `families::laya::arch`. FP16-stored shards upcast to FP32 arrays at
//! load. The candle CPU path remains the correctness oracle.

use std::path::Path;

use mlx_rs::ops;
use mlx_rs::Array;

use crate::families::laya::LayaProfile;
use crate::families::mlx_modernbert::{
    gelu, layer_norm_biased, op, scalar, ModernBertMlx, ModernBertMlxConfig, PinnedDtype, Shard,
};
use crate::qwen35::mlx::MlxError;

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
            .map_err(op("attn q reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("attn q transpose"))?;
        let k = parts[1]
            .reshape(&head_shape)
            .map_err(op("attn k reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("attn k transpose"))?;
        let v = parts[2]
            .reshape(&head_shape)
            .map_err(op("attn v reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(op("attn v transpose"))?;
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
    body: ModernBertMlx,
    head_type_emb: Array,
    head_layers: Vec<HeadLayer>,
    scorer_norm_weight: Array,
    scorer_norm_bias: Array,
    scorer_fc1: Array,
    scorer_fc1_bias: Array,
    scorer_fc2: Array,
    scorer_fc2_bias: Array,
    head_num_heads: usize,
    head_head_dim: usize,
}

// SAFETY: every array evaluation in this model's methods runs under the
// process-wide `MlxRuntime` execution mutex (see the qwen35 `mlx` module
// docs); outside the lock, array handles are immutable refcounted values
// whose handle-only operations (clone, shape, dtype) are safe concurrently.
// The body's window-mask cache is additionally guarded by its own mutex.
// This is the same soundness argument as `unsafe impl Sync for
// MlxQwen35Backbone`, and it is what lets the bounded engine share one
// loaded model across the blocking-pool threads.
unsafe impl Send for LayaMlxModel {}
unsafe impl Sync for LayaMlxModel {}

impl LayaMlxModel {
    /// Load the digest-verified checkpoint into FP32 MLX arrays. Must run
    /// inside [`MlxRuntime::execute`]; every required tensor is checked
    /// against the pinned shape and any missing tensor fails closed.
    pub(super) fn load(profile: &LayaProfile, checkpoint: &Path) -> Result<Self, MlxError> {
        let mut shard = Shard::open(checkpoint, PinnedDtype::F16)?;
        let hidden = profile.encoder.hidden_size;
        let body_config = ModernBertMlxConfig {
            vocab_size: profile.encoder.vocab_size,
            hidden_size: hidden,
            num_attention_heads: profile.encoder.num_attention_heads,
            num_hidden_layers: profile.encoder.num_hidden_layers,
            intermediate_size: profile.encoder.intermediate_size,
            local_attention: 128,
            global_attn_every_n_layers: 3,
            global_rope_theta: profile.encoder.global_rope_theta,
            local_rope_theta: profile.encoder.local_rope_theta,
            norm_eps: 1e-5,
            max_sequence_tokens: profile.max_sequence_tokens,
        };
        let body = ModernBertMlx::load(&mut shard, &body_config, "encoder")?;
        let head_layers = (0..2)
            .map(|index| HeadLayer::load(&mut shard, hidden, &format!("head.layers.{index}")))
            .collect::<Result<Vec<_>, _>>()?;
        let head_num_heads = (hidden / 64).max(1);
        Ok(Self {
            body,
            head_type_emb: shard.f32_tensor("type_emb.weight", &[3, hidden])?,
            head_layers,
            scorer_norm_weight: shard.f32_tensor("scorer.0.weight", &[hidden])?,
            scorer_norm_bias: shard.f32_tensor("scorer.0.bias", &[hidden])?,
            scorer_fc1: shard.transposed_f32_tensor("scorer.1.weight", hidden, hidden)?,
            scorer_fc1_bias: shard.f32_tensor("scorer.1.bias", &[hidden])?,
            scorer_fc2: shard.transposed_f32_tensor("scorer.3.weight", 1, hidden)?,
            scorer_fc2_bias: shard.f32_tensor("scorer.3.bias", &[1])?,
            head_num_heads,
            head_head_dim: hidden / head_num_heads,
        })
    }

    /// One forward pass over the rendered sequence: encoder, type embedding,
    /// head layers, and one logit per marker in marker order.
    pub(super) fn option_logits(
        &self,
        token_ids: &[u32],
        markers: &[usize],
        qtype: usize,
    ) -> Result<Vec<f64>, MlxError> {
        let mut xs = self.body.forward_hidden(token_ids)?;

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
