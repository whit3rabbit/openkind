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
    /// Returns the attention output and, for layers with their own
    /// projections, the post-rope K/V for later KV-shared consumers.
    fn forward(
        &self,
        xs: &Tensor,
        donor_kv: Option<&(Tensor, Tensor)>,
        causal_mask: &Tensor,
        sliding_mask: &Tensor,
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
        let (k, v, own_kv) = match (&self.k_proj, &self.v_proj, donor_kv) {
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
                (k.clone(), v.clone(), Some((k, v)))
            }
            (None, None, Some((dk, dv))) => ((*dk).clone(), (*dv).clone(), None),
            _ => candle_core::bail!("attention needs own projections or donor K/V"),
        };
        let k = repeat_kv(k.contiguous()?, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v.contiguous()?, self.num_heads / self.num_kv_heads)?;
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let mask = if self.is_sliding {
            sliding_mask
        } else {
            causal_mask
        };
        let scores = scores.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn = weights.matmul(&v)?;
        let out = attn
            .transpose(1, 2)?
            .reshape((b, seq, ()))?
            .apply(&self.o_proj)?;
        Ok((out, own_kv))
    }
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    candle_transformers::utils::repeat_kv(xs, n_rep)
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
        causal_mask: &Tensor,
        sliding_mask: &Tensor,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let (xs, own_kv) = self
            .self_attn
            .forward(&normed, donor_kv, causal_mask, sliding_mask)?;
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
    sliding_window: usize,
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
        Ok(Self {
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
            sliding_window: cfg.sliding_window,
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

    /// Attention masks for the full sequence: the plain causal mask and the
    /// sliding-window mask (a query attends keys at most `sliding_window`
    /// positions back). Shapes are `(1, 1, seq, seq)` for broadcasting over
    /// batch and heads.
    fn attention_masks(&self, seq: usize, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
        let causal: Vec<f32> = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let sliding = self.sliding_window;
        let windowed: Vec<f32> = (0..seq)
            .flat_map(|i| {
                (0..seq).map(move |j| {
                    if i < j || j + sliding < i {
                        f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
            })
            .collect();
        let causal = Tensor::from_slice(&causal, (seq, seq), dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(dtype)?;
        let windowed = Tensor::from_slice(&windowed, (seq, seq), dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(dtype)?;
        Ok((causal, windowed))
    }

    /// Full-sequence forward returning the final-position logits (after
    /// softcapping when configured) as `(1, vocab)`.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor> {
        let seq = input_ids.len();
        if seq == 0 || seq > crate::families::gemma4::MAX_SEQUENCE_TOKENS {
            candle_core::bail!(
                "sequence length {seq} outside (0, {}]",
                crate::families::gemma4::MAX_SEQUENCE_TOKENS
            );
        }
        let dev = &self.embed_rows.device;
        let xs = self.embed_tokens(input_ids)?;
        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;
        let (causal_mask, sliding_mask) = self.attention_masks(seq, xs.dtype(), dev)?;

        // Post-rope K/V of the layers that donate to KV-shared layers. The
        // pinned geometry's donors sit before every consumer, so the entries
        // are filled in by the time a shared layer reads them.
        let mut donors: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
        let mut xs = xs;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let donor_kv = match layer.self_attn.kv_donor {
                Some(donor) => Some(
                    donors[donor]
                        .clone()
                        .expect("donor K/V computed before its consumer"),
                ),
                None => None,
            };
            let pli = per_layer_inputs.as_ref().map(|inputs| &inputs[layer_idx]);
            let (out, own_kv) =
                layer.forward(&xs, donor_kv.as_ref(), pli, &causal_mask, &sliding_mask)?;
            if own_kv.is_some() {
                donors[layer_idx] = own_kv;
            }
            xs = out;
            if std::env::var_os("OPENKIND_GEMMA4_NO_SCALE").is_some() {
                xs = xs.affine(1.0, 0.0)?;
            }
            if std::env::var_os("OPENKIND_GEMMA4_DEBUG").is_some() {
                let flat = xs.flatten_all()?.to_vec1::<f32>()?;
                let max_abs = flat.iter().fold(0f32, |acc, v| acc.max(v.abs()));
                let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
                eprintln!("layer {layer_idx}: max_abs={max_abs:.4} non_finite={non_finite}");
            }
        }
        let last = xs.narrow(1, seq - 1, 1)?.squeeze(1)?;
        let logits = self.lm_head.forward(&self.norm.forward(&last)?)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(logits.affine(1f64 / sc, 0f64)?.tanh()?.affine(sc, 0f64)?),
        }
    }

    /// Forward the full prompt once and return the final-position logits
    /// restricted to `letter_ids` (after softcapping). The answer slot is
    /// the last prompt position; nothing is sampled and no continuation
    /// tokens exist.
    pub fn letter_logits(&self, prompt_ids: &[u32], letter_ids: &[u32]) -> Result<Vec<f64>> {
        if prompt_ids.is_empty() {
            candle_core::bail!("prompt token ids are empty");
        }
        if letter_ids.is_empty() {
            candle_core::bail!("letter token ids are empty");
        }
        let logits = self.forward(prompt_ids)?.squeeze(0)?.to_vec1::<f32>()?;
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
}
