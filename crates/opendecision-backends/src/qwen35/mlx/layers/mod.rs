//! Qwen3.5 decoder layers over MLX arrays (Phase 3M reference-ops path).
//!
//! The math is a direct port of the parity-proven Candle CPU oracle in
//! [`crate::qwen35::backbone::layer0`]: offset-RMSNorm (folded at load into
//! `(1 + w)`), per-token Gated DeltaNet recurrence vectorized across the 32
//! value heads, causal depthwise convolution folded into the recurrence
//! loop, grouped-query attention with per-head offset-RMSNorm q/k
//! normalization, partial rotary embedding over the **first** 64 dims with
//! NeoX pairing, sigmoid attention gating, and SiLU-gated MLP.
//!
//! This is the `ReferenceOps` DeltaNet implementation: ordinary array
//! operations, one token at a time. The fused Gated-DeltaNet Metal kernel
//! is the separate 3M.5 path and must compare against this one first.
//!
//! MLX arrays are immutable values with cheap refcounted clones; every
//! "mutation" below produces new arrays, so continuation states are isolated
//! structurally. Std-operator arithmetic (`mul`/`add`/`sub`/`div`) is
//! infallible in mlx-rs; fallible ops map their exception text into
//! [`MlxError::Operation`].

use std::collections::BTreeMap;
use std::ops::{Add, Mul};

use mlx_rs::Array;

use super::{MlxError, MlxPrecision};

mod full_attention;
mod linear_attention;
mod ops;

#[cfg(all(test, feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod differential_tests;

pub(crate) use full_attention::*;
pub(crate) use linear_attention::*;
pub(crate) use ops::*;

pub(crate) const HIDDEN_SIZE: usize = 2_560;
const INTERMEDIATE_SIZE: usize = 9_216;
pub(crate) const KEY_HEADS: usize = 16;
pub(crate) const VALUE_HEADS: usize = 32;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const KEY_SIZE: usize = KEY_HEADS * HEAD_DIM;
pub(crate) const VALUE_SIZE: usize = VALUE_HEADS * HEAD_DIM;
pub(crate) const QKV_SIZE: usize = KEY_SIZE * 2 + VALUE_SIZE;
pub(crate) const CONV_KERNEL: usize = 4;
const RMS_EPSILON: f32 = 1e-6;
const ATTENTION_HEADS: usize = 16;
pub(crate) const KV_HEADS: usize = 4;
pub(crate) const ATTENTION_HEAD_DIM: usize = 256;
const ATTENTION_SIZE: usize = ATTENTION_HEADS * ATTENTION_HEAD_DIM;
const KV_ATTN_SIZE: usize = KV_HEADS * ATTENTION_HEAD_DIM;
const ROTARY_DIM: usize = 64;
const ROPE_THETA: f32 = 10_000_000.0;

/// One linear-attention layer's continuation state.
#[derive(Clone)]
pub struct MlxLinearState {
    /// Causal-conv window `[QKV_SIZE, CONV_KERNEL]`.
    pub(crate) conv: Array,
    /// DeltaNet recurrent state `[VALUE_HEADS, HEAD_DIM, HEAD_DIM]`.
    pub(crate) recurrent: Array,
}

/// One full-attention layer's continuation state.
#[derive(Clone)]
pub struct MlxFullState {
    /// Keys `[positions, KV_HEADS, ATTENTION_HEAD_DIM]`.
    pub(crate) keys: Array,
    /// Values `[positions, KV_HEADS, ATTENTION_HEAD_DIM]`.
    pub(crate) values: Array,
}

/// Per-layer continuation state matching the layer's token mixer.
#[derive(Clone)]
pub(crate) enum MlxLayerState {
    /// DeltaNet conv + recurrent state for linear-attention layers.
    Linear(MlxLinearState),
    /// Attention key/value cache for full-attention layers.
    Full(MlxFullState),
}

enum TokenMixer {
    Linear(LinearAttention),
    Full(FullAttention),
}

/// One Qwen3.5 decoder block holding MLX weight arrays.
pub(crate) struct MlxDecoderLayer {
    input_layernorm: Array,
    mixer: TokenMixer,
    post_attention_layernorm: Array,
    mlp_gate_proj: Array,
    mlp_up_proj: Array,
    mlp_down_proj: Array,
}

impl MlxDecoderLayer {
    /// Build layer `layer_index` from the verified tensor map.
    pub(crate) fn load(
        tensors: &BTreeMap<String, Array>,
        layer_index: usize,
        precision: MlxPrecision,
    ) -> Result<Self, MlxError> {
        let prefix = format!("model.language_model.layers.{layer_index}");
        let matrix = |suffix: &str, shape: [usize; 2]| -> Result<Array, MlxError> {
            tensor_2d(tensors, &format!("{prefix}.{suffix}"), shape)
        };
        let vector = |suffix: &str, width: usize| -> Result<Array, MlxError> {
            tensor_vector(tensors, &format!("{prefix}.{suffix}"), width)
        };
        let folded = |suffix: &str, width: usize| -> Result<Array, MlxError> {
            folded_vector(tensors, &format!("{prefix}.{suffix}"), width, precision)
        };
        let mixer = if layer_index % 4 == 3 {
            TokenMixer::Full(FullAttention {
                q_proj: matrix("self_attn.q_proj.weight", [ATTENTION_SIZE * 2, HIDDEN_SIZE])?,
                k_proj: matrix("self_attn.k_proj.weight", [KV_ATTN_SIZE, HIDDEN_SIZE])?,
                v_proj: matrix("self_attn.v_proj.weight", [KV_ATTN_SIZE, HIDDEN_SIZE])?,
                q_norm: folded("self_attn.q_norm.weight", ATTENTION_HEAD_DIM)?,
                k_norm: folded("self_attn.k_norm.weight", ATTENTION_HEAD_DIM)?,
                o_proj: matrix("self_attn.o_proj.weight", [HIDDEN_SIZE, ATTENTION_SIZE])?,
            })
        } else {
            TokenMixer::Linear(LinearAttention {
                in_proj_qkv: matrix("linear_attn.in_proj_qkv.weight", [QKV_SIZE, HIDDEN_SIZE])?,
                in_proj_z: matrix("linear_attn.in_proj_z.weight", [VALUE_SIZE, HIDDEN_SIZE])?,
                in_proj_b: matrix("linear_attn.in_proj_b.weight", [VALUE_HEADS, HIDDEN_SIZE])?,
                in_proj_a: matrix("linear_attn.in_proj_a.weight", [VALUE_HEADS, HIDDEN_SIZE])?,
                conv1d: conv_weight(tensors, &format!("{prefix}.linear_attn.conv1d.weight"))?,
                dt_bias: vector("linear_attn.dt_bias", VALUE_HEADS)?,
                a_log: vector("linear_attn.A_log", VALUE_HEADS)?,
                delta_norm: vector("linear_attn.norm.weight", HEAD_DIM)?,
                out_proj: matrix("linear_attn.out_proj.weight", [HIDDEN_SIZE, VALUE_SIZE])?,
                precision,
            })
        };
        Ok(Self {
            input_layernorm: folded("input_layernorm.weight", HIDDEN_SIZE)?,
            mixer,
            post_attention_layernorm: folded("post_attention_layernorm.weight", HIDDEN_SIZE)?,
            mlp_gate_proj: matrix("mlp.gate_proj.weight", [INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            mlp_up_proj: matrix("mlp.up_proj.weight", [INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            mlp_down_proj: matrix("mlp.down_proj.weight", [HIDDEN_SIZE, INTERMEDIATE_SIZE])?,
        })
    }

    /// Evaluate every weight array so no unevaluated lazy graph crosses the
    /// loading thread's boundary.
    ///
    /// MLX graphs record the stream of the thread that created them, and a
    /// different thread cannot resolve that stream's encoder. Forwards may
    /// run on worker threads that never loaded the model (the engine's
    /// blocking pool), so load-time transforms such as the conv-weight
    /// reshape must be evaluated here while still on the loading thread.
    ///
    /// # Errors
    /// Returns [`MlxError`] when a weight array cannot be evaluated.
    pub(crate) fn materialize(&self) -> Result<(), MlxError> {
        fn eval(array: &Array, label: &str) -> Result<(), MlxError> {
            array.eval().map_err(|error| MlxError::Operation {
                operation: "weight materialization",
                message: format!("{label}: {error}"),
            })
        }
        eval(&self.input_layernorm, "input_layernorm")?;
        eval(&self.post_attention_layernorm, "post_attention_layernorm")?;
        eval(&self.mlp_gate_proj, "mlp_gate_proj")?;
        eval(&self.mlp_up_proj, "mlp_up_proj")?;
        eval(&self.mlp_down_proj, "mlp_down_proj")?;
        match &self.mixer {
            TokenMixer::Linear(mixer) => {
                eval(&mixer.in_proj_qkv, "in_proj_qkv")?;
                eval(&mixer.in_proj_z, "in_proj_z")?;
                eval(&mixer.in_proj_b, "in_proj_b")?;
                eval(&mixer.in_proj_a, "in_proj_a")?;
                eval(&mixer.conv1d, "conv1d")?;
                eval(&mixer.dt_bias, "dt_bias")?;
                eval(&mixer.a_log, "a_log")?;
                eval(&mixer.delta_norm, "delta_norm")?;
                eval(&mixer.out_proj, "out_proj")?;
            }
            TokenMixer::Full(mixer) => {
                eval(&mixer.q_proj, "q_proj")?;
                eval(&mixer.k_proj, "k_proj")?;
                eval(&mixer.v_proj, "v_proj")?;
                eval(&mixer.q_norm, "q_norm")?;
                eval(&mixer.k_norm, "k_norm")?;
                eval(&mixer.o_proj, "o_proj")?;
            }
        }
        Ok(())
    }

    /// Advance the layer, returning the new hidden state and continuation
    /// state. `hidden` is `[rows, HIDDEN_SIZE]` and is never mutated.
    pub(crate) fn forward(
        &self,
        hidden: &Array,
        rows: usize,
        position_start: usize,
        previous: Option<&MlxLayerState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let normalized = rms_norm(hidden, &self.input_layernorm)?;
        let (attention, state) = match (&self.mixer, previous) {
            (TokenMixer::Linear(mixer), Some(MlxLayerState::Linear(state))) => {
                mixer.forward(&normalized, rows, Some(state))?
            }
            (TokenMixer::Linear(mixer), None) => mixer.forward(&normalized, rows, None)?,
            (TokenMixer::Full(mixer), Some(MlxLayerState::Full(state))) => {
                mixer.forward(&normalized, rows, position_start, Some(state))?
            }
            (TokenMixer::Full(mixer), None) => {
                mixer.forward(&normalized, rows, position_start, None)?
            }
            _ => {
                return Err(MlxError::InvalidState(
                    "cache kind does not match the layer mixer".to_owned(),
                ));
            }
        };
        let hidden = hidden.clone().add(&attention);
        let mlp_input = rms_norm(&hidden, &self.post_attention_layernorm)?;
        let gate = linear(&mlp_input, &self.mlp_gate_proj)?;
        let up = linear(&mlp_input, &self.mlp_up_proj)?;
        let activated = mlx_rs::nn::silu(&gate).map_err(op("mlp silu"))?.mul(&up);
        let mlp = linear(&activated, &self.mlp_down_proj)?;
        let hidden = hidden.add(&mlp);
        Ok((hidden, state))
    }
}
