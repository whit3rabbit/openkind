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

use super::{MlxError, MlxGatedDeltaKernel, MlxPrecision};

mod full_attention;
mod gated_delta_kernel;
mod linear_attention;
mod ops;

#[cfg(all(test, feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod differential_tests;

pub(crate) use full_attention::*;
pub(crate) use linear_attention::*;
pub(crate) use ops::*;

/// Arrays retained only by opt-in per-layer parity diagnostics.
pub(crate) type LayerOperationArrays = Vec<(String, Array)>;

pub(crate) fn trace_operation(
    trace: &mut Option<&mut LayerOperationArrays>,
    name: &str,
    array: &Array,
) {
    if let Some(trace) = trace.as_deref_mut() {
        trace.push((name.to_owned(), array.clone()));
    }
}

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
        gated_delta_kernel: MlxGatedDeltaKernel,
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
                gated_delta_kernel,
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
    pub(crate) fn materialize(&self, layer_index: usize) -> Result<(), MlxError> {
        self.visit_weight_arrays(|label, array| {
            array.eval().map_err(|error| MlxError::Operation {
                operation: "weight materialization",
                message: format!("layer {layer_index} {label}: {error}"),
            })
        })
    }

    pub(crate) fn supports_reference_batch(&self) -> bool {
        match &self.mixer {
            TokenMixer::Linear(mixer) => {
                mixer.gated_delta_kernel == MlxGatedDeltaKernel::ReferenceOps
            }
            TokenMixer::Full(_) => true,
        }
    }

    /// Visit every retained weight array exactly once.
    ///
    /// Keeping materialization and accounting on one inventory prevents a new
    /// layer field from silently escaping load-time evaluation.
    fn visit_weight_arrays(
        &self,
        mut visit: impl FnMut(&'static str, &Array) -> Result<(), MlxError>,
    ) -> Result<(), MlxError> {
        visit("input_layernorm", &self.input_layernorm)?;
        visit("post_attention_layernorm", &self.post_attention_layernorm)?;
        visit("mlp_gate_proj", &self.mlp_gate_proj)?;
        visit("mlp_up_proj", &self.mlp_up_proj)?;
        visit("mlp_down_proj", &self.mlp_down_proj)?;
        match &self.mixer {
            TokenMixer::Linear(mixer) => {
                visit("in_proj_qkv", &mixer.in_proj_qkv)?;
                visit("in_proj_z", &mixer.in_proj_z)?;
                visit("in_proj_b", &mixer.in_proj_b)?;
                visit("in_proj_a", &mixer.in_proj_a)?;
                visit("conv1d", &mixer.conv1d)?;
                visit("dt_bias", &mixer.dt_bias)?;
                visit("a_log", &mixer.a_log)?;
                visit("delta_norm", &mixer.delta_norm)?;
                visit("out_proj", &mixer.out_proj)?;
            }
            TokenMixer::Full(mixer) => {
                visit("q_proj", &mixer.q_proj)?;
                visit("k_proj", &mixer.k_proj)?;
                visit("v_proj", &mixer.v_proj)?;
                visit("q_norm", &mixer.q_norm)?;
                visit("k_norm", &mixer.k_norm)?;
                visit("o_proj", &mixer.o_proj)?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn weight_array_count(&self) -> usize {
        let mut count = 0;
        self.visit_weight_arrays(|_, _| {
            count += 1;
            Ok(())
        })
        .expect("weight visitor cannot fail while counting");
        count
    }

    /// Advance the layer, returning the new hidden state and continuation
    /// state. `hidden` is `[rows, HIDDEN_SIZE]` and is never mutated.
    #[cfg(test)]
    pub(crate) fn forward(
        &self,
        hidden: &Array,
        rows: usize,
        position_start: usize,
        previous: Option<&MlxLayerState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        self.forward_traced(hidden, rows, position_start, previous, None)
    }

    pub(crate) fn forward_traced(
        &self,
        hidden: &Array,
        rows: usize,
        position_start: usize,
        previous: Option<&MlxLayerState>,
        mut operation_trace: Option<&mut LayerOperationArrays>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        trace_operation(&mut operation_trace, "input", hidden);
        let normalized = rms_norm(hidden, &self.input_layernorm)?;
        trace_operation(&mut operation_trace, "input_rms_norm", &normalized);
        let (attention, state) = match (&self.mixer, previous) {
            (TokenMixer::Linear(mixer), Some(MlxLayerState::Linear(state))) => mixer
                .forward_traced(
                    &normalized,
                    rows,
                    Some(state),
                    operation_trace.as_deref_mut(),
                )?,
            (TokenMixer::Linear(mixer), None) => {
                mixer.forward_traced(&normalized, rows, None, operation_trace.as_deref_mut())?
            }
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
        trace_operation(&mut operation_trace, "mixer_output", &attention);
        let hidden = hidden.clone().add(&attention);
        trace_operation(&mut operation_trace, "attention_residual", &hidden);
        let mlp_input = rms_norm(&hidden, &self.post_attention_layernorm)?;
        trace_operation(&mut operation_trace, "post_attention_rms_norm", &mlp_input);
        let gate = linear(&mlp_input, &self.mlp_gate_proj)?;
        trace_operation(&mut operation_trace, "mlp_gate_projection", &gate);
        let up = linear(&mlp_input, &self.mlp_up_proj)?;
        trace_operation(&mut operation_trace, "mlp_up_projection", &up);
        let activated = mlx_rs::nn::silu(&gate).map_err(op("mlp silu"))?.mul(&up);
        trace_operation(&mut operation_trace, "mlp_silu_times_up", &activated);
        let mlp = linear(&activated, &self.mlp_down_proj)?;
        trace_operation(&mut operation_trace, "mlp_down_projection", &mlp);
        let hidden = hidden.add(&mlp);
        trace_operation(&mut operation_trace, "output", &hidden);
        Ok((hidden, state))
    }

    /// Advance one layer with a lane axis and right-padded suffix rows.
    pub(crate) fn forward_batched(
        &self,
        hidden: &Array,
        rows: usize,
        lengths: &[usize],
        position_start: usize,
        previous: &MlxLayerState,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let lanes = lengths.len();
        if hidden.shape() != [lanes as i32, rows as i32, HIDDEN_SIZE as i32] {
            return Err(MlxError::InvalidState(format!(
                "batched decoder layer expects [{lanes}, {rows}, {HIDDEN_SIZE}], found {:?}",
                hidden.shape()
            )));
        }
        let normalized = rms_norm(hidden, &self.input_layernorm)?;
        let (attention, state) = match (&self.mixer, previous) {
            (TokenMixer::Linear(mixer), MlxLayerState::Linear(state)) => {
                mixer.forward_batched(&normalized, rows, lengths, state)?
            }
            (TokenMixer::Full(mixer), MlxLayerState::Full(state)) => {
                mixer.forward_batched(&normalized, rows, position_start, lengths, state)?
            }
            _ => {
                return Err(MlxError::InvalidState(
                    "batched cache kind does not match the layer mixer".to_owned(),
                ));
            }
        };
        let hidden = hidden.clone().add(&attention);
        let mlp_input = rms_norm(&hidden, &self.post_attention_layernorm)?;
        let gate = linear(&mlp_input, &self.mlp_gate_proj)?;
        let up = linear(&mlp_input, &self.mlp_up_proj)?;
        let activated = mlx_rs::nn::silu(&gate)
            .map_err(op("batched mlp silu"))?
            .mul(&up);
        let mlp = linear(&activated, &self.mlp_down_proj)?;
        Ok((hidden.add(&mlp), state))
    }
}
