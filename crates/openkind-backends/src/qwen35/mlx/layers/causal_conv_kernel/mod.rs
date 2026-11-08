//! Fused Metal kernel for causal depthwise convolution with SiLU activation.
//!
//! Inspired by TurboSpark's `gdn_conv_mix_prefill` and `short_conv` Metal kernels,
//! adapted to OpenKind's `[channels, 4]` convolution state representation.
//!
//! Replaces the multi-operation concatenation, take_axis, stack, mul, and sum_axis
//! pipeline with a single GPU dispatch per layer that produces both the convolved
//! output sequence and the updated carried state.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype};

use super::metal_kernel::{scoped_execution_stream, MetalKernel, MetalKernelConfig};
use super::MlxError;
use sources::{CAUSAL_CONV_SOURCE, INPUTS, OUTPUTS};

mod sources;

#[cfg(test)]
mod tests;

/// Environment switch controlling whether fused causal convolution is enabled.
pub const FUSED_CONV_ENV: &str = "OPENKIND_QWEN35_MLX_FUSED_CONV";

/// Whether fused causal depthwise convolution is enabled.
///
/// Enabled by default. Setting `OPENKIND_QWEN35_MLX_FUSED_CONV=0` falls back
/// to the multi-operation hoisted convolution path.
pub fn fused_conv_enabled() -> bool {
    match std::env::var(FUSED_CONV_ENV) {
        Ok(val) => val != "0" && !val.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn causal_conv_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "fused_causal_conv_silu_v1",
            INPUTS,
            OUTPUTS,
            CAUSAL_CONV_SOURCE,
        )
    })
}

/// Execute the fused causal depthwise convolution with SiLU activation.
///
/// # Arguments
/// - `raw_qkv`: `[rows, channels]` input activations.
/// - `conv`: `[channels, 4]` previous carried state window.
/// - `conv_w`: `[channels, 4]` convolution weights.
///
/// # Returns
/// - `output`: `[rows, channels]` SiLU-activated convolution result.
/// - `next_conv`: `[channels, 4]` updated carried state window.
pub(crate) fn fused_causal_conv_silu(
    raw_qkv: &Array,
    conv: &Array,
    conv_w: &Array,
) -> Result<(Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;

    let raw_shape = raw_qkv.shape();
    if raw_shape.len() != 2 {
        return Err(MlxError::InvalidState(format!(
            "fused causal conv expects 2D raw_qkv [rows, channels], got {raw_shape:?}"
        )));
    }
    let rows = raw_shape[0] as usize;
    let channels = raw_shape[1] as usize;
    if rows == 0 {
        return Err(MlxError::InvalidState(
            "fused causal conv requires at least 1 row".to_owned(),
        ));
    }

    let conv_shape = conv.shape();
    if conv_shape != [channels as i32, 4] {
        return Err(MlxError::InvalidState(format!(
            "fused causal conv expects conv shape [{channels}, 4], got {conv_shape:?}"
        )));
    }

    let conv_w_shape = conv_w.shape();
    if conv_w_shape != [channels as i32, 4] {
        return Err(MlxError::InvalidState(format!(
            "fused causal conv expects conv_w shape [{channels}, 4], got {conv_w_shape:?}"
        )));
    }

    if raw_qkv.dtype() != Dtype::Float32
        || conv.dtype() != Dtype::Float32
        || conv_w.dtype() != Dtype::Float32
    {
        return Err(MlxError::InvalidState(
            "fused causal conv currently requires Float32 tensors".to_owned(),
        ));
    }

    let row_count = Array::from_slice(&[rows as u32], &[1]);
    let channel_count = Array::from_slice(&[channels as u32], &[1]);

    let config = MetalKernelConfig::new();
    config.output(&[rows as i32, channels as i32], Dtype::Float32)?;
    config.output(&[channels as i32, 4], Dtype::Float32)?;

    let grid_x = channels.div_ceil(256) * 256;
    config.grid(grid_x, rows, 1)?;
    config.threadgroup(256, 1, 1)?;

    let outputs = causal_conv_kernel().apply(
        &[raw_qkv, conv, conv_w, &row_count, &channel_count],
        &config,
        &stream,
    )?;

    if outputs.len() != 2 {
        return Err(MlxError::Operation {
            operation: "fused causal conv",
            message: format!("expected 2 outputs, got {}", outputs.len()),
        });
    }

    let mut iter = outputs.into_iter();
    Ok((iter.next().unwrap(), iter.next().unwrap()))
}
