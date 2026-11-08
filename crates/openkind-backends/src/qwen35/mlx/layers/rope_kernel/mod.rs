//! Fused Metal kernel for Qwen 3.5 partial RoPE.
//!
//! Inspired by TurboSpark's `rope_neox_subdim_rows` in `rope.metal`.
//!
//! Replaces the host CPU sin/cos generation, memory allocations, splits,
//! and elementwise MLX math with a single GPU dispatch per attention projection.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype};

use super::metal_kernel::{scoped_execution_stream, MetalKernel, MetalKernelConfig};
use super::{MlxError, ATTENTION_HEAD_DIM, ROPE_THETA};
use sources::{INPUTS, OUTPUTS, SOURCE};

mod sources;

#[cfg(test)]
pub(crate) mod tests;

/// Environment switch controlling whether fused RoPE is enabled.
pub const FUSED_ROPE_ENV: &str = "OPENKIND_QWEN35_MLX_FUSED_ROPE";

/// Whether fused RoPE is enabled.
///
/// Enabled by default. Setting `OPENKIND_QWEN35_MLX_FUSED_ROPE=0` falls
/// back to the multi-operation split, norm, and repeat path.
pub fn fused_rope_enabled() -> bool {
    match std::env::var(FUSED_ROPE_ENV) {
        Ok(val) => val != "0" && !val.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn rope_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new("qwen35_rope_neox_subdim_fp32_v1", INPUTS, OUTPUTS, SOURCE)
    })
}

/// Execute fused partial rotary embedding on `values`.
///
/// # Arguments
/// - `values`: `[rows, heads, 256]` Float32 tensor.
/// - `position_start`: starting token position.
/// - `heads`: number of heads.
///
/// # Returns
/// - `out`: `[rows, heads, 256]` Float32 tensor with partial RoPE applied.
pub(crate) fn fused_apply_rotary(
    values: &Array,
    rows: usize,
    position_start: usize,
    heads: usize,
) -> Result<Array, MlxError> {
    let stream = scoped_execution_stream()?;

    let shape = values.shape();
    if shape.len() != 3 {
        return Err(MlxError::InvalidState(format!(
            "fused rope expects 3D tensor [rows, heads, {ATTENTION_HEAD_DIM}], got {shape:?}"
        )));
    }
    if shape[0] as usize != rows
        || shape[1] as usize != heads
        || shape[2] as usize != ATTENTION_HEAD_DIM
    {
        return Err(MlxError::InvalidState(format!(
            "fused rope shape mismatch: expected [{rows}, {heads}, {ATTENTION_HEAD_DIM}], got {shape:?}"
        )));
    }
    if rows == 0 {
        return Err(MlxError::InvalidState(
            "fused rope requires rows > 0".to_owned(),
        ));
    }
    if values.dtype() != Dtype::Float32 {
        return Err(MlxError::InvalidState(
            "fused rope requires Float32 tensor".to_owned(),
        ));
    }

    let row_count = Array::from_slice(&[rows as u32], &[1]);
    let head_count = Array::from_slice(&[heads as u32], &[1]);
    let pos_start = Array::from_slice(&[position_start as u32], &[1]);
    let theta_arr = Array::from_slice(&[ROPE_THETA], &[1]);

    let config = MetalKernelConfig::new();
    let out_shape = [rows as i32, heads as i32, ATTENTION_HEAD_DIM as i32];
    config.output(&out_shape, Dtype::Float32)?;

    // Launch: (128, heads, rows) grid with (128, 1, 1) threadgroups
    config.grid(128, heads, rows)?;
    config.threadgroup(128, 1, 1)?;

    let outputs = rope_kernel().apply(
        &[values, &row_count, &head_count, &pos_start, &theta_arr],
        &config,
        &stream,
    )?;

    if outputs.len() != 1 {
        return Err(MlxError::Operation {
            operation: "fused rope",
            message: format!("expected 1 output, got {}", outputs.len()),
        });
    }

    Ok(outputs.into_iter().next().unwrap())
}
