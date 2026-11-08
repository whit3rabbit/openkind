//! Fused Metal kernel for QK normalization, scaling, and head expansion.
//!
//! Inspired by TurboSpark's `gdn_qk_norm` Metal kernel in `gdn.metal`.
//!
//! Replaces the 15-operation pipeline (`split`, `reshape`, `square`, `sum_axis`,
//! `recip_sqrt_eps`, query scaling, and `repeat_heads_rows`) with a single
//! GPU launch with 32-lane coalesced SIMD reductions and head broadcasting.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype};

use super::metal_kernel::{scoped_execution_stream, MetalKernel, MetalKernelConfig};
use super::{MlxError, HEAD_DIM, QKV_SIZE, VALUE_HEADS};
use sources::{INPUTS, OUTPUTS, SOURCE};

mod sources;

#[cfg(test)]
pub(crate) mod tests;

/// Environment switch controlling whether fused QK norm is enabled.
pub const FUSED_QK_NORM_ENV: &str = "OPENKIND_QWEN35_MLX_FUSED_QK_NORM";

/// Whether fused QK norm is enabled.
///
/// Enabled by default. Setting `OPENKIND_QWEN35_MLX_FUSED_QK_NORM=0` falls
/// back to the multi-operation split, norm, and repeat path.
pub fn fused_qk_norm_enabled() -> bool {
    match std::env::var(FUSED_QK_NORM_ENV) {
        Ok(val) => val != "0" && !val.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn qk_norm_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new("gdn_qk_norm_split_expand_fp32_v1", INPUTS, OUTPUTS, SOURCE)
    })
}

/// Execute fused QK normalization, scaling, and head expansion on `qkv`.
///
/// # Arguments
/// - `qkv`: `[rows, 8192]` Float32 tensor.
/// - `eps`: RMSNorm epsilon (`1e-6`).
///
/// # Returns
/// - `query_hat`: `[rows, 32, 128]` Float32 tensor.
/// - `key_hat`: `[rows, 32, 128]` Float32 tensor.
/// - `value`: `[rows, 32, 128]` Float32 tensor.
pub(crate) fn fused_qk_norm_split_expand(
    qkv: &Array,
    eps: f32,
) -> Result<(Array, Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;

    let shape = qkv.shape();
    if shape.len() != 2 {
        return Err(MlxError::InvalidState(format!(
            "fused qk norm expects 2D qkv [rows, {QKV_SIZE}], got {shape:?}"
        )));
    }
    let rows = shape[0] as usize;
    let channels = shape[1] as usize;

    if channels != QKV_SIZE {
        return Err(MlxError::InvalidState(format!(
            "fused qk norm expects channels={QKV_SIZE}, got {channels}"
        )));
    }
    if rows == 0 {
        return Err(MlxError::InvalidState(
            "fused qk norm requires rows > 0".to_owned(),
        ));
    }
    if qkv.dtype() != Dtype::Float32 {
        return Err(MlxError::InvalidState(
            "fused qk norm requires Float32 tensor".to_owned(),
        ));
    }

    let row_count = Array::from_slice(&[rows as u32], &[1]);
    let eps_arr = Array::from_slice(&[eps], &[1]);

    let config = MetalKernelConfig::new();
    let out_shape = [rows as i32, VALUE_HEADS as i32, HEAD_DIM as i32];
    config.output(&out_shape, Dtype::Float32)?;
    config.output(&out_shape, Dtype::Float32)?;
    config.output(&out_shape, Dtype::Float32)?;

    // Launch: (32 * 32, rows, 1) grid with (32, 1, 1) threadgroups
    config.grid(32 * VALUE_HEADS, rows, 1)?;
    config.threadgroup(32, 1, 1)?;

    let outputs = qk_norm_kernel().apply(&[qkv, &row_count, &eps_arr], &config, &stream)?;

    if outputs.len() != 3 {
        return Err(MlxError::Operation {
            operation: "fused qk norm",
            message: format!("expected 3 outputs, got {}", outputs.len()),
        });
    }

    let mut iter = outputs.into_iter();
    Ok((
        iter.next().unwrap(),
        iter.next().unwrap(),
        iter.next().unwrap(),
    ))
}
