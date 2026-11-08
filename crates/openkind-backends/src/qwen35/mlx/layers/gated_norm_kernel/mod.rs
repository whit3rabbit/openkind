//! Fused Metal kernel for gated output normalization.
//!
//! Inspired by TurboSpark's `gdn_gated_norm` Metal kernel in `gdn.metal`.
//!
//! Computes `out = rmsnorm(y; weight, eps) * silu(z)` per value head in a
//! single GPU launch with 32-lane coalesced SIMD reductions, eliminating
//! separate MLX operations for `fast::rms_norm`, `silu`, `mul`, and
//! intermediate allocations.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype};

use super::metal_kernel::{scoped_execution_stream, MetalKernel, MetalKernelConfig};
use super::MlxError;
use sources::{INPUTS, OUTPUTS, SOURCE};

mod sources;

#[cfg(test)]
pub(crate) mod tests;

/// Environment switch controlling whether fused gated norm is enabled.
pub const FUSED_GATED_NORM_ENV: &str = "OPENKIND_QWEN35_MLX_FUSED_GATED_NORM";

/// Whether fused gated norm is enabled.
///
/// Enabled by default. Setting `OPENKIND_QWEN35_MLX_FUSED_GATED_NORM=0` falls
/// back to the separate `fast::rms_norm` and `silu` path.
pub fn fused_gated_norm_enabled() -> bool {
    match std::env::var(FUSED_GATED_NORM_ENV) {
        Ok(val) => val != "0" && !val.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn gated_norm_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| MetalKernel::new("gdn_gated_norm_silu_fp32_v1", INPUTS, OUTPUTS, SOURCE))
}

/// Execute fused gated output norm: `out = rmsnorm(y; weight, eps) * silu(z)`.
///
/// # Arguments
/// - `y`: `[rows, v_heads, value_dim]` or `[rows, v_heads * value_dim]` Float32 tensor.
/// - `z`: Float32 tensor matching `y`'s total element count.
/// - `weight`: `[value_dim]` Float32 normalization scale weights.
/// - `eps`: RMSNorm epsilon (typically `1e-6`).
///
/// # Returns
/// - `out`: `[rows, v_heads * value_dim]` Float32 tensor.
pub(crate) fn fused_gated_norm_silu(
    y: &Array,
    z: &Array,
    weight: &Array,
    eps: f32,
) -> Result<Array, MlxError> {
    let stream = scoped_execution_stream()?;

    let y_shape = y.shape();
    let weight_shape = weight.shape();

    if weight_shape.len() != 1 {
        return Err(MlxError::InvalidState(format!(
            "fused gated norm expects 1D weight [value_dim], got {weight_shape:?}"
        )));
    }
    let value_dim = weight_shape[0] as usize;
    if value_dim == 0 {
        return Err(MlxError::InvalidState(
            "fused gated norm requires value_dim > 0".to_owned(),
        ));
    }

    let (rows, v_heads) = match y_shape.len() {
        2 => {
            let total = y_shape[1] as usize;
            if !total.is_multiple_of(value_dim) {
                return Err(MlxError::InvalidState(format!(
                    "fused gated norm 2D shape [{}, {total}] indivisible by value_dim {value_dim}",
                    y_shape[0]
                )));
            }
            (y_shape[0] as usize, total / value_dim)
        }
        3 => {
            if y_shape[2] as usize != value_dim {
                return Err(MlxError::InvalidState(format!(
                    "fused gated norm 3D shape {y_shape:?} last dim does not match weight dim {value_dim}"
                )));
            }
            (y_shape[0] as usize, y_shape[1] as usize)
        }
        _ => {
            return Err(MlxError::InvalidState(format!(
                "fused gated norm expects 2D or 3D tensor y, got shape {y_shape:?}"
            )));
        }
    };

    if rows == 0 || v_heads == 0 {
        return Err(MlxError::InvalidState(
            "fused gated norm requires rows > 0 and v_heads > 0".to_owned(),
        ));
    }

    let total_elements = rows * v_heads * value_dim;
    if z.shape().iter().map(|&d| d as usize).product::<usize>() != total_elements {
        return Err(MlxError::InvalidState(format!(
            "fused gated norm z elements {} do not match y elements {total_elements}",
            z.shape().iter().map(|&d| d as usize).product::<usize>()
        )));
    }

    if y.dtype() != Dtype::Float32
        || z.dtype() != Dtype::Float32
        || weight.dtype() != Dtype::Float32
    {
        return Err(MlxError::InvalidState(
            "fused gated norm requires Float32 tensors".to_owned(),
        ));
    }

    let row_count = Array::from_slice(&[rows as u32], &[1]);
    let v_heads_arr = Array::from_slice(&[v_heads as u32], &[1]);
    let value_dim_arr = Array::from_slice(&[value_dim as u32], &[1]);
    let eps_arr = Array::from_slice(&[eps], &[1]);

    let config = MetalKernelConfig::new();
    let total_size = (v_heads * value_dim) as i32;
    config.output(&[rows as i32, total_size], Dtype::Float32)?;

    // Launch: (32 * v_heads, rows, 1) grid with (32, 1, 1) threadgroups
    config.grid(32 * v_heads, rows, 1)?;
    config.threadgroup(32, 1, 1)?;

    let outputs = gated_norm_kernel().apply(
        &[
            y,
            z,
            weight,
            &row_count,
            &v_heads_arr,
            &value_dim_arr,
            &eps_arr,
        ],
        &config,
        &stream,
    )?;

    if outputs.len() != 1 {
        return Err(MlxError::Operation {
            operation: "fused gated norm",
            message: format!("expected 1 output, got {}", outputs.len()),
        });
    }

    Ok(outputs.into_iter().next().unwrap())
}
