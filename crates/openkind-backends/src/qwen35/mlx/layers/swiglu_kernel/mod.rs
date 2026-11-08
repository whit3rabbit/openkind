//! Fused Metal kernel for SwiGLU activation (`silu(gate) * up`).
//!
//! Inspired by TurboSpark's `silu_mul_fp16` in `utility.metal`.
//!
//! Fuses the separate `silu` activation, intermediate allocation, and
//! elementwise multiplication into a single GPU dispatch with vectorized
//! float4 loads and stores.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype};

use super::metal_kernel::{scoped_execution_stream, MetalKernel, MetalKernelConfig};
use super::MlxError;
use sources::{INPUTS, OUTPUTS, SOURCE};

mod sources;

#[cfg(test)]
pub(crate) mod tests;

/// Environment switch controlling whether fused SwiGLU is enabled.
pub const FUSED_SWIGLU_ENV: &str = "OPENKIND_QWEN35_MLX_FUSED_SWIGLU";

/// Whether fused SwiGLU is enabled.
///
/// Enabled by default. Setting `OPENKIND_QWEN35_MLX_FUSED_SWIGLU=0` falls
/// back to separate `silu` and `mul` operations.
pub fn fused_swiglu_enabled() -> bool {
    match std::env::var(FUSED_SWIGLU_ENV) {
        Ok(val) => val != "0" && !val.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn swiglu_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| MetalKernel::new("qwen35_swiglu_fp32_v1", INPUTS, OUTPUTS, SOURCE))
}

/// Execute fused SwiGLU activation: `out = silu(gate) * up`.
///
/// # Arguments
/// - `gate`: Float32 tensor.
/// - `up`: Float32 tensor with the same shape as `gate`.
///
/// # Returns
/// - `out`: Float32 tensor with `silu(gate) * up`.
pub(crate) fn fused_swiglu(gate: &Array, up: &Array) -> Result<Array, MlxError> {
    let stream = scoped_execution_stream()?;

    let shape = gate.shape();
    if shape != up.shape() {
        return Err(MlxError::InvalidState(format!(
            "fused swiglu shape mismatch: gate {shape:?} vs up {:?}",
            up.shape()
        )));
    }
    if gate.dtype() != Dtype::Float32 || up.dtype() != Dtype::Float32 {
        return Err(MlxError::InvalidState(
            "fused swiglu requires Float32 tensors".to_owned(),
        ));
    }

    let total_elements = shape.iter().map(|&d| d as usize).product::<usize>();
    if total_elements == 0 {
        return Err(MlxError::InvalidState(
            "fused swiglu requires total_elements > 0".to_owned(),
        ));
    }

    let element_count = Array::from_slice(&[total_elements as u32], &[1]);

    let config = MetalKernelConfig::new();
    config.output(shape, Dtype::Float32)?;

    // Launch: (total_elements + 3) / 4 threads, 256 threads per threadgroup
    let vec_threads = total_elements.div_ceil(4);
    config.grid(vec_threads, 1, 1)?;
    config.threadgroup(256.min(vec_threads), 1, 1)?;

    let outputs = swiglu_kernel().apply(&[gate, up, &element_count], &config, &stream)?;

    if outputs.len() != 1 {
        return Err(MlxError::Operation {
            operation: "fused swiglu",
            message: format!("expected 1 output, got {}", outputs.len()),
        });
    }

    Ok(outputs.into_iter().next().unwrap())
}
