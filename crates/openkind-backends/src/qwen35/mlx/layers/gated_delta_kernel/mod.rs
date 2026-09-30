//! Fused Metal kernels for one Gated DeltaNet recurrent step.
//!
//! These kernels are derived from the recurrence itself, not copied from
//! mlx-lm. The generic kernel owns one threadgroup per value head and uses a
//! fixed binary reduction tree over the key dimension. Compile-time flags
//! select masked lanes and scalar versus vector decay gates. The packed
//! specialization handles the pinned FP32 `Dk = Dv = 128` shape four value
//! columns at a time.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, Stream};

use super::MlxError;
use ffi::{scoped_execution_stream, MetalKernel, MetalKernelConfig};
use sources::{
    GENERIC_INPUTS, GENERIC_SOURCE, OUTPUTS, PACKED_SEQUENCE_SOURCE, PACKED_SOURCE, SEQUENCE_INPUTS,
};

mod ffi;
mod sources;

#[cfg(test)]
mod tests;

const MAX_KEY_DIM: usize = 256;

/// Gate layout accepted by the generic kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateLayout {
    /// One decay value per value head: `[heads]`.
    Scalar,
    /// One decay value per value-head column: `[heads, value_dim]`.
    Vector,
}

/// Execute the generic reduction-tree kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generic_gated_delta(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
    mask: Option<&Array>,
    gate_layout: GateLayout,
) -> Result<(Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;
    let shape = validate_inputs(query, key, value, beta, gate, state, mask, gate_layout)?;
    if !shape.key_dim.is_power_of_two() || shape.key_dim > MAX_KEY_DIM {
        return Err(MlxError::InvalidState(format!(
            "generic gated-delta key dimension {} must be a power of two no larger than {MAX_KEY_DIM}",
            shape.key_dim
        )));
    }
    let has_mask = mask.is_some();
    let fallback_mask;
    let mask = match mask {
        Some(mask) => mask,
        None => {
            fallback_mask = Array::from_slice(&vec![true; shape.heads], &[shape.heads as i32]);
            &fallback_mask
        }
    };
    apply_kernel(
        generic_kernel(),
        [query, key, value, beta, gate, state, mask],
        shape,
        has_mask,
        gate_layout,
        false,
        &stream,
    )
}

/// Execute the packed FP32 `Dk = Dv = 128` specialization.
pub(crate) fn packed_gated_delta_128(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
) -> Result<(Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;
    let shape = validate_inputs(
        query,
        key,
        value,
        beta,
        gate,
        state,
        None,
        GateLayout::Scalar,
    )?;
    if shape.key_dim != 128 || shape.value_dim != 128 || state.dtype() != Dtype::Float32 {
        return Err(MlxError::InvalidState(
            "packed gated-delta requires FP32 state with Dk = Dv = 128".to_owned(),
        ));
    }
    let mask = Array::from_slice(&vec![true; shape.heads], &[shape.heads as i32]);
    apply_kernel(
        packed_kernel(),
        [query, key, value, beta, gate, state, &mask],
        shape,
        false,
        GateLayout::Scalar,
        true,
        &stream,
    )
}

/// Execute the packed FP32 `Dk = Dv = 128` recurrence for a complete suffix.
///
/// One threadgroup owns one value head. Each lane keeps four state columns in
/// registers while the kernel walks the sequence, removing the per-token graph
/// evaluation boundary required by the diagnostic step kernel.
pub(crate) fn packed_gated_delta_sequence_128(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
) -> Result<(Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;
    let query_shape = query.shape();
    if query_shape.len() != 3 {
        return Err(MlxError::InvalidState(format!(
            "packed gated-delta sequence query must be [rows, heads, 128], found {query_shape:?}"
        )));
    }
    let rows = query_shape[0] as usize;
    let heads = query_shape[1] as usize;
    if rows == 0 || heads != 32 || query_shape[2] != 128 {
        return Err(MlxError::InvalidState(format!(
            "packed gated-delta sequence requires [rows, 32, 128], found {query_shape:?}"
        )));
    }
    let dtype = Dtype::Float32;
    for (name, array, expected) in [
        ("key", key, vec![rows as i32, heads as i32, 128]),
        ("value", value, vec![rows as i32, heads as i32, 128]),
        ("beta", beta, vec![rows as i32, heads as i32]),
        ("gate", gate, vec![rows as i32, heads as i32]),
        ("state", state, vec![heads as i32, 128, 128]),
    ] {
        if array.shape() != expected || array.dtype() != dtype {
            return Err(MlxError::InvalidState(format!(
                "packed gated-delta sequence {name} expected shape {expected:?} and FP32, found {:?} and {:?}",
                array.shape(),
                array.dtype()
            )));
        }
    }
    if query.dtype() != dtype {
        return Err(MlxError::InvalidState(format!(
            "packed gated-delta sequence query must be FP32, found {:?}",
            query.dtype()
        )));
    }

    let row_count = Array::from_slice(&[rows as u32], &[1]);
    let config = MetalKernelConfig::new();
    config.output(&[rows as i32, heads as i32, 128], dtype)?;
    config.output(&[heads as i32, 128, 128], dtype)?;
    config.grid(
        128_usize.checked_mul(heads).ok_or_else(|| {
            MlxError::InvalidState("packed sequence Metal grid overflowed usize".to_owned())
        })?,
        1,
        1,
    )?;
    config.threadgroup(128, 1, 1)?;
    let outputs = packed_sequence_kernel().apply(
        &[query, key, value, beta, gate, state, &row_count],
        &config,
        &stream,
    )?;
    if outputs.len() != 2 {
        return Err(MlxError::Operation {
            operation: "packed gated-delta sequence Metal kernel",
            message: format!("expected 2 outputs, found {}", outputs.len()),
        });
    }
    let mut outputs = outputs.into_iter();
    Ok((
        outputs.next().expect("length checked"),
        outputs.next().expect("length checked"),
    ))
}

#[derive(Debug, Clone, Copy)]
struct KernelShape {
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    dtype: Dtype,
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
    mask: Option<&Array>,
    gate_layout: GateLayout,
) -> Result<KernelShape, MlxError> {
    let state_shape = state.shape();
    if state_shape.len() != 3 {
        return Err(MlxError::InvalidState(format!(
            "gated-delta state must be [heads, Dk, Dv], found {state_shape:?}"
        )));
    }
    let [heads, key_dim, value_dim] = [
        state_shape[0] as usize,
        state_shape[1] as usize,
        state_shape[2] as usize,
    ];
    let dtype = state.dtype();
    if !matches!(dtype, Dtype::Float32 | Dtype::Bfloat16) {
        return Err(MlxError::InvalidState(format!(
            "gated-delta state dtype must be FP32 or BF16, found {dtype:?}"
        )));
    }
    for (name, array, expected) in [
        ("query", query, vec![heads as i32, key_dim as i32]),
        ("key", key, vec![heads as i32, key_dim as i32]),
        ("value", value, vec![heads as i32, value_dim as i32]),
        ("beta", beta, vec![heads as i32]),
    ] {
        if array.shape() != expected || array.dtype() != dtype {
            return Err(MlxError::InvalidState(format!(
                "gated-delta {name} expected shape {expected:?} and dtype {dtype:?}, found {:?} and {:?}",
                array.shape(),
                array.dtype()
            )));
        }
    }
    let expected_gate = match gate_layout {
        GateLayout::Scalar => vec![heads as i32],
        GateLayout::Vector => vec![heads as i32, value_dim as i32],
    };
    if gate.shape() != expected_gate || gate.dtype() != dtype {
        return Err(MlxError::InvalidState(format!(
            "gated-delta gate expected shape {expected_gate:?} and dtype {dtype:?}, found {:?} and {:?}",
            gate.shape(),
            gate.dtype()
        )));
    }
    if let Some(mask) = mask {
        if mask.shape() != [heads as i32] || mask.dtype() != Dtype::Bool {
            return Err(MlxError::InvalidState(format!(
                "gated-delta mask expected bool shape [{heads}], found {:?} and {:?}",
                mask.shape(),
                mask.dtype()
            )));
        }
    }
    Ok(KernelShape {
        heads,
        key_dim,
        value_dim,
        dtype,
    })
}

fn apply_kernel(
    kernel: &'static MetalKernel,
    inputs: [&Array; 7],
    shape: KernelShape,
    has_mask: bool,
    gate_layout: GateLayout,
    packed: bool,
    stream: &Stream,
) -> Result<(Array, Array), MlxError> {
    let config = MetalKernelConfig::new();
    config.output(&[shape.heads as i32, shape.value_dim as i32], shape.dtype)?;
    config.output(
        &[
            shape.heads as i32,
            shape.key_dim as i32,
            shape.value_dim as i32,
        ],
        shape.dtype,
    )?;
    let grid_width = shape.key_dim.checked_mul(shape.heads).ok_or_else(|| {
        MlxError::InvalidState("gated-delta Metal grid width overflowed usize".to_owned())
    })?;
    config.grid(grid_width, 1, 1)?;
    config.threadgroup(shape.key_dim, 1, 1)?;
    if !packed {
        config.template_dtype("T", shape.dtype)?;
        config.template_int("DK", shape.key_dim)?;
        config.template_int("DV", shape.value_dim)?;
        config.template_bool("HAS_MASK", has_mask)?;
        config.template_bool("VECTOR_GATE", gate_layout == GateLayout::Vector)?;
    }
    let outputs = kernel.apply(&inputs, &config, stream)?;
    if outputs.len() != 2 {
        return Err(MlxError::Operation {
            operation: "gated-delta Metal kernel",
            message: format!("expected 2 outputs, found {}", outputs.len()),
        });
    }
    let mut outputs = outputs.into_iter();
    let output = outputs.next().expect("length checked");
    let next_state = outputs.next().expect("length checked");
    // Evaluating either sibling executes the complete multi-output primitive.
    // Do that before constructing the next recurrent invocation: it preserves
    // deterministic state chaining without a GPU-to-host readback.
    output.eval().map_err(|error| MlxError::Operation {
        operation: "materialize gated-delta Metal step",
        message: error.to_string(),
    })?;
    Ok((output, next_state))
}

fn generic_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "openkind_gated_delta_tree",
            GENERIC_INPUTS,
            OUTPUTS,
            GENERIC_SOURCE,
        )
    })
}

fn packed_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "openkind_gated_delta_packed_128",
            GENERIC_INPUTS,
            OUTPUTS,
            PACKED_SOURCE,
        )
    })
}

fn packed_sequence_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "openkind_gated_delta_packed_sequence_128",
            SEQUENCE_INPUTS,
            OUTPUTS,
            PACKED_SEQUENCE_SOURCE,
        )
    })
}
