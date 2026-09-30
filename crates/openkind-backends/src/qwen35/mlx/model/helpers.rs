//! Internal tensor conversion, materialization, and metadata helpers.

use std::collections::BTreeMap;

use mlx_rs::Array;

use crate::qwen35::mlx::layers::{row_of, MlxLayerState, HIDDEN_SIZE};
use crate::qwen35::mlx::runtime::MlxRuntime;
use crate::qwen35::mlx::weights::MlxCheckpointFormat;
use crate::qwen35::mlx::{
    MlxError, MlxGatedDeltaKernel, MlxPrecision, MLX_ARITHMETIC_ID_BF16_METAL_TREE,
    MLX_ARITHMETIC_ID_BF16_REFERENCE, MLX_ARITHMETIC_ID_FP32_METAL_TREE,
    MLX_ARITHMETIC_ID_FP32_REFERENCE,
};

/// Arithmetic identity including precision, kernel family, and the linked
/// Xcode/Metal toolchain that affects native-BF16 behavior.
pub(super) fn arithmetic_id(
    precision: MlxPrecision,
    gated_delta_kernel: MlxGatedDeltaKernel,
    runtime: &MlxRuntime,
    checkpoint_format: MlxCheckpointFormat,
) -> String {
    let base = match (precision, gated_delta_kernel) {
        (MlxPrecision::Fp32, MlxGatedDeltaKernel::ReferenceOps) => MLX_ARITHMETIC_ID_FP32_REFERENCE,
        (MlxPrecision::NativeBf16, MlxGatedDeltaKernel::ReferenceOps) => {
            MLX_ARITHMETIC_ID_BF16_REFERENCE
        }
        (MlxPrecision::Fp32, MlxGatedDeltaKernel::MetalTree) => MLX_ARITHMETIC_ID_FP32_METAL_TREE,
        (MlxPrecision::NativeBf16, MlxGatedDeltaKernel::MetalTree) => {
            MLX_ARITHMETIC_ID_BF16_METAL_TREE
        }
    };
    format!(
        "{base};checkpoint={};toolchain={}",
        checkpoint_format.as_str(),
        runtime.toolchain_identity()
    )
}

pub(super) fn effective_gated_delta_kernel(
    precision: MlxPrecision,
    requested: MlxGatedDeltaKernel,
) -> MlxGatedDeltaKernel {
    match (precision, requested) {
        // The generic BF16 Metal kernel is directly tested, but its current
        // model-backed result exceeds the frozen probability gate. Keep the
        // candidate implementation out of serving/benchmark dispatch until a
        // distinct arithmetic profile qualifies it.
        (MlxPrecision::NativeBf16, MlxGatedDeltaKernel::MetalTree) => {
            MlxGatedDeltaKernel::ReferenceOps
        }
        _ => requested,
    }
}

/// Build `[rows, 2560]` embedding array from host-widened rows.
pub(super) fn embed_array(values: &[f32], precision: MlxPrecision) -> Array {
    match precision {
        MlxPrecision::Fp32 => Array::from_slice(
            values,
            &[(values.len() / HIDDEN_SIZE) as i32, HIDDEN_SIZE as i32],
        ),
        MlxPrecision::NativeBf16 => Array::from_slice(
            &values
                .iter()
                .map(|value| half::bf16::from_f32(*value))
                .collect::<Vec<_>>(),
            &[(values.len() / HIDDEN_SIZE) as i32, HIDDEN_SIZE as i32],
        ),
    }
}

/// Materialize one array to host FP32 (eval first).
pub(super) fn array_to_host(array: &Array) -> Result<Vec<f32>, MlxError> {
    array.eval().map_err(|error| MlxError::Operation {
        operation: "boundary eval",
        message: error.to_string(),
    })?;
    let values = array
        .to_vec_cast::<f32>()
        .map_err(|error| MlxError::Operation {
            operation: "boundary read",
            message: error.to_string(),
        })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MlxError::Operation {
            operation: "boundary finite check",
            message: "MLX produced a non-finite value".to_owned(),
        });
    }
    Ok(values)
}

/// Host FP32 of the final row of a `[rows, width]` array.
pub(super) fn last_row_to_host(array: &Array) -> Result<Vec<f32>, MlxError> {
    let rows = array.shape()[0] as usize;
    let row = row_of(array, rows - 1)?;
    array_to_host(&row)
}

/// Materialize every continuation tensor before it leaves the runtime lock.
pub(super) fn materialize_state(layers: &[MlxLayerState]) -> Result<(), MlxError> {
    for layer in layers {
        let arrays: [&Array; 2] = match layer {
            MlxLayerState::Linear(state) => [&state.conv, &state.recurrent],
            MlxLayerState::Full(state) => [&state.keys, &state.values],
        };
        for array in arrays {
            array.eval().map_err(|error| MlxError::Operation {
                operation: "continuation state eval",
                message: error.to_string(),
            })?;
        }
    }
    Ok(())
}

/// Final RMSNorm weight, folded `(1 + w)` like every other norm.
pub(super) fn folded_final_norm(
    tensors: &BTreeMap<String, Array>,
    precision: MlxPrecision,
) -> Result<Array, MlxError> {
    let raw = tensors
        .get("model.language_model.norm.weight")
        .ok_or_else(|| MlxError::InvalidState("final norm weight was not loaded".to_owned()))?;
    let values = raw
        .to_vec_cast::<f32>()
        .map_err(|error| MlxError::Operation {
            operation: "final norm read",
            message: error.to_string(),
        })?;
    if values.len() != HIDDEN_SIZE {
        return Err(MlxError::InvalidState(format!(
            "final norm weight has {} values, expected {HIDDEN_SIZE}",
            values.len()
        )));
    }
    let folded: Vec<f32> = values.iter().map(|value| 1.0 + value).collect();
    Ok(match precision {
        MlxPrecision::Fp32 => Array::from_slice(&folded, &[HIDDEN_SIZE as i32]),
        MlxPrecision::NativeBf16 => Array::from_slice(
            &folded
                .iter()
                .map(|value| half::bf16::from_f32(*value))
                .collect::<Vec<_>>(),
            &[HIDDEN_SIZE as i32],
        ),
    })
}
