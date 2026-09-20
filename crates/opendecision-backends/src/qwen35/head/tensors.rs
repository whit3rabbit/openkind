//! Safetensors validation and deserialization helpers for the fitted head.

use safetensors::{tensor::Dtype, SafeTensors};

use super::types::SCORE_SUMMARY_WIDTH;
use crate::qwen35::Qwen35Error;

pub(crate) fn validate_tensor(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<(), Qwen35Error> {
    let tensor = tensors.tensor(name)?;
    if tensor.dtype() != Dtype::F32 {
        return Err(Qwen35Error::InvalidTensor {
            name: name.into(),
            message: format!("expected F32, found {:?}", tensor.dtype()),
        });
    }
    if tensor.shape() != shape {
        return Err(Qwen35Error::InvalidTensor {
            name: name.into(),
            message: format!("expected shape {shape:?}, found {:?}", tensor.shape()),
        });
    }
    Ok(())
}

// `slice::as_chunks` is newer than the workspace's Rust 1.75 minimum.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub(crate) fn tensor_values(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<Vec<f64>, Qwen35Error> {
    let tensor = tensors.tensor(name)?;
    let chunks = tensor.data().chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(Qwen35Error::InvalidTensor {
            name: name.into(),
            message: "F32 byte length is not divisible by four".into(),
        });
    }
    let values: Vec<f64> = chunks
        .map(|bytes| {
            f64::from(f32::from_le_bytes(
                bytes.try_into().expect("four-byte chunk"),
            ))
        })
        .collect();
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Qwen35Error::InvalidTensor {
            name: name.into(),
            message: format!("element {index} is not finite: {value}"),
        });
    }
    Ok(values)
}

pub(crate) fn scalar(tensors: &SafeTensors<'_>, name: &str) -> Result<f64, Qwen35Error> {
    let values = tensor_values(tensors, name)?;
    values
        .first()
        .copied()
        .ok_or_else(|| Qwen35Error::InvalidTensor {
            name: name.into(),
            message: "scalar tensor has no value".into(),
        })
}

pub(crate) fn array6(
    values: Vec<f64>,
    name: &str,
) -> Result<[f64; SCORE_SUMMARY_WIDTH], Qwen35Error> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| Qwen35Error::InvalidTensor {
            name: name.into(),
            message: format!("expected six values, found {}", values.len()),
        })
}

pub(crate) fn validate_positive_std(name: &str, values: &[f64]) -> Result<(), Qwen35Error> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value <= 0.0)
    {
        Err(Qwen35Error::InvalidTensor {
            name: name.into(),
            message: format!("element {index} must be greater than zero, found {value}"),
        })
    } else {
        Ok(())
    }
}
