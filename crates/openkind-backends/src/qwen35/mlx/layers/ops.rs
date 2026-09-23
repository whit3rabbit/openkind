use std::collections::BTreeMap;
use std::ops::Add;

use mlx_rs::fast;
use mlx_rs::ops::{broadcast_to, concatenate};
use mlx_rs::{Array, Dtype};

use super::{MlxError, MlxPrecision};
use super::{CONV_KERNEL, KEY_HEADS, QKV_SIZE, RMS_EPSILON, VALUE_HEADS};

pub(crate) fn op<E: std::fmt::Display>(name: &'static str) -> impl Fn(E) -> MlxError {
    move |error| MlxError::Operation {
        operation: name,
        message: error.to_string(),
    }
}

/// Row-major linear: `x @ Wᵀ` for `[rows, in]` against `[out, in]`.
pub(crate) fn linear(input: &Array, weight: &Array) -> Result<Array, MlxError> {
    input
        .matmul(&weight.clone().transpose().map_err(op("weight t"))?)
        .map_err(op("linear matmul"))
}

pub(crate) fn rms_norm(input: &Array, folded_weight: &Array) -> Result<Array, MlxError> {
    fast::rms_norm(input, Some(folded_weight), RMS_EPSILON).map_err(op("rms norm"))
}

/// `1 / sqrt(x + eps)` elementwise.
pub(crate) fn recip_sqrt_eps(x: &Array) -> Result<Array, MlxError> {
    x.clone()
        .add(scalar_like(x, RMS_EPSILON))
        .sqrt()
        .map_err(op("norm sqrt"))?
        .reciprocal()
        .map_err(op("norm recip"))
}

/// Extract row `index` as a flat `[width]` array from `[rows, width]`.
pub(crate) fn row_of(matrix: &Array, index: usize) -> Result<Array, MlxError> {
    matrix
        .take_axis(Array::from_slice(&[index as u32], &[1]), 0)
        .map_err(op("row take"))?
        .reshape(&[matrix.shape()[1]])
        .map_err(op("row reshape"))
}

/// Select the final `length` rows from a token-major array for diagnostics.
pub(crate) fn tail_rows(matrix: &Array, length: usize) -> Result<Array, MlxError> {
    let shape = matrix.shape();
    let available = shape.first().copied().unwrap_or_default().max(0) as usize;
    if shape.len() < 2 || length == 0 || length > available {
        return Err(MlxError::InvalidState(format!(
            "cannot select {length} diagnostic rows from shape {shape:?}"
        )));
    }
    let indices = (available - length..available)
        .map(|index| index as u32)
        .collect::<Vec<_>>();
    matrix
        .take_axis(Array::from_slice(&indices, &[length as i32]), 0)
        .map_err(op("diagnostic tail rows"))
}

/// Extract one sequence row from `[lanes, rows, width]`, preserving lanes.
pub(crate) fn batch_row_of(matrix: &Array, index: usize) -> Result<Array, MlxError> {
    let shape = matrix.shape();
    if shape.len() != 3 {
        return Err(MlxError::InvalidState(format!(
            "batched row extraction expects rank 3, found shape {shape:?}"
        )));
    }
    matrix
        .take_axis(Array::from_slice(&[index as u32], &[1]), 1)
        .map_err(op("batch row take"))?
        .reshape(&[shape[0], shape[2]])
        .map_err(op("batch row reshape"))
}

/// Gather one lane's last real suffix token from `[lanes, rows, width]`.
pub(crate) fn gather_last_real_token(
    hidden: &Array,
    lane: usize,
    true_length: usize,
) -> Result<Array, MlxError> {
    let shape = hidden.shape();
    if shape.len() != 3
        || lane >= shape[0] as usize
        || true_length == 0
        || true_length > shape[1] as usize
    {
        return Err(MlxError::InvalidState(format!(
            "invalid lane {lane} or true length {true_length} for batched hidden shape {shape:?}"
        )));
    }
    lane_of(&batch_row_of(hidden, true_length - 1)?, lane)
}

/// Stack equal-shaped arrays on a new leading lane axis.
pub(crate) fn stack_lanes(lanes: &[&Array]) -> Result<Array, MlxError> {
    let Some(first) = lanes.first() else {
        return Err(MlxError::InvalidState(
            "cannot stack an empty MLX lane batch".to_owned(),
        ));
    };
    let mut shape = Vec::with_capacity(first.shape().len() + 1);
    shape.push(1);
    shape.extend_from_slice(first.shape());
    let mut stacked = (*first)
        .clone()
        .reshape(&shape)
        .map_err(op("lane stack reshape"))?;
    for lane in &lanes[1..] {
        if lane.shape() != first.shape() || lane.dtype() != first.dtype() {
            return Err(MlxError::InvalidState(
                "batched continuation tensors have different shapes or dtypes".to_owned(),
            ));
        }
        let mut shape = Vec::with_capacity(lane.shape().len() + 1);
        shape.push(1);
        shape.extend_from_slice(lane.shape());
        let next = (*lane)
            .clone()
            .reshape(&shape)
            .map_err(op("lane stack reshape"))?;
        stacked = concatenate(&[&stacked, &next], 0).map_err(op("lane stack concat"))?;
    }
    Ok(stacked)
}

/// Extract one lane from an array whose leading dimension is the lane axis.
pub(crate) fn lane_of(array: &Array, lane: usize) -> Result<Array, MlxError> {
    let shape = array.shape();
    if shape.is_empty() {
        return Err(MlxError::InvalidState(
            "cannot select a lane from a scalar MLX array".to_owned(),
        ));
    }
    let mut output_shape = shape[1..].to_vec();
    if output_shape.is_empty() {
        output_shape.push(1);
    }
    array
        .take_axis(Array::from_slice(&[lane as u32], &[1]), 0)
        .map_err(op("lane take"))?
        .reshape(&output_shape)
        .map_err(op("lane reshape"))
}

/// Keep the first `length` elements of axis zero.
pub(crate) fn prefix_axis0(array: &Array, length: usize) -> Result<Array, MlxError> {
    let shape = array.shape();
    if shape.is_empty() || length > shape[0] as usize {
        return Err(MlxError::InvalidState(format!(
            "cannot take length {length} from axis zero of shape {shape:?}"
        )));
    }
    let indices = (0..length as u32).collect::<Vec<_>>();
    array
        .take_axis(Array::from_slice(&indices, &[length as i32]), 0)
        .map_err(op("axis prefix take"))
}

/// Stack `[lanes, width]` token outputs as sequence-major `[lanes, rows, width]`.
pub(crate) fn stack_batch_time(rows: &[Array]) -> Result<Array, MlxError> {
    let Some(first) = rows.first() else {
        return Err(MlxError::InvalidState(
            "cannot stack an empty MLX sequence".to_owned(),
        ));
    };
    if first.shape().len() != 2 {
        return Err(MlxError::InvalidState(format!(
            "batched sequence row must have rank 2, found {:?}",
            first.shape()
        )));
    }
    let [lanes, width] = [first.shape()[0], first.shape()[1]];
    let mut stacked = first
        .clone()
        .reshape(&[lanes, 1, width])
        .map_err(op("batch sequence reshape"))?;
    for row in &rows[1..] {
        if row.shape() != [lanes, width] || row.dtype() != first.dtype() {
            return Err(MlxError::InvalidState(
                "batched sequence rows have different shapes or dtypes".to_owned(),
            ));
        }
        let row = row
            .clone()
            .reshape(&[lanes, 1, width])
            .map_err(op("batch sequence reshape"))?;
        stacked = concatenate(&[&stacked, &row], 1).map_err(op("batch sequence concat"))?;
    }
    Ok(stacked)
}

/// Split a flat row at cumulative widths; returns the pieces flattened.
pub(crate) fn split_last(row: &Array, widths: &[usize]) -> Result<Vec<Array>, MlxError> {
    let mut indices = Vec::new();
    let mut cumulative = 0_i32;
    for width in widths {
        cumulative += *width as i32;
        indices.push(cumulative);
    }
    let reshaped = row
        .clone()
        .reshape(&[1, row.shape()[0]])
        .map_err(op("split reshape"))?;
    let parts = reshaped
        .split_at_indices(&indices, 1)
        .map_err(op("split"))?;
    parts
        .iter()
        .map(|part| {
            part.clone()
                .reshape(&[part.size() as i32])
                .map_err(op("split flatten"))
        })
        .collect()
}

/// Repeat each of `KEY_HEADS` heads `VALUE_HEADS / KEY_HEADS` times.
pub(crate) fn repeat_heads(heads: &Array) -> Result<Array, MlxError> {
    let shape = heads.shape().to_vec();
    let times = (VALUE_HEADS / KEY_HEADS) as i32;
    let reshaped = heads
        .clone()
        .reshape(&[shape[0], 1, shape[1]])
        .map_err(op("repeat reshape"))?;
    broadcast_to(&reshaped, &[shape[0], times, shape[1]])
        .map_err(op("repeat broadcast"))?
        .reshape(&[shape[0] * times, shape[1]])
        .map_err(op("repeat flatten"))
}

/// Shift the conv window: drop the oldest tap column, append the raw (pre-
/// silu) current row as the newest column.
///
/// The model path evaluates the causal depthwise conv as this explicit
/// shifted window rather than native `mlx_conv1d`: the per-token recurrence
/// carries the last-K raw-input window as continuation state, so the
/// shifted-add form transcribes the oracle's tap order directly at every
/// position. Native grouped `mlx_conv1d` — including its true-convolution
/// kernel flip relative to torch/candle cross-correlation — is qualified in
/// the 3M.0 gate and is the intended primitive for the fused 3M.5 path.
pub(crate) fn shift_window(conv: &Array, raw_current: &Array) -> Result<Array, MlxError> {
    // Split at 1 drops the single oldest tap: [`oldest`, `remaining K-1`].
    // Keep the remaining columns and append the current raw row.
    let columns = conv.split_at_indices(&[1], 1).map_err(op("conv split"))?;
    let current = raw_current
        .clone()
        .reshape(&[QKV_SIZE as i32, 1])
        .map_err(op("conv current"))?;
    concatenate(&[&columns[1], &current], 1).map_err(op("conv window"))
}

/// Stack per-token flat rows into `[rows, width]`.
pub(crate) fn stack_rows(rows: &[Array], width: usize) -> Result<Array, MlxError> {
    let mut stacked = rows[0]
        .clone()
        .reshape(&[1, width as i32])
        .map_err(op("stack reshape"))?;
    for row in &rows[1..] {
        stacked = concatenate(
            &[
                &stacked,
                &row.clone()
                    .reshape(&[1, width as i32])
                    .map_err(op("stack row"))?,
            ],
            0,
        )
        .map_err(op("stack concat"))?;
    }
    Ok(stacked)
}

pub(crate) fn zeros(shape: &[i32]) -> Array {
    let count: usize = shape.iter().map(|dim| *dim as usize).product();
    Array::from_slice(&vec![0.0_f32; count], shape)
}

pub(crate) fn zeros_for_precision(shape: &[i32], precision: MlxPrecision) -> Array {
    match precision {
        MlxPrecision::Fp32 => zeros(shape),
        MlxPrecision::NativeBf16 => {
            let count: usize = shape.iter().map(|dim| *dim as usize).product();
            Array::from_slice(&vec![half::bf16::from_f32(0.0); count], shape)
        }
    }
}

pub(crate) fn scalar_like(array: &Array, value: f32) -> Array {
    if array.dtype() == Dtype::Bfloat16 {
        Array::from_slice(&[half::bf16::from_f32(value)], &[1])
    } else {
        Array::from_f32(value)
    }
}

pub(crate) fn tensor_2d(
    tensors: &BTreeMap<String, Array>,
    name: &str,
    shape: [usize; 2],
) -> Result<Array, MlxError> {
    let tensor = tensors.get(name).ok_or_else(|| {
        MlxError::InvalidState(format!("checkpoint tensor `{name}` was not loaded"))
    })?;
    let dims = tensor.shape();
    if dims != [shape[0] as i32, shape[1] as i32] {
        return Err(MlxError::InvalidState(format!(
            "checkpoint tensor `{name}` has shape {dims:?}, expected [{}, {}]",
            shape[0], shape[1]
        )));
    }
    Ok(tensor.clone())
}

/// Load the conv1d checkpoint tensor as `[C, K]`.
///
/// The pinned PyTorch safetensors use `[C, 1, K]`; the MLX-community export
/// uses the equivalent singleton-axis layout `[C, K, 1]`. Both are contiguous
/// `[C, K]` payloads, so the adapter can preserve the values without a copy.
pub(crate) fn conv_weight(
    tensors: &BTreeMap<String, Array>,
    name: &str,
) -> Result<Array, MlxError> {
    let tensor = tensors.get(name).ok_or_else(|| {
        MlxError::InvalidState(format!("checkpoint tensor `{name}` was not loaded"))
    })?;
    let shape = tensor.shape();
    if shape != [QKV_SIZE as i32, 1, CONV_KERNEL as i32]
        && shape != [QKV_SIZE as i32, CONV_KERNEL as i32, 1]
    {
        return Err(MlxError::InvalidState(format!(
            "checkpoint tensor `{name}` has shape {shape:?}, expected [{QKV_SIZE}, 1, {CONV_KERNEL}] or [{QKV_SIZE}, {CONV_KERNEL}, 1]"
        )));
    }
    tensor
        .clone()
        .reshape(&[QKV_SIZE as i32, CONV_KERNEL as i32])
        .map_err(op("conv reshape"))
}

pub(crate) fn tensor_vector(
    tensors: &BTreeMap<String, Array>,
    name: &str,
    width: usize,
) -> Result<Array, MlxError> {
    let tensor = tensors.get(name).ok_or_else(|| {
        MlxError::InvalidState(format!("checkpoint tensor `{name}` was not loaded"))
    })?;
    if tensor.shape() != [width as i32] {
        return Err(MlxError::InvalidState(format!(
            "checkpoint tensor `{name}` has shape {:?}, expected [{width}]",
            tensor.shape()
        )));
    }
    Ok(tensor.clone())
}

/// Load a norm-weight vector and fold `(1 + w)` exactly as the oracle
/// computes it at runtime, then materialize at the execution precision.
pub(crate) fn folded_vector(
    tensors: &BTreeMap<String, Array>,
    name: &str,
    width: usize,
    precision: MlxPrecision,
) -> Result<Array, MlxError> {
    let raw = tensor_vector(tensors, name, width)?;
    let values = raw.to_vec_cast::<f32>().map_err(op("norm read"))?;
    let folded: Vec<f32> = values.iter().map(|value| 1.0 + value).collect();
    Ok(match precision {
        MlxPrecision::Fp32 => Array::from_slice(&folded, &[width as i32]),
        MlxPrecision::NativeBf16 => Array::from_slice(
            &folded
                .iter()
                .map(|value| half::bf16::from_f32(*value))
                .collect::<Vec<_>>(),
            &[width as i32],
        ),
    })
}

#[cfg(test)]
mod vectorized_tests {
    use super::{gather_last_real_token, lane_of, prefix_axis0};
    use mlx_rs::Array;

    #[test]
    fn gather_uses_each_lanes_true_last_token_and_truncates_padding() {
        let hidden = Array::from_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            ],
            &[2, 3, 2],
        );
        assert_eq!(
            gather_last_real_token(&hidden, 0, 3)
                .unwrap()
                .to_vec_cast::<f32>()
                .unwrap(),
            vec![5.0, 6.0]
        );
        assert_eq!(
            gather_last_real_token(&hidden, 1, 1)
                .unwrap()
                .to_vec_cast::<f32>()
                .unwrap(),
            vec![10.0, 20.0]
        );

        let padded_cache = Array::from_slice(&[1.0_f32, 2.0, 3.0, 99.0], &[1, 4, 1]);
        let lane_cache = lane_of(&padded_cache, 0).unwrap();
        let truncated = prefix_axis0(&lane_cache, 3).unwrap();
        assert_eq!(truncated.shape(), [3, 1]);
        assert_eq!(truncated.to_vec_cast::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }
}
