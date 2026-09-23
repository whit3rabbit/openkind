use std::ops::{Add, Mul, Sub};

use mlx_rs::fast;
use mlx_rs::ops::maximum;
use mlx_rs::Array;

use super::gated_delta_kernel::{
    generic_gated_delta, packed_gated_delta_128, packed_gated_delta_sequence_128, GateLayout,
};
use super::ops::{
    batch_row_of, linear, op, recip_sqrt_eps, repeat_heads, row_of, scalar_like, shift_window,
    split_last, stack_batch_time, stack_rows, zeros_for_precision,
};
use super::{trace_operation, LayerOperationArrays};
use super::{
    MlxError, MlxGatedDeltaKernel, MlxLayerState, MlxLinearState, MlxPrecision, CONV_KERNEL,
    HEAD_DIM, KEY_HEADS, KEY_SIZE, QKV_SIZE, RMS_EPSILON, VALUE_HEADS, VALUE_SIZE,
};

pub(crate) struct LinearAttention {
    pub(crate) in_proj_qkv: Array,
    pub(crate) in_proj_z: Array,
    pub(crate) in_proj_b: Array,
    pub(crate) in_proj_a: Array,
    pub(crate) conv1d: Array,
    pub(crate) dt_bias: Array,
    pub(crate) a_log: Array,
    pub(crate) delta_norm: Array,
    pub(crate) out_proj: Array,
    pub(crate) precision: MlxPrecision,
    pub(crate) gated_delta_kernel: MlxGatedDeltaKernel,
}

impl LinearAttention {
    #[cfg(test)]
    pub(crate) fn forward(
        &self,
        normalized: &Array,
        rows: usize,
        previous: Option<&MlxLinearState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        self.forward_traced(normalized, rows, previous, None)
    }

    pub(crate) fn forward_traced(
        &self,
        normalized: &Array,
        rows: usize,
        previous: Option<&MlxLinearState>,
        mut operation_trace: Option<&mut LayerOperationArrays>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let raw_qkv = linear(normalized, &self.in_proj_qkv)?;
        trace_operation(&mut operation_trace, "linear_qkv_projection", &raw_qkv);
        let z_all = linear(normalized, &self.in_proj_z)?;
        trace_operation(&mut operation_trace, "linear_z_projection", &z_all);
        let beta_all = linear(normalized, &self.in_proj_b)?;
        trace_operation(&mut operation_trace, "linear_beta_projection", &beta_all);
        let decay_all = linear(normalized, &self.in_proj_a)?;
        trace_operation(&mut operation_trace, "linear_decay_projection", &decay_all);

        let mut conv = match previous {
            Some(state) => state.conv.clone(),
            None => zeros_for_precision(&[QKV_SIZE as i32, CONV_KERNEL as i32], self.precision),
        };
        let mut recurrent = match previous {
            Some(state) => state.recurrent.clone(),
            None => zeros_for_precision(
                &[VALUE_HEADS as i32, HEAD_DIM as i32, HEAD_DIM as i32],
                self.precision,
            ),
        };

        if self.gated_delta_kernel == MlxGatedDeltaKernel::MetalTree
            && recurrent.dtype() == mlx_rs::Dtype::Float32
        {
            return self.forward_metal_tree(
                rows, &raw_qkv, &z_all, &beta_all, &decay_all, conv, recurrent,
            );
        }

        let mut mixed_rows = Vec::with_capacity(rows);
        let mut conv_sum_rows = operation_trace.as_ref().map(|_| Vec::with_capacity(rows));
        let mut conv_silu_rows = operation_trace.as_ref().map(|_| Vec::with_capacity(rows));
        for row in 0..rows {
            // ---- causal depthwise conv over the window [x[r-K+1] .. x[r]] ----
            // The window drops the oldest tap and includes the CURRENT raw
            // row before the weighted sum; the outgoing state is the same
            // window (last K raw inputs).
            let raw_row = row_of(&raw_qkv, row)?;
            let window = shift_window(&conv, &raw_row)?;
            let summed = window
                .clone()
                .mul(&self.conv1d)
                .sum_axis(1, false)
                .map_err(op("conv sum"))?;
            let qkv_row = mlx_rs::nn::silu(&summed).map_err(op("conv silu"))?;
            if let Some(trace_rows) = &mut conv_sum_rows {
                trace_rows.push(summed.clone());
            }
            if let Some(trace_rows) = &mut conv_silu_rows {
                trace_rows.push(qkv_row.clone());
            }
            // ---- per-token projections ----
            let beta_row = row_of(&beta_all, row)?;
            let decay_row = row_of(&decay_all, row)?;
            let z_row = row_of(&z_all, row)?;
            // ---- DeltaNet recurrence, vectorized across value heads ----
            let mixed = gated_delta_step(
                &qkv_row,
                &z_row,
                &beta_row,
                &decay_row,
                &self.dt_bias,
                &self.a_log,
                &self.delta_norm,
                &mut recurrent,
                self.gated_delta_kernel,
            )?;
            mixed_rows.push(mixed);
            conv = window;
        }
        let mixed = stack_rows(&mixed_rows, VALUE_SIZE)?;
        if let Some(trace_rows) = conv_sum_rows {
            let conv_sum = stack_rows(&trace_rows, QKV_SIZE)?;
            trace_operation(&mut operation_trace, "linear_conv_window_sum", &conv_sum);
        }
        if let Some(trace_rows) = conv_silu_rows {
            let conv_silu = stack_rows(&trace_rows, QKV_SIZE)?;
            trace_operation(&mut operation_trace, "linear_conv_silu", &conv_silu);
        }
        trace_operation(&mut operation_trace, "linear_gated_delta_output", &mixed);
        let output = linear(&mixed, &self.out_proj)?;
        trace_operation(&mut operation_trace, "linear_out_projection", &output);
        Ok((
            output,
            MlxLayerState::Linear(MlxLinearState { conv, recurrent }),
        ))
    }

    /// Vectorized lane-axis continuation using reference array operations.
    /// Inputs are `[lanes, rows, hidden]`; padded lanes are frozen after their
    /// true length so their recurrent and convolution state remains exact.
    pub(crate) fn forward_batched(
        &self,
        normalized: &Array,
        rows: usize,
        lengths: &[usize],
        previous: &MlxLinearState,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        if self.gated_delta_kernel != MlxGatedDeltaKernel::ReferenceOps {
            return Err(MlxError::InvalidState(
                "vectorized MLX continuation supports only reference-ops DeltaNet".to_owned(),
            ));
        }
        let lanes = lengths.len();
        if normalized.shape() != [lanes as i32, rows as i32, super::HIDDEN_SIZE as i32]
            || previous.conv.shape() != [lanes as i32, QKV_SIZE as i32, CONV_KERNEL as i32]
            || previous.recurrent.shape()
                != [
                    lanes as i32,
                    VALUE_HEADS as i32,
                    HEAD_DIM as i32,
                    HEAD_DIM as i32,
                ]
        {
            return Err(MlxError::InvalidState(
                "invalid batched DeltaNet input or state shape".to_owned(),
            ));
        }

        let raw_qkv = linear(normalized, &self.in_proj_qkv)?;
        let z_all = linear(normalized, &self.in_proj_z)?;
        let beta_all = linear(normalized, &self.in_proj_b)?;
        let decay_all = linear(normalized, &self.in_proj_a)?;
        let mut conv = previous.conv.clone();
        let mut recurrent = previous.recurrent.clone();
        let mut mixed_rows = Vec::with_capacity(rows);

        for row in 0..rows {
            let raw_row = batch_row_of(&raw_qkv, row)?;
            let window = shift_window_batched(&conv, &raw_row)?;
            let summed = window
                .clone()
                .mul(&self.conv1d)
                .sum_axis(2, false)
                .map_err(op("batched conv sum"))?;
            let qkv_row = mlx_rs::nn::silu(&summed).map_err(op("batched conv silu"))?;
            let prepared = prepare_gated_delta_inputs_batched(
                &qkv_row,
                &batch_row_of(&z_all, row)?,
                &batch_row_of(&beta_all, row)?,
                &batch_row_of(&decay_all, row)?,
                &self.dt_bias,
                &self.a_log,
            )?;
            let (mixed, candidate_recurrent) = reference_gated_delta_batched(
                &prepared.query,
                &prepared.key,
                &prepared.value,
                &prepared.beta,
                &prepared.gate,
                &recurrent,
            )?;
            let gated = rms_norm_batched_heads(&mixed, &self.delta_norm)?
                .mul(&mlx_rs::nn::silu(&prepared.z).map_err(op("batched z silu"))?);
            let gated = gated
                .reshape(&[lanes as i32, VALUE_SIZE as i32])
                .map_err(op("batched delta flatten"))?;
            mixed_rows.push(gated);

            let active = lengths
                .iter()
                .map(|length| if row < *length { 1.0_f32 } else { 0.0_f32 })
                .collect::<Vec<_>>();
            let active_conv = Array::from_slice(&active, &[lanes as i32, 1, 1]);
            let active_recurrent = Array::from_slice(&active, &[lanes as i32, 1, 1, 1]);
            conv = candidate_or_previous(&window, &conv, &active_conv);
            recurrent = candidate_or_previous(&candidate_recurrent, &recurrent, &active_recurrent);
        }

        let mixed = stack_batch_time(&mixed_rows)?;
        let output = linear(&mixed, &self.out_proj)?;
        Ok((
            output,
            MlxLayerState::Linear(MlxLinearState { conv, recurrent }),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_metal_tree(
        &self,
        rows: usize,
        raw_qkv: &Array,
        z_all: &Array,
        beta_all: &Array,
        decay_all: &Array,
        mut conv: Array,
        recurrent: Array,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let mut queries = Vec::with_capacity(rows);
        let mut keys = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(rows);
        let mut betas = Vec::with_capacity(rows);
        let mut gates = Vec::with_capacity(rows);
        let mut z_rows = Vec::with_capacity(rows);

        for row in 0..rows {
            let raw_row = row_of(raw_qkv, row)?;
            let window = shift_window(&conv, &raw_row)?;
            let summed = window
                .clone()
                .mul(&self.conv1d)
                .sum_axis(1, false)
                .map_err(op("conv sum"))?;
            let qkv_row = mlx_rs::nn::silu(&summed).map_err(op("conv silu"))?;
            let prepared = prepare_gated_delta_inputs(
                &qkv_row,
                &row_of(z_all, row)?,
                &row_of(beta_all, row)?,
                &row_of(decay_all, row)?,
                &self.dt_bias,
                &self.a_log,
            )?;
            queries.push(prepared.query);
            keys.push(prepared.key);
            values.push(prepared.value);
            betas.push(prepared.beta);
            gates.push(prepared.gate);
            z_rows.push(prepared.z);
            conv = window;
        }

        let sequence_shape = [rows as i32, VALUE_HEADS as i32, HEAD_DIM as i32];
        let query = stack_rows(&queries, VALUE_SIZE)?
            .reshape(&sequence_shape)
            .map_err(op("sequence query reshape"))?;
        let key = stack_rows(&keys, VALUE_SIZE)?
            .reshape(&sequence_shape)
            .map_err(op("sequence key reshape"))?;
        let value = stack_rows(&values, VALUE_SIZE)?
            .reshape(&sequence_shape)
            .map_err(op("sequence value reshape"))?;
        let beta = stack_rows(&betas, VALUE_HEADS)?;
        let gate = stack_rows(&gates, VALUE_HEADS)?;
        let z = stack_rows(&z_rows, VALUE_SIZE)?
            .reshape(&sequence_shape)
            .map_err(op("sequence z reshape"))?;
        let (mixed, recurrent) =
            packed_gated_delta_sequence_128(&query, &key, &value, &beta, &gate, &recurrent)?;
        let gated = fast::rms_norm(&mixed, Some(&self.delta_norm), RMS_EPSILON)
            .map_err(op("delta norm"))?
            .mul(&mlx_rs::nn::silu(&z).map_err(op("z silu"))?);
        let mixed = gated
            .reshape(&[rows as i32, VALUE_SIZE as i32])
            .map_err(op("sequence output reshape"))?;
        let output = linear(&mixed, &self.out_proj)?;
        Ok((
            output,
            MlxLayerState::Linear(MlxLinearState { conv, recurrent }),
        ))
    }
}

fn shift_window_batched(conv: &Array, raw_current: &Array) -> Result<Array, MlxError> {
    if conv.shape().len() != 3 || raw_current.shape() != [conv.shape()[0], conv.shape()[1]] {
        return Err(MlxError::InvalidState(
            "invalid batched convolution state or row shape".to_owned(),
        ));
    }
    let columns = conv
        .split_at_indices(&[1], 2)
        .map_err(op("batched conv split"))?;
    let current = raw_current
        .clone()
        .reshape(&[conv.shape()[0], conv.shape()[1], 1])
        .map_err(op("batched conv current"))?;
    mlx_rs::ops::concatenate(&[&columns[1], &current], 2).map_err(op("batched conv window"))
}

fn candidate_or_previous(candidate: &Array, previous: &Array, active: &Array) -> Array {
    let inactive = scalar_like(active, 1.0).sub(active);
    candidate
        .clone()
        .mul(active)
        .add(&previous.clone().mul(&inactive))
}

fn prepare_gated_delta_inputs_batched(
    qkv_row: &Array,
    z_row: &Array,
    beta_row: &Array,
    decay_row: &Array,
    dt_bias: &Array,
    a_log: &Array,
) -> Result<GatedDeltaInputs, MlxError> {
    let lanes = qkv_row.shape()[0] as usize;
    let split_at = [KEY_SIZE as i32, (KEY_SIZE * 2) as i32];
    let parts = qkv_row
        .clone()
        .split_at_indices(&split_at, 1)
        .map_err(op("batched qkv split"))?;
    let query = parts[0]
        .clone()
        .reshape(&[lanes as i32, KEY_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("batched q reshape"))?;
    let key = parts[1]
        .clone()
        .reshape(&[lanes as i32, KEY_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("batched k reshape"))?;
    let value = parts[2]
        .clone()
        .reshape(&[lanes as i32, VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("batched v reshape"))?;

    let q_sq = query
        .clone()
        .square()
        .map_err(op("batched q square"))?
        .sum_axis(-1, true)
        .map_err(op("batched q sum"))?;
    let k_sq = key
        .clone()
        .square()
        .map_err(op("batched k square"))?
        .sum_axis(-1, true)
        .map_err(op("batched k sum"))?;
    let query_norm =
        recip_sqrt_eps(&q_sq)?.mul(scalar_like(&q_sq, (HEAD_DIM as f32).sqrt().recip()));
    let key_norm = recip_sqrt_eps(&k_sq)?;
    let query_hat = repeat_heads_batched(&query)?.mul(&repeat_heads_batched(&query_norm)?);
    let key_hat = repeat_heads_batched(&key)?.mul(&repeat_heads_batched(&key_norm)?);

    let beta = mlx_rs::nn::sigmoid(beta_row).map_err(op("batched beta sigmoid"))?;
    let dt = decay_row.clone().add(dt_bias);
    let softplus = mlx_rs::ops::maximum(&dt, scalar_like(&dt, 0.0))
        .map_err(op("batched dt maximum"))?
        .add(
            &dt.abs()
                .map_err(op("batched dt abs"))?
                .negative()
                .map_err(op("batched dt abs negate"))?
                .exp()
                .map_err(op("batched dt exp"))?
                .log1p()
                .map_err(op("batched dt log1p"))?,
        );
    let gate = a_log
        .clone()
        .exp()
        .map_err(op("batched a_log exp"))?
        .mul(&softplus)
        .negative()
        .map_err(op("batched decay negate"))?
        .exp()
        .map_err(op("batched decay exp"))?;
    let z = z_row
        .clone()
        .reshape(&[lanes as i32, VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("batched z reshape"))?;
    Ok(GatedDeltaInputs {
        query: query_hat,
        key: key_hat,
        value,
        beta,
        gate,
        z,
    })
}

fn repeat_heads_batched(heads: &Array) -> Result<Array, MlxError> {
    let shape = heads.shape();
    if shape.len() != 3 || shape[1] as usize != KEY_HEADS {
        return Err(MlxError::InvalidState(format!(
            "invalid batched key-head shape {:?}",
            shape
        )));
    }
    let times = (VALUE_HEADS / KEY_HEADS) as i32;
    let reshaped = heads
        .clone()
        .reshape(&[shape[0], shape[1], 1, shape[2]])
        .map_err(op("batched repeat reshape"))?;
    mlx_rs::ops::broadcast_to(&reshaped, &[shape[0], shape[1], times, shape[2]])
        .map_err(op("batched repeat broadcast"))?
        .reshape(&[shape[0], (VALUE_HEADS) as i32, shape[2]])
        .map_err(op("batched repeat flatten"))
}

fn reference_gated_delta_batched(
    query_hat: &Array,
    key_hat: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
) -> Result<(Array, Array), MlxError> {
    let shape = state.shape();
    let lanes = shape[0];
    let decay_heads = gate
        .reshape(&[lanes, VALUE_HEADS as i32, 1, 1])
        .map_err(op("batched decay reshape"))?;
    let decayed = state.clone().mul(&decay_heads);
    let key_col = key_hat
        .clone()
        .reshape(&[lanes, VALUE_HEADS as i32, HEAD_DIM as i32, 1])
        .map_err(op("batched key col"))?;
    let memory = decayed
        .clone()
        .transpose_axes(&[0, 1, 3, 2])
        .map_err(op("batched state t"))?
        .matmul(&key_col)
        .map_err(op("batched memory matmul"))?
        .reshape(&[lanes, VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("batched memory reshape"))?;
    let beta_heads = beta
        .clone()
        .reshape(&[lanes, VALUE_HEADS as i32, 1])
        .map_err(op("batched beta reshape"))?;
    let delta = value.clone().sub(&memory).mul(&beta_heads);
    let delta_row = delta
        .reshape(&[lanes, VALUE_HEADS as i32, 1, HEAD_DIM as i32])
        .map_err(op("batched delta row"))?;
    let next_state = decayed.add(&key_col.mul(&delta_row));
    let query_col = query_hat
        .clone()
        .reshape(&[lanes, VALUE_HEADS as i32, HEAD_DIM as i32, 1])
        .map_err(op("batched query col"))?;
    let out = next_state
        .clone()
        .transpose_axes(&[0, 1, 3, 2])
        .map_err(op("batched state t out"))?
        .matmul(&query_col)
        .map_err(op("batched retrieval matmul"))?
        .reshape(&[lanes, VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("batched out reshape"))?;
    Ok((out, next_state))
}

fn rms_norm_batched_heads(values: &Array, weight: &Array) -> Result<Array, MlxError> {
    let shape = values.shape();
    if shape.len() != 3 {
        return Err(MlxError::InvalidState(format!(
            "batched head RMSNorm expects rank 3, found {:?}",
            shape
        )));
    }
    let lanes_heads = shape[0] * shape[1];
    fast::rms_norm(
        &values
            .clone()
            .reshape(&[lanes_heads, shape[2]])
            .map_err(op("batched norm reshape"))?,
        Some(weight),
        RMS_EPSILON,
    )
    .map_err(op("batched delta norm"))?
    .reshape(shape)
    .map_err(op("batched norm restore"))
}

struct GatedDeltaInputs {
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    gate: Array,
    z: Array,
}

#[allow(clippy::too_many_arguments)]
fn prepare_gated_delta_inputs(
    qkv_row: &Array,
    z_row: &Array,
    beta_row: &Array,
    decay_row: &Array,
    dt_bias: &Array,
    a_log: &Array,
) -> Result<GatedDeltaInputs, MlxError> {
    let query_scale = (HEAD_DIM as f32).sqrt().recip();
    let parts = split_last(qkv_row, &[KEY_SIZE, KEY_SIZE])?;
    let query = parts[0]
        .clone()
        .reshape(&[KEY_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("q reshape"))?;
    let key = parts[1]
        .clone()
        .reshape(&[KEY_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("k reshape"))?;
    let value = parts[2]
        .clone()
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("v reshape"))?;

    let q_sq = query
        .clone()
        .square()
        .map_err(op("q square"))?
        .sum_axis(1, true)
        .map_err(op("q sum"))?;
    let k_sq = key
        .clone()
        .square()
        .map_err(op("k square"))?
        .sum_axis(1, true)
        .map_err(op("k sum"))?;
    let query_norm = recip_sqrt_eps(&q_sq)?.mul(scalar_like(&q_sq, query_scale));
    let key_norm = recip_sqrt_eps(&k_sq)?;
    let query_vh = repeat_heads(&query)?;
    let query_norm_vh = repeat_heads(&query_norm)?;
    let key_vh = repeat_heads(&key)?;
    let key_norm_vh = repeat_heads(&key_norm)?;

    let beta = mlx_rs::nn::sigmoid(beta_row).map_err(op("beta sigmoid"))?;
    let dt = decay_row.clone().add(dt_bias);
    let softplus = maximum(&dt, scalar_like(&dt, 0.0))
        .map_err(op("dt maximum"))?
        .add(
            &dt.abs()
                .map_err(op("dt abs"))?
                .negative()
                .map_err(op("dt abs negate"))?
                .exp()
                .map_err(op("dt exp"))?
                .log1p()
                .map_err(op("dt log1p"))?,
        );
    let gate = a_log
        .clone()
        .exp()
        .map_err(op("a_log exp"))?
        .mul(&softplus)
        .negative()
        .map_err(op("decay negate"))?
        .exp()
        .map_err(op("decay exp"))?;
    let z = z_row
        .clone()
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("z reshape"))?;
    Ok(GatedDeltaInputs {
        query: query_vh.mul(&query_norm_vh),
        key: key_vh.mul(&key_norm_vh),
        value,
        beta,
        gate,
        z,
    })
}

/// One Gated DeltaNet recurrence step over all 32 value heads.
///
/// Direct port of the oracle's `gated_delta_recurrent_with_state` for one
/// token: query/key head grouping (16 key heads shared pairwise), per-head
/// L2 normalization with the fixed `1/sqrt(128)` query scale, per-head
/// decayed state, `(v - Sᵀk̂)·β` delta, outer update `S += k̂⊗δ`,
/// retrieval `Sᵀq̂`, per-head RMSNorm with the **raw** (unfolded) norm
/// weight, and the SiLU gate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_step(
    qkv_row: &Array,
    z_row: &Array,
    beta_row: &Array,
    decay_row: &Array,
    dt_bias: &Array,
    a_log: &Array,
    delta_norm: &Array,
    recurrent: &mut Array,
    kernel: MlxGatedDeltaKernel,
) -> Result<Array, MlxError> {
    let prepared = prepare_gated_delta_inputs(qkv_row, z_row, beta_row, decay_row, dt_bias, a_log)?;
    let (out, next_state) = match kernel {
        MlxGatedDeltaKernel::MetalTree if recurrent.dtype() == mlx_rs::Dtype::Float32 => {
            packed_gated_delta_128(
                &prepared.query,
                &prepared.key,
                &prepared.value,
                &prepared.beta,
                &prepared.gate,
                recurrent,
            )?
        }
        MlxGatedDeltaKernel::MetalTree => generic_gated_delta(
            &prepared.query,
            &prepared.key,
            &prepared.value,
            &prepared.beta,
            &prepared.gate,
            recurrent,
            None,
            GateLayout::Scalar,
        )?,
        MlxGatedDeltaKernel::ReferenceOps => reference_gated_delta(
            &prepared.query,
            &prepared.key,
            &prepared.value,
            &prepared.beta,
            &prepared.gate,
            recurrent,
        )?,
    };
    *recurrent = next_state;

    // per-head RMSNorm (raw weight) * silu(z)
    let gated = fast::rms_norm(&out, Some(delta_norm), RMS_EPSILON)
        .map_err(op("delta norm"))?
        .mul(&mlx_rs::nn::silu(&prepared.z).map_err(op("z silu"))?);
    gated
        .reshape(&[VALUE_SIZE as i32])
        .map_err(op("out flatten"))
}

/// Independent ordinary-ops comparator and fallback for the fused kernels.
fn reference_gated_delta(
    query_hat: &Array,
    key_hat: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
) -> Result<(Array, Array), MlxError> {
    let decay_heads = gate
        .reshape(&[VALUE_HEADS as i32, 1, 1])
        .map_err(op("decay reshape"))?;
    let decayed = state.clone().mul(&decay_heads);
    let key_col = key_hat
        .clone()
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32, 1])
        .map_err(op("key col"))?;
    let memory = decayed
        .clone()
        .transpose_axes(&[0, 2, 1])
        .map_err(op("state t"))?
        .matmul(&key_col)
        .map_err(op("memory matmul"))?
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("memory reshape"))?;
    let beta_heads = beta
        .clone()
        .reshape(&[VALUE_HEADS as i32, 1])
        .map_err(op("beta reshape"))?;
    let delta = value.clone().sub(&memory).mul(&beta_heads);
    let delta_row = delta
        .reshape(&[VALUE_HEADS as i32, 1, HEAD_DIM as i32])
        .map_err(op("delta row"))?;
    let next_state = decayed.add(&key_col.mul(&delta_row));
    let query_col = query_hat
        .clone()
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32, 1])
        .map_err(op("query col"))?;
    let out = next_state
        .clone()
        .transpose_axes(&[0, 2, 1])
        .map_err(op("state t out"))?
        .matmul(&query_col)
        .map_err(op("retrieval matmul"))?
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("out reshape"))?;
    Ok((out, next_state))
}

#[cfg(test)]
mod vectorized_tests {
    use super::candidate_or_previous;
    use mlx_rs::Array;

    #[test]
    fn inactive_lane_keeps_its_state_and_source_siblings_are_unchanged() {
        let source = Array::from_slice(&[5.0_f32, 7.0], &[2, 1, 1]);
        let sibling = source.clone();
        let candidate = Array::from_slice(&[15.0_f32, 25.0], &[2, 1, 1]);
        let active = Array::from_slice(&[1.0_f32, 0.0], &[2, 1, 1]);
        let advanced = candidate_or_previous(&candidate, &source, &active);

        assert_eq!(advanced.to_vec_cast::<f32>().unwrap(), vec![15.0, 7.0]);
        assert_eq!(source.to_vec_cast::<f32>().unwrap(), vec![5.0, 7.0]);
        assert_eq!(sibling.to_vec_cast::<f32>().unwrap(), vec![5.0, 7.0]);
    }
}
