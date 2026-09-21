use std::ops::{Add, Mul, Sub};

use mlx_rs::fast;
use mlx_rs::ops::maximum;
use mlx_rs::Array;

use super::ops::{
    linear, op, recip_sqrt_eps, repeat_heads, row_of, scalar_like, shift_window, split_last,
    stack_rows, zeros_for_precision,
};
use super::{
    MlxError, MlxLayerState, MlxLinearState, MlxPrecision, CONV_KERNEL, HEAD_DIM, KEY_HEADS,
    KEY_SIZE, QKV_SIZE, RMS_EPSILON, VALUE_HEADS, VALUE_SIZE,
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
}

impl LinearAttention {
    pub(crate) fn forward(
        &self,
        normalized: &Array,
        rows: usize,
        previous: Option<&MlxLinearState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let raw_qkv = linear(normalized, &self.in_proj_qkv)?;
        let z_all = linear(normalized, &self.in_proj_z)?;
        let beta_all = linear(normalized, &self.in_proj_b)?;
        let decay_all = linear(normalized, &self.in_proj_a)?;

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

        let mut mixed_rows = Vec::with_capacity(rows);
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
            )?;
            mixed_rows.push(mixed);
            conv = window;
        }
        let mixed = stack_rows(&mixed_rows, VALUE_SIZE)?;
        let output = linear(&mixed, &self.out_proj)?;
        Ok((
            output,
            MlxLayerState::Linear(MlxLinearState { conv, recurrent }),
        ))
    }
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
) -> Result<Array, MlxError> {
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

    // Per-key-head L2 norms; the query additionally carries 1/sqrt(128).
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

    // Expand 16 key heads to 32 value heads (key head = value head / 2).
    let query_vh = repeat_heads(&query)?;
    let query_norm_vh = repeat_heads(&query_norm)?;
    let key_vh = repeat_heads(&key)?;
    let key_norm_vh = repeat_heads(&key_norm)?;

    let beta = mlx_rs::nn::sigmoid(beta_row).map_err(op("beta sigmoid"))?;
    // decay = exp(-exp(a_log) * softplus(decay_proj + dt_bias))
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
    let decay = a_log
        .clone()
        .exp()
        .map_err(op("a_log exp"))?
        .mul(&softplus)
        .negative()
        .map_err(op("decay negate"))?
        .exp()
        .map_err(op("decay exp"))?;

    // state *= decay (per head)
    let decay_heads = decay
        .reshape(&[VALUE_HEADS as i32, 1, 1])
        .map_err(op("decay reshape"))?;
    *recurrent = recurrent.clone().mul(&decay_heads);

    // delta = (v - Sᵀ k̂) * β
    let key_hat = key_vh.mul(&key_norm_vh);
    let key_col = key_hat
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32, 1])
        .map_err(op("key col"))?;
    let state_t = recurrent
        .clone()
        .transpose_axes(&[0, 2, 1])
        .map_err(op("state t"))?;
    let memory = state_t
        .matmul(&key_col)
        .map_err(op("memory matmul"))?
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("memory reshape"))?;
    let beta_heads = beta
        .reshape(&[VALUE_HEADS as i32, 1])
        .map_err(op("beta reshape"))?;
    let delta = value.clone().sub(&memory).mul(&beta_heads);

    // S += k̂ ⊗ delta
    let delta_row = delta
        .clone()
        .reshape(&[VALUE_HEADS as i32, 1, HEAD_DIM as i32])
        .map_err(op("delta row"))?;
    let outer = key_col.mul(&delta_row);
    *recurrent = recurrent.clone().add(&outer);

    // out = Sᵀ q̂
    let query_hat = query_vh.mul(&query_norm_vh);
    let query_col = query_hat
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32, 1])
        .map_err(op("query col"))?;
    let state_t = recurrent
        .clone()
        .transpose_axes(&[0, 2, 1])
        .map_err(op("state t out"))?;
    let z_hat = z_row
        .clone()
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("z reshape"))?;
    let out = state_t
        .matmul(&query_col)
        .map_err(op("retrieval matmul"))?
        .reshape(&[VALUE_HEADS as i32, HEAD_DIM as i32])
        .map_err(op("out reshape"))?;
    // per-head RMSNorm (raw weight) * silu(z)
    let gated = fast::rms_norm(&out, Some(delta_norm), RMS_EPSILON)
        .map_err(op("delta norm"))?
        .mul(&mlx_rs::nn::silu(&z_hat).map_err(op("z silu"))?);
    gated
        .reshape(&[VALUE_SIZE as i32])
        .map_err(op("out flatten"))
}
