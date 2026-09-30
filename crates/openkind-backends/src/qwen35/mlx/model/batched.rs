//! Vectorized batched continuation for the MLX Qwen3.5 backbone.

use mlx_rs::fast;
use mlx_rs::Array;

use crate::qwen35::mlx::layers::{
    gather_last_real_token, lane_of, prefix_axis0, stack_lanes, MlxDecoderLayer, MlxFullState,
    MlxLayerState, MlxLinearState, HIDDEN_SIZE,
};
use crate::qwen35::mlx::{MlxError, MlxPrecision};

use super::helpers::{array_to_host, materialize_state};
use super::state::MlxBackboneState;
use super::types::{MlxBackboneOutput, MAX_VECTOR_BATCH_LANES};
use super::MlxQwen35Backbone;

impl MlxQwen35Backbone {
    /// Whether this instance can execute the current vectorized continuation
    /// implementation without changing its arithmetic profile.
    #[must_use]
    pub(crate) fn supports_vectorized_batch(&self) -> bool {
        self.precision == MlxPrecision::Fp32
            && self
                .layers
                .iter()
                .all(MlxDecoderLayer::supports_reference_batch)
    }

    pub(crate) fn can_vectorize_batch(
        &self,
        states: &[&MlxBackboneState],
        suffix_ids: &[&[u32]],
    ) -> bool {
        if !self.supports_vectorized_batch()
            || !(2..=MAX_VECTOR_BATCH_LANES).contains(&states.len())
            || states.len() != suffix_ids.len()
            || suffix_ids.iter().any(|suffix| suffix.is_empty())
        {
            return false;
        }
        let position = states[0].position;
        states.iter().all(|state| state.position == position)
    }

    /// Continue multiple equal-position states through one FP32 reference-ops
    /// graph with right-padded suffixes. The returned states preserve each
    /// lane's true position and lineage.
    pub fn continue_batch_vectorized(
        &self,
        states: &[&MlxBackboneState],
        suffix_ids: &[&[u32]],
    ) -> Result<Vec<(MlxBackboneOutput, MlxBackboneState)>, MlxError> {
        if !self.supports_vectorized_batch() {
            return Err(MlxError::InvalidState(
                "vectorized continuation currently supports only FP32 reference-ops".to_owned(),
            ));
        }
        let lengths = suffix_ids
            .iter()
            .map(|suffix| suffix.len())
            .collect::<Vec<_>>();
        let positions = states
            .iter()
            .map(|state| state.position)
            .collect::<Vec<_>>();
        let (position_start, rows) =
            validate_vectorized_batch_inputs(states.len(), &lengths, &positions)?;
        for (lane, state) in states.iter().enumerate() {
            if state.identity != self.identity {
                return Err(MlxError::InvalidState(format!(
                    "continuation lane {lane} has a different arithmetic identity"
                )));
            }
            if !std::sync::Arc::ptr_eq(&state.runtime, &self.runtime) {
                return Err(MlxError::InvalidState(format!(
                    "continuation lane {lane} belongs to a different MLX runtime"
                )));
            }
            self.validate_state(state)?;
        }
        let lanes = states.len();
        let next_position = position_start.checked_add(rows).ok_or_else(|| {
            MlxError::InvalidState("vectorized continuation position overflow".to_owned())
        })?;
        i32::try_from(next_position).map_err(|_| {
            MlxError::InvalidState(
                "vectorized continuation position does not fit MLX dimensions".to_owned(),
            )
        })?;

        let mut padded_embeddings = vec![0.0_f32; lanes * rows * HIDDEN_SIZE];
        for (lane, suffix) in suffix_ids.iter().enumerate() {
            let embedding = self.embedding.embed(suffix).map_err(MlxError::from_qwen)?;
            let start = lane * rows * HIDDEN_SIZE;
            let end = start + suffix.len() * HIDDEN_SIZE;
            padded_embeddings[start..end].copy_from_slice(embedding.values());
        }

        self.runtime.execute(|| -> Result<_, MlxError> {
            let mut hidden = Array::from_slice(
                &padded_embeddings,
                &[lanes as i32, rows as i32, HIDDEN_SIZE as i32],
            );
            let mut next_layers = Vec::with_capacity(self.layers.len());
            for (layer_index, layer) in self.layers.iter().enumerate() {
                let previous = stack_previous_layer(states, layer_index)?;
                let (next_hidden, next_state) =
                    layer.forward_batched(&hidden, rows, &lengths, position_start, &previous)?;
                hidden = next_hidden;
                hidden.eval().map_err(|error| MlxError::Operation {
                    operation: "batched decoder layer eval",
                    message: format!("layer {layer_index}: {error}"),
                })?;
                next_layers.push(next_state);
            }
            let final_array =
                fast::rms_norm(&hidden, Some(&self.final_norm), 1e-6).map_err(|error| {
                    MlxError::Operation {
                        operation: "batched final rms norm",
                        message: error.to_string(),
                    }
                })?;
            materialize_state(&next_layers)?;

            let mut outputs = Vec::with_capacity(lanes);
            for (lane, state) in states.iter().enumerate() {
                let feature =
                    array_to_host(&gather_last_real_token(&final_array, lane, lengths[lane])?)?;
                let lane_layers =
                    split_batch_layer_states(&next_layers, lane, position_start, lengths[lane])?;
                outputs.push((
                    MlxBackboneOutput {
                        token_count: lengths[lane],
                        final_feature: feature,
                    },
                    MlxBackboneState {
                        runtime: std::sync::Arc::clone(&self.runtime),
                        identity: self.identity.clone(),
                        lineage: state.lineage,
                        position: position_start + lengths[lane],
                        layers: lane_layers,
                    },
                ));
            }
            Ok(outputs)
        })?
    }
}

pub(super) fn validate_vectorized_batch_inputs(
    lanes: usize,
    suffix_lengths: &[usize],
    positions: &[usize],
) -> Result<(usize, usize), MlxError> {
    if !(2..=MAX_VECTOR_BATCH_LANES).contains(&lanes)
        || suffix_lengths.len() != lanes
        || positions.len() != lanes
    {
        return Err(MlxError::InvalidState(format!(
            "vectorized continuation supports 2..={MAX_VECTOR_BATCH_LANES} matching lanes, found {lanes} states, {} suffixes, and {} positions",
            suffix_lengths.len(),
            positions.len()
        )));
    }
    if suffix_lengths.contains(&0) {
        return Err(MlxError::InvalidState(
            "vectorized continuation contains an empty suffix".to_owned(),
        ));
    }
    let position_start = positions[0];
    if positions.iter().any(|position| *position != position_start) {
        return Err(MlxError::InvalidState(
            "vectorized continuation requires equal starting positions".to_owned(),
        ));
    }
    let rows = suffix_lengths.iter().copied().max().unwrap_or(0);
    Ok((position_start, rows))
}

pub(super) fn stack_previous_layer(
    states: &[&MlxBackboneState],
    layer_index: usize,
) -> Result<MlxLayerState, MlxError> {
    match states[0].layers.get(layer_index) {
        Some(MlxLayerState::Linear(_)) => {
            let mut conv = Vec::with_capacity(states.len());
            let mut recurrent = Vec::with_capacity(states.len());
            for state in states {
                match state.layers.get(layer_index) {
                    Some(MlxLayerState::Linear(layer)) => {
                        conv.push(&layer.conv);
                        recurrent.push(&layer.recurrent);
                    }
                    _ => {
                        return Err(MlxError::InvalidState(format!(
                            "continuation cache kind differs at layer {layer_index}"
                        )));
                    }
                }
            }
            Ok(MlxLayerState::Linear(MlxLinearState {
                conv: stack_lanes(&conv)?,
                recurrent: stack_lanes(&recurrent)?,
            }))
        }
        Some(MlxLayerState::Full(_)) => {
            let mut keys = Vec::with_capacity(states.len());
            let mut values = Vec::with_capacity(states.len());
            for state in states {
                match state.layers.get(layer_index) {
                    Some(MlxLayerState::Full(layer)) => {
                        keys.push(&layer.keys);
                        values.push(&layer.values);
                    }
                    _ => {
                        return Err(MlxError::InvalidState(format!(
                            "continuation cache kind differs at layer {layer_index}"
                        )));
                    }
                }
            }
            Ok(MlxLayerState::Full(MlxFullState {
                keys: stack_lanes(&keys)?,
                values: stack_lanes(&values)?,
            }))
        }
        None => Err(MlxError::InvalidState(format!(
            "continuation state has no cache for layer {layer_index}"
        ))),
    }
}

pub(super) fn split_batch_layer_states(
    batched: &[MlxLayerState],
    lane: usize,
    position_start: usize,
    suffix_length: usize,
) -> Result<Vec<MlxLayerState>, MlxError> {
    batched
        .iter()
        .enumerate()
        .map(|(layer_index, state)| match state {
            MlxLayerState::Linear(layer) => Ok(MlxLayerState::Linear(MlxLinearState {
                conv: lane_of(&layer.conv, lane)?,
                recurrent: lane_of(&layer.recurrent, lane)?,
            })),
            MlxLayerState::Full(layer) => {
                let valid_position =
                    position_start.checked_add(suffix_length).ok_or_else(|| {
                        MlxError::InvalidState(format!(
                            "attention position overflow at layer {layer_index}"
                        ))
                    })?;
                let keys = prefix_axis0(&lane_of(&layer.keys, lane)?, valid_position)?;
                let values = prefix_axis0(&lane_of(&layer.values, lane)?, valid_position)?;
                Ok(MlxLayerState::Full(MlxFullState { keys, values }))
            }
        })
        .collect()
}
