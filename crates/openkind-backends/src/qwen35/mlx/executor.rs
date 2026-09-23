//! `SequentialNestedExecutor` binding for the MLX backbone (Phase 3M).
//!
//! With this impl, the backend-neutral nested runners
//! (`run_sequential_nested`, `run_with_scheduler`) drive the MLX backbone
//! exactly like the Candle oracle; only the arithmetic identity differs, so
//! cross-backend state mixing fails closed as designed.
//!
//! Vectorized dispatch is limited to explicitly qualified FP32 reference-ops
//! calls with 2..=8 lanes at both fan-out stages. Unsupported shapes use the
//! trait's physical per-lane mode, which the scheduler records explicitly.

use crate::qwen35::backbone::{BatchContinuation, SequentialNestedExecutor};
use crate::qwen35::{ExecutionControl, Qwen35Error};
use openkind_runtime::BatchForwardMode;

use super::model::{MlxBackboneState, MlxQwen35Backbone};

impl SequentialNestedExecutor for MlxQwen35Backbone {
    type State = MlxBackboneState;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        let (output, state) =
            MlxQwen35Backbone::prefill(self, input_ids).map_err(Qwen35Error::Mlx)?;
        Ok((output.feature().to_vec(), state))
    }

    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        let (output, next) =
            MlxQwen35Backbone::continue_from(self, state, suffix_ids).map_err(Qwen35Error::Mlx)?;
        Ok((output.feature().to_vec(), next))
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        validate_batch_inputs(states, suffix_ids)?;
        if self.can_vectorize_batch(states, suffix_ids) {
            let outputs = MlxQwen35Backbone::continue_batch_vectorized(self, states, suffix_ids)
                .map_err(Qwen35Error::Mlx)?;
            let lanes = outputs
                .into_iter()
                .map(|(output, state)| (output.feature().to_vec(), state))
                .collect();
            return Ok(BatchContinuation::new(lanes, BatchForwardMode::Vectorized));
        }
        continue_batch_per_lane(self, states, suffix_ids)
    }

    fn continue_batch_from_controlled(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
        control: &ExecutionControl,
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        validate_batch_inputs(states, suffix_ids)?;
        if self.can_vectorize_batch(states, suffix_ids) {
            control.check()?;
            let outputs = MlxQwen35Backbone::continue_batch_vectorized(self, states, suffix_ids)
                .map_err(Qwen35Error::Mlx)?;
            control.check()?;
            let lanes = outputs
                .into_iter()
                .map(|(output, state)| (output.feature().to_vec(), state))
                .collect();
            return Ok(BatchContinuation::new(lanes, BatchForwardMode::Vectorized));
        }

        let mut lanes = Vec::with_capacity(states.len());
        for (&state, &suffix) in states.iter().zip(suffix_ids) {
            control.check()?;
            let (output, next) =
                MlxQwen35Backbone::continue_from(self, state, suffix).map_err(Qwen35Error::Mlx)?;
            control.check()?;
            lanes.push((output.feature().to_vec(), next));
        }
        Ok(BatchContinuation::new(lanes, BatchForwardMode::PerLane))
    }
}

fn validate_batch_inputs(
    states: &[&MlxBackboneState],
    suffix_ids: &[&[u32]],
) -> Result<(), Qwen35Error> {
    if states.is_empty() || states.len() != suffix_ids.len() {
        return Err(Qwen35Error::InvalidInput(format!(
            "MLX batch continuation has {} states and {} suffixes",
            states.len(),
            suffix_ids.len()
        )));
    }
    if suffix_ids.iter().any(|suffix| suffix.is_empty()) {
        return Err(Qwen35Error::InvalidInput(
            "MLX batch continuation contains an empty suffix".into(),
        ));
    }
    Ok(())
}

fn continue_batch_per_lane(
    executor: &MlxQwen35Backbone,
    states: &[&MlxBackboneState],
    suffix_ids: &[&[u32]],
) -> Result<BatchContinuation<MlxBackboneState>, Qwen35Error> {
    let mut lanes = Vec::with_capacity(states.len());
    for (&state, &suffix) in states.iter().zip(suffix_ids) {
        let (output, next) =
            MlxQwen35Backbone::continue_from(executor, state, suffix).map_err(Qwen35Error::Mlx)?;
        lanes.push((output.feature().to_vec(), next));
    }
    Ok(BatchContinuation::new(lanes, BatchForwardMode::PerLane))
}
