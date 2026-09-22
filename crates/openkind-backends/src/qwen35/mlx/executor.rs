//! `SequentialNestedExecutor` binding for the MLX backbone (Phase 3M).
//!
//! With this impl, the backend-neutral nested runners
//! (`run_sequential_nested`, `run_with_scheduler`) drive the MLX backbone
//! exactly like the Candle oracle; only the arithmetic identity differs, so
//! cross-backend state mixing fails closed as designed.

use crate::qwen35::backbone::SequentialNestedExecutor;
use crate::qwen35::Qwen35Error;

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
}
