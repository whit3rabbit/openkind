//! Diagnostic tracing methods for the MLX Qwen3.5 backbone.

use super::state::MlxBackboneState;
use super::types::MlxBackboneTrace;
use super::{MlxError, MlxQwen35Backbone};

impl MlxQwen35Backbone {
    /// Prefill while collecting per-layer host vectors for parity diagnostics.
    ///
    /// This is intentionally separate from [`Self::prefill`] so benchmark and
    /// serving paths cannot accidentally pay for 32 diagnostic readbacks.
    pub fn prefill_trace(
        &self,
        input_ids: &[u32],
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        let execution = self.execute(input_ids, None, true, None, None)?;
        let (
            embedding_last_token,
            layer_last_tokens,
            layer_output_dtypes,
            operation_trace_layer,
            operation_trace,
        ) = execution.trace.ok_or_else(|| {
            MlxError::InvalidState("diagnostic trace was not collected".to_owned())
        })?;
        Ok((
            MlxBackboneTrace {
                output: execution.output,
                embedding_last_token,
                layer_last_tokens,
                layer_output_dtypes,
                operation_trace_layer,
                operation_trace,
            },
            execution.state,
        ))
    }

    /// Prefill while retaining detailed operation arrays for a suffix of the
    /// full input. This is intended for one-off cache/full diagnostics.
    pub fn prefill_trace_tail(
        &self,
        input_ids: &[u32],
        trace_tail_rows: usize,
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        self.prefill_trace_layer_tail(input_ids, trace_tail_rows, 1)
    }

    /// Prefill while tracing a selected layer over the final input rows.
    pub fn prefill_trace_layer_tail(
        &self,
        input_ids: &[u32],
        trace_tail_rows: usize,
        operation_trace_layer: usize,
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        if trace_tail_rows == 0 || trace_tail_rows > input_ids.len() {
            return Err(MlxError::InvalidState(format!(
                "diagnostic tail length {trace_tail_rows} is invalid for {} input rows",
                input_ids.len()
            )));
        }
        if operation_trace_layer >= self.layers.len() {
            return Err(MlxError::InvalidState(format!(
                "diagnostic layer {operation_trace_layer} is out of range for {} layers",
                self.layers.len()
            )));
        }
        let execution = self.execute(
            input_ids,
            None,
            true,
            Some(trace_tail_rows),
            Some(operation_trace_layer),
        )?;
        let (
            embedding_last_token,
            layer_last_tokens,
            layer_output_dtypes,
            operation_trace_layer,
            operation_trace,
        ) = execution.trace.ok_or_else(|| {
            MlxError::InvalidState("diagnostic trace was not collected".to_owned())
        })?;
        Ok((
            MlxBackboneTrace {
                output: execution.output,
                embedding_last_token,
                layer_last_tokens,
                layer_output_dtypes,
                operation_trace_layer,
                operation_trace,
            },
            execution.state,
        ))
    }

    /// Continue a state while collecting the final hidden vector and dtype
    /// after each layer. This is for targeted parity localization only.
    pub fn continue_from_trace(
        &self,
        state: &MlxBackboneState,
        suffix_ids: &[u32],
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        self.continue_from_trace_layer(state, suffix_ids, 1)
    }

    /// Continue a state while tracing one selected decoder layer.
    pub fn continue_from_trace_layer(
        &self,
        state: &MlxBackboneState,
        suffix_ids: &[u32],
        operation_trace_layer: usize,
    ) -> Result<(MlxBackboneTrace, MlxBackboneState), MlxError> {
        if state.identity != self.identity {
            return Err(MlxError::InvalidState(format!(
                "continuation state arithmetic identity `{}` does not match this executor `{}`",
                state.identity.arithmetic_id(),
                self.identity.arithmetic_id()
            )));
        }
        if state.layers.len() != self.layers.len() {
            return Err(MlxError::InvalidState(format!(
                "continuation state has {} layers, expected {}",
                state.layers.len(),
                self.layers.len()
            )));
        }
        if !std::sync::Arc::ptr_eq(&state.runtime, &self.runtime) {
            return Err(MlxError::InvalidState(
                "continuation state belongs to a different MLX runtime".to_owned(),
            ));
        }
        self.validate_state(state)?;
        if operation_trace_layer >= self.layers.len() {
            return Err(MlxError::InvalidState(format!(
                "diagnostic layer {operation_trace_layer} is out of range for {} layers",
                self.layers.len()
            )));
        }
        let execution = self.execute(
            suffix_ids,
            Some(state),
            true,
            Some(suffix_ids.len()),
            Some(operation_trace_layer),
        )?;
        let (
            embedding_last_token,
            layer_last_tokens,
            layer_output_dtypes,
            operation_trace_layer,
            operation_trace,
        ) = execution.trace.ok_or_else(|| {
            MlxError::InvalidState("diagnostic trace was not collected".to_owned())
        })?;
        Ok((
            MlxBackboneTrace {
                output: execution.output,
                embedding_last_token,
                layer_last_tokens,
                layer_output_dtypes,
                operation_trace_layer,
                operation_trace,
            },
            execution.state,
        ))
    }
}
