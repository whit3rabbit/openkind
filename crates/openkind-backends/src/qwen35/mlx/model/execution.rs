//! Forward prefill and continuation execution for the MLX Qwen3.5 backbone.

use mlx_rs::fast;
use openkind_runtime::branch::StateLineage;

use crate::qwen35::mlx::layers::{row_of, tail_rows, LayerOperationArrays};
use crate::qwen35::mlx::runtime::SharedMlxRuntime;
use crate::qwen35::mlx::{MlxError, MlxQwen35Backbone};

use super::helpers::{array_to_host, embed_array, last_row_to_host, materialize_state};
use super::state::MlxBackboneState;
use super::types::{BackboneExecution, DeviceExecution, MlxBackboneOutput, MlxOperationTrace};

impl MlxQwen35Backbone {
    /// Prefill a fresh root sequence.
    pub fn prefill(
        &self,
        input_ids: &[u32],
    ) -> Result<(MlxBackboneOutput, MlxBackboneState), MlxError> {
        let execution = self.execute(input_ids, None, false, None, None)?;
        Ok((execution.output, execution.state))
    }

    /// Read the final host embedding row without executing decoder layers.
    ///
    /// This keeps the embedding-only parity diagnostic bounded to tokenizer
    /// and embedding work. Weight loading remains complete because the
    /// verified loader must validate both checkpoint shards before use.
    pub fn embedding_last_token(&self, input_ids: &[u32]) -> Result<Vec<f32>, MlxError> {
        if input_ids.is_empty() {
            return Err(MlxError::InvalidState(
                "embedding input must contain at least one token".to_owned(),
            ));
        }
        self.embedding
            .embed(input_ids)
            .map(|embedding| embedding.last_token().to_vec())
            .map_err(MlxError::from_qwen)
    }

    /// Continue `state` with a suffix without mutating it.
    pub fn continue_from(
        &self,
        state: &MlxBackboneState,
        suffix_ids: &[u32],
    ) -> Result<(MlxBackboneOutput, MlxBackboneState), MlxError> {
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
        let execution = self.execute(suffix_ids, Some(state), false, None, None)?;
        Ok((execution.output, execution.state))
    }

    pub(super) fn execute(
        &self,
        input_ids: &[u32],
        previous_state: Option<&MlxBackboneState>,
        collect_trace: bool,
        trace_tail_rows: Option<usize>,
        operation_trace_layer: Option<usize>,
    ) -> Result<BackboneExecution, MlxError> {
        if input_ids.is_empty() {
            return Err(MlxError::InvalidState(
                "execution input must contain at least one token".to_owned(),
            ));
        }
        let token_count = input_ids.len();
        let position_start = previous_state.map_or(0, |state| state.position);
        let lineage = previous_state.map_or_else(StateLineage::new_root, |state| state.lineage);

        let embedding = self
            .embedding
            .embed(input_ids)
            .map_err(MlxError::from_qwen)?;
        let embedding_last_token = collect_trace.then(|| embedding.last_token().to_vec());

        let device = self
            .runtime
            .execute(|| -> Result<DeviceExecution, MlxError> {
                let mut hidden = embed_array(embedding.values(), self.precision);
                let mut layer_last_tokens =
                    collect_trace.then(|| Vec::with_capacity(self.layers.len()));
                let mut layer_output_dtypes =
                    collect_trace.then(|| Vec::with_capacity(self.layers.len()));
                let mut operation_traces = collect_trace.then(Vec::new);
                let mut next_layers = Vec::with_capacity(self.layers.len());
                for (layer_index, layer) in self.layers.iter().enumerate() {
                    let previous = previous_state.map(|state| &state.layers[layer_index]);
                    let mut operation_arrays = LayerOperationArrays::new();
                    let collect_layer_operations = collect_trace
                        && trace_tail_rows.is_some()
                        && operation_trace_layer == Some(layer_index);
                    let operation_trace = collect_layer_operations.then_some(&mut operation_arrays);
                    let (next_hidden, next_state) = layer.forward_traced(
                        &hidden,
                        token_count,
                        position_start,
                        previous,
                        operation_trace,
                    )?;
                    hidden = next_hidden;
                    if let Some(trace) = &mut layer_last_tokens {
                        trace.push(last_row_to_host(&hidden)?);
                        layer_output_dtypes
                            .as_mut()
                            .expect("layer dtype trace accompanies vector trace")
                            .push(format!("{:?}", hidden.dtype()));
                        if collect_layer_operations {
                            let tail_length = trace_tail_rows.ok_or_else(|| {
                                MlxError::InvalidState(
                                    "layer operation trace has no requested tail length".to_owned(),
                                )
                            })?;
                            let mut layer_operations = Vec::with_capacity(operation_arrays.len());
                            for (operation, array) in operation_arrays {
                                let tail = tail_rows(&array, tail_length)?;
                                layer_operations.push(MlxOperationTrace {
                                    operation,
                                    dtype: format!("{:?}", tail.dtype()),
                                    shape: tail.shape().to_vec(),
                                    values: array_to_host(&tail)?,
                                });
                            }
                            *operation_traces
                                .as_mut()
                                .expect("layer operation trace accompanies vector trace") =
                                layer_operations;
                        }
                    } else {
                        // Bound the lazy graph without a host readback.
                        hidden.eval().map_err(|error| MlxError::Operation {
                            operation: "decoder layer eval",
                            message: format!("layer {layer_index}: {error}"),
                        })?;
                    }
                    next_layers.push(next_state);
                }
                let final_array =
                    fast::rms_norm(&hidden, Some(&self.final_norm), 1e-6).map_err(|error| {
                        MlxError::Operation {
                            operation: "final rms norm",
                            message: error.to_string(),
                        }
                    })?;
                let final_row = row_of(&final_array, token_count - 1)?;
                let output = MlxBackboneOutput {
                    token_count,
                    final_feature: array_to_host(&final_row)?,
                };
                materialize_state(&next_layers)?;
                Ok(DeviceExecution {
                    output,
                    layers: next_layers,
                    layer_trace: layer_last_tokens
                        .zip(layer_output_dtypes)
                        .zip(operation_traces)
                        .map(|((layers, dtypes), operations)| {
                            (layers, dtypes, operation_trace_layer, operations)
                        }),
                })
            })??;

        Ok(BackboneExecution {
            output: device.output,
            state: MlxBackboneState {
                runtime: SharedMlxRuntime::clone(&self.runtime),
                identity: self.identity.clone(),
                lineage,
                position: position_start + token_count,
                layers: device.layers,
            },
            trace: device.layer_trace.map(|layers| {
                (
                    embedding_last_token.expect("trace embedding accompanies trace layers"),
                    layers.0,
                    layers.1,
                    layers.2,
                    layers.3,
                )
            }),
        })
    }
}
