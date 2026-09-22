//! Shared synthetic executor for the offline nested/batched orchestration
//! tests. Produces the real `BackboneState` shape with token-derived content
//! so features and state fingerprints are path-dependent and deterministic.

use super::layer0::{LayerState, CONV_KERNEL, HEAD_DIM, KV_SIZE, QKV_SIZE, VALUE_HEADS};
use super::model::BackboneState;
use super::nested::{NestedQuestion, SequentialNestedExecutor};
use crate::qwen35::Qwen35Error;
use openkind_runtime::branch::{StateIdentity, StateLineage};

pub(super) const LAYER_COUNT: usize = 32;

pub(super) const ROOT_IDS: &[u32] = &[10, 11, 12];

pub(super) fn plan() -> Vec<NestedQuestion<'static>> {
    vec![
        NestedQuestion {
            question_ids: &[20, 21],
            candidate_suffix_ids: &[&[30], &[31, 32]],
        },
        NestedQuestion {
            question_ids: &[40],
            candidate_suffix_ids: &[&[50, 51]],
        },
    ]
}

pub(super) fn test_identity() -> StateIdentity {
    StateIdentity::new(
        "test-profile",
        "test-model",
        "test-revision",
        "state_first",
        "tokenizer-digest",
        "cpu-fp32",
    )
    .expect("test identity is valid")
}

/// Deterministic token-derived KV content keyed by absolute position.
pub(super) fn token_values(tokens: &[u32], start_position: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(tokens.len() * KV_SIZE);
    for (offset, &token) in tokens.iter().enumerate() {
        for index in 0..KV_SIZE {
            let raw = token as f32 + (index % 17) as f32 + (start_position + offset) as f32;
            values.push(raw / 1_003.0);
        }
    }
    values
}

pub(super) fn synthetic_root(tokens: &[u32]) -> BackboneState {
    let kv = token_values(tokens, 0);
    let layers = (0..LAYER_COUNT)
        .map(|layer_index| {
            if layer_index % 4 == 3 {
                LayerState::Full {
                    keys: kv.clone(),
                    values: kv.clone(),
                }
            } else {
                LayerState::Linear {
                    conv: vec![0.0; QKV_SIZE * CONV_KERNEL],
                    recurrent: vec![0.0; VALUE_HEADS * HEAD_DIM * HEAD_DIM],
                }
            }
        })
        .collect();
    BackboneState {
        identity: test_identity(),
        lineage: StateLineage::new_root(),
        position: tokens.len(),
        layers,
    }
}

pub(super) fn advance(state: &BackboneState, suffix: &[u32]) -> BackboneState {
    let mut next = state.clone();
    let growth = token_values(suffix, state.position());
    for layer in &mut next.layers {
        if let LayerState::Full { keys, values } = layer {
            keys.extend_from_slice(&growth);
            values.extend_from_slice(&growth);
        }
    }
    next.position += suffix.len();
    next
}

pub(super) fn feature(tokens: &[u32], position_after: usize) -> Vec<f32> {
    vec![
        tokens[0] as f32,
        *tokens.last().expect("non-empty tokens") as f32,
        tokens.len() as f32,
        position_after as f32,
    ]
}

/// Executor whose feature depends on the executed tokens and the resulting
/// position, and whose state content depends on every executed token.
pub(super) struct SyntheticExecutor;

impl SequentialNestedExecutor for SyntheticExecutor {
    type State = BackboneState;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        if input_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput("empty prefill".into()));
        }
        let state = synthetic_root(input_ids);
        Ok((feature(input_ids, state.position()), state))
    }

    fn continue_from(
        &self,
        state: &BackboneState,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        if suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput("empty suffix".into()));
        }
        let next = advance(state, suffix_ids);
        Ok((feature(suffix_ids, next.position()), next))
    }
}

/// Executor that reports one position too far, to pin the fail-closed gates.
pub(super) struct DriftingExecutor;

impl SequentialNestedExecutor for DriftingExecutor {
    type State = BackboneState;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        SyntheticExecutor.prefill(input_ids)
    }

    fn continue_from(
        &self,
        state: &BackboneState,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        let (values, mut next) = SyntheticExecutor.continue_from(state, suffix_ids)?;
        next.position += 1;
        Ok((values, next))
    }
}
