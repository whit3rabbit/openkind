//! Offline Phase 3.5 sequential nested execution tests.
//!
//! These tests exercise the nested orchestration semantics — prefill once,
//! fork question, advance, fork candidate, advance, immutable sources,
//! deterministic replay — against a synthetic executor that produces the real
//! `BackboneState` shape with token-derived content. Checkpoint-gated native
//! parity runs in the `qwen35_nested_parity` example.

use super::layer0::{LayerState, CONV_KERNEL, HEAD_DIM, KV_SIZE, QKV_SIZE, VALUE_HEADS};
use super::model::BackboneState;
use super::nested::{run_sequential_nested, NestedQuestion, SequentialNestedExecutor};
use crate::qwen35::Qwen35Error;

const LAYER_COUNT: usize = 32;

const ROOT_IDS: &[u32] = &[10, 11, 12];

fn plan() -> Vec<NestedQuestion<'static>> {
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

fn test_identity() -> crate::branch::StateIdentity {
    crate::branch::StateIdentity::new(
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
fn token_values(tokens: &[u32], start_position: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(tokens.len() * KV_SIZE);
    for (offset, &token) in tokens.iter().enumerate() {
        for index in 0..KV_SIZE {
            let raw = token as f32 + (index % 17) as f32 + (start_position + offset) as f32;
            values.push(raw / 1_003.0);
        }
    }
    values
}

fn synthetic_root(tokens: &[u32]) -> BackboneState {
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
        lineage: crate::branch::StateLineage::new_root(),
        position: tokens.len(),
        layers,
    }
}

fn advance(state: &BackboneState, suffix: &[u32]) -> BackboneState {
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

fn feature(tokens: &[u32], position_after: usize) -> Vec<f32> {
    vec![
        tokens[0] as f32,
        *tokens.last().expect("non-empty tokens") as f32,
        tokens.len() as f32,
        position_after as f32,
    ]
}

/// Executor whose feature depends on the executed tokens and the resulting
/// position, and whose state content depends on every executed token.
struct SyntheticExecutor;

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

/// Executor that reports one position too far, to pin the fail-closed gate.
struct DriftingExecutor;

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

#[test]
fn nested_run_matches_direct_replay_and_keeps_sources_immutable() {
    let executor = SyntheticExecutor;
    let run = run_sequential_nested(&executor, ROOT_IDS, &plan()).expect("nested run");

    assert_eq!(run.root_feature(), feature(ROOT_IDS, ROOT_IDS.len()));
    assert_eq!(run.root_state().position(), ROOT_IDS.len());
    assert_eq!(run.questions().len(), 2);

    // Independent single-stage replay of every question and candidate.
    let (_, fresh_root) = executor.prefill(ROOT_IDS).expect("prefill");
    assert_eq!(
        run.root_state().strict_fingerprint(),
        fresh_root.strict_fingerprint(),
        "the run's root holds exactly the prefilled content"
    );

    let (_, question_one_state) = executor
        .continue_from(&fresh_root, &[20, 21])
        .expect("question replay");
    let question_one = &run.questions()[0];
    assert_eq!(
        question_one.question_feature(),
        feature(&[20, 21], ROOT_IDS.len() + 2)
    );
    assert_eq!(
        question_one.question_state().strict_fingerprint(),
        question_one_state.strict_fingerprint()
    );

    let (candidate_zero, candidate_zero_state) = executor
        .continue_from(&question_one_state, &[30])
        .expect("candidate replay");
    let (candidate_one, candidate_one_state) = executor
        .continue_from(&question_one_state, &[31, 32])
        .expect("candidate replay");
    assert_eq!(question_one.candidates()[0].feature(), candidate_zero);
    assert_eq!(question_one.candidates()[1].feature(), candidate_one);
    assert_eq!(
        question_one.candidates()[0].state().strict_fingerprint(),
        candidate_zero_state.strict_fingerprint()
    );
    assert_eq!(
        question_one.candidates()[1].state().strict_fingerprint(),
        candidate_one_state.strict_fingerprint()
    );

    // Question two replaying from the same root proves question one's
    // candidate work never disturbed the shared root.
    let (_, question_two_state) = executor
        .continue_from(&fresh_root, &[40])
        .expect("question replay");
    let question_two = &run.questions()[1];
    assert_eq!(
        question_two.question_feature(),
        feature(&[40], ROOT_IDS.len() + 1)
    );
    assert_eq!(
        question_two.question_state().strict_fingerprint(),
        question_two_state.strict_fingerprint()
    );
    assert_eq!(
        question_two.candidates()[0].feature(),
        feature(&[50, 51], ROOT_IDS.len() + 3)
    );
}

#[test]
fn nested_run_positions_track_root_question_and_candidate_lengths() {
    let run = run_sequential_nested(&SyntheticExecutor, ROOT_IDS, &plan()).expect("nested run");

    assert_eq!(run.root_state().position(), 3);
    assert_eq!(run.questions()[0].question_state().position(), 5);
    assert_eq!(run.questions()[0].candidates()[0].state().position(), 6);
    assert_eq!(run.questions()[0].candidates()[1].state().position(), 7);
    assert_eq!(run.questions()[1].question_state().position(), 4);
    assert_eq!(run.questions()[1].candidates()[0].state().position(), 6);
}

#[test]
fn nested_run_rejects_empty_plans_and_suffixes() {
    let executor = SyntheticExecutor;
    assert!(matches!(
        run_sequential_nested(&executor, &[], &plan()),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("non-empty root")
    ));
    assert!(matches!(
        run_sequential_nested(&executor, ROOT_IDS, &[]),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("at least one question")
    ));

    let empty_question_ids = [NestedQuestion {
        question_ids: &[],
        candidate_suffix_ids: &[&[30]],
    }];
    assert!(matches!(
        run_sequential_nested(&executor, ROOT_IDS, &empty_question_ids),
        Err(Qwen35Error::InvalidInput(message))
            if message.contains("question 0 has an empty token suffix")
    ));

    let no_candidates = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &[],
    }];
    assert!(matches!(
        run_sequential_nested(&executor, ROOT_IDS, &no_candidates),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("no candidate suffixes")
    ));

    let empty_candidate = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &[&[]],
    }];
    assert!(matches!(
        run_sequential_nested(&executor, ROOT_IDS, &empty_candidate),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("empty suffix")
    ));
}

#[test]
fn nested_run_candidate_features_follow_suffixes_regardless_of_plan_order() {
    let executor = SyntheticExecutor;
    let forward = run_sequential_nested(&executor, ROOT_IDS, &plan()).expect("forward plan");
    let reversed_plan = vec![NestedQuestion {
        question_ids: &[20, 21],
        candidate_suffix_ids: &[&[31, 32], &[30]],
    }];
    let reversed = run_sequential_nested(&executor, ROOT_IDS, &reversed_plan)
        .expect("reversed candidate order");

    let forward_question = &forward.questions()[0];
    let reversed_question = &reversed.questions()[0];
    assert_eq!(
        forward_question.candidates()[0].feature(),
        reversed_question.candidates()[1].feature(),
        "candidate 0 keeps its feature under a different sibling order"
    );
    assert_eq!(
        forward_question.candidates()[1].feature(),
        reversed_question.candidates()[0].feature(),
        "candidate 1 keeps its feature under a different sibling order"
    );
    assert_eq!(
        forward_question.candidates()[0]
            .state()
            .strict_fingerprint(),
        reversed_question.candidates()[1]
            .state()
            .strict_fingerprint()
    );
}

#[test]
fn nested_run_replays_deterministically_across_fresh_executors() {
    let first = run_sequential_nested(&SyntheticExecutor, ROOT_IDS, &plan()).expect("first run");
    let second = run_sequential_nested(&SyntheticExecutor, ROOT_IDS, &plan()).expect("second run");

    assert_eq!(first.root_feature(), second.root_feature());
    for (first_question, second_question) in first.questions().iter().zip(second.questions()) {
        assert_eq!(
            first_question.question_feature(),
            second_question.question_feature()
        );
        assert_eq!(
            first_question.question_state().strict_fingerprint(),
            second_question.question_state().strict_fingerprint()
        );
        for (first_candidate, second_candidate) in first_question
            .candidates()
            .iter()
            .zip(second_question.candidates())
        {
            assert_eq!(first_candidate.feature(), second_candidate.feature());
            assert_eq!(
                first_candidate.state().strict_fingerprint(),
                second_candidate.state().strict_fingerprint()
            );
        }
    }
}

#[test]
fn nested_run_fail_closed_on_executor_position_drift() {
    assert!(matches!(
        run_sequential_nested(&DriftingExecutor, ROOT_IDS, &plan()),
        Err(Qwen35Error::InvalidInput(message))
            if message.contains("question 0 advanced to position 6, expected 5")
    ));
}
