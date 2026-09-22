//! Offline Phase 3.5 sequential nested execution tests.
//!
//! These tests exercise the nested orchestration semantics — prefill once,
//! fork question, advance, fork candidate, advance, immutable sources,
//! deterministic replay — against the shared synthetic executor, which
//! produces the real `BackboneState` shape with token-derived content.
//! Checkpoint-gated native parity runs in the `qwen35_nested_parity` example.

use super::nested::{run_sequential_nested, NestedQuestion, SequentialNestedExecutor};
use super::test_support::{feature, plan, DriftingExecutor, SyntheticExecutor, ROOT_IDS};
use crate::qwen35::Qwen35Error;

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
