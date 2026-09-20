//! Offline Phase 3.6/3.7 breadth-first batched execution tests.
//!
//! These tests pin the batched orchestration semantics — one root, `Q`
//! isolated question lanes, `K` isolated candidate lanes per question,
//! immutable sources, exact equivalence with the sequential nested baseline,
//! and gather/reorder equivalence — against the shared synthetic executor.
//! Checkpoint-gated native parity runs in the `qwen35_batched_parity`
//! example.

use super::batched::{run_batched_candidates, run_batched_nested, run_batched_questions};
use super::nested::{run_sequential_nested, SequentialNestedExecutor};
use super::test_support::{advance, plan, DriftingExecutor, SyntheticExecutor, ROOT_IDS};
use crate::qwen35::Qwen35Error;
use opendecision_runtime::branch::{BranchBatch, BranchableState};

#[test]
fn batched_questions_match_direct_forks_and_keep_sources_immutable() {
    let executor = SyntheticExecutor;
    let (_, root) = executor.prefill(ROOT_IDS).expect("prefill");
    let root_fingerprint = root.scheduling_fingerprint();
    let root_strict = root.strict_fingerprint();

    let batched = run_batched_questions(&executor, &root, &plan()).expect("batched questions");

    assert_eq!(batched.batch_bytes(), 2 * root.tensor_storage_bytes());
    assert_eq!(
        root.scheduling_fingerprint(),
        root_fingerprint,
        "root unchanged by fan-out"
    );
    assert_eq!(
        root.strict_fingerprint(),
        root_strict,
        "root content unchanged"
    );

    for (index, plan) in plan().iter().enumerate() {
        let (expected_feature, expected_state) = executor
            .continue_from(&root, plan.question_ids)
            .expect("direct fork replay");
        assert_eq!(&batched.question_features()[index], &expected_feature);
        assert_eq!(
            batched.question_states()[index].position(),
            expected_state.position()
        );
        // Fork depth is excluded from the structural fingerprint, so a
        // batch lane is scheduling-equivalent to a single fork of the same
        // root at the same position.
        assert_eq!(
            batched.question_states()[index].scheduling_fingerprint(),
            expected_state.scheduling_fingerprint()
        );
        assert_eq!(
            batched.question_states()[index].strict_fingerprint(),
            expected_state.strict_fingerprint()
        );
    }
    assert_eq!(
        root.scheduling_fingerprint(),
        root_fingerprint,
        "root unchanged after all lanes"
    );
}

#[test]
fn batched_nested_run_matches_the_sequential_baseline_exactly() {
    let executor = SyntheticExecutor;
    let sequential = run_sequential_nested(&executor, ROOT_IDS, &plan()).expect("sequential");
    let batched = run_batched_nested(&executor, ROOT_IDS, &plan()).expect("batched");

    assert_eq!(batched.root_feature(), sequential.root_feature());
    assert_eq!(
        batched.question_batch_bytes(),
        2 * sequential.root_state().tensor_storage_bytes()
    );

    for (sequential_question, batched_question) in
        sequential.questions().iter().zip(batched.questions())
    {
        assert_eq!(
            batched_question.question_feature(),
            sequential_question.question_feature()
        );
        assert_eq!(
            batched_question.question_state().position(),
            sequential_question.question_state().position()
        );
        // The runs prefill separate roots, so process-local lineage differs
        // while content and position stay identical.
        assert_ne!(
            batched_question.question_state().scheduling_fingerprint(),
            sequential_question
                .question_state()
                .scheduling_fingerprint()
        );
        assert_eq!(
            batched_question.question_state().strict_fingerprint(),
            sequential_question.question_state().strict_fingerprint()
        );
        for (sequential_candidate, batched_candidate) in sequential_question
            .candidates()
            .iter()
            .zip(batched_question.candidates())
        {
            assert_eq!(batched_candidate.feature(), sequential_candidate.feature());
            assert_eq!(
                batched_candidate.state().strict_fingerprint(),
                sequential_candidate.state().strict_fingerprint()
            );
        }
    }
}

#[test]
fn batched_questions_compose_from_a_foreign_sequential_root() {
    let executor = SyntheticExecutor;
    let sequential = run_sequential_nested(&executor, ROOT_IDS, &plan()).expect("sequential");

    // Fan out from the sequential run's own root: lineage is then shared, so
    // even the structural fingerprints must match the sequential states.
    let batched =
        run_batched_questions(&executor, sequential.root_state(), &plan()).expect("batched");
    for (sequential_question, batched_state) in
        sequential.questions().iter().zip(batched.question_states())
    {
        assert_eq!(
            batched_state.scheduling_fingerprint(),
            sequential_question
                .question_state()
                .scheduling_fingerprint()
        );
        assert_eq!(
            batched_state.strict_fingerprint(),
            sequential_question.question_state().strict_fingerprint()
        );
    }
}

#[test]
fn batched_candidates_follow_suffix_order_and_support_gather_reordering() {
    let executor = SyntheticExecutor;
    let (_, root) = executor.prefill(ROOT_IDS).expect("prefill");
    let (_, question_state) = executor.continue_from(&root, &[20, 21]).expect("question");
    let question_fingerprint = question_state.scheduling_fingerprint();

    let suffix_a: &[u32] = &[30];
    let suffix_b: &[u32] = &[31, 32];
    let suffix_c: &[u32] = &[32, 33, 34];
    let suffixes: &[&[u32]] = &[suffix_a, suffix_b, suffix_c];

    let forward = run_batched_candidates(&executor, &question_state, suffixes)
        .expect("forward candidate fan-out");
    let reversed_suffixes: &[&[u32]] = &[suffix_c, suffix_b, suffix_a];
    let reversed = run_batched_candidates(&executor, &question_state, reversed_suffixes)
        .expect("reversed candidate fan-out");

    assert_eq!(
        forward.batch_bytes(),
        3 * question_state.tensor_storage_bytes()
    );
    assert_eq!(
        forward.candidate_features()[0],
        reversed.candidate_features()[2]
    );
    assert_eq!(
        forward.candidate_features()[1],
        reversed.candidate_features()[1]
    );
    assert_eq!(
        forward.candidate_features()[2],
        reversed.candidate_features()[0]
    );
    assert_eq!(
        forward.candidate_states()[0].strict_fingerprint(),
        reversed.candidate_states()[2].strict_fingerprint()
    );
    assert_eq!(
        question_state.scheduling_fingerprint(),
        question_fingerprint,
        "question state unchanged by both fan-outs"
    );

    // Gathering unadvanced lanes then advancing equals advancing in order:
    // the reorder pattern length-bucketed execution uses.
    let (feature_after_advance, state_after_advance) = executor
        .continue_from(&question_state, suffix_c)
        .expect("direct advance");
    let batch = question_state.fork_batch(3).expect("fan-out");
    let gathered = batch.gather(&[2, 0, 1]).expect("gather reorder");
    let (feature_from_gathered, state_from_gathered) = executor
        .continue_from(&gathered.select(0).expect("lane"), suffix_c)
        .expect("advance gathered lane");
    assert_eq!(feature_from_gathered, feature_after_advance);
    assert_eq!(
        state_from_gathered.scheduling_fingerprint(),
        state_after_advance.scheduling_fingerprint()
    );
    assert_eq!(
        state_from_gathered.strict_fingerprint(),
        advance(&question_state, suffix_c).strict_fingerprint()
    );
}

#[test]
fn batched_run_positions_and_candidate_batches_track_plan_shape() {
    let batched = run_batched_nested(&SyntheticExecutor, ROOT_IDS, &plan()).expect("batched run");

    assert_eq!(batched.root_state().position(), 3);
    assert_eq!(batched.questions().len(), 2);
    assert_eq!(batched.questions()[0].question_state().position(), 5);
    assert_eq!(batched.questions()[0].candidates()[0].state().position(), 6);
    assert_eq!(batched.questions()[0].candidates()[1].state().position(), 7);
    assert_eq!(batched.questions()[1].question_state().position(), 4);
    assert_eq!(batched.questions()[1].candidates()[0].state().position(), 6);
    assert_eq!(
        batched.questions()[0].candidate_batch_bytes(),
        2 * batched.questions()[0]
            .question_state()
            .tensor_storage_bytes()
    );
}

#[test]
fn batched_run_rejects_empty_plans_and_suffixes() {
    let executor = SyntheticExecutor;
    let (_, root) = executor.prefill(ROOT_IDS).expect("prefill");
    let (_, question_state) = executor.continue_from(&root, &[20, 21]).expect("question");

    assert!(matches!(
        run_batched_nested(&executor, &[], &plan()),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("non-empty root")
    ));
    assert!(matches!(
        run_batched_questions(&executor, &root, &[]),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("at least one question")
    ));
    assert!(matches!(
        run_batched_candidates(&executor, &question_state, &[]),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("at least one candidate suffix")
    ));

    let empty_question_ids = [super::nested::NestedQuestion {
        question_ids: &[],
        candidate_suffix_ids: &[&[30]],
    }];
    assert!(matches!(
        run_batched_questions(&executor, &root, &empty_question_ids),
        Err(Qwen35Error::InvalidInput(message))
            if message.contains("question 0 has an empty token suffix")
    ));

    let empty_suffix: &[&[u32]] = &[&[]];
    assert!(matches!(
        run_batched_candidates(&executor, &question_state, empty_suffix),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("candidate 0 has an empty suffix")
    ));
}

#[test]
fn batched_run_fail_closed_on_executor_position_drift() {
    assert!(matches!(
        run_batched_nested(&DriftingExecutor, ROOT_IDS, &plan()),
        Err(Qwen35Error::InvalidInput(message))
            if message.contains("question 0 advanced to position 6, expected 5")
    ));
}
