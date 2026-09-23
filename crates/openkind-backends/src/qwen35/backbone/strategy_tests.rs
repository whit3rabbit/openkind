//! Offline Phase 3.8 execution-strategy tests.
//!
//! These tests pin the strategy layer's semantics — the repeated-full oracle,
//! cross-strategy feature equality, forward-call accounting, and the measured
//! selection policy — against the shared synthetic executor.
//! Checkpoint-gated measurements run in the `qwen35_scheduler_bench` example.

use super::nested::{NestedQuestion, SequentialNestedExecutor};
use super::strategy::{
    choose_strategy, run_repeated_full, run_strategy, run_with_scheduler, ExecutionStrategy,
    ProcessMemoryEnvelope, SchedulerConfig, StrategyRequest,
};
use super::test_support::{plan, SyntheticExecutor, ROOT_IDS};
use crate::qwen35::Qwen35Error;
use openkind_runtime::{BackendCapabilities, BatchForwardMode};

fn scheduler_config(min_shared_savings_ratio: f64) -> SchedulerConfig {
    SchedulerConfig::for_pinned_profile(min_shared_savings_ratio, None)
}

fn vectorized_scheduler_config(min_shared_savings_ratio: f64) -> SchedulerConfig {
    scheduler_config(min_shared_savings_ratio)
        .with_backend_capabilities(BackendCapabilities::fully_vectorized())
}

#[test]
fn repeated_full_matches_shared_strategies_exactly() {
    let executor = SyntheticExecutor;
    let plans = plan();

    let repeated = run_repeated_full(&executor, ROOT_IDS, &plans).expect("repeated full");
    let sequential = run_strategy(
        &executor,
        ExecutionStrategy::NestedSequential,
        ROOT_IDS,
        &plans,
    )
    .expect("sequential");
    let batched = run_strategy(
        &executor,
        ExecutionStrategy::NestedBatched,
        ROOT_IDS,
        &plans,
    )
    .expect("batched");

    assert_eq!(repeated.strategy(), ExecutionStrategy::RepeatedFull);
    // The synthetic feature reflects only the tokens the executor was handed
    // (suffix for shared paths, full sequence for repeated_full), so compare
    // repeated_full against a direct full-sequence prefill and compare the
    // continuation states across strategies — the real backbone's parity
    // invariant (identical final-token features and states) is exercised by
    // the checkpoint gates.
    for (question_index, plan) in plans.iter().enumerate() {
        for (candidate_index, suffix) in plan.candidate_suffix_ids.iter().enumerate() {
            let mut full_ids = ROOT_IDS.to_vec();
            full_ids.extend_from_slice(plan.question_ids);
            full_ids.extend_from_slice(suffix);
            let (feature, _) = executor.prefill(&full_ids).expect("direct full replay");
            assert_eq!(
                repeated.question_features()[question_index][candidate_index],
                feature
            );
        }
    }
    for ((repeated_question, sequential_question), batched_question) in repeated
        .question_states()
        .iter()
        .zip(sequential.question_states())
        .zip(batched.question_states())
    {
        for ((repeated_state, sequential_state), batched_state) in repeated_question
            .iter()
            .zip(sequential_question)
            .zip(batched_question)
        {
            assert_eq!(
                repeated_state.strict_fingerprint(),
                sequential_state.strict_fingerprint()
            );
            assert_eq!(
                repeated_state.strict_fingerprint(),
                batched_state.strict_fingerprint()
            );
        }
    }
}

#[test]
fn strategy_outputs_account_calls_and_staged_tokens() {
    let executor = SyntheticExecutor;
    let plans = plan();

    let repeated = run_repeated_full(&executor, ROOT_IDS, &plans).expect("repeated full");
    // One prefill per candidate: 3 candidates, each a full sequence.
    assert_eq!(repeated.forward_calls(), 3);
    let repeated_tokens: usize = plans
        .iter()
        .flat_map(|plan| {
            let question_tokens = plan.question_ids.len();
            plan.candidate_suffix_ids
                .iter()
                .map(move |suffix| ROOT_IDS.len() + question_tokens + suffix.len())
        })
        .sum();
    assert_eq!(repeated.staged_tokens(), repeated_tokens);
    assert!(repeated.retained_tensor_bytes() > 0);

    let batched = run_strategy(
        &executor,
        ExecutionStrategy::NestedBatched,
        ROOT_IDS,
        &plans,
    )
    .expect("batched");
    // One prefill + 2 question advances + 3 candidate advances.
    assert_eq!(batched.forward_calls(), 6);
    let shared_tokens: usize = plans
        .iter()
        .map(|plan| {
            plan.question_ids.len()
                + plan
                    .candidate_suffix_ids
                    .iter()
                    .map(|s| s.len())
                    .sum::<usize>()
        })
        .sum();
    assert_eq!(batched.staged_tokens(), shared_tokens);
}

#[test]
fn cpu_backend_prefers_sequential_when_sharing_clearly_wins() {
    let config = scheduler_config(1.5);
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };

    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::NestedSequential);
    assert!(decision.estimates.savings_ratio > 1.5);
    assert!(
        decision.retention.nested_batched_tensor_bytes
            > decision.retention.repeated_full_tensor_bytes
    );
}

#[test]
fn vectorized_backend_can_select_batched() {
    let config = vectorized_scheduler_config(1.5);
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };

    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::NestedBatched);
    assert!(decision.admitted);
}

#[test]
fn choose_strategy_repeated_full_when_sharing_does_not_pay() {
    let config = scheduler_config(3.0);
    let request = StrategyRequest {
        root_tokens: 3,
        question_tokens: vec![2, 1],
        suffix_tokens: vec![vec![1, 2], vec![2]],
    };

    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::RepeatedFull);
    assert!(decision.estimates.savings_ratio < 3.0);
}

#[test]
fn choose_strategy_handles_degenerate_zero_token_request() {
    let config = scheduler_config(1.5);
    let request = StrategyRequest {
        root_tokens: 0,
        question_tokens: Vec::new(),
        suffix_tokens: Vec::new(),
    };

    // No tokens means no shared work to amortize (ratio zero, not a division
    // by zero) and an empty retention footprint, so the scheduler falls back
    // to repeated-full and admits the request.
    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.estimates.shared_tokens, 0);
    assert_eq!(decision.estimates.repeated_tokens, 0);
    assert_eq!(decision.estimates.savings_ratio, 0.0);
    assert_eq!(
        decision.retention.repeated_full_tensor_bytes, 0,
        "nothing is retained for an empty request"
    );
    assert_eq!(decision.strategy, ExecutionStrategy::RepeatedFull);
    assert!(decision.admitted);
}

#[test]
fn choose_strategy_falls_back_through_the_memory_ceiling() {
    let config = vectorized_scheduler_config(1.5);
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };
    let unconstrained = choose_strategy(&config, &request);
    assert_eq!(unconstrained.strategy, ExecutionStrategy::NestedBatched);

    // A ceiling between the sequential and batched estimates selects the
    // sequential shared path.
    let between = (unconstrained.retention.nested_sequential_tensor_bytes
        + unconstrained.retention.nested_batched_tensor_bytes)
        / 2;
    let constrained = choose_strategy(
        &SchedulerConfig::for_pinned_profile(1.5, Some(between))
            .with_backend_capabilities(BackendCapabilities::fully_vectorized()),
        &request,
    );
    assert_eq!(constrained.strategy, ExecutionStrategy::NestedSequential);

    // A ceiling below every estimate reports repeated-full as the last-resort
    // strategy but rejects the request.
    let tight = choose_strategy(
        &SchedulerConfig::for_pinned_profile(1.5, Some(1_000)),
        &request,
    );
    assert_eq!(tight.strategy, ExecutionStrategy::RepeatedFull);
    assert!(!tight.admitted);
}

#[test]
fn repeated_full_admission_accounts_for_every_retained_candidate_state() {
    let request = StrategyRequest {
        root_tokens: 149,
        question_tokens: vec![5; 64],
        suffix_tokens: vec![vec![17, 17]; 64],
    };
    let per_candidate_bytes = 53_477_376 + 65_536 * (149 + 5 + 17);
    let expected_retained_bytes = per_candidate_bytes * 128;
    let config = SchedulerConfig::for_pinned_profile(1.5, Some(80 * 1024 * 1024));

    let decision = choose_strategy(&config, &request);

    assert_eq!(
        decision.retention.repeated_full_tensor_bytes,
        expected_retained_bytes
    );
    assert_eq!(decision.strategy, ExecutionStrategy::RepeatedFull);
    assert!(!decision.admitted);
}

#[test]
fn process_envelope_refreshes_loaded_baseline_and_splits_concurrent_headroom() {
    let envelope = ProcessMemoryEnvelope {
        observed_resident_bytes: 100,
        forward_scratch_bytes: 10,
        allocator_headroom_bytes: 20,
        max_process_bytes: 1_000,
    }
    .for_concurrent_requests(400, 3);

    assert_eq!(envelope.observed_resident_bytes, 400);
    assert_eq!(envelope.max_process_bytes, 600);

    let already_over = ProcessMemoryEnvelope {
        observed_resident_bytes: 100,
        forward_scratch_bytes: 10,
        allocator_headroom_bytes: 20,
        max_process_bytes: 300,
    }
    .for_concurrent_requests(400, 0);
    assert_eq!(already_over.observed_resident_bytes, 400);
    assert_eq!(already_over.max_process_bytes, 400);
}

#[test]
fn run_with_scheduler_executes_the_chosen_strategy() {
    let executor = SyntheticExecutor;
    let config = scheduler_config(1.5);
    let (decision, output) =
        run_with_scheduler(&executor, &config, ROOT_IDS, &plan()).expect("adaptive run");

    assert_eq!(decision.strategy, output.strategy());
    assert_eq!(output.strategy(), ExecutionStrategy::NestedSequential);
    let direct = run_strategy(
        &executor,
        ExecutionStrategy::NestedSequential,
        ROOT_IDS,
        &plan(),
    )
    .expect("direct batched");
    assert_eq!(output.question_features(), direct.question_features());
}

#[test]
fn process_peak_admission_uses_resident_scratch_and_allocator_headroom() {
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };
    let baseline = choose_strategy(&scheduler_config(1.5), &request);
    let repeated_peak = baseline.retention.repeated_full_tensor_bytes + 300_000_000;
    let config = scheduler_config(1.5).with_process_memory(ProcessMemoryEnvelope {
        observed_resident_bytes: 200_000_000,
        forward_scratch_bytes: 75_000_000,
        allocator_headroom_bytes: 25_000_000,
        max_process_bytes: repeated_peak,
    });
    let decision = choose_strategy(&config, &request);

    assert_eq!(decision.strategy, ExecutionStrategy::RepeatedFull);
    assert!(decision.admitted);
    assert_eq!(
        decision.retention.repeated_full_process_peak_bytes,
        Some(repeated_peak)
    );
}

#[test]
fn scheduler_state_stress_covers_high_cardinality_without_allocating_model_state() {
    for candidates in [32, 64, 128, 255] {
        let request = StrategyRequest {
            root_tokens: 256,
            question_tokens: vec![8, 13, 21],
            suffix_tokens: vec![
                vec![7; candidates],
                vec![11; candidates],
                vec![17; candidates],
            ],
        };
        let decision = choose_strategy(&scheduler_config(1.5), &request);
        assert_eq!(decision.strategy, ExecutionStrategy::NestedSequential);
        assert!(decision.admitted);
        assert!(
            decision.retention.nested_batched_tensor_bytes
                > decision.retention.nested_sequential_tensor_bytes
        );
    }
}

#[test]
fn vectorized_lane_limit_falls_back_before_high_k_fanout() {
    let request = StrategyRequest {
        root_tokens: 256,
        question_tokens: vec![8, 13],
        suffix_tokens: vec![vec![7; 64], vec![11; 255]],
    };
    let config = scheduler_config(1.5).with_backend_capabilities(
        BackendCapabilities::fully_vectorized().with_lane_limits(8, 128),
    );
    let decision = choose_strategy(&config, &request);

    assert_eq!(decision.strategy, ExecutionStrategy::NestedSequential);
    assert!(decision.admitted);
}

#[test]
fn repeated_full_rejects_plans_without_candidates() {
    let executor = SyntheticExecutor;
    let empty = [NestedQuestion {
        question_ids: &[20],
        candidate_suffix_ids: &[],
    }];
    assert!(matches!(
        run_repeated_full(&executor, ROOT_IDS, &empty),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("no candidate suffixes")
    ));
}

#[test]
fn forced_strategy_bypasses_profitability_but_not_admission() {
    // A 3.0 crossover would never share for this shape; the diagnostic
    // override forces the shared sequential plan through anyway.
    let config =
        scheduler_config(3.0).with_forced_strategy(Some(ExecutionStrategy::NestedSequential));
    let request = StrategyRequest {
        root_tokens: 3,
        question_tokens: vec![2, 1],
        suffix_tokens: vec![vec![1, 2], vec![2]],
    };

    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::NestedSequential);
    assert!(decision.forced);
    assert!(decision.admitted);
    assert_eq!(decision.batch_forward_mode, BatchForwardMode::PerLane);
    assert!(decision.rationale.contains("forced override"));
    assert!(decision.rationale.contains("admission=true"));
}

#[test]
fn forced_nested_batched_records_the_physical_forward_mode() {
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };

    // Forcing the batched topology on a per-lane CPU backend is legal (it is
    // state topology; compute advances lane by lane) and must be recorded as
    // per-lane so it can never be mistaken for a vectorized run.
    let per_lane =
        scheduler_config(1.5).with_forced_strategy(Some(ExecutionStrategy::NestedBatched));
    let decision = choose_strategy(&per_lane, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::NestedBatched);
    assert!(decision.forced);
    assert_eq!(decision.batch_forward_mode, BatchForwardMode::PerLane);

    // The same override on a vectorized backend records the vectorized mode.
    let vectorized = vectorized_scheduler_config(1.5)
        .with_forced_strategy(Some(ExecutionStrategy::NestedBatched));
    let decision = choose_strategy(&vectorized, &request);
    assert_eq!(decision.batch_forward_mode, BatchForwardMode::Vectorized);

    // Auto decisions carry the same field: the per-lane CPU default selects
    // sequential and executes per-lane.
    let auto = choose_strategy(&scheduler_config(1.5), &request);
    assert!(!auto.forced);
    assert_eq!(auto.strategy, ExecutionStrategy::NestedSequential);
    assert_eq!(auto.batch_forward_mode, BatchForwardMode::PerLane);
}

#[test]
fn vectorized_mode_requires_every_nested_fanout_to_have_multiple_lanes() {
    let config = vectorized_scheduler_config(1.5)
        .with_forced_strategy(Some(ExecutionStrategy::NestedBatched));
    let auto_config = vectorized_scheduler_config(1.5);
    let one_question = StrategyRequest {
        root_tokens: 32,
        question_tokens: vec![8],
        suffix_tokens: vec![vec![4, 5]],
    };
    let one_candidate_fanout = StrategyRequest {
        root_tokens: 32,
        question_tokens: vec![8, 8],
        suffix_tokens: vec![vec![4, 5], vec![6]],
    };
    let unequal_question_lengths = StrategyRequest {
        root_tokens: 32,
        question_tokens: vec![8, 9],
        suffix_tokens: vec![vec![4, 5], vec![6, 7]],
    };
    let fully_batched = StrategyRequest {
        root_tokens: 32,
        question_tokens: vec![8, 8],
        suffix_tokens: vec![vec![4, 5], vec![6, 7]],
    };

    assert_eq!(
        choose_strategy(&config, &one_question).batch_forward_mode,
        BatchForwardMode::PerLane
    );
    assert_eq!(
        choose_strategy(&config, &one_candidate_fanout).batch_forward_mode,
        BatchForwardMode::PerLane
    );
    assert_eq!(
        choose_strategy(&config, &unequal_question_lengths).batch_forward_mode,
        BatchForwardMode::Vectorized
    );
    assert_eq!(
        choose_strategy(&config, &fully_batched).batch_forward_mode,
        BatchForwardMode::Vectorized
    );
    assert_eq!(
        choose_strategy(&auto_config, &one_question).strategy,
        ExecutionStrategy::NestedSequential
    );
    assert_eq!(
        choose_strategy(&auto_config, &one_candidate_fanout).strategy,
        ExecutionStrategy::NestedSequential
    );
    assert_eq!(
        choose_strategy(&auto_config, &unequal_question_lengths).strategy,
        ExecutionStrategy::NestedBatched
    );
    assert_eq!(
        choose_strategy(&auto_config, &fully_batched).strategy,
        ExecutionStrategy::NestedBatched
    );
}

#[test]
fn forced_strategy_still_fails_closed_on_admission() {
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };
    let config = SchedulerConfig::for_pinned_profile(1.5, Some(1_000))
        .with_forced_strategy(Some(ExecutionStrategy::NestedSequential));

    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::NestedSequential);
    assert!(!decision.admitted);
    assert!(decision.rationale.contains("admission=false"));

    let executor = SyntheticExecutor;
    assert!(matches!(
        run_with_scheduler(&executor, &config, ROOT_IDS, &plan()),
        Err(Qwen35Error::InvalidInput(message)) if message.contains("forced override")
    ));
}

#[test]
fn forced_strategy_executes_and_matches_a_direct_run() {
    let executor = SyntheticExecutor;
    let config = scheduler_config(1.5).with_forced_strategy(Some(ExecutionStrategy::NestedBatched));

    let (decision, output) =
        run_with_scheduler(&executor, &config, ROOT_IDS, &plan()).expect("forced run");
    assert_eq!(decision.strategy, ExecutionStrategy::NestedBatched);
    assert_eq!(output.strategy(), ExecutionStrategy::NestedBatched);

    let direct = run_strategy(
        &executor,
        ExecutionStrategy::NestedBatched,
        ROOT_IDS,
        &plan(),
    )
    .expect("direct batched");
    assert_eq!(output.question_features(), direct.question_features());
}
