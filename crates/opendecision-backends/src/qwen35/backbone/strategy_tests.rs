//! Offline Phase 3.8 execution-strategy tests.
//!
//! These tests pin the strategy layer's semantics — the repeated-full oracle,
//! cross-strategy feature equality, forward-call accounting, and the measured
//! selection policy — against the shared synthetic executor.
//! Checkpoint-gated measurements run in the `qwen35_scheduler_bench` example.

use super::nested::{NestedQuestion, SequentialNestedExecutor};
use super::strategy::{
    choose_strategy, run_repeated_full, run_strategy, run_with_scheduler, ExecutionStrategy,
    SchedulerConfig, StrategyRequest,
};
use super::test_support::{plan, SyntheticExecutor, ROOT_IDS};
use crate::qwen35::Qwen35Error;

fn scheduler_config(min_shared_savings_ratio: f64) -> SchedulerConfig {
    SchedulerConfig::for_pinned_profile(min_shared_savings_ratio, None)
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
    assert!(repeated.retained_state_bytes() > 0);

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
fn choose_strategy_prefers_batched_when_sharing_clearly_wins() {
    let config = scheduler_config(1.5);
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };

    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::NestedBatched);
    assert!(decision.estimates.savings_ratio > 1.5);
    assert!(decision.retention.nested_batched_bytes > decision.retention.repeated_full_bytes);
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
    assert_eq!(
        decision.retention.repeated_full_bytes,
        request
            .question_tokens
            .iter()
            .zip(&request.suffix_tokens)
            .flat_map(|(&question, suffixes)| suffixes.iter().map(move |&suffix| {
                config.state_fixed_bytes
                    + config.state_bytes_per_token * (request.root_tokens + question + suffix)
            }))
            .sum()
    );
}

#[test]
fn choose_strategy_falls_back_through_the_memory_ceiling() {
    let config = scheduler_config(1.5);
    let request = StrategyRequest {
        root_tokens: 512,
        question_tokens: vec![10, 10],
        suffix_tokens: vec![vec![16, 16], vec![16, 16]],
    };
    let unconstrained = choose_strategy(&config, &request);
    assert_eq!(unconstrained.strategy, ExecutionStrategy::NestedBatched);

    // A ceiling between the sequential and batched estimates selects the
    // sequential shared path.
    let between = (unconstrained.retention.nested_sequential_bytes
        + unconstrained.retention.nested_batched_bytes)
        / 2;
    let constrained = choose_strategy(
        &SchedulerConfig::for_pinned_profile(1.5, Some(between)),
        &request,
    );
    assert_eq!(constrained.strategy, ExecutionStrategy::NestedSequential);

    // A ceiling below every estimate leaves repeated-full as the decision;
    // run_with_scheduler rejects that decision before execution.
    let tight = choose_strategy(
        &SchedulerConfig::for_pinned_profile(1.5, Some(1_000)),
        &request,
    );
    assert_eq!(tight.strategy, ExecutionStrategy::RepeatedFull);
}

#[test]
fn run_with_scheduler_executes_the_chosen_strategy() {
    let executor = SyntheticExecutor;
    let config = scheduler_config(1.5);
    let (decision, output) =
        run_with_scheduler(&executor, &config, ROOT_IDS, &plan()).expect("adaptive run");

    assert_eq!(decision.strategy, output.strategy());
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

#[test]
fn run_with_scheduler_rejects_when_aggregate_repeated_state_exceeds_ceiling() {
    let executor = SyntheticExecutor;
    let plans = plan();
    let config = SchedulerConfig::for_pinned_profile(10.0, Some(100_000_000));

    let request = StrategyRequest::from_plans(ROOT_IDS.len(), &plans);
    let decision = choose_strategy(&config, &request);
    assert_eq!(decision.strategy, ExecutionStrategy::RepeatedFull);
    assert!(decision.retention.repeated_full_bytes > 100_000_000);

    assert!(matches!(
        run_with_scheduler(&executor, &config, ROOT_IDS, &plans),
        Err(Qwen35Error::InvalidInput(message))
            if message.contains("no execution strategy fits the state-memory ceiling")
    ));
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
