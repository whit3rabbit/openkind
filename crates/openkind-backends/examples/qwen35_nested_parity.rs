//! Phase 3.5 native gate: sequential nested `state → question → candidate`
//! execution parity for the pinned Qwen3.5 profile.
//!
//! For every exported fixture case the shared state is prefilled exactly once,
//! then each question and candidate runs through immutable `BranchableState`
//! forks. Declared gates: head probability parity against the golden fixtures
//! (`0.005`), zero argmax changes, zero policy changes, exact root-state
//! content identity against an independent prefill, exact question and
//! candidate replay from the retained fork states, candidate-sibling order
//! independence, and cached-versus-`repeated_full` agreement within the Phase
//! 3B `1e-4` self-consistency guard. Hidden-vector deltas are reported as
//! localization diagnostics, not new tolerances.
//!
//! Usage: `qwen35_nested_parity <checkpoint-root> <phase3b-reference-root>
//! <head-bundle-root>`. The run executes the real checkpoint on CPU and is
//! not part of `cargo test`.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::branch::BranchableState;
use openkind_backends::qwen35::{
    BackboneReference, NestedQuestion, PolicyAction, PrimitiveKind, Qwen35Backbone,
    ReferenceBundle, ORDERING_TOLERANCE, POLICY_THRESHOLD, PROBABILITY_TOLERANCE,
};
use serde::Deserialize;

/// Phase 3B's declared cached-versus-full self-consistency guard.
const CACHED_VS_FULL_GUARD: f64 = 0.000_1;

#[derive(Debug, Deserialize)]
struct TokenFixtures {
    records: Vec<TokenRecord>,
}

#[derive(Debug, Deserialize)]
struct TokenRecord {
    fixture_case: usize,
    question_id: String,
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<Vec<u32>>,
    full_candidate_ids: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    request: GoldenRequest,
    head_and_token_fixtures: Vec<HeadFixture>,
}

#[derive(Debug, Deserialize)]
struct GoldenRequest {
    questions: Vec<GoldenQuestion>,
}

#[derive(Debug, Deserialize)]
struct GoldenQuestion {
    id: String,
    primitive: String,
    options: Vec<GoldenOption>,
}

#[derive(Debug, Deserialize)]
struct GoldenOption {
    id: String,
}

#[derive(Debug, Deserialize)]
struct HeadFixture {
    question_id: String,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct ProbabilityReference {
    records: Vec<ProbabilityRecord>,
}

#[derive(Debug, Deserialize)]
struct ProbabilityRecord {
    fixture_case: usize,
    answers: Vec<ExpectedAnswer>,
}

#[derive(Debug, Deserialize)]
struct ExpectedAnswer {
    question_id: String,
    selected_id: String,
    top_probability: f64,
}

#[derive(Debug, Deserialize)]
struct ContinuationTrace {
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<u32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = required_path(&mut arguments, "checkpoint-root")?;
    let reference_root = required_path(&mut arguments, "phase3b-reference-root")?;
    let head_bundle_root = required_path(&mut arguments, "head-bundle-root")?;
    if arguments.next().is_some() {
        return Err("qwen35_nested_parity accepts exactly three paths".into());
    }

    let reference = BackboneReference::load(&reference_root)?;
    let backbone = Qwen35Backbone::load(checkpoint_root)?;
    let bundle = ReferenceBundle::load(&head_bundle_root)?;
    let token_fixtures: TokenFixtures = read_json(&reference_root.join("TOKEN_FIXTURES.json"))?;
    let golden_cases: Vec<GoldenCase> = read_json(&head_bundle_root.join("golden.json"))?;
    let probability_reference: ProbabilityReference =
        read_json(&reference_root.join("PROBABILITY_REFERENCE.json"))?;
    let trace: ContinuationTrace =
        serde_json::from_slice(&fs::read(reference_root.join("CONTINUATION_TRACE.json"))?)?;

    let expected_answers: HashMap<_, _> = probability_reference
        .records
        .iter()
        .flat_map(|record| {
            record
                .answers
                .iter()
                .map(move |answer| ((record.fixture_case, answer.question_id.as_str()), answer))
        })
        .collect();

    let mut grouped: BTreeMap<usize, Vec<&TokenRecord>> = BTreeMap::new();
    for record in &token_fixtures.records {
        grouped.entry(record.fixture_case).or_default().push(record);
    }

    let mut maximum_probability_delta = 0.0_f64;
    let mut maximum_logit_delta = 0.0_f64;
    let mut maximum_feature_delta_vs_reference = 0.0_f64;
    let mut maximum_cached_vs_full_feature_delta = 0.0_f64;
    let mut maximum_nested_vs_full_probability_delta = 0.0_f64;
    let mut argmax_changes = 0_usize;
    let mut policy_changes = 0_usize;
    let mut candidate_count = 0_usize;
    let mut root_immutable_all_cases = true;
    let mut replay_exact_all_records = true;
    let mut cached_state_equals_full_state_all = true;
    let mut positions_match_fixtures = true;
    let mut case_results = Vec::new();

    for (&fixture_case, records) in &grouped {
        let root_ids = &records[0].root_ids;
        if records.iter().any(|record| record.root_ids != *root_ids) {
            return Err(format!("fixture case {fixture_case} records disagree on root_ids").into());
        }

        // Plan the nested run: one root prefill, per-record question forks,
        // per-candidate forks of each question state. The per-record suffix
        // storage must be pushed before the plan borrows it.
        let mut suffix_store = Vec::with_capacity(records.len());
        for record in records {
            let suffixes: Vec<&[u32]> = record
                .candidate_suffix_ids
                .iter()
                .map(Vec::as_slice)
                .collect();
            suffix_store.push(suffixes);
        }
        let plans: Vec<NestedQuestion> = records
            .iter()
            .zip(&suffix_store)
            .map(|(record, suffixes)| NestedQuestion {
                question_ids: record.question_ids.as_slice(),
                candidate_suffix_ids: suffixes.as_slice(),
            })
            .collect();
        let run = backbone.evaluate_nested(root_ids, &plans)?;

        // Root immutability: the retained root must hold exactly the content
        // of an independent prefill of the same tokens.
        let (fresh_root_output, fresh_root_state) = backbone.prefill(root_ids)?;
        let root_feature_delta = maximum_abs(run.root_feature(), fresh_root_output.final_token());
        let root_content_identical =
            run.root_state().strict_fingerprint() == fresh_root_state.strict_fingerprint();
        root_immutable_all_cases &= root_feature_delta == 0.0 && root_content_identical;
        positions_match_fixtures &= run.root_state().position() == root_ids.len();

        let root_trace_matches = fixture_case == 0 && root_ids == &trace.root_ids;
        let mut trace_root_delta = None;
        if root_trace_matches {
            let comparison =
                reference.compare("continuation.root_last_hidden", run.root_feature())?;
            trace_root_delta = Some(comparison.max_abs());
        }

        let mut record_results = Vec::with_capacity(records.len());
        for (record, question) in records.iter().zip(run.questions()) {
            let golden_case = golden_cases
                .get(record.fixture_case)
                .ok_or("fixture case is out of range")?;
            let golden_question = golden_case
                .request
                .questions
                .iter()
                .find(|question| question.id == record.question_id)
                .ok_or("question is missing from golden case")?;
            let head_fixture = golden_case
                .head_and_token_fixtures
                .iter()
                .find(|fixture| fixture.question_id == record.question_id)
                .ok_or("head fixture is missing from golden case")?;
            let expected = expected_answers
                .get(&(record.fixture_case, record.question_id.as_str()))
                .ok_or("probability answer is missing")?;

            positions_match_fixtures &=
                question.question_state().position() == root_ids.len() + record.question_ids.len();
            let trace_question_matches = record.question_ids == trace.question_ids;

            // Question replay from the retained root fork must be exact.
            let question_branch = run.root_state().fork_one()?;
            let (replayed_question, replayed_question_state) =
                backbone.continue_from(&question_branch, &record.question_ids)?;
            let question_replay_delta =
                maximum_abs(replayed_question.final_token(), question.question_feature());
            let question_replay_exact = question_replay_delta == 0.0
                && replayed_question_state.strict_fingerprint()
                    == question.question_state().strict_fingerprint();
            replay_exact_all_records &= question_replay_exact;

            let mut trace_question_delta = None;
            if trace_question_matches {
                let comparison = reference.compare(
                    "continuation.question_last_hidden",
                    question.question_feature(),
                )?;
                trace_question_delta = Some(comparison.max_abs());
            }

            let mut features = Vec::with_capacity(question.candidates().len());
            let mut full_features = Vec::with_capacity(question.candidates().len());
            let mut candidate_results = Vec::with_capacity(question.candidates().len());
            for (candidate_index, candidate) in question.candidates().iter().enumerate() {
                let suffix_ids = &record.candidate_suffix_ids[candidate_index];
                let mut full_ids = root_ids.clone();
                full_ids.extend_from_slice(&record.question_ids);
                full_ids.extend_from_slice(suffix_ids);
                if full_ids != record.full_candidate_ids[candidate_index] {
                    return Err(format!(
                        "fixture case {fixture_case} question {} candidate {candidate_index} \
                         segments do not concatenate to the exported full sequence",
                        record.question_id
                    )
                    .into());
                }
                positions_match_fixtures &= candidate.state().position()
                    == record.full_candidate_ids[candidate_index].len();

                // Correctness oracle: independent full-sequence execution of
                // the same finalized tokens.
                let (full_output, full_state) = backbone.prefill(&full_ids)?;
                let cached_vs_full_delta =
                    maximum_abs(candidate.feature(), full_output.final_token());
                let cached_state_equals_full_state =
                    candidate.state().strict_fingerprint() == full_state.strict_fingerprint();
                cached_state_equals_full_state_all &= cached_state_equals_full_state;
                maximum_cached_vs_full_feature_delta =
                    maximum_cached_vs_full_feature_delta.max(cached_vs_full_delta);

                // Frozen reference vector comparison.
                let expected_record = reference
                    .full_sequence_records()
                    .iter()
                    .find(|candidate_record| {
                        candidate_record.fixture_case() == record.fixture_case
                            && candidate_record.question_id() == record.question_id
                            && candidate_record.candidate_index() == candidate_index
                    })
                    .ok_or("candidate feature reference is missing")?;
                let reference_comparison =
                    reference.compare(expected_record.tensor_key(), candidate.feature())?;
                maximum_feature_delta_vs_reference =
                    maximum_feature_delta_vs_reference.max(reference_comparison.max_abs());

                // Candidate replay from the retained question-state fork must
                // be exact.
                let candidate_branch = question.question_state().fork_one()?;
                let (replayed_candidate, replayed_candidate_state) =
                    backbone.continue_from(&candidate_branch, suffix_ids)?;
                let candidate_replay_delta =
                    maximum_abs(replayed_candidate.final_token(), candidate.feature());
                let candidate_replay_exact = candidate_replay_delta == 0.0
                    && replayed_candidate_state.strict_fingerprint()
                        == candidate.state().strict_fingerprint();
                replay_exact_all_records &= candidate_replay_exact;

                let mut trace_candidate_delta = None;
                if trace_question_matches
                    && suffix_ids.as_slice() == trace.candidate_suffix_ids.as_slice()
                {
                    let comparison = reference
                        .compare("continuation.candidate_last_hidden", candidate.feature())?;
                    trace_candidate_delta = Some(comparison.max_abs());
                }

                features.push(candidate.feature().to_vec());
                full_features.push(full_output.final_token().to_vec());
                candidate_results.push(serde_json::json!({
                    "candidate_index": candidate_index,
                    "suffix_tokens": suffix_ids.len(),
                    "feature_vs_reference_max_abs": reference_comparison.max_abs(),
                    "cached_vs_full_max_abs": cached_vs_full_delta,
                    "cached_state_equals_full_state": cached_state_equals_full_state,
                    "replay_max_abs": candidate_replay_delta,
                    "replay_exact": candidate_replay_exact,
                    "trace_candidate_max_abs": trace_candidate_delta,
                }));
                candidate_count += 1;
            }

            // Sibling-order independence: re-advance every candidate from a
            // fresh fork of the question state in reverse order.
            let mut reversed_max_abs = 0.0_f64;
            for (candidate_index, candidate) in question.candidates().iter().enumerate().rev() {
                let branch = question.question_state().fork_one()?;
                let (replayed, _) = backbone
                    .continue_from(&branch, &record.candidate_suffix_ids[candidate_index])?;
                reversed_max_abs =
                    reversed_max_abs.max(maximum_abs(replayed.final_token(), candidate.feature()));
            }
            replay_exact_all_records &= reversed_max_abs == 0.0;

            // Head readout over the nested candidate features.
            let evaluation = bundle
                .head()
                .evaluate(primitive(&golden_question.primitive)?, &features)?;
            let question_logit_delta =
                maximum_delta(&evaluation.full_logits(), &head_fixture.logits)?;
            let question_probability_delta = maximum_delta(
                &evaluation.full_probabilities(),
                &head_fixture.probabilities,
            )?;
            maximum_logit_delta = maximum_logit_delta.max(question_logit_delta);
            maximum_probability_delta = maximum_probability_delta.max(question_probability_delta);

            // Oracle readout: the same head over `repeated_full` features.
            let full_evaluation = bundle
                .head()
                .evaluate(primitive(&golden_question.primitive)?, &full_features)?;
            let nested_vs_full_probability_delta = maximum_delta(
                &evaluation.full_probabilities(),
                &full_evaluation.full_probabilities(),
            )?;
            maximum_nested_vs_full_probability_delta =
                maximum_nested_vs_full_probability_delta.max(nested_vs_full_probability_delta);

            let actual_selected_id = evaluation
                .selected_candidate_index()
                .map(|index| golden_question.options[index].id.as_str());
            let argmax_changed = actual_selected_id != Some(expected.selected_id.as_str());
            argmax_changes += usize::from(argmax_changed);
            let expected_index = golden_question
                .options
                .iter()
                .position(|option| option.id == expected.selected_id)
                .ok_or("selected option is missing")?;
            let expected_policy = if expected.top_probability >= POLICY_THRESHOLD {
                PolicyAction::Accept {
                    candidate_index: expected_index,
                }
            } else {
                PolicyAction::Review
            };
            let policy_changed = evaluation.policy_action() != expected_policy;
            policy_changes += usize::from(policy_changed);

            record_results.push(serde_json::json!({
                "question_id": record.question_id,
                "candidate_count": features.len(),
                "question_replay_max_abs": question_replay_delta,
                "trace_question_max_abs": trace_question_delta,
                "logit_max_abs": question_logit_delta,
                "probability_max_abs": question_probability_delta,
                "nested_vs_full_probability_max_abs": nested_vs_full_probability_delta,
                "reversed_sibling_max_abs": reversed_max_abs,
                "selected_id": actual_selected_id,
                "argmax_changed": argmax_changed,
                "policy_changed": policy_changed,
                "candidates": candidate_results,
            }));
        }

        case_results.push(serde_json::json!({
            "fixture_case": fixture_case,
            "records": records.len(),
            "root_tokens": root_ids.len(),
            "root_feature_vs_fresh_prefill_max_abs": root_feature_delta,
            "root_content_identical": root_content_identical,
            "root_position": run.root_state().position(),
            "root_strict_fingerprint": run.root_state().strict_fingerprint().hex(),
            "trace_root_max_abs": trace_root_delta,
            "questions": record_results,
        }));
    }

    let decision_gate_passed = maximum_probability_delta <= PROBABILITY_TOLERANCE
        && argmax_changes == 0
        && policy_changes == 0;
    let oracle_gate_passed = maximum_cached_vs_full_feature_delta <= CACHED_VS_FULL_GUARD
        && maximum_nested_vs_full_probability_delta <= PROBABILITY_TOLERANCE;
    let isolation_gate_passed = root_immutable_all_cases
        && replay_exact_all_records
        && positions_match_fixtures
        && cached_state_equals_full_state_all;
    let phase35_gate_passed = decision_gate_passed && oracle_gate_passed && isolation_gate_passed;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "phase35_nested_gate_passed": phase35_gate_passed,
            "decision_gate_passed": decision_gate_passed,
            "oracle_gate_passed": oracle_gate_passed,
            "isolation_gate_passed": isolation_gate_passed,
            "questions": grouped.values().map(Vec::len).sum::<usize>(),
            "candidates": candidate_count,
            "maximum_probability_delta": maximum_probability_delta,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "maximum_logit_delta": maximum_logit_delta,
            "ordering_tolerance": ORDERING_TOLERANCE,
            "maximum_feature_delta_vs_reference": maximum_feature_delta_vs_reference,
            "maximum_cached_vs_full_feature_delta": maximum_cached_vs_full_feature_delta,
            "cached_vs_full_guard": CACHED_VS_FULL_GUARD,
            "cached_state_equals_full_state_all": cached_state_equals_full_state_all,
            "maximum_nested_vs_full_probability_delta": maximum_nested_vs_full_probability_delta,
            "argmax_changes": argmax_changes,
            "policy_changes": policy_changes,
            "root_immutable_all_cases": root_immutable_all_cases,
            "replay_exact_all_records": replay_exact_all_records,
            "positions_match_fixtures": positions_match_fixtures,
            "cases": case_results,
        }))?
    );

    if !phase35_gate_passed {
        return Err("native sequential nested parity gate failed".into());
    }
    Ok(())
}

fn required_path(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    name: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing required {name}").into())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn primitive(value: &str) -> Result<PrimitiveKind, Box<dyn Error>> {
    match value {
        "choice" => Ok(PrimitiveKind::Choice),
        "noul" => Ok(PrimitiveKind::Noul),
        "score" => Ok(PrimitiveKind::Score),
        other => Err(format!("unexpected primitive {other}").into()),
    }
}

fn maximum_abs(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from((left - right).abs()))
        .fold(0.0, f64::max)
}

fn maximum_delta(actual: &[f64], expected: &[f64]) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "vector length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max))
}
