//! Phase 3.6/3.7 native gate: breadth-first batched Q/K execution parity for
//! the pinned Qwen3.5 profile.
//!
//! For every exported fixture case this gate prefills the shared state once
//! and executes the graph twice in-process: the Phase 3.5 sequential nested
//! baseline and the Phase 3.6/3.7 batched path (`fork_batch` question lanes
//! advanced breadth-first, then a `fork_batch` candidate fan-out per question
//! state). Declared gates: the batched run must match the sequential
//! baseline exactly (features and strict state fingerprints), the frozen head
//! must reach the same probability parity against the golden fixtures
//! (`0.005`) with zero argmax and zero policy changes under both strategies,
//! and the retained root must equal an independent fresh prefill exactly.
//! Position checks and fan-out byte accounting are reported alongside.
//! Combined with the Phase 3.5 gate (sequential versus `repeated_full`
//! agreement exact on the same fixtures and machine), exact batched-versus-
//! sequential equality chains the batched path to the full-sequence oracle.
//! This is a CPU/Candle result; it does not establish Metal, vectorized
//! suffix kernels, service registration, or release promotion.
//!
//! Usage: `qwen35_batched_parity <checkpoint-root> <phase3b-reference-root>
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
    ReferenceBundle, POLICY_THRESHOLD, PROBABILITY_TOLERANCE,
};
use serde::Deserialize;

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

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = required_path(&mut arguments, "checkpoint-root")?;
    let reference_root = required_path(&mut arguments, "phase3b-reference-root")?;
    let head_bundle_root = required_path(&mut arguments, "head-bundle-root")?;
    if arguments.next().is_some() {
        return Err("qwen35_batched_parity accepts exactly three paths".into());
    }

    let reference = BackboneReference::load(&reference_root)?;
    let backbone = Qwen35Backbone::load(checkpoint_root)?;
    let bundle = ReferenceBundle::load(&head_bundle_root)?;
    let token_fixtures: TokenFixtures = read_json(&reference_root.join("TOKEN_FIXTURES.json"))?;
    let golden_cases: Vec<GoldenCase> = read_json(&head_bundle_root.join("golden.json"))?;
    let probability_reference: ProbabilityReference =
        read_json(&reference_root.join("PROBABILITY_REFERENCE.json"))?;

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
    let mut argmax_changes = 0_usize;
    let mut policy_changes = 0_usize;
    let mut strategy_decision_changes = 0_usize;
    let mut candidate_count = 0_usize;
    let mut batched_equals_sequential = true;
    let mut root_immutable_all_cases = true;
    let mut positions_match_fixtures = true;
    let mut case_results = Vec::new();

    for (&fixture_case, records) in &grouped {
        let root_ids = &records[0].root_ids;
        if records.iter().any(|record| record.root_ids != *root_ids) {
            return Err(format!("fixture case {fixture_case} records disagree on root_ids").into());
        }

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

        // Independent prefill for the root-immutability anchor.
        let (fresh_root_output, fresh_root_state) = backbone.prefill(root_ids)?;

        // The Phase 3.5 sequential baseline and the Phase 3.6/3.7 batched
        // path over the same finalized tokens.
        let sequential = backbone.evaluate_nested(root_ids, &plans)?;
        let batched = backbone.evaluate_batched_nested(root_ids, &plans)?;

        let root_feature_delta =
            maximum_abs(batched.root_feature(), fresh_root_output.final_token());
        let root_content_identical =
            batched.root_state().strict_fingerprint() == fresh_root_state.strict_fingerprint();
        root_immutable_all_cases &= root_feature_delta == 0.0 && root_content_identical;
        positions_match_fixtures &= batched.root_state().position() == root_ids.len();

        let mut record_results = Vec::with_capacity(records.len());
        for (record_index, record) in records.iter().enumerate() {
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

            let sequential_question = &sequential.questions()[record_index];
            let batched_question = &batched.questions()[record_index];

            // Batched question lane must equal the sequential fork exactly.
            let question_feature_delta = maximum_abs(
                batched_question.question_feature(),
                sequential_question.question_feature(),
            );
            let question_state_identical = batched_question.question_state().strict_fingerprint()
                == sequential_question.question_state().strict_fingerprint();
            batched_equals_sequential &= question_feature_delta == 0.0 && question_state_identical;
            positions_match_fixtures &= batched_question.question_state().position()
                == root_ids.len() + record.question_ids.len();

            let mut features = Vec::with_capacity(batched_question.candidates().len());
            let mut candidate_results = Vec::with_capacity(batched_question.candidates().len());
            for (candidate_index, batched_candidate) in
                batched_question.candidates().iter().enumerate()
            {
                let sequential_candidate = &sequential_question.candidates()[candidate_index];
                let candidate_feature_delta =
                    maximum_abs(batched_candidate.feature(), sequential_candidate.feature());
                let candidate_state_identical = batched_candidate.state().strict_fingerprint()
                    == sequential_candidate.state().strict_fingerprint();
                batched_equals_sequential &=
                    candidate_feature_delta == 0.0 && candidate_state_identical;
                positions_match_fixtures &= batched_candidate.state().position()
                    == root_ids.len()
                        + record.question_ids.len()
                        + record.candidate_suffix_ids[candidate_index].len();

                let reference_record = reference
                    .full_sequence_records()
                    .iter()
                    .find(|candidate_record| {
                        candidate_record.fixture_case() == record.fixture_case
                            && candidate_record.question_id() == record.question_id
                            && candidate_record.candidate_index() == candidate_index
                    })
                    .ok_or("candidate feature reference is missing")?;
                let reference_comparison = reference
                    .compare(reference_record.tensor_key(), batched_candidate.feature())?;
                maximum_feature_delta_vs_reference =
                    maximum_feature_delta_vs_reference.max(reference_comparison.max_abs());

                features.push(batched_candidate.feature().to_vec());
                candidate_results.push(serde_json::json!({
                    "candidate_index": candidate_index,
                    "suffix_tokens": record.candidate_suffix_ids[candidate_index].len(),
                    "vs_sequential_max_abs": candidate_feature_delta,
                    "state_identical_to_sequential": candidate_state_identical,
                    "feature_vs_reference_max_abs": reference_comparison.max_abs(),
                }));
                candidate_count += 1;
            }

            // Frozen head over the batched candidate features.
            let evaluation = bundle
                .head()
                .evaluate(primitive(&golden_question.primitive)?, &features)?;
            let sequential_evaluation = bundle.head().evaluate(
                primitive(&golden_question.primitive)?,
                &sequential_question
                    .candidates()
                    .iter()
                    .map(|candidate| candidate.feature().to_vec())
                    .collect::<Vec<_>>(),
            )?;
            let question_logit_delta =
                maximum_delta(&evaluation.full_logits(), &head_fixture.logits)?;
            let question_probability_delta = maximum_delta(
                &evaluation.full_probabilities(),
                &head_fixture.probabilities,
            )?;
            maximum_logit_delta = maximum_logit_delta.max(question_logit_delta);
            maximum_probability_delta = maximum_probability_delta.max(question_probability_delta);

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
            let strategy_changed = evaluation.policy_action()
                != sequential_evaluation.policy_action()
                || evaluation.selected_candidate_index()
                    != sequential_evaluation.selected_candidate_index();
            strategy_decision_changes += usize::from(strategy_changed);

            record_results.push(serde_json::json!({
                "question_id": record.question_id,
                "candidate_count": features.len(),
                "question_vs_sequential_max_abs": question_feature_delta,
                "question_state_identical_to_sequential": question_state_identical,
                "logit_max_abs": question_logit_delta,
                "probability_max_abs": question_probability_delta,
                "selected_id": actual_selected_id,
                "argmax_changed": argmax_changed,
                "policy_changed": policy_changed,
                "strategy_decision_changed": strategy_changed,
                "candidates": candidate_results,
            }));
        }

        case_results.push(serde_json::json!({
            "fixture_case": fixture_case,
            "records": records.len(),
            "root_tokens": root_ids.len(),
            "question_fanout_lanes": plans.len(),
            "question_batch_bytes": batched.question_batch_bytes(),
            "root_bytes": batched.root_state().tensor_storage_bytes(),
            "root_feature_vs_fresh_prefill_max_abs": root_feature_delta,
            "root_content_identical": root_content_identical,
            "questions": record_results,
        }));
    }

    let batched_parity_gate_passed = batched_equals_sequential && root_immutable_all_cases;
    let decision_gate_passed = maximum_probability_delta <= PROBABILITY_TOLERANCE
        && argmax_changes == 0
        && policy_changes == 0;
    let phase37_gate_passed = batched_parity_gate_passed
        && decision_gate_passed
        && strategy_decision_changes == 0
        && positions_match_fixtures;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "phase36_phase37_batched_gate_passed": phase37_gate_passed,
            "batched_parity_gate_passed": batched_parity_gate_passed,
            "decision_gate_passed": decision_gate_passed,
            "batched_equals_sequential": batched_equals_sequential,
            "questions": grouped.values().map(Vec::len).sum::<usize>(),
            "candidates": candidate_count,
            "maximum_probability_delta": maximum_probability_delta,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "maximum_logit_delta": maximum_logit_delta,
            "maximum_feature_delta_vs_reference": maximum_feature_delta_vs_reference,
            "argmax_changes": argmax_changes,
            "policy_changes": policy_changes,
            "strategy_decision_changes": strategy_decision_changes,
            "root_immutable_all_cases": root_immutable_all_cases,
            "positions_match_fixtures": positions_match_fixtures,
            "cases": case_results,
        }))?
    );

    if !phase37_gate_passed {
        return Err("native batched Q/K parity gate failed".into());
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
