//! Full 32-layer backbone parity evaluation against the frozen Phase 3B bundle.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use openkind_backends::qwen35::mlx::MlxQwen35Backbone;
use openkind_backends::qwen35::{
    BackboneReference, PolicyAction, ReferenceBundle, POLICY_THRESHOLD, PROBABILITY_TOLERANCE,
};

use super::fixtures::{
    git_commit, maximum_delta, primitive, read_json, GoldenCase, ProbabilityReference,
    TokenFixtures,
};

pub(crate) struct FullResult {
    pub(crate) gate_passed: bool,
    pub(crate) json: serde_json::Value,
}

pub(crate) fn full_stage(
    backbone: &MlxQwen35Backbone,
    reference: &BackboneReference,
    reference_root: &Path,
    head_bundle_root: &Path,
) -> Result<FullResult, Box<dyn Error>> {
    let token_fixtures: TokenFixtures = read_json(&reference_root.join("TOKEN_FIXTURES.json"))?;
    let golden_cases: Vec<GoldenCase> = read_json(&head_bundle_root.join("golden.json"))?;
    let probability_reference: ProbabilityReference =
        read_json(&reference_root.join("PROBABILITY_REFERENCE.json"))?;
    let bundle = ReferenceBundle::load(head_bundle_root)?;

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

    let mut maximum_feature_delta = 0.0_f64;
    let mut maximum_probability_delta = 0.0_f64;
    let mut argmax_changes = 0_usize;
    let mut policy_changes = 0_usize;
    let mut candidate_count = 0_usize;
    let mut question_results = Vec::new();

    for record in &token_fixtures.records {
        let golden_case = golden_cases
            .get(record.fixture_case)
            .ok_or("fixture case is out of range")?;
        let question = golden_case
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

        let mut features = Vec::with_capacity(record.full_candidate_ids.len());
        let mut question_feature_delta = 0.0_f64;
        for (candidate_index, input_ids) in record.full_candidate_ids.iter().enumerate() {
            let (output, _state) = backbone
                .prefill(input_ids)
                .map_err(|error| format!("MLX candidate forward failed: {error}"))?;
            let feature = output.feature().to_vec();
            let expected_record = reference
                .full_sequence_records()
                .iter()
                .find(|candidate| {
                    candidate.fixture_case() == record.fixture_case
                        && candidate.question_id() == record.question_id
                        && candidate.candidate_index() == candidate_index
                })
                .ok_or("candidate feature reference is missing")?;
            let comparison = reference.compare(expected_record.tensor_key(), &feature)?;
            question_feature_delta = question_feature_delta.max(comparison.max_abs());
            maximum_feature_delta = maximum_feature_delta.max(comparison.max_abs());
            features.push(feature);
            candidate_count += 1;
        }

        let evaluation = bundle
            .head()
            .evaluate(primitive(&question.primitive)?, &features)?;
        let actual_probabilities = evaluation.full_probabilities();
        let question_probability_delta =
            maximum_delta(&actual_probabilities, &head_fixture.probabilities)?;
        maximum_probability_delta = maximum_probability_delta.max(question_probability_delta);

        let actual_selected_id = evaluation
            .selected_candidate_index()
            .map(|index| question.options[index].id.as_str());
        let argmax_changed = actual_selected_id != Some(expected.selected_id.as_str());
        argmax_changes += usize::from(argmax_changed);
        let expected_index = question
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
        question_results.push(serde_json::json!({
            "fixture_case": record.fixture_case,
            "question_id": record.question_id,
            "candidate_count": features.len(),
            "feature_max_abs": question_feature_delta,
            "probability_max_abs": question_probability_delta,
            "selected_id": actual_selected_id,
            "argmax_changed": argmax_changed,
            "policy_changed": policy_changed,
        }));
    }

    let gate_passed = maximum_probability_delta <= PROBABILITY_TOLERANCE
        && argmax_changes == 0
        && policy_changes == 0;
    Ok(FullResult {
        gate_passed,
        json: serde_json::json!({
            "precision": backbone.arithmetic_id(),
            "gate_passed": gate_passed,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "maximum_feature_delta": maximum_feature_delta,
            "maximum_probability_delta": maximum_probability_delta,
            "argmax_changes": argmax_changes,
            "policy_changes": policy_changes,
            "questions": question_results.len(),
            "candidates": candidate_count,
            "results": question_results,
            "git_commit": git_commit(),
        }),
    })
}
