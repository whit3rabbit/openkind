use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use opendecision_backends::qwen35::{
    BackboneReference, PolicyAction, PrimitiveKind, Qwen35Backbone, ReferenceBundle,
    ORDERING_TOLERANCE, POLICY_THRESHOLD, PROBABILITY_TOLERANCE,
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

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = required_path(&mut arguments, "checkpoint-root")?;
    let reference_root = required_path(&mut arguments, "phase3b-reference-root")?;
    let head_bundle_root = required_path(&mut arguments, "head-bundle-root")?;
    if arguments.next().is_some() {
        return Err("qwen35_full_parity accepts exactly three paths".into());
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

    let mut maximum_feature_delta = 0.0_f64;
    let mut maximum_logit_delta = 0.0_f64;
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
            let output = backbone.forward(input_ids)?;
            let feature = output.final_token().to_vec();
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
        let actual_logits = evaluation.full_logits();
        let actual_probabilities = evaluation.full_probabilities();
        let question_logit_delta = maximum_delta(&actual_logits, &head_fixture.logits)?;
        let question_probability_delta =
            maximum_delta(&actual_probabilities, &head_fixture.probabilities)?;
        maximum_logit_delta = maximum_logit_delta.max(question_logit_delta);
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
            "logit_max_abs": question_logit_delta,
            "probability_max_abs": question_probability_delta,
            "selected_id": actual_selected_id,
            "argmax_changed": argmax_changed,
            "policy_changed": policy_changed,
        }));
    }

    let absolute_logit_diagnostic_passed = maximum_logit_delta <= ORDERING_TOLERANCE;
    // Phase 3.1 applies the absolute-logit tolerance to frozen input features.
    // Phase 3B deliberately exports fresh backbone features and declares its
    // backbone gate over probabilities and discrete decisions instead.
    let phase3b_decision_gate_passed = maximum_probability_delta <= PROBABILITY_TOLERANCE
        && argmax_changes == 0
        && policy_changes == 0;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "phase3b_decision_gate_passed": phase3b_decision_gate_passed,
            "questions": question_results.len(),
            "candidates": candidate_count,
            "maximum_feature_delta": maximum_feature_delta,
            "maximum_logit_delta": maximum_logit_delta,
            "ordering_tolerance": ORDERING_TOLERANCE,
            "absolute_logit_diagnostic_passed": absolute_logit_diagnostic_passed,
            "maximum_probability_delta": maximum_probability_delta,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "argmax_changes": argmax_changes,
            "policy_changes": policy_changes,
            "results": question_results,
        }))?
    );
    if !phase3b_decision_gate_passed {
        return Err("native full-sequence backbone parity gate failed".into());
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
