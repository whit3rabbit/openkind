use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use opendecision_backends::qwen35::{
    BackboneReference, PrimitiveKind, ReferenceBundle, ORDERING_TOLERANCE, PROBABILITY_TOLERANCE,
};
use serde::Deserialize;

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
}

#[derive(Debug, Deserialize)]
struct HeadFixture {
    question_id: String,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let reference_root = required_path(&mut arguments, "phase3b-reference-root")?;
    let head_bundle_root = required_path(&mut arguments, "head-bundle-root")?;
    if arguments.next().is_some() {
        return Err("qwen35_reference_head_probe accepts exactly two paths".into());
    }
    let reference = BackboneReference::load(&reference_root)?;
    let bundle = ReferenceBundle::load(&head_bundle_root)?;
    let cases: Vec<GoldenCase> = read_json(&head_bundle_root.join("golden.json"))?;
    let mut maximum_logit_delta = 0.0_f64;
    let mut maximum_probability_delta = 0.0_f64;
    let mut results = Vec::new();
    for (fixture_case, case) in cases.iter().enumerate() {
        for fixture in &case.head_and_token_fixtures {
            let question = case
                .request
                .questions
                .iter()
                .find(|question| question.id == fixture.question_id)
                .ok_or("golden question is missing")?;
            let mut records: Vec<_> = reference
                .full_sequence_records()
                .iter()
                .filter(|candidate| {
                    candidate.fixture_case() == fixture_case
                        && candidate.question_id() == fixture.question_id
                })
                .collect();
            records.sort_by_key(|candidate| candidate.candidate_index());
            let features: Vec<_> = records
                .iter()
                .map(|candidate| {
                    reference
                        .vector(candidate.tensor_key())
                        .ok_or("candidate vector is missing")
                        .map(<[f32]>::to_vec)
                })
                .collect::<Result<_, _>>()?;
            let evaluation = bundle
                .head()
                .evaluate(primitive(&question.primitive)?, &features)?;
            let logit_delta = maximum_delta(&evaluation.full_logits(), &fixture.logits)?;
            let probability_delta =
                maximum_delta(&evaluation.full_probabilities(), &fixture.probabilities)?;
            maximum_logit_delta = maximum_logit_delta.max(logit_delta);
            maximum_probability_delta = maximum_probability_delta.max(probability_delta);
            results.push(serde_json::json!({
                "fixture_case": fixture_case,
                "question_id": fixture.question_id,
                "logit_max_abs": logit_delta,
                "probability_max_abs": probability_delta,
            }));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "phase3b_reference_vs_phase31_head_fixture": {
                "maximum_logit_delta": maximum_logit_delta,
                "ordering_tolerance": ORDERING_TOLERANCE,
                "absolute_logit_gate_passed": maximum_logit_delta <= ORDERING_TOLERANCE,
                "maximum_probability_delta": maximum_probability_delta,
                "probability_tolerance": PROBABILITY_TOLERANCE,
                "probability_gate_passed": maximum_probability_delta <= PROBABILITY_TOLERANCE,
            },
            "results": results,
        }))?
    );
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
        return Err("comparison vector length mismatch".into());
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max))
}
