//! Offline fixture generator and calibration helper for the pinned
//! `schema-scorer` profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_schema_scorer_fixture -- \
//!   --model-root ~/.cache/openkind/msmarco-minilm --fit
//! # pin the printed temperature in families/schema_scorer/mod.rs
//! cargo run -p openkind-backends --bin gen_schema_scorer_fixture -- \
//!   --model-root ~/.cache/openkind/msmarco-minilm \
//!   --output crates/openkind-backends/tests/fixtures/schema_scorer_5a7350af556f0ee66566/golden.json
//! ```
//!
//! The calibration set is the shared letter-family set in
//! `families/calibration.rs`; the fit assumes the profile temperature
//! constant is `1.0` while running.

use std::path::PathBuf;

use openkind_backends::families::calibration::calibration_cases;
use openkind_backends::families::schema_scorer::{
    SchemaScorerEngine, SchemaScorerEngineConfig, BACKBONE_ID, BACKBONE_REVISION,
    EXECUTION_ARITHMETIC_ID, PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::{Answer, State, SystemRequest};
use openkind_engine::DecisionEngine;
use serde_json::json;

fn correct_probability(
    case: &openkind_backends::families::calibration::CalibrationCase,
    answer: &Answer,
) -> Option<f64> {
    match answer {
        Answer::Choice(choice) => choice.probabilities.get(&case.truth_label).copied(),
        Answer::Score(score) => score.probabilities.get(&case.truth_label).copied(),
        Answer::Noul(noul) => {
            if case.truth_label == "true" {
                Some(noul.noul)
            } else {
                Some(1.0 - noul.noul)
            }
        }
    }
}

fn distribution(
    case: &openkind_backends::families::calibration::CalibrationCase,
    answer: &Answer,
) -> Option<Vec<f64>> {
    match answer {
        Answer::Choice(choice) => case
            .option_labels
            .iter()
            .map(|label| choice.probabilities.get(label).copied())
            .collect(),
        Answer::Score(score) => case
            .option_labels
            .iter()
            .map(|label| score.probabilities.get(label).copied())
            .collect(),
        Answer::Noul(noul) => Some(vec![1.0 - noul.noul, noul.noul]),
    }
}

/// Mean NLL over correct options for candidate temperature `t`, given
/// distributions recorded at temperature `t0 = 1.0`.
fn mean_nll(distributions: &[Vec<f64>], truths: &[usize], t: f64) -> f64 {
    let mut total = 0.0;
    for (distribution, &truth) in distributions.iter().zip(truths) {
        let rescaled: Vec<f64> = distribution
            .iter()
            .map(|p| {
                if *p <= f64::EPSILON {
                    f64::NEG_INFINITY
                } else {
                    p.ln() / t
                }
            })
            .collect();
        let max = rescaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = rescaled.iter().map(|v| (v - max).exp()).sum();
        total -= ((rescaled[truth] - max).exp() / sum).ln();
    }
    total / distributions.len() as f64
}

/// Two-pass multiplicative grid search over temperature.
fn fit_temperature(distributions: &[Vec<f64>], truths: &[usize]) -> f64 {
    let scan = |from: f64, to: f64, step: f64| {
        let mut best_t = from;
        let mut best_nll = f64::INFINITY;
        let mut t = from;
        while t <= to {
            let nll = mean_nll(distributions, truths, t);
            if nll < best_nll {
                best_nll = nll;
                best_t = t;
            }
            t *= step;
        }
        (best_t, best_nll)
    };
    let (coarse, _) = scan(0.05, 20.0, 1.05);
    let (fine, _) = scan(coarse / 1.05, coarse * 1.05, 1.002);
    fine
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_root = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/openkind/msmarco-minilm");
    let mut fit = false;
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => model_root = PathBuf::from(args.next().expect("--model-root value")),
            "--fit" => fit = true,
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let engine = SchemaScorerEngine::load(SchemaScorerEngineConfig {
        model_root: model_root.clone(),
        limits: FamilyLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 0,
            retry_after_ms: 100,
            evaluation_timeout: None,
        },
    })?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let case_list = calibration_cases();
    let mut distributions = Vec::new();
    let mut truths = Vec::new();
    let mut rendered_cases = Vec::new();
    for (index, case) in case_list.iter().enumerate() {
        let request: SystemRequest = serde_json::from_value(json!({
            "state": State::Text(case.state.clone()),
            "model": "fixture",
            "questions": { "q0": case.question },
        }))?;
        let request_value = serde_json::to_value(&request)?;
        let response = runtime.block_on(engine.evaluate(request))?;
        let answer = response.answers.get("q0").expect("answer for q0");
        let probability = correct_probability(case, answer)
            .ok_or_else(|| format!("case {} produced no correct-option probability", case.name))?;
        let distribution = distribution(case, answer)
            .ok_or_else(|| format!("case {} produced an incomplete distribution", case.name))?;
        let truth_index = case
            .option_labels
            .iter()
            .position(|label| *label == case.truth_label)
            .expect("truth label is an option");
        eprintln!(
            "[fixture {:>2}] {:<16} p(correct) = {probability:.6}",
            index, case.name
        );
        distributions.push(distribution);
        truths.push(truth_index);
        rendered_cases.push((
            case.name.clone(),
            request_value,
            serde_json::to_value(&response)?,
        ));
    }

    if fit {
        let temperature = fit_temperature(&distributions, &truths);
        println!("{temperature:.17}");
        return Ok(());
    }

    let output = output.ok_or_else(|| "--output is required without --fit".to_owned())?;
    let case_values: Vec<serde_json::Value> = rendered_cases
        .into_iter()
        .map(|(name, request, response)| {
            json!({ "name": name, "request": request, "response": response })
        })
        .collect();
    let golden = json!({
        "profile_id": PROFILE_ID,
        "backbone": { "id": BACKBONE_ID, "revision": BACKBONE_REVISION },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "calibration_temperature":
            openkind_backends::families::schema_scorer::CALIBRATION_TEMPERATURE,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-schema-scorer-fixture",
            "case_construction": "correct option stated verbatim in the state document",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
