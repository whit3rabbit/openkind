//! Offline fixture generator and calibration helper for the pinned
//! `encoder-nli` profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_encoder_nli_fixture -- \
//!   --model-root ~/.cache/openkind/distilbert-mnli --fit
//! # pin the printed temperature in families/encoder_nli/mod.rs
//! cargo run -p openkind-backends --bin gen_encoder_nli_fixture -- \
//!   --model-root ~/.cache/openkind/distilbert-mnli \
//!   --output crates/openkind-backends/tests/fixtures/encoder_nli_1041a4c362338a61b820/golden.json
//! ```
//!
//! The calibration set is a fixed list of synthetic premise–hypothesis cases
//! whose correct option is entailed by the state document. The fit assumes
//! the profile temperature constant is `1.0` while running.

use std::path::PathBuf;

use openkind_backends::families::encoder_nli::{
    EncoderNliEngine, EncoderNliEngineConfig, BACKBONE_ID, BACKBONE_REVISION,
    EXECUTION_ARITHMETIC_ID, PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::{Answer, Question, State, SystemRequest};
use openkind_engine::DecisionEngine;
use serde_json::json;

/// One labeled calibration case.
struct Case {
    name: String,
    /// Ordered candidate labels as they appear in the answer map.
    option_labels: Vec<String>,
    /// Candidate label carrying the correct answer.
    truth_label: String,
    state: String,
    question: Question,
}

fn choice_case(name: &str, state: &str, options: &[&str], truth: usize) -> Case {
    Case {
        name: name.to_owned(),
        option_labels: options.iter().map(|option| option.to_string()).collect(),
        truth_label: options[truth].to_string(),
        state: state.to_owned(),
        question: Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which option is explicitly recorded in the state?"),
            criteria: options
                .iter()
                .map(|option| (option.to_string(), None))
                .collect(),
        }),
    }
}

fn noul_case(name: &str, state: &str, instruction: &'static str, truth: bool) -> Case {
    Case {
        name: name.to_owned(),
        option_labels: vec!["false".into(), "true".into()],
        truth_label: if truth { "true".into() } else { "false".into() },
        state: state.to_owned(),
        question: Question::Noul(openkind_core::NoulQuestion {
            instructions: json!(instruction),
            criteria: None,
        }),
    }
}

fn score_case(name: &str, state: &str, levels: usize, truth: usize) -> Case {
    Case {
        name: name.to_owned(),
        option_labels: (0..levels).map(|level| level.to_string()).collect(),
        truth_label: truth.to_string(),
        state: state.to_owned(),
        question: Question::Score(openkind_core::ScoreQuestion {
            instructions: json!("What is the documented impact level?"),
            criteria: (0..levels).map(|level| format!("level {level}")).collect(),
        }),
    }
}

fn cases() -> Vec<Case> {
    vec![
        choice_case(
            "asset-a",
            "Fictional incident record. Affected asset: application server. The asset operating system is Linux. These statements are the full evidence record; do not infer missing facts.",
            &["workstation", "network appliance", "application server", "database cluster"],
            2,
        ),
        choice_case(
            "asset-b",
            "Fictional incident record. Affected asset: storage array. These statements are the full evidence record; do not infer missing facts.",
            &["workstation", "router", "application server", "storage array"],
            3,
        ),
        choice_case(
            "os-a",
            "Fictional incident record. The asset operating system is Windows. These statements are the full evidence record; do not infer missing facts.",
            &["Linux", "Windows", "macOS", "BSD"],
            1,
        ),
        choice_case(
            "os-b",
            "Fictional incident record. The asset operating system is Linux. These statements are the full evidence record; do not infer missing facts.",
            &["Linux", "Windows Server 2022", "macOS", "Solaris"],
            0,
        ),
        choice_case(
            "owner-a",
            "Fictional incident record. The owner of record is the platform team. These statements are the full evidence record; do not infer missing facts.",
            &["security team", "platform team", "external vendor", "data team"],
            1,
        ),
        choice_case(
            "region-a",
            "Fictional incident record. The affected deployment runs in us-east-1. These statements are the full evidence record; do not infer missing facts.",
            &["us-east-1", "eu-west-2", "ap-southeast-1", "us-west-1"],
            0,
        ),
        choice_case(
            "protocol-a",
            "Fictional incident record. The observed connection used SSH. These statements are the full evidence record; do not infer missing facts.",
            &["HTTPS", "SSH", "SMTP", "DNS"],
            1,
        ),
        choice_case(
            "status",
            "Fictional incident record. The incident status is investigating. These statements are the full evidence record; do not infer missing facts.",
            &["open", "investigating", "mitigated", "closed"],
            1,
        ),
        noul_case(
            "external-true",
            "Fictional incident record. An external destination was observed in the connection logs. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that an external destination was observed?",
            true,
        ),
        noul_case(
            "external-false",
            "Fictional incident record. An external destination was not observed. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that an external destination was observed?",
            false,
        ),
        noul_case(
            "mfa-true",
            "Fictional incident record. The record states that two-factor authentication was enabled on the affected account. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that two-factor authentication was enabled?",
            true,
        ),
        noul_case(
            "export-false",
            "Fictional incident record. No data export was approved during the window. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that a data export was approved?",
            false,
        ),
        score_case(
            "impact-level-0",
            "Fictional incident record. The documented impact level is 0. These statements are the full evidence record; do not infer missing facts.",
            5,
            0,
        ),
        score_case(
            "impact-level-2",
            "Fictional incident record. The documented impact level is 2. These statements are the full evidence record; do not infer missing facts.",
            5,
            2,
        ),
        score_case(
            "impact-level-4",
            "Fictional incident record. The documented impact level is 4. These statements are the full evidence record; do not infer missing facts.",
            5,
            4,
        ),
    ]
}

/// Probability of the correct option under the engine's current calibration.
fn correct_probability(case: &Case, answer: &Answer) -> Option<f64> {
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

/// Full candidate distribution in candidate order, for temperature refitting.
fn distribution(case: &Case, answer: &Answer) -> Option<Vec<f64>> {
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
/// distributions recorded at temperature `t0 = 1.0` (`q_i(t) ∝ p_i^(1/t)`).
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
    let mut model_root = PathBuf::from(
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .unwrap_or_default(),
    )
    .join(".cache/openkind/distilbert-mnli");
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

    let engine = EncoderNliEngine::load(EncoderNliEngineConfig {
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

    let case_list = cases();
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
            openkind_backends::families::encoder_nli::CALIBRATION_TEMPERATURE,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-encoder-nli-fixture",
            "case_construction": "correct option entailed by the state document",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
