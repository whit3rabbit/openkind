//! Offline fixture generator and calibration helper for the pinned `kev`
//! profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_kev_fixture -- \
//!   --model-root ~/.cache/openkind/kev-0.6b \
//!   --base-root ~/.cache/openkind/qwen3-0.6b-base --fit
//! # pin the printed temperature in families/kev/mod.rs
//! cargo run -p openkind-backends --bin gen_kev_fixture -- \
//!   --model-root ~/.cache/openkind/kev-0.6b \
//!   --base-root ~/.cache/openkind/qwen3-0.6b-base \
//!   --output crates/openkind-backends/tests/fixtures/kev_39d88c11faeb4ac165fa/golden.json
//! ```

use std::path::PathBuf;

use openkind_backends::families::kev::{
    KevEngine, KevEngineConfig, BACKBONE_ID, BACKBONE_REVISION, BASE_MODEL_ID, BASE_MODEL_REVISION,
    EXECUTION_ARITHMETIC_ID, PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::{Answer, Question, State, SystemRequest};
use openkind_engine::DecisionEngine;
use serde_json::json;

/// One labeled calibration case.
struct Case {
    name: String,
    /// Candidate label carrying the correct answer.
    truth_label: String,
    state: serde_json::Value,
    question: Question,
}

fn choice_case(name: &str, state: &str, options: &[&str], truth: usize) -> Case {
    Case {
        name: name.to_owned(),
        truth_label: options[truth].to_string(),
        state: json!(state),
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
        truth_label: if truth { "true".into() } else { "false".into() },
        state: json!(state),
        question: Question::Noul(openkind_core::NoulQuestion {
            instructions: json!(instruction),
            criteria: None,
        }),
    }
}

fn score_case(name: &str, state: &str, levels: &[&str], truth: usize) -> Case {
    Case {
        name: name.to_owned(),
        truth_label: truth.to_string(),
        state: json!(state),
        question: Question::Score(openkind_core::ScoreQuestion {
            instructions: json!("What is the documented impact level?"),
            criteria: levels.iter().map(|level| level.to_string()).collect(),
        }),
    }
}

/// The pinned stated-fact calibration set.
fn cases() -> Vec<Case> {
    vec![
        choice_case(
            "os-a",
            "Fictional incident record. The asset operating system is Windows.",
            &["Linux", "Windows", "macOS", "BSD"],
            1,
        ),
        choice_case(
            "os-b",
            "Fictional incident record. The asset operating system is Linux.",
            &["Linux", "Windows Server 2022", "macOS", "Solaris"],
            0,
        ),
        choice_case(
            "region-a",
            "Fictional incident record. The affected deployment runs in us-east-1.",
            &["us-east-1", "eu-west-2", "ap-southeast-1", "us-west-1"],
            0,
        ),
        choice_case(
            "protocol-a",
            "Fictional incident record. The observed connection used SSH.",
            &["HTTPS", "SSH", "SMTP", "DNS"],
            1,
        ),
        choice_case(
            "owner-a",
            "Fictional incident record. The owner of record is the platform team.",
            &["security team", "platform team", "external vendor", "data team"],
            1,
        ),
        choice_case(
            "status",
            "Fictional incident record. The incident status is investigating.",
            &["open", "investigating", "mitigated", "closed"],
            1,
        ),
        noul_case(
            "external-true",
            "Fictional incident record. An external destination was observed in the connection logs.",
            "Does the record explicitly state that an external destination was observed?",
            true,
        ),
        noul_case(
            "external-false",
            "Fictional incident record. The record does not mention external destinations.",
            "Does the record explicitly state that an external destination was observed?",
            false,
        ),
        noul_case(
            "mfa-true",
            "Fictional incident record. The record states that two-factor authentication was enabled on the affected account.",
            "Does the record explicitly state that two-factor authentication was enabled?",
            true,
        ),
        noul_case(
            "export-false",
            "Fictional incident record. No data export was approved during the window.",
            "Does the record explicitly state that a data export was approved?",
            false,
        ),
        noul_case(
            "patch-true",
            "Fictional incident record. The patch level is current as of the report date.",
            "Is the patch level described as current?",
            true,
        ),
        score_case(
            "impact-minor",
            "Fictional incident record. The documented impact level is minor.",
            &["minor", "moderate", "major", "critical"],
            0,
        ),
        score_case(
            "impact-moderate",
            "Fictional incident record. The documented impact level is moderate.",
            &["minor", "moderate", "major", "critical"],
            1,
        ),
        score_case(
            "impact-major",
            "Fictional incident record. The documented impact level is major.",
            &["minor", "moderate", "major", "critical"],
            2,
        ),
        score_case(
            "impact-critical",
            "Fictional incident record. The documented impact level is critical.",
            &["minor", "moderate", "major", "critical"],
            3,
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

/// Full candidate distribution in the sorted wire candidate order, for
/// temperature refitting (aligned with the truth index).
fn distribution(question: &Question, answer: &Answer) -> Option<Vec<f64>> {
    let ordered: Vec<String> = openkind_backends::families::wire::unpack_question("q0", question)
        .ok()?
        .labels;
    match answer {
        Answer::Choice(choice) => ordered
            .iter()
            .map(|label| choice.probabilities.get(label).copied())
            .collect(),
        Answer::Score(score) => ordered
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    let mut model_root = home.join(".cache/openkind/kev-0.6b");
    let mut base_root = home.join(".cache/openkind/qwen3-0.6b-base");
    let mut fit = false;
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => model_root = PathBuf::from(args.next().expect("--model-root value")),
            "--base-root" => base_root = PathBuf::from(args.next().expect("--base-root value")),
            "--fit" => fit = true,
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let engine = KevEngine::load(KevEngineConfig {
        model_root: model_root.clone(),
        base_root: base_root.clone(),
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
            "state": State::Text(case.state.as_str().expect("string state").to_owned()),
            "model": "fixture",
            "questions": { "q0": case.question },
        }))?;
        let request_value = serde_json::to_value(&request)?;
        let response = runtime.block_on(engine.evaluate(request))?;
        let answer = response.answers.get("q0").expect("answer for q0");
        let probability = correct_probability(case, answer)
            .ok_or_else(|| format!("case {} produced no correct-option probability", case.name))?;
        let distribution = distribution(&case.question, answer)
            .ok_or_else(|| format!("case {} produced an incomplete distribution", case.name))?;
        // The wire distribution follows the unpacked candidate order
        // (labels sorted for Choice), not the case declaration order.
        let truth_index = openkind_backends::families::wire::unpack_question("q0", &case.question)?
            .labels
            .iter()
            .position(|label| *label == case.truth_label)
            .expect("truth label is an offered candidate");
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
        let (temperature, nll) = {
            let scan = |from: f64, to: f64, step: f64| {
                let mut best_t = from;
                let mut best_nll = f64::INFINITY;
                let mut t = from;
                while t <= to {
                    let nll = mean_nll(&distributions, &truths, t);
                    if nll < best_nll {
                        best_nll = nll;
                        best_t = t;
                    }
                    t *= step;
                }
                (best_t, best_nll)
            };
            let (coarse, _) = scan(0.05, 20.0, 1.05);
            scan(coarse / 1.05, coarse * 1.05, 1.002)
        };
        println!("{temperature:.17}");
        eprintln!("[fit] mean NLL at optimum: {nll:.6}");
        eprintln!(
            "[fit] mean NLL at T=1.0:    {:.6}",
            mean_nll(&distributions, &truths, 1.0)
        );
        return Ok(());
    }

    let output = output.ok_or_else(|| "--output is required without --fit".to_owned())?;
    let case_values: Vec<serde_json::Value> = rendered_cases
        .into_iter()
        .map(|(name, request, response)| json!({ "name": name, "request": request, "response": response }))
        .collect();
    let golden = json!({
        "profile_id": PROFILE_ID,
        "backbone": { "id": BACKBONE_ID, "revision": BACKBONE_REVISION },
        "base_model": { "id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "calibration_temperature":
            openkind_backends::families::kev::CALIBRATION_TEMPERATURE,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-kev-fixture",
            "case_construction": "correct option stated verbatim in the state document",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
