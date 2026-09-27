//! Offline fixture generator and calibration helper for the pinned
//! `encoder-instruct-label` profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_encoder_instruct_label_fixture -- \
//!   --model-root ~/.cache/openkind/gliclass-modern-base-v3.0 --fit
//! # pin the printed temperature in families/encoder_instruct_label/mod.rs
//! cargo run -p openkind-backends --bin gen_encoder_instruct_label_fixture -- \
//!   --model-root ~/.cache/openkind/gliclass-modern-base-v3.0 \
//!   --output crates/openkind-backends/tests/fixtures/encoder_instruct_label_<profile>/golden.json
//! ```
//!
//! The calibration set is a fixed list of synthetic stated-fact decision
//! cases whose correct option is stated verbatim in the state document. The
//! fit re-derives the exact wire distribution of each primitive
//! (renormalized sigmoids for `Choice`, softmax over level logits for
//! `Score`, the support sigmoid for `Noul`) at candidate temperatures, so
//! the pinned temperature is optimal for the shipped readout, not for raw
//! logits.

use std::path::PathBuf;

use openkind_backends::families::encoder_instruct_label::model::{
    EncoderInstructLabelModel, VerifiedArtifacts,
};
use openkind_backends::families::encoder_instruct_label::renderer::EncoderInstructLabelRenderer;
use openkind_backends::families::encoder_instruct_label::{
    EncoderInstructLabelEngine, EncoderInstructLabelEngineConfig, BACKBONE_ID, BACKBONE_REVISION,
    EXECUTION_ARITHMETIC_ID, PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_backends::families::wire::unpack_question;
use openkind_core::{Question, State};
use openkind_engine::DecisionEngine;
use serde_json::json;

/// One labeled calibration case.
struct Case {
    name: String,
    /// Candidate label carrying the correct answer.
    truth_label: String,
    state: String,
    question: Question,
}

fn choice_case(name: &str, state: &str, options: &[&str], truth: usize) -> Case {
    Case {
        name: name.to_owned(),
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
        truth_label: if truth { "true".into() } else { "false".into() },
        state: state.to_owned(),
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
        state: state.to_owned(),
        question: Question::Score(openkind_core::ScoreQuestion {
            instructions: json!("What is the documented impact level?"),
            criteria: levels.iter().map(|level| level.to_string()).collect(),
        }),
    }
}

/// The pinned stated-fact calibration set. Score levels carry meaningful
/// level words: the marker readout scores criterion text against the state,
/// so a state saying "major" discriminates the "major" marker, not "level 2".
fn cases() -> Vec<Case> {
    vec![
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
            "owner-a",
            "Fictional incident record. The owner of record is the platform team. These statements are the full evidence record; do not infer missing facts.",
            &["security team", "platform team", "external vendor", "data team"],
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
            "Fictional incident record. The record does not mention external destinations. These statements are the full evidence record; do not infer missing facts.",
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
        noul_case(
            "patch-true",
            "Fictional incident record. The patch level is current as of the report date. These statements are the full evidence record; do not infer missing facts.",
            "Is the patch level described as current?",
            true,
        ),
        score_case(
            "impact-minor",
            "Fictional incident record. The documented impact level is minor. These statements are the full evidence record; do not infer missing facts.",
            &["minor", "moderate", "major", "critical"],
            0,
        ),
        score_case(
            "impact-moderate",
            "Fictional incident record. The documented impact level is moderate. These statements are the full evidence record; do not infer missing facts.",
            &["minor", "moderate", "major", "critical"],
            1,
        ),
        score_case(
            "impact-major",
            "Fictional incident record. The documented impact level is major. These statements are the full evidence record; do not infer missing facts.",
            &["minor", "moderate", "major", "critical"],
            2,
        ),
        score_case(
            "impact-critical",
            "Fictional incident record. The documented impact level is critical. These statements are the full evidence record; do not infer missing facts.",
            &["minor", "moderate", "major", "critical"],
            3,
        ),
    ]
}

/// Wire readout of one question at temperature `t` from raw marker logits.
/// Mirrors the engine's evaluate path exactly.
fn wire_distribution(
    primitive: openkind_backends::families::wire::QuestionPrimitive,
    logits: &[f64],
    t: f64,
) -> Vec<f64> {
    match primitive {
        openkind_backends::families::wire::QuestionPrimitive::Noul => {
            let logit = logits[0];
            let scaled = logit / t;
            let support = 1.0 / (1.0 + (-scaled).exp());
            vec![1.0 - support, support]
        }
        openkind_backends::families::wire::QuestionPrimitive::Choice => {
            let supports: Vec<f64> = logits
                .iter()
                .map(|logit| 1.0 / (1.0 + (-(logit / t)).exp()))
                .collect();
            let sum: f64 = supports.iter().sum();
            supports.iter().map(|support| support / sum).collect()
        }
        openkind_backends::families::wire::QuestionPrimitive::Score => {
            let scaled: Vec<f64> = logits.iter().map(|logit| logit / t).collect();
            let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f64> = scaled.iter().map(|value| (value - max).exp()).collect();
            let sum: f64 = exponentials.iter().sum();
            exponentials.iter().map(|value| value / sum).collect()
        }
    }
}

/// One case reduced for the temperature fit.
struct FittedCase {
    primitive: openkind_backends::families::wire::QuestionPrimitive,
    logits: Vec<f64>,
    truth_index: usize,
}

/// Mean NLL of the correct option across all cases at temperature `t`.
fn mean_nll(per_case: &[FittedCase], t: f64) -> f64 {
    let mut total = 0.0;
    for case in per_case {
        let distribution = wire_distribution(case.primitive, &case.logits, t);
        total -= distribution[case.truth_index].ln();
    }
    total / per_case.len() as f64
}

/// Two-pass multiplicative grid search over temperature.
fn fit_temperature(per_case: &[FittedCase]) -> f64 {
    let scan = |from: f64, to: f64, step: f64| {
        let mut best_t = from;
        let mut best_nll = f64::INFINITY;
        let mut t = from;
        while t <= to {
            let nll = mean_nll(per_case, t);
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
        .join(".cache/openkind/gliclass-modern-base-v3.0");
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

    let limits = FamilyLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 0,
        retry_after_ms: 100,
        evaluation_timeout: None,
    };
    let artifacts = VerifiedArtifacts::verify(&model_root)?;
    let renderer = EncoderInstructLabelRenderer::load(&artifacts.tokenizer)?;
    let model = EncoderInstructLabelModel::load(&artifacts)?;
    let engine = EncoderInstructLabelEngine::load(EncoderInstructLabelEngineConfig {
        model_root: model_root.clone(),
        limits,
    })?;

    let case_list = cases();
    let mut per_case: Vec<FittedCase> = Vec::new();
    let mut rendered_cases = Vec::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    for (index, case) in case_list.iter().enumerate() {
        let question = &case.question;
        let unpacked = unpack_question("q0", question)?;
        let instruction = openkind_backends::families::wire::instruction_text(question)?;
        let explicit = matches!(question, Question::Noul(noul) if noul.criteria.is_some());
        let markers: Vec<String> = match unpacked.primitive {
            openkind_backends::families::wire::QuestionPrimitive::Noul => {
                if explicit {
                    vec![unpacked
                        .labels
                        .iter()
                        .position(|label| label == "true")
                        .and_then(|index| unpacked.criteria.get(index))
                        .expect("noul unpacking provides a true criterion")
                        .clone()]
                } else {
                    vec![instruction.clone()]
                }
            }
            _ => unpacked.criteria.clone(),
        };
        let ids = renderer.render(&markers, &case.state)?;
        let logits = model.marker_logits(&ids)?;
        // The wire distribution follows the unpacked candidate order
        // (labels sorted for Choice), not the case declaration order.
        let truth_index = unpacked
            .labels
            .iter()
            .position(|label| *label == case.truth_label)
            .expect("truth label is an offered candidate");
        let correct = wire_distribution(unpacked.primitive, &logits, 1.0)[truth_index];
        eprintln!(
            "[fixture {:>2}] {:<16} p(correct)@T=1 = {correct:.6}",
            index, case.name
        );
        per_case.push(FittedCase {
            primitive: unpacked.primitive,
            logits,
            truth_index,
        });

        // Full engine pass for the golden response (at the pinned constant).
        let request: openkind_core::SystemRequest = serde_json::from_value(json!({
            "state": State::Text(case.state.clone()),
            "model": "fixture",
            "questions": { "q0": case.question },
        }))?;
        let request_value = serde_json::to_value(&request)?;
        let response = runtime.block_on(engine.evaluate(request.clone()))?;
        rendered_cases.push((
            case.name.clone(),
            request_value,
            serde_json::to_value(&response)?,
        ));
    }

    if fit {
        let temperature = fit_temperature(&per_case);
        println!("{temperature:.17}");
        eprintln!(
            "[fit] mean NLL at optimum: {:.6}",
            mean_nll(&per_case, temperature)
        );
        eprintln!(
            "[fit] mean NLL at T=1.0:    {:.6}",
            mean_nll(&per_case, 1.0)
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
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "calibration_temperature":
            openkind_backends::families::encoder_instruct_label::CALIBRATION_TEMPERATURE,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-encoder-instruct-label-fixture",
            "case_construction": "correct option stated verbatim in the state document",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
