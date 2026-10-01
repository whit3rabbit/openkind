//! Offline fixture generator for the pinned `decoder-logit-qwen35` profiles.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --release --bin gen-decoder-logit-qwen35-fixture -- \
//!   --profile decoder-logit-qwen35 \
//!   --model-root ~/.cache/openkind/jevk5 \
//!   --output crates/openkind-backends/tests/fixtures/decoder_logit_qwen35_415bcf4a064e6dadcf85/golden.json
//! ```
//!
//! The cases are synthetic decisions whose correct option is stated verbatim
//! in the state document; they exercise the letter readout (Choice, Noul,
//! Score) and, for knockout profiles, the knockout combination (a 17-option
//! Choice spanning two groups plus a final pass). The `plumb-4b` profile
//! serves at most 16 options in one pass, so its wide case stays inside one
//! read. Temperatures are the pinned runtime values, so no calibration fit
//! runs here.

use std::path::PathBuf;

use openkind_backends::families::decoder_logit_qwen35::{
    profile_by_loader_id, DecoderLogitQwen35Engine, DecoderLogitQwen35EngineConfig,
    EXECUTION_ARITHMETIC_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::{Answer, Question, SystemRequest};
use openkind_engine::DecisionEngine;
use serde_json::json;

struct Case {
    name: String,
    truth_label: String,
    state: serde_json::Value,
    question: Question,
}

fn text_case(name: &str, state: &str, truth: &str, question: Question) -> Case {
    Case {
        name: name.to_owned(),
        truth_label: truth.to_owned(),
        state: json!(state),
        question,
    }
}

fn object_case(name: &str, state: serde_json::Value, truth: &str, question: Question) -> Case {
    Case {
        name: name.to_owned(),
        truth_label: truth.to_owned(),
        state,
        question,
    }
}

fn stated_choice(name: &str, state: &str, options: &[&str], truth: usize) -> Case {
    text_case(
        name,
        state,
        options[truth],
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which option is explicitly recorded in the state?"),
            criteria: options
                .iter()
                .map(|option| (option.to_string(), None))
                .collect(),
        }),
    )
}

fn stated_noul(name: &str, state: &str, instruction: &str, truth: bool) -> Case {
    text_case(
        name,
        state,
        if truth { "true" } else { "false" },
        Question::Noul(openkind_core::NoulQuestion {
            instructions: json!(instruction),
            criteria: None,
        }),
    )
}

fn stated_score(name: &str, state: &str, levels: usize, truth: usize) -> Case {
    text_case(
        name,
        state,
        &truth.to_string(),
        Question::Score(openkind_core::ScoreQuestion {
            instructions: json!("What is the documented impact level?"),
            criteria: (0..levels).map(|level| format!("level {level}")).collect(),
        }),
    )
}

fn cases(with_knockout: bool) -> Vec<Case> {
    let mut cases = vec![
        stated_choice(
            "os",
            "Fictional incident record. The asset operating system is Linux. These statements are the full evidence record; do not infer missing facts.",
            &["Linux", "Windows", "macOS", "BSD"],
            0,
        ),
        stated_choice(
            "region",
            "Fictional incident record. The affected deployment runs in ap-southeast-1. These statements are the full evidence record; do not infer missing facts.",
            &["us-east-1", "eu-west-2", "ap-southeast-1", "us-west-1"],
            2,
        ),
        stated_noul(
            "mfa-true",
            "Fictional incident record. The record states that two-factor authentication was enabled on the affected account. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that two-factor authentication was enabled?",
            true,
        ),
        stated_noul(
            "export-false",
            "Fictional incident record. No data export was approved during the window. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that a data export was approved?",
            false,
        ),
        stated_score(
            "impact-3",
            "Fictional incident record. The documented impact level is 3. These statements are the full evidence record; do not infer missing facts.",
            5,
            3,
        ),
        stated_score(
            "severity-1",
            "Fictional incident record. The documented severity is 1. These statements are the full evidence record; do not infer missing facts.",
            3,
            1,
        ),
        // A structured state exercises the JSON evidence payload.
        object_case(
            "refund-json",
            json!({
                "ticket": "Fictional billing ticket. The customer was billed twice for the same order and asks for the duplicate charge back.",
                "policy": "Refunds are permitted for duplicate charges.",
            }),
            "true",
            Question::Noul(openkind_core::NoulQuestion {
                instructions: json!("Does the record support permitting the refund?"),
                criteria: None,
            }),
        ),
        // Explicit Noul criteria flow into the option descriptions.
        text_case(
            "compromise-criteria",
            "Fictional incident record. The record states that the server was compromised. These statements are the full evidence record; do not infer missing facts.",
            "true",
            Question::Noul(openkind_core::NoulQuestion {
                instructions: json!("Was the server compromised according to the record?"),
                criteria: Some(openkind_core::NoulCriteria {
                    r#true: "The record states the server was compromised.".into(),
                    r#false: "The record does not state the server was compromised.".into(),
                }),
            }),
        ),
    ];
    // A wide Choice: knockout profiles run two groups (9 + 8) plus a
    // 16-finalist final pass; single-read profiles cap at 16 options in one
    // pass, so their wide case stays within the pass width.
    let wide_count = if with_knockout { 17 } else { 16 };
    let wide_options: Vec<String> = (0..wide_count)
        .map(|index| format!("site-{index:02}"))
        .collect();
    cases.push(text_case(
        "wide-choice",
        "Fictional deployment record. The affected site is site-13. These statements are the full evidence record; do not infer missing facts.",
        "site-13",
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which site is explicitly recorded in the state?"),
            criteria: wide_options
                .iter()
                .map(|option| (option.clone(), None))
                .collect(),
        }),
    ));
    cases
}

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile_id = "decoder-logit-qwen35".to_owned();
    let mut model_root =
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/openkind/jevk5");
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => profile_id = args.next().expect("--profile value"),
            "--model-root" => model_root = PathBuf::from(args.next().expect("--model-root value")),
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let output = output.ok_or_else(|| "--output is required".to_owned())?;
    let profile = profile_by_loader_id(&profile_id)
        .ok_or_else(|| format!("unknown profile loader id {profile_id}"))?;

    let engine = DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
        profile,
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

    let case_list = cases(profile.knockout_temperature.is_some());
    let mut rendered_cases = Vec::new();
    for (index, case) in case_list.iter().enumerate() {
        let request: SystemRequest = serde_json::from_value(json!({
            "state": case.state,
            "model": "fixture",
            "questions": { "q0": case.question },
        }))?;
        let request_value = serde_json::to_value(&request)?;
        let started = std::time::Instant::now();
        let response = runtime.block_on(engine.evaluate(request))?;
        let answer = response.answers.get("q0").expect("answer for q0");
        let probability = correct_probability(case, answer)
            .ok_or_else(|| format!("case {} produced no correct-option probability", case.name))?;
        eprintln!(
            "[fixture {:>2}] {:<22} p(correct) = {probability:.6}  ({:.1}s)",
            index,
            case.name,
            started.elapsed().as_secs_f32(),
        );
        rendered_cases.push((
            case.name.clone(),
            request_value,
            serde_json::to_value(&response)?,
        ));
    }

    let case_values: Vec<serde_json::Value> = rendered_cases
        .into_iter()
        .map(|(name, request, response)| json!({ "name": name, "request": request, "response": response }))
        .collect();
    let calibration = match profile.calibration {
        openkind_backends::families::decoder_logit_qwen35::Calibration::Uniform(temperature) => {
            json!({ "uniform": temperature })
        }
        openkind_backends::families::decoder_logit_qwen35::Calibration::ByType {
            choice,
            score,
            noul,
        } => json!({ "choice": choice, "score": score, "noul": noul }),
    };
    let golden = json!({
        "profile_id": profile.profile_id,
        "loader_id": profile.loader_id,
        "backbone": { "id": profile.backbone_id, "revision": profile.backbone_revision },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "calibration": calibration,
        "knockout_temperature": profile.knockout_temperature,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-decoder-logit-qwen35-fixture",
            "case_construction": "correct option stated verbatim in the state document",
            "reference_runtime": "github.com/allebee/jevk5 prompt.py (pinned bytes)",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
