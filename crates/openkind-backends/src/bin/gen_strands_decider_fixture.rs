//! Offline fixture generator for the pinned `strands-decider-2b` profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_strands_decider_fixture -- \
//!   --model-root <strands-decider-2b install>/adapter \
//!   --base-root <strands-decider-2b install>/base \
//!   --output crates/openkind-backends/tests/fixtures/strands_decider_6a02bb0d1c6b25cae74b/golden.json
//! ```
//!
//! No `--fit` temperature search exists here (unlike `gen_kev_fixture`):
//! the per-type temperatures ship fitted in the release's
//! `hobson_config.json` and are pinned verbatim in
//! `families/strands_decider/mod.rs`.

use std::path::PathBuf;

use openkind_backends::families::strands_decider::{
    StrandsDeciderEngine, StrandsDeciderEngineConfig, BACKBONE_ID, BACKBONE_REVISION,
    BASE_MODEL_ID, BASE_MODEL_REVISION, EXECUTION_ARITHMETIC_ID, PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::{Question, SystemRequest};
use openkind_engine::DecisionEngine;
use serde_json::json;

/// One labeled fixture case.
struct Case {
    name: String,
    state: serde_json::Value,
    questions: serde_json::Map<String, serde_json::Value>,
}

fn single_question(name: &str, state: serde_json::Value, question: Question) -> Case {
    Case {
        name: name.to_owned(),
        state,
        questions: std::iter::once((
            "q0".to_owned(),
            serde_json::to_value(question).expect("question"),
        ))
        .collect(),
    }
}

fn choice_case(name: &str, state: &str, options: &[&str], descriptions: &[Option<&str>]) -> Case {
    single_question(
        name,
        json!(state),
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which option is explicitly recorded in the state?"),
            criteria: options
                .iter()
                .zip(descriptions.iter().chain(std::iter::repeat(&None)))
                .map(|(option, description)| {
                    (option.to_string(), (*description).map(str::to_owned))
                })
                .collect(),
        }),
    )
}

fn noul_case(name: &str, state: &str, instruction: &'static str) -> Case {
    single_question(
        name,
        json!(state),
        Question::Noul(openkind_core::NoulQuestion {
            instructions: json!(instruction),
            criteria: None,
        }),
    )
}

fn noul_criteria_case(name: &str, state: &str, instruction: &'static str) -> Case {
    single_question(
        name,
        json!(state),
        Question::Noul(openkind_core::NoulQuestion {
            instructions: json!(instruction),
            criteria: Some(openkind_core::NoulCriteria {
                r#true: "the record states the export was approved".into(),
                r#false: "the record does not state an approved export".into(),
            }),
        }),
    )
}

fn score_case(name: &str, state: &str, levels: &[&str]) -> Case {
    single_question(
        name,
        json!(state),
        Question::Score(openkind_core::ScoreQuestion {
            instructions: json!("What is the documented impact level?"),
            criteria: levels.iter().map(|level| level.to_string()).collect(),
        }),
    )
}

/// The pinned stated-fact calibration set: every case states its correct
/// option verbatim in the state document, so the truth label is
/// deterministic without any model inference at generation time.
fn cases() -> Vec<Case> {
    let impact_levels = ["minor", "moderate", "major", "critical"];
    vec![
        choice_case(
            "os-a",
            "Fictional incident record. The asset operating system is Windows.",
            &["Linux", "Windows", "macOS", "BSD"],
            &[],
        ),
        choice_case(
            "os-b",
            "Fictional incident record. The asset operating system is Linux.",
            &["Linux", "Windows Server 2022", "macOS", "Solaris"],
            &[],
        ),
        choice_case(
            "region-a",
            "Fictional incident record. The affected deployment runs in us-east-1.",
            &["us-east-1", "eu-west-2", "ap-southeast-1", "us-west-1"],
            &[],
        ),
        choice_case(
            "protocol-a",
            "Fictional incident record. The observed connection used SSH.",
            &["HTTPS", "SSH", "SMTP", "DNS"],
            &[],
        ),
        choice_case(
            "owner-a",
            "Fictional incident record. The owner of record is the platform team.",
            &["external vendor", "platform team", "security team"],
            &[],
        ),
        choice_case(
            "descriptions",
            "Fictional incident record. The routing layer involved is the gateway — the edge routing product.",
            &["gateway", "workstation"],
            &[Some("the edge routing product"), Some("an endpoint device")],
        ),
        noul_case(
            "external-true",
            "Fictional incident record. An external destination was observed in the connection logs.",
            "Does the record explicitly state that an external destination was observed?",
        ),
        noul_case(
            "external-false",
            "Fictional incident record. The record does not mention external destinations.",
            "Does the record explicitly state that an external destination was observed?",
        ),
        noul_case(
            "mfa-true",
            "Fictional incident record. The record states that two-factor authentication was enabled on the affected account.",
            "Does the record explicitly state that two-factor authentication was enabled?",
        ),
        noul_case(
            "export-false",
            "Fictional incident record. No data export was approved during the window.",
            "Was a data export approved during the window?",
        ),
        noul_criteria_case(
            "export-criteria",
            "Fictional incident record. The record states a data export was approved on Tuesday.",
            "Was a data export approved?",
        ),
        score_case(
            "impact-minor",
            "Fictional incident record. The documented impact level is minor.",
            &impact_levels,
        ),
        score_case(
            "impact-moderate",
            "Fictional incident record. The documented impact level is moderate.",
            &impact_levels,
        ),
        score_case(
            "impact-major",
            "Fictional incident record. The documented impact level is major.",
            &impact_levels,
        ),
        score_case(
            "impact-critical",
            "Fictional incident record. The documented impact level is critical.",
            &impact_levels,
        ),
        Case {
            name: "structured-multi".to_owned(),
            state: json!({
                "record": "Fictional incident record",
                "affected_asset": "application server",
                "operating_system": "Linux",
                "impact_level": "moderate",
                "data_export_approved": false
            }),
            questions: [
                (
                    "os".to_owned(),
                    serde_json::to_value(Question::Choice(openkind_core::ChoiceQuestion {
                        instructions: json!("Which operating system does the record state?"),
                        criteria: [
                            ("Linux".to_owned(), None),
                            ("Windows".to_owned(), None),
                            ("macOS".to_owned(), None),
                        ]
                        .into_iter()
                        .collect(),
                    }))
                    .expect("question"),
                ),
                (
                    "export".to_owned(),
                    serde_json::to_value(Question::Noul(openkind_core::NoulQuestion {
                        instructions: json!(
                            "Does the record state that a data export was approved?"
                        ),
                        criteria: None,
                    }))
                    .expect("question"),
                ),
            ]
            .into_iter()
            .collect(),
        },
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_root: Option<PathBuf> = None;
    let mut base_root: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => {
                model_root = Some(PathBuf::from(args.next().expect("--model-root value")))
            }
            "--base-root" => {
                base_root = Some(PathBuf::from(args.next().expect("--base-root value")))
            }
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let model_root = model_root.ok_or("--model-root is required")?;
    let base_root = base_root.ok_or("--base-root is required")?;
    let output = output.ok_or("--output is required")?;

    let engine = StrandsDeciderEngine::load(StrandsDeciderEngineConfig {
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

    let mut case_values = Vec::new();
    for (index, case) in cases().into_iter().enumerate() {
        let request: SystemRequest = serde_json::from_value(json!({
            "state": case.state,
            "model": "fixture",
            "questions": case.questions,
        }))?;
        let request_value = serde_json::to_value(&request)?;
        let response = runtime.block_on(engine.evaluate(request))?;
        eprintln!("[fixture {:>2}] {} ok", index, case.name);
        case_values.push(json!({
            "name": case.name,
            "request": request_value,
            "response": serde_json::to_value(&response)?,
        }));
    }

    let golden = json!({
        "profile_id": PROFILE_ID,
        "backbone": { "id": BACKBONE_ID, "revision": BACKBONE_REVISION },
        "base_model": { "id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-strands-decider-fixture",
            "case_construction": "correct option stated verbatim in the state document",
            "reference_crosscheck": "17/17 argmax agreement against the pinned PyTorch \
              reference (transformers 5.17.0 + peft 0.21.1, CPU fp32) on identical \
              prompts; probability drift <= 8.5e-4 on Noul and Score, and 2e-5..5.9e-2 \
              on Choice (one-sided softening amplified by the 0.734 choice temperature) \
              from cross-implementation hybrid-recurrence arithmetic",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
