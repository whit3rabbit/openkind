//! Offline fixture generator for the pinned `drex-v15` profile (Drex v1.5).
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --release --bin gen_drex_fixture -- \
//!   --model-root ~/.cache/openkind/drex-v1.5 \
//!   --output crates/openkind-backends/tests/fixtures/drex_2d495e07ea056d6c4317/golden.json
//! ```
//!
//! The cases are synthetic decisions whose correct option is stated verbatim
//! in the state document. They cover Noul (default and explicit criteria),
//! Choice (bare keys and descriptions), Score, and one structured JSON state
//! with several independent questions. Choice criteria are written in sorted
//! key order so the bundled Python reference, which keeps caller order,
//! scores the same option order as this engine.

use std::path::PathBuf;

use openkind_backends::families::drex::{
    DrexEngine, DrexEngineConfig, BACKBONE_ID, BACKBONE_REVISION, CALIBRATION_TEMPERATURE,
    EXECUTION_ARITHMETIC_ID, PROFILE_ID,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::SystemRequest;
use openkind_engine::DecisionEngine;
use serde_json::json;

fn cases() -> Vec<(&'static str, serde_json::Value)> {
    let record = "Fictional incident record.";
    vec![
        (
            "choice-os",
            json!({
                "state": format!("{record} The asset operating system is Windows."),
                "questions": { "q0": {
                    "type": "choice",
                    "instructions": "Which option is explicitly recorded in the state?",
                    "criteria": { "BSD": null, "Linux": null, "Windows": null, "macOS": null }
                }}
            }),
        ),
        (
            "choice-region",
            json!({
                "state": format!("{record} The affected deployment runs in eu-west-2."),
                "questions": { "q0": {
                    "type": "choice",
                    "instructions": "Which region does the affected deployment run in?",
                    "criteria": { "ap-southeast-1": null, "eu-west-2": null, "us-east-1": null }
                }}
            }),
        ),
        (
            "choice-described",
            json!({
                "state": format!("{record} A user reported a phishing email that captured their password."),
                "questions": { "q0": {
                    "type": "choice",
                    "instructions": "Which category does the report belong to?",
                    "criteria": {
                        "billing": "Charges, refunds, invoices",
                        "performance": "Latency and outages",
                        "security": "Account takeover, phishing"
                    }
                }}
            }),
        ),
        (
            "noul-true",
            json!({
                "state": format!("{record} Two-factor authentication was enabled on the affected account."),
                "questions": { "q0": {
                    "type": "noul",
                    "instructions": "Does the record state that two-factor authentication was enabled?"
                }}
            }),
        ),
        (
            "noul-false",
            json!({
                "state": format!("{record} No data export was approved during the window."),
                "questions": { "q0": {
                    "type": "noul",
                    "instructions": "Does the record state that a data export was approved?"
                }}
            }),
        ),
        (
            "noul-criteria",
            json!({
                "state": format!("{record} The message asks the user to wire money to an unknown account."),
                "questions": { "q0": {
                    "type": "noul",
                    "instructions": "Classify the message.",
                    "criteria": { "false": "a legitimate request", "true": "a likely scam" }
                }}
            }),
        ),
        (
            "score-minor",
            json!({
                "state": format!("{record} The documented impact level is minor."),
                "questions": { "q0": {
                    "type": "score",
                    "instructions": "What is the documented impact level?",
                    "criteria": ["minor", "moderate", "major", "critical"]
                }}
            }),
        ),
        (
            "score-critical",
            json!({
                "state": format!("{record} The documented impact level is critical."),
                "questions": { "q0": {
                    "type": "score",
                    "instructions": "What is the documented impact level?",
                    "criteria": ["minor", "moderate", "major", "critical"]
                }}
            }),
        ),
        (
            "structured-multi",
            json!({
                "state": {
                    "ticket": "A-771",
                    "priority": "high",
                    "category": "billing",
                    "refund_approved": true,
                    "amount_usd": 412.5
                },
                "questions": {
                    "billing_refund": {
                        "type": "noul",
                        "instructions": "Was the refund approved?"
                    },
                    "category": {
                        "type": "choice",
                        "instructions": "Which category does the ticket belong to?",
                        "criteria": {
                            "billing": "Charges, refunds, invoices",
                            "performance": "Latency and outages",
                            "security": "Account takeover, phishing"
                        }
                    },
                    "severity": {
                        "type": "score",
                        "instructions": "Rate the ticket priority.",
                        "criteria": ["low", "medium", "high", "urgent"]
                    }
                }
            }),
        ),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let mut model_root = home.join(".cache/openkind/drex-v1.5");
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => model_root = PathBuf::from(args.next().expect("--model-root value")),
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let output = output.ok_or("--output is required")?;

    let engine = DrexEngine::load(DrexEngineConfig {
        model_root,
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

    let mut rendered = Vec::new();
    for (name, mut request) in cases() {
        request["model"] = json!("fixture");
        let request: SystemRequest = serde_json::from_value(request)?;
        let request_value = serde_json::to_value(&request)?;
        let started = std::time::Instant::now();
        let response = runtime.block_on(engine.evaluate(request))?;
        eprintln!(
            "[fixture] {name:<18} {:>7.1} s  {}",
            started.elapsed().as_secs_f64(),
            serde_json::to_string(&response.answers)?
        );
        rendered.push(json!({
            "name": name,
            "request": request_value,
            "response": serde_json::to_value(&response)?,
        }));
    }

    let golden = json!({
        "profile_id": PROFILE_ID,
        "backbone": { "id": BACKBONE_ID, "revision": BACKBONE_REVISION },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "calibration_temperature": CALIBRATION_TEMPERATURE,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-drex-fixture",
            "case_construction": "correct option stated verbatim in the state document",
        },
        "cases": rendered,
    });
    std::fs::create_dir_all(output.parent().ok_or("output has no parent")?)?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
