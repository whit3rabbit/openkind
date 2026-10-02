//! Offline fixture generator for the pinned `clef` profiles.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --release --bin gen_clef_fixture -- \
//!   --profile clef-flash-gguf \
//!   --model-root ~/.cache/openkind/models/clef-flash-gguf-q4 \
//!   --output crates/openkind-backends/tests/fixtures/clef_c330d9ee7e9cc658ad45/golden.json
//! ```
//!
//! The cases are synthetic decisions whose correct option is stated verbatim
//! in the state document; they exercise the joint-schema readout (Noul,
//! Choice, Score) and a multi-question record, which is the readout mode the
//! joint head is trained for.

use std::path::PathBuf;

use openkind_backends::families::clef::{profile_by_loader_id, ClefEngine};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::{Answer, SystemRequest};
use openkind_engine::DecisionEngine;
use serde_json::json;

struct Args {
    profile: String,
    model_root: PathBuf,
    output: PathBuf,
}

impl Args {
    fn parse() -> Self {
        let mut profile = None;
        let mut model_root = None;
        let mut output = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--profile" => profile = Some(args.next().expect("--profile value")),
                "--model-root" => model_root = Some(args.next().expect("--model-root value")),
                "--output" => output = Some(args.next().expect("--output value")),
                other => panic!("unknown argument {other}"),
            }
        }
        Self {
            profile: profile.expect("--profile"),
            model_root: PathBuf::from(model_root.expect("--model-root")),
            output: PathBuf::from(output.expect("--output")),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let profile = profile_by_loader_id(&args.profile)
        .ok_or_else(|| format!("unknown clef profile {}", args.profile))?;

    let requests = vec![
        // One multi-question record exercising all three primitives jointly
        // — the joint head's designed operating mode.
        json!({
            "model": "clef-fixture",
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
                        "security": "Account takeover, phishing",
                        "performance": "Latency and outages"
                    }
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate operational severity.",
                    "criteria": ["low", "moderate", "high", "critical"]
                }
            }
        }),
        // A stated-verbatim choice over four options.
        json!({
            "model": "clef-fixture",
            "state": "Incident log: the failing component is the scheduler. All other components healthy.",
            "questions": {
                "failing_component": {
                    "type": "choice",
                    "instructions": "Which component is failing?",
                    "criteria": {
                        "api_gateway": "Front door",
                        "scheduler": "Job scheduler",
                        "storage": "Object storage"
                    }
                }
            }
        }),
        // Text passthrough state (a string, not JSON).
        json!({
            "model": "clef-fixture",
            "state": "The customer explicitly declined the renewal offer on 2026-09-14.",
            "questions": {
                "renewal": {
                    "type": "noul",
                    "instructions": "Did the customer decline the renewal?"
                }
            }
        }),
    ];

    let requests = requests
        .into_iter()
        .map(serde_json::from_value::<SystemRequest>)
        .collect::<Result<Vec<_>, _>>()?;

    let engine = ClefEngine::load(
        &args.model_root,
        profile,
        FamilyLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 0,
            retry_after_ms: 250,
            evaluation_timeout: None,
        },
    )?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut case_outputs = Vec::with_capacity(requests.len());
    for request in &requests {
        let response = runtime.block_on(engine.evaluate(request.clone()))?;
        case_outputs.push(serde_json::to_value(&response)?);
        for (id, answer) in &response.answers {
            match answer {
                Answer::Choice(choice) => {
                    println!("{id}: choice={} p={:.4}", choice.choice, choice.confidence)
                }
                Answer::Noul(noul) => println!("{id}: noul={:.4}", noul.noul),
                Answer::Score(score) => println!(
                    "{id}: score={:.4} confidence={:.4}",
                    score.score, score.confidence
                ),
            }
        }
        println!("---");
    }

    let golden = json!({
        "family": "clef",
        "profile": profile.loader_id,
        "profile_id": profile.profile_id,
        "backbone": profile.backbone_id,
        "backbone_revision": profile.backbone_revision,
        "arithmetic": profile.cpu_backend_id,
        "cases": case_outputs,
    });
    std::fs::create_dir_all(args.output.parent().ok_or("output path has no parent")?)?;
    std::fs::write(
        &args.output,
        serde_json::to_string_pretty(&golden)?.as_bytes(),
    )?;
    println!("wrote {}", args.output.display());
    Ok(())
}
