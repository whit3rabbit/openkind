//! Offline fixture generator for the pinned `sieve-4b` profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_sieve_4b_fixture -- \
//!   --model-root <sieve-4b install root> \
//!   --output crates/openkind-backends/tests/fixtures/sieve_8b97b56524b55c41fbf4/golden.json
//! ```
//!
//! The golden replays document the engine's own behavior for regression
//! tests; external reference evidence is recorded in `docs/families/sieve.md`
//! (prompt byte-parity against the shipped `sieve4b/prompt.py` and the
//! LoRA merge's shared-backend semantics).

use std::path::PathBuf;

use openkind_backends::families::sieve::{
    SieveEngine, SieveEngineConfig, SieveRenderer, EXECUTION_ARITHMETIC_ID, FAMILY_SLUG, SIEVE_4B,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::{Question, SystemRequest};
use openkind_engine::DecisionEngine;
use serde_json::json;

/// One labeled fixture case with the engine-planned prompt inputs.
struct Case {
    name: String,
    request: serde_json::Value,
    state: serde_json::Value,
    /// (kind, instructions, option texts, listing order) per branch, in
    /// engine order (noul renders two branches).
    plans: Vec<(&'static str, serde_json::Value, Vec<String>, Vec<usize>)>,
}

fn choice_plan(
    instructions: &str,
    criteria: &[(&str, Option<&str>)],
) -> (&'static str, serde_json::Value, Vec<String>, Vec<usize>) {
    // Wire options render in the wire's sorted label order.
    let mut labels: Vec<&str> = criteria.iter().map(|(label, _)| *label).collect();
    labels.sort_unstable();
    let options = labels
        .iter()
        .map(
            |label| match criteria.iter().find(|(key, _)| key == label) {
                Some((_, Some(description))) if !description.is_empty() => {
                    format!("{label} — {description}")
                }
                _ => (*label).to_owned(),
            },
        )
        .collect();
    let listing = (0..labels.len()).collect();
    ("choice", serde_json::json!(instructions), options, listing)
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut push = |name: &str, state: serde_json::Value, question: Question, plans| {
        let request = json!({
            "state": state,
            "model": "fixture",
            "questions": {
                "q0": serde_json::to_value(question).expect("question")
            }
        });
        cases.push(Case {
            name: name.to_owned(),
            request,
            state,
            plans,
        });
    };

    push(
        "choice-two",
        json!("The kitchen light is on. The office light is off."),
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which option is explicitly recorded in the state?"),
            criteria: [
                (
                    "kitchen".to_owned(),
                    Some("the room whose light is on".to_owned()),
                ),
                ("office".to_owned(), None),
            ]
            .into_iter()
            .collect(),
        }),
        vec![choice_plan(
            "Which option is explicitly recorded in the state?",
            &[
                ("kitchen", Some("the room whose light is on")),
                ("office", None),
            ],
        )],
    );
    push(
        "choice-five-object-state",
        json!("Incident record: severity high, service billing, region eu-west, status mitigated, owner oncall."),
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which option is explicitly recorded in the state?"),
            criteria: [
                ("owner".to_owned(), None),
                ("region".to_owned(), None),
                ("service".to_owned(), None),
                ("severity".to_owned(), None),
                ("status".to_owned(), None),
            ]
            .into_iter()
            .collect(),
        }),
        vec![choice_plan(
            "Which option is explicitly recorded in the state?",
            &[
                ("owner", None),
                ("region", None),
                ("service", None),
                ("severity", None),
                ("status", None),
            ],
        )],
    );
    push(
        "noul-default-criteria",
        json!({"ticket": {"priority": "p1", "escalated": true}}),
        Question::Noul(openkind_core::NoulQuestion {
            instructions: json!("Is the ticket escalated?"),
            criteria: None,
        }),
        vec![
            (
                "noul",
                json!("Is the ticket escalated?"),
                vec!["No".to_owned(), "Yes".to_owned()],
                vec![1, 0],
            ),
            (
                "noul",
                json!("Is the ticket escalated?"),
                vec!["No".to_owned(), "Yes".to_owned()],
                vec![0, 1],
            ),
        ],
    );
    push(
        "noul-described-criteria",
        json!("Sensor report: temperature 41.2C, threshold 40C."),
        Question::Noul(openkind_core::NoulQuestion {
            instructions: json!("Did the temperature exceed the threshold?"),
            criteria: Some(openkind_core::NoulCriteria {
                r#true: "The temperature is above the threshold.".to_owned(),
                r#false: "The temperature is at or below the threshold.".to_owned(),
            }),
        }),
        vec![
            (
                "noul",
                json!("Did the temperature exceed the threshold?"),
                vec![
                    "No — The temperature is at or below the threshold.".to_owned(),
                    "Yes — The temperature is above the threshold.".to_owned(),
                ],
                vec![1, 0],
            ),
            (
                "noul",
                json!("Did the temperature exceed the threshold?"),
                vec![
                    "No — The temperature is at or below the threshold.".to_owned(),
                    "Yes — The temperature is above the threshold.".to_owned(),
                ],
                vec![0, 1],
            ),
        ],
    );
    push(
        "score-three-levels",
        json!("Queue depth is 420 events; the SLO allows up to 500 with a soft warn at 350."),
        Question::Score(openkind_core::ScoreQuestion {
            instructions: json!("How loaded is the queue?"),
            criteria: vec!["low".to_owned(), "moderate".to_owned(), "high".to_owned()],
        }),
        vec![(
            "score",
            json!("How loaded is the queue?"),
            vec!["low".to_owned(), "moderate".to_owned(), "high".to_owned()],
            vec![0, 1, 2],
        )],
    );
    push(
        "choice-empty-state",
        json!(""),
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Reply with any option."),
            criteria: [("alpha".to_owned(), None), ("beta".to_owned(), None)]
                .into_iter()
                .collect(),
        }),
        vec![choice_plan(
            "Reply with any option.",
            &[("alpha", None), ("beta", None)],
        )],
    );
    push(
        "choice-none-option",
        json!("The request matches neither the allowlist nor the denylist."),
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which list applies?"),
            criteria: [
                (
                    "__none__".to_owned(),
                    Some("neither list applies".to_owned()),
                ),
                ("allowlist".to_owned(), None),
                ("denylist".to_owned(), None),
            ]
            .into_iter()
            .collect(),
        }),
        vec![choice_plan(
            "Which list applies?",
            &[
                ("__none__", Some("neither list applies")),
                ("allowlist", None),
                ("denylist", None),
            ],
        )],
    );
    cases
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_root: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => {
                model_root = Some(PathBuf::from(args.next().expect("--model-root value")))
            }
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let model_root = model_root.ok_or("--model-root is required")?;
    let output = output.ok_or("--output is required")?;

    let engine = SieveEngine::load(SieveEngineConfig {
        model_root: model_root.clone(),
        limits: FamilyLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 0,
            retry_after_ms: 100,
            evaluation_timeout: None,
        },
        profile: &SIEVE_4B,
    })?;
    let renderer = SieveRenderer::load(
        &model_root.join("checkpoint").join("tokenizer.json"),
        &{
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(model_root.join("checkpoint/sieve_config.json"))
                    .expect("read sieve config"),
            )?;
            serde_json::from_value::<Vec<String>>(config["codes"].clone())?
        },
        &{
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(model_root.join("checkpoint/sieve_config.json"))
                    .expect("read sieve config"),
            )?;
            serde_json::from_value::<Vec<u64>>(config["code_ids"].clone())?
        },
        &SIEVE_4B,
    )?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let mut case_values = Vec::new();
    let mut prompt_stats = Vec::new();
    let built_cases = cases();
    for (index, case) in built_cases.iter().enumerate() {
        let request: SystemRequest = serde_json::from_value(case.request.clone())?;
        let request_value = serde_json::to_value(&request)?;
        let response = runtime.block_on(engine.evaluate(request.clone()))?;
        eprintln!("[fixture {:>2}] {} ok", index, case.name);
        case_values.push(json!({
            "name": case.name,
            "request": request_value,
            "response": serde_json::to_value(&response)?,
        }));

        // Render every branch through the engine's own planning path for
        // the reference-parity anchor.
        let prefix = renderer.render_prefix(&case.state)?;
        let mut branches = Vec::new();
        for (kind, instructions, options, listing) in &case.plans {
            let row = renderer.render_branch(&prefix, kind, instructions, options, listing)?;
            let digest = {
                use sha2::Digest as _;
                use std::fmt::Write as _;
                let joined = row
                    .prompt_ids()
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let mut hasher = sha2::Sha256::new();
                hasher.update(joined.as_bytes());
                let digest = hasher.finalize();
                let mut text = String::new();
                for byte in digest {
                    write!(&mut text, "{byte:02x}").expect("hex");
                }
                text
            };
            branches.push(json!({
                "token_count": row.prompt_ids().len(),
                "branch_ids_sha256": digest,
            }));
        }
        prompt_stats.push(json!({
            "name": case.name,
            "prefix_token_count": prefix.len(),
            "branches": branches,
        }));
    }

    let golden = json!({
        "profile_id": SIEVE_4B.profile_id,
        "family": FAMILY_SLUG,
        "backbone": {
            "id": SIEVE_4B.backbone_id,
            "revision": SIEVE_4B.backbone_revision,
        },
        "base": {
            "id": SIEVE_4B.base_id,
            "revision": SIEVE_4B.base_revision,
        },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-sieve-4b-fixture",
            "case_construction": "correct option stated verbatim in the state document; \
              noul cases answer in both listing orders and average, an empty-state case \
              exercises the fenced empty state, and a __none__ case scores the reserved \
              option as an ordinary candidate",
        },
        "cases": case_values,
        "prompt_stats": prompt_stats,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
