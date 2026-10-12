//! Offline fixture generator for the pinned `d3-edge` profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_d3_edge_fixture -- \
//!   --model-root <d3-edge model root> \
//!   --output crates/openkind-backends/tests/fixtures/d3_664f9ead734372935286/golden.json
//! ```
//!
//! The golden replays document the engine's own behavior for regression
//! tests; external reference evidence is recorded in
//! `docs/families/d3.md` (tokenizer/prompt byte-parity against the shipped
//! `d3_format.py` and the noncausal kernel's naive-reference test).

use std::path::PathBuf;

use openkind_backends::families::d3::{
    profile_by_short_name, D3Engine, D3EngineConfig, D3Renderer, EXECUTION_ARITHMETIC_ID,
    FAMILY_SLUG,
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
    /// The engine-planned prompt inputs of the case's single question: the
    /// state and instruction values plus the option texts in engine code
    /// order (wire Choice renders the wire's sorted label order; Noul the
    /// fixed false/true pair; Score the ordered levels).
    prompt: PromptPlan,
}

struct PromptPlan {
    state: serde_json::Value,
    instructions: serde_json::Value,
    options: Vec<String>,
}

fn single_question(name: &str, state: serde_json::Value, question: Question) -> Case {
    let prompt_state = state.clone();
    let prompt_instructions = match &question {
        Question::Choice(inner) => inner.instructions.clone(),
        Question::Noul(inner) => inner.instructions.clone(),
        Question::Score(inner) => inner.instructions.clone(),
    };
    let prompt_options = match &question {
        Question::Choice(inner) => {
            let mut labels: Vec<&String> = inner.criteria.keys().collect();
            labels.sort();
            labels
                .iter()
                .map(|label| match inner.criteria.get(*label) {
                    Some(Some(description)) => format!("{label}: {description}"),
                    _ => (*label).clone(),
                })
                .collect()
        }
        Question::Noul(inner) => {
            let (r#false, r#true) = inner
                .criteria
                .as_ref()
                .map(|c| (c.r#false.clone(), c.r#true.clone()))
                .unwrap_or_default();
            let fallback = |text: &str, default: &str| {
                if text.is_empty() {
                    default.to_owned()
                } else {
                    text.to_owned()
                }
            };
            vec![
                fallback(&r#false, "No / false"),
                fallback(&r#true, "Yes / true"),
            ]
        }
        Question::Score(inner) => inner.criteria.clone(),
    };
    Case {
        name: name.to_owned(),
        state: prompt_state.clone(),
        questions: std::iter::once((
            "q0".to_owned(),
            serde_json::to_value(question).expect("question"),
        ))
        .collect(),
        prompt: PromptPlan {
            state: prompt_state,
            instructions: prompt_instructions,
            options: prompt_options,
        },
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

fn cases() -> Vec<Case> {
    vec![
        choice_case(
            "choice-two",
            "The kitchen light is on. The office light is off.",
            &["kitchen", "office"],
            &[Some("the room whose light is on"), None],
        ),
        choice_case(
            "choice-five-object-state",
            "Incident record: severity high, service billing, region eu-west, status mitigated, owner oncall.",
            &["severity", "service", "region", "status", "owner"],
            &[None, None, None, None, None],
        ),
        single_question(
            "noul-default-criteria",
            json!({"ticket": {"priority": "p1", "escalated": true}}),
            Question::Noul(openkind_core::NoulQuestion {
                instructions: json!("Is the ticket escalated?"),
                criteria: None,
            }),
        ),
        single_question(
            "noul-described-criteria",
            json!("Sensor report: temperature 41.2C, threshold 40C."),
            Question::Noul(openkind_core::NoulQuestion {
                instructions: json!("Did the temperature exceed the threshold?"),
                criteria: Some(openkind_core::NoulCriteria {
                    r#true: "The temperature is above the threshold.".to_owned(),
                    r#false: "The temperature is at or below the threshold.".to_owned(),
                }),
            }),
        ),
        single_question(
            "score-three-levels",
            json!("Queue depth is 420 events; the SLO allows up to 500 with a soft warn at 350."),
            Question::Score(openkind_core::ScoreQuestion {
                instructions: json!("How loaded is the queue?"),
                criteria: vec!["low".to_owned(), "moderate".to_owned(), "high".to_owned()],
            }),
        ),
        single_question(
            "choice-empty-state",
            json!(""),
            Question::Choice(openkind_core::ChoiceQuestion {
                instructions: json!("Reply with any option."),
                criteria: [
                    ("alpha".to_owned(), None),
                    ("beta".to_owned(), None),
                ]
                .into_iter()
                .collect(),
            }),
        ),
        {
            let mut case = single_question(
                "choice-none-option",
                json!("The request matches neither the allowlist nor the denylist."),
                Question::Choice(openkind_core::ChoiceQuestion {
                    instructions: json!("Which list applies?"),
                    criteria: [
                        ("allowlist".to_owned(), None),
                        ("denylist".to_owned(), None),
                        (
                            "__none__".to_owned(),
                            Some("neither list applies".to_owned()),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                }),
            );
            case.name = "choice-none-option".to_owned();
            case
        },
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_root: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut profile_loader = "edge";
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => {
                model_root = Some(PathBuf::from(args.next().expect("--model-root value")))
            }
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            "--profile" => {
                profile_loader = Box::leak(
                    args.next()
                        .expect("--profile value")
                        .trim()
                        .to_owned()
                        .into_boxed_str(),
                );
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let model_root = model_root.ok_or("--model-root is required")?;
    let output = output.ok_or("--output is required")?;
    let profile = profile_by_short_name(profile_loader)
        .ok_or_else(|| format!("unknown --profile {profile_loader}; known: edge, lite"))?;

    let engine = D3Engine::load(D3EngineConfig {
        model_root: model_root.clone(),
        limits: FamilyLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 0,
            retry_after_ms: 100,
            evaluation_timeout: None,
        },
        profile,
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

    // Render every case's prompt through the engine's own planning path and
    // record byte/token statistics; these anchor the golden to the reference
    // prompts computed by research/handoffs/registry-expansion-2026-10-12/
    // d3_reference_parity.py (the shipped d3_format.py over the real
    // tokenizer and chat template).
    let renderer = D3Renderer::load(
        &model_root.join("tokenizer.json"),
        &{
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(model_root.join("decision_config.json"))
                    .expect("read decision config"),
            )?;
            let codes: Vec<String> = serde_json::from_value(config["codes"].clone())?;
            codes
        },
        &{
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(model_root.join("decision_config.json"))
                    .expect("read decision config"),
            )?;
            let token_ids: Vec<u64> = serde_json::from_value(config["token_ids"].clone())?;
            token_ids
        },
        profile,
    )?;
    let mut prompt_stats = Vec::new();
    for case in case_values.iter().zip(cases()) {
        let (case_value, case) = case;
        let row = renderer.render_row(
            &case.prompt.state,
            &case.prompt.instructions,
            &case.prompt.options,
        )?;
        let ids = row.prompt_ids();
        let digest = {
            // Same serialization as the reference-parity script: the
            // comma-joined decimal ids, UTF-8 hashed.
            use sha2::Digest as _;
            use std::fmt::Write as _;
            let joined = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
            let mut hasher = sha2::Sha256::new();
            hasher.update(joined.as_bytes());
            let digest = hasher.finalize();
            let mut text = String::new();
            for byte in digest {
                write!(&mut text, "{byte:02x}").expect("hex");
            }
            text
        };
        prompt_stats.push(json!({
            "name": case_value["name"],
            "token_count": ids.len(),
            "token_ids_sha256": digest,
        }));
    }

    let golden = json!({
        "profile_id": profile.profile_id,
        "family": FAMILY_SLUG,
        "backbone": {
            "id": profile.backbone_id,
            "revision": profile.backbone_revision,
        },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-d3-edge-fixture",
            "case_construction": "correct option stated verbatim in the state document; \
              an empty-state case exercises the (empty) placeholder and a __none__ case \
              exercises the reserved option scored as an ordinary candidate",
        },
        "cases": case_values,
        "prompt_stats": prompt_stats,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
