//! Offline fixture generator for the pinned `decider` (`decider-4b`)
//! profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --release --bin gen_decider_fixture -- \
//!   --model-root ~/.cache/openkind/decider-4b \
//!   --output crates/openkind-backends/tests/fixtures/decider_4b_0529bf6f2bed84641701/golden.json
//! ```
//!
//! The cases are synthetic decisions whose correct option is stated verbatim
//! in the state document; they exercise the narrow Choice read, Noul, the
//! isolated-level Score rows, the wide (label-table) Choice rendering, and a
//! long JSON array state with `_index` annotations. Temperatures are the
//! pinned `decider_config.json` values, so no calibration fit runs here.

use std::path::PathBuf;

use openkind_backends::families::decider::{
    DeciderEngine, DeciderEngineConfig, DECIDER_4B, EXECUTION_ARITHMETIC_ID, FAMILY_SLUG,
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

fn cases() -> Vec<Case> {
    let mut cases = vec![
        stated_choice(
            "os",
            "Fictional incident record. The asset operating system is Linux. These statements are the full evidence record; do not infer missing facts.",
            &["Linux", "Windows", "macOS", "BSD"],
            0,
        ),
        stated_noul(
            "mfa-true",
            "Fictional incident record. The record states that two-factor authentication was enabled on the affected account. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that two-factor authentication was enabled?",
            true,
        ),
        stated_score(
            "impact-3",
            "Fictional incident record. The documented impact level is 3. These statements are the full evidence record; do not infer missing facts.",
            5,
            3,
        ),
        // A wide Choice (12 options > NARROW_OPTIONS) exercises the
        // label-table rendering path.
        {
            let mut options: Vec<String> =
                (0..12).map(|index| format!("site-{index:02}")).collect();
            options.push("site-none".to_owned());
            let _ = options.pop();
            text_case(
                "wide-choice",
                "Fictional deployment record. The affected site is site-07. These statements are the full evidence record; do not infer missing facts.",
                "site-07",
                Question::Choice(openkind_core::ChoiceQuestion {
                    instructions: json!("Which site is explicitly recorded in the state?"),
                    criteria: (0..12)
                        .map(|index| (format!("site-{index:02}"), None))
                        .collect(),
                }),
            )
        },
    ];
    // A JSON object state with a long array exercises `_index` annotation and
    // the compact Python-json state serialization.
    let events: Vec<serde_json::Value> = (0..10)
        .map(|index| {
            json!({
                "seq": index,
                "detail": if index == 6 { "the database failover was approved" } else { "routine heartbeat" },
            })
        })
        .collect();
    cases.push(Case {
        name: "json-state".to_owned(),
        truth_label: "true".to_owned(),
        state: json!({ "events": events }),
        question: Question::Noul(openkind_core::NoulQuestion {
            instructions: json!("Does the record state that a database failover was approved?"),
            criteria: None,
        }),
    });
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
    let mut model_root = PathBuf::from(
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .unwrap_or_default(),
    )
    .join(".cache/openkind/decider-4b");
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => model_root = PathBuf::from(args.next().expect("--model-root value")),
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let output = output.ok_or_else(|| "--output is required".to_owned())?;

    let engine = DeciderEngine::load(DeciderEngineConfig {
        profile: &DECIDER_4B,
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
    let calibration = match DECIDER_4B.calibration {
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
        "profile_id": DECIDER_4B.profile_id,
        "loader_id": DECIDER_4B.loader_id,
        "backbone": { "id": DECIDER_4B.backbone_id, "revision": DECIDER_4B.backbone_revision },
        "execution_arithmetic": EXECUTION_ARITHMETIC_ID,
        "family": FAMILY_SLUG,
        "calibration": calibration,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen_decider_fixture",
            "case_construction": "correct option stated verbatim in the state document",
            "reference_runtime": "github.com/Mapika/decider decider package (in-repo, pinned revision)",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
