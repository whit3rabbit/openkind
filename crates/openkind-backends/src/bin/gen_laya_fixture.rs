//! Offline fixture generator, reference comparator, and shipped-temperature
//! verifier for the pinned laya profiles.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! # 1. vendor the engine's golden replay fixture
//! cargo run -p openkind-backends --bin gen_laya_fixture -- --profile english \
//!   --output crates/openkind-backends/tests/fixtures/laya-english_<profile-id>/golden.json
//!
//! # 2. run the Python reference over the same requests (see the script's
//! #    header in scripts/laya_reference.py) producing reference.json
//!
//! # 3. quantify parity against the reference and record it on the family page
//! cargo run -p openkind-backends --bin gen_laya_fixture -- --profile english \
//!   --compare reference.json
//! ```
//!
//! There is no temperature fit: the checkpoints ship fitted per-type and
//! per-bucket temperatures in `rl_agent_config.json`, and the loader's
//! contract check verifies them; decode applies the reference's `[0.5, 5.0]`
//! clamp, under which the shipped `choice:11+` sharpening bucket resolves to
//! 0.5. The case set is a fixed list of synthetic stated-fact decisions that
//! exercises every wire primitive, both noul criteria modes, the `__none__`
//! reserved key, object/array/long states, head-budget overflow, mask-token
//! sanitization, and non-Latin script.

use std::path::PathBuf;

use openkind_backends::families::laya::{
    LayaEngine, LayaEngineConfig, LayaProfile, LAYA_ENGLISH, LAYA_MULTILINGUAL,
    LAYA_TYPED_DECISIONS,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::Question;
use openkind_engine::DecisionEngine;
use serde_json::{json, Value};

/// One replay case: a name, a state, and one question.
struct Case {
    name: String,
    state: Value,
    question: Question,
}

fn choice_case(name: &str, state: Value, options: &[(&str, Option<&str>)]) -> Case {
    Case {
        name: name.to_owned(),
        state,
        question: Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("Which option is explicitly recorded in the state?"),
            criteria: options
                .iter()
                .map(|(label, description)| {
                    (label.to_string(), description.map(|text| text.to_string()))
                })
                .collect(),
        }),
    }
}

fn noul_case(name: &str, state: Value, instruction: &str, criteria: Option<(&str, &str)>) -> Case {
    Case {
        name: name.to_owned(),
        state,
        question: Question::Noul(openkind_core::NoulQuestion {
            instructions: json!(instruction),
            criteria: criteria.map(|(r#false, r#true)| openkind_core::NoulCriteria {
                r#false: r#false.to_owned(),
                r#true: r#true.to_owned(),
            }),
        }),
    }
}

fn score_case(name: &str, state: Value, levels: &[&str]) -> Case {
    Case {
        name: name.to_owned(),
        state,
        question: Question::Score(openkind_core::ScoreQuestion {
            instructions: json!("What is the documented impact level?"),
            criteria: levels.iter().map(|level| level.to_string()).collect(),
        }),
    }
}

const LONG_STATE: &str = "Fictional incident record. The asset operating system is Windows and the affected deployment runs in us-east-1. The observed connection used SSH from an external destination. The owner of record is the platform team. The incident status is investigating. The documented impact level is moderate. Two-factor authentication was enabled on the affected account. No data export was approved during the window. The patch level is current as of the report date. These statements are the full evidence record; do not infer missing facts. Background: the record was opened by the on-call rotation after an automated alert fired twice within ten minutes. Initial triage confirmed the alert source, correlated the timeline, and attached the raw telemetry excerpts below. The excerpts include connection metadata, authentication traces, and a short narrative from the responding engineer. Nothing in this paragraph changes any fact stated above.";

/// The pinned replay/parity case set, shared by every laya profile.
fn cases() -> Vec<Case> {
    let english_state = "Fictional incident record. The asset operating system is Windows. These statements are the full evidence record; do not infer missing facts.";
    vec![
        choice_case(
            "choice-two",
            json!(english_state),
            &[("Linux", None), ("Windows", None)],
        ),
        choice_case(
            "choice-described",
            json!(english_state),
            &[
                ("Linux", Some("a freely licensed kernel family")),
                ("Windows", Some("a proprietary Microsoft operating system")),
                ("macOS", Some("Apple's desktop operating system")),
                ("BSD", Some("a Berkeley Unix descendant")),
            ],
        ),
        choice_case(
            "choice-none",
            json!(english_state),
            &[
                ("Linux", None),
                ("Windows", None),
                ("__none__", Some("none of the listed options is recorded")),
            ],
        ),
        choice_case(
            "choice-twelve",
            json!(
                "Fictional incident record. The affected queue is billing. These statements are the full evidence record; do not infer missing facts."
            ),
            &[
                ("billing", None),
                ("tech", None),
                ("account", None),
                ("shipping", None),
                ("returns", None),
                ("payments", None),
                ("marketing", None),
                ("legal", None),
                ("facilities", None),
                ("security", None),
                ("partnerships", None),
                ("other", None),
            ],
        ),
        choice_case(
            "choice-head-overflow",
            json!(LONG_STATE),
            &[
                (
                    "operating system",
                    Some(
                        "the recorded platform family of the affected asset with full provenance detail",
                    ),
                ),
                (
                    "deployment region",
                    Some(
                        "the recorded cloud region the affected deployment runs in with full provenance detail",
                    ),
                ),
                (
                    "connection protocol",
                    Some(
                        "the recorded protocol the observed connection used with full provenance detail",
                    ),
                ),
                (
                    "owning team",
                    Some(
                        "the recorded owner of record for the incident with full provenance detail",
                    ),
                ),
                (
                    "incident status",
                    Some(
                        "the recorded lifecycle status of the incident with full provenance detail",
                    ),
                ),
                (
                    "impact level",
                    Some(
                        "the recorded severity of the documented impact with full provenance detail",
                    ),
                ),
                (
                    "authentication state",
                    Some(
                        "the recorded two-factor state of the affected account with full provenance detail",
                    ),
                ),
                (
                    "export approval",
                    Some(
                        "the recorded approval state of any data export with full provenance detail",
                    ),
                ),
            ],
        ),
        score_case(
            "score-four",
            json!(
                "Fictional incident record. The documented impact level is moderate. These statements are the full evidence record; do not infer missing facts."
            ),
            &["minor", "moderate", "major", "critical"],
        ),
        score_case(
            "score-six",
            json!(
                "Fictional incident record. The documented impact level is major. These statements are the full evidence record; do not infer missing facts."
            ),
            &[
                "negligible",
                "minor",
                "moderate",
                "major",
                "severe",
                "critical",
            ],
        ),
        noul_case(
            "noul-default",
            json!(
                "Fictional incident record. An external destination was observed in the connection logs. These statements are the full evidence record; do not infer missing facts."
            ),
            "Does the record explicitly state that an external destination was observed?",
            None,
        ),
        noul_case(
            "noul-explicit",
            json!(
                "Fictional incident record. The record does not mention external destinations. These statements are the full evidence record; do not infer missing facts."
            ),
            "Does the record explicitly state that an external destination was observed?",
            Some((
                "the record names no external destination",
                "the record names an external destination",
            )),
        ),
        // Object state: exercises the Python-dumps serialization contract.
        choice_case(
            "state-object",
            json!({
                "record": "fictional incident summary",
                "asset": {"os": "Windows", "region": "us-east-1"},
                "tags": ["infra", "tier-2"],
                "open": true
            }),
            &[("Linux", None), ("Windows", None), ("macOS", None)],
        ),
        // Array state: exercises left (newest-keeping) truncation.
        choice_case(
            "state-array",
            json!([
                "Conversation turn one: the customer asks about the invoice total.",
                "Conversation turn two: the customer repeats the billing question.",
                "Conversation turn three: the customer reports the invoice total is wrong.",
                "Conversation turn four: the customer requests a correction of the billing error."
            ]),
            &[
                (
                    "billing",
                    Some("the discussion concerns invoices or charges"),
                ),
                ("tech", Some("the discussion concerns technical defects")),
                ("other", Some("the discussion concerns another topic")),
            ],
        ),
        // Long state: exercises keep-first state truncation.
        choice_case(
            "state-long",
            json!(LONG_STATE),
            &[("Linux", None), ("Windows", None)],
        ),
        // Literal mask token in user text: sanitized before tokenizing.
        noul_case(
            "mask-literal",
            json!(
                "Fictional incident record. The token [MASK] appears literally in the state text and the patch level is current. These statements are the full evidence record; do not infer missing facts."
            ),
            "Is the patch level described as current?",
            None,
        ),
        // Non-Latin script: the multilingual checkpoint's home ground and a
        // byte-level stress for the English tokenizer.
        choice_case(
            "non-latin",
            json!(
                "Fiktiver Vorfallbericht. Das Betriebssystem der Anlage ist Windows. Diese Angaben sind der vollständige Befund; keine fehlenden Fakten ableiten."
            ),
            &[("Linux", None), ("Windows", None), ("macOS", None)],
        ),
        choice_case(
            "cyrillic",
            json!(
                "Вымышленный отчёт об инциденте. Операционная система актива — Windows. Эти утверждения являются полным отчётом; не выводите отсутствующие факты."
            ),
            &[("billing", None), ("tech", None), ("other", None)],
        ),
    ]
}

fn profile_by_name(name: &str) -> &'static LayaProfile {
    match name {
        "english" => &LAYA_ENGLISH,
        "multilingual" => &LAYA_MULTILINGUAL,
        "typed-decisions" => &LAYA_TYPED_DECISIONS,
        other => panic!("unknown profile `{other}` (english|multilingual|typed-decisions)"),
    }
}

fn default_model_root(profile: &LayaProfile) -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/openkind")
        .join(match profile.loader_id {
            "laya-english" => "laya-english",
            "laya-multilingual" => "laya-multilingual",
            _ => "laya-typed-decisions",
        })
}

/// Walk one response's answers and collect every probability/confidence
/// delta against a reference response with the same shape.
fn answer_deltas(
    ours: &Value,
    reference: &Value,
    case_name: &str,
    deltas: &mut Vec<(String, f64)>,
) {
    let ours_answers = ours
        .get("answers")
        .and_then(Value::as_object)
        .expect("our response carries answers");
    let reference_answers = reference
        .get("answers")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("reference case {case_name} carries answers"));
    for (qid, our_answer) in ours_answers {
        let reference_answer = reference_answers
            .get(qid)
            .unwrap_or_else(|| panic!("reference case {case_name} answers {qid}"));
        for key in ["noul", "confidence", "score"] {
            if let (Some(a), Some(b)) = (our_answer.get(key), reference_answer.get(key)) {
                let (a, b) = (a.as_f64().expect("f64"), b.as_f64().expect("f64"));
                deltas.push((format!("{case_name}/{qid}.{key}"), (a - b).abs()));
            }
        }
        if let (Some(a), Some(b)) = (
            our_answer.get("probabilities").and_then(Value::as_object),
            reference_answer
                .get("probabilities")
                .and_then(Value::as_object),
        ) {
            for (option, our_p) in a {
                let reference_p = b
                    .get(option)
                    .unwrap_or_else(|| panic!("reference {case_name}/{qid} has {option}"))
                    .as_f64()
                    .expect("f64 probability");
                deltas.push((
                    format!("{case_name}/{qid}.p[{option}]"),
                    (our_p.as_f64().expect("f64") - reference_p).abs(),
                ));
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile_name = "english".to_owned();
    let mut model_root: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut compare: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => profile_name = args.next().expect("--profile value"),
            "--model-root" => {
                model_root = Some(PathBuf::from(args.next().expect("--model-root value")))
            }
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            "--compare" => compare = Some(PathBuf::from(args.next().expect("--compare value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let profile = profile_by_name(&profile_name);
    let model_root = model_root.unwrap_or_else(|| default_model_root(profile));

    let limits = FamilyLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 0,
        retry_after_ms: 100,
        evaluation_timeout: None,
    };
    let engine = LayaEngine::load(LayaEngineConfig {
        profile,
        model_root: model_root.clone(),
        limits,
    })?;
    eprintln!(
        "[fixture] loaded {} ({}) from {}",
        profile.loader_id,
        profile.profile_id,
        model_root.display()
    );

    let case_list = cases();
    let mut rendered_cases = Vec::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    for case in &case_list {
        let request: openkind_core::SystemRequest = serde_json::from_value(json!({
            "state": case.state,
            "model": "fixture",
            "questions": { "q0": serde_json::to_value(&case.question)? },
        }))?;
        let request_value = serde_json::to_value(&request)?;
        let response = runtime.block_on(engine.evaluate(request))?;
        rendered_cases.push((
            case.name.clone(),
            request_value,
            serde_json::to_value(&response)?,
        ));
    }

    if let Some(reference_path) = compare {
        // Reference comparison: replay our engine against the Python
        // reference's responses over the identical request set.
        let reference_text = std::fs::read_to_string(&reference_path)?;
        let reference: Value = serde_json::from_str(&reference_text)?;
        let reference_cases = reference
            .get("cases")
            .and_then(Value::as_array)
            .expect("reference carries cases");
        let mut deltas = Vec::new();
        let mut selection_flips = Vec::new();
        for (name, request, response) in &rendered_cases {
            let reference_case = reference_cases
                .iter()
                .find(|case| case.get("name").and_then(Value::as_str) == Some(name.as_str()))
                .unwrap_or_else(|| panic!("reference carries case {name}"));
            let reference_request = reference_case.get("request").expect("reference request");
            if reference_request != request {
                return Err(
                    format!("case {name}: reference request differs from local case").into(),
                );
            }
            let reference_response = reference_case.get("response").expect("reference response");
            let our_choice = response
                .pointer("/answers/q0/choice")
                .and_then(Value::as_str);
            let reference_choice = reference_response
                .pointer("/answers/q0/choice")
                .and_then(Value::as_str);
            if our_choice != reference_choice {
                selection_flips.push(format!(
                    "{name}: ours={:?} reference={:?}",
                    our_choice, reference_choice
                ));
            }
            answer_deltas(response, reference_response, name, &mut deltas);
        }
        deltas.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("[compare] cases: {}", rendered_cases.len());
        eprintln!("[compare] selection flips: {}", selection_flips.len());
        for flip in &selection_flips {
            eprintln!("[compare]   FLIP {flip}");
        }
        if let Some((worst, delta)) = deltas.first() {
            eprintln!("[compare] max |delta| = {delta:.3e} at {worst}");
        }
        let over_tolerance: Vec<&(String, f64)> =
            deltas.iter().filter(|(_, delta)| *delta > 0.005).collect();
        eprintln!(
            "[compare] entries beyond 0.005 tolerance: {}",
            over_tolerance.len()
        );
        for (name, delta) in over_tolerance.iter().take(10) {
            eprintln!("[compare]   {delta:.3e} {name}");
        }
        return Ok(());
    }

    let output = output.expect("--output is required without --compare");
    let case_values: Vec<Value> = rendered_cases
        .into_iter()
        .map(|(name, request, response)| {
            json!({ "name": name, "request": request, "response": response })
        })
        .collect();
    let golden = json!({
        "profile_id": profile.profile_id,
        "backbone": { "id": profile.backbone_id, "revision": profile.backbone_revision },
        "execution_arithmetic": openkind_backends::families::laya::EXECUTION_ARITHMETIC_ID,
        "calibration": {
            "source": "shipped rl_agent_config.json temperature tables with the reference [0.5, 5.0] clamp",
            "temperature": profile.temperature,
        },
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-laya-fixture",
            "case_construction": "correct option stated verbatim in the state document; readout-surface cases shared across profiles",
        },
        "cases": case_values,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
