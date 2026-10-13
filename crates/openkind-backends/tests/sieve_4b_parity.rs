//! Offline parity and failure-mode tests for the pinned `sieve-4b` profile
//! (the `sieve` family's causal LoRA-merged code-readout engine).
//!
//! Unit-level contract checks always run. The full golden replay runs only
//! when the pinned artifacts are present locally (point
//! `OPENKIND_SIEVE_MODEL_ROOT` at the installation root); tests never
//! download model artifacts. The golden replay is a Rust self-replay: it
//! pins this engine's own outputs and is not a forward-pass cross-check
//! against the torch reference.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::sieve::{
    derive_sieve_answer_codes, plan_sieve_request, SieveEngine, SieveEngineConfig, SieveRenderer,
    FAMILY_SLUG, SIEVE_4B,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/sieve_8b97b56524b55c41fbf4";
const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_SIEVE_MODEL_ROOT")
        .map(PathBuf::from)
        .filter(|root| root.is_dir())
}

fn limits() -> FamilyLimits {
    FamilyLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 0,
        retry_after_ms: 100,
        evaluation_timeout: None,
    }
}

#[derive(Debug, Deserialize)]
struct Golden {
    cases: Vec<GoldenCase>,
    prompt_stats: Vec<PromptStats>,
}

/// Engine-side prompt evidence recorded per case; the branch digests equal
/// the shipped `sieve4b/prompt.py` digests in the reference prompts file.
#[derive(Debug, Deserialize)]
struct PromptStats {
    name: String,
    prefix_token_count: usize,
    branches: Vec<BranchStats>,
}

#[derive(Debug, Deserialize)]
struct BranchStats {
    token_count: usize,
    branch_ids_sha256: String,
}

/// SHA-256 of the comma-joined token ids (the generator's digest rule).
fn ids_digest(ids: &[u32]) -> String {
    use sha2::Digest as _;
    let joined = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    sha2::Sha256::digest(joined.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn golden() -> Golden {
    serde_json::from_slice(
        &fs::read(fixture_dir().join("golden.json")).expect("read golden fixture"),
    )
    .expect("decode golden fixture")
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    name: String,
    request: openkind_core::SystemRequest,
    response: GoldenResponse,
}

#[derive(Debug, Deserialize)]
struct GoldenResponse {
    answers: std::collections::BTreeMap<String, serde_json::Value>,
    usage: GoldenUsage,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoldenUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn assert_answer_matches(label: &str, actual: &serde_json::Value, expected: &serde_json::Value) {
    assert_eq!(
        actual["type"], expected["type"],
        "{label}: answer type drifted"
    );
    match expected["type"].as_str().expect("tagged answer") {
        "noul" => {
            let actual = actual["noul"].as_f64().expect("noul");
            let expected = expected["noul"].as_f64().expect("noul");
            assert!(
                (actual - expected).abs() <= PROBABILITY_TOLERANCE,
                "{label}: noul {actual} vs {expected}"
            );
        }
        "choice" => {
            assert_eq!(
                actual["choice"], expected["choice"],
                "{label}: selected option drifted"
            );
            let probabilities = expected["probabilities"]
                .as_object()
                .expect("probabilities");
            for (key, expected_probability) in probabilities {
                let actual_probability = actual["probabilities"][key]
                    .as_f64()
                    .unwrap_or_else(|| panic!("{label}: missing probability for {key}"));
                let expected_probability = expected_probability.as_f64().expect("probability");
                assert!(
                    (actual_probability - expected_probability).abs() <= PROBABILITY_TOLERANCE,
                    "{label}: {key} {actual_probability} vs {expected_probability}"
                );
            }
        }
        "score" => {
            let actual_score = actual["score"].as_f64().expect("score");
            let expected_score = expected["score"].as_f64().expect("score");
            assert!(
                (actual_score - expected_score).abs() <= PROBABILITY_TOLERANCE,
                "{label}: score {actual_score} vs {expected_score}"
            );
            let probabilities = expected["probabilities"]
                .as_object()
                .expect("probabilities");
            for (key, expected_probability) in probabilities {
                let actual_probability = actual["probabilities"][key]
                    .as_f64()
                    .unwrap_or_else(|| panic!("{label}: missing probability for level {key}"));
                let expected_probability = expected_probability.as_f64().expect("probability");
                assert!(
                    (actual_probability - expected_probability).abs() <= PROBABILITY_TOLERANCE,
                    "{label}: level {key} {actual_probability} vs {expected_probability}"
                );
            }
        }
        other => panic!("{label}: unexpected answer type {other}"),
    }
}

#[test]
fn profile_constants_are_self_consistent() {
    assert_eq!(
        SIEVE_4B.profile_id,
        openkind_backends::families::support::derive_profile_id(
            FAMILY_SLUG,
            SIEVE_4B.backbone_id,
            SIEVE_4B.backbone_revision,
        )
    );
    assert_eq!(SIEVE_4B.backbone_id, "sthanika-ai/Sieve-4B");
    assert_eq!(
        SIEVE_4B.backbone_revision,
        "ea7cd1edee29590a82ed9fbfcb644227a8eed5f1"
    );
    // The base is the causal Qwen3.5-4B hybrid (the pinned qwen3.5 dims).
    const {
        assert!(SIEVE_4B.geometry.causal);
        assert!(SIEVE_4B.geometry.hidden_size == 2_560);
        assert!(SIEVE_4B.lora_rank == 32);
        assert!(SIEVE_4B.lora_scale == 2.0);
    }
}

/// The synthetic WordLevel sample tokenizer. It exercises the code-table
/// derivation's selection and ordering logic and the request planner; its
/// token ids and counts are not the pinned Qwen3.5 tokenizer's. The real
/// code table is checked at load against `sieve_config.json`.
fn sample_tokenizer() -> tokenizers::Tokenizer {
    let bytes = fs::read(fixture_dir().join("tokenizer-vocab-sample.json"))
        .expect("vendored tokenizer sample is checked in");
    tokenizers::Tokenizer::from_bytes(bytes).expect("sample parses")
}

/// A renderer over the sample tokenizer for weight-free planning tests.
fn sample_renderer() -> SieveRenderer {
    let (codes, ids) =
        derive_sieve_answer_codes(&sample_tokenizer()).expect("sample provides the code table");
    let ids: Vec<u64> = ids.into_iter().map(u64::from).collect();
    SieveRenderer::load(
        &fixture_dir().join("tokenizer-vocab-sample.json"),
        &codes,
        &ids,
    )
    .expect("sample renderer loads")
}

fn request(state: serde_json::Value, questions: serde_json::Value) -> openkind_core::SystemRequest {
    serde_json::from_value(serde_json::json!({
        "state": state,
        "model": "fixture",
        "questions": questions,
    }))
    .expect("wire request")
}

/// `count` whitespace-separated words: one sample-tokenizer token each.
fn words(count: usize) -> String {
    vec!["w"; count].join(" ")
}

#[test]
fn renderer_derives_the_reference_code_table() {
    let tokenizer = sample_tokenizer();
    let derived =
        derive_sieve_answer_codes(&tokenizer).expect("sample tokenizer provides the code table");
    assert_eq!(derived.0.len(), 255);
    assert_eq!(derived.0[0], "A");
    assert_eq!(derived.0[26], "AA");
    for (code, id) in derived.0.iter().zip(derived.1.iter()) {
        let decoded = tokenizer.decode(&[*id], false).expect("decode");
        assert_eq!(&decoded, code, "code {code} must round-trip");
    }
}

#[test]
fn planning_accepts_a_one_option_choice() {
    // Reference `contract.py:check_limits` and `prompt.py:branch` serve
    // 1..=255 options, as does core validation.
    let plan = plan_sieve_request(
        &sample_renderer(),
        &SIEVE_4B,
        &request(
            serde_json::json!("state"),
            serde_json::json!({"q": {"type": "choice", "instructions": "Pick.", "criteria": {"only": null}}}),
        ),
    )
    .expect("a one-option choice is served");
    assert_eq!(plan.rows().len(), 1);
}

#[test]
fn planning_rejects_a_later_invalid_question_without_the_model() {
    // The planner owns every validation and runs no forward;
    // `SieveEngine::evaluate` scores only a plan that succeeded, so the
    // valid question `a` never reaches the model.
    let error = match plan_sieve_request(
        &sample_renderer(),
        &SIEVE_4B,
        &request(
            serde_json::json!("state"),
            serde_json::json!({
                "a": {"type": "choice", "instructions": "Pick.", "criteria": {"x": null, "y": null}},
                "z": {"type": "score", "instructions": "Rate.", "criteria": (0..11).map(|level| level.to_string()).collect::<Vec<_>>()},
            }),
        ),
    ) {
        Err(error) => error,
        Ok(_) => panic!("an 11-level score must fail planning"),
    };
    assert!(
        error.to_string().contains("question `z`") && error.to_string().contains("11 levels"),
        "unexpected error: {error}"
    );
}

#[test]
fn planning_enforces_the_per_question_token_cap() {
    // 32,768 state words alone reach the cap; any branch exceeds it.
    let error = match plan_sieve_request(
        &sample_renderer(),
        &SIEVE_4B,
        &request(
            serde_json::json!(words(32_768)),
            serde_json::json!({"q": {"type": "noul", "instructions": "True?"}}),
        ),
    ) {
        Err(error) => error,
        Ok(_) => panic!("an over-length state must fail planning"),
    };
    assert!(
        error.to_string().contains("per question"),
        "unexpected error: {error}"
    );
}

#[test]
fn planning_enforces_the_per_request_token_cap() {
    // Three ~22k-token questions each fit with the state, but together
    // exceed 65,536 tokens.
    let question = serde_json::json!({
        "type": "choice",
        "instructions": words(22_000),
        "criteria": {"x": null, "y": null},
    });
    let error = match plan_sieve_request(
        &sample_renderer(),
        &SIEVE_4B,
        &request(
            serde_json::json!("state"),
            serde_json::json!({"a": question, "b": question, "c": question}),
        ),
    ) {
        Err(error) => error,
        Ok(_) => panic!("an over-length request must fail planning"),
    };
    assert!(
        error.to_string().contains("per request"),
        "unexpected error: {error}"
    );
}

#[test]
fn billing_counts_the_state_once_plus_each_questions_longest_branch() {
    let renderer = sample_renderer();
    let plan = plan_sieve_request(
        &renderer,
        &SIEVE_4B,
        &request(
            serde_json::json!("The ticket is escalated."),
            serde_json::json!({
                "a": {"type": "noul", "instructions": "Is it escalated?"},
                "b": {"type": "choice", "instructions": "Pick.", "criteria": {"x": null, "y": "why"}},
            }),
        ),
    )
    .expect("plan");
    let prefix = plan.prefix_ids().len();
    let preamble = renderer.preamble_token_count();
    assert!(preamble > 0 && preamble < prefix);
    let rows = plan.rows();
    assert_eq!(rows.len(), 3, "noul plans two listings, choice one");
    assert!(rows
        .iter()
        .all(|(_, row)| row.starts_with(plan.prefix_ids())));
    let longest = |id: &str| {
        rows.iter()
            .filter(|(question, _)| *question == id)
            .map(|(_, row)| row.len() - prefix)
            .max()
            .expect("question has rows")
    };
    // Reference `engine.py:evaluate`: preamble excluded, state once, noul's
    // two listings billed as one branch.
    let expected = (prefix - preamble) + longest("a") + longest("b");
    assert_eq!(plan.input_tokens(), expected as u64);
}

#[test]
fn load_fails_closed_on_artifact_digest_mismatch() {
    // Every required artifact exists but holds placeholder bytes, so the
    // loader must reject the first digest it checks; no weights needed.
    let root = tempfile::tempdir().expect("tempdir");
    let base = root.path().join("base");
    let checkpoint = root.path().join("checkpoint");
    fs::create_dir_all(&base).expect("create base");
    fs::create_dir_all(&checkpoint).expect("create checkpoint");
    fs::write(base.join("config.json"), b"{}").expect("stage base config");
    for (name, _, _) in SIEVE_4B.base_shards {
        fs::write(base.join(name), b"placeholder").expect("stage base shard");
    }
    for (name, _, _) in SIEVE_4B.checkpoint_files {
        fs::write(checkpoint.join(name), b"placeholder").expect("stage checkpoint file");
    }
    let error = match SieveEngine::load(SieveEngineConfig {
        model_root: root.path().to_path_buf(),
        limits: limits(),
        profile: &SIEVE_4B,
    }) {
        Err(error) => error,
        Ok(_) => panic!("placeholder artifacts must fail closed"),
    };
    assert!(
        error.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {error}"
    );
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!("openkind-sieve-missing-{}", std::process::id()));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match SieveEngine::load(SieveEngineConfig {
        model_root: empty.clone(),
        limits: limits(),
        profile: &SIEVE_4B,
    }) {
        Err(error) => error,
        Ok(_) => panic!("missing artifacts must fail closed"),
    };
    assert!(
        error.to_string().contains("missing") || error.to_string().contains("read"),
        "unexpected error: {error}"
    );
    let _ = fs::remove_dir_all(&empty);
}

#[test]
fn planner_reproduces_the_recorded_prompt_stats_and_usage() {
    // Needs only `checkpoint/tokenizer.json` and `checkpoint/sieve_config.json`
    // under the model root, not weights. The tokenizer digest is not
    // checked here: reproducing every recorded row digest is the check.
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_SIEVE_MODEL_ROOT is not set");
        return;
    };
    let checkpoint = root.join("checkpoint");
    let sieve_config: serde_json::Value = serde_json::from_slice(
        &fs::read(checkpoint.join("sieve_config.json")).expect("read sieve_config.json"),
    )
    .expect("decode sieve_config.json");
    let codes: Vec<String> =
        serde_json::from_value(sieve_config["codes"].clone()).expect("pinned codes");
    let code_ids: Vec<u64> =
        serde_json::from_value(sieve_config["code_ids"].clone()).expect("pinned code ids");
    let renderer = SieveRenderer::load(&checkpoint.join("tokenizer.json"), &codes, &code_ids)
        .expect("tokenizer derives the pinned code table");
    let golden = golden();
    for case in &golden.cases {
        let stats = golden
            .prompt_stats
            .iter()
            .find(|stats| stats.name == case.name)
            .unwrap_or_else(|| panic!("{}: no prompt stats", case.name));
        let plan = plan_sieve_request(&renderer, &SIEVE_4B, &case.request)
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(
            plan.prefix_ids().len(),
            stats.prefix_token_count,
            "{}: prefix length",
            case.name
        );
        let rows = plan.rows();
        assert_eq!(rows.len(), stats.branches.len(), "{}: row count", case.name);
        for ((_, row), branch) in rows.iter().zip(&stats.branches) {
            assert_eq!(row.len(), branch.token_count, "{}: row length", case.name);
            assert_eq!(
                ids_digest(row),
                branch.branch_ids_sha256,
                "{}: row digest",
                case.name
            );
        }
        assert_eq!(
            plan.input_tokens(),
            u64::from(case.response.usage.input_tokens),
            "{}: billed input tokens",
            case.name
        );
    }
}

#[test]
fn golden_replay_matches_the_pinned_checkpoint() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_SIEVE_MODEL_ROOT is not set");
        return;
    };
    let golden = golden();
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let engine = SieveEngine::load(SieveEngineConfig {
        model_root: root,
        limits: limits(),
        profile: &SIEVE_4B,
    })
    .expect("load pinned engine");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    for case in &golden.cases {
        let response = runtime
            .block_on(engine.evaluate(case.request.clone()))
            .unwrap_or_else(|error| panic!("case {}: {error}", case.name));
        for (question_id, expected) in &case.response.answers {
            let actual = response
                .answers
                .get(question_id)
                .unwrap_or_else(|| panic!("case {}: missing answer {question_id}", case.name));
            let actual = serde_json::to_value(actual).expect("serialize answer");
            assert_answer_matches(&case.name, &actual, expected);
        }
        assert_eq!(
            response.usage.output_tokens, 0,
            "{}: decision engines never generate output tokens",
            case.name
        );
        assert_eq!(
            response.usage.input_tokens, case.response.usage.input_tokens,
            "{}: usage drifted",
            case.name
        );
    }
}
