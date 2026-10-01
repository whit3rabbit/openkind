//! Offline parity and failure-mode tests for the pinned `decoder-logit-qwen3`
//! raw controls.
//!
//! Unit-level contract checks always run. The golden replays run only when
//! the pinned checkpoints are present locally (point
//! `OPENKIND_DECODER_LOGIT_QWEN3_06B_MODEL_ROOT`,
//! `OPENKIND_DECODER_LOGIT_QWEN3_17B_MODEL_ROOT`, or
//! `OPENKIND_DECODER_LOGIT_QWEN3_4B_MODEL_ROOT` at the model root); tests
//! never download model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::decoder_logit_qwen3::{
    profile_by_loader_id, AssistantTail, DecoderLogitQwen3Engine, DecoderLogitQwen3EngineConfig,
    PROFILES, QWEN3_06B, QWEN3_17B, QWEN3_4B,
};
use openkind_backends::families::support::{derive_profile_id, FamilyLimits};
use openkind_engine::DecisionEngine;
use serde::Deserialize;
use serde_json::json;

const PROBABILITY_TOLERANCE: f64 = 0.005;

fn fixture_dir(profile_id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/decoder_logit_qwen3_{profile_id}"))
}

fn model_root(env: &str) -> Option<PathBuf> {
    std::env::var_os(env)
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
    for profile in PROFILES {
        assert_eq!(
            profile.profile_id,
            derive_profile_id(
                "decoder-logit-qwen3",
                profile.backbone_id,
                profile.backbone_revision
            ),
            "{}",
            profile.loader_id
        );
        assert_eq!(
            profile.calibration(),
            openkind_backends::families::decoder_logit_qwen35::Calibration::Uniform(1.0)
        );
    }
    assert_eq!(QWEN3_06B.assistant_tail, AssistantTail::EmptyThinkBlock);
    assert_eq!(QWEN3_17B.assistant_tail, AssistantTail::EmptyThinkBlock);
    assert_eq!(QWEN3_4B.assistant_tail, AssistantTail::BareGenerationPrompt);
    assert_eq!(QWEN3_06B.num_hidden_layers, 28);
    assert_eq!(QWEN3_17B.num_hidden_layers, 28);
    assert_eq!(QWEN3_4B.num_hidden_layers, 36);
    assert_eq!(QWEN3_4B.num_attention_heads, 32);
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!(
        "openkind-decoder-logit-qwen3-missing-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match DecoderLogitQwen3Engine::load(DecoderLogitQwen3EngineConfig {
        profile: &QWEN3_06B,
        model_root: empty.clone(),
        limits: limits(),
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
fn load_fails_closed_on_wide_questions_without_a_model() {
    // The renderer rejects >16-option passes before any checkpoint access.
    let error = match DecoderLogitQwen3Engine::load(DecoderLogitQwen3EngineConfig {
        profile: &QWEN3_06B,
        model_root: std::env::temp_dir().join(format!("missing-{}", std::process::id())),
        limits: limits(),
    }) {
        Err(error) => error,
        Ok(_) => panic!("missing root must fail closed"),
    };
    assert!(error.to_string().contains("missing") || error.to_string().contains("read"));
}

fn golden_replay(loader_id: &'static str, env: &str) {
    let profile = profile_by_loader_id(loader_id).expect("pinned profile");
    let Some(root) = model_root(env) else {
        eprintln!("skipping: {env} is not set");
        return;
    };
    let golden: Golden = serde_json::from_slice(
        &fs::read(fixture_dir(profile.profile_id).join("golden.json"))
            .expect("read golden fixture"),
    )
    .expect("decode golden fixture");
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let engine = DecoderLogitQwen3Engine::load(DecoderLogitQwen3EngineConfig {
        profile,
        model_root: root,
        limits: limits(),
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

#[test]
fn golden_replay_06b_matches_the_pinned_checkpoint() {
    golden_replay(
        QWEN3_06B.loader_id,
        "OPENKIND_DECODER_LOGIT_QWEN3_06B_MODEL_ROOT",
    );
}

#[test]
fn golden_replay_17b_matches_the_pinned_checkpoint() {
    golden_replay(
        QWEN3_17B.loader_id,
        "OPENKIND_DECODER_LOGIT_QWEN3_17B_MODEL_ROOT",
    );
}

#[test]
fn golden_replay_4b_matches_the_pinned_checkpoint() {
    golden_replay(
        QWEN3_4B.loader_id,
        "OPENKIND_DECODER_LOGIT_QWEN3_4B_MODEL_ROOT",
    );
}

#[test]
fn renderer_declares_the_letter_pass_payload() {
    // The payload shape mirrors the jevk5 letter pass; the tokenizer for it
    // is the vendored synthetic one from the sibling fixture, proving the
    // renderer is loadable offline.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/decoder_logit_qwen35_415bcf4a064e6dadcf85/synthetic_control_tokenizer.json",
    );
    openkind_backends::families::support::verify_digest(
        &path,
        "790e5d78a52353fcb7766098a8e0044c6d229448602eb78b05a81c3f875315c4",
    )
    .expect("synthetic tokenizer digest");
    let renderer =
        openkind_backends::families::decoder_logit_qwen3::renderer::Qwen3ControlRenderer::load(
            &path,
            AssistantTail::EmptyThinkBlock,
        )
        .expect("renderer");
    let pass = renderer
        .render_pass(
            &json!("evidence"),
            "Pick one",
            &[
                ("alpha".to_owned(), "Alpha text".to_owned()),
                ("beta".to_owned(), "Beta text".to_owned()),
            ],
        )
        .expect("render");
    assert_eq!(pass.letter_ids().len(), 2);
    assert!(!pass.prompt_ids().is_empty());
}
