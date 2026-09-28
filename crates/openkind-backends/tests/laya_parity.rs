//! Offline parity and failure-mode tests for the pinned laya profiles.
//!
//! Unit-level contract checks always run. The golden replays run only when a
//! pinned checkpoint is present locally (point
//! `OPENKIND_LAYA_<PROFILE>_MODEL_ROOT` at the model root, with `<PROFILE>`
//! one of `ENGLISH`, `MULTILINGUAL`, `TYPED_DECISIONS`); tests never download
//! model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::laya::{
    LayaEngine, LayaEngineConfig, LayaProfile, FAMILY_SLUG, LAYA_ENGLISH, LAYA_MULTILINGUAL,
    LAYA_TYPED_DECISIONS,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;
use serde_json::json;

const PROBABILITY_TOLERANCE: f64 = 0.005;

/// One profile under test: its identity, fixture directory, and env var.
struct ProfileSpec {
    profile: &'static LayaProfile,
    fixture_dir: &'static str,
    env_var: &'static str,
}

const PROFILES: &[ProfileSpec] = &[
    ProfileSpec {
        profile: &LAYA_ENGLISH,
        fixture_dir: "tests/fixtures/laya-english_c8ea29bf1e33a343c4b7",
        env_var: "OPENKIND_LAYA_ENGLISH_MODEL_ROOT",
    },
    ProfileSpec {
        profile: &LAYA_MULTILINGUAL,
        fixture_dir: "tests/fixtures/laya-multilingual_f4064eb56fb7f7d325e1",
        env_var: "OPENKIND_LAYA_MULTILINGUAL_MODEL_ROOT",
    },
    ProfileSpec {
        profile: &LAYA_TYPED_DECISIONS,
        fixture_dir: "tests/fixtures/laya-typed-decisions_9d28cfa9567902801ed1",
        env_var: "OPENKIND_LAYA_TYPED_DECISIONS_MODEL_ROOT",
    },
];

fn fixture_dir(spec: &ProfileSpec) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(spec.fixture_dir)
}

fn model_root(spec: &ProfileSpec) -> Option<PathBuf> {
    std::env::var_os(spec.env_var)
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

fn replay_golden(spec: &ProfileSpec) {
    let Some(root) = model_root(spec) else {
        eprintln!("skipping: {} is not set", spec.env_var);
        return;
    };
    let golden: Golden = serde_json::from_slice(
        &fs::read(fixture_dir(spec).join("golden.json")).expect("read golden fixture"),
    )
    .expect("decode golden fixture");
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let engine = LayaEngine::load(LayaEngineConfig {
        profile: spec.profile,
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
fn profile_constants_are_self_consistent() {
    for spec in PROFILES {
        let profile = spec.profile;
        assert_eq!(
            profile.profile_id,
            openkind_backends::families::support::derive_profile_id(
                FAMILY_SLUG,
                profile.backbone_id,
                profile.backbone_revision,
            ),
            "{}",
            profile.loader_id
        );
        assert!(profile
            .temperature
            .iter()
            .all(|t| t.is_finite() && *t > 0.0));
        assert!(profile.head_max_len < profile.max_sequence_tokens);
    }
}

#[test]
fn load_fails_closed_without_artifacts() {
    for spec in PROFILES {
        let empty = std::env::temp_dir().join(format!(
            "openkind-laya-missing-{}-{}",
            spec.profile.loader_id,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&empty);
        fs::create_dir_all(&empty).expect("create empty root");
        let error = match LayaEngine::load(LayaEngineConfig {
            profile: spec.profile,
            model_root: empty.clone(),
            limits: limits(),
        }) {
            Err(error) => error,
            Ok(_) => panic!(
                "{}: missing artifacts must fail closed",
                spec.profile.loader_id
            ),
        };
        assert!(
            error.to_string().contains("missing") || error.to_string().contains("read"),
            "{}: unexpected error: {error}",
            spec.profile.loader_id
        );
        let _ = fs::remove_dir_all(&empty);
    }
}

#[test]
fn load_fails_closed_on_config_digest_mismatch() {
    for spec in PROFILES {
        let Some(root) = model_root(spec) else {
            eprintln!("skipping: {} is not set", spec.env_var);
            continue;
        };
        let staged = std::env::temp_dir().join(format!(
            "openkind-laya-config-{}-{}",
            spec.profile.loader_id,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&staged);
        fs::create_dir_all(staged.join("encoder")).expect("stage encoder dir");
        fs::create_dir_all(staged.join("tokenizer")).expect("stage tokenizer dir");
        // Copy only the cheap artifacts; the checkpoint is deliberately absent
        // so a malformed config fails before any large file is streamed.
        fs::copy(
            root.join("rl_agent_config.json"),
            staged.join("rl_agent_config.json"),
        )
        .expect("stage agent config");
        fs::copy(
            root.join("encoder/config.json"),
            staged.join("encoder/config.json"),
        )
        .expect("stage encoder config");
        for file in ["tokenizer.json", "tokenizer_config.json"] {
            fs::copy(
                root.join("tokenizer").join(file),
                staged.join("tokenizer").join(file),
            )
            .expect("stage tokenizer artifact");
        }
        // A placeholder shard satisfies the existence check without staging
        // the real multi-hundred-megabyte file; the drifted agent config
        // fails its digest before the shard digest is ever streamed.
        fs::write(staged.join("model.safetensors"), b"placeholder").expect("stage placeholder");
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("rl_agent_config.json")).expect("config"))
                .expect("decode agent config");
        config["max_len"] = json!(config["max_len"].as_u64().expect("max_len") + 1);
        fs::write(staged.join("rl_agent_config.json"), config.to_string())
            .expect("write drifted config");

        let error = match LayaEngine::load(LayaEngineConfig {
            profile: spec.profile,
            model_root: staged.clone(),
            limits: limits(),
        }) {
            Err(error) => error,
            Ok(_) => panic!(
                "{}: drifted config must fail closed",
                spec.profile.loader_id
            ),
        };
        assert!(
            error.to_string().contains("SHA-256 mismatch"),
            "{}: unexpected error: {error}",
            spec.profile.loader_id
        );
        let _ = fs::remove_dir_all(&staged);
    }
}

#[test]
fn golden_replay_matches_laya_english() {
    replay_golden(&PROFILES[0]);
}

#[test]
fn golden_replay_matches_laya_multilingual() {
    replay_golden(&PROFILES[1]);
}

#[test]
fn golden_replay_matches_laya_typed_decisions() {
    replay_golden(&PROFILES[2]);
}
