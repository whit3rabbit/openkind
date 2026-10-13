//! Offline parity and failure-mode tests for the pinned `d3-edge` profile
//! (the `d3` family's noncausal code-readout engine).
//!
//! Unit-level contract checks always run: the profile constants, the
//! code-table derivation, the golden fixtures' internal consistency, and the
//! fail-closed load paths (missing artifacts, and dummy artifacts that fail
//! the first digest check). The noncausal kernel's naive-reference test lives
//! beside the kernel. The golden replay, the prompt-statistics check against
//! the real tokenizer, and the staged decision-config drift run only when the
//! pinned artifacts are present locally (point `OPENKIND_D3_MODEL_ROOT` at
//! the model root); tests never download model artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::families::d3::{
    plan_prompt_ids, D3Engine, D3EngineConfig, D3Renderer, D3_EDGE, FAMILY_SLUG, PROFILES,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_engine::DecisionEngine;
use serde::Deserialize;

const DEFAULT_FIXTURE_DIR: &str = "d3_664f9ead734372935286";
const PROBABILITY_TOLERANCE: f64 = 0.005;

/// The fixture to replay. Set `OPENKIND_D3_FIXTURE` to the profile's
/// fixture directory name (for example `d3_e0dd32dbc16b27546692`) when the
/// downloaded weights belong to another family profile.
fn fixture_dir() -> PathBuf {
    let name =
        std::env::var("OPENKIND_D3_FIXTURE").unwrap_or_else(|_| DEFAULT_FIXTURE_DIR.to_owned());
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("OPENKIND_D3_MODEL_ROOT")
        .map(PathBuf::from)
        .filter(|root| root.is_dir())
}

/// The profile id recorded in a fixture's golden, for fail-closed tests.
fn golden_profile_id(dir: &Path) -> String {
    #[derive(Deserialize)]
    struct Id {
        profile_id: String,
    }
    serde_json::from_slice::<Id>(&fs::read(dir.join("golden.json")).expect("read golden"))
        .expect("golden profile id")
        .profile_id
}

fn profile_for_fixture(profile_id: &str) -> &'static openkind_backends::families::d3::D3Profile {
    openkind_backends::families::d3::profile_by_id(profile_id)
        .unwrap_or_else(|| panic!("golden names unknown profile {profile_id}"))
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
    profile_id: String,
    cases: Vec<GoldenCase>,
    /// Per-case prompt token counts and token-id digests. These were checked
    /// against the shipped reference renderer over the real tokenizer (see
    /// `docs/families/d3.md`), so they are the golden's reference-anchored
    /// evidence.
    prompt_stats: Vec<PromptStat>,
}

#[derive(Debug, Deserialize)]
struct PromptStat {
    name: String,
    token_count: usize,
    token_ids_sha256: String,
}

/// Digest of a prompt in the reference-parity script's serialization: the
/// comma-joined decimal ids, UTF-8 hashed.
fn token_ids_sha256(ids: &[u32]) -> String {
    use sha2::Digest as _;
    let joined = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    sha2::Sha256::digest(joined.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_golden(dir: &Path) -> Golden {
    serde_json::from_slice(&fs::read(dir.join("golden.json")).expect("read golden fixture"))
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
        D3_EDGE.profile_id,
        openkind_backends::families::support::derive_profile_id(
            FAMILY_SLUG,
            D3_EDGE.backbone_id,
            D3_EDGE.backbone_revision,
        )
    );
    assert_eq!(D3_EDGE.backbone_id, "vllm-sr/d3-edge");
    assert_eq!(
        D3_EDGE.backbone_revision,
        "18d61df78684675463af9ae732872478845c5a79"
    );
    // The d3-edge prune: the 0.8B hybrid at 12 layers with the causal mask
    // lifted in the full-attention layers.
    const {
        assert!(!D3_EDGE.geometry.causal);
        assert!(D3_EDGE.geometry.layer_count == 12);
        assert!(D3_EDGE.geometry.hidden_size == 1_024);
        assert!(D3_EDGE.temperature == 1.0);
        assert!(D3_EDGE.max_candidates == 255);
    }
}

#[test]
fn renderer_derives_the_reference_code_table() {
    let bytes = fs::read(fixture_dir().join("tokenizer-vocab-sample.json"))
        .expect("vendored tokenizer sample is checked in");
    let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).expect("sample parses");
    let derived = openkind_backends::families::d3::derive_answer_codes(&tokenizer)
        .expect("sample tokenizer provides the code table");
    let letters: Vec<String> = (b'A'..=b'Z')
        .map(|letter| (letter as char).to_string())
        .collect();
    let two_letter: Vec<String> = (0u8..255)
        .map(|index| {
            format!(
                "{}{}",
                (b'A' + index / 26) as char,
                (b'A' + index % 26) as char
            )
        })
        .collect();
    let mut expected = letters;
    expected.extend(two_letter);
    expected.truncate(255);
    assert_eq!(derived.0.len(), 255);
    assert_eq!(
        derived.0, expected,
        "derivation must follow d3_format order"
    );
    assert_eq!(derived.0[0], "A");
    assert_eq!(derived.0[26], "AA");
    // Every derived id decodes back to its code with the sample tokenizer.
    for (code, id) in derived.0.iter().zip(derived.1.iter()) {
        let decoded = tokenizer.decode(&[*id], false).expect("decode");
        assert_eq!(&decoded, code, "code {code} must round-trip");
    }
}

#[test]
fn load_fails_closed_without_artifacts() {
    let empty = std::env::temp_dir().join(format!("openkind-d3-missing-{}", std::process::id()));
    let _ = fs::remove_dir_all(&empty);
    fs::create_dir_all(&empty).expect("create empty root");
    let error = match D3Engine::load(D3EngineConfig {
        model_root: empty.clone(),
        limits: limits(),
        profile: &D3_EDGE,
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
fn golden_prompt_stats_cover_every_case() {
    // Always on: every profile's golden pairs each case with its
    // reference-anchored prompt statistics, and the usage the replay pins is
    // exactly that token count, so the replay's usage check is anchored to
    // the reference prompts too.
    for profile in PROFILES {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("{FAMILY_SLUG}_{}", profile.profile_id));
        let golden = read_golden(&dir);
        assert_eq!(golden.profile_id, profile.profile_id);
        let case_names: Vec<&str> = golden.cases.iter().map(|case| case.name.as_str()).collect();
        let stat_names: Vec<&str> = golden
            .prompt_stats
            .iter()
            .map(|stat| stat.name.as_str())
            .collect();
        assert_eq!(case_names, stat_names, "{}", profile.loader_id);
        for (case, stat) in golden.cases.iter().zip(&golden.prompt_stats) {
            assert_eq!(
                u64::from(case.response.usage.input_tokens),
                stat.token_count as u64,
                "{} {}: usage must equal the prompt token count",
                profile.loader_id,
                case.name
            );
            assert!(
                stat.token_ids_sha256.len() == 64
                    && stat
                        .token_ids_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit()),
                "{} {}: malformed token digest",
                profile.loader_id,
                case.name
            );
        }
    }
}

#[test]
fn golden_replay_matches_the_pinned_checkpoint() {
    let Some(root) = model_root() else {
        eprintln!("skipping: OPENKIND_D3_MODEL_ROOT is not set");
        return;
    };
    let golden = read_golden(&fixture_dir());
    assert!(!golden.cases.is_empty(), "golden fixture has cases");

    let profile = openkind_backends::families::d3::profile_by_id(&golden.profile_id)
        .unwrap_or_else(|| panic!("golden names unknown profile {}", golden.profile_id));
    let engine = D3Engine::load(D3EngineConfig {
        model_root: root.clone(),
        limits: limits(),
        profile,
    })
    .expect("load pinned engine");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    // The load above verified every artifact digest, so the renderer can
    // read the real tokenizer and pinned code table in place. Prompt
    // statistics need the real tokenizer, which is why this check is gated
    // with the replay.
    let decision_config: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("decision_config.json")).expect("read decision config"),
    )
    .expect("decode decision config");
    let codes: Vec<String> =
        serde_json::from_value(decision_config["codes"].clone()).expect("pinned codes");
    let token_ids: Vec<u64> =
        serde_json::from_value(decision_config["token_ids"].clone()).expect("pinned token ids");
    let renderer = D3Renderer::load(&root.join("tokenizer.json"), &codes, &token_ids, profile)
        .expect("load renderer");
    assert_eq!(golden.prompt_stats.len(), golden.cases.len());

    for (case, stat) in golden.cases.iter().zip(&golden.prompt_stats) {
        assert_eq!(case.name, stat.name, "prompt stats follow case order");
        let prompts = plan_prompt_ids(profile, &renderer, &case.request)
            .unwrap_or_else(|error| panic!("case {}: {error}", case.name));
        let [(_, ids)] = prompts.as_slice() else {
            panic!("case {} must plan exactly one question", case.name);
        };
        assert_eq!(
            ids.len(),
            stat.token_count,
            "{}: prompt token count drifted from the reference",
            case.name
        );
        assert_eq!(
            token_ids_sha256(ids),
            stat.token_ids_sha256,
            "{}: prompt token ids drifted from the reference",
            case.name
        );

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
fn load_fails_closed_on_artifact_digest_mismatch() {
    // Always on: dummy artifacts are present but carry the wrong bytes. The
    // loader verifies the small configs before any shard, and `config.json`
    // is the first digest it checks, so every profile fails closed there
    // without weights.
    let dummy = std::env::temp_dir().join(format!("openkind-d3-dummy-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dummy);
    fs::create_dir_all(&dummy).expect("stage dummy root");
    for file in [
        "config.json",
        "decision_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        fs::write(dummy.join(file), b"{}").expect("write dummy artifact");
    }
    let config_path = format!("{}`", dummy.join("config.json").display());
    for profile in PROFILES.iter().copied() {
        let error = match D3Engine::load(D3EngineConfig {
            model_root: dummy.clone(),
            limits: limits(),
            profile,
        }) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("{}: dummy artifacts must fail closed", profile.loader_id),
        };
        assert!(
            error.contains("SHA-256 mismatch") && error.contains(&config_path),
            "{}: unexpected error: {error}",
            profile.loader_id
        );
    }
    let _ = fs::remove_dir_all(&dummy);

    // With the pinned artifacts present, also drift `decision_config.json`
    // alone so the check fails past a genuine `config.json`.
    let Some(root) = model_root() else {
        eprintln!("skipping the staged-drift half: OPENKIND_D3_MODEL_ROOT is not set");
        return;
    };
    let staged = std::env::temp_dir().join(format!("openkind-d3-drift-{}", std::process::id()));
    let _ = fs::remove_dir_all(&staged);
    fs::create_dir_all(&staged).expect("stage model root");
    // Stage the cheap artifacts verbatim; the drifted config fails before
    // the checkpoint shard digest ever streams.
    for file in [
        "config.json",
        "decision_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "readout.safetensors",
    ] {
        if root.join(file).exists() {
            fs::copy(root.join(file), staged.join(file)).expect("stage artifact");
        }
    }
    let mut config = fs::read(staged.join("decision_config.json")).expect("read config");
    config.push(b' ');
    fs::write(staged.join("decision_config.json"), config).expect("write drifted config");

    let error = match D3Engine::load(D3EngineConfig {
        model_root: staged.clone(),
        limits: limits(),
        profile: profile_for_fixture(&golden_profile_id(&fixture_dir())),
    }) {
        Err(error) => error,
        Ok(_) => panic!("drifted decision config must fail closed"),
    };
    assert!(
        error.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {error}"
    );
    let _ = fs::remove_dir_all(&staged);
}
