//! Prompt-parity and fail-closed tests for the Perplexity Decider v1.3
//! profiles (`pplx-decider-flash`, `pplx-decider-27b`), the chunked-readout
//! layout of the `d3` family.
//!
//! `reference-prompts.json` holds token ids produced by an independent
//! implementation: the checkpoint's real `chat_template.jinja` rendered with
//! jinja2 and tokenized with the pinned `tokenizer.json` through the
//! `tokenizers` Python bindings, following the shipped
//! `pplx_decider.protocol` (version 4). The byte-parity test replays those
//! cases through the Rust renderer when `OPENKIND_PPLX_FLASH_MODEL_ROOT`
//! names a directory holding the pinned `tokenizer.json` and
//! `decision_config.json`; otherwise it skips. Tests never download
//! artifacts. A skipped parity test is not evidence.

use std::path::{Path, PathBuf};

use openkind_backends::families::d3::{
    plan_prompt_ids, profile_by_id, profile_by_short_name, D3Engine, D3EngineConfig, D3Renderer,
    PromptLayout, PPLX_27B, PPLX_FLASH,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_core::SystemRequest;
use serde::Deserialize;
use serde_json::json;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/d3_3722c64ec29836dc7608/reference-prompts.json")
}

#[derive(Deserialize)]
struct ReferenceCase {
    state: serde_json::Value,
    question: serde_json::Value,
    ids: Vec<u32>,
}

#[test]
fn profiles_are_registered_with_the_chunked_layout() {
    for (short, profile) in [("pplx-flash", &PPLX_FLASH), ("pplx-27b", &PPLX_27B)] {
        assert_eq!(profile_by_short_name(short), Some(profile));
        assert_eq!(profile_by_id(profile.profile_id), Some(profile));
        assert_eq!(profile.layout, PromptLayout::PplxChunked);
        assert!(!profile.geometry.causal);
    }
    assert_eq!(PPLX_FLASH.geometry.hidden_size, 2_560);
    assert_eq!(PPLX_FLASH.geometry.layer_count, 32);
    assert_eq!(PPLX_27B.geometry.hidden_size, 5_120);
    assert_eq!(PPLX_27B.geometry.layer_count, 64);
    // Distilled student and teacher have separately fitted temperatures.
    assert_ne!(PPLX_FLASH.temperature, PPLX_27B.temperature);
}

#[test]
fn loading_without_artifacts_fails_closed() {
    let empty = tempfile::tempdir().expect("tempdir");
    for profile in [&PPLX_FLASH, &PPLX_27B] {
        let result = D3Engine::load(D3EngineConfig {
            profile,
            model_root: empty.path().to_path_buf(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 1,
                retry_after_ms: 1_000,
                evaluation_timeout: None,
            },
        });
        assert!(result.is_err(), "{} must not load", profile.loader_id);
    }
}

#[test]
fn reference_prompts_match_the_rust_renderer() {
    let Some(root) = std::env::var_os("OPENKIND_PPLX_FLASH_MODEL_ROOT").map(PathBuf::from) else {
        eprintln!("skipped: OPENKIND_PPLX_FLASH_MODEL_ROOT is unset (not parity evidence)");
        return;
    };
    let decision_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("decision_config.json")).expect("decision_config.json"),
    )
    .expect("decision config json");
    let codes: Vec<String> = serde_json::from_value(decision_config["codes"].clone()).unwrap();
    let token_ids: Vec<u64> = serde_json::from_value(decision_config["token_ids"].clone()).unwrap();
    let renderer = D3Renderer::load(
        &root.join("tokenizer.json"),
        &codes,
        &token_ids,
        &PPLX_FLASH,
    )
    .expect("renderer over the pinned tokenizer");

    let cases: std::collections::BTreeMap<String, ReferenceCase> =
        serde_json::from_slice(&std::fs::read(fixture()).expect("fixture")).expect("cases");
    assert!(cases.len() >= 9);
    for (name, case) in cases {
        // The wire requires the `instructions` key; the reference reads a
        // missing key and a null alike through `question.get(...) or default`.
        let mut question = case.question.clone();
        question
            .as_object_mut()
            .expect("question object")
            .entry("instructions")
            .or_insert(serde_json::Value::Null);
        let request: SystemRequest = serde_json::from_value(json!({
            "model": "m",
            "state": case.state,
            "questions": {"q": question},
        }))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let planned = plan_prompt_ids(&PPLX_FLASH, &renderer, &request)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(planned.len(), 1);
        assert_eq!(
            planned[0].1, case.ids,
            "{name}: token ids differ from the reference"
        );
    }
}

/// Runs the README example request through the real checkpoint when the
/// model root holds the pinned shards. Prints the answers and, when
/// `OPENKIND_PPLX_OUT` names a file, writes them there as JSON so a reference
/// run can be compared. Skips otherwise; a skip is not parity evidence.
#[test]
fn engine_answers_the_readme_example_with_real_weights() {
    let Some(root) = std::env::var_os("OPENKIND_PPLX_FLASH_MODEL_ROOT").map(PathBuf::from) else {
        eprintln!("skipped: OPENKIND_PPLX_FLASH_MODEL_ROOT is unset");
        return;
    };
    if !root.join(PPLX_FLASH.shards[0].0).is_file() {
        eprintln!("skipped: no checkpoint shards under the model root");
        return;
    }
    let engine = D3Engine::load(D3EngineConfig::new(&PPLX_FLASH, &root)).expect("load engine");
    check_readme_example(&engine, true);
}

/// The same check on the MLX backend (FP32 for this profile).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mlx_engine_answers_the_readme_example_with_real_weights() {
    let Some(root) = std::env::var_os("OPENKIND_PPLX_FLASH_MODEL_ROOT").map(PathBuf::from) else {
        eprintln!("skipped: OPENKIND_PPLX_FLASH_MODEL_ROOT is unset");
        return;
    };
    if !root.join(PPLX_FLASH.shards[0].0).is_file() {
        eprintln!("skipped: no checkpoint shards under the model root");
        return;
    }
    let engine =
        openkind_backends::families::d3::D3MlxEngine::load(D3EngineConfig::new(&PPLX_FLASH, &root))
            .expect("load mlx engine");
    check_readme_example(&engine, true);
}

fn check_readme_example(engine: &impl openkind_engine::DecisionEngine, compare_reference: bool) {
    let request: SystemRequest = serde_json::from_value(json!({
        "model": "m",
        "state": "My integration keeps failing with a 401 since we rotated our API key. Please help.",
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this request?",
                "criteria": {"billing": "Charges and refunds", "sales": "Questions about buying a product", "support": "Technical integration errors"}},
            "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"},
            "frustration": {"type": "score", "instructions": null, "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]},
        },
    }))
    .expect("request");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let started = std::time::Instant::now();
    let response = runtime
        .block_on(engine.evaluate(request))
        .expect("evaluate");
    eprintln!("evaluated in {:?}", started.elapsed());
    let answers = serde_json::to_value(&response.answers).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&answers).unwrap());
    if let Some(out) = std::env::var_os("OPENKIND_PPLX_OUT") {
        std::fs::write(out, serde_json::to_vec_pretty(&answers).unwrap()).unwrap();
    }
    assert_eq!(response.usage.output_tokens, 0);

    if compare_reference {
        // Compare with the shipped PyTorch reference's answers for the same
        // request. Both ran FP32 compute on CPU; the acceptance tolerance is the
        // project's 0.005 probability tolerance (observed differences are far
        // smaller, about 2e-6).
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/d3_3722c64ec29836dc7608/pytorch-reference-answers.json"),
            )
            .expect("reference answers"),
        )
        .expect("reference json");
        assert_eq!(
            u64::from(response.usage.input_tokens),
            reference["input_tokens"].as_u64().unwrap(),
            "usage must equal the reference's token count"
        );
        let close = |label: &str, actual: &serde_json::Value, expected: &serde_json::Value| {
            let (a, e) = (actual.as_f64().unwrap(), expected.as_f64().unwrap());
            assert!((a - e).abs() <= 0.005, "{label}: {a} vs reference {e}");
        };
        for (question, key) in [("team", "team"), ("frustration", "frustration")] {
            for (option, expected) in reference[key].as_object().unwrap() {
                close(
                    &format!("{question}/{option}"),
                    &answers[question]["probabilities"][option],
                    expected,
                );
            }
        }
        close("urgent", &answers["urgent"]["noul"], &reference["urgent"]);
    }
    assert_eq!(answers["team"]["choice"], "support");
}

/// The 27B profile on MLX (affine 4-bit at load). No PyTorch reference exists
/// for it on this host (BF16 weights exceed its memory), so this checks that
/// the pinned artifacts verify, the quantized engine loads and answers, and
/// the obvious argmax holds. It is not parity evidence.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mlx_27b_engine_answers_the_readme_example_with_real_weights() {
    let Some(root) = std::env::var_os("OPENKIND_PPLX_27B_MODEL_ROOT").map(PathBuf::from) else {
        eprintln!("skipped: OPENKIND_PPLX_27B_MODEL_ROOT is unset");
        return;
    };
    if !root.join(PPLX_27B.shards[0].0).is_file() {
        eprintln!("skipped: no checkpoint shards under the model root");
        return;
    }
    let engine =
        openkind_backends::families::d3::D3MlxEngine::load(D3EngineConfig::new(&PPLX_27B, &root))
            .expect("load 27B mlx engine");
    check_readme_example(&engine, false);
}
