use std::fs;
use std::path::PathBuf;

use super::records::*;
use super::time::*;
use super::writer::*;

fn temp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "opendecision-evidence-{tag}-{}-{}",
        std::process::id(),
        unix_now()
    ))
}

fn invocation() -> SanitizedInvocation {
    SanitizedInvocation {
        harness: "qwen35_scheduler_stress".to_owned(),
        command: "qwen35_scheduler_stress".to_owned(),
        parameters: serde_json::json!({"k_cells": [32, 64, 128, 255]}),
        raw_argv_recorded: false,
    }
}

fn environment() -> RunEnvironment {
    RunEnvironment {
        git_commit: Some("35c481a".to_owned()),
        git_dirty: Some(false),
        rustc_version: Some("rustc 1.98.0".to_owned()),
        os: "macos".to_owned(),
        arch: "aarch64".to_owned(),
    }
}

#[test]
fn writer_produces_run_profile_and_verifiable_checksums() {
    let root = temp_root("full");
    let profile = ProfileRecord {
        profile_id: "a047d".to_owned(),
        bundle_sha256: "0".repeat(64),
        backbone_id: "Qwen/Qwen3.5-4B-Base".to_owned(),
        backbone_revision: "1001bb4".to_owned(),
        renderer_id: "state_first".to_owned(),
        tokenizer_digest: "1".repeat(64),
        arithmetic_id: "candle-cpu-fp32".to_owned(),
        calibration_temperature: 1.818_679_991_044_277_7,
        policy_threshold: 0.98,
        probability_space: "offered_options_plus_semantic_none".to_owned(),
        backend: BackendRecord {
            backend_id: "qwen35-native-cpu".to_owned(),
            backend_version: "candle-core 0.8.0".to_owned(),
            allocator: "system".to_owned(),
            capabilities: serde_json::json!({
                "vectorized_question_forward": false,
                "vectorized_candidate_forward": false
            }),
        },
        execution: ExecutionRecord {
            execution_plan: "nested_sequential".to_owned(),
            batch_forward_mode: "per_lane".to_owned(),
            forced: false,
            scheduler: serde_json::json!({"min_shared_savings_ratio": 2.52}),
        },
    };

    let directory = NativeRunWriter::begin(&root, "20260920T000000Z", invocation(), environment())
        .expect("begin")
        .profile(profile)
        .parity(serde_json::json!({"passed": true}))
        .predictions(vec![serde_json::json!({"k": 32, "admitted": true})])
        .finish()
        .expect("finish");

    assert_eq!(directory, root.join("20260920T000000Z"));
    let run: RunRecord =
        serde_json::from_str(&fs::read_to_string(directory.join("RUN.json")).unwrap()).unwrap();
    assert_eq!(run.schema, NATIVE_RUN_SCHEMA);
    assert!(!run.invocation.raw_argv_recorded);
    assert!(!run.contains_input_content);
    assert!(!run.contains_sensitive_paths);
    assert!(run.finished_utc.ends_with('Z'));
    assert!(directory.join("PROFILE.json").exists());
    assert!(directory.join("PARITY.json").exists());
    assert!(!directory.join("PERFORMANCE.json").exists());
    let rows = fs::read_to_string(directory.join("predictions.jsonl")).unwrap();
    assert_eq!(rows.lines().count(), 1);

    let checksums: ChecksumsRecord =
        serde_json::from_str(&fs::read_to_string(directory.join("checksums.json")).unwrap())
            .unwrap();
    assert_eq!(checksums.schema, CHECKSUMS_SCHEMA);
    for (name, expected) in &checksums.files {
        let bytes = fs::read(directory.join(name)).unwrap();
        assert_eq!(&sha256_hex(&bytes), expected, "checksum matches {name}");
    }
    assert!(checksums.files.contains_key("RUN.json"));
    assert!(checksums.files.contains_key("predictions.jsonl"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn writer_rejects_traversal_run_ids() {
    let root = temp_root("traversal");
    assert!(matches!(
        NativeRunWriter::begin(&root, "../escape", invocation(), environment()),
        Err(EvidenceError::InvalidRunId(_))
    ));
    assert!(matches!(
        NativeRunWriter::begin(&root, "run/with/slashes", invocation(), environment()),
        Err(EvidenceError::InvalidRunId(_))
    ));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn utc_formatting_covers_epoch_and_leap_days() {
    assert_eq!(format_utc_timestamp(0), "1970-01-01T00:00:00Z");
    assert_eq!(format_utc_timestamp(86_400), "1970-01-02T00:00:00Z");
    assert_eq!(format_utc_timestamp(951_782_400), "2000-02-29T00:00:00Z");
    // 2026-09-20T00:00:00Z in Unix seconds (documentation baseline date).
    assert_eq!(format_utc_timestamp(1_789_862_400), "2026-09-20T00:00:00Z");
    assert_eq!(generate_run_id_at(1_789_862_400), "20260920T000000Z");
    let run_id = generate_run_id();
    assert!(run_id.starts_with("20") && run_id.ends_with('Z') && run_id.len() == 16);
}
