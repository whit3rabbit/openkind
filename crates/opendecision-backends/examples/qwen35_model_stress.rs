//! Phase 3.9b representative model-backed high-K stress.
//!
//! This streams candidate continuations and discards each candidate state so
//! it measures model execution without allocating a naive K-wide state fan-out.
//! It is intentionally checkpoint-gated and never downloads artifacts.
//!
//! Emits the JSON report to stdout and records a native-run evidence
//! directory (`opendecision-native-run/v1`), defaulting to the ephemeral
//! `target/verification/native-runs`; pass `--evidence-root research/native`
//! to record committed evidence.

use std::env;
use std::error::Error;
use std::path::PathBuf;

use opendecision_backends::qwen35::{
    choose_strategy, native_profile_record, Qwen35Backbone, SchedulerConfig, StrategyRequest,
};
use opendecision_runtime::branch::BranchableState;
use opendecision_runtime::evidence::{
    generate_run_id, NativeRunWriter, RunEnvironment, SanitizedInvocation,
};
use opendecision_runtime::peak_resident_bytes;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = PathBuf::from(arguments.next().ok_or("missing checkpoint root")?);
    let candidates: usize = arguments
        .next()
        .ok_or("missing candidate count (32, 64, 128, or 255)")?
        .to_string_lossy()
        .parse()?;
    if ![32, 64, 128, 255].contains(&candidates) {
        return Err("candidate count must be 32, 64, 128, or 255".into());
    }
    let mut evidence_root = PathBuf::from("target/verification/native-runs");
    while let Some(flag) = arguments.next() {
        match flag.to_string_lossy().as_ref() {
            "--evidence-root" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "missing --evidence-root value".to_string())?;
                evidence_root = PathBuf::from(value);
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    let backbone = Qwen35Backbone::load(checkpoint_root)?;
    let root_ids: Vec<u32> = (0..98)
        .map(|index| ((index * 7_919 + 13) % 248_320) as u32)
        .collect();
    let question_ids = [41_u32, 42, 43, 44];
    let (_, root_state) = backbone.prefill(&root_ids)?;
    let root_fingerprint = root_state.strict_fingerprint();
    let (_, question_state) = backbone.continue_from(&root_state, &question_ids)?;
    let peak_before_candidates = peak_resident_bytes()?;
    let mut checksum = 0.0_f64;
    for candidate in 0..candidates {
        let suffix = [u32::try_from(1000 + candidate)?];
        let (output, candidate_state) = backbone.continue_from(&question_state, &suffix)?;
        checksum += output
            .final_token()
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        drop(candidate_state);
    }
    let root_immutable = root_state.strict_fingerprint() == root_fingerprint;
    let peak_after_candidates = peak_resident_bytes()?;
    let report = serde_json::json!({
        "schema": "opendecision-qwen35-model-stress/v1",
        "scope": "representative streaming CPU continuations; not vectorized K-wide throughput",
        "candidates": candidates,
        "root_tensor_bytes": root_state.tensor_storage_bytes(),
        "question_tensor_bytes": question_state.tensor_storage_bytes(),
        "peak_resident_before_candidates_bytes": peak_before_candidates,
        "peak_resident_after_candidates_bytes": peak_after_candidates,
        "root_immutable": root_immutable,
        "feature_checksum": checksum,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !root_immutable || !checksum.is_finite() {
        return Err("model-backed stress invariant failed".into());
    }

    // The harness streams from one shared root, so the recorded execution is
    // the pinned CPU policy's decision for this exact shape.
    let scheduler = SchedulerConfig::for_pinned_profile(
        SchedulerConfig::LOWEST_MEASURED_SHARED_SAVINGS_RATIO,
        None,
    );
    let decision = choose_strategy(
        &scheduler,
        &StrategyRequest {
            root_tokens: root_ids.len(),
            question_tokens: vec![question_ids.len()],
            suffix_tokens: vec![vec![1; candidates]],
        },
    );
    let directory = NativeRunWriter::begin(
        &evidence_root,
        generate_run_id(),
        SanitizedInvocation {
            harness: "qwen35-model-stress".to_owned(),
            command: "qwen35_model_stress".to_owned(),
            parameters: serde_json::json!({
                "candidates": candidates,
                "scope": "representative streaming CPU continuations; not vectorized K-wide throughput",
            }),
            raw_argv_recorded: false,
        },
        RunEnvironment::capture(),
    )?
    .profile(native_profile_record(&scheduler, &decision))
    .parity(report.clone())
    .memory(serde_json::json!({
        "peak_resident_before_candidates_bytes": peak_before_candidates,
        "peak_resident_after_candidates_bytes": peak_after_candidates,
        "root_tensor_bytes": root_state.tensor_storage_bytes(),
        "question_tensor_bytes": question_state.tensor_storage_bytes(),
    }))
    .predictions(vec![report])
    .finish()?;
    eprintln!("evidence recorded: {}", directory.display());
    Ok(())
}
