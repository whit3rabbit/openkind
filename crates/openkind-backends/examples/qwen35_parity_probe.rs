use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::branch::{BranchBatch, BranchableState};
use openkind_backends::qwen35::{
    choose_strategy, native_profile_record, BackboneReference, BackboneState, Qwen35Backbone,
    Qwen35Embedding, Qwen35Layer0, SchedulerConfig, StrategyRequest, EXECUTION_ARITHMETIC_ID,
    PROFILE_ID,
};
use openkind_runtime::evidence::{
    generate_run_id, NativeRunWriter, RunEnvironment, SanitizedInvocation,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ContinuationTrace {
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<u32>,
    root_cache_bytes: usize,
    root_position: usize,
    question_position: usize,
    candidate_position: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: qwen35_parity_probe <checkpoint-root> <phase3b-reference-root>")?;
    let reference_root = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: qwen35_parity_probe <checkpoint-root> <phase3b-reference-root>")?;
    let stop_after = arguments
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "embedding".to_owned());
    let snapshot_path = arguments.next().map(PathBuf::from);
    let mut evidence_root = PathBuf::from("target/verification/native-runs");
    while let Some(flag) = arguments.next() {
        match flag.to_string_lossy().as_ref() {
            "--evidence-root" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "missing --evidence-root value".to_string())?;
                evidence_root = PathBuf::from(value);
            }
            other => {
                return Err(format!("unknown argument `{other}`").into());
            }
        }
    }

    let reference = BackboneReference::load(&reference_root)?;
    if stop_after == "persist-save" {
        let snapshot_path = snapshot_path.ok_or("persist-save requires a snapshot path")?;
        let trace: ContinuationTrace =
            serde_json::from_slice(&fs::read(reference_root.join("CONTINUATION_TRACE.json"))?)?;
        let backbone = Qwen35Backbone::load(checkpoint_root)?;
        let (_, root_state) = backbone.prefill(&trace.root_ids)?;
        root_state.persist_pinned(&snapshot_path)?;
        let report = serde_json::json!({
            "schema": "openkind-qwen35-persist-save/v1",
            "snapshot": snapshot_path,
            "position": root_state.position(),
            "tensor_storage_bytes": root_state.tensor_storage_bytes(),
            "content_fingerprint": root_state.strict_fingerprint().hex(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        record_persist_evidence(&evidence_root, "persist-save", &trace, redact(report))?;
        return Ok(());
    }
    if stop_after == "persist-replay" {
        let snapshot_path = snapshot_path.ok_or("persist-replay requires a snapshot path")?;
        let trace: ContinuationTrace =
            serde_json::from_slice(&fs::read(reference_root.join("CONTINUATION_TRACE.json"))?)?;
        let backbone = Qwen35Backbone::load(checkpoint_root)?;
        let root_state = BackboneState::restore_pinned(&snapshot_path)?;
        let before = root_state.strict_fingerprint();
        let (question_output, question_state) =
            backbone.continue_from(&root_state, &trace.question_ids)?;
        let (candidate_output, candidate_state) =
            backbone.continue_from(&question_state, &trace.candidate_suffix_ids)?;
        let mut full_ids = trace.root_ids.clone();
        full_ids.extend_from_slice(&trace.question_ids);
        full_ids.extend_from_slice(&trace.candidate_suffix_ids);
        let full_output = backbone.forward(&full_ids)?;
        let cached_vs_full = maximum_abs(candidate_output.final_token(), full_output.final_token());
        let root_immutable = before == root_state.strict_fingerprint();
        let passed = root_state.position() == trace.root_position
            && root_state.tensor_storage_bytes() == trace.root_cache_bytes
            && question_state.position() == trace.question_position
            && candidate_state.position() == trace.candidate_position
            && root_immutable
            && cached_vs_full == 0.0;
        let report = serde_json::json!({
            "schema": "openkind-qwen35-persist-replay/v1",
            "passed": passed,
            "snapshot": snapshot_path,
            "root_content_fingerprint": before.hex(),
            "root_immutable": root_immutable,
            "question_position": question_state.position(),
            "candidate_position": candidate_state.position(),
            "cached_vs_native_full_max_abs": cached_vs_full,
            "question_reference_max_abs": reference
                .compare("continuation.question_last_hidden", question_output.final_token())?
                .max_abs(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        record_persist_evidence(&evidence_root, "persist-replay", &trace, redact(report))?;
        if !passed {
            return Err("fresh-process persisted-state replay gate failed".into());
        }
        return Ok(());
    }
    if snapshot_path.is_some() {
        return Err("snapshot path is only valid for persist-save or persist-replay".into());
    }
    if stop_after == "branch" {
        return branch_parity_stage(checkpoint_root, &reference_root, &reference);
    }
    if stop_after == "continuation" {
        let trace: ContinuationTrace =
            serde_json::from_slice(&fs::read(reference_root.join("CONTINUATION_TRACE.json"))?)?;
        let backbone = Qwen35Backbone::load(checkpoint_root)?;
        let (root_output, root_state) = backbone.prefill(&trace.root_ids)?;
        let root_snapshot = root_state.clone();
        let root_comparison =
            reference.compare("continuation.root_last_hidden", root_output.final_token())?;
        let (question_output, question_state) =
            backbone.continue_from(&root_state, &trace.question_ids)?;
        let root_immutable = root_state == root_snapshot;
        let question_comparison = reference.compare(
            "continuation.question_last_hidden",
            question_output.final_token(),
        )?;
        let (candidate_output, candidate_state) =
            backbone.continue_from(&question_state, &trace.candidate_suffix_ids)?;
        let candidate_comparison = reference.compare(
            "continuation.candidate_last_hidden",
            candidate_output.final_token(),
        )?;
        let mut full_ids = trace.root_ids.clone();
        full_ids.extend_from_slice(&trace.question_ids);
        full_ids.extend_from_slice(&trace.candidate_suffix_ids);
        let full_output = backbone.forward(&full_ids)?;
        let cached_vs_full = maximum_abs(candidate_output.final_token(), full_output.final_token());
        let passed = root_state.position() == trace.root_position
            && question_state.position() == trace.question_position
            && candidate_state.position() == trace.candidate_position
            && root_state.byte_len() == trace.root_cache_bytes
            && root_state.layer_count() == 32
            && root_immutable;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "passed_structural_gates": passed,
                "root": {
                    "position": root_state.position(),
                    "cache_bytes": root_state.byte_len(),
                    "max_abs": root_comparison.max_abs(),
                    "rms": root_comparison.rms(),
                    "cosine": root_comparison.cosine(),
                },
                "question": {
                    "position": question_state.position(),
                    "max_abs": question_comparison.max_abs(),
                    "rms": question_comparison.rms(),
                    "cosine": question_comparison.cosine(),
                },
                "candidate": {
                    "position": candidate_state.position(),
                    "max_abs": candidate_comparison.max_abs(),
                    "rms": candidate_comparison.rms(),
                    "cosine": candidate_comparison.cosine(),
                    "cached_vs_native_full_max_abs": cached_vs_full,
                },
                "layer_states": root_state.layer_count(),
                "root_immutable": root_immutable,
            }))?
        );
        if !passed {
            return Err("native continuation structural gate failed".into());
        }
        return Ok(());
    }
    if stop_after == "trace" {
        let backbone = Qwen35Backbone::load(checkpoint_root)?;
        let output = backbone.forward(reference.trace_input_ids())?;
        let mut stages = Vec::with_capacity(34);
        let embedding = reference.compare("diagnostic.embedding", output.embedding_last_token())?;
        stages.push(serde_json::json!({
            "stage": "embedding",
            "max_abs": embedding.max_abs(),
            "rms": embedding.rms(),
            "cosine": embedding.cosine(),
        }));
        for layer_index in 0..32 {
            let stage = format!("layer_{layer_index:02}");
            let comparison = reference.compare(
                &format!("diagnostic.{stage}"),
                output
                    .layer_last_token(layer_index)
                    .ok_or("missing native layer output")?,
            )?;
            stages.push(serde_json::json!({
                "stage": stage,
                "max_abs": comparison.max_abs(),
                "rms": comparison.rms(),
                "cosine": comparison.cosine(),
            }));
        }
        let final_norm = reference.compare("diagnostic.final_norm", output.final_token())?;
        stages.push(serde_json::json!({
            "stage": "final_norm",
            "max_abs": final_norm.max_abs(),
            "rms": final_norm.rms(),
            "cosine": final_norm.cosine(),
        }));
        println!("{}", serde_json::to_string_pretty(&stages)?);
        return Ok(());
    }
    let (stage, token_count, hidden_size, comparison) = match stop_after.as_str() {
        "embedding" => {
            let embedding = Qwen35Embedding::load(checkpoint_root)?;
            let output = embedding.embed(reference.trace_input_ids())?;
            (
                "embedding",
                output.token_count(),
                output.hidden_size(),
                reference.compare("diagnostic.embedding", output.last_token())?,
            )
        }
        "layer_00" => {
            let layer = Qwen35Layer0::load(checkpoint_root)?;
            let output = layer.forward(reference.trace_input_ids())?;
            (
                "layer_00",
                output.token_count(),
                output.last_token().len(),
                reference.compare("diagnostic.layer_00", output.last_token())?,
            )
        }
        "final_norm" => {
            let backbone = Qwen35Backbone::load(checkpoint_root)?;
            let output = backbone.forward(reference.trace_input_ids())?;
            (
                "final_norm",
                output.token_count(),
                output.final_token().len(),
                reference.compare("diagnostic.final_norm", output.final_token())?,
            )
        }
        other if other.starts_with("layer_") => {
            let layer_index = other
                .strip_prefix("layer_")
                .expect("prefix checked")
                .parse::<usize>()?;
            let backbone = Qwen35Backbone::load(checkpoint_root)?;
            let output = backbone.forward(reference.trace_input_ids())?;
            let values = output
                .layer_last_token(layer_index)
                .ok_or_else(|| format!("layer index {layer_index} is out of range"))?;
            (
                other,
                output.token_count(),
                values.len(),
                reference.compare(&format!("diagnostic.{other}"), values)?,
            )
        }
        other => return Err(format!("unsupported stop stage `{other}`").into()),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "stage": stage,
            "token_count": token_count,
            "hidden_size": hidden_size,
            "max_abs": comparison.max_abs(),
            "rms": comparison.rms(),
            "cosine": comparison.cosine(),
            "exact": comparison.max_abs() == 0.0,
        }))?
    );
    if stage == "embedding" && comparison.max_abs() != 0.0 {
        return Err("native embedding does not exactly match diagnostic.embedding".into());
    }
    Ok(())
}

fn maximum_abs(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from((left - right).abs()))
        .fold(0.0, f64::max)
}

/// Replace the operator-chosen snapshot path before copying a report into an
/// evidence artifact, so evidence never records potentially sensitive paths.
fn redact(mut report: serde_json::Value) -> serde_json::Value {
    if report.get("snapshot").is_some() {
        report["snapshot"] = serde_json::json!("<redacted>");
    }
    report
}

/// Record one persist-gate invocation as a native-run evidence directory.
fn record_persist_evidence(
    evidence_root: &Path,
    stage: &str,
    trace: &ContinuationTrace,
    report: serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let scheduler = SchedulerConfig::for_pinned_profile(
        SchedulerConfig::LOWEST_MEASURED_SHARED_SAVINGS_RATIO,
        None,
    );
    let decision = choose_strategy(
        &scheduler,
        &StrategyRequest {
            root_tokens: trace.root_ids.len(),
            question_tokens: vec![trace.question_ids.len()],
            suffix_tokens: vec![vec![trace.candidate_suffix_ids.len()]],
        },
    );
    let directory = NativeRunWriter::begin(
        evidence_root,
        generate_run_id(),
        SanitizedInvocation {
            harness: "qwen35-parity-probe".to_owned(),
            command: "qwen35_parity_probe".to_owned(),
            parameters: serde_json::json!({
                "stage": stage,
                "root_tokens": trace.root_ids.len(),
                "question_tokens": trace.question_ids.len(),
                "candidate_suffix_tokens": trace.candidate_suffix_ids.len(),
            }),
            raw_argv_recorded: false,
        },
        RunEnvironment::capture(),
    )?
    .profile(native_profile_record(&scheduler, &decision))
    .parity(report)
    .finish()?;
    eprintln!("evidence recorded: {}", directory.display());
    Ok(())
}

/// Phase 3.4 checkpoint: root -> question -> candidate parity through the
/// backend-neutral `BranchableState` contract, plus fork, fan-out, gather,
/// identity, and exact byte-accounting gates.
fn branch_parity_stage(
    checkpoint_root: PathBuf,
    reference_root: &Path,
    reference: &BackboneReference,
) -> Result<(), Box<dyn Error>> {
    let trace: ContinuationTrace =
        serde_json::from_slice(&fs::read(reference_root.join("CONTINUATION_TRACE.json"))?)?;
    let backbone = Qwen35Backbone::load(checkpoint_root)?;

    let (root_output, root_state) = backbone.prefill(&trace.root_ids)?;
    let root_comparison =
        reference.compare("continuation.root_last_hidden", root_output.final_token())?;

    // 3.4a: profile-bound identity and lineage.
    let identity_matches_pinned_profile = root_state.profile_id().as_str() == PROFILE_ID
        && root_state.identity().arithmetic_id() == EXECUTION_ARITHMETIC_ID
        && root_state.lineage().fork_depth() == 0;
    let root_fingerprint = root_state.scheduling_fingerprint();
    let root_strict = root_state.strict_fingerprint();
    let untouched_root = root_state.clone();

    // 3.4b: exact byte accounting over the three tensor families.
    let breakdown = root_state.tensor_storage_breakdown();

    // 3.4c: single fork root -> question -> candidate.
    let branch = root_state.fork_one()?;
    let fork_preserved_fingerprint = branch.scheduling_fingerprint() == root_fingerprint;
    let (question_output, question_state) = backbone.continue_from(&branch, &trace.question_ids)?;
    let question_comparison = reference.compare(
        "continuation.question_last_hidden",
        question_output.final_token(),
    )?;
    let (candidate_output, candidate_state) =
        backbone.continue_from(&question_state, &trace.candidate_suffix_ids)?;
    let candidate_comparison = reference.compare(
        "continuation.candidate_last_hidden",
        candidate_output.final_token(),
    )?;

    // 3.4f: the cached candidate equals the independent full sequence exactly.
    let mut full_ids = trace.root_ids.clone();
    full_ids.extend_from_slice(&trace.question_ids);
    full_ids.extend_from_slice(&trace.candidate_suffix_ids);
    let (full_output, full_state) = backbone.prefill(&full_ids)?;
    let cached_vs_full_features =
        maximum_abs(candidate_output.final_token(), full_output.final_token());
    let cached_state_equals_full_state =
        candidate_state.strict_fingerprint() == full_state.strict_fingerprint();

    // 3.4d: batched fan-out with independently advanced lanes.
    let batch = root_state.fork_batch(2)?;
    let batch_bytes = batch.tensor_storage_bytes();
    let lane_root = batch.select(0)?;
    let (lane_question_output, lane_question_state) =
        backbone.continue_from(&lane_root, &trace.question_ids)?;
    let lane_candidate = batch.select(1)?;
    let (_, lane_candidate_state) =
        backbone.continue_from(&lane_candidate, &trace.candidate_suffix_ids)?;
    let lanes_advanced_independently = lane_question_state.position() == trace.question_position
        && lane_candidate_state.position()
            == trace.root_position + trace.candidate_suffix_ids.len()
        && lane_question_state.scheduling_fingerprint()
            != lane_candidate_state.scheduling_fingerprint()
        && lane_question_state.scheduling_fingerprint() != root_fingerprint
        && lane_candidate_state.scheduling_fingerprint() != root_fingerprint
        && batch.select(0)?.position() == trace.root_position
        && batch.select(1)?.scheduling_fingerprint() == root_fingerprint;
    let lane_replay_matches_single_fork = lane_question_state == question_state;
    let lane_feature_matches_single_fork = maximum_abs(
        lane_question_output.final_token(),
        question_output.final_token(),
    ) == 0.0;

    // 3.4e: gather with reorder and duplicate indices preserves lane identity.
    let gathered = batch.gather(&[1, 0, 1])?;
    let gather_preserved = gathered.select(0)? == batch.select(1)?
        && gathered.select(1)? == batch.select(0)?
        && gathered.select(2)? == batch.select(1)?;

    let root_immutable = root_state == untouched_root
        && root_state.scheduling_fingerprint() == root_fingerprint
        && root_state.strict_fingerprint() == root_strict;

    let structural_gates_passed = identity_matches_pinned_profile
        && fork_preserved_fingerprint
        && root_immutable
        && root_state.position() == trace.root_position
        && question_state.position() == trace.question_position
        && candidate_state.position() == trace.candidate_position
        && root_state.tensor_storage_bytes() == trace.root_cache_bytes
        && breakdown.tensor_storage_bytes() == trace.root_cache_bytes
        && root_state.layer_count() == 32
        && batch_bytes == 2 * trace.root_cache_bytes
        && lanes_advanced_independently
        && lane_replay_matches_single_fork
        && lane_feature_matches_single_fork
        && gather_preserved
        && cached_state_equals_full_state;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed_structural_gates": structural_gates_passed,
            "identity_matches_pinned_profile": identity_matches_pinned_profile,
            "root": {
                "position": root_state.position(),
                "tensor_storage_bytes": root_state.tensor_storage_bytes(),
                "attention_kv_bytes": breakdown.attention_kv_bytes,
                "recurrent_bytes": breakdown.recurrent_bytes,
                "convolution_bytes": breakdown.convolution_bytes,
                "scheduling_fingerprint": root_fingerprint.hex(),
                "max_abs": root_comparison.max_abs(),
                "rms": root_comparison.rms(),
                "cosine": root_comparison.cosine(),
            },
            "question": {
                "position": question_state.position(),
                "max_abs": question_comparison.max_abs(),
                "rms": question_comparison.rms(),
                "cosine": question_comparison.cosine(),
            },
            "candidate": {
                "position": candidate_state.position(),
                "max_abs": candidate_comparison.max_abs(),
                "rms": candidate_comparison.rms(),
                "cosine": candidate_comparison.cosine(),
                "cached_vs_native_full_max_abs": cached_vs_full_features,
                "cached_state_equals_full_state": cached_state_equals_full_state,
            },
            "fork": {
                "preserved_root_fingerprint": fork_preserved_fingerprint,
                "batch_bytes": batch_bytes,
                "lanes_advanced_independently": lanes_advanced_independently,
                "lane_replay_matches_single_fork": lane_replay_matches_single_fork,
                "gather_preserved_lane_identity": gather_preserved,
            },
            "root_immutable": root_immutable,
        }))?
    );
    if !structural_gates_passed {
        return Err("native branch-state structural gate failed".into());
    }
    Ok(())
}
