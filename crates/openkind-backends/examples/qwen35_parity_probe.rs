use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::branch::{BranchBatch, BranchableState};
use openkind_backends::qwen35::{
    choose_strategy, native_profile_record, BackboneReference, BackboneState, PolicyAction,
    PrimitiveKind, Qwen35Backbone, Qwen35Embedding, Qwen35Layer0, ReferenceBundle, SchedulerConfig,
    StrategyRequest, EXECUTION_ARITHMETIC_ID, PROBABILITY_TOLERANCE, PROFILE_ID,
};
use openkind_runtime::evidence::{
    generate_run_id, NativeRunWriter, RunEnvironment, SanitizedInvocation,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ContinuationTrace {
    question_id: String,
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<u32>,
    root_cache_bytes: usize,
    root_position: usize,
    question_position: usize,
    candidate_position: usize,
}

#[derive(Debug, Deserialize)]
struct TokenFixtures {
    records: Vec<TokenFixtureRecord>,
}

#[derive(Debug, Deserialize)]
struct TokenFixtureRecord {
    fixture_case: usize,
    question_id: String,
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<Vec<u32>>,
    full_candidate_ids: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    request: GoldenRequest,
}

#[derive(Debug, Deserialize)]
struct GoldenRequest {
    questions: Vec<GoldenQuestion>,
}

#[derive(Debug, Deserialize)]
struct GoldenQuestion {
    id: String,
    primitive: String,
}

/// Phase 3.5 cached-versus-full feature self-consistency guard.
const PERSIST_FEATURE_TOLERANCE: f64 = 0.000_1;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: qwen35_parity_probe <checkpoint-root> <phase3b-reference-root> [stage [snapshot-path [head-bundle-root]]]")?;
    let reference_root = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: qwen35_parity_probe <checkpoint-root> <phase3b-reference-root>")?;
    let stop_after = arguments
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "embedding".to_owned());
    let snapshot_path = arguments.next().map(PathBuf::from);
    let head_bundle_root = if stop_after == "persist-replay" {
        Some(
            arguments
                .next()
                .map(PathBuf::from)
                .ok_or("persist-replay requires a head bundle path")?,
        )
    } else {
        None
    };
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
        let head_bundle_root = head_bundle_root.expect("persist-replay bundle path checked above");
        let trace: ContinuationTrace =
            serde_json::from_slice(&fs::read(reference_root.join("CONTINUATION_TRACE.json"))?)?;
        let backbone = Qwen35Backbone::load(checkpoint_root)?;
        let root_state = BackboneState::restore_pinned(&snapshot_path)?;
        let report = persist_replay(
            &backbone,
            &root_state,
            &trace,
            &reference,
            &reference_root,
            &head_bundle_root,
            &snapshot_path,
        )?;
        let passed = report["passed"].as_bool().unwrap_or(false);
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

/// Replay the persisted root through every candidate in its fixture case and
/// compare the restored branches with independent full-sequence forwards.
fn persist_replay(
    backbone: &Qwen35Backbone,
    root_state: &BackboneState,
    trace: &ContinuationTrace,
    reference: &BackboneReference,
    reference_root: &Path,
    head_bundle_root: &Path,
    snapshot_path: &Path,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let token_fixtures: TokenFixtures =
        serde_json::from_slice(&fs::read(reference_root.join("TOKEN_FIXTURES.json"))?)?;
    let golden_cases: Vec<GoldenCase> =
        serde_json::from_slice(&fs::read(head_bundle_root.join("golden.json"))?)?;
    let bundle = ReferenceBundle::load(head_bundle_root)?;
    let matching_records: Vec<_> = token_fixtures
        .records
        .iter()
        .filter(|record| record.root_ids == trace.root_ids)
        .collect();
    if matching_records.is_empty() {
        return Err("no token fixture records match the persisted root IDs".into());
    }

    let before = root_state.strict_fingerprint();
    let (_, independent_root_state) = backbone.prefill(&trace.root_ids)?;
    let restored_root_matches_independent_prefill =
        before == independent_root_state.strict_fingerprint();
    let root_position_matches = root_state.position() == trace.root_position;
    let root_bytes_match = root_state.tensor_storage_bytes() == trace.root_cache_bytes;
    let mut maximum_candidate_feature_delta = 0.0_f64;
    let mut maximum_probability_delta = 0.0_f64;
    let mut argmax_changes = 0_usize;
    let mut policy_changes = 0_usize;
    let mut candidate_count = 0_usize;
    let mut trace_question_position_matches = false;
    let mut trace_candidate_position_matches = false;
    let mut trace_question_reference_max_abs = None;
    let mut trace_candidate_reference_max_abs = None;
    let mut trace_candidate_matches = 0_usize;
    let mut question_results = Vec::with_capacity(matching_records.len());

    for record in matching_records {
        if record.candidate_suffix_ids.len() != record.full_candidate_ids.len() {
            return Err(format!(
                "fixture case {} question {} has mismatched candidate segment counts",
                record.fixture_case, record.question_id
            )
            .into());
        }
        let golden_case = golden_cases
            .get(record.fixture_case)
            .ok_or("fixture case is out of range in head bundle")?;
        let golden_question = golden_case
            .request
            .questions
            .iter()
            .find(|question| question.id == record.question_id)
            .ok_or("fixture question is missing from the head bundle")?;
        let primitive = primitive(&golden_question.primitive)?;
        let (question_output, question_state) =
            backbone.continue_from(root_state, &record.question_ids)?;
        if question_state.position() != trace.root_position + record.question_ids.len() {
            return Err(format!(
                "restored question position mismatch for fixture question {}",
                record.question_id
            )
            .into());
        }

        if record.question_id == trace.question_id && record.question_ids == trace.question_ids {
            trace_question_position_matches = question_state.position() == trace.question_position;
            trace_question_reference_max_abs = Some(
                reference
                    .compare(
                        "continuation.question_last_hidden",
                        question_output.final_token(),
                    )?
                    .max_abs(),
            );
        }

        let mut restored_features = Vec::with_capacity(record.candidate_suffix_ids.len());
        let mut full_features = Vec::with_capacity(record.candidate_suffix_ids.len());
        let mut candidate_results = Vec::with_capacity(record.candidate_suffix_ids.len());
        let mut question_feature_delta = 0.0_f64;
        for (candidate_index, suffix_ids) in record.candidate_suffix_ids.iter().enumerate() {
            let expected_full_ids = &record.full_candidate_ids[candidate_index];
            let mut full_ids = record.root_ids.clone();
            full_ids.extend_from_slice(&record.question_ids);
            full_ids.extend_from_slice(suffix_ids);
            if &full_ids != expected_full_ids {
                return Err(format!(
                    "fixture case {} question {} candidate {candidate_index} segments do not match the exported full sequence",
                    record.fixture_case, record.question_id
                )
                .into());
            }

            let (restored_output, restored_candidate_state) =
                backbone.continue_from(&question_state, suffix_ids)?;
            if restored_candidate_state.position() != expected_full_ids.len() {
                return Err(format!(
                    "restored candidate position mismatch for fixture question {} candidate {candidate_index}",
                    record.question_id
                )
                .into());
            }

            // This starts from token zero and does not use the restored state.
            let full_output = backbone.forward(expected_full_ids)?;
            let feature_delta =
                maximum_abs(restored_output.final_token(), full_output.final_token());
            maximum_candidate_feature_delta = maximum_candidate_feature_delta.max(feature_delta);
            question_feature_delta = question_feature_delta.max(feature_delta);
            restored_features.push(restored_output.final_token().to_vec());
            full_features.push(full_output.final_token().to_vec());
            candidate_results.push(serde_json::json!({
                "candidate_index": candidate_index,
                "suffix_tokens": suffix_ids.len(),
                "restored_vs_full_feature_max_abs": feature_delta,
            }));

            if record.question_id == trace.question_id
                && record.question_ids == trace.question_ids
                && *suffix_ids == trace.candidate_suffix_ids
            {
                trace_candidate_matches += 1;
                trace_candidate_position_matches =
                    restored_candidate_state.position() == trace.candidate_position;
                trace_candidate_reference_max_abs = Some(
                    reference
                        .compare(
                            "continuation.candidate_last_hidden",
                            restored_output.final_token(),
                        )?
                        .max_abs(),
                );
            }
            candidate_count += 1;
        }

        let restored_evaluation = bundle.head().evaluate(primitive, &restored_features)?;
        let full_evaluation = bundle.head().evaluate(primitive, &full_features)?;
        let restored_probabilities = restored_evaluation.full_probabilities();
        let full_probabilities = full_evaluation.full_probabilities();
        let probability_delta = maximum_delta(&restored_probabilities, &full_probabilities)?;
        maximum_probability_delta = maximum_probability_delta.max(probability_delta);

        let argmax_changed = restored_evaluation.selected_candidate_index()
            != full_evaluation.selected_candidate_index();
        let policy_changed = restored_evaluation.policy_action() != full_evaluation.policy_action();
        argmax_changes += usize::from(argmax_changed);
        policy_changes += usize::from(policy_changed);
        question_results.push(serde_json::json!({
            "fixture_case": record.fixture_case,
            "question_id": record.question_id,
            "candidate_count": restored_features.len(),
            "candidate_feature_max_abs_delta": question_feature_delta,
            "full_probability_max_abs_delta": probability_delta,
            "restored_selected_candidate_index": restored_evaluation.selected_candidate_index(),
            "full_selected_candidate_index": full_evaluation.selected_candidate_index(),
            "argmax_changed": argmax_changed,
            "restored_policy": policy_action_report(restored_evaluation.policy_action()),
            "full_policy": policy_action_report(full_evaluation.policy_action()),
            "policy_changed": policy_changed,
            "candidates": candidate_results,
        }));
    }

    let root_immutable = before == root_state.strict_fingerprint();
    let trace_record_found = trace_candidate_matches == 1;
    let feature_gate_passed =
        candidate_count > 0 && maximum_candidate_feature_delta <= PERSIST_FEATURE_TOLERANCE;
    let decision_gate_passed = maximum_probability_delta <= PROBABILITY_TOLERANCE
        && argmax_changes == 0
        && policy_changes == 0;
    let structure_gate_passed = root_position_matches
        && root_bytes_match
        && root_immutable
        && restored_root_matches_independent_prefill
        && trace_question_position_matches
        && trace_candidate_position_matches
        && trace_record_found;
    let passed = structure_gate_passed && feature_gate_passed && decision_gate_passed;

    Ok(serde_json::json!({
        "schema": "openkind-qwen35-persist-replay/v2",
        "passed": passed,
        "structure_gate_passed": structure_gate_passed,
        "feature_gate_passed": feature_gate_passed,
        "decision_gate_passed": decision_gate_passed,
        "snapshot": snapshot_path,
        "fixture_case_questions": question_results.len(),
        "candidate_count": candidate_count,
        "root_content_fingerprint": before.hex(),
        "restored_root_matches_independent_prefill": restored_root_matches_independent_prefill,
        "root_immutable": root_immutable,
        "root_position": root_state.position(),
        "root_position_matches_fixture": root_position_matches,
        "root_tensor_storage_bytes": root_state.tensor_storage_bytes(),
        "root_tensor_storage_bytes_match_fixture": root_bytes_match,
        "trace_question_position_matches_fixture": trace_question_position_matches,
        "trace_candidate_position_matches_fixture": trace_candidate_position_matches,
        "trace_question_reference_max_abs": trace_question_reference_max_abs,
        "trace_candidate_reference_max_abs": trace_candidate_reference_max_abs,
        "maximum_candidate_feature_max_abs_delta": maximum_candidate_feature_delta,
        "candidate_feature_tolerance": PERSIST_FEATURE_TOLERANCE,
        "maximum_full_probability_max_abs_delta": maximum_probability_delta,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "argmax_changes": argmax_changes,
        "policy_changes": policy_changes,
        "questions": question_results,
    }))
}

fn primitive(value: &str) -> Result<PrimitiveKind, Box<dyn Error>> {
    match value {
        "choice" => Ok(PrimitiveKind::Choice),
        "noul" => Ok(PrimitiveKind::Noul),
        "score" => Ok(PrimitiveKind::Score),
        other => Err(format!("unexpected primitive {other}").into()),
    }
}

fn maximum_delta(left: &[f64], right: &[f64]) -> Result<f64, Box<dyn Error>> {
    if left.len() != right.len() {
        return Err(format!(
            "probability vector lengths differ: restored={}, full={}",
            left.len(),
            right.len()
        )
        .into());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max))
}

fn policy_action_report(action: PolicyAction) -> serde_json::Value {
    match action {
        PolicyAction::Accept { candidate_index } => serde_json::json!({
            "action": "accept",
            "candidate_index": candidate_index,
        }),
        PolicyAction::Review => serde_json::json!({"action": "review"}),
    }
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
