//! Phase 3.9a scheduler/admission stress with no model weights or large state allocation.
//!
//! Emits the JSON report to stdout and records a native-run evidence
//! directory (`openkind-native-run/v1`). The evidence root defaults to
//! the ephemeral `target/verification/native-runs` so verification runs never
//! dirty the working tree; pass `--evidence-root research/native` to record
//! committed evidence.

use std::error::Error;
use std::path::PathBuf;

use openkind_backends::qwen35::{
    choose_strategy, native_profile_record, ProcessMemoryEnvelope, SchedulerConfig,
    StrategyDecision, StrategyRequest,
};
use openkind_runtime::evidence::{
    generate_run_id, NativeRunWriter, RunEnvironment, SanitizedInvocation,
};
use openkind_runtime::{peak_resident_bytes, BackendCapabilities};

fn main() -> Result<(), Box<dyn Error>> {
    let mut evidence_root = PathBuf::from("target/verification/native-runs");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--evidence-root" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing --evidence-root value".to_owned())?;
                evidence_root = PathBuf::from(value);
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    let observed = peak_resident_bytes()?;
    let envelope = ProcessMemoryEnvelope {
        observed_resident_bytes: observed,
        forward_scratch_bytes: 1024 * 1024 * 1024,
        allocator_headroom_bytes: 512 * 1024 * 1024,
        max_process_bytes: 12 * 1024 * 1024 * 1024,
    };
    let mut rows = Vec::new();
    let mut representative: Option<(SchedulerConfig, StrategyDecision)> = None;
    for candidates in [32, 64, 128, 255] {
        let request = StrategyRequest {
            root_tokens: 256,
            question_tokens: vec![8, 13, 21],
            suffix_tokens: vec![
                vec![7; candidates],
                vec![11; candidates],
                vec![17; candidates],
            ],
        };
        let cpu = SchedulerConfig::for_pinned_profile(
            SchedulerConfig::LOWEST_MEASURED_SHARED_SAVINGS_RATIO,
            Some(8 * 1024 * 1024 * 1024),
        )
        .with_process_memory(envelope);
        let limited_vectorized = cpu.clone().with_backend_capabilities(
            BackendCapabilities::fully_vectorized().with_lane_limits(8, 128),
        );
        let cpu_decision = choose_strategy(&cpu, &request);
        let vectorized_decision = choose_strategy(&limited_vectorized, &request);
        rows.push(serde_json::json!({
            "questions": 3,
            "candidates_per_question": candidates,
            "cpu_plan": cpu_decision.strategy.as_str(),
            "cpu_batch_forward_mode": cpu_decision.batch_forward_mode.as_str(),
            "cpu_admitted": cpu_decision.admitted,
            "limited_vectorized_plan": vectorized_decision.strategy.as_str(),
            "limited_vectorized_batch_forward_mode":
                vectorized_decision.batch_forward_mode.as_str(),
            "limited_vectorized_admitted": vectorized_decision.admitted,
            "repeated_full_tensor_bytes": cpu_decision.retention.repeated_full_tensor_bytes,
            "nested_sequential_tensor_bytes": cpu_decision.retention.nested_sequential_tensor_bytes,
            "nested_batched_tensor_bytes": cpu_decision.retention.nested_batched_tensor_bytes,
            "nested_sequential_process_peak_bytes": cpu_decision.retention.nested_sequential_process_peak_bytes,
            "nested_batched_process_peak_bytes": cpu_decision.retention.nested_batched_process_peak_bytes,
        }));
        representative = Some((cpu, cpu_decision));
    }
    let (profile_config, profile_decision) = representative.expect("stress cells are non-empty");
    let report = serde_json::json!({
        "schema": "openkind-qwen35-scheduler-stress/v1",
        "scope": "state and scheduler estimates only; no model forward",
        "observed_process_peak_resident_bytes": observed,
        "rows": rows.clone(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);

    let directory = NativeRunWriter::begin(
        &evidence_root,
        generate_run_id(),
        SanitizedInvocation {
            harness: "qwen35-scheduler-stress".to_owned(),
            command: "qwen35_scheduler_stress".to_owned(),
            parameters: serde_json::json!({
                "k_cells": [32, 64, 128, 255],
                "scope": "state and scheduler estimates only; no model forward",
            }),
            raw_argv_recorded: false,
        },
        RunEnvironment::capture(),
    )?
    .profile(native_profile_record(&profile_config, &profile_decision))
    .parity(report)
    .memory(serde_json::json!({
        "observed_process_peak_resident_bytes": observed,
        "admission_envelope_bytes": {
            "forward_scratch_bytes": envelope.forward_scratch_bytes,
            "allocator_headroom_bytes": envelope.allocator_headroom_bytes,
            "max_process_bytes": envelope.max_process_bytes,
        },
    }))
    .predictions(rows)
    .finish()?;
    eprintln!("evidence recorded: {}", directory.display());
    Ok(())
}
