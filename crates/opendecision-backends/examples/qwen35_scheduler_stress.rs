//! Phase 3.9a scheduler/admission stress with no model weights or large state allocation.

use std::error::Error;

use opendecision_backends::qwen35::{
    choose_strategy, ProcessMemoryEnvelope, SchedulerConfig, StrategyRequest,
};
use opendecision_runtime::{peak_resident_bytes, BackendCapabilities};

fn main() -> Result<(), Box<dyn Error>> {
    let observed = peak_resident_bytes()?;
    let mut rows = Vec::new();
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
        .with_process_memory(ProcessMemoryEnvelope {
            observed_resident_bytes: observed,
            forward_scratch_bytes: 1024 * 1024 * 1024,
            allocator_headroom_bytes: 512 * 1024 * 1024,
            max_process_bytes: 12 * 1024 * 1024 * 1024,
        });
        let limited_vectorized = cpu.clone().with_backend_capabilities(
            BackendCapabilities::fully_vectorized().with_lane_limits(8, 128),
        );
        let cpu_decision = choose_strategy(&cpu, &request);
        let vectorized_decision = choose_strategy(&limited_vectorized, &request);
        rows.push(serde_json::json!({
            "questions": 3,
            "candidates_per_question": candidates,
            "cpu_plan": cpu_decision.strategy.as_str(),
            "cpu_admitted": cpu_decision.admitted,
            "limited_vectorized_plan": vectorized_decision.strategy.as_str(),
            "limited_vectorized_admitted": vectorized_decision.admitted,
            "repeated_full_tensor_bytes": cpu_decision.retention.repeated_full_tensor_bytes,
            "nested_sequential_tensor_bytes": cpu_decision.retention.nested_sequential_tensor_bytes,
            "nested_batched_tensor_bytes": cpu_decision.retention.nested_batched_tensor_bytes,
            "nested_sequential_process_peak_bytes": cpu_decision.retention.nested_sequential_process_peak_bytes,
            "nested_batched_process_peak_bytes": cpu_decision.retention.nested_batched_process_peak_bytes,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "opendecision-qwen35-scheduler-stress/v1",
            "scope": "state and scheduler estimates only; no model forward",
            "observed_process_peak_resident_bytes": observed,
            "rows": rows,
        }))?
    );
    Ok(())
}
