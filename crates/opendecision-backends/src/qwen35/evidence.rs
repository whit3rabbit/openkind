//! Qwen3.5 mapping onto the backend-neutral native-run evidence schema.
//!
//! Everything profile-, backend-, or execution-specific lives here; the
//! schema and writer in `opendecision_runtime::evidence` stay backend-neutral
//! so a future accelerated backend produces comparable artifacts from the
//! same schema.

use opendecision_runtime::evidence::{BackendRecord, ExecutionRecord, ProfileRecord};
use serde_json::json;

use super::{
    ProcessMemoryEnvelope, SchedulerConfig, StrategyDecision, BACKBONE_ID, BACKBONE_REVISION,
    BUNDLE_SHA256, CALIBRATION_TEMPERATURE, DECLARED_PROBABILITY_SPACE, EXECUTION_ARITHMETIC_ID,
    POLICY_THRESHOLD, PROFILE_ID, STATE_FIRST_RENDERER_ID, TOKENIZER_JSON_SHA256,
};

/// Backend implementation identity recorded in evidence artifacts.
///
/// `candle-core` is pinned to `=0.8.0` in the workspace manifest; this string
/// must be updated together with that pin. An accelerated backend records its
/// own implementation commit here (for example an MLX-LM revision).
pub const BACKEND_IMPLEMENTATION: &str = "candle-core 0.8.0";

/// Map the pinned profile, this backend, and one scheduler decision onto the
/// evidence `PROFILE.json` record.
#[must_use]
pub fn native_profile_record(
    scheduler: &SchedulerConfig,
    decision: &StrategyDecision,
) -> ProfileRecord {
    ProfileRecord {
        profile_id: PROFILE_ID.to_owned(),
        bundle_sha256: BUNDLE_SHA256.to_owned(),
        backbone_id: BACKBONE_ID.to_owned(),
        backbone_revision: BACKBONE_REVISION.to_owned(),
        renderer_id: STATE_FIRST_RENDERER_ID.to_owned(),
        tokenizer_digest: TOKENIZER_JSON_SHA256.to_owned(),
        arithmetic_id: EXECUTION_ARITHMETIC_ID.to_owned(),
        calibration_temperature: CALIBRATION_TEMPERATURE,
        policy_threshold: POLICY_THRESHOLD,
        probability_space: DECLARED_PROBABILITY_SPACE.as_str().to_owned(),
        backend: BackendRecord {
            backend_id: "qwen35-native-cpu".to_owned(),
            backend_version: BACKEND_IMPLEMENTATION.to_owned(),
            allocator: "default system allocator".to_owned(),
            capabilities: json!({
                "vectorized_question_forward":
                    scheduler.backend_capabilities.supports_vectorized_question_forward(),
                "vectorized_candidate_forward":
                    scheduler.backend_capabilities.supports_vectorized_candidate_forward(),
                "max_vectorized_question_lanes":
                    scheduler.backend_capabilities.max_vectorized_question_lanes(),
                "max_vectorized_candidate_lanes":
                    scheduler.backend_capabilities.max_vectorized_candidate_lanes(),
            }),
        },
        execution: ExecutionRecord {
            execution_plan: decision.strategy.as_str().to_owned(),
            batch_forward_mode: decision.batch_forward_mode.as_str().to_owned(),
            forced: decision.forced,
            scheduler: scheduler_summary(scheduler),
        },
    }
}

fn scheduler_summary(scheduler: &SchedulerConfig) -> serde_json::Value {
    json!({
        "min_shared_savings_ratio": scheduler.min_shared_savings_ratio,
        "max_tensor_storage_bytes": scheduler.max_tensor_storage_bytes,
        "state_fixed_tensor_bytes": scheduler.state_fixed_tensor_bytes,
        "state_tensor_bytes_per_token": scheduler.state_tensor_bytes_per_token,
        "forced_strategy": scheduler.forced_strategy.map(|plan| plan.as_str()),
        "process_memory": scheduler.process_memory.map(process_memory_summary),
    })
}

fn process_memory_summary(envelope: ProcessMemoryEnvelope) -> serde_json::Value {
    json!({
        "observed_resident_bytes": envelope.observed_resident_bytes,
        "forward_scratch_bytes": envelope.forward_scratch_bytes,
        "allocator_headroom_bytes": envelope.allocator_headroom_bytes,
        "max_process_bytes": envelope.max_process_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::{RetentionEstimates, StrategyEstimates};
    use opendecision_runtime::{BackendCapabilities, BatchForwardMode, ExecutionPlan};
    use serde_json::Value;

    #[test]
    fn profile_record_carries_plan_and_physical_mode_as_siblings() {
        let scheduler = SchedulerConfig::for_pinned_profile(2.52, None)
            .with_backend_capabilities(BackendCapabilities::per_lane());
        let decision = StrategyDecision {
            strategy: ExecutionPlan::NestedBatched,
            batch_forward_mode: BatchForwardMode::PerLane,
            forced: true,
            rationale: "forced override".to_owned(),
            estimates: StrategyEstimates {
                repeated_tokens: 10,
                shared_tokens: 4,
                savings_ratio: 2.5,
            },
            retention: RetentionEstimates {
                repeated_full_tensor_bytes: 1,
                nested_sequential_tensor_bytes: 2,
                nested_batched_tensor_bytes: 3,
                repeated_full_process_peak_bytes: None,
                nested_sequential_process_peak_bytes: None,
                nested_batched_process_peak_bytes: None,
            },
            admitted: true,
        };

        let record = native_profile_record(&scheduler, &decision);
        assert_eq!(record.profile_id, PROFILE_ID);
        assert_eq!(
            record.probability_space,
            "offered_options_plus_semantic_none"
        );
        assert_eq!(record.backend.backend_version, BACKEND_IMPLEMENTATION);
        assert_eq!(record.execution.execution_plan, "nested_batched");
        assert_eq!(record.execution.batch_forward_mode, "per_lane");
        assert!(record.execution.forced);
        let scheduler_json = record.execution.scheduler;
        assert_eq!(
            scheduler_json
                .get("forced_strategy")
                .and_then(Value::as_str),
            None
        );
        assert_eq!(
            scheduler_json
                .get("min_shared_savings_ratio")
                .and_then(Value::as_f64),
            Some(2.52)
        );
    }
}
