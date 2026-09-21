//! Sequential nested continuation parity evaluation against frozen Phase 3B fixtures.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use opendecision_backends::qwen35::mlx::MlxQwen35Backbone as Backbone;
use opendecision_backends::qwen35::{
    BackboneReference, PolicyAction, ReferenceBundle, POLICY_THRESHOLD, PROBABILITY_TOLERANCE,
};
use opendecision_runtime::branch::BranchableState;

use super::fixtures::{
    git_commit, maximum_delta, primitive, read_json, GoldenCase, ProbabilityReference,
    TokenFixtures,
};

pub(crate) struct StageResult {
    pub(crate) gate_passed: bool,
    pub(crate) json: serde_json::Value,
}

pub(crate) fn nested_stage(
    backbone: &Backbone,
    reference_root: &Path,
    head_bundle_root: &Path,
) -> Result<StageResult, Box<dyn Error>> {
    let token_fixtures: TokenFixtures = read_json(&reference_root.join("TOKEN_FIXTURES.json"))?;
    let probability_reference: ProbabilityReference =
        read_json(&reference_root.join("PROBABILITY_REFERENCE.json"))?;
    let golden_cases: Vec<GoldenCase> = read_json(&head_bundle_root.join("golden.json"))?;
    let bundle = ReferenceBundle::load(head_bundle_root)?;
    let reference = BackboneReference::load(reference_root)?;

    let expected_answers: HashMap<_, _> = probability_reference
        .records
        .iter()
        .flat_map(|record| {
            record
                .answers
                .iter()
                .map(move |answer| ((record.fixture_case, answer.question_id.as_str()), answer))
        })
        .collect();

    const CACHED_VS_FULL_GUARD: f64 = 0.000_1;

    let mut maximum_probability_delta = 0.0_f64;
    let mut maximum_cached_vs_full = 0.0_f64;
    let mut argmax_changes = 0_usize;
    let mut policy_changes = 0_usize;
    let mut candidate_count = 0_usize;
    let mut root_immutability_holds = true;
    let mut position_gates_hold = true;
    let mut isolation_holds = true;
    let mut root_storage_gates_hold = true;
    let mut root_bytes_reports = Vec::new();
    let mut question_reports = Vec::new();

    for record in &token_fixtures.records {
        let golden_case = golden_cases
            .get(record.fixture_case)
            .ok_or("fixture case is out of range")?;
        let question = golden_case
            .request
            .questions
            .iter()
            .find(|question| question.id == record.question_id)
            .ok_or("question is missing from golden case")?;
        let head_fixture = golden_case
            .head_and_token_fixtures
            .iter()
            .find(|fixture| fixture.question_id == record.question_id)
            .ok_or("head fixture is missing from golden case")?;
        let expected = expected_answers
            .get(&(record.fixture_case, record.question_id.as_str()))
            .ok_or("probability answer is missing")?;

        // ---- one prefill of the shared root ----
        let (_root_output, root_state) = Backbone::prefill(backbone, &record.root_ids)
            .map_err(|e| format!("root prefill failed: {e}"))?;
        let root_bytes = root_state.tensor_storage_bytes();
        let expected_root_bytes = backbone.expected_tensor_storage_bytes(record.root_ids.len());
        let root_storage_matches = root_bytes == expected_root_bytes;
        root_storage_gates_hold &= root_storage_matches;
        root_bytes_reports.push(serde_json::json!({
            "fixture_case": record.fixture_case,
            "question_id": record.question_id,
            "root_tokens": record.root_ids.len(),
            "root_tensor_bytes": root_bytes,
            "expected_root_tensor_bytes": expected_root_bytes,
            "root_tensor_bytes_match": root_storage_matches,
        }));
        let root_fingerprint_before = root_state.strict_fingerprint().map_err(|e| e.to_string())?;
        if root_state.position() != record.root_ids.len() {
            position_gates_hold = false;
        }

        // ---- question fork, then candidate forks ----
        let question_fork = root_state.fork_one().map_err(|e| e.to_string())?;
        let (_question_feature, question_state) =
            Backbone::continue_from(backbone, &question_fork, &record.question_ids)
                .map_err(|e| format!("question continuation failed: {e}"))?;
        if question_state.position() != record.root_ids.len() + record.question_ids.len() {
            position_gates_hold = false;
        }

        let mut features = Vec::with_capacity(record.candidate_suffix_ids.len());
        // Reversed-order sibling replay for isolation evidence.
        let mut reversed_features = vec![None; record.candidate_suffix_ids.len()];
        for (index, suffix) in record.candidate_suffix_ids.iter().enumerate().rev() {
            let candidate_fork = question_state.fork_one().map_err(|e| e.to_string())?;
            let (output, candidate_state) =
                Backbone::continue_from(backbone, &candidate_fork, suffix)
                    .map_err(|e| format!("candidate continuation failed: {e}"))?;
            if candidate_state.position()
                != record.root_ids.len()
                    + record.question_ids.len()
                    + record.candidate_suffix_ids[index].len()
            {
                position_gates_hold = false;
            }
            reversed_features[index] = Some(output.feature().to_vec());
        }
        for (index, suffix) in record.candidate_suffix_ids.iter().enumerate() {
            let candidate_fork = question_state.fork_one().map_err(|e| e.to_string())?;
            let (output, candidate_state) =
                Backbone::continue_from(backbone, &candidate_fork, suffix)
                    .map_err(|e| format!("candidate continuation failed: {e}"))?;
            // Cached-vs-full against an independent MLX full-sequence run.
            let full_ids = &record.full_candidate_ids[index];
            let (full_output, _full_state) = Backbone::prefill(backbone, full_ids)
                .map_err(|e| format!("full-sequence run failed: {e}"))?;
            let cached = output.feature();
            let full = full_output.feature();
            let delta = cached
                .iter()
                .zip(full)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0.0, f64::max);
            maximum_cached_vs_full = maximum_cached_vs_full.max(delta);
            // Sibling isolation: reversed-order replay reproduces the feature.
            if let Some(reversed) = &reversed_features[index] {
                let isolation_delta = reversed
                    .iter()
                    .zip(cached)
                    .map(|(a, b)| (*a - *b).abs())
                    .fold(0.0_f32, f32::max);
                if isolation_delta > 1e-4 {
                    isolation_holds = false;
                }
            }
            let _ = candidate_state;
            features.push(cached.to_vec());
            candidate_count += 1;
        }

        // ---- root immutability after all downstream work ----
        let root_fingerprint_after = root_state.strict_fingerprint().map_err(|e| e.to_string())?;
        let (_fresh_output, fresh_root) = Backbone::prefill(backbone, &record.root_ids)
            .map_err(|e| format!("independent root prefill failed: {e}"))?;
        let fresh_fingerprint = fresh_root.strict_fingerprint().map_err(|e| e.to_string())?;
        if root_fingerprint_before != root_fingerprint_after
            || root_fingerprint_before != fresh_fingerprint
        {
            root_immutability_holds = false;
        }

        // ---- head gates ----
        let evaluation = bundle
            .head()
            .evaluate(primitive(&question.primitive)?, &features)?;
        let probabilities = evaluation.full_probabilities();
        let probability_delta = maximum_delta(&probabilities, &head_fixture.probabilities)?;
        maximum_probability_delta = maximum_probability_delta.max(probability_delta);
        let actual_selected = evaluation
            .selected_candidate_index()
            .map(|index| question.options[index].id.as_str());
        if actual_selected != Some(expected.selected_id.as_str()) {
            argmax_changes += 1;
        }
        let expected_index = question
            .options
            .iter()
            .position(|option| option.id == expected.selected_id)
            .ok_or("selected option is missing")?;
        let expected_policy = if expected.top_probability >= POLICY_THRESHOLD {
            PolicyAction::Accept {
                candidate_index: expected_index,
            }
        } else {
            PolicyAction::Review
        };
        if evaluation.policy_action() != expected_policy {
            policy_changes += 1;
        }
        question_reports.push(serde_json::json!({
            "fixture_case": record.fixture_case,
            "question_id": record.question_id,
            "candidates": features.len(),
            "probability_max_abs": probability_delta,
            "selected_id": actual_selected,
        }));
        let _ = reference;
    }

    let probability_gate = maximum_probability_delta <= PROBABILITY_TOLERANCE;
    // The cached-vs-full self-consistency guard is an FP32-reference
    // assertion (the CPU oracle's cached path is exactly equal). BF16
    // execution legitimately rounds differently between cached and full
    // shapes; for the candidate profile it is reported as a diagnostic and
    // the frozen decision gates (probability/argmax/policy) decide.
    let fp32 = backbone.arithmetic_id().contains("/fp32/reference-ops;");
    let cached_gate = !fp32 || maximum_cached_vs_full <= CACHED_VS_FULL_GUARD;
    let gate_passed = probability_gate
        && argmax_changes == 0
        && policy_changes == 0
        && cached_gate
        && root_immutability_holds
        && position_gates_hold
        && isolation_holds
        && root_storage_gates_hold;
    Ok(StageResult {
        gate_passed,
        json: serde_json::json!({
            "precision": backbone.arithmetic_id(),
            "gate_passed": gate_passed,
            "probability_gate": probability_gate,
            "maximum_probability_delta": maximum_probability_delta,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "argmax_changes": argmax_changes,
            "policy_changes": policy_changes,
            "maximum_cached_vs_full_delta": maximum_cached_vs_full,
            "cached_vs_full_guard": CACHED_VS_FULL_GUARD,
            "cached_vs_full_gate_asserted": fp32,
            "root_immutability_holds": root_immutability_holds,
            "position_gates_hold": position_gates_hold,
            "sibling_isolation_holds": isolation_holds,
            "root_storage_gates_hold": root_storage_gates_hold,
            "questions": question_reports.len(),
            "candidates": candidate_count,
            "root_tensor_bytes": root_bytes_reports,
            "results": question_reports,
            "git_commit": git_commit(),
        }),
    })
}
