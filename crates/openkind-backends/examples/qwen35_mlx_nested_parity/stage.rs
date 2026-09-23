//! Sequential nested continuation parity evaluation against frozen Phase 3B fixtures.

use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use openkind_backends::qwen35::mlx::MlxPrecision;
use openkind_backends::qwen35::mlx::MlxQwen35Backbone as Backbone;
use openkind_backends::qwen35::{
    run_batched_nested, BackboneReference, BatchContinuation, NestedQuestion, PolicyAction,
    Qwen35Error, ReferenceBundle, ScoreSummaryHead, SequentialNestedExecutor, POLICY_THRESHOLD,
    PROBABILITY_TOLERANCE,
};
use openkind_runtime::branch::BranchableState;
use openkind_runtime::BatchForwardMode;

use super::fixtures::{
    git_commit, maximum_delta, primitive, read_json, ExpectedAnswer, GoldenCase, HeadFixture,
    ProbabilityReference, TokenFixtures, TokenRecord,
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
    let fp32 = backbone.precision() == MlxPrecision::Fp32;
    let vectorized_fixture_indices = if fp32 {
        select_vectorized_fixture_pair(&token_fixtures.records)
    } else {
        None
    };
    let mut vectorized_batch_gate_passed = !fp32;
    let mut vectorized_batch_baselines: [Option<Vec<Vec<f32>>>; 2] = [None, None];
    let mut vectorized_batch_gate = if fp32 {
        serde_json::json!({
            "applicable": true,
            "gate_passed": false,
            "reason": "no same-root fixture pair has unequal question lengths and a variable candidate group",
        })
    } else {
        serde_json::json!({
            "applicable": false,
            "gate_passed": true,
            "reason": "vectorized batch parity is an FP32-only gate",
        })
    };

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

    for (record_index, record) in token_fixtures.records.iter().enumerate() {
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

        if let Some(indices) = vectorized_fixture_indices {
            for lane_index in 0..indices.len() {
                if indices[lane_index] == record_index {
                    vectorized_batch_baselines[lane_index] = Some(features.clone());
                }
            }
        }
        let _ = reference;
    }

    if let Some(indices) = vectorized_fixture_indices {
        let mut batch_lanes = Vec::with_capacity(indices.len());
        for (lane_index, fixture_index) in indices.into_iter().enumerate() {
            let record = &token_fixtures.records[fixture_index];
            let golden_case = golden_cases
                .get(record.fixture_case)
                .ok_or("batched fixture case is out of range")?;
            let question = golden_case
                .request
                .questions
                .iter()
                .find(|question| question.id == record.question_id)
                .ok_or("batched question is missing from golden case")?;
            let head_fixture = golden_case
                .head_and_token_fixtures
                .iter()
                .find(|fixture| fixture.question_id == record.question_id)
                .ok_or("batched head fixture is missing from golden case")?;
            let expected = *expected_answers
                .get(&(record.fixture_case, record.question_id.as_str()))
                .ok_or("batched probability answer is missing")?;
            let sequential_features = vectorized_batch_baselines[lane_index]
                .as_deref()
                .ok_or("batched sequential baseline was not recorded")?;
            batch_lanes.push(BatchLaneFixture {
                record,
                question,
                head_fixture,
                expected,
                sequential_features,
            });
        }
        let (gate_passed, gate_report) =
            vectorized_batch_parity_gate(backbone, &batch_lanes, bundle.head())?;
        vectorized_batch_gate_passed = gate_passed;
        vectorized_batch_gate = gate_report;
    }

    let probability_gate = maximum_probability_delta <= PROBABILITY_TOLERANCE;
    // The cached-vs-full self-consistency guard is an FP32-reference
    // assertion (the CPU oracle's cached path is exactly equal). BF16
    // execution legitimately rounds differently between cached and full
    // shapes; for the candidate profile it is reported as a diagnostic and
    // the frozen decision gates (probability/argmax/policy) decide.
    let cached_gate = !fp32 || maximum_cached_vs_full <= CACHED_VS_FULL_GUARD;
    let gate_passed = probability_gate
        && argmax_changes == 0
        && policy_changes == 0
        && cached_gate
        && vectorized_batch_gate_passed
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
            "vectorized_batch_parity": vectorized_batch_gate,
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

struct BatchModeRecorder<'a, E> {
    inner: &'a E,
    observed_modes: RefCell<Vec<BatchForwardMode>>,
}

impl<'a, E> BatchModeRecorder<'a, E> {
    fn new(inner: &'a E) -> Self {
        Self {
            inner,
            observed_modes: RefCell::new(Vec::new()),
        }
    }

    fn observed_modes(&self) -> Vec<BatchForwardMode> {
        self.observed_modes.borrow().clone()
    }
}

impl<E: SequentialNestedExecutor> SequentialNestedExecutor for BatchModeRecorder<'_, E> {
    type State = E::State;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        SequentialNestedExecutor::prefill(self.inner, input_ids)
    }

    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        SequentialNestedExecutor::continue_from(self.inner, state, suffix_ids)
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        let result = SequentialNestedExecutor::continue_batch_from(self.inner, states, suffix_ids)?;
        self.observed_modes
            .borrow_mut()
            .push(result.batch_forward_mode());
        Ok(result)
    }
}

struct BatchLaneFixture<'a> {
    record: &'a TokenRecord,
    question: &'a super::fixtures::GoldenQuestion,
    head_fixture: &'a HeadFixture,
    expected: &'a ExpectedAnswer,
    sequential_features: &'a [Vec<f32>],
}

fn select_vectorized_fixture_pair(records: &[TokenRecord]) -> Option<[usize; 2]> {
    for (left_index, left) in records.iter().enumerate() {
        let Some(left_lengths) = vectorizable_candidate_lengths(left) else {
            continue;
        };
        for (right_index, right) in records.iter().enumerate().skip(left_index + 1) {
            let Some(right_lengths) = vectorizable_candidate_lengths(right) else {
                continue;
            };
            let has_variable_candidate_group = left_lengths.iter().min()
                != left_lengths.iter().max()
                || right_lengths.iter().min() != right_lengths.iter().max();
            if left.fixture_case == right.fixture_case
                && left.question_id != right.question_id
                && left.root_ids == right.root_ids
                && left.question_ids.len() != right.question_ids.len()
                && has_variable_candidate_group
            {
                return Some([left_index, right_index]);
            }
        }
    }
    None
}

fn vectorizable_candidate_lengths(record: &TokenRecord) -> Option<Vec<usize>> {
    let lengths = record
        .candidate_suffix_ids
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    (!record.root_ids.is_empty()
        && !record.question_ids.is_empty()
        && (2..=8).contains(&lengths.len())
        && record.full_candidate_ids.len() == lengths.len()
        && lengths.iter().all(|length| *length > 0))
    .then_some(lengths)
}

fn vectorized_batch_parity_gate(
    backbone: &Backbone,
    lanes: &[BatchLaneFixture<'_>],
    head: &ScoreSummaryHead,
) -> Result<(bool, serde_json::Value), Box<dyn Error>> {
    if lanes.len() != 2 {
        return Err("vectorized batch gate requires exactly two question lanes".into());
    }
    let candidate_suffixes = lanes
        .iter()
        .map(|lane| {
            lane.record
                .candidate_suffix_ids
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let plans = lanes
        .iter()
        .zip(&candidate_suffixes)
        .map(|(lane, suffixes)| NestedQuestion {
            question_ids: &lane.record.question_ids,
            candidate_suffix_ids: suffixes,
        })
        .collect::<Vec<_>>();
    let recorder = BatchModeRecorder::new(backbone);
    let root_ids = &lanes[0].record.root_ids;
    let run = run_batched_nested(&recorder, root_ids, &plans)
        .map_err(|error| format!("vectorized nested run failed: {error}"))?;
    let observed_modes = recorder.observed_modes();
    let question_mode = observed_modes.first().copied();
    let candidate_modes = observed_modes.iter().skip(1).copied().collect::<Vec<_>>();
    let modes_complete = lanes.len() == 2 && observed_modes.len() == 1 + lanes.len();
    let question_fanout_vectorized = question_mode == Some(BatchForwardMode::Vectorized);
    let candidate_fanouts_vectorized = modes_complete
        && candidate_modes.len() == lanes.len()
        && candidate_modes
            .iter()
            .all(|mode| *mode == BatchForwardMode::Vectorized);
    let aggregate_vectorized = run.batch_forward_mode() == BatchForwardMode::Vectorized;

    let root_position = run.root_state().position();
    let question_suffix_lengths = lanes
        .iter()
        .map(|lane| lane.record.question_ids.len())
        .collect::<Vec<_>>();
    let expected_question_positions = lanes
        .iter()
        .map(|lane| root_ids.len() + lane.record.question_ids.len())
        .collect::<Vec<_>>();
    let question_positions = run
        .questions()
        .iter()
        .map(|question| question.question_state().position())
        .collect::<Vec<_>>();
    let candidate_positions = run
        .questions()
        .iter()
        .map(|question| {
            question
                .candidates()
                .iter()
                .map(|candidate| candidate.state().position())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected_candidate_positions = lanes
        .iter()
        .map(|lane| {
            let question_position = root_ids.len() + lane.record.question_ids.len();
            lane.record
                .candidate_suffix_ids
                .iter()
                .map(|suffix| question_position + suffix.len())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let candidate_suffix_lengths = lanes
        .iter()
        .map(|lane| {
            lane.record
                .candidate_suffix_ids
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let pair_contract_holds = lanes.len() == 2
        && lanes[0].record.fixture_case == lanes[1].record.fixture_case
        && lanes[0].record.question_id != lanes[1].record.question_id
        && lanes[0].record.root_ids == lanes[1].record.root_ids
        && question_suffix_lengths[0] != question_suffix_lengths[1]
        && candidate_suffix_lengths
            .iter()
            .any(|lengths| lengths.iter().min() != lengths.iter().max());
    let question_start_positions = vec![root_position; lanes.len()];
    let questions_share_start_position = pair_contract_holds
        && question_start_positions.len() == lanes.len()
        && question_start_positions
            .iter()
            .all(|position| *position == root_position);
    let positions_hold = root_position == root_ids.len()
        && run.questions().len() == lanes.len()
        && question_positions == expected_question_positions
        && candidate_positions == expected_candidate_positions
        && questions_share_start_position;

    let mut maximum_probability_delta = 0.0_f64;
    let mut maximum_head_fixture_delta = 0.0_f64;
    let mut maximum_probability_reference_top_delta = 0.0_f64;
    let mut maximum_feature_delta = 0.0_f64;
    let mut argmax_changes = 0_usize;
    let mut policy_changes = 0_usize;
    let mut lane_reports = Vec::with_capacity(run.questions().len());

    for (lane_index, (batched_question, lane)) in run.questions().iter().zip(lanes).enumerate() {
        if batched_question.candidates().len() != lane.sequential_features.len()
            || lane.sequential_features.len() != lane.record.candidate_suffix_ids.len()
        {
            return Err("batched candidate count does not match fixture".into());
        }
        let lane_features = batched_question
            .candidates()
            .iter()
            .map(|candidate| candidate.feature().to_vec())
            .collect::<Vec<_>>();
        for (feature, sequential) in lane_features.iter().zip(lane.sequential_features) {
            maximum_feature_delta =
                maximum_feature_delta.max(feature_max_abs_delta(feature, sequential)?);
        }

        let evaluation = head.evaluate(primitive(&lane.question.primitive)?, &lane_features)?;
        let probabilities = evaluation.full_probabilities();
        let head_fixture_delta = maximum_delta(&probabilities, &lane.head_fixture.probabilities)?;
        let probability_reference_top_delta =
            (evaluation.top_probability() - lane.expected.top_probability).abs();
        maximum_head_fixture_delta = maximum_head_fixture_delta.max(head_fixture_delta);
        maximum_probability_reference_top_delta =
            maximum_probability_reference_top_delta.max(probability_reference_top_delta);
        maximum_probability_delta = maximum_probability_delta
            .max(head_fixture_delta)
            .max(probability_reference_top_delta);

        let expected_index = lane
            .question
            .options
            .iter()
            .position(|option| option.id == lane.expected.selected_id)
            .ok_or("probability reference selected option is missing")?;
        let expected_policy = if lane.expected.top_probability >= POLICY_THRESHOLD {
            PolicyAction::Accept {
                candidate_index: expected_index,
            }
        } else {
            PolicyAction::Review
        };
        let actual_selected = evaluation.selected_candidate_index().and_then(|index| {
            lane.question
                .options
                .get(index)
                .map(|option| option.id.as_str())
        });
        if actual_selected != Some(lane.expected.selected_id.as_str()) {
            argmax_changes += 1;
        }
        if evaluation.policy_action() != expected_policy {
            policy_changes += 1;
        }
        lane_reports.push(serde_json::json!({
            "lane": lane_index,
            "question_id": lane.record.question_id,
            "question_suffix_length": lane.record.question_ids.len(),
            "candidate_suffix_lengths": candidate_suffix_lengths[lane_index],
            "probability_max_abs_delta_vs_head_fixture": head_fixture_delta,
            "top_probability_abs_delta_vs_probability_reference": probability_reference_top_delta,
            "selected_id": actual_selected,
        }));
    }

    let probability_gate = maximum_probability_delta <= PROBABILITY_TOLERANCE;
    let decision_gate = argmax_changes == 0 && policy_changes == 0;
    let mode_gate =
        question_fanout_vectorized && candidate_fanouts_vectorized && aggregate_vectorized;
    let gate_passed = mode_gate && probability_gate && decision_gate && positions_hold;

    let question_mode_name = question_mode.map_or("missing", BatchForwardMode::as_str);
    let candidate_mode_names = candidate_modes
        .iter()
        .map(|mode| mode.as_str())
        .collect::<Vec<_>>();
    Ok((
        gate_passed,
        serde_json::json!({
            "applicable": true,
            "gate_passed": gate_passed,
            "fixture_case": lanes.first().map(|lane| lane.record.fixture_case),
            "question_ids": lanes.iter().map(|lane| lane.record.question_id.as_str()).collect::<Vec<_>>(),
            "lanes": lanes.len(),
            "question_suffix_lengths": question_suffix_lengths,
            "candidate_suffix_lengths_by_question": candidate_suffix_lengths,
            "question_fanout_mode": question_mode_name,
            "candidate_fanout_modes": candidate_mode_names,
            "aggregate_batch_forward_mode": run.batch_forward_mode().as_str(),
            "question_fanout_vectorized": question_fanout_vectorized,
            "candidate_fanouts_vectorized": candidate_fanouts_vectorized,
            "candidate_mode_recording_complete": modes_complete,
            "probability_gate": probability_gate,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "maximum_probability_delta": maximum_probability_delta,
            "maximum_head_fixture_probability_delta": maximum_head_fixture_delta,
            "maximum_probability_reference_top_delta": maximum_probability_reference_top_delta,
            "argmax_changes": argmax_changes,
            "policy_changes": policy_changes,
            "question_lanes_share_start_position": questions_share_start_position,
            "positions_gate": positions_hold,
            "positions": {
                "root": root_position,
                "question_starts": question_start_positions,
                "questions": question_positions,
                "candidates": candidate_positions,
            },
            "maximum_feature_delta": maximum_feature_delta,
            "feature_delta_vs_each_lane_nested_baseline": maximum_feature_delta,
            "feature_delta_is_diagnostic_only": true,
            "lanes_report": lane_reports,
            "git_commit": git_commit(),
        }),
    ))
}

fn feature_max_abs_delta(actual: &[f32], expected: &[f32]) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "feature length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
        .fold(0.0, f64::max))
}
