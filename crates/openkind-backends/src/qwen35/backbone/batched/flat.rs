//! Diagnostic shared-prefix execution with complete question-candidate suffixes.
//!
//! Each candidate lane starts at the same immutable root, as in the Python
//! field-batching pattern. The normal nested runner retains question states
//! and avoids repeating question tokens for every candidate.

use crate::qwen35::Qwen35Error;
use openkind_runtime::branch::{BranchBatch, BranchableState};
use openkind_runtime::BatchForwardMode;

use crate::qwen35::backbone::nested::{NestedQuestion, SequentialNestedExecutor};

/// Features restored to question and candidate order after flat execution.
pub struct FlatBatchRun {
    candidate_features: Vec<Vec<Vec<f32>>>,
    batch_forward_mode: BatchForwardMode,
    peak_batch_bytes: usize,
}

impl FlatBatchRun {
    /// Per-question candidate features in the input order.
    #[must_use]
    pub fn candidate_features(&self) -> &[Vec<Vec<f32>>] {
        &self.candidate_features
    }

    /// Physical forward mode across all flat suffix batches.
    #[must_use]
    pub const fn batch_forward_mode(&self) -> BatchForwardMode {
        self.batch_forward_mode
    }

    /// Largest simultaneous root fan-out tensor payload.
    #[must_use]
    pub const fn peak_batch_bytes(&self) -> usize {
        self.peak_batch_bytes
    }
}

/// Prefill once, then batch complete question-candidate suffixes from the root.
///
/// This is an opt-in diagnostic path. It repeats question tokens per candidate
/// so the final feature and readout stay equivalent to nested execution.
///
/// # Errors
/// Returns an error for invalid plans, state drift, branch mutation, or an
/// executor failure.
pub fn run_flat_batched_candidates<E: SequentialNestedExecutor>(
    executor: &E,
    root_ids: &[u32],
    plans: &[NestedQuestion<'_>],
    max_lanes: usize,
) -> Result<FlatBatchRun, Qwen35Error> {
    if !(2..=8).contains(&max_lanes) {
        return Err(Qwen35Error::InvalidInput(
            "flat batching requires 2..=8 lanes".into(),
        ));
    }
    if root_ids.is_empty() || plans.is_empty() {
        return Err(Qwen35Error::InvalidInput(
            "flat batching requires a root and questions".into(),
        ));
    }

    let mut suffixes = Vec::new();
    for (q, plan) in plans.iter().enumerate() {
        if plan.question_ids.is_empty() || plan.candidate_suffix_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(format!(
                "flat question {q} requires a suffix and candidates"
            )));
        }
        for (k, candidate_ids) in plan.candidate_suffix_ids.iter().enumerate() {
            if candidate_ids.is_empty() {
                return Err(Qwen35Error::InvalidInput(format!(
                    "flat candidate {q}:{k} has an empty suffix"
                )));
            }
            let mut ids = Vec::with_capacity(plan.question_ids.len() + candidate_ids.len());
            ids.extend_from_slice(plan.question_ids);
            ids.extend_from_slice(candidate_ids);
            suffixes.push((q, k, ids));
        }
    }

    let (_, root) = executor.prefill(root_ids)?;
    let root_fingerprint = root.scheduling_fingerprint();
    let mut outputs = plans
        .iter()
        .map(|plan| vec![None; plan.candidate_suffix_ids.len()])
        .collect::<Vec<Vec<Option<Vec<f32>>>>>();
    let mut mode = BatchForwardMode::Vectorized;
    let mut peak_batch_bytes = 0;

    for group in suffixes.chunks(max_lanes) {
        let batch = root.fork_batch(group.len())?;
        if batch.lanes() != group.len() {
            return Err(Qwen35Error::InvalidInput(format!(
                "flat fan-out produced {} lanes, expected {}",
                batch.lanes(),
                group.len()
            )));
        }
        peak_batch_bytes = peak_batch_bytes.max(batch.tensor_storage_bytes());
        let lanes = (0..group.len())
            .map(|lane| batch.select(lane))
            .collect::<Result<Vec<_>, _>>()?;
        let fingerprints = lanes
            .iter()
            .map(BranchableState::scheduling_fingerprint)
            .collect::<Vec<_>>();
        let lane_refs = lanes.iter().collect::<Vec<_>>();
        let ids = group
            .iter()
            .map(|(_, _, ids)| ids.as_slice())
            .collect::<Vec<_>>();
        let continuation = executor.continue_batch_from(&lane_refs, &ids)?;
        if continuation.batch_forward_mode() == BatchForwardMode::PerLane {
            mode = BatchForwardMode::PerLane;
        }
        let advanced = continuation.into_lanes();
        if advanced.len() != group.len() {
            return Err(Qwen35Error::InvalidInput(format!(
                "flat continuation returned {} lanes, expected {}",
                advanced.len(),
                group.len()
            )));
        }
        for (lane, ((q, k, ids), (feature, state))) in group.iter().zip(advanced).enumerate() {
            let expected = root.position().checked_add(ids.len()).ok_or_else(|| {
                Qwen35Error::InvalidInput("flat continuation position overflow".into())
            })?;
            if state.position() != expected {
                return Err(Qwen35Error::InvalidInput(format!(
                    "flat candidate {q}:{k} advanced to position {}, expected {expected}",
                    state.position()
                )));
            }
            outputs[*q][*k] = Some(feature);
            if lanes[lane].scheduling_fingerprint() != fingerprints[lane]
                || batch.select(lane)?.scheduling_fingerprint() != fingerprints[lane]
            {
                return Err(Qwen35Error::InvalidInput(format!(
                    "flat source lane {lane} changed during continuation"
                )));
            }
        }
        if root.scheduling_fingerprint() != root_fingerprint {
            return Err(Qwen35Error::InvalidInput(
                "shared root changed during flat continuation".into(),
            ));
        }
    }

    let candidate_features = outputs
        .into_iter()
        .enumerate()
        .map(|(q, slots)| {
            slots
                .into_iter()
                .enumerate()
                .map(|(k, slot)| {
                    slot.ok_or_else(|| {
                        Qwen35Error::InvalidInput(format!(
                            "flat candidate {q}:{k} produced no result"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FlatBatchRun {
        candidate_features,
        batch_forward_mode: mode,
        peak_batch_bytes,
    })
}
