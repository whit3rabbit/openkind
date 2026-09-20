//! Backend-neutral `BranchableState` implementation for the Qwen3.5 state.
//!
//! The complete-state contract binds attention KV, DeltaNet recurrent state,
//! convolution state, logical position, and the pinned profile identity.
//! Fork operations deep-copy every tensor family so a branch can never
//! mutate its source root through attention masks or partial cache reuse.

use std::mem::size_of_val;

use super::layer0::LayerState;
use super::model::BackboneState;
use opendecision_runtime::branch::{
    BranchBatch, BranchableState, ContentFingerprint, ContentFingerprintBuilder, ProfileId,
    SchedulingFingerprint, SchedulingFingerprintBuilder, StateError, StateIdentity,
    TensorStorageBreakdown,
};

const STRUCTURAL_DOMAIN: &str = "opendecision-qwen35-branch-state-v1";
const CONTENT_DOMAIN: &str = "opendecision-qwen35-branch-state-content-v1";

impl BranchableState for BackboneState {
    type Batch = Qwen35BranchBatch;

    fn profile_id(&self) -> &ProfileId {
        self.identity.profile()
    }

    fn position(&self) -> usize {
        self.position
    }

    fn tensor_storage_bytes(&self) -> usize {
        self.tensor_storage_breakdown().tensor_storage_bytes()
    }

    fn scheduling_fingerprint(&self) -> SchedulingFingerprint {
        let mut material = SchedulingFingerprint::builder(STRUCTURAL_DOMAIN);
        mix_scheduling_identity(&mut material, &self.identity);
        // Lineage participates through the process-local root only; fork
        // depth is excluded so a forked child stays scheduling-equivalent to
        // its parent until it advances.
        material
            .value(self.lineage.root_id())
            .value(self.position as u64);
        material.value(self.layers.len() as u64);
        for layer in &self.layers {
            match layer {
                LayerState::Linear { conv, recurrent } => {
                    material
                        .field(b"linear")
                        .value(conv.len() as u64)
                        .value(recurrent.len() as u64);
                }
                LayerState::Full { keys, values } => {
                    material
                        .field(b"full")
                        .value(keys.len() as u64)
                        .value(values.len() as u64);
                }
            }
        }
        material.finish()
    }

    fn fork_one(&self) -> Result<Self, StateError> {
        Ok(Self {
            identity: self.identity.clone(),
            lineage: self.lineage.forked(),
            position: self.position,
            layers: self.layers.clone(),
        })
    }

    fn fork_batch(&self, lanes: usize) -> Result<Self::Batch, StateError> {
        if lanes == 0 {
            return Err(StateError::InvalidLaneCount { requested: 0 });
        }
        let mut batch = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            batch.push(self.fork_one()?);
        }
        Ok(Qwen35BranchBatch { lanes: batch })
    }
}

/// Ordered set of independent Qwen3.5 continuation lanes forked from one root.
///
/// Every lane owns its own attention KV, DeltaNet recurrent, and convolution
/// tensors; advancing or mutating one lane leaves the remaining lanes and the
/// source root unchanged.
#[derive(Debug, Clone)]
pub struct Qwen35BranchBatch {
    lanes: Vec<BackboneState>,
}

impl BranchBatch for Qwen35BranchBatch {
    type State = BackboneState;

    fn lanes(&self) -> usize {
        self.lanes.len()
    }

    fn tensor_storage_bytes(&self) -> usize {
        self.lanes
            .iter()
            .map(BranchableState::tensor_storage_bytes)
            .sum()
    }

    fn select(&self, index: usize) -> Result<Self::State, StateError> {
        self.lanes
            .get(index)
            .cloned()
            .ok_or(StateError::LaneIndexOutOfBounds {
                lanes: self.lanes.len(),
                index,
            })
    }

    fn gather(&self, indices: &[usize]) -> Result<Self, StateError> {
        if indices.is_empty() {
            return Err(StateError::EmptyGather);
        }
        let mut lanes = Vec::with_capacity(indices.len());
        for &index in indices {
            lanes.push(self.select(index)?);
        }
        Ok(Self { lanes })
    }
}

impl BackboneState {
    /// Exact byte accounting over the three hybrid tensor families.
    ///
    /// [`TensorStorageBreakdown::tensor_storage_bytes`] equals
    /// [`BackboneState::byte_len`]
    /// and the exported Phase 3B `root_cache_bytes` contract.
    #[must_use]
    pub fn tensor_storage_breakdown(&self) -> TensorStorageBreakdown {
        let float_bytes = size_of_val(&0_f32);
        let mut breakdown = TensorStorageBreakdown {
            attention_kv_bytes: 0,
            recurrent_bytes: 0,
            convolution_bytes: 0,
        };
        for layer in &self.layers {
            match layer {
                LayerState::Linear { conv, recurrent } => {
                    breakdown.convolution_bytes += conv.len() * float_bytes;
                    breakdown.recurrent_bytes += recurrent.len() * float_bytes;
                }
                LayerState::Full { keys, values } => {
                    breakdown.attention_kv_bytes += (keys.len() + values.len()) * float_bytes;
                }
            }
        }
        breakdown
    }

    /// Strict content fingerprint over exact little-endian tensor bytes,
    /// identity, and position.
    ///
    /// Unlike the structural fingerprint, the strict fingerprint excludes
    /// lineage and is reproducible across processes: two states hash equally
    /// exactly when every tensor, the identity, and the position match. The
    /// cost is linear in state bytes, so it is a fixture/replay gate rather
    /// than a scheduling operation.
    #[must_use]
    pub fn strict_fingerprint(&self) -> ContentFingerprint {
        let mut material = ContentFingerprint::builder(CONTENT_DOMAIN);
        mix_content_identity(&mut material, &self.identity);
        material.value(self.position as u64);
        material.value(self.layers.len() as u64);
        for layer in &self.layers {
            match layer {
                LayerState::Linear { conv, recurrent } => {
                    material.field(b"linear");
                    mix_tensor(&mut material, conv);
                    mix_tensor(&mut material, recurrent);
                }
                LayerState::Full { keys, values } => {
                    material.field(b"full");
                    mix_tensor(&mut material, keys);
                    mix_tensor(&mut material, values);
                }
            }
        }
        material.finish()
    }
}

fn mix_scheduling_identity(material: &mut SchedulingFingerprintBuilder, identity: &StateIdentity) {
    mix_identity_fields(material, identity);
}

fn mix_content_identity(material: &mut ContentFingerprintBuilder, identity: &StateIdentity) {
    mix_identity_fields(material, identity);
}

trait IdentityFingerprintBuilder {
    fn field(&mut self, bytes: &[u8]) -> &mut Self;
}

impl IdentityFingerprintBuilder for SchedulingFingerprintBuilder {
    fn field(&mut self, bytes: &[u8]) -> &mut Self {
        SchedulingFingerprintBuilder::field(self, bytes)
    }
}

impl IdentityFingerprintBuilder for ContentFingerprintBuilder {
    fn field(&mut self, bytes: &[u8]) -> &mut Self {
        ContentFingerprintBuilder::field(self, bytes)
    }
}

fn mix_identity_fields<B: IdentityFingerprintBuilder>(material: &mut B, identity: &StateIdentity) {
    material
        .field(identity.profile().as_str().as_bytes())
        .field(identity.backbone_id().as_bytes())
        .field(identity.backbone_revision().as_bytes())
        .field(identity.renderer_id().as_bytes())
        .field(identity.tokenizer_digest().as_bytes())
        .field(identity.arithmetic_id().as_bytes());
}

fn mix_tensor(material: &mut ContentFingerprintBuilder, values: &[f32]) {
    material.value(values.len() as u64);
    let mut bytes = Vec::with_capacity(size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    material.field(&bytes);
}
