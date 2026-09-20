//! Backend-neutral `BranchableState` implementation for the Qwen3.5 state.
//!
//! The complete-state contract binds attention KV, DeltaNet recurrent state,
//! convolution state, logical position, and the pinned profile identity.
//! Fork operations deep-copy every tensor family so a branch can never
//! mutate its source root through attention masks or partial cache reuse.

use std::mem::{size_of, size_of_val};

use super::layer0::LayerState;
use super::model::BackboneState;
use crate::branch::{
    BranchBatch, BranchableState, FingerprintMaterial, ProfileId, StateError, StateFingerprint,
    StateIdentity, StateLineage, StorageBreakdown,
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

    fn storage_bytes(&self) -> usize {
        self.storage_breakdown().tensor_bytes()
    }

    fn fingerprint(&self) -> StateFingerprint {
        let mut material = FingerprintMaterial::new(STRUCTURAL_DOMAIN);
        mix_identity(&mut material, &self.identity);
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

    fn storage_bytes(&self) -> usize {
        self.lanes.iter().map(BranchableState::storage_bytes).sum()
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
    /// Exact byte accounting over the three hybrid tensor families plus the
    /// fixed-size metadata bookkeeping.
    ///
    /// [`StorageBreakdown::tensor_bytes`] equals [`BackboneState::byte_len`]
    /// and the exported Phase 3B `root_cache_bytes` contract.
    #[must_use]
    pub fn storage_breakdown(&self) -> StorageBreakdown {
        let float_bytes = size_of::<f32>();
        let mut breakdown = StorageBreakdown {
            attention_kv_bytes: 0,
            recurrent_bytes: 0,
            convolution_bytes: 0,
            metadata_bytes: 0,
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
        breakdown.metadata_bytes = metadata_bytes();
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
    pub fn strict_fingerprint(&self) -> StateFingerprint {
        let mut material = FingerprintMaterial::new(CONTENT_DOMAIN);
        mix_identity(&mut material, &self.identity);
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

/// Fixed-size identity, lineage, and position bookkeeping bytes.
fn metadata_bytes() -> usize {
    size_of::<StateIdentity>() + size_of::<StateLineage>() + size_of::<usize>()
}

fn mix_identity(material: &mut FingerprintMaterial, identity: &StateIdentity) {
    material
        .field(identity.profile().as_str().as_bytes())
        .field(identity.backbone_id().as_bytes())
        .field(identity.backbone_revision().as_bytes())
        .field(identity.renderer_id().as_bytes())
        .field(identity.tokenizer_digest().as_bytes())
        .field(identity.arithmetic_id().as_bytes());
}

fn mix_tensor(material: &mut FingerprintMaterial, values: &[f32]) {
    material.value(values.len() as u64);
    let mut bytes = Vec::with_capacity(size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    material.field(&bytes);
}
