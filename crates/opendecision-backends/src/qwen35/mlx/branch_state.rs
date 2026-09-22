//! Backend-neutral `BranchableState` binding for the MLX Qwen3.5 state.
//!
//! MLX arrays are immutable, refcounted values: "advancing" a state always
//! produces *new* arrays, and [`Array::clone`](mlx_rs::Array) is the C API's
//! refcounted handle copy (`mlx_array_set`). Forking therefore isolates all
//! three tensor families (attention KV, DeltaNet recurrent, convolution)
//! structurally — a branch cannot mutate its source root because no in-place
//! mutation path exists — while remaining an O(lanes) handle copy instead of
//! a tensor-payload copy.
//!
//! `MlxBackboneState` additionally stores its [`SharedMlxRuntime`]: array
//! *evaluation* is only legal under the runtime's execution lock, which is
//! what makes the `Send + Sync` requirement (MLX arrays are `Send` but not
//! `Sync`) sound. Handle-only operations — clone, shape, byte accounting —
//! are safe concurrently because the underlying values are immutable and
//! the refcount updates are atomic.

use opendecision_runtime::branch::{
    BranchBatch, BranchableState, ContentFingerprint, ContentFingerprintBuilder, ProfileId,
    SchedulingFingerprint, SchedulingFingerprintBuilder, StateError, StateIdentity,
    TensorStorageBreakdown,
};

use super::layers::MlxLayerState;
use super::model::MlxBackboneState;

const STRUCTURAL_DOMAIN: &str = "opendecision-qwen35-mlx-branch-state-v1";
const CONTENT_DOMAIN: &str = "opendecision-qwen35-mlx-branch-state-content-v1";

// SAFETY: every array evaluation, handle clone, shape read, and byte
// accounting operation on this state runs under the process-wide
// `MlxRuntime` execution mutex (see the module docs). MLX arrays are `Send`
// but not `Sync`, so the lock is required even for these otherwise read-only
// operations.
unsafe impl Sync for MlxBackboneState {}

impl BranchableState for MlxBackboneState {
    type Batch = MlxBranchBatch;

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
        self.runtime
            .execute(|| {
                let mut material = SchedulingFingerprint::builder(STRUCTURAL_DOMAIN);
                mix_identity_fields(&mut material, &self.identity);
                // Lineage participates through the process-local root only;
                // fork depth is excluded so a forked child stays
                // scheduling-equivalent to its parent until it advances.
                material
                    .value(self.lineage.root_id())
                    .value(self.position as u64);
                material.value(self.layers.len() as u64);
                for layer in &self.layers {
                    match layer {
                        MlxLayerState::Linear(state) => {
                            material
                                .field(b"linear")
                                .value(state.conv.size() as u64)
                                .value(state.recurrent.size() as u64);
                        }
                        MlxLayerState::Full(state) => {
                            material
                                .field(b"full")
                                .value(state.keys.size() as u64)
                                .value(state.values.size() as u64);
                        }
                    }
                }
                material.finish()
            })
            .unwrap_or_else(|error| panic!("failed to fingerprint MLX scheduling state: {error}"))
    }

    fn fork_one(&self) -> Result<Self, StateError> {
        self.runtime
            .execute(|| Self {
                runtime: super::runtime::SharedMlxRuntime::clone(&self.runtime),
                identity: self.identity.clone(),
                lineage: self.lineage.forked(),
                position: self.position,
                layers: self.layers.clone(),
            })
            .map_err(|error| StateError::IdentityMismatch {
                expected: "mlx execution".to_owned(),
                actual: error.to_string(),
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
        Ok(MlxBranchBatch { lanes: batch })
    }
}

/// Ordered set of independent MLX continuation lanes forked from one root.
///
/// Every lane holds its own refcounted immutable arrays; advancing one lane
/// constructs new arrays and leaves the remaining lanes and the source root
/// unchanged.
pub struct MlxBranchBatch {
    lanes: Vec<MlxBackboneState>,
}

impl BranchBatch for MlxBranchBatch {
    type State = MlxBackboneState;

    fn lanes(&self) -> usize {
        self.lanes.len()
    }

    fn tensor_storage_bytes(&self) -> usize {
        self.lanes
            .iter()
            .map(BranchableState::tensor_storage_bytes)
            .fold(0_usize, usize::saturating_add)
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

impl MlxBackboneState {
    /// Exact logical byte accounting over the three hybrid tensor families.
    ///
    /// This is the same quantity the runtime's `TensorStorageBreakdown`
    /// tracks; it is *not* the MLX allocator's active/peak/cache memory,
    /// which the runtime reports separately.
    #[must_use]
    pub fn tensor_storage_breakdown(&self) -> TensorStorageBreakdown {
        self.runtime
            .execute(|| {
                let mut breakdown = TensorStorageBreakdown {
                    attention_kv_bytes: 0,
                    recurrent_bytes: 0,
                    convolution_bytes: 0,
                };
                for layer in &self.layers {
                    match layer {
                        MlxLayerState::Linear(state) => {
                            breakdown.convolution_bytes = breakdown
                                .convolution_bytes
                                .saturating_add(state.conv.nbytes());
                            breakdown.recurrent_bytes = breakdown
                                .recurrent_bytes
                                .saturating_add(state.recurrent.nbytes());
                        }
                        MlxLayerState::Full(state) => {
                            breakdown.attention_kv_bytes =
                                breakdown.attention_kv_bytes.saturating_add(
                                    state.keys.nbytes().saturating_add(state.values.nbytes()),
                                );
                        }
                    }
                }
                breakdown
            })
            .unwrap_or_else(|error| panic!("failed to account for MLX state storage: {error}"))
    }

    /// Strict content fingerprint over exact little-endian tensor bytes,
    /// identity, and position.
    ///
    /// Reproducible across processes: two states hash equally exactly when
    /// every tensor value, the identity, and the position match. Evaluation
    /// and host reads run under the state's runtime execution lock; the cost
    /// is linear in state bytes, so this is a fixture/replay gate rather
    /// than a scheduling operation.
    pub fn strict_fingerprint(&self) -> Result<ContentFingerprint, StateError> {
        self.runtime
            .execute(|| -> Result<ContentFingerprint, StateError> {
                let mut material = ContentFingerprint::builder(CONTENT_DOMAIN);
                mix_identity_fields(&mut material, &self.identity);
                material.value(self.position as u64);
                material.value(self.layers.len() as u64);
                for layer in &self.layers {
                    match layer {
                        MlxLayerState::Linear(state) => {
                            material.field(b"linear");
                            mix_array(&mut material, &state.conv)?;
                            mix_array(&mut material, &state.recurrent)?;
                        }
                        MlxLayerState::Full(state) => {
                            material.field(b"full");
                            mix_array(&mut material, &state.keys)?;
                            mix_array(&mut material, &state.values)?;
                        }
                    }
                }
                Ok(material.finish())
            })
            .map_err(|error| StateError::IdentityMismatch {
                expected: "mlx execution".to_owned(),
                actual: error.to_string(),
            })?
    }
}

fn mix_array(
    material: &mut ContentFingerprintBuilder,
    array: &mlx_rs::Array,
) -> Result<(), StateError> {
    array.eval().map_err(|error| StateError::IdentityMismatch {
        expected: "mlx eval".to_owned(),
        actual: error.to_string(),
    })?;
    let values = array
        .to_vec_cast::<f32>()
        .map_err(|error| StateError::IdentityMismatch {
            expected: "mlx read".to_owned(),
            actual: error.to_string(),
        })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(StateError::IdentityMismatch {
            expected: "finite MLX state tensors".to_owned(),
            actual: "state fingerprint input contains a non-finite value".to_owned(),
        });
    }
    material.value(values.len() as u64);
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    material.field(&bytes);
    Ok(())
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
