use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use super::super::{ExecutionControl, Qwen35Error};
use super::embedding::{verify_decoder_shard, Qwen35Embedding};
use super::layer0::{rms_norm_zero_centered, DecoderLayer, LayerState, PINNED};
use openkind_runtime::branch::{StateError, StateIdentity, StateLineage};

const HIDDEN_SIZE: usize = 2_560;
const LAYER_COUNT: usize = 32;

/// Native FP32 full-sequence result for the pinned Qwen3.5 backbone.
#[derive(Debug, Clone, PartialEq)]
pub struct BackboneOutput {
    token_count: usize,
    embedding_last_token: Vec<f32>,
    layer_last_tokens: Vec<Vec<f32>>,
    final_values: Vec<f32>,
}

impl BackboneOutput {
    /// Number of positions in the evaluated sequence.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    /// Final-token embedding before decoder execution.
    #[must_use]
    pub fn embedding_last_token(&self) -> &[f32] {
        &self.embedding_last_token
    }

    /// Final-token hidden vector after a decoder layer.
    #[must_use]
    pub fn layer_last_token(&self, layer_index: usize) -> Option<&[f32]> {
        self.layer_last_tokens.get(layer_index).map(Vec::as_slice)
    }

    /// Row-major `[token_count, 2560]` output after final RMSNorm.
    #[must_use]
    pub fn final_values(&self) -> &[f32] {
        &self.final_values
    }

    /// Final-token candidate feature after final RMSNorm.
    #[must_use]
    pub fn final_token(&self) -> &[f32] {
        let start = (self.token_count - 1) * HIDDEN_SIZE;
        &self.final_values[start..]
    }
}

/// Verified native FP32 executor for the frozen 32-layer Qwen3.5 backbone.
#[derive(Debug)]
pub struct Qwen35Backbone {
    embedding: Qwen35Embedding,
    decoder_shard: PathBuf,
    identity: StateIdentity,
    device: Device,
}

/// Arithmetic identity of the candle backbone on a CUDA device.
///
/// CUDA execution is a separate identity from the pinned CPU oracle so CPU
/// and CUDA continuation states can never be mixed (mirroring the MLX
/// precision/kernel identity discipline).
pub(crate) fn candle_cuda_arithmetic_id() -> String {
    "candle-cuda-fp32".to_owned()
}

/// Complete Qwen3.5 continuation state after a native prefix evaluation.
///
/// The state carries the pinned profile/model/tokenizer/renderer/arithmetic
/// identity, branch lineage, logical position, and every mutable tensor a
/// continuation reads: attention KV, DeltaNet recurrent, and convolution
/// state. Every clone owns independent tensors, so continuing a branch cannot
/// mutate its source root.
#[derive(Debug, Clone, PartialEq)]
pub struct BackboneState {
    pub(super) identity: StateIdentity,
    pub(super) lineage: StateLineage,
    pub(super) position: usize,
    pub(super) layers: Vec<LayerState>,
}

impl BackboneState {
    /// Pinned profile/model/tokenizer/renderer/arithmetic identity.
    #[must_use]
    pub fn identity(&self) -> &StateIdentity {
        &self.identity
    }

    /// Branch lineage shared with the prefill root this state derives from.
    #[must_use]
    pub const fn lineage(&self) -> StateLineage {
        self.lineage
    }

    /// Next absolute token position for a continuation suffix.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Exact bytes held by attention KV, recurrent, and convolution tensors.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.layers.iter().map(LayerState::byte_len).sum()
    }

    /// Number of layer-local state records.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Qwen35Backbone {
    pub(crate) fn embedding_rows(&self, ids: &[u32]) -> Result<Vec<f32>, Qwen35Error> {
        Ok(self.embedding.embed(ids)?.values().to_vec())
    }

    /// Load and verify both shards of the immutable base-model revision.
    ///
    /// The reference execution device is the CPU; see
    /// [`Self::load_with_device`] for accelerated loads.
    pub fn load(checkpoint_root: impl AsRef<Path>) -> Result<Self, Qwen35Error> {
        Self::load_with_device(checkpoint_root, Device::Cpu)
    }

    /// Load and verify both shards, executing on `device`.
    ///
    /// The continuation-state arithmetic identity follows the device: CPU
    /// keeps the pinned `"candle-cpu-fp32"` identity, CUDA carries
    /// `"candle-cuda-fp32"` so CPU and CUDA states can never be mixed
    /// (mirroring the MLX identity discipline).
    pub fn load_with_device(
        checkpoint_root: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, Qwen35Error> {
        let identity = match &device {
            Device::Cpu => super::super::pinned_state_identity(),
            Device::Cuda(_) => {
                use super::super::{
                    BACKBONE_ID, BACKBONE_REVISION, PROFILE_ID, STATE_FIRST_RENDERER_ID,
                    TOKENIZER_JSON_SHA256,
                };
                StateIdentity::new(
                    PROFILE_ID,
                    BACKBONE_ID,
                    BACKBONE_REVISION,
                    STATE_FIRST_RENDERER_ID,
                    TOKENIZER_JSON_SHA256,
                    candle_cuda_arithmetic_id(),
                )
                .expect("pinned identity constants are non-empty")
            }
            Device::Metal(_) => {
                return Err(Qwen35Error::InvalidInput(
                    "the candle backbone supports CPU and CUDA; use the MLX backend for Metal"
                        .to_owned(),
                ))
            }
        };
        let checkpoint_root = checkpoint_root.as_ref();
        let embedding = Qwen35Embedding::load(checkpoint_root)?;
        let decoder_shard = verify_decoder_shard(checkpoint_root)?;
        Ok(Self {
            embedding,
            decoder_shard,
            identity,
            device,
        })
    }

    /// The device this backbone executes its forward passes on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Pinned state identity every state produced by this backbone carries.
    #[must_use]
    pub fn identity(&self) -> &StateIdentity {
        &self.identity
    }

    /// Execute embedding, all 32 decoder layers, and final RMSNorm in FP32.
    pub fn forward(&self, input_ids: &[u32]) -> Result<BackboneOutput, Qwen35Error> {
        self.execute(input_ids, None, None)
            .map(|(output, _)| output)
    }

    /// Evaluate a prefix and return its complete immutable continuation state.
    pub fn prefill(
        &self,
        input_ids: &[u32],
    ) -> Result<(BackboneOutput, BackboneState), Qwen35Error> {
        self.execute(input_ids, None, None)
    }

    /// Evaluate a prefix with cooperative cancellation checks between decoder layers.
    pub(crate) fn prefill_controlled(
        &self,
        input_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(BackboneOutput, BackboneState), Qwen35Error> {
        self.execute(input_ids, None, Some(control))
    }

    /// Continue from complete prefix state without mutating the source.
    ///
    /// The source state must carry this backbone's pinned identity; a state
    /// produced under a different profile, model revision, renderer,
    /// tokenizer, or arithmetic path is rejected instead of silently mixed.
    pub fn continue_from(
        &self,
        state: &BackboneState,
        suffix_ids: &[u32],
    ) -> Result<(BackboneOutput, BackboneState), Qwen35Error> {
        if state.identity != self.identity {
            return Err(StateError::IdentityMismatch {
                expected: self.identity.to_string(),
                actual: state.identity.to_string(),
            }
            .into());
        }
        if state.layers.len() != LAYER_COUNT {
            return Err(Qwen35Error::InvalidInput(format!(
                "continuation state has {} layers, expected {LAYER_COUNT}",
                state.layers.len()
            )));
        }
        self.execute(suffix_ids, Some(state), None)
    }

    /// Continue a prefix with cooperative cancellation checks between decoder layers.
    pub(crate) fn continue_from_controlled(
        &self,
        state: &BackboneState,
        suffix_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(BackboneOutput, BackboneState), Qwen35Error> {
        if state.identity != self.identity {
            return Err(StateError::IdentityMismatch {
                expected: self.identity.to_string(),
                actual: state.identity.to_string(),
            }
            .into());
        }
        if state.layers.len() != LAYER_COUNT {
            return Err(Qwen35Error::InvalidInput(format!(
                "continuation state has {} layers, expected {LAYER_COUNT}",
                state.layers.len()
            )));
        }
        self.execute(suffix_ids, Some(state), Some(control))
    }

    fn execute(
        &self,
        input_ids: &[u32],
        previous_state: Option<&BackboneState>,
        control: Option<&ExecutionControl>,
    ) -> Result<(BackboneOutput, BackboneState), Qwen35Error> {
        if let Some(control) = control {
            control.check()?;
        }
        let embedding = self.embedding.embed(input_ids)?;
        if let Some(control) = control {
            control.check()?;
        }
        let token_count = embedding.token_count();
        let position_start = previous_state.map_or(0, |state| state.position);
        // Continuation extends the source state's own timeline and keeps its
        // lineage; a fresh prefill starts a new root.
        let lineage = previous_state.map_or_else(StateLineage::new_root, |state| state.lineage);
        let embedding_last_token = embedding.last_token().to_vec();
        let mut hidden = embedding.values().to_vec();
        let device = self.device.clone();
        // SAFETY: load() verified the immutable size and SHA-256 of both
        // read-only shards. The VarBuilder owns the mapped tensor storage.
        let variables = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[
                    self.embedding.verified_shard_path(),
                    self.decoder_shard.as_path(),
                ],
                DType::F32,
                &device,
            )?
        }
        .pp("model")
        .pp("language_model");
        let layers = variables.pp("layers");
        let mut layer_last_tokens = Vec::with_capacity(LAYER_COUNT);
        let mut next_layers = Vec::with_capacity(LAYER_COUNT);
        for layer_index in 0..LAYER_COUNT {
            if let Some(control) = control {
                control.check()?;
            }
            let layer = DecoderLayer::load(&layers, layer_index, &device, PINNED)?;
            let previous_layer = previous_state.map(|state| &state.layers[layer_index]);
            let (next_hidden, next_state) = layer.forward_with_state(
                &hidden,
                token_count,
                layer_index,
                position_start,
                previous_layer,
            )?;
            hidden = next_hidden;
            next_layers.push(next_state);
            let start = (token_count - 1) * HIDDEN_SIZE;
            layer_last_tokens.push(hidden[start..].to_vec());
        }
        if let Some(control) = control {
            control.check()?;
        }
        let norm = variables
            .get(HIDDEN_SIZE, "norm.weight")?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let final_values = rms_norm_zero_centered(&hidden, token_count, HIDDEN_SIZE, &norm);
        if let Some((index, value)) = final_values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(Qwen35Error::Numerical(format!(
                "final_norm output element {index} is not finite: {value}"
            )));
        }
        Ok((
            BackboneOutput {
                token_count,
                embedding_last_token,
                layer_last_tokens,
                final_values,
            },
            BackboneState {
                identity: self.identity.clone(),
                lineage,
                position: position_start + token_count,
                layers: next_layers,
            },
        ))
    }
}
