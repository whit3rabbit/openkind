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
    /// Decoder layers and final-norm weights held across forwards, when the
    /// operator opted in. `None` keeps the default: every forward re-maps the
    /// shards and holds one layer's widened weights at a time.
    resident: Option<ResidentWeights>,
}

#[derive(Debug)]
struct ResidentWeights {
    layers: Vec<DecoderLayer>,
    final_norm: Vec<f32>,
}

/// Whether a backbone keeps its widened FP32 decoder weights resident.
///
/// Each forward otherwise copies every decoder layer out of the mapped shard
/// and widens it from BF16, which dominated the profile of the pinned 4B
/// checkpoint on CPU. Resident weights remove that work at the cost of about
/// the FP32 decoder size (roughly 14 to 17 GB for the pinned 4B) held for the
/// backbone's lifetime. Results are bit-identical in every mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidentWeightsMode {
    /// Re-map and widen one layer at a time on every forward.
    #[default]
    Off,
    /// Always keep the weights resident, whatever the free memory.
    On,
    /// Keep them resident when the host has room for them and the process
    /// ceiling allows it; otherwise stay off and log why.
    Auto,
}

static RESIDENT_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static RESIDENT_CEILING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Free memory that must remain after the resident copy under
/// [`ResidentWeightsMode::Auto`], for the OS, page cache, and forward scratch.
const RESIDENT_AUTO_HEADROOM_BYTES: u64 = 4 << 30;

/// Set the policy for backbones loaded after this call. The library default
/// is [`ResidentWeightsMode::Off`], so tests and examples stay deterministic;
/// the daemon selects its own default. `max_process_bytes` is the operator's
/// process-memory ceiling, which `Auto` respects.
pub fn set_resident_weights_policy(mode: ResidentWeightsMode, max_process_bytes: Option<usize>) {
    use std::sync::atomic::Ordering::Relaxed;
    RESIDENT_MODE.store(
        match mode {
            ResidentWeightsMode::Off => 0,
            ResidentWeightsMode::On => 1,
            ResidentWeightsMode::Auto => 2,
        },
        Relaxed,
    );
    RESIDENT_CEILING.store(max_process_bytes.unwrap_or(0), Relaxed);
}

fn resident_weights_mode() -> ResidentWeightsMode {
    match RESIDENT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => ResidentWeightsMode::On,
        2 => ResidentWeightsMode::Auto,
        _ => ResidentWeightsMode::Off,
    }
}

/// Decide whether `Auto` may keep `estimate` bytes resident. `Err` carries the
/// reason, which the caller logs.
fn auto_resident_decision(
    estimate: u64,
    available: Option<u64>,
    peak_resident: Option<u64>,
    process_ceiling: Option<u64>,
) -> Result<(), String> {
    let Some(available) = available else {
        return Err("free memory could not be read on this host".to_owned());
    };
    let needed = estimate.saturating_add(RESIDENT_AUTO_HEADROOM_BYTES);
    if available < needed {
        return Err(format!(
            "{:.1} GiB free is below the {:.1} GiB needed (resident weights {:.1} GiB plus {:.1} GiB headroom)",
            gib(available),
            gib(needed),
            gib(estimate),
            gib(RESIDENT_AUTO_HEADROOM_BYTES),
        ));
    }
    if let (Some(ceiling), Some(peak)) = (process_ceiling, peak_resident) {
        if peak.saturating_add(estimate) > ceiling {
            return Err(format!(
                "resident weights ({:.1} GiB) on top of {:.1} GiB already observed would exceed the {:.1} GiB process-memory ceiling",
                gib(estimate),
                gib(peak),
                gib(ceiling),
            ));
        }
    }
    Ok(())
}

/// Pure residency verdict for backbones that decide at construction.
///
/// `Ok(true)` keeps the widened weights resident, `Ok(false)` leaves the
/// per-forward path in place with nothing to report, and `Err` carries the
/// reason an `Auto` request was declined, which the caller logs. `Auto` only
/// applies when the footprint can be estimated from mapped shards and the
/// forward runs on the CPU, because the host-memory guard says nothing about
/// device memory and an adapter source has no on-disk size to read.
fn text_resident_verdict(
    mode: ResidentWeightsMode,
    estimate: Option<u64>,
    on_cpu: bool,
    available: Option<u64>,
    peak_resident: Option<u64>,
    process_ceiling: Option<u64>,
) -> Result<bool, String> {
    match mode {
        ResidentWeightsMode::Off => Ok(false),
        ResidentWeightsMode::On => Ok(true),
        ResidentWeightsMode::Auto => {
            let Some(estimate) = estimate.filter(|_| on_cpu) else {
                return Err(
                    "automatic residency needs shard-backed weights on the CPU; use `on` to force it"
                        .to_owned(),
                );
            };
            auto_resident_decision(estimate, available, peak_resident, process_ceiling)
                .map(|()| true)
        }
    }
}

/// Residency verdict for a [`TextBackbone`](super::text::TextBackbone) under
/// the process-wide policy set by [`set_resident_weights_policy`].
pub(super) fn text_resident_decision(estimate: Option<u64>, on_cpu: bool) -> Result<bool, String> {
    let ceiling = RESIDENT_CEILING.load(std::sync::atomic::Ordering::Relaxed);
    text_resident_verdict(
        resident_weights_mode(),
        estimate,
        on_cpu,
        openkind_runtime::available_memory_bytes(),
        openkind_runtime::peak_resident_bytes()
            .ok()
            .map(|bytes| bytes as u64),
        (ceiling > 0).then_some(ceiling as u64),
    )
}

#[allow(clippy::cast_precision_loss)]
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1_u64 << 30) as f64
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
        let backbone = Self {
            embedding,
            decoder_shard,
            identity,
            device,
            resident: None,
        };
        match resident_weights_mode() {
            ResidentWeightsMode::Off => Ok(backbone),
            ResidentWeightsMode::On => backbone.with_resident_weights(),
            ResidentWeightsMode::Auto => {
                let ceiling = RESIDENT_CEILING.load(std::sync::atomic::Ordering::Relaxed);
                let estimate = backbone.resident_estimate_bytes();
                let verdict = estimate
                    .map_err(|error| error.to_string())
                    .and_then(|estimate| {
                        auto_resident_decision(
                            estimate,
                            openkind_runtime::available_memory_bytes(),
                            openkind_runtime::peak_resident_bytes()
                                .ok()
                                .map(|bytes| bytes as u64),
                            (ceiling > 0).then_some(ceiling as u64),
                        )
                        .map(|()| estimate)
                    });
                match verdict {
                    Ok(estimate) => {
                        tracing::info!(
                            resident_bytes = estimate,
                            "keeping Qwen3.5 decoder weights resident (free memory is sufficient)"
                        );
                        backbone.with_resident_weights()
                    }
                    Err(reason) => {
                        tracing::warn!(
                            %reason,
                            "not keeping Qwen3.5 decoder weights resident; each forward will re-map and widen them (set --qwen35-resident-weights on to force)"
                        );
                        Ok(backbone)
                    }
                }
            }
        }
    }

    /// Upper bound on the FP32 bytes [`Self::with_resident_weights`] holds:
    /// both shards' BF16 bytes less the embedding matrix, widened to FP32. It
    /// includes any non-decoder tensors the shards carry, so it errs high.
    fn resident_estimate_bytes(&self) -> Result<u64, Qwen35Error> {
        let length = |path: &Path| {
            std::fs::metadata(path)
                .map(|metadata| metadata.len())
                .map_err(|source| Qwen35Error::Io {
                    path: path.to_path_buf(),
                    source,
                })
        };
        let shards = length(self.embedding.verified_shard_path())?
            .saturating_add(length(self.decoder_shard.as_path())?);
        let embedding_matrix = (super::embedding::VOCAB_SIZE * HIDDEN_SIZE * 2) as u64;
        Ok(shards.saturating_sub(embedding_matrix).saturating_mul(2))
    }

    /// Load every decoder layer and the final norm once and keep them for all
    /// later forwards. See [`set_resident_weights_default`] for the memory cost.
    pub fn with_resident_weights(mut self) -> Result<Self, Qwen35Error> {
        let variables = self.variables()?;
        let layers_root = variables.pp("layers");
        let layers = (0..LAYER_COUNT)
            .map(|layer_index| DecoderLayer::load(&layers_root, layer_index, &self.device, PINNED))
            .collect::<Result<Vec<_>, _>>()?;
        let final_norm = variables
            .get(HIDDEN_SIZE, "norm.weight")?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let held: usize = layers.iter().map(DecoderLayer::resident_bytes).sum();
        tracing::info!(
            layers = layers.len(),
            resident_bytes = held,
            "Qwen3.5 decoder weights are resident"
        );
        self.resident = Some(ResidentWeights { layers, final_norm });
        Ok(self)
    }

    /// FP32 view of both shards, scoped to the language model.
    fn variables(&self) -> Result<VarBuilder<'static>, Qwen35Error> {
        // SAFETY: load() verified the immutable size and SHA-256 of both
        // read-only shards. The VarBuilder owns the mapped tensor storage.
        let variables = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[
                    self.embedding.verified_shard_path(),
                    self.decoder_shard.as_path(),
                ],
                DType::F32,
                &self.device,
            )?
        };
        Ok(variables.pp("model").pp("language_model"))
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
        // Resident weights skip the per-forward shard mapping entirely.
        let transient = if self.resident.is_none() {
            Some(self.variables()?)
        } else {
            None
        };
        let layers_root = transient.as_ref().map(|variables| variables.pp("layers"));
        let mut layer_last_tokens = Vec::with_capacity(LAYER_COUNT);
        let mut next_layers = Vec::with_capacity(LAYER_COUNT);
        for layer_index in 0..LAYER_COUNT {
            if let Some(control) = control {
                control.check()?;
            }
            let loaded;
            let layer = match (&self.resident, &layers_root) {
                (Some(resident), _) => &resident.layers[layer_index],
                (None, Some(root)) => {
                    loaded = DecoderLayer::load(root, layer_index, &device, PINNED)?;
                    &loaded
                }
                (None, None) => unreachable!("transient weights are built when none are resident"),
            };
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
        let loaded_norm;
        let norm: &[f32] = match (&self.resident, &transient) {
            (Some(resident), _) => &resident.final_norm,
            (None, Some(variables)) => {
                loaded_norm = variables
                    .get(HIDDEN_SIZE, "norm.weight")?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                &loaded_norm
            }
            (None, None) => unreachable!("transient weights are built when none are resident"),
        };
        let final_values = rms_norm_zero_centered(&hidden, token_count, HIDDEN_SIZE, norm);
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

#[cfg(test)]
mod resident_policy_tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn auto_requires_the_estimate_plus_headroom_to_be_free() {
        let estimate = 14 * GIB;
        assert!(auto_resident_decision(estimate, Some(20 * GIB), None, None).is_ok());
        assert!(auto_resident_decision(estimate, Some(18 * GIB), None, None).is_ok());
        let error = auto_resident_decision(estimate, Some(17 * GIB), None, None).unwrap_err();
        assert!(error.contains("free is below"), "{error}");
    }

    #[test]
    fn auto_stays_off_when_free_memory_is_unreadable() {
        let error = auto_resident_decision(GIB, None, None, None).unwrap_err();
        assert!(error.contains("could not be read"), "{error}");
    }

    #[test]
    fn auto_respects_the_process_memory_ceiling() {
        let estimate = 14 * GIB;
        let free = Some(64 * GIB);
        assert!(auto_resident_decision(estimate, free, Some(8 * GIB), Some(24 * GIB)).is_ok());
        let error =
            auto_resident_decision(estimate, free, Some(12 * GIB), Some(24 * GIB)).unwrap_err();
        assert!(error.contains("process-memory ceiling"), "{error}");
        // No observation means the ceiling cannot be judged here; the scheduler
        // still enforces it per request.
        assert!(auto_resident_decision(estimate, free, None, Some(24 * GIB)).is_ok());
    }

    #[test]
    fn text_verdict_follows_the_mode() {
        let plenty = Some(64 * GIB);
        let verdict = |mode, estimate, on_cpu| {
            text_resident_verdict(mode, estimate, on_cpu, plenty, None, None)
        };
        assert_eq!(
            verdict(ResidentWeightsMode::Off, Some(GIB), true),
            Ok(false)
        );
        // `On` is an explicit operator choice: no estimate and no CPU needed.
        assert_eq!(verdict(ResidentWeightsMode::On, None, false), Ok(true));
        assert_eq!(
            verdict(ResidentWeightsMode::Auto, Some(14 * GIB), true),
            Ok(true)
        );
    }

    #[test]
    fn text_auto_declines_what_it_cannot_size_or_guard() {
        let plenty = Some(64 * GIB);
        // An adapter source has no on-disk size to estimate from.
        let error =
            text_resident_verdict(ResidentWeightsMode::Auto, None, true, plenty, None, None)
                .unwrap_err();
        assert!(error.contains("shard-backed"), "{error}");
        // The host-memory guard says nothing about device memory.
        let error = text_resident_verdict(
            ResidentWeightsMode::Auto,
            Some(GIB),
            false,
            plenty,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.contains("on the CPU"), "{error}");
        // The usual free-memory guard still applies.
        let error = text_resident_verdict(
            ResidentWeightsMode::Auto,
            Some(30 * GIB),
            true,
            Some(32 * GIB),
            None,
            None,
        )
        .unwrap_err();
        assert!(error.contains("free is below"), "{error}");
    }

    #[test]
    fn policy_round_trips_and_defaults_to_off() {
        assert_eq!(ResidentWeightsMode::default(), ResidentWeightsMode::Off);
        set_resident_weights_policy(ResidentWeightsMode::Auto, Some(7));
        assert_eq!(resident_weights_mode(), ResidentWeightsMode::Auto);
        set_resident_weights_policy(ResidentWeightsMode::On, None);
        assert_eq!(resident_weights_mode(), ResidentWeightsMode::On);
        set_resident_weights_policy(ResidentWeightsMode::Off, None);
        assert_eq!(resident_weights_mode(), ResidentWeightsMode::Off);
    }
}
