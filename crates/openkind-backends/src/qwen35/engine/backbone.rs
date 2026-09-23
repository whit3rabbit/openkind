//! Execution-backend selection for the direct native engine.
//!
//! The request pipeline — state-first tokenization, admission, strategy
//! scheduling, head readout, Jev mapping — is backend-neutral. This module
//! selects the backbone that executes tokens: the Candle FP32 CPU reference
//! backend, or (behind the `mlx` feature on macOS arm64) the MLX reference-ops
//! backend. Both sit behind the same
//! [`SequentialNestedExecutor`](crate::qwen35::backbone::SequentialNestedExecutor)
//! and [`BranchableState`](openkind_runtime::branch::BranchableState)
//! contracts the strategy layer already drives, so the engine adapter and the
//! benchmark harness dispatch them identically.
//!
//! Without the `mlx` feature the Candle CPU backbone binds directly. With it,
//! a closed enum dispatches per request; the CPU and MLX continuation-state
//! types do not unify, so cross-backend state mixing fails to compile instead
//! of failing at runtime.

use std::path::Path;

use crate::qwen35::{Qwen35Backbone, Qwen35Error, SchedulerConfig};

/// Execution backend for a [`Qwen35DecisionEngine`](super::Qwen35DecisionEngine).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Qwen35Backend {
    /// Candle FP32 CPU reference backend — the frozen Phase 3B oracle.
    #[default]
    NativeCpu,
    /// MLX FP32 reference-ops backend (Phase 3M parity-qualified).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxFp32,
    /// MLX native-BF16 reference-ops backend: the separately gated
    /// `mlx-native-bf16` candidate profile. Loading runs the runtime BF16
    /// preflight and fails closed when it does not pass.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxBf16,
}

impl Qwen35Backend {
    /// Stable backend identifier used on the wire and in bench summaries.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            match self {
                Self::NativeCpu => "qwen35-native-cpu",
                Self::MlxFp32 => "qwen35-mlx-fp32",
                Self::MlxBf16 => "qwen35-mlx-bf16",
            }
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        {
            "qwen35-native-cpu"
        }
    }
}

/// Load the selected backbone and keep `scheduler` admission inputs truthful
/// for the backend actually loaded.
///
/// The pinned scheduler constants describe the FP32 CPU continuation state.
/// An MLX backbone therefore re-derives per-state tensor-size inputs from the
/// loaded model. Vectorized MLX batching remains an explicit forced-plan path
/// until parity and performance gates support automatic selection.
///
/// # Errors
/// Returns [`Qwen35Error`] when artifact loading, or for BF16 the runtime
/// preflight, fails.
pub(super) fn load_backbone(
    backend: Qwen35Backend,
    checkpoint_root: &Path,
    scheduler: &mut SchedulerConfig,
) -> Result<EngineBackbone, Qwen35Error> {
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    {
        match backend {
            Qwen35Backend::NativeCpu => {
                Ok(EngineBackbone::Cpu(Qwen35Backbone::load(checkpoint_root)?))
            }
            Qwen35Backend::MlxFp32 => Ok(EngineBackbone::Mlx(load_mlx(
                checkpoint_root,
                mlx::MlxPrecision::Fp32,
                false,
                scheduler,
            )?)),
            Qwen35Backend::MlxBf16 => Ok(EngineBackbone::Mlx(load_mlx(
                checkpoint_root,
                mlx::MlxPrecision::NativeBf16,
                true,
                scheduler,
            )?)),
        }
    }
    #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
    {
        debug_assert!(
            backend == Qwen35Backend::NativeCpu,
            "the only selectable backend without the mlx feature is the CPU backend"
        );
        // The pinned scheduler constants already describe the FP32 CPU state.
        let _ = scheduler;
        Qwen35Backbone::load(checkpoint_root)
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::qwen35::mlx;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn load_mlx(
    checkpoint_root: &Path,
    precision: mlx::MlxPrecision,
    qualify_bf16: bool,
    scheduler: &mut SchedulerConfig,
) -> Result<mlx::MlxQwen35Backbone, Qwen35Error> {
    use crate::qwen35::backbone::ExecutionStrategy;
    use openkind_runtime::BackendCapabilities;

    let runtime = std::sync::Arc::new(mlx::MlxRuntime::new(mlx::MlxRuntimeConfig::default())?);
    if qualify_bf16 {
        runtime.qualify_bf16()?;
    }
    let backbone = mlx::MlxQwen35Backbone::load(checkpoint_root, runtime, precision)?;
    validate_mlx_profile(&backbone)?;
    scheduler.state_fixed_tensor_bytes = backbone.expected_tensor_storage_bytes(0);
    scheduler.state_tensor_bytes_per_token =
        backbone.expected_tensor_storage_bytes(1) - backbone.expected_tensor_storage_bytes(0);
    let forced_vectorized_batch = precision == mlx::MlxPrecision::Fp32
        && scheduler.forced_strategy == Some(ExecutionStrategy::NestedBatched)
        && backbone.supports_vectorized_batch();
    scheduler.backend_capabilities = if forced_vectorized_batch {
        BackendCapabilities::fully_vectorized().with_lane_limits(8, 8)
    } else {
        BackendCapabilities::per_lane()
    };
    Ok(backbone)
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn validate_mlx_profile(backbone: &mlx::MlxQwen35Backbone) -> Result<(), Qwen35Error> {
    use crate::qwen35::mlx::MlxCheckpointFormat;
    use crate::qwen35::{
        require_equal, BACKBONE_ID, BACKBONE_REVISION, PROFILE_ID, STATE_FIRST_RENDERER_ID,
        TOKENIZER_JSON_SHA256,
    };

    let identity = backbone.identity();
    require_equal(
        "checkpoint.format",
        MlxCheckpointFormat::PinnedQwen35Base.as_str(),
        backbone.checkpoint_format().as_str(),
    )?;
    require_equal("profile_id", PROFILE_ID, identity.profile().as_str())?;
    require_equal("backbone_id", BACKBONE_ID, identity.backbone_id())?;
    require_equal(
        "backbone_revision",
        BACKBONE_REVISION,
        identity.backbone_revision(),
    )?;
    require_equal(
        "renderer_id",
        STATE_FIRST_RENDERER_ID,
        identity.renderer_id(),
    )?;
    require_equal(
        "tokenizer_digest",
        TOKENIZER_JSON_SHA256,
        identity.tokenizer_digest(),
    )
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod dispatch {
    //! Closed backbone/state/batch enums unifying the CPU and MLX executors.

    use openkind_runtime::branch::{
        BranchBatch, BranchableState, ProfileId, SchedulingFingerprint, StateError,
    };

    use crate::qwen35::backbone::{BatchContinuation, SequentialNestedExecutor};
    use crate::qwen35::mlx::{MlxBackboneState, MlxBranchBatch, MlxQwen35Backbone};
    use crate::qwen35::ExecutionControl;
    use crate::qwen35::{BackboneState, Qwen35Backbone, Qwen35BranchBatch, Qwen35Error};

    /// One loaded backbone behind the backend-neutral request pipeline.
    pub enum EngineBackbone {
        /// Candle FP32 CPU reference backend.
        Cpu(Qwen35Backbone),
        /// MLX reference-ops backend.
        Mlx(MlxQwen35Backbone),
    }

    /// Continuation state of the loaded engine backbone.
    #[derive(Clone)]
    pub enum EngineBackboneState {
        /// Candle CPU continuation state.
        Cpu(BackboneState),
        /// MLX continuation state.
        Mlx(MlxBackboneState),
    }

    /// Batch handle of the loaded engine backbone's continuation state.
    pub enum EngineBranchBatch {
        /// Candle CPU lane batch.
        Cpu(Qwen35BranchBatch),
        /// MLX lane batch.
        Mlx(MlxBranchBatch),
    }

    impl SequentialNestedExecutor for EngineBackbone {
        type State = EngineBackboneState;

        fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
            match self {
                Self::Cpu(backbone) => {
                    let (feature, state) = SequentialNestedExecutor::prefill(backbone, input_ids)?;
                    Ok((feature, EngineBackboneState::Cpu(state)))
                }
                Self::Mlx(backbone) => {
                    let (feature, state) = SequentialNestedExecutor::prefill(backbone, input_ids)?;
                    Ok((feature, EngineBackboneState::Mlx(state)))
                }
            }
        }

        fn continue_from(
            &self,
            state: &Self::State,
            suffix_ids: &[u32],
        ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
            match (self, state) {
                (Self::Cpu(backbone), EngineBackboneState::Cpu(state)) => {
                    let (feature, next) =
                        SequentialNestedExecutor::continue_from(backbone, state, suffix_ids)?;
                    Ok((feature, EngineBackboneState::Cpu(next)))
                }
                (Self::Mlx(backbone), EngineBackboneState::Mlx(state)) => {
                    let (feature, next) =
                        SequentialNestedExecutor::continue_from(backbone, state, suffix_ids)?;
                    Ok((feature, EngineBackboneState::Mlx(next)))
                }
                _ => Err(Qwen35Error::InvalidInput(
                    "engine backbone and continuation state backends do not match".into(),
                )),
            }
        }

        fn continue_batch_from(
            &self,
            states: &[&Self::State],
            suffix_ids: &[&[u32]],
        ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
            match self {
                Self::Cpu(backbone) => {
                    let states = states
                        .iter()
                        .map(|state| match *state {
                            EngineBackboneState::Cpu(state) => Ok(state),
                            EngineBackboneState::Mlx(_) => Err(Qwen35Error::InvalidInput(
                                "CPU engine received an MLX continuation state".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let result = SequentialNestedExecutor::continue_batch_from(
                        backbone, &states, suffix_ids,
                    )?;
                    let (lanes, mode) = result.into_parts();
                    Ok(BatchContinuation::new(
                        lanes
                            .into_iter()
                            .map(|(feature, state)| (feature, EngineBackboneState::Cpu(state)))
                            .collect(),
                        mode,
                    ))
                }
                Self::Mlx(backbone) => {
                    let states = states
                        .iter()
                        .map(|state| match *state {
                            EngineBackboneState::Mlx(state) => Ok(state),
                            EngineBackboneState::Cpu(_) => Err(Qwen35Error::InvalidInput(
                                "MLX engine received a CPU continuation state".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let result = SequentialNestedExecutor::continue_batch_from(
                        backbone, &states, suffix_ids,
                    )?;
                    let (lanes, mode) = result.into_parts();
                    Ok(BatchContinuation::new(
                        lanes
                            .into_iter()
                            .map(|(feature, state)| (feature, EngineBackboneState::Mlx(state)))
                            .collect(),
                        mode,
                    ))
                }
            }
        }

        fn continue_batch_from_controlled(
            &self,
            states: &[&Self::State],
            suffix_ids: &[&[u32]],
            control: &ExecutionControl,
        ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
            match self {
                Self::Cpu(backbone) => {
                    let states = states
                        .iter()
                        .map(|state| match *state {
                            EngineBackboneState::Cpu(state) => Ok(state),
                            EngineBackboneState::Mlx(_) => Err(Qwen35Error::InvalidInput(
                                "CPU engine received an MLX continuation state".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let result = SequentialNestedExecutor::continue_batch_from_controlled(
                        backbone, &states, suffix_ids, control,
                    )?;
                    let (lanes, mode) = result.into_parts();
                    Ok(BatchContinuation::new(
                        lanes
                            .into_iter()
                            .map(|(feature, state)| (feature, EngineBackboneState::Cpu(state)))
                            .collect(),
                        mode,
                    ))
                }
                Self::Mlx(backbone) => {
                    let states = states
                        .iter()
                        .map(|state| match *state {
                            EngineBackboneState::Mlx(state) => Ok(state),
                            EngineBackboneState::Cpu(_) => Err(Qwen35Error::InvalidInput(
                                "MLX engine received a CPU continuation state".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let result = SequentialNestedExecutor::continue_batch_from_controlled(
                        backbone, &states, suffix_ids, control,
                    )?;
                    let (lanes, mode) = result.into_parts();
                    Ok(BatchContinuation::new(
                        lanes
                            .into_iter()
                            .map(|(feature, state)| (feature, EngineBackboneState::Mlx(state)))
                            .collect(),
                        mode,
                    ))
                }
            }
        }
    }

    impl BranchableState for EngineBackboneState {
        type Batch = EngineBranchBatch;

        fn profile_id(&self) -> &ProfileId {
            match self {
                Self::Cpu(state) => state.profile_id(),
                Self::Mlx(state) => state.profile_id(),
            }
        }

        fn position(&self) -> usize {
            match self {
                Self::Cpu(state) => state.position(),
                Self::Mlx(state) => state.position(),
            }
        }

        fn tensor_storage_bytes(&self) -> usize {
            match self {
                Self::Cpu(state) => state.tensor_storage_bytes(),
                Self::Mlx(state) => state.tensor_storage_bytes(),
            }
        }

        fn scheduling_fingerprint(&self) -> SchedulingFingerprint {
            match self {
                Self::Cpu(state) => state.scheduling_fingerprint(),
                Self::Mlx(state) => state.scheduling_fingerprint(),
            }
        }

        fn fork_one(&self) -> Result<Self, StateError> {
            match self {
                Self::Cpu(state) => Ok(Self::Cpu(state.fork_one()?)),
                Self::Mlx(state) => Ok(Self::Mlx(state.fork_one()?)),
            }
        }

        fn fork_batch(&self, lanes: usize) -> Result<Self::Batch, StateError> {
            match self {
                Self::Cpu(state) => Ok(EngineBranchBatch::Cpu(state.fork_batch(lanes)?)),
                Self::Mlx(state) => Ok(EngineBranchBatch::Mlx(state.fork_batch(lanes)?)),
            }
        }
    }

    impl BranchBatch for EngineBranchBatch {
        type State = EngineBackboneState;

        fn lanes(&self) -> usize {
            match self {
                Self::Cpu(batch) => batch.lanes(),
                Self::Mlx(batch) => batch.lanes(),
            }
        }

        fn tensor_storage_bytes(&self) -> usize {
            match self {
                Self::Cpu(batch) => batch.tensor_storage_bytes(),
                Self::Mlx(batch) => batch.tensor_storage_bytes(),
            }
        }

        fn select(&self, index: usize) -> Result<Self::State, StateError> {
            match self {
                Self::Cpu(batch) => Ok(EngineBackboneState::Cpu(batch.select(index)?)),
                Self::Mlx(batch) => Ok(EngineBackboneState::Mlx(batch.select(index)?)),
            }
        }

        fn gather(&self, indices: &[usize]) -> Result<Self, StateError> {
            match self {
                Self::Cpu(batch) => Ok(Self::Cpu(batch.gather(indices)?)),
                Self::Mlx(batch) => Ok(Self::Mlx(batch.gather(indices)?)),
            }
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) use dispatch::EngineBackbone;

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(super) use crate::qwen35::Qwen35Backbone as EngineBackbone;
