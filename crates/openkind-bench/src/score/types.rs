//! Argument and configuration types for the `score` benchmark subcommand.

use std::path::PathBuf;

use anyhow::{bail, Result};
use openkind_backends::qwen35::{ExecutionStrategy, Qwen35Backend};

/// Model alias used for every bench request inside the run-local registry.
pub(crate) const BENCH_ALIAS: &str = "bench";

/// The `engine` choice for `score`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// In-process deterministic mock; plumbing and summary smoke.
    Mock,
    /// Pinned Qwen3.5 native CPU engine; real scoring and timing.
    Qwen35,
    /// Pinned Qwen3.5 MLX FP32 reference-ops engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Qwen35MlxFp32,
    /// Pinned Qwen3.5 MLX native-BF16 candidate engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Qwen35MlxBf16,
}

/// Native backend bound to a non-mock [`EngineKind`].
///
/// # Panics
/// Panics when called for [`EngineKind::Mock`]; callers gate on the mock
/// engine before reaching this mapping.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn native_backend(engine: EngineKind) -> Qwen35Backend {
    match engine {
        EngineKind::Mock => panic!("the mock engine has no native backend"),
        EngineKind::Qwen35 => Qwen35Backend::NativeCpu,
        EngineKind::Qwen35MlxFp32 => Qwen35Backend::MlxFp32,
        EngineKind::Qwen35MlxBf16 => Qwen35Backend::MlxBf16,
    }
}

/// Native backend bound to a non-mock [`EngineKind`].
///
/// # Panics
/// Panics when called for [`EngineKind::Mock`]; callers gate on the mock
/// engine before reaching this mapping.
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn native_backend(engine: EngineKind) -> Qwen35Backend {
    match engine {
        EngineKind::Mock => panic!("the mock engine has no native backend"),
        EngineKind::Qwen35 => Qwen35Backend::NativeCpu,
    }
}

/// File-stem slug for the summary and prediction outputs. The CPU engine
/// keeps its historical `qwen35` slug; MLX engines get distinct slugs so
/// sweeps of different backends never overwrite each other.
pub(crate) fn engine_slug(engine: EngineKind) -> &'static str {
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    {
        match engine {
            EngineKind::Mock => "mock",
            EngineKind::Qwen35 => "qwen35",
            EngineKind::Qwen35MlxFp32 => "qwen35-mlx-fp32",
            EngineKind::Qwen35MlxBf16 => "qwen35-mlx-bf16",
        }
    }
    #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
    {
        match engine {
            EngineKind::Mock => "mock",
            EngineKind::Qwen35 => "qwen35",
        }
    }
}

/// One entry of the `--strategies` sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategySpec {
    /// Let the measured scheduler choose.
    ChooseStrategy,
    /// Force a concrete execution plan.
    Forced(ExecutionStrategy),
}

impl StrategySpec {
    /// Parse one comma-separated strategy token.
    ///
    /// # Errors
    /// Returns an error for unknown tokens.
    pub fn parse(token: &str) -> Result<Self> {
        match token {
            "choose_strategy" | "auto" => Ok(Self::ChooseStrategy),
            "repeated_full" => Ok(Self::Forced(ExecutionStrategy::RepeatedFull)),
            "nested_sequential" => Ok(Self::Forced(ExecutionStrategy::NestedSequential)),
            "nested_batched" => Ok(Self::Forced(ExecutionStrategy::NestedBatched)),
            other => bail!(
                "unknown strategy `{other}`; expected one of repeated_full, \
                 nested_sequential, nested_batched, choose_strategy"
            ),
        }
    }

    /// Label name for this strategy spec.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::ChooseStrategy => "choose_strategy",
            Self::Forced(ExecutionStrategy::RepeatedFull) => "repeated_full",
            Self::Forced(ExecutionStrategy::NestedSequential) => "nested_sequential",
            Self::Forced(ExecutionStrategy::NestedBatched) => "nested_batched",
        }
    }

    /// Diagnostic forced strategy override for the scheduler.
    pub(crate) fn scheduler_override(self) -> Option<ExecutionStrategy> {
        match self {
            Self::ChooseStrategy => None,
            Self::Forced(strategy) => Some(strategy),
        }
    }
}

/// Arguments for a scoring run (mirrors the CLI flags).
#[derive(Debug, Clone)]
pub struct ScoreArgs {
    /// Workload JSONL path.
    pub input: PathBuf,
    /// Engine under test.
    pub engine: EngineKind,
    /// Output directory for summary and predictions.
    pub output_dir: PathBuf,
    /// Comma-separated strategy sweep (native engine only).
    pub strategies: Vec<StrategySpec>,
    /// Timed repetitions per group; predictions come from the last rep.
    pub reps: usize,
    /// Group rows sharing one state into one request (default).
    pub group: bool,
    /// Attribution label for the host, recorded in the summary.
    pub host: Option<String>,
    /// Commit hash under measurement, recorded in the summary.
    pub commit: Option<String>,
    /// Write the summary as pretty JSON.
    pub pretty: bool,
    /// Native engine artifacts (required for `EngineKind::Qwen35`).
    pub bundle_root: Option<PathBuf>,
    /// Native engine artifacts (required for `EngineKind::Qwen35`).
    pub checkpoint_root: Option<PathBuf>,
    /// Native engine artifacts (required for `EngineKind::Qwen35`).
    pub tokenizer_path: Option<PathBuf>,
}

/// Run-level output of [`run_score`](super::run_score).
#[derive(Debug, Clone)]
pub struct ScoreOutcome {
    /// The summary document (also written to `summary-<engine>.json`).
    pub summary: serde_json::Value,
}

/// strategies accepted by `--strategies`, for CLI help text.
pub const STRATEGY_HELP: &str = "repeated_full, nested_sequential, nested_batched, choose_strategy";

/// Default sweep for the native engine.
pub const DEFAULT_STRATEGIES: [StrategySpec; 4] = [
    StrategySpec::Forced(ExecutionStrategy::RepeatedFull),
    StrategySpec::Forced(ExecutionStrategy::NestedSequential),
    StrategySpec::Forced(ExecutionStrategy::NestedBatched),
    StrategySpec::ChooseStrategy,
];

/// Reject strategy/backend combinations that cannot produce qualifying evidence.
pub(crate) fn validate_strategy_selection(
    engine: EngineKind,
    strategies: &[StrategySpec],
) -> Result<()> {
    if engine != EngineKind::Mock && strategies.is_empty() {
        bail!("native benchmark runs require at least one execution strategy");
    }
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    if engine == EngineKind::Qwen35MlxBf16
        && strategies
            .iter()
            .any(|strategy| *strategy != StrategySpec::Forced(ExecutionStrategy::RepeatedFull))
    {
        bail!(
            "qwen35-mlx-bf16 is restricted to --strategies repeated_full until nested BF16 continuation is qualified"
        );
    }
    Ok(())
}
