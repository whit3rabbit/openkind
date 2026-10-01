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
    /// Pinned decoder-logit-letter engine (Qwen2.5-0.5B-Instruct).
    DecoderLetter,
    /// Pinned encoder-nli engine (DistilBERT MNLI).
    EncoderNli,
    /// Pinned encoder-instruct-label engine (GLiClass label markers).
    EncoderInstructLabel,
    /// Pinned decoder-logit-llm engine (GGUF q8_0).
    DecoderLlm,
    /// Pinned schema-scorer engine (MS MARCO cross-encoder).
    SchemaScorer,
    /// Router-script composite over mock siblings; routing-overhead only.
    RouterScript,
    /// Pinned qwen3guard engine (Qwen3Guard-Stream 0.6B).
    Qwen3Guard,
    /// Pinned kev engine (Kev-0.6B pointer readout).
    Kev,
    /// Pinned decoder-logit-qwen35 engine (JevK5 letter-logit readout).
    DecoderLogitQwen35,
    /// Pinned plumb-4b engine (Plumb-4B letter-logit readout, single read).
    Plumb4b,
    /// Pinned decider-4b engine (slot-logit readout, isolated score levels).
    Decider4b,
    /// Pinned laya-english engine (English ModernBERT-large decision encoder).
    LayaEnglish,
    /// Pinned laya-multilingual engine (mmBERT-base decision encoder).
    LayaMultilingual,
    /// Pinned laya-typed-decisions engine (fine-tuned ModernBERT-large).
    LayaTypedDecisions,
    /// Pinned von engine (von-1.1 option-marker ModernBERT-large).
    Von,
    /// Winnow learned router over mock siblings; router-cost only.
    Winnow,
    /// Pinned Qwen3.5 MLX FP32 reference-ops engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Qwen35MlxFp32,
    /// Pinned Qwen3.5 MLX native-BF16 candidate engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Qwen35MlxBf16,
    /// Pinned laya-english MLX FP32 engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    LayaEnglishMlxFp32,
    /// Pinned laya-multilingual MLX FP32 engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    LayaMultilingualMlxFp32,
    /// Pinned laya-typed-decisions MLX FP32 engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    LayaTypedDecisionsMlxFp32,
    /// Pinned encoder-instruct-label MLX FP32 engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    EncoderInstructLabelMlxFp32,
    /// Pinned decoder-logit-qwen35 MLX FP32 engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    DecoderLogitQwen35MlxFp32,
    /// Pinned plumb-4b MLX FP32 engine (`mlx` feature).
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Plumb4bMlxFp32,
}

/// Native backend bound to a non-mock [`EngineKind`].
///
/// # Panics
/// Panics when called for [`EngineKind::Mock`]; callers gate on the mock
/// engine before reaching this mapping.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn native_backend(engine: EngineKind) -> Qwen35Backend {
    match engine {
        EngineKind::Mock
        | EngineKind::DecoderLetter
        | EngineKind::EncoderNli
        | EngineKind::EncoderInstructLabel
        | EngineKind::DecoderLlm
        | EngineKind::SchemaScorer
        | EngineKind::RouterScript
        | EngineKind::Qwen3Guard
        | EngineKind::Winnow
        | EngineKind::Kev
        | EngineKind::DecoderLogitQwen35
        | EngineKind::Plumb4b
        | EngineKind::Decider4b
        | EngineKind::LayaEnglish
        | EngineKind::LayaMultilingual
        | EngineKind::LayaTypedDecisions
        | EngineKind::Von
        | EngineKind::LayaEnglishMlxFp32
        | EngineKind::LayaMultilingualMlxFp32
        | EngineKind::LayaTypedDecisionsMlxFp32
        | EngineKind::EncoderInstructLabelMlxFp32
        | EngineKind::DecoderLogitQwen35MlxFp32 => {
            panic!("the mock and surveyed-family engines have no Qwen35 native backend")
        }
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
        EngineKind::Mock
        | EngineKind::DecoderLetter
        | EngineKind::EncoderNli
        | EngineKind::EncoderInstructLabel
        | EngineKind::DecoderLlm
        | EngineKind::SchemaScorer
        | EngineKind::RouterScript
        | EngineKind::Qwen3Guard
        | EngineKind::Winnow
        | EngineKind::Kev
        | EngineKind::DecoderLogitQwen35
        | EngineKind::Plumb4b
        | EngineKind::Decider4b
        | EngineKind::LayaEnglish
        | EngineKind::LayaMultilingual
        | EngineKind::LayaTypedDecisions
        | EngineKind::Von => {
            panic!("the mock and surveyed-family engines have no Qwen35 native backend")
        }
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
            EngineKind::DecoderLetter => "decoder-letter",
            EngineKind::EncoderNli => "encoder-nli",
            EngineKind::EncoderInstructLabel => "encoder-instruct-label",
            EngineKind::DecoderLlm => "decoder-llm",
            EngineKind::SchemaScorer => "schema-scorer",
            EngineKind::RouterScript => "router-script",
            EngineKind::Qwen3Guard => "qwen3guard",
            EngineKind::Kev => "kev",
            EngineKind::DecoderLogitQwen35 => "decoder-logit-qwen35",
            EngineKind::Plumb4b => "plumb-4b",
            EngineKind::Decider4b => "decider-4b",
            EngineKind::LayaEnglish => "laya-english",
            EngineKind::LayaMultilingual => "laya-multilingual",
            EngineKind::LayaTypedDecisions => "laya-typed-decisions",
            EngineKind::Von => "von",
            EngineKind::Winnow => "winnow",
            EngineKind::Qwen35MlxFp32 => "qwen35-mlx-fp32",
            EngineKind::Qwen35MlxBf16 => "qwen35-mlx-bf16",
            EngineKind::LayaEnglishMlxFp32 => "laya-english-mlx-fp32",
            EngineKind::LayaMultilingualMlxFp32 => "laya-multilingual-mlx-fp32",
            EngineKind::LayaTypedDecisionsMlxFp32 => "laya-typed-decisions-mlx-fp32",
            EngineKind::EncoderInstructLabelMlxFp32 => "encoder-instruct-label-mlx-fp32",
            EngineKind::DecoderLogitQwen35MlxFp32 => "decoder-logit-qwen35-mlx-fp32",
            EngineKind::Plumb4bMlxFp32 => "plumb-4b-mlx-fp32",
        }
    }
    #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
    {
        match engine {
            EngineKind::Mock => "mock",
            EngineKind::Qwen35 => "qwen35",
            EngineKind::DecoderLetter => "decoder-letter",
            EngineKind::EncoderNli => "encoder-nli",
            EngineKind::EncoderInstructLabel => "encoder-instruct-label",
            EngineKind::DecoderLlm => "decoder-llm",
            EngineKind::SchemaScorer => "schema-scorer",
            EngineKind::RouterScript => "router-script",
            EngineKind::Qwen3Guard => "qwen3guard",
            EngineKind::Kev => "kev",
            EngineKind::DecoderLogitQwen35 => "decoder-logit-qwen35",
            EngineKind::Plumb4b => "plumb-4b",
            EngineKind::Decider4b => "decider-4b",
            EngineKind::LayaEnglish => "laya-english",
            EngineKind::LayaMultilingual => "laya-multilingual",
            EngineKind::LayaTypedDecisions => "laya-typed-decisions",
            EngineKind::Von => "von",
            EngineKind::Winnow => "winnow",
        }
    }
}

/// Surveyed-family engines that load from a `--model-root` directory.
pub fn is_family_engine_public(engine: EngineKind) -> bool {
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    if matches!(
        engine,
        EngineKind::LayaEnglishMlxFp32
            | EngineKind::LayaMultilingualMlxFp32
            | EngineKind::LayaTypedDecisionsMlxFp32
            | EngineKind::EncoderInstructLabelMlxFp32
            | EngineKind::DecoderLogitQwen35MlxFp32
            | EngineKind::Plumb4bMlxFp32
    ) {
        return true;
    }
    matches!(
        engine,
        EngineKind::DecoderLetter
            | EngineKind::EncoderNli
            | EngineKind::EncoderInstructLabel
            | EngineKind::DecoderLlm
            | EngineKind::SchemaScorer
            | EngineKind::RouterScript
            | EngineKind::Qwen3Guard
            | EngineKind::Winnow
            | EngineKind::Kev
            | EngineKind::DecoderLogitQwen35
            | EngineKind::Plumb4b
            | EngineKind::Decider4b
            | EngineKind::LayaEnglish
            | EngineKind::LayaMultilingual
            | EngineKind::LayaTypedDecisions
            | EngineKind::Von
    )
}

pub(crate) use is_family_engine_public as is_family_engine;

/// `(engine slug, profile id, backbone revision)` provenance for a
/// surveyed-family engine. Returns `None` for the mock and Qwen35 engines.
pub(crate) fn family_identity(
    engine: EngineKind,
) -> Option<(&'static str, &'static str, &'static str)> {
    match engine {
        EngineKind::DecoderLetter => Some((
            openkind_backends::families::decoder_logit_letter::FAMILY_SLUG,
            openkind_backends::families::decoder_logit_letter::PROFILE_ID,
            openkind_backends::families::decoder_logit_letter::BACKBONE_REVISION,
        )),
        EngineKind::EncoderNli => Some((
            openkind_backends::families::encoder_nli::FAMILY_SLUG,
            openkind_backends::families::encoder_nli::PROFILE_ID,
            openkind_backends::families::encoder_nli::BACKBONE_REVISION,
        )),
        EngineKind::EncoderInstructLabel => Some((
            openkind_backends::families::encoder_instruct_label::FAMILY_SLUG,
            openkind_backends::families::encoder_instruct_label::PROFILE_ID,
            openkind_backends::families::encoder_instruct_label::BACKBONE_REVISION,
        )),
        EngineKind::DecoderLlm => Some((
            openkind_backends::families::decoder_logit_llm::FAMILY_SLUG,
            openkind_backends::families::decoder_logit_llm::PROFILE_ID,
            openkind_backends::families::decoder_logit_llm::BACKBONE_REVISION,
        )),
        EngineKind::SchemaScorer => Some((
            openkind_backends::families::schema_scorer::FAMILY_SLUG,
            openkind_backends::families::schema_scorer::PROFILE_ID,
            openkind_backends::families::schema_scorer::BACKBONE_REVISION,
        )),
        EngineKind::RouterScript => Some((
            openkind_backends::families::router_script::FAMILY_SLUG,
            "n/a",
            "n/a",
        )),
        EngineKind::Qwen3Guard => Some((
            openkind_backends::families::qwen3guard::FAMILY_SLUG,
            openkind_backends::families::qwen3guard::PROFILE_ID,
            openkind_backends::families::qwen3guard::BACKBONE_REVISION,
        )),
        EngineKind::Winnow => Some((
            openkind_backends::families::winnow::FAMILY_SLUG,
            openkind_backends::families::winnow::PROFILE_ID,
            openkind_backends::families::winnow::BACKBONE_REVISION,
        )),
        EngineKind::Kev => Some((
            openkind_backends::families::kev::FAMILY_SLUG,
            openkind_backends::families::kev::PROFILE_ID,
            openkind_backends::families::kev::BACKBONE_REVISION,
        )),
        EngineKind::DecoderLogitQwen35 => Some((
            openkind_backends::families::decoder_logit_qwen35::FAMILY_SLUG,
            openkind_backends::families::decoder_logit_qwen35::PROFILE_ID,
            openkind_backends::families::decoder_logit_qwen35::BACKBONE_REVISION,
        )),
        EngineKind::Plumb4b => Some(qwen35_logit_identity(
            &openkind_backends::families::decoder_logit_qwen35::PLUMB_4B,
        )),
        EngineKind::LayaEnglish => Some(laya_identity(
            &openkind_backends::families::laya::LAYA_ENGLISH,
        )),
        EngineKind::LayaMultilingual => Some(laya_identity(
            &openkind_backends::families::laya::LAYA_MULTILINGUAL,
        )),
        EngineKind::LayaTypedDecisions => Some(laya_identity(
            &openkind_backends::families::laya::LAYA_TYPED_DECISIONS,
        )),
        EngineKind::Von => Some((
            openkind_backends::families::von::FAMILY_SLUG,
            openkind_backends::families::von::PROFILE_ID,
            openkind_backends::families::von::VON.backbone_revision,
        )),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaEnglishMlxFp32 => Some(laya_identity(
            &openkind_backends::families::laya::LAYA_ENGLISH,
        )),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaMultilingualMlxFp32 => Some(laya_identity(
            &openkind_backends::families::laya::LAYA_MULTILINGUAL,
        )),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaTypedDecisionsMlxFp32 => Some(laya_identity(
            &openkind_backends::families::laya::LAYA_TYPED_DECISIONS,
        )),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::EncoderInstructLabelMlxFp32 => Some((
            openkind_backends::families::encoder_instruct_label::FAMILY_SLUG,
            openkind_backends::families::encoder_instruct_label::PROFILE_ID,
            openkind_backends::families::encoder_instruct_label::BACKBONE_REVISION,
        )),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::DecoderLogitQwen35MlxFp32 => Some((
            openkind_backends::families::decoder_logit_qwen35::FAMILY_SLUG,
            openkind_backends::families::decoder_logit_qwen35::PROFILE_ID,
            openkind_backends::families::decoder_logit_qwen35::BACKBONE_REVISION,
        )),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::Plumb4bMlxFp32 => Some(qwen35_logit_identity(
            openkind_backends::families::decoder_logit_qwen35::PLUMB_4B,
        )),
        _ => None,
    }
}

/// `(family slug, profile id, backbone revision)` provenance for one pinned
/// decoder-logit-qwen35 family profile.
fn qwen35_logit_identity(
    profile: &openkind_backends::families::decoder_logit_qwen35::Qwen35LogitProfile,
) -> (&'static str, &'static str, &'static str) {
    (
        openkind_backends::families::decoder_logit_qwen35::FAMILY_SLUG,
        profile.profile_id,
        profile.backbone_revision,
    )
}

/// `(family slug, profile id, backbone revision)` provenance for one pinned
/// laya profile.
fn laya_identity(
    profile: &openkind_backends::families::laya::LayaProfile,
) -> (&'static str, &'static str, &'static str) {
    (
        openkind_backends::families::laya::FAMILY_SLUG,
        profile.profile_id,
        profile.backbone_revision,
    )
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
    /// Run the untimed warmup pass before measured requests.
    pub warmup: bool,
    /// Execute two workload rows as A, B, A with identical first and last requests.
    pub history_aba: bool,
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
    /// Family model root (required for surveyed-family engines).
    pub model_root: Option<PathBuf>,
    /// LoRA adapter path (winnow engine).
    pub adapter: Option<PathBuf>,
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
    if is_family_engine(engine) {
        // Surveyed-family engines have a single execution plan: no sweep.
        if !strategies.is_empty() {
            bail!("--strategies does not apply to surveyed-family engines");
        }
        return Ok(());
    }
    if engine != EngineKind::Mock && strategies.is_empty() {
        bail!("native benchmark runs require at least one execution strategy");
    }
    if engine != EngineKind::Mock {
        for (index, strategy) in strategies.iter().enumerate() {
            // Strategy names also identify result files and digest keys.
            // Repeated aliases would overwrite evidence within one run.
            if strategies[..index].contains(strategy) {
                bail!("duplicate execution strategy `{}`", strategy.name());
            }
        }
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
