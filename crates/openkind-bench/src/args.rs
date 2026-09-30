use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::gen::GeneratedWorkload;
use crate::score::{EngineKind, StrategySpec, DEFAULT_STRATEGIES, STRATEGY_HELP};

/// CLI front end for the offline benchmark harness.
#[derive(Parser, Debug)]
#[command(
    name = "openkind-bench",
    about = "openkind — offline scoring and timing benchmarks over JSONL decision workloads"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Subcommands supported by the `openkind-bench` benchmark harness CLI.
#[allow(clippy::large_enum_variant)] // the Score variant legitimately carries the artifact paths
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Compare the fitted scorer with experimental joint-option scoring on labeled Choice rows.
    CompareChoice {
        /// JSONL workload with gold, task, and source_group fields on every row.
        input: PathBuf,
        #[arg(long)]
        bundle_root: PathBuf,
        #[arg(long)]
        checkpoint_root: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, value_enum, default_value_t = ProbeBackendArg::Cpu)]
        backend: ProbeBackendArg,
        #[arg(long, default_value = "bench-output/joint-choice")]
        output_dir: PathBuf,
        /// Hardware attribution, required for quality and latency evidence.
        #[arg(long)]
        host: String,
        /// Commit hash under measurement (record dirty source separately).
        #[arg(long)]
        commit: String,
    },
    /// Generate a seeded deterministic state × criterion workload JSONL.
    GenWorkload {
        /// Number of distinct states.
        #[arg(long, default_value_t = 37)]
        states: usize,
        /// Binary criteria scored per state.
        #[arg(long, default_value_t = 21)]
        criteria: usize,
        /// Seed for deterministic generation.
        #[arg(long, default_value_t = 291_607)]
        seed: u64,
        /// Output JSONL path.
        #[arg(long)]
        output: PathBuf,
    },

    /// Score a workload end to end, timing each request and recording
    /// per-row predictions plus a provenance summary.
    Score {
        /// Workload JSONL path.
        input: PathBuf,
        /// Engine under test.
        #[arg(long, value_enum, default_value_t = EngineArg::Mock)]
        engine: EngineArg,
        /// Output directory for the summary and predictions.
        #[arg(long, default_value = "bench-output")]
        output_dir: PathBuf,
        /// Comma-separated strategy sweep (native engine only).
        #[arg(long)]
        strategies: Option<String>,
        /// Timed repetitions per group.
        #[arg(long, default_value_t = 1)]
        reps: usize,
        /// Disable state grouping (fresh per-row requests).
        #[arg(long)]
        no_group: bool,
        /// Skip the untimed warmup pass. Used by cold and history probes.
        #[arg(long)]
        no_warmup: bool,
        /// Execute the first of two rows again after the second, with the same request ID.
        #[arg(long)]
        history_aba: bool,
        /// Host attribution label recorded in the summary.
        #[arg(long)]
        host: Option<String>,
        /// Commit hash under measurement, recorded in the summary.
        #[arg(long)]
        commit: Option<String>,
        /// Profile bundle root with the fitted head (native engine).
        #[arg(long)]
        bundle_root: Option<PathBuf>,
        /// Pinned Qwen checkpoint root (native engine).
        #[arg(long)]
        checkpoint_root: Option<PathBuf>,
        /// Digest-locked tokenizer JSON (native engine).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Family model root directory (surveyed-family engines).
        #[arg(long)]
        model_root: Option<PathBuf>,
        /// LoRA adapter path (winnow engine).
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Write the summary as pretty JSON.
        #[arg(long)]
        pretty: bool,
    },
}

/// FP32 arithmetic paths supported by the paired readout experiment.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ProbeBackendArg {
    Cpu,
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxFp32,
}

impl From<ProbeBackendArg> for openkind_backends::qwen35::Qwen35Backend {
    fn from(value: ProbeBackendArg) -> Self {
        match value {
            ProbeBackendArg::Cpu => Self::NativeCpu,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            ProbeBackendArg::MlxFp32 => Self::MlxFp32,
        }
    }
}

/// Engine choices accepted on the command line.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum EngineArg {
    /// Deterministic in-process mock; smoke runs and CI.
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
    /// Pinned laya-english engine (English ModernBERT-large decision encoder).
    LayaEnglish,
    /// Pinned laya-multilingual engine (mmBERT-base decision encoder).
    LayaMultilingual,
    /// Pinned laya-typed-decisions engine (fine-tuned ModernBERT-large).
    LayaTypedDecisions,
    /// Winnow learned router over mock siblings; router-cost only.
    Winnow,
    /// Pinned Qwen3.5 MLX FP32 reference-ops engine; requires `--features mlx`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Qwen35MlxFp32,
    /// Pinned Qwen3.5 MLX native-BF16 candidate engine; requires `--features mlx`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Qwen35MlxBf16,
    /// Pinned laya-english MLX FP32 engine; requires `--features mlx`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    LayaEnglishMlxFp32,
    /// Pinned laya-multilingual MLX FP32 engine; requires `--features mlx`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    LayaMultilingualMlxFp32,
    /// Pinned laya-typed-decisions MLX FP32 engine; requires `--features mlx`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    LayaTypedDecisionsMlxFp32,
    /// Pinned encoder-instruct-label MLX FP32 engine; requires `--features mlx`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    EncoderInstructLabelMlxFp32,
    /// Pinned decoder-logit-qwen35 MLX FP32 engine; requires `--features mlx`.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    DecoderLogitQwen35MlxFp32,
}

impl From<EngineArg> for EngineKind {
    fn from(value: EngineArg) -> Self {
        match value {
            EngineArg::Mock => EngineKind::Mock,
            EngineArg::Qwen35 => EngineKind::Qwen35,
            EngineArg::DecoderLetter => EngineKind::DecoderLetter,
            EngineArg::EncoderNli => EngineKind::EncoderNli,
            EngineArg::EncoderInstructLabel => EngineKind::EncoderInstructLabel,
            EngineArg::DecoderLlm => EngineKind::DecoderLlm,
            EngineArg::SchemaScorer => EngineKind::SchemaScorer,
            EngineArg::RouterScript => EngineKind::RouterScript,
            EngineArg::Qwen3Guard => EngineKind::Qwen3Guard,
            EngineArg::Kev => EngineKind::Kev,
            EngineArg::DecoderLogitQwen35 => EngineKind::DecoderLogitQwen35,
            EngineArg::LayaEnglish => EngineKind::LayaEnglish,
            EngineArg::LayaMultilingual => EngineKind::LayaMultilingual,
            EngineArg::LayaTypedDecisions => EngineKind::LayaTypedDecisions,
            EngineArg::Winnow => EngineKind::Winnow,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineArg::Qwen35MlxFp32 => EngineKind::Qwen35MlxFp32,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineArg::Qwen35MlxBf16 => EngineKind::Qwen35MlxBf16,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineArg::LayaEnglishMlxFp32 => EngineKind::LayaEnglishMlxFp32,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineArg::LayaMultilingualMlxFp32 => EngineKind::LayaMultilingualMlxFp32,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineArg::LayaTypedDecisionsMlxFp32 => EngineKind::LayaTypedDecisionsMlxFp32,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineArg::EncoderInstructLabelMlxFp32 => EngineKind::EncoderInstructLabelMlxFp32,
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineArg::DecoderLogitQwen35MlxFp32 => EngineKind::DecoderLogitQwen35MlxFp32,
        }
    }
}

/// Parse the `--strategies` list, defaulting to the full sweep.
///
/// # Errors
/// Returns an error listing the accepted tokens when any token is unknown.
pub fn parse_strategies(spec: Option<&str>) -> anyhow::Result<Vec<StrategySpec>> {
    let Some(spec) = spec else {
        return Ok(DEFAULT_STRATEGIES.to_vec());
    };
    let mut strategies = Vec::new();
    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let strategy = StrategySpec::parse(token)
            .map_err(|error| anyhow::anyhow!("{error} (accepted: {STRATEGY_HELP}, auto)"))?;
        anyhow::ensure!(
            !strategies.contains(&strategy),
            "duplicate execution strategy `{}`",
            strategy.name()
        );
        strategies.push(strategy);
    }
    if strategies.is_empty() {
        anyhow::bail!("--strategies listed no strategies (accepted: {STRATEGY_HELP}, auto)");
    }
    Ok(strategies)
}

/// Result of a `gen-workload` invocation, for command reporting.
pub struct GenWorkloadOutcome {
    /// Where the workload was written.
    pub output: PathBuf,
    /// The generated workload.
    pub workload: GeneratedWorkload,
}
