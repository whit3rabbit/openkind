use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::gen::GeneratedWorkload;
use crate::score::{EngineKind, StrategySpec, DEFAULT_STRATEGIES, STRATEGY_HELP};

/// CLI front end for the offline benchmark harness.
#[derive(Parser, Debug)]
#[command(
    name = "opendecision-bench",
    about = "opendecision — offline scoring and timing benchmarks over JSONL decision workloads"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
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
        /// Write the summary as pretty JSON.
        #[arg(long)]
        pretty: bool,
    },
}

/// Engine choices accepted on the command line.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum EngineArg {
    /// Deterministic in-process mock; smoke runs and CI.
    Mock,
    /// Pinned Qwen3.5 native CPU engine; real scoring and timing.
    Qwen35,
}

impl From<EngineArg> for EngineKind {
    fn from(value: EngineArg) -> Self {
        match value {
            EngineArg::Mock => EngineKind::Mock,
            EngineArg::Qwen35 => EngineKind::Qwen35,
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
        strategies.push(
            StrategySpec::parse(token)
                .map_err(|error| anyhow::anyhow!("{error} (accepted: {STRATEGY_HELP}, auto)"))?,
        );
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
