//! The `score` subcommand: time a decision workload end to end and emit
//! row-level predictions plus a provenance summary.
//!
//! Timing scope follows `docs/BENCHMARKS.md`: the measured region covers
//! request construction, validation, engine dispatch, and answer extraction.
//! Model loading and result-file writes are excluded; model-load wall time is
//! reported separately per strategy. Runs are warm-process.
//!
//! Execution paths mirror the prior-art systems benchmark mapping: fresh
//! per-row requests (`--no-group`) correspond to repeated-full direct
//! scoring; state-grouped requests let the shared-state strategies
//! (`nested_sequential`, `nested_batched`, and the measured scheduler) pay
//! the root prefill once. On the native engine, strategies other than
//! `choose_strategy` are forced through the scheduler's diagnostic override;
//! cross-strategy answer equality is asserted per workload.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use opendecision_backends::qwen35::{
    ExecutionStrategy, Qwen35DecisionEngine, Qwen35EngineConfig, SchedulerConfig, PROFILE_ID,
};
use opendecision_engine::{dispatch, EngineRegistry, MockEngine};
use opendecision_runtime::peak_resident_bytes;
use serde_json::{json, Value};

use crate::workload::{self, Workload, WorkloadRow};

/// Model alias used for every bench request inside the run-local registry.
const BENCH_ALIAS: &str = "bench";

/// The `engine` choice for `score`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// In-process deterministic mock; plumbing and summary smoke.
    Mock,
    /// Pinned Qwen3.5 native CPU engine; real scoring and timing.
    Qwen35,
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

    fn name(self) -> &'static str {
        match self {
            Self::ChooseStrategy => "choose_strategy",
            Self::Forced(ExecutionStrategy::RepeatedFull) => "repeated_full",
            Self::Forced(ExecutionStrategy::NestedSequential) => "nested_sequential",
            Self::Forced(ExecutionStrategy::NestedBatched) => "nested_batched",
        }
    }

    fn scheduler_override(self) -> Option<ExecutionStrategy> {
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

/// Run-level output of [`run_score`].
#[derive(Debug, Clone)]
pub struct ScoreOutcome {
    /// The summary document (also written to `summary-<engine>.json`).
    pub summary: Value,
}

/// Execute one scoring run.
///
/// # Errors
/// Returns an error for workload, artifact, or I/O failures. Timed engine
/// errors surface as errors too: a benchmark run that cannot score is not a
/// benchmark result.
pub fn run_score(args: &ScoreArgs) -> Result<ScoreOutcome> {
    let workload = workload::load_workload(&args.input)?;
    eprintln!("[bench] workload {}: {}", args.input.display(), workload);
    let groups = if args.group {
        workload::state_groups(&workload.rows)?
    } else {
        (0..workload.rows.len()).map(|index| vec![index]).collect()
    };
    eprintln!(
        "[bench] {} groups over {} rows (grouping={})",
        groups.len(),
        workload.rows.len(),
        args.group
    );
    anyhow::ensure!(args.reps >= 1, "reps must be at least 1");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    let mut strategy_reports = Vec::new();
    let mut predictions_per_strategy = Vec::new();
    let mut parity_clean: Option<bool> = None;
    let mut answers_by_row: Option<Vec<Value>> = None;

    match args.engine {
        EngineKind::Mock => {
            let registry = mock_registry();
            let (report, predictions) = runtime.block_on(run_strategy_pass(
                &registry,
                &workload.rows,
                &groups,
                args,
                StrategyPass {
                    label: "mock",
                    forced: None,
                    warmup: false,
                },
            ))?;
            strategy_reports.push(report);
            predictions_per_strategy.push(predictions);
        }
        EngineKind::Qwen35 => {
            let bundle_root = args.bundle_root.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--bundle-root is required for the qwen35 engine")
            })?;
            let checkpoint_root = args.checkpoint_root.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--checkpoint-root is required for the qwen35 engine")
            })?;
            let tokenizer_path = args
                .tokenizer_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("--tokenizer is required for the qwen35 engine"))?;
            for &spec in &args.strategies {
                let scheduler = SchedulerConfig::for_pinned_profile(
                    SchedulerConfig::LOWEST_MEASURED_SHARED_SAVINGS_RATIO,
                    None,
                )
                .with_forced_strategy(spec.scheduler_override());
                let load_started = Instant::now();
                let engine = Qwen35DecisionEngine::load(Qwen35EngineConfig {
                    bundle_root: bundle_root.clone(),
                    checkpoint_root: checkpoint_root.clone(),
                    tokenizer_path: tokenizer_path.clone(),
                    scheduler,
                    max_concurrent_requests: 1,
                    max_queued_requests: 0,
                    retry_after_ms: 250,
                })
                .map_err(|error| anyhow::anyhow!("load native engine: {error}"))?;
                let model_load_seconds = load_started.elapsed().as_secs_f64();
                let mut registry = EngineRegistry::new();
                registry.register(BENCH_ALIAS, std::sync::Arc::new(engine));

                eprintln!("[bench] strategy {} starting", spec.name());
                let (report, predictions) = runtime.block_on(run_strategy_pass(
                    &registry,
                    &workload.rows,
                    &groups,
                    args,
                    StrategyPass {
                        label: spec.name(),
                        forced: Some(spec),
                        warmup: true,
                    },
                ))?;
                strategy_reports.push(report_with_load(report, model_load_seconds));
                predictions_per_strategy.push(predictions);

                let answers = answers_of(&predictions_per_strategy);
                if let Some(previous) = &answers_by_row {
                    if previous != &answers && parity_clean != Some(false) {
                        eprintln!(
                            "[bench] strategy {}: answer parity violation against earlier \
                             strategies",
                            spec.name()
                        );
                        parity_clean = Some(false);
                    }
                } else {
                    answers_by_row = Some(answers);
                }
                if parity_clean.is_none() {
                    parity_clean = Some(true);
                }
            }
        }
    }

    let summary = build_summary(args, &workload, &groups, &strategy_reports, parity_clean);
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;
    let summary_path = args
        .output_dir
        .join(format!("summary-{}.json", engine_slug(args.engine)));
    write_json(&summary_path, &summary, args.pretty)?;
    let mut prediction_paths = Vec::with_capacity(predictions_per_strategy.len());
    for (report, predictions) in strategy_reports.iter().zip(&predictions_per_strategy) {
        let path = args.output_dir.join(format!(
            "predictions-{}-{}.jsonl",
            engine_slug(args.engine),
            report["strategy"].as_str().unwrap_or("strategy")
        ));
        fs::write(&path, predictions).with_context(|| format!("write {}", path.display()))?;
        prediction_paths.push(path);
    }
    eprintln!("[bench] summary {}", summary_path.display());
    Ok(ScoreOutcome { summary })
}

struct StrategyPass<'a> {
    label: &'a str,
    forced: Option<StrategySpec>,
    /// Untimed pass over every group before timing (mmap page-in, allocator,
    /// cache state). Warmup is setup, not the measured region.
    warmup: bool,
}

async fn run_strategy_pass(
    registry: &EngineRegistry,
    rows: &[WorkloadRow],
    groups: &[Vec<usize>],
    args: &ScoreArgs,
    pass: StrategyPass<'_>,
) -> Result<(Value, String)> {
    let mut rep_totals = Vec::with_capacity(args.reps);
    let mut input_tokens_total = 0_u64;
    let mut last_request_latencies = vec![0.0_f64; rows.len()];
    let mut last_answers: Vec<Option<Value>> = vec![None; rows.len()];
    let mut last_group_sizes = vec![0_usize; rows.len()];

    if pass.warmup {
        for group in groups {
            let request = workload::build_request(BENCH_ALIAS, rows, group)?;
            dispatch(request, registry)
                .await
                .map_err(|error| anyhow::anyhow!("warmup dispatch failed: {error}"))?;
        }
        eprintln!("[bench] strategy {} warmup complete", pass.label);
    }

    for rep in 0..args.reps {
        let rep_started = Instant::now();
        for group in groups {
            let request = workload::build_request(BENCH_ALIAS, rows, group)?;
            let started = Instant::now();
            let response = dispatch(request, registry)
                .await
                .map_err(|error| anyhow::anyhow!("dispatch failed: {error}"))?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            input_tokens_total += u64::from(response.usage.input_tokens);
            if rep + 1 == args.reps {
                for &index in group {
                    let answer = response.answers.get(&rows[index].id).ok_or_else(|| {
                        anyhow::anyhow!("missing answer for `{}`", rows[index].id)
                    })?;
                    last_answers[index] =
                        Some(serde_json::to_value(answer).context("serialize answer")?);
                    last_request_latencies[index] = elapsed_ms;
                    last_group_sizes[index] = group.len();
                }
            }
        }
        rep_totals.push(rep_started.elapsed().as_secs_f64());
        eprintln!(
            "[bench] strategy {} rep {rep}: {:.3} s",
            pass.label, rep_totals[rep]
        );
    }

    let decisions = rows.len();
    let mut sorted_totals = rep_totals.clone();
    sorted_totals.sort_by(|left, right| left.total_cmp(right));
    let p50 = sorted_totals[sorted_totals.len() / 2];
    let p95 = sorted_totals[(0.95 * sorted_totals.len() as f64).ceil() as usize - 1];
    let forced = pass.forced.map(|spec| match spec {
        StrategySpec::ChooseStrategy => json!({ "strategy": "choose_strategy" }),
        StrategySpec::Forced(strategy) => json!({ "strategy": strategy.as_str() }),
    });
    let report = json!({
        "strategy": pass.label,
        "forced": forced,
        "rows": decisions,
        "groups": groups.len(),
        "reps": args.reps,
        "samples_seconds": sorted_totals,
        "p50_seconds": p50,
        "p95_seconds": p95,
        "decisions_per_second": decisions as f64 / p50,
        "input_tokens_total": input_tokens_total,
    });

    let mut predictions = String::with_capacity(decisions * 256);
    for (index, row) in rows.iter().enumerate() {
        let line = json!({
            "id": row.id,
            "strategy": pass.label,
            "group_size": last_group_sizes[index],
            "request_latency_ms": last_request_latencies[index],
            "answer": last_answers[index],
        });
        predictions.push_str(&serde_json::to_string(&line).context("serialize prediction")?);
        predictions.push('\n');
    }
    Ok((report, predictions))
}

fn answers_of(predictions_per_strategy: &[String]) -> Vec<Value> {
    predictions_per_strategy
        .last()
        .map(|predictions| {
            predictions
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter_map(|line| line.get("answer").cloned())
                .collect()
        })
        .unwrap_or_default()
}

fn report_with_load(mut report: Value, model_load_seconds: f64) -> Value {
    if let Some(object) = report.as_object_mut() {
        object.insert("model_load_seconds".into(), json!(model_load_seconds));
    }
    report
}

fn mock_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::new();
    registry.register(BENCH_ALIAS, std::sync::Arc::new(MockEngine::new()));
    registry
}

fn build_summary(
    args: &ScoreArgs,
    fixture: &Workload,
    groups: &[Vec<usize>],
    strategy_reports: &[Value],
    parity_clean: Option<bool>,
) -> Value {
    let (engine_id, profile, model_revision, bundle_version) = match args.engine {
        EngineKind::Mock => ("mock", Value::Null, Value::Null, Value::Null),
        EngineKind::Qwen35 => (
            "qwen35-native-cpu",
            json!(PROFILE_ID),
            model_revision(args),
            bundle_version(args),
        ),
    };
    json!({
        "schema": "opendecision-bench/v1",
        "engine": engine_id,
        "profile_id": profile,
        "model_revision": model_revision,
        "bundle_version": bundle_version,
        "fixture": {
            "path": args.input.display().to_string(),
            "sha256": fixture.sha256,
            "rows": fixture.rows.len(),
            "groups": groups.len(),
        },
        "grouping": if args.group { "per-state" } else { "per-row" },
        "reps": args.reps,
        "host": args.host.clone().unwrap_or_else(default_host),
        "commit": args.commit.clone(),
        "timing_scope": "request construction, validation, dispatch, and answer extraction; \
                         excludes model load (reported per strategy), result writes, and the \
                         untimed warmup pass; warm process",
        "peak_resident_bytes": peak_resident_bytes().ok(),
        "cross_strategy_answer_parity_clean": parity_clean,
        "strategies": strategy_reports,
        "notes": {
            "percentiles": "p50 is the median rep total and p95 is the ceil(0.95*n)-1 sample; \
                            small rep counts are reported alongside as samples_seconds",
            "request_latency_ms": "latency is per request; grouped rows share their \
                                   request's latency and are not independent timings",
        },
    })
}

fn model_revision(args: &ScoreArgs) -> Value {
    manifest_field(args, &["base", "revision"])
}

fn bundle_version(args: &ScoreArgs) -> Value {
    manifest_field(args, &["version"])
}

fn manifest_field(args: &ScoreArgs, path: &[&str]) -> Value {
    let Some(bundle_root) = &args.bundle_root else {
        return Value::Null;
    };
    let Ok(raw) = fs::read(bundle_root.join("BUNDLE_MANIFEST.json")) else {
        return Value::Null;
    };
    let mut current = serde_json::from_slice::<Value>(&raw).unwrap_or(Value::Null);
    for key in path {
        current = current.get(key).cloned().unwrap_or(Value::Null);
    }
    current
}

fn default_host() -> String {
    format!(
        "{}-{} (unattributed; pass --host to attribute)",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

fn engine_slug(engine: EngineKind) -> &'static str {
    match engine {
        EngineKind::Mock => "mock",
        EngineKind::Qwen35 => "qwen35",
    }
}

fn write_json(path: &Path, value: &Value, pretty: bool) -> Result<()> {
    let body = if pretty {
        serde_json::to_string_pretty(value).context("serialize summary")?
    } else {
        serde_json::to_string(value).context("serialize summary")?
    };
    fs::write(path, body).with_context(|| format!("write {}", path.display()))?;
    Ok(())
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
