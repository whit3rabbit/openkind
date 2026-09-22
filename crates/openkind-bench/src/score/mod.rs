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

mod execution;
mod summary;
mod types;

use std::fs;
use std::time::Instant;

use anyhow::{Context, Result};
use openkind_backends::qwen35::{Qwen35DecisionEngine, Qwen35EngineConfig, SchedulerConfig};
use openkind_engine::EngineRegistry;
use serde_json::Value;

pub(crate) use types::validate_strategy_selection;
pub use types::{
    EngineKind, ScoreArgs, ScoreOutcome, StrategySpec, DEFAULT_STRATEGIES, STRATEGY_HELP,
};

use crate::workload;

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
    validate_strategy_selection(args.engine, &args.strategies)?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    let mut strategy_reports = Vec::new();
    let mut predictions_per_strategy = Vec::new();
    let mut parity_clean: Option<bool> = None;
    let mut answers_by_row: Option<Vec<Value>> = None;

    if args.engine == EngineKind::Mock {
        let registry = execution::mock_registry();
        let (report, predictions) = runtime.block_on(execution::run_strategy_pass(
            &registry,
            &workload.rows,
            &groups,
            args,
            execution::StrategyPass {
                label: "mock",
                forced: None,
                warmup: false,
            },
        ))?;
        strategy_reports.push(report);
        predictions_per_strategy.push(predictions);
    } else {
        let backend = types::native_backend(args.engine);
        let bundle_root = args.bundle_root.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--bundle-root is required for the {} engine",
                backend.as_str()
            )
        })?;
        let checkpoint_root = args.checkpoint_root.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--checkpoint-root is required for the {} engine",
                backend.as_str()
            )
        })?;
        let tokenizer_path = args.tokenizer_path.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--tokenizer is required for the {} engine",
                backend.as_str()
            )
        })?;
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
                backend,
                scheduler,
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 250,
            })
            .map_err(|error| anyhow::anyhow!("load {} engine: {error}", backend.as_str()))?;
            let model_load_seconds = load_started.elapsed().as_secs_f64();
            let mut registry = EngineRegistry::new();
            registry.register(types::BENCH_ALIAS, std::sync::Arc::new(engine));

            eprintln!(
                "[bench] engine {} strategy {} starting",
                backend.as_str(),
                spec.name()
            );
            let (report, predictions) = runtime.block_on(execution::run_strategy_pass(
                &registry,
                &workload.rows,
                &groups,
                args,
                execution::StrategyPass {
                    label: spec.name(),
                    forced: Some(spec),
                    warmup: true,
                },
            ))?;
            strategy_reports.push(execution::report_with_load(report, model_load_seconds));
            predictions_per_strategy.push(predictions);

            let answers = execution::answers_of(&predictions_per_strategy);
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

    let summary = summary::build_summary(args, &workload, &groups, &strategy_reports, parity_clean);
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;
    let summary_path = args
        .output_dir
        .join(format!("summary-{}.json", types::engine_slug(args.engine)));
    summary::write_json(&summary_path, &summary, args.pretty)?;
    let mut prediction_paths = Vec::with_capacity(predictions_per_strategy.len());
    for (report, predictions) in strategy_reports.iter().zip(&predictions_per_strategy) {
        let path = args.output_dir.join(format!(
            "predictions-{}-{}.jsonl",
            types::engine_slug(args.engine),
            report["strategy"].as_str().unwrap_or("strategy")
        ));
        fs::write(&path, predictions).with_context(|| format!("write {}", path.display()))?;
        prediction_paths.push(path);
    }
    eprintln!("[bench] summary {}", summary_path.display());
    Ok(ScoreOutcome { summary })
}
