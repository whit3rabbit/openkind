//! Engine execution, warmup, timing passes, and answer collection.

use std::time::Instant;

use anyhow::{Context, Result};
use opendecision_engine::{dispatch, EngineRegistry, MockEngine};
use serde_json::{json, Value};

use super::types::{ScoreArgs, StrategySpec, BENCH_ALIAS};
use crate::workload::{self, WorkloadRow};

pub(crate) struct StrategyPass<'a> {
    pub(crate) label: &'a str,
    pub(crate) forced: Option<StrategySpec>,
    /// Untimed pass over every group before timing (mmap page-in, allocator,
    /// cache state). Warmup is setup, not the measured region.
    pub(crate) warmup: bool,
}

pub(crate) async fn run_strategy_pass(
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

pub(crate) fn answers_of(predictions_per_strategy: &[String]) -> Vec<Value> {
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

pub(crate) fn report_with_load(mut report: Value, model_load_seconds: f64) -> Value {
    if let Some(object) = report.as_object_mut() {
        object.insert("model_load_seconds".into(), json!(model_load_seconds));
    }
    report
}

pub(crate) fn mock_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::new();
    registry.register(BENCH_ALIAS, std::sync::Arc::new(MockEngine::new()));
    registry
}
