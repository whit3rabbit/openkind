//! Engine execution, warmup, timing passes, and answer collection.

use std::time::Instant;

use anyhow::{Context, Result};
use openkind_engine::{dispatch, EngineRegistry, MockEngine};
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
    let backend_id = registry
        .get(BENCH_ALIAS)
        .context("benchmark engine is not registered")?
        .backend_id()
        .to_owned();
    let mut rep_totals = Vec::new();
    rep_totals
        .try_reserve(args.reps)
        .context("reserve repetition samples")?;
    let mut input_tokens_total = 0_u64;
    let mut last_request_latencies = vec![0.0_f64; rows.len()];
    let mut last_answers: Vec<Option<Value>> = vec![None; rows.len()];
    let mut last_group_sizes = vec![0_usize; rows.len()];
    let mut history_predictions = String::new();

    if pass.warmup {
        for group in groups {
            let request = workload::build_request(BENCH_ALIAS, rows, group)?;
            dispatch(request, registry)
                .await
                .map_err(|error| anyhow::anyhow!("warmup dispatch failed: {error}"))?;
        }
        eprintln!("[bench] strategy {} warmup complete", pass.label);
    }

    // CPU accounting starts after warmup so the measured region excludes
    // setup work, matching the timing scope.
    let cpu_started = openkind_runtime::cpu_time_seconds().ok();

    for rep in 0..args.reps {
        let rep_started = Instant::now();
        for (sequence_index, group) in groups.iter().enumerate() {
            let started = Instant::now();
            let request = workload::build_request(BENCH_ALIAS, rows, group)?;
            let response = dispatch(request, registry)
                .await
                .map_err(|error| anyhow::anyhow!("dispatch failed: {error}"))?;
            // Extract every repetition's answers within the advertised
            // timing scope, retaining predictions only for the final rep.
            let answers: Vec<Value> = group
                .iter()
                .map(|&index| {
                    let answer = response.answers.get(&rows[index].id).ok_or_else(|| {
                        anyhow::anyhow!("missing answer for `{}`", rows[index].id)
                    })?;
                    serde_json::to_value(answer).context("serialize answer")
                })
                .collect::<Result<_>>()?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            input_tokens_total = input_tokens_total
                .checked_add(u64::from(response.usage.input_tokens))
                .context("total input token count exceeds the report range")?;
            if rep + 1 == args.reps {
                for (&index, answer) in group.iter().zip(answers) {
                    last_answers[index] = Some(answer.clone());
                    last_request_latencies[index] = elapsed_ms;
                    last_group_sizes[index] = group.len();
                    if args.history_aba {
                        let line = json!({
                            "id": rows[index].id,
                            "strategy": pass.label,
                            "sequence_index": sequence_index,
                            "group_size": group.len(),
                            "request_latency_ms": elapsed_ms,
                            "answer": answer,
                        });
                        history_predictions.push_str(
                            &serde_json::to_string(&line)
                                .context("serialize history prediction")?,
                        );
                        history_predictions.push('\n');
                    }
                }
            }
        }
        rep_totals.push(rep_started.elapsed().as_secs_f64());
        eprintln!(
            "[bench] strategy {} rep {rep}: {:.3} s",
            pass.label, rep_totals[rep]
        );
    }

    let decisions = if args.history_aba {
        groups.len()
    } else {
        rows.len()
    };
    let mut sorted_totals = rep_totals.clone();
    sorted_totals.sort_by(|left, right| left.total_cmp(right));
    let p50 = median(&sorted_totals);
    let p95 = sorted_totals[(0.95 * sorted_totals.len() as f64).ceil() as usize - 1];
    let timed_wall_seconds: f64 = rep_totals.iter().sum();
    let cpu_time_seconds = cpu_started
        .zip(openkind_runtime::cpu_time_seconds().ok())
        .map(|(started, ended)| (ended - started).max(0.0));
    let avg_cpu_percent = cpu_time_seconds
        .filter(|_| timed_wall_seconds > 0.0)
        .map(|cpu| 100.0 * cpu / timed_wall_seconds);
    let forced = pass.forced.map(|spec| match spec {
        StrategySpec::ChooseStrategy => json!({ "strategy": "choose_strategy" }),
        StrategySpec::Forced(strategy) => json!({ "strategy": strategy.as_str() }),
    });
    let report = json!({
        "strategy": pass.label,
        "backend_id": backend_id,
        "forced": forced,
        "rows": decisions,
        "groups": groups.len(),
        "reps": args.reps,
        "samples_seconds": sorted_totals,
        "p50_seconds": p50,
        "p95_seconds": p95,
        "decisions_per_second": decisions as f64 / p50,
        "input_tokens_total": input_tokens_total,
        "input_tokens_per_second": input_tokens_total as f64 / timed_wall_seconds,
        "cpu_time_seconds": cpu_time_seconds,
        "avg_cpu_percent": avg_cpu_percent,
    });

    if args.history_aba {
        return Ok((report, history_predictions));
    }
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

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        let lower = sorted[middle - 1];
        lower + (sorted[middle] - lower) / 2.0
    } else {
        sorted[middle]
    }
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

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_averages_the_middle_pair_for_even_samples() {
        assert_eq!(median(&[1.0, 9.0]), 5.0);
        assert_eq!(median(&[1.0, 5.0, 9.0]), 5.0);
    }
}
