//! Phase 3.8 measurement harness: adaptive-scheduler crossover and
//! Q-amortization on the named M4 Max host.
//!
//! For each workload this harness executes all three parity-proven strategies
//! (`repeated_full`, `nested_sequential`, `nested_batched`), asserts that the
//! candidate features are identical across strategies, and records wall-time
//! samples plus cost accounting: forward calls, staged tokens, retained
//! branch-state bytes, `T(Q)/T(1)`, marginal ms/question, questions/s, and
//! the prefix-reuse fraction.
//!
//! Workloads: the Phase 3B semantic fixture requests (98-token root with
//! Q=3/Q=1 and the 345-token root with Q=1) plus two synthetic mechanics
//! cells (short L=98/Q=4/K=4 and long L=256/Q=2/K=2) with deterministic
//! pseudo-token content. All runs are warm-process; cold-start and
//! memory-pressure cells belong to the later lifecycle work.
//!
//! Usage: `qwen35_scheduler_bench <checkpoint-root> <phase3b-reference-root>
//! [reps]` — the default is 3 repetitions for semantic workloads and 2 for
//! synthetic ones. With 3 samples, p50 is the median and p95 is the maximum;
//! sample counts are reported alongside. The run executes the real checkpoint
//! on CPU for roughly 35-45 minutes at the default settings and is not part
//! of `cargo test`.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use opendecision_backends::qwen35::{
    run_strategy, BackboneReference, ExecutionStrategy, NestedQuestion, Qwen35Backbone,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TokenFixtures {
    records: Vec<TokenRecord>,
}

#[derive(Debug, Deserialize)]
struct TokenRecord {
    fixture_case: usize,
    #[allow(dead_code)]
    question_id: String,
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<Vec<u32>>,
}

struct Workload {
    name: String,
    root_ids: Vec<u32>,
    question_ids: Vec<Vec<u32>>,
    suffix_ids: Vec<Vec<Vec<u32>>>,
    questions: usize,
    reps: usize,
    synthetic: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = required_path(&mut arguments, "checkpoint-root")?;
    let reference_root = required_path(&mut arguments, "phase3b-reference-root")?;
    let extra_reps: Option<usize> = arguments
        .next()
        .and_then(|value| value.to_string_lossy().into_owned().parse().ok());
    if arguments.next().is_some() {
        return Err(
            "qwen35_scheduler_bench accepts two paths and an optional reps override".into(),
        );
    }

    let backbone = Qwen35Backbone::load(&checkpoint_root)?;
    let _ = BackboneReference::load(&reference_root)?;
    let token_fixtures: TokenFixtures = read_json(&reference_root.join("TOKEN_FIXTURES.json"))?;

    // Semantic workloads from the frozen Phase 3B token fixtures.
    let mut grouped: BTreeMap<usize, Vec<&TokenRecord>> = BTreeMap::new();
    for record in &token_fixtures.records {
        grouped.entry(record.fixture_case).or_default().push(record);
    }
    let case0 = &grouped[&0];
    let case1 = &grouped[&1];

    let mut workloads = vec![
        semantic_workload("case0_q3", case0, extra_reps.unwrap_or(3)),
        semantic_workload("case0_q1", &case0[..1], extra_reps.unwrap_or(3)),
        semantic_workload("case1_q1", case1, extra_reps.unwrap_or(3)),
    ];
    workloads.push(synthetic_workload(
        "synthetic_l98_q4_k4",
        98,
        4,
        4,
        12,
        extra_reps.unwrap_or(2),
    ));
    workloads.push(synthetic_workload(
        "synthetic_l256_q2_k2",
        256,
        2,
        2,
        16,
        extra_reps.unwrap_or(2),
    ));

    let mut workload_results = Vec::new();
    for workload in &workloads {
        eprintln!("[bench] workload {} starting", workload.name);
        let plans = build_plans(workload);
        // Warm-up pass, untimed: mmap pages, allocator, cache state.
        for strategy in ExecutionStrategy::all() {
            run_strategy(&backbone, strategy, &workload.root_ids, &plans)?;
        }

        let mut per_strategy = Vec::new();
        let mut oracle_features: Option<Vec<Vec<Vec<f32>>>> = None;
        let mut parity_clean = true;
        for strategy in ExecutionStrategy::all() {
            let mut samples = Vec::with_capacity(workload.reps);
            let mut output = None;
            for rep in 0..workload.reps {
                let started = Instant::now();
                let result = run_strategy(&backbone, strategy, &workload.root_ids, &plans)?;
                let elapsed = started.elapsed();
                if let Some(previous) = &oracle_features {
                    if previous != result.question_features() {
                        parity_clean = false;
                        eprintln!(
                            "[bench] workload {} strategy {} rep {rep}: feature parity violation",
                            workload.name,
                            strategy.as_str()
                        );
                    }
                } else {
                    oracle_features = Some(result.question_features().to_vec());
                }
                samples.push(elapsed);
                output = Some(result);
                eprintln!(
                    "[bench] workload {} strategy {} rep {rep}: {:.3} s",
                    workload.name,
                    strategy.as_str(),
                    elapsed.as_secs_f64()
                );
            }
            let output = output.expect("at least one rep");
            per_strategy.push(strategy_metrics(strategy, &samples, &output));
        }
        eprintln!("[bench] workload {} complete", workload.name);

        workload_results.push(serde_json::json!({
            "workload": workload.name,
            "root_tokens": workload.root_ids.len(),
            "questions": workload.questions,
            "candidates_per_question": workload.suffix_ids.iter().map(Vec::len).collect::<Vec<_>>(),
            "synthetic": workload.synthetic,
            "feature_parity_across_strategies": parity_clean,
            "strategies": per_strategy,
        }));
    }

    // Q-amortization from the paired case-0 workloads (same root, Q=3 vs Q=1).
    let q3 = strategy_metrics_by_name(&workload_results, "case0_q3");
    let q1 = strategy_metrics_by_name(&workload_results, "case0_q1");
    let amortization: Vec<_> = ExecutionStrategy::all()
        .iter()
        .map(|strategy| {
            let name = strategy.as_str();
            let t_q = median_seconds(q3.get(name).map(|m| m["p50_seconds"].as_f64()));
            let t_one = median_seconds(q1.get(name).map(|m| m["p50_seconds"].as_f64()));
            let (t_q, t_one) = match (t_q, t_one) {
                (Some(t_q), Some(t_one)) if t_one > 0.0 => (t_q, t_one),
                _ => {
                    return serde_json::json!({
                        "strategy": name,
                        "error": "missing samples",
                    })
                }
            };
            let ratio = t_q / t_one;
            serde_json::json!({
                "strategy": name,
                "t_q_seconds": t_q,
                "t_1_seconds": t_one,
                "t_q_over_t_1": ratio,
                "marginal_ms_per_question": (t_q - t_one) * 1000.0 / 2.0,
                "questions_per_second_at_q3": 3.0 / t_q,
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "opendecision-qwen35-scheduler-bench/v1",
            "profile_id": opendecision_backends::qwen35::PROFILE_ID,
            "host": "Mac16,5 Apple M4 Max 36 GiB (named Mac), candle-cpu-fp32, warm process",
            "workloads": workload_results,
            "q_amortization_case0_q3_vs_q1": amortization,
            "notes": {
                "sample_counts": "small; p50 is the median sample and p95 is the maximum of the reported samples",
                "oracle": "feature parity across strategies asserted per workload per repetition",
                "open_cells": "cold-start, cache-warmth, memory-pressure and suffix-bucket packing cells remain later work",
            },
        }))?
    );
    Ok(())
}

fn build_plans(workload: &Workload) -> Vec<NestedQuestion<'static>> {
    let mut plans = Vec::with_capacity(workload.question_ids.len());
    for (question_ids, suffix_lists) in workload.question_ids.iter().zip(&workload.suffix_ids) {
        let leaked_question: &'static [u32] = Box::leak(question_ids.clone().into_boxed_slice());
        let leaked_suffixes: Vec<&'static [u32]> = suffix_lists
            .iter()
            .map(|suffix| Box::leak(suffix.clone().into_boxed_slice()) as &'static [u32])
            .collect();
        plans.push(NestedQuestion {
            question_ids: leaked_question,
            candidate_suffix_ids: Box::leak(leaked_suffixes.into_boxed_slice())
                as &'static [&'static [u32]],
        });
    }
    plans
}

fn semantic_workload(name: &str, records: &[&TokenRecord], reps: usize) -> Workload {
    Workload {
        name: name.to_owned(),
        root_ids: records[0].root_ids.clone(),
        question_ids: records
            .iter()
            .map(|record| record.question_ids.clone())
            .collect(),
        suffix_ids: records
            .iter()
            .map(|record| record.candidate_suffix_ids.clone())
            .collect(),
        questions: records.len(),
        reps,
        synthetic: false,
    }
}

fn synthetic_workload(
    name: &str,
    root_tokens: usize,
    questions: usize,
    candidates: usize,
    suffix_tokens: usize,
    reps: usize,
) -> Workload {
    let token = |index: usize| ((index * 7_919 + 13) % 248_320) as u32;
    let root_ids: Vec<u32> = (0..root_tokens).map(&token).collect();
    let question_ids: Vec<Vec<u32>> = (0..questions)
        .map(|question_index| {
            (0..10)
                .map(|offset| token(root_tokens + question_index * 31 + offset))
                .collect()
        })
        .collect();
    let suffix_ids: Vec<Vec<Vec<u32>>> = (0..questions)
        .map(|question_index| {
            (0..candidates)
                .map(|candidate_index| {
                    (0..suffix_tokens)
                        .map(|offset| {
                            token(
                                root_tokens
                                    + questions * 31
                                    + question_index * 17
                                    + candidate_index * 13
                                    + offset,
                            )
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    Workload {
        name: name.to_owned(),
        root_ids,
        question_ids,
        suffix_ids,
        questions,
        reps,
        synthetic: true,
    }
}

fn strategy_metrics(
    strategy: ExecutionStrategy,
    samples: &[Duration],
    output: &opendecision_backends::qwen35::StrategyOutput<
        opendecision_backends::qwen35::BackboneState,
    >,
) -> serde_json::Value {
    let mut seconds: Vec<f64> = samples.iter().map(|sample| sample.as_secs_f64()).collect();
    seconds.sort_by(|left, right| left.total_cmp(right));
    let p50 = seconds[seconds.len() / 2];
    let p95 = seconds[(0.95 * seconds.len() as f64).ceil() as usize - 1];
    let total_tokens = output.staged_tokens();
    serde_json::json!({
        "strategy": strategy.as_str(),
        "samples_seconds": seconds,
        "p50_seconds": p50,
        "p95_seconds": p95,
        "forward_calls": output.forward_calls(),
        "staged_tokens": total_tokens,
        "retained_state_bytes": output.retained_state_bytes(),
        "tokens_per_second_p50": total_tokens as f64 / p50,
    })
}

fn strategy_metrics_by_name(
    workload_results: &[serde_json::Value],
    workload: &str,
) -> BTreeMap<String, serde_json::Value> {
    let mut map = BTreeMap::new();
    for case in workload_results {
        if case["workload"].as_str() == Some(workload) {
            if let Some(strategies) = case["strategies"].as_array() {
                for entry in strategies {
                    if let Some(name) = entry["strategy"].as_str() {
                        map.insert(name.to_owned(), entry.clone());
                    }
                }
            }
        }
    }
    map
}

fn median_seconds(value: Option<Option<f64>>) -> Option<f64> {
    value.and_then(std::convert::identity)
}

fn required_path(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    name: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing required {name}").into())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
