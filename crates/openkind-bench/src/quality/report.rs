use std::collections::{BTreeMap, BTreeSet};

use openkind_backends::qwen35::experimental::{
    Qwen35ScoringProbe, CATALOGUE_RENDERER_ID, JOINT_RENDERER_ID,
};
use openkind_backends::qwen35::{
    BACKBONE_ID, BACKBONE_REVISION, CALIBRATION_TEMPERATURE, PROFILE_ID, TOKENIZER_JSON_SHA256,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};

use super::{metrics, CompareArgs, Record, COMPARISON_METHODS, METHOD_DEFINITIONS};

pub(super) fn summary(
    args: &CompareArgs,
    probe: &Qwen35ScoringProbe,
    digest: String,
    load_seconds: f64,
    records: &[Record],
) -> Value {
    let all: Vec<_> = records.iter().collect();
    let overall: BTreeMap<_, _> = COMPARISON_METHODS
        .iter()
        .map(|method| (*method, metrics::metrics(&all, method)))
        .collect();
    let tasks: BTreeSet<_> = records.iter().map(|record| record.task.as_str()).collect();
    let per_task: BTreeMap<_, BTreeMap<_, _>> = tasks
        .into_iter()
        .map(|task| {
            let subset: Vec<_> = records
                .iter()
                .filter(|record| record.task == task)
                .collect();
            (
                task,
                COMPARISON_METHODS
                    .iter()
                    .map(|method| (*method, metrics::metrics(&subset, method)))
                    .collect(),
            )
        })
        .collect();
    let paired_changes = |first: &str, second: &str| -> Value {
        let flips = records
            .iter()
            .filter(|record| {
                metrics::prediction(&record.methods[first].score.probabilities, false).0
                    != metrics::prediction(&record.methods[second].score.probabilities, false).0
            })
            .count();
        let max_shift = records
            .iter()
            .flat_map(|record| {
                record.methods[first]
                    .score
                    .probabilities
                    .iter()
                    .map(|(label, p)| (p - record.methods[second].score.probabilities[label]).abs())
            })
            .fold(0.0_f64, f64::max);
        json!({"selection_flips": flips, "count": records.len(), "max_probability_shift": max_shift})
    };
    let comparisons: BTreeMap<_, _> = COMPARISON_METHODS[1..]
        .iter()
        .map(|method| (*method, paired(records, method)))
        .collect();
    let method_definitions: BTreeMap<_, _> = METHOD_DEFINITIONS
        .iter()
        .map(|(name, kind, members)| (*name, json!({"kind": kind, "ensemble_members": members})))
        .collect();
    json!({"schema": "openkind-choice-comparison/v1", "host": args.host, "commit": args.commit,
    "backbone": BACKBONE_ID, "backbone_revision": BACKBONE_REVISION, "tokenizer_sha256": TOKENIZER_JSON_SHA256,
    "baseline_profile": PROFILE_ID, "baseline_temperature": CALIBRATION_TEMPERATURE,
    "joint_renderer": JOINT_RENDERER_ID, "joint_temperature": 1.0,
    "catalogue_renderer": CATALOGUE_RENDERER_ID,
    "backend": probe.backend().as_str(), "arithmetic_id": probe.arithmetic_id(),
    "workload_sha256": digest, "load_seconds": load_seconds,
    "timing_scope": "warm full-forward scorer only; excludes load, preparation, warmup, admission, wire mapping and writes; ensemble costs equal the sum of member passes; execution order rotates across rows after one warmup",
    "evidence_scope": "paired diagnostic quality on supplied labeled Choice rows; the catalogue arm keeps the fitted head and frozen temperature but changes the prompt, the joint arms change prompt and readout together; no temperature, offset, or threshold fit; no production promotion",
    "method_definitions": method_definitions,
    "nll_probability_floor": 1e-15, "ece_bins": 10,
    "overall": overall, "per_task": per_task, "paired_delta_from_independent": comparisons,
    "order_sensitivity": {
        "coupled_order_and_codes": paired_changes("joint_forward", "joint_reverse"),
        "position_factor_fixed_codes": paired_changes("joint_forward", "joint_text_rotate"),
        "code_factor_fixed_positions": paired_changes("joint_forward", "joint_code_rotate"),
    }})
}

pub(super) fn paired(records: &[Record], method: &str) -> Value {
    paired_vs(records, "independent_fitted", method)
}

pub(super) fn paired_vs(records: &[Record], baseline_method: &str, method: &str) -> Value {
    // Resample whole source groups together, preserving paired predictions.
    let mut groups: BTreeMap<&str, ([f64; 3], usize)> = BTreeMap::new();
    for record in records {
        let baseline = metrics::losses(record, baseline_method);
        let candidate = metrics::losses(record, method);
        let group = groups.entry(&record.source_group).or_default();
        for index in 0..3 {
            group.0[index] += candidate[index] - baseline[index];
        }
        group.1 += 1;
    }
    let groups: Vec<_> = groups.values().collect();
    let point: Vec<_> = (0..3)
        .map(|index| groups.iter().map(|group| group.0[index]).sum::<f64>() / records.len() as f64)
        .collect();
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    let mut rng = StdRng::seed_from_u64(29_160_717);
    for _ in 0..2000 {
        let mut sum = [0.0; 3];
        let mut count = 0;
        for _ in 0..groups.len() {
            let group = groups[rng.random_range(0..groups.len())];
            for (index, value) in sum.iter_mut().enumerate() {
                *value += group.0[index];
            }
            count += group.1;
        }
        for index in 0..3 {
            samples[index].push(sum[index] / count as f64);
        }
    }
    let intervals: BTreeMap<_, _> = ["accuracy", "nll", "brier"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            samples[index].sort_by(f64::total_cmp);
            (
                name,
                json!({"delta": point[index], "ci_95": [samples[index][50], samples[index][1949]]}),
            )
        })
        .collect();
    json!({"candidate_minus_baseline": intervals, "source_groups": groups.len(), "bootstrap_replicates": 2000, "bootstrap_seed": 29_160_717})
}
