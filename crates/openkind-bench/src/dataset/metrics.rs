//! Accuracy metrics for dataset evaluation.
//!
//! Extends the existing quality-report quantities (accuracy, per-class
//! recall, NLL, Brier, fixed 10-bin ECE, fixed risk/coverage points) with
//! the statistics the public-dataset protocol requires and the repo lacked:
//! AUROC for Noul answers, Spearman/Pearson correlation for Score rubrics,
//! macro-F1, AURC, accuracy at target coverage, and source-group bootstrap
//! intervals.

use std::collections::BTreeMap;

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};

const NLL_FLOOR: f64 = 1e-15;
const NONE: &str = "__none__";
const BOOTSTRAP_SEED: u64 = 29_160_717;
const BOOTSTRAP_REPLICATES: usize = 2_000;

/// 95% percentile bootstrap interval over source-group clusters.
pub fn bootstrap_ci(correct: &[bool], groups: &[String]) -> Value {
    let mut cluster_ids: Vec<&String> = Vec::new();
    let mut cluster_correct: Vec<Vec<bool>> = Vec::new();
    for (group, value) in groups.iter().zip(correct.iter()) {
        match cluster_ids.iter().position(|known| *known == group) {
            Some(index) => cluster_correct[index].push(*value),
            None => {
                cluster_ids.push(group);
                cluster_correct.push(vec![*value]);
            }
        }
    }
    let point = correct.iter().filter(|c| **c).count() as f64 / correct.len().max(1) as f64;
    let mut generator = StdRng::seed_from_u64(BOOTSTRAP_SEED);
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut hits = 0usize;
        let mut total = 0usize;
        for _ in 0..cluster_ids.len() {
            let index = generator.random_range(0..cluster_ids.len());
            hits += cluster_correct[index].iter().filter(|c| **c).count();
            total += cluster_correct[index].len();
        }
        if total > 0 {
            samples.push(hits as f64 / total as f64);
        }
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let lower = samples.get(50).copied().unwrap_or(point);
    let upper = samples
        .get(BOOTSTRAP_REPLICATES - 51)
        .copied()
        .unwrap_or(point);
    json!({
        "point": point,
        "ci_95": [lower, upper],
        "replicates": samples.len(),
        "clusters": cluster_ids.len(),
        "seed": BOOTSTRAP_SEED,
    })
}

/// Equal-width binned expected calibration error on the winner probability,
/// comparing bin accuracy against the bin's mean confidence (the same fixed
/// 10-bin estimator the authored-panel diagnostics use).
pub fn ece(winner_probabilities: &[f64], correct: &[bool], bins: usize) -> f64 {
    let mut bin_confidence = vec![0.0; bins];
    let mut bin_correct = vec![0usize; bins];
    let mut bin_total = vec![0usize; bins];
    for (probability, hit) in winner_probabilities.iter().zip(correct.iter()) {
        let index = (((*probability) * bins as f64).floor() as usize).min(bins - 1);
        bin_total[index] += 1;
        bin_confidence[index] += probability;
        if *hit {
            bin_correct[index] += 1;
        }
    }
    let mut error = 0.0;
    for index in 0..bins {
        if bin_total[index] == 0 {
            continue;
        }
        let fraction = bin_correct[index] as f64 / bin_total[index] as f64;
        let confidence = bin_confidence[index] / bin_total[index] as f64;
        error += bin_total[index] as f64 / winner_probabilities.len() as f64
            * (fraction - confidence).abs();
    }
    error
}

/// Area under the risk-coverage curve: mean over all prefixes (highest
/// confidence first) of the running error rate.
pub fn aurc(confidences: &[f64], correct: &[bool]) -> f64 {
    let mut order: Vec<usize> = (0..confidences.len()).collect();
    order.sort_by(|a, b| confidences[*b].total_cmp(&confidences[*a]));
    let mut errors = 0usize;
    let mut risk_sum = 0.0;
    for (position, index) in order.iter().enumerate() {
        if !correct[*index] {
            errors += 1;
        }
        risk_sum += errors as f64 / (position + 1) as f64;
    }
    if order.is_empty() {
        0.0
    } else {
        risk_sum / order.len() as f64
    }
}

/// Accuracy when only the most confident `coverage` fraction is answered.
pub fn accuracy_at_coverage(confidences: &[f64], correct: &[bool], coverage: f64) -> Option<Value> {
    let keep = ((confidences.len() as f64 * coverage).ceil() as usize).min(confidences.len());
    if keep == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..confidences.len()).collect();
    order.sort_by(|a, b| confidences[*b].total_cmp(&confidences[*a]));
    let hits = order[..keep]
        .iter()
        .filter(|index| correct[**index])
        .count();
    Some(json!({
        "coverage": keep as f64 / confidences.len() as f64,
        "accuracy": hits as f64 / keep as f64,
        "answered": keep,
    }))
}

/// Mann-Whitney AUROC with tie-corrected ranks. `None` when single-class.
pub fn auroc(probabilities: &[f64], gold: &[bool]) -> Option<f64> {
    let positives = gold.iter().filter(|g| **g).count();
    let negatives = gold.len() - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }
    let mut ranked: Vec<(f64, bool)> = probabilities
        .iter()
        .copied()
        .zip(gold.iter().copied())
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum = 0.0;
    let mut index = 0;
    while index < ranked.len() {
        let mut end = index + 1;
        while end < ranked.len() && ranked[end].0 == ranked[index].0 {
            end += 1;
        }
        let average_rank = (index + 1 + end) as f64 / 2.0;
        for entry in &ranked[index..end] {
            if entry.1 {
                rank_sum += average_rank;
            }
        }
        index = end;
    }
    Some(
        (rank_sum - positives as f64 * (positives as f64 + 1.0) / 2.0)
            / (positives as f64 * negatives as f64),
    )
}

fn rank_values(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut ranks = vec![0.0; values.len()];
    let mut index = 0;
    while index < order.len() {
        let mut end = index + 1;
        while end < order.len() && values[order[end]] == values[order[index]] {
            end += 1;
        }
        let average = (index + 1 + end) as f64 / 2.0;
        for position in &order[index..end] {
            ranks[*position] = average;
        }
        index = end;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (a, b) in x.iter().zip(y.iter()) {
        covariance += (a - mean_x) * (b - mean_y);
        variance_x += (a - mean_x).powi(2);
        variance_y += (b - mean_y).powi(2);
    }
    (variance_x > 0.0 && variance_y > 0.0).then(|| covariance / (variance_x * variance_y).sqrt())
}

pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    pearson(&rank_values(x), &rank_values(y))
}

/// Best accuracy threshold on a tuning set (paper practice for Noul rows);
/// returns the threshold and its tuning accuracy.
pub fn best_threshold_tuned_on(probabilities: &[f64], gold: &[bool]) -> Option<(f64, f64)> {
    if probabilities.len() != gold.len() || probabilities.is_empty() {
        return None;
    }
    let mut candidates: Vec<f64> = probabilities.to_vec();
    candidates.push(0.5);
    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup();
    let mut best = (f64::NAN, f64::NAN);
    for threshold in candidates {
        let hits = probabilities
            .iter()
            .zip(gold.iter())
            .filter(|(p, g)| (**p >= threshold) == **g)
            .count();
        let accuracy = hits as f64 / probabilities.len() as f64;
        if matches!(
            accuracy.partial_cmp(&best.1),
            Some(std::cmp::Ordering::Greater) | None
        ) {
            best = (threshold, accuracy);
        }
    }
    Some(best)
}

/// Per-class recall over (task, label) pairs, mirroring the quality report.
pub fn per_class_recall(golds: &[String], predictions: &[String]) -> Value {
    let mut classes: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (gold, prediction) in golds.iter().zip(predictions.iter()) {
        let entry = classes.entry(gold.clone()).or_insert((0, 0));
        entry.1 += 1;
        if prediction == gold {
            entry.0 += 1;
        }
    }
    let mut out = Vec::new();
    for (label, (hits, total)) in classes {
        out.push(json!({
            "label": label,
            "count": total,
            "recall": hits as f64 / total as f64,
        }));
    }
    Value::Array(out)
}

/// Macro-F1 over the gold classes observed in the split.
pub fn macro_f1(golds: &[String], predictions: &[String]) -> Option<f64> {
    let mut classes: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for (gold, prediction) in golds.iter().zip(predictions.iter()) {
        let entry = classes.entry(gold.as_str()).or_insert((0, 0, 0));
        entry.0 += 1; // support
        if prediction == gold {
            entry.1 += 1; // true positive
        }
        let entry = classes.entry(prediction.as_str()).or_insert((0, 0, 0));
        entry.2 += 1; // predicted
    }
    let mut scores = Vec::new();
    for (_, (support, hits, predicted)) in classes {
        if support == 0 {
            continue;
        }
        let precision = hits as f64 / predicted as f64;
        let recall = hits as f64 / support as f64;
        if precision + recall > 0.0 {
            scores.push(2.0 * precision * recall / (precision + recall));
        } else {
            scores.push(0.0);
        }
    }
    (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64)
}

/// Multiclass Choice metrics over full offered distributions.
pub struct ChoiceRow {
    pub probabilities: BTreeMap<String, f64>,
    pub gold: String,
}

pub fn choice_metrics(rows: &[ChoiceRow], groups: &[String]) -> Value {
    let mut correct = Vec::new();
    let mut groups_for_bootstrap: Vec<String> = Vec::new();
    let mut golds = Vec::new();
    let mut predictions = Vec::new();
    let mut winner_probabilities = Vec::new();
    let mut confidences = Vec::new();
    let mut nll = Vec::new();
    let mut brier = Vec::new();
    let mut answerable = 0;
    let mut ranking_correct = 0;
    let mut none_gold = 0;
    let mut none_hits = 0;
    let mut false_none = 0;
    for (row, group) in rows.iter().zip(groups.iter()) {
        let mut winner = ("", f64::NEG_INFINITY);
        let mut winner_real = ("", f64::NEG_INFINITY);
        for (label, probability) in &row.probabilities {
            if *probability > winner.1 {
                winner = (label.as_str(), *probability);
            }
            if label.as_str() != NONE && *probability > winner_real.1 {
                winner_real = (label.as_str(), *probability);
            }
        }
        let hit = winner.0 == row.gold;
        correct.push(hit);
        groups_for_bootstrap.push(group.clone());
        golds.push(row.gold.clone());
        predictions.push(winner.0.to_owned());
        winner_probabilities.push(winner.1);
        confidences.push(winner.1);
        nll.push(
            -row.probabilities
                .get(&row.gold)
                .copied()
                .unwrap_or(0.0)
                .max(NLL_FLOOR)
                .ln(),
        );
        brier.push(
            row.probabilities
                .iter()
                .map(|(label, p)| (p - f64::from(label.as_str() == row.gold)).powi(2))
                .sum::<f64>(),
        );
        if row.gold != NONE {
            answerable += 1;
            if winner_real.0 == row.gold {
                ranking_correct += 1;
            }
            if winner.0 == NONE {
                false_none += 1;
            }
        } else {
            none_gold += 1;
            if winner.0 == NONE {
                none_hits += 1;
            }
        }
    }
    let hits = correct.iter().filter(|c| **c).count();
    let mut risk_coverage = Vec::new();
    for threshold in [0.5, 0.75, 0.9, 0.98] {
        let answered = confidences
            .iter()
            .zip(correct.iter())
            .filter(|(confidence, _)| **confidence >= threshold)
            .count();
        let answered_correct = confidences
            .iter()
            .zip(correct.iter())
            .filter(|(confidence, hit)| **confidence >= threshold && **hit)
            .count();
        risk_coverage.push(json!({
            "threshold": threshold,
            "answered": answered,
            "accepted_error": if answered > 0 {
                Value::from(1.0 - answered_correct as f64 / answered as f64)
            } else {
                Value::Null
            },
        }));
    }
    json!({
        "rows": rows.len(),
        "accuracy": hits as f64 / rows.len() as f64,
        "accuracy_bootstrap": bootstrap_ci(&correct, &groups_for_bootstrap),
        "answerable_ranking_accuracy": if answerable > 0 {
            Value::from(ranking_correct as f64 / answerable as f64)
        } else {
            Value::Null
        },
        "false_none_rate": if answerable > 0 {
            Value::from(false_none as f64 / answerable as f64)
        } else {
            Value::Null
        },
        "none_recall": if none_gold > 0 {
            Value::from(none_hits as f64 / none_gold as f64)
        } else {
            Value::Null
        },
        "macro_f1": macro_f1(&golds, &predictions),
        "nll": nll.iter().sum::<f64>() / rows.len() as f64,
        "brier": brier.iter().sum::<f64>() / rows.len() as f64,
        "ece_10_bins": ece(&winner_probabilities, &correct, 10),
        "aurc": aurc(&confidences, &correct),
        "accuracy_at_coverage": {
            "0.5": accuracy_at_coverage(&confidences, &correct, 0.5),
            "0.8": accuracy_at_coverage(&confidences, &correct, 0.8),
        },
        "risk_coverage": risk_coverage,
        "per_class_recall": per_class_recall(&golds, &predictions),
    })
}

/// Binary Noul metrics over P(yes).
pub struct NoulRow {
    pub probability: f64,
    pub gold: bool,
    pub group: String,
}

pub fn noul_metrics(rows: &[NoulRow]) -> Value {
    let probabilities: Vec<f64> = rows.iter().map(|row| row.probability).collect();
    let gold: Vec<bool> = rows.iter().map(|row| row.gold).collect();
    let groups: Vec<String> = rows.iter().map(|row| row.group.clone()).collect();
    let correct: Vec<bool> = rows
        .iter()
        .map(|row| (row.probability >= 0.5) == row.gold)
        .collect();
    let confidences: Vec<f64> = probabilities
        .iter()
        .map(|p| (p - 0.5).abs() + 0.5)
        .collect();
    let hits = correct.iter().filter(|c| **c).count();
    let brier: Vec<f64> = rows
        .iter()
        .map(|row| (row.probability - f64::from(row.gold)).powi(2))
        .collect();
    let mut positive_recall = None;
    let mut negative_recall = None;
    let positives = rows.iter().filter(|row| row.gold).count();
    let negatives = rows.len() - positives;
    if positives > 0 {
        positive_recall = Some(
            rows.iter()
                .filter(|row| row.gold && row.probability >= 0.5)
                .count() as f64
                / positives as f64,
        );
    }
    if negatives > 0 {
        negative_recall = Some(
            rows.iter()
                .filter(|row| !row.gold && row.probability < 0.5)
                .count() as f64
                / negatives as f64,
        );
    }
    json!({
        "rows": rows.len(),
        "accuracy_at_0.5": hits as f64 / rows.len() as f64,
        "accuracy_bootstrap": bootstrap_ci(&correct, &groups),
        "auroc": auroc(&probabilities, &gold),
        "brier": brier.iter().sum::<f64>() / rows.len() as f64,
        "ece_10_bins": ece(&probabilities, &correct, 10),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "aurc": aurc(&confidences, &correct),
        "accuracy_at_coverage": {
            "0.5": accuracy_at_coverage(&confidences, &correct, 0.5),
            "0.8": accuracy_at_coverage(&confidences, &correct, 0.8),
        },
    })
}

/// Ordinal Score metrics: expected score versus gold on the level scale.
pub struct ScoreRow {
    pub expected: f64,
    pub gold: f64,
    pub levels: usize,
    /// Probability mass of the level nearest the gold.
    pub gold_level_probability: f64,
}

pub fn score_metrics(rows: &[ScoreRow]) -> Value {
    let expected: Vec<f64> = rows.iter().map(|row| row.expected).collect();
    let gold: Vec<f64> = rows.iter().map(|row| row.gold).collect();
    let clamp = |value: f64, levels: usize| value.round().clamp(0.0, levels as f64 - 1.0);
    let level_hits = rows
        .iter()
        .filter(|row| clamp(row.expected, row.levels) == clamp(row.gold, row.levels))
        .count();
    let mae = rows
        .iter()
        .map(|row| (row.expected - row.gold).abs())
        .sum::<f64>()
        / rows.len() as f64;
    let nll = rows
        .iter()
        .map(|row| -row.gold_level_probability.max(NLL_FLOOR).ln())
        .sum::<f64>()
        / rows.len() as f64;
    json!({
        "rows": rows.len(),
        "argmax_level_accuracy": level_hits as f64 / rows.len() as f64,
        "spearman": spearman(&expected, &gold),
        "pearson": pearson(&expected, &gold),
        "mae": mae,
        "nll": nll,
    })
}

#[cfg(test)]
mod metric_tests {
    use super::*;

    fn noul_row(probability: f64, gold: bool) -> NoulRow {
        NoulRow {
            probability,
            gold,
            group: "g".to_owned(),
        }
    }

    #[test]
    fn noul_metrics_computes_hand_computed_binary_stats() {
        // p=0.9/true (hit), p=0.2/false (hit), p=0.4/true (miss at 0.5).
        let rows = vec![
            noul_row(0.9, true),
            noul_row(0.2, false),
            noul_row(0.4, true),
        ];
        let metrics = noul_metrics(&rows);
        assert_eq!(metrics["rows"], 3);
        let accuracy = metrics["accuracy_at_0.5"].as_f64().unwrap();
        assert!((accuracy - 2.0 / 3.0).abs() < 1e-12);
        let brier = metrics["brier"].as_f64().unwrap();
        let expected_brier: f64 =
            ((0.9_f64 - 1.0).powi(2) + (0.2_f64 - 0.0).powi(2) + (0.4_f64 - 1.0).powi(2)) / 3.0;
        assert!((brier - expected_brier).abs() < 1e-12);
        assert!((metrics["positive_recall"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert!((metrics["negative_recall"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        // A perfectly separable ordering gives the top AUROC.
        assert_eq!(metrics["auroc"].as_f64().unwrap(), 1.0);
    }

    #[test]
    fn noul_metrics_renders_null_recalls_when_a_class_is_absent() {
        let all_positive = vec![noul_row(0.9, true), noul_row(0.3, true)];
        let metrics = noul_metrics(&all_positive);
        assert!(metrics["negative_recall"].is_null());
        assert!((metrics["positive_recall"].as_f64().unwrap() - 0.5).abs() < 1e-12);

        let all_negative = vec![noul_row(0.9, false), noul_row(0.3, false)];
        let metrics = noul_metrics(&all_negative);
        assert!(metrics["positive_recall"].is_null());
        assert!((metrics["negative_recall"].as_f64().unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn score_metrics_clamps_levels_and_flags_constant_input_correlations() {
        let rows = vec![ScoreRow {
            expected: 5.0, // rounds to 5, clamps to levels-1 = 2
            gold: 1.6,     // rounds to 2
            levels: 3,
            gold_level_probability: 0.5,
        }];
        let metrics = score_metrics(&rows);
        assert_eq!(metrics["rows"], 1);
        assert_eq!(metrics["argmax_level_accuracy"].as_f64().unwrap(), 1.0);
        assert!((metrics["mae"].as_f64().unwrap() - 3.4).abs() < 1e-12);
        // The floor keeps the NLL finite when the gold level carried no mass.
        let nll = metrics["nll"].as_f64().unwrap();
        assert!(nll.is_finite() && nll > 0.0);

        // Constant vectors cannot be correlated: both correlations are null.
        let flat = vec![
            ScoreRow {
                expected: 1.0,
                gold: 0.5,
                levels: 3,
                gold_level_probability: 1.0,
            },
            ScoreRow {
                expected: 1.0,
                gold: 2.0,
                levels: 3,
                gold_level_probability: 1.0,
            },
        ];
        let metrics = score_metrics(&flat);
        assert!(metrics["spearman"].is_null());
        assert!(metrics["pearson"].is_null());

        // A perfectly monotone pair correlates at 1.0 on both scales.
        let ordered = vec![
            ScoreRow {
                expected: 0.0,
                gold: 0.0,
                levels: 3,
                gold_level_probability: 1.0,
            },
            ScoreRow {
                expected: 2.0,
                gold: 2.0,
                levels: 3,
                gold_level_probability: 1.0,
            },
        ];
        let metrics = score_metrics(&ordered);
        assert_eq!(metrics["spearman"].as_f64().unwrap(), 1.0);
        assert!((metrics["pearson"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_ci_on_empty_input_falls_back_to_point_defaults() {
        let empty: Vec<bool> = vec![];
        let groups: Vec<String> = vec![];
        let ci = bootstrap_ci(&empty, &groups);
        assert_eq!(ci["point"], 0.0);
        assert_eq!(ci["replicates"], 0);

        let ci = bootstrap_ci(&[true, true, false], &["g"; 3].map(String::from));
        assert!((ci["point"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn coverage_and_risk_curves_degenerate_to_none_or_zero() {
        // Coverage 0 keeps no rows.
        assert!(accuracy_at_coverage(&[0.9, 0.1], &[true, false], 0.0).is_none());
        // An empty AURC is zero risk.
        assert_eq!(aurc(&[], &[]), 0.0);
    }
}
