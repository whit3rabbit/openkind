use std::collections::BTreeMap;

use anyhow::Result;
use openkind_backends::qwen35::SEMANTIC_NONE_OPTION;
use serde_json::{json, Value};

use super::Record;

pub(super) fn validate(probabilities: &BTreeMap<String, f64>, gold: &str) -> Result<()> {
    anyhow::ensure!(
        probabilities.contains_key(gold) && probabilities.contains_key(SEMANTIC_NONE_OPTION),
        "distribution omits gold or semantic none"
    );
    anyhow::ensure!(
        probabilities
            .values()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
        "invalid probability"
    );
    anyhow::ensure!(
        (probabilities.values().sum::<f64>() - 1.0).abs() < 1e-8,
        "probabilities do not sum to one"
    );
    Ok(())
}

// Lexical first wins ties, consistently for every method and permutation.
pub(super) fn prediction(probabilities: &BTreeMap<String, f64>, exclude_none: bool) -> (&str, f64) {
    let mut best = ("", f64::NEG_INFINITY);
    for (label, &probability) in probabilities {
        if exclude_none && label == SEMANTIC_NONE_OPTION {
            continue;
        }
        if probability > best.1 {
            best = (label.as_str(), probability);
        }
    }
    best
}

pub(super) fn losses(record: &Record, method: &str) -> [f64; 3] {
    let probabilities = &record.methods[method].score.probabilities;
    let correct = prediction(probabilities, false).0 == record.gold;
    let nll = -probabilities[&record.gold].max(1e-15).ln();
    let brier: f64 = probabilities
        .iter()
        .map(|(label, p)| (p - f64::from(label == &record.gold)).powi(2))
        .sum();
    [f64::from(correct), nll, brier]
}

fn fraction(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

pub(super) fn metrics(records: &[&Record], method: &str) -> Value {
    let mut correct = 0;
    let mut answerable = 0;
    let mut ranking_correct = 0;
    let mut none = 0;
    let mut none_correct = 0;
    let mut false_none = 0;
    let mut nll = 0.0;
    let mut brier = 0.0;
    let mut elapsed_ms = 0.0;
    let mut input_tokens = 0;
    let mut forwards = 0;
    let mut classes: BTreeMap<(&str, &str), (usize, usize)> = BTreeMap::new();
    let mut bins = [(0_usize, 0.0, 0.0); 10];
    let mut coverage = [(0_usize, 0_usize); 4];
    let thresholds = [0.5, 0.75, 0.9, 0.98];
    for record in records {
        let result = &record.methods[method];
        let (predicted, probability) = prediction(&result.score.probabilities, false);
        let is_correct = predicted == record.gold;
        correct += usize::from(is_correct);
        let class = classes.entry((&record.task, &record.gold)).or_default();
        class.0 += usize::from(is_correct);
        class.1 += 1;
        if record.gold == SEMANTIC_NONE_OPTION {
            none += 1;
            none_correct += usize::from(is_correct);
        } else {
            answerable += 1;
            ranking_correct +=
                usize::from(prediction(&result.score.probabilities, true).0 == record.gold);
            false_none += usize::from(predicted == SEMANTIC_NONE_OPTION);
        }
        let loss = losses(record, method);
        nll += loss[1];
        brier += loss[2];
        elapsed_ms += result.elapsed_ms;
        input_tokens += result.score.input_tokens;
        forwards += result.score.forwards;
        let bin = &mut bins[((probability * 10.0) as usize).min(9)];
        bin.0 += 1;
        bin.1 += probability;
        bin.2 += f64::from(is_correct);
        for (index, threshold) in thresholds.iter().enumerate() {
            // A semantic-none winner always goes to review, even when certain.
            if predicted != SEMANTIC_NONE_OPTION && probability >= *threshold {
                coverage[index].0 += 1;
                coverage[index].1 += usize::from(!is_correct);
            }
        }
    }
    let count = records.len();
    let ece = bins
        .iter()
        .filter(|bin| bin.0 > 0)
        .map(|bin| (bin.1 - bin.2).abs())
        .sum::<f64>()
        / count as f64;
    let recalls: Vec<_> = classes.iter().map(|((task, label), (hits, total))|
        json!({"task": task, "label": label, "count": total, "recall": *hits as f64 / *total as f64})).collect();
    let balanced_accuracy = classes
        .values()
        .map(|(hits, total)| *hits as f64 / *total as f64)
        .sum::<f64>()
        / classes.len() as f64;
    let risk_coverage: Vec<_> = thresholds.iter().zip(coverage).map(|(threshold, (accepted, errors))|
        json!({"threshold": threshold, "accepted": accepted, "coverage": fraction(accepted, count), "error_rate": fraction(errors, accepted)})).collect();
    json!({"count": count, "accuracy": fraction(correct, count), "balanced_accuracy": balanced_accuracy,
        "answerable_count": answerable, "answerable_ranking_accuracy": fraction(ranking_correct, answerable),
        "none_count": none, "none_recall": fraction(none_correct, none), "false_none_rate": fraction(false_none, answerable),
        "nll": nll / count as f64, "brier": brier / count as f64, "ece_10_bins": ece,
        "per_class_recall": recalls, "risk_coverage": risk_coverage,
        "scoring_seconds": elapsed_ms / 1000.0, "mean_scoring_ms": elapsed_ms / count as f64,
        "decisions_per_second": count as f64 * 1000.0 / elapsed_ms, "input_tokens": input_tokens, "forwards": forwards})
}
