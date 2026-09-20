//! Mathematical operations for logits, score-summaries, and stable softmax.

use super::types::SCORE_SUMMARY_WIDTH;
use crate::qwen35::Qwen35Error;

pub(crate) fn score_summary(logits: &[f64]) -> Result<[f64; SCORE_SUMMARY_WIDTH], Qwen35Error> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::Numerical(
            "score summary requires finite non-empty logits".into(),
        ));
    }
    let mut ordered = logits.to_vec();
    ordered.sort_by(|left, right| right.total_cmp(left));
    let maximum = ordered[0];
    let gap = ordered.get(1).map_or(0.0, |second| maximum - second);
    let count = logits.len() as f64;
    let mean = logits.iter().sum::<f64>() / count;
    let variance = logits
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / count;
    let log_mean_exp = maximum
        + logits
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
        - count.ln();
    let summary = [
        maximum,
        gap,
        mean,
        variance.sqrt(),
        log_mean_exp,
        count.ln(),
    ];
    if summary.iter().all(|value| value.is_finite()) {
        Ok(summary)
    } else {
        Err(Qwen35Error::Numerical(
            "score-summary features are not finite".into(),
        ))
    }
}

pub(crate) fn stable_softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>, Qwen35Error> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::Numerical(
            "softmax requires finite non-empty logits".into(),
        ));
    }
    let scaled: Vec<f64> = logits.iter().map(|value| value / temperature).collect();
    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = scaled.iter().map(|value| (value - maximum).exp()).collect();
    let total = exponentials.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Qwen35Error::Numerical(
            "softmax normalization is not finite and positive".into(),
        ));
    }
    let probabilities: Vec<f64> = exponentials.iter().map(|value| value / total).collect();
    if probabilities.iter().all(|value| value.is_finite()) {
        Ok(probabilities)
    } else {
        Err(Qwen35Error::Numerical(
            "softmax probabilities are not finite".into(),
        ))
    }
}

pub(crate) fn first_argmax(values: &[f64]) -> usize {
    let mut selected = 0;
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > values[selected] {
            selected = index;
        }
    }
    selected
}
