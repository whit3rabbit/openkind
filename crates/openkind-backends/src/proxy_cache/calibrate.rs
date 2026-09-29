//! Confidence-threshold calibration under a statistical budget.
//!
//! The task tolerates at most `budget = 1 - target_agreement` probability
//! mass of (answered and disagreed). Candidate thresholds come from a grid
//! fixed before calibration rows are seen; the scan walks strictest-first
//! and keeps the loosest threshold whose one-sided Clopper–Pearson upper
//! bound on the observed disagreement count still fits the budget — a
//! fixed-sequence test, so the guarantee holds at confidence `1 - delta`.

use serde::{Deserialize, Serialize};

use super::ProxyCacheError;

/// Lanczos approximation of the log-gamma function (g=7, n=9 coefficients).
pub fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection formula for the small-x branch.
        let series = std::f64::consts::PI / (std::f64::consts::PI * x).sin().max(1e-300);
        return std::f64::consts::PI.ln() - series.ln() - lgamma(1.0 - x);
    }
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = COEFFICIENTS[0];
    let t = x + 7.5;
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        a += coefficient / (x + index as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// `log P(X ≤ k | n, p)` via logsumexp over binomial terms.
fn log_binomial_cdf(k: usize, n: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return if k >= n { 0.0 } else { f64::NEG_INFINITY };
    }
    let log_p = p.ln();
    let log_1m_p = (1.0 - p).ln();
    let mut terms = Vec::with_capacity(k + 1);
    for i in 0..=k {
        let log_choose =
            lgamma(n as f64 + 1.0) - lgamma(i as f64 + 1.0) - lgamma((n - i) as f64 + 1.0);
        terms.push(log_choose + i as f64 * log_p + (n - i) as f64 * log_1m_p);
    }
    let max = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = terms.iter().map(|t| (t - max).exp()).sum();
    max + sum.ln()
}

/// The smallest success probability `p` with `P(X ≤ k | n, p) ≤ delta`
/// (bisection, 60 iterations). This is the one-sided Clopper–Pearson upper
/// bound for `k` disagreements in `n` answered rows.
pub fn clopper_pearson_upper(k: usize, n: usize, delta: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }
    if delta >= 1.0 {
        return 1.0;
    }
    // Monotone: the CDF decreases in p, so bisect on `log cdf ≤ log delta`.
    let log_delta = delta.ln();
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for _ in 0..60 {
        let mid = (low + high) / 2.0;
        if log_binomial_cdf(k, n, mid) <= log_delta {
            high = mid;
        } else {
            low = mid;
        }
    }
    (low + high) / 2.0
}

/// One-sided Clopper–Pearson lower bound for `k` successes in `n` rows.
pub fn clopper_pearson_lower(k: usize, n: usize, delta: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    1.0 - clopper_pearson_upper(n - k, n, delta)
}

/// The candidate threshold grid, fixed before calibration rows are seen:
/// `1 - geomspace(1e-4, 1 - 1/max(n_labels, 2), size)`, strictest first,
/// dense near 1.
pub fn threshold_grid(n_labels: usize, size: usize) -> Vec<f64> {
    let size = size.max(2);
    let high = 1.0 - 1.0 / (n_labels.max(2) as f64);
    let low = 1e-4_f64;
    if high <= low {
        return vec![1.0 - high];
    }
    let log_low = low.ln();
    let log_high = high.ln();
    (0..size)
        .map(|step| {
            let fraction = step as f64 / (size - 1) as f64;
            1.0 - (log_low + fraction * (log_high - log_low)).exp()
        })
        .collect()
}

/// One calibration observation: the student's routing-side confidence and
/// predicted label, the teacher's answer, and the gate score.
#[derive(Debug, Clone)]
pub struct CalibRow {
    /// Routing confidence (max softmax probability).
    pub student_confidence: f64,
    /// Student argmax label.
    pub student_label: String,
    /// Teacher's label.
    pub teacher_label: String,
    /// Teacher's reported confidence (for floor-aware disagreement).
    pub teacher_confidence: f64,
}

/// The calibrated routing decision for one student version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingPolicy {
    /// Loosest threshold fitting the budget; `None` = unusable policy.
    pub conf_threshold: Option<f64>,
    /// kNN gate threshold (may be infinite; stored as a JSON number or the
    /// string "inf" for readability).
    pub ood_threshold: f64,
    /// Share of calibration rows the policy would answer.
    pub expected_coverage: f64,
    /// The budget the policy was fitted against (after headroom).
    pub disagreement_ub: f64,
    /// Expected disagreement rate at the chosen threshold.
    pub expected_system_disagreement: f64,
    /// Full budget of the task.
    pub budget: f64,
    /// Statistical confidence (1 − delta) of the bound.
    pub delta: f64,
    /// Calibration rows used.
    pub n_calib: usize,
    /// Labels the policy refuses to serve (rare classes).
    pub deferred_labels: Vec<String>,
    /// Reported-confidence floor mirrored from the task.
    pub confidence_floor: Option<f64>,
}

impl RoutingPolicy {
    /// Whether the policy can answer anything.
    pub fn usable(&self) -> bool {
        self.conf_threshold.is_some()
    }

    /// Serialize to pretty JSON for the version directory.
    pub fn to_json(&self) -> Result<String, ProxyCacheError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| ProxyCacheError::Version(format!("serialize policy: {error}")))
    }

    /// Restore from JSON.
    pub fn from_json(text: &str) -> Result<Self, ProxyCacheError> {
        serde_json::from_str(text)
            .map_err(|error| ProxyCacheError::Version(format!("deserialize policy: {error}")))
    }
}

/// Whether one answered row counts as a disagreement.
pub fn disagrees(
    student_label: &str,
    teacher_label: &str,
    teacher_confidence: f64,
    confidence_floor: Option<f64>,
) -> bool {
    if student_label != teacher_label {
        return true;
    }
    // A confidence floor means unsure teacher answers count as disagreement
    // too: the student must inherit the teacher's doubt.
    match confidence_floor {
        Some(floor) => floor > 0.0 && teacher_confidence < floor,
        None => false,
    }
}

/// Outcome of one `fit_policy` scan.
#[derive(Debug, Clone)]
pub struct FitOutcome {
    /// The fitted policy (`conf_threshold = None` when nothing fit).
    pub policy: RoutingPolicy,
    /// Whether the scan found any usable threshold.
    pub usable: bool,
    /// Rejection reason when unusable.
    pub reason: Option<&'static str>,
}

/// Fit the routing policy over calibration rows.
///
/// * `rows` — calibration observations (student outputs already scored).
/// * `ood_scores` — gate score per row (parallel to `rows`).
/// * `ood_threshold` — gate threshold from the training-only reference.
/// * `n_labels` — number of task classes (bounds the meaningful threshold
///   range: max softmax probability is at least `1/n_labels`).
/// * `budget` — full task disagreement budget.
/// * `headroom` — fraction shaved off the budget for the fit (shadow judges
///   at the full budget).
/// * `delta` — `1 - confidence`.
/// * `deferred_labels` — labels the policy must not serve.
/// * `confidence_floor` — floor-aware disagreement counting.
#[allow(clippy::too_many_arguments)]
pub fn fit_policy(
    rows: &[CalibRow],
    ood_scores: &[f64],
    ood_threshold: f64,
    n_labels: usize,
    budget: f64,
    headroom: f64,
    delta: f64,
    deferred_labels: &[String],
    confidence_floor: Option<f64>,
) -> Result<FitOutcome, ProxyCacheError> {
    if rows.len() != ood_scores.len() {
        return Err(ProxyCacheError::Contract(
            "calibration rows and ood scores must be parallel".into(),
        ));
    }
    let effective_budget = (budget * (1.0 - headroom.clamp(0.0, 0.99))).max(1e-9);
    let n_total = rows.len();

    // An infinite gate threshold cannot serialize into the policy; clamp it
    // to the largest calibration score (the gate then passes everything it
    // saw, which is what an unfitted gate means).
    let gate = if ood_threshold.is_finite() || rows.is_empty() {
        ood_threshold
    } else {
        ood_scores.iter().cloned().fold(0.0_f64, f64::max)
    };

    // Per-row loss: 1[answered AND disagreed]. The bound is over ALL N
    // calibration rows — the budget limits the system disagreement rate over
    // every request, so unanswered rows simply contribute zero loss. Losses
    // are fixed before the scan; the threshold only selects.
    let losses: Vec<bool> = rows
        .iter()
        .zip(ood_scores)
        .map(|(row, &ood)| {
            let selected = ood <= gate
                && !deferred_labels
                    .iter()
                    .any(|label| label == &row.student_label);
            selected
                && disagrees(
                    &row.student_label,
                    &row.teacher_label,
                    row.teacher_confidence,
                    confidence_floor,
                )
        })
        .collect();

    let mut best: Option<(f64, usize, usize, f64)> = None; // (threshold, answered, disagreed, bound)
    if n_total > 0 {
        let grid = threshold_grid(n_labels, 400);
        for &threshold in &grid {
            let mut answered = 0usize;
            let mut disagreed = 0usize;
            // Selection differs per threshold only through the confidence
            // comparison; the gate and deferral filters are fixed.
            for ((row, &ood), &loss) in rows.iter().zip(ood_scores).zip(&losses) {
                if row.student_confidence < threshold || ood > gate {
                    continue;
                }
                if deferred_labels
                    .iter()
                    .any(|label| label == &row.student_label)
                {
                    continue;
                }
                answered += 1;
                if loss {
                    disagreed += 1;
                }
            }
            let bound = clopper_pearson_upper(disagreed, n_total, delta);
            if bound > effective_budget {
                // Fixed-sequence testing: losses only grow as the threshold
                // loosens, so the first failure retires every looser
                // candidate without a multiple-testing penalty.
                break;
            }
            if answered > 0 {
                best = Some((threshold, answered, disagreed, bound));
            }
        }
    }

    let (threshold, answered, disagreed, bound) = match best {
        Some(found) => found,
        None => {
            return Ok(FitOutcome {
                policy: RoutingPolicy {
                    conf_threshold: None,
                    ood_threshold: gate,
                    expected_coverage: 0.0,
                    disagreement_ub: 1.0,
                    expected_system_disagreement: 0.0,
                    budget,
                    delta,
                    n_calib: n_total,
                    deferred_labels: deferred_labels.to_vec(),
                    confidence_floor,
                },
                usable: false,
                reason: Some("no threshold satisfies the budget"),
            });
        }
    };

    let policy = RoutingPolicy {
        conf_threshold: Some(threshold),
        ood_threshold: gate,
        expected_coverage: answered as f64 / n_total as f64,
        disagreement_ub: bound,
        expected_system_disagreement: if answered > 0 {
            disagreed as f64 / answered as f64
        } else {
            0.0
        },
        budget,
        delta,
        n_calib: n_total,
        deferred_labels: deferred_labels.to_vec(),
        confidence_floor,
    };
    Ok(FitOutcome {
        policy,
        usable: true,
        reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lgamma_matches_known_values() {
        assert!((lgamma(1.0) - 0.0).abs() < 1e-9);
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-9);
        assert!((lgamma(0.5) - 0.5723649429247001).abs() < 1e-9);
        assert!((lgamma(10.0) - 362_880.0_f64.ln()).abs() < 1e-8);
    }

    #[test]
    fn closed_form_zero_disagreements() {
        // CP upper bound for k=0 is exactly 1 - delta^(1/n).
        for n in [5usize, 50, 500] {
            let expected = 1.0 - 0.05_f64.powf(1.0 / n as f64);
            let actual = clopper_pearson_upper(0, n, 0.05);
            assert!(
                (actual - expected).abs() < 1e-6,
                "n={n} expected {expected} got {actual}"
            );
        }
    }

    #[test]
    fn bound_is_monotone_in_disagreements_and_rows() {
        let a = clopper_pearson_upper(1, 100, 0.05);
        let b = clopper_pearson_upper(3, 100, 0.05);
        let c = clopper_pearson_upper(1, 500, 0.05);
        assert!(a < b, "more disagreements loosens the bound");
        assert!(c < a, "more rows tightens the bound");
        assert!((clopper_pearson_upper(100, 100, 0.05) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lower_bound_mirrors_upper() {
        let lower = clopper_pearson_lower(30, 100, 0.05);
        let upper = clopper_pearson_upper(70, 100, 0.05);
        assert!((lower - (1.0 - upper)).abs() < 1e-9);
    }

    #[test]
    fn grid_is_descending_and_dense_near_one() {
        let grid = threshold_grid(5, 400);
        assert_eq!(grid.len(), 400);
        assert!(grid[0] > grid[100]);
        assert!(grid[0] > grid[399]);
        // The grid is 1 − geomspace(1e-4, 1 − 1/n): strictest ≈ 0.9999,
        // loosest exactly 1/n (max-probability floor for any argmax).
        let last = grid[grid.len() - 1];
        assert!((last - 1.0 / 5.0).abs() < 1e-9, "last {last}");
        // Strictest threshold is close to 1.
        assert!(grid[0] >= 0.9999, "strictest {}", grid[0]);
    }

    fn row(conf: f64, student: &str, teacher: &str) -> CalibRow {
        CalibRow {
            student_confidence: conf,
            student_label: student.to_string(),
            teacher_label: teacher.to_string(),
            teacher_confidence: 0.99,
        }
    }

    #[test]
    fn policy_respects_the_budget() {
        // 200 calibration rows; 10% of confident answers disagree.
        let mut rows = Vec::new();
        let mut oods = Vec::new();
        for i in 0..200 {
            if i % 10 == 0 {
                rows.push(row(0.99, "a", "b"));
            } else {
                rows.push(row(0.99, "a", "a"));
            }
            oods.push(0.1);
        }
        let outcome = fit_policy(&rows, &oods, 0.5, 2, 0.02, 0.0, 0.05, &[], None).unwrap();
        // 10% disagreement rate cannot fit a 2% budget at any threshold.
        assert!(!outcome.usable);
        assert!(outcome.policy.conf_threshold.is_none());
    }

    #[test]
    fn policy_finds_the_loosest_passing_threshold() {
        // Disagreements only among the low-confidence half.
        let mut rows = Vec::new();
        let mut oods = Vec::new();
        for i in 0..200 {
            let conf = if i % 2 == 0 { 0.95 } else { 0.8 };
            let teacher = if conf < 0.9 && i % 4 == 1 { "b" } else { "a" };
            rows.push(row(conf, "a", teacher));
            oods.push(0.1);
        }
        let outcome = fit_policy(&rows, &oods, 0.5, 2, 0.05, 0.0, 0.05, &[], None).unwrap();
        assert!(outcome.usable, "reason {:?}", outcome.reason);
        let threshold = outcome.policy.conf_threshold.unwrap();
        assert!(
            threshold > 0.8 && threshold <= 0.95,
            "threshold {threshold} should sit between the two clusters"
        );
        // The chosen threshold answers ~half the rows with zero disagreement.
        assert!((outcome.policy.expected_coverage - 0.5).abs() < 0.05);
        assert!(outcome.policy.expected_system_disagreement <= 0.05 + 1e-9);
    }

    #[test]
    fn headroom_tightens_or_rejects_the_fit() {
        // 7 disagreements in 300 confident rows: the full-budget upper bound
        // squeaks under a 5% budget, the 15%-headroom budget does not.
        let mut rows = Vec::new();
        let mut oods = Vec::new();
        for i in 0..300 {
            let teacher = if i % 43 == 0 { "b" } else { "a" };
            rows.push(row(0.99, "a", teacher));
            oods.push(0.1);
        }
        let disagreed = rows
            .iter()
            .filter(|r| r.student_label != r.teacher_label)
            .count();
        let bound = clopper_pearson_upper(disagreed, 300, 0.05);
        let loose = fit_policy(&rows, &oods, 0.5, 2, 0.05, 0.0, 0.05, &[], None).unwrap();
        let tight = fit_policy(&rows, &oods, 0.5, 2, 0.05, 0.15, 0.05, &[], None).unwrap();
        if bound <= 0.05 {
            assert!(loose.usable, "bound {bound} should fit the 5% budget");
            if bound > 0.05 * 0.85 {
                assert!(!tight.usable, "bound {bound} must fail the headroom budget");
            } else if tight.usable {
                assert!(
                    tight.policy.conf_threshold.unwrap() >= loose.policy.conf_threshold.unwrap()
                );
            }
        } else {
            assert!(!loose.usable, "bound {bound} should not fit the 5% budget");
        }
    }

    #[test]
    fn ood_gate_and_deferred_labels_exclude_rows() {
        let mut rows = Vec::new();
        let mut oods = Vec::new();
        for i in 0..100 {
            rows.push(row(0.99, "a", if i == 0 { "b" } else { "a" }));
            oods.push(if i % 2 == 0 { 0.1 } else { 0.9 });
        }
        // With the gate at 0.5, only the in-distribution half is judged.
        let outcome = fit_policy(&rows, &oods, 0.5, 2, 0.15, 0.0, 0.05, &[], None).unwrap();
        assert!(outcome.usable, "reason {:?}", outcome.reason);
        assert!((outcome.policy.expected_coverage - 0.5).abs() < 0.05);

        // Deferring "a" answers nothing.
        let deferred = fit_policy(
            &rows,
            &oods,
            0.5,
            2,
            0.15,
            0.0,
            0.05,
            &["a".to_string()],
            None,
        )
        .unwrap();
        assert!(!deferred.usable);
    }

    #[test]
    fn confidence_floor_counts_unsure_teacher_rows() {
        // 200 rows: the zero-disagreement CP bound (≈1.5%) fits the 2%
        // budget, so the floor is what makes the policy unusable.
        let mut rows = Vec::new();
        let mut oods = Vec::new();
        for _ in 0..200 {
            rows.push(CalibRow {
                student_confidence: 0.99,
                student_label: "a".into(),
                teacher_label: "a".into(),
                teacher_confidence: 0.3,
            });
            oods.push(0.1);
        }
        let without_floor = fit_policy(&rows, &oods, 0.5, 2, 0.02, 0.0, 0.05, &[], None).unwrap();
        assert!(without_floor.usable);
        let with_floor = fit_policy(&rows, &oods, 0.5, 2, 0.02, 0.0, 0.05, &[], Some(0.6)).unwrap();
        assert!(
            !with_floor.usable,
            "unsure teachers must count as disagreements"
        );
    }

    #[test]
    fn policy_json_roundtrip() {
        let mut rows = Vec::new();
        let mut oods = Vec::new();
        for _ in 0..100 {
            rows.push(row(0.99, "a", "a"));
            oods.push(0.1);
        }
        let outcome = fit_policy(
            &rows,
            &oods,
            0.5,
            2,
            0.02,
            0.15,
            0.05,
            &["rare".to_string()],
            Some(0.4),
        )
        .unwrap();
        let json = outcome.policy.to_json().unwrap();
        let restored = RoutingPolicy::from_json(&json).unwrap();
        assert_eq!(restored, outcome.policy);
    }
}
