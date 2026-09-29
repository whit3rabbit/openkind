//! Task identity and per-task engine configuration.
//!
//! A task is one (tenant, requested model, instructions, criteria) tuple.
//! The fingerprint is order-insensitive: criteria keys and JSON object keys
//! are canonicalized before hashing, so a reordered question is the same
//! task and any wording change is a new task that trains from scratch.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::text::canonical_json;

/// Upper bound on criteria labels one task can carry.
pub const MAX_CLASSES: usize = 255;

/// Operation mode of one task's engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TaskMode {
    /// Route with the production student when calibrated (default).
    #[default]
    Auto,
    /// Always forward to the teacher, recording rows for training.
    TeacherOnly,
}

/// What happens when the teacher's resolved model changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TeacherChangePolicy {
    /// Force fallback to teacher-only until a student of the new lineage
    /// passes shadow (default).
    #[default]
    Fallback,
    /// Keep serving the current student with an elevated audit rate.
    Audit,
}

/// How classes below `min_samples_per_class` are handled at training time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RareClasses {
    /// Train on them but never serve them (they become deferred labels).
    #[default]
    Defer,
    /// Block training until every class has enough rows.
    Wait,
}

/// Student target: the teacher's full distribution or a one-hot argmax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LabelTarget {
    /// Learn the teacher's soft probabilities (default).
    #[default]
    Probs,
    /// Learn one-hot argmax labels.
    Hard,
}

/// All knobs of one task's collect → train → calibrate → shadow → monitor
/// loop. Defaults mirror the reference design's measured settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TaskConfig {
    /// Statistical confidence for the Clopper–Pearson bound (1 − delta).
    pub confidence: f64,
    /// Fraction of eligible requests always answered by the teacher.
    pub audit_rate: f64,
    /// Audit rate while a shadow candidate exists.
    pub audit_rate_shadow: f64,
    /// Audit rate while drift is suspected.
    pub audit_rate_elevated: f64,
    /// Fraction of IID requests reserved for calibration (per request).
    pub calib_fraction: f64,
    /// Teacher-labelled train rows required for the first fit.
    pub min_train_samples: usize,
    /// Per-class teacher-labelled train rows required (rare → defer).
    pub min_samples_per_class: usize,
    /// Teacher-labelled calibration rows required for the first fit.
    pub min_calib_samples: usize,
    /// New teacher answers that trigger a retrain.
    pub min_new_samples: usize,
    /// Superseded versions kept besides the production one.
    pub keep_versions: usize,
    /// Shadow rows required before a candidate is judged.
    pub shadow_min_samples: usize,
    /// OOD threshold quantile over leave-one-out reference scores.
    pub ood_quantile: f64,
    /// Neighbors for the kNN out-of-distribution gate.
    pub ood_k: usize,
    /// Reference rows kept by the kNN gate (stratified by class).
    pub ood_max_ref: usize,
    /// Newest teacher-labelled train rows used per fit.
    pub max_train_samples: usize,
    /// Newest calibration rows used per calibration.
    pub max_calib_samples: usize,
    /// Newest student-served rows kept (no teacher label).
    pub keep_local_rows: usize,
    /// Rolling audit window for drift monitoring.
    pub drift_window: usize,
    /// Audit rows required before drift is scored.
    pub drift_min_samples: usize,
    /// Slack between the observed upper bound and the target before fallback.
    pub drift_margin: f64,
    /// New audit rows between drift checks.
    pub drift_check_every: usize,
    /// Operator mode.
    pub mode: TaskMode,
    /// RNG seed for audit draws, splits, and subsampling.
    pub seed: u64,
    /// Upper bound of full-batch Adam epochs per fit.
    pub student_epochs: usize,
    /// Student learning rate.
    pub student_lr: f64,
    /// L2 regularization on the linear weights.
    pub student_l2: f64,
    /// Early-stopping patience (non-improving evaluations).
    pub student_patience: usize,
    /// Held-out fraction for early stopping (0 disables; requires n ≥ 50).
    pub val_fraction: f64,
    /// Epochs between early-stopping evaluations.
    pub eval_every: usize,
    /// Student target kind.
    pub label_target: LabelTarget,
    /// Up-weight audit rows to keep training traffic-shaped.
    pub importance_weighting: bool,
    /// Exponent for audit importance weights.
    pub weight_power: f64,
    /// Cap on audit importance weights.
    pub max_weight: f64,
    /// Fit the policy at this fraction shaved off the budget; shadow judges
    /// at the full budget.
    pub fit_headroom: f64,
    /// Teacher-change reaction.
    pub teacher_change: TeacherChangePolicy,
    /// Consecutive different-model teacher answers before a lineage switch.
    pub teacher_change_confirm: usize,
    /// Rare-class handling.
    pub rare_classes: RareClasses,
    /// Store request text (false keeps only a salted hash + embedding).
    pub store_text: bool,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            audit_rate: 0.02,
            audit_rate_shadow: 0.10,
            audit_rate_elevated: 0.10,
            calib_fraction: 0.20,
            min_train_samples: 1000,
            min_samples_per_class: 50,
            min_calib_samples: 500,
            min_new_samples: 2000,
            keep_versions: 3,
            shadow_min_samples: 1000,
            ood_quantile: 0.99,
            ood_k: 10,
            ood_max_ref: 5000,
            max_train_samples: 50_000,
            max_calib_samples: 20_000,
            keep_local_rows: 10_000,
            drift_window: 500,
            drift_min_samples: 200,
            drift_margin: 0.0,
            drift_check_every: 20,
            mode: TaskMode::Auto,
            seed: 0,
            student_epochs: 2000,
            student_lr: 0.05,
            student_l2: 1e-6,
            student_patience: 4,
            val_fraction: 0.1,
            eval_every: 25,
            label_target: LabelTarget::Probs,
            importance_weighting: true,
            weight_power: 0.5,
            max_weight: 20.0,
            fit_headroom: 0.15,
            teacher_change: TeacherChangePolicy::Fallback,
            teacher_change_confirm: 20,
            rare_classes: RareClasses::Defer,
            store_text: true,
        }
    }
}

impl TaskConfig {
    /// The disagreement budget: the task tolerates at most `1 - agreement`
    /// probability mass of (answered and disagreed).
    pub fn budget(&self, target_agreement: f64) -> f64 {
        (1.0 - target_agreement).clamp(1e-6, 1.0)
    }
}

/// One choice task's learned surface: instructions plus the criteria map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSpec {
    /// Instructions JSON value (canonicalized on the wire).
    pub instructions: serde_json::Value,
    /// Criteria option name → description (sorted map on identity).
    pub criteria: std::collections::BTreeMap<String, Option<String>>,
}

impl TaskSpec {
    /// Sorted class labels (criteria keys). Order is stable across runs.
    pub fn classes(&self) -> Vec<String> {
        self.criteria.keys().cloned().collect()
    }

    /// Full 256-bit fingerprint: sha256 over canonical `{"c": criteria,
    /// "i": instructions}`. Key order in both maps is irrelevant.
    pub fn fingerprint(&self) -> String {
        let value = serde_json::json!({
            "c": self.criteria,
            "i": self.instructions,
        });
        let mut hasher = Sha256::new();
        hasher.update(canonical_json(&value).as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Short stable version tag (first 12 hex of the fingerprint).
    pub fn version(&self) -> String {
        self.fingerprint()[..12].to_owned()
    }

    /// The 20-hex store key for one task.
    pub fn task_key(&self, tenant: &str, model: &str) -> String {
        let value = serde_json::json!({
            "model": model,
            "question": self.fingerprint(),
            "tenant": tenant,
            "type": "choice",
        });
        let mut hasher = Sha256::new();
        hasher.update(canonical_json(&value).as_bytes());
        format!("{:x}", hasher.finalize())[..20].to_owned()
    }
}

/// Jev's own reported-confidence definition: linear rescale of the max
/// probability so uniform = 0 and fully peaked = 1.
pub fn peakedness(probabilities: &[f64]) -> f64 {
    let classes = probabilities.len();
    if classes < 2 {
        return 0.0;
    }
    let max = probabilities.iter().cloned().fold(0.0_f64, f64::max);
    ((classes as f64 * max - 1.0) / (classes as f64 - 1.0)).clamp(0.0, 1.0)
}

/// Why a request item was routed the way it was (engine-side reasons; the
/// proxy layer adds forward-only reasons such as `not_understood`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingReason {
    /// No production student yet.
    Bootstrap,
    /// Forced fallback after drift or a teacher change.
    Fallback,
    /// Random audit slice always answered by the teacher.
    Audit,
    /// Student answered within the calibrated policy.
    Confident,
    /// Student confidence below the calibrated threshold.
    LowConfidence,
    /// Embedding outside the calibrated distribution.
    Ood,
    /// Predicted label is deferred (rare class).
    RareClass,
}

impl RoutingReason {
    /// Stable wire/log token.
    pub fn as_str(&self) -> &'static str {
        match self {
            RoutingReason::Bootstrap => "bootstrap",
            RoutingReason::Fallback => "fallback",
            RoutingReason::Audit => "audit",
            RoutingReason::Confident => "confident",
            RoutingReason::LowConfidence => "low_confidence",
            RoutingReason::Ood => "ood",
            RoutingReason::RareClass => "rare_class",
        }
    }
}

/// Which process served the answer and what the row is used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    /// Teacher-only traffic before any production student.
    Bootstrap,
    /// Teacher-served audit slice (student output recorded alongside).
    Audit,
    /// Teacher-served traffic while forced fallback is active.
    Fallback,
    /// Answerable question forwarded because its request had to go upstream.
    CoDeferred,
    /// Locally served student answer (no teacher label).
    Student,
}

impl Channel {
    /// Stable store token.
    pub fn as_str(&self) -> &'static str {
        match self {
            Channel::Bootstrap => "bootstrap",
            Channel::Audit => "audit",
            Channel::Fallback => "fallback",
            Channel::CoDeferred => "co_deferred",
            Channel::Student => "student",
        }
    }

    /// Parse the store token back.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "bootstrap" => Some(Channel::Bootstrap),
            "audit" => Some(Channel::Audit),
            "fallback" => Some(Channel::Fallback),
            "co_deferred" => Some(Channel::CoDeferred),
            "student" => Some(Channel::Student),
            _ => None,
        }
    }

    /// Independent-and-identically-distributed channels: the only rows
    /// eligible for the calibration split. Co-deferred rows correlate with
    /// the rest of their request, so they always land in train.
    pub fn is_iid(self) -> bool {
        matches!(
            self,
            Channel::Bootstrap | Channel::Audit | Channel::Fallback
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn spec(instructions: serde_json::Value, classes: &[&str]) -> TaskSpec {
        TaskSpec {
            instructions,
            criteria: classes
                .iter()
                .map(|name| (name.to_string(), Some(format!("desc {name}"))))
                .collect(),
        }
    }

    #[test]
    fn fingerprint_is_order_insensitive_and_wording_sensitive() {
        let a = spec(json!("Which team?"), &["x", "y"]);
        let mut reordered = TaskSpec {
            instructions: json!("Which team?"),
            criteria: BTreeMap::new(),
        };
        reordered.criteria.insert("y".into(), Some("desc y".into()));
        reordered.criteria.insert("x".into(), Some("desc x".into()));
        assert_eq!(a.fingerprint(), reordered.fingerprint());
        assert_eq!(
            a.task_key("t", "jev-latest"),
            reordered.task_key("t", "jev-latest")
        );

        let different_instructions = spec(json!("Which team wins?"), &["x", "y"]);
        assert_ne!(a.fingerprint(), different_instructions.fingerprint());

        let mut different_description = TaskSpec {
            instructions: json!("Which team?"),
            criteria: BTreeMap::new(),
        };
        different_description
            .criteria
            .insert("x".into(), Some("other".into()));
        different_description
            .criteria
            .insert("y".into(), Some("desc y".into()));
        assert_ne!(a.fingerprint(), different_description.fingerprint());
    }

    #[test]
    fn task_key_separates_tenant_and_model() {
        let task = spec(json!("i"), &["x", "y"]);
        assert_ne!(task.task_key("a", "m"), task.task_key("b", "m"));
        assert_ne!(task.task_key("a", "m"), task.task_key("a", "n"));
        assert_eq!(task.task_key("a", "m").len(), 20);
    }

    #[test]
    fn peakedness_matches_jev_definition() {
        assert!((peakedness(&[0.5, 0.5]) - 0.0).abs() < 1e-12);
        assert!((peakedness(&[1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!((peakedness(&[0.75, 0.25]) - 0.5).abs() < 1e-12);
        assert_eq!(peakedness(&[1.0]), 0.0);
    }

    #[test]
    fn channel_iid_membership() {
        assert!(Channel::Bootstrap.is_iid());
        assert!(Channel::Audit.is_iid());
        assert!(Channel::Fallback.is_iid());
        assert!(!Channel::CoDeferred.is_iid());
        assert!(!Channel::Student.is_iid());
        assert_eq!(Channel::parse("co_deferred"), Some(Channel::CoDeferred));
    }
}
