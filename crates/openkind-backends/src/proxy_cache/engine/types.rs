//! Types, states, and data bundles for the proxy-cache task engine.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::super::calibrate::RoutingPolicy;
use super::super::ood::KnnOod;
use super::super::registry::VersionRegistry;
use super::super::store::{SampleRow, SampleStore};
use super::super::student::{FitReport, LinearStudent};
use super::super::task::{Channel, RoutingReason, TaskConfig, TaskMode, TaskSpec};

/// What the engine decided for one question item.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    /// Why this item routed this way.
    pub reason: RoutingReason,
    /// Serving channel for the eventual row.
    pub channel: Channel,
    /// The student's local answer when the item may be served locally.
    pub local: Option<LocalPrediction>,
    /// Student version that produced the answer (`student-v3`).
    pub student_version: Option<String>,
}

/// The student's answer for one item (pre-teacher).
#[derive(Debug, Clone)]
pub struct LocalPrediction {
    /// Argmax label.
    pub label: String,
    /// Full distribution aligned with the task's sorted classes.
    pub probabilities: Vec<f64>,
    /// Routing confidence (max probability).
    pub routing_confidence: f64,
    /// kNN gate score.
    pub ood: f64,
}

/// A recorded teacher answer completing one routed item.
#[derive(Debug, Clone)]
pub struct TeacherAnswer {
    /// Teacher's argmax label (already mapped onto this task's classes).
    pub label: String,
    /// Teacher's distribution aligned with the task's sorted classes.
    pub probabilities: Vec<f64>,
    /// Teacher's reported confidence.
    pub confidence: f64,
    /// Resolved teacher model from the upstream response (`jev-1.13.0`).
    pub model: String,
    /// Round-trip latency in milliseconds.
    pub latency_ms: f64,
}

/// Engine output of a maintenance tick.
#[derive(Debug, Default)]
pub struct TickOutcome {
    /// A fit was requested (the manager should schedule it).
    pub train_requested: bool,
    /// Lifecycle events to log.
    pub events: Vec<(String, serde_json::Value)>,
}

/// Production student state.
pub(super) struct Production {
    pub(super) version: String,
    pub(super) student: LinearStudent,
    pub(super) ood: KnnOod,
    pub(super) policy: RoutingPolicy,
    /// Coverage the candidate showed on its calibration rows (persisted in
    /// the version meta and restored on reload).
    pub(super) calib_coverage: f64,
}

/// Shadow candidate state.
#[derive(Clone)]
pub(super) struct ShadowCandidate {
    pub(super) version: String,
    pub(super) student: LinearStudent,
    pub(super) ood: KnnOod,
    pub(super) policy: RoutingPolicy,
    /// Candidate calibration counts (pooled into shadow judgement).
    pub(super) calib_accepted: usize,
    pub(super) calib_disagree: usize,
    pub(super) calib_coverage: f64,
}

/// One audit observation for drift monitoring: whether the student (as it
/// was at serving time) matched the teacher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AuditObservation {
    pub(super) agree: bool,
    pub(super) production_version: String,
}

/// Per-task engine state machine.
pub struct TaskEngine {
    /// 20-hex store key.
    pub key: String,
    /// Tenant label (`default` for shared tenancy).
    pub tenant: String,
    /// Requested model alias (`jev-latest`).
    pub model: String,
    /// Task identity (instructions + criteria).
    pub spec: TaskSpec,
    /// Configuration (defaults + operator overrides).
    pub config: TaskConfig,
    /// Disagreement budget derived from the target agreement.
    pub target_agreement: f64,
    /// Optional reported-confidence floor.
    pub confidence_floor: Option<f64>,

    pub(super) store: SampleStore,
    pub(super) versions: VersionRegistry,
    pub(super) classes: Vec<String>,
    pub(super) task_version: String,
    pub(super) rng: fastrand::Rng,

    pub(super) production: Option<Production>,
    pub(super) shadow: Option<ShadowCandidate>,
    pub(super) deferred_labels: Vec<String>,

    pub(super) mode: TaskMode,
    pub(super) forced_fallback: bool,
    pub(super) suspicious: bool,

    // Training bookkeeping.
    pub(super) teacher_answers_since_train: usize,
    pub(super) answers_at_last_failure: Option<(usize, usize)>,
    pub(super) training_in_flight: bool,
    /// A fit is wanted (readiness passed, retrain due, or an event asked for
    /// one). Cleared when a fit completes; re-armed by ticks and events.
    pub(super) needs_fit: bool,
    /// Recovery horizon: training data before this row id is discarded
    /// after a drift fallback (0 = full history).
    pub(super) last_train_id: i64,

    // Teacher lineage.
    pub(super) lineage_model: Option<String>,
    pub(super) lineage_candidate: Option<String>,
    pub(super) lineage_candidate_streak: usize,

    // Drift monitoring over the audit channel.
    pub(super) audit_window: VecDeque<AuditObservation>,
    pub(super) audit_since_check: usize,

    // Per-request split draw state (a seeded RNG re-seeded per request).
    pub(super) request_rng: Option<fastrand::Rng>,
}

/// Extra visibility for tests and status endpoints.
#[derive(Debug, Clone, Serialize)]
pub struct TaskStatus {
    /// Reported-confidence floor.
    pub confidence_floor: Option<f64>,
    /// Store key.
    pub key: String,
    /// Task version.
    pub task_version: String,
    /// Classes.
    pub classes: Vec<String>,
    /// Deferred labels.
    pub deferred_labels: Vec<String>,
    /// Production version.
    pub production_version: Option<String>,
    /// Confidence threshold.
    pub production_threshold: Option<f64>,
    /// OOD threshold.
    pub ood_threshold: Option<f64>,
    /// Expected coverage.
    pub expected_coverage: Option<f64>,
    /// Shadow version.
    pub shadow_version: Option<String>,
    /// Teacher lineage.
    pub teacher_model: Option<String>,
    /// Forced fallback flag.
    pub forced_fallback: bool,
    /// Suspicious flag.
    pub suspicious: bool,
    /// Mode.
    pub mode: TaskMode,
}

/// Data bundle for one fit, extracted under the engine lock.
pub struct FitInput {
    /// Newest teacher-labelled train rows.
    pub train_rows: Vec<SampleRow>,
    /// Teacher-labelled calibration rows (IID channels only).
    pub calib_rows: Vec<SampleRow>,
    /// Per-class train counts.
    pub label_counts: Vec<(String, i64)>,
    /// Recovery horizon (0 = full history).
    pub since_id: i64,
}

/// Result of one fit, ready to apply.
pub struct FitOutput {
    /// Fitted student.
    pub student: LinearStudent,
    /// Fitted gate.
    pub ood: KnnOod,
    /// Fitted policy.
    pub policy: RoutingPolicy,
    /// Whether the policy fit the budget.
    pub usable: bool,
    /// Rejection reason.
    pub reject_reason: Option<String>,
    /// Fit report.
    pub report: FitReport,
    /// Train rows used.
    pub n_train: usize,
    /// Calibration rows used.
    pub n_calib: usize,
    /// Rows the policy would answer on calibration data.
    pub calib_accepted: usize,
    /// Disagreements on calibration data.
    pub calib_disagree: usize,
    /// Per-class train counts.
    pub label_counts: Vec<(String, i64)>,
    /// Recovery horizon used.
    pub since_id: i64,
    /// Teacher lineage of the fit.
    pub teacher_model: String,
}
