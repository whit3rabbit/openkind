//! The per-task lifecycle loop: route → record → train → calibrate → shadow
//! → promote → monitor.
//!
//! One [`TaskEngine`] owns one task's store, production student, shadow
//! candidate, audit bookkeeping, teacher lineage, and drift state. The
//! engine never performs I/O on the upstream API and never embeds text —
//! the caller supplies embeddings and teacher answers; the engine decides
//! what may be answered locally and when to train.
//!
//! The guarantee: the probability that the student answers and disagrees
//! with the teacher stays at or below the task's budget at confidence
//! `1 - delta`, maintained by (a) calibration rows drawn only from IID
//! channels, (b) a fixed threshold grid tested strictest-first with a
//! Clopper–Pearson bound over all calibration rows, (c) shadow judgement at
//! the full budget pooled with the candidate's own calibration counts, and
//! (d) an always-on audit slice scored as served.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::calibrate::{self, CalibRow, RoutingPolicy};
use super::ood::KnnOod;
use super::registry::{
    write_json_atomic, RegistryIndex, VersionEntry, VersionMeta, VersionRegistry,
};
use super::store::{SampleRow, SampleStore, StoreCounts};
use super::student::{FitReport, LinearStudent};
use super::task::{
    Channel, LabelTarget, RareClasses, RoutingReason, TaskConfig, TaskMode, TaskSpec,
    TeacherChangePolicy,
};
use super::ProxyCacheError;

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
struct Production {
    version: String,
    student: LinearStudent,
    ood: KnnOod,
    policy: RoutingPolicy,
    /// Coverage the candidate showed on its calibration rows (persisted in
    /// the version meta and restored on reload).
    calib_coverage: f64,
}

/// Shadow candidate state.
#[derive(Clone)]
struct ShadowCandidate {
    version: String,
    student: LinearStudent,
    ood: KnnOod,
    policy: RoutingPolicy,
    /// Candidate calibration counts (pooled into shadow judgement).
    calib_accepted: usize,
    calib_disagree: usize,
    calib_coverage: f64,
}

/// One audit observation for drift monitoring: whether the student (as it
/// was at serving time) matched the teacher.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditObservation {
    agree: bool,
    production_version: String,
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

    store: SampleStore,
    versions: VersionRegistry,
    classes: Vec<String>,
    task_version: String,
    rng: fastrand::Rng,

    production: Option<Production>,
    shadow: Option<ShadowCandidate>,
    deferred_labels: Vec<String>,

    mode: TaskMode,
    forced_fallback: bool,
    suspicious: bool,

    // Training bookkeeping.
    teacher_answers_since_train: usize,
    answers_at_last_failure: Option<(usize, usize)>,
    training_in_flight: bool,
    /// A fit is wanted (readiness passed, retrain due, or an event asked for
    /// one). Cleared when a fit completes; re-armed by ticks and events.
    needs_fit: bool,
    /// Recovery horizon: training data before this row id is discarded
    /// after a drift fallback (0 = full history).
    last_train_id: i64,

    // Teacher lineage.
    lineage_model: Option<String>,
    lineage_candidate: Option<String>,
    lineage_candidate_streak: usize,

    // Drift monitoring over the audit channel.
    audit_window: VecDeque<AuditObservation>,
    audit_since_check: usize,

    // Per-request split draw state (a seeded RNG re-seeded per request).
    request_rng: Option<fastrand::Rng>,
}

impl TaskEngine {
    /// Create a brand-new task engine (writes task.json beside the store).
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        dir: &std::path::Path,
        key: String,
        tenant: String,
        model: String,
        spec: TaskSpec,
        config: TaskConfig,
        target_agreement: f64,
        confidence_floor: Option<f64>,
    ) -> Result<Self, ProxyCacheError> {
        std::fs::create_dir_all(dir).map_err(|error| {
            ProxyCacheError::Store(format!("create {}: {error}", dir.display()))
        })?;
        let store = SampleStore::open(&dir.join("samples.sqlite"))?;
        let versions = VersionRegistry::new(dir.join("versions"));
        let task_version = spec.version();
        let seed = split_seed(&config.seed, &task_version);
        let engine = Self {
            key: key.clone(),
            tenant: tenant.clone(),
            model: model.clone(),
            spec: spec.clone(),
            config: config.clone(),
            target_agreement,
            confidence_floor,
            store,
            versions,
            classes: spec.classes(),
            task_version: task_version.clone(),
            rng: fastrand::Rng::with_seed(seed),
            production: None,
            shadow: None,
            deferred_labels: Vec::new(),
            mode: config.mode,
            forced_fallback: false,
            suspicious: false,
            teacher_answers_since_train: 0,
            answers_at_last_failure: None,
            training_in_flight: false,
            needs_fit: false,
            last_train_id: 0,
            lineage_model: None,
            lineage_candidate: None,
            lineage_candidate_streak: 0,
            audit_window: VecDeque::new(),
            audit_since_check: 0,
            request_rng: None,
        };
        let task_info = serde_json::json!({
            "key": key,
            "tenant": tenant,
            "instructions": spec.instructions,
            "criteria": spec.criteria,
            "target_agreement": target_agreement,
            "confidence_floor": confidence_floor,
            "model": model,
            "task_version": task_version,
            "mode": config.mode,
        });
        write_json_atomic(&dir.join("task.json"), &task_info)?;
        engine.store.insert_event(
            "created",
            &serde_json::json!({"task_version": task_version, "model": model}),
        )?;
        Ok(engine)
    }

    /// Restore an engine from its task directory.
    pub fn load(
        dir: &std::path::Path,
        key: String,
        config: TaskConfig,
        target_agreement: f64,
        confidence_floor: Option<f64>,
    ) -> Result<Self, ProxyCacheError> {
        let info_path = dir.join("task.json");
        let info_text = std::fs::read_to_string(&info_path).map_err(|error| {
            ProxyCacheError::Store(format!("read {}: {error}", info_path.display()))
        })?;
        let info: serde_json::Value = serde_json::from_str(&info_text)
            .map_err(|error| ProxyCacheError::Store(format!("decode task.json: {error}")))?;
        let spec = TaskSpec {
            instructions: info
                .get("instructions")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
            criteria: serde_json::from_value(info.get("criteria").cloned().unwrap_or_default())
                .map_err(|error| ProxyCacheError::Store(format!("decode criteria: {error}")))?,
        };
        let store = SampleStore::open(&dir.join("samples.sqlite"))?;
        let versions = VersionRegistry::new(dir.join("versions"));
        let task_version = spec.version();
        let seed = split_seed(&config.seed, &task_version);

        // Restore the production version (and any interrupted shadow).
        let index = versions.index()?;
        let mut production = None;
        if let Some(name) = &index.production {
            if let Ok((student, ood, policy, meta)) = versions.load_version(name) {
                production = Some(Production {
                    version: name.clone(),
                    student,
                    ood,
                    policy,
                    calib_coverage: meta.calib_coverage,
                });
            }
        }
        let mut shadow = None;
        if let Some(entry) = index
            .versions
            .iter()
            .rev()
            .find(|entry| entry.state == "shadow")
        {
            if let Ok((student, ood, policy, meta)) = versions.load_version(&entry.name) {
                shadow = Some(ShadowCandidate {
                    version: entry.name.clone(),
                    student,
                    ood,
                    policy,
                    calib_accepted: meta.calib_accepted,
                    calib_disagree: meta.calib_disagree,
                    calib_coverage: meta.calib_coverage,
                });
            }
        }

        let persisted_mode = info
            .get("mode")
            .cloned()
            .map(serde_json::from_value::<TaskMode>)
            .unwrap_or(Ok(TaskMode::Auto))
            .unwrap_or(TaskMode::Auto);

        // A persisted `fallback` event means the forced fallback survives
        // restarts until a passing candidate (or an operator) clears it.
        let recent = store.recent_events(200)?;
        let forced_fallback = recent.iter().any(|(kind, _)| kind == "fallback")
            && !recent.iter().any(|(kind, _)| kind == "fallback_cleared");

        let mut engine = Self {
            key,
            tenant: info
                .get("tenant")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("default")
                .to_owned(),
            model: info
                .get("model")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            spec,
            config,
            target_agreement,
            confidence_floor,
            store,
            versions,
            classes: Vec::new(),
            task_version,
            rng: fastrand::Rng::with_seed(seed),
            production,
            shadow,
            deferred_labels: Vec::new(),
            mode: persisted_mode,
            forced_fallback,
            suspicious: false,
            teacher_answers_since_train: 0,
            answers_at_last_failure: None,
            training_in_flight: false,
            needs_fit: false,
            last_train_id: 0,
            lineage_model: None,
            lineage_candidate: None,
            lineage_candidate_streak: 0,
            audit_window: VecDeque::new(),
            audit_since_check: 0,
            request_rng: None,
        };
        engine.classes = engine.spec.classes();
        // Lineage continues from the newest teacher-labelled row.
        engine.lineage_model = engine.store.latest_teacher_model(&engine.task_version)?;
        Ok(engine)
    }

    /// Begin one request: seed the per-request split draw. Every item of a
    /// request shares one draw so siblings never straddle the split.
    pub fn begin_request(&mut self) {
        let seed = self.rng.u64(..)
            ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0);
        self.request_rng = Some(fastrand::Rng::with_seed(seed));
    }

    fn split_for(&mut self, channel: Channel) -> &'static str {
        if !channel.is_iid() {
            return "train";
        }
        let draw = self
            .request_rng
            .as_mut()
            .map(|rng| rng.f64())
            .unwrap_or_else(|| self.rng.f64());
        if draw < self.config.calib_fraction {
            "calib"
        } else {
            "train"
        }
    }

    fn effective_audit_rate(&self) -> f64 {
        let mut rate = self.config.audit_rate;
        if self.shadow.is_some() {
            rate = rate.max(self.config.audit_rate_shadow);
        }
        if self.suspicious {
            rate = rate.max(self.config.audit_rate_elevated);
        }
        rate
    }

    /// Route one item. `embedding` is the request state's embedding.
    pub fn route(&mut self, embedding: &[f32]) -> RouteDecision {
        let teacher_only = match self.mode {
            TaskMode::TeacherOnly => true,
            TaskMode::Auto => self.forced_fallback,
        };
        if teacher_only {
            let (reason, channel) = if self.production.is_some() {
                (RoutingReason::Fallback, Channel::Fallback)
            } else {
                (RoutingReason::Bootstrap, Channel::Bootstrap)
            };
            return RouteDecision {
                reason,
                channel,
                local: None,
                student_version: None,
            };
        }
        if self.production.is_none() {
            return RouteDecision {
                reason: RoutingReason::Bootstrap,
                channel: Channel::Bootstrap,
                local: None,
                student_version: None,
            };
        }
        let production = self.production.as_ref().expect("production checked");
        let draw = self.rng.f64();
        if draw < self.effective_audit_rate() {
            // Audit: the teacher serves, but the student's would-be answer is
            // recorded as-is and scored against the teacher for drift.
            // Callers must never serve an audit item locally.
            let local = self.score_with_student(production, embedding);
            return RouteDecision {
                reason: RoutingReason::Audit,
                channel: Channel::Audit,
                local: Some(local),
                student_version: Some(production.version.clone()),
            };
        }
        let local = self.score_with_student(production, embedding);
        let accept = production.policy.usable()
            && local.routing_confidence
                >= production.policy.conf_threshold.unwrap_or(f64::INFINITY)
            && local.ood <= production.policy.ood_threshold
            && !self
                .deferred_labels
                .iter()
                .any(|label| label == &local.label);
        let reason = if accept {
            RoutingReason::Confident
        } else if local.ood > production.policy.ood_threshold {
            RoutingReason::Ood
        } else if self
            .deferred_labels
            .iter()
            .any(|label| label == &local.label)
        {
            RoutingReason::RareClass
        } else {
            RoutingReason::LowConfidence
        };
        RouteDecision {
            reason,
            channel: Channel::Student,
            local: if accept { Some(local) } else { None },
            student_version: Some(production.version.clone()),
        }
    }

    fn score_with_student(&self, production: &Production, embedding: &[f32]) -> LocalPrediction {
        let probs = production.student.predict_proba(embedding);
        let best_index = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let label = self.classes.get(best_index).cloned().unwrap_or_default();
        LocalPrediction {
            label,
            routing_confidence: probs.iter().cloned().fold(0.0_f64, f64::max),
            probabilities: probs,
            ood: production.ood.score(embedding),
        }
    }

    /// Record one teacher answer for a routed item and observe the teacher
    /// lineage. Returns lifecycle events for logging.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_teacher(
        &mut self,
        decision: &RouteDecision,
        embedding: &[f32],
        state_text: &str,
        state_type: &str,
        encoder_id: &str,
        answer: &TeacherAnswer,
    ) -> Result<Vec<(String, serde_json::Value)>, ProxyCacheError> {
        let mut events = Vec::new();
        // Tempered Horvitz-Thompson: audit rows are up-weighted so training
        // tracks traffic rather than the hard audit slice.
        let weight: f64 = if self.config.importance_weighting && decision.channel == Channel::Audit
        {
            (1.0_f64 / self.effective_audit_rate())
                .powf(self.config.weight_power)
                .min(self.config.max_weight)
        } else {
            1.0
        };
        let split = self.split_for(decision.channel).to_owned();
        let (shadow_version, shadow_label, shadow_confidence) = self.shadow_scores(embedding);
        let row = SampleRow {
            id: 0,
            task_version: self.task_version.clone(),
            split,
            channel: decision.channel.as_str().into(),
            routing_reason: decision.reason.as_str().into(),
            served_by: "teacher".into(),
            text: if self.config.store_text {
                state_text.to_owned()
            } else {
                String::new()
            },
            text_hash: super::text::text_hash(state_text, self.task_version.as_bytes()),
            encoder_id: encoder_id.to_owned(),
            embedding: embedding.to_vec(),
            state_type: state_type.to_owned(),
            weight,
            teacher_label: Some(answer.label.clone()),
            teacher_probs: Some(probs_json(&self.classes, &answer.probabilities)),
            teacher_confidence: Some(answer.confidence),
            teacher_model: Some(answer.model.clone()),
            teacher_latency_ms: Some(answer.latency_ms),
            student_version: decision.student_version.clone(),
            student_label: decision.local.as_ref().map(|local| local.label.clone()),
            student_probs: decision
                .local
                .as_ref()
                .map(|local| probs_json(&self.classes, &local.probabilities)),
            student_confidence: decision
                .local
                .as_ref()
                .map(|local| local.routing_confidence),
            ood_score: decision.local.as_ref().map(|local| local.ood),
            shadow_version,
            shadow_label,
            shadow_confidence,
        };
        self.store.insert(&row)?;
        self.teacher_answers_since_train += 1;
        events.extend(self.observe_teacher_lineage(&answer.model)?);
        if decision.channel == Channel::Audit {
            if let Some(local) = &decision.local {
                let agree = !calibrate::disagrees(
                    &local.label,
                    &answer.label,
                    answer.confidence,
                    self.confidence_floor,
                );
                self.record_audit_observation(agree);
            }
        }
        Ok(events)
    }

    /// Score one embedding with the current shadow candidate, if any.
    fn shadow_scores(&self, embedding: &[f32]) -> (Option<String>, Option<String>, Option<f64>) {
        let Some(candidate) = &self.shadow else {
            return (None, None, None);
        };
        let probs = candidate.student.predict_proba(embedding);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index);
        let label = best.and_then(|index| self.classes.get(index).cloned());
        let confidence = probs.iter().cloned().fold(0.0_f64, f64::max);
        (Some(candidate.version.clone()), label, Some(confidence))
    }

    /// Record one locally served answer (channel `student`).
    #[allow(clippy::too_many_arguments)]
    pub fn observe_student_served(
        &mut self,
        decision: &RouteDecision,
        embedding: &[f32],
        state_text: &str,
        state_type: &str,
        encoder_id: &str,
    ) -> Result<(), ProxyCacheError> {
        let Some(local) = &decision.local else {
            return Ok(());
        };
        let row = SampleRow {
            id: 0,
            task_version: self.task_version.clone(),
            split: "train".into(),
            channel: Channel::Student.as_str().into(),
            routing_reason: decision.reason.as_str().into(),
            served_by: "student".into(),
            text: if self.config.store_text {
                state_text.to_owned()
            } else {
                String::new()
            },
            text_hash: super::text::text_hash(state_text, self.task_version.as_bytes()),
            encoder_id: encoder_id.to_owned(),
            embedding: embedding.to_vec(),
            state_type: state_type.to_owned(),
            weight: 1.0,
            teacher_label: None,
            teacher_probs: None,
            teacher_confidence: None,
            teacher_model: None,
            teacher_latency_ms: None,
            student_version: decision.student_version.clone(),
            student_label: Some(local.label.clone()),
            student_probs: Some(probs_json(&self.classes, &local.probabilities)),
            student_confidence: Some(local.routing_confidence),
            ood_score: Some(local.ood),
            shadow_version: None,
            shadow_label: None,
            shadow_confidence: None,
        };
        self.store.insert(&row)?;
        Ok(())
    }

    /// Observe the teacher lineage: a different resolved model must answer
    /// `teacher_change_confirm` times consecutively before the lineage
    /// switches (no flapping). Returns events.
    fn observe_teacher_lineage(
        &mut self,
        model: &str,
    ) -> Result<Vec<(String, serde_json::Value)>, ProxyCacheError> {
        let mut events = Vec::new();
        if self.lineage_candidate.as_deref() == Some(model) {
            self.lineage_candidate_streak += 1;
        } else if self.lineage_model.as_deref() == Some(model) {
            self.lineage_candidate = None;
            self.lineage_candidate_streak = 0;
        } else if self.lineage_model.is_none() {
            self.lineage_model = Some(model.to_owned());
            self.lineage_candidate = None;
            self.lineage_candidate_streak = 0;
            return Ok(events);
        } else {
            self.lineage_candidate = Some(model.to_owned());
            self.lineage_candidate_streak = 1;
        }
        if self.lineage_candidate_streak >= self.config.teacher_change_confirm.max(1) {
            let old = self.lineage_model.clone().unwrap_or_default();
            self.lineage_model = self.lineage_candidate.clone();
            self.lineage_candidate = None;
            self.lineage_candidate_streak = 0;
            events.push((
                "teacher_changed".into(),
                serde_json::json!({"from": old, "to": model}),
            ));
            // The old lineage's shadow can never pass for the new one.
            if self.shadow.is_some() {
                self.drop_shadow("teacher_changed")?;
            }
            match self.config.teacher_change {
                TeacherChangePolicy::Fallback => {
                    if self.production.is_some() && !self.forced_fallback {
                        self.forced_fallback = true;
                        self.store.insert_event(
                            "fallback",
                            &serde_json::json!({"reason": "teacher_change", "from": old, "to": model}),
                        )?;
                    }
                }
                TeacherChangePolicy::Audit => {
                    self.suspicious = true;
                }
            }
            self.request_retrain();
            self.needs_fit = true;
        }
        Ok(events)
    }

    fn record_audit_observation(&mut self, agree: bool) {
        let version = self
            .production
            .as_ref()
            .map(|production| production.version.clone())
            .unwrap_or_default();
        self.audit_window.push_back(AuditObservation {
            agree,
            production_version: version,
        });
        while self.audit_window.len() > self.config.drift_window.max(1) {
            self.audit_window.pop_front();
        }
        self.audit_since_check += 1;
    }

    /// Cheap maintenance run on the request path: drift monitoring, shadow
    /// judgement, and fit requests.
    pub fn tick(&mut self) -> Result<TickOutcome, ProxyCacheError> {
        let mut outcome = TickOutcome::default();

        // Drift monitoring over the audit channel, scored as served.
        if self.production.is_some()
            && self.audit_since_check >= self.config.drift_check_every.max(1)
        {
            self.audit_since_check = 0;
            let current_version = self
                .production
                .as_ref()
                .map(|production| production.version.clone())
                .unwrap_or_default();
            let observations: Vec<&AuditObservation> = self
                .audit_window
                .iter()
                .filter(|observation| observation.production_version == current_version)
                .collect();
            let n = observations.len();
            if n >= self.config.drift_min_samples {
                let target = self.target_agreement;
                let disagreements = observations
                    .iter()
                    .filter(|observation| !observation.agree)
                    .count();
                let point = 1.0 - (n - disagreements) as f64 / n as f64;
                if point < target && !self.suspicious {
                    self.suspicious = true;
                    self.store.insert_event(
                        "suspicious",
                        &serde_json::json!({"agreement": point, "target": target}),
                    )?;
                    outcome.events.push((
                        "suspicious".into(),
                        serde_json::json!({"agreement": point, "n": n}),
                    ));
                    self.request_retrain();
                    self.needs_fit = true;
                }
                let upper = calibrate::clopper_pearson_upper(
                    disagreements,
                    n,
                    1.0 - self.config.confidence,
                );
                if upper < target - self.config.drift_margin {
                    // Broken: force fallback and restart training data here.
                    self.forced_fallback = true;
                    self.suspicious = false;
                    self.last_train_id = self.store.counts(&self.task_version)?.max_id;
                    if self.shadow.is_some() {
                        self.drop_shadow("drift_broken")?;
                    }
                    self.store.insert_event(
                        "fallback",
                        &serde_json::json!({
                            "reason": "drift",
                            "agreement_ub": upper,
                            "target": target,
                            "since_id": self.last_train_id,
                        }),
                    )?;
                    outcome.events.push((
                        "fallback".into(),
                        serde_json::json!({"reason": "drift", "since_id": self.last_train_id}),
                    ));
                    self.request_retrain();
                    self.needs_fit = true;
                }
            }
        }

        // Shadow judgement.
        if self.shadow.is_some() {
            if let Some(events) = self.judge_shadow()? {
                outcome.events.extend(events);
                return Ok(outcome);
            }
        }

        // Readiness for the first fit: request exactly one fit and let the
        // candidate run through shadow before considering another.
        if self.production.is_none() {
            if self.shadow.is_some() || self.needs_fit {
                if self.needs_fit && !self.training_in_flight {
                    outcome.train_requested = true;
                }
                return Ok(outcome);
            }
            let counts = self.store.counts(&self.task_version)?;
            if counts.train_labelled < self.config.min_train_samples as i64
                || counts.calib_labelled < self.config.min_calib_samples as i64
            {
                return Ok(outcome);
            }
            if !self.check_class_readiness(&counts)? {
                return Ok(outcome);
            }
            self.request_retrain();
            self.needs_fit = true;
            outcome.train_requested = true;
            return Ok(outcome);
        }

        // Retrain when enough new teacher answers accumulated and no
        // candidate is being judged.
        if self.shadow.is_none() && self.teacher_answers_since_train >= self.config.min_new_samples
        {
            self.request_retrain();
            self.needs_fit = true;
        }
        if self.needs_fit && !self.training_in_flight {
            outcome.train_requested = true;
        }
        Ok(outcome)
    }

    fn check_class_readiness(&mut self, _counts: &StoreCounts) -> Result<bool, ProxyCacheError> {
        let label_counts = self.store.train_label_counts(&self.task_version)?;
        match self.config.rare_classes {
            RareClasses::Wait => {
                let ready = !label_counts.is_empty()
                    && label_counts
                        .iter()
                        .all(|(_, count)| *count >= self.config.min_samples_per_class as i64);
                Ok(ready)
            }
            RareClasses::Defer => {
                // At least two classes must clear the bar; the rest defer.
                let sufficient: Vec<String> = label_counts
                    .iter()
                    .filter(|(_, count)| *count >= self.config.min_samples_per_class as i64)
                    .map(|(label, _)| label.clone())
                    .collect();
                if sufficient.len() < 2 {
                    return Ok(false);
                }
                let deferred: Vec<String> = self
                    .classes
                    .iter()
                    .filter(|class| !sufficient.contains(class))
                    .cloned()
                    .collect();
                if deferred != self.deferred_labels {
                    self.deferred_labels = deferred.clone();
                    self.store.insert_event(
                        "deferred_labels",
                        &serde_json::json!({"labels": deferred}),
                    )?;
                }
                Ok(true)
            }
        }
    }

    fn request_retrain(&mut self) {
        self.teacher_answers_since_train = 0;
    }

    /// Prepare the fit job data (runs under the engine lock; the heavy fit
    /// itself runs outside it via [`TaskEngine::run_fit`]).
    pub fn prepare_fit(&mut self) -> Result<Option<FitInput>, ProxyCacheError> {
        if self.training_in_flight || !self.needs_fit {
            return Ok(None);
        }
        let counts = self.store.counts(&self.task_version)?;
        if counts.train_labelled < self.config.min_train_samples as i64
            || counts.calib_labelled < self.config.min_calib_samples as i64
        {
            return Ok(None);
        }
        // Backoff after a failed candidate: wait for 25% more teacher answers
        // (capped at min_new_samples) before retrying.
        if let Some((at_failure, required)) = self.answers_at_last_failure {
            let current = self.teacher_answers_since_train;
            if current < required && (counts.teacher_labelled as usize) <= at_failure {
                return Ok(None);
            }
        }
        let (train_rows, calib_rows) = self.store.labelled_rows(
            &self.task_version,
            self.config.max_train_samples,
            self.config.max_calib_samples,
        )?;
        if train_rows.len() < self.config.min_train_samples
            || calib_rows.len() < self.config.min_calib_samples
        {
            return Ok(None);
        }
        // Recovery training after a drift fallback only uses rows from the
        // recovery horizon onward.
        let train_rows: Vec<SampleRow> = if self.forced_fallback && self.last_train_id > 0 {
            train_rows
                .into_iter()
                .filter(|row| row.id >= self.last_train_id)
                .collect()
        } else {
            train_rows
        };
        if train_rows.len() < self.config.min_train_samples {
            return Ok(None);
        }
        let label_counts = self.store.train_label_counts(&self.task_version)?;
        let since_id = if self.forced_fallback && self.last_train_id > 0 {
            self.last_train_id
        } else {
            0
        };
        self.training_in_flight = true;
        Ok(Some(FitInput {
            train_rows,
            calib_rows,
            label_counts,
            since_id,
        }))
    }

    /// Run the fit for a prepared job (CPU-heavy; it only reads immutable
    /// task state, so holding the engine lock is fine).
    pub fn run_fit(&self, input: &FitInput) -> Result<FitOutput, ProxyCacheError> {
        let dim = input
            .train_rows
            .iter()
            .find_map(|row| {
                if row.embedding.is_empty() {
                    None
                } else {
                    Some(row.embedding.len())
                }
            })
            .unwrap_or(1);
        let class_index =
            |label: &str| -> Option<usize> { self.classes.iter().position(|class| class == label) };

        // Train matrix.
        let usable: Vec<&SampleRow> = input
            .train_rows
            .iter()
            .filter(|row| class_index(row.teacher_label.as_deref().unwrap_or("")).is_some())
            .collect();
        let n = usable.len();
        if n < self.config.min_train_samples {
            return Err(ProxyCacheError::Contract(format!(
                "not enough labelled train rows for the task's classes ({n})"
            )));
        }
        let mut xs = ndarray::Array2::<f64>::zeros((n, dim));
        let mut targets = ndarray::Array2::<f64>::zeros((n, self.classes.len()));
        let mut weights = vec![1.0_f64; n];
        for (row_index, row) in usable.iter().enumerate() {
            let label =
                class_index(row.teacher_label.as_deref().unwrap_or("")).expect("filtered above");
            for (dim_index, &value) in row.embedding.iter().enumerate().take(dim) {
                xs[[row_index, dim_index]] = value as f64;
            }
            let probs: Vec<f64> = row
                .teacher_probs
                .as_deref()
                .and_then(|text| {
                    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(text).ok()
                })
                .map(|map| {
                    self.classes
                        .iter()
                        .map(|class| {
                            map.get(class)
                                .and_then(serde_json::Value::as_f64)
                                .unwrap_or(0.0)
                        })
                        .collect()
                })
                .unwrap_or_else(|| {
                    let mut one_hot = vec![0.0; self.classes.len()];
                    one_hot[label] = 1.0;
                    one_hot
                });
            let normalized: Vec<f64> = match self.config.label_target {
                LabelTarget::Probs => {
                    let sum: f64 = probs.iter().sum();
                    if sum > 1e-9 {
                        probs.iter().map(|p| p / sum).collect()
                    } else {
                        probs
                    }
                }
                LabelTarget::Hard => {
                    let mut one_hot = vec![0.0; self.classes.len()];
                    one_hot[label] = 1.0;
                    one_hot
                }
            };
            for (index, value) in normalized.iter().enumerate() {
                targets[[row_index, index]] = *value;
            }
            weights[row_index] = row.weight.max(0.0);
        }

        let (student, report) = LinearStudent::fit(
            &xs,
            &targets,
            &weights,
            self.config.student_lr,
            self.config.student_l2,
            self.config.student_epochs,
            self.config.student_patience,
            self.config.val_fraction,
            self.config.eval_every,
            self.config.seed,
        )?;

        // OOD reference from the training embeddings, stratified by class.
        let rows_for_ood: Vec<(&[f32], usize)> = usable
            .iter()
            .map(|row| {
                (
                    row.embedding.as_slice(),
                    class_index(row.teacher_label.as_deref().unwrap_or("")).unwrap_or(0),
                )
            })
            .collect();
        let ood = KnnOod::fit(
            &rows_for_ood,
            dim,
            self.config.ood_k,
            self.config.ood_quantile,
            self.config.ood_max_ref,
        );

        // Score calibration rows with the candidate.
        let mut calib = Vec::with_capacity(input.calib_rows.len());
        let mut calib_scores = Vec::with_capacity(input.calib_rows.len());
        let mut calib_accepted = 0usize;
        let mut calib_disagree = 0usize;
        for row in &input.calib_rows {
            if class_index(row.teacher_label.as_deref().unwrap_or("")).is_none() {
                continue;
            }
            let probs = student.predict_proba(&row.embedding);
            let best = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);
            let confidence = probs.iter().cloned().fold(0.0_f64, f64::max);
            let ood_score = ood.score(&row.embedding);
            let student_label = self.classes.get(best).cloned().unwrap_or_default();
            calib.push(CalibRow {
                student_confidence: confidence,
                student_label: student_label.clone(),
                teacher_label: row.teacher_label.clone().unwrap_or_default(),
                teacher_confidence: row.teacher_confidence.unwrap_or(1.0),
            });
            calib_scores.push(ood_score);
            if ood_score <= ood.threshold()
                && !self
                    .deferred_labels
                    .iter()
                    .any(|label| label == &student_label)
            {
                calib_accepted += 1;
            }
            if calibrate::disagrees(
                &student_label,
                row.teacher_label.as_deref().unwrap_or(""),
                row.teacher_confidence.unwrap_or(1.0),
                self.confidence_floor,
            ) {
                calib_disagree += 1;
            }
        }

        let budget = self.config.budget(self.target_agreement);
        let outcome = calibrate::fit_policy(
            &calib,
            &calib_scores,
            ood.threshold(),
            self.classes.len(),
            budget,
            self.config.fit_headroom,
            1.0 - self.config.confidence,
            &self.deferred_labels,
            self.confidence_floor,
        )?;

        Ok(FitOutput {
            student,
            ood,
            policy: outcome.policy,
            usable: outcome.usable,
            reject_reason: outcome.reason.map(str::to_owned),
            report,
            n_train: n,
            n_calib: calib.len(),
            calib_accepted,
            calib_disagree,
            label_counts: input.label_counts.clone(),
            since_id: input.since_id,
            teacher_model: self
                .lineage_model
                .clone()
                .unwrap_or_else(|| "unknown".into()),
        })
    }

    /// Apply a finished fit: save the version, enter shadow (or record a
    /// rejection). Returns events.
    pub fn apply_fit(
        &mut self,
        output: FitOutput,
    ) -> Result<Vec<(String, serde_json::Value)>, ProxyCacheError> {
        self.training_in_flight = false;
        self.needs_fit = false;
        let mut events = Vec::new();
        let number = self.versions.next_version_number()?;
        let name = format!("student-v{number}");
        let counts = self.store.counts(&self.task_version)?;
        if !output.usable {
            let mut index = self.versions.index()?;
            index.versions.push(VersionEntry {
                name: name.clone(),
                state: "rejected".into(),
                created: now(),
            });
            self.versions.write_index(&index)?;
            self.store.insert_event(
                "rejected",
                &serde_json::json!({"version": name, "reason": output.reject_reason}),
            )?;
            events.push((
                "rejected".into(),
                serde_json::json!({"version": name, "reason": output.reject_reason}),
            ));
            // Backoff: wait for 25% more teacher answers before retrying.
            let extra = ((self.config.min_new_samples as f64 * 0.25).ceil() as usize)
                .min(self.config.min_new_samples);
            self.answers_at_last_failure = Some((counts.teacher_labelled as usize, extra));
            return Ok(events);
        }

        let candidate_coverage = if output.n_calib > 0 {
            output.calib_accepted as f64 / output.n_calib as f64
        } else {
            0.0
        };
        let meta = VersionMeta {
            name: name.clone(),
            n_train: output.n_train,
            n_calib: output.n_calib,
            train_loss: output.report.train_loss,
            epochs: output.report.epochs,
            train_label_counts: output.label_counts.clone(),
            encoder_id: self.current_encoder_id(),
            teacher_model: output.teacher_model.clone(),
            confidence_floor: self.confidence_floor,
            target_agreement: self.target_agreement,
            trained_to_id: counts.max_id,
            calib_accepted: output.calib_accepted,
            calib_disagree: output.calib_disagree,
            calib_coverage: candidate_coverage,
            config: serde_json::to_value(&self.config).unwrap_or(serde_json::Value::Null),
            since_id: output.since_id,
        };
        self.versions.save_version(
            &name,
            &output.student,
            &output.ood,
            &output.policy,
            &meta,
            &output.report,
        )?;
        let mut index = self.versions.index()?;
        index.versions.push(VersionEntry {
            name: name.clone(),
            state: "shadow".into(),
            created: now(),
        });
        self.versions.write_index(&index)?;
        self.shadow = Some(ShadowCandidate {
            version: name.clone(),
            student: output.student,
            ood: output.ood,
            policy: output.policy,
            calib_accepted: output.calib_accepted,
            calib_disagree: output.calib_disagree,
            calib_coverage: candidate_coverage,
        });
        self.answers_at_last_failure = None;
        self.store.insert_event(
            "training_queued",
            &serde_json::json!({"version": name, "state": "shadow"}),
        )?;
        events.push(("shadow".into(), serde_json::json!({"version": name})));
        Ok(events)
    }

    fn current_encoder_id(&self) -> String {
        // The lineage is whatever the newest teacher-labelled rows carry.
        self.store
            .labelled_rows(&self.task_version, 1, 0)
            .ok()
            .and_then(|(train, _)| train.first().map(|row| row.encoder_id.clone()))
            .unwrap_or_default()
    }

    /// Judge the shadow candidate when enough shadow rows exist. Returns
    /// events when a decision was reached.
    ///
    /// The guarantee: shadow rows are scored as the candidate would have
    /// served them, pooled with the candidate's own calibration counts, and
    /// tested against the FULL budget with a Clopper–Pearson bound;
    /// coverage must not fall below 95% of production's calibrated coverage.
    fn judge_shadow(
        &mut self,
    ) -> Result<Option<Vec<(String, serde_json::Value)>>, ProxyCacheError> {
        let Some(shadow) = self.shadow.clone() else {
            return Ok(None);
        };
        let shadow_rows = self.store.shadow_rows(
            &self.task_version,
            &shadow.version,
            self.config.shadow_min_samples * 2,
        )?;
        if shadow_rows.len() < self.config.shadow_min_samples {
            return Ok(None);
        }
        let mut accepted = 0usize;
        let mut disagreed = 0usize;
        for row in &shadow_rows {
            let Some(student_label) = &row.shadow_label else {
                continue;
            };
            let Some(teacher_label) = &row.teacher_label else {
                continue;
            };
            let confidence = row.shadow_confidence.unwrap_or(0.0);
            let ood = shadow.ood.score(&row.embedding);
            let accepted_here = ood <= shadow.ood.threshold()
                && confidence >= shadow.policy.conf_threshold.unwrap_or(f64::INFINITY)
                && !self
                    .deferred_labels
                    .iter()
                    .any(|label| label == student_label);
            if !accepted_here {
                continue;
            }
            accepted += 1;
            if calibrate::disagrees(
                student_label,
                teacher_label,
                row.teacher_confidence.unwrap_or(1.0),
                self.confidence_floor,
            ) {
                disagreed += 1;
            }
        }

        // Pool shadow and calibration counts; judge at the full budget.
        let n = accepted + shadow.calib_accepted;
        let k = disagreed + shadow.calib_disagree;
        let delta = 1.0 - self.config.confidence;
        let upper = calibrate::clopper_pearson_upper(k, n, delta);
        let budget = self.config.budget(self.target_agreement);
        let production_coverage = self
            .production
            .as_ref()
            .map(|production| production.calib_coverage)
            .unwrap_or(0.0);
        let candidate_coverage = shadow.calib_coverage;
        let coverage_floor = if self.production.is_some() {
            0.95 * production_coverage
        } else {
            0.0
        };

        let mut events = Vec::new();
        if upper <= budget && candidate_coverage >= coverage_floor {
            self.promote(&shadow.version)?;
            events.push((
                "promoted".into(),
                serde_json::json!({
                    "version": shadow.version,
                    "accepted": accepted,
                    "disagreed": disagreed,
                    "ub": upper,
                    "budget": budget,
                }),
            ));
        } else {
            self.reject_shadow(&shadow.version)?;
            events.push((
                "rejected".into(),
                serde_json::json!({
                    "version": shadow.version,
                    "ub": upper,
                    "budget": budget,
                    "coverage": candidate_coverage,
                    "coverage_floor": coverage_floor,
                }),
            ));
            let counts = self.store.counts(&self.task_version)?;
            let extra = ((self.config.min_new_samples as f64 * 0.25).ceil() as usize)
                .min(self.config.min_new_samples);
            self.answers_at_last_failure = Some((counts.teacher_labelled as usize, extra));
        }
        Ok(Some(events))
    }

    fn promote(&mut self, version: &str) -> Result<(), ProxyCacheError> {
        let (student, ood, policy, meta) = self.versions.load_version(version)?;
        let mut index = self.versions.index()?;
        for entry in &mut index.versions {
            if entry.name == version {
                entry.state = "production".into();
            } else if entry.state == "production" || entry.state == "shadow" {
                entry.state = "superseded".into();
            }
        }
        index.production = Some(version.to_owned());
        self.versions.write_index(&index)?;
        self.production = Some(Production {
            version: version.to_owned(),
            student,
            ood,
            policy,
            calib_coverage: meta.calib_coverage,
        });
        self.shadow = None;
        self.suspicious = false;
        tracing::info!(
            task = %self.key,
            version = %version,
            calib_accepted = meta.calib_accepted,
            calib_disagree = meta.calib_disagree,
            n_calib = meta.n_calib,
            teacher_model = %meta.teacher_model,
            "proxy-cache production updated"
        );
        // A passing candidate ends a drift fallback and clears the recovery
        // horizon: the new student is calibrated on fresh data.
        if self.forced_fallback {
            self.forced_fallback = false;
            self.last_train_id = 0;
            self.store
                .insert_event("fallback_cleared", &serde_json::json!({"version": version}))?;
        }
        self.enforce_keep_versions(&mut index)?;
        Ok(())
    }

    fn reject_shadow(&mut self, version: &str) -> Result<(), ProxyCacheError> {
        let mut index = self.versions.index()?;
        for entry in &mut index.versions {
            if entry.name == version {
                entry.state = "rejected".into();
            }
        }
        self.versions.write_index(&index)?;
        self.shadow = None;
        self.store.insert_event(
            "rejected",
            &serde_json::json!({"version": version, "stage": "shadow"}),
        )?;
        Ok(())
    }

    fn drop_shadow(&mut self, reason: &str) -> Result<(), ProxyCacheError> {
        if let Some(shadow) = self.shadow.take() {
            let mut index = self.versions.index()?;
            for entry in &mut index.versions {
                if entry.name == shadow.version {
                    entry.state = "rejected".into();
                }
            }
            self.versions.write_index(&index)?;
            self.store.insert_event(
                "rejected",
                &serde_json::json!({"version": shadow.version, "reason": reason}),
            )?;
        }
        Ok(())
    }

    fn enforce_keep_versions(&self, index: &mut RegistryIndex) -> Result<(), ProxyCacheError> {
        let superseded: Vec<String> = index
            .versions
            .iter()
            .filter(|entry| entry.state == "superseded")
            .map(|entry| entry.name.clone())
            .collect();
        let keep = self.config.keep_versions.max(1);
        if superseded.len() > keep {
            let to_delete: Vec<String> = superseded[..superseded.len() - keep].to_vec();
            for name in &to_delete {
                self.versions.delete_version(name)?;
            }
            index
                .versions
                .retain(|entry| !to_delete.contains(&entry.name));
            self.versions.write_index(index)?;
        }
        Ok(())
    }

    /// Production status for telemetry.
    pub fn status(&self) -> TaskStatus {
        TaskStatus {
            confidence_floor: self.confidence_floor,
            key: self.key.clone(),
            task_version: self.task_version.clone(),
            classes: self.classes.clone(),
            deferred_labels: self.deferred_labels.clone(),
            production_version: self.production.as_ref().map(|p| p.version.clone()),
            production_threshold: self
                .production
                .as_ref()
                .and_then(|p| p.policy.conf_threshold),
            ood_threshold: self.production.as_ref().map(|p| p.policy.ood_threshold),
            expected_coverage: self.production.as_ref().map(|p| p.policy.expected_coverage),
            shadow_version: self.shadow.as_ref().map(|shadow| shadow.version.clone()),
            teacher_model: self.lineage_model.clone(),
            forced_fallback: self.forced_fallback,
            suspicious: self.suspicious,
            mode: self.mode,
        }
    }

    /// Newest teacher-labelled rows per split (parity and integration tests;
    /// the manager reads the store directly).
    pub fn labelled_rows_for_test(
        &self,
        max_train: usize,
        max_calib: usize,
    ) -> Result<(Vec<SampleRow>, Vec<SampleRow>), ProxyCacheError> {
        self.store
            .labelled_rows(&self.task_version, max_train, max_calib)
    }

    /// The task's store handle (manager-internal use).
    pub fn store(&self) -> &SampleStore {
        &self.store
    }

    /// Whether a production student can answer right now.
    pub fn has_production(&self) -> bool {
        self.production.is_some()
    }

    /// Clear the in-flight flag when a fit did not run (backoff, empty input,
    /// or the fit itself failed). `keep_needs_fit` false also abandons the
    /// wanted fit so a failing fit cannot spin the worker.
    pub fn clear_training_in_flight(&mut self, keep_needs_fit: bool) {
        self.training_in_flight = false;
        if !keep_needs_fit {
            self.needs_fit = false;
        }
    }

    /// Whether a fit is queued or running.
    pub fn training_in_flight(&self) -> bool {
        self.training_in_flight
    }
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

fn split_seed(config_seed: &u64, task_version: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(config_seed.to_le_bytes());
    hasher.update(task_version.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"))
}

fn probs_json(classes: &[String], probabilities: &[f64]) -> String {
    let mut map = serde_json::Map::new();
    for (class, probability) in classes.iter().zip(probabilities) {
        map.insert(class.clone(), serde_json::json!(probability));
    }
    serde_json::Value::Object(map).to_string()
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}
