//! Teacher and served student observation and lineage tracking for TaskEngine.

use super::super::calibrate;
use super::super::store::SampleRow;
use super::super::task::{Channel, TeacherChangePolicy};
use super::super::ProxyCacheError;
use super::init::probs_json;
use super::types::{AuditObservation, RouteDecision, TaskEngine, TeacherAnswer};

impl TaskEngine {
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
            text_hash: super::super::text::text_hash(state_text, self.task_version.as_bytes()),
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
    pub(super) fn shadow_scores(
        &self,
        embedding: &[f32],
    ) -> (Option<String>, Option<String>, Option<f64>) {
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
            text_hash: super::super::text::text_hash(state_text, self.task_version.as_bytes()),
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
    pub(super) fn observe_teacher_lineage(
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

    pub(super) fn record_audit_observation(&mut self, agree: bool) {
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
}
