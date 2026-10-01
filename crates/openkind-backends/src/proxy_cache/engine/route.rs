//! Request routing and student scoring for TaskEngine.

use super::super::task::{Channel, RoutingReason, TaskMode};
use super::types::{LocalPrediction, Production, RouteDecision, TaskEngine};

impl TaskEngine {
    /// Begin one request: seed the per-request split draw. Every item of a
    /// request shares one draw so siblings never straddle the split.
    pub fn begin_request(&mut self) {
        let seed = self.rng.u64(..)
            ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0);
        self.request_split_draw = Some(fastrand::Rng::with_seed(seed).f64());
    }

    pub(super) fn split_for(&mut self, channel: Channel) -> &'static str {
        if !channel.is_iid() {
            return "train";
        }
        let draw = self.request_split_draw.unwrap_or_else(|| self.rng.f64());
        if draw < self.config.calib_fraction {
            "calib"
        } else {
            "train"
        }
    }

    pub(super) fn effective_audit_rate(&self) -> f64 {
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

    pub(super) fn score_with_student(
        &self,
        production: &Production,
        embedding: &[f32],
    ) -> LocalPrediction {
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
}
