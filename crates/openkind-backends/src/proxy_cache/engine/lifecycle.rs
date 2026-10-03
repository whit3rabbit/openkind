//! Background maintenance ticks, shadow judgment, and version lifecycle for TaskEngine.

use super::super::calibrate;
use super::super::registry::RegistryIndex;
use super::super::ProxyCacheError;
use super::types::{AuditObservation, Production, TaskEngine, TickOutcome};

impl TaskEngine {
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

    /// Judge the shadow candidate when enough shadow rows exist. Returns
    /// events when a decision was reached.
    ///
    /// The guarantee: shadow rows are scored as the candidate would have
    /// served them, pooled with the candidate's own calibration counts, and
    /// tested against the FULL budget with a Clopper–Pearson bound;
    /// coverage must not fall below 95% of production's calibrated coverage.
    pub(super) fn judge_shadow(
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

    pub(super) fn promote(&mut self, version: &str) -> Result<(), ProxyCacheError> {
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

    pub(super) fn reject_shadow(&mut self, version: &str) -> Result<(), ProxyCacheError> {
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

    pub(super) fn drop_shadow(&mut self, reason: &str) -> Result<(), ProxyCacheError> {
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

    pub(super) fn enforce_keep_versions(
        &self,
        index: &mut RegistryIndex,
    ) -> Result<(), ProxyCacheError> {
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
}
