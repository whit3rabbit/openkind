//! Student model fitting, calibration, and training preparation for TaskEngine.

use super::super::calibrate::{self, CalibRow};
use super::super::ood::KnnOod;
use super::super::registry::{VersionEntry, VersionMeta};
use super::super::store::SampleRow;
use super::super::student::LinearStudent;
use super::super::task::LabelTarget;
use super::super::ProxyCacheError;
use super::init::now;
use super::types::{FitInput, FitOutput, ShadowCandidate, TaskEngine};

impl TaskEngine {
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

    pub(super) fn current_encoder_id(&self) -> String {
        // The lineage is whatever the newest teacher-labelled rows carry.
        self.store
            .labelled_rows(&self.task_version, 1, 0)
            .ok()
            .and_then(|(train, _)| train.first().map(|row| row.encoder_id.clone()))
            .unwrap_or_default()
    }
}
