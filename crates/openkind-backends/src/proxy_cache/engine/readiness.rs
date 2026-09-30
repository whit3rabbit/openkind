//! Class readiness, training requests, and status telemetry for TaskEngine.

use super::super::store::{SampleRow, SampleStore, StoreCounts};
use super::super::task::RareClasses;
use super::super::ProxyCacheError;
use super::types::{TaskEngine, TaskStatus};

impl TaskEngine {
    pub(super) fn check_class_readiness(
        &mut self,
        _counts: &StoreCounts,
    ) -> Result<bool, ProxyCacheError> {
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

    pub(super) fn request_retrain(&mut self) {
        self.teacher_answers_since_train = 0;
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
