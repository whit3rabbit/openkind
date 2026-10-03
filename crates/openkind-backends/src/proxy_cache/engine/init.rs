//! Initialization and disk persistence for TaskEngine.

use std::collections::VecDeque;

use sha2::{Digest, Sha256};

use super::super::registry::{write_json_atomic, VersionRegistry};
use super::super::store::SampleStore;
use super::super::task::{TaskConfig, TaskMode, TaskSpec};
use super::super::ProxyCacheError;
use super::types::{Production, ShadowCandidate, TaskEngine};

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
            text_hash_salt: Vec::new(),
            active_encoder_id: String::new(),
            active_embedding_dim: 0,
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
            request_split_draw: None,
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
                let embedding_dim = student.dim();
                production = Some(Production {
                    version: name.clone(),
                    student,
                    ood,
                    policy,
                    encoder_id: meta.encoder_id.clone(),
                    embedding_dim,
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
                let embedding_dim = student.dim();
                shadow = Some(ShadowCandidate {
                    version: entry.name.clone(),
                    student,
                    ood,
                    policy,
                    encoder_id: meta.encoder_id.clone(),
                    embedding_dim,
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
            text_hash_salt: Vec::new(),
            active_encoder_id: String::new(),
            active_embedding_dim: 0,
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
            request_split_draw: None,
        };
        engine.classes = engine.spec.classes();
        // Lineage continues from the newest teacher-labelled row.
        engine.lineage_model = engine.store.latest_teacher_model(&engine.task_version)?;
        Ok(engine)
    }
}

impl TaskEngine {
    pub(in crate::proxy_cache) fn set_text_hash_salt(&mut self, salt: &[u8]) {
        self.text_hash_salt.clear();
        self.text_hash_salt.extend_from_slice(salt);
    }

    /// Bind restored state and future fits to the manager's active encoder.
    /// Incompatible persisted candidates are ignored so routing falls back to
    /// the teacher while enough rows are collected for a matching fit.
    pub(in crate::proxy_cache) fn set_embedder_identity(
        &mut self,
        encoder_id: &str,
        embedding_dim: usize,
    ) -> bool {
        self.active_encoder_id = encoder_id.to_owned();
        self.active_embedding_dim = embedding_dim;
        let production_incompatible = self.production.as_ref().is_some_and(|production| {
            !version_matches_embedder(
                &production.encoder_id,
                production.embedding_dim,
                production.student.dim(),
                production.ood.dim(),
                encoder_id,
                embedding_dim,
            )
        });
        let shadow_incompatible = self.shadow.as_ref().is_some_and(|shadow| {
            !version_matches_embedder(
                &shadow.encoder_id,
                shadow.embedding_dim,
                shadow.student.dim(),
                shadow.ood.dim(),
                encoder_id,
                embedding_dim,
            )
        });
        if production_incompatible {
            self.production = None;
        }
        if shadow_incompatible {
            self.shadow = None;
        }
        production_incompatible || shadow_incompatible
    }
}

fn version_matches_embedder(
    version_encoder_id: &str,
    version_dim: usize,
    student_dim: usize,
    ood_dim: usize,
    encoder_id: &str,
    embedding_dim: usize,
) -> bool {
    !encoder_id.is_empty()
        && version_encoder_id == encoder_id
        && embedding_dim > 0
        && version_dim == embedding_dim
        && student_dim == embedding_dim
        && ood_dim == embedding_dim
}

pub(super) fn split_seed(config_seed: &u64, task_version: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(config_seed.to_le_bytes());
    hasher.update(task_version.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"))
}

pub(super) fn probs_json(classes: &[String], probabilities: &[f64]) -> String {
    let mut map = serde_json::Map::new();
    for (class, probability) in classes.iter().zip(probabilities) {
        map.insert(class.clone(), serde_json::json!(probability));
    }
    serde_json::Value::Object(map).to_string()
}

pub(super) fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::version_matches_embedder;

    #[test]
    fn restored_students_require_matching_encoder_lineage_and_dimensions() {
        assert!(version_matches_embedder(
            "bert-checkpoint-a",
            768,
            768,
            768,
            "bert-checkpoint-a",
            768,
        ));
        assert!(!version_matches_embedder(
            "bert-checkpoint-a",
            768,
            768,
            768,
            "bert-checkpoint-b",
            768,
        ));
        assert!(!version_matches_embedder(
            "bert-checkpoint-a",
            768,
            768,
            768,
            "bert-checkpoint-a",
            384,
        ));
        assert!(!version_matches_embedder(
            "bert-checkpoint-a",
            768,
            768,
            384,
            "bert-checkpoint-a",
            768,
        ));
    }
}
