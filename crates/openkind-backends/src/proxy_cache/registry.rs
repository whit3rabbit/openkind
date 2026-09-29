//! Immutable student version directories.
//!
//! Every fitted candidate lands in `tasks/<key>/versions/student-v<N>/`
//! containing the student head, the OOD reference, the calibrated policy,
//! and fit metadata. A `registry.json` index points at the production
//! version; writes are atomic (temp file + rename) so a crash leaves either
//! the old or the new index, never a torn one.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::calibrate::RoutingPolicy;
use super::ood::KnnOod;
use super::student::{FitReport, LinearStudent};
use super::ProxyCacheError;

/// On-disk fit metadata for one version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMeta {
    /// Version name (`student-v3`).
    pub name: String,
    /// Train rows used.
    pub n_train: usize,
    /// Calibration rows used.
    pub n_calib: usize,
    /// Final training loss.
    pub train_loss: f64,
    /// Epochs run (early stopping may cut the budget).
    pub epochs: usize,
    /// Per-class train counts at fit time.
    pub train_label_counts: Vec<(String, i64)>,
    /// Encoder lineage the fit was trained on.
    pub encoder_id: String,
    /// Resolved teacher model of the training rows.
    pub teacher_model: String,
    /// Confidence floor at fit time (policies are floor-specific).
    pub confidence_floor: Option<f64>,
    /// Target agreement at fit time.
    pub target_agreement: f64,
    /// Store row id the training data reached.
    pub trained_to_id: i64,
    /// Calibration rows the candidate's policy would answer.
    #[serde(default)]
    pub calib_accepted: usize,
    /// Calibration rows the candidate disagrees with the teacher on.
    #[serde(default)]
    pub calib_disagree: usize,
    /// Candidate coverage on its calibration rows.
    #[serde(default)]
    pub calib_coverage: f64,
    /// Task config snapshot (JSON).
    pub config: serde_json::Value,
    /// First store row id covered by the fit (recovery training).
    #[serde(default)]
    pub since_id: i64,
}

/// One entry of the registry index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Version name (`student-v3`; `:` never appears in directory names).
    pub name: String,
    /// `production`, `shadow`, `superseded`, or `rejected`.
    pub state: String,
    /// Creation timestamp (unix seconds).
    pub created: f64,
}

/// The registry index file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryIndex {
    /// Production version name, if any.
    pub production: Option<String>,
    /// All known versions, newest last.
    pub versions: Vec<VersionEntry>,
}

/// Root of one task's version tree.
pub struct VersionRegistry {
    root: PathBuf,
}

impl VersionRegistry {
    /// Registry rooted at `tasks/<key>/versions`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Directory of one version.
    pub fn version_dir(&self, name: &str) -> PathBuf {
        self.root.join(name.replace(':', "-"))
    }

    /// Read the index; a missing file reads as an empty registry.
    pub fn index(&self) -> Result<RegistryIndex, ProxyCacheError> {
        let path = self.root.join("registry.json");
        if !path.exists() {
            return Ok(RegistryIndex::default());
        }
        let text = std::fs::read_to_string(&path).map_err(|error| {
            ProxyCacheError::Version(format!("read {}: {error}", path.display()))
        })?;
        serde_json::from_str(&text).map_err(|error| {
            ProxyCacheError::Version(format!("decode {}: {error}", path.display()))
        })
    }

    /// Atomically replace the index.
    pub fn write_index(&self, index: &RegistryIndex) -> Result<(), ProxyCacheError> {
        std::fs::create_dir_all(&self.root).map_err(|error| {
            ProxyCacheError::Version(format!("create {}: {error}", self.root.display()))
        })?;
        let path = self.root.join("registry.json");
        let body = serde_json::to_string_pretty(index)
            .map_err(|error| ProxyCacheError::Version(format!("encode index: {error}")))?;
        atomic_write(&path, body.as_bytes())?;
        Ok(())
    }

    /// The next free version number (one past the highest existing).
    pub fn next_version_number(&self) -> Result<u64, ProxyCacheError> {
        let index = self.index()?;
        let mut max = 0u64;
        for entry in &index.versions {
            if let Some(number) = entry
                .name
                .strip_prefix("student-v")
                .and_then(|rest| rest.parse::<u64>().ok())
            {
                max = max.max(number);
            }
        }
        Ok(max + 1)
    }

    /// Persist one candidate version to its directory.
    #[allow(clippy::too_many_arguments)]
    pub fn save_version(
        &self,
        name: &str,
        student: &LinearStudent,
        ood: &KnnOod,
        policy: &RoutingPolicy,
        meta: &VersionMeta,
        report: &FitReport,
    ) -> Result<PathBuf, ProxyCacheError> {
        let dir = self.version_dir(name);
        std::fs::create_dir_all(&dir).map_err(|error| {
            ProxyCacheError::Version(format!("create {}: {error}", dir.display()))
        })?;
        write_bytes(&dir.join("head.safetensors"), &student.to_safetensors()?)?;
        write_bytes(&dir.join("ood.safetensors"), &ood.to_safetensors()?)?;
        atomic_write(&dir.join("policy.json"), policy.to_json()?.as_bytes())?;
        let mut meta_with_report = serde_json::to_value(meta)
            .map_err(|error| ProxyCacheError::Version(format!("encode meta: {error}")))?;
        if let Some(object) = meta_with_report.as_object_mut() {
            object.insert(
                "train_loss_reported".into(),
                serde_json::json!(report.train_loss),
            );
            object.insert("val_loss".into(), serde_json::json!(report.val_loss));
        }
        atomic_write(
            &dir.join("meta.json"),
            serde_json::to_string_pretty(&meta_with_report)
                .map_err(|error| ProxyCacheError::Version(format!("encode meta: {error}")))?
                .as_bytes(),
        )?;
        Ok(dir)
    }

    /// Load a version's student, gate, and policy.
    pub fn load_version(
        &self,
        name: &str,
    ) -> Result<(LinearStudent, KnnOod, RoutingPolicy, VersionMeta), ProxyCacheError> {
        let dir = self.version_dir(name);
        let head = std::fs::read(dir.join("head.safetensors")).map_err(|error| {
            ProxyCacheError::Version(format!(
                "read {}: {error}",
                dir.join("head.safetensors").display()
            ))
        })?;
        let ood_bytes = std::fs::read(dir.join("ood.safetensors")).map_err(|error| {
            ProxyCacheError::Version(format!(
                "read {}: {error}",
                dir.join("ood.safetensors").display()
            ))
        })?;
        let policy_text = std::fs::read_to_string(dir.join("policy.json")).map_err(|error| {
            ProxyCacheError::Version(format!(
                "read {}: {error}",
                dir.join("policy.json").display()
            ))
        })?;
        let meta_text = std::fs::read_to_string(dir.join("meta.json")).map_err(|error| {
            ProxyCacheError::Version(format!("read {}: {error}", dir.join("meta.json").display()))
        })?;
        let student = LinearStudent::from_safetensors(&head)?;
        let ood = KnnOod::from_safetensors(&ood_bytes)?;
        let policy = RoutingPolicy::from_json(&policy_text)?;
        let meta: VersionMeta = serde_json::from_str(&meta_text)
            .map_err(|error| ProxyCacheError::Version(format!("decode meta: {error}")))?;
        Ok((student, ood, policy, meta))
    }

    /// Delete a version directory (keeps the index entry marked `deleted`).
    pub fn delete_version(&self, name: &str) -> Result<(), ProxyCacheError> {
        let dir = self.version_dir(name);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|error| {
                ProxyCacheError::Version(format!("remove {}: {error}", dir.display()))
            })?;
        }
        Ok(())
    }
}

/// Write a JSON document atomically (temp file + rename).
pub fn write_json_atomic(path: &Path, value: &serde_json::Value) -> Result<(), ProxyCacheError> {
    let body = serde_json::to_string_pretty(value)
        .map_err(|error| ProxyCacheError::Version(format!("encode json: {error}")))?;
    atomic_write(path, body.as_bytes())
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), ProxyCacheError> {
    let mut file = std::fs::File::create(path)
        .map_err(|error| ProxyCacheError::Version(format!("create {}: {error}", path.display())))?;
    file.write_all(bytes)
        .map_err(|error| ProxyCacheError::Version(format!("write {}: {error}", path.display())))?;
    file.sync_all()
        .map_err(|error| ProxyCacheError::Version(format!("sync {}: {error}", path.display())))?;
    Ok(())
}

/// Write via a sibling temp file + rename so readers never see partial data.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ProxyCacheError> {
    let temp = path.with_extension("tmp");
    write_bytes(&temp, bytes)?;
    std::fs::rename(&temp, path)
        .map_err(|error| ProxyCacheError::Version(format!("rename {}: {error}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_registry() -> (tempfile::TempDir, VersionRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let registry = VersionRegistry::new(dir.path().join("versions"));
        (dir, registry)
    }

    fn sample_policy() -> RoutingPolicy {
        RoutingPolicy {
            conf_threshold: Some(0.9),
            ood_threshold: 0.4,
            expected_coverage: 0.8,
            disagreement_ub: 0.02,
            expected_system_disagreement: 0.01,
            budget: 0.02,
            delta: 0.05,
            n_calib: 100,
            deferred_labels: vec!["rare".into()],
            confidence_floor: None,
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let (_dir, registry) = temp_registry();
        let student = LinearStudent::new(4, 3);
        let ood = KnnOod::fit(&[], 4, 5, 0.99, 100);
        let policy = sample_policy();
        let meta = VersionMeta {
            name: "student-v1".into(),
            n_train: 100,
            n_calib: 50,
            train_loss: 0.1,
            epochs: 25,
            train_label_counts: vec![("a".into(), 50), ("b".into(), 50)],
            encoder_id: "hash-512-0-bi".into(),
            teacher_model: "jev-1.0.0".into(),
            confidence_floor: None,
            target_agreement: 0.98,
            trained_to_id: 150,
            calib_accepted: 40,
            calib_disagree: 1,
            calib_coverage: 0.8,
            config: serde_json::json!({}),
            since_id: 0,
        };
        let report = FitReport {
            epochs: 25,
            train_loss: 0.1,
            val_loss: Some(0.12),
        };
        registry
            .save_version("student-v1", &student, &ood, &policy, &meta, &report)
            .unwrap();
        let (loaded_student, loaded_ood, loaded_policy, loaded_meta) =
            registry.load_version("student-v1").unwrap();
        assert_eq!(loaded_student.dim(), 4);
        assert_eq!(loaded_student.classes(), 3);
        assert_eq!(loaded_policy, policy);
        assert_eq!(loaded_meta.name, "student-v1");
        assert_eq!(loaded_meta.trained_to_id, 150);
        assert_eq!(loaded_ood.reference_rows(), 0);
    }

    #[test]
    fn index_atomic_and_version_numbering() {
        let (_dir, registry) = temp_registry();
        assert_eq!(registry.next_version_number().unwrap(), 1);
        let mut index = registry.index().unwrap();
        index.versions.push(VersionEntry {
            name: "student-v1".into(),
            state: "superseded".into(),
            created: 1.0,
        });
        index.production = Some("student-v1".into());
        registry.write_index(&index).unwrap();
        assert_eq!(
            registry.index().unwrap().production,
            Some("student-v1".into())
        );
        assert_eq!(registry.next_version_number().unwrap(), 2);

        // A rejected version still advances the counter.
        let mut index = registry.index().unwrap();
        index.versions.push(VersionEntry {
            name: "student-v2".into(),
            state: "rejected".into(),
            created: 2.0,
        });
        registry.write_index(&index).unwrap();
        assert_eq!(registry.next_version_number().unwrap(), 3);
    }

    #[test]
    fn delete_version_removes_directory() {
        let (_dir, registry) = temp_registry();
        let student = LinearStudent::new(2, 2);
        let ood = KnnOod::fit(&[], 2, 5, 0.99, 100);
        let policy = sample_policy();
        let meta = VersionMeta {
            name: "student-v1".into(),
            n_train: 10,
            n_calib: 5,
            train_loss: 0.1,
            epochs: 10,
            train_label_counts: vec![],
            encoder_id: "hash".into(),
            teacher_model: "jev-1.0.0".into(),
            confidence_floor: None,
            target_agreement: 0.98,
            trained_to_id: 15,
            calib_accepted: 4,
            calib_disagree: 0,
            calib_coverage: 0.8,
            config: serde_json::json!({}),
            since_id: 0,
        };
        let report = FitReport {
            epochs: 10,
            train_loss: 0.1,
            val_loss: None,
        };
        registry
            .save_version("student-v1", &student, &ood, &policy, &meta, &report)
            .unwrap();
        assert!(registry.version_dir("student-v1").exists());
        registry.delete_version("student-v1").unwrap();
        assert!(!registry.version_dir("student-v1").exists());
        assert!(registry.load_version("student-v1").is_err());
    }
}
