//! Registry schema and validation for pinned evaluation datasets.
//!
//! The registry is checked into this repository (no public mirror: serving
//! dataset bytes from a mirror would be redistribution). Every entry pins the
//! upstream Hugging Face repository, the exact main-branch and
//! `refs/convert/parquet` revisions, and the size plus SHA-256 of each
//! downloaded parquet shard.

use std::collections::BTreeMap;

use crate::{Error, Result};

pub const DATASET_REGISTRY_SCHEMA: &str = "openkind-dataset-registry/v1";

/// Installation size ceiling per dataset, mirroring the model store's cap.
pub const MAX_DATASET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

const PRIMITIVES: [&str; 3] = ["choice", "noul", "score"];

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRegistry {
    pub schema: String,
    pub datasets: Vec<DatasetEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetEntry {
    /// Pull name, unique in the registry (`[a-z0-9_]+`).
    pub name: String,
    pub description: String,
    /// Hugging Face dataset repository (`owner/name`).
    pub hf_repo: String,
    /// Pinned 40-hex commit of the repository `main` branch.
    pub hf_revision: String,
    /// Pinned 40-hex commit of the repository `refs/convert/parquet` branch.
    pub convert_revision: String,
    /// Parquet shards actually downloaded, relative to the convert-branch root.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<DatasetFile>,
    /// Materialization split name to upstream split name (`eval` required).
    pub splits: BTreeMap<String, String>,
    /// Convert-layout config prefixes whose shards are pinned.
    pub configs: Vec<String>,
    pub license: String,
    pub gated: Gated,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub access_note: String,
    pub task_family: String,
    pub primitives: Vec<String>,
    pub template: TemplateRef,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetFile {
    /// Path relative to the `refs/convert/parquet` root (`config/split/file.parquet`).
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// Whether upstream access needs more than an anonymous request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gated {
    /// Anonymous download works.
    None,
    /// Hugging Face click-through terms after login; token download works.
    Auto,
    /// Separate out-of-band access request governs use.
    Form,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemplateRef {
    pub id: String,
    pub version: u32,
    pub source: String,
}

impl DatasetRegistry {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let registry: DatasetRegistry = serde_json::from_slice(bytes)?;
        registry.validate()?;
        Ok(registry)
    }

    /// The registry compiled into this build.
    pub fn embedded() -> Result<Self> {
        Self::parse(crate::DATASETS_JSON.as_bytes())
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != DATASET_REGISTRY_SCHEMA {
            return Err(Error::Invalid(format!(
                "unknown dataset registry schema {}",
                self.schema
            )));
        }
        let mut names = Vec::new();
        for entry in &self.datasets {
            entry.validate()?;
            names.push(entry.name.as_str());
        }
        names.sort_unstable();
        for pair in names.windows(2) {
            if pair[0] == pair[1] {
                return Err(Error::Invalid(format!("duplicate dataset {}", pair[0])));
            }
        }
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<&DatasetEntry> {
        self.datasets.iter().find(|entry| entry.name == name)
    }
}

impl DatasetEntry {
    pub fn validate(&self) -> Result<()> {
        if !valid_name(&self.name) {
            return Err(Error::Invalid(format!(
                "invalid dataset name {}",
                self.name
            )));
        }
        if self.description.trim().is_empty() {
            return Err(Error::Invalid("empty description".into()));
        }
        if !valid_repo(&self.hf_repo) {
            return Err(Error::Invalid(format!(
                "invalid dataset repository {}",
                self.hf_repo
            )));
        }
        if !valid_revision(&self.hf_revision) || !valid_revision(&self.convert_revision) {
            return Err(Error::Invalid(format!(
                "dataset {} has an unpinned revision",
                self.name
            )));
        }
        let eval_split = self
            .splits
            .get("eval")
            .ok_or_else(|| Error::Invalid(format!("dataset {} has no eval split", self.name)))?;
        let mut used_splits: Vec<&String> = vec![eval_split];
        if let Some(dev) = self.splits.get("dev") {
            if dev == eval_split {
                return Err(Error::Invalid("dev and eval splits coincide".into()));
            }
            used_splits.push(dev);
        }
        for (name, value) in &self.splits {
            if name != "eval" && name != "dev" {
                return Err(Error::Invalid(format!("unknown split name {name}")));
            }
            if value.trim().is_empty() {
                return Err(Error::Invalid(format!("empty split {name}")));
            }
        }
        if self.configs.is_empty() {
            return Err(Error::Invalid(format!(
                "dataset {} has no configs",
                self.name
            )));
        }
        let mut configs = self.configs.clone();
        configs.sort_unstable();
        for pair in configs.windows(2) {
            if pair[0] == pair[1] {
                return Err(Error::Invalid(format!("duplicate config {}", pair[0])));
            }
        }
        if self.license.trim().is_empty() {
            return Err(Error::Invalid("empty license".into()));
        }
        if self.task_family.trim().is_empty() {
            return Err(Error::Invalid("empty task family".into()));
        }
        if self.primitives.is_empty()
            || self
                .primitives
                .iter()
                .any(|primitive| !PRIMITIVES.contains(&primitive.as_str()))
        {
            return Err(Error::Invalid(format!(
                "dataset {} declares no valid primitive",
                self.name
            )));
        }
        if self.template.id.trim().is_empty() || self.template.source.trim().is_empty() {
            return Err(Error::Invalid("incomplete template reference".into()));
        }
        let mut total = 0u64;
        let mut paths = Vec::new();
        for file in &self.files {
            if !valid_relative_path(&file.path) {
                return Err(Error::Invalid(format!("invalid file path {}", file.path)));
            }
            let mut segments = file.path.split('/');
            let config = segments.next().unwrap_or_default();
            let split = segments.next().unwrap_or_default();
            if !self.configs.iter().any(|declared| declared == config) {
                return Err(Error::Invalid(format!(
                    "file {} is outside the declared configs",
                    file.path
                )));
            }
            if !used_splits.iter().any(|declared| *declared == split) {
                return Err(Error::Invalid(format!(
                    "file {} belongs to split {split}, which the entry does not use",
                    file.path
                )));
            }
            if !file.path.ends_with(".parquet") {
                return Err(Error::Invalid(format!("file {} is not parquet", file.path)));
            }
            if file.size == 0 {
                return Err(Error::Invalid(format!("file {} has no size", file.path)));
            }
            if !valid_sha256(&file.sha256) {
                return Err(Error::Invalid(format!("file {} has no digest", file.path)));
            }
            total = total
                .checked_add(file.size)
                .filter(|total| *total <= MAX_DATASET_BYTES)
                .ok_or_else(|| Error::Invalid("dataset size exceeds 2 GiB".into()))?;
            paths.push(file.path.as_str());
        }
        paths.sort_unstable();
        for pair in paths.windows(2) {
            if pair[0] == pair[1] {
                return Err(Error::Invalid(format!("duplicate file path {}", pair[0])));
            }
        }
        Ok(())
    }

    /// Files for one (config, materialization split) pair, in path order.
    pub fn split_files(&self, config: &str, split: &str) -> Vec<&DatasetFile> {
        let upstream = match self.splits.get(split) {
            Some(upstream) => upstream,
            None => return Vec::new(),
        };
        let prefix = format!("{config}/{upstream}/");
        self.files
            .iter()
            .filter(|file| file.path.starts_with(&prefix))
            .collect()
    }

    pub fn gated(&self) -> bool {
        self.gated != Gated::None
    }
}

/// Lowercase slug names, matching the pull-name shape of the model store.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// `owner/name` repository shape.
pub fn valid_repo(repo: &str) -> bool {
    match repo.split_once('/') {
        Some((owner, name)) => {
            !owner.is_empty()
                && !name.is_empty()
                && !name.contains('/')
                && repo
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
        }
        None => false,
    }
}

/// 64 lowercase hex characters (a file digest).
pub fn valid_sha256(sha: &str) -> bool {
    sha.len() == 64 && sha.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

/// 40 lowercase hex characters (a pinned git commit).
pub fn valid_revision(sha: &str) -> bool {
    sha.len() == 40 && sha.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

/// Relative slash-separated path with a restricted charset per segment.
pub fn valid_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.ends_with('/')
        && path.split('/').all(|segment| {
            !segment.is_empty()
                && !segment.contains('\\')
                && segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
                && segment != "."
                && segment != ".."
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_json(name: &str) -> String {
        format!(
            r#"{{
                "name": "{name}",
                "description": "test dataset",
                "hf_repo": "owner/data",
                "hf_revision": "{}",
                "convert_revision": "{}",
                "files": [{{"path": "default/test/0000.parquet", "size": 10, "sha256": "{}"}}],
                "splits": {{"dev": "train", "eval": "test"}},
                "configs": ["default"],
                "license": "mit",
                "gated": "none",
                "task_family": "classification",
                "primitives": ["choice"],
                "template": {{"id": "t", "version": 1, "source": "src"}}
            }}"#,
            "a".repeat(40),
            "b".repeat(40),
            "c".repeat(64),
        )
    }

    #[test]
    fn parses_and_validates_a_minimal_entry() {
        let registry: DatasetRegistry = serde_json::from_str(&format!(
            r#"{{"schema": "{DATASET_REGISTRY_SCHEMA}", "datasets": [{}]}}"#,
            entry_json("demo")
        ))
        .unwrap();
        registry.validate().unwrap();
        assert_eq!(
            registry
                .find("demo")
                .unwrap()
                .split_files("default", "eval")
                .len(),
            1
        );
        assert!(registry
            .find("demo")
            .unwrap()
            .split_files("default", "dev")
            .is_empty());
    }

    #[test]
    fn rejects_files_outside_used_splits() {
        let bad =
            entry_json("demo").replace("default/test/0000.parquet", "default/other/0000.parquet");
        let entry: DatasetEntry = serde_json::from_str(&bad).unwrap();
        assert!(entry.validate().is_err());
    }

    #[test]
    fn rejects_bad_names_repos_and_digests() {
        for bad in ["Demo", "", "with-dash", "x".repeat(65).as_str()] {
            assert!(!valid_name(bad));
        }
        assert!(valid_name("sst2"));
        assert!(valid_name("commonsense_qa"));
        assert!(!valid_repo("nodomain"));
        assert!(!valid_repo("a/b/c"));
        assert!(!valid_sha256(&"A".repeat(64)));
        assert!(!valid_revision(&"a".repeat(64)));
        assert!(valid_revision(&"a".repeat(40)));
        assert!(!valid_relative_path("../escape.parquet"));
        assert!(!valid_relative_path("a//b.parquet"));
    }

    #[test]
    fn entry_validate_rejects_each_drifted_field() {
        let parse = |json: &str| -> DatasetEntry { serde_json::from_str(json).unwrap() };
        // Patches are applied to the JSON so every rejection path runs
        // through the same deserialization a real registry file takes.
        let reject = |patch: &[(&str, &str)], label: &str| {
            let mut json = entry_json("demo");
            for (from, to) in patch {
                assert!(
                    json.contains(from),
                    "{label}: fixture lost `{from}`; update the patch"
                );
                json = json.replace(from, to);
            }
            let entry = parse(&json);
            assert!(
                entry.validate().is_err(),
                "{label}: a drifted entry must be rejected"
            );
        };

        reject(
            &[("\"description\": \"test dataset\"", "\"description\": \"\"")],
            "empty description",
        );
        reject(
            &[("\"hf_repo\": \"owner/data\"", "\"hf_repo\": \"nodomain\"")],
            "unslashed repo",
        );
        // Unpinned revisions: a 39-char revision fails the length rule.
        reject(
            &[(
                "\"hf_revision\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
                "\"hf_revision\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
            )],
            "short hf revision",
        );
        reject(
            &[(
                "\"splits\": {\"dev\": \"train\", \"eval\": \"test\"}",
                "\"splits\": {\"dev\": \"train\", \"train\": \"test\"}",
            )],
            "missing eval split",
        );
        reject(
            &[(
                "\"splits\": {\"dev\": \"train\", \"eval\": \"test\"}",
                "\"splits\": {\"dev\": \"train\", \"eval\": \"train\"}",
            )],
            "dev equals eval",
        );
        reject(
            &[("\"configs\": [\"default\"]", "\"configs\": []")],
            "empty configs",
        );
        reject(
            &[("\"license\": \"mit\"", "\"license\": \"\"")],
            "empty license",
        );
        reject(
            &[(
                "\"task_family\": \"classification\"",
                "\"task_family\": \"\"",
            )],
            "empty task family",
        );
        reject(
            &[("\"primitives\": [\"choice\"]", "\"primitives\": []")],
            "empty primitives",
        );
        reject(
            &[(
                "\"primitives\": [\"choice\"]",
                "\"primitives\": [\"alchemy\"]",
            )],
            "unknown primitive",
        );
        reject(
            &[(
                "\"template\": {\"id\": \"t\", \"version\": 1, \"source\": \"src\"}",
                "\"template\": {\"id\": \"\", \"version\": 1, \"source\": \"src\"}",
            )],
            "empty template id",
        );
        reject(
            &[(
                "\"files\": [{\"path\": \"default/test/0000.parquet\", \"size\": 10",
                "\"files\": [{\"path\": \"default/test/0000.parquet\", \"size\": 0",
            )],
            "zero-size file",
        );
        reject(
            &[("\"sha256\": \"cccc", "\"sha256\": \"CCCC")],
            "uppercase file digest",
        );
    }

    #[test]
    fn registry_validate_rejects_wrong_schema_and_duplicate_names() {
        let duplicated = format!(
            r#"{{"schema": "{DATASET_REGISTRY_SCHEMA}", "datasets": [{}, {}]}}"#,
            entry_json("demo"),
            entry_json("demo")
        );
        let registry: DatasetRegistry = serde_json::from_str(&duplicated).unwrap();
        assert!(registry.validate().is_err(), "duplicate names must fail");

        let wrong_schema = format!(
            r#"{{"schema": "other/v9", "datasets": [{}]}}"#,
            entry_json("demo")
        );
        let registry: DatasetRegistry = serde_json::from_str(&wrong_schema).unwrap();
        assert!(registry.validate().is_err(), "wrong schema must fail");

        assert!(serde_json::from_str::<DatasetRegistry>("not json").is_err());
    }
}
