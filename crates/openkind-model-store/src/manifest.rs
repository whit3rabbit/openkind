use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const CATALOG_SCHEMA: &str = "openkind-catalog/v1";
pub const MANIFEST_SCHEMA: &str = "openkind-model/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub schema: String,
    pub models: Vec<CatalogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    pub name: String,
    pub profile_id: String,
    pub loader_id: String,
    pub description: String,
    pub support_status: String,
    pub manifest_path: String,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub name: String,
    pub profile_id: String,
    pub loader_id: String,
    pub description: String,
    pub release_date: String,
    pub support_status: String,
    pub question_types: Vec<String>,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// A path below the model installation, such as `checkpoint/config.json`.
    pub path: String,
    pub size: u64,
    pub sha256: String,
    pub source: Source,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// `huggingface` for author artifacts, `github` for the pinned OpenKind bundle.
    pub kind: String,
    pub repository: String,
    pub revision: String,
    pub path: String,
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn valid_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
}

pub(crate) fn valid_name(name: &str) -> bool {
    let Some((family, version)) = name.split_once(':') else {
        return false;
    };
    !family.starts_with('.')
        && !version.starts_with('.')
        && !version.contains(':')
        && [family, version].iter().all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
        })
}

impl Catalog {
    pub fn validate(&self) -> Result<()> {
        if self.schema != CATALOG_SCHEMA {
            return Err(Error::Invalid("unsupported catalog schema".into()));
        }
        let mut names = HashSet::new();
        for entry in &self.models {
            if !valid_name(&entry.name)
                || !names.insert(&entry.name)
                || !valid_relative_path(&entry.manifest_path)
                || !entry.manifest_path.starts_with("manifests/")
                || !valid_sha256(&entry.manifest_sha256)
                || entry.profile_id.is_empty()
                || entry.loader_id.is_empty()
            {
                return Err(Error::Invalid(format!(
                    "invalid catalog entry {}",
                    entry.name
                )));
            }
        }
        Ok(())
    }
}

impl Manifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema != MANIFEST_SCHEMA || !valid_name(&self.name) {
            return Err(Error::Invalid(
                "unsupported manifest schema or model name".into(),
            ));
        }
        if self.profile_id.is_empty()
            || self.loader_id.is_empty()
            || self.description.is_empty()
            || self.release_date.len() != 10
            || !["rust-loadable", "task-qualified", "release-promoted"]
                .contains(&self.support_status.as_str())
        {
            return Err(Error::Invalid(
                "incomplete profile identity or status".into(),
            ));
        }
        if self.question_types.is_empty()
            || self
                .question_types
                .iter()
                .any(|kind| !["choice", "score", "noul"].contains(&kind.as_str()))
        {
            return Err(Error::Invalid("invalid question types".into()));
        }
        let mut paths: HashSet<&str> = HashSet::new();
        for artifact in &self.artifacts {
            if !valid_relative_path(&artifact.path)
                || artifact.path == "manifest.json"
                || artifact.path.starts_with("manifest.json/")
                || !paths.insert(&artifact.path)
                || artifact.size == 0
                || !valid_sha256(&artifact.sha256)
                || !artifact.source.validate()
            {
                return Err(Error::Invalid(format!(
                    "invalid artifact {}",
                    artifact.path
                )));
            }
        }
        if self.artifacts.is_empty() {
            return Err(Error::Invalid("manifest has no artifacts".into()));
        }
        // A file cannot also be another artifact's parent in the staged tree.
        for path in &paths {
            let mut parent = path.rsplit_once('/');
            while let Some((prefix, _)) = parent {
                if paths.contains(prefix) {
                    return Err(Error::Invalid(format!(
                        "artifact path overlaps another artifact: {path}"
                    )));
                }
                parent = prefix.rsplit_once('/');
            }
        }
        Ok(())
    }
}

impl Source {
    fn validate(&self) -> bool {
        matches!(self.kind.as_str(), "huggingface" | "github")
            && self.repository.split('/').count() == 2
            && self.repository.split('/').all(valid_relative_path)
            && self.revision.len() == 40
            && self.revision.bytes().all(|b| b.is_ascii_hexdigit())
            && valid_relative_path(&self.path)
    }

    pub(crate) fn url(&self) -> String {
        if self.kind == "github" {
            format!(
                "https://raw.githubusercontent.com/{}/{}/{}",
                self.repository, self.revision, self.path
            )
        } else {
            format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                self.repository, self.revision, self.path
            )
        }
    }
}

impl CatalogEntry {
    pub(crate) fn matches(&self, manifest: &Manifest) -> bool {
        self.name == manifest.name
            && self.profile_id == manifest.profile_id
            && self.loader_id == manifest.loader_id
            && self.description == manifest.description
            && self.support_status == manifest.support_status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_that_escape_or_replace_store_metadata() {
        for path in [
            "../secret",
            "/absolute",
            "bundle/../secret",
            "bundle\\secret",
            "bundle//file",
        ] {
            assert!(!valid_relative_path(path), "{path}");
        }
        assert!(valid_relative_path("bundle/model/head.safetensors"));
    }

    #[test]
    fn requires_immutable_model_names() {
        assert!(valid_name("qwen35-state-first:a047d6802c3f06f085b8"));
        for name in ["qwen35", "../bad:v1", "a:b:c", "A:b", "a:", ".a:v1"] {
            assert!(!valid_name(name), "{name}");
        }
    }

    #[test]
    fn rejects_artifacts_that_replace_install_metadata_or_parent_paths() {
        let artifact = |path: &str| Artifact {
            path: path.into(),
            size: 1,
            sha256: "a".repeat(64),
            source: Source {
                kind: "github".into(),
                repository: "example/models".into(),
                revision: "b".repeat(40),
                path: "source.bin".into(),
            },
        };
        let mut manifest = Manifest {
            schema: MANIFEST_SCHEMA.into(),
            name: "fixture:v1".into(),
            profile_id: "test-profile".into(),
            loader_id: "test-loader".into(),
            description: "fixture".into(),
            release_date: "2026-09-25".into(),
            support_status: "rust-loadable".into(),
            question_types: vec!["choice".into()],
            artifacts: vec![artifact("manifest.json")],
        };
        assert!(manifest.validate().is_err());
        manifest.artifacts = vec![artifact("manifest.json/subfile")];
        assert!(manifest.validate().is_err());
        manifest.artifacts = vec![artifact("bundle"), artifact("bundle/head.bin")];
        assert!(manifest.validate().is_err());
    }
}
