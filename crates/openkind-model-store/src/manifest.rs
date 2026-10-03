use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const CATALOG_SCHEMA: &str = "openkind-catalog/v1";
pub const MANIFEST_SCHEMA: &str = "openkind-model/v1";
/// Upper bound for all artifacts in one model installation.
///
/// Sized above the largest pinned artifact (the Cloudflare Clef 27B GGUF
/// Q4_K_M backbone at 17.2 GB) with headroom for its companion artifacts.
pub const MAX_MODEL_BYTES: u64 = 32 * 1024 * 1024 * 1024;

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
    /// Alternative pull names shaped `name:tag` (the schema the ollaya
    /// decision-model registry uses). An alias resolves to `name` at pull
    /// time; installations are always keyed by the canonical `name`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    pub profile_id: String,
    pub loader_id: String,
    pub description: String,
    /// Maximum total input sequence length in tokens that the pinned
    /// checkpoint's own configuration declares (`max_position_embeddings`
    /// in Hugging Face configs, `context_length` in GGUF metadata).
    pub context_limit: u64,
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
                || entry.context_limit == 0
            {
                return Err(Error::Invalid(format!(
                    "invalid catalog entry {}",
                    entry.name
                )));
            }
        }
        // An alias must be a well-formed `name:tag`, must not collide with a
        // canonical name or another alias, and must stay unique so a pull
        // request resolves to exactly one curated profile.
        let mut aliases = HashSet::new();
        for entry in &self.models {
            for alias in &entry.aliases {
                if !valid_name(alias) || names.contains(alias) || !aliases.insert(alias.as_str()) {
                    return Err(Error::Invalid(format!(
                        "invalid catalog alias {alias} for {}",
                        entry.name
                    )));
                }
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
        let mut total_size = 0_u64;
        for artifact in &self.artifacts {
            // The metadata write must not alias a linked blob on case-insensitive filesystems.
            let metadata_alias = artifact
                .path
                .split('/')
                .next()
                .is_some_and(|root| root.eq_ignore_ascii_case("manifest.json"));
            if !valid_relative_path(&artifact.path)
                || metadata_alias
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
            total_size = total_size
                .checked_add(artifact.size)
                .ok_or_else(|| Error::Invalid("model artifact size overflow".into()))?;
            if total_size > MAX_MODEL_BYTES {
                return Err(Error::Invalid(format!(
                    "model artifacts exceed the {MAX_MODEL_BYTES}-byte installation limit"
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

    fn catalog_with_entries(entries: &[(&str, &[&str])]) -> Catalog {
        Catalog {
            schema: CATALOG_SCHEMA.into(),
            models: entries
                .iter()
                .map(|(name, aliases)| CatalogEntry {
                    name: (*name).into(),
                    aliases: aliases.iter().map(|a| (*a).to_owned()).collect(),
                    profile_id: "profile".into(),
                    loader_id: "loader".into(),
                    description: "fixture".into(),
                    context_limit: 8192,
                    support_status: "rust-loadable".into(),
                    manifest_path: "manifests/fixture.json".into(),
                    manifest_sha256: "a".repeat(64),
                })
                .collect(),
        }
    }

    #[test]
    fn catalog_accepts_unique_well_formed_aliases() {
        let catalog = catalog_with_entries(&[
            ("fixture:aaaa", &["alias:one", "alias:two"]),
            ("other:bbbb", &[]),
        ]);
        catalog.validate().expect("unique aliases validate");
    }

    #[test]
    fn catalog_rejects_alias_collisions_and_bad_shapes() {
        let cases: &[&[(&str, &[&str])]] = &[
            // An alias may not collide with a canonical name.
            &[("fixture:aaaa", &["other:bbbb"]), ("other:bbbb", &[])],
            // An alias may not collide with another entry's alias.
            &[
                ("fixture:aaaa", &["alias:one"]),
                ("other:bbbb", &["alias:one"]),
            ],
            // Aliases must be valid `name:tag` model names.
            &[("fixture:aaaa", &["Alias:one"])],
            &[("fixture:aaaa", &["no-tag"])],
            &[("fixture:aaaa", &["a:b:c"])],
        ];
        for entries in cases {
            let catalog = catalog_with_entries(entries);
            assert!(
                catalog.validate().is_err(),
                "catalog must reject {entries:?}"
            );
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
        for path in ["MANIFEST.JSON", "Manifest.Json/subfile"] {
            manifest.artifacts = vec![artifact(path)];
            assert!(
                manifest.validate().is_err(),
                "reserved metadata alias: {path}"
            );
        }
        manifest.artifacts = vec![artifact("bundle"), artifact("bundle/head.bin")];
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn rejects_models_over_the_installation_size_limit() {
        let artifact = |path: &str, size| Artifact {
            path: path.into(),
            size,
            sha256: "a".repeat(64),
            source: Source {
                kind: "github".into(),
                repository: "example/models".into(),
                revision: "b".repeat(40),
                path: "source.bin".into(),
            },
        };
        let manifest = Manifest {
            schema: MANIFEST_SCHEMA.into(),
            name: "fixture:v1".into(),
            profile_id: "test-profile".into(),
            loader_id: "test-loader".into(),
            description: "fixture".into(),
            release_date: "2026-09-25".into(),
            support_status: "rust-loadable".into(),
            question_types: vec!["choice".into()],
            artifacts: vec![artifact("one.bin", MAX_MODEL_BYTES), artifact("two.bin", 1)],
        };
        assert!(manifest.validate().is_err());
    }
}

#[cfg(test)]
mod context_limit_tests {
    use super::*;

    /// The context-window floor is a catalog-ingestion invariant: an entry
    /// without a positive `context_limit` is not installable.
    #[test]
    fn catalog_rejects_zero_context_limits() {
        let mut catalog = Catalog {
            schema: CATALOG_SCHEMA.into(),
            models: vec![CatalogEntry {
                name: "fixture:aaaa".into(),
                aliases: vec![],
                profile_id: "profile".into(),
                loader_id: "loader".into(),
                description: "fixture".into(),
                context_limit: 0,
                support_status: "rust-loadable".into(),
                manifest_path: "manifests/fixture.json".into(),
                manifest_sha256: "a".repeat(64),
            }],
        };
        let error = catalog.validate().expect_err("zero context must fail");
        assert!(
            error.to_string().contains("fixture:aaaa"),
            "unexpected error: {error}"
        );

        catalog.models[0].context_limit = 1;
        catalog
            .validate()
            .expect("the smallest positive limit passes");
    }

    #[test]
    fn catalog_rejects_bad_schema_duplicate_names_and_metadata() {
        let base = Catalog {
            schema: CATALOG_SCHEMA.into(),
            models: vec![CatalogEntry {
                name: "fixture:aaaa".into(),
                aliases: vec![],
                profile_id: "profile".into(),
                loader_id: "loader".into(),
                description: "fixture".into(),
                context_limit: 8192,
                support_status: "rust-loadable".into(),
                manifest_path: "manifests/fixture.json".into(),
                manifest_sha256: "a".repeat(64),
            }],
        };
        let mut catalog = base.clone();
        catalog.schema = "other/v1".into();
        assert!(catalog.validate().is_err(), "wrong schema must fail");

        let mut catalog = base.clone();
        catalog.models.push(CatalogEntry {
            name: "fixture:aaaa".into(),
            ..base.models[0].clone()
        });
        assert!(catalog.validate().is_err(), "duplicate names must fail");

        let mut catalog = base.clone();
        catalog.models[0].manifest_path = "elsewhere/fixture.json".into();
        assert!(catalog.validate().is_err(), "non-manifests path must fail");

        let mut catalog = base.clone();
        catalog.models[0].manifest_path = "../fixture.json".into();
        assert!(catalog.validate().is_err(), "escaping path must fail");

        let mut catalog = base.clone();
        catalog.models[0].manifest_sha256 = "A".repeat(64);
        assert!(catalog.validate().is_err(), "uppercase digest must fail");

        let mut catalog = base.clone();
        catalog.models[0].profile_id = String::new();
        assert!(catalog.validate().is_err(), "empty profile id must fail");

        let mut catalog = base.clone();
        catalog.models[0].loader_id = String::new();
        assert!(catalog.validate().is_err(), "empty loader id must fail");

        let mut catalog = base.clone();
        catalog.models[0].name = "Fixture:aaaa".into();
        assert!(catalog.validate().is_err(), "invalid name must fail");
    }

    #[test]
    fn valid_sha256_rejects_wrong_length_charset_and_case() {
        assert!(valid_sha256(&"a".repeat(64)));
        assert!(!valid_sha256(&"A".repeat(64)), "uppercase must fail");
        assert!(!valid_sha256(&"g".repeat(64)), "non-hex must fail");
        assert!(!valid_sha256(&"a".repeat(63)), "short digest must fail");
        assert!(!valid_sha256(&"a".repeat(65)), "long digest must fail");
    }

    #[test]
    fn manifest_rejects_bad_shapes_field_by_field() {
        let valid_artifact = || Artifact {
            path: "bundle/head.bin".into(),
            size: 1,
            sha256: "a".repeat(64),
            source: Source {
                kind: "github".into(),
                repository: "example/models".into(),
                revision: "b".repeat(40),
                path: "source.bin".into(),
            },
        };
        let valid_manifest = || Manifest {
            schema: MANIFEST_SCHEMA.into(),
            name: "fixture:v1".into(),
            profile_id: "test-profile".into(),
            loader_id: "test-loader".into(),
            description: "fixture".into(),
            release_date: "2026-09-25".into(),
            support_status: "rust-loadable".into(),
            question_types: vec!["choice".into()],
            artifacts: vec![valid_artifact()],
        };
        valid_manifest().validate().expect("fixture must validate");

        let mut manifest = valid_manifest();
        manifest.artifacts.clear();
        assert!(manifest.validate().is_err(), "empty artifacts must fail");

        let mut manifest = valid_manifest();
        manifest.artifacts.push(valid_artifact());
        assert!(manifest.validate().is_err(), "duplicate paths must fail");

        let mut manifest = valid_manifest();
        manifest.artifacts[0].size = 0;
        assert!(manifest.validate().is_err(), "zero size must fail");

        let mut manifest = valid_manifest();
        manifest.artifacts[0].sha256 = "A".repeat(64);
        assert!(
            manifest.validate().is_err(),
            "bad artifact digest must fail"
        );

        let mut manifest = valid_manifest();
        manifest.description = String::new();
        assert!(manifest.validate().is_err(), "empty description must fail");

        let mut manifest = valid_manifest();
        manifest.release_date = "2026-9-5".into();
        assert!(manifest.validate().is_err(), "short release date must fail");

        let mut manifest = valid_manifest();
        manifest.support_status = "beta".into();
        assert!(
            manifest.validate().is_err(),
            "unknown support status must fail"
        );

        let mut manifest = valid_manifest();
        manifest.question_types = vec![];
        assert!(
            manifest.validate().is_err(),
            "empty question types must fail"
        );

        let mut manifest = valid_manifest();
        manifest.question_types = vec!["alchemy".into()];
        assert!(
            manifest.validate().is_err(),
            "unknown question type must fail"
        );

        let mut manifest = valid_manifest();
        manifest.artifacts[0].size = MAX_MODEL_BYTES;
        manifest.artifacts[0].path = "bundle/one.bin".into();
        manifest.artifacts.push(valid_artifact());
        manifest.artifacts[1].size = 1;
        manifest.artifacts[1].path = "bundle/two.bin".into();
        assert!(
            manifest.validate().is_err(),
            "overflowing total size must fail"
        );
    }

    #[test]
    fn source_rejects_bad_shapes_field_by_field() {
        let source = |kind: &str, repository: &str, revision: &str, path: &str| Source {
            kind: kind.into(),
            repository: repository.into(),
            revision: revision.into(),
            path: path.into(),
        };
        for source in [
            source("gitlab", "example/models", &"b".repeat(40), "source.bin"),
            source("", "example/models", &"b".repeat(40), "source.bin"),
            source("github", "example", &"b".repeat(40), "source.bin"),
            source("github", "a/b/c", &"b".repeat(40), "source.bin"),
            source("github", "example/models", &"b".repeat(39), "source.bin"),
            source("github", "example/models", &"g".repeat(40), "source.bin"),
            source("github", "example/models", &"b".repeat(40), "../escape"),
        ] {
            let manifest = Manifest {
                schema: MANIFEST_SCHEMA.into(),
                name: "fixture:v1".into(),
                profile_id: "test-profile".into(),
                loader_id: "test-loader".into(),
                description: "fixture".into(),
                release_date: "2026-09-25".into(),
                support_status: "rust-loadable".into(),
                question_types: vec!["choice".into()],
                artifacts: vec![Artifact {
                    path: "bundle/head.bin".into(),
                    size: 1,
                    sha256: "a".repeat(64),
                    source,
                }],
            };
            assert!(
                manifest.validate().is_err(),
                "source must be rejected: {:?}",
                manifest.artifacts[0].source
            );
        }
    }

    #[test]
    fn source_urls_pin_github_and_huggingface_layouts() {
        let github = Source {
            kind: "github".into(),
            repository: "example/models".into(),
            revision: "b".repeat(40),
            path: "weights/head.bin".into(),
        };
        assert_eq!(
            github.url(),
            format!(
                "https://raw.githubusercontent.com/example/models/{}/weights/head.bin",
                "b".repeat(40)
            )
        );
        let huggingface = Source {
            kind: "huggingface".into(),
            repository: "example/models".into(),
            revision: "c".repeat(40),
            path: "weights/head.bin".into(),
        };
        assert_eq!(
            huggingface.url(),
            format!(
                "https://huggingface.co/example/models/resolve/{}/weights/head.bin",
                "c".repeat(40)
            )
        );
    }
}
