//! `registry/v1/jev-gev-mlx-models.json` pins two JEV-protocol MLX conversions
//! as supplemental research metadata. Like the other supplemental indexes it
//! is not a catalog: nothing in it may claim to be installable or loadable,
//! every revision must look like a real commit, and every file carries the
//! size and SHA-256 later qualification will verify.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use openkind_model_store::Catalog;
use serde::Deserialize;

#[derive(Deserialize)]
struct Index {
    schema: String,
    installable: bool,
    reference_implementation: serde_json::Value,
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    name: String,
    protocol: String,
    repository: String,
    revision: String,
    license: String,
    source_model: SourceModel,
    decision_config: serde_json::Value,
    files: Vec<File>,
    openkind_loadable: bool,
    blocker: String,
}

#[derive(Deserialize)]
struct SourceModel {
    repository: String,
    revision: String,
}

#[derive(Deserialize)]
struct File {
    path: String,
    size: u64,
    sha256: String,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn index() -> Index {
    let bytes = std::fs::read(repo_root().join("registry/v1/jev-gev-mlx-models.json"))
        .expect("registry/v1/jev-gev-mlx-models.json is checked in");
    serde_json::from_slice(&bytes).expect("jev-gev-mlx-models.json parses")
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[test]
fn index_is_research_metadata_not_a_catalog() {
    let index = index();
    assert_eq!(index.schema, "openkind-jev-gev-mlx/v1");
    assert!(!index.installable);
    let protocols: Vec<&str> = index.entries.iter().map(|e| e.protocol.as_str()).collect();
    assert_eq!(protocols, ["jev", "gev"]);
    for entry in &index.entries {
        assert!(
            !entry.openkind_loadable && !entry.blocker.is_empty(),
            "{} must record why it is not loadable",
            entry.name
        );
    }
}

#[test]
fn every_revision_is_a_full_commit_sha() {
    let index = index();
    for key in ["jev", "gev"] {
        let reference = &index.reference_implementation[key];
        assert!(
            is_hex(reference["revision"].as_str().unwrap(), 40),
            "{key} reference"
        );
    }
    for entry in &index.entries {
        assert!(is_hex(&entry.revision, 40), "{} revision", entry.name);
        assert!(
            is_hex(&entry.source_model.revision, 40),
            "{} source",
            entry.name
        );
        assert!(!entry.repository.is_empty() && !entry.source_model.repository.is_empty());
        assert_eq!(entry.license, "apache-2.0");
    }
}

#[test]
fn files_pin_size_and_digest_and_cover_the_loader_inputs() {
    for entry in &index().entries {
        let paths: BTreeSet<&str> = entry.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths.len(),
            entry.files.len(),
            "{} repeats a file",
            entry.name
        );
        for required in [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ] {
            assert!(paths.contains(required), "{} lacks {required}", entry.name);
        }
        let shards = entry
            .files
            .iter()
            .filter(|f| f.path.ends_with(".safetensors"))
            .count();
        assert_eq!(shards, 6, "{} pins six weight shards", entry.name);
        for file in &entry.files {
            assert!(
                file.size > 0 && is_hex(&file.sha256, 64),
                "{} {}",
                entry.name,
                file.path
            );
        }
    }
}

#[test]
fn decision_configs_have_the_three_pinned_slot_ranges() {
    for entry in &index().entries {
        let config = &entry.decision_config;
        assert_eq!(
            config["ranges"],
            serde_json::json!({"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}),
            "{}",
            entry.name
        );
        match entry.protocol.as_str() {
            "jev" => assert!(config["verbalizer_ids"].is_array() && config["softcap"].is_null()),
            "gev" => assert!(config["softcap"].is_number() && config["verbalizer_ids"].is_null()),
            other => panic!("unknown protocol {other}"),
        }
    }
}

#[test]
fn entries_never_collide_with_curated_catalog_names() {
    let bytes = std::fs::read(repo_root().join("registry/v1/catalog.json")).unwrap();
    let catalog: Catalog = serde_json::from_slice(&bytes).unwrap();
    for entry in &index().entries {
        assert!(
            !catalog.models.iter().any(|m| m.name.contains(&entry.name)),
            "{} must not shadow a curated catalog name",
            entry.name
        );
    }
}
