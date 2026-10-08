//! `registry/v1/jev-decision-index.json` is the supplemental research index
//! for the top open reproductions on the Jev Decision Index leaderboard
//! (https://huggingface.co/spaces/multimodalart/jev-decision-index). It is
//! research metadata, not a catalog: nothing in it may claim to be
//! installable or loadable, and every pinned repository revision must look
//! like a real Git/Hugging Face commit. These tests keep the file structurally
//! honest so later qualification starts from immutable sources.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use openkind_model_store::Catalog;
use serde::Deserialize;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn index_bytes() -> &'static Vec<u8> {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        std::fs::read(repo_root().join("registry/v1/jev-decision-index.json"))
            .expect("registry/v1/jev-decision-index.json is checked in")
    })
}

#[derive(Deserialize)]
struct Index {
    schema: String,
    #[allow(dead_code)]
    checked_at: String,
    installable: bool,
    source: Source,
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct Source {
    #[allow(dead_code)]
    bundle_sha256: String,
    #[allow(dead_code)]
    space_revision: String,
    ranking_field: String,
}

#[derive(Deserialize)]
struct Entry {
    rank: u32,
    engine: String,
    decision_index: f64,
    #[serde(default)]
    decision_index_raw: Option<f64>,
    base_model: BaseModel,
    #[serde(default)]
    weights: Option<Weights>,
    openkind_loadable: bool,
    blocker: String,
    #[serde(default)]
    mlx_leads: Vec<Lead>,
}

#[derive(Deserialize)]
struct BaseModel {
    repository: String,
    #[serde(default)]
    revision: Option<String>,
}

#[derive(Deserialize)]
struct Weights {
    repository: String,
    revision: String,
    format: String,
    license: String,
}

#[derive(Deserialize)]
struct Lead {
    repository: String,
    revision: String,
    relationship: String,
}

fn parsed_index() -> &'static Index {
    static INDEX: OnceLock<Index> = OnceLock::new();
    INDEX.get_or_init(|| {
        serde_json::from_slice(index_bytes()).expect("jev-decision-index.json parses")
    })
}

fn is_commit_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

#[test]
fn index_declares_research_metadata_not_a_catalog() {
    let index = parsed_index();
    assert_eq!(index.schema, "openkind-jev-decision-index/v1");
    assert!(
        !index.installable,
        "the supplemental index must never mark its entries installable"
    );
    assert_eq!(
        index.source.ranking_field, "scores.balanced_skill",
        "the Space ranks on suite.headline = balanced_skill; re-verify the ranking field when the edition changes"
    );
}

#[test]
fn entries_cover_the_top_ten_in_rank_order() {
    let index = parsed_index();
    assert_eq!(index.entries.len(), 10, "the scope is the top 10 minus Jev");
    for (expected, entry) in (1u32..=10).zip(&index.entries) {
        assert_eq!(entry.rank, expected, "ranks must be contiguous from 1");
        assert!(
            !entry.engine.is_empty() && !entry.blocker.is_empty(),
            "every entry names its engine and records why it is not loadable"
        );
        assert!(
            !entry.openkind_loadable,
            "{} must stay non-loadable until a family adapter and parity fixtures land",
            entry.engine
        );
        assert!(
            entry.decision_index > 0.0 && entry.decision_index < 100.0,
            "decision_index is a chance-corrected percentage"
        );
        if let Some(raw) = entry.decision_index_raw {
            assert!(
                raw >= entry.decision_index,
                "balanced_raw is never chance-corrected, so it dominates balanced_skill"
            );
        }
    }
    let ranks_descending = index
        .entries
        .windows(2)
        .all(|pair| pair[0].decision_index > pair[1].decision_index);
    assert!(
        ranks_descending,
        "entries must stay sorted by the headline score"
    );
}

#[test]
fn every_pinned_revision_is_a_full_commit_sha() {
    let index = parsed_index();
    let mut repositories = BTreeSet::new();
    for entry in &index.entries {
        assert!(
            !entry.base_model.repository.is_empty(),
            "every entry records its base model"
        );
        if let Some(revision) = &entry.base_model.revision {
            assert!(
                is_commit_revision(revision),
                "{} base revision must be a 40-hex commit sha",
                entry.engine
            );
        }
        if let Some(weights) = &entry.weights {
            assert!(
                !weights.format.is_empty() && !weights.license.is_empty(),
                "pinned weights describe their format and license"
            );
            assert!(
                is_commit_revision(&weights.revision),
                "{} weights revision must be a 40-hex commit sha",
                entry.engine
            );
            assert!(
                repositories.insert(weights.repository.clone()),
                "duplicate pinned weights repository in the index"
            );
        }
        for lead in &entry.mlx_leads {
            assert!(
                !lead.repository.is_empty(),
                "{} mlx leads name their repository",
                entry.engine
            );
            assert!(
                is_commit_revision(&lead.revision),
                "{} mlx lead revision must be a 40-hex commit sha",
                entry.engine
            );
            assert!(
                matches!(
                    lead.relationship.as_str(),
                    "base-model-only" | "fine-tune-conversion"
                ),
                "mlx lead relationship is one of the two surveyed kinds"
            );
        }
    }
}

#[test]
fn technique_entries_carry_no_weights_and_fine_tunes_do() {
    const WEIGHTS_FREE_ENGINES: [&str; 3] = [
        "simple-jev-qwen3.8-27b",
        "reflex-27b-v2",
        "joshua-diffusion-full",
    ];
    let index = parsed_index();
    for entry in &index.entries {
        let expects_weights = !WEIGHTS_FREE_ENGINES.contains(&entry.engine.as_str());
        assert_eq!(
            entry.weights.is_some(),
            expects_weights,
            "{} must {} pinned weights",
            entry.engine,
            if expects_weights {
                "carry"
            } else {
                "not carry"
            }
        );
        assert!(
            !entry.base_model.repository.is_empty(),
            "every entry names the checkpoint it serves"
        );
    }
}

#[test]
fn entries_never_collide_with_curated_catalog_names() {
    let catalog_bytes = std::fs::read(repo_root().join("registry/v1/catalog.json"))
        .expect("registry/v1/catalog.json is checked in");
    let catalog: Catalog = serde_json::from_slice(&catalog_bytes).expect("catalog parses");
    let catalog_names: BTreeSet<String> = catalog.models.iter().map(|m| m.name.clone()).collect();
    let index = parsed_index();
    for entry in &index.entries {
        assert!(
            !catalog_names
                .iter()
                .any(|name| name.contains(&entry.engine)),
            "supplemental entry {} must not shadow a curated catalog name",
            entry.engine
        );
    }
}
