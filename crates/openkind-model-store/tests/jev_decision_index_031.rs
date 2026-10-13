//! `registry/v1/jev-decision-index-0.3.1.json` is the supplemental research
//! index for the Decision Index 0.3.1 leaders (superseding the 0.2.1 file,
//! which is retained as history). Like the 0.2.1 index it is research
//! metadata, not a catalog: nothing in it may claim to be installable, only
//! entries that landed as catalog profiles may claim to be loadable, and every
//! pinned repository revision must look like a real Git / Hugging Face commit.
//! These tests keep the file structurally honest.

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
        std::fs::read(repo_root().join("registry/v1/jev-decision-index-0.3.1.json"))
            .expect("registry/v1/jev-decision-index-0.3.1.json is checked in")
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
    base_models: Vec<BaseEntry>,
    excluded: Vec<Excluded>,
}

#[derive(Deserialize)]
struct Source {
    #[allow(dead_code)]
    bundle_sha256: String,
    #[allow(dead_code)]
    space_revision: String,
    ranking_field: String,
    #[allow(dead_code)]
    supersedes: String,
}

#[derive(Deserialize)]
struct Entry {
    engine: String,
    tier: u8,
    board: Option<Board>,
    #[serde(default)]
    off_board_scores: Option<OffBoard>,
    #[serde(default)]
    already_catalog: bool,
    base_model: BaseModel,
    weights: Weights,
    openkind_loadable: bool,
    #[serde(default)]
    catalog_profile: Option<String>,
    #[serde(default)]
    blocker: Option<String>,
    #[allow(dead_code)]
    nearest_openkind_path: String,
    #[serde(default)]
    mlx_leads: Vec<Lead>,
}

#[derive(Deserialize)]
struct Board {
    rank: u32,
    full: f64,
    #[allow(dead_code)]
    public: f64,
    #[allow(dead_code)]
    same_skills: f64,
    #[allow(dead_code)]
    new_domains: f64,
}

#[derive(Deserialize)]
struct OffBoard {
    #[allow(dead_code)]
    source: String,
    #[serde(default)]
    #[allow(dead_code)]
    full: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    public: Option<f64>,
    #[allow(dead_code)]
    note: String,
}

#[derive(Deserialize)]
struct BaseModel {
    #[allow(dead_code)]
    repository: String,
    #[serde(default)]
    revision: Option<String>,
    #[allow(dead_code)]
    architecture: String,
}

#[derive(Deserialize)]
struct Weights {
    repository: String,
    revision: String,
    format: String,
    license: String,
    #[serde(default)]
    head_at_survey: String,
    #[serde(default)]
    equivalent_revisions: Vec<String>,
}

#[derive(Deserialize)]
struct Lead {
    #[allow(dead_code)]
    repository: String,
    revision: String,
    relationship: String,
}

#[derive(Deserialize)]
struct BaseEntry {
    repository: String,
    revision: String,
    #[serde(default)]
    used_by: Vec<String>,
}

#[derive(Deserialize)]
struct Excluded {
    #[allow(dead_code)]
    entry: String,
    #[allow(dead_code)]
    reason: String,
}

fn parsed_index() -> &'static Index {
    static INDEX: OnceLock<Index> = OnceLock::new();
    INDEX.get_or_init(|| {
        serde_json::from_slice(index_bytes()).expect("jev-decision-index-0.3.1.json parses")
    })
}

fn is_commit_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

#[test]
fn index_declares_research_metadata_not_a_catalog() {
    let index = parsed_index();
    assert_eq!(index.schema, "openkind-jev-decision-index/v2");
    assert!(
        !index.installable,
        "the supplemental index must never mark its entries installable"
    );
    assert_eq!(
        index.source.ranking_field, "v",
        "the 0.3.1 Space ranks on the Full score v; re-verify the ranking field when the edition changes"
    );
    assert_eq!(
        index.source.supersedes, "registry/v1/jev-decision-index.json",
        "the 0.2.1 file stays as history and must be named"
    );
}

#[test]
fn tier1_leaders_come_first_and_boards_are_consistent() {
    let index = parsed_index();
    let tier1_count = index.entries.iter().filter(|entry| entry.tier == 1).count();
    assert_eq!(
        tier1_count, 12,
        "the handoff pins twelve tier-1 repositories"
    );
    let mut board_ranks = BTreeSet::new();
    for (position, entry) in index.entries.iter().enumerate() {
        assert!(
            (entry.tier == 1 && position < tier1_count)
                || (entry.tier == 2 && position >= tier1_count),
            "{}: tier 1 entries must precede tier 2",
            entry.engine
        );
        assert!(!entry.engine.is_empty(), "every entry names its engine");
        if let Some(board) = &entry.board {
            assert!(
                board.full > 0.0 && board.full < 100.0,
                "{}: board scores are percentages",
                entry.engine
            );
            assert!(
                board_ranks.insert(board.rank),
                "{}: board ranks must be unique",
                entry.engine
            );
        } else {
            let off_board = entry.off_board_scores.as_ref();
            assert!(
                entry.already_catalog || off_board.is_some(),
                "{}: off-board entries record where their number comes from",
                entry.engine
            );
            if let Some(off_board) = off_board {
                assert!(
                    !off_board.note.is_empty(),
                    "{}: off-board entries explain their source",
                    entry.engine
                );
            }
        }
    }
}

#[test]
fn every_pinned_revision_is_a_full_commit_sha() {
    let index = parsed_index();
    let mut repositories = BTreeSet::new();
    let mut engines = BTreeSet::new();
    for entry in &index.entries {
        assert!(
            engines.insert(entry.engine.clone()),
            "duplicate engine {}",
            entry.engine
        );
        if let Some(revision) = &entry.base_model.revision {
            assert!(
                is_commit_revision(revision),
                "{} base revision must be a 40-hex commit sha",
                entry.engine
            );
        }
        assert!(
            !entry.weights.format.is_empty() && !entry.weights.license.is_empty(),
            "{} weights describe their format and license",
            entry.engine
        );
        for revision in std::iter::once(&entry.weights.revision)
            .chain(std::iter::once(&entry.weights.head_at_survey))
            .chain(entry.weights.equivalent_revisions.iter())
        {
            assert!(
                is_commit_revision(revision),
                "{} weights revision must be a 40-hex commit sha",
                entry.engine
            );
        }
        assert!(
            repositories.insert(entry.weights.repository.clone()),
            "duplicate pinned weights repository in the index"
        );
        for lead in &entry.mlx_leads {
            assert!(
                is_commit_revision(&lead.revision),
                "{} mlx lead revision must be a 40-hex commit sha",
                entry.engine
            );
            assert!(
                matches!(
                    lead.relationship.as_str(),
                    "exact-conversion-unverified" | "base-model-only"
                ),
                "{} mlx lead relationship is one of the two surveyed kinds",
                entry.engine
            );
        }
    }
    for model in &index.base_models {
        assert!(
            is_commit_revision(&model.revision),
            "base model {} must be pinned to a 40-hex commit sha",
            model.repository
        );
        for user in &model.used_by {
            assert!(
                engines.contains(user),
                "base model {} names unknown user {}",
                model.repository,
                user
            );
        }
    }
}

#[test]
fn board_rows_match_the_pinned_bundle_leaders() {
    // Spot pins transcribed from the hashed 0.3.1 bundle (data/v03.json at
    // space revision f6571b3); the tie band means ranks 1-2 are one group.
    let index = parsed_index();
    let expected = [
        ("vllm-sr-d3", 1u32, 65.65),
        ("decider-31b", 2, 65.56),
        ("darwin-27b-ztc", 3, 63.21),
        ("pplx-decider-v1.1-27b", 4, 62.75),
        ("torchcast-decision-27b", 12, 59.91),
    ];
    for (engine, rank, full) in expected {
        let entry = index
            .entries
            .iter()
            .find(|entry| entry.engine == engine)
            .unwrap_or_else(|| panic!("{engine} is pinned"));
        let board = entry
            .board
            .as_ref()
            .expect("{engine} carries its board row");
        assert_eq!(board.rank, rank, "{engine} rank");
        assert!((board.full - full).abs() < 1e-9, "{engine} Full score");
    }
}

#[test]
fn catalog_families_are_marked_and_other_engines_do_not_shadow_the_catalog() {
    let catalog_bytes = std::fs::read(repo_root().join("registry/v1/catalog.json"))
        .expect("registry/v1/catalog.json is checked in");
    let catalog: Catalog = serde_json::from_slice(&catalog_bytes).expect("catalog parses");
    let catalog_names: BTreeSet<String> = catalog.models.iter().map(|m| m.name.clone()).collect();
    let index = parsed_index();
    for entry in &index.entries {
        if entry.already_catalog {
            const CURATED_ENGINES: [&str; 8] = [
                "clef",
                "clef-flash",
                "d3-edge",
                "d3-lite",
                "d3-nano",
                "d3-mini",
                "d3-flash",
                "sieve-4b",
            ];
            assert!(
                CURATED_ENGINES.contains(&entry.engine.as_str()),
                "only entries with a curated catalog profile may overlap catalog families: {}",
                entry.engine
            );
            continue;
        }
        assert!(
            !catalog_names
                .iter()
                .any(|name| name.contains(&entry.engine)),
            "supplemental entry {} must not shadow a curated catalog name",
            entry.engine
        );
    }
}

#[test]
fn only_landed_catalog_profiles_are_loadable() {
    // These six landed with a family adapter, parity fixtures, and a catalog
    // profile over the exact pinned weights revision; every other entry stays
    // non-loadable and records its blocker.
    const LANDED: [(&str, &str); 6] = [
        ("d3-edge", "d3-edge:664f9ead734372935286"),
        ("d3-lite", "d3-lite:e0dd32dbc16b27546692"),
        ("d3-nano", "d3-nano:0198af5b5c8e8835605e"),
        ("d3-mini", "d3-mini:434dbc50d3b581663e52"),
        ("d3-flash", "d3-flash:b2a4750f8bf4a1fa58cb"),
        ("sieve-4b", "sieve-4b:8b97b56524b55c41fbf4"),
    ];
    let catalog_bytes = std::fs::read(repo_root().join("registry/v1/catalog.json"))
        .expect("registry/v1/catalog.json is checked in");
    let catalog: Catalog = serde_json::from_slice(&catalog_bytes).expect("catalog parses");
    let index = parsed_index();
    for entry in &index.entries {
        match LANDED.iter().find(|(engine, _)| *engine == entry.engine) {
            Some((_, profile)) => {
                assert!(entry.openkind_loadable, "{} is loadable", entry.engine);
                assert!(
                    entry.already_catalog,
                    "{} is marked as a catalog family",
                    entry.engine
                );
                assert_eq!(entry.catalog_profile.as_deref(), Some(*profile));
                assert!(
                    entry.blocker.is_none(),
                    "{}: a landed entry has no blocker",
                    entry.engine
                );
                let model = catalog
                    .models
                    .iter()
                    .find(|model| model.name == *profile)
                    .unwrap_or_else(|| panic!("{profile} is a curated catalog profile"));
                let manifest: openkind_model_store::Manifest = serde_json::from_slice(
                    &std::fs::read(repo_root().join("registry/v1").join(&model.manifest_path))
                        .expect("catalog manifest is checked in"),
                )
                .expect("manifest parses");
                assert!(
                    manifest.artifacts.iter().any(|artifact| {
                        artifact.source.repository == entry.weights.repository
                            && artifact.source.revision == entry.weights.revision
                    }),
                    "{profile} must load the index's pinned {} revision",
                    entry.weights.repository
                );
            }
            None => {
                assert!(
                    !entry.openkind_loadable && entry.catalog_profile.is_none(),
                    "{} must stay non-loadable until a family adapter and parity fixtures land",
                    entry.engine
                );
                assert!(
                    entry
                        .blocker
                        .as_deref()
                        .is_some_and(|blocker| !blocker.trim().is_empty()),
                    "{} records why it is not loadable",
                    entry.engine
                );
            }
        }
    }
    assert_eq!(
        index
            .entries
            .iter()
            .filter(|entry| entry.openkind_loadable)
            .count(),
        LANDED.len()
    );
}

#[test]
fn excluded_rows_record_why_entries_are_absent() {
    let index = parsed_index();
    assert!(
        index.excluded.len() >= 5,
        "the handoff's exclusion reasons are recorded"
    );
    assert!(
        index
            .excluded
            .iter()
            .any(|row| row.entry.contains("Fastino")),
        "the no-weights board entry is explained"
    );
}
