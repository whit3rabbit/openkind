//! `docs/MODELS.md` mirrors the curated catalog in `registry/v1/catalog.json`
//! for operators. These tests fail when the page and the catalog drift apart
//! in either direction: every curated model must be named by pull name and
//! carry a table row, and the page may not name models the catalog does not
//! carry. `tests/catalog.rs` separately pins the catalog digest to the
//! production mirror, so this transitively covers the public registry too.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use openkind_model_store::Catalog;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn curated_catalog() -> Catalog {
    let bytes = std::fs::read(repo_root().join("registry/v1/catalog.json"))
        .expect("registry/v1/catalog.json is checked in");
    let catalog: Catalog = serde_json::from_slice(&bytes).expect("catalog parses");
    catalog.validate().expect("catalog validates");
    catalog
}

/// Tokens shaped like a catalog model name: `loader-id:profile-id`, where the
/// loader is lowercase-hyphen and the profile ID is 16-20 hex digits. Split
/// on the punctuation that can sit next to a name in prose, tables, or links.
fn catalog_shaped_names(doc: &str) -> BTreeSet<String> {
    doc.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '`' | '|' | '.' | ',' | '(' | ')' | '[' | ']' | '"' | '\''
            )
    })
    .filter(|token| {
        let (loader, profile_id) = match token.split_once(':') {
            Some(parts) => parts,
            None => return false,
        };
        !loader.is_empty()
            && loader
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            && (16..=20).contains(&profile_id.len())
            && profile_id.bytes().all(|b| b.is_ascii_hexdigit())
    })
    .map(str::to_owned)
    .collect()
}

fn models_doc() -> String {
    std::fs::read_to_string(repo_root().join("docs/MODELS.md"))
        .expect("docs/MODELS.md is checked in")
}

#[test]
fn models_doc_names_exactly_the_curated_catalog_models() {
    let catalog = curated_catalog();
    let doc = models_doc();

    let catalog_names: BTreeSet<String> = catalog.models.iter().map(|m| m.name.clone()).collect();
    assert_eq!(
        catalog_shaped_names(&doc),
        catalog_names,
        "docs/MODELS.md pull names must equal the curated catalog exactly; \
         update the All-models table in the same change as registry/v1"
    );
}

#[test]
fn models_doc_has_a_table_row_for_every_curated_model() {
    let catalog = curated_catalog();
    let doc = models_doc();

    for model in &catalog.models {
        let named = format!("`{}`", model.name);
        let row = doc
            .lines()
            .find(|line| line.trim_start().starts_with('|') && line.contains(&named));
        assert!(
            row.is_some(),
            "docs/MODELS.md needs an All-models table row naming {named}"
        );
    }
}

/// Parse the `## Pull-name aliases` section of docs/MODELS.md into
/// `(alias, canonical pull name)` pairs. Table prose is stripped of
/// backticks; header and separator rows are skipped.
fn documented_alias_pairs(doc: &str) -> BTreeSet<(String, String)> {
    let section = doc
        .split_once("## Pull-name aliases")
        .expect("docs/MODELS.md needs a `## Pull-name aliases` section")
        .1;
    let section = section.split("\n## ").next().expect("section body");
    let mut pairs = BTreeSet::new();
    for line in section.lines().filter(|l| l.trim_start().starts_with('|')) {
        let cells: Vec<String> = line
            .split('|')
            .skip(1)
            .map(|cell| cell.trim().trim_matches('`').to_owned())
            .collect();
        if cells.len() < 2 || cells[0].is_empty() || cells[0] == "Alias" {
            continue;
        }
        if cells[0].bytes().all(|b| b == b'-' || b == b' ') {
            continue;
        }
        pairs.insert((cells[0].clone(), cells[1].clone()));
    }
    pairs
}

#[test]
fn models_doc_alias_table_equals_the_catalog_aliases() {
    let catalog = curated_catalog();
    let doc = models_doc();

    let expected: BTreeSet<(String, String)> = catalog
        .models
        .iter()
        .flat_map(|model| {
            model
                .aliases
                .iter()
                .map(|alias| (alias.clone(), model.name.clone()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        documented_alias_pairs(&doc),
        expected,
        "the docs/MODELS.md alias table must equal the catalog aliases in both directions"
    );
}
