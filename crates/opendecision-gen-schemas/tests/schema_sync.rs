//! Guards the wire-contract invariant from AGENTS.md: the committed JSON
//! Schemas under `crates/opendecision-core/schemas/` must exactly match what
//! `schemars` derives from the `opendecision-core` types. If a struct field
//! changes without re-running `cargo run -p opendecision-gen-schemas -- --write`,
//! this test fails.

use opendecision_core::{SystemRequest, SystemResponse};

fn schema_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR for this test is crates/opendecision-gen-schemas.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&manifest)
        .join("../opendecision-core/schemas")
        .canonicalize()
        .expect("schemas directory must exist next to opendecision-core")
}

#[test]
fn committed_request_schema_matches_core_types() {
    let generated = serde_json::to_value(schemars::schema_for!(SystemRequest)).unwrap();
    let raw = std::fs::read_to_string(schema_dir().join("jev-v1-request.json"))
        .expect("committed request schema must exist");
    let committed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        generated, committed,
        "jev-v1-request.json is stale — run `cargo run -p opendecision-gen-schemas -- --write`"
    );
}

#[test]
fn committed_response_schema_matches_core_types() {
    let generated = serde_json::to_value(schemars::schema_for!(SystemResponse)).unwrap();
    let raw = std::fs::read_to_string(schema_dir().join("jev-v1-response.json"))
        .expect("committed response schema must exist");
    let committed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        generated, committed,
        "jev-v1-response.json is stale — run `cargo run -p opendecision-gen-schemas -- --write`"
    );
}

#[test]
fn committed_schemas_are_draft_2020_12_objects() {
    for name in ["jev-v1-request.json", "jev-v1-response.json"] {
        let raw = std::fs::read_to_string(schema_dir().join(name)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(
            v.is_object(),
            "{name} must be a JSON Schema object, got {v:?}"
        );
        assert_eq!(
            v.get("$schema").and_then(|s| s.as_str()),
            Some("https://json-schema.org/draft/2020-12/schema"),
            "{name} must declare Draft 2020-12"
        );
    }
}
