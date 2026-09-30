//! Guards the wire-contract invariant from AGENTS.md: the committed JSON
//! Schemas under `crates/openkind-core/schemas/` must exactly match what
//! `schemars` derives from the `openkind-core` types. If a struct field
//! changes without re-running `cargo run -p openkind-gen-schemas -- --write`,
//! this test fails.

use openkind_core::{SystemRequest, SystemResponse};

fn schema_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR for this test is crates/openkind-gen-schemas.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../openkind-core/schemas")
        .canonicalize()
        .expect("schemas directory must exist next to openkind-core")
}

#[test]
fn committed_request_schema_matches_core_types() {
    let schema = schemars::schema_for!(SystemRequest);
    let generated = serde_json::to_value(&schema).unwrap();
    let raw = std::fs::read_to_string(schema_dir().join("jev-v1-request.json"))
        .expect("committed request schema must exist");
    let committed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        generated, committed,
        "jev-v1-request.json is stale — run `cargo run -p openkind-gen-schemas -- --write`"
    );
    assert_eq!(
        raw,
        format!("{}\n", serde_json::to_string_pretty(&schema).unwrap())
    );
}

#[test]
fn committed_response_schema_matches_core_types() {
    let schema = schemars::schema_for!(SystemResponse);
    let generated = serde_json::to_value(&schema).unwrap();
    let raw = std::fs::read_to_string(schema_dir().join("jev-v1-response.json"))
        .expect("committed response schema must exist");
    let committed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        generated, committed,
        "jev-v1-response.json is stale — run `cargo run -p openkind-gen-schemas -- --write`"
    );
    assert_eq!(
        raw,
        format!("{}\n", serde_json::to_string_pretty(&schema).unwrap())
    );
}

#[test]
fn cli_rejects_unknown_flags_before_writing() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_openkind-gen-schemas"))
        .args(["--write", "--writ"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("unknown argument"));
}

#[test]
fn cli_prints_deterministic_schemas_from_other_directory_and_environment() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_openkind-gen-schemas"))
        .current_dir(std::env::temp_dir())
        .env("CARGO_MANIFEST_DIR", "/nonexistent/openkind-schema-source")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let request = std::fs::read_to_string(schema_dir().join("jev-v1-request.json")).unwrap();
    let response = std::fs::read_to_string(schema_dir().join("jev-v1-response.json")).unwrap();
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("REQUEST:{request}RESPONSE:{response}")
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
