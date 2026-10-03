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

/// The committed files must be in sync already (the tests above guarantee
/// it), so `--write` is an idempotent regeneration: it succeeds, emits the
/// same stdout as print mode, and leaves the committed bytes untouched.
#[test]
fn cli_write_mode_is_an_idempotent_regeneration() {
    let before_request = std::fs::read(schema_dir().join("jev-v1-request.json")).unwrap();
    let before_response = std::fs::read(schema_dir().join("jev-v1-response.json")).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_openkind-gen-schemas"))
        .arg("--write")
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Successfully updated"),
        "write mode reports its target directory: {stderr}"
    );

    let after_request = std::fs::read(schema_dir().join("jev-v1-request.json")).unwrap();
    let after_response = std::fs::read(schema_dir().join("jev-v1-response.json")).unwrap();
    assert_eq!(
        before_request, after_request,
        "--write must be a no-op on a synced tree"
    );
    assert_eq!(
        before_response, after_response,
        "--write must be a no-op on a synced tree"
    );

    // Write mode prints the same payload as print mode.
    let print = std::process::Command::new(env!("CARGO_BIN_EXE_openkind-gen-schemas"))
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    assert!(print.status.success());
    assert_eq!(output.stdout, print.stdout);
}

/// `--help` prints usage and generates nothing.
#[test]
fn cli_help_prints_usage_without_touching_schemas() {
    let before_request = std::fs::read(schema_dir().join("jev-v1-request.json")).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_openkind-gen-schemas"))
        .arg("--help")
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Usage:") && stdout.contains("--write"),
        "usage names the write flag: {stdout}"
    );
    assert!(
        !stdout.contains("REQUEST:"),
        "help must not generate schemas"
    );
    assert!(output.stderr.is_empty());
    assert_eq!(
        std::fs::read(schema_dir().join("jev-v1-request.json")).unwrap(),
        before_request,
        "help must not touch the committed schemas"
    );
}

/// Consecutive runs emit byte-identical output: schema generation is a pure
/// function of the compiled wire types.
#[test]
fn consecutive_runs_produce_identical_output() {
    let first = std::process::Command::new(env!("CARGO_BIN_EXE_openkind-gen-schemas"))
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    let second = std::process::Command::new(env!("CARGO_BIN_EXE_openkind-gen-schemas"))
        .current_dir(std::env::temp_dir())
        .env("CARGO_MANIFEST_DIR", "/nonexistent/openkind-schema-source")
        .output()
        .unwrap();
    assert!(first.status.success() && second.status.success());
    assert_eq!(
        first.stdout, second.stdout,
        "generation must not depend on the caller's environment"
    );
}
