//! Characterization of the laya load-time precedence.
//!
//! Existence runs over all five artifacts first (the checkpoint is skipped
//! on an ONNX-only load when absent), then agent-config digest and contract,
//! encoder-config digest and contract, the two tokenizer digests, and the
//! checkpoint digest last.

use serde_json::{json, Value};

use super::VerifiedArtifacts;
use crate::families::laya::{LayaProfile, LAYA_ENGLISH};
use crate::families::support::testing::{config_from_pins, json_bytes, leak, put, sha};
use crate::families::support::FamilyError;

struct Fixture {
    dir: tempfile::TempDir,
    profile: LayaProfile,
}

impl Fixture {
    fn new() -> Self {
        Self::with_configs(None, None)
    }

    fn with_configs(agent: Option<Value>, encoder: Option<Value>) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut profile = LAYA_ENGLISH.clone();
        let agent = agent.unwrap_or_else(|| config_from_pins(&profile.pinned_agent_config()));
        let encoder = encoder.unwrap_or_else(|| config_from_pins(&profile.pinned_encoder_config()));
        let root = dir.path();
        let agent_bytes = json_bytes(&agent);
        put(root, "rl_agent_config.json", &agent_bytes);
        profile.agent_config_sha256 = leak(sha(&agent_bytes));
        let encoder_bytes = json_bytes(&encoder);
        put(root, "encoder/config.json", &encoder_bytes);
        profile.encoder_config_sha256 = leak(sha(&encoder_bytes));
        put(root, "tokenizer/tokenizer.json", b"tokenizer");
        profile.tokenizer_json_sha256 = leak(sha(b"tokenizer"));
        put(root, "tokenizer/tokenizer_config.json", b"tokenizer-config");
        profile.tokenizer_config_sha256 = leak(sha(b"tokenizer-config"));
        put(root, "model.safetensors", b"weights");
        profile.checkpoint_sha256 = leak(sha(b"weights"));
        Self { dir, profile }
    }

    fn verify(&self, native: bool) -> Result<(), FamilyError> {
        VerifiedArtifacts::verify_artifacts(self.dir.path(), &self.profile, native).map(drop)
    }

    fn tamper(&self, name: &str) {
        put(self.dir.path(), name, b"tampered");
    }

    fn remove(&self, name: &str) {
        std::fs::remove_file(self.dir.path().join(name)).expect("remove fixture file");
    }
}

fn not_found(error: &FamilyError) -> Option<(String, String)> {
    match error {
        FamilyError::Io { path, source } if source.kind() == std::io::ErrorKind::NotFound => {
            Some((
                path.file_name().unwrap().to_string_lossy().into_owned(),
                source.to_string(),
            ))
        }
        _ => None,
    }
}

fn digest_mismatch_for(error: &FamilyError, name: &str) -> bool {
    matches!(error, FamilyError::DigestMismatch { path, .. } if path.ends_with(name))
}

#[test]
fn a_complete_root_verifies() {
    Fixture::new().verify(true).expect("complete pinned root");
}

#[test]
fn existence_runs_in_declared_order_before_any_digest() {
    // Order: checkpoint, tokenizer, tokenizer_config, encoder config, agent.
    let fixture = Fixture::new();
    fixture.tamper("model.safetensors");
    fixture.remove("rl_agent_config.json");
    fixture.remove("encoder/config.json");
    let error = fixture.verify(true).expect_err("missing artifact");
    assert_eq!(
        not_found(&error),
        Some((
            "config.json".to_owned(),
            "required pinned artifact is missing".to_owned()
        )),
        "{error}"
    );
    assert!(
        matches!(&error, FamilyError::Io { path, .. } if path.ends_with("encoder/config.json")),
        "{error}"
    );
}

#[test]
fn a_missing_checkpoint_is_the_first_existence_failure_on_a_native_load() {
    let fixture = Fixture::new();
    fixture.remove("model.safetensors");
    fixture.remove("tokenizer/tokenizer.json");
    let error = fixture.verify(true).expect_err("missing artifact");
    assert_eq!(
        not_found(&error).map(|found| found.0),
        Some("model.safetensors".to_owned()),
        "{error}"
    );
}

#[test]
fn agent_digest_is_checked_first() {
    let fixture = Fixture::new();
    fixture.tamper("encoder/config.json");
    fixture.tamper("rl_agent_config.json");
    let error = fixture.verify(true).expect_err("bad digest");
    assert!(
        digest_mismatch_for(&error, "rl_agent_config.json"),
        "{error}"
    );
}

#[test]
fn agent_contract_runs_before_the_encoder_digest() {
    let probe = Fixture::new();
    let mut agent = config_from_pins(&probe.profile.pinned_agent_config());
    agent["head_layers"] = json!(7);
    let fixture = Fixture::with_configs(Some(agent), None);
    fixture.tamper("encoder/config.json");
    let error = fixture.verify(true).expect_err("contract mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "head_layers", expected, actual }
                if expected == "2" && actual == "7"
        ),
        "{error}"
    );
}

#[test]
fn missing_field_reports_missing() {
    let probe = Fixture::new();
    let mut agent = config_from_pins(&probe.profile.pinned_agent_config());
    agent.as_object_mut().unwrap().remove("head_layers");
    let fixture = Fixture::with_configs(Some(agent), None);
    let error = fixture.verify(true).expect_err("missing field");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "head_layers", expected, actual }
                if expected == "2" && actual == "missing"
        ),
        "{error}"
    );
}

#[test]
fn encoder_contract_runs_before_the_tokenizer_digests() {
    let probe = Fixture::new();
    let mut encoder = config_from_pins(&probe.profile.pinned_encoder_config());
    encoder["model_type"] = json!("wrong");
    let fixture = Fixture::with_configs(None, Some(encoder));
    fixture.tamper("tokenizer/tokenizer_config.json");
    let error = fixture.verify(true).expect_err("contract mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch {
                field: "model_type",
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn tokenizer_config_digest_precedes_the_tokenizer_digest() {
    let fixture = Fixture::new();
    fixture.tamper("tokenizer/tokenizer.json");
    fixture.tamper("tokenizer/tokenizer_config.json");
    let error = fixture.verify(true).expect_err("bad digest");
    assert!(
        digest_mismatch_for(&error, "tokenizer_config.json"),
        "{error}"
    );
}

#[test]
fn floats_match_within_the_parser_epsilon_and_integers_exactly() {
    let probe = Fixture::new();
    let mut encoder = config_from_pins(&probe.profile.pinned_encoder_config());
    encoder["norm_eps"] = json!(1.00005e-5);
    Fixture::with_configs(None, Some(encoder))
        .verify(true)
        .expect("float within tolerance");

    let mut encoder = config_from_pins(&probe.profile.pinned_encoder_config());
    encoder["norm_eps"] = json!(1.0002e-5);
    let error = Fixture::with_configs(None, Some(encoder))
        .verify(true)
        .expect_err("float outside tolerance");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch {
                field: "norm_eps",
                ..
            }
        ),
        "{error}"
    );

    let mut encoder = config_from_pins(&probe.profile.pinned_encoder_config());
    encoder["local_attention"] = json!(129);
    let error = Fixture::with_configs(None, Some(encoder))
        .verify(true)
        .expect_err("integer mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "local_attention", expected, actual }
                if expected == "128" && actual == "129"
        ),
        "{error}"
    );
}

#[test]
fn a_non_native_load_skips_an_absent_checkpoint_but_verifies_a_present_one() {
    let fixture = Fixture::new();
    fixture.remove("model.safetensors");
    fixture.verify(false).expect("optional checkpoint");

    fixture.tamper("model.safetensors");
    let error = fixture
        .verify(false)
        .expect_err("present bytes still verify");
    assert!(digest_mismatch_for(&error, "model.safetensors"), "{error}");
}
