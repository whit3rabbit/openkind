//! Characterization of the d3 load-time precedence: which error wins when
//! more than one pinned artifact is wrong.
//!
//! The order is existence of the five small artifacts, their digests, the
//! `config.json` contract, the `decision_config.json` contract, then per
//! shard existence, size, and digest, then the readout. Helper extraction
//! must keep it, so each test names one adjacent pair.

use serde_json::{json, Value};

use super::model::VerifiedArtifacts;
use super::{pinned_config, pinned_decision_config, D3Profile, D3_LITE};
use crate::families::support::testing::{config_from_pins, json_bytes, leak, put, sha};
use crate::families::support::FamilyError;

const SHARD_BYTES: &[u8] = b"shard-bytes";

struct Fixture {
    dir: tempfile::TempDir,
    profile: D3Profile,
}

impl Fixture {
    /// A complete, self-consistent model root under a profile whose digests
    /// were computed from the fabricated files.
    fn new() -> Self {
        Self::with_configs(None, None)
    }

    fn with_configs(config: Option<Value>, decision: Option<Value>) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut profile = D3_LITE;
        let config = config.unwrap_or_else(|| config_from_pins(&pinned_config(&profile)));
        let decision =
            decision.unwrap_or_else(|| config_from_pins(&pinned_decision_config(&profile)));
        let root = dir.path();
        let config_bytes = json_bytes(&config);
        put(root, "config.json", &config_bytes);
        profile.config_json_sha256 = leak(sha(&config_bytes));
        let decision_bytes = json_bytes(&decision);
        put(root, "decision_config.json", &decision_bytes);
        profile.decision_config_sha256 = leak(sha(&decision_bytes));
        put(root, "tokenizer.json", b"tokenizer");
        profile.tokenizer_json_sha256 = leak(sha(b"tokenizer"));
        put(root, "tokenizer_config.json", b"tokenizer-config");
        profile.tokenizer_config_json_sha256 = leak(sha(b"tokenizer-config"));
        put(root, "chat_template.jinja", b"template");
        profile.chat_template_sha256 = leak(sha(b"template"));
        put(root, "shard.safetensors", SHARD_BYTES);
        profile.shards = Box::leak(Box::new([(
            "shard.safetensors",
            SHARD_BYTES.len() as u64,
            leak(sha(SHARD_BYTES)),
        )]));
        put(root, "readout.safetensors", b"readout");
        profile.readout_sha256 = leak(sha(b"readout"));
        Self { dir, profile }
    }

    fn verify(&self) -> Result<(), FamilyError> {
        VerifiedArtifacts::verify(self.dir.path(), &self.profile).map(drop)
    }

    fn tamper(&self, name: &str) {
        put(self.dir.path(), name, b"tampered");
    }

    fn remove(&self, name: &str) {
        std::fs::remove_file(self.dir.path().join(name)).expect("remove fixture file");
    }
}

/// File name and message of a NotFound `Io` error.
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
    let fixture = Fixture::new();
    let artifacts = VerifiedArtifacts::verify(fixture.dir.path(), &fixture.profile)
        .expect("complete pinned root");
    assert_eq!(artifacts.shards.len(), 1);
}

#[test]
fn missing_small_artifact_beats_an_earlier_bad_digest() {
    let fixture = Fixture::new();
    fixture.tamper("config.json");
    fixture.remove("chat_template.jinja");
    let error = fixture.verify().expect_err("missing artifact");
    assert_eq!(
        not_found(&error),
        Some((
            "chat_template.jinja".to_owned(),
            "required pinned artifact chat_template.jinja is missing".to_owned()
        )),
        "{error}"
    );
}

#[test]
fn small_artifacts_are_checked_for_existence_in_declared_order() {
    let fixture = Fixture::new();
    fixture.remove("tokenizer.json");
    fixture.remove("decision_config.json");
    let error = fixture.verify().expect_err("missing artifact");
    assert_eq!(
        not_found(&error).map(|found| found.0),
        Some("decision_config.json".to_owned()),
        "{error}"
    );
}

#[test]
fn small_digests_are_checked_in_declared_order() {
    let fixture = Fixture::new();
    fixture.tamper("tokenizer.json");
    fixture.tamper("config.json");
    let error = fixture.verify().expect_err("bad digest");
    assert!(digest_mismatch_for(&error, "config.json"), "{error}");
}

#[test]
fn every_small_digest_is_checked_before_any_contract() {
    // The config fails its contract, but the chat template digest is wrong
    // too, and the digest pass covers every small artifact first.
    let probe = Fixture::new();
    let mut config = config_from_pins(&pinned_config(&probe.profile));
    config["model_type"] = json!("wrong");
    let fixture = Fixture::with_configs(Some(config), None);
    fixture.tamper("chat_template.jinja");
    let error = fixture.verify().expect_err("digest");
    assert!(
        digest_mismatch_for(&error, "chat_template.jinja"),
        "{error}"
    );
}

#[test]
fn config_contract_beats_a_missing_shard() {
    let probe = Fixture::new();
    let mut config = config_from_pins(&pinned_config(&probe.profile));
    config["model_type"] = json!("wrong");
    let fixture = Fixture::with_configs(Some(config), None);
    fixture.remove("shard.safetensors");
    let error = fixture.verify().expect_err("contract mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "model_type", expected, actual }
                if expected == "\"qwen3_5\"" && actual == "\"wrong\""
        ),
        "{error}"
    );
}

#[test]
fn missing_config_field_reports_missing() {
    let probe = Fixture::new();
    let mut config = config_from_pins(&pinned_config(&probe.profile));
    config.as_object_mut().unwrap().remove("model_type");
    let fixture = Fixture::with_configs(Some(config), None);
    let error = fixture.verify().expect_err("missing field");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "model_type", expected, actual }
                if expected == "\"qwen3_5\"" && actual == "missing"
        ),
        "{error}"
    );
}

#[test]
fn config_contract_is_checked_before_the_decision_contract() {
    let probe = Fixture::new();
    let mut config = config_from_pins(&pinned_config(&probe.profile));
    config["model_type"] = json!("wrong");
    let mut decision = config_from_pins(&pinned_decision_config(&probe.profile));
    decision["pooling"] = json!("mean");
    let fixture = Fixture::with_configs(Some(config), Some(decision));
    let error = fixture.verify().expect_err("contract mismatch");
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
fn decision_contract_beats_a_missing_shard() {
    let probe = Fixture::new();
    let mut decision = config_from_pins(&pinned_decision_config(&probe.profile));
    decision["pooling"] = json!("mean");
    let fixture = Fixture::with_configs(None, Some(decision));
    fixture.remove("shard.safetensors");
    let error = fixture.verify().expect_err("contract mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "pooling", expected, actual }
                if expected == "\"last\"" && actual == "\"mean\""
        ),
        "{error}"
    );
}

#[test]
fn rms_epsilon_compares_through_f32() {
    // The config's decimal `1e-6` is the f64 nearest that value, not the
    // widened f32 the profile pins, so a strict comparison would reject it.
    let probe = Fixture::new();
    let mut config = config_from_pins(&pinned_config(&probe.profile));
    config["text_config"]["rms_norm_eps"] = serde_json::from_str("0.000001").unwrap();
    Fixture::with_configs(Some(config), None)
        .verify()
        .expect("decimal epsilon matches the f32 pin");

    let mut config = config_from_pins(&pinned_config(&probe.profile));
    config["text_config"]["rms_norm_eps"] = serde_json::from_str("0.00001").unwrap();
    let error = Fixture::with_configs(Some(config), None)
        .verify()
        .expect_err("different epsilon");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "text_config.rms_norm_eps", expected, actual }
                if expected == "9.999999974752427e-7" && actual == "0.00001"
        ),
        "{error}"
    );

    // A non-numeric epsilon fails closed rather than matching.
    let mut config = config_from_pins(&pinned_config(&probe.profile));
    config["text_config"]["rms_norm_eps"] = json!("1e-6");
    let error = Fixture::with_configs(Some(config), None)
        .verify()
        .expect_err("string epsilon");
    assert!(matches!(
        error,
        FamilyError::ContractMismatch {
            field: "text_config.rms_norm_eps",
            ..
        }
    ));
}

#[test]
fn missing_shard_reports_the_named_message() {
    let fixture = Fixture::new();
    fixture.remove("shard.safetensors");
    let error = fixture.verify().expect_err("missing shard");
    assert_eq!(
        not_found(&error),
        Some((
            "shard.safetensors".to_owned(),
            "required pinned shard shard.safetensors is missing".to_owned()
        )),
        "{error}"
    );
}

#[test]
fn shard_size_is_checked_before_its_digest() {
    let fixture = Fixture::new();
    put(fixture.dir.path(), "shard.safetensors", b"short");
    let error = fixture.verify().expect_err("size mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "checkpoint shard size", expected, actual }
                if expected == "shard.safetensors is 11 bytes" && actual == "5 bytes"
        ),
        "{error}"
    );

    // Same length, different bytes: the size passes and the digest fails.
    put(fixture.dir.path(), "shard.safetensors", b"SHARD-BYTES");
    let error = fixture.verify().expect_err("digest mismatch");
    assert!(digest_mismatch_for(&error, "shard.safetensors"), "{error}");
}

#[test]
fn shard_digest_beats_a_missing_readout() {
    let fixture = Fixture::new();
    put(fixture.dir.path(), "shard.safetensors", b"SHARD-BYTES");
    fixture.remove("readout.safetensors");
    let error = fixture.verify().expect_err("digest mismatch");
    assert!(digest_mismatch_for(&error, "shard.safetensors"), "{error}");
}

#[test]
fn readout_existence_precedes_its_digest() {
    let fixture = Fixture::new();
    fixture.remove("readout.safetensors");
    let error = fixture.verify().expect_err("missing readout");
    assert_eq!(
        not_found(&error),
        Some((
            "readout.safetensors".to_owned(),
            "required pinned artifact readout.safetensors is missing".to_owned()
        )),
        "{error}"
    );

    let fixture = Fixture::new();
    fixture.tamper("readout.safetensors");
    let error = fixture.verify().expect_err("digest mismatch");
    assert!(
        digest_mismatch_for(&error, "readout.safetensors"),
        "{error}"
    );
}
