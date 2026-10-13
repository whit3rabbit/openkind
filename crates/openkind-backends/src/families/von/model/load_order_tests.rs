//! Characterization of the von load-time precedence.
//!
//! Von interleaves digests and contracts: existence of the four small
//! artifacts, then `config.json` digest and contract, then calibration digest
//! and contract, then the two tokenizer digests, then the checkpoint. A
//! helper that batched all digests ahead of the contracts would change which
//! error wins, so each test names one adjacent pair.

use serde_json::{json, Value};

use super::VerifiedArtifacts;
use crate::families::support::testing::{config_from_pins, json_bytes, leak, put, sha};
use crate::families::support::FamilyError;
use crate::families::von::{VonProfile, VON};

struct Fixture {
    dir: tempfile::TempDir,
    profile: VonProfile,
}

impl Fixture {
    fn new() -> Self {
        Self::with_configs(None, None)
    }

    fn with_configs(config: Option<Value>, calibration: Option<Value>) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut profile = VON.clone();
        let config = config.unwrap_or_else(|| config_from_pins(&profile.pinned_config()));
        let calibration =
            calibration.unwrap_or_else(|| config_from_pins(&profile.pinned_calibration()));
        let root = dir.path();
        let config_bytes = json_bytes(&config);
        put(root, "checkpoint/config.json", &config_bytes);
        profile.config_sha256 = leak(sha(&config_bytes));
        let calibration_bytes = json_bytes(&calibration);
        put(
            root,
            "checkpoint/marker_calibration.json",
            &calibration_bytes,
        );
        profile.calibration_sha256 = leak(sha(&calibration_bytes));
        put(root, "checkpoint/tokenizer.json", b"tokenizer");
        profile.tokenizer_json_sha256 = leak(sha(b"tokenizer"));
        put(
            root,
            "checkpoint/tokenizer_config.json",
            b"tokenizer-config",
        );
        profile.tokenizer_config_sha256 = leak(sha(b"tokenizer-config"));
        put(root, "checkpoint/option_marker.pt", b"weights");
        profile.checkpoint_sha256 = leak(sha(b"weights"));
        Self { dir, profile }
    }

    fn verify(&self, native: bool) -> Result<(), FamilyError> {
        VerifiedArtifacts::verify_artifacts(self.dir.path(), &self.profile, native).map(drop)
    }

    fn tamper(&self, name: &str) {
        put(self.dir.path(), &format!("checkpoint/{name}"), b"tampered");
    }

    fn remove(&self, name: &str) {
        std::fs::remove_file(self.dir.path().join("checkpoint").join(name))
            .expect("remove fixture file");
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
fn an_empty_root_reports_the_first_required_artifact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error = match VerifiedArtifacts::verify(dir.path(), &VON) {
        Ok(_) => panic!("empty root must fail"),
        Err(error) => error,
    };
    assert_eq!(
        not_found(&error),
        Some((
            "config.json".to_owned(),
            "required pinned artifact is missing".to_owned()
        )),
        "{error}"
    );
}

#[test]
fn every_small_artifact_must_exist_before_any_digest_runs() {
    let fixture = Fixture::new();
    fixture.tamper("config.json");
    fixture.remove("marker_calibration.json");
    let error = fixture.verify(true).expect_err("missing artifact");
    assert_eq!(
        not_found(&error).map(|found| found.0),
        Some("marker_calibration.json".to_owned()),
        "{error}"
    );
}

#[test]
fn small_artifacts_are_checked_for_existence_in_declared_order() {
    let fixture = Fixture::new();
    fixture.remove("tokenizer_config.json");
    fixture.remove("tokenizer.json");
    let error = fixture.verify(true).expect_err("missing artifact");
    assert_eq!(
        not_found(&error).map(|found| found.0),
        Some("tokenizer.json".to_owned()),
        "{error}"
    );
}

#[test]
fn config_digest_is_checked_first() {
    let fixture = Fixture::new();
    fixture.tamper("marker_calibration.json");
    fixture.tamper("config.json");
    let error = fixture.verify(true).expect_err("bad digest");
    assert!(digest_mismatch_for(&error, "config.json"), "{error}");
}

#[test]
fn config_contract_runs_before_the_calibration_digest() {
    let probe = Fixture::new();
    let mut config = config_from_pins(&probe.profile.pinned_config());
    config["model_type"] = json!("wrong");
    let fixture = Fixture::with_configs(Some(config), None);
    fixture.tamper("marker_calibration.json");
    let error = fixture.verify(true).expect_err("contract mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "model_type", expected, actual }
                if expected == "\"modernbert\"" && actual == "\"wrong\""
        ),
        "{error}"
    );
}

#[test]
fn missing_config_field_reports_missing() {
    let probe = Fixture::new();
    let mut config = config_from_pins(&probe.profile.pinned_config());
    config
        .as_object_mut()
        .unwrap()
        .remove("rope_parameters")
        .expect("pinned nested table");
    let fixture = Fixture::with_configs(Some(config), None);
    let error = fixture.verify(true).expect_err("missing field");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch {
                field: "rope_parameters.full_attention.rope_theta",
                actual,
                ..
            } if actual == "missing"
        ),
        "{error}"
    );
}

#[test]
fn calibration_contract_runs_before_the_tokenizer_digests() {
    let probe = Fixture::new();
    let mut calibration = config_from_pins(&probe.profile.pinned_calibration());
    calibration["calibration_map"]["hi"] = json!(99.0);
    let fixture = Fixture::with_configs(None, Some(calibration));
    fixture.tamper("tokenizer_config.json");
    let error = fixture.verify(true).expect_err("contract mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch {
                field: "calibration_map.hi",
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn tokenizer_config_digest_precedes_the_tokenizer_digest() {
    let fixture = Fixture::new();
    fixture.tamper("tokenizer.json");
    fixture.tamper("tokenizer_config.json");
    let error = fixture.verify(true).expect_err("bad digest");
    assert!(
        digest_mismatch_for(&error, "tokenizer_config.json"),
        "{error}"
    );
}

#[test]
fn floats_match_within_the_parser_epsilon_and_integers_exactly() {
    let probe = Fixture::new();
    // 5e-10 apart is inside the 1e-9 tolerance von and laya share.
    let mut config = config_from_pins(&probe.profile.pinned_config());
    config["norm_eps"] = json!(1.00005e-5);
    Fixture::with_configs(Some(config), None)
        .verify(true)
        .expect("float within tolerance");

    // 2e-9 apart is outside it, and the message prints both documents' values.
    let mut config = config_from_pins(&probe.profile.pinned_config());
    config["norm_eps"] = json!(1.0002e-5);
    let error = Fixture::with_configs(Some(config), None)
        .verify(true)
        .expect_err("float outside tolerance");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch { field: "norm_eps", expected, actual }
                if expected == "0.00001" && actual == "0.000010002"
        ),
        "{error}"
    );

    // Integers never get the float tolerance.
    let mut config = config_from_pins(&probe.profile.pinned_config());
    config["hidden_size"] = json!(probe.profile.encoder.hidden_size + 1);
    let error = Fixture::with_configs(Some(config), None)
        .verify(true)
        .expect_err("integer mismatch");
    assert!(
        matches!(
            &error,
            FamilyError::ContractMismatch {
                field: "hidden_size",
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_native_load_without_a_checkpoint_fails_on_the_digest_open() {
    let fixture = Fixture::new();
    fixture.remove("option_marker.pt");
    let error = fixture.verify(true).expect_err("missing checkpoint");
    // The checkpoint is not in the existence pass: the digest opens it.
    assert!(
        matches!(&error, FamilyError::Io { path, source }
            if path.ends_with("option_marker.pt")
                && source.kind() == std::io::ErrorKind::NotFound
                && source.to_string() != "required pinned artifact is missing"),
        "{error}"
    );
}

#[test]
fn a_non_native_load_tolerates_a_missing_checkpoint_but_not_a_bad_one() {
    let fixture = Fixture::new();
    fixture.remove("option_marker.pt");
    fixture.verify(false).expect("optional checkpoint");

    fixture.tamper("option_marker.pt");
    let error = fixture
        .verify(false)
        .expect_err("present bytes still verify");
    assert!(digest_mismatch_for(&error, "option_marker.pt"), "{error}");
}
