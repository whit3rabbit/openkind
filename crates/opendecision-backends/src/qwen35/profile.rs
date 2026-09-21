use std::collections::HashMap;
use std::fs;
use std::path::Path;

use opendecision_engine::{
    ArtifactIdentity, ExecutionSemantics, ModelExecutionProfile, ParityContract, ProbabilitySpace,
    ProfileSource,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;

use super::{
    require_equal, sha256_hex, Qwen35Error, ScoreSummaryHead, BACKBONE_ID, BACKBONE_REVISION,
    BUNDLE_SHA256, CALIBRATION_TEMPERATURE, HEAD_FILE, MANIFEST_SHA256, ORDERING_TOLERANCE,
    POLICY_THRESHOLD, PROBABILITY_TOLERANCE, PROFILE_ID, REFERENCE_REPOSITORY, REFERENCE_REVISION,
};

const MANIFEST_SCHEMA: &str = "opendecision-model-bundle/v1";
const MANIFEST_VERSION: &str = "2ij.2.0";
const PROFILE_NAME: &str = "score_summary_seed17";
const PROFILE_METHOD: &str = "frozen";
const PROFILE_LAYOUT: &str = "state_first";
const PROFILE_REJECTION: &str = "score_summary";
const ARTIFACT_DIR: &str = "model";

/// Probability space declared by the selected profile.
///
/// Single source of truth for the profile construction and the evidence
/// mapping: the adapter's none-mass behavior and the recorded declaration can
/// never drift apart.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::OfferedOptionsPlusSemanticNone;

const REQUIRED_MANIFEST_FILES: &[&str] = &[
    "HEAD_GRAPH.json",
    "MODEL_CONTRACT.json",
    "golden.json",
    "inference_protocol.json",
    "profile.json",
    "model/score_summary_seed17.safetensors",
];

// These receipts were published at the pinned repository revision after the
// bundle manifest. Their revision-specific hashes are recorded in SOURCE.md.
const SUPPLEMENTAL_FILES: &[(&str, &str)] = &[
    (
        "HEAD_ALGEBRA_CHECK.json",
        "0ea4c52a86cb46662444ef978a312940cf01b2ea01acc50094cb42927bfd8937",
    ),
    (
        "RELOAD_CHECK.json",
        "c465a65559476571f0fd502fc5d6646a51c4e6588008fccc9f2958c586f3b314",
    ),
];

/// Verified selected-profile metadata and fitted readout.
#[derive(Debug, Clone)]
pub struct ReferenceBundle {
    profile: ModelExecutionProfile,
    head: ScoreSummaryHead,
}

impl ReferenceBundle {
    /// Load the selected Phase 3.1 artifacts from a bundle directory.
    ///
    /// Every artifact used by this tranche is checked against its pinned
    /// manifest or source-revision digest before its contents are trusted.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, Qwen35Error> {
        let root = root.as_ref();
        let manifest_path = root.join("BUNDLE_MANIFEST.json");
        let manifest_bytes = read(&manifest_path)?;
        verify_digest("BUNDLE_MANIFEST.json", MANIFEST_SHA256, &manifest_bytes)?;
        let manifest: BundleManifest = decode_json(&manifest_path, &manifest_bytes)?;
        validate_manifest(&manifest)?;

        let mut verified = HashMap::new();
        for relative in REQUIRED_MANIFEST_FILES {
            verified.insert(*relative, read_verified(root, &manifest, relative)?);
        }
        for (relative, digest) in SUPPLEMENTAL_FILES {
            verify_digest(relative, digest, &read(&root.join(relative))?)?;
        }

        let profile_path = root.join("profile.json");
        let profile_json: ProfileJson = decode_json(
            &profile_path,
            verified
                .get("profile.json")
                .expect("profile is a required manifest file"),
        )?;
        validate_profile(&profile_json)?;

        let protocol_path = root.join("inference_protocol.json");
        let protocol: InferenceProtocol = decode_json(
            &protocol_path,
            verified
                .get("inference_protocol.json")
                .expect("protocol is a required manifest file"),
        )?;
        validate_protocol(&protocol)?;

        let source = ProfileSource::new(
            PROFILE_ID,
            REFERENCE_REPOSITORY,
            REFERENCE_REVISION,
            BUNDLE_SHA256,
        )?;
        let backbone = ArtifactIdentity::new(BACKBONE_ID, BACKBONE_REVISION)?;
        let execution = ExecutionSemantics::new(
            PROFILE_LAYOUT,
            format!("{ARTIFACT_DIR}/{HEAD_FILE}"),
            PROFILE_REJECTION,
            // The selected profile's scores are probabilities over the offered
            // options plus explicit semantic-none mass; the adapter reports
            // none mass under `__none__` without discarding or renormalizing.
            DECLARED_PROBABILITY_SPACE,
        )?;
        let parity = ParityContract::new(
            profile_json.calibration.selected,
            profile_json.policy.selected.threshold,
            protocol.probability_tolerance,
            protocol.order_tolerance,
        )?;
        let profile = ModelExecutionProfile::new(source, backbone, execution, parity);
        let head = ScoreSummaryHead::from_safetensors(
            verified
                .get("model/score_summary_seed17.safetensors")
                .expect("head is a required manifest file"),
            profile.parity().calibration_temperature(),
            profile.parity().policy_threshold(),
        )?;

        Ok(Self { profile, head })
    }

    /// Immutable model/execution contract represented by this bundle.
    pub fn profile(&self) -> &ModelExecutionProfile {
        &self.profile
    }

    /// Validated selected score-summary head.
    pub fn head(&self) -> &ScoreSummaryHead {
        &self.head
    }
}

fn read(path: &Path) -> Result<Vec<u8>, Qwen35Error> {
    fs::read(path).map_err(|source| Qwen35Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn decode_json<T: DeserializeOwned>(path: &Path, bytes: &[u8]) -> Result<T, Qwen35Error> {
    serde_json::from_slice(bytes).map_err(|source| Qwen35Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn read_verified(
    root: &Path,
    manifest: &BundleManifest,
    relative: &str,
) -> Result<Vec<u8>, Qwen35Error> {
    let expected = manifest
        .files
        .get(relative)
        .ok_or_else(|| Qwen35Error::MissingManifestEntry(relative.into()))?;
    let path = root.join(relative);
    let bytes = read(&path)?;
    verify_digest(relative, expected, &bytes)?;
    Ok(bytes)
}

fn verify_digest(relative: &str, expected: &str, bytes: &[u8]) -> Result<(), Qwen35Error> {
    let actual = sha256_hex(bytes);
    if actual == expected {
        Ok(())
    } else {
        Err(Qwen35Error::DigestMismatch {
            path: relative.into(),
            expected: expected.into(),
            actual,
        })
    }
}

fn validate_manifest(manifest: &BundleManifest) -> Result<(), Qwen35Error> {
    require_equal("manifest.schema", MANIFEST_SCHEMA, &manifest.schema)?;
    require_equal("manifest.version", MANIFEST_VERSION, &manifest.version)?;
    require_equal("manifest.profile_id", PROFILE_ID, &manifest.profile_id)?;
    require_equal(
        "manifest.base_weights_included",
        false,
        manifest.base_weights_included,
    )?;
    require_equal("manifest.base.id", BACKBONE_ID, &manifest.base.id)?;
    require_equal(
        "manifest.base.revision",
        BACKBONE_REVISION,
        &manifest.base.revision,
    )?;
    Ok(())
}

fn validate_profile(profile: &ProfileJson) -> Result<(), Qwen35Error> {
    require_equal("profile.profile_id", PROFILE_ID, &profile.profile_id)?;
    require_equal("profile.name", PROFILE_NAME, &profile.name)?;
    require_equal("profile.kind", PROFILE_REJECTION, &profile.kind)?;
    require_equal("profile.method", PROFILE_METHOD, &profile.method)?;
    require_equal("profile.layout", PROFILE_LAYOUT, &profile.layout)?;
    require_equal("profile.artifact_dir", ARTIFACT_DIR, &profile.artifact_dir)?;
    require_equal("profile.head_file", HEAD_FILE, &profile.head_file)?;
    require_equal("profile.spec.id", BACKBONE_ID, &profile.spec.id)?;
    require_equal(
        "profile.spec.revision",
        BACKBONE_REVISION,
        &profile.spec.revision,
    )?;
    require_equal(
        "profile.calibration.selected",
        CALIBRATION_TEMPERATURE,
        profile.calibration.selected,
    )?;
    require_equal(
        "profile.policy.selected.threshold",
        POLICY_THRESHOLD,
        profile.policy.selected.threshold,
    )?;
    Ok(())
}

fn validate_protocol(protocol: &InferenceProtocol) -> Result<(), Qwen35Error> {
    require_equal(
        "protocol.probability_tolerance",
        PROBABILITY_TOLERANCE,
        protocol.probability_tolerance,
    )?;
    require_equal(
        "protocol.order_tolerance",
        ORDERING_TOLERANCE,
        protocol.order_tolerance,
    )?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct BundleManifest {
    schema: String,
    version: String,
    profile_id: String,
    files: HashMap<String, String>,
    base_weights_included: bool,
    base: BackboneJson,
}

#[derive(Debug, Deserialize)]
struct BackboneJson {
    id: String,
    revision: String,
}

#[derive(Debug, Deserialize)]
struct ProfileJson {
    profile_id: String,
    name: String,
    kind: String,
    method: String,
    layout: String,
    artifact_dir: String,
    head_file: String,
    spec: BackboneJson,
    calibration: CalibrationJson,
    policy: PolicyJson,
}

#[derive(Debug, Deserialize)]
struct CalibrationJson {
    selected: f64,
}

#[derive(Debug, Deserialize)]
struct PolicyJson {
    selected: SelectedPolicyJson,
}

#[derive(Debug, Deserialize)]
struct SelectedPolicyJson {
    threshold: f64,
}

#[derive(Debug, Deserialize)]
struct InferenceProtocol {
    probability_tolerance: f64,
    order_tolerance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_digest_verification_fails_closed() {
        let bytes = b"pinned artifact";
        let digest = sha256_hex(bytes);
        assert!(verify_digest("artifact", &digest, bytes).is_ok());
        assert!(matches!(
            verify_digest("artifact", &"0".repeat(64), bytes),
            Err(Qwen35Error::DigestMismatch { .. })
        ));
    }
}
