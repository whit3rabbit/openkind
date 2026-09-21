//! Immutable model/execution identity and numerical parity contracts.

use thiserror::Error;

/// Validation errors for model execution profile metadata.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ProfileValidationError {
    /// A required text field was empty.
    #[error("profile field `{field}` cannot be empty")]
    EmptyField {
        /// Name of the invalid field.
        field: &'static str,
    },

    /// The reference bundle digest was not a lowercase SHA-256 value.
    #[error("bundle SHA-256 must contain exactly 64 lowercase hexadecimal characters")]
    InvalidBundleSha256,

    /// Calibration temperature must be finite and strictly positive.
    #[error("calibration temperature must be finite and greater than zero")]
    InvalidCalibrationTemperature,

    /// Policy threshold must be a finite probability.
    #[error("policy threshold must be finite and lie in [0, 1]")]
    InvalidPolicyThreshold,

    /// A parity tolerance must be finite and non-negative.
    #[error("{field} must be finite and non-negative")]
    InvalidTolerance {
        /// Name of the invalid tolerance.
        field: &'static str,
    },
}

fn required(
    value: impl Into<String>,
    field: &'static str,
) -> Result<String, ProfileValidationError> {
    let value = value.into();
    if value.trim().is_empty() {
        Err(ProfileValidationError::EmptyField { field })
    } else {
        Ok(value)
    }
}

/// Immutable identity of a model or fitted artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactIdentity {
    id: String,
    revision: String,
}

impl ArtifactIdentity {
    /// Construct an artifact identity from a stable ID and immutable revision.
    pub fn new(
        id: impl Into<String>,
        revision: impl Into<String>,
    ) -> Result<Self, ProfileValidationError> {
        Ok(Self {
            id: required(id, "artifact.id")?,
            revision: required(revision, "artifact.revision")?,
        })
    }

    /// Stable model or artifact identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Immutable artifact revision.
    pub fn revision(&self) -> &str {
        &self.revision
    }
}

/// Provenance for the exported model bundle used as the parity reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSource {
    profile_id: String,
    repository: String,
    repository_revision: String,
    bundle_sha256: String,
}

impl ProfileSource {
    /// Construct pinned reference-bundle provenance.
    pub fn new(
        profile_id: impl Into<String>,
        repository: impl Into<String>,
        repository_revision: impl Into<String>,
        bundle_sha256: impl Into<String>,
    ) -> Result<Self, ProfileValidationError> {
        let bundle_sha256 = bundle_sha256.into();
        if bundle_sha256.len() != 64
            || !bundle_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ProfileValidationError::InvalidBundleSha256);
        }

        Ok(Self {
            profile_id: required(profile_id, "profile_id")?,
            repository: required(repository, "source.repository")?,
            repository_revision: required(repository_revision, "source.repository_revision")?,
            bundle_sha256,
        })
    }

    /// Immutable model/execution profile ID.
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    /// Repository containing the exported reference bundle.
    pub fn repository(&self) -> &str {
        &self.repository
    }

    /// Immutable repository revision from which fixtures were taken.
    pub fn repository_revision(&self) -> &str {
        &self.repository_revision
    }

    /// SHA-256 of the selected exported reference-bundle archive.
    pub fn bundle_sha256(&self) -> &str {
        &self.bundle_sha256
    }
}

/// Probability semantics a profile's scores live in.
///
/// This makes the meaning of returned probabilities an explicit, versioned
/// profile property instead of an implicit adapter behavior. An adapter that
/// cannot serve the declared space must fail explicitly rather than discard
/// or renormalize mass the space promises — silently dropping semantic-none
/// mass and renormalizing the remainder is exactly the drift this declaration
/// exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilitySpace {
    /// Probabilities are conditional on the offered options only: the
    /// distribution over those options sums to one and no none mass is
    /// modeled. A finite-token or direct-logit profile may legitimately
    /// declare this space.
    ConditionalOnOfferedOptions,
    /// Probabilities cover the offered options plus explicit semantic-none
    /// mass: the full distribution includes the reserved none class and sums
    /// to one over options-plus-none. The selected native profile declares
    /// this space.
    OfferedOptionsPlusSemanticNone,
}

impl ProbabilitySpace {
    /// Stable lowercase identifier used in reports and evidence artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConditionalOnOfferedOptions => "conditional_on_offered_options",
            Self::OfferedOptionsPlusSemanticNone => "offered_options_plus_semantic_none",
        }
    }
}

/// Names the learned input and readout semantics of a profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionSemantics {
    renderer: String,
    head: String,
    rejection: String,
    probability_space: ProbabilitySpace,
}

impl ExecutionSemantics {
    /// Construct the renderer, head, rejection, and probability-space
    /// identities.
    pub fn new(
        renderer: impl Into<String>,
        head: impl Into<String>,
        rejection: impl Into<String>,
        probability_space: ProbabilitySpace,
    ) -> Result<Self, ProfileValidationError> {
        Ok(Self {
            renderer: required(renderer, "execution.renderer")?,
            head: required(head, "execution.head")?,
            rejection: required(rejection, "execution.rejection")?,
            probability_space,
        })
    }

    /// Renderer identity, including ordering semantics.
    pub fn renderer(&self) -> &str {
        &self.renderer
    }

    /// Fitted head artifact identity.
    pub fn head(&self) -> &str {
        &self.head
    }

    /// Rejection mechanism identity.
    pub fn rejection(&self) -> &str {
        &self.rejection
    }

    /// Probability space the profile's scores live in.
    #[must_use]
    pub const fn probability_space(&self) -> ProbabilitySpace {
        self.probability_space
    }
}

/// Numerical calibration, policy, and parity requirements for one profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityContract {
    calibration_temperature: f64,
    policy_threshold: f64,
    probability_tolerance: f64,
    ordering_tolerance: f64,
}

impl ParityContract {
    /// Construct numerical parity requirements.
    pub fn new(
        calibration_temperature: f64,
        policy_threshold: f64,
        probability_tolerance: f64,
        ordering_tolerance: f64,
    ) -> Result<Self, ProfileValidationError> {
        if !calibration_temperature.is_finite() || calibration_temperature <= 0.0 {
            return Err(ProfileValidationError::InvalidCalibrationTemperature);
        }
        if !policy_threshold.is_finite() || !(0.0..=1.0).contains(&policy_threshold) {
            return Err(ProfileValidationError::InvalidPolicyThreshold);
        }
        for (field, value) in [
            ("probability_tolerance", probability_tolerance),
            ("ordering_tolerance", ordering_tolerance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ProfileValidationError::InvalidTolerance { field });
            }
        }

        Ok(Self {
            calibration_temperature,
            policy_threshold,
            probability_tolerance,
            ordering_tolerance,
        })
    }

    /// Selected softmax temperature.
    pub fn calibration_temperature(&self) -> f64 {
        self.calibration_temperature
    }

    /// Minimum top probability required for the selected application policy to accept.
    pub fn policy_threshold(&self) -> f64 {
        self.policy_threshold
    }

    /// Maximum full-distribution delta accepted by the exported fixture contract.
    pub fn probability_tolerance(&self) -> f64 {
        self.probability_tolerance
    }

    /// Tolerance used by exported ordering and logit comparisons.
    pub fn ordering_tolerance(&self) -> f64 {
        self.ordering_tolerance
    }
}

/// Immutable contract binding one model, renderer, readout, and numerical policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelExecutionProfile {
    source: ProfileSource,
    backbone: ArtifactIdentity,
    execution: ExecutionSemantics,
    parity: ParityContract,
}

impl ModelExecutionProfile {
    /// Construct a complete immutable execution profile.
    pub fn new(
        source: ProfileSource,
        backbone: ArtifactIdentity,
        execution: ExecutionSemantics,
        parity: ParityContract,
    ) -> Self {
        Self {
            source,
            backbone,
            execution,
            parity,
        }
    }

    /// Reference-bundle provenance.
    pub fn source(&self) -> &ProfileSource {
        &self.source
    }

    /// Pinned base-model identity.
    pub fn backbone(&self) -> &ArtifactIdentity {
        &self.backbone
    }

    /// Renderer, head, and rejection semantics.
    pub fn execution(&self) -> &ExecutionSemantics {
        &self.execution
    }

    /// Calibration, policy, and parity tolerances.
    pub fn parity(&self) -> &ParityContract {
        &self.parity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_contract_is_constructed_from_valid_components() {
        let source = ProfileSource::new(
            "profile",
            "owner/repository",
            "revision",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        let backbone = ArtifactIdentity::new("model", "model-revision").unwrap();
        let execution = ExecutionSemantics::new(
            "state-first",
            "head.safetensors",
            "score-summary",
            ProbabilitySpace::OfferedOptionsPlusSemanticNone,
        )
        .unwrap();
        let parity = ParityContract::new(1.5, 0.98, 0.005, 0.00001).unwrap();

        let profile = ModelExecutionProfile::new(source, backbone, execution, parity);

        assert_eq!(profile.source().profile_id(), "profile");
        assert_eq!(profile.backbone().id(), "model");
        assert_eq!(profile.execution().renderer(), "state-first");
        assert_eq!(profile.parity().policy_threshold(), 0.98);
        assert_eq!(
            profile.execution().probability_space(),
            ProbabilitySpace::OfferedOptionsPlusSemanticNone
        );
        assert_eq!(
            ProbabilitySpace::ConditionalOnOfferedOptions.as_str(),
            "conditional_on_offered_options"
        );
        assert_eq!(
            ProbabilitySpace::OfferedOptionsPlusSemanticNone.as_str(),
            "offered_options_plus_semantic_none"
        );
    }

    #[test]
    fn profile_components_reject_invalid_values() {
        assert!(matches!(
            ArtifactIdentity::new("", "revision"),
            Err(ProfileValidationError::EmptyField { .. })
        ));
        assert_eq!(
            ProfileSource::new("profile", "repository", "revision", "ABC"),
            Err(ProfileValidationError::InvalidBundleSha256)
        );
        assert_eq!(
            ParityContract::new(0.0, 0.98, 0.005, 0.00001),
            Err(ProfileValidationError::InvalidCalibrationTemperature)
        );
        assert_eq!(
            ParityContract::new(1.0, 1.1, 0.005, 0.00001),
            Err(ProfileValidationError::InvalidPolicyThreshold)
        );
        assert!(matches!(
            ParityContract::new(1.0, 0.98, f64::NAN, 0.00001),
            Err(ProfileValidationError::InvalidTolerance { .. })
        ));
    }
}
