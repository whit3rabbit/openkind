use serde::Deserialize;

/// One expected full-sequence output vector from the Phase 3B export.
#[derive(Debug, Clone, Deserialize)]
pub struct FullSequenceRecord {
    pub(super) fixture_case: usize,
    pub(super) question_id: String,
    pub(super) candidate_index: usize,
    pub(super) token_count: usize,
    pub(super) tensor_key: String,
    pub(super) max_abs_delta_vs_existing_bundle_feature: f64,
}

impl FullSequenceRecord {
    /// Index of the source golden request.
    #[must_use]
    pub const fn fixture_case(&self) -> usize {
        self.fixture_case
    }

    /// Question identifier within the source request.
    #[must_use]
    pub fn question_id(&self) -> &str {
        &self.question_id
    }

    /// Candidate index within the question.
    #[must_use]
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    /// Full rendered token count.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    /// Key of the expected final hidden vector.
    #[must_use]
    pub fn tensor_key(&self) -> &str {
        &self.tensor_key
    }

    /// Diagnostic drift between the fresh Python run and older bundle feature.
    #[must_use]
    pub const fn reference_feature_delta(&self) -> f64 {
        self.max_abs_delta_vs_existing_bundle_feature
    }
}

/// One embedding, decoder-layer, or final-normalization trace stage.
#[derive(Debug, Clone, Deserialize)]
pub struct TraceStage {
    pub(super) stage: String,
    pub(super) tensor_key: String,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: String,
    pub(super) sha256: String,
}

impl TraceStage {
    /// Human-readable stage name such as `embedding` or `layer_03`.
    #[must_use]
    pub fn stage(&self) -> &str {
        &self.stage
    }

    /// Key of the expected diagnostic vector.
    #[must_use]
    pub fn tensor_key(&self) -> &str {
        &self.tensor_key
    }
}

/// Numeric diagnostics for one actual vector against its frozen reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageComparison {
    pub(super) max_abs: f64,
    pub(super) rms: f64,
    pub(super) cosine: f64,
}

impl StageComparison {
    /// Largest elementwise absolute difference.
    #[must_use]
    pub const fn max_abs(self) -> f64 {
        self.max_abs
    }

    /// Root-mean-square elementwise difference.
    #[must_use]
    pub const fn rms(self) -> f64 {
        self.rms
    }

    /// Cosine similarity between actual and reference vectors.
    #[must_use]
    pub const fn cosine(self) -> f64 {
        self.cosine
    }
}
