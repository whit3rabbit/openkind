//! Type definitions, evaluation structures, and profile constants for the Qwen 3.5 head.

/// Hidden width of the selected Qwen3.5-4B backbone.
pub const FEATURE_WIDTH: usize = 2560;

pub(crate) const SCORE_SUMMARY_WIDTH: usize = 6;

pub(crate) const TENSOR_SPECS: &[(&str, &[usize])] = &[
    ("rank.linear.bias", &[1]),
    ("rank.linear.weight", &[1, FEATURE_WIDTH]),
    ("rank.mean", &[FEATURE_WIDTH]),
    ("rank.none", &[]),
    ("rank.std", &[FEATURE_WIDTH]),
    ("reject.mean", &[SCORE_SUMMARY_WIDTH]),
    ("reject.net.bias", &[1]),
    ("reject.net.weight", &[1, SCORE_SUMMARY_WIDTH]),
    ("reject.std", &[SCORE_SUMMARY_WIDTH]),
];

/// Primitive whose candidate-conditioned features are being evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    /// Choice over one or more caller candidates, plus native semantic none.
    Choice,
    /// Ordered binary false/true decision.
    Noul,
    /// Ordered score rubric with at least two levels.
    Score,
}

/// Frozen application-policy action over a head distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyAction {
    /// Accept the selected caller candidate at this zero-based index.
    Accept {
        /// Candidate index in the supplied feature order.
        candidate_index: usize,
    },
    /// Route to review, including when native semantic none wins.
    Review,
}

/// Complete result of the selected fitted readout.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadEvaluation {
    pub(crate) candidate_logits: Vec<f64>,
    pub(crate) none_logit: Option<f64>,
    pub(crate) candidate_probabilities: Vec<f64>,
    pub(crate) none_probability: Option<f64>,
    pub(crate) selected_candidate_index: Option<usize>,
    pub(crate) top_probability: f64,
    pub(crate) policy_action: PolicyAction,
}

impl HeadEvaluation {
    /// Construct a new `HeadEvaluation` with explicit components.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        candidate_logits: Vec<f64>,
        none_logit: Option<f64>,
        candidate_probabilities: Vec<f64>,
        none_probability: Option<f64>,
        selected_candidate_index: Option<usize>,
        top_probability: f64,
        policy_action: PolicyAction,
    ) -> Self {
        Self {
            candidate_logits,
            none_logit,
            candidate_probabilities,
            none_probability,
            selected_candidate_index,
            top_probability,
            policy_action,
        }
    }

    /// One fitted scalar logit per caller candidate.
    pub fn candidate_logits(&self) -> &[f64] {
        &self.candidate_logits
    }

    /// Native semantic-none logit for Choice, absent for Noul and Score.
    pub fn none_logit(&self) -> Option<f64> {
        self.none_logit
    }

    /// Unconditional probability assigned to each caller candidate.
    pub fn candidate_probabilities(&self) -> &[f64] {
        &self.candidate_probabilities
    }

    /// Native semantic-none probability for Choice.
    pub fn none_probability(&self) -> Option<f64> {
        self.none_probability
    }

    /// Winning caller-candidate index, or `None` when semantic none wins.
    pub fn selected_candidate_index(&self) -> Option<usize> {
        self.selected_candidate_index
    }

    /// Maximum probability across caller candidates and native none.
    pub fn top_probability(&self) -> f64 {
        self.top_probability
    }

    /// Frozen threshold-policy action.
    pub fn policy_action(&self) -> PolicyAction {
        self.policy_action
    }

    /// Full distribution in selected-profile order, including Choice none last.
    pub fn full_probabilities(&self) -> Vec<f64> {
        let mut probabilities = self.candidate_probabilities.clone();
        if let Some(none) = self.none_probability {
            probabilities.push(none);
        }
        probabilities
    }

    /// Full logits in selected-profile order, including Choice none last.
    pub fn full_logits(&self) -> Vec<f64> {
        let mut logits = self.candidate_logits.clone();
        if let Some(none) = self.none_logit {
            logits.push(none);
        }
        logits
    }
}
