//! Types for breadth-first batched execution results.

use openkind_runtime::branch::BranchableState;
use openkind_runtime::BatchForwardMode;

/// Question fan-out (Phase 3.6): `Q` isolated lanes advanced breadth-first.
#[derive(Debug)]
pub struct BatchedQuestions<S: BranchableState> {
    pub(crate) question_features: Vec<Vec<f32>>,
    pub(crate) question_states: Vec<S>,
    pub(crate) batch_bytes: usize,
    pub(crate) batch_forward_mode: BatchForwardMode,
}

impl<S: BranchableState> BatchedQuestions<S> {
    /// Per-lane question features in plan order.
    #[must_use]
    pub fn question_features(&self) -> &[Vec<f32>] {
        &self.question_features
    }

    /// Advanced question states in plan order.
    #[must_use]
    pub fn question_states(&self) -> &[S] {
        &self.question_states
    }

    /// Continuation tensor bytes held by the fan-out lanes before advancing.
    #[must_use]
    pub const fn batch_bytes(&self) -> usize {
        self.batch_bytes
    }

    /// Physical forward mode used to advance the question lanes.
    #[must_use]
    pub const fn batch_forward_mode(&self) -> BatchForwardMode {
        self.batch_forward_mode
    }
}

/// Candidate fan-out (Phase 3.7): `K` isolated lanes advanced breadth-first.
#[derive(Debug)]
pub struct BatchedCandidates<S: BranchableState> {
    pub(crate) candidate_features: Vec<Vec<f32>>,
    pub(crate) candidate_states: Vec<S>,
    pub(crate) batch_bytes: usize,
    pub(crate) batch_forward_mode: BatchForwardMode,
}

impl<S: BranchableState> BatchedCandidates<S> {
    /// Per-lane candidate features in suffix order.
    #[must_use]
    pub fn candidate_features(&self) -> &[Vec<f32>] {
        &self.candidate_features
    }

    /// Advanced candidate states in suffix order.
    #[must_use]
    pub fn candidate_states(&self) -> &[S] {
        &self.candidate_states
    }

    /// Continuation tensor bytes held by the fan-out lanes before advancing.
    #[must_use]
    pub const fn batch_bytes(&self) -> usize {
        self.batch_bytes
    }

    /// Physical forward mode used to advance the candidate lanes.
    #[must_use]
    pub const fn batch_forward_mode(&self) -> BatchForwardMode {
        self.batch_forward_mode
    }
}

/// One question lane of a complete batched nested run.
#[derive(Debug)]
pub struct BatchedQuestionResult<S: BranchableState> {
    pub(crate) question_feature: Vec<f32>,
    pub(crate) question_state: S,
    pub(crate) candidate_batch_bytes: usize,
    pub(crate) candidates: Vec<BatchedCandidateResult<S>>,
}

impl<S: BranchableState> BatchedQuestionResult<S> {
    /// Final-token feature after the question suffix.
    #[must_use]
    pub fn question_feature(&self) -> &[f32] {
        &self.question_feature
    }

    /// Question continuation state; the candidate fan-out derives from it.
    #[must_use]
    pub fn question_state(&self) -> &S {
        &self.question_state
    }

    /// Continuation tensor bytes the candidate fan-out held before advancing.
    #[must_use]
    pub const fn candidate_batch_bytes(&self) -> usize {
        self.candidate_batch_bytes
    }

    /// Per-candidate results in suffix order.
    #[must_use]
    pub fn candidates(&self) -> &[BatchedCandidateResult<S>] {
        &self.candidates
    }
}

/// One candidate lane advanced from a batched question state.
#[derive(Debug)]
pub struct BatchedCandidateResult<S: BranchableState> {
    pub(crate) feature: Vec<f32>,
    pub(crate) state: S,
}

impl<S: BranchableState> BatchedCandidateResult<S> {
    /// Final-token candidate feature feeding the score-summary readout.
    #[must_use]
    pub fn feature(&self) -> &[f32] {
        &self.feature
    }

    /// Candidate continuation state.
    #[must_use]
    pub fn state(&self) -> &S {
        &self.state
    }
}

/// Complete batched nested run: one prefill, `Q` question lanes, and a `K`
/// candidate fan-out per question lane.
#[derive(Debug)]
pub struct BatchedNestedRun<S: BranchableState> {
    pub(crate) root_feature: Vec<f32>,
    pub(crate) root_state: S,
    pub(crate) question_batch_bytes: usize,
    pub(crate) batch_forward_mode: BatchForwardMode,
    pub(crate) questions: Vec<BatchedQuestionResult<S>>,
}

impl<S: BranchableState> BatchedNestedRun<S> {
    /// Final-token feature of the shared prefilled root.
    #[must_use]
    pub fn root_feature(&self) -> &[f32] {
        &self.root_feature
    }

    /// Immutable shared root state every fan-out derives from.
    #[must_use]
    pub fn root_state(&self) -> &S {
        &self.root_state
    }

    /// Continuation tensor bytes the question fan-out held before advancing.
    #[must_use]
    pub const fn question_batch_bytes(&self) -> usize {
        self.question_batch_bytes
    }

    /// Physical mode used across all question and candidate lane forwards.
    ///
    /// The complete run reports `Vectorized` only when every batched stage
    /// used a true multi-lane forward.
    #[must_use]
    pub const fn batch_forward_mode(&self) -> BatchForwardMode {
        self.batch_forward_mode
    }

    /// Per-question results in plan order.
    #[must_use]
    pub fn questions(&self) -> &[BatchedQuestionResult<S>] {
        &self.questions
    }
}
