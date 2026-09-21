//! Execution identity for one native evaluation.
//!
//! The finalized token sequences — rendered state root, question branches,
//! candidate suffixes — are the execution contract of a request. This struct
//! binds their typed digests to the profile, renderer, tokenizer, arithmetic,
//! and physical execution selections behind one evaluation, so a bug report
//! or evidence artifact can state exactly which computation it discusses.
//!
//! Digest values are emitted at `debug` level in the daemon (never in default
//! production logs, because low-entropy inputs can be guessed offline from
//! raw digests) and into offline evidence artifacts, where they are the
//! reproducibility identity.

use opendecision_runtime::{
    BatchForwardMode, CandidateTokenDigest, ExecutionInputDigest, QuestionTokenDigest,
    SemanticSetDigest, StateTokenDigest,
};
use serde::Serialize;

use super::{
    ExecutionStrategy, EXECUTION_ARITHMETIC_ID, PROFILE_ID, STATE_FIRST_RENDERER_ID,
    TOKENIZER_JSON_SHA256,
};

/// Everything needed to identify the exact computation behind one evaluation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExecutionIdentity {
    /// Selected immutable model/execution profile ID.
    pub profile_id: String,
    /// State-first renderer identity (includes structured-state ordering
    /// semantics).
    pub renderer_id: String,
    /// SHA-256 of the exported tokenizer artifact.
    pub tokenizer_digest: String,
    /// Arithmetic/device identity of the execution path.
    pub arithmetic_id: String,
    /// Selected execution plan (state topology) as a stable identifier.
    pub execution_plan: String,
    /// Physical compute mode the lanes actually advanced in (`per_lane` or
    /// `vectorized`). A plan name alone does not say this; an accelerated
    /// backend must not be credited with the CPU reference's graph or vice
    /// versa.
    pub batch_forward_mode: String,
    /// Whether the plan came from a diagnostic override rather than the
    /// measured policy.
    pub forced: bool,
    /// Digest of the finalized immutable state-root token sequence.
    pub state_token_digest: String,
    /// Digests of each question branch, in execution order.
    pub question_token_digests: Vec<String>,
    /// Digests of each candidate suffix, grouped per question in execution
    /// order.
    pub candidate_token_digests: Vec<Vec<String>>,
    /// Order-sensitive digest over the exact finalized execution ordering.
    /// This is the reproducibility identity.
    pub execution_input_digest: String,
    /// Canonical order-independent digest of the same role-labeled segment
    /// set. For invariance and isolation testing; two requests that differ
    /// only in question order share it.
    pub semantic_set_digest: String,
}

impl ExecutionIdentity {
    /// Compute the execution identity of one finalized native request.
    ///
    /// `questions` holds, per question in execution order, its token IDs and
    /// its candidate-suffix token IDs in execution order. The digests are
    /// domain-separated per role and hashed over fixed-width little-endian
    /// token IDs, so the identity never depends on any textual rendering.
    #[must_use]
    pub fn compute(
        root_ids: &[u32],
        questions: &[(&[u32], &[&[u32]])],
        strategy: ExecutionStrategy,
        batch_forward_mode: BatchForwardMode,
        forced: bool,
    ) -> Self {
        let state_token_digest = StateTokenDigest::from_token_ids(root_ids);
        let question_token_digests = questions
            .iter()
            .map(|(question_ids, _)| QuestionTokenDigest::from_token_ids(question_ids).hex())
            .collect();
        let candidate_token_digests = questions
            .iter()
            .map(|(_, candidate_ids)| {
                candidate_ids
                    .iter()
                    .map(|ids| CandidateTokenDigest::from_token_ids(ids).hex())
                    .collect()
            })
            .collect();
        let execution_input_digest =
            ExecutionInputDigest::from_execution_order(root_ids, questions).hex();
        let semantic_set_digest = SemanticSetDigest::from_segments(root_ids, questions).hex();
        Self {
            profile_id: PROFILE_ID.to_owned(),
            renderer_id: STATE_FIRST_RENDERER_ID.to_owned(),
            tokenizer_digest: TOKENIZER_JSON_SHA256.to_owned(),
            arithmetic_id: EXECUTION_ARITHMETIC_ID.to_owned(),
            execution_plan: strategy.as_str().to_owned(),
            batch_forward_mode: batch_forward_mode.as_str().to_owned(),
            forced,
            state_token_digest: state_token_digest.hex(),
            question_token_digests,
            candidate_token_digests,
            execution_input_digest,
            semantic_set_digest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opendecision_runtime::BatchForwardMode;

    #[test]
    fn compute_records_profile_plan_and_physical_mode() {
        let identity = ExecutionIdentity::compute(
            &[1, 2, 3],
            &[(&[10, 11], &[&[20, 21], &[22, 23]])],
            ExecutionStrategy::NestedBatched,
            BatchForwardMode::PerLane,
            true,
        );
        assert_eq!(identity.profile_id, PROFILE_ID);
        assert_eq!(identity.renderer_id, STATE_FIRST_RENDERER_ID);
        assert_eq!(identity.tokenizer_digest, TOKENIZER_JSON_SHA256);
        assert_eq!(identity.arithmetic_id, EXECUTION_ARITHMETIC_ID);
        assert_eq!(identity.execution_plan, "nested_batched");
        assert_eq!(identity.batch_forward_mode, "per_lane");
        assert!(identity.forced);
        assert_eq!(identity.question_token_digests.len(), 1);
        assert_eq!(identity.candidate_token_digests.len(), 1);
        assert_eq!(identity.candidate_token_digests[0].len(), 2);
        assert_ne!(
            identity.candidate_token_digests[0][0], identity.candidate_token_digests[0][1],
            "distinct suffixes produce distinct digests"
        );
        assert_eq!(identity.execution_input_digest.len(), 64);
        assert_eq!(identity.semantic_set_digest.len(), 64);
    }

    #[test]
    fn execution_input_digest_is_order_sensitive_semantic_set_is_not() {
        let forward: &[(&[u32], &[&[u32]])] = &[(&[10, 11], &[&[20, 21], &[22, 23]])];
        let reversed: &[(&[u32], &[&[u32]])] = &[(&[10, 11], &[&[22, 23], &[20, 21]])];
        let a = ExecutionIdentity::compute(
            &[1, 2, 3],
            forward,
            ExecutionStrategy::NestedSequential,
            BatchForwardMode::PerLane,
            false,
        );
        let b = ExecutionIdentity::compute(
            &[1, 2, 3],
            reversed,
            ExecutionStrategy::NestedSequential,
            BatchForwardMode::PerLane,
            false,
        );
        assert_ne!(
            a.execution_input_digest, b.execution_input_digest,
            "candidate order is part of the exact execution"
        );
        assert_eq!(
            a.semantic_set_digest, b.semantic_set_digest,
            "candidate order is not part of the semantic set"
        );
        assert_eq!(a.state_token_digest, b.state_token_digest);
    }
}
