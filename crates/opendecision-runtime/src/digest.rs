//! Typed digests over finalized execution-input token sequences.
//!
//! The finalized token sequences — the rendered state root, question
//! branches, and candidate suffixes — are the actual execution contract of a
//! request: two inputs with identical sequences execute identically, and two
//! with different sequences do not. These digests give that identity a
//! stable, domain-separated SHA-256 form for telemetry, reproducibility
//! artifacts, and invariance tests.
//!
//! Token IDs are hashed as fixed-width little-endian `u32`s in
//! length-prefixed fields; no incidental textual or JSON representation
//! enters the digest. Each role (state root, question, candidate) has its own
//! type and domain, so equal token streams under different roles never share
//! a digest.
//!
//! Two distinct whole-request identities are provided:
//!
//! * [`ExecutionInputDigest`] — order-sensitive; identifies exactly what ran,
//!   including question and candidate ordering. This is the reproducibility
//!   identity for evidence artifacts.
//! * [`SemanticSetDigest`] — order-independent; the canonical digest of the
//!   same set of role-labeled segments. This is for invariance and isolation
//!   testing where permutation must not change identity.

use sha2::{Digest, Sha256};

use crate::branch::fingerprint_type;

/// Domain of the immutable state-root token digest.
const STATE_ROOT_DOMAIN: &str = "opendecision-input-state-root-v1";
/// Domain of the question-branch token digest.
const QUESTION_DOMAIN: &str = "opendecision-input-question-v1";
/// Domain of the candidate-suffix token digest.
const CANDIDATE_DOMAIN: &str = "opendecision-input-candidate-v1";
/// Domain of the exact-order whole-input digest.
const EXECUTION_INPUT_DOMAIN: &str = "opendecision-execution-input-v1";
/// Domain of the order-independent whole-set digest.
const SEMANTIC_SET_DOMAIN: &str = "opendecision-semantic-set-v1";
/// Domain of one role-labeled record inside a [`SemanticSetDigest`].
const SEMANTIC_SET_RECORD_DOMAIN: &str = "opendecision-semantic-set-record-v1";

fingerprint_type!(
    StateTokenDigest,
    StateTokenDigestBuilder,
    "Digest over the finalized immutable state-root token sequence."
);
fingerprint_type!(
    QuestionTokenDigest,
    QuestionTokenDigestBuilder,
    "Digest over one finalized question-branch token sequence."
);
fingerprint_type!(
    CandidateTokenDigest,
    CandidateTokenDigestBuilder,
    "Digest over one finalized candidate-suffix token sequence."
);
fingerprint_type!(
    ExecutionInputDigest,
    ExecutionInputDigestBuilder,
    "Order-sensitive digest over the exact finalized execution input: state root, then each question and its candidate suffixes in execution order."
);
fingerprint_type!(
    SemanticSetDigest,
    SemanticSetDigestBuilder,
    "Order-independent digest over the canonical set of role-labeled input segments."
);

/// Fixed-width little-endian bytes of one token-ID sequence.
fn little_endian_tokens(tokens: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

impl StateTokenDigest {
    /// Digest the finalized state-root token sequence.
    #[must_use]
    pub fn from_token_ids(tokens: &[u32]) -> Self {
        Self::builder(STATE_ROOT_DOMAIN)
            .field(&little_endian_tokens(tokens))
            .finish()
    }
}

impl QuestionTokenDigest {
    /// Digest one finalized question-branch token sequence.
    #[must_use]
    pub fn from_token_ids(tokens: &[u32]) -> Self {
        Self::builder(QUESTION_DOMAIN)
            .field(&little_endian_tokens(tokens))
            .finish()
    }
}

impl CandidateTokenDigest {
    /// Digest one finalized candidate-suffix token sequence.
    #[must_use]
    pub fn from_token_ids(tokens: &[u32]) -> Self {
        Self::builder(CANDIDATE_DOMAIN)
            .field(&little_endian_tokens(tokens))
            .finish()
    }
}

impl ExecutionInputDigest {
    /// Digest the exact execution: the state root, then for each question in
    /// execution order its branch followed by each candidate suffix in
    /// execution order. Each segment is hashed under an explicit role label,
    /// so reordering any two segments changes the digest.
    #[must_use]
    pub fn from_execution_order(root: &[u32], questions: &[(&[u32], &[&[u32]])]) -> Self {
        let mut builder = Self::builder(EXECUTION_INPUT_DOMAIN);
        builder.field(b"state").field(&little_endian_tokens(root));
        for (question_ids, candidate_ids) in questions {
            builder
                .field(b"question")
                .field(&little_endian_tokens(question_ids));
            for candidate in *candidate_ids {
                builder
                    .field(b"candidate")
                    .field(&little_endian_tokens(candidate));
            }
        }
        builder.finish()
    }
}

impl SemanticSetDigest {
    /// Digest the same role-labeled segments independent of order: each
    /// record (role label plus token bytes) is digested separately, the
    /// record digests are sorted, and the sorted list is digested.
    #[must_use]
    pub fn from_segments(root: &[u32], questions: &[(&[u32], &[&[u32]])]) -> Self {
        let mut records = vec![record_digest(b"state", root)];
        for (question_ids, candidate_ids) in questions {
            records.push(record_digest(b"question", question_ids));
            for candidate in *candidate_ids {
                records.push(record_digest(b"candidate", candidate));
            }
        }
        records.sort_unstable();
        let mut builder = Self::builder(SEMANTIC_SET_DOMAIN);
        builder.value(records.len() as u64);
        for record in &records {
            builder.field(record);
        }
        builder.finish()
    }
}

fn record_digest(role: &[u8], tokens: &[u32]) -> [u8; 32] {
    let token_bytes = little_endian_tokens(tokens);
    let mut hasher = Sha256::new();
    hasher.update(SEMANTIC_SET_RECORD_DOMAIN.as_bytes());
    hasher.update((role.len() as u64).to_le_bytes());
    hasher.update(role);
    hasher.update((token_bytes.len() as u64).to_le_bytes());
    hasher.update(&token_bytes);
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&hasher.finalize());
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question_shape_a() -> Vec<(&'static [u32], &'static [&'static [u32]])> {
        vec![
            (&[10, 11, 12], &[&[20, 21], &[22, 23]]),
            (&[13, 14], &[&[24]]),
        ]
    }

    fn question_shape_b() -> Vec<(&'static [u32], &'static [&'static [u32]])> {
        vec![
            (&[13, 14], &[&[24]]),
            (&[10, 11, 12], &[&[22, 23], &[20, 21]]),
        ]
    }

    #[test]
    fn role_domains_separate_equal_token_streams() {
        let tokens = [7_u32, 8, 9];
        let state = StateTokenDigest::from_token_ids(&tokens);
        let question = QuestionTokenDigest::from_token_ids(&tokens);
        let candidate = CandidateTokenDigest::from_token_ids(&tokens);
        // The types themselves prevent cross-role comparison, so the domain
        // separation is asserted over the underlying digests.
        assert_ne!(state.hex(), question.hex());
        assert_ne!(state.hex(), candidate.hex());
        assert_ne!(question.hex(), candidate.hex());
        assert_eq!(state.hex().len(), 64);
        assert_eq!(
            state,
            StateTokenDigest::from_token_ids(&tokens),
            "digests are deterministic"
        );
        assert_ne!(
            state,
            StateTokenDigest::from_token_ids(&[7, 8, 10]),
            "token content changes the digest"
        );
    }

    #[test]
    fn execution_input_digest_is_order_sensitive() {
        let root = [1_u32, 2, 3];
        let a = ExecutionInputDigest::from_execution_order(&root, &question_shape_a());
        let b = ExecutionInputDigest::from_execution_order(&root, &question_shape_b());
        assert_eq!(
            a,
            ExecutionInputDigest::from_execution_order(&root, &question_shape_a()),
            "exact-order digest is deterministic"
        );
        assert_ne!(
            a, b,
            "reordering questions or candidates is a different execution"
        );
    }

    #[test]
    fn semantic_set_digest_is_order_independent() {
        let root = [1_u32, 2, 3];
        let a = SemanticSetDigest::from_segments(&root, &question_shape_a());
        let b = SemanticSetDigest::from_segments(&root, &question_shape_b());
        assert_eq!(
            a, b,
            "permuting questions and candidates preserves the semantic set"
        );
        assert_ne!(
            a,
            SemanticSetDigest::from_segments(&[1, 2, 4], &question_shape_a()),
            "different content changes the set digest"
        );
        assert_ne!(
            SemanticSetDigest::from_segments(&root, &question_shape_a()),
            SemanticSetDigest::from_segments(&root, &[(&[1, 2, 3], &[&[20, 21], &[22, 23]])]),
            "moving the root tokens into a question lane is a different set"
        );
    }
}
