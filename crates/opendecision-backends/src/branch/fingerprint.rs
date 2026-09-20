//! Stable branch-state fingerprints.

use std::fmt;

use sha2::{Digest, Sha256};

/// 256-bit fingerprint over branch-state material.
///
/// The structural fingerprint hashes identity, lineage, position, and tensor
/// layout only, so scheduling operations can call it without reading tensor
/// contents. It is process-local: root IDs come from a process counter, and
/// equal fingerprints mean "interchangeable for scheduling", not "provably
/// equal contents". The strict content fingerprint hashes exact tensor bytes
/// and is the cross-process integrity identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateFingerprint([u8; 32]);

impl StateFingerprint {
    /// Lowercase hexadecimal digest.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut hex = String::with_capacity(64);
        for byte in self.0 {
            hex.push_str(&format!("{byte:02x}"));
        }
        hex
    }
}

impl fmt::Display for StateFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.hex())
    }
}

/// Incremental constructor for [`StateFingerprint`] digests.
///
/// Material is domain-separated and length-prefixed so field boundaries are
/// unambiguous.
pub(crate) struct FingerprintMaterial {
    hasher: Sha256,
}

impl FingerprintMaterial {
    /// Start a digest under a fixed domain-separation tag.
    pub(crate) fn new(domain: &'static str) -> Self {
        let mut material = Self {
            hasher: Sha256::new(),
        };
        material.field(domain.as_bytes());
        material
    }

    /// Mix a length-prefixed byte field into the digest.
    pub(crate) fn field(&mut self, bytes: &[u8]) -> &mut Self {
        self.hasher.update((bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
        self
    }

    /// Mix one little-endian integer into the digest.
    pub(crate) fn value(&mut self, integer: u64) -> &mut Self {
        self.hasher.update(integer.to_le_bytes());
        self
    }

    /// Finish and return the fingerprint.
    #[must_use]
    pub(crate) fn finish(&self) -> StateFingerprint {
        let mut fingerprint = [0_u8; 32];
        fingerprint.copy_from_slice(&self.hasher.clone().finalize());
        StateFingerprint(fingerprint)
    }
}
