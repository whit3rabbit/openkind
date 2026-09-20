//! Typed branch-state fingerprints.

use std::fmt;

use sha2::{Digest, Sha256};

macro_rules! fingerprint_type {
    ($name:ident, $builder:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name([u8; 32]);

        impl $name {
            /// Start an incremental, domain-separated digest.
            #[must_use]
            pub fn builder(domain: &'static str) -> $builder {
                $builder::new(domain)
            }

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

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.hex())
            }
        }

        #[doc = concat!("Incremental constructor for [`", stringify!($name), "`].")]
        pub struct $builder {
            hasher: Sha256,
        }

        impl $builder {
            fn new(domain: &'static str) -> Self {
                let mut builder = Self {
                    hasher: Sha256::new(),
                };
                builder.field(domain.as_bytes());
                builder
            }

            /// Mix a length-prefixed byte field into the digest.
            pub fn field(&mut self, bytes: &[u8]) -> &mut Self {
                self.hasher.update((bytes.len() as u64).to_le_bytes());
                self.hasher.update(bytes);
                self
            }

            /// Mix one little-endian integer into the digest.
            pub fn value(&mut self, integer: u64) -> &mut Self {
                self.hasher.update(integer.to_le_bytes());
                self
            }

            /// Finish the digest.
            #[must_use]
            pub fn finish(&self) -> $name {
                let mut fingerprint = [0_u8; 32];
                fingerprint.copy_from_slice(&self.hasher.clone().finalize());
                $name(fingerprint)
            }
        }
    };
}

fingerprint_type!(
    SchedulingFingerprint,
    SchedulingFingerprintBuilder,
    "Cheap process-local fingerprint for execution lineage, layout, and position. It is not a content key."
);
fingerprint_type!(
    ContentFingerprint,
    ContentFingerprintBuilder,
    "Strict cross-process fingerprint over execution identity, position, and exact tensor contents."
);
