//! Fixed artifact identity for the sole supported 4-bit package.
use crate::families::support::{verify_digest, FamilyError};
use std::path::{Path, PathBuf};

/// Files required for runtime execution, pinned independently of the package manifest.
pub const FILES: &[(&str, u64, &str)] = &[
    (
        "model.safetensors",
        5038161163,
        "a7f3c503106f38d7bff3a55cb16b82bb5085fd8247893ade8a1de6ed1750e1f1",
    ),
    (
        "model.safetensors.index.json",
        81194,
        "215f9abadc1cea4d465b01c036dffcac5a84c2fb2386777863eff6928b91ed0b",
    ),
    (
        "config.json",
        2080,
        "6cc3a4bc9bd804159d7849476564026451ace6009b7fc51cdf14908aaacf333a",
    ),
    (
        "tokenizer.json",
        19989325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    (
        "tokenizer_config.json",
        1161,
        "95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679",
    ),
    (
        "chat_template.jinja",
        7756,
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
    (
        "export.json",
        2582,
        "608d8aa594bc9cc6cd37e691eb3b5ec0e661866be0ae38f21659f6d70531e8a1",
    ),
    (
        "head.safetensors",
        8391016,
        "ceac931470abafbc62723a297291f2c3d96627679794433585e658981798343e",
    ),
];

/// Verified paths. Checkpoint tensors are read in place, never staged in a temporary directory.
pub struct VerifiedArtifacts {
    /// Packed affine-4 backbone.
    pub(crate) model: PathBuf,
    /// FP32 pointer head.
    pub(crate) head: PathBuf,
    /// Exact fast tokenizer.
    pub(crate) tokenizer: PathBuf,
}
impl VerifiedArtifacts {
    /// Verified packed backbone location.
    pub fn model_path(&self) -> &Path {
        &self.model
    }

    /// Verified FP32 pointer-head location.
    pub fn head_path(&self) -> &Path {
        &self.head
    }

    /// Verified tokenizer location for explicit renderer replay.
    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer
    }

    /// Verify every execution dependency before constructing any device arrays.
    pub fn verify(root: &Path) -> Result<Self, FamilyError> {
        for (name, bytes, digest) in FILES {
            let path = root.join(name);
            let actual = std::fs::metadata(&path)
                .map_err(|source| FamilyError::Io {
                    path: path.clone(),
                    source,
                })?
                .len();
            if actual != *bytes {
                return Err(FamilyError::ContractMismatch {
                    field: "artifact.bytes",
                    expected: bytes.to_string(),
                    actual: actual.to_string(),
                });
            }
            verify_digest(&path, digest)?;
        }
        Ok(Self {
            model: root.join("model.safetensors"),
            head: root.join("head.safetensors"),
            tokenizer: root.join("tokenizer.json"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_or_truncated_weights_fail_before_device_loading() {
        let root = tempfile::tempdir().unwrap();
        assert!(matches!(
            VerifiedArtifacts::verify(root.path()),
            Err(FamilyError::Io { .. })
        ));
        std::fs::write(root.path().join("model.safetensors"), b"bad").unwrap();
        assert!(matches!(
            VerifiedArtifacts::verify(root.path()),
            Err(FamilyError::ContractMismatch {
                field: "artifact.bytes",
                ..
            })
        ));
    }
}
