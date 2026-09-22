//! Versioned, digest-checked persistence for pinned Qwen continuation state.

use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use openkind_runtime::branch::{BranchableState, StateIdentity, StateLineage};
use sha2::{Digest, Sha256};

use super::layer0::{LayerState, CONV_KERNEL, HEAD_DIM, KV_SIZE, QKV_SIZE, VALUE_HEADS};
use super::model::BackboneState;
use crate::qwen35::{pinned_state_identity, Qwen35Error, MAX_SEQUENCE_TOKENS};

const MAGIC: &[u8; 16] = b"ODQ35STATEv1\0\0\0\0";
const FORMAT_VERSION: u32 = 1;
const LAYER_COUNT: usize = 32;
const DIGEST_BYTES: usize = 32;
const MAX_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);

impl BackboneState {
    /// Atomically persist a pinned continuation state with an envelope digest.
    ///
    /// Only the selected CPU profile is accepted. Process-local lineage is not
    /// persisted; a restored state receives a fresh root lineage.
    pub fn persist_pinned(&self, path: impl AsRef<Path>) -> Result<(), Qwen35Error> {
        let path = path.as_ref();
        let pinned = pinned_state_identity();
        if self.identity != pinned {
            return Err(Qwen35Error::StatePersistence(format!(
                "state identity `{}` does not match pinned identity `{pinned}`",
                self.identity
            )));
        }
        validate_layout(self.position, &self.layers)?;

        let mut bytes = Vec::with_capacity(self.tensor_storage_bytes().saturating_add(4 * 1024));
        bytes.extend_from_slice(MAGIC);
        push_u32(&mut bytes, FORMAT_VERSION);
        push_identity(&mut bytes, &self.identity)?;
        push_u64(&mut bytes, usize_to_u64(self.position, "position")?);
        push_u32(&mut bytes, usize_to_u32(self.layers.len(), "layer count")?);
        for layer in &self.layers {
            match layer {
                LayerState::Linear { conv, recurrent } => {
                    bytes.push(0);
                    push_f32_slice(&mut bytes, conv)?;
                    push_f32_slice(&mut bytes, recurrent)?;
                }
                LayerState::Full { keys, values } => {
                    bytes.push(1);
                    push_f32_slice(&mut bytes, keys)?;
                    push_f32_slice(&mut bytes, values)?;
                }
            }
        }
        bytes.extend_from_slice(self.strict_fingerprint().hex().as_bytes());
        let digest = Sha256::digest(&bytes);
        bytes.extend_from_slice(&digest);
        if bytes.len() > MAX_SNAPSHOT_BYTES {
            return Err(Qwen35Error::StatePersistence(format!(
                "snapshot is {} bytes, above the {} byte format limit",
                bytes.len(),
                MAX_SNAPSHOT_BYTES
            )));
        }

        let temporary = temporary_path(path);
        if let Err(source) = fs::write(&temporary, &bytes) {
            let _ = fs::remove_file(&temporary);
            return Err(Qwen35Error::Io {
                path: temporary,
                source,
            });
        }
        if let Err(source) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(Qwen35Error::Io {
                path: path.to_path_buf(),
                source,
            });
        }
        Ok(())
    }

    /// Restore a pinned continuation state and verify format, identity, layout,
    /// envelope digest, and strict tensor-content fingerprint.
    pub fn restore_pinned(path: impl AsRef<Path>) -> Result<Self, Qwen35Error> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| Qwen35Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if bytes.len() > MAX_SNAPSHOT_BYTES {
            return Err(Qwen35Error::StatePersistence(format!(
                "snapshot is {} bytes, above the {} byte format limit",
                bytes.len(),
                MAX_SNAPSHOT_BYTES
            )));
        }
        if bytes.len() < MAGIC.len() + DIGEST_BYTES {
            return Err(Qwen35Error::StatePersistence(
                "snapshot is shorter than the format envelope".into(),
            ));
        }
        let (body, stored_digest) = bytes.split_at(bytes.len() - DIGEST_BYTES);
        let actual_digest = Sha256::digest(body);
        if &actual_digest[..] != stored_digest {
            return Err(Qwen35Error::StatePersistence(
                "snapshot envelope SHA-256 mismatch".into(),
            ));
        }

        let mut reader = SnapshotReader::new(body);
        if reader.take(MAGIC.len(), "magic")? != MAGIC {
            return Err(Qwen35Error::StatePersistence(
                "snapshot magic does not match the Qwen state format".into(),
            ));
        }
        let version = reader.u32("format version")?;
        if version != FORMAT_VERSION {
            return Err(Qwen35Error::StatePersistence(format!(
                "unsupported snapshot version {version}"
            )));
        }
        let identity = reader.identity()?;
        let pinned = pinned_state_identity();
        if identity != pinned {
            return Err(Qwen35Error::StatePersistence(format!(
                "snapshot identity `{identity}` does not match pinned identity `{pinned}`"
            )));
        }
        let position = reader.usize("position")?;
        if position > MAX_SEQUENCE_TOKENS {
            return Err(Qwen35Error::StatePersistence(format!(
                "snapshot position {position} exceeds {MAX_SEQUENCE_TOKENS}"
            )));
        }
        let layer_count = reader.usize_u32("layer count")?;
        if layer_count != LAYER_COUNT {
            return Err(Qwen35Error::StatePersistence(format!(
                "snapshot has {layer_count} layers, expected {LAYER_COUNT}"
            )));
        }
        let mut layers = Vec::with_capacity(layer_count);
        for layer_index in 0..layer_count {
            let tag = reader.byte("layer tag")?;
            let first = reader.f32_vec("first layer tensor")?;
            let second = reader.f32_vec("second layer tensor")?;
            let layer = match tag {
                0 => LayerState::Linear {
                    conv: first,
                    recurrent: second,
                },
                1 => LayerState::Full {
                    keys: first,
                    values: second,
                },
                other => {
                    return Err(Qwen35Error::StatePersistence(format!(
                        "layer {layer_index} has unknown tag {other}"
                    )))
                }
            };
            layers.push(layer);
        }
        let fingerprint = reader.take(64, "strict content fingerprint")?;
        let fingerprint = std::str::from_utf8(fingerprint).map_err(|_| {
            Qwen35Error::StatePersistence("content fingerprint is not ASCII hex".into())
        })?;
        if !reader.is_empty() {
            return Err(Qwen35Error::StatePersistence(
                "snapshot contains trailing body bytes".into(),
            ));
        }
        validate_layout(position, &layers)?;
        let state = Self {
            identity,
            lineage: StateLineage::new_root(),
            position,
            layers,
        };
        let restored = state.strict_fingerprint().hex();
        if restored != fingerprint {
            return Err(Qwen35Error::StatePersistence(format!(
                "restored content fingerprint {restored} does not match stored {fingerprint}"
            )));
        }
        Ok(state)
    }
}

fn validate_layout(position: usize, layers: &[LayerState]) -> Result<(), Qwen35Error> {
    if position > MAX_SEQUENCE_TOKENS {
        return Err(Qwen35Error::StatePersistence(format!(
            "state position {position} exceeds {MAX_SEQUENCE_TOKENS}"
        )));
    }
    if layers.len() != LAYER_COUNT {
        return Err(Qwen35Error::StatePersistence(format!(
            "state has {} layers, expected {LAYER_COUNT}",
            layers.len()
        )));
    }
    for (index, layer) in layers.iter().enumerate() {
        let full_expected = index % 4 == 3;
        match layer {
            LayerState::Linear { conv, recurrent } if !full_expected => {
                expect_len(index, "convolution", conv.len(), QKV_SIZE * CONV_KERNEL)?;
                expect_len(
                    index,
                    "recurrent",
                    recurrent.len(),
                    VALUE_HEADS * HEAD_DIM * HEAD_DIM,
                )?;
            }
            LayerState::Full { keys, values } if full_expected => {
                expect_len(index, "keys", keys.len(), KV_SIZE * position)?;
                expect_len(index, "values", values.len(), KV_SIZE * position)?;
            }
            LayerState::Linear { .. } => {
                return Err(Qwen35Error::StatePersistence(format!(
                    "layer {index} is linear but the pinned layout requires full attention"
                )))
            }
            LayerState::Full { .. } => {
                return Err(Qwen35Error::StatePersistence(format!(
                "layer {index} is full attention but the pinned layout requires linear attention"
            )))
            }
        }
    }
    Ok(())
}

fn expect_len(
    layer: usize,
    tensor: &str,
    actual: usize,
    expected: usize,
) -> Result<(), Qwen35Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Qwen35Error::StatePersistence(format!(
            "layer {layer} {tensor} has {actual} values, expected {expected}"
        )))
    }
}

fn push_identity(bytes: &mut Vec<u8>, identity: &StateIdentity) -> Result<(), Qwen35Error> {
    for value in [
        identity.profile().as_str(),
        identity.backbone_id(),
        identity.backbone_revision(),
        identity.renderer_id(),
        identity.tokenizer_digest(),
        identity.arithmetic_id(),
    ] {
        push_u32(bytes, usize_to_u32(value.len(), "identity field length")?);
        bytes.extend_from_slice(value.as_bytes());
    }
    Ok(())
}

fn push_f32_slice(bytes: &mut Vec<u8>, values: &[f32]) -> Result<(), Qwen35Error> {
    push_u64(bytes, usize_to_u64(values.len(), "tensor length")?);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32, Qwen35Error> {
    u32::try_from(value)
        .map_err(|_| Qwen35Error::StatePersistence(format!("{label} {value} does not fit in u32")))
}

fn usize_to_u64(value: usize, label: &str) -> Result<u64, Qwen35Error> {
    u64::try_from(value)
        .map_err(|_| Qwen35Error::StatePersistence(format!("{label} {value} does not fit in u64")))
}

fn temporary_path(path: &Path) -> PathBuf {
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("qwen-state");
    path.with_file_name(format!(".{name}.{}.{}.tmp", std::process::id(), id))
}

struct SnapshotReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SnapshotReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize, label: &str) -> Result<&'a [u8], Qwen35Error> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Qwen35Error::StatePersistence(format!("{label} length overflow")))?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            Qwen35Error::StatePersistence(format!("snapshot ended while reading {label}"))
        })?;
        self.offset = end;
        Ok(value)
    }

    fn byte(&mut self, label: &str) -> Result<u8, Qwen35Error> {
        Ok(self.take(1, label)?[0])
    }

    fn u32(&mut self, label: &str) -> Result<u32, Qwen35Error> {
        let bytes: [u8; 4] = self
            .take(4, label)?
            .try_into()
            .expect("slice length checked");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self, label: &str) -> Result<u64, Qwen35Error> {
        let bytes: [u8; 8] = self
            .take(8, label)?
            .try_into()
            .expect("slice length checked");
        Ok(u64::from_le_bytes(bytes))
    }

    fn usize(&mut self, label: &str) -> Result<usize, Qwen35Error> {
        usize::try_from(self.u64(label)?)
            .map_err(|_| Qwen35Error::StatePersistence(format!("{label} does not fit in usize")))
    }

    fn usize_u32(&mut self, label: &str) -> Result<usize, Qwen35Error> {
        usize::try_from(self.u32(label)?)
            .map_err(|_| Qwen35Error::StatePersistence(format!("{label} does not fit in usize")))
    }

    fn string(&mut self, label: &str) -> Result<String, Qwen35Error> {
        let length = self.usize_u32(label)?;
        let value = self.take(length, label)?;
        String::from_utf8(value.to_vec())
            .map_err(|_| Qwen35Error::StatePersistence(format!("{label} is not valid UTF-8")))
    }

    fn identity(&mut self) -> Result<StateIdentity, Qwen35Error> {
        StateIdentity::new(
            self.string("profile identity")?,
            self.string("backbone identity")?,
            self.string("backbone revision")?,
            self.string("renderer identity")?,
            self.string("tokenizer digest")?,
            self.string("arithmetic identity")?,
        )
        .map_err(Qwen35Error::from)
    }

    fn f32_vec(&mut self, label: &str) -> Result<Vec<f32>, Qwen35Error> {
        let length = self.usize(label)?;
        let byte_len = length.checked_mul(size_of::<f32>()).ok_or_else(|| {
            Qwen35Error::StatePersistence(format!("{label} byte length overflow"))
        })?;
        let bytes = self.take(byte_len, label)?;
        let mut values = Vec::with_capacity(length);
        for index in 0..length {
            let start = index * size_of::<f32>();
            values.push(f32::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ]));
        }
        Ok(values)
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
