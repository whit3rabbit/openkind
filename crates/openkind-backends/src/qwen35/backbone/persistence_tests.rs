//! Offline persistence-envelope tests for pinned Qwen continuation state.
//!
//! The zero-content round trip (persist → restore → identical fingerprint
//! with a fresh lineage) lives in `branch_tests.rs`; these tests pin the
//! fail-closed envelope gates: exact nonzero tensor content, identity
//! mismatches on both the persist and restore sides, layout violations, and
//! corrupted or resealed snapshots.

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::layer0::{LayerState, CONV_KERNEL, HEAD_DIM, KV_SIZE, QKV_SIZE, VALUE_HEADS};
use super::model::BackboneState;
use crate::qwen35::{pinned_state_identity, Qwen35Error, PROFILE_ID};
use openkind_runtime::branch::{StateIdentity, StateLineage};

const LAYER_COUNT: usize = 32;
const MAGIC_LEN: usize = 16;
/// Body offset of the u32 length prefix of the first identity field.
const PROFILE_LENGTH_OFFSET: usize = MAGIC_LEN + 4;
/// Body offset of the first identity field (the profile id).
const PROFILE_OFFSET: usize = PROFILE_LENGTH_OFFSET + 4;
const DIGEST_BYTES: usize = 32;

fn test_identity() -> StateIdentity {
    StateIdentity::new(
        "test-profile",
        "test-model",
        "test-revision",
        "state_first",
        "tokenizer-digest",
        "cpu-fp32",
    )
    .expect("test identity is valid")
}

fn layer_state(layer_index: usize, position: usize) -> LayerState {
    if layer_index % 4 == 3 {
        LayerState::Full {
            keys: vec![0.0; KV_SIZE * position],
            values: vec![0.0; KV_SIZE * position],
        }
    } else {
        LayerState::Linear {
            conv: vec![0.0; QKV_SIZE * CONV_KERNEL],
            recurrent: vec![0.0; VALUE_HEADS * HEAD_DIM * HEAD_DIM],
        }
    }
}

fn root_state(position: usize) -> BackboneState {
    let layers = (0..LAYER_COUNT)
        .map(|layer_index| layer_state(layer_index, position))
        .collect();
    BackboneState {
        identity: test_identity(),
        lineage: StateLineage::new_root(),
        position,
        layers,
    }
}

fn pinned_root_state(position: usize) -> BackboneState {
    let mut state = root_state(position);
    state.identity = pinned_state_identity();
    state
}

/// Fill every tensor with nonzero, index-derived content so a restore that
/// swaps, truncates, or reorders payloads cannot pass by accident.
fn with_nonzero_tensors(state: &mut BackboneState) {
    let mut value = 0.25f32;
    let mut bump = || {
        value += 0.125;
        value
    };
    for (layer_index, layer) in state.layers.iter_mut().enumerate() {
        match layer {
            LayerState::Linear { conv, recurrent } => {
                for (index, slot) in conv.iter_mut().enumerate() {
                    *slot = bump() + (index % 13) as f32;
                }
                for (index, slot) in recurrent.iter_mut().enumerate() {
                    *slot = -bump() - (index % 7) as f32;
                }
            }
            LayerState::Full { keys, values } => {
                for (index, slot) in keys.iter_mut().enumerate() {
                    *slot = (layer_index + index % 17) as f32 / 29.0;
                }
                for (index, slot) in values.iter_mut().enumerate() {
                    *slot = -((index % 11) as f32) / 31.0;
                }
            }
        }
    }
}

fn snapshot_path(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "openkind-qwen-persistence-{label}-{}-{nonce}.bin",
        std::process::id()
    ))
}

fn error_message(result: Result<(), Qwen35Error>) -> String {
    match result {
        Ok(()) => panic!("expected the persistence gate to reject the state"),
        Err(err) => err.to_string(),
    }
}

fn restore_error(path: &std::path::Path) -> String {
    match BackboneState::restore_pinned(path) {
        Ok(_) => panic!("expected the persistence gate to reject the snapshot"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn nonzero_state_round_trips_with_exact_tensor_content() {
    let mut state = pinned_root_state(3);
    with_nonzero_tensors(&mut state);
    let path = snapshot_path("roundtrip");

    state.persist_pinned(&path).expect("persist state");
    let restored = BackboneState::restore_pinned(&path).expect("restore state");
    assert_eq!(restored.position(), state.position());
    assert_eq!(restored.identity(), state.identity());
    assert_eq!(restored.strict_fingerprint(), state.strict_fingerprint());
    assert_eq!(
        restored.layers, state.layers,
        "tensor payloads must round trip exactly"
    );
    assert_ne!(restored.lineage().root_id(), state.lineage().root_id());
    fs::remove_file(&path).expect("remove state snapshot");
}

#[test]
fn persist_rejects_state_without_the_pinned_identity() {
    let path = snapshot_path("identity");
    let err = error_message(root_state(1).persist_pinned(&path));
    assert!(err.contains("does not match pinned identity"), "{err}");
    assert!(
        !path.exists(),
        "a rejected state must not leave a snapshot behind"
    );
}

#[test]
fn persist_rejects_state_with_wrong_tensor_layout() {
    let mut state = pinned_root_state(1);
    for layer in &mut state.layers {
        if let LayerState::Linear { conv, .. } = layer {
            conv.pop();
            break;
        }
    }
    let path = snapshot_path("layout");
    let err = error_message(state.persist_pinned(&path));
    assert!(err.contains("convolution"), "{err}");
    assert!(
        !path.exists(),
        "a rejected state must not leave a snapshot behind"
    );
}

#[test]
fn restore_rejects_flipped_payload_bytes_via_envelope_digest() {
    let path = snapshot_path("corrupt");
    pinned_root_state(1)
        .persist_pinned(&path)
        .expect("persist state");

    let mut bytes = fs::read(&path).expect("read snapshot");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xFF;
    fs::write(&path, &bytes).expect("write corrupted snapshot");

    let err = restore_error(&path);
    assert!(err.contains("envelope SHA-256 mismatch"), "{err}");
    fs::remove_file(&path).expect("remove state snapshot");
}

#[test]
fn restore_rejects_resealed_snapshot_with_foreign_identity() {
    let path = snapshot_path("reseal");
    pinned_root_state(1)
        .persist_pinned(&path)
        .expect("persist state");
    let mut bytes = fs::read(&path).expect("read snapshot");

    // Rebuild the envelope around a tampered profile field so the payload
    // digest stays valid and the identity gate is what fails.
    let body_len = bytes.len() - DIGEST_BYTES;
    let (body, stored_digest) = bytes.split_at_mut(body_len);
    let length = u32::from_le_bytes(
        body[PROFILE_LENGTH_OFFSET..PROFILE_OFFSET]
            .try_into()
            .expect("length prefix"),
    ) as usize;
    assert_eq!(
        &body[PROFILE_OFFSET..PROFILE_OFFSET + length],
        PROFILE_ID.as_bytes(),
        "expected the profile id as the first identity field"
    );
    body[PROFILE_OFFSET] = if body[PROFILE_OFFSET] == b'a' {
        b'b'
    } else {
        b'a'
    };
    let digest = Sha256::digest(body);
    stored_digest.copy_from_slice(&digest);
    fs::write(&path, &bytes).expect("write resealed snapshot");

    let err = restore_error(&path);
    assert!(err.contains("does not match pinned identity"), "{err}");
    fs::remove_file(&path).expect("remove state snapshot");
}

#[test]
fn restore_reports_io_error_for_missing_snapshot() {
    let path = snapshot_path("missing");
    assert!(matches!(
        BackboneState::restore_pinned(&path),
        Err(Qwen35Error::Io { .. })
    ));
}
