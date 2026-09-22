//! Offline Phase 3.4 branch-state contract tests.
//!
//! These tests exercise identity, fingerprint, storage accounting, fork,
//! batch, select, and gather semantics on synthetic states built with the
//! real per-layer tensor shapes. Checkpoint-gated native parity runs in the
//! `qwen35_parity_probe` example.

use std::fs;
use std::mem::size_of;
use std::time::{SystemTime, UNIX_EPOCH};

use super::layer0::{LayerState, CONV_KERNEL, HEAD_DIM, KV_SIZE, QKV_SIZE, VALUE_HEADS};
use super::model::BackboneState;
use crate::qwen35::{
    pinned_state_identity, Qwen35Error, BACKBONE_ID, BACKBONE_REVISION, EXECUTION_ARITHMETIC_ID,
    PROFILE_ID, STATE_FIRST_RENDERER_ID, TOKENIZER_JSON_SHA256,
};
use openkind_runtime::branch::{
    BranchBatch, BranchableState, StateError, StateIdentity, StateLineage,
};

const LAYER_COUNT: usize = 32;
const FLOAT_BYTES: usize = size_of::<f32>();
/// Phase 3B `CONTINUATION_TRACE.json` root_cache_bytes for position 98.
const PHASE3B_ROOT_CACHE_BYTES: usize = 59_899_904;

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

#[test]
fn pinned_identity_matches_selected_profile_constants() {
    let identity = pinned_state_identity();
    assert_eq!(identity.profile().as_str(), PROFILE_ID);
    assert_eq!(identity.backbone_id(), BACKBONE_ID);
    assert_eq!(identity.backbone_revision(), BACKBONE_REVISION);
    assert_eq!(identity.renderer_id(), STATE_FIRST_RENDERER_ID);
    assert_eq!(identity.tokenizer_digest(), TOKENIZER_JSON_SHA256);
    assert_eq!(identity.arithmetic_id(), EXECUTION_ARITHMETIC_ID);
}

#[test]
fn pinned_state_snapshot_round_trips_with_exact_content_identity() {
    let state = pinned_root_state(1);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "openkind-qwen-state-{}-{nonce}.bin",
        std::process::id()
    ));

    state.persist_pinned(&path).expect("persist state");
    let restored = BackboneState::restore_pinned(&path).expect("restore state");
    assert_eq!(restored.identity(), state.identity());
    assert_eq!(restored.position(), state.position());
    assert_ne!(restored.lineage().root_id(), state.lineage().root_id());
    assert_eq!(restored.strict_fingerprint(), state.strict_fingerprint());
    assert_eq!(
        restored.tensor_storage_bytes(),
        state.tensor_storage_bytes()
    );

    fs::write(&path, b"truncated").expect("replace with malformed snapshot");
    assert!(matches!(
        BackboneState::restore_pinned(&path),
        Err(Qwen35Error::StatePersistence(_))
    ));
    fs::remove_file(&path).expect("remove state snapshot");
}

#[test]
fn state_identity_rejects_empty_fields() {
    assert!(matches!(
        StateIdentity::new("", "model", "rev", "state_first", "digest", "cpu"),
        Err(StateError::EmptyIdentityField {
            field: "profile_id"
        })
    ));
    assert!(matches!(
        StateIdentity::new("profile", "model", "rev", "state_first", "", "cpu"),
        Err(StateError::EmptyIdentityField {
            field: "tokenizer_digest"
        })
    ));
}

#[test]
fn real_shape_root_state_matches_phase3b_byte_contract() {
    let state = root_state(98);
    let breakdown = state.tensor_storage_breakdown();

    let expected_convolution = 24 * QKV_SIZE * CONV_KERNEL * FLOAT_BYTES;
    let expected_recurrent = 24 * VALUE_HEADS * HEAD_DIM * HEAD_DIM * FLOAT_BYTES;
    let expected_kv = 8 * 2 * KV_SIZE * 98 * FLOAT_BYTES;
    assert_eq!(expected_convolution, 3_145_728);
    assert_eq!(expected_recurrent, 50_331_648);
    assert_eq!(expected_kv, 6_422_528);

    assert_eq!(breakdown.convolution_bytes, expected_convolution);
    assert_eq!(breakdown.recurrent_bytes, expected_recurrent);
    assert_eq!(breakdown.attention_kv_bytes, expected_kv);
    assert_eq!(breakdown.tensor_storage_bytes(), PHASE3B_ROOT_CACHE_BYTES);
    assert_eq!(state.tensor_storage_bytes(), PHASE3B_ROOT_CACHE_BYTES);
    assert_eq!(state.byte_len(), state.tensor_storage_bytes());
}

#[test]
fn structural_fingerprint_is_stable_across_clones_and_forks() {
    let state = root_state(4);
    let fingerprint = state.scheduling_fingerprint();

    assert_eq!(state.clone().scheduling_fingerprint(), fingerprint);
    let child = state.fork_one().expect("fork");
    assert_eq!(
        child.scheduling_fingerprint(),
        fingerprint,
        "fork preserves structure"
    );
    assert_eq!(child.strict_fingerprint(), state.strict_fingerprint());

    let mut advanced = state.fork_one().expect("fork");
    advanced.position += 1;
    assert_ne!(advanced.scheduling_fingerprint(), fingerprint);

    let mut reshaped = state.fork_one().expect("fork");
    reshaped.layers[0] = match &reshaped.layers[0] {
        LayerState::Linear { recurrent, .. } => LayerState::Linear {
            conv: vec![0.0; 1],
            recurrent: recurrent.clone(),
        },
        other => panic!("layer zero is linear, found {other:?}"),
    };
    assert_ne!(reshaped.scheduling_fingerprint(), fingerprint);
}

#[test]
fn structural_fingerprint_separates_roots_but_strict_is_content_only() {
    let first = root_state(4);
    let second = root_state(4);
    assert_ne!(
        first.lineage().root_id(),
        second.lineage().root_id(),
        "each synthetic root gets a fresh lineage"
    );
    assert_ne!(
        first.scheduling_fingerprint(),
        second.scheduling_fingerprint()
    );
    assert_eq!(first.strict_fingerprint(), second.strict_fingerprint());
}

#[test]
fn strict_fingerprint_tracks_tensor_contents_and_position() {
    let mut state = root_state(2);
    let baseline = state.strict_fingerprint();

    if let LayerState::Linear { conv, .. } = &mut state.layers[0] {
        conv[0] = 1.0;
    }
    assert_ne!(state.strict_fingerprint(), baseline);

    state.position += 1;
    let after_position = state.strict_fingerprint();
    assert_ne!(after_position, baseline);

    if let LayerState::Linear { conv, .. } = &mut state.layers[0] {
        conv[0] = 0.0;
    }
    // Position still differs from the baseline, so compare against the
    // state captured after the position change.
    let mut restored = state.clone();
    restored.position -= 1;
    assert_eq!(restored.strict_fingerprint(), baseline);
}

#[test]
fn fork_one_produces_independent_child_over_an_untouched_root() {
    let root = root_state(4);
    let root_fingerprint = root.scheduling_fingerprint();
    let root_strict = root.strict_fingerprint();
    let child = root.fork_one().expect("fork");

    assert_eq!(child.identity(), root.identity());
    assert_eq!(child.position(), root.position());
    assert_eq!(child.layers, root.layers);
    assert_eq!(child.lineage().root_id(), root.lineage().root_id());
    assert_eq!(
        child.lineage().fork_depth(),
        root.lineage().fork_depth() + 1
    );
    assert_eq!(child.profile_id().as_str(), "test-profile");
    assert_eq!(child.tensor_storage_bytes(), root.tensor_storage_bytes());

    let mut child = child;
    if let LayerState::Linear { conv, .. } = &mut child.layers[0] {
        conv[0] = 7.5;
    }
    child.position += 3;
    assert_eq!(
        root.scheduling_fingerprint(),
        root_fingerprint,
        "root is unchanged"
    );
    assert_eq!(
        root.strict_fingerprint(),
        root_strict,
        "root tensor contents are unchanged"
    );
    assert_ne!(child, root);
    if let LayerState::Linear { conv, .. } = &root.layers[0] {
        assert_eq!(conv[0], 0.0);
    }
}

#[test]
fn fork_batch_creates_independent_lanes_over_an_untouched_root() {
    let root = root_state(4);
    let root_fingerprint = root.scheduling_fingerprint();
    let lane_bytes = root.tensor_storage_bytes();

    let batch = root.fork_batch(3).expect("fan-out");
    assert_eq!(batch.lanes(), 3);
    assert_eq!(batch.tensor_storage_bytes(), 3 * lane_bytes);

    for lane_index in 0..3 {
        let lane = batch.select(lane_index).expect("lane");
        assert_eq!(lane.scheduling_fingerprint(), root_fingerprint);
        assert_eq!(lane.lineage().fork_depth(), 1);
    }

    let mut first = batch.select(0).expect("lane");
    if let LayerState::Full { keys, .. } = &mut first.layers[3] {
        keys[0] = -1.0;
    }
    first.position += 2;
    let sibling = batch.select(1).expect("lane");
    if let LayerState::Full { keys, .. } = &sibling.layers[3] {
        assert_eq!(keys[0], 0.0, "lanes are isolated from each other");
    }
    assert_eq!(
        root.scheduling_fingerprint(),
        root_fingerprint,
        "root is unchanged"
    );

    assert!(matches!(
        root.fork_batch(0),
        Err(StateError::InvalidLaneCount { requested: 0 })
    ));
}

#[test]
fn select_copies_lanes_and_rejects_out_of_range_indices() {
    let batch = root_state(4).fork_batch(3).expect("fan-out");
    let lane_one = batch.select(1).expect("lane");
    assert_eq!(
        lane_one.scheduling_fingerprint(),
        batch.select(1).expect("lane").scheduling_fingerprint()
    );
    assert!(matches!(
        batch.select(3),
        Err(StateError::LaneIndexOutOfBounds { lanes: 3, index: 3 })
    ));
}

#[test]
fn gather_reorders_and_duplicates_lanes_with_preserved_identity() {
    let root = root_state(4);
    let batch = root.fork_batch(2).expect("fan-out");
    let source_fingerprints: Vec<_> = (0..2)
        .map(|index| batch.select(index).expect("lane").scheduling_fingerprint())
        .collect();

    let gathered = batch.gather(&[1, 0, 1]).expect("gather");
    assert_eq!(gathered.lanes(), 3);
    assert_eq!(
        gathered.tensor_storage_bytes(),
        3 * root.tensor_storage_bytes()
    );
    let gathered_fingerprints: Vec<_> = (0..3)
        .map(|index| {
            gathered
                .select(index)
                .expect("lane")
                .scheduling_fingerprint()
        })
        .collect();
    assert_eq!(
        gathered_fingerprints,
        vec![
            source_fingerprints[1],
            source_fingerprints[0],
            source_fingerprints[1]
        ]
    );
    assert_eq!(
        gathered.select(0).expect("lane"),
        batch.select(1).expect("lane")
    );

    assert!(matches!(batch.gather(&[]), Err(StateError::EmptyGather)));
    assert!(matches!(
        batch.gather(&[0, 2]),
        Err(StateError::LaneIndexOutOfBounds { lanes: 2, index: 2 })
    ));
}

#[test]
fn generic_contract_callers_can_fork_and_gather_without_concrete_types() {
    fn lanes_from<B: BranchableState>(state: &B, lanes: usize) -> Result<Vec<B>, StateError> {
        let batch = state.fork_batch(lanes)?;
        let selected = batch.select(0)?;
        let gathered = batch.gather(&[0, 0])?;
        Ok(vec![state.fork_one()?, selected, gathered.select(1)?])
    }

    let root = root_state(4);
    let lanes = lanes_from(&root, 2).expect("generic fork and gather");
    assert_eq!(lanes.len(), 3);
    for lane in &lanes {
        assert_eq!(lane.profile_id().as_str(), "test-profile");
        assert_eq!(lane.position(), 4);
        assert_eq!(lane.tensor_storage_bytes(), root.tensor_storage_bytes());
        assert_eq!(lane.scheduling_fingerprint(), root.scheduling_fingerprint());
    }
    assert_eq!(root.lineage().fork_depth(), 0);
    assert!(lanes.iter().all(|lane| lane.lineage().fork_depth() >= 1));
}
