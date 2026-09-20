use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::embedding::*;
use crate::qwen35::{BackboneReference, Qwen35Error};

const TOKEN_25_ROW_SHA256: &str =
    "84ce40703c960b305ad72adbdc0f6fc5b8df6fd279e5b18d5dd194e4764377c7";
const TOKEN_25_ROW_HEX: &str = include_str!(
    "../../../tests/fixtures/qwen35_backbone_phase3b_a047d6802c3f06f085b8/embedding_token_25.bf16.hex"
);

#[test]
fn seek_lookup_preserves_token_order_and_widens_bf16() {
    let root = temp_dir("lookup");
    let shard_path = root.join("tiny.safetensors");
    write_tiny_shard(&shard_path, &[[1.0, -2.0], [3.5, 4.0], [-0.5, 8.0]]);
    let layout = EmbeddingLayout::read(&shard_path, 3, 2).expect("read tiny layout");
    let embedding = Qwen35Embedding::from_layout(shard_path, layout);

    let output = embedding.embed(&[2, 0]).expect("embed tokens");
    assert_eq!(output.token_count(), 2);
    assert_eq!(output.hidden_size(), 2);
    assert_eq!(output.values(), &[-0.5, 8.0, 1.0, -2.0]);
    assert_eq!(output.last_token(), &[1.0, -2.0]);
    assert!(matches!(
        embedding.embed(&[3]),
        Err(Qwen35Error::InvalidInput(_))
    ));
    assert!(matches!(
        embedding.embed(&[]),
        Err(Qwen35Error::InvalidInput(_))
    ));
    drop(embedding);
    fs::remove_dir_all(root).expect("remove temp directory");
}

#[test]
fn tensor_layout_rejects_wrong_dtype_and_shape() {
    let root = temp_dir("invalid-layout");
    let rows = [[1.0, -2.0], [3.5, 4.0], [-0.5, 8.0]];
    let wrong_dtype = root.join("wrong-dtype.safetensors");
    write_tiny_shard_with_layout(&wrong_dtype, "F32", [3, 2], &rows);
    assert!(matches!(
        EmbeddingLayout::read(&wrong_dtype, 3, 2),
        Err(Qwen35Error::InvalidTensor { .. })
    ));

    let wrong_shape = root.join("wrong-shape.safetensors");
    write_tiny_shard_with_layout(&wrong_shape, "BF16", [2, 3], &rows);
    assert!(matches!(
        EmbeddingLayout::read(&wrong_shape, 3, 2),
        Err(Qwen35Error::InvalidTensor { .. })
    ));
    fs::remove_dir_all(root).expect("remove temp directory");
}

#[test]
fn pinned_checkpoint_row_matches_phase3b_embedding_exactly() {
    let bytes = decode_hex(TOKEN_25_ROW_HEX);
    assert_eq!(bytes.len(), HIDDEN_SIZE * 2);
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), TOKEN_25_ROW_SHA256);
    let mut actual = Vec::with_capacity(HIDDEN_SIZE);
    decode_bf16_row(&bytes, 25, &mut actual).expect("decode pinned BF16 row");

    let phase3b_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z");
    let reference = BackboneReference::load(phase3b_root).expect("load Phase 3B reference");
    let comparison = reference
        .compare("diagnostic.embedding", &actual)
        .expect("compare embedding");
    assert_eq!(comparison.max_abs(), 0.0);
    assert_eq!(comparison.rms(), 0.0);
    assert!((comparison.cosine() - 1.0).abs() <= f64::EPSILON);
}

#[test]
fn non_finite_bf16_is_rejected() {
    let mut output = Vec::new();
    assert!(matches!(
        decode_bf16_row(&0x7f80_u16.to_le_bytes(), 0, &mut output),
        Err(Qwen35Error::Numerical(_))
    ));
}

#[test]
fn decoder_shard_verification_rejects_wrong_size() {
    let root = temp_dir("decoder-shard-size");
    fs::write(root.join(DECODER_SHARD), b"not a model shard").expect("write tiny shard");
    assert!(matches!(
        verify_decoder_shard(&root),
        Err(Qwen35Error::InvalidTensor { .. })
    ));
    fs::remove_dir_all(root).expect("remove temp directory");
}

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "opendecision-qwen35-embedding-{}-{label}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create temp directory");
    root
}

fn write_tiny_shard(path: &Path, rows: &[[f32; 2]; 3]) {
    write_tiny_shard_with_layout(path, "BF16", [3, 2], rows);
}

fn write_tiny_shard_with_layout(path: &Path, dtype: &str, shape: [usize; 2], rows: &[[f32; 2]; 3]) {
    let mut header = serde_json::to_vec(&serde_json::json!({
        EMBEDDING_TENSOR: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, 12]
        }
    }))
    .expect("serialize header");
    let padding = (8 - header.len() % 8) % 8;
    header.resize(header.len() + padding, b' ');
    let mut file = File::create(path).expect("create tiny shard");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write header length");
    file.write_all(&header).expect("write header");
    for row in rows {
        for value in row {
            let bf16 = (value.to_bits() >> 16) as u16;
            file.write_all(&bf16.to_le_bytes()).expect("write BF16");
        }
    }
}

fn decode_hex(value: &str) -> Vec<u8> {
    let value = value.trim();
    assert_eq!(value.len() % 2, 0, "hex fixture must have byte pairs");
    let bytes = value.as_bytes();
    (0..bytes.len() / 2)
        .map(|index| {
            let offset = index * 2;
            let pair = std::str::from_utf8(&bytes[offset..offset + 2]).expect("ASCII hex");
            u8::from_str_radix(pair, 16).expect("valid hex")
        })
        .collect()
}
