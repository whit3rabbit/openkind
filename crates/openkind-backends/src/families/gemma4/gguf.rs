//! Pinned Winnow-E4B GGUF (q8_0) checkpoint loading for the Gemma 4
//! backbone.
//!
//! The GGUF is verified in place, its metadata is checked against the
//! pinned architecture contract, and the backbone consumes it in two
//! memory-bounded ways:
//!
//! - Every compute weight (attention/FFN/PLE projections, the tied LM head)
//!   stays a quantized `QTensor` executed through candle's `QMatMul`
//!   kernels — the same binding the `decoder-logit-llm` family uses — so
//!   resident weights stay near the on-disk q8_0 size instead of a
//!   dequantized FP32 copy.
//! - The two giant embedding tables (the token embedding and the
//!   `(vocab, layers x 256)` per-layer embedding table) are never fully
//!   resident: a file-backed row reader dequantizes only the rows a prompt
//!   references, in the checkpoint's own q8_0 blocks, arithmetic identical
//!   to candle's block dequantization.
//!
//! The KV-shared trailing layers' dead `attn_k`/`attn_v` tensors and the
//! recomputed `rope_freqs` are skipped, matching what the reference runtimes
//! load. Nothing is copied or staged: the checkpoint is streamed where it
//! lives.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::gguf_file as gguf;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};

use crate::families::support::{verify_digest, FamilyError};

use super::backbone::Gemma4TextConfig;
use super::{pinned_metadata, CHECKPOINT_SHA256, GGUF_VARIANT, TOKENIZER_JSON_SHA256};

/// Digest-verified, contract-checked artifacts required by the loader.
pub struct VerifiedArtifacts {
    /// Path to the verified GGUF checkpoint.
    pub checkpoint: PathBuf,
    /// Path to the verified tokenizer.
    pub tokenizer: PathBuf,
}

impl VerifiedArtifacts {
    /// Verify the pinned artifacts in place.
    ///
    /// The GGUF checkpoint is a multi-gigabyte file: verification streams it
    /// where it lives and nothing is copied.
    pub fn verify(model_root: &Path) -> Result<Self, FamilyError> {
        let checkpoint = model_root.join(GGUF_VARIANT);
        let tokenizer = model_root.join("tokenizer.json");
        for path in [&checkpoint, &tokenizer] {
            if !path.is_file() {
                return Err(FamilyError::Io {
                    path: path.to_path_buf(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "required pinned artifact is missing",
                    ),
                });
            }
        }
        verify_digest(&tokenizer, TOKENIZER_JSON_SHA256)?;
        verify_digest(&checkpoint, CHECKPOINT_SHA256)?;
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }
}

/// Single-letter labels (`A`..`Z`) followed by double-letter labels (`AA`,
/// `AB`, ...) up to the frozen candidate cap. The same discovery order the
/// reference engine uses.
pub(crate) fn letter_labels(cap: usize) -> Vec<String> {
    let mut labels = Vec::with_capacity(cap);
    for a in b'A'..=b'Z' {
        labels.push((a as char).to_string());
    }
    for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            labels.push(format!("{}{}", a as char, b as char));
        }
    }
    labels.truncate(cap);
    labels
}

/// GGUF tensor name -> candle model name. Names not in the map are skipped.
fn map_tensor_name(name: &str) -> Option<String> {
    let mapped = if name == "token_embd.weight" {
        "embed_tokens.weight".to_owned()
    } else if name == "output_norm.weight" {
        "norm.weight".to_owned()
    } else if name == "per_layer_token_embd.weight" {
        "embed_tokens_per_layer.weight".to_owned()
    } else if name == "per_layer_model_proj.weight" {
        "per_layer_model_projection.weight".to_owned()
    } else if name == "per_layer_proj_norm.weight" {
        "per_layer_projection_norm.weight".to_owned()
    } else if let Some(rest) = name.strip_prefix("blk.") {
        let (layer, tensor) = rest.split_once('.')?;
        format!("layers.{layer}.{tensor}")
    } else {
        return None;
    };
    let mapped = match mapped {
        s if s.ends_with("attn_q.weight") => s.replace("attn_q.weight", "self_attn.q_proj.weight"),
        s if s.ends_with("attn_k.weight") => s.replace("attn_k.weight", "self_attn.k_proj.weight"),
        s if s.ends_with("attn_v.weight") => s.replace("attn_v.weight", "self_attn.v_proj.weight"),
        s if s.ends_with("attn_output.weight") => {
            s.replace("attn_output.weight", "self_attn.o_proj.weight")
        }
        s if s.ends_with("attn_q_norm.weight") => {
            s.replace("attn_q_norm.weight", "self_attn.q_norm.weight")
        }
        s if s.ends_with("attn_k_norm.weight") => {
            s.replace("attn_k_norm.weight", "self_attn.k_norm.weight")
        }
        s if s.ends_with("attn_norm.weight") => {
            s.replace("attn_norm.weight", "input_layernorm.weight")
        }
        s if s.ends_with("ffn_norm.weight") => {
            s.replace("ffn_norm.weight", "pre_feedforward_layernorm.weight")
        }
        s if s.ends_with("post_attention_norm.weight") => s.replace(
            "post_attention_norm.weight",
            "post_attention_layernorm.weight",
        ),
        s if s.ends_with("post_ffw_norm.weight") => {
            s.replace("post_ffw_norm.weight", "post_feedforward_layernorm.weight")
        }
        s if s.ends_with("post_norm.weight") => {
            s.replace("post_norm.weight", "post_per_layer_input_norm.weight")
        }
        s if s.ends_with("ffn_gate.weight") => s.replace("ffn_gate.weight", "mlp.gate_proj.weight"),
        s if s.ends_with("ffn_up.weight") => s.replace("ffn_up.weight", "mlp.up_proj.weight"),
        s if s.ends_with("ffn_down.weight") => s.replace("ffn_down.weight", "mlp.down_proj.weight"),
        s if s.ends_with("inp_gate.weight") => {
            s.replace("inp_gate.weight", "per_layer_input_gate.weight")
        }
        s if s.ends_with(".proj.weight") && !s.ends_with("mlp.gate_proj.weight") => {
            s.replace(".proj.weight", ".per_layer_projection.weight")
        }
        s => s,
    };
    Some(mapped)
}

fn layer_index_of(model_name: &str) -> Option<usize> {
    let rest = model_name.strip_prefix("layers.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse().ok()
}

/// File-backed q8_0 row lookup over one GGUF tensor: dequantizes only the
/// rows a prompt references, so the giant embedding tables never become
/// fully resident. The block arithmetic is candle's own q8_0 dequantization
/// (`d * q` per 32-element block with an f16 scale), applied to a bounded
/// row slice read from the verified checkpoint.
#[derive(Clone)]
pub struct GgufRowTable {
    path: PathBuf,
    /// Absolute file offset of row 0.
    start: u64,
    /// Q8_0 blocks per row (width / 32).
    row_blocks: usize,
    /// Elements per row.
    width: usize,
    pub(crate) device: Device,
}

const QK8_0: usize = 32;
/// GGUF q8_0 block size in bytes: a 2-byte f16 scale followed by 32 int8
/// values (matches candle's `GgmlDType::Q8_0.type_size()`).
const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

impl GgufRowTable {
    /// Read `ids.len()` rows as an F32 tensor of shape `(ids.len(), width)`
    /// on the loading device.
    pub fn rows(&self, ids: &[u32]) -> candle_core::Result<Tensor> {
        let row_bytes = self.row_blocks * Q8_0_BLOCK_BYTES;
        let mut block_bytes = vec![0u8; row_bytes];
        let mut values = Vec::with_capacity(ids.len() * self.width);
        let mut file = std::fs::File::open(&self.path).map_err(candle_core::Error::wrap)?;
        for id in ids {
            let row = u64::from(*id);
            file.seek(SeekFrom::Start(self.start + row * row_bytes as u64))
                .map_err(candle_core::Error::wrap)?;
            file.read_exact(&mut block_bytes)
                .map_err(candle_core::Error::wrap)?;
            values.reserve(self.width);
            for block in block_bytes.chunks_exact(Q8_0_BLOCK_BYTES) {
                let scale_bits = u16::from_le_bytes([block[0], block[1]]);
                let scale = half::f16::from_bits(scale_bits).to_f32();
                for byte in &block[2..] {
                    values.push(scale * f32::from(*byte as i8));
                }
            }
        }
        Tensor::from_vec(values, (ids.len(), self.width), &self.device)
    }
}

/// The pinned checkpoint in execution form: quantized compute weights, an
/// LM-head matmul tied to the token embedding, and file-backed row lookups
/// for the two embedding tables.
pub struct Gemma4Checkpoint {
    tensors: HashMap<String, Arc<QTensor>>,
    /// Input-token embedding rows (also the tied LM-head source).
    pub token_embd_rows: GgufRowTable,
    /// Per-layer embedding rows.
    pub ple_rows: GgufRowTable,
    /// Tied LM head over the quantized token embedding.
    pub lm_head: QMatMul,
}

impl Gemma4Checkpoint {
    /// Quantized weight by candle model name (e.g.
    /// `layers.3.self_attn.q_proj.weight`).
    pub fn q_tensor(&self, name: &str) -> candle_core::Result<Arc<QTensor>> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("missing quantized tensor {name}")))
    }

    /// Small F32 tensor (norm weights, layer scalars), dequantized at load.
    pub fn f32_tensor(&self, name: &str) -> candle_core::Result<Tensor> {
        let qtensor = self.q_tensor(name)?;
        qtensor.dequantize(&self.token_embd_rows.device)
    }

    /// Dequantized F32 bias-free linear. Used where the graph amplifies
    /// quantization noise: candle's `QMatMul` kernels dequantize blocks to
    /// f16, and Gemma 4's scale-1.0 attention sits deep in softmax
    /// saturation, so an f16-weighted projection completely reshuffles the
    /// attention argmax. Attention projections therefore run as F32
    /// linears (~2.9 GB for the pinned profile); the residual-stream paths
    /// (FFN, PLE, LM head) tolerate the quantized kernels.
    pub fn f32_linear(&self, name: &str) -> candle_core::Result<candle_nn::Linear> {
        let qtensor = self.q_tensor(name)?;
        let weight = qtensor.dequantize(&self.token_embd_rows.device)?;
        Ok(candle_nn::Linear::new(weight, None))
    }
}

/// Read the pinned GGUF, validate its pinned metadata, and stage the
/// backbone's execution form on `device`.
///
/// KV-shared layers skip their dead K/V tensors: the loader drops them so a
/// missing-or-present checkpoint row cannot silently change the executed
/// graph (the reference runtimes ignore them too). The per-layer embedding
/// table is not loaded as a resident tensor at all — only its row geometry
/// is staged for the file-backed reader.
pub(crate) fn load_checkpoint(
    artifacts: &VerifiedArtifacts,
    cfg: &Gemma4TextConfig,
    device: &Device,
) -> Result<Gemma4Checkpoint, FamilyError> {
    let file = std::fs::File::open(&artifacts.checkpoint).map_err(|source| FamilyError::Io {
        path: artifacts.checkpoint.clone(),
        source,
    })?;
    let mut reader = std::io::BufReader::new(file);
    let content = gguf::Content::read(&mut reader).map_err(FamilyError::Candle)?;
    for (key, expected) in pinned_metadata() {
        let value = content
            .metadata
            .get(key)
            .ok_or_else(|| FamilyError::ContractMismatch {
                field: "gguf.metadata",
                expected: format!("`{key}` present"),
                actual: "missing".to_owned(),
            })?;
        let actual = match value {
            gguf::Value::String(value) => value.clone(),
            gguf::Value::U32(value) => value.to_string(),
            gguf::Value::U64(value) => value.to_string(),
            gguf::Value::F32(value) => value.to_string(),
            gguf::Value::F64(value) => value.to_string(),
            gguf::Value::Bool(value) => value.to_string(),
            other => format!("{other:?}"),
        };
        if actual != expected {
            return Err(FamilyError::contract(key, expected, actual));
        }
    }
    let dead_kv_layers: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|layer| cfg.kv_donor(*layer).is_some())
        .collect();
    let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::new();
    let row_table =
        |gguf_name: &str, width: usize| -> Result<GgufRowTable, FamilyError> {
            let info = content.tensor_infos.get(gguf_name).ok_or_else(|| {
                FamilyError::ContractMismatch {
                    field: "gguf.tensor",
                    expected: format!("`{gguf_name}` present"),
                    actual: "missing".to_owned(),
                }
            })?;
            if info.ggml_dtype != candle_core::quantized::GgmlDType::Q8_0 {
                return Err(FamilyError::contract(
                    "gguf.tensor.dtype",
                    "Q8_0",
                    format!("{:?}", info.ggml_dtype),
                ));
            }
            Ok(GgufRowTable {
                path: artifacts.checkpoint.clone(),
                start: content.tensor_data_offset + info.offset,
                row_blocks: width / QK8_0,
                width,
                device: device.clone(),
            })
        };
    let token_embd_rows = row_table("token_embd.weight", cfg.hidden_size)?;
    let ple_rows = row_table(
        "per_layer_token_embd.weight",
        cfg.num_hidden_layers * cfg.per_layer_input_dim,
    )?;
    for name in content.tensor_infos.keys() {
        if name == "per_layer_token_embd.weight" {
            // File-backed rows only; never resident.
            continue;
        }
        let Some(mapped) = map_tensor_name(name) else {
            continue;
        };
        if let Some(layer) = layer_index_of(&mapped) {
            let is_kv = mapped.ends_with("self_attn.k_proj.weight")
                || mapped.ends_with("self_attn.v_proj.weight")
                || mapped.ends_with("self_attn.k_norm.weight");
            if is_kv && dead_kv_layers.contains(&layer) {
                continue;
            }
        }
        let qtensor = content
            .tensor(&mut reader, name, device)
            .map_err(FamilyError::Candle)?;
        tensors.insert(mapped, Arc::new(qtensor));
    }
    let lm_head =
        QMatMul::from_arc(tensors.get("embed_tokens.weight").cloned().ok_or_else(|| {
            FamilyError::ContractMismatch {
                field: "gguf.tensor",
                expected: "`token_embd.weight` present".to_owned(),
                actual: "missing".to_owned(),
            }
        })?)
        .map_err(FamilyError::Candle)?;
    Ok(Gemma4Checkpoint {
        tensors,
        token_embd_rows,
        ple_rows,
        lm_head,
    })
}

/// Letter token ids cross-checked between the pinned `tokenizer.json` and
/// the GGUF-embedded vocabulary: one id per discovered label.
pub(crate) fn gguf_letter_token_ids(
    artifacts: &VerifiedArtifacts,
    cap: usize,
) -> Result<Vec<u32>, FamilyError> {
    let file = std::fs::File::open(&artifacts.checkpoint).map_err(|source| FamilyError::Io {
        path: artifacts.checkpoint.clone(),
        source,
    })?;
    let mut reader = std::io::BufReader::new(file);
    let content = gguf::Content::read(&mut reader).map_err(FamilyError::Candle)?;
    let tokens = content
        .metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| FamilyError::ContractMismatch {
            field: "gguf.tokenizer.tokens",
            expected: "embedded token list present".to_owned(),
            actual: "missing".to_owned(),
        })?;
    let gguf::Value::Array(tokens) = tokens else {
        return Err(FamilyError::ContractMismatch {
            field: "gguf.tokenizer.tokens",
            expected: "array of token strings".to_owned(),
            actual: "non-array value".to_owned(),
        });
    };
    let mut ids = Vec::with_capacity(cap);
    for label in letter_labels(cap) {
        let id = tokens.iter().position(|token| match token {
            gguf::Value::String(text) => text == &label,
            _ => false,
        });
        match id {
            Some(id) => ids.push(
                u32::try_from(id)
                    .map_err(|_| FamilyError::Numerical("GGUF token id exceeds u32".to_owned()))?,
            ),
            None => {
                // Labels beyond the single-token vocabulary end the contract;
                // the discovered prefix is what the profile offers.
                break;
            }
        }
    }
    if ids.len() < 2 {
        return Err(FamilyError::ContractMismatch {
            field: "gguf.letter_tokens",
            expected: "at least two verified letter tokens".to_owned(),
            actual: format!("{} discovered", ids.len()),
        });
    }
    Ok(ids)
}

// Keep DType referenced for the row-reader contract even though rows are
// always materialized as F32.
const _: Option<DType> = Some(DType::F32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_labels_follow_the_reference_discovery_order() {
        let labels = letter_labels(6);
        assert_eq!(labels, ["A", "B", "C", "D", "E", "F"]);
        let labels = letter_labels(28);
        assert_eq!(labels[26], "AA");
        assert_eq!(labels[27], "AB");
        assert_eq!(letter_labels(64).len(), 64);
    }

    #[test]
    fn tensor_names_map_to_the_candle_layout() {
        assert_eq!(
            map_tensor_name("blk.3.attn_q.weight").as_deref(),
            Some("layers.3.self_attn.q_proj.weight")
        );
        assert_eq!(
            map_tensor_name("blk.5.attn_k.weight").as_deref(),
            Some("layers.5.self_attn.k_proj.weight")
        );
        assert_eq!(
            map_tensor_name("blk.7.attn_output.weight").as_deref(),
            Some("layers.7.self_attn.o_proj.weight")
        );
        assert_eq!(
            map_tensor_name("blk.9.attn_norm.weight").as_deref(),
            Some("layers.9.input_layernorm.weight")
        );
        assert_eq!(
            map_tensor_name("blk.11.ffn_norm.weight").as_deref(),
            Some("layers.11.pre_feedforward_layernorm.weight")
        );
        assert_eq!(
            map_tensor_name("blk.13.post_ffw_norm.weight").as_deref(),
            Some("layers.13.post_feedforward_layernorm.weight")
        );
        assert_eq!(
            map_tensor_name("blk.13.post_norm.weight").as_deref(),
            Some("layers.13.post_per_layer_input_norm.weight")
        );
        assert_eq!(
            map_tensor_name("blk.15.ffn_gate.weight").as_deref(),
            Some("layers.15.mlp.gate_proj.weight")
        );
        assert_eq!(
            map_tensor_name("blk.17.inp_gate.weight").as_deref(),
            Some("layers.17.per_layer_input_gate.weight")
        );
        assert_eq!(
            map_tensor_name("blk.19.proj.weight").as_deref(),
            Some("layers.19.per_layer_projection.weight")
        );
        assert_eq!(
            map_tensor_name("token_embd.weight").as_deref(),
            Some("embed_tokens.weight")
        );
        assert_eq!(
            map_tensor_name("per_layer_token_embd.weight").as_deref(),
            Some("embed_tokens_per_layer.weight")
        );
        assert_eq!(
            map_tensor_name("per_layer_model_proj.weight").as_deref(),
            Some("per_layer_model_projection.weight")
        );
        assert_eq!(
            map_tensor_name("per_layer_proj_norm.weight").as_deref(),
            Some("per_layer_projection_norm.weight")
        );
        assert_eq!(map_tensor_name("rope_freqs.weight"), None);
        assert_eq!(
            map_tensor_name("output_norm.weight").as_deref(),
            Some("norm.weight")
        );
    }

    #[test]
    fn layer_index_parses_model_names() {
        assert_eq!(
            layer_index_of("layers.23.self_attn.q_proj.weight"),
            Some(23)
        );
        assert_eq!(layer_index_of("embed_tokens.weight"), None);
    }
}
