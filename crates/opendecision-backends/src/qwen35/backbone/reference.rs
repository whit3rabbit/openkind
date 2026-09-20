use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{tensor::Dtype, SafeTensors};
use serde::de::DeserializeOwned;
use serde::Deserialize;

use super::super::{sha256_hex, Qwen35Error};
use super::contract::{validate_identity, ArchitectureContract, BackboneContract, LayerKind};
use super::types::{FullSequenceRecord, StageComparison, TraceStage};

const GOLDEN_FILE: &str = "QWEN35_BACKBONE_GOLDEN.safetensors";
const VECTOR_WIDTH: usize = 2_560;
const TENSOR_COUNT: usize = 47;

/// Validated offline Phase 3B architecture and golden-vector bundle.
pub struct BackboneReference {
    layer_kinds: Vec<LayerKind>,
    trace_input_ids: Vec<u32>,
    trace_stages: Vec<TraceStage>,
    full_sequence_records: Vec<FullSequenceRecord>,
    vectors: BTreeMap<String, Vec<f32>>,
}

impl BackboneReference {
    /// Load and validate the Phase 3B contract, architecture, trace, index, and tensors.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, Qwen35Error> {
        let root = root.as_ref();
        let contract: BackboneContract = read_json(root.join("BACKBONE_PARITY_CONTRACT.json"))?;
        contract.validate()?;
        let architecture: ArchitectureContract = read_json(root.join("QWEN35_ARCHITECTURE.json"))?;
        let layer_kinds = architecture.validate()?;
        let index: TensorIndex = read_json(root.join("BACKBONE_TENSOR_INDEX.json"))?;
        index.validate()?;
        let trace: LayerTrace = read_json(root.join("LAYER_TRACE.json"))?;
        trace.validate(&index.tensors)?;
        let continuation: ContinuationTrace = read_json(root.join("CONTINUATION_TRACE.json"))?;
        continuation.validate(&index.tensors)?;

        let tensor_bytes = read_bytes(root.join(GOLDEN_FILE))?;
        let actual_digest = sha256_hex(&tensor_bytes);
        if actual_digest != index.safetensors_sha256 {
            return Err(Qwen35Error::DigestMismatch {
                path: GOLDEN_FILE.to_owned(),
                expected: index.safetensors_sha256,
                actual: actual_digest,
            });
        }
        let vectors = load_vectors(&tensor_bytes, &index.tensors)?;
        Ok(Self {
            layer_kinds,
            trace_input_ids: trace.input_ids,
            trace_stages: trace.stages,
            full_sequence_records: index.full_sequence_records,
            vectors,
        })
    }

    /// Frozen sequence of 24 DeltaNet and 8 full-attention layers.
    #[must_use]
    pub fn layer_kinds(&self) -> &[LayerKind] {
        &self.layer_kinds
    }

    /// Exact full-sequence IDs used for the 34-stage diagnostic trace.
    #[must_use]
    pub fn trace_input_ids(&self) -> &[u32] {
        &self.trace_input_ids
    }

    /// Ordered embedding, 32-layer, and final-normalization trace stages.
    #[must_use]
    pub fn trace_stages(&self) -> &[TraceStage] {
        &self.trace_stages
    }

    /// Ten full-sequence candidate feature records.
    #[must_use]
    pub fn full_sequence_records(&self) -> &[FullSequenceRecord] {
        &self.full_sequence_records
    }

    /// Return one validated reference vector by tensor key.
    #[must_use]
    pub fn vector(&self, tensor_key: &str) -> Option<&[f32]> {
        self.vectors.get(tensor_key).map(Vec::as_slice)
    }

    /// Compare an actual FP32 stage vector to its frozen reference.
    pub fn compare(
        &self,
        tensor_key: &str,
        actual: &[f32],
    ) -> Result<StageComparison, Qwen35Error> {
        let expected = self.vectors.get(tensor_key).ok_or_else(|| {
            Qwen35Error::InvalidInput(format!("unknown Phase 3B tensor key `{tensor_key}`"))
        })?;
        if actual.len() != expected.len() {
            return Err(Qwen35Error::InvalidInput(format!(
                "tensor `{tensor_key}` expected width {}, found {}",
                expected.len(),
                actual.len()
            )));
        }
        if let Some((index, value)) = actual
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(Qwen35Error::Numerical(format!(
                "tensor `{tensor_key}` element {index} is not finite: {value}"
            )));
        }

        let mut max_abs = 0.0_f64;
        let mut squared_error = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut actual_norm = 0.0_f64;
        let mut expected_norm = 0.0_f64;
        for (&actual, &expected) in actual.iter().zip(expected) {
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            let delta = actual - expected;
            max_abs = max_abs.max(delta.abs());
            squared_error += delta * delta;
            dot += actual * expected;
            actual_norm += actual * actual;
            expected_norm += expected * expected;
        }
        let cosine = match (actual_norm == 0.0, expected_norm == 0.0) {
            (true, true) => 1.0,
            (true, false) | (false, true) => 0.0,
            (false, false) => dot / (actual_norm * expected_norm).sqrt(),
        };
        Ok(StageComparison {
            max_abs,
            rms: (squared_error / actual.len() as f64).sqrt(),
            cosine,
        })
    }
}

#[derive(Debug, Deserialize)]
struct TensorIndex {
    schema: String,
    profile_id: String,
    bundle_sha256: String,
    safetensors_sha256: String,
    tensors: BTreeMap<String, TensorRecord>,
    full_sequence_records: Vec<FullSequenceRecord>,
}

#[derive(Debug, Deserialize)]
struct TensorRecord {
    shape: Vec<usize>,
    dtype: String,
    sha256: String,
}

impl TensorIndex {
    fn validate(&self) -> Result<(), Qwen35Error> {
        super::super::require_equal(
            "backbone tensor index schema",
            "opendecision-qwen35-backbone-tensor-index/v1",
            &self.schema,
        )?;
        validate_identity(&self.profile_id, &self.bundle_sha256, None)?;
        if self.tensors.len() != TENSOR_COUNT || self.full_sequence_records.len() != 10 {
            return Err(Qwen35Error::InvalidInput(format!(
                "Phase 3B export requires {TENSOR_COUNT} tensors and 10 full-sequence records"
            )));
        }
        for (name, tensor) in &self.tensors {
            if tensor.shape != [VECTOR_WIDTH] || tensor.dtype != "float32" {
                return Err(Qwen35Error::InvalidTensor {
                    name: name.clone(),
                    message: format!(
                        "expected float32 [{VECTOR_WIDTH}], found {} {:?}",
                        tensor.dtype, tensor.shape
                    ),
                });
            }
        }
        for record in &self.full_sequence_records {
            if !self.tensors.contains_key(&record.tensor_key)
                || !record.max_abs_delta_vs_existing_bundle_feature.is_finite()
                || record.max_abs_delta_vs_existing_bundle_feature > 0.000_1
            {
                return Err(Qwen35Error::InvalidInput(format!(
                    "invalid full-sequence record for {} candidate {}",
                    record.question_id, record.candidate_index
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct LayerTrace {
    schema: String,
    profile_id: String,
    input_ids: Vec<u32>,
    token_count: usize,
    stages: Vec<TraceStage>,
}

impl LayerTrace {
    fn validate(&self, tensors: &BTreeMap<String, TensorRecord>) -> Result<(), Qwen35Error> {
        super::super::require_equal(
            "layer trace schema",
            "opendecision-qwen35-layer-trace/v1",
            &self.schema,
        )?;
        super::super::require_equal(
            "layer trace profile ID",
            super::super::PROFILE_ID,
            &self.profile_id,
        )?;
        if self.input_ids.len() != self.token_count || self.stages.len() != 34 {
            return Err(Qwen35Error::InvalidInput(
                "layer trace must contain its declared tokens and exactly 34 stages".to_owned(),
            ));
        }
        for (index, stage) in self.stages.iter().enumerate() {
            let expected_stage = match index {
                0 => "embedding".to_owned(),
                33 => "final_norm".to_owned(),
                layer => format!("layer_{:02}", layer - 1),
            };
            let expected_key = format!("diagnostic.{expected_stage}");
            super::super::require_equal("trace stage order", expected_stage, &stage.stage)?;
            super::super::require_equal("trace tensor key", expected_key, &stage.tensor_key)?;
            let indexed = tensors
                .get(&stage.tensor_key)
                .ok_or_else(|| Qwen35Error::MissingManifestEntry(stage.tensor_key.clone()))?;
            if stage.shape != [VECTOR_WIDTH]
                || stage.dtype != "float32"
                || stage.sha256 != indexed.sha256
            {
                return Err(Qwen35Error::InvalidTensor {
                    name: stage.tensor_key.clone(),
                    message: "layer trace metadata does not match tensor index".to_owned(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct ContinuationTrace {
    schema: String,
    profile_id: String,
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<u32>,
    root_position: usize,
    question_position: usize,
    candidate_position: usize,
    cached_candidate_max_abs_delta_vs_full_sequence: f64,
}

impl ContinuationTrace {
    fn validate(&self, tensors: &BTreeMap<String, TensorRecord>) -> Result<(), Qwen35Error> {
        super::super::require_equal(
            "continuation trace schema",
            "opendecision-qwen35-continuation-trace/v1",
            &self.schema,
        )?;
        super::super::require_equal(
            "continuation trace profile ID",
            super::super::PROFILE_ID,
            &self.profile_id,
        )?;
        if self.root_position != self.root_ids.len()
            || self.question_position != self.root_ids.len() + self.question_ids.len()
            || self.candidate_position
                != self.root_ids.len() + self.question_ids.len() + self.candidate_suffix_ids.len()
            || !self
                .cached_candidate_max_abs_delta_vs_full_sequence
                .is_finite()
            || self.cached_candidate_max_abs_delta_vs_full_sequence > 0.000_1
        {
            return Err(Qwen35Error::InvalidInput(
                "continuation trace positions or exported self-consistency guard failed".to_owned(),
            ));
        }
        for key in [
            "continuation.root_last_hidden",
            "continuation.question_last_hidden",
            "continuation.candidate_last_hidden",
        ] {
            if !tensors.contains_key(key) {
                return Err(Qwen35Error::MissingManifestEntry(key.to_owned()));
            }
        }
        Ok(())
    }
}

// `slice::as_chunks` is newer than the workspace's Rust 1.75 minimum.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn load_vectors(
    bytes: &[u8],
    index: &BTreeMap<String, TensorRecord>,
) -> Result<BTreeMap<String, Vec<f32>>, Qwen35Error> {
    let tensors = SafeTensors::deserialize(bytes)?;
    let expected: BTreeSet<_> = index.keys().cloned().collect();
    let actual: BTreeSet<_> = tensors.names().into_iter().cloned().collect();
    if expected != actual {
        return Err(Qwen35Error::TensorSetMismatch {
            expected: expected.into_iter().collect(),
            actual: actual.into_iter().collect(),
        });
    }

    let mut vectors = BTreeMap::new();
    for (name, record) in index {
        let tensor = tensors.tensor(name)?;
        if tensor.dtype() != Dtype::F32 || tensor.shape() != [VECTOR_WIDTH] {
            return Err(Qwen35Error::InvalidTensor {
                name: name.clone(),
                message: format!("expected F32 [{VECTOR_WIDTH}]"),
            });
        }
        if sha256_hex(tensor.data()) != record.sha256 {
            return Err(Qwen35Error::DigestMismatch {
                path: name.clone(),
                expected: record.sha256.clone(),
                actual: sha256_hex(tensor.data()),
            });
        }
        let mut values = Vec::with_capacity(VECTOR_WIDTH);
        for bytes in tensor.data().chunks_exact(4) {
            let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
            if !value.is_finite() {
                return Err(Qwen35Error::InvalidTensor {
                    name: name.clone(),
                    message: "reference vector contains a non-finite value".to_owned(),
                });
            }
            values.push(value);
        }
        vectors.insert(name.clone(), values);
    }
    Ok(vectors)
}

fn read_json<T: DeserializeOwned>(path: PathBuf) -> Result<T, Qwen35Error> {
    let bytes = read_bytes(path.clone())?;
    serde_json::from_slice(&bytes).map_err(|source| Qwen35Error::Json { path, source })
}

fn read_bytes(path: PathBuf) -> Result<Vec<u8>, Qwen35Error> {
    fs::read(&path).map_err(|source| Qwen35Error::Io { path, source })
}
