//! Offline packed-weight model, independent question state, and selected-row readout.
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::ops::{Add, Mul};
use std::path::Path;
use std::sync::Arc;

use super::artifacts::VerifiedArtifacts;
use super::mixers::{materialize_state, tensor, Layer, Projection};
use super::renderer::JeevesRow;
use super::{HIDDEN, MAX_TOKENS, TEMPERATURE};
use crate::families::support::{temperature_softmax, FamilyControl, FamilyError};
use crate::qwen35::mlx::layers::{linear, op, rms_norm, MlxLayerState};
use crate::qwen35::mlx::{MlxError, MlxRuntime, MlxRuntimeConfig};

/// Final-normalized selected vectors and pointer scores for explicit offline evidence.
#[derive(Debug)]
pub struct JeevesScores {
    /// Option rows followed by the decide row, each 4096 floats.
    pub selected_hidden: Vec<Vec<f32>>,
    /// Uncalibrated pointer logits, before dividing by the pinned temperature.
    pub logits: Vec<f64>,
    /// Distribution conditional on supplied options.
    pub probabilities: Vec<f64>,
}

/// Native MLX model with packed affine-4 projections and FP32 arithmetic.
pub struct JeevesModel {
    runtime: Arc<MlxRuntime>,
    embedding: Projection,
    layers: Vec<Layer>,
    final_norm: Array,
    q: Array,
    q_bias: Array,
    k: Array,
    k_bias: Array,
}

impl JeevesModel {
    /// Load caller-verified local files. All array construction and evaluation
    /// uses the shared serialized runtime, including on worker threads.
    pub fn load(artifacts: &VerifiedArtifacts) -> Result<Self, FamilyError> {
        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        runtime
            .execute(|| -> Result<Self, MlxError> {
                let tensors = load_mapped(&artifacts.model, "language_model.model.")?;
                let embedding = Projection::load(
                    &tensors,
                    "language_model.model.embed_tokens",
                    248320,
                    HIDDEN,
                )?;
                let layers = (0..32)
                    .map(|index| Layer::load(&tensors, index))
                    .collect::<Result<Vec<_>, _>>()?;
                let final_norm = tensor(
                    &tensors,
                    "language_model.model.norm.weight",
                    &[HIDDEN],
                    Dtype::Bfloat16,
                )?;
                let head = load_mapped(&artifacts.head, "")?;
                if head.len() != 4 {
                    return Err(MlxError::InvalidState("unexpected pointer tensors".into()));
                }
                let model = Self {
                    runtime: Arc::clone(&runtime),
                    embedding,
                    layers,
                    final_norm,
                    q: tensor(&head, "q.weight", &[256, HIDDEN], Dtype::Float32)?,
                    k: tensor(&head, "k.weight", &[256, HIDDEN], Dtype::Float32)?,
                    q_bias: tensor(&head, "q.bias", &[256], Dtype::Float32)?,
                    k_bias: tensor(&head, "k.bias", &[256], Dtype::Float32)?,
                };
                model.embedding.materialize()?;
                for vector in [
                    &model.final_norm,
                    &model.q,
                    &model.k,
                    &model.q_bias,
                    &model.k_bias,
                ] {
                    vector.eval().map_err(op("pointer materialize"))?;
                }
                Ok(model)
            })?
            .map_err(FamilyError::from)
    }

    /// Score a single row with fresh KV, DeltaNet, and convolution states.
    /// Independent rows need no prefix sharing or mutable branch cloning.
    pub fn score(
        &self,
        row: &JeevesRow,
        control: Option<&FamilyControl>,
    ) -> Result<JeevesScores, FamilyError> {
        if row.ids.is_empty()
            || row.ids.len() > MAX_TOKENS
            || row.ids.iter().any(|id| *id >= 248320)
            || row.decide_position != row.ids.len() - 1
            || row.ids[row.decide_position] != 248062
            || row.option_positions.is_empty()
            || row.option_positions.len() > 255
            || row
                .option_positions
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || row
                .option_positions
                .iter()
                .any(|pos| *pos >= row.decide_position || row.ids[*pos] != 248050)
        {
            return Err(FamilyError::InvalidInput(
                "invalid Jeeves readout row".into(),
            ));
        }
        self.runtime
            .execute(|| -> Result<JeevesScores, FamilyError> {
                let positions: Vec<_> = row
                    .option_positions
                    .iter()
                    .copied()
                    .chain([row.decide_position])
                    .collect();
                let mut selected = vec![Vec::new(); positions.len()];
                let mut states: Vec<Option<MlxLayerState>> = (0..32).map(|_| None).collect();
                // Match the reference's small prefill chunks and bound each lazy graph.
                for (chunk_index, ids) in row.ids.chunks(16).enumerate() {
                    let start = chunk_index * 16;
                    let mut hidden = self.embedding.embed(ids)?;
                    for (layer, state) in self.layers.iter().zip(states.iter_mut()) {
                        if let Some(control) = control {
                            control.check()?;
                        }
                        let (next, next_state) = layer.forward(&hidden, start, state.as_ref())?;
                        next.eval().map_err(op("Jeeves layer eval"))?;
                        materialize_state(&next_state)?;
                        hidden = next;
                        *state = Some(next_state);
                    }
                    let final_hidden = rms_norm(&hidden, &self.final_norm)?;
                    for (output, position) in selected.iter_mut().zip(&positions) {
                        if *position >= start && *position < start + ids.len() {
                            *output = crate::qwen35::mlx::layers::row_of(
                                &final_hidden,
                                *position - start,
                            )?
                            .to_vec_cast::<f32>()
                            .map_err(op("selected hidden read"))?;
                        }
                    }
                }
                if selected.iter().any(|values| {
                    values.len() != HIDDEN as usize || values.iter().any(|v| !v.is_finite())
                }) {
                    return Err(FamilyError::Numerical(
                        "invalid selected hidden states".into(),
                    ));
                }
                let count = row.option_positions.len();
                let options =
                    Array::from_slice(&selected[..count].concat(), &[count as i32, HIDDEN]);
                let decide = Array::from_slice(&selected[count], &[1, HIDDEN]);
                let keys = linear(&options, &self.k)?.add(&self.k_bias);
                let query = linear(&decide, &self.q)?.add(&self.q_bias);
                let logits = keys
                    .matmul(&query.transpose().map_err(op("pointer query transpose"))?)
                    .map_err(op("pointer dot product"))?
                    .mul(Array::from_f32(1.0 / 16.0))
                    .reshape(&[count as i32])
                    .map_err(op("pointer logits reshape"))?
                    .to_vec_cast::<f32>()
                    .map_err(op("pointer logits read"))?
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                let probabilities = temperature_softmax(&logits, TEMPERATURE)?;
                Ok(JeevesScores {
                    selected_hidden: selected,
                    logits,
                    probabilities,
                })
            })?
    }

    /// Active allocator peak for explicitly requested offline evidence.
    pub fn peak_memory_bytes(&self) -> Result<usize, FamilyError> {
        Ok(self.runtime.peak_memory_bytes()?)
    }
}

fn load_mapped(path: &Path, prefix: &str) -> Result<HashMap<String, Array>, MlxError> {
    // SAFETY: the caller verifies immutable local source files before load.
    // The mapping stays alive until MLX has copied each selected tensor into
    // its owned array. Checkpoint bytes are never staged in another directory.
    let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
        .map_err(op("Jeeves mmap"))?;
    let mut tensors = HashMap::new();
    for (name, view) in mapped.tensors() {
        if !name.starts_with(prefix) {
            continue;
        }
        let dtype = match candle_core::DType::try_from(view.dtype()).map_err(op("mapped dtype"))? {
            candle_core::DType::U32 => Dtype::Uint32,
            candle_core::DType::BF16 => Dtype::Bfloat16,
            candle_core::DType::F32 => Dtype::Float32,
            other => {
                return Err(MlxError::InvalidState(format!(
                    "unsupported Jeeves dtype {other:?}"
                )))
            }
        };
        let shape = view
            .shape()
            .iter()
            .map(|dimension| i32::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(op("mapped shape"))?;
        // SAFETY: safetensors validates data length against dtype and shape;
        // shape conversion is checked above. mlx_array_new_data copies the
        // bytes synchronously, so the result does not borrow the mapping.
        let array = unsafe { Array::from_raw_data(view.data().as_ptr().cast(), &shape, dtype) };
        tensors.insert(name, array);
    }
    Ok(tensors)
}
