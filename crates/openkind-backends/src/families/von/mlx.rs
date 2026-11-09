//! MLX/Metal execution backend for the pinned `von` profile (feature
//! `mlx`, macOS arm64).
//!
//! A second execution backend behind the same frozen family contract: the
//! same digest-verified artifacts, renderer, calibrated temperature map, and
//! noul debias feed the shared ModernBERT MLX body (`families::
//! mlx_modernbert`) with the von scorer head. The checkpoint is the author's
//! torch pickle (`option_marker.pt`) — the same digest-verified file the CPU
//! oracle reads; candle's pickle interpreter widens every tensor to FP32
//! host values and the arrays are built from those exact values. The Candle
//! CPU path remains the correctness oracle; parity gates compare against
//! the committed golden fixtures.
//!
//! Every MLX operation — model load and per-pass forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::collections::BTreeMap;
use std::sync::Arc;

use mlx_rs::Array;

use crate::families::mlx_modernbert::{
    gelu, layer_norm_biased, map_transposed, ModernBertMlx, ModernBertMlxConfig,
};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::{MlxError, MlxRuntime};

use super::model::VerifiedArtifacts;
use super::VonProfile;

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// FP32 pickle tensors computed on the Metal GPU through the shared
/// ModernBERT MLX body.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-von";

/// MLX backend id reported by the engine serving this profile.
pub const MLX_BACKEND_ID: &str = "von/mlx-fp32";

/// `torch.nn.LayerNorm` default semantics, matching the CPU scorer.
const LN_EPS: f64 = 1e-5;

/// The pinned von model on the MLX ModernBERT body plus the calibrated MLP
/// scorer.
pub struct VonMlxModel {
    body: ModernBertMlx,
    /// Scorer head arrays: biased linears consume transposed `(input,
    /// output)` weights like the body's projections.
    input_norm_weight: Array,
    input_norm_bias: Array,
    dense_weight: Array,
    dense_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
    out_proj_weight: Array,
    out_proj_bias: Array,
    runtime: Arc<MlxRuntime>,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared ModernBERT MLX body).
unsafe impl Send for VonMlxModel {}
unsafe impl Sync for VonMlxModel {}

/// Read the verified torch pickle into a map of FP32 MLX arrays. candle's
/// interpreter decodes the opcodes into plain tensors without executing
/// anything; every tensor widens to FP32 exactly (the checkpoint stores
/// FP32). Must run inside [`MlxRuntime::execute`].
fn pickle_tensor_map(
    checkpoint: &std::path::Path,
    required: &[String],
) -> Result<BTreeMap<String, Array>, MlxError> {
    let tensors = candle_core::pickle::read_all(checkpoint).map_err(|error| {
        MlxError::InvalidState(format!(
            "failed to read the torch pickle {}: {error}",
            checkpoint.display()
        ))
    })?;
    let tensors: std::collections::HashMap<String, candle_core::Tensor> =
        tensors.into_iter().collect();
    let mut map = BTreeMap::new();
    for name in required {
        let tensor = tensors.get(name).ok_or_else(|| {
            MlxError::InvalidState(format!("pinned checkpoint is missing tensor `{name}`"))
        })?;
        let tensor = tensor
            .to_dtype(candle_core::DType::F32)
            .map_err(|error| MlxError::InvalidState(format!("tensor `{name}`: {error}")))?;
        let shape: Vec<i32> = tensor.dims().iter().map(|dim| *dim as i32).collect();
        let values = tensor
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| MlxError::InvalidState(format!("tensor `{name}`: {error}")))?;
        map.insert(name.clone(), Array::from_slice(&values, &shape));
    }
    Ok(map)
}

/// The body tensor names under the `encoder` namespace plus the scorer keys.
fn required_tensor_names(profile: &VonProfile) -> Vec<String> {
    let mut names = vec![
        "encoder.embeddings.tok_embeddings.weight".to_owned(),
        "encoder.embeddings.norm.weight".to_owned(),
        "encoder.final_norm.weight".to_owned(),
        "scorer.input_norm.weight".to_owned(),
        "scorer.input_norm.bias".to_owned(),
        "scorer.dense.weight".to_owned(),
        "scorer.dense.bias".to_owned(),
        "scorer.norm.weight".to_owned(),
        "scorer.norm.bias".to_owned(),
        "scorer.out_proj.weight".to_owned(),
        "scorer.out_proj.bias".to_owned(),
    ];
    for layer in 0..profile.encoder.num_hidden_layers {
        if layer > 0 {
            names.push(format!("encoder.layers.{layer}.attn_norm.weight"));
        }
        names.push(format!("encoder.layers.{layer}.attn.Wqkv.weight"));
        names.push(format!("encoder.layers.{layer}.attn.Wo.weight"));
        names.push(format!("encoder.layers.{layer}.mlp_norm.weight"));
        names.push(format!("encoder.layers.{layer}.mlp.Wi.weight"));
        names.push(format!("encoder.layers.{layer}.mlp.Wo.weight"));
    }
    names
}

impl VonMlxModel {
    /// Load the digest-verified pickle into FP32 MLX arrays and build the
    /// body plus scorer.
    pub fn load(
        profile: &'static VonProfile,
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, MlxError> {
        runtime.execute(|| {
            let names = required_tensor_names(profile);
            let map = pickle_tensor_map(&artifacts.checkpoint, &names)?;
            let hidden = profile.encoder.hidden_size;
            let scorer_width = hidden / 2;
            let body_config = ModernBertMlxConfig {
                vocab_size: profile.encoder.vocab_size,
                hidden_size: hidden,
                num_attention_heads: profile.encoder.num_attention_heads,
                num_hidden_layers: profile.encoder.num_hidden_layers,
                intermediate_size: profile.encoder.intermediate_size,
                local_attention: 128,
                global_attn_every_n_layers: 3,
                global_rope_theta: profile.encoder.global_rope_theta,
                local_rope_theta: profile.encoder.local_rope_theta,
                norm_eps: 1e-5,
                max_sequence_tokens: profile.max_sequence_tokens,
            };
            let host_vector = |name: &str, len: usize| -> Result<Array, MlxError> {
                let array = map.get(name).ok_or_else(|| {
                    MlxError::InvalidState(format!("pinned checkpoint is missing tensor `{name}`"))
                })?;
                let flat: Vec<i32> = [len as i32].to_vec();
                if array.shape() != flat.as_slice() {
                    return Err(MlxError::InvalidState(format!(
                        "tensor `{name}` has shape {:?} but the pinned contract requires [{len}]",
                        array.shape()
                    )));
                }
                Ok(array.clone())
            };
            let body = ModernBertMlx::from_tensor_map(&map, &body_config, "encoder")?;
            Ok(Self {
                input_norm_weight: host_vector("scorer.input_norm.weight", hidden)?,
                input_norm_bias: host_vector("scorer.input_norm.bias", hidden)?,
                dense_weight: map_transposed(&map, "scorer.dense.weight", scorer_width, hidden)?,
                dense_bias: host_vector("scorer.dense.bias", scorer_width)?,
                norm_weight: host_vector("scorer.norm.weight", scorer_width)?,
                norm_bias: host_vector("scorer.norm.bias", scorer_width)?,
                out_proj_weight: map_transposed(&map, "scorer.out_proj.weight", 1, scorer_width)?,
                out_proj_bias: host_vector("scorer.out_proj.bias", 1)?,
                body,
                runtime: Arc::clone(runtime),
            })
        })?
    }

    /// One logit per option marker in marker order, before calibration.
    pub fn option_logits(
        &self,
        token_ids: &[u32],
        markers: &[usize],
    ) -> Result<Vec<f64>, MlxError> {
        if markers.is_empty() {
            return Ok(Vec::new());
        }
        let values = self.runtime.execute(|| -> Result<Vec<f32>, MlxError> {
            let hidden = self.body.forward_hidden(token_ids)?;
            let seq = hidden.shape()[0];
            let indices: Vec<u32> = markers
                .iter()
                .map(|position| {
                    u32::try_from(*position).map_err(|_| {
                        MlxError::InvalidState(format!(
                            "marker position {position} exceeds the sequence"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if markers.iter().any(|position| *position >= seq as usize) {
                return Err(MlxError::InvalidState(format!(
                    "a marker position exceeds the {}-token sequence",
                    seq
                )));
            }
            let idx = Array::from_slice(&indices, &[indices.len() as i32]);
            let rows = hidden
                .take_axis(&idx, 0)
                .map_err(|error| MlxError::Operation {
                    operation: "marker gather",
                    message: error.to_string(),
                })?;
            let normed = layer_norm_biased(
                &rows,
                &self.input_norm_weight,
                &self.input_norm_bias,
                LN_EPS,
            )?;
            let projected =
                normed
                    .matmul(&self.dense_weight)
                    .map_err(|error| MlxError::Operation {
                        operation: "scorer dense",
                        message: error.to_string(),
                    })?;
            let projected = projected + self.dense_bias.clone();
            let activated = gelu(&projected)?;
            let normed = layer_norm_biased(&activated, &self.norm_weight, &self.norm_bias, LN_EPS)?;
            let scored =
                normed
                    .matmul(&self.out_proj_weight)
                    .map_err(|error| MlxError::Operation {
                        operation: "scorer out_proj",
                        message: error.to_string(),
                    })?;
            let scored = scored + self.out_proj_bias.clone();
            let logits =
                scored
                    .reshape(&[markers.len() as i32])
                    .map_err(|error| MlxError::Operation {
                        operation: "scorer flatten",
                        message: error.to_string(),
                    })?;
            logits.eval().map_err(|error| MlxError::Operation {
                operation: "scorer eval",
                message: error.to_string(),
            })?;
            logits
                .to_vec_cast::<f32>()
                .map_err(|error| MlxError::Operation {
                    operation: "scorer host read",
                    message: error.to_string(),
                })
        })??;
        let logits: Vec<f64> = values.into_iter().map(f64::from).collect();
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(MlxError::InvalidState(
                "option-marker readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }

    /// [`FamilyError`]-mapped readout for the engine's shared logit-source
    /// trait impl in `engine.rs`.
    pub(crate) fn option_logits_family(
        &self,
        token_ids: &[u32],
        markers: &[usize],
    ) -> Result<Vec<f64>, FamilyError> {
        self.option_logits(token_ids, markers)
            .map_err(FamilyError::from)
    }
}
