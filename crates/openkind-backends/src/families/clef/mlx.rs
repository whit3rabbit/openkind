//! MLX/Metal 4-bit execution for the `clef-flash-mlx-4bit` profile
//! (feature `mlx`, macOS arm64).
//!
//! The `mlx-community/clef-flash-4bit` checkpoint stores the backbone as
//! affine-quantized MLX tensors (4-bit, group 64: a packed U32 `weight`
//! plus BF16 `scales`/`biases`) under the `language_model.model.*` key
//! namespace, with the untied `language_model.lm_head` quantized as well.
//! Projections run through `mlx_rs::quantized_matmul` over the on-disk
//! packing; the gated DeltaNet recurrence follows the reference per-token
//! array ops (the same equations as the CPU oracle) with FP32 recurrent
//! state per the checkpoint's `mamba_ssm_dtype`; the joint schema head
//! executes on the CPU oracle's kernel over the final-norm hidden states.
//!
//! Every MLX operation runs under the process-wide [`MlxRuntime`]
//! execution lock, per the workspace MLX discipline.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_rs::ops::indexing::TryIndexOp as _;
use mlx_rs::Array;

use crate::families::support::{verify_digest, FamilyControl, FamilyError};
use crate::qwen35::mlx::runtime::SharedMlxRuntime;
use crate::qwen35::Qwen35Geometry;

use super::head::{JointSchemaHead, LexicalLookup};
use super::model::ClefExecutionModel;
use super::renderer::EncodedRecord;

/// Affine quantization parameters of the 4-bit checkpoint.
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// One loaded tensor: affine-quantized, or a dense array.
enum MlxWeight {
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
    },
    Dense(Array),
}

fn mlx_error(operation: &'static str) -> impl Fn(mlx_rs::error::Exception) -> FamilyError {
    move |error| FamilyError::Mlx(format!("{operation}: {error}"))
}

/// FP32 scalar constant (f64 arrays are unsupported on the GPU).
fn scalar_f32(value: f64) -> Array {
    Array::from_slice(&[value as f32], &[])
}

/// Loaded 4-bit Clef model.
pub struct MlxClefModel {
    tensors: BTreeMap<String, MlxWeight>,
    geometry: Qwen35Geometry,
    head: JointSchemaHead,
    runtime: SharedMlxRuntime,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared qwen35 MLX backbone).
unsafe impl Sync for MlxClefModel {}
unsafe impl Send for MlxClefModel {}

impl MlxClefModel {
    /// Digest-verify and load the pinned profile under `model_root`.
    pub fn load(
        model_root: impl AsRef<Path>,
        profile: &'static super::ClefProfile,
        runtime: SharedMlxRuntime,
    ) -> Result<Self, FamilyError> {
        let model_root = model_root.as_ref();
        let tensors = runtime.execute(|| load_tensors(model_root, profile))??;
        let head = JointSchemaHead::load(
            &profile.joint_head_config,
            &model_root.join(profile.joint_head.0),
            &candle_core::Device::Cpu,
        )?;
        Ok(Self {
            tensors,
            geometry: profile.geometry,
            head,
            runtime,
        })
    }

    fn get(&self, name: &str) -> Result<&MlxWeight, FamilyError> {
        self.tensors.get(name).ok_or_else(|| {
            FamilyError::InvalidInput(format!("tensor `{name}` missing from checkpoint"))
        })
    }

    /// `x @ w.T` through the quantized or dense kernel.
    fn project(&self, name: &str, x: &Array) -> Result<Array, FamilyError> {
        match self.get(name)? {
            MlxWeight::Quantized {
                weight,
                scales,
                biases,
            } => mlx_rs::ops::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                true,
                GROUP_SIZE,
                BITS,
            )
            .map_err(mlx_error("quantized matmul")),
            MlxWeight::Dense(tensor) => {
                let transposed = tensor.transpose().map_err(mlx_error("weight transpose"))?;
                Ok(x * transposed)
            }
        }
    }

    /// Raw (unfolded) norm weight as a dense array.
    fn norm_weight(&self, name: &str, width: usize) -> Result<Array, FamilyError> {
        let tensor = match self.get(name)? {
            MlxWeight::Dense(tensor) => tensor,
            MlxWeight::Quantized { .. } => {
                return Err(FamilyError::InvalidInput(format!(
                    "norm tensor `{name}` unexpectedly quantized"
                )));
            }
        };
        let shape = tensor.shape();
        if shape.len() != 1 || shape[0] as usize != width {
            return Err(FamilyError::ContractMismatch {
                field: "norm_shape",
                expected: format!("[{width}]"),
                actual: format!("{shape:?}"),
            });
        }
        Ok(tensor.clone())
    }

    /// Per-head host scalars (dt_bias, A_log).
    fn f32_vector(&self, name: &str, width: usize) -> Result<Vec<f32>, FamilyError> {
        let tensor = self
            .norm_weight(name, width)?
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(mlx_error("vector widen"))?;
        tensor.eval().map_err(mlx_error("vector eval"))?;
        tensor
            .to_vec_cast::<f32>()
            .map_err(|error| FamilyError::Mlx(format!("vector read: {error}")))
    }

    /// The layer prefix namespace of this checkpoint.
    fn prefix(&self, layer: usize) -> String {
        format!("language_model.model.layers.{layer}")
    }

    fn layer_kernels(
        &self,
        prefix: &str,
        geometry: Qwen35Geometry,
    ) -> Result<(Array, Array, Array, Array), FamilyError> {
        let conv = match self.get(&format!("{prefix}.linear_attn.conv1d.weight"))? {
            MlxWeight::Dense(array) => {
                // The checkpoint stores the taps as [channels, kernel, 1];
                // the conv math wants [channels, kernel].
                let widened = array
                    .as_dtype(mlx_rs::Dtype::Float32)
                    .map_err(mlx_error("conv widen"))?;
                widened.eval().map_err(mlx_error("conv eval"))?;
                widened
                    .reshape(&[geometry.qkv_size() as i32, geometry.conv_kernel as i32])
                    .map_err(mlx_error("conv reshape"))?
            }
            MlxWeight::Quantized { .. } => {
                return Err(FamilyError::InvalidInput("conv1d quantized".into()))
            }
        };
        let dt_bias = Array::from_slice(
            &self.f32_vector(
                &format!("{prefix}.linear_attn.dt_bias"),
                geometry.value_heads,
            )?,
            &[geometry.value_heads as i32],
        );
        let a_log = Array::from_slice(
            &self.f32_vector(&format!("{prefix}.linear_attn.A_log"), geometry.value_heads)?,
            &[geometry.value_heads as i32],
        );
        let delta_norm = Array::from_slice(
            &self.f32_vector(
                &format!("{prefix}.linear_attn.norm.weight"),
                geometry.head_dim,
            )?,
            &[geometry.head_dim as i32],
        );
        Ok((conv, dt_bias, a_log, delta_norm))
    }

    /// Run the whole backbone, returning the final-norm hidden states as
    /// host FP32 row-major `[token_count, hidden_size]` values.
    fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let geometry = self.geometry;
        let hidden_size = geometry.hidden_size;
        let token_count = input_ids.len();
        let ids = Array::from_slice(input_ids, &[token_count as i32]);
        let hidden = match self.get("language_model.model.embed_tokens.weight")? {
            MlxWeight::Quantized {
                weight,
                scales,
                biases,
            } => {
                let rows = weight.take_axis(&ids, 0).map_err(mlx_error("embed rows"))?;
                let row_scales = scales
                    .take_axis(&ids, 0)
                    .map_err(mlx_error("embed scales"))?;
                let row_biases = biases
                    .as_ref()
                    .ok_or_else(|| FamilyError::InvalidInput("embed needs biases".into()))?
                    .take_axis(&ids, 0)
                    .map_err(mlx_error("embed biases"))?;
                mlx_rs::ops::dequantize(&rows, &row_scales, Some(&row_biases), GROUP_SIZE, BITS)
                    .map_err(mlx_error("embed dequantize"))?
            }
            MlxWeight::Dense(tensor) => {
                tensor.take_axis(&ids, 0).map_err(mlx_error("embed rows"))?
            }
        };
        let mut hidden = hidden;
        for layer in 0..geometry.layer_count {
            let prefix = self.prefix(layer);
            let normalized = rms_norm(
                &hidden,
                &self.norm_weight(&format!("{prefix}.input_layernorm.weight"), hidden_size)?,
            )?;
            let attention = if geometry.is_full_attention(layer) {
                self.full_attention(&prefix, &normalized, token_count, geometry)?
            } else {
                self.linear_attention(&prefix, &normalized, token_count, geometry)?
            };
            let mixed = hidden + attention;
            let post = rms_norm(
                &mixed,
                &self.norm_weight(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    hidden_size,
                )?,
            )?;
            let gate = self.project(&format!("{prefix}.mlp.gate_proj.weight"), &post)?;
            let up = self.project(&format!("{prefix}.mlp.up_proj.weight"), &post)?;
            let activated = mlx_rs::ops::sigmoid(&gate).map_err(mlx_error("sigmoid"))? * gate;
            let down =
                self.project(&format!("{prefix}.mlp.down_proj.weight"), &(activated * up))?;
            hidden = mixed + down;
        }
        let final_hidden = rms_norm(
            &hidden,
            &self.norm_weight("language_model.model.norm.weight", hidden_size)?,
        )?;
        let flattened = final_hidden
            .reshape(&[(token_count * hidden_size) as i32])
            .map_err(mlx_error("final flatten"))?;
        host_f32(&flattened)
    }

    fn linear_attention(
        &self,
        prefix: &str,
        normalized: &Array,
        token_count: usize,
        geometry: Qwen35Geometry,
    ) -> Result<Array, FamilyError> {
        let raw_qkv = self.project(
            &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
            normalized,
        )?;
        let z_all = self.project(
            &format!("{prefix}.linear_attn.in_proj_z.weight"),
            normalized,
        )?;
        let beta_all = self.project(
            &format!("{prefix}.linear_attn.in_proj_b.weight"),
            normalized,
        )?;
        let decay_all = self.project(
            &format!("{prefix}.linear_attn.in_proj_a.weight"),
            normalized,
        )?;
        let (conv1d, dt_bias, a_log, delta_norm) = self.layer_kernels(prefix, geometry)?;

        let value_heads = geometry.value_heads;
        let head_dim = geometry.head_dim;
        let key_heads = geometry.key_heads;
        let qkv_size = geometry.qkv_size();
        let value_size = geometry.value_size();
        let key_size = geometry.key_size();
        let kernel = geometry.conv_kernel;

        // Per-token reference-ops DeltaNet with FP32 recurrent state, the
        // same equations as the shared qwen35 MLX layer kernel.
        let zeros_conv = vec![0_f32; qkv_size * kernel];
        let mut conv = Array::from_slice(&zeros_conv, &[qkv_size as i32, kernel as i32]);
        let zeros_recurrent = vec![0_f32; value_heads * head_dim * head_dim];
        let mut recurrent = Array::from_slice(
            &zeros_recurrent,
            &[value_heads as i32, head_dim as i32, head_dim as i32],
        );
        // Value head v reads key head v / (value_heads / key_heads).
        let groups = value_heads / key_heads;
        let key_index: Vec<u32> = (0..value_heads)
            .map(|head| (head / groups) as u32)
            .collect();
        let key_index = Array::from_slice(&key_index, &[value_heads as i32]);
        if false && prefix.ends_with("layers.0") {
            let dump = |array: &Array| -> Vec<f32> {
                host_f32(
                    &array
                        .reshape(&[-1])
                        .map_err(mlx_error("dump flat"))
                        .unwrap_or_else(|_| Array::from_slice(&[f32::NAN], &[])),
                )
                .unwrap_or_default()
            };
            let qkv_values = dump(&raw_qkv);
            let z_values = dump(&z_all);
            let beta_values = dump(&beta_all);
            let decay_values = dump(&decay_all);
            let payload = serde_json::json!({
                "token_count": token_count,
                "qkv": qkv_values,
                "z": z_values,
                "beta": beta_values,
                "decay": decay_values,
            });
            std::fs::write(
                "/tmp/clef-mlx-layer0-inputs.json",
                serde_json::to_vec(&payload).map_err(|error| FamilyError::Json {
                    path: std::path::PathBuf::from("dump"),
                    source: error,
                })?,
            )
            .map_err(|source| FamilyError::Io {
                path: std::path::PathBuf::from("/tmp/clef-mlx-layer0-inputs.json"),
                source,
            })?;
            eprintln!("mlx debug: dumped layer 0 inputs");
        }
        let mut mixed_rows: Vec<Array> = Vec::with_capacity(token_count);
        for row in 0..token_count {
            let index = Array::from_slice(&[row as u32], &[1]);
            let raw_row = raw_qkv
                .take_axis(&index, 0)
                .map_err(mlx_error("qkv row"))?
                .reshape(&[qkv_size as i32, 1])
                .map_err(mlx_error("qkv row col"))?;
            // Causal depthwise conv window: drop the oldest tap, append the
            // current raw row, then sum taps with silu.
            let keep = kernel - 1;
            let window_indices: Vec<u32> = (1..kernel as u32).collect();
            let shifted = conv
                .take_axis(Array::from_slice(&window_indices, &[keep as i32]), 1)
                .map_err(mlx_error("conv shift"))?;
            let window = mlx_rs::ops::concatenate(&[shifted, raw_row], 1)
                .map_err(mlx_error("conv window"))?;
            if std::env::var("OPENKIND_CLEF_MLX_DEBUG").is_ok() && row == 0 {
                eprintln!(
                    "mlx debug: conv1d {:?} window {:?} summed_in {:?}",
                    conv1d.shape(),
                    window.shape(),
                    (window.clone() * conv1d.clone()).shape(),
                );
            }
            let summed = (window.clone() * conv1d.clone())
                .sum_axis(1, false)
                .map_err(mlx_error("conv sum"))?;
            let qkv_row = mlx_rs::nn::silu(&summed).map_err(mlx_error("conv silu"))?;
            if std::env::var("OPENKIND_CLEF_MLX_DEBUG").is_ok()
                && row == 0
                && prefix.ends_with("layers.0")
            {
                let conv_host = host_f32(&qkv_row.reshape(&[-1]).map_err(mlx_error("c flat"))?)?;
                eprintln!(
                    "mlx debug: model conv row0 ch0..3 {:?} |v|max {:.4}",
                    &conv_host[..4],
                    conv_host[4096..]
                        .iter()
                        .copied()
                        .fold(0.0_f32, |m, v| m.max(v.abs())),
                );
            }
            // Split the fused q/k/v row into per-head blocks.
            let queries = qkv_row
                .try_index((0..key_size as i32,))
                .map_err(mlx_error("q slice"))?
                .reshape(&[key_heads as i32, head_dim as i32])
                .map_err(mlx_error("q heads"))?;
            let keys = qkv_row
                .try_index((key_size as i32..(key_size * 2) as i32,))
                .map_err(mlx_error("k slice"))?
                .reshape(&[key_heads as i32, head_dim as i32])
                .map_err(mlx_error("k heads"))?;
            let values = qkv_row
                .try_index(((key_size * 2) as i32..qkv_size as i32,))
                .map_err(mlx_error("v slice"))?
                .reshape(&[value_heads as i32, head_dim as i32])
                .map_err(mlx_error("v heads"))?;
            let beta_row = beta_all
                .take_axis(&index, 0)
                .map_err(mlx_error("beta row"))?
                .reshape(&[value_heads as i32])
                .map_err(mlx_error("beta heads"))?;
            let decay_row = decay_all
                .take_axis(&index, 0)
                .map_err(mlx_error("decay row"))?
                .reshape(&[value_heads as i32])
                .map_err(mlx_error("decay heads"))?;
            let z_row = z_all
                .take_axis(&index, 0)
                .map_err(mlx_error("z row"))?
                .reshape(&[value_heads as i32, head_dim as i32])
                .map_err(mlx_error("z heads"))?;
            // Offset RMSNorm over each head, then expand q/k per value head.
            let normalize = |heads: &Array| -> Result<Array, FamilyError> {
                let scale = (heads.clone() * heads.clone())
                    .mean_axis(-1, Some(true))
                    .map_err(mlx_error("qk mean"))?
                    + scalar_f32(1e-6);
                let scale = scale
                    .sqrt()
                    .map_err(mlx_error("qk sqrt"))?
                    .reciprocal()
                    .map_err(mlx_error("qk recip"))?;
                Ok(heads * scale)
            };
            let queries = normalize(&queries)?;
            let keys = normalize(&keys)?;
            let queries_e = queries
                .take_axis(&key_index, 0)
                .map_err(mlx_error("q expand"))?;
            let keys_e = keys
                .take_axis(&key_index, 0)
                .map_err(mlx_error("k expand"))?;
            // Decay and beta gates: decay = exp(-exp(A_log) * softplus(dt)).
            let beta = mlx_rs::ops::sigmoid(&beta_row).map_err(mlx_error("beta sigmoid"))?;
            let decay_argument = mlx_rs::nn::softplus(&(decay_row + dt_bias.clone()))
                .map_err(mlx_error("dt softplus"))?;
            let decay = mlx_rs::ops::exp(
                &(-mlx_rs::ops::exp(&a_log).map_err(mlx_error("a exp"))? * decay_argument),
            )
            .map_err(mlx_error("decay exp"))?;
            let decay = decay
                .reshape(&[value_heads as i32, 1, 1])
                .map_err(mlx_error("decay shape"))?;
            recurrent = recurrent.clone() * decay.clone();
            // The state layout is S[key, value]: the readout contracts the
            // KEY axis, so memory = k^T @ S per head.
            let keys_col = keys_e
                .reshape(&[value_heads as i32, head_dim as i32, 1])
                .map_err(mlx_error("k col"))?;
            let keys_row = keys_col
                .transpose_axes(&[0, 2, 1])
                .map_err(mlx_error("k row"))?;
            let memory = keys_row
                .matmul(&recurrent)
                .map_err(mlx_error("memory matmul"))?
                .transpose_axes(&[0, 2, 1])
                .map_err(mlx_error("memory shape"))?;
            let beta = beta
                .reshape(&[value_heads as i32, 1, 1])
                .map_err(mlx_error("beta shape"))?;
            let values_col = values
                .reshape(&[value_heads as i32, head_dim as i32, 1])
                .map_err(mlx_error("v col"))?;
            let delta = (values_col - memory) * beta.clone();
            // S[v] += k[v] outer delta[v]
            let update = keys_col
                .matmul(
                    &delta
                        .transpose_axes(&[0, 2, 1])
                        .map_err(mlx_error("delta t"))?,
                )
                .map_err(mlx_error("outer matmul"))?;
            recurrent += update;
            if std::env::var("OPENKIND_CLEF_MLX_DEBUG").is_ok()
                && row == 0
                && prefix.ends_with("layers.0")
            {
                let k_host = host_f32(&keys_e.reshape(&[-1]).map_err(mlx_error("k flat"))?)?;
                for vh in 0..3 {
                    let kh = vh / 2;
                    let k_row = &k_host[kh * head_dim..(kh + 1) * head_dim];
                    let k_hat_norm = k_row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let k_exp_norm = k_host[vh * head_dim..(vh + 1) * head_dim]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt();
                    eprintln!(
                        "mlx debug: row0 vh {vh} |k_hat[{kh}]| {k_hat_norm:.3} |k_exp| {k_exp_norm:.3}"
                    );
                }
                let delta_host = host_f32(&delta.reshape(&[-1]).map_err(mlx_error("d flat"))?)?;
                let delta_max = delta_host.iter().copied().fold(0.0_f32, |m, v| {
                    if v.is_finite() {
                        m.max(v.abs())
                    } else {
                        m
                    }
                });
                eprintln!("mlx debug: row0 |delta|max {delta_max:.3}");
            }
            if std::env::var("OPENKIND_CLEF_MLX_DEBUG").is_ok()
                && (row < 3 || row % 60 == 0 || row == token_count - 1)
            {
                let state_vals = host_f32(&recurrent.reshape(&[-1]).map_err(mlx_error("s flat"))?)?;
                let decay_vals = host_f32(&decay.reshape(&[-1]).map_err(mlx_error("d flat"))?)?;
                let beta_vals = host_f32(&beta.reshape(&[-1]).map_err(mlx_error("b flat"))?)?;
                let s_max = state_vals.iter().copied().fold(0.0_f32, |m, v| {
                    if v.is_finite() {
                        m.max(v.abs())
                    } else {
                        m
                    }
                });
                let s_nan = state_vals.iter().filter(|v| v.is_nan()).count();
                eprintln!(
                    "mlx debug: row {row} decay[0..4]={:?} beta[0..4]={:?} state_absmax {s_max:.4} state_nan {s_nan}",
                    &decay_vals[..4.min(decay_vals.len())],
                    &beta_vals[..4.min(beta_vals.len())],
                );
            }
            // out[v] = q^T @ S per head, then offset RMSNorm * silu(z).
            let queries_row = queries_e
                .reshape(&[value_heads as i32, 1, head_dim as i32])
                .map_err(mlx_error("q row"))?;
            let out = queries_row
                .matmul(&recurrent)
                .map_err(mlx_error("out matmul"))?
                .reshape(&[value_heads as i32, head_dim as i32])
                .map_err(mlx_error("out reshape"))?;
            let out_scale = (out.clone() * out.clone())
                .mean_axis(-1, Some(true))
                .map_err(mlx_error("out mean"))?
                + scalar_f32(1e-6);
            let out_scale = out_scale
                .sqrt()
                .map_err(mlx_error("out sqrt"))?
                .reciprocal()
                .map_err(mlx_error("out recip"))?;
            let gated = (out * out_scale)
                * delta_norm.clone()
                * mlx_rs::nn::silu(&z_row).map_err(mlx_error("z silu"))?;
            mixed_rows.push(
                gated
                    .reshape(&[value_size as i32])
                    .map_err(mlx_error("mixed row reshape"))?,
            );
            conv = window;
        }
        let stacked = mlx_rs::ops::stack(&mixed_rows, 0).map_err(mlx_error("mixed stack"))?;
        self.project(&format!("{prefix}.linear_attn.out_proj.weight"), &stacked)
    }

    fn full_attention(
        &self,
        prefix: &str,
        normalized: &Array,
        token_count: usize,
        geometry: Qwen35Geometry,
    ) -> Result<Array, FamilyError> {
        let heads = geometry.attention_heads;
        let kv_heads = geometry.kv_heads;
        let head_dim = geometry.attention_head_dim;
        let rotary_dim = geometry.rotary_dim();
        let projected_q = self.project(&format!("{prefix}.self_attn.q_proj.weight"), normalized)?;
        let keys = self.project(&format!("{prefix}.self_attn.k_proj.weight"), normalized)?;
        let values = self.project(&format!("{prefix}.self_attn.v_proj.weight"), normalized)?;
        // The query projection fuses per-head query and sigmoid-gate rows.
        let q_reshaped = projected_q
            .reshape(&[token_count as i32, heads as i32, 2, head_dim as i32])
            .map_err(mlx_error("q reshape"))?;
        let query = q_reshaped
            .take_axis(Array::from_slice(&[0_u32], &[1]), 2)
            .map_err(mlx_error("q split"))?
            .reshape(&[token_count as i32, heads as i32, head_dim as i32])
            .map_err(mlx_error("q split reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("q heads"))?;
        let gates = q_reshaped
            .take_axis(Array::from_slice(&[1_u32], &[1]), 2)
            .map_err(mlx_error("gate split"))?
            .reshape(&[token_count as i32, heads as i32, head_dim as i32])
            .map_err(mlx_error("gate split reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("gate heads"))?;
        let keys_h = keys
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("k reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("k heads"))?;
        let values_h = values
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("v reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("v heads"))?;
        // Per-head offset RMSNorm on q/k.
        let q_norm = self.norm_weight(&format!("{prefix}.self_attn.q_norm.weight"), head_dim)?;
        let k_norm = self.norm_weight(&format!("{prefix}.self_attn.k_norm.weight"), head_dim)?;
        let normalize_heads = |x: &Array, weight: &Array| -> Result<Array, FamilyError> {
            let scale = (x.clone() * x.clone())
                .mean_axis(-1, Some(true))
                .map_err(mlx_error("head mean"))?
                + scalar_f32(1e-6);
            let scale = scale
                .sqrt()
                .map_err(mlx_error("head sqrt"))?
                .reciprocal()
                .map_err(mlx_error("head recip"))?;
            let folded = weight + scalar_f32(1.0);
            Ok((x / scale) * folded)
        };
        let query = normalize_heads(&query, &q_norm)?;
        let keys_h = normalize_heads(&keys_h, &k_norm)?;
        // Partial rotary over the first `rotary_dim` entries, NeoX pairing
        // at theta 1e7. cos/sin are [token, rotary/2], broadcast over heads.
        let rotate = |x: &Array| -> Result<Array, FamilyError> {
            let rotary_half = (rotary_dim / 2) as i32;
            let mut cos_rows = Vec::with_capacity(token_count);
            let mut sin_rows = Vec::with_capacity(token_count);
            for token in 0..token_count {
                let mut cos_row = Vec::with_capacity(rotary_dim / 2);
                let mut sin_row = Vec::with_capacity(rotary_dim / 2);
                for index in 0..rotary_dim / 2 {
                    let frequency = 1e7_f64.powf(-((2 * index) as f64) / rotary_dim as f64);
                    let (sin, cos) = ((token as f64) * frequency).sin_cos();
                    cos_row.push(cos as f32);
                    sin_row.push(sin as f32);
                }
                cos_rows.push(Array::from_slice(&cos_row, &[1, (rotary_dim / 2) as i32]));
                sin_rows.push(Array::from_slice(&sin_row, &[1, (rotary_dim / 2) as i32]));
            }
            let cos = mlx_rs::ops::concatenate(&cos_rows, 0)
                .map_err(mlx_error("rope cos"))?
                .reshape(&[token_count as i32, 1, rotary_half])
                .map_err(mlx_error("rope cos reshape"))?;
            let sin = mlx_rs::ops::concatenate(&sin_rows, 0)
                .map_err(mlx_error("rope sin"))?
                .reshape(&[token_count as i32, 1, rotary_half])
                .map_err(mlx_error("rope sin reshape"))?;
            // x layout [heads, tokens, head_dim]; move tokens to axis 0 for
            // the broadcast, rotate the leading half against the trailing.
            let tokens_axis = x.transpose_axes(&[1, 0, 2]).map_err(mlx_error("rope t1"))?;
            let half = (rotary_dim / 2) as i32;
            let first = tokens_axis
                .try_index((.., .., 0..half))
                .map_err(mlx_error("rope first"))?;
            let second = tokens_axis
                .try_index((.., .., half..rotary_dim as i32))
                .map_err(mlx_error("rope second"))?;
            let head_count = x.shape()[0];
            let broadcast_shape = [token_count as i32, head_count, half];
            let cos_b =
                mlx_rs::ops::broadcast_to(&cos, &broadcast_shape).map_err(mlx_error("cos b"))?;
            let sin_b =
                mlx_rs::ops::broadcast_to(&sin, &broadcast_shape).map_err(mlx_error("sin b"))?;
            let rotated_first = first.clone() * cos_b.clone() - second.clone() * sin_b.clone();
            let rotated_second = second * cos_b + first * sin_b;
            let rotated = mlx_rs::ops::concatenate(&[rotated_first, rotated_second], 2)
                .map_err(mlx_error("rope cat"))?;
            let tail = tokens_axis
                .try_index((.., .., rotary_dim as i32..head_dim as i32))
                .map_err(mlx_error("rope tail"))?;
            let full =
                mlx_rs::ops::concatenate(&[rotated, tail], 2).map_err(mlx_error("rope full"))?;
            full.transpose_axes(&[1, 0, 2])
                .map_err(mlx_error("rope t2"))
        };
        let query = rotate(&query)?;
        let keys_h = rotate(&keys_h)?;
        // Grouped-query attention.
        let group = heads / kv_heads;
        // Put the KV-head axis before the repeat axis so query head `h`
        // uses KV head `h / group`, matching contiguous GQA head mapping.
        let keys_r = mlx_rs::ops::stack(&(0..group).map(|_| keys_h.clone()).collect::<Vec<_>>(), 0)
            .map_err(mlx_error("k repeat"))?
            .transpose_axes(&[1, 0, 2, 3])
            .map_err(mlx_error("k repeat transpose"))?
            .reshape(&[heads as i32, token_count as i32, head_dim as i32])
            .map_err(mlx_error("k repeat reshape"))?;
        let values_r =
            mlx_rs::ops::stack(&(0..group).map(|_| values_h.clone()).collect::<Vec<_>>(), 0)
                .map_err(mlx_error("v repeat"))?
                .transpose_axes(&[1, 0, 2, 3])
                .map_err(mlx_error("v repeat transpose"))?
                .reshape(&[heads as i32, token_count as i32, head_dim as i32])
                .map_err(mlx_error("v repeat reshape"))?;
        let scores = query
            .matmul(
                &keys_r
                    .transpose_axes(&[0, 2, 1])
                    .map_err(mlx_error("k t"))?,
            )
            .map_err(mlx_error("scores"))?;
        let scores = scores * scalar_f32(1.0 / (head_dim as f64).sqrt());
        let probabilities =
            mlx_rs::ops::softmax_axis(&scores, -1, false).map_err(mlx_error("softmax"))?;
        let mixed = probabilities
            .matmul(&values_r)
            .map_err(mlx_error("mix"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("mix heads"))?
            .reshape(&[token_count as i32, geometry.attention_size() as i32])
            .map_err(mlx_error("mix flatten"))?;
        // Sigmoid attention gate over the fused per-head query/gate rows.
        let gates_flat = gates
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("gate t"))?
            .reshape(&[token_count as i32, geometry.attention_size() as i32])
            .map_err(mlx_error("gate flat"))?;
        let gated = mixed * mlx_rs::ops::sigmoid(&gates_flat).map_err(mlx_error("gate sigmoid"))?;
        self.project(&format!("{prefix}.self_attn.o_proj.weight"), &gated)
    }
}

impl ClefExecutionModel for MlxClefModel {
    fn evaluate_record(
        &self,
        encoded: &EncodedRecord,
        control: &FamilyControl,
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        control.check()?;
        let hidden_size = self.geometry.hidden_size;
        let token_count = encoded.input_ids.len();
        let hidden = self
            .runtime
            .execute(|| self.forward_hidden(&encoded.input_ids))??;
        if hidden.len() != token_count * hidden_size {
            return Err(FamilyError::InvalidInput(
                "mlx backbone output width does not match the pinned geometry".into(),
            ));
        }
        control.check()?;
        // Collect every option-span token id and resolve its lm_head row
        // once (the lexical prior reads the untied output embedding).
        let mut lexical_ids: Vec<u32> = Vec::new();
        for question in &encoded.questions {
            for (start, end) in &question.option_spans {
                lexical_ids.extend_from_slice(&encoded.input_ids[*start..*end]);
            }
        }
        let lexical = self
            .runtime
            .execute(|| ResolvedLexical::resolve(&self.tensors, &lexical_ids, hidden_size))??;
        let result = self.head.forward(
            &hidden,
            token_count,
            encoded,
            &lexical,
            &candle_core::Device::Cpu,
        )?;
        control.check()?;
        Ok(result)
    }
}

/// Offset RMSNorm over the last axis with the folded `(1 + w)` weight.
fn rms_norm(x: &Array, weight: &Array) -> Result<Array, FamilyError> {
    let variance = (x.clone() * x.clone())
        .mean_axis(-1, Some(true))
        .map_err(mlx_error("rms mean"))?;
    // `scale` is the reciprocal RMS; the normalized activation is `x * scale`.
    let scale = (variance + scalar_f32(1e-6))
        .sqrt()
        .map_err(mlx_error("rms sqrt"))?
        .reciprocal()
        .map_err(mlx_error("rms recip"))?;
    let folded = weight + scalar_f32(1.0);
    Ok((x * scale) * folded)
}

fn host_f32(array: &Array) -> Result<Vec<f32>, FamilyError> {
    let widened = array
        .as_dtype(mlx_rs::Dtype::Float32)
        .map_err(mlx_error("host widen"))?;
    widened.eval().map_err(mlx_error("host eval"))?;
    widened
        .to_vec_cast::<f32>()
        .map_err(|error| FamilyError::Mlx(format!("host read: {error}")))
}

/// Pre-resolved lexical rows: every option-span token id of the record is
/// dequantized once on the MLX stream before the head reads them.
struct ResolvedLexical {
    rows: BTreeMap<u32, Vec<f32>>,
    hidden_size: usize,
}

impl ResolvedLexical {
    fn resolve(
        tensors: &BTreeMap<String, MlxWeight>,
        token_ids: &[u32],
        hidden_size: usize,
    ) -> Result<Self, FamilyError> {
        let mut unique: Vec<u32> = token_ids.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique.is_empty() {
            return Ok(Self {
                rows: BTreeMap::new(),
                hidden_size,
            });
        }
        let index = Array::from_slice(&unique, &[unique.len() as i32]);
        let rows_flat = match tensors.get("language_model.lm_head.weight") {
            Some(MlxWeight::Quantized {
                weight,
                scales,
                biases,
            }) => {
                let rows = weight.take_axis(&index, 0).map_err(mlx_error("lm rows"))?;
                let row_scales = scales
                    .take_axis(&index, 0)
                    .map_err(mlx_error("lm scales"))?;
                let row_biases = biases
                    .as_ref()
                    .ok_or_else(|| FamilyError::InvalidInput("lm_head needs biases".into()))?
                    .take_axis(&index, 0)
                    .map_err(mlx_error("lm biases"))?;
                let dequantized = mlx_rs::ops::dequantize(
                    &rows,
                    &row_scales,
                    Some(&row_biases),
                    GROUP_SIZE,
                    BITS,
                )
                .map_err(mlx_error("lm dequantize"))?;
                host_f32(&dequantized.reshape(&[-1]).map_err(mlx_error("lm flat"))?)?
            }
            Some(MlxWeight::Dense(tensor)) => {
                let rows = tensor.take_axis(&index, 0).map_err(mlx_error("lm rows"))?;
                host_f32(&rows.reshape(&[-1]).map_err(mlx_error("lm flat"))?)?
            }
            None => {
                return Err(FamilyError::InvalidInput(
                    "lm_head.weight missing from checkpoint".into(),
                ));
            }
        };
        let mut rows = BTreeMap::new();
        for (position, id) in unique.iter().enumerate() {
            rows.insert(
                *id,
                rows_flat[position * hidden_size..(position + 1) * hidden_size].to_vec(),
            );
        }
        Ok(Self { rows, hidden_size })
    }
}

impl LexicalLookup for ResolvedLexical {
    fn rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let mut out = Vec::with_capacity(token_ids.len() * self.hidden_size);
        for id in token_ids {
            out.extend_from_slice(
                self.rows
                    .get(id)
                    .ok_or_else(|| FamilyError::InvalidInput("lexical row missing".into()))?,
            );
        }
        Ok(out)
    }
}

/// Read every checkpoint tensor, grouping quantized triplets.
fn load_tensors(
    model_root: &Path,
    profile: &'static super::ClefProfile,
) -> Result<BTreeMap<String, MlxWeight>, FamilyError> {
    for (name, digest, bytes) in profile.checkpoint_shards {
        let path = model_root.join(name);
        let length = std::fs::metadata(&path)
            .map_err(|source| FamilyError::Io {
                path: path.clone(),
                source,
            })?
            .len();
        if length != *bytes {
            return Err(FamilyError::ContractMismatch {
                field: "checkpoint_size",
                expected: format!("{name} is {bytes} bytes"),
                actual: format!("{length} bytes"),
            });
        }
        verify_digest(&path, digest)?;
    }
    let (head_name, head_digest, head_bytes) = profile.joint_head;
    let head_path = model_root.join(head_name);
    let head_length = std::fs::metadata(&head_path)
        .map_err(|source| FamilyError::Io {
            path: head_path.clone(),
            source,
        })?
        .len();
    if head_length != head_bytes {
        return Err(FamilyError::ContractMismatch {
            field: "joint_head_size",
            expected: format!("{head_name} is {head_bytes} bytes"),
            actual: format!("{head_length} bytes"),
        });
    }
    verify_digest(&head_path, head_digest)?;
    verify_digest(
        &model_root.join(profile.tokenizer_path),
        profile.tokenizer_json_sha256,
    )?;
    verify_digest(
        &model_root.join(profile.config_path),
        profile.config_json_sha256,
    )?;

    let mut tensors = BTreeMap::new();
    for (shard_name, _, _) in profile.checkpoint_shards {
        for (name, weight) in read_safetensors(&model_root.join(shard_name))? {
            if tensors.insert(name.clone(), weight).is_some() {
                return Err(FamilyError::InvalidInput(format!(
                    "duplicate tensor `{name}` across shards"
                )));
            }
        }
    }
    Ok(tensors)
}

fn read_safetensors(path: &Path) -> Result<BTreeMap<String, MlxWeight>, FamilyError> {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let mut file = std::fs::File::open(path).map_err(|source| FamilyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
        .map_err(|_| FamilyError::InvalidInput("safetensors header too large".into()))?;
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return Err(FamilyError::InvalidInput(format!(
            "invalid safetensors header length {header_len}"
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|source| FamilyError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    #[derive(serde::Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [u64; 2],
    }

    impl Entry {
        fn from_value(value: &serde_json::Value, tensor: &str) -> Result<Entry, FamilyError> {
            serde_json::from_value(value.clone()).map_err(|error| FamilyError::Json {
                path: std::path::PathBuf::from(tensor),
                source: error,
            })
        }
    }

    let entries: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|error| FamilyError::Json {
            path: path.to_path_buf(),
            source: error,
        })?;
    let data_start = 8_u64 + header_len as u64;

    let mut read_array = |entry: &Entry| -> Result<Array, FamilyError> {
        file.seek(SeekFrom::Start(data_start + entry.data_offsets[0]))
            .map_err(|source| FamilyError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let len = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
        let mut bytes = vec![0_u8; len];
        file.read_exact(&mut bytes)
            .map_err(|source| FamilyError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let shape: Vec<i32> = entry.shape.iter().map(|dim| *dim as i32).collect();
        let array = match entry.dtype.as_str() {
            "U32" => {
                let values: Vec<u32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            "F16" => {
                let values: Vec<half::f16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            "BF16" => {
                let values: Vec<half::bf16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            "F32" => {
                let values: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            other => {
                return Err(FamilyError::InvalidInput(format!(
                    "unsupported safetensors dtype {other}"
                )));
            }
        };
        array.eval().map_err(mlx_error("tensor materialization"))?;
        Ok(array)
    };

    let mut weights = BTreeMap::new();
    for (name, value) in &entries {
        if name.starts_with("__metadata__") || name.starts_with("vision_tower.") {
            continue;
        }
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        let entry = Entry::from_value(value, name)?;
        // MLX pairs a packed `X.weight` with sibling `X.scales`/`X.biases`
        // named after `X` without the `.weight` suffix.
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let scales_name = format!("{base}.scales");
        if let Some(scales_value) = entries.get(&scales_name) {
            let biases_name = format!("{base}.biases");
            let biases_value = entries.get(&biases_name).ok_or_else(|| {
                FamilyError::InvalidInput(format!("quantized tensor `{name}` missing biases"))
            })?;
            weights.insert(
                name.clone(),
                MlxWeight::Quantized {
                    weight: read_array(&entry)?,
                    scales: read_array(&Entry::from_value(scales_value, &scales_name)?)?,
                    biases: Some(read_array(&Entry::from_value(biases_value, &biases_name)?)?),
                },
            );
        } else {
            weights.insert(name.clone(), MlxWeight::Dense(read_array(&entry)?));
        }
    }
    Ok(weights)
}
