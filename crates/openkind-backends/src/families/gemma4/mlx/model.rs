//! MLX/Metal execution of the pinned Winnow-E4B GGUF (q8_0) decoder
//! (feature `mlx`, macOS arm64).
//!
//! The digest-verified checkpoint loads through the family's own GGUF
//! loader ([`super::gguf::load_checkpoint`]) — the same artifact
//! verification, pinned metadata contract, KV-shared dead-tensor skipping,
//! and file-backed embedding row readers the CPU oracle uses. Every compute
//! weight is then dequantized once to the exact FP32 values candle's block
//! arithmetic produces and loaded into FP32 MLX arrays; the two giant
//! embedding tables never become resident and stream only the rows a prompt
//! or the letter readout references.
//!
//! Arithmetic identity: the CPU oracle multiplies the q8_0 blocks of the
//! FFN/PLE/LM-head weights inside candle's quantized kernels (f16 block
//! scales) and runs the attention projections as dequantized F32 linears,
//! while this backend multiplies every weight fully dequantized in FP32 — a
//! different arithmetic identity whose parity gate is the workspace 0.005
//! probability budget, named by the backend id
//! (`gemma4-decision/mlx-fp32-deq`). The forward mirrors `super::backbone`
//! op for op: RMSNorm applying the stored weight directly (no `+1` shift),
//! attention softmax scale `1.0` (the learned Q/K norms carry the scale),
//! per-head Q/K RMSNorm, value RMS normalization without a learned weight,
//! HF `repeat_kv` GQA expansion, banded attention over query chunks of
//! [`ATTENTION_QUERY_CHUNK`], full-dimension RoPE (base `1e4`) on sliding
//! layers and proportional partial RoPE (first quarter of the head dims,
//! rotary factor `0.25`, base `1e6`) on full-attention layers, the trailing
//! KV-shared layers attending over their same-type donor's post-RoPE K/V,
//! `sqrt(hidden)`-scaled embeddings, the per-layer-embedding injection
//! (gate, activation, multiply, projection, post-norm, add), the optional
//! per-layer output scale, and the final-position readout with tanh
//! softcapping at `30`.
//!
//! Letter readout: the CPU oracle computes the full-vocabulary logits row
//! through the tied-head `QMatMul` and indexes the letter ids. Because row
//! selection only indexes the logits row and the softcap is elementwise,
//! computing `30 * tanh(dot(hidden, letter_row) / 30)` per letter
//! reproduces the indexed values exactly — the per-letter dots run
//! host-side in f64 against the file-backed q8_0 letter rows (byte-identical
//! to candle's block dequantization, cross-checked by the family's
//! `quantization_crosscheck` test). Only the kernel arithmetic differs,
//! which is the declared parity difference the backend id names.
//!
//! Memory: the dequantized FP32 compute weights of the pinned profile
//! (~15.9 GB resident MLX arrays in unified memory —
//! [`estimated_fp32_weight_bytes`]) are larger than the ~4.9 GB q8_0
//! artifact, which is the accepted cost of dequantized execution; the
//! precomputed RoPE tables add ~25 MB. Neither embedding table ever becomes
//! resident, and attention stays banded exactly like the oracle, so
//! per-forward scratch grows linearly with the prompt and no
//! `heads x seq x seq` tensor is ever materialized.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::collections::BTreeMap;
use std::ops::{Div, Sub};
use std::sync::Arc;

use candle_core::Device;
use mlx_rs::Array;

use crate::families::qwen35_survey_mlx::mlx_error;
use crate::families::support::{FamilyControl, FamilyError};
use crate::qwen35::mlx::MlxRuntime;

use super::super::backbone::Gemma4TextConfig;
use super::super::gguf::{load_checkpoint, Gemma4Checkpoint, GgufRowTable, VerifiedArtifacts};
use super::super::MAX_SEQUENCE_TOKENS;

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// q8_0 checkpoint dequantized to exact FP32 arrays computed on the Metal
/// GPU (the `-deq` marks the dequantized execution the backend id names).
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-deq-gemma4";

/// Query-block size of the banded attention, mirroring the oracle's
/// `ATTENTION_QUERY_CHUNK`: every score tensor the forward materializes is
/// bounded by `(heads, CHUNK, CHUNK + sliding_window)` on sliding layers and
/// `(heads, CHUNK, seq)` on full-attention layers.
const ATTENTION_QUERY_CHUNK: usize = 256;

/// The pinned family model on MLX: FP32-dequantized compute weights plus
/// the file-backed embedding row readers.
pub(super) struct Gemma4MlxModel {
    /// Compute weights keyed by their candle model names (the same names
    /// `super::gguf` maps the GGUF tensor names to).
    tensors: BTreeMap<String, Array>,
    /// Input-token embedding rows (also the tied LM-head source), streamed
    /// from the verified GGUF per prompt and per letter readout.
    token_embd_rows: GgufRowTable,
    /// Per-layer embedding rows, streamed from the verified GGUF per prompt.
    ple_rows: GgufRowTable,
    config: Gemma4TextConfig,
    /// Precomputed RoPE tables `(MAX_SEQUENCE_TOKENS, head_dim / 2)`:
    /// full-dimension base `1e4` for sliding layers and the proportional
    /// partial (rotary factor `0.25`, base `1e6`) tables for full-attention
    /// layers (identity rotation on the unrotated pairs).
    rope_local_cos: Array,
    rope_local_sin: Array,
    rope_global_cos: Array,
    rope_global_sin: Array,
    /// Layers whose own post-RoPE K/V a later KV-shared consumer reads —
    /// only these retain their K/V to the end of a forward, mirroring the
    /// oracle's `retains_donor_kv`.
    retains_donor_kv: Vec<bool>,
    /// Cap on the estimated per-forward transient scratch; a forward fails
    /// closed before allocating anything when a sequence's estimate exceeds
    /// it. The dequantized resident weights are not part of this estimate.
    scratch_budget: Option<u64>,
    runtime: Arc<MlxRuntime>,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared qwen35 MLX backbone).
unsafe impl Sync for Gemma4MlxModel {}
unsafe impl Send for Gemma4MlxModel {}

/// Conservative resident size, in bytes, of the FP32-dequantized compute
/// weights this backend loads for `cfg` (~15.9 GB for the pinned profile,
/// against the ~4.9 GB q8_0 checkpoint the CPU oracle maps). The two
/// embedding tables and the precomputed RoPE tables are not included: the
/// tables stay file-backed and the rope tables are ~25 MB.
pub fn estimated_fp32_weight_bytes(cfg: &Gemma4TextConfig) -> u64 {
    let hidden = cfg.hidden_size as u64;
    let inter = cfg.intermediate_size as u64;
    let ple = cfg.per_layer_input_dim as u64;
    let mut bytes = hidden
        .saturating_add(ple.saturating_mul(hidden + 1))
        .saturating_mul(4);
    for layer in 0..cfg.num_hidden_layers {
        let head_dim = if cfg.is_sliding(layer) {
            cfg.head_dim
        } else {
            cfg.global_head_dim
        } as u64;
        let q_out = cfg.num_attention_heads as u64 * head_dim;
        let mut layer_bytes = 4u64
            .saturating_mul(hidden)
            .saturating_add(head_dim)
            .saturating_add(2u64.saturating_mul(q_out.saturating_mul(hidden)))
            .saturating_add(3u64.saturating_mul(inter.saturating_mul(hidden)))
            .saturating_add(2u64.saturating_mul(ple.saturating_mul(hidden)))
            .saturating_add(hidden);
        if cfg.kv_donor(layer).is_none() {
            layer_bytes = layer_bytes
                .saturating_add(2u64.saturating_mul(
                    (cfg.num_key_value_heads as u64).saturating_mul(head_dim) * hidden,
                ))
                .saturating_add(head_dim);
        }
        bytes = bytes.saturating_add(layer_bytes.saturating_mul(4));
    }
    bytes
}

/// Dequantize one checkpoint tensor to its exact FP32 host values and shape
/// (candle's block arithmetic, on the CPU device).
fn dequantized(
    checkpoint: &Gemma4Checkpoint,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), FamilyError> {
    let qtensor = checkpoint.q_tensor(name)?;
    let shape = qtensor.shape().dims().to_vec();
    let values = qtensor
        .dequantize(&Device::Cpu)
        .and_then(|tensor| {
            tensor
                .flatten_all()
                .and_then(|tensor| tensor.to_vec1::<f32>())
        })
        .map_err(FamilyError::Candle)?;
    Ok((values, shape))
}

impl Gemma4MlxModel {
    /// Load the pinned checkpoint through the family GGUF loader,
    /// dequantize every compute weight to FP32, and build the MLX arrays.
    ///
    /// Must run inside [`MlxRuntime::execute`]: the caller holds the
    /// process-wide execution lock for array construction. Each tensor is
    /// dequantized to a transient host vector, immediately copied into an
    /// evaluated MLX array, and released, so no two FP32 tensors are ever
    /// host-resident together and the 262144-row embedding tables are never
    /// dequantized at all.
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        cfg: &Gemma4TextConfig,
        scratch_budget: Option<u64>,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let checkpoint = load_checkpoint(artifacts, cfg, &Device::Cpu)?;
        let token_embd_rows = checkpoint.token_embd_rows.clone();
        let ple_rows = checkpoint.ple_rows.clone();
        let mut tensors = BTreeMap::new();
        let mut put = |name: String, expected: Vec<usize>| -> Result<(), FamilyError> {
            let (values, shape) = dequantized(&checkpoint, &name)?;
            if shape != expected {
                return Err(FamilyError::contract(
                    "gguf.tensor.shape",
                    format!("`{name}` shaped {expected:?}"),
                    format!("{shape:?}"),
                ));
            }
            let dims: Vec<i32> = expected.iter().map(|extent| *extent as i32).collect();
            let array = Array::from_slice(&values, &dims);
            array.eval().map_err(mlx_error("weight materialization"))?;
            if tensors.insert(name, array).is_some() {
                return Err(FamilyError::InvalidInput(
                    "duplicate decoder tensor in the checkpoint".to_owned(),
                ));
            }
            Ok(())
        };
        put("norm.weight".to_owned(), vec![cfg.hidden_size])?;
        for layer in 0..cfg.num_hidden_layers {
            let head_dim = if cfg.is_sliding(layer) {
                cfg.head_dim
            } else {
                cfg.global_head_dim
            };
            let q_out = cfg.num_attention_heads * head_dim;
            let kv_out = cfg.num_key_value_heads * head_dim;
            let hidden = cfg.hidden_size;
            let prefix = format!("layers.{layer}");
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ] {
                put(format!("{prefix}.{suffix}"), vec![hidden])?;
            }
            put(
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_out, hidden],
            )?;
            put(format!("{prefix}.self_attn.q_norm.weight"), vec![head_dim])?;
            put(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, q_out],
            )?;
            // KV-shared layers reuse their donor's K/V; their own dead K/V
            // tensors were already dropped by `load_checkpoint`, matching
            // what the reference runtimes load.
            if cfg.kv_donor(layer).is_none() {
                put(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![kv_out, hidden],
                )?;
                put(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![kv_out, hidden],
                )?;
                put(format!("{prefix}.self_attn.k_norm.weight"), vec![head_dim])?;
            }
            for suffix in ["gate_proj.weight", "up_proj.weight"] {
                put(
                    format!("{prefix}.mlp.{suffix}"),
                    vec![cfg.intermediate_size, hidden],
                )?;
            }
            put(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, cfg.intermediate_size],
            )?;
            if cfg.per_layer_input_dim > 0 {
                put(
                    format!("{prefix}.per_layer_input_gate.weight"),
                    vec![cfg.per_layer_input_dim, hidden],
                )?;
                put(
                    format!("{prefix}.per_layer_projection.weight"),
                    vec![hidden, cfg.per_layer_input_dim],
                )?;
                // The post-PLE norm normalizes the projected branch at the
                // hidden width (the oracle applies it to
                // `per_layer_projection`'s output, not to the gated
                // per-layer slice).
                put(
                    format!("{prefix}.post_per_layer_input_norm.weight"),
                    vec![hidden],
                )?;
            }
            // Optional per-layer output scale: the oracle treats it as
            // present-when-loaded and broadcast-multiplies it.
            if checkpoint
                .q_tensor(&format!("{prefix}.layer_output_scale.weight"))
                .is_ok()
            {
                let (_, shape) =
                    dequantized(&checkpoint, &format!("{prefix}.layer_output_scale.weight"))?;
                let broadcastable = shape.len() == 1 && (shape[0] == hidden || shape[0] == 1);
                if !broadcastable {
                    return Err(FamilyError::contract(
                        "gguf.tensor.shape",
                        format!("a broadcastable per-layer output scale against width {hidden}"),
                        format!("{shape:?}"),
                    ));
                }
                put(format!("{prefix}.layer_output_scale.weight"), shape)?;
            }
        }
        if cfg.per_layer_input_dim > 0 {
            put(
                "per_layer_model_projection.weight".to_owned(),
                vec![
                    cfg.num_hidden_layers * cfg.per_layer_input_dim,
                    cfg.hidden_size,
                ],
            )?;
            put(
                "per_layer_projection_norm.weight".to_owned(),
                vec![cfg.per_layer_input_dim],
            )?;
        }
        let (rope_local_cos, rope_local_sin) = rope_tables(
            cfg.head_dim,
            cfg.rope_local_base_freq,
            cfg.head_dim / 2,
            MAX_SEQUENCE_TOKENS,
        );
        let (rope_global_cos, rope_global_sin) = rope_tables(
            cfg.global_head_dim,
            cfg.rope_theta,
            // Proportional partial rotary: only the first
            // `partial_rotary_factor * head_dim / 2` pair frequencies
            // rotate; the rest carry identity rotation.
            (cfg.partial_rotary_factor * cfg.global_head_dim as f64 / 2.0) as usize,
            MAX_SEQUENCE_TOKENS,
        );
        let retains_donor_kv = (0..cfg.num_hidden_layers)
            .map(|layer| {
                (layer + 1..cfg.num_hidden_layers)
                    .any(|consumer| cfg.kv_donor(consumer) == Some(layer))
            })
            .collect();
        Ok(Self {
            tensors,
            token_embd_rows,
            ple_rows,
            config: cfg.clone(),
            rope_local_cos,
            rope_local_sin,
            rope_global_cos,
            rope_global_sin,
            retains_donor_kv,
            scratch_budget,
            runtime: Arc::clone(runtime),
        })
    }

    /// Forward the prompt once and return the final-position letter logits
    /// (after softcapping, in candidate order). The answer slot is the last
    /// prompt position; nothing is sampled and no continuation tokens exist.
    pub(super) fn letter_logits_controlled(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f64>, FamilyError> {
        if prompt_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        if letter_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "letter token ids are empty".to_owned(),
            ));
        }
        let seq = prompt_ids.len();
        if seq > MAX_SEQUENCE_TOKENS {
            return Err(FamilyError::InvalidInput(format!(
                "sequence length {seq} outside (0, {MAX_SEQUENCE_TOKENS}]"
            )));
        }
        if let Some(budget) = self.scratch_budget {
            let estimate = self.config.estimated_peak_scratch_bytes(seq);
            if estimate > budget {
                return Err(FamilyError::InvalidInput(format!(
                    "estimated peak scratch of a {seq}-token forward is {estimate} bytes, \
                     over the configured {budget}-byte budget; refusing to allocate \
                     (lower the prompt or raise {})",
                    super::super::SCRATCH_BUDGET_ENV
                )));
            }
        }
        control.check()?;
        // A single MLX forward is uninterruptible mid-graph, but the
        // per-layer checks inside it abort a long pass early when the
        // caller disconnects or the deadline elapses.
        let slot = self
            .runtime
            .execute(|| self.final_normed_slot(prompt_ids, control))??;
        if slot.len() != self.config.hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "backbone output width does not match the pinned geometry: {} values, \
                 expected {}",
                slot.len(),
                self.config.hidden_size
            )));
        }
        if slot.iter().any(|value| !value.is_finite()) {
            return Err(FamilyError::Numerical(
                "backbone produced a non-finite hidden state".to_owned(),
            ));
        }
        control.check()?;
        // The tied head: the letter rows stream from the verified GGUF
        // (q8_0 blocks, byte-identical to candle's dequantization), and the
        // dots run host-side in f64 — mathematically the indexed values of
        // the oracle's full-vocab logits row (see the module docs).
        let rows = self
            .token_embd_rows
            .rows(letter_ids)?
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(FamilyError::Candle)?;
        if rows.len() != letter_ids.len() * self.config.hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "letter embedding rows have unexpected geometry: {} values for {} letters",
                rows.len(),
                letter_ids.len()
            )));
        }
        let logits: Vec<f64> = rows
            .chunks_exact(self.config.hidden_size)
            .map(|row| {
                row.iter()
                    .zip(&slot)
                    .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                    .sum::<f64>()
            })
            .collect();
        // Elementwise tanh softcapping, mirroring the oracle's
        // `logits.affine(1/c).tanh().affine(c)`; it commutes with the row
        // selection above.
        let capped: Vec<f64> = match self.config.final_logit_softcapping {
            None => logits,
            Some(cap) => logits
                .into_iter()
                .map(|logit| cap * (logit / cap).tanh())
                .collect(),
        };
        if capped.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "letter readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(capped)
    }

    /// Full-sequence forward returning the final position's post-final-norm
    /// hidden row (the answer slot, before the tied-head readout).
    ///
    /// Must run inside [`MlxRuntime::execute`].
    fn final_normed_slot(
        &self,
        input_ids: &[u32],
        control: &FamilyControl,
    ) -> Result<Vec<f32>, FamilyError> {
        let cfg = &self.config;
        let seq = input_ids.len();
        let hidden_size = cfg.hidden_size;
        let values = self
            .token_embd_rows
            .rows(input_ids)?
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(FamilyError::Candle)?;
        if values.len() != seq * hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "token embedding rows have unexpected geometry: {} values for {} tokens",
                values.len(),
                seq
            )));
        }
        // Input embeddings scaled by `sqrt(hidden_size)`.
        let mut hidden = Array::from_slice(&values, &[seq as i32, hidden_size as i32])
            .multiply(Array::from_f32((hidden_size as f64).sqrt() as f32))
            .map_err(mlx_error("embed scale"))?;
        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &hidden)?;
        // Post-RoPE K/V of the layers that donate to KV-shared layers. Only
        // layers a later consumer actually reads retain their K/V.
        let mut donors: Vec<Option<(Array, Array)>> = vec![None; cfg.num_hidden_layers];
        for layer in 0..cfg.num_hidden_layers {
            control.check()?;
            let prefix = format!("layers.{layer}");
            let residual = hidden.clone();
            let normed = rms_norm(
                &hidden,
                self.get(&format!("{prefix}.input_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )?;
            let donor_kv = cfg.kv_donor(layer).map(|donor| {
                donors[donor]
                    .clone()
                    .expect("donor K/V computed before its consumer")
            });
            let (attention, own_kv) = self.attention(layer, &normed, donor_kv.as_ref())?;
            let mixed = rms_norm(
                &attention,
                self.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )? + residual;
            let residual = mixed.clone();
            let normed = rms_norm(
                &mixed,
                self.get(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )?;
            let gate = mlx_rs::nn::gelu_approximate(
                &self.project(&format!("{prefix}.mlp.gate_proj.weight"), &normed)?,
            )
            .map_err(mlx_error("mlp gelu"))?;
            let up = self.project(&format!("{prefix}.mlp.up_proj.weight"), &normed)?;
            let activated = gate.multiply(&up).map_err(mlx_error("mlp product"))?;
            let ffn = self.project(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            let mut xs = rms_norm(
                &ffn,
                self.get(&format!("{prefix}.post_feedforward_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )? + residual;
            if let (Some(gate_weight), Some(projection), Some(norm), Some(pli)) = (
                self.tensors
                    .get(&format!("{prefix}.per_layer_input_gate.weight")),
                self.tensors
                    .get(&format!("{prefix}.per_layer_projection.weight")),
                self.tensors
                    .get(&format!("{prefix}.post_per_layer_input_norm.weight")),
                per_layer_inputs
                    .as_ref()
                    .and_then(|inputs| inputs.get(layer)),
            ) {
                let gated = mlx_rs::nn::gelu_approximate(
                    &xs.matmul(&gate_weight.transpose().map_err(mlx_error("gate t"))?)
                        .map_err(mlx_error("gate matmul"))?,
                )
                .map_err(mlx_error("ple gelu"))?;
                let gated = gated.multiply(pli).map_err(mlx_error("ple product"))?;
                let projected = rms_norm(
                    &gated
                        .matmul(&projection.transpose().map_err(mlx_error("projection t"))?)
                        .map_err(mlx_error("projection matmul"))?,
                    norm,
                    cfg.rms_norm_eps,
                )?;
                xs = projected.add(&xs).map_err(mlx_error("ple residual"))?;
            }
            if let Some(scale) = self
                .tensors
                .get(&format!("{prefix}.layer_output_scale.weight"))
            {
                xs = xs.multiply(scale).map_err(mlx_error("layer scale"))?;
            }
            if let Some((key, value)) = own_kv {
                if self.retains_donor_kv[layer] {
                    donors[layer] = Some((key, value));
                }
            }
            hidden = xs;
            // Materialize per layer so no forward graph spans more than one
            // block.
            hidden.eval().map_err(mlx_error("layer materialization"))?;
        }
        control.check()?;
        // The answer slot: the last prompt position, final-normed.
        let indices = [(seq - 1) as u32];
        let last = hidden
            .take_axis(Array::from_slice(&indices, &[1]), 0)
            .map_err(mlx_error("slot slice"))?;
        let normed = rms_norm(&last, self.get("norm.weight")?, cfg.rms_norm_eps)?;
        let slot = normed
            .reshape(&[hidden_size as i32])
            .map_err(mlx_error("slot flatten"))?;
        slot.eval().map_err(mlx_error("slot materialization"))?;
        slot.to_vec_cast::<f32>()
            .map_err(|error| FamilyError::Mlx(format!("slot read: {error}")))
    }

    /// Compute the shared per-layer inputs, mirroring the oracle's
    /// `compute_per_layer_inputs`:
    /// `(rms_norm(per_layer_model_projection(xs) / sqrt(hidden)) +
    /// ple_embedding(ids) * sqrt(ple_dim)) / sqrt(2)`, one `(seq, ple_dim)`
    /// slice per layer.
    ///
    /// Must run inside [`MlxRuntime::execute`].
    fn compute_per_layer_inputs(
        &self,
        input_ids: &[u32],
        hidden: &Array,
    ) -> Result<Option<Vec<Array>>, FamilyError> {
        let ple_dim = self.config.per_layer_input_dim;
        if ple_dim == 0 {
            return Ok(None);
        }
        let seq = input_ids.len();
        let layer_count = self.config.num_hidden_layers;
        let values = self
            .ple_rows
            .rows(input_ids)?
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(FamilyError::Candle)?;
        if values.len() != seq * layer_count * ple_dim {
            return Err(FamilyError::InvalidInput(format!(
                "per-layer embedding rows have unexpected geometry: {} values for \
                 {} tokens x {layer_count} layers x {ple_dim}",
                values.len(),
                seq
            )));
        }
        let embedded =
            Array::from_slice(&values, &[seq as i32, layer_count as i32, ple_dim as i32])
                .multiply(Array::from_f32((ple_dim as f64).sqrt() as f32))
                .map_err(mlx_error("ple embed scale"))?;
        let projected = self
            .project("per_layer_model_projection.weight", hidden)?
            .multiply(Array::from_f32(
                (1.0 / (self.config.hidden_size as f64).sqrt()) as f32,
            ))
            .map_err(mlx_error("ple projection scale"))?
            .reshape(&[seq as i32, layer_count as i32, ple_dim as i32])
            .map_err(mlx_error("ple projection reshape"))?;
        let projected = rms_norm(
            &projected,
            self.get("per_layer_projection_norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let combined = projected
            .add(&embedded)
            .map_err(mlx_error("ple combine"))?
            .multiply(Array::from_f32(std::f64::consts::FRAC_1_SQRT_2 as f32))
            .map_err(mlx_error("ple combine scale"))?;
        combined.eval().map_err(mlx_error("ple materialization"))?;
        // (seq, layers, ple_dim) -> one (seq, ple_dim) slice per layer.
        let transposed = combined
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("ple transpose"))?;
        let slices = transposed
            .split_equal(layer_count as i32, 0)
            .map_err(mlx_error("ple split"))?;
        let per_layer = slices
            .iter()
            .map(|slice| {
                slice
                    .reshape(&[seq as i32, ple_dim as i32])
                    .map_err(mlx_error("ple slice reshape"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(per_layer))
    }

    fn get(&self, name: &str) -> Result<&Array, FamilyError> {
        self.tensors.get(name).ok_or_else(|| {
            FamilyError::InvalidInput(format!("tensor `{name}` missing from the checkpoint"))
        })
    }

    /// `x @ w.T` over a dense FP32 projection.
    fn project(&self, name: &str, x: &Array) -> Result<Array, FamilyError> {
        let transposed = self.get(name)?.transpose().map_err(mlx_error("weight t"))?;
        x.matmul(&transposed).map_err(mlx_error("linear matmul"))
    }

    /// Gemma 4 hybrid attention for one layer, mirroring the oracle's
    /// `Attention::forward`: softmax scale `1.0`, per-head Q/K RMSNorm,
    /// value RMS normalization without a learned weight, RoPE before the
    /// banded softmax, HF `repeat_kv` GQA expansion, and — for KV-shared
    /// layers — the donor's post-RoPE K/V instead of own projections.
    /// Returns the attention output and, for layers with their own
    /// projections, the post-RoPE K/V for later KV-shared consumers.
    fn attention(
        &self,
        layer: usize,
        normed: &Array,
        donor_kv: Option<&(Array, Array)>,
    ) -> Result<(Array, Option<(Array, Array)>), FamilyError> {
        let cfg = &self.config;
        let seq = normed.shape()[0] as usize;
        let is_sliding = cfg.is_sliding(layer);
        let head_dim = if is_sliding {
            cfg.head_dim
        } else {
            cfg.global_head_dim
        };
        let prefix = format!("layers.{layer}.self_attn");
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let rep = heads / kv_heads;
        let query = self
            .project(&format!("{prefix}.q_proj.weight"), normed)?
            .reshape(&[seq as i32, heads as i32, head_dim as i32])
            .map_err(mlx_error("q reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("q heads"))?;
        let query = norm_last_axis(
            &query,
            self.get(&format!("{prefix}.q_norm.weight"))?,
            cfg.rms_norm_eps,
        )?;
        let rope_cos = if is_sliding {
            &self.rope_local_cos
        } else {
            &self.rope_global_cos
        };
        let rope_sin = if is_sliding {
            &self.rope_local_sin
        } else {
            &self.rope_global_sin
        };
        let query = apply_rotary(&query, rope_cos, rope_sin)?;
        let (keys, values, own_kv) = match donor_kv {
            None => {
                let keys = self
                    .project(&format!("{prefix}.k_proj.weight"), normed)?
                    .reshape(&[seq as i32, kv_heads as i32, head_dim as i32])
                    .map_err(mlx_error("k reshape"))?
                    .transpose_axes(&[1, 0, 2])
                    .map_err(mlx_error("k heads"))?;
                let keys = norm_last_axis(
                    &keys,
                    self.get(&format!("{prefix}.k_norm.weight"))?,
                    cfg.rms_norm_eps,
                )?;
                // Values receive a pure RMS normalization without a learned
                // weight.
                let values = mlx_rs::fast::rms_norm(
                    &self
                        .project(&format!("{prefix}.v_proj.weight"), normed)?
                        .reshape(&[seq as i32, kv_heads as i32, head_dim as i32])
                        .map_err(mlx_error("v reshape"))?
                        .transpose_axes(&[1, 0, 2])
                        .map_err(mlx_error("v heads"))?,
                    None,
                    cfg.rms_norm_eps as f32,
                )
                .map_err(mlx_error("value norm"))?;
                let keys = apply_rotary(&keys, rope_cos, rope_sin)?;
                let own_kv = self.retains_donor_kv[layer].then(|| (keys.clone(), values.clone()));
                (keys, values, own_kv)
            }
            Some((donor_key, donor_value)) => (donor_key.clone(), donor_value.clone(), None),
        };
        let keys = repeat_kv(&keys, rep, seq, head_dim)?;
        let values = repeat_kv(&values, rep, seq, head_dim)?;
        let window = is_sliding.then_some(cfg.sliding_window);
        let attended = chunked_attention(&query, &keys, &values, window)?;
        let mixed = attended
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("mix heads"))?
            .reshape(&[seq as i32, (heads * head_dim) as i32])
            .map_err(mlx_error("mix flatten"))?;
        let out = self.project(&format!("{prefix}.o_proj.weight"), &mixed)?;
        Ok((out, own_kv))
    }
}

/// Grouped-query expansion of K/V to query heads, mirroring the oracle's
/// `repeat_kv` (HF interleave: query head `i` pairs with KV head
/// `i / rep`) over `(kv_heads, seq, head_dim)` values.
fn repeat_kv(xs: &Array, n_rep: usize, seq: usize, head_dim: usize) -> Result<Array, FamilyError> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let kv_heads = xs.shape()[0];
    mlx_rs::ops::stack(&(0..n_rep).map(|_| xs.clone()).collect::<Vec<_>>(), 0)
        .map_err(mlx_error("kv repeat"))?
        .transpose_axes(&[1, 0, 2, 3])
        .map_err(mlx_error("kv repeat transpose"))?
        .reshape(&[kv_heads * n_rep as i32, seq as i32, head_dim as i32])
        .map_err(mlx_error("kv repeat reshape"))
}

/// Key span `[key_start, key_end)` that the query block
/// `[query_start, query_end)` can attend to, mirroring the oracle's
/// `block_key_span`: the trailing causal range, narrowed to the last
/// `window` positions on sliding layers.
fn block_key_span(
    query_start: usize,
    query_end: usize,
    sliding_window: Option<usize>,
) -> (usize, usize) {
    match sliding_window {
        Some(window) => (query_start.saturating_sub(window), query_end),
        None => (0, query_end),
    }
}

/// Compact block-local attention mask shaped `(1, 1, query_len, key_len)`:
/// `0` where the query may attend the key and `-inf` where the dense mask
/// would have blocked it (causal `key <= query`, plus
/// `key + window >= query` on sliding layers). Every row keeps at least one
/// `0` (the query's own position always lies inside the span), so the
/// softmax never sees an all-masked row.
fn attention_block_mask(
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
    sliding_window: Option<usize>,
) -> Array {
    let mut data = Vec::with_capacity(query_len * key_len);
    for qi in 0..query_len {
        let query = query_start + qi;
        for ki in 0..key_len {
            let key = key_start + ki;
            let allowed = key <= query && sliding_window.is_none_or(|window| key + window >= query);
            data.push(if allowed { 0f32 } else { f32::NEG_INFINITY });
        }
    }
    Array::from_slice(&data, &[1, 1, query_len as i32, key_len as i32])
}

/// Banded attention over query chunks of [`ATTENTION_QUERY_CHUNK`],
/// mirroring the oracle's `chunked_attention`: each block of queries
/// multiplies only the keys it can attend to and applies a compact
/// block-local mask before the explicitly max-subtracted softmax. Masked
/// keys contribute `exp(-inf) = 0`, so this is the dense masked attention's
/// math with linear-in-seq scratch.
fn chunked_attention(
    query: &Array,
    keys: &Array,
    values: &Array,
    sliding_window: Option<usize>,
) -> Result<Array, FamilyError> {
    let seq = query.shape()[1] as usize;
    let mut blocks = Vec::with_capacity(seq.div_ceil(ATTENTION_QUERY_CHUNK));
    for query_start in (0..seq).step_by(ATTENTION_QUERY_CHUNK) {
        let query_end = (query_start + ATTENTION_QUERY_CHUNK).min(seq);
        let (key_start, key_end) = block_key_span(query_start, query_end, sliding_window);
        let key_len = key_end - key_start;
        let query_indices: Vec<u32> = (query_start as u32..query_end as u32).collect();
        let key_indices: Vec<u32> = (key_start as u32..key_end as u32).collect();
        let q_block = query
            .take_axis(
                Array::from_slice(&query_indices, &[query_indices.len() as i32]),
                1,
            )
            .map_err(mlx_error("q block"))?;
        let k_block = keys
            .take_axis(
                Array::from_slice(&key_indices, &[key_indices.len() as i32]),
                1,
            )
            .map_err(mlx_error("k block"))?;
        let v_block = values
            .take_axis(
                Array::from_slice(&key_indices, &[key_indices.len() as i32]),
                1,
            )
            .map_err(mlx_error("v block"))?;
        // Softmax scale is the Gemma 4 constant `1.0`: the learned Q/K norms
        // carry the magnitude and both reference runtimes pin the scale.
        let scores = q_block
            .matmul(
                &k_block
                    .transpose_axes(&[0, 2, 1])
                    .map_err(mlx_error("k block t"))?,
            )
            .map_err(mlx_error("scores matmul"))?;
        let scores = scores
            + attention_block_mask(
                query_start,
                query_end - query_start,
                key_start,
                key_len,
                sliding_window,
            );
        // Explicit max-subtracted softmax, matching the oracle's numerics.
        let maximum = scores
            .max_axis(-1, true)
            .map_err(mlx_error("softmax max"))?;
        let exponentiated = scores
            .sub(&maximum)
            .exp()
            .map_err(mlx_error("softmax exp"))?;
        let denominator = exponentiated
            .sum_axis(-1, true)
            .map_err(mlx_error("softmax sum"))?;
        let weights = exponentiated.div(&denominator);
        let mixed = weights
            .matmul(&v_block)
            .map_err(mlx_error("mix block matmul"))?;
        // MLX builds a lazy graph. Evaluate before building the next block
        // so its score, mask, and softmax buffers do not accumulate.
        mixed.eval().map_err(mlx_error("mix block eval"))?;
        blocks.push(mixed);
    }
    mlx_rs::ops::concatenate(&blocks, 1).map_err(mlx_error("block concat"))
}

/// RMSNorm applying the stored weight directly over the last axis (Gemma 4
/// dropped the historical Gemma `weight + 1` shift), the same math as the
/// oracle's `RmsNorm`.
fn rms_norm(x: &Array, weight: &Array, eps: f64) -> Result<Array, FamilyError> {
    mlx_rs::fast::rms_norm(x, Some(weight), eps as f32).map_err(mlx_error("rms norm"))
}

/// RMSNorm over the last axis with arbitrary leading dimensions (the
/// oracle's `(b, heads, seq, head_dim)` Q/K norms flatten the same way).
fn norm_last_axis(values: &Array, weight: &Array, eps: f64) -> Result<Array, FamilyError> {
    let dims = values.shape().to_vec();
    let rows: i32 = dims.iter().take(dims.len() - 1).product();
    let width = *dims.last().expect("non-empty shape");
    mlx_rs::fast::rms_norm(
        &values
            .clone()
            .reshape(&[rows, width])
            .map_err(mlx_error("norm reshape"))?,
        Some(weight),
        eps as f32,
    )
    .map_err(mlx_error("head norm"))?
    .reshape(&dims)
    .map_err(mlx_error("norm restore"))
}

/// Host FP32 RoPE tables at `(max_seq, head_dim / 2)`, the same math as the
/// oracle's `RotaryEmbedding`/`ProportionalRotaryEmbedding`: `inv_freq[i] =
/// 1 / base^(2i / head_dim)` for the first `rotated_pairs` pair frequencies
/// and `0` (identity rotation) beyond — full-dimension tables pass
/// `rotated_pairs = head_dim / 2`.
fn rope_tables(head_dim: usize, base: f64, rotated_pairs: usize, max_seq: usize) -> (Array, Array) {
    let half = head_dim / 2;
    let mut cos_values = Vec::with_capacity(max_seq * half);
    let mut sin_values = Vec::with_capacity(max_seq * half);
    for position in 0..max_seq {
        for index in 0..half {
            let inv_freq = if index < rotated_pairs {
                1.0 / base.powf((2 * index) as f64 / head_dim as f64)
            } else {
                0.0
            };
            let angle = position as f32 * inv_freq as f32;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }
    let shape = [max_seq as i32, half as i32];
    let cos = Array::from_slice(&cos_values, &shape);
    let sin = Array::from_slice(&sin_values, &shape);
    cos.eval().expect("rope cos materializes");
    sin.eval().expect("rope sin materializes");
    (cos, sin)
}

/// Half-split (NeoX) rotary embedding over `(heads, seq, head_dim)` values
/// at absolute positions — the exact pairing candle's `rotary_emb::rope`
/// kernel applies (`out1 = x1*cos - x2*sin`, `out2 = x2*cos + x1*sin` over
/// the two halves of each head), which the proportional tables extend with
/// identity rotation on the unrotated pairs.
fn apply_rotary(values: &Array, rope_cos: &Array, rope_sin: &Array) -> Result<Array, FamilyError> {
    let dims = values.shape().to_vec();
    let seq = dims[1];
    let indices: Vec<u32> = (0..seq as u32).collect();
    let idx = Array::from_slice(&indices, &[seq]);
    let cos = rope_cos
        .take_axis(&idx, 0)
        .map_err(mlx_error("rope slice"))?;
    let sin = rope_sin
        .take_axis(&idx, 0)
        .map_err(mlx_error("rope slice"))?;
    let half = dims[2] / 2;
    let split = values.split_equal(2, -1).map_err(mlx_error("rope split"))?;
    let (x1, x2) = (&split[0], &split[1]);
    let broadcast_shape = [dims[0], seq, half];
    let cos = mlx_rs::ops::broadcast_to(
        &cos.reshape(&[seq, half])
            .map_err(mlx_error("rope cos reshape"))?,
        &broadcast_shape,
    )
    .map_err(mlx_error("rope cos broadcast"))?;
    let sin = mlx_rs::ops::broadcast_to(
        &sin.reshape(&[seq, half])
            .map_err(mlx_error("rope sin reshape"))?,
        &broadcast_shape,
    )
    .map_err(mlx_error("rope sin broadcast"))?;
    let out1 = x1
        .multiply(&cos)
        .map_err(mlx_error("rope out1 mul"))?
        .subtract(&x2.multiply(&sin).map_err(mlx_error("rope out2 mul"))?)
        .map_err(mlx_error("rope out1"))?;
    let out2 = x2
        .multiply(&cos)
        .map_err(mlx_error("rope out2 mul"))?
        .add(&x1.multiply(&sin).map_err(mlx_error("rope out1 mul"))?)
        .map_err(mlx_error("rope out2"))?;
    mlx_rs::ops::concatenate(&[out1, out2], -1).map_err(mlx_error("rope concat"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_mlx(f: impl FnOnce()) {
        let runtime =
            MlxRuntime::new(crate::qwen35::mlx::MlxRuntimeConfig::default()).expect("MLX runtime");
        runtime.execute(f).expect("MLX execution");
    }

    #[test]
    fn fp32_weight_estimate_matches_the_pinned_footprint() {
        let cfg = Gemma4TextConfig::winnow_e4b();
        let bytes = estimated_fp32_weight_bytes(&cfg);
        // The q8_0 compute weights are ~4.9 GB (34 bytes per 32 elements);
        // full FP32 dequantization roughly triples the resident compute
        // weights. Either bound failing means the estimator drifted from
        // the load path's tensor list.
        assert!(
            ((12u64 << 30)..(20u64 << 30)).contains(&bytes),
            "estimate {bytes} outside the known pinned footprint"
        );
        // Dropping KV sharing adds the shared layers' dead K/V back into
        // the estimate (the load path stops skipping them).
        let without_shared = estimated_fp32_weight_bytes(&Gemma4TextConfig {
            shared_kv_layers: 0,
            ..Gemma4TextConfig::winnow_e4b()
        });
        assert!(
            without_shared > bytes,
            "dropping KV sharing must add the shared layers' dead K/V to the estimate"
        );
    }

    #[test]
    fn block_mask_matches_the_dense_predicate() {
        on_mlx(|| {
            // A middle block of a sliding layer: query range [256, 512), key
            // range [192, 512) — causal corner plus windowed head, the general
            // case (mirrors the oracle's mask test).
            let mask = attention_block_mask(256, 256, 192, 320, Some(64));
            let values = mask.to_vec_cast::<f32>().expect("mask read");
            assert_eq!(values.len(), 256 * 320);
            for (qi, row) in values.as_chunks::<320>().0.iter().enumerate() {
                let query = 256 + qi;
                for (ki, &value) in row.iter().enumerate() {
                    let key = 192 + ki;
                    let allowed = key <= query && key + 64 >= query;
                    assert_eq!(value == 0f32, allowed, "query {query} key {key}");
                    assert_eq!(value == f32::NEG_INFINITY, !allowed);
                }
                assert!(row.contains(&0f32));
            }
            // Full-attention layers keep the whole causal prefix.
            let mask = attention_block_mask(256, 256, 0, 512, None);
            let values = mask.to_vec_cast::<f32>().expect("mask read");
            for (qi, row) in values.as_chunks::<512>().0.iter().enumerate() {
                let query = 256 + qi;
                for (ki, &value) in row.iter().enumerate() {
                    assert_eq!(value == 0f32, ki <= query, "query {query} key {ki}");
                }
            }
        });
    }

    #[test]
    fn key_span_bounds_the_attention_band() {
        for start in (0..600).step_by(ATTENTION_QUERY_CHUNK) {
            let end = (start + ATTENTION_QUERY_CHUNK).min(600);
            let (key_start, key_end) = block_key_span(start, end, Some(64));
            assert_eq!(key_start, start.saturating_sub(64));
            assert!(key_end - key_start <= ATTENTION_QUERY_CHUNK + 64);
            let (key_start, key_end) = block_key_span(start, end, None);
            assert_eq!((key_start, key_end), (0, end));
        }
    }

    #[test]
    fn rope_tables_extend_identity_rotation_beyond_the_rotated_pairs() {
        on_mlx(|| {
            // The proportional partial tables: the first quarter of the pairs
            // rotates at base 1e6, the rest stay at angle 0 (cos 1, sin 0).
            let head_dim = 512usize;
            let rotated_pairs = head_dim / 8;
            let (cos, sin) = rope_tables(head_dim, 1_000_000.0, rotated_pairs, 8);
            let cos = cos.to_vec_cast::<f32>().expect("cos read");
            let sin = sin.to_vec_cast::<f32>().expect("sin read");
            for (row, chunk) in cos.chunks_exact(head_dim / 2).enumerate() {
                // At position 0 every angle is 0, so the rotated pairs also sit
                // at cos 1; a later position must move them.
                let rotated = &chunk[..rotated_pairs];
                if row > 0 {
                    assert!(
                        rotated.iter().any(|v| *v != 1.0),
                        "row {row}: rotated pairs must leave identity"
                    );
                }
                assert!(
                    chunk[rotated_pairs..].iter().all(|v| *v == 1.0),
                    "row {row}: unrotated cos must stay 1"
                );
            }
            for chunk in sin.chunks_exact(head_dim / 2) {
                assert!(
                    chunk[rotated_pairs..].iter().all(|v| *v == 0.0),
                    "unrotated sin must stay 0"
                );
            }
            // The full-dimension local table rotates every pair.
            let (cos, sin) = rope_tables(256, 10_000.0, 128, 8);
            let cos = cos.to_vec_cast::<f32>().expect("cos read");
            let sin = sin.to_vec_cast::<f32>().expect("sin read");
            assert!(cos.iter().any(|v| *v != 1.0));
            assert!(sin.iter().any(|v| *v != 0.0));
        });
    }
}
