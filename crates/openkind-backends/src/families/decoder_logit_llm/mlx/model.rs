//! MLX/Metal execution of the pinned GGUF (q8_0) decoder checkpoint
//! (feature `mlx`, macOS arm64).
//!
//! The pinned GGUF streams in through candle's GGUF reader exactly like the
//! CPU oracle's binding; every quantized tensor is then dequantized to the
//! exact FP32 values candle's block arithmetic produces and loaded into
//! FP32 MLX arrays. llama.cpp stores 2-D weight matrices as `(in, out)` and
//! candle reverses the GGUF dimension order at read, so the dequantized
//! tensors are already in the HF `(out, in)` row-major layout the dense
//! forward consumes — the same layout the unquantized safetensors families
//! load (candle's `QMatMul` applies the transpose inside its forward).
//!
//! This module deliberately duplicates the shared dense body's forward
//! instead of reusing it: the GGUF container (not safetensors shards) needs
//! its own load path, and the pinned quantized checkpoint carries q/k/v
//! attention biases that the CPU quantized runner adds unconditionally —
//! a projection shape the shared safetensors-oriented loader does not
//! express.
//!
//! Arithmetic identity: the CPU oracle multiplies q8_0 blocks inside
//! candle's quantized kernels (f16 block scales), while this backend
//! multiplies the fully dequantized FP32 weights — a different arithmetic
//! identity whose parity gate is the workspace 0.005 probability budget,
//! named by the backend id (`decoder-logit-llm/mlx-fp32`). The letter
//! readout mirrors the CPU: the tied head (the GGUF carries no separate
//! `output.weight`; its presence fails the load) makes the full-vocab
//! logits row the oracle indexes exactly `hidden · embedding.T`, so the
//! per-letter dots against host-resident dequantized embedding rows
//! reproduce the indexed values. The embedding table is dequantized once at
//! load and held on the host (151936 x 896 FP32, ~545 MB), replacing the
//! CPU path's per-row reads; nothing is staged in temporary storage.
//!
//! Every MLX operation — model load and per-request forward — runs through
//! [`MlxRuntime::execute`] on the process-wide serialized GPU stream, per
//! the workspace MLX discipline.

use std::collections::BTreeMap;
use std::sync::Arc;

use candle_core::quantized::gguf_file as gguf;
use candle_core::Device;
use mlx_rs::Array;

use crate::families::qwen35_survey_mlx::mlx_error;
use crate::families::qwen_dense_mlx::{causal_attention, DenseQwenConfig};
use crate::families::support::FamilyError;
use crate::qwen35::mlx::MlxRuntime;

use super::super::model::VerifiedArtifacts;
use super::super::pinned_metadata;

/// Arithmetic/device identity of the family MLX execution path: the pinned
/// q8_0 checkpoint dequantized to exact FP32 arrays computed on the Metal
/// GPU.
pub const MLX_EXECUTION_ARITHMETIC_ID: &str = "mlx-gpu-fp32-decoder-logit-llm";

/// The pinned GGUF model on MLX: dequantized decoder tensors plus the
/// host-resident tied embedding table.
pub(super) struct DecoderLlmMlxModel {
    /// Decoder tensors keyed by their HF suffix (`layers.N.<rest>`,
    /// `norm.weight`, and the q/k/v attention biases).
    tensors: BTreeMap<String, Array>,
    /// Dequantized `(vocab, hidden)` row-major embedding table, retained on
    /// the host for both the forward's embedding lookup and the letter
    /// readout.
    embedding: Vec<f32>,
    vocab_size: usize,
    config: DenseQwenConfig,
    /// Precomputed RoPE tables: `cos`/`sin` at `(max_sequence_tokens,
    /// head_dim / 2)`.
    rope_cos: Array,
    rope_sin: Array,
    runtime: Arc<MlxRuntime>,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared qwen35 MLX backbone).
unsafe impl Sync for DecoderLlmMlxModel {}
unsafe impl Send for DecoderLlmMlxModel {}

/// Insert one suffix-keyed array, failing closed on duplicates.
fn insert_tensor(
    tensors: &mut BTreeMap<String, Array>,
    suffix: String,
    array: Array,
) -> Result<(), FamilyError> {
    if tensors.insert(suffix, array).is_some() {
        Err(FamilyError::InvalidInput(
            "duplicate decoder tensor in the checkpoint".to_owned(),
        ))
    } else {
        Ok(())
    }
}

impl DecoderLlmMlxModel {
    /// Load the digest-verified GGUF checkpoint, dequantize it to FP32, and
    /// build the MLX arrays. Must run inside [`MlxRuntime::execute`].
    pub(super) fn load(
        artifacts: &VerifiedArtifacts,
        runtime: &Arc<MlxRuntime>,
    ) -> Result<Self, FamilyError> {
        let file =
            std::fs::File::open(&artifacts.checkpoint).map_err(|source| FamilyError::Io {
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
            let actual = metadata_text(value);
            if actual != expected {
                return Err(FamilyError::contract(key, expected, actual));
            }
        }
        // The CPU quantized runner reads these metadata values (the epsilon
        // is required there, the rope base defaults to 10_000 when absent);
        // this path enforces the same contract so a drifted checkpoint
        // fails closed instead of silently changing answers.
        let hidden_size = metadata_u32(&content, "qwen2.embedding_length")? as usize;
        let num_hidden_layers = metadata_u32(&content, "qwen2.block_count")? as usize;
        let heads = metadata_u32(&content, "qwen2.attention.head_count")? as usize;
        let kv_heads = metadata_u32(&content, "qwen2.attention.head_count_kv")? as usize;
        if heads == 0 || !hidden_size.is_multiple_of(heads) {
            return Err(FamilyError::contract(
                "qwen2.attention.head_count",
                format!("a divisor of embedding length {hidden_size}"),
                heads.to_string(),
            ));
        }
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(FamilyError::contract(
                "qwen2.attention.head_count_kv",
                format!("a divisor of head count {heads}"),
                kv_heads.to_string(),
            ));
        }
        let head_dim = hidden_size / heads;
        let kv_dim = kv_heads * head_dim;
        let rms_norm_eps = metadata_f32(&content, "qwen2.attention.layer_norm_rms_epsilon")? as f64;
        // llama.cpp stores the profile's rope theta; the CPU runner defaults
        // to 10_000 when the key is absent, and this path mirrors that
        // fallback so the executed math stays the oracle's.
        let rope_theta = match content.metadata.get("qwen2.rope.freq_base") {
            Some(value) => metadata_f64(value)?,
            None => 10_000.0,
        };
        // The CPU runner requires the context length (its rope table is
        // built to it); require the key here for the same contract.
        metadata_u32(&content, "qwen2.context_length")?;
        // The tied-head contract: a separate `output.weight` would make the
        // letter dots diverge from the CPU readout, so its presence fails
        // the load instead of silently changing answers.
        if content.tensor_infos.contains_key("output.weight") {
            return Err(FamilyError::ContractMismatch {
                field: "gguf.tied_head",
                expected: "no separate `output.weight` (tied embeddings)".to_owned(),
                actual: "`output.weight` present".to_owned(),
            });
        }

        let mut tensors: BTreeMap<String, Array> = BTreeMap::new();

        // Dequantized 2-D weights arrive (out, in): llama.cpp stores
        // `(in, out)` and candle reverses the GGUF dimension order at read,
        // the same layout the unquantized safetensors families load.
        let (embedding, vocab_size) = {
            let (values, shape) = dequantized(&mut reader, &content, "token_embd.weight")?;
            if shape.len() != 2 || shape[1] != hidden_size || shape[0] == 0 {
                return Err(FamilyError::contract(
                    "token_embd.weight",
                    format!("[vocab, {hidden_size}]"),
                    format!("{shape:?}"),
                ));
            }
            (values, shape[0])
        };
        let (norm_values, norm_shape) = dequantized(&mut reader, &content, "output_norm.weight")?;
        if norm_shape != vec![hidden_size] {
            return Err(FamilyError::contract(
                "output_norm.weight",
                format!("[{hidden_size}]"),
                format!("{norm_shape:?}"),
            ));
        }
        insert_tensor(
            &mut tensors,
            "norm.weight".to_owned(),
            Array::from_slice(&norm_values, &[hidden_size as i32]),
        )?;

        let mut intermediate: Option<usize> = None;
        for layer in 0..num_hidden_layers {
            let prefix = format!("blk.{layer}");
            let suffix = format!("layers.{layer}");
            // Norms (1-D, exact width).
            for (gguf_name, name) in [
                ("attn_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight", "post_attention_layernorm.weight"),
            ] {
                let (values, shape) =
                    dequantized(&mut reader, &content, &format!("{prefix}.{gguf_name}"))?;
                if shape != vec![hidden_size] {
                    return Err(FamilyError::InvalidInput(format!(
                        "GGUF tensor `{prefix}.{gguf_name}` has shape {shape:?}, expected \
                         [{hidden_size}]"
                    )));
                }
                insert_tensor(
                    &mut tensors,
                    format!("{suffix}.{name}"),
                    Array::from_slice(&values, &[hidden_size as i32]),
                )?;
            }
            // Attention projections: (out, in) after the dimension reversal;
            // the non-square k/v shapes pin the orientation.
            for (gguf_name, name, out) in [
                ("attn_q.weight", "self_attn.q_proj.weight", hidden_size),
                ("attn_k.weight", "self_attn.k_proj.weight", kv_dim),
                ("attn_v.weight", "self_attn.v_proj.weight", kv_dim),
                ("attn_output.weight", "self_attn.o_proj.weight", hidden_size),
            ] {
                let (values, shape) =
                    dequantized(&mut reader, &content, &format!("{prefix}.{gguf_name}"))?;
                if shape != vec![out, hidden_size] {
                    return Err(FamilyError::InvalidInput(format!(
                        "GGUF tensor `{prefix}.{gguf_name}` has shape {shape:?}, expected \
                         [{out}, {hidden_size}]"
                    )));
                }
                insert_tensor(
                    &mut tensors,
                    format!("{suffix}.{name}"),
                    Array::from_slice(&values, &[out as i32, hidden_size as i32]),
                )?;
            }
            // The q/k/v biases the CPU quantized runner adds unconditionally
            // (required keys).
            for (gguf_name, name, out) in [
                ("attn_q.bias", "self_attn.q_proj.bias", hidden_size),
                ("attn_k.bias", "self_attn.k_proj.bias", kv_dim),
                ("attn_v.bias", "self_attn.v_proj.bias", kv_dim),
            ] {
                let (values, shape) =
                    dequantized(&mut reader, &content, &format!("{prefix}.{gguf_name}"))?;
                if shape != vec![out] {
                    return Err(FamilyError::InvalidInput(format!(
                        "GGUF tensor `{prefix}.{gguf_name}` has shape {shape:?}, expected \
                         [{out}]"
                    )));
                }
                insert_tensor(
                    &mut tensors,
                    format!("{suffix}.{name}"),
                    Array::from_slice(&values, &[out as i32]),
                )?;
            }
            // MLP projections. The intermediate width comes from the
            // checkpoint itself; the three tensors must agree with it.
            let (gate_values, gate_shape) =
                dequantized(&mut reader, &content, &format!("{prefix}.ffn_gate.weight"))?;
            if gate_shape.len() != 2 || gate_shape[1] != hidden_size || gate_shape[0] == 0 {
                return Err(FamilyError::InvalidInput(format!(
                    "GGUF tensor `{prefix}.ffn_gate.weight` has shape {gate_shape:?}, expected \
                     [intermediate, {hidden_size}]"
                )));
            }
            let width = gate_shape[0];
            match intermediate {
                None => intermediate = Some(width),
                Some(previous) if previous != width => {
                    return Err(FamilyError::InvalidInput(format!(
                        "GGUF tensor `{prefix}.ffn_gate.weight` has intermediate width \
                         {width}, expected {previous}"
                    )));
                }
                Some(_) => {}
            }
            insert_tensor(
                &mut tensors,
                format!("{suffix}.mlp.gate_proj.weight"),
                Array::from_slice(&gate_values, &[width as i32, hidden_size as i32]),
            )?;
            let (up_values, up_shape) =
                dequantized(&mut reader, &content, &format!("{prefix}.ffn_up.weight"))?;
            if up_shape != vec![width, hidden_size] {
                return Err(FamilyError::InvalidInput(format!(
                    "GGUF tensor `{prefix}.ffn_up.weight` has shape {up_shape:?}, expected \
                     [{width}, {hidden_size}]"
                )));
            }
            insert_tensor(
                &mut tensors,
                format!("{suffix}.mlp.up_proj.weight"),
                Array::from_slice(&up_values, &[width as i32, hidden_size as i32]),
            )?;
            let (down_values, down_shape) =
                dequantized(&mut reader, &content, &format!("{prefix}.ffn_down.weight"))?;
            if down_shape != vec![hidden_size, width] {
                return Err(FamilyError::InvalidInput(format!(
                    "GGUF tensor `{prefix}.ffn_down.weight` has shape {down_shape:?}, expected \
                     [{hidden_size}, {width}]"
                )));
            }
            insert_tensor(
                &mut tensors,
                format!("{suffix}.mlp.down_proj.weight"),
                Array::from_slice(&down_values, &[hidden_size as i32, width as i32]),
            )?;
        }
        let intermediate = intermediate.ok_or_else(|| {
            FamilyError::InvalidInput("checkpoint carries no MLP tensors".to_owned())
        })?;

        let config = DenseQwenConfig {
            vocab_size,
            hidden_size,
            head_dim,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            num_hidden_layers,
            intermediate_size: intermediate,
            rope_theta,
            rms_norm_eps,
            // Qwen2 has no per-head q/k RMSNorm.
            qk_norm: false,
            max_sequence_tokens: super::super::MAX_SEQUENCE_TOKENS,
        };
        let (rope_cos, rope_sin) = rope_tables(&config);
        let model = Self {
            tensors,
            embedding,
            vocab_size,
            config,
            rope_cos,
            rope_sin,
            runtime: Arc::clone(runtime),
        };
        // Materialize every retained array so no unevaluated lazy graph
        // crosses the loading thread's boundary.
        for array in model.tensors.values() {
            array.eval().map_err(mlx_error("weight materialization"))?;
        }
        Ok(model)
    }

    /// Forward the prompt once and return the next-token logits restricted
    /// to `letter_ids`. The answer slot is the final prompt position;
    /// nothing is sampled and no continuation tokens exist.
    ///
    /// The tied-embedding identity: the CPU oracle's readout row is
    /// `hidden @ output.T` with `output` bound to the dequantized
    /// `token_embd.weight`, so indexing that row at a letter id equals a dot
    /// of the same hidden row against that id's embedding row. (The two
    /// paths still differ in kernel arithmetic — the CPU multiplies q8_0
    /// blocks in f16-scaled kernels — which is the declared parity
    /// difference this backend id names.)
    pub(super) fn letter_logits(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<Vec<f64>, FamilyError> {
        if prompt_ids.is_empty() {
            return Err(FamilyError::InvalidInput(
                "prompt token ids are empty".to_owned(),
            ));
        }
        let hidden_size = self.config.hidden_size;
        // Fail closed on out-of-vocabulary letters exactly like the CPU
        // `letter_token_contract` check.
        for token in letter_ids {
            let in_range = usize::try_from(*token)
                .map(|index| index < self.vocab_size)
                .unwrap_or(false);
            if !in_range {
                return Err(FamilyError::ContractMismatch {
                    field: "letter_token_contract",
                    expected: "letter token inside the model vocabulary".to_owned(),
                    actual: format!("letter token id {token} out of range"),
                });
            }
        }
        let rows = self.embed_rows(letter_ids)?;
        if rows.len() != letter_ids.len() * hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "letter embedding rows have unexpected geometry: {} values for {} letters",
                rows.len(),
                letter_ids.len()
            )));
        }
        let hidden = self.runtime.execute(|| self.forward_hidden(prompt_ids))??;
        let token_count = prompt_ids.len();
        if hidden.len() != token_count * hidden_size {
            return Err(FamilyError::InvalidInput(format!(
                "backbone output width does not match the pinned geometry: {} values for \
                 {token_count} tokens",
                hidden.len()
            )));
        }
        // Last prompt position: the answer slot.
        let slot = &hidden[(token_count - 1) * hidden_size..];
        let logits: Vec<f64> = rows
            .chunks_exact(hidden_size)
            .map(|row| {
                row.iter()
                    .zip(slot)
                    .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                    .sum::<f64>()
            })
            .collect();
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(FamilyError::Numerical(
                "letter readout produced a non-finite logit".to_owned(),
            ));
        }
        Ok(logits)
    }

    /// Host-resident tied-embedding rows for `token_ids` from the
    /// dequantized table.
    fn embed_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let hidden_size = self.config.hidden_size;
        let mut rows = Vec::with_capacity(token_ids.len() * hidden_size);
        for token in token_ids {
            let index = usize::try_from(*token).map_err(|_| FamilyError::ContractMismatch {
                field: "letter_token_contract",
                expected: "letter token inside the model vocabulary".to_owned(),
                actual: format!("letter token id {token} out of range"),
            })?;
            if index >= self.vocab_size {
                return Err(FamilyError::ContractMismatch {
                    field: "letter_token_contract",
                    expected: "letter token inside the model vocabulary".to_owned(),
                    actual: format!("letter token id {token} out of range"),
                });
            }
            rows.extend_from_slice(&self.embedding[index * hidden_size..(index + 1) * hidden_size]);
        }
        Ok(rows)
    }

    /// Causal full-sequence forward, returning the final-normed hidden
    /// states as host FP32 row-major `[token_count, hidden]` values.
    ///
    /// Must run inside [`MlxRuntime::execute`].
    fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, FamilyError> {
        let config = self.config;
        let hidden_size = config.hidden_size;
        let token_count = input_ids.len();
        if token_count == 0 {
            return Err(FamilyError::InvalidInput(
                "decoder-llm prompts carry at least one token".to_owned(),
            ));
        }
        let embedded = self.embed_rows(input_ids)?;
        let mut hidden = Array::from_slice(&embedded, &[token_count as i32, hidden_size as i32]);
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            let normalized = rms_norm(
                &hidden,
                self.get(&format!("{prefix}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            let attention = self.attention(&prefix, &normalized, token_count)?;
            let mixed = hidden.clone() + attention;
            let post = rms_norm(
                &mixed,
                self.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            )?;
            let gate = self.project(&format!("{prefix}.mlp.gate_proj.weight"), &post)?;
            let up = self.project(&format!("{prefix}.mlp.up_proj.weight"), &post)?;
            let activated = mlx_rs::nn::silu(&gate).map_err(mlx_error("mlp silu"))? * up;
            let down = self.project(&format!("{prefix}.mlp.down_proj.weight"), &activated)?;
            hidden = mixed + down;
            // Materialize per layer so no forward graph spans more than one
            // block.
            hidden.eval().map_err(mlx_error("layer materialization"))?;
        }
        let final_hidden = rms_norm(&hidden, self.get("norm.weight")?, config.rms_norm_eps)?;
        let flattened = final_hidden
            .reshape(&[(token_count * hidden_size) as i32])
            .map_err(mlx_error("final flatten"))?;
        flattened.eval().map_err(mlx_error("host eval"))?;
        flattened
            .to_vec_cast::<f32>()
            .map_err(|error| FamilyError::Mlx(format!("host read: {error}")))
    }

    fn get(&self, name: &str) -> Result<&Array, FamilyError> {
        self.tensors.get(name).ok_or_else(|| {
            FamilyError::InvalidInput(format!("tensor `{name}` missing from checkpoint"))
        })
    }

    /// `x @ w.T` over a dense FP32 projection.
    fn project(&self, name: &str, x: &Array) -> Result<Array, FamilyError> {
        let transposed = self.get(name)?.transpose().map_err(mlx_error("weight t"))?;
        x.matmul(&transposed).map_err(mlx_error("linear matmul"))
    }

    /// Add the named 1-D bias when the checkpoint carries it (the
    /// quantized runner's q/k/v bias adds; o/mlp have none).
    fn add_bias(&self, name: &str, out: Array) -> Result<Array, FamilyError> {
        match self.tensors.get(name) {
            Some(bias) => Ok(out + bias),
            None => Ok(out),
        }
    }

    /// Causal grouped-query attention with full-dimension NeoX rotary
    /// embedding at absolute positions and the quantized runner's q/k/v
    /// bias adds.
    fn attention(
        &self,
        prefix: &str,
        normalized: &Array,
        token_count: usize,
    ) -> Result<Array, FamilyError> {
        let config = self.config;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let projected_q = self.add_bias(
            &format!("{prefix}.self_attn.q_proj.bias"),
            self.project(&format!("{prefix}.self_attn.q_proj.weight"), normalized)?,
        )?;
        let keys = self.add_bias(
            &format!("{prefix}.self_attn.k_proj.bias"),
            self.project(&format!("{prefix}.self_attn.k_proj.weight"), normalized)?,
        )?;
        let values = self.add_bias(
            &format!("{prefix}.self_attn.v_proj.bias"),
            self.project(&format!("{prefix}.self_attn.v_proj.weight"), normalized)?,
        )?;
        let query = projected_q
            .reshape(&[token_count as i32, heads as i32, head_dim as i32])
            .map_err(mlx_error("q reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("q heads"))?;
        let keys = keys
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("k reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("k heads"))?;
        let values = values
            .reshape(&[token_count as i32, kv_heads as i32, head_dim as i32])
            .map_err(mlx_error("v reshape"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("v heads"))?;

        let query = apply_rotary(&query, &self.rope_cos, &self.rope_sin)?;
        let keys = apply_rotary(&keys, &self.rope_cos, &self.rope_sin)?;
        // HF `repeat_kv` interleave: query head `i` pairs with KV head
        // `i / group`.
        let group = heads / kv_heads;
        let keys_r = mlx_rs::ops::stack(&(0..group).map(|_| keys.clone()).collect::<Vec<_>>(), 0)
            .map_err(mlx_error("k repeat"))?
            .transpose_axes(&[1, 0, 2, 3])
            .map_err(mlx_error("k repeat transpose"))?
            .reshape(&[heads as i32, token_count as i32, head_dim as i32])
            .map_err(mlx_error("k repeat reshape"))?;
        let values_r =
            mlx_rs::ops::stack(&(0..group).map(|_| values.clone()).collect::<Vec<_>>(), 0)
                .map_err(mlx_error("v repeat"))?
                .transpose_axes(&[1, 0, 2, 3])
                .map_err(mlx_error("v repeat transpose"))?
                .reshape(&[heads as i32, token_count as i32, head_dim as i32])
                .map_err(mlx_error("v repeat reshape"))?;

        let mixed = causal_attention(&query, &keys_r, &values_r, head_dim)?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("mix heads"))?
            .reshape(&[token_count as i32, (heads * head_dim) as i32])
            .map_err(mlx_error("mix flatten"))?;
        self.project(&format!("{prefix}.self_attn.o_proj.weight"), &mixed)
    }
}

/// Dequantize one GGUF tensor to its exact FP32 host values and shape
/// (candle's block arithmetic, on the CPU device).
fn dequantized<R: std::io::Seek + std::io::Read>(
    reader: &mut R,
    content: &gguf::Content,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), FamilyError> {
    let tensor = content
        .tensor(reader, name, &Device::Cpu)
        .map_err(FamilyError::Candle)?;
    let shape = tensor.shape().dims().to_vec();
    let values = tensor
        .dequantize(&Device::Cpu)
        .and_then(|tensor| {
            tensor
                .flatten_all()
                .and_then(|tensor| tensor.to_vec1::<f32>())
        })
        .map_err(FamilyError::Candle)?;
    Ok((values, shape))
}

/// Render one GGUF metadata value the way the pinned-contract comparison
/// reads it.
fn metadata_text(value: &gguf::Value) -> String {
    match value {
        gguf::Value::String(value) => value.clone(),
        gguf::Value::U32(value) => value.to_string(),
        gguf::Value::U64(value) => value.to_string(),
        gguf::Value::F32(value) => value.to_string(),
        gguf::Value::F64(value) => value.to_string(),
        gguf::Value::Bool(value) => value.to_string(),
        other => format!("{other:?}"),
    }
}

fn metadata_u32(content: &gguf::Content, key: &'static str) -> Result<u32, FamilyError> {
    match content.metadata.get(key) {
        Some(gguf::Value::U32(value)) => Ok(*value),
        Some(gguf::Value::U64(value)) => u32::try_from(*value)
            .map_err(|_| FamilyError::contract(key, "a u32 metadata value", value.to_string())),
        Some(other) => Err(FamilyError::contract(
            key,
            "a u32 metadata value",
            metadata_text(other),
        )),
        None => Err(FamilyError::ContractMismatch {
            field: "gguf.metadata",
            expected: format!("`{key}` present"),
            actual: "missing".to_owned(),
        }),
    }
}

fn metadata_f32(content: &gguf::Content, key: &'static str) -> Result<f32, FamilyError> {
    match content.metadata.get(key) {
        Some(gguf::Value::F32(value)) => Ok(*value),
        Some(gguf::Value::F64(value)) => Ok(*value as f32),
        Some(other) => Err(FamilyError::contract(
            key,
            "an f32 metadata value",
            metadata_text(other),
        )),
        None => Err(FamilyError::ContractMismatch {
            field: "gguf.metadata",
            expected: format!("`{key}` present"),
            actual: "missing".to_owned(),
        }),
    }
}

fn metadata_f64(value: &gguf::Value) -> Result<f64, FamilyError> {
    match value {
        gguf::Value::F32(value) => Ok(f64::from(*value)),
        gguf::Value::F64(value) => Ok(*value),
        other => Err(FamilyError::InvalidInput(format!(
            "gguf metadata is not numeric: {other:?}"
        ))),
    }
}

/// Plain RMSNorm over the last axis (`x · w · rsqrt(mean(x²) + eps)`), the
/// Qwen norm without the Qwen3.5 offset fold.
fn rms_norm(x: &Array, weight: &Array, eps: f64) -> Result<Array, FamilyError> {
    mlx_rs::fast::rms_norm(x, Some(weight), eps as f32).map_err(mlx_error("rms norm"))
}

/// Host FP32 RoPE tables at `(max_sequence_tokens, head_dim / 2)`, the same
/// math as the candle oracle's `precomput_freqs_cis`.
fn rope_tables(config: &DenseQwenConfig) -> (Array, Array) {
    let half = config.head_dim / 2;
    let mut cos_values = Vec::with_capacity(config.max_sequence_tokens * half);
    let mut sin_values = Vec::with_capacity(config.max_sequence_tokens * half);
    for position in 0..config.max_sequence_tokens {
        for index in 0..half {
            let exponent = (2 * index) as f64 / config.head_dim as f64;
            let inv_freq = 1.0 / config.rope_theta.powf(exponent);
            let angle = position as f32 * inv_freq as f32;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }
    let shape = [config.max_sequence_tokens as i32, half as i32];
    let cos = Array::from_slice(&cos_values, &shape);
    let sin = Array::from_slice(&sin_values, &shape);
    cos.eval().expect("rope cos materializes");
    sin.eval().expect("rope sin materializes");
    (cos, sin)
}

/// Full-dimension NeoX rotary embedding over `(heads, seq, head_dim)`
/// values: split each head's width into halves, rotate the halves against
/// each other at the absolute positions' angles.
fn apply_rotary(values: &Array, rope_cos: &Array, rope_sin: &Array) -> Result<Array, FamilyError> {
    let dims = values.shape();
    let seq = dims[1];
    let indices: Vec<u32> = (0..seq as u32).collect();
    let idx = Array::from_slice(&indices, &[seq]);
    let cos = rope_cos
        .take_axis(&idx, 0)
        .map_err(mlx_error("rope cos slice"))?;
    let sin = rope_sin
        .take_axis(&idx, 0)
        .map_err(mlx_error("rope sin slice"))?;
    let head_dim = *dims.last().expect("head dim") as usize;
    let half = (head_dim / 2) as i32;
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
