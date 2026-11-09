//! Shared post-LN MLX encoder body for the BERT-family backends (feature
//! `mlx`, macOS arm64).
//!
//! Mirrors the candle CPU arithmetic of the `distilbert` and `bert` models
//! op for op (FP32 compute over digest-verified FP32 shards): learned word +
//! absolute-position embeddings (plus the BERT token-type row-0 add), biased
//! LayerNorm everywhere, biased Q/K/V/output/FFN linears, full bidirectional
//! attention with no mask (both served families pass an unpadded single pair,
//! so the CPU paths mask nothing), and the post-residual LayerNorm order —
//! normalize AFTER the attention residual and again AFTER the FFN residual,
//! the opposite order from ModernBERT. The FFN activation is the exact
//! erf-based GELU both pinned checkpoints train with. The candle CPU paths
//! remain the correctness oracle.
//!
//! Checkpoint namespaces differ (`distilbert.transformer.layer.N...` vs
//! `bert.encoder.layer.N...`), so callers normalize tensor names onto this
//! body's BERT-style internal names at load through
//! [`BertMlxKeyScheme`]; one forward then serves every family whose backbone
//! is a post-LN BERT-shaped encoder.
//!
//! Every MLX operation — model load and per-pair forward — runs through the
//! process-wide `MlxRuntime` execution lock, per the workspace MLX
//! discipline. The load and the [`BertMlxModel::forward`] body itself do not
//! take the lock (the lock is not reentrant): callers wrap them in
//! `MlxRuntime::execute`.

use std::collections::BTreeMap;
use std::ops::{Div, Sub};

use mlx_rs::Array;

use crate::families::mlx_modernbert::{
    gelu, layer_norm_biased, map_tensor, map_transposed, scalar,
};
use crate::families::qwen35_survey_mlx::mlx_error;
use crate::families::support::FamilyError;

/// Configuration subset the shared body needs, pinned by each family from
/// its digest-verified checkpoint contract.
pub(crate) struct BertMlxConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    /// Biased LayerNorm epsilon (candle `distilbert` pins 1e-12 in code; the
    /// pinned BERT checkpoint carries 1e-12 in its digest-verified config).
    pub(crate) layer_norm_eps: f64,
    /// Token-type vocabulary size (`Some(2)` for BERT checkpoints whose
    /// embeddings add the token-type row 0, `None` for DistilBERT which has
    /// no token-type table).
    pub(crate) token_type_vocab: Option<usize>,
}

/// Checkpoint tensor-name scheme this body normalizes onto its internal
/// BERT-style names.
pub(crate) enum BertMlxKeyScheme {
    /// `distilbert.embeddings.*` / `distilbert.transformer.layer.N.*`
    /// (DistilBERT naming: `q_lin`, `sa_layer_norm`, `ffn.lin1`, ...).
    DistilBert,
    /// `bert.embeddings.*` / `bert.encoder.layer.N.*` (HF BERT naming:
    /// `attention.self.query`, `attention.output.LayerNorm`, ...).
    Bert,
}

impl BertMlxKeyScheme {
    /// Map one checkpoint tensor name onto the shared body's internal name,
    /// or `None` when the checkpoint carries a tensor this body does not
    /// consume. Every mapped tensor is shape-checked at load, and a required
    /// tensor the scheme fails to map fails the load closed.
    pub(crate) fn internal_name(&self, key: &str) -> Option<String> {
        match self {
            Self::DistilBert => {
                let rest = key.strip_prefix("distilbert.")?;
                if let Some(suffix) = rest.strip_prefix("embeddings.") {
                    return match suffix {
                        "word_embeddings.weight"
                        | "position_embeddings.weight"
                        | "LayerNorm.weight"
                        | "LayerNorm.bias" => Some(format!("embeddings.{suffix}")),
                        _ => None,
                    };
                }
                let layer = rest.strip_prefix("transformer.layer.")?;
                let (index, tensor) = layer.split_once('.')?;
                let internal = match tensor {
                    "attention.q_lin.weight" => "attention.self.query.weight",
                    "attention.q_lin.bias" => "attention.self.query.bias",
                    "attention.k_lin.weight" => "attention.self.key.weight",
                    "attention.k_lin.bias" => "attention.self.key.bias",
                    "attention.v_lin.weight" => "attention.self.value.weight",
                    "attention.v_lin.bias" => "attention.self.value.bias",
                    "attention.out_lin.weight" => "attention.output.dense.weight",
                    "attention.out_lin.bias" => "attention.output.dense.bias",
                    "sa_layer_norm.weight" => "attention.output.LayerNorm.weight",
                    "sa_layer_norm.bias" => "attention.output.LayerNorm.bias",
                    "ffn.lin1.weight" => "intermediate.dense.weight",
                    "ffn.lin1.bias" => "intermediate.dense.bias",
                    "ffn.lin2.weight" => "output.dense.weight",
                    "ffn.lin2.bias" => "output.dense.bias",
                    "output_layer_norm.weight" => "output.LayerNorm.weight",
                    "output_layer_norm.bias" => "output.LayerNorm.bias",
                    _ => return None,
                };
                Some(format!("layers.{index}.{internal}"))
            }
            Self::Bert => {
                let rest = key.strip_prefix("bert.")?;
                if let Some(suffix) = rest.strip_prefix("embeddings.") {
                    return match suffix {
                        "word_embeddings.weight"
                        | "position_embeddings.weight"
                        | "token_type_embeddings.weight"
                        | "LayerNorm.weight"
                        | "LayerNorm.bias" => Some(format!("embeddings.{suffix}")),
                        _ => None,
                    };
                }
                let layer = rest.strip_prefix("encoder.layer.")?;
                let (index, tensor) = layer.split_once('.')?;
                match tensor {
                    "attention.self.query.weight"
                    | "attention.self.query.bias"
                    | "attention.self.key.weight"
                    | "attention.self.key.bias"
                    | "attention.self.value.weight"
                    | "attention.self.value.bias"
                    | "attention.output.dense.weight"
                    | "attention.output.dense.bias"
                    | "attention.output.LayerNorm.weight"
                    | "attention.output.LayerNorm.bias"
                    | "intermediate.dense.weight"
                    | "intermediate.dense.bias"
                    | "output.dense.weight"
                    | "output.dense.bias"
                    | "output.LayerNorm.weight"
                    | "output.LayerNorm.bias" => Some(format!("layers.{index}.{tensor}")),
                    _ => None,
                }
            }
        }
    }
}

/// One post-LN encoder layer: biased Q/K/V/output linears with full
/// bidirectional attention, then the post-residual LayerNorms around the FFN.
struct BertMlxLayer {
    query_w: Array,
    query_b: Array,
    key_w: Array,
    key_b: Array,
    value_w: Array,
    value_b: Array,
    output_w: Array,
    output_b: Array,
    attention_norm_weight: Array,
    attention_norm_bias: Array,
    ffn_in_w: Array,
    ffn_in_b: Array,
    ffn_out_w: Array,
    ffn_out_b: Array,
    output_norm_weight: Array,
    output_norm_bias: Array,
    heads: usize,
    head_dim: usize,
    norm_eps: f64,
}

impl BertMlxLayer {
    /// Load one layer from the caller-normalized tensor map. Every required
    /// tensor is checked against the pinned shape and any missing tensor
    /// fails closed.
    fn from_tensor_map(
        tensors: &BTreeMap<String, Array>,
        config: &BertMlxConfig,
        index: usize,
    ) -> Result<Self, FamilyError> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let prefix = format!("layers.{index}");
        let tensor = |name: String, expected: &[usize]| map_tensor(tensors, &name, expected);
        let transposed =
            |name: String, rows: usize, cols: usize| map_transposed(tensors, &name, rows, cols);
        Ok(Self {
            query_w: transposed(
                format!("{prefix}.attention.self.query.weight"),
                hidden,
                hidden,
            )?,
            query_b: tensor(format!("{prefix}.attention.self.query.bias"), &[hidden])?,
            key_w: transposed(
                format!("{prefix}.attention.self.key.weight"),
                hidden,
                hidden,
            )?,
            key_b: tensor(format!("{prefix}.attention.self.key.bias"), &[hidden])?,
            value_w: transposed(
                format!("{prefix}.attention.self.value.weight"),
                hidden,
                hidden,
            )?,
            value_b: tensor(format!("{prefix}.attention.self.value.bias"), &[hidden])?,
            output_w: transposed(
                format!("{prefix}.attention.output.dense.weight"),
                hidden,
                hidden,
            )?,
            output_b: tensor(format!("{prefix}.attention.output.dense.bias"), &[hidden])?,
            attention_norm_weight: tensor(
                format!("{prefix}.attention.output.LayerNorm.weight"),
                &[hidden],
            )?,
            attention_norm_bias: tensor(
                format!("{prefix}.attention.output.LayerNorm.bias"),
                &[hidden],
            )?,
            ffn_in_w: transposed(
                format!("{prefix}.intermediate.dense.weight"),
                intermediate,
                hidden,
            )?,
            ffn_in_b: tensor(format!("{prefix}.intermediate.dense.bias"), &[intermediate])?,
            ffn_out_w: transposed(
                format!("{prefix}.output.dense.weight"),
                hidden,
                intermediate,
            )?,
            ffn_out_b: tensor(format!("{prefix}.output.dense.bias"), &[hidden])?,
            output_norm_weight: tensor(format!("{prefix}.output.LayerNorm.weight"), &[hidden])?,
            output_norm_bias: tensor(format!("{prefix}.output.LayerNorm.bias"), &[hidden])?,
            heads: config.num_attention_heads,
            head_dim: hidden / config.num_attention_heads,
            norm_eps: config.layer_norm_eps,
        })
    }

    /// One full bidirectional attention block over a `[seq, hidden]` input.
    /// No mask: the served pairs are unpadded single sequences, and the CPU
    /// oracle masks nothing over them.
    fn attention(&self, xs: &Array, seq: i32, hidden: i32) -> Result<Array, FamilyError> {
        let project = |weight: &Array,
                       bias: &Array,
                       input: &Array,
                       name: &'static str|
         -> Result<Array, FamilyError> {
            input
                .matmul(weight)
                .map_err(mlx_error(name))?
                .add(bias)
                .map_err(mlx_error(name))?
                .reshape(&[seq, self.heads as i32, self.head_dim as i32])
                .map_err(mlx_error(name))?
                .transpose_axes(&[1, 0, 2])
                .map_err(mlx_error(name))
        };
        let query = project(&self.query_w, &self.query_b, xs, "attn query")?;
        let keys = project(&self.key_w, &self.key_b, xs, "attn key")?;
        let values = project(&self.value_w, &self.value_b, xs, "attn value")?;

        // Scores over every position pair, scaled by 1/sqrt(head_dim) after
        // the matmul (the candle `bert` order; the `distilbert` oracle scales
        // q first — an FP32 reordering, not a semantic difference).
        let scores = query
            .matmul(
                &keys
                    .transpose_axes(&[0, 2, 1])
                    .map_err(mlx_error("attn k t"))?,
            )
            .map_err(mlx_error("attn scores"))?
            .divide(scalar(1.0 / (self.head_dim as f64).sqrt() as f32))
            .map_err(mlx_error("attn scale"))?;
        // Explicit max-subtracted softmax, matching the other MLX modules.
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
        let probabilities = exponentiated.div(&denominator);
        let mixed = probabilities
            .matmul(&values)
            .map_err(mlx_error("attn mix"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(mlx_error("attn heads"))?
            .reshape(&[seq, hidden])
            .map_err(mlx_error("attn flatten"))?;
        mixed
            .matmul(&self.output_w)
            .map_err(mlx_error("attn output"))?
            .add(&self.output_b)
            .map_err(mlx_error("attn output"))
    }

    /// One post-LN encoder layer: attention residual, LayerNorm, FFN
    /// residual, LayerNorm.
    fn forward(&self, xs: &Array) -> Result<Array, FamilyError> {
        let shape = xs.shape();
        let seq = shape[0];
        let hidden = shape[1];
        let attended = self.attention(xs, seq, hidden)?;
        let post_attention = layer_norm_biased(
            &xs.add(&attended).map_err(mlx_error("attention residual"))?,
            &self.attention_norm_weight,
            &self.attention_norm_bias,
            self.norm_eps,
        )?;
        let ffn_in = post_attention
            .matmul(&self.ffn_in_w)
            .map_err(mlx_error("ffn in"))?
            .add(&self.ffn_in_b)
            .map_err(mlx_error("ffn in"))?;
        let activated = gelu(&ffn_in)?;
        let ffn_out = activated
            .matmul(&self.ffn_out_w)
            .map_err(mlx_error("ffn out"))?
            .add(&self.ffn_out_b)
            .map_err(mlx_error("ffn out"))?;
        let post_ffn = post_attention
            .add(&ffn_out)
            .map_err(mlx_error("ffn residual"))?;
        Ok(layer_norm_biased(
            &post_ffn,
            &self.output_norm_weight,
            &self.output_norm_bias,
            self.norm_eps,
        )?)
    }
}

/// The shared post-LN encoder body in MLX arrays: embeddings and layers.
/// There is no final norm — the BERT/DistilBERT encoders return the last
/// layer's post-residual LayerNorm output directly, and the family heads
/// consume it from there.
pub(crate) struct BertMlxModel {
    word_embeddings: Array,
    position_embeddings: Array,
    token_type_embeddings: Option<Array>,
    embedding_norm_weight: Array,
    embedding_norm_bias: Array,
    layers: Vec<BertMlxLayer>,
    max_position_embeddings: usize,
    hidden_size: usize,
    norm_eps: f64,
}

// SAFETY: MLX array handles are immutable refcounted values whose
// handle-only operations are thread-safe, and every array evaluation in
// this model's methods runs under the process-wide `MlxRuntime` execution
// lock (the same discipline as the shared qwen35 MLX backbone).
unsafe impl Sync for BertMlxModel {}
unsafe impl Send for BertMlxModel {}

impl BertMlxModel {
    /// Build the body from a caller-normalized FP32 tensor map whose keys
    /// follow [`BertMlxKeyScheme`]'s internal names. Must run inside
    /// [`MlxRuntime::execute`]; every required tensor is checked against the
    /// pinned shape and any missing tensor fails closed.
    pub(crate) fn from_tensor_map(
        tensors: &BTreeMap<String, Array>,
        config: &BertMlxConfig,
    ) -> Result<Self, FamilyError> {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        if heads == 0 || !hidden.is_multiple_of(heads) {
            return Err(FamilyError::InvalidInput(format!(
                "hidden size {hidden} does not divide into {heads} attention heads"
            )));
        }
        let token_type_embeddings = match config.token_type_vocab {
            Some(vocab) => Some(map_tensor(
                tensors,
                "embeddings.token_type_embeddings.weight",
                &[vocab, hidden],
            )?),
            None => None,
        };
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertMlxLayer::from_tensor_map(tensors, config, index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            word_embeddings: map_tensor(
                tensors,
                "embeddings.word_embeddings.weight",
                &[config.vocab_size, hidden],
            )?,
            position_embeddings: map_tensor(
                tensors,
                "embeddings.position_embeddings.weight",
                &[config.max_position_embeddings, hidden],
            )?,
            token_type_embeddings,
            embedding_norm_weight: map_tensor(tensors, "embeddings.LayerNorm.weight", &[hidden])?,
            embedding_norm_bias: map_tensor(tensors, "embeddings.LayerNorm.bias", &[hidden])?,
            layers,
            max_position_embeddings: config.max_position_embeddings,
            hidden_size: hidden,
            norm_eps: config.layer_norm_eps,
        })
    }

    /// The hidden width the body was pinned to (the family heads matvec
    /// against it).
    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// One full-sequence forward over an unpadded single pair: embeddings,
    /// every layer, and the final `[seq, hidden]` hidden states (callers
    /// gather row 0, the CLS position). Must run inside
    /// [`MlxRuntime::execute`].
    pub(crate) fn forward(&self, token_ids: &[u32]) -> Result<Array, FamilyError> {
        let seq = token_ids.len();
        if seq == 0 {
            return Err(FamilyError::InvalidInput(
                "encoded pair is empty".to_owned(),
            ));
        }
        if seq > self.max_position_embeddings {
            return Err(FamilyError::InvalidInput(format!(
                "encoded pair has {seq} tokens but the pinned position table holds {}",
                self.max_position_embeddings
            )));
        }
        let ids = Array::from_slice(token_ids, &[seq as i32]);
        let mut xs = self
            .word_embeddings
            .take_axis(&ids, 0)
            .map_err(mlx_error("embed tokens"))?;
        let positions: Vec<u32> = (0..seq as u32).collect();
        let positions = Array::from_slice(&positions, &[seq as i32]);
        let position_embeddings = self
            .position_embeddings
            .take_axis(&positions, 0)
            .map_err(mlx_error("embed positions"))?;
        xs = xs
            .add(&position_embeddings)
            .map_err(mlx_error("embed add"))?;
        if let Some(token_type_embeddings) = &self.token_type_embeddings {
            // The CPU path passes all-zero token-type ids, so the row-0
            // embedding broadcasts over every position.
            let zeros = Array::from_slice(&[0_u32], &[1]);
            let row = token_type_embeddings
                .take_axis(&zeros, 0)
                .map_err(mlx_error("embed token type"))?;
            xs = xs.add(&row).map_err(mlx_error("embed token type add"))?;
        }
        xs = layer_norm_biased(
            &xs,
            &self.embedding_norm_weight,
            &self.embedding_norm_bias,
            self.norm_eps,
        )?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

/// Gather the row-0 (CLS) hidden state of a `[seq, hidden]` forward as a
/// host FP32 row. Must run inside [`MlxRuntime::execute`].
pub(crate) fn cls_row(hidden: &Array, width: usize) -> Result<Vec<f32>, FamilyError> {
    let zeros = Array::from_slice(&[0_u32], &[1]);
    let cls = hidden
        .take_axis(&zeros, 0)
        .map_err(mlx_error("cls row"))?
        .reshape(&[width as i32])
        .map_err(mlx_error("cls reshape"))?;
    cls.eval().map_err(mlx_error("cls eval"))?;
    cls.to_vec_cast::<f32>()
        .map_err(|error| FamilyError::Mlx(format!("cls read: {error}")))
}

/// Row-major biased matvec: `[out_dim, in_dim] @ row + bias` in FP32, the
/// arithmetic the CPU heads' candle linears compute. Shapes fail closed.
pub(crate) fn biased_matvec(
    weight: &[f32],
    bias: &[f32],
    in_dim: usize,
    row: &[f32],
) -> Result<Vec<f32>, FamilyError> {
    if row.len() != in_dim {
        return Err(FamilyError::InvalidInput(format!(
            "head input has {} values but the pinned head expects {in_dim}",
            row.len()
        )));
    }
    let out_dim = bias.len();
    if weight.len() != out_dim.saturating_mul(in_dim) {
        return Err(FamilyError::InvalidInput(format!(
            "head weight has {} values but the pinned head expects {}x{in_dim}",
            weight.len(),
            out_dim
        )));
    }
    let mut output = Vec::with_capacity(out_dim);
    for (out_index, bias) in bias.iter().enumerate() {
        let weights = &weight[out_index * in_dim..(out_index + 1) * in_dim];
        output.push(
            weights
                .iter()
                .zip(row)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                + bias,
        );
    }
    Ok(output)
}
