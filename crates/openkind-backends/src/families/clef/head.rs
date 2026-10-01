//! Joint schema head: a candle port of the reference `JointSchemaHead`.
//!
//! The head reads the backbone's final-norm hidden states plus the encoded
//! record's token spans and produces one logit per allowed option for every
//! question in a single pass. The math replicates the reference PyTorch
//! module exactly: evidence routing over the state memory, per-question
//! option summaries, pre-norm decoder layers across questions, and a
//! lexical-prior plus gated residual-joint logit combination. All compute is
//! FP32; the checkpoint stores BF16 weights that candle widens at load.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::families::support::FamilyError;

use super::renderer::EncodedRecord;

/// Rows of the backbone output-embedding matrix for the requested token ids,
/// row-major `[ids.len(), hidden_size]` FP32 values.
pub trait LexicalLookup: Send + Sync {
    fn rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, FamilyError>;
}

/// Joint-head geometry from the pinned `joint_head_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JointHeadConfig {
    pub(crate) hidden_size: usize,
    pub(crate) width: usize,
    pub(crate) routing_layers: usize,
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) feedforward: usize,
}

/// torch `torch.nn.LayerNorm(width)` epsilon default.
const LAYER_NORM_EPS: f32 = 1e-5;
/// torch `F.normalize` L2 epsilon default.
const NORMALIZE_EPS: f32 = 1e-12;
/// torch `F.cosine_similarity` epsilon default.
const COSINE_EPS: f32 = 1e-8;

/// Loaded joint schema head.
pub(crate) struct JointSchemaHead {
    config: JointHeadConfig,
    hidden_norm: LayerNorm,
    memory_projection: Tensor,
    question_projection: Tensor,
    option_question_projection: Tensor,
    global_projection: Tensor,
    option_context_projection: Tensor,
    option_lexical_projection: Tensor,
    type_embedding: Tensor,
    evidence_layers: Vec<EvidenceRoutingLayer>,
    option_summary_norm: LayerNorm,
    decoder_layers: Vec<TransformerDecoderLayer>,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    residual_scorer_0: Tensor,
    residual_scorer_0_bias: Tensor,
    residual_scorer_3: Tensor,
    residual_scorer_3_bias: Tensor,
    prior_logit_scale: f32,
    joint_logit_scale: f32,
    residual_gate: f32,
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn apply(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        candle_nn::ops::layer_norm(x, &self.weight, &self.bias, LAYER_NORM_EPS)
    }
}


/// Linear projection that accepts any batch rank: flattens leading axes to
/// 2-D, applies the weight and bias, and restores the shape. candle matmul
/// requires matching ranks, so every head projection funnels through here.
fn batched_linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor, FamilyError> {
    let shape = x.dims().to_vec();
    let width = *shape.last().ok_or_else(|| {
        FamilyError::InvalidInput("batched_linear requires rank >= 1".into())
    })?;
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let flat = x.reshape((rows, width))?;
    let mut out = flat.matmul(&weight.t()?)?;
    if let Some(bias) = bias {
        out = out.broadcast_add(bias)?;
    }
    let last = *out.dims().last().unwrap_or(&width);
    let mut out_shape = shape;
    let row_count = out_shape.len();
    out_shape[row_count - 1] = last;
    Ok(out.reshape(out_shape)?)
}

/// torch `nn.MultiheadAttention` (batch_first, packed in-projection, biases).
struct PackedAttention {
    heads: usize,
    width: usize,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
}

impl PackedAttention {
    fn load(variables: &VarBuilder<'_>, width: usize, heads: usize) -> Result<Self, FamilyError> {
        Ok(Self {
            heads,
            width,
            in_proj_weight: variables
                .get((width * 3, width), "in_proj_weight")
                ?,
            in_proj_bias: variables
                .get(width * 3, "in_proj_bias")
                ?,
            out_proj_weight: variables
                .get((width, width), "out_proj.weight")
                ?,
            out_proj_bias: variables
                .get(width, "out_proj.bias")
                ?,
        })
    }

    fn apply(&self, queries: &Tensor, keys: &Tensor, values: &Tensor) -> Result<Tensor, FamilyError> {
        let width = self.width;
        let w_q = self
            .in_proj_weight
            .narrow(0, 0, width)
            ?;
        let w_k = self
            .in_proj_weight
            .narrow(0, width, width)
            ?;
        let w_v = self
            .in_proj_weight
            .narrow(0, width * 2, width)
            ?;
        let b_q = self.in_proj_bias.narrow(0, 0, width)?;
        let b_k = self
            .in_proj_bias
            .narrow(0, width, width)
            ?;
        let b_v = self
            .in_proj_bias
            .narrow(0, width * 2, width)
            ?;
        // candle matmul requires matching ranks: project in 2-D
        // [batch * length, width], then restore the batch axis.
        let (batch_queries, query_length, _) = queries.dims3()?;
        let flatten = |x: &Tensor| -> Result<Tensor, FamilyError> {
            Ok(x.reshape((batch_queries * query_length, width))?)
        };
        let q = flatten(queries)?
            .matmul(&w_q.t()?)?
            .broadcast_add(&b_q)?
            .reshape((batch_queries, query_length, width))?;
        let (batch_keys, key_length, _) = keys.dims3()?;
        let flatten_keys = |x: &Tensor| -> Result<Tensor, FamilyError> {
            Ok(x.reshape((batch_keys * key_length, width))?)
        };
        let k = flatten_keys(keys)?
            .matmul(&w_k.t()?)?
            .broadcast_add(&b_k)?
            .reshape((batch_keys, key_length, width))?;
        let v = flatten_keys(values)?
            .matmul(&w_v.t()?)?
            .broadcast_add(&b_v)?
            .reshape((batch_keys, key_length, width))?;
        let head_dim = width / self.heads;
        // (B, L, W) -> (B, L, H, hd) -> (B, H, L, hd)
        let (batch, query_len, _) = q.dims3()?;
        let (batch_keys, key_len, _) = k.dims3()?;
        if batch_keys != batch {
            return Err(FamilyError::InvalidInput(
                "attention query and key batch sizes differ".into(),
            ));
        }
        let to_heads = |x: &Tensor, len: usize| -> Result<Tensor, FamilyError> {
            Ok(x.reshape((batch, len, self.heads, head_dim))?.transpose(1, 2)?)
        };
        let q = to_heads(&q, query_len)?.contiguous()?;
        let k = to_heads(&k, key_len)?.contiguous()?;
        let v = to_heads(&v, key_len)?.contiguous()?;
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)
            ?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)
            ?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let mixed = probabilities
            .matmul(&v)
            ?
            .transpose(1, 2)
            ?
            .reshape((batch, query_len, width))
            ?;
        let (out_batch, out_length, _) = mixed.dims3()?;
        Ok(mixed
            .reshape((out_batch * out_length, width))?
            .matmul(&self.out_proj_weight.t()?)?
            .broadcast_add(&self.out_proj_bias)?
            .reshape((out_batch, out_length, width))?)
    }
}

struct EvidenceRoutingLayer {
    query_norm: LayerNorm,
    memory_norm: LayerNorm,
    attention: PackedAttention,
    feedforward_norm: LayerNorm,
    feedforward_0: Tensor,
    feedforward_0_bias: Tensor,
    feedforward_3: Tensor,
    feedforward_3_bias: Tensor,
}

impl EvidenceRoutingLayer {
    fn load(
        variables: &VarBuilder<'_>,
        width: usize,
        heads: usize,
        feedforward: usize,
    ) -> Result<Self, FamilyError> {
        Ok(Self {
            query_norm: load_layer_norm(variables, "query_norm", width)?,
            memory_norm: load_layer_norm(variables, "memory_norm", width)?,
            attention: PackedAttention::load(
                &variables.pp("attention"),
                width,
                heads,
            )?,
            feedforward_norm: load_layer_norm(variables, "feedforward_norm", width)?,
            feedforward_0: variables
                .get((feedforward, width), "feedforward.0.weight")
                ?,
            feedforward_0_bias: variables
                .get(feedforward, "feedforward.0.bias")
                ?,
            feedforward_3: variables
                .get((width, feedforward), "feedforward.3.weight")
                ?,
            feedforward_3_bias: variables
                .get(width, "feedforward.3.bias")
                ?,
        })
    }

    fn apply(&self, queries: &Tensor, memory: &Tensor) -> Result<Tensor, FamilyError> {
        let normalized_queries = self.query_norm.apply(queries)?;
        let normalized_memory = self.memory_norm.apply(memory)?;
        let routed = self
            .attention
            .apply(&normalized_queries, &normalized_memory, &normalized_memory)?;
        let queries = queries.broadcast_add(&routed)?;
        let ff_input = self
            .feedforward_norm
            .apply(&queries)
            ?;
        let ff = batched_linear(
            &batched_linear(&ff_input, &self.feedforward_0, Some(&self.feedforward_0_bias))?
                .gelu()?,
            &self.feedforward_3,
            Some(&self.feedforward_3_bias),
        )?;
        Ok(queries.broadcast_add(&ff)?)
    }
}

/// torch `nn.TransformerDecoderLayer(norm_first=True, activation="gelu")`
/// with the dropout disabled (evaluation mode).
struct TransformerDecoderLayer {
    self_attn: PackedAttention,
    cross_attn: PackedAttention,
    linear1: Tensor,
    linear1_bias: Tensor,
    linear2: Tensor,
    linear2_bias: Tensor,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl TransformerDecoderLayer {
    fn load(
        variables: &VarBuilder<'_>,
        width: usize,
        heads: usize,
        feedforward: usize,
    ) -> Result<Self, FamilyError> {
        Ok(Self {
            self_attn: PackedAttention::load(
                &variables.pp("self_attn"),
                width,
                heads,
            )?,
            cross_attn: PackedAttention::load(
                &variables.pp("multihead_attn"),
                width,
                heads,
            )?,
            linear1: variables
                .get((feedforward, width), "linear1.weight")
                ?,
            linear1_bias: variables
                .get(feedforward, "linear1.bias")
                ?,
            linear2: variables
                .get((width, feedforward), "linear2.weight")
                ?,
            linear2_bias: variables
                .get(width, "linear2.bias")
                ?,
            norm1: load_layer_norm(variables, "norm1", width)?,
            norm2: load_layer_norm(variables, "norm2", width)?,
            norm3: load_layer_norm(variables, "norm3", width)?,
        })
    }

    fn apply(&self, x: &Tensor, memory: &Tensor) -> Result<Tensor, FamilyError> {
        let h = self.norm1.apply(x)?;
        let x = x
            .broadcast_add(&self.self_attn.apply(&h, &h, &h)?)
            ?;
        let h = self.norm2.apply(&x)?;
        let x = x
            .broadcast_add(&self.cross_attn.apply(&h, memory, memory)?)
            ?;
        let h = self.norm3.apply(&x)?;
        let ff = batched_linear(
            &batched_linear(&h, &self.linear1, Some(&self.linear1_bias))?.gelu()?,
            &self.linear2,
            Some(&self.linear2_bias),
        )?;
        Ok(x.broadcast_add(&ff)?)
    }
}

fn load_layer_norm(
    variables: &VarBuilder<'_>,
    prefix: &str,
    width: usize,
) -> Result<LayerNorm, FamilyError> {
    let variables = variables.pp(prefix);
    Ok(LayerNorm {
        weight: variables.get(width, "weight")?,
        bias: variables.get(width, "bias")?,
    })
}

fn scalar(variables: &VarBuilder<'_>, name: &str) -> Result<f32, FamilyError> {
    Ok(variables
        .get((), name)
        ?
        .to_scalar::<f32>()
        ?)
}

fn l2_normalize(x: &Tensor, eps: f32) -> Result<Tensor, FamilyError> {
    let norm = x
        .sqr()
        ?
        .sum_keepdim(candle_core::D::Minus1)
        ?
        .sqrt()
        ?
        .clamp(eps, f32::MAX)
        ?;
    Ok(x.broadcast_div(&norm)?)
}

impl JointSchemaHead {
    /// Load the head from a digest-verified `joint_head.safetensors` file.
    pub(crate) fn load(
        joint_head_config: &JointHeadConfig,
        weights_path: &std::path::Path,
        device: &Device,
    ) -> Result<Self, FamilyError> {
        // SAFETY: the caller digest-verified this read-only safetensors file
        // in place; the VarBuilder only maps it.
        let variables = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                ?
        };
        let config = *joint_head_config;
        let width = config.width;
        let hidden = config.hidden_size;
        let mut evidence_layers = Vec::with_capacity(config.routing_layers);
        for index in 0..config.routing_layers {
            evidence_layers.push(EvidenceRoutingLayer::load(
                &variables.pp(format!("evidence_layers.{index}")),
                width,
                config.heads,
                config.feedforward,
            )?);
        }
        let mut decoder_layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            decoder_layers.push(TransformerDecoderLayer::load(
                &variables.pp(format!("layers.{index}")),
                width,
                config.heads,
                config.feedforward,
            )?);
        }
        Ok(Self {
            config: config,
            hidden_norm: load_layer_norm(&variables, "hidden_norm", hidden)?,
            memory_projection: variables
                .get((width, hidden), "memory_projection.weight")
                ?,
            question_projection: variables
                .get((width, hidden), "question_projection.weight")
                ?,
            option_question_projection: variables
                .get((width, hidden), "option_question_projection.weight")
                ?,
            global_projection: variables
                .get((width, hidden), "global_projection.weight")
                ?,
            option_context_projection: variables
                .get((width, hidden), "option_context_projection.weight")
                ?,
            option_lexical_projection: variables
                .get((width, hidden), "option_lexical_projection.weight")
                ?,
            type_embedding: variables
                .get((3, width), "type_embedding.weight")
                ?,
            evidence_layers,
            option_summary_norm: load_layer_norm(&variables, "option_summary_norm", width)?,
            decoder_layers,
            field_norm: load_layer_norm(&variables, "field_norm", width)?,
            option_norm: load_layer_norm(&variables, "option_norm", width)?,
            residual_scorer_0: variables
                .get((width, width * 4), "residual_scorer.0.weight")
                ?,
            residual_scorer_0_bias: variables
                .get(width, "residual_scorer.0.bias")
                ?,
            residual_scorer_3: variables
                .get((1, width), "residual_scorer.3.weight")
                ?,
            residual_scorer_3_bias: variables
                .get(1, "residual_scorer.3.bias")
                ?,
            prior_logit_scale: scalar(&variables, "prior_logit_scale")?,
            joint_logit_scale: scalar(&variables, "joint_logit_scale")?,
            residual_gate: scalar(&variables, "residual_gate")?,
        })
    }

    /// One logit vector per question, in encoded-question order.
    ///
    /// `hidden` is the row-major `[token_count, hidden_size]` final-norm
    /// backbone output, `record` the encoded record whose spans index into
    /// it, and `lexical` the output-embedding row reader.
    pub(crate) fn forward(
        &self,
        hidden: &[f32],
        token_count: usize,
        record: &EncodedRecord,
        lexical: &dyn LexicalLookup,
        device: &Device,
    ) -> Result<Vec<Vec<f64>>, FamilyError> {
        let hidden_size = self.config.hidden_size;
        let width = self.config.width;
        let hidden_tensor = Tensor::from_vec(hidden.to_vec(), (token_count, hidden_size), device)
            ?;
        let normalized_hidden = self
            .hidden_norm
            .apply(&hidden_tensor)
            ?;
        // memory: [1, T, width]
        let memory = normalized_hidden
            .matmul(&self.memory_projection.t()?)
            ?
            .unsqueeze(0)
            ?;
        // global vector: last real token's hidden state [hidden]
        let global_vector = normalized_hidden
            .narrow(0, token_count - 1, 1)
            ?
            .squeeze(0)
            ?;

        let question_count = record.questions.len();
        let mut question_vectors = Vec::with_capacity(question_count);
        for question in &record.questions {
            question_vectors.push(mean_span(&normalized_hidden, question.question_span)?);
        }
        let question_vectors = Tensor::stack(&question_vectors, 0)?;

        // Option contexts and lexical vectors per question.
        let mut option_contexts: Vec<Tensor> = Vec::with_capacity(question_count);
        let mut lexical_options: Vec<Tensor> = Vec::with_capacity(question_count);
        let mut option_counts: Vec<usize> = Vec::with_capacity(question_count);
        for question in &record.questions {
            let mut contexts = Vec::with_capacity(question.option_spans.len());
            let mut lexical_rows: Vec<Tensor> = Vec::with_capacity(question.option_spans.len());
            for (start, end) in &question.option_spans {
                contexts.push(mean_span(&normalized_hidden, (*start, *end))?);
                let ids = &record.input_ids[*start..*end];
                let rows = lexical.rows(ids)?;
                let expected = ids.len() * hidden_size;
                if rows.len() != expected {
                    return Err(FamilyError::InvalidInput(format!(
                        "lexical lookup returned {} values for {expected}",
                        rows.len()
                    )));
                }
                lexical_rows.push(
                    Tensor::from_vec(rows, (ids.len(), hidden_size), device)
                        ?
                        .mean(0)
                        ?,
                );
            }
            option_contexts.push(Tensor::stack(&contexts, 0)?);
            lexical_options.push(Tensor::stack(&lexical_rows, 0)?);
            option_counts.push(question.option_spans.len());
        }

        // Option queries: context + lexical + parent-question projections.
        let mut option_queries: Vec<Tensor> = Vec::new();
        for (question_index, (context, lexical)) in
            option_contexts.iter().zip(&lexical_options).enumerate()
        {
            let question_part = self
                .option_question_projection
                .matmul(
                    &question_vectors
                        .narrow(0, question_index, 1)
                        ?
                        .squeeze(0)
                        ?
                        .unsqueeze(1)
                        ?,
                )
                ?
                .squeeze(1)
                ?;
            let query = context
                .matmul(&self.option_context_projection.t()?)
                ?
                .broadcast_add(
                    &lexical
                        .matmul(&self.option_lexical_projection.t()?)
                        ?,
                )
                ?
                .broadcast_add(&question_part)
                ?;
            option_queries.push(query);
        }
        // routed: [1, total_options, width]
        let mut routed = Tensor::cat(&option_queries, 0)
            ?
            .unsqueeze(0)
            ?;
        for layer in &self.evidence_layers {
            routed = layer.apply(&routed, &memory)?;
        }
        let routed = routed.squeeze(0)?;
        let split_options = split_at(&routed, &option_counts)?;

        // Field vectors from question projections, option summaries, global
        // vector, and the type embedding.
        let base_fields = question_vectors
            .matmul(&self.question_projection.t()?)
            ?;
        let mut base_fields_rows: Vec<Tensor> = Vec::with_capacity(question_count);
        for index in 0..question_count {
            base_fields_rows.push(base_fields.narrow(0, index, 1)?.squeeze(0)?);
        }
        let mut summaries: Vec<Tensor> = Vec::with_capacity(question_count);
        for (question_index, (field, options)) in
            (0..question_count).zip(base_fields_rows.iter().zip(&split_options))
        {
            let _ = question_index;
            let scores = options
                .matmul(&field.unsqueeze(1)?)
                ?
                .affine(1.0 / (width as f64).sqrt(), 0.0)
                ?;
            let routing_weights = candle_nn::ops::softmax(&scores, 0)?;
            summaries.push(
                routing_weights
                    .transpose(0, 1)?
                    .matmul(options)?
                    .squeeze(0)?
                    .squeeze(0)?,
            );
        }
        let summaries = self
            .option_summary_norm
            .apply(&Tensor::stack(&summaries, 0)?)?;
        let global_part = self
            .global_projection
            .matmul(&global_vector.unsqueeze(1)?)
            ?
            .squeeze(1)
            ?;
        let type_ids: Vec<Tensor> = record
            .questions
            .iter()
            .map(|question| -> Result<Tensor, FamilyError> {
                Ok(self
                    .type_embedding
                    .narrow(0, usize::try_from(question.question_type.type_id()).unwrap_or(0), 1)?
                    .squeeze(0)?)
            })
            .collect::<Result<Vec<_>, FamilyError>>()?;
        let mut fields = base_fields
            .broadcast_add(&summaries)
            ?
            .broadcast_add(&global_part)
            ?
            .broadcast_add(&Tensor::stack(&type_ids, 0)?)
            ?
            .unsqueeze(0)
            ?;
        for layer in &self.decoder_layers {
            fields = layer.apply(&fields, &memory)?;
        }
        let fields = self.field_norm.apply(&fields.squeeze(0)?)?;
        let mut fields_rows: Vec<Tensor> = Vec::with_capacity(question_count);
        for index in 0..question_count {
            fields_rows.push(fields.narrow(0, index, 1)?.squeeze(0)?);
        }

        // Per-option logits: lexical prior plus gated residual joint term.
        let prior_scale = self.prior_logit_scale.clamp(0.0, 100.0_f64.ln() as f32).exp();
        let joint_scale = self.joint_logit_scale.clamp(0.0, 100.0_f64.ln() as f32).exp();
        let gate = 1.0 / (1.0 + (-self.residual_gate).exp());
        let mut record_logits: Vec<Vec<f64>> = Vec::with_capacity(question_count);
        for (question_index, ((field, lexical), options)) in fields_rows
            .iter()
            .zip(lexical_options.iter())
            .zip(split_options.iter())
            .enumerate()
        {
            let anchor = l2_normalize(
                &question_vectors
                    .narrow(0, question_index, 1)
                    ?
                    .squeeze(0)
                    ?
                    .broadcast_add(&global_vector)
                    ?,
                NORMALIZE_EPS,
            )?;
            let lexical_anchor = l2_normalize(lexical, NORMALIZE_EPS)?;
            let prior = lexical_anchor
                .matmul(&anchor.unsqueeze(1)?)
                ?
                .affine(f64::from(prior_scale), 0.0)
                ?;
            let options = self.option_norm.apply(&options)?;
            let repeated_field = field
                .unsqueeze(0)
                ?
                .broadcast_as(options.shape().to_owned())?;
            let cosine = cosine_similarity(&repeated_field, &options)?;
            let features = Tensor::cat(
                &[
                    repeated_field.clone(),
                    options.clone(),
                    repeated_field.mul(&options)?,
                    repeated_field.sub(&options)?.abs()?,
                ],
                candle_core::D::Minus1,
            )?;
            let residual = batched_linear(
                &batched_linear(&features, &self.residual_scorer_0, Some(&self.residual_scorer_0_bias))?
                    .gelu()?,
                &self.residual_scorer_3,
                Some(&self.residual_scorer_3_bias),
            )?
            .squeeze(1)?;
            let joint = cosine
                .affine(f64::from(joint_scale), 0.0)?
                .broadcast_add(&residual)?;
            let logits = prior
                .squeeze(1)?
                .broadcast_add(&joint.affine(f64::from(gate), 0.0)?)?;
            let values = logits
                .to_dtype(DType::F64)
                ?
                .flatten_all()
                ?
                .to_vec1::<f64>()
                ?;
            record_logits.push(values);
        }
        Ok(record_logits)
    }
}

fn mean_span(hidden: &Tensor, span: (usize, usize)) -> Result<Tensor, FamilyError> {
    let (start, end) = span;
    if end <= start {
        return Err(FamilyError::InvalidInput(format!(
            "empty token span [{start}, {end})"
        )));
    }
    Ok(hidden.narrow(0, start, end - start)?.mean(0)?)
}

fn split_at(tensor: &Tensor, counts: &[usize]) -> Result<Vec<Tensor>, FamilyError> {
    let mut parts = Vec::with_capacity(counts.len());
    let mut offset = 0;
    for count in counts {
        parts.push(
            tensor
                .narrow(0, offset, *count)
                ?,
        );
        offset += count;
    }
    Ok(parts)
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor, FamilyError> {
    let dot = a.mul(b)?.sum(candle_core::D::Minus1)?;
    let norm_a = a.sqr()?.sum(candle_core::D::Minus1)?.sqrt()?;
    let norm_b = b.sqr()?.sum(candle_core::D::Minus1)?.sqrt()?;
    let denominator = norm_a.mul(&norm_b)?.clamp(COSINE_EPS, f32::MAX)?;
    Ok(dot.broadcast_div(&denominator)?)
}
