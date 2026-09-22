use serde::Deserialize;

use super::super::{
    require_equal, Qwen35Error, BACKBONE_ID, BACKBONE_REVISION, BUNDLE_SHA256, ORDERING_TOLERANCE,
    PROBABILITY_TOLERANCE, PROFILE_ID, REFERENCE_REPOSITORY,
};
use crate::qwen35::tokenizer::STATE_FIRST_RENDERER_ID;

const PARAMETER_COUNT: u64 = 4_205_751_296;
const PARAMETER_TENSOR_COUNT: usize = 426;
const LAYER_COUNT: usize = 32;

#[derive(Debug, Deserialize)]
pub(crate) struct BackboneContract {
    schema: String,
    profile_id: String,
    bundle_sha256: String,
    hf_reference_repo: String,
    base: BaseContract,
    exact_gates: ExactGates,
    decision_gate: DecisionGate,
    execution_reference: ExecutionReference,
}

#[derive(Debug, Deserialize)]
struct BaseContract {
    model_id: String,
    revision: String,
    parameter_count: u64,
    hidden_size: usize,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct ExactGates {
    profile_id: bool,
    base_revision: bool,
    renderer_layout: String,
    token_ids: bool,
    tensor_shapes: bool,
    layer_order: bool,
}

#[derive(Debug, Deserialize)]
struct DecisionGate {
    max_probability_delta: f64,
    order_tolerance: f64,
    require_zero_argmax_changes: bool,
    require_zero_policy_changes: bool,
}

#[derive(Debug, Deserialize)]
struct ExecutionReference {
    full_sequence_first: bool,
    cached_continuation_second: bool,
    branchable_state_after_full_backbone_parity: bool,
    no_generation: bool,
}

impl BackboneContract {
    pub(crate) fn validate(&self) -> Result<(), Qwen35Error> {
        require_equal(
            "backbone contract schema",
            "openkind-qwen35-backbone-parity-contract/v1",
            &self.schema,
        )?;
        validate_identity(
            &self.profile_id,
            &self.bundle_sha256,
            Some(&self.hf_reference_repo),
        )?;
        require_equal("base model ID", BACKBONE_ID, &self.base.model_id)?;
        require_equal(
            "base model revision",
            BACKBONE_REVISION,
            &self.base.revision,
        )?;
        require_equal(
            "parameter count",
            PARAMETER_COUNT,
            self.base.parameter_count,
        )?;
        require_equal("hidden size", 2_560, self.base.hidden_size)?;
        require_equal("backbone dtype", "float32", &self.base.dtype)?;
        require_equal(
            "renderer layout",
            STATE_FIRST_RENDERER_ID,
            &self.exact_gates.renderer_layout,
        )?;
        require_equal(
            "probability tolerance",
            PROBABILITY_TOLERANCE,
            self.decision_gate.max_probability_delta,
        )?;
        require_equal(
            "ordering tolerance",
            ORDERING_TOLERANCE,
            self.decision_gate.order_tolerance,
        )?;

        let required = [
            self.exact_gates.profile_id,
            self.exact_gates.base_revision,
            self.exact_gates.token_ids,
            self.exact_gates.tensor_shapes,
            self.exact_gates.layer_order,
            self.decision_gate.require_zero_argmax_changes,
            self.decision_gate.require_zero_policy_changes,
            self.execution_reference.full_sequence_first,
            self.execution_reference.cached_continuation_second,
            self.execution_reference
                .branchable_state_after_full_backbone_parity,
            self.execution_reference.no_generation,
        ];
        if required.into_iter().all(|gate| gate) {
            Ok(())
        } else {
            Err(Qwen35Error::InvalidInput(
                "Phase 3B contract disabled a required exact or execution gate".to_owned(),
            ))
        }
    }
}

/// Token-mixer family for one frozen Qwen 3.5 decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet recurrent linear-attention block.
    LinearAttention,
    /// Conventional grouped-query causal-attention block.
    FullAttention,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ArchitectureContract {
    schema: String,
    profile_id: String,
    bundle_sha256: String,
    base_model: String,
    base_revision: String,
    parameter_count: u64,
    model_class: String,
    config: ArchitectureConfig,
    layers: Vec<ArchitectureLayer>,
    parameter_inventory: Vec<ParameterEntry>,
}

#[derive(Debug, Deserialize)]
struct ArchitectureConfig {
    dtype: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    head_dim: usize,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    layer_types: Vec<String>,
    attn_output_gate: bool,
    full_attention_interval: usize,
    model_type: String,
    mamba_ssm_dtype: String,
}

#[derive(Debug, Deserialize)]
struct ArchitectureLayer {
    index: usize,
    declared_type: String,
    module_class: String,
    token_mixer_class: String,
    mlp_class: String,
}

#[derive(Debug, Deserialize)]
struct ParameterEntry {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    numel: u64,
}

impl ArchitectureContract {
    pub(crate) fn validate(&self) -> Result<Vec<LayerKind>, Qwen35Error> {
        require_equal(
            "architecture schema",
            "openkind-qwen35-backbone-architecture/v1",
            &self.schema,
        )?;
        validate_identity(&self.profile_id, &self.bundle_sha256, None)?;
        require_equal("architecture model ID", BACKBONE_ID, &self.base_model)?;
        require_equal(
            "architecture revision",
            BACKBONE_REVISION,
            &self.base_revision,
        )?;
        require_equal(
            "architecture parameter count",
            PARAMETER_COUNT,
            self.parameter_count,
        )?;
        require_equal("model class", "Qwen3_5TextModel", &self.model_class)?;

        let config = &self.config;
        for (field, expected, actual) in [
            ("vocab size", 248_320, config.vocab_size),
            ("hidden size", 2_560, config.hidden_size),
            ("intermediate size", 9_216, config.intermediate_size),
            ("layer count", LAYER_COUNT, config.num_hidden_layers),
            ("attention heads", 16, config.num_attention_heads),
            ("key/value heads", 4, config.num_key_value_heads),
            ("max positions", 262_144, config.max_position_embeddings),
            ("attention head width", 256, config.head_dim),
            ("linear convolution width", 4, config.linear_conv_kernel_dim),
            ("linear key head width", 128, config.linear_key_head_dim),
            ("linear value head width", 128, config.linear_value_head_dim),
            ("linear key heads", 16, config.linear_num_key_heads),
            ("linear value heads", 32, config.linear_num_value_heads),
            ("full attention interval", 4, config.full_attention_interval),
        ] {
            require_equal(field, expected, actual)?;
        }
        require_equal("RMSNorm epsilon", 0.000_001, config.rms_norm_eps)?;
        require_equal("config dtype", "float32", &config.dtype)?;
        require_equal("recurrent state dtype", "float32", &config.mamba_ssm_dtype)?;
        require_equal("model type", "qwen3_5_text", &config.model_type)?;
        require_equal("attention output gate", true, config.attn_output_gate)?;

        let kinds = validate_layers(&config.layer_types, &self.layers)?;
        validate_parameter_inventory(&self.parameter_inventory, self.parameter_count)?;
        Ok(kinds)
    }
}

fn validate_layers(
    declared: &[String],
    layers: &[ArchitectureLayer],
) -> Result<Vec<LayerKind>, Qwen35Error> {
    if declared.len() != LAYER_COUNT || layers.len() != LAYER_COUNT {
        return Err(Qwen35Error::InvalidInput(format!(
            "architecture requires {LAYER_COUNT} declared and described layers"
        )));
    }
    let mut kinds = Vec::with_capacity(LAYER_COUNT);
    for (index, (declared, layer)) in declared.iter().zip(layers).enumerate() {
        let expected = if index % 4 == 3 {
            (
                LayerKind::FullAttention,
                "full_attention",
                "Qwen3_5Attention",
            )
        } else {
            (
                LayerKind::LinearAttention,
                "linear_attention",
                "Qwen3_5GatedDeltaNet",
            )
        };
        require_equal("layer index", index, layer.index)?;
        require_equal("declared layer type", expected.1, declared)?;
        require_equal("layer descriptor type", expected.1, &layer.declared_type)?;
        require_equal(
            "decoder layer class",
            "Qwen3_5DecoderLayer",
            &layer.module_class,
        )?;
        require_equal("token mixer class", expected.2, &layer.token_mixer_class)?;
        require_equal("MLP class", "Qwen3_5MLP", &layer.mlp_class)?;
        kinds.push(expected.0);
    }
    Ok(kinds)
}

fn validate_parameter_inventory(
    parameters: &[ParameterEntry],
    expected_total: u64,
) -> Result<(), Qwen35Error> {
    require_equal(
        "parameter tensor count",
        PARAMETER_TENSOR_COUNT,
        parameters.len(),
    )?;
    let mut total = 0_u64;
    for parameter in parameters {
        require_equal("parameter dtype", "float32", &parameter.dtype)?;
        let shape_numel = parameter.shape.iter().try_fold(1_u64, |product, size| {
            product
                .checked_mul(*size as u64)
                .ok_or_else(|| Qwen35Error::InvalidTensor {
                    name: parameter.name.clone(),
                    message: "shape element count overflowed u64".to_owned(),
                })
        })?;
        if shape_numel != parameter.numel {
            return Err(Qwen35Error::InvalidTensor {
                name: parameter.name.clone(),
                message: format!(
                    "shape contains {shape_numel} elements, inventory declares {}",
                    parameter.numel
                ),
            });
        }
        total = total.checked_add(parameter.numel).ok_or_else(|| {
            Qwen35Error::InvalidInput("parameter count overflowed u64".to_owned())
        })?;
    }
    require_equal("parameter inventory total", expected_total, total)
}

pub(crate) fn validate_identity(
    profile_id: &str,
    bundle_sha256: &str,
    repository: Option<&str>,
) -> Result<(), Qwen35Error> {
    require_equal("Phase 3B profile ID", PROFILE_ID, profile_id)?;
    require_equal("Phase 3B bundle SHA-256", BUNDLE_SHA256, bundle_sha256)?;
    if let Some(repository) = repository {
        require_equal(
            "Phase 3B reference repository",
            REFERENCE_REPOSITORY,
            repository,
        )?;
    }
    Ok(())
}
