//! Generic full-sequence forward over digest-verified Qwen3.5 text
//! checkpoints.
//!
//! The pinned [`Qwen35Backbone`](super::model::Qwen35Backbone) serves the
//! selected `encoder-state-first` profile and binds its continuation-state
//! identity to that profile. Third-party survey-profile checkpoints (same
//! frozen 32-layer Qwen3.5 text architecture, different weights) only need
//! the forward pass: embedding lookup, 32 decoder layers, and final RMSNorm.
//! This module exposes exactly that over shard paths the caller has already
//! digest-verified in place, reusing the pinned layer implementations so the
//! numerical path cannot drift from the parity-verified oracle.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;

use super::super::Qwen35Error;
use super::embedding::Qwen35Embedding;
use super::geometry::Qwen35Geometry;
use super::layer0::{rms_norm_zero_centered, DecoderLayer};
use super::model::text_resident_decision;

/// Where layer weights come from during forward.
///
/// The default source memory-maps the caller-verified shard paths; the
/// adapter source wraps a caller-built backend (for example a LoRA-merged
/// view over a mapped checkpoint) with the same lazy per-tensor contract.
enum WeightSource {
    Shards(Vec<PathBuf>),
    Adapter(Arc<dyn SimpleBackend>),
}

impl std::fmt::Debug for WeightSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shards(shards) => formatter.debug_tuple("Shards").field(shards).finish(),
            Self::Adapter(_) => formatter.debug_tuple("Adapter").finish(),
        }
    }
}

/// Delegating backend so a shared [`SimpleBackend`] can be boxed per forward
/// (candle's `VarBuilder::from_backend` consumes the backend).
struct SharedBackend(Arc<dyn SimpleBackend>);

impl std::fmt::Debug for SharedBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("SharedBackend").finish()
    }
}

impl SimpleBackend for SharedBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.0.get(s, name, h, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }
}

/// Full-sequence Qwen3.5 text forward over caller-verified checkpoint shards.
///
/// The caller owns artifact trust: every shard path must have been verified
/// against its pinned SHA-256 digest before it reaches this loader. No
/// continuation state, branch lineage, or profile identity is produced; the
/// survey-family engines that build on this type evaluate independent
/// full-sequence prompts and retain nothing between requests.
/// Decoder namespace of the published `Qwen3_5ForConditionalGeneration`
/// checkpoints every pinned profile except the d3 family serves.
const DEFAULT_TENSOR_PREFIX: &[&str] = &["model", "language_model"];

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TextBackbone {
    embedding: Qwen35Embedding,
    source: WeightSource,
    geometry: Qwen35Geometry,
    /// Safetensors namespace of the decoder stack below the shard root, split
    /// on `.`. Published `Qwen3_5ForConditionalGeneration` checkpoints nest
    /// under `["model", "language_model"]`; text-only `Qwen3_5Model` releases
    /// (the d3 family) nest directly under `["language_model"]`.
    tensor_prefix: &'static [&'static str],
    /// Storage dtype of the mapped shards. BF16 shards stay mapped in their
    /// native width and are widened per use inside the layer kernels; F32
    /// (the pinned default) maps directly.
    dtype: DType,
    device: Device,
    /// Decoder layers and final-norm weights held across forwards, when the
    /// process-wide residency policy asked for it at construction. `None`
    /// keeps the per-forward path: re-map the shards and widen one layer at a
    /// time. Both paths run the same layer code on the same tensor values, so
    /// outputs are bit-identical.
    resident: Option<ResidentDecoder>,
}

#[derive(Debug)]
struct ResidentDecoder {
    layers: Vec<DecoderLayer>,
    final_norm: Vec<f32>,
}

impl TextBackbone {
    /// Assemble a backbone from a verified embedding reader and verified
    /// shard paths.
    ///
    /// The geometry must match the checkpoint. Unless the residency policy
    /// keeps the widened weights in memory (see
    /// [`set_resident_weights_policy`](super::model::set_resident_weights_policy)),
    /// every layer is loaded lazily through the memory-mapped shards during
    /// forward, so no weights are held outside each layer's execution.
    pub(crate) fn new(
        embedding: Qwen35Embedding,
        shards: Vec<PathBuf>,
        geometry: Qwen35Geometry,
        device: Device,
    ) -> Self {
        Self::new_with_tensor_prefix(embedding, shards, geometry, DEFAULT_TENSOR_PREFIX, device)
    }

    /// Assemble a backbone whose decoder tensors live under a non-default
    /// safetensors namespace (text-only `Qwen3_5Model` checkpoints).
    #[allow(dead_code)]
    pub(crate) fn new_with_tensor_prefix(
        embedding: Qwen35Embedding,
        shards: Vec<PathBuf>,
        geometry: Qwen35Geometry,
        tensor_prefix: &'static [&'static str],
        device: Device,
    ) -> Self {
        let mut backbone = Self::unsettled(embedding, shards, geometry, DType::F32, device);
        backbone.tensor_prefix = tensor_prefix;
        backbone.settle_residency()
    }

    /// Assemble a backbone whose mapped shards keep `dtype` storage.
    pub(crate) fn new_with_dtype(
        embedding: Qwen35Embedding,
        shards: Vec<PathBuf>,
        geometry: Qwen35Geometry,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self::unsettled(embedding, shards, geometry, dtype, device).settle_residency()
    }

    /// Backbone with its final tensor prefix still to be applied and its
    /// residency undecided. Residency reads the tensors, so it must run only
    /// after every constructor has set the prefix.
    fn unsettled(
        embedding: Qwen35Embedding,
        shards: Vec<PathBuf>,
        geometry: Qwen35Geometry,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            embedding,
            source: WeightSource::Shards(shards),
            geometry,
            tensor_prefix: DEFAULT_TENSOR_PREFIX,
            dtype,
            device,
            resident: None,
        }
    }

    /// Assemble a backbone whose layer weights come from a caller-built
    /// backend (for example a LoRA-merged view over a mapped checkpoint).
    ///
    /// The backend must serve the checkpoint's own full tensor names; the
    /// same lazy per-tensor loading contract applies, and the caller owns
    /// artifact trust exactly as with shard paths.
    #[allow(dead_code)]
    pub(crate) fn new_with_adapter_backend(
        embedding: Qwen35Embedding,
        backend: Arc<dyn SimpleBackend>,
        geometry: Qwen35Geometry,
        device: Device,
    ) -> Self {
        Self {
            embedding,
            source: WeightSource::Adapter(backend),
            geometry,
            tensor_prefix: DEFAULT_TENSOR_PREFIX,
            dtype: DType::F32,
            device,
            resident: None,
        }
        .settle_residency()
    }

    /// Apply the process-wide residency policy once, after construction.
    ///
    /// Every failure falls back to the per-forward path, which reports the
    /// same error on its first forward, so a declined or failed request for
    /// residency never changes an answer.
    fn settle_residency(mut self) -> Self {
        // BF16-stored backbones keep their mapped view and widen per matmul
        // inside the kernels, so a resident copy would hold the same bytes.
        if self.dtype != DType::F32 {
            return self;
        }
        let estimate = self.resident_estimate_bytes();
        match text_resident_decision(estimate, matches!(self.device, Device::Cpu)) {
            Ok(false) => {}
            Ok(true) => match self.load_resident() {
                Ok(resident) => self.resident = Some(resident),
                Err(error) => tracing::warn!(
                    %error,
                    "text backbone could not keep its decoder weights resident; each forward will load them"
                ),
            },
            Err(reason) => tracing::warn!(
                %reason,
                "not keeping text backbone decoder weights resident; each forward will load them (set --qwen35-resident-weights on to force)"
            ),
        }
        self
    }

    /// Upper bound on the FP32 bytes residency would hold, from the mapped
    /// shards' on-disk size widened from BF16. It includes the embedding
    /// table and errs high. An adapter source has no size to read.
    fn resident_estimate_bytes(&self) -> Option<u64> {
        let WeightSource::Shards(shards) = &self.source else {
            return None;
        };
        shards.iter().try_fold(0_u64, |total, shard| {
            let length = std::fs::metadata(shard).ok()?.len();
            Some(total.saturating_add(length.saturating_mul(2)))
        })
    }

    /// Load every decoder layer and the final norm once, exactly as a forward
    /// would, and keep them.
    fn load_resident(&self) -> Result<ResidentDecoder, Qwen35Error> {
        let variables = self.variables()?;
        let layers_root = variables.pp("layers");
        let layers = (0..self.geometry.layer_count)
            .map(|layer_index| {
                DecoderLayer::load(&layers_root, layer_index, &self.device, self.geometry)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_norm = self.final_norm_weights(&variables)?;
        let held: usize = layers.iter().map(DecoderLayer::resident_bytes).sum();
        tracing::info!(
            layers = layers.len(),
            resident_bytes = held,
            "text backbone decoder weights are resident"
        );
        Ok(ResidentDecoder { layers, final_norm })
    }

    /// Force residency regardless of policy, for tests that compare both
    /// paths.
    #[cfg(test)]
    fn with_resident_weights(mut self) -> Result<Self, Qwen35Error> {
        self.resident = Some(self.load_resident()?);
        Ok(self)
    }

    /// Force the per-forward path regardless of policy, so a test's baseline
    /// cannot be flipped by another test that sets the global policy.
    #[cfg(test)]
    fn without_resident_weights(mut self) -> Self {
        self.resident = None;
        self
    }

    /// FP32-widened final RMSNorm weight.
    fn final_norm_weights(&self, variables: &VarBuilder<'_>) -> Result<Vec<f32>, Qwen35Error> {
        Ok(variables
            .get(self.geometry.hidden_size, "norm.weight")?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?)
    }

    /// Decoder-scoped weight view over the shards or the adapter backend.
    fn variables(&self) -> Result<VarBuilder<'static>, Qwen35Error> {
        let device = self.device.clone();
        // SAFETY: the caller verified the immutable size and SHA-256 of every
        // read-only shard before constructing this backbone. The VarBuilder
        // owns the mapped tensor storage.
        let variables = match &self.source {
            WeightSource::Shards(shards) => unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &shards
                        .iter()
                        .map(|shard| shard.as_path())
                        .collect::<Vec<_>>(),
                    self.dtype,
                    &device,
                )?
            },
            WeightSource::Adapter(backend) => VarBuilder::from_backend(
                Box::new(SharedBackend(backend.clone())),
                self.dtype,
                device,
            ),
        };
        let mut variables = variables;
        for namespace in self.tensor_prefix {
            variables = variables.pp(namespace);
        }
        Ok(variables)
    }

    /// Run embedding, all decoder layers, and final RMSNorm in FP32, returning
    /// the row-major `[token_count, 2560]` final-norm hidden states.
    ///
    /// The tied output embedding is intentionally NOT applied: survey-profile
    /// readouts select their own output rows through
    /// [`Self::embedding_rows`].
    #[allow(dead_code)]
    pub(crate) fn forward_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>, Qwen35Error> {
        self.forward_hidden_with_check(input_ids, || Ok(()))
    }

    pub(crate) fn forward_hidden_with_check<E>(
        &self,
        input_ids: &[u32],
        check: impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<f32>, E>
    where
        E: From<Qwen35Error>,
    {
        self.forward_hidden_checked(input_ids, check, false, false)
    }

    /// Final-norm hidden state of the last token only, `[hidden_size]`.
    ///
    /// Equivalent to the last row of [`Self::forward_hidden_with_check`] for
    /// callers that read nothing else. The last decoder layer then runs its
    /// query path, output projection, and MLP for that one row. Earlier layers
    /// still run every row, because the next layer needs all their outputs.
    /// The result matches the full pass up to the matmul's summation order at
    /// one row against many, so a caller holds it to its probability tolerance
    /// rather than to bit identity.
    pub(crate) fn forward_last_hidden_with_check<E>(
        &self,
        input_ids: &[u32],
        check: impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<f32>, E>
    where
        E: From<Qwen35Error>,
    {
        self.forward_hidden_checked(input_ids, check, false, true)
    }

    /// [`Self::forward_last_hidden_with_check`] that also runs `check` inside
    /// the full-attention kernel, not only between layers.
    ///
    /// The scalar attention kernel is quadratic in the prompt length, so on
    /// long prompts one layer can outlast a deadline or disconnect by a wide
    /// margin. Numerics are identical; only cancellation granularity
    /// changes, which is why callers opt in per family.
    pub(crate) fn forward_last_hidden_with_attention_check<E>(
        &self,
        input_ids: &[u32],
        check: impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<f32>, E>
    where
        E: From<Qwen35Error>,
    {
        self.forward_hidden_checked(input_ids, check, true, true)
    }

    fn forward_hidden_checked<E>(
        &self,
        input_ids: &[u32],
        mut check: impl FnMut() -> Result<(), E>,
        check_inside_attention: bool,
        last_row_only: bool,
    ) -> Result<Vec<f32>, E>
    where
        E: From<Qwen35Error>,
    {
        check()?;
        if input_ids.is_empty() {
            return Err(Qwen35Error::InvalidInput(
                "text backbone input must contain at least one token".to_owned(),
            )
            .into());
        }
        if matches!(self.source, WeightSource::Shards(ref shards) if shards.is_empty()) {
            return Err(Qwen35Error::InvalidInput(
                "text backbone requires at least one verified shard".to_owned(),
            )
            .into());
        }
        let embedding = self.embedding.embed(input_ids).map_err(E::from)?;
        let token_count = embedding.token_count();
        let mut hidden = embedding.values().to_vec();
        let device = self.device.clone();
        // Resident weights skip the per-forward shard mapping and widening.
        let transient = match self.resident {
            Some(_) => None,
            None => Some(self.variables().map_err(E::from)?),
        };
        let layers_root = transient.as_ref().map(|variables| variables.pp("layers"));
        let trace_layers = std::env::var_os("OPENKIND_QWEN35_LAYER_TRACE").is_some();
        // Rows in `hidden`: the whole sequence until a last-row layer prunes it.
        let mut rows = token_count;
        for layer_index in 0..self.geometry.layer_count {
            check()?;
            let loaded;
            let layer = match (&self.resident, &layers_root) {
                (Some(resident), _) => &resident.layers[layer_index],
                (None, Some(root)) => {
                    loaded = DecoderLayer::load(root, layer_index, &device, self.geometry)
                        .map_err(E::from)?;
                    &loaded
                }
                (None, None) => unreachable!("transient weights are built when none are resident"),
            };
            hidden = if last_row_only && layer_index + 1 == self.geometry.layer_count {
                // Nothing reads the other rows past this layer.
                rows = 1;
                if check_inside_attention {
                    layer.forward_last_row_checked(
                        &hidden,
                        token_count,
                        layer_index,
                        Some(&mut check as &mut dyn FnMut() -> Result<(), E>),
                    )?
                } else {
                    layer.forward_last_row_checked::<E>(&hidden, token_count, layer_index, None)?
                }
            } else if check_inside_attention {
                layer.forward_checked(&hidden, token_count, layer_index, &mut check)?
            } else {
                layer
                    .forward(&hidden, token_count, layer_index)
                    .map_err(E::from)?
            };
            if trace_layers {
                let start = (rows - 1) * self.geometry.hidden_size;
                let head: Vec<f32> = hidden[start..start + 4].to_vec();
                eprintln!("trace: layer {layer_index} last {head:?}");
            }
            check()?;
        }
        let loaded_norm;
        let norm: &[f32] = match (&self.resident, &transient) {
            (Some(resident), _) => &resident.final_norm,
            (None, Some(variables)) => {
                loaded_norm = self.final_norm_weights(variables).map_err(E::from)?;
                &loaded_norm
            }
            (None, None) => unreachable!("transient weights are built when none are resident"),
        };
        let final_values = rms_norm_zero_centered(&hidden, rows, self.geometry.hidden_size, norm);
        if let Some((index, value)) = final_values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(Qwen35Error::Numerical(format!(
                "text backbone final_norm output element {index} is not finite: {value}"
            ))
            .into());
        }
        check()?;
        Ok(final_values)
    }

    /// FP32-widened embedding rows for the given token IDs, row-major
    /// `[ids.len(), hidden_size]`.
    ///
    /// For tied-head profiles these rows are the output-head weights of the
    /// requested vocabulary items.
    pub(crate) fn embedding_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, Qwen35Error> {
        self.embedding
            .embed(token_ids)
            .map(|rows| rows.values().to_vec())
    }
}

#[cfg(test)]
mod resident_tests {
    use std::collections::HashMap;
    use std::path::Path;

    use candle_core::Tensor;

    use super::super::embedding::EmbeddingLayout;
    use super::*;

    const VOCAB: usize = 32;
    const PREFIX: &[&str] = &["language_model"];

    /// Small hybrid geometry that exercises both mixers: layers 0-2 are
    /// gated DeltaNet and layer 3 is full attention.
    fn tiny_geometry() -> Qwen35Geometry {
        Qwen35Geometry {
            hidden_size: 16,
            intermediate_size: 24,
            layer_count: 4,
            full_attention_interval: 4,
            full_attention_layers: Qwen35Geometry::interval_layers(4, 4),
            key_heads: 2,
            value_heads: 4,
            head_dim: 8,
            conv_kernel: 4,
            attention_heads: 2,
            kv_heads: 1,
            attention_head_dim: 8,
            rms_epsilon: 1e-6,
            causal: true,
        }
    }

    /// Deterministic values in `[-scale, scale)`, distinct per tensor.
    fn values(count: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1_u32 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Write a BF16 checkpoint in the layout `TextBackbone` loads, the same
    /// storage dtype the published shards use, so the F32 request below goes
    /// through candle's widening exactly as on a real install.
    fn write_checkpoint(path: &Path, geometry: Qwen35Geometry) {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut seed = 0_u32;
        let mut add = |name: String, shape: &[usize], scale: f32| {
            seed += 1;
            let count: usize = shape.iter().product();
            let tensor = Tensor::from_vec(values(count, seed, scale), shape, &Device::Cpu)
                .and_then(|tensor| tensor.to_dtype(DType::BF16))
                .expect("synthetic tensor");
            tensors.insert(format!("language_model.{name}"), tensor);
        };
        let hidden = geometry.hidden_size;
        add("embed_tokens.weight".into(), &[VOCAB, hidden], 0.5);
        add("norm.weight".into(), &[hidden], 0.1);
        for layer in 0..geometry.layer_count {
            let base = format!("layers.{layer}");
            add(format!("{base}.input_layernorm.weight"), &[hidden], 0.1);
            add(
                format!("{base}.post_attention_layernorm.weight"),
                &[hidden],
                0.1,
            );
            for (name, shape) in [
                ("mlp.gate_proj", [geometry.intermediate_size, hidden]),
                ("mlp.up_proj", [geometry.intermediate_size, hidden]),
                ("mlp.down_proj", [hidden, geometry.intermediate_size]),
            ] {
                add(format!("{base}.{name}.weight"), &shape, 0.2);
            }
            if geometry.is_full_attention(layer) {
                add(
                    format!("{base}.self_attn.q_proj.weight"),
                    &[geometry.attention_size() * 2, hidden],
                    0.2,
                );
                add(
                    format!("{base}.self_attn.k_proj.weight"),
                    &[geometry.kv_size(), hidden],
                    0.2,
                );
                add(
                    format!("{base}.self_attn.v_proj.weight"),
                    &[geometry.kv_size(), hidden],
                    0.2,
                );
                add(
                    format!("{base}.self_attn.q_norm.weight"),
                    &[geometry.attention_head_dim],
                    0.1,
                );
                add(
                    format!("{base}.self_attn.k_norm.weight"),
                    &[geometry.attention_head_dim],
                    0.1,
                );
                add(
                    format!("{base}.self_attn.o_proj.weight"),
                    &[hidden, geometry.attention_size()],
                    0.2,
                );
            } else {
                add(
                    format!("{base}.linear_attn.in_proj_qkv.weight"),
                    &[geometry.qkv_size(), hidden],
                    0.2,
                );
                add(
                    format!("{base}.linear_attn.in_proj_z.weight"),
                    &[geometry.value_size(), hidden],
                    0.2,
                );
                add(
                    format!("{base}.linear_attn.in_proj_b.weight"),
                    &[geometry.value_heads, hidden],
                    0.2,
                );
                add(
                    format!("{base}.linear_attn.in_proj_a.weight"),
                    &[geometry.value_heads, hidden],
                    0.2,
                );
                add(
                    format!("{base}.linear_attn.conv1d.weight"),
                    &[geometry.qkv_size(), 1, geometry.conv_kernel],
                    0.3,
                );
                add(
                    format!("{base}.linear_attn.dt_bias"),
                    &[geometry.value_heads],
                    0.1,
                );
                add(
                    format!("{base}.linear_attn.A_log"),
                    &[geometry.value_heads],
                    0.1,
                );
                add(
                    format!("{base}.linear_attn.norm.weight"),
                    &[geometry.head_dim],
                    0.1,
                );
                add(
                    format!("{base}.linear_attn.out_proj.weight"),
                    &[hidden, geometry.value_size()],
                    0.2,
                );
            }
        }
        candle_core::safetensors::save(&tensors, path).expect("write synthetic checkpoint");
    }

    fn backbone(path: &Path, geometry: Qwen35Geometry) -> TextBackbone {
        let layout = EmbeddingLayout::read_named(
            path,
            "language_model.embed_tokens.weight",
            VOCAB,
            geometry.hidden_size,
        )
        .expect("embedding layout");
        let embedding = Qwen35Embedding::from_layout(path.to_path_buf(), layout);
        TextBackbone::new_with_tensor_prefix(
            embedding,
            vec![path.to_path_buf()],
            geometry,
            PREFIX,
            Device::Cpu,
        )
    }

    fn assert_bit_identical(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "{what}[{index}]: {a} vs {e}");
        }
    }

    #[test]
    fn resident_forward_is_bit_identical_to_the_per_forward_path() {
        let geometry = tiny_geometry();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("synthetic.safetensors");
        write_checkpoint(&path, geometry);

        let per_forward = backbone(&path, geometry).without_resident_weights();
        let resident = backbone(&path, geometry)
            .with_resident_weights()
            .expect("resident load");
        assert!(per_forward.resident.is_none());
        assert_eq!(
            resident.resident.as_ref().map(|held| held.layers.len()),
            Some(geometry.layer_count)
        );

        // Distinct lengths exercise the prefill shape and a single row.
        for ids in [&[3_u32, 1, 4, 1, 5, 9, 2][..], &[7][..]] {
            let expected = per_forward.forward_hidden(ids).expect("per-forward");
            let first = resident.forward_hidden(ids).expect("resident");
            assert_bit_identical(&first, &expected, "resident");
            assert!(first.iter().any(|value| *value != 0.0), "degenerate output");
            // The weights are shared across forwards and must not be mutated.
            let again = resident.forward_hidden(ids).expect("resident again");
            assert_bit_identical(&again, &expected, "resident repeat");
            // The attention-checked variant takes the same layers.
            let checked = resident
                .forward_hidden_checked(ids, || -> Result<(), Qwen35Error> { Ok(()) }, true, false)
                .expect("resident checked");
            assert_bit_identical(&checked, &expected, "resident checked");
        }
    }

    #[test]
    fn residency_is_skipped_for_bf16_storage() {
        let geometry = tiny_geometry();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("synthetic.safetensors");
        write_checkpoint(&path, geometry);
        let layout = EmbeddingLayout::read_named(
            &path,
            "language_model.embed_tokens.weight",
            VOCAB,
            geometry.hidden_size,
        )
        .expect("embedding layout");
        let embedding = Qwen35Embedding::from_layout(path.clone(), layout);
        let bf16 =
            TextBackbone::new_with_dtype(embedding, vec![path], geometry, DType::BF16, Device::Cpu);
        // Whatever the process-wide policy says, a BF16-mapped backbone widens
        // per matmul, so there is nothing to keep.
        assert!(bf16.resident.is_none());
    }

    /// Geometry whose last layer is a DeltaNet mixer: with an interval of 4
    /// and three layers, no layer is full attention.
    fn linear_last_geometry() -> Qwen35Geometry {
        Qwen35Geometry {
            layer_count: 3,
            full_attention_layers: Qwen35Geometry::interval_layers(4, 3),
            ..tiny_geometry()
        }
    }

    fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// The last-row path must reproduce the last row of the full pass. The
    /// two differ only in the matmul's summation order at one row against
    /// many, so they are held to a tight tolerance rather than bit identity.
    #[test]
    fn last_row_forward_matches_the_last_row_of_the_full_pass() {
        for (name, base) in [
            ("full-attention last layer", tiny_geometry()),
            ("deltanet last layer", linear_last_geometry()),
        ] {
            for causal in [true, false] {
                let geometry = base.with_causal(causal);
                let directory = tempfile::tempdir().expect("tempdir");
                let path = directory.path().join("synthetic.safetensors");
                write_checkpoint(&path, geometry);
                let backbone = backbone(&path, geometry).without_resident_weights();
                let resident =
                    super::TextBackbone::with_resident_weights(self::backbone(&path, geometry))
                        .expect("resident load");
                let hidden = geometry.hidden_size;
                for ids in [&[3_u32, 1, 4, 1, 5, 9, 2, 6, 5][..], &[7, 7][..], &[11][..]] {
                    let full = backbone.forward_hidden(ids).expect("full");
                    let expected = &full[(ids.len() - 1) * hidden..];
                    let tag = format!("{name} causal={causal} rows={}", ids.len());
                    let pruned = backbone
                        .forward_last_hidden_with_check(ids, || -> Result<(), Qwen35Error> {
                            Ok(())
                        })
                        .expect("last row");
                    assert_eq!(pruned.len(), hidden, "{tag}");
                    let difference = max_abs_difference(&pruned, expected);
                    assert!(difference <= 1e-5, "{tag}: max diff {difference}");
                    assert!(
                        pruned.iter().any(|value| *value != 0.0),
                        "{tag}: degenerate"
                    );

                    let checked = backbone
                        .forward_last_hidden_with_attention_check(
                            ids,
                            || -> Result<(), Qwen35Error> { Ok(()) },
                        )
                        .expect("last row, attention check");
                    assert_eq!(checked, pruned, "{tag}: checked variant is the same kernel");

                    // Residency does not change the pruned result either.
                    let from_resident = resident
                        .forward_last_hidden_with_check(ids, || -> Result<(), Qwen35Error> {
                            Ok(())
                        })
                        .expect("resident last row");
                    assert_bit_identical(&from_resident, &pruned, &tag);
                }
            }
        }
    }

    #[test]
    fn last_row_forward_runs_its_cancellation_checks() {
        let geometry = tiny_geometry();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("synthetic.safetensors");
        write_checkpoint(&path, geometry);
        let backbone = backbone(&path, geometry).without_resident_weights();
        let ids = [3_u32, 1, 4, 1, 5];
        let mut calls = 0_usize;
        backbone
            .forward_last_hidden_with_attention_check(&ids, || -> Result<(), Qwen35Error> {
                calls += 1;
                Ok(())
            })
            .expect("last row");
        assert!(calls >= geometry.layer_count, "checks ran {calls} times");
        // A failing check stops the pass, including inside the pruned layer.
        let mut remaining = geometry.layer_count * 2;
        let outcome = backbone.forward_last_hidden_with_attention_check(
            &ids,
            || -> Result<(), Qwen35Error> {
                if remaining == 0 {
                    return Err(Qwen35Error::InvalidInput("cancelled".into()));
                }
                remaining -= 1;
                Ok(())
            },
        );
        assert!(
            matches!(outcome, Err(Qwen35Error::InvalidInput(message)) if message == "cancelled")
        );
    }
}
