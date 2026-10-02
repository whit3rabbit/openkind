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
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TextBackbone {
    embedding: Qwen35Embedding,
    source: WeightSource,
    geometry: Qwen35Geometry,
    /// Storage dtype of the mapped shards. BF16 shards stay mapped in their
    /// native width and are widened per use inside the layer kernels; F32
    /// (the pinned default) maps directly.
    dtype: DType,
    device: Device,
}

impl TextBackbone {
    /// Assemble a backbone from a verified embedding reader and verified
    /// shard paths.
    ///
    /// The geometry must match the checkpoint; every layer is loaded lazily
    /// through the memory-mapped shards during forward, so no weights are
    /// copied or held outside each layer's execution.
    pub(crate) fn new(
        embedding: Qwen35Embedding,
        shards: Vec<PathBuf>,
        geometry: Qwen35Geometry,
        device: Device,
    ) -> Self {
        Self::new_with_dtype(embedding, shards, geometry, DType::F32, device)
    }

    /// Assemble a backbone whose mapped shards keep `dtype` storage.
    pub(crate) fn new_with_dtype(
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
            dtype,
            device,
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
            dtype: DType::F32,
            device,
        }
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
        mut check: impl FnMut() -> Result<(), E>,
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
                )
                .map_err(Qwen35Error::from)
                .map_err(E::from)?
            },
            WeightSource::Adapter(backend) => VarBuilder::from_backend(
                Box::new(SharedBackend(backend.clone())),
                self.dtype,
                device.clone(),
            ),
        }
        .pp("model")
        .pp("language_model");
        let layers = variables.pp("layers");
        for layer_index in 0..self.geometry.layer_count {
            check()?;
            let layer = DecoderLayer::load(&layers, layer_index, &device, self.geometry)
                .map_err(E::from)?;
            hidden = layer
                .forward(&hidden, token_count, layer_index)
                .map_err(E::from)?;
            if std::env::var("OPENKIND_QWEN35_LAYER_TRACE").is_ok() {
                let start = (token_count - 1) * self.geometry.hidden_size;
                let head: Vec<f32> = hidden[start..start + 4].to_vec();
                eprintln!("trace: layer {layer_index} last {head:?}");
            }
            check()?;
        }
        let norm = variables
            .get(self.geometry.hidden_size, "norm.weight")
            .and_then(|value| value.flatten_all())
            .and_then(|value| value.to_dtype(DType::F32))
            .and_then(|value| value.to_vec1::<f32>())
            .map_err(Qwen35Error::from)
            .map_err(E::from)?;
        let final_values =
            rms_norm_zero_centered(&hidden, token_count, self.geometry.hidden_size, &norm);
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
