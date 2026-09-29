//! candle CPU embedder for BERT-family sentence-transformer checkpoints
//! (`BAAI/bge-small-en-v1.5` and its base/large siblings).
//!
//! The checkpoint stays in place and is memory-mapped FP32; pooling is CLS
//! (the BGE convention) and outputs are L2-normalized, matching the
//! reference proxy design's encoder contract. The encoder `id` pins the
//! checkpoint content hash, so a different checkpoint starts a new training
//! lineage.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::BertModel;
use tokenizers::Tokenizer;

use super::encoder::{l2_normalize, TextEmbedder};
use super::ProxyCacheError;

/// Requested context window per text (the reference default).
pub const MAX_LENGTH: usize = 256;

/// Filesystem layout of one BERT embedding checkpoint.
#[derive(Debug, Clone)]
pub struct BertEmbedderArtifacts {
    /// `model.safetensors`.
    pub checkpoint: std::path::PathBuf,
    /// `config.json`.
    pub config_json: std::path::PathBuf,
    /// `tokenizer.json`.
    pub tokenizer_json: std::path::PathBuf,
    /// Repository identity baked into the encoder id (e.g. `BAAI/bge-small-en-v1.5`).
    pub repository: String,
}

impl BertEmbedderArtifacts {
    /// Standard single-root layout produced by the model store.
    pub fn from_model_root(root: &Path, repository: &str) -> Self {
        Self {
            checkpoint: root.join("model.safetensors"),
            config_json: root.join("config.json"),
            tokenizer_json: root.join("tokenizer.json"),
            repository: repository.to_owned(),
        }
    }

    /// Verify all three artifacts exist and read the checkpoint's content
    /// hash (the encoder id's lineage component).
    pub fn verify(&self) -> Result<String, ProxyCacheError> {
        for path in [&self.checkpoint, &self.config_json, &self.tokenizer_json] {
            if !path.is_file() {
                return Err(ProxyCacheError::Encoder(format!(
                    "embedding checkpoint artifact {} is missing",
                    path.display()
                )));
            }
        }
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        let mut file = std::fs::File::open(&self.checkpoint).map_err(|error| {
            ProxyCacheError::Encoder(format!("open {}: {error}", self.checkpoint.display()))
        })?;
        std::io::copy(&mut file, &mut hasher)
            .map_err(|error| ProxyCacheError::Encoder(format!("hash checkpoint: {error}")))?;
        let digest = hasher.finalize();
        Ok(digest.iter().take(4).map(|b| format!("{b:02x}")).collect())
    }
}

/// candle-backed BERT sentence embedder.
pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    id: String,
    dim: usize,
}

impl BertEmbedder {
    /// Load and verify the checkpoint (mmap FP32 CPU).
    pub fn load(artifacts: &BertEmbedderArtifacts) -> Result<Self, ProxyCacheError> {
        let content_hash = artifacts.verify()?;

        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&artifacts.config_json).map_err(|error| {
                ProxyCacheError::Encoder(format!(
                    "read {}: {error}",
                    artifacts.config_json.display()
                ))
            })?,
        )
        .map_err(|error| ProxyCacheError::Encoder(format!("decode config.json: {error}")))?;
        let architectures = config_json
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let is_bert = architectures
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|arch| arch.eq_ignore_ascii_case("BertModel"))
            || config_json
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|kind| kind.eq_ignore_ascii_case("bert"));
        if !is_bert {
            return Err(ProxyCacheError::Encoder(format!(
                "checkpoint {} is not a BertModel checkpoint",
                artifacts.config_json.display()
            )));
        }

        let dim = config_json
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                ProxyCacheError::Encoder("bert config.json is missing hidden_size".into())
            })? as usize;
        let config: candle_transformers::models::bert::Config = serde_json::from_value(config_json)
            .map_err(|error| ProxyCacheError::Encoder(format!("decode bert config: {error}")))?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&artifacts.checkpoint),
                DType::F32,
                &device,
            )
        }
        .map_err(candle_error)?;
        let model = BertModel::load(vb, &config).map_err(candle_error)?;

        let mut tokenizer = Tokenizer::from_file(&artifacts.tokenizer_json).map_err(|error| {
            ProxyCacheError::Encoder(format!(
                "load tokenizer {}: {error}",
                artifacts.tokenizer_json.display()
            ))
        })?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|error| ProxyCacheError::Encoder(format!("set truncation: {error}")))?;

        let id = format!(
            "candle-bert:{}:cls:{}:{content_hash}",
            artifacts.repository, MAX_LENGTH
        );
        Ok(Self {
            model,
            tokenizer,
            device,
            id,
            dim,
        })
    }

    /// Encode one batch of already-tokenized inputs (used by tests too).
    fn forward_batch(
        &self,
        encodings: &[tokenizers::Encoding],
    ) -> Result<Vec<Vec<f32>>, ProxyCacheError> {
        if encodings.is_empty() {
            return Ok(Vec::new());
        }
        let batch = encodings.len();
        let width = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(0)
            .max(1);
        let mut input_ids = vec![0_u32; batch * width];
        let mut token_type_ids = vec![0_u32; batch * width];
        let mut attention_mask = vec![0_u32; batch * width];
        for (row, encoding) in encodings.iter().enumerate() {
            for (column, &token) in encoding.get_ids().iter().enumerate() {
                input_ids[row * width + column] = token;
                attention_mask[row * width + column] = 1;
            }
            for (column, &type_id) in encoding.get_type_ids().iter().enumerate() {
                let index = row * width + column;
                if index < token_type_ids.len() {
                    token_type_ids[index] = type_id;
                }
            }
        }
        let shape = (batch, width);
        let input_ids = Tensor::from_vec(input_ids, shape, &self.device).map_err(candle_error)?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids, shape, &self.device).map_err(candle_error)?;
        let attention_mask =
            Tensor::from_vec(attention_mask, shape, &self.device).map_err(candle_error)?;
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(candle_error)?;
        // CLS pooling: hidden state of position 0, per row.
        let cls = hidden
            .i((.., 0))
            .and_then(|row| row.to_dtype(DType::F32))
            .map_err(candle_error)?;
        let rows: Vec<Vec<f32>> = cls.to_vec2().map_err(candle_error)?;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
        for vector in rows {
            let mut owned = vector;
            l2_normalize(&mut owned);
            out.push(owned);
        }
        Ok(out)
    }
}

fn candle_error(error: candle_core::Error) -> ProxyCacheError {
    ProxyCacheError::Encoder(format!("candle execution failed: {error}"))
}

impl TextEmbedder for BertEmbedder {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProxyCacheError> {
        let mut encodings = Vec::with_capacity(texts.len());
        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|error| ProxyCacheError::Encoder(format!("tokenize: {error}")))?;
            encodings.push(encoding);
        }
        self.forward_batch(&encodings)
    }

    fn backend_id(&self) -> &str {
        "encoder-embedding/candle-fp32"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_root() -> Option<std::path::PathBuf> {
        std::env::var_os("OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT")
            .map(std::path::PathBuf::from)
            .filter(|root| root.is_dir())
    }

    /// Env-gated parity test: requires a local bge-small checkpoint
    /// (`model.safetensors`, `config.json`, `tokenizer.json`). Never runs in
    /// CI or the default battery — tests must not download model assets.
    #[test]
    fn embeds_text_with_pinned_checkpoint_when_env_gated() {
        let Some(root) = model_root() else {
            eprintln!("skipping: OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT not set");
            return;
        };
        let artifacts = BertEmbedderArtifacts::from_model_root(&root, "BAAI/bge-small-en-v1.5");
        let embedder = BertEmbedder::load(&artifacts).expect("load bge checkpoint");
        assert_eq!(embedder.dim(), 384);
        assert!(embedder
            .id()
            .starts_with("candle-bert:BAAI/bge-small-en-v1.5:cls:256:"));

        let batch = embedder
            .encode(&vec!["the capital of France is Paris".to_string(); 2])
            .unwrap();
        assert_eq!(batch.len(), 2);
        let norm: f32 = batch[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");

        // Similar sentences should be closer than dissimilar ones.
        let different = embedder
            .encode(&["quantum chromodynamics gauge symmetry".to_string()])
            .unwrap();
        let similarity =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let same_pair = similarity(&batch[0], &batch[1]);
        let cross_pair = similarity(&batch[0], &different[0]);
        assert!(
            same_pair > cross_pair,
            "same {same_pair} vs cross {cross_pair}"
        );
    }
}
