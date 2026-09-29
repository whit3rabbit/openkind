//! The `TextEmbedder` contract and the dependency-free hash embedder.
//!
//! Model-backed embedders live in [`super::bert_encoder`] (candle CPU) and
//! [`super::mlx_bert_encoder`] (MLX). The hash embedder needs no weights and
//! no downloads: it is the fallback encoder (`--proxy-cache-encoder hash`)
//! and the deterministic encoder used by the test battery.

use sha2::{Digest, Sha256};

use super::ProxyCacheError;

/// A frozen text encoder producing L2-normalized float32 embeddings.
///
/// `id` is the encoder's lineage identity: rows are keyed by it, so changing
/// it starts a new training lineage for every task (the encoder id is stored
/// alongside every sample).
pub trait TextEmbedder: Send + Sync {
    /// Stable lineage identity including a content hash of the weights.
    fn id(&self) -> String;
    /// Embedding dimensionality.
    fn dim(&self) -> usize;
    /// Embed a batch of texts into L2-normalized vectors (one per input).
    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProxyCacheError>;

    /// Human-readable backend identity for telemetry (`encoder-embedding/candle-fp32`).
    fn backend_id(&self) -> &str {
        "proxy-cache-encoder"
    }
}

/// L2-normalize a vector in place; the zero vector maps to itself.
pub(crate) fn l2_normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 1e-12 {
        for value in vector.iter_mut() {
            *value = (*value as f64 / norm) as f32;
        }
    }
}

/// Keyed hashed bag of words + bigrams, L2-normalized.
///
/// Each token (and adjacent token pair) is hashed into one of `dim` slots
/// with a ±1 sign; collisions are the dimensionality budget. Deterministic
/// across processes for a given (dim, seed) pair, so a corpus embedded on
/// different machines is comparable.
pub struct HashEmbedder {
    dim: usize,
    seed: u64,
    bigrams: bool,
    id: String,
}

impl HashEmbedder {
    /// Build a hash embedder. `dim` must be non-zero.
    pub fn new(dim: usize, seed: u64, bigrams: bool) -> Result<Self, ProxyCacheError> {
        if dim == 0 {
            return Err(ProxyCacheError::Contract(
                "hash embedder dimension must be non-zero".into(),
            ));
        }
        let id = format!("hash-{dim}-{seed}{}", if bigrams { "-bi" } else { "-uni" });
        Ok(Self {
            dim,
            seed,
            bigrams,
            id,
        })
    }
}

impl TextEmbedder for HashEmbedder {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProxyCacheError> {
        let mut batch = Vec::with_capacity(texts.len());
        for text in texts {
            let mut slots = vec![0.0_f32; self.dim];
            let tokens = tokenize(text);
            for token in &tokens {
                hash_feature(token, self.seed, self.dim, &mut slots);
            }
            if self.bigrams {
                for window in tokens.windows(2) {
                    let pair = format!("{} {}", window[0], window[1]);
                    hash_feature(&pair, self.seed, self.dim, &mut slots);
                }
            }
            l2_normalize(&mut slots);
            batch.push(slots);
        }
        Ok(batch)
    }
}

/// Lowercase alphanumeric tokenization; runs of non-alphanumerics split.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn hash_feature(feature: &str, seed: u64, dim: usize, slots: &mut [f32]) {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update([0]);
    hasher.update(feature.as_bytes());
    let digest = hasher.finalize();
    let slot = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize % dim;
    let sign = if digest[5] & 1 == 0 { 1.0 } else { -1.0 };
    slots[slot] += sign;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_deterministic_l2_normalized() {
        let embedder = HashEmbedder::new(512, 0, true).unwrap();
        let texts = vec!["hello world".to_string(), "hello world".to_string()];
        let batch = embedder.encode(&texts).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], batch[1]);
        assert_eq!(batch[0].len(), 512);
        let norm: f32 = batch[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
    }

    #[test]
    fn different_text_diverges_and_seed_changes_embedding() {
        let embedder = HashEmbedder::new(512, 0, true).unwrap();
        let batch = embedder
            .encode(&["hello world".to_string(), "goodbye world".to_string()])
            .unwrap();
        assert_ne!(batch[0], batch[1]);

        let other = HashEmbedder::new(512, 7, true).unwrap();
        let batch_other = other.encode(&["hello world".to_string()]).unwrap();
        assert_ne!(batch[0], batch_other[0]);
    }

    #[test]
    fn zero_text_maps_to_zero_vector() {
        let embedder = HashEmbedder::new(64, 0, true).unwrap();
        let batch = embedder.encode(&["".to_string()]).unwrap();
        assert!(batch[0].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn rejects_zero_dim() {
        assert!(HashEmbedder::new(0, 0, true).is_err());
    }
}
