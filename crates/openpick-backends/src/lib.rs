//! `openpick-backends`: Decision engine backend drivers and providers.
//!
//! # Architecture & Responsibilities
//! `openpick-backends` contains driver implementations that fulfill the `DecisionEngine` trait
//! from `openpick-engine`.
//!
//! Per `docs/ARCHITECTURE.md` and `docs/RESEARCH.md`:
//! - **Phase 1**: Features an in-memory, deterministic [`BackendType::Mock`] engine for testing wire protocols.
//! - **Phase 2 & Phase 3**: Implements real inference drivers for Candle (Hugging Face tensors targeting
//!   frozen Qwen3.5-4B + linear decision head), quantized GGUF via llama.cpp, ONNX runtime, and remote provider proxies.

#![warn(missing_docs)]

use std::fmt;

/// Backend engine provider types supported by openpick.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// In-memory mock decision engine.
    Mock,
    /// Remote Jev-compatible HTTP/REST or gRPC proxy backend.
    Remote {
        /// Base URL of the remote API endpoint (e.g. `https://api.typesafe.ai`).
        endpoint: String,
    },
    /// Candle native transformer backend.
    Candle {
        /// Path to the model weights file (e.g. `.safetensors`).
        model_path: String,
    },
    /// GGUF quantized model backend.
    Gguf {
        /// Path to the GGUF quantized model file.
        model_path: String,
    },
    /// ONNX runtime graph backend.
    Onnx {
        /// Path to the ONNX graph model file.
        model_path: String,
    },
}

impl BackendType {
    /// Validate backend configuration parameters to protect against SSRF and invalid paths.
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            BackendType::Mock => Ok(()),
            BackendType::Remote { endpoint } => {
                if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                    return Err("remote backend endpoint must use http:// or https:// scheme");
                }
                Ok(())
            }
            BackendType::Candle { model_path }
            | BackendType::Gguf { model_path }
            | BackendType::Onnx { model_path } => {
                if model_path.is_empty() {
                    return Err("model_path cannot be empty");
                }
                if model_path.contains('\0') {
                    return Err("model_path cannot contain null bytes");
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Mock => write!(f, "mock"),
            BackendType::Remote { endpoint } => write!(f, "remote({endpoint})"),
            BackendType::Candle { model_path } => write!(f, "candle({model_path})"),
            BackendType::Gguf { model_path } => write!(f, "gguf({model_path})"),
            BackendType::Onnx { model_path } => write!(f, "onnx({model_path})"),
        }
    }
}

/// Configuration describing an inference backend to register.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendConfig {
    /// Model alias to expose (e.g. "jev-latest", "mock", "qwen-3b").
    pub alias: String,
    /// Concrete backend driver.
    pub backend: BackendType,
    /// Concurrency or batch limit (None = unbounded).
    pub max_batch_size: Option<usize>,
}

impl BackendConfig {
    /// Construct a new `BackendConfig` with the specified alias and backend driver.
    pub fn new(alias: impl Into<String>, backend: BackendType) -> Self {
        Self {
            alias: alias.into(),
            backend,
            max_batch_size: None,
        }
    }

    /// Set an optional maximum batch size limit for the backend.
    pub fn with_max_batch_size(mut self, limit: usize) -> Self {
        self.max_batch_size = Some(limit);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_type_formatting() {
        assert_eq!(BackendType::Mock.to_string(), "mock");
        assert_eq!(
            BackendType::Remote {
                endpoint: "https://api.typesafe.ai".into()
            }
            .to_string(),
            "remote(https://api.typesafe.ai)"
        );
        assert_eq!(
            BackendType::Candle {
                model_path: "qwen.safetensors".into()
            }
            .to_string(),
            "candle(qwen.safetensors)"
        );
    }

    #[test]
    fn backend_config_builder() {
        let config = BackendConfig::new("my-model", BackendType::Mock).with_max_batch_size(16);
        assert_eq!(config.alias, "my-model");
        assert_eq!(config.backend, BackendType::Mock);
        assert_eq!(config.max_batch_size, Some(16));
    }

    #[test]
    fn backend_type_validation() {
        assert!(BackendType::Mock.validate().is_ok());
        assert!(BackendType::Remote {
            endpoint: "https://api.typesafe.ai".into()
        }
        .validate()
        .is_ok());
        assert!(BackendType::Remote {
            endpoint: "http://localhost:8080".into()
        }
        .validate()
        .is_ok());
        assert!(BackendType::Remote {
            endpoint: "ftp://malicious".into()
        }
        .validate()
        .is_err());
        assert!(BackendType::Remote {
            endpoint: "file:///etc/passwd".into()
        }
        .validate()
        .is_err());

        assert!(BackendType::Candle {
            model_path: "weights.safetensors".into()
        }
        .validate()
        .is_ok());
        assert!(BackendType::Candle {
            model_path: "".into()
        }
        .validate()
        .is_err());
        assert!(BackendType::Candle {
            model_path: "bad\0path".into()
        }
        .validate()
        .is_err());
    }
}
