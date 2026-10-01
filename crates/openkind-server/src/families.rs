//! Daemon configuration and registration for surveyed-family engines.
//!
//! Each family exposes `--<family>-aliases` plus the artifact paths its
//! loader requires. An alias is only served when it appears in `--models`;
//! requesting a family alias without its artifact configuration fails the
//! daemon startup instead of falling back to the mock engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::args::{
    CudaOnlyBackendArg, DecoderLogitQwen35BackendArg, EncoderInstructLabelBackendArg,
    FamilyBackendArg, LayaBackendArg,
};
use anyhow::{bail, Context, Result};
use clap::Parser;
use openkind_backends::families::decider::{DeciderEngine, DeciderEngineConfig, DECIDER_4B};
use openkind_backends::families::decoder_logit_letter::{
    DecoderLetterEngine, DecoderLetterEngineConfig,
};
use openkind_backends::families::decoder_logit_llm::{DecoderLlmEngine, DecoderLlmEngineConfig};
use openkind_backends::families::decoder_logit_qwen3::{
    DecoderLogitQwen3Engine, DecoderLogitQwen3EngineConfig, QWEN3_06B, QWEN3_17B, QWEN3_4B,
};
use openkind_backends::families::decoder_logit_qwen35::{
    DecoderLogitQwen35Engine, DecoderLogitQwen35EngineConfig, PLUMB_4B,
};
use openkind_backends::families::gemma4::{Gemma4DecisionEngine, Gemma4EngineConfig};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::families::decoder_logit_qwen35::{
    DecoderLogitQwen35MlxEngine, DecoderLogitQwen35MlxEngineConfig,
};
use openkind_backends::families::encoder_instruct_label::{
    EncoderInstructLabelEngine, EncoderInstructLabelEngineConfig,
};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::families::encoder_instruct_label::{
    EncoderInstructLabelMlxEngine, EncoderInstructLabelMlxEngineConfig,
};
use openkind_backends::families::encoder_nli::{EncoderNliEngine, EncoderNliEngineConfig};
use openkind_backends::families::kev::{KevEngine, KevEngineConfig};
use openkind_backends::families::laya::{
    LayaEngine, LayaEngineConfig, LAYA_ENGLISH, LAYA_MULTILINGUAL, LAYA_TYPED_DECISIONS,
};
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::families::laya::{LayaMlxEngine, LayaMlxEngineConfig};
use openkind_backends::families::qwen3guard::{Qwen3GuardEngine, Qwen3GuardEngineConfig};
use openkind_backends::families::router_script::ScriptRuleTable;
use openkind_backends::families::schema_scorer::{SchemaScorerEngine, SchemaScorerEngineConfig};
use openkind_backends::families::support::FamilyLimits;
use openkind_backends::families::von::{VonEngine, VonEngineConfig, VON};
use openkind_engine::DecisionEngine;

/// Admission defaults shared by all surveyed-family engines.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FamilyAdmission {
    pub(crate) concurrency: usize,
    pub(crate) queue: usize,
    pub(crate) timeout_ms: u64,
}

impl FamilyAdmission {
    fn limits(self) -> FamilyLimits {
        FamilyLimits {
            max_concurrent_requests: self.concurrency,
            max_queued_requests: self.queue,
            retry_after_ms: 1_000,
            evaluation_timeout: Some(Duration::from_millis(self.timeout_ms)),
        }
    }
}

/// Surveyed-family daemon configuration.
#[derive(Debug, Parser)]
pub(crate) struct FamilyArgs {
    /// Aliases in `--models` that should use the pinned decoder-logit-letter
    /// engine (Qwen2.5-0.5B-Instruct letter readout).
    #[arg(
        long,
        env = "OPENKIND_DECODER_LETTER_ALIASES",
        value_delimiter = ',',
        default_value = "decoder-letter-native"
    )]
    pub(crate) decoder_letter_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`, and
    /// `tokenizer.json` required by decoder-letter aliases.
    #[arg(long, env = "OPENKIND_DECODER_LETTER_MODEL_ROOT")]
    pub(crate) decoder_letter_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned encoder-nli engine
    /// (DistilBERT MNLI entailment readout).
    #[arg(
        long,
        env = "OPENKIND_ENCODER_NLI_ALIASES",
        value_delimiter = ',',
        default_value = "encoder-nli-native"
    )]
    pub(crate) encoder_nli_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`, and
    /// `vocab.txt` required by encoder-nli aliases.
    #[arg(long, env = "OPENKIND_ENCODER_NLI_MODEL_ROOT")]
    pub(crate) encoder_nli_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned
    /// encoder-instruct-label engine (GLiClass label-marker readout).
    #[arg(
        long,
        env = "OPENKIND_ENCODER_INSTRUCT_LABEL_ALIASES",
        value_delimiter = ',',
        default_value = "encoder-instruct-label-native"
    )]
    pub(crate) encoder_instruct_label_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`, and
    /// `tokenizer.json` required by encoder-instruct-label aliases.
    #[arg(long, env = "OPENKIND_ENCODER_INSTRUCT_LABEL_MODEL_ROOT")]
    pub(crate) encoder_instruct_label_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned kev engine
    /// (Kev-0.6B pointer readout on Qwen3-0.6B-Base).
    #[arg(
        long,
        env = "OPENKIND_KEV_ALIASES",
        value_delimiter = ',',
        default_value = "kev-native"
    )]
    pub(crate) kev_aliases: Vec<String>,

    /// Model root with the pinned kev checkpoint artifacts
    /// (`adapter_model.safetensors`, `tokenizer.json`, `head.safetensors`).
    #[arg(long, env = "OPENKIND_KEV_MODEL_ROOT")]
    pub(crate) kev_model_root: Option<PathBuf>,

    /// Base-model root with the pinned `Qwen3-0.6B-Base`
    /// (`model.safetensors`, `config.json`).
    #[arg(long, env = "OPENKIND_KEV_BASE_ROOT")]
    pub(crate) kev_base_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned decoder-logit-llm
    /// engine (GGUF q8_0 letter readout).
    #[arg(
        long,
        env = "OPENKIND_DECODER_LLM_ALIASES",
        value_delimiter = ',',
        default_value = "decoder-llm-native"
    )]
    pub(crate) decoder_llm_aliases: Vec<String>,

    /// Model root with the pinned `qwen2.5-0.5b-instruct-q8_0.gguf` and
    /// `tokenizer.json` required by decoder-llm aliases.
    #[arg(long, env = "OPENKIND_DECODER_LLM_MODEL_ROOT")]
    pub(crate) decoder_llm_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned schema-scorer engine
    /// (MS MARCO cross-encoder scalar readout).
    #[arg(
        long,
        env = "OPENKIND_SCHEMA_SCORER_ALIASES",
        value_delimiter = ',',
        default_value = "schema-scorer-native"
    )]
    pub(crate) schema_scorer_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`, and
    /// `tokenizer.json` required by schema-scorer aliases.
    #[arg(long, env = "OPENKIND_SCHEMA_SCORER_MODEL_ROOT")]
    pub(crate) schema_scorer_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the router-script composite.
    #[arg(
        long,
        env = "OPENKIND_ROUTER_SCRIPT_ALIASES",
        value_delimiter = ',',
        default_value = "router-script"
    )]
    pub(crate) router_script_aliases: Vec<String>,

    /// Rule table for router-script aliases: comma-separated
    /// `script=alias` pairs plus a `default=alias` route. Sibling aliases
    /// must also be served through `--models`.
    #[arg(
        long,
        env = "OPENKIND_ROUTER_SCRIPT_RULES",
        value_delimiter = ';',
        default_value = "latin=decoder-letter-native,cyrillic=encoder-nli-native,default=decoder-letter-native"
    )]
    pub(crate) router_script_rules: Vec<String>,

    /// Aliases in `--models` that should use the pinned qwen3guard engine
    /// (Qwen3Guard-Stream risk-level readout).
    #[arg(
        long,
        env = "OPENKIND_QWEN3GUARD_ALIASES",
        value_delimiter = ',',
        default_value = "qwen3guard-native"
    )]
    pub(crate) qwen3guard_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`, and
    /// `tokenizer.json` required by qwen3guard aliases.
    #[arg(long, env = "OPENKIND_QWEN3GUARD_MODEL_ROOT")]
    pub(crate) qwen3guard_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned von engine
    /// (von-1.1 option-marker readout).
    #[arg(
        long,
        env = "OPENKIND_VON_ALIASES",
        value_delimiter = ',',
        default_value = "von-native"
    )]
    pub(crate) von_aliases: Vec<String>,

    /// Model root with the pinned `checkpoint/` directory required by von
    /// aliases.
    #[arg(long, env = "OPENKIND_VON_MODEL_ROOT")]
    pub(crate) von_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the winnow learned router.
    #[arg(
        long,
        env = "OPENKIND_WINNOW_ALIASES",
        value_delimiter = ',',
        default_value = "winnow-router"
    )]
    pub(crate) winnow_aliases: Vec<String>,

    /// Model root with the pinned base checkpoint and tokenizer required by
    /// winnow aliases.
    #[arg(long, env = "OPENKIND_WINNOW_MODEL_ROOT")]
    pub(crate) winnow_model_root: Option<PathBuf>,

    /// Path to the pinned winnow LoRA adapter.
    #[arg(long, env = "OPENKIND_WINNOW_ADAPTER")]
    pub(crate) winnow_adapter: Option<PathBuf>,

    /// Routing labels for winnow aliases: `A=<sibling alias>,B=<sibling
    /// alias>`. Sibling aliases must also be served through `--models`.
    #[arg(
        long,
        env = "OPENKIND_WINNOW_SIBLINGS",
        value_delimiter = ';',
        default_value = "A=decoder-letter-native,B=encoder-nli-native"
    )]
    pub(crate) winnow_siblings: Vec<String>,

    /// Aliases in `--models` that should use the pinned
    /// decoder-logit-qwen35 engine (JevK5 letter-logit readout on the
    /// Qwen3.5 hybrid backbone).
    #[arg(
        long,
        env = "OPENKIND_DECODER_LOGIT_QWEN35_ALIASES",
        value_delimiter = ',',
        default_value = "jevk5-native"
    )]
    pub(crate) decoder_logit_qwen35_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`,
    /// `jevk5_config.json`, and `tokenizer.json` required by
    /// decoder-logit-qwen35 aliases.
    #[arg(long, env = "OPENKIND_DECODER_LOGIT_QWEN35_MODEL_ROOT")]
    pub(crate) decoder_logit_qwen35_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned plumb-4b engine
    /// (Plumb-4B letter-logit readout on the Qwen3.5 hybrid backbone).
    #[arg(
        long,
        env = "OPENKIND_PLUMB_4B_ALIASES",
        value_delimiter = ',',
        default_value = "plumb-4b-native"
    )]
    pub(crate) plumb_4b_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`,
    /// `jevk5_config.json`, and `tokenizer.json` required by plumb-4b
    /// aliases.
    #[arg(long, env = "OPENKIND_PLUMB_4B_MODEL_ROOT")]
    pub(crate) plumb_4b_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned decider-4b engine
    /// (slot-logit readout on the Qwen3.5 hybrid backbone).
    #[arg(
        long,
        env = "OPENKIND_DECIDER_4B_ALIASES",
        value_delimiter = ',',
        default_value = "decider-4b-native"
    )]
    pub(crate) decider_4b_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `config.json`,
    /// `decider_config.json`, and `tokenizer.json` required by decider-4b
    /// aliases.
    #[arg(long, env = "OPENKIND_DECIDER_4B_MODEL_ROOT")]
    pub(crate) decider_4b_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned raw decoder-logit-
    /// qwen3 controls (letter-logit readout on the dense Qwen3 checkpoints).
    #[arg(
        long,
        env = "OPENKIND_DECODER_LOGIT_QWEN3_ALIASES",
        value_delimiter = ',',
        help = "comma-separated alias lists of the form                 06b=<alias>[,alias…];17b=<alias>[,alias…];4b=<alias>[,alias…]"
    )]
    pub(crate) decoder_logit_qwen3_aliases: Vec<String>,

    /// Model roots for the raw decoder-logit-qwen3 controls, of the form
    /// `06b=<path>;17b=<path>;4b=<path>`.
    #[arg(long, env = "OPENKIND_DECODER_LOGIT_QWEN3_MODEL_ROOTS")]
    pub(crate) decoder_logit_qwen3_model_roots: Vec<String>,

    /// Aliases in `--models` that should use the pinned laya-english engine
    /// (English ModernBERT-large decision encoder).
    #[arg(
        long,
        env = "OPENKIND_LAYA_ENGLISH_ALIASES",
        value_delimiter = ',',
        default_value = "laya-english-native"
    )]
    pub(crate) laya_english_aliases: Vec<String>,

    /// Model root with the pinned `model.safetensors`, `encoder/`,
    /// `tokenizer/`, and `rl_agent_config.json` required by laya-english
    /// aliases.
    #[arg(long, env = "OPENKIND_LAYA_ENGLISH_MODEL_ROOT")]
    pub(crate) laya_english_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned laya-multilingual
    /// engine (mmBERT-base decision encoder, 100+ languages).
    #[arg(
        long,
        env = "OPENKIND_LAYA_MULTILINGUAL_ALIASES",
        value_delimiter = ',',
        default_value = "laya-multilingual-native"
    )]
    pub(crate) laya_multilingual_aliases: Vec<String>,

    /// Model root with the pinned multilingual laya artifacts.
    #[arg(long, env = "OPENKIND_LAYA_MULTILINGUAL_MODEL_ROOT")]
    pub(crate) laya_multilingual_model_root: Option<PathBuf>,

    /// Aliases in `--models` that should use the pinned
    /// laya-typed-decisions engine (fine-tuned ModernBERT-large).
    #[arg(
        long,
        env = "OPENKIND_LAYA_TYPED_DECISIONS_ALIASES",
        value_delimiter = ',',
        default_value = "laya-typed-decisions-native"
    )]
    pub(crate) laya_typed_decisions_aliases: Vec<String>,

    /// Model root with the pinned typed-decisions laya artifacts.
    #[arg(long, env = "OPENKIND_LAYA_TYPED_DECISIONS_MODEL_ROOT")]
    pub(crate) laya_typed_decisions_model_root: Option<PathBuf>,

    /// Laya decision-encoder backend. `cuda` requires the daemon's `cuda`
    /// feature; `onnx`/`onnx-cuda` require the ONNX features and `model.onnx`
    /// in the model root; `mlx-fp32` is available on macOS arm64 with the
    /// daemon's `mlx` feature enabled.
    #[arg(
        long,
        env = "OPENKIND_LAYA_BACKEND",
        value_enum,
        default_value_t = LayaBackendArg::NativeCpu
    )]
    pub(crate) laya_backend: LayaBackendArg,

    /// Encoder-instruct-label backend. `cuda` requires the daemon's `cuda`
    /// feature; `onnx`/`onnx-cuda` require the ONNX features and `model.onnx`
    /// in the model root; `mlx-fp32` is available on macOS arm64 with the
    /// daemon's `mlx` feature enabled.
    #[arg(
        long,
        env = "OPENKIND_ENCODER_INSTRUCT_LABEL_BACKEND",
        value_enum,
        default_value_t = EncoderInstructLabelBackendArg::NativeCpu
    )]
    pub(crate) encoder_instruct_label_backend: EncoderInstructLabelBackendArg,

    /// decoder-logit-qwen35 backend. `cuda` requires the daemon's `cuda`
    /// feature; `mlx-fp32` is available on macOS arm64 with the daemon's
    /// `mlx` feature enabled.
    #[arg(
        long,
        env = "OPENKIND_DECODER_LOGIT_QWEN35_BACKEND",
        value_enum,
        default_value_t = DecoderLogitQwen35BackendArg::NativeCpu
    )]
    pub(crate) decoder_logit_qwen35_backend: DecoderLogitQwen35BackendArg,

    /// Encoder-NLI backend. `cuda` requires the daemon's `cuda` feature;
    /// `onnx`/`onnx-cuda` require the ONNX features and `model.onnx` in the
    /// model root.
    #[arg(
        long,
        env = "OPENKIND_ENCODER_NLI_BACKEND",
        value_enum,
        default_value_t = FamilyBackendArg::NativeCpu
    )]
    pub(crate) encoder_nli_backend: FamilyBackendArg,

    /// Decoder-logit-letter backend. `cuda` requires the daemon's `cuda`
    /// feature; `onnx`/`onnx-cuda` require the ONNX features and `model.onnx`
    /// in the model root.
    #[arg(
        long,
        env = "OPENKIND_DECODER_LETTER_BACKEND",
        value_enum,
        default_value_t = FamilyBackendArg::NativeCpu
    )]
    pub(crate) decoder_letter_backend: FamilyBackendArg,

    /// Kev backend. The pointer-head readout has no ONNX export; `cuda`
    /// requires the daemon's `cuda` feature.
    #[arg(
        long,
        env = "OPENKIND_KEV_BACKEND",
        value_enum,
        default_value_t = CudaOnlyBackendArg::NativeCpu
    )]
    pub(crate) kev_backend: CudaOnlyBackendArg,

    /// Decoder-logit-llm backend. The quantized GGUF readout has no ONNX
    /// export; `cuda` requires the daemon's `cuda` feature.
    #[arg(
        long,
        env = "OPENKIND_DECODER_LLM_BACKEND",
        value_enum,
        default_value_t = CudaOnlyBackendArg::NativeCpu
    )]
    pub(crate) decoder_llm_backend: CudaOnlyBackendArg,

    /// Schema-scorer backend. `cuda` requires the daemon's `cuda` feature;
    /// `onnx`/`onnx-cuda` require the ONNX features and `model.onnx` in the
    /// model root.
    #[arg(
        long,
        env = "OPENKIND_SCHEMA_SCORER_BACKEND",
        value_enum,
        default_value_t = FamilyBackendArg::NativeCpu
    )]
    pub(crate) schema_scorer_backend: FamilyBackendArg,

    /// Qwen3Guard backend. `cuda` requires the daemon's `cuda` feature;
    /// `onnx`/`onnx-cuda` require the ONNX features and `model.onnx` in the
    /// model root.
    #[arg(
        long,
        env = "OPENKIND_QWEN3GUARD_BACKEND",
        value_enum,
        default_value_t = FamilyBackendArg::NativeCpu
    )]
    pub(crate) qwen3guard_backend: FamilyBackendArg,

    /// Von backend. `cuda` requires the daemon's `cuda` feature;
    /// `onnx`/`onnx-cuda` require the ONNX features and `model.onnx` in the
    /// model root.
    #[arg(
        long,
        env = "OPENKIND_VON_BACKEND",
        value_enum,
        default_value_t = FamilyBackendArg::NativeCpu
    )]
    pub(crate) von_backend: FamilyBackendArg,

    /// Raw decoder-logit-qwen3 controls backend (applies to every size).
    /// `cuda` requires the daemon's `cuda` feature; `onnx`/`onnx-cuda`
    /// require the ONNX features and `model.onnx` in each model root.
    #[arg(
        long,
        env = "OPENKIND_DECODER_LOGIT_QWEN3_BACKEND",
        value_enum,
        default_value_t = FamilyBackendArg::NativeCpu
    )]
    pub(crate) decoder_logit_qwen3_backend: FamilyBackendArg,

    /// Decider-4b backend. The hybrid-backbone slot-logit readout has no
    /// ONNX export; `cuda` requires the daemon's `cuda` feature.
    #[arg(
        long,
        env = "OPENKIND_DECIDER_4B_BACKEND",
        value_enum,
        default_value_t = CudaOnlyBackendArg::NativeCpu
    )]
    pub(crate) decider_4b_backend: CudaOnlyBackendArg,

    /// Aliases in `--models` that should use the pinned winnow-e4b engine
    /// (Gemma 4 backbone letter readout on the Winnow-E4B Q8_0 GGUF).
    #[arg(
        long,
        env = "OPENKIND_WINNOW_E4B_ALIASES",
        value_delimiter = ',',
        default_value = "winnow-e4b-native"
    )]
    pub(crate) winnow_e4b_aliases: Vec<String>,

    /// Model root with the pinned `Winnow-E4B-Q8_0.gguf` and
    /// `tokenizer.json` required by winnow-e4b aliases.
    #[arg(long, env = "OPENKIND_WINNOW_E4B_MODEL_ROOT")]
    pub(crate) winnow_e4b_model_root: Option<PathBuf>,

    /// Winnow-e4b backend. The Gemma 4 backbone has no ONNX export; `cuda`
    /// requires the daemon's `cuda` feature.
    #[arg(
        long,
        env = "OPENKIND_WINNOW_E4B_BACKEND",
        value_enum,
        default_value_t = CudaOnlyBackendArg::NativeCpu
    )]
    pub(crate) winnow_e4b_backend: CudaOnlyBackendArg,

    /// Winnow router backend. The routing decoder has no ONNX export;
    /// `cuda` requires the daemon's `cuda` feature.
    #[arg(
        long,
        env = "OPENKIND_WINNOW_BACKEND",
        value_enum,
        default_value_t = CudaOnlyBackendArg::NativeCpu
    )]
    pub(crate) winnow_backend: CudaOnlyBackendArg,

    /// Maximum concurrent model evaluations per surveyed-family engine.
    #[arg(long, env = "OPENKIND_FAMILY_CONCURRENCY", default_value_t = 1)]
    pub(crate) family_concurrency: usize,

    /// Additional family requests allowed to wait for execution.
    #[arg(long, env = "OPENKIND_FAMILY_QUEUE", default_value_t = 2)]
    pub(crate) family_queue: usize,

    /// Queue-inclusive deadline for one family evaluation, in milliseconds.
    #[arg(long, env = "OPENKIND_FAMILY_TIMEOUT_MS", default_value_t = 600_000)]
    pub(crate) family_timeout_ms: u64,
}

impl FamilyArgs {
    fn admission(&self) -> FamilyAdmission {
        FamilyAdmission {
            concurrency: self.family_concurrency,
            queue: self.family_queue,
            timeout_ms: self.family_timeout_ms,
        }
    }

    /// Load every family engine whose aliases intersect `--models`.
    ///
    /// Returns `(alias, engine)` pairs ready for `EngineRegistry`
    /// registration. Missing artifact configuration fails closed.
    pub(crate) fn load_requested(
        &self,
        models: &[String],
        cuda_device: usize,
    ) -> Result<Vec<(String, Arc<dyn DecisionEngine>)>> {
        let admission = self.admission();
        let mut engines = Vec::new();

        let decoder_letter: Vec<_> = self
            .decoder_letter_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !decoder_letter.is_empty() {
            let model_root = self.decoder_letter_model_root.clone().context(
                "decoder-letter alias requested but --decoder-letter-model-root is missing",
            )?;
            let execution = self.decoder_letter_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                DecoderLetterEngine::load_with_execution(
                    DecoderLetterEngineConfig {
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load decoder-logit-letter engine: {error}"))?,
            );
            for alias in decoder_letter {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let encoder_nli: Vec<_> = self
            .encoder_nli_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !encoder_nli.is_empty() {
            let model_root = self
                .encoder_nli_model_root
                .clone()
                .context("encoder-nli alias requested but --encoder-nli-model-root is missing")?;
            let execution = self.encoder_nli_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                EncoderNliEngine::load_with_execution(
                    EncoderNliEngineConfig {
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load encoder-nli engine: {error}"))?,
            );
            for alias in encoder_nli {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let encoder_instruct_label: Vec<_> = self
            .encoder_instruct_label_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !encoder_instruct_label.is_empty() {
            let model_root = self
                .encoder_instruct_label_model_root
                .clone()
                .context(
                    "encoder-instruct-label alias requested but                      --encoder-instruct-label-model-root is missing",
                )?;
            let engine: Arc<dyn DecisionEngine> = match self.encoder_instruct_label_backend {
                EncoderInstructLabelBackendArg::NativeCpu => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::NativeCpu.to_execution(cuda_device)?,
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("load encoder-instruct-label engine: {error}")
                    })?,
                ),
                #[cfg(feature = "cuda")]
                EncoderInstructLabelBackendArg::Cuda => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::Cuda.to_execution(cuda_device)?,
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("load encoder-instruct-label engine: {error}")
                    })?,
                ),
                #[cfg(feature = "onnx")]
                EncoderInstructLabelBackendArg::Onnx => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::Onnx.to_execution(cuda_device)?,
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("load encoder-instruct-label engine: {error}")
                    })?,
                ),
                #[cfg(feature = "onnx")]
                EncoderInstructLabelBackendArg::OnnxCuda => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::OnnxCuda.to_execution(cuda_device)?,
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("load encoder-instruct-label engine: {error}")
                    })?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                EncoderInstructLabelBackendArg::MlxFp32 => Arc::new(
                    EncoderInstructLabelMlxEngine::load(EncoderInstructLabelMlxEngineConfig {
                        model_root,
                        limits: admission.limits(),
                    })
                    .map_err(|error| {
                        anyhow::anyhow!("load encoder-instruct-label mlx engine: {error}")
                    })?,
                ),
            };
            for alias in encoder_instruct_label {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let decoder_llm: Vec<_> = self
            .decoder_llm_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !decoder_llm.is_empty() {
            let model_root = self
                .decoder_llm_model_root
                .clone()
                .context("decoder-llm alias requested but --decoder-llm-model-root is missing")?;
            let execution = self.decoder_llm_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                DecoderLlmEngine::load_with_execution(
                    DecoderLlmEngineConfig {
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load decoder-logit-llm engine: {error}"))?,
            );
            for alias in decoder_llm {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let kev: Vec<_> = self
            .kev_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !kev.is_empty() {
            let model_root = self
                .kev_model_root
                .clone()
                .context("kev alias requested but --kev-model-root is missing")?;
            let base_root = self
                .kev_base_root
                .clone()
                .context("kev alias requested but --kev-base-root is missing")?;
            let execution = self.kev_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                KevEngine::load_with_execution(
                    KevEngineConfig {
                        model_root,
                        base_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load kev engine: {error}"))?,
            );
            for alias in kev {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let schema_scorer: Vec<_> = self
            .schema_scorer_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !schema_scorer.is_empty() {
            let model_root = self.schema_scorer_model_root.clone().context(
                "schema-scorer alias requested but --schema-scorer-model-root is missing",
            )?;
            let execution = self.schema_scorer_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                SchemaScorerEngine::load_with_execution(
                    SchemaScorerEngineConfig {
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load schema-scorer engine: {error}"))?,
            );
            for alias in schema_scorer {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let qwen3guard: Vec<_> = self
            .qwen3guard_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !qwen3guard.is_empty() {
            let model_root = self
                .qwen3guard_model_root
                .clone()
                .context("qwen3guard alias requested but --qwen3guard-model-root is missing")?;
            let execution = self.qwen3guard_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                Qwen3GuardEngine::load_with_execution(
                    Qwen3GuardEngineConfig {
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load qwen3guard engine: {error}"))?,
            );
            for alias in qwen3guard {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let von: Vec<_> = self
            .von_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !von.is_empty() {
            let model_root = self
                .von_model_root
                .clone()
                .context("von alias requested but --von-model-root is missing")?;
            let execution = self.von_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                VonEngine::load_with_execution(
                    VonEngineConfig {
                        profile: &VON,
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load von engine: {error}"))?,
            );
            for alias in von {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let decoder_logit_qwen35: Vec<_> = self
            .decoder_logit_qwen35_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !decoder_logit_qwen35.is_empty() {
            let model_root = self
                .decoder_logit_qwen35_model_root
                .clone()
                .context(
                    "decoder-logit-qwen35 alias requested but                      --decoder-logit-qwen35-model-root is missing",
                )?;
            let engine: Arc<dyn DecisionEngine> = match self.decoder_logit_qwen35_backend {
                DecoderLogitQwen35BackendArg::NativeCpu => Arc::new(
                    DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
                        profile: &openkind_backends::families::decoder_logit_qwen35::JEVK5,
                        model_root,
                        limits: admission.limits(),
                    })
                    .map_err(|error| {
                        anyhow::anyhow!("load decoder-logit-qwen35 engine: {error}")
                    })?,
                ),
                #[cfg(feature = "cuda")]
                DecoderLogitQwen35BackendArg::Cuda => Arc::new(
                    DecoderLogitQwen35Engine::load_with_execution(
                        DecoderLogitQwen35EngineConfig {
                            profile: &openkind_backends::families::decoder_logit_qwen35::JEVK5,
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::Cuda.to_execution(cuda_device)?,
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("load decoder-logit-qwen35 engine: {error}")
                    })?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                DecoderLogitQwen35BackendArg::MlxFp32 => Arc::new(
                    DecoderLogitQwen35MlxEngine::load(DecoderLogitQwen35MlxEngineConfig {
                        profile: &openkind_backends::families::decoder_logit_qwen35::JEVK5,
                        model_root,
                        limits: admission.limits(),
                    })
                    .map_err(|error| {
                        anyhow::anyhow!("load decoder-logit-qwen35 mlx engine: {error}")
                    })?,
                ),
            };
            for alias in decoder_logit_qwen35 {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let plumb_4b: Vec<_> = self
            .plumb_4b_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !plumb_4b.is_empty() {
            let model_root = self
                .plumb_4b_model_root
                .clone()
                .context("plumb-4b alias requested but --plumb-4b-model-root is missing")?;
            let engine: Arc<dyn DecisionEngine> = match self.decoder_logit_qwen35_backend {
                DecoderLogitQwen35BackendArg::NativeCpu => Arc::new(
                    DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
                        profile: &PLUMB_4B,
                        model_root,
                        limits: admission.limits(),
                    })
                    .map_err(|error| anyhow::anyhow!("load plumb-4b engine: {error}"))?,
                ),
                #[cfg(feature = "cuda")]
                DecoderLogitQwen35BackendArg::Cuda => Arc::new(
                    DecoderLogitQwen35Engine::load_with_execution(
                        DecoderLogitQwen35EngineConfig {
                            profile: &PLUMB_4B,
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::Cuda.to_execution(cuda_device)?,
                    )
                    .map_err(|error| anyhow::anyhow!("load plumb-4b engine: {error}"))?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                DecoderLogitQwen35BackendArg::MlxFp32 => Arc::new(
                    DecoderLogitQwen35MlxEngine::load(DecoderLogitQwen35MlxEngineConfig {
                        profile: &PLUMB_4B,
                        model_root,
                        limits: admission.limits(),
                    })
                    .map_err(|error| anyhow::anyhow!("load plumb-4b mlx engine: {error}"))?,
                ),
            };
            for alias in plumb_4b {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        // Raw decoder-logit-qwen3 controls: size-keyed alias lists and model
        // roots (`06b=`, `17b=`, `4b=` prefixes).
        let mut qwen3_controls: Vec<(
            &std::string::String,
            PathBuf,
            &'static openkind_backends::families::decoder_logit_qwen3::Qwen3LogitProfile,
            &str,
        )> = Vec::new();
        for entry in &self.decoder_logit_qwen3_model_roots {
            let Some((size, root)) = entry.split_once('=') else {
                bail!(
                    "--decoder-logit-qwen3-model-roots entries must look like 06b=<path>; got {entry}"
                );
            };
            let profile = match size {
                "06b" => &QWEN3_06B,
                "17b" => &QWEN3_17B,
                "4b" => &QWEN3_4B,
                other => {
                    bail!("unknown decoder-logit-qwen3 control size `{other}`; expected 06b, 17b, or 4b")
                }
            };
            for alias in &self.decoder_logit_qwen3_aliases {
                let Some((alias_size, alias_names)) = alias.split_once('=') else {
                    bail!(
                        "--decoder-logit-qwen3-aliases entries must look like 06b=<alias>[,alias…]; got {alias}"
                    );
                };
                if alias_size != size {
                    continue;
                }
                for alias_name in alias_names.split(',') {
                    if let Some(alias_string) = models.iter().find(|model| model == &alias_name) {
                        qwen3_controls.push((alias_string, PathBuf::from(root), profile, size));
                    }
                }
            }
        }
        if !qwen3_controls.is_empty() {
            let qwen3_execution = self.decoder_logit_qwen3_backend.to_execution(cuda_device)?;
            for (alias, model_root, profile, size) in &qwen3_controls {
                let engine: Arc<dyn DecisionEngine> = Arc::new(
                    DecoderLogitQwen3Engine::load_with_execution(
                        DecoderLogitQwen3EngineConfig {
                            profile,
                            model_root: model_root.clone(),
                            limits: admission.limits(),
                        },
                        qwen3_execution,
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("load decoder-logit-qwen3-{size} engine: {error}")
                    })?,
                );
                engines.push(((*alias).clone(), Arc::clone(&engine)));
            }
        }

        let decider_4b: Vec<_> = self
            .decider_4b_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !decider_4b.is_empty() {
            let model_root = self
                .decider_4b_model_root
                .clone()
                .context("decider-4b alias requested but --decider-4b-model-root is missing")?;
            let execution = self.decider_4b_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                DeciderEngine::load_with_execution(
                    DeciderEngineConfig {
                        profile: &DECIDER_4B,
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load decider-4b engine: {error}"))?,
            );
            for alias in decider_4b {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let winnow_e4b: Vec<_> = self
            .winnow_e4b_aliases
            .iter()
            .filter(|alias| models.contains(alias))
            .collect();
        if !winnow_e4b.is_empty() {
            let model_root = self
                .winnow_e4b_model_root
                .clone()
                .context("winnow-e4b alias requested but --winnow-e4b-model-root is missing")?;
            let execution = self.winnow_e4b_backend.to_execution(cuda_device)?;
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                Gemma4DecisionEngine::load_with_execution(
                    Gemma4EngineConfig {
                        model_root,
                        limits: admission.limits(),
                    },
                    execution,
                )
                .map_err(|error| anyhow::anyhow!("load winnow-e4b engine: {error}"))?,
            );
            for alias in winnow_e4b {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        let mut laya_requested: Vec<(
            Vec<&String>,
            PathBuf,
            &openkind_backends::families::laya::LayaProfile,
            &str,
        )> = Vec::new();
        for (aliases, model_root, profile, name) in [
            (
                &self.laya_english_aliases,
                self.laya_english_model_root.clone(),
                &LAYA_ENGLISH,
                "laya-english",
            ),
            (
                &self.laya_multilingual_aliases,
                self.laya_multilingual_model_root.clone(),
                &LAYA_MULTILINGUAL,
                "laya-multilingual",
            ),
            (
                &self.laya_typed_decisions_aliases,
                self.laya_typed_decisions_model_root.clone(),
                &LAYA_TYPED_DECISIONS,
                "laya-typed-decisions",
            ),
        ] {
            let requested: Vec<_> = aliases
                .iter()
                .filter(|alias| models.contains(alias))
                .collect();
            if requested.is_empty() {
                continue;
            }
            let model_root = model_root.context(format!(
                "{name} alias requested but --{name}-model-root is missing"
            ))?;
            laya_requested.push((requested, model_root, profile, name));
        }
        for (requested, model_root, profile, name) in laya_requested {
            let engine: Arc<dyn DecisionEngine> = match self.laya_backend {
                LayaBackendArg::NativeCpu => Arc::new(
                    LayaEngine::load_with_execution(
                        LayaEngineConfig {
                            profile,
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::NativeCpu.to_execution(cuda_device)?,
                    )
                    .map_err(|error| anyhow::anyhow!("load {name} engine: {error}"))?,
                ),
                #[cfg(feature = "cuda")]
                LayaBackendArg::Cuda => Arc::new(
                    LayaEngine::load_with_execution(
                        LayaEngineConfig {
                            profile,
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::Cuda.to_execution(cuda_device)?,
                    )
                    .map_err(|error| anyhow::anyhow!("load {name} engine: {error}"))?,
                ),
                #[cfg(feature = "onnx")]
                LayaBackendArg::Onnx => Arc::new(
                    LayaEngine::load_with_execution(
                        LayaEngineConfig {
                            profile,
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::Onnx.to_execution(cuda_device)?,
                    )
                    .map_err(|error| anyhow::anyhow!("load {name} engine: {error}"))?,
                ),
                #[cfg(feature = "onnx")]
                LayaBackendArg::OnnxCuda => Arc::new(
                    LayaEngine::load_with_execution(
                        LayaEngineConfig {
                            profile,
                            model_root,
                            limits: admission.limits(),
                        },
                        FamilyBackendArg::OnnxCuda.to_execution(cuda_device)?,
                    )
                    .map_err(|error| anyhow::anyhow!("load {name} engine: {error}"))?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                LayaBackendArg::MlxFp32 => Arc::new(
                    LayaMlxEngine::load(LayaMlxEngineConfig {
                        profile,
                        model_root,
                        limits: admission.limits(),
                    })
                    .map_err(|error| anyhow::anyhow!("load {name} mlx engine: {error}"))?,
                ),
            };
            for alias in requested {
                engines.push((alias.to_string(), Arc::clone(&engine)));
            }
        }

        Ok(engines)
    }

    /// Winnow aliases requested through `--models`, with their sibling
    /// label bindings. Routing composition happens after the sibling
    /// engines are registered, so this returns configuration only.
    pub(crate) fn winnow_requested(&self, models: &[String]) -> Result<Vec<(String, Vec<String>)>> {
        let mut requested = Vec::new();
        for (alias, siblings) in self.winnow_aliases.iter().zip(
            self.winnow_siblings
                .iter()
                .chain(std::iter::repeat(&self.winnow_siblings[0])),
        ) {
            if !models.contains(alias) {
                continue;
            }
            let mut labels: Vec<String> = Vec::new();
            let mut bindings = std::collections::BTreeMap::new();
            for token in siblings.split(',') {
                let (label, sibling) = token.split_once('=').ok_or_else(|| {
                    anyhow::anyhow!("expected `A=alias` in --winnow-siblings, found `{token}`")
                })?;
                let label = label.trim();
                match label {
                    "A" | "B" => {}
                    other => anyhow::bail!("unknown winnow label `{other}`"),
                }
                let sibling = sibling.trim();
                if sibling.is_empty() {
                    anyhow::bail!("winnow label `{label}` requires a non-empty sibling alias");
                }
                if bindings.insert(label, sibling).is_some() {
                    anyhow::bail!("duplicate winnow label `{label}`");
                }
            }
            if bindings.len() != 2 {
                anyhow::bail!("--winnow-siblings needs exactly two labels A and B");
            }
            // Learned logits are ordered A, B regardless of flag token order.
            for label in ["A", "B"] {
                labels.push(bindings[label].to_owned());
            }
            requested.push((alias.clone(), labels));
        }
        Ok(requested)
    }

    /// Artifact configuration for winnow aliases.
    pub(crate) fn winnow_artifacts(&self, alias: &str) -> Result<(PathBuf, PathBuf)> {
        let model_root = self
            .winnow_model_root
            .clone()
            .context("winnow alias requested but --winnow-model-root is missing")?;
        let adapter = self
            .winnow_adapter
            .clone()
            .context("winnow alias requested but --winnow-adapter is missing")?;
        let _ = alias;
        Ok((model_root, adapter))
    }

    /// Router-script aliases requested through `--models`, with their rule
    /// tables. Routing composition happens after the sibling engines are
    /// registered, so this returns configuration only.
    pub(crate) fn router_script_requested(
        &self,
        models: &[String],
    ) -> Result<Vec<(String, ScriptRuleTable)>> {
        let mut requested = Vec::new();
        for (alias, rules) in self.router_script_aliases.iter().zip(
            self.router_script_rules
                .iter()
                .chain(std::iter::repeat(&self.router_script_rules[0])),
        ) {
            if models.contains(alias) {
                let table = ScriptRuleTable::parse(rules).map_err(|error| {
                    anyhow::anyhow!("parse --router-script-rules for `{alias}`: {error}")
                })?;
                requested.push((alias.clone(), table));
            }
        }
        Ok(requested)
    }

    /// Reject duplicate native or family aliases that would shadow each other.
    pub(crate) fn validate(&self, native_aliases: &[String]) -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for alias in self
            .decoder_letter_aliases
            .iter()
            .chain(&self.encoder_nli_aliases)
            .chain(&self.encoder_instruct_label_aliases)
            .chain(&self.decoder_llm_aliases)
            .chain(&self.schema_scorer_aliases)
            .chain(&self.router_script_aliases)
            .chain(&self.qwen3guard_aliases)
            .chain(&self.von_aliases)
            .chain(&self.winnow_aliases)
            .chain(&self.kev_aliases)
            .chain(&self.decoder_logit_qwen35_aliases)
            .chain(&self.plumb_4b_aliases)
            .chain(&self.decider_4b_aliases)
            .chain(self.decoder_logit_qwen3_aliases.iter())
            .chain(&self.laya_english_aliases)
            .chain(&self.laya_multilingual_aliases)
            .chain(&self.laya_typed_decisions_aliases)
            .chain(native_aliases)
        {
            if !seen.insert(alias.as_str()) {
                bail!("alias `{alias}` is assigned to more than one family engine");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winnow_label_bindings_follow_labels_instead_of_input_order() {
        let args = FamilyArgs::parse_from(["openkindd", "--winnow-siblings", "B=second,A=first"]);
        assert_eq!(
            args.winnow_requested(&["winnow-router".into()]).unwrap(),
            vec![(
                "winnow-router".into(),
                vec!["first".into(), "second".into()]
            )]
        );
    }

    #[test]
    fn winnow_rejects_duplicate_missing_and_empty_label_bindings() {
        for bindings in [
            "A=first,A=second",
            "B=first,B=second",
            "A=first",
            "A=,B=second",
        ] {
            let args = FamilyArgs::parse_from(["openkindd", "--winnow-siblings", bindings]);
            assert!(
                args.winnow_requested(&["winnow-router".into()]).is_err(),
                "{bindings}"
            );
        }
    }

    #[test]
    fn native_aliases_cannot_shadow_family_aliases() {
        let args = FamilyArgs::parse_from(["openkindd"]);
        assert!(args.validate(&["qwen35-native".into()]).is_ok());
        assert!(args.validate(&["decoder-letter-native".into()]).is_err());
        assert!(args.validate(&["router-script".into()]).is_err());
    }
}
