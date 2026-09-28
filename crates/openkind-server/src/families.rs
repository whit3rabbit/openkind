//! Daemon configuration and registration for surveyed-family engines.
//!
//! Each family exposes `--<family>-aliases` plus the artifact paths its
//! loader requires. An alias is only served when it appears in `--models`;
//! requesting a family alias without its artifact configuration fails the
//! daemon startup instead of falling back to the mock engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use openkind_backends::families::decoder_logit_letter::{
    DecoderLetterEngine, DecoderLetterEngineConfig,
};
use openkind_backends::families::decoder_logit_llm::{DecoderLlmEngine, DecoderLlmEngineConfig};
use openkind_backends::families::decoder_logit_qwen35::{
    DecoderLogitQwen35Engine, DecoderLogitQwen35EngineConfig,
};
use openkind_backends::families::encoder_instruct_label::{
    EncoderInstructLabelEngine, EncoderInstructLabelEngineConfig,
};
use openkind_backends::families::encoder_nli::{EncoderNliEngine, EncoderNliEngineConfig};
use openkind_backends::families::kev::{KevEngine, KevEngineConfig};
use openkind_backends::families::laya::{
    LayaEngine, LayaEngineConfig, LAYA_ENGLISH, LAYA_MULTILINGUAL, LAYA_TYPED_DECISIONS,
};
use openkind_backends::families::qwen3guard::{Qwen3GuardEngine, Qwen3GuardEngineConfig};
use openkind_backends::families::router_script::ScriptRuleTable;
use openkind_backends::families::schema_scorer::{SchemaScorerEngine, SchemaScorerEngineConfig};
use openkind_backends::families::support::FamilyLimits;
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                DecoderLetterEngine::load(DecoderLetterEngineConfig {
                    model_root,
                    limits: admission.limits(),
                })
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                EncoderNliEngine::load(EncoderNliEngineConfig {
                    model_root,
                    limits: admission.limits(),
                })
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                EncoderInstructLabelEngine::load(EncoderInstructLabelEngineConfig {
                    model_root,
                    limits: admission.limits(),
                })
                .map_err(|error| anyhow::anyhow!("load encoder-instruct-label engine: {error}"))?,
            );
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                DecoderLlmEngine::load(DecoderLlmEngineConfig {
                    model_root,
                    limits: admission.limits(),
                })
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                KevEngine::load(KevEngineConfig {
                    model_root,
                    base_root,
                    limits: admission.limits(),
                })
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                SchemaScorerEngine::load(SchemaScorerEngineConfig {
                    model_root,
                    limits: admission.limits(),
                })
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                Qwen3GuardEngine::load(Qwen3GuardEngineConfig {
                    model_root,
                    limits: admission.limits(),
                })
                .map_err(|error| anyhow::anyhow!("load qwen3guard engine: {error}"))?,
            );
            for alias in qwen3guard {
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
                    model_root,
                    limits: admission.limits(),
                })
                .map_err(|error| anyhow::anyhow!("load decoder-logit-qwen35 engine: {error}"))?,
            );
            for alias in decoder_logit_qwen35 {
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
            let engine: Arc<dyn DecisionEngine> = Arc::new(
                LayaEngine::load(LayaEngineConfig {
                    profile,
                    model_root,
                    limits: admission.limits(),
                })
                .map_err(|error| anyhow::anyhow!("load {name} engine: {error}"))?,
            );
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
            for token in siblings.split(',') {
                let (label, sibling) = token.split_once('=').ok_or_else(|| {
                    anyhow::anyhow!("expected `A=alias` in --winnow-siblings, found `{token}`")
                })?;
                match label.trim() {
                    "A" | "B" => {}
                    other => anyhow::bail!("unknown winnow label `{other}`"),
                }
                labels.push(sibling.trim().to_owned());
            }
            if labels.len() != 2 {
                anyhow::bail!("--winnow-siblings needs exactly two labels A and B");
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

    /// Reject duplicate family aliases that would shadow each other.
    pub(crate) fn validate(&self) -> Result<()> {
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
            .chain(&self.winnow_aliases)
            .chain(&self.kev_aliases)
            .chain(&self.decoder_logit_qwen35_aliases)
            .chain(&self.laya_english_aliases)
            .chain(&self.laya_multilingual_aliases)
            .chain(&self.laya_typed_decisions_aliases)
        {
            if !seen.insert(alias.as_str()) {
                bail!("alias `{alias}` is assigned to more than one family engine");
            }
        }
        Ok(())
    }
}
