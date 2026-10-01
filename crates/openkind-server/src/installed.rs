//! Shared loading of installed catalog models for daemon startup and the
//! playground. A manifest is only loadable when its `(name, loader,
//! profile)` identity matches a compiled-in loader; artifact bytes are
//! digest-verified by `ModelStore::acquire_serving` before this module runs,
//! and each family loader re-verifies its pinned digests at load.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use openkind_backends::families::decider::{DeciderEngine, DeciderEngineConfig, DECIDER_4B};
use openkind_backends::families::decoder_logit_letter::{
    DecoderLetterEngine, DecoderLetterEngineConfig, PROFILE_ID as DECODER_LETTER_PROFILE,
};
use openkind_backends::families::decoder_logit_llm::{
    DecoderLlmEngine, DecoderLlmEngineConfig, PROFILE_ID as DECODER_LLM_PROFILE,
};
use openkind_backends::families::decoder_logit_qwen35::{
    DecoderLogitQwen35Engine, DecoderLogitQwen35EngineConfig, Qwen35LogitProfile, JEVK5, PLUMB_4B,
    PROFILE_ID as DECODER_LOGIT_QWEN35_PROFILE,
};
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::families::decoder_logit_qwen35::{
    DecoderLogitQwen35MlxEngine, DecoderLogitQwen35MlxEngineConfig,
};
use openkind_backends::families::encoder_instruct_label::{
    EncoderInstructLabelEngine, EncoderInstructLabelEngineConfig,
    PROFILE_ID as ENCODER_INSTRUCT_LABEL_PROFILE,
};
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::families::encoder_instruct_label::{
    EncoderInstructLabelMlxEngine, EncoderInstructLabelMlxEngineConfig,
};
use openkind_backends::families::encoder_nli::{
    EncoderNliEngine, EncoderNliEngineConfig, PROFILE_ID as ENCODER_NLI_PROFILE,
};
use openkind_backends::families::kev::{KevEngine, KevEngineConfig, PROFILE_ID as KEV_PROFILE};
use openkind_backends::families::laya::{
    LayaEngine, LayaEngineConfig, LayaProfile, LAYA_ENGLISH, LAYA_MULTILINGUAL,
    LAYA_TYPED_DECISIONS,
};
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::families::laya::{LayaMlxEngine, LayaMlxEngineConfig};
use openkind_backends::families::qwen3guard::{
    Qwen3GuardEngine, Qwen3GuardEngineConfig, PROFILE_ID as QWEN3GUARD_PROFILE,
};
use openkind_backends::families::schema_scorer::{
    SchemaScorerEngine, SchemaScorerEngineConfig, PROFILE_ID as SCHEMA_SCORER_PROFILE,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_backends::families::von::{
    VonEngine, VonEngineConfig, PROFILE_ID as VON_PROFILE, VON,
};
use openkind_backends::families::winnow::{
    WinnowEngine, WinnowEngineConfig, PROFILE_ID as WINNOW_PROFILE,
};
use openkind_engine::{DecisionEngine, EngineRegistry};
use openkind_model_store::{
    Manifest, DECIDER_4B_MODEL_NAME, DECODER_LOGIT_LETTER_MODEL_NAME, DECODER_LOGIT_LLM_MODEL_NAME,
    DECODER_LOGIT_QWEN35_MODEL_NAME, ENCODER_INSTRUCT_LABEL_MODEL_NAME, ENCODER_NLI_MODEL_NAME,
    KEV_MODEL_NAME, LAYA_ENGLISH_MODEL_NAME, LAYA_MULTILINGUAL_MODEL_NAME,
    LAYA_TYPED_DECISIONS_MODEL_NAME, PLUMB_4B_MODEL_NAME, QWEN35_STATE_FIRST_MODEL_NAME,
    QWEN3GUARD_MODEL_NAME, SCHEMA_SCORER_MODEL_NAME, VON_MODEL_NAME, WINNOW_MODEL_NAME,
};

use crate::args::{
    Args, DecoderLogitQwen35BackendArg, EncoderInstructLabelBackendArg, LayaBackendArg,
};

/// The installed model a manifest describes, matched against the pinned
/// identities this build can load.
#[derive(Debug, Clone, Copy)]
pub(crate) enum InstalledKind {
    Qwen35StateFirst,
    Laya(&'static LayaProfile),
    DecoderLetter,
    EncoderNli,
    EncoderInstructLabel,
    DecoderLlm,
    SchemaScorer,
    Qwen3Guard,
    Kev,
    DecoderLogitQwen35(&'static Qwen35LogitProfile),
    Decider4b,
    Von,
    Winnow,
}

/// Classify an installed-model manifest. `None` means this build has no
/// loader for the manifest identity.
pub(crate) fn installed_kind(manifest: &Manifest) -> Option<InstalledKind> {
    let profile = manifest.profile_id.as_str();
    match (manifest.name.as_str(), manifest.loader_id.as_str()) {
        (QWEN35_STATE_FIRST_MODEL_NAME, "qwen35-state-first")
            if profile == openkind_backends::qwen35::PROFILE_ID =>
        {
            Some(InstalledKind::Qwen35StateFirst)
        }
        (LAYA_ENGLISH_MODEL_NAME, "laya-english") if profile == LAYA_ENGLISH.profile_id => {
            Some(InstalledKind::Laya(&LAYA_ENGLISH))
        }
        (LAYA_MULTILINGUAL_MODEL_NAME, "laya-multilingual")
            if profile == LAYA_MULTILINGUAL.profile_id =>
        {
            Some(InstalledKind::Laya(&LAYA_MULTILINGUAL))
        }
        (LAYA_TYPED_DECISIONS_MODEL_NAME, "laya-typed-decisions")
            if profile == LAYA_TYPED_DECISIONS.profile_id =>
        {
            Some(InstalledKind::Laya(&LAYA_TYPED_DECISIONS))
        }
        (DECODER_LOGIT_LETTER_MODEL_NAME, "decoder-logit-letter")
            if profile == DECODER_LETTER_PROFILE =>
        {
            Some(InstalledKind::DecoderLetter)
        }
        (ENCODER_NLI_MODEL_NAME, "encoder-nli") if profile == ENCODER_NLI_PROFILE => {
            Some(InstalledKind::EncoderNli)
        }
        (ENCODER_INSTRUCT_LABEL_MODEL_NAME, "encoder-instruct-label")
            if profile == ENCODER_INSTRUCT_LABEL_PROFILE =>
        {
            Some(InstalledKind::EncoderInstructLabel)
        }
        (DECODER_LOGIT_LLM_MODEL_NAME, "decoder-logit-llm") if profile == DECODER_LLM_PROFILE => {
            Some(InstalledKind::DecoderLlm)
        }
        (SCHEMA_SCORER_MODEL_NAME, "schema-scorer") if profile == SCHEMA_SCORER_PROFILE => {
            Some(InstalledKind::SchemaScorer)
        }
        (QWEN3GUARD_MODEL_NAME, "qwen3guard") if profile == QWEN3GUARD_PROFILE => {
            Some(InstalledKind::Qwen3Guard)
        }
        (KEV_MODEL_NAME, "kev") if profile == KEV_PROFILE => Some(InstalledKind::Kev),
        (DECODER_LOGIT_QWEN35_MODEL_NAME, "decoder-logit-qwen35")
            if profile == DECODER_LOGIT_QWEN35_PROFILE =>
        {
            Some(InstalledKind::DecoderLogitQwen35(&JEVK5))
        }
        (PLUMB_4B_MODEL_NAME, "plumb-4b") if profile == PLUMB_4B.profile_id => {
            Some(InstalledKind::DecoderLogitQwen35(&PLUMB_4B))
        }
        (DECIDER_4B_MODEL_NAME, "decider-4b") if profile == DECIDER_4B.profile_id => {
            Some(InstalledKind::Decider4b)
        }
        (VON_MODEL_NAME, "von") if profile == VON_PROFILE => Some(InstalledKind::Von),
        (WINNOW_MODEL_NAME, "winnow") if profile == WINNOW_PROFILE => Some(InstalledKind::Winnow),
        _ => None,
    }
}

/// Winnow's routing labels bound to sibling engines, in label order. For
/// each label the loader prefers the sibling's installed catalog name and
/// falls back to the artifact-path family alias used by `--models`.
pub(crate) const WINNOW_SIBLING_DEFAULTS: &[(&str, &str, &str)] = &[
    (
        "A",
        DECODER_LOGIT_LETTER_MODEL_NAME,
        "decoder-letter-native",
    ),
    ("B", ENCODER_NLI_MODEL_NAME, "encoder-nli-native"),
];

/// Build the engine for one classified installation. The registry is only
/// consulted for the winnow router, whose sibling engines must already be
/// registered (through `--models` or earlier `--installed-models` entries).
pub(crate) fn load_installed_engine(
    args: &Args,
    kind: InstalledKind,
    root: &Path,
    registry: &EngineRegistry,
) -> Result<Arc<dyn DecisionEngine>> {
    let limits = FamilyLimits {
        max_concurrent_requests: args.family_args.family_concurrency,
        max_queued_requests: args.family_args.family_queue,
        retry_after_ms: 1_000,
        evaluation_timeout: Some(std::time::Duration::from_millis(
            args.family_args.family_timeout_ms,
        )),
    };
    let engine: Arc<dyn DecisionEngine> = match kind {
        InstalledKind::Qwen35StateFirst => crate::load_qwen(
            args,
            root.join("bundle"),
            root.join("checkpoint"),
            root.join("checkpoint/tokenizer.json"),
        )?,
        InstalledKind::Laya(profile) => match args.family_args.laya_backend {
            LayaBackendArg::NativeCpu => Arc::new(
                LayaEngine::load(LayaEngineConfig {
                    profile,
                    model_root: root.to_path_buf(),
                    limits,
                })
                .map_err(|error| anyhow!("load installed laya model: {error}"))?,
            ),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            LayaBackendArg::MlxFp32 => Arc::new(
                LayaMlxEngine::load(LayaMlxEngineConfig {
                    profile,
                    model_root: root.to_path_buf(),
                    limits,
                })
                .map_err(|error| anyhow!("load installed laya mlx model: {error}"))?,
            ),
        },
        InstalledKind::DecoderLetter => Arc::new(
            DecoderLetterEngine::load(DecoderLetterEngineConfig {
                model_root: root.join("checkpoint"),
                limits,
            })
            .map_err(|error| anyhow!("load decoder-logit-letter engine: {error}"))?,
        ),
        InstalledKind::EncoderNli => Arc::new(
            EncoderNliEngine::load(EncoderNliEngineConfig {
                model_root: root.join("checkpoint"),
                limits,
            })
            .map_err(|error| anyhow!("load encoder-nli engine: {error}"))?,
        ),
        InstalledKind::EncoderInstructLabel => {
            match args.family_args.encoder_instruct_label_backend {
                EncoderInstructLabelBackendArg::NativeCpu => Arc::new(
                    EncoderInstructLabelEngine::load(EncoderInstructLabelEngineConfig {
                        model_root: root.join("checkpoint"),
                        limits,
                    })
                    .map_err(|error| anyhow!("load encoder-instruct-label engine: {error}"))?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                EncoderInstructLabelBackendArg::MlxFp32 => Arc::new(
                    EncoderInstructLabelMlxEngine::load(EncoderInstructLabelMlxEngineConfig {
                        model_root: root.join("checkpoint"),
                        limits,
                    })
                    .map_err(|error| anyhow!("load encoder-instruct-label mlx engine: {error}"))?,
                ),
            }
        }
        InstalledKind::DecoderLlm => Arc::new(
            DecoderLlmEngine::load(DecoderLlmEngineConfig {
                model_root: root.join("checkpoint"),
                limits,
            })
            .map_err(|error| anyhow!("load decoder-logit-llm engine: {error}"))?,
        ),
        InstalledKind::SchemaScorer => Arc::new(
            SchemaScorerEngine::load(SchemaScorerEngineConfig {
                model_root: root.join("checkpoint"),
                limits,
            })
            .map_err(|error| anyhow!("load schema-scorer engine: {error}"))?,
        ),
        InstalledKind::Qwen3Guard => Arc::new(
            Qwen3GuardEngine::load(Qwen3GuardEngineConfig {
                model_root: root.join("checkpoint"),
                limits,
            })
            .map_err(|error| anyhow!("load qwen3guard engine: {error}"))?,
        ),
        InstalledKind::Kev => Arc::new(
            KevEngine::load(KevEngineConfig {
                model_root: root.join("adapter"),
                base_root: root.join("base"),
                limits,
            })
            .map_err(|error| anyhow!("load kev engine: {error}"))?,
        ),
        InstalledKind::DecoderLogitQwen35(profile) => {
            match args.family_args.decoder_logit_qwen35_backend {
                DecoderLogitQwen35BackendArg::NativeCpu => Arc::new(
                    DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
                        profile,
                        model_root: root.join("checkpoint"),
                        limits,
                    })
                    .map_err(|error| anyhow!("load decoder-logit-qwen35 engine: {error}"))?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                DecoderLogitQwen35BackendArg::MlxFp32 => Arc::new(
                    DecoderLogitQwen35MlxEngine::load(DecoderLogitQwen35MlxEngineConfig {
                        profile,
                        model_root: root.join("checkpoint"),
                        limits,
                    })
                    .map_err(|error| anyhow!("load decoder-logit-qwen35 mlx engine: {error}"))?,
                ),
            }
        }
        InstalledKind::Von => Arc::new(
            VonEngine::load(VonEngineConfig {
                profile: &VON,
                model_root: root.to_path_buf(),
                limits,
            })
            .map_err(|error| anyhow!("load von engine: {error}"))?,
        ),
        InstalledKind::Decider4b => Arc::new(
            DeciderEngine::load(DeciderEngineConfig {
                profile: &DECIDER_4B,
                model_root: root.join("checkpoint"),
                limits,
            })
            .map_err(|error| anyhow!("load decider-4b engine: {error}"))?,
        ),
        InstalledKind::Winnow => {
            let mut siblings: Vec<(String, Arc<dyn DecisionEngine>)> = Vec::new();
            for &(label, catalog_sibling, family_alias) in WINNOW_SIBLING_DEFAULTS {
                let bound = registry
                    .get(catalog_sibling)
                    .map(|engine| (catalog_sibling, engine))
                    .or_else(|| {
                        registry
                            .get(family_alias)
                            .map(|engine| (family_alias, engine))
                    });
                let (sibling_alias, engine) = bound.ok_or_else(|| {
                    anyhow!(
                        "winnow router label {label} references unregistered sibling \
                         `{catalog_sibling}` (or `{family_alias}`); serve it through \
                         --models or --installed-models"
                    )
                })?;
                siblings.push((sibling_alias.to_owned(), engine));
            }
            Arc::new(
                WinnowEngine::load(
                    WinnowEngineConfig {
                        model_root: root.join("checkpoint"),
                        adapter_path: root.join("adapter.safetensors"),
                        limits,
                    },
                    siblings,
                )
                .map_err(|error| anyhow!("load winnow engine: {error}"))?,
            )
        }
    };
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn empty_model_error(args: &Args, kind: InstalledKind) -> String {
        let root = tempfile::tempdir().unwrap();
        match load_installed_engine(args, kind, root.path(), &EngineRegistry::new()) {
            Ok(_) => panic!("empty model directory must fail loading"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn installed_accelerated_families_default_to_cpu() {
        let args = Args::parse_from(["openkindd"]);
        for (kind, context) in [
            (
                InstalledKind::Laya(&LAYA_ENGLISH),
                "load installed laya model:",
            ),
            (
                InstalledKind::EncoderInstructLabel,
                "load encoder-instruct-label engine:",
            ),
            (
                InstalledKind::DecoderLogitQwen35(&JEVK5),
                "load decoder-logit-qwen35 engine:",
            ),
        ] {
            let error = empty_model_error(&args, kind);
            assert!(error.starts_with(context), "{error}");
        }
    }

    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn installed_accelerated_families_honor_mlx_selectors() {
        for (kind, selector, context) in [
            (
                InstalledKind::Laya(&LAYA_ENGLISH),
                "--laya-backend",
                "load installed laya mlx model:",
            ),
            (
                InstalledKind::EncoderInstructLabel,
                "--encoder-instruct-label-backend",
                "load encoder-instruct-label mlx engine:",
            ),
            (
                InstalledKind::DecoderLogitQwen35(&JEVK5),
                "--decoder-logit-qwen35-backend",
                "load decoder-logit-qwen35 mlx engine:",
            ),
        ] {
            let args = Args::parse_from(["openkindd", selector, "mlx-fp32"]);
            let error = empty_model_error(&args, kind);
            assert!(error.starts_with(context), "{error}");
        }
    }
}
