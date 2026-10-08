//! Shared loading of installed catalog models for daemon startup and the
//! playground. A manifest is only loadable when its `(name, loader,
//! profile)` identity matches a compiled-in loader; artifact bytes are
//! digest-verified by `ModelStore::acquire_serving` before this module runs,
//! and each family loader re-verifies its pinned digests at load.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use openkind_backends::families::clef::{
    ClefEngine, ClefProfile, CLEF_27B_GGUF, CLEF_FLASH, CLEF_FLASH_GGUF,
};
use openkind_backends::families::decider::{DeciderEngine, DeciderEngineConfig, DECIDER_4B};
use openkind_backends::families::decoder_logit_letter::{
    DecoderLetterEngine, DecoderLetterEngineConfig, PROFILE_ID as DECODER_LETTER_PROFILE,
};
use openkind_backends::families::decoder_logit_llm::{
    DecoderLlmEngine, DecoderLlmEngineConfig, PROFILE_ID as DECODER_LLM_PROFILE,
};
use openkind_backends::families::decoder_logit_qwen3::{
    DecoderLogitQwen3Engine, DecoderLogitQwen3EngineConfig, Qwen3LogitProfile, QWEN3_06B,
    QWEN3_17B, QWEN3_4B,
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
use openkind_backends::families::gemma4::{Gemma4DecisionEngine, Gemma4EngineConfig};
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
use openkind_backends::families::strands_decider::{
    StrandsDeciderEngine, StrandsDeciderEngineConfig, StrandsDeciderProfile, HOBSON_V19, HOBSON_V21,
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
    Manifest, CLEF_27B_GGUF_MODEL_NAME, CLEF_FLASH_GGUF_MODEL_NAME, CLEF_FLASH_MODEL_NAME,
    DECIDER_4B_MODEL_NAME, DECODER_LOGIT_LETTER_MODEL_NAME, DECODER_LOGIT_LLM_MODEL_NAME,
    DECODER_LOGIT_QWEN35_MODEL_NAME, DECODER_LOGIT_QWEN3_06B_MODEL_NAME,
    DECODER_LOGIT_QWEN3_17B_MODEL_NAME, DECODER_LOGIT_QWEN3_4B_MODEL_NAME,
    ENCODER_INSTRUCT_LABEL_MODEL_NAME, ENCODER_NLI_MODEL_NAME, KEV_MODEL_NAME,
    LAYA_ENGLISH_MODEL_NAME, LAYA_MULTILINGUAL_MODEL_NAME, LAYA_TYPED_DECISIONS_MODEL_NAME,
    PLUMB_4B_MODEL_NAME, QWEN35_STATE_FIRST_MODEL_NAME, QWEN3GUARD_MODEL_NAME,
    SCHEMA_SCORER_MODEL_NAME, STRANDS_DECIDER_2B_MODEL_NAME, STRANDS_DECIDER_2B_V21_MODEL_NAME,
    VON_MODEL_NAME, WINNOW_E4B_MODEL_NAME, WINNOW_MODEL_NAME,
};

use crate::args::{
    Args, DecoderLogitQwen35BackendArg, EncoderInstructLabelBackendArg, FamilyBackendArg,
    LayaBackendArg,
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
    StrandsDecider2b(&'static StrandsDeciderProfile),
    DecoderLogitQwen35(&'static Qwen35LogitProfile),
    DecoderLogitQwen3(&'static Qwen3LogitProfile),
    Clef(&'static ClefProfile),
    Decider4b,
    Von,
    Winnow,
    WinnowE4b,
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
        (STRANDS_DECIDER_2B_MODEL_NAME, "strands-decider-2b")
            if profile == HOBSON_V19.profile_id =>
        {
            Some(InstalledKind::StrandsDecider2b(&HOBSON_V19))
        }
        (STRANDS_DECIDER_2B_V21_MODEL_NAME, "strands-decider-2b")
            if profile == HOBSON_V21.profile_id =>
        {
            Some(InstalledKind::StrandsDecider2b(&HOBSON_V21))
        }
        (DECODER_LOGIT_QWEN35_MODEL_NAME, "decoder-logit-qwen35")
            if profile == DECODER_LOGIT_QWEN35_PROFILE =>
        {
            Some(InstalledKind::DecoderLogitQwen35(&JEVK5))
        }
        (PLUMB_4B_MODEL_NAME, "plumb-4b") if profile == PLUMB_4B.profile_id => {
            Some(InstalledKind::DecoderLogitQwen35(&PLUMB_4B))
        }
        (DECODER_LOGIT_QWEN3_06B_MODEL_NAME, "decoder-logit-qwen3-06b")
            if profile == QWEN3_06B.profile_id =>
        {
            Some(InstalledKind::DecoderLogitQwen3(&QWEN3_06B))
        }
        (DECODER_LOGIT_QWEN3_17B_MODEL_NAME, "decoder-logit-qwen3-17b")
            if profile == QWEN3_17B.profile_id =>
        {
            Some(InstalledKind::DecoderLogitQwen3(&QWEN3_17B))
        }
        (DECODER_LOGIT_QWEN3_4B_MODEL_NAME, "decoder-logit-qwen3-4b")
            if profile == QWEN3_4B.profile_id =>
        {
            Some(InstalledKind::DecoderLogitQwen3(&QWEN3_4B))
        }
        (CLEF_FLASH_MODEL_NAME, "clef-flash") if profile == CLEF_FLASH.profile_id => {
            Some(InstalledKind::Clef(&CLEF_FLASH))
        }
        (CLEF_FLASH_GGUF_MODEL_NAME, "clef-flash-gguf")
            if profile == CLEF_FLASH_GGUF.profile_id =>
        {
            Some(InstalledKind::Clef(&CLEF_FLASH_GGUF))
        }
        (CLEF_27B_GGUF_MODEL_NAME, "clef-27b-gguf") if profile == CLEF_27B_GGUF.profile_id => {
            Some(InstalledKind::Clef(&CLEF_27B_GGUF))
        }
        (DECIDER_4B_MODEL_NAME, "decider-4b") if profile == DECIDER_4B.profile_id => {
            Some(InstalledKind::Decider4b)
        }
        (VON_MODEL_NAME, "von") if profile == VON_PROFILE => Some(InstalledKind::Von),
        (WINNOW_MODEL_NAME, "winnow") if profile == WINNOW_PROFILE => Some(InstalledKind::Winnow),
        (WINNOW_E4B_MODEL_NAME, "winnow-e4b")
            if profile == openkind_backends::families::gemma4::PROFILE_ID =>
        {
            Some(InstalledKind::WinnowE4b)
        }
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
    macro_rules! selected {
        ($field:ident, $model_root:expr) => {
            crate::backend::load(
                args.family_args.$field,
                $model_root,
                args.device_ordinals(),
                |backend| {
                    let mut explicit = args.clone();
                    explicit.family_args.$field = backend;
                    load_installed_explicit(&explicit, kind, root, registry)
                },
            )
        };
    }
    match kind {
        InstalledKind::Qwen35StateFirst | InstalledKind::Clef(_) => {
            load_installed_explicit(args, kind, root, registry)
        }
        InstalledKind::Laya(_) => selected!(laya_backend, root),
        InstalledKind::DecoderLetter => selected!(decoder_letter_backend, &root.join("checkpoint")),
        InstalledKind::EncoderNli => selected!(encoder_nli_backend, &root.join("checkpoint")),
        InstalledKind::EncoderInstructLabel => {
            selected!(encoder_instruct_label_backend, &root.join("checkpoint"))
        }
        InstalledKind::DecoderLlm => selected!(decoder_llm_backend, &root.join("checkpoint")),
        InstalledKind::SchemaScorer => selected!(schema_scorer_backend, &root.join("checkpoint")),
        InstalledKind::Qwen3Guard => selected!(qwen3guard_backend, &root.join("checkpoint")),
        InstalledKind::Kev => selected!(kev_backend, root),
        InstalledKind::StrandsDecider2b(_) => selected!(strands_decider_backend, root),
        InstalledKind::DecoderLogitQwen35(_) => {
            selected!(decoder_logit_qwen35_backend, &root.join("checkpoint"))
        }
        InstalledKind::DecoderLogitQwen3(_) => {
            selected!(decoder_logit_qwen3_backend, &root.join("checkpoint"))
        }
        InstalledKind::Decider4b => selected!(decider_4b_backend, &root.join("checkpoint")),
        InstalledKind::Von => selected!(von_backend, root),
        InstalledKind::Winnow => selected!(winnow_backend, &root.join("checkpoint")),
        InstalledKind::WinnowE4b => selected!(winnow_e4b_backend, &root.join("checkpoint")),
    }
}

fn load_installed_explicit(
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
            LayaBackendArg::Auto => unreachable!("auto resolves before loading"),
            LayaBackendArg::NativeCpu => Arc::new(
                LayaEngine::load_with_execution(
                    LayaEngineConfig {
                        profile,
                        model_root: root.to_path_buf(),
                        limits,
                    },
                    FamilyBackendArg::NativeCpu.to_execution(args.device_ordinals())?,
                )
                .context("load installed laya model")?,
            ),
            #[cfg(feature = "cuda")]
            LayaBackendArg::Cuda => Arc::new(
                LayaEngine::load_with_execution(
                    LayaEngineConfig {
                        profile,
                        model_root: root.to_path_buf(),
                        limits,
                    },
                    FamilyBackendArg::Cuda.to_execution(args.device_ordinals())?,
                )
                .context("load installed laya model")?,
            ),
            #[cfg(feature = "onnx")]
            LayaBackendArg::Onnx => Arc::new(
                LayaEngine::load_with_execution(
                    LayaEngineConfig {
                        profile,
                        model_root: root.to_path_buf(),
                        limits,
                    },
                    FamilyBackendArg::Onnx.to_execution(args.device_ordinals())?,
                )
                .context("load installed laya model")?,
            ),
            #[cfg(feature = "onnx")]
            LayaBackendArg::OnnxCuda => Arc::new(
                LayaEngine::load_with_execution(
                    LayaEngineConfig {
                        profile,
                        model_root: root.to_path_buf(),
                        limits,
                    },
                    FamilyBackendArg::OnnxCuda.to_execution(args.device_ordinals())?,
                )
                .context("load installed laya model")?,
            ),
            #[cfg(feature = "onnx")]
            LayaBackendArg::OnnxRocm => Arc::new(
                LayaEngine::load_with_execution(
                    LayaEngineConfig {
                        profile,
                        model_root: root.to_path_buf(),
                        limits,
                    },
                    FamilyBackendArg::OnnxRocm.to_execution(args.device_ordinals())?,
                )
                .context("load installed laya model")?,
            ),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            LayaBackendArg::MlxFp32 => Arc::new(
                LayaMlxEngine::load(LayaMlxEngineConfig {
                    profile,
                    model_root: root.to_path_buf(),
                    limits,
                })
                .context("load installed laya mlx model")?,
            ),
        },
        InstalledKind::DecoderLetter => Arc::new(
            DecoderLetterEngine::load_with_execution(
                DecoderLetterEngineConfig {
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .decoder_letter_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load decoder-logit-letter engine")?,
        ),
        InstalledKind::EncoderNli => Arc::new(
            EncoderNliEngine::load_with_execution(
                EncoderNliEngineConfig {
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .encoder_nli_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load encoder-nli engine")?,
        ),
        InstalledKind::EncoderInstructLabel => {
            match args.family_args.encoder_instruct_label_backend {
                EncoderInstructLabelBackendArg::Auto => {
                    unreachable!("auto resolves before loading")
                }
                EncoderInstructLabelBackendArg::NativeCpu => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root: root.join("checkpoint"),
                            limits,
                        },
                        FamilyBackendArg::NativeCpu.to_execution(args.device_ordinals())?,
                    )
                    .context("load encoder-instruct-label engine")?,
                ),
                #[cfg(feature = "cuda")]
                EncoderInstructLabelBackendArg::Cuda => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root: root.join("checkpoint"),
                            limits,
                        },
                        FamilyBackendArg::Cuda.to_execution(args.device_ordinals())?,
                    )
                    .context("load encoder-instruct-label engine")?,
                ),
                #[cfg(feature = "onnx")]
                EncoderInstructLabelBackendArg::Onnx => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root: root.join("checkpoint"),
                            limits,
                        },
                        FamilyBackendArg::Onnx.to_execution(args.device_ordinals())?,
                    )
                    .context("load encoder-instruct-label engine")?,
                ),
                #[cfg(feature = "onnx")]
                EncoderInstructLabelBackendArg::OnnxCuda => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root: root.join("checkpoint"),
                            limits,
                        },
                        FamilyBackendArg::OnnxCuda.to_execution(args.device_ordinals())?,
                    )
                    .context("load encoder-instruct-label engine")?,
                ),
                #[cfg(feature = "onnx")]
                EncoderInstructLabelBackendArg::OnnxRocm => Arc::new(
                    EncoderInstructLabelEngine::load_with_execution(
                        EncoderInstructLabelEngineConfig {
                            model_root: root.join("checkpoint"),
                            limits,
                        },
                        FamilyBackendArg::OnnxRocm.to_execution(args.device_ordinals())?,
                    )
                    .context("load encoder-instruct-label engine")?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                EncoderInstructLabelBackendArg::MlxFp32 => Arc::new(
                    EncoderInstructLabelMlxEngine::load(EncoderInstructLabelMlxEngineConfig {
                        model_root: root.join("checkpoint"),
                        limits,
                    })
                    .context("load encoder-instruct-label mlx engine")?,
                ),
            }
        }
        InstalledKind::DecoderLlm => Arc::new(
            DecoderLlmEngine::load_with_execution(
                DecoderLlmEngineConfig {
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .decoder_llm_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load decoder-logit-llm engine")?,
        ),
        InstalledKind::SchemaScorer => Arc::new(
            SchemaScorerEngine::load_with_execution(
                SchemaScorerEngineConfig {
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .schema_scorer_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load schema-scorer engine")?,
        ),
        InstalledKind::Qwen3Guard => Arc::new(
            Qwen3GuardEngine::load_with_execution(
                Qwen3GuardEngineConfig {
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .qwen3guard_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load qwen3guard engine")?,
        ),
        InstalledKind::Kev => Arc::new(
            KevEngine::load_with_execution(
                KevEngineConfig {
                    model_root: root.join("adapter"),
                    base_root: root.join("base"),
                    limits,
                },
                args.family_args
                    .kev_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load kev engine")?,
        ),
        InstalledKind::StrandsDecider2b(profile) => Arc::new(
            StrandsDeciderEngine::load_with_execution(
                StrandsDeciderEngineConfig {
                    model_root: root.join("adapter"),
                    base_root: root.join("base"),
                    limits,
                    profile: Some(profile),
                },
                args.family_args
                    .strands_decider_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load strands-decider-2b engine")?,
        ),
        InstalledKind::DecoderLogitQwen35(profile) => {
            match args.family_args.decoder_logit_qwen35_backend {
                DecoderLogitQwen35BackendArg::Auto => unreachable!("auto resolves before loading"),
                DecoderLogitQwen35BackendArg::NativeCpu => Arc::new(
                    DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
                        profile,
                        model_root: root.join("checkpoint"),
                        limits,
                    })
                    .context("load decoder-logit-qwen35 engine")?,
                ),
                #[cfg(feature = "cuda")]
                DecoderLogitQwen35BackendArg::Cuda => Arc::new(
                    DecoderLogitQwen35Engine::load_with_execution(
                        DecoderLogitQwen35EngineConfig {
                            profile,
                            model_root: root.join("checkpoint"),
                            limits,
                        },
                        FamilyBackendArg::Cuda.to_execution(args.device_ordinals())?,
                    )
                    .context("load decoder-logit-qwen35 engine")?,
                ),
                #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
                DecoderLogitQwen35BackendArg::MlxFp32 => Arc::new(
                    DecoderLogitQwen35MlxEngine::load(DecoderLogitQwen35MlxEngineConfig {
                        profile,
                        model_root: root.join("checkpoint"),
                        limits,
                    })
                    .context("load decoder-logit-qwen35 mlx engine")?,
                ),
            }
        }
        InstalledKind::Von => Arc::new(
            VonEngine::load_with_execution(
                VonEngineConfig {
                    profile: &VON,
                    model_root: root.to_path_buf(),
                    limits,
                },
                args.family_args
                    .von_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load von engine")?,
        ),
        InstalledKind::DecoderLogitQwen3(profile) => Arc::new(
            DecoderLogitQwen3Engine::load_with_execution(
                DecoderLogitQwen3EngineConfig {
                    profile,
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .decoder_logit_qwen3_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load decoder-logit-qwen3 engine")?,
        ),
        InstalledKind::Clef(profile) => Arc::new(
            ClefEngine::load(root.join("checkpoint"), profile, limits)
                .context("load clef engine")?,
        ),
        InstalledKind::Decider4b => Arc::new(
            DeciderEngine::load_with_execution(
                DeciderEngineConfig {
                    profile: &DECIDER_4B,
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .decider_4b_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load decider-4b engine")?,
        ),
        InstalledKind::WinnowE4b => Arc::new(
            Gemma4DecisionEngine::load_with_execution(
                Gemma4EngineConfig {
                    model_root: root.join("checkpoint"),
                    limits,
                },
                args.family_args
                    .winnow_e4b_backend
                    .to_execution(args.device_ordinals())?,
            )
            .context("load winnow-e4b engine")?,
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
                WinnowEngine::load_with_execution(
                    WinnowEngineConfig {
                        model_root: root.join("checkpoint"),
                        adapter_path: root.join("adapter.safetensors"),
                        limits,
                    },
                    siblings,
                    args.family_args
                        .winnow_backend
                        .to_execution(args.device_ordinals())?,
                )
                .context("load winnow engine")?,
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
            Err(error) => format!("{error:#}"),
        }
    }

    #[test]
    fn installed_accelerated_families_auto_uses_cpu_when_probes_are_unavailable() {
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

#[cfg(test)]
mod classifier_tests {
    use super::*;

    fn manifest(name: &str, loader: &str, profile: &str) -> Manifest {
        Manifest {
            schema: "openkind-model/v1".into(),
            name: name.into(),
            profile_id: profile.into(),
            loader_id: loader.into(),
            description: "fixture".into(),
            release_date: "2026-01-01".into(),
            support_status: "rust-loadable".into(),
            question_types: vec!["choice".into()],
            artifacts: vec![],
        }
    }

    /// Every curated identity classifies to exactly one loader kind, and
    /// any drift in the (name, loader, profile) triple fails closed.
    #[test]
    fn installed_kind_classifies_curated_identities_and_rejects_drift() {
        let cases: Vec<(Manifest, &str)> = vec![
            (
                manifest(
                    QWEN35_STATE_FIRST_MODEL_NAME,
                    "qwen35-state-first",
                    openkind_backends::qwen35::PROFILE_ID,
                ),
                "qwen35",
            ),
            (
                manifest(
                    LAYA_ENGLISH_MODEL_NAME,
                    "laya-english",
                    LAYA_ENGLISH.profile_id,
                ),
                "laya",
            ),
            (
                manifest(
                    LAYA_MULTILINGUAL_MODEL_NAME,
                    "laya-multilingual",
                    LAYA_MULTILINGUAL.profile_id,
                ),
                "laya",
            ),
            (
                manifest(
                    LAYA_TYPED_DECISIONS_MODEL_NAME,
                    "laya-typed-decisions",
                    LAYA_TYPED_DECISIONS.profile_id,
                ),
                "laya",
            ),
            (
                manifest(
                    DECODER_LOGIT_LETTER_MODEL_NAME,
                    "decoder-logit-letter",
                    DECODER_LETTER_PROFILE,
                ),
                "letter",
            ),
            (
                manifest(ENCODER_NLI_MODEL_NAME, "encoder-nli", ENCODER_NLI_PROFILE),
                "encoder-nli",
            ),
            (
                manifest(
                    ENCODER_INSTRUCT_LABEL_MODEL_NAME,
                    "encoder-instruct-label",
                    ENCODER_INSTRUCT_LABEL_PROFILE,
                ),
                "label",
            ),
            (
                manifest(
                    DECODER_LOGIT_LLM_MODEL_NAME,
                    "decoder-logit-llm",
                    DECODER_LLM_PROFILE,
                ),
                "llm",
            ),
            (
                manifest(
                    SCHEMA_SCORER_MODEL_NAME,
                    "schema-scorer",
                    SCHEMA_SCORER_PROFILE,
                ),
                "schema",
            ),
            (
                manifest(QWEN3GUARD_MODEL_NAME, "qwen3guard", QWEN3GUARD_PROFILE),
                "guard",
            ),
            (manifest(KEV_MODEL_NAME, "kev", KEV_PROFILE), "kev"),
            (
                manifest(
                    STRANDS_DECIDER_2B_MODEL_NAME,
                    "strands-decider-2b",
                    HOBSON_V19.profile_id,
                ),
                "strands-v19",
            ),
            (
                manifest(
                    STRANDS_DECIDER_2B_V21_MODEL_NAME,
                    "strands-decider-2b",
                    HOBSON_V21.profile_id,
                ),
                "strands-v21",
            ),
            (
                manifest(
                    DECODER_LOGIT_QWEN35_MODEL_NAME,
                    "decoder-logit-qwen35",
                    DECODER_LOGIT_QWEN35_PROFILE,
                ),
                "qwen35-logit",
            ),
            (
                manifest(PLUMB_4B_MODEL_NAME, "plumb-4b", PLUMB_4B.profile_id),
                "plumb",
            ),
            (
                manifest(
                    DECODER_LOGIT_QWEN3_06B_MODEL_NAME,
                    "decoder-logit-qwen3-06b",
                    QWEN3_06B.profile_id,
                ),
                "qwen3",
            ),
            (
                manifest(CLEF_FLASH_MODEL_NAME, "clef-flash", CLEF_FLASH.profile_id),
                "clef",
            ),
            (
                manifest(
                    CLEF_FLASH_GGUF_MODEL_NAME,
                    "clef-flash-gguf",
                    CLEF_FLASH_GGUF.profile_id,
                ),
                "clef-gguf",
            ),
            (
                manifest(
                    CLEF_27B_GGUF_MODEL_NAME,
                    "clef-27b-gguf",
                    CLEF_27B_GGUF.profile_id,
                ),
                "clef-27b",
            ),
            (
                manifest(DECIDER_4B_MODEL_NAME, "decider-4b", DECIDER_4B.profile_id),
                "decider",
            ),
            (manifest(VON_MODEL_NAME, "von", VON_PROFILE), "von"),
            (
                manifest(WINNOW_MODEL_NAME, "winnow", WINNOW_PROFILE),
                "winnow",
            ),
            (
                manifest(
                    WINNOW_E4B_MODEL_NAME,
                    "winnow-e4b",
                    openkind_backends::families::gemma4::PROFILE_ID,
                ),
                "gemma4",
            ),
        ];
        assert_eq!(cases.len(), 23, "one row per curated installed identity");
        for (manifest, label) in &cases {
            assert!(
                installed_kind(manifest).is_some(),
                "{label}: a curated identity must classify"
            );
        }

        // Mutating any leg of the triple must fail closed so a tampered or
        // foreign installation never silently loads.
        let good = manifest(
            QWEN35_STATE_FIRST_MODEL_NAME,
            "qwen35-state-first",
            openkind_backends::qwen35::PROFILE_ID,
        );
        let mut drifted = good.clone();
        drifted.profile_id = "0".repeat(20);
        assert!(installed_kind(&drifted).is_none(), "wrong profile");

        let mut drifted = good.clone();
        drifted.loader_id = "impostor".into();
        assert!(installed_kind(&drifted).is_none(), "wrong loader");

        let mut drifted = good.clone();
        drifted.name = "fixture:v1".into();
        assert!(installed_kind(&drifted).is_none(), "wrong name");

        assert!(installed_kind(&manifest("fixture:v1", "fixture", "p")).is_none());
    }
}
