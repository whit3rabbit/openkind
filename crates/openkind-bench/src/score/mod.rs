//! The `score` subcommand: time a decision workload end to end and emit
//! row-level predictions plus a provenance summary.
//!
//! Timing scope follows `docs/BENCHMARKS.md`: the measured region covers
//! request construction, validation, engine dispatch, and answer extraction.
//! Model loading and result-file writes are excluded; model-load wall time is
//! reported separately per strategy. An explicit no-warmup mode supports
//! first-request and history probes.
//!
//! Execution paths mirror the prior-art systems benchmark mapping: fresh
//! per-row requests (`--no-group`) correspond to repeated-full direct
//! scoring; state-grouped requests let the shared-state strategies
//! (`nested_sequential`, `nested_batched`, and the measured scheduler) pay
//! the root prefill once. On the native engine, strategies other than
//! `choose_strategy` are forced through the scheduler's diagnostic override;
//! cross-strategy answer equality is asserted per workload.

mod execution;
mod summary;
mod types;

use std::fs;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use openkind_backends::families::decoder_logit_letter::{
    DecoderLetterEngine, DecoderLetterEngineConfig,
};
use openkind_backends::families::decoder_logit_llm::{DecoderLlmEngine, DecoderLlmEngineConfig};
use openkind_backends::families::decoder_logit_qwen35::{
    DecoderLogitQwen35Engine, DecoderLogitQwen35EngineConfig,
};
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
use openkind_backends::families::qwen3guard::{Qwen3GuardEngine, Qwen3GuardEngineConfig};
use openkind_backends::families::schema_scorer::{SchemaScorerEngine, SchemaScorerEngineConfig};
use openkind_backends::families::strands_decider::{
    StrandsDeciderEngine, StrandsDeciderEngineConfig,
};
use openkind_backends::families::support::FamilyLimits;
use openkind_backends::qwen35::{Qwen35DecisionEngine, Qwen35EngineConfig, SchedulerConfig};
use openkind_engine::{DecisionEngine, EngineRegistry};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(crate) use types::validate_strategy_selection;
pub use types::{
    is_family_engine_public, EngineKind, ScoreArgs, ScoreOutcome, StrategySpec, DEFAULT_STRATEGIES,
    STRATEGY_HELP,
};

use crate::workload;

/// Load the pinned laya engine for one `EngineKind`, choosing the MLX
/// backend for the `-mlx-fp32` kinds (feature `mlx`, macOS arm64) and the
/// candle CPU backend otherwise.
fn load_laya_engine(
    engine: EngineKind,
    model_root: std::path::PathBuf,
    limits: FamilyLimits,
) -> Result<Arc<dyn DecisionEngine>> {
    use openkind_backends::families::laya;
    let (profile, mlx) = match engine {
        EngineKind::LayaEnglish => (&laya::LAYA_ENGLISH, false),
        EngineKind::LayaMultilingual => (&laya::LAYA_MULTILINGUAL, false),
        EngineKind::LayaTypedDecisions => (&laya::LAYA_TYPED_DECISIONS, false),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaEnglishMlxFp32 => (&laya::LAYA_ENGLISH, true),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaMultilingualMlxFp32 => (&laya::LAYA_MULTILINGUAL, true),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaTypedDecisionsMlxFp32 => (&laya::LAYA_TYPED_DECISIONS, true),
        _ => unreachable!("non-laya engine reached the laya loader"),
    };
    if mlx {
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        return Ok(Arc::new(
            laya::LayaMlxEngine::load(laya::LayaMlxEngineConfig {
                profile,
                model_root,
                limits,
            })
            .map_err(|error| anyhow::anyhow!("load {} mlx engine: {error}", profile.loader_id))?,
        ) as Arc<dyn DecisionEngine>);
    }
    Ok(Arc::new(
        laya::LayaEngine::load(laya::LayaEngineConfig {
            profile,
            model_root,
            limits,
        })
        .map_err(|error| anyhow::anyhow!("load {} engine: {error}", profile.loader_id))?,
    ))
}

/// Execute one scoring run.
///
/// # Errors
/// Returns an error for workload, artifact, or I/O failures. Timed engine
/// errors surface as errors too: a benchmark run that cannot score is not a
/// benchmark result.
pub fn run_score(args: &ScoreArgs) -> Result<ScoreOutcome> {
    let workload = workload::load_workload(&args.input)?;
    eprintln!("[bench] workload {}: {}", args.input.display(), workload);
    if args.history_aba {
        anyhow::ensure!(
            workload.rows.len() == 2,
            "--history-aba requires exactly two rows"
        );
        anyhow::ensure!(!args.group, "--history-aba requires --no-group");
        anyhow::ensure!(!args.warmup, "--history-aba requires --no-warmup");
        anyhow::ensure!(args.reps == 1, "--history-aba requires --reps 1");
        anyhow::ensure!(
            args.engine == EngineKind::Mock
                || types::is_family_engine(args.engine)
                || args.strategies.len() == 1,
            "--history-aba requires one native execution strategy"
        );
    }
    let groups = if args.history_aba {
        vec![vec![0], vec![1], vec![0]]
    } else if args.group {
        workload::state_groups(&workload.rows)?
    } else {
        (0..workload.rows.len()).map(|index| vec![index]).collect()
    };
    eprintln!(
        "[bench] {} groups over {} rows (grouping={})",
        groups.len(),
        workload.rows.len(),
        args.group
    );
    anyhow::ensure!(args.reps >= 1, "reps must be at least 1");
    validate_strategy_selection(args.engine, &args.strategies)?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    let mut strategy_reports = Vec::new();
    let mut predictions_per_strategy = Vec::new();
    let mut parity_clean: Option<bool> = None;
    let mut answers_by_row: Option<Vec<Value>> = None;

    if args.engine == EngineKind::Mock {
        let registry = execution::mock_registry();
        let (report, predictions) = runtime.block_on(execution::run_strategy_pass(
            &registry,
            &workload.rows,
            &groups,
            args,
            execution::StrategyPass {
                label: "mock",
                forced: None,
                warmup: false,
            },
        ))?;
        strategy_reports.push(report);
        predictions_per_strategy.push(predictions);
    } else if types::is_family_engine(args.engine) {
        let needs_model_root =
            !matches!(args.engine, EngineKind::RouterScript | EngineKind::Winnow);
        let model_root = if needs_model_root {
            Some(args.model_root.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--model-root is required for the {} engine",
                    types::engine_slug(args.engine)
                )
            })?)
        } else {
            None
        };
        let load_started = Instant::now();
        let engine: Arc<dyn DecisionEngine> = match args.engine {
            EngineKind::DecoderLetter => Arc::new(
                DecoderLetterEngine::load(DecoderLetterEngineConfig {
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load decoder-letter engine: {error}"))?,
            ),
            EngineKind::DecoderLogitQwen35 => Arc::new(
                DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
                    profile: &openkind_backends::families::decoder_logit_qwen35::JEVK5,
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load decoder-logit-qwen35 engine: {error}"))?,
            ),
            EngineKind::Plumb4b => Arc::new(
                DecoderLogitQwen35Engine::load(DecoderLogitQwen35EngineConfig {
                    profile: &openkind_backends::families::decoder_logit_qwen35::PLUMB_4B,
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load plumb-4b engine: {error}"))?,
            ),
            EngineKind::DecoderLogitQwen306b
            | EngineKind::DecoderLogitQwen317b
            | EngineKind::DecoderLogitQwen34b => {
                let profile = match args.engine {
                    EngineKind::DecoderLogitQwen306b => {
                        &openkind_backends::families::decoder_logit_qwen3::QWEN3_06B
                    }
                    EngineKind::DecoderLogitQwen317b => {
                        &openkind_backends::families::decoder_logit_qwen3::QWEN3_17B
                    }
                    _ => &openkind_backends::families::decoder_logit_qwen3::QWEN3_4B,
                };
                Arc::new(
                    openkind_backends::families::decoder_logit_qwen3::DecoderLogitQwen3Engine::load(
                        openkind_backends::families::decoder_logit_qwen3::DecoderLogitQwen3EngineConfig {
                            profile,
                            model_root: model_root.expect("gated").clone(),
                            limits: FamilyLimits {
                                max_concurrent_requests: 1,
                                max_queued_requests: 0,
                                retry_after_ms: 250,
                                evaluation_timeout: None,
                            },
                        },
                    )
                    .map_err(|error| {
                        anyhow::anyhow!("load decoder-logit-qwen3 engine: {error}")
                    })?,
                )
            }
            EngineKind::ClefFlash | EngineKind::ClefFlashGguf | EngineKind::Clef27bGguf => {
                let profile = match args.engine {
                    EngineKind::ClefFlash => &openkind_backends::families::clef::CLEF_FLASH,
                    EngineKind::ClefFlashGguf => {
                        &openkind_backends::families::clef::CLEF_FLASH_GGUF
                    }
                    _ => &openkind_backends::families::clef::CLEF_27B_GGUF,
                };
                Arc::new(
                    openkind_backends::families::clef::ClefEngine::load(
                        model_root.expect("gated").clone(),
                        profile,
                        FamilyLimits {
                            max_concurrent_requests: 1,
                            max_queued_requests: 0,
                            retry_after_ms: 250,
                            evaluation_timeout: None,
                        },
                    )
                    .map_err(|error| anyhow::anyhow!("load clef engine: {error}"))?,
                )
            }
            EngineKind::Decider4b => Arc::new(
                openkind_backends::families::decider::DeciderEngine::load(
                    openkind_backends::families::decider::DeciderEngineConfig {
                        profile: &openkind_backends::families::decider::DECIDER_4B,
                        model_root: model_root.expect("gated").clone(),
                        limits: FamilyLimits {
                            max_concurrent_requests: 1,
                            max_queued_requests: 0,
                            retry_after_ms: 250,
                            evaluation_timeout: None,
                        },
                    },
                )
                .map_err(|error| anyhow::anyhow!("load decider-4b engine: {error}"))?,
            ),
            EngineKind::WinnowE4b => Arc::new(
                openkind_backends::families::gemma4::Gemma4DecisionEngine::load(
                    openkind_backends::families::gemma4::Gemma4EngineConfig {
                        model_root: model_root.expect("gated").clone(),
                        limits: FamilyLimits {
                            max_concurrent_requests: 1,
                            max_queued_requests: 0,
                            retry_after_ms: 250,
                            evaluation_timeout: None,
                        },
                    },
                )
                .map_err(|error| anyhow::anyhow!("load winnow-e4b engine: {error}"))?,
            ),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineKind::EncoderInstructLabelMlxFp32 => Arc::new(
                EncoderInstructLabelMlxEngine::load(EncoderInstructLabelMlxEngineConfig {
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| {
                    anyhow::anyhow!("load encoder-instruct-label mlx engine: {error}")
                })?,
            ),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineKind::DecoderLogitQwen35MlxFp32 => Arc::new(
                DecoderLogitQwen35MlxEngine::load(DecoderLogitQwen35MlxEngineConfig {
                    profile: &openkind_backends::families::decoder_logit_qwen35::JEVK5,
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| {
                    anyhow::anyhow!("load decoder-logit-qwen35 mlx engine: {error}")
                })?,
            ),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineKind::Plumb4bMlxFp32 => Arc::new(
                DecoderLogitQwen35MlxEngine::load(DecoderLogitQwen35MlxEngineConfig {
                    profile: &openkind_backends::families::decoder_logit_qwen35::PLUMB_4B,
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load plumb-4b mlx engine: {error}"))?,
            ),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            EngineKind::LayaEnglishMlxFp32
            | EngineKind::LayaMultilingualMlxFp32
            | EngineKind::LayaTypedDecisionsMlxFp32
            | EngineKind::LayaEnglish
            | EngineKind::LayaMultilingual
            | EngineKind::LayaTypedDecisions => {
                let limits = FamilyLimits {
                    max_concurrent_requests: 1,
                    max_queued_requests: 0,
                    retry_after_ms: 250,
                    evaluation_timeout: None,
                };
                load_laya_engine(args.engine, model_root.expect("gated").clone(), limits)?
            }
            #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
            EngineKind::LayaEnglish
            | EngineKind::LayaMultilingual
            | EngineKind::LayaTypedDecisions => {
                let limits = FamilyLimits {
                    max_concurrent_requests: 1,
                    max_queued_requests: 0,
                    retry_after_ms: 250,
                    evaluation_timeout: None,
                };
                load_laya_engine(args.engine, model_root.expect("gated").clone(), limits)?
            }
            EngineKind::Von => Arc::new(
                openkind_backends::families::von::VonEngine::load(
                    openkind_backends::families::von::VonEngineConfig {
                        profile: &openkind_backends::families::von::VON,
                        model_root: model_root.expect("gated").clone(),
                        limits: FamilyLimits {
                            max_concurrent_requests: 1,
                            max_queued_requests: 0,
                            retry_after_ms: 250,
                            evaluation_timeout: None,
                        },
                    },
                )
                .map_err(|error| anyhow::anyhow!("load von engine: {error}"))?,
            ),
            EngineKind::EncoderNli => Arc::new(
                EncoderNliEngine::load(EncoderNliEngineConfig {
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load encoder-nli engine: {error}"))?,
            ),
            EngineKind::EncoderInstructLabel => Arc::new(
                EncoderInstructLabelEngine::load(EncoderInstructLabelEngineConfig {
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load encoder-instruct-label engine: {error}"))?,
            ),
            EngineKind::DecoderLlm => Arc::new(
                DecoderLlmEngine::load(DecoderLlmEngineConfig {
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load decoder-llm engine: {error}"))?,
            ),
            EngineKind::SchemaScorer => Arc::new(
                SchemaScorerEngine::load(SchemaScorerEngineConfig {
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load schema-scorer engine: {error}"))?,
            ),
            // The router composite is benchmarked over mock siblings: this
            // run measures routing overhead only, not sibling throughput.
            EngineKind::RouterScript => {
                let rules = openkind_backends::families::router_script::ScriptRuleTable::parse(
                    "latin=mock-latin,cyrillic=mock-cyrillic,default=mock-latin",
                )
                .map_err(|error| anyhow::anyhow!("parse bench router rules: {error}"))?;
                let mut siblings: std::collections::HashMap<String, Arc<dyn DecisionEngine>> =
                    std::collections::HashMap::new();
                siblings.insert(
                    "mock-latin".to_owned(),
                    Arc::new(openkind_engine::MockEngine::new()),
                );
                siblings.insert(
                    "mock-cyrillic".to_owned(),
                    Arc::new(openkind_engine::MockEngine::new()),
                );
                Arc::new(
                    openkind_backends::families::router_script::RouterScriptEngine::new(
                        rules, &siblings,
                    )
                    .map_err(|error| anyhow::anyhow!("compose router-script engine: {error}"))?,
                )
            }
            EngineKind::Qwen3Guard => Arc::new(
                Qwen3GuardEngine::load(Qwen3GuardEngineConfig {
                    model_root: model_root.expect("gated").clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                })
                .map_err(|error| anyhow::anyhow!("load qwen3guard engine: {error}"))?,
            ),
            // The winnow router is benchmarked over mock siblings: this run
            // measures the learned routing pass only, not sibling throughput.
            EngineKind::Kev => {
                let model_root = args.model_root.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--model-root is required for the kev engine")
                })?;
                let base_root = args.checkpoint_root.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--checkpoint-root is required for the kev engine")
                })?;
                Arc::new(
                    KevEngine::load(KevEngineConfig {
                        model_root: model_root.clone(),
                        base_root: base_root.clone(),
                        limits: FamilyLimits {
                            max_concurrent_requests: 1,
                            max_queued_requests: 0,
                            retry_after_ms: 250,
                            evaluation_timeout: None,
                        },
                    })
                    .map_err(|error| anyhow::anyhow!("load kev engine: {error}"))?,
                )
            }
            EngineKind::StrandsDecider2b => {
                let model_root = args.model_root.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--model-root is required for the strands-decider-2b engine")
                })?;
                let base_root = args.checkpoint_root.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "--checkpoint-root is required for the strands-decider-2b engine"
                    )
                })?;
                Arc::new(
                    StrandsDeciderEngine::load(StrandsDeciderEngineConfig {
                        model_root: model_root.clone(),
                        base_root: base_root.clone(),
                        limits: FamilyLimits {
                            max_concurrent_requests: 1,
                            max_queued_requests: 0,
                            retry_after_ms: 250,
                            evaluation_timeout: None,
                        },
                    })
                    .map_err(|error| anyhow::anyhow!("load strands-decider-2b engine: {error}"))?,
                )
            }
            EngineKind::Winnow => {
                let model_root = args.model_root.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--model-root is required for the winnow engine")
                })?;
                let adapter = args.adapter.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--adapter is required for the winnow engine")
                })?;
                let config = openkind_backends::families::winnow::WinnowEngineConfig {
                    model_root: model_root.clone(),
                    adapter_path: adapter.clone(),
                    limits: FamilyLimits {
                        max_concurrent_requests: 1,
                        max_queued_requests: 0,
                        retry_after_ms: 250,
                        evaluation_timeout: None,
                    },
                };
                let siblings = vec![
                    (
                        "mock-english".to_owned(),
                        Arc::new(openkind_engine::MockEngine::new()) as Arc<dyn DecisionEngine>,
                    ),
                    (
                        "mock-multilingual".to_owned(),
                        Arc::new(openkind_engine::MockEngine::new()) as Arc<dyn DecisionEngine>,
                    ),
                ];
                Arc::new(
                    openkind_backends::families::winnow::WinnowEngine::load(config, siblings)
                        .map_err(|error| anyhow::anyhow!("load winnow engine: {error}"))?,
                )
            }
            other => anyhow::bail!("engine {} is not wired for family scoring", other as u32),
        };
        let model_load_seconds = load_started.elapsed().as_secs_f64();
        let mut registry = EngineRegistry::new();
        registry.register(types::BENCH_ALIAS, engine);
        eprintln!(
            "[bench] engine {} starting",
            types::engine_slug(args.engine)
        );
        let (report, predictions) = runtime.block_on(execution::run_strategy_pass(
            &registry,
            &workload.rows,
            &groups,
            args,
            execution::StrategyPass {
                label: types::engine_slug(args.engine),
                forced: None,
                warmup: args.warmup,
            },
        ))?;
        strategy_reports.push(execution::report_with_load(report, model_load_seconds));
        predictions_per_strategy.push(predictions);
    } else {
        let backend = types::native_backend(args.engine);
        let bundle_root = args.bundle_root.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--bundle-root is required for the {} engine",
                backend.as_str()
            )
        })?;
        let checkpoint_root = args.checkpoint_root.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--checkpoint-root is required for the {} engine",
                backend.as_str()
            )
        })?;
        let tokenizer_path = args.tokenizer_path.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--tokenizer is required for the {} engine",
                backend.as_str()
            )
        })?;
        for &spec in &args.strategies {
            let scheduler = SchedulerConfig::for_pinned_profile(
                SchedulerConfig::LOWEST_MEASURED_SHARED_SAVINGS_RATIO,
                None,
            )
            .with_forced_strategy(spec.scheduler_override());
            let load_started = Instant::now();
            let engine = Qwen35DecisionEngine::load(Qwen35EngineConfig {
                bundle_root: bundle_root.clone(),
                checkpoint_root: checkpoint_root.clone(),
                tokenizer_path: tokenizer_path.clone(),
                backend,
                scheduler,
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 250,
                evaluation_timeout: None,
            })
            .map_err(|error| anyhow::anyhow!("load {} engine: {error}", backend.as_str()))?;
            let model_load_seconds = load_started.elapsed().as_secs_f64();
            let mut registry = EngineRegistry::new();
            registry.register(types::BENCH_ALIAS, std::sync::Arc::new(engine));

            eprintln!(
                "[bench] engine {} strategy {} starting",
                backend.as_str(),
                spec.name()
            );
            let (report, predictions) = runtime.block_on(execution::run_strategy_pass(
                &registry,
                &workload.rows,
                &groups,
                args,
                execution::StrategyPass {
                    label: spec.name(),
                    forced: Some(spec),
                    warmup: args.warmup,
                },
            ))?;
            strategy_reports.push(execution::report_with_load(report, model_load_seconds));
            predictions_per_strategy.push(predictions);

            let answers = execution::answers_of(&predictions_per_strategy);
            if let Some(previous) = &answers_by_row {
                if previous != &answers && parity_clean != Some(false) {
                    eprintln!(
                        "[bench] strategy {}: answer parity violation against earlier \
                         strategies",
                        spec.name()
                    );
                    parity_clean = Some(false);
                }
            } else {
                answers_by_row = Some(answers);
            }
            if parity_clean.is_none() {
                parity_clean = Some(true);
            }
        }
    }

    let mut summary =
        summary::build_summary(args, &workload, &groups, &strategy_reports, parity_clean);
    let prediction_hashes: serde_json::Map<String, Value> = strategy_reports
        .iter()
        .zip(&predictions_per_strategy)
        .map(|(report, predictions)| {
            (
                report["strategy"].as_str().unwrap_or("strategy").to_owned(),
                Value::String(format!("{:x}", Sha256::digest(predictions.as_bytes()))),
            )
        })
        .collect();
    summary["prediction_sha256"] = Value::Object(prediction_hashes);
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;
    let summary_path = args
        .output_dir
        .join(format!("summary-{}.json", types::engine_slug(args.engine)));
    summary::write_json(&summary_path, &summary, args.pretty)?;
    let mut prediction_paths = Vec::with_capacity(predictions_per_strategy.len());
    for (report, predictions) in strategy_reports.iter().zip(&predictions_per_strategy) {
        let path = args.output_dir.join(format!(
            "predictions-{}-{}.jsonl",
            types::engine_slug(args.engine),
            report["strategy"].as_str().unwrap_or("strategy")
        ));
        fs::write(&path, predictions).with_context(|| format!("write {}", path.display()))?;
        prediction_paths.push(path);
    }
    eprintln!("[bench] summary {}", summary_path.display());
    Ok(ScoreOutcome { summary })
}
