//! Summary construction, metadata extraction, and JSON file output.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use openkind_backends::qwen35::PROFILE_ID;
use openkind_runtime::peak_resident_bytes;
use serde_json::{json, Value};

use super::types::{engine_slug, family_identity, native_backend, EngineKind, ScoreArgs};
use crate::workload::Workload;

pub(crate) fn build_summary(
    args: &ScoreArgs,
    fixture: &Workload,
    groups: &[Vec<usize>],
    strategy_reports: &[Value],
    parity_clean: Option<bool>,
) -> Value {
    let (engine_id, profile, model_revision, bundle_version) =
        if let Some((slug, family_profile, family_revision)) = family_identity(args.engine) {
            (
                slug,
                json!(family_profile),
                json!(family_revision),
                Value::Null,
            )
        } else if args.engine == EngineKind::Mock {
            ("mock", Value::Null, Value::Null, Value::Null)
        } else {
            (
                native_backend(args.engine).as_str(),
                json!(PROFILE_ID),
                model_revision(args),
                bundle_version(args),
            )
        };
    json!({
        "schema": "openkind-bench/v1",
        "engine": engine_id,
        "engine_variant": engine_slug(args.engine),
        "measurement_scope": match args.engine {
            EngineKind::RouterScript | EngineKind::Winnow => "routing_overhead_with_mock_siblings",
            EngineKind::Mock => "mock_request_path",
            _ => "model_request_path",
        },
        "profile_id": profile,
        "model_revision": model_revision,
        "bundle_version": bundle_version,
        "fixture": {
            "path": args.input.display().to_string(),
            "sha256": fixture.sha256,
            "rows": fixture.rows.len(),
            "groups": groups.len(),
        },
        "grouping": if args.group { "per-state" } else { "per-row" },
        "reps": args.reps,
        "warmup": args.warmup && args.engine != EngineKind::Mock,
        "history_aba": args.history_aba,
        "host": args.host.clone().unwrap_or_else(default_host),
        "commit": args.commit.clone(),
        "timing_scope": if args.warmup && args.engine != EngineKind::Mock {
            "request construction, validation, dispatch, and answer extraction; \
             excludes model load (reported per strategy), result writes, and the \
             untimed warmup pass; warm process"
        } else {
            "request construction, validation, dispatch, and answer extraction; \
             excludes model load (reported per strategy) and result writes; no warmup pass"
        },
        "peak_resident_bytes": peak_resident_bytes().ok(),
        "host_hardware": serde_json::to_value(openkind_runtime::host_hardware())
            .unwrap_or(Value::Null),
        "context": context_limits(args.engine),
        "cross_strategy_answer_parity_clean": parity_clean,
        "strategies": strategy_reports,
        "notes": {
            "percentiles": "p50 is the median rep total and p95 is the ceil(0.95*n)-1 sample; \
                            small rep counts are reported alongside as samples_seconds",
            "request_latency_ms": "latency is per request; grouped rows share their \
                                   request's latency and are not independent timings; \
                                   includes request construction through answer extraction",
            "input_tokens_per_second": "total input tokens across all timed repetitions \
                                        divided by their total wall time",
            "cpu_time_seconds": "process-wide user+system CPU time diff across the timed \
                                 region, excluding warmup and model load; \
                                 avg_cpu_percent exceeds 100 when multiple threads run",
        },
    })
}

pub(crate) fn write_json(path: &Path, value: &Value, pretty: bool) -> Result<()> {
    let body = if pretty {
        serde_json::to_string_pretty(value).context("serialize summary")?
    } else {
        serde_json::to_string(value).context("serialize summary")?
    };
    fs::write(path, body).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Context-token limits the engine accepts, so request-path evidence can be
/// compared across profiles with different frozen input budgets.
fn context_limits(engine: EngineKind) -> Value {
    match engine {
        EngineKind::Qwen35 => json!({
            "max_candidate_sequence_tokens": openkind_backends::qwen35::MAX_SEQUENCE_TOKENS,
        }),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::Qwen35MlxFp32 | EngineKind::Qwen35MlxBf16 => json!({
            "max_candidate_sequence_tokens": openkind_backends::qwen35::MAX_SEQUENCE_TOKENS,
        }),
        EngineKind::LayaEnglish => laya_context(&openkind_backends::families::laya::LAYA_ENGLISH),
        EngineKind::LayaMultilingual => {
            laya_context(&openkind_backends::families::laya::LAYA_MULTILINGUAL)
        }
        EngineKind::LayaTypedDecisions => {
            laya_context(&openkind_backends::families::laya::LAYA_TYPED_DECISIONS)
        }
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaEnglishMlxFp32 => {
            laya_context(&openkind_backends::families::laya::LAYA_ENGLISH)
        }
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaMultilingualMlxFp32 => {
            laya_context(&openkind_backends::families::laya::LAYA_MULTILINGUAL)
        }
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        EngineKind::LayaTypedDecisionsMlxFp32 => {
            laya_context(&openkind_backends::families::laya::LAYA_TYPED_DECISIONS)
        }
        _ => Value::Null,
    }
}

/// Frozen per-sequence input budgets of one laya profile.
fn laya_context(profile: &openkind_backends::families::laya::LayaProfile) -> Value {
    json!({
        "max_sequence_tokens": profile.max_sequence_tokens,
        "head_max_len_tokens": profile.head_max_len,
    })
}

fn model_revision(args: &ScoreArgs) -> Value {
    manifest_field(args, &["base", "revision"])
}

fn bundle_version(args: &ScoreArgs) -> Value {
    manifest_field(args, &["version"])
}

fn manifest_field(args: &ScoreArgs, path: &[&str]) -> Value {
    let Some(bundle_root) = &args.bundle_root else {
        return Value::Null;
    };
    let Ok(raw) = fs::read(bundle_root.join("BUNDLE_MANIFEST.json")) else {
        return Value::Null;
    };
    let mut current = serde_json::from_slice::<Value>(&raw).unwrap_or(Value::Null);
    for key in path {
        current = current.get(key).cloned().unwrap_or(Value::Null);
    }
    current
}

fn default_host() -> String {
    format!(
        "{}-{} (unattributed; pass --host to attribute)",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::{parse_workload, state_groups};

    fn profile_summary(engine: EngineKind) -> Value {
        let fixture = parse_workload(
            "smoke",
            include_bytes!("../../fixtures/decisions_smoke.jsonl"),
        )
        .expect("parse fixture");
        let groups = state_groups(&fixture.rows).expect("group fixture");
        let args = ScoreArgs {
            input: "smoke.jsonl".into(),
            engine,
            output_dir: "unused".into(),
            strategies: vec![],
            reps: 1,
            group: true,
            warmup: true,
            history_aba: false,
            host: Some("offline-test".into()),
            commit: None,
            pretty: false,
            bundle_root: None,
            checkpoint_root: None,
            tokenizer_path: None,
            model_root: None,
            adapter: None,
        };
        // Report construction must use the pinned family identity without
        // loading weights or reaching the native Qwen backend mapping.
        build_summary(&args, &fixture, &groups, &[], None)
    }

    #[test]
    fn expansion_profiles_report_their_own_provenance_without_model_assets() {
        for (engine, family, variant, profile_id, revision) in [
            (
                EngineKind::Plumb4b,
                "decoder-logit-qwen35",
                "plumb-4b",
                "c1f080794d38e94a0bc2",
                "24f7bf77e7ee258a2d158c61ea2dce2b60321010",
            ),
            (
                EngineKind::Decider4b,
                "decider",
                "decider-4b",
                "0529bf6f2bed84641701",
                "eb5fbdfc9448473ec25e399882912863afbdb70e",
            ),
            (
                EngineKind::WinnowE4b,
                "gemma4-decision",
                "winnow-e4b",
                "656ac636ce450cf79c7d",
                "1b257e8fa80b270a62338362a8b35e37f7890273",
            ),
        ] {
            let summary = profile_summary(engine);
            assert_eq!(summary["engine"], family);
            assert_eq!(summary["engine_variant"], variant);
            assert_eq!(summary["profile_id"], profile_id);
            assert_eq!(summary["model_revision"], revision);
            assert_eq!(summary["measurement_scope"], "model_request_path");
            assert!(summary["bundle_version"].is_null());
        }
    }

    #[test]
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    fn plumb_mlx_reports_the_cpu_profile_with_a_distinct_engine_variant() {
        let cpu = profile_summary(EngineKind::Plumb4b);
        let mlx = profile_summary(EngineKind::Plumb4bMlxFp32);
        for key in [
            "engine",
            "profile_id",
            "model_revision",
            "measurement_scope",
        ] {
            assert_eq!(mlx[key], cpu[key], "{key}");
        }
        assert_eq!(mlx["engine_variant"], "plumb-4b-mlx-fp32");
    }
}
