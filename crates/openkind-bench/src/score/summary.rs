//! Summary construction, metadata extraction, and JSON file output.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use openkind_backends::qwen35::PROFILE_ID;
use openkind_runtime::peak_resident_bytes;
use serde_json::{json, Value};

use super::types::{native_backend, EngineKind, ScoreArgs};
use crate::workload::Workload;

pub(crate) fn build_summary(
    args: &ScoreArgs,
    fixture: &Workload,
    groups: &[Vec<usize>],
    strategy_reports: &[Value],
    parity_clean: Option<bool>,
) -> Value {
    let (engine_id, profile, model_revision, bundle_version) = if args.engine == EngineKind::Mock {
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
        "host": args.host.clone().unwrap_or_else(default_host),
        "commit": args.commit.clone(),
        "timing_scope": "request construction, validation, dispatch, and answer extraction; \
                         excludes model load (reported per strategy), result writes, and the \
                         untimed warmup pass; warm process",
        "peak_resident_bytes": peak_resident_bytes().ok(),
        "cross_strategy_answer_parity_clean": parity_clean,
        "strategies": strategy_reports,
        "notes": {
            "percentiles": "p50 is the median rep total and p95 is the ceil(0.95*n)-1 sample; \
                            small rep counts are reported alongside as samples_seconds",
            "request_latency_ms": "latency is per request; grouped rows share their \
                                   request's latency and are not independent timings",
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
