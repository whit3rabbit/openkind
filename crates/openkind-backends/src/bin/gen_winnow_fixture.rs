//! Offline fixture generator for the pinned `winnow` profile.
//!
//! One-time operator workflow (never part of builds or tests):
//!
//! ```text
//! cargo run -p openkind-backends --bin gen_winnow_fixture -- \
//!   --model-root ~/.cache/openkind/qwen25-05b-instruct \
//!   --output crates/openkind-backends/tests/fixtures/winnow_4dff8c5b03cfbf680db6/golden.json
//! ```
//!
//! Records the routing probabilities of the pinned states; the adapter is
//! the vendored fixture artifact and is verified by digest at load.

use std::path::PathBuf;

use openkind_backends::families::support::FamilyLimits;
use openkind_backends::families::winnow::{WinnowEngine, WinnowEngineConfig};
use serde_json::json;

fn states() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("english-incident", "english", "Fictional incident record. Affected asset: application server. The asset operating system is Linux."),
        ("english-report", "english", "The quarterly report lists revenue by region and product line. No anomalies were recorded."),
        ("english-question", "english", "A user reported that the search bar stopped returning results after the latest deploy."),
        ("russian-record", "multilingual", "Сводка инцидента: затронутый актив — сервер приложений. Операционная система Linux."),
        ("chinese-record", "multilingual", "事故记录：受影响资产为应用服务器。操作系统为Linux。文档记录的影响级别为2。"),
        ("russian-journal", "multilingual", "Журнал подключений: использовался SSH из внутренней подсети. Пароль не был скомпрометирован."),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_root = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/openkind/qwen25-05b-instruct");
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut adapter = crate_root.join("tests/fixtures/winnow_adapter/adapters.safetensors");
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => model_root = PathBuf::from(args.next().expect("--model-root value")),
            "--adapter" => adapter = PathBuf::from(args.next().expect("--adapter value")),
            "--output" => output = Some(PathBuf::from(args.next().expect("--output value"))),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let engine = WinnowEngine::load(
        WinnowEngineConfig {
            model_root: model_root.clone(),
            adapter_path: adapter,
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                retry_after_ms: 100,
                evaluation_timeout: None,
            },
        },
        vec![
            (
                "A".to_string(),
                std::sync::Arc::new(openkind_engine::MockEngine::new())
                    as std::sync::Arc<dyn openkind_engine::DecisionEngine>,
            ),
            (
                "B".to_string(),
                std::sync::Arc::new(openkind_engine::MockEngine::new())
                    as std::sync::Arc<dyn openkind_engine::DecisionEngine>,
            ),
        ],
    )?;

    let mut cases = Vec::new();
    for (name, expected, state) in states() {
        let probabilities = engine.debug_route_probabilities(state)?;
        let routed = if probabilities[0] >= probabilities[1] {
            "A"
        } else {
            "B"
        };
        eprintln!(
            "[fixture] {:<18} expected {expected:<12} routed {routed} p = {probabilities:.6?}",
            name
        );
        assert_eq!(
            routed,
            match expected {
                "english" => "A",
                _ => "B",
            },
            "case {name} routed to the wrong label"
        );
        cases.push(json!({
            "name": name,
            "expected_label": expected,
            "state": state,
            "probabilities": probabilities,
        }));
    }

    let output = output.ok_or_else(|| "--output is required".to_owned())?;
    let golden = json!({
        "profile_id": openkind_backends::families::winnow::PROFILE_ID,
        "backbone": {
            "id": openkind_backends::families::winnow::BACKBONE_ID,
            "revision": openkind_backends::families::winnow::BACKBONE_REVISION,
        },
        "execution_arithmetic": openkind_backends::families::winnow::EXECUTION_ARITHMETIC_ID,
        "calibration_temperature": openkind_backends::families::winnow::CALIBRATION_TEMPERATURE,
        "probability_tolerance": 0.005,
        "provenance": {
            "generator": "gen-winnow-fixture",
            "adapter": "tests/fixtures/winnow_adapter/adapters.safetensors",
        },
        "cases": cases,
    });
    std::fs::create_dir_all(output.parent().expect("output parent"))?;
    std::fs::write(&output, serde_json::to_string_pretty(&golden)? + "\n")?;
    eprintln!("[fixture] wrote {}", output.display());
    Ok(())
}
