//! Operator smoke run for the pinned JEV-27B-VL 8-bit MLX profile (never part
//! of builds or tests). Loads a local checkpoint directory, evaluates the
//! requests in `tests/fixtures/jev_gev/smoke_requests.json` (or `--requests`),
//! and prints one JSON document with the wire answers and wall time.
//!
//! ```text
//! cargo run -p openkind-backends --release --features mlx --bin jev_mlx_smoke -- \
//!   --model-root ~/.cache/openkind/jev-27b-vl-mlx-8bit > rust.json
//! python3 scripts/jev-mlx-compare.py --model-root ~/.cache/openkind/jev-27b-vl-mlx-8bit \
//!   --rust rust.json
//! ```
//!
//! The script runs the mlx-vlm `feat/jev` reference on the same requests and
//! reports the largest probability difference per question.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use std::time::Instant;

    use openkind_backends::families::jev_protocol::mlx_engine;
    use openkind_backends::families::support::FamilyLimits;
    use openkind_core::SystemRequest;
    use openkind_engine::DecisionEngine;

    let mut model_root = None;
    let mut requests_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-root" => {
                model_root = Some(PathBuf::from(args.next().ok_or("--model-root value")?))
            }
            "--requests" => {
                requests_path = Some(PathBuf::from(args.next().ok_or("--requests value")?))
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let model_root = model_root.ok_or("--model-root is required")?;
    let requests: Vec<SystemRequest> = match requests_path {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => serde_json::from_str(include_str!(
            "../../tests/fixtures/jev_gev/smoke_requests.json"
        ))?,
    };

    let started = Instant::now();
    let engine = mlx_engine::load(
        &model_root,
        FamilyLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 0,
            retry_after_ms: 250,
            evaluation_timeout: None,
        },
    )?;
    let load_seconds = started.elapsed().as_secs_f64();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut cases = Vec::new();
    for request in requests {
        let started = Instant::now();
        let response = runtime.block_on(engine.evaluate(request.clone()))?;
        cases.push(serde_json::json!({
            "request": request,
            "response": response,
            "seconds": started.elapsed().as_secs_f64(),
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "load_seconds": load_seconds,
            "cases": cases,
        }))?
    );
    Ok(())
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("jev_mlx_smoke requires the `mlx` feature on macOS arm64");
    std::process::exit(2);
}
