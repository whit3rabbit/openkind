//! Targeted BF16 cache/full trace for the frozen q4897 Phase 3B fixture.
//!
//! Usage: `qwen35_mlx_bf16_trace <checkpoint-root> <phase3b-reference-root> [layer-index]`
//! with `--features mlx` on macOS arm64. Reports per-layer cached-versus-full
//! hidden deltas and MLX output dtypes for the question and candidate suffixes.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::error::Error;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::fs;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use std::path::PathBuf;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::qwen35::mlx::{
    MlxPrecision, MlxQwen35Backbone, MlxRuntime, MlxRuntimeConfig,
};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = run() {
        eprintln!("qwen35_mlx_bf16_trace: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("qwen35_mlx_bf16_trace requires --features mlx on macOS arm64");
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args.next().ok_or("missing checkpoint root")?;
    let reference = PathBuf::from(args.next().ok_or("missing Phase 3B reference root")?);
    let operation_trace_layer = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    if args.next().is_some() {
        return Err(
            "usage: qwen35_mlx_bf16_trace <checkpoint-root> <phase3b-reference-root> [layer-index]"
                .into(),
        );
    }

    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.join("TOKEN_FIXTURES.json"))?)?;
    let record = fixture["records"]
        .as_array()
        .ok_or("fixture records are missing")?
        .iter()
        .find(|record| record["fixture_case"] == 1 && record["question_id"] == "q4897_a26093")
        .ok_or("q4897 fixture record is missing")?;
    let root_ids = token_ids(&record["root_ids"])?;
    let question_ids = token_ids(&record["question_ids"])?;
    let runtime = std::sync::Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
    runtime.qualify_bf16()?;
    let backbone = MlxQwen35Backbone::load(&checkpoint, runtime, MlxPrecision::NativeBf16)?;
    println!("backend: {}", backbone.arithmetic_id());

    let (_, root_state) = backbone.prefill(&root_ids)?;
    let (question_cached, _) =
        backbone.continue_from_trace_layer(&root_state, &question_ids, operation_trace_layer)?;
    let mut full_question_ids = root_ids.clone();
    full_question_ids.extend_from_slice(&question_ids);
    let (question_full, _) = backbone.prefill_trace_layer_tail(
        &full_question_ids,
        question_ids.len(),
        operation_trace_layer,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&compare_traces(
            "question",
            None,
            &question_cached,
            &question_full,
        ))?
    );

    Ok(())
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn token_ids(value: &serde_json::Value) -> Result<Vec<u32>, Box<dyn Error>> {
    value
        .as_array()
        .ok_or("token IDs are not an array")?
        .iter()
        .map(|token| {
            token
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .ok_or_else(|| "token ID is not a u32".into())
        })
        .collect()
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn compare_traces(
    stage: &str,
    candidate: Option<usize>,
    cached: &openkind_backends::qwen35::mlx::MlxBackboneTrace,
    full: &openkind_backends::qwen35::mlx::MlxBackboneTrace,
) -> serde_json::Value {
    let layers = cached
        .layer_last_tokens
        .iter()
        .zip(&full.layer_last_tokens)
        .enumerate()
        .map(|(layer, (cached_layer, full_layer))| {
            let (maximum_absolute, squared_sum) = cached_layer
                .iter()
                .zip(full_layer)
                .map(|(cached, full)| {
                    let delta = f64::from(*cached) - f64::from(*full);
                    (delta.abs(), delta * delta)
                })
                .fold((0.0_f64, 0.0_f64), |(max, sum), (delta, square)| {
                    (max.max(delta), sum + square)
                });
            serde_json::json!({
                "layer": layer,
                "cached_dtype": cached.layer_output_dtypes[layer],
                "full_dtype": full.layer_output_dtypes[layer],
                "max_abs_cached_vs_full": maximum_absolute,
                "rms_cached_vs_full": (squared_sum / cached_layer.len() as f64).sqrt(),
            })
        })
        .collect::<Vec<_>>();
    let operations = cached
        .operation_trace
        .iter()
        .zip(&full.operation_trace)
        .map(|(cached, full)| {
            let (maximum_absolute, squared_sum) = cached
                .values
                .iter()
                .zip(&full.values)
                .map(|(cached, full)| {
                    let delta = f64::from(*cached) - f64::from(*full);
                    (delta.abs(), delta * delta)
                })
                .fold((0.0_f64, 0.0_f64), |(max, sum), (delta, square)| {
                    (max.max(delta), sum + square)
                });
            serde_json::json!({
                "operation": cached.operation,
                "cached_dtype": cached.dtype,
                "full_dtype": full.dtype,
                "cached_shape": cached.shape,
                "full_shape": full.shape,
                "max_abs_cached_vs_full": maximum_absolute,
                "rms_cached_vs_full": (squared_sum / cached.values.len() as f64).sqrt(),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "stage": stage,
        "candidate_index": candidate,
        "operation_trace_layer": cached.operation_trace_layer,
        "cached_feature_max_abs_vs_full": cached
            .output
            .feature()
            .iter()
            .zip(full.output.feature())
            .map(|(cached, full)| (f64::from(*cached) - f64::from(*full)).abs())
            .fold(0.0_f64, f64::max),
        "layers": layers,
        "operations": operations,
    })
}
