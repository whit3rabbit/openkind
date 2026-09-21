//! Staged trace comparison against frozen Phase 3B reference vectors.

use std::error::Error;

use opendecision_backends::qwen35::mlx::MlxQwen35Backbone;
use opendecision_backends::qwen35::BackboneReference;

use super::fixtures::{maximum_abs, rms_delta};

pub(crate) struct TraceResult {
    pub(crate) summary: Vec<serde_json::Value>,
    pub(crate) embedding_exact: bool,
    pub(crate) layer0_max_abs: f64,
}

pub(crate) fn trace_stage(
    backbone: &MlxQwen35Backbone,
    reference: &BackboneReference,
    embedding_only: bool,
) -> Result<TraceResult, Box<dyn Error>> {
    let input_ids = reference.trace_input_ids();
    let embedding = if embedding_only {
        Some(
            backbone
                .embedding_last_token(input_ids)
                .map_err(|error| format!("MLX embedding read failed: {error}"))?,
        )
    } else {
        None
    };
    let output = if embedding_only {
        None
    } else {
        Some(
            backbone
                .prefill(input_ids)
                .map_err(|error| format!("MLX prefill failed: {error}"))?
                .0,
        )
    };

    let mut embedding_exact = false;
    let mut layer0_max_abs = f64::NAN;
    let mut summary = Vec::new();
    for (stage_index, stage) in reference.trace_stages().iter().enumerate() {
        if embedding_only && stage.stage() != "embedding" {
            continue;
        }
        let Some(golden) = reference.vector(stage.tensor_key()) else {
            continue;
        };
        let actual: Vec<f32> = match stage.stage() {
            "embedding" => embedding
                .clone()
                .or_else(|| {
                    output
                        .as_ref()
                        .map(|value| value.embedding_last_token.clone())
                })
                .ok_or("embedding output missing")?,
            stage if stage.starts_with("layer_") => {
                let layer = stage
                    .strip_prefix("layer_")
                    .and_then(|index| index.parse::<usize>().ok())
                    .ok_or("bad layer stage name")?;
                output
                    .as_ref()
                    .ok_or("layer output unavailable for embedding-only stage")?
                    .layer_last_tokens
                    .get(layer)
                    .ok_or("layer output missing")?
                    .clone()
            }
            "final_norm" => output
                .as_ref()
                .ok_or("final norm output unavailable for embedding-only stage")?
                .feature()
                .to_vec(),
            other => return Err(format!("unexpected trace stage {other}").into()),
        };
        let max_abs = maximum_abs(&actual, golden)?;
        let rms = rms_delta(&actual, golden);
        summary.push(serde_json::json!({
            "index": stage_index,
            "stage": stage.stage(),
            "max_abs": max_abs,
            "rms": rms,
        }));
        match stage.stage() {
            "embedding" => embedding_exact = max_abs == 0.0,
            "layer_00" => layer0_max_abs = max_abs,
            _ => {}
        }
    }
    Ok(TraceResult {
        summary,
        embedding_exact,
        layer0_max_abs,
    })
}
