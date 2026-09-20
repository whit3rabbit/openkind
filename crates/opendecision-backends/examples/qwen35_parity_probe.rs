use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use opendecision_backends::qwen35::{
    BackboneReference, Qwen35Backbone, Qwen35Embedding, Qwen35Layer0,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ContinuationTrace {
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<u32>,
    root_cache_bytes: usize,
    root_position: usize,
    question_position: usize,
    candidate_position: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: qwen35_parity_probe <checkpoint-root> <phase3b-reference-root>")?;
    let reference_root = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: qwen35_parity_probe <checkpoint-root> <phase3b-reference-root>")?;
    let stop_after = arguments
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "embedding".to_owned());
    if arguments.next().is_some() {
        return Err("qwen35_parity_probe accepts two paths and one optional stage".into());
    }

    let reference = BackboneReference::load(&reference_root)?;
    if stop_after == "continuation" {
        let trace: ContinuationTrace =
            serde_json::from_slice(&fs::read(reference_root.join("CONTINUATION_TRACE.json"))?)?;
        let backbone = Qwen35Backbone::load(checkpoint_root)?;
        let (root_output, root_state) = backbone.prefill(&trace.root_ids)?;
        let root_snapshot = root_state.clone();
        let root_comparison =
            reference.compare("continuation.root_last_hidden", root_output.final_token())?;
        let (question_output, question_state) =
            backbone.continue_from(&root_state, &trace.question_ids)?;
        let root_immutable = root_state == root_snapshot;
        let question_comparison = reference.compare(
            "continuation.question_last_hidden",
            question_output.final_token(),
        )?;
        let (candidate_output, candidate_state) =
            backbone.continue_from(&question_state, &trace.candidate_suffix_ids)?;
        let candidate_comparison = reference.compare(
            "continuation.candidate_last_hidden",
            candidate_output.final_token(),
        )?;
        let mut full_ids = trace.root_ids.clone();
        full_ids.extend_from_slice(&trace.question_ids);
        full_ids.extend_from_slice(&trace.candidate_suffix_ids);
        let full_output = backbone.forward(&full_ids)?;
        let cached_vs_full = maximum_abs(candidate_output.final_token(), full_output.final_token());
        let passed = root_state.position() == trace.root_position
            && question_state.position() == trace.question_position
            && candidate_state.position() == trace.candidate_position
            && root_state.byte_len() == trace.root_cache_bytes
            && root_state.layer_count() == 32
            && root_immutable;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "passed_structural_gates": passed,
                "root": {
                    "position": root_state.position(),
                    "cache_bytes": root_state.byte_len(),
                    "max_abs": root_comparison.max_abs(),
                    "rms": root_comparison.rms(),
                    "cosine": root_comparison.cosine(),
                },
                "question": {
                    "position": question_state.position(),
                    "max_abs": question_comparison.max_abs(),
                    "rms": question_comparison.rms(),
                    "cosine": question_comparison.cosine(),
                },
                "candidate": {
                    "position": candidate_state.position(),
                    "max_abs": candidate_comparison.max_abs(),
                    "rms": candidate_comparison.rms(),
                    "cosine": candidate_comparison.cosine(),
                    "cached_vs_native_full_max_abs": cached_vs_full,
                },
                "layer_states": root_state.layer_count(),
                "root_immutable": root_immutable,
            }))?
        );
        if !passed {
            return Err("native continuation structural gate failed".into());
        }
        return Ok(());
    }
    if stop_after == "trace" {
        let backbone = Qwen35Backbone::load(checkpoint_root)?;
        let output = backbone.forward(reference.trace_input_ids())?;
        let mut stages = Vec::with_capacity(34);
        let embedding = reference.compare("diagnostic.embedding", output.embedding_last_token())?;
        stages.push(serde_json::json!({
            "stage": "embedding",
            "max_abs": embedding.max_abs(),
            "rms": embedding.rms(),
            "cosine": embedding.cosine(),
        }));
        for layer_index in 0..32 {
            let stage = format!("layer_{layer_index:02}");
            let comparison = reference.compare(
                &format!("diagnostic.{stage}"),
                output
                    .layer_last_token(layer_index)
                    .ok_or("missing native layer output")?,
            )?;
            stages.push(serde_json::json!({
                "stage": stage,
                "max_abs": comparison.max_abs(),
                "rms": comparison.rms(),
                "cosine": comparison.cosine(),
            }));
        }
        let final_norm = reference.compare("diagnostic.final_norm", output.final_token())?;
        stages.push(serde_json::json!({
            "stage": "final_norm",
            "max_abs": final_norm.max_abs(),
            "rms": final_norm.rms(),
            "cosine": final_norm.cosine(),
        }));
        println!("{}", serde_json::to_string_pretty(&stages)?);
        return Ok(());
    }
    let (stage, token_count, hidden_size, comparison) = match stop_after.as_str() {
        "embedding" => {
            let embedding = Qwen35Embedding::load(checkpoint_root)?;
            let output = embedding.embed(reference.trace_input_ids())?;
            (
                "embedding",
                output.token_count(),
                output.hidden_size(),
                reference.compare("diagnostic.embedding", output.last_token())?,
            )
        }
        "layer_00" => {
            let layer = Qwen35Layer0::load(checkpoint_root)?;
            let output = layer.forward(reference.trace_input_ids())?;
            (
                "layer_00",
                output.token_count(),
                output.last_token().len(),
                reference.compare("diagnostic.layer_00", output.last_token())?,
            )
        }
        "final_norm" => {
            let backbone = Qwen35Backbone::load(checkpoint_root)?;
            let output = backbone.forward(reference.trace_input_ids())?;
            (
                "final_norm",
                output.token_count(),
                output.final_token().len(),
                reference.compare("diagnostic.final_norm", output.final_token())?,
            )
        }
        other if other.starts_with("layer_") => {
            let layer_index = other
                .strip_prefix("layer_")
                .expect("prefix checked")
                .parse::<usize>()?;
            let backbone = Qwen35Backbone::load(checkpoint_root)?;
            let output = backbone.forward(reference.trace_input_ids())?;
            let values = output
                .layer_last_token(layer_index)
                .ok_or_else(|| format!("layer index {layer_index} is out of range"))?;
            (
                other,
                output.token_count(),
                values.len(),
                reference.compare(&format!("diagnostic.{other}"), values)?,
            )
        }
        other => return Err(format!("unsupported stop stage `{other}`").into()),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "stage": stage,
            "token_count": token_count,
            "hidden_size": hidden_size,
            "max_abs": comparison.max_abs(),
            "rms": comparison.rms(),
            "cosine": comparison.cosine(),
            "exact": comparison.max_abs() == 0.0,
        }))?
    );
    if stage == "embedding" && comparison.max_abs() != 0.0 {
        return Err("native embedding does not exactly match diagnostic.embedding".into());
    }
    Ok(())
}

fn maximum_abs(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from((left - right).abs()))
        .fold(0.0, f64::max)
}
