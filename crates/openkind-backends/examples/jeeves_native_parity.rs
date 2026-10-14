//! Explicit local-checkpoint parity smoke. Never downloads any assets.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use openkind_backends::families::jeeves::{
        artifacts::VerifiedArtifacts, model::JeevesModel, renderer::JeevesRenderer, MODEL_REVISION,
        PROFILE_ID,
    };
    use std::{path::PathBuf, time::Instant};
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: jeeves_native_parity <local-model-root>")?,
    );
    eprintln!("Verifying pinned local artifacts");
    let artifacts = VerifiedArtifacts::verify(&root)?;
    let renderer = JeevesRenderer::load(artifacts.tokenizer_path())?;
    let start = Instant::now();
    let model = JeevesModel::load(&artifacts)?;
    let load_seconds = start.elapsed().as_secs_f64();
    eprintln!("Native model loaded in {load_seconds:.3}s");
    let fixtures: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../tests/fixtures/jeeves_15fb3b95801f2f039636/golden.json"
    ))?;
    let mut cases = Vec::new();
    let mut passed = true;
    for fixture in fixtures {
        let options: Vec<String> = serde_json::from_value(fixture["options"].clone())?;
        let row = renderer.encode(
            fixture["state_text"].as_str().ok_or("missing state")?,
            fixture["instruction_text"]
                .as_str()
                .ok_or("missing instructions")?,
            &options,
        )?;
        let expected_ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone())?;
        let expected_positions: Vec<usize> =
            serde_json::from_value(fixture["option_positions"].clone())?;
        if row.ids != expected_ids
            || row.option_positions != expected_positions
            || row.decide_position
                != fixture["decide_position"]
                    .as_u64()
                    .ok_or("missing decide")? as usize
        {
            return Err(format!("renderer mismatch: {}", fixture["id"]).into());
        }
        let start = Instant::now();
        eprintln!("Scoring {} ({} tokens)", fixture["id"], row.ids.len());
        let result = model.score(&row, None)?;
        let elapsed = start.elapsed().as_secs_f64();
        let reference: Vec<f64> = serde_json::from_value(fixture["probabilities"].clone())?;
        let delta = result
            .probabilities
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let argmax = |p: &[f64]| {
            p.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
        };
        let agrees = argmax(&result.probabilities) == argmax(&reference);
        passed &= delta <= 0.005 && agrees;
        cases.push(serde_json::json!({"id":fixture["id"],"token_readout_parity":true,"probabilities":result.probabilities,"reference_probabilities":reference,"max_probability_delta":delta,"argmax_agreement":agrees,"elapsed_seconds":elapsed,"raw_logits":result.logits,"selected_hidden":result.selected_hidden}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"profile":PROFILE_ID,"revision":MODEL_REVISION,"backend":"native-rust-mlx-affine4-fp32","reference":"MLX-LM 94cdcae13 / MLX 0.32.3 / Metal","load_seconds":load_seconds,"peak_active_mlx_bytes":model.peak_memory_bytes()?,"probability_tolerance":0.005,"passed":passed,"cases":cases})
        )?
    );
    if !passed {
        return Err("Jeeves native probability parity failed".into());
    }
    Ok(())
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("jeeves_native_parity requires --features mlx on macOS arm64");
    std::process::exit(1);
}
