//! Phase 3.9b representative model-backed high-K stress.
//!
//! This streams candidate continuations and discards each candidate state so
//! it measures model execution without allocating a naive K-wide state fan-out.
//! It is intentionally checkpoint-gated and never downloads artifacts.

use std::env;
use std::error::Error;
use std::path::PathBuf;

use opendecision_backends::qwen35::Qwen35Backbone;
use opendecision_runtime::branch::BranchableState;
use opendecision_runtime::peak_resident_bytes;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let checkpoint_root = PathBuf::from(arguments.next().ok_or("missing checkpoint root")?);
    let candidates: usize = arguments
        .next()
        .ok_or("missing candidate count (32, 64, 128, or 255)")?
        .to_string_lossy()
        .parse()?;
    if ![32, 64, 128, 255].contains(&candidates) {
        return Err("candidate count must be 32, 64, 128, or 255".into());
    }
    if arguments.next().is_some() {
        return Err("expected checkpoint root and candidate count only".into());
    }

    let backbone = Qwen35Backbone::load(checkpoint_root)?;
    let root_ids: Vec<u32> = (0..98)
        .map(|index| ((index * 7_919 + 13) % 248_320) as u32)
        .collect();
    let question_ids = [41_u32, 42, 43, 44];
    let (_, root_state) = backbone.prefill(&root_ids)?;
    let root_fingerprint = root_state.strict_fingerprint();
    let (_, question_state) = backbone.continue_from(&root_state, &question_ids)?;
    let peak_before_candidates = peak_resident_bytes()?;
    let mut checksum = 0.0_f64;
    for candidate in 0..candidates {
        let suffix = [u32::try_from(1000 + candidate)?];
        let (output, candidate_state) = backbone.continue_from(&question_state, &suffix)?;
        checksum += output
            .final_token()
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        drop(candidate_state);
    }
    let root_immutable = root_state.strict_fingerprint() == root_fingerprint;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "opendecision-qwen35-model-stress/v1",
            "scope": "representative streaming CPU continuations; not vectorized K-wide throughput",
            "candidates": candidates,
            "root_tensor_bytes": root_state.tensor_storage_bytes(),
            "question_tensor_bytes": question_state.tensor_storage_bytes(),
            "peak_resident_before_candidates_bytes": peak_before_candidates,
            "peak_resident_after_candidates_bytes": peak_resident_bytes()?,
            "root_immutable": root_immutable,
            "feature_checksum": checksum,
        }))?
    );
    if !root_immutable || !checksum.is_finite() {
        return Err("model-backed stress invariant failed".into());
    }
    Ok(())
}
