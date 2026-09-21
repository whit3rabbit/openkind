//! Fixture schemas and comparison utilities for MLX nested sequential parity.

use std::error::Error;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use opendecision_backends::qwen35::PrimitiveKind;

#[derive(Debug, Deserialize)]
pub(crate) struct TokenFixtures {
    pub(crate) records: Vec<TokenRecord>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TokenRecord {
    pub(crate) fixture_case: usize,
    pub(crate) question_id: String,
    pub(crate) root_ids: Vec<u32>,
    pub(crate) question_ids: Vec<u32>,
    pub(crate) candidate_suffix_ids: Vec<Vec<u32>>,
    pub(crate) full_candidate_ids: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GoldenCase {
    pub(crate) request: GoldenRequest,
    pub(crate) head_and_token_fixtures: Vec<HeadFixture>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GoldenRequest {
    pub(crate) questions: Vec<GoldenQuestion>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GoldenQuestion {
    pub(crate) id: String,
    pub(crate) primitive: String,
    pub(crate) options: Vec<GoldenOption>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GoldenOption {
    pub(crate) id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct HeadFixture {
    pub(crate) question_id: String,
    #[allow(dead_code)]
    pub(crate) logits: Vec<f64>,
    pub(crate) probabilities: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProbabilityReference {
    pub(crate) records: Vec<ProbabilityRecord>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProbabilityRecord {
    pub(crate) fixture_case: usize,
    pub(crate) answers: Vec<ExpectedAnswer>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ExpectedAnswer {
    pub(crate) question_id: String,
    pub(crate) selected_id: String,
    pub(crate) top_probability: f64,
}

pub(crate) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub(crate) fn primitive(value: &str) -> Result<PrimitiveKind, Box<dyn Error>> {
    match value {
        "choice" => Ok(PrimitiveKind::Choice),
        "noul" => Ok(PrimitiveKind::Noul),
        "score" => Ok(PrimitiveKind::Score),
        other => Err(format!("unexpected primitive {other}").into()),
    }
}

pub(crate) fn maximum_delta(actual: &[f64], expected: &[f64]) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "vector length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max))
}

pub(crate) fn git_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub(crate) fn require_clean_worktree() -> Result<(), Box<dyn Error>> {
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()?;
    if !output.status.success() {
        return Err("formal evidence could not inspect git worktree state".into());
    }
    if !output.stdout.is_empty() {
        return Err(
            "formal parity evidence requires a clean worktree; rerun after committing the implementation".into(),
        );
    }
    Ok(())
}
