//! Paired labeled Choice comparison, separate from synthetic timing workloads.

mod metrics;
mod report;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use openkind_backends::qwen35::experimental::{average_orders, Qwen35ScoringProbe, ScoringResult};
use openkind_backends::qwen35::{Qwen35Backend, SEMANTIC_NONE_OPTION};
use openkind_core::Question;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::workload::{parse_workload, WorkloadRow};

pub struct CompareArgs {
    pub input: PathBuf,
    pub bundle_root: PathBuf,
    pub checkpoint_root: PathBuf,
    pub tokenizer: PathBuf,
    pub backend: Qwen35Backend,
    pub output_dir: PathBuf,
    pub host: String,
    pub commit: String,
}

#[derive(Debug, Deserialize)]
struct LabeledRow {
    #[serde(flatten)]
    row: WorkloadRow,
    gold: String,
    task: String,
    source_group: String,
    #[serde(default)]
    split: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TimedScore {
    #[serde(flatten)]
    score: ScoringResult,
    elapsed_ms: f64,
}

#[derive(Debug, Serialize)]
struct Record {
    id: String,
    gold: String,
    task: String,
    source_group: String,
    methods: BTreeMap<String, TimedScore>,
}

fn load(path: &Path) -> Result<(Vec<LabeledRow>, String)> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    // Use the normal harness's wire validation and duplicate-ID checks as well.
    let workload = parse_workload(&path.display().to_string(), &raw)?;
    let mut rows = Vec::new();
    for line in std::str::from_utf8(&raw)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let row: LabeledRow = serde_json::from_str(line).context("labeled Choice row")?;
        anyhow::ensure!(
            row.split
                .as_deref()
                .is_none_or(|split| !split.to_ascii_lowercase().contains("final")),
            "row {}: final partitions are not accepted by this diagnostic command",
            row.row.id
        );
        let Question::Choice(question) = row.row.question()? else {
            anyhow::bail!("row {}: comparison accepts Choice only", row.row.id);
        };
        anyhow::ensure!(
            !row.task.trim().is_empty() && !row.source_group.trim().is_empty(),
            "row {}: task and source_group must be nonempty",
            row.row.id
        );
        anyhow::ensure!(
            question.criteria.contains_key(&row.gold),
            "row {}: gold must be an offered option",
            row.row.id
        );
        anyhow::ensure!(
            question.criteria.len() >= 3,
            "row {}: baseline requires at least two real candidates",
            row.row.id
        );
        // Require explicit none text rather than silently choosing the fallback
        // for the new readout. Its meaning is part of this experiment's input.
        let crate::workload::QuestionSpec::Choice { options, .. } = &row.row.question else {
            unreachable!()
        };
        anyhow::ensure!(
            options
                .iter()
                .any(|option| option.id == SEMANTIC_NONE_OPTION
                    && option
                        .description
                        .as_deref()
                        .is_some_and(|text| !text.trim().is_empty())),
            "row {}: explicit __none__ description required",
            row.row.id
        );
        rows.push(row);
    }
    Ok((rows, workload.sha256))
}

fn timed(
    score: impl FnOnce() -> Result<ScoringResult, openkind_backends::qwen35::Qwen35Error>,
) -> Result<TimedScore> {
    let start = Instant::now();
    let score = score()?;
    Ok(TimedScore {
        score,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

pub fn run(args: &CompareArgs) -> Result<serde_json::Value> {
    anyhow::ensure!(
        !args.host.trim().is_empty() && !args.commit.trim().is_empty(),
        "host and commit attribution required"
    );
    let (rows, digest) = load(&args.input)?;
    let start = Instant::now();
    let probe = Qwen35ScoringProbe::load(
        &args.bundle_root,
        &args.checkpoint_root,
        &args.tokenizer,
        args.backend,
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    // Reject any incompatible prompt before starting the scored panel.
    let prepared = rows
        .iter()
        .map(|row| Ok(probe.prepare(&row.row.state, &row.row.question()?)?))
        .collect::<Result<Vec<_>>>()?;
    probe.independent(&prepared[0])?;
    probe.joint(&prepared[0], false)?;
    probe.joint(&prepared[0], true)?;
    fs::create_dir_all(&args.output_dir)?;
    let mut writer = BufWriter::new(File::create(args.output_dir.join("predictions.jsonl"))?);
    let mut records = Vec::new();
    for (index, (row, input)) in rows.iter().zip(&prepared).enumerate() {
        let (independent, forward, reverse) = if index % 2 == 0 {
            (
                timed(|| probe.independent(input))?,
                timed(|| probe.joint(input, false))?,
                timed(|| probe.joint(input, true))?,
            )
        } else {
            let reverse = timed(|| probe.joint(input, true))?;
            let forward = timed(|| probe.joint(input, false))?;
            (timed(|| probe.independent(input))?, forward, reverse)
        };
        let average = TimedScore {
            score: average_orders(&forward.score, &reverse.score)?,
            elapsed_ms: forward.elapsed_ms + reverse.elapsed_ms,
        };
        let record = Record {
            id: row.row.id.clone(),
            gold: row.gold.clone(),
            task: row.task.clone(),
            source_group: row.source_group.clone(),
            methods: [
                ("independent_fitted".into(), independent),
                ("joint_forward".into(), forward),
                ("joint_reverse".into(), reverse),
                ("joint_average".into(), average),
            ]
            .into_iter()
            .collect(),
        };
        for score in record.methods.values() {
            metrics::validate(&score.score.probabilities, &record.gold)?;
        }
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
        writer.flush()?;
        records.push(record);
        if index % 8 == 0 || index + 1 == rows.len() {
            eprintln!("paired Choice scoring: {}/{}", index + 1, rows.len());
        }
    }
    let mut summary = report::summary(args, &probe, digest, load_seconds, &records);
    summary["executable_sha256"] = serde_json::json!(format!(
        "{:x}",
        Sha256::digest(fs::read(std::env::current_exe()?)?)
    ));
    fs::write(
        args.output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}
