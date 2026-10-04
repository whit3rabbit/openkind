//! Paired labeled Choice comparison, separate from synthetic timing workloads.

mod calibrate;
pub(crate) mod evidence;
mod metrics;
mod report;

#[cfg(test)]
mod tests;

pub(crate) use calibrate::{run as run_calibration, CalibrateArgs};

use std::collections::BTreeMap;
use std::fs;
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

/// Every scoring method recorded by one paired comparison, in output order.
pub(crate) const COMPARISON_METHODS: [&str; 9] = [
    "independent_fitted",
    "catalogue_fitted",
    "joint_forward",
    "joint_reverse",
    "joint_text_rotate",
    "joint_code_rotate",
    "joint_average",
    "joint_pair_text",
    "joint_ensemble_four",
];

/// How each comparison method is produced from the shared prepared inputs.
pub(crate) const METHOD_DEFINITIONS: [(&str, &str, &[&str]); 9] = [
    ("independent_fitted", "single", &[]),
    ("catalogue_fitted", "single", &[]),
    ("joint_forward", "single", &[]),
    ("joint_reverse", "single", &[]),
    ("joint_text_rotate", "single", &[]),
    ("joint_code_rotate", "single", &[]),
    (
        "joint_average",
        "ensemble",
        &["joint_forward", "joint_reverse"],
    ),
    (
        "joint_pair_text",
        "ensemble",
        &["joint_forward", "joint_text_rotate"],
    ),
    (
        "joint_ensemble_four",
        "ensemble",
        &[
            "joint_forward",
            "joint_reverse",
            "joint_text_rotate",
            "joint_code_rotate",
        ],
    ),
];

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
    let [predictions_file, mut summary_file] =
        evidence::reserve_outputs(&args.output_dir, ["predictions.jsonl", "summary.json"])?;
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
    probe.catalogue(&prepared[0])?;
    probe.joint(&prepared[0], false)?;
    probe.joint(&prepared[0], true)?;
    probe.joint_text_rotate(&prepared[0])?;
    probe.joint_code_rotate(&prepared[0])?;
    let mut writer = BufWriter::new(predictions_file);
    let mut records = Vec::new();
    for (index, (row, input)) in rows.iter().zip(&prepared).enumerate() {
        // Rotate execution order across rows after the shared warmup so no
        // method systematically runs first or last.
        let mut computed: BTreeMap<&'static str, TimedScore> = BTreeMap::new();
        let names: [&'static str; 6] = [
            "independent_fitted",
            "catalogue_fitted",
            "joint_forward",
            "joint_reverse",
            "joint_text_rotate",
            "joint_code_rotate",
        ];
        for step in 0..6 {
            let slot = (step + index) % 6;
            let score = match slot {
                0 => timed(|| probe.independent(input))?,
                1 => timed(|| probe.catalogue(input))?,
                2 => timed(|| probe.joint(input, false))?,
                3 => timed(|| probe.joint(input, true))?,
                4 => timed(|| probe.joint_text_rotate(input))?,
                _ => timed(|| probe.joint_code_rotate(input))?,
            };
            computed.insert(names[slot], score);
        }
        let mut take = |name: &str| computed.remove(name).expect("computed method recorded");
        let independent = take("independent_fitted");
        let catalogue = take("catalogue_fitted");
        let forward = take("joint_forward");
        let reverse = take("joint_reverse");
        let text_rotate = take("joint_text_rotate");
        let code_rotate = take("joint_code_rotate");
        let ensemble = |members: [&TimedScore; 2]| -> Result<TimedScore> {
            Ok(TimedScore {
                score: average_orders(&members[0].score, &members[1].score)?,
                elapsed_ms: members[0].elapsed_ms + members[1].elapsed_ms,
            })
        };
        let joint_average = ensemble([&forward, &reverse])?;
        let joint_pair_text = ensemble([&forward, &text_rotate])?;
        let upper = ensemble([&forward, &reverse])?;
        let lower = ensemble([&text_rotate, &code_rotate])?;
        let joint_ensemble_four = TimedScore {
            score: average_orders(&upper.score, &lower.score)?,
            elapsed_ms: upper.elapsed_ms + lower.elapsed_ms,
        };
        let record = Record {
            id: row.row.id.clone(),
            gold: row.gold.clone(),
            task: row.task.clone(),
            source_group: row.source_group.clone(),
            methods: [
                ("independent_fitted".into(), independent),
                ("catalogue_fitted".into(), catalogue),
                ("joint_forward".into(), forward),
                ("joint_reverse".into(), reverse),
                ("joint_text_rotate".into(), text_rotate),
                ("joint_code_rotate".into(), code_rotate),
                ("joint_average".into(), joint_average),
                ("joint_pair_text".into(), joint_pair_text),
                ("joint_ensemble_four".into(), joint_ensemble_four),
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
    serde_json::to_writer_pretty(&mut summary_file, &summary)?;
    summary_file.sync_all()?;
    Ok(summary)
}
