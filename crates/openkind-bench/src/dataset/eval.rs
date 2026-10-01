//! Dataset evaluation: materialize one split, score it through the normal
//! harness request path, and join predictions back to gold for the accuracy
//! report.
//!
//! Reports are model-quality evidence on a pinned public dataset. They never
//! promote a model or policy, and every report carries dataset provenance
//! (revisions, digests, license, template identity), engine provenance, and
//! the workload/prediction digests that bind the numbers to exact bytes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use openkind_datasets::DatasetStore;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::materialize::{materialize, DatasetRow};
use super::metrics::{choice_metrics, noul_metrics, score_metrics, ChoiceRow, NoulRow, ScoreRow};
use crate::score::{
    is_family_engine_public as is_family_engine, run_score, EngineKind, ScoreArgs, StrategySpec,
};
use crate::workload::{QuestionSpec, WorkloadRow};

pub struct EvalArgs {
    pub name: String,
    pub split: String,
    pub limit: Option<usize>,
    pub datasets_dir: Option<PathBuf>,
    pub engine: EngineKind,
    pub output_dir: PathBuf,
    pub host: Option<String>,
    pub commit: Option<String>,
    pub pretty: bool,
    pub bundle_root: Option<PathBuf>,
    pub checkpoint_root: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub model_root: Option<PathBuf>,
    pub adapter: Option<PathBuf>,
    /// Tune a Noul decision threshold on the dev split and apply it here.
    pub tune_threshold: bool,
}

/// Run one dataset evaluation and write the report.
///
/// # Errors
/// Returns an error when the dataset is not installed, the engine artifacts
/// are missing, prediction coverage is incomplete, or any answer fails wire
/// validation.
pub fn run_eval(args: &EvalArgs) -> Result<Value> {
    let store = DatasetStore::new(
        args.datasets_dir
            .clone()
            .map_or_else(openkind_datasets::default_datasets_dir, Ok)?,
    )?;
    let installed = store.installed(&args.name)?;
    let entry = installed.entry.clone();
    let primitive = entry.primitives.first().cloned().unwrap_or_default();
    ensure!(
        ["choice", "noul", "score"].contains(&primitive.as_str()),
        "dataset {} declares no evaluable primitive",
        args.name
    );

    let tuned = if args.tune_threshold && primitive == "noul" && args.split == "eval" {
        Some(tune_threshold_on_dev(&store, args)?)
    } else {
        None
    };

    let rows = materialize(&installed, &args.split, args.limit)?;
    if rows.rows.is_empty() {
        bail!("dataset {} produced no {} rows", args.name, args.split);
    }
    fs::create_dir_all(&args.output_dir)?;
    let workload_path = args
        .output_dir
        .join(format!("workload-{}-{}.jsonl", args.name, args.split));
    write_workload(&workload_path, &rows.rows)?;

    let score_args = ScoreArgs {
        input: workload_path.clone(),
        engine: args.engine,
        output_dir: args.output_dir.clone(),
        strategies: strategy_selection(args.engine)?,
        reps: 1,
        group: true,
        warmup: true,
        history_aba: false,
        host: args.host.clone(),
        commit: args.commit.clone(),
        pretty: args.pretty,
        bundle_root: args.bundle_root.clone(),
        checkpoint_root: args.checkpoint_root.clone(),
        tokenizer_path: args.tokenizer_path.clone(),
        model_root: args.model_root.clone(),
        adapter: args.adapter.clone(),
    };
    let outcome = run_score(&score_args)?;
    let summary = outcome.summary;
    let engine_slug = summary["engine"]
        .as_str()
        .context("summary engine slug")?
        .to_owned();
    let strategy = summary["strategies"][0]["strategy"]
        .as_str()
        .context("summary strategy")?
        .to_owned();
    let predictions_path = args
        .output_dir
        .join(format!("predictions-{engine_slug}-{strategy}.jsonl"));
    let predictions_bytes = fs::read(&predictions_path)?;
    let predictions_sha256 = format!("{:x}", Sha256::digest(&predictions_bytes));
    // Digest binding: the summary pins the exact prediction bytes.
    ensure!(
        summary["prediction_sha256"][&strategy].as_str() == Some(predictions_sha256.as_str()),
        "predictions do not match the summary digest binding"
    );
    let predictions = parse_predictions(&predictions_bytes)?;

    let mut by_id: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    for prediction in &predictions {
        let id = prediction["id"]
            .as_str()
            .context("prediction id")?
            .to_owned();
        ensure!(
            by_id.insert(id.clone(), prediction.clone()).is_none(),
            "duplicate prediction {id}"
        );
    }
    ensure!(
        by_id.len() == rows.rows.len(),
        "prediction coverage mismatch: {} predictions for {} workload rows",
        by_id.len(),
        rows.rows.len()
    );

    let metrics = match primitive.as_str() {
        "choice" => {
            let mut scored = Vec::new();
            let mut groups = Vec::new();
            for row in &rows.rows {
                let answer = &by_id[&row.row.id]["answer"];
                let probabilities = parse_distribution(answer)?;
                scored.push(ChoiceRow {
                    probabilities,
                    gold: row.gold.clone(),
                });
                groups.push(row.source_group.clone());
            }
            choice_metrics(&scored, &groups)
        }
        "noul" => {
            let mut scored = Vec::new();
            for row in &rows.rows {
                let answer = &by_id[&row.row.id]["answer"];
                let probability = answer["noul"]
                    .as_f64()
                    .with_context(|| format!("row {}: noul answer", row.row.id))?;
                ensure!(
                    (0.0..=1.0).contains(&probability),
                    "row {}: noul probability out of range",
                    row.row.id
                );
                scored.push(NoulRow {
                    probability,
                    gold: row.gold == "true",
                    group: row.source_group.clone(),
                });
            }
            let mut metrics = noul_metrics(&scored);
            if let Some((threshold, tuning_accuracy, dev_rows)) = tuned {
                let hits = scored
                    .iter()
                    .filter(|row| (row.probability >= threshold) == row.gold)
                    .count();
                metrics
                    .as_object_mut()
                    .expect("noul metrics object")
                    .insert(
                        "tuned_threshold".to_owned(),
                        json!({
                            "threshold": threshold,
                            "tuning_accuracy_on_dev": tuning_accuracy,
                            "dev_rows": dev_rows,
                            "eval_accuracy": hits as f64 / scored.len() as f64,
                            "protocol": "threshold selected on dev, applied to eval",
                        }),
                    );
            }
            metrics
        }
        "score" => {
            let mut scored = Vec::new();
            for row in &rows.rows {
                let answer = &by_id[&row.row.id]["answer"];
                let levels = level_count(&row.row);
                let probabilities = answer["probabilities"]
                    .as_object()
                    .with_context(|| format!("row {}: score probabilities", row.row.id))?;
                ensure!(
                    probabilities.len() == levels,
                    "row {}: expected {levels} level probabilities",
                    row.row.id
                );
                let mut expected = 0.0;
                let mut total = 0.0;
                let mut level_probabilities = vec![0.0; levels];
                for (index, probability) in probabilities {
                    let index: usize = index.parse().context("level index")?;
                    let probability = probability.as_f64().context("level probability")?;
                    expected += index as f64 * probability;
                    total += probability;
                    level_probabilities[index] = probability;
                }
                ensure!(
                    (total - 1.0).abs() < 1e-6,
                    "row {}: probabilities do not sum to one",
                    row.row.id
                );
                let gold: f64 = row.gold.parse().context("gold level")?;
                let nearest = gold.round().clamp(0.0, levels as f64 - 1.0) as usize;
                scored.push(ScoreRow {
                    expected: expected / total,
                    gold,
                    levels,
                    gold_level_probability: level_probabilities[nearest],
                });
            }
            score_metrics(&scored)
        }
        other => bail!("no metrics for primitive {other}"),
    };

    let workload_sha256 = format!("{:x}", Sha256::digest(fs::read(&workload_path)?));
    let executable_sha256 = executable_digest()?;
    let report = json!({
        "schema": "openkind-dataset-eval/v1",
        "dataset": {
            "name": entry.name,
            "description": entry.description,
            "hf_repo": entry.hf_repo,
            "hf_revision": entry.hf_revision,
            "convert_revision": entry.convert_revision,
            "license": entry.license,
            "gated": entry.gated,
            "files": entry.files.len(),
            "task_family": entry.task_family,
        },
        "template": entry.template,
        "split": args.split,
        "limit": args.limit,
        "rows": rows.rows.len(),
        "skipped_duplicates": rows.skipped_duplicates,
        "skipped_template_rows": rows.skipped_rows,
        "engine": {
            "engine": summary["engine"],
            "profile_id": summary["profile_id"],
            "model_revision": summary["model_revision"],
            "bundle_version": summary["bundle_version"],
            "strategy": strategy,
        },
        "provenance": {
            "workload_path": workload_path,
            "workload_sha256": workload_sha256,
            "predictions_path": predictions_path,
            "predictions_sha256": predictions_sha256,
            "summary_sha256": format!("{:x}", Sha256::digest(serde_json::to_vec(&summary)?)),
            "executable_sha256": executable_sha256,
            "host": summary["host"],
            "commit": summary["commit"],
        },
        "metrics": metrics,
        "comparability": {
            "forced_none_option": primitive == "choice",
            "note": "Choice rows always carry an injected __none__ option; accuracy includes it, answerable_ranking_accuracy excludes it for external comparison. Request wording follows the pinned template source.",
        },
        "evidence_class": "model-quality (public-dataset); no model or policy promotion",
    });
    let report_path = args
        .output_dir
        .join(format!("dataset-eval-{}-{}.json", args.name, args.split));
    let bytes = if args.pretty {
        serde_json::to_vec_pretty(&report)?
    } else {
        serde_json::to_vec(&report)?
    };
    fs::write(&report_path, bytes)?;
    Ok(report)
}

fn strategy_selection(engine: EngineKind) -> Result<Vec<StrategySpec>> {
    // Native engines take the measured scheduler choice; mock and family
    // engines run their single pinned plan.
    if engine != EngineKind::Mock && !is_family_engine(engine) {
        Ok(vec![
            StrategySpec::parse("choose_strategy").map_err(|error| anyhow::anyhow!("{error}"))?
        ])
    } else {
        Ok(Vec::new())
    }
}

fn level_count(row: &WorkloadRow) -> usize {
    match &row.question {
        QuestionSpec::Score { levels, .. } => levels.len(),
        _ => 0,
    }
}

fn parse_distribution(answer: &Value) -> Result<BTreeMap<String, f64>> {
    let map = answer
        .get("probabilities")
        .and_then(Value::as_object)
        .context("choice probabilities")?;
    let mut out = BTreeMap::new();
    let mut total = 0.0;
    for (label, probability) in map {
        let probability = probability
            .as_f64()
            .with_context(|| format!("probability for {label}"))?;
        ensure!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "invalid probability for {label}"
        );
        total += probability;
        out.insert(label.clone(), probability);
    }
    ensure!(
        (total - 1.0).abs() < 1e-6,
        "choice probabilities do not sum to one"
    );
    Ok(out)
}

fn parse_predictions(bytes: &[u8]) -> Result<Vec<Map<String, Value>>> {
    let mut out = Vec::new();
    for line in std::str::from_utf8(bytes)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line).context("prediction row")?);
    }
    Ok(out)
}

fn write_workload(path: &Path, rows: &[DatasetRow]) -> Result<()> {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend_from_slice(serde_json::to_string(row)?.as_bytes());
        bytes.push(b'\n');
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn executable_digest() -> Result<String> {
    let executable = std::env::current_exe()?;
    let bytes = fs::read(&executable)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Tune a Noul threshold on the dev split with the same engine, following the
/// paper's protocol: parameters are locked on dev before any eval row is
/// scored.
fn tune_threshold_on_dev(store: &DatasetStore, args: &EvalArgs) -> Result<(f64, f64, usize)> {
    let installed = store.installed(&args.name)?;
    let dev = materialize(&installed, "dev", args.limit)?;
    if dev.rows.is_empty() {
        bail!("threshold tuning requires dev rows for {}", args.name);
    }
    let dev_dir = args.output_dir.join(format!("tuning-{}", args.name));
    fs::create_dir_all(&dev_dir)?;
    let workload_path = dev_dir.join(format!("workload-{}-dev.jsonl", args.name));
    write_workload(&workload_path, &dev.rows)?;
    let score_args = ScoreArgs {
        input: workload_path,
        engine: args.engine,
        output_dir: dev_dir.clone(),
        strategies: strategy_selection(args.engine)?,
        reps: 1,
        group: true,
        warmup: true,
        history_aba: false,
        host: args.host.clone(),
        commit: args.commit.clone(),
        pretty: false,
        bundle_root: args.bundle_root.clone(),
        checkpoint_root: args.checkpoint_root.clone(),
        tokenizer_path: args.tokenizer_path.clone(),
        model_root: args.model_root.clone(),
        adapter: args.adapter.clone(),
    };
    let outcome = run_score(&score_args)?;
    let summary = outcome.summary;
    let engine_slug = summary["engine"]
        .as_str()
        .context("summary engine")?
        .to_owned();
    let strategy = summary["strategies"][0]["strategy"]
        .as_str()
        .context("strategy")?
        .to_owned();
    let predictions_bytes =
        fs::read(dev_dir.join(format!("predictions-{engine_slug}-{strategy}.jsonl")))?;
    let predictions = parse_predictions(&predictions_bytes)?;
    ensure!(
        predictions.len() == dev.rows.len(),
        "dev prediction coverage mismatch"
    );
    let mut probabilities = Vec::new();
    let mut gold = Vec::new();
    for row in &dev.rows {
        let answer = predictions
            .iter()
            .find(|prediction| prediction["id"].as_str() == Some(row.row.id.as_str()))
            .with_context(|| format!("dev row {}", row.row.id))?;
        probabilities.push(answer["answer"]["noul"].as_f64().context("noul answer")?);
        gold.push(row.gold == "true");
    }
    let (threshold, accuracy) =
        super::metrics::best_threshold_tuned_on(&probabilities, &gold).context("tune threshold")?;
    Ok((threshold, accuracy, dev.rows.len()))
}
