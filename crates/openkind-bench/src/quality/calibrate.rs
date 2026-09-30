//! Post-hoc joint-distribution calibration, fitted on a calibration partition
//! and evaluated locked on a disjoint gate partition.
//!
//! Three post-hoc arms are fitted by deterministic NLL minimization over the
//! calibration partition's forward joint logits: a positive temperature, a
//! semantic-none logit offset, and a coordinate-descent combination of both.
//! All parameters are locked before any gate row is scored. A temperature
//! alone preserves the winning class; the none offset can change rejection but
//! cannot supply missing evidence. Raw distributions are scored alongside as
//! the unlocked control.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use openkind_backends::qwen35::experimental::{PreparedChoice, Qwen35ScoringProbe, ScoringResult};
use openkind_backends::qwen35::{
    Qwen35Backend, BACKBONE_ID, BACKBONE_REVISION, CALIBRATION_TEMPERATURE, PROFILE_ID,
    SEMANTIC_NONE_OPTION, TOKENIZER_JSON_SHA256,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::metrics::{self};
use super::{load, Record, TimedScore};

/// One locked post-hoc calibration arm.
#[derive(Debug, Clone, Copy)]
struct Arm {
    name: &'static str,
    temperature: f64,
    none_offset: f64,
}

pub struct CalibrateArgs {
    pub calibration: PathBuf,
    pub gate: PathBuf,
    pub bundle_root: PathBuf,
    pub checkpoint_root: PathBuf,
    pub tokenizer: PathBuf,
    pub backend: Qwen35Backend,
    pub output_dir: PathBuf,
    pub host: String,
    pub commit: String,
}

/// One calibration-partition row reduced to raw forward logits and gold index.
struct FitRow {
    logits: Vec<f64>,
    gold: usize,
}

const TEMPERATURE_GRID: (f64, f64, usize) = (0.25, 8.0, 512);
const OFFSET_GRID: (f64, f64, usize) = (-8.0, 8.0, 1024);
const REFINE_ITERATIONS: usize = 80;
const COORDINATE_SWEEPS: usize = 3;
const NLL_PROBABILITY_FLOOR: f64 = 1e-15;

pub fn run(args: &CalibrateArgs) -> Result<Value> {
    anyhow::ensure!(
        !args.host.trim().is_empty() && !args.commit.trim().is_empty(),
        "host and commit attribution required"
    );
    let (calibration_rows, calibration_digest) = load(&args.calibration)
        .with_context(|| format!("calibration partition {}", args.calibration.display()))?;
    let (gate_rows, gate_digest) =
        load(&args.gate).with_context(|| format!("gate partition {}", args.gate.display()))?;
    let calibration_groups: BTreeSet<&str> = calibration_rows
        .iter()
        .map(|row| row.source_group.as_str())
        .collect();
    let gate_groups: BTreeSet<&str> = gate_rows
        .iter()
        .map(|row| row.source_group.as_str())
        .collect();
    let shared_groups: Vec<&str> = calibration_groups
        .intersection(&gate_groups)
        .copied()
        .collect();
    anyhow::ensure!(
        shared_groups.is_empty(),
        "calibration and gate partitions share source groups: {shared_groups:?}"
    );
    let calibration_ids: BTreeSet<&str> = calibration_rows
        .iter()
        .map(|row| row.row.id.as_str())
        .collect();
    let gate_ids: BTreeSet<&str> = gate_rows.iter().map(|row| row.row.id.as_str()).collect();
    let shared_ids: Vec<&str> = calibration_ids.intersection(&gate_ids).copied().collect();
    anyhow::ensure!(
        shared_ids.is_empty(),
        "calibration and gate partitions share row ids: {shared_ids:?}"
    );
    let calibration_tasks: BTreeSet<&str> = calibration_rows
        .iter()
        .map(|row| row.task.as_str())
        .collect();
    let gate_tasks: BTreeSet<&str> = gate_rows.iter().map(|row| row.task.as_str()).collect();
    let shared_tasks: Vec<&str> = calibration_tasks
        .intersection(&gate_tasks)
        .copied()
        .collect();

    let start = Instant::now();
    let probe = Qwen35ScoringProbe::load(
        &args.bundle_root,
        &args.checkpoint_root,
        &args.tokenizer,
        args.backend,
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    let calibration_prepared = calibration_rows
        .iter()
        .map(|row| Ok(probe.prepare(&row.row.state, &row.row.question()?)?))
        .collect::<Result<Vec<_>>>()?;
    let gate_prepared = gate_rows
        .iter()
        .map(|row| Ok(probe.prepare(&row.row.state, &row.row.question()?)?))
        .collect::<Result<Vec<_>>>()?;

    // ---- Fit phase: calibration partition only --------------------------
    let mut fit_rows = Vec::with_capacity(calibration_rows.len());
    let mut forward_tokens = 0usize;
    for (row, input) in calibration_rows.iter().zip(&calibration_prepared) {
        let (logits, _) = timed_logits(|| probe.joint_logits(input, false))?;
        let labels = row_labels(input);
        let gold = labels
            .iter()
            .position(|label| label == &row.gold)
            .expect("gold is an offered option");
        forward_tokens += logits.input_tokens;
        fit_rows.push(FitRow {
            logits: logits.values,
            gold,
        });
    }
    let raw_in_sample_nll = nll_mean(&fit_rows, 1.0, 0.0)?;
    let temperature = fit_scalar(
        TEMPERATURE_GRID.0,
        TEMPERATURE_GRID.1,
        TEMPERATURE_GRID.2,
        true,
        &|value| nll_mean(&fit_rows, value, 0.0).expect("finite temperature objective"),
    );
    let offset = fit_scalar(
        OFFSET_GRID.0,
        OFFSET_GRID.1,
        OFFSET_GRID.2,
        false,
        &|value| nll_mean(&fit_rows, 1.0, value).expect("finite offset objective"),
    );
    // Coordinate descent for the combined arm, always from the raw start.
    let mut combined = (1.0_f64, 0.0_f64);
    for _ in 0..COORDINATE_SWEEPS {
        combined.0 = fit_scalar(
            TEMPERATURE_GRID.0,
            TEMPERATURE_GRID.1,
            TEMPERATURE_GRID.2,
            true,
            &|value| nll_mean(&fit_rows, value, combined.1).expect("finite temperature objective"),
        );
        combined.1 = fit_scalar(
            OFFSET_GRID.0,
            OFFSET_GRID.1,
            OFFSET_GRID.2,
            false,
            &|value| nll_mean(&fit_rows, combined.0, value).expect("finite offset objective"),
        );
    }
    // ---- Parameters are locked from here on -----------------------------
    let arms = [
        Arm {
            name: "joint_raw",
            temperature: 1.0,
            none_offset: 0.0,
        },
        Arm {
            name: "joint_temperature",
            temperature,
            none_offset: 0.0,
        },
        Arm {
            name: "joint_none_offset",
            temperature: 1.0,
            none_offset: offset,
        },
        Arm {
            name: "joint_temperature_offset",
            temperature: combined.0,
            none_offset: combined.1,
        },
    ];
    let arm_fits = arms
        .iter()
        .map(|arm| {
            Ok(json!({
                "name": arm.name,
                "temperature": arm.temperature,
                "none_offset": arm.none_offset,
                "in_sample_nll": nll_mean(&fit_rows, arm.temperature, arm.none_offset)?,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let fit_block = json!({
        "method": "deterministic grid search plus golden-section NLL refinement; no randomness",
        "temperature_grid": {"low": TEMPERATURE_GRID.0, "high": TEMPERATURE_GRID.1, "points": TEMPERATURE_GRID.2, "scale": "log"},
        "offset_grid": {"low": OFFSET_GRID.0, "high": OFFSET_GRID.1, "points": OFFSET_GRID.2, "scale": "linear"},
        "refine_iterations": REFINE_ITERATIONS,
        "coordinate_sweeps": COORDINATE_SWEEPS,
        "raw_in_sample_nll": raw_in_sample_nll,
        "arms": arm_fits,
    });

    // ---- Locked evaluation: calibration partition (in-sample diagnostic) -
    let calibration_records = score_partition(
        &probe,
        "calibration",
        &calibration_rows,
        &calibration_prepared,
        &arms,
        false,
    )?;

    // ---- Locked evaluation: disjoint gate, forward and reverse renders ---
    let gate_records = score_partition(&probe, "gate", &gate_rows, &gate_prepared, &arms, false)?;
    let gate_reverse_records = score_partition(
        &probe,
        "gate-reverse",
        &gate_rows,
        &gate_prepared,
        &arms,
        true,
    )?;

    fs::create_dir_all(&args.output_dir)?;
    write_predictions(
        &args.output_dir.join("calibration-predictions.jsonl"),
        &calibration_records,
    )?;
    write_predictions(&args.output_dir.join("predictions.jsonl"), &gate_records)?;
    write_predictions(
        &args.output_dir.join("reverse-transfer-predictions.jsonl"),
        &gate_reverse_records,
    )?;

    let gate_refs: Vec<_> = gate_records.iter().collect();
    let overall: BTreeMap<_, _> = arms
        .iter()
        .map(|arm| (arm.name, metrics::metrics(&gate_refs, arm.name)))
        .collect();
    let tasks: BTreeSet<&str> = gate_records
        .iter()
        .map(|record| record.task.as_str())
        .collect();
    let per_task: BTreeMap<_, BTreeMap<_, _>> = tasks
        .into_iter()
        .map(|task| {
            let subset: Vec<_> = gate_records
                .iter()
                .filter(|record| record.task == task)
                .collect();
            (
                task,
                arms.iter()
                    .map(|arm| (arm.name, metrics::metrics(&subset, arm.name)))
                    .collect(),
            )
        })
        .collect();
    let paired_deltas: BTreeMap<_, _> = arms[1..]
        .iter()
        .map(|arm| {
            (
                arm.name,
                super::report::paired_vs(&gate_records, "joint_raw", arm.name),
            )
        })
        .collect();
    let reverse_refs: Vec<_> = gate_reverse_records.iter().collect();
    let reverse_overall: BTreeMap<_, _> = arms
        .iter()
        .map(|arm| (arm.name, metrics::metrics(&reverse_refs, arm.name)))
        .collect();
    let calibration_refs: Vec<_> = calibration_records.iter().collect();
    let in_sample_overall: BTreeMap<_, _> = arms
        .iter()
        .map(|arm| (arm.name, metrics::metrics(&calibration_refs, arm.name)))
        .collect();
    let mut summary = json!({
        "schema": "openkind-choice-calibration/v1",
        "host": args.host, "commit": args.commit,
        "backbone": BACKBONE_ID, "backbone_revision": BACKBONE_REVISION,
        "tokenizer_sha256": TOKENIZER_JSON_SHA256,
        "baseline_profile": PROFILE_ID, "baseline_temperature": CALIBRATION_TEMPERATURE,
        "backend": probe.backend().as_str(), "arithmetic_id": probe.arithmetic_id(),
        "calibration_workload": {"sha256": calibration_digest, "rows": calibration_rows.len(), "source_groups": calibration_groups.len()},
        "gate_workload": {"sha256": gate_digest, "rows": gate_rows.len(), "source_groups": gate_groups.len()},
        "partition_isolation": {"shared_source_groups": 0, "shared_row_ids": 0, "shared_tasks": shared_tasks},
        "fit": fit_block,
        "lock": "all temperatures and none offsets were fitted on the calibration partition and fixed before any gate row was scored",
        "scoring_tokens": {"calibration_forward_input_tokens": forward_tokens},
        "nll_probability_floor": NLL_PROBABILITY_FLOOR, "ece_bins": 10,
        "timing_scope": "warm full-forward scorer only; every locked arm reads the same forward logits, so calibrated arms add post-hoc softmax time only; the reverse transfer costs one extra forward per gate row",
        "evidence_scope": "post-hoc calibration diagnostic on supplied labeled Choice rows; fitted parameters qualify only if the locked gate improves; no production promotion and no rejection-threshold fit",
        "calibration_partition_in_sample": {"overall": in_sample_overall},
        "gate": {"overall": overall, "per_task": per_task, "paired_delta_from_raw": paired_deltas},
        "gate_reverse_transfer": {"overall": reverse_overall},
    });
    summary["executable_sha256"] = serde_json::json!(format!(
        "{:x}",
        Sha256::digest(fs::read(std::env::current_exe()?)?)
    ));
    summary["load_seconds"] = json!(load_seconds);
    fs::write(
        args.output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}

fn timed_logits(
    score: impl FnOnce() -> Result<
        openkind_backends::qwen35::experimental::JointLogits,
        openkind_backends::qwen35::Qwen35Error,
    >,
) -> Result<(openkind_backends::qwen35::experimental::JointLogits, f64)> {
    let start = Instant::now();
    let logits = score()?;
    Ok((logits, start.elapsed().as_secs_f64() * 1000.0))
}

fn row_labels(input: &PreparedChoice) -> Vec<String> {
    let mut labels = input.labels().to_vec();
    labels.push(SEMANTIC_NONE_OPTION.into());
    labels
}

fn nll_mean(rows: &[FitRow], temperature: f64, none_offset: f64) -> Result<f64> {
    if rows.is_empty() {
        anyhow::bail!("calibration partition is empty");
    }
    let mut total = 0.0;
    for row in rows {
        let probabilities =
            Qwen35ScoringProbe::calibrated_probabilities(&row.logits, temperature, none_offset)?;
        total += -probabilities[row.gold].max(NLL_PROBABILITY_FLOOR).ln();
    }
    Ok(total / rows.len() as f64)
}

/// Deterministic 1-D minimization: grid argmin followed by golden-section
/// refinement inside the winning grid cell. `log_scale` searches in log space.
fn fit_scalar(
    low: f64,
    high: f64,
    points: usize,
    log_scale: bool,
    objective: &dyn Fn(f64) -> f64,
) -> f64 {
    let to_parameter = |fraction: f64| {
        if log_scale {
            (low.ln() + fraction * (high.ln() - low.ln())).exp()
        } else {
            low + fraction * (high - low)
        }
    };
    let evaluate = |fraction: f64| objective(to_parameter(fraction));
    let mut best_fraction = 0.0;
    let mut best_value = f64::INFINITY;
    for index in 0..points {
        let fraction = index as f64 / (points - 1) as f64;
        let value = evaluate(fraction);
        if value < best_value {
            best_value = value;
            best_fraction = fraction;
        }
    }
    let cell = 1.0 / (points - 1) as f64;
    let mut lo = (best_fraction - cell).max(0.0);
    let mut hi = (best_fraction + cell).min(1.0);
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut left = hi - golden * (hi - lo);
    let mut right = lo + golden * (hi - lo);
    let mut left_value = evaluate(left);
    let mut right_value = evaluate(right);
    for _ in 0..REFINE_ITERATIONS {
        if left_value < right_value {
            hi = right;
            right = left;
            right_value = left_value;
            left = hi - golden * (hi - lo);
            left_value = evaluate(left);
        } else {
            lo = left;
            left = right;
            left_value = right_value;
            right = lo + golden * (hi - lo);
            right_value = evaluate(right);
        }
    }
    to_parameter((lo + hi) / 2.0)
}

fn score_partition(
    probe: &Qwen35ScoringProbe,
    partition: &'static str,
    rows: &[super::LabeledRow],
    prepared: &[PreparedChoice],
    arms: &[Arm],
    reversed: bool,
) -> Result<Vec<Record>> {
    let mut records = Vec::with_capacity(rows.len());
    for (index, (row, input)) in rows.iter().zip(prepared).enumerate() {
        let (logits, elapsed_ms) = timed_logits(|| probe.joint_logits(input, reversed))?;
        let labels = row_labels(input);
        let mut methods = BTreeMap::new();
        for arm in arms {
            let start = Instant::now();
            let probabilities = Qwen35ScoringProbe::calibrated_probabilities(
                &logits.values,
                arm.temperature,
                arm.none_offset,
            )?;
            let applied_ms = start.elapsed().as_secs_f64() * 1000.0;
            let score = ScoringResult {
                probabilities: labels.iter().cloned().zip(probabilities).collect(),
                input_tokens: logits.input_tokens,
                forwards: 1,
            };
            metrics::validate(&score.probabilities, &row.gold)?;
            // The raw arm owns the forward cost; calibrated arms add only the
            // post-hoc softmax application over the same locked logits.
            methods.insert(
                arm.name.to_owned(),
                TimedScore {
                    score,
                    elapsed_ms: if arm.name == "joint_raw" {
                        elapsed_ms
                    } else {
                        applied_ms
                    },
                },
            );
        }
        records.push(Record {
            id: row.row.id.clone(),
            gold: row.gold.clone(),
            task: row.task.clone(),
            source_group: row.source_group.clone(),
            methods,
        });
        if index % 8 == 0 || index + 1 == rows.len() {
            eprintln!(
                "calibration scoring ({partition}): {}/{}",
                index + 1,
                rows.len()
            );
        }
    }
    Ok(records)
}

fn write_predictions(path: &std::path::Path, records: &[Record]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_rows() -> Vec<FitRow> {
        // Deterministic logits; even rows put gold at the argmax (sharpening
        // pressure) and odd rows at the argmin (flattening pressure), so the
        // NLL optimum sits strictly inside the grid.
        let rows: Vec<Vec<f64>> = (0..40)
            .map(|row| {
                if row % 2 == 0 {
                    // A clear winner rewards some sharpening.
                    (0..5)
                        .map(|slot| {
                            if slot == 0 {
                                1.2
                            } else {
                                -0.2 - 0.1 * slot as f64
                            }
                        })
                        .collect()
                } else {
                    // Mildly spread pseudo-random logits punish sharpening
                    // when the gold is the argmin, but not enough to push the
                    // optimum past the grid edge.
                    (0..5)
                        .map(|slot| ((row * 7 + slot * 13) % 11) as f64 / 12.0 - 0.375)
                        .collect()
                }
            })
            .collect();
        rows.into_iter()
            .enumerate()
            .map(|(index, logits)| {
                let gold = if index % 2 == 0 {
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                } else {
                    logits
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                };
                FitRow { logits, gold }
            })
            .collect()
    }

    #[test]
    fn temperature_fit_is_optimal_and_scales_with_the_logits() {
        let rows = synthetic_rows();
        let fitted = fit_scalar(
            TEMPERATURE_GRID.0,
            TEMPERATURE_GRID.1,
            TEMPERATURE_GRID.2,
            true,
            &|value| nll_mean(&rows, value, 0.0).expect("finite objective"),
        );
        assert!(
            fitted > TEMPERATURE_GRID.0 && fitted < TEMPERATURE_GRID.1,
            "optimum must be interior, got {fitted}"
        );
        // The fit never loses against the raw distribution.
        assert!(nll_mean(&rows, fitted, 0.0).unwrap() <= nll_mean(&rows, 1.0, 0.0).unwrap());
        // Equivariance: logits scaled by s move the optimum to s * T. The
        // temperature arm stays a pure property of the logit scale.
        let doubled: Vec<FitRow> = rows
            .iter()
            .map(|row| FitRow {
                logits: row.logits.iter().map(|z| z * 2.0).collect(),
                gold: row.gold,
            })
            .collect();
        let refitted = fit_scalar(
            TEMPERATURE_GRID.0,
            TEMPERATURE_GRID.1,
            TEMPERATURE_GRID.2,
            true,
            &|value| nll_mean(&doubled, value, 0.0).expect("finite objective"),
        );
        assert!(
            (refitted - 2.0 * fitted).abs() < 2e-3,
            "refitted {refitted} versus 2 * {fitted}"
        );
    }

    #[test]
    fn none_offset_fit_is_optimal_and_tracks_the_none_logit_shift() {
        let rows = synthetic_rows();
        let fitted = fit_scalar(
            OFFSET_GRID.0,
            OFFSET_GRID.1,
            OFFSET_GRID.2,
            false,
            &|value| nll_mean(&rows, 1.0, value).expect("finite objective"),
        );
        assert!(
            fitted > OFFSET_GRID.0 && fitted < OFFSET_GRID.1,
            "optimum must be interior, got {fitted}"
        );
        assert!(nll_mean(&rows, 1.0, fitted).unwrap() <= nll_mean(&rows, 1.0, 0.0).unwrap());
        // Equivariance: raising every none logit by d moves the fitted offset
        // to offset - d.
        let shifted: Vec<FitRow> = rows
            .iter()
            .map(|row| {
                let mut logits = row.logits.clone();
                let last = logits.len() - 1;
                logits[last] += 1.25;
                FitRow {
                    logits,
                    gold: row.gold,
                }
            })
            .collect();
        let refitted = fit_scalar(
            OFFSET_GRID.0,
            OFFSET_GRID.1,
            OFFSET_GRID.2,
            false,
            &|value| nll_mean(&shifted, 1.0, value).expect("finite objective"),
        );
        assert!(
            (refitted - (fitted - 1.25)).abs() < 2e-3,
            "refitted {refitted} versus {} - 1.25",
            fitted
        );
    }

    #[test]
    fn calibrated_probabilities_agree_with_the_probe_math() {
        let logits = vec![0.7, -1.2, 0.1, 0.9];
        let probabilities =
            Qwen35ScoringProbe::calibrated_probabilities(&logits, 0.5, -0.3).unwrap();
        // Hand-computed softmax over [1.4, -2.4, 0.2, (0.9 - 0.3) / 0.5].
        let scaled = [1.4_f64, -2.4, 0.2, 1.2];
        let total: f64 = scaled.iter().map(|v| v.exp()).sum();
        for (probability, value) in probabilities.iter().zip(scaled) {
            assert!((probability - value.exp() / total).abs() < 1e-12);
        }
    }
}
