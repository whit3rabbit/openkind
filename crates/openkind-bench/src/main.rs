//! `openkind-bench`: offline scoring and timing benchmarks for
//! openkind engines over JSONL decision workloads.
//!
//! Methodology, timing scope, and recorded results live in
//! `docs/BENCHMARKS.md`. The harness never downloads artifacts: the mock
//! engine is fully offline, and the native engine loads locally pinned
//! checkpoint, bundle, and tokenizer paths only.

mod args;
mod dataset;
mod gen;
mod quality;
mod score;
mod workload;

#[cfg(test)]
mod tests;

use anyhow::{Context, Result};
use clap::Parser;

use crate::args::{Cli, Commands, DatasetCommand, GenWorkloadOutcome};
use crate::score::{run_score, EngineKind, ScoreArgs, StrategySpec};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::CompareChoice {
            input,
            bundle_root,
            checkpoint_root,
            tokenizer,
            backend,
            output_dir,
            host,
            commit,
        } => {
            let summary = quality::run(&quality::CompareArgs {
                input,
                bundle_root,
                checkpoint_root,
                tokenizer,
                backend: backend.into(),
                output_dir,
                host,
                commit,
            })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
        Commands::CalibrateChoice {
            calibration,
            gate,
            bundle_root,
            checkpoint_root,
            tokenizer,
            backend,
            output_dir,
            host,
            commit,
        } => {
            let summary = quality::run_calibration(&quality::CalibrateArgs {
                calibration,
                gate,
                bundle_root,
                checkpoint_root,
                tokenizer,
                backend: backend.into(),
                output_dir,
                host,
                commit,
            })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
        Commands::GenWorkload {
            states,
            criteria,
            seed,
            output,
        } => {
            let generated = gen::generate_workload(states, criteria, seed, &output)?;
            let outcome = GenWorkloadOutcome {
                output,
                workload: generated,
            };
            println!(
                "{}",
                serde_json::json!({
                    "output": outcome.output.display().to_string(),
                    "rows": outcome.workload.rows.len(),
                    "sha256": outcome.workload.sha256,
                })
            );
            Ok(())
        }
        Commands::Score {
            input,
            engine,
            output_dir,
            strategies,
            reps,
            no_group,
            no_warmup,
            history_aba,
            host,
            commit,
            bundle_root,
            checkpoint_root,
            tokenizer,
            model_root,
            adapter,
            pretty,
        } => {
            let kind = EngineKind::from(engine);
            let strategy_specs: Vec<StrategySpec> = if types_family_engine(kind) {
                if strategies.is_some() {
                    anyhow::bail!("--strategies does not apply to surveyed-family engines");
                }
                Vec::new()
            } else {
                args::parse_strategies(strategies.as_deref())?
            };
            let score_args = ScoreArgs {
                input,
                engine: kind,
                output_dir,
                strategies: strategy_specs,
                reps,
                group: !no_group,
                warmup: !no_warmup,
                history_aba,
                host,
                commit,
                pretty,
                bundle_root,
                checkpoint_root,
                tokenizer_path: tokenizer,
                model_root,
                adapter,
            };
            let outcome = run_score(&score_args)
                .with_context(|| format!("score run over {}", score_args.input.display()))?;
            println!("{}", outcome.summary);
            Ok(())
        }
        Commands::Dataset { action } => {
            let action = match action {
                DatasetCommand::List { datasets_dir } => {
                    dataset::DatasetAction::List { datasets_dir }
                }
                DatasetCommand::Pull { name, datasets_dir } => {
                    dataset::DatasetAction::Pull { name, datasets_dir }
                }
                DatasetCommand::Rm { name, datasets_dir } => {
                    dataset::DatasetAction::Rm { name, datasets_dir }
                }
                DatasetCommand::Verify { name, datasets_dir } => {
                    dataset::DatasetAction::Verify { name, datasets_dir }
                }
                DatasetCommand::Pin { name, datasets_dir } => {
                    dataset::DatasetAction::Pin { name, datasets_dir }
                }
                DatasetCommand::Build {
                    name,
                    split,
                    datasets_dir,
                    limit,
                    output,
                } => dataset::DatasetAction::Build {
                    name,
                    split,
                    datasets_dir,
                    limit,
                    output,
                },
                DatasetCommand::Eval {
                    name,
                    split,
                    datasets_dir,
                    limit,
                    engine,
                    output_dir,
                    host,
                    commit,
                    bundle_root,
                    checkpoint_root,
                    tokenizer,
                    model_root,
                    adapter,
                    tune_threshold,
                    pretty,
                } => dataset::DatasetAction::Eval {
                    name,
                    split,
                    datasets_dir,
                    limit,
                    engine,
                    output_dir,
                    host,
                    commit,
                    bundle_root,
                    checkpoint_root,
                    tokenizer,
                    model_root,
                    adapter,
                    tune_threshold,
                    pretty,
                },
            };
            dataset::run_command(action)
        }
    }
}

fn types_family_engine(kind: EngineKind) -> bool {
    crate::score::is_family_engine_public(kind)
}
