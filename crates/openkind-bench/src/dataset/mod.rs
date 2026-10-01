//! `dataset` subcommands: curated public-dataset acquisition, materialization,
//! and accuracy evaluation. Acquisition is always explicit and networked;
//! materialization and evaluation read only the verified local install.
//!
//! Dataset bytes are never committed to this repository and never
//! redistributed: the registry pins identities, revisions, sizes, and digests
//! only. Methodology is owned by `docs/BENCHMARKS.md`.

pub mod eval;
pub mod materialize;
pub mod metrics;
pub mod templates;

#[cfg(test)]
mod tests;

use anyhow::{Context, Result};
use openkind_datasets::{default_datasets_dir, DatasetStore};

use self::eval::EvalArgs;

/// Run one `dataset` subcommand.
///
/// # Errors
/// Returns an error for unknown datasets, failed downloads or verification,
/// or materialization and evaluation failures.
pub fn run_command(action: DatasetAction) -> Result<()> {
    match action {
        DatasetAction::List { datasets_dir } => {
            let store = store(datasets_dir)?;
            let installed: Vec<String> =
                store.list()?.into_iter().map(|entry| entry.name).collect();
            let mut out = Vec::new();
            for entry in store.registry()?.datasets {
                out.push(serde_json::json!({
                    "name": entry.name,
                    "description": entry.description,
                    "hf_repo": entry.hf_repo,
                    "primitives": entry.primitives,
                    "license": entry.license,
                    "gated": entry.gated,
                    "installed": installed.contains(&entry.name),
                }));
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        DatasetAction::Pull { name, datasets_dir } => {
            let store = store(datasets_dir)?;
            eprintln!("[dataset] auth: {}", store.token_source().describe());
            let runtime = tokio::runtime::Runtime::new()?;
            let entry = runtime.block_on(store.pull(&name, |path, present, total| {
                eprintln!("[dataset] {path}: {present}/{total} bytes");
            }))?;
            eprintln!(
                "[dataset] installed {} ({} files, revision {})",
                entry.name,
                entry.files.len(),
                &entry.hf_revision[..12]
            );
        }
        DatasetAction::Rm { name, datasets_dir } => {
            let store = store(datasets_dir)?;
            store.rm(&name)?;
            eprintln!("[dataset] removed {name}");
        }
        DatasetAction::Verify { name, datasets_dir } => {
            let store = store(datasets_dir)?;
            let entry = store.verify(&name)?;
            eprintln!(
                "[dataset] verified {} ({} files, revision {})",
                entry.name,
                entry.files.len(),
                &entry.hf_revision[..12]
            );
        }
        DatasetAction::Pin { name, datasets_dir } => {
            let store = store(datasets_dir)?;
            eprintln!("[dataset] auth: {}", store.token_source().describe());
            let runtime = tokio::runtime::Runtime::new()?;
            let entry = runtime.block_on(store.pin(&name, |path, present, total| {
                eprintln!("[dataset] {path}: {present}/{total} bytes");
            }))?;
            // The emitted entry is what belongs in registry/v1/datasets.json.
            println!("{}", serde_json::to_string_pretty(&entry)?);
        }
        DatasetAction::Build {
            name,
            split,
            datasets_dir,
            limit,
            output,
        } => {
            let store = store(datasets_dir)?;
            let installed = store.installed(&name)?;
            let rows = materialize::materialize(&installed, split.as_str(), limit)?;
            let path = output.unwrap_or_else(|| {
                std::path::PathBuf::from(format!(
                    "bench-output/dataset-{}-{}.jsonl",
                    name,
                    split.as_str()
                ))
            });
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut bytes = Vec::new();
            for row in &rows.rows {
                bytes.extend_from_slice(serde_json::to_string(row)?.as_bytes());
                bytes.push(b'\n');
            }
            std::fs::write(&path, bytes)?;
            eprintln!(
                "[dataset] wrote {} rows ({} duplicates, {} template-skipped) to {}",
                rows.rows.len(),
                rows.skipped_duplicates,
                rows.skipped_rows,
                path.display()
            );
        }
        DatasetAction::Eval {
            name,
            split,
            datasets_dir,
            limit,
            engine,
            output_dir,
            host,
            commit,
            pretty,
            bundle_root,
            checkpoint_root,
            tokenizer,
            model_root,
            adapter,
            tune_threshold,
        } => {
            let args = EvalArgs {
                name,
                split: split.as_str().to_owned(),
                limit,
                datasets_dir,
                engine: engine.into(),
                output_dir,
                host,
                commit,
                pretty,
                bundle_root,
                checkpoint_root,
                tokenizer_path: tokenizer,
                model_root,
                adapter,
                tune_threshold,
            };
            let report = eval::run_eval(&args)?;
            let text = if pretty {
                serde_json::to_string_pretty(&report)?
            } else {
                serde_json::to_string(&report)?
            };
            println!("{text}");
        }
    }
    Ok(())
}

fn store(datasets_dir: Option<std::path::PathBuf>) -> Result<DatasetStore> {
    let root = datasets_dir.map_or_else(default_datasets_dir, Ok)?;
    DatasetStore::new(root).context("open the dataset store")
}

/// `dataset` subcommand payloads (mirrors the clap definitions in `args.rs`).
// Windows `PathBuf`s push the variant size ratio over the lint threshold.
#[allow(clippy::large_enum_variant)]
pub enum DatasetAction {
    List {
        datasets_dir: Option<std::path::PathBuf>,
    },
    Pull {
        name: String,
        datasets_dir: Option<std::path::PathBuf>,
    },
    Rm {
        name: String,
        datasets_dir: Option<std::path::PathBuf>,
    },
    Verify {
        name: String,
        datasets_dir: Option<std::path::PathBuf>,
    },
    Pin {
        name: String,
        datasets_dir: Option<std::path::PathBuf>,
    },
    Build {
        name: String,
        split: crate::args::SplitArg,
        datasets_dir: Option<std::path::PathBuf>,
        limit: Option<usize>,
        output: Option<std::path::PathBuf>,
    },
    Eval {
        name: String,
        split: crate::args::SplitArg,
        datasets_dir: Option<std::path::PathBuf>,
        limit: Option<usize>,
        engine: crate::args::EngineArg,
        output_dir: std::path::PathBuf,
        host: Option<String>,
        commit: Option<String>,
        pretty: bool,
        bundle_root: Option<std::path::PathBuf>,
        checkpoint_root: Option<std::path::PathBuf>,
        tokenizer: Option<std::path::PathBuf>,
        model_root: Option<std::path::PathBuf>,
        adapter: Option<std::path::PathBuf>,
        tune_threshold: bool,
    },
}
