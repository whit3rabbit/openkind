use std::path::PathBuf;

use anyhow::{Context, Result};
use openkind_model_store::{default_models_dir, ModelStore};

fn store(dir: Option<PathBuf>) -> Result<ModelStore> {
    let dir = match dir {
        Some(dir) => dir,
        None => default_models_dir().context("model directory")?,
    };
    Ok(ModelStore::new(dir)?)
}

pub(super) fn catalog() -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let catalog = runtime.block_on(store(None)?.catalog())?;
    for model in catalog.models {
        println!(
            "{}\t{}\t{}",
            model.name, model.support_status, model.description
        );
    }
    Ok(())
}

pub(super) fn pull(name: &str, dir: Option<PathBuf>) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let mut last_path = String::new();
    let mut last_report = 0;
    let mut last_seen = 0;
    let mut downloaded = 0_u64;
    let manifest = runtime.block_on(store(dir)?.pull(name, |path, done, total| {
        if path != last_path {
            last_seen = done;
        } else {
            downloaded += done.saturating_sub(last_seen);
            last_seen = done;
        }
        if path != last_path || done == total || done.saturating_sub(last_report) >= 8 * 1024 * 1024
        {
            eprintln!("{path}: {done}/{total} bytes present");
            last_path = path.to_owned();
            last_report = done;
        }
    }))?;
    println!(
        "installed {} ({}); {} artifact bytes downloaded",
        manifest.name, manifest.support_status, downloaded
    );
    Ok(())
}

pub(super) fn list(dir: Option<PathBuf>) -> Result<()> {
    for model in store(dir)?.list()? {
        println!(
            "{}\t{}\t{}",
            model.name, model.support_status, model.description
        );
    }
    Ok(())
}

pub(super) fn show(name: &str, dir: Option<PathBuf>) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&store(dir)?.show(name)?)?
    );
    Ok(())
}

pub(super) fn rm(name: &str, dir: Option<PathBuf>) -> Result<()> {
    store(dir)?.rm(name)?;
    println!("removed {name}");
    Ok(())
}
