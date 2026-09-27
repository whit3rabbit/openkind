use std::path::PathBuf;

use anyhow::{Context, Result};
use openkind_model_store::{default_models_dir, Manifest, ModelStore};

use crate::output::{self, PullProgress};

fn store(dir: Option<PathBuf>) -> Result<ModelStore> {
    let dir = match dir {
        Some(dir) => dir,
        None => default_models_dir().context("model directory")?,
    };
    Ok(ModelStore::new(dir)?)
}

pub(super) fn catalog(json: bool) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let catalog = runtime.block_on(store(None)?.catalog())?;
    if json {
        println!("{}", serde_json::to_string(&catalog)?);
        return Ok(());
    }
    let rows: Vec<Vec<String>> = catalog
        .models
        .into_iter()
        .map(|model| {
            vec![
                model.name,
                model.profile_id,
                model.support_status,
                model.description,
            ]
        })
        .collect();
    if rows.is_empty() {
        output::print_heading("Curated catalog profiles available to pull");
        println!("No curated profiles are available.");
    } else {
        output::print_table(
            "Curated catalog profiles available to pull",
            &["NAME", "PROFILE", "STATUS", "DESCRIPTION"],
            &rows,
        );
    }
    Ok(())
}

pub(super) fn pull(name: &str, dir: Option<PathBuf>) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let mut progress = PullProgress::new();
    let result = runtime.block_on(store(dir)?.pull(name, |path, done, total| {
        progress.update(path, done, total);
    }));
    let manifest = match result {
        Ok(manifest) => manifest,
        Err(error) => {
            let artifact_error = matches!(
                &error,
                openkind_model_store::Error::DigestMismatch(_) | openkind_model_store::Error::Io(_)
            );
            progress.finish_failure(artifact_error);
            return Err(error.into());
        }
    };
    let already_installed = !progress.has_progress();
    let downloaded = progress.downloaded_bytes();
    progress.finish_success();
    if already_installed {
        eprintln!("{} is already installed and verified.", manifest.name);
    }
    println!(
        "Installed {} ({}){}",
        manifest.name,
        manifest.support_status,
        if already_installed {
            String::new()
        } else {
            format!("; {} downloaded", output::format_bytes(downloaded))
        }
    );
    println!(
        "To activate it, restart the daemon with: openkind serve --installed-models {}",
        manifest.name
    );
    Ok(())
}

pub(super) fn list(dir: Option<PathBuf>, json: bool) -> Result<()> {
    let models = store(dir)?.list()?;
    if json {
        println!("{}", serde_json::to_string(&models)?);
        return Ok(());
    }
    if models.is_empty() {
        output::print_heading("Installed model profiles");
        println!("No models are installed.");
        return Ok(());
    }
    let rows: Vec<Vec<String>> = models
        .into_iter()
        .map(|model| {
            vec![
                model.name,
                model.profile_id,
                model.support_status,
                model.description,
            ]
        })
        .collect();
    output::print_table(
        "Installed model profiles",
        &["NAME", "PROFILE", "STATUS", "DESCRIPTION"],
        &rows,
    );
    Ok(())
}

pub(super) fn show(name: &str, dir: Option<PathBuf>, json: bool) -> Result<()> {
    let manifest = store(dir)?.show(name)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    print_manifest(&manifest);
    Ok(())
}

fn print_manifest(manifest: &Manifest) {
    output::print_heading("Installed model profile");
    output::print_key_value("Name", &manifest.name);
    output::print_key_value("Status", &manifest.support_status);
    output::print_key_value("Profile", &manifest.profile_id);
    output::print_key_value("Loader", &manifest.loader_id);
    output::print_key_value("Release date", &manifest.release_date);
    output::print_key_value("Description", &manifest.description);
    output::print_key_value("Question types", &manifest.question_types.join(", "));

    let rows: Vec<Vec<String>> = manifest
        .artifacts
        .iter()
        .map(|artifact| {
            vec![
                artifact.path.clone(),
                output::format_bytes(artifact.size),
                artifact.sha256.clone(),
                format!(
                    "{}: {}@{}/{}",
                    artifact.source.kind,
                    artifact.source.repository,
                    artifact.source.revision,
                    artifact.source.path
                ),
            ]
        })
        .collect();
    output::print_table(
        "Pinned artifacts",
        &["PATH", "SIZE", "SHA-256", "SOURCE"],
        &rows,
    );
    println!(
        "Daemon activation requires a restart with --installed-models {}.",
        manifest.name
    );
}

pub(super) fn rm(name: &str, dir: Option<PathBuf>) -> Result<()> {
    store(dir)?.rm(name)?;
    println!("Removed installed model {name}");
    Ok(())
}
