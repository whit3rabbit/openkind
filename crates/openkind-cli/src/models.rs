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
    let rows = catalog_rows(&catalog.models);
    if rows.is_empty() {
        output::print_heading("Curated catalog profiles available to pull");
        println!("No curated profiles are available.");
    } else {
        output::print_table(
            "Curated catalog profiles available to pull",
            &[
                "NAME",
                "ALIASES",
                "PROFILE",
                "STATUS",
                "CONTEXT",
                "DESCRIPTION",
            ],
            &rows,
        );
    }
    Ok(())
}

/// One row per curated profile: name, aliases (or a placeholder), profile
/// ID, support status, context window, and description.
fn catalog_rows(models: &[openkind_model_store::CatalogEntry]) -> Vec<Vec<String>> {
    models
        .iter()
        .map(|model| {
            vec![
                model.name.clone(),
                if model.aliases.is_empty() {
                    "—".to_owned()
                } else {
                    model.aliases.join(", ")
                },
                model.profile_id.to_owned(),
                model.support_status.to_owned(),
                model.context_limit.to_string(),
                model.description.to_owned(),
            ]
        })
        .collect()
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
    if name != manifest.name {
        println!(
            "Resolved alias {name} to the curated profile {}.",
            manifest.name
        );
    }
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
    let rows = list_rows(&models);
    output::print_table(
        "Installed model profiles",
        &["NAME", "PROFILE", "STATUS", "DESCRIPTION"],
        &rows,
    );
    Ok(())
}

/// One row per installed profile: name, profile ID, support status, and
/// description.
fn list_rows(models: &[Manifest]) -> Vec<Vec<String>> {
    models
        .iter()
        .map(|model| {
            vec![
                model.name.clone(),
                model.profile_id.clone(),
                model.support_status.clone(),
                model.description.clone(),
            ]
        })
        .collect()
}

/// One row per pinned artifact: path, size, digest, and upstream source.
fn artifact_rows(manifest: &Manifest) -> Vec<Vec<String>> {
    manifest
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
        .collect()
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

    let rows = artifact_rows(manifest);
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

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_model_store::{Artifact, CatalogEntry, Source};

    fn entry() -> CatalogEntry {
        CatalogEntry {
            name: "fixture:abc123".into(),
            aliases: vec!["short:1".into()],
            profile_id: "profile".into(),
            loader_id: "loader".into(),
            description: "Offline fixture".into(),
            context_limit: 262_144,
            support_status: "rust-loadable".into(),
            manifest_path: "manifests/fixture.json".into(),
            manifest_sha256: "a".repeat(64),
        }
    }

    fn manifest() -> Manifest {
        Manifest {
            schema: "openkind-model/v1".into(),
            name: "fixture:abc123".into(),
            profile_id: "profile".into(),
            loader_id: "loader".into(),
            description: "Offline fixture".into(),
            release_date: "2026-09-25".into(),
            support_status: "rust-loadable".into(),
            question_types: vec!["choice".into()],
            artifacts: vec![Artifact {
                path: "bundle/head.bin".into(),
                size: 2048,
                sha256: "a".repeat(64),
                source: Source {
                    kind: "github".into(),
                    repository: "example/models".into(),
                    revision: "b".repeat(40),
                    path: "head.bin".into(),
                },
            }],
        }
    }

    #[test]
    fn catalog_rows_render_the_context_column_and_alias_placeholder() {
        let rows = catalog_rows(&[entry()]);
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0],
            vec![
                "fixture:abc123".to_owned(),
                "short:1".to_owned(),
                "profile".to_owned(),
                "rust-loadable".to_owned(),
                "262144".to_owned(),
                "Offline fixture".to_owned(),
            ],
            "the CONTEXT column carries the profile's context window"
        );

        let mut aliasless = entry();
        aliasless.aliases.clear();
        let rows = catalog_rows(&[aliasless]);
        assert_eq!(rows[0][1], "—", "an empty alias list renders a placeholder");

        assert!(catalog_rows(&[]).is_empty());
    }

    #[test]
    fn list_rows_render_one_row_per_installed_profile() {
        let rows = list_rows(&[manifest()]);
        assert_eq!(
            rows[0],
            vec![
                "fixture:abc123".to_owned(),
                "profile".to_owned(),
                "rust-loadable".to_owned(),
                "Offline fixture".to_owned(),
            ]
        );
        assert!(list_rows(&[]).is_empty());
    }

    #[test]
    fn artifact_rows_render_sizes_digests_and_sources() {
        let rows = artifact_rows(&manifest());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], "bundle/head.bin");
        assert_eq!(rows[0][1], "2.0 KiB");
        assert_eq!(rows[0][2], "a".repeat(64));
        assert_eq!(
            rows[0][3],
            format!("github: example/models@{}/head.bin", "b".repeat(40))
        );
    }
}
