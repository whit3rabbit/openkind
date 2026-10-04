//! Reserve evidence destinations before expensive work without replacing prior evidence.

use std::fs::{self, File, OpenOptions};
use std::path::Path;

use anyhow::{Context, Result};

pub(crate) fn reserve_outputs<const N: usize>(dir: &Path, names: [&str; N]) -> Result<[File; N]> {
    let paths: Vec<_> = names.iter().map(|name| dir.join(name)).collect();
    reserve_paths(&paths)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("evidence reservation count mismatch"))
}

pub(crate) fn reserve_paths(paths: &[std::path::PathBuf]) -> Result<Vec<File>> {
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        // Parent creation must use the same rollback as an output-file conflict.
        let reservation = (|| {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            OpenOptions::new().write(true).create_new(true).open(path)
        })();
        match reservation {
            Ok(file) => files.push(file),
            Err(error) => {
                let reserved = files.len();
                drop(files);
                // Only our successful reservations may be removed on conflict.
                for created in &paths[..reserved] {
                    let _ = fs::remove_file(created);
                }
                return Err(error).with_context(|| {
                    format!(
                        "reserve evidence {}; choose a fresh --output-dir",
                        path.display()
                    )
                });
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflicts_preserve_prior_evidence_and_release_only_new_reservations() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("summary.json"), b"prior evidence").unwrap();
        assert!(reserve_outputs(dir.path(), ["predictions.jsonl", "summary.json"]).is_err());
        assert!(!dir.path().join("predictions.jsonl").exists());
        assert_eq!(
            fs::read(dir.path().join("summary.json")).unwrap(),
            b"prior evidence"
        );
    }

    #[test]
    fn parent_directory_failure_releases_earlier_reservations() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("predictions.jsonl");
        let blocker = dir.path().join("blocked");
        fs::write(&blocker, b"prior evidence").unwrap();
        assert!(reserve_paths(&[first.clone(), blocker.join("summary.json")]).is_err());
        assert!(!first.exists());
        assert_eq!(fs::read(blocker).unwrap(), b"prior evidence");
    }
}
