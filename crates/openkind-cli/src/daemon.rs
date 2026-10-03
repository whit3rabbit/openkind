//! Resolve the daemon from the same installation as the CLI.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

fn resolve(override_path: Option<PathBuf>, executable: &Path) -> PathBuf {
    if let Some(path) = override_path {
        return path;
    }
    let filename = if cfg!(windows) {
        "openkindd.exe"
    } else {
        "openkindd"
    };
    if let Some(parent) = executable.parent() {
        let sibling = parent.join(filename);
        if sibling.is_file() {
            return sibling;
        }
    }
    PathBuf::from(filename)
}

pub(crate) fn executable() -> Result<PathBuf> {
    let current = std::env::current_exe().context("locate the CLI installation")?;
    Ok(resolve(
        std::env::var_os("OPENKINDD_BINARY")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        &current,
    ))
}

pub(crate) fn doctor(
    json: bool,
    cuda_device: usize,
    rocm_device: usize,
    runtime: Option<PathBuf>,
) -> Result<()> {
    let mut command = std::process::Command::new(executable()?);
    command.args([
        "--diagnose-backends",
        "--cuda-device",
        &cuda_device.to_string(),
        "--rocm-device",
        &rocm_device.to_string(),
    ]);
    if json {
        command.arg("--json");
    }
    if let Some(runtime) = runtime {
        command.arg("--onnx-runtime").arg(runtime);
    }
    let status = command
        .status()
        .context("run daemon diagnostics (install openkindd beside openkind)")?;
    anyhow::ensure!(status.success(), "daemon diagnostics failed with {status}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn override_sibling_and_path_precedence() {
        let directory =
            std::env::temp_dir().join(format!("openkind-daemon-resolver-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let cli = directory.join("openkind");
        let name = if cfg!(windows) {
            "openkindd.exe"
        } else {
            "openkindd"
        };
        assert_eq!(resolve(None, &cli), PathBuf::from(name));
        let sibling = directory.join(name);
        std::fs::write(&sibling, []).unwrap();
        assert_eq!(resolve(None, &cli), sibling);
        let explicit = PathBuf::from("custom-daemon");
        assert_eq!(resolve(Some(explicit.clone()), &cli), explicit);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
