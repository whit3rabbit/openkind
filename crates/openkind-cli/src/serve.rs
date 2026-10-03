use anyhow::{Context, Result};
use std::path::PathBuf;

/// Runs the `openkindd` daemon binary with configured flags and credentials.
///
/// On Unix the wrapper `exec`s into the daemon so the supervisor-targeted PID
/// is preserved. On Windows the wrapper stays resident and blocks on the
/// daemon; Ctrl-C reaches both processes through the shared console and the
/// daemon drains gracefully, though the wrapper's own exit code is not
/// forwarded for signal-driven exits.
pub fn cmd_serve(
    http_addr: String,
    grpc_addr: String,
    models: String,
    installed_models: String,
    models_dir: Option<PathBuf>,
    api_key: Option<String>,
) -> Result<()> {
    let mut cmd = std::process::Command::new(crate::daemon::executable()?);
    cmd.arg("--http-addr").arg(http_addr);
    cmd.arg("--grpc-addr").arg(grpc_addr);
    cmd.arg("--models").arg(models);
    // Forward empty selections too, so an inherited environment setting
    // cannot restore installations explicitly cleared by the CLI caller.
    cmd.arg("--installed-models").arg(installed_models);
    if let Some(dir) = models_dir {
        cmd.arg("--models-dir").arg(dir);
    }
    if let Some(key) = api_key {
        // Pass via environment variable to avoid leaking the secret in
        // process table listings (e.g. ps aux / /proc/*/cmdline).
        cmd.env("OPENKIND_API_KEY", key);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Replacing the wrapper preserves the PID targeted by supervisors
        // and lets the daemon receive shutdown signals and drain requests.
        Err(cmd.exec()).context("execute openkindd (is openkindd built and on PATH?)")
    }
    #[cfg(not(unix))]
    {
        let status = cmd
            .status()
            .context("execute openkindd (is openkindd built and on PATH?)")?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
        Ok(())
    }
}
