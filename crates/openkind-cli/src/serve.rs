use anyhow::{Context, Result};

/// Spawns and supervises the `openkindd` daemon binary with configured flags and credentials.
pub fn cmd_serve(
    http_addr: String,
    grpc_addr: String,
    models: String,
    api_key: Option<String>,
) -> Result<()> {
    let mut cmd = std::process::Command::new("openkindd");
    cmd.arg("--http-addr").arg(http_addr);
    cmd.arg("--grpc-addr").arg(grpc_addr);
    cmd.arg("--models").arg(models);
    if let Some(key) = api_key {
        // Pass via environment variable to avoid leaking the secret in
        // process table listings (e.g. ps aux / /proc/*/cmdline).
        cmd.env("OPENKIND_API_KEY", key);
    }
    let status = cmd
        .status()
        .context("execute openkindd (is openkindd built and on PATH?)")?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
