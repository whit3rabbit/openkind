use std::path::PathBuf;

use anyhow::{Context, Result};
use opendecision_core::{validate_request, SystemRequest};

/// Maximum allowed input file size (32 MB) to prevent local memory exhaustion.
pub const MAX_CLI_INPUT_BYTES: u64 = 32 * 1024 * 1024;

pub fn cmd_inspect(file: PathBuf) -> Result<()> {
    let meta = std::fs::metadata(&file).with_context(|| format!("stat {}", file.display()))?;
    if meta.len() > MAX_CLI_INPUT_BYTES {
        anyhow::bail!(
            "file {} exceeds maximum allowed size ({} bytes, limit {} bytes)",
            file.display(),
            meta.len(),
            MAX_CLI_INPUT_BYTES
        );
    }
    let raw = std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
    let req: SystemRequest =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", file.display()))?;
    validate_request(&req).with_context(|| format!("validate {}", file.display()))?;
    println!(
        "{} ✓ (model={}, {} questions)",
        file.display(),
        req.model,
        req.questions.len()
    );
    Ok(())
}
