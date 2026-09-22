use std::path::PathBuf;

use anyhow::{Context, Result};
use openkind_core::{validate_request, SystemRequest, ValidationError};

/// Maximum allowed input file size (32 MB) to prevent local memory exhaustion.
pub const MAX_CLI_INPUT_BYTES: u64 = 32 * 1024 * 1024;

enum RequestInputError {
    Parse(serde_json::Error),
    Validate(ValidationError),
}

fn parse_and_validate(raw: &str) -> Result<SystemRequest, RequestInputError> {
    let request = serde_json::from_str(raw).map_err(RequestInputError::Parse)?;
    validate_request(&request).map_err(RequestInputError::Validate)?;
    Ok(request)
}

/// Parse and validate one in-memory Jev request.
///
/// File-size enforcement remains the caller's responsibility because this
/// pure path is also used by benchmarks and embedders that already own the
/// input buffer.
pub fn parse_and_validate_request(raw: &str) -> Result<SystemRequest> {
    match parse_and_validate(raw) {
        Ok(request) => Ok(request),
        Err(RequestInputError::Parse(error)) => Err(error).context("parse request JSON"),
        Err(RequestInputError::Validate(error)) => Err(error).context("validate request"),
    }
}

/// Validates a decision request JSON file against the canonical schema and domain constraints.
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
    let req = match parse_and_validate(&raw) {
        Ok(request) => request,
        Err(RequestInputError::Parse(error)) => {
            return Err(error).with_context(|| format!("parse {}", file.display()));
        }
        Err(RequestInputError::Validate(error)) => {
            return Err(error).with_context(|| format!("validate {}", file.display()));
        }
    };
    println!(
        "{} ✓ (model={}, {} questions)",
        file.display(),
        req.model,
        req.questions.len()
    );
    Ok(())
}
