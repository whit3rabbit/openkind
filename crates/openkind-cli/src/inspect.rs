use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use openkind_core::{validate_request, SystemRequest, ValidationError};

/// Maximum allowed input file size (32 MB) to prevent local memory exhaustion.
pub const MAX_CLI_INPUT_BYTES: u64 = 32 * 1024 * 1024;

pub(crate) fn read_bounded_input(reader: impl Read, limit: u64) -> Result<String> {
    let mut bytes = Vec::new();
    // One extra byte distinguishes an exact-size document from a truncated
    // valid prefix, including stdin and files that grow after metadata checks.
    reader
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        anyhow::bail!("input exceeds maximum allowed size (limit {limit} bytes)");
    }
    String::from_utf8(bytes).context("input must be UTF-8")
}

pub(crate) fn read_request_file(file: &Path) -> Result<String> {
    let reader = std::fs::File::open(file).with_context(|| format!("read {}", file.display()))?;
    let meta = reader
        .metadata()
        .with_context(|| format!("stat {}", file.display()))?;
    if meta.len() > MAX_CLI_INPUT_BYTES {
        anyhow::bail!(
            "file {} exceeds maximum allowed size ({} bytes, limit {} bytes)",
            file.display(),
            meta.len(),
            MAX_CLI_INPUT_BYTES
        );
    }
    read_bounded_input(reader, MAX_CLI_INPUT_BYTES)
        .with_context(|| format!("read {}", file.display()))
}

enum RequestInputError {
    Parse(serde_json::Error),
    Validate(ValidationError),
}

#[inline]
fn parse_and_validate(raw: &str) -> Result<SystemRequest, RequestInputError> {
    // sonic-rs parses the same serde data model measurably faster for
    // CLI-sized documents; on any parse failure re-parse with serde_json
    // so error text stays byte-identical with the wire-standard parser.
    let request = match sonic_rs::from_str(raw) {
        Ok(request) => request,
        Err(_) => serde_json::from_str(raw).map_err(RequestInputError::Parse)?,
    };
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
    let raw = read_request_file(&file)?;
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
        crate::output::terminal_safe(&req.model),
        req.questions.len()
    );
    Ok(())
}
