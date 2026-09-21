use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::records::*;
use super::time::{format_utc_timestamp, unix_now};

/// Incremental builder for one `opendecision-native-run/v1` directory.
#[derive(Debug)]
pub struct NativeRunWriter {
    directory: PathBuf,
    run: RunRecord,
    profile: Option<ProfileRecord>,
    parity: Option<serde_json::Value>,
    performance: Option<serde_json::Value>,
    memory: Option<serde_json::Value>,
    predictions: Vec<serde_json::Value>,
}

impl NativeRunWriter {
    /// Begin a run under `root` in a new `<root>/<run_id>/` directory.
    ///
    /// # Errors
    /// Returns [`EvidenceError::InvalidRunId`] for run IDs outside
    /// `[A-Za-z0-9._-]` (path traversal is rejected, not sanitized) and
    /// [`EvidenceError::Io`] if the directory cannot be created.
    pub fn begin(
        root: impl AsRef<Path>,
        run_id: impl Into<String>,
        invocation: SanitizedInvocation,
        environment: RunEnvironment,
    ) -> Result<Self, EvidenceError> {
        let run_id = run_id.into();
        if run_id.is_empty()
            || !run_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
        {
            return Err(EvidenceError::InvalidRunId(run_id));
        }
        let directory = root.as_ref().join(&run_id);
        fs::create_dir_all(&directory).map_err(|source| EvidenceError::Io {
            path: directory.clone(),
            source,
        })?;
        Ok(Self {
            directory,
            run: RunRecord {
                schema: NATIVE_RUN_SCHEMA.to_owned(),
                run_id,
                started_utc: format_utc_timestamp(unix_now()),
                finished_utc: String::new(),
                invocation,
                environment,
                contains_input_content: false,
                contains_sensitive_paths: false,
            },
            profile: None,
            parity: None,
            performance: None,
            memory: None,
            predictions: Vec::new(),
        })
    }

    /// Attach the profile/backend/execution identity.
    #[must_use]
    pub fn profile(mut self, record: ProfileRecord) -> Self {
        self.profile = Some(record);
        self
    }

    /// Attach a parity report (`PARITY.json`).
    #[must_use]
    pub fn parity(mut self, report: serde_json::Value) -> Self {
        self.parity = Some(report);
        self
    }

    /// Attach a performance report (`PERFORMANCE.json`).
    #[must_use]
    pub fn performance(mut self, report: serde_json::Value) -> Self {
        self.performance = Some(report);
        self
    }

    /// Attach a memory report (`MEMORY.json`).
    #[must_use]
    pub fn memory(mut self, report: serde_json::Value) -> Self {
        self.memory = Some(report);
        self
    }

    /// Attach row-level outputs (`predictions.jsonl`), one JSON object per
    /// line. Only fixture or synthetic benchmark rows belong here.
    #[must_use]
    pub fn predictions(mut self, rows: Vec<serde_json::Value>) -> Self {
        self.predictions = rows;
        self
    }

    /// Declare whether any artifact row records input document content.
    #[must_use]
    pub fn contains_input_content(mut self, yes: bool) -> Self {
        self.run.contains_input_content = yes;
        self
    }

    /// Declare whether any artifact records potentially sensitive paths.
    #[must_use]
    pub fn contains_sensitive_paths(mut self, yes: bool) -> Self {
        self.run.contains_sensitive_paths = yes;
        self
    }

    /// Write every provided artifact plus `RUN.json` and `checksums.json`.
    /// Returns the run directory.
    ///
    /// Files are written to a temporary sibling and renamed into place, so a
    /// half-written artifact is never visible under its final name.
    ///
    /// # Errors
    /// Returns [`EvidenceError`] on any write or serialization failure.
    pub fn finish(mut self) -> Result<PathBuf, EvidenceError> {
        self.run.finished_utc = format_utc_timestamp(unix_now());
        write_json(&self.directory, "RUN.json", &self.run)?;
        if let Some(profile) = &self.profile {
            write_json(&self.directory, "PROFILE.json", profile)?;
        }
        if let Some(parity) = &self.parity {
            write_json(&self.directory, "PARITY.json", parity)?;
        }
        if let Some(performance) = &self.performance {
            write_json(&self.directory, "PERFORMANCE.json", performance)?;
        }
        if let Some(memory) = &self.memory {
            write_json(&self.directory, "MEMORY.json", memory)?;
        }
        if !self.predictions.is_empty() {
            let mut body = String::new();
            for row in &self.predictions {
                body.push_str(&serde_json::to_string(row)?);
                body.push('\n');
            }
            write_bytes(&self.directory, "predictions.jsonl", body.as_bytes())?;
        }

        let mut files = BTreeMap::new();
        for name in [
            "RUN.json",
            "PROFILE.json",
            "PARITY.json",
            "PERFORMANCE.json",
            "MEMORY.json",
            "predictions.jsonl",
        ] {
            let path = self.directory.join(name);
            if let Ok(bytes) = fs::read(&path) {
                files.insert(name.to_owned(), sha256_hex(&bytes));
            }
        }
        let checksums = ChecksumsRecord {
            schema: CHECKSUMS_SCHEMA.to_owned(),
            files,
        };
        write_json(&self.directory, "checksums.json", &checksums)?;
        Ok(self.directory)
    }
}

pub(crate) fn write_json<T: Serialize>(
    directory: &Path,
    name: &str,
    record: &T,
) -> Result<(), EvidenceError> {
    let body = serde_json::to_string_pretty(record)?;
    write_bytes(directory, name, body.as_bytes())
}

pub(crate) fn write_bytes(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), EvidenceError> {
    let path = directory.join(name);
    let temporary = directory.join(format!("{name}.tmp"));
    fs::write(&temporary, bytes).map_err(|source| EvidenceError::Io {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, &path).map_err(|source| EvidenceError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}
