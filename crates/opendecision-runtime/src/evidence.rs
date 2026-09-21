//! Backend-neutral native-run evidence artifacts.
//!
//! One schema, `opendecision-native-run/v1`, records what a native execution
//! harness ran and produced: sanitized invocation, machine environment,
//! profile/backend/execution identity, optional parity/performance/memory
//! reports, row-level outputs, and checksums over every sibling file. The
//! writer lives in the runtime — not in any model backend — so the Candle CPU
//! path and a future accelerated backend emit byte-comparable evidence from
//! the same schema and can be compared mechanically instead of through
//! ad-hoc benchmark formats.
//!
//! Sanitization is part of the contract: the invocation record is structured
//! (harness name plus explicitly non-sensitive parameters), never raw `argv`,
//! and `contains_input_content` / `contains_sensitive_paths` flags on
//! [`RunRecord`] state whether anything input- or path-derived was written.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Schema identifier written to every `RUN.json`.
pub const NATIVE_RUN_SCHEMA: &str = "opendecision-native-run/v1";
/// Schema identifier written to every `checksums.json`.
pub const CHECKSUMS_SCHEMA: &str = "opendecision-native-run-checksums/v1";

/// Errors raised while writing an evidence directory.
#[derive(Debug, Error)]
pub enum EvidenceError {
    /// A file could not be written.
    #[error("failed to write evidence file `{path}`: {source}")]
    Io {
        /// Evidence file path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// A record could not be serialized.
    #[error("failed to serialize evidence record: {0}")]
    Json(#[from] serde_json::Error),
    /// The run ID contains characters outside `[A-Za-z0-9._-]`.
    #[error("invalid run ID `{0}`: only [A-Za-z0-9._-] is allowed")]
    InvalidRunId(String),
}

/// Sanitized description of how a harness was invoked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SanitizedInvocation {
    /// Harness identifier, e.g. `qwen35_model_stress`.
    pub harness: String,
    /// Binary or example that ran, e.g. `qwen35_scheduler_stress`.
    pub command: String,
    /// Explicitly non-sensitive named parameters (counts, strategy names,
    /// fixture digests). Never raw `argv`.
    pub parameters: serde_json::Value,
    /// Always `false`: raw argv is deliberately not recorded.
    pub raw_argv_recorded: bool,
}

/// Machine and toolchain environment of one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEnvironment {
    /// Commit of the working repository, if `git` is available.
    pub git_commit: Option<String>,
    /// Whether the working tree had uncommitted changes.
    pub git_dirty: Option<bool>,
    /// `rustc --version` output.
    pub rustc_version: Option<String>,
    /// Operating system, e.g. `macos`.
    pub os: String,
    /// CPU architecture, e.g. `aarch64`.
    pub arch: String,
}

impl RunEnvironment {
    /// Capture environment evidence from the local machine.
    ///
    /// Runs read-only `git` and `rustc` queries; each degrades to `None`
    /// instead of failing the run. Harness unit tests inject values rather
    /// than capture.
    #[must_use]
    pub fn capture() -> Self {
        let output = |command: &mut Command| {
            command
                .output()
                .ok()
                .filter(|result| result.status.success())
                .map(|result| String::from_utf8_lossy(&result.stdout).trim().to_owned())
        };
        let git_commit =
            output(Command::new("git").args(["rev-parse", "HEAD"])).filter(|s| !s.is_empty());
        let git_dirty = Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .filter(|result| result.status.success())
            .map(|result| !result.stdout.is_empty());
        let rustc_version = output(Command::new("rustc").arg("--version"));
        Self {
            git_commit,
            git_dirty,
            rustc_version,
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
        }
    }
}

/// `RUN.json`: identifies one harness invocation and its environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    /// Always [`NATIVE_RUN_SCHEMA`].
    pub schema: String,
    /// UTC run identifier, also the directory name.
    pub run_id: String,
    /// ISO-8601 UTC start time.
    pub started_utc: String,
    /// ISO-8601 UTC finish time.
    pub finished_utc: String,
    /// Sanitized invocation.
    pub invocation: SanitizedInvocation,
    /// Machine and toolchain environment.
    pub environment: RunEnvironment,
    /// Whether any artifact row records input document content (as opposed to
    /// digests, counts, and synthetic fixtures).
    pub contains_input_content: bool,
    /// Whether any artifact records filesystem paths that could be sensitive.
    pub contains_sensitive_paths: bool,
}

/// `PROFILE.json` backend block: concrete implementation identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendRecord {
    /// Backend identifier, e.g. `qwen35-native-cpu`.
    pub backend_id: String,
    /// Backend implementation version/commit, e.g. `candle-core 0.8.0` today
    /// or an MLX-LM commit for an accelerated backend.
    pub backend_version: String,
    /// Allocator configuration descriptor.
    pub allocator: String,
    /// Advertised compute capabilities, backend-supplied.
    pub capabilities: serde_json::Value,
}

/// `PROFILE.json` execution block: plan and physical compute mode.
///
/// `execution_plan` (state topology) and `batch_forward_mode` (how lanes
/// physically advanced) are siblings on purpose: `nested_batched` +
/// `per_lane` and `nested_batched` + `vectorized` are different physical
/// execution graphs and must not be conflated when comparing backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Selected execution plan, e.g. `nested_sequential`.
    pub execution_plan: String,
    /// Physical compute mode: `per_lane` or `vectorized`.
    pub batch_forward_mode: String,
    /// Whether the plan came from a diagnostic override.
    pub forced: bool,
    /// Backend-supplied scheduler configuration summary.
    pub scheduler: serde_json::Value,
}

/// `PROFILE.json`: the profile, backend, and execution identity behind a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileRecord {
    /// Selected model/execution profile ID.
    pub profile_id: String,
    /// SHA-256 of the selected reference bundle.
    pub bundle_sha256: String,
    /// Base model ID.
    pub backbone_id: String,
    /// Base model revision.
    pub backbone_revision: String,
    /// Renderer identity, including ordering semantics.
    pub renderer_id: String,
    /// SHA-256 of the exported tokenizer artifact.
    pub tokenizer_digest: String,
    /// Arithmetic/device identity of the execution path.
    pub arithmetic_id: String,
    /// Calibration temperature.
    pub calibration_temperature: f64,
    /// Application-policy threshold.
    pub policy_threshold: f64,
    /// Declared probability space, e.g. `offered_options_plus_semantic_none`.
    pub probability_space: String,
    /// Concrete backend implementation identity.
    pub backend: BackendRecord,
    /// Plan and physical execution identity.
    pub execution: ExecutionRecord,
}

/// `checksums.json`: SHA-256 over every sibling artifact file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChecksumsRecord {
    /// Always [`CHECKSUMS_SCHEMA`].
    pub schema: String,
    /// File name to lowercase-hex SHA-256, sorted.
    pub files: BTreeMap<String, String>,
}

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

fn write_json<T: Serialize>(directory: &Path, name: &str, record: &T) -> Result<(), EvidenceError> {
    let body = serde_json::to_string_pretty(record)?;
    write_bytes(directory, name, body.as_bytes())
}

fn write_bytes(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), EvidenceError> {
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Format Unix seconds as ISO-8601 UTC, e.g. `2026-09-20T15:22:06Z`.
#[must_use]
pub fn format_utc_timestamp(unix_seconds: u64) -> String {
    let (year, month, day, hour, minute, second) = civil_from_unix(unix_seconds);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Generate a UTC run identifier, e.g. `20260920T152206Z`.
#[must_use]
pub fn generate_run_id() -> String {
    generate_run_id_at(unix_now())
}

fn generate_run_id_at(unix_seconds: u64) -> String {
    let (year, month, day, hour, minute, second) = civil_from_unix(unix_seconds);
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

/// Civil date-time from Unix seconds (Howard Hinnant's `civil_from_days`).
fn civil_from_unix(unix_seconds: u64) -> (u64, u64, u64, u64, u64, u64) {
    let days = (unix_seconds / 86_400) as i64;
    let seconds_of_day = unix_seconds % 86_400;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    (
        year.unsigned_abs(),
        month.unsigned_abs(),
        day.unsigned_abs(),
        seconds_of_day / 3_600,
        (seconds_of_day % 3_600) / 60,
        seconds_of_day % 60,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "opendecision-evidence-{tag}-{}-{}",
            std::process::id(),
            unix_now()
        ))
    }

    fn invocation() -> SanitizedInvocation {
        SanitizedInvocation {
            harness: "qwen35_scheduler_stress".to_owned(),
            command: "qwen35_scheduler_stress".to_owned(),
            parameters: serde_json::json!({"k_cells": [32, 64, 128, 255]}),
            raw_argv_recorded: false,
        }
    }

    fn environment() -> RunEnvironment {
        RunEnvironment {
            git_commit: Some("35c481a".to_owned()),
            git_dirty: Some(false),
            rustc_version: Some("rustc 1.98.0".to_owned()),
            os: "macos".to_owned(),
            arch: "aarch64".to_owned(),
        }
    }

    #[test]
    fn writer_produces_run_profile_and_verifiable_checksums() {
        let root = temp_root("full");
        let profile = ProfileRecord {
            profile_id: "a047d".to_owned(),
            bundle_sha256: "0".repeat(64),
            backbone_id: "Qwen/Qwen3.5-4B-Base".to_owned(),
            backbone_revision: "1001bb4".to_owned(),
            renderer_id: "state_first".to_owned(),
            tokenizer_digest: "1".repeat(64),
            arithmetic_id: "candle-cpu-fp32".to_owned(),
            calibration_temperature: 1.818_679_991_044_277_7,
            policy_threshold: 0.98,
            probability_space: "offered_options_plus_semantic_none".to_owned(),
            backend: BackendRecord {
                backend_id: "qwen35-native-cpu".to_owned(),
                backend_version: "candle-core 0.8.0".to_owned(),
                allocator: "system".to_owned(),
                capabilities: serde_json::json!({
                    "vectorized_question_forward": false,
                    "vectorized_candidate_forward": false
                }),
            },
            execution: ExecutionRecord {
                execution_plan: "nested_sequential".to_owned(),
                batch_forward_mode: "per_lane".to_owned(),
                forced: false,
                scheduler: serde_json::json!({"min_shared_savings_ratio": 2.52}),
            },
        };

        let directory =
            NativeRunWriter::begin(&root, "20260920T000000Z", invocation(), environment())
                .expect("begin")
                .profile(profile)
                .parity(serde_json::json!({"passed": true}))
                .predictions(vec![serde_json::json!({"k": 32, "admitted": true})])
                .finish()
                .expect("finish");

        assert_eq!(directory, root.join("20260920T000000Z"));
        let run: RunRecord =
            serde_json::from_str(&fs::read_to_string(directory.join("RUN.json")).unwrap()).unwrap();
        assert_eq!(run.schema, NATIVE_RUN_SCHEMA);
        assert!(!run.invocation.raw_argv_recorded);
        assert!(!run.contains_input_content);
        assert!(!run.contains_sensitive_paths);
        assert!(run.finished_utc.ends_with('Z'));
        assert!(directory.join("PROFILE.json").exists());
        assert!(directory.join("PARITY.json").exists());
        assert!(!directory.join("PERFORMANCE.json").exists());
        let rows = fs::read_to_string(directory.join("predictions.jsonl")).unwrap();
        assert_eq!(rows.lines().count(), 1);

        let checksums: ChecksumsRecord =
            serde_json::from_str(&fs::read_to_string(directory.join("checksums.json")).unwrap())
                .unwrap();
        assert_eq!(checksums.schema, CHECKSUMS_SCHEMA);
        for (name, expected) in &checksums.files {
            let bytes = fs::read(directory.join(name)).unwrap();
            assert_eq!(&sha256_hex(&bytes), expected, "checksum matches {name}");
        }
        assert!(checksums.files.contains_key("RUN.json"));
        assert!(checksums.files.contains_key("predictions.jsonl"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn writer_rejects_traversal_run_ids() {
        let root = temp_root("traversal");
        assert!(matches!(
            NativeRunWriter::begin(&root, "../escape", invocation(), environment()),
            Err(EvidenceError::InvalidRunId(_))
        ));
        assert!(matches!(
            NativeRunWriter::begin(&root, "run/with/slashes", invocation(), environment()),
            Err(EvidenceError::InvalidRunId(_))
        ));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn utc_formatting_covers_epoch_and_leap_days() {
        assert_eq!(format_utc_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_utc_timestamp(86_400), "1970-01-02T00:00:00Z");
        assert_eq!(format_utc_timestamp(951_782_400), "2000-02-29T00:00:00Z");
        // 2026-09-20T00:00:00Z in Unix seconds (documentation baseline date).
        assert_eq!(format_utc_timestamp(1_789_862_400), "2026-09-20T00:00:00Z");
        assert_eq!(generate_run_id_at(1_789_862_400), "20260920T000000Z");
        let run_id = generate_run_id();
        assert!(run_id.starts_with("20") && run_id.ends_with('Z') && run_id.len() == 16);
    }
}
