use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Schema identifier written to every `RUN.json`.
pub const NATIVE_RUN_SCHEMA: &str = "openkind-native-run/v1";
/// Schema identifier written to every `checksums.json`.
pub const CHECKSUMS_SCHEMA: &str = "openkind-native-run-checksums/v1";

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
