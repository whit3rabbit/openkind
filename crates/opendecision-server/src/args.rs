use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use opendecision_runtime::ExecutionPlan;

#[derive(Parser, Debug)]
#[command(name = "opendecisiond", about = "opendecision inference daemon")]
pub(crate) struct Args {
    /// Address to bind the HTTP server on.
    #[arg(long, env = "OPENDECISION_HTTP_ADDR")]
    pub(crate) http_addr: Option<SocketAddr>,

    /// Deprecated pre-rename HTTP listener environment variable.
    #[arg(long, env = "OPENPICK_HTTP_ADDR", hide = true)]
    pub(crate) legacy_http_addr: Option<SocketAddr>,

    /// Address to bind the gRPC server on. Use `0` to disable the gRPC listener.
    #[arg(long, env = "OPENDECISION_GRPC_ADDR")]
    pub(crate) grpc_addr: Option<String>,

    /// Deprecated pre-rename gRPC listener environment variable.
    #[arg(long, env = "OPENPICK_GRPC_ADDR", hide = true)]
    pub(crate) legacy_grpc_addr: Option<String>,

    /// Comma-separated model aliases to expose. Aliases also listed in
    /// `--qwen35-aliases` use the native engine; all others use the mock.
    /// Register `jev-latest` to accept the SDK's default alias.
    #[arg(
        long,
        env = "OPENDECISION_MODELS",
        value_delimiter = ',',
        default_value = "mock,jev-latest"
    )]
    pub(crate) models: Vec<String>,

    /// Aliases in `--models` that should use the native Qwen3.5 engine.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_ALIASES",
        value_delimiter = ',',
        default_value = "qwen35-native"
    )]
    pub(crate) qwen35_aliases: Vec<String>,

    /// Offline selected-profile bundle root required by native aliases.
    #[arg(long, env = "OPENDECISION_QWEN35_BUNDLE_ROOT")]
    pub(crate) qwen35_bundle_root: Option<PathBuf>,

    /// Offline pinned Qwen checkpoint root required by native aliases.
    #[arg(long, env = "OPENDECISION_QWEN35_CHECKPOINT_ROOT")]
    pub(crate) qwen35_checkpoint_root: Option<PathBuf>,

    /// Digest-locked tokenizer JSON required by native aliases.
    #[arg(long, env = "OPENDECISION_QWEN35_TOKENIZER")]
    pub(crate) qwen35_tokenizer: Option<PathBuf>,

    /// Maximum concurrent native model evaluations.
    #[arg(long, env = "OPENDECISION_QWEN35_CONCURRENCY", default_value_t = 1)]
    pub(crate) qwen35_concurrency: usize,

    /// Additional native requests allowed to wait for execution.
    #[arg(long, env = "OPENDECISION_QWEN35_QUEUE", default_value_t = 2)]
    pub(crate) qwen35_queue: usize,

    /// Execution-plan override for diagnostics and reproducibility.
    /// Overrides adaptive scheduling only; memory and backend capability
    /// admission still apply.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_EXECUTION",
        value_enum,
        default_value_t = ExecutionArg::Auto
    )]
    pub(crate) qwen35_execution: ExecutionArg,

    /// Optional continuation tensor-payload ceiling per request.
    #[arg(long, env = "OPENDECISION_QWEN35_MAX_TENSOR_BYTES")]
    pub(crate) qwen35_max_tensor_bytes: Option<usize>,

    /// Optional process-memory ceiling for native admission.
    #[arg(long, env = "OPENDECISION_QWEN35_MAX_PROCESS_BYTES")]
    pub(crate) qwen35_max_process_bytes: Option<usize>,

    /// Forward scratch budget added to observed process memory.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_SCRATCH_BYTES",
        default_value_t = 1_073_741_824
    )]
    pub(crate) qwen35_scratch_bytes: usize,

    /// Allocator and runtime headroom added to observed process memory.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_ALLOCATOR_HEADROOM_BYTES",
        default_value_t = 536_870_912
    )]
    pub(crate) qwen35_allocator_headroom_bytes: usize,

    /// Optional bearer token required for `/v1/*`. If unset, the env
    /// `OPENDECISION_API_KEY` is consulted; if both are unset, auth is off
    /// (matches Phase 1 dev behavior).
    #[arg(long, env = "OPENDECISION_API_KEY")]
    pub(crate) api_key: Option<String>,

    /// Deprecated pre-rename API-key environment variable.
    #[arg(long, env = "OPENPICK_API_KEY", hide = true)]
    pub(crate) legacy_api_key: Option<String>,

    /// Per-client-IP request budget per minute on `/v1/*` routes.
    /// `0` disables rate limiting entirely.
    #[arg(long, env = "OPENDECISION_RATE_LIMIT_RPM", default_value_t = 120)]
    pub(crate) rate_limit_rpm: u32,

    /// Log filter. Standard `tracing_subscriber::EnvFilter` syntax.
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    pub(crate) log_filter: String,
}

/// CLI surface for `--qwen35-execution`: the three execution plans plus the
/// adaptive `auto` default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ExecutionArg {
    /// Adaptive scheduling from the measured policy (default).
    Auto,
    /// One full-sequence evaluation per candidate.
    RepeatedFull,
    /// Prefill once, advance question and candidate lanes sequentially.
    NestedSequential,
    /// Prefill once, breadth-first question/candidate state fan-out.
    NestedBatched,
}

impl From<ExecutionArg> for Option<ExecutionPlan> {
    fn from(value: ExecutionArg) -> Self {
        match value {
            ExecutionArg::Auto => None,
            ExecutionArg::RepeatedFull => Some(ExecutionPlan::RepeatedFull),
            ExecutionArg::NestedSequential => Some(ExecutionPlan::NestedSequential),
            ExecutionArg::NestedBatched => Some(ExecutionPlan::NestedBatched),
        }
    }
}

/// Resolve a renamed setting while refusing ambiguous migration configuration.
pub(crate) fn resolve_alias<T: PartialEq>(
    current: Option<T>,
    legacy: Option<T>,
    current_name: &str,
    legacy_name: &str,
) -> Result<Option<T>> {
    match (current, legacy) {
        (Some(current), Some(legacy)) if current != legacy => anyhow::bail!(
            "conflicting values for {current_name} and deprecated {legacy_name}; remove {legacy_name} after migration"
        ),
        (Some(current), _) => Ok(Some(current)),
        (None, legacy) => Ok(legacy),
    }
}

/// Parse the `--grpc-addr` value. The literal `0` (also `off`/`none`/`disabled`,
/// case-insensitive) disables the gRPC listener; anything else must be a
/// `host:port` socket address.
pub(crate) fn parse_grpc_addr(
    value: &str,
) -> std::result::Result<Option<SocketAddr>, std::net::AddrParseError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "off" | "none" | "disabled" => Ok(None),
        _ => value.trim().parse::<SocketAddr>().map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_default_values() {
        let args = Args::try_parse_from(["opendecisiond"]).unwrap();
        assert_eq!(args.http_addr, None);
        assert_eq!(args.grpc_addr, None);
        assert_eq!(args.models, vec!["mock", "jev-latest"]);
        assert_eq!(args.api_key, None);
        assert_eq!(args.rate_limit_rpm, 120);
        assert_eq!(args.log_filter, "info");
        assert_eq!(args.qwen35_execution, ExecutionArg::Auto);
    }

    #[test]
    fn args_qwen35_execution_override_parses_and_maps() {
        let forced =
            Args::try_parse_from(["opendecisiond", "--qwen35-execution", "nested-batched"])
                .unwrap();
        assert_eq!(forced.qwen35_execution, ExecutionArg::NestedBatched);
        assert_eq!(
            Option::<ExecutionPlan>::from(forced.qwen35_execution),
            Some(ExecutionPlan::NestedBatched)
        );

        for (flag, expected) in [
            ("repeated-full", ExecutionPlan::RepeatedFull),
            ("nested-sequential", ExecutionPlan::NestedSequential),
            ("nested-batched", ExecutionPlan::NestedBatched),
        ] {
            let args = Args::try_parse_from(["opendecisiond", "--qwen35-execution", flag]).unwrap();
            assert_eq!(
                Option::<ExecutionPlan>::from(args.qwen35_execution),
                Some(expected),
                "--qwen35-execution {flag}"
            );
        }

        let auto = Args::try_parse_from(["opendecisiond"]).unwrap();
        assert_eq!(Option::<ExecutionPlan>::from(auto.qwen35_execution), None);
        assert!(Args::try_parse_from(["opendecisiond", "--qwen35-execution", "turbo"]).is_err());
    }

    #[test]
    fn args_custom_values() {
        let args = Args::try_parse_from([
            "opendecisiond",
            "--http-addr",
            "127.0.0.1:18080",
            "--grpc-addr",
            "127.0.0.1:19090",
            "--models",
            "mock,candle,jev-latest",
            "--api-key",
            "secret-token",
            "--rate-limit-rpm",
            "0",
            "--log-filter",
            "debug",
        ])
        .unwrap();

        assert_eq!(args.http_addr, Some("127.0.0.1:18080".parse().unwrap()));
        assert_eq!(args.grpc_addr.as_deref(), Some("127.0.0.1:19090"));
        assert_eq!(args.models, vec!["mock", "candle", "jev-latest"]);
        assert_eq!(args.api_key.as_deref(), Some("secret-token"));
        assert_eq!(args.rate_limit_rpm, 0);
        assert_eq!(args.log_filter, "debug");
    }

    #[test]
    fn parse_grpc_addr_disable_sentinels() {
        assert_eq!(parse_grpc_addr("0").unwrap(), None);
        assert_eq!(parse_grpc_addr(" off ").unwrap(), None);
        assert_eq!(parse_grpc_addr("None").unwrap(), None);
        assert_eq!(parse_grpc_addr("disabled").unwrap(), None);
        assert_eq!(parse_grpc_addr("  DISABLED  ").unwrap(), None);
        assert_eq!(
            parse_grpc_addr("127.0.0.1:19090").unwrap(),
            Some("127.0.0.1:19090".parse().unwrap())
        );
        assert!(parse_grpc_addr("not-an-addr").is_err());
        assert!(parse_grpc_addr("").is_err());
    }

    #[test]
    fn renamed_setting_uses_legacy_fallback() {
        assert_eq!(
            resolve_alias(None, Some("legacy"), "NEW", "OLD").unwrap(),
            Some("legacy")
        );
        assert_eq!(
            resolve_alias(Some("same"), Some("same"), "NEW", "OLD").unwrap(),
            Some("same")
        );
    }

    #[test]
    fn renamed_setting_rejects_conflicting_values() {
        let error = resolve_alias(Some("new"), Some("old"), "NEW", "OLD").unwrap_err();
        assert!(error.to_string().contains("conflicting values for NEW"));
    }
}
