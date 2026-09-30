use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use openkind_runtime::ExecutionPlan;

use crate::families::FamilyArgs;

#[derive(Parser, Debug)]
#[command(name = "openkindd", about = "openkind inference daemon")]
pub(crate) struct Args {
    /// Surveyed-family engine configuration (aliases and artifact paths).
    #[command(flatten)]
    pub(crate) family_args: FamilyArgs,

    /// Address to bind the HTTP server on.
    #[arg(long, env = "OPENKIND_HTTP_ADDR")]
    pub(crate) http_addr: Option<SocketAddr>,

    /// Deprecated pre-rename HTTP listener environment variable.
    #[arg(long, env = "OPENPICK_HTTP_ADDR", hide = true)]
    pub(crate) legacy_http_addr: Option<SocketAddr>,

    /// Address to bind the gRPC server on. Use `0` to disable the gRPC listener.
    #[arg(long, env = "OPENKIND_GRPC_ADDR")]
    pub(crate) grpc_addr: Option<String>,

    /// Deprecated pre-rename gRPC listener environment variable.
    #[arg(long, env = "OPENPICK_GRPC_ADDR", hide = true)]
    pub(crate) legacy_grpc_addr: Option<String>,

    /// Comma-separated model aliases to expose. Aliases also listed in
    /// `--qwen35-aliases` use the native engine; all others use the mock.
    /// Register `jev-latest` to accept the SDK's default alias.
    #[arg(
        long,
        env = "OPENKIND_MODELS",
        value_delimiter = ',',
        default_value = "mock,jev-latest"
    )]
    pub(crate) models: Vec<String>,

    /// Installed profile names loaded explicitly at startup.
    #[arg(long, env = "OPENKIND_INSTALLED_MODELS", value_delimiter = ',')]
    pub(crate) installed_models: Vec<String>,

    /// Shared model store directory.
    #[arg(long, env = "OPENKIND_MODELS_DIR")]
    pub(crate) models_dir: Option<PathBuf>,

    /// Aliases in `--models` that should use the native Qwen3.5 engine.
    #[arg(
        long,
        env = "OPENKIND_QWEN35_ALIASES",
        value_delimiter = ',',
        default_value = "qwen35-native"
    )]
    pub(crate) qwen35_aliases: Vec<String>,

    /// Offline selected-profile bundle root required by native aliases.
    #[arg(long, env = "OPENKIND_QWEN35_BUNDLE_ROOT")]
    pub(crate) qwen35_bundle_root: Option<PathBuf>,

    /// Offline pinned Qwen checkpoint root required by native aliases.
    #[arg(long, env = "OPENKIND_QWEN35_CHECKPOINT_ROOT")]
    pub(crate) qwen35_checkpoint_root: Option<PathBuf>,

    /// Digest-locked tokenizer JSON required by native aliases.
    #[arg(long, env = "OPENKIND_QWEN35_TOKENIZER")]
    pub(crate) qwen35_tokenizer: Option<PathBuf>,

    /// Maximum concurrent native model evaluations.
    #[arg(long, env = "OPENKIND_QWEN35_CONCURRENCY", default_value_t = 1)]
    pub(crate) qwen35_concurrency: usize,

    /// Additional native requests allowed to wait for execution.
    #[arg(long, env = "OPENKIND_QWEN35_QUEUE", default_value_t = 2)]
    pub(crate) qwen35_queue: usize,

    /// Queue-inclusive deadline for one native evaluation, in milliseconds.
    #[arg(long, env = "OPENKIND_QWEN35_TIMEOUT_MS", default_value_t = 600_000)]
    pub(crate) qwen35_timeout_ms: u64,

    /// Native model backend. `mlx-fp32` is available on macOS arm64 with the
    /// daemon's `mlx` feature enabled.
    #[arg(
        long,
        env = "OPENKIND_QWEN35_BACKEND",
        value_enum,
        default_value_t = Qwen35BackendArg::NativeCpu
    )]
    pub(crate) qwen35_backend: Qwen35BackendArg,

    /// Execution-plan override for diagnostics and reproducibility.
    /// Overrides adaptive scheduling only; memory and backend capability
    /// admission still apply.
    #[arg(
        long,
        env = "OPENKIND_QWEN35_EXECUTION",
        value_enum,
        default_value_t = ExecutionArg::Auto
    )]
    pub(crate) qwen35_execution: ExecutionArg,

    /// Optional continuation tensor-payload ceiling per request.
    #[arg(long, env = "OPENKIND_QWEN35_MAX_TENSOR_BYTES")]
    pub(crate) qwen35_max_tensor_bytes: Option<usize>,

    /// Optional process-memory ceiling for native admission.
    #[arg(long, env = "OPENKIND_QWEN35_MAX_PROCESS_BYTES")]
    pub(crate) qwen35_max_process_bytes: Option<usize>,

    /// Forward scratch budget added to observed process memory.
    #[arg(
        long,
        env = "OPENKIND_QWEN35_SCRATCH_BYTES",
        default_value_t = 1_073_741_824
    )]
    pub(crate) qwen35_scratch_bytes: usize,

    /// Allocator and runtime headroom added to observed process memory.
    #[arg(
        long,
        env = "OPENKIND_QWEN35_ALLOCATOR_HEADROOM_BYTES",
        default_value_t = 536_870_912
    )]
    pub(crate) qwen35_allocator_headroom_bytes: usize,

    /// Optional bearer token required for `/v1/*`. If unset, the env
    /// `OPENKIND_API_KEY` is consulted; if both are unset, auth is off
    /// (matches Phase 1 dev behavior).
    #[arg(long, env = "OPENKIND_API_KEY")]
    pub(crate) api_key: Option<String>,

    /// Deprecated pre-rename API-key environment variable.
    #[arg(long, env = "OPENPICK_API_KEY", hide = true)]
    pub(crate) legacy_api_key: Option<String>,

    /// Per-client-IP request budget per minute on `/v1/*` routes.
    /// `0` disables rate limiting entirely.
    #[arg(long, env = "OPENKIND_RATE_LIMIT_RPM", default_value_t = 120)]
    pub(crate) rate_limit_rpm: u32,

    /// Serve the embedded playground and authenticated local model controls.
    #[arg(
        long,
        env = "OPENKIND_PLAYGROUND",
        value_enum,
        default_value_t = PlaygroundArg::Off
    )]
    pub(crate) playground: PlaygroundArg,

    /// Serve the unofficial bulk Arrow IPC endpoint (`POST /v1/arrow`).
    /// Outside the TypeSafe wire contract; see `docs/ARROW.md`.
    #[arg(
        long,
        env = "OPENKIND_ARROW",
        value_enum,
        default_value_t = ArrowArg::Off
    )]
    pub(crate) arrow: ArrowArg,

    /// Upstream Jev-compatible API base URL. Setting this enables proxy
    /// cache mode: proxied aliases are answered from the distilling cache
    /// when confident and forwarded upstream otherwise.
    #[arg(long, env = "OPENKIND_PROXY_CACHE_UPSTREAM")]
    pub(crate) proxy_cache_upstream: Option<String>,

    /// Model aliases the proxy cache intercepts. Aliases listed here are
    /// served by the cache (or forwarded upstream), not by local engines.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_MODELS",
        value_delimiter = ',',
        default_value = "jev-latest"
    )]
    pub(crate) proxy_cache_models: Vec<String>,

    /// Embedding model for the proxy cache. `hash` needs no weights;
    /// anything else is an `openkind pull` name (regular or MLX profile).
    #[arg(long, env = "OPENKIND_PROXY_CACHE_ENCODER", default_value = "hash")]
    pub(crate) proxy_cache_encoder: String,

    /// Proxy-cache encoder backend. `mlx-fp32` requires the daemon's `mlx`
    /// feature on macOS arm64.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_ENCODER_BACKEND",
        value_enum,
        default_value_t = ProxyCacheEncoderBackendArg::Cpu
    )]
    pub(crate) proxy_cache_encoder_backend: ProxyCacheEncoderBackendArg,

    /// Proxy-cache state directory (task stores, student versions, key salt).
    #[arg(long, env = "OPENKIND_PROXY_CACHE_DATA_DIR")]
    pub(crate) proxy_cache_data_dir: Option<PathBuf>,

    /// Bearer key used for upstream calls instead of the caller's own key.
    #[arg(long, env = "OPENKIND_PROXY_CACHE_UPSTREAM_KEY")]
    pub(crate) proxy_cache_upstream_key: Option<String>,

    /// Per-attempt upstream timeout in milliseconds.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_UPSTREAM_TIMEOUT_MS",
        default_value_t = 9_000
    )]
    pub(crate) proxy_cache_upstream_timeout_ms: u64,

    /// Target agreement for every task: the cache tolerates at most
    /// `1 - agreement` probability mass of answered-and-disagreed.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_TARGET_AGREEMENT",
        default_value_t = 0.98
    )]
    pub(crate) proxy_cache_target_agreement: f64,

    /// Store request text in the cache's training rows. Disable to keep only
    /// salted hashes and embeddings.
    #[arg(long, env = "OPENKIND_PROXY_CACHE_STORE_TEXT", default_value_t = true)]
    pub(crate) proxy_cache_store_text: bool,

    /// Requests a new task must observe before an engine is created.
    #[arg(long, env = "OPENKIND_PROXY_CACHE_ADMISSION_MIN", default_value_t = 50)]
    pub(crate) proxy_cache_admission_min: usize,

    /// Teacher-labelled train rows required before the first fit.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_MIN_TRAIN_SAMPLES",
        default_value_t = 1000
    )]
    pub(crate) proxy_cache_min_train_samples: usize,

    /// Teacher-labelled calibration rows required before the first fit.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_MIN_CALIB_SAMPLES",
        default_value_t = 500
    )]
    pub(crate) proxy_cache_min_calib_samples: usize,

    /// Shadow observations required before a candidate is judged.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_SHADOW_MIN_SAMPLES",
        default_value_t = 1000
    )]
    pub(crate) proxy_cache_shadow_min_samples: usize,

    /// Fraction of teacher-answered requests reserved for calibration.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_CALIB_FRACTION",
        default_value_t = 0.2
    )]
    pub(crate) proxy_cache_calib_fraction: f64,

    /// New teacher answers that trigger a retrain of the production student.
    #[arg(
        long,
        env = "OPENKIND_PROXY_CACHE_MIN_NEW_SAMPLES",
        default_value_t = 2000
    )]
    pub(crate) proxy_cache_min_new_samples: usize,

    /// Log filter. Standard `tracing_subscriber::EnvFilter` syntax.
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    pub(crate) log_filter: String,
}

/// CLI surface for `--playground`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum PlaygroundArg {
    /// Serve the playground UI and explicit model load/unload controls.
    On,
    /// Do not serve the playground route.
    Off,
}

/// CLI surface for `--arrow`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ArrowArg {
    /// Serve the unofficial `POST /v1/arrow` bulk Arrow IPC endpoint.
    On,
    /// Do not serve the Arrow endpoint.
    Off,
}

/// Proxy-cache encoder backend choices exposed by the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ProxyCacheEncoderBackendArg {
    /// Candle FP32 CPU reference backend.
    Cpu,
    /// MLX FP32 backend on macOS arm64 when the optional feature is enabled.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxFp32,
}

/// Native backbone choices exposed by the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Qwen35BackendArg {
    /// Candle FP32 CPU reference backend.
    NativeCpu,
    /// MLX FP32 backend on macOS arm64 when the optional feature is enabled.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxFp32,
}

/// Laya decision-encoder backend choices exposed by the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum LayaBackendArg {
    /// Candle FP32 CPU reference backend.
    NativeCpu,
    /// MLX FP32 backend on macOS arm64 when the optional feature is enabled.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxFp32,
}

/// Encoder-instruct-label backend choices exposed by the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum EncoderInstructLabelBackendArg {
    /// Candle FP32 CPU reference backend.
    NativeCpu,
    /// MLX FP32 backend on macOS arm64 when the optional feature is enabled.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxFp32,
}

/// Decoder-logit-qwen35 backend choices exposed by the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DecoderLogitQwen35BackendArg {
    /// Candle FP32 CPU reference backend.
    NativeCpu,
    /// MLX FP32 backend on macOS arm64 when the optional feature is enabled.
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    MlxFp32,
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
    use clap::CommandFactory;
    use std::ffi::OsStr;

    #[test]
    fn args_default_values() {
        let args = Args::try_parse_from(["openkindd"]).unwrap();
        assert_eq!(args.http_addr, None);
        assert_eq!(args.grpc_addr, None);
        assert_eq!(args.models, vec!["mock", "jev-latest"]);
        assert!(args.installed_models.is_empty());
        assert_eq!(args.api_key, None);
        assert_eq!(args.rate_limit_rpm, 120);
        assert_eq!(args.playground, PlaygroundArg::Off);
        assert_eq!(args.arrow, ArrowArg::Off);
        assert_eq!(args.log_filter, "info");
        assert_eq!(args.qwen35_execution, ExecutionArg::Auto);
        assert_eq!(args.qwen35_backend, Qwen35BackendArg::NativeCpu);
        assert_eq!(args.qwen35_timeout_ms, 600_000);
    }

    #[test]
    fn playground_flag_parses_and_environment_alias_is_declared() {
        let args = Args::try_parse_from(["openkindd", "--playground", "on"]).unwrap();
        assert_eq!(args.playground, PlaygroundArg::On);

        let off = Args::try_parse_from(["openkindd", "--playground", "off"]).unwrap();
        assert_eq!(off.playground, PlaygroundArg::Off);

        let error = Args::try_parse_from(["openkindd", "--playground", "maybe"])
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid value 'maybe'"), "{error}");

        let command = Args::command();
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_long() == Some("playground"))
            .expect("--playground argument");
        assert_eq!(arg.get_env(), Some(OsStr::new("OPENKIND_PLAYGROUND")));
    }

    #[test]
    fn arrow_flag_parses_and_environment_alias_is_declared() {
        let args = Args::try_parse_from(["openkindd", "--arrow", "on"]).unwrap();
        assert_eq!(args.arrow, ArrowArg::On);

        let off = Args::try_parse_from(["openkindd", "--arrow", "off"]).unwrap();
        assert_eq!(off.arrow, ArrowArg::Off);

        let command = Args::command();
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_long() == Some("arrow"))
            .expect("--arrow argument");
        assert_eq!(arg.get_env(), Some(OsStr::new("OPENKIND_ARROW")));
    }

    #[test]
    fn installed_models_are_explicit_and_have_a_shared_store_root() {
        let args = Args::try_parse_from([
            "openkindd",
            "--installed-models",
            "fixture:v1,other:v2",
            "--models-dir",
            "/tmp/openkind-model-test",
        ])
        .unwrap();
        assert_eq!(args.installed_models, vec!["fixture:v1", "other:v2"]);
        assert_eq!(
            args.models_dir.unwrap(),
            PathBuf::from("/tmp/openkind-model-test")
        );
    }

    #[test]
    fn qwen35_backend_cli_values_and_diagnostics() {
        let args = Args::try_parse_from(["openkindd", "--qwen35-backend", "native-cpu"]).unwrap();
        assert_eq!(args.qwen35_backend, Qwen35BackendArg::NativeCpu);

        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            let args = Args::try_parse_from(["openkindd", "--qwen35-backend", "mlx-fp32"]).unwrap();
            assert_eq!(args.qwen35_backend, Qwen35BackendArg::MlxFp32);
        }

        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        {
            let error = Args::try_parse_from(["openkindd", "--qwen35-backend", "mlx-fp32"])
                .unwrap_err()
                .to_string();
            assert!(error.contains("native-cpu"), "{error}");
        }

        let error = Args::try_parse_from(["openkindd", "--qwen35-backend", "mlx-bf16"])
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid value 'mlx-bf16'"), "{error}");
        assert!(error.contains("native-cpu"), "{error}");
    }

    #[test]
    fn qwen35_backend_environment_alias_is_declared() {
        let command = Args::command();
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_long() == Some("qwen35-backend"))
            .expect("--qwen35-backend argument");
        assert_eq!(arg.get_env(), Some(OsStr::new("OPENKIND_QWEN35_BACKEND")));
    }

    #[test]
    fn laya_backend_cli_values_and_diagnostics() {
        let default = Args::try_parse_from(["openkindd"]).unwrap();
        assert_eq!(default.family_args.laya_backend, LayaBackendArg::NativeCpu);

        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            let args = Args::try_parse_from(["openkindd", "--laya-backend", "mlx-fp32"]).unwrap();
            assert_eq!(args.family_args.laya_backend, LayaBackendArg::MlxFp32);
        }

        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        {
            let error = Args::try_parse_from(["openkindd", "--laya-backend", "mlx-fp32"])
                .unwrap_err()
                .to_string();
            assert!(error.contains("native-cpu"), "{error}");
        }

        let error = Args::try_parse_from(["openkindd", "--laya-backend", "mlx-bf16"])
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid value 'mlx-bf16'"), "{error}");

        let command = Args::command();
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_long() == Some("laya-backend"))
            .expect("--laya-backend argument");
        assert_eq!(arg.get_env(), Some(OsStr::new("OPENKIND_LAYA_BACKEND")));
    }

    #[test]
    fn args_qwen35_execution_override_parses_and_maps() {
        let forced =
            Args::try_parse_from(["openkindd", "--qwen35-execution", "nested-batched"]).unwrap();
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
            let args = Args::try_parse_from(["openkindd", "--qwen35-execution", flag]).unwrap();
            assert_eq!(
                Option::<ExecutionPlan>::from(args.qwen35_execution),
                Some(expected),
                "--qwen35-execution {flag}"
            );
        }

        let auto = Args::try_parse_from(["openkindd"]).unwrap();
        assert_eq!(Option::<ExecutionPlan>::from(auto.qwen35_execution), None);
        assert!(Args::try_parse_from(["openkindd", "--qwen35-execution", "turbo"]).is_err());
    }

    #[test]
    fn args_custom_values() {
        let args = Args::try_parse_from([
            "openkindd",
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
