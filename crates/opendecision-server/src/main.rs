//! opendecisiond — the long-running daemon.
//!
//! Wires together:
//! - config (clap + serde)
//! - tracing (env-filtered JSON or pretty)
//! - the engine registry (mock by default, direct native Qwen3.5 when explicit
//!   offline artifact paths are configured)
//! - HTTP (axum) on one port
//! - gRPC (tonic) on another port (or the same port via SO_REUSEPORT —
//!   not done here; we use separate ports and document it)
//! - graceful shutdown on SIGINT/SIGTERM

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use opendecision_api::{grpc, http, AppState, AuthConfig};
use opendecision_backends::qwen35::{Qwen35DecisionEngine, Qwen35EngineConfig, SchedulerConfig};
use opendecision_engine::{DecisionEngine, EngineRegistry, MockEngine};
use opendecision_runtime::{peak_resident_bytes, BackendCapabilities, ExecutionPlan};
use tokio::net::TcpListener;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "opendecisiond", about = "opendecision inference daemon")]
struct Args {
    /// Address to bind the HTTP server on.
    #[arg(long, env = "OPENDECISION_HTTP_ADDR")]
    http_addr: Option<SocketAddr>,

    /// Deprecated pre-rename HTTP listener environment variable.
    #[arg(long, env = "OPENPICK_HTTP_ADDR", hide = true)]
    legacy_http_addr: Option<SocketAddr>,

    /// Address to bind the gRPC server on. Use `0` to disable the gRPC listener.
    #[arg(long, env = "OPENDECISION_GRPC_ADDR")]
    grpc_addr: Option<String>,

    /// Deprecated pre-rename gRPC listener environment variable.
    #[arg(long, env = "OPENPICK_GRPC_ADDR", hide = true)]
    legacy_grpc_addr: Option<String>,

    /// Comma-separated model aliases to expose. Aliases also listed in
    /// `--qwen35-aliases` use the native engine; all others use the mock.
    /// Register `jev-latest` to accept the SDK's default alias.
    #[arg(
        long,
        env = "OPENDECISION_MODELS",
        value_delimiter = ',',
        default_value = "mock,jev-latest"
    )]
    models: Vec<String>,

    /// Aliases in `--models` that should use the native Qwen3.5 engine.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_ALIASES",
        value_delimiter = ',',
        default_value = "qwen35-native"
    )]
    qwen35_aliases: Vec<String>,

    /// Offline selected-profile bundle root required by native aliases.
    #[arg(long, env = "OPENDECISION_QWEN35_BUNDLE_ROOT")]
    qwen35_bundle_root: Option<PathBuf>,

    /// Offline pinned Qwen checkpoint root required by native aliases.
    #[arg(long, env = "OPENDECISION_QWEN35_CHECKPOINT_ROOT")]
    qwen35_checkpoint_root: Option<PathBuf>,

    /// Digest-locked tokenizer JSON required by native aliases.
    #[arg(long, env = "OPENDECISION_QWEN35_TOKENIZER")]
    qwen35_tokenizer: Option<PathBuf>,

    /// Maximum concurrent native model evaluations.
    #[arg(long, env = "OPENDECISION_QWEN35_CONCURRENCY", default_value_t = 1)]
    qwen35_concurrency: usize,

    /// Additional native requests allowed to wait for execution.
    #[arg(long, env = "OPENDECISION_QWEN35_QUEUE", default_value_t = 2)]
    qwen35_queue: usize,

    /// Execution-plan override for diagnostics and reproducibility.
    /// Overrides adaptive scheduling only; memory and backend capability
    /// admission still apply.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_EXECUTION",
        value_enum,
        default_value_t = ExecutionArg::Auto
    )]
    qwen35_execution: ExecutionArg,

    /// Optional continuation tensor-payload ceiling per request.
    #[arg(long, env = "OPENDECISION_QWEN35_MAX_TENSOR_BYTES")]
    qwen35_max_tensor_bytes: Option<usize>,

    /// Optional process-memory ceiling for native admission.
    #[arg(long, env = "OPENDECISION_QWEN35_MAX_PROCESS_BYTES")]
    qwen35_max_process_bytes: Option<usize>,

    /// Forward scratch budget added to observed process memory.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_SCRATCH_BYTES",
        default_value_t = 1_073_741_824
    )]
    qwen35_scratch_bytes: usize,

    /// Allocator and runtime headroom added to observed process memory.
    #[arg(
        long,
        env = "OPENDECISION_QWEN35_ALLOCATOR_HEADROOM_BYTES",
        default_value_t = 536_870_912
    )]
    qwen35_allocator_headroom_bytes: usize,

    /// Optional bearer token required for `/v1/*`. If unset, the env
    /// `OPENDECISION_API_KEY` is consulted; if both are unset, auth is off
    /// (matches Phase 1 dev behavior).
    #[arg(long, env = "OPENDECISION_API_KEY")]
    api_key: Option<String>,

    /// Deprecated pre-rename API-key environment variable.
    #[arg(long, env = "OPENPICK_API_KEY", hide = true)]
    legacy_api_key: Option<String>,

    /// Per-client-IP request budget per minute on `/v1/*` routes.
    /// `0` disables rate limiting entirely.
    #[arg(long, env = "OPENDECISION_RATE_LIMIT_RPM", default_value_t = 120)]
    rate_limit_rpm: u32,

    /// Log filter. Standard `tracing_subscriber::EnvFilter` syntax.
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_filter: String,
}

/// CLI surface for `--qwen35-execution`: the three execution plans plus the
/// adaptive `auto` default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ExecutionArg {
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    init_tracing(&args.log_filter)?;

    let http_addr = resolve_alias(
        args.http_addr,
        args.legacy_http_addr,
        "OPENDECISION_HTTP_ADDR",
        "OPENPICK_HTTP_ADDR",
    )?
    .unwrap_or_else(|| "0.0.0.0:8080".parse().expect("valid default HTTP address"));
    let grpc_addr_value = resolve_alias(
        args.grpc_addr,
        args.legacy_grpc_addr,
        "OPENDECISION_GRPC_ADDR",
        "OPENPICK_GRPC_ADDR",
    )?
    .unwrap_or_else(|| "0.0.0.0:9090".to_owned());
    let grpc_addr = parse_grpc_addr(&grpc_addr_value)
        .context("invalid --grpc-addr (expected host:port, or `0` to disable)")?;

    let api_key = resolve_alias(
        args.api_key.filter(|s| !s.is_empty()),
        args.legacy_api_key.filter(|s| !s.is_empty()),
        "OPENDECISION_API_KEY",
        "OPENPICK_API_KEY",
    )?;
    let auth = api_key
        .filter(|s| !s.is_empty())
        .map(|k| AuthConfig::new(Some(k)))
        .unwrap_or_else(AuthConfig::from_env);
    if auth.is_required() {
        info!("api key auth: enabled (gate on /v1/* and gRPC)");
    } else {
        if http_addr.ip().is_unspecified() || grpc_addr.is_some_and(|g| g.ip().is_unspecified()) {
            tracing::warn!(
                "SECURITY WARNING: Server is binding to a public interface without authentication! Anyone with network access can execute inference queries."
            );
        } else {
            info!("api key auth: disabled (neither OPENDECISION_API_KEY nor TYPESAFE_API_KEY set)");
        }
    }

    info!(
        http = %http_addr,
        grpc = ?grpc_addr,
        models = ?args.models,
        "starting opendecisiond"
    );

    // Engine registry.
    let mut registry = EngineRegistry::new();
    let mock = Arc::new(MockEngine::new());
    let native_requested = args
        .models
        .iter()
        .any(|alias| args.qwen35_aliases.contains(alias));
    let native: Option<Arc<dyn DecisionEngine>> = if native_requested {
        let bundle_root = args
            .qwen35_bundle_root
            .clone()
            .context("native alias requested but --qwen35-bundle-root is missing")?;
        let checkpoint_root = args
            .qwen35_checkpoint_root
            .clone()
            .context("native alias requested but --qwen35-checkpoint-root is missing")?;
        let tokenizer_path = args
            .qwen35_tokenizer
            .clone()
            .context("native alias requested but --qwen35-tokenizer is missing")?;
        let mut scheduler = SchedulerConfig::for_pinned_profile(
            SchedulerConfig::LOWEST_MEASURED_SHARED_SAVINGS_RATIO,
            args.qwen35_max_tensor_bytes,
        )
        .with_backend_capabilities(BackendCapabilities::per_lane())
        .with_forced_strategy(args.qwen35_execution.into());
        if let Some(max_process_bytes) = args.qwen35_max_process_bytes {
            scheduler = scheduler.with_process_memory(
                opendecision_backends::qwen35::ProcessMemoryEnvelope {
                    observed_resident_bytes: peak_resident_bytes()
                        .context("read process peak RSS for native admission")?,
                    forward_scratch_bytes: args.qwen35_scratch_bytes,
                    allocator_headroom_bytes: args.qwen35_allocator_headroom_bytes,
                    max_process_bytes,
                },
            );
        }
        Some(Arc::new(
            Qwen35DecisionEngine::load(Qwen35EngineConfig {
                bundle_root,
                checkpoint_root,
                tokenizer_path,
                scheduler,
                max_concurrent_requests: args.qwen35_concurrency,
                max_queued_requests: args.qwen35_queue,
                retry_after_ms: 1_000,
            })
            .context("load native Qwen3.5 engine")?,
        ))
    } else {
        None
    };
    for alias in &args.models {
        let engine: Arc<dyn DecisionEngine> = if args.qwen35_aliases.contains(alias) {
            native
                .as_ref()
                .expect("native engine loaded when a native alias is requested")
                .clone()
        } else {
            mock.clone()
        };
        info!(alias, backend = engine.backend_id(), "registered model");
        registry.register(alias.clone(), engine);
    }

    // Install metrics recorder once, shared across HTTP/gRPC.
    opendecision_api::http::install_metrics_recorder().context("metrics recorder")?;

    let state = AppState::new(registry);

    // Build shutdown coordination channels.
    let (shutdown_tx, mut shutdown_rx_http) = tokio::sync::watch::channel(false);
    let mut shutdown_rx_grpc = shutdown_tx.subscribe();

    // Spawn signal watcher.
    let sig_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = sig_tx.send(true);
    });

    // Spawn HTTP server.
    let http_state = state.clone();
    let http_auth = auth.clone();
    let http_tx = shutdown_tx.clone();
    let rate_limiter = if args.rate_limit_rpm > 0 {
        opendecision_api::RateLimiter::new(opendecision_api::RateLimitConfig {
            max_requests: args.rate_limit_rpm,
            window: std::time::Duration::from_secs(60),
        })
    } else {
        opendecision_api::RateLimiter::disabled()
    };
    let http_handle = tokio::spawn(async move {
        let router = http::router_with_state_auth_rate_limit(
            http_state,
            http_auth,
            opendecision_api::http::MAX_PAYLOAD_SIZE_BYTES,
            rate_limiter,
        );
        let listener = match TcpListener::bind(http_addr).await {
            Ok(l) => l,
            Err(e) => {
                let _ = http_tx.send(true);
                return Err(anyhow::anyhow!("bind http {http_addr}: {e}"));
            }
        };
        info!(%http_addr, "http listening");
        let res = axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx_http.wait_for(|&v| v).await;
        })
        .await
        .context("http serve");
        let _ = http_tx.send(true);
        res
    });

    // Spawn gRPC server (unless disabled with `--grpc-addr 0`).
    let grpc_state = state.clone();
    let grpc_tx = shutdown_tx.clone();
    let grpc_handle = if let Some(grpc_addr) = grpc_addr {
        let svc = grpc::service_with_auth((*grpc_state.registry).clone(), auth.clone());
        Some(tokio::spawn(async move {
            info!(%grpc_addr, "grpc listening");
            let res = Server::builder()
                .add_service(svc)
                .serve_with_shutdown(grpc_addr, async move {
                    let _ = shutdown_rx_grpc.wait_for(|&v| v).await;
                })
                .await
                .context("grpc serve");
            let _ = grpc_tx.send(true);
            res
        }))
    } else {
        info!("grpc disabled (--grpc-addr 0)");
        None
    };

    // Wait for either to exit.
    if let Some(handle) = grpc_handle {
        let (h, g) = tokio::join!(http_handle, handle);
        h.context("http task")??;
        g.context("grpc task")??;
    } else {
        http_handle
            .await
            .context("http task")?
            .context("http serve")?;
    }

    info!("opendecisiond exited cleanly");
    Ok(())
}

/// Resolve a renamed setting while refusing ambiguous migration configuration.
fn resolve_alias<T: PartialEq>(
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

fn init_tracing(filter: &str) -> Result<()> {
    let env_filter = EnvFilter::try_new(filter).context("invalid log filter")?;
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .init();
    Ok(())
}

/// Parse the `--grpc-addr` value. The literal `0` (also `off`/`none`/`disabled`,
/// case-insensitive) disables the gRPC listener; anything else must be a
/// `host:port` socket address.
fn parse_grpc_addr(
    value: &str,
) -> std::result::Result<Option<SocketAddr>, std::net::AddrParseError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "off" | "none" | "disabled" => Ok(None),
        _ => value.trim().parse::<SocketAddr>().map(Some),
    }
}

/// Future that resolves on SIGINT (Ctrl-C) or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("received SIGINT"),
        _ = terminate => info!("received SIGTERM"),
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
