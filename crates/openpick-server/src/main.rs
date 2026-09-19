//! openpickd — the long-running daemon.
//!
//! Wires together:
//! - config (clap + serde)
//! - tracing (env-filtered JSON or pretty)
//! - the engine registry (Phase 1: just the mock)
//! - HTTP (axum) on one port
//! - gRPC (tonic) on another port (or the same port via SO_REUSEPORT —
//!   not done here; we use separate ports and document it)
//! - graceful shutdown on SIGINT/SIGTERM

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use openpick_api::{grpc, http, AppState, AuthConfig};
use openpick_engine::{DecisionEngine, EngineRegistry, MockEngine};
use tokio::net::TcpListener;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "openpickd", about = "openpick inference daemon")]
struct Args {
    /// Address to bind the HTTP server on.
    #[arg(long, env = "OPENPICK_HTTP_ADDR", default_value = "0.0.0.0:8080")]
    http_addr: SocketAddr,

    /// Address to bind the gRPC server on. Use `0` to disable the gRPC listener.
    #[arg(long, env = "OPENPICK_GRPC_ADDR", default_value = "0.0.0.0:9090")]
    grpc_addr: String,

    /// Comma-separated model aliases to expose. For Phase 1, all point
    /// at the mock engine. Register `jev-latest` to accept the SDK's
    /// default alias.
    #[arg(
        long,
        env = "OPENPICK_MODELS",
        value_delimiter = ',',
        default_value = "mock,jev-latest"
    )]
    models: Vec<String>,

    /// Optional bearer token required for `/v1/*`. If unset, the env
    /// `OPENPICK_API_KEY` is consulted; if both are unset, auth is off
    /// (matches Phase 1 dev behavior).
    #[arg(long, env = "OPENPICK_API_KEY")]
    api_key: Option<String>,

    /// Per-client-IP request budget per minute on `/v1/*` routes.
    /// `0` disables rate limiting entirely.
    #[arg(long, env = "OPENPICK_RATE_LIMIT_RPM", default_value_t = 120)]
    rate_limit_rpm: u32,

    /// Log filter. Standard `tracing_subscriber::EnvFilter` syntax.
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_filter: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    init_tracing(&args.log_filter)?;

    let grpc_addr = parse_grpc_addr(&args.grpc_addr)
        .context("invalid --grpc-addr (expected host:port, or `0` to disable)")?;

    let auth = args
        .api_key
        .filter(|s| !s.is_empty())
        .map(|k| AuthConfig::new(Some(k)))
        .unwrap_or_else(AuthConfig::from_env);
    if auth.is_required() {
        info!("api key auth: enabled (gate on /v1/* and gRPC)");
    } else {
        if args.http_addr.ip().is_unspecified()
            || grpc_addr.is_some_and(|g| g.ip().is_unspecified())
        {
            tracing::warn!(
                "SECURITY WARNING: Server is binding to a public interface without authentication! Anyone with network access can execute inference queries."
            );
        } else {
            info!("api key auth: disabled (neither OPENPICK_API_KEY nor TYPESAFE_API_KEY set)");
        }
    }

    info!(
        http = %args.http_addr,
        grpc = ?grpc_addr,
        models = ?args.models,
        "starting openpickd"
    );

    // Engine registry.
    let mut registry = EngineRegistry::new();
    let mock = Arc::new(MockEngine::new());
    for alias in &args.models {
        registry.register(alias.clone(), mock.clone());
        info!(alias, backend = mock.backend_id(), "registered model");
    }

    // Install metrics recorder once, shared across HTTP/gRPC.
    openpick_api::http::install_metrics_recorder().context("metrics recorder")?;

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
    let http_addr = args.http_addr;
    let http_auth = auth.clone();
    let http_tx = shutdown_tx.clone();
    let rate_limiter = if args.rate_limit_rpm > 0 {
        openpick_api::RateLimiter::new(openpick_api::RateLimitConfig {
            max_requests: args.rate_limit_rpm,
            window: std::time::Duration::from_secs(60),
        })
    } else {
        openpick_api::RateLimiter::disabled()
    };
    let http_handle = tokio::spawn(async move {
        let router = http::router_with_state_auth_rate_limit(
            http_state,
            http_auth,
            openpick_api::http::MAX_PAYLOAD_SIZE_BYTES,
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

    info!("openpickd exited cleanly");
    Ok(())
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
        let args = Args::try_parse_from(["openpickd"]).unwrap();
        assert_eq!(args.http_addr, "0.0.0.0:8080".parse().unwrap());
        assert_eq!(args.grpc_addr, "0.0.0.0:9090");
        assert_eq!(args.models, vec!["mock", "jev-latest"]);
        assert_eq!(args.api_key, None);
        assert_eq!(args.rate_limit_rpm, 120);
        assert_eq!(args.log_filter, "info");
    }

    #[test]
    fn args_custom_values() {
        let args = Args::try_parse_from([
            "openpickd",
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

        assert_eq!(args.http_addr, "127.0.0.1:18080".parse().unwrap());
        assert_eq!(args.grpc_addr, "127.0.0.1:19090");
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
}
