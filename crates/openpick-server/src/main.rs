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

    /// Address to bind the gRPC server on. Use 0 to disable.
    #[arg(long, env = "OPENPICK_GRPC_ADDR", default_value = "0.0.0.0:9090")]
    grpc_addr: SocketAddr,

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

    /// Log filter. Standard `tracing_subscriber::EnvFilter` syntax.
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_filter: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    init_tracing(&args.log_filter)?;

    let auth = args
        .api_key
        .filter(|s| !s.is_empty())
        .map(|k| AuthConfig::new(Some(k)))
        .unwrap_or_else(AuthConfig::from_env);
    if auth.is_required() {
        info!("api key auth: enabled (gate on /v1/*)");
    } else {
        info!("api key auth: disabled (neither OPENPICK_API_KEY nor TYPESAFE_API_KEY set)");
    }

    info!(
        http = %args.http_addr,
        grpc = %args.grpc_addr,
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

    // Build shutdown signal future.
    let shutdown = shutdown_signal();

    // Spawn HTTP server.
    let http_state = state.clone();
    let http_addr = args.http_addr;
    let http_auth = auth.clone();
    let http_handle = tokio::spawn(async move {
        let router = http::router_with_state(http_state, http_auth);
        let listener = TcpListener::bind(http_addr)
            .await
            .with_context(|| format!("bind http {http_addr}"))?;
        info!(%http_addr, "http listening");
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown)
            .await
            .context("http serve")
    });

    // Spawn gRPC server (if port is non-zero).
    let grpc_state = state.clone();
    let grpc_addr = args.grpc_addr;
    let grpc_handle = if grpc_addr.port() != 0 {
        let svc = grpc::service((*grpc_state.registry).clone());
        Some(tokio::spawn(async move {
            info!(%grpc_addr, "grpc listening");
            Server::builder()
                .add_service(svc)
                .serve_with_shutdown(grpc_addr, shutdown_signal())
                .await
                .context("grpc serve")
        }))
    } else {
        info!("grpc disabled (port 0)");
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
        assert_eq!(args.grpc_addr, "0.0.0.0:9090".parse().unwrap());
        assert_eq!(args.models, vec!["mock", "jev-latest"]);
        assert_eq!(args.api_key, None);
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
            "--log-filter",
            "debug",
        ])
        .unwrap();

        assert_eq!(args.http_addr, "127.0.0.1:18080".parse().unwrap());
        assert_eq!(args.grpc_addr, "127.0.0.1:19090".parse().unwrap());
        assert_eq!(args.models, vec!["mock", "candle", "jev-latest"]);
        assert_eq!(args.api_key.as_deref(), Some("secret-token"));
        assert_eq!(args.log_filter, "debug");
    }
}
