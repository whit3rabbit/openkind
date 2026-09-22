//! openkindd — the long-running daemon.
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

mod args;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use openkind_api::{grpc, http, AppState, AuthConfig};
use openkind_backends::qwen35::{
    Qwen35Backend, Qwen35DecisionEngine, Qwen35EngineConfig, SchedulerConfig,
};
use openkind_engine::{DecisionEngine, EngineRegistry, MockEngine};
use openkind_runtime::{peak_resident_bytes, BackendCapabilities};
use tokio::net::TcpListener;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::args::{parse_grpc_addr, resolve_alias, Args};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    init_tracing(&args.log_filter)?;

    let http_addr = resolve_alias(
        args.http_addr,
        args.legacy_http_addr,
        "OPENKIND_HTTP_ADDR",
        "OPENPICK_HTTP_ADDR",
    )?
    .unwrap_or_else(|| "0.0.0.0:8080".parse().expect("valid default HTTP address"));
    let grpc_addr_value = resolve_alias(
        args.grpc_addr,
        args.legacy_grpc_addr,
        "OPENKIND_GRPC_ADDR",
        "OPENPICK_GRPC_ADDR",
    )?
    .unwrap_or_else(|| "0.0.0.0:9090".to_owned());
    let grpc_addr = parse_grpc_addr(&grpc_addr_value)
        .context("invalid --grpc-addr (expected host:port, or `0` to disable)")?;

    let api_key = resolve_alias(
        args.api_key.filter(|s| !s.is_empty()),
        args.legacy_api_key.filter(|s| !s.is_empty()),
        "OPENKIND_API_KEY",
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
            info!("api key auth: disabled (neither OPENKIND_API_KEY nor TYPESAFE_API_KEY set)");
        }
    }

    info!(
        http = %http_addr,
        grpc = ?grpc_addr,
        models = ?args.models,
        "starting openkindd"
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
            scheduler =
                scheduler.with_process_memory(openkind_backends::qwen35::ProcessMemoryEnvelope {
                    observed_resident_bytes: peak_resident_bytes()
                        .context("read process peak RSS for native admission")?,
                    forward_scratch_bytes: args.qwen35_scratch_bytes,
                    allocator_headroom_bytes: args.qwen35_allocator_headroom_bytes,
                    max_process_bytes,
                });
        }
        Some(Arc::new(
            Qwen35DecisionEngine::load(Qwen35EngineConfig {
                bundle_root,
                checkpoint_root,
                tokenizer_path,
                backend: Qwen35Backend::NativeCpu,
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
    openkind_api::http::install_metrics_recorder().context("metrics recorder")?;

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
        openkind_api::RateLimiter::new(openkind_api::RateLimitConfig {
            max_requests: args.rate_limit_rpm,
            window: std::time::Duration::from_secs(60),
        })
    } else {
        openkind_api::RateLimiter::disabled()
    };
    let http_handle = tokio::spawn(async move {
        let router = http::router_with_state_auth_rate_limit(
            http_state,
            http_auth,
            openkind_api::http::MAX_PAYLOAD_SIZE_BYTES,
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

    info!("openkindd exited cleanly");
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
