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
mod families;
mod installed;
mod playground;
mod proxy;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use openkind_api::{grpc, http, AppState, AuthConfig};
use openkind_backends::qwen35::{
    Qwen35Backend, Qwen35DecisionEngine, Qwen35EngineConfig, SchedulerConfig,
};
use openkind_engine::{DecisionEngine, EngineRegistry, MockEngine};
use openkind_model_store::{default_models_dir, ModelStore};
use openkind_runtime::{peak_resident_bytes, BackendCapabilities};
use tokio::net::TcpListener;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::args::{
    parse_grpc_addr, resolve_alias, Args, ArrowArg, PlaygroundArg, Qwen35BackendArg,
};

fn backend_from_arg(backend: Qwen35BackendArg) -> Qwen35Backend {
    match backend {
        Qwen35BackendArg::NativeCpu => Qwen35Backend::NativeCpu,
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        Qwen35BackendArg::MlxFp32 => Qwen35Backend::MlxFp32,
    }
}

fn load_qwen(
    args: &Args,
    bundle_root: PathBuf,
    checkpoint_root: PathBuf,
    tokenizer_path: PathBuf,
) -> Result<Arc<dyn DecisionEngine>> {
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
    Ok(Arc::new(
        Qwen35DecisionEngine::load(Qwen35EngineConfig {
            bundle_root,
            checkpoint_root,
            tokenizer_path,
            backend: backend_from_arg(args.qwen35_backend),
            scheduler,
            max_concurrent_requests: args.qwen35_concurrency,
            max_queued_requests: args.qwen35_queue,
            retry_after_ms: 1_000,
            evaluation_timeout: Some(std::time::Duration::from_millis(args.qwen35_timeout_ms)),
        })
        .context("load native Qwen3.5 engine")?,
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mut aliases = HashSet::new();
    for alias in &args.models {
        if !aliases.insert(alias) {
            anyhow::bail!("duplicate --models alias `{alias}`");
        }
    }
    for name in &args.installed_models {
        if !aliases.insert(name) {
            if args.models.contains(name) {
                anyhow::bail!("installed model alias `{name}` collides with --models");
            }
            anyhow::bail!("duplicate --installed-models alias `{name}`");
        }
    }

    init_tracing(&args.log_filter)?;

    let http_addr = resolve_alias(
        args.http_addr,
        args.legacy_http_addr,
        "OPENKIND_HTTP_ADDR",
        "OPENPICK_HTTP_ADDR",
    )?
    .unwrap_or_else(|| "0.0.0.0:8080".parse().expect("valid default HTTP address"));
    let grpc_addr_value = resolve_alias(
        args.grpc_addr.clone(),
        args.legacy_grpc_addr.clone(),
        "OPENKIND_GRPC_ADDR",
        "OPENPICK_GRPC_ADDR",
    )?
    .unwrap_or_else(|| "0.0.0.0:9090".to_owned());
    let grpc_addr = parse_grpc_addr(&grpc_addr_value)
        .context("invalid --grpc-addr (expected host:port, or `0` to disable)")?;

    let api_key = resolve_alias(
        args.api_key.clone().filter(|s| !s.is_empty()),
        args.legacy_api_key.clone().filter(|s| !s.is_empty()),
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
        playground = matches!(args.playground, PlaygroundArg::On),
        "starting openkindd"
    );

    // Engine registry.
    let mut registry = EngineRegistry::new();
    let mock = Arc::new(MockEngine::new());
    args.family_args.validate(&args.qwen35_aliases)?;
    let family_engines = args.family_args.load_requested(&args.models)?;
    for (alias, engine) in &family_engines {
        info!(
            alias,
            backend = engine.backend_id(),
            "registered family engine"
        );
    }
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
        Some(load_qwen(
            &args,
            bundle_root,
            checkpoint_root,
            tokenizer_path,
        )?)
    } else {
        None
    };
    for alias in &args.models {
        // A composite must resolve real registered siblings. Mock placeholders
        // would let self-references and unresolved composites silently serve demos.
        if args.family_args.winnow_aliases.contains(alias)
            || args.family_args.router_script_aliases.contains(alias)
        {
            continue;
        }
        let engine: Arc<dyn DecisionEngine> = if args.qwen35_aliases.contains(alias) {
            native
                .as_ref()
                .expect("native engine loaded when a native alias is requested")
                .clone()
        } else if let Some((_, family_engine)) = family_engines
            .iter()
            .find(|(family_alias, _)| family_alias == alias)
        {
            family_engine.clone()
        } else {
            mock.clone()
        };
        info!(alias, backend = engine.backend_id(), "registered model");
        registry.register(alias.clone(), engine);
    }

    // Winnow routers register after their siblings so they can hold handles
    // to the registered engines.
    for (alias, sibling_labels) in args.family_args.winnow_requested(&args.models)? {
        let (model_root, adapter) = args.family_args.winnow_artifacts(&alias)?;
        let mut siblings: Vec<(String, Arc<dyn DecisionEngine>)> = Vec::new();
        for sibling_alias in &sibling_labels {
            let engine = registry.get(sibling_alias).ok_or_else(|| {
                anyhow::anyhow!(
                    "winnow alias `{alias}` references unregistered sibling `{sibling_alias}`"
                )
            })?;
            siblings.push((sibling_alias.clone(), engine));
        }
        let engine = openkind_backends::families::winnow::WinnowEngine::load(
            openkind_backends::families::winnow::WinnowEngineConfig {
                model_root,
                adapter_path: adapter,
                limits: openkind_backends::families::support::FamilyLimits {
                    max_concurrent_requests: args.family_args.family_concurrency,
                    max_queued_requests: args.family_args.family_queue,
                    retry_after_ms: 1_000,
                    evaluation_timeout: Some(std::time::Duration::from_millis(
                        args.family_args.family_timeout_ms,
                    )),
                },
            },
            siblings,
        )
        .map_err(|error| anyhow::anyhow!("compose winnow alias `{alias}`: {error}"))?;
        info!(alias, backend = engine.backend_id(), "registered model");
        registry.register(alias, Arc::new(engine));
    }

    // Router-script composites register after their siblings so they can
    // hold handles to the registered engines.
    for (alias, rules) in args.family_args.router_script_requested(&args.models)? {
        let siblings: std::collections::HashMap<String, Arc<dyn DecisionEngine>> = rules
            .referenced_aliases()
            .into_iter()
            .filter_map(|sibling_alias| {
                registry
                    .get(sibling_alias)
                    .map(|engine| (sibling_alias.to_owned(), engine))
            })
            .collect();
        let router = openkind_backends::families::router_script::RouterScriptEngine::new(
            rules, &siblings,
        )
        .map_err(|error| anyhow::anyhow!("compose router-script alias `{alias}`: {error}"))?;
        info!(alias, backend = router.backend_id(), "registered model");
        registry.register(alias, Arc::new(router));
    }

    // Keep each installation's shared lock alive until shutdown. A concurrent
    // `openkind rm` then fails instead of removing files under a live engine.
    let mut _installed_guards = Vec::new();
    if !args.installed_models.is_empty() {
        let dir = match &args.models_dir {
            Some(dir) => dir.clone(),
            None => default_models_dir()?,
        };
        let store = ModelStore::new(dir)?;
        // Winnow routers register after every other installation so they can
        // hold handles to sibling engines registered earlier in this loop.
        let mut deferred_winnow = Vec::new();
        for name in &args.installed_models {
            let installed = store.acquire_serving(name)?;
            let kind = installed::installed_kind(&installed.manifest)
                .with_context(|| format!("unsupported installed model profile `{name}`"))?;
            if matches!(kind, installed::InstalledKind::Winnow) {
                deferred_winnow.push((name.clone(), installed));
                continue;
            }
            let engine = installed::load_installed_engine(&args, kind, &installed.root, &registry)?;
            info!(
                alias = name,
                backend = engine.backend_id(),
                "registered installed model"
            );
            registry.register(name.clone(), engine);
            _installed_guards.push(installed);
        }
        for (name, installed) in deferred_winnow {
            let engine = installed::load_installed_engine(
                &args,
                installed::InstalledKind::Winnow,
                &installed.root,
                &registry,
            )?;
            info!(
                alias = name,
                backend = engine.backend_id(),
                "registered installed model"
            );
            registry.register(name, engine);
            _installed_guards.push(installed);
        }
    }

    // Proxy cache mode: resolve the encoder (one explicit pull attempt when
    // the model is not installed), build the manager, and install the proxy
    // hook. Proxied aliases forward over gRPC through the same service.
    let mut _proxy_encoder_guard: Option<openkind_model_store::InstalledModel> = None;
    let proxy_service: Option<Arc<proxy::ProxyService>> = match &args.proxy_cache_upstream {
        Some(upstream) => {
            if !(0.0..1.0).contains(&args.proxy_cache_target_agreement) {
                anyhow::bail!(
                    "--proxy-cache-target-agreement must be in (0, 1), got {}",
                    args.proxy_cache_target_agreement
                );
            }
            let (embedder, guard) = proxy::resolve_encoder(
                &args.proxy_cache_encoder,
                args.proxy_cache_encoder_backend,
                args.models_dir.as_deref(),
            )
            .await?;
            _proxy_encoder_guard = guard;
            let task_config = openkind_backends::proxy_cache::TaskConfig {
                store_text: args.proxy_cache_store_text,
                min_train_samples: args.proxy_cache_min_train_samples,
                min_calib_samples: args.proxy_cache_min_calib_samples,
                shadow_min_samples: args.proxy_cache_shadow_min_samples,
                calib_fraction: args.proxy_cache_calib_fraction,
                min_new_samples: args.proxy_cache_min_new_samples,
                ..Default::default()
            };
            let data_dir = match &args.proxy_cache_data_dir {
                Some(dir) => dir.clone(),
                None => proxy::default_proxy_cache_dir()
                    .map_err(|error| anyhow::anyhow!("proxy cache data dir: {error}"))?,
            };
            let manager = openkind_backends::proxy_cache::ProxyCacheManager::new(
                openkind_backends::proxy_cache::ProxyCacheManagerConfig {
                    data_dir,
                    task_config,
                    target_agreement: args.proxy_cache_target_agreement,
                    confidence_floor: None,
                    admission_min_requests: args.proxy_cache_admission_min,
                    ..Default::default()
                },
                embedder,
            )
            .map_err(|error| anyhow::anyhow!("proxy cache manager: {error}"))?;
            let service = Arc::new(proxy::ProxyService::new(
                manager,
                proxy::ProxyCacheServiceConfig {
                    upstream: upstream.clone(),
                    upstream_key: args.proxy_cache_upstream_key.clone(),
                    upstream_timeout_ms: args.proxy_cache_upstream_timeout_ms,
                    proxied_models: args.proxy_cache_models.clone(),
                },
            ));
            for alias in &args.proxy_cache_models {
                registry.register(
                    alias.clone(),
                    Arc::new(proxy::ProxyForwardEngine::new(
                        service.clone(),
                        alias.clone(),
                    )),
                );
                info!(
                    alias,
                    backend = "proxy-cache/upstream-forward",
                    "registered proxied model (gRPC forwards upstream; HTTP answers via the cache hook)"
                );
            }
            info!(
                upstream = %upstream,
                encoder = %args.proxy_cache_encoder,
                models = ?args.proxy_cache_models,
                "proxy cache enabled"
            );
            Some(service)
        }
        None => None,
    };

    // Install metrics recorder once, shared across HTTP/gRPC.
    openkind_api::http::install_metrics_recorder().context("metrics recorder")?;

    let args = Arc::new(args);
    let mut state = AppState::new(registry);
    state.proxy = proxy_service.map(|service| service as Arc<dyn openkind_api::SystemProxy>);
    if matches!(args.playground, PlaygroundArg::On) {
        state.playground_models = Some(Arc::new(playground::LocalModels::new(
            args.clone(),
            state.registry.clone(),
            _installed_guards,
        )?));
    }

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
    let playground_enabled = matches!(args.playground, PlaygroundArg::On);
    let arrow_enabled = matches!(args.arrow, ArrowArg::On);
    let http_handle = tokio::spawn(async move {
        let _shutdown = ShutdownOnDrop(http_tx);
        let router = http::router_daemon_with_arrow(
            http_state,
            http_auth,
            openkind_api::http::MAX_PAYLOAD_SIZE_BYTES,
            rate_limiter,
            playground_enabled,
            arrow_enabled,
        );
        let listener = TcpListener::bind(http_addr)
            .await
            .with_context(|| format!("bind http {http_addr}"))?;
        info!(http_addr = %listener.local_addr()?, "http listening");
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx_http.wait_for(|&v| v).await;
        })
        .await
        .context("http serve")
    });

    // Spawn gRPC server (unless disabled with `--grpc-addr 0`).
    let grpc_state = state.clone();
    let grpc_tx = shutdown_tx.clone();
    let grpc_handle = if let Some(grpc_addr) = grpc_addr {
        let svc = grpc::service_with_auth((*grpc_state.registry).clone(), auth.clone());
        Some(tokio::spawn(async move {
            let _shutdown = ShutdownOnDrop(grpc_tx);
            info!(%grpc_addr, "grpc listening");
            Server::builder()
                .add_service(svc)
                .serve_with_shutdown(grpc_addr, async move {
                    let _ = shutdown_rx_grpc.wait_for(|&v| v).await;
                })
                .await
                .context("grpc serve")
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

struct ShutdownOnDrop(tokio::sync::watch::Sender<bool>);

impl Drop for ShutdownOnDrop {
    fn drop(&mut self) {
        // A panicking listener must also wake its peer so join! can finish
        // and main can propagate the listener task failure.
        let _ = self.0.send(true);
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

    #[tokio::test]
    async fn listener_panic_notifies_peer_shutdown() {
        let (tx, mut rx) = tokio::sync::watch::channel(false);
        let listener = tokio::spawn(async move {
            let _shutdown = ShutdownOnDrop(tx);
            panic!("simulated listener failure");
        });
        assert!(listener.await.unwrap_err().is_panic());
        tokio::time::timeout(std::time::Duration::from_secs(1), rx.wait_for(|&v| v))
            .await
            .unwrap()
            .unwrap();
    }
}
