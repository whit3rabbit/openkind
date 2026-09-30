//! `openkind playground` — launch the local web playground.
//!
//! Connects to an already-healthy daemon when one is listening on the
//! target address; otherwise spawns a loopback `openkindd` with the
//! playground route enabled, waits for `/health`, opens the browser, and
//! supervises the child until Ctrl-C.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

/// How long to wait for a freshly spawned daemon to answer `/health`.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(600);
/// Delay between `/health` probes while waiting for the daemon.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(150);
/// Timeout for the probe that decides whether a daemon is already running.
const EXISTING_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Entry point for the `playground` CLI subcommand.
pub fn cmd_playground(
    http_addr: String,
    models: String,
    installed_models: String,
    models_dir: Option<PathBuf>,
    api_key: Option<String>,
    no_open: bool,
) -> Result<()> {
    let addr: SocketAddr = http_addr
        .parse()
        .with_context(|| format!("invalid --http-addr `{http_addr}` (expected host:port)"))?;
    if addr.port() == 0 {
        bail!("--http-addr needs a fixed, non-zero port so the playground can open it");
    }
    if !addr.ip().is_loopback() {
        bail!("--http-addr must be loopback because the playground can change loaded models");
    }
    let origin = format!("http://{addr}");
    let url = playground_url(&origin);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    if rt.block_on(health_probe(&origin, EXISTING_PROBE_TIMEOUT)) {
        if !rt.block_on(playground_probe(&origin)) {
            bail!("the daemon at {origin} has no playground; restart it with --playground on, or choose another --http-addr");
        }
        println!("openkind playground: {url}");
        println!("using the daemon already running at {origin}");
        open_browser(&url, no_open);
        return Ok(());
    }

    let mut signals = {
        let _entered = rt.enter();
        ShutdownSignals::new()?
    };
    let mut child = ChildGuard(Some(spawn_daemon(
        &addr,
        &models,
        &installed_models,
        models_dir.as_deref(),
        api_key.as_deref(),
    )?));

    println!("waiting for openkindd at {origin} …");
    if rt.block_on(wait_until_ready(
        child.0.as_mut().unwrap(),
        &origin,
        &mut signals,
    ))? {
        println!("openkind playground: {url}");
        println!("Ctrl-C stops the daemon.");
        open_browser(&url, no_open);
    }

    let status = rt.block_on(wait_for_exit(child.0.as_mut().unwrap(), &mut signals))?;
    child.0 = None;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

struct ChildGuard(Option<Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        // Startup, probe, or supervision failures must not orphan a daemon.
        // Normal and signal-driven exits are reaped before reaching this guard.
        if let Some(child) = self.0.as_mut() {
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

struct ShutdownSignals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
}

impl ShutdownSignals {
    fn new() -> Result<Self> {
        Ok(Self {
            #[cfg(unix)]
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                .context("install SIGINT handler")?,
            #[cfg(unix)]
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .context("install SIGTERM handler")?,
        })
    }

    async fn recv(&mut self) -> Result<i32> {
        #[cfg(unix)]
        {
            tokio::select! {
                value = self.interrupt.recv() => value.context("SIGINT stream closed").map(|_| libc::SIGINT),
                value = self.terminate.recv() => value.context("SIGTERM stream closed").map(|_| libc::SIGTERM),
            }
        }
        #[cfg(not(unix))]
        std::future::pending().await
    }
}

fn forward_shutdown(child: &mut Child, signal: i32) -> Result<()> {
    #[cfg(unix)]
    {
        let pid = libc::pid_t::try_from(child.id()).context("invalid daemon PID")?;
        // The child remains owned and unreaped here, preventing PID reuse.
        // kill is async-signal-safe; Tokio delivers signals outside the handler.
        if unsafe { libc::kill(pid, signal) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error).context("forward shutdown to openkindd");
            }
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = signal;
        child.kill().context("stop openkindd")
    }
}

async fn wait_until_ready(
    child: &mut Child,
    origin: &str,
    signals: &mut ShutdownSignals,
) -> Result<bool> {
    let start = Instant::now();
    while start.elapsed() < HEALTH_TIMEOUT {
        if let Some(status) = child.try_wait().context("poll openkindd")? {
            bail!("openkindd exited before becoming healthy (status {status})");
        }
        tokio::select! {
            signal = signals.recv() => {
                forward_shutdown(child, signal?)?;
                return Ok(false);
            }
            healthy = health_probe(origin, HEALTH_POLL_INTERVAL) => {
                if healthy { return Ok(true); }
            }
        }
        tokio::select! {
            signal = signals.recv() => {
                forward_shutdown(child, signal?)?;
                return Ok(false);
            }
            _ = tokio::time::sleep(HEALTH_POLL_INTERVAL) => {}
        }
    }
    bail!(
        "openkindd did not answer /health within {} s",
        HEALTH_TIMEOUT.as_secs()
    )
}

async fn wait_for_exit(child: &mut Child, signals: &mut ShutdownSignals) -> Result<ExitStatus> {
    loop {
        if let Some(status) = child.try_wait().context("wait for openkindd")? {
            return Ok(status);
        }
        tokio::select! {
            signal = signals.recv() => forward_shutdown(child, signal?)?,
            _ = tokio::time::sleep(HEALTH_POLL_INTERVAL) => {}
        }
    }
}

/// Probe `{origin}/health`; `true` when the target daemon reports healthy.
async fn health_probe(origin: &str, timeout: Duration) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let Ok(response) = client.get(format!("{origin}/health")).send().await else {
        return false;
    };
    response.status().is_success()
        && response
            .json::<serde_json::Value>()
            .await
            .is_ok_and(|body| body.get("status").and_then(|status| status.as_str()) == Some("ok"))
}

async fn playground_probe(origin: &str) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(EXISTING_PROBE_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
    else {
        return false;
    };
    match client.get(playground_url(origin)).send().await {
        Ok(response) if response.status().is_success() => response
            .text()
            .await
            .is_ok_and(|body| body.contains("<title>openkind playground</title>")),
        _ => false,
    }
}

/// Spawn the playground daemon: loopback bind, no gRPC listener, playground
/// route on, rate limiting off so benchmark bursts are not throttled.
fn spawn_daemon(
    addr: &SocketAddr,
    models: &str,
    installed_models: &str,
    models_dir: Option<&std::path::Path>,
    api_key: Option<&str>,
) -> Result<std::process::Child> {
    let mut cmd = Command::new("openkindd");
    cmd.arg("--http-addr")
        .arg(addr.to_string())
        .arg("--grpc-addr")
        .arg("0")
        .arg("--playground")
        .arg("on")
        .arg("--rate-limit-rpm")
        .arg("0")
        .arg("--models")
        .arg(models);
    cmd.arg("--installed-models").arg(installed_models);
    if let Some(dir) = models_dir {
        cmd.arg("--models-dir").arg(dir);
    }
    if let Some(key) = api_key {
        // Pass via environment so the secret never lands in the process
        // table (ps aux / /proc/*/cmdline).
        cmd.env("OPENKIND_API_KEY", key);
    }
    cmd.stdin(Stdio::null());
    cmd.spawn()
        .context("execute openkindd (is openkindd built and on PATH?)")
}

/// Launch the platform browser at `url` detached; no-op with `no_open`.
fn open_browser(url: &str, no_open: bool) {
    if no_open {
        return;
    }
    let (program, args) = opener_argv(current_os(), url);
    let opened = Command::new(program)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok();
    if !opened {
        eprintln!("could not launch a browser via `{program}`; open {url} manually");
    }
}

/// The current OS family, as consumed by [`opener_argv`].
fn current_os() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "xdg"
    }
}

/// Resolve the browser opener program and arguments for an OS family
/// (`macos`, `windows`, anything else is an XDG desktop).
fn opener_argv(os: &str, url: &str) -> (&'static str, Vec<String>) {
    match os {
        "macos" => ("open", vec![url.to_owned()]),
        "windows" => (
            "cmd",
            vec![
                "/c".to_owned(),
                "start".to_owned(),
                String::new(),
                url.to_owned(),
            ],
        ),
        _ => ("xdg-open", vec![url.to_owned()]),
    }
}

/// Join a daemon origin and the playground path without double slashes.
fn playground_url(origin: &str) -> String {
    format!("{}/playground", origin.trim_end_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_requires_the_daemon_status_body() {
        use std::io::{Read, Write};
        for (status, body, expected) in [
            ("200 OK", r#"{"status":"ok"}"#, true),
            ("200 OK", r#"{"status":"failed"}"#, false),
            ("200 OK", "unrelated application", false),
            ("503 Service Unavailable", r#"{"status":"ok"}"#, false),
        ] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let worker = std::thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut buffer = [0; 2048];
                let size = socket.read(&mut buffer).unwrap();
                assert!(String::from_utf8_lossy(&buffer[..size]).starts_with("GET /health "));
                write!(
                    socket,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            });
            assert_eq!(
                health_probe(&origin, Duration::from_secs(2)).await,
                expected
            );
            worker.join().unwrap();
        }
    }

    #[tokio::test]
    async fn existing_daemon_must_serve_the_playground_asset() {
        use std::io::{Read, Write};
        for (status, body, expected) in [
            ("200 OK", "<title>openkind playground</title>", true),
            ("404 Not Found", "not found", false),
            ("200 OK", "unrelated application", false),
        ] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let worker = std::thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut buffer = [0; 2048];
                let size = socket.read(&mut buffer).unwrap();
                assert!(String::from_utf8_lossy(&buffer[..size]).starts_with("GET /playground "));
                write!(
                    socket,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            });
            assert_eq!(playground_probe(&origin).await, expected);
            worker.join().unwrap();
        }
    }

    #[test]
    fn playground_url_joins_without_double_slash() {
        assert_eq!(
            playground_url("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/playground"
        );
        assert_eq!(
            playground_url("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080/playground"
        );
    }

    #[test]
    fn opener_argv_matches_the_platform_conventions() {
        assert_eq!(
            opener_argv("macos", "http://x/p"),
            ("open", vec!["http://x/p".to_owned()])
        );
        assert_eq!(
            opener_argv("windows", "http://x/p"),
            (
                "cmd",
                vec![
                    "/c".to_owned(),
                    "start".to_owned(),
                    String::new(),
                    "http://x/p".to_owned()
                ]
            )
        );
        assert_eq!(
            opener_argv("linux", "http://x/p"),
            ("xdg-open", vec!["http://x/p".to_owned()])
        );
        let (program, _) = opener_argv(current_os(), "http://x/p");
        assert!(!program.is_empty());
    }
}
