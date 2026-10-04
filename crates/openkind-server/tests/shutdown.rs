//! A second operator signal interrupts a stuck graceful drain with a failure status.
#![cfg(unix)]

use std::{process::Stdio, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    sync::Notify,
};

async fn log_until(
    lines: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    marker: &str,
) {
    tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(line) = lines.next_line().await.unwrap() {
            if line.contains(marker) {
                return;
            }
        }
        panic!("daemon exited before {marker}");
    })
    .await
    .expect("daemon log timeout");
}

#[tokio::test]
async fn second_signal_forces_nonzero_exit_while_first_signal_drains() {
    let entered = Arc::new(Notify::new());
    let notify = entered.clone();
    let app = axum::Router::new().route(
        "/v1/systemone",
        axum::routing::post(move || {
            let notify = notify.clone();
            async move {
                notify.notify_one();
                std::future::pending::<axum::http::StatusCode>().await
            }
        }),
    );
    let upstream = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_addr = upstream.local_addr().unwrap();
    let upstream_task = tokio::spawn(async move {
        axum::serve(upstream, app).await.unwrap();
    });
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = socket.local_addr().unwrap();
    drop(socket);
    let data = tempfile::tempdir().unwrap();
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_openkindd"))
        .args([
            "--http-addr",
            &addr.to_string(),
            "--grpc-addr",
            "0",
            "--models",
            "mock",
            "--rate-limit-rpm",
            "0",
            "--api-key",
            "local",
            "--proxy-cache-upstream-key",
            "upstream",
            "--proxy-cache-upstream",
            &format!("http://{upstream_addr}"),
            "--proxy-cache-models",
            "remote",
            "--proxy-cache-encoder",
            "hash",
            "--proxy-cache-data-dir",
            data.path().to_str().unwrap(),
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    log_until(&mut lines, "http listening").await;
    let pending = tokio::spawn(async move {
        reqwest::Client::new().post(format!("http://{addr}/v1/systemone")).bearer_auth("local")
            .json(&serde_json::json!({"model":"remote","state":"state","questions":{"q":{"type":"noul","instructions":"?"}}}))
            .send().await
    });
    tokio::time::timeout(Duration::from_secs(10), entered.notified())
        .await
        .unwrap();
    let pid = child.id().unwrap().to_string();
    assert!(std::process::Command::new("kill")
        .args(["-TERM", &pid])
        .status()
        .unwrap()
        .success());
    log_until(&mut lines, "draining listeners").await;
    assert!(
        child.try_wait().unwrap().is_none(),
        "the first signal must wait for the active request"
    );
    assert!(std::process::Command::new("kill")
        .args(["-INT", &pid])
        .status()
        .unwrap()
        .success());
    let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.code(), Some(130));
    pending.abort();
    upstream_task.abort();
}
