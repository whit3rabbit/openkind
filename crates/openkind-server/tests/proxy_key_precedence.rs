//! Integration test verifying upstream key precedence and credential isolation:
//! 1. When `--proxy-cache-upstream-key` is configured alongside `--api-key`,
//!    the upstream receives the configured fixed key on first-use, cache-miss,
//!    plain-forward, and model discovery paths. The local daemon credential is NEVER
//!    disclosed to the upstream.
//! 2. When `--proxy-cache-upstream-key` is NOT configured, caller-key forwarding
//!    remains active.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::routing::{get, post};
use openkind_core::{
    Answer, ChoiceAnswer, NoulAnswer, SystemRequest, SystemResponse, Usage, WireHashState,
};
use serde_json::json;

#[derive(Default)]
struct UpstreamLog {
    received_keys: Mutex<Vec<Option<String>>>,
}

#[tokio::main]
async fn main() {
    test_fixed_upstream_key_precedence().await;
    test_caller_key_forwarding_when_no_fixed_key().await;
    println!("proxy_key_precedence: PASS");
}

async fn free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

async fn wait_for_ready(base_url: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if Instant::now() > deadline {
            panic!("daemon did not become healthy at {base_url}");
        }
        if tokio::net::TcpStream::connect(base_url.trim_start_matches("http://"))
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn spawn_fake_upstream() -> (SocketAddr, Arc<UpstreamLog>) {
    let log = Arc::new(UpstreamLog::default());
    let upstream_state = log.clone();

    let app = axum::Router::new()
        .route(
            "/v1/systemone",
            post(
                |State(log): State<Arc<UpstreamLog>>,
                 headers: axum::http::HeaderMap,
                 axum::Json(request): axum::Json<SystemRequest>| async move {
                    let key = headers
                        .get(axum::http::header::AUTHORIZATION)
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_owned);
                    log.received_keys.lock().unwrap().push(key);

                    let mut answers = std::collections::HashMap::with_hasher(WireHashState::default());
                    for (id, q) in &request.questions {
                        match q {
                            openkind_core::Question::Choice(_) => {
                                answers.insert(
                                    id.clone(),
                                    Answer::Choice(ChoiceAnswer {
                                        choice: "alpha".to_string(),
                                        probabilities: [("alpha".to_string(), 1.0)]
                                            .into_iter()
                                            .collect(),
                                        confidence: 1.0,
                                    }),
                                );
                            }
                            openkind_core::Question::Noul(_) => {
                                answers.insert(
                                    id.clone(),
                                    Answer::Noul(NoulAnswer {
                                        noul: 1.0,
                                    }),
                                );
                            }
                            openkind_core::Question::Score(_) => {
                                answers.insert(
                                    id.clone(),
                                    Answer::Score(openkind_core::ScoreAnswer {
                                        score: 1.0,
                                        legend: Default::default(),
                                        probabilities: Default::default(),
                                        confidence: 1.0,
                                    }),
                                );
                            }
                        }
                    }

                    axum::Json(SystemResponse {
                        model: "jev-resolved-1.0".to_string(),
                        answers,
                        usage: Usage {
                            input_tokens: 10,
                            output_tokens: 1,
                        },
                    })
                },
            ),
        )
        .route(
            "/v1/models",
            get(|State(log): State<Arc<UpstreamLog>>, headers: axum::http::HeaderMap| async move {
                let key = headers
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned);
                log.received_keys.lock().unwrap().push(key);
                axum::Json(json!({
                    "models": [
                        {"name": "jev-latest", "description": "alias", "release_date": "2026-01-01"}
                    ]
                }))
            }),
        )
        .with_state(upstream_state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, log)
}

async fn test_fixed_upstream_key_precedence() {
    let (upstream_addr, upstream_log) = spawn_fake_upstream().await;
    let http_port = free_port().await;
    let data_dir = tempfile::tempdir().unwrap();

    let exe = env!("CARGO_BIN_EXE_openkindd");
    let mut child = std::process::Command::new(exe)
        .args([
            "--http-addr",
            &format!("127.0.0.1:{http_port}"),
            "--grpc-addr",
            "0",
            "--rate-limit-rpm",
            "0",
            "--api-key",
            "local-daemon-secret-key",
            "--models",
            "mock",
            "--proxy-cache-upstream",
            &format!("http://{upstream_addr}"),
            "--proxy-cache-upstream-key",
            "fixed-upstream-secret-key",
            "--proxy-cache-models",
            "jev-latest",
            "--proxy-cache-encoder",
            "hash",
            "--proxy-cache-data-dir",
            data_dir.path().to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn openkindd");

    let daemon_url = format!("http://127.0.0.1:{http_port}");
    wait_for_ready(&daemon_url).await;

    let http = reqwest::Client::new();

    // 1. First request / cache miss:
    let choice_req = json!({
        "model": "jev-latest",
        "state": "sample text with alpha",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "classify",
                "criteria": {"alpha": "is alpha"}
            }
        }
    });

    let resp = http
        .post(format!("{daemon_url}/v1/systemone"))
        .bearer_auth("local-daemon-secret-key")
        .json(&choice_req)
        .send()
        .await
        .expect("choice request");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    {
        let keys = upstream_log.received_keys.lock().unwrap();
        assert_eq!(
            keys.last().and_then(|o| o.as_deref()),
            Some("Bearer fixed-upstream-secret-key"),
            "upstream must receive the fixed configured key, not the daemon key"
        );
        assert!(
            !keys
                .iter()
                .any(|k| k.as_deref() == Some("Bearer local-daemon-secret-key")),
            "upstream must NEVER see the local daemon key"
        );
    }

    // 2. Plain forward path (no choice questions, e.g. noul question):
    let noul_req = json!({
        "model": "jev-latest",
        "state": "sample text",
        "questions": {
            "q2": {
                "type": "noul",
                "instructions": "is this true?"
            }
        }
    });

    let resp = http
        .post(format!("{daemon_url}/v1/systemone"))
        .bearer_auth("local-daemon-secret-key")
        .json(&noul_req)
        .send()
        .await
        .expect("noul request");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    {
        let keys = upstream_log.received_keys.lock().unwrap();
        assert_eq!(
            keys.last().and_then(|o| o.as_deref()),
            Some("Bearer fixed-upstream-secret-key"),
            "plain forward path must also use fixed upstream key"
        );
        assert!(
            !keys
                .iter()
                .any(|k| k.as_deref() == Some("Bearer local-daemon-secret-key")),
            "upstream must NEVER see the local daemon key"
        );
    }

    // 3. Upstream /v1/models listing:
    let resp = http
        .get(format!("{daemon_url}/v1/models"))
        .bearer_auth("local-daemon-secret-key")
        .send()
        .await
        .expect("models request");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    {
        let keys = upstream_log.received_keys.lock().unwrap();
        assert_eq!(
            keys.last().and_then(|o| o.as_deref()),
            Some("Bearer fixed-upstream-secret-key"),
            "models discovery must use fixed upstream key"
        );
    }

    let _ = child.kill();
    let _ = child.wait();
}

async fn test_caller_key_forwarding_when_no_fixed_key() {
    let (upstream_addr, upstream_log) = spawn_fake_upstream().await;
    let http_port = free_port().await;
    let data_dir = tempfile::tempdir().unwrap();

    let exe = env!("CARGO_BIN_EXE_openkindd");
    let mut child = std::process::Command::new(exe)
        .args([
            "--http-addr",
            &format!("127.0.0.1:{http_port}"),
            "--grpc-addr",
            "0",
            "--rate-limit-rpm",
            "0",
            "--models",
            "mock",
            "--proxy-cache-upstream",
            &format!("http://{upstream_addr}"),
            "--proxy-cache-models",
            "jev-latest",
            "--proxy-cache-encoder",
            "hash",
            "--proxy-cache-data-dir",
            data_dir.path().to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn openkindd");

    let daemon_url = format!("http://127.0.0.1:{http_port}");
    wait_for_ready(&daemon_url).await;

    let http = reqwest::Client::new();

    // 1. Caller provides their own key:
    let choice_req = json!({
        "model": "jev-latest",
        "state": "sample text with alpha",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "classify",
                "criteria": {"alpha": "is alpha"}
            }
        }
    });

    let resp = http
        .post(format!("{daemon_url}/v1/systemone"))
        .bearer_auth("caller-provided-upstream-key")
        .json(&choice_req)
        .send()
        .await
        .expect("choice request with caller key");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    {
        let keys = upstream_log.received_keys.lock().unwrap();
        assert_eq!(
            keys.last().and_then(|o| o.as_deref()),
            Some("Bearer caller-provided-upstream-key"),
            "in caller-key mode, caller key must reach upstream"
        );
    }

    // 2. Caller does not provide a key in caller-key mode:
    let resp = http
        .post(format!("{daemon_url}/v1/systemone"))
        .json(&choice_req)
        .send()
        .await
        .expect("choice request without key");

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::UNAUTHORIZED,
        "missing caller key in caller-key mode must return 401"
    );

    let _ = child.kill();
    let _ = child.wait();
}
