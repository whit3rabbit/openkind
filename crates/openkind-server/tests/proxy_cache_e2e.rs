//! Black-box proxy-cache e2e test: the real `openkindd` binary runs in
//! proxy mode against a fake Jev upstream, and the test drives the full
//! lifecycle over HTTP — forward with the caller's key, collect, train,
//! promote, then serve confident requests locally with `x-openkind-cache:
//! local`.
//!
//! The cache runs the dependency-free hash embedder, so no model assets
//! are downloaded (workspace invariant #6). The tuning flags shrink the
//! bootstrap thresholds so the promotion lands within a small request loop.

use std::time::{Duration, Instant};

use axum::extract::State;
use axum::routing::post;
use openkind_core::{Answer, SystemRequest, SystemResponse};
use serde_json::json;

/// What the fake upstream saw.
#[derive(Default)]
struct UpstreamLog {
    evaluations: std::sync::atomic::AtomicUsize,
    last_key: std::sync::Mutex<Option<String>>,
}

#[tokio::main]
async fn main() {
    let exit = run().await;
    std::process::exit(exit);
}

async fn run() -> i32 {
    // 1. Fake upstream: answers choice questions by keyword, resolving the
    // alias to a concrete model version, and records the caller key it saw.
    let log = std::sync::Arc::new(UpstreamLog::default());
    let upstream_state = log.clone();
    let upstream = axum::Router::new()
        .route(
            "/v1/systemone",
            post(move |State(log): State<std::sync::Arc<UpstreamLog>>, headers: axum::http::HeaderMap, JsonForUpstream(request): JsonForUpstream| {
                let log = log.clone();
                async move {
                    log.evaluations
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let key = headers
                        .get(axum::http::header::AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_owned);
                    *log.last_key.lock().unwrap() = key;
                    let mut answers: std::collections::HashMap<
                        String,
                        Answer,
                        openkind_core::WireHashState,
                    > = std::collections::HashMap::with_hasher(openkind_core::WireHashState::default());
                    // The teacher answers from the request STATE (deterministic
                    // per cluster), never from the instructions: instructions
                    // are fixed per task, so they carry no signal.
                    let state_text = match &request.state {
                        openkind_core::State::Text(text) => text.to_lowercase(),
                        other => format!("{other:?}").to_lowercase(),
                    };
                    for (id, question) in &request.questions {
                        let openkind_core::Question::Choice(_) = question else {
                            continue;
                        };
                        let (label, p, other) = if state_text.contains("alpha") {
                            ("alpha", 0.96, "beta")
                        } else if state_text.contains("beta") {
                            ("beta", 0.94, "alpha")
                        } else {
                            ("alpha", 0.55, "beta")
                        };
                        answers.insert(
                            id.clone(),
                            Answer::Choice(openkind_core::ChoiceAnswer {
                                choice: label.to_string(),
                                probabilities: [
                                    (label.to_string(), p),
                                    (other.to_string(), 1.0 - p),
                                ]
                                .into_iter()
                                .collect(),
                                confidence: p,
                            }),
                        );
                    }
                    axum::Json(SystemResponse {
                        model: "jev-test-1.0".to_string(),
                        answers,
                        usage: openkind_core::Usage {
                            input_tokens: 11,
                            output_tokens: 1,
                        },
                    })
                }
            }),
        )
        .route(
            "/v1/models",
            axum::routing::get(|| async {
                axum::Json(json!({
                    "models": [
                        {"name": "jev-latest", "description": "alias", "release_date": "2026-01-01"},
                        {"name": "jev-test-1.0", "description": "resolved", "release_date": "2026-01-01"}
                    ]
                }))
            }),
        )
        .with_state(upstream_state);
    let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_addr = upstream_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(upstream_listener, upstream).await.unwrap();
    });

    // 2. Start the daemon in proxy mode against the fake upstream.
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
            "--proxy-cache-target-agreement",
            "0.85",
            "--proxy-cache-admission-min",
            "1",
            "--proxy-cache-min-train-samples",
            "30",
            "--proxy-cache-min-calib-samples",
            "12",
            "--proxy-cache-shadow-min-samples",
            "8",
            "--proxy-cache-calib-fraction",
            "0.3",
            "--proxy-cache-min-new-samples",
            "30",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn openkindd");

    let result = drive(&format!("http://127.0.0.1:{http_port}"), &log).await;
    let _ = child.kill();
    let _ = child.wait();
    match result {
        Ok(()) => {
            println!("proxy-cache e2e: PASS");
            0
        }
        Err(error) => {
            eprintln!("proxy-cache e2e: FAIL: {error}");
            1
        }
    }
}

struct JsonForUpstream(SystemRequest);

impl<S: Send + Sync> axum::extract::FromRequest<S> for JsonForUpstream {
    type Rejection = (axum::http::StatusCode, &'static str);

    async fn from_request(
        request: axum::extract::Request,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        let bytes = axum::body::to_bytes(request.into_body(), 16 * 1024 * 1024)
            .await
            .map_err(|_| (axum::http::StatusCode::BAD_REQUEST, "body read failed"))?;
        let request: SystemRequest = serde_json::from_slice(&bytes)
            .map_err(|_| (axum::http::StatusCode::BAD_REQUEST, "bad request json"))?;
        Ok(Self(request))
    }
}

async fn free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

async fn drive(base: &str, upstream_log: &std::sync::Arc<UpstreamLog>) -> Result<(), String> {
    let client = openkind_client::Client::builder()
        .api_key("sk-caller-key-1")
        .base_url(base)
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|error| format!("client: {error}"))?;

    // Wait for startup.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if Instant::now() > deadline {
            return Err("daemon did not become healthy".into());
        }
        if tokio::net::TcpStream::connect(base.trim_start_matches("http://"))
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let request = |text: &str, id: &str| SystemRequest {
        state: openkind_core::State::Text(text.to_string()),
        model: "jev-latest".to_string(),
        questions: [(id.to_string(), choice_question())].into_iter().collect(),
    };

    // The first request must forward upstream with the caller's key.
    let response = client
        .evaluate(request(CLUSTER_A_1, "q1"))
        .await
        .map_err(|error| format!("first evaluate: {error}"))?;
    assert_eq!(
        response.model, "jev-test-1.0",
        "upstream resolves the alias"
    );
    assert_eq!(
        upstream_log.last_key.lock().unwrap().as_deref(),
        Some("Bearer sk-caller-key-1"),
        "the caller's own key reaches the upstream"
    );
    assert!(
        upstream_log
            .evaluations
            .load(std::sync::atomic::Ordering::SeqCst)
            >= 1
    );

    // Drive the loop until the cache answers this cluster locally.
    let request_for = |index: usize| {
        if index.is_multiple_of(2) {
            request(&cluster_text(0, index), "qa")
        } else {
            request(&cluster_text(1, index), "qb")
        }
    };
    // Debug builds run the fit 20-50x slower and hold the engine lock while
    // fitting, so a single request can stall briefly during promotion.
    let deadline = Instant::now() + Duration::from_secs(240);
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|error| format!("raw client: {error}"))?;
    let mut forwarded = 0usize;
    let mut local_answer = None;
    while Instant::now() < deadline {
        // Mixed traffic on both clusters keeps the task's classes balanced.
        // Local answers carry the resolved upstream model name too, so
        // detect them by their zero token usage (the fake upstream
        // reports real usage on forwarded responses).
        let raw = http
            .post(format!("{base}/v1/systemone"))
            .bearer_auth("sk-caller-key-1")
            .json(&request_for(forwarded))
            .send()
            .await
            .map_err(|error| format!("loop evaluate at {forwarded}: {error}"))?;
        assert_eq!(raw.status(), axum::http::StatusCode::OK);
        let cache_header = raw
            .headers()
            .get("x-openkind-cache")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_default();
        let detail = raw
            .headers()
            .get("x-openkind-cache-detail")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_default();
        let response: SystemResponse = raw
            .json()
            .await
            .map_err(|error| format!("loop response at {forwarded}: {error}"))?;
        forwarded += 1;
        if response.usage.input_tokens == 0 {
            // Check the same response: a repeat can legitimately audit upstream.
            assert_eq!(cache_header, "local", "confident traffic is served locally");
            assert!(
                detail.contains("student-v"),
                "detail names the student version: {detail}"
            );
            local_answer = Some(response);
            break;
        }
    }
    let Some(local) = local_answer else {
        return Err(format!(
            "cache never answered locally after {forwarded} requests"
        ));
    };
    assert!(forwarded >= 40, "promotion should need some traffic");
    let answer = local
        .answers
        .get("qa")
        .or_else(|| local.answers.get("qb"))
        .expect("answer present");
    let Answer::Choice(choice) = answer else {
        return Err("expected a choice answer".into());
    };
    assert!(
        choice.choice == "alpha" || choice.choice == "beta",
        "served label {}",
        choice.choice
    );
    assert!(
        choice.confidence > 0.8,
        "reported confidence {} should be the peakedness",
        choice.confidence
    );
    let probability_sum: f64 = choice.probabilities.values().sum();
    assert!(
        (probability_sum - 1.0).abs() < 1e-6,
        "probabilities must sum to one"
    );
    assert_eq!(local.usage.input_tokens, 0, "local answers cost no tokens");

    // Out-of-distribution text forwards upstream.
    let novel = http
        .post(format!("{base}/v1/systemone"))
        .bearer_auth("sk-caller-key-1")
        .json(&request("zzz unheard vocabulary qqq 完全不同的主题", "qo"))
        .send()
        .await
        .map_err(|error| format!("ood evaluate: {error}"))?;
    assert_eq!(
        novel
            .headers()
            .get("x-openkind-cache")
            .and_then(|value| value.to_str().ok()),
        Some("upstream"),
        "OOD text must forward"
    );

    // A different caller key is unverified: its first request forwards.
    let other = openkind_client::Client::builder()
        .api_key("sk-caller-key-2")
        .base_url(base)
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|error| format!("client 2: {error}"))?;
    let response = other
        .evaluate(request(CLUSTER_A_1, "q2"))
        .await
        .map_err(|error| format!("second key evaluate: {error}"))?;
    assert_eq!(response.model, "jev-test-1.0", "unverified key forwards");

    // /v1/models falls back to the local registry when no upstream key is
    // configured (the upstream listing needs credentials). With
    // --proxy-cache-upstream-key set, the upstream listing is proxied.
    let models = other
        .list_models()
        .await
        .map_err(|error| format!("models: {error}"))?;
    assert!(
        models.models.iter().any(|model| model.name == "jev-latest"),
        "local registry answers model discovery: {:?}",
        models
            .models
            .iter()
            .map(|model| model.name.clone())
            .collect::<Vec<_>>()
    );
    Ok(())
}

fn cluster_text(cluster: usize, index: usize) -> String {
    match cluster {
        0 => format!("alpha report number {index} discusses alpha topics and alpha findings"),
        _ => format!("beta memo number {index} covers beta matters and beta outcomes"),
    }
}

const CLUSTER_A_1: &str = "alpha report number 1 discusses alpha topics and alpha findings";

/// Fixed instructions: the task fingerprint covers instructions and
/// criteria, so per-request variation must live in the state alone.
fn choice_question() -> openkind_core::Question {
    openkind_core::Question::Choice(openkind_core::ChoiceQuestion {
        instructions: json!("Classify the document topic."),
        criteria: [
            ("alpha".to_string(), Some("alpha documents".to_string())),
            ("beta".to_string(), Some("beta documents".to_string())),
        ]
        .into_iter()
        .collect(),
    })
}
