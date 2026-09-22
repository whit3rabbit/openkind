//! Shared stub-server plumbing for the SDK-parity integration tests.
//!
//! Each test binary declares `mod common;` and uses [`spawn`] to boot an
//! axum stub whose behavior is defined per-request by a closure, mirroring
//! `httpx2.MockTransport` from the Python SDK's test suite. The returned
//! [`Requests`] handle exposes exactly what the Python tests assert on: the
//! `x-call` marker header, `x-typesafe-retry-count`, the parsed JSON body,
//! method, path, and full headers.

#![allow(dead_code)]

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Method, Request as HttpRequest, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use openkind_client::{question, Client, RetryPolicy, SystemRequest};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, MutexGuard};

/// The Python SDK test suite's canonical `RESULT` fixture: a full
/// three-answer SystemOne response.
pub fn result() -> Value {
    json!({
        "model": "jev-latest",
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "answers": {
            "spam": {"type": "noul", "noul": 0.98},
            "tone": {
                "type": "choice",
                "choice": "friendly",
                "confidence": 0.9,
                "probabilities": {"friendly": 0.9, "hostile": 0.1}
            },
            "quality": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.8,
                "legend": {"0": "bad", "1": "ok", "2": "great"},
                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}
            }
        }
    })
}

/// The Python test suite's `CARD` model-metadata fixture.
pub fn card() -> Value {
    json!({"name": "jev-latest", "description": "Fast model", "release_date": "2026-08-01"})
}

/// A request as the stub saw it.
#[derive(Clone, Default, Debug)]
pub struct Captured {
    /// The `x-call` marker header used to route behavior per logical call.
    pub call: Option<String>,
    /// `x-typesafe-retry-count` (`None` on the initial attempt).
    pub retry_count: Option<String>,
    /// Parsed JSON body, when present and valid.
    pub body: Option<Value>,
    pub method: String,
    pub path: String,
    pub headers: HeaderMap,
}

impl Captured {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }
}

/// What the stub answers for one request.
pub struct Outcome {
    status: u16,
    body: Value,
    retry_after_ms: Option<u64>,
    retry_after_raw: Option<String>,
    request_id: Option<String>,
}

impl Outcome {
    pub fn success(body: Value) -> Self {
        Self {
            status: 200,
            body,
            retry_after_ms: None,
            retry_after_raw: None,
            request_id: None,
        }
    }

    pub fn error(status: u16, body: Value) -> Self {
        Self {
            status,
            body,
            retry_after_ms: None,
            retry_after_raw: None,
            request_id: None,
        }
    }

    /// Attach `retry-after-ms` (and the derived integer-seconds `Retry-After`),
    /// exactly like the openkind server does on 429/529.
    pub fn with_retry_after_ms(mut self, ms: u64) -> Self {
        self.retry_after_ms = Some(ms);
        self
    }

    /// Attach a raw `Retry-After` header value (seconds or HTTP-date).
    pub fn with_retry_after_raw(mut self, value: impl Into<String>) -> Self {
        self.retry_after_raw = Some(value.into());
        self
    }

    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    /// Render into an axum response with the server's retry-header contract.
    pub fn into_response(self) -> Response {
        let mut response =
            (StatusCode::from_u16(self.status).unwrap(), Json(self.body)).into_response();
        let headers = response.headers_mut();
        if let Some(ms) = self.retry_after_ms {
            headers.insert("retry-after-ms", ms.to_string().parse().unwrap());
            headers.insert(
                "retry-after",
                ms.div_ceil(1000).to_string().parse().unwrap(),
            );
        }
        if let Some(raw) = self.retry_after_raw {
            headers.insert("retry-after", raw.parse().unwrap());
        }
        if let Some(id) = self.request_id {
            headers.insert("x-typesafe-request-id", id.parse().unwrap());
        }
        response
    }
}

type Handler = Arc<dyn Fn(&Captured) -> Outcome + Send + Sync>;

#[derive(Clone)]
struct StubState {
    requests: Arc<Mutex<Vec<Captured>>>,
    handler: Handler,
}

/// Live handle to the requests the stub has received.
#[derive(Clone)]
pub struct Requests(Arc<Mutex<Vec<Captured>>>);

impl Requests {
    pub fn snapshot(&self) -> Vec<Captured> {
        self.0.lock().unwrap().clone()
    }

    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    pub fn last(&self) -> Option<Captured> {
        self.0.lock().unwrap().last().cloned()
    }

    /// Retry-count header sequence across captured requests, e.g.
    /// `[None, "1", "2"]` — the exact list the Python tests assert.
    pub fn retry_sequence(&self) -> Vec<Option<String>> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.retry_count.clone())
            .collect()
    }

    /// Captured `x-call` values grouped per logical call.
    pub fn calls(&self, name: &str) -> Vec<Captured> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.call.as_deref() == Some(name))
            .cloned()
            .collect()
    }
}

/// Boot the stub and return its base URL plus the capture handle.
pub async fn spawn(
    handler: impl Fn(&Captured) -> Outcome + Send + Sync + 'static,
) -> (String, Requests) {
    let state = StubState {
        requests: Arc::new(Mutex::new(Vec::new())),
        handler: Arc::new(handler),
    };
    let requests = Requests(state.requests.clone());
    let app = Router::new()
        .route("/v1/systemone", axum::routing::any(handle))
        .route("/v1/models", axum::routing::any(handle))
        .with_state(state);
    let url = spawn_router(app).await;
    (url, requests)
}

/// Boot an arbitrary router (for bespoke slow/closing-endpoint tests).
pub async fn spawn_router(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn handle(
    State(state): State<StubState>,
    method: Method,
    request: HttpRequest<Body>,
) -> Response {
    let (parts, body) = request.into_parts();
    let bytes = axum::body::to_bytes(body, 1 << 20)
        .await
        .unwrap_or_default();
    let captured = Captured {
        call: parts
            .headers
            .get("x-call")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned),
        retry_count: parts
            .headers
            .get("x-typesafe-retry-count")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned),
        body: serde_json::from_slice(&bytes).ok(),
        method: method.to_string(),
        path: parts.uri.path().to_owned(),
        headers: parts.headers.clone(),
    };
    let outcome = {
        let mut guard: MutexGuard<Vec<Captured>> = state.requests.lock().unwrap();
        guard.push(captured);
        let last = guard.last().expect("just pushed");
        (state.handler)(last)
    };
    outcome.into_response()
}

/// A client with the Python suite's standard settings.
pub fn client(url: &str, policy: RetryPolicy) -> Client {
    Client::builder()
        .api_key("test-key")
        .base_url(url)
        .default_model("client-model")
        .timeout(std::time::Duration::from_secs(5))
        .retry(policy)
        .build()
        .unwrap()
}

/// The canonical `system_one(state="hello", questions={"q": Noul("?")})` call.
pub fn evaluate_request() -> SystemRequest {
    SystemRequest {
        state: openkind_client::State::Text("hello".into()),
        model: "client-model".into(),
        questions: std::collections::HashMap::from([("q".to_string(), question::noul("?"))]),
    }
}
