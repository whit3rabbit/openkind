//! `opendecision-client`: an async Rust client for the `opendecisiond`
//! daemon and the TypeSafe Jev SystemOne HTTP API.
//!
//! Built directly on the `opendecision-core` wire types, so requests and
//! responses are validated against the same canonical structs the server
//! uses (`crates/opendecision-core/schemas/`). Wire-compatible with
//! `https://api.typesafe.ai` and the `typesafe_sdk` Python SDK: the same
//! retry taxonomy, `Retry-After` handling, error envelope parsing, request
//! ids, and env-var resolution — with defaults mirroring the Python
//! `RetryPolicy` (2 retries, 0.5s→5s exponential backoff, 25% jitter,
//! 408/429/5xx retried, 30s total budget).
//!
//! # Quick start
//! ```no_run
//! use opendecision_client::{question, Client};
//!
//! # async fn demo() -> Result<(), opendecision_client::Error> {
//! let client = Client::builder()
//!     .api_key("dev-key")
//!     .base_url("http://127.0.0.1:8080")
//!     .build()?;
//!
//! let response = client
//!     .system_one(
//!         "I was charged twice. Please help.",
//!         [
//!             ("billing", question::noul("Is this about billing?")),
//!             (
//!                 "tone",
//!                 question::choice("What is the tone?", [("calm", None), ("angry", None)]),
//!             ),
//!         ],
//!     )
//!     .await?;
//!
//! let billing = match &response.answers["billing"] {
//!     opendecision_client::Answer::Noul(answer) => answer.noul,
//!     _ => unreachable!(),
//! };
//! println!("billing probability: {billing}");
//! # Ok(())
//! # }
//! ```
//!
//! # Rate limiting, overload, and retries
//!
//! 429 (`rate_limited`) and 529 (`overloaded`) responses carry
//! `retry-after-ms` / `Retry-After` headers. The client reads them and waits
//! exactly as long as the server asked, falling back to exponential backoff
//! with jitter for other transient failures (408, 5xx, connection errors,
//! timeouts). See [`RetryPolicy`] to tune or disable.
//!
//! # Errors
//!
//! All failures surface as [`Error`]; server rejections as
//! [`Error::Api`](`Error::Api`) with the status, error `code`, `message`,
//! `x-typesafe-request-id`, and parsed `retry_after`. See [`error`].
//!
//! # Environment variables
//!
//! | Builder option   | Primary                 | Python-SDK fallback   |
//! |------------------|-------------------------|-----------------------|
//! | `api_key`        | `OPENDECISION_API_KEY`  | `TYPESAFE_API_KEY`    |
//! | `base_url`       | `OPENDECISION_BASE_URL` | `TYPESAFE_BASE_URL`   |
//! | `default_model`  | `OPENDECISION_DEFAULT_MODEL` | `TYPESAFE_DEFAULT_MODEL` |
//!
//! Explicit builder values win; empty/whitespace env values are ignored.

#![warn(missing_docs)]

pub mod client;
pub mod error;
pub mod question;
pub mod retry;

pub use client::{
    Client, ClientBuilder, Health, IntoState, RequestOptions, DEFAULT_BASE_URL, DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
};
pub use error::{ApiError, ApiErrorKind, Error};
pub use retry::RetryPolicy;

// Wire types re-exported from `opendecision-core` — the single source of
// truth for the Jev protocol. Client and server share these structs, which
// is what guarantees schema conformance (`jev-v1-request.json` /
// `jev-v1-response.json`).
pub use opendecision_core::{
    Answer, ChoiceAnswer, ChoiceQuestion, ModelInfo, ModelsResponse, NoulAnswer, NoulCriteria,
    NoulQuestion, Question, ScoreAnswer, ScoreQuestion, State, SystemRequest, SystemResponse,
    Usage, API_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The re-exported wire types must round-trip through the pinned Jev
    /// request schema shape: `state`, `model`, `questions` with `type` tags.
    #[test]
    fn system_request_serializes_to_jev_wire_shape() {
        let request = SystemRequest {
            state: State::Text("hello".into()),
            model: "jev-latest".into(),
            questions: HashMap::from([
                ("b".to_string(), question::noul("Is this billing?")),
                (
                    "c".to_string(),
                    question::choice("Tone?", [("calm", None), ("angry", None)]),
                ),
                (
                    "s".to_string(),
                    question::score("Severity?", ["low", "high"]),
                ),
            ]),
        };
        let wire = serde_json::to_value(&request).unwrap();
        assert_eq!(wire["state"], "hello");
        assert_eq!(wire["model"], "jev-latest");
        assert_eq!(wire["questions"]["b"]["type"], "noul");
        assert_eq!(wire["questions"]["c"]["type"], "choice");
        assert_eq!(wire["questions"]["s"]["type"], "score");
        // Response side: noul answers carry no confidence field per spec.
        let response: SystemResponse = serde_json::from_value(serde_json::json!({
            "model": "jev-latest",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "answers": {"b": {"type": "noul", "noul": 0.5}}
        }))
        .unwrap();
        match &response.answers["b"] {
            Answer::Noul(answer) => assert_eq!(answer.noul, 0.5),
            other => panic!("expected noul answer, got {other:?}"),
        }
    }
}
