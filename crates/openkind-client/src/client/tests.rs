//! Unit tests for client builder and configuration precedence.

use std::time::Duration;

use super::builder::{clean_env_value, resolve_lookup};
use super::core::Client;
use crate::error::Error;

/// Port of the Python SDK's `test_resolution` precedence matrix, using an
/// injected lookup so no process-global environment state is touched.
#[test]
fn resolve_lookup_precedence() {
    let primary = "OPENKIND_API_KEY";
    let fallback = "TYPESAFE_API_KEY";
    let env = |wanted: &'static str, value: &'static str| {
        move |name: &str| (name == wanted).then(|| value.to_owned())
    };

    // Explicit wins over both env vars.
    assert_eq!(
        resolve_lookup(
            Some("explicit".into()),
            env(primary, "p"),
            primary,
            fallback
        ),
        Some("explicit".into())
    );
    // Primary env beats fallback env.
    assert_eq!(
        resolve_lookup(None, env(primary, "p"), primary, fallback),
        Some("p".into())
    );
    // Fallback env is used when primary is absent (TYPESAFE_* parity).
    assert_eq!(
        resolve_lookup(None, env(fallback, "f"), primary, fallback),
        Some("f".into())
    );
    // Default applies only when nothing resolves (None via |_: Option<_>|).
    assert_eq!(resolve_lookup(None, |_| None, primary, fallback), None);
    // Empty/whitespace env values are ignored, like the Python SDK
    // (`os.environ.get(env, "").strip() or default`); the lookup layer
    // (`non_empty_env`) applies that filter before resolution.
    assert_eq!(clean_env_value("   ".into()), None);
    assert_eq!(clean_env_value("".into()), None);
    assert_eq!(
        clean_env_value("  real-value  ".into()),
        Some("real-value".into())
    );
}

#[test]
fn client_debug_does_not_leak_api_key() {
    let client = Client::new("super-secret-key").unwrap();
    let rendered = format!("{client:?}");
    assert!(!rendered.contains("super-secret-key"), "{rendered}");
}

#[test]
fn base_url_trailing_slashes_trimmed() {
    let client = Client::builder()
        .api_key("k")
        .base_url("http://example.test/prefix///")
        .build()
        .unwrap();
    assert_eq!(client.base_url(), "http://example.test/prefix");
}

#[test]
fn invalid_base_url_rejected_at_build() {
    let err = Client::builder()
        .api_key("k")
        .base_url("ftp://example.test")
        .build()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err:?}");
}

#[test]
fn zero_timeout_rejected_at_build() {
    let err = Client::builder()
        .api_key("k")
        .base_url("http://example.test")
        .timeout(Duration::ZERO)
        .build()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err:?}");
}
