//! Unit tests for client builder and configuration precedence.

use std::time::Duration;

use super::builder::{clean_env_value, resolve_api_key, resolve_lookup, resolve_optional_api_key};
use super::core::Client;
use super::transport::is_user_overridable;
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

/// Ports of the Python SDK's `test_api_key_whitespace`,
/// `test_invalid_explicit_key_does_not_fall_back_to_env`, and
/// `test_invalid_api_key` parametrizations, using an injected lookup so no
/// process-global environment state is touched.
#[test]
fn api_key_resolution_matches_python_sdk() {
    // The injected lookup composes clean_env_value exactly like the
    // production `non_empty_env`, so the env source arrives trimmed.
    let env_key = |_: &str| clean_env_value("  env-key  ".to_owned());

    // Explicit keys are trimmed before use (constructor source).
    for padding in ["", "\n", "\r\n", " \t\r\n "] {
        let key = resolve_api_key(Some(format!("{padding}test-key{padding}")), |_| None).unwrap();
        assert_eq!(key, "test-key");
    }
    // Environment keys are trimmed too (env source).
    assert_eq!(resolve_api_key(None, env_key).unwrap(), "env-key");

    // An explicit empty or invalid key errors instead of falling back to
    // the environment, and the error never echoes the env credential.
    for key in ["", " \t\r\n ", "\0private", "private\0"] {
        let err = resolve_api_key(Some(key.to_owned()), env_key).unwrap_err();
        assert!(!err.contains("env-key"), "{err}");
    }

    // Non-printable, whitespace, and non-ASCII characters are rejected from
    // both sources without echoing the credential.
    for character in ["\n", "\r", "\t", "\u{1f}", "\u{7f}", " ", "é", "\u{200b}"] {
        let explicit = resolve_api_key(Some(format!("ts_live_private{character}suffix")), |_| None)
            .unwrap_err();
        let from_env = resolve_api_key(None, |_| Some(format!("ts_live_private{character}suffix")))
            .unwrap_err();
        for err in [explicit, from_env] {
            assert!(err.contains("printable ASCII"), "{err}");
            assert!(!err.contains("ts_live_private"), "{err}");
        }
    }

    // No key anywhere names both variables (test_missing_key).
    let err = resolve_api_key(None, |_| None).unwrap_err();
    assert!(err.contains("OPENKIND_API_KEY"), "{err}");
    assert!(err.contains("TYPESAFE_API_KEY"), "{err}");
}

#[test]
fn client_debug_does_not_leak_api_key() {
    let client = Client::new("super-secret-key").unwrap();
    let rendered = format!("{client:?}");
    assert!(!rendered.contains("super-secret-key"), "{rendered}");
}

#[test]
fn unauthenticated_mode_only_allows_a_missing_key() {
    assert_eq!(resolve_optional_api_key(None, |_| None).unwrap(), None);
    assert!(resolve_api_key(None, |_| None).is_err());
    assert_eq!(
        resolve_optional_api_key(None, |_| Some("env-key".into())).unwrap(),
        Some("env-key".into())
    );
    assert_eq!(
        resolve_optional_api_key(Some("explicit".into()), |_| Some("env-key".into())).unwrap(),
        Some("explicit".into())
    );
    for key in ["", "   ", "invalid key", "invalid\nkey", "sécret"] {
        assert!(resolve_optional_api_key(Some(key.into()), |_| Some("valid".into())).is_err());
        assert!(Client::builder()
            .allow_unauthenticated()
            .api_key(key)
            .build()
            .is_err());
    }
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
fn base_url_surrounding_whitespace_is_trimmed_before_validation() {
    let client = Client::builder()
        .api_key("k")
        .base_url("  http://example.test/prefix/  ")
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
fn base_url_rejects_components_that_hide_endpoint_paths_or_credentials() {
    for url in [
        "http://example.test?query=1",
        "http://example.test/#fragment",
        "http://user:password@example.test",
    ] {
        assert!(matches!(
            Client::builder().api_key("k").base_url(url).build(),
            Err(Error::Config(_))
        ));
    }
}

#[test]
fn base_url_with_invalid_port_rejected_at_build() {
    // Passes the http(s) prefix check but fails URL parsing, so endpoint
    // resolution must reject it at construction time instead of per request.
    let err = Client::builder()
        .api_key("k")
        .base_url("http://example.test:not-a-port")
        .build()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err:?}");
    assert!(
        err.to_string().contains("invalid base URL"),
        "unexpected error: {err}"
    );
}

#[test]
fn protected_headers_are_not_user_overridable() {
    // SDK identification/protocol headers always win over per-call extras.
    for name in [
        "authorization",
        "accept",
        "user-agent",
        "x-typesafe-sdk",
        "x-typesafe-runtime",
        "x-typesafe-retry-count",
        "host",
        "content-length",
        "transfer-encoding",
    ] {
        assert!(!is_user_overridable(name, false), "{name} with no body");
        assert!(!is_user_overridable(name, true), "{name} with body");
    }
    // Content-type belongs to the JSON body: only settable on bodiless calls.
    assert!(!is_user_overridable("content-type", true));
    assert!(is_user_overridable("content-type", false));
    // Anything else passes through to the request.
    assert!(is_user_overridable("x-custom", true));
    assert!(is_user_overridable("x-custom", false));
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
