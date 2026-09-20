//! Extraction helpers for error codes and human-readable messages from API response envelopes.

use serde_json::Value;

/// Matches the Python SDK's `MAX_ERROR_BODY_LENGTH`: raw bodies longer than
/// this are truncated in error messages.
pub(crate) const MAX_ERROR_BODY_LENGTH: usize = 200;

/// Extract the machine-readable error code the server puts in
/// `{"error": {"code": ...}}`, with lenient fallbacks for proxies that
/// reshape the envelope.
pub(crate) fn extract_code(body: Option<&Value>) -> Option<String> {
    let body = body?;
    if let Some(code) = str_field(&body["error"], "code") {
        return Some(code);
    }
    if let Some(code) = str_scalar(&body["error"]) {
        return Some(code);
    }
    str_field(body, "code")
}

/// Extract a human-readable message from an error body, accepting the
/// server's `error.message` shape plus the lenient fallbacks the Python SDK
/// tolerates (`message`, `detail` as string, object with `message`, or
/// FastAPI-style detail list).
pub(crate) fn extract_message(body: Option<&Value>) -> Option<String> {
    let body = body?;
    if let Some(error) = str_scalar(&body["error"]) {
        return Some(error);
    }
    if let Some(message) = str_field(&body["error"], "message") {
        return Some(message);
    }
    if let Some(message) = str_field(body, "message") {
        return Some(message);
    }
    if let Some(detail) = str_scalar(&body["detail"]) {
        return Some(detail);
    }
    if let Some(message) = str_field(&body["detail"], "message") {
        return Some(message);
    }
    if let Some(detail) = body["detail"].as_array() {
        // FastAPI-style validation errors: [{"loc": [...], "msg": "..."}, ...]
        let parts: Vec<String> = detail
            .iter()
            .filter_map(|entry| {
                let msg = str_field(entry, "msg")?;
                Some(match entry["loc"].as_array() {
                    Some(loc) => {
                        let path = loc
                            .iter()
                            .filter(|&seg| seg != "body")
                            .map(|seg| match seg {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join(".");
                        if path.is_empty() {
                            msg
                        } else {
                            format!("{path}: {msg}")
                        }
                    }
                    None => msg,
                })
            })
            .collect();
        if !parts.is_empty() {
            return Some(parts.join("; "));
        }
    }
    None
}

fn str_scalar(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

fn str_field(value: &Value, field: &str) -> Option<String> {
    value[field].as_str().map(str::to_owned)
}
