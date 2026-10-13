//! Python `json.dumps` and `repr` byte shapes shared by the prompt renderers.
//!
//! Several reference runtimes serialize state payloads and option text with
//! Python's `json.dumps(ensure_ascii=False)`, so the renderers reproduce those
//! bytes exactly. The helpers live here, not in a family module, so no family
//! has to import another family's renderer for them.

/// Serialize a JSON value exactly like Python `json.dumps(value,
/// ensure_ascii=False)` with default separators.
///
/// Strings escape the two mandatory characters plus the named control
/// characters and use `\uXXXX` (lowercase, no leading-zero suppression) for
/// every other control character; non-ASCII passes through as UTF-8. Floats
/// follow Python `repr`: shortest round-trip digits, `.0` on integral
/// values, and `e+NN`/`e-NN` exponents outside the fixed notation range.
pub(crate) fn python_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_owned(),
        serde_json::Value::Bool(true) => "true".to_owned(),
        serde_json::Value::Bool(false) => "false".to_owned(),
        serde_json::Value::Number(number) => python_number(number),
        serde_json::Value::String(text) => python_string(text),
        serde_json::Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(python_json).collect();
            format!("[{}]", rendered.join(", "))
        }
        serde_json::Value::Object(entries) => {
            let rendered: Vec<String> = entries
                .iter()
                .map(|(key, value)| format!("{}: {}", python_string(key), python_json(value)))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
    }
}

fn python_number(number: &serde_json::Number) -> String {
    if let Some(integer) = number.as_i64() {
        return integer.to_string();
    }
    if let Some(unsigned) = number.as_u64() {
        return unsigned.to_string();
    }
    // A JSON number is always i64, u64, or f64.
    python_float(number.as_f64().unwrap_or_default())
}

pub(crate) fn python_float(value: f64) -> String {
    if !value.is_finite() {
        // Wire states cannot carry non-finite numbers; callers validate
        // before reaching this point.
        return "null".to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }
    let magnitude = value.abs();
    // Python switches to exponent notation outside [1e-4, 1e16).
    if (1e-4..1e16).contains(&magnitude) {
        let rendered = format!("{value}");
        return if rendered.contains('.') {
            rendered
        } else {
            format!("{rendered}.0")
        };
    }
    // Shortest round-trip digits via Rust's `{:e}`, then re-exponentiate in
    // Python's `de+NN` shape. Python keeps the shortest mantissa in exponent
    // notation: `repr(1e-05)` is `1e-05`, not `1.0e-05`.
    let rendered = format!("{value:e}");
    let (mantissa, exponent) = rendered.split_once('e').unwrap_or((rendered.as_str(), "0"));
    let exponent_value: i32 = exponent.parse().unwrap_or(0);
    if exponent_value < 0 {
        format!("{mantissa}e-{:02}", -exponent_value)
    } else {
        format!("{mantissa}e+{exponent_value:02}")
    }
}

pub(crate) fn python_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            character if (character as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_matches_reference_separator_and_escape_shapes() {
        assert_eq!(python_json(&json!(null)), "null");
        assert_eq!(python_json(&json!(true)), "true");
        assert_eq!(python_json(&json!(7)), "7");
        assert_eq!(python_json(&json!(-3)), "-3");
        assert_eq!(python_json(&json!(1.5)), "1.5");
        assert_eq!(python_json(&json!(1.0)), "1.0");
        assert_eq!(python_json(&json!("a\"b\\c\nd")), "\"a\\\"b\\\\c\\nd\"");
        // Control characters use named or \uXXXX escapes; non-ASCII is raw.
        assert_eq!(python_json(&json!("a\u{1}é")), "\"a\\u0001é\"");
        assert_eq!(python_json(&json!([1, "x"])), "[1, \"x\"]");
        // Object keys render in the wire map's (sorted) order with Python
        // separators; the profile declares sorted keys as its deterministic
        // order.
        assert_eq!(
            python_json(&json!({"criterion": "c", "evidence": "s"})),
            "{\"criterion\": \"c\", \"evidence\": \"s\"}"
        );
    }

    #[test]
    fn python_float_matches_reference_notation() {
        assert_eq!(python_float(0.1), "0.1");
        assert_eq!(python_float(-2.5), "-2.5");
        assert_eq!(python_float(0.0001), "0.0001");
        assert_eq!(python_float(0.00001), "1e-05");
        assert_eq!(python_float(1e16), "1e+16");
        assert_eq!(python_float(1e15), "1000000000000000.0");
        assert_eq!(python_float(123_456.789), "123456.789");
    }
}
