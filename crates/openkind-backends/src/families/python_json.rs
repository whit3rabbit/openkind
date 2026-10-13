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
    let mut out = String::new();
    write_python_json(value, &mut out);
    out
}

/// [`python_json`] appended to `out`, so nested values share one buffer
/// instead of allocating a string per element.
pub(crate) fn write_python_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(true) => out.push_str("true"),
        serde_json::Value::Bool(false) => out.push_str("false"),
        serde_json::Value::Number(number) => write_python_number(number, out),
        serde_json::Value::String(text) => write_python_string(text, out),
        serde_json::Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(entries) => {
            out.push('{');
            for (index, (key, value)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_string(key, out);
                out.push_str(": ");
                write_python_json(value, out);
            }
            out.push('}');
        }
    }
}

fn write_python_number(number: &serde_json::Number, out: &mut String) {
    use std::fmt::Write;
    // Writing to a String cannot fail.
    if let Some(integer) = number.as_i64() {
        let _ = write!(out, "{integer}");
    } else if let Some(unsigned) = number.as_u64() {
        let _ = write!(out, "{unsigned}");
    } else {
        // A JSON number is always i64, u64, or f64.
        out.push_str(&python_float(number.as_f64().unwrap_or_default()));
    }
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
    write_python_string(text, &mut out);
    out
}

/// [`python_string`] appended to `out`.
pub(crate) fn write_python_string(text: &str, out: &mut String) {
    out.reserve(text.len() + 2);
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
                use std::fmt::Write;
                // Writing to a String cannot fail.
                let _ = write!(out, "\\u{:04x}", character as u32);
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

/// Reproducible JSON corpus shared by the writer equivalence tests.
#[cfg(test)]
pub(crate) mod corpus {
    /// Deterministic xorshift stream so the corpus is reproducible without
    /// a random-number dependency.
    pub(crate) struct Stream(pub(crate) u64);

    impl Stream {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        pub(crate) fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    /// Strings that exercise every escape class: quotes, backslashes, named
    /// and numeric control characters, DEL, multi-byte scalars, astral
    /// characters, and plain text.
    pub(crate) fn random_text(stream: &mut Stream) -> String {
        const PIECES: &[&str] = &[
            "plain",
            " ",
            "\"",
            "\\",
            "\n",
            "\r",
            "\t",
            "\u{8}",
            "\u{c}",
            "\u{1}",
            "\u{1f}",
            "\u{7f}",
            "é",
            "日本",
            "\u{2028}",
            "\u{1f600}",
            "<|im_end|>",
            "key",
            "0",
        ];
        let count = stream.below(6);
        (0..count)
            .map(|_| PIECES[stream.below(PIECES.len() as u64) as usize])
            .collect()
    }

    fn random_number(stream: &mut Stream) -> serde_json::Value {
        const FLOATS: &[f64] = &[
            0.0,
            -0.0,
            1.0,
            -1.5,
            0.1,
            1e-4,
            9.999e-5,
            1e-5,
            1e15,
            1e16,
            1.5e300,
            5e-324,
            f64::MAX,
            f64::MIN_POSITIVE,
            123_456.789,
        ];
        match stream.below(5) {
            0 => serde_json::json!(stream.next() as i64),
            1 => serde_json::json!(stream.next()),
            2 => serde_json::json!(FLOATS[stream.below(FLOATS.len() as u64) as usize]),
            3 => {
                // A raw bit pattern; skip the non-finite ones JSON cannot hold.
                let value = f64::from_bits(stream.next());
                serde_json::Number::from_f64(value)
                    .map_or(serde_json::Value::Null, serde_json::Value::Number)
            }
            _ => serde_json::json!(stream.below(1000)),
        }
    }

    pub(crate) fn random_value(stream: &mut Stream, depth: u32) -> serde_json::Value {
        let kinds = if depth == 0 { 5 } else { 7 };
        match stream.below(kinds) {
            0 => serde_json::Value::Null,
            1 => serde_json::Value::Bool(stream.below(2) == 0),
            2 => random_number(stream),
            3 | 4 => serde_json::Value::String(random_text(stream)),
            5 => serde_json::Value::Array(
                (0..stream.below(5))
                    .map(|_| random_value(stream, depth - 1))
                    .collect(),
            ),
            _ => serde_json::Value::Object(
                (0..stream.below(5))
                    .map(|_| (random_text(stream), random_value(stream, depth - 1)))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::corpus::{random_value, Stream};
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

    /// The pre-buffer implementation, kept as the oracle for the single
    /// buffer writer: one `String` per element joined afterwards.
    fn legacy_python_json(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Null => "null".to_owned(),
            serde_json::Value::Bool(true) => "true".to_owned(),
            serde_json::Value::Bool(false) => "false".to_owned(),
            serde_json::Value::Number(number) => {
                if let Some(integer) = number.as_i64() {
                    integer.to_string()
                } else if let Some(unsigned) = number.as_u64() {
                    unsigned.to_string()
                } else {
                    python_float(number.as_f64().unwrap_or_default())
                }
            }
            serde_json::Value::String(text) => legacy_python_string(text),
            serde_json::Value::Array(items) => {
                let rendered: Vec<String> = items.iter().map(legacy_python_json).collect();
                format!("[{}]", rendered.join(", "))
            }
            serde_json::Value::Object(entries) => {
                let rendered: Vec<String> = entries
                    .iter()
                    .map(|(key, value)| {
                        format!(
                            "{}: {}",
                            legacy_python_string(key),
                            legacy_python_json(value)
                        )
                    })
                    .collect();
                format!("{{{}}}", rendered.join(", "))
            }
        }
    }

    fn legacy_python_string(text: &str) -> String {
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

    #[test]
    fn single_buffer_writer_matches_the_legacy_join_implementation() {
        let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
        let mut nested = 0;
        for _ in 0..5_000 {
            let value = random_value(&mut stream, 4);
            if matches!(
                value,
                serde_json::Value::Array(_) | serde_json::Value::Object(_)
            ) {
                nested += 1;
            }
            assert_eq!(python_json(&value), legacy_python_json(&value), "{value}");
        }
        assert!(nested > 1_000, "the corpus must exercise containers");

        // Empty containers and a deeply nested document.
        for value in [
            json!([]),
            json!({}),
            json!([[], {}, [[{}]]]),
            json!({"a": {"b": {"c": [1, [2, [3, {"d": "e\n"}]]]}}}),
        ] {
            assert_eq!(python_json(&value), legacy_python_json(&value));
        }
    }

    #[test]
    fn string_writer_matches_the_legacy_escape_for_every_scalar_value() {
        for code in 0..=0x10_FFFF_u32 {
            let Some(character) = char::from_u32(code) else {
                continue;
            };
            let text = character.to_string();
            assert_eq!(
                python_string(&text),
                legacy_python_string(&text),
                "{code:#x}"
            );
        }
        let mut appended = String::from("prefix:");
        write_python_string("a\"b", &mut appended);
        assert_eq!(appended, "prefix:\"a\\\"b\"");
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
