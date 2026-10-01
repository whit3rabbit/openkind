//! Prompt renderer and letter-token contract for the pinned JevK5 profile.
//!
//! The bytes replicate the reference `jevk5` runtime exactly: the pinned
//! system instruction, the decision payload as a Python-`json.dumps`-shaped
//! object (`ensure_ascii=False`, `", "`/`": "` separators), and Qwen3.5's
//! chat template rendered with thinking off. Because the wire `criteria` map
//! is a hash map, options are rendered in the profile's deterministic sorted
//! label order — a declared renderer difference from the reference runtime,
//! which preserves the caller's insertion order.

use tokenizers::Tokenizer;

use crate::families::support::FamilyError;

use super::MAX_OPTIONS_PER_PASS;

pub(crate) const SYSTEM_INSTRUCTION: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// Qwen3.5 chat template with thinking off, rendered once and pinned so the
/// native path is token-identical to the reference `transformers` rendering.
const CHAT_TEMPLATE: &str = "<|im_start|>system\n{system}<|im_end|>\n\
<|im_start|>user\n{user}<|im_end|>\n\
<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Default Noul criterion text used by the reference runtime when the caller
/// supplies no explicit criteria.
pub(crate) const NOUL_DEFAULT_DESCRIPTIONS: [&str; 2] =
    ["The proposition is false.", "The proposition is true."];

/// One rendered forward pass: prompt ids plus the letter id per option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPass {
    prompt_ids: Vec<u32>,
    letter_ids: Vec<u32>,
}

impl RenderedPass {
    /// Prompt token ids ending at the assistant answer slot.
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    /// One letter-token id per option in this pass, in option order.
    pub fn letter_ids(&self) -> &[u32] {
        &self.letter_ids
    }
}

/// Offline renderer for the pinned letter-pass profiles of this family.
pub struct Jevk5Renderer {
    tokenizer: Tokenizer,
    letter_token_ids: Vec<u32>,
    /// Frozen maximum rendered prompt length per pass of the loaded profile.
    max_sequence_tokens: usize,
}

impl Jevk5Renderer {
    /// Load a digest-verified tokenizer and resolve the letter-token contract.
    ///
    /// Every option letter must tokenize to exactly one token; a tokenizer
    /// that splits a letter fails the load instead of corrupting the readout.
    pub fn load(
        tokenizer_path: &std::path::Path,
        max_sequence_tokens: usize,
    ) -> Result<Self, FamilyError> {
        let bytes = std::fs::read(tokenizer_path).map_err(|source| FamilyError::Io {
            path: tokenizer_path.to_path_buf(),
            source,
        })?;
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let letter_token_ids = super::LETTERS
            .iter()
            .map(|letter| {
                let text = (*letter as char).to_string();
                let encoding = tokenizer
                    .encode(text.as_str(), false)
                    .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
                let ids = encoding.get_ids();
                if ids.len() != 1 {
                    return Err(FamilyError::ContractMismatch {
                        field: "letter_token_contract",
                        expected: format!("letter {text} tokenizes to exactly one token"),
                        actual: format!("letter {text} tokenizes to {} tokens", ids.len()),
                    });
                }
                Ok(ids[0])
            })
            .collect::<Result<Vec<u32>, FamilyError>>()?;
        if letter_token_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != letter_token_ids.len()
        {
            return Err(FamilyError::ContractMismatch {
                field: "letter_token_contract",
                expected: "distinct letter tokens".to_owned(),
                actual: "duplicate letter tokens".to_owned(),
            });
        }
        Ok(Self {
            tokenizer,
            letter_token_ids,
            max_sequence_tokens,
        })
    }

    /// Render one knockout pass into a prompt and its per-option letter ids.
    ///
    /// `state` is the wire state as a JSON value (`evidence`), `criterion`
    /// the question instruction text, and `options` the `(label,
    /// description)` pairs scored by this pass in order.
    pub fn render_pass(
        &self,
        state: &serde_json::Value,
        criterion: &str,
        options: &[(String, String)],
    ) -> Result<RenderedPass, FamilyError> {
        if criterion.trim().is_empty() {
            return Err(FamilyError::InvalidInput(
                "question instruction must contain non-whitespace text".to_owned(),
            ));
        }
        if options.len() < super::MIN_CANDIDATES {
            return Err(FamilyError::InvalidInput(format!(
                "a pass needs at least {} options, found {}",
                super::MIN_CANDIDATES,
                options.len()
            )));
        }
        if options.len() > MAX_OPTIONS_PER_PASS {
            return Err(FamilyError::InvalidInput(format!(
                "a pass scores at most {MAX_OPTIONS_PER_PASS} options, found {}",
                options.len()
            )));
        }
        if options
            .iter()
            .any(|(_, description)| description.trim().is_empty())
        {
            return Err(FamilyError::InvalidInput(
                "option descriptions must contain non-whitespace text".to_owned(),
            ));
        }
        // The reference maps every option to `f"{label}: {description}"`
        // (jevk5 `decision_options`) and serializes each option object with
        // `letter` before `description`; the bytes are built by hand so the
        // sorted wire map cannot reorder them.
        let payload_options: String = options
            .iter()
            .enumerate()
            .map(|(index, (label, description))| {
                let rendered_description = python_json(&serde_json::Value::String(format!(
                    "{label}: {description}"
                )));
                format!(
                    "{{\"letter\": \"{}\", \"description\": {}}}",
                    super::LETTERS[index] as char,
                    rendered_description
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let payload = format!(
            "{{\"evidence\": {}, \"criterion\": {}, \"options\": [{}]}}",
            python_json(state),
            python_json(&serde_json::Value::String(criterion.to_owned())),
            payload_options,
        );
        let prompt = CHAT_TEMPLATE
            .replace("{system}", SYSTEM_INSTRUCTION)
            .replace("{user}", &payload);
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.len() > self.max_sequence_tokens {
            return Err(FamilyError::InvalidInput(format!(
                "rendered prompt length {} exceeds frozen maximum {}; truncation is forbidden",
                prompt_ids.len(),
                self.max_sequence_tokens
            )));
        }
        let letter_ids = options
            .iter()
            .enumerate()
            .map(|(index, _)| self.letter_token_ids[index])
            .collect();
        Ok(RenderedPass {
            prompt_ids,
            letter_ids,
        })
    }

    /// Token count of one rendered pass for usage accounting.
    pub fn pass_token_count(&self, pass: &RenderedPass) -> u32 {
        u32::try_from(pass.prompt_ids().len()).unwrap_or(u32::MAX)
    }
}

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

fn python_float(value: f64) -> String {
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

fn python_string(text: &str) -> String {
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
