//! Unicode script detection and the fixed routing rule table.

/// Script families the router distinguishes.
///
/// The router is honest about its limit: script detection cannot separate
/// two languages sharing a script (German versus Turkish Latin text both
/// route as `Latin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Script {
    /// Latin alphabet.
    Latin,
    /// Cyrillic alphabet.
    Cyrillic,
    /// Greek alphabet.
    Greek,
    /// Han ideographs.
    Han,
    /// Hiragana and Katakana.
    Kana,
    /// Hangul.
    Hangul,
    /// Arabic script.
    Arabic,
    /// Hebrew script.
    Hebrew,
    /// Devanagari script.
    Devanagari,
    /// Thai script.
    Thai,
    /// Every other codepoint, including punctuation, digits, and symbols.
    Other,
}

impl Script {
    /// Stable lowercase identifier used in rule tables and telemetry.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Latin => "latin",
            Self::Cyrillic => "cyrillic",
            Self::Greek => "greek",
            Self::Han => "han",
            Self::Kana => "kana",
            Self::Hangul => "hangul",
            Self::Arabic => "arabic",
            Self::Hebrew => "hebrew",
            Self::Devanagari => "devanagari",
            Self::Thai => "thai",
            Self::Other => "other",
        }
    }

    /// Parse a rule-table script identifier.
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "latin" => Self::Latin,
            "cyrillic" => Self::Cyrillic,
            "greek" => Self::Greek,
            "han" => Self::Han,
            "kana" => Self::Kana,
            "hangul" => Self::Hangul,
            "arabic" => Self::Arabic,
            "hebrew" => Self::Hebrew,
            "devanagari" => Self::Devanagari,
            "thai" => Self::Thai,
            "other" => Self::Other,
            _ => return None,
        })
    }
}

/// Map a single codepoint to its script family.
///
/// Ranges follow the Unicode general-script blocks; the table covers the
/// scripts the router distinguishes and falls back to `Other` for
/// everything else, so new Unicode versions cannot panic the router.
pub fn detect_script(character: char) -> Script {
    let code = character as u32;
    match code {
        0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF => Script::Latin,
        0x0400..=0x04FF | 0x0500..=0x052F | 0x2DE0..=0x2DFF => Script::Cyrillic,
        0x0370..=0x03FF | 0x1F00..=0x1FFF => Script::Greek,
        0x2E80..=0x2EFF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => Script::Han,
        0x3040..=0x309F | 0x30A0..=0x30FF | 0x31F0..=0x31FF => Script::Kana,
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => Script::Hangul,
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF => Script::Arabic,
        0x0590..=0x05FF => Script::Hebrew,
        0x0900..=0x097F => Script::Devanagari,
        0x0E00..=0x0E7F => Script::Thai,
        _ => Script::Other,
    }
}

/// Detect the dominant script of a text sample.
///
/// The dominant script is the script with the most assigned codepoints,
/// ignoring `Other` (punctuation, digits, whitespace, symbols). Ties and
/// `Other`-only samples resolve to [`Script::Other`], which rule tables
/// route through their `default` entry. The result is a pure function of
/// the input: no randomness, no environment.
pub fn dominant_script(text: &str) -> Script {
    let mut counts: [usize; 11] = [0; 11];
    for character in text.chars() {
        let script = detect_script(character);
        if script != Script::Other {
            counts[script as usize] += 1;
        }
    }
    let mut best = Script::Other;
    let mut best_count = 0;
    for (index, &count) in counts.iter().enumerate() {
        if count > best_count {
            best_count = count;
            best = match index {
                0 => Script::Latin,
                1 => Script::Cyrillic,
                2 => Script::Greek,
                3 => Script::Han,
                4 => Script::Kana,
                5 => Script::Hangul,
                6 => Script::Arabic,
                7 => Script::Hebrew,
                8 => Script::Devanagari,
                9 => Script::Thai,
                _ => Script::Other,
            };
        }
    }
    best
}

/// Fixed script → sibling-alias routing table.
#[derive(Debug, Clone, PartialEq)]
pub struct ScriptRuleTable {
    rules: Vec<(Script, String)>,
    default: Option<String>,
}

impl ScriptRuleTable {
    /// Build a rule table from `script=alias` pairs plus a `default=alias`.
    ///
    /// Accepted script identifiers are the lowercase names from
    /// [`Script::as_str`]; later duplicates of the same script replace
    /// earlier ones, so the table stays a function.
    pub fn parse(spec: &str) -> Result<Self, super::RouterScriptError> {
        let mut rules: Vec<(Script, String)> = Vec::new();
        let mut default = None;
        for token in spec.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let (key, alias) = token.split_once('=').ok_or_else(|| {
                super::RouterScriptError::InvalidRules(format!(
                    "expected `script=alias`, found `{token}`"
                ))
            })?;
            let alias = alias.trim().to_owned();
            if alias.is_empty() {
                return Err(super::RouterScriptError::InvalidRules(format!(
                    "empty alias in `{token}`"
                )));
            }
            if key == "default" {
                default = Some(alias);
            } else {
                let script = Script::parse(key).ok_or_else(|| {
                    super::RouterScriptError::InvalidRules(format!(
                        "unknown script `{key}` in `{token}`"
                    ))
                })?;
                rules.retain(|(existing, _)| *existing != script);
                rules.push((script, alias));
            }
        }
        Ok(Self { rules, default })
    }

    /// Route a detected script to its sibling alias.
    pub fn route(&self, script: Script) -> Option<&str> {
        self.rules
            .iter()
            .find(|(candidate, _)| *candidate == script)
            .map(|(_, alias)| alias.as_str())
            .or(self.default.as_deref())
    }

    /// Every alias the table can produce, including the default.
    pub fn referenced_aliases(&self) -> Vec<&str> {
        let mut aliases: Vec<&str> = self.rules.iter().map(|(_, alias)| alias.as_str()).collect();
        if let Some(default) = &self.default {
            aliases.push(default);
        }
        aliases.sort_unstable();
        aliases.dedup();
        aliases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_detection_covers_the_documented_families() {
        assert_eq!(detect_script('a'), Script::Latin);
        assert_eq!(detect_script('Ж'), Script::Cyrillic);
        assert_eq!(detect_script('Ω'), Script::Greek);
        assert_eq!(detect_script('国'), Script::Han);
        assert_eq!(detect_script('あ'), Script::Kana);
        assert_eq!(detect_script('한'), Script::Hangul);
        assert_eq!(detect_script('ع'), Script::Arabic);
        assert_eq!(detect_script('א'), Script::Hebrew);
        assert_eq!(detect_script('न'), Script::Devanagari);
        assert_eq!(detect_script('ก'), Script::Thai);
        assert_eq!(detect_script('7'), Script::Other);
        assert_eq!(detect_script('!'), Script::Other);
    }

    #[test]
    fn dominant_script_counts_assigned_codepoints_only() {
        assert_eq!(dominant_script("Fictional incident record."), Script::Latin);
        assert_eq!(dominant_script("Сводка инцидента"), Script::Cyrillic);
        assert_eq!(dominant_script("事故レポート"), Script::Kana);
        assert_eq!(dominant_script("事故报告"), Script::Han);
        assert_eq!(dominant_script("사고 기록"), Script::Hangul);
        // Digits and punctuation never win.
        assert_eq!(dominant_script("1234567890 !!!"), Script::Other);
        // A single Han character inside Latin text stays Latin.
        assert_eq!(dominant_script("incident 国 record"), Script::Latin);
    }

    #[test]
    fn rule_table_parses_and_routes_deterministically() {
        let table = ScriptRuleTable::parse(
            "latin=decoder-letter-native,cyrillic=encoder-nli-native,default=decoder-letter-native",
        )
        .expect("parse rules");
        assert_eq!(table.route(Script::Latin), Some("decoder-letter-native"));
        assert_eq!(table.route(Script::Cyrillic), Some("encoder-nli-native"));
        assert_eq!(table.route(Script::Han), Some("decoder-letter-native"));
        assert_eq!(table.route(Script::Other), Some("decoder-letter-native"));
        assert_eq!(
            table.referenced_aliases(),
            vec!["decoder-letter-native", "encoder-nli-native"]
        );
    }

    #[test]
    fn rule_table_rejects_malformed_rules() {
        assert!(ScriptRuleTable::parse("latin").is_err());
        assert!(ScriptRuleTable::parse("klingon=x").is_err());
        assert!(ScriptRuleTable::parse("latin=").is_err());
        assert!(ScriptRuleTable::parse("latin=x,default=").is_err());
        // No default: unrouted scripts return None.
        let table = ScriptRuleTable::parse("latin=x").expect("parse");
        assert_eq!(table.route(Script::Cyrillic), None);
    }
}
