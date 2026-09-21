//! Decision workload model: JSONL rows, digest-checked loading, state
//! grouping, and Jev request construction.
//!
//! A workload row is one decision: `{"id", "state", "primitive", ...}` where
//! `state` is a Jev `State` (string, object, or array) and the question spec
//! is flattened onto the row behind its `primitive` tag (`noul`, `choice`, or
//! `score`). The shape is row-per-decision so it stays directly comparable to
//! prior-art scored workloads (see `docs/BENCHMARKS.md`).

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use opendecision_backends::qwen35::SEMANTIC_NONE_OPTION;
use opendecision_core::{
    ChoiceQuestion, NoulCriteria, NoulQuestion, Question, ScoreQuestion, State, SystemRequest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Default instructions when a row omits the question text.
const DEFAULT_INSTRUCTIONS: &str = "Decide.";

/// Fallback description injected for Choice rows without an explicit
/// semantic-none option, so native evaluation always keeps none mass on wire.
const DEFAULT_NONE_DESCRIPTION: &str = "None of the listed options applies.";

/// One scored option of a Choice question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    /// Option key returned in the choice answer.
    pub id: String,
    /// Optional rubric description; falls back to the option id.
    #[serde(default)]
    pub description: Option<String>,
}

/// Wire shape of `noul` boolean criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulCriteriaWire {
    /// Meaning of a true/yes decision.
    #[serde(rename = "true")]
    pub r#true: String,
    /// Meaning of a false/no decision.
    #[serde(rename = "false")]
    pub r#false: String,
}

/// Typed question spec inside one workload row.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "primitive", rename_all = "lowercase")]
pub enum QuestionSpec {
    /// Boolean probability question.
    Noul {
        /// Instructions text; defaults to [`DEFAULT_INSTRUCTIONS`].
        #[serde(default = "default_instructions")]
        text: String,
        /// Optional true/false criterion descriptions.
        #[serde(default)]
        criteria: Option<NoulCriteriaWire>,
    },
    /// Categorical pick-one question.
    Choice {
        /// Instructions text; defaults to [`DEFAULT_INSTRUCTIONS`].
        #[serde(default = "default_instructions")]
        text: String,
        /// Scored options; `__none__` is injected when absent.
        options: Vec<DecisionOption>,
    },
    /// Ordinal rating question.
    Score {
        /// Instructions text; defaults to [`DEFAULT_INSTRUCTIONS`].
        #[serde(default = "default_instructions")]
        text: String,
        /// Ordered rubric level descriptions (at least two).
        levels: Vec<String>,
    },
}

fn default_instructions() -> String {
    DEFAULT_INSTRUCTIONS.to_owned()
}

/// One workload row: a single decision over one state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadRow {
    /// Decision id; unique across the workload and used as the Jev question id.
    pub id: String,
    /// State to evaluate (string, object, or array).
    pub state: State,
    /// Typed question.
    #[serde(flatten)]
    pub question: QuestionSpec,
}

impl WorkloadRow {
    /// The Jev question for this row.
    ///
    /// # Errors
    /// Returns an error when a score rubric has fewer than two levels or a
    /// choice row defines `__none__` with an empty description.
    pub fn question(&self) -> Result<Question> {
        match &self.question {
            QuestionSpec::Noul { text, criteria } => Ok(Question::Noul(NoulQuestion {
                instructions: serde_json::json!(text),
                criteria: criteria.as_ref().map(|criteria| NoulCriteria {
                    r#true: criteria.r#true.clone(),
                    r#false: criteria.r#false.clone(),
                }),
            })),
            QuestionSpec::Choice { text, options } => {
                if options.is_empty() {
                    bail!("row `{}`: choice question has no options", self.id);
                }
                let mut map: HashMap<String, Option<String>> = HashMap::new();
                for option in options {
                    if option.id == SEMANTIC_NONE_OPTION {
                        let description = option
                            .description
                            .clone()
                            .unwrap_or_else(|| DEFAULT_NONE_DESCRIPTION.to_owned());
                        if description.trim().is_empty() {
                            bail!(
                                "row `{}`: `{SEMANTIC_NONE_OPTION}` must have a non-empty \
                                 description",
                                self.id
                            );
                        }
                    }
                    if map
                        .insert(option.id.clone(), option.description.clone())
                        .is_some()
                    {
                        bail!("row `{}`: duplicate option id `{}`", self.id, option.id);
                    }
                }
                map.entry(SEMANTIC_NONE_OPTION.to_owned())
                    .or_insert_with(|| Some(DEFAULT_NONE_DESCRIPTION.to_owned()));
                Ok(Question::Choice(ChoiceQuestion {
                    instructions: serde_json::json!(text),
                    criteria: map,
                }))
            }
            QuestionSpec::Score { text, levels } => {
                if levels.len() < 2 {
                    bail!(
                        "row `{}`: score rubric needs at least two levels, got {}",
                        self.id,
                        levels.len()
                    );
                }
                Ok(Question::Score(ScoreQuestion {
                    instructions: serde_json::json!(text),
                    criteria: levels.clone(),
                }))
            }
        }
    }

    /// Canonical grouping key: rows sharing this key share one state root and
    /// can be evaluated in one grouped Jev request.
    pub fn group_key(&self) -> Result<String> {
        serde_json::to_string(&self.state).context("serialize state for grouping")
    }
}

/// A loaded workload: rows plus their fixture digest.
#[derive(Debug, Clone)]
pub struct Workload {
    /// Rows in file order.
    pub rows: Vec<WorkloadRow>,
    /// SHA-256 of the raw workload bytes, recorded in every run summary.
    pub sha256: String,
}

impl fmt::Display for Workload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} rows (sha256 {})", self.rows.len(), self.sha256)
    }
}

/// Read and validate a JSONL workload file.
///
/// # Errors
/// Returns an error for unreadable files or parse/validation failures from
/// [`parse_workload`].
pub fn load_workload(path: &Path) -> Result<Workload> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_workload(&path.display().to_string(), &raw)
}

/// Parse and validate raw workload bytes.
///
/// # Errors
/// Returns an error for invalid UTF-8, malformed JSONL, duplicate row ids, an
/// empty workload, or per-row validation failures.
pub fn parse_workload(label: &str, raw: &[u8]) -> Result<Workload> {
    let sha256 = hex(&Sha256::digest(raw));
    let text = String::from_utf8(raw.to_vec()).with_context(|| format!("decode {label}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: WorkloadRow =
            serde_json::from_str(line).with_context(|| format!("{label} line {}", index + 1))?;
        // Fail early on rows the wire validator would reject later anyway.
        row.question()?;
        rows.push(row);
    }
    if rows.is_empty() {
        bail!("{label}: workload has no rows");
    }
    let mut seen = std::collections::HashSet::new();
    for row in &rows {
        if !seen.insert(row.id.clone()) {
            bail!("{label}: duplicate decision id `{}`", row.id);
        }
    }
    Ok(Workload { rows, sha256 })
}

/// Groups of row indices sharing one state, in first-appearance order.
pub fn state_groups(rows: &[WorkloadRow]) -> Result<Vec<Vec<usize>>> {
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, row) in rows.iter().enumerate() {
        let key = row.group_key()?;
        let entry = groups.entry(key.clone()).or_default();
        if entry.is_empty() {
            order.push(key);
        }
        entry.push(index);
    }
    Ok(order
        .into_iter()
        .map(|key| groups.remove(&key).expect("recorded key"))
        .collect())
}

/// Build one Jev request covering `group` (row indices into `rows`).
///
/// All rows in a group must share one state so the state-first renderer
/// produces a single root prefix; [`state_groups`] guarantees this.
pub fn build_request(model: &str, rows: &[WorkloadRow], group: &[usize]) -> Result<SystemRequest> {
    let first = &rows[*group.first().expect("non-empty group")];
    let mut questions = HashMap::with_capacity(group.len());
    for &index in group {
        let row = &rows[index];
        if row.group_key()? != first.group_key()? {
            bail!(
                "grouping violation: row `{}` does not share the group state",
                row.id
            );
        }
        questions.insert(row.id.clone(), row.question()?);
    }
    Ok(SystemRequest {
        state: first.state.clone(),
        model: model.to_owned(),
        questions,
    })
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
