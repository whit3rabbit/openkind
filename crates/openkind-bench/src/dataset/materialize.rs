//! Dataset materialization: convert installed parquet shards into labeled
//! workload rows using the frozen templates.
//!
//! The ported task semantics follow `jev_benchmarking` (MIT): one request per
//! example, gold on every row, deterministic hash-ordered rows so smaller
//! `--limit` values select subsets of larger ones, and identical-request
//! deduplication. Split discipline follows the paper: `dev` is for prompt and
//! threshold work, `eval` is the reported split.

use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use openkind_datasets::rows::read_parquet_rows;
use openkind_datasets::InstalledDataset;
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::templates as t;
use crate::workload::{DecisionOption, NoulCriteriaWire, QuestionSpec, WorkloadRow};

/// One labeled dataset row. The flattened question keeps the normal harness
/// wire shape; the label fields ride along as extra keys the plain `score`
/// command ignores.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetRow {
    #[serde(flatten)]
    pub row: WorkloadRow,
    pub gold: String,
    pub task: String,
    pub source_group: String,
    pub split: String,
    /// Upstream configuration (subject or split variant).
    pub subset: String,
}

pub struct Materialized {
    pub rows: Vec<DatasetRow>,
    pub skipped_duplicates: usize,
    pub skipped_rows: usize,
}

/// Materialize one split of an installed dataset.
///
/// # Errors
/// Returns an error for unknown datasets, missing shards, or rows the
/// template cannot render.
pub fn materialize(
    installed: &InstalledDataset,
    split: &str,
    limit: Option<usize>,
) -> Result<Materialized> {
    let entry = &installed.entry;
    let mut raw = Vec::new();
    for config in &entry.configs {
        let files = entry.split_files(config, split);
        if files.is_empty() {
            bail!(
                "dataset {} has no {split} files for config {config}",
                entry.name
            );
        }
        for file in files {
            let rows = read_parquet_rows(&installed.file_path(file))
                .with_context(|| format!("read {}", file.path))?;
            raw.push((config.clone(), rows));
        }
    }
    let banking_intents = if entry.name == "banking77" {
        Some(banking77_intents(installed)?)
    } else {
        None
    };
    let built = match entry.name.as_str() {
        "sst2" => sst2(&raw),
        "ag_news" => ag_news(&raw),
        "banking77" => banking77(&raw, banking_intents.as_ref().expect("intents resolved")),
        "clinc150" => clinc150(&raw),
        "arc" | "commonsense_qa" => labeled_choices(&raw, &entry.name),
        "hellaswag" => hellaswag(&raw),
        "winogrande" => winogrande(&raw),
        "paws" => paws(&raw),
        "boolq" => boolq(&raw),
        "stsb" => stsb(&raw),
        "sst5" => sst5(&raw),
        other => bail!("no materializer for dataset {other}"),
    }?;
    let total_upstream = raw.iter().map(|(_, rows)| rows.len()).sum::<usize>();
    // Identical-request dedup (paper practice): the exact state and question
    // bytes decide, never the row id.
    let mut seen = HashSet::new();
    let mut unique = Vec::with_capacity(built.len());
    let mut skipped_duplicates = 0;
    for example in built {
        let mut request = Sha256::new();
        request.update(
            serde_json::to_string(&example.state)
                .unwrap_or_default()
                .as_bytes(),
        );
        request.update(
            serde_json::to_string(&question_wire(&example))
                .unwrap_or_default()
                .as_bytes(),
        );
        if seen.insert(format!("{:x}", request.finalize())) {
            unique.push(example);
        } else {
            skipped_duplicates += 1;
        }
    }
    // Deterministic hash ordering: `--limit N` selects a subset of any larger
    // limit, matching the upstream sampling contract.
    unique.sort_by_key(|example| {
        let mut hasher = Sha256::new();
        hasher.update(example.uid.as_bytes());
        format!("{:x}", hasher.finalize())
    });
    let mut rows: Vec<DatasetRow> = unique
        .into_iter()
        .map(|example| {
            Ok(DatasetRow {
                row: WorkloadRow {
                    id: format!("{}/{}", entry.name, example.uid),
                    state: serde_json::from_value(example.state)
                        .with_context(|| format!("state for {}", example.uid))?,
                    question: example.question,
                },
                gold: example.gold,
                task: entry.name.clone(),
                // Single-example groups give the example-level bootstrap the
                // paper reports; grouped datasets can override this later.
                source_group: format!("{}/{}", entry.name, example.uid),
                split: split.to_owned(),
                subset: example.subset,
            })
        })
        .collect::<Result<Vec<DatasetRow>>>()?;
    let skipped_rows = total_upstream.saturating_sub(rows.len() + skipped_duplicates);
    if let Some(limit) = limit {
        rows.truncate(limit);
    }
    Ok(Materialized {
        skipped_rows,
        rows,
        skipped_duplicates,
    })
}

/// The banking77 option list is derived from the eval split's `label_text`
/// values, sorted and deduplicated, exactly as upstream derives it; both
/// splits must be offered the same option set.
fn banking77_intents(installed: &InstalledDataset) -> Result<Vec<String>> {
    let entry = &installed.entry;
    let mut intents = HashSet::new();
    for config in &entry.configs {
        for file in entry.split_files(config, "eval") {
            for row in read_parquet_rows(&installed.file_path(file))
                .with_context(|| format!("read {}", file.path))?
            {
                intents.insert(text(&row, "label_text")?.to_owned());
            }
        }
    }
    let mut intents: Vec<String> = intents.into_iter().collect();
    intents.sort();
    anyhow::ensure!(intents.len() >= 2, "banking77 intents did not resolve");
    Ok(intents)
}

struct Example {
    uid: String,
    state: Value,
    question: QuestionSpec,
    gold: String,
    subset: String,
}

/// The question half of the dedup key, in wire shape.
fn question_wire(example: &Example) -> Value {
    match &example.question {
        QuestionSpec::Noul { text, criteria } => serde_json::json!({
            "primitive": "noul", "text": text, "criteria": criteria,
        }),
        QuestionSpec::Choice { text, options } => serde_json::json!({
            "primitive": "choice", "text": text, "options": options,
        }),
        QuestionSpec::Score { text, levels } => serde_json::json!({
            "primitive": "score", "text": text, "levels": levels,
        }),
    }
}

fn text<'a>(row: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    row.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("column {key} is not a string"))
}

fn integer(row: &Map<String, Value>, key: &str) -> Result<i64> {
    row.get(key)
        .and_then(Value::as_i64)
        .with_context(|| format!("column {key} is not an integer"))
}

fn number(row: &Map<String, Value>, key: &str) -> Result<f64> {
    row.get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("column {key} is not a number"))
}

fn bool_field(row: &Map<String, Value>, key: &str) -> Result<bool> {
    row.get(key)
        .and_then(Value::as_bool)
        .with_context(|| format!("column {key} is not a boolean"))
}

fn choice_question(instruction: &str, options: Vec<DecisionOption>) -> QuestionSpec {
    QuestionSpec::Choice {
        text: instruction.to_owned(),
        options,
    }
}

fn described(instruction: &str, criteria: &[(&str, &str)]) -> QuestionSpec {
    choice_question(
        instruction,
        criteria
            .iter()
            .map(|(id, description)| DecisionOption {
                id: (*id).to_owned(),
                description: Some((*description).to_owned()),
            })
            .collect(),
    )
}

fn plain(instruction: &str, ids: &[String]) -> QuestionSpec {
    choice_question(
        instruction,
        ids.iter()
            .map(|id| DecisionOption {
                id: id.clone(),
                description: None,
            })
            .collect(),
    )
}

fn indexed(config: &str, index: usize) -> String {
    if config == "default" {
        index.to_string()
    } else {
        format!("{config}/{index}")
    }
}

fn sst2(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let Some(entry) = t::SENTIMENT_2.get(integer(row, "label")?.unsigned_abs() as usize)
            else {
                continue;
            };
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({ "review": text(row, "sentence")?.trim() }),
                question: described(t::SST2_INSTRUCTION, &t::SENTIMENT_2),
                gold: entry.0.to_owned(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn ag_news(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let Some(gold) = t::AG_NEWS_LABELS.get(integer(row, "label")?.unsigned_abs() as usize)
            else {
                continue;
            };
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({ "article": text(row, "text")? }),
                question: described(t::AG_NEWS_INSTRUCTION, &t::AG_NEWS_CRITERIA),
                gold: (*gold).to_owned(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn banking77(
    raw: &[(String, Vec<Map<String, Value>>)],
    intents: &[String],
) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let gold = text(row, "label_text")?.to_owned();
            if !intents.contains(&gold) {
                continue;
            }
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({ "query": text(row, "text")? }),
                question: plain(t::BANKING77_INSTRUCTION, intents),
                gold,
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn clinc150(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let options: Vec<DecisionOption> = t::CLINC150_INTENTS
        .iter()
        .map(|intent| DecisionOption {
            id: (*intent).to_owned(),
            description: (*intent == t::CLINC150_OOS.0).then(|| t::CLINC150_OOS.1.to_owned()),
        })
        .collect();
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let Some(gold) =
                t::CLINC150_INTENTS.get(integer(row, "intent")?.unsigned_abs() as usize)
            else {
                continue;
            };
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({ "utterance": text(row, "text")? }),
                question: choice_question(t::CLINC150_INSTRUCTION, options.clone()),
                gold: (*gold).to_owned(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

/// ARC and CommonsenseQA: options arrive with their own labels and an
/// `answerKey` (`tasks/multiple_choice.py`, `_LabeledChoices`).
fn labeled_choices(raw: &[(String, Vec<Map<String, Value>>)], name: &str) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let Some(choices) = row.get("choices").and_then(Value::as_object) else {
                bail!("{name}: choices column is not a struct");
            };
            let labels = choices
                .get("label")
                .and_then(Value::as_array)
                .context("choices.label is missing")?;
            let texts = choices
                .get("text")
                .and_then(Value::as_array)
                .context("choices.text is missing")?;
            let answer_key = text(row, "answerKey")?.to_owned();
            let mut options = Vec::new();
            for (label, option_text) in labels.iter().zip(texts.iter()) {
                options.push(DecisionOption {
                    id: label
                        .as_str()
                        .context("choice label is not a string")?
                        .to_owned(),
                    description: Some(
                        option_text
                            .as_str()
                            .context("choice text is not a string")?
                            .to_owned(),
                    ),
                });
            }
            if !options.iter().any(|option| option.id == answer_key) {
                continue;
            }
            out.push(Example {
                uid: format!("{config}/{index}"),
                state: serde_json::json!({ "question": text(row, "question")? }),
                question: choice_question(t::MC_INSTRUCTION, options),
                gold: answer_key,
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

/// Port of `_hellaswag_clean` (`tasks/multiple_choice.py`): drop "[title]"
/// markup and bracketed groups, collapse doubled spaces and dot runs.
pub(crate) fn hellaswag_clean(input: &str) -> String {
    let step = input.trim().replace(" [title]", ". ");
    // Remove each "[...]" group that has a closing bracket (regex `\[.*?\]`).
    let characters: Vec<char> = step.chars().collect();
    let mut without_brackets = String::with_capacity(step.len());
    let mut index = 0;
    while index < characters.len() {
        if characters[index] == '[' {
            if let Some(close) = characters[index + 1..].iter().position(|c| *c == ']') {
                index += close + 2;
                continue;
            }
        }
        without_brackets.push(characters[index]);
        index += 1;
    }
    let single_spaced = without_brackets.replace("  ", " ");
    let mut collapsed = String::with_capacity(single_spaced.len());
    let mut dots = 0;
    for character in single_spaced.chars() {
        if character == '.' {
            dots += 1;
        } else {
            if dots > 0 {
                collapsed.push('.');
            }
            dots = 0;
            collapsed.push(character);
        }
    }
    if dots > 0 {
        collapsed.push('.');
    }
    collapsed.trim().to_owned()
}

fn capitalize(sentence: &str) -> String {
    let mut characters = sentence.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

fn hellaswag(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let Ok(label) = text(row, "label")?.parse::<usize>() else {
                continue;
            };
            let context = hellaswag_clean(&format!(
                "{}: {} {}",
                text(row, "activity_label")?,
                text(row, "ctx_a")?,
                capitalize(text(row, "ctx_b")?)
            ));
            let endings = row
                .get("endings")
                .and_then(Value::as_array)
                .context("endings is missing")?;
            if endings.len() != 4 || label >= 4 {
                continue;
            }
            let keys = t::option_keys(4);
            let options: Vec<DecisionOption> = keys
                .iter()
                .zip(endings.iter())
                .map(|(id, ending)| DecisionOption {
                    id: id.clone(),
                    description: Some(hellaswag_clean(ending.as_str().unwrap_or_default())),
                })
                .collect();
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({ "context": context }),
                question: choice_question(t::HELLASWAG_INSTRUCTION, options),
                gold: keys[label].clone(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn winogrande(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let answer = text(row, "answer")?;
            if answer != "1" && answer != "2" {
                continue;
            }
            let options = vec![
                DecisionOption {
                    id: "A".to_owned(),
                    description: Some(text(row, "option1")?.to_owned()),
                },
                DecisionOption {
                    id: "B".to_owned(),
                    description: Some(text(row, "option2")?.to_owned()),
                },
            ];
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({ "sentence": text(row, "sentence")? }),
                question: choice_question(t::WINOGRANDE_INSTRUCTION, options),
                gold: if answer == "1" { "A" } else { "B" }.to_owned(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn paws(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let label = integer(row, "label")?;
            if label != 0 && label != 1 {
                continue;
            }
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({
                    "sentence_1": text(row, "sentence1")?,
                    "sentence_2": text(row, "sentence2")?,
                }),
                question: QuestionSpec::Noul {
                    text: t::PAWS_INSTRUCTION.to_owned(),
                    criteria: Some(NoulCriteriaWire {
                        r#true: t::PAWS_CRITERIA.0.to_owned(),
                        r#false: t::PAWS_CRITERIA.1.to_owned(),
                    }),
                },
                gold: if label == 1 { "true" } else { "false" }.to_owned(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn boolq(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let mut question = capitalize(text(row, "question")?.trim());
            if !question.ends_with('?') {
                question.push('?');
            }
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({
                    "passage": text(row, "passage")?,
                    "question": question,
                }),
                question: QuestionSpec::Noul {
                    text: t::BOOLQ_INSTRUCTION.to_owned(),
                    criteria: None,
                },
                gold: if bool_field(row, "answer")? {
                    "true"
                } else {
                    "false"
                }
                .to_owned(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn stsb(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            // The upstream copy stores scores normalized to 0-1; gold is on
            // the original 0-5 level scale.
            let gold = number(row, "score")? * 5.0;
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({
                    "sentence_1": text(row, "sentence1")?,
                    "sentence_2": text(row, "sentence2")?,
                }),
                question: QuestionSpec::Score {
                    text: t::STSB_INSTRUCTION.to_owned(),
                    levels: t::STSB_LEVELS
                        .iter()
                        .map(|level| (*level).to_owned())
                        .collect(),
                },
                gold: format!("{gold}"),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}

fn sst5(raw: &[(String, Vec<Map<String, Value>>)]) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (config, rows) in raw {
        for (index, row) in rows.iter().enumerate() {
            let label = integer(row, "label")?;
            if !(0..5).contains(&label) {
                continue;
            }
            out.push(Example {
                uid: indexed(config, index),
                state: serde_json::json!({ "sentence": text(row, "text")? }),
                question: QuestionSpec::Score {
                    text: t::SST5_INSTRUCTION.to_owned(),
                    levels: t::SST5_LEVELS
                        .iter()
                        .map(|level| (*level).to_owned())
                        .collect(),
                },
                gold: label.to_string(),
                subset: config.clone(),
            });
        }
    }
    Ok(out)
}
