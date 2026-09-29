//! Validate the mapping before reconstructing request-bound Jev answers.

use super::*;
use arrow_array::{Array, ArrayRef, StructArray, UInt8Array};
use openkind_core::{
    Answer, ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, ResponseContract, ScoreAnswer,
    ScoreQuestion, SystemResponse, Usage,
};
use std::collections::HashSet;

fn malformed(message: impl std::fmt::Display) -> ApiError {
    ApiError::Internal(format!("invalid Arrow mapping: {message}"))
}

fn strings(field: &Field, key: &str) -> Result<Vec<String>, ApiError> {
    let raw = field
        .metadata()
        .get(key)
        .ok_or_else(|| malformed(format!("missing `{key}` metadata")))?;
    serde_json::from_str(raw).map_err(malformed)
}

fn non_null(field: &Field, array: &ArrayRef) -> Result<(), ApiError> {
    if field.is_nullable() || array.null_count() != 0 {
        return Err(malformed(
            "nullable fields or null values are not supported",
        ));
    }
    match field.data_type() {
        DataType::Struct(children) => {
            let values = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| malformed("expected struct"))?;
            for (child, values) in children.iter().zip(values.columns()) {
                non_null(child, values)?;
            }
        }
        DataType::FixedSizeList(item, _) => {
            if item.is_nullable() {
                return Err(malformed("nullable probability items"));
            }
            let values = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| malformed("expected fixed-size probabilities"))?;
            // Slices may share backing storage containing unrelated nulls.
            // Only the list values visible in this batch affect its answers.
            for row in 0..values.len() {
                if values.value(row).null_count() != 0 {
                    return Err(malformed("null probability value"));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

enum Values<'a> {
    Noul(&'a Float64Array),
    Choice {
        labels: Vec<String>,
        index: &'a UInt8Array,
        confidence: &'a Float64Array,
        probabilities: &'a FixedSizeListArray,
    },
    Score {
        legend: Vec<String>,
        score: &'a Float64Array,
        confidence: &'a Float64Array,
        probabilities: &'a FixedSizeListArray,
    },
}

fn float64(array: &ArrayRef) -> Result<&Float64Array, ApiError> {
    array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| malformed("expected float64"))
}

impl Values<'_> {
    fn answer(&self, row: usize) -> Result<Answer, ApiError> {
        Ok(match self {
            Self::Noul(values) => Answer::Noul(NoulAnswer {
                noul: values.value(row),
            }),
            Self::Choice {
                labels,
                index,
                confidence,
                probabilities,
            } => {
                let choice = labels
                    .get(usize::from(index.value(row)))
                    .ok_or_else(|| malformed("choice index outside labels"))?
                    .clone();
                let row_probs = probabilities.value(row);
                let values = float64(&row_probs)?;
                Answer::Choice(ChoiceAnswer {
                    choice,
                    confidence: confidence.value(row),
                    probabilities: labels
                        .iter()
                        .enumerate()
                        .map(|(i, label)| (label.clone(), values.value(i)))
                        .collect(),
                })
            }
            Self::Score {
                legend,
                score,
                confidence,
                probabilities,
            } => {
                let row_probs = probabilities.value(row);
                let values = float64(&row_probs)?;
                Answer::Score(ScoreAnswer {
                    score: score.value(row),
                    confidence: confidence.value(row),
                    probabilities: legend
                        .iter()
                        .enumerate()
                        .map(|(i, _)| (i.to_string(), values.value(i)))
                        .collect(),
                    legend: legend
                        .iter()
                        .enumerate()
                        .map(|(i, label)| (i.to_string(), label.clone()))
                        .collect(),
                })
            }
        })
    }
}

/// Reconstruct Jev answers from a version-1 batch produced by this module.
///
/// Validates column identities, schema types, non-null values, metadata and
/// Jev numeric rules. Malformed batches return an error without substituting
/// defaults or discarding duplicate keys. Valid sliced arrays are supported.
/// Row `i` contains the answers for `states[i]`, keyed by question ID.
pub fn answers_from_batch(batch: &RecordBatch) -> Result<Vec<BTreeMap<String, Answer>>, ApiError> {
    let schema = batch.schema();
    if schema
        .metadata()
        .get(META_ARROW_VERSION)
        .map(String::as_str)
        != Some(ARROW_MAPPING_VERSION)
    {
        return Err(malformed("missing or unsupported mapping version"));
    }
    if batch.num_rows() > MAX_ARROW_STATES {
        return Err(malformed("too many rows"));
    }
    let mut names = HashSet::new();
    let mut questions = BTreeMap::new();
    let mut columns = Vec::new();
    for (field, array) in schema.fields().iter().zip(batch.columns()) {
        if !names.insert(field.name()) {
            return Err(malformed("duplicate question column ID"));
        }
        let kind = field
            .metadata()
            .get(META_JEV_TYPE)
            .ok_or_else(|| malformed("missing question type"))?;
        let instructions = serde_json::json!("Arrow mapping validation");
        let question = match kind.as_str() {
            "noul" => Question::Noul(NoulQuestion {
                instructions,
                criteria: None,
            }),
            "choice" => {
                let labels = strings(field, META_LABELS)?;
                if labels.is_empty()
                    || labels.len() > MAX_CHOICE_LABELS
                    || labels.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return Err(malformed(
                        "Choice labels must be unique, sorted and have width 1 through 256",
                    ));
                }
                Question::Choice(ChoiceQuestion {
                    instructions,
                    criteria: labels.into_iter().map(|label| (label, None)).collect(),
                })
            }
            "score" => Question::Score(ScoreQuestion {
                instructions,
                criteria: strings(field, META_LEGEND)?,
            }),
            _ => return Err(malformed("unknown question type")),
        };
        // Run core validation before casting widths or constructing expected fields.
        let representative = SystemRequest {
            state: State::Text(String::new()),
            model: String::new(),
            questions: [(field.name().clone(), question.clone())]
                .into_iter()
                .collect(),
        };
        openkind_core::validate_request(&representative).map_err(malformed)?;
        let expected = field_for_question(field.name(), &question)?;
        if field.data_type() != expected.data_type() {
            return Err(malformed("unexpected field or child schema"));
        }
        non_null(field, array)?;
        let values = match &question {
            Question::Noul(_) => Values::Noul(float64(array)?),
            Question::Choice(q) => {
                let values = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| malformed("expected Choice struct"))?;
                Values::Choice {
                    labels: sorted_choice_labels(q),
                    index: values
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt8Array>()
                        .ok_or_else(|| malformed("expected uint8 index"))?,
                    confidence: float64(values.column(1))?,
                    probabilities: values
                        .column(2)
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .ok_or_else(|| malformed("expected probabilities"))?,
                }
            }
            Question::Score(q) => {
                let values = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| malformed("expected Score struct"))?;
                Values::Score {
                    legend: q.criteria.clone(),
                    score: float64(values.column(0))?,
                    confidence: float64(values.column(1))?,
                    probabilities: values
                        .column(2)
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .ok_or_else(|| malformed("expected probabilities"))?,
                }
            }
        };
        questions.insert(field.name().clone(), question);
        columns.push((field.name(), values));
    }
    projected_column_bytes(&questions, batch.num_rows())?;
    let request = SystemRequest {
        state: State::Text(String::new()),
        model: String::new(),
        questions: questions.into_iter().collect(),
    };
    let contract = ResponseContract::from_request(&request).map_err(malformed)?;
    let mut rows = Vec::new();
    for row in 0..batch.num_rows() {
        let mut response = SystemResponse {
            model: String::new(),
            answers: Default::default(),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        };
        for (id, values) in &columns {
            response.answers.insert((*id).clone(), values.answer(row)?);
        }
        contract.validate(&response).map_err(malformed)?;
        rows.push(response.answers.into_iter().collect());
    }
    Ok(rows)
}
