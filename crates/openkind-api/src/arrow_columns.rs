//! Numeric column builders retain no per-row question text or legends.

use super::*;
use arrow_array::{ArrayRef, StructArray, UInt8Array};
use openkind_core::{Answer, SystemResponse};

enum Column {
    Noul(Vec<f64>),
    Choice {
        labels: Vec<String>,
        indices: Vec<u8>,
        confidence: Vec<f64>,
        probabilities: Vec<f64>,
    },
    Score {
        legend: Vec<String>,
        values: Vec<f64>,
        confidence: Vec<f64>,
        probabilities: Vec<f64>,
    },
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, ApiError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ApiError::Overloaded {
            retry_after_ms: 1000,
        })?;
    Ok(values)
}

impl Column {
    fn new(question: &Question, rows: usize) -> Result<Self, ApiError> {
        Ok(match question {
            Question::Noul(_) => Self::Noul(reserved(rows)?),
            Question::Choice(q) => Self::Choice {
                labels: sorted_choice_labels(q),
                indices: reserved(rows)?,
                confidence: reserved(rows)?,
                probabilities: reserved(
                    rows.checked_mul(q.criteria.len()).ok_or_else(size_error)?,
                )?,
            },
            Question::Score(q) => Self::Score {
                legend: q.criteria.clone(),
                values: reserved(rows)?,
                confidence: reserved(rows)?,
                probabilities: reserved(
                    rows.checked_mul(q.criteria.len()).ok_or_else(size_error)?,
                )?,
            },
        })
    }

    fn push(&mut self, id: &str, answer: &Answer) -> Result<(), ApiError> {
        let invalid = || {
            ApiError::Internal(format!(
                "backend answer for `{id}` does not match the Arrow question mapping"
            ))
        };
        match (self, answer) {
            (Self::Noul(values), Answer::Noul(answer)) => values.push(answer.noul),
            (
                Self::Choice {
                    labels,
                    indices,
                    confidence,
                    probabilities,
                },
                Answer::Choice(answer),
            ) => {
                if answer.probabilities.len() != labels.len() {
                    return Err(invalid());
                }
                let index = labels
                    .binary_search(&answer.choice)
                    .map_err(|_| invalid())?;
                indices.push(u8::try_from(index).map_err(|_| invalid())?);
                confidence.push(answer.confidence);
                for label in labels {
                    probabilities.push(*answer.probabilities.get(label).ok_or_else(invalid)?);
                }
            }
            (
                Self::Score {
                    legend,
                    values,
                    confidence,
                    probabilities,
                },
                Answer::Score(answer),
            ) => {
                // Keep each row bound to the schema even when a caller builds
                // a batch directly without dispatch's request-bound validation.
                if answer.probabilities.len() != legend.len() || answer.legend.len() != legend.len()
                {
                    return Err(invalid());
                }
                for (level, description) in legend.iter().enumerate() {
                    let key = level.to_string();
                    if answer.legend.get(&key) != Some(description) {
                        return Err(invalid());
                    }
                    probabilities.push(*answer.probabilities.get(&key).ok_or_else(invalid)?);
                }
                values.push(answer.score);
                confidence.push(answer.confidence);
            }
            _ => return Err(invalid()),
        }
        Ok(())
    }

    fn finish(self, field: &Field) -> Result<ArrayRef, ApiError> {
        let arrays: Vec<ArrayRef> = match self {
            Self::Noul(values) => return Ok(Arc::new(Float64Array::from(values))),
            Self::Choice {
                labels,
                indices,
                confidence,
                probabilities,
            } => vec![
                Arc::new(UInt8Array::from(indices)),
                Arc::new(Float64Array::from(confidence)),
                Arc::new(fixed_probabilities_column(probabilities, labels.len())?),
            ],
            Self::Score {
                legend,
                values,
                confidence,
                probabilities,
            } => vec![
                Arc::new(Float64Array::from(values)),
                Arc::new(Float64Array::from(confidence)),
                Arc::new(fixed_probabilities_column(probabilities, legend.len())?),
            ],
        };
        let DataType::Struct(fields) = field.data_type() else {
            return Err(ApiError::Internal(
                "unexpected Arrow builder field".to_string(),
            ));
        };
        Ok(Arc::new(
            StructArray::try_new(fields.clone(), arrays, None)
                .map_err(|e| ApiError::Internal(format!("assemble Arrow column: {e}")))?,
        ))
    }
}

pub(super) struct BatchBuilder {
    fields: Vec<FieldRef>,
    columns: Vec<Column>,
    fallback_model: String,
    model: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
}

impl BatchBuilder {
    pub(super) fn new(
        questions: &BTreeMap<String, Question>,
        fallback_model: &str,
        rows: usize,
    ) -> Result<Self, ApiError> {
        projected_column_bytes(questions, rows)?;
        // Validate every mapping field before reserving any column buffers.
        let fields: Vec<FieldRef> = questions
            .iter()
            .map(|(id, q)| field_for_question(id, q).map(Arc::new))
            .collect::<Result<_, _>>()?;
        let columns = questions
            .values()
            .map(|q| Column::new(q, rows))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            fields,
            columns,
            fallback_model: fallback_model.to_string(),
            model: None,
            input_tokens: 0,
            output_tokens: 0,
        })
    }

    pub(super) fn empty_batch(&self) -> Result<(SchemaRef, RecordBatch), ApiError> {
        let schema = Arc::new(
            Schema::new(self.fields.clone()).with_metadata(schema_metadata(
                &self.fallback_model,
                0,
                0,
            )),
        );
        let batch = RecordBatch::new_empty(schema.clone());
        Ok((schema, batch))
    }

    pub(super) fn push(&mut self, response: &SystemResponse) -> Result<(), ApiError> {
        if self
            .model
            .as_ref()
            .is_some_and(|model| model != &response.model)
        {
            return Err(ApiError::Internal(
                "backend model metadata changed within an Arrow batch".to_string(),
            ));
        }
        if response.answers.len() != self.fields.len() {
            return Err(ApiError::Internal(
                "backend answer coverage differs from the Arrow schema".to_string(),
            ));
        }
        for (field, column) in self.fields.iter().zip(&mut self.columns) {
            let answer = response.answers.get(field.name()).ok_or_else(|| {
                ApiError::Internal("backend answer is missing from the Arrow schema".to_string())
            })?;
            column.push(field.name(), answer)?;
        }
        self.model.get_or_insert_with(|| response.model.clone());
        self.input_tokens = self
            .input_tokens
            .checked_add(u64::from(response.usage.input_tokens))
            .ok_or_else(|| ApiError::Internal("Arrow usage overflow".to_string()))?;
        self.output_tokens = self
            .output_tokens
            .checked_add(u64::from(response.usage.output_tokens))
            .ok_or_else(|| ApiError::Internal("Arrow usage overflow".to_string()))?;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<(SchemaRef, RecordBatch), ApiError> {
        let columns = self
            .columns
            .into_iter()
            .zip(&self.fields)
            .map(|(column, field)| column.finish(field))
            .collect::<Result<Vec<_>, _>>()?;
        let model = self.model.as_deref().unwrap_or(&self.fallback_model);
        let schema = Arc::new(Schema::new(self.fields).with_metadata(schema_metadata(
            model,
            self.input_tokens,
            self.output_tokens,
        )));
        let batch = RecordBatch::try_new(schema.clone(), columns)
            .map_err(|e| ApiError::Internal(format!("assemble Arrow batch: {e}")))?;
        Ok((schema, batch))
    }
}
