//! Wire question unpacking and distribution-to-answer mapping.

use std::collections::HashMap;

use openkind_core::{Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer};

use super::SEMANTIC_NONE_OPTION;
use crate::qwen35::{HeadEvaluation, PrimitiveKind, Qwen35Error};

pub(crate) struct EncodedQuestion {
    pub(crate) id: String,
    pub(crate) primitive: PrimitiveKind,
    pub(crate) labels: Vec<String>,
    pub(crate) criteria: Vec<String>,
    pub(crate) question_ids: Vec<u32>,
    pub(crate) candidate_suffix_ids: Vec<Vec<u32>>,
}

pub(crate) fn question_candidates(
    question: &Question,
) -> Result<(PrimitiveKind, Vec<String>, Vec<String>), Qwen35Error> {
    match question {
        Question::Noul(question) => {
            let (false_criterion, true_criterion) = question
                .criteria
                .as_ref()
                .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                .unwrap_or_else(|| {
                    (
                        "The answer is false or no.".into(),
                        "The answer is true or yes.".into(),
                    )
                });
            Ok((
                PrimitiveKind::Noul,
                vec!["false".into(), "true".into()],
                vec![false_criterion, true_criterion],
            ))
        }
        Question::Choice(question) => {
            let none = question.criteria.get(SEMANTIC_NONE_OPTION).ok_or_else(|| {
                Qwen35Error::InvalidInput(format!(
                    "native Choice requires explicit `{SEMANTIC_NONE_OPTION}` criteria so semantic-none mass remains on wire"
                ))
            })?;
            if none
                .as_deref()
                .is_none_or(|criterion| criterion.trim().is_empty())
            {
                return Err(Qwen35Error::InvalidInput(format!(
                    "`{SEMANTIC_NONE_OPTION}` must have a non-empty description"
                )));
            }
            let mut labels: Vec<_> = question
                .criteria
                .keys()
                .filter(|label| label.as_str() != SEMANTIC_NONE_OPTION)
                .cloned()
                .collect();
            labels.sort();
            let criteria = labels
                .iter()
                .map(|label| {
                    question.criteria[label]
                        .clone()
                        .unwrap_or_else(|| label.clone())
                })
                .collect();
            Ok((PrimitiveKind::Choice, labels, criteria))
        }
        Question::Score(question) => Ok((
            PrimitiveKind::Score,
            (0..question.criteria.len())
                .map(|index| index.to_string())
                .collect(),
            question.criteria.clone(),
        )),
    }
}

pub(crate) fn answer_from_distribution(
    question: &EncodedQuestion,
    evaluation: &HeadEvaluation,
) -> Result<Answer, Qwen35Error> {
    match question.primitive {
        PrimitiveKind::Noul => Ok(Answer::Noul(NoulAnswer {
            noul: evaluation.candidate_probabilities()[1],
        })),
        PrimitiveKind::Choice => {
            let mut probabilities: HashMap<String, f64> = question
                .labels
                .iter()
                .cloned()
                .zip(evaluation.candidate_probabilities().iter().copied())
                .collect();
            let none = evaluation.none_probability().ok_or_else(|| {
                Qwen35Error::Numerical("Choice evaluation omitted semantic-none mass".into())
            })?;
            probabilities.insert(SEMANTIC_NONE_OPTION.into(), none);
            let choice = evaluation
                .selected_candidate_index()
                .map(|index| question.labels[index].clone())
                .unwrap_or_else(|| SEMANTIC_NONE_OPTION.into());
            Ok(Answer::Choice(ChoiceAnswer {
                choice,
                confidence: distribution_confidence(&evaluation.full_probabilities()),
                probabilities,
            }))
        }
        PrimitiveKind::Score => {
            let probabilities: HashMap<String, f64> = question
                .labels
                .iter()
                .cloned()
                .zip(evaluation.candidate_probabilities().iter().copied())
                .collect();
            let score = evaluation
                .candidate_probabilities()
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            let legend = question
                .labels
                .iter()
                .cloned()
                .zip(question.criteria.iter().cloned())
                .collect();
            Ok(Answer::Score(ScoreAnswer {
                score,
                legend,
                confidence: distribution_confidence(evaluation.candidate_probabilities()),
                probabilities,
            }))
        }
    }
}

pub(crate) fn distribution_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() <= 1 {
        return 1.0;
    }
    let entropy = probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

pub(crate) fn instruction_text(question: &Question) -> Result<String, Qwen35Error> {
    let instructions = match question {
        Question::Noul(question) => &question.instructions,
        Question::Choice(question) => &question.instructions,
        Question::Score(question) => &question.instructions,
    };
    match instructions {
        serde_json::Value::String(text) => Ok(text.clone()),
        value => serde_json::to_string(value).map_err(|error| {
            Qwen35Error::Tokenizer(format!("instruction serialization failed: {error}"))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_core::ChoiceQuestion;

    #[test]
    fn choice_requires_explicit_semantic_none_and_never_sends_it_to_the_model() {
        let missing = Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: [
                ("a".into(), Some("A".into())),
                ("b".into(), Some("B".into())),
            ]
            .into_iter()
            .collect(),
        });
        assert!(matches!(
            question_candidates(&missing),
            Err(Qwen35Error::InvalidInput(message)) if message.contains(SEMANTIC_NONE_OPTION)
        ));

        let explicit = Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: [
                ("b".into(), Some("B".into())),
                (SEMANTIC_NONE_OPTION.into(), Some("Neither applies".into())),
                ("a".into(), None),
            ]
            .into_iter()
            .collect(),
        });
        let (_, labels, criteria) = question_candidates(&explicit).expect("map choice");
        assert_eq!(labels, ["a", "b"]);
        assert_eq!(criteria, ["a", "B"]);
    }

    #[test]
    fn confidence_is_entropy_based_not_top_probability() {
        let uniform = distribution_confidence(&[0.5, 0.5]);
        let peaked = distribution_confidence(&[0.99, 0.01]);
        assert!(uniform.abs() <= f64::EPSILON);
        assert!(peaked > 0.9);
        assert_ne!(peaked, 0.99);
    }
}
