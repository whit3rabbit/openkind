//! Shared Jev wire unpacking and answer mapping for surveyed families.
//!
//! These helpers serve profiles that declare
//! [`ProbabilitySpace::ConditionalOnOfferedOptions`]: the model returns one
//! probability per offered candidate and the distribution over those options
//! sums to one. Semantic-none mass is never invented; when a request offers
//! the reserved `__none__` key it is scored like any other offered option.

use std::collections::HashMap;

use openkind_core::{Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer};

use crate::families::support::FamilyError;

/// Reserved Choice criteria key for explicit semantic-none mass.
///
/// Conditional-space families have no none mass of their own. When the
/// caller offers this key it is treated as an ordinary candidate so the
/// reserved key survives the wire round trip instead of being silently
/// dropped or renormalized.
pub const RESERVED_NONE_OPTION: &str = "__none__";

/// Wire primitive of an unpacked question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionPrimitive {
    /// Boolean probability question.
    Noul,
    /// Categorical choice over discrete options.
    Choice,
    /// Ordinal rating along an ordered rubric.
    Score,
}

/// Question primitive with its unpacked candidate labels and criteria.
#[derive(Debug, Clone)]
pub struct UnpackedQuestion {
    /// Question id from the request.
    pub id: String,
    /// Wire primitive the question arrived as.
    pub primitive: QuestionPrimitive,
    /// Candidate labels in evaluation order.
    pub labels: Vec<String>,
    /// Candidate criteria (hypothesis/criterion text) in evaluation order.
    pub criteria: Vec<String>,
    /// True for `Score` questions: candidates are ordered ordinal levels.
    pub ordered: bool,
}

impl UnpackedQuestion {
    /// `true` when the candidate at `index` is the reserved none option.
    pub fn is_reserved_none(&self, index: usize) -> bool {
        self.labels
            .get(index)
            .is_some_and(|label| label == RESERVED_NONE_OPTION)
    }
}

/// Unpack one wire question into candidate labels and criteria.
///
/// `Noul` maps to the fixed `[false, true]` candidate pair, using the
/// reserved criteria descriptions when the caller supplied them and the
/// reference default descriptions otherwise. `Choice` maps every offered
/// option (including a reserved `__none__` key, when present) to a candidate;
/// labels sort lexicographically for deterministic evaluation order and null
/// criteria fall back to the label text. `Score` maps the ordered level list
/// positionally.
pub fn unpack_question(id: &str, question: &Question) -> Result<UnpackedQuestion, FamilyError> {
    match question {
        Question::Noul(question) => {
            let (false_criterion, true_criterion) = question
                .criteria
                .as_ref()
                .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                .unwrap_or_else(|| {
                    (
                        "The stated proposition is false according to the record.".into(),
                        "The stated proposition is true according to the record.".into(),
                    )
                });
            Ok(UnpackedQuestion {
                id: id.to_owned(),
                primitive: QuestionPrimitive::Noul,
                labels: vec!["false".into(), "true".into()],
                criteria: vec![false_criterion, true_criterion],
                ordered: false,
            })
        }
        Question::Choice(question) => {
            if question.criteria.is_empty() {
                return Err(FamilyError::InvalidInput(format!(
                    "choice question `{id}` offers no options"
                )));
            }
            if question
                .criteria
                .get(RESERVED_NONE_OPTION)
                .is_some_and(|criterion| {
                    criterion
                        .as_deref()
                        .is_none_or(|text| text.trim().is_empty())
                })
            {
                return Err(FamilyError::InvalidInput(format!(
                    "choice question `{id}` offers `{RESERVED_NONE_OPTION}` without a non-empty description"
                )));
            }
            let mut labels: Vec<String> = question.criteria.keys().cloned().collect();
            labels.sort();
            let criteria = labels
                .iter()
                .map(|label| {
                    question
                        .criteria
                        .get(label)
                        .and_then(|criterion| criterion.clone())
                        .unwrap_or_else(|| label.clone())
                })
                .collect();
            Ok(UnpackedQuestion {
                id: id.to_owned(),
                primitive: QuestionPrimitive::Choice,
                labels,
                criteria,
                ordered: false,
            })
        }
        Question::Score(question) => {
            if question.criteria.len() < 2 {
                return Err(FamilyError::InvalidInput(format!(
                    "score question `{id}` needs at least two ordered levels, found {}",
                    question.criteria.len()
                )));
            }
            Ok(UnpackedQuestion {
                id: id.to_owned(),
                primitive: QuestionPrimitive::Score,
                labels: (0..question.criteria.len())
                    .map(|index| index.to_string())
                    .collect(),
                criteria: question.criteria.clone(),
                ordered: true,
            })
        }
    }
}

/// Map a per-candidate probability distribution back onto the wire answer.
///
/// `probabilities` must align with `question.labels`. The reserved none key,
/// when offered, keeps its model probability; no mass is added or removed.
pub fn answer_from_probabilities(
    question: &UnpackedQuestion,
    probabilities: &[f64],
) -> Result<Answer, FamilyError> {
    let valid_candidate_count = match question.primitive {
        QuestionPrimitive::Noul => question.labels.len() == 2,
        QuestionPrimitive::Choice => !question.labels.is_empty(),
        QuestionPrimitive::Score => question.labels.len() >= 2,
    };
    if !valid_candidate_count {
        return Err(FamilyError::InvalidInput(format!(
            "question `{}` has an invalid {:?} candidate count: {}",
            question.id,
            question.primitive,
            question.labels.len()
        )));
    }
    if probabilities.len() != question.labels.len() {
        return Err(FamilyError::InvalidInput(format!(
            "question `{}` expected {} candidate probabilities, found {}",
            question.id,
            question.labels.len(),
            probabilities.len()
        )));
    }
    if probabilities
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(FamilyError::Numerical(format!(
            "question `{}` produced a non-finite or out-of-range candidate probability",
            question.id
        )));
    }
    // Reject malformed model output at the backend boundary using the wire
    // contract's sum tolerance. Renormalizing here would hide a readout bug.
    let sum: f64 = probabilities.iter().sum();
    if (sum - 1.0).abs() > 1e-3 {
        return Err(FamilyError::Numerical(format!(
            "question `{}` produced candidate probabilities that sum to {sum}, expected 1",
            question.id
        )));
    }
    // Strictly-greater scan: ties resolve to the earliest candidate index.
    let mut argmax = 0;
    for (index, probability) in probabilities.iter().enumerate() {
        if *probability > probabilities[argmax] {
            argmax = index;
        }
    }
    let per_label: HashMap<String, f64> = question
        .labels
        .iter()
        .cloned()
        .zip(probabilities.iter().copied())
        .collect();
    match question.primitive {
        QuestionPrimitive::Noul => {
            let _ = question.ordered;
            Ok(Answer::Noul(NoulAnswer {
                noul: probabilities[1],
            }))
        }
        QuestionPrimitive::Choice => {
            let _ = question.ordered;
            Ok(Answer::Choice(ChoiceAnswer {
                choice: question.labels[argmax].clone(),
                confidence: distribution_confidence(probabilities),
                probabilities: per_label,
            }))
        }
        QuestionPrimitive::Score => {
            let score: f64 = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            if !score.is_finite() {
                return Err(FamilyError::Numerical(format!(
                    "question `{}` produced a non-finite ordinal score",
                    question.id
                )));
            }
            let legend = question
                .labels
                .iter()
                .cloned()
                .zip(question.criteria.iter().cloned())
                .collect();
            Ok(Answer::Score(ScoreAnswer {
                score,
                legend,
                confidence: distribution_confidence(probabilities),
                probabilities: per_label,
            }))
        }
    }
}

/// Entropy-based normalized confidence in `[0, 1]`.
pub fn distribution_confidence(probabilities: &[f64]) -> f64 {
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

/// Serialize the question `instructions` payload to prompt text.
pub fn instruction_text(question: &Question) -> Result<String, FamilyError> {
    let instructions = match question {
        Question::Noul(question) => &question.instructions,
        Question::Choice(question) => &question.instructions,
        Question::Score(question) => &question.instructions,
    };
    match instructions {
        serde_json::Value::String(text) => Ok(text.clone()),
        value => serde_json::to_string(value).map_err(|error| {
            FamilyError::Tokenizer(format!("instruction serialization failed: {error}"))
        }),
    }
}

/// Render the wire state into prompt text.
pub fn state_text(state: &openkind_core::State) -> Result<String, FamilyError> {
    match state {
        openkind_core::State::Text(text) => Ok(text.clone()),
        openkind_core::State::Object(map) => serde_json::to_string_pretty(map).map_err(|error| {
            FamilyError::InvalidInput(format!("state serialization failed: {error}"))
        }),
        openkind_core::State::Array(items) => {
            serde_json::to_string_pretty(items).map_err(|error| {
                FamilyError::InvalidInput(format!("state serialization failed: {error}"))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_core::{ChoiceQuestion, NoulQuestion, ScoreQuestion};

    #[test]
    fn noul_defaults_to_false_true_pair_and_uses_supplied_criteria() {
        let default_question = Question::Noul(NoulQuestion {
            instructions: serde_json::json!("Is it true?"),
            criteria: None,
        });
        let unpacked = unpack_question("q1", &default_question).expect("unpack");
        assert_eq!(unpacked.labels, ["false", "true"]);
        assert!(!unpacked.ordered);

        let explicit = Question::Noul(NoulQuestion {
            instructions: serde_json::json!("Is it true?"),
            criteria: Some(openkind_core::NoulCriteria {
                r#true: "The server was compromised.".into(),
                r#false: "The server was not compromised.".into(),
            }),
        });
        let unpacked = unpack_question("q1", &explicit).expect("unpack");
        assert_eq!(unpacked.criteria[0], "The server was not compromised.");
        assert_eq!(unpacked.criteria[1], "The server was compromised.");
    }

    #[test]
    fn choice_maps_offered_options_and_rejects_undescribed_none() {
        let question = Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: [
                ("b".into(), Some("Option B".into())),
                ("a".into(), None),
                (RESERVED_NONE_OPTION.into(), Some("Neither applies".into())),
            ]
            .into_iter()
            .collect(),
        });
        let unpacked = unpack_question("q1", &question).expect("unpack");
        assert_eq!(unpacked.labels, [RESERVED_NONE_OPTION, "a", "b"]);
        assert_eq!(unpacked.criteria, ["Neither applies", "a", "Option B"]);
        assert!(unpacked.is_reserved_none(0));
        assert!(!unpacked.is_reserved_none(1));

        let empty_none = Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: [
                ("a".into(), None),
                (RESERVED_NONE_OPTION.into(), Some("  ".into())),
            ]
            .into_iter()
            .collect(),
        });
        assert!(unpack_question("q1", &empty_none).is_err());
    }

    #[test]
    fn answers_map_distribution_without_inventing_mass() {
        let question = Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: [
                ("a".into(), None),
                ("b".into(), None),
                (RESERVED_NONE_OPTION.into(), Some("none".into())),
            ]
            .into_iter()
            .collect(),
        });
        let unpacked = unpack_question("q1", &question).expect("unpack");
        // Candidate order after sorting: [__none__, a, b].
        let answer = answer_from_probabilities(&unpacked, &[0.3, 0.1, 0.6]).expect("answer");
        match answer {
            Answer::Choice(choice) => {
                assert_eq!(choice.choice, "b");
                let sum: f64 = choice.probabilities.values().sum();
                assert!((sum - 1.0).abs() < 1e-9);
                assert_eq!(choice.probabilities.get(RESERVED_NONE_OPTION), Some(&0.3));
                assert_eq!(choice.probabilities.get("a"), Some(&0.1));
            }
            other => panic!("unexpected answer {other:?}"),
        }
    }

    #[test]
    fn noul_answer_uses_the_true_candidate() {
        let question = Question::Noul(NoulQuestion {
            instructions: serde_json::json!("True?"),
            criteria: None,
        });
        let unpacked = unpack_question("q1", &question).expect("unpack");
        let answer = answer_from_probabilities(&unpacked, &[0.25, 0.75]).expect("answer");
        match answer {
            Answer::Noul(noul) => assert_eq!(noul.noul, 0.75),
            other => panic!("unexpected answer {other:?}"),
        }
    }

    #[test]
    fn score_answer_uses_expected_level_value() {
        let question = Question::Score(ScoreQuestion {
            instructions: serde_json::json!("Rate"),
            criteria: vec!["low".into(), "mid".into(), "high".into()],
        });
        let unpacked = unpack_question("q1", &question).expect("unpack");
        let answer = answer_from_probabilities(&unpacked, &[0.2, 0.3, 0.5]).expect("answer");
        match answer {
            Answer::Score(score) => {
                assert!((score.score - (0.3 + 1.0)).abs() < 1e-9);
                assert_eq!(score.legend.get("2"), Some(&"high".to_string()));
            }
            other => panic!("unexpected answer {other:?}"),
        }
    }

    #[test]
    fn rejects_malformed_distributions() {
        let question = Question::Score(ScoreQuestion {
            instructions: serde_json::json!("Rate"),
            criteria: vec!["low".into(), "high".into()],
        });
        let unpacked = unpack_question("q1", &question).expect("unpack");
        assert!(answer_from_probabilities(&unpacked, &[0.5]).is_err());
        assert!(answer_from_probabilities(&unpacked, &[0.5, f64::NAN]).is_err());
        assert!(answer_from_probabilities(&unpacked, &[-0.1, 1.1]).is_err());
        assert!(answer_from_probabilities(&unpacked, &[0.0, 0.0]).is_err());
        assert!(answer_from_probabilities(&unpacked, &[0.6, 0.6]).is_err());
    }

    #[test]
    fn rejects_invalid_unpacked_cardinality_without_panicking() {
        for (primitive, labels, probabilities) in [
            (QuestionPrimitive::Choice, vec![], vec![]),
            (QuestionPrimitive::Noul, vec!["true".into()], vec![1.0]),
            (QuestionPrimitive::Score, vec!["0".into()], vec![1.0]),
        ] {
            let question = UnpackedQuestion {
                id: "q1".into(),
                primitive,
                criteria: labels.clone(),
                labels,
                ordered: primitive == QuestionPrimitive::Score,
            };
            assert!(matches!(
                answer_from_probabilities(&question, &probabilities),
                Err(FamilyError::InvalidInput(_))
            ));
        }
    }
}
