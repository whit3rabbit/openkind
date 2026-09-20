//! Unit tests for response validation rules in opendecision-core.

use std::collections::HashMap;

use crate::answer::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
use crate::error::types::ValidationError;
use crate::error::validate::validate_response;
use crate::response::{SystemResponse, Usage};

#[test]
fn validate_response_rejects_probability_keys_mismatch() {
    let mut answers = HashMap::new();
    let mut probs = HashMap::new();
    probs.insert("option_a".into(), 0.6);
    probs.insert("option_x".into(), 0.4);
    answers.insert(
        "ch".into(),
        Answer::Choice(ChoiceAnswer {
            choice: "option_a".into(),
            probabilities: probs,
            confidence: 0.8,
        }),
    );
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 5,
        },
    };
    let mut criteria = HashMap::new();
    criteria.insert("ch".into(), vec!["option_a".into(), "option_b".into()]);
    assert!(matches!(
        validate_response(&resp, &criteria).unwrap_err(),
        ValidationError::ProbabilityKeysMismatch(_)
    ));
}

#[test]
fn validate_response_rejects_noul_out_of_range() {
    for invalid_noul in [-0.01, 1.05] {
        let mut answers = HashMap::new();
        answers.insert(
            "noul_q".into(),
            Answer::Noul(NoulAnswer { noul: invalid_noul }),
        );
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let criteria = HashMap::new();
        assert!(
            matches!(
                validate_response(&resp, &criteria).unwrap_err(),
                ValidationError::NoulOutOfRange { .. }
            ),
            "failed to reject out-of-range noul: {}",
            invalid_noul
        );
    }
}

#[test]
fn validate_response_rejects_confidence_out_of_range() {
    for invalid_conf in [-0.1, 1.2] {
        let mut answers = HashMap::new();
        let mut probs = HashMap::new();
        probs.insert("0".into(), 0.7);
        probs.insert("1".into(), 0.3);
        let mut legend = HashMap::new();
        legend.insert("0".into(), "Low".into());
        legend.insert("1".into(), "High".into());
        answers.insert(
            "sc".into(),
            Answer::Score(ScoreAnswer {
                score: 0.3,
                legend,
                probabilities: probs,
                confidence: invalid_conf,
            }),
        );
        let resp = SystemResponse {
            model: "mock".into(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let criteria = HashMap::new();
        assert!(
            matches!(
                validate_response(&resp, &criteria).unwrap_err(),
                ValidationError::ConfidenceOutOfRange { .. }
            ),
            "failed to reject out-of-range confidence: {}",
            invalid_conf
        );
    }
}

#[test]
fn validate_response_happy_path_all_three_types() {
    let mut answers = HashMap::new();
    answers.insert("noul".into(), Answer::Noul(NoulAnswer { noul: 0.85 }));
    let mut c_probs = HashMap::new();
    c_probs.insert("a".into(), 0.7);
    c_probs.insert("b".into(), 0.3);
    answers.insert(
        "choice".into(),
        Answer::Choice(ChoiceAnswer {
            choice: "a".into(),
            probabilities: c_probs,
            confidence: 0.75,
        }),
    );
    let mut s_probs = HashMap::new();
    s_probs.insert("0".into(), 0.2);
    s_probs.insert("1".into(), 0.8);
    let mut legend = HashMap::new();
    legend.insert("0".into(), "No".into());
    legend.insert("1".into(), "Yes".into());
    answers.insert(
        "score".into(),
        Answer::Score(ScoreAnswer {
            score: 0.8,
            legend,
            probabilities: s_probs,
            confidence: 0.9,
        }),
    );
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 20,
            output_tokens: 6,
        },
    };
    let mut criteria = HashMap::new();
    criteria.insert("choice".into(), vec!["a".into(), "b".into()]);
    criteria.insert("noul".into(), Vec::new());
    criteria.insert("score".into(), Vec::new());
    assert!(validate_response(&resp, &criteria).is_ok());
}

#[test]
fn validate_response_rejects_missing_answer() {
    let mut answers = HashMap::new();
    answers.insert("noul".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    };
    let mut criteria = HashMap::new();
    criteria.insert("noul".into(), Vec::new());
    criteria.insert("dropped".into(), Vec::new());
    assert!(matches!(
        validate_response(&resp, &criteria).unwrap_err(),
        ValidationError::MissingAnswer(ref id) if id == "dropped"
    ));
}

#[test]
fn validate_response_rejects_unexpected_answer() {
    let mut answers = HashMap::new();
    answers.insert("noul".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
    answers.insert("ghost".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    };
    let mut criteria = HashMap::new();
    criteria.insert("noul".into(), Vec::new());
    assert!(matches!(
        validate_response(&resp, &criteria).unwrap_err(),
        ValidationError::UnexpectedAnswer(ref id) if id == "ghost"
    ));
}

#[test]
fn validate_response_empty_criteria_skips_coverage_checks() {
    let mut answers = HashMap::new();
    answers.insert("anything".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    };
    assert!(validate_response(&resp, &HashMap::new()).is_ok());
}

#[test]
fn validate_response_rejects_nan_probabilities() {
    let mut answers = HashMap::new();
    let mut c_probs = HashMap::new();
    c_probs.insert("a".into(), f64::NAN);
    c_probs.insert("b".into(), 0.5);
    answers.insert(
        "choice".into(),
        Answer::Choice(ChoiceAnswer {
            choice: "a".into(),
            probabilities: c_probs,
            confidence: 0.5,
        }),
    );
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 2,
        },
    };
    let mut criteria = HashMap::new();
    criteria.insert("choice".into(), vec!["a".into(), "b".into()]);
    assert!(matches!(
        validate_response(&resp, &criteria).unwrap_err(),
        ValidationError::ProbabilityOutOfRange { .. }
    ));
}

#[test]
fn validate_response_rejects_negative_probabilities() {
    let mut answers = HashMap::new();
    let mut c_probs = HashMap::new();
    c_probs.insert("a".into(), -0.2);
    c_probs.insert("b".into(), 1.2);
    answers.insert(
        "choice".into(),
        Answer::Choice(ChoiceAnswer {
            choice: "b".into(),
            probabilities: c_probs,
            confidence: 0.5,
        }),
    );
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 2,
        },
    };
    let mut criteria = HashMap::new();
    criteria.insert("choice".into(), vec!["a".into(), "b".into()]);
    assert!(matches!(
        validate_response(&resp, &criteria).unwrap_err(),
        ValidationError::ProbabilityOutOfRange { .. }
    ));
}

#[test]
fn validate_response_rejects_nan_and_out_of_range_score() {
    let mut legend = HashMap::new();
    legend.insert("0".into(), "Low".into());
    legend.insert("1".into(), "High".into());

    let mut score_probs = HashMap::new();
    score_probs.insert("0".into(), 0.5);
    score_probs.insert("1".into(), 0.5);

    // Test NaN score
    let mut answers_nan = HashMap::new();
    answers_nan.insert(
        "score_q".into(),
        Answer::Score(ScoreAnswer {
            score: f64::NAN,
            legend: legend.clone(),
            probabilities: score_probs.clone(),
            confidence: 0.8,
        }),
    );
    let resp_nan = SystemResponse {
        model: "mock".into(),
        answers: answers_nan,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 4,
        },
    };
    assert!(matches!(
        validate_response(&resp_nan, &HashMap::new()).unwrap_err(),
        ValidationError::ScoreOutOfRange { .. }
    ));

    // Test out of range score (score 2.5 when max index is 1)
    let mut answers_oor = HashMap::new();
    answers_oor.insert(
        "score_q".into(),
        Answer::Score(ScoreAnswer {
            score: 2.5,
            legend,
            probabilities: score_probs,
            confidence: 0.8,
        }),
    );
    let resp_oor = SystemResponse {
        model: "mock".into(),
        answers: answers_oor,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 4,
        },
    };
    assert!(matches!(
        validate_response(&resp_oor, &HashMap::new()).unwrap_err(),
        ValidationError::ScoreOutOfRange { .. }
    ));
}

#[test]
fn validate_response_rejects_nan_noul() {
    let mut answers = HashMap::new();
    answers.insert("noul_q".into(), Answer::Noul(NoulAnswer { noul: f64::NAN }));
    let resp = SystemResponse {
        model: "mock".into(),
        answers,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 1,
        },
    };
    assert!(matches!(
        validate_response(&resp, &HashMap::new()).unwrap_err(),
        ValidationError::NoulOutOfRange { .. }
    ));
}
