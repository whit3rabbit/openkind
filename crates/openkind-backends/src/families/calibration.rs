//! Shared synthetic calibration cases for letter-readout family fixtures.
//!
//! Used only by the one-time fixture generator binaries; never part of
//! builds or tests. Every case states its correct option verbatim in the
//! state document so the truth label is deterministic.

use openkind_core::{ChoiceQuestion, NoulQuestion, Question, ScoreQuestion};
use serde_json::json;

/// One labeled calibration case.
pub struct CalibrationCase {
    /// Stable case name recorded in the golden fixture.
    pub name: String,
    /// Ordered candidate labels as they appear in the answer map.
    pub option_labels: Vec<String>,
    /// Candidate label carrying the correct answer.
    pub truth_label: String,
    /// State document text.
    pub state: String,
    /// Wire question.
    pub question: Question,
}

fn choice_case(name: &str, state: &str, options: &[&str], truth: usize) -> CalibrationCase {
    CalibrationCase {
        name: name.to_owned(),
        option_labels: options.iter().map(|option| option.to_string()).collect(),
        truth_label: options[truth].to_string(),
        state: state.to_owned(),
        question: Question::Choice(ChoiceQuestion {
            instructions: json!("Which option is explicitly recorded in the state?"),
            criteria: options
                .iter()
                .map(|option| (option.to_string(), None))
                .collect(),
        }),
    }
}

fn noul_case(name: &str, state: &str, instruction: &'static str, truth: bool) -> CalibrationCase {
    CalibrationCase {
        name: name.to_owned(),
        option_labels: vec!["false".into(), "true".into()],
        truth_label: if truth { "true".into() } else { "false".into() },
        state: state.to_owned(),
        question: Question::Noul(NoulQuestion {
            instructions: json!(instruction),
            criteria: None,
        }),
    }
}

fn score_case(name: &str, state: &str, levels: usize, truth: usize) -> CalibrationCase {
    CalibrationCase {
        name: name.to_owned(),
        option_labels: (0..levels).map(|level| level.to_string()).collect(),
        truth_label: truth.to_string(),
        state: state.to_owned(),
        question: Question::Score(ScoreQuestion {
            instructions: json!("What is the documented impact level?"),
            criteria: (0..levels).map(|level| format!("level {level}")).collect(),
        }),
    }
}

/// The full calibration set shared by every letter-readout family.
pub fn calibration_cases() -> Vec<CalibrationCase> {
    vec![
        choice_case(
            "asset-a",
            "Fictional incident record. Affected asset: application server. The asset operating system is Linux. These statements are the full evidence record; do not infer missing facts.",
            &[
                "workstation",
                "network appliance",
                "application server",
                "database cluster",
            ],
            2,
        ),
        choice_case(
            "asset-b",
            "Fictional incident record. Affected asset: storage array. These statements are the full evidence record; do not infer missing facts.",
            &[
                "workstation",
                "router",
                "application server",
                "storage array",
            ],
            3,
        ),
        choice_case(
            "os-a",
            "Fictional incident record. The asset operating system is Windows. These statements are the full evidence record; do not infer missing facts.",
            &["Linux", "Windows", "macOS", "BSD"],
            1,
        ),
        choice_case(
            "os-b",
            "Fictional incident record. The asset operating system is Linux. These statements are the full evidence record; do not infer missing facts.",
            &["Linux", "Windows Server 2022", "macOS", "Solaris"],
            0,
        ),
        choice_case(
            "owner-a",
            "Fictional incident record. The owner of record is the platform team. These statements are the full evidence record; do not infer missing facts.",
            &[
                "security team",
                "platform team",
                "external vendor",
                "data team",
            ],
            1,
        ),
        choice_case(
            "owner-b",
            "Fictional incident record. The owner of record is an external vendor. These statements are the full evidence record; do not infer missing facts.",
            &[
                "security team",
                "platform team",
                "external vendor",
                "data team",
            ],
            2,
        ),
        choice_case(
            "region-a",
            "Fictional incident record. The affected deployment runs in us-east-1. These statements are the full evidence record; do not infer missing facts.",
            &["us-east-1", "eu-west-2", "ap-southeast-1", "us-west-1"],
            0,
        ),
        choice_case(
            "region-b",
            "Fictional incident record. The affected deployment runs in ap-southeast-1. These statements are the full evidence record; do not infer missing facts.",
            &["us-east-1", "eu-west-2", "ap-southeast-1", "us-west-1"],
            2,
        ),
        choice_case(
            "protocol-a",
            "Fictional incident record. The observed connection used SSH. These statements are the full evidence record; do not infer missing facts.",
            &["HTTPS", "SSH", "SMTP", "DNS"],
            1,
        ),
        choice_case(
            "protocol-b",
            "Fictional incident record. The observed connection used DNS. These statements are the full evidence record; do not infer missing facts.",
            &["HTTPS", "SSH", "SMTP", "DNS", "RDP"],
            3,
        ),
        choice_case(
            "impact-word",
            "Fictional incident record. The documented impact level is major. These statements are the full evidence record; do not infer missing facts.",
            &["minor", "moderate", "major", "critical"],
            2,
        ),
        choice_case(
            "status",
            "Fictional incident record. The incident status is investigating. These statements are the full evidence record; do not infer missing facts.",
            &["open", "investigating", "mitigated", "closed"],
            1,
        ),
        noul_case(
            "external-true",
            "Fictional incident record. An external destination was observed in the connection logs. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that an external destination was observed?",
            true,
        ),
        noul_case(
            "external-false",
            "Fictional incident record. An external destination was not observed. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that an external destination was observed?",
            false,
        ),
        noul_case(
            "mfa-true",
            "Fictional incident record. The record states that two-factor authentication was enabled on the affected account. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that two-factor authentication was enabled?",
            true,
        ),
        noul_case(
            "mfa-false",
            "Fictional incident record. The record does not mention multi-factor authentication. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that multi-factor authentication was enabled?",
            false,
        ),
        noul_case(
            "export-false",
            "Fictional incident record. No data export was approved during the window. These statements are the full evidence record; do not infer missing facts.",
            "Does the record explicitly state that a data export was approved?",
            false,
        ),
        noul_case(
            "patch-true",
            "Fictional incident record. The patch level is current as of the report date. These statements are the full evidence record; do not infer missing facts.",
            "Is the patch level described as current?",
            true,
        ),
        score_case(
            "impact-level-0",
            "Fictional incident record. The documented impact level is 0. These statements are the full evidence record; do not infer missing facts.",
            5,
            0,
        ),
        score_case(
            "impact-level-1",
            "Fictional incident record. The documented impact level is 1. These statements are the full evidence record; do not infer missing facts.",
            5,
            1,
        ),
        score_case(
            "impact-level-2",
            "Fictional incident record. The documented impact level is 2. These statements are the full evidence record; do not infer missing facts.",
            5,
            2,
        ),
        score_case(
            "impact-level-3",
            "Fictional incident record. The documented impact level is 3. These statements are the full evidence record; do not infer missing facts.",
            5,
            3,
        ),
        score_case(
            "impact-level-4",
            "Fictional incident record. The documented impact level is 4. These statements are the full evidence record; do not infer missing facts.",
            5,
            4,
        ),
    ]
}
