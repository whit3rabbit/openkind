//! Unit tests for the unofficial Arrow bulk mapping (`super` = [`crate::arrow`]).
//!
//! The round-trip tests are the executable definition of the mapping:
//! building a batch, encoding it as an Arrow IPC stream, decoding it with a
//! stock reader, and reconstructing answers via [`answers_from_batch`] must
//! reproduce the exact answer objects the engine returned.

use std::collections::{BTreeMap, HashMap};
use std::io::Cursor;

use arrow_ipc::reader::StreamReader;
use arrow_schema::DataType;
use openkind_core::{
    Answer, ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, Question, ScoreAnswer,
    ScoreQuestion, SystemResponse, Usage, WireHashState,
};
use serde_json::json;

use super::*;

/// Noul question with a bare-string instruction.
pub(super) fn noul_question() -> Question {
    Question::Noul(NoulQuestion {
        instructions: json!("Is a refund being requested?"),
        criteria: None,
    })
}

/// Choice question whose criteria keys are deliberately inserted unsorted;
/// the mapping must project them into sorted label order.
pub(super) fn choice_question() -> Question {
    let mut criteria = HashMap::new();
    criteria.insert("shipping".to_string(), None);
    criteria.insert("billing".to_string(), Some("Payments, refunds".to_string()));
    criteria.insert("other".to_string(), None);
    Question::Choice(ChoiceQuestion {
        instructions: json!("Which department does this belong to?"),
        criteria,
    })
}

/// Score question with a three-level rubric.
fn score_question() -> Question {
    Question::Score(ScoreQuestion {
        instructions: json!("How urgent is this?"),
        criteria: vec![
            "Can wait".to_string(),
            "Within a few days".to_string(),
            "Today".to_string(),
        ],
    })
}

/// One answer per question with hand-picked values (no NaN, no arithmetic),
/// so byte-exact round-trips compare equal through `PartialEq`.
pub(super) fn sample_answers() -> HashMap<String, Answer, WireHashState> {
    let answers: Vec<(String, Answer)> = vec![
        (
            "refund".to_string(),
            Answer::Noul(NoulAnswer { noul: 0.7312 }),
        ),
        (
            "department".to_string(),
            Answer::Choice(ChoiceAnswer {
                choice: "shipping".to_string(),
                probabilities: [
                    ("billing".to_string(), 0.125),
                    ("other".to_string(), 0.25),
                    ("shipping".to_string(), 0.625),
                ]
                .into_iter()
                .collect(),
                confidence: 0.4,
            }),
        ),
        (
            "urgency".to_string(),
            Answer::Score(ScoreAnswer {
                score: 1.25,
                legend: [
                    ("0".to_string(), "Can wait".to_string()),
                    ("1".to_string(), "Within a few days".to_string()),
                    ("2".to_string(), "Today".to_string()),
                ]
                .into_iter()
                .collect(),
                probabilities: [
                    ("0".to_string(), 0.125),
                    ("1".to_string(), 0.5),
                    ("2".to_string(), 0.375),
                ]
                .into_iter()
                .collect(),
                confidence: 0.55,
            }),
        ),
    ];
    answers.into_iter().collect()
}

pub(super) fn response(
    model: &str,
    answers: HashMap<String, Answer, WireHashState>,
) -> SystemResponse {
    SystemResponse {
        model: model.to_string(),
        answers,
        usage: Usage {
            input_tokens: 11,
            output_tokens: 3,
        },
    }
}

pub(super) fn questions() -> BTreeMap<String, Question> {
    BTreeMap::from([
        ("urgency".to_string(), score_question()),
        ("refund".to_string(), noul_question()),
        ("department".to_string(), choice_question()),
    ])
}

/// Encode one batch and decode it back with a stock Arrow IPC stream reader.
fn encode_decode(
    questions: &BTreeMap<String, Question>,
    responses: &[SystemResponse],
) -> (arrow_schema::SchemaRef, RecordBatch) {
    let (schema, batch) = build_batch(questions, responses, "fallback-model").unwrap();
    let bytes = encode_ipc_stream(schema.clone(), batch).unwrap();
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
    let decoded_schema = reader.schema().clone();
    let batch = reader.next().unwrap().unwrap();
    // The stream ends with the end-of-stream marker and no trailing batches.
    assert!(reader.next().is_none());
    assert_eq!(decoded_schema, schema);
    (schema, batch)
}

#[test]
fn noul_column_is_plain_non_null_float64() {
    let field = field_for_question("refund", &noul_question()).unwrap();
    assert_eq!(field.data_type(), &DataType::Float64);
    assert!(!field.is_nullable());
    assert_eq!(field.metadata().get(META_JEV_TYPE).unwrap(), "noul");
}

#[test]
fn choice_column_schema_and_labels_are_sorted_option_keys() {
    let field = field_for_question("department", &choice_question()).unwrap();
    let DataType::Struct(children) = field.data_type() else {
        panic!(
            "choice column must be a struct, got {:?}",
            field.data_type()
        );
    };
    assert!(!field.is_nullable());
    let names: Vec<String> = children
        .iter()
        .map(|child| child.name().to_string())
        .collect();
    assert_eq!(names, ["choice", "confidence", "probabilities"]);
    assert_eq!(children[0].data_type(), &DataType::UInt8);
    assert!(!children[0].is_nullable());
    assert_eq!(children[1].data_type(), &DataType::Float64);
    assert_eq!(
        children[2].data_type(),
        &DataType::FixedSizeList(
            std::sync::Arc::new(arrow_schema::Field::new("item", DataType::Float64, false)),
            3,
        )
    );
    // `criteria` was inserted shipping/billing/other; metadata is sorted.
    let labels: Vec<String> =
        serde_json::from_str(field.metadata().get(META_LABELS).unwrap()).unwrap();
    assert_eq!(labels, vec!["billing", "other", "shipping"]);
    assert_eq!(field.metadata().get(META_JEV_TYPE).unwrap(), "choice");
}

#[test]
fn score_column_schema_and_legend_follow_rubric_order() {
    let field = field_for_question("urgency", &score_question()).unwrap();
    let DataType::Struct(children) = field.data_type() else {
        panic!("score column must be a struct, got {:?}", field.data_type());
    };
    let names: Vec<String> = children
        .iter()
        .map(|child| child.name().to_string())
        .collect();
    assert_eq!(names, ["score", "confidence", "probabilities"]);
    assert_eq!(children[0].data_type(), &DataType::Float64);
    let legend: Vec<String> =
        serde_json::from_str(field.metadata().get(META_LEGEND).unwrap()).unwrap();
    assert_eq!(legend, ["Can wait", "Within a few days", "Today"]);
    assert_eq!(field.metadata().get(META_JEV_TYPE).unwrap(), "score");
}

#[test]
fn choice_with_more_than_256_options_is_rejected() {
    let mut criteria = HashMap::new();
    for i in 0..=256 {
        criteria.insert(format!("option-{i:03}"), None);
    }
    let question = Question::Choice(ChoiceQuestion {
        instructions: json!("Pick one"),
        criteria,
    });
    let error = field_for_question("wide", &question).unwrap_err();
    match error {
        ApiError::InvalidBody(message) => {
            assert!(message.contains("uint8"), "{message}");
            assert!(message.contains("257"), "{message}");
        }
        other => panic!("expected InvalidBody, got {other:?}"),
    }
}

#[test]
fn request_accepts_string_object_and_array_states_and_ignores_extras() {
    let body = json!({
        "model": "mock",
        "states": ["plain text", {"cart": ["shoes"]}, [1, "two"]],
        "questions": {"refund": {"type": "noul", "instructions": "refund?"}},
        "trace_id": "tr-123",
        "client_metadata": {"service": "ci"}
    });
    let request: ArrowBatchRequest = serde_json::from_value(body).unwrap();
    assert_eq!(request.model, "mock");
    assert_eq!(request.states.len(), 3);
    assert!(matches!(request.states[0], State::Text(_)));
    assert!(matches!(request.states[1], State::Object(_)));
    assert!(matches!(request.states[2], State::Array(_)));
    assert_eq!(request.questions.len(), 1);
}

#[test]
fn request_requires_model_states_and_questions() {
    let missing_questions = json!({"model": "mock", "states": ["s"]});
    assert!(serde_json::from_value::<ArrowBatchRequest>(missing_questions).is_err());

    let missing_model = json!({
        "states": ["s"],
        "questions": {"refund": {"type": "noul", "instructions": "refund?"}}
    });
    assert!(serde_json::from_value::<ArrowBatchRequest>(missing_model).is_err());

    let missing_states = json!({
        "model": "mock",
        "questions": {"refund": {"type": "noul", "instructions": "refund?"}}
    });
    assert!(serde_json::from_value::<ArrowBatchRequest>(missing_states).is_err());
}

#[test]
fn roundtrip_reconstructs_exact_answers_and_metadata() {
    let questions = questions();
    let responses = vec![
        response("mock", sample_answers()),
        response("mock", sample_answers()),
    ];
    let (schema, batch) = encode_decode(&questions, &responses);

    // Columns follow the sorted question ids, not insertion order.
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    assert_eq!(names, ["department", "refund", "urgency"]);
    assert_eq!(batch.num_rows(), 2);

    let metadata = schema.metadata();
    assert_eq!(
        metadata.get(META_ARROW_VERSION).unwrap(),
        ARROW_MAPPING_VERSION
    );
    assert_eq!(metadata.get(META_MODEL).unwrap(), "mock");
    // Aggregate usage: 11 input / 3 output tokens per state, two states.
    assert_eq!(metadata.get(META_USAGE_INPUT_TOKENS).unwrap(), "22");
    assert_eq!(metadata.get(META_USAGE_OUTPUT_TOKENS).unwrap(), "6");

    let rows = answers_from_batch(&batch).unwrap();
    let expected = sample_answers();
    for row in &rows {
        assert_eq!(row.len(), 3);
        assert_eq!(row["refund"], expected["refund"]);
        assert_eq!(row["department"], expected["department"]);
        assert_eq!(row["urgency"], expected["urgency"]);
    }
}

#[test]
fn roundtrip_with_empty_states_keeps_the_schema() {
    let questions = questions();
    let (schema, batch) = build_batch(&questions, &[], "jev-latest").unwrap();
    assert_eq!(batch.num_rows(), 0);
    let bytes = encode_ipc_stream(schema.clone(), batch).unwrap();
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert_eq!(batch.num_rows(), 0);
    // No responses means the requested model name lands in the metadata.
    assert_eq!(
        batch.schema().metadata().get(META_MODEL).unwrap(),
        "jev-latest"
    );
    let rows = answers_from_batch(&batch).unwrap();
    assert!(rows.is_empty());
}

#[test]
fn decoder_rejects_columns_without_question_type_metadata() {
    let schema = std::sync::Arc::new(Schema::new(arrow_schema::Fields::from(vec![
        arrow_schema::Field::new("refund", DataType::Float64, false),
    ])));
    let batch = RecordBatch::try_new(
        schema,
        vec![std::sync::Arc::new(Float64Array::from(vec![0.5]))],
    )
    .unwrap();
    let error = answers_from_batch(&batch).unwrap_err();
    assert!(matches!(error, ApiError::Internal(_)));
}

#[test]
fn column_order_follows_sorted_question_ids() {
    let questions = questions();
    let ids = question_ids(&questions);
    assert_eq!(ids, vec!["department", "refund", "urgency"]);
}
