//! Unofficial Arrow bulk endpoint: `POST /v1/arrow`.
//!
//! Inspired by the proposal in ["What if Jev spoke Arrow?"](https://columnar.tech/blog/what-if-jev-spoke-arrow/):
//! one request carries many independent states plus a single shared question
//! map, and the answer comes back as an [Apache Arrow](https://arrow.apache.org/)
//! IPC stream: one row per state, one column per question.
//!
//! This surface is **not** part of the TypeSafe wire contract. It is
//! flag-gated (`openkindd --arrow on`), absent from `openapi.yaml`, and must
//! never leak into the `/v1/systemone` shapes or the SDK compatibility
//! tests. See [`docs/ARROW.md`](../../../docs/ARROW.md) for the full
//! mapping and its limits.
//!
//! Wire shape (JSON in, Arrow out):
//!
//! ```json
//! {
//!   "model": "jev-latest",
//!   "states": ["Please refund the shoes.", {"cart": ["shoes"]}],
//!   "questions": { "refund": { "type": "noul", "instructions": "Is a refund being requested?" } }
//! }
//! ```
//!
//! Each question id becomes an Arrow column whose type encodes the answer:
//!
//! | Jev type | Arrow type | Field metadata |
//! |---|---|---|
//! | Noul | `float64` | `jev.type = "noul"` |
//! | Choice | `struct<choice: uint8, confidence: float64, probabilities: fixed_size_list<float64>[N]>` | `jev.type = "choice"`, `labels` (JSON array, sorted option keys) |
//! | Score | `struct<score: float64, confidence: float64, probabilities: fixed_size_list<float64>[N]>` | `jev.type = "score"`, `legend` (JSON array of level descriptions in rubric order) |
//!
//! Columns are ordered by question id (lexicographic). Row `i` is the answer
//! for `states[i]`. All fields are non-nullable: the endpoint evaluates
//! every state or fails the whole request.

use std::collections::{BTreeMap, HashMap};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{FixedSizeListArray, Float64Array, RecordBatch};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use axum::extract::rejection::JsonRejection;
use axum::extract::State as AxumState;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use openkind_core::{Question, State, SystemRequest};
use openkind_engine::{dispatch, EngineError, EngineRegistry};
use serde::Deserialize;

use crate::error::ApiError;
use crate::AppState;

#[path = "arrow_columns.rs"]
mod columns;
#[path = "arrow_decoder.rs"]
mod decoder;
use columns::BatchBuilder;
pub use decoder::answers_from_batch;

/// Maximum projected column-buffer size and encoded IPC response size (64 MiB).
pub const MAX_ARROW_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
/// Queue-inclusive deadline for the complete bulk evaluation and encoding.
pub const ARROW_BATCH_TIMEOUT: Duration = Duration::from_secs(600);

/// `Content-Type` of the Arrow IPC stream response.
pub const ARROW_CONTENT_TYPE: &str = "application/vnd.apache.arrow.stream";

/// Maximum number of states in one bulk request. Mirrors the DoS budget of
/// [`openkind_core::MAX_QUESTIONS_PER_REQUEST`]; beyond this, callers chunk
/// their states across requests.
pub const MAX_ARROW_STATES: usize = 10_000;

/// Maximum number of Choice options a question may have on this endpoint.
/// The `choice` child column is a `uint8` index into `labels`, so more than
/// 256 options cannot be represented; core allows up to
/// [`openkind_core::MAX_CRITERIA_OPTIONS`] on `/v1/systemone`.
pub const MAX_CHOICE_LABELS: usize = u8::MAX as usize + 1;

/// Schema metadata: format version of this unofficial mapping.
pub const META_ARROW_VERSION: &str = "openkind.arrow.version";
/// Schema metadata: model that performed the evaluation.
pub const META_MODEL: &str = "openkind.model";
/// Schema metadata: aggregate input tokens across all states.
pub const META_USAGE_INPUT_TOKENS: &str = "openkind.usage.input_tokens";
/// Schema metadata: aggregate output tokens across all states.
pub const META_USAGE_OUTPUT_TOKENS: &str = "openkind.usage.output_tokens";
/// Field metadata on every question column: the Jev question type.
pub const META_JEV_TYPE: &str = "jev.type";
/// Field metadata on Choice columns: JSON array of option keys, sorted.
pub const META_LABELS: &str = "labels";
/// Field metadata on Score columns: JSON array of level descriptions, in rubric order.
pub const META_LEGEND: &str = "legend";

/// Value of [`META_ARROW_VERSION`] in every response stream.
pub const ARROW_MAPPING_VERSION: &str = "1";

/// Bulk evaluation request body for `POST /v1/arrow`.
///
/// The shape mirrors [`SystemRequest`] with `state` generalized to
/// `states`; the question map is shared by every state, exactly as the
/// article proposes. Unknown top-level fields are ignored, matching
/// `/v1/systemone` tolerance for extra metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct ArrowBatchRequest {
    /// Required. `"jev-latest"` or a registered model alias.
    pub model: String,
    /// Required. One entry per output row; may be empty (empty batch).
    pub states: Vec<State>,
    /// Required. Map of user-chosen id → typed question, shared by all states.
    pub questions: BTreeMap<String, Question>,
}

impl ArrowBatchRequest {
    /// Build the per-state [`SystemRequest`] fanned out to the engine.
    ///
    /// The question map is cloned once per state because `dispatch` takes
    /// ownership; column/row semantics are unaffected.
    fn system_request_for(&self, state: &State) -> SystemRequest {
        SystemRequest {
            state: state.clone(),
            model: self.model.clone(),
            questions: self
                .questions
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

#[cfg(test)]
/// Ordered question ids of this request (lexicographic: the column order).
fn question_ids(questions: &BTreeMap<String, Question>) -> Vec<&str> {
    questions.keys().map(String::as_str).collect()
}

/// Build a Jev/Arrow field metadata map.
fn field_metadata(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

/// Build the Arrow field for one question column, including its metadata.
fn field_for_question(id: &str, question: &Question) -> Result<Field, ApiError> {
    match question {
        Question::Noul(_) => Ok(Field::new(id, DataType::Float64, false)
            .with_metadata(field_metadata(&[(META_JEV_TYPE, "noul")]))),
        Question::Choice(choice) => {
            let labels = sorted_choice_labels(choice);
            if labels.len() > MAX_CHOICE_LABELS {
                return Err(ApiError::InvalidBody(format!(
                    "choice question `{id}` has {} options; the Arrow mapping indexes `choice` into a uint8, so at most {MAX_CHOICE_LABELS} options are supported",
                    labels.len()
                )));
            }
            let children = Fields::from(vec![
                Field::new("choice", DataType::UInt8, false),
                Field::new("confidence", DataType::Float64, false),
                fixed_probabilities_field(labels.len()),
            ]);
            let labels_json = serde_json::to_string(&labels)
                .map_err(|e| ApiError::Internal(format!("encode labels metadata: {e}")))?;
            Ok(
                Field::new_struct(id, children, false).with_metadata(field_metadata(&[
                    (META_JEV_TYPE, "choice"),
                    (META_LABELS, labels_json.as_str()),
                ])),
            )
        }
        Question::Score(score) => {
            let legend = score.criteria.clone();
            let children = Fields::from(vec![
                Field::new("score", DataType::Float64, false),
                Field::new("confidence", DataType::Float64, false),
                fixed_probabilities_field(legend.len()),
            ]);
            let legend_json = serde_json::to_string(&legend)
                .map_err(|e| ApiError::Internal(format!("encode legend metadata: {e}")))?;
            Ok(
                Field::new_struct(id, children, false).with_metadata(field_metadata(&[
                    (META_JEV_TYPE, "score"),
                    (META_LEGEND, legend_json.as_str()),
                ])),
            )
        }
    }
}

/// Sorted option keys of a Choice question: the canonical label order.
///
/// `criteria` is a wire `HashMap`, so the sorted order is the only
/// deterministic projection shared by producer and consumer.
fn sorted_choice_labels(choice: &openkind_core::ChoiceQuestion) -> Vec<String> {
    let mut labels: Vec<String> = choice.criteria.keys().cloned().collect();
    labels.sort();
    labels
}

/// Non-null `fixed_size_list<float64>[n]` child for the probability vector.
fn fixed_probabilities_field(n: usize) -> Field {
    // Core validation bounds the width before the schema is constructed.
    let item = Field::new("item", DataType::Float64, false);
    Field::new(
        "probabilities",
        DataType::FixedSizeList(Arc::new(item), n as i32),
        false,
    )
}

/// Build a validated non-null probability array.
fn fixed_probabilities_column(
    flat: Vec<f64>,
    list_size: usize,
) -> Result<FixedSizeListArray, ApiError> {
    FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float64, false)),
        list_size as i32,
        Arc::new(Float64Array::from(flat)),
        None,
    )
    .map_err(|e| ApiError::Internal(format!("assemble probabilities: {e}")))
}

/// Canonical metadata map for the response schema.
fn schema_metadata(model: &str, input_tokens: u64, output_tokens: u64) -> HashMap<String, String> {
    HashMap::from([
        (
            META_ARROW_VERSION.to_string(),
            ARROW_MAPPING_VERSION.to_string(),
        ),
        (META_MODEL.to_string(), model.to_string()),
        (
            META_USAGE_INPUT_TOKENS.to_string(),
            input_tokens.to_string(),
        ),
        (
            META_USAGE_OUTPUT_TOKENS.to_string(),
            output_tokens.to_string(),
        ),
    ])
}

/// Checked buffer projection, before allocating columns or evaluating states.
fn projected_column_bytes(
    questions: &BTreeMap<String, Question>,
    rows: usize,
) -> Result<usize, ApiError> {
    let mut per_row = 0usize;
    for question in questions.values() {
        let (fixed, width) = match question {
            Question::Noul(_) => (8usize, 0usize),
            Question::Choice(q) => (9, q.criteria.len()),
            Question::Score(q) => (16, q.criteria.len()),
        };
        let bytes = width
            .checked_mul(8)
            .and_then(|v| v.checked_add(fixed))
            .ok_or_else(size_error)?;
        per_row = per_row.checked_add(bytes).ok_or_else(size_error)?;
    }
    let bytes = per_row.checked_mul(rows).ok_or_else(size_error)?;
    if bytes > MAX_ARROW_RESPONSE_BYTES {
        return Err(size_error());
    }
    Ok(bytes)
}

fn size_error() -> ApiError {
    ApiError::InvalidBody(format!("Arrow buffers and encoded response must each fit within {MAX_ARROW_RESPONSE_BYTES} bytes; chunk the batch"))
}

fn deadline_error() -> ApiError {
    EngineError::DeadlineExceeded {
        backend: "arrow".to_string(),
        timeout_ms: ARROW_BATCH_TIMEOUT.as_millis() as u64,
    }
    .into()
}

/// Limit the writer itself so schema overhead cannot escape the projection.
struct CappedWriter {
    bytes: Vec<u8>,
    limit: usize,
    deadline: tokio::time::Instant,
    exceeded: bool,
}

impl Write for CappedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if tokio::time::Instant::now() >= self.deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Arrow batch deadline exceeded",
            ));
        }
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.limit)
        {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "Arrow response limit exceeded",
            ));
        }
        self.bytes
            .try_reserve_exact(bytes.len())
            .map_err(|e| io::Error::new(io::ErrorKind::OutOfMemory, e))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode_ipc_stream_until(
    schema: SchemaRef,
    batch: RecordBatch,
    deadline: tokio::time::Instant,
) -> Result<Vec<u8>, ApiError> {
    encode_ipc_stream_with_limit(schema, batch, deadline, MAX_ARROW_RESPONSE_BYTES)
}

fn encode_ipc_stream_with_limit(
    schema: SchemaRef,
    batch: RecordBatch,
    deadline: tokio::time::Instant,
    limit: usize,
) -> Result<Vec<u8>, ApiError> {
    let mut sink = CappedWriter {
        bytes: Vec::new(),
        limit,
        deadline,
        exceeded: false,
    };
    let result = (|| {
        let mut writer = StreamWriter::try_new(&mut sink, &schema)?;
        writer.write(&batch)?;
        writer.finish()
    })();
    if tokio::time::Instant::now() >= deadline {
        return Err(deadline_error());
    }
    if sink.exceeded {
        return Err(size_error());
    }
    result.map_err(|e| ApiError::Internal(format!("encode Arrow stream: {e}")))?;
    Ok(sink.bytes)
}

#[cfg(test)]
fn encode_ipc_stream(schema: SchemaRef, batch: RecordBatch) -> Result<Vec<u8>, ApiError> {
    encode_ipc_stream_until(
        schema,
        batch,
        tokio::time::Instant::now() + ARROW_BATCH_TIMEOUT,
    )
}

#[cfg(test)]
#[path = "arrow_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "arrow_regression_tests.rs"]
mod regression_tests;

/// Handler for the unofficial `POST /v1/arrow` bulk endpoint.
///
/// Evaluates every state against the shared question map and returns one
/// Arrow IPC stream. Errors are all-or-nothing: any per-state failure fails
/// the whole request with the standard JSON error envelope, before any
/// Arrow bytes are written.
pub async fn arrow_batch(
    AxumState(state): AxumState<Arc<AppState>>,
    req: Result<Json<ArrowBatchRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = match req {
        Ok(json) => json,
        Err(rejection) => match rejection {
            JsonRejection::BytesRejection(e) => {
                return Err(ApiError::PayloadTooLarge(e.to_string()))
            }
            JsonRejection::JsonSyntaxError(e) => return Err(ApiError::BadJson(e.to_string())),
            JsonRejection::JsonDataError(e) => return Err(ApiError::InvalidBody(e.to_string())),
            other => return Err(ApiError::InvalidBody(other.to_string())),
        },
    };

    evaluate_batch(state, req, ARROW_BATCH_TIMEOUT).await
}

async fn evaluate_batch(
    state: Arc<AppState>,
    req: ArrowBatchRequest,
    timeout: Duration,
) -> Result<Response, ApiError> {
    let deadline = tokio::time::Instant::now() + timeout;
    tokio::time::timeout_at(deadline, evaluate_batch_until(state, req, deadline))
        .await
        .map_err(|_| deadline_error())?
}

async fn evaluate_batch_until(
    state: Arc<AppState>,
    req: ArrowBatchRequest,
    deadline: tokio::time::Instant,
) -> Result<Response, ApiError> {
    if req.states.len() > MAX_ARROW_STATES {
        return Err(ApiError::InvalidBody(format!(
            "states array has {} entries; at most {MAX_ARROW_STATES} are supported",
            req.states.len()
        )));
    }
    // Empty batches still have a question contract. No engine is evaluated here.
    let representative = req.system_request_for(&State::Text(String::new()));
    openkind_core::validate_request(&representative)
        .map_err(|e| ApiError::Engine(EngineError::Invalid(e)))?;
    drop(representative);
    let engine = state
        .registry
        .get(&req.model)
        .ok_or_else(|| EngineError::UnknownModel(req.model.clone()))?;
    // Pin the handle for the whole batch while preserving dispatch validation,
    // usage estimation and telemetry. Playground updates affect later batches.
    let mut registry = EngineRegistry::new();
    registry.register(req.model.clone(), engine);
    projected_column_bytes(&req.questions, req.states.len())?;
    let mut builder = BatchBuilder::new(&req.questions, &req.model, req.states.len())?;
    // Check schema size before doing model work, even for a zero-row batch.
    let (schema, empty) = builder.empty_batch()?;
    tokio::task::spawn_blocking(move || encode_ipc_stream_until(schema, empty, deadline))
        .await
        .map_err(|e| ApiError::Internal(format!("Arrow schema task failed: {e}")))??;
    for jev_state in &req.states {
        // Mock engines can return ready futures; yield so cancellation and
        // other requests remain observable between independent evaluations.
        tokio::task::yield_now().await;
        if tokio::time::Instant::now() >= deadline {
            return Err(deadline_error());
        }
        let response = dispatch(req.system_request_for(jev_state), &registry).await?;
        builder.push(&response)?;
    }
    let (schema, batch) = builder.finish()?;
    let bytes =
        tokio::task::spawn_blocking(move || encode_ipc_stream_until(schema, batch, deadline))
            .await
            .map_err(|e| ApiError::Internal(format!("Arrow encoding task failed: {e}")))??;
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, ARROW_CONTENT_TYPE)],
        bytes,
    )
        .into_response())
}

#[cfg(test)]
fn build_batch(
    questions: &BTreeMap<String, Question>,
    responses: &[openkind_core::SystemResponse],
    fallback_model: &str,
) -> Result<(SchemaRef, RecordBatch), ApiError> {
    projected_column_bytes(questions, responses.len())?;
    let mut builder = BatchBuilder::new(questions, fallback_model, responses.len())?;
    for response in responses {
        builder.push(response)?;
    }
    builder.finish()
}
