//! Behavioral regressions for bulk admission, mapping validation and decoding.

use super::*;
use arrow_array::{ArrayRef, StructArray, UInt8Array};
use axum::{body::Body, http::Request, Router};
use http_body_util::BodyExt;
use openkind_core::{Answer, NoulAnswer, SystemResponse};
use openkind_engine::{DecisionEngine, EngineResult, MockEngine};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use tower::ServiceExt;

struct TestEngine {
    calls: AtomicUsize,
    finished: AtomicUsize,
    slot: Arc<tokio::sync::Semaphore>,
    fail_at: Option<usize>,
    pending: bool,
    replace_registry: Option<EngineRegistry>,
    changed_model: bool,
    changed_score: u8,
}

struct Finish<'a>(&'a AtomicUsize);
impl Drop for Finish<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl TestEngine {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            slot: Arc::new(tokio::sync::Semaphore::new(1)),
            fail_at: None,
            pending: false,
            replace_registry: None,
            changed_model: false,
            changed_score: 0,
        }
    }
}

#[async_trait::async_trait]
impl DecisionEngine for TestEngine {
    fn backend_id(&self) -> &str {
        "arrow-test"
    }
    async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
        let _slot = self
            .slot
            .clone()
            .try_acquire_owned()
            .map_err(|_| EngineError::Overloaded {
                backend: self.backend_id().into(),
                retry_after_ms: 750,
            })?;
        let ordinal = self.calls.fetch_add(1, Ordering::SeqCst);
        let _finish = Finish(&self.finished);
        if self.pending {
            std::future::pending::<()>().await;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
        if self.fail_at == Some(ordinal) {
            return Err(EngineError::Overloaded {
                backend: self.backend_id().into(),
                retry_after_ms: 750,
            });
        }
        if ordinal == 0 {
            if let Some(registry) = &self.replace_registry {
                registry.unregister(&req.model);
                registry.register_if_absent(req.model.clone(), Arc::new(MockEngine::new()));
            }
        }
        let value = match &req.state {
            State::Text(text) => text.parse().unwrap_or(0.5),
            _ => 0.5,
        };
        let mut response = MockEngine::new().evaluate(req).await?;
        for answer in response.answers.values_mut() {
            match answer {
                Answer::Noul(answer) => answer.noul = value,
                Answer::Score(answer) if self.changed_score == 1 => {
                    answer
                        .legend
                        .insert("0".into(), "backend changed the rubric".into());
                }
                Answer::Score(answer) if self.changed_score == 2 => {
                    answer.legend.insert("2".into(), "extra level".into());
                    answer.probabilities.insert("2".into(), 0.0);
                }
                Answer::Score(answer) if self.changed_score == 3 => {
                    let legend = answer.legend.remove("0").unwrap();
                    let probability = answer.probabilities.remove("0").unwrap();
                    answer.legend.insert("00".into(), legend);
                    answer.probabilities.insert("00".into(), probability);
                }
                _ => {}
            }
        }
        if self.changed_model && ordinal > 0 {
            response.model = "different-model".into();
        }
        Ok(response)
    }
}

fn state(engine: Arc<TestEngine>) -> Arc<AppState> {
    let mut registry = EngineRegistry::new();
    registry.register("mock", engine);
    Arc::new(AppState::new(registry))
}

fn body(states: serde_json::Value) -> serde_json::Value {
    json!({"model":"mock", "states": states, "questions":{"q":{"type":"noul","instructions":"test"}}})
}

fn app(engine: Arc<TestEngine>, limit: usize, limiter: crate::RateLimiter) -> Router {
    crate::router_daemon_with_arrow(
        (*state(engine)).clone(),
        crate::AuthConfig::default(),
        limit,
        limiter,
        false,
        true,
    )
}

async fn post(router: Router, value: serde_json::Value) -> Response {
    send(router, serde_json::to_vec(&value).unwrap()).await
}

async fn send(router: Router, bytes: Vec<u8>) -> Response {
    let mut req = Request::builder()
        .method("POST")
        .uri("/v1/arrow")
        .header("content-type", "application/json")
        .header("x-typesafe-request-id", "arrow-regression")
        .body(Body::from(bytes))
        .unwrap();
    req.extensions_mut().insert(axum::extract::ConnectInfo(
        "127.0.0.1:1234".parse::<std::net::SocketAddr>().unwrap(),
    ));
    let response = router.oneshot(req).await.unwrap();
    assert!(response.headers().contains_key("x-typesafe-request-id"));
    response
}

async fn error(response: Response, status: StatusCode, code: &str) {
    assert_eq!(response.status(), status);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    let value: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(value["error"]["code"], code);
}

async fn decoded(response: Response) -> RecordBatch {
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        ARROW_CONTENT_TYPE
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let mut reader =
        arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    batch
}

#[tokio::test]
async fn empty_batches_validate_questions_and_models_without_evaluation() {
    let engine = Arc::new(TestEngine::new());
    let router = app(
        engine.clone(),
        crate::http::MAX_PAYLOAD_SIZE_BYTES,
        crate::RateLimiter::disabled(),
    );
    for question in [
        json!({"type":"noul","instructions":""}),
        json!({"type":"choice","instructions":"pick","criteria":{}}),
        json!({"type":"score","instructions":"rate","criteria":[]}),
        json!({"type":"score","instructions":"rate","criteria":["one"]}),
        json!({"type":"score","instructions":"rate","criteria":["one",""]}),
        json!({"type":"score","instructions":"rate","criteria":vec!["x";10_001]}),
    ] {
        let mut value = body(json!([]));
        value["questions"]["q"] = question;
        error(
            post(router.clone(), value).await,
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_body",
        )
        .await;
    }
    let mut value = body(json!([]));
    value["model"] = json!("absent");
    error(
        post(router.clone(), value).await,
        StatusCode::NOT_FOUND,
        "unknown_model",
    )
    .await;
    let mut value = body(json!([]));
    value["questions"] = json!((0..10_001)
        .map(|i| (i.to_string(), json!({"type":"noul","instructions":"x"})))
        .collect::<serde_json::Map<_, _>>());
    error(
        post(router.clone(), value).await,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    assert_eq!(
        decoded(post(router, body(json!([]))).await)
            .await
            .num_rows(),
        0
    );
    assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn sequential_batch_preserves_state_order_and_pins_the_engine() {
    let registry = EngineRegistry::new();
    let engine = Arc::new(TestEngine {
        replace_registry: Some(registry.clone()),
        ..TestEngine::new()
    });
    registry.register_if_absent("mock".into(), engine.clone());
    let response = evaluate_batch(
        Arc::new(AppState::new(registry.clone())),
        serde_json::from_value(body(json!(["0.1", "0.2", "0.3"]))).unwrap(),
        None,
        ARROW_BATCH_TIMEOUT,
    )
    .await
    .unwrap();
    let rows = answers_from_batch(&decoded(response).await).unwrap();
    for (row, value) in rows.iter().zip([0.1, 0.2, 0.3]) {
        assert_eq!(row["q"], Answer::Noul(NoulAnswer { noul: value }));
    }
    assert_eq!(engine.calls.load(Ordering::SeqCst), 3);
    assert_eq!(registry.get("mock").unwrap().backend_id(), "mock");
}

#[tokio::test]
async fn first_failure_stops_later_states_and_keeps_retry_headers() {
    let engine = Arc::new(TestEngine {
        fail_at: Some(1),
        ..TestEngine::new()
    });
    let response = post(
        app(engine.clone(), 1024, crate::RateLimiter::disabled()),
        body(json!(["0.1", "0.2", "0.3", "0.4"])),
    )
    .await;
    assert_eq!(response.headers().get("retry-after-ms").unwrap(), "750");
    assert_eq!(response.headers().get("retry-after").unwrap(), "1");
    error(response, StatusCode::from_u16(529).unwrap(), "overloaded").await;
    assert_eq!(engine.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn batch_deadline_cancels_active_evaluation_without_starting_later_states() {
    let engine = Arc::new(TestEngine {
        pending: true,
        ..TestEngine::new()
    });
    let result = evaluate_batch(
        state(engine.clone()),
        serde_json::from_value(body(json!(["0.1", "0.2"]))).unwrap(),
        None,
        Duration::from_millis(50),
    )
    .await;
    error(
        result.unwrap_err().into_response(),
        StatusCode::GATEWAY_TIMEOUT,
        "deadline_exceeded",
    )
    .await;
    assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.finished.load(Ordering::SeqCst), 1);
    assert_eq!(engine.slot.available_permits(), 1);
}

#[tokio::test]
async fn changing_response_model_or_score_mapping_fails_the_whole_batch() {
    for changed_score in 1..=3 {
        let engine = Arc::new(TestEngine {
            changed_score,
            ..TestEngine::new()
        });
        let mut value = body(json!(["s", "t"]));
        value["questions"]["q"] =
            json!({"type":"score","instructions":"rate","criteria":["low","high"]});
        error(
            post(
                app(engine.clone(), 1024, crate::RateLimiter::disabled()),
                value,
            )
            .await,
            StatusCode::INTERNAL_SERVER_ERROR,
            "backend_error",
        )
        .await;
        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    }
    let engine = Arc::new(TestEngine {
        changed_model: true,
        ..TestEngine::new()
    });
    error(
        post(
            app(engine, 1024, crate::RateLimiter::disabled()),
            body(json!(["0.1", "0.2"])),
        )
        .await,
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
    )
    .await;
}

#[tokio::test]
async fn projected_output_and_state_limits_reject_before_evaluation() {
    let engine = Arc::new(TestEngine::new());
    let router = app(
        engine.clone(),
        crate::http::MAX_PAYLOAD_SIZE_BYTES,
        crate::RateLimiter::disabled(),
    );
    error(
        post(router.clone(), body(json!(vec!["s"; MAX_ARROW_STATES + 1]))).await,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    let mut value = body(json!(vec!["s"; 1000]));
    value["questions"]["q"] =
        json!({"type":"score","instructions":"rate","criteria":vec!["level";10_000]});
    error(
        post(router, value).await,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn payload_syntax_and_rate_limit_errors_use_standard_json() {
    let engine = Arc::new(TestEngine::new());
    error(
        post(
            app(engine.clone(), 16, crate::RateLimiter::disabled()),
            body(json!(["s"])),
        )
        .await,
        StatusCode::PAYLOAD_TOO_LARGE,
        "payload_too_large",
    )
    .await;
    error(
        send(
            app(engine.clone(), 1024, crate::RateLimiter::disabled()),
            b"{".to_vec(),
        )
        .await,
        StatusCode::BAD_REQUEST,
        "bad_json",
    )
    .await;
    let limiter = crate::RateLimiter::new(crate::RateLimitConfig {
        max_requests: 1,
        window: Duration::from_secs(60),
    });
    error(
        post(
            app(engine.clone(), 1024, limiter),
            body(json!(["0.1", "0.2"])),
        )
        .await,
        StatusCode::TOO_MANY_REQUESTS,
        "rate_limited",
    )
    .await;
    assert_eq!(engine.calls.load(Ordering::SeqCst), 0);

    let limiter = crate::RateLimiter::new(crate::RateLimitConfig {
        max_requests: 2,
        window: Duration::from_secs(60),
    });
    let router = app(engine.clone(), 1024, limiter);
    decoded(post(router.clone(), body(json!(["0.1", "0.2"]))).await).await;
    let response = post(router, body(json!(["0.1"]))).await;
    assert!(response.headers().contains_key("retry-after"));
    error(response, StatusCode::TOO_MANY_REQUESTS, "rate_limited").await;
    assert_eq!(engine.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn projected_buffer_boundary_and_overflow_are_checked() {
    let questions = serde_json::from_value::<ArrowBatchRequest>(body(json!([])))
        .unwrap()
        .questions;
    assert_eq!(
        projected_column_bytes(&questions, MAX_ARROW_RESPONSE_BYTES / 8).unwrap(),
        MAX_ARROW_RESPONSE_BYTES
    );
    assert!(projected_column_bytes(&questions, MAX_ARROW_RESPONSE_BYTES / 8 + 1).is_err());
    assert!(projected_column_bytes(&questions, usize::MAX).is_err());
    assert_eq!(arrow_work_units(MAX_ARROW_STATES, 8), ARROW_ADMISSION_UNITS);
    assert_eq!(
        arrow_work_units(1, MAX_ARROW_RESPONSE_BYTES),
        ARROW_ADMISSION_UNITS
    );
}

#[test]
fn capped_writer_accepts_boundary_and_rejects_overflow_or_deadline() {
    let mut writer = CappedWriter {
        bytes: Vec::new(),
        limit: 8,
        deadline: tokio::time::Instant::now() + ARROW_BATCH_TIMEOUT,
        exceeded: false,
    };
    writer.write_all(&[0; 8]).unwrap();
    assert_eq!(
        writer.write(&[0]).unwrap_err().kind(),
        io::ErrorKind::FileTooLarge
    );
    assert_eq!(writer.bytes.len(), 8);
    writer.deadline = tokio::time::Instant::now();
    assert_eq!(
        writer.write(&[]).unwrap_err().kind(),
        io::ErrorKind::TimedOut
    );
}

fn batch(fields: Vec<Field>, arrays: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(fields).with_metadata(schema_metadata("mock", 0, 0))),
        arrays,
    )
    .unwrap()
}

#[test]
fn decoder_rejects_null_noul_duplicate_ids_and_unknown_versions() {
    let field = field_for_question("q", &super::tests::noul_question()).unwrap();
    let null = batch(
        vec![field.clone().with_nullable(true)],
        vec![Arc::new(Float64Array::from(vec![None]))],
    );
    assert!(answers_from_batch(&null).is_err());
    let duplicate = batch(
        vec![field.clone(), field.clone()],
        vec![
            Arc::new(Float64Array::from(vec![0.5])),
            Arc::new(Float64Array::from(vec![0.5])),
        ],
    );
    assert!(answers_from_batch(&duplicate).is_err());
    let unknown = RecordBatch::try_new(
        Arc::new(
            Schema::new(vec![field])
                .with_metadata(HashMap::from([(META_ARROW_VERSION.into(), "2".into())])),
        ),
        vec![Arc::new(Float64Array::from(vec![0.5]))],
    )
    .unwrap();
    assert!(answers_from_batch(&unknown).is_err());
}

#[test]
fn decoder_rejects_invalid_numeric_values() {
    let field = field_for_question("q", &super::tests::noul_question()).unwrap();
    for value in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(answers_from_batch(&batch(
            vec![field.clone()],
            vec![Arc::new(Float64Array::from(vec![value]))]
        ))
        .is_err());
    }
}

#[test]
fn decoder_preserves_sliced_rows_and_repeated_score_descriptions() {
    let mut questions = super::tests::questions();
    if let Question::Score(q) = questions.get_mut("urgency").unwrap() {
        q.criteria[1] = q.criteria[0].clone();
    }
    let mut first = super::tests::sample_answers();
    if let Answer::Score(a) = first.get_mut("urgency").unwrap() {
        a.legend.insert("1".into(), "Can wait".into());
    }
    let mut second = first.clone();
    second.insert("refund".into(), Answer::Noul(NoulAnswer { noul: 0.2 }));
    let responses = vec![
        super::tests::response("mock", first),
        super::tests::response("mock", second.clone()),
    ];
    let (_, full) = build_batch(&questions, &responses, "mock").unwrap();
    let rows = answers_from_batch(&full.slice(1, 1)).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["refund"], second["refund"]);
    assert_eq!(rows[0]["urgency"], second["urgency"]);
}

#[test]
fn decoder_rejects_choice_index_labels_width_and_child_schema_errors() {
    let question = super::tests::choice_question();
    let field = field_for_question("q", &question).unwrap();
    let make = |field: Field, index: u8, width: usize, confidence: f64, probs: Vec<f64>| {
        let children = Fields::from(vec![
            Field::new("choice", DataType::UInt8, false),
            Field::new("confidence", DataType::Float64, false),
            fixed_probabilities_field(width),
        ]);
        let array = StructArray::new(
            children.clone(),
            vec![
                Arc::new(UInt8Array::from(vec![index])),
                Arc::new(Float64Array::from(vec![confidence])),
                Arc::new(fixed_probabilities_column(probs, width).unwrap()),
            ],
            None,
        );
        batch(
            vec![Field::new_struct("q", children, false).with_metadata(field.metadata().clone())],
            vec![Arc::new(array)],
        )
    };
    assert!(answers_from_batch(&make(field.clone(), 3, 3, 0.5, vec![0.2, 0.3, 0.5])).is_err());
    assert!(answers_from_batch(&make(field.clone(), 0, 2, 0.5, vec![0.5, 0.5])).is_err());
    assert!(answers_from_batch(&make(field.clone(), 0, 3, 1.1, vec![0.2, 0.3, 0.5])).is_err());
    assert!(answers_from_batch(&make(field.clone(), 0, 3, 0.5, vec![0.2, 0.3, 0.1])).is_err());
    for labels in [json!([]), json!(["a", "a", "b"]), json!(["z", "a", "b"])] {
        let mut metadata = field.metadata().clone();
        metadata.insert(META_LABELS, labels.to_string());
        assert!(answers_from_batch(&make(
            field.clone().with_metadata(metadata),
            0,
            3,
            0.5,
            vec![0.2, 0.3, 0.5]
        ))
        .is_err());
    }
    let fields = Fields::from(vec![Field::new("choice", DataType::UInt8, false)]);
    let array = StructArray::new(
        fields.clone(),
        vec![Arc::new(UInt8Array::from(vec![0]))],
        None,
    );
    assert!(answers_from_batch(&batch(
        vec![Field::new_struct("q", fields, false).with_metadata(field.metadata().clone())],
        vec![Arc::new(array)]
    ))
    .is_err());
}

#[test]
fn decoder_rejects_nulls_at_every_struct_nesting_level() {
    use arrow_array::builder::FixedSizeListBuilder;
    use arrow_array::builder::Float64Builder;
    use arrow_array::builder::StructBuilder;
    use arrow_array::builder::UInt8Builder;
    for level in 0..5 {
        let fields = Fields::from(vec![
            Field::new("choice", DataType::UInt8, level == 1),
            Field::new("confidence", DataType::Float64, level == 4),
            Field::new(
                "probabilities",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float64, level == 3)),
                    3,
                ),
                level == 2,
            ),
        ]);
        let item = Arc::new(Field::new("item", DataType::Float64, level == 3));
        let mut builder = StructBuilder::new(
            fields.clone(),
            vec![
                Box::new(UInt8Builder::new()),
                Box::new(Float64Builder::new()),
                Box::new(FixedSizeListBuilder::new(Float64Builder::new(), 3).with_field(item)),
            ],
        );
        builder
            .field_builder::<UInt8Builder>(0)
            .unwrap()
            .append_option(if level == 1 { None } else { Some(0) });
        builder
            .field_builder::<Float64Builder>(1)
            .unwrap()
            .append_option(if level == 4 { None } else { Some(0.5) });
        let probabilities = builder
            .field_builder::<FixedSizeListBuilder<Float64Builder>>(2)
            .unwrap();
        probabilities
            .values()
            .append_option(if level == 3 { None } else { Some(0.2) });
        probabilities.values().append_value(0.3);
        probabilities.values().append_value(0.5);
        probabilities.append(level != 2);
        builder.append(level != 0);
        let field = Field::new_struct("q", fields, level == 0).with_metadata(
            field_for_question("q", &super::tests::choice_question())
                .unwrap()
                .metadata()
                .clone(),
        );
        assert!(answers_from_batch(&batch(vec![field], vec![Arc::new(builder.finish())])).is_err());
    }
}

#[tokio::test]
async fn maximum_state_count_succeeds_with_mock_dispatch() {
    let mut registry = EngineRegistry::new();
    registry.register("mock", Arc::new(MockEngine::new()));
    let router = crate::router_daemon_with_arrow(
        AppState::new(registry),
        crate::AuthConfig::default(),
        crate::http::MAX_PAYLOAD_SIZE_BYTES,
        crate::RateLimiter::disabled(),
        false,
        true,
    );
    assert_eq!(
        decoded(post(router, body(json!(vec!["s"; MAX_ARROW_STATES]))).await)
            .await
            .num_rows(),
        MAX_ARROW_STATES
    );
}

#[test]
fn choice_256_labels_roundtrips_the_last_uint8_index() {
    let criteria: HashMap<String, Option<String>> =
        (0..256).map(|i| (format!("option-{i:03}"), None)).collect();
    let questions = BTreeMap::from([(
        "q".into(),
        Question::Choice(openkind_core::ChoiceQuestion {
            instructions: json!("pick"),
            criteria,
        }),
    )]);
    let answer = Answer::Choice(openkind_core::ChoiceAnswer {
        choice: "option-255".into(),
        confidence: 1.0,
        probabilities: (0..256)
            .map(|i| (format!("option-{i:03}"), if i == 255 { 1.0 } else { 0.0 }))
            .collect(),
    });
    let response =
        super::tests::response("mock", [("q".into(), answer.clone())].into_iter().collect());
    let (_, batch) = build_batch(&questions, &[response], "mock").unwrap();
    assert_eq!(answers_from_batch(&batch).unwrap()[0]["q"], answer);
}

#[test]
fn encoder_cap_includes_schema_batch_and_end_marker() {
    let questions = super::tests::questions();
    let (schema, batch) = build_batch(
        &questions,
        &[super::tests::response(
            "mock",
            super::tests::sample_answers(),
        )],
        "mock",
    )
    .unwrap();
    let deadline = tokio::time::Instant::now() + ARROW_BATCH_TIMEOUT;
    let bytes = encode_ipc_stream_until(schema.clone(), batch.clone(), deadline).unwrap();
    assert_eq!(
        encode_ipc_stream_with_limit(schema.clone(), batch.clone(), deadline, bytes.len())
            .unwrap()
            .len(),
        bytes.len()
    );
    assert!(matches!(
        encode_ipc_stream_with_limit(schema.clone(), batch.clone(), deadline, bytes.len() - 1),
        Err(ApiError::InvalidBody(_))
    ));
    assert!(matches!(
        encode_ipc_stream_until(schema, batch, tokio::time::Instant::now()),
        Err(ApiError::Engine(EngineError::DeadlineExceeded { .. }))
    ));
}

#[tokio::test]
async fn maximum_score_width_is_valid_even_without_states() {
    let engine = Arc::new(TestEngine::new());
    let mut value = body(json!([]));
    value["questions"]["q"] = json!({"type":"score","instructions":"rate","criteria":vec!["repeated description";10_000]});
    let batch = decoded(
        post(
            app(
                engine.clone(),
                crate::http::MAX_PAYLOAD_SIZE_BYTES,
                crate::RateLimiter::disabled(),
            ),
            value,
        )
        .await,
    )
    .await;
    assert!(answers_from_batch(&batch).unwrap().is_empty());
    assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn decoder_rejects_missing_and_malformed_type_metadata() {
    let field = Field::new("q", DataType::Float64, false);
    assert!(answers_from_batch(&batch(
        vec![field.clone()],
        vec![Arc::new(Float64Array::from(vec![0.5]))]
    ))
    .is_err());
    assert!(answers_from_batch(&batch(
        vec![field.with_metadata(HashMap::from([(META_JEV_TYPE.into(), "text".into())]))],
        vec![Arc::new(Float64Array::from(vec![0.5]))]
    ))
    .is_err());
}
