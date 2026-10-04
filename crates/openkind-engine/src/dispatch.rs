//! Request dispatching, metrics tracking, and token estimation.

use openkind_core::{Answer, ResponseContract, SystemRequest, SystemResponse};
use tracing::instrument;

use crate::error::{EngineError, EngineResult};
use crate::registry::EngineRegistry;

/// Validate + dispatch. The HTTP and gRPC layers both call this — it
/// contains the cross-cutting logic (validation, telemetry, routing).
#[instrument(skip(req, registry), fields(model = %req.model, n_questions = req.questions.len()))]
pub async fn dispatch(
    req: SystemRequest,
    registry: &EngineRegistry,
) -> EngineResult<SystemResponse> {
    metrics::counter!("openkind_requests_total").increment(1);

    let mut observation = DispatchObservation {
        start: std::time::Instant::now(),
        outcome: "cancelled",
    };
    let result = dispatch_inner(req, registry).await;
    observation.outcome = match &result {
        Ok(_) => "success",
        Err(EngineError::Invalid(_)) => "invalid_request",
        Err(EngineError::UnknownModel(_)) => "unknown_model",
        Err(EngineError::Unsupported { .. }) => "unsupported",
        Err(EngineError::Overloaded { .. }) => "overloaded",
        Err(EngineError::DeadlineExceeded { .. }) => "deadline_exceeded",
        Err(EngineError::BackendValidation { .. }) => "backend_validation",
        Err(EngineError::Backend { .. }) => "backend_error",
    };
    result
}

struct DispatchObservation {
    start: std::time::Instant,
    outcome: &'static str,
}

impl Drop for DispatchObservation {
    fn drop(&mut self) {
        // Drop also observes callers abandoning an in-flight dispatch.
        metrics::histogram!("openkind_request_duration_ms")
            .record(duration_millis(self.start.elapsed()));
        metrics::counter!("openkind_request_outcomes_total", "outcome" => self.outcome)
            .increment(1);
    }
}

async fn dispatch_inner(
    req: SystemRequest,
    registry: &EngineRegistry,
) -> EngineResult<SystemResponse> {
    let requested_model = req.model.clone();
    let engine = registry
        .get(&req.model)
        .ok_or_else(|| EngineError::UnknownModel(req.model.clone()))?;

    let contract = ResponseContract::from_request(&req)?;
    let input_tokens = engine.estimate_input_tokens(&req);

    let mut resp = engine.evaluate(req).await?;

    if resp.model != requested_model {
        return Err(EngineError::Backend {
            backend: engine.backend_id().to_string(),
            message: format!(
                "backend returned model `{}`, expected registered alias `{requested_model}`",
                resp.model
            ),
        });
    }

    // Never forward a contract-violating engine response to the client:
    // a bad answer shape is a backend fault, so it maps to BackendValidation (500),
    // not to a client-facing 422.
    if let Err(validation) = contract.validate(&resp) {
        return Err(EngineError::BackendValidation {
            backend: engine.backend_id().to_string(),
            source: validation,
        });
    }

    // If the backend didn't fill in usage, do it from the estimator.
    // Real backends will fill it precisely.
    if resp.usage.input_tokens == 0 {
        resp.usage.input_tokens = input_tokens;
    }
    if resp.usage.output_tokens == 0 {
        resp.usage.output_tokens = estimate_output_tokens(&resp);
    }

    metrics::counter!("openkind_responses_total").increment(1);

    Ok(resp)
}

fn estimate_output_tokens(resp: &SystemResponse) -> u32 {
    // Noul = 1 token. Choice = 1 (just the picked label).
    // Score = ~ level descriptions worth of tokens.
    let sum: usize = resp
        .answers
        .values()
        .map(|a| match a {
            Answer::Noul(_) => 1,
            Answer::Choice(_) => 1,
            Answer::Score(_) => 4,
        })
        .sum();
    u32::try_from(sum).unwrap_or(u32::MAX)
}

fn duration_millis(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod telemetry_tests {
    use super::*;
    use crate::{DecisionEngine, MockEngine};
    use async_trait::async_trait;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use openkind_core::{NoulQuestion, Question, State};
    use std::{
        future::Future,
        sync::Arc,
        task::{Context, Poll, Waker},
        time::Duration,
    };

    struct OutcomeEngine(&'static str);
    #[async_trait]
    impl DecisionEngine for OutcomeEngine {
        fn backend_id(&self) -> &str {
            "outcome-test"
        }
        async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
            match self.0 {
                "unsupported" => Err(EngineError::Unsupported {
                    backend: "test".into(),
                    message: "unsupported".into(),
                }),
                "overloaded" => Err(EngineError::Overloaded {
                    backend: "test".into(),
                    retry_after_ms: 1,
                }),
                "deadline_exceeded" => Err(EngineError::DeadlineExceeded {
                    backend: "test".into(),
                    timeout_ms: 1,
                }),
                "backend_error" => Err(EngineError::Backend {
                    backend: "test".into(),
                    message: "error".into(),
                }),
                "backend_validation" => {
                    let mut response = MockEngine::new().evaluate(req).await?;
                    response.answers.clear();
                    Ok(response)
                }
                "cancelled" => std::future::pending().await,
                _ => unreachable!(),
            }
        }
    }

    fn request() -> SystemRequest {
        SystemRequest {
            model: "mock".into(),
            state: State::Text("private text".into()),
            questions: [(
                "q".into(),
                Question::Noul(NoulQuestion {
                    instructions: serde_json::json!("?"),
                    criteria: None,
                }),
            )]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn duration_keeps_sub_millisecond_precision() {
        assert_eq!(duration_millis(Duration::from_micros(125)), 0.125);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn every_dispatch_observes_duration_and_fixed_outcome_including_cancellation() {
        let recorder = DebuggingRecorder::new();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let mut registry = EngineRegistry::new();
        registry.register("mock", Arc::new(MockEngine::new()));
        dispatch(request(), &registry).await.unwrap();
        let mut invalid = request();
        invalid.questions.clear();
        assert!(dispatch(invalid, &registry).await.is_err());
        assert!(dispatch(request(), &EngineRegistry::new()).await.is_err());
        for outcome in [
            "unsupported",
            "overloaded",
            "deadline_exceeded",
            "backend_error",
            "backend_validation",
        ] {
            registry.register("mock", Arc::new(OutcomeEngine(outcome)));
            assert!(dispatch(request(), &registry).await.is_err());
        }
        registry.register("mock", Arc::new(OutcomeEngine("cancelled")));
        let mut future = Box::pin(dispatch(request(), &registry));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        drop(future);

        let snapshot = recorder.snapshotter().snapshot().into_vec();
        let mut outcomes = std::collections::BTreeSet::new();
        for (key, _, _, value) in snapshot {
            match key.key().name() {
                "openkind_requests_total" => assert_eq!(value, DebugValue::Counter(9)),
                "openkind_responses_total" => assert_eq!(value, DebugValue::Counter(1)),
                "openkind_request_duration_ms" => {
                    let DebugValue::Histogram(values) = value else {
                        panic!("expected histogram")
                    };
                    assert_eq!(values.len(), 9);
                    assert!(values.iter().all(|value| value.0 > 0.0));
                    assert!(values.iter().any(|value| value.0.fract() > 0.0));
                }
                "openkind_request_outcomes_total" => {
                    assert_eq!(value, DebugValue::Counter(1));
                    let labels: Vec<_> = key.key().labels().collect();
                    assert_eq!(labels.len(), 1);
                    assert_eq!(labels[0].key(), "outcome");
                    outcomes.insert(labels[0].value().to_owned());
                }
                _ => panic!("unexpected metric"),
            }
        }
        assert_eq!(
            outcomes,
            [
                "success",
                "invalid_request",
                "unknown_model",
                "unsupported",
                "overloaded",
                "deadline_exceeded",
                "backend_error",
                "backend_validation",
                "cancelled"
            ]
            .into_iter()
            .map(str::to_owned)
            .collect()
        );
    }
}
