//! Request dispatching, metrics tracking, and token estimation.

use std::collections::HashMap;

use opendecision_core::{
    validate_request, validate_response, Answer, Question, SystemRequest, SystemResponse,
};
use tracing::instrument;

use crate::error::{EngineError, EngineResult};
use crate::registry::EngineRegistry;

/// Per-question reference data for response validation: question id → the
/// choice-criteria keys a choice answer must respect (empty for noul/score).
/// Passing the full id set lets `validate_response` also enforce that the
/// engine answered exactly the requested questions, no more and no less.
fn response_criteria(req: &SystemRequest) -> HashMap<String, Vec<String>> {
    req.questions
        .iter()
        .map(|(id, q)| {
            let keys = match q {
                Question::Noul(_) => Vec::new(),
                Question::Choice(c) => {
                    let mut keys: Vec<String> = c.criteria.keys().cloned().collect();
                    keys.sort();
                    keys
                }
                Question::Score(s) => (0..s.criteria.len()).map(|i| i.to_string()).collect(),
            };
            (id.clone(), keys)
        })
        .collect()
}

/// Validate + dispatch. The HTTP and gRPC layers both call this — it
/// contains the cross-cutting logic (validation, telemetry, routing).
#[instrument(skip(req, registry), fields(model = %req.model, n_questions = req.questions.len()))]
pub async fn dispatch(
    req: SystemRequest,
    registry: &EngineRegistry,
) -> EngineResult<SystemResponse> {
    metrics::counter!("opendecision_requests_total").increment(1);

    let start = std::time::Instant::now();
    let engine = registry
        .get(&req.model)
        .ok_or_else(|| EngineError::UnknownModel(req.model.clone()))?;

    validate_request(&req)?;
    let criteria = response_criteria(&req);
    let input_tokens = engine.estimate_input_tokens(&req);

    let mut resp = engine.evaluate(req).await?;

    // Never forward a contract-violating engine response to the client:
    // a bad answer shape is a backend fault, so it maps to Backend (500),
    // not to a client-facing 422.
    if let Err(validation) = validate_response(&resp, &criteria) {
        return Err(EngineError::Backend {
            backend: engine.backend_id().to_string(),
            message: format!("backend returned an invalid response: {validation}"),
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

    let elapsed_ms = start.elapsed().as_millis() as f64;
    metrics::histogram!("opendecision_request_duration_ms").record(elapsed_ms);
    metrics::counter!("opendecision_responses_total").increment(1);

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
