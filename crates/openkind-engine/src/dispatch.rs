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

    let start = std::time::Instant::now();
    let engine = registry
        .get(&req.model)
        .ok_or_else(|| EngineError::UnknownModel(req.model.clone()))?;

    let contract = ResponseContract::from_request(&req)?;
    let input_tokens = engine.estimate_input_tokens(&req);

    let mut resp = engine.evaluate(req).await?;

    // Never forward a contract-violating engine response to the client:
    // a bad answer shape is a backend fault, so it maps to Backend (500),
    // not to a client-facing 422.
    if let Err(validation) = contract.validate(&resp) {
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
    metrics::histogram!("openkind_request_duration_ms").record(elapsed_ms);
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
