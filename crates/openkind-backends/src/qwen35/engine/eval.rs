//! Offline model evaluation task execution.

use std::collections::HashMap;

use openkind_core::{SystemRequest, SystemResponse, Usage};
use openkind_engine::EngineError;
use openkind_runtime::peak_resident_bytes;

use super::canonical::state_text;
use super::mapping::{
    answer_from_distribution, instruction_text, question_candidates, EncodedQuestion,
};
use super::EngineInner;
use crate::qwen35::backbone::{ControlledExecutor, NestedQuestion, RootCachingExecutor};
use crate::qwen35::{
    choose_strategy, choose_strategy_with_cached_root, run_strategy, CandidateText,
    ExecutionControl, Qwen35Error, StrategyRequest,
};

pub(super) fn evaluate_request(
    inner: &EngineInner,
    request: SystemRequest,
    control: &ExecutionControl,
) -> Result<SystemResponse, Qwen35Error> {
    control.check()?;
    let state = state_text(&request.state);
    let mut question_ids: Vec<_> = request.questions.keys().cloned().collect();
    question_ids.sort();
    let mut encoded = Vec::with_capacity(question_ids.len());
    let mut root_ids: Option<Vec<u32>> = None;
    for id in question_ids {
        let question = request
            .questions
            .get(&id)
            .expect("question id came from the request map");
        let (primitive, labels, criteria) = question_candidates(question)?;
        let instruction = instruction_text(question)?;
        let candidates: Vec<_> = labels
            .iter()
            .zip(&criteria)
            .map(|(label, criterion)| CandidateText::new(label, criterion))
            .collect();
        // The root depends on the state alone, so encode it once, on the first
        // question, after that question's own mapping errors have surfaced.
        let root_len = match &root_ids {
            Some(root) => root.len(),
            None => {
                let root = inner.tokenizer.encode_state_root(&state)?;
                let len = root.len();
                root_ids = Some(root);
                len
            }
        };
        let (question_ids, candidate_suffix_ids) =
            inner
                .tokenizer
                .encode_state_first_branch(root_len, &instruction, &candidates)?;
        encoded.push(EncodedQuestion {
            id,
            primitive,
            labels,
            criteria,
            question_ids,
            candidate_suffix_ids,
        });
    }
    let suffix_refs: Vec<Vec<&[u32]>> = encoded
        .iter()
        .map(|question| {
            question
                .candidate_suffix_ids
                .iter()
                .map(Vec::as_slice)
                .collect()
        })
        .collect();
    let plans: Vec<_> = encoded
        .iter()
        .zip(&suffix_refs)
        .map(|(question, suffixes)| NestedQuestion {
            question_ids: &question.question_ids,
            candidate_suffix_ids: suffixes,
        })
        .collect();
    let logical_input_tokens = root_ids.as_ref().map_or(0, Vec::len)
        + plans
            .iter()
            .map(|plan| {
                plan.question_ids.len()
                    + plan
                        .candidate_suffix_ids
                        .iter()
                        .map(|suffix| suffix.len())
                        .sum::<usize>()
            })
            .sum::<usize>();
    let root_ids = root_ids.ok_or_else(|| Qwen35Error::InvalidInput("no questions".into()))?;
    if let Ok(peak_bytes) = peak_resident_bytes() {
        metrics::gauge!("openkind_native_process_peak_rss_bytes").set(peak_bytes as f64);
    }
    let mut scheduler = inner.scheduler.clone();
    if let Some(process_memory) = scheduler.process_memory {
        scheduler.process_memory = Some(
            process_memory
                .for_concurrent_requests(peak_resident_bytes()?, inner.max_concurrent_requests),
        );
    }
    // Choose explicitly (rather than through `run_with_scheduler`) so the
    // decision itself is telemetered on both the admitted and rejected paths.
    let strategy_request = StrategyRequest::from_plans(root_ids.len(), &plans);
    // A root already held by the cache costs no forward under the shared
    // strategies, which the scheduler must see or it would price work that
    // will not run. The entry can expire before execution. The request is then
    // still correct, only slower than planned.
    let root_cached = inner.root_cache.contains(&root_ids);
    let decision = if root_cached {
        choose_strategy_with_cached_root(&scheduler, &strategy_request)
    } else {
        choose_strategy(&scheduler, &strategy_request)
    };
    if !decision.admitted {
        metrics::counter!(
            "openkind_execution_admission_rejected_total",
            "strategy" => decision.strategy.as_str(),
        )
        .increment(1);
        tracing::warn!(
            strategy = decision.strategy.as_str(),
            forced = decision.forced,
            rationale = %decision.rationale,
            "execution plan rejected by admission ceilings"
        );
        return Err(Qwen35Error::InvalidInput(decision.rationale));
    }
    tracing::debug!(
        rationale = %decision.rationale,
        savings_ratio = decision.estimates.savings_ratio,
        root_cached,
        repeated_tokens = decision.estimates.repeated_tokens,
        shared_tokens = decision.estimates.shared_tokens,
        "scheduler decision detail"
    );
    let controlled = ControlledExecutor::new(&inner.backbone, control);
    let output = if inner.root_cache.config().is_enabled() {
        // Only the exact-root prefill is served from the cache. Every other
        // forward, including the repeated-full strategy's longer prefills,
        // runs unchanged.
        let caching = RootCachingExecutor::new(&controlled, &inner.root_cache, &root_ids);
        run_strategy(&caching, decision.strategy, &root_ids, &plans)?
    } else {
        run_strategy(&controlled, decision.strategy, &root_ids, &plans)?
    };
    if output.batch_forward_mode() != decision.batch_forward_mode {
        return Err(Qwen35Error::InvalidInput(format!(
            "scheduler selected {} batch forwarding but executor reported {}",
            decision.batch_forward_mode.as_str(),
            output.batch_forward_mode().as_str()
        )));
    }
    metrics::counter!(
        "openkind_execution_strategy_total",
        "strategy" => decision.strategy.as_str(),
        "batch_forward_mode" => output.batch_forward_mode().as_str(),
        "forced" => if decision.forced { "true" } else { "false" },
    )
    .increment(1);
    tracing::info!(
        strategy = decision.strategy.as_str(),
        batch_forward_mode = output.batch_forward_mode().as_str(),
        admitted = decision.admitted,
        forced = decision.forced,
        "execution strategy completed"
    );

    let mut answers: HashMap<String, openkind_core::Answer, _> =
        HashMap::with_capacity_and_hasher(encoded.len(), Default::default());
    for (question, features) in encoded.iter().zip(output.question_features()) {
        control.check()?;
        let evaluation = inner.head.evaluate(question.primitive, features)?;
        let answer = answer_from_distribution(question, &evaluation)?;
        answers.insert(question.id.clone(), answer);
    }
    control.check()?;
    Ok(SystemResponse {
        model: request.model,
        answers,
        usage: Usage {
            input_tokens: u32::try_from(logical_input_tokens).unwrap_or(u32::MAX),
            output_tokens: 0,
        },
    })
}

pub(super) fn map_evaluation_error(
    backend: &str,
    error: Qwen35Error,
    timeout_ms: Option<u64>,
) -> EngineError {
    match error {
        Qwen35Error::InvalidInput(message) => EngineError::Unsupported {
            backend: backend.into(),
            message,
        },
        Qwen35Error::ExecutionDeadlineExceeded => EngineError::DeadlineExceeded {
            backend: backend.into(),
            timeout_ms: timeout_ms.unwrap_or_default(),
        },
        Qwen35Error::ExecutionCancelled => EngineError::Backend {
            backend: backend.into(),
            message: "native evaluation cancelled after caller disconnect".into(),
        },
        error => EngineError::Backend {
            backend: backend.into(),
            message: error.to_string(),
        },
    }
}
