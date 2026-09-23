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
use crate::qwen35::backbone::NestedQuestion;
use crate::qwen35::{choose_strategy, run_strategy, CandidateText, Qwen35Error, StrategyRequest};

pub(super) fn evaluate_request(
    inner: &EngineInner,
    request: SystemRequest,
) -> Result<SystemResponse, Qwen35Error> {
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
        let segments = inner
            .tokenizer
            .encode_state_first(&state, &instruction, &candidates)?;
        if let Some(expected) = &root_ids {
            if expected != segments.root_ids() {
                return Err(Qwen35Error::InvalidInput(
                    "state-first root tokenization changed between questions".into(),
                ));
            }
        } else {
            root_ids = Some(segments.root_ids().to_vec());
        }
        encoded.push(EncodedQuestion {
            id,
            primitive,
            labels,
            criteria,
            question_ids: segments.question_ids().to_vec(),
            candidate_suffix_ids: segments.candidate_suffix_ids().to_vec(),
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
    let decision = choose_strategy(&scheduler, &strategy_request);
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
    metrics::counter!(
        "openkind_execution_strategy_total",
        "strategy" => decision.strategy.as_str(),
        "batch_forward_mode" => decision.batch_forward_mode.as_str(),
        "forced" => if decision.forced { "true" } else { "false" },
    )
    .increment(1);
    tracing::info!(
        strategy = decision.strategy.as_str(),
        batch_forward_mode = decision.batch_forward_mode.as_str(),
        admitted = decision.admitted,
        forced = decision.forced,
        "execution strategy selected"
    );
    tracing::debug!(
        rationale = %decision.rationale,
        savings_ratio = decision.estimates.savings_ratio,
        repeated_tokens = decision.estimates.repeated_tokens,
        shared_tokens = decision.estimates.shared_tokens,
        "scheduler decision detail"
    );
    let output = run_strategy(&inner.backbone, decision.strategy, &root_ids, &plans)?;

    let mut answers: HashMap<String, openkind_core::Answer, _> =
        HashMap::with_capacity_and_hasher(encoded.len(), Default::default());
    for (question, features) in encoded.iter().zip(output.question_features()) {
        let evaluation = inner.head.evaluate(question.primitive, features)?;
        let answer = answer_from_distribution(question, &evaluation)?;
        answers.insert(question.id.clone(), answer);
    }
    Ok(SystemResponse {
        model: request.model,
        answers,
        usage: Usage {
            input_tokens: u32::try_from(logical_input_tokens).unwrap_or(u32::MAX),
            output_tokens: 0,
        },
    })
}

pub(super) fn map_evaluation_error(backend: &str, error: Qwen35Error) -> EngineError {
    match error {
        Qwen35Error::InvalidInput(message) => EngineError::Unsupported {
            backend: backend.into(),
            message,
        },
        error => EngineError::Backend {
            backend: backend.into(),
            message: error.to_string(),
        },
    }
}
