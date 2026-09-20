//! Direct Jev wire adapter for the pinned native Qwen3.5 profile.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use opendecision_core::{
    Answer, ChoiceAnswer, ModelInfo, NoulAnswer, Question, ScoreAnswer, State, SystemRequest,
    SystemResponse, Usage,
};
use opendecision_engine::{DecisionEngine, EngineError, EngineResult};
use opendecision_runtime::peak_resident_bytes;
use tokio::sync::Semaphore;

use super::backbone::NestedQuestion;
use super::{
    run_with_scheduler, CandidateText, PrimitiveKind, Qwen35Backbone, Qwen35Error, Qwen35Tokenizer,
    ReferenceBundle, SchedulerConfig, ScoreSummaryHead,
};

/// Reserved Choice criteria key that explicitly exposes the model's semantic-none mass.
///
/// Native Choice requests must include this key with a non-empty description. The adapter
/// never appends an unrequested option and never discards or renormalizes none probability.
pub const SEMANTIC_NONE_OPTION: &str = "__none__";

/// Filesystem and admission settings for the direct native engine.
#[derive(Debug, Clone)]
pub struct Qwen35EngineConfig {
    /// Selected-profile bundle directory containing the fitted head.
    pub bundle_root: PathBuf,
    /// Pinned Qwen checkpoint directory containing both safetensors shards.
    pub checkpoint_root: PathBuf,
    /// Digest-locked exported tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Adaptive execution policy and memory admission settings.
    pub scheduler: SchedulerConfig,
    /// Maximum model evaluations executing concurrently.
    pub max_concurrent_requests: usize,
    /// Maximum additional requests allowed to wait for execution.
    pub max_queued_requests: usize,
    /// Suggested backoff returned when admission is full.
    pub retry_after_ms: u64,
}

struct EngineInner {
    tokenizer: Qwen35Tokenizer,
    backbone: Qwen35Backbone,
    head: ScoreSummaryHead,
    scheduler: SchedulerConfig,
    max_concurrent_requests: usize,
}

/// Native CPU implementation registered directly behind [`DecisionEngine`].
pub struct Qwen35DecisionEngine {
    inner: Arc<EngineInner>,
    execution_slots: Arc<Semaphore>,
    admission_slots: Arc<Semaphore>,
    retry_after_ms: u64,
}

impl Qwen35DecisionEngine {
    /// Load every pinned artifact offline and construct the bounded native engine.
    pub fn load(config: Qwen35EngineConfig) -> Result<Self, Qwen35Error> {
        let tokenizer = Qwen35Tokenizer::from_file(&config.tokenizer_path)?;
        let backbone = Qwen35Backbone::load(&config.checkpoint_root)?;
        let bundle = ReferenceBundle::load(&config.bundle_root)?;
        let concurrent = config.max_concurrent_requests.max(1);
        let admitted = concurrent.saturating_add(config.max_queued_requests);
        Ok(Self {
            inner: Arc::new(EngineInner {
                tokenizer,
                backbone,
                head: bundle.head().clone(),
                scheduler: config.scheduler,
                max_concurrent_requests: concurrent,
            }),
            execution_slots: Arc::new(Semaphore::new(concurrent)),
            admission_slots: Arc::new(Semaphore::new(admitted)),
            retry_after_ms: config.retry_after_ms,
        })
    }
}

#[async_trait]
impl DecisionEngine for Qwen35DecisionEngine {
    fn backend_id(&self) -> &str {
        "qwen35-native-cpu"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description:
                "Pinned Qwen3.5-4B native CPU reference engine with explicit semantic none.".into(),
            release_date: "2026-09-20".into(),
        }
    }

    async fn evaluate(&self, request: SystemRequest) -> EngineResult<SystemResponse> {
        let admission = self
            .admission_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| EngineError::Overloaded {
                backend: self.backend_id().to_owned(),
                retry_after_ms: self.retry_after_ms,
            })?;
        let execution = self
            .execution_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| EngineError::Backend {
                backend: self.backend_id().to_owned(),
                message: "execution admission closed during shutdown".into(),
            })?;
        let inner = self.inner.clone();
        let backend = self.backend_id().to_owned();
        let result = run_blocking_with_permits(execution, admission, move || {
            evaluate_request(&inner, request)
        })
        .await
        .map_err(|join| EngineError::Backend {
            backend: backend.clone(),
            message: format!("native evaluation task failed: {join}"),
        })?;
        result.map_err(|error| map_evaluation_error(&backend, error))
    }
}

async fn run_blocking_with_permits<T, F>(
    execution: tokio::sync::OwnedSemaphorePermit,
    admission: tokio::sync::OwnedSemaphorePermit,
    work: F,
) -> Result<T, tokio::task::JoinError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        // The blocking task owns both permits. Dropping a cancelled HTTP/gRPC
        // future therefore cannot admit replacement work while native compute
        // is still running in the blocking pool.
        let _execution = execution;
        let _admission = admission;
        work()
    })
    .await
}

struct EncodedQuestion {
    id: String,
    primitive: PrimitiveKind,
    labels: Vec<String>,
    criteria: Vec<String>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<Vec<u32>>,
}

fn evaluate_request(
    inner: &EngineInner,
    request: SystemRequest,
) -> Result<SystemResponse, Qwen35Error> {
    let state = state_text(&request.state)?;
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
    let (_, output) = run_with_scheduler(&inner.backbone, &scheduler, &root_ids, &plans)?;

    let mut answers = HashMap::with_capacity(encoded.len());
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

fn question_candidates(
    question: &Question,
) -> Result<(PrimitiveKind, Vec<String>, Vec<String>), Qwen35Error> {
    match question {
        Question::Noul(question) => {
            let (false_criterion, true_criterion) = question
                .criteria
                .as_ref()
                .map(|criteria| (criteria.r#false.clone(), criteria.r#true.clone()))
                .unwrap_or_else(|| {
                    (
                        "The answer is false or no.".into(),
                        "The answer is true or yes.".into(),
                    )
                });
            Ok((
                PrimitiveKind::Noul,
                vec!["false".into(), "true".into()],
                vec![false_criterion, true_criterion],
            ))
        }
        Question::Choice(question) => {
            let none = question.criteria.get(SEMANTIC_NONE_OPTION).ok_or_else(|| {
                Qwen35Error::InvalidInput(format!(
                    "native Choice requires explicit `{SEMANTIC_NONE_OPTION}` criteria so semantic-none mass remains on wire"
                ))
            })?;
            if none
                .as_deref()
                .is_none_or(|criterion| criterion.trim().is_empty())
            {
                return Err(Qwen35Error::InvalidInput(format!(
                    "`{SEMANTIC_NONE_OPTION}` must have a non-empty description"
                )));
            }
            let mut labels: Vec<_> = question
                .criteria
                .keys()
                .filter(|label| label.as_str() != SEMANTIC_NONE_OPTION)
                .cloned()
                .collect();
            labels.sort();
            let criteria = labels
                .iter()
                .map(|label| {
                    question.criteria[label]
                        .clone()
                        .unwrap_or_else(|| label.clone())
                })
                .collect();
            Ok((PrimitiveKind::Choice, labels, criteria))
        }
        Question::Score(question) => Ok((
            PrimitiveKind::Score,
            (0..question.criteria.len())
                .map(|index| index.to_string())
                .collect(),
            question.criteria.clone(),
        )),
    }
}

fn answer_from_distribution(
    question: &EncodedQuestion,
    evaluation: &super::HeadEvaluation,
) -> Result<Answer, Qwen35Error> {
    match question.primitive {
        PrimitiveKind::Noul => Ok(Answer::Noul(NoulAnswer {
            noul: evaluation.candidate_probabilities()[1],
        })),
        PrimitiveKind::Choice => {
            let mut probabilities: HashMap<String, f64> = question
                .labels
                .iter()
                .cloned()
                .zip(evaluation.candidate_probabilities().iter().copied())
                .collect();
            let none = evaluation.none_probability().ok_or_else(|| {
                Qwen35Error::Numerical("Choice evaluation omitted semantic-none mass".into())
            })?;
            probabilities.insert(SEMANTIC_NONE_OPTION.into(), none);
            let choice = evaluation
                .selected_candidate_index()
                .map(|index| question.labels[index].clone())
                .unwrap_or_else(|| SEMANTIC_NONE_OPTION.into());
            Ok(Answer::Choice(ChoiceAnswer {
                choice,
                confidence: distribution_confidence(&evaluation.full_probabilities()),
                probabilities,
            }))
        }
        PrimitiveKind::Score => {
            let probabilities: HashMap<String, f64> = question
                .labels
                .iter()
                .cloned()
                .zip(evaluation.candidate_probabilities().iter().copied())
                .collect();
            let score = evaluation
                .candidate_probabilities()
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            let legend = question
                .labels
                .iter()
                .cloned()
                .zip(question.criteria.iter().cloned())
                .collect();
            Ok(Answer::Score(ScoreAnswer {
                score,
                legend,
                confidence: distribution_confidence(evaluation.candidate_probabilities()),
                probabilities,
            }))
        }
    }
}

fn distribution_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() <= 1 {
        return 1.0;
    }
    let entropy = probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

fn state_text(state: &State) -> Result<String, Qwen35Error> {
    match state {
        State::Text(text) => Ok(text.clone()),
        State::Object(value) => serde_json::to_string(value),
        State::Array(value) => serde_json::to_string(value),
    }
    .map_err(|error| Qwen35Error::Tokenizer(format!("state serialization failed: {error}")))
}

fn instruction_text(question: &Question) -> Result<String, Qwen35Error> {
    let instructions = match question {
        Question::Noul(question) => &question.instructions,
        Question::Choice(question) => &question.instructions,
        Question::Score(question) => &question.instructions,
    };
    match instructions {
        serde_json::Value::String(text) => Ok(text.clone()),
        value => serde_json::to_string(value).map_err(|error| {
            Qwen35Error::Tokenizer(format!("instruction serialization failed: {error}"))
        }),
    }
}

fn map_evaluation_error(backend: &str, error: Qwen35Error) -> EngineError {
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

#[cfg(test)]
mod tests {
    use super::*;
    use opendecision_core::ChoiceQuestion;
    use std::time::Duration;

    #[test]
    fn choice_requires_explicit_semantic_none_and_never_sends_it_to_the_model() {
        let missing = Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: [
                ("a".into(), Some("A".into())),
                ("b".into(), Some("B".into())),
            ]
            .into_iter()
            .collect(),
        });
        assert!(matches!(
            question_candidates(&missing),
            Err(Qwen35Error::InvalidInput(message)) if message.contains(SEMANTIC_NONE_OPTION)
        ));

        let explicit = Question::Choice(ChoiceQuestion {
            instructions: serde_json::json!("Pick"),
            criteria: [
                ("b".into(), Some("B".into())),
                (SEMANTIC_NONE_OPTION.into(), Some("Neither applies".into())),
                ("a".into(), None),
            ]
            .into_iter()
            .collect(),
        });
        let (_, labels, criteria) = question_candidates(&explicit).expect("map choice");
        assert_eq!(labels, ["a", "b"]);
        assert_eq!(criteria, ["a", "B"]);
    }

    #[test]
    fn confidence_is_entropy_based_not_top_probability() {
        let uniform = distribution_confidence(&[0.5, 0.5]);
        let peaked = distribution_confidence(&[0.99, 0.01]);
        assert!(uniform.abs() <= f64::EPSILON);
        assert!(peaked > 0.9);
        assert_ne!(peaked, 0.99);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_keeps_permits_until_blocking_work_finishes() {
        let execution = Arc::new(Semaphore::new(1));
        let admission = Arc::new(Semaphore::new(1));
        let execution_permit = execution
            .clone()
            .acquire_owned()
            .await
            .expect("execution permit");
        let admission_permit = admission
            .clone()
            .acquire_owned()
            .await
            .expect("admission permit");
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let request = tokio::spawn(run_blocking_with_permits(
            execution_permit,
            admission_permit,
            move || {
                let _ = started_tx.send(());
                let _ = release_rx.blocking_recv();
            },
        ));

        started_rx.await.expect("blocking work started");
        request.abort();
        assert_eq!(execution.available_permits(), 0);
        assert_eq!(admission.available_permits(), 0);

        release_tx.send(()).expect("release blocking work");
        tokio::time::timeout(Duration::from_secs(2), async {
            while execution.available_permits() == 0 || admission.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permits released after native work completed");
    }
}
