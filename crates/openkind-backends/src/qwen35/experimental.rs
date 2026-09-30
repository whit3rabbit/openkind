//! Offline joint-option readout experiment, never registered by the daemon.
//!
//! The baseline replays the frozen state-first renderer and fitted head with
//! independent full forwards. The alternative changes both prompt and readout:
//! all options share one prompt, and tied output rows score single-token letters.
//! Neither temperature nor rejection thresholds are fitted on evaluation cases.

use std::collections::BTreeMap;
use std::path::Path;

use openkind_core::{Question, State};
use serde::Serialize;

use super::engine::{instruction_text, question_candidates, state_text};
use super::head::stable_softmax;
use super::{
    CandidateText, PrimitiveKind, Qwen35Backbone, Qwen35Backend, Qwen35Error, Qwen35Tokenizer,
    ReferenceBundle, ScoreSummaryHead, FEATURE_WIDTH, MAX_CANDIDATES, MAX_SEQUENCE_TOKENS,
    SEMANTIC_NONE_OPTION,
};

/// Renderer identity separate from the frozen state-first model profile.
pub const JOINT_RENDERER_ID: &str = "joint_option_letter/v1";

/// Prepared prompts in canonical label order, with an explicit reversed pass.
pub struct PreparedChoice {
    independent_ids: Vec<Vec<u32>>,
    forward_ids: Vec<u32>,
    reverse_ids: Vec<u32>,
    labels: Vec<String>,
}

/// Complete option probabilities and actual forward-work accounting.
#[derive(Debug, Clone, Serialize)]
pub struct ScoringResult {
    /// All offered real options plus `__none__`, without renormalizing it away.
    pub probabilities: BTreeMap<String, f64>,
    /// Total prompt positions processed by full forwards.
    pub input_tokens: usize,
    /// Number of backbone forwards needed by this scoring method.
    pub forwards: usize,
}

enum ProbeBackbone {
    Cpu(Qwen35Backbone),
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Mlx(super::mlx::MlxQwen35Backbone),
}

impl ProbeBackbone {
    fn final_feature(&self, ids: &[u32]) -> Result<Vec<f32>, Qwen35Error> {
        match self {
            Self::Cpu(backbone) => Ok(backbone.forward(ids)?.final_token().to_vec()),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            Self::Mlx(backbone) => Ok(backbone.prefill(ids)?.0.feature().to_vec()),
        }
    }

    fn output_rows(&self, ids: &[u32]) -> Result<Vec<f32>, Qwen35Error> {
        match self {
            Self::Cpu(backbone) => backbone.embedding_rows(ids),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            Self::Mlx(backbone) => Ok(backbone.embedding_rows(ids)?),
        }
    }
}

/// One locally verified checkpoint shared by both experimental scoring paths.
pub struct Qwen35ScoringProbe {
    tokenizer: Qwen35Tokenizer,
    head: ScoreSummaryHead,
    backbone: ProbeBackbone,
    output_rows: Vec<f32>,
    backend: Qwen35Backend,
}

impl Qwen35ScoringProbe {
    /// Verify local artifacts in place and load FP32 CPU or FP32 ReferenceOps MLX.
    pub fn load(
        bundle_root: &Path,
        checkpoint_root: &Path,
        tokenizer_path: &Path,
        backend: Qwen35Backend,
    ) -> Result<Self, Qwen35Error> {
        let tokenizer = Qwen35Tokenizer::from_file(tokenizer_path)?;
        let head = ReferenceBundle::load(bundle_root)?.head().clone();
        let letters = letter_ids(&tokenizer)?;
        let backbone = match backend {
            Qwen35Backend::NativeCpu => ProbeBackbone::Cpu(Qwen35Backbone::load(checkpoint_root)?),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            Qwen35Backend::MlxFp32 => {
                use super::mlx::{MlxPrecision, MlxQwen35Backbone, MlxRuntime, MlxRuntimeConfig};
                let runtime = std::sync::Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
                ProbeBackbone::Mlx(MlxQwen35Backbone::load(
                    checkpoint_root,
                    runtime,
                    MlxPrecision::Fp32,
                )?)
            }
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            Qwen35Backend::MlxBf16 => {
                return Err(Qwen35Error::InvalidInput(
                    "joint scoring comparison requires FP32 arithmetic".into(),
                ));
            }
        };
        // This checkpoint ties lm_head to embed_tokens, verified by its pinned
        // config. Gathering only action rows avoids a full-vocabulary projection.
        let output_rows = backbone.output_rows(&letters)?;
        Ok(Self {
            tokenizer,
            head,
            backbone,
            output_rows,
            backend,
        })
    }

    /// Arithmetic path under measurement.
    pub fn arithmetic_id(&self) -> &str {
        match &self.backbone {
            ProbeBackbone::Cpu(backbone) => backbone.identity().arithmetic_id(),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            ProbeBackbone::Mlx(backbone) => backbone.arithmetic_id(),
        }
    }

    /// Backend used by both paths.
    pub fn backend(&self) -> Qwen35Backend {
        self.backend
    }

    /// Validate and tokenize every path before evaluating any labeled case.
    pub fn prepare(
        &self,
        state: &State,
        question: &Question,
    ) -> Result<PreparedChoice, Qwen35Error> {
        prepare(&self.tokenizer, state, question)
    }

    /// Frozen fitted head over independent candidate prompts (`repeated_full`).
    pub fn independent(&self, input: &PreparedChoice) -> Result<ScoringResult, Qwen35Error> {
        let features = input
            .independent_ids
            .iter()
            .map(|ids| self.backbone.final_feature(ids))
            .collect::<Result<Vec<_>, _>>()?;
        let evaluation = self.head.evaluate(PrimitiveKind::Choice, &features)?;
        let mut probabilities: BTreeMap<_, _> = input
            .labels
            .iter()
            .cloned()
            .zip(evaluation.candidate_probabilities().iter().copied())
            .collect();
        probabilities.insert(
            SEMANTIC_NONE_OPTION.into(),
            evaluation
                .none_probability()
                .ok_or_else(|| Qwen35Error::Numerical("baseline omitted none mass".into()))?,
        );
        Ok(ScoringResult {
            probabilities,
            input_tokens: input.independent_ids.iter().map(Vec::len).sum(),
            forwards: input.independent_ids.len(),
        })
    }

    /// One joint prompt and raw softmax over letters A..P plus semantic-none Z.
    pub fn joint(
        &self,
        input: &PreparedChoice,
        reversed: bool,
    ) -> Result<ScoringResult, Qwen35Error> {
        let ids = if reversed {
            &input.reverse_ids
        } else {
            &input.forward_ids
        };
        let hidden = self.backbone.final_feature(ids)?;
        let mut logits = self
            .output_rows
            .as_chunks::<FEATURE_WIDTH>()
            .0
            .iter()
            .take(input.labels.len())
            .map(|row| project(&hidden, row))
            .collect::<Result<Vec<_>, _>>()?;
        logits.push(project(
            &hidden,
            &self.output_rows[MAX_CANDIDATES * FEATURE_WIDTH..],
        )?);
        let probabilities = stable_softmax(&logits, 1.0)?;
        let mut labels = input.labels.clone();
        if reversed {
            labels.reverse();
        }
        labels.push(SEMANTIC_NONE_OPTION.into());
        Ok(ScoringResult {
            probabilities: labels.into_iter().zip(probabilities).collect(),
            input_tokens: ids.len(),
            forwards: 1,
        })
    }
}

/// Average remapped distributions, preserving semantic-none mass.
pub fn average_orders(
    first: &ScoringResult,
    second: &ScoringResult,
) -> Result<ScoringResult, Qwen35Error> {
    if first.probabilities.keys().ne(second.probabilities.keys()) {
        return Err(Qwen35Error::InvalidInput(
            "option sets differ between orders".into(),
        ));
    }
    Ok(ScoringResult {
        probabilities: first
            .probabilities
            .iter()
            .map(|(label, probability)| {
                (
                    label.clone(),
                    (probability + second.probabilities[label]) / 2.0,
                )
            })
            .collect(),
        input_tokens: first.input_tokens + second.input_tokens,
        forwards: first.forwards + second.forwards,
    })
}

fn prepare(
    tokenizer: &Qwen35Tokenizer,
    state: &State,
    question: &Question,
) -> Result<PreparedChoice, Qwen35Error> {
    let Question::Choice(choice) = question else {
        return Err(Qwen35Error::InvalidInput(
            "joint probe accepts Choice questions only".into(),
        ));
    };
    let (_, labels, criteria) = question_candidates(question)?;
    let candidates: Vec<_> = labels
        .iter()
        .zip(&criteria)
        .map(|(label, criterion)| CandidateText::new(label, criterion))
        .collect();
    let state = state_text(state);
    let instruction = instruction_text(question)?;
    let independent = tokenizer.encode_state_first(&state, &instruction, &candidates)?;
    let none = choice.criteria[SEMANTIC_NONE_OPTION]
        .as_deref()
        .expect("validated none");
    let forward_ids = render_joint(tokenizer, &state, &instruction, &candidates, none)?;
    let reversed: Vec<_> = candidates.into_iter().rev().collect();
    let reverse_ids = render_joint(tokenizer, &state, &instruction, &reversed, none)?;
    Ok(PreparedChoice {
        independent_ids: independent.full_candidate_ids().to_vec(),
        forward_ids,
        reverse_ids,
        labels,
    })
}

fn render_joint(
    tokenizer: &Qwen35Tokenizer,
    state: &str,
    instruction: &str,
    candidates: &[CandidateText<'_>],
    none: &str,
) -> Result<Vec<u32>, Qwen35Error> {
    let mut ids = tokenizer.encode(&format!("Context:\n{state}\n\n"))?;
    let mut suffix = format!("Question: {instruction}\nChoose the single best option supported by the context. Choose Z if none is supported.\n");
    for (index, candidate) in candidates.iter().enumerate() {
        suffix.push_str(&format!(
            "{}. {}: {}\n",
            (b'A' + index as u8) as char,
            candidate.label,
            candidate.criteria
        ));
    }
    suffix.push_str(&format!("Z. {none}\nAnswer:"));
    ids.extend(tokenizer.encode(&suffix)?);
    if ids.len() > MAX_SEQUENCE_TOKENS {
        return Err(Qwen35Error::InvalidInput(format!(
            "joint prompt length {} exceeds maximum {MAX_SEQUENCE_TOKENS}; truncation is forbidden",
            ids.len()
        )));
    }
    Ok(ids)
}

fn letter_ids(tokenizer: &Qwen35Tokenizer) -> Result<Vec<u32>, Qwen35Error> {
    let ids = (0..MAX_CANDIDATES)
        .map(|index| b'A' + index as u8)
        .chain(*b"Z")
        .map(|letter| {
            let encoded = tokenizer.encode(&format!(" {}", letter as char))?;
            if encoded.len() != 1 {
                return Err(Qwen35Error::Tokenizer(format!(
                    "space-prefixed letter {} must be one token",
                    letter as char
                )));
            }
            Ok(encoded[0])
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.iter().collect::<std::collections::BTreeSet<_>>().len() != ids.len() {
        return Err(Qwen35Error::Tokenizer(
            "action token IDs must be distinct".into(),
        ));
    }
    Ok(ids)
}

fn project(hidden: &[f32], row: &[f32]) -> Result<f64, Qwen35Error> {
    if hidden.len() != FEATURE_WIDTH || row.len() != FEATURE_WIDTH {
        return Err(Qwen35Error::InvalidInput(
            "selected-output projection width mismatch".into(),
        ));
    }
    let logit: f64 = hidden
        .iter()
        .zip(row)
        .map(|(feature, weight)| f64::from(*feature) * f64::from(*weight))
        .sum();
    if !logit.is_finite() {
        return Err(Qwen35Error::Numerical("letter logit is not finite".into()));
    }
    Ok(logit)
}

#[cfg(test)]
mod tests;
