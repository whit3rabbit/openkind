//! Offline joint-option readout experiment, never registered by the daemon.
//!
//! The baseline replays the frozen state-first renderer and fitted head with
//! independent full forwards. The joint alternatives share one prompt over all
//! options and score tied single-token letter rows. The catalogue alternative
//! keeps the fitted head and frozen temperature but shows every option's
//! description in each candidate prompt. Position and code layouts separate
//! text order from output-code assignment, and calibration applies a locked
//! positive temperature and none-logit offset to raw joint logits. No
//! temperature, offset, or rejection threshold is fitted on evaluation cases.

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

/// Renderer identity of the catalogue variant, which keeps the fitted head and
/// frozen temperature but adds all option descriptions to every candidate
/// prompt. Distinct from both the frozen `state_first` and joint identities.
pub const CATALOGUE_RENDERER_ID: &str = "catalogue_state_first/v1";

/// One rendered joint prompt: token ids plus each option's output-code slot.
///
/// Slots index the gathered letter rows: `0..MAX_CANDIDATES` render the
/// space-prefixed letters `A..P`, and `MAX_CANDIDATES` renders `Z`.
#[derive(Debug, Clone)]
pub struct JointRender {
    ids: Vec<u32>,
    code_of: Vec<usize>,
}

/// Prepared prompts for one Choice question, in canonical label order.
pub struct PreparedChoice {
    independent_ids: Vec<Vec<u32>>,
    catalogue_ids: Vec<Vec<u32>>,
    forward: JointRender,
    reverse: JointRender,
    text_rotate: JointRender,
    code_rotate: JointRender,
    labels: Vec<String>,
}

impl PreparedChoice {
    /// Canonical real-option labels in sorted order, matching logit order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
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
    #[cfg(feature = "cuda")]
    Cuda(Qwen35Backbone),
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    Mlx(super::mlx::MlxQwen35Backbone),
}

impl ProbeBackbone {
    fn final_feature(&self, ids: &[u32]) -> Result<Vec<f32>, Qwen35Error> {
        match self {
            Self::Cpu(backbone) => Ok(backbone.forward(ids)?.final_token().to_vec()),
            #[cfg(feature = "cuda")]
            Self::Cuda(backbone) => Ok(backbone.forward(ids)?.final_token().to_vec()),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            Self::Mlx(backbone) => Ok(backbone.prefill(ids)?.0.feature().to_vec()),
        }
    }

    fn output_rows(&self, ids: &[u32]) -> Result<Vec<f32>, Qwen35Error> {
        match self {
            Self::Cpu(backbone) => backbone.embedding_rows(ids),
            #[cfg(feature = "cuda")]
            Self::Cuda(backbone) => backbone.embedding_rows(ids),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            Self::Mlx(backbone) => Ok(backbone.embedding_rows(ids)?),
        }
    }
}

/// Raw letter logits of one recorded joint render, ready for post-hoc
/// calibration: real options in sorted label order, none logit last.
pub struct JointLogits {
    /// Raw logits, real options in sorted label order, none last.
    pub values: Vec<f64>,
    /// Prompt positions processed by the single full forward.
    pub input_tokens: usize,
}

/// One locally verified checkpoint shared by every experimental scoring path.
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
            #[cfg(feature = "cuda")]
            Qwen35Backend::Cuda { device_id } => {
                ProbeBackbone::Cuda(Qwen35Backbone::load_with_device(
                    checkpoint_root,
                    candle_core::Device::new_cuda(device_id)?,
                )?)
            }
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
            #[cfg(feature = "cuda")]
            ProbeBackbone::Cuda(backbone) => backbone.identity().arithmetic_id(),
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            ProbeBackbone::Mlx(backbone) => backbone.arithmetic_id(),
        }
    }

    /// Backend used by every path.
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
        self.fitted_scores(&input.independent_ids, input)
    }

    /// Frozen fitted head over independent candidate prompts that also carry a
    /// full option catalogue in the question branch. The head and temperature
    /// are unchanged; only the renderer gains rival descriptions.
    pub fn catalogue(&self, input: &PreparedChoice) -> Result<ScoringResult, Qwen35Error> {
        self.fitted_scores(&input.catalogue_ids, input)
    }

    /// One joint prompt and raw softmax over letter rows at temperature 1.0.
    ///
    /// `reversed` selects the recorded reversal, which changes text order and
    /// letter assignment together and keeps the none option last.
    pub fn joint(
        &self,
        input: &PreparedChoice,
        reversed: bool,
    ) -> Result<ScoringResult, Qwen35Error> {
        let render = if reversed {
            &input.reverse
        } else {
            &input.forward
        };
        self.score_render(input, render)
    }

    /// Joint render with rotated text order and each option's forward code.
    ///
    /// The full displayed list, including the none option, rotates by one
    /// position; letter codes stay bound to their options.
    pub fn joint_text_rotate(&self, input: &PreparedChoice) -> Result<ScoringResult, Qwen35Error> {
        self.score_render(input, &input.text_rotate)
    }

    /// Joint render with forward text order and rotated letter codes.
    ///
    /// Display positions are unchanged; each option, including the none
    /// option, takes the next letter slot in the used set.
    pub fn joint_code_rotate(&self, input: &PreparedChoice) -> Result<ScoringResult, Qwen35Error> {
        self.score_render(input, &input.code_rotate)
    }

    /// Raw letter logits of one recorded joint render: real options in sorted
    /// label order, then the semantic-none logit last.
    pub fn joint_logits(
        &self,
        input: &PreparedChoice,
        reversed: bool,
    ) -> Result<JointLogits, Qwen35Error> {
        let render = if reversed {
            &input.reverse
        } else {
            &input.forward
        };
        let values = self.render_logits(render)?;
        Ok(JointLogits {
            values,
            input_tokens: render.ids.len(),
        })
    }

    /// Softmax over raw joint logits with a positive temperature and an
    /// additive offset on the semantic-none logit, applied before the softmax.
    ///
    /// A temperature alone preserves the winning class; the none offset can
    /// change rejection. Both parameters must be locked on a separate
    /// calibration partition before any gate evaluation.
    pub fn calibrated_probabilities(
        logits: &[f64],
        temperature: f64,
        none_offset: f64,
    ) -> Result<Vec<f64>, Qwen35Error> {
        if logits.len() < 2 {
            return Err(Qwen35Error::InvalidInput(
                "calibration requires at least one real option and the none logit".into(),
            ));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Qwen35Error::InvalidInput(
                "calibration temperature must be finite and greater than zero".into(),
            ));
        }
        if !none_offset.is_finite() {
            return Err(Qwen35Error::InvalidInput(
                "none offset must be finite".into(),
            ));
        }
        let mut scaled: Vec<f64> = logits.iter().map(|logit| logit / temperature).collect();
        let last = scaled.len() - 1;
        scaled[last] += none_offset / temperature;
        stable_softmax(&scaled, 1.0)
    }

    fn fitted_scores(
        &self,
        ids: &[Vec<u32>],
        input: &PreparedChoice,
    ) -> Result<ScoringResult, Qwen35Error> {
        let features = ids
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
                .ok_or_else(|| Qwen35Error::Numerical("fitted head omitted none mass".into()))?,
        );
        Ok(ScoringResult {
            probabilities,
            input_tokens: ids.iter().map(Vec::len).sum(),
            forwards: ids.len(),
        })
    }

    fn score_render(
        &self,
        input: &PreparedChoice,
        render: &JointRender,
    ) -> Result<ScoringResult, Qwen35Error> {
        let logits = self.render_logits(render)?;
        let probabilities = stable_softmax(&logits, 1.0)?;
        let mut labels = input.labels.clone();
        labels.push(SEMANTIC_NONE_OPTION.into());
        Ok(ScoringResult {
            probabilities: labels.into_iter().zip(probabilities).collect(),
            input_tokens: render.ids.len(),
            forwards: 1,
        })
    }

    fn render_logits(&self, render: &JointRender) -> Result<Vec<f64>, Qwen35Error> {
        let hidden = self.backbone.final_feature(&render.ids)?;
        render
            .code_of
            .iter()
            .map(|&slot| {
                let start = slot
                    .checked_mul(FEATURE_WIDTH)
                    .ok_or_else(|| Qwen35Error::InvalidInput("letter row offset overflow".into()))?;
                let row = self.output_rows.get(start..start + FEATURE_WIDTH).ok_or_else(
                    || Qwen35Error::InvalidInput(format!(
                        "letter slot {slot} is outside the {MAX_CANDIDATES}-option and none action rows"
                    )),
                )?;
                project(&hidden, row)
            })
            .collect()
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
    let none = choice.criteria[SEMANTIC_NONE_OPTION]
        .as_deref()
        .expect("validated none");
    let independent = tokenizer.encode_state_first(&state, &instruction, &candidates)?;
    let catalogue = tokenizer.encode_state_first_catalogue(
        &state,
        &instruction,
        &catalogue_block(&labels, &criteria, none),
        &candidates,
    )?;
    let count = labels.len();
    let (display, codes) = forward_layout(count);
    let forward = render_layout(
        tokenizer,
        &state,
        &instruction,
        &labels,
        &criteria,
        none,
        &display,
        &codes,
    )?;
    let (display, codes) = reverse_layout(count);
    let reverse = render_layout(
        tokenizer,
        &state,
        &instruction,
        &labels,
        &criteria,
        none,
        &display,
        &codes,
    )?;
    let (display, codes) = text_rotate_layout(count);
    let text_rotate = render_layout(
        tokenizer,
        &state,
        &instruction,
        &labels,
        &criteria,
        none,
        &display,
        &codes,
    )?;
    let (display, codes) = code_rotate_layout(count);
    let code_rotate = render_layout(
        tokenizer,
        &state,
        &instruction,
        &labels,
        &criteria,
        none,
        &display,
        &codes,
    )?;
    Ok(PreparedChoice {
        independent_ids: independent.full_candidate_ids().to_vec(),
        catalogue_ids: catalogue.full_candidate_ids().to_vec(),
        forward,
        reverse,
        text_rotate,
        code_rotate,
        labels,
    })
}

/// Canonical option catalogue added to every catalogue-variant prompt: all
/// real options in sorted label order plus the semantic-none description, with
/// the same letter formatting as the joint renderer.
fn catalogue_block(labels: &[String], criteria: &[String], none: &str) -> String {
    let mut block = String::from("\nOptions:\n");
    for (index, (label, criterion)) in labels.iter().zip(criteria).enumerate() {
        block.push_str(&format!(
            "{}. {}: {}\n",
            letter_char(index),
            label,
            criterion
        ));
    }
    block.push_str(&format!("{}. {none}\n", letter_char(MAX_CANDIDATES)));
    block
}

fn letter_char(slot: usize) -> char {
    if slot == MAX_CANDIDATES {
        'Z'
    } else {
        (b'A' + slot as u8) as char
    }
}

/// Forward render: options in sorted order, letter codes bound to positions,
/// and the none option last with `Z`. Identical to the recorded joint prompt.
fn forward_layout(count: usize) -> (Vec<usize>, Vec<usize>) {
    let display = (0..=count).collect();
    let code_of = (0..count).chain([MAX_CANDIDATES]).collect();
    (display, code_of)
}

/// Recorded reversal: text order reverses and letter codes follow the new
/// positions, so order and codes change together and `Z` stays last.
fn reverse_layout(count: usize) -> (Vec<usize>, Vec<usize>) {
    let display = (0..count).rev().chain([count]).collect();
    let code_of = (0..count).rev().chain([MAX_CANDIDATES]).collect();
    (display, code_of)
}

/// Text-rotation render: the full displayed list, including the none option,
/// rotates by one position while every option keeps its forward letter code.
fn text_rotate_layout(count: usize) -> (Vec<usize>, Vec<usize>) {
    let display = (1..=count).chain([0]).collect();
    let code_of = (0..count).chain([MAX_CANDIDATES]).collect();
    (display, code_of)
}

/// Code-rotation render: text order is unchanged while letter codes rotate by
/// one over the used set, moving the none option off `Z`.
fn code_rotate_layout(count: usize) -> (Vec<usize>, Vec<usize>) {
    let display = (0..=count).collect();
    let code_of = (0..count)
        .map(|option| (option + 1) % (count + 1))
        .chain([0])
        .collect();
    (display, code_of)
}

#[allow(clippy::too_many_arguments)]
fn render_layout(
    tokenizer: &Qwen35Tokenizer,
    state: &str,
    instruction: &str,
    labels: &[String],
    criteria: &[String],
    none: &str,
    display: &[usize],
    code_of: &[usize],
) -> Result<JointRender, Qwen35Error> {
    let total = labels.len() + 1;
    if display.len() != total || code_of.len() != total {
        return Err(Qwen35Error::InvalidInput(
            "joint layout must cover every real option and none exactly once".into(),
        ));
    }
    let mut seen = [false; MAX_CANDIDATES + 1];
    for slot in code_of {
        if *slot > MAX_CANDIDATES || seen[*slot] {
            return Err(Qwen35Error::InvalidInput(format!(
                "joint layout code slot {slot} is repeated or outside the verified letter rows"
            )));
        }
        seen[*slot] = true;
    }
    if display
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != total
    {
        return Err(Qwen35Error::InvalidInput(
            "joint layout display order repeats an option".into(),
        ));
    }
    let none_letter = letter_char(code_of[labels.len()]);
    let mut ids = tokenizer.encode(&format!("Context:\n{state}\n\n"))?;
    let mut suffix = format!("Question: {instruction}\nChoose the single best option supported by the context. Choose {none_letter} if none is supported.\n");
    for &option in display {
        let letter = letter_char(code_of[option]);
        if option == labels.len() {
            suffix.push_str(&format!("{letter}. {none}\n"));
        } else {
            suffix.push_str(&format!(
                "{letter}. {}: {}\n",
                labels[option], criteria[option]
            ));
        }
    }
    suffix.push_str("Answer:");
    ids.extend(tokenizer.encode(&suffix)?);
    if ids.len() > MAX_SEQUENCE_TOKENS {
        return Err(Qwen35Error::InvalidInput(format!(
            "joint prompt length {} exceeds maximum {MAX_SEQUENCE_TOKENS}; truncation is forbidden",
            ids.len()
        )));
    }
    Ok(JointRender {
        ids,
        code_of: code_of.to_vec(),
    })
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
