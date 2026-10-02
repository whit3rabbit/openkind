//! Family: `gemma4-decision`.
//!
//! The Gemma 4 text backbone (the E-series hybrid attention decoder with
//! per-layer embeddings) fine-tuned for typed decisions, executing a pinned
//! GGUF checkpoint as the FP32 candle reference graph. The readout is the
//! Winnow letter protocol: options are rendered as letter-labelled lines and
//! the answer is the temperature-calibrated distribution over verified
//! single-token letter logits at the answer slot. No output token is ever
//! sampled.
//!
//! One pinned profile:
//!
//! | Profile | Checkpoint | Reference runtime | Calibration |
//! |---|---|---|---|
//! | `winnow-e4b` | `EldanRing/Winnow-E4B` Q8_0 GGUF | winnow-inference (llama.cpp, MIT) | Q8 letter softmax 1.2574172017327816 |
//!
//! Profile semantics:
//! - Probability space: [`ConditionalOnOfferedOptions`] — the distribution
//!   over offered options sums to one and no semantic-none mass exists. An
//!   offered `__none__` key is scored as an ordinary option.
//! - Continuation state: none — every question is an independent
//!   full-sequence forward and nothing is retained across questions.
//! - Text generation: none.
//!
//! Reference contract: the `winnow` native decision engine shipped by the
//! checkpoint authors (`native/protocol.h` and `native/engine.h` in
//! `github.com/EldanRing/winnow-inference`, MIT), whose system prompt,
//! state serialization, option rendering, boundary markers, and verified
//! letter-token discovery this family reproduces. Two declared determinism
//! differences, as in the other surveyed families: structured state
//! serializes in canonical byte-lexicographic key order (the repository's
//! `state_first` invariant), and the reference's llama.cpp KV reuse across
//! questions is replaced by fresh full-sequence forwards, which is
//! numerically identical to the reference's cold-prefix path.
//!
//! Architecture reference: this backbone ports the Gemma 4 text decoder from
//! two agreeing implementations — mistral.rs
//! (`mistralrs-core/src/vision_models/gemma4/`, day-0 Gemma 4 support) and
//! llama.cpp (`src/models/gemma4.cpp` plus the `Gemma4Model` converter with
//! `norm_shift = 0`). Where the in-tree candle 0.8.0 release line and
//! candle `main`'s `gemma4` example disagree with those references — the
//! RMSNorm `+1` weight shift and the `1/sqrt(head_dim)` attention scale —
//! the references win: the converter stores unshifted norms and both
//! references set the attention softmax scale to `1.0`.

mod backbone;
mod engine;
mod gguf;
mod renderer;

use std::path::PathBuf;
use std::time::Duration;

use openkind_engine::ProbabilitySpace;
use thiserror::Error;

pub use self::engine::Gemma4DecisionEngine;
use crate::families::support::FamilyLimits;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "gemma4-decision";
/// Pinned backbone repository (GGUF export).
pub const BACKBONE_ID: &str = "EldanRing/Winnow-E4B";
/// Pinned immutable backbone revision.
pub const BACKBONE_REVISION: &str = "1b257e8fa80b270a62338362a8b35e37f7890273";
/// Pinned GGUF quantization variant.
pub const GGUF_VARIANT: &str = "Winnow-E4B-Q8_0.gguf";
/// Loader/profile name of the pinned Winnow E4B profile.
pub const WINNOW_E4B: &str = "winnow-e4b";
/// Stable derived profile ID for the pinned profile.
pub const PROFILE_ID: &str = "656ac636ce450cf79c7d";
/// SHA-256 of the pinned GGUF checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2";
/// SHA-256 of the pinned `tokenizer.json` (the Gemma 4 E4B tokenizer
/// exported from `mistralrs-community/gemma-4-E4B-it-UQFF` at revision
/// `a1789f4cb2d036e0e15c5cdb10317536e4a2d847`; the GGUF embeds the same
/// vocabulary, cross-checked at load time).
pub const TOKENIZER_JSON_SHA256: &str =
    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
/// Arithmetic/device identity of the family execution path: candle's
/// quantized q8_0 `QMatMul` kernels on CPU, the binding the
/// `decoder-logit-llm` family established for GGUF checkpoints.
pub const EXECUTION_ARITHMETIC_ID: &str = "candle-cpu-q8_0-gemma4";
/// Declared probability space of the profile.
pub const DECLARED_PROBABILITY_SPACE: ProbabilitySpace =
    ProbabilitySpace::ConditionalOnOfferedOptions;

/// Frozen maximum rendered prompt length. Longer requests fail closed;
/// truncation is forbidden.
pub const MAX_SEQUENCE_TOKENS: usize = 8_192;

/// Provisional application-policy threshold recorded with the profile.
///
/// This is not a model-quality result: no M2 reviewed-decision gate has run
/// for this profile.
pub const POLICY_THRESHOLD: f64 = 0.60;

/// Calibration temperature for the pinned Q8_0 profile.
///
/// The checkpoint authors fitted `1.2574172017327816` on 778 calibration
/// questions for the Q8_0 artifact (the BF16 artifact carries a different
/// fit and must not be reused). Loaders reject any other temperature.
pub const CALIBRATION_TEMPERATURE: f64 = 1.257_417_201_732_781_6;

/// GGUF metadata values the loader enforces before trusting the checkpoint.
///
/// These pin the architecture the backbone implements: the E4B hybrid
/// decoder with per-layer embeddings and shared-KV layers.
pub(crate) fn pinned_metadata() -> Vec<(&'static str, &'static str)> {
    vec![
        ("general.architecture", "gemma4"),
        ("gemma4.block_count", "42"),
        ("gemma4.embedding_length", "2560"),
        ("gemma4.embedding_length_per_layer_input", "256"),
        ("gemma4.feed_forward_length", "10240"),
        ("gemma4.attention.head_count", "8"),
        ("gemma4.attention.head_count_kv", "2"),
        ("gemma4.attention.key_length", "512"),
        ("gemma4.attention.key_length_swa", "256"),
        ("gemma4.attention.sliding_window", "512"),
        ("gemma4.attention.shared_kv_layers", "18"),
        ("gemma4.final_logit_softcapping", "30"),
        ("gemma4.rope.freq_base", "1000000"),
        ("gemma4.rope.freq_base_swa", "10000"),
    ]
}

/// Errors raised while loading or evaluating the pinned Gemma 4 profile.
#[derive(Debug, Error)]
pub enum Gemma4Error {
    /// Shared family loader/evaluator failure.
    #[error(transparent)]
    Family(#[from] crate::families::support::FamilyError),
}

/// Filesystem configuration for the pinned gemma4-decision profile.
#[derive(Debug, Clone)]
pub struct Gemma4EngineConfig {
    /// Model root containing the pinned `Winnow-E4B-Q8_0.gguf` and
    /// `tokenizer.json` at [`BACKBONE_REVISION`]. Artifacts are verified in
    /// place; nothing is copied or downloaded.
    pub model_root: PathBuf,
    /// Admission and deadline settings.
    pub limits: FamilyLimits,
}

impl Gemma4EngineConfig {
    /// Fill admission defaults around the given model root.
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            limits: FamilyLimits {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                retry_after_ms: 1_000,
                evaluation_timeout: Some(Duration::from_secs(900)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_id_matches_the_derivation_rule() {
        assert_eq!(
            PROFILE_ID,
            crate::families::support::derive_profile_id(
                FAMILY_SLUG,
                BACKBONE_ID,
                BACKBONE_REVISION
            )
        );
    }

    #[test]
    fn declared_probability_space_is_conditional() {
        assert_eq!(
            DECLARED_PROBABILITY_SPACE.as_str(),
            ProbabilitySpace::ConditionalOnOfferedOptions.as_str()
        );
    }

    #[test]
    fn calibration_temperature_is_the_published_q8_fit() {
        assert_eq!(CALIBRATION_TEMPERATURE, 1.257_417_201_732_781_6);
    }

    #[test]
    fn pinned_metadata_pin_the_e4b_geometry() {
        let metadata = pinned_metadata();
        assert!(metadata.contains(&("gemma4.block_count", "42")));
        assert!(metadata.contains(&("gemma4.attention.shared_kv_layers", "18")));
        assert_eq!(metadata.len(), 14);
    }

    /// Operator-gated smoke run against the locally downloaded checkpoint:
    /// `OPENKIND_GEMMA4_MODEL_ROOT=<dir> cargo test -p openkind-backends --
    /// --nocapture gemma4::tests::pinned_checkpoint_smoke --ignored`. Skipped
    /// (passing trivially) without the env var so CI stays offline. The
    /// cases assert the argmax of obvious ground-truth questions across all
    /// three wire primitives; they are sanity signals for the port, not a
    /// model-quality gate.
    /// Operator-gated numeric crosscheck of the execution layer: the
    /// file-backed q8_0 row reader and candle's `QMatMul` kernels versus
    /// candle's own full dequantization.
    #[test]
    #[ignore]
    fn quantization_crosscheck() {
        use candle_core::Module as _;
        let Ok(root) = std::env::var("OPENKIND_GEMMA4_MODEL_ROOT") else {
            return;
        };
        let artifacts = gguf::VerifiedArtifacts::verify(std::path::Path::new(&root))
            .expect("verify pinned artifacts");
        let device = candle_core::Device::Cpu;
        let cfg = backbone::Gemma4TextConfig::winnow_e4b();
        let checkpoint = gguf::load_checkpoint(&artifacts, &cfg, &device).expect("load gguf");

        // Row reader vs full dequantize, embedding table.
        let ids = [2u32, 105, 236776, 236799, 12345];
        let via_rows = checkpoint.token_embd_rows.rows(&ids).expect("rows");
        let dequantized = checkpoint
            .q_tensor("embed_tokens.weight")
            .expect("emb tensor")
            .dequantize(&device)
            .expect("dequantize");
        let reference = dequantized
            .index_select(
                &candle_core::Tensor::from_vec(ids.to_vec(), (ids.len(),), &device).unwrap(),
                0,
            )
            .expect("select rows");
        let a = via_rows.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!("row reader vs dequantize: max_diff={max_diff:e}");
        assert_eq!(
            max_diff, 0.0,
            "row reader must match candle dequantization exactly"
        );

        // QMatMul vs dequantized matmul, first attention projection.
        let xs =
            candle_core::Tensor::randn(1f32, 1.0, (1, 7, cfg.hidden_size), &device).expect("input");
        let via_qmatmul = candle_core::quantized::QMatMul::from_arc(
            checkpoint
                .q_tensor("layers.0.self_attn.q_proj.weight")
                .expect("q tensor"),
        )
        .expect("qmatmul")
        .forward(&xs)
        .expect("qmatmul forward");
        let weight = checkpoint
            .q_tensor("layers.0.self_attn.q_proj.weight")
            .unwrap()
            .dequantize(&device)
            .expect("dequantize");
        let via_matmul = xs
            .apply(&candle_nn::Linear::new(weight, None))
            .expect("linear");
        let a = via_qmatmul.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = via_matmul.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let max_abs = a.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        println!("qmatmul vs matmul: max_diff={max_diff:e} (max_abs={max_abs:e})");
        // candle's quantized kernels dequantize q8_0 blocks to f16, so a
        // ~0.5% relative deviation from the F32 matmul is the documented
        // kernel behavior (and the reason the family's attention
        // projections run as F32 linears), not a loader bug.
        assert!(
            max_diff < 5e-2 + 5e-3 * max_abs,
            "QMatMul diverges beyond the documented f16 kernel noise"
        );
    }

    #[test]
    #[ignore]
    fn pinned_checkpoint_smoke() {
        let Ok(root) = std::env::var("OPENKIND_GEMMA4_MODEL_ROOT") else {
            return;
        };
        let artifacts = gguf::VerifiedArtifacts::verify(std::path::Path::new(&root))
            .expect("verify pinned artifacts");
        let gguf_letters = gguf::gguf_letter_token_ids(&artifacts, renderer::LETTER_MAX_OPTIONS)
            .expect("gguf letter ids");
        let renderer = renderer::WinnowRenderer::load(
            &artifacts.tokenizer,
            MAX_SEQUENCE_TOKENS,
            Some(&gguf_letters),
        )
        .expect("load renderer");
        println!("verified letter tokens: {}", gguf_letters.len());
        let device = candle_core::Device::Cpu;
        let cfg = backbone::Gemma4TextConfig::winnow_e4b();
        let checkpoint = gguf::load_checkpoint(&artifacts, &cfg, &device).expect("load gguf");
        let model = backbone::Gemma4TextModel::new(&cfg, &checkpoint).expect("build model");

        let state = serde_json::json!({
            "record": {
                "order_id": "A-2231",
                "destination": "Reykjavik, Iceland",
                "carrier": "Northern Air",
                "departures": ["09:30", "14:05"],
                "cancelled_departures": ["09:30"],
                "active_departure": "14:05",
            }
        });
        #[allow(clippy::type_complexity)]
        let cases: &[(
            &str,
            serde_json::Value,
            Vec<String>,
            Vec<String>,
            bool,
            usize,
        )] = &[
            (
                "noul true",
                serde_json::json!("Does the record still list an active departure?"),
                vec!["false".into(), "true".into()],
                vec![
                    "The stated proposition is false according to the record.".into(),
                    "The stated proposition is true according to the record.".into(),
                ],
                false,
                1,
            ),
            (
                "noul false",
                serde_json::json!("Was the 09:30 departure cancelled?"),
                vec!["false".into(), "true".into()],
                vec![
                    "The stated proposition is false according to the record.".into(),
                    "The stated proposition is true according to the record.".into(),
                ],
                false,
                1,
            ),
            (
                "choice destination",
                serde_json::json!("Which city is the order shipping to?"),
                vec!["lisbon".into(), "reykjavik".into()],
                vec![
                    "lisbon: The order ships to Lisbon, Portugal.".into(),
                    "reykjavik: The order ships to Reykjavik, Iceland.".into(),
                ],
                false,
                1,
            ),
            (
                "score cancelled count",
                serde_json::json!("How many departures were cancelled?"),
                vec!["0".into(), "1".into(), "2".into()],
                vec![
                    "no departure was cancelled".into(),
                    "exactly one departure was cancelled".into(),
                    "two or more departures were cancelled".into(),
                ],
                true,
                1,
            ),
        ];
        let mut failures = Vec::new();
        // Backbone health probe: plain chat continuation, no decision
        // protocol. A healthy Gemma-4 chat model answers the cloze with
        // the city name.
        {
            let chat = "<|turn>user\nThe capital of France is<turn|>\n<|turn>model\n";
            let ids = renderer.encode_debug(chat);
            let logits = model.forward(&ids).expect("chat forward");
            let flat = logits.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
            let mut indexed: Vec<(u32, f32)> = flat
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (i as u32, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_text: Vec<String> = indexed[..8]
                .iter()
                .map(|(id, _)| renderer.decode_debug(&[*id]))
                .collect();
            println!("chat probe top-8 decoded: {top_text:?}");
        }
        if std::env::var_os("OPENKIND_GEMMA4_SKIP_CASES").is_some() {
            return;
        }
        for (name, instruction, labels, criteria, ordered, expected_argmax) in cases {
            let rendered = renderer
                .render(&state, instruction, labels, criteria, *ordered)
                .expect("render prompt");
            let ids = rendered.prompt_ids();
            assert_eq!(ids[0], 2, "prompt starts with the Gemma BOS token");
            let logits = model
                .letter_logits(&ids, rendered.letter_ids())
                .expect("forward");
            let probabilities =
                crate::families::support::temperature_softmax(&logits, CALIBRATION_TEMPERATURE)
                    .expect("softmax");
            let mut argmax = 0;
            for (index, probability) in probabilities.iter().enumerate() {
                if *probability > probabilities[argmax] {
                    argmax = index;
                }
            }
            println!(
                "{name}: prompt_tokens={} logits={logits:?} probabilities={probabilities:?} argmax={argmax}",
                ids.len()
            );
            if argmax != *expected_argmax {
                failures.push(format!(
                    "{name}: expected candidate {expected_argmax} to win, got {argmax}"
                ));
            } else if probabilities[argmax] <= 0.6 {
                failures.push(format!(
                    "{name}: winning probability {} is not decisive",
                    probabilities[argmax]
                ));
            }
        }
        assert!(failures.is_empty(), "smoke failures: {failures:?}");
    }
}
