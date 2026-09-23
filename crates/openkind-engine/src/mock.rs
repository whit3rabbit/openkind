//! `MockEngine` — Phase 1 placeholder for a real model.
//!
//! It returns deterministic but plausibly-shaped answers so we can test the
//! full HTTP/gRPC/CLI path without loading any weights. Drop in the candle
//! or GGUF engine in Phase 2 and the server is unchanged.

use std::collections::HashMap;
use std::hash::Hasher;

use async_trait::async_trait;
use openkind_core::ModelInfo;
use openkind_core::{
    Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer, SystemRequest, SystemResponse, Usage,
};

use crate::{DecisionEngine, EngineResult};

/// Hash the question id and instructions into a deterministic seed so
/// repeated requests get the same answer. Determinism > realism here —
/// we just want the server to be testable.
///
/// The instructions value is hashed structurally (variant-tagged fields)
/// instead of being serialized to a JSON string first: same determinism,
/// no per-question formatting or allocation on the request path.
fn seed_for_question(id: &str, instructions: &serde_json::Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    let mut h = DefaultHasher::new();
    id.hash(&mut h);
    hash_value(&mut h, instructions);
    h.finish()
}

/// Fold a JSON value into a hasher with variant tags so distinct shapes
/// (for example the string `"1"` and the number `1`) hash differently.
fn hash_value<H: Hasher>(h: &mut H, value: &serde_json::Value) {
    use std::hash::Hash;
    match value {
        serde_json::Value::Null => h.write_u8(0),
        serde_json::Value::Bool(b) => {
            h.write_u8(1);
            b.hash(h);
        }
        serde_json::Value::Number(n) => {
            h.write_u8(2);
            if let Some(i) = n.as_i64() {
                h.write_u8(0);
                i.hash(h);
            } else if let Some(u) = n.as_u64() {
                h.write_u8(1);
                u.hash(h);
            } else if let Some(f) = n.as_f64() {
                h.write_u8(2);
                f.to_bits().hash(h);
            }
        }
        serde_json::Value::String(s) => {
            h.write_u8(3);
            s.hash(h);
        }
        serde_json::Value::Array(a) => {
            h.write_u8(4);
            a.len().hash(h);
            for item in a {
                hash_value(h, item);
            }
        }
        serde_json::Value::Object(m) => {
            h.write_u8(5);
            m.len().hash(h);
            for (key, item) in m {
                key.hash(h);
                hash_value(h, item);
            }
        }
    }
}

/// Small deterministic SplitMix64 generator. Mock answers only need stable
/// per-question determinism, not cryptographic quality, and seeding a
/// ChaCha-backed `StdRng` per question dominated mock dispatch profiles.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of resolution.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)` via multiply-shift.
    fn next_below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// A mock that returns distributions seeded from `(id, instructions)`.
/// Always satisfies sum-to-1 and confidence-in-range so the validator
/// passes.
pub struct MockEngine {
    backend: String,
}

impl MockEngine {
    /// Create a new `MockEngine` with the default backend identifier (`"mock"`).
    pub fn new() -> Self {
        Self {
            backend: "mock".into(),
        }
    }

    /// Create a new `MockEngine` with a custom backend identifier string.
    pub fn with_backend(backend: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
        }
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DecisionEngine for MockEngine {
    fn backend_id(&self) -> &str {
        &self.backend
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(), // EngineRegistry overrides with the alias.
            description: "Deterministic fake answers for testing the wire protocol.".into(),
            release_date: "2026-01-01".into(),
        }
    }

    async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse> {
        let mut answers: HashMap<String, Answer, _> = HashMap::default();

        for (id, q) in &req.questions {
            let mut rng = SplitMix64::new(seed_for_question(id, instructions_of(q)));
            let answer = match q {
                Question::Noul(_) => Answer::Noul(NoulAnswer {
                    noul: 0.05 + rng.next_f64() * 0.90,
                }),
                Question::Choice(cq) => {
                    let mut keys: Vec<&String> = cq.criteria.keys().collect();
                    keys.sort(); // HashMap iteration is non-deterministic
                    if keys.is_empty() {
                        // `dispatch` validates requests first, but the engine is
                        // also a public library API — refuse instead of panicking.
                        return Err(crate::EngineError::Backend {
                            backend: self.backend.clone(),
                            message: format!("choice question `{id}` has empty criteria"),
                        });
                    }
                    let n = keys.len() as f64;
                    let raw: Vec<f64> = (0..keys.len()).map(|_| rng.next_f64()).collect();
                    let sum: f64 = raw.iter().sum();
                    let probs: HashMap<String, f64> = if sum <= 0.0 || !sum.is_finite() {
                        let uniform = 1.0 / n;
                        keys.iter().map(|k| ((*k).clone(), uniform)).collect()
                    } else {
                        keys.iter()
                            .zip(raw.iter())
                            .map(|(k, v)| ((*k).clone(), v / sum))
                            .collect()
                    };
                    let max_p = probs.values().cloned().fold(f64::NEG_INFINITY, f64::max);
                    // Tie-break on the keys' sorted order so we always
                    // pick the same option for the same (id, instructions)
                    // seed even with float noise.
                    let mut best_key: Option<String> = None;
                    for k in keys.iter() {
                        let p = probs[*k];
                        match &best_key {
                            None => best_key = Some((*k).clone()),
                            Some(bk) => {
                                if p > probs[bk] || (p == probs[bk] && *k < bk) {
                                    best_key = Some((*k).clone());
                                }
                            }
                        }
                    }
                    let choice = best_key.unwrap_or_else(|| keys[0].clone());
                    let raw_conf = max_p - (1.0 - max_p) / (n - 1.0).max(1.0);
                    let confidence = if raw_conf.is_nan() {
                        0.0
                    } else {
                        raw_conf.clamp(0.0, 1.0)
                    };
                    Answer::Choice(ChoiceAnswer {
                        choice,
                        probabilities: probs,
                        confidence,
                    })
                }
                Question::Score(sq) => {
                    let n = sq.criteria.len();
                    if n == 0 {
                        return Err(crate::EngineError::Backend {
                            backend: self.backend.clone(),
                            message: format!("score question `{id}` has empty criteria"),
                        });
                    }
                    let mut probs = vec![0.0f64; n];
                    // Bias the peak toward the middle of the range — feels
                    // more realistic than uniform for a mock.
                    let peak = rng.next_below(n);
                    for (i, p) in probs.iter_mut().enumerate() {
                        let d = (i as f64 - peak as f64).abs();
                        *p = (-d * 1.5).exp();
                    }
                    let sum: f64 = probs.iter().sum();
                    let probs: Vec<f64> = if sum <= 0.0 || !sum.is_finite() {
                        let uniform = 1.0 / (n as f64);
                        vec![uniform; n]
                    } else {
                        probs.iter().map(|p| p / sum).collect()
                    };
                    let score: f64 = probs
                        .iter()
                        .enumerate()
                        .map(|(i, p)| i as f64 * p)
                        .sum::<f64>();
                    let legend: HashMap<String, String> = sq
                        .criteria
                        .iter()
                        .enumerate()
                        .map(|(i, l)| (i.to_string(), l.clone()))
                        .collect();
                    let probs_map: HashMap<String, f64> = probs
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| (i.to_string(), p))
                        .collect();
                    let max_p = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let raw_conf = max_p - (1.0 - max_p) / (n as f64 - 1.0).max(1.0);
                    let confidence = if raw_conf.is_nan() {
                        0.0
                    } else {
                        raw_conf.clamp(0.0, 1.0)
                    };
                    Answer::Score(ScoreAnswer {
                        score,
                        legend,
                        probabilities: probs_map,
                        confidence,
                    })
                }
            };
            answers.insert(id.clone(), answer);
        }

        Ok(SystemResponse {
            model: req.model,
            answers,
            // Filled in by `dispatch` via the estimator.
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        })
    }
}

fn instructions_of(q: &Question) -> &serde_json::Value {
    match q {
        Question::Noul(n) => &n.instructions,
        Question::Choice(c) => &c.instructions,
        Question::Score(s) => &s.instructions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_core::{State, SystemRequest};

    fn req(model: &str) -> SystemRequest {
        let mut questions = HashMap::default();
        questions.insert(
            "is_urgent".into(),
            Question::Noul(openkind_core::NoulQuestion {
                instructions: serde_json::json!("Does this convey urgency?"),
                criteria: None,
            }),
        );
        questions.insert(
            "dept".into(),
            Question::Choice(openkind_core::ChoiceQuestion {
                instructions: serde_json::json!("Which team?"),
                criteria: [
                    ("billing".to_string(), Some("pay".to_string())),
                    ("technical".to_string(), Some("bugs".to_string())),
                ]
                .into_iter()
                .collect(),
            }),
        );
        questions.insert(
            "frust".into(),
            Question::Score(openkind_core::ScoreQuestion {
                instructions: serde_json::json!("How frustrated?"),
                criteria: vec!["Calm".into(), "Angry".into()],
            }),
        );
        SystemRequest {
            state: State::Text("Help!".into()),
            model: model.into(),
            questions,
        }
    }

    #[tokio::test]
    async fn mock_returns_one_answer_per_question() {
        let engine = MockEngine::new();
        let resp = engine.evaluate(req("mock")).await.unwrap();
        assert_eq!(resp.answers.len(), 3);
        assert!(matches!(resp.answers["is_urgent"], Answer::Noul(_)));
        assert!(matches!(resp.answers["dept"], Answer::Choice(_)));
        assert!(matches!(resp.answers["frust"], Answer::Score(_)));
    }

    #[tokio::test]
    async fn mock_is_deterministic_for_same_inputs() {
        let engine = MockEngine::new();
        let a = engine.evaluate(req("mock")).await.unwrap();
        let b = engine.evaluate(req("mock")).await.unwrap();
        match (&a.answers["dept"], &b.answers["dept"]) {
            (Answer::Choice(x), Answer::Choice(y)) => {
                assert_eq!(x.choice, y.choice);
                assert_eq!(x.probabilities, y.probabilities);
            }
            _ => panic!("expected choice answers"),
        }
    }

    #[tokio::test]
    async fn choice_probabilities_sum_to_one() {
        let engine = MockEngine::new();
        let resp = engine.evaluate(req("mock")).await.unwrap();
        if let Answer::Choice(c) = &resp.answers["dept"] {
            let sum: f64 = c.probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-4);
            assert!((0.0..=1.0).contains(&c.confidence));
        } else {
            panic!("expected choice");
        }
    }

    #[tokio::test]
    async fn score_probabilities_sum_to_one_and_fields_valid() {
        let engine = MockEngine::new();
        let resp = engine.evaluate(req("mock")).await.unwrap();
        if let Answer::Score(s) = &resp.answers["frust"] {
            let sum: f64 = s.probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-4);
            assert!((0.0..=1.0).contains(&s.confidence));
            assert!(s.score >= 0.0 && s.score <= 1.0);
            let expected_score: f64 = s
                .probabilities
                .iter()
                .map(|(k, &p)| k.parse::<f64>().unwrap() * p)
                .sum();
            assert!((expected_score - s.score).abs() < 1e-6);
            assert_eq!(s.legend.len(), 2);
            assert_eq!(s.legend.get("0").unwrap(), "Calm");
            assert_eq!(s.legend.get("1").unwrap(), "Angry");
        } else {
            panic!("expected score");
        }
    }

    #[tokio::test]
    async fn noul_in_range() {
        let engine = MockEngine::new();
        let resp = engine.evaluate(req("mock")).await.unwrap();
        if let Answer::Noul(n) = &resp.answers["is_urgent"] {
            assert!(n.noul >= 0.05 && n.noul <= 0.95);
        } else {
            panic!("expected noul");
        }
    }

    #[tokio::test]
    async fn empty_criteria_returns_error_not_panic() {
        let engine = MockEngine::new();
        let mut questions = HashMap::default();
        questions.insert(
            "bad_choice".into(),
            Question::Choice(openkind_core::ChoiceQuestion {
                instructions: serde_json::json!("pick"),
                criteria: HashMap::default(),
            }),
        );
        questions.insert(
            "bad_score".into(),
            Question::Score(openkind_core::ScoreQuestion {
                instructions: serde_json::json!("rate"),
                criteria: Vec::new(),
            }),
        );
        let request = SystemRequest {
            state: openkind_core::State::Text("x".into()),
            model: "mock".into(),
            questions,
        };
        let err = engine.evaluate(request).await.unwrap_err();
        assert!(
            matches!(err, crate::EngineError::Backend { .. }),
            "expected Backend error, got {err:?}"
        );
    }

    #[test]
    fn mock_backend_custom_id_and_metadata() {
        let engine = MockEngine::with_backend("my-custom-engine");
        assert_eq!(engine.backend_id(), "my-custom-engine");
        let meta = engine.model_metadata();
        assert_eq!(meta.release_date, "2026-01-01");
        assert!(meta.description.contains("wire protocol"));
    }

    #[test]
    fn seed_for_question_distinguishes_json_shapes() {
        // One representative per `Value` variant, plus the integer/float
        // number encodings: the variant tags must keep every seed distinct.
        let shapes = [
            serde_json::json!("1"),
            serde_json::json!(1),
            serde_json::json!(1.0),
            serde_json::json!(true),
            serde_json::Value::Null,
            serde_json::json!([1]),
            serde_json::json!({"a": 1}),
        ];
        let seeds: Vec<u64> = shapes.iter().map(|v| seed_for_question("q", v)).collect();
        let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(
            unique.len(),
            seeds.len(),
            "variant tags must keep seeds distinct, got {seeds:?}"
        );
        // Same input always hashes to the same seed.
        for value in &shapes {
            assert_eq!(seed_for_question("q", value), seed_for_question("q", value));
        }
        // Deeply nested values recurse without panicking and stay stable.
        let nested = serde_json::json!([[1, {"b": [2.5]}], {"a": null, "c": [true, "x"]}]);
        assert_eq!(
            seed_for_question("deep", &nested),
            seed_for_question("deep", &nested)
        );
    }

    #[test]
    fn splitmix64_values_stay_in_range() {
        let mut rng = SplitMix64::new(0xDEAD_BEEF);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "next_f64 escaped [0, 1): {x}");
        }
        for n in [1usize, 2, 7, 100] {
            for _ in 0..10_000 {
                let i = rng.next_below(n);
                assert!(i < n, "next_below({n}) escaped range: {i}");
            }
        }
        // The degenerate single-criterion score draw has exactly one
        // valid index.
        assert_eq!(rng.next_below(1), 0);
        assert_eq!(rng.next_below(1), 0);
    }

    #[tokio::test]
    async fn mock_is_deterministic_for_non_string_instructions() {
        let request = |instructions: serde_json::Value| {
            let mut questions = HashMap::default();
            questions.insert(
                "structured".into(),
                Question::Noul(openkind_core::NoulQuestion {
                    instructions,
                    criteria: None,
                }),
            );
            SystemRequest {
                state: State::Text("x".into()),
                model: "mock".into(),
                questions,
            }
        };
        let engine = MockEngine::new();

        let structured = serde_json::json!([{"role": "user", "text": "urgent?"}]);
        let a = engine.evaluate(request(structured.clone())).await.unwrap();
        let b = engine.evaluate(request(structured)).await.unwrap();
        match (&a.answers["structured"], &b.answers["structured"]) {
            (Answer::Noul(x), Answer::Noul(y)) => assert_eq!(x.noul, y.noul),
            _ => panic!("expected noul answers"),
        }

        // The string `"1"` and the number `1` hash differently, so their
        // answers must not alias.
        let number = engine
            .evaluate(request(serde_json::json!(1)))
            .await
            .unwrap();
        let string = engine
            .evaluate(request(serde_json::json!("1")))
            .await
            .unwrap();
        match (&number.answers["structured"], &string.answers["structured"]) {
            (Answer::Noul(x), Answer::Noul(y)) => assert_ne!(x.noul, y.noul),
            _ => panic!("expected noul answers"),
        }
    }
}
