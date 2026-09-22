use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::qwen35::{
    BackboneReference, LayerKind, PolicyAction, PrimitiveKind, Qwen35Error, ReferenceBundle,
    FEATURE_WIDTH, POLICY_THRESHOLD, PROBABILITY_TOLERANCE,
};
use serde::Deserialize;

const PHASE3B_DIR: &str = "research/OpenKind_Phase3B_BackboneParity_20260920T152206Z";

#[derive(Debug, Deserialize)]
struct TokenFixtures {
    records: Vec<TokenRecord>,
}

#[derive(Debug, Deserialize)]
struct TokenRecord {
    full_candidate_ids: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    request: GoldenRequest,
}

#[derive(Debug, Deserialize)]
struct GoldenRequest {
    questions: Vec<GoldenQuestion>,
}

#[derive(Debug, Deserialize)]
struct GoldenQuestion {
    id: String,
    primitive: String,
    options: Vec<GoldenOption>,
}

#[derive(Debug, Deserialize)]
struct GoldenOption {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ProbabilityReference {
    records: Vec<ProbabilityRecord>,
}

#[derive(Debug, Deserialize)]
struct ProbabilityRecord {
    fixture_case: usize,
    answers: Vec<ExpectedAnswer>,
}

#[derive(Debug, Deserialize)]
struct ExpectedAnswer {
    question_id: String,
    selected_id: String,
    option_probabilities: HashMap<String, f64>,
    none_probability: Option<f64>,
    top_probability: f64,
}

fn phase3b_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(PHASE3B_DIR)
}

fn head_fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8")
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    serde_json::from_slice(
        &fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("decode {}: {error}", path.display()))
}

fn primitive(value: &str) -> PrimitiveKind {
    match value {
        "choice" => PrimitiveKind::Choice,
        "noul" => PrimitiveKind::Noul,
        "score" => PrimitiveKind::Score,
        other => panic!("unexpected primitive {other}"),
    }
}

fn assert_close(label: &str, actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= PROBABILITY_TOLERANCE,
        "{label}: actual {actual}, expected {expected}, delta {delta}"
    );
}

#[test]
fn phase3b_reference_loads_all_strict_identity_and_tensor_gates() {
    let root = phase3b_root();
    let reference = BackboneReference::load(&root).expect("load Phase 3B reference");
    let token_fixtures: TokenFixtures = read_json(&root.join("TOKEN_FIXTURES.json"));

    assert_eq!(reference.layer_kinds().len(), 32);
    assert_eq!(
        reference
            .layer_kinds()
            .iter()
            .filter(|kind| **kind == LayerKind::LinearAttention)
            .count(),
        24
    );
    assert_eq!(
        reference
            .layer_kinds()
            .iter()
            .filter(|kind| **kind == LayerKind::FullAttention)
            .count(),
        8
    );
    assert_eq!(reference.trace_stages().len(), 34);
    assert_eq!(reference.trace_stages()[0].stage(), "embedding");
    assert_eq!(reference.trace_stages()[33].stage(), "final_norm");
    assert_eq!(reference.full_sequence_records().len(), 10);
    assert!(reference
        .full_sequence_records()
        .iter()
        .all(|record| record.reference_feature_delta() <= 0.000_1));
    assert_eq!(
        reference.trace_input_ids(),
        token_fixtures.records[0].full_candidate_ids[0]
    );

    for stage in reference.trace_stages() {
        let vector = reference
            .vector(stage.tensor_key())
            .unwrap_or_else(|| panic!("missing {}", stage.tensor_key()));
        assert_eq!(vector.len(), FEATURE_WIDTH);
        let comparison = reference
            .compare(stage.tensor_key(), vector)
            .expect("compare reference to itself");
        assert_eq!(comparison.max_abs(), 0.0);
        assert_eq!(comparison.rms(), 0.0);
        assert!((comparison.cosine() - 1.0).abs() <= f64::EPSILON);
    }
}

#[test]
fn phase3b_candidate_vectors_replay_the_frozen_probability_contract() {
    let root = phase3b_root();
    let reference = BackboneReference::load(&root).expect("load Phase 3B reference");
    let bundle = ReferenceBundle::load(head_fixture_root()).expect("load selected head");
    let cases: Vec<GoldenCase> = read_json(&head_fixture_root().join("golden.json"));
    let probabilities: ProbabilityReference = read_json(&root.join("PROBABILITY_REFERENCE.json"));
    let mut replayed = 0;

    for record in probabilities.records {
        let questions: HashMap<_, _> = cases[record.fixture_case]
            .request
            .questions
            .iter()
            .map(|question| (question.id.as_str(), question))
            .collect();
        for expected in record.answers {
            replayed += 1;
            let question = questions[expected.question_id.as_str()];
            let mut feature_records: Vec<_> = reference
                .full_sequence_records()
                .iter()
                .filter(|candidate| {
                    candidate.fixture_case() == record.fixture_case
                        && candidate.question_id() == expected.question_id
                })
                .collect();
            feature_records.sort_by_key(|candidate| candidate.candidate_index());
            let features: Vec<_> = feature_records
                .iter()
                .map(|candidate| {
                    reference
                        .vector(candidate.tensor_key())
                        .expect("candidate feature")
                        .to_vec()
                })
                .collect();
            assert_eq!(features.len(), question.options.len());

            let evaluation = bundle
                .head()
                .evaluate(primitive(&question.primitive), &features)
                .unwrap_or_else(|error| panic!("evaluate {}: {error}", expected.question_id));
            let selected_index = evaluation
                .selected_candidate_index()
                .expect("Phase 3B references select a real candidate");
            assert_eq!(question.options[selected_index].id, expected.selected_id);
            for (index, option) in question.options.iter().enumerate() {
                assert_close(
                    &format!("{} option {}", expected.question_id, option.id),
                    evaluation.candidate_probabilities()[index],
                    expected.option_probabilities[&option.id],
                );
            }
            match (evaluation.none_probability(), expected.none_probability) {
                (Some(actual), Some(expected_none)) => assert_close(
                    &format!("{} none", expected.question_id),
                    actual,
                    expected_none,
                ),
                (None, None) => {}
                pair => panic!("{} none mismatch: {pair:?}", expected.question_id),
            }
            assert_close(
                &format!("{} top", expected.question_id),
                evaluation.top_probability(),
                expected.top_probability,
            );
            let expected_action = if expected.top_probability >= POLICY_THRESHOLD {
                PolicyAction::Accept {
                    candidate_index: selected_index,
                }
            } else {
                PolicyAction::Review
            };
            assert_eq!(evaluation.policy_action(), expected_action);
        }
    }
    assert_eq!(replayed, 4);
}

#[test]
fn stage_comparison_reports_drift_and_rejects_invalid_vectors() {
    let reference = BackboneReference::load(phase3b_root()).expect("load Phase 3B reference");
    let key = "diagnostic.embedding";
    let mut actual = reference.vector(key).expect("embedding vector").to_vec();
    actual[0] += 1.0;
    let comparison = reference.compare(key, &actual).expect("compare drift");
    assert_eq!(comparison.max_abs(), 1.0);
    assert!((comparison.rms() - 1.0 / (FEATURE_WIDTH as f64).sqrt()).abs() < 1e-12);
    assert!(comparison.cosine() < 1.0);

    let zeros = vec![0.0; FEATURE_WIDTH];
    assert_eq!(
        reference
            .compare(key, &zeros)
            .expect("compare zeros")
            .cosine(),
        0.0
    );

    assert!(matches!(
        reference.compare(key, &actual[..FEATURE_WIDTH - 1]),
        Err(Qwen35Error::InvalidInput(_))
    ));
    actual[0] = f32::NAN;
    assert!(matches!(
        reference.compare(key, &actual),
        Err(Qwen35Error::Numerical(_))
    ));
    assert!(matches!(
        reference.compare("unknown", &actual),
        Err(Qwen35Error::InvalidInput(_))
    ));
}
