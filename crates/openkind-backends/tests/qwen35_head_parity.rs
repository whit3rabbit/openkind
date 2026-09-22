use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use openkind_backends::qwen35::{
    PolicyAction, PrimitiveKind, ReferenceBundle, BACKBONE_ID, BACKBONE_REVISION, BUNDLE_SHA256,
    CALIBRATION_TEMPERATURE, FEATURE_WIDTH, ORDERING_TOLERANCE, POLICY_THRESHOLD,
    PROBABILITY_TOLERANCE, PROFILE_ID, REFERENCE_REVISION,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const FIXTURE_DIR: &str = "tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8";

#[derive(Debug, Deserialize)]
struct Manifest {
    files: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    request: GoldenRequest,
    expected_native_answers: Vec<ExpectedAnswer>,
    head_and_token_fixtures: Vec<HeadFixture>,
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
struct ExpectedAnswer {
    question_id: String,
    selected_id: Option<String>,
    option_probabilities: HashMap<String, f64>,
    none_probability: Option<f64>,
    top_probability: f64,
}

#[derive(Debug, Deserialize)]
struct HeadFixture {
    question_id: String,
    candidate_features: Vec<Vec<f32>>,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("decode {}: {error}", path.display()))
}

fn sha256_file(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}

fn primitive(value: &str) -> PrimitiveKind {
    match value {
        "choice" => PrimitiveKind::Choice,
        "noul" => PrimitiveKind::Noul,
        "score" => PrimitiveKind::Score,
        other => panic!("unexpected primitive {other}"),
    }
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "{label}: actual {actual}, expected {expected}, delta {delta}, tolerance {tolerance}"
    );
}

#[test]
fn vendored_parity_artifacts_match_original_manifest_digests() {
    let root = fixture_root();
    let manifest: Manifest = read_json(&root.join("BUNDLE_MANIFEST.json"));

    for relative in ["golden.json", "model/score_summary_seed17.safetensors"] {
        let expected = manifest
            .files
            .get(relative)
            .unwrap_or_else(|| panic!("manifest entry for {relative}"));
        assert_eq!(sha256_file(&root.join(relative)), *expected, "{relative}");
    }
}

#[test]
fn exported_head_fixtures_match_f64_rust_algebra_and_policy() {
    let root = fixture_root();
    let bundle = ReferenceBundle::load(&root).expect("load verified reference bundle");
    let profile = bundle.profile();

    assert_eq!(profile.source().profile_id(), PROFILE_ID);
    assert_eq!(profile.source().repository_revision(), REFERENCE_REVISION);
    assert_eq!(profile.source().bundle_sha256(), BUNDLE_SHA256);
    assert_eq!(profile.backbone().id(), BACKBONE_ID);
    assert_eq!(profile.backbone().revision(), BACKBONE_REVISION);
    assert_eq!(
        profile.parity().calibration_temperature(),
        CALIBRATION_TEMPERATURE
    );
    assert_eq!(profile.parity().policy_threshold(), POLICY_THRESHOLD);
    assert_eq!(
        profile.parity().probability_tolerance(),
        PROBABILITY_TOLERANCE
    );
    assert_eq!(profile.parity().ordering_tolerance(), ORDERING_TOLERANCE);

    let cases: Vec<GoldenCase> = read_json(&root.join("golden.json"));
    let mut replayed = 0;
    let mut selected_id_changes = 0;
    let mut policy_changes = 0;

    for case in cases {
        let questions: HashMap<_, _> = case
            .request
            .questions
            .iter()
            .map(|question| (question.id.as_str(), question))
            .collect();
        let answers: HashMap<_, _> = case
            .expected_native_answers
            .iter()
            .map(|answer| (answer.question_id.as_str(), answer))
            .collect();

        for fixture in &case.head_and_token_fixtures {
            replayed += 1;
            let question = questions[fixture.question_id.as_str()];
            let expected = answers[fixture.question_id.as_str()];
            let primitive = primitive(&question.primitive);

            assert!(
                fixture
                    .candidate_features
                    .iter()
                    .all(|feature| feature.len() == FEATURE_WIDTH),
                "{} feature width",
                fixture.question_id
            );

            let evaluation = bundle
                .head()
                .evaluate(primitive, &fixture.candidate_features)
                .unwrap_or_else(|error| panic!("evaluate {}: {error}", fixture.question_id));
            let actual_logits = evaluation.full_logits();
            let actual_probabilities = evaluation.full_probabilities();

            assert_eq!(actual_logits.len(), fixture.logits.len());
            assert_eq!(actual_probabilities.len(), fixture.probabilities.len());
            for (index, (actual, expected)) in actual_logits.iter().zip(&fixture.logits).enumerate()
            {
                assert_close(
                    &format!("{} logit {index}", fixture.question_id),
                    *actual,
                    *expected,
                    ORDERING_TOLERANCE,
                );
            }
            for (index, (actual, expected)) in actual_probabilities
                .iter()
                .zip(&fixture.probabilities)
                .enumerate()
            {
                assert_close(
                    &format!("{} probability {index}", fixture.question_id),
                    *actual,
                    *expected,
                    PROBABILITY_TOLERANCE,
                );
            }

            let actual_selected_id = evaluation
                .selected_candidate_index()
                .map(|index| question.options[index].id.as_str());
            if actual_selected_id != expected.selected_id.as_deref() {
                selected_id_changes += 1;
            }

            for (index, option) in question.options.iter().enumerate() {
                assert_close(
                    &format!("{} option {}", fixture.question_id, option.id),
                    evaluation.candidate_probabilities()[index],
                    expected.option_probabilities[&option.id],
                    PROBABILITY_TOLERANCE,
                );
            }
            assert_eq!(
                evaluation.none_probability().is_some(),
                primitive == PrimitiveKind::Choice
            );
            match (evaluation.none_probability(), expected.none_probability) {
                (Some(actual), Some(expected)) => assert_close(
                    &format!("{} none", fixture.question_id),
                    actual,
                    expected,
                    PROBABILITY_TOLERANCE,
                ),
                (None, None) => {}
                pair => panic!(
                    "{} none probability mismatch: {pair:?}",
                    fixture.question_id
                ),
            }
            assert_close(
                &format!("{} top probability", fixture.question_id),
                evaluation.top_probability(),
                expected.top_probability,
                PROBABILITY_TOLERANCE,
            );

            let expected_candidate_index = expected.selected_id.as_deref().map(|selected_id| {
                question
                    .options
                    .iter()
                    .position(|option| option.id == selected_id)
                    .unwrap_or_else(|| panic!("expected option {selected_id}"))
            });
            let expected_action = match expected_candidate_index {
                Some(candidate_index) if expected.top_probability >= POLICY_THRESHOLD => {
                    PolicyAction::Accept { candidate_index }
                }
                _ => PolicyAction::Review,
            };
            if evaluation.policy_action() != expected_action {
                policy_changes += 1;
            }
        }
    }

    assert_eq!(replayed, 4, "published golden fixture count");
    assert_eq!(selected_id_changes, 0);
    assert_eq!(policy_changes, 0);
}
