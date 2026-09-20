use std::fs;
use std::path::{Path, PathBuf};

use opendecision_backends::qwen35::{
    CandidateText, Qwen35Error, Qwen35Tokenizer, BUNDLE_SHA256, PROFILE_ID,
    STATE_FIRST_RENDERER_ID, TOKENIZER_BACKEND_SHA256,
};
use serde::Deserialize;

const PHASE3B_DIR: &str = "research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z";

#[derive(Debug, Deserialize)]
struct GoldenCase {
    request: GoldenRequest,
}

#[derive(Debug, Deserialize)]
struct GoldenRequest {
    state: String,
    questions: Vec<GoldenQuestion>,
}

#[derive(Debug, Deserialize)]
struct GoldenQuestion {
    id: String,
    instruction: String,
    options: Vec<GoldenOption>,
}

#[derive(Debug, Deserialize)]
struct GoldenOption {
    label: String,
    criteria: String,
}

#[derive(Debug, Deserialize)]
struct TokenFixtureFile {
    schema: String,
    profile_id: String,
    bundle_sha256: String,
    tokenizer_backend_hash: String,
    layout: String,
    segmentation: String,
    records: Vec<TokenRecord>,
}

#[derive(Debug, Deserialize)]
struct TokenRecord {
    fixture_case: usize,
    question_id: String,
    root_ids: Vec<u32>,
    question_ids: Vec<u32>,
    candidate_suffix_ids: Vec<Vec<u32>>,
    full_candidate_ids: Vec<Vec<u32>>,
    checks: TokenChecks,
}

#[derive(Debug, Deserialize)]
struct TokenChecks {
    root_ids_exact: bool,
    question_ids_exact: bool,
    candidate_suffix_ids_exact: bool,
    full_candidate_ids_exact: bool,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn phase3b_root() -> PathBuf {
    workspace_root().join(PHASE3B_DIR)
}

fn tokenizer_path() -> PathBuf {
    phase3b_root().join("backbone_runtime/tokenizer/tokenizer.json")
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8/golden.json")
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("decode {}: {error}", path.display()))
}

#[test]
fn phase3b_state_first_segments_match_all_four_exported_records_exactly() {
    let fixtures: TokenFixtureFile = read_json(&phase3b_root().join("TOKEN_FIXTURES.json"));
    let cases: Vec<GoldenCase> = read_json(&golden_path());

    assert_eq!(fixtures.schema, "opendecision-qwen35-token-fixtures/v1");
    assert_eq!(fixtures.profile_id, PROFILE_ID);
    assert_eq!(fixtures.bundle_sha256, BUNDLE_SHA256);
    assert_eq!(fixtures.tokenizer_backend_hash, TOKENIZER_BACKEND_SHA256);
    assert_eq!(fixtures.layout, STATE_FIRST_RENDERER_ID);
    assert_eq!(
        fixtures.segmentation,
        "segments encoded independently with add_special_tokens=False; concatenate exact IDs"
    );
    assert_eq!(fixtures.records.len(), 4);

    let tokenizer = Qwen35Tokenizer::from_file(tokenizer_path()).expect("load pinned tokenizer");
    for record in fixtures.records {
        assert!(record.checks.root_ids_exact);
        assert!(record.checks.question_ids_exact);
        assert!(record.checks.candidate_suffix_ids_exact);
        assert!(record.checks.full_candidate_ids_exact);

        let request = &cases[record.fixture_case].request;
        let question = request
            .questions
            .iter()
            .find(|question| question.id == record.question_id)
            .unwrap_or_else(|| panic!("missing question {}", record.question_id));
        let candidates: Vec<_> = question
            .options
            .iter()
            .map(|option| CandidateText::new(&option.label, &option.criteria))
            .collect();
        let actual = tokenizer
            .encode_state_first(&request.state, &question.instruction, &candidates)
            .unwrap_or_else(|error| panic!("encode {}: {error}", record.question_id));

        assert_eq!(
            actual.root_ids(),
            record.root_ids,
            "{} root",
            record.question_id
        );
        assert_eq!(
            actual.question_ids(),
            record.question_ids,
            "{} question",
            record.question_id
        );
        assert_eq!(
            actual.candidate_suffix_ids(),
            record.candidate_suffix_ids,
            "{} suffixes",
            record.question_id
        );
        assert_eq!(
            actual.full_candidate_ids(),
            record.full_candidate_ids,
            "{} full sequences",
            record.question_id
        );
    }
}

#[test]
fn tokenizer_loader_rejects_an_unpinned_artifact() {
    let error = Qwen35Tokenizer::from_file(phase3b_root().join("TOKEN_FIXTURES.json"))
        .err()
        .expect("wrong artifact must fail closed");
    assert!(matches!(error, Qwen35Error::DigestMismatch { .. }));
}

#[test]
fn renderer_rejects_invalid_cardinality_and_truncation() {
    let tokenizer = Qwen35Tokenizer::from_file(tokenizer_path()).expect("load pinned tokenizer");
    let one = [CandidateText::new("one", "only candidate")];
    assert!(matches!(
        tokenizer.encode_state_first("state", "question", &one),
        Err(Qwen35Error::InvalidInput(_))
    ));

    let two = [
        CandidateText::new("one", "first candidate"),
        CandidateText::new("two", "second candidate"),
    ];
    let overlong_state = "evidence ".repeat(4_000);
    let error = tokenizer
        .encode_state_first(&overlong_state, "question", &two)
        .expect_err("frozen renderer must reject instead of truncate");
    assert!(matches!(error, Qwen35Error::InvalidInput(_)));
}
