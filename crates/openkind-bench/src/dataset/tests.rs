//! Offline dataset tests: metric identities, template cleaning, synthetic
//! parquet materialization, and a mock-engine evaluation end to end. Nothing
//! here touches the network; the synthetic parquet shards are authored by the
//! test itself.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use serde_json::{Map, Value};
use sha2::Digest;

use super::eval::EvalArgs;
use super::materialize::materialize;
use super::metrics::{
    accuracy_at_coverage, aurc, auroc, best_threshold_tuned_on, ece, macro_f1, spearman,
};
use super::templates::CLINC150_INTENTS;

fn approx(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

#[test]
fn auroc_matches_hand_computation() {
    approx(
        auroc(&[0.1, 0.4, 0.35, 0.8], &[false, false, true, true]).unwrap(),
        0.75,
    );
    approx(auroc(&[0.5, 0.5], &[false, true]).unwrap(), 0.5);
    approx(auroc(&[0.2, 0.9], &[true, false]).unwrap(), 0.0);
    assert!(auroc(&[0.5, 0.5], &[true, true]).is_none());
}

#[test]
fn spearman_matches_hand_computation() {
    approx(spearman(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap(), 1.0);
    approx(spearman(&[1.0, 2.0, 3.0], &[6.0, 4.0, 2.0]).unwrap(), -1.0);
    // Ties: ranks [1.5, 1.5, 3] vs [1, 2.5, 2.5] give exactly 0.5.
    approx(spearman(&[1.0, 1.0, 2.0], &[1.0, 2.0, 2.0]).unwrap(), 0.5);
    assert!(spearman(&[1.0], &[1.0]).is_none());
}

#[test]
fn macro_f1_matches_hand_computation() {
    approx(
        macro_f1(
            &["a".to_owned(), "a".to_owned(), "b".to_owned()],
            &["a".to_owned(), "b".to_owned(), "b".to_owned()],
        )
        .unwrap(),
        2.0 / 3.0,
    );
}

#[test]
fn aurc_and_coverage_match_hand_computation() {
    approx(aurc(&[0.9, 0.1], &[true, false]), 0.25);
    let at_two_thirds =
        accuracy_at_coverage(&[0.9, 0.8, 0.2], &[true, false, true], 2.0 / 3.0).unwrap();
    approx(at_two_thirds["accuracy"].as_f64().unwrap(), 0.5);
    assert_eq!(at_two_thirds["answered"].as_u64().unwrap(), 2);
}

#[test]
fn ece_matches_hand_computation() {
    // Mean confidence 0.65 against perfect accuracy: miscalibrated by 0.35.
    approx(ece(&[0.65; 4], &[true, true, true, true], 10), 0.35);
    // Confidence matches the empirical accuracy exactly.
    approx(ece(&[0.5; 4], &[true, true, false, false], 10), 0.0);
}

#[test]
fn tuned_threshold_finds_the_best_cut() {
    let (threshold, accuracy) =
        best_threshold_tuned_on(&[0.2, 0.6, 0.4], &[false, true, false]).unwrap();
    approx(accuracy, 1.0);
    assert!((0.4..=0.5).contains(&threshold), "{threshold}");
}

#[test]
fn hellaswag_clean_matches_the_reference_pipeline() {
    use super::materialize::hellaswag_clean;
    assert_eq!(
        hellaswag_clean("A [title] test. [keep] removed.. dots  double"),
        "A. test. removed. dots double"
    );
    assert_eq!(hellaswag_clean("trailing [unclosed"), "trailing [unclosed");
    assert_eq!(hellaswag_clean("  spaced out  "), "spaced out");
}

#[test]
fn clinc150_intent_table_is_well_formed() {
    assert_eq!(CLINC150_INTENTS.len(), 151);
    let unique: std::collections::HashSet<&&str> = CLINC150_INTENTS.iter().collect();
    assert_eq!(unique.len(), 151);
    assert!(CLINC150_INTENTS.contains(&"oos"));
}

fn write_parquet(path: &std::path::Path, batch: RecordBatch) -> u64 {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    std::fs::metadata(path).unwrap().len()
}

fn sst2_batch(rows: &[(&str, i64)]) -> RecordBatch {
    let sentences: Vec<&str> = rows.iter().map(|(text, _)| *text).collect();
    let labels: Vec<i64> = rows.iter().map(|(_, label)| *label).collect();
    RecordBatch::try_from_iter(vec![
        (
            "sentence",
            Arc::new(StringArray::from(sentences)) as ArrayRef,
        ),
        ("label", Arc::new(Int64Array::from(labels)) as ArrayRef),
    ])
    .unwrap()
}

/// Install a synthetic dataset into a store root with real digests.
fn install_synthetic(
    root: &std::path::Path,
    name: &str,
    files: &[(&str, RecordBatch)],
    splits: &[(&str, &str)],
    configs: &[&str],
    primitives: &[&str],
) {
    use openkind_datasets::{DatasetEntry, DatasetFile};
    use std::collections::BTreeMap;
    let mut entries = Vec::new();
    for (path, batch) in files {
        let dir = root.join("datasets").join(name).join("files");
        let relative: Vec<&str> = path.split('/').collect();
        let file_dir = dir.join(relative[..relative.len() - 1].join("/"));
        std::fs::create_dir_all(&file_dir).unwrap();
        let full = dir.join(path);
        let size = write_parquet(&full, batch.clone());
        let bytes = std::fs::read(&full).unwrap();
        let sha = format!("{:x}", sha2::Sha256::digest(&bytes));
        entries.push(DatasetFile {
            path: (*path).to_owned(),
            size,
            sha256: sha,
        });
    }
    let mut split_map = BTreeMap::new();
    for (ours, upstream) in splits {
        split_map.insert((*ours).to_owned(), (*upstream).to_owned());
    }
    let entry = DatasetEntry {
        name: name.to_owned(),
        description: "synthetic".to_owned(),
        hf_repo: "owner/data".to_owned(),
        hf_revision: "a".repeat(40),
        convert_revision: "b".repeat(40),
        files: entries,
        splits: split_map,
        configs: configs.iter().map(|c| (*c).to_owned()).collect(),
        license: "synthetic".to_owned(),
        gated: openkind_datasets::Gated::None,
        access_note: String::new(),
        task_family: "synthetic".to_owned(),
        primitives: primitives.iter().map(|p| (*p).to_owned()).collect(),
        template: openkind_datasets::definitions::template(),
    };
    std::fs::create_dir_all(root.join("datasets").join(name)).unwrap();
    std::fs::write(
        root.join("datasets").join(name).join("entry.json"),
        serde_json::to_vec(&entry).unwrap(),
    )
    .unwrap();
}

#[test]
fn materializes_synthetic_sst2_with_dedup_and_nested_limits() {
    let dir = tempfile::tempdir().unwrap();
    let batch = sst2_batch(&[
        ("a charming journey . ", 1),
        ("a terrible mess . ", 0),
        ("a charming journey . ", 1),
    ]);
    install_synthetic(
        dir.path(),
        "sst2",
        &[("default/test/0000.parquet", batch)],
        &[("eval", "test"), ("dev", "train")],
        &["default"],
        &["choice"],
    );
    let store = openkind_datasets::DatasetStore::new(dir.path().to_owned()).unwrap();
    let installed = store.installed("sst2").unwrap();
    let full = materialize(&installed, "eval", None).unwrap();
    assert_eq!(full.rows.len(), 2, "duplicate request dropped");
    assert_eq!(full.skipped_duplicates, 1);
    let limited = materialize(&installed, "eval", Some(1)).unwrap();
    assert_eq!(limited.rows.len(), 1);
    // Nested sampling: the smaller limit selects the first of the larger set.
    assert_eq!(limited.rows[0].row.id, full.rows[0].row.id);
    let row = &full.rows[0];
    let crate::workload::QuestionSpec::Choice { text, options } = &row.row.question else {
        panic!("expected a choice question");
    };
    assert_eq!(
        text,
        "What is the overall sentiment of `review` toward the movie?"
    );
    let ids: Vec<&str> = options.iter().map(|o| o.id.as_str()).collect();
    assert_eq!(ids, vec!["negative", "positive"]);
    assert!(options.iter().all(|o| o.description.is_some()));
    assert!(row.gold == "negative" || row.gold == "positive");
    assert_eq!(row.subset, "default");
}

#[test]
fn materializes_boolq_and_stsb_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let boolq_batch = RecordBatch::try_from_iter(vec![
        (
            "question",
            Arc::new(StringArray::from(vec!["does ethanol take energy"])) as ArrayRef,
        ),
        (
            "answer",
            Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
        ),
        (
            "passage",
            Arc::new(StringArray::from(vec!["All biomass goes through steps."])) as ArrayRef,
        ),
    ])
    .unwrap();
    install_synthetic(
        dir.path(),
        "boolq",
        &[("default/validation/0000.parquet", boolq_batch)],
        &[("eval", "validation"), ("dev", "train")],
        &["default"],
        &["noul"],
    );
    let store = openkind_datasets::DatasetStore::new(dir.path().to_owned()).unwrap();
    let rows = materialize(&store.installed("boolq").unwrap(), "eval", None)
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert!(matches!(
        rows[0].row.question,
        crate::workload::QuestionSpec::Noul { .. }
    ));
    assert_eq!(rows[0].gold, "true");
    let openkind_core::State::Object(state) = &rows[0].row.state else {
        panic!("expected an object state");
    };
    let question = state["question"].as_str().unwrap();
    assert!(question.starts_with('D') && question.ends_with('?'));

    let stsb_batch = RecordBatch::try_from_iter(vec![
        (
            "sentence1",
            Arc::new(StringArray::from(vec!["A girl is styling her hair."])) as ArrayRef,
        ),
        (
            "sentence2",
            Arc::new(StringArray::from(vec!["A girl is brushing her hair."])) as ArrayRef,
        ),
        ("score", Arc::new(Float64Array::from(vec![0.5])) as ArrayRef),
    ])
    .unwrap();
    install_synthetic(
        dir.path(),
        "stsb",
        &[("default/test/0000.parquet", stsb_batch)],
        &[("eval", "test"), ("dev", "validation")],
        &["default"],
        &["score"],
    );
    let rows = materialize(&store.installed("stsb").unwrap(), "eval", None)
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    let crate::workload::QuestionSpec::Score { levels, .. } = &rows[0].row.question else {
        panic!("expected a score question");
    };
    assert_eq!(levels.len(), 6);
    approx(rows[0].gold.parse::<f64>().unwrap(), 2.5);
}

#[test]
fn materializes_arc_structured_choices() {
    let dir = tempfile::tempdir().unwrap();
    let labels: Vec<Option<String>> = vec![Some("A".into()), Some("B".into()), Some("C".into())];
    let texts: Vec<Option<String>> = vec![
        Some("Planetary density will decrease.".into()),
        Some("Planetary years will increase.".into()),
        Some("Planetary years will decrease.".into()),
    ];
    let label_list_stem = Arc::new(Field::new("item", DataType::Utf8, true));
    let label_array = arrow_array::ListArray::new(
        label_list_stem.clone(),
        OffsetBuffer::new(vec![0i32, 3].try_into().unwrap()),
        Arc::new(StringArray::from(labels)),
        None,
    );
    let text_array = arrow_array::ListArray::new(
        label_list_stem,
        OffsetBuffer::new(vec![0i32, 3].try_into().unwrap()),
        Arc::new(StringArray::from(texts)),
        None,
    );
    let choices = arrow_array::StructArray::new(
        Fields::from(vec![
            Field::new("label", label_array.data_type().clone(), true),
            Field::new("text", text_array.data_type().clone(), true),
        ]),
        vec![Arc::new(label_array) as ArrayRef, Arc::new(text_array)],
        None,
    );
    let question = StringArray::from(vec!["A planet rotates faster after an impact."]);
    let answer_key = StringArray::from(vec!["C"]);
    let schema = Schema::new(vec![
        Field::new("question", DataType::Utf8, true),
        Field::new("choices", choices.data_type().clone(), true),
        Field::new("answerKey", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(question) as ArrayRef,
            Arc::new(choices),
            Arc::new(answer_key),
        ],
    )
    .unwrap();
    install_synthetic(
        dir.path(),
        "arc",
        &[("ARC-Challenge/test/0000.parquet", batch)],
        &[("eval", "test"), ("dev", "validation")],
        &["ARC-Challenge"],
        &["choice"],
    );
    let store = openkind_datasets::DatasetStore::new(dir.path().to_owned()).unwrap();
    let rows = materialize(&store.installed("arc").unwrap(), "eval", None)
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    let crate::workload::QuestionSpec::Choice { options, text } = &rows[0].row.question else {
        panic!("expected a choice question");
    };
    assert_eq!(text, "Which option correctly answers `question`?");
    assert_eq!(options.len(), 3);
    assert_eq!(options[2].id, "C");
    assert_eq!(rows[0].gold, "C");
    assert_eq!(rows[0].subset, "ARC-Challenge");
}

#[test]
fn eval_reports_metrics_with_provenance_for_the_mock_engine() {
    let dir = tempfile::tempdir().unwrap();
    let batch = sst2_batch(&[
        ("a charming journey . ", 1),
        ("a terrible mess . ", 0),
        ("an ordinary sentence . ", 1),
        ("another charming trip . ", 1),
    ]);
    install_synthetic(
        dir.path(),
        "sst2",
        &[("default/test/0000.parquet", batch)],
        &[("eval", "test"), ("dev", "train")],
        &["default"],
        &["choice"],
    );
    let output = dir.path().join("eval-out");
    let args = EvalArgs {
        name: "sst2".to_owned(),
        split: "eval".to_owned(),
        limit: None,
        datasets_dir: Some(dir.path().to_owned()),
        engine: crate::score::EngineKind::Mock,
        output_dir: output.clone(),
        host: Some("test-host".to_owned()),
        commit: Some("testcommit".to_owned()),
        pretty: false,
        bundle_root: None,
        checkpoint_root: None,
        tokenizer_path: None,
        model_root: None,
        adapter: None,
        tune_threshold: false,
    };
    let report = super::eval::run_eval(&args).unwrap();
    assert_eq!(report["schema"], "openkind-dataset-eval/v1");
    assert_eq!(report["dataset"]["name"], "sst2");
    assert_eq!(report["rows"].as_u64().unwrap(), 4);
    assert_eq!(report["engine"]["engine"], "mock");
    let metrics = &report["metrics"];
    assert!(metrics["accuracy"].as_f64().is_some());
    assert!(metrics["answerable_ranking_accuracy"].as_f64().is_some());
    assert!(metrics["auroc"].as_null().is_some() || metrics["auroc"].is_null());
    assert!(metrics["macro_f1"].as_f64().is_some() || metrics["macro_f1"].is_null());
    assert!(metrics["ece_10_bins"].as_f64().is_some());
    assert!(metrics["aurc"].as_f64().is_some());
    let provenance = &report["provenance"];
    for key in [
        "workload_sha256",
        "predictions_sha256",
        "summary_sha256",
        "executable_sha256",
    ] {
        let digest = provenance[key].as_str().unwrap();
        assert_eq!(digest.len(), 64, "{key}");
    }
    assert!(output.join("dataset-eval-sst2-eval.json").is_file());
    assert!(output.join("workload-sst2-eval.jsonl").is_file());
    // The written workload re-parses through the normal harness loader.
    let raw = std::fs::read(output.join("workload-sst2-eval.jsonl")).unwrap();
    let workload = crate::workload::parse_workload("sst2-eval", &raw).unwrap();
    assert_eq!(workload.rows.len(), 4);
}

#[test]
fn mock_engine_predictions_reparse_as_choices() {
    // Guard the assumption the eval join depends on: mock answers carry full
    // probability distributions including the semantic-none option.
    let dir = tempfile::tempdir().unwrap();
    let batch = sst2_batch(&[("a charming journey . ", 1), ("a terrible mess . ", 0)]);
    install_synthetic(
        dir.path(),
        "sst2",
        &[("default/test/0000.parquet", batch)],
        &[("eval", "test"), ("dev", "train")],
        &["default"],
        &["choice"],
    );
    let store = openkind_datasets::DatasetStore::new(dir.path().to_owned()).unwrap();
    let rows = materialize(&store.installed("sst2").unwrap(), "eval", None)
        .unwrap()
        .rows;
    let mut bytes = Vec::new();
    for row in &rows {
        bytes.extend_from_slice(serde_json::to_string(row).unwrap().as_bytes());
        bytes.push(b'\n');
    }
    let workload_path = dir.path().join("workload.jsonl");
    std::fs::write(&workload_path, bytes).unwrap();
    let args = crate::score::ScoreArgs {
        input: workload_path,
        engine: crate::score::EngineKind::Mock,
        output_dir: dir.path().join("out"),
        strategies: Vec::new(),
        reps: 1,
        group: true,
        warmup: true,
        history_aba: false,
        host: None,
        commit: None,
        pretty: false,
        bundle_root: None,
        checkpoint_root: None,
        tokenizer_path: None,
        model_root: None,
        adapter: None,
    };
    let outcome = crate::score::run_score(&args).unwrap();
    let strategy = outcome.summary["strategies"][0]["strategy"]
        .as_str()
        .unwrap()
        .to_owned();
    let predictions_text = std::fs::read_to_string(
        dir.path()
            .join("out")
            .join(format!("predictions-mock-{strategy}.jsonl")),
    )
    .unwrap();
    let mut saw_none = false;
    for line in predictions_text.lines() {
        let row: Map<String, Value> = serde_json::from_str(line).unwrap();
        let probabilities: BTreeMap<String, f64> =
            serde_json::from_value(row["answer"]["probabilities"].clone()).unwrap();
        assert!(probabilities.contains_key("__none__"));
        saw_none = true;
    }
    assert!(saw_none);
}
