use super::metrics::{losses, metrics, prediction, validate};
use super::*;

fn record(gold: &str, values: [f64; 3]) -> Record {
    let score = TimedScore {
        score: ScoringResult {
            probabilities: [
                ("a".into(), values[0]),
                ("b".into(), values[1]),
                (SEMANTIC_NONE_OPTION.into(), values[2]),
            ]
            .into_iter()
            .collect(),
            input_tokens: 12,
            forwards: 1,
        },
        elapsed_ms: 2.0,
    };
    Record {
        id: "q".into(),
        gold: gold.into(),
        task: "task".into(),
        source_group: "group".into(),
        methods: [
            ("independent_fitted".into(), score.clone()),
            ("joint_forward".into(), score.clone()),
            ("joint_reverse".into(), score.clone()),
            ("joint_average".into(), score),
        ]
        .into_iter()
        .collect(),
    }
}

#[test]
fn proper_scores_and_none_errors_have_known_values() {
    let first = record("a", [0.6, 0.3, 0.1]);
    let second = record(SEMANTIC_NONE_OPTION, [0.1, 0.2, 0.7]);
    let third = record("b", [0.1, 0.2, 0.7]);
    let losses = losses(&first, "joint_forward");
    assert!((losses[1] + 0.6_f64.ln()).abs() < 1e-12);
    assert!((losses[2] - 0.26).abs() < 1e-12);
    let result = metrics(&[&first, &second, &third], "joint_forward");
    assert_eq!(result["accuracy"], 2.0 / 3.0);
    assert_eq!(result["none_recall"], 1.0);
    assert_eq!(result["false_none_rate"], 0.5);
    assert_eq!(result["answerable_ranking_accuracy"], 1.0);
    assert!((result["ece_10_bins"].as_f64().unwrap() - 0.8 / 3.0).abs() < 1e-12);
    assert_eq!(result["risk_coverage"][0]["accepted"], 1);
    assert!(result["risk_coverage"][3]["error_rate"].is_null());
}

#[test]
fn validation_rejects_missing_none_bad_mass_and_nonfinite_values() {
    for values in [[0.2, 0.2, 0.2], [f64::NAN, 0.5, 0.5], [-0.1, 0.5, 0.6]] {
        assert!(validate(
            &record("a", values).methods["joint_forward"]
                .score
                .probabilities,
            "a"
        )
        .is_err());
    }
    let mut missing = record("a", [0.5, 0.5, 0.0]).methods["joint_forward"]
        .score
        .probabilities
        .clone();
    missing.remove(SEMANTIC_NONE_OPTION);
    assert!(validate(&missing, "a").is_err());
}

#[test]
fn ties_and_absent_answerable_denominators_are_explicit() {
    let record = record(SEMANTIC_NONE_OPTION, [0.25, 0.25, 0.5]);
    let p = &record.methods["joint_forward"].score.probabilities;
    assert_eq!(prediction(p, true).0, "a");
    let result = metrics(&[&record], "joint_forward");
    assert!(result["answerable_ranking_accuracy"].is_null());
    assert!(result["false_none_rate"].is_null());
}

#[test]
fn paired_bootstrap_is_grouped_and_zero_for_identical_predictions() {
    let mut first = record("a", [0.6, 0.3, 0.1]);
    let mut second = record("b", [0.6, 0.3, 0.1]);
    first.source_group = "shared".into();
    second.source_group = "shared".into();
    let report = super::report::paired(&[first, second], "joint_average");
    assert_eq!(report["source_groups"], 1);
    assert_eq!(
        report["candidate_minus_baseline"]["accuracy"]["ci_95"],
        serde_json::json!([0.0, 0.0])
    );
}

#[test]
fn labeled_panel_is_balanced_and_wire_valid() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/joint_choice_diagnostic.jsonl");
    let (rows, _) = load(&path).unwrap();
    assert_eq!(rows.len(), 96);
    let mut counts = BTreeMap::new();
    for row in rows {
        *counts.entry((row.task, row.gold)).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 16);
    assert!(counts.values().all(|count| *count == 6));
}

#[test]
fn comparison_rejects_final_rows_and_unoffered_gold_before_model_load() {
    let original: serde_json::Value = serde_json::from_str(
        include_str!("../../fixtures/joint_choice_diagnostic.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!(
        "openkind-quality-invalid-{}.jsonl",
        std::process::id()
    ));
    for (field, value) in [
        ("split", "reserved_final"),
        ("gold", "missing"),
        ("source_group", ""),
    ] {
        let mut row = original.clone();
        row[field] = serde_json::json!(value);
        fs::write(&path, serde_json::to_vec(&row).unwrap()).unwrap();
        assert!(load(&path).is_err());
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn reference_card_intervention_changes_only_the_reference_option() {
    let (rows, _) = load(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/joint_reference_card_diagnostic.jsonl"),
    )
    .unwrap();
    assert_eq!(rows.len(), 24);
    let mut groups: BTreeMap<String, Vec<LabeledRow>> = BTreeMap::new();
    for row in rows {
        groups
            .entry(row.source_group.clone())
            .or_default()
            .push(row);
    }
    assert_eq!(groups.len(), 6);
    for rows in groups.values() {
        assert_eq!(rows.len(), 4);
        let stripped = |row: &LabeledRow| {
            let mut q = serde_json::to_value(&row.row).unwrap();
            q["options"]
                .as_array_mut()
                .unwrap()
                .retain(|option| option["id"] != "reference");
            q.as_object_mut().unwrap().remove("id");
            q
        };
        for row in &rows[1..] {
            assert_eq!(stripped(&rows[0]), stripped(row));
        }
    }
}
