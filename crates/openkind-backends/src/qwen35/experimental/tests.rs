use super::*;
use openkind_core::ChoiceQuestion;

fn tokenizer() -> Qwen35Tokenizer {
    Qwen35Tokenizer::from_file(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/14_phase3b_backbone_parity_results/backbone_runtime/tokenizer/tokenizer.json"))
        .expect("vendored digest-locked tokenizer")
}

fn choice() -> Question {
    Question::Choice(ChoiceQuestion {
        instructions: serde_json::json!("Which option has the higher value?"),
        criteria: [
            ("small".into(), Some("value 3".into())),
            ("large".into(), Some("value 9".into())),
            (
                SEMANTIC_NONE_OPTION.into(),
                Some("Neither value is given.".into()),
            ),
        ]
        .into_iter()
        .collect(),
    })
}

#[test]
fn joint_prompt_contains_rival_evidence_and_reversal_changes_letter_assignment() {
    let tokenizer = tokenizer();
    let input = prepare(
        &tokenizer,
        &State::Text("Compare the supplied values.".into()),
        &choice(),
    )
    .unwrap();
    assert_eq!(input.labels, ["large", "small"]);
    let decode = |ids: &[u32]| tokenizer.inner.decode(ids, false).unwrap();
    let forward = decode(&input.forward_ids);
    let reverse = decode(&input.reverse_ids);
    assert!(forward.contains("A. large: value 9\nB. small: value 3"));
    assert!(reverse.contains("A. small: value 3\nB. large: value 9"));
    assert!(forward.contains("Z. Neither value is given.\nAnswer:"));
    assert!(!decode(&input.independent_ids[0]).contains("value 3"));
    assert!(!decode(&input.independent_ids[1]).contains("value 9"));
    let actions = letter_ids(&tokenizer).unwrap();
    assert_eq!(decode(&[actions[0]]), " A");
    assert_eq!(decode(&[actions[MAX_CANDIDATES]]), " Z");
}

#[test]
fn joint_rejects_long_option_sets_instead_of_truncating_rivals() {
    let tokenizer = tokenizer();
    let mut question = choice();
    {
        let Question::Choice(ref mut choice) = question else {
            unreachable!()
        };
        for index in 0..16 {
            choice
                .criteria
                .insert(format!("opt{index}"), Some("word ".repeat(120)));
        }
    }
    assert!(prepare(&tokenizer, &State::Text("state".into()), &question).is_err());
    {
        let Question::Choice(ref mut choice) = question else {
            unreachable!()
        };
        // Each independent prompt fits, but the combined options exceed the bound.
        choice.criteria.remove("large");
        choice.criteria.remove("small");
    }
    let error = prepare(&tokenizer, &State::Text("state".into()), &question)
        .err()
        .unwrap();
    assert!(error.to_string().contains("joint prompt length"));
}

#[test]
fn order_average_preserves_none_and_rejects_mismatched_keys() {
    let score = |a, b, none| ScoringResult {
        probabilities: [
            ("a".into(), a),
            ("b".into(), b),
            (SEMANTIC_NONE_OPTION.into(), none),
        ]
        .into_iter()
        .collect(),
        input_tokens: 10,
        forwards: 1,
    };
    let first = score(0.6, 0.1, 0.3);
    let second = score(0.2, 0.4, 0.4);
    let average = average_orders(&first, &second).unwrap();
    assert!((average.probabilities[SEMANTIC_NONE_OPTION] - 0.35).abs() < 1e-12);
    assert!((average.probabilities.values().sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(average.forwards, 2);
    let mut mismatched = second;
    mismatched.probabilities.remove("a");
    assert!(average_orders(&first, &mismatched).is_err());
}

#[test]
fn projection_checks_width_and_nonfinite_values() {
    let hidden = vec![1.0; FEATURE_WIDTH];
    let row = vec![0.5; FEATURE_WIDTH];
    assert_eq!(project(&hidden, &row).unwrap(), FEATURE_WIDTH as f64 / 2.0);
    assert!(project(&hidden[..1], &row).is_err());
    let mut bad = hidden;
    bad[0] = f32::NAN;
    assert!(project(&bad, &row).is_err());
}
