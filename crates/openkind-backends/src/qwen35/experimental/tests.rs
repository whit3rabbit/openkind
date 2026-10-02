use super::*;
use openkind_core::ChoiceQuestion;

fn tokenizer() -> Qwen35Tokenizer {
    use tokenizers::{
        models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel, AddedToken, Tokenizer,
    };
    // These tests check renderer semantics. Pretrained token-ID parity has its own qualification.
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocab = alphabet
        .into_iter()
        .enumerate()
        .map(|(id, ch)| (ch.to_string(), id as u32))
        .collect();
    let mut inner = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    inner.with_decoder(Some(ByteLevel::new(false, true, true)));
    inner.add_tokens(
        &(b'A'..=b'Z')
            .map(|letter| AddedToken::from(format!(" {}", letter as char), false))
            .collect::<Vec<_>>(),
    );
    Qwen35Tokenizer { inner }
}

#[test]
fn state_first_renderer_rejects_invalid_cardinality_and_truncation() {
    let tokenizer = tokenizer();
    let one = [CandidateText::new("one", "only candidate")];
    assert!(matches!(
        tokenizer.encode_state_first("state", "question", &one),
        Err(Qwen35Error::InvalidInput(_))
    ));
    let two = [
        CandidateText::new("one", "first"),
        CandidateText::new("two", "second"),
    ];
    assert!(matches!(
        tokenizer.encode_state_first(&"evidence ".repeat(4_000), "question", &two),
        Err(Qwen35Error::InvalidInput(_))
    ));
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
    let forward = decode(&input.forward.ids);
    let reverse = decode(&input.reverse.ids);
    assert!(forward.contains("A. large: value 9\nB. small: value 3"));
    assert!(reverse.contains("A. small: value 3\nB. large: value 9"));
    assert!(forward.contains("Z. Neither value is given.\nAnswer:"));
    assert!(reverse.contains("Z. Neither value is given.\nAnswer:"));
    assert!(!decode(&input.independent_ids[0]).contains("value 3"));
    assert!(!decode(&input.independent_ids[1]).contains("value 9"));
    let actions = letter_ids(&tokenizer).unwrap();
    assert_eq!(decode(&[actions[0]]), " A");
    assert_eq!(decode(&[actions[MAX_CANDIDATES]]), " Z");
}

#[test]
fn catalogue_prompt_shows_every_description_and_keeps_the_frozen_root() {
    let tokenizer = tokenizer();
    let state = State::Text("Compare the supplied values.".into());
    let input = prepare(&tokenizer, &state, &choice()).unwrap();
    let decode = |ids: &[u32]| tokenizer.inner.decode(ids, false).unwrap();
    let catalogue = decode(&input.catalogue_ids[0]);
    assert!(catalogue.contains(
        "\nOptions:\nA. large: value 9\nB. small: value 3\nZ. Neither value is given.\n"
    ));
    assert!(catalogue.contains("\nCandidate:\nlarge — value 9\nMatch assessment:"));
    assert!(decode(&input.catalogue_ids[1])
        .contains("\nCandidate:\nsmall — value 3\nMatch assessment:"));
    // The catalogue lives in the question branch: the shared root matches the
    // frozen renderer's root byte for byte.
    let frozen_root_len = {
        let frozen = tokenizer
            .encode_state_first(
                "Compare the supplied values.",
                "Which option has the higher value?",
                &[
                    CandidateText::new("large", "value 9"),
                    CandidateText::new("small", "value 3"),
                ],
            )
            .unwrap();
        frozen.root_ids().len()
    };
    assert_eq!(
        decode(&input.catalogue_ids[0][..frozen_root_len]),
        decode(&input.independent_ids[0][..frozen_root_len])
    );
}

#[test]
fn text_rotation_moves_positions_while_codes_stay_bound_to_options() {
    let tokenizer = tokenizer();
    let input = prepare(
        &tokenizer,
        &State::Text("Compare the supplied values.".into()),
        &choice(),
    )
    .unwrap();
    let decode = |ids: &[u32]| tokenizer.inner.decode(ids, false).unwrap();
    let rotated = decode(&input.text_rotate.ids);
    // The full list, none included, rotates by one; codes follow their options.
    assert!(rotated.contains("Question: Which option has the higher value?\nChoose the single best option supported by the context. Choose Z if none is supported.\nB. small: value 3\nZ. Neither value is given.\nA. large: value 9\nAnswer:"));
    assert_eq!(input.text_rotate.code_of, input.forward.code_of);
    assert_ne!(input.text_rotate.ids, input.forward.ids);
}

#[test]
fn code_rotation_permutes_letters_while_positions_stay_fixed() {
    let tokenizer = tokenizer();
    let input = prepare(
        &tokenizer,
        &State::Text("Compare the supplied values.".into()),
        &choice(),
    )
    .unwrap();
    let decode = |ids: &[u32]| tokenizer.inner.decode(ids, false).unwrap();
    let rotated = decode(&input.code_rotate.ids);
    // Two real options plus none: codes shift to B, C, and A.
    assert!(rotated
        .contains("B. large: value 9\nC. small: value 3\nA. Neither value is given.\nAnswer:"));
    assert!(rotated.contains("Choose A if none is supported."));
    assert_eq!(input.code_rotate.code_of, vec![1, 2, 0]);
    // Display order is unchanged from the forward render.
    let options = |ids: &[u32]| {
        decode(ids)
            .lines()
            .filter(|line| {
                line.len() >= 3
                    && line.as_bytes()[1] == b'.'
                    && line.as_bytes()[2] == b' '
                    && (line.as_bytes()[0] as char).is_ascii_uppercase()
            })
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let stripped = |lines: &[String]| {
        lines
            .iter()
            .map(|line| line.split_once(". ").unwrap().1.to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        stripped(&options(&input.code_rotate.ids)),
        stripped(&options(&input.forward.ids))
    );
}

#[test]
fn calibration_math_preserves_argmax_and_moves_none_mass() {
    let logits = vec![1.5, -0.5, 0.25];
    let probabilities = Qwen35ScoringProbe::calibrated_probabilities(&logits, 1.0, 0.0).unwrap();
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let argmax = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .reduce(|best, (index, value)| {
                if *value > *best.1 {
                    (index, value)
                } else {
                    best
                }
            })
            .unwrap()
            .0
    };
    // The temperature arm preserves the winning class.
    let hottest = Qwen35ScoringProbe::calibrated_probabilities(&logits, 4.0, 0.0).unwrap();
    assert_eq!(argmax(&probabilities), argmax(&hottest));
    // Higher temperature flattens, so the winner keeps less mass.
    assert!(hottest[argmax(&hottest)] < probabilities[argmax(&probabilities)]);
    // The none offset moves rejection mass without inventing evidence.
    let shifted = Qwen35ScoringProbe::calibrated_probabilities(&logits, 1.0, -2.0).unwrap();
    assert!(shifted[2] < probabilities[2]);
    assert!((shifted.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(Qwen35ScoringProbe::calibrated_probabilities(&logits, 0.0, 0.0).is_err());
    assert!(Qwen35ScoringProbe::calibrated_probabilities(&logits, 1.0, f64::NAN).is_err());
    assert!(Qwen35ScoringProbe::calibrated_probabilities(&[1.0], 1.0, 0.0).is_err());
}

#[test]
fn ensemble_average_preserves_none_and_rejects_mismatched_keys() {
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
        // Each independent prompt fits at ~170 tokens, but the catalogue shows
        // every rival in every prompt and exceeds the bound instead.
        for index in 0..16 {
            choice.criteria.remove(&format!("opt{index}"));
        }
        choice
            .criteria
            .insert("wide".into(), Some("word ".repeat(1200)));
    }
    let error = prepare(&tokenizer, &State::Text("state".into()), &question)
        .err()
        .unwrap();
    assert!(error.to_string().contains("exceeds"));
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
