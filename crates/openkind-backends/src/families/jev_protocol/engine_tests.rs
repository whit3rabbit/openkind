use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use openkind_core::{
    Answer, ChoiceQuestion, NoulCriteria, NoulQuestion, Question, ScoreQuestion, State,
    SystemRequest,
};
use serde_json::json;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

use super::engine::{python_json, state_text};
use super::*;
use crate::families::support::{FamilyControl, FamilyError, FamilyEvaluator};

const JEV: &[u8] = include_bytes!("../../../tests/fixtures/jev_gev/jev_decision_config.json");

/// Records each call and returns logits that favour one option position.
struct Recorder {
    favoured: usize,
    calls: Mutex<Vec<(usize, Vec<u32>)>>,
}

impl JevForward for Recorder {
    fn option_logits(&self, input_ids: &[u32], token_ids: &[u32]) -> Result<Vec<f64>, FamilyError> {
        self.calls
            .lock()
            .unwrap()
            .push((input_ids.len(), token_ids.to_vec()));
        Ok((0..token_ids.len())
            .map(|index| if index == self.favoured { 6.0 } else { 0.0 })
            .collect())
    }
}

fn tokenizer() -> Tokenizer {
    let vocab = [("[UNK]".to_owned(), 0_u32)].into_iter().collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
}

fn engine(favoured: usize) -> (JevEngine, Arc<Recorder>) {
    let recorder = Arc::new(Recorder {
        favoured,
        calls: Mutex::new(Vec::new()),
    });
    let config = DecisionConfig::from_json(Protocol::Jev, JEV).unwrap();
    let engine = JevEngine::evaluator(
        recorder.clone(),
        tokenizer(),
        config,
        ChoiceLabels::synthetic(),
    )
    .unwrap();
    (engine, recorder)
}

fn request(questions: Vec<(&str, Question)>) -> SystemRequest {
    SystemRequest {
        state: State::Text("The parcel arrived damaged.".into()),
        model: "jev".into(),
        questions: questions
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    }
}

fn choice(options: &[(&str, Option<&str>)]) -> Question {
    Question::Choice(ChoiceQuestion {
        instructions: json!("Pick one"),
        criteria: options
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.map(str::to_owned)))
            .collect(),
    })
}

fn evaluate(
    engine: &JevEngine,
    request: SystemRequest,
) -> Result<openkind_core::SystemResponse, FamilyError> {
    let control = FamilyControl::new(Arc::new(AtomicBool::new(false)), None, 0);
    engine.evaluate(request, &control)
}

#[test]
fn choice_options_are_sorted_and_the_favoured_slot_wins() {
    let (engine, recorder) = engine(2);
    let response = evaluate(
        &engine,
        request(vec![(
            "route",
            choice(&[
                ("b", Some("second")),
                ("a", None),
                ("__none__", Some("neither")),
            ]),
        )]),
    )
    .unwrap();
    // Sorted labels are `__none__`, `a`, `b`; slot 2 is `b`.
    let Answer::Choice(answer) = &response.answers["route"] else {
        panic!("choice answer expected");
    };
    assert_eq!(answer.choice, "b");
    assert!((answer.probabilities.values().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(response.usage.output_tokens, 0);
    assert!(response.usage.input_tokens > 0);
    let calls = recorder.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].1,
        [1_000, 1_001, 1_002],
        "choice slots use the label tokens"
    );
}

#[test]
fn boolean_reads_the_true_slot_and_rejects_criteria() {
    let (engine, recorder) = engine(1);
    let noul = Question::Noul(NoulQuestion {
        instructions: json!("Refund?"),
        criteria: None,
    });
    let response = evaluate(&engine, request(vec![("refund", noul)])).unwrap();
    let Answer::Noul(answer) = &response.answers["refund"] else {
        panic!("noul answer expected");
    };
    assert!(answer.noul > 0.9);
    assert_eq!(recorder.calls.lock().unwrap()[0].1, [3721, 1802]);

    let with_criteria = Question::Noul(NoulQuestion {
        instructions: json!("Refund?"),
        criteria: Some(NoulCriteria {
            r#true: "yes".into(),
            r#false: "no".into(),
        }),
    });
    assert!(evaluate(&engine, request(vec![("refund", with_criteria)])).is_err());
}

#[test]
fn score_needs_six_levels_and_reports_the_expected_level() {
    let (engine, recorder) = engine(5);
    let six = Question::Score(ScoreQuestion {
        instructions: json!("Rate"),
        criteria: (0..6).map(|i| format!("level {i}")).collect(),
    });
    let response = evaluate(&engine, request(vec![("quality", six)])).unwrap();
    let Answer::Score(answer) = &response.answers["quality"] else {
        panic!("score answer expected");
    };
    assert!(answer.score > 4.5, "expected level {}", answer.score);
    assert_eq!(
        recorder.calls.lock().unwrap()[0].1,
        [15, 16, 17, 18, 19, 20]
    );

    let five = Question::Score(ScoreQuestion {
        instructions: json!("Rate"),
        criteria: (0..5).map(|i| format!("level {i}")).collect(),
    });
    assert!(evaluate(&engine, request(vec![("quality", five)])).is_err());
}

#[test]
fn choice_over_the_cap_is_rejected_before_any_forward() {
    let (engine, recorder) = engine(0);
    let options: Vec<(String, Option<&str>)> = (0..257)
        .map(|index| (format!("option-{index:03}"), None))
        .collect();
    let borrowed: Vec<(&str, Option<&str>)> =
        options.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert!(evaluate(&engine, request(vec![("big", choice(&borrowed))])).is_err());
    assert!(recorder.calls.lock().unwrap().is_empty());
}

#[test]
fn two_hundred_fifty_six_options_run_in_one_pass() {
    let (engine, recorder) = engine(200);
    let options: Vec<(String, Option<&str>)> = (0..256)
        .map(|index| (format!("option-{index:03}"), None))
        .collect();
    let borrowed: Vec<(&str, Option<&str>)> =
        options.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let response = evaluate(&engine, request(vec![("big", choice(&borrowed))])).unwrap();
    let Answer::Choice(answer) = &response.answers["big"] else {
        panic!("choice answer expected");
    };
    assert_eq!(answer.choice, "option-200");
    assert_eq!(recorder.calls.lock().unwrap().len(), 1);
    assert_eq!(recorder.calls.lock().unwrap()[0].1.len(), 256);
}

#[test]
fn python_json_matches_json_dumps_without_ascii_escaping() {
    let value = json!({"b": [1, 2.5, "é\n\"q\"", true, null], "a": {"z": "\u{1}"}});
    assert_eq!(
        python_json(&value),
        r#"{"a": {"z": "\u0001"}, "b": [1, 2.5, "é\n\"q\"", true, null]}"#
    );
    let State::Object(map) = State::Object(json!({"k": 1}).as_object().unwrap().clone()) else {
        unreachable!()
    };
    assert_eq!(state_text(&State::Object(map)).unwrap(), r#"{"k": 1}"#);
    assert_eq!(state_text(&State::Text("plain".into())).unwrap(), "plain");
}

#[test]
fn pins_equal_the_registry_index() {
    let index: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../registry/v1/jev-gev-mlx-models.json"
    ))
    .unwrap();
    let entry = &index["entries"][0];
    assert_eq!(entry["repository"], pins::REPOSITORY);
    assert_eq!(entry["revision"], pins::REVISION);
    assert_eq!(entry["source_model"]["revision"], pins::SOURCE_REVISION);
    let files = entry["files"].as_array().unwrap();
    let lookup = |name: &str| {
        files
            .iter()
            .find(|file| file["path"] == name)
            .unwrap_or_else(|| panic!("{name} is pinned in the index"))
    };
    for (name, size, digest) in [pins::CONFIG, pins::INDEX, pins::TOKENIZER]
        .into_iter()
        .chain(pins::SHARDS)
    {
        let file = lookup(name);
        assert_eq!(file["size"], size, "{name}");
        assert_eq!(file["sha256"], digest, "{name}");
    }
    let shard_count = files
        .iter()
        .filter(|f| f["path"].as_str().unwrap().ends_with(".safetensors"))
        .count();
    assert_eq!(shard_count, pins::SHARDS.len());
}

const CONFIG: &[u8] = include_bytes!("../../../tests/fixtures/jev_gev/jev_config.json");
const MANIFEST: &[u8] = include_bytes!("../../../tests/fixtures/jev_gev/jev_tensor_manifest.json");

#[test]
fn pinned_config_parses_to_the_27b_geometry() {
    use sha2::{Digest, Sha256};
    assert_eq!(format!("{:x}", Sha256::digest(CONFIG)), pins::CONFIG.2);
    let config = JevConfig::parse(CONFIG).unwrap();
    assert_eq!(config.geometry, crate::qwen35::Qwen35Geometry::CLEF);
    assert_eq!(
        config.quant,
        QuantParams {
            group_size: 64,
            bits: 8
        }
    );
    assert_eq!(config.vocab_size, 248_320);
    // A different geometry or quantization fails closed.
    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    value["text_config"]["num_hidden_layers"] = json!(60);
    assert!(JevConfig::parse(value.to_string().as_bytes()).is_err());
    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    value["quantization"]["bits"] = json!(4);
    assert!(JevConfig::parse(value.to_string().as_bytes()).is_err());
}

/// The loader's expected tensors must equal the real checkpoint's tensors
/// exactly (names, dtypes, shapes), apart from the unloaded vision tower.
#[test]
fn expected_tensors_equal_the_real_checkpoint_layout() {
    let config = JevConfig::parse(CONFIG).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(MANIFEST).unwrap();
    assert_eq!(manifest["revision"], pins::REVISION);
    let tensors = manifest["tensors"].as_object().unwrap();
    let group = config.quant.group_size as usize;
    let packing = 32 / config.quant.bits as usize;
    let mut expected: std::collections::BTreeMap<String, (&str, Vec<usize>)> = Default::default();
    for tensor in expected_tensors(&config.geometry, config.vocab_size) {
        match tensor.storage {
            Storage::Quantized { rows, cols } => {
                let base = tensor.name.strip_suffix(".weight").unwrap().to_owned();
                expected.insert(tensor.name, ("U32", vec![rows, cols / packing]));
                expected.insert(format!("{base}.scales"), ("BF16", vec![rows, cols / group]));
                expected.insert(format!("{base}.biases"), ("BF16", vec![rows, cols / group]));
            }
            Storage::Dense(shape) => {
                expected.insert(tensor.name, ("BF16", shape));
            }
        }
    }
    let actual: std::collections::BTreeMap<String, (String, Vec<usize>)> = tensors
        .iter()
        .filter(|(name, _)| !name.starts_with(VISION_PREFIX))
        .map(|(name, value)| {
            let dtype = value[0].as_str().unwrap().to_owned();
            let shape = value[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            (name.clone(), (dtype, shape))
        })
        .collect();
    let missing: Vec<_> = expected
        .keys()
        .filter(|name| !actual.contains_key(*name))
        .collect();
    let unexpected: Vec<_> = actual
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .collect();
    assert!(missing.is_empty(), "missing from checkpoint: {missing:?}");
    assert!(
        unexpected.is_empty(),
        "unexpected in checkpoint: {unexpected:?}"
    );
    for (name, (dtype, shape)) in &expected {
        let (actual_dtype, actual_shape) = &actual[name];
        assert_eq!(
            (actual_dtype.as_str(), actual_shape),
            (*dtype, shape),
            "{name}"
        );
    }
}
