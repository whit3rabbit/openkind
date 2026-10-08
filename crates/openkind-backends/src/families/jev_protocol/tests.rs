use super::*;
use crate::families::support::FamilyError;

// Fixtures are the `decision_config` objects from the pinned checkpoints'
// root config.json (nativ-community/JEV-27B-VL-MLX-8bit @ a871d5f8 and
// nativ-community/GEV-26B-Decide-MLX-8bit @ 8f04bf3f). Expected values come
// from a line-for-line Python port of the mlx-vlm `feat/jev` and `feat/gev`
// readout (mlx_vlm/models/jev/jev.py, gev/gev.py), run in f64.
const JEV: &[u8] = include_bytes!("../../../tests/fixtures/jev_gev/jev_decision_config.json");
const GEV: &[u8] = include_bytes!("../../../tests/fixtures/jev_gev/gev_decision_config.json");
const TOLERANCE: f64 = 1e-10;

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < TOLERANCE, "{a} vs {e}");
    }
}

fn jev() -> DecisionConfig {
    DecisionConfig::from_json(Protocol::Jev, JEV).expect("pinned jev config validates")
}

fn gev() -> DecisionConfig {
    DecisionConfig::from_json(Protocol::Gev, GEV).expect("pinned gev config validates")
}

/// Deterministic stand-in for one forward pass over the given options.
fn mock_pass(options: &[usize]) -> Result<Vec<f64>, FamilyError> {
    let logits: Vec<f64> = options
        .iter()
        .enumerate()
        .map(|(position, &option)| 3.0 * (1.7 * option as f64 + 0.3).sin() + 0.25 * position as f64)
        .collect();
    crate::families::temperature_softmax(&logits, 1.0)
}

#[test]
fn pinned_configs_validate_and_reject_the_other_protocol() {
    assert_eq!(jev().lora_scale(), 2.0);
    assert_eq!(jev().slot_width(), 24);
    assert_eq!(gev().softcap(), Some(30.0));
    assert!(DecisionConfig::from_json(Protocol::Gev, JEV).is_err());
    assert!(DecisionConfig::from_json(Protocol::Jev, GEV).is_err());
}

#[test]
fn malformed_configs_fail_closed() {
    let mut value: serde_json::Value = serde_json::from_slice(JEV).unwrap();
    value["ranges"]["choice"] = serde_json::json!([8, 23]);
    assert!(DecisionConfig::from_json(Protocol::Jev, value.to_string().as_bytes()).is_err());
    let mut value: serde_json::Value = serde_json::from_slice(JEV).unwrap();
    value["temperature_by_type"]["score"] = serde_json::json!(0.0);
    assert!(DecisionConfig::from_json(Protocol::Jev, value.to_string().as_bytes()).is_err());
    let mut value: serde_json::Value = serde_json::from_slice(JEV).unwrap();
    value["bias"].as_array_mut().unwrap().pop();
    assert!(DecisionConfig::from_json(Protocol::Jev, value.to_string().as_bytes()).is_err());
}

#[test]
fn jev_token_ids_follow_the_pinned_verbalizers() {
    let config = jev();
    assert_eq!(
        config.jev_token_ids(DecisionKind::Noul, 2, &[]).unwrap(),
        [3721, 1802]
    );
    assert_eq!(
        config.jev_token_ids(DecisionKind::Score, 6, &[]).unwrap(),
        [15, 16, 17, 18, 19, 20]
    );
    assert_eq!(
        config
            .jev_token_ids(DecisionKind::Choice, 3, &[32, 33, 34, 35])
            .unwrap(),
        [32, 33, 34]
    );
    assert!(config.jev_token_ids(DecisionKind::Score, 5, &[]).is_err());
    assert!(config
        .jev_token_ids(DecisionKind::Choice, 257, &[0; 300])
        .is_err());
}

#[test]
fn jev_probabilities_match_the_reference() {
    let config = jev();
    close(
        &config
            .jev_probabilities(DecisionKind::Noul, &[1.25, -0.5])
            .unwrap(),
        &[0.8488411674637608, 0.15115883253623913],
    );
    close(
        &config
            .jev_probabilities(DecisionKind::Score, &[0.1, 0.9, 1.7, 2.4, 1.1, -0.3])
            .unwrap(),
        &[
            0.046579769848460836,
            0.10346190839443088,
            0.22980379130949594,
            0.462384174090427,
            0.12650980676132648,
            0.031260549595858896,
        ],
    );
    // 20 options: the last four slots lie beyond the choice range and take zero bias.
    let gathered: Vec<f64> = (0..20)
        .map(|i| 2.0 - 0.35 * i as f64 + if i % 7 == 3 { 0.6 } else { 0.0 })
        .collect();
    close(
        &config
            .jev_probabilities(DecisionKind::Choice, &gathered)
            .unwrap(),
        &[
            0.26728723453317765,
            0.18941617027664107,
            0.1342217449338726,
            0.17169425972680577,
            0.06739964902194968,
            0.04776458758227444,
            0.0337707638502389,
            0.023820959619859704,
            0.016887255546066588,
            0.011986645116330382,
            0.01536196122039557,
            0.0060319784277146704,
            0.004273293117482377,
            0.0030280504700066395,
            0.0021449951299881906,
            0.0015211770746485034,
            0.00108045604409511,
            0.00138183896412849,
            0.0005425330205032943,
            0.0003844463238202991,
        ],
    );
}

#[test]
fn gev_softcap_and_score_match_the_reference() {
    let config = gev();
    close(
        &config
            .gev_softcap(&[40.0, -35.0, 3.0, -2.0, 0.5, 12.0, -7.5, 31.0])
            .unwrap(),
        &[
            26.101849852280157,
            -24.696019367576493,
            2.9900398387486744,
            -1.997042295067877,
            0.49995370884715834,
            11.398468867656748,
            -7.347559872111274,
            23.257253439518596,
        ],
    );
    assert_eq!(config.gev_window(DecisionKind::Score, 6).unwrap(), 2..8);
    assert_eq!(config.gev_window(DecisionKind::Choice, 16).unwrap(), 8..24);
    assert!(config.gev_window(DecisionKind::Choice, 17).is_err());
    close(
        &config
            .gev_probabilities(DecisionKind::Score, &[0.1, 0.9, 1.7, 2.4, 1.1, -0.3])
            .unwrap(),
        &[
            0.04636866848739882,
            0.10325398533028071,
            0.22992649636862378,
            0.4632452182263863,
            0.12613260720512243,
            0.031073024382187925,
        ],
    );
}

#[test]
fn tournament_matches_the_reference() {
    let single = gev_choice_tournament(5, mock_pass).unwrap();
    close(&single, &mock_pass(&[0, 1, 2, 3, 4]).unwrap());
    close(
        &gev_choice_tournament(17, mock_pass).unwrap(),
        &[
            0.0022375862772593623,
            0.01811480193599218,
            0.0003101586579413349,
            0.00019215434444758202,
            0.022325828355994653,
            0.018607945906760598,
            0.00029515631848797024,
            0.0018114891890817083,
            0.12582091850841218,
            0.01208660805068374,
            0.0004372361250150279,
            0.017610031990164525,
            0.2578286671774477,
            0.005619091863613426,
            0.0018088948926742883,
            0.19545734597571934,
            0.3194360844303044,
        ],
    );
    close(
        &gev_choice_tournament(40, mock_pass).unwrap(),
        &[
            0.00025979551887966526,
            0.006678644645427508,
            3.601104918000433e-05,
            2.2310128609604037e-05,
            0.004992465667513081,
            0.004161078800800626,
            3.426919877525948e-05,
            0.00021032340902573228,
            0.017065268108959582,
            0.0014033186750986469,
            6.497554332393435e-05,
            0.0018601449561134991,
            0.027234402246370616,
            0.0008377059855834822,
            3.571940091781078e-05,
            0.0038596047407244713,
            0.015938584761343032,
            9.04457800867103e-05,
            0.00029055474433907735,
            0.04042598822838424,
            0.006924527284837179,
            0.00013789305648935816,
            0.003478927941418796,
            0.08109917145540997,
            0.0026507979402403915,
            0.0004573180730367766,
            0.023468961050428606,
            0.07480458196058194,
            0.0002763356738807807,
            0.0004552408258684855,
            0.11556933319039597,
            0.015692186014497164,
            0.000343167342828104,
            0.0050582736972329114,
            0.22598181484632474,
            0.00936359289620728,
            0.0009054000047632364,
            0.04255018559486638,
            0.25919269943248896,
            0.006087980128746381,
        ],
    );
    let full = gev_choice_tournament(256, mock_pass).unwrap();
    let expected = serde_json::json!({"argmax": 252, "p0": 1.6880286127434254e-05, "p100": 0.0006263867130352357, "p255": 0.03630418696911386, "weighted": 199.91529319154375, "sum": 1.0});
    let argmax = (0..256)
        .max_by(|&a, &b| full[a].total_cmp(&full[b]))
        .unwrap();
    assert_eq!(argmax, expected["argmax"].as_u64().unwrap() as usize);
    for (index, key) in [(0, "p0"), (100, "p100"), (255, "p255")] {
        assert!((full[index] - expected[key].as_f64().unwrap()).abs() < TOLERANCE);
    }
    let weighted: f64 = full.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
    assert!((weighted - expected["weighted"].as_f64().unwrap()).abs() < 1e-8);
    assert!((full.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn tournament_passes_stay_within_sixteen_options() {
    let mut widest = 0;
    gev_choice_tournament(256, |options| {
        widest = widest.max(options.len());
        mock_pass(options)
    })
    .unwrap();
    assert_eq!(widest, GEV_MAX_GROUP);
    assert!(gev_choice_tournament(0, mock_pass).is_err());
    assert!(gev_choice_tournament(257, mock_pass).is_err());
    assert!(gev_choice_tournament(20, |_| Ok(vec![1.0])).is_err());
}

#[test]
fn prompts_match_the_reference_format() {
    let options = vec!["refund: wants money back".to_owned(), "keep".to_owned()];
    let labels = ChoiceLabels::gev();
    assert_eq!(
        render_prompt(
            Protocol::Gev,
            DecisionKind::Choice,
            "S",
            "Q",
            &options,
            &labels
        )
        .unwrap(),
        "<bos>[kind] choice
[state] S
[question] Q
[options]
A) refund: wants money back
B) keep
[decision]:"
    );
    let levels: Vec<String> = (0..6).map(|i| i.to_string()).collect();
    assert_eq!(
        render_prompt(
            Protocol::Jev,
            DecisionKind::Score,
            "S",
            "Q",
            &levels,
            &labels
        )
        .unwrap(),
        "[kind] score
[state] S
[question] Q
[options]
0
1
2
3
4
5
[decision]:"
    );
    assert_eq!(choice_option_text("keep", "keep"), "keep");
    assert_eq!(choice_option_text("keep", ""), "keep");
    assert_eq!(choice_option_text("keep", "stay"), "keep: stay");
}

#[test]
fn choice_labels_use_single_token_letters_seen_in_the_prompt_form() {
    // Toy tokenizer: a bare label is one token (`Q` gets a private id), and the
    // prompt form drops `Q`'s token, so `Q` never qualifies.
    let labels = ChoiceLabels::derive(|text| {
        if text.starts_with("x\n") {
            return Ok(text
                .chars()
                .map(|c| c as u32)
                .filter(|&c| c != 'Q' as u32 && c != 999)
                .collect());
        }
        if text == "Q" {
            return Ok(vec![999]);
        }
        Ok(vec![text.chars().fold(0, |acc, c| acc * 100 + c as u32)])
    })
    .unwrap();
    assert_eq!(labels.names()[0], "A");
    assert_eq!(labels.token_ids()[0], 'A' as u32);
    assert!(!labels.names().contains(&"Q".to_owned()));
    // Two-letter labels encode to one token alone but split in the prompt form,
    // so only the 25 single letters other than `Q` qualify in this toy tokenizer.
    assert_eq!(labels.names().len(), 25);
}

#[test]
fn vendored_fixtures_equal_the_pinned_registry_index() {
    let index: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../registry/v1/jev-gev-mlx-models.json"
    ))
    .unwrap();
    for (entry, fixture) in index["entries"].as_array().unwrap().iter().zip([JEV, GEV]) {
        let vendored: serde_json::Value = serde_json::from_slice(fixture).unwrap();
        assert_eq!(entry["decision_config"], vendored, "{}", entry["name"]);
    }
}
