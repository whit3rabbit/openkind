//! Small offline reference records. Real checkpoint replay is an explicit example.
use openkind_backends::families::jeeves::{PROFILE_ID, TEMPERATURE};
use openkind_runtime::hashing::sha256_hex;
use serde_json::Value;

const GOLDEN: &[u8] = include_bytes!("fixtures/jeeves_15fb3b95801f2f039636/golden.json");

#[test]
fn frozen_reference_identity_and_readout_positions_are_consistent() {
    assert_eq!(
        sha256_hex(GOLDEN),
        "6fc3351114fd130faf13120727aac448088d4c1facbd45947f3ceded52b35ed5"
    );
    let identity = "cowWhySo/jeeves-mlx@6f4220a267a94c5279b3fcbc5499de734b9a55b0/markers-v3-plainchains/lexicographic-options/no-thinking/affine4-g64/fp32-reference";
    assert_eq!(&sha256_hex(identity.as_bytes())[..20], PROFILE_ID);
    assert_eq!(TEMPERATURE, 1.8589280843734741);
    let cases: Vec<Value> = serde_json::from_slice(GOLDEN).unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let positions: Vec<usize> =
            serde_json::from_value(case["option_positions"].clone()).unwrap();
        let decide = case["decide_position"].as_u64().unwrap() as usize;
        assert_eq!(decide, ids.len() - 1);
        assert_eq!(ids[decide], 248062);
        assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
        for position in &positions {
            assert!(*position < decide);
            assert_eq!(ids[*position], 248050);
        }
        let probabilities: Vec<f64> =
            serde_json::from_value(case["probabilities"].clone()).unwrap();
        assert_eq!(positions.len(), probabilities.len());
        assert_eq!(positions.len(), case["options"].as_array().unwrap().len());
        assert!(probabilities.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
