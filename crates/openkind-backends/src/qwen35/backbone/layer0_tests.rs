use super::layer0::*;

#[test]
fn causal_depthwise_convolution_uses_left_padding_and_silu() {
    let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0];
    let weight = [1.0_f32, 2.0, 1.0, -1.0];
    let output = causal_depthwise_conv_silu(&input, 3, 2, &weight, 2);
    let expected_raw = [2.0_f32, -10.0, 5.0, -10.0, 8.0, -10.0];
    for (actual, expected) in output.iter().zip(expected_raw) {
        assert!((actual - silu(expected)).abs() < 1e-6);
    }
}

#[test]
fn zero_centered_rms_norm_applies_one_plus_weight() {
    let output = rms_norm_zero_centered(&[3.0, 4.0], 1, 2, &[0.0, 1.0]);
    let scale = ((25.0_f32 / 2.0) + RMS_EPSILON).sqrt().recip();
    assert!((output[0] - 3.0 * scale).abs() < 1e-6);
    assert!((output[1] - 8.0 * scale).abs() < 1e-6);
}

#[test]
fn activations_are_finite_at_extremes() {
    for value in [-100.0_f32, -1.0, 0.0, 1.0, 100.0] {
        assert!(sigmoid(value).is_finite());
        assert!(silu(value).is_finite());
        assert!(softplus(value).is_finite());
    }
}

#[test]
fn cached_convolution_matches_one_pass_execution() {
    let channels = 2;
    let kernel = 4;
    let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0];
    let weight = [1.0_f32, 0.5, -0.25, 2.0, -1.0, 0.25, 0.5, 1.0];
    let (full, _) =
        causal_depthwise_conv_silu_with_state(&input, 5, channels, &weight, kernel, None);
    let (prefix, state) =
        causal_depthwise_conv_silu_with_state(&input[..6], 3, channels, &weight, kernel, None);
    let (suffix, _) = causal_depthwise_conv_silu_with_state(
        &input[6..],
        2,
        channels,
        &weight,
        kernel,
        Some(&state),
    );
    assert_eq!([prefix, suffix].concat(), full);
}

#[test]
fn rotary_positions_are_continuation_stable() {
    let mut full = vec![0.25_f32; 3 * ATTENTION_HEAD_DIM];
    let mut prefix = full[..2 * ATTENTION_HEAD_DIM].to_vec();
    let mut suffix = full[2 * ATTENTION_HEAD_DIM..].to_vec();
    apply_rotary(&mut full, 1, 3, 0, PINNED);
    apply_rotary(&mut prefix, 1, 2, 0, PINNED);
    apply_rotary(&mut suffix, 1, 1, 2, PINNED);
    assert_eq!([prefix, suffix].concat(), full);
}

#[test]
fn cached_causal_attention_matches_one_pass_execution() {
    let queries = vec![0.125_f32; 2 * ATTENTION_SIZE];
    let keys = vec![0.25_f32; 2 * KV_SIZE];
    let values: Vec<_> = (0..2 * KV_SIZE)
        .map(|index| index as f32 / 10_000.0)
        .collect();
    let full = causal_grouped_query_attention(&queries, &keys, &values, 2, 0, PINNED);
    let prefix = causal_grouped_query_attention(
        &queries[..ATTENTION_SIZE],
        &keys[..KV_SIZE],
        &values[..KV_SIZE],
        1,
        0,
        PINNED,
    );
    let suffix =
        causal_grouped_query_attention(&queries[ATTENTION_SIZE..], &keys, &values, 1, 1, PINNED);
    assert_eq!([prefix, suffix].concat(), full);
}

#[test]
fn cached_delta_recurrence_matches_one_pass_execution() {
    let rows = 3;
    let qkv = vec![0.01_f32; rows * QKV_SIZE];
    let z = vec![0.2_f32; rows * VALUE_SIZE];
    let beta = vec![0.1_f32; rows * VALUE_HEADS];
    let decay = vec![0.05_f32; rows * VALUE_HEADS];
    let dt_bias = vec![0.0_f32; VALUE_HEADS];
    let a_log = vec![0.0_f32; VALUE_HEADS];
    let norm = vec![1.0_f32; HEAD_DIM];
    let (full, full_state) = gated_delta_recurrent_with_state(
        &qkv, &z, &beta, &decay, &dt_bias, &a_log, &norm, rows, PINNED, None,
    );
    let (prefix, prefix_state) = gated_delta_recurrent_with_state(
        &qkv[..2 * QKV_SIZE],
        &z[..2 * VALUE_SIZE],
        &beta[..2 * VALUE_HEADS],
        &decay[..2 * VALUE_HEADS],
        &dt_bias,
        &a_log,
        &norm,
        2,
        PINNED,
        None,
    );
    let (suffix, suffix_state) = gated_delta_recurrent_with_state(
        &qkv[2 * QKV_SIZE..],
        &z[2 * VALUE_SIZE..],
        &beta[2 * VALUE_HEADS..],
        &decay[2 * VALUE_HEADS..],
        &dt_bias,
        &a_log,
        &norm,
        1,
        PINNED,
        Some(&prefix_state),
    );
    assert_eq!([prefix, suffix].concat(), full);
    assert_eq!(suffix_state, full_state);
}
