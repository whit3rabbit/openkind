use super::*;

#[test]
fn generic_masked_vector_gate_matches_explicit_tree_host() {
    const HEADS: usize = 2;
    const DK: usize = 8;
    const DV: usize = 8;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let query = seeded(HEADS * DK, 1);
    let key = seeded(HEADS * DK, 2);
    let value = seeded(HEADS * DV, 3);
    let beta = vec![0.25, 0.75];
    let gate = seeded(HEADS * DV, 4)
        .into_iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let state = seeded(HEADS * DK * DV, 5);
    let mask = vec![true, false];
    let expected = host_step(
        HEADS,
        DK,
        DV,
        &query,
        &key,
        &value,
        &beta,
        &gate,
        &state,
        Some(&mask),
        GateLayout::Vector,
    );
    let actual = runtime
        .execute(|| {
            let (output, next) = generic_gated_delta(
                &Array::from_slice(&query, &[HEADS as i32, DK as i32]),
                &Array::from_slice(&key, &[HEADS as i32, DK as i32]),
                &Array::from_slice(&value, &[HEADS as i32, DV as i32]),
                &Array::from_slice(&beta, &[HEADS as i32]),
                &Array::from_slice(&gate, &[HEADS as i32, DV as i32]),
                &Array::from_slice(&state, &[HEADS as i32, DK as i32, DV as i32]),
                Some(&Array::from_slice(&mask, &[HEADS as i32])),
                GateLayout::Vector,
            )
            .expect("generic kernel");
            output.eval().expect("output eval");
            next.eval().expect("state eval");
            (
                output.to_vec_cast::<f32>().expect("output read"),
                next.to_vec_cast::<f32>().expect("state read"),
            )
        })
        .expect("execute");
    assert_close(&actual.0, &expected.0, 1e-6);
    assert_close(&actual.1, &expected.1, 1e-6);
}

#[test]
fn packed_128_matches_generic_scalar_kernel() {
    const HEADS: usize = 1;
    const D: usize = 128;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let query = seeded(HEADS * D, 11);
    let key = seeded(HEADS * D, 12);
    let value = seeded(HEADS * D, 13);
    let beta = vec![0.4];
    let gate = vec![0.9];
    let state = seeded(HEADS * D * D, 14);
    let (packed, generic) = runtime
        .execute(|| {
            let query = Array::from_slice(&query, &[HEADS as i32, D as i32]);
            let key = Array::from_slice(&key, &[HEADS as i32, D as i32]);
            let value = Array::from_slice(&value, &[HEADS as i32, D as i32]);
            let beta = Array::from_slice(&beta, &[HEADS as i32]);
            let gate = Array::from_slice(&gate, &[HEADS as i32]);
            let state = Array::from_slice(&state, &[HEADS as i32, D as i32, D as i32]);
            let packed = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
                .expect("packed kernel");
            let generic = generic_gated_delta(
                &query,
                &key,
                &value,
                &beta,
                &gate,
                &state,
                None,
                GateLayout::Scalar,
            )
            .expect("generic kernel");
            for array in [&packed.0, &packed.1, &generic.0, &generic.1] {
                array.eval().expect("eval");
            }
            (
                (
                    packed.0.to_vec_cast::<f32>().expect("read"),
                    packed.1.to_vec_cast::<f32>().expect("read"),
                ),
                (
                    generic.0.to_vec_cast::<f32>().expect("read"),
                    generic.1.to_vec_cast::<f32>().expect("read"),
                ),
            )
        })
        .expect("execute");
    // `float4` lowering may fuse the multiply-add where the scalar generic
    // kernel does not; the reduction tree and resulting state still agree
    // within two FP32 ulps at the observed magnitude.
    assert_close(&packed.0, &generic.0, 2e-6);
    assert_close(&packed.1, &generic.1, 2e-6);
}

#[test]
fn packed_128_preserves_shared_input_and_replays_exactly() {
    const HEADS: usize = 32;
    const D: usize = 128;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let query = seeded(HEADS * D, 31);
    let key = seeded(HEADS * D, 32);
    let value = seeded(HEADS * D, 33);
    let beta = seeded(HEADS, 34)
        .into_iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let gate = seeded(HEADS, 35)
        .into_iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let state_values = seeded(HEADS * D * D, 36);
    runtime
        .execute(|| {
            let query = Array::from_slice(&query, &[HEADS as i32, D as i32]);
            let key = Array::from_slice(&key, &[HEADS as i32, D as i32]);
            let value = Array::from_slice(&value, &[HEADS as i32, D as i32]);
            let beta = Array::from_slice(&beta, &[HEADS as i32]);
            let gate = Array::from_slice(&gate, &[HEADS as i32]);
            let state = Array::from_slice(&state_values, &[HEADS as i32, D as i32, D as i32]);
            state.eval().expect("input eval");
            let input_before = state.to_vec_cast::<f32>().expect("input read");
            let first = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
                .expect("first packed kernel");
            for array in [&first.0, &first.1] {
                array.eval().expect("first eval");
            }
            let first_output = first.0.to_vec_cast::<f32>().expect("first output read");
            let first_state = first.1.to_vec_cast::<f32>().expect("first state read");
            let second = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
                .expect("second packed kernel");
            for array in [&second.0, &second.1] {
                array.eval().expect("second eval");
            }
            let second_output = second.0.to_vec_cast::<f32>().expect("second output read");
            let second_state = second.1.to_vec_cast::<f32>().expect("second state read");
            let input_after = state.to_vec_cast::<f32>().expect("input reread");
            assert_eq!(
                input_before, input_after,
                "kernel mutated shared input state"
            );
            assert_close(&first_output, &second_output, 0.0);
            assert_close(&first_state, &second_state, 0.0);

            let replay_chain = || {
                let mut current = state.clone();
                let mut outputs = Vec::new();
                for step in 0..8_u32 {
                    let query = Array::from_slice(
                        &seeded(HEADS * D, 40 + step * 5),
                        &[HEADS as i32, D as i32],
                    );
                    let key = Array::from_slice(
                        &seeded(HEADS * D, 41 + step * 5),
                        &[HEADS as i32, D as i32],
                    );
                    let value = Array::from_slice(
                        &seeded(HEADS * D, 42 + step * 5),
                        &[HEADS as i32, D as i32],
                    );
                    let beta = Array::from_slice(
                        &seeded(HEADS, 43 + step * 5)
                            .into_iter()
                            .map(|value| value.abs())
                            .collect::<Vec<_>>(),
                        &[HEADS as i32],
                    );
                    let gate = Array::from_slice(
                        &seeded(HEADS, 44 + step * 5)
                            .into_iter()
                            .map(|value| value.abs())
                            .collect::<Vec<_>>(),
                        &[HEADS as i32],
                    );
                    let next = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &current)
                        .expect("packed chain step");
                    outputs.push(next.0);
                    current = next.1;
                }
                for output in &outputs {
                    output.eval().expect("chain output eval");
                }
                current.eval().expect("chain state eval");
                (
                    outputs
                        .last()
                        .expect("non-empty chain")
                        .to_vec_cast::<f32>()
                        .expect("chain output read"),
                    current.to_vec_cast::<f32>().expect("chain state read"),
                )
            };
            let chain_first = replay_chain();
            let chain_second = replay_chain();
            assert_close(&chain_first.0, &chain_second.0, 0.0);
            assert_close(&chain_first.1, &chain_second.1, 0.0);
        })
        .expect("execute");
}

#[test]
fn generic_128_replays_exactly_across_all_heads() {
    const HEADS: usize = 32;
    const D: usize = 128;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| {
            let query = Array::from_slice(&seeded(HEADS * D, 51), &[32, 128]);
            let key = Array::from_slice(&seeded(HEADS * D, 52), &[32, 128]);
            let value = Array::from_slice(&seeded(HEADS * D, 53), &[32, 128]);
            let beta = Array::from_slice(&seeded(HEADS, 54), &[32]);
            let gate = Array::from_slice(&seeded(HEADS, 55), &[32]);
            let state = Array::from_slice(
                &seeded(HEADS * D * D, 56),
                &[HEADS as i32, D as i32, D as i32],
            );
            let run = || {
                let result = generic_gated_delta(
                    &query,
                    &key,
                    &value,
                    &beta,
                    &gate,
                    &state,
                    None,
                    GateLayout::Scalar,
                )
                .expect("generic kernel");
                result.0.eval().expect("output eval");
                result.1.eval().expect("state eval");
                (
                    result.0.to_vec_cast::<f32>().expect("output read"),
                    result.1.to_vec_cast::<f32>().expect("state read"),
                )
            };
            let first = run();
            let second = run();
            assert_close(&first.0, &second.0, 0.0);
            assert_close(&first.1, &second.1, 0.0);
        })
        .expect("execute");
}

#[test]
fn generic_bf16_scalar_kernel_is_finite() {
    const HEADS: usize = 2;
    const D: usize = 8;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime.qualify_bf16().expect("BF16 preflight");
    let bf16 = |values: Vec<f32>| {
        values
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>()
    };
    let output = runtime
        .execute(|| {
            let query = Array::from_slice(&bf16(seeded(HEADS * D, 21)), &[2, 8]);
            let key = Array::from_slice(&bf16(seeded(HEADS * D, 22)), &[2, 8]);
            let value = Array::from_slice(&bf16(seeded(HEADS * D, 23)), &[2, 8]);
            let beta = Array::from_slice(&bf16(vec![0.3, 0.7]), &[2]);
            let gate = Array::from_slice(&bf16(vec![0.8, 0.9]), &[2]);
            let state = Array::from_slice(&bf16(seeded(HEADS * D * D, 24)), &[2, 8, 8]);
            let (output, next) = generic_gated_delta(
                &query,
                &key,
                &value,
                &beta,
                &gate,
                &state,
                None,
                GateLayout::Scalar,
            )
            .expect("generic BF16 kernel");
            output.eval().expect("output eval");
            next.eval().expect("state eval");
            output.to_vec_cast::<f32>().expect("output read")
        })
        .expect("execute");
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn custom_kernels_require_the_explicit_runtime_stream_scope() {
    const HEADS: usize = 1;
    const D: usize = 128;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let (query, key, value, beta, gate, state) = runtime
        .execute(|| {
            (
                Array::from_slice(&vec![0.0_f32; HEADS * D], &[1, 128]),
                Array::from_slice(&vec![0.0_f32; HEADS * D], &[1, 128]),
                Array::from_slice(&vec![0.0_f32; HEADS * D], &[1, 128]),
                Array::from_slice(&[0.0_f32; HEADS], &[1]),
                Array::from_slice(&[1.0_f32; HEADS], &[1]),
                Array::from_slice(&vec![0.0_f32; HEADS * D * D], &[1, 128, 128]),
            )
        })
        .expect("create arrays");
    let error = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
        .expect_err("kernel call outside MlxRuntime::execute must fail");
    assert!(
        matches!(error, MlxError::InvalidState(message) if message.contains("MlxRuntime::execute"))
    );
}
