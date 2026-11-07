use super::*;

#[test]
fn packed_sequence_128_matches_materialized_step_chain_and_replays_exactly() {
    const ROWS: usize = 4;
    const HEADS: usize = 32;
    const D: usize = 128;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let query = seeded(ROWS * HEADS * D, 61);
    let key = seeded(ROWS * HEADS * D, 62);
    let value = seeded(ROWS * HEADS * D, 63);
    let beta = seeded(ROWS * HEADS, 64)
        .into_iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let gate = seeded(ROWS * HEADS, 65)
        .into_iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let state_values = seeded(HEADS * D * D, 66);

    runtime
        .execute(|| {
            let state = Array::from_slice(&state_values, &[HEADS as i32, D as i32, D as i32]);
            let run_sequence = || {
                let result = packed_gated_delta_sequence_128(
                    &Array::from_slice(&query, &[ROWS as i32, HEADS as i32, D as i32]),
                    &Array::from_slice(&key, &[ROWS as i32, HEADS as i32, D as i32]),
                    &Array::from_slice(&value, &[ROWS as i32, HEADS as i32, D as i32]),
                    &Array::from_slice(&beta, &[ROWS as i32, HEADS as i32]),
                    &Array::from_slice(&gate, &[ROWS as i32, HEADS as i32]),
                    &state,
                )
                .expect("packed sequence kernel");
                result.0.eval().expect("sequence output eval");
                result.1.eval().expect("sequence state eval");
                (
                    result.0.to_vec_cast::<f32>().expect("sequence output read"),
                    result.1.to_vec_cast::<f32>().expect("sequence state read"),
                )
            };

            let first = run_sequence();
            let second = run_sequence();
            let assert_named = |name: &str, left: &[f32], right: &[f32]| {
                let max = left
                    .iter()
                    .zip(right)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                assert_eq!(max, 0.0, "{name} max_abs {max}");
            };
            assert_named("sequence replay output", &first.0, &second.0);
            assert_named("sequence replay state", &first.1, &second.1);

            let row_width = HEADS * D;
            let head_width = HEADS;
            let mut current = state;
            let mut step_outputs = Vec::with_capacity(ROWS * row_width);
            for row in 0..ROWS {
                let vector_range = row * row_width..(row + 1) * row_width;
                let head_range = row * head_width..(row + 1) * head_width;
                let result = packed_gated_delta_128(
                    &Array::from_slice(&query[vector_range.clone()], &[HEADS as i32, D as i32]),
                    &Array::from_slice(&key[vector_range.clone()], &[HEADS as i32, D as i32]),
                    &Array::from_slice(&value[vector_range], &[HEADS as i32, D as i32]),
                    &Array::from_slice(&beta[head_range.clone()], &[HEADS as i32]),
                    &Array::from_slice(&gate[head_range], &[HEADS as i32]),
                    &current,
                )
                .expect("packed step kernel");
                step_outputs.extend(
                    result
                        .0
                        .to_vec_cast::<f32>()
                        .expect("packed step output read"),
                );
                current = result.1;
            }
            current.eval().expect("packed step state eval");
            let step_state = current
                .to_vec_cast::<f32>()
                .expect("packed step state read");
            assert_named("sequence versus step output", &first.0, &step_outputs);
            assert_named("sequence versus step state", &first.1, &step_state);
        })
        .expect("execute");
}

/// Model-shaped recurrence inputs: unit-L2 keys, queries scaled by
/// `1/sqrt(Dk)` as the hoisted preparation produces, and gate and beta in
/// (0, 1). Raw seeded keys make the delta rule grow without bound over long
/// sequences, which would test overflow rather than the kernel.
struct ModelShapedInputs {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    beta: Vec<f32>,
    gate: Vec<f32>,
    state: Vec<f32>,
}

fn model_shaped_inputs(rows: usize, seed: u32) -> ModelShapedInputs {
    const HEADS: usize = 32;
    const D: usize = 128;
    let normalize = |values: Vec<f32>, scale: f32| {
        values
            .chunks(D)
            .flat_map(|chunk| {
                let norm = chunk.iter().map(|value| value * value).sum::<f32>().sqrt();
                chunk
                    .iter()
                    .map(move |value| value / norm * scale)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let unit_interval = |values: Vec<f32>| {
        values
            .into_iter()
            .map(|value| 0.05 + 0.9 * (value * 0.5 + 0.5))
            .collect::<Vec<_>>()
    };
    ModelShapedInputs {
        query: normalize(seeded(rows * HEADS * D, seed), (D as f32).sqrt().recip()),
        key: normalize(seeded(rows * HEADS * D, seed + 1), 1.0),
        value: seeded(rows * HEADS * D, seed + 2),
        beta: unit_interval(seeded(rows * HEADS, seed + 3)),
        gate: unit_interval(seeded(rows * HEADS, seed + 4)),
        state: seeded(HEADS * D * D, seed + 5)
            .into_iter()
            .map(|value| value * 0.1)
            .collect(),
    }
}

fn run_simd_sequence(inputs: &ModelShapedInputs, rows: usize) -> (Vec<f32>, Vec<f32>) {
    const HEADS: i32 = 32;
    const D: i32 = 128;
    let rows = rows as i32;
    let (output, state) = simd_gated_delta_sequence_128(
        &Array::from_slice(&inputs.query, &[rows, HEADS, D]),
        &Array::from_slice(&inputs.key, &[rows, HEADS, D]),
        &Array::from_slice(&inputs.value, &[rows, HEADS, D]),
        &Array::from_slice(&inputs.beta, &[rows, HEADS]),
        &Array::from_slice(&inputs.gate, &[rows, HEADS]),
        &Array::from_slice(&inputs.state, &[HEADS, D, D]),
    )
    .expect("SIMD sequence kernel");
    output.eval().expect("SIMD output eval");
    state.eval().expect("SIMD state eval");
    (
        output.to_vec_cast::<f32>().expect("SIMD output read"),
        state.to_vec_cast::<f32>().expect("SIMD state read"),
    )
}

/// The host recurrence chained over every row, through [`host_step`].
fn host_sequence(inputs: &ModelShapedInputs, rows: usize) -> (Vec<f32>, Vec<f32>) {
    const HEADS: usize = 32;
    const D: usize = 128;
    let mut state = inputs.state.clone();
    let mut outputs = Vec::with_capacity(rows * HEADS * D);
    for row in 0..rows {
        let vectors = row * HEADS * D..(row + 1) * HEADS * D;
        let scalars = row * HEADS..(row + 1) * HEADS;
        let (output, next) = host_step(
            HEADS,
            D,
            D,
            &inputs.query[vectors.clone()],
            &inputs.key[vectors.clone()],
            &inputs.value[vectors],
            &inputs.beta[scalars.clone()],
            &inputs.gate[scalars],
            &state,
            None,
            GateLayout::Scalar,
        );
        outputs.extend(output);
        state = next;
    }
    (outputs, state)
}

/// Largest absolute difference, divided by the reference's largest magnitude
/// (at least 1), so the bound does not depend on how large states grow.
fn relative_max_difference(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let scale = expected
        .iter()
        .fold(1.0_f32, |max, value| max.max(value.abs()));
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max)
        / scale
}

#[test]
fn simd_sequence_128_matches_the_host_recurrence() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| {
            for (rows, seed) in [(1_usize, 101_u32), (7, 111), (48, 121)] {
                let inputs = model_shaped_inputs(rows, seed);
                let (output, state) = run_simd_sequence(&inputs, rows);
                let (host_output, host_state) = host_sequence(&inputs, rows);
                let output_error = relative_max_difference(&output, &host_output);
                let state_error = relative_max_difference(&state, &host_state);
                // simd_sum and the host's binary tree sum 128 FP32 products in
                // different orders; the bound is a few ULP per step.
                assert!(
                    output_error <= 2e-6,
                    "rows {rows} output relative {output_error}"
                );
                assert!(
                    state_error <= 2e-6,
                    "rows {rows} state relative {state_error}"
                );
            }
        })
        .expect("execute");
}

#[test]
fn simd_sequence_128_replays_and_matches_its_one_row_chain_exactly() {
    const HEADS: i32 = 32;
    const D: i32 = 128;
    const ROWS: usize = 9;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let inputs = model_shaped_inputs(ROWS, 131);
    runtime
        .execute(|| {
            let first = run_simd_sequence(&inputs, ROWS);
            let second = run_simd_sequence(&inputs, ROWS);
            assert_close(&first.0, &second.0, 0.0);
            assert_close(&first.1, &second.1, 0.0);

            // The step path under `MetalSimd` is this kernel with one row.
            let row_width = (HEADS * D) as usize;
            let head_width = HEADS as usize;
            let mut state = Array::from_slice(&inputs.state, &[HEADS, D, D]);
            let mut chained = Vec::with_capacity(ROWS * row_width);
            for row in 0..ROWS {
                let vectors = row * row_width..(row + 1) * row_width;
                let scalars = row * head_width..(row + 1) * head_width;
                let (output, next) = simd_gated_delta_sequence_128(
                    &Array::from_slice(&inputs.query[vectors.clone()], &[1, HEADS, D]),
                    &Array::from_slice(&inputs.key[vectors.clone()], &[1, HEADS, D]),
                    &Array::from_slice(&inputs.value[vectors], &[1, HEADS, D]),
                    &Array::from_slice(&inputs.beta[scalars.clone()], &[1, HEADS]),
                    &Array::from_slice(&inputs.gate[scalars], &[1, HEADS]),
                    &state,
                )
                .expect("one-row SIMD kernel");
                chained.extend(output.to_vec_cast::<f32>().expect("one-row output read"));
                state = next;
            }
            state.eval().expect("chained state eval");
            let chained_state = state.to_vec_cast::<f32>().expect("chained state read");
            assert_close(&first.0, &chained, 0.0);
            assert_close(&first.1, &chained_state, 0.0);
        })
        .expect("execute");
}

#[test]
fn simd_sequence_128_refuses_other_shapes_and_dtypes() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| {
            let vectors = Array::from_slice(&vec![0.0_f32; 2 * 16 * 128], &[2, 16, 128]);
            let scalars = Array::from_slice(&[0.5_f32; 2 * 16], &[2, 16]);
            let state = Array::from_slice(&vec![0.0_f32; 16 * 128 * 128], &[16, 128, 128]);
            assert!(simd_gated_delta_sequence_128(
                &vectors, &vectors, &vectors, &scalars, &scalars, &state
            )
            .is_err());
        })
        .expect("execute");
}
