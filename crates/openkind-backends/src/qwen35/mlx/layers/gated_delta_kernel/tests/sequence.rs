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
