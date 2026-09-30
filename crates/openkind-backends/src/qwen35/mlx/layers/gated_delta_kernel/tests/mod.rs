use super::*;
use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

mod sequence;
mod step;

fn seeded(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn tree_sum(mut values: Vec<f32>) -> f32 {
    let mut stride = values.len() / 2;
    while stride > 0 {
        for index in 0..stride {
            values[index] += values[index + stride];
        }
        stride /= 2;
    }
    values[0]
}

#[allow(clippy::too_many_arguments)]
fn host_step(
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    beta: &[f32],
    gate: &[f32],
    state: &[f32],
    mask: Option<&[bool]>,
    layout: GateLayout,
) -> (Vec<f32>, Vec<f32>) {
    let mut output = vec![0.0; heads * value_dim];
    let mut next = state.to_vec();
    for head in 0..heads {
        let active = mask.is_none_or(|values| values[head]);
        for column in 0..value_dim {
            let gate_value = match layout {
                GateLayout::Scalar => gate[head],
                GateLayout::Vector => gate[head * value_dim + column],
            };
            let decayed: Vec<f32> = (0..key_dim)
                .map(|lane| {
                    let old = state[(head * key_dim + lane) * value_dim + column];
                    if active {
                        old * gate_value
                    } else {
                        old
                    }
                })
                .collect();
            let memory = tree_sum(
                decayed
                    .iter()
                    .enumerate()
                    .map(|(lane, value)| value * key[head * key_dim + lane])
                    .collect(),
            );
            let correction = (value[head * value_dim + column] - memory) * beta[head];
            let mut products = Vec::with_capacity(key_dim);
            for (lane, decayed) in decayed.into_iter().enumerate() {
                let index = (head * key_dim + lane) * value_dim + column;
                let updated = if active {
                    decayed + key[head * key_dim + lane] * correction
                } else {
                    state[index]
                };
                next[index] = updated;
                products.push(if active {
                    updated * query[head * key_dim + lane]
                } else {
                    0.0
                });
            }
            output[head * value_dim + column] = tree_sum(products);
        }
    }
    (output, next)
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    let max = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(max <= tolerance, "max_abs {max} exceeds {tolerance}");
}
