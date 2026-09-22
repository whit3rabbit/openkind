use mlx_rs::ops::concatenate;
use mlx_rs::{Array, Dtype};

use super::super::full_attention::{apply_rotary, grouped_query_attention};
use super::super::ops::op;
use super::super::*;
use super::host_reference::seeded;
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

#[test]
fn full_attention_prefix_cache_matches_one_pass() {
    const ROWS: usize = 3;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let query_values = seeded(&[ROWS * ATTENTION_HEADS * ATTENTION_HEAD_DIM], 501);
    let key_values = seeded(&[ROWS * KV_HEADS * ATTENTION_HEAD_DIM], 502);
    let value_values = seeded(&[ROWS * KV_HEADS * ATTENTION_HEAD_DIM], 503);
    let per_query = ATTENTION_HEADS * ATTENTION_HEAD_DIM;
    let per_kv = KV_HEADS * ATTENTION_HEAD_DIM;
    let (full, cached) = runtime
        .execute(|| -> Result<(Vec<f32>, Vec<f32>), MlxError> {
            let queries = Array::from_slice(
                &query_values,
                &[
                    ROWS as i32,
                    ATTENTION_HEADS as i32,
                    ATTENTION_HEAD_DIM as i32,
                ],
            );
            let keys = Array::from_slice(
                &key_values,
                &[ROWS as i32, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let values = Array::from_slice(
                &value_values,
                &[ROWS as i32, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let full = grouped_query_attention(&queries, &keys, &values, ROWS, 0)?;
            let first_queries = Array::from_slice(
                &query_values[..per_query],
                &[1, ATTENTION_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let first_keys = Array::from_slice(
                &key_values[..per_kv],
                &[1, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let first_values = Array::from_slice(
                &value_values[..per_kv],
                &[1, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let first = grouped_query_attention(&first_queries, &first_keys, &first_values, 1, 0)?;
            let suffix_queries = Array::from_slice(
                &query_values[per_query..],
                &[2, ATTENTION_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let suffix = grouped_query_attention(&suffix_queries, &keys, &values, 2, 1)?;
            let cached = concatenate(&[&first, &suffix], 0).map_err(op("cache join"))?;
            full.eval().map_err(op("full attention eval"))?;
            cached.eval().map_err(op("cached attention eval"))?;
            let full_values = full
                .to_vec_cast::<f32>()
                .map_err(op("full attention read"))?;
            let cached_values = cached
                .to_vec_cast::<f32>()
                .map_err(op("cached attention read"))?;
            Ok((full_values, cached_values))
        })
        .expect("execute")
        .expect("full attention");
    let max_abs = full
        .iter()
        .zip(&cached)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 1e-5,
        "full attention prefix cache diverged from one pass: max_abs {max_abs:.6}"
    );
}

#[test]
fn full_attention_preserves_bfloat16_dtype() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    runtime
        .execute(|| -> Result<(), MlxError> {
            let queries = Array::from_slice(
                &vec![half::bf16::ZERO; ATTENTION_HEADS * ATTENTION_HEAD_DIM],
                &[1, ATTENTION_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let keys = Array::from_slice(
                &vec![half::bf16::ZERO; KV_HEADS * ATTENTION_HEAD_DIM],
                &[1, KV_HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let values = keys.clone();
            let output = grouped_query_attention(&queries, &keys, &values, 1, 0)?;
            assert_eq!(output.dtype(), Dtype::Bfloat16);
            Ok(())
        })
        .expect("execute")
        .expect("bfloat16 attention");
}

#[test]
fn full_attention_rotary_matches_host_reference() {
    const ROWS: usize = 2;
    const HEADS: usize = 3;
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let values = seeded(&[ROWS * HEADS * ATTENTION_HEAD_DIM], 504);
    let actual = runtime
        .execute(|| -> Result<Vec<f32>, MlxError> {
            let input = Array::from_slice(
                &values,
                &[ROWS as i32, HEADS as i32, ATTENTION_HEAD_DIM as i32],
            );
            let rotated = apply_rotary(&input, ROWS, 7, HEADS)?;
            rotated.eval().map_err(op("rope eval"))?;
            rotated.to_vec_cast::<f32>().map_err(op("rope read"))
        })
        .expect("execute")
        .expect("rope");
    let mut expected = values.clone();
    for row in 0..ROWS {
        for head in 0..HEADS {
            let base = (row * HEADS + head) * ATTENTION_HEAD_DIM;
            for index in 0..ROTARY_DIM / 2 {
                let frequency = ROPE_THETA.powf(-((2 * index) as f32) / ROTARY_DIM as f32);
                let angle = (7 + row) as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let lo = values[base + index];
                let hi = values[base + ROTARY_DIM / 2 + index];
                expected[base + index] = lo * cos - hi * sin;
                expected[base + ROTARY_DIM / 2 + index] = hi * cos + lo * sin;
            }
        }
    }
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 2e-5,
        "MLX rotary embedding diverged from host reference: max_abs {max_abs:.6}"
    );
}
