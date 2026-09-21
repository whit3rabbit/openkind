use std::collections::BTreeMap;

use mlx_rs::Array;

use crate::qwen35::mlx::layers::*;
use crate::qwen35::mlx::weights::shape_i32;

pub(crate) const ROWS: usize = 8;

pub(crate) struct HostLinearWeights {
    pub(crate) input_layernorm: Vec<f32>,
    pub(crate) in_proj_qkv: Vec<f32>,
    pub(crate) in_proj_z: Vec<f32>,
    pub(crate) in_proj_b: Vec<f32>,
    pub(crate) in_proj_a: Vec<f32>,
    pub(crate) conv1d: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) a_log: Vec<f32>,
    pub(crate) delta_norm: Vec<f32>,
    pub(crate) out_proj: Vec<f32>,
    pub(crate) post_attention_layernorm: Vec<f32>,
    pub(crate) gate_proj: Vec<f32>,
    pub(crate) up_proj: Vec<f32>,
    pub(crate) down_proj: Vec<f32>,
}

pub(crate) fn seeded(values: &[usize], seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    values
        .iter()
        .map(|count| {
            let mut generated = Vec::with_capacity(*count);
            for _ in 0..*count {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
                generated.push(unit * 2.0 - 1.0);
            }
            generated
        })
        .next()
        .unwrap_or_default()
}

pub(crate) fn synthetic_tensors() -> BTreeMap<String, Array> {
    let mut tensors = BTreeMap::new();
    let mut insert = |name: String, shape: &[usize], seed: u64| {
        let count: usize = shape.iter().product();
        tensors.insert(
            name,
            Array::from_slice(&seeded(&[count], seed), &shape_i32(shape)),
        );
    };
    let layer = "model.language_model.layers.0";
    insert(
        format!("{layer}.input_layernorm.weight"),
        &[HIDDEN_SIZE],
        11,
    );
    insert(
        format!("{layer}.linear_attn.in_proj_qkv.weight"),
        &[QKV_SIZE, HIDDEN_SIZE],
        12,
    );
    insert(
        format!("{layer}.linear_attn.in_proj_z.weight"),
        &[VALUE_SIZE, HIDDEN_SIZE],
        13,
    );
    insert(
        format!("{layer}.linear_attn.in_proj_b.weight"),
        &[VALUE_HEADS, HIDDEN_SIZE],
        14,
    );
    insert(
        format!("{layer}.linear_attn.in_proj_a.weight"),
        &[VALUE_HEADS, HIDDEN_SIZE],
        15,
    );
    insert(
        format!("{layer}.linear_attn.conv1d.weight"),
        &[QKV_SIZE, 1, CONV_KERNEL],
        16,
    );
    insert(format!("{layer}.linear_attn.dt_bias"), &[VALUE_HEADS], 17);
    insert(format!("{layer}.linear_attn.A_log"), &[VALUE_HEADS], 18);
    insert(format!("{layer}.linear_attn.norm.weight"), &[HEAD_DIM], 19);
    insert(
        format!("{layer}.linear_attn.out_proj.weight"),
        &[HIDDEN_SIZE, VALUE_SIZE],
        20,
    );
    insert(
        format!("{layer}.post_attention_layernorm.weight"),
        &[HIDDEN_SIZE],
        21,
    );
    insert(
        format!("{layer}.mlp.gate_proj.weight"),
        &[9216, HIDDEN_SIZE],
        22,
    );
    insert(
        format!("{layer}.mlp.up_proj.weight"),
        &[9216, HIDDEN_SIZE],
        23,
    );
    insert(
        format!("{layer}.mlp.down_proj.weight"),
        &[HIDDEN_SIZE, 9216],
        24,
    );
    tensors
}

pub(crate) fn host_weights() -> HostLinearWeights {
    HostLinearWeights {
        input_layernorm: seeded(&[HIDDEN_SIZE], 11),
        in_proj_qkv: seeded(&[QKV_SIZE * HIDDEN_SIZE], 12),
        in_proj_z: seeded(&[VALUE_SIZE * HIDDEN_SIZE], 13),
        in_proj_b: seeded(&[VALUE_HEADS * HIDDEN_SIZE], 14),
        in_proj_a: seeded(&[VALUE_HEADS * HIDDEN_SIZE], 15),
        conv1d: seeded(&[QKV_SIZE * CONV_KERNEL], 16),
        dt_bias: seeded(&[VALUE_HEADS], 17),
        a_log: seeded(&[VALUE_HEADS], 18),
        delta_norm: seeded(&[HEAD_DIM], 19),
        out_proj: seeded(&[HIDDEN_SIZE * VALUE_SIZE], 20),
        post_attention_layernorm: seeded(&[HIDDEN_SIZE], 21),
        gate_proj: seeded(&[9216 * HIDDEN_SIZE], 22),
        up_proj: seeded(&[9216 * HIDDEN_SIZE], 23),
        down_proj: seeded(&[HIDDEN_SIZE * 9216], 24),
    }
}
