//! Jeeves projections reuse the qualified 128-dimensional hybrid mixer math.
use mlx_rs::ops::{concatenate, dequantize, quantized_matmul};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::ops::{Add, Mul};

use super::{HIDDEN, INTERMEDIATE};
use crate::qwen35::mlx::layers::{
    apply_rotary, gated_delta_step, grouped_query_attention, norm_heads, op, rms_norm, row_of,
    shift_window, stack_rows, MlxFullState, MlxLayerState, MlxLinearState, QKV_SIZE, VALUE_SIZE,
};
use crate::qwen35::mlx::{MlxError, MlxGatedDeltaKernel};

pub(super) fn tensor(
    tensors: &HashMap<String, Array>,
    name: &str,
    shape: &[i32],
    dtype: Dtype,
) -> Result<Array, MlxError> {
    let value = tensors
        .get(name)
        .ok_or_else(|| MlxError::InvalidState(format!("missing tensor {name}")))?;
    if value.shape() != shape || value.dtype() != dtype {
        return Err(MlxError::InvalidState(format!(
            "tensor {name}: expected {shape:?}/{dtype:?}, found {:?}/{:?}",
            value.shape(),
            value.dtype()
        )));
    }
    if dtype == Dtype::Bfloat16 {
        value.as_dtype(Dtype::Float32).map_err(op("weight widen"))
    } else {
        Ok(value.clone())
    }
}

pub(super) struct Projection {
    weight: Array,
    scales: Array,
    biases: Array,
}
impl Projection {
    pub(super) fn load(
        tensors: &HashMap<String, Array>,
        name: &str,
        output: i32,
        input: i32,
    ) -> Result<Self, MlxError> {
        Ok(Self {
            weight: tensor(
                tensors,
                &format!("{name}.weight"),
                &[output, input / 8],
                Dtype::Uint32,
            )?,
            scales: tensor(
                tensors,
                &format!("{name}.scales"),
                &[output, input / 64],
                Dtype::Bfloat16,
            )?,
            biases: tensor(
                tensors,
                &format!("{name}.biases"),
                &[output, input / 64],
                Dtype::Bfloat16,
            )?,
        })
    }
    pub(super) fn forward(&self, input: &Array) -> Result<Array, MlxError> {
        quantized_matmul(input, &self.weight, &self.scales, &self.biases, true, 64, 4)
            .map_err(op("affine-4 projection"))
    }
    pub(super) fn embed(&self, ids: &[u32]) -> Result<Array, MlxError> {
        let indices = Array::from_slice(ids, &[ids.len() as i32]);
        let w = self
            .weight
            .take_axis(&indices, 0)
            .map_err(op("embedding packed rows"))?;
        let s = self
            .scales
            .take_axis(&indices, 0)
            .map_err(op("embedding scale rows"))?;
        let b = self
            .biases
            .take_axis(&indices, 0)
            .map_err(op("embedding bias rows"))?;
        dequantize(&w, &s, &b, 64, 4).map_err(op("affine-4 embedding"))
    }
    pub(super) fn materialize(&self) -> Result<(), MlxError> {
        for array in [&self.weight, &self.scales, &self.biases] {
            array.eval().map_err(op("projection materialize"))?;
        }
        Ok(())
    }
}

struct Delta {
    qkv: Projection,
    z: Projection,
    b: Projection,
    a: Projection,
    out: Projection,
    conv: Array,
    dt: Array,
    a_log: Array,
    norm: Array,
}
impl Delta {
    fn forward(
        &self,
        input: &Array,
        previous: Option<&MlxLayerState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let qkv = self.qkv.forward(input)?;
        let z = self.z.forward(input)?;
        let b = self.b.forward(input)?;
        let a = self.a.forward(input)?;
        let (mut conv, mut recurrent) = match previous {
            Some(MlxLayerState::Linear(state)) => (state.conv.clone(), state.recurrent.clone()),
            None => (
                mlx_rs::ops::zeros::<f32>(&[QKV_SIZE as i32, 4]).map_err(op("conv state"))?,
                mlx_rs::ops::zeros::<f32>(&[32, 128, 128]).map_err(op("DeltaNet state"))?,
            ),
            _ => return Err(MlxError::InvalidState("wrong DeltaNet state kind".into())),
        };
        let mut mixed = Vec::with_capacity(input.shape()[0] as usize);
        for index in 0..input.shape()[0] as usize {
            conv = shift_window(&conv, &row_of(&qkv, index)?)?;
            let summed = conv
                .clone()
                .mul(&self.conv)
                .sum_axis(1, false)
                .map_err(op("depthwise conv"))?;
            let qkv = mlx_rs::nn::silu(&summed).map_err(op("conv silu"))?;
            mixed.push(gated_delta_step(
                &qkv,
                &row_of(&z, index)?,
                &row_of(&b, index)?,
                &row_of(&a, index)?,
                &self.dt,
                &self.a_log,
                &self.norm,
                &mut recurrent,
                MlxGatedDeltaKernel::ReferenceOps,
            )?);
        }
        let mixed = stack_rows(&mixed, VALUE_SIZE)?;
        Ok((
            self.out.forward(&mixed)?,
            MlxLayerState::Linear(MlxLinearState { conv, recurrent }),
        ))
    }
}

struct Attention {
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    q_norm: Array,
    k_norm: Array,
}
impl Attention {
    fn forward(
        &self,
        input: &Array,
        position: usize,
        previous: Option<&MlxLayerState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let rows = input.shape()[0] as usize;
        let q = self
            .q
            .forward(input)?
            .reshape(&[rows as i32, 16, 2, 256])
            .map_err(op("query gate reshape"))?;
        let parts = q
            .split_at_indices(&[1], 2)
            .map_err(op("query gate split"))?;
        let q = parts[0]
            .reshape(&[rows as i32, 16, 256])
            .map_err(op("query reshape"))?;
        let gate = parts[1]
            .reshape(&[rows as i32, 4096])
            .map_err(op("attention gate reshape"))?;
        let k = self
            .k
            .forward(input)?
            .reshape(&[rows as i32, 4, 256])
            .map_err(op("key reshape"))?;
        let v = self
            .v
            .forward(input)?
            .reshape(&[rows as i32, 4, 256])
            .map_err(op("value reshape"))?;
        let q = apply_rotary(&norm_heads(&q, &self.q_norm)?, rows, position, 16)?;
        let k = apply_rotary(&norm_heads(&k, &self.k_norm)?, rows, position, 4)?;
        let (keys, values) = match previous {
            Some(MlxLayerState::Full(state)) => (
                concatenate(&[&state.keys, &k], 0).map_err(op("KV keys"))?,
                concatenate(&[&state.values, &v], 0).map_err(op("KV values"))?,
            ),
            None => (k, v),
            _ => return Err(MlxError::InvalidState("wrong attention state kind".into())),
        };
        let mixed = grouped_query_attention(&q, &keys, &values, rows, position)?
            .mul(&mlx_rs::nn::sigmoid(&gate).map_err(op("attention sigmoid gate"))?);
        Ok((
            self.out.forward(&mixed)?,
            MlxLayerState::Full(MlxFullState { keys, values }),
        ))
    }
}

enum Mixer {
    Delta(Delta),
    Attention(Attention),
}
pub(super) struct Layer {
    input_norm: Array,
    post_norm: Array,
    gate: Projection,
    up: Projection,
    down: Projection,
    mixer: Mixer,
}
impl Layer {
    pub(super) fn load(tensors: &HashMap<String, Array>, index: usize) -> Result<Self, MlxError> {
        let prefix = format!("language_model.model.layers.{index}");
        let projection = |name: &str, output, input| {
            Projection::load(tensors, &format!("{prefix}.{name}"), output, input)
        };
        let vector = |name: &str, size| {
            tensor(
                tensors,
                &format!("{prefix}.{name}"),
                &[size],
                Dtype::Bfloat16,
            )
        };
        let mixer = if index % 4 == 3 {
            Mixer::Attention(Attention {
                q: projection("self_attn.q_proj", 8192, HIDDEN)?,
                k: projection("self_attn.k_proj", 1024, HIDDEN)?,
                v: projection("self_attn.v_proj", 1024, HIDDEN)?,
                out: projection("self_attn.o_proj", HIDDEN, 4096)?,
                q_norm: vector("self_attn.q_norm.weight", 256)?,
                k_norm: vector("self_attn.k_norm.weight", 256)?,
            })
        } else {
            Mixer::Delta(Delta {
                qkv: projection("linear_attn.in_proj_qkv", QKV_SIZE as i32, HIDDEN)?,
                z: projection("linear_attn.in_proj_z", 4096, HIDDEN)?,
                b: projection("linear_attn.in_proj_b", 32, HIDDEN)?,
                a: projection("linear_attn.in_proj_a", 32, HIDDEN)?,
                out: projection("linear_attn.out_proj", HIDDEN, 4096)?,
                conv: tensor(
                    tensors,
                    &format!("{prefix}.linear_attn.conv1d.weight"),
                    &[8192, 4, 1],
                    Dtype::Bfloat16,
                )?
                .reshape(&[8192, 4])
                .map_err(op("converted conv layout"))?,
                dt: vector("linear_attn.dt_bias", 32)?,
                a_log: vector("linear_attn.A_log", 32)?,
                norm: vector("linear_attn.norm.weight", 128)?,
            })
        };
        // The MLX export already folds offset-RMSNorm. Adding one again changes the model.
        let layer = Self {
            input_norm: vector("input_layernorm.weight", HIDDEN)?,
            post_norm: vector("post_attention_layernorm.weight", HIDDEN)?,
            gate: projection("mlp.gate_proj", INTERMEDIATE, HIDDEN)?,
            up: projection("mlp.up_proj", INTERMEDIATE, HIDDEN)?,
            down: projection("mlp.down_proj", HIDDEN, INTERMEDIATE)?,
            mixer,
        };
        layer.materialize()?;
        Ok(layer)
    }
    fn materialize(&self) -> Result<(), MlxError> {
        let mut projections = vec![&self.gate, &self.up, &self.down];
        let mut vectors = vec![&self.input_norm, &self.post_norm];
        match &self.mixer {
            Mixer::Delta(d) => {
                projections.extend([&d.qkv, &d.z, &d.b, &d.a, &d.out]);
                vectors.extend([&d.conv, &d.dt, &d.a_log, &d.norm]);
            }
            Mixer::Attention(a) => {
                projections.extend([&a.q, &a.k, &a.v, &a.out]);
                vectors.extend([&a.q_norm, &a.k_norm]);
            }
        }
        for projection in projections {
            projection.materialize()?;
        }
        for vector in vectors {
            vector.eval().map_err(op("layer vector materialize"))?;
        }
        Ok(())
    }
    pub(super) fn forward(
        &self,
        hidden: &Array,
        position: usize,
        previous: Option<&MlxLayerState>,
    ) -> Result<(Array, MlxLayerState), MlxError> {
        let normalized = rms_norm(hidden, &self.input_norm)?;
        let (mixed, state) = match &self.mixer {
            Mixer::Delta(d) => d.forward(&normalized, previous)?,
            Mixer::Attention(a) => a.forward(&normalized, position, previous)?,
        };
        let residual = hidden.clone().add(&mixed);
        let normalized = rms_norm(&residual, &self.post_norm)?;
        let gated = mlx_rs::nn::silu(&self.gate.forward(&normalized)?)
            .map_err(op("MLP silu"))?
            .mul(&self.up.forward(&normalized)?);
        Ok((residual.add(&self.down.forward(&gated)?), state))
    }
}

pub(super) fn materialize_state(state: &MlxLayerState) -> Result<(), MlxError> {
    let arrays = match state {
        MlxLayerState::Linear(s) => [&s.conv, &s.recurrent],
        MlxLayerState::Full(s) => [&s.keys, &s.values],
    };
    for a in arrays {
        a.eval().map_err(op("state materialize"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

    #[test]
    fn packed_affine4_embedding_and_projection_match_scalar_reference() {
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).unwrap();
        runtime
            .execute(|| {
                // Low nibble is the first coefficient, with a new affine group every 64.
                let packed: Vec<u32> = (0..16)
                    .map(|word| {
                        if word % 2 == 0 {
                            0x76543210
                        } else {
                            0xfedcba98
                        }
                    })
                    .collect();
                let projection = Projection {
                    weight: Array::from_slice(&packed, &[2, 8]),
                    scales: Array::from_slice(&[0.5_f32, 0.25], &[2, 1]),
                    biases: Array::from_slice(&[-1.0_f32, 2.0], &[2, 1]),
                };
                let expected: Vec<f32> = (0..2)
                    .flat_map(|row| {
                        (0..64).map(move |column| {
                            (column % 16) as f32 * [0.5, 0.25][row] + [-1.0, 2.0][row]
                        })
                    })
                    .collect();
                let embedded = projection
                    .embed(&[0, 1])
                    .unwrap()
                    .to_vec_cast::<f32>()
                    .unwrap();
                assert_eq!(embedded, expected);
                let input: Vec<f32> = (0..64).map(|i| (i % 3) as f32 - 1.0).collect();
                let expected_products: Vec<f32> = expected
                    .chunks(64)
                    .map(|weights| weights.iter().zip(&input).map(|(w, x)| w * x).sum())
                    .collect();
                let products = projection
                    .forward(&Array::from_slice(&input, &[1, 64]))
                    .unwrap()
                    .to_vec_cast::<f32>()
                    .unwrap();
                assert_eq!(products, expected_products);
            })
            .unwrap();
    }
}
