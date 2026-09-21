use crate::qwen35::mlx::layers::*;

pub(crate) fn host_rms_norm(input: &[f32], weight: &[f32]) -> Vec<f32> {
    let width = HIDDEN_SIZE;
    input
        .chunks_exact(width)
        .flat_map(|row| {
            let variance = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
            let scale = (variance + RMS_EPSILON).sqrt().recip();
            row.iter()
                .zip(weight)
                .map(|(v, w)| v * scale * (1.0 + w))
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(crate) fn host_matmul(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    out_width: usize,
) -> Vec<f32> {
    let in_width = input.len() / rows;
    let mut out = vec![0.0_f32; rows * out_width];
    for r in 0..rows {
        for o in 0..out_width {
            let mut acc = 0.0_f32;
            for i in 0..in_width {
                acc += input[r * in_width + i] * weight[o * in_width + i];
            }
            out[r * out_width + o] = acc;
        }
    }
    out
}

pub(crate) fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

pub(crate) fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

pub(crate) fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else if v < -20.0 {
        v.exp()
    } else {
        v.exp().ln_1p()
    }
}
