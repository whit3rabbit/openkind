//! Parity of the quantized MLX Qwen3.5 backbone against the Candle CPU oracle.
//!
//! A tiny random hybrid model (three DeltaNet layers, one full-attention
//! layer, one more DeltaNet layer) is quantized to affine 8-bit with MLX. The
//! MLX backbone runs the packed triples through `quantized_matmul`; the
//! oracle runs the dequantized values of the same triples in FP32 through the
//! shared [`TextBackbone`]. Both must agree, which exercises every equation
//! the 27B checkpoint relies on (offset norms, causal convolution, DeltaNet
//! recurrence, causal GQA with the sigmoid gate, partial rotary, SiLU MLP)
//! without any model download.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_rs::Array;
use safetensors::tensor::{Dtype as StDtype, TensorView};

use super::qwen35_quantized::{read_safetensors, NormConvention, QuantizedQwen35, Weight};
use super::QuantParams;
use crate::qwen35::{EmbeddingLayout, Qwen35Embedding, Qwen35Geometry, TextBackbone};

const TINY: Qwen35Geometry = Qwen35Geometry {
    hidden_size: 64,
    intermediate_size: 128,
    layer_count: 5,
    full_attention_interval: 4,
    key_heads: 2,
    value_heads: 4,
    head_dim: 64,
    conv_kernel: 4,
    attention_heads: 2,
    kv_heads: 1,
    attention_head_dim: 64,
    rms_epsilon: 1e-6,
};
const VOCAB: usize = 96;
const QUANT: QuantParams = QuantParams {
    group_size: 64,
    bits: 8,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
    }

    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.next(scale)).collect()
    }
}

/// Everything needed to run both executions of one random model.
struct TinyModel {
    /// MLX tensors with norms stored as the raw `w` (`Offset` convention).
    mlx: BTreeMap<String, Weight>,
    /// The same tensors in Hugging Face naming and FP32 values, for the oracle.
    oracle: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    /// Names of norm tensors, for building the `Folded` variant.
    norms: Vec<String>,
}

fn bf16_array(values: &[f32], shape: &[i32]) -> Array {
    Array::from_slice(values, shape)
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap()
}

fn host(array: &Array) -> Vec<f32> {
    let widened = array.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    widened.eval().unwrap();
    widened.as_slice::<f32>().to_vec()
}

impl TinyModel {
    fn build(seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut model = Self {
            mlx: BTreeMap::new(),
            oracle: BTreeMap::new(),
            norms: Vec::new(),
        };
        let g = TINY;
        model.projection("embed_tokens.weight", VOCAB, g.hidden_size, &mut rng);
        for layer in 0..g.layer_count {
            let p = format!("layers.{layer}");
            model.norm(
                &format!("{p}.input_layernorm.weight"),
                g.hidden_size,
                &mut rng,
            );
            model.norm(
                &format!("{p}.post_attention_layernorm.weight"),
                g.hidden_size,
                &mut rng,
            );
            if g.is_full_attention(layer) {
                let a = format!("{p}.self_attn");
                model.projection(
                    &format!("{a}.q_proj.weight"),
                    g.attention_size() * 2,
                    g.hidden_size,
                    &mut rng,
                );
                model.projection(
                    &format!("{a}.k_proj.weight"),
                    g.kv_size(),
                    g.hidden_size,
                    &mut rng,
                );
                model.projection(
                    &format!("{a}.v_proj.weight"),
                    g.kv_size(),
                    g.hidden_size,
                    &mut rng,
                );
                model.projection(
                    &format!("{a}.o_proj.weight"),
                    g.hidden_size,
                    g.attention_size(),
                    &mut rng,
                );
                model.norm(
                    &format!("{a}.q_norm.weight"),
                    g.attention_head_dim,
                    &mut rng,
                );
                model.norm(
                    &format!("{a}.k_norm.weight"),
                    g.attention_head_dim,
                    &mut rng,
                );
            } else {
                let l = format!("{p}.linear_attn");
                model.projection(
                    &format!("{l}.in_proj_qkv.weight"),
                    g.qkv_size(),
                    g.hidden_size,
                    &mut rng,
                );
                model.projection(
                    &format!("{l}.in_proj_z.weight"),
                    g.value_size(),
                    g.hidden_size,
                    &mut rng,
                );
                model.projection(
                    &format!("{l}.in_proj_b.weight"),
                    g.value_heads,
                    g.hidden_size,
                    &mut rng,
                );
                model.projection(
                    &format!("{l}.in_proj_a.weight"),
                    g.value_heads,
                    g.hidden_size,
                    &mut rng,
                );
                model.projection(
                    &format!("{l}.out_proj.weight"),
                    g.hidden_size,
                    g.value_size(),
                    &mut rng,
                );
                // conv1d is [channels, 1, kernel] in the oracle and
                // [channels, kernel, 1] in MLX; the flat data is identical.
                let conv = rng.values(g.qkv_size() * g.conv_kernel, 0.5);
                model.dense_pair(
                    &format!("{l}.conv1d.weight"),
                    &[g.qkv_size() as i32, g.conv_kernel as i32, 1],
                    &[g.qkv_size(), 1, g.conv_kernel],
                    &conv,
                );
                model.dense_pair(
                    &format!("{l}.dt_bias"),
                    &[g.value_heads as i32],
                    &[g.value_heads],
                    &rng.values(g.value_heads, 0.5),
                );
                model.dense_pair(
                    &format!("{l}.A_log"),
                    &[g.value_heads as i32],
                    &[g.value_heads],
                    &rng.values(g.value_heads, 0.5),
                );
                let delta: Vec<f32> = rng
                    .values(g.head_dim, 0.2)
                    .into_iter()
                    .map(|v| 1.0 + v)
                    .collect();
                model.dense_pair(
                    &format!("{l}.norm.weight"),
                    &[g.head_dim as i32],
                    &[g.head_dim],
                    &delta,
                );
            }
            let m = format!("{p}.mlp");
            model.projection(
                &format!("{m}.gate_proj.weight"),
                g.intermediate_size,
                g.hidden_size,
                &mut rng,
            );
            model.projection(
                &format!("{m}.up_proj.weight"),
                g.intermediate_size,
                g.hidden_size,
                &mut rng,
            );
            model.projection(
                &format!("{m}.down_proj.weight"),
                g.hidden_size,
                g.intermediate_size,
                &mut rng,
            );
        }
        model.norm("norm.weight", g.hidden_size, &mut rng);
        // Dense output embedding (the checkpoint's lm_head is unquantized).
        let lm_head = rng.values(VOCAB * g.hidden_size, 0.3);
        model.mlx.insert(
            "language_model.lm_head.weight".into(),
            Weight::Dense(bf16_array(&lm_head, &[VOCAB as i32, g.hidden_size as i32])),
        );
        model
    }

    /// Quantized projection: MLX gets the packed triple, the oracle its
    /// dequantized FP32 values.
    fn projection(&mut self, suffix: &str, rows: usize, cols: usize, rng: &mut Rng) {
        // Every table is quantized from BF16 values, so scales and biases are
        // BF16 exactly as in the real checkpoint. The oracle's weights are the
        // exact FP32 dequantization of the same triple: `dequantize` itself
        // rounds to the scales' dtype, whereas `quantized_matmul` promotes the
        // scales to the activation dtype and does not.
        let dense = bf16_array(&rng.values(rows * cols, 0.25), &[rows as i32, cols as i32]);
        let (weight, scales, biases) =
            mlx_rs::ops::quantize(&dense, QUANT.group_size, QUANT.bits).unwrap();
        let to_f32 = |array: &Array| array.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let dequantized = if suffix == "embed_tokens.weight" {
            // The oracle reads the embedding as BF16, as the engine's embed does.
            mlx_rs::ops::dequantize(&weight, &scales, &biases, QUANT.group_size, QUANT.bits)
                .unwrap()
        } else {
            mlx_rs::ops::dequantize(
                &weight,
                to_f32(&scales),
                &to_f32(&biases),
                QUANT.group_size,
                QUANT.bits,
            )
            .unwrap()
        };
        for tensor in [&weight, &scales, &biases, &dequantized] {
            tensor.eval().unwrap();
        }
        self.mlx.insert(
            format!("language_model.model.{suffix}"),
            Weight::Quantized {
                weight,
                scales,
                biases,
            },
        );
        self.oracle.insert(
            format!("model.language_model.{suffix}"),
            (vec![rows, cols], host(&dequantized)),
        );
    }

    /// Norm tensor stored as raw `w`; `Folded` copies add one.
    fn norm(&mut self, suffix: &str, width: usize, rng: &mut Rng) {
        let raw = rng.values(width, 0.2);
        self.dense_pair(suffix, &[width as i32], &[width], &raw);
        self.norms.push(format!("language_model.model.{suffix}"));
    }

    fn dense_pair(
        &mut self,
        suffix: &str,
        mlx_shape: &[i32],
        oracle_shape: &[usize],
        values: &[f32],
    ) {
        self.mlx.insert(
            format!("language_model.model.{suffix}"),
            Weight::Dense(Array::from_slice(values, mlx_shape)),
        );
        self.oracle.insert(
            format!("model.language_model.{suffix}"),
            (oracle_shape.to_vec(), values.to_vec()),
        );
    }

    fn mlx_backbone(&self, convention: NormConvention) -> QuantizedQwen35 {
        let tensors = self
            .mlx
            .iter()
            .map(|(name, weight)| {
                let weight = match (weight, convention) {
                    (Weight::Dense(array), NormConvention::Folded) if self.norms.contains(name) => {
                        Weight::Dense(array + mlx_rs::Array::from_slice(&[1.0_f32], &[]))
                    }
                    (weight, _) => weight.clone(),
                };
                (name.clone(), weight)
            })
            .collect();
        QuantizedQwen35::new(
            tensors,
            TINY,
            QUANT,
            convention,
            "language_model.model",
            "language_model.lm_head.weight",
        )
    }

    /// Write the oracle checkpoint (embedding as BF16, the rest FP32).
    fn write_oracle(&self, path: &Path) {
        let mut buffers: Vec<(String, StDtype, Vec<usize>, Vec<u8>)> = Vec::new();
        for (name, (shape, values)) in &self.oracle {
            if name.ends_with("embed_tokens.weight") {
                let bytes = values
                    .iter()
                    .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                    .collect();
                buffers.push((name.clone(), StDtype::BF16, shape.clone(), bytes));
            } else {
                let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                buffers.push((name.clone(), StDtype::F32, shape.clone(), bytes));
            }
        }
        let views: Vec<(String, TensorView<'_>)> = buffers
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, path).unwrap();
    }

    fn oracle_last_hidden(&self, ids: &[u32], dir: &Path) -> Vec<f32> {
        let path = dir.join("oracle.safetensors");
        self.write_oracle(&path);
        let layout = EmbeddingLayout::read(&path, VOCAB, TINY.hidden_size).unwrap();
        let embedding = Qwen35Embedding::from_layout(path.clone(), layout);
        let backbone = TextBackbone::new(embedding, vec![path], TINY, candle_core::Device::Cpu);
        let all = backbone.forward_hidden(ids).unwrap();
        all[(ids.len() - 1) * TINY.hidden_size..].to_vec()
    }
}

fn max_difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

const IDS: [u32; 9] = [3, 17, 5, 42, 8, 8, 71, 20, 1];

#[test]
fn quantized_backbone_matches_the_cpu_oracle() {
    let model = TinyModel::build(0x9e37_79b9_7f4a_7c15);
    let dir = tempfile::tempdir().unwrap();
    model
        .mlx_backbone(NormConvention::Folded)
        .validate_layout(VOCAB)
        .expect("the random model follows the layout contract");
    let oracle = model.oracle_last_hidden(&IDS, dir.path());
    let magnitude = oracle.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(magnitude > 0.05, "oracle output is degenerate: {magnitude}");

    let offset = model
        .mlx_backbone(NormConvention::Offset)
        .forward_last_hidden(&IDS)
        .unwrap();
    let folded = model
        .mlx_backbone(NormConvention::Folded)
        .forward_last_hidden(&IDS)
        .unwrap();
    assert!(
        max_difference(&offset, &oracle) < 5e-4,
        "offset: {}",
        max_difference(&offset, &oracle)
    );
    assert!(
        max_difference(&folded, &oracle) < 5e-4,
        "folded: {}",
        max_difference(&folded, &oracle)
    );
    // Both conventions describe the same effective weights.
    assert!(max_difference(&offset, &folded) < 1e-4);
}

#[test]
fn quantized_backbone_is_causal_and_handles_one_token() {
    let model = TinyModel::build(0x1234_5678_9abc_def1);
    let backbone = model.mlx_backbone(NormConvention::Folded);
    let dir = tempfile::tempdir().unwrap();
    // A single token exercises the empty-padding conv and 1x1 attention.
    let single = backbone.forward_last_hidden(&IDS[..1]).unwrap();
    assert!(max_difference(&single, &model.oracle_last_hidden(&IDS[..1], dir.path())) < 5e-4);
    // The final hidden state must not depend on tokens after it: the prefix
    // result equals the oracle's result for the same prefix.
    let prefix = backbone.forward_last_hidden(&IDS[..5]).unwrap();
    assert!(max_difference(&prefix, &model.oracle_last_hidden(&IDS[..5], dir.path())) < 5e-4);
}

#[test]
fn output_logits_gather_the_requested_rows() {
    let model = TinyModel::build(0x0bad_cafe_0bad_cafe);
    let backbone = model.mlx_backbone(NormConvention::Folded);
    let hidden = backbone.forward_last_hidden(&IDS).unwrap();
    let tokens = [4_u32, 90, 4, 31];
    let logits = backbone.logits_for_tokens(&hidden, &tokens).unwrap();
    let Some(Weight::Dense(lm_head)) = model.mlx.get("language_model.lm_head.weight") else {
        panic!("lm_head is dense");
    };
    let table = host(lm_head);
    for (logit, token) in logits.iter().zip(tokens) {
        let row =
            &table[token as usize * TINY.hidden_size..(token as usize + 1) * TINY.hidden_size];
        let expected: f64 = row.iter().zip(&hidden).map(|(a, b)| f64::from(a * b)).sum();
        assert!((logit - expected).abs() < 1e-3, "{logit} vs {expected}");
    }
    assert!(backbone.logits_for_tokens(&hidden[1..], &tokens).is_err());
}

#[test]
fn safetensors_round_trip_runs_the_same_forward() {
    let model = TinyModel::build(0x5555_aaaa_5555_aaaa);
    let dir = tempfile::tempdir().unwrap();
    // Serialize the MLX-side tensors as the checkpoint stores them.
    let mut buffers: Vec<(String, StDtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut push = |name: &str, array: &Array| {
        array.eval().unwrap();
        let shape: Vec<usize> = array.shape().iter().map(|d| *d as usize).collect();
        let (dtype, bytes) = match array.dtype() {
            mlx_rs::Dtype::Uint32 => (
                StDtype::U32,
                array
                    .as_slice::<u32>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            mlx_rs::Dtype::Bfloat16 => (
                StDtype::BF16,
                array
                    .as_slice::<half::bf16>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            mlx_rs::Dtype::Float32 => (
                StDtype::F32,
                array
                    .as_slice::<f32>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            other => panic!("unexpected dtype {other:?}"),
        };
        buffers.push((name.to_owned(), dtype, shape, bytes));
    };
    for (name, weight) in &model.mlx {
        match weight {
            Weight::Dense(array) => push(name, array),
            Weight::Quantized {
                weight,
                scales,
                biases,
            } => {
                let base = name.strip_suffix(".weight").unwrap();
                push(name, weight);
                push(&format!("{base}.scales"), scales);
                push(&format!("{base}.biases"), biases);
            }
        }
    }
    // A vision tensor must be skipped, not loaded.
    push(
        "vision_tower.patch_embed.proj.weight",
        &Array::from_slice(&[0.0_f32; 4], &[4]),
    );
    let views: Vec<(String, TensorView<'_>)> = buffers
        .iter()
        .map(|(n, d, s, b)| (n.clone(), TensorView::new(*d, s.clone(), b).unwrap()))
        .collect();
    let path = dir.path().join("model.safetensors");
    safetensors::serialize_to_file(views, &None, &path).unwrap();

    let mut loaded = BTreeMap::new();
    read_safetensors(&path, &["vision_tower."], &mut loaded).unwrap();
    assert!(!loaded.keys().any(|name| name.starts_with("vision_tower.")));
    assert_eq!(loaded.len(), model.mlx.len());
    let from_disk = QuantizedQwen35::new(
        loaded,
        TINY,
        QUANT,
        NormConvention::Offset,
        "language_model.model",
        "language_model.lm_head.weight",
    );
    let direct = model.mlx_backbone(NormConvention::Offset);
    let a = from_disk.forward_last_hidden(&IDS).unwrap();
    let b = direct.forward_last_hidden(&IDS).unwrap();
    // The dense norm tensors round-trip as BF16-free FP32, so this is exact.
    assert!(max_difference(&a, &b) < 1e-6);
}
