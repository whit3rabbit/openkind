//! The fused SIMD-group DeltaNet path against its own row loop and against
//! the reference operations, at the pinned layer shapes on seeded weights.
//!
//! Within `MetalSimd` the fused launch and the row loop run identical per-row
//! instructions, so they must agree bit for bit. Against `ReferenceOps` the
//! recurrence reduces in a different order, so outputs and recurrent states
//! are held to a relative tolerance while the convolution window, which no
//! kernel touches, must stay bit-identical.

use mlx_rs::Array;

use super::super::*;
use super::host_reference::{seeded, synthetic_tensors};
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

fn host(array: &Array) -> Vec<f32> {
    array.eval().expect("eval");
    array.to_vec_cast::<f32>().expect("host read")
}

fn linear_state(state: MlxLayerState) -> MlxLinearState {
    let MlxLayerState::Linear(state) = state else {
        panic!("a DeltaNet layer returns a linear state");
    };
    state
}

struct Mixer {
    layer: MlxDecoderLayer,
}

impl Mixer {
    fn new(kernel: MlxGatedDeltaKernel) -> Self {
        let layer = MlxDecoderLayer::load(&synthetic_tensors(), 0, MlxPrecision::Fp32, kernel)
            .expect("layer");
        Self { layer }
    }

    fn mixer(&self) -> &LinearAttention {
        let TokenMixer::Linear(mixer) = &self.layer.mixer else {
            panic!("layer 0 is a DeltaNet layer");
        };
        mixer
    }

    /// `(output, state)`; a trace selects the per-row loop.
    fn run(
        &self,
        input: &Array,
        rows: usize,
        previous: Option<&MlxLinearState>,
        looped: bool,
    ) -> (Vec<f32>, MlxLinearState) {
        let mut trace = LayerOperationArrays::new();
        let (output, state) = self
            .mixer()
            .forward_traced(input, rows, previous, looped.then_some(&mut trace))
            .expect("forward");
        (host(&output), linear_state(state))
    }
}

fn input(rows: usize, seed: u64) -> Array {
    Array::from_slice(
        &seeded(&[rows * HIDDEN_SIZE], seed),
        &[rows as i32, HIDDEN_SIZE as i32],
    )
}

fn assert_bits_equal(left: &[f32], right: &[f32], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: length");
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}[{index}]: {a} vs {b}");
    }
}

fn relative_deviation(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1e-30_f32, |m, v| m.max(v.abs()));
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
        / scale
}

#[test]
fn fused_simd_path_is_bit_identical_to_its_row_loop() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let simd = Mixer::new(MlxGatedDeltaKernel::MetalSimd);
    runtime
        .execute(|| {
            for rows in [1_usize, 2, 4, 5, 37] {
                let x = input(rows, 1_100 + rows as u64);
                let (fused_out, fused_state) = simd.run(&x, rows, None, false);
                let (looped_out, looped_state) = simd.run(&x, rows, None, true);
                let tag = format!("rows={rows}");
                assert_bits_equal(&fused_out, &looped_out, &format!("{tag} output"));
                assert_bits_equal(
                    &host(&fused_state.conv),
                    &host(&looped_state.conv),
                    &format!("{tag} conv"),
                );
                assert_bits_equal(
                    &host(&fused_state.recurrent),
                    &host(&looped_state.recurrent),
                    &format!("{tag} recurrent"),
                );
            }
        })
        .expect("execute");
}

#[test]
fn fused_simd_path_matches_reference_ops() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let simd = Mixer::new(MlxGatedDeltaKernel::MetalSimd);
    let reference = Mixer::new(MlxGatedDeltaKernel::ReferenceOps);
    runtime
        .execute(|| {
            for rows in [1_usize, 3, 4, 8, 37, 192] {
                let x = input(rows, 1_200 + rows as u64);
                let (simd_out, simd_state) = simd.run(&x, rows, None, false);
                let (ref_out, ref_state) = reference.run(&x, rows, None, false);
                let tag = format!("rows={rows}");
                // No kernel touches the convolution window.
                assert_bits_equal(
                    &host(&simd_state.conv),
                    &host(&ref_state.conv),
                    &format!("{tag} conv"),
                );
                let output = relative_deviation(&simd_out, &ref_out);
                let recurrent =
                    relative_deviation(&host(&simd_state.recurrent), &host(&ref_state.recurrent));
                assert!(output <= 1e-5, "{tag}: output relative deviation {output}");
                assert!(
                    recurrent <= 1e-5,
                    "{tag}: state relative deviation {recurrent}"
                );
            }
        })
        .expect("execute");
}

#[test]
fn simd_continuation_is_close_to_one_pass() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let simd = Mixer::new(MlxGatedDeltaKernel::MetalSimd);
    runtime
        .execute(|| {
            for prefix in [1_usize, 4, 9] {
                let rows = prefix + 6;
                let x = input(rows, 1_300 + prefix as u64);
                let values = host(&x);
                let head = Array::from_slice(
                    &values[..prefix * HIDDEN_SIZE],
                    &[prefix as i32, HIDDEN_SIZE as i32],
                );
                let tail = Array::from_slice(
                    &values[prefix * HIDDEN_SIZE..],
                    &[(rows - prefix) as i32, HIDDEN_SIZE as i32],
                );
                let (_, carried) = simd.run(&head, prefix, None, false);
                let (tail_out, _) = simd.run(&tail, rows - prefix, Some(&carried), false);
                let (whole_out, _) = simd.run(&x, rows, None, false);
                // Same bound as the reference continuation test: projections
                // are matmuls whose summation order depends on the row count.
                let deviation = relative_deviation(&tail_out, &whole_out[prefix * HIDDEN_SIZE..]);
                assert!(
                    deviation <= 1e-4,
                    "prefix={prefix}: continuation deviates {deviation} from one pass"
                );
            }
        })
        .expect("execute");
}

/// Layer-level timing of the recurrence, the documented MLX bottleneck, on
/// seeded weights. Diagnostic only; run with `--ignored --nocapture`.
#[test]
#[ignore = "timing diagnostic, not a correctness gate"]
fn print_reference_versus_simd_layer_timing() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let reference = Mixer::new(MlxGatedDeltaKernel::ReferenceOps);
    let simd = Mixer::new(MlxGatedDeltaKernel::MetalSimd);
    runtime
        .execute(|| {
            for rows in [12_usize, 192, 1_792] {
                let x = input(rows, 1_400 + rows as u64);
                let median = |mixer: &Mixer| {
                    let _ = mixer.run(&x, rows, None, false);
                    let mut samples: Vec<f64> = (0..5)
                        .map(|_| {
                            let start = std::time::Instant::now();
                            let (_, state) = mixer.run(&x, rows, None, false);
                            host(&state.recurrent);
                            start.elapsed().as_secs_f64() * 1_000.0
                        })
                        .collect();
                    samples.sort_by(f64::total_cmp);
                    samples[samples.len() / 2]
                };
                let reference_ms = median(&reference);
                let simd_ms = median(&simd);
                println!(
                    "rows={rows:>5}  reference-ops {reference_ms:>9.2} ms  metal-simd-v1 {simd_ms:>9.2} ms  speedup {:.2}x",
                    reference_ms / simd_ms
                );
            }
        })
        .expect("execute");
}
