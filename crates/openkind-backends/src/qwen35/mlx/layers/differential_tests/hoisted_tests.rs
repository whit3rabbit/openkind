//! The hoisted DeltaNet path against the per-row loop it replaces.
//!
//! Both run the same FP32 reference operations at the pinned layer shapes on
//! seeded weights. The hoisted path does the row-independent work once over
//! all rows, so the claim to defend is that no output or carried state moves
//! by even one bit. Asking for a trace selects the loop, which is how these
//! tests reach it without a second copy of the code.

use mlx_rs::Array;

use super::super::*;
use super::host_reference::{seeded, synthetic_tensors};
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig};

fn host(array: &Array) -> Vec<f32> {
    array.eval().expect("eval");
    array.to_vec_cast::<f32>().expect("host read")
}

fn assert_bits_equal(left: &[f32], right: &[f32], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: length");
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}[{index}]: hoisted {a} vs loop {b}"
        );
    }
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
    fn new() -> Self {
        let layer = MlxDecoderLayer::load(
            &synthetic_tensors(),
            0,
            MlxPrecision::Fp32,
            MlxGatedDeltaKernel::ReferenceOps,
        )
        .expect("layer");
        Self { layer }
    }

    fn mixer(&self) -> &LinearAttention {
        let TokenMixer::Linear(mixer) = &self.layer.mixer else {
            panic!("layer 0 is a DeltaNet layer");
        };
        mixer
    }

    /// `(output, conv, recurrent)` from the per-row loop, selected by asking
    /// for a trace.
    fn looped(
        &self,
        input: &Array,
        rows: usize,
        previous: Option<&MlxLinearState>,
    ) -> (Vec<f32>, MlxLinearState) {
        let mut trace = LayerOperationArrays::new();
        let (output, state) = self
            .mixer()
            .forward_traced(input, rows, previous, Some(&mut trace))
            .expect("loop");
        (host(&output), linear_state(state))
    }

    fn hoisted(
        &self,
        input: &Array,
        rows: usize,
        previous: Option<&MlxLinearState>,
    ) -> (Vec<f32>, MlxLinearState) {
        let (output, state) = self
            .mixer()
            .forward_traced(input, rows, previous, None)
            .expect("hoisted");
        (host(&output), linear_state(state))
    }
}

fn input(rows: usize, seed: u64) -> Array {
    Array::from_slice(
        &seeded(&[rows * HIDDEN_SIZE], seed),
        &[rows as i32, HIDDEN_SIZE as i32],
    )
}

fn assert_states_equal(hoisted: &MlxLinearState, looped: &MlxLinearState, tag: &str) {
    assert_bits_equal(
        &host(&hoisted.conv),
        &host(&looped.conv),
        &format!("{tag} conv"),
    );
    assert_bits_equal(
        &host(&hoisted.recurrent),
        &host(&looped.recurrent),
        &format!("{tag} recurrent"),
    );
}

#[test]
fn hoisted_path_is_bit_identical_to_the_row_loop() {
    assert!(
        !crate::qwen35::mlx::layers::linear_attention::legacy_row_loop(),
        "OPENKIND_QWEN35_MLX_ROW_LOOP=1 would make both sides the loop"
    );
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let fixture = Mixer::new();
    runtime
        .execute(|| {
            // Row counts below, at, and above the convolution window of four.
            for rows in [1_usize, 2, 3, 4, 5, 8, 37] {
                let tag = format!("fresh rows={rows}");
                let x = input(rows, 700 + rows as u64);
                let (looped_out, looped_state) = fixture.looped(&x, rows, None);
                let (hoisted_out, hoisted_state) = fixture.hoisted(&x, rows, None);
                assert_bits_equal(&hoisted_out, &looped_out, &format!("{tag} output"));
                assert_states_equal(&hoisted_state, &looped_state, &tag);
            }
        })
        .expect("execute");
}

#[test]
fn hoisted_continuation_is_bit_identical_to_the_loop_and_close_to_one_pass() {
    let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
    let fixture = Mixer::new();
    runtime
        .execute(|| {
            // A carried state of every shape the window can take: fewer rows
            // than taps, exactly the taps, and many more.
            for prefix in [1_usize, 3, 4, 9] {
                let rows = prefix + 6;
                let x = input(rows, 900 + prefix as u64);
                let head = Array::from_slice(
                    &host(&x)[..prefix * HIDDEN_SIZE],
                    &[prefix as i32, HIDDEN_SIZE as i32],
                );
                let tail = Array::from_slice(
                    &host(&x)[prefix * HIDDEN_SIZE..],
                    &[(rows - prefix) as i32, HIDDEN_SIZE as i32],
                );
                let (_, carried_looped) = fixture.looped(&head, prefix, None);
                let (_, carried_hoisted) = fixture.hoisted(&head, prefix, None);
                let tag = format!("prefix={prefix}");
                assert_states_equal(&carried_hoisted, &carried_looped, &format!("{tag} carried"));

                let (looped_out, looped_state) =
                    fixture.looped(&tail, rows - prefix, Some(&carried_looped));
                let (hoisted_out, hoisted_state) =
                    fixture.hoisted(&tail, rows - prefix, Some(&carried_hoisted));
                assert_bits_equal(&hoisted_out, &looped_out, &format!("{tag} suffix output"));
                assert_states_equal(&hoisted_state, &looped_state, &format!("{tag} suffix"));

                // Splitting a sequence is held to a tolerance, not to bits: the
                // projections are matmuls whose summation order depends on the
                // row count, on the loop and the hoisted path alike.
                let (whole_out, _) = fixture.hoisted(&x, rows, None);
                let whole_tail = &whole_out[prefix * HIDDEN_SIZE..];
                let scale = whole_tail.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                let deviation = hoisted_out
                    .iter()
                    .zip(whole_tail)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    deviation <= 1e-4 * scale,
                    "{tag}: continuation deviates {deviation} from one pass on scale {scale}"
                );
            }
        })
        .expect("execute");
}
