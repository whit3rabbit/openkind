use super::batched::validate_vectorized_batch_inputs;
use super::helpers::effective_gated_delta_kernel;
use crate::qwen35::mlx::{MlxGatedDeltaKernel, MlxPrecision};

#[test]
fn fused_kernel_promotion_is_precision_scoped() {
    assert_eq!(
        effective_gated_delta_kernel(MlxPrecision::Fp32, MlxGatedDeltaKernel::MetalTree),
        MlxGatedDeltaKernel::MetalTree,
    );
    assert_eq!(
        effective_gated_delta_kernel(MlxPrecision::NativeBf16, MlxGatedDeltaKernel::MetalTree),
        MlxGatedDeltaKernel::ReferenceOps,
    );
}

#[test]
fn vectorized_batch_accepts_right_padded_lengths_and_enforces_lane_limit() {
    assert_eq!(
        validate_vectorized_batch_inputs(3, &[2, 7, 4], &[19, 19, 19]).unwrap(),
        (19, 7)
    );
    assert!(validate_vectorized_batch_inputs(1, &[2], &[19]).is_err());
    assert!(validate_vectorized_batch_inputs(9, &[1; 9], &[19; 9]).is_err());
    assert!(validate_vectorized_batch_inputs(2, &[1], &[19, 19]).is_err());
    assert!(validate_vectorized_batch_inputs(2, &[1, 2], &[19, 20]).is_err());
    assert!(validate_vectorized_batch_inputs(2, &[1, 0], &[19, 19]).is_err());
}
