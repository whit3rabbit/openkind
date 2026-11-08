//! Metal shader sources and parameter signatures for fused SwiGLU.

pub(super) const INPUTS: &[&str] = &["gate", "up", "element_count"];

pub(super) const OUTPUTS: &[&str] = &["out"];

/// Fused SwiGLU activation: `out = silu(gate) * up` in FP32.
///
/// Derived from TurboSpark's `silu_mul_fp16` in `utility.metal`.
///
/// Uses float4 vectorized loads and stores with in-register `precise::exp` evaluation.
pub(super) const SOURCE: &str = r#"
    const uint tid = thread_position_in_grid.x;
    const uint total_elements = element_count[0];
    const uint vec_index = tid * 4u;

    if (vec_index >= total_elements) {
        return;
    }

    // Fully aligned float4 vector path
    if (vec_index + 4u <= total_elements) {
        float4 g = float4(gate[vec_index + 0u], gate[vec_index + 1u], gate[vec_index + 2u], gate[vec_index + 3u]);
        float4 u = float4(up[vec_index + 0u], up[vec_index + 1u], up[vec_index + 2u], up[vec_index + 3u]);

        float4 sigmoid_g = 1.0f / (1.0f + metal::precise::exp(-g));
        float4 silu_g = g * sigmoid_g;
        float4 result = silu_g * u;

        out[vec_index + 0u] = result.x;
        out[vec_index + 1u] = result.y;
        out[vec_index + 2u] = result.z;
        out[vec_index + 3u] = result.w;
    } else {
        // Scalar tail if element count is not a multiple of 4
        for (uint i = vec_index; i < total_elements; ++i) {
            float g = gate[i];
            float u = up[i];
            float sigmoid_g = 1.0f / (1.0f + metal::precise::exp(-g));
            out[i] = (g * sigmoid_g) * u;
        }
    }
"#;
