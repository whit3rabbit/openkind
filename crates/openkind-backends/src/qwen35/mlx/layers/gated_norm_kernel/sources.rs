//! Metal shader sources and parameter signatures for fused gated output norm.

pub(super) const INPUTS: &[&str] = &[
    "y",
    "z",
    "weight",
    "row_count",
    "v_heads",
    "value_dim",
    "eps",
];

pub(super) const OUTPUTS: &[&str] = &["out"];

/// Fused Gated Output Norm: `out = rmsnorm(y; weight, eps) * silu(z)`
///
/// Designed after TurboSpark's `gdn_gated_norm` in `gdn.metal`.
///
/// Geometry:
///   - Grid: (32 * v_heads, rows, 1) threads.
///   - Threadgroup: (32, 1, 1) threads (single SIMD group).
///   - Memory accesses: perfectly coalesced across the 32 lanes (stride 32).
///   - Cross-lane reduction: `simd_sum` across the warp (no shared memory or barriers).
pub(super) const SOURCE: &str = r#"
    const uint head = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint num_heads = v_heads[0];
    const uint num_rows = row_count[0];
    const uint dim = value_dim[0];
    const float epsilon = eps[0];

    if (head >= num_heads || row >= num_rows) {
        return;
    }

    const uint lane = thread_index_in_simdgroup;
    const uint base = (row * num_heads + head) * dim;

    // 1. Coalesced sum of squares across dim
    float sumsq = 0.0f;
    for (uint i = lane; i < dim; i += 32u) {
        const float val = y[base + i];
        sumsq += val * val;
    }
    sumsq = metal::simd_sum(sumsq);

    // 2. Inverse RMS
    const float mean_sq = sumsq / float(dim);
    const float inv_rms = metal::rsqrt(mean_sq + epsilon);

    // 3. Fused RMSNorm scale and precise SiLU gate
    for (uint i = lane; i < dim; i += 32u) {
        const float val = y[base + i];
        const float w = weight[i];
        const float zv = z[base + i];
        const float silu_z = zv / (1.0f + metal::precise::exp(-zv));
        out[base + i] = (val * inv_rms * w) * silu_z;
    }
"#;
