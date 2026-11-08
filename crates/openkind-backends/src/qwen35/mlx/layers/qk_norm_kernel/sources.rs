//! Metal shader sources and parameter signatures for fused QKV normalization,
//! scaling, and head expansion.

pub(super) const INPUTS: &[&str] = &["qkv", "row_count", "eps"];

pub(super) const OUTPUTS: &[&str] = &["query_hat", "key_hat", "value"];

/// Fused QK RMSNorm, query scaling (1/sqrt(128)), and head broadcast (16 -> 32).
///
/// Derived from TurboSpark's `gdn_qk_norm` in `gdn.metal`.
///
/// Geometry:
///   - Grid: (32 * 32, rows, 1) threads, i.e., 32 threadgroups along x.
///   - Threadgroups: (32, 1, 1) threads (1 SIMD group per value head).
///   - Head mapping: value head `h` (0..31) reads key head `hk = h / 2` (0..15).
///   - Cross-lane reduction: `simd_sum` across the 32 lanes (zero shared memory, zero barriers).
pub(super) const SOURCE: &str = r#"
    const uint h = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint rows = row_count[0];
    const float epsilon = eps[0];

    if (h >= 32u || r >= rows) {
        return;
    }

    const uint lane = thread_index_in_simdgroup;
    const uint hk = h / 2u;

    const uint qkv_row_base = r * 8192u;
    const uint q_base = qkv_row_base + hk * 128u;
    const uint k_base = qkv_row_base + 2048u + hk * 128u;
    const uint v_base = qkv_row_base + 4096u + h * 128u;

    const uint out_base = r * 4096u + h * 128u;

    // 1. Load 4 floats per lane for query
    float q0 = qkv[q_base + lane * 4u + 0u];
    float q1 = qkv[q_base + lane * 4u + 1u];
    float q2 = qkv[q_base + lane * 4u + 2u];
    float q3 = qkv[q_base + lane * 4u + 3u];

    float q_sumsq = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
    q_sumsq = metal::simd_sum(q_sumsq);
    float q_scale = (1.0f / metal::precise::sqrt(q_sumsq + epsilon)) * 0.08838834764831845f; // 1.0 / sqrt(128)

    query_hat[out_base + lane * 4u + 0u] = q0 * q_scale;
    query_hat[out_base + lane * 4u + 1u] = q1 * q_scale;
    query_hat[out_base + lane * 4u + 2u] = q2 * q_scale;
    query_hat[out_base + lane * 4u + 3u] = q3 * q_scale;

    // 2. Load 4 floats per lane for key
    float k0 = qkv[k_base + lane * 4u + 0u];
    float k1 = qkv[k_base + lane * 4u + 1u];
    float k2 = qkv[k_base + lane * 4u + 2u];
    float k3 = qkv[k_base + lane * 4u + 3u];

    float k_sumsq = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
    k_sumsq = metal::simd_sum(k_sumsq);
    float k_scale = 1.0f / metal::precise::sqrt(k_sumsq + epsilon);

    key_hat[out_base + lane * 4u + 0u] = k0 * k_scale;
    key_hat[out_base + lane * 4u + 1u] = k1 * k_scale;
    key_hat[out_base + lane * 4u + 2u] = k2 * k_scale;
    key_hat[out_base + lane * 4u + 3u] = k3 * k_scale;

    // 3. Load 4 floats per lane for value and store directly
    value[out_base + lane * 4u + 0u] = qkv[v_base + lane * 4u + 0u];
    value[out_base + lane * 4u + 1u] = qkv[v_base + lane * 4u + 1u];
    value[out_base + lane * 4u + 2u] = qkv[v_base + lane * 4u + 2u];
    value[out_base + lane * 4u + 3u] = qkv[v_base + lane * 4u + 3u];
"#;
