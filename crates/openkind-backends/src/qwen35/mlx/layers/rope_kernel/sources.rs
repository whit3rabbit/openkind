//! Metal shader sources and parameter signatures for fused partial RoPE.

pub(super) const INPUTS: &[&str] = &[
    "values",
    "row_count",
    "head_count",
    "position_start",
    "theta",
];

pub(super) const OUTPUTS: &[&str] = &["out"];

/// Fused partial rotary embedding for Qwen 3.5.
///
/// Derived from TurboSpark's `rope_neox_subdim_rows` in `rope.metal`.
///
/// Head dimension is 256, rotary dimension is 64 (32 pairs), theta is 10,000,000.
///
/// Geometry:
///   - Grid: (128, heads, rows) threads.
///   - Threadgroups: (128, 1, 1) threads (1 threadgroup per head per row).
///   - Pair < 32: computes partial RoPE on pairs (i, i + 32).
///   - Pair >= 32: coalesced passthrough for elements 64..255 (2 elements per thread).
pub(super) const SOURCE: &str = r#"
    const uint pair = thread_position_in_grid.x;
    const uint head = thread_position_in_grid.y;
    const uint row = thread_position_in_grid.z;
    const uint rows = row_count[0];
    const uint heads = head_count[0];
    const uint start_pos = position_start[0];
    const float th = theta[0];

    if (pair >= 128u || head >= heads || row >= rows) {
        return;
    }

    const uint base = (row * heads + head) * 256u;

    if (pair < 32u) {
        const float exponent = -float(2u * pair) / 64.0f;
        const float angle = float(start_pos + row) * metal::precise::pow(th, exponent);
        const float cosine = metal::precise::cos(angle);
        const float sine = metal::precise::sin(angle);

        const float lo = values[base + pair];
        const float hi = values[base + 32u + pair];

        out[base + pair] = lo * cosine - hi * sine;
        out[base + 32u + pair] = hi * cosine + lo * sine;
    } else {
        const uint offset = 64u + (pair - 32u) * 2u;
        out[base + offset] = values[base + offset];
        out[base + offset + 1u] = values[base + offset + 1u];
    }
"#;
