//! Metal shader sources and parameter signatures for fused causal convolution.

pub(super) const INPUTS: &[&str] = &["raw_qkv", "conv", "conv_w", "row_count", "channel_count"];

pub(super) const OUTPUTS: &[&str] = &["output", "next_conv"];

/// Fused causal depthwise 1D convolution with SiLU activation and carried state update.
///
/// Designed after TurboSpark's `gdn_conv_mix_prefill` and `short_conv`, adapted
/// to OpenKind's `[channels, 4]` convolution state representation.
///
/// Each thread `(c, r)` computes the convolution for channel `c` at sequence row `r`
/// across the 4 taps:
/// - tap 0: virtual index `r - 3`
/// - tap 1: virtual index `r - 2`
/// - tap 2: virtual index `r - 1`
/// - tap 3: virtual index `r` (current row in `raw_qkv`)
///
/// Virtual indices `< 0` index the incoming `conv` buffer.
/// Summation order matches the reference host loop (`tap 0 + tap 1 + tap 2 + tap 3`).
/// The final row `r == rows - 1` records the updated 4-tap carried window to `next_conv`.
pub(super) const CAUSAL_CONV_SOURCE: &str = r#"
    const uint c = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    const uint rows = row_count[0];
    const uint channels = channel_count[0];
    if (c >= channels || r >= rows) return;

    float v0 = (r >= 3) ? raw_qkv[(r - 3) * channels + c] : conv[c * 4 + (1 + r)];
    float v1 = (r >= 2) ? raw_qkv[(r - 2) * channels + c] : conv[c * 4 + (2 + r)];
    float v2 = (r >= 1) ? raw_qkv[(r - 1) * channels + c] : conv[c * 4 + (3 + r)];
    float v3 = raw_qkv[r * channels + c];

    float w0 = conv_w[c * 4 + 0];
    float w1 = conv_w[c * 4 + 1];
    float w2 = conv_w[c * 4 + 2];
    float w3 = conv_w[c * 4 + 3];

    float acc = ((v0 * w0 + v1 * w1) + v2 * w2) + v3 * w3;
    output[r * channels + c] = acc / (1.0f + metal::precise::exp(-acc));

    if (r == rows - 1) {
        for (uint j = 0; j < 4; ++j) {
            int t = int(rows) - 4 + int(j);
            float val = (t >= 0) ? raw_qkv[uint(t) * channels + c] : conv[c * 4 + uint(4 + t)];
            next_conv[c * 4 + j] = val;
        }
    }
"#;
