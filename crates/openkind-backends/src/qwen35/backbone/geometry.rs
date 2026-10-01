//! Frozen Qwen3.5 text-backbone geometry shared by every execution path.
//!
//! The parity-verified CPU oracle and the MLX backend were both written
//! against the pinned `Qwen/Qwen3.5-4B` dimensions. The Cloudflare Clef
//! family reuses the same hybrid decoder architecture (gated DeltaNet with
//! causal depthwise convolution, grouped-query attention with partial
//! rotary embedding, SiLU-gated MLP) at different widths, so every
//! dimension that enters the forward equations is carried by this struct
//! instead of a module constant.
//!
//! Rope parameters are shared by every supported checkpoint: rotary over
//! the first quarter of the full-attention head dim with NeoX pairing at
//! theta 1e7 (`mrope_interleaved` collapses to this for text-only input;
//! this was verified against the pinned-profile parity fixtures).

/// Full-sequence geometry of one frozen Qwen3.5 text backbone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35Geometry {
    /// Hidden width of the residual stream.
    pub hidden_size: usize,
    /// MLP intermediate width.
    pub intermediate_size: usize,
    /// Decoder block count.
    pub layer_count: usize,
    /// Every layer whose index satisfies `index % interval == interval - 1`
    /// is a full-attention layer; the others are gated-DeltaNet layers.
    pub full_attention_interval: usize,
    /// Gated-DeltaNet key heads.
    pub key_heads: usize,
    /// Gated-DeltaNet value heads.
    pub value_heads: usize,
    /// Gated-DeltaNet key/value head width.
    pub head_dim: usize,
    /// Causal depthwise convolution kernel width.
    pub conv_kernel: usize,
    /// Full-attention query heads.
    pub attention_heads: usize,
    /// Full-attention grouped key/value heads.
    pub kv_heads: usize,
    /// Full-attention head width.
    pub attention_head_dim: usize,
    /// RMSNorm epsilon (shared by input/post norms, DeltaNet norms, and the
    /// final norm).
    pub rms_epsilon: f32,
}

impl Qwen35Geometry {
    /// Geometry of the pinned `Qwen/Qwen3.5-4B` text backbone that the CPU
    /// oracle, MLX backend, and every pinned survey profile execute.
    pub const PINNED: Self = Self {
        hidden_size: 2_560,
        intermediate_size: 9_216,
        layer_count: 32,
        full_attention_interval: 4,
        key_heads: 16,
        value_heads: 32,
        head_dim: 128,
        conv_kernel: 4,
        attention_heads: 16,
        kv_heads: 4,
        attention_head_dim: 256,
        rms_epsilon: 1e-6,
    };

    /// Geometry of the `Cloudflare/clef-flash` text backbone.
    pub const CLEF_FLASH: Self = Self {
        hidden_size: 4_096,
        intermediate_size: 12_288,
        layer_count: 32,
        full_attention_interval: 4,
        key_heads: 16,
        value_heads: 32,
        head_dim: 128,
        conv_kernel: 4,
        attention_heads: 16,
        kv_heads: 4,
        attention_head_dim: 256,
        rms_epsilon: 1e-6,
    };

    /// Geometry of the `Cloudflare/clef` (27B) text backbone.
    pub const CLEF: Self = Self {
        hidden_size: 5_120,
        intermediate_size: 17_408,
        layer_count: 64,
        full_attention_interval: 4,
        key_heads: 16,
        value_heads: 48,
        head_dim: 128,
        conv_kernel: 4,
        attention_heads: 24,
        kv_heads: 4,
        attention_head_dim: 256,
        rms_epsilon: 1e-6,
    };

    /// Gated-DeltaNet key projection width.
    pub const fn key_size(&self) -> usize {
        self.key_heads * self.head_dim
    }

    /// Gated-DeltaNet value projection width.
    pub const fn value_size(&self) -> usize {
        self.value_heads * self.head_dim
    }

    /// Gated-DeltaNet fused q/k/v projection width.
    pub const fn qkv_size(&self) -> usize {
        self.key_size() * 2 + self.value_size()
    }

    /// Full-attention query projection width (queries carry a sigmoid gate).
    pub const fn attention_size(&self) -> usize {
        self.attention_heads * self.attention_head_dim
    }

    /// Full-attention grouped key/value width.
    pub const fn kv_size(&self) -> usize {
        self.kv_heads * self.attention_head_dim
    }

    /// Rotary width: the first quarter of the full-attention head dim
    /// (`partial_rotary_factor` 0.25 on every supported checkpoint).
    pub const fn rotary_dim(&self) -> usize {
        self.attention_head_dim / 4
    }

    /// Rotary base shared by every supported checkpoint.
    pub const ROPE_THETA: f32 = 10_000_000.0;

    /// True when `layer_index` is a full-attention layer.
    pub const fn is_full_attention(&self, layer_index: usize) -> bool {
        layer_index % self.full_attention_interval == self.full_attention_interval - 1
    }
}
