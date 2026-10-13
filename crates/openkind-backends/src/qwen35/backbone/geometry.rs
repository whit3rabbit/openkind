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
    /// Documentation/verification field: the declared 3:1 interval of the
    /// published uniform checkpoints. The authoritative per-layer split is
    /// [`Self::full_attention_layers`]; pruned checkpoints (the d3 family)
    /// keep the declared interval while their actual table is irregular.
    pub full_attention_interval: usize,
    /// Per-layer mixer table: `true` marks a full-attention layer, `false`
    /// a gated-DeltaNet layer. Indexes at and beyond `layer_count` are
    /// ignored.
    pub full_attention_layers: [bool; 64],
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
    /// Whether full-attention layers apply the causal mask. Every published
    /// Qwen3.5 base checkpoint is causal; decision checkpoints of the d3
    /// family opt out (bidirectional full attention) through
    /// [`Self::with_causal`]. Gated-DeltaNet layers stay causal regardless:
    /// their recurrent scan is causal by construction.
    pub causal: bool,
}

impl Qwen35Geometry {
    /// Layer table of the published uniform checkpoints: every layer whose
    /// index satisfies `index % interval == interval - 1` takes full
    /// attention.
    pub const fn interval_layers(interval: usize, layer_count: usize) -> [bool; 64] {
        let mut layers = [false; 64];
        let mut index = interval - 1;
        while index < layer_count {
            layers[index] = true;
            index += interval;
        }
        layers
    }

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
        full_attention_layers: Self::interval_layers(4, 32),
        causal: true,
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
        full_attention_layers: Self::interval_layers(4, 32),
        causal: true,
    };

    /// Geometry of the `Qwen/Qwen3.5-2B-Base` text backbone (the base every
    /// published 2B decision checkpoint adapts).
    pub const BASE_2B: Self = Self {
        hidden_size: 2_048,
        intermediate_size: 6_144,
        layer_count: 24,
        full_attention_interval: 4,
        key_heads: 16,
        value_heads: 16,
        head_dim: 128,
        conv_kernel: 4,
        attention_heads: 8,
        kv_heads: 2,
        attention_head_dim: 256,
        rms_epsilon: 1e-6,
        full_attention_layers: Self::interval_layers(4, 24),
        causal: true,
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
        full_attention_layers: Self::interval_layers(4, 64),
        causal: true,
    };

    /// Geometry of the `Qwen/Qwen3.5-0.8B` text backbone (the base the d3
    /// lite and edge checkpoints prune and fine-tune).
    pub const BASE_08B: Self = Self {
        hidden_size: 1_024,
        intermediate_size: 3_584,
        layer_count: 24,
        full_attention_interval: 4,
        key_heads: 16,
        value_heads: 16,
        head_dim: 128,
        conv_kernel: 4,
        attention_heads: 8,
        kv_heads: 2,
        attention_head_dim: 256,
        rms_epsilon: 1e-6,
        full_attention_layers: Self::interval_layers(4, 24),
        causal: true,
    };

    /// Copy of this geometry with the causal flag set. A const so profile
    /// definitions can derive noncausal variants from the published base
    /// geometries at compile time.
    pub const fn with_causal(self, causal: bool) -> Self {
        let mut geometry = self;
        geometry.causal = causal;
        geometry
    }

    /// Geometry with the decoder block count replaced. The per-layer table
    /// is unchanged; pruned checkpoints supply their own table through a
    /// struct-update at the profile definition.
    pub const fn with_layer_count(self, layer_count: usize) -> Self {
        let mut geometry = self;
        geometry.layer_count = layer_count;
        geometry
    }

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

    /// True when `layer_index` is a full-attention layer. Callers only pass
    /// indexes below `layer_count`, where the pinned table is defined.
    pub const fn is_full_attention(&self, layer_index: usize) -> bool {
        self.full_attention_layers[layer_index]
    }

    /// True when full attention is causal and the mixer table is the
    /// uniform [`Self::interval_layers`] layout of the published base
    /// checkpoints.
    ///
    /// The MLX backend, continuation-state persistence, and the pinned
    /// architecture contract hard-code exactly this layout (`index % 4 ==
    /// 3`) and the causal mask, so they must reject any geometry for which
    /// this is false instead of silently computing a different model.
    pub const fn has_uniform_causal_layout(&self) -> bool {
        if !self.causal
            || self.full_attention_interval == 0
            || self.layer_count > self.full_attention_layers.len()
        {
            return false;
        }
        let uniform = Self::interval_layers(self.full_attention_interval, self.layer_count);
        let mut index = 0;
        while index < self.layer_count {
            if self.full_attention_layers[index] != uniform[index] {
                return false;
            }
            index += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen35Geometry;

    #[test]
    fn uniform_causal_layout_covers_the_published_bases_only() {
        for geometry in [
            Qwen35Geometry::PINNED,
            Qwen35Geometry::CLEF_FLASH,
            Qwen35Geometry::BASE_2B,
            Qwen35Geometry::CLEF,
            Qwen35Geometry::BASE_08B,
        ] {
            assert!(geometry.has_uniform_causal_layout(), "{geometry:?}");
            assert!(!geometry.with_causal(false).has_uniform_causal_layout());
        }
        // An irregular table (a pruned checkpoint keeping a different
        // full-attention subset) is rejected even when causal.
        let mut irregular = Qwen35Geometry::BASE_08B.with_layer_count(12);
        irregular.full_attention_layers = [false; 64];
        irregular.full_attention_layers[2] = true;
        assert!(!irregular.has_uniform_causal_layout());
        // Entries at and beyond `layer_count` are ignored, as documented.
        assert!(Qwen35Geometry::BASE_08B
            .with_layer_count(12)
            .has_uniform_causal_layout());
    }
}
