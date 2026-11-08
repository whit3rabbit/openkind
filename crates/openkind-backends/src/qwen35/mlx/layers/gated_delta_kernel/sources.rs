//! Metal shader sources and parameter signatures for Gated DeltaNet kernels.

pub(super) const GENERIC_INPUTS: &[&str] =
    &["query", "key", "value", "beta", "gate", "state", "mask"];

pub(super) const SEQUENCE_INPUTS: &[&str] = &[
    "query",
    "key",
    "value",
    "beta",
    "gate",
    "state",
    "row_count",
];

pub(super) const OUTPUTS: &[&str] = &["output", "next_state"];

pub(super) const GENERIC_SOURCE: &str = r#"
    const uint lane = thread_position_in_threadgroup.x;
    const uint head = threadgroup_position_in_grid.x;
    threadgroup float partial[256];
    threadgroup float memory_value;
    const bool active = !HAS_MASK || mask[head];
    const uint qk_base = head * DK;
    const uint state_base = head * DK * DV;
    const uint value_base = head * DV;
    const float q_lane = static_cast<float>(query[qk_base + lane]);
    const float k_lane = static_cast<float>(key[qk_base + lane]);

    for (uint column = 0; column < DV; ++column) {
        const uint state_index = state_base + lane * DV + column;
        const float old_state = static_cast<float>(state[state_index]);
        const float gate_value = VECTOR_GATE
            ? static_cast<float>(gate[value_base + column])
            : static_cast<float>(gate[head]);
        // Match the ordinary-ops path's dtype boundary after each elementwise
        // operation. These casts are no-ops for FP32 and intentional BF16
        // rounding points for differential parity.
        const T decayed_t = active
            ? static_cast<T>(old_state * gate_value)
            : static_cast<T>(old_state);
        const float decayed = static_cast<float>(decayed_t);
        partial[lane] = decayed * k_lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = DK >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            memory_value = partial[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const T residual_t = static_cast<T>(
            static_cast<float>(value[value_base + column]) - memory_value);
        const T correction_t = static_cast<T>(
            static_cast<float>(residual_t) * static_cast<float>(beta[head]));
        const T update_t = static_cast<T>(k_lane * static_cast<float>(correction_t));
        const T updated_t = active
            ? static_cast<T>(decayed + static_cast<float>(update_t))
            : static_cast<T>(old_state);
        next_state[state_index] = updated_t;
        partial[lane] = active ? static_cast<float>(updated_t) * q_lane : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = DK >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            output[value_base + column] = static_cast<T>(partial[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"#;

pub(super) const PACKED_SOURCE: &str = r#"
    const uint lane = thread_position_in_threadgroup.x;
    const uint head = threadgroup_position_in_grid.x;
    threadgroup float4 partial[128];
    threadgroup float4 memory_value;
    const uint qk_base = head * 128;
    const uint state_base = head * 128 * 128;
    const uint value_base = head * 128;
    const float q_lane = query[qk_base + lane];
    const float k_lane = key[qk_base + lane];
    const float gate_value = gate[head];
    const float beta_value = beta[head];
    const device float4* state4 = reinterpret_cast<const device float4*>(state);
    const device float4* value4 = reinterpret_cast<const device float4*>(value);
    device float4* output4 = reinterpret_cast<device float4*>(output);
    device float4* next_state4 = reinterpret_cast<device float4*>(next_state);

    for (uint column = 0; column < 128; column += 4) {
        const uint state_index = state_base + lane * 128 + column;
        const float4 old_state = state4[state_index >> 2];
        const float4 decayed = old_state * gate_value;
        partial[lane] = decayed * k_lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            memory_value = partial[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float4 values = value4[(value_base + column) >> 2];
        const float4 correction = (values - memory_value) * beta_value;
        const float4 updated = decayed + correction * k_lane;
        next_state4[state_index >> 2] = updated;
        partial[lane] = updated * q_lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            output4[(value_base + column) >> 2] = partial[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"#;

/// FP32 `Dk = Dv = 128`, 32-head whole-sequence recurrence, one SIMD group
/// per (value head, value column).
///
/// Every value column of the state evolves independently: its two reductions
/// (`Sᵀk` and `Sᵀq`) run over the key dimension only. So one 32-lane SIMD
/// group owns one column, each lane keeps four key-dimension entries of it in
/// registers for the whole sequence, and both reductions are `simd_sum`s.
/// There is no threadgroup memory and no barrier, and the dispatch is
/// 32 heads x 128 columns = 4,096 independent SIMD groups rather than one
/// threadgroup per head. The layout follows TurboSpark's FP16-activation
/// `gdn_delta_step_prefill`, kept FP32 end to end here. State is
/// `[heads, Dk, Dv]`, so only the one-time load and store are strided.
///
/// Arithmetic per row matches the reference recurrence: decay, `Sᵀk`,
/// `(v - Sᵀk)·β`, `S += k ⊗ δ`, `Sᵀq`. The reduction order is `simd_sum`'s,
/// so this is its own arithmetic identity, not the reduction-tree kernels'.
/// A one-row call executes exactly one iteration of the same loop, so a chain
/// of one-row calls and one many-row call produce identical bits.
pub(super) const SIMD_SEQUENCE_SOURCE: &str = r#"
    const uint lane = thread_position_in_threadgroup.x;
    const uint head = threadgroup_position_in_grid.x;
    const uint column = threadgroup_position_in_grid.y * 4 + thread_position_in_threadgroup.y;
    const uint rows = row_count[0];
    const uint state_base = head * 128 * 128;
    const device float4* query4 = reinterpret_cast<const device float4*>(query);
    const device float4* key4 = reinterpret_cast<const device float4*>(key);

    float4 s;
    s.x = state[state_base + (lane * 4 + 0) * 128 + column];
    s.y = state[state_base + (lane * 4 + 1) * 128 + column];
    s.z = state[state_base + (lane * 4 + 2) * 128 + column];
    s.w = state[state_base + (lane * 4 + 3) * 128 + column];

    for (uint row = 0; row < rows; ++row) {
        const uint head_index = row * 32 + head;
        const uint vector_index = (head_index * 128 + lane * 4) >> 2;
        const float4 k = key4[vector_index];
        const float4 q = query4[vector_index];
        s = s * gate[head_index];
        const float memory = simd_sum(dot(s, k));
        const float delta = (value[head_index * 128 + column] - memory) * beta[head_index];
        s = s + k * delta;
        const float retrieved = simd_sum(dot(s, q));
        if (lane == 0) {
            output[head_index * 128 + column] = retrieved;
        }
    }

    next_state[state_base + (lane * 4 + 0) * 128 + column] = s.x;
    next_state[state_base + (lane * 4 + 1) * 128 + column] = s.y;
    next_state[state_base + (lane * 4 + 2) * 128 + column] = s.z;
    next_state[state_base + (lane * 4 + 3) * 128 + column] = s.w;
"#;

pub(super) const PACKED_SEQUENCE_SOURCE: &str = r#"
    const uint lane = thread_position_in_threadgroup.x;
    const uint head = threadgroup_position_in_grid.x;
    const uint rows = row_count[0];
    threadgroup float4 partial[128];
    threadgroup float4 memory_value;
    const uint state_base = head * 128 * 128;
    const device float4* state4 = reinterpret_cast<const device float4*>(state);
    const device float4* value4 = reinterpret_cast<const device float4*>(value);
    device float4* output4 = reinterpret_cast<device float4*>(output);
    device float4* next_state4 = reinterpret_cast<device float4*>(next_state);

    for (uint column = 0; column < 128; column += 4) {
        const uint state_index = state_base + lane * 128 + column;
        float4 current = state4[state_index >> 2];
        for (uint row = 0; row < rows; ++row) {
            const uint head_index = row * 32 + head;
            const uint qk_base = head_index * 128;
            const uint value_base = head_index * 128;
            const float q_lane = query[qk_base + lane];
            const float k_lane = key[qk_base + lane];
            const float4 decayed = current * gate[head_index];
            partial[lane] = decayed * k_lane;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 64; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                memory_value = partial[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float4 values = value4[(value_base + column) >> 2];
            const float4 correction = (values - memory_value) * beta[head_index];
            current = decayed + correction * k_lane;
            partial[lane] = current * q_lane;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 64; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                output4[(value_base + column) >> 2] = partial[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        next_state4[state_index >> 2] = current;
    }
"#;
