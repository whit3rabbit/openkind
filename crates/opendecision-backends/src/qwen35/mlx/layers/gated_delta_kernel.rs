//! Fused Metal kernels for one Gated DeltaNet recurrent step.
//!
//! These kernels are derived from the recurrence itself, not copied from
//! mlx-lm. The generic kernel owns one threadgroup per value head and uses a
//! fixed binary reduction tree over the key dimension. Compile-time flags
//! select masked lanes and scalar versus vector decay gates. The packed
//! specialization handles the pinned FP32 `Dk = Dv = 128` shape four value
//! columns at a time.

use std::ffi::{c_char, CString};
use std::sync::OnceLock;

use mlx_rs::{thread_local_default_stream, Array, Dtype, Stream};

use super::MlxError;

const MAX_KEY_DIM: usize = 256;

/// Gate layout accepted by the generic kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateLayout {
    /// One decay value per value head: `[heads]`.
    Scalar,
    /// One decay value per value-head column: `[heads, value_dim]`.
    Vector,
}

/// Execute the generic reduction-tree kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generic_gated_delta(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
    mask: Option<&Array>,
    gate_layout: GateLayout,
) -> Result<(Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;
    let shape = validate_inputs(query, key, value, beta, gate, state, mask, gate_layout)?;
    if !shape.key_dim.is_power_of_two() || shape.key_dim > MAX_KEY_DIM {
        return Err(MlxError::InvalidState(format!(
            "generic gated-delta key dimension {} must be a power of two no larger than {MAX_KEY_DIM}",
            shape.key_dim
        )));
    }
    let has_mask = mask.is_some();
    let fallback_mask;
    let mask = match mask {
        Some(mask) => mask,
        None => {
            fallback_mask = Array::from_slice(&vec![true; shape.heads], &[shape.heads as i32]);
            &fallback_mask
        }
    };
    apply_kernel(
        generic_kernel(),
        [query, key, value, beta, gate, state, mask],
        shape,
        has_mask,
        gate_layout,
        false,
        &stream,
    )
}

/// Execute the packed FP32 `Dk = Dv = 128` specialization.
pub(crate) fn packed_gated_delta_128(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
) -> Result<(Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;
    let shape = validate_inputs(
        query,
        key,
        value,
        beta,
        gate,
        state,
        None,
        GateLayout::Scalar,
    )?;
    if shape.key_dim != 128 || shape.value_dim != 128 || state.dtype() != Dtype::Float32 {
        return Err(MlxError::InvalidState(
            "packed gated-delta requires FP32 state with Dk = Dv = 128".to_owned(),
        ));
    }
    let mask = Array::from_slice(&vec![true; shape.heads], &[shape.heads as i32]);
    apply_kernel(
        packed_kernel(),
        [query, key, value, beta, gate, state, &mask],
        shape,
        false,
        GateLayout::Scalar,
        true,
        &stream,
    )
}

/// Execute the packed FP32 `Dk = Dv = 128` recurrence for a complete suffix.
///
/// One threadgroup owns one value head. Each lane keeps four state columns in
/// registers while the kernel walks the sequence, removing the per-token graph
/// evaluation boundary required by the diagnostic step kernel.
pub(crate) fn packed_gated_delta_sequence_128(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
) -> Result<(Array, Array), MlxError> {
    let stream = scoped_execution_stream()?;
    let query_shape = query.shape();
    if query_shape.len() != 3 {
        return Err(MlxError::InvalidState(format!(
            "packed gated-delta sequence query must be [rows, heads, 128], found {query_shape:?}"
        )));
    }
    let rows = query_shape[0] as usize;
    let heads = query_shape[1] as usize;
    if rows == 0 || heads != 32 || query_shape[2] != 128 {
        return Err(MlxError::InvalidState(format!(
            "packed gated-delta sequence requires [rows, 32, 128], found {query_shape:?}"
        )));
    }
    let dtype = Dtype::Float32;
    for (name, array, expected) in [
        ("key", key, vec![rows as i32, heads as i32, 128]),
        ("value", value, vec![rows as i32, heads as i32, 128]),
        ("beta", beta, vec![rows as i32, heads as i32]),
        ("gate", gate, vec![rows as i32, heads as i32]),
        ("state", state, vec![heads as i32, 128, 128]),
    ] {
        if array.shape() != expected || array.dtype() != dtype {
            return Err(MlxError::InvalidState(format!(
                "packed gated-delta sequence {name} expected shape {expected:?} and FP32, found {:?} and {:?}",
                array.shape(),
                array.dtype()
            )));
        }
    }
    if query.dtype() != dtype {
        return Err(MlxError::InvalidState(format!(
            "packed gated-delta sequence query must be FP32, found {:?}",
            query.dtype()
        )));
    }

    let row_count = Array::from_slice(&[rows as u32], &[1]);
    let config = MetalKernelConfig::new();
    config.output(&[rows as i32, heads as i32, 128], dtype)?;
    config.output(&[heads as i32, 128, 128], dtype)?;
    config.grid(
        128_usize.checked_mul(heads).ok_or_else(|| {
            MlxError::InvalidState("packed sequence Metal grid overflowed usize".to_owned())
        })?,
        1,
        1,
    )?;
    config.threadgroup(128, 1, 1)?;
    let outputs = packed_sequence_kernel().apply(
        &[query, key, value, beta, gate, state, &row_count],
        &config,
        &stream,
    )?;
    if outputs.len() != 2 {
        return Err(MlxError::Operation {
            operation: "packed gated-delta sequence Metal kernel",
            message: format!("expected 2 outputs, found {}", outputs.len()),
        });
    }
    let mut outputs = outputs.into_iter();
    Ok((
        outputs.next().expect("length checked"),
        outputs.next().expect("length checked"),
    ))
}

#[derive(Debug, Clone, Copy)]
struct KernelShape {
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    dtype: Dtype,
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    query: &Array,
    key: &Array,
    value: &Array,
    beta: &Array,
    gate: &Array,
    state: &Array,
    mask: Option<&Array>,
    gate_layout: GateLayout,
) -> Result<KernelShape, MlxError> {
    let state_shape = state.shape();
    if state_shape.len() != 3 {
        return Err(MlxError::InvalidState(format!(
            "gated-delta state must be [heads, Dk, Dv], found {state_shape:?}"
        )));
    }
    let [heads, key_dim, value_dim] = [
        state_shape[0] as usize,
        state_shape[1] as usize,
        state_shape[2] as usize,
    ];
    let dtype = state.dtype();
    if !matches!(dtype, Dtype::Float32 | Dtype::Bfloat16) {
        return Err(MlxError::InvalidState(format!(
            "gated-delta state dtype must be FP32 or BF16, found {dtype:?}"
        )));
    }
    for (name, array, expected) in [
        ("query", query, vec![heads as i32, key_dim as i32]),
        ("key", key, vec![heads as i32, key_dim as i32]),
        ("value", value, vec![heads as i32, value_dim as i32]),
        ("beta", beta, vec![heads as i32]),
    ] {
        if array.shape() != expected || array.dtype() != dtype {
            return Err(MlxError::InvalidState(format!(
                "gated-delta {name} expected shape {expected:?} and dtype {dtype:?}, found {:?} and {:?}",
                array.shape(),
                array.dtype()
            )));
        }
    }
    let expected_gate = match gate_layout {
        GateLayout::Scalar => vec![heads as i32],
        GateLayout::Vector => vec![heads as i32, value_dim as i32],
    };
    if gate.shape() != expected_gate || gate.dtype() != dtype {
        return Err(MlxError::InvalidState(format!(
            "gated-delta gate expected shape {expected_gate:?} and dtype {dtype:?}, found {:?} and {:?}",
            gate.shape(),
            gate.dtype()
        )));
    }
    if let Some(mask) = mask {
        if mask.shape() != [heads as i32] || mask.dtype() != Dtype::Bool {
            return Err(MlxError::InvalidState(format!(
                "gated-delta mask expected bool shape [{heads}], found {:?} and {:?}",
                mask.shape(),
                mask.dtype()
            )));
        }
    }
    Ok(KernelShape {
        heads,
        key_dim,
        value_dim,
        dtype,
    })
}

fn apply_kernel(
    kernel: &'static MetalKernel,
    inputs: [&Array; 7],
    shape: KernelShape,
    has_mask: bool,
    gate_layout: GateLayout,
    packed: bool,
    stream: &Stream,
) -> Result<(Array, Array), MlxError> {
    let config = MetalKernelConfig::new();
    config.output(&[shape.heads as i32, shape.value_dim as i32], shape.dtype)?;
    config.output(
        &[
            shape.heads as i32,
            shape.key_dim as i32,
            shape.value_dim as i32,
        ],
        shape.dtype,
    )?;
    let grid_width = shape.key_dim.checked_mul(shape.heads).ok_or_else(|| {
        MlxError::InvalidState("gated-delta Metal grid width overflowed usize".to_owned())
    })?;
    config.grid(grid_width, 1, 1)?;
    config.threadgroup(shape.key_dim, 1, 1)?;
    if !packed {
        config.template_dtype("T", shape.dtype)?;
        config.template_int("DK", shape.key_dim)?;
        config.template_int("DV", shape.value_dim)?;
        config.template_bool("HAS_MASK", has_mask)?;
        config.template_bool("VECTOR_GATE", gate_layout == GateLayout::Vector)?;
    }
    let outputs = kernel.apply(&inputs, &config, stream)?;
    if outputs.len() != 2 {
        return Err(MlxError::Operation {
            operation: "gated-delta Metal kernel",
            message: format!("expected 2 outputs, found {}", outputs.len()),
        });
    }
    let mut outputs = outputs.into_iter();
    let output = outputs.next().expect("length checked");
    let next_state = outputs.next().expect("length checked");
    // Evaluating either sibling executes the complete multi-output primitive.
    // Do that before constructing the next recurrent invocation: it preserves
    // deterministic state chaining without a GPU-to-host readback.
    output.eval().map_err(|error| MlxError::Operation {
        operation: "materialize gated-delta Metal step",
        message: error.to_string(),
    })?;
    Ok((output, next_state))
}

fn generic_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "opendecision_gated_delta_tree",
            GENERIC_INPUTS,
            OUTPUTS,
            GENERIC_SOURCE,
        )
    })
}

fn packed_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "opendecision_gated_delta_packed_128",
            GENERIC_INPUTS,
            OUTPUTS,
            PACKED_SOURCE,
        )
    })
}

fn packed_sequence_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "opendecision_gated_delta_packed_sequence_128",
            SEQUENCE_INPUTS,
            OUTPUTS,
            PACKED_SEQUENCE_SOURCE,
        )
    })
}

const GENERIC_INPUTS: &[&str] = &["query", "key", "value", "beta", "gate", "state", "mask"];
const SEQUENCE_INPUTS: &[&str] = &[
    "query",
    "key",
    "value",
    "beta",
    "gate",
    "state",
    "row_count",
];
const OUTPUTS: &[&str] = &["output", "next_state"];

const GENERIC_SOURCE: &str = r#"
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

const PACKED_SOURCE: &str = r#"
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

const PACKED_SEQUENCE_SOURCE: &str = r#"
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

struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
}

// SAFETY: kernel application is only reachable inside `MlxRuntime::execute`,
// whose process-wide lock serializes the MLX stream and compiled-kernel handle.
unsafe impl Send for MetalKernel {}
// SAFETY: same serialization contract as the `Send` implementation.
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    fn new(name: &str, input_names: &[&str], output_names: &[&str], source: &str) -> Self {
        let name = CString::new(name).expect("static kernel name has no NUL");
        let source = CString::new(source).expect("static kernel source has no NUL");
        let header = CString::new("").expect("empty header has no NUL");
        let inputs = StringVector::new(input_names);
        let outputs = StringVector::new(output_names);
        let raw = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                inputs.raw,
                outputs.raw,
                source.as_ptr(),
                header.as_ptr(),
                true,
                false,
            )
        };
        Self { raw }
    }

    fn apply(
        &self,
        inputs: &[&Array],
        config: &MetalKernelConfig,
        stream: &Stream,
    ) -> Result<Vec<Array>, MlxError> {
        let inputs = ArrayVector::new(inputs);
        let mut outputs = unsafe { mlx_sys::mlx_vector_array_new() };
        let status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                &mut outputs,
                self.raw,
                inputs.raw,
                config.raw,
                stream.as_ptr(),
            )
        };
        if status != 0 {
            let _ = unsafe { mlx_sys::mlx_vector_array_free(outputs) };
            return Err(MlxError::Operation {
                operation: "gated-delta Metal kernel",
                message: format!("status {status}"),
            });
        }
        arrays_from_vector(outputs)
    }
}

struct MetalKernelConfig {
    raw: mlx_sys::mlx_fast_metal_kernel_config,
}

impl MetalKernelConfig {
    fn new() -> Self {
        Self {
            raw: unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() },
        }
    }

    fn output(&self, shape: &[i32], dtype: Dtype) -> Result<(), MlxError> {
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    self.raw,
                    shape.as_ptr(),
                    shape.len(),
                    dtype.into(),
                )
            },
            "add Metal kernel output",
        )
    }

    fn grid(&self, x: usize, y: usize, z: usize) -> Result<(), MlxError> {
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_grid(
                    self.raw, x as i32, y as i32, z as i32,
                )
            },
            "set Metal kernel grid",
        )
    }

    fn threadgroup(&self, x: usize, y: usize, z: usize) -> Result<(), MlxError> {
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                    self.raw, x as i32, y as i32, z as i32,
                )
            },
            "set Metal kernel threadgroup",
        )
    }

    fn template_dtype(&self, name: &str, dtype: Dtype) -> Result<(), MlxError> {
        let name = CString::new(name).expect("static template name has no NUL");
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                    self.raw,
                    name.as_ptr(),
                    dtype.into(),
                )
            },
            "set Metal dtype template",
        )
    }

    fn template_int(&self, name: &str, value: usize) -> Result<(), MlxError> {
        let name = CString::new(name).expect("static template name has no NUL");
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    self.raw,
                    name.as_ptr(),
                    value as i32,
                )
            },
            "set Metal integer template",
        )
    }

    fn template_bool(&self, name: &str, value: bool) -> Result<(), MlxError> {
        let name = CString::new(name).expect("static template name has no NUL");
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                    self.raw,
                    name.as_ptr(),
                    value,
                )
            },
            "set Metal boolean template",
        )
    }
}

impl Drop for MetalKernelConfig {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(self.raw) };
    }
}

struct StringVector {
    raw: mlx_sys::mlx_vector_string,
}

impl StringVector {
    fn new(values: &[&str]) -> Self {
        let strings: Vec<CString> = values
            .iter()
            .map(|value| CString::new(*value).expect("static kernel name has no NUL"))
            .collect();
        let mut pointers: Vec<*const c_char> = strings.iter().map(|value| value.as_ptr()).collect();
        let raw =
            unsafe { mlx_sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        Self { raw }
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_vector_string_free(self.raw) };
    }
}

struct ArrayVector {
    raw: mlx_sys::mlx_vector_array,
}

impl ArrayVector {
    fn new(values: &[&Array]) -> Self {
        let handles: Vec<mlx_sys::mlx_array> = values.iter().map(|value| value.as_ptr()).collect();
        let raw = unsafe { mlx_sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) };
        Self { raw }
    }
}

impl Drop for ArrayVector {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_vector_array_free(self.raw) };
    }
}

fn arrays_from_vector(raw: mlx_sys::mlx_vector_array) -> Result<Vec<Array>, MlxError> {
    struct OwnedVector(mlx_sys::mlx_vector_array);
    impl Drop for OwnedVector {
        fn drop(&mut self) {
            let _ = unsafe { mlx_sys::mlx_vector_array_free(self.0) };
        }
    }
    let raw = OwnedVector(raw);
    let count = unsafe { mlx_sys::mlx_vector_array_size(raw.0) };
    let mut arrays = Vec::with_capacity(count);
    for index in 0..count {
        let mut array = unsafe { mlx_sys::mlx_array_new() };
        let result = unsafe { mlx_sys::mlx_vector_array_get(&mut array, raw.0, index) };
        if result != 0 {
            let _ = unsafe { mlx_sys::mlx_array_free(array) };
            return Err(MlxError::Operation {
                operation: "read Metal kernel output",
                message: format!("status {result} for output {index}"),
            });
        }
        // SAFETY: `mlx_vector_array_get` initialized a fresh independently
        // owned handle, which `Array` now owns and will free.
        arrays.push(unsafe { Array::from_ptr(array) });
    }
    Ok(arrays)
}

fn scoped_execution_stream() -> Result<Stream, MlxError> {
    thread_local_default_stream().ok_or_else(|| {
        MlxError::InvalidState(
            "custom Metal kernels must run inside MlxRuntime::execute".to_owned(),
        )
    })
}

fn status(code: i32, operation: &'static str) -> Result<(), MlxError> {
    if code == 0 {
        Ok(())
    } else {
        Err(MlxError::Operation {
            operation,
            message: format!("status {code}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::mlx::{MlxRuntime, MlxRuntimeConfig};

    fn seeded(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn tree_sum(mut values: Vec<f32>) -> f32 {
        let mut stride = values.len() / 2;
        while stride > 0 {
            for index in 0..stride {
                values[index] += values[index + stride];
            }
            stride /= 2;
        }
        values[0]
    }

    #[allow(clippy::too_many_arguments)]
    fn host_step(
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        beta: &[f32],
        gate: &[f32],
        state: &[f32],
        mask: Option<&[bool]>,
        layout: GateLayout,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut output = vec![0.0; heads * value_dim];
        let mut next = state.to_vec();
        for head in 0..heads {
            let active = mask.is_none_or(|values| values[head]);
            for column in 0..value_dim {
                let gate_value = match layout {
                    GateLayout::Scalar => gate[head],
                    GateLayout::Vector => gate[head * value_dim + column],
                };
                let decayed: Vec<f32> = (0..key_dim)
                    .map(|lane| {
                        let old = state[(head * key_dim + lane) * value_dim + column];
                        if active {
                            old * gate_value
                        } else {
                            old
                        }
                    })
                    .collect();
                let memory = tree_sum(
                    decayed
                        .iter()
                        .enumerate()
                        .map(|(lane, value)| value * key[head * key_dim + lane])
                        .collect(),
                );
                let correction = (value[head * value_dim + column] - memory) * beta[head];
                let mut products = Vec::with_capacity(key_dim);
                for (lane, decayed) in decayed.into_iter().enumerate() {
                    let index = (head * key_dim + lane) * value_dim + column;
                    let updated = if active {
                        decayed + key[head * key_dim + lane] * correction
                    } else {
                        state[index]
                    };
                    next[index] = updated;
                    products.push(if active {
                        updated * query[head * key_dim + lane]
                    } else {
                        0.0
                    });
                }
                output[head * value_dim + column] = tree_sum(products);
            }
        }
        (output, next)
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        let max = left
            .iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(max <= tolerance, "max_abs {max} exceeds {tolerance}");
    }

    #[test]
    fn generic_masked_vector_gate_matches_explicit_tree_host() {
        const HEADS: usize = 2;
        const DK: usize = 8;
        const DV: usize = 8;
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
        let query = seeded(HEADS * DK, 1);
        let key = seeded(HEADS * DK, 2);
        let value = seeded(HEADS * DV, 3);
        let beta = vec![0.25, 0.75];
        let gate = seeded(HEADS * DV, 4)
            .into_iter()
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        let state = seeded(HEADS * DK * DV, 5);
        let mask = vec![true, false];
        let expected = host_step(
            HEADS,
            DK,
            DV,
            &query,
            &key,
            &value,
            &beta,
            &gate,
            &state,
            Some(&mask),
            GateLayout::Vector,
        );
        let actual = runtime
            .execute(|| {
                let (output, next) = generic_gated_delta(
                    &Array::from_slice(&query, &[HEADS as i32, DK as i32]),
                    &Array::from_slice(&key, &[HEADS as i32, DK as i32]),
                    &Array::from_slice(&value, &[HEADS as i32, DV as i32]),
                    &Array::from_slice(&beta, &[HEADS as i32]),
                    &Array::from_slice(&gate, &[HEADS as i32, DV as i32]),
                    &Array::from_slice(&state, &[HEADS as i32, DK as i32, DV as i32]),
                    Some(&Array::from_slice(&mask, &[HEADS as i32])),
                    GateLayout::Vector,
                )
                .expect("generic kernel");
                output.eval().expect("output eval");
                next.eval().expect("state eval");
                (
                    output.to_vec_cast::<f32>().expect("output read"),
                    next.to_vec_cast::<f32>().expect("state read"),
                )
            })
            .expect("execute");
        assert_close(&actual.0, &expected.0, 1e-6);
        assert_close(&actual.1, &expected.1, 1e-6);
    }

    #[test]
    fn packed_128_matches_generic_scalar_kernel() {
        const HEADS: usize = 1;
        const D: usize = 128;
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
        let query = seeded(HEADS * D, 11);
        let key = seeded(HEADS * D, 12);
        let value = seeded(HEADS * D, 13);
        let beta = vec![0.4];
        let gate = vec![0.9];
        let state = seeded(HEADS * D * D, 14);
        let (packed, generic) = runtime
            .execute(|| {
                let query = Array::from_slice(&query, &[HEADS as i32, D as i32]);
                let key = Array::from_slice(&key, &[HEADS as i32, D as i32]);
                let value = Array::from_slice(&value, &[HEADS as i32, D as i32]);
                let beta = Array::from_slice(&beta, &[HEADS as i32]);
                let gate = Array::from_slice(&gate, &[HEADS as i32]);
                let state = Array::from_slice(&state, &[HEADS as i32, D as i32, D as i32]);
                let packed = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
                    .expect("packed kernel");
                let generic = generic_gated_delta(
                    &query,
                    &key,
                    &value,
                    &beta,
                    &gate,
                    &state,
                    None,
                    GateLayout::Scalar,
                )
                .expect("generic kernel");
                for array in [&packed.0, &packed.1, &generic.0, &generic.1] {
                    array.eval().expect("eval");
                }
                (
                    (
                        packed.0.to_vec_cast::<f32>().expect("read"),
                        packed.1.to_vec_cast::<f32>().expect("read"),
                    ),
                    (
                        generic.0.to_vec_cast::<f32>().expect("read"),
                        generic.1.to_vec_cast::<f32>().expect("read"),
                    ),
                )
            })
            .expect("execute");
        // `float4` lowering may fuse the multiply-add where the scalar generic
        // kernel does not; the reduction tree and resulting state still agree
        // within two FP32 ulps at the observed magnitude.
        assert_close(&packed.0, &generic.0, 2e-6);
        assert_close(&packed.1, &generic.1, 2e-6);
    }

    #[test]
    fn packed_sequence_128_matches_materialized_step_chain_and_replays_exactly() {
        const ROWS: usize = 4;
        const HEADS: usize = 32;
        const D: usize = 128;
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
        let query = seeded(ROWS * HEADS * D, 61);
        let key = seeded(ROWS * HEADS * D, 62);
        let value = seeded(ROWS * HEADS * D, 63);
        let beta = seeded(ROWS * HEADS, 64)
            .into_iter()
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        let gate = seeded(ROWS * HEADS, 65)
            .into_iter()
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        let state_values = seeded(HEADS * D * D, 66);

        runtime
            .execute(|| {
                let state = Array::from_slice(&state_values, &[HEADS as i32, D as i32, D as i32]);
                let run_sequence = || {
                    let result = packed_gated_delta_sequence_128(
                        &Array::from_slice(&query, &[ROWS as i32, HEADS as i32, D as i32]),
                        &Array::from_slice(&key, &[ROWS as i32, HEADS as i32, D as i32]),
                        &Array::from_slice(&value, &[ROWS as i32, HEADS as i32, D as i32]),
                        &Array::from_slice(&beta, &[ROWS as i32, HEADS as i32]),
                        &Array::from_slice(&gate, &[ROWS as i32, HEADS as i32]),
                        &state,
                    )
                    .expect("packed sequence kernel");
                    result.0.eval().expect("sequence output eval");
                    result.1.eval().expect("sequence state eval");
                    (
                        result.0.to_vec_cast::<f32>().expect("sequence output read"),
                        result.1.to_vec_cast::<f32>().expect("sequence state read"),
                    )
                };

                let first = run_sequence();
                let second = run_sequence();
                let assert_named = |name: &str, left: &[f32], right: &[f32]| {
                    let max = left
                        .iter()
                        .zip(right)
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f32, f32::max);
                    assert_eq!(max, 0.0, "{name} max_abs {max}");
                };
                assert_named("sequence replay output", &first.0, &second.0);
                assert_named("sequence replay state", &first.1, &second.1);

                let row_width = HEADS * D;
                let head_width = HEADS;
                let mut current = state;
                let mut step_outputs = Vec::with_capacity(ROWS * row_width);
                for row in 0..ROWS {
                    let vector_range = row * row_width..(row + 1) * row_width;
                    let head_range = row * head_width..(row + 1) * head_width;
                    let result = packed_gated_delta_128(
                        &Array::from_slice(&query[vector_range.clone()], &[HEADS as i32, D as i32]),
                        &Array::from_slice(&key[vector_range.clone()], &[HEADS as i32, D as i32]),
                        &Array::from_slice(&value[vector_range], &[HEADS as i32, D as i32]),
                        &Array::from_slice(&beta[head_range.clone()], &[HEADS as i32]),
                        &Array::from_slice(&gate[head_range], &[HEADS as i32]),
                        &current,
                    )
                    .expect("packed step kernel");
                    step_outputs.extend(
                        result
                            .0
                            .to_vec_cast::<f32>()
                            .expect("packed step output read"),
                    );
                    current = result.1;
                }
                current.eval().expect("packed step state eval");
                let step_state = current
                    .to_vec_cast::<f32>()
                    .expect("packed step state read");
                assert_named("sequence versus step output", &first.0, &step_outputs);
                assert_named("sequence versus step state", &first.1, &step_state);
            })
            .expect("execute");
    }

    #[test]
    fn packed_128_preserves_shared_input_and_replays_exactly() {
        const HEADS: usize = 32;
        const D: usize = 128;
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
        let query = seeded(HEADS * D, 31);
        let key = seeded(HEADS * D, 32);
        let value = seeded(HEADS * D, 33);
        let beta = seeded(HEADS, 34)
            .into_iter()
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        let gate = seeded(HEADS, 35)
            .into_iter()
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        let state_values = seeded(HEADS * D * D, 36);
        runtime
            .execute(|| {
                let query = Array::from_slice(&query, &[HEADS as i32, D as i32]);
                let key = Array::from_slice(&key, &[HEADS as i32, D as i32]);
                let value = Array::from_slice(&value, &[HEADS as i32, D as i32]);
                let beta = Array::from_slice(&beta, &[HEADS as i32]);
                let gate = Array::from_slice(&gate, &[HEADS as i32]);
                let state = Array::from_slice(&state_values, &[HEADS as i32, D as i32, D as i32]);
                state.eval().expect("input eval");
                let input_before = state.to_vec_cast::<f32>().expect("input read");
                let first = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
                    .expect("first packed kernel");
                for array in [&first.0, &first.1] {
                    array.eval().expect("first eval");
                }
                let first_output = first.0.to_vec_cast::<f32>().expect("first output read");
                let first_state = first.1.to_vec_cast::<f32>().expect("first state read");
                let second = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
                    .expect("second packed kernel");
                for array in [&second.0, &second.1] {
                    array.eval().expect("second eval");
                }
                let second_output = second.0.to_vec_cast::<f32>().expect("second output read");
                let second_state = second.1.to_vec_cast::<f32>().expect("second state read");
                let input_after = state.to_vec_cast::<f32>().expect("input reread");
                assert_eq!(
                    input_before, input_after,
                    "kernel mutated shared input state"
                );
                assert_close(&first_output, &second_output, 0.0);
                assert_close(&first_state, &second_state, 0.0);

                let replay_chain = || {
                    let mut current = state.clone();
                    let mut outputs = Vec::new();
                    for step in 0..8_u32 {
                        let query = Array::from_slice(
                            &seeded(HEADS * D, 40 + step * 5),
                            &[HEADS as i32, D as i32],
                        );
                        let key = Array::from_slice(
                            &seeded(HEADS * D, 41 + step * 5),
                            &[HEADS as i32, D as i32],
                        );
                        let value = Array::from_slice(
                            &seeded(HEADS * D, 42 + step * 5),
                            &[HEADS as i32, D as i32],
                        );
                        let beta = Array::from_slice(
                            &seeded(HEADS, 43 + step * 5)
                                .into_iter()
                                .map(|value| value.abs())
                                .collect::<Vec<_>>(),
                            &[HEADS as i32],
                        );
                        let gate = Array::from_slice(
                            &seeded(HEADS, 44 + step * 5)
                                .into_iter()
                                .map(|value| value.abs())
                                .collect::<Vec<_>>(),
                            &[HEADS as i32],
                        );
                        let next =
                            packed_gated_delta_128(&query, &key, &value, &beta, &gate, &current)
                                .expect("packed chain step");
                        outputs.push(next.0);
                        current = next.1;
                    }
                    for output in &outputs {
                        output.eval().expect("chain output eval");
                    }
                    current.eval().expect("chain state eval");
                    (
                        outputs
                            .last()
                            .expect("non-empty chain")
                            .to_vec_cast::<f32>()
                            .expect("chain output read"),
                        current.to_vec_cast::<f32>().expect("chain state read"),
                    )
                };
                let chain_first = replay_chain();
                let chain_second = replay_chain();
                assert_close(&chain_first.0, &chain_second.0, 0.0);
                assert_close(&chain_first.1, &chain_second.1, 0.0);
            })
            .expect("execute");
    }

    #[test]
    fn generic_128_replays_exactly_across_all_heads() {
        const HEADS: usize = 32;
        const D: usize = 128;
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
        runtime
            .execute(|| {
                let query = Array::from_slice(&seeded(HEADS * D, 51), &[32, 128]);
                let key = Array::from_slice(&seeded(HEADS * D, 52), &[32, 128]);
                let value = Array::from_slice(&seeded(HEADS * D, 53), &[32, 128]);
                let beta = Array::from_slice(&seeded(HEADS, 54), &[32]);
                let gate = Array::from_slice(&seeded(HEADS, 55), &[32]);
                let state = Array::from_slice(
                    &seeded(HEADS * D * D, 56),
                    &[HEADS as i32, D as i32, D as i32],
                );
                let run = || {
                    let result = generic_gated_delta(
                        &query,
                        &key,
                        &value,
                        &beta,
                        &gate,
                        &state,
                        None,
                        GateLayout::Scalar,
                    )
                    .expect("generic kernel");
                    result.0.eval().expect("output eval");
                    result.1.eval().expect("state eval");
                    (
                        result.0.to_vec_cast::<f32>().expect("output read"),
                        result.1.to_vec_cast::<f32>().expect("state read"),
                    )
                };
                let first = run();
                let second = run();
                assert_close(&first.0, &second.0, 0.0);
                assert_close(&first.1, &second.1, 0.0);
            })
            .expect("execute");
    }

    #[test]
    fn generic_bf16_scalar_kernel_is_finite() {
        const HEADS: usize = 2;
        const D: usize = 8;
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
        runtime.qualify_bf16().expect("BF16 preflight");
        let bf16 = |values: Vec<f32>| {
            values
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>()
        };
        let output = runtime
            .execute(|| {
                let query = Array::from_slice(&bf16(seeded(HEADS * D, 21)), &[2, 8]);
                let key = Array::from_slice(&bf16(seeded(HEADS * D, 22)), &[2, 8]);
                let value = Array::from_slice(&bf16(seeded(HEADS * D, 23)), &[2, 8]);
                let beta = Array::from_slice(&bf16(vec![0.3, 0.7]), &[2]);
                let gate = Array::from_slice(&bf16(vec![0.8, 0.9]), &[2]);
                let state = Array::from_slice(&bf16(seeded(HEADS * D * D, 24)), &[2, 8, 8]);
                let (output, next) = generic_gated_delta(
                    &query,
                    &key,
                    &value,
                    &beta,
                    &gate,
                    &state,
                    None,
                    GateLayout::Scalar,
                )
                .expect("generic BF16 kernel");
                output.eval().expect("output eval");
                next.eval().expect("state eval");
                output.to_vec_cast::<f32>().expect("output read")
            })
            .expect("execute");
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn custom_kernels_require_the_explicit_runtime_stream_scope() {
        const HEADS: usize = 1;
        const D: usize = 128;
        let runtime = MlxRuntime::new(MlxRuntimeConfig::default()).expect("runtime");
        let (query, key, value, beta, gate, state) = runtime
            .execute(|| {
                (
                    Array::from_slice(&vec![0.0_f32; HEADS * D], &[1, 128]),
                    Array::from_slice(&vec![0.0_f32; HEADS * D], &[1, 128]),
                    Array::from_slice(&vec![0.0_f32; HEADS * D], &[1, 128]),
                    Array::from_slice(&[0.0_f32; HEADS], &[1]),
                    Array::from_slice(&[1.0_f32; HEADS], &[1]),
                    Array::from_slice(&vec![0.0_f32; HEADS * D * D], &[1, 128, 128]),
                )
            })
            .expect("create arrays");
        let error = packed_gated_delta_128(&query, &key, &value, &beta, &gate, &state)
            .expect_err("kernel call outside MlxRuntime::execute must fail");
        assert!(
            matches!(error, MlxError::InvalidState(message) if message.contains("MlxRuntime::execute"))
        );
    }
}
