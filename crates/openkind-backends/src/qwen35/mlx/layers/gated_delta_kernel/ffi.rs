//! C FFI bindings and helper wrappers for MLX Metal custom kernels.

use std::ffi::{c_char, CString};

use mlx_rs::{thread_local_default_stream, Array, Dtype, Stream};

use crate::qwen35::mlx::layers::MlxError;

pub(super) struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
}

// SAFETY: kernel application is only reachable inside `MlxRuntime::execute`,
// whose process-wide lock serializes the MLX stream and compiled-kernel handle.
unsafe impl Send for MetalKernel {}
// SAFETY: same serialization contract as the `Send` implementation.
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    pub(super) fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
    ) -> Self {
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

    pub(super) fn apply(
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

pub(super) struct MetalKernelConfig {
    raw: mlx_sys::mlx_fast_metal_kernel_config,
}

impl MetalKernelConfig {
    pub(super) fn new() -> Self {
        Self {
            raw: unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() },
        }
    }

    pub(super) fn output(&self, shape: &[i32], dtype: Dtype) -> Result<(), MlxError> {
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

    pub(super) fn grid(&self, x: usize, y: usize, z: usize) -> Result<(), MlxError> {
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_grid(
                    self.raw, x as i32, y as i32, z as i32,
                )
            },
            "set Metal kernel grid",
        )
    }

    pub(super) fn threadgroup(&self, x: usize, y: usize, z: usize) -> Result<(), MlxError> {
        status(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                    self.raw, x as i32, y as i32, z as i32,
                )
            },
            "set Metal kernel threadgroup",
        )
    }

    pub(super) fn template_dtype(&self, name: &str, dtype: Dtype) -> Result<(), MlxError> {
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

    pub(super) fn template_int(&self, name: &str, value: usize) -> Result<(), MlxError> {
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

    pub(super) fn template_bool(&self, name: &str, value: bool) -> Result<(), MlxError> {
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

pub(super) fn scoped_execution_stream() -> Result<Stream, MlxError> {
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
