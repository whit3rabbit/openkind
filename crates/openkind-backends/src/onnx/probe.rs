//! Check runtime initialization and execution without a checkpoint or filesystem graph.
use super::*;

/// Run the vendored tiny Gemm graph using exactly the requested provider.
///
/// # Errors
/// A missing library, provider, or incorrect result fails the probe. Run in
/// an isolated process because vendor initialization is outside Rust's control.
pub fn probe_runtime(acceleration: OnnxAcceleration) -> Result<(), OnnxError> {
    ensure_environment(&OnnxRuntimeSettings::system())?;
    let mut builder = ort::session::Session::builder()?.with_intra_threads(1)?;
    builder =
        match acceleration {
            OnnxAcceleration::Cpu => builder
                .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])?,
            #[cfg(feature = "onnx-cuda")]
            OnnxAcceleration::Cuda { device_id } => builder
                .with_execution_providers([ort::ep::CUDA::default()
                    .with_device_id(i32::try_from(device_id).map_err(|_| {
                        OnnxError::CudaProvider("device ordinal exceeds i32".into())
                    })?)
                    .build()
                    .error_on_failure()])
                .map_err(|error| OnnxError::CudaProvider(error.to_string()))?
                .with_config_entry("session.disable_cpu_ep_fallback", "1")?,
            #[cfg(feature = "onnx-rocm")]
            OnnxAcceleration::Rocm { device_id } => builder
                .with_execution_providers([ort::ep::ROCm::default()
                    .with_device_id(i32::try_from(device_id).map_err(|_| {
                        OnnxError::RocmProvider("device ordinal exceeds i32".into())
                    })?)
                    .build()
                    .error_on_failure()])
                .map_err(|error| OnnxError::RocmProvider(error.to_string()))?
                .with_config_entry("session.disable_cpu_ep_fallback", "1")?,
        };
    let mut session = builder.commit_from_memory(&proto::tiny_gemm_model())?;
    let input = ort::value::Tensor::from_array(([1usize, 2], vec![1.0f32, 2.0]))?;
    let outputs = session.run(ort::inputs!["x" => input])?;
    let (_, values) = outputs["y"].try_extract_tensor::<f32>()?;
    if values != [9.0, 12.0, 15.0] {
        return Err(OnnxError::Runtime(
            "tiny Gemm probe produced invalid arithmetic".into(),
        ));
    }
    Ok(())
}

/// Minimal protobuf wire-format writer for the hand-encoded test graph.
pub(super) mod proto {
    pub(crate) fn push_varint(buffer: &mut Vec<u8>, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                buffer.push(byte);
                break;
            }
            buffer.push(byte | 0x80);
        }
    }

    pub(crate) fn push_tag(buffer: &mut Vec<u8>, field: u32, wire_type: u8) {
        push_varint(buffer, ((u64::from(field)) << 3) | u64::from(wire_type));
    }

    pub(crate) fn push_len_delimited(buffer: &mut Vec<u8>, field: u32, payload: &[u8]) {
        push_tag(buffer, field, 2);
        push_varint(buffer, payload.len() as u64);
        buffer.extend_from_slice(payload);
    }

    pub(crate) fn push_string(buffer: &mut Vec<u8>, field: u32, value: &str) {
        push_len_delimited(buffer, field, value.as_bytes());
    }

    pub(crate) fn push_int64(buffer: &mut Vec<u8>, field: u32, value: i64) {
        push_tag(buffer, field, 0);
        push_varint(buffer, value as u64);
    }

    pub(crate) fn push_packed_int64(buffer: &mut Vec<u8>, field: u32, values: &[i64]) {
        let mut packed = Vec::new();
        for &value in values {
            push_varint(&mut packed, value as u64);
        }
        push_len_delimited(buffer, field, &packed);
    }

    pub(crate) fn push_packed_float(buffer: &mut Vec<u8>, field: u32, values: &[f32]) {
        let mut packed = Vec::new();
        for &value in values {
            packed.extend_from_slice(&value.to_le_bytes());
        }
        push_len_delimited(buffer, field, &packed);
    }

    fn tensor_type_message(elem_type: i32, shape: &[i64]) -> Vec<u8> {
        // TypeProto.tensor_type = 1
        let mut tensor = Vec::new();
        push_int64(&mut tensor, 1, i64::from(elem_type));
        // TensorShapeProto.dim = 1 (repeated Dimension{dim_value=1}). A
        // negative entry encodes a dynamic dimension as an empty Dimension
        // message, the on-wire form ONNX Runtime reports back as -1.
        let mut shape_message = Vec::new();
        for &dim in shape {
            let mut dimension = Vec::new();
            if dim >= 0 {
                push_int64(&mut dimension, 1, dim);
            }
            push_len_delimited(&mut shape_message, 1, &dimension);
        }
        push_len_delimited(&mut tensor, 2, &shape_message);
        let mut type_proto = Vec::new();
        push_len_delimited(&mut type_proto, 1, &tensor);
        type_proto
    }

    fn value_info_message(name: &str, elem_type: i32, shape: &[i64]) -> Vec<u8> {
        let mut message = Vec::new();
        push_string(&mut message, 1, name);
        push_len_delimited(&mut message, 2, &tensor_type_message(elem_type, shape));
        message
    }

    fn initializer_message(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
        let mut message = Vec::new();
        push_packed_int64(&mut message, 1, dims);
        // data_type = 1 (FLOAT)
        push_int64(&mut message, 2, 1);
        push_packed_float(&mut message, 4, data);
        push_string(&mut message, 8, name);
        message
    }

    /// Encode `y = x @ w + b` over float32 tensors:
    /// `x` input `[1, 2]`, `w` initializer `[2, 3]`, `b` initializer
    /// `[3]`, `y` output `[1, 3]`.
    pub(crate) fn tiny_gemm_model() -> Vec<u8> {
        tiny_gemm_model_shaped(&[1, 2], &[1, 3], false)
    }

    /// Same Gemm graph with caller-chosen declared shapes; a `-1` dimension
    /// is exported as dynamic.
    pub(crate) fn tiny_gemm_model_shaped(
        input_shape: &[i64],
        output_shape: &[i64],
        extra_input: bool,
    ) -> Vec<u8> {
        let mut graph = Vec::new();
        // GraphProto.node = 1: Gemm(x, w, b) -> y
        let mut node = Vec::new();
        push_string(&mut node, 1, "x");
        push_string(&mut node, 1, "w");
        push_string(&mut node, 1, "b");
        push_string(&mut node, 2, "y");
        push_string(&mut node, 4, "Gemm");
        push_string(&mut node, 3, "gemm_node");
        push_len_delimited(&mut graph, 1, &node);
        // GraphProto.name = 2
        push_string(&mut graph, 2, "tiny_gemm");
        // GraphProto.initializer = 5
        push_len_delimited(
            &mut graph,
            5,
            &initializer_message("w", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        push_len_delimited(
            &mut graph,
            5,
            &initializer_message("b", &[3], &[0.0, 0.0, 0.0]),
        );
        // GraphProto.input = 11
        push_len_delimited(&mut graph, 11, &value_info_message("x", 1, input_shape));
        if extra_input {
            push_len_delimited(&mut graph, 11, &value_info_message("extra", 1, &[1]));
        }
        // GraphProto.output = 12
        push_len_delimited(&mut graph, 12, &value_info_message("y", 1, output_shape));

        let mut model = Vec::new();
        // ModelProto.ir_version = 1 (IR 8 = ONNX 1.13)
        push_int64(&mut model, 1, 8);
        // ModelProto.graph = 7
        push_len_delimited(&mut model, 7, &graph);
        // ModelProto.opset_import = 8: {domain: "", version: 13}
        let mut opset = Vec::new();
        push_string(&mut opset, 1, "");
        push_int64(&mut opset, 2, 13);
        push_len_delimited(&mut model, 8, &opset);
        model
    }
}
