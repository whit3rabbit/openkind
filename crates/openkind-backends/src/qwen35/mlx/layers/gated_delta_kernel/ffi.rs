//! C FFI bindings and helper wrappers for MLX Metal custom kernels.
//! Re-exports the shared Metal kernel abstractions from [`super::super::metal_kernel`].

pub(super) use super::super::metal_kernel::{
    scoped_execution_stream, MetalKernel, MetalKernelConfig,
};
