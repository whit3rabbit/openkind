//! Native no-thinking decisions for the pinned Jeeves MLX affine-4 checkpoint.
//!
//! Packed weights stay packed. Rust renders the exact empty-chain sequence,
//! executes the hybrid backbone, and maps the FP32 pointer distribution to
//! typed Jev answers. No Python process or generated tokens are involved.

pub mod artifacts;
pub mod renderer;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod engine;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod mixers;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod model;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub use engine::{JeevesEngine, JeevesEngineConfig};

/// Distinct identity for the pinned checkpoint, sorted options, empty chain,
/// and native FP32 arithmetic. It never aliases the frozen 4B profile.
pub const PROFILE_ID: &str = "15fb3b95801f2f039636";
/// Operator-owned MLX conversion used by this profile.
pub const MODEL_ID: &str = "cowWhySo/jeeves-mlx";
/// Immutable 4-bit package revision.
pub const MODEL_REVISION: &str = "6f4220a267a94c5279b3fcbc5499de734b9a55b0";
/// Original fused Jeeves export.
pub const SOURCE_REVISION: &str = "8622b7d1652a9dcb8629486b84dce9e8d690c5cd";
/// Original calibration, retained without a new quality claim.
pub const TEMPERATURE: f64 = 1.8589280843734741;
/// Fail closed rather than truncating a rendered request.
pub const MAX_TOKENS: usize = 4096;
/// Bound the independent question passes retained by one request.
pub const MAX_QUESTIONS: usize = 64;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) const HIDDEN: i32 = 4096;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) const INTERMEDIATE: i32 = 12288;
