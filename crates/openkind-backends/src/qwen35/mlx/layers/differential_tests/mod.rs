//! Synthetic full-width differential test: the MLX layer forward against
//! an independently transcribed host FP32 reference of the Candle
//! oracle's math. No checkpoint is loaded; weights come from a fixed
//! seeded generator, so any divergence is a transcription bug in this
//! port, not weight loading.

mod attention_tests;
mod host_reference;
mod linear_tests;
