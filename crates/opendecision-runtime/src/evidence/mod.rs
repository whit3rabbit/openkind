//! Backend-neutral native-run evidence artifacts.
//!
//! One schema, `opendecision-native-run/v1`, records what a native execution
//! harness ran and produced: sanitized invocation, machine environment,
//! profile/backend/execution identity, optional parity/performance/memory
//! reports, row-level outputs, and checksums over every sibling file. The
//! writer lives in the runtime — not in any model backend — so the Candle CPU
//! path and a future accelerated backend emit byte-comparable evidence from
//! the same schema and can be compared mechanically instead of through
//! ad-hoc benchmark formats.
//!
//! Sanitization is part of the contract: the invocation record is structured
//! (harness name plus explicitly non-sensitive parameters), never raw `argv`,
//! and `contains_input_content` / `contains_sensitive_paths` flags on
//! [`RunRecord`] state whether anything input- or path-derived was written.

mod records;
mod time;
mod writer;

#[cfg(test)]
mod tests;

pub use records::{
    BackendRecord, ChecksumsRecord, EvidenceError, ExecutionRecord, ProfileRecord, RunEnvironment,
    RunRecord, SanitizedInvocation, CHECKSUMS_SCHEMA, NATIVE_RUN_SCHEMA,
};
pub use time::{format_utc_timestamp, generate_run_id};
pub use writer::NativeRunWriter;
