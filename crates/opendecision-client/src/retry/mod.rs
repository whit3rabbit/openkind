//! Retry policy with exponential backoff and `Retry-After` support,
//! mirroring the TypeSafe Python SDK's tenacity-based `RetryPolicy` defaults:
//! 2 retries, 0.5s initial backoff doubling to a 5s cap, 25% subtractive
//! jitter, retrying 408/429/5xx (including the 529 overload status) plus
//! connection and timeout errors, within a 30s total budget per call.

mod policy;
#[cfg(test)]
mod tests;

pub use policy::{RetryPolicy, RetryPredicate, DEFAULT_RETRY_STATUSES};
