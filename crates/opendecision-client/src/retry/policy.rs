use std::time::Duration;

use crate::error::Error;

/// Default statuses retried in addition to all 5xx: request timeout and rate
/// limit (529 is covered by the 5xx rule).
pub const DEFAULT_RETRY_STATUSES: [u16; 2] = [408, 429];

/// Extra retry opt-in mirroring the Python SDK's `predicate` / `exceptions`
/// options: called with every error; returning `true` retries it even when
/// the built-in rules would not.
pub type RetryPredicate = std::sync::Arc<dyn Fn(&Error) -> bool + Send + Sync>;

/// Configuration for client retry behavior.
///
/// # Examples
/// ```
/// use std::time::Duration;
/// use opendecision_client::RetryPolicy;
///
/// // Three quick retries, honoring the server's Retry-After.
/// let policy = RetryPolicy::new()
///     .max_retries(3)
///     .backoff_initial(Duration::from_millis(100))
///     .backoff_max(Duration::from_secs(2));
/// ```
#[derive(Clone)]
pub struct RetryPolicy {
    pub(crate) max_retries: u32,
    pub(crate) backoff_initial: Duration,
    pub(crate) backoff_max: Duration,
    pub(crate) backoff_jitter: f64,
    pub(crate) retry_statuses: Vec<u16>,
    pub(crate) retry_server_errors: bool,
    pub(crate) respect_retry_after: bool,
    pub(crate) retry_connection_errors: bool,
    pub(crate) retry_timeout_errors: bool,
    pub(crate) total_timeout: Option<Duration>,
    pub(crate) retry_predicate: Option<RetryPredicate>,
}

// Manual impl: the predicate closure is not `Debug` (it may capture secrets);
// everything else is.
impl std::fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_retries", &self.max_retries)
            .field("backoff_initial", &self.backoff_initial)
            .field("backoff_max", &self.backoff_max)
            .field("backoff_jitter", &self.backoff_jitter)
            .field("retry_statuses", &self.retry_statuses)
            .field("retry_server_errors", &self.retry_server_errors)
            .field("respect_retry_after", &self.respect_retry_after)
            .field("retry_connection_errors", &self.retry_connection_errors)
            .field("retry_timeout_errors", &self.retry_timeout_errors)
            .field("total_timeout", &self.total_timeout)
            .field("retry_predicate", &self.retry_predicate.is_some())
            .finish()
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            backoff_initial: Duration::from_millis(500),
            backoff_max: Duration::from_secs(5),
            backoff_jitter: 0.25,
            retry_statuses: DEFAULT_RETRY_STATUSES.to_vec(),
            retry_server_errors: true,
            respect_retry_after: true,
            retry_connection_errors: true,
            retry_timeout_errors: true,
            total_timeout: Some(Duration::from_secs(30)),
            retry_predicate: None,
        }
    }
}

impl RetryPolicy {
    /// A policy with the SDK-default settings (same as [`RetryPolicy::default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum retries after the initial attempt; `0` disables retries.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// First backoff delay, doubled each attempt up to [`backoff_max`](Self::backoff_max).
    /// Zero disables backoff delays entirely.
    pub fn backoff_initial(mut self, initial: Duration) -> Self {
        self.backoff_initial = initial;
        self
    }

    /// Upper bound on exponential backoff delays. Zero disables backoff delays entirely.
    pub fn backoff_max(mut self, max: Duration) -> Self {
        self.backoff_max = max;
        self
    }

    /// Fraction of each backoff delay randomly subtracted, in `0.0..=1.0`.
    /// `0.0` makes delays fully deterministic (useful in tests).
    pub fn backoff_jitter(mut self, jitter: f64) -> Self {
        self.backoff_jitter = jitter;
        self
    }

    /// Retry these status codes (in addition to 5xx while
    /// [`retry_server_errors`](Self::retry_server_errors) is set). The
    /// default retries 408 and 429.
    pub fn retry_status(mut self, status: u16) -> Self {
        if !self.retry_statuses.contains(&status) {
            self.retry_statuses.push(status);
        }
        self
    }

    /// Replace the retried-status list wholesale, mirroring the Python SDK's
    /// `http_statuses=[...]` semantics: only these codes (plus 5xx while
    /// [`retry_server_errors`](Self::retry_server_errors) is set) are retried.
    ///
    /// `RetryPolicy::new().with_retry_statuses([409]).retry_server_errors(false)`
    /// retries 409 and nothing else.
    pub fn with_retry_statuses(mut self, statuses: impl IntoIterator<Item = u16>) -> Self {
        self.retry_statuses = statuses.into_iter().collect();
        self
    }

    /// Opt extra errors into retrying, mirroring the Python SDK's
    /// `predicate`/`exceptions` options: `Some(predicate)` retries any error
    /// for which the predicate returns `true`, even non-transient statuses
    /// like 404. `None` (the default) uses only the built-in rules.
    pub fn retry_predicate(mut self, predicate: Option<RetryPredicate>) -> Self {
        self.retry_predicate = predicate;
        self
    }

    /// Whether any 5xx status (including 529 Overloaded) is retried. Default `true`.
    pub fn retry_server_errors(mut self, retry: bool) -> Self {
        self.retry_server_errors = retry;
        self
    }

    /// Whether to honor `Retry-After` / `retry-after-ms` response headers
    /// instead of the computed backoff. Default `true`.
    pub fn respect_retry_after(mut self, respect: bool) -> Self {
        self.respect_retry_after = respect;
        self
    }

    /// Whether connection failures are retried. Default `true`.
    pub fn retry_connection_errors(mut self, retry: bool) -> Self {
        self.retry_connection_errors = retry;
        self
    }

    /// Whether per-attempt timeouts are retried. Default `true`.
    pub fn retry_timeout_errors(mut self, retry: bool) -> Self {
        self.retry_timeout_errors = retry;
        self
    }

    /// Total budget per call, including the initial attempt and all delays.
    /// A retry whose delay would reach or exceed the remaining budget is not
    /// attempted; the last error is returned instead. `None` disables the limit.
    pub fn total_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.total_timeout = timeout;
        self
    }

    /// Validate jitter bounds. Called when the client is built so misconfig
    /// surfaces at construction, not mid-call.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !(0.0..=1.0).contains(&self.backoff_jitter) || !self.backoff_jitter.is_finite() {
            return Err(Error::Config(format!(
                "backoff_jitter must be between 0.0 and 1.0, got {}",
                self.backoff_jitter
            )));
        }
        Ok(())
    }

    /// Whether this error should be retried under the policy.
    pub fn is_retryable(&self, error: &Error) -> bool {
        let builtin = match error {
            Error::Connection(_) => self.retry_connection_errors,
            Error::Timeout { .. } => self.retry_timeout_errors,
            Error::Api(api) => {
                self.retry_server_errors && (500..600).contains(&api.status)
                    || self.retry_statuses.contains(&api.status)
            }
            // Configuration and decode failures are deterministic; retrying
            // cannot change the outcome.
            Error::Config(_) | Error::Decode { .. } | Error::ResponseTooLarge { .. } => false,
        };
        builtin || self.retry_predicate.as_ref().is_some_and(|f| f(error))
    }

    /// How long to wait before the retry numbered `retry_index`
    /// (0-based: `0` is the wait before the first retry).
    ///
    /// A server-requested `Retry-After` takes precedence over computed
    /// backoff when [`respect_retry_after`](Self::respect_retry_after) is set.
    pub fn delay_for(&self, error: &Error, retry_index: u32) -> Duration {
        if self.respect_retry_after {
            if let Some(delay) = error.retry_after() {
                return delay;
            }
        }
        self.backoff_delay(retry_index)
    }

    /// Exponential backoff with subtractive jitter:
    /// `min(backoff_max, backoff_initial * 2^retry_index) * (1 - U * jitter)`.
    ///
    /// Nanosecond precision throughout, so sub-millisecond initial delays
    /// and very large caps behave like the Python SDK's float math.
    pub fn backoff_delay(&self, retry_index: u32) -> Duration {
        if self.backoff_initial.is_zero() || self.backoff_max.is_zero() {
            return Duration::ZERO;
        }
        let initial_ns = self.backoff_initial.as_nanos() as u64;
        let exponential = initial_ns.saturating_mul(1u64 << retry_index.min(63));
        let capped = exponential.min(self.backoff_max.as_nanos() as u64);
        let jitter_ns = (capped as f64 * self.backoff_jitter * fastrand::f64()) as u64;
        Duration::from_nanos(capped.saturating_sub(jitter_ns))
    }
}
