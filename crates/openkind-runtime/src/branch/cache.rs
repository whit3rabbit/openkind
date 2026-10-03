//! Tenant-isolated, byte-bounded in-process branch-state reuse.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

use super::{BranchableState, ContentFingerprint, StateError};

/// Persistent-reuse key that can only be built from a strict content fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCacheKey {
    tenant: String,
    content: ContentFingerprint,
}

impl StateCacheKey {
    /// Build a tenant-scoped key. Empty tenant identifiers are rejected.
    pub fn new(tenant: impl Into<String>, content: ContentFingerprint) -> Result<Self, CacheError> {
        let tenant = tenant.into();
        if tenant.trim().is_empty() {
            return Err(CacheError::EmptyTenant);
        }
        Ok(Self { tenant, content })
    }

    /// Tenant namespace owning this entry.
    #[must_use]
    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    /// Strict content identity of the cached state.
    #[must_use]
    pub const fn content_fingerprint(&self) -> ContentFingerprint {
        self.content
    }
}

impl Hash for StateCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tenant.hash(state);
        self.content.hash(state);
    }
}

/// Cache admission failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CacheError {
    /// Tenant namespaces must be explicit.
    #[error("state cache tenant must not be empty")]
    EmptyTenant,
    /// One state exceeds the entire tensor-payload budget.
    #[error("state tensor payload {state_bytes} exceeds cache budget {max_bytes}")]
    StateTooLarge {
        /// State tensor payload bytes.
        state_bytes: usize,
        /// Configured cache budget.
        max_bytes: usize,
    },
    /// A previous thread poisoned the cache lock.
    #[error("state cache lock poisoned")]
    Poisoned,
    /// The cached state could not be forked into an independent reader state.
    #[error("failed to fork cached state: {0}")]
    State(#[from] StateError),
}

struct Entry<S> {
    state: Arc<S>,
    tensor_bytes: usize,
    inserted_at: Instant,
    last_used: u64,
}

struct CacheInner<S> {
    entries: HashMap<StateCacheKey, Entry<S>>,
    tensor_bytes: usize,
    clock: u64,
}

/// In-process branch-state cache with strict tenant isolation, TTL, and LRU eviction.
///
/// The byte budget covers continuation tensor payload only. Process-level admission
/// must separately account for model weights, scratch, allocator overhead, and RSS.
pub struct BranchStateCache<S> {
    inner: Mutex<CacheInner<S>>,
    max_tensor_bytes: usize,
    ttl: Duration,
}

impl<S: BranchableState> BranchStateCache<S> {
    /// Create a cache. A zero byte budget admits no states; a zero TTL expires immediately.
    #[must_use]
    pub fn new(max_tensor_bytes: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                tensor_bytes: 0,
                clock: 0,
            }),
            max_tensor_bytes,
            ttl,
        }
    }

    /// Insert or replace one tenant-scoped state, evicting least-recently-used entries.
    pub fn insert(&self, key: StateCacheKey, state: S) -> Result<(), CacheError> {
        let state_bytes = state.tensor_storage_bytes();
        if state_bytes > self.max_tensor_bytes {
            return Err(CacheError::StateTooLarge {
                state_bytes,
                max_bytes: self.max_tensor_bytes,
            });
        }
        if self.max_tensor_bytes == 0 || self.ttl.is_zero() {
            return Ok(());
        }
        let mut inner = self.inner.lock().map_err(|_| CacheError::Poisoned)?;
        let now = Instant::now();
        purge_expired(&mut inner, now, self.ttl);
        if let Some(previous) = inner.entries.remove(&key) {
            inner.tensor_bytes = inner.tensor_bytes.saturating_sub(previous.tensor_bytes);
        }
        // Evict before adding so saturation cannot conceal an over-budget total.
        let remaining_budget = self.max_tensor_bytes - state_bytes;
        while inner.tensor_bytes > remaining_budget {
            let Some(eviction_key) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(evicted) = inner.entries.remove(&eviction_key) {
                inner.tensor_bytes = inner.tensor_bytes.saturating_sub(evicted.tensor_bytes);
            }
        }
        inner.clock = inner.clock.wrapping_add(1);
        let entry = Entry {
            state: Arc::new(state),
            tensor_bytes: state_bytes,
            inserted_at: now,
            last_used: inner.clock,
        };
        inner.tensor_bytes += state_bytes;
        inner.entries.insert(key, entry);
        Ok(())
    }

    /// Fork one unexpired state into an independent reader state from its tenant namespace.
    pub fn get(&self, key: &StateCacheKey) -> Result<Option<S>, CacheError> {
        let state = {
            let mut inner = self.inner.lock().map_err(|_| CacheError::Poisoned)?;
            purge_expired(&mut inner, Instant::now(), self.ttl);
            inner.clock = inner.clock.wrapping_add(1);
            let clock = inner.clock;
            inner.entries.get_mut(key).map(|entry| {
                entry.last_used = clock;
                Arc::clone(&entry.state)
            })
        };
        // State implementations may re-enter this cache while forking. Keep
        // arbitrary trait code outside the shared cache mutex.
        state
            .map(|state| state.fork_one().map(Some))
            .unwrap_or(Ok(None))
            .map_err(CacheError::State)
    }

    /// Current entry count and exact cached tensor payload bytes.
    pub fn usage(&self) -> Result<(usize, usize), CacheError> {
        let mut inner = self.inner.lock().map_err(|_| CacheError::Poisoned)?;
        purge_expired(&mut inner, Instant::now(), self.ttl);
        Ok((inner.entries.len(), inner.tensor_bytes))
    }
}

fn purge_expired<S>(inner: &mut CacheInner<S>, now: Instant, ttl: Duration) {
    inner.entries.retain(|_, entry| {
        // Comparing age avoids overflow for TTLs beyond Instant's representable range.
        let retain = now.saturating_duration_since(entry.inserted_at) < ttl;
        if !retain {
            inner.tensor_bytes = inner.tensor_bytes.saturating_sub(entry.tensor_bytes);
        }
        retain
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch::{BranchBatch, ProfileId, SchedulingFingerprint, StateError};
    use std::sync::Arc;

    #[derive(Clone)]
    struct DummyState {
        profile: ProfileId,
        bytes: usize,
        value: Arc<Mutex<usize>>,
        fork_error: Option<StateError>,
    }

    #[derive(Clone)]
    struct DummyBatch(Vec<DummyState>);

    impl BranchableState for DummyState {
        type Batch = DummyBatch;

        fn profile_id(&self) -> &ProfileId {
            &self.profile
        }

        fn position(&self) -> usize {
            0
        }

        fn tensor_storage_bytes(&self) -> usize {
            self.bytes
        }

        fn scheduling_fingerprint(&self) -> SchedulingFingerprint {
            SchedulingFingerprint::builder("dummy").finish()
        }

        fn fork_one(&self) -> Result<Self, StateError> {
            if let Some(error) = &self.fork_error {
                return Err(error.clone());
            }
            Ok(Self {
                profile: self.profile.clone(),
                bytes: self.bytes,
                value: Arc::new(Mutex::new(*self.value.lock().expect("value"))),
                fork_error: None,
            })
        }

        fn fork_batch(&self, lanes: usize) -> Result<Self::Batch, StateError> {
            (0..lanes)
                .map(|_| self.fork_one())
                .collect::<Result<Vec<_>, _>>()
                .map(DummyBatch)
        }
    }

    impl BranchBatch for DummyBatch {
        type State = DummyState;

        fn lanes(&self) -> usize {
            self.0.len()
        }

        fn tensor_storage_bytes(&self) -> usize {
            self.0.iter().map(|state| state.bytes).sum()
        }

        fn select(&self, index: usize) -> Result<Self::State, StateError> {
            self.0
                .get(index)
                .ok_or(StateError::LaneIndexOutOfBounds {
                    lanes: self.0.len(),
                    index,
                })?
                .fork_one()
        }

        fn gather(&self, indices: &[usize]) -> Result<Self, StateError> {
            indices
                .iter()
                .map(|&index| self.select(index))
                .collect::<Result<Vec<_>, _>>()
                .map(Self)
        }
    }

    fn state(bytes: usize) -> DummyState {
        DummyState {
            profile: ProfileId::new("profile").expect("profile"),
            bytes,
            value: Arc::new(Mutex::new(0)),
            fork_error: None,
        }
    }

    fn key(tenant: &str, value: u64) -> StateCacheKey {
        StateCacheKey::new(
            tenant,
            ContentFingerprint::builder("cache-test")
                .value(value)
                .finish(),
        )
        .expect("key")
    }

    #[test]
    fn tenant_keys_are_isolated_and_lru_is_byte_bounded() {
        let cache = BranchStateCache::new(20, Duration::from_secs(60));
        cache.insert(key("alpha", 1), state(10)).expect("insert");
        cache.insert(key("beta", 1), state(10)).expect("insert");
        assert!(cache.get(&key("alpha", 1)).expect("get").is_some());
        cache.insert(key("alpha", 2), state(10)).expect("insert");

        assert!(cache.get(&key("alpha", 1)).expect("get").is_some());
        assert!(cache.get(&key("beta", 1)).expect("get").is_none());
        assert_eq!(cache.usage().expect("usage"), (2, 20));
    }

    #[test]
    fn ttl_and_single_state_admission_fail_closed() {
        let cache = BranchStateCache::new(8, Duration::ZERO);
        cache.insert(key("alpha", 1), state(8)).expect("insert");
        assert!(cache.get(&key("alpha", 1)).expect("get").is_none());
        assert_eq!(
            cache.insert(key("alpha", 2), state(9)),
            Err(CacheError::StateTooLarge {
                state_bytes: 9,
                max_bytes: 8,
            })
        );
        assert_eq!(
            StateCacheKey::new("", ContentFingerprint::builder("x").finish()),
            Err(CacheError::EmptyTenant)
        );
    }

    #[test]
    fn cache_evicts_before_the_byte_total_can_overflow() {
        let cache = BranchStateCache::new(usize::MAX, Duration::from_secs(60));
        cache
            .insert(key("alpha", 1), state(usize::MAX))
            .expect("insert");
        cache.insert(key("alpha", 2), state(1)).expect("insert");
        assert!(cache.get(&key("alpha", 1)).expect("get").is_none());
        assert_eq!(cache.usage().expect("usage"), (1, 1));
        cache
            .insert(key("alpha", 3), state(usize::MAX))
            .expect("insert");
        assert!(cache.get(&key("alpha", 2)).expect("get").is_none());
        assert_eq!(cache.usage().expect("usage"), (1, usize::MAX));
    }

    #[test]
    fn unrepresentable_ttl_does_not_panic_or_poison_the_cache() {
        let cache = BranchStateCache::new(8, Duration::MAX);
        cache.insert(key("alpha", 1), state(8)).expect("insert");
        assert!(cache.get(&key("alpha", 1)).expect("get").is_some());
        assert_eq!(cache.usage().expect("usage"), (1, 8));
    }

    #[test]
    fn disabled_cache_does_not_retain_zero_byte_states() {
        let cache = BranchStateCache::new(0, Duration::from_secs(60));
        cache.insert(key("alpha", 1), state(0)).expect("insert");
        assert!(cache.get(&key("alpha", 1)).expect("get").is_none());
        assert_eq!(cache.usage().expect("usage"), (0, 0));
    }

    #[test]
    fn cache_reads_fork_instead_of_sharing_a_shallow_clone() {
        let cache = BranchStateCache::new(8, Duration::from_secs(60));
        cache.insert(key("alpha", 1), state(8)).expect("insert");
        let first = cache.get(&key("alpha", 1)).expect("get").expect("present");
        *first.value.lock().expect("value") = 7;
        let second = cache.get(&key("alpha", 1)).expect("get").expect("present");
        assert_eq!(*second.value.lock().expect("value"), 0);
        assert_eq!(cache.usage().expect("usage"), (1, 8));
    }

    #[test]
    fn fork_failure_is_returned_without_poisoning_the_cache() {
        let cache = BranchStateCache::new(8, Duration::from_secs(60));
        let error = StateError::IdentityMismatch {
            expected: "profile".to_owned(),
            actual: "invalid".to_owned(),
        };
        let mut invalid = state(8);
        invalid.fork_error = Some(error.clone());
        cache.insert(key("alpha", 1), invalid).expect("insert");
        assert!(matches!(
            cache.get(&key("alpha", 1)),
            Err(CacheError::State(actual)) if actual == error
        ));
        assert_eq!(cache.usage().expect("usage after failed fork"), (1, 8));
        cache.insert(key("alpha", 1), state(8)).expect("replace");
        assert!(cache.get(&key("alpha", 1)).expect("get").is_some());
    }

    #[test]
    fn active_reader_clone_survives_concurrent_cache_eviction() {
        let cache = Arc::new(BranchStateCache::new(10, Duration::from_secs(60)));
        cache
            .insert(key("alpha", 1), state(10))
            .expect("insert root");
        let reader_cache = cache.clone();
        let (read_tx, read_rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let active = reader_cache
                .get(&key("alpha", 1))
                .expect("read root")
                .expect("root present");
            read_tx.send(()).expect("signal acquired reader clone");
            std::thread::sleep(Duration::from_millis(20));
            active.tensor_storage_bytes()
        });

        read_rx.recv().expect("reader cloned active state");
        cache
            .insert(key("alpha", 2), state(10))
            .expect("evict cached root");
        assert!(cache
            .get(&key("alpha", 1))
            .expect("lookup evicted root")
            .is_none());
        assert_eq!(reader.join().expect("active reader remains valid"), 10);
    }
}
