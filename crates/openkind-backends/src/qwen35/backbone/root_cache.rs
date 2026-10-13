//! Cross-request cache of the shared-root prefill.
//!
//! Every native request prefills the same root (the rendered state) before it
//! branches into questions and candidates, and the root is the largest single
//! forward of a typical request. Repeated requests over one state therefore
//! recompute identical activations. This cache keeps the root's final-token
//! feature and continuation state, keyed on the exact root token ids, so a
//! later request can branch from it directly.
//!
//! The cache is off unless an operator sets a byte budget. A hit is visible
//! to the caller as lower latency, and the Jev wire request carries no tenant
//! to partition on, so enabling the cache lets any caller learn whether a
//! recent request used the same state. Entries also keep activations derived
//! from that state in memory until they expire. Nothing is written to disk
//! and the key is never logged (digests of low-entropy inputs can be
//! guessed). A hit returns a [`BranchableState::fork_one`] copy, so a request
//! can never mutate a cached state, and the hit path must reproduce what a
//! miss computes bit for bit.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use openkind_runtime::branch::BranchableState;
use sha2::{Digest, Sha256};

use super::nested::{BatchContinuation, SequentialNestedExecutor};
use crate::qwen35::{ExecutionControl, Qwen35Error};

/// Byte budget and lifetime of a [`RootPrefixCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootCacheConfig {
    /// Ceiling on the continuation tensor payload held by cached roots. Zero
    /// disables the cache.
    pub max_tensor_bytes: usize,
    /// How long one entry may be served after it was inserted. Zero disables
    /// the cache.
    pub ttl: Duration,
}

impl RootCacheConfig {
    /// No caching, the default.
    pub const DISABLED: Self = Self {
        max_tensor_bytes: 0,
        ttl: Duration::ZERO,
    };

    /// Whether this configuration can ever hold an entry.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.max_tensor_bytes > 0 && !self.ttl.is_zero()
    }
}

impl Default for RootCacheConfig {
    fn default() -> Self {
        Self::DISABLED
    }
}

type RootDigest = [u8; 32];

struct Entry<S> {
    /// The exact ids the entry was computed from. A digest match alone never
    /// serves an entry, so a hash collision cannot return another root.
    ids: Box<[u32]>,
    feature: Vec<f32>,
    state: Arc<S>,
    tensor_bytes: usize,
    inserted_at: Instant,
    last_used: u64,
}

struct Inner<S> {
    entries: HashMap<RootDigest, Entry<S>>,
    tensor_bytes: usize,
    clock: u64,
}

/// Process-local, byte-bounded LRU of shared-root prefills with a TTL.
pub struct RootPrefixCache<S> {
    inner: Mutex<Inner<S>>,
    config: RootCacheConfig,
}

fn digest(ids: &[u32]) -> RootDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"openkind.qwen35.root-prefix.v1");
    hasher.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    hasher.finalize().into()
}

impl<S: BranchableState> RootPrefixCache<S> {
    /// Create a cache. A disabled configuration never stores or serves.
    #[must_use]
    pub fn new(config: RootCacheConfig) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                tensor_bytes: 0,
                clock: 0,
            }),
            config,
        }
    }

    /// The configuration this cache was built with.
    #[must_use]
    pub const fn config(&self) -> RootCacheConfig {
        self.config
    }

    /// Entry count and cached tensor payload bytes after dropping expired
    /// entries.
    #[must_use]
    pub fn usage(&self) -> (usize, usize) {
        let Ok(mut inner) = self.inner.lock() else {
            return (0, 0);
        };
        purge_expired(&mut inner, Instant::now(), self.config.ttl);
        (inner.entries.len(), inner.tensor_bytes)
    }

    /// An independent copy of the cached root for exactly `ids`, if one is
    /// held and unexpired.
    ///
    /// # Errors
    /// Returns an error if the cached state cannot be forked.
    pub fn get(&self, ids: &[u32]) -> Result<Option<(Vec<f32>, S)>, Qwen35Error> {
        if !self.config.is_enabled() {
            return Ok(None);
        }
        let key = digest(ids);
        let hit = {
            // A poisoned lock means another request panicked mid-update.
            // Serve nothing rather than trust the entries.
            let Ok(mut inner) = self.inner.lock() else {
                return Ok(None);
            };
            purge_expired(&mut inner, Instant::now(), self.config.ttl);
            inner.clock = inner.clock.wrapping_add(1);
            let clock = inner.clock;
            inner
                .entries
                .get_mut(&key)
                .filter(|entry| &*entry.ids == ids)
                .map(|entry| {
                    entry.last_used = clock;
                    (entry.feature.clone(), Arc::clone(&entry.state))
                })
        };
        // Fork outside the lock: state code is arbitrary and may be slow.
        match hit {
            Some((feature, state)) => Ok(Some((feature, state.fork_one()?))),
            None => Ok(None),
        }
    }

    /// Remember the root computed for `ids`, evicting least-recently-used
    /// entries to stay within budget. A state larger than the whole budget is
    /// skipped.
    ///
    /// # Errors
    /// Returns an error if the state cannot be forked into the cache.
    pub fn insert(&self, ids: &[u32], feature: &[f32], state: &S) -> Result<(), Qwen35Error> {
        if !self.config.is_enabled() {
            return Ok(());
        }
        let tensor_bytes = state.tensor_storage_bytes();
        if tensor_bytes > self.config.max_tensor_bytes {
            return Ok(());
        }
        // The cache owns a private copy, never the request's state.
        let stored = Arc::new(state.fork_one()?);
        let key = digest(ids);
        let Ok(mut inner) = self.inner.lock() else {
            return Ok(());
        };
        let now = Instant::now();
        purge_expired(&mut inner, now, self.config.ttl);
        if let Some(previous) = inner.entries.remove(&key) {
            inner.tensor_bytes = inner.tensor_bytes.saturating_sub(previous.tensor_bytes);
        }
        let remaining = self.config.max_tensor_bytes - tensor_bytes;
        while inner.tensor_bytes > remaining {
            let Some(victim) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(evicted) = inner.entries.remove(&victim) {
                inner.tensor_bytes = inner.tensor_bytes.saturating_sub(evicted.tensor_bytes);
            }
        }
        inner.clock = inner.clock.wrapping_add(1);
        let last_used = inner.clock;
        inner.tensor_bytes += tensor_bytes;
        inner.entries.insert(
            key,
            Entry {
                ids: ids.into(),
                feature: feature.to_vec(),
                state: stored,
                tensor_bytes,
                inserted_at: now,
                last_used,
            },
        );
        Ok(())
    }
}

fn purge_expired<S>(inner: &mut Inner<S>, now: Instant, ttl: Duration) {
    inner.entries.retain(|_, entry| {
        // Comparing an age avoids overflow for TTLs beyond the clock's range.
        let keep = now.saturating_duration_since(entry.inserted_at) < ttl;
        if !keep {
            inner.tensor_bytes = inner.tensor_bytes.saturating_sub(entry.tensor_bytes);
        }
        keep
    });
}

/// Executor adapter that serves the exact root prefill from a
/// [`RootPrefixCache`] and fills it on a miss.
///
/// Only a prefill of exactly `root_ids` is cached. The repeated-full strategy
/// prefills longer sequences, which never match, so it runs unchanged.
pub(crate) struct RootCachingExecutor<'a, E: SequentialNestedExecutor> {
    inner: &'a E,
    cache: &'a RootPrefixCache<E::State>,
    root_ids: &'a [u32],
}

impl<'a, E: SequentialNestedExecutor> RootCachingExecutor<'a, E> {
    pub(crate) const fn new(
        inner: &'a E,
        cache: &'a RootPrefixCache<E::State>,
        root_ids: &'a [u32],
    ) -> Self {
        Self {
            inner,
            cache,
            root_ids,
        }
    }

    fn prefill_through_cache(
        &self,
        input_ids: &[u32],
        compute: impl FnOnce() -> Result<(Vec<f32>, E::State), Qwen35Error>,
    ) -> Result<(Vec<f32>, E::State), Qwen35Error> {
        if input_ids != self.root_ids || !self.cache.config().is_enabled() {
            return compute();
        }
        if let Some(hit) = self.cache.get(input_ids)? {
            metrics::counter!("openkind_native_root_cache_total", "result" => "hit").increment(1);
            return Ok(hit);
        }
        metrics::counter!("openkind_native_root_cache_total", "result" => "miss").increment(1);
        let (feature, state) = compute()?;
        // A failed insert only costs a later hit. The request already has its
        // answer, so it must not fail on the cache.
        if let Err(error) = self.cache.insert(input_ids, &feature, &state) {
            tracing::warn!(%error, "root prefix cache could not store a root");
        } else {
            let (_, bytes) = self.cache.usage();
            metrics::gauge!("openkind_native_root_cache_tensor_bytes").set(bytes as f64);
        }
        Ok((feature, state))
    }
}

impl<E: SequentialNestedExecutor> SequentialNestedExecutor for RootCachingExecutor<'_, E> {
    type State = E::State;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.prefill_through_cache(input_ids, || self.inner.prefill(input_ids))
    }

    fn prefill_controlled(
        &self,
        input_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        control.check()?;
        let output = self.prefill_through_cache(input_ids, || {
            self.inner.prefill_controlled(input_ids, control)
        })?;
        control.check()?;
        Ok(output)
    }

    fn continue_from(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.inner.continue_from(state, suffix_ids)
    }

    fn continue_from_controlled(
        &self,
        state: &Self::State,
        suffix_ids: &[u32],
        control: &ExecutionControl,
    ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
        self.inner
            .continue_from_controlled(state, suffix_ids, control)
    }

    fn continue_batch_from(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        self.inner.continue_batch_from(states, suffix_ids)
    }

    fn continue_batch_from_controlled(
        &self,
        states: &[&Self::State],
        suffix_ids: &[&[u32]],
        control: &ExecutionControl,
    ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
        self.inner
            .continue_batch_from_controlled(states, suffix_ids, control)
    }
}

#[cfg(test)]
mod digest_guard_tests {
    use super::*;
    use crate::qwen35::backbone::test_support::{SyntheticExecutor, ROOT_IDS};

    /// A digest match alone must not serve an entry. Forge a collision by
    /// filing one root's entry under another root's digest, which no real
    /// SHA-256 input could produce, and require a miss.
    #[test]
    fn a_digest_collision_never_serves_a_different_root() {
        let cache = RootPrefixCache::new(RootCacheConfig {
            max_tensor_bytes: usize::MAX / 2,
            ttl: Duration::from_secs(60),
        });
        let (feature, state) = SyntheticExecutor.prefill(ROOT_IDS).expect("prefill");
        cache.insert(ROOT_IDS, &feature, &state).expect("insert");
        let other = [7_u32, 8, 9];
        {
            let mut inner = cache.inner.lock().expect("lock");
            let entry = inner.entries.remove(&digest(ROOT_IDS)).expect("entry");
            inner.entries.insert(digest(&other), entry);
        }
        assert!(
            cache.get(&other).expect("get").is_none(),
            "a colliding digest with different ids must miss"
        );
    }
}
