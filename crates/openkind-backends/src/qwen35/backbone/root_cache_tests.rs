//! Offline tests for the cross-request root prefix cache, driven by the
//! shared synthetic executor so state content is deterministic.

use std::cell::Cell;
use std::time::Duration;

use openkind_runtime::branch::BranchableState;

use super::model::BackboneState;
use super::nested::{run_sequential_nested, NestedQuestion, SequentialNestedExecutor};
use super::root_cache::{RootCacheConfig, RootCachingExecutor, RootPrefixCache};
use super::strategy::{run_strategy, ExecutionStrategy};
use super::test_support::{plan, SyntheticExecutor, ROOT_IDS};

/// One question with one candidate: the tests below only need a branch, and
/// each synthetic state is tens of megabytes to fingerprint in a debug build.
fn small_plan() -> Vec<NestedQuestion<'static>> {
    vec![NestedQuestion {
        question_ids: &[20, 21],
        candidate_suffix_ids: &[&[30]],
    }]
}
use crate::qwen35::Qwen35Error;

const AMPLE: usize = usize::MAX / 2;

fn config(max_tensor_bytes: usize) -> RootCacheConfig {
    RootCacheConfig {
        max_tensor_bytes,
        ttl: Duration::from_secs(60),
    }
}

/// Synthetic executor that counts prefills and can be told to fail them.
#[derive(Default)]
struct Counting {
    prefills: Cell<usize>,
    fail: Cell<bool>,
}

impl SequentialNestedExecutor for Counting {
    type State = BackboneState;

    fn prefill(&self, input_ids: &[u32]) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        self.prefills.set(self.prefills.get() + 1);
        if self.fail.get() {
            return Err(Qwen35Error::InvalidInput("forced prefill failure".into()));
        }
        SyntheticExecutor.prefill(input_ids)
    }

    fn continue_from(
        &self,
        state: &BackboneState,
        suffix_ids: &[u32],
    ) -> Result<(Vec<f32>, BackboneState), Qwen35Error> {
        SyntheticExecutor.continue_from(state, suffix_ids)
    }
}

fn assert_runs_equal(
    left: &super::nested::NestedRun<BackboneState>,
    right: &super::nested::NestedRun<BackboneState>,
) {
    assert_eq!(left.root_feature(), right.root_feature());
    assert_eq!(
        left.root_state().strict_fingerprint(),
        right.root_state().strict_fingerprint()
    );
    for (a, b) in left.questions().iter().zip(right.questions()) {
        assert_eq!(a.question_feature(), b.question_feature());
        assert_eq!(
            a.question_state().strict_fingerprint(),
            b.question_state().strict_fingerprint()
        );
        for (x, y) in a.candidates().iter().zip(b.candidates()) {
            assert_eq!(x.feature(), y.feature());
            assert_eq!(
                x.state().strict_fingerprint(),
                y.state().strict_fingerprint()
            );
        }
    }
}

#[test]
fn a_hit_reproduces_the_miss_and_skips_the_root_forward() {
    let inner = Counting::default();
    let cache = RootPrefixCache::new(config(AMPLE));
    let uncached =
        run_sequential_nested(&SyntheticExecutor, ROOT_IDS, &small_plan()).expect("uncached");

    let caching = RootCachingExecutor::new(&inner, &cache, ROOT_IDS);
    let miss = run_sequential_nested(&caching, ROOT_IDS, &small_plan()).expect("miss");
    assert_eq!(
        inner.prefills.get(),
        1,
        "the first request computes the root"
    );
    assert_eq!(cache.usage().0, 1);
    let hit = run_sequential_nested(&caching, ROOT_IDS, &small_plan()).expect("hit");
    assert_eq!(
        inner.prefills.get(),
        1,
        "the second request reuses the root"
    );

    assert_runs_equal(&miss, &uncached);
    assert_runs_equal(&hit, &uncached);
}

#[test]
fn only_the_exact_root_ids_are_served() {
    let inner = Counting::default();
    let cache = RootPrefixCache::new(config(AMPLE));
    let caching = RootCachingExecutor::new(&inner, &cache, ROOT_IDS);
    run_sequential_nested(&caching, ROOT_IDS, &small_plan()).expect("fill");

    // A shorter, longer, or different root is a different key, whether or not
    // the shared executor is told it is the root.
    for ids in [&ROOT_IDS[..2], &[10, 11, 12, 13][..], &[10, 11, 99][..]] {
        let before = inner.prefills.get();
        let other = RootCachingExecutor::new(&inner, &cache, ids);
        run_sequential_nested(&other, ids, &small_plan()).expect("other root");
        assert_eq!(inner.prefills.get(), before + 1, "ids {ids:?} must miss");
    }
    assert_eq!(cache.usage().0, 4, "each distinct root is its own entry");
}

#[test]
fn prefills_that_are_not_the_root_never_touch_the_cache() {
    let inner = Counting::default();
    let cache = RootPrefixCache::new(config(AMPLE));
    let caching = RootCachingExecutor::new(&inner, &cache, ROOT_IDS);
    // Repeated-full prefills root + question + candidate, never the bare root.
    run_strategy(&caching, ExecutionStrategy::RepeatedFull, ROOT_IDS, &plan())
        .expect("repeated full");
    assert_eq!(cache.usage(), (0, 0));
    assert_eq!(inner.prefills.get(), 3, "one full prefill per candidate");
}

#[test]
fn run_strategy_matches_with_and_without_the_cache() {
    let inner = Counting::default();
    let cache = RootPrefixCache::new(config(AMPLE));
    let caching = RootCachingExecutor::new(&inner, &cache, ROOT_IDS);
    let plain = run_strategy(
        &SyntheticExecutor,
        ExecutionStrategy::NestedSequential,
        ROOT_IDS,
        &plan(),
    )
    .expect("plain");
    for round in 0..2 {
        let cached = run_strategy(
            &caching,
            ExecutionStrategy::NestedSequential,
            ROOT_IDS,
            &plan(),
        )
        .expect("cached");
        assert_eq!(cached.question_features(), plain.question_features());
        assert_eq!(
            cached.retained_tensor_bytes(),
            plain.retained_tensor_bytes()
        );
        for (a, b) in cached.question_states().iter().zip(plain.question_states()) {
            let a: Vec<_> = a.iter().map(BackboneState::strict_fingerprint).collect();
            let b: Vec<_> = b.iter().map(BackboneState::strict_fingerprint).collect();
            assert_eq!(a, b, "round {round}");
        }
    }
    assert_eq!(inner.prefills.get(), 1);
}

#[test]
fn a_returned_state_cannot_change_what_the_cache_serves() {
    let cache = RootPrefixCache::new(config(AMPLE));
    let (feature, state) = SyntheticExecutor.prefill(ROOT_IDS).expect("prefill");
    let pristine = state.strict_fingerprint();
    cache.insert(ROOT_IDS, &feature, &state).expect("insert");

    // Mutating the state the caller kept does not reach the cached copy.
    let mut kept = state;
    kept.position += 5;
    let (_, first) = cache.get(ROOT_IDS).expect("get").expect("hit");
    assert_eq!(first.strict_fingerprint(), pristine);

    // Nor does mutating a state a previous hit returned.
    let mut served = first;
    served.position += 9;
    let (served_feature, second) = cache.get(ROOT_IDS).expect("get").expect("hit");
    assert_eq!(second.strict_fingerprint(), pristine);
    assert_eq!(served_feature, feature);
    assert_eq!(second.lineage().root_id(), kept.lineage().root_id());
}

#[test]
fn a_failed_prefill_is_not_cached() {
    let inner = Counting::default();
    inner.fail.set(true);
    let cache = RootPrefixCache::new(config(AMPLE));
    let caching = RootCachingExecutor::new(&inner, &cache, ROOT_IDS);
    assert!(caching.prefill(ROOT_IDS).is_err());
    assert_eq!(cache.usage(), (0, 0));
    inner.fail.set(false);
    caching.prefill(ROOT_IDS).expect("retry succeeds");
    assert_eq!(cache.usage().0, 1);
}

#[test]
fn a_disabled_cache_never_stores_or_serves() {
    for disabled in [
        RootCacheConfig::DISABLED,
        RootCacheConfig::default(),
        RootCacheConfig {
            max_tensor_bytes: AMPLE,
            ttl: Duration::ZERO,
        },
        RootCacheConfig {
            max_tensor_bytes: 0,
            ttl: Duration::from_secs(60),
        },
    ] {
        assert!(!disabled.is_enabled());
        let cache = RootPrefixCache::new(disabled);
        let (feature, state) = SyntheticExecutor.prefill(ROOT_IDS).expect("prefill");
        cache.insert(ROOT_IDS, &feature, &state).expect("insert");
        assert_eq!(cache.usage(), (0, 0));
        assert!(cache.get(ROOT_IDS).expect("get").is_none());
    }
}

#[test]
fn the_byte_budget_bounds_the_cache_and_evicts_least_recently_used() {
    let (feature, one) = SyntheticExecutor.prefill(ROOT_IDS).expect("prefill");
    let bytes = one.tensor_storage_bytes();
    // Room for exactly two roots of this size.
    let cache = RootPrefixCache::new(config(bytes * 2));
    let ids = |n: u32| [n, n + 1, n + 2];
    let state = |n: u32| SyntheticExecutor.prefill(&ids(n)).expect("prefill").1;
    let first = state(100);
    assert_eq!(first.tensor_storage_bytes(), bytes, "same-length roots");

    cache.insert(&ids(100), &feature, &first).expect("a");
    cache.insert(&ids(200), &feature, &state(200)).expect("b");
    // Touch A so B is the least recently used.
    assert!(cache.get(&ids(100)).expect("get").is_some());
    cache.insert(&ids(300), &feature, &state(300)).expect("c");

    assert_eq!(cache.usage(), (2, bytes * 2));
    assert!(cache.get(&ids(100)).expect("get").is_some(), "A survives");
    assert!(cache.get(&ids(200)).expect("get").is_none(), "B is evicted");
    assert!(cache.get(&ids(300)).expect("get").is_some(), "C is held");

    // A state larger than the whole budget is skipped, not stored.
    let small = RootPrefixCache::new(config(bytes - 1));
    small.insert(ROOT_IDS, &feature, &one).expect("skipped");
    assert_eq!(small.usage(), (0, 0));
}

#[test]
fn entries_expire_after_the_ttl() {
    let cache = RootPrefixCache::new(RootCacheConfig {
        max_tensor_bytes: AMPLE,
        ttl: Duration::from_millis(30),
    });
    let (feature, state) = SyntheticExecutor.prefill(ROOT_IDS).expect("prefill");
    cache.insert(ROOT_IDS, &feature, &state).expect("insert");
    assert!(cache.get(ROOT_IDS).expect("get").is_some());
    std::thread::sleep(Duration::from_millis(120));
    assert!(cache.get(ROOT_IDS).expect("get").is_none());
    assert_eq!(cache.usage(), (0, 0), "expired bytes are released");
}

#[test]
fn reinserting_a_root_replaces_it_without_double_counting() {
    let cache = RootPrefixCache::new(config(AMPLE));
    let (feature, state) = SyntheticExecutor.prefill(ROOT_IDS).expect("prefill");
    cache.insert(ROOT_IDS, &feature, &state).expect("first");
    let once = cache.usage();
    cache.insert(ROOT_IDS, &feature, &state).expect("second");
    assert_eq!(cache.usage(), once);
}

#[test]
fn contains_reports_a_live_root_without_refreshing_it() {
    let cache = RootPrefixCache::new(RootCacheConfig {
        max_tensor_bytes: AMPLE,
        ttl: Duration::from_millis(80),
    });
    assert!(!cache.contains(ROOT_IDS));
    let (feature, state) = SyntheticExecutor.prefill(ROOT_IDS).expect("prefill");
    cache.insert(ROOT_IDS, &feature, &state).expect("insert");
    assert!(cache.contains(ROOT_IDS));
    assert!(!cache.contains(&ROOT_IDS[..2]), "exact ids only");
    // Polling must not extend the lifetime: the entry still expires on schedule.
    for _ in 0..4 {
        std::thread::sleep(Duration::from_millis(30));
        let _ = cache.contains(ROOT_IDS);
    }
    assert!(!cache.contains(ROOT_IDS));
    assert!(!RootPrefixCache::<BackboneState>::new(RootCacheConfig::DISABLED).contains(ROOT_IDS));
}
