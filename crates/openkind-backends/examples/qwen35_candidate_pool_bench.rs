//! Diagnostic FP32 MLX comparison for same-position candidate pooling.
//!
//! Run each strategy in a fresh process. This replays frozen token suffixes,
//! so it measures backbone plus readout time, not HTTP or tokenization.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod supported {
    use std::cell::RefCell;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use openkind_backends::qwen35::mlx::{
        MlxPrecision, MlxQwen35Backbone, MlxRuntime, MlxRuntimeConfig,
    };
    use openkind_backends::qwen35::{
        run_batched_nested, run_batched_nested_pooled, run_sequential_nested, BatchContinuation,
        NestedQuestion, PrimitiveKind, Qwen35Error, ReferenceBundle, SequentialNestedExecutor,
        BACKBONE_REVISION, PROFILE_ID,
    };
    use openkind_runtime::branch::BranchableState;
    use openkind_runtime::{peak_resident_bytes, BatchForwardMode};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct TokenFixtures {
        records: Vec<TokenRecord>,
    }

    #[derive(Clone, Deserialize)]
    struct TokenRecord {
        fixture_case: usize,
        question_id: String,
        root_ids: Vec<u32>,
        question_ids: Vec<u32>,
        candidate_suffix_ids: Vec<Vec<u32>>,
    }

    #[derive(Default)]
    struct StageStats {
        prefill: Duration,
        question: Duration,
        candidate: Duration,
        physical_forwards: usize,
        padding_tokens: usize,
    }

    struct TimedExecutor<'a> {
        inner: &'a MlxQwen35Backbone,
        root_position: usize,
        stats: RefCell<StageStats>,
    }

    impl SequentialNestedExecutor for TimedExecutor<'_> {
        type State = <MlxQwen35Backbone as SequentialNestedExecutor>::State;

        fn prefill(&self, ids: &[u32]) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
            let started = Instant::now();
            let output = SequentialNestedExecutor::prefill(self.inner, ids)?;
            let mut stats = self.stats.borrow_mut();
            stats.prefill += started.elapsed();
            stats.physical_forwards += 1;
            Ok(output)
        }

        fn continue_from(
            &self,
            state: &Self::State,
            ids: &[u32],
        ) -> Result<(Vec<f32>, Self::State), Qwen35Error> {
            let started = Instant::now();
            let output = SequentialNestedExecutor::continue_from(self.inner, state, ids)?;
            let mut stats = self.stats.borrow_mut();
            let elapsed = started.elapsed();
            if state.position() == self.root_position {
                stats.question += elapsed;
            } else {
                stats.candidate += elapsed;
            }
            stats.physical_forwards += 1;
            Ok(output)
        }

        fn continue_batch_from(
            &self,
            states: &[&Self::State],
            suffixes: &[&[u32]],
        ) -> Result<BatchContinuation<Self::State>, Qwen35Error> {
            let started = Instant::now();
            let output =
                SequentialNestedExecutor::continue_batch_from(self.inner, states, suffixes)?;
            let mut stats = self.stats.borrow_mut();
            let elapsed = started.elapsed();
            if states[0].position() == self.root_position {
                stats.question += elapsed;
            } else {
                stats.candidate += elapsed;
            }
            if output.batch_forward_mode() == BatchForwardMode::Vectorized {
                stats.physical_forwards += 1;
                let longest = suffixes.iter().map(|ids| ids.len()).max().unwrap_or(0);
                let real_tokens: usize = suffixes.iter().map(|ids| ids.len()).sum();
                stats.padding_tokens += longest * suffixes.len() - real_tokens;
            } else {
                stats.physical_forwards += states.len();
            }
            Ok(output)
        }
    }

    fn ms(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1000.0
    }

    fn readout_primitive(record: &TokenRecord, k: usize) -> PrimitiveKind {
        match record.question_id.as_str() {
            "external" if k == 2 => PrimitiveKind::Noul,
            "urgency" if k >= 2 => PrimitiveKind::Score,
            _ => PrimitiveKind::Choice,
        }
    }

    fn option(args: &[String], name: &str) -> Result<String, Box<dyn Error>> {
        let index = args
            .iter()
            .position(|arg| arg == name)
            .ok_or_else(|| format!("{name} is required"))?;
        Ok(args
            .get(index + 1)
            .ok_or_else(|| format!("{name} requires a value"))?
            .clone())
    }

    fn load_tokens(path: &Path) -> Result<Vec<TokenRecord>, Box<dyn Error>> {
        let fixtures: TokenFixtures = serde_json::from_slice(&fs::read(path)?)?;
        let records = fixtures
            .records
            .into_iter()
            .filter(|record| record.fixture_case == 0)
            .collect::<Vec<_>>();
        if records.len() != 3 || records.iter().any(|r| r.root_ids != records[0].root_ids) {
            return Err("expected three same-root Phase 3B records".into());
        }
        Ok(records)
    }

    fn workload(records: &[TokenRecord], q: usize, k: usize) -> Vec<TokenRecord> {
        // The equal-position pair pools across questions. Other Q values add
        // the unequal-position record and repeat records as a load shape.
        let order = if q == 3 { [0, 1, 2] } else { [1, 2, 0] };
        (0..q)
            .map(|index| {
                let mut record = records[order[index % 3]].clone();
                record.candidate_suffix_ids = (0..k)
                    .map(|candidate| {
                        record.candidate_suffix_ids[candidate % record.candidate_suffix_ids.len()]
                            .clone()
                    })
                    .collect();
                record
            })
            .collect()
    }

    pub(super) fn run() -> Result<(), Box<dyn Error>> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let checkpoint = PathBuf::from(option(&args, "--checkpoint")?);
        let strategy = option(&args, "--strategy")?;
        let q: usize = option(&args, "--q")?.parse()?;
        let k: usize = option(&args, "--k")?.parse()?;
        let iterations: usize = option(&args, "--iterations")?.parse()?;
        let max_lanes: usize = option(&args, "--max-lanes")?.parse()?;
        if q == 0 || k == 0 || iterations == 0 || !(2..=8).contains(&max_lanes) {
            return Err("q, k, and iterations must be positive; max-lanes must be 2..=8".into());
        }
        if !["nested_sequential", "nested_batched", "pooled"].contains(&strategy.as_str()) {
            return Err("unknown strategy".into());
        }

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo = manifest
            .parent()
            .and_then(Path::parent)
            .ok_or("repo root")?;
        let records = load_tokens(
            &repo.join("research/14_phase3b_backbone_parity_results/TOKEN_FIXTURES.json"),
        )?;
        let workload = workload(&records, q, k);
        let root_ids = &workload[0].root_ids;
        let suffix_refs = workload
            .iter()
            .map(|record| {
                record
                    .candidate_suffix_ids
                    .iter()
                    .map(Vec::as_slice)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let plans = workload
            .iter()
            .zip(&suffix_refs)
            .map(|(record, suffixes)| NestedQuestion {
                question_ids: &record.question_ids,
                candidate_suffix_ids: suffixes,
            })
            .collect::<Vec<_>>();

        let runtime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let backbone = MlxQwen35Backbone::load(&checkpoint, runtime.clone(), MlxPrecision::Fp32)?;
        let bundle = ReferenceBundle::load(
            manifest.join("tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8"),
        )?;
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let timed = TimedExecutor {
                inner: &backbone,
                root_position: root_ids.len(),
                stats: RefCell::new(StageStats::default()),
            };
            let started = Instant::now();
            let features = match strategy.as_str() {
                "nested_sequential" => run_sequential_nested(&timed, root_ids, &plans)?
                    .questions()
                    .iter()
                    .map(|q| {
                        q.candidates()
                            .iter()
                            .map(|c| c.feature().to_vec())
                            .collect()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>(),
                "nested_batched" => run_batched_nested(&timed, root_ids, &plans)?
                    .questions()
                    .iter()
                    .map(|q| {
                        q.candidates()
                            .iter()
                            .map(|c| c.feature().to_vec())
                            .collect()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>(),
                _ => run_batched_nested_pooled(&timed, root_ids, &plans, max_lanes)?
                    .questions()
                    .iter()
                    .map(|q| {
                        q.candidates()
                            .iter()
                            .map(|c| c.feature().to_vec())
                            .collect()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>(),
            };
            let readout_started = Instant::now();
            let decisions = features
                .iter()
                .zip(&workload)
                .map(|(question, record)| {
                    bundle
                        .head()
                        .evaluate(readout_primitive(record, k), question)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let readout = readout_started.elapsed();
            let total = started.elapsed();
            let stats = timed.stats.into_inner();
            samples.push(serde_json::json!({
                "total_ms": ms(total),
                "prefill_ms": ms(stats.prefill),
                "question_ms": ms(stats.question),
                "candidate_ms": ms(stats.candidate),
                "readout_ms": ms(readout),
                "physical_forwards": stats.physical_forwards,
                "padding_tokens": stats.padding_tokens,
                "peak_process_bytes": peak_resident_bytes()?,
                "peak_mlx_active_bytes": runtime.peak_memory_bytes()?,
                "selected_indices": decisions.iter().map(|d| d.selected_candidate_index()).collect::<Vec<_>>(),
                "probabilities": decisions.iter().map(|d| d.full_probabilities()).collect::<Vec<_>>(),
            }));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "profile": PROFILE_ID,
                "revision": BACKBONE_REVISION,
                "arithmetic": backbone.arithmetic_id(),
                "strategy": strategy,
                "q": q,
                "k": k,
                "max_lanes": max_lanes,
            "source_question_ids": workload.iter().map(|r| &r.question_id).collect::<Vec<_>>(),
            "readout_primitives": workload.iter().map(|r| format!("{:?}", readout_primitive(r, k))).collect::<Vec<_>>(),
                "suffix_lengths": workload.iter().map(|r| r.candidate_suffix_ids.iter().map(Vec::len).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "samples": samples,
            }))?
        );
        Ok(())
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    supported::run()
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("qwen35_candidate_pool_bench requires --features mlx on macOS arm64");
}
