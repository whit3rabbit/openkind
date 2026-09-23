//! Bounded FP32 allocator and process-footprint stress probe for Phase 3M.8.
//!
//! The repeated token ID is intentional: this probe measures retained
//! continuation state and allocator behavior, not model quality.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod implementation {

    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use openkind_backends::qwen35::mlx::model::{MlxBackboneState, MlxQwen35Backbone};
    use openkind_backends::qwen35::mlx::{
        MlxPrecision, MlxRuntime, MlxRuntimeConfig, SharedMlxRuntime,
    };
    use openkind_backends::qwen35::PROFILE_ID;
    use openkind_runtime::peak_resident_bytes;
    use serde_json::{json, Value};

    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    const TOKEN_ID: u32 = 1;
    const SMALL: RequestShape = RequestShape {
        name: "small",
        root_tokens: 32,
        questions: 1,
        question_tokens: 8,
        candidate_tokens: 8,
        candidates: 2,
    };
    const DEFAULT_LARGE: RequestShape = RequestShape {
        name: "large",
        root_tokens: 512,
        questions: 1,
        question_tokens: 32,
        candidate_tokens: 24,
        candidates: 64,
    };
    // Bound live continuation payload to a measured fraction of the named
    // 36-GiB development host. The model and forward scratch remain outside this
    // logical cap and are recorded separately through process/MLX telemetry.
    const DEFAULT_MAX_RETAINED_STATE_BYTES: usize = 8 * 1024 * 1024 * 1024;
    const MAX_CYCLES: usize = 8;
    const MAX_QUESTIONS: usize = 4;
    const MAX_CANDIDATES: usize = 255;
    const MAX_SEQUENCE_TOKENS: usize = 1024;
    const K_SWEEP: [usize; 4] = [32, 64, 128, 255];
    const Q_SWEEP: [usize; 3] = [1, 2, 4];

    #[derive(Clone, Copy)]
    struct RequestShape {
        name: &'static str,
        root_tokens: usize,
        questions: usize,
        question_tokens: usize,
        candidate_tokens: usize,
        candidates: usize,
    }

    struct Options {
        checkpoint_root: PathBuf,
        cycles: usize,
        large: RequestShape,
        max_retained_state_bytes: usize,
        run_recovery: bool,
        run_qk_sweep: bool,
        only_q: Option<usize>,
        only_k: Option<usize>,
    }

    struct HeldRequest {
        _root: MlxBackboneState,
        _questions: Vec<MlxBackboneState>,
        _candidates: Vec<Vec<MlxBackboneState>>,
        estimated_tensor_bytes: usize,
        retained_tensor_bytes: usize,
    }

    pub(super) fn run() -> Result<()> {
        let options = parse_options()?;
        let host_physical_memory_bytes = host_physical_memory_bytes()?;
        validate_shape(
            options.cycles,
            options.large,
            options.max_retained_state_bytes,
            host_physical_memory_bytes,
        )?;
        validate_selection(&options)?;
        let (subject_sha, working_tree_dirty) = checkout_identity()?;

        let runtime: SharedMlxRuntime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
        let backbone = MlxQwen35Backbone::load(
            &options.checkpoint_root,
            Arc::clone(&runtime),
            MlxPrecision::Fp32,
        )?;
        runtime.synchronize()?;

        emit(json!({
            "schema": "openkind-mlx-memory-stress/v1",
            "record": "run",
            "subject_sha": subject_sha,
            "working_tree_dirty": working_tree_dirty,
            "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "profile_id": PROFILE_ID,
            "precision": "fp32",
            "kernel": runtime.config().gated_delta_kernel.as_str(),
            "arithmetic_id": backbone.arithmetic_id(),
            "mlx_version": runtime.version(),
            "toolchain_identity": runtime.toolchain_identity(),
            "inactive_cache_limit_bytes": runtime.config().inactive_cache_limit_bytes,
            "process_id": std::process::id(),
            "os": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "host_physical_memory_bytes": host_physical_memory_bytes,
            "retained_state_cap_fraction_of_physical_memory":
                options.max_retained_state_bytes as f64 / host_physical_memory_bytes as f64,
            "input_mode": "synthetic_repeated_token_id",
            "token_id": TOKEN_ID,
            "evidence_scope": "allocator_and_process_memory_only; not parity, throughput, or service evidence",
            "cycles": options.cycles,
            "small_request": shape_json(SMALL),
            "large_request": shape_json(options.large),
            "max_retained_state_bytes": options.max_retained_state_bytes,
            "admission_estimate": "one root state + Q question states + Q*K candidate states, using expected_tensor_storage_bytes",
            "admission_scope": "retained continuation-state payload; excludes model weights and process baseline",
            "qk_sweep": {
                "questions": Q_SWEEP,
                "candidate_counts": K_SWEEP,
                "runs_after_preflight": true,
                "only_q": options.only_q,
                "only_k": options.only_k,
            },
        }))?;

        sample(
            "baseline_after_load",
            None,
            "model_loaded",
            &runtime,
            0,
            0,
            json!({}),
        )?;

        if options.run_recovery {
            // Warm one small request so first-use graph setup does not contaminate
            // the alternating request samples. Keep the post-warmup baseline explicit.
            runtime.reset_peak_memory()?;
            drop(run_request(
                &backbone,
                SMALL,
                options.max_retained_state_bytes,
            )?);
            sample(
                "baseline_after_warmup",
                None,
                "small_warmup_released",
                &runtime,
                0,
                estimated_retained_state_bytes(&backbone, SMALL)?,
                json!({}),
            )?;

            for cycle in 0..options.cycles {
                for shape in [SMALL, options.large] {
                    run_and_sample_request(
                        &backbone,
                        &runtime,
                        shape,
                        cycle,
                        options.max_retained_state_bytes,
                    )?;
                }
            }
        }

        if options.run_qk_sweep {
            run_qk_sweep(
                &backbone,
                &runtime,
                options.large,
                options.max_retained_state_bytes,
                options.only_q,
                options.only_k,
            )?;
        }

        Ok(())
    }

    fn parse_options() -> Result<Options> {
        let mut args = std::env::args_os().skip(1);
        let checkpoint_root = args.next().map(PathBuf::from).ok_or(
        "usage: qwen35_mlx_memory_stress <checkpoint-root> [--cycles N] [--large-root-tokens N] [--large-questions N] [--large-question-tokens N] [--large-candidate-tokens N] [--large-candidates N] [--max-retained-state-bytes N] [--only-q Q] [--only-k K] [--skip-recovery] [--skip-qk-sweep]",
    )?;
        let mut options = Options {
            checkpoint_root,
            cycles: 3,
            large: DEFAULT_LARGE,
            max_retained_state_bytes: DEFAULT_MAX_RETAINED_STATE_BYTES,
            run_recovery: true,
            run_qk_sweep: true,
            only_q: None,
            only_k: None,
        };

        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| "option name must be valid UTF-8")?;
            match flag.as_str() {
                "--skip-recovery" => {
                    options.run_recovery = false;
                    continue;
                }
                "--skip-qk-sweep" => {
                    options.run_qk_sweep = false;
                    continue;
                }
                _ => {}
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?
                .into_string()
                .map_err(|_| format!("value for {flag} must be valid UTF-8"))?;
            let parsed = value
                .parse::<usize>()
                .map_err(|error| format!("invalid value for {flag}: {error}"))?;
            match flag.as_str() {
                "--cycles" => options.cycles = parsed,
                "--large-root-tokens" => options.large.root_tokens = parsed,
                "--large-questions" => options.large.questions = parsed,
                "--large-question-tokens" => options.large.question_tokens = parsed,
                "--large-candidate-tokens" => options.large.candidate_tokens = parsed,
                "--large-candidates" => options.large.candidates = parsed,
                "--max-retained-state-bytes" => options.max_retained_state_bytes = parsed,
                "--only-q" => options.only_q = Some(parsed),
                "--only-k" => options.only_k = Some(parsed),
                _ => return Err(format!("unknown option {flag}").into()),
            }
        }
        Ok(options)
    }

    fn checkout_identity() -> Result<(String, bool)> {
        let root = env!("CARGO_MANIFEST_DIR");
        let commit = Command::new("git")
            .args(["-C", root, "rev-parse", "HEAD"])
            .output()?;
        if !commit.status.success() {
            return Err(format!(
                "git rev-parse failed: {}",
                String::from_utf8_lossy(&commit.stderr).trim()
            )
            .into());
        }
        let subject_sha = String::from_utf8(commit.stdout)?.trim().to_owned();
        if subject_sha.is_empty() {
            return Err("git rev-parse returned an empty subject SHA".into());
        }

        let status = Command::new("git")
            .args([
                "-C",
                root,
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ])
            .output()?;
        if !status.status.success() {
            return Err(format!(
                "git status failed: {}",
                String::from_utf8_lossy(&status.stderr).trim()
            )
            .into());
        }
        Ok((subject_sha, !status.stdout.is_empty()))
    }

    fn host_physical_memory_bytes() -> Result<usize> {
        let output = Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()?;
        if !output.status.success() {
            return Err(format!(
                "sysctl could not read host physical memory: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )
            .into());
        }
        let bytes = String::from_utf8(output.stdout)?.trim().parse::<usize>()?;
        if bytes == 0 {
            return Err("sysctl returned zero host physical memory".into());
        }
        Ok(bytes)
    }

    fn validate_shape(
        cycles: usize,
        shape: RequestShape,
        max_retained_state_bytes: usize,
        host_physical_memory_bytes: usize,
    ) -> Result<()> {
        if !(1..=MAX_CYCLES).contains(&cycles) {
            return Err(format!("--cycles must be between 1 and {MAX_CYCLES}").into());
        }
        if shape.root_tokens == 0
            || shape.question_tokens == 0
            || shape.candidate_tokens == 0
            || !(1..=MAX_QUESTIONS).contains(&shape.questions)
            || !(1..=MAX_CANDIDATES).contains(&shape.candidates)
        {
            return Err(format!(
            "large request dimensions must be positive and candidate count must be at most {MAX_CANDIDATES}"
        )
        .into());
        }
        if shape.root_tokens > MAX_SEQUENCE_TOKENS
            || shape.question_tokens > MAX_SEQUENCE_TOKENS
            || shape.candidate_tokens > MAX_SEQUENCE_TOKENS
        {
            return Err(format!(
                "each large request segment is capped at {MAX_SEQUENCE_TOKENS} tokens"
            )
            .into());
        }
        if max_retained_state_bytes == 0
            || max_retained_state_bytes > DEFAULT_MAX_RETAINED_STATE_BYTES
            || max_retained_state_bytes > host_physical_memory_bytes / 4
        {
            return Err(format!(
            "--max-retained-state-bytes must be in 1..={} and no more than one quarter of host physical memory",
            DEFAULT_MAX_RETAINED_STATE_BYTES.min(host_physical_memory_bytes / 4)
        )
        .into());
        }
        Ok(())
    }

    fn validate_selection(options: &Options) -> Result<()> {
        if let Some(q) = options.only_q {
            if !Q_SWEEP.contains(&q) {
                return Err(format!("--only-q must be one of {Q_SWEEP:?}").into());
            }
        }
        if let Some(k) = options.only_k {
            if !K_SWEEP.contains(&k) {
                return Err(format!("--only-k must be one of {K_SWEEP:?}").into());
            }
        }
        if (options.only_q.is_some() || options.only_k.is_some()) && !options.run_qk_sweep {
            return Err("--only-q and --only-k require the Q/K sweep to be enabled".into());
        }
        Ok(())
    }

    fn run_request(
        backbone: &MlxQwen35Backbone,
        shape: RequestShape,
        max_retained_state_bytes: usize,
    ) -> Result<HeldRequest> {
        let estimated_bytes = estimated_retained_state_bytes(backbone, shape)?;
        if estimated_bytes > max_retained_state_bytes {
            return Err(format!(
            "{} request Q={} K={} estimates {estimated_bytes} retained state bytes, above --max-retained-state-bytes {max_retained_state_bytes}",
            shape.name, shape.questions, shape.candidates
        )
        .into());
        }

        let root_ids = vec![TOKEN_ID; shape.root_tokens];
        let question_ids = vec![TOKEN_ID; shape.question_tokens];
        let candidate_ids = vec![TOKEN_ID; shape.candidate_tokens];

        let (root_output, root) = backbone.prefill(&root_ids)?;
        drop(root_output);
        let mut questions = Vec::with_capacity(shape.questions);
        let mut candidates = Vec::with_capacity(shape.questions);
        for _ in 0..shape.questions {
            let (question_output, question) = backbone.continue_from(&root, &question_ids)?;
            drop(question_output);

            let mut question_candidates = Vec::with_capacity(shape.candidates);
            for _ in 0..shape.candidates {
                let (candidate_output, candidate) =
                    backbone.continue_from(&question, &candidate_ids)?;
                drop(candidate_output);
                question_candidates.push(candidate);
            }
            questions.push(question);
            candidates.push(question_candidates);
        }

        let mut retained_tensor_bytes = root.tensor_storage_bytes();
        for (question, question_candidates) in questions.iter().zip(&candidates) {
            retained_tensor_bytes = retained_tensor_bytes
                .checked_add(question.tensor_storage_bytes())
                .ok_or("retained tensor byte count overflowed")?;
            for candidate in question_candidates {
                retained_tensor_bytes = retained_tensor_bytes
                    .checked_add(candidate.tensor_storage_bytes())
                    .ok_or("retained tensor byte count overflowed")?;
            }
        }
        if retained_tensor_bytes > max_retained_state_bytes {
            return Err(format!(
            "{} request Q={} K={} retains {retained_tensor_bytes} tensor bytes, above --max-retained-state-bytes {max_retained_state_bytes}",
            shape.name, shape.questions, shape.candidates
        )
        .into());
        }

        Ok(HeldRequest {
            _root: root,
            _questions: questions,
            _candidates: candidates,
            estimated_tensor_bytes: estimated_bytes,
            retained_tensor_bytes,
        })
    }

    fn estimated_retained_state_bytes(
        backbone: &MlxQwen35Backbone,
        shape: RequestShape,
    ) -> Result<usize> {
        let question_position = shape
            .root_tokens
            .checked_add(shape.question_tokens)
            .ok_or("question position overflowed")?;
        let candidate_position = question_position
            .checked_add(shape.candidate_tokens)
            .ok_or("candidate position overflowed")?;
        let root_bytes = backbone.expected_tensor_storage_bytes(shape.root_tokens);
        let question_bytes = backbone
            .expected_tensor_storage_bytes(question_position)
            .checked_mul(shape.questions)
            .ok_or("question state byte estimate overflowed")?;
        let candidate_bytes = backbone
            .expected_tensor_storage_bytes(candidate_position)
            .checked_mul(shape.questions)
            .and_then(|bytes| bytes.checked_mul(shape.candidates))
            .ok_or("candidate state byte estimate overflowed")?;
        root_bytes
            .checked_add(question_bytes)
            .and_then(|bytes| bytes.checked_add(candidate_bytes))
            .ok_or_else(|| "retained-state byte estimate overflowed".into())
    }

    fn max_admitted_candidates(
        backbone: &MlxQwen35Backbone,
        template: RequestShape,
        questions: usize,
        cap_bytes: usize,
    ) -> Result<usize> {
        let mut low = 0;
        let mut high = MAX_CANDIDATES;
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            let shape = RequestShape {
                name: template.name,
                questions,
                candidates: middle,
                ..template
            };
            if estimated_retained_state_bytes(backbone, shape)? <= cap_bytes {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(low)
    }

    fn run_and_sample_request(
        backbone: &MlxQwen35Backbone,
        runtime: &MlxRuntime,
        requested_shape: RequestShape,
        cycle: usize,
        max_retained_state_bytes: usize,
    ) -> Result<()> {
        let requested_estimate = estimated_retained_state_bytes(backbone, requested_shape)?;
        let admitted = requested_estimate <= max_retained_state_bytes;
        let fallback_candidates = if admitted {
            requested_shape.candidates
        } else {
            max_admitted_candidates(
                backbone,
                requested_shape,
                requested_shape.questions,
                max_retained_state_bytes,
            )?
        };
        let actual_shape = RequestShape {
            candidates: if admitted {
                requested_shape.candidates
            } else {
                fallback_candidates
            },
            ..requested_shape
        };
        emit_admission(
            runtime,
            "recovery_admission",
            Some((
                Some(cycle),
                requested_shape.name,
                requested_shape.questions,
                requested_shape.candidates,
            )),
            requested_estimate,
            max_retained_state_bytes,
            admitted,
            fallback_candidates,
            "alternating_recovery",
        )?;
        if fallback_candidates == 0 {
            return Ok(());
        }

        let actual_estimate = estimated_retained_state_bytes(backbone, actual_shape)?;
        runtime.reset_peak_memory()?;
        let held = run_request(backbone, actual_shape, max_retained_state_bytes)?;
        sample(
            "request_memory",
            Some((
                Some(cycle),
                actual_shape.name,
                actual_shape.questions,
                actual_shape.candidates,
            )),
            "live",
            runtime,
            held.retained_tensor_bytes,
            held.estimated_tensor_bytes,
            json!({
                "requested_candidate_count": requested_shape.candidates,
                "admitted": admitted,
                "fallback_candidate_count": fallback_candidates,
                "estimated_retained_state_bytes_for_requested_shape": requested_estimate,
                "estimated_retained_state_bytes_for_actual_shape": actual_estimate,
            }),
        )?;
        drop(held);
        sample(
            "request_memory",
            Some((
                Some(cycle),
                actual_shape.name,
                actual_shape.questions,
                actual_shape.candidates,
            )),
            "released",
            runtime,
            0,
            actual_estimate,
            json!({
                "requested_candidate_count": requested_shape.candidates,
                "admitted": admitted,
                "fallback_candidate_count": fallback_candidates,
                "estimated_retained_state_bytes_for_requested_shape": requested_estimate,
                "estimated_retained_state_bytes_for_actual_shape": actual_estimate,
            }),
        )
    }

    fn run_qk_sweep(
        backbone: &MlxQwen35Backbone,
        runtime: &MlxRuntime,
        template: RequestShape,
        max_retained_state_bytes: usize,
        only_q: Option<usize>,
        only_k: Option<usize>,
    ) -> Result<()> {
        let questions_to_run: Vec<usize> = only_q.map_or_else(|| Q_SWEEP.to_vec(), |q| vec![q]);
        let candidates_to_run: Vec<usize> = only_k.map_or_else(|| K_SWEEP.to_vec(), |k| vec![k]);

        for questions in questions_to_run {
            for &candidates in &candidates_to_run {
                let shape = RequestShape {
                    name: "qk_sweep",
                    questions,
                    candidates,
                    ..template
                };
                let estimate = estimated_retained_state_bytes(backbone, shape)?;
                let admitted = estimate <= max_retained_state_bytes;
                let max_admitted =
                    max_admitted_candidates(backbone, shape, questions, max_retained_state_bytes)?;
                emit_admission(
                    runtime,
                    "admission_case",
                    Some((None, "qk_sweep", questions, candidates)),
                    estimate,
                    max_retained_state_bytes,
                    admitted,
                    max_admitted,
                    "qk_sweep",
                )?;
                if !admitted {
                    continue;
                }

                runtime.reset_peak_memory()?;
                let held = run_request(backbone, shape, max_retained_state_bytes)?;
                sample(
                    "qk_sweep_memory",
                    Some((None, "qk_sweep", questions, candidates)),
                    "live",
                    runtime,
                    held.retained_tensor_bytes,
                    held.estimated_tensor_bytes,
                    json!({
                        "admitted": true,
                        "max_admitted_candidate_count_for_q": max_admitted,
                    }),
                )?;
                drop(held);
                sample(
                    "qk_sweep_memory",
                    Some((None, "qk_sweep", questions, candidates)),
                    "released",
                    runtime,
                    0,
                    estimate,
                    json!({
                        "admitted": true,
                        "max_admitted_candidate_count_for_q": max_admitted,
                    }),
                )?;
            }

            let fallback_request = *candidates_to_run.last().ok_or("empty K sweep")?;
            let fallback_shape = RequestShape {
                name: "qk_sweep_fallback",
                questions,
                candidates: fallback_request,
                ..template
            };
            let fallback_estimate = estimated_retained_state_bytes(backbone, fallback_shape)?;
            let fallback_k = max_admitted_candidates(
                backbone,
                fallback_shape,
                questions,
                max_retained_state_bytes,
            )?;
            if fallback_estimate > max_retained_state_bytes && fallback_k > 0 {
                let admitted_shape = RequestShape {
                    candidates: fallback_k,
                    ..fallback_shape
                };
                let admitted_estimate = estimated_retained_state_bytes(backbone, admitted_shape)?;
                emit_admission(
                    runtime,
                    "admission_fallback",
                    Some((None, "qk_sweep_fallback", questions, fallback_request)),
                    admitted_estimate,
                    max_retained_state_bytes,
                    true,
                    fallback_k,
                    "qk_sweep_fallback",
                )?;
                runtime.reset_peak_memory()?;
                let held = run_request(backbone, admitted_shape, max_retained_state_bytes)?;
                sample(
                    "qk_fallback_memory",
                    Some((None, "qk_sweep_fallback", questions, fallback_k)),
                    "live",
                    runtime,
                    held.retained_tensor_bytes,
                    held.estimated_tensor_bytes,
                    json!({
                        "requested_candidate_count": fallback_request,
                        "fallback_candidate_count": fallback_k,
                        "admitted": true,
                    }),
                )?;
                drop(held);
                sample(
                    "qk_fallback_memory",
                    Some((None, "qk_sweep_fallback", questions, fallback_k)),
                    "released",
                    runtime,
                    0,
                    admitted_estimate,
                    json!({
                        "requested_candidate_count": fallback_request,
                        "fallback_candidate_count": fallback_k,
                        "admitted": true,
                    }),
                )?;
            } else if fallback_estimate > max_retained_state_bytes {
                emit_admission(
                    runtime,
                    "admission_fallback",
                    Some((None, "qk_sweep_fallback", questions, fallback_request)),
                    fallback_estimate,
                    max_retained_state_bytes,
                    false,
                    0,
                    "no_candidate_count_fits_cap",
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_admission(
        runtime: &MlxRuntime,
        record: &str,
        request: Option<(Option<usize>, &str, usize, usize)>,
        estimated_bytes: usize,
        cap_bytes: usize,
        admitted: bool,
        fallback_candidates: usize,
        reason: &str,
    ) -> Result<()> {
        let requested_candidates = request.map(|(_, _, _, candidates)| candidates);
        sample(
            record,
            request,
            if admitted {
                "preflight_admitted"
            } else {
                "preflight_rejected"
            },
            runtime,
            0,
            estimated_bytes,
            json!({
                "admitted": admitted,
                "admission_reason": if admitted { "estimated_retained_state_within_cap" } else { reason },
                "estimated_retained_state_bytes": estimated_bytes,
                "max_retained_state_bytes": cap_bytes,
                "requested_candidate_count": requested_candidates,
                "fallback_candidate_count": fallback_candidates,
            }),
        )
    }

    fn sample(
        record: &str,
        request: Option<(Option<usize>, &str, usize, usize)>,
        phase: &str,
        runtime: &MlxRuntime,
        retained_tensor_bytes: usize,
        estimated_tensor_bytes: usize,
        extra: Value,
    ) -> Result<()> {
        runtime.synchronize()?;
        let mlx = runtime.memory_snapshot();
        let active_bytes = mlx
            .active_bytes
            .ok_or("MLX active-memory query failed during sampling")?;
        let cache_bytes = mlx
            .cache_bytes
            .ok_or("MLX cache-memory query failed during sampling")?;
        let peak_active_bytes = mlx
            .peak_bytes
            .ok_or("MLX peak-memory query failed during sampling")?;
        let process = process_memory_snapshot()?;
        let rss_peak_bytes = peak_resident_bytes().ok();
        let (cycle, request_name, questions, candidates) = request.map_or(
            (Value::Null, Value::Null, Value::Null, Value::Null),
            |(cycle, name, q, k)| {
                (
                    cycle.map_or(Value::Null, |cycle| json!(cycle + 1)),
                    json!(name),
                    json!(q),
                    json!(k),
                )
            },
        );

        let mut sample = json!({
            "schema": "openkind-mlx-memory-stress/v1",
            "record": record,
            "timestamp_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            "cycle": cycle,
            "request": request_name,
            "question_count": questions,
            "candidate_count": candidates,
            "phase": phase,
            "retained_tensor_state_bytes": retained_tensor_bytes,
            "estimated_retained_state_bytes": estimated_tensor_bytes,
            "mlx_active_bytes": active_bytes,
            "mlx_cache_bytes": cache_bytes,
            "mlx_peak_active_bytes_since_request_reset": peak_active_bytes,
            "process_rss_bytes": process.rss_bytes,
            "process_rss_high_water_bytes": rss_peak_bytes,
            "process_footprint_bytes": process.footprint_bytes,
            "process_phys_footprint_bytes": process.phys_footprint_bytes,
            "process_phys_footprint_peak_bytes": process.phys_footprint_peak_bytes,
        });
        if let (Some(sample), Some(extra)) = (sample.as_object_mut(), extra.as_object()) {
            for (key, value) in extra {
                sample.insert(key.clone(), value.clone());
            }
        }
        emit(sample)
    }

    struct ProcessMemorySnapshot {
        rss_bytes: usize,
        footprint_bytes: usize,
        phys_footprint_bytes: usize,
        phys_footprint_peak_bytes: usize,
    }

    fn process_memory_snapshot() -> Result<ProcessMemorySnapshot> {
        let pid = std::process::id();
        let pid_arg = pid.to_string();
        let rss = Command::new("/bin/ps")
            .args(["-o", "rss=", "-p", pid_arg.as_str()])
            .output()?;
        if !rss.status.success() {
            return Err(format!(
                "ps failed to read RSS for pid {pid}: {}",
                String::from_utf8_lossy(&rss.stderr).trim()
            )
            .into());
        }
        let rss_kib = String::from_utf8(rss.stdout)?.trim().parse::<usize>()?;
        let rss_bytes = rss_kib
            .checked_mul(1024)
            .ok_or("RSS conversion overflowed")?;

        // `footprint` reports the current process's physical footprint in bytes.
        // Its JSON also includes a process high-water mark for the same measure.
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!("openkind-footprint-{pid}-{unique}.json"));
        let footprint = Command::new("/usr/bin/footprint")
            .args([
                "--pid",
                pid_arg.as_str(),
                "--format",
                "bytes",
                "--noCategories",
                "--json",
            ])
            .arg(&path)
            .output()?;
        if !footprint.status.success() {
            let _ = fs::remove_file(&path);
            return Err(format!(
                "footprint failed for pid {pid}: {}",
                String::from_utf8_lossy(&footprint.stderr).trim()
            )
            .into());
        }
        let raw = fs::read(&path);
        let _ = fs::remove_file(&path);
        let value: Value = serde_json::from_slice(&raw?)?;
        let process = value["processes"]
            .as_array()
            .and_then(|processes| {
                processes
                    .iter()
                    .find(|process| process["pid"].as_u64() == Some(u64::from(pid)))
            })
            .ok_or("footprint JSON did not include the probe process")?;

        Ok(ProcessMemorySnapshot {
            rss_bytes,
            footprint_bytes: required_bytes(process, "footprint")?,
            phys_footprint_bytes: required_bytes(&process["auxiliary"], "phys_footprint")?,
            phys_footprint_peak_bytes: required_bytes(
                &process["auxiliary"],
                "phys_footprint_peak",
            )?,
        })
    }

    fn required_bytes(value: &Value, field: &str) -> Result<usize> {
        let bytes = value[field]
            .as_u64()
            .ok_or_else(|| format!("footprint JSON is missing {field}"))?;
        usize::try_from(bytes).map_err(|_| format!("{field} does not fit usize").into())
    }

    fn shape_json(shape: RequestShape) -> Value {
        json!({
            "root_tokens": shape.root_tokens,
            "question_count": shape.questions,
            "question_tokens": shape.question_tokens,
            "candidate_tokens": shape.candidate_tokens,
            "candidate_count": shape.candidates,
        })
    }

    fn emit(value: Value) -> Result<()> {
        use std::io::Write;

        let stdout = std::io::stdout();
        let mut output = stdout.lock();
        serde_json::to_writer(&mut output, &value)?;
        writeln!(output)?;
        output.flush()?;
        Ok(())
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = implementation::run() {
        eprintln!("qwen35_mlx_memory_stress: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("qwen35_mlx_memory_stress requires --features mlx on macOS arm64");
}
