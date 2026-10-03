//! Offline tests for the benchmark harness. All runs use the deterministic
//! mock engine: no test may download artifacts or require a local checkpoint
//! (workspace invariant 7). Native-engine timing is exercised by the
//! checkpoint-gated commands documented in `docs/BENCHMARKS.md`.

use std::fs;
use std::path::PathBuf;

use openkind_backends::qwen35::SEMANTIC_NONE_OPTION;
use serde_json::Value;
use sha2::Digest;

use crate::gen::generate_workload;
use crate::score::{run_score, EngineKind, ScoreArgs, StrategySpec};
use crate::workload::{build_request, parse_workload, state_groups};

/// The checked-in smoke fixture: 4 states × 3 primitives = 12 rows.
const SMOKE_FIXTURE: &str = include_str!("../fixtures/decisions_smoke.jsonl");

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "openkind-bench-test-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn smoke_fixture_parses_and_groups_by_state() {
    let workload = parse_workload("smoke", SMOKE_FIXTURE.as_bytes()).expect("parse fixture");
    assert_eq!(workload.rows.len(), 12);
    let groups = state_groups(&workload.rows).expect("group");
    assert_eq!(groups.len(), 4, "four distinct tickets");
    assert!(groups.iter().all(|group| group.len() == 3));
    assert!(workload.sha256.len() == 64);
}

#[test]
fn choice_rows_carry_semantic_none_on_wire() {
    let workload = parse_workload("smoke", SMOKE_FIXTURE.as_bytes()).expect("parse fixture");
    let route = workload
        .rows
        .iter()
        .find(|row| row.id == "t01.route")
        .expect("route row");
    let question = route.question().expect("question");
    let openkind_core::Question::Choice(choice) = question else {
        panic!("expected a choice question");
    };
    assert!(choice.criteria.contains_key(SEMANTIC_NONE_OPTION));
    // The explicit-none row (t02.route) keeps its own description.
    let explicit = workload
        .rows
        .iter()
        .find(|row| row.id == "t02.route")
        .expect("explicit none row");
    let openkind_core::Question::Choice(choice) = explicit.question().expect("question") else {
        panic!("expected a choice question");
    };
    assert_eq!(
        choice.criteria[SEMANTIC_NONE_OPTION].as_deref(),
        Some("No owned queue applies; close without routing")
    );
}

#[test]
fn grouped_request_shares_one_state_and_unique_ids() {
    let workload = parse_workload("smoke", SMOKE_FIXTURE.as_bytes()).expect("parse fixture");
    let groups = state_groups(&workload.rows).expect("group");
    let request = build_request("bench", &workload.rows, &groups[0]).expect("request");
    assert_eq!(request.questions.len(), 3);
    assert_eq!(request.model, "bench");
    assert!(build_request("bench", &workload.rows, &[]).is_err());
    assert!(build_request("bench", &workload.rows, &[usize::MAX]).is_err());
    assert!(build_request("bench", &workload.rows, &[0, usize::MAX]).is_err());
    assert!(build_request("bench", &workload.rows, &[0, 0]).is_err());
}

#[test]
fn explicit_none_without_a_description_gets_the_wire_fallback() {
    let raw = r#"{"id":"q","state":"evidence","primitive":"choice","options":[{"id":"a"},{"id":"__none__","description":null}]}"#;
    let workload = parse_workload("none", raw.as_bytes()).expect("parse");
    let openkind_core::Question::Choice(question) = workload.rows[0].question().expect("question")
    else {
        panic!("expected choice");
    };
    assert!(question.criteria[SEMANTIC_NONE_OPTION]
        .as_deref()
        .is_some_and(|text| !text.trim().is_empty()));
}

#[test]
fn workload_loading_rejects_invalid_wire_questions_before_scoring() {
    for row in [
        serde_json::json!({"primitive":"noul", "text":""}),
        serde_json::json!({"primitive":"noul", "text":"   "}),
        serde_json::json!({"primitive":"noul", "text":"q", "criteria":{"true":"", "false":"no"}}),
        serde_json::json!({"primitive":"choice", "text":"", "options":[{"id":"a"}]}),
        serde_json::json!({"primitive":"score", "text":"q", "levels":["low", ""]}),
    ] {
        let mut row = row.as_object().expect("row").clone();
        row.insert("id".into(), serde_json::json!("q"));
        row.insert("state".into(), serde_json::json!("evidence"));
        let raw = serde_json::to_vec(&row).expect("row JSON");
        let error = parse_workload("invalid", &raw).expect_err("reject invalid wire question");
        assert!(error.to_string().contains("invalid decision"), "{error}");
    }
}

#[test]
fn generated_workloads_are_seed_deterministic() {
    let first = temp_dir("gen-a");
    let second = temp_dir("gen-b");
    let left = generate_workload(3, 5, 42, &first.join("w.jsonl")).expect("generate");
    let right = generate_workload(3, 5, 42, &second.join("w.jsonl")).expect("generate");
    let left_bytes = fs::read(first.join("w.jsonl")).expect("read left");
    let right_bytes = fs::read(second.join("w.jsonl")).expect("read right");
    assert_eq!(
        left_bytes, right_bytes,
        "same seed must emit identical bytes"
    );
    assert_eq!(left.sha256, right.sha256);
    assert_eq!(left.rows.len(), 15);
    let third = temp_dir("gen-c");
    let other = generate_workload(3, 5, 43, &third.join("w.jsonl")).expect("generate");
    assert_ne!(left.sha256, other.sha256, "different seed must differ");
    for dir in [first, second, third] {
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn oversized_generation_fails_without_overwriting_existing_output() {
    let dir = temp_dir("gen-overflow");
    let output = dir.join("workload.jsonl");
    fs::write(&output, "existing evidence\n").expect("existing output");
    for (states, criteria) in [(usize::MAX, 2), (usize::MAX, 1), (usize::MAX / 256, 1)] {
        assert!(generate_workload(states, criteria, 42, &output).is_err());
        assert_eq!(fs::read_to_string(&output).unwrap(), "existing evidence\n");
    }
    fs::remove_dir_all(dir).expect("cleanup");
}

#[test]
fn duplicate_ids_are_rejected() {
    let raw = concat!(
        "{\"id\":\"dup\",\"state\":\"a\",\"primitive\":\"noul\",\"text\":\"q\"}\n",
        "{\"id\":\"dup\",\"state\":\"a\",\"primitive\":\"noul\",\"text\":\"q\"}\n",
    );
    assert!(parse_workload("dup", raw.as_bytes()).is_err());
}

#[test]
fn mock_score_run_end_to_end_writes_summary_and_predictions() {
    let dir = temp_dir("score");
    let input = dir.join("decisions_smoke.jsonl");
    fs::write(&input, SMOKE_FIXTURE).expect("write fixture");
    let output_dir = dir.join("out");
    let args = ScoreArgs {
        input: input.clone(),
        engine: EngineKind::Mock,
        output_dir: output_dir.clone(),
        strategies: vec![StrategySpec::ChooseStrategy],
        reps: 2,
        group: true,
        warmup: true,
        history_aba: false,
        host: Some("test-host".into()),
        commit: Some("0000000000000000000000000000000000000000".into()),
        pretty: false,
        bundle_root: None,
        checkpoint_root: None,
        tokenizer_path: None,
        model_root: None,
        adapter: None,
    };

    run_score(&args).expect("mock score run");
    let summary_path = output_dir.join("summary-mock.json");
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(&summary_path).expect("read summary"))
            .expect("summary is JSON");
    assert_eq!(summary["schema"], "openkind-bench/v1");
    assert_eq!(summary["engine"], "mock");
    assert_eq!(summary["engine_variant"], "mock");
    assert_eq!(summary["measurement_scope"], "mock_request_path");
    assert_eq!(summary["host"], "test-host");
    assert_eq!(summary["fixture"]["rows"], 12);
    assert_eq!(summary["fixture"]["groups"], 4);
    assert_eq!(summary["grouping"], "per-state");
    assert_eq!(summary["cross_strategy_answer_parity_clean"], Value::Null);
    assert_eq!(summary["warmup"], false, "mock has no warmup pass");
    assert!(summary["peak_resident_bytes"].is_null() || summary["peak_resident_bytes"].is_u64());
    let strategies = summary["strategies"].as_array().expect("strategies");
    assert_eq!(strategies.len(), 1);
    assert_eq!(strategies[0]["strategy"], "mock");
    assert_eq!(strategies[0]["backend_id"], "mock");
    assert_eq!(strategies[0]["rows"], 12);
    assert_eq!(strategies[0]["reps"], 2);
    assert!(strategies[0]["p50_seconds"].as_f64().expect("p50") > 0.0);
    assert!(strategies[0]["decisions_per_second"].as_f64().expect("dps") > 0.0);
    let tokens = strategies[0]["input_tokens_total"]
        .as_u64()
        .expect("tokens");
    assert!(tokens > 0, "dispatch estimates mock input tokens");
    let timed_seconds: f64 = strategies[0]["samples_seconds"]
        .as_array()
        .expect("samples")
        .iter()
        .map(|value| value.as_f64().expect("seconds"))
        .sum();
    let token_rate = strategies[0]["input_tokens_per_second"]
        .as_f64()
        .expect("token rate");
    let expected_rate = tokens as f64 / timed_seconds;
    assert!((token_rate - expected_rate).abs() <= expected_rate * 1e-12);
    // CPU telemetry and host hardware are recorded with every summary.
    assert!(
        strategies[0]["cpu_time_seconds"].as_f64().is_some(),
        "cpu_time_seconds must be present (null when the OS query fails)"
    );
    assert!(
        strategies[0]["avg_cpu_percent"].as_f64().is_some(),
        "avg_cpu_percent must be present (null when the OS query fails)"
    );
    let hardware = &summary["host_hardware"];
    assert!(
        hardware.is_object(),
        "host_hardware block is always present"
    );
    assert!(
        hardware["logical_cores"].is_null() || hardware["logical_cores"].as_u64().unwrap_or(0) > 0
    );
    // The mock engine has no context budget contract; other engines record
    // their frozen per-sequence token limits here.
    assert!(summary["context"].is_null() || summary["context"].is_object());

    let prediction_path = output_dir.join("predictions-mock-mock.jsonl");
    let predictions = fs::read_to_string(&prediction_path).expect("predictions");
    let expected_digest = format!("{:x}", sha2::Sha256::digest(predictions.as_bytes()));
    assert_eq!(summary["prediction_sha256"]["mock"], expected_digest);
    let lines: Vec<&str> = predictions.lines().collect();
    assert_eq!(lines.len(), 12, "one prediction row per decision");
    for line in &lines {
        let row: Value = serde_json::from_str(line).expect("prediction row is JSON");
        assert!(row["request_latency_ms"].as_f64().is_some());
        assert!(row["answer"].is_object());
    }
    let route_line = lines
        .iter()
        .copied()
        .find(|line| serde_json::from_str::<Value>(line).expect("row")["id"] == "t01.route")
        .expect("route prediction");
    let choice_row: Value = serde_json::from_str(route_line).expect("route row");
    let probabilities = choice_row["answer"]["probabilities"]
        .as_object()
        .expect("probabilities object");
    let total: f64 = probabilities.values().filter_map(Value::as_f64).sum();
    assert!((total - 1.0).abs() < 1e-6, "choice distribution sums to 1");

    let mut router_args = args.clone();
    router_args.engine = EngineKind::RouterScript;
    router_args.strategies.clear();
    router_args.output_dir = dir.join("router");
    let router = run_score(&router_args).expect("offline router over mock siblings");
    assert_eq!(router.summary["engine_variant"], "router-script");
    assert_eq!(
        router.summary["measurement_scope"],
        "routing_overhead_with_mock_siblings"
    );
    assert_eq!(
        router.summary["strategies"][0]["backend_id"],
        "router-script/detector"
    );

    let mut oversized_reps = args.clone();
    oversized_reps.reps = usize::MAX;
    let error =
        run_score(&oversized_reps).expect_err("oversized repetition count must return an error");
    assert!(error.to_string().contains("reserve repetition samples"));
    let preserved: Value =
        serde_json::from_str(&fs::read_to_string(&summary_path).expect("preserved summary"))
            .expect("summary JSON");
    assert_eq!(
        preserved, summary,
        "failed run must preserve existing evidence"
    );

    let _ = fs::remove_dir_all(dir);
    // Silence unused-variable lint when fixture constant changes shape.
    let _ = &input;
}

#[test]
fn history_aba_repeats_the_identical_request() {
    let dir = temp_dir("history-aba");
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/qwen_history_smoke.jsonl");
    let output_dir = dir.join("out");
    let args = ScoreArgs {
        input,
        engine: EngineKind::Mock,
        output_dir: output_dir.clone(),
        strategies: vec![StrategySpec::ChooseStrategy],
        reps: 1,
        group: false,
        warmup: false,
        history_aba: true,
        host: Some("test-host".into()),
        commit: None,
        pretty: false,
        bundle_root: None,
        checkpoint_root: None,
        tokenizer_path: None,
        model_root: None,
        adapter: None,
    };
    run_score(&args).expect("history run");
    let summary: Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("summary-mock.json")).expect("summary"),
    )
    .expect("summary JSON");
    assert_eq!(summary["history_aba"], true);
    assert_eq!(summary["strategies"][0]["rows"], 3);
    let raw =
        fs::read_to_string(output_dir.join("predictions-mock-mock.jsonl")).expect("predictions");
    let rows: Vec<Value> = raw
        .lines()
        .map(|line| serde_json::from_str(line).expect("row"))
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["id"], rows[2]["id"]);
    assert_ne!(rows[0]["id"], rows[1]["id"]);
    assert_eq!(rows[0]["sequence_index"], 0);
    assert_eq!(rows[1]["sequence_index"], 1);
    assert_eq!(rows[2]["sequence_index"], 2);
    assert_eq!(rows[0]["answer"], rows[2]["answer"]);
    let mut mock_defaults = args.clone();
    mock_defaults.strategies = crate::score::DEFAULT_STRATEGIES.to_vec();
    mock_defaults.output_dir = dir.join("mock-defaults");
    run_score(&mock_defaults).expect("mock has one effective plan regardless of native sweep");

    let mut router_args = args.clone();
    router_args.engine = EngineKind::RouterScript;
    router_args.strategies.clear();
    router_args.output_dir = dir.join("router");
    let router = run_score(&router_args).expect("family history probe uses its single pinned plan");
    assert_eq!(router.summary["strategies"][0]["rows"], 3);
    assert_eq!(router.summary["history_aba"], true);
    let _ = fs::remove_dir_all(dir);
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn bf16_benchmarks_fail_closed_for_unqualified_nested_strategies() {
    use openkind_backends::qwen35::ExecutionStrategy;

    let allowed = [StrategySpec::Forced(ExecutionStrategy::RepeatedFull)];
    crate::score::validate_strategy_selection(EngineKind::Qwen35MlxBf16, &allowed)
        .expect("repeated_full remains qualifying");

    for rejected in [
        StrategySpec::Forced(ExecutionStrategy::NestedSequential),
        StrategySpec::Forced(ExecutionStrategy::NestedBatched),
        StrategySpec::ChooseStrategy,
    ] {
        let error =
            crate::score::validate_strategy_selection(EngineKind::Qwen35MlxBf16, &[rejected])
                .expect_err("nested BF16 strategy must fail closed");
        assert!(error.to_string().contains("repeated_full"));
    }
}

#[test]
fn strategy_tokens_parse_to_matching_specs() {
    use openkind_backends::qwen35::ExecutionStrategy;

    assert_eq!(
        StrategySpec::parse("choose_strategy").unwrap(),
        StrategySpec::ChooseStrategy
    );
    assert_eq!(
        StrategySpec::parse("auto").unwrap(),
        StrategySpec::ChooseStrategy
    );
    assert_eq!(
        StrategySpec::parse("repeated_full").unwrap(),
        StrategySpec::Forced(ExecutionStrategy::RepeatedFull)
    );
    assert_eq!(
        StrategySpec::parse("nested_sequential").unwrap(),
        StrategySpec::Forced(ExecutionStrategy::NestedSequential)
    );
    assert_eq!(
        StrategySpec::parse("nested_batched").unwrap(),
        StrategySpec::Forced(ExecutionStrategy::NestedBatched)
    );

    for unknown in ["RepeatedFull", "fast", "nested"] {
        let err = StrategySpec::parse(unknown).unwrap_err();
        assert!(
            err.to_string().contains("unknown strategy"),
            "`{unknown}`: {err}"
        );
    }
}

#[test]
fn parse_strategies_defaults_explicit_lists_and_rejects_empty() {
    use crate::args::parse_strategies;
    use crate::score::DEFAULT_STRATEGIES;
    use openkind_backends::qwen35::ExecutionStrategy;

    // No flag: the full default sweep.
    assert_eq!(parse_strategies(None).unwrap(), DEFAULT_STRATEGIES.to_vec());

    // Explicit list: trimmed tokens, `auto` alias included.
    let parsed = parse_strategies(Some("repeated_full, auto ,nested_batched")).unwrap();
    assert_eq!(
        parsed,
        vec![
            StrategySpec::Forced(ExecutionStrategy::RepeatedFull),
            StrategySpec::ChooseStrategy,
            StrategySpec::Forced(ExecutionStrategy::NestedBatched),
        ]
    );

    // Unknown tokens surface the accepted set.
    let err = parse_strategies(Some("repeated_full,turbo")).unwrap_err();
    assert!(err.to_string().contains("accepted"), "{err}");

    // Present-but-empty tokens yield no strategies, not the default sweep.
    let err = parse_strategies(Some(",,")).unwrap_err();
    assert!(err.to_string().contains("listed no strategies"), "{err}");

    for repeated in ["repeated_full,repeated_full", "auto,choose_strategy"] {
        let err = parse_strategies(Some(repeated)).expect_err("reject duplicate strategy aliases");
        assert!(err.to_string().contains("duplicate"), "{err}");
    }
}

#[test]
fn native_runs_require_at_least_one_strategy() {
    // The mock engine ignores strategies, so an empty sweep is fine there...
    crate::score::validate_strategy_selection(EngineKind::Mock, &[]).unwrap();
    // ...but a native run with no strategy cannot produce evidence.
    let err = crate::score::validate_strategy_selection(EngineKind::Qwen35, &[])
        .expect_err("native run with no strategies must fail");
    assert!(
        err.to_string().contains("at least one execution strategy"),
        "{err}"
    );
    let repeated = [StrategySpec::ChooseStrategy, StrategySpec::ChooseStrategy];
    let error = crate::score::validate_strategy_selection(EngineKind::Qwen35, &repeated)
        .expect_err("duplicate strategies would overwrite result files");
    assert!(error.to_string().contains("duplicate"), "{error}");
}

#[test]
fn workload_rows_reject_question_shape_violations() {
    let row = |question: serde_json::Value| {
        let mut object = question.as_object().expect("row").clone();
        object.insert("id".into(), serde_json::json!("q"));
        object.insert("state".into(), serde_json::json!("evidence"));
        serde_json::to_vec(&object).expect("row JSON")
    };

    for (raw, needle) in [
        // Choice with an empty option list.
        (
            row(serde_json::json!({
                "primitive": "choice", "text": "q", "options": []
            })),
            "has no options",
        ),
        // Duplicate option ids.
        (
            row(serde_json::json!({
                "primitive": "choice", "text": "q",
                "options": [{"id": "a"}, {"id": "a"}]
            })),
            "duplicate option id",
        ),
        // A `__none__` option with a blank description.
        (
            row(serde_json::json!({
                "primitive": "choice", "text": "q",
                "options": [{"id": "a"}, {"id": "__none__", "description": "  "}]
            })),
            "must have a non-empty",
        ),
        // Score rubrics need at least two levels.
        (
            row(serde_json::json!({
                "primitive": "score", "text": "q", "levels": ["only"]
            })),
            "at least two levels",
        ),
    ] {
        let error = parse_workload("invalid", &raw).expect_err("reject the bad row");
        // Every per-row bail is wrapped in the line's "invalid decision"
        // context; the specific message survives in the cause chain.
        let chain = format!("{error:?}");
        assert!(chain.contains("invalid decision"), "{chain}");
        if chain.contains(needle) {
            continue;
        }
        // The wire validator rejects some shapes before the row-level bail
        // (empty options collapse to empty criteria), which is fine — the
        // request still fails closed.
        assert!(
            needle == "has no options" || needle == "duplicate option id",
            "expected `{needle}` in the chain: {chain}"
        );
    }
}

#[test]
fn parse_workload_rejects_invalid_utf8_and_empty_inputs() {
    let error = parse_workload("binary", &[0xFF, 0xFE]).expect_err("invalid UTF-8");
    assert!(error.to_string().contains("decode"), "{error}");

    let error = parse_workload("empty", b"\n  \n").expect_err("empty workload");
    assert!(error.to_string().contains("no rows"), "{error}");

    let row = serde_json::json!({
        "id": "q", "state": "s", "primitive": "noul", "text": "q"
    });
    let error =
        parse_workload("dupes", format!("{row}\n{row}\n").as_bytes()).expect_err("duplicate ids");
    assert!(
        error.to_string().contains("duplicate decision id"),
        "{error}"
    );
}

#[test]
fn generated_workloads_reject_out_of_range_dimensions() {
    let dir = temp_dir("gen-zero");
    let error = match generate_workload(0, 5, 7, &dir.join("w.jsonl")) {
        Err(error) => error,
        Ok(_) => panic!("zero states must fail"),
    };
    assert!(
        error.to_string().contains("states must be at least 1"),
        "{error}"
    );

    for criteria in [0_usize, 22] {
        let error = match generate_workload(3, criteria, 7, &dir.join("w.jsonl")) {
            Err(error) => error,
            Ok(_) => panic!("criteria {criteria} must fail"),
        };
        assert!(
            error.to_string().contains("criteria must be between"),
            "{error}"
        );
    }
    let _ = fs::remove_dir_all(dir);
}
