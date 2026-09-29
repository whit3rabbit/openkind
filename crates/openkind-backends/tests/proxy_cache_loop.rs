//! End-to-end lifecycle test for the proxy-cache task engine:
//! bootstrap collection → first fit → shadow judgement → promotion →
//! confident local serving with OOD rejection → teacher-change fallback.
//!
//! The test is fully offline: a hash embedder replaces the model-backed
//! encoder, and the synthetic teacher labels two well-separated text
//! clusters, so no model assets are downloaded (workspace invariant #6).

use openkind_backends::proxy_cache::encoder::HashEmbedder;
use openkind_backends::proxy_cache::engine::{TaskEngine, TeacherAnswer, TickOutcome};
use openkind_backends::proxy_cache::task::{
    Channel, RoutingReason, TaskConfig, TaskSpec, TeacherChangePolicy,
};
use openkind_backends::proxy_cache::text::state_text;
use openkind_backends::proxy_cache::ProxyCacheError;
use openkind_backends::proxy_cache::TextEmbedder;
use openkind_core::State;
use serde_json::json;

fn two_class_spec() -> TaskSpec {
    TaskSpec {
        instructions: json!("Classify the document topic."),
        criteria: [
            ("alpha".to_string(), Some("alpha documents".to_string())),
            ("beta".to_string(), Some("beta documents".to_string())),
        ]
        .into_iter()
        .collect(),
    }
}

fn small_config() -> TaskConfig {
    TaskConfig {
        min_train_samples: 60,
        min_samples_per_class: 10,
        min_calib_samples: 30,
        min_new_samples: 500,
        shadow_min_samples: 30,
        audit_rate: 0.05,
        audit_rate_shadow: 0.10,
        calib_fraction: 0.3,
        ood_k: 5,
        student_epochs: 400,
        student_patience: 3,
        eval_every: 20,
        teacher_change: TeacherChangePolicy::Fallback,
        teacher_change_confirm: 5,
        ..Default::default()
    }
}

/// Deterministic cluster texts: shared prefix tokens per cluster so the
/// hash embedder separates them cleanly.
fn cluster_text(cluster: usize, index: usize) -> String {
    match cluster {
        0 => format!("alpha report number {index} discusses alpha topics and alpha findings"),
        _ => format!("beta memo number {index} covers beta matters and beta outcomes"),
    }
}

fn teacher_for(cluster: usize) -> TeacherAnswer {
    let (label, probability) = match cluster {
        0 => ("alpha", 0.96),
        _ => ("beta", 0.94),
    };
    TeacherAnswer {
        label: label.to_string(),
        probabilities: match cluster {
            0 => vec![probability, 1.0 - probability],
            _ => vec![1.0 - probability, probability],
        },
        confidence: probability,
        model: "jev-1.0.0".to_string(),
        latency_ms: 1.0,
    }
}

/// Feed one sample through the engine the way the proxy does.
fn feed(
    engine: &mut TaskEngine,
    embedder: &HashEmbedder,
    text: &str,
    cluster: usize,
) -> Result<Vec<(String, serde_json::Value)>, ProxyCacheError> {
    let embedding = embedder
        .encode(&[text.to_string()])?
        .into_iter()
        .next()
        .expect("one embedding");
    engine.begin_request();
    let decision = engine.route(&embedding);
    if decision.channel == Channel::Bootstrap {
        assert!(
            decision.local.is_none(),
            "bootstrap items must never be served locally"
        );
    }
    let state = State::Text(text.to_string());
    let state_text_value = state_text(&state);
    engine.observe_teacher(
        &decision,
        &embedding,
        &state_text_value,
        "text",
        &embedder.id(),
        &teacher_for(cluster),
    )
}

fn request_training_quietly(engine: &mut TaskEngine) -> bool {
    match engine.tick() {
        Ok(outcome) => outcome.train_requested,
        Err(error) => panic!("tick failed: {error}"),
    }
}

#[test]
fn full_loop_bootstrap_train_promote_serve_locally() {
    let dir = tempfile::tempdir().unwrap();
    let embedder = HashEmbedder::new(64, 0, true).unwrap();
    let mut engine = TaskEngine::create(
        dir.path(),
        "testkey0000000000000".to_string(),
        "default".to_string(),
        "jev-latest".to_string(),
        two_class_spec(),
        small_config(),
        // budget = 1 - 0.9 = 0.1 so zero-disagreement shadow rows can pass
        // quickly at 95% confidence.
        0.9,
        None,
    )
    .unwrap();

    // Bootstrap: alternate clusters until the first fit is ready. 160
    // samples keep ~48 calibration rows, enough for the fixed-sequence
    // Clopper-Pearson scan to certify a 10% budget with zero disagreements
    // (needs ~29 answered rows after the 15% fit headroom).
    for index in 0..160 {
        let cluster = index % 2;
        feed(
            &mut engine,
            &embedder,
            &cluster_text(cluster, index),
            cluster,
        )
        .unwrap();
    }
    assert!(!engine.has_production());
    assert!(
        request_training_quietly(&mut engine),
        "readiness should request the first fit after 160 samples"
    );

    // Fit (the manager runs this off the engine lock in production).
    let input = engine.prepare_fit().unwrap().expect("fit input ready");
    let output = engine.run_fit(&input).expect("fit succeeds");
    assert!(output.usable, "clean clusters must fit the 10% budget");
    let events = engine.apply_fit(output).unwrap();
    assert!(
        events.iter().any(|(kind, _)| kind == "shadow"),
        "first candidate enters shadow: {events:?}"
    );
    assert!(!engine.has_production());

    // Keep feeding: shadow rows accumulate (bootstrap channel continues
    // because production is still empty).
    let mut promoted = false;
    for index in 120..260 {
        let cluster = index % 2;
        feed(
            &mut engine,
            &embedder,
            &cluster_text(cluster, index),
            cluster,
        )
        .unwrap();
        let outcome = engine.tick().unwrap();
        if outcome.events.iter().any(|(kind, _)| kind == "promoted") {
            promoted = true;
            break;
        }
    }
    assert!(
        promoted,
        "candidate should promote within 140 further samples"
    );
    assert!(engine.has_production());

    // Confident in-distribution request is served locally.
    let status = engine.status();
    assert_eq!(
        status.production_version.as_deref(),
        Some("student-v1"),
        "first promoted version"
    );
    let confident_text = cluster_text(0, 999);
    let embedding = embedder
        .encode(std::slice::from_ref(&confident_text))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    engine.begin_request();
    let decision = engine.route(&embedding);
    assert_eq!(decision.reason, RoutingReason::Confident);
    let local = decision.local.as_ref().expect("served locally");
    assert_eq!(local.label, "alpha");
    // Reported confidence uses Jev's peakedness definition.
    let reported = peakedness_floor(&local.probabilities);
    assert!(reported > 0.9, "peaked cluster reports high confidence");

    // Out-of-distribution text is forwarded (no local answer).
    let novel = embedder
        .encode(&["完全不同的主题 zzz qqq unheard vocabulary".to_string()])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    engine.begin_request();
    let decision = engine.route(&novel);
    assert!(
        decision.local.is_none(),
        "OOD text must not be answered locally"
    );
    assert!(
        matches!(
            decision.reason,
            RoutingReason::Ood | RoutingReason::LowConfidence
        ),
        "novel routing reason: {:?}",
        decision.reason
    );

    // Persistence: reload from disk and confirm the production student is
    // restored with its policy.
    let reloaded = TaskEngine::load(
        dir.path(),
        "testkey0000000000000".to_string(),
        small_config(),
        0.9,
        None,
    )
    .unwrap();
    assert!(reloaded.has_production(), "production survives restart");
    let reloaded_status = reloaded.status();
    assert_eq!(
        reloaded_status.production_version.as_deref(),
        Some("student-v1")
    );
}

fn peakedness_floor(probabilities: &[f64]) -> f64 {
    openkind_backends::proxy_cache::task::peakedness(probabilities)
}

#[test]
fn teacher_change_falls_back_and_new_lineage_blocks_old_shadow() {
    let dir = tempfile::tempdir().unwrap();
    let embedder = HashEmbedder::new(64, 0, true).unwrap();
    let mut config = small_config();
    config.teacher_change_confirm = 3;
    let mut engine = TaskEngine::create(
        dir.path(),
        "lineagekey0000000000".to_string(),
        "default".to_string(),
        "jev-latest".to_string(),
        two_class_spec(),
        config,
        0.9,
        None,
    )
    .unwrap();

    // Bootstrap and promote a production student of the first lineage.
    for index in 0..160 {
        let cluster = index % 2;
        feed(
            &mut engine,
            &embedder,
            &cluster_text(cluster, index),
            cluster,
        )
        .unwrap();
    }
    assert!(request_training_quietly(&mut engine));
    let input = engine.prepare_fit().unwrap().expect("fit ready");
    let output = engine.run_fit(&input).unwrap();
    engine.apply_fit(output).unwrap();
    let mut promoted = false;
    for index in 160..300 {
        let cluster = index % 2;
        feed(
            &mut engine,
            &embedder,
            &cluster_text(cluster, index),
            cluster,
        )
        .unwrap();
        if engine
            .tick()
            .unwrap()
            .events
            .iter()
            .any(|(kind, _)| kind == "promoted")
        {
            promoted = true;
            break;
        }
    }
    assert!(
        promoted,
        "first lineage should promote before the teacher moves"
    );
    assert!(engine.has_production());

    // The teacher moves to a new resolved model; after 3 consecutive answers
    // the lineage switches and the fallback policy forces teacher-only
    // routing (the old lineage's shadow, if any, can never promote).
    for index in 400..412 {
        let cluster = index % 2;
        let embedding = embedder
            .encode(&[cluster_text(cluster, index)])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        engine.begin_request();
        // Pre-switch answers may be Confident or Audit; the switch lands
        // mid-loop (confirm=3), after which routing is Fallback. The final
        // assertions below pin the post-switch behavior.
        let _decision_for_record = engine.route(&embedding);
        let decision = _decision_for_record.clone();
        let mut answer = teacher_for(cluster);
        answer.model = "jev-2.0.0".to_string();
        let state = State::Text(cluster_text(cluster, index));
        engine
            .observe_teacher(
                &decision,
                &embedding,
                &state_text(&state),
                "text",
                &embedder.id(),
                &answer,
            )
            .unwrap();
    }
    engine.tick().unwrap();
    let status = engine.status();
    assert!(
        status.forced_fallback,
        "teacher change under the fallback policy forces teacher-only routing"
    );
    // Routing now reports fallback and never answers locally.
    let embedding = embedder
        .encode(&[cluster_text(0, 500)])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    engine.begin_request();
    let decision = engine.route(&embedding);
    assert_eq!(decision.reason, RoutingReason::Fallback);
    assert!(decision.local.is_none());
}

#[test]
fn co_deferred_rows_are_teacher_labelled_and_never_calib() {
    let dir = tempfile::tempdir().unwrap();
    let embedder = HashEmbedder::new(64, 0, true).unwrap();
    let mut engine = TaskEngine::create(
        dir.path(),
        "codeferkey0000000000".to_string(),
        "default".to_string(),
        "jev-latest".to_string(),
        two_class_spec(),
        small_config(),
        0.9,
        None,
    )
    .unwrap();

    let text = cluster_text(1, 7);
    let embedding = embedder
        .encode(std::slice::from_ref(&text))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let decision = openkind_backends::proxy_cache::engine::RouteDecision {
        reason: RoutingReason::Bootstrap,
        channel: Channel::CoDeferred,
        local: None,
        student_version: None,
    };
    // begin_request draws one split for the whole request; feed several
    // requests and confirm co-deferred rows always land in train.
    for _ in 0..30 {
        engine.begin_request();
        engine
            .observe_teacher(
                &decision,
                &embedding,
                &text,
                "text",
                &embedder.id(),
                &teacher_for(1),
            )
            .unwrap();
    }
    let (train, calib) = engine.labelled_rows_for_test(100, 100).unwrap();
    // The engine wrote every co-deferred row into the train split.
    assert!(!train.is_empty() || !calib.is_empty());
    assert!(
        calib.is_empty(),
        "co-deferred rows must never enter the calibration split"
    );
    assert_eq!(train.len(), 30);
    assert!(train.iter().all(|row| row.teacher_label.is_some()));
    let _ = TickOutcome::default();
}
