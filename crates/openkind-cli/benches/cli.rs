use std::collections::HashMap;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openkind_cli::{parse_and_validate_request, Cli};
use openkind_core::{NoulQuestion, Question, State, SystemRequest};

const QUESTION_COUNTS: [usize; 3] = [1, 8, 32];

fn request_json(question_count: usize) -> String {
    let questions = (0..question_count)
        .map(|index| {
            (
                format!("q{index}"),
                Question::Noul(NoulQuestion {
                    instructions: serde_json::json!(format!("Question {index}?")),
                    criteria: None,
                }),
            )
        })
        .collect::<HashMap<_, _, _>>();
    serde_json::to_string(&SystemRequest {
        state: State::Text("A deterministic CLI benchmark state.".into()),
        model: "mock".into(),
        questions,
    })
    .expect("serialize benchmark request")
}

fn bench_argument_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("cli_parse");
    let cases: [(&str, &[&str]); 4] = [
        ("inspect", &["openkind", "inspect", "request.json"]),
        (
            "evaluate",
            &[
                "openkind",
                "evaluate",
                "request.json",
                "--server",
                "http://127.0.0.1:8080",
                "--api-key",
                "bench-key",
                "--pretty",
            ],
        ),
        (
            "serve",
            &[
                "openkind",
                "serve",
                "--http-addr",
                "127.0.0.1:8080",
                "--grpc-addr",
                "127.0.0.1:9090",
                "--models",
                "mock",
                "--api-key",
                "bench-key",
            ],
        ),
        ("version", &["openkind", "version"]),
    ];

    for (name, args) in cases {
        group.bench_function(name, |b| {
            b.iter(|| black_box(Cli::try_parse_from(args).expect("parse benchmark command")))
        });
    }
    group.finish();
}

fn bench_inspect(c: &mut Criterion) {
    let mut group = c.benchmark_group("cli_inspect");
    for question_count in QUESTION_COUNTS {
        let input = request_json(question_count);
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("questions", question_count),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(
                        parse_and_validate_request(black_box(input.as_str()))
                            .expect("parse and validate benchmark request"),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_argument_parsing, bench_inspect);
criterion_main!(benches);
