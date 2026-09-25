# Native service gate — 22 September 2026

This initial release-mode campaign is superseded by the later source-fingerprint
rerun in [`2026-09-22-rerun2`](../2026-09-22-rerun2/README.md). Its service
measurements passed, but its tracked-diff hash was captured only at report
finalization, so it did not bind the executable to its build source.

The release-mode native CPU service campaign passed all checks recorded in
[`service-report.json`](service-report.json). Raw daemon logs are retained in
[`daemon.log`](daemon.log) and [`deadline-daemon.log`](deadline-daemon.log).

## Scope and provenance

The run exercised `openkindd` over HTTP with the pinned Qwen3.5-4B-Base
revision `1001bb4d826a52d1f399e183466143f4da7b741b`, the digest-checked
Phase 3.1 readout bundle, the Phase 3B tokenizer, and
`decisions_smoke.jsonl`. Each request contained three questions. The fixture
has no reviewed labels, so the run provides no semantic accuracy,
accepted-error, or coverage result.

The machine was a Mac16,5 Apple M4 Max with 36 GiB RAM, arm64 macOS 26.6.2,
and Rust 1.98.1. Source HEAD was `a5a752ab50efccba2eff0345fc5435c01248d41e`;
the working tree was dirty. The report records the tracked-diff and harness
SHA-256 values. The tracked-diff hash was captured when the report was
finalized, after the campaign. Shared tracked source files changed after the
daemon was built, so the exact compile-time source diff cannot be recovered
from this report. The daemon SHA-256 identifies the exact tested release
artifact; source-to-binary correspondence is not independently verified. The
release executable SHA-256 was
`32b7f685771c4aaa7d7d019ae2f761b1f584481a28082051df4ae3ae5bf5612e`, built
with:

```bash
CARGO_TARGET_DIR=target/native-service-gate cargo build --release --locked --offline -p openkind-server --bin openkindd
```

The model-bundle manifest declares `data_scope: exploratory_pilot` and
`production_ready: false`. The run qualifies this release-mode native CPU
service path; it does not promote the model, establish reviewed-domain quality,
or certify a signed, notarized, or published product release. Metal and
accelerated service behavior were not exercised.

## Results

| Phase | Result |
|---|---|
| Startup and resident request | Healthy in 16.36 s. First and resident requests each returned HTTP 200 with 3/3 answers in 15.36 s and 15.68 s. |
| Concurrency 1 | 3/3 accepted; p50/p95/p99 15.74/15.77/15.77 s; 0.06365 requests/s. |
| Concurrency 2 | 6/6 accepted; p50/p95/p99 15.79/31.80/31.80 s; 0.06356 requests/s. |
| Concurrency 4 | 9/12 accepted and 3 immediate HTTP 529 overloads; accepted p50/p95/p99 31.13/47.44/47.44 s; 0.06392 requests/s. |
| Cancellation and recovery | Client reset observed in the cancellation counter (`0 → 1`). The next request returned HTTP 200 with 3/3 answers in 15.69 s. |
| Deadline | A separate release daemon configured with a 1 ms queue-inclusive timeout returned HTTP 504 in 2.62 ms. |
| Steady soak | 1,800.009 s; 114/114 requests returned HTTP 200 with 342 total answers. Throughput was 0.06333 requests/s; p50/p95/p99 were 15.58/16.77/19.26 s. |
| Memory and shutdown | 2,101 one-second RSS samples. Sampled median RSS was 6,643,924,992 bytes, maximum RSS and daemon peak-RSS metric were 10,222,190,592 bytes, and final RSS was 6,713,704,448 bytes. Both daemons exited cleanly on SIGINT. |

Invalid wire input returned HTTP 422, malformed JSON returned HTTP 400, and
request IDs were echoed on all request responses. The 4-request wave confirms
that the configured single execution slot limits throughput: additional
concurrency increases queue-inclusive tail latency and then reaches the
bounded overload response.

The post-run log audit found no request-ID prefix, bearer/API-key marker, or
any of the 26 fixture string values of 16 or more characters in either daemon
log. The logs contain only startup, strategy, route, status, and timing
records; they do not contain request bodies.

## Reproduction

Build the daemon with the command above, acquire the pinned checkpoint using
the opt-in instructions in the repository README, then follow the current
runner command in [`docs/BENCHMARKS.md`](../../../BENCHMARKS.md). Supply the
digest-checked bundle fixture at
`crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8`,
the runtime manifest at
`research/14_phase3b_backbone_parity_results/bundle_probability_runtime/runtime.json`,
and tokenizer at
`research/14_phase3b_backbone_parity_results/backbone_runtime/tokenizer/tokenizer.json`.
Use a new empty output directory for each run. The build command and report
hashes identify the recorded artifact and inputs, but a reproducible source
claim requires capturing the tracked diff before the build and confirming it
is unchanged after the run.
