# Jeeves native Rust MLX benchmark (2026-09-29)

Release-build request-path measurements for `jeeves:15fb3b95801f2f039636`
on Apple M4 Max, 14 logical cores, 36 GiB, macOS 26.6.2. The engine uses
packed affine4/g64 weights, FP32 arithmetic, and the exported pointer head.
It scores an empty reasoning chain and executes one fresh hybrid-backbone
pass per question.

## Results

| Workload / process | Whole workload p50 / p95 | Decisions/s | Input tok/s | Peak RSS | Peak physical footprint | Load |
|---|---:|---:|---:|---:|---:|---:|
| run-a (12 decisions) | 13.062 / 13.096 s | 0.919 | 152.2 | 9.18 GB | 5.74 GB | 9.99 s |
| run-b (12 decisions) | 13.169 / 13.214 s | 0.911 | 151.3 | 9.18 GB | 5.74 GB | 10.41 s |
| shape21-a (21 decisions) | 28.394 / 28.648 s | 0.740 | 156.2 | 9.18 GB | 5.74 GB | 9.86 s |
| shape21-b (21 decisions) | 29.490 / 29.575 s | 0.712 | 150.8 | 9.18 GB | 5.74 GB | 10.44 s |

Each p50/p95 describes the whole workload repetition. With three samples
per process, p95 is the maximum observed repetition, not a population tail
estimate. Grouped predictions share request latency. Loading, file writes,
and one full-workload warmup are excluded from timing. Loading includes
artifact size and digest verification.

Peak RSS is the process high-water mark, including loading and mapped
checkpoint pages. `/usr/bin/time -l` separately records peak physical memory
footprint. The earlier
[parity replay](../../verification/jeeves-native/2026-09-29/REPORT.md) reports
MLX active allocation, a different metric on a smaller workload.

## Provenance and repeatability

- Benchmark source: `5c9105864b74ff47ee9bc21833cf4bbf5080de7b`; its Rust
  implementation is identical to `7d91b3c865ae36bfc56e0341e1a9d69795667c30`.
  Documentation and recommendation-data edits were pending during the runs.
- Checkpoint: `cowWhySo/jeeves-mlx` at
  `6f4220a267a94c5279b3fcbc5499de734b9a55b0`, verified locally before loading.
- `mlx-rs 0.32.0`, MLX core `0.32.2`, Rust `1.98.1`, Xcode `27.0`, SDK `27.0`.
  [provenance.json](provenance.json) pins release binary hashes and toolchain.
- Mixed workload: four states × Noul/Choice/Score, 12 decisions, four requests;
  SHA-256 `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb`.
- Generated workload: one state × 21 Noul criteria, seed `291607`, one request;
  SHA-256 `a5539013310ea996c6aee1ba5f9d1f782230cd58572cb69e2aab091b62f4592e`.
- Two independent processes per workload, full-workload warmup, three timed
  repetitions each. See the raw summaries, predictions, and `run.log` in
  `run-a/`, `run-b/`, `shape21-a/`, and `shape21-b/`.
- All 12 mixed and 21 generated final answers match exactly between their
  paired fresh processes; [repeatability.json](repeatability.json) records
  this check. Raw prediction digests remain bound by each summary.
- A live desktop host, with background Finder, Spotlight, and renderer activity in
  the saved process snapshots. No concurrent model benchmark or build ran
  during timed passes. These measurements are a local baseline.

## Reproduce

Acquire the pinned package or pull the catalog profile explicitly first.
`JEEVES_ROOT` points to its execution files, or an installation's `checkpoint`
directory. Tests and builds do not download model assets.

```bash
SDKROOT="$(xcrun --show-sdk-path)" cargo build --release --locked \
  -p openkind-bench --features mlx
target/release/openkind-bench gen-workload \
  --states 1 --criteria 21 --seed 291607 \
  --output bench-output/jeeves-shape21.jsonl

# Run each command in two separate processes with distinct output directories.
target/release/openkind-bench score \
  crates/openkind-bench/fixtures/decisions_smoke.jsonl \
  --engine jeeves-mlx-affine4 --model-root "$JEEVES_ROOT" \
  --reps 3 --host "<host label>" --commit "$(git rev-parse HEAD)" \
  --output-dir bench-output/jeeves-mixed-a --pretty
target/release/openkind-bench score bench-output/jeeves-shape21.jsonl \
  --engine jeeves-mlx-affine4 --model-root "$JEEVES_ROOT" \
  --reps 3 --host "<host label>" --commit "$(git rev-parse HEAD)" \
  --output-dir bench-output/jeeves-shape21-a --pretty
```

Qwen strategy overrides do not apply. Summary context records the 4096-token
per-question limit, 64-question request limit, quantization, and arithmetic.
[recommendation-data.json](recommendation-data.json) joins these summaries to
the immutable catalog name and labels the limited no-thinking parity gate.

## Public registry and installed serving

The public registry is published at
[955cc49](https://github.com/whit3rabbit/openkind-model-registry/commit/955cc49d7a378f4999213149c50ce1062cd351bf).
`sync-model-registry.py --remote` verified 16 metadata files and 13 pinned
GitHub assets over public HTTPS. Jeeves' 11 manifest artifacts come directly
from its pinned Hugging Face revision and total 5,066,655,549 bytes.

The CLI catalog pull reused and SHA-256 verified the cached 5,038,161,163-byte
weight blob in persistent storage and downloaded the remaining 27.2 MiB.
`openkind serve --installed-models jeeves:15fb3b95801f2f039636` loaded that
installation. `/v1/models` listed its immutable name, and `/v1/systemone`
returned Noul, Choice with described `__none__`, and Score answers.
[registry.json](registry.json) records this smoke, its input, and its response.
The smoke establishes the install-and-serve path, not a full cold weight transfer.

## Verification and evidence boundary

Format, default and MLX workspace clippy, default workspace tests
(791 reported passes in the final serial run), workspace benches
(524 reported passes), release MLX build, schema generation, and
`git diff --check` pass. Schemas are unchanged. Focused MLX benchmark,
server, and model-store unit tests report 53 passes. Prediction digests,
exact paired answers, four recommendation-data registry joins, and local
Markdown links were checked.

Two earlier workspace attempts failed in existing tests: the CLI contracts
suite hit a temporary-directory `AlreadyExists` collision, and the first
serial retry hit the proxy-cache confidence assertion at
`crates/openkind-backends/tests/proxy_cache_loop.rs:211`. The proxy-cache
case passed in isolation; the subsequent full serial workspace run passed.
[verification.json](verification.json) preserves the failed attempts and
successful retries separately. No unrelated test implementation was changed.

The [native parity report](../../verification/jeeves-native/2026-09-29/REPORT.md)
records four exact-token reference rows, matching selections, and maximum
probability difference `0.004152` against the live Python MLX reference.
That is a limited reference gate, not broad task qualification.

These smoke-scale measurements are outside the shape777 comparison table.
The seeded 21-question workload matches the earlier JevK5 smoke fixture, but
checkpoint size, renderer, arithmetic, and execution path differ. No matched
cross-engine speed or model-quality claim follows. Reviewed task quality,
semantic-none calibration, larger source-aligned parity coverage, and service
soak remain separate gates.
