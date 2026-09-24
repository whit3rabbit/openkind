# Criterion component optimization, 23 September 2026

This record compares a fresh same-host snapshot with a working-tree candidate.
The requested threshold is at least `1.20x` lower time in **each** of the 15
existing Criterion cases. Eleven cases meet it. The four sequential client
cases remain below it and are left as measured. The new parallel client case
is reported separately because it was not in the starting suite.

All cases use `MockEngine`. The server target invokes Axum in memory; the
client target uses a warm localhost TCP connection. These numbers measure
component overhead, not native-model throughput or classification quality.
See [the methodology](../../BENCHMARKS.md#criterion-microbenchmarks) for the
measured regions and exclusions.

## Run identity

- Baseline source commit: `0dd269e74d1d73fd9002435561ec640ba5e798e8`, clean checkout before edits.
- Candidate: working tree based on that commit. SHA-256 of the binary Git diff
  for the 14 changed Cargo and Rust source files, excluding this report:
  `679bb3257d053973a5cdc66b6880800f4547c2dd72862c6e90b56ad7a0b1579b`.
  This identifies the measured source patch, not a committed revision.
  Reproduce the digest with
  `git diff --binary -- Cargo.toml Cargo.lock crates/ | shasum -a 256`.
- Host: Mac16,5, Apple M4 Max, 36 GiB RAM, macOS 26.6.2 (25G83), AC power.
- Toolchain: `rustc 1.98.1`, `cargo 1.98.1`, Criterion `0.8.2`.
- Cargo `bench` profile, 3-second warmup, 5-second measurement, 100 samples,
  95% confidence interval, plots disabled. No threshold was applied by the
  harness. Runs were serial, without another benchmark or profiler active.
- Values below are Criterion slope point estimates and 95% intervals, in
  microseconds. Speedup is the baseline estimate divided by the candidate
  estimate. Raw local reports are under ignored `target/criterion/`.

```bash
# Clean starting tree, before source edits
cargo bench --locked -p openkind-cli --bench cli -- --noplot --save-baseline criterion-0dd269e-20260923-baseline
cargo bench --locked -p openkind-server --bench server -- --noplot --save-baseline criterion-0dd269e-20260923-baseline
cargo bench --locked -p openkind-client --bench client -- --noplot --save-baseline criterion-0dd269e-20260923-baseline

# Working-tree candidate
cargo bench --locked -p openkind-cli --bench cli -- --noplot --save-baseline criterion-0dd269e-20260923-single-pass
cargo bench --locked -p openkind-server --bench server -- --noplot --save-baseline criterion-0dd269e-20260923-single-pass
cargo bench --locked -p openkind-client --bench client -- --noplot --save-baseline criterion-0dd269e-20260923-chunk-fix
```

## Existing cases

| Target | Case | Baseline, µs (95% interval) | Candidate, µs (95% interval) | Speedup | Meets 1.20x |
|---|---|---:|---:|---:|---|
| CLI | parse `inspect` | 0.1225 (0.1219–0.1230) | 0.0890 (0.0887–0.0894) | 1.376x | yes |
| CLI | parse `evaluate` | 0.3881 (0.3867–0.3895) | 0.2600 (0.2587–0.2614) | 1.493x | yes |
| CLI | parse `serve` | 0.4482 (0.4459–0.4512) | 0.3292 (0.3282–0.3302) | 1.361x | yes |
| CLI | parse `version` | 0.0842 (0.0837–0.0848) | 0.0487 (0.0485–0.0489) | 1.730x | yes |
| CLI | inspect, 1 question | 0.2422 (0.2410–0.2434) | 0.1977 (0.1971–0.1982) | 1.225x | yes |
| CLI | inspect, 8 questions | 1.1091 (1.1061–1.1128) | 0.7632 (0.7615–0.7649) | 1.453x | yes |
| CLI | inspect, 32 questions | 4.3177 (4.3086–4.3274) | 2.8786 (2.8724–2.8851) | 1.500x | yes |
| Server | `/health` | 2.0479 (2.0423–2.0537) | 1.5899 (1.5872–1.5929) | 1.288x | yes |
| Server | `/v1/systemone`, 1 question | 3.4989 (3.4882–3.5099) | 2.8829 (2.8756–2.8901) | 1.214x | yes |
| Server | `/v1/systemone`, 8 questions | 7.4362 (7.4173–7.4554) | 5.7745 (5.7578–5.7920) | 1.288x | yes |
| Server | `/v1/systemone`, 32 questions | 21.2002 (21.1391–21.2639) | 15.6553 (15.6170–15.6979) | 1.354x | yes |
| Client | `health` | 60.6095 (60.2204–60.9270) | 59.0158 (58.7921–59.2537) | 1.027x | no |
| Client | `evaluate`, 1 question | 66.0121 (65.7931–66.2420) | 63.5657 (63.4769–63.6619) | 1.038x | no |
| Client | `evaluate`, 8 questions | 73.2169 (72.9615–73.5331) | 70.9850 (70.4449–71.7336) | 1.031x | no |
| Client | `evaluate`, 32 questions | 96.2481 (95.8693–96.6394) | 88.2983 (88.0751–88.5483) | 1.090x | no |

The optimization review concludes at 11 of 15 cases. The four sequential
client results remain below `1.20x`; this record does not claim that the
original per-case target was met.

The CLI parser now keeps common argv and flag collections inline. The core
question deserializer uses fixed field and tag enums. Response-contract
validation retains a sized map, combines request checks with contract capture,
and avoids rebuilding Choice-key maps. The router omits a disabled rate-limit
layer. The client lets separate requests
use separate pooled HTTP/1.1 connections concurrently; the pool still locks
its idle list. A chunked-response parser fix handles size lines split across
socket reads. These changes are covered by the workspace and client overlap
tests, but the measured sequential client improvement is small.

## Added client concurrency case

The candidate adds `client_http/health_parallel_4`: four simultaneous health
requests using the same client and benchmark server. The focused before run
was taken after adding the case but before removing the outer client transport
lock; the focused after run followed that lock change. The complete final
client suite confirmed the result. This is a separate diagnostic, not one of
the original 15 threshold cases.

| Stage | Time, µs (95% interval) |
|---|---:|
| Before lock change | 245.0285 (244.2627–245.8018) |
| After lock change | 109.9232 (109.1372–110.6531) |
| Complete final suite | 109.87 (109.73–110.01) |

The focused ratio is `2.23x`. Each iteration finishes four requests, so its
per-iteration time is not comparable to the sequential health case as a
single-request latency.

```bash
cargo bench --locked -p openkind-client --bench client -- client_http/health_parallel_4 --noplot --save-baseline criterion-0dd269e-20260923-parallel-before
cargo bench --locked -p openkind-client --bench client -- client_http/health_parallel_4 --noplot --save-baseline criterion-0dd269e-20260923-parallel-after
```

## Rejected client latency experiment

A separate, reverted experiment polled the socket for up to 50 µs while
waiting for the response head. The filtered `client_http/health` case measured
55.278 µs (55.065–55.481 µs), only `1.096x` against the fresh 60.610 µs
baseline. It missed the requested `1.20x` and consumed CPU while waiting, so
the candidate retains the asynchronous read path. This exploratory timing is
not included in the 15-case result table.

To separate HTTP work from loopback scheduling, an ignored temporary Tokio
program under `target/tmp-loopback-floor/` timed 100,000 warm sequential
one-byte TCP echo exchanges on the same host. It used Tokio `1.53.1`, a
two-worker runtime, one client task on the calling thread, one server task,
and `TCP_NODELAY` on both sockets. Four serial runs had medians of 56.625,
56.458, 56.625, and 56.583 µs (p95 64.750–65.583 µs). The client's `1.20x`
health threshold is 50.508 µs. Replacing only the echo server with a blocking
thread measured 58.542 µs median. With blocking sockets on both sides, four
serial medians were 50.042, 49.917, 49.792, and 49.875 µs, before any HTTP
parsing, authorization, response generation, or JSON decoding. These different
workloads are scheduling diagnostics, not rigorous lower bounds for the HTTP
client or substitutes for its Criterion case.

```bash
cargo bench --locked -p openkind-client --bench client -- 'client_http/health$' --noplot --save-baseline criterion-0dd269e-20260923-spin50-experiment
```

The [22 September clean-commit record](../2026-09-22-criterion-microbenchmarks/)
predates the starting source commit. Its times are historical context; they
are not the denominator for this `1.20x` target. No native request-path,
classification-quality, or peak-RSS run was made for this candidate.
