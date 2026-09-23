# Criterion component benchmarks, 22 September 2026

This is a local timing baseline for the CLI, in-memory HTTP server, and Rust
client. All cases use `MockEngine`. The 1, 8, and 32-question cases use one
fixed text state and `Noul` questions. These timings describe component
overhead. They are not native-model throughput or classification accuracy.

## Run identity

- Source commit: `78bcd94ca34381627dbe3c92e7062643ee8e8bf7`, clean checkout
  before the record was written.
- Host: Mac16,5, Apple M4 Max, 36 GiB RAM, macOS 26.6.2 (25G83), AC power.
- Toolchain: `rustc 1.98.1`, `cargo 1.98.1`, Criterion `0.8.2`.
- Cargo `bench` profile, 3-second warmup, 5-second measurement, 100 samples,
  95% confidence interval, plots disabled. No timing threshold was applied.
- Results: Criterion's slope point estimate and 95% interval, converted from
  nanoseconds to microseconds. Raw local reports are under ignored
  `target/criterion/`. An overlapping profiling run invalidated the first
  client attempt, so only the later client baseline is reported below.

```bash
cargo bench --locked -p openkind-cli --bench cli -- --noplot --save-baseline criterion-78bcd94-20260922
cargo bench --locked -p openkind-server --bench server -- --noplot --save-baseline criterion-78bcd94-20260922
cargo bench --locked -p openkind-client --bench client -- --noplot --save-baseline criterion-78bcd94-20260922-client-idle
```

## Results

| Target | Case | Time, µs (95% interval) | Throughput |
|---|---|---:|---:|
| CLI | parse `inspect` | 4.0110 (3.9816–4.0416) | n/a |
| CLI | parse `evaluate` | 6.5215 (6.4087–6.6655) | n/a |
| CLI | parse `serve` | 6.1544 (6.1314–6.1767) | n/a |
| CLI | parse `version` | 3.3289 (3.3169–3.3398) | n/a |
| CLI | inspect, 1 question | 0.2988 (0.2974–0.3003) | 405.35 MiB/s |
| CLI | inspect, 8 questions | 1.5091 (1.5012–1.5172) | 301.44 MiB/s |
| CLI | inspect, 32 questions | 5.7463 (5.6938–5.8043) | 285.62 MiB/s |
| Server | `/health` | 3.0630 (3.0468–3.0808) | n/a |
| Server | `/v1/systemone`, 1 question | 4.7339 (4.7153–4.7530) | 211.24k questions/s |
| Server | `/v1/systemone`, 8 questions | 10.1015 (10.0646–10.1403) | 791.96k questions/s |
| Server | `/v1/systemone`, 32 questions | 28.2679 (28.1441–28.3996) | 1.1320M questions/s |
| Client | `health` | 79.9724 (79.4189–80.4919) | n/a |
| Client | `evaluate`, 1 question | 90.0815 (89.0786–91.1192) | 11.101k questions/s |
| Client | `evaluate`, 8 questions | 100.4558 (99.5662–101.4029) | 79.637k questions/s |
| Client | `evaluate`, 32 questions | 134.0008 (132.8646–135.2871) | 238.80k questions/s |

CLI inspect throughput counts input JSON bytes. Server and client throughput
count questions answered per request, not concurrent requests. The server
target runs Axum in memory. The client target uses a warmed localhost TCP
connection with retries and rate limiting disabled. For the exact measured
regions and excluded work, see [the methodology](../../BENCHMARKS.md#criterion-microbenchmarks).

## Peak resident memory

Each row below profiles only its 32-question case for 5 seconds in a separate
process under macOS `/usr/bin/time -l`. The helper built the exact bench target
with `--locked` and ran its executable directly. The working tree was dirty
only because this documentation record had been added; the benchmark source
and Cargo files still matched the commit above.

| Target | Exact filter | Maximum resident set size |
|---|---|---:|
| CLI | `cli_inspect/questions/32` | 7,749,632 bytes (7.39 MiB) |
| Server | `server_http_systemone/questions/32` | 59,031,552 bytes (56.30 MiB) |
| Client | `client_http_systemone/questions/32` | 75,169,792 bytes (71.69 MiB) |

```bash
scripts/bench-rss.sh --package openkind-cli --bench cli --filter cli_inspect/questions/32 --duration 5
scripts/bench-rss.sh --package openkind-server --bench server --filter server_http_systemone/questions/32 --duration 5
scripts/bench-rss.sh --package openkind-client --bench client --filter client_http_systemone/questions/32 --duration 5
```

The provenance logs are under ignored `target/bench-rss/78bcd94ca343/`.
These values include the benchmark process, Criterion, runtime, and allocator
retention. They are whole-process high-water marks, not memory per request.
