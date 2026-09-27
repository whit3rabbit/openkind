# Rust flat-field batching diagnostic

The Python field-batching schedule was ported as an opt-in Qwen3.5 Rust
diagnostic. It prefills one root, concatenates each question and candidate
suffix, and advances complete field lanes from that root in right-padded MLX
batches of at most eight. The existing `nested_batched` path instead advances
questions once and reuses each question state for its candidates. Both feed the
same score-summary readout. Neither the service path nor automatic scheduler
uses the flat runner.

## Identity and method

- Host: `Mac16,5`, Apple M4 Max, 36 GiB RAM.
- Profile: `a047d6802c3f06f085b8`, pinned
  `Qwen/Qwen3.5-4B-Base` revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Runtime: MLX 0.32.2 FP32 `ReferenceOps`, Xcode 27.0 build 27A266a,
  Metal 32023.921. The exact arithmetic identity is in each raw run.
- Source: [`run_flat_batched_candidates`](../../../crates/openkind-backends/src/qwen35/backbone/batched/flat.rs)
  and [`qwen35_candidate_pool_bench`](../../../crates/openkind-backends/examples/qwen35_candidate_pool_bench.rs).
- Each strategy ran in a fresh process. Two process orders were used:
  `nested_batched, flat` and `flat, nested_batched`.
- Each process ran one untimed-for-summary warmup sample, then three Q2/K2 or
  two Q8/K4 timed samples. The table reports each process's timed median.
- The timed region includes prefill, continuation, and readout. It excludes
  model load, rendering, tokenization, validation, and HTTP/gRPC handling.
  Q8/K4 repeats frozen token suffixes to create a load shape, not a semantic
  quality dataset.

## Paired results

| Shape | Pair | Nested batched median | Flat median | Flat change |
|---|---:|---:|---:|---:|
| Q2/K2 | 1 | 1,341.9 ms | 1,422.1 ms | +6.0% |
| Q2/K2 | 2 | 1,268.5 ms | 1,421.5 ms | +12.1% |
| Q8/K4 | 1 | 5,004.9 ms | 8,221.8 ms | +64.3% |
| Q8/K4 | 2 | 5,004.8 ms | 8,231.9 ms | +64.5% |

At Q2/K2, flat suffix continuation takes 867.6 to 875.0 ms after warmup,
while the current question plus candidate stages take 737.3 to 795.2 ms.
At Q8/K4, flat suffix continuation takes 7,593.9 to 7,598.0 ms, while the
current question plus candidate stages take 4,433.9 to 4,435.7 ms. The flat
path cuts physical forwards from four to two at Q2/K2 and from ten to five
at Q8/K4, but repeats question tokens per candidate. Right-padding grows
from zero to two token slots at Q2/K2 and from 14 to 104 at Q8/K4.

Selections match in both shapes. Maximum probability difference is
`2.71e-6` at Q2/K2 and `8.08e-7` at Q8/K4, below the frozen `0.005`
tolerance. Every top probability is at least `0.01099` from the `0.98`
policy threshold, so the unchanged selections imply unchanged policy actions.
The flat runner's mixed-length, chunking, ordering, and singleton fallback
tests pass. These checks establish implementation parity on the tested
inputs, not classification quality.

Q2/K2 peak active MLX memory is about 14.0 GiB nested versus 14.38 GiB flat.
Q8/K4 is 16.49 GiB nested versus 15.43 GiB flat. Process peak RSS is mixed
across pairs. With only two paired processes and two or three timed samples
per process, these runs do not establish tail latency or a stable memory
ratio. The latency loss alone rejects promotion. Retain the flat runner as a
diagnostic and keep the current scheduler choice.

The raw paired runs are [`q2k2.json`](q2k2.json) and
[`q8k4.json`](q8k4.json); their checksums are in [`SHA256SUMS`](SHA256SUMS).

## Reproduce

Build against the pinned local MLX runtime, then run one shape at a time on a
quiet host. The script starts a new process for every strategy run and writes
the evidence file after each process.

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo build --release \
  -p openkind-backends --features mlx --example qwen35_candidate_pool_bench
python3 scripts/bench_qwen35_flat_field.py \
  --binary target/release/examples/qwen35_candidate_pool_bench \
  --checkpoint <pinned-checkpoint-dir> \
  --output docs/benchmarks/2026-09-27-python-flat-field/q2k2.json \
  --q 2 --k 2 --iterations 4 --pairs 2
python3 scripts/bench_qwen35_flat_field.py \
  --binary target/release/examples/qwen35_candidate_pool_bench \
  --checkpoint <pinned-checkpoint-dir> \
  --output docs/benchmarks/2026-09-27-python-flat-field/q8k4.json \
  --q 8 --k 4 --iterations 3 --pairs 2
```
