# FP32 MLX candidate pooling diagnostic

Candidate pooling combines candidate lanes from different questions when
their continuation positions match. This record rejects promotion on the
tested shapes. The service and automatic scheduler continue to use the
existing per-question batched runner.

## Identity and checks

- Subject: `df1d4702e065286cb32722b022414ac7318a72eb` on `Mac16,5`
  (Apple M4 Max, 36 GiB).
- Profile: `a047d6802c3f06f085b8`, pinned
  `Qwen/Qwen3.5-4B-Base` revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Runtime: MLX 0.32.2 FP32 `ReferenceOps`, Xcode 27.0 build 27A266a,
  Metal 32023.921. The local runtime qualification passed.
- The full-sequence parity gate passed with maximum probability error
  `1.514845e-6`, zero selection changes, and zero policy changes. The
  nested gate passed with maximum probability error `1.378739e-5`, complete
  state isolation/position checks, and the unequal-length vectorized gate.
- The pooled frozen three-question replay had maximum probability error
  `5.669357e-7` against the golden distributions. It kept all selections
  and policy actions (`review`, `accept:0`, `accept:2`), derived with the
  unchanged `0.98` threshold. The pinned gate details are in
  [`parity_summary.json`](parity_summary.json).
- The workspace test and bench-smoke batteries, the MLX feature tests, and
  workspace and MLX clippy passes completed. These tests do not measure
  service load or semantic classification quality.

## Stage replay

Each strategy ran in its own process. Two paired orders were used:
`sequential → batched → pooled` and `pooled → batched → sequential`.
Q2/K2 has five timed iterations per process; Q8/K4 has three. The table
uses the median of all timed iterations for each shape and strategy. The
reported process and MLX memory peaks in
[`stage_samples.json`](stage_samples.json) include model load and earlier
iterations in that process.

| Shape | Strategy | Median (ms) | Candidate stage (ms) | Physical forwards | Padded token slots |
|---|---|---:|---:|---:|---:|
| Q2/K2 | `nested_sequential` | 1,334.5 | 543.6 | 7 | 0 |
| Q2/K2 | `nested_batched` | 1,290.1 | 518.4 | 4 | 0 |
| Q2/K2 | `pooled` | 1,309.7 | 519.7 | 3 | 2 |
| Q8/K4 | `nested_sequential` | 5,753.0 | 4,182.1 | 41 | 0 |
| Q8/K4 | `nested_batched` | 4,948.3 | 3,534.7 | 10 | 14 |
| Q8/K4 | `pooled` | 5,332.5 | 3,882.9 | 6 | 18 |

Pooling is 1.5% slower than current batching at Q2/K2 and 7.8% slower at
Q8/K4. Against sequential execution it is only 1.9% and 7.3% faster,
respectively. Its Q8/K4 observed p95 is 7,787 ms versus 7,049 ms for
current batching; with six samples per strategy, p95 is the maximum sample,
not a stable tail estimate. Pooled peak active MLX memory is also higher in
both shapes. Fewer physical forwards did not translate into lower latency.

The Q8/K4 workload repeats frozen question and candidate token sequences to
create a matched load shape with unequal candidate suffix lengths. Its
predictions are implementation-parity diagnostics, not semantic-quality
evidence. Maximum pooled probability differences from sequential execution
were `1.99e-5` for Q2/K2 and `7.59e-7` for Q8/K4, with unchanged selected
indices.

The external Qwen2.5 Python field-batching pattern has a direct counterpart
in the existing Rust question stage: [`run_batched_questions`](../../../crates/openkind-backends/src/qwen35/backbone/batched.rs)
forks the shared root and [`continue_batch_vectorized`](../../../crates/openkind-backends/src/qwen35/mlx/model.rs)
right-pads unequal suffixes, executes one MLX graph, and gathers each last
real token. The paired stage samples show that this question stage improved
from `247.0` to `233.4` ms and `248.3` to `233.2` ms at Q2 (5.5% to 6.1%), and
from `949.3` to `832.0` ms and `971.0` to `822.4` ms at Q8 (12.4% to 15.3%).
The current Rust vectorized path accepts 2 to 8 lanes, so the Python Q28
single-batch result does not transfer to this backend.
Those timings are in [`stage_samples.json`](stage_samples.json). They cover
the pinned Qwen3.5 FP32 Rust backend, not the separate quantized Qwen2.5
checkpoint. The full-request results below do not establish a speedup from
this question-stage gain.

## Full request baseline

The [`q2k2.jsonl`](q2k2.jsonl) and [`q8k4.jsonl`](q8k4.jsonl) workloads
put same-length question text and unequal candidate descriptions in one
state-grouped request. The existing `openkind-bench score` path was forced
to `nested_sequential` or `nested_batched`, one strategy per fresh process,
with an untimed warmup and three timed repetitions. It includes rendering,
tokenization, validation, backbone, and readout. The pooled diagnostic is not
wired into this request path.

| Shape | Pair 1 sequential / batched p50 (s) | Pair 2 sequential / batched p50 (s) |
|---|---:|---:|
| Q2/K2 | 1.330 / 1.395 | 1.373 / 1.409 |
| Q8/K4 | 6.802 / 7.221 | 7.778 / 15.985 |

The second Q8/K4 batched process drifted sharply across its repetitions
(`9.765`, `15.985`, `16.858` s), so these full-request runs do not establish
a stable strategy ratio. The forced strategies kept the same selections;
maximum wire probability differences were `2.69e-8` at Q2/K2 and
`3.28e-6` at Q8/K4. Per-process timings, load times, memory peaks, and
fixture checksums are in
[`full_request_baseline.json`](full_request_baseline.json).
Per-file evidence checksums are in [`SHA256SUMS`](SHA256SUMS).

## Decision and reproduction

The pooled path misses the required 10% median improvement even in the
smaller stage diagnostic, and it loses to the current batched path on both
shapes. Retain it as an opt-in diagnostic only. Do not advertise pooled
capability in the automatic scheduler or route service requests through it.
No pooled full-request or queue-inclusive service claim follows from this
record.

Build the diagnostic with `SDKROOT=$(xcrun --show-sdk-path)` and
`cargo build --release -p openkind-backends --features mlx --example
qwen35_candidate_pool_bench`. Run the resulting binary with
`--checkpoint <pinned-checkpoint-dir> --q 2 --k 2 --iterations 5
--max-lanes 8 --strategy <nested_sequential|nested_batched|pooled>`.
Repeat with `--q 8 --k 4 --iterations 3`, reversing process order for the
second pair. `--q 3 --k 0` replays the natural frozen fixture for decision
parity. The command and output fields are described in
[`../../BENCHMARKS.md`](../../BENCHMARKS.md).
