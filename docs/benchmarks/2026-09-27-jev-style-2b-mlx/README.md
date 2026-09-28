# Jev-Style 2B MLX survey

This record checks the upstream MLX runtime at Hub revision
[`b86e4cbc6f420f5d7d1691299d62db272934ebf4`](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX/tree/b86e4cbc6f420f5d7d1691299d62db272934ebf4)
and compares it with OpenKind's pinned Qwen3.5-4B MLX engine on the local
Apple M4 Max. It is throughput evidence only. The models have different
readouts and inference contracts, and these runs do not compare task quality.

Both precision folders passed the upstream runtime's selected-folder SHA-256
manifest verification and returned finite probabilities summing to one on the
same local choice request. The BF16 and 8-bit smokes selected the same option.
The upstream parity and task-quality numbers remain author-reported; OpenKind
did not reproduce those evaluations.

## Local M4 Max timings

| Serialized state tokens | Questions per call | Jev-Style BF16 p50 | Jev-Style 8-bit p50 | OpenKind Qwen3.5-4B p50 |
|---:|---:|---:|---:|---:|
| 878 | 1 | 0.291 s | 0.334 s | 10.642 s |
| 878 | 10 | 0.568 s | 0.714 s | 16.207 s |
| 3,950 | 1 | 1.207 s | 1.645 s | Rejected by the profile limit |
| 3,950 | 10 | 1.552 s | 2.023 s | Rejected by the profile limit |
| 24,436 | 1 | 9.040 s | 10.772 s | Not run, same profile limit |
| 24,436 | 10 | 9.591 s | 10.819 s | Not run, same profile limit |

Jev-Style timings are the median of three warmed `score_many` calls. Each call
recomputed the state. The one-question case is a choice question. The ten
questions are four choice, four true/false, and two score questions about the
same state. The largest local Jev-Style input was 24,518 tokens, within its
25,600-token budget. Calls include tokenization and rendering; model load and
manifest verification are recorded separately.

The final column is the local `qwen35-mlx-fp32` production path with the
`nested_sequential` strategy. At 878 state tokens its p95 values were 13.615 s
for one question and 23.404 s for ten, from three samples. The OpenKind loader
reports 21.888 to 24.182 s of model load separately. Its 3,950-token request
failed closed with `candidate sequence length 3999 exceeds frozen maximum
1792; truncation is forbidden`. This limit is part of the pinned OpenKind
protocol, so the 4B comparator does not cover the longer Jev-Style rows.

At 878 state tokens, the measured OpenKind 4B p50 was about 36.6 times the
Jev-Style BF16 p50 for one question and 28.5 times for ten questions. These
ratios combine checkpoint size, FP32 versus BF16 compute, renderer, and readout
differences. The ten-question OpenKind samples ranged from 16.022 s to 23.404 s.

BF16 was faster in each of the six local rows. The 8-bit weights use about
half the disk size and less MLX memory, but the 8-bit session ran under higher
background load, so these relative timings are indicative. The runs were on a
shared M4 Max, with the 1-minute load average recorded at 7.3 for BF16 and 16.6
for 8-bit. There were only three samples per cell.

| Precision | Weight file | Load plus manifest verification | MLX active after load | MLX peak during run | Process peak RSS |
|---|---:|---:|---:|---:|---:|
| BF16 | 3,763,691,755 bytes | 2.633 s | 4,085,614,218 bytes | 5,946,771,126 bytes | 4,210,114,560 bytes |
| 8-bit | 2,000,043,057 bytes | 2.033 s | 2,321,917,578 bytes | 4,350,797,494 bytes | 2,426,880,000 bytes |

After precomputing the state, a further question took 0.043 / 0.048 / 0.090 s
for BF16 at 878 / 3,950 / 24,436 tokens. The corresponding 8-bit times were
0.043 / 0.052 / 0.108 s. The runtime reuses the cached state only for an exact
prefix match.

## Upstream M1 Max reference

The pinned upstream README reports these median call times on an Apple M1 Max
with 64 GB memory. It used English documentation and source-code states, three
calls per cell, and the same one-question or ten-question mix. Model loading
(3.1 to 3.7 s) is excluded. The author notes that other jobs shared that host,
so these are reference measurements rather than a clean baseline.

| State tokens | Questions per call | BF16 | 8-bit |
|---:|---:|---:|---:|
| 878 | 1 | 0.57 s | 0.72 s |
| 878 | 10 | 1.07 s | 1.27 s |
| 3,950 | 1 | 2.26 s | 2.96 s |
| 3,950 | 10 | 2.80 s | 3.54 s |
| 24,436 | 1 | 15.5 s | 19.9 s |
| 24,436 | 10 | 16.2 s | 20.8 s |
| 24,436, state already computed | 1 | 0.15 s | 0.16 s |

The local generated states use repeated prose and source-like code to match
these token-count shapes. They are not the upstream author's exact texts. The
local 2B and 4B runs use the same generated state and typed question rows, but
different renderers and model execution paths. Treat the table as a same-host
throughput comparison, not a head-to-head model-quality result.

## Reproduce

Follow the pinned download and environment commands in the
[family profile page](../../families/jev-style.md), then run the benchmark
script once per precision:

```bash
python docs/benchmarks/2026-09-27-jev-style-2b-mlx/run_local_bench.py \
  --model-dir "$HOME/.cache/openkind/jev-style-2b-v3-mlx" \
  --precision bf16 \
  --output-dir "$HOME/.cache/openkind/benchmarks/jev-style-2b/m4-max" \
  --reps 3

python docs/benchmarks/2026-09-27-jev-style-2b-mlx/run_local_bench.py \
  --model-dir "$HOME/.cache/openkind/jev-style-2b-v3-mlx" \
  --precision 8bit \
  --output-dir "$HOME/.cache/openkind/benchmarks/jev-style-2b/m4-max" \
  --reps 3
```

The script verifies the selected precision manifest, writes six matching
OpenKind JSONL workloads, warms each shape once, clears the cached state before
each timed call, and writes a JSON result. To repeat the short-context OpenKind
comparison, use the generated `openkind-s878-q1.jsonl` and
`openkind-s878-q10.jsonl` with `openkind-bench score`,
`--engine qwen35-mlx-fp32`, and `--strategies nested_sequential`. Include
`--host` and `--commit` in the command. The 3,950 and 24,436 token workloads
cannot run through this OpenKind profile because its frozen per-candidate
maximum is 1,792 tokens.

The detailed local measurements, samples, memory counters, and OpenKind fixture
hashes are in [`measurements.json`](./measurements.json). This run used checkout
revision `03caf4ee650ac5c6a67a0506a9e0e53258d734b9` with a dirty working tree;
it is exploratory and does not qualify an OpenKind release or the Jev-Style
model's quality.
