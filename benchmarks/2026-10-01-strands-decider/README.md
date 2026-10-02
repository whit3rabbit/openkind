# 2026-10-01: strands-decider-2b bring-up, cross-check, and CPU timing

Records for the new [`strands-decider`](../../docs/families/strands-decider.md)
family: the pinned Strands Decider 2B (Hobson v19) profile
(`strands-decider-2b:6a02bb0d1c6b25cae74b`) landing as a Rust-loadable
prototype. Request-path timing and offline parity only — no model-quality
claim. The pinned-source review and its evidence limits live in
[`docs/RESEARCH.md`](../../docs/RESEARCH.md).

## Artifacts and verification

All artifacts were downloaded from the pinned revisions and verified
byte-for-byte against the release's own `MANIFEST.sha256` before any digest
was pinned in `registry/v1/manifests/strands-decider-2b-6a02bb0d1c6b25cae74b.json`:

| Artifact | Source | SHA-256 verified |
|---|---|---|
| `base/model.safetensors` (4,548,221,488 B) | `Qwen/Qwen3.5-2B-Base@b1485b2f` | `928acbf1…5fdf2d` (matches the LFS OID) |
| `adapter/adapter_model.safetensors` (67,324,872 B) | `StrandsAgents/…@bb282d78` | `701bdb89…0a4aebc` |
| `adapter/head.safetensors` (4,213,224 B) | same | `daad0727…38b5287` |
| `adapter/tokenizer.json`, `adapter_config.json`, `hobson_config.json` | same | all six match `MANIFEST.sha256` |

## Golden fixture and reference cross-check

`tests/fixtures/strands_decider_6a02bb0d1c6b25cae74b/golden.json` pins 16
stated-fact cases (6 Choice, 5 Noul, 4 Score, plus a structured object state
with two questions). Generation log: every case assigns the correct option
the highest probability (p 0.59–0.996 single-question; the multi-question
case answers both correctly).

The same 16 requests were replayed through the pinned PyTorch reference
(`strands-labs/strands-decider` @ `f91487ab`, transformers 5.17.0 +
peft 0.21.1, CPU fp32 upcast, identical prompts):

- **Argmax agrees in 17/17 answers.**
- Noul and Score probability drift ≤ 8.5e-4 — inside the 0.005 parity gate.
- Choice drift spans 2e-5 (`descriptions`) to 5.9e-2
  (`structured-multi/os`), one-sided: the reference is consistently
  sharper. The 0.734 choice temperature amplifies a small logit-level
  difference between the two hybrid-recurrence implementations
  (transformers' eager gated-DeltaNet reference vs the candle port); the
  drift is input-dependent rather than option-count- or order-dependent
  (2-, 3-, and 4-option probes all drift on the operating-system states
  while other states sit at ≤ 2e-3). This is the same class of
  cross-implementation residual the `kev` family recorded (fp32 GEMM
  accumulation order there, ≤ 3e-2 on one near-degenerate case), and the
  reference itself reports answer-level flips from BF16 rounding on GPU.
  The committed golden fixture is the self-consistency regression gate; the
  cross-check numbers above are recorded here as the cross-implementation
  residual.

## CPU timing (Apple M4 Max, fp32, request path)

`openkind-bench score` over the seeded shape777 workload
(`benchmarks/2026-10-01-local-model-suite/shape777.jsonl`), `--reps 1`,
warm process, model load reported separately:

| Workload | Rows | Decisions/s | Input tok/s | Peak RSS | Model load |
|---|---:|---:|---:|---:|---:|
| shape-smoke-12 | 12 | 0.336 | 93.8 | 9.13 GB | 9.1 s |
| shape777 (`be397bfc…`) | 777 | 0.305 | 85.3 | 9.01 GB | 10.3 s |

Summaries: `summary-strands-decider-2b.json` (shape777) and
`smoke/summary-strands-decider-2b.json` (smoke-12), with per-row predictions.
Both runs are single-rep warm-process request-path timings on
`openkind-mac-arm64-local` (Apple M4 Max, 14 logical cores); the shape777 run
shared the machine with a concurrent benchmark process, so its absolute
timings carry that contention.

Throughput is not comparable to the encoder families (a 2B hybrid decoder
forward per row) and sits between the 0.6B pointer engine (`kev`, 4.84/s)
and the 4B slot-logit engine (`decider-4b`, 0.06/s smoke). The 9.13 GB peak
RSS reflects the FP32-widened 2B torso plus the mmap'd checkpoint.

## What this record does not say

No task-quality measurement: rust-loadable status carries no quality claim,
and the authors' published JevBench numbers are development-exposed
observations per the research review. No CUDA, ONNX, or MLX execution was
measured; the pointer-head readout has no ONNX export, and no MLX module
exists for this family.
