# 2026-11-07: drex-v15 bring-up and reference cross-check

The pinned Drex v1.5 profile (`drex-v15:2d495e07ea056d6c4317`) matches Nace.AI's
own reference code to within 1.2e-7 in probability on every fixture answer, with
no argmax flip. This record covers offline parity and request-path cost on one
host. It makes no model-quality claim: the fixture cases state the correct
answer verbatim. The source review, license reading, and inference-only policy
live in [`docs/RESEARCH.md`](../../docs/RESEARCH.md#drex-naceais-open-decision-models-and-tech-report-reviewed-2026-11-07).

## Host and code

- Apple M4 Max, 36 GiB, macOS 26.6.2.
- OpenKind at `b9061ee3` plus the uncommitted Drex changes this record
  describes.
- Execution: `drex-v15/cpu-bf16w-fp32c` (BF16 weights widened per use, FP32
  compute, candle CPU).

## Artifacts and verification

`hf download nace-ai/drex-v1.5 --revision 70a8a22d9b36bc91977efe296c3f59c9f756e6c2`
fetched the checkpoint. The local SHA-256 of every LFS object (four shards,
`head.pt`, `tokenizer.json`) equals the Hub's LFS OID at that revision, and
the loader verifies each digest in place before reading weights.

Nothing is mirrored. The loader reads the original `head.pt` through candle's pickle
reader after its digest check. The pickle names only `collections.OrderedDict`,
`torch._utils._rebuild_tensor_v2`, and `torch.FloatStorage`. Its metadata
records `state_bidir: false`, `option_isolation: false`, `lora: 0`,
`head_dim: 256`, and `temperature: 1.0`.

## Golden fixture

[`golden.json`](../../crates/openkind-backends/tests/fixtures/drex_2d495e07ea056d6c4317/golden.json)
pins 9 requests with 11 answers: 3 Choice (bare keys and descriptions), 3 Noul
(default and explicit criteria), 2 Score, and one structured JSON state with
a Noul, a Choice, and a Score. Every answer gives the stated option the highest
probability. The lowest correct-option probability is 0.9808, on `noul-false`
(p(true) = 0.0192).

`gen_drex_fixture` (release build) took 3.2 to 3.9 s per single-question
request and 9.7 s for the three-question request, with a 15.0 GB maximum
resident set. The resident set includes mapped shard pages. These are
fixture-run timings, not the seeded shape777 benchmark, which has not been
run for this profile.

## Reference cross-check

[`scripts/drex_reference.py`](../../scripts/drex_reference.py) replayed the
same 9 requests through the checkpoint's bundled `code/kev/` package (torch
2.8.0, transformers 5.17.0, Python 3.12.11, CPU). Weights load in BF16, and
each decoder layer, the embedding output, and the final norm run in FP32. That
is the same arithmetic as the Rust oracle. Choice criteria were sorted before
scoring to match the engine's declared option order.

- Largest probability difference over all 11 answers: 1.15e-7.
- Argmax flips: none.
- Reference wall time: 8.7 to 17.5 s per request (transformers' pure-PyTorch
  gated-DeltaNet and causal-conv fallbacks).

The per-answer numbers are in
[`reference-crosscheck.json`](reference-crosscheck.json). The parity gate in
[`drex_parity.rs`](../../crates/openkind-backends/tests/drex_parity.rs) uses
the family-standard 0.005 tolerance and replays only when
`OPENKIND_DREX_V15_MODEL_ROOT` points at the checkpoint.

## Not covered

- Task quality on labeled datasets.
- The shape777 throughput and memory benchmark.
- Requests near the 16,384-token frozen state and row limits.
- Any execution path other than CPU. CUDA, MLX, and ONNX fail closed.
