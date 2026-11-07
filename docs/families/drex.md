# Family: drex

> Nace.AI's four open decision checkpoints. Three reuse the `kev` pointer
> readout, and Drex Max reuses the `d3` 255-code readout at 27B. Drex v1.5
> is Rust-loadable on CPU. The other three are surveyed only. Support is
> inference-only under the project's license policy.

## Status in openkind

**Drex v1.5: Rust-loadable prototype (CPU).** Profile
`drex-v15:2d495e07ea056d6c4317` (alias `drex:v1.5`), loader
[`DrexEngine`](../../crates/openkind-backends/src/families/drex/mod.rs),
daemon aliases through `--drex-aliases` and `--drex-model-root`, and
catalog-installable with `openkind pull drex:v1.5`. It carries no
model-quality claim. Parity evidence is in the
[bring-up record](../../benchmarks/2026-11-07-drex-v15/README.md).

**Drex v2.0, DLM, and Max: surveyed only.** No profile, no catalog entry, no
fixtures. The analysis, reported results, and license reading live in
[`../RESEARCH.md`](../RESEARCH.md#drex-naceais-open-decision-models-and-tech-report-reviewed-2026-11-07).

| Checkpoint | Revision | Backbone | Readout | License |
|---|---|---|---|---|
| [`nace-ai/drex-v1.5`](https://huggingface.co/nace-ai/drex-v1.5/tree/70a8a22d9b36bc91977efe296c3f59c9f756e6c2) | `70a8a22d…` | Qwen3.5 9B hybrid (32 layers, hidden 4096, 3:1 DeltaNet) | Kev pointer head, `head.pt` | Nace AI Open RAIL-M (modified) |
| [`nace-ai/drex-v2.0`](https://huggingface.co/nace-ai/drex-v2.0/tree/e646674476744e3826caf62b834054151e2aba51) | `e6466744…` | Qwen3.5 MoE (40 layers, hidden 2048, 256 experts, 8 active) | Kev pointer head, `head.pt` | Nace AI Open RAIL-M (modified) |
| [`nace-ai/drex-dlm`](https://huggingface.co/nace-ai/drex-dlm/tree/9947862a2ab9e37a947aae8422c05601d6aa03c3) | `9947862a…` | Efficient-DLM-8B bidirectional encoder (custom `edlm` code) | Kev pointer head with `state_bidir` | CC BY-NC 4.0 |
| [`nace-ai/drex-max-v1`](https://huggingface.co/nace-ai/drex-max-v1/tree/a67fa2808b5d91a4db62629beb14eba6d3f3e53c) | `a67fa280…` | Qwen3.8-27B hybrid (64 layers, hidden 5120) plus ViT | 255-code readout, BF16 `[255, 5120]`, noncausal full attention | Nace AI Open RAIL-M (modified) |

## Architectural shape

| Aspect | Pointer models (v1.5, v2.0, DLM) | Drex Max |
|---|---|---|
| Prompt | Kev row form: state, instruction, `<\|box_start\|>` option `<\|box_end\|>` per option, `<\|fim_suffix\|>` decide token | Images, state, instruction, `A:` to `IW:` coded options, answer token |
| Readout | `k(h_opt) · q(h_decide) / 16`, `d_p = 256`, four FP32 tensors | Bias-free 255-row linear map at the last position, softmax over the first K codes |
| Continuation state | State prefilled once, reused across question rows | None. The noncausal mask makes the state depend on the question |
| Calibration | `T = 1` shipped | `T = 1.0087…` in `decision_config.json` |
| Semantic none | Not reported. Training inserts distractor options | Not reported |
| Text generation | None | None |

## Drex v1.5 implementation

| Aspect | Implementation |
|---|---|
| Backbone | Shared Qwen3.5 text backbone at the `CLEF_FLASH` geometry, BF16 shards mapped in place and widened per use, FP32 compute (`drex-v15/cpu-bf16w-fp32c`) |
| Prompt | The kev renderer under Drex's frozen limits: 16,384 tokens for the state and for each row (the reference serving default). Longer requests fail closed |
| Head | Read from the digest-pinned `head.pt` with candle's pickle reader, which interprets opcodes and never imports or calls anything. No converted head is needed, so nothing is mirrored |
| Questions | One causal row per question, the reference row form for hybrid backbones. No state reuse across questions yet |
| Execution | CPU only. CUDA, MLX, and ONNX selections fail closed |
| Choice order | Sorted label order (the reference keeps caller order), a declared difference |

Parity: [`drex_parity.rs`](../../crates/openkind-backends/tests/drex_parity.rs)
replays the 9-request golden fixture when `OPENKIND_DREX_V15_MODEL_ROOT` is
set. The bundled reference matched every answer to within 1.15e-7.

## What a loader would need for the rest

- **Max:** the `d3` engine with 27B dims, a BF16 readout, and the vision
  encoder. Text-only serving would be a separate, weaker profile.
- **v2.0:** a Qwen3.5 MoE forward path.
- **DLM:** a bidirectional encoder path for the custom `edlm` architecture.
  Its pointer readout is single-pass, so the multi-step refinement concern in
  [`diffusion-decision`](./diffusion-decision.md) does not apply to it.

## License policy

Owner decision, 7 November 2026: support Drex as inference-only. The full
reasoning is in the
[license policy](../RESEARCH.md#license-policy-inference-only-no-mirroring-no-distillation).
Any Drex profile must meet the rules below. Drex v1.5 does: its manifest
sources every file from `nace-ai`, `openkind pull` prints the notice from
`openkind_model_store::license_notice` before downloading, and the proxy
cache refuses to record answers from a teacher whose model name contains
`drex`.

- Pin the `nace-ai` revisions and digests directly, with no Drex file in the
  public registry mirror (weights, configs, tokenizer, or converted heads).
- Show the license name and link at `openkind pull`, warn that the license
  forbids distillation and teacher use of Drex output, and note the
  commercial thresholds and the competing-product exclusion.
- Refuse use as a distillation teacher, including for the proxy-cache
  distilling student. The proxy cache only sees the upstream's reported model
  name, so a Drex deployment behind an alias without `drex` in it gets past
  the check. The license obligation stays with that operator.

## What this page does not say

No accuracy, calibration, or throughput claims. The report's figures are
author-reported and summarized in [`../RESEARCH.md`](../RESEARCH.md). The
fixture timings in the bring-up record are single-host request-path numbers,
not a benchmark. Measured runs belong in [`../BENCHMARKS.md`](../BENCHMARKS.md).
