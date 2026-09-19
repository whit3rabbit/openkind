# opendecision — Roadmap

> Jev-compatible, open-source decision-inference engine in Rust.
> Wire spec: <https://docs.typesafe.ai/api>

This is the single source of truth for what's done, what's next, and
what the open questions are. Each phase ships with passing tests at
HEAD.

---

## Phase 0 — wire contract ✅ DONE

**Goal:** lock the Jev schema before any server exists. Every TypeSafe
spec example must round-trip through our types without data loss.

- [x] `opendecision-core` crate: `SystemRequest` / `SystemResponse` /
      `Answer` with `f64` (not `f32`) precision on `probabilities`,
      `score`, `noul`, `confidence`. `0.92f32` round-trips to
      `0.9200000166893005` — that was the trap.
- [x] `validate_request()` covering every documented validation rule.
- [x] **23** conformance tests in
      `crates/opendecision-core/tests/conformance.rs` — one per Jev spec
      example fixture.
- [x] `gen-schemas` binary regenerates
      `crates/opendecision-core/schemas/jev-v1-{request,response}.json`
      from the Rust types via `schemars`.
- [x] **8** example JSON fixtures committed in `examples/`
      (`01_noul.json` … `08_response_score.json`).

---

## Phase 1 — daemon + SDK compatibility ✅ DONE

**Goal:** a self-hostable server that the future `typesafe_sdk` Python
client can speak to without modification.

- [x] **HTTP** (axum 0.8): `POST /v1/systemone`, `GET /v1/models`,
      `GET /health`, `GET /metrics`.
- [x] **gRPC** (tonic 0.14): `opendecision.system_one.SystemOne` with one
      `evaluate` RPC, request_id propagated via tonic interceptor.
- [x] `opendecisiond` daemon + `opendecision` CLI (`serve`, `evaluate`,
      `inspect`, `version`).
- [x] `DecisionEngine` trait (`backend_id`, `model_metadata`,
      `evaluate`) + `EngineRegistry` alias → `Arc<dyn …>`.
- [x] `MockEngine` — deterministic, seeded RNG, slightly jittered.
- [x] **SDK compatibility surface** (the bits the future client
      actually depends on):
  - `x-typesafe-request-id` stamped on **every** response, including
    401s (middleware runs outermost).
  - Bearer auth opt-in via `OPENDECISION_API_KEY`, constant-time compare.
  - `Retry-After` / `retry-after-ms` headers on 429 + 529.
  - Error envelope `{"error":{"code",message}}` mapped to the Python
    SDK's `TypeSafe{Authentication,RateLimit,BadRequest,…}Error`.
  - `/v1/models` returns `{"models":[{name,description,release_date}]}`.

### Test totals at HEAD

| Suite                                              | Tests |
|----------------------------------------------------|-------|
| `opendecision-core` unit + conformance                 | 33    |
| `opendecision-engine` unit (MockEngine, dispatch)      | 14    |
| `opendecision-api` unit (middleware, error mapping)    | 22    |
| `opendecision-api/tests/sdk_compat.rs`                 | 56    |
| `opendecision-api/tests/grpc_roundtrip.rs`             | 8     |
| `opendecision-cli` unit                                | 7     |
| `opendecision-server` unit                             | 2     |
| `opendecision-proto` unit                              | 2     |
| **Total at HEAD**                                  | **144** |

(Exact unit-test count drifts with engine/HTTP additions; SDK-compat
and conformance are the pinned contract.)

---

## Phase 2 — real model backends 🚧 STARTED

**Goal:** go from "mock" to "does useful work" without changing the
wire format. The hypothesis: Jev decisions are a single forward pass
on a 2560-dim hidden vector + a small classifier head, *not* a
generation loop.

### Phase 2A — Python exploration (DONE)

A PyTorch notebook on Colab validated the architecture hypothesis
before we burn time on the Rust implementation.

**Model probed:** `Qwen/Qwen3.5-4B-Base` (via
`Qwen3_5ForConditionalGeneration`).

| Property                         | Value                          |
|----------------------------------|--------------------------------|
| Backbone params (`AutoModel`)    | 4.539B                         |
| Full params (`AutoModelForCausalLM`) | 4.206B                     |
| Hidden size                      | 2560                           |
| Decoder layers                   | 32                             |
| Vocab                            | 248,320                        |
| Token embedding                  | `Embedding(248320, 2560)`      |

**Key finding 1 — generation vs decision latency**

| Path                          | Wall time |
|-------------------------------|-----------|
| Generative (100 tokens out)   | 7.48 s    |
| Decision (1 forward pass)     | 0.26 s    |
| Speedup                       | **~28.6×** |

Note this is a generous bound for the decision path (1 forward, no
KV-cache walk-off, no token-by-token I/O). Real-world speedup will
depend on sequence length and batch composition — but the scaling
regime changes fundamentally: linear-in-tokens → constant per query.

**Key finding 2 — Qwen3.5 layer mix**

The model alternates `Qwen3_5GatedDeltaNet` (recurrent-style state)
and `Qwen3_5Attention` (standard Q/K/V) layers. This is *relevant*
to OpenDecision because it suggests the backbone already produces a
reusable state representation — which has direct implications for
the shared-state cache design in the Phase 2F reference-engine workstream below.

**Key finding 3 — embeddings are tied**

`embed_tokens` lives inside the backbone. Stripping `AutoModelForCausalLM`
does *not* remove billions of params — it removes the generation
loop, not the LM head. The win is in compute, not in disk.

### Phase 2B — decision head + benchmarks (DONE / MEASURED)

Run ID: `20260917T205849Z` (archived in `research/opendecision_phase2b_20260917T205849Z`).
Evaluated on **NVIDIA L4 GPU** using native `torch.bfloat16`. LoRA was disabled.

Core question answered: **Yes, a frozen Qwen3.5-4B backbone + a 7,683-parameter linear classifier head achieves 87.67% matched / 87.33% mismatched test accuracy on MultiNLI** (trained on 2,400 examples).

**1. Head & Pooling Hierarchy:**

| Head & Pooling | Matched Acc | Mismatched Acc | Matched NLL | Status |
|---|---|---|---|---|
| **Last token + linear** (`winner`) | **87.67%** | **87.33%** | **0.3345** | Selected by dev NLL (0.3459) |
| Last token + MLP | 87.83% | 88.67% | 0.3361 | Retained for multi-seed eval |
| Max + MLP | 86.00% | 86.50% | 0.3745 | Viable baseline |
| Learned attention + MLP | 86.00% | 84.83% | 0.3591 | Underperformed last-token |
| Max + linear | 84.83% | 85.33% | 0.3920 | |
| Mean + MLP | 75.33% | 76.17% | 0.5952 | Poor |
| Mean + linear | 74.67% | 76.33% | 0.6048 | Poor |
| *Untuned finite-code baseline* | *78.83%* | *81.50%* | *0.5122* | Original LM projection |
| *Class-prior baseline* | *33.17%* | *33.00%* | *1.0995* | ECE ~0.010 (misleading) |

- **Supervised Head Gain**: Improves accuracy by **+8.83 pp** (matched) and **+5.83 pp** (mismatched) over untuned token projection.
- **Pooling Choice**: Last-token pooling won decisively; learned attention pooling underperformed and added unnecessary complexity.
- **Selection**: `last_linear_seed17` was selected via dev NLL. Both last-token linear and MLP remain viable under sampling error (95% CI: 85.2–90.2%).

**2. Parameter & Memory Accounting:**
- Text backbone: **4,205,751,296 parameters**.
- Weight tying: `embed_tokens` is tied to `lm_head`; dropping the LM head interface saves compute during inference but **0 marginal weights** on disk.
- Linear head size: $2560 \times 3 + 3 =$ **7,683 parameters** (~51.5 KB safetensors).
- Resident VRAM: **8,039 MiB (~7.85 GiB)** allocated on L4; peak dynamic memory during single-sample inference was **~37 MiB**.

**3. Latency & Workload Scaling:**
- Decision head median latency: **80.75 ms** (Python end-to-end: **81.05 ms**).
- Versus generation:
  - Generate 1 token: 91.32 ms (**1.13×** decision latency, ~10.5 ms difference).
  - Generate 8 tokens: 533.54 ms (**6.61×** decision latency).
  - Generate 32 tokens: 2,035.35 ms (**25.21×** decision latency).
- Takeaway: speedup scales with tokens avoided. Avoiding a paragraph is a ~25× win; avoiding 1 token is a modest ~13% win. The primary win over 1-token generation is accuracy (+8.83 pp).
- Independent batching throughput: 12.38 qps ($B=1$), 25.79 qps ($B=4$), 24.24 qps ($B=8$), with varying sequence lengths (94, 104, 128 tokens).

**4. Calibration & Batching Findings:**
- Temperature scaling ($T=1.1441$ on 300 calibration examples) did not improve test NLL (0.3345 → 0.3392) or Brier score (0.1834 → 0.1854). API must preserve raw probabilities and label scaling `temperature_scaled`.
- Batching invariance: ~1.27% probability difference observed between standalone and padded batch execution. Requires 3-tier diagnostic (same-length duplicate, variable padding, heterogeneous batch) before declaring production invariance.

**5. Exported Reference Artifacts (`research/opendecision_phase2b_20260917T205849Z/frozen_export/`):**
- `manifest.json`: Tokenization and prompt format (`nli-described-abc-v1`).
- `head.safetensors`: Weights for the 7,683-parameter head.
- `golden_head_inputs.npz`: Reference input activations and expected logits for Rust unit testing.

### Phase 2C — Qwen model research: dynamic schemas, batching invariance & model scaling (MEASURED / NEEDS REVIEW)

> **Principle**: Phase 2 is model research in Python/Colab. We do not jump into cutting the Rust inference engine before the model architecture handles arbitrary dynamic schemas, candidate scoring, and verified batching behavior.

Run ID: `20260917T222948Z` (archived in `research/opendecision_phase2c_20260917T222948Z/`). The run used `Qwen/Qwen3.5-4B-Base` on an NVIDIA L4 with native BF16. The dynamic-choice stage completed, but the overall stability status is `needs_review`; LoRA was disabled.

The run supports continuing with the frozen Qwen3.5-4B backbone and the development-selected last-token linear head. It does not yet support batch-invariant production serving, a general abstention detector, shared-state branching across different questions, ordinal `Score` training, or Rust/Metal execution.

#### Replicated frozen-backbone baseline

On fresh 1,000-example MultiNLI test partitions, the selected linear head achieved **87.00% accuracy, 0.3400 NLL, and 0.0194 ECE** on matched data, and **88.80%, 0.3246, and 0.0247** on mismatched data. The calibration gate retained temperature **1.0**. Across three seeds, linear scored **86.87% ± 0.42 pp** matched and **88.70% ± 0.36 pp** mismatched; the MLP scored **87.90% ± 0.36 pp** and **87.77% ± 0.29 pp**. Keep `last_linear` as the reference implementation and retain the MLP as a comparison only.

#### Batching stability is a serving gate

The diagnostic used 200 examples per scenario and a predeclared probability tolerance of 0.5 percentage points. Probability differences are percentage points, not relative percentages:

| Scenario | 95th percentile | Maximum | Selected-class changes |
|---|---:|---:|---:|
| Repeat the example alone | 0.00 pp | 0.00 pp | 0/200 |
| Duplicate in same-length batch of two | 2.97 pp | 5.59 pp | **2/200** |
| Add 32 right-padding tokens | 2.97 pp | 4.94 pp | 0/200 |
| Add 128 right-padding tokens | 2.42 pp | 6.08 pp | 0/200 |
| Mix shorter and longer examples | 3.42 pp | 5.86 pp | **2/200** |
| Left-pad with explicit positions | 3.22 pp | 5.21 pp | **3/200** |

The duplicate result shows that padding alone is not a sufficient explanation. The underlying execution-shape sensitivity is not isolated yet. A separate math-attention diagnostic changed probabilities by up to 2.48 pp across 12 examples, but covered only the full-attention path; the complete FP32 diagnostic was disabled. Treat single-example, unpadded inference as the reference path. Any optimized batch path needs an explicit comparison tolerance and decision-change test. This is an execution-consistency result, not evidence that questions semantically contaminate one another.

#### Dynamic candidate scoring: useful transfer, weak missing-answer handling

The dynamic scorer was trained on 600 message/choice episodes in the Banking77 domain. Each evaluation partition contains 480 episodes from 160 messages, with the listed real candidates plus a `none of these` option:

| Real candidates | Seen-label accuracy | Held-out-label accuracy |
|---:|---:|---:|
| 2 + none | **96.88%** | **88.13%** |
| 4 + none | **86.25%** | **83.13%** |
| 8 + none | **82.50%** | **75.63%** |
| Overall | **88.54%** | **82.29%** |

When the correct candidate was present, accuracy was **95.16%** for seen labels and **93.01%** for held-out labels. When it was absent, `none` recall fell to **65.74%** and **45.37%**. In the eight-candidate held-out slice, the model selected `none` for only **8 of 36** omitted-answer episodes and selected a wrong offered candidate in the other **28**, accounting for approximately 72% of that slice's 39 errors.

The exported design uses a shared message/instruction/candidate scorer plus one global learned `none` scalar. The next ablation should keep the scorer fixed and compare that scalar with a candidate-set-conditioned `none` head using candidate-count and permutation-invariant score summaries. False selection when the correct answer is absent is a primary metric, not an optimization detail. The current `none` target means an omitted Banking77 intent, not insufficient evidence or a general safety abstention.

Candidate order handling passed the narrow reference test: unbatched probability change was 0, batched reordering changed probabilities by at most 1.87 pp, there were no selected-choice changes in 24 episodes, and arbitrary candidate keys did not change tokenization. This does not clear the general batching gate or establish instruction robustness.

#### Calibration and throughput findings

For the selected NLI linear head, a fitted temperature of approximately 0.9816 failed the separate calibration gate, so temperature 1.0 remains selected. The dynamic-choice gate selected **1.0749**, but only on a narrow NLL improvement of approximately **0.00222** against a required **0.002** on 150 validation episodes. Aggregate held-out ECE around 0.0273 did not prevent eight-candidate `none` recall from falling to 22.22%; the corresponding ECE was around 0.0709.

At fixed 128-token inputs, excluding tokenization and server overhead, L4/BF16 throughput was:

| Batch size | Median batch latency | Decisions/second |
|---:|---:|---:|
| 1 | **77.69 ms** | **12.87** |
| 2 | **91.81 ms** | **21.79** |
| 4 | **166.84 ms** | **23.97** |
| 8 | **333.15 ms** | **24.01** |

For this workload, batch eight roughly doubles batch-four latency without improving throughput. Benchmark small, length-aware batches, and validate them against the single-example reference. These are fixed-shape NLI timings, not complete dynamic-choice request timings; the current candidate scorer re-encodes the message for each candidate. Optional flash-linear-attention, FLA, causal-conv1d, and flash-attn packages were absent, so their performance and numerical effects remain unmeasured.

#### Reference artifacts and remaining limits

The archive includes fresh NLI heads and predictions, a dynamic candidate head with golden inputs, and a folded linear projection. The supplied head-parity fixtures show a maximum logit error of approximately **1.8e-6**. This validates the exported head only, not a Rust implementation of the Qwen backbone. Qwen pretraining decontamination, LoRA, Score training, shared-prefix branching, and Rust/Metal execution remain untested.

---

### Phase 2D — numerical reference, rejection policy & complete requests (MEASURED / NEEDS REVIEW)

Run ID: `20260917T234417Z` (archived in `research/opendecision_phase2d_20260917T234417Z/`), built from the Phase 2C archive. Qwen3.5, the NLI head, and the real-candidate scorer remained frozen. Only small `none` heads and the temperature option were fitted. The run does not include KV branching, LoRA, quantization, Rust/Metal, or HTTP validation.

#### FP32 is a numerical reference, not yet a production default

The same shape diagnostics were run under four numerical modes:

| Numerical mode | Largest probability difference | Selected-class changes |
|---|---:|---:|
| BF16, default attention | 6.75 pp | Yes |
| BF16, math attention | 6.99 pp | Yes |
| BF16, strict math settings | 6.91 pp | Yes |
| **FP32, strict math settings** | **0.000727 pp** | **No** |

The FP32 result stayed within the declared tolerance across this diagnostic. Layer traces showed the target embedding unchanged, with differences already visible at the first decoder block in all 20 traced changed-shape comparisons. The trace does not isolate linear attention, projections, normalization, feed-forward, or residual operations. FP32 single-example predictions still differed from BF16 by up to 2.66 pp, including one selected-class change, so FP32 has not been shown to improve task accuracy.

Use the tested FP32 configuration as the numerical reference for follow-up experiments. It roughly doubles weight-only storage from 7.83 GiB in BF16 to 15.67 GiB in FP32 before activations and runtime overhead, and its complete-request latency was not measured. The next goal is selective higher-precision computation that approaches the FP32 agreement without paying the full FP32 resource cost.

#### Set-conditioned `none` handling is promising but policy-dependent

The development-selected model is `set_linear`, a seven-coefficient head over frozen candidate scores. On paired 1,600-episode tests, with 100 distinct messages per partition and a 50% absent-answer stress construction:

| Metric | Seen labels: original → set-conditioned | Held-out labels: original → set-conditioned |
|---|---:|---:|
| Overall accuracy | **64.44% → 83.94%** | **54.56% → 74.56%** |
| False answers when absent | **64.38% → 20.63%** | **74.00% → 26.88%** |
| False abstentions when present | 1.25% → 7.50% | 3.50% → 18.00% |
| Correct answer when present | 93.25% → 88.50% | 83.13% → 76.00% |

On held-out labels, incorrect offered answers fell from 592 to 215, about a 64% relative reduction, while false abstentions rose from 28 to 144. The real-candidate logits did not change. Refitting only the original global scalar captured most of the gain, reaching 83.19% seen-label and 72.56% held-out-label accuracy versus 83.94% and 74.56% for `set_linear`. Candidate-set features add a smaller improvement beyond moving the rejection operating point.

The new head is therefore a candidate for missing-answer-heavy workloads, not a universal default. The 5% versus 25% absent-answer reweightings reversed the preferred NLL model, and these are scenario reweightings of recorded cases, not separate deployment populations:

| Evaluation scenario | Original NLL | Set-conditioned NLL |
|---|---:|---:|
| Seen labels, 5% absent | **0.3477** | 0.3815 |
| Seen labels, 25% absent | 0.7420 | **0.4551** |
| Held-out labels, 5% absent | **0.5684** | 0.6743 |
| Held-out labels, 25% absent | 0.9214 | **0.6836** |

Keep model output and application policy separate. `none` still means that the annotated Banking77 intent was omitted, not that evidence is insufficient or that the domain is unfamiliar. The temperature gate rejected its fitted value and retained 1.0.

#### Candidate batching improves latency but repeats state computation

The complete-request benchmark measured the median of four message-specific medians, without server or network overhead:

| Real candidates | Sequential, batch 1 | Candidate batch 2 | Candidate batch 4 |
|---:|---:|---:|---:|
| 2 | 156 ms | 81 ms | 81 ms |
| 4 | 308 ms | 159 ms | 92 ms |
| 8 | 618 ms | 319 ms | 184 ms |
| 16 | 1,229 ms | 631 ms | 365 ms |

For 4, 8, and 16 candidates, batch four was approximately 3.3 to 3.4 times faster than sequential evaluation. The path still performs K full state encodings, and batching changed probabilities by up to approximately 3.86 pp relative to sequential, unpadded inference. No selected choices changed in this small four-message benchmark, which used the correct intent as present and did not broadly test rejection-boundary cases. Shared-prefix reuse was not implemented or measured.

The Phase 2D result is therefore: preserve two references, a numerical FP32 reference and a decision reference containing the frozen scorer, original global `none`, selected set-conditioned `none`, fixtures, and calibration assumptions. Do not promise universal BF16 use or batch-invariant answers.

### Phase 2E: selective precision, shared-prefix parity & rejection policy (MEASURED / DONE)

Run `20260918T114914072764Z` completed on an NVIDIA L4 with fresh FP32 and
BF16 workers. The [expanded result archive](../research/opendecision_phase2e_expanded_20260918T114914072764Z/)
and its [results README](../research/opendecision_phase2e_expanded_20260918T114914072764Z/README_results.md)
contain the saved raw rows and detailed evidence. An independent reconstruction of 3,072 probability
distributions, policy actions, parity counts, and timing aggregates agreed
with the report. This validates the saved calculations; Qwen was not rerun
for that reconstruction.

- [x] **FP32 cached execution is the numerical reference.** Full-prompt
      batch-four, shared-prefix sequential suffixes, and shared-prefix
      equal-length suffix batches all stayed within the 0.005 probability
      tolerance across 128 episodes. Their largest absolute differences were
      0.00000928, 0.00000776, and 0.00001072, respectively, with 0/128
      tolerance failures, selected-outcome changes, and answer/review changes.
- [x] **Complete hybrid cache isolation passed.** Reusable caches stayed
      unchanged, repeated branches reproduced probabilities, and reversed
      candidate order stayed within the stricter order tolerance. These were
      eight isolation checks per cached strategy, separate from the 128
      numerical comparisons.
- [x] **BF16 is not behavior-preserving for this reference.** Depending on
      the strategy, 78/128 to 80/128 episodes exceeded tolerance, selected
      outcomes changed in 10/128 to 14/128 episodes, and answer/review
      decisions changed in 7/128 to 11/128 episodes. FP32 full-sequential
      versus BF16 full-sequential also changed a selected outcome in 18/128
      episodes and a saved policy in 8/128.
- [x] **FP32 suffix batching provides a useful cached-request speedup.** For
      real candidate counts 2, 4, 8, and 16, shared-prefix batched suffixes
      measured 195, 378, 493, and 764 ms. At 16 candidates this was 2.08x
      faster than sequential full prompts and 1.46x faster than full-prompt
      batch four. Full-prompt batching remained faster at two and four
      candidates, so the scheduler must measure request shape rather than use
      a fixed candidate-count cutoff.
- [x] **Long shared prefixes amplify the benefit.** At 1,024 common-prefix
      tokens and eight synthetic candidates, FP32 cached suffix batching took
      1,270 ms versus 8,855 ms for full sequential and 8,908 ms for
      full-prompt batch four, approximately a 7x speedup. The largest
      probability difference was approximately 0.00000185 with no policy
      output change.
- [x] **Profiling identifies model execution as the main cost.** FP32 cached
      configurations spent approximately 92–94% of instrumented request time
      in prefix and suffix model execution, versus 4.6–6.6% in cache cloning
      and expansion. Sixteen-candidate cached requests still used seven or
      eight model calls because suffixes were grouped by exact length.
- [x] **Direct FP32 loading was reproduced in process isolation.** The fresh
      worker loaded FP32 directly rather than converting a resident BF16 model
      or offloading to CPU. Post-load PyTorch allocation was 15.67 GiB in
      FP32 versus 7.83 GiB in BF16, with 6.13 GiB versus 13.97 GiB of driver-
      reported free memory. These are snapshots, not a memory-soak result.

The panel contains eight distinct messages expanded into 128 factorial
episodes and reuses archived examples for execution regression. It supports
the cache implementation in the tested FP32 configuration, not exact
equivalence for every input or task-quality superiority over BF16. Preserve
the pinned checkpoint, tokenizer and exact token sequences, frozen heads,
policies, FP32 arithmetic configuration, and full-sequential outputs as
reference artifacts. Keep BF16 outputs as a separate execution reference.

The validated execution structure is:

```text
Finalize exact candidate token sequences
  -> find the common token prefix
  -> prefill once
  -> create isolated hybrid cache branches
  -> group and evaluate candidate suffixes
  -> restore original candidate order
  -> frozen scorer -> none head -> probabilities -> application policy
```

Complete hybrid state isolation includes recurrent and convolution state, not
only attention keys and values. Shared-prefix reuse across different
questions, padded or packed suffix schemes, Rust, Metal, HTTP, concurrent
requests, and long-document decision quality remain unvalidated.

### Phase 2F: cache compression, prefix reuse & persistent LRU caching (MEASURED / DONE)

Run ID: `20260918T224427722898Z` (version `2f.1.0`, archived in `research/opendecision_phase2f_20260918T224427722898Z/`).
Evaluated on **NVIDIA L4 GPU** comparing `fp32_strict_math` and `bf16_default` workers in process isolation. Backbone weights, candidate scorer, set-linear `none` head, and 9 application policies remained frozen; no retraining or threshold fitting occurred.

The evaluation covered 128 episodes from 8 archived messages (32 episodes for compression regression, 16 for complete cold requests across $K \in \{2, 4, 8, 16\}$, two 32-request traces for cache locality, and synthetic prefixes of 64, 256, and 1,024 tokens).

**1. Snapshot Codec Equivalence vs. Gate Failure:**

| FP32 Snapshot Codec | Max Delta vs. Full | Outcome Changes | Policy Changes | Accepted / 32 | Status |
|---|---:|---:|---:|---:|---|
| **Lossless** | **0.00000880** | **0** | **0** | **32 / 32** | Passed all gates |
| **FP16 Attention KV** (`kv_fp16`) | **0.00042450** | **0** | **0** | **32 / 32** | Passed all gates |
| TurboQuant $k=3, v=4, r=0$ | 0.21882282 | 8 | 2 | 5 / 32 | Failed |
| TurboQuant $k=3, v=4, r=32$ | 0.18655649 | 8 | 5 | 6 / 32 | Failed |
| TurboQuant $k=4, v=4, r=32$ | 0.12008530 | 2 | 1 | 12 / 32 | Failed |
| TurboQuant $k=3, v=2, r=32$ | 0.47489768 | 13 | 6 | 5 / 32 | Failed |

- **Lossless & FP16 KV Storage Passed**: In FP32, lossless snapshots reproduced identical outputs under the same chunking; FP16 attention-KV storage had maximum codec-only delta of 0.00042351 with 0 outcome changes and 0 policy changes.
- **TurboQuant Low-Bit Codecs Failed**: All four tested TurboQuant snapshot configurations failed the 0.005 probability gate and altered decisions (changing argmax in 2–13 episodes and policies in 1–6 episodes). Retaining a 32-token exact tail ($r=32$) or increasing nominal key bits did not restore parity.
- **BF16 Round-Trip Divergence**: Under BF16, even lossless snapshot restoration failed comparison against BF16 full sequential (max delta 0.074689, 13/32 accepted, 1 policy change), confirming that snapshot fidelity cannot fix underlying execution-shape divergence.

**2. Memory Accounting & Recurrent State Dominance:**
- **Hybrid Cache Composition**: For a representative 42-token prefix, the FP32 root contains **48.0 MiB recurrent state** (24 DeltaNet blocks $\times 32 \times 128 \times 128$ floats), **3.0 MiB convolution state**, and only **2.625 MiB attention KV** (53.625 MiB total).
- **Short-Prefix Compression Overhead**: Attention KV accounts for only 4.9% of a 42-token root. Compressing KV saves negligible bytes while TurboQuant codebook tables add ~4 MiB overhead, resulting in snapshot entries that are *larger* than uncompressed roots.
- **Long-Prefix Amortization**: At 1,024 tokens, attention KV grows to 64 MiB (FP32). FP16-KV storage reduces root storage from 115 MiB to 83 MiB (**27.8% root reduction**), passing all numerical and policy checks.

**3. Cold Request Latency & Suffix Batching:**

| Candidates ($K$) | Full Sequential | Full Batch 4 | Shared Batch 4 | Shared Batch 8 | Lossless Snapshot | FP16-KV Snapshot |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 195.4 ms | **147.2 ms** | 271.5 ms | 271.7 ms | 274.6 ms | 287.3 ms |
| 4 | 392.1 ms | **281.4 ms** | 373.7 ms | 373.3 ms | 376.7 ms | 389.2 ms |
| 8 | 793.2 ms | 582.5 ms | **576.6 ms** | 576.9 ms | 578.8 ms | 590.8 ms |
| 16 | 1,595.3 ms | 1,180.4 ms | 724.5 ms | **686.4 ms** | 727.3 ms | 738.8 ms |

- Full batching is faster on short requests ($K=2, 4$). Shared-prefix suffix batching becomes faster at $K=8$ and decisive at $K=16$ (**$2.32\times$ faster** than full sequential, **$1.72\times$ faster** than full batch 4).

**4. Persistent LRU Cache Traces:**
- On 32-request traces across 8 messages with a 384 MiB budget, GPU lossless LRU reduced total service time by **13.91%** (grouped locality) and **11.88%** (shuffled locality) with 0 policy changes.
- **Cold vs. Warm Mechanics**: At 1,024 synthetic tokens, cold shared-prefix execution took 1,295.9 ms (vs. 9,217.4 ms full sequential), while warm lossless reuse took **272.9 ms** (a **$33.8\times$ latency reduction** over full sequential after cache population).
- **Multi-Token Prediction (MTP)** is not applicable: no output tokens are generated, and all candidate suffix tokens are supplied upfront.

---

### Phase 2G: fresh decisions, TF32 arithmetic & cache lifecycle (MEASURED / DONE)

Run ID: `20260919T005142584348Z` (version `2g.1.0`, archived in `research/opendecision_phase2g_20260919T005142584348Z/`).
Evaluated on **NVIDIA L4 GPU** comparing `fp32_strict_math` and `fp32_tf32_allowed` workers in process isolation. Backbone, heads, and policies remained frozen.

The evaluation introduced 416 fresh sampled-choice episodes across 112 messages (excluding 2,912 prior Banking message hashes), spanning Banking77 head-training labels, Banking77 held-out labels, CLINC non-financial domains, and CLINC author-labeled out-of-scope (OOS) requests.

**1. Strict-FP32 Parity Survives Fresh Inputs & Controlled Context:**

| Panel | Strategy | Episodes / Messages | Max Prob Delta | Outcome Changes | Policy Changes | Accepted |
|---|---|---:|---:|---:|---:|---:|
| **Fresh Inputs** | Full batch 4 | 56 / 16 | 0.00000731 | 0 | 0 | 56 / 56 |
| **Fresh Inputs** | Shared lossless | 56 / 16 | 0.00000774 | 0 | 0 | 56 / 56 |
| **Fresh Inputs** | Shared FP16-KV | 56 / 16 | 0.00045509 | 0 | 0 | 56 / 56 |
| **Controlled Context** | Full batch 4 | 72 / 6 | 0.00003582 | 0 | 0 | 72 / 72 |
| **Controlled Context** | Shared lossless | 72 / 6 | 0.00001442 | 0 | 0 | 72 / 72 |
| **Controlled Context** | Shared FP16-KV | 72 / 6 | 0.00054408 | 0 | 0 | 72 / 72 |

- Strict-FP32 execution fidelity reproduced its full-prompt reference across all fresh and context-expanded tests. FP16-KV storage remained bounded within 0.00055 max delta with 0 policy drift.

**2. TF32 Speedup vs. Equivalence Gate Failures:**
- **Performance Win**: TF32 permission (`float32_matmul_precision = high`, `matmul.allow_tf32 = true`) roughly doubled full batching speed on short requests:
  - $K=4$: 265.1 ms $\rightarrow$ **131.7 ms** ($2.01\times$).
  - $K=16$: 1,118.3 ms $\rightarrow$ **518.5 ms** ($2.16\times$).
- **Memory Invariance**: Post-load parameter allocation was identical (16,043.7 MiB). TF32 does not save weight storage.
- **Equivalence Failures**: Across 416 fresh full-sequential episodes, TF32 changed head argmax in **3 episodes** (2 moving from correct `none` to wrong candidate in refitted global; set-linear head had 0 argmax changes). In controlled context (72 episodes), 2 episodes exceeded the 0.005 tolerance.
- **Conclusion**: TF32 is a separately versioned performance configuration, not an exact equivalent drop-in replacement.

**3. Fresh Semantic Quality & Rejection Transfer Limits:**

| Family | Frozen None Head | Accuracy | NLL | Answerable Acc | None Recall |
|---|---|---:|---:|---:|---:|
| Banking, training labels | Set-linear | 69.53% | 0.8821 | 70.31% | 68.75% |
| Banking, held-out labels | Set-linear | 75.78% | 0.6471 | 82.81% | 68.75% |
| CLINC, non-financial | Set-linear | 66.41% | 1.0366 | **93.75%** | **39.06%** |
| CLINC, author OOS | Set-linear | 46.88% | 1.3385 | N/A | **46.88%** |

- **Rejection Transfer Bottleneck**: In CLINC, answerable accuracy was 93.75% (60/64), but omitted-intent recall dropped to **39.06%** (25/64), and author-labeled OOS recall was only **46.88%** (15/32). Matching offered candidates transfers well; detecting absent or out-of-domain intents remains weak.

**4. Criteria Ambiguity & High-Confidence Error Case:**
- Message: `"How can I get a physical card"` (gold label: `order_physical_card`, "order physical card").
- Model behavior: When presented with `get_physical_card` ("get physical card"), the model selected it with **>98.8% to >99.8% confidence** across all 4 episodes.
- Implication: Terse, overlapping descriptions cause confident failure. Candidate criteria must be documented, versioned semantic descriptions rather than raw class labels.

**5. Context Length & Evidence Position Degradation:**
- Wrapping 6 requests in administrative background showed significant semantic sensitivity:
  - Minimal wrapper: **91.67% accuracy**, 0.28–0.34 NLL.
  - 1,024 state tokens, request first: **66.67% accuracy**, 0.5932 NLL.
  - 1,024 state tokens, request last: **83.33% accuracy**, 0.3342 NLL.
- Execution fidelity and semantic validity diverge: cached paths faithfully reproduce the full-sequential decision, but the decision itself degrades under distracting background context.

**6. Cache Lifecycle with Real TTL Expiry:**
- 192 MiB budget with 20s TTL on a 48-request trace achieved **6.19% service time reduction** in strict FP32, recording 15 hits, 33 misses, 21 capacity evictions, and 9 TTL expiry events.
- 7 CPU tests verified exact-boundary expiry, byte-bounded eviction, oversized-entry bypass, tenant namespace separation, and branch isolation without root mutation.

---

### Phase 2H: decision criteria, rejection transfer & prompt architecture (NEXT)

Shift immediate model research from repetitive cache testing to decision criteria, rejection transfer, and prompt structure under the frozen strict-FP32 reference:

- [ ] **P0: Criteria and annotation review**: Audit confusing candidate pairs (e.g. `order_physical_card` vs. `get_physical_card`). Decouple stable machine IDs from rich semantic descriptions. Establish versioned description contracts.
- [ ] **P0: Rejection and calibration transfer**: Compare fixed scorer with refitted set-aware `none` models and multi-domain candidate heads across both omitted in-scope intents and author-OOS distributions.
- [ ] **P0: Irrelevant-context and instruction robustness**: Train/evaluate under realistic background text and variable evidence positions to prevent attention dilution.
- [ ] **P1: TF32 serving path validation**: Measure TF32 under explicit application quality and behavioral contracts to determine if the $2\times$ batching speedup can be safely captured.
- [ ] **P1: State-first prompt contract**: Explore `[State, Instruction, Candidates]` prompt layout to evaluate whether prefill can be amortized across *different* questions, rather than only across candidates of the same question.
- [ ] **P1: Matched baselines**: Benchmark against finite-token classification baselines (SALSA, GLiClass).
- [ ] **P1: Dedicated Noul & Score evaluation**: Train and validate binary truth/yes (`Noul`) calibration and ordinal rubric levels (`Score`).
- [ ] **P1: Model scaling and adaptation**: Compare frozen 4B backbone against LoRA adaptation and smaller backbones (e.g. Qwen2.5-1.5B/3B).

---

## Phase 3 — Rust Engine & Production Backends (PLANNED / GATED ON 2H)

Once the Phase 2H criteria, rejection transfer, and prompt contracts are pinned, implement the production Rust engine following the **five-stage Parity Ladder** from Section 11.3 of the research whitepaper:

### The Rust Parity Ladder

1. **Deterministic Head Algebra**:
   - Folded linear projection: $W' = W / \sigma$ and $b' = b - W(\mu / \sigma)$.
   - Numerically stable softmax with temperature scaling.
   - Seven-parameter set-linear `none` feature extraction (max score, top-two gap, mean, std, log-mean-exp, $\log K$).
   - Verify against exported fixtures (`golden_head_inputs.npz`, max logit error $\le 1.8 \times 10^{-6}$).
2. **Tokenization & Mask Conventions**:
   - Segmented encoding (`instruction`, `state`, delimiters, `candidate`) with `add_special_tokens=False`.
   - Longest common token-prefix calculation.
   - Attention masks, position offset tracking, and explicit truncation handling.
3. **Full-Backbone Hidden Vector Parity**:
   - Qwen3.5 hybrid architecture: 24 recurrent DeltaNet blocks + 8 full-attention blocks.
   - Hidden state verification at the last non-padding token against Python FP32 reference.
4. **Hybrid Cache Branching & Snapshot Isolation**:
   - Immutable root snapshot: recurrent state ($32 \times 128 \times 128 \times 4$ bytes $= 48$ MiB), convolution state (3 MiB), and attention KV.
   - Independent branch isolation for candidate suffixes (prevent mutable cross-branch contamination).
   - FP16 attention-KV snapshot storage for long prefixes ($L \ge 256$).
5. **Suffix Batching & Cache Lifecycle**:
   - Group candidate suffixes by exact token length for unpadded execution.
   - Byte-bounded persistent LRU cache with tenant namespace isolation and TTL expiry.
6. **Daemon Integration & Production Backends**:
   - `opendecision-runtime`: Device discovery, worker pools, VRAM budgeting.
   - `opendecision-backends`: Candle, GGUF/llama.cpp, and ONNX backends behind `DecisionEngine`.
   - Axum HTTP & Tonic gRPC request scheduler with deadline cancellation.

---

## Status snapshot

| Phase | Description | Status | Tests / Milestone |
|---|---|---|---|
| **Phase 0** | Wire contract & core types | done | 33 tests |
| **Phase 1** | Daemon, HTTP/gRPC transports & SDK compat | done | 91 tests |
| **Phase 2A** | Python Qwen3.5-4B exploration & parameter audit | done | Colab probe |
| **Phase 2B** | Fixed NLI head benchmark & baseline readout | done | Run 20260917T205849Z (87.67% acc) |
| **Phase 2C** | Qwen model research: dynamic schemas, batching & scaling | measured, needs review | Run 20260917T222948Z |
| **Phase 2D** | Numerical reference, rejection policy & complete requests | measured, needs review | Run 20260917T234417Z |
| **Phase 2E** | Selective precision, shared-prefix parity & rejection policy | measured, done | Drive run 20260918T114914072764Z |
| **Phase 2F** | Cache compression, prefix reuse & persistent LRU caching | measured, done | Run 20260918T224427722898Z |
| **Phase 2G** | Fresh decisions, TF32 arithmetic & cache lifecycle | measured, done | Run 20260919T005142584348Z |
| **Phase 2H** | Decision criteria, rejection transfer & prompt architecture | next | Gated on Phase 2G evidence |
| **Phase 3** | Rust engine, production backends & Parity Ladder | planned | Gated on Phase 2H |

`cargo test --workspace`: **173 tests passing, 0 failing.**

---

## Conventions for agents

(Same as `docs/AGENTS.md` — duplicated here so this file is
self-contained for a roadmap reader.)

- **Wire types live in `opendecision-core`.** Any change there is a
  breaking change for every future SDK caller. Bump
  `JevRequest::SCHEMA_VERSION`, add a round-trip test, regenerate
  schemas, update `docs/ARCHITECTURE.md`.
- **`sdk_compat.rs` is the contract.** If your change would break
  one of those 56 tests, you are touching the wire contract.
- **New backend = new `DecisionEngine`.** Wire it in
  `opendecision-server/src/main.rs` behind a CLI flag. Add at least
  one round-trip test in `opendecision-engine`. Add a `sdk_compat.rs`
  case that exercises the alias end-to-end via `/v1/systemone`.
- **Phase 2 owns model research in Python; Phase 2H owns criteria & rejection transfer; Phase 3 owns production Rust engine.** Do not start production runtime/backends work until the Phase 3 Parity Ladder reproduces the declared FP32, rejection, and cache-isolation contracts.

## Quick reference

```bash
# Build & test
cargo build --workspace
cargo test --workspace          # 82 tests

# Regenerate JSON Schema
cargo run -p opendecision-gen-schemas

# Run the daemon (mock engine, dev mode, no auth)
cargo run -p opendecisiond -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest

# Verify a request against a running daemon
curl -sS -H 'content-type: application/json' \
     -X POST http://127.0.0.1:18080/v1/systemone \
     -d @examples/04_mixed.json | jq
```
