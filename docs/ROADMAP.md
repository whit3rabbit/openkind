# opendecision — Roadmap

> Qwen-led, open-source typed decision inference, with a Rust service and an explicitly pinned Jev-compatible interface target.
> **Revision 0.7.2 · 20 September 2026 · Rust Phase 3.3 CPU parity, Phase 3.4 branch-state, and Phase 3.5 sequential nested gates passed; batched question execution is next.**
> Contract reference: <https://docs.typesafe.ai/api>. A live documentation URL does not replace pinned schemas and conformance fixtures.

**Current evidence:** B–G remain historical research. **Phase 2H is completed through the `2h.1.2` continuation** and the exploratory 2I/2J screen selected/exported a provisional Qwen3.5-4B state-first integration profile. **Phase 3A is completed in Python** for run `20260920T024056Z`: the selected profile remained unchanged, semantic batched parity passed, high-K systems parity passed, and the recorded same-process repeatability delta was zero. **Phase 3B is completed in Python** for run `20260920T152206Z`: it exports exact tokens, 34 ordered diagnostic stages, 10 candidate vectors, and 3 continuation vectors without changing the model or bundle. Rust Phase 3.1 head/probability parity, Phase 3.2 exact tokenizer/rendering parity, Phase 3.3 CPU backbone parity, Phase 3.4 backend-neutral branch-state contract parity, and Phase 3.5 sequential nested execution parity now pass. The native path executes embedding, 32 decoder layers, final RMSNorm, all 10 candidate sequences, Qwen-specific cached continuation, profile-bound branchable state with fork/fan-out/gather, and one-prefill sequential nested `state → question → candidate` execution while keeping the model and Colab frozen. **Current direction:** batch the question layer, then batch candidate execution, and build the adaptive scheduler on the named Mac. CPU native parity does not imply Metal or accelerated parity. Reviewed model promotion, natural-data confirmation, efficient external baselines and Track S service work continue in parallel. [H; HF; IJ2; P3A; P3B; WP §§13.3–13.5, 15–17]

**Latest systems workbench:** `OpenDecision_Phase3A_BranchableState_BatchedQ`, run `20260920T024056Z`, completed notebook scope against selected profile `a047d6802c3f06f085b8` without model changes, training or reselection. It validates the Python branch/batch reference and sharpens the scheduler requirement; it does not establish Rust/Metal parity or release quality. The earlier `2ij.2.0` model-selection screen remains the model-selection authority, while the blocked `2ij.1.0` review-gated checkpoint remains preserved as evidence for the still-open independent-review path. [IJ; IJ2; P3A; WP §§14–16]

**Latest backbone workbench:** `OpenDecision_Phase3B_Qwen35_Backbone_Parity`, run `20260920T152206Z`, completed its Python/Qwen reference scope without training, model selection, model modification, or bundle change. The export contains 47 FP32 vectors: 34 trace stages, 10 full candidate features, and 3 continuation vectors. Fresh full-sequence features differ from the earlier bundle features by at most `4.9591e-05`. Cached continuation differs from fresh full sequence by at most `1.9073e-05`. These are localization diagnostics under the notebook's `1e-4` self-consistency guard, not a new Rust acceptance tolerance. [P3B]

This file is the task/status authority; the companion [whitepaper](whitepaper/OpenDecision_Whitepaper_v0.7.2.md) is the evidence and interpretation authority. Neither updates the underlying repository or notebook by itself. Checkbox completion below is source-reported historical work or an explicitly stated saved-artifact result; open work remains unchecked. Execution completion, statistical quality, numerical equivalence and deployment readiness are separate gates.

**Review capture (v0.5.2):** the historical `OpenDecision_Review_Followup_Traceability.md` review-to-work/test matrix maps the recovered architecture discussion and Laya/R4T follow-ups to the tasks below. ModernBERT-style joint-candidate scoring and a separate released-Laya baseline are explicit early comparisons; conditional **P2.1–P2.3** checkboxes are restored from the earlier roadmap. This is source-based documentation, not a claim of a complete shared-chat transcript or completed future experiments. [RC; WP §11.7]

**Execution snapshot boundary:** H now has a completed `2h.1.2` continuation with final outputs, as recorded in HF. The original `2h.1.1` failed attempt remains unchanged under H. Recovery-notebook availability and completed evaluation are distinct events; both are now documented. [H; HF; HR]

**Reading order:** [2I/2J checkpoint and blockers](#phase-2ij-workbench-checkpoint-preparation-recorded--review-gated) → [Phase 2H status](#phase-2h--criteria-and-rejection-transfer-completed--required-scope-closed) → [current priorities](#strategic-assessment--project-rebalancing) → [status snapshot](#status-snapshot). The source register records scope and verification limits. Historical “next” language in 2A–2G is not the current queue.

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

### Reported test snapshot — current HEAD not verified

| Suite                                              | Tests |
|----------------------------------------------------|-------|
| `opendecision-core` unit + conformance             | 43    |
| `opendecision-engine` unit (MockEngine, dispatch)  | 20    |
| `opendecision-api` unit, integration & sdk_compat  | 108   |
| `opendecision-cli` unit                            | 10    |
| `opendecision-server` unit                         | 3     |
| `opendecision-proto` unit                          | 2     |
| `opendecision-runtime` unit                        | 6     |
| `opendecision-gen-schemas` schema sync             | 3     |
| **Reported total**                                  | **195** |

The supplied roadmap reports this **195-test snapshot**, but does not identify an exact commit. No Rust source, build or test was inspected or executed for this revision. Treat the counts as supplied historical implementation evidence—not a fresh `cargo test` result or proof of semantic model quality. Track S.1 requires a commit-stamped replacement. (SDK-compat and conformance fixtures describe the pinned wire surface; unit tests cover internal subsystems.) [R0]

---

## Phase 2 — real model backends 🚧 STARTED

**Goal:** go from mock responses to useful, versioned decision inference while preserving the declared wire contract. The measured OpenDecision hypothesis is that Qwen's 2,560-dimensional features can support small decision heads without an answer-generation loop. This does not reconstruct Jev's proprietary neural architecture, establish one pass per complete dynamic request, or remove work that scales with input length, Q or K. [WP §§2, 4–5]

Phases 2A–2G below retain historical measurements. Their experiment-local “next” recommendations are historical; the current status, closeout tasks and refocused program later in this document control new work.

### Phase 2A — Python exploration (DONE)

A PyTorch notebook on Colab established exploratory hidden-state/head plumbing. It did not validate a proprietary architecture or general decision quality; the initial head used two fitted examples. [WP §4.1]

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

**Corrected interpretation:** this is an exploratory comparison with a long generated answer, not an equal-quality benchmark or proof of constant-time decisions. Avoiding output-token decoding removes that loop; prefill, candidate evaluation and independent-question work still depend on the actual input and execution graph. The multimodal `AutoModel` and text-only causal-LM parameter rows also count different module sets, not removable output-head weights. [WP §§4.1–4.2, 8.3]

**Key finding 2 — Qwen3.5 layer mix**

The model alternates `Qwen3_5GatedDeltaNet` (recurrent-style state)
and `Qwen3_5Attention` (standard Q/K/V) layers. This is *relevant*
to OpenDecision because it suggests the backbone already produces a
reusable state representation — which has direct implications for
the candidate-prefix cache studies in Phases 2E–2G and proposed question-level sharing in Phase 2I. It does not by itself establish an instruction-independent state representation.

**Key finding 3 — embeddings are tied**

`embed_tokens` lives inside the backbone and shares weight storage with the vocabulary projection. Bypassing the LM output projection and generation loop saves computation, but retaining the input embedding leaves **zero marginal removable tied weights** in the measured text model. The later text-only audit, not the wrapper comparison, is the memory authority. [WP §4.2]

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
- Interpretation: the reported latency ratios depend on the output tokens avoided, not an equal-quality end-to-end workload. The +8.83 pp matched accuracy comparison is a trained head versus an **untuned** finite-code baseline, not an isolated effect of removing generation. A trained finite-token control remains necessary. [WP §§4.4, 8.3]
- Independent batching throughput: 12.38 qps ($B=1$), 25.79 qps ($B=4$), 24.24 qps ($B=8$), with varying sequence lengths (94, 104, 128 tokens).

**4. Calibration & Batching Findings:**
- Temperature scaling ($T=1.1441$ on 300 calibration examples) did not improve test NLL (0.3345 → 0.3392) or Brier score (0.1834 → 0.1854). API must preserve raw probabilities and label scaling `temperature_scaled`.
- Batching invariance: ~1.27% probability difference observed between standalone and padded batch execution. Requires 3-tier diagnostic (same-length duplicate, variable padding, heterogeneous batch) before declaring production invariance.

**5. Exported Reference Artifacts (`research/opendecision_phase2b_20260917T205849Z/frozen_export/`):**
- `manifest.json`: Tokenization and prompt format (`nli-described-abc-v1`).
- `head.safetensors`: Weights for the 7,683-parameter head.
- `golden_head_inputs.npz`: Reference input activations and expected logits for Rust unit testing.

### Phase 2C — Qwen model research: dynamic schemas, batching invariance & model scaling (MEASURED / NEEDS REVIEW)

> **Historical scope, updated gate:** Phase 2 researches models in Python/Colab. A complete native backend needs a chosen supported profile and a parity ladder, not a claim of arbitrary-schema competence. The thin resident-worker service in Track S can proceed earlier with explicit task limits; do not block all real integration on the full modeling program. [WP §§11.3, 11.8, 13.3]

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

Run `20260918T114914072764Z` completed on an NVIDIA L4 with fresh FP32 and BF16 workers. The [saved expanded-run summary](https://drive.google.com/file/d/1SxOG4VY4TlqK0e4eqBgZ_KwfDYjbXeFX/view) and [whitepaper discussion](whitepaper/OpenDecision_Whitepaper_v0.7.2.md#8-shared-prefix-execution-what-works-and-when) provide the source result and detailed interpretation. The historical review reconstructed 3,072 probability distributions, policy actions, parity counts and timing aggregates with agreement to the report; it did not rerun Qwen. That earlier saved-calculation audit is retained as historical evidence, not repeated by this v0.6 documentation update. [WP §§7–8; E5]

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

**4. Criteria Ambiguity & High-Probability Error Case:**
- Message: `"How can I get a physical card"` (gold label: `order_physical_card`, "order physical card").
- Model behavior: When presented with `get_physical_card` ("get physical card"), the model selected it with **candidate probability >0.988 to >0.998** across all 4 episodes (distinguishing candidate probability from the separate confidence statistic).
- Implication: Terse, overlapping descriptions cause confident failure. Candidate criteria must be documented, versioned semantic descriptions rather than raw class labels. That is a warning about the combination of ambiguous criteria, ranking, and policy transfer—not four independent demonstrations of failure, and not grounds to silently alter gold labels.

**5. Context Length & Evidence Position Degradation:**
- Wrapping 6 requests in administrative background showed observed semantic sensitivity (not a reported significance test):
  - Minimal wrapper: **91.67% accuracy**, 0.28–0.34 NLL.
  - 1,024 state tokens, request first: **66.67% accuracy**, 0.5932 NLL.
  - 1,024 state tokens, request last: **83.33% accuracy**, 0.3342 NLL.
- Execution fidelity and semantic validity diverge: cached paths faithfully reproduce the full-sequential decision, but the decision itself degrades under distracting background context.

**6. Cache Lifecycle with Real TTL Expiry:**
- 192 MiB budget with 20s TTL on a 48-request trace achieved **6.19% service time reduction** in strict FP32, recording 15 hits, 33 misses, 21 capacity evictions, and 9 TTL expiry events.
- 7 CPU tests verified exact-boundary expiry, byte-bounded eviction, oversized-entry bypass, tenant namespace separation, and branch isolation without root mutation.

---

## Phase 2H — Criteria and Rejection Transfer (COMPLETED / REQUIRED SCOPE CLOSED)

**Result authority:** continuation `20260919T040612625670Z__finish_2h_1_2`, version `2h.1.2`, saved around 14:45 UTC on 19 September 2026. **Both `eval_qwen4b_strict` and `eval_qwen4b_tf32` completed.** The original `20260919T040612625670Z` failed attempt is preserved unchanged; its fitted artifacts, selection, criteria and original reserved final data were validated and reused without retraining. Execution completion, numerical acceptance and model/deployment quality remain separate decisions. [H; HF]

**Detailed evidence belongs in the [whitepaper §13.1](whitepaper/OpenDecision_Whitepaper_v0.7.2.md#131-phase-2h-completed-continuation-and-retained-development-history).** It retains the development tables, full final comparisons, conditional rejection, calibration/policies, TF32 and cache gates, robustness, primitive scores, complete-request timing, memory limitations and the audit scope. Original H development details are no longer duplicated here.

[Completed summary](https://drive.google.com/file/d/1rhO2B5ro3rQxEGJ3ZSt6JhflluV7-NPz/view) · [Compact summary](https://drive.google.com/file/d/1072Mee3vu6JFjMXADPBe-GWm_BhCbGGE/view) · [Continuation archive](https://drive.google.com/file/d/1kSnKjnOinF6fvyc-ZO9UF3rbfNT5eMCR/view) · [Recovery validation](https://drive.google.com/file/d/139rnY0E9EK6lIXrXDE0PhmSESZrPNiIu/view) · [Final lock](https://drive.google.com/file/d/15nnlgaO__RDAzVQOKh96LzF9XvIBWkeG/view).

### Closeout summary and implications

| Finding | Roadmap consequence |
|---|---|
| All 14 saved profiles and both required arithmetic workers have final results over 320 messages / 1,152 episodes; controlled robustness, bounded primitives and request benchmarks completed | Close original-study recovery/evaluation, not the unrelated service or new-model tasks |
| Preselected support/seed43 reaches **78.91% raw pooled accuracy**, but held-out Banking/CLINC accuracy is **63.67% / 66.02%**; a retained original-criteria joint control has stronger pooled transfer | Carry the original-criteria control and support ablations into fresh 2I/2J work; do not choose a new default retrospectively on H final results |
| All sampled author-OOS episodes are correctly rejected, while omitted-intent, context and frozen-policy errors remain | Keep independent criteria review, missing-evidence labels and richer applicability/rejection open |
| Same-mode selected-profile batching/lossless/FP16-KV subset gates pass; full-panel TF32 still has numeric failures and controlled contexts show additional drift | Close the measurement; do not promote TF32 as an interchangeable implementation |
| BoolQ/SST-5 bounded final probes are available; natural documents, state-first, smaller Qwen, LoRA and real service/native execution are absent or disabled | Use H as a bounded baseline; keep broader 2I/2J/Track S/P2 and Phase 3 scope open |

Source and full interpretation: HF / WP §§13.1.8–13.1.16. The pooled headline is a raw episode average; the separately weighted NLL and policy scenarios are explained in the whitepaper. Source-message counts, repeated variants and independent task evidence are not interchangeable.

### Closed original-study tasks

- [x] **2H-C1 — Preserve and verify recovery inputs.** Completed: original ZIP/failure record retained; saved fit files, criteria, split/final-input identities and selection verified against the original attempt. [HF; WP §13.1.8]
- [x] **2H-C2 — Repair worker finalization and validate recovery.** Completed through the versioned `2h.1.2` repair/recovery path, with separately recorded artifact validation and successful final-worker completion; no manual rewriting of the original failed worker. [HF; HR]
- [x] **2H-C3 — Lock artifacts and complete original final evaluation.** Completed: locked original reserved final inputs, unchanged fitted artifacts/policies, both required final workers and retained prediction outputs. [HF]
- [x] **2H-C4 — Publish original readout and unresolved scope.** Completed: all retained final comparisons, rejection/policy/probability metrics, bounded primitives, controlled robustness, arithmetic and request/resource records saved; disabled or absent arms identified. [HF; WP §§13.1.9–13.1.15]
- [x] **2H-C5 — Close the evidence handoff.** Completed by the version 0.6 source-linked whitepaper and roadmap with saved-output/lineage checks. Final outcomes become historical/regression evidence after they inform the next study. [HF; V06]

**Closed means the required experiment and handoff are complete—not that every hypothesis passed.** Preserve all failures, pre-final model selection and historical labels. No new low-bit cache, LoRA, state-first, compact-model, independent-review or HTTP result is inferred. Continue with the reviewed-data/selection protocol below; do not rerun H merely to get a more favorable final result.

---

## Strategic Assessment & Project Rebalancing

### Destination and evidence boundary

**Keep the existing 4B strict-FP32 system as a reference, not a mandatory shipping model.** The destination is a compact, useful engine answering several independent, well-scoped questions over one state, with measured uncertainty, automation coverage, latency and memory. The project is Qwen-led but model-comparative. [WP §§1, 11–13]

Completed B–G results establish the original reference and its limits. H now adds **completed paired final readout-learning evidence**, including a support-conditioned development/final transfer reversal, persistent omission/context/policy limits and incomplete TF32 equivalence. It does not establish backbone adaptation, general question following, a smaller-model winner or a real service. Keep these evidence levels distinct. [HF; WP §§13.1.8–13.1.16]

Rust Phase 3.1 through 3.5 parity now passes against the frozen selected profile and Phase 3B fixtures: head/probability, exact tokens, CPU full-sequence backbone, cached continuation, the backend-neutral `BranchableState` contract, and sequential nested execution. The near-term sequence is batched Q/K parity and then target-Mac scheduler measurements. CPU native parity does not imply Metal or accelerated parity. Independent review/natural-data confirmation, 2J.5–2J.6 release confirmation and Track S service work continue in parallel; they are promotion/integration gates, not reasons to change the frozen implementation target. Do not tune new model choices on exposed H/E11 final data. [HF; IJ2; P3A; P3B; RUST3; RUST4; RUST5; WP §§13.3, 16–17]

### External benchmark implications — execution shape, not architecture proof

Two September 2026 external sources refine the systems target without changing the selected integration profile.

**Parallel Constrained Decoding (PCD) community Qwen implementation.** The public `harshatheg/Qwen-2.5-1B-RLCD` project is most useful as an inference-pattern reference: one shared prefill, cache broadcasting across fields, constrained candidate-token scoring and programmatic JSON assembly. Its model card reports M4 Max examples around 68–75 ms for four fields, 270 ms for 28 fields and 89 ms for one 255-choice field, with 5.6–7.0× latency reductions versus its autoregressive JSON baseline. Treat these as author-reported PCD measurements, not OpenDecision results and not evidence that TypeSafe's RLCD training procedure has been reproduced. The repository's useful lesson is **batched branch execution**; a softmax over sliced logits is a normalized distribution, not proof of calibration. [EXT1; WP §11.7.1]

**DGX Spark Jev-style comparison.** The More Than a Machine study reports whole-request Q-scaling for several readers. Within its own campaign, Jev 1.13 moved from 105.1 ms p50 at one question to 109.2 ms at four questions (~1.04×); Laya moved from 16.4 to 29.1 ms (~1.77×); the tuned Qwen3.5 wrapper moved from 167.0 to 665.1 ms (~3.98×). Absolute values are not directly comparable because Jev includes a hosted HTTP round trip while local readers are warm/in-process. The relevant systems target is the **slope**: additional questions should add substantially less than a full repeated model evaluation once the state is shared. The same study also reports task-order reversals and separate Brier/ECE results, reinforcing that speed, accuracy and calibration must remain separate axes. [EXT2; WP §11.7.1]

**Architectural consequence:** do not replace state-first OpenDecision with the public schema-first PCD prompt. In the public PCD design, semantic schema descriptions are part of the shared prefix, so changing the question set can change the state representation used by every field. OpenDecision's state-first root remains the stronger isolation contract. Borrow the **breadth-first batching strategy**, not the schema-conditioned state representation.

**New primary systems question:** after backend-neutral branching and nested parity, measure whether `T(Q)` grows sublinearly enough to make multi-question use materially cheaper than repeating Q independent requests. Report `T(Q)/T(1)`, marginal milliseconds per added question, questions/s, forward-call count, state-prefill share and peak branch-state bytes for sequential-nested and batched-nested execution. No external timing is a Mac or OpenDecision target by itself.

### Two acceptance tracks

| Track | Required question | Gate | Not a substitute |
|---|---|---|---|
| **Implementation equivalence** | Does a cache, batching strategy or backend preserve one specified model/execution profile? | Same finalized tokens, weights, heads, criteria, normalization, calibration and fixed policies; declared probability, argmax and directed-action tolerances | More speed, unchanged pooled accuracy, or a new model name |
| **New-model quality** | Does a separately trained/model-sized/rendered profile improve useful decisions? | Fresh labeled state/question/rubric families; predeclared accepted-risk, coverage and resource requirements; its own versioned execution reference | Agreement with every old answer, improvements only on inspected regression cases, or rejecting everything |

The historical E/F/G tolerance of **0.005** and their no-outcome/no-policy-change requirements are not relaxed retrospectively. A new model can legitimately correct old errors, but must subsequently have its own tested implementation contract. Failed BF16/TF32/codec equivalence results are not universal claims about model quality and are not automatically accepted under a new label. [WP §§7, 9–10, 13.2]

### Priorities and dependencies

| Priority | Work package | Start condition | Evidence needed before the next decision |
|---|---|---|---|
| **Closed** | **2H-C1–C5:** original-study recovery and closeout | Completed continuation and evidence handoff | Preserved E8 failure plus HF final outputs; model/numerical promotion remains separate |
| **P0 — now** | **Phase 3.6:** batched question execution; 3.1–3.5 complete | Rust Phase 3.5 sequential nested pass + completed Phase 3A Python reference | Batched question parity against the sequential baseline, question isolation, and exact profile/state identity [P3A; RUST5] |
| **P0 — parallel contract work** | **S.1–S.2:** one contract/status authority and probability semantics | Existing wire types and selected profile | Versioned supported contract/capabilities and source-backed status |
| **P1 — native optimization** | **Phase 3.6–3.10:** batched Q/K, adaptive scheduler, high-K/repeatability | 3.1–3.5 parity | Native parity plus target-Mac crossover/Q-amortization and branch-memory evidence |
| **P1 — release confirmation in parallel** | **2I.1–2I.2 + 2J.5–2J.6:** reviewed data, natural cases, efficient external baselines and scoped promotion | Existing draft intake plus frozen E11 profile | Distinct review, explicit release limits, fresh held-out evidence and comparable resource results [IJ; I0; IJ2] |
| **P1 — service bridge in parallel** | **S.3–S.5:** thin Rust service with resident reference worker | S.1–S.2 and one supported loadable profile | Real probabilities, bounded work, honest errors and queue-inclusive measurements |
| **P2 — conditional** | **P2.1–P2.3:** optimized kernels, weight precision or teacher/distillation | Specific unmet target after native baseline | Verified capabilities, fresh quality or same-model equivalence gates as appropriate |

Phase 3A removes Python execution mechanics from the critical path, Rust Phase 3.3 removes full-sequence CPU backbone reproduction from it, Rust Phase 3.4 removes backend-neutral branch-state semantics from it, and Rust Phase 3.5 removes sequential nested execution from it. The immediate blocker is batched question execution parity, not another model search or Colab export. Reviewed release confirmation remains necessary before production promotion, but it proceeds independently of the fixed implementation target. [IJ2; P3A; P3B; RUST3; RUST5; WP §§13.3, 16–17]

### Work to pause or keep conditional

Pause additional low-bit KV snapshot sweeps on the same short-prefix workload. Keep lossless caching and bounded FP16-KV storage as reference infrastructure; require a changed workload, codec or memory hypothesis before reopening that branch. A failed KV-snapshot codec does not reject model-weight quantization, which targets different tensors. No universal prefix-length cutoff for FP16-KV promotion is established. [WP §§9, 13.5]

Optimize model work before host serialization: the historical cached profiles place approximately **92–94%** of instrumented time in model execution and **4.6–6.6%** in cloning/expansion. Fewer forward invocations, more useful suffix packing, smaller backbones and verified faster kernels are the relevant hypotheses—not claims of measured gains in this revision. No CUDA timing is a Mac prediction. [WP §§8, 11, 13]

Do not prioritize MTP in the no-output-decoding graph, a large RLCD/RL effort before supervised baselines, or a diffusion rewrite based on R4T. Keep the later teacher-to-student hypothesis separate from the early compact bidirectional comparison. A purpose-built shared-encoder/query redesign is conditional on simpler measured approaches missing the target. Laya is motivation for a comparator, not proof of broad question competence or an accepted replacement. [WP §§11.7, 13.5]

---

## Next Milestone Target

> **Demonstrate a real, versioned OpenDecision model answering several independent questions over one state, with measured rejection, calibration, useful automation coverage, complete-request latency and peak memory on a named deployment machine.**

A request with Q independently answered questions is not evidence of one shared state encoding. A single framework forward call is not proof of shared computation. Both semantic usefulness and amortized computation must be measured, with **Q independent questions** separated from **K candidate alternatives within each Choice question**. [WP §§2.4, 13.4]

The selection objective is **correct automated decisions per second subject to predeclared accepted-error, minimum-coverage, latency and memory requirements**. Exact bounds and target hardware must be recorded before final model selection; this roadmap does not invent an unsupported safety threshold or resource budget. Report source-state counts and clustered uncertainty. Conditional accepted-error is undefined when nothing is accepted. [WP §13.4]

---

## Phase 2IJ workbench checkpoint (PREPARATION RECORDED / REVIEW GATED)

**Recorded invocation:** `2ij_reviewed_multiquestion_v1`, workbench `2ij.1.0`, exported around 16:10 UTC on 19 September 2026. The overall **`blocked`** status is a review/protocol gate, not a training crash. Full technical evidence and the coverage audit live in [whitepaper §14](whitepaper/OpenDecision_Whitepaper_v0.7.2.md#14-phase-2i2j-workbench-preparation-mechanical-evidence-and-the-review-gate). [IJ]

| Recorded contribution | Status | Roadmap boundary |
|---|---|---|
| Historical H identity/head checks and exclusions | Completed for this invocation | H stays closed; no retraining or new test-set selection |
| Review package and model revision entries | Draft prepared; unsigned | Partial preparation for 2I.1–2I.2 and 2J.1, not approved data or model comparison |
| Native probability contract and worker handoff | Draft exported | Partial S.1–S.2; no Jev/Rust/HTTP conformance result |
| Qwen4B/L4 synthetic nested-feature and root checks | Small mechanics probe completed | Partial 2I.4/2I.6 only; no trained question quality, probability/policy gate or resource grid |
| New semantic training, final selection, smaller/LoRA/ModernBERT results | Not run | Full 2I/2J tasks remain unchecked |

**Source-coverage finding:** the reported `unrelated_question_addition` fixture appends the first synthetic question again. Preserve the zero-drift duplicate-append observation, but add a unique, genuinely different question and rerun that separately named test under **2I.6**. This documentation revision does not repair the notebook or claim that new test passed. [IJ; WP §14.3]

### Immediate unblock work

In `Colab Notebooks / OpenDecision_Phase2IJ_review`, finalize the case/protocol definitions and **specify `target_device` plus all five bounds**: `max_family_macro_nll`, `max_panel_accepted_error`, `min_policy_coverage`, `max_request_p95_ms`, `max_peak_allocated_gib`. These values remain null in the inspected intake; do not infer them from the smoke-test hardware. [I0]

Run the revision-resolution/review-template refresh step after finalizing those choices. Have a **distinct independent reviewer** actually inspect and sign the matching `review.json` and protocol attestations. The existing `review.json` points to an old protocol hash; `review_template.refreshed.json` matches the inspected current protocol but is also unsigned and must be refreshed again after further changes. Changing approval booleans without review is not completion. [I0; WP §14.4]

Rerun the same study ID only while it remains unregistered and under the intended finalized design; registered inputs are immutable. Resume from complete Drive snapshots rather than this compact report ZIP. Preserve the blocked snapshot, historical H sources and protected final annotations. No parent 2I/2J/S task is closed by this checkpoint, and no new human review is supplied by this update. [IJ; WP §14.5]

---


## Phase 2IJ model-selection screen closeout (COMPLETED EXPLORATORY SCOPE)

Study `2ij_model_selection_screen_v2` / workbench `2ij.2.0` completed its requested scope. All 13 configured fit jobs, all 31 locked final profiles, bounded multi-question scaling and the selected-model export completed. Selection was frozen before final evaluation and was not changed afterward.

**Provisional integration target:** `a047d6802c3f06f085b8` — `Qwen/Qwen3.5-4B-Base`, frozen backbone, **state-first** rendering, **score-summary** rejection. Final exploratory panel: **95.0% accuracy, 0.13006 NLL, 0.06168 family-macro NLL**, with the natural MultiRC slice at **83.33% accuracy / 0.40361 NLL**. A 0.98 policy accepted 214/320 episodes with one wrong accepted decision. The point estimates are not release gates because promotion limits were unset and the corpus is not independently reviewed.

The selected bundle is exported and reload-verified. Bundle SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`. Fresh Python reload maximum probability delta is `3.6673555e-6`, with zero selected-ID changes and a passed independent NumPy/f64 head-algebra check. Base Qwen weights remain a pinned external dependency rather than being duplicated in the bundle.

**Roadmap effect:** model identity no longer blocks Rust/native engineering. Start Phase 3 parity work against this exact profile and bundle. Keep reviewed promotion, natural-document confirmation, explicit release bounds, released Laya/GLiClass adapters and conditional P2 experiments separate so implementation progress is not confused with production certification.


## Phase 2I — Genuine Multi-Question Evaluation & State-First Execution (BOUNDED PILOT COMPLETE / REVIEWED CONFIRMATION OPEN)

This workstream owns the **common evaluation foundation** before model comparison, then tests question semantics and state sharing. It must not reduce to another intent-label benchmark. H’s selection/transfer and evidence-position findings are inputs to protocol design; new interventions require fresh final data. [HF; WP §§11.6, 13.3–13.4]

- [ ] **2I.1 — Freeze the evaluation and selection contract.** Name supported task/rubric families, criteria/none semantics, truncation, target machine and quality/resource requirements. Separate training, development, calibration, policy selection and untouched final data; group related states/documents. Include development diagnostics for the intended transfer conditions without consuming final holdouts. H’s development-selected support arm did not have the strongest final transfer, so fitting-family development loss alone must not stand in for the new deployment objective. Keep H final as historical/regression evidence, not reusable fresh selection data. [HF; WP §§13.1.10, 13.1.16] **Checkpoint:** a draft exists, but device/limits and approval remain unset; follow the unblock steps above. [IJ; I0]
- [ ] **2I.2 — Build reviewed multi-task, multi-question cases.** Include the same state under instructions requiring different judgments; explicit inclusion/exclusion boundaries; negation, contradictions, irrelevant context and insufficient evidence. Keep omitted valid options, author-OOS, evidence insufficiency and application review separately labeled. Add independently labeled natural documents; report them separately from constructed wrappers. Review confusing criteria without retroactively changing G's labels or H's frozen criteria. **Checkpoint:** the review manifest lists 92 draft cases; independent review and operational natural-document evidence are not established. [I0]
- [ ] **2I.3 — Compare trained instruction-first and state-first profiles.** **Exploratory checkpoint:** state-first materially outperformed instruction-first for the tested Qwen profiles and is part of the selected integration contract; retain this task open for independently reviewed confirmation.  Fit compatible heads/normalization/calibration under each rendering using the same semantic comparison contract. State-first is a new causal input/model contract, not a cache-only patch expected to preserve the old head's behavior. Compare each optimized implementation to its own full-prompt reference.
- [ ] **2I.4 — Implement and verify nested sharing.** **Exploratory checkpoint:** nested Q=1/Q=4 semantic execution and all 12 synthetic mechanics cells passed their declared parity checks; keep open for broader natural/reviewed coverage and native parity.  On the Qwen path, prefill stable format + state once, fork isolated question states, then fork isolated candidate suffixes. Isolate recurrent DeltaNet, convolution and attention-KV state at both branch points. Attention masks alone do not isolate recurrent streams. For a different architecture, document and test its actual sharing mechanism rather than claiming Qwen cache semantics. **Checkpoint:** one Q = 2, K = 2 synthetic GPU feature probe and root check completed. Retain this partial evidence without closing semantic/probability or broader scaling requirements. [IJ; WP §14.2]
- [ ] **2I.5 — Measure Q/K scaling, amortization and resource cost.** **Exploratory checkpoint:** 64/256/1,024-token state mechanics with Q∈{1,4,16} and K∈{2,4,8,16} were sampled in the predeclared 12-cell screen; long-prefix reuse showed large execution savings, but semantic Q=16 and service-inclusive latency remain open. Start with proposed state-length targets **64, 256 and 1,024 tokens**, **Q ∈ {1,4,16}**, and **K ∈ {2,4,8,16}** where applicable. Compare three execution shapes on the same profile: repeated full requests, sequential nested sharing, and batched nested question execution. Report `T(Q)/T(1)`, marginal latency per additional question, questions/s, peak/resident memory, branch-state bytes, forward calls, state-prefill share, accepted-error and coverage. Separate startup, warm-model cold-state, warm-state reuse and queueing. External Jev/Laya Q-scaling is context, not an acceptance threshold. [EXT2]
- [ ] **2I.6 — Test semantic and execution isolation.** **Fixture correction completed:** distinct-question-v2 was executed; bounded question-order, added-question, opaque-ID and candidate-order checks showed no recorded policy changes. Keep the broader task open for reviewed/natural cases and native execution.  Test question-order changes, adding/removing unrelated questions, opaque-ID renaming, candidate permutation, evidence placement and declared truncation. Expected changes from altering the candidate set are not equivalent to unwanted cross-question influence. Test root immutability and independent branches, not only output shape. **Required fixture correction:** use a unique, genuinely different added question and assert non-duplication; keep the old duplicate-append result as separate regression evidence. [IJ; WP §14.3]

- [ ] **2I.7 — Add state/evidence-sufficiency strata.** Treat deliberately omitted candidates, author-OOS, insufficient evidence, unobservable state and application review as separate targets. Add reviewed cases where the requested judgment cannot be supported from the visible state, including temporally incomplete or lossy summaries. The DGX Doom study shows that changing the textual observation/history representation can materially change controller behavior even when the underlying environment is the same; use that as motivation for explicit state-sufficiency evaluation, not as OpenDecision task evidence. [EXT2; WP §§5.4, 11.6]
- [ ] **2I.8 — Add high-cardinality systems stress.** Extend systems-only K sweeps through **32, 64, 128 and 255** options before claiming compatibility with high-cardinality Choice. Measure latency, branch memory, suffix/token-tree behavior and scheduler saturation separately from semantic quality. Follow with a much smaller reviewed high-K quality panel. The public PCD project's 255-choice timing demonstrates feasibility of a constrained-logit path, not accuracy at 255-way dynamic semantics. [EXT1]

**Exit evidence:** versioned data and rendering manifests; clearly bounded question-family quality; full-request Q/K measurements; per-profile numerical/action comparisons; and an honest result when sharing costs more or reduces semantic quality. Report both batched repeated-state and genuinely shared-state baselines. Qwen's nested path and a compact model need not have the same optimal execution graph. [WP §§11.2, 13.4; proposed gate]

---

## Phase 2J — Early Matched Adaptation, Rejection & Model Sizing (EXPLORATORY SCREEN COMPLETE / RELEASE CONFIRMATION OPEN)

The exploratory 2J screen has already completed the matched Qwen4B/Qwen2B/ModernBERT/rejection comparison needed to choose a provisional integration target. The remaining 2J work is **release confirmation and fair external baselines**, not a prerequisite for Rust parity. Retain the original-criteria joint control and support-example ablations for future reviewed confirmation; do not retrospectively promote a different arm from exposed final data. [HF; IJ2; WP §§13.1.10, 15–16]

- [x] **2J.1 — Pin the comparison matrix.** **Closed for the exploratory screen:** immutable Qwen4B/Qwen2B/ModernBERT revisions and the declared screen matrix were locked before fitting.  Compare frozen 4B + newly fitted multi-task heads; limited 4B LoRA; `Qwen/Qwen3.5-2B-Base` under an appropriate matched supervised recipe; and a compact bidirectional joint-candidate scorer, with a **ModernBERT-style arm motivated by Laya**. Resolve immutable revisions before fitting, including tokenizer/renderer/head versions and inference dependencies. Do not assume equal token IDs across model families, or infer actual 2B memory by halving 4B measurements. **Checkpoint:** three revisions and nine planned arms are present in the unsigned intake; no approved comparison or new model profile exists yet. [I0]
- [x] **2J.2 — Run attributable supervised adaptation.** **Closed for the exploratory screen:** frozen, online-head, 2B LoRA, ModernBERT full-encoder and finite-token adaptation treatments completed under the registered pilot.  Hold criteria, split semantics and loss/update budgets fixed within each intended ablation; disclose unavoidable cross-family training differences. Refit normalization and calibration when representations change. For the optional H-style online LoRA recipe, compare against its matched online-head-only BCE control—not directly against joint-CE results as an equal-training comparison. No RL/RLCD reproduction is claimed.
- [x] **2J.3 — Compare feature-aware applicability/rejection.** **Closed for the exploratory screen:** constant, score-summary and semantic-feature rejection variants were retained and evaluated; score-summary was selected pre-final for the integration candidate.  Retain constant-none and score-summary set-linear controls. Add a small head that can use already-computed candidate-conditioned semantic features, with explicit applicability and absent-answer supervision. Test whether it improves rejection without unacceptable false-none or ranking losses; do not preselect it as a replacement. Separate omitted options, OOS and insufficient evidence in evaluation, and measure any additional feature/cache cost.
- [x] **2J.4 — Extend bounded Noul/Score evidence.** **Closed for this pilot scope:** dedicated binary and described ordinal questions were included with full distributions and ordinal diagnostics; operational/general primitive validation remains outside this closure.  H’s BoolQ/SST-5 final evaluation is closed and supplies a bounded baseline. This task remains open for new binary questions, explicit insufficient-evidence cases and described ordinal rubrics on held-out families. Start from a proper distribution loss such as NLL and assess ordinal action costs separately; an expected-distance penalty is not itself a proper probability score or calibration guarantee. Treat cumulative-probability Brier training as a separate experiment. [HF; WP §13.1.14]
- [ ] **2J.5 — Use fair efficient baselines.** **Partial:** lexical, frozen/trained finite-token and newly trained ModernBERT controls completed. Released Laya and GLiClass adapters remain deferred, so the broader baseline task stays open.  Retain H's inexpensive lexical and unadapted finite-code controls, clearly labeled. Add a trained finite-token/SALSA-style control, a **state-first parallel-constrained-decoding (PCD) latency-floor baseline** where candidate tokenization permits it, and a dynamic-label/GLiClass-style comparator where task-compatible. The PCD baseline should use OpenDecision's state-first isolation contract rather than copying a schema-before-state shared prefix. Avoid long JSON generation as the only baseline; disclose input budgets, supervision, trainable parameters and complete request costs. External benchmark numbers are not OpenDecision results.
- [ ] **2J.6 — Select a scoped quality/resource operating point.** **Exploratory candidate selected:** Qwen4B/state-first/score-summary is locked and exported, but release selection remains open because the pilot is not independently reviewed and promotion bounds were unset.  Use untouched final data, held-out state/question/rubric families, rejection strata, calibrated-distribution diagnostics, accepted-error/coverage and scenario costs. Measure actual resident/peak memory and latency on the named device. Promote a model only within its supported scope; close an unsuccessful arm without changing selection rules after seeing its final outcome.

**Compact-arm specification (2J.1–2J.2).** Retain a full-fine-tuning ModernBERT-style joint-candidate treatment in the early comparison, disclosing its trainable parameters and training budget rather than claiming identical adaptation cost to LoRA. Evaluate the **released Laya checkpoint separately** as an external baseline under 2J.5; its prior supervision is not a matched training treatment. Test held-out question/criteria families, omission/OOS/insufficient-evidence behavior, calibration, accepted-error/coverage and complete latency/memory. Do not silently truncate decisive evidence, substitute maximum probability for the defined confidence statistic, or treat question batching as proof of one shared state encoding. Architecture and marker scoring are test candidates, not adopted superiority claims. [WP §§11.2, 11.7; RC RQ-06]

One **proposed**, not yet fitted, applicability formulation is:

```text
a = P(at least one offered candidate is valid | state, question, candidates)
P(none) = 1 - a
P(candidate_j) = a × P(candidate_j | an offered candidate is valid, state, question, candidates)
```

The possible benefit is richer evidence and supervision, not the reparameterization alone. Rejection cannot repair the relative ranking of two candidates unless the ranking model also changes. Test both. Keep semantic none separate from an external review action and from the vendor's Noul primitive. [WP §§5.4, 11.6; proposed experiment]

**Exit evidence:** matched comparison manifests, learning/selection histories, independent final results, calibrated/rejection behavior and measured resource trade-offs. A smaller Qwen or compact encoder is an early candidate, not an assumed winner; the 4B reference may remain useful as an evaluator or later teacher without being the deployment choice. [WP §13.3]

---

## Integration Track S — Contract Hardening & Thin Real-Service Bridge (PLANNED / PARALLEL)

The supplied roadmap describes implemented mock transports and wire fixtures. It does **not** establish a resident real-model service. Keep that distinction visible in `/v1/models`, documentation and tests. Track S is separate from the numbered H experiment so notebook completion cannot accidentally mark HTTP/security work done. [R0; WP §§11.8–11.9]

- [ ] **S.1 — Consolidate contracts and current status.** Define the decision specification; model/execution profile; backend capabilities; and evaluation context (tenant, budget, deadline, cancellation and tracing). Keep this roadmap as task/status authority and the whitepaper as evidence/interpretation authority. Pin compatibility fixtures/schema revision; record code test totals only with commit and environment. Synchronize `AGENTS.md`, architecture docs and research instructions without inventing a current-HEAD test claim.
- [ ] **S.2 — Resolve probability and candidate semantics before integration.** Choose an explicit supported mapping for internal none: caller-supplied semantic option, separately versioned native representation, or a review mechanism outside the probability vector. Do not silently append an unrequested class or drop none and renormalize while calling the probabilities unconditional. Separate stable machine IDs from semantic labels/descriptions; define null-description behavior and adapter semantics. Preserve the separate definition of `confidence` rather than replacing it with maximum candidate probability. **Checkpoint:** a native draft was exported with an explicit non-conformance flag; approval and real wire integration remain open. [IJ]
- [ ] **S.3 — Wire a resident reference worker to the existing Rust service.** Start with one persistent Python worker and one bounded queue behind `DecisionEngine` / `POST /v1/systemone`; load a supported, pinned reference profile once and return actual probabilities in validated responses. Reference operation is not a deployment-quality endorsement. An H profile remains non-production despite completed offline evaluation; advertise only its actually validated scope. Unsupported primitives or shapes must fail explicitly, never return mock answers disguised as model output.
- [ ] **S.4 — Test admission, lifecycle and deployment safeguards.** Bound request bytes, state length, Q, K, work and memory; define cancellation before/after dispatch, deadlines, queue rejection and worker-failure recovery. Require explicit opt-in for unauthenticated non-loopback serving, protect metrics and redact sensitive inputs. Test tenant separation and active-reader state lifetime. These are requirements, not confirmed vulnerabilities or already-implemented protections.
- [ ] **S.5 — Measure the real service.** Record queue-inclusive end-to-end latency distributions, throughput, accepted-error/coverage where labels exist, startup versus resident requests, peak/process memory and model-profile identifiers. Test concurrency, request IDs, validation, cancellation and leak/soak behavior. Do not substitute offline Colab medians for these measurements or extrapolate L4 latency to Apple Silicon.

**Exit evidence:** real non-mock request/response fixtures, supported-capability declarations, bounded-work and failure tests, and a named-machine service report. This milestone does not require a native Qwen port, all primitive semantics, or arbitrary-domain generalization. [WP §11.8; proposed integration gate]

---

## P2 — conditional efficiency and teacher-to-student studies (PLANNED / AFTER A PROMISING PROFILE)

Retain these experiments, but do not put them ahead of the reviewed-data and early
model-selection round. A small verified-backend experiment may proceed earlier
when it answers a specific bottleneck question. [WP §§11.7, 13.3, 13.5]

- [ ] **P2.1 — Verify optimized execution.** Pin an environment and establish
      actual causal-convolution/linear-attention kernel dispatch, supported
      arithmetic, and access to required hidden states/cache operations. Package
      installation is not evidence of a fast path. Compare complete costs and
      same-model equivalence when claimed; use new-profile quality gates when
      behavior changes.
- [ ] **P2.2 — Weight-precision/quantization candidate.** After selecting a
      promising model and backend, test its own calibration, rejection, policy
      behavior, latency and total memory. Weight quantization is not the failed
      attention-KV snapshot experiment; neither result establishes the other.
- [ ] **P2.3 — Bounded teacher/student comparison.** For the same student,
      compare reviewed labels alone, labels plus reviewed teacher-generated
      examples, then an additional distribution-distillation treatment. Keep
      final labels independent of the teacher, preserve candidate/none semantics,
      and report teacher-generation cost as well as student serving cost. The 4B
      reference is a possible teacher, not automatically a better one. R4T is
      motivation for offline supervision in the whitepaper, not evidence for a
      diffusion-based OpenDecision architecture.

**Paused or conditional:** no further low-bit KV snapshot sweep on the same
short-prefix workload without a new hypothesis; no MTP work for the current
no-output-decoding graph; no large RLCD-inspired program before strong supervised
baselines; no diffusion rewrite from a retrieval analogy. A purpose-built shared
encoder/query network remains a later fork only if simpler paths miss the target.
Retain lossless and bounded FP16-KV infrastructure and all failed configurations
as historical evidence. There is no universal prefix-length cutoff at which
FP16-KV must be selected. [WP §§11.2, 13.5]

---

## Phase 3A — Python BranchableState & Batched-Q Systems Validation ✅ COMPLETED NOTEBOOK SCOPE

**Run authority:** `20260920T024056Z`, schema `opendecision-phase3a-summary/v1`, against profile `a047d6802c3f06f085b8` and bundle SHA-256 `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`. The run explicitly records `model_changed=false`, `training_performed=false`, and `selection_performed=false`. It is an implementation/systems experiment over the already selected profile, not another model-selection stage. [P3A; WP §16]

- [x] **3A.1 — Validate full-hybrid fan-out/select semantics in Python.** The first generic Transformers cache-batching attempt exposed that Qwen3.5 `LinearAttentionLayer` does not implement the top-level `batch_repeat_interleave` helper. The corrected reference deep-copies the complete hybrid cache and uses cache-wide `reorder_cache(indices)`, so repeated indices fan out attention KV, recurrent DeltaNet state and convolution state together. Root immutability/fan-out/select is checked before semantic benchmarks.
- [x] **3A.2 — Validate semantic batched parity.** Three nonfinal semantic smoke states passed the notebook's batched parity gate.
- [x] **3A.3 — Validate high-K systems parity.** Systems-only high-cardinality K stress passed parity. This is not high-K semantic-quality evidence.
- [x] **3A.4 — Record repeatability.** The recorded same-process repeatability maximum probability delta is **0.0** with **0 argmax changes**.
- [x] **3A.5 — Measure the Q/state-length crossover.** The result is deliberately mixed:
  - short semantic Q=4: median-of-case `repeated_full` ≈ **869.3 ms**, `nested_batched_warm` ≈ **962.4 ms** — warm sharing is about **10.7% slower** here;
  - L=256/Q=4/K=4 mechanics: **2,934.3 → 507.6 ms** repeated-full to warm batched (**5.78×**);
  - L=1024/Q=4/K=4: **9,448.1 → 516.5 ms** (**18.29×**);
  - L=1024/Q=16/K=2: **18,928.6 → 1,922.0 ms** (**9.85×**).
- [x] **3A.6 — Record memory trade-off.** At L=1024/Q=16/K=2, peak allocated memory rises from roughly **16.42 GiB** for repeated-full to **18.67 GiB** for batched-cold.
- [x] **3A.7 — Export the Rust handoff contract.** The Python reference defines complete-state root identity, immutable fork, batched fork/select, explicit position, byte accounting, exact/declared length bucketing and profile-bound state identity. Rust may use a different representation but must reproduce the semantics and fixtures.

**Systems conclusion:** `BranchableState` is no longer merely an architectural proposal; its required semantics have a Python reference. The optimization policy remains unresolved because short semantic requests and long shared-state mechanics have different winners. Rust must preserve explicit `repeated_full`, `nested_sequential`, and `nested_batched` strategies until a target-machine scheduler is measured.

**Public reference artifact:** <https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>. Publication does not change the `release_quality_claim=false` or `rust_metal_parity_claim=false` boundary recorded by Phase 3A. [PUB1]

**Phase 3A.1 follow-up is conditional, not the next blocker.** Begin Rust parity now. Create a dedicated scheduler/crossover Colab only if native component profiling cannot derive stable crossover rules from state length, Q, K, suffix buckets, cache warmth and memory pressure. Such a notebook must keep the model/profile frozen and must not become another model-search stage.

---

## Phase 3B — Python Qwen Backbone Parity Reference ✅ COMPLETED NOTEBOOK SCOPE

**Run authority:** `20260920T152206Z`, schema `opendecision-phase3b-backbone-parity-summary/v1`, against profile `a047d6802c3f06f085b8`, bundle SHA-256 `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`, and base revision `1001bb4d826a52d1f399e183466143f4da7b741b`. The run records no training, model selection, or model modification. [P3B]

- [x] **3B.1 — Export exact segmented token fixtures.** Four records cover root, question, candidate suffix, and full candidate IDs.
- [x] **3B.2 — Export ordered layer diagnostics.** The trace contains embedding, 32 decoder outputs, and final RMSNorm, for 34 stages total.
- [x] **3B.3 — Export final candidate features.** Ten candidate vectors cover the four published decision fixtures.
- [x] **3B.4 — Export continuation diagnostics.** Root, question, and candidate continuation vectors are present with complete hybrid-cache identity and position metadata.
- [x] **3B.5 — Preserve the probability contract.** Fresh full-sequence outputs reproduce the selected head within the existing `0.005` probability and zero-decision-change gates.

The Rust diagnostic loader verifies the Phase 3B model/config identity, 426-entry parameter inventory, 24 DeltaNet plus 8 full-attention layer order, 47-vector tensor inventory, tensor hashes, trace order, and continuation positions. The native loader verifies the pinned checkpoint config, safetensors index, both shard sizes and SHA-256 digests, BF16 tensor layouts, token bounds, and finite values. The diagnostic embedding matches exactly. The 34-stage native trace reaches final RMSNorm with maximum absolute diagnostic error `5.8174e-05`, RMS `1.1531e-05`, and cosine `0.999999999993`; these hidden-state values remain localization diagnostics, not new acceptance tolerances. Across 10 native candidate sequences, maximum feature delta is `1.0300e-04`, maximum probability delta is `4.5869e-06`, and there are zero argmax or policy changes. Qwen-specific cached continuation preserves the root, matches positions `98 → 109 → 121`, accounts for exactly `59,899,904` root-state bytes, and matches native full-sequence output exactly for the exported candidate. This is CPU/Candle evidence, not Metal, service, or release evidence.

---

## Phase 3 — Production Rust Engine & Selected Native Backend (IN PROGRESS / INTEGRATION PROFILE LOCKED)

**Integration profile for parity work:** `a047d6802c3f06f085b8` (`Qwen/Qwen3.5-4B-Base`, frozen backbone, state-first renderer, score-summary rejection). The A100-verified model bundle is the model/probability reference contract; completed Phase 3A adds the Python full-hybrid branching/batching reference and Rust handoff fixtures. A later reviewed confirmation may change release promotion without invalidating parity work against this versioned profile. The same integration line is published at <https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>. [P3A; PUB1]

The new external benchmarks do **not** reopen model selection. They sharpen the implementation objective: after exact native parity, OpenDecision should turn the already validated state-first/nested graph into a **breadth-first batched execution graph** and measure how question latency scales with Q. The public PCD implementation demonstrates practical cache broadcasting and constrained-logit batching on Apple Silicon; the DGX study shows that near-flat Q=1→4 latency is a meaningful Jev-style systems property. Neither reveals Jev's private architecture or supplies an OpenDecision acceptance threshold. [EXT1; EXT2; WP §11.7.1]

### Phase 3 execution sequence

- [x] **3.1 — Head/probability parity.** Rust reproduces normalization, candidate/primitive projections, score-summary rejection, stable softmax, calibration, and policy semantics across all four exported fixtures. This is host-side readout parity, not backbone parity.
- [x] **3.2 — Exact tokenizer and state-first renderer.** The digest-locked offline tokenizer reproduces all four exported `root_ids`, `question_ids`, `candidate_suffix_ids`, and `full_candidate_ids` records exactly. Segment boundaries, no-special-token behavior, candidate order, `max_tokens=1792`, and reject-without-truncation behavior are executable contracts. Phase 3.3 executes the corresponding positional encoding and causal-mask math.
- [x] **3.3 — Full Qwen3.5 backbone parity on the target local runtime.** The correctness-first Candle `0.8.0` CPU path on an Apple M4 Max (`Mac16,5`, 36 GiB, macOS 26.6.2) verifies both immutable BF16 checkpoint shards, widens weights for FP32 execution, and runs embedding, `layer_00` through `layer_31`, and final RMSNorm. The 34-stage trace has exact embedding and reaches final norm with the diagnostic values recorded above. All 10 candidate features preserve all four full distributions within `0.005`, with zero argmax and directed-policy changes. The four-question run took `399.75 s`; `/usr/bin/time -l` reported `7.77 GB` maximum resident accounting including mapped model shards and a `680 MB` peak memory footprint. Those numbers describe the current CPU reference implementation, not optimized throughput. The `1e-5` absolute-logit gate remains the Phase 3.1 fixed-feature algebra gate: Phase 3B's own freshly exported features miss that older fixture by up to `9.6905e-05`, and the native features by up to `2.5652e-04`, while the declared Phase 3B probability and discrete-decision gates pass. The older absolute comparison is not relabeled as a backbone failure or widened silently.
- [x] **3.4 — Implement native `BranchableState` against the validated Python contract.** Represent every mutable continuation component required by the backend. For Qwen3.5 this includes full-attention KV, DeltaNet recurrent state, convolution state, logical position and profile identity. Required operations support immutable-root verification, single fork, batched fork, gather/select, byte accounting and stable cache identity. The backend-neutral contract (`crates/opendecision-backends/src/branch`) and the `BackboneState` binding now carry profile/model/tokenizer/renderer/arithmetic identity, process-local branch lineage, a structural scheduling fingerprint plus a strict content fingerprint, and exact attention-KV/recurrent/convolution/metadata byte accounting. **Native checkpoint:** the checkpoint-gated `branch` stage replays the Phase 3B `root → question → candidate` continuation through the contract at positions `98 → 109 → 121` with the exact `59,899,904`-byte root (`6,422,528` attention-KV, `50,331,648` recurrent, `3,145,728` convolution tensor bytes), an unchanged root under fork/fan-out/gather, two-lane fan-out whose independently advanced lanes exactly match single-fork replay, preserved lane identity under reorder/duplicate gather, and a cached continuation state exactly equal to the independent full-sequence state (maximum absolute delta `0.0`). Hidden-vector deltas (root `1.1444e-04`, question `7.8201e-05`, candidate `6.8665e-05`) remain localization diagnostics. This is a CPU/Candle result; it does not establish Metal, sequential or batched nested execution, service registration, or release promotion. **Phase 3A checkpoint:** the Python reference had already validated complete-state fan-out/select using cache-wide reindexing; Rust reproduces, not rediscovers, those semantics. [P3A; RUST4]
- [x] **3.5 — Reproduce sequential nested execution.** Implement `state → question → candidate` branching against the Phase 3A/Python reference and pass full probability/argmax/policy parity plus root-immutability/isolation tests. This is the native correctness baseline for later vectorization. **Native checkpoint:** the checkpoint-gated `qwen35_nested_parity` stage prefills each fixture case's shared state exactly once, then executes all four Phase 3B questions and all 10 candidates through immutable `BranchableState` forks (`run_sequential_nested`/`BackboneState`). Declared gates pass: maximum probability delta `4.5869e-06` against the golden head fixtures under the `0.005` contract, zero argmax changes, and zero policy changes against `PROBABILITY_REFERENCE.json`. Isolation gates pass: the retained root equals an independent fresh prefill exactly (strict content fingerprint and root feature delta `0.0`) after all question and candidate work, question and candidate replay from the retained fork states is exact (`0.0`), reversed sibling-order advancement is exact, and every position matches the exported token fixtures. The `repeated_full` oracle agrees exactly: cached-versus-full candidate features are `0.0` for all 10 candidates and every cached continuation state's strict fingerprint equals its independent full-sequence state. Feature deltas against the frozen Phase 3B vectors reach `1.0300e-04`, and the traced continuation vectors replay at root `1.1444e-04`, question `7.8201e-05`, and candidate `6.8665e-05` — the same localization diagnostics recorded by Phase 3.3/3.4, not new tolerances. This is a CPU/Candle result; it does not establish Metal, batched Q/K execution, service registration, or release promotion. [P3A; RUST5]
- [ ] **3.6 — Batch the question layer.** Fork the immutable state root into a Q batch and evaluate compatible question suffixes breadth-first. Preserve question isolation and compare against 3.5. **Phase 3A checkpoint:** Python semantic batched parity passed; the native task is parity plus target-machine performance.
- [ ] **3.7 — Batch candidate branches within question states.** Use exact/declared length-aware buckets, retaining candidate permutation invariance and score-summary rejection semantics. **Phase 3A checkpoint:** high-K Python systems parity passed; native K batching still requires its own parity and memory evidence.
- [ ] **3.8 — Build the adaptive scheduler and Mac Q-amortization curve.** Compare `repeated_full`, `nested_sequential`, and `nested_batched` on the named Mac. Phase 3A disproves an “always share” rule: the short semantic Q=4 warm-batched path was ~10.7% slower than repeated-full, while long-state mechanics reached 5.78×–18.29× warm speedups. Select by measured state length, Q, K, suffix-length buckets, cache warmth, available memory and backend/precision. Report p50/p95, `T(Q)/T(1)`, marginal ms/question, questions/s, forward calls, state-prefill fraction and branch-state bytes. [P3A]
- [ ] **3.9 — High-cardinality and scheduler stress.** Reproduce K=32/64/128/255 systems fixtures natively, add mixed Q/K lengths and memory-pressure cases, and verify scheduler fallback. Keep semantic high-K quality under 2I.8.
- [ ] **3.10 — Repeatability and persistence.** Reproduce Phase 3A's exact same-process replay result where the native runtime permits it, then broaden to fresh-process and persistent-state replay. Distinguish deterministic equality from declared floating-point tolerance. [P3A; EXT2]
- [ ] **3.11 — Production backend and service lifecycle.** Integrate the selected supported backend with device discovery, queueing, bounded admission, persistent state reuse, TTL/tenant separation, cancellation, load shedding, recovery, telemetry and soak tests. Select supported Candle/GGUF/ONNX/other paths by actual Qwen3.5 hidden-state/branch capabilities rather than promising all backends.

### Per-profile parity requirements

Every optimization that claims to preserve `a047d6802c3f06f085b8` must keep the selected finalized tokens, weights, renderer, heads, rejection semantics, calibration and policy fixed and must pass declared probability, argmax and directed-policy checks. For the measured 4B profile, recurrent branch storage remains a material cost; total admission accounting must include model weights, recurrent/convolution/KV state, temporary branch expansion, restoration buffers and allocator overhead. FP16 attention-KV remains a tested storage candidate only for configurations where its profile-specific gate passes.

**Release gate:** commit-stamped fixtures/tests; named-machine parity, Q-amortization and load/soak results; explicit task/capability limits; a memory and latency envelope; repeatability results; and no unreviewed probability/API changes. Kernel, precision, model, adapter, renderer, head, policy or cache changes trigger the appropriate equivalence or new-model review. This revision asserts Rust head, exact-token, and correctness-first CPU backbone/continuation parity for the frozen fixtures. It does not assert backend-neutral branching, batched execution, service integration, full Rust parity, release promotion, or Metal parity. [WP §§11.3–11.4, 13.2, 13.4; EXT1; EXT2]

---

## Status snapshot

| Phase / track | Current description | Status / authority |
|---|---|---|
| Phase 0 | Wire contract and core types | Reported done in supplied roadmap; 43 historical tests, not rerun here |
| Phase 1 | Mock daemon, HTTP/gRPC transports and SDK compatibility fixtures | Reported done; 152 other workspace tests, 195 reported total; current HEAD unverified |
| Phase 2A | Initial Qwen probe | Historical exploratory result; not an equal-quality benchmark |
| Phase 2B | Fixed NLI heads and architectural audit | Measured, done — `20260917T205849Z` |
| Phase 2C | Dynamic schemas and batching diagnostics | Measured, needs review — `20260917T222948Z`; no completed LoRA/scaling evidence |
| Phase 2D | Precision, rejection and full requests | Measured, needs review — `20260917T234417Z` |
| Expanded Phase 2E | Shared-prefix agreement and request cost | Measured, done — `20260918T114914072764Z`; acceptance differs by precision |
| Phase 2F | Snapshot compression and prefix persistence | Measured, done — `20260918T224427722898Z`; four low-bit codec gates failed |
| Phase 2G | Fresh decisions, TF32 and lifecycle | Measured, done — `20260919T005142584348Z`; semantic and arithmetic limits remain |
| **Phase 2H** | **Criteria/rejection transfer and bounded primitive readouts** | **Completed required scope — `20260919T040612625670Z__finish_2h_1_2`; both workers complete; original failed attempt preserved** |
| **2H-C1–C5** | Preserve/recover/close the original H study | **Closed** — continuation, original artifact/selection lock, final readout and v0.6 handoff; acceptance remains bounded |
| **Phase 2I** | Multi-question evaluation, rendering and sharing | **Bounded exploratory evidence completed**: corrected isolation fixture, Q=1/Q=4 semantic sharing and 12-cell Q/K/length mechanics grid measured; independent reviewed/natural-data confirmation and broader generality remain open [IJ2] |
| **Phase 2J** | Matched adaptation, smaller/compact models and feature-aware rejection | **Exploratory screen completed**: 13 fit jobs, 31 final profiles, pre-final selection of Qwen4B/state-first/score-summary; release selection remains open pending reviewed confirmation and declared promotion bounds [IJ2] |
| **Track S** | Contracts and resident-reference-worker service bridge | Contract/model-bundle handoff available; real Rust HTTP/native model integration, none/confidence mapping and service measurements remain open |
| **P2.1–P2.3** | Verified optimized execution, weight precision and teacher/student tests | Conditional after a promising profile; restored tracked work, not completed experiments |
| **Phase 3A (Python)** | Full-hybrid BranchableState reference, batched Q/K systems validation and crossover measurement | **Completed notebook scope — `20260920T024056Z`**; semantic/high-K parity passed, exact recorded same-process replay; no model change, release promotion or Rust/Metal claim [P3A] |
| **Phase 3B (Python)** | Exact Qwen token, layer, candidate, and continuation reference export | **Completed notebook scope — `20260920T152206Z`**; 47 FP32 vectors, no model or bundle change, no Rust/Metal claim [P3B] |
| **Phase 3 (Rust/native)** | Native parity, branchable hybrid state, sequential nested execution, adaptive batched Q/K execution and production lifecycle | **In progress:** 3.1 head/probability, 3.2 exact-token, 3.3 CPU full-sequence/cached-continuation, 3.4 backend-neutral branch-state, and 3.5 sequential nested gates pass. Batched question execution is next |

### Superseded task mapping

The uploaded roadmap assigned “Phase 2H” to proposed engineering work, while the notebook had already used that phase name for criteria/rejection transfer. This revision restores the experiment's name and explicitly relocates the uncompleted proposals. Existing issue links should retain this mapping rather than implying that a newly reused number means old work shipped. [R0; H]

| Old roadmap task | Current location | Disposition |
|---|---|---|
| Old 2H.1 — contract/status and regression suite | S.1 and 2H-C1 | Still open operational work; historical evidence retained |
| Old 2H.2 — independent criteria/annotation audit | 2I.2; H closeout preserves H's frozen criteria | Still open; H's completed support-example study does not satisfy independent review |
| Old 2H.3 — feature-conditioned rejection replacement | 2J.3 | Planned comparison, not predetermined replacement or completed H result |
| Old 2H.4 — Python worker/HTTP bridge | S.3–S.5 | Planned integration, not a Colab experiment outcome |
| Old 2H.5 — probability/API pinning | S.2 | Still open |
| Old 2I.1 / 2I.2 / 2I.3 | 2I.3 / 2I.4 / 2I.5–2I.6 | Preserved scope, now preceded by shared evaluation/data tasks |
| Old 2J.1–2J.2 — 2B and adaptation | 2J.1–2J.2 | Brought forward; compact bidirectional comparator added |
| Old 2J.3 — multi-task supervision | 2I.1–2I.2 | Moved before matched training so data is not built after comparison starts |
| Old 2J.4 / 2J.5 — primitives / competitors | 2J.4 / 2J.5 | Retained; H fitting distinguished from final validation |

---

## Conventions for agents

Read the **current status**, **two acceptance tracks**, and **task crosswalk** before treating historical “next” paragraphs as current assignments. Historical result sections retain their original experiment-specific conclusions; the current program above controls new work.

- **Wire types live in `opendecision-core`.** Treat the pinned schema and compatibility fixtures as contracts. A wire change needs explicit versioning, round-trip tests, regenerated schemas and architecture-document updates. Confirm actual type/constant names in the repository rather than copying an unverified identifier from prose.
- **Evidence and implementation status are separate.** A passed head, loader, or embedding-only fixture is not full-backbone parity. The frozen 34-stage CPU trace plus candidate/probability/decision and continuation gates establish the bounded Phase 3.3 CPU checkpoint, not Metal or full Rust parity. A source-reported test total is not a current CI result; a saved fit is not a completed final evaluation. Record commit, command, environment and result for any new code/test claim.
- **A new backend implements the declared engine contract.** Wire it through the existing registry/service boundary with fixture and end-to-end tests. Advertise only supported tasks, shapes and precision modes; never mask missing capabilities with mock or semantically different fallback output.
- **Phase 2 owns modeling; Track S owns the thin bridge; Phase 3 owns native production promotion.** Preserve the reference while comparing new models on fresh quality gates. Do not block independent data/service work on every H stage, or bypass a selected profile's required quality/parity validation.
- **Preserve evaluation lineage.** Do not open protected final stimuli for exploratory tuning, change old gold labels, overwrite a failed attempt, delete reservations, or turn generic “requested” status text into completed checkboxes. Update the roadmap and whitepaper together when result authority changes.

## Quick reference

```bash
# Build and test the checked-out repository; record the exact commit and result.
git rev-parse HEAD
cargo build --workspace
cargo test --workspace          # Prior supplied snapshot: 195; current count must be measured.

# Regenerate JSON Schema using the existing project command.
cargo run -p opendecision-gen-schemas -- --write

# Existing mock daemon example, loopback development only; not a real Qwen backend.
cargo run -p opendecisiond -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest

# Exercise the existing request surface; a valid response does not prove model quality.
curl -sS -H 'content-type: application/json' \
     -X POST http://127.0.0.1:18080/v1/systemone \
     -d @examples/04_mixed.json | jq
```

The commands are retained as project usage examples, not commands run during this documentation revision. The `jev-latest` mock alias does not identify TypeSafe's proprietary model or a validated OpenDecision model.

## Source and revision register

**R0 — Uploaded roadmap.** `ROADMAP(20260919-123538).md`. Authority for supplied code milestones and historical test counts, not independent current-repository verification. Its numerical tables in the 2A–2G history are retained unchanged. Corrections to the exploratory interpretation and current scope are explicitly recorded in the companion change log.

**WP: Latest whitepaper lineage.** The current [whitepaper v0.7.2](whitepaper/OpenDecision_Whitepaper_v0.7.2.md) retains the delivered v0.6/v0.6.1 lineage. Older v0.3 and pre-H files surfaced in the conversation remain historical, not replacement bases. E1–E8 preserve earlier evidence, E9 the completed H continuation, and E10/I0 the blocked workbench and unsigned intake. The v0.4 refocus, 16-group review mapping and all prior results remain intact; B–H experiments were not rerun here.

**H — Saved Phase 2H attempt.** `Google Drive / Colab Notebooks / OpenDecision_Phase2H_results / 20260919T040612625670Z`. [Full summary](https://drive.google.com/file/d/1IT4cJN74vgOW0td7haE_bl2KfviHiaP1/view), [compact summary](https://drive.google.com/file/d/1525L3h-0hVKtgm50c_IAX1dCAKHNZNUe/view), [attempt archive](https://drive.google.com/file/d/1X8JP-8hhb3lu_PmMCWNLovMMdXPK6eDO/view), and [saved Colab notebook](https://colab.research.google.com/drive/1fhRJTek7Ura3aSdItwBjJTJubXUIee4a). The notebook and exported result snapshots, saved around 05:03–05:04 UTC on 19 September 2026, describe a partial attempt. No inaccessible live runtime or later unsaved continuation is inferred.

**V — Version 0.5.1 documentation checks.** The companion source reconciliation checks the saved notebook's `finish()` output against the archive; the standalone summary against its archive copy; profile/development/selection metadata; fitted-file presence; all 37 source hashes; and the absence of a final lock/evaluation worker in the retained inventory. It does not run Qwen, recompute development losses from absent feature/logit caches, inspect reserved final stimuli, repair Colab, test Rust, or modify persistent files. Hashes, exact diffs and the limited audit scope are included in the revision package.

**Revision 0.6 outcome:** required H recovery/evaluation and 2H-C1–C5 closed; technical H detail consolidated in WP §13.1; original failed attempt and historical results preserved; current priorities move to reviewed multi-question data, transfer-aware selection, early 2J comparisons, genuine Q sharing and Track S. All 16 review groups and P2.1–P2.3 remain captured. No repository, notebook or source result was changed by this documentation revision.


### Historical revision 0.5.2 — recovered-review coverage and restored conditional tasks

**RC: Review capture.** The historical `OpenDecision_Review_Followup_Traceability.md` maps sixteen recovered recommendation groups to existing tasks and test evidence. It uses the preserved R3 review and WP §11.7’s retained Laya/R4T interpretation. The shared page exposes no readable full transcript; follow-up coverage is recommendation-level, not a verbatim complete archive. The source snapshot and hashes are in the companion package.

**RP2 — Earlier roadmap task detail.** The v0.4 roadmap revision package contains explicit P2.1 verified-execution, P2.2 weight-precision and P2.3 teacher/student checkboxes. Their full conditional section is restored unchanged here after v0.5.1 retained only its summarized priorities. This restores tracking, not a new research priority or a promotion to completed work.

**HR — Historical recovery deliverable.** `OpenDecision_Phase2H_Recovery_Notes.md`, version `2h.1.2`, records the delivered Finish notebook, original-notebook repair and bounded local validation before the final run. Those notes alone did not establish GPU completion; HF now supplies the completed continuation. No notebook was changed during this documentation update.

The v0.5.2 review-capture revision left historical metrics and H-development tables unchanged while restoring explicit compact-comparator/P2 tasks. Version 0.6 now summarizes the closed H study here and retains its full development/final detail in the whitepaper. Neither documentation revision edits source notebooks, results or repository code.

**HF — Completed H Finish authority.** `OpenDecision_Phase2H_Finish_results / 20260919T040612625670Z__finish_2h_1_2`, version `2h.1.2`, both required workers completed. Use the linked summary/archive/recovery/lock in the H section and whitepaper E9; H remains the preserved original-attempt source. No retraining, criteria reset or post-final model reselection is reported.

**V06 — Version 0.6 checks.** Read-only saved-output and lineage checks (`audit_phase2h_finish.py`, `completed_h_audit.json`, `check_additional_lineage.py`, `additional_lineage.json`) plus document checks, source hashes and exact diffs in the companion package. This checks retained distributions/metrics, policies, parity, timings and artifact identities; it does not run Qwen, repair Colab, test the Rust repository or validate deployment. Policy intervals remain source-reported. See WP §3.2 and V2 for the exact scope.

**IJ — First 2I/2J workbench checkpoint.** `OpenDecision_Phase2IJ_results / 2ij_reviewed_multiquestion_v1`, version `2ij.1.0`; overall blocked. [Summary](https://drive.google.com/file/d/11X6HKs18bjjgGTDKbQwTe8PWK4nLXwfh/view), [report archive](https://drive.google.com/file/d/1bIyJfim0n-XsphCIEZu11cs8NEb90kO8/view), [mechanics output](https://drive.google.com/file/d/178mr-qS7CARw6qgfxGqZpghOvfjohmf_/view). WP E10/V3 records hashes, exact source scope and the duplicate/unrelated fixture finding. The source's 48 CPU tests are recorded, not rerun by this revision.

**I0 — Unsigned review intake.** [Protocol](https://drive.google.com/file/d/1zDdBv1JtGFTzMKYZshx-T8olFj5dVM9c/view), [review](https://drive.google.com/file/d/1_eIzpj9722xYDLBOmBb5lG1j0fng-rqO/view), [refreshed template](https://drive.google.com/file/d/1rk2jLwBnI_nsgJ9xXwqylOfTmC5GJsvk/view). Device/limits, actual review and matching signed manifests remain required. Cases/final annotations were not opened or approved during this documentation update.

**EXT1 — Harsha Gundala / `harshatheg/Qwen-2.5-1B-RLCD`.** Public Hugging Face model/repository documentation reviewed 20 September 2026. It describes Parallel Constrained Decoding with one shared prefill, KV-cache broadcasting, constrained candidate-token logit slicing, token-tree disambiguation and programmatic JSON assembly, and reports author-measured Apple Silicon M4 Max timings including 68–75 ms four-field cases, 270 ms for 28 fields and 89 ms for a 255-choice case. OpenDecision treats it as an inference-pattern and latency-floor reference, **not** evidence that TypeSafe's RLCD training method has been reproduced or that the returned softmax probabilities are calibrated.

**EXT2 — Nishaanth Reddy, “Jev-style models on DGX Spark,” More Than a Machine, 19 September 2026.** External comparison reviewed 20 September 2026. Relevant observations are the within-system Q=1→4 latency scaling (Jev 105.1→109.2 ms p50, Laya 16.4→29.1 ms, tuned Qwen3.5 167.0→665.1 ms), task-dependent quality ranking, separate probability-quality metrics, Doom state-representation sensitivity and campaign-to-campaign Jev API variation. Hosted Jev and local Spark absolute latencies are not directly comparable; OpenDecision uses the benchmark as motivation to measure its own Q-amortization, state sufficiency and repeatability.

**P3A — Completed Phase 3A Python systems/reference run.** `OpenDecision_Phase3A_BranchableState_BatchedQ`, run `20260920T024056Z`, schema `opendecision-phase3a-summary/v1`, result folder `Colab Notebooks/OpenDecision_Phase3A_results/20260920T024056Z`. Selected profile and E11 bundle hash were unchanged; no training or selection ran. The summary records semantic batched parity passed, high-K parity passed, same-process repeatability maximum probability delta 0.0 / zero argmax changes, workload-shape latency/memory measurements, `release_quality_claim=false`, and `rust_metal_parity_claim=false`. The corrected Python hybrid fan-out uses a deep copy plus cache-wide reindex/select rather than the unsupported generic repeat helper. Full interpretation is in WP §16.

**P3B — Completed Phase 3B Python backbone-reference run.** `research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z`, run `20260920T152206Z`, schema `opendecision-phase3b-backbone-parity-summary/v1`. Profile, bundle, and base revision remain unchanged. The checked-in export contains exact token fixtures, a 34-stage layer trace, 10 full candidate vectors, 3 continuation vectors, architecture and tensor indexes, and probability references. Its maximum fresh-feature delta is `4.9591064453125e-05`. Its maximum cached-continuation delta is `1.9073486328125e-05`. These values describe Python reference self-consistency, not native acceptance or Metal evidence.

**RUST3 — Rust Phase 3.3 CPU backbone checkpoint.** `Qwen35Backbone`, the `qwen35_parity_probe` example, and the `qwen35_full_parity` example provide the executable authority. On the named M4 Max host, the 34-stage trace, 10-candidate decision replay, and root/question/candidate cached-continuation trace produced the values recorded under 3.3. The implementation uses Candle `0.8.0` CPU/Accelerate matmuls plus explicit FP32 Qwen normalization, attention, DeltaNet, convolution, RoPE, and cache algebra. This checkpoint does not claim Metal, optimized throughput, backend-neutral branching, service integration, or release promotion.

**RUST4 — Rust Phase 3.4 branch-state checkpoint.** `crates/opendecision-backends/src/branch` defines the backend-neutral `BranchableState`/`BranchBatch` contract, and `crates/opendecision-backends/src/qwen35/backbone/branch.rs` binds `BackboneState` to it. The `qwen35_parity_probe` `branch` stage provides the executable authority on the named M4 Max host: profile-bound identity, structural and strict fingerprints, exact byte accounting, immutable-root fork, two-lane fan-out with independently advanced lanes, reorder/duplicate gather, and exact cached-versus-full continuation-state equality for the Phase 3B root/question/candidate fixtures. This checkpoint does not claim sequential or batched nested execution, Metal, service integration, or release promotion.

**RUST5 — Rust Phase 3.5 sequential nested checkpoint.** `crates/opendecision-backends/src/qwen35/backbone/nested.rs` implements the sequential nested graph behind the `SequentialNestedExecutor` trait (`run_sequential_nested` plus `Qwen35Backbone::evaluate_nested`), and the `qwen35_nested_parity` example provides the executable authority on the named M4 Max host: one prefill per fixture case, immutable question and candidate forks for all four Phase 3B questions and 10 candidates, maximum probability delta `4.5869e-06` with zero argmax and policy changes, root content identity against an independent prefill, exact replay determinism and sibling-order independence, and exact cached-versus-full feature and state equality (`0.0`). Hidden-vector deltas remain localization diagnostics. This checkpoint does not claim batched Q/K execution, Metal, service integration, or release promotion.

**PUB1 — Public OpenDecision state-first reference repository.** <https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>. This project-owned publication exposes the integration line publicly. Publication does not by itself establish release-quality model promotion, native parity, or TypeSafe RLCD reproduction.

**Revision 0.7.2 outcome:** Phase 3A and Phase 3B are closed for their Python reference scopes. Rust head/probability, exact renderer/tokenizer, CPU full-sequence backbone, and Qwen-specific cached-continuation parity pass, the Phase 3.4 amendment adds the backend-neutral branch-state contract with its native checkpoint, and the Phase 3.5 amendment adds sequential nested execution parity with its native checkpoint. The remaining queue starts with batched Q/K parity, then continues through adaptive scheduling and Mac performance. CPU native parity does not imply Metal or accelerated parity. Phase 3A.1 remains conditional cost-model work.

**Revision 0.7.1 outcome:** selected model identity remains unchanged. The forward implementation path is now explicit: native parity → backend-neutral branchable state → sequential nested parity → batched question execution → candidate batching → Q-amortization/high-K/repeatability measurements → production lifecycle. Reviewed release promotion remains a separate gate.

**Revision 0.6.1 outcome:** preparation and narrow mechanics are credited without closing 2I/2J/S; H remains closed. The exact blockers and a source-coverage correction are summarized here, with detailed measurements and reasoning in WP §14. All existing task checkboxes retain their state. The companion contains 34 saved-artifact checks, document integrity, diffs and hashes; no model inference, notebook repair, signed review, repository test or persistent Drive modification is claimed.
