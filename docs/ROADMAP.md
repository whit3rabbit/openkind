# openpick — Roadmap

> Jev-compatible, open-source decision-inference engine in Rust.
> Wire spec: <https://docs.typesafe.ai/api>

This is the single source of truth for what's done, what's next, and
what the open questions are. Each phase ships with passing tests at
HEAD.

---

## Phase 0 — wire contract ✅ DONE

**Goal:** lock the Jev schema before any server exists. Every TypeSafe
spec example must round-trip through our types without data loss.

- [x] `openpick-core` crate: `SystemRequest` / `SystemResponse` /
      `Answer` with `f64` (not `f32`) precision on `probabilities`,
      `score`, `noul`, `confidence`. `0.92f32` round-trips to
      `0.9200000166893005` — that was the trap.
- [x] `validate_request()` covering every documented validation rule.
- [x] **23** conformance tests in
      `crates/openpick-core/tests/conformance.rs` — one per Jev spec
      example fixture.
- [x] `gen-schemas` binary regenerates
      `crates/openpick-core/schemas/jev-v1-{request,response}.json`
      from the Rust types via `schemars`.
- [x] **8** example JSON fixtures committed in `examples/`
      (`01_noul.json` … `08_response_score.json`).

---

## Phase 1 — daemon + SDK compatibility ✅ DONE

**Goal:** a self-hostable server that the future `typesafe_sdk` Python
client can speak to without modification.

- [x] **HTTP** (axum 0.8): `POST /v1/systemone`, `GET /v1/models`,
      `GET /health`, `GET /metrics`.
- [x] **gRPC** (tonic 0.14): `openpick.system_one.SystemOne` with one
      `evaluate` RPC, request_id propagated via tonic interceptor.
- [x] `openpickd` daemon + `openpick` CLI (`serve`, `evaluate`,
      `inspect`, `version`).
- [x] `DecisionEngine` trait (`backend_id`, `model_metadata`,
      `evaluate`) + `EngineRegistry` alias → `Arc<dyn …>`.
- [x] `MockEngine` — deterministic, seeded RNG, slightly jittered.
- [x] **SDK compatibility surface** (the bits the future client
      actually depends on):
  - `x-typesafe-request-id` stamped on **every** response, including
    401s (middleware runs outermost).
  - Bearer auth opt-in via `OPENPICK_API_KEY`, constant-time compare.
  - `Retry-After` / `retry-after-ms` headers on 429 + 529.
  - Error envelope `{"error":{"code",message}}` mapped to the Python
    SDK's `TypeSafe{Authentication,RateLimit,BadRequest,…}Error`.
  - `/v1/models` returns `{"models":[{name,description,release_date}]}`.

### Test totals at HEAD

| Suite                                              | Tests |
|----------------------------------------------------|-------|
| `openpick-core` unit + conformance                 | 33    |
| `openpick-engine` unit (MockEngine, dispatch)      | 14    |
| `openpick-api` unit (middleware, error mapping)    | 22    |
| `openpick-api/tests/sdk_compat.rs`                 | 56    |
| `openpick-api/tests/grpc_roundtrip.rs`             | 8     |
| `openpick-cli` unit                                | 7     |
| `openpick-server` unit                             | 2     |
| `openpick-proto` unit                              | 2     |
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
the shared-state cache design (Phase 2D below).

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

### Phase 2C — Qwen model research: dynamic schemas, batching invariance & model scaling (IN PROGRESS)

> **Principle**: Phase 2 is model research in Python/Colab. We do not jump into cutting the Rust inference engine before the model architecture handles arbitrary dynamic schemas, candidate scoring, and verified batching behavior.

Phase 2B established that a frozen Qwen3.5-4B backbone + small linear head achieves 87.67% on fixed 3-class NLI. Phase 2C bridges the gap between fixed NLI classification and a true Jev-compatible decision engine:

#### 1. Batching Invariance Diagnostic & Multi-Seed Stabilization
- The Phase 2B audit found a **~1.27% probability shift** between standalone and padded batch execution.
- Implement a systematic 3-tier diagnostic to isolate root causes across hundreds of examples:
  1. *Duplicate in same-length batch*: isolate kernel summation order without padding.
  2. *Identical sample with varied padding*: isolate attention mask and positional encoding handling.
  3. *Heterogeneous production batches*: isolate real-world serving batch interactions.
- Run multi-seed evaluations across `last_linear` and `last_mlp` to quantify training-seed variance (test accuracy intervals span 85.2%–90.2%).

#### 2. Dynamic Candidate Scoring & Arbitrary Schemas (The True Jev Workload)
- Fixed 3-class NLI (`entailment`, `neutral`, `contradiction`) does not support callers specifying arbitrary business categories.
- Implement dynamic candidate encoding for all three Jev question types:
  - **`Choice`**: dynamically encode caller-supplied alternatives ($K \le 255$) with natural language descriptions; produce normalized probability distribution.
  - **`Noul`**: binary probability ($K=2$) with explicit True/False criteria.
  - **`Score`**: ordinal rubric levels ($\ge 2$ ordered levels) with calibrated probability distributions.
- Test candidate permutation invariance: verify that answer distributions do not depend on the order of presented choices.
- Implement explicit handling of unanswerable / insufficient evidence / abstention cases.

#### 3. Controlled LoRA Comparison
- Phase 2B had LoRA disabled.
- Execute an identical-split, controlled evaluation with LoRA adapters enabled on Q/K/V.
- Quantify accuracy, NLL, Brier, and ECE deltas against the frozen baseline to definitively decide whether fine-tuning is required before committing to the serving architecture.

#### 4. Model Scaling & Memory Reduction (2B vs 4B vs 9B, Quantization)
- The Phase 2B resident allocation was **~7.85 GiB in BF16**.
- Evaluate smaller backbones: `Qwen/Qwen3.5-2B-Base` and `Qwen3.5-0.8B-Base`.
- Answer the core research question: *Can a smaller 2B model or quantized model (GGUF / AWQ / INT8) match the decision accuracy while fitting within consumer VRAM (< 4 GiB)?*
- Explore teacher-student distillation using Qwen 9B or frontier models as teachers.

#### 5. Synthetic Decision Benchmark Datasets
- MultiNLI may be contaminated in Qwen pretraining; split separation does not guarantee out-of-domain generalization.
- Generate domain-specific decision evaluation sets (security triage, customer routing, code review) with controlled ambiguity and adversarial cases using strong LLMs (Claude/GPT).

---

### Phase 2D — Shared-State Prefill & KV-Cache Branching (Python/PyTorch)

Followup from Phase 2A finding 2 (GatedDeltaNet + Attention hybrid):

> Qwen3.5 may already produce a state representation that is reusable
> across many questions for the same state.

Before writing a complex scheduler in Rust, validate the core Jev execution model in Python:

```text
State string (e.g. 50k character document)
   │
tokenize
   │
Qwen3.5 backbone (prefill once, cache hidden/KV state)
   │
cached state representation (2560-dim / recurrent state)
   │
   ├─────────── branch 1: Question A + candidates A ──────────► answer A
   ├─────────── branch 2: Question B + candidates B ──────────► answer B
   └─────────── branch 3: Question C + candidates C ──────────► answer C
```

- Measure memory and latency savings of prefilling shared state once vs re-evaluating per question.
- Empirically verify zero-interference isolation: ensure Question A and Question B do not attend to each other.

---

## Phase 3 — Rust Engine & Production Backends (PLANNED)

Once the Qwen decision architecture, dynamic candidate scoring, quantization, and KV-cache branching are validated in Python (Phases 2A–2D), port the complete inference pipeline to Rust:

- [ ] `openpick-runtime` — device discovery, VRAM accounting, worker pools, shared state cache.
- [ ] `openpick-backends` — candle (GGUF), ONNX runtime, optional remote-provider passthrough, all behind `DecisionEngine`.
- [ ] Parity test suite against Python reference vectors (`golden_head_inputs.npz`).
- [ ] Request scheduler — batch incoming `system_one` calls, execute branched question evaluations against cached state.
- [ ] Cancel tokens / timeout propagation from gRPC deadline headers.
- [ ] A real Python client (`openpick` or `typesafe_sdk`) exercising a self-hosted `openpickd` daemon end-to-end.
- [ ] Eval harness: feed labeled datasets through `openpickd`, capture side-by-side comparison with the hosted TypeSafe API.

---

## Status snapshot

| Phase | Description | Status | Tests / Milestone |
|---|---|---|---|
| **Phase 0** | Wire contract & core types | done | 33 tests |
| **Phase 1** | Daemon, HTTP/gRPC transports & SDK compat | done | 91 tests (124 total) |
| **Phase 2A** | Python Qwen3.5-4B exploration & parameter audit | done | Colab probe |
| **Phase 2B** | Fixed NLI head benchmark & baseline readout | done | Run 20260917T205849Z (87.67% acc) |
| **Phase 2C** | Qwen model research: dynamic schemas, batching & scaling | in progress / planned | Python / Colab |
| **Phase 2D** | Shared-state prefill & KV-cache branching | planned | Python / Colab |
| **Phase 3** | Rust engine, runtime & production backends | planned | Gated on Phase 2 |

`cargo test --workspace`: **124 tests passing, 0 failing.**

---

## Conventions for agents

(Same as `docs/AGENTS.md` — duplicated here so this file is
self-contained for a roadmap reader.)

- **Wire types live in `openpick-core`.** Any change there is a
  breaking change for every future SDK caller. Bump
  `JevRequest::SCHEMA_VERSION`, add a round-trip test, regenerate
  schemas, update `docs/ARCHITECTURE.md`.
- **`sdk_compat.rs` is the contract.** If your change would break
  one of those 37 tests, you are touching the wire contract.
- **New backend = new `DecisionEngine`.** Wire it in
  `openpick-server/src/main.rs` behind a CLI flag. Add at least
  one round-trip test in `openpick-engine`. Add a `sdk_compat.rs`
  case that exercises the alias end-to-end via `/v1/systemone`.
- **Phase 2 owns model research in Python; Phase 3 owns Rust engine code.**
  Do not start cutting the runtime/backends crates until Phase 2
  validates dynamic schemas, LoRA, and cache branching in Python.

## Quick reference

```bash
# Build & test
cargo build --workspace
cargo test --workspace          # 82 tests

# Regenerate JSON Schema
cargo run -p openpick-gen-schemas

# Run the daemon (mock engine, dev mode, no auth)
cargo run -p openpickd -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest

# Verify a request against a running daemon
curl -sS -H 'content-type: application/json' \
     -X POST http://127.0.0.1:18080/v1/systemone \
     -d @examples/04_mixed.json | jq
```