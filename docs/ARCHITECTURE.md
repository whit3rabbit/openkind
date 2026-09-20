# opendecision — Architecture

> An open-source decision-inference engine in Rust targeting the Jev wire contract, with independently designed model and runtime internals.
>
> **Revision 0.8.0 · 20 September 2026 · Native CPU reference engine through safe adaptive scheduling, state/scheduler high-K stress, and direct service registration. Model-backed high-K, fresh-process repeatability, Metal, and load/soak remain open.**
>
> Wire spec: https://docs.typesafe.ai/api
> Reference client SDK target: https://docs.typesafe.ai/sdk/python/api

## Overview

`opendecision` accepts structured decision questions (Noul, Choice, Score) over a shared state and returns typed answers and probability distributions rather than generated answer prose. The surrounding service provides the tracing, metrics, authentication, admission, and lifecycle surface required by a public SDK.

The repository separates the **wire contract**, **decision/model execution**, and **transport/runtime** layers so that a model backend can change without changing the public request/response schema. The project targets Jev-compatible wire behavior where explicitly supported; it does **not** claim to reproduce Jev's private neural architecture or RLCD training procedure.

The current model-integration reference is profile `a047d6802c3f06f085b8`: `Qwen/Qwen3.5-4B-Base`, frozen backbone, **state-first rendering**, and **score-summary rejection**. The exploratory `2ij.2.0` study selected and exported this profile before final evaluation, and the exported bundle passed a clean Python reload/head-algebra parity check. Phase 3A kept that profile immutable and validated the Python reference mechanics for full-hybrid-state branching, batched question/candidate execution, high-K systems stress, and same-process repeatability. Phase 3B kept the same profile, bundle, and base revision and exported exact segmented tokens, a 34-stage embedding-to-final-norm trace, 10 candidate features, and 3 continuation vectors. Rust Phase 3.3 now reproduces the correctness-first CPU backbone and Qwen-specific cached continuation on the named M4 Max. That makes the profile the fixed Phase 3 implementation target. It remains a **provisional integration profile**, not a release-quality model: independent review, natural-data confirmation, explicit promotion limits, optimized execution, and Metal evidence remain open.

Rust now reproduces the selected head/probability algebra and all four exported token records. It loads and validates the Phase 3B architecture and 47-vector reference bundle, verifies both immutable checkpoint shards, and executes the complete 32-layer Qwen text backbone plus final RMSNorm through Candle's CPU backend in FP32. All 34 stage diagnostics are localized, all 10 candidate sequences pass the declared probability/argmax/policy gate, and Qwen-specific cached continuation carries attention KV, DeltaNet recurrent state, convolution state, and absolute position without mutating its root. This is correctness-first CPU evidence, not Metal, production-service, or release-quality evidence.

A public OpenDecision reference repository for this integration line is published at <https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>. Publication improves inspectability and handoff; it does **not** change the release-quality or Rust/Metal parity boundary.

The immediate architecture objective is therefore no longer “choose a model.” It is:

```text
completed Python Phase 3A branch/batch reference + Phase 3B backbone reference
  → completed Rust head/probability + exact-token + CPU backbone parity
  → completed Qwen-specific full-hybrid cached continuation
  → completed backend-neutral Rust BranchableState with fork/gather/profile identity
  → completed sequential state→question→candidate parity
  → completed batched question and candidate execution
  → completed adaptive workload scheduler measured on the named Mac
  → completed state/scheduler high-K admission stress + cache/snapshot persistence contracts
  → completed direct native DecisionEngine registration + explicit none mapping
  → model-backed high-K / fresh-process repeatability
  → production load/soak lifecycle
```

Model promotion and implementation equivalence remain separate decisions.

---

## Current status boundary

The architecture document distinguishes implemented/reported repository behavior from research artifacts and planned work:

- **Phase 0 / Phase 1:** wire types, schemas, mock engine, HTTP/gRPC surfaces, CLI and SDK-compatibility fixtures are implemented and covered by the repository verification battery.
- **Phase 2H:** required criteria/rejection-transfer study is complete through the preserved `2h.1.2` continuation. The old failed attempt remains historical evidence.
- **Phase 2I:** bounded state-first multi-question mechanics and Q=1/Q=4 semantic sharing are measured; independently reviewed/natural-data confirmation remains open.
- **Phase 2J:** the exploratory model-selection screen is complete; the Qwen4B state-first/score-summary profile is the provisional integration target. Release promotion remains open.
- **Track S:** the service path now supports direct `Qwen35DecisionEngine` registration through `EngineRegistry`; the Python implementation remains a differential-test oracle, not a deployment bridge.
- **Phase 3A (Python reference):** run `20260920T024056Z` completed notebook scope with the selected profile unchanged; semantic batched parity passed, high-K parity passed, and the recorded same-process repeatability delta was zero. The run is systems/reference evidence, not Rust/Metal parity or release certification.
- **Phase 3B (Python reference):** run `20260920T152206Z` completed notebook scope with no training, model selection, model modification, or bundle change. It exports 4 token records and 47 FP32 vectors for layer, candidate, and continuation localization. Its hidden-vector deltas are diagnostics, not Rust acceptance tolerances.
- **Phase 3 (Rust/native):** head/probability, exact tokenizer/state-first token, full CPU decoder/final-normalization, Qwen-specific cached-continuation, backend-neutral branch-state, sequential nested execution, batched Q/K, and the measured adaptive scheduler pass against the frozen fixtures. State/scheduler high-K stress, process-peak admission, tenant-isolated cache semantics, direct service registration, and explicit semantic-none mapping are implemented. Model-backed high-K execution, fresh-process replay, Metal, load/soak, and release promotion remain open.

The roadmap is the task/status authority; the whitepaper is the evidence/interpretation authority. This file defines the intended software and execution architecture.

---

## Workspace layout

```text
opendecision/
├── Cargo.toml                    # workspace manifest + shared deps
├── crates/
│   ├── opendecision-core/        # wire types (request, response, errors)
│   ├── opendecision-engine/      # DecisionEngine + profile/execution boundary
│   ├── opendecision-api/         # HTTP (axum) + gRPC (tonic)
│   ├── opendecision-server/      # opendecisiond binary
│   ├── opendecision-cli/         # opendecision binary
│   ├── opendecision-runtime/     # device/scheduler/state/cache lifecycle
│   ├── opendecision-backends/    # supported native model drivers
│   └── opendecision-gen-schemas/ # JSON Schema codegen
├── proto/opendecision.proto      # gRPC service definition
├── examples/                     # wire-format fixtures
├── docs/ARCHITECTURE.md          # this file
└── crates/opendecision-core/schemas/
```

### Layering

```text
┌──────────────────────────────────────────────────────────────┐
│ opendecisiond / opendecision CLI                            │
├──────────────────────────────────────────────────────────────┤
│ opendecision-api                                            │
│   HTTP axum + gRPC tonic                                    │
│   request IDs · auth · validation · errors · tracing        │
├──────────────────────────────────────────────────────────────┤
│ opendecision-engine                                         │
│   DecisionEngine · EngineRegistry                           │
│   DecisionSpecification · ModelExecutionProfile             │
│   probability / rejection / policy contracts                │
├──────────────────────────────────────────────────────────────┤
│ opendecision-runtime                                        │
│   admission · scheduling · device/runtime lifecycle          │
│   BranchableState · batching · persistent state reuse        │
├──────────────────────────────────────────────────────────────┤
│ opendecision-backends                                       │
│   selected supported Qwen/runtime implementation(s)          │
│   profile-specific hidden-state and branch operations        │
├──────────────────────────────────────────────────────────────┤
│ opendecision-core                                           │
│   SystemRequest · SystemResponse · Answer · validation       │
│   generated schemas                                          │
└──────────────────────────────────────────────────────────────┘
```

The public dependency direction remains one-way. `core` owns wire types; `engine` owns semantic/model contracts; `runtime` and `backends` implement execution; `api` exposes the service; `server/cli` compose those pieces. Backends must not redefine public probability semantics merely because their internal execution graph differs.

---

## Crate responsibilities

### `opendecision-core`

The single source of truth for supported request/response wire types across HTTP and gRPC.

Public types include:

- `SystemRequest`, `SystemResponse`
- `Question` (Noul, Choice, Score variants), `Answer`
- `State`, `Instructions`
- `ValidationError` + `validate_request(req)`
- `ModelInfo`, `ModelsResponse`

Wire floating-point values such as `probabilities`, `score`, `noul`, and `confidence` remain `f64`. Do not replace wire precision with an implementation backend's internal dtype.

Wire compatibility and model capability are separate. A syntactically valid request can still be unsupported by a specific profile because of primitive, context-length, Q/K, memory, or execution-mode limits.

### `opendecision-engine`

Anything callable from `/v1/systemone` implements the public engine boundary:

```rust
#[async_trait]
pub trait DecisionEngine: Send + Sync + 'static {
    fn backend_id(&self) -> &str;
    fn model_metadata(&self) -> opendecision_core::ModelInfo;

    async fn evaluate(
        &self,
        req: SystemRequest,
    ) -> Result<SystemResponse, EngineError>;
}
```

`EngineRegistry` maps aliases to `Arc<dyn DecisionEngine>`.

Behind that public boundary, execution is governed by four versioned contracts:

1. **Decision Specification** — state semantics, question instructions, candidate/rubric descriptions, semantic-none rules, insufficient-evidence behavior, truncation, and supported primitive semantics.
2. **Model / Execution Profile** — checkpoint revision, adapter, tokenizer, renderer, normalization, heads, rejection model, calibration, policy mapping, arithmetic mode, kernels, and allowed execution strategies.
3. **Backend Capabilities** — supported primitives, context/Q/K limits, branch/batch operations, precision modes, state-storage behavior, and device-memory requirements.
4. **Evaluation Context** — tenant identity, admission/work budget, deadline/cancellation, tracing identifiers, and any request-scoped scheduling metadata.

The **selected Phase 3 profile** is currently:

```text
profile_id: a047d6802c3f06f085b8
backbone:   Qwen/Qwen3.5-4B-Base
revision:   1001bb4d826a52d1f399e183466143f4da7b741b
weights:    frozen
renderer:   state-first
rejection:  score-summary
bundle:     4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332
status:     provisional integration target
```

`ModelExecutionProfile` records this identity and its reference bundle, renderer,
head, rejection, calibration, policy, and numerical tolerances without changing
the public `DecisionEngine` trait.

`score-summary` is part of this profile; it must **not** be hard-coded into the engine architecture. The exploratory screen also produced semantic-feature rejection variants, and future profiles may use different applicability/rejection heads. Rejection is therefore profile-owned behavior behind a stable probability contract.

### `opendecision-runtime`

Owns device/runtime lifecycle and execution mechanics that are independent of one particular neural head:

- device discovery and backend selection;
- bounded request admission and worker pools;
- request scheduling across Q questions and K candidates;
- `BranchableState` lifecycle;
- exact-prefix/persistent state reuse where supported;
- memory accounting for retained and transient branch state;
- cancellation, deadlines, TTL, eviction, and active-reader lifetime;
- repeatability/profile identity telemetry;
- queue-inclusive performance measurement.

The runtime must not assume that reusable model state is an ordinary Transformer KV cache. Qwen3.5 requires a hybrid continuation state.

### `opendecision-backends`

Contains only backends that can satisfy the selected profile's required capabilities. Candle now supplies the correctness-first CPU tensor path. Candle Metal, GGUF/llama.cpp, ONNX, MLX-backed bridges, or other production implementations remain candidates, not promises.

A backend is promotable only if it can reproduce the required tokenizer/rendering and hidden-state path and expose enough control over continuation state to support the selected profile's parity ladder. If a backend cannot expose recurrent/convolution state or the required hidden representations, it must report the unsupported capability rather than silently substitute a different decision function.

The current `qwen35` module implements the deterministic contracts through the
complete correctness-first CPU backbone and Qwen-specific continuation stage:

- fail-closed selected-profile and head-artifact loading;
- f64 normalization, projection, rejection, calibration, stable softmax, and
  policy evaluation;
- digest-locked offline tokenizer loading and exact segmented state-first IDs;
- Phase 3B architecture, 426-entry parameter inventory, tensor index, trace,
  continuation metadata, and golden-vector validation;
- max-absolute, RMS, and cosine comparisons across all 34 native stages;
- exact two-shard checkpoint validation, BF16 loading, FP32 host execution,
  24 DeltaNet blocks, 8 full-attention blocks, and final RMSNorm;
- 10-candidate probability/argmax/policy replay;
- immutable Qwen-specific continuation state with attention KV, recurrent,
  convolution, position, and byte-accounting components;
- profile/model/tokenizer/renderer/arithmetic state identity, distinct scheduling
  and strict content fingerprint types, exact tensor-payload accounting,
  immutable-root fork, batched fork, and gather/select through the
  backend-neutral `opendecision-runtime::branch` contracts;
- sequential nested and breadth-first Q/K execution, with the current CPU
  backend explicitly advertising per-lane rather than vectorized forward;
- scheduler admission over tensor payload plus observed process peak, scratch,
  allocator headroom, and backend lane limits;
- `Qwen35DecisionEngine` with bounded execution/queue admission and direct
  `EngineRegistry` registration;
- explicit Choice semantic-none mapping through caller-supplied `__none__`,
  with none mass preserved and confidence derived from normalized entropy.

It does not yet implement vectorized batch-forward kernels, Metal execution,
fresh-process replay proof, or production load/soak validation.

---

## Track S — direct native service integration

The service now connects the existing Rust boundary directly to the native backend:

```text
HTTP / gRPC client
  │
  ▼
opendecisiond
  │  request ID · auth · validation · admission · deadline
  ▼
Qwen35DecisionEngine : DecisionEngine
  │  bounded queue · explicit offline profile paths
  ▼
Native tokenizer + Qwen backbone + fitted head
  │  selected pinned model profile
  │  real probability/rejection computation
  ▼
validated typed SystemResponse
```

The daemon still defaults to mock aliases. Native aliases require explicit bundle, checkpoint, and tokenizer paths and never download artifacts. Queue admission, unsupported-input errors, and direct typed responses are implemented. Cancellation after a blocking model call starts, recovery, service load/soak, and deployment promotion remain open. The Python reference worker is retained only as an optional differential oracle.

---

## State-first execution contract

State-first rendering is part of the selected learned profile, not merely a caching trick.

The selected architecture intentionally keeps the shared state representation independent of which questions happen to be submitted in the same request:

```text
stable format + state
        │
        ▼
immutable shared state root
        │
        ├── question A suffix → isolated question-A state
        │       ├── candidate A1 suffix → feature/score
        │       └── candidate A2 suffix → feature/score
        │
        └── question B suffix → isolated question-B state
                ├── candidate B1 suffix → feature/score
                └── candidate B2 suffix → feature/score
```

This is deliberately different from a schema-first parallel-constrained-decoding pattern in which the complete semantic field/question catalog appears before the state and is therefore part of the shared cached representation. That design can be efficient, but adding/reordering questions can alter the prefix used by every field. OpenDecision borrows **breadth-first branch batching** from public PCD implementations while retaining a **state-first isolation contract**.

### Why `KVCache` is not the right abstraction

Qwen3.5 mixes full-attention blocks with recurrent DeltaNet blocks. A valid continuation branch includes at least:

- full-attention key/value state;
- DeltaNet recurrent state;
- convolution state;
- logical/token position state;
- execution/profile identity required to prevent cross-profile reuse.

A branch that clones only attention KV can still share or corrupt mutable recurrent state. Attention masks do not make recurrent streams independent.

---

## `BranchableState` target abstraction

Phase 3 exposes a backend-neutral continuation-state abstraction rather than Qwen-specific cache tensors through the engine boundary.

Phase 3A supplies the working Python reference for the required semantics. The first notebook draft attempted Transformers 5.17.0's generic `Cache.batch_repeat_interleave()`, which fails on Qwen3.5 because its `LinearAttentionLayer` does not implement that generic helper. The corrected reference instead deep-copies the complete hybrid cache and calls cache-wide `reorder_cache(indices)`. Repeated indices such as `[0,0,0,0]` fan one immutable root into multiple lanes; selected indices gather lanes back out. This reaches both ordinary attention KV layers and the linear-attention layers carrying recurrent and convolution state. The environment remains pinned rather than monkey-patching only the KV portion of the cache.

The Python contract validates immutable-root fan-out/select before the semantic benchmark. Rust does not copy the Python object model, but it reproduces the **same complete-state semantics** and fixture outputs.

The Phase 3.4 Rust contract is owned by `crates/opendecision-runtime/src/branch` and bound to `BackboneState` in `crates/opendecision-backends/src/qwen35/backbone/branch.rs`:

```rust
pub trait BranchableState: Send + Sync {
    type Batch: BranchBatch<State = Self>;

    fn profile_id(&self) -> &ProfileId;
    fn position(&self) -> usize;
    fn tensor_storage_bytes(&self) -> usize;
    fn scheduling_fingerprint(&self) -> SchedulingFingerprint;

    fn fork_one(&self) -> Result<Self, StateError>
    where
        Self: Sized;

    fn fork_batch(&self, lanes: usize) -> Result<Self::Batch, StateError>
    where
        Self: Sized;
}

pub trait BranchBatch {
    type State: BranchableState<Batch = Self>;

    fn lanes(&self) -> usize;
    fn tensor_storage_bytes(&self) -> usize;
    fn select(&self, index: usize) -> Result<Self::State, StateError>;
    fn gather(&self, indices: &[usize]) -> Result<Self, StateError>
    where
        Self: Sized;
}
```

The semantic requirements are more important than the exact trait syntax:

1. **Immutable-root semantics** — branching must not mutate a reusable root.
2. **Complete-state isolation** — all mutable continuation components are isolated.
3. **Single and batched fork** — a backend can create one continuation or a vectorized set of lanes.
4. **Split/gather/rebatch support where valid** — batching is an execution optimization, not a semantic merge.
5. **Qualified memory accounting** — `tensor_storage_bytes()` covers attention KV, recurrent, and convolution payload exactly. Admission separately adds observed process memory, forward scratch, allocator headroom, and fan-out peaks. The direct engine refreshes peak RSS after model load and before each evaluation, then apportions remaining headroom across the configured concurrency limit.
6. **Stable identity** — state cannot be reused across incompatible model, renderer, adapter, arithmetic, tokenizer, or position contracts.
7. **Backend neutrality** — future encoders/query models can implement a different reusable-state representation without pretending to be a Transformer KV cache.

The Rust implementation uses two compile-time-distinct fingerprint types.
`SchedulingFingerprint` hashes identity, lineage root, position, and tensor
layout only, so scheduling can call it without reading tens of MiB of tensor
bytes. It is process-local and cannot be passed where a persistent content key
is required. `ContentFingerprint` hashes exact little-endian tensor bytes,
identity, and position; it is the cross-process replay/integrity identity used
by fixture gates and tenant-scoped persistent-state keys, not per-fork scheduling.

---

## Breadth-first Q/K execution

The correctness reference is **sequential nested execution**, now implemented natively in `crates/opendecision-backends/src/qwen35/backbone/nested.rs` (`run_sequential_nested` / `Qwen35Backbone::evaluate_nested`, Phase 3.5): one immutable state prefill, then `fork question → advance question → fork candidate → advance candidate` through `BranchableState`, with fail-closed position and immutability verification and exact `repeated_full` agreement. The breadth-first lane topology is also native (`crates/opendecision-backends/src/qwen35/backbone/batched.rs`, `run_batched_nested`, Phases 3.6/3.7): `fork_batch` question lanes from the same root, then a `fork_batch` candidate fan-out per question state, proven exactly equal to the sequential baseline. This is state-semantic batching, not compute-vectorized forward. `BackendCapabilities` keeps those claims separate, and the current CPU scheduler defaults to `NestedSequential`. `NestedBatched` is eligible only when a backend advertises vectorized question and candidate forward support within explicit lane limits.

### Reference graph

```text
state_tokens = render_state(state)
S0 = prefill_state(state_tokens)

for question in questions:
    Sq = fork(S0)
    Sq = run_question_suffix(Sq, question)

    for candidate in question.candidates:
        Sc = fork(Sq)
        feature[candidate] = run_candidate_suffix(Sc, candidate)

    scores = candidate_head(feature[*])
    rejection = profile.rejection(scores, feature[*])
    distribution = profile.normalize(scores, rejection)
    decision = profile.policy(distribution)
```

### Optimized graph

```text
S0 = prefill_state(state)                       # once

Q_batch = S0.fork_batch(Q)                      # isolated Q lanes
Q_batch = run_question_suffixes(Q_batch)        # breadth-first
question_states = split_or_view(Q_batch)

for compatible question bucket:
    C_batch = fork_candidates(question_states)  # K lanes/question
    candidate_features = run_candidate_suffixes(C_batch)

scores
  → profile-specific rejection
  → calibrated distribution
  → application policy
  → wire adapter
```

Question batching and candidate batching must remain independently switchable. If an optimized path changes probabilities or policy outputs, the system must be able to localize whether the difference entered at state prefill, Q batching, K batching, rejection, arithmetic, or the wire adapter.

### Scheduler modes

Every supported profile should have explicit execution modes rather than a single opaque “fast” path:

- `repeated_full` — Q complete independent evaluations; slow but conceptually simple baseline.
- `nested_sequential` — state prefilled once, questions/candidates branched sequentially; primary sharing correctness reference.
- `nested_batched` — state prefilled once, compatible Q and K suffix work vectorized; target high-throughput path.

The scheduler may choose among these only within capabilities already accepted for the profile. The choice can depend on state length, Q, K, suffix-length distribution, available memory, and arithmetic mode.

---

## Q-amortization is a first-class systems metric

Single-question latency is insufficient for a Jev-style multi-question engine. Phase 3 must measure how total request time grows with independent question count.

For the same pinned profile and semantic workload, report at minimum:

- complete-request p50 / p95 latency;
- `T(Q) / T(1)`;
- marginal milliseconds per additional question;
- questions/second and decisions/second;
- actual model/forward invocation count;
- state-prefill fraction of total model time;
- branch-state bytes, peak allocation, and resident memory;
- cold-state versus warm-state reuse;
- queue-inclusive service latency separately from model-only timings.

Primary comparison:

```text
                    Q=1    Q=4    Q=16
repeated_full        x      x       x
nested_sequential    x      x       x
nested_batched       x      x       x
```

The external DGX Spark comparison is useful because its within-system slope differs sharply across implementations: the published campaign reports Jev 1.13 at roughly 105.1 → 109.2 ms p50 from Q=1 to Q=4, while a sequential tuned-Qwen wrapper reports roughly 167.0 → 665.1 ms. These absolute values are not OpenDecision targets and are not directly comparable across hosted/local environments.

### Phase 3A measured crossover

Phase 3A shows why the scheduler cannot equate “shared state” with “always faster.” Across the three real semantic smoke states, median-of-case timings were approximately:

| Execution mode | Q=1 median | Q=4 median | `T(4)/T(1)` |
|---|---:|---:|---:|
| `repeated_full` | 265.0 ms | 869.3 ms | 3.28× |
| `nested_sequential_cold` | 456.0 ms | 1,260.9 ms | 2.77× |
| `nested_batched_cold` | 380.8 ms | 1,058.4 ms | 2.78× |
| `nested_batched_warm` | 288.4 ms | 962.4 ms | 3.34× |

On these short semantic Q=4 requests, warm nested batching was about **10.7% slower** than repeated-full execution. Phase 3A therefore does **not** establish Jev-like near-flat Q scaling for short semantic requests.

The synthetic mechanics grid demonstrates the complementary result: sharing becomes strongly advantageous as the common state dominates request cost.

| Mechanics cell | Repeated full | Nested batched cold | Nested batched warm | Warm speedup vs repeated |
|---|---:|---:|---:|---:|
| L=64, Q=4, K=4 | 1,375.4 ms | 597.9 ms | 514.9 ms | 2.67× |
| L=256, Q=4, K=4 | 2,934.3 ms | 677.0 ms | 507.6 ms | 5.78× |
| L=1024, Q=4, K=4 | 9,448.1 ms | 1,101.2 ms | 516.5 ms | 18.29× |
| L=1024, Q=16, K=2 | 18,928.6 ms | 2,506.3 ms | 1,922.0 ms | 9.85× |

At L=1024/Q=16/K=2, batched-cold peak allocation is about **18.67 GiB**, versus about **16.42 GiB** for repeated-full. Latency and branch memory therefore belong in the same scheduling decision.

**Architecture consequence:** retain `repeated_full`, `nested_sequential`, and `nested_batched` as explicit strategies. The runtime planner should choose among them using measured state length, Q, K, suffix-length buckets, available memory, precision/backend identity, and cache warmth. A Phase 3A.1 Colab cost-model study is conditional: run it if early Rust profiling cannot determine stable crossover rules; it should not block head/tokenizer/backbone parity.

---

## Candidate cardinality and constrained-token baselines

High K and high Q are different scaling problems.

The selected OpenDecision candidate scorer uses semantic candidate descriptions and candidate-conditioned features. A public Qwen parallel-constrained-decoding implementation demonstrates a much cheaper alternative for finite answer tokens: prefill once, broadcast state, slice logits to allowed tokens, and serialize the result in host code. OpenDecision should use that idea as an **efficient baseline / latency floor**, not silently replace the selected semantic scorer.

Phase 3 systems stress should cover K = 32, 64, 128, and 255 where supported, measuring:

- candidate-branch memory;
- model invocations and suffix grouping;
- scheduler/batch saturation;
- latency and throughput;
- behavior under mixed Q/K request shapes.

Semantic quality at high K requires a separate reviewed evaluation. Surviving a 255-choice systems test does not establish 255-way decision accuracy or calibration.

---

## Rejection, evidence sufficiency, and application review

The runtime must not collapse several different “do not answer” concepts into one bit:

1. **Offered-option omission** — a valid semantic answer exists but is not among the offered candidates.
2. **Out of domain / unsupported task** — the question lies outside the model's supported scope.
3. **Insufficient evidence** — the requested judgment is meaningful, but the visible state does not support a determinate answer.
4. **Unobservable state** — required facts are absent because the state representation is incomplete/lossy.
5. **Application review** — the model may have a valid semantic answer, but policy declines automation because risk/cost exceeds the operating threshold.

These categories are evaluation and decision-semantics concerns first; they must not be inserted into the public wire probability vector without an explicit versioned contract.

The selected profile currently uses **score-summary rejection**. Future profiles may use semantic-feature applicability or another rejection model, so the engine should expose a profile-specific rejection interface rather than a single universal equation such as `P(none) = 1 - a` baked into runtime code.

---

## Wire & API contract resolutions

### Choice `none` / rejection mass

OpenDecision must not silently append an unrequested `none` choice or drop/renormalize internally modeled mass while calling the resulting probabilities unchanged unconditional probabilities.

Supported mappings must be explicit and versioned, for example:

- caller supplies a semantic `other` / `none` option;
- a native OpenDecision response extension represents semantic-none separately;
- an application policy returns a review/routing action outside the semantic probability vector.

Application review is not the same as semantic none.

### Candidate identifiers and semantic descriptions

Machine IDs and semantic meaning remain separate. IDs are routing/stability keys; model input comes from the declared names/descriptions/criteria according to the selected profile. Opaque ID renaming must not alter model meaning.

### Confidence

Do not substitute top probability for a separately defined compatibility `confidence` statistic. Any compatibility adapter must implement the documented meaning explicitly and version that mapping.

---

## HTTP and gRPC service boundary

### `opendecision-api`

Two transports share the same validated engine behavior:

- **HTTP** (`src/http.rs`)
  - `POST /v1/systemone`
  - `POST /v1/system_one` compatibility alias where retained
  - `GET /v1/models`
  - `GET /health`
  - `GET /metrics`
- **gRPC** (`src/grpc.rs`)
  - one evaluate RPC through the same engine registry and semantic contract.

Reported middleware behavior includes request-ID propagation, bearer authentication where enabled, validation, tracing, and a common error taxonomy. Request IDs must remain available on failures so service/model traces can be joined.

Transport success is not model success. `/v1/models` should advertise the exact supported profile/capabilities rather than imply arbitrary primitive, Q/K, context, or precision support.

---

## Phase 3 parity ladder

The exported selected bundle is the model/probability reference contract. Phase 3A adds the Python branch/batch execution reference, and Phase 3B adds exact token, layer, candidate, and continuation diagnostics. Native work advances in this order:

1. **Head/probability algebra, complete:** Rust matches the exported fixtures. It reproduces normalization, projection, rejection, calibration, stable softmax, and policy semantics.
2. **Exact tokenizer + state-first token rendering, complete:** all four exported root, question, candidate-suffix, and full-sequence ID records match exactly. The implementation rejects overlength inputs rather than silently truncating them.
3. **Full Qwen3.5 CPU backbone parity, complete for frozen fixtures:** exact embedding, all 32 decoder blocks, final RMSNorm, 34-stage diagnostics, 10 candidate features, and probability/decision replay pass. Qwen-specific cached continuation also matches native full-sequence output exactly for the exported branch.
4. **`BranchableState`, complete for the CPU path:** the backend-neutral contract lives in `opendecision-runtime`; Qwen state carries profile/model/tokenizer/renderer/arithmetic identity, lineage, explicit position, attention KV, recurrent and convolution tensors, clone isolation, exact tensor-payload accounting, distinct scheduling/content fingerprints, single and batched fork, and gather/select. Metal remains open.
5. **Sequential nested parity, complete for the CPU path:** `state → question → candidate` against the Python reference with exact `repeated_full` agreement, root immutability, and replay determinism.
6. **Batched question layer, complete for the CPU path:** Q breadth-first `fork_batch` execution with lane isolation and exact sequential-baseline parity.
7. **Batched candidate layer, complete for the CPU path:** K breadth-first fan-out per question state with permutation/rejection parity; vectorized suffix kernels remain open.
8. **Adaptive scheduler / Q-amortization, measured for the warm CPU path:** `run_strategy` accounts forward calls and staged tokens across all three strategies; `choose_strategy` uses `2.52`, the lowest token-work ratio actually measured, rather than claiming evidence below it. In the commit-stamped named-Mac rerun, the CPU-default `NestedSequential` path beat repeated-full in every measured cell by 1.29x-2.02x, and `NestedBatched` stayed within 2.9% of it. The current CPU backend advertises per-lane forward and therefore defaults to `NestedSequential`; `NestedBatched` requires real vectorized capability. The exact medians and provenance are in [`verification/2026-09-20-v0.8.0-35c481a.md`](verification/2026-09-20-v0.8.0-35c481a.md).
9. **High-K state/scheduler stress, implemented without full decoder fan-out:** estimator/admission gates cover K=32/64/128/255, mixed Q/K, tensor-payload ceilings, process-peak envelopes, vectorized lane limits, and fallback. `qwen35_model_stress` defines the explicit checkpoint-gated representative model run, which remains unverified on this checkout.
10. **Repeatability/persistence contract, implementation complete but checkpoint evidence open:** exact same-process state replay is pinned by parity tests. `BranchStateCache` adds tenant isolation, TTL, tensor-byte LRU eviction, and strict `ContentFingerprint` keys. `BackboneState::persist_pinned`/`restore_pinned` add an atomic, versioned, envelope-digested snapshot with pinned execution identity, exact layout validation, fresh process-local lineage, and strict restored-content verification. The two-invocation `persist-save`/`persist-replay` probe defines the fresh-process model gate; its named-machine run remains open.
11. **Direct service adapter, implemented but not production-promoted:** `Qwen35DecisionEngine` loads artifacts offline, registers through `EngineRegistry`, preserves explicit semantic-none mass, and bounds concurrent/queued requests. Cancellation after blocking execution begins, recovery, load shedding telemetry, and soak/load tests remain open.

A change to model, tokenizer, renderer, adapter, head, rejection, calibration, precision, kernels, batching, cache/state representation, or policy triggers the relevant equivalence or new-model review. “Same checkpoint” alone is not sufficient identity.

---

## Repeatability and profile identity

A self-hosted engine should make execution identity stronger, not weaker.

Every returned/model telemetry record should be able to identify the relevant combination of:

```text
model revision
+ tokenizer revision
+ renderer version
+ adapter/head/rejection versions
+ calibration/policy version
+ backend/runtime version
+ arithmetic/precision mode
+ kernel family
+ execution strategy
```

Pinned replay campaigns should distinguish:

- exact deterministic replay where the runtime supports it;
- bounded floating-point differences that preserve the declared contract;
- true profile/runtime changes that must receive a new identity.

The external DGX study observed campaign-to-campaign differences from identical hosted Jev requests. That does not establish a cause, but it is sufficient motivation to measure and version repeatability explicitly rather than assume a service/model label guarantees identical numerical behavior.

---

## Testing strategy

### Current repository tests

- `opendecision-core` — serde/wire round trips and generated-schema checks.
- `opendecision-engine`: deterministic mock/dispatch behavior plus immutable selected-profile identity and validation.
- `opendecision-api` — middleware and SDK-compatibility tests.
- `opendecision-api` — gRPC round-trip tests.
- `opendecision-server` / CLI — launch/parsing/integration behavior.
- `opendecision-backends`: head algebra, artifact validation, exact-token replay,
  Phase 3B reference loading, and candidate-feature probability replay.

Use the verification commands in the root `AGENTS.md` instead of copying a test
total into this document. The backend parity suite must also pass under the
workspace's Rust 1.75 minimum.

### Remaining Phase 3 model/runtime tests

Extend the current fixtures with:

- full-backbone hidden features and distributions;
- immutable state-root fingerprints;
- single versus batched branch storage isolation;
- question order / unrelated-question addition/removal;
- candidate permutation and opaque-ID renaming;
- repeated-full versus nested-sequential parity;
- nested-sequential versus nested-batched Q parity;
- candidate-batch K parity;
- precision/kernel/profile identity changes;
- persistent state restore across processes (in-process TTL, byte eviction, and tenant separation are implemented);
- cancellation and abandoned-reader safety;
- exact/bounded replay repeatability;
- Q=1/4/16 scaling and model-backed high-K stress (state/scheduler K=32/64/128/255 is implemented);
- queue-inclusive load and memory-soak behavior.

Probability delta, argmax changes, and directed application-policy changes must remain separate assertions. Aggregate accuracy is not an implementation-equivalence test.

---

## Operational notes

- **Bound work before dispatch.** Admission should account for state tokens, Q, K, model/profile limits, branch-state bytes, temporary batch expansion, and currently available memory.
- **Do not equate cache budget with process memory.** Model weights, recurrent/convolution/KV state, restored branches, allocator reservations, and backend scratch buffers all matter.
- **Auth and metrics exposure are deployment decisions.** Non-loopback unauthenticated serving and sensitive telemetry require explicit opt-in and redaction rules.
- **Cancellation must be defined across queue and model execution.** Cancelling an HTTP future is not sufficient if GPU work or retained state remains alive.
- **Cache identity includes semantic execution identity.** Similar text is not enough; token/profile/position contracts must match exactly.
- **No silent semantic fallback.** If a backend lacks a required primitive, state operation, precision, or context/Q/K shape, return a precise unsupported-path error.
- **Mac performance must be measured on Mac.** CUDA/L4/A100/DGX results inform hypotheses, not Apple-Silicon latency or memory guarantees.

---

## Public reference artifact

The selected state-first integration line is published at:

<https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>

Treat this repository as a public **OpenDecision reference artifact** tied to the provisional integration profile. It is not evidence that anyone retrained the frozen Qwen backbone. Phase 3A and Phase 3B record no model change, training, or selection. Any adapted or quantized profile requires a distinct identity and its own quality/equivalence evidence.

---

## Architecture lessons from external systems

### Parallel constrained decoding

The public `harshatheg/Qwen-2.5-1B-RLCD` artifact is useful primarily as an inference pattern: shared prefill, cache broadcasting, constrained candidate-token logits, limited token-tree continuation, and host-side structured assembly. It is **not** treated here as evidence that TypeSafe's RLCD training procedure was reproduced, and normalized softmax outputs are not by themselves a calibration result.

OpenDecision adopts the execution lesson—**batch branches breadth-first**—while retaining its state-first root. A schema/question catalog placed before state would weaken the intended question-set isolation contract.

Reference: https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD

### Jev-style DGX Spark benchmark

The September 2026 More Than a Machine comparison is useful as a systems-shape benchmark. Its published Q=1→4 p50 results are approximately:

- Jev 1.13: 105.1 → 109.2 ms;
- Laya: 16.4 → 29.1 ms;
- tuned Qwen3.5 wrapper: 167.0 → 665.1 ms.

The absolute numbers are not apples-to-apples because hosted Jev and warm local readers have different boundaries. OpenDecision therefore adopts **no external latency threshold** from this table. The architectural takeaway is to measure the within-system Q-scaling slope and make additional questions cheaper than Q repeated full evaluations where shared state is substantial.

Reference: https://morethanamachine.com/posts/jev-style-decisions-dgx-spark/

---

## Phasing & what's left

- **Phase 2H — COMPLETED REQUIRED SCOPE.** Criteria/rejection transfer, bounded primitive probes, locked continuation and evidence handoff are complete. Historical failures/limits remain preserved.
- **Phase 2I — BOUNDED PILOT COMPLETE / REVIEWED CONFIRMATION OPEN.** State-first rendering and nested Q sharing have exploratory semantic/mechanical evidence; independent review, natural documents, broader isolation/evidence-sufficiency and semantic high-K confirmation remain open.
- **Phase 2J — EXPLORATORY SCREEN COMPLETE / RELEASE CONFIRMATION OPEN.** Thirteen fit jobs and 31 final profiles completed; Qwen4B/state-first/score-summary is the provisional integration target. Released external baselines and release promotion remain open.
- **Track S — DIRECT NATIVE ADAPTER IMPLEMENTED / SERVICE EVIDENCE OPEN.** `Qwen35DecisionEngine` registers directly, and Choice semantic none is explicit through `__none__`; the Python worker is only a differential oracle. Load/soak, cancellation during active model work, recovery, and deployment promotion remain open.
- **Phase 3A — COMPLETED PYTHON SYSTEMS/REFERENCE SCOPE.** Run `20260920T024056Z` validated full-hybrid-state branch fan-out/select, semantic batched parity, high-K systems parity and exact recorded same-process replay for the frozen selected profile. Its short semantic benchmark also establishes that sharing is workload-dependent rather than universally faster.
- **Phase 3B: COMPLETED PYTHON BACKBONE-REFERENCE SCOPE.** Run `20260920T152206Z` keeps the selected profile unchanged. It exports exact tokens, 34 trace stages, 10 candidate features, and 3 continuation vectors.
- **Phase 3: IN PROGRESS / CPU REFERENCE, SAFE SCHEDULER, AND DIRECT ADAPTER IMPLEMENTED.** Rust head/probability, exact-token, full-sequence CPU backbone, Qwen-specific cached continuation, backend-neutral `BranchableState`, sequential nested execution, lane-topology Q/K parity, measured scheduling, state/scheduler high-K admission, cache lifecycle, and direct engine registration are implemented. Model-backed high-K, fresh-process replay, real vectorized forward, Metal, and service load/soak remain open. CPU native parity does not imply Metal or accelerated parity.
- **Phase 3A.1 — CONDITIONAL SCHEDULER/CROSSOVER STUDY.** Create the additional Colab only if early Rust profiling does not provide enough component timing to derive stable strategy crossover rules. It does not block 3.1–3.5.
- **P2.1–P2.3 — CONDITIONAL.** Optimized kernels, model-weight precision/quantization, and teacher/student work require a specific unmet target and their own parity/quality evidence.

The current architecture decision is conservative. **Keep the selected state-first Qwen profile fixed while validating the native service.** The Phase 3B trace localized and closed the correctness-first CPU full-sequence and cached-continuation gates. The runtime-owned branch contract, sequential nested path, breadth-first lane topology, measured scheduler, safe CPU default, high-K admission model, strict-content cache keys, and direct adapter are now implemented. The next evidence is model-backed high-K, fresh-process replay, and queue-inclusive service load/soak before kernel optimization or Metal. CPU native parity does not imply Metal or accelerated parity. Let reviewed release-quality evidence decide whether the provisional profile ships.
