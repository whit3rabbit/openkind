# opendecision

> **Independent, open-source decision inference engine speaking the Jev protocol** — providing wire- and SDK-compatible judgment-envelope interfaces matching TypeSafe's System One models.
>
> Wire specification: <https://docs.typesafe.ai/api>
> Client SDK reference: <https://docs.typesafe.ai/sdk/python/api>

---

## 🎯 Executive Overview

`opendecision` is a high-throughput, non-generative decision engine implemented in Rust. Instead of generating free-form text via an autoregressive loop and parsing it after the fact, `opendecision` takes a **shared state** and one or more **typed questions** (`Choice`, `Score`, `Noul`) and directly outputs deterministic, schema-validated answers with calibrated probability distributions.

- **Wire & SDK Compatible**: Drop-in compatible with TypeSafe's hosted API (`https://api.typesafe.ai`). Client applications built with `typesafe_sdk` (Python) or [`opendecision-client`](crates/opendecision-client) (Rust) can target either hosted TypeSafe or a local `opendecisiond` daemon without code changes.
- **Independent Architecture**: TypeSafe identifies Jev as its proprietary System One model family. `opendecision` is an independently designed, open-source engine providing a compatible judgment-envelope interface.
- **Dual Transports**: First-class HTTP/REST (`/v1/systemone`, aliased to `/v1/system_one`) via Axum 0.8 and gRPC (`opendecision.SystemOne/Evaluate`) via Tonic 0.14.
- **State-First Hybrid Execution**: Extensively researched and optimized for models with hybrid recurrent/attention backends (such as Qwen 3.5), amortizing prefill computation across multiple questions and candidate alternatives.

---

## 🧭 Repository Navigation

- [`AGENTS.md`](AGENTS.md) — Master navigation briefing, invariant checklists, crate catalog, and engineering guidelines.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Implementation roadmap, phasing checklist (Phases 0 through 4), and milestone progress.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Layered crate topology, transport protocols, data flow, and runtime state management.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — Research dossier, reverse-engineering analysis, benchmarks, and prior art audit.
- [`docs/whitepaper/OpenDecision_Whitepaper_v0.7.1.md`](docs/whitepaper/OpenDecision_Whitepaper_v0.7.1.md) — Canonical scientific whitepaper detailing Phases 2A through 2J, empirical benchmarks, and architectural conclusions.

---

## 📊 Project Status & Phased Roadmap

| Phase | Description | Status | Verification / Artifacts |
|---|---|:---:|---|
| **Phase 0** | Jev Wire Contract & Canonical Types | **DONE** ✅ | 43 tests; JSON Schema Draft 2020-12; 8 canonical examples |
| **Phase 1** | Daemon, Transports & SDK Parity | **DONE** ✅ | Axum HTTP, Tonic gRPC, `MockEngine`, CLI, client SDK (259 tests) |
| **Phase 2** | Empirical Model Research & Benchmarks | **COMPLETED** ✅ | Phases 2A–2J completed; provisional model profile selected & locked |
| **Track S** | Contract Hardening & Thin Real-Service Bridge | **IN PROGRESS** 🚧 | Reference worker bridge & deployment safeguard architecture |
| **Phase 3** | Production Rust Engine & Native Backend | **READY / NEXT** 🚀 | Native parity against profile `a047d6802c3f06f085b8` |
| **Phase 4** | Autonomous Deployment & Edge Packaging | **PLANNED** 📋 | Embedded runtimes, Apple Silicon / Metal optimizations |

### Current Test Suite Status: 302 Tests Passing at HEAD
The workspace maintains **302 automated tests** across 9 crates and doc-tests (`cargo test --workspace`):

```text
Suite                                                Tests   Status
-------------------------------------------------------------------
opendecision-core (unit + Jev conformance)              43   PASS
opendecision-engine (unit + dispatch + MockEngine)      20   PASS
opendecision-api (unit + grpc_roundtrip + sdk_compat)  103   PASS
opendecision-client (unit + live + retry + parity)     107   PASS
opendecision-cli (unit + command line interface)        10   PASS
opendecision-backends (driver unit tests)                5   PASS
opendecision-runtime (device detection + limits)         6   PASS
opendecision-server (daemon configuration tests)         3   PASS
opendecision-gen-schemas (schema synchronization)        3   PASS
opendecision-proto (protobuf wire roundtrips)            2   PASS
-------------------------------------------------------------------
Total Automated Workspace Tests                        302   PASS
```

---

## 🔬 Empirical Research & Findings (Phases 2A–2J)

Between September 17 and September 20, 2026, extensive empirical investigations using Qwen 3.5, ModernBERT, and alternative architectures produced a rigorous foundation for native engine design:

1. **Non-Generative Decision Feasibility (Phases 2A–2C)**:
   - A frozen `Qwen/Qwen3.5-4B-Base` backbone (4.2B parameters, hidden dimension 2,560) combined with a compact linear classification head (7,683 parameters) achieved **87.67% matched / 87.33% mismatched accuracy on MultiNLI** without autoregressive token generation.
   - Avoiding the generation loop yielded a **~28.6× latency speedup** over generative prompt completions.
   - Evaluated dynamic candidate transfer on Banking77: achieved **88.54% accuracy** on seen labels and **82.29% accuracy** on labels withheld from head fitting.

2. **Precision & Numerical Equivalence (Phases 2D–2E)**:
   - Evaluated FP32, TF32, and BF16 execution modes against an exact full-sequential FP32 reference.
   - FP32 batching and lossless cache reuse maintained within **1.1 × 10⁻⁵ maximum probability delta** with zero policy flips.
   - BF16 failed the declared 0.005 probability tolerance and altered discrete decision actions.
   - TF32 provided ~2× batching speedup but introduced drift on edge cases; therefore, FP32 remains the canonical numerical reference.

3. **Cache Compression & Prefix Reuse (Phase 2F)**:
   - Evaluated 6 cache codecs on Qwen 3.5: FP16 attention-KV storage passed all numerical and policy gates.
   - All four low-bit TurboQuant codecs failed tolerance gates; low-bit compression was rejected for decision inference.
   - Lossless exact-prefix reuse delivered **11.9%–13.9% wall-clock latency reduction** on controlled multi-request workloads.

4. **Rejection & Criteria Transfer (Phases 2G–2H)**:
   - Investigated the missing-option problem ("none" handling) and out-of-scope (OOS) rejection across Banking77 and CLINC150.
   - Demonstrated that semantic "none", author-OOS, and evidence insufficiency are separate failure modes requiring structured rejection heads rather than arbitrary post-hoc thresholding.
   - Phase 2H completed all required evaluation workers across 14 comparison arms, locking development artifacts.

5. **Exploratory Model Selection Screen (Phases 2I–2J / Workbench `2ij.2.0`)**:
   - Completed 13 fit jobs and evaluated 31 locked final profiles across Qwen 3.5 4B, Qwen 3.5 2B (frozen, online-head, LoRA), ModernBERT-large (frozen, fully fine-tuned), and controls.
   - **Key Finding: State-First Rendering is Learned**: Placing the state before the question/candidate prompt (`state → question → candidate`) significantly outperformed instruction-first rendering (95.0% vs 86.25% accuracy on 4B; 91.56% vs 85.31% on 2B LoRA). State-first is a learned model contract, not merely a caching trick.
   - **ModernBERT Compact Arm**: ModernBERT was very fast (~220 ms) but struggled with dynamic multi-choice generalization (42%–51% accuracy), demonstrating that bidirectional encoders require different training formulations.
   - **Provisional Integration Candidate Selected**:
     - **Profile ID**: `a047d6802c3f06f085b8`
     - **Architecture**: `Qwen/Qwen3.5-4B-Base`, frozen backbone, **state-first rendering**, **score-summary rejection head**.
     - **Performance**: **95.00% accuracy**, 0.13006 NLL, 0.06168 family-macro NLL, 0.01661 ECE on 320 held-out episodes (with natural MultiRC slice at 83.33%).
     - **Reload Parity**: Passed clean A100 reload checks with maximum probability delta **3.67 × 10⁻⁶**, zero selected-ID changes, and passed NumPy/f64 head-algebra verification.
     - **Exported Reference Bundle**: SHA-256 `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`.

6. **External Benchmarks & Architectural Prompts (v0.7.1)**:
   - Community Parallel Constrained Decoding (PCD) demonstrated one-prefill/batched-field execution on Apple Silicon.
   - DGX Spark benchmark demonstrated near-flat Q=1→4 latency on Jev hosted endpoints, proving that multi-question amortization is achievable and essential.

---

## 🚀 The Phase 3 Native Implementation Direction

With the provisional integration profile `a047d6802c3f06f085b8` locked and its reference bundle verified, model selection no longer blocks native systems engineering. Phase 3 focuses on native execution in Rust:

```text
Phase 3 Execution Pipeline:
1. Head / Probability Parity (opendecision-engine / opendecision-backends)
   └── Reproduce normalization, score-summary rejection, stable softmax, calibration
2. Exact Tokenizer & State-First Renderer (opendecision-backends)
   └── Match exact token IDs, segment markers, positional encodings, and truncation
3. Full Qwen 3.5 Backbone Parity (opendecision-runtime / opendecision-backends)
   └── Verify hidden features, argmax outcomes, and policy actions against golden fixtures
4. Backend-Neutral BranchableState Abstraction (opendecision-runtime)
   └── Isolate recurrent DeltaNet state, Conv state, and Attention KV state at branch points
5. Sequential Nested Execution Baseline
   └── state → question → candidate tree evaluation with root immutability guarantees
6. Breadth-First Batched Question Execution
   └── Fork immutable state root into a Q-batch; evaluate question suffixes concurrently
7. Batched Candidate Execution
   └── Length-bucketed candidate suffix evaluation per question
8. Empirical Q-Amortization Measurement
   └── Measure T(Q)/T(1) scaling curves, throughput (questions/s), and memory overhead
9. High-Cardinality Systems Stress
   └── Exercise K ∈ {32, 64, 128, 255} candidate sets
10. Production Service Lifecycle
    └── Queueing, admission limits, tenant isolation, and cancellation safeguards
```

---

## 📦 Workspace Architecture & Crates

```text
opendecision/
├── crates/
│   ├── opendecision-core/        # Zero-dependency Jev types, serde, validation & errors
│   ├── opendecision-engine/      # DecisionEngine trait, MockEngine, execution contracts
│   ├── opendecision-api/         # Axum 0.8 HTTP router & Tonic 0.14 gRPC service
│   ├── opendecision-server/      # opendecisiond server daemon binary
│   ├── opendecision-cli/         # opendecision operator CLI utility
│   ├── opendecision-client/      # Async Rust client SDK (typesafe_sdk counterpart)
│   ├── opendecision-runtime/     # Hardware detection, device memory & state management
│   ├── opendecision-backends/    # Native model drivers (Candle, GGUF, ONNX)
│   └── opendecision-gen-schemas/ # Tooling for JSON Schema Draft 2020-12 generation
├── proto/                        # opendecision.proto Protobuf & gRPC definitions
├── examples/                     # Canonical Jev JSON request/response fixtures
├── research/                     # Jupyter research notebooks (Phases 2A–2J) and archives
└── docs/                         # Whitepaper, architecture, roadmap, and research dossiers
```

---

## ⚡ Quickstart

### Prerequisites
- Rust 1.80+ (`cargo`, `rustc`)
- Protocol Buffers compiler (`protoc`) for gRPC codegen

### Build & Run Tests
```bash
# Clone the repository
git clone https://github.com/your-org/opendecision.git
cd opendecision

# Build workspace
cargo build --workspace

# Run all 302 workspace tests (unit, conformance, SDK parity, doc tests)
cargo test --workspace

# Verify JSON Schema generation
cargo run -p opendecision-gen-schemas -- --write
```

### Launch the Daemon
```bash
# Run opendecisiond with mock backend in development mode
cargo run -p opendecisiond -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest
```

### Query the API
```bash
# Execute a multi-question evaluation request
curl -sS -X POST http://127.0.0.1:18080/v1/systemone \
     -H 'Content-Type: application/json' \
     -d @examples/04_mixed.json | jq
```

### Use the Rust Client SDK
```rust
use opendecision_client::{Client, question};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder()
        .base_url("http://127.0.0.1:18080")
        .build()?;

    let response = client
        .system_one(
            "Customer states: 'I cannot log in and password reset fails.'",
            [
                ("urgent", question::noul("Is immediate escalation required?")),
                ("department", question::choice("Route ticket to:", [
                    ("auth_support", Some("Authentication & login issues")),
                    ("billing", Some("Billing & subscription issues")),
                    ("general", None),
                ])),
            ],
        )
        .await?;

    println!("Response ID: {}", response.model);
    Ok(())
}
```

---

## 🛡️ Architectural & Engineering Invariants

1. **Path Discipline**: Never use absolute machine paths (`/Users/...` or `file:///...`) in documentation, code, or comments. Always use relative paths (`crates/opendecision-core`, `docs/ROADMAP.md`).
2. **Wire Format Stability**: `f64` precision is strictly enforced across all probability and score fields. Wire types must match TypeSafe's Jev JSON schema Draft 2020-12 and `opendecision.proto`.
3. **Clean Dependency Flow**: `core` → `engine` → `runtime`/`backends` → `api` → `server`/`cli`. No circular dependencies.
4. **Hybrid State Isolation**: Qwen 3.5 utilizes both attention and recurrent DeltaNet layers. Branching optimizations must isolate recurrent and convolution states alongside KV caches.
5. **Quality vs Equivalence**: Model selection and numerical equivalence are distinct gates. Never alter model contracts or selection criteria after inspecting test outcomes.
