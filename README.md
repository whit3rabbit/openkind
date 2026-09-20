# opendecision

`opendecision` is an independent Rust decision-inference engine for typed `Noul`, `Choice`, and `Score` answers over Jev-compatible public interfaces. It is built to answer structured questions without depending on an autoregressive text-generation loop.

The wire, service, and SDK layers are implemented. The selected Qwen 3.5 native path has passed Rust head, tokenizer, correctness-first CPU backbone, and cached-continuation parity gates. Backend-neutral branch execution, Metal validation, and production model serving remain open.

[Quickstart](#quickstart) | [Research dossier](docs/RESEARCH.md) | [Whitepaper](docs/whitepaper/OpenDecision_Whitepaper_v0.7.2.md) | [Roadmap](docs/ROADMAP.md) | [Architecture](docs/ARCHITECTURE.md) | [Jev wire reference](https://docs.typesafe.ai/api)

> [!IMPORTANT]
> The daemon currently maps configured model aliases to `MockEngine`. The native Qwen backend is not registered yet. The quickstart below verifies the wire and service path, not model quality or native Qwen execution.

## Quickstart

You need Rust 1.75 or newer and `protoc` for gRPC code generation.

```bash
git clone https://github.com/whit3rabbit/opendecision.git
cd opendecision
cargo build --workspace
cargo run -p opendecision-server --bin opendecisiond -- \
    --http-addr 127.0.0.1:18080 \
    --grpc-addr 127.0.0.1:19090 \
    --models mock,jev-latest
```

In another terminal, send the canonical mixed-question fixture:

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/systemone \
    -H 'Content-Type: application/json' \
    -d @examples/04_mixed.json
```

The same service is available through gRPC at `opendecision.SystemOne/Evaluate`.

## What it provides

- Typed Jev request and response models with JSON Schema generation.
- HTTP endpoints at `/v1/systemone` and `/v1/system_one`.
- A gRPC `SystemOne/Evaluate` service using Protobuf `double` values.
- A Rust client with retry, configuration, and wire-parity coverage.
- A deterministic mock engine for service and integration testing.
- Native Qwen 3.5 contracts for model identity, head algebra, tokenization, and reference tensors.

The public interfaces target TypeSafe System One wire compatibility where the repository has explicit contract coverage. The neural implementation is independent. This project does not claim to reproduce TypeSafe's private architecture or training process.

## Research record

This repository carries the research behind the implementation, not only the implementation plan. Phases 2A through 2J tested model families, rendering order, precision, rejection, cache reuse, batching, transfer, and model selection. Phases 3A and 3B then converted that work into branch-state and backbone reference artifacts for the Rust port.

| Work | Evidence produced | Boundary |
|---|---|---|
| Phases 2A through 2C | Frozen-backbone feasibility, compact readout heads, dynamic candidate transfer | Research evidence, not general decision competence |
| Phases 2D through 2H | FP32, BF16, TF32, cache, rejection, policy, and complete-request studies | Python and GPU evidence, not native Rust parity |
| Phases 2I and 2J | 13 fit jobs and 31 locked final profiles across Qwen 3.5, ModernBERT, LoRA, and controls | Selected a provisional integration profile, not a release model |
| Phase 3A | Full-hybrid branch-state and batched question/candidate reference behavior | Python systems reference, not Rust or Metal proof |
| Phase 3B | Four exact token records and 47 FP32 vectors across 34 trace stages, candidates, and continuations | Backbone localization fixtures, not native execution |
| Rust Phase 3.1 through 3.3 | Head, probability, tokenizer, state-first rendering, CPU backbone, and cached continuation | Frozen-fixture parity, not Metal or production service proof |

Start with the [research dossier](docs/RESEARCH.md) for the study sequence and the [whitepaper](docs/whitepaper/OpenDecision_Whitepaper_v0.7.2.md) for methods, results, and interpretation. The [roadmap](docs/ROADMAP.md) is the current status authority. The public [Qwen 3.5 reference repository](https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst) exposes the selected integration line.

### Locked native integration target

| Field | Value |
|---|---|
| Profile | `a047d6802c3f06f085b8` |
| Backbone | `Qwen/Qwen3.5-4B-Base` |
| Revision | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Renderer | State first |
| Readout | Score-summary rejection head |
| Bundle SHA-256 | `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332` |
| Calibration temperature | `1.8186799910442777` |
| Policy threshold | `0.98` |
| Probability tolerance | `0.005` |

These values define the parity target. They do not promote the profile to release quality. Independent review, natural-data confirmation, backend-neutral execution, Metal, and production validation remain separate gates.

## Current implementation status

| Area | Status |
|---|---|
| Jev types, validation, schemas, and fixtures | Implemented |
| HTTP, gRPC, CLI, daemon, and Rust client | Implemented with mock-engine service coverage |
| Native head and probability algebra | Parity gate passed |
| Offline tokenizer and state-first renderer | Exact token gate passed |
| Phase 3B architecture and reference-vector loader | Validation gate passed |
| Native Qwen CPU embedding and decoder execution | Frozen Phase 3B parity gate passed |
| Qwen-specific full-hybrid continuation state | Cached continuation gate passed |
| Backend-neutral `BranchableState` | In progress |
| Batched question and candidate execution | Python reference complete, Rust implementation open |
| Metal and production service validation | Open |

Implementation equivalence and release promotion are different decisions. Passing a parity fixture does not establish model quality, hardware support, or production readiness.

## Architecture

The workspace keeps wire types independent from model execution and transport code:

```text
HTTP / gRPC / Rust client
          |
          v
Jev validation and typed requests
          |
          v
EngineRegistry and DecisionEngine
          |
          +---- MockEngine                     current service path
          |
          +---- native Qwen 3.5 backend        integration path
                         |
                         v
             typed answers and probabilities
```

The native execution order is state first:

```text
state -> question -> candidate -> backbone -> readout -> policy
```

Qwen 3.5 branch state is more than attention KV. Correct isolation also requires DeltaNet recurrent state, convolution state, logical position, and profile identity. The [architecture document](docs/ARCHITECTURE.md) covers the crate boundaries and full execution plan.

### Workspace map

| Area | Crates |
|---|---|
| Wire types and engine contracts | `opendecision-core`, `opendecision-engine` |
| HTTP and gRPC | `opendecision-api`, `opendecision-server`, `opendecision-proto` |
| Clients and operator tools | `opendecision-client`, `opendecision-cli` |
| Hardware and model execution | `opendecision-runtime`, `opendecision-backends` |
| Schema generation | `opendecision-gen-schemas` |

## Rust client

```rust
use opendecision_client::{question, Client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder()
        .base_url("http://127.0.0.1:18080")
        .build()?;

    let response = client
        .system_one(
            "Customer cannot log in after resetting a password.",
            [
                ("urgent", question::noul("Is immediate escalation required?")),
                (
                    "department",
                    question::choice(
                        "Route this ticket.",
                        [("auth_support", None), ("billing", None)],
                    ),
                ),
            ],
        )
        .await?;

    println!("{}", response.model);
    Ok(())
}
```

See the [`opendecision-client` guide](crates/opendecision-client/README.md) for configuration, authentication, retries, and request options.

## Development

Run the repository verification battery before submitting changes:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
env -u RUST_LOG cargo test --workspace
cargo run -p opendecision-gen-schemas -- --write
git diff --check
```

Tests and builds must not download model artifacts. Native parity fixtures are vendored and digest-checked. After schema generation, confirm that unrelated schema files did not change.

For module-specific invariants and focused checks, start with [`AGENTS.md`](AGENTS.md).

## Documentation

- [Roadmap](docs/ROADMAP.md): current phase status, gates, and remaining work.
- [Architecture](docs/ARCHITECTURE.md): crate boundaries, data flow, and runtime state.
- [Research dossier](docs/RESEARCH.md): experiment sequence, evidence, and prior art.
- [Whitepaper](docs/whitepaper/OpenDecision_Whitepaper_v0.7.2.md): scientific rationale and measured results.

## License

Cargo metadata declares `MIT OR Apache-2.0`. The checked-in [license text](LICENSE) contains the MIT terms.
