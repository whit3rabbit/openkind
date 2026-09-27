# openkind

`openkind` is an independent Rust decision-inference engine for typed `Noul`, `Choice`, and `Score` answers over Jev-compatible public interfaces. It is built to answer structured questions without depending on an autoregressive text-generation loop.

The selected Qwen 3.5 engine can reuse a prefilled state across question and
candidate branches, then computes typed answers in Rust. Native CPU and FP32
MLX pass their pinned implementation-parity gates.

Reviewed model quality, practical accelerated request latency, and
product-release promotion remain open. See
[current status](#current-implementation-status) and the
[roadmap](docs/ROADMAP.md).

[Quickstart](#quickstart) | [Decision engine features](#decision-engine-features) | [Model registry](docs/MODEL_REGISTRY.md) | [Research dossier](docs/RESEARCH.md) | [Whitepaper](docs/whitepaper/WHITEPAPER.md) | [Roadmap](docs/ROADMAP.md) | [Architecture](docs/ARCHITECTURE.md) | [MLX backend](docs/MLX.md) | [Jev wire reference](https://docs.typesafe.ai/api)

> [!IMPORTANT]
> The daemon defaults to `MockEngine`. A native alias can be registered directly with explicit offline bundle, checkpoint, and tokenizer paths. The quickstart below still verifies the mock wire/service path, not model quality or native Qwen execution.

Model catalog metadata is checked into this repository and published through
the separate public [OpenKind model registry](https://github.com/whit3rabbit/openkind-model-registry).
That repository holds the small pinned profile assets; checkpoint shards stay
with their authors. See the [registry guide](docs/MODEL_REGISTRY.md) for pulls,
explicit serving, and the script used to verify both repositories stay aligned.

## Quickstart

You need Rust 1.88 or newer and `protoc` for gRPC code generation.

```bash
git clone https://github.com/whit3rabbit/openkind.git
cd openkind
cargo build --workspace
cargo run -p openkind-server --bin openkindd -- \
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

The same service is available through gRPC at `openkind.SystemOne/Evaluate`.

## Decision engine features

The selected Qwen 3.5 engine evaluates a state and its questions as decision
work. It does not decode an answer document token by token:

- **One shared state prefill:** State-first tokenization forms an immutable root
  for the shared document. The nested strategies reuse that root across
  questions, so adding a question does not change the document prefix.
- **Question and candidate branches:** In nested execution, each question
  advances from the root. Candidates then advance from that question state,
  so the question suffix is computed once per question instead of once per
  candidate. Every branch has its own attention KV, DeltaNet recurrent state,
  convolution state, and position.
- **Vectorized MLX continuation:** An explicitly forced FP32 `nested_batched`
  run can advance compatible groups of 2–8 question or candidate lanes in one
  right-padded MLX forward and restore each lane's true length. CPU batched
  execution has the same branch topology but advances lanes separately.
  Automatic MLX scheduling remains per-lane.
- **Decision readout, then typed output:** Candidate-conditioned features feed
  the score-summary head. Choice includes explicit `__none__` rejection mass;
  Rust assembles `Noul`, `Choice`, or `Score` answers and their probabilities.
  The model does not generate JSON keys, punctuation, or prose responses.
- **Workload-aware execution:** The scheduler can choose independent full
  evaluation or shared-root execution. It checks backend capabilities and
  retained-state and process-memory limits. Queue and execution permits stay
  with the running task through cancellation.

These are execution and interface features, not a claim of better semantic
accuracy than a token-logit classifier or a text-generating model. The
[architecture guide](docs/ARCHITECTURE.md) covers the state graph and the
[MLX guide](docs/MLX.md) covers physical batching.

### Measured behavior

- **CPU sharing:** On a 36-GiB M4 Max, warm-process FP32
  `nested_sequential` was 1.294×–2.018× faster than `repeated_full` across
  five pinned workloads. This is native compute evidence, not a service-wide
  speedup or model-quality result ([commit-stamped run](docs/verification/2026-09-20-v0.8.0-35c481a.md)).
- **FP32 MLX question batching:** In paired stage replay, forced vectorization
  reduced question-stage time by 5.5%–6.1% at Q2 and 12.4%–15.3% at Q8.
  Full, nested, and unequal-length vectorized parity pass at the pinned
  `0.005` probability tolerance. The stage gain did not establish a complete
  request speedup ([candidate-pooling record](docs/benchmarks/2026-09-27-candidate-pooling/README.md)).
- **Cross-question candidate pooling:** In the same native compute replay,
  pooling was 1.5% slower than existing nested batching at Q2/K2 and 7.8%
  slower at Q8/K4. It remains a diagnostic
  ([candidate-pooling record](docs/benchmarks/2026-09-27-candidate-pooling/README.md)).
- **Flat field batching:** A diagnostic Rust port of shared-root field
  traversal reduced physical forwards but was 6.0%–12.1% slower at Q2/K2 and
  64.3%–64.5% slower at Q8/K4 than nested batching. It repeated question
  tokens and increased padding; it is not in automatic scheduling
  ([paired record](docs/benchmarks/2026-09-27-python-flat-field/README.md)).

The MLX native compute timings include prefill, continuation, and readout on
frozen token workloads. They exclude model load, rendering, tokenization,
validation, transport, and queueing. Q8/K4 repeats fixture suffixes as a load
shape.

No Qwen 3.5 comparison against autoregressive JSON generation or complete
request-path MLX speedup follows from these measurements. See the
[benchmark guide](docs/BENCHMARKS.md) for methods and other recorded runs.

### Public interfaces

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
| Rust Phase 3.1 through 3.8 | Head, probability, tokenizer, state-first rendering, CPU backbone, cached continuation, branch-state contract, sequential nested execution, batched Q/K, and the measured adaptive scheduler | Frozen-fixture parity and warm-host measurements, not Metal or production service proof |

Start with the [research dossier](docs/RESEARCH.md) for the study sequence and the [whitepaper](docs/whitepaper/WHITEPAPER.md) for methods, results, and interpretation. The [roadmap](docs/ROADMAP.md) is the current status authority. The public [model registry](https://github.com/whit3rabbit/openkind-model-registry) distributes the selected Qwen profile bundle and exported tokenizer; the [bundle source note](https://github.com/whit3rabbit/openkind-model-registry/blob/main/assets/qwen35-state-first/a047d6802c3f06f085b8/bundle/SOURCE.md) records reference provenance.

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

### Optional model download for local parity and benchmarking

Normal builds, tests, and the `openkind-bench` harness never download model
artifacts. For an explicit real-checkpoint run, install the Hugging Face CLI
and download the pinned base checkpoint locally:

```bash
# Run this only if the `hf` command is not already installed.
python3 -m pip install --user --upgrade huggingface_hub
MODEL_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/openkind"

hf download Qwen/Qwen3.5-4B-Base \
  --revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --local-dir "$MODEL_CACHE_DIR/qwen35-4b-base-1001bb4d826a52d1f399e183466143f4da7b741b"
```

The optional MLX-community comparison artifact can be downloaded with its
independent revision:

```bash
hf download mlx-community/Qwen3.5-4B-MLX-bf16 \
  --revision 475632ded9a95863da4e4b235ab9ccbc5d3cc6bf \
  --local-dir "$MODEL_CACHE_DIR/mlx-community-qwen35-4b-mlx-bf16-475632d"
```

The pinned base directory is the parity target. The community directory is a
compatibility and model-difference comparison only. The [community model
card](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16) identifies it
as `Qwen/Qwen3.5-4B` converted through an `mlx-vlm` fix branch, not the pinned
`Qwen/Qwen3.5-4B-Base` revision. See the backend briefing and
[`docs/MLX.md`](docs/MLX.md) for the backend contract and
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for recorded results.

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
| Backend-neutral `BranchableState` | CPU and MLX structural gates pass; pinned-base variable-length FP32 vectorized batch parity passes in a forced diagnostic path |
| MLX/Metal parity backend (Phase 3M, `--features mlx`) | Pinned-base FP32 full/nested/vectorized batch gates pass; BF16 Gate B fails probability tolerance; daemon request-path and bounded memory stress pass; packed kernel remains opt-in and automatic vectorized scheduling remains disabled after negative diagnostic comparisons |
| Sequential nested execution (CPU) | Phase 3.5 gate passed |
| Breadth-first batched Q/K execution (CPU) | Phase 3.6/3.7 gate passed; vectorized kernels open |
| Adaptive scheduler (CPU, named Mac) | Phase 3.8 measured gate passed; 2.52 is the lowest measured boundary |
| High-K state/scheduler stress and state reuse | K=32/64/128/255 bounded model-backed correctness/memory campaign passed; semantic quality and practical latency remain open |
| Native CPU service lifecycle (Phase 3.11, S.4–S.5) | Named-machine release-mode endpoint, queue/load, deadline, recovery, memory, and 30-minute soak gates passed; reviewed quality and product-release promotion remain open |
| Accelerated production and service validation | MLX FP32 parity, one forced daemon request, bounded memory stress, and negative candidate-pooling/flat-field native compute comparisons are recorded; complete request-path performance, service load/soak, and production promotion remain open |

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
          +---- MockEngine                     default service path
          |
          +---- Qwen35DecisionEngine           explicit native alias
                         |
                         v
             typed answers and probabilities
```

The native execution order is state first:

```text
state -> question -> candidate -> backbone -> readout -> policy
```

Qwen 3.5 branch state is more than attention KV. Correct isolation also requires DeltaNet recurrent state, convolution state, logical position, and profile identity. The [architecture document](docs/ARCHITECTURE.md) covers the crate boundaries and full execution plan.

Native Choice requests must include an explicit `__none__` criteria key. The adapter returns the model's semantic-none mass under that key without appending a hidden class, discarding mass, or renormalizing. Choice and Score confidence use normalized distribution entropy, not maximum probability.

### Workspace map

| Area | Crates |
|---|---|
| Wire types and engine contracts | `openkind-core`, `openkind-engine` |
| HTTP and gRPC | `openkind-api`, `openkind-server`, `openkind-proto` |
| Clients and operator tools | `openkind-client`, `openkind-cli` |
| Hardware and model execution | `openkind-runtime`, `openkind-backends` |
| Schema generation | `openkind-gen-schemas` |

## Rust client

```rust
use openkind_client::{question, Client};

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

See the [`openkind-client` guide](crates/openkind-client/README.md) for configuration, authentication, retries, and request options.

## TypeScript, Python, and Swift clients

Source packages in [`bindings/`](bindings/README.md) call a running `openkindd` over the same System One HTTP API as the CLI. They cover evaluation, model listing, and health, with request-bound answer checks. Each also has a local process wrapper that can start and stop `openkindd`. See the package guides for platform limits and tests.

## Development

Run the repository verification battery before submitting changes:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
env -u RUST_LOG cargo test --workspace
cargo run -p openkind-gen-schemas -- --write
git diff --check
```

Tests and builds must not download model artifacts. Native parity fixtures are vendored and digest-checked. After schema generation, confirm that unrelated schema files did not change.

For a clean-commit verification with retained logs and the checkpoint-backed
Qwen parity/benchmark gates, use:

```bash
scripts/verify-commit.sh --checkpoint-root path/to/Qwen3.5-4B-Base
```

The script refuses a dirty tree and never downloads weights. Use
`--offline-only` only when the resulting record will explicitly remain a
partial, non-model verification.

For module-specific invariants and focused checks, start with [`AGENTS.md`](AGENTS.md).

## Documentation

- [Roadmap](docs/ROADMAP.md): current phase status, gates, and remaining work.
- [Architecture](docs/ARCHITECTURE.md): crate boundaries, data flow, and runtime state.
- [MLX backend](docs/MLX.md): explicit-stream execution, weights, branch state, kernels, limitations, and enhancement priorities.
- [Benchmarks](docs/BENCHMARKS.md): methodology, harness commands, and recorded runs.
- [Research dossier](docs/RESEARCH.md): experiment sequence, evidence, and prior art.
- [Whitepaper](docs/whitepaper/WHITEPAPER.md): scientific rationale and measured results.

## License

Cargo metadata declares `MIT OR Apache-2.0`. The checked-in [license text](LICENSE) contains the MIT terms.
