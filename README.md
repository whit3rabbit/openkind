# openkind

`openkind` runs a Typesafe server and cli (Jev) with local hosted models for the server.

I highly recommend using: https://github.com/ollaya-dev/ollaya

We seem to have had the same idea when I started working and their public release is a lot more polished.

OpenKind researches a custom Qwen3.5 decision model and a Rust inference engine that shares input processing across questions.

[![CI](https://github.com/whit3rabbit/openkind/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/openkind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Install

Install the `openkind` CLI and `openkindd` server using Homebrew, Cargo, or prebuilt binaries. Keep both binaries on `PATH`: `openkind serve` starts the server. The standard distributions run models on the CPU; Apple silicon users who want GPU acceleration can [build from source with MLX](#apple-silicon-and-mlx).

### Homebrew (macOS and Linux)

```bash
brew install whit3rabbit/tap/openkind
openkind version
```

### Cargo

Install from crates.io with Rust 1.88+:

```bash
cargo install --locked openkind-cli openkind-server
openkind version
```

### Releases

Download prebuilt binary archives for macOS (Apple silicon or Intel) and Linux (`x86_64` musl) from [GitHub Releases](https://github.com/whit3rabbit/openkind/releases). Unpack the tarball and add `openkind` and `openkindd` to your `PATH`.

## Run a model

Browse the catalog, pull a profile, and serve it locally. This example uses the Laya English decision encoder:

```bash
openkind catalog
openkind pull laya-english:c8ea29bf1e33a343c4b7
openkind serve \
  --installed-models laya-english:c8ea29bf1e33a343c4b7 \
  --http-addr 127.0.0.1:18080 \
  --grpc-addr 0
```

`pull` downloads pinned model files and verifies their SHA-256 digests. Builds and tests do not download weights. See the [model guide](docs/MODELS.md) for download sizes, memory requirements, and available backends.

In another terminal, save a request as `request.json`:

```json
{
  "state": "My card was charged twice. Please refund the extra payment.",
  "model": "laya-english:c8ea29bf1e33a343c4b7",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, charges, and refunds",
        "technical": "Bugs, outages, and integrations",
        "__none__": "None of these teams"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?"
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

Submit it with the CLI:

```bash
openkind evaluate request.json --server http://127.0.0.1:18080 --format text --verbose
```

The output shows each answer and its available probabilities. Omit `--format text --verbose` to get JSON for scripts.

| Question | Returns | Example use |
|---|---|---|
| `choice` | A selected option and probability distribution | Route a ticket to a team |
| `noul` | A yes/no probability | Detect a refund request |
| `score` | A numeric score against an ordered rubric | Rate urgency or frustration |

Native Qwen Choice questions require a non-empty `__none__` option. Keeping it in your requests gives that model a way to reject the offered choices.

<details>
<summary>Try the API without downloading a model</summary>

From the checkout, start a mock server:

```bash
openkindd --models jev-latest --http-addr 127.0.0.1:18080 --grpc-addr 0
```

In another terminal:

```bash
openkind evaluate examples/04_mixed.json --server http://127.0.0.1:18080 --pretty
```

Stop any server already using that port first. In this setup, `jev-latest` returns deterministic mock answers for integration testing.

</details>

## CLI and playground

Manage local models and inspect the running server:

```bash
openkind list
openkind show laya-english:c8ea29bf1e33a343c4b7
openkind status --server http://127.0.0.1:18080
openkind inspect request.json
```

For a browser interface, start the playground with an installed model:

```bash
openkind playground --installed-models laya-english:c8ea29bf1e33a343c4b7
```

It opens a local page where you can edit requests, inspect probabilities, and load or unload installed models. Running `openkind playground` alone uses mock models. The daemon rejects `--playground on` unless its HTTP listener is bound to a loopback address, keeping model lifecycle controls local even when inference authentication is disabled or an API key is shared. See the [CLI guide](crates/openkind-cli/README.md) for commands and the [registry guide](docs/MODEL_REGISTRY.md) for model storage and lifecycle.

## Use from Rust

Use [`openkind-client`](crates/openkind-client/README.md) to call the local server. For an application beside the cloned `openkind` directory, add these dependencies to `Cargo.toml`:

```toml
[dependencies]
openkind-client = { path = "../openkind/crates/openkind-client" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

With the Laya server above running, put this in `src/main.rs` and run `cargo run`:

```rust
use std::time::Duration;

use openkind_client::{question, Client, RetryPolicy};

#[tokio::main]
async fn main() -> Result<(), openkind_client::Error> {
    let client = Client::builder()
        .api_key("local") // The client requires a key; this local server has auth disabled.
        .base_url("http://127.0.0.1:18080")
        .default_model("laya-english:c8ea29bf1e33a343c4b7")
        .timeout(Duration::from_secs(600))
        .retry(RetryPolicy::new().total_timeout(Some(Duration::from_secs(600))))
        .build()?;

    let response = client
        .system_one(
            "My card was charged twice. Please refund the extra payment.",
            [(
                "refund_requested",
                question::noul("Is the customer asking for a refund?"),
            )],
        )
        .await?;

    println!("{:?}", response.answers);
    Ok(())
}
```

For inference inside your Rust process, use [`openkind-backends`](crates/openkind-backends/README.md) through the [`DecisionEngine` interface](crates/openkind-engine/README.md). [`openkind-core`](crates/openkind-core/README.md) provides the shared request types and validation. The HTTP client connects to a server; the backend crates load and execute models.

TypeScript, Python, and Swift HTTP clients and local server wrappers are also available. See the [bindings guide](bindings/README.md).

## API compatibility

`openkindd` serves Jev's typed `Noul`, `Choice`, and `Score` contract over HTTP and gRPC. The HTTP API includes `POST /v1/systemone` and `GET /v1/models`. The same request works with `curl`:

```bash
curl -sS http://127.0.0.1:18080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

Existing TypeSafe clients can target the local server with a supported request and a registered model name. The Rust client also supports TypeSafe, OpenRouter System One, and Cloudflare Workers AI. Accepted fields and provider routes differ; the [compatibility matrix](docs/JEV_COMPATIBILITY.md) records those limits.

API compatibility does not imply the same predictions as TypeSafe's Jev. OpenKind implements its own models and inference paths. Your application still decides which actions an answer may trigger.

## Apple silicon and MLX

Enable the optional MLX/Metal backend on macOS arm64 to run supported models on the GPU:

```bash
export SDKROOT=$(xcrun --show-sdk-path)
cargo build --release --locked -p openkind-cli -p openkind-server --features openkind-server/mlx
openkindd \
  --installed-models laya-english:c8ea29bf1e33a343c4b7 \
  --laya-backend mlx-fp32 \
  --http-addr 127.0.0.1:18080 \
  --grpc-addr 0
```

Stop the CPU server before starting this one on the same port. FP32 MLX paths are available for the native Qwen3.5 profile, the three Laya profiles, GLiClass (`encoder-instruct-label`), and JevK5 (`decoder-logit-qwen35`). Other profiles use the CPU backend.

The [MLX guide](docs/MLX.md) covers build requirements and parity checks. The [model guide](docs/MODELS.md) lists measured performance and memory use. BF16 Qwen execution remains experimental and has not passed the FP32 probability tolerance.

## Models and research

The catalog includes Laya decision encoders, NLI and GLiClass classifiers, Qwen-based decision decoders, and the native `qwen35-state-first` research profile. Use `openkind catalog` for pull names and the [model guide](docs/MODELS.md) for each profile's purpose and limits.

OpenKind's custom model work explores learned decision readouts and training recipes over open Qwen backbones. Its native Rust engine can process shared state once, then branch question and candidate work while isolating attention and recurrent state. The research tests calibration, precision, cache reuse, and whether those changes preserve decisions.

The new mixed-task Qwen3.5 decision LoRA trainer has been checked with tiny models; full 4B training and Mac qualification are still unrun. Fixture parity and throughput measurements establish specific implementation behavior. They do not establish general task accuracy.

| Read more | What it covers |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Rust crates, inference paths, and shared-state execution |
| [Research dossier](docs/RESEARCH.md) | Evidence, related systems, and reproduction limits |
| [Whitepaper](docs/whitepaper/WHITEPAPER.md) | Methods and measured results |
| [Research notebooks](research/README.md) | Experiments, training recipe, and supporting artifacts |
| [Benchmarks](docs/BENCHMARKS.md) | Measurement methods and recorded runs |


## Development

### Build from source

To build from source, install Rust 1.88 or newer and the Protocol Buffers compiler (`protoc`):

```bash
git clone https://github.com/whit3rabbit/openkind.git
cd openkind
cargo build --release --locked -p openkind-cli -p openkind-server
export PATH="$PWD/target/release:$PATH"
openkind version
```

Apple silicon users can enable the optional GPU-accelerated backend with `--features openkind-server/mlx` (see [Apple silicon and MLX](#apple-silicon-and-mlx)).

### Verification

Start with the [architecture guide](docs/ARCHITECTURE.md) and [family integration guide](docs/families/NEW_FAMILY.md). The repository's [agent guide](AGENTS.md#verification) lists the full verification battery:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
env -u RUST_LOG cargo test --workspace
```

Report bugs or propose changes through [GitHub issues](https://github.com/whit3rabbit/openkind/issues).

## License

See the [MIT license](LICENSE). Cargo metadata declares `MIT OR Apache-2.0`. Each model retains its own license, recorded in the [model catalog](registry/v1/catalog.json) and linked manifests.
