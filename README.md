# openkind

`openkind` is a Rust decision engine that scores candidate answers and returns typed `Noul` (yes/no probability), `Choice`, and `Score` results through the Jev System One API.  It reuses a state prefix across questions and assembles JSON responses directly, without generating answer text token by token.

The current native model is a pinned Qwen3.5-4B-Base research profile.  CPU implementation parity has passed its frozen fixtures, but reviewed task quality and release approval remain open.  See the [roadmap](docs/ROADMAP.md) before using its decisions in an application.

## Install

**From source, available now.**  You need Rust 1.88 or newer and `protoc`.  The build creates both `openkind` (CLI) and `openkindd` (daemon); keep them together on `PATH` because `openkind serve` starts `openkindd`.

```bash
git clone https://github.com/whit3rabbit/openkind.git
cd openkind
cargo build --release --locked -p openkind-cli -p openkind-server
export PATH="$PWD/target/release:$PATH"
openkind version
```

Homebrew and crates.io distribution are planned.  Once published, the intended commands are `brew install whit3rabbit/tap/openkind` or `cargo install openkind-cli` plus `cargo install openkind-server`.  Until then, use the source build above.

## Run a model

The curated catalog has one Rust-loadable profile.  `pull` downloads its pinned checkpoint, tokenizer, and readout bundle, then verifies the files.  This is an explicit multi-gigabyte download; normal builds and tests do not fetch weights.

```bash
openkind catalog
openkind pull qwen35-state-first:a047d6802c3f06f085b8
openkind list
openkind serve \
  --installed-models qwen35-state-first:a047d6802c3f06f085b8 \
  --http-addr 127.0.0.1:18080 \
  --grpc-addr 0
```

In another terminal, submit a request.  The `model` field names the installed profile explicitly; the daemon also registers mock aliases by default.

```bash
curl -sS http://127.0.0.1:18080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "state": "A customer says their payment failed twice and asks for help.",
  "model": "qwen35-state-first:a047d6802c3f06f085b8",
  "questions": {
    "billing": {
      "type": "noul",
      "instructions": "Is this a billing issue?",
      "criteria": {
        "true": "The issue concerns a payment or charge.",
        "false": "The issue does not concern a payment or charge."
      }
    }
  }
}
JSON
```

The response contains a `billing` answer with a `noul` probability.  For native `Choice` questions, include a non-empty `__none__` criterion so the model can report that none of the offered options fit.

Use `openkind status --server http://127.0.0.1:18080` to see the aliases active in the daemon.  A new pull becomes available after restarting it or explicitly loading it in the playground.  The [model registry guide](docs/MODEL_REGISTRY.md) covers the store location and verification lifecycle.

To try the wire API without downloading a model, stop the daemon above and use the checked-in fixture:

```bash
openkind serve --models jev-latest --http-addr 127.0.0.1:18080 --grpc-addr 0
# In another terminal:
openkind evaluate examples/04_mixed.json --server http://127.0.0.1:18080 --pretty
```

`jev-latest` runs the mock engine in this setup.  Its answers check integration behaviour, not model quality.

## Playground

The daemon ships an embedded web playground for poking at the wire API and eyeballing speed locally.  It is off by default. Evaluation uses `POST /v1/systemone`; authentication and rate limits still apply.

### Quick start

To start the playground with mock models (no weights download required):

```bash
openkind playground
```

This spawns a loopback-only daemon at `http://127.0.0.1:8080`, disables the gRPC listener and rate limits, and automatically opens the playground in your browser.

### Start and load a model from CLI

To launch the playground with an installed model pre-loaded and ready for evaluation, pass `--installed-models`:

```bash
# Pull the model if not already downloaded:
openkind pull qwen35-state-first:a047d6802c3f06f085b8

# Launch the playground with the model loaded at startup:
openkind playground --installed-models qwen35-state-first:a047d6802c3f06f085b8
```

You can also dynamically load or unload any installed profile at runtime from the **Local Models** panel in the web interface without restarting. See the [model lifecycle](docs/MODEL_REGISTRY.md) for limits.

### Options and existing daemons

```bash
# Print the URL without opening a browser:
openkind playground --no-open

# Bind to a custom port or specify mock models:
openkind playground --http-addr 127.0.0.1:18080 --models mock,jev-latest

# Enable the playground route on a manually started daemon:
openkindd --playground on
```

`openkind playground` binds `127.0.0.1` only, disables the gRPC listener, and turns off the per-IP rate limit so benchmark bursts are not throttled; Ctrl-C stops it.  If a daemon is already listening on the target address it checks that the playground is enabled before opening it.  API keys stay in page memory until the page closes. The page offers preset and saved requests (saved examples live in the browser's `localStorage`), a form or raw-JSON editor, typed answer cards with probability bars, and a benchmark tab that measures client-side round trips across selected models.  Those timings are informational; recorded measurements come from `openkind-bench score` ([benchmarks](docs/BENCHMARKS.md)).

## Features

- **Typed decisions.**  `Noul` returns a yes/no probability, `Choice` returns an offered option and its distribution, and `Score` uses an ordered rubric.  The [wire contract](crates/openkind-api/openapi.yaml) defines the request and response shapes.
- **Shared state execution.**  The native runner can prefill a document once, then branch question and candidate work from it.  It tracks attention, DeltaNet, and convolution state when branching.  The [architecture guide](docs/ARCHITECTURE.md) explains the execution plans.
- **Local model lifecycle.**  The CLI lists curated profiles, verifies downloads, inspects local installations, and starts explicitly selected models.  There is no implicit model download or hot reload.
- **Service and clients.**  The daemon serves HTTP and gRPC.  Rust, TypeScript, Python, and Swift clients are available in this repository; the latter three call the HTTP API.  Clients check returned answers against submitted question IDs and options.
- **Optional Apple MLX backend.**  It is feature gated and has separate parity and performance evidence.  The normal source build above uses the CPU backend.  See the [MLX guide](docs/MLX.md) for its current limits.

The native profile's fixture parity and measured execution behaviour do not establish general classification accuracy or production readiness.  See [benchmarks](docs/BENCHMARKS.md) for what each run measures.

## Set up a client

The Python client requires Python 3.11 or newer.  It installs from this checkout and calls the running daemon over HTTP.  Start the model as shown above, then, from the repository root:

```bash
python3 -m pip install ./bindings/python
python3 - <<'PY'
from openkind_client import Client

client = Client(base_url="http://127.0.0.1:18080", timeout=600)
result = client.system_one(
    "A customer says their payment failed twice and asks for help.",
    {"billing": {"type": "noul", "instructions": "Is this a billing issue?"}},
    model="qwen35-state-first:a047d6802c3f06f085b8",
)
print(result.data["answers"]["billing"])
PY
```

For Rust, use [`openkind-client`](crates/openkind-client/README.md) and set its base URL, default model, and request timeout for local CPU inference.  The [bindings guide](bindings/README.md) has TypeScript, Python, and Swift setup and local server wrappers.

These clients submit requests; they do not load weights in the application process.  Check a model's output against your own allowed actions before acting on it.

## Crates

| Crate | Role |
|---|---|
| [`openkind-core`](crates/openkind-core/README.md) | Jev wire types, validation, and schema definitions. |
| [`openkind-engine`](crates/openkind-engine/README.md) | Engine interface, registry, mock engine, and execution profiles. |
| [`openkind-runtime`](crates/openkind-runtime/README.md) | Hardware limits, branch state, scheduling, and run evidence. |
| [`openkind-backends`](crates/openkind-backends/README.md) | Native Qwen execution and decision readouts. |
| [`openkind-model-store`](crates/openkind-model-store/Cargo.toml) | Curated manifests and verified local model installations. |
| [`openkind-api`](crates/openkind-api/README.md) | HTTP and gRPC routes. |
| [`openkind-server`](crates/openkind-server/README.md) | `openkindd` process, model registration, and service lifecycle. |
| [`openkind-cli`](crates/openkind-cli/README.md) | `openkind` commands for models, requests, status, and serving. |
| [`openkind-client`](crates/openkind-client/README.md) | Async Rust HTTP client. |
| [`openkind-bench`](crates/openkind-bench/Cargo.toml) | Native request-path scoring and timing harness. |
| [`openkind-gen-schemas`](crates/openkind-gen-schemas/README.md) | JSON Schema generator. |
| [`openkind-proto`](proto/Cargo.toml) | Protobuf service and message definitions. |

## Design and evidence

Start with the [research dossier](docs/RESEARCH.md) for the experiment sequence and the [whitepaper](docs/whitepaper/WHITEPAPER.md) for methods and measured results.  The [working paper](docs/whitepaper/WORKING_PAPER.md) records newer open questions.

Developers can use the [architecture guide](docs/ARCHITECTURE.md), [family integration guide](docs/families/NEW_FAMILY.md), and [roadmap](docs/ROADMAP.md) to trace design choices, supported profiles, and remaining gates.

## License

See the [MIT license text](LICENSE).  Cargo metadata currently declares `MIT OR Apache-2.0`.
