# openkind

`openkind` runs a Typesafe server and cli (Jev) with local hosted models for the server.

I highly recommend using: https://github.com/ollaya-dev/ollaya

We seem to have had the same idea when I started working and their public release is a lot more polished.

OpenKind researches a custom Qwen3.5 decision model and a Rust inference engine that shares input processing across questions.

[![CI](https://github.com/whit3rabbit/openkind/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/openkind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Install

Install the `openkind` CLI and `openkindd` server using Homebrew, Cargo, or prebuilt binaries. Keep both binaries on `PATH`: `openkind serve` starts the server. macOS, Linux, and Windows (`x86_64`) are supported host platforms; the standard distributions run models on the CPU, and Apple silicon users who want GPU acceleration can [build from source with MLX](#apple-silicon-and-mlx).

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

Download prebuilt binary archives for macOS (Apple silicon or Intel), Linux (`x86_64` musl), and Windows (`x86_64` MSVC) from [GitHub Releases](https://github.com/whit3rabbit/openkind/releases). Unpack the tarball — or the zip on Windows — and add `openkind` and `openkindd` to your `PATH`.

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

## Choose a local model

For an Apple silicon Mac, start with **Laya Typed Decisions** (`laya:typed-decisions`) for mixed Choice, Noul, and Score requests. It balances memory, speed, and the two classification panels below. **GLiClass** (`encoder-instruct-label`) is the smaller, faster option for label classification.

Keep Laya Choice questions under about 20 options, and validate its confidence on your own data before using probability thresholds.

For CPU-only machines, try **Kev 0.6B** for short Choice requests. **DistilBERT NLI** (`encoder-nli`) uses much less memory, with lower accuracy on these panels. These are models to try on an 8-16 GB machine, not capacity guarantees: the measurements came from a 36 GiB Mac. Leave memory for the OS and other applications.

The first three rows are suggested starting points. The rest are grouped by measured MLX, CPU, and GGUF execution, sorted by catalog loader name, followed by auxiliary models. Model links lead to Hugging Face; short pull aliases appear where the catalog defines one. Use the [model guide](docs/MODELS.md#all-models) for full pinned pull names.

Numbers come from the [local suite](benchmarks/2026-10-01-local-model-suite/): Apple M4 Max, 14 cores, 36 GiB RAM, macOS arm64. Timing uses one warm repetition of the same 12-decision, four-request smoke fixture unless marked. Model loading is excluded. Peak RSS is measured process memory in decimal GB, not download size, dedicated VRAM, or a maximum-context memory estimate.

**Accuracy is SST-2 / AG News, 50 eval rows each**, using pinned zero-shot templates and the injected `__none__` option. These small public-dataset samples establish limited task evidence, not general accuracy or release qualification. Small percentage differences are not stable rankings. Accuracy links open reports with confidence intervals; speed links open timing summaries.

**Context means the current runtime's encoded input budget**, including state, instructions, and options unless a state-only limit is shown. It is often smaller than the checkpoint's advertised window. Laya uses 512 or 1,024 tokens despite an 8,192-token backbone. Input tokens/s is harness-reported input throughput; OpenKind does not generate output tokens.

The measured backend is bold. CPU is available for every checkpoint row. **MLX** requires an Apple silicon source build.

`CUDA*` and `ONNX*` are optional candidates without comparable hardware measurements or completed backend qualification here. ONNX requires a separate pinned export and can use CPU, CUDA, or Linux ROCm providers. See the [CUDA](docs/CUDA.md), [ONNX](docs/ONNX.md), and [ROCm](docs/ROCM.md) guides. GGUF describes the weights, not GPU acceleration.

| Model / alias (Hugging Face) | Execution support | Peak RSS (GB) | Runtime context (tokens) | Decisions/s | Input tokens/s | Accuracy: SST-2 / AG News |
|---|---|---:|---|---:|---:|---|
| [`laya:typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | CPU, **MLX**, CUDA*, ONNX* | 2.12 | 1,024 | [34.74](benchmarks/2026-10-01-local-model-suite/performance/laya-typed-decisions/laya-typed-decisions-mlx-fp32/summary-laya-typed-decisions-mlx-fp32.json) | 7,116 | [92%](benchmarks/2026-10-01-local-model-suite/quality/laya-typed-decisions/sst2/dataset-eval-sst2-eval.json) / [90%](benchmarks/2026-10-01-local-model-suite/quality/laya-typed-decisions/ag_news/dataset-eval-ag_news-eval.json) |
| [GLiClass (`encoder-instruct-label`)](https://huggingface.co/knowledgator/gliclass-modern-base-v3.0) | CPU, **MLX**, CUDA*, ONNX* | 1.01 | 1,024 | [72.32](benchmarks/2026-10-01-local-model-suite/performance/encoder-instruct-label/encoder-instruct-label-mlx-fp32/summary-encoder-instruct-label-mlx-fp32.json) | 15,536 | [92%](benchmarks/2026-10-01-local-model-suite/quality/encoder-instruct-label/sst2/dataset-eval-sst2-eval.json) / [78%](benchmarks/2026-10-01-local-model-suite/quality/encoder-instruct-label/ag_news/dataset-eval-ag_news-eval.json) |
| [`kev`](https://huggingface.co/jaredpalmer/kev-0.6b) | **CPU**, CUDA* | 4.21 | 1,024 (state 384) | [5.48](benchmarks/2026-10-01-local-model-suite/performance/kev/kev/summary-kev.json) | 844 | [86%](benchmarks/2026-10-01-local-model-suite/quality/kev/sst2/dataset-eval-sst2-eval.json) / [88%](benchmarks/2026-10-01-local-model-suite/quality/kev/ag_news/dataset-eval-ag_news-eval.json) |
| [`jevk5:4b`](https://huggingface.co/alibiserikbay/JevK5) | CPU, **MLX**, CUDA* | 14.75 | 512 | [0.55](benchmarks/2026-10-01-local-model-suite/performance/decoder-logit-qwen35/decoder-logit-qwen35-mlx-fp32/summary-decoder-logit-qwen35-mlx-fp32.json) | 149 | [98%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen35/sst2/dataset-eval-sst2-eval.json) / [86%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen35/ag_news/dataset-eval-ag_news-eval.json) |
| [`laya:en`](https://huggingface.co/convaiinnovations/laya) | CPU, **MLX**, CUDA*, ONNX* | 2.12 | 512 | [33.42](benchmarks/2026-10-01-local-model-suite/performance/laya-english/laya-english-mlx-fp32/summary-laya-english-mlx-fp32.json) | 6,845 | [82%](benchmarks/2026-10-01-local-model-suite/quality/laya-english/sst2/dataset-eval-sst2-eval.json) / [86%](benchmarks/2026-10-01-local-model-suite/quality/laya-english/ag_news/dataset-eval-ag_news-eval.json) |
| [`laya:multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | CPU, **MLX**, CUDA*, ONNX* | 2.94 | 1,024 | [71.11](benchmarks/2026-10-01-local-model-suite/performance/laya-multilingual/laya-multilingual-mlx-fp32/summary-laya-multilingual-mlx-fp32.json) | 14,494 | [30%](benchmarks/2026-10-01-local-model-suite/quality/laya-multilingual/sst2/dataset-eval-sst2-eval.json) / [88%](benchmarks/2026-10-01-local-model-suite/quality/laya-multilingual/ag_news/dataset-eval-ag_news-eval.json) |
| [`plumb:4b`](https://huggingface.co/crh225/plumb-4b) | CPU, **MLX**, CUDA* | 11.99 | 16,384 | [0.54](benchmarks/2026-10-01-local-model-suite/performance/plumb-4b/plumb-4b-mlx-fp32/summary-plumb-4b-mlx-fp32.json) (777) | 146 | [96%](benchmarks/2026-10-01-local-model-suite/quality/plumb-4b/sst2/dataset-eval-sst2-eval.json) / [86%](benchmarks/2026-10-01-local-model-suite/quality/plumb-4b/ag_news/dataset-eval-ag_news-eval.json) |
| [`qwen35-state-first`](https://huggingface.co/cowWhySo/OpenKind-Qwen3.5-4B-StateFirst) | CPU, **MLX**, CUDA* | 12.31 | 1,792 | [0.33](benchmarks/2026-10-01-local-model-suite/performance/qwen35-state-first/qwen35-mlx-fp32/summary-qwen35-mlx-fp32.json) | 36 | [2%](benchmarks/2026-10-01-local-model-suite/quality/qwen35-state-first/sst2/dataset-eval-sst2-eval.json) / [70%](benchmarks/2026-10-01-local-model-suite/quality/qwen35-state-first/ag_news/dataset-eval-ag_news-eval.json) |
| [`clef:flash`](https://huggingface.co/Cloudflare/clef-flash) | **CPU** (BF16 weights, FP32 compute) | 15.59 | 16,384 | [0.22](benchmarks/2026-10-01-local-model-suite/performance/clef-flash/clef-flash/summary-clef-flash.json) | 37 | [96%](benchmarks/2026-10-01-local-model-suite/quality/clef-flash/sst2/dataset-eval-sst2-eval.json) / [90%](benchmarks/2026-10-01-local-model-suite/quality/clef-flash/ag_news/dataset-eval-ag_news-eval.json) |
| [`decider:4b`](https://huggingface.co/Mapika/decider-4b) | **CPU**, CUDA* | 8.05 | 32,768 state | [0.18](benchmarks/2026-10-01-local-model-suite/performance/decider-4b/decider-4b/summary-decider-4b.json) | 36 | [96%](benchmarks/2026-10-01-local-model-suite/quality/decider-4b/sst2/dataset-eval-sst2-eval.json) / [90%](benchmarks/2026-10-01-local-model-suite/quality/decider-4b/ag_news/dataset-eval-ag_news-eval.json) |
| [`decoder-logit-letter`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | **CPU**, CUDA*, ONNX* | 3.34 | 8,192 | [6.63](benchmarks/2026-10-01-local-model-suite/performance/decoder-logit-letter/decoder-letter/summary-decoder-letter.json) | 1,794 | [72%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-letter/sst2/dataset-eval-sst2-eval.json) / [72%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-letter/ag_news/dataset-eval-ag_news-eval.json) |
| [`decoder-logit-qwen3-06b`](https://huggingface.co/Qwen/Qwen3-0.6B) | **CPU**, CUDA*, ONNX* | 4.21 | 512 | [3.55](benchmarks/2026-10-01-local-model-suite/performance/decoder-logit-qwen3-06b/decoder-logit-qwen3-06b/summary-decoder-logit-qwen3-06b.json) (777) | 967 | [88%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen3-06b/sst2/dataset-eval-sst2-eval.json) / [72%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen3-06b/ag_news/dataset-eval-ag_news-eval.json) |
| [`decoder-logit-qwen3-17b`](https://huggingface.co/Qwen/Qwen3-1.7B) | **CPU**, CUDA*, ONNX* | 11.46 | 512 | [2.09](benchmarks/2026-10-01-local-model-suite/performance/decoder-logit-qwen3-17b/decoder-logit-qwen3-17b/summary-decoder-logit-qwen3-17b.json) | 570 | [96%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen3-17b/sst2/dataset-eval-sst2-eval.json) / [6%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen3-17b/ag_news/dataset-eval-ag_news-eval.json) |
| [`qwen3:4b`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | **CPU**, CUDA*, ONNX* | 22.88 | 512 | [0.85](benchmarks/2026-10-01-local-model-suite/performance/decoder-logit-qwen3-4b/decoder-logit-qwen3-4b/summary-decoder-logit-qwen3-4b.json) | 230 | [98%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen3-4b/sst2/dataset-eval-sst2-eval.json) / [86%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-qwen3-4b/ag_news/dataset-eval-ag_news-eval.json) |
| [DistilBERT NLI (`encoder-nli`)](https://huggingface.co/typeform/distilbert-base-uncased-mnli) | **CPU**, CUDA*, ONNX* | 0.56 | 512 | [26.04](benchmarks/2026-10-01-local-model-suite/performance/encoder-nli/encoder-nli/summary-encoder-nli.json) | 5,306 | [74%](benchmarks/2026-10-01-local-model-suite/quality/encoder-nli/sst2/dataset-eval-sst2-eval.json) / [64%](benchmarks/2026-10-01-local-model-suite/quality/encoder-nli/ag_news/dataset-eval-ag_news-eval.json) |
| [`qwen3guard`](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) | **CPU**, CUDA*, ONNX* | 4.21 | 8,192 | [5.12](benchmarks/2026-10-01-local-model-suite/performance/qwen3guard/qwen3-guard/summary-qwen3guard.json) | 949 | n/a (safety head) |
| [`schema-scorer`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | **CPU**, CUDA*, ONNX* | 0.26 | 512 | [12.19](benchmarks/2026-10-01-local-model-suite/performance/schema-scorer/schema-scorer/summary-schema-scorer.json) | 5,260 | [50%](benchmarks/2026-10-01-local-model-suite/quality/schema-scorer/sst2/dataset-eval-sst2-eval.json) / [16%](benchmarks/2026-10-01-local-model-suite/quality/schema-scorer/ag_news/dataset-eval-ag_news-eval.json) |
| [`strands-decider-2b`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) | **CPU**, CUDA* | 9.15 | 4,096 | [0.42](benchmarks/2026-10-01-local-model-suite/performance/strands-decider-2b/strands-decider-2b/summary-strands-decider-2b.json) | 117 | [96%](benchmarks/2026-10-01-local-model-suite/quality/strands-decider-2b/sst2/dataset-eval-sst2-eval.json) / [88%](benchmarks/2026-10-01-local-model-suite/quality/strands-decider-2b/ag_news/dataset-eval-ag_news-eval.json) |
| [`von:1.1`](https://huggingface.co/wfzyx/von) | **CPU**, CUDA*, ONNX* | 1.81 | 8,192 | [2.43](benchmarks/2026-10-01-local-model-suite/performance/von/von/summary-von.json) | 534 | [94%](benchmarks/2026-10-01-local-model-suite/quality/von/sst2/dataset-eval-sst2-eval.json) / [66%](benchmarks/2026-10-01-local-model-suite/quality/von/ag_news/dataset-eval-ag_news-eval.json) |
| [`winnow`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (in-house LoRA router) | **CPU**, CUDA* | n/a | 8,192 | n/a (routing pass) | n/a | n/a (router) |
| [`winnow:e4b`](https://huggingface.co/EldanRing/Winnow-E4B) | **CPU**, CUDA* | 8.42 | 8,192 | [0.10](benchmarks/2026-10-01-local-model-suite/performance/winnow-e4b/winnow-e4b/summary-winnow-e4b.json) | 27 | [94%](benchmarks/2026-10-01-local-model-suite/quality/winnow-e4b/sst2/dataset-eval-sst2-eval.json) / [80%](benchmarks/2026-10-01-local-model-suite/quality/winnow-e4b/ag_news/dataset-eval-ag_news-eval.json) |
| [`clef:27b`](https://huggingface.co/bartowski/Cloudflare_clef-GGUF) (Q4_K_M + [head](https://huggingface.co/Cloudflare/clef)) | CPU | pending | 16,384 | pending | pending | pending / pending |
| [`clef:flash-gguf`](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF) (Q4_K_M + [head](https://huggingface.co/Cloudflare/clef-flash)) | CPU | pending | 16,384 | pending | pending | [96%](benchmarks/2026-10-01-local-model-suite/quality/clef-flash-gguf/sst2/dataset-eval-sst2-eval.json) / pending |
| [`decoder-logit-llm`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (GGUF Q8_0) | **CPU**, CUDA* | 1.56 | 8,192 | [0.44](benchmarks/2026-10-01-local-model-suite/performance/decoder-logit-llm/decoder-llm/summary-decoder-llm.json) | 120 | [72%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-llm/sst2/dataset-eval-sst2-eval.json) / [66%](benchmarks/2026-10-01-local-model-suite/quality/decoder-logit-llm/ag_news/dataset-eval-ag_news-eval.json) |
| [BGE-small (`encoder-embedding`)](https://huggingface.co/BAAI/bge-small-en-v1.5) | CPU, **MLX** | pending | 256 | [289.98 emb/s](benchmarks/2026-10-01-local-model-suite/performance/encoder-embedding-mlx.json) | n/a | n/a (embedder) |
| [`router-script`](docs/families/router-script.md) | Rules only | n/a | n/a | n/a | n/a | n/a (router) |

- `(777)` marks the 777-decision shape fixture for Plumb and Qwen3 0.6B. Those speed and memory rows are not directly comparable to the 12-decision runs. Plumb accuracy was measured on CPU, while its timing uses MLX. The native state-first timing uses `repeated_full`; earlier scheduler-selected results measure a different execution strategy.
- BGE is a proxy-cache embedding component, measured separately with 30 calls after five warmups. Its rate is embeddings/s. Winnow and `router-script` compose other engines; a routing-pass rate does not measure the complete decision path.
- Pending cells have no completed, checked-in measurement. GGUF can reduce weight storage without improving these CPU kernels' speed; do not transfer Clef BF16 memory or accuracy to a quantized profile.
- Laya Multilingual's 30% SST-2 result and the native state-first profile's 2% result make them poor defaults for that task. For wide-label classification, the suite's [Banking77](benchmarks/2026-10-01-local-model-suite/quality/clef-flash/banking77/dataset-eval-banking77-eval.json) and [CLINC150](benchmarks/2026-10-01-local-model-suite/quality/clef-flash/clinc150/dataset-eval-clinc150-eval.json) panels favor Clef Flash (98% / 94%, 50 rows each), at substantial CPU latency and memory cost. Test your actual labels before choosing a larger model.

The [benchmark guide](docs/BENCHMARKS.md#dataset-accuracy-evaluation) explains scoring, calibration metrics, and evidence limits. Public model-card and leaderboard scores use different harnesses and are not included in this table.

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
