# Environment Variables

> Every environment variable openkind reads, grouped by component. Component
> guides own per-flag detail; this page is the aggregated reference for
> operator configuration and TypeSafe System One SDK compatibility.

## Resolution Convention

Where a client setting can come from the environment, resolution is:

1. Explicit argument or builder value.
2. `OPENKIND_*` environment variable.
3. `TYPESAFE_*` environment variable.
4. Built-in default.

Empty and whitespace-only environment values are ignored, matching the
[`typesafe-sdk` Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python).
The `TYPESAFE_*` names exist so scripts written against the official
`typesafe_sdk` package run unchanged against `openkind`; see
[`JEV_COMPATIBILITY.md`](JEV_COMPATIBILITY.md) for the full protocol
compatibility story.

## TypeSafe SDK Compatibility Variables

The official Python SDK reads four public environment variables (v0.7.2,
`src/typesafe_sdk/constants.py`). openkind supports all four:

| Variable | Meaning | Default | Read by |
|---|---|---|---|
| `TYPESAFE_API_KEY` | Bearer token for `/v1/*` | none | `openkindd` auth, `openkind` CLI, Rust client, Python binding |
| `TYPESAFE_BASE_URL` | API root URL | `https://api.typesafe.ai` (Rust client), `http://127.0.0.1:18080` (Python binding) | Rust client, Python binding |
| `TYPESAFE_DEFAULT_MODEL` | Model alias when a request omits `model` | `jev-latest` | Rust client, Python binding |
| `TYPESAFE_LOG_LEVEL` | Binding log level: `debug`, `info`, `warn`, `warning`, `error`, `off` | unset (library default) | Python binding |

Notes:

- The Rust client requires an API key: building without
  `ClientBuilder::api_key`, `OPENKIND_API_KEY`, or `TYPESAFE_API_KEY` fails.
  The Python binding sends `Authorization: Bearer <key>` only when a key
  resolves.
- `openkindd` consults `TYPESAFE_API_KEY` as a fallback for the expected
  token when `--api-key` and `OPENKIND_API_KEY` are unset (precedence:
  `OPENKIND_API_KEY` → `TYPESAFE_API_KEY` → `OPENPICK_API_KEY`). With no key
  anywhere, auth is off.
- The Python binding applies `TYPESAFE_LOG_LEVEL` to its `openkind_client`
  logger once at import and ignores unknown values, mirroring the official
  SDK. The Rust client emits `tracing` events to the host application's
  subscriber and has no log-level variable of its own.
- The TypeScript and Swift clients read no environment variables; all
  settings come from constructor options. Their server wrappers scrub
  `OPENKIND_API_KEY`, `OPENPICK_API_KEY`, and `TYPESAFE_API_KEY` from the
  child process environment and re-set `OPENKIND_API_KEY` when a key is
  configured.

A hosted-SDK-style workflow works as-is against a local daemon:

```bash
export TYPESAFE_BASE_URL=http://localhost:8888
export TYPESAFE_API_KEY=sk-unsloth-YOUR_KEY
```

Each variable also has an `OPENKIND_*` primary name: `OPENKIND_API_KEY`,
`OPENKIND_BASE_URL`, `OPENKIND_DEFAULT_MODEL`, `OPENKIND_LOG_LEVEL`.

## Daemon (`openkindd`)

### Listeners, models, and auth

| Variable | Meaning | Default |
|---|---|---|
| `OPENKIND_HTTP_ADDR` | HTTP bind address | `0.0.0.0:8080` |
| `OPENKIND_GRPC_ADDR` | gRPC bind address; `0`, `off`, `none`, or `disabled` (case-insensitive) disables the listener | `0.0.0.0:9090` |
| `OPENKIND_MODELS` | Comma-separated model aliases to expose | `mock,jev-latest` |
| `OPENKIND_INSTALLED_MODELS` | Comma-separated installed profile names loaded at startup | empty |
| `OPENKIND_MODELS_DIR` | Shared model store directory | platform default (below) |
| `OPENKIND_API_KEY` | Bearer token required for `/v1/*`; unset disables auth | unset |
| `OPENKIND_RATE_LIMIT_RPM` | Per-client-IP request budget per minute on `/v1/*`; `0` disables | `120` |
| `OPENKIND_PLAYGROUND` | Serve the embedded playground and local model controls (`on`/`off`) | `off` |
| `OPENKIND_ARROW` | Serve the unofficial [Arrow bulk endpoint](ARROW.md) (`on`/`off`) | `off` |
| `RUST_LOG` | Log filter, `tracing_subscriber::EnvFilter` syntax | `info` |

### Native Qwen3.5 engine

| Variable | Meaning | Default |
|---|---|---|
| `OPENKIND_QWEN35_ALIASES` | Comma-separated aliases routed to the native engine | `qwen35-native` |
| `OPENKIND_QWEN35_BUNDLE_ROOT` | Offline selected-profile bundle root | unset |
| `OPENKIND_QWEN35_CHECKPOINT_ROOT` | Offline pinned Qwen checkpoint root | unset |
| `OPENKIND_QWEN35_TOKENIZER` | Digest-locked tokenizer JSON path | unset |
| `OPENKIND_QWEN35_CONCURRENCY` | Maximum concurrent native evaluations | `1` |
| `OPENKIND_QWEN35_QUEUE` | Additional queued native requests | `2` |
| `OPENKIND_QWEN35_TIMEOUT_MS` | Queue-inclusive deadline per native evaluation | `600000` |
| `OPENKIND_QWEN35_BACKEND` | `native-cpu`, or `mlx-fp32` on macOS arm64 with the `mlx` feature | `native-cpu` |
| `OPENKIND_QWEN35_EXECUTION` | Execution plan override: `auto`, `repeated-full`, `nested-sequential`, `nested-batched` | `auto` |
| `OPENKIND_QWEN35_MAX_TENSOR_BYTES` | Continuation tensor-payload ceiling per request | unset (policy decides) |
| `OPENKIND_QWEN35_MAX_PROCESS_BYTES` | Process-memory admission ceiling | unset (policy decides) |
| `OPENKIND_QWEN35_SCRATCH_BYTES` | Forward scratch budget added to observed process memory | `1073741824` |
| `OPENKIND_QWEN35_ALLOCATOR_HEADROOM_BYTES` | Allocator headroom added to observed process memory | `536870912` |

### Surveyed-family engines

Each surveyed family takes a `<NAME>_ALIASES` variable selecting aliases in
`--models` and, unless noted, a `<NAME>_MODEL_ROOT` variable pointing at its
read-only artifact root:

| Family token | Variables |
|---|---|
| `DECODER_LETTER` | `_ALIASES`, `_MODEL_ROOT` |
| `ENCODER_NLI` | `_ALIASES`, `_MODEL_ROOT` |
| `ENCODER_INSTRUCT_LABEL` | `_ALIASES`, `_MODEL_ROOT` |
| `KEV` | `_ALIASES`, `_MODEL_ROOT`, plus `OPENKIND_KEV_BASE_ROOT` (base checkpoint) |
| `DECODER_LLM` | `_ALIASES`, `_MODEL_ROOT` |
| `SCHEMA_SCORER` | `_ALIASES`, `_MODEL_ROOT` |
| `ROUTER_SCRIPT` | `_ALIASES`, plus `OPENKIND_ROUTER_SCRIPT_RULES` (rules file; no model root) |
| `QWEN3GUARD` | `_ALIASES`, `_MODEL_ROOT` |
| `WINNOW` | `_ALIASES`, `_MODEL_ROOT`, `_ADAPTER`, `_SIBLINGS` |
| `DECODER_LOGIT_QWEN35` | `_ALIASES`, `_MODEL_ROOT` |
| `LAYA_ENGLISH` | `_ALIASES`, `_MODEL_ROOT` |
| `LAYA_MULTILINGUAL` | `_ALIASES`, `_MODEL_ROOT` |
| `LAYA_TYPED_DECISIONS` | `_ALIASES`, `_MODEL_ROOT` |

Family token variables are prefixed `OPENKIND_`, for example
`OPENKIND_LAYA_ENGLISH_MODEL_ROOT`. Shared admission knobs:
`OPENKIND_FAMILY_CONCURRENCY` (default `1`), `OPENKIND_FAMILY_QUEUE`
(default `2`), `OPENKIND_FAMILY_TIMEOUT_MS` (default `600000`).

### Surveyed-family execution backends

Families with an MLX path take a backend selector: `native-cpu` default, or
`mlx-fp32` on macOS arm64 when the daemon's `mlx` feature is enabled.

| Variable | Meaning | Default |
|---|---|---|
| `OPENKIND_LAYA_BACKEND` | Backend for every served laya alias | `native-cpu` |
| `OPENKIND_ENCODER_INSTRUCT_LABEL_BACKEND` | Backend for served encoder-instruct-label aliases | `native-cpu` |
| `OPENKIND_DECODER_LOGIT_QWEN35_BACKEND` | Backend for served decoder-logit-qwen35 aliases | `native-cpu` |

## CLI (`openkind`)

- `serve` mirrors the daemon flags above: `OPENKIND_HTTP_ADDR` (default
  `0.0.0.0:8080`), `OPENKIND_GRPC_ADDR`, `OPENKIND_MODELS`,
  `OPENKIND_INSTALLED_MODELS`, `OPENKIND_MODELS_DIR`, `OPENKIND_API_KEY`.
- Artifact subcommands (`pull`, `list`, `show`, `rm`) read
  `OPENKIND_MODELS_DIR`.
- `evaluate` and `status` resolve the API key as `--api-key` →
  `OPENKIND_API_KEY` → `TYPESAFE_API_KEY`.
- `NO_COLOR` disables colored output when set (any value, including empty).

## Model Store Locations

`OPENKIND_MODELS_DIR` overrides the store root (an empty value is an
error). Without it, the platform default applies:

| Platform | Default |
|---|---|
| macOS | `$HOME/Library/Application Support/openkind/models` |
| Windows | `%APPDATA%\openkind\models`, falling back to `%USERPROFILE%\AppData\Roaming\openkind\models` |
| Linux and other | `$XDG_DATA_HOME/openkind/models`, or `$HOME/.local/share/openkind/models` when `XDG_DATA_HOME` is unset |

## Proxy Variables

The Rust client, `openkindd` egress, and the Python binding honor the
standard proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`,
`NO_PROXY`, and lowercase forms) through their HTTP stacks. The Rust client
additionally bypasses its localhost fast path whenever a proxy variable is
set so requests still traverse the proxy.

## Deprecated Variables

`OPENPICK_HTTP_ADDR`, `OPENPICK_GRPC_ADDR`, and `OPENPICK_API_KEY` are
pre-rename fallbacks accepted by `openkindd` when the `OPENKIND_*` variable
is unset. Setting both to conflicting values fails startup; remove the
`OPENPICK_*` names after migrating. The binding wrappers scrub
`OPENPICK_API_KEY` alongside `OPENKIND_API_KEY` from child processes.

## Test and Offline Evidence Variables

- `OPENKIND_TEST_URL`, `OPENKIND_TEST_API_KEY`, and `OPENKIND_TEST_BINARY`
  gate live integration tests in the bindings and Rust client;
  [`../bindings/README.md`](../bindings/README.md) owns the procedure.
- The family `OPENKIND_<NAME>_MODEL_ROOT` variables also gate offline parity
  tests in `crates/openkind-backends`; see that crate's `AGENTS.md`. Tests
  and builds never download model assets; fixtures are vendored and
  digest-checked.

## Benchmark Dataset Variables

Read by `openkind-bench dataset` commands only (see
[`BENCHMARKS.md`](BENCHMARKS.md#dataset-accuracy-evaluation)); no test or
build path consults them.

| Variable | Meaning | Default |
|---|---|---|
| `OPENKIND_DATASETS_DIR` | Evaluation-dataset cache root (installs, content-addressed blobs, locks) | platform default: `~/Library/Application Support/openkind/datasets` (macOS), `%APPDATA%/openkind/datasets` (Windows), `${XDG_DATA_HOME:-~/.local/share}/openkind/datasets` (Linux) |
| `HF_TOKEN` | Hugging Face bearer token for gated dataset downloads | unset |
| `HF_TOKEN_PATH` | Explicit path to a Hugging Face token file | unset |
| `HF_HOME` | Locates the `token` file written by `hf auth login` when `HF_TOKEN`/`HF_TOKEN_PATH` are unset | `~/.cache/huggingface` |

Token resolution order is `HF_TOKEN`, then `HF_TOKEN_PATH`, then the
`hf auth login` token file; downloads are anonymous when none is present.
The token value is never logged — only its source.

## Verification

Client-side `TYPESAFE_*` resolution is covered by
`crates/openkind-client/tests/sdk_parity_config.rs` (Rust) and
`bindings/python/tests/` (Python). After editing this page, run
`git diff --check -- docs/` for whitespace issues.
