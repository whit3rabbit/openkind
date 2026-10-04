# HTTP clients and local server wrappers

The TypeScript, Python, and Swift packages let applications submit typed
decisions to a running `openkindd`. They use the System One HTTP API that the
`openkind` CLI uses, without launching the CLI or loading model weights.
Each package also has a wrapper that starts and stops a local `openkindd`
process, then provides a client for it. Build or obtain `openkindd` separately.
The wrappers do not embed the Rust server or expose a foreign-function ABI.

## Quickstart

From the repository root, start the mock engine without model artifacts:

```bash
cargo run -p openkind-server --bin openkindd -- \
  --http-addr 127.0.0.1:18080 \
  --grpc-addr 0 \
  --models mock
```

Use `http://127.0.0.1:18080` as the client base URL. Authentication is off
for this example; omit the client's API key. Each package has a runnable example:

| Language | Package guide | Requirement |
|---|---|---|
| TypeScript | [typescript](typescript/README.md) | Node 18+ for the client; Node process APIs for the server wrapper |
| Python | [python](python/README.md) | Python 3.11+ |
| Swift | [swift](swift/README.md) | SwiftPM on macOS 12+ or iOS 15+; server wrapper on macOS only |

## API keys

Key checking is opt-in. Define a key with the daemon's `--api-key` flag or
`OPENKIND_API_KEY`, then supply the same key to each client. The
[`openkind keygen` command](../crates/openkind-cli/README.md#keygen) prints a
random key for shell configuration, including PowerShell.

The bindings also generate keys without invoking the CLI:

| Language | Generator | Import |
|---|---|---|
| TypeScript (Node) | `generateApiKey()` | `@openkind/client/server` |
| Python | `generate_api_key()` | `openkind_client` |
| Swift (macOS and iOS) | `try OpenKindAPIKey.generate()` | `OpenKind` |

Each generator uses 32 bytes of operating-system randomness and returns `ok_`
followed by 64 lowercase hexadecimal characters. Generation does not start a
server, enable authentication, or save the key. Pass the returned value to the
local server wrapper's `apiKey` or `api_key` option to enable authentication;
the wrapper configures its client with the same key. Omit that option to leave
authentication off. Configured wrapper keys must be nonempty visible ASCII
without whitespace and are checked before launching the daemon.

For an existing daemon, configure its key separately and reuse that exact
value in the client. A newly generated client key will not match a daemon
already configured with another key. Keys remain caller-managed and are not
saved to `~/.openkind` or another configuration file.

## Shared contract

All three expose evaluation (`POST /v1/systemone`), model listing
(`GET /v1/models`), and health (`GET /health`). The health probe omits bearer
authentication.

An explicit `evaluate` request carries `state`, `model`, and a map of question
IDs to tagged `noul`, `choice`, or `score` questions. The `systemOne` or
`system_one` convenience method supplies the configured default model because
`model` is required on the wire.

Successful results carry the response body and the
`x-typesafe-request-id` header. HTTP failures expose the status, error code,
message, and request ID when the server provides them.

The clients check answer IDs and types against the submitted questions. Choice
selections and probability keys must match the submitted criteria. Malformed
probabilities and Score legends fail validation. The
[OpenAPI contract](../crates/openkind-api/openapi.yaml) owns the wire shapes.

## Local server lifecycle

The server wrappers bind HTTP to an explicit `127.0.0.1` port, disable gRPC,
and pass a configured API key through the child environment. Inherited API-key
variables are cleared so an omitted wrapper key keeps authentication off.
They wait for `/health` before returning a client. `stop()` terminates only the process the wrapper
started. They reject an occupied port before launch and do not control an
already running daemon. For other bind addresses, gRPC, or deployment, start
`openkindd` yourself and use a client with its URL.

For a native Qwen Choice request, include a non-empty `__none__` option in
the criteria so semantic-none mass remains explicit. A decision does not
authorize an action. The caller determines eligible actions and verifies any
action it takes.

## Verify the packages

Run from the repository root:

```bash
(cd bindings/typescript && npm install && npm test)
(cd bindings/python && python3 -m unittest discover -s tests)
(cd bindings/swift && swift test)
```

All three packages also have opt-in live daemon tests for the covered HTTP
routes. Start the daemon above, then run:

```bash
(cd bindings/typescript && OPENKIND_TEST_URL=http://127.0.0.1:18080 npm test)
(cd bindings/python && OPENKIND_TEST_URL=http://127.0.0.1:18080 python3 -m unittest discover -s tests)
(cd bindings/swift && OPENKIND_TEST_URL=http://127.0.0.1:18080 swift test)
```

When the daemon has authentication enabled, set `OPENKIND_TEST_API_KEY` to
its configured key in those commands.

Set `OPENKIND_TEST_BINARY` to an absolute `openkindd` path in those commands
to also test each server wrapper against the real binary. The wrappers use
their own free loopback ports and the mock engine.

## Scope

Resumable JSONL jobs are managed locally by
[`openkind batch`](../crates/openkind-cli/README.md#batch). Job directories,
row ranges, stop/status, and resume are CLI features; these packages expose
the daemon's HTTP requests.

These packages are not published to npm, PyPI, or a Swift package registry.
The clients cover the canonical HTTP evaluation, models, and health routes.
They do not wrap `/v1/system_one`, `/metrics`, or gRPC, and do not include the
[Rust client's](../crates/openkind-client/README.md) retry policy,
OpenRouter adaptation, or Cloudflare envelope. TypeScript and Swift JSON
numbers use IEEE 754 doubles, so general state integers above `2^53 - 1`
cannot be represented exactly. Local contract and mock-daemon checks establish
the covered HTTP behavior, not full API parity or native model quality.

## License

The repository is [MIT licensed](../LICENSE).
