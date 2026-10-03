# openkind-cli

> Command-line client for the `openkind` decision engine.

`openkind-cli` provides the `openkind` binary for validating request files offline, evaluating them against a running daemon, launching `openkindd` and the web playground, generating API keys, checking daemon status, and managing curated model installations.

## Installation & Build

### Homebrew (macOS & Linux)

```bash
brew install whit3rabbit/tap/openkind
```

### Cargo

Install from crates.io with Rust 1.88 or newer:

```bash
cargo install --locked openkind-cli
```

### Build from source

```bash
cargo build --release -p openkind-cli
# Binary produced at target/release/openkind
```

## Subcommands

### `inspect`

Validates a JSON request file against the Jev request schema:

```bash
openkind inspect examples/01_noul.json
# Output: examples/01_noul.json ✓ (model=jev-latest, 1 questions)
```

Returns exit code `0` on success, non-zero with error context if JSON is malformed or violates schema validation rules (e.g. empty questions, missing instructions).

### `evaluate`

POSTs a JSON request file to a running `openkindd` instance. Pass `-` as the file to read the request from stdin. Successful responses are JSON on stdout by default, so they can be piped directly to tools such as `jq`:

```bash
openkind evaluate examples/04_mixed.json | jq '.answers'
openkind evaluate examples/04_mixed.json --pretty
openkind evaluate examples/04_mixed.json --format text --verbose
```

Flags:
- `--server <URL>`: Daemon base URL (default: `http://127.0.0.1:8080`).
- `--format json|text`: JSON is the default. Text prints one row per question.
- `--pretty`: Pretty-print JSON output.
- `--verbose`: Add available answer probabilities and request timing to text output. For JSON output, timing and token counts go to stderr.
- `--api-key <KEY>`: Bearer key for protected endpoints. `OPENKIND_API_KEY` and `TYPESAFE_API_KEY` are also read.

HTTP status and error details go to stderr. Noul text rows show their Noul value and label confidence as not provided.

### `serve`

Launches `openkindd`, which must be built and on `PATH`. On Unix the CLI process replaces itself with the daemon, so shutdown signals reach the daemon directly:

```bash
openkind serve --installed-models qwen35-state-first:a047d6802c3f06f085b8
```

Flags:
- `--http-addr <ADDR>`: Address to bind the HTTP server on (default: `0.0.0.0:8080`).
- `--grpc-addr <ADDR>`: Address to bind the gRPC server on (default: `0.0.0.0:9090`).
- `--models <ALIASES>`: Comma-separated model aliases to expose (default: `mock,jev-latest`).
- `--installed-models <MODELS>`: Comma-separated installed models to load at daemon startup.
- `--models-dir <PATH>`: Directory shared by model commands and the daemon.
- `--api-key <KEY>`: Bearer token required for `/v1/*` and gRPC. Forwarded through the daemon's environment instead of the process table.

Each flag also reads its `OPENKIND_*` environment variable (`OPENKIND_HTTP_ADDR`, `OPENKIND_GRPC_ADDR`, `OPENKIND_MODELS`, `OPENKIND_INSTALLED_MODELS`, `OPENKIND_MODELS_DIR`, `OPENKIND_API_KEY`). The [server guide](../openkind-server/README.md) documents the daemon's full option set.

### `keygen`

Generates one API key from 32 bytes of operating-system randomness (256 bits):

```bash
openkind keygen
```

Stdout contains only the `ok_` prefix, 64 lowercase hexadecimal characters, and a newline. The command does not start a daemon or save the key.

Authentication remains disabled when no API key is configured. To enable it on macOS or Linux, generate a key in the current shell and start the daemon:

```bash
export OPENKIND_API_KEY="$(openkind keygen)"
openkind serve --http-addr 127.0.0.1:8080 --grpc-addr 127.0.0.1:9090
```

On Windows, use PowerShell:

```powershell
$env:OPENKIND_API_KEY = (openkind keygen)
openkind serve --http-addr 127.0.0.1:8080 --grpc-addr 127.0.0.1:9090
```

You can also define a key with `--api-key <KEY>` on `openkind serve` or `openkindd`. Configured keys must be nonempty visible ASCII with no whitespace; invalid keys fail daemon startup.

Clients send `Authorization: Bearer <KEY>` for `/v1/*` and gRPC. `/health` and `/metrics` remain public. `openkind evaluate` and `openkind status` read `OPENKIND_API_KEY` or accept `--api-key <KEY>`. In the web playground, enter the key under Settings; it stays in browser memory until the page closes.

Keys are not saved to `~/.openkind` or a configuration file. Supply the same key again when restarting the daemon in a new shell or environment. See the [environment reference](../../docs/ENV.md) for key resolution and compatibility aliases.

### Model profiles

`catalog` shows curated profiles available to pull. `list` shows local installations, and `show` prints one installed profile and its pinned artifacts. These commands use labeled text output by default; add `--json` for machine-readable output.

```bash
openkind catalog
openkind pull qwen35-state-first:a047d6802c3f06f085b8
openkind list
openkind show qwen35-state-first:a047d6802c3f06f085b8 --json
openkind rm qwen35-state-first:a047d6802c3f06f085b8
```

Pull progress appears on stderr. Terminals get a per-artifact progress bar with transfer rate and SHA-256 verification state; redirected output gets concise milestones. A successful pull prints the `openkind serve --installed-models NAME` command needed to load the profile at daemon startup. Restart the daemon after changing its installed model list.

### `status`

Checks a daemon's unauthenticated health endpoint and lists aliases registered with that running process:

```bash
openkind status --server http://127.0.0.1:18080
openkind status --watch --server http://127.0.0.1:18080
```

Use `--api-key`, `OPENKIND_API_KEY`, or `TYPESAFE_API_KEY` when the daemon protects `/v1/models`. Registered aliases are the profiles available to evaluations on that daemon; local installations are shown separately by `openkind list`.

`--watch` opens a live terminal view, refreshes every five seconds, and exits with `q`, `Esc`, or `Ctrl-C`. It requires an interactive terminal.

### `playground`

Launches the local web playground. Connects to an existing running daemon if one is already healthy at the target address; otherwise spawns a loopback-only `openkindd` with the playground route enabled. The daemon rejects `--playground on` unless its HTTP listener is bound to a loopback address, keeping model lifecycle controls local. `Ctrl-C` stops a spawned daemon.

```bash
openkind playground
openkind playground --installed-models qwen35-state-first:a047d6802c3f06f085b8
openkind playground --no-open --http-addr 127.0.0.1:18080
```

Flags:
- `--http-addr <ADDR>`: Loopback address to bind or connect to (default: `127.0.0.1:8080`). Non-loopback addresses are rejected.
- `--installed-models <MODELS>`: Comma-separated installed models to load at daemon startup.
- `--models <ALIASES>`: Comma-separated model aliases to expose (default: `mock,jev-latest`), so running `openkind playground` alone uses mock models.
- `--no-open`: Print the playground URL instead of opening a browser.
- `--models-dir <PATH>`: Directory shared by model commands and the daemon.
- `--api-key <KEY>`: Optional API key for the spawned daemon's bearer authentication.

### `version`

Prints the wire API contract version:

```bash
openkind version
# Output: openkind jev-compatible-0.1
```

## Testing

```bash
cargo test -p openkind-cli
```

## License

See the [MIT license](../../LICENSE). Cargo metadata declares `MIT OR Apache-2.0`.
