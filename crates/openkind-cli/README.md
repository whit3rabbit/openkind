# openkind-cli

> Command-line client for inspecting schemas and evaluating requests against `openkind`.

`openkind-cli` provides the `openkind` binary for inspecting requests, evaluating them against a running daemon, and managing curated model installations.

## Installation & Build

```bash
cargo build --release -p openkind-cli
# Binary produced at target/release/openkind
```

## Subcommands

### 1. `inspect`
Validates a JSON request file against the Jev request schema:

```bash
openkind inspect examples/01_noul.json
# Output: examples/01_noul.json ✓ (model=jev-latest, 1 questions)
```

Returns exit code `0` on success, non-zero with error context if JSON is malformed or violates schema validation rules (e.g. empty questions, missing instructions).

### 2. `evaluate`
POSTs a JSON request file to a running `openkindd` instance. Successful responses are JSON on stdout by default, so they can be piped directly to tools such as `jq`:

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

### 3. Model profiles

`catalog` shows curated profiles available to pull. `list` shows local installations, and `show` prints one installed profile and its pinned artifacts. These commands use labeled text output by default; add `--json` for machine-readable output.

```bash
openkind catalog
openkind pull qwen35-state-first:a047d6802c3f06f085b8
openkind list
openkind show qwen35-state-first:a047d6802c3f06f085b8 --json
openkind rm qwen35-state-first:a047d6802c3f06f085b8
```

Pull progress appears on stderr. Terminals get a per-artifact progress bar with transfer rate and SHA-256 verification state; redirected output gets concise milestones. A successful pull prints the `openkind serve --installed-models NAME` command needed to load the profile at daemon startup. Restart the daemon after changing its installed model list.

### 4. `status`

Checks a daemon's unauthenticated health endpoint and lists aliases registered with that running process:

```bash
openkind status --server http://127.0.0.1:18080
openkind status --watch --server http://127.0.0.1:18080
```

Use `--api-key` or `OPENKIND_API_KEY` when the daemon protects `/v1/models`. Registered aliases are the profiles available to evaluations on that daemon; local installations are shown separately by `openkind list`.

`--watch` opens a live terminal view, refreshes every five seconds, and exits with `q`, `Esc`, or `Ctrl-C`. It requires an interactive terminal.

### 5. `version`
Prints the wire API contract version:

```bash
openkind version
# Output: openkind jev-compatible-0.1
```

## Testing

```bash
cargo test -p openkind-cli
```
