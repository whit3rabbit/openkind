# openkind-cli

> Command-line client for inspecting schemas and evaluating requests against `openkind`.

`openkind-cli` provides the `openkind` binary for interacting with request fixtures and running `openkindd` instances from the command line.

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
POSTs a JSON request file to a running `openkindd` instance:

```bash
openkind evaluate examples/04_mixed.json --server http://127.0.0.1:18080 --pretty
```

Flags:
- `--server <URL>`: Daemon base URL (default: `http://127.0.0.1:8080`).
- `--pretty`: Pretty-print the returned JSON response.

### 3. `version`
Prints the wire API contract version:

```bash
openkind version
# Output: openkind jev-compatible-0.1
```

## Testing

```bash
cargo test -p openkind-cli
```
