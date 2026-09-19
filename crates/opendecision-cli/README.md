# opendecision-cli

> Command-line client for inspecting schemas and evaluating requests against `opendecision`.

`opendecision-cli` provides the `opendecision` binary for interacting with request fixtures and running `opendecisiond` instances from the command line.

## Installation & Build

```bash
cargo build --release -p opendecision-cli
# Binary produced at target/release/opendecision
```

## Subcommands

### 1. `inspect`
Validates a JSON request file against the Jev request schema:

```bash
opendecision inspect examples/01_noul.json
# Output: examples/01_noul.json ✓ (model=jev-latest, 1 questions)
```

Returns exit code `0` on success, non-zero with error context if JSON is malformed or violates schema validation rules (e.g. empty questions, missing instructions).

### 2. `evaluate`
POSTs a JSON request file to a running `opendecisiond` instance:

```bash
opendecision evaluate examples/04_mixed.json --server http://127.0.0.1:18080 --pretty
```

Flags:
- `--server <URL>`: Daemon base URL (default: `http://127.0.0.1:8080`).
- `--pretty`: Pretty-print the returned JSON response.

### 3. `version`
Prints the wire API contract version:

```bash
opendecision version
# Output: opendecision jev-compatible-0.1
```

## Testing

```bash
cargo test -p opendecision-cli
```
