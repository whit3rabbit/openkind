# opendecision-gen-schemas

> Utility binary to generate draft-2020-12 JSON Schemas from `opendecision-core` wire types.

`opendecision-gen-schemas` uses `schemars` to inspect the canonical Rust structs in `opendecision-core` (`SystemRequest` and `SystemResponse`) and output the official JSON Schemas.

The resulting schemas live under `crates/opendecision-core/schemas/`:
- `jev-v1-request.json`
- `jev-v1-response.json`

## Usage

Run the generator with `--write` to update the committed schema files directly:

```bash
cargo run -p opendecision-gen-schemas -- --write
```

Or run without flags to print the schemas to stdout:

```bash
cargo run -p opendecision-gen-schemas
```

This ensures that schemas used by MCP adapters, OpenAPI documentation, and external clients remain synchronized with the Rust wire types.
