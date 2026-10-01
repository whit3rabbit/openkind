# openkind-gen-schemas

> One-shot binary that generates the committed Draft 2020-12 JSON Schemas from the `openkind-core` wire types.

`openkind-gen-schemas` is a binary-only workspace utility, not a library and not part of the server request path: no other workspace crate depends on it, and its Cargo manifest sets `publish = false`. It uses `schemars` to derive schemas from the canonical `SystemRequest` and `SystemResponse` structs so the committed schema files stay synchronized with the Rust wire types.

The generated files live under `crates/openkind-core/schemas/`:

- `jev-v1-request.json`
- `jev-v1-response.json`

## Usage

Run the generator with `--write` to update the committed schema files in place. This also prints the schemas to stdout:

```bash
cargo run -p openkind-gen-schemas -- --write
```

Run without flags to print both schemas to stdout, each prefixed with `REQUEST:` or `RESPONSE:`:

```bash
cargo run -p openkind-gen-schemas
```

`-w` is a shorthand for `--write`, `--help` prints usage, and unknown arguments are rejected. The schema directory is resolved from the build's own source tree, so the output does not depend on the caller's working directory.

## Keeping the schemas in sync

Never hand-edit the generated JSON files. Change the `openkind-core` struct definitions, then regenerate with the `--write` command above.

When the committed files drift from what `schemars` derives from the core types, the schema-sync test (`tests/schema_sync.rs`) fails, so a stale schema breaks `cargo test --workspace`. This is the workspace invariant behind the regeneration step in the [agent guide](../../AGENTS.md#verification).

Consumers inherit the same contract structurally: the committed schemas and the [OpenAPI document](../../crates/openkind-api/openapi.yaml) both describe the Jev wire contract, and `openkind-client` re-exports the same core types, which is what guarantees its conformance with these files.

## Benchmark context

This crate only emits JSON Schemas. It does not load a model or execute a benchmark. MLX throughput evidence is recorded in [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md), produced by `openkind-bench`; the full records and newer campaigns live there.

The table below reproduces one recorded comparison so the numbers are not misattributed to this crate. It uses `Qwen/Qwen3.5-4B-Base` at revision `1001bb4d826a52d1f399e183466143f4da7b741b`, the pinned `qwen35_statefirst_a047d6802c3f06f085b8` profile bundle, the native Candle CPU backend, and the MLX FP32 backend built with the pinned MLX 0.32.2 toolchain.

On the named M4 Max host, one warm 12-row smoke run measured:

| Backend | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` |
|---|---:|---:|---:|---:|
| Candle CPU | 119.22 s | 70.34 s | 71.41 s | 68.83 s |
| MLX FP32 | 29.42 s | 7.91 s | 7.92 s | 7.94 s |

That is approximately `4.05x`, `8.89x`, `9.02x`, and `8.67x` faster for the four columns respectively. These are single-sample throughput observations, not schema-generation performance or a model-parity release gate.

The separately downloaded `mlx-community/Qwen3.5-4B-MLX-bf16` checkpoint was within roughly `0.4%` of pinned MLX throughput on these cells, but produced different Choice selections because it is a different source model and conversion. The reproducible MLX command and the community-checkpoint comparison are in [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

## License

See the [MIT license](../../LICENSE). Cargo metadata declares `MIT OR Apache-2.0`.
