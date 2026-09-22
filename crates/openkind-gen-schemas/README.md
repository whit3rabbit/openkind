# openkind-gen-schemas

> Utility binary to generate draft-2020-12 JSON Schemas from `openkind-core` wire types.

`openkind-gen-schemas` uses `schemars` to inspect the canonical Rust structs in `openkind-core` (`SystemRequest` and `SystemResponse`) and output the official JSON Schemas.

The resulting schemas live under `crates/openkind-core/schemas/`:
- `jev-v1-request.json`
- `jev-v1-response.json`

## Usage

Run the generator with `--write` to update the committed schema files directly:

```bash
cargo run -p openkind-gen-schemas -- --write
```

Or run without flags to print the schemas to stdout:

```bash
cargo run -p openkind-gen-schemas
```

This ensures that schemas used by MCP adapters, OpenAPI documentation, and external clients remain synchronized with the Rust wire types.

## Benchmark context

This crate only emits JSON Schemas. It does not load a model or execute a
benchmark. Current MLX throughput evidence is recorded in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) and is produced by
`openkind-bench`.

The comparison uses `Qwen/Qwen3.5-4B-Base` at revision
`1001bb4d826a52d1f399e183466143f4da7b741b`, with the pinned profile bundle,
the native Candle CPU backend, and the MLX FP32 backend built with the pinned
MLX 0.32.2 toolchain. On the named M4 Max host, one warm 12-row smoke run
measured:

| Backend | `repeated_full` | `nested_sequential` | `nested_batched` | `choose_strategy` |
|---|---:|---:|---:|---:|
| Candle CPU | 119.22 s | 70.34 s | 71.41 s | 68.83 s |
| MLX FP32 | 29.42 s | 7.91 s | 7.92 s | 7.94 s |

That is approximately `4.05x`, `8.89x`, `9.02x`, and `8.67x` faster for the
four columns respectively. These are single-sample throughput observations,
not schema-generation performance or a model-parity release gate. The
separately downloaded `mlx-community/Qwen3.5-4B-MLX-bf16` checkpoint was
within roughly `0.4%` of pinned MLX throughput on these cells, but produced
different Choice selections because it is a different source model and
conversion. The reproducible MLX command and the community-checkpoint
comparison are in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).
