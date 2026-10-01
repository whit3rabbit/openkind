# openkind-bench

`openkind-bench` is the offline scoring and timing harness for openkind engines over JSONL decision workloads (Noul, Choice, and Score). `openkind-bench score` measures pinned native request-path performance and memory, and `dataset eval` scores labeled datasets for model quality. The two evidence classes stay separate: model quality is never inferred from throughput or predictions alone.

## Quickstart

The harness is a workspace crate, not a published package. From a checkout of the repository, run the fully offline mock smoke test:

```bash
cargo run -p openkind-bench -- score \
  crates/openkind-bench/fixtures/decisions_smoke.jsonl \
  --engine mock --reps 3
```

The mock engine needs no model assets, and builds and tests never download anything. `gen-workload` produces a seeded state × criterion grid that is byte-identical for identical seeds:

```bash
cargo run -p openkind-bench -- gen-workload \
  --states 37 --criteria 21 --seed 291607 --output bench-output/shape777.jsonl
```

## Scoring and timing

`score` runs a workload end to end through a real engine path, prints a provenance summary, and writes per-row predictions under `--output-dir`.

Timing covers request construction, validation, dispatch, and answer extraction. Model load is excluded and reported separately per strategy. Rows sharing one state are grouped into a single request by default (`--no-group` emits one request per row), so a grouped row reports its shared request latency, not a per-decision time.

```bash
cargo run --release -p openkind-bench -- score <workload.jsonl> \
  --engine qwen35 \
  --bundle-root <profile-bundle-dir> \
  --checkpoint-root <pinned-checkpoint-dir> \
  --tokenizer <digest-locked-tokenizer.json> \
  --strategies repeated_full,nested_sequential,nested_batched,choose_strategy \
  --reps 1 --host "<host label>" --commit <hash>
```

`--engine mock` is the default. The native Qwen 3.5 engine loads only locally pinned checkpoint, bundle, and tokenizer paths. Surveyed-family engines take `--model-root` instead, and winnow takes `--adapter`. The `--strategies` sweep is native-only, and those engines reject it. Optional `mlx`, `cuda`, and `onnx` features add further engines.

`--host` and `--commit` are recorded as caller-supplied attribution that the harness does not verify.

`--no-warmup` skips the untimed warmup pass for cold and history probes. `--history-aba` executes two rows as A, B, A with identical first and last requests. The paired `compare-choice` and `calibrate-choice` diagnostics compare the fitted scorer against experimental joint-option scoring and fit post-hoc calibration on disjoint partitions.

## Dataset evaluation

[`openkind-datasets`](../openkind-datasets/) backs the `dataset` subcommands: `list`, `pull`, `rm`, `verify`, `pin`, `build`, and `eval`. Downloads are explicit and networked, pinned to exact Hugging Face revisions, and verified against per-file SHA-256 digests before install.

Evaluation then reads only the verified local install, and dataset bytes are never committed to the repository or redistributed. The cache root follows `--datasets-dir` or `OPENKIND_DATASETS_DIR`.

```bash
cargo run -p openkind-bench -- dataset list
cargo run -p openkind-bench -- dataset pull sst2
cargo run -p openkind-bench -- dataset eval sst2 --engine mock --output-dir bench-output
```

`build` materializes a labeled workload JSONL from an installed dataset, with `--split dev` for prompt and threshold work and `--split eval` to report. `eval` scores it through the normal request path and reports accuracy with provenance as an `openkind-dataset-eval/v1` report.

## Methodology and recorded runs

Timing scope, warm policy, strategy definitions, evidence retention, and recorded results are owned by [the benchmarks guide](../../docs/BENCHMARKS.md), which also covers the workspace criterion microbenchmarks over mock and component overhead. Quote a recorded number only with its attribution and the caveats recorded there.

## License

See the [MIT license](../../LICENSE).
