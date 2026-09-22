# 2026-09-20 — mock smoke run (harness validation)

Purpose: validate the `openkind-bench` harness end to end (workload
loading, state grouping, dispatch, predictions and summary emission). This is
**not performance evidence** — the mock engine is a deterministic placeholder,
not a model.

- Schema: `openkind-bench/v1` (`summary-mock.json`)
- Engine: `mock` (in-process `MockEngine` behind `dispatch`)
- Fixture: `crates/openkind-bench/fixtures/decisions_smoke.jsonl`,
  SHA-256 `3a673e843690b942658b4c9de6cc594770185756098d356e78dcd3efd5cffeeb`,
  12 rows, 4 state groups
- Host: Mac16,5 Apple M4 Max 36 GiB (named Mac), development build
- Commit: `9d086107bb017bf721d815bb5bcb8ba516ce0e6e`
- Reps: 3, grouping: per-state

Timings here measure only request construction, validation, dispatch, and
answer extraction against the mock. Do not quote them as engine performance.
