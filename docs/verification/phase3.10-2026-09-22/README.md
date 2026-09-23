# Phase 3.10 persisted-state replay verification

Recorded 22 September 2026 on the named Mac host. This dirty-tree follow-up is
anchored at base revision `a5a752ab50efccba2eff0345fc5435c01248d41e`. The
replay implementation used for this run has SHA-256
`94a4b6451dbffc8013f0b20dc78850dc3ddcaeb212c1cce527e71eef04b9d8df`. The
Qwen backbone executor source used by the run has SHA-256
`581119b9d2f056befee7a87081cb21ebc42d76b6b6657222713c03bd2cfbb498`.

## Provenance

- Host: Apple M4 Max (`Mac16,5`), 36 GiB, macOS 26.6.2 (`25G83`), `aarch64`.
- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`.
- Profile: `a047d6802c3f06f085b8`.
- Checkpoint: `Qwen/Qwen3.5-4B-Base`, revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Bundle SHA-256:
  `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`.
- Tree state: dirty. The native run records `git_dirty: true` and omits raw
  argv and sensitive paths.
- Process A run: `20260923T021707Z` (`persist-save`).
- Process B run: `20260923T021814Z` (`persist-replay`).

The sanitized `openkind-native-run/v1` records and checksum manifests are
archived in [`persist-save/`](persist-save/) and
[`persist-replay/`](persist-replay/).

## Commands

Each command was a separate `cargo run` invocation, so process B restored the
snapshot after process A had exited.

```bash
cargo run --release --locked -p openkind-backends --example qwen35_parity_probe -- \
  <pinned-checkpoint-root> \
  research/OpenKind_Phase3B_BackboneParity_20260920T152206Z \
  persist-save target/verification/qwen35-persist-root-state.bin \
  --evidence-root target/verification/native-runs

cargo run --release --locked -p openkind-backends --example qwen35_parity_probe -- \
  <pinned-checkpoint-root> \
  research/OpenKind_Phase3B_BackboneParity_20260920T152206Z \
  persist-replay target/verification/qwen35-persist-root-state.bin \
  crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8 \
  --evidence-root target/verification/native-runs
```

## Results

The restored root passed its content-fingerprint comparison against an
independent fresh prefill. The root stayed immutable during replay. Position
and storage checks passed at root `98`, question `109`, and candidate `121`,
with `59,899,904` tensor-storage bytes.

The replay covered all 3 questions and 8 candidates sharing that root. Each
restored candidate feature matched an independent full-sequence forward at
maximum absolute difference `0.0` (guard `1e-4`). The complete head
probability vector, including semantic-none mass for Choice, matched at
maximum absolute difference `0.0` (tolerance `0.005`). Argmax and policy
actions had zero changes.

| Question | Candidates | Selected index, restored/full | Policy, restored/full |
|---|---:|---:|---|
| `asset` | 3 | 2 / 2 | review / review |
| `external` | 2 | 0 / 0 | accept 0 / accept 0 |
| `urgency` | 3 | 2 / 2 | accept 2 / accept 2 |

Trace-hidden diagnostics were `0.0000782012939453125` for the question and
`0.00006866455078125` for the candidate. They localize backbone differences
against the frozen trace and are not acceptance thresholds.

The expanded Phase 3.10 persistence gate passes for this CPU FP32 profile. It
does not establish Metal parity, practical high-K latency, production
load/soak, or release promotion.

## Focused checks

| Check | Result |
|---|---|
| `cargo check -p openkind-backends --example qwen35_parity_probe --locked` | Pass |
| `cargo clippy --locked -p openkind-backends --example qwen35_parity_probe -- -D warnings` | Pass |
| Separate `persist-save` invocation | Pass; run `20260923T021707Z` |
| Separate `persist-replay` invocation | Pass; run `20260923T021814Z` |
| Workspace `cargo fmt --check` | Pass |
| `git diff --check` | Pass |

The workspace test suite and workspace-wide Clippy battery were not run.
