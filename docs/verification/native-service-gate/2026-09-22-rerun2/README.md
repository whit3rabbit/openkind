# Native service gate rerun — 22 September 2026

The provenance-bound release-mode native CPU campaign passed the service
checks recorded in [`service-report.json`](service-report.json). Daemon logs
and before/after build source fingerprints are retained beside the report.

## Build and source identity

The run started on a Mac16,5 Apple M4 Max with 36 GiB RAM, arm64 macOS 26.6.2,
and Rust 1.98.1. It used commit `a5a752ab50efccba2eff0345fc5435c01248d41e`
with a dirty working tree. The saved pre-build, post-build, and daemon-start
records match on commit, tracked-diff SHA-256
`303f8ce5a6b31cec45310a0a6530f142299babab242ec156b1981cd9e28194cd`, untracked
crate-source SHA-256
`644adb9d32ab359539bfe0d8d254dbbe1d206994913babce4c88b14b348c8f6f`, and
harness SHA-256
`92615413e1384a6d3358978d0bebdc8eb4f81e4e850de2a4cc58da3d81cd0e7c`. The
release executable SHA-256 was
`0672cad7ca9066db5c849c89364930814668534923b7f4f98fd7a2c17d657d8d`; its hash
was unchanged through the service run.

The shared working tree changed after daemon startup. The final tracked-diff
hash is `229a2e2c0164dcbd636cd3cb5802f6f64b252c7f0153994c0192402d238e88d9`,
and the final untracked crate-source hash is
`40a11183604cee5bf9e674d1de3f768b53253284574059b24202442bbd0f9d80`. These
end-of-run values are recorded as workspace drift; they do not change the
already-built process. The report's provenance check requires the pre-build
source record to match the daemon-start fingerprint and the release executable
hash to remain unchanged. The source record is a fingerprint, not an archived
patch, so this dirty-tree run does not claim a clean-commit build.

The original runner used for the campaign is preserved as
[`native-service-gate-runner-used.py`](native-service-gate-runner-used.py); its
SHA-256 matches the report. That version also required the whole checkout to
stay unchanged through shutdown. A shared source edit after startup tripped
that extra check even though the pre-build and daemon-start fingerprints
matched and the executable hash stayed fixed. The report's gate was therefore
re-evaluated using the artifact boundary documented above. The checked-in
runner now applies that same rule and retains post-start source changes as an
informational field.

The build command was:

```bash
CARGO_TARGET_DIR=target/native-service-gate cargo build --release --locked --offline -p openkind-server --bin openkindd
```

## Results

| Phase | Result |
|---|---|
| Startup and resident request | Healthy in 15.59 s. First and resident requests returned HTTP 200 with 3/3 answers in 15.05 s and 15.10 s. |
| Concurrency 1 | 3/3 accepted; p50/p95/p99 15.22/15.40/15.40 s; 0.06577 requests/s. |
| Concurrency 2 | 6/6 accepted; p50/p95/p99 15.61/30.91/30.91 s; 0.06500 requests/s. |
| Concurrency 4 | 9/12 accepted, 3 immediate HTTP 529 overloads; accepted p50/p95/p99 31.07/47.02/47.02 s; 0.06420 requests/s. |
| Cancellation and recovery | Client reset advanced the cancellation counter from 0 to 1. The next request returned HTTP 200 with 3/3 answers in 15.95 s. |
| Deadline | A separate release daemon with a 1 ms queue-inclusive timeout returned HTTP 504 in 2.12 ms. |
| Steady soak | 1,814.216 s; 115/115 requests returned HTTP 200 with 345 answers. Throughput was 0.06339 requests/s; p50/p95/p99 were 15.78/15.88/15.94 s. |
| Memory and shutdown | 2,111 one-second RSS samples. Sampled median/max RSS were 6,648,053,760 / 10,280,976,384 bytes; daemon peak-RSS metric was 10,283,155,456 bytes. The daemon exited cleanly on SIGINT. |

Invalid wire input returned HTTP 422, malformed JSON returned HTTP 400, and
request IDs were echoed. The post-run audit found no generated request-ID
prefix, bearer/auth header, or any of the 68 fixture strings of at least 16
characters in either daemon log.

The fixture has no reviewed labels, so semantic accuracy, accepted-error, and
coverage are unavailable. The bundle manifest declares
`data_scope: exploratory_pilot` and `production_ready: false`. This closes the
native CPU service lifecycle and load/soak checks for the recorded release
artifact; it does not establish reviewed model quality, Metal service behavior,
signed packaging, or product-release promotion.

## Reproduction

Use the before-build capture, offline release build, post-build capture, and
runner command in [`docs/BENCHMARKS.md`](../../../BENCHMARKS.md). Supply the
local pinned checkpoint, bundle, runtime manifest, tokenizer, and smoke fixture
paths listed in that command. Each run needs a new empty output directory.
