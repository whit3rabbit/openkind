# Native Jeeves MLX smoke

The Rust backend loads and executes the pinned Jeeves 4-bit checkpoint on
Apple M4 Max. Four reference rows pass exact token/readout-position checks,
argmax agreement, and the unchanged `0.005` probability tolerance. The
daemon serves Noul, Choice, and Score without a Python inference process.

## Identity

| Component | Pin |
|---|---|
| Profile | `15fb3b95801f2f039636`, no-thinking, lexicographic Choice labels, affine4/g64 weights, FP32 arithmetic |
| Converted package | `cowWhySo/jeeves-mlx`, `6f4220a267a94c5279b3fcbc5499de734b9a55b0` |
| Original fused export | `PostHog/jeeves`, `8622b7d1652a9dcb8629486b84dce9e8d690c5cd` |
| Upstream source | `PostHog/jeeves`, `f04ec5567301450dcaae0210dd54deeb4f647f87` |
| Native runtime | Rust 1.98.1, mlx-rs 0.32.0, mlx-sys 0.6.0, MLX core 0.32.2 |
| Reference | Bundled `run_jeeves.py`, MLX-LM `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`, MLX 0.32.3, Metal |
| Host | Apple M4 Max, arm64, 36 GiB RAM, macOS 26.6.2 (25G83), Xcode 27.0 (27A266a), SDK 27.0 |

The loader verifies eight [execution files](../../../../crates/openkind-backends/src/families/jeeves/artifacts.rs)
in place, then mmaps the source checkpoint. MLX owns copies of selected
packed tensors. Norms and quantization scales widen to FP32; the model never
expands the complete backbone into FP32 weights. Every question starts with
fresh attention KV, DeltaNet, and convolution state.

## Reference replay

The [frozen fixture](../../../../crates/openkind-backends/tests/fixtures/jeeves_15fb3b95801f2f039636/golden.json)
contains the package's four toy token rows and their live Metal reference
probabilities. Its SHA-256 is
`6fc3351114fd130faf13120727aac448088d4c1facbd45947f3ceded52b35ed5`;
the offline test checks that digest. The direct replay preserves each row's
original candidate order. The HTTP comparison separately checks sorted
Choice labels with a described `__none__`.

| Row | Tokens | Max probability difference | Same argmax |
|---|---:|---:|---|
| Billing Noul | 48 | 0.00091145 | yes |
| Delivery Choice | 49 | 0.00008323 | yes |
| Sentiment Score | 50 | 0.00024487 | yes |
| Literal control-marker text | 71 | 0.00415172 | yes |

See [parity.json](parity.json) for distributions, raw logits, and timings.
The renderer must tokenize prompt, empty-chain newline, and suffix
separately. Encoding their concatenated text merges the two newline tokens
and fails the reference check. The exported offset norms are already folded
and must not receive another `+1`.

Peak active MLX allocation was 5,470,499,840 bytes (5.095 GiB). This is
allocator telemetry, not peak process RSS. Loading the mapped model took
0.865 seconds after artifact verification. Individual forwards took
1.43 to 2.11 seconds. These are uncontrolled debug-build smoke timings;
they do not establish throughput or a speed advantage over MLX-LM.

## HTTP behavior

[http.json](http.json) records the three-question request, native answers,
and separately run Python reference. All answer keys and selected options
agree; the largest probability difference is `0.00032607`. Input usage is
251 tokens. The host adapter reports six output tokens for structured answer
usage accounting; the model generates no tokens.

Separate requests reproduce the multi-question distributions exactly.
Repeating the request with reversed question and Choice-key insertion order
also produces identical answers. Missing `__none__` and a 5037-token row
both return HTTP 422 before inference. The configured daemon uses one
execution slot and no queue.

## Reproduce

Follow the [family setup](../../../families/jeeves.md#run-the-native-engine)
to acquire the pinned package explicitly. With `JEEVES_ROOT` pointing at the
local package, run:

```bash
SDKROOT="$(xcrun --show-sdk-path)" cargo \
  --config 'profile.dev.package.sha2.opt-level=3' run \
  -p openkind-backends --features mlx --example jeeves_native_parity \
  -- "$JEEVES_ROOT"
```

The Cargo override only optimizes SHA-256 artifact verification for this
debug run. It neither skips digest checks nor changes model arithmetic.
The example emits selected normalized hidden vectors for localization;
the compact checked-in report omits those vectors.

## Repository checks

The default workspace suite passes with `--test-threads=1`; its initial
parallel run hit an existing CLI temporary-directory name collision.
Workspace benchmark tests pass with `--benches --locked`. Default and MLX
workspace clippy pass with `--all-targets -- -D warnings`. Formatting and
diff checks pass, and schema regeneration leaves the schema files unchanged.

The MLX backend and server test run reports 338 passing tests before the
proxy-cache E2E harness fails its local-promotion header assertion. That
harness passes on a separate retry. The five Jeeves module tests, frozen
fixture digest test, and daemon configuration checks pass. These checks
remain offline; the real-model runs above require an explicitly prepared
local checkpoint.

[source.json](source.json) records the worktree's base commit and SHA-256
hashes of the implementation and test files used for this evidence.

## Limits

This is a prototype execution and wire smoke. The four toy labels are not
a reviewed task benchmark. Broader source parity, semantic-none calibration,
task quality, matched performance and process-memory campaigns, native
timeout/disconnect stress, and service soak remain open. The stored MLX CPU
reference differs from the live Metal reference, so this result does not
qualify CPU execution. Generated reasoning, speculative drafting, other
quantizations, and catalog installation are outside this profile.
