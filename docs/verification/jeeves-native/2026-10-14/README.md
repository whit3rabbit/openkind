# Jeeves native MLX parity re-run (2026-10-14)

The [2026-09-29 replay](../2026-09-29/REPORT.md) ran on the original
`codex/jeeves-native-mlx` branch. This re-run checks the same code after its
port onto the 0.3.x line (source commit `75e1250`), on the same host,
toolchain, and MLX versions.

The `jeeves_native_parity` example read the pinned `cowWhySo/jeeves-mlx`
package at `6f4220a267a94c5279b3fcbc5499de734b9a55b0` after verifying every
execution file's size and SHA-256 in place. All four frozen reference rows
reproduced their token IDs and readout positions exactly, kept the reference
argmax, and stayed inside the `0.005` probability tolerance.

Every case's probabilities and raw logits are bit-identical to the
2026-09-29 record, and peak active MLX allocation is unchanged at
5,470,499,840 bytes, so the port did not change the arithmetic. The
2026-09-29 [benchmark](../../../benchmarks/2026-09-29-jeeves-native/REPORT.md)
therefore still describes this code. Its timings were not re-measured.

| Row | Tokens | Max probability difference | Same argmax |
|---|---:|---:|---|
| case0 | 48 | 0.00091145 | yes |
| case1 | 49 | 0.00008323 | yes |
| case2 | 50 | 0.00024487 | yes |
| case3 | 71 | 0.00415172 | yes |

[parity.json](parity.json) holds the distributions and raw logits. It omits
the per-row selected hidden states, as the 2026-09-29 record does.
