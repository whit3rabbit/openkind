# Owner decision record: remaining Decision Index 0.3.1 entries stay as pinned metadata

Written 2026-10-12 after the registry expansion and the feasible loader
campaigns. The handoff
([REGISTRY_EXPANSION_2026-10-12.md](REGISTRY_EXPANSION_2026-10-12.md)) asked
for immutable pins for every 0.3.1 leader; the session goal asked for
`openkind pull` support per model. This file records where that work stops
and why.

## Landed with full `openkind pull` support

| Profile | Commit | Evidence |
|---|---|---|
| `d3-edge:664f9ead734372935286` | `20ed3e8` | prompt byte-parity 7/7, golden replay |
| `d3-lite:e0dd32dbc16b27546692` | `ae75dba` | prompt byte-parity 7/7, golden replay |
| `d3-nano:0198af5b5c8e8835605e` | `f4275cc` | prompt byte-parity 7/7, golden replay |
| `d3-mini:434dbc50d3b581663e52` | `ea86471` | prompt byte-parity 7/7, golden replay |
| `d3-flash:b2a4750f8bf4a1fa58cb` | `7d9e19a` | prompt byte-parity 7/7, golden replay |
| `sieve-4b:8b97b56524b55c41fbf4` | `f62b5a3` | prompt byte-parity 7/7, golden replay |

Plus `441e330`: the 0.3.1 registry expansion pinning all surveyed entries,
base models, and MLX leads with re-resolved revisions and pin-vs-HEAD
equivalence checks.

## Decision: the remaining entries stay as pinned metadata

The entries below keep `openkind_loadable: false` and
`installable: false` in
[`registry/v1/jev-decision-index-0.3.1.json`](../../registry/v1/jev-decision-index-0.3.1.json).
Each blocker is recorded per entry in that file; grouped:

1. **Above the 36 GB reference host (27B-31B BF16):** d3 27B, pplx v1.1,
   Darwin-27B-ZTC, Sieve-27B, Kev-27b, Shisa DE-2, Matilda, RSI-Jev v6.1
   27B, the noncausal LoRA, linne-fern, Quyet, Vega. Loading any of them
   needs either a larger qualifying host or a quantized release, plus the
   per-entry readout adapter and multi-gigabyte parity evidence
   (docs/families/NEW_FAMILY.md qualification).
2. **No execution path:** Decider 31B and Blink (NVFP4 weights; openkind
   has no NVFP4 backend).
3. **Family gap:** Shisa DE-2 and Xor (gemma4 MoE routing the gemma4
   family's E4B-pinned loader rejects); Torchcast 27B/12B/4B and StartLux
   27B/9B/4B (the StartLux letter readout plus per-type calibration policy
   — also CC BY-NC 4.0, which the catalog distribution policy would have to
   admit first); Sieve-27B and Sieve-9B-Plus (same adapter pattern as
   sieve-4b but over 27B/9B bases with a torch-pickle `head.pt` needing
   safe re-expression); Bespoke Nimble v3 (adapter whose readout is defined
   by the non-commercial Nimble serving stack); garas (adapter plus custom
   engine code needing re-expression).
4. **Superseded:** Decision 2.0 Vega/Lux/Nox — the d3 family supersedes
   them at every size; kept as predecessor baselines.
5. **Unresolvable provenance:** Matilda's declared base repository
   (`Maincode/MATILDA`) returned HTTP 401 on 2026-10-12; Darwin's base
   lineage is author-disclosed only.

## What would change the decision

- A 64 GB-class qualifying host (or official quantized releases) unlocks
  group 1 with the d3-family adapter already in tree for the Qwen3.8-27B
  entries.
- An NVFP4 backend unlocks group 2.
- A gemma4 MoE config and the StartLux letter readout unlock their groups.
- Any of these reuse the pins in the 0.3.1 registry file as immutable
  starting sources, which is exactly what the pins were recorded for.

Signed: agent run 2026-10-12, per the session goal's "after exhausting
feasible entries" clause. Feasible entries are exhausted: every handoff
model that fits the reference host, has an execution path, carries a
re-expressible contract, and is license-compatible now has a loadable,
pullable catalog profile.
