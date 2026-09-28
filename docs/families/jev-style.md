# Jev-Style 2B decision profile

## Pinned source

| Item | Pin |
|---|---|
| MLX repository | [`chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX`](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX) |
| Immutable Hub revision | `b86e4cbc6f420f5d7d1691299d62db272934ebf4` |
| Base profile | `chaoliangUNSW/Jev-Style-2B-Decision-v3`, 1.88B parameters, based on Qwen3.5-2B |
| Precision folders | `bf16/` (3.76 GB) and affine `8bit/` (group size 64, 2.00 GB) |
| Upstream runtime | `mlx==0.32.2`, `mlx-lm==0.31.3`, Python 3.12 |
| Weights license | Apache-2.0; review the upstream training-data and third-party notices before redistribution or product use |

The revision contains both weight formats, the custom `jev_style_decision_mlx.py`
runtime, `readout_config.json`, `release_config.json`, and a SHA-256 manifest.
The tokenizer and `macjev_norms_fp32.safetensors` sidecar are required in the
selected precision folder. The current upstream manifest check passed for both
folders, and both returned finite, normalized probabilities on the local M4
Max. Local timings are recorded in
[`../benchmarks/2026-09-27-jev-style-2b-mlx/`](../benchmarks/2026-09-27-jev-style-2b-mlx/).

## Inference contract

This is a decision-specific Qwen3.5-2B profile, not a generic causal language
model checkpoint. Its 24 layers use 18 Gated-DeltaNet layers and 6 full
attention layers. The runtime renders candidate verdict slots and scores each
option as `logit(" yes") - logit(" no")`, then applies one global calibrated
temperature. It uses 2,048-token block-causal attention: tokens in a block can
attend to the entire block and all preceding blocks. The total state, question,
options, and readout budget is 25,600 tokens.

The upstream runtime patches two `mlx-lm` Qwen3.5 details: the Gated-DeltaNet
query/key normalization epsilon and the use of exact FP32 shifted RMSNorm
weights. It checks source hashes for the functions it patches or relies on and
refuses other `mlx-lm` versions. Keep `mlx-lm==0.31.3`, the FP32 norm sidecar,
the custom runtime, and the matching readout together. Stock
`mlx_lm.generate` does not implement this scoring contract or block mask.

## Upstream-reported evidence

The model card reports the following comparison against its PyTorch FP32 CPU
reference. The author used 1,000 development rows of up to 4,096 tokens and a
long fixture with 43 questions up to 25,600 tokens. These are upstream claims,
not OpenKind parity results.

| Precision | Same top-1 as FP32 | Max absolute probability difference | Accuracy (FP32 80.8%) | Long fixture |
|---|---:|---:|---:|---:|
| BF16 | 99.7% | 0.035 | 80.5% | 43 / 43 |
| 8-bit | 99.6% | 0.162 | 80.8% | 43 / 43 |

The model card also reports 73.6% on its self-run JevBench v1.4.1 set of 231
public items, with a 95% Wilson interval of 67.6% to 78.9%. Its card notes that
the training pool includes train splits or format-imitating data from the
benchmarks it lists, and that Decision Index 0.2 was not run. Treat those
results as upstream evidence with that overlap disclosed.

## OpenKind status

**Surveyed only.** The model and upstream Python MLX runtime load locally, but
OpenKind has no Rust loader, offline parity fixtures, `DecisionEngine` adapter,
daemon alias, or installable catalog entry for this profile. The existing
`Qwen35DecisionEngine` is pinned to a different Qwen3.5-4B architecture and
score-summary readout. This 2B profile has a different layer layout, block
mask, yes/no logit-difference readout, calibration, and required norm sidecar.

The profile is listed in the surveyed-family index only. Per the
[model-store contract](../MODEL_REGISTRY.md), `registry/v1` remains limited to
profiles with a compiled-in loader. A Rust integration would need its own
loader and offline qualification before an alias or pullable catalog entry is
valid.

The model card describes a reduced 60M-token training pool and flags some
training data as restrictive or unclear, including model-generated rows. The
weights carry Apache-2.0 metadata, but that metadata alone does not resolve the
upstream training-data caveat.

## Run the upstream MLX profile

Download the pinned source and one precision folder. This downloads model
weights only when explicitly invoked:

```bash
MODEL_DIR="$HOME/.cache/openkind/jev-style-2b-v3-mlx"
hf download chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX \
  --revision b86e4cbc6f420f5d7d1691299d62db272934ebf4 \
  --include "bf16/*" --include "jev_style_decision_mlx.py" \
  --include "readout_config.json" --include "release_config.json" \
  --include "manifest.json" --include "requirements.txt" --include "config.json" \
  --include "LICENSE" --include "NOTICE" \
  --include "THIRD_PARTY_NOTICES.md" --local-dir "$MODEL_DIR"
uv venv --python 3.12 "$MODEL_DIR/.venv"
uv pip install --python "$MODEL_DIR/.venv/bin/python" -r "$MODEL_DIR/requirements.txt"
"$MODEL_DIR/.venv/bin/python" "$MODEL_DIR/jev_style_decision_mlx.py" \
  --model-dir "$MODEL_DIR" --precision bf16 --verify \
  --state "The service rollout is complete, but intermittent timeouts remain." \
  --question "What should the support team do next?" \
  --options '{"investigate":"inspect the traces","close":"close without checking","__none__":"none of these"}'
```

Use `--include "8bit/*"` and `--precision 8bit` for the smaller format. The
upstream `requirements.txt` intentionally pins MLX and MLX-LM versions.
