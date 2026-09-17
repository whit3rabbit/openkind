# OpenDecision Phase 2B — measured readout

Run: 20260917T205849Z  
Model: Qwen/Qwen3.5-4B-Base @ 1001bb4d826a52d1f399e183466143f4da7b741b  
GPU: NVIDIA L4; dtype: torch.bfloat16

## Selection and quality
Selected on **dev NLL**: `last_linear_seed17`.

| Evaluation | Accuracy | Macro F1 | NLL | Brier (sum) | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| Matched, raw | 0.8767 | 0.8775 | 0.3345 | 0.1834 | 0.0298 |
| Matched, calibrated | 0.8767 | 0.8775 | 0.3392 | 0.1854 | 0.0380 |
| Mismatched, calibrated | 0.8733 | 0.8728 | 0.3233 | 0.1818 | 0.0398 |

Matched accuracy 95% premise-cluster bootstrap interval: [0.8519, 0.9022].
Temperature: 1.1441, fitted on a separate calibration set.
Matched test NLL change after calibration: +0.0047 (negative is better; no guarantee under shift).

## Model and resources
Text-backbone parameters: 4,205,751,296.
Logical vocabulary head: 635,699,200; marginal removable parameters: 0.
Input/output weights tied: True. Vision tower loaded: False.

Frozen run: completed. LoRA: disabled.
This run does not choose a production architecture automatically; compare quality, calibration and latency together.

## Limitations
- This is a fixed three-class NLI experiment, not a trained arbitrary-schema Jev replacement.
- No shared-prefix cache fork, inter-question isolation architecture, dynamic candidate head, or ordinal Score training is implemented.
- The pretraining corpus may contain MultiNLI. Split separation here does not establish decontamination of Qwen pretraining.
- Removing a tied LM head does not remove the input embedding or most text parameters.
- No universal calibration or production latency guarantee follows from these sampled tasks and short timing runs.
- Generation timings and classification timings do not demonstrate equal decision quality.
- Attention pooling can differ because of extra trainable parameters, not only pooling choice.
- No quantization, Rust, Metal, or cross-device parity benchmark was run by this notebook.
- Only one head-training seed was evaluated; accuracy intervals quantify test sampling, not training-seed uncertainty.
- LoRA was disabled; this run cannot conclude whether LoRA improves the backbone.
- train: truncated 2 premises and 0 hypotheses.
- test_matched: truncated 1 premises and 0 hypotheses.
- test_mismatched: truncated 1 premises and 0 hypotheses.