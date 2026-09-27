# OpenKind — cache and finite-decision follow-up

Status: **PARTIAL**
Run key: `7047c6b31436f8e9b5aa85a5dad9ea4378d16eaa0912ebba288fae273c7e12ae`

A completed study may contain failed scientific gates. No model, probability policy or cache path is promoted.

## Answers from this run
- vLLM repeatability: 1/4 measured mode/profile rows passed.
- Cache/performance: 6/40 rows qualified independently by shape.
- The cache table requires observed reuse, a stable cold baseline, ≤0.005 raw probability drift and no decision/diagnostic-action flips. Negative short-prefix controls are expected to be unqualified.
- Native branching compares one branch slot with 24 reserved sequences and compares padding enabled/disabled on the same GGUF. See native_branching.csv and runtime memory records.
- Decode calls and rounds are separate counters; neither counts GPU kernels or internal microbatches.
- Complete native tree probabilities describe a locally constrained token policy. Greedy mode has no fabricated full distribution.
- Native cache telemetry refers to instructions/catalogue reuse; per-request context trunks are a separate sharing mechanism.
- Native JSON comparison holds model, quantization, allowed choices, descriptions, state and rule information fixed. The required output/prompt format differs.
- Native GGUF Q4 and vLLM BF16/GPTQ results are distinct system profiles. Do not attribute cross-backend changes solely to MoE architecture or kernels.

## Questions still open
- A trained state-only head matching candidate-conditioned reasoning
- RLCD training efficacy
- Sealed production generalization
- Quantization versus architecture in isolation
- Actual GPU microbatch/kernel count (decode-call counters are not kernel counters)

## Evidence
Raw job JSON, exact token fixtures, source, runtime/launch records, tables and charts are included.
Prior run: https://drive.google.com/drive/folders/1LIKE7JSmEcO2fvrhf8Qx4_ZJfv-heGwD
Method: https://www.privatemode.ai/blog/system-one-from-glm-flash
Native branch: https://github.com/thecodacus/llama.cpp/tree/ad129b08d9f134cd298d1f8a85efc52b1b66e18e/tools/parallel-decision
Batch invariance: https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/