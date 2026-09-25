# OpenKind Research & Empirical Notebooks

This directory contains the experimental notebooks, frozen reference artifacts, benchmark datasets, and model selection studies that form the scientific foundation of **OpenKind**.

OpenKind evaluates typed decisions (`Choice`, `Score`, `Noul`) directly from neural representations without an autoregressive token-generation loop. The research journey documented here progresses from initial feasibility probes on `Qwen/Qwen3.5-4B-Base`, through precision contracts, prefix caching, cache compression, multi-domain criteria transfer, and multi-model screening, to the native Rust backbone handoff and multi-question natural-document architectures.

Detailed analysis, theoretical foundations, and mathematical formulations are documented in the [OpenKind Whitepaper](../docs/whitepaper/WHITEPAPER.md) and [Research Dossier](../docs/RESEARCH.md).

---

## Chronological Experiment Index

| # | Notebook | Phase / Run ID | Target HW | Core Research Focus | Status & Outcome | Supporting Artifacts |
|---|---|---|---|---|---|---|
| **1** | [`01_phase2_qwen35_probe.ipynb`](./01_phase2_qwen35_probe.ipynb) | Phase 2 Probe | GPU (L4/A100) | Frozen Qwen3.5 feature extraction & linear decision head | Feasibility proven; identified text vs vision parameter count | [Whitepaper §3.1](../docs/whitepaper/WHITEPAPER.md#31-research-chronology) |
| **2** | [`02_phase2b_benchmark.ipynb`](./02_phase2b_benchmark.ipynb) | Phase 2B (`20260917T205849Z`) | NVIDIA L4 (BF16) | MultiNLI classification, pooling strategies, calibration, generation vs decision timing | Last-token + linear won (87.7% matched); 44.6× speedup over generation; temperature scaling rejected | [`02_phase2b_benchmark_results/`](./02_phase2b_benchmark_results) |
| **3** | [`03_phase2c_stability_dynamic_choice.ipynb`](./03_phase2c_stability_dynamic_choice.ipynb) | Phase 2C (`20260917T222948Z`) | NVIDIA L4 (BF16) | Multi-seed stability, dynamic candidate scoring (Banking77), global `__none__` logit | Stable across seeds (87.0% / 88.8%); 80.8% seen / 64.6% unseen accuracy; 0.817 AUROC for none | [`03_phase2c_stability_results/`](./03_phase2c_stability_results) |
| **4** | [`04_phase2d_numerics_none_handling.ipynb`](./04_phase2d_numerics_none_handling.ipynb) | Phase 2D (`20260917T234417Z`) | NVIDIA L4 | Precision diagnostics (BF16 vs FP32 vs TF32), none-head alternatives, request latency vs K | Isolated layer divergence in DeltaNet/conv; linear latency scaling without prefix caching | [`04_phase2d_numerics_results/`](./04_phase2d_numerics_results) |
| **5** | [`05_phase2e_selective_precision_initial.ipynb`](./05_phase2e_selective_precision_initial.ipynb) | Phase 2E initial (`20260918T032049180933Z`, v2e.1.1) | GPU (L4/A100) | Module-level FP32 promotion, shared-prefix KV branching, application policies | Full FP32 stage halted by memory guard; isolated hybrid state isolation requirement | [`05_phase2e_selective_precision_results/`](./05_phase2e_selective_precision_results) |
| **6** | [`06_phase2e_expanded_batched_prefix_parity.ipynb`](./06_phase2e_expanded_batched_prefix_parity.ipynb) | Phase 2E expanded (`20260918T114914072764Z`, v2e.2.0) | GPU (isolated processes) | Strict FP32 reference, batched prefix reuse parity vs BF16, component breakdown | Strict FP32 prefix reuse passed ($Δp \le 1.1 \times 10^{-5}$); BF16 failed tolerance ($>0.005$) and flipped actions | [`06_phase2e_expanded_parity_results/`](./06_phase2e_expanded_parity_results) |
| **7** | [`07_phase2f_cache_compression_prefix_reuse.ipynb`](./07_phase2f_cache_compression_prefix_reuse.ipynb) | Phase 2F (`20260918T224427722898Z`, v2f.1.0) | NVIDIA L4 / A100 | TurboQuant low-bit KV compression (2/3/4-bit) vs FP16/lossless, GPU LRU prefix cache | Lossless & FP16-KV passed all gates; all 4 low-bit TurboQuant variants failed and flipped actions | [`07_phase2f_cache_compression_results/`](./07_phase2f_cache_compression_results) |
| **8** | [`08_phase2g_fresh_evidence_tf32_cache.ipynb`](./08_phase2g_fresh_evidence_tf32_cache.ipynb) | Phase 2G (`20260919T005142584348Z`, v2g.1.0) | NVIDIA L4 / A100 | Generalization to fresh Banking/CLINC/OOS data, TF32 execution, cache TTL & eviction | TF32 cut latency ~2× but flipped 3 argmax decisions; revealed candidate head narrowness on OOS | [`08_phase2g_fresh_evidence_results/`](./08_phase2g_fresh_evidence_results) |
| **9** | [`09_phase2h_criteria_rejection_multidomain.ipynb`](./09_phase2h_criteria_rejection_multidomain.ipynb) | Phase 2H (`20260919T040612625670Z`, v2h.1.2) | NVIDIA L4 / A100 | Multi-domain candidate fitting, criteria augmentation (support examples), rejection transfer | Original criteria transferred better (83.8% acc) than support examples (78.9% acc); held to pre-registration | [`09_phase2h_criteria_rejection_results/`](./09_phase2h_criteria_rejection_results) |
| **10** | [`10_phase2ij_gated_workbench.ipynb`](./10_phase2ij_gated_workbench.ipynb) | Phase 2I/2J Workbench (`2ij.1.0`) | NVIDIA L4 / A100 | Multi-question reviewed study gate, state-first vs instruction-first GPU mechanics probe | Strict independent-review gate blocked unverified training; probe validated state-first prefill mechanics | [`10_phase2ij_gated_workbench_results/`](./10_phase2ij_gated_workbench_results) |
| **11** | [`11_phase2ij_model_selection_screen.ipynb`](./11_phase2ij_model_selection_screen.ipynb) | Phase 2I/2J Screen (`2ij.2.0`) | NVIDIA L4 / A100 | 13 fit jobs, 31 evaluation profiles across Qwen 3.5 4B, Qwen 2.5 1.5B/2B, ModernBERT | Selected & locked profile `a047d6802c3f06f085b8` (Qwen3.5-4B state-first score-summary head) | [`11_phase2ij_model_selection_results/`](./11_phase2ij_model_selection_results) |
| **12** | [`12_phase2ij_a100_bundle_export.ipynb`](./12_phase2ij_a100_bundle_export.ipynb) | Phase 2I/2J Continuation | NVIDIA A100 | Clean selected-model bundle export and checksum verification on high-memory GPU | Produced golden bundle `4d9ffdee...3332` with zero OOM risk; validated reference repo | Informs profile `a047d6802c3f06f085b8` |
| **13** | [`13_phase3a_branchable_state_batched_q.ipynb`](./13_phase3a_branchable_state_batched_q.ipynb) | Phase 3A (`20260920T024056Z`) | NVIDIA L4 / A100 | Hybrid state branching (KV + DeltaNet + conv), batched-Q/K topologies, crossover behavior | Zero-leakage branch isolation confirmed; batched-Q faster for long state/high Q; informed Rust scheduler | [Whitepaper §15](../docs/whitepaper/WHITEPAPER.md#15-phase-3a-branchable-hybrid-state-and-batched-q-execution) |
| **14** | [`14_phase3b_backbone_parity.ipynb`](./14_phase3b_backbone_parity.ipynb) | Phase 3B (`20260920T152206Z`) | GPU (FP32) | Layer-by-layer backbone traces, golden token fixtures, Rust handoff contract | Exported 47 FP32 vectors across 34 stages; maximum cached continuation drift $\le 1.91 \times 10^{-5}$ | [`14_phase3b_backbone_parity_results/`](./14_phase3b_backbone_parity_results) |
| **15** | [`15_phase4a_statequery_workbench.ipynb`](./15_phase4a_statequery_workbench.ipynb) | Phase 4A.0 / 4A.1 (`20260921T013558Z`, v4a.0.2) | GPU (L4/A100) | Natural-document corpus lock (ContractNLI + QASPER) & StateQuery B1 representation probe | Locked 2,192 states / 15,368 Qs; B1 scored 0.7796 macro-acc but collapsed QASPER none-recall to 0.0 | [Whitepaper §18](../docs/whitepaper/WHITEPAPER.md#18-phase-4a-natural-document-multi-question-benchmark-and-statequery-readout) |
| **16** | [`16_phase4a2_statequery_model_comparison.ipynb`](./16_phase4a2_statequery_model_comparison.ipynb) | Phase 4A.2 (`4a.2.0`) | GPU (L4/A100) | Matched B0 control, source-balanced B1, factorized B2 without reopening final splits | B0 matched collapse (0.0 none-recall); balanced B1 raised none-recall to 0.168 (gate threshold: 0.30) | [Whitepaper §18.4](../docs/whitepaper/WHITEPAPER.md#184-completed-non-final-comparison-aggregate-improvement-hides-a-qasper-collapse) |
| **17** | [`17_phase4b2_statequery_model_comparison.ipynb`](./17_phase4b2_statequery_model_comparison.ipynb) | Phase 4B.2 (`4b.2.0` / `4b.2.1`) | GPU (L4/A100) | Scalar-weight saturation (cap 12) & source/class-stratified applicability | Realized weight saturated at 8.5602; stratified objective raised QASPER recall to 0.3111 gate, but gate policy cost failed (0.11675) | [`17_phase4b2_statequery_model_comparison_results/`](./17_phase4b2_statequery_model_comparison_results) |
| **18** | [`18_phase4b3_applicability_ranking_sweep.ipynb`](./18_phase4b3_applicability_ranking_sweep.ipynb) | Phase 4B.3 (`4b.3.0`) | GPU (L4/A100) | Within-state pairwise applicability ranking loss ($w \in \{0.25, 0.50, 1.00\}$) | Cleared development recall (0.3451), but failed gate false-none (>0.21) and policy cost (0.10221); rejected | [Whitepaper §18.10](../docs/whitepaper/WHITEPAPER.md#1810-phase-4b3-pairwise-applicability-passes-development-but-not-transfer) |
| **19** | [`19_phase4c_decoupled_applicability_head.ipynb`](./19_phase4c_decoupled_applicability_head.ipynb) | Phase 4C (`4c.0.0`) | GPU (L4/A100) | Isolated applicability head fine-tuning over frozen B2 representations | Child epochs regressed NLL (+2.77%) and Brier (+2.10%) for only +2 true positives; parent epoch 0 retained | [`19_phase4c_decoupled_applicability_results/`](./19_phase4c_decoupled_applicability_results) |
| **20** | [`20_phase4d_evidence_aware_applicability.ipynb`](./20_phase4d_evidence_aware_applicability.ipynb) | Phase 4D (`4d.0.0`) | GPU (L4/A100) | 19-dim handcrafted evidence/uncertainty diagnostic residual over frozen parent | Strict JSON enforced; child gained only +1 true positive while NLL worsened +6.08%; child rejected | [`20_phase4d_evidence_aware_applicability_results/`](./20_phase4d_evidence_aware_applicability_results) |
| **21** | [`21_phase4e_qasper_error_audit.ipynb`](./21_phase4e_qasper_error_audit.ipynb) | Phase 4E (`4e.0.0`) | CPU | Blinded human audit of 150 QASPER false-negative, false-positive, and control cases | Prepares double-blind adjudication pack to separate representation failure from label ambiguity before 4E-B | [`21_phase4e_qasper_audit_results/`](./21_phase4e_qasper_audit_results) |
| **22** | [`22_phase4e_a2_qasper_audit_repair.ipynb`](./22_phase4e_a2_qasper_audit_repair.ipynb) | Phase 4E-A2 (`20260921T013558Z`, `qasper_error_audit_repair_s17`) | CPU | Repaired QASPER error/evidence audit with complete paper text; primary blinded review & adjudication packet generation | Validated hash chain, 66.4% decided agreement, 51 adjudication rows, asymmetric disagreement (44 challenged semantic_none, 6 representation/serialization defects); Phase 4E-B blocked pending benchmark/representation repair | [`22_phase4e_a2_qasper_audit_repair_results/`](./22_phase4e_a2_qasper_audit_repair_results) |
| **23** | [`23_phase4e_a3_qasper_followup.ipynb`](./23_phase4e_a3_qasper_followup.ipynb) | Phase 4E-A3 (`20260921T013558Z`, `qasper_followup_cpu_s17_v1`) | CPU | QASPER review follow-up, 87 numeric cell restorations, frozen-state evidence candidates, and external representation leads | Restored 87 numeric cells; 51-row V2 adjudication (33 answerable, 13 semantic-none, 5 ambiguous); 13 exact state candidate spans; 6 non-independent assistant follow-ups scoped; no benchmark mutations; Phase 4E-B training unauthorized | [`23_phase4e_a3_qasper_followup_results/`](./23_phase4e_a3_qasper_followup_results) |
| **24** | [`24_phase4e_a4_qasper_source_alignment.ipynb`](./24_phase4e_a4_qasper_source_alignment.ipynb) | Phase 4E-A4 (`20260921T013558Z`, `qasper_source_alignment_cpu_s17_v1`) | CPU | QASPER upstream source alignment preflight, span partition verification, and audit ledger V2 | Verified 12 development candidate spans (7 Qs) and 1 gate diagnostic span; quarantined 3 unresolved evidence cases; aligned with upstream allenai/qasper (train/val only); final unopened; training remains unauthorized | [`24_phase4e_a4_qasper_source_alignment_results/`](./24_phase4e_a4_qasper_source_alignment_results) |
| **25** | [`25_phase4e_b1_candidate_option_logit_audit.ipynb`](./25_phase4e_b1_candidate_option_logit_audit.ipynb) | Phase 4E-B.1 (`candidate_option_logit_gate16_s17_v2`) | GPU / CPU (FP32) | Constrained next-token candidate option readout and uncalibrated `Z` rejection over 16-state gate sample (N=325) | Answerable ContractNLI ranking improved (87.8% vs 31.1% StateQuery ref); uncalibrated `Z` failed semantic none (0/124 none recall); order reversal flipped winners on 6.6% of questions; QASPER accuracy 77.4% (below 84.9% majority baseline); model promotion rejected | [`25_phase4e_b1_candidate_option_logit_results/`](./25_phase4e_b1_candidate_option_logit_results) |
| **26** | [`26_phase4e_b2_candidate_ranking_sweep.ipynb`](./26_phase4e_b2_candidate_ranking_sweep.ipynb) | Phase 4E-B.2 (`candidate_detection_ranking_sweep_s17_n12_v1`) | GPU / CPU (FP32) | Candidate ranking (3 rankers) and logistic-calibrated semantic-none detection (3 detectors) sweep across 12-state splits (N=741) | Cache parity verified ($\le 3.3 \times 10^{-6}$); order-averaged ranking hit 87.0% gate accuracy on ContractNLI; ContractNLI none recall hit 75.0% but gate false-none was 42.6% (violating $\le 0.20$ guardrail); QASPER selected grid collapsed to 0/5 none recall; cross-source calibration diverged; final closed | [`26_phase4e_b2_candidate_ranking_sweep_results/`](./26_phase4e_b2_candidate_ranking_sweep_results) |

---

## Detailed Notebook Breakdown

### 1. Phase 2 Initial Feasibility Probe
* **File**: [`01_phase2_qwen35_probe.ipynb`](./01_phase2_qwen35_probe.ipynb)
* **Goal**: Establish basic end-to-end viability of using `Qwen/Qwen3.5-4B-Base` as a frozen feature extractor for a non-autoregressive decision classification head.
* **Methodology**:
  - Loaded `Qwen/Qwen3.5-4B-Base` in BF16 on Google Colab.
  - Extracted 2,560-dimensional hidden states from the final layer.
  - Implemented pooling functions (last token, mean pooling, max pooling).
  - Built and trained a toy PyTorch `DecisionHead` module (Linear layer with CrossEntropyLoss) over two synthetic decision examples.
* **Findings & Discoveries**:
  - Identified critical parameter count difference: `AutoModel` loaded a multimodal wrapper with vision tower (4.539B parameters), whereas `AutoModelForCausalLM` loaded the pure text causal LM backbone (4.206B parameters across 32 layers). Subsequent phases pinned the exact text backbone.
  - Confirmed that last-token hidden representations retain strong syntactic and semantic signals for discriminative scoring.

---

### 2. Phase 2B Benchmark: Frozen Decision Heads & Readout Baseline
* **File**: [`02_phase2b_benchmark.ipynb`](./02_phase2b_benchmark.ipynb)
* **Run ID**: `20260917T205849Z` (NVIDIA L4 GPU, native BF16)
* **What it Measured**:
  - Performance of frozen backbone feature pooling (last-token, mean-pool, max-pool) paired with Linear vs 2-layer MLP classification heads on MultiNLI (3-class natural language inference).
  - Empirical calibration via post-hoc temperature scaling (Guo et al., 2017) evaluated by negative log-likelihood (NLL) and Expected Calibration Error (ECE).
  - Wall-clock latency, token throughput, and memory consumption comparing direct decision head forward passes against token-by-token autoregressive generation.
* **Results**:
  - **Decision Accuracy**: Last-token pooling with a Linear head achieved **87.67% matched** and **87.33% mismatched** accuracy, outperforming mean pooling (74.67%) and max pooling (84.83%). MLP offered negligible advantage over linear.
  - **Latency / Throughput**: Decision forward pass completed in **~38.4 ms** versus **~1,714 ms** for autoregressive generation (~44.6× speedup; ~250× fewer FLOPs).
  - **Calibration Insight**: Fitted temperature ($T = 0.887$) worsened matched test NLL (0.380 $\to$ 0.428) and ECE. Established rule: post-hoc calibration must pass held-out validation gates before being applied.
* **Supporting Directory**:
  - [`02_phase2b_benchmark_results/`](./02_phase2b_benchmark_results)
  - Key files: [`openkind_phase2b_summary.md`](./02_phase2b_benchmark_results/openkind_phase2b_summary.md), `decision_benchmark.json`, `generation_benchmark.json`, `metrics.json`.

---

### 3. Phase 2C: Stability, Dynamic Choices, and Serving Gates
* **File**: [`03_phase2c_stability_dynamic_choice.ipynb`](./03_phase2c_stability_dynamic_choice.ipynb)
* **Run ID**: `20260917T222948Z` (NVIDIA L4 GPU)
* **What it Measured**:
  - Multi-seed training stability (seeds 17, 29, 43) across newly partitioned MultiNLI splits strictly decontaminated from Phase 2B premise groups.
  - Dynamic candidate choice scoring on Banking77: evaluating arbitrary runtime candidate descriptions rather than a fixed classification head.
  - Evaluated candidate-conditioned last token with shared affine projection (2,562 parameters) and a learned global `__none__` option logit.
  - Tested intent discovery on 57 seen training labels vs 20 withheld evaluation labels.
* **Results**:
  - MultiNLI stability confirmed: 87.0% matched / 88.8% mismatched test accuracy; dev NLL was 0.35297 (linear) vs 0.35472 (MLP).
  - Calibration gate strictly enforced: temperature scaling was rejected ($T=1.0$ retained) as it failed held-out gate criteria.
  - Dynamic choice head scored **80.8% accuracy on seen labels** and **64.6% on withheld labels**. The global `__none__` scalar yielded an **AUROC of 0.817** for identifying unrepresented intents.
* **Supporting Directory**:
  - [`03_phase2c_stability_results/`](./03_phase2c_stability_results)
  - Key files: [`openkind_phase2c_summary.md`](./03_phase2c_stability_results/openkind_phase2c_summary.md), `stability.json`, `dynamic_predictions_test_seen.json`, `dynamic_predictions_test_unseen.json`.

---

### 4. Phase 2D: Numerical Reference, None-Handling, and Request Benchmarks
* **File**: [`04_phase2d_numerics_none_handling.ipynb`](./04_phase2d_numerics_none_handling.ipynb)
* **Run ID**: `20260917T234417Z` (NVIDIA L4 GPU)
* **What it Measured**:
  - Numerical divergence across datatypes: default BF16, strict FP32 math, and TF32.
  - Layer-by-layer representation drift across the 32 decoder blocks.
  - Missing-answer (`__none__`) modeling: scalar logit vs dedicated MLP none-head vs post-hoc threshold rejection.
  - End-to-end request latency as candidate cardinality $K$ scales ($K \in \{2, 4, 8, 16, 32\}$).
* **Results**:
  - Numerical drift begins in early hybrid layers (DeltaNet recurrent states and 1D depthwise convolutions) and compounds across depth.
  - Dedicated none-head failed to improve out-of-scope discrimination over the simpler global scalar logit.
  - Without prefix caching, latency scaled linearly with $K$ (~25 ms per candidate), confirming the theoretical necessity of shared-prefix continuation caching.
* **Supporting Directory**:
  - [`04_phase2d_numerics_results/`](./04_phase2d_numerics_results)
  - Key files: [`openkind_phase2d_summary.md`](./04_phase2d_numerics_results/openkind_phase2d_summary.md), `layer_divergence.json`, `none_head_comparison.json`, `request_benchmarks.json`.

---

### 5. Phase 2E (Initial): Selective Precision & Shared-Prefix Policy
* **File**: [`05_phase2e_selective_precision_initial.ipynb`](./05_phase2e_selective_precision_initial.ipynb)
* **Run ID**: `20260918T032049180933Z` (Version `2e.1.1`)
* **What it Measured**:
  - Feasibility of module-level selective precision (promoting specific linear-attention or normalization modules to FP32 while leaving others in BF16).
  - Implementation of state-prefill KV caching where shared message context is prefilled once and candidate suffixes are evaluated by cloning cache state.
  - Operational acceptance/abstention policies based on minimum winning margin and entropy thresholds.
* **Results**:
  - An in-process memory guard halted the full FP32 stage due to running the notebook coordinator and GPU tensors in the same process space.
  - Demonstrated that cloning Qwen attention KV tensors alone is insufficient for state branching: hybrid recurrent DeltaNet states must also be isolated.
* **Supporting Directory**:
  - [`05_phase2e_selective_precision_results/`](./05_phase2e_selective_precision_results)
  - Key files: [`README_results.md`](./05_phase2e_selective_precision_results/README_results.md).

---

### 6. Phase 2E (Expanded): Isolated FP32 Reference & Batched Prefix Parity
* **File**: [`06_phase2e_expanded_batched_prefix_parity.ipynb`](./06_phase2e_expanded_batched_prefix_parity.ipynb)
* **Run ID**: `20260918T114914072764Z` (Version `2e.2.0`)
* **What it Measured**:
  - Executed coordinator and GPU workers in completely isolated Python processes, successfully executing the full FP32 reference path.
  - 128-episode factorial panel evaluating:
    1. Full-sequential execution (FP32 & BF16)
    2. Shared-prefix branching (FP32 & BF16)
    3. Suffix batching across candidates
  - Fine-grained breakdown: state prefill, state cloning, host-to-device transfer, suffix evaluation, and head readout.
* **Results**:
  - **Foundational Precision Invariant Established**:
    - Under strict FP32, shared-prefix branching matched sequential evaluation with **maximum probability delta $\le 1.1 \times 10^{-5}$** and zero class flips or policy changes.
    - Under BF16, shared-prefix branching **violated the 0.005 parity tolerance** and altered application-policy decisions.
    - Established that strict FP32 is the required mathematical parity reference for OpenKind engine verification.
* **Supporting Directory**:
  - [`06_phase2e_expanded_parity_results/`](./06_phase2e_expanded_parity_results)
  - Key files: [`README_results.md`](./06_phase2e_expanded_parity_results/README_results.md), `fp32_strict_math/parity_rows.json`, `bf16_default/parity_rows.json`, `component_profiles.json`.

---

### 7. Phase 2F: Cache Compression, TurboQuant Codecs, and Serving Trade-Offs
* **File**: [`07_phase2f_cache_compression_prefix_reuse.ipynb`](./07_phase2f_cache_compression_prefix_reuse.ipynb)
* **Run ID**: `20260918T224427722898Z` (Version `2f.1.0`)
* **What it Measured**:
  - Evaluated low-bit quantization of stored continuation states using external GPL-3.0 **0xSero/TurboQuant** codecs (4-bit, 3-bit, 2-bit variants) versus lossless FP32 and FP16-KV snapshots.
  - Byte-bounded GPU/CPU LRU prefix cache across multi-request traces under cold, warm, and zero-expiry conditions.
  - Acceptance gates: probability delta $\le 0.005$, identical top-1 argmax, and zero downstream policy action changes across 32 regression episodes.
* **Results**:
  - **Lossless FP32 and FP16-KV passed all 32 episodes with zero policy flips**.
  - **All four low-bit TurboQuant codecs failed the gates**, introducing severe probability distortions that flipped decisions and downstream application actions.
  - Proved that low-bit quantization of continuation state cannot be applied without end-to-end retraining. Bounded lossless prefix caching was adopted.
* **Supporting Directory**:
  - [`07_phase2f_cache_compression_results/`](./07_phase2f_cache_compression_results)
  - Key files: [`README_results.md`](./07_phase2f_cache_compression_results/README_results.md), `codec_regression.json`, `lru_trace_results.json`, `storage_benchmarks.json`.

---

### 8. Phase 2G: Fresh Evidence, TF32 Arithmetic, and Cache Lifecycle
* **File**: [`08_phase2g_fresh_evidence_tf32_cache.ipynb`](./08_phase2g_fresh_evidence_tf32_cache.ipynb)
* **Run ID**: `20260919T005142584348Z` (Version `2g.1.0`)
* **What it Measured**:
  - Generalization to fresh labeled messages: Banking77 fresh partition (416 episodes, 112 messages), CLINC150 non-financial domain transfer, and author-written out-of-scope (OOS) queries.
  - TF32-permitted FP32 execution on Ampere/Ada Tensor Cores versus strict FP32 IEEE math.
  - Persistent cache lifecycle: TTL expirations, eviction policies, and cache size bounds.
* **Results**:
  - **Rejection Limitation Identified**: On the CLINC panel, answerable accuracy was 93.75%, but omitted-intent recall was only 39.06% (author OOS recall was 46.88%). Revealed that candidate-conditioned scoring trained on in-domain tasks does not automatically generalize to out-of-domain rejection.
  - **TF32 Trade-off**: TF32 halved latency ($K=4$: 265 ms $\to$ 132 ms; $K=16$: 1,118 ms $\to$ 518 ms), but caused 3 argmax flips across 416 episodes. TF32 is valuable for high-throughput serving but cannot be used as the reference parity authority.
* **Supporting Directory**:
  - [`08_phase2g_fresh_evidence_results/`](./08_phase2g_fresh_evidence_results)
  - Key files: [`README_results.md`](./08_phase2g_fresh_evidence_results/README_results.md), `fresh_quality_summary.json`, `tf32_comparison.json`, `cache_lifecycle_trace.json`.

---

### 9. Phase 2H: Criteria Augmentation, Rejection, and Multi-Domain Transfer
* **File**: [`09_phase2h_criteria_rejection_multidomain.ipynb`](./09_phase2h_criteria_rejection_multidomain.ipynb)
* **Run ID**: `20260919T040612625670Z` (Version `2h.1.2`, continuation)
* **What it Measured**:
  - Multi-domain candidate head fitting and criteria transfer across Banking77 and CLINC150.
  - Comparing original criteria descriptions vs criteria augmented with few-shot training support examples.
  - Rigorous pre-registration discipline: development selection rule vs held-out final evaluation.
* **Results**:
  - The support-example joint head achieved 78.91% raw pooled accuracy (1,152 episodes), 85.55% on Banking fitting labels, and 89.84% on CLINC fitting domains, but dropped to 63.67% on held-out Banking labels and 66.02% on held-out CLINC domains.
  - **Methodological Victory**: The original-criteria control head transferred significantly better on held-out data (83.77% accuracy / 0.541 NLL) than the development-selected support-augmented head. Following pre-registration rules, the project did *not* retroactively swap winners, using this as motivation for architectural model selection in Phase 2I/2J.
* **Supporting Directory**:
  - [`09_phase2h_criteria_rejection_results/`](./09_phase2h_criteria_rejection_results)
  - Key files: [`README_results.md`](./09_phase2h_criteria_rejection_results/README_results.md), `final_quality_report.json`, `comparison_arms.json`.

---

### 10. Phase 2I/2J: Gated Multi-Question Workbench
* **File**: [`10_phase2ij_gated_workbench.ipynb`](./10_phase2ij_gated_workbench.ipynb)
* **Version**: `2ij.1.0` (`10_phase2ij_gated_workbench_results`)
* **What it Measured**:
  - Setup of a formal review gate requiring independent human audit of task criteria and split manifests before launching expensive training.
  - Small synthetic GPU probe validating **state-first segmented tokenization**:
    $$\text{Tokens} = [\text{State Prefix}] + [\text{Question Suffix}] + [\text{Candidate Suffix}]$$
    contrasted with standard instruction-first rendering.
* **Results**:
  - The review gate successfully blocked unreviewed multi-question training, preventing unverified claims.
  - Synthetic GPU probe validated state-first mechanics: prefilling the state document once and branching questions/candidates reduced compute without token leakage.
* **Supporting Directory**:
  - [`10_phase2ij_gated_workbench_results/`](./10_phase2ij_gated_workbench_results)
  - Key files: [`REPORT.md`](./10_phase2ij_gated_workbench_results/REPORT.md), `gate_report.json`, `contracts/SERVICE_HANDOFF.md`.

---

### 11. Phase 2I/2J: Model Selection Screen & Architectural Decision
* **File**: [`11_phase2ij_model_selection_screen.ipynb`](./11_phase2ij_model_selection_screen.ipynb)
* **Version**: `2ij.2.0` (`11_phase2ij_model_selection_results`)
* **What it Measured**:
  - Extensive screening across 13 model fit jobs and 31 evaluation profiles.
  - Model families evaluated:
    - `Qwen/Qwen3.5-4B-Base` (frozen state-first vs instruction-first, score-summary head)
    - `Qwen/Qwen2.5-1.5B` & `2B` (frozen, online adaptation, LoRA)
    - `ModernBERT` (frozen & fine-tuned bidirectional encoder baselines)
    - Lexical baseline controls
* **Results**:
  - **Locked Native Profile `a047d6802c3f06f085b8`**:
    - Backbone: `Qwen/Qwen3.5-4B-Base` (commit `1001bb4d826a52d1f399e183466143f4da7b741b`).
    - Rendering: `state_first`.
    - Readout Head: Score-summary rejection head with normalization, projection, rejection scalar, temperature calibration ($T = 1.81868$), and policy threshold (0.98).
  - ModernBERT and Qwen2.5 baselines trailed Qwen3.5-4B on dynamic candidate expressivity and state prefill compatibility.
  - This locked profile became the permanent target for native Rust and Metal implementations.
* **Supporting Directory**:
  - [`11_phase2ij_model_selection_results/`](./11_phase2ij_model_selection_results)
  - Key files: [`REPORT.md`](./11_phase2ij_model_selection_results/REPORT.md), [`MODEL_DECISION.md`](./11_phase2ij_model_selection_results/MODEL_DECISION.md), `contracts/SERVICE_HANDOFF.md`.

---

### 12. Phase 2I/2J: A100 Selected-Model Bundle Export
* **File**: [`12_phase2ij_a100_bundle_export.ipynb`](./12_phase2ij_a100_bundle_export.ipynb)
* **Goal**: High-memory export continuation on an NVIDIA A100 GPU to export the full model bundle for profile `a047d6802c3f06f085b8` without risk of VRAM fragmentation or host memory limits.
* **Outputs**:
  - Exported complete self-contained reference bundle:
    - SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`.
    - Published public reference repository: `cowWhySo/OpenKind-Qwen3.5-4B-StateFirst`.
  - Exported token fixtures, configuration JSONs, and model weights.

---

### 13. Phase 3A: Branchable Hybrid State & Batched-Q Execution
* **File**: [`13_phase3a_branchable_state_batched_q.ipynb`](./13_phase3a_branchable_state_batched_q.ipynb)
* **Run ID**: `20260920T024056Z` (NVIDIA L4 / A100)
* **What it Measured**:
  - Formalized the complete hybrid state branching abstraction for Qwen 3.5: attention KV cache + DeltaNet recurrent state + convolution state.
  - Compared three execution strategies across $Q \in \{1, 4, 16\}$ questions and $K \in [2, 255]$ candidates:
    1. `repeated_full`: Sequential independent evaluation of every complete sequence.
    2. `nested_sequential`: Prefills root state once, sequentially forks question lanes and candidate lanes.
    3. `nested_batched`: Prefills root state once, fans out question lanes (`fork_batch`) and candidate lanes in parallel batches.
* **Results**:
  - Validated that cloning all three state families provides strict mathematical isolation (zero cross-branch leakage).
  - **Workload Crossover Discovery**:
    - For short documents and small $Q$ ($Q \le 4$), `nested_sequential` is faster due to low kernel dispatch overhead.
    - As state length grows ($>500$ tokens) and $Q$ expands ($Q \ge 8$), `nested_batched` delivers massive speedups at the expense of peak VRAM.
    - Directly informed the implementation of the capability-aware `choose_strategy` scheduler in `openkind-runtime`.
* **Supporting Documentation**:
  - [OpenKind Whitepaper §15](../docs/whitepaper/WHITEPAPER.md#15-phase-3a-branchable-hybrid-state-and-batched-q-execution).

---

### 14. Phase 3B: Qwen3.5 Backbone Parity Reference for Rust
* **File**: [`14_phase3b_backbone_parity.ipynb`](./14_phase3b_backbone_parity.ipynb)
* **Run ID**: `20260920T152206Z` (FP32 CPU & GPU)
* **What it Measured**:
  - Exported exact ground-truth golden fixtures from PyTorch/Transformers to guide the pure Rust CPU backbone implementation (`crates/openkind-backends`).
  - Traced intermediate activations across all 32 layers for a diagnostic candidate sequence.
  - Verified cached continuation against fresh full-sequence inference.
* **Results**:
  - Exported 4 exact token-fixture records and 47 FP32 golden vectors (34 layer trace stages, 10 candidate features, 3 continuation vectors).
  - Largest fresh-feature difference between Python and Rust reference was $4.96 \times 10^{-5}$.
  - Cached continuation differed from fresh full-sequence execution by at most $1.91 \times 10^{-5}$.
  - Locked the parity gates: Bundle SHA-256 match, max probability difference $\le 0.005$, argmax rank ordering tolerance $\le 10^{-5}$.
* **Supporting Directory**:
  - [`14_phase3b_backbone_parity_results/`](./14_phase3b_backbone_parity_results)
  - Key files: [`RUST_BACKBONE_HANDOFF.md`](./14_phase3b_backbone_parity_results/RUST_BACKBONE_HANDOFF.md), `TOKEN_FIXTURES.json`, `PROBABILITY_REFERENCE.json`, `CONTINUATION_TRACE.json`, `QWEN35_BACKBONE_GOLDEN.safetensors`.

---

### 15. Phase 4A.0 / 4A.1: Natural-Document Corpus & StateQuery Workbench
* **File**: [`15_phase4a_statequery_workbench.ipynb`](./15_phase4a_statequery_workbench.ipynb)
* **Run ID**: `20260921T013558Z` (Version `4a.0.2`)
* **What it Measured**:
  - **Phase 4A.0**: Assembled and locked a natural multi-question document benchmark combining **ContractNLI** (legal non-disclosure contracts) and **QASPER** (NLP research papers): 2,192 states, 15,368 questions, 25,687 options, 21,894 evidence rows. Partitions split strictly by document group (zero test leakage).
  - **Phase 4A.1**: Investigated whether a small learned query network (**StateQuery B1**) can evaluate questions directly over a **frozen, state-only Qwen representation** without re-encoding candidate descriptions.
* **Results**:
  - Corpus locked with SHA-256 table manifests before evaluation.
  - StateQuery B1 improved source-macro accuracy (0.7796 vs 0.3766 for historical candidate reference) and reduced NLL (0.5427 vs 1.6734).
  - **Critical Failure Mode Discovered**: StateQuery B1 collapsed on QASPER semantic-none recall to **0.0** (defaulted to predicting answerable options). Proved that aggregate macro accuracy can obscure catastrophic rejection collapse. Model promotion was halted.
* **Supporting Documentation**:
  - [OpenKind Whitepaper §18](../docs/whitepaper/WHITEPAPER.md#18-phase-4a-natural-document-multi-question-benchmark-and-statequery-readout).

---

### 16. Phase 4A.2: StateQuery Model Comparison & Balance Optimization
* **File**: [`16_phase4a2_statequery_model_comparison.ipynb`](./16_phase4a2_statequery_model_comparison.ipynb)
* **Version**: `4a.2.0`
* **What it Measured**:
  - Evaluated three model architectures under identical data partitions without opening final test splits:
    1. **B0 (Matched Control)**: Candidate-conditioned architecture with newly supervised heads.
    2. **B1 (Balanced StateQuery)**: Direct StateQuery with balanced loss weighting and semantic-none penalty.
    3. **B2 (Factorized StateQuery)**: StateQuery with factorized evidence slots.
* **Results**:
  - B0 matched control achieved source-macro accuracy 0.7283 / NLL 0.6211, but also collapsed to 0.0 QASPER none-recall.
  - Balanced B1 training raised development QASPER none-recall to 0.1681, moving toward the required 0.30 floor, but remained below the promotion threshold.
  - Reaffirmed project governance: final test splits remain locked until development models satisfy all gate conditions.
* **Supporting Documentation**:
  - [OpenKind Whitepaper §18.4](../docs/whitepaper/WHITEPAPER.md#184-completed-non-final-comparison-aggregate-improvement-hides-a-qasper-collapse).

---

### 17. Phase 4B.2: StateQuery Model Comparison & Scalar-Weight Saturation
* **File**: [`17_phase4b2_statequery_model_comparison.ipynb`](./17_phase4b2_statequery_model_comparison.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4b.2.0`, arm `b2q_weight12_s17`, followed by `4b.2.1` `b2_applicability_stratified_s17`)
* **Target HW**: GPU (NVIDIA L4 / A100)
* **What it Measured**:
  - Testing whether raising the configured scalar QASPER semantic-none weight cap from 8 to 12 can rescue missing-answer recall.
  - Evaluating empirical effective loss weight saturation against the natural answerable-to-none class imbalance ratio.
  - Follow-on objective experiment (`4b.2.1`): replacing the single joint scalar choice loss with decoupled answerable candidate ranking plus source/class-stratified binary applicability BCE.
* **What Was Confirmed**:
  - **Saturation Confirmed**: Raising the cap to 12 saturated at an empirical realized weight of **8.5602**; higher caps cannot increase weight under empirical sampling.
  - **Stratified Objective Confirmed**: The source/class-stratified applicability loss substantially improved QASPER operating recall to **0.2564** on policy development and **0.3111** on the calibration gate, while ContractNLI maintained 0.5751 recall with only 0.1626 false-none.
  - **In-Scope Discrimination Confirmed**: Factorized B2 architecture reliably learned answerable candidate ranking across both legal and scientific text.
* **What Was Not Confirmed / Disproved**:
  - **Scalar Recovery Disproved**: Larger scalar weights failed to resolve the cross-source conflict. While ContractNLI none recall rose to 0.8399, 278 of 701 answerable ContractNLI questions were falsely rejected (0.3966 false-none rate). The scalar weight sweep was permanently closed.
  - **Policy Transfer Disproved**: The stratified model accepted 14 development decisions with 0 errors (cost 0.09858), but on the untouched calibration gate accepted 32 decisions with 7 errors, driving cost to **0.11675** (worse than the 0.10 review-all ceiling). The model remained ineligible for release promotion.
* **Supporting Directory**:
  - [`17_phase4b2_statequery_model_comparison_results/`](./17_phase4b2_statequery_model_comparison_results/)
  - Key files: [`b2q_weight12_s17/TRAINING_REPORT.json`](./17_phase4b2_statequery_model_comparison_results/20260921T013558Z/b2q_weight12_s17/TRAINING_REPORT.json), [`b2_applicability_stratified_s17/NONFINAL_RESULT_LOCK.json`](./17_phase4b2_statequery_model_comparison_results/20260921T013558Z/b2_applicability_stratified_s17/NONFINAL_RESULT_LOCK.json), `comparison_snapshot_20260922T115817Z.json`.

---

### 18. Phase 4B.3: Applicability Ranking & Pairwise Loss Sweep
* **File**: [`18_phase4b3_applicability_ranking_sweep.ipynb`](./18_phase4b3_applicability_ranking_sweep.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4b.3.0`, locked parent `b2_applicability_stratified_s17`, arm `b2_app_pairwise_w025_s17`)
* **Target HW**: GPU (NVIDIA L4 / A100)
* **What it Measured**:
  - Evaluating whether QASPER unanswerability is fundamentally a score-separation ranking problem.
  - Added a source-balanced within-state pairwise margin loss ($L_{\text{pairwise}}$ with weights 0.25, 0.50, 1.00) between answerable and unanswerable questions over the same state document.
  - Evaluated development operating points under a strict $\le 0.20$ false-none guardrail.
* **What Was Confirmed**:
  - **Development Separation Confirmed**: Under the 0.25 pairwise weight, development QASPER none recall cleared the pre-registered floor for the first time, reaching **0.3451** (39/113) with a 0.1936 false-none rate, while ContractNLI reached 0.5811 recall (0.1971 false-none).
  - **Within-State Consistency Confirmed**: Pairwise loss prevented probability mass from pooling uniformly across questions on dense states.
* **What Was Not Confirmed / Disproved**:
  - **Threshold Generalization Disproved**: The development gains failed to transfer to the calibration gate. Both sources exceeded the 0.20 false-none ceiling (ContractNLI: 0.2111; QASPER: 0.2110).
  - **Gate Discrimination Disproved**: Raw-choice QASPER none recall collapsed to 0.0 on the calibration gate, and ROC AUC remained weak (0.5925).
  - **Policy Transfer Disproved**: The transferred policy accepted 8 gate decisions with 1 error, yielding a cost of **0.10221** ($>0.10$). Pairwise loss was officially rejected as a non-generalizing local fit.
* **Supporting Documentation**:
  - [OpenKind Whitepaper §18.10](../docs/whitepaper/WHITEPAPER.md#1810-phase-4b3-pairwise-applicability-passes-development-but-not-transfer).

---

### 19. Phase 4C: Decoupled Applicability Head Recovery
* **File**: [`19_phase4c_decoupled_applicability_head.ipynb`](./19_phase4c_decoupled_applicability_head.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4c.0.0`, locked parent Phase 4B.2.1 epoch 8, arm `b2_app_headonly_guard18_s17`)
* **Target HW**: GPU (NVIDIA L4 / A100)
* **What it Measured**:
  - Investigating whether multi-task gradient interference between candidate ranking and applicability heads degraded rejection performance.
  - Froze all underlying B2 representations, query attention stacks, and candidate heads; fine-tuned only the 1,771,009-parameter applicability MLP head using source/class-balanced BCE.
  - Parent checkpoint was evaluated and registered as epoch 0, establishing an explicit non-regression baseline.
* **What Was Confirmed**:
  - **Parent Optimality Confirmed**: The unchanged parent (epoch 0) outperformed every fine-tuned child on development NLL (parent: 0.56457; child epochs: 0.58351, 0.59079, 0.58022).
  - **Representation Bottleneck Confirmed**: Multi-task gradient conflict was ruled out as the primary cause of rejection failure; the limitation lies in the pooled feature representations themselves.
* **What Was Not Confirmed / Disproved**:
  - **Head-Only Recovery Disproved**: Fine-tuning the head in isolation failed to yield viable candidates. The best child (epoch 3) gained only 2 additional QASPER true positives (34 $\to$ 36 of 113) while degrading source-macro NLL by +2.77% and Brier by +2.10%.
  - **Child Promotion Disproved**: All child checkpoints were rejected; epoch 0 was retained. Concluded that further head-only hyperparameter tuning on frozen pooled features is an unproductive route.
* **Supporting Directory**:
  - [`19_phase4c_decoupled_applicability_results/`](./19_phase4c_decoupled_applicability_results/)
  - Key files: [`b2_app_headonly_guard18_s17/TRAINING_REPORT.json`](./19_phase4c_decoupled_applicability_results/20260921T013558Z/b2_app_headonly_guard18_s17/TRAINING_REPORT.json), [`b2_app_headonly_guard18_s17/NONFINAL_RESULT_LOCK.json`](./19_phase4c_decoupled_applicability_results/20260921T013558Z/b2_app_headonly_guard18_s17/NONFINAL_RESULT_LOCK.json), `comparison_snapshot_20260922T172543Z.json`.

---

### 20. Phase 4D: Evidence-Aware Applicability Residual
* **File**: [`20_phase4d_evidence_aware_applicability.ipynb`](./20_phase4d_evidence_aware_applicability.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4d.0.0`, locked parent Phase 4C epoch 0, arm `b2_evidence_residual_s17`)
* **Target HW**: GPU (NVIDIA L4 / A100)
* **What it Measured**:
  - Tested whether 19 runtime diagnostic features (candidate entropy, rank margin, question-candidate cosine stats, question-state cosine stats, and cross-attention entropy/top-4 mass) could furnish the missing applicability signal.
  - Added a zero-initialized 19 $\to$ 64 $\to$ 1 shallow residual MLP (1,345 parameters) to the frozen parent applicability logit.
  - Implemented strict JSON data hygiene (mapping non-standard floating-point `NaN`/`Inf` to standard JSON `null`).
* **What Was Confirmed**:
  - **Strict JSON Contract Confirmed**: Zero IEEE non-standard floats; complete compatibility with strict JSON parsers and audit hashing.
  - **Zero-Initialization Safety Confirmed**: Epoch 0 replicated parent predictions with bit-level mathematical identity.
* **What Was Not Confirmed / Disproved**:
  - **Diagnostic Feature Sufficiency Disproved**: Handcrafted statistical and attention heuristics failed to separate applicable from unanswerable questions. The best child (epoch 1) gained only a single QASPER true positive (34 $\to$ 35 of 113), falling 5 short of the 40 required.
  - **Proper-Score Stability Disproved**: Source-macro development NLL deteriorated by **+6.08%** (0.56457 $\to$ 0.59888) and Brier by **+6.60%** (0.33934 $\to$ 0.36172).
  - **Training Signal Generalization Disproved**: Training loss proxy steadily decreased from 0.75076 to 0.65405 while validation metrics worsened, demonstrating rapid memorization of surface heuristics. All 8 child epochs were rejected; status locked as `nonfinal_failed_final_unavailable`.
* **Supporting Directory**:
  - [`20_phase4d_evidence_aware_applicability_results/`](./20_phase4d_evidence_aware_applicability_results/)
  - Key files: [`b2_evidence_residual_s17/TRAINING_REPORT.json`](./20_phase4d_evidence_aware_applicability_results/20260921T013558Z/b2_evidence_residual_s17/TRAINING_REPORT.json), [`b2_evidence_residual_s17/NONFINAL_RESULT_LOCK.json`](./20_phase4d_evidence_aware_applicability_results/20260921T013558Z/b2_evidence_residual_s17/NONFINAL_RESULT_LOCK.json), `comparison_snapshot_20260922T200741Z.json`.

---

### 21. Phase 4E: QASPER Error & Evidence Audit
* **File**: [`21_phase4e_qasper_error_audit.ipynb`](./21_phase4e_qasper_error_audit.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4e.0.0`, run label `qasper_error_audit_s17`)
* **Target HW**: CPU (zero GPU compute; analytical audit)
* **What it Measured**:
  - Conducted a blinded human audit across 150 QASPER episodes to determine what portion of missing-answer errors stem from ground-truth annotation ambiguity versus genuine model representation failure.
  - Partition breakdown:
    - 70 policy-development errors (29 false negatives, 41 false positives).
    - 60 calibration-gate sampled errors (30 false negatives, 30 false positives via deterministic state round-robin).
    - 20 policy-development controls (10 true positives, 10 true negatives).
  - Complete double-blind protocol: hidden labels, model scores, predictions, and error classes with cryptographically locked key (`AUDIT_KEY.parquet`).
* **What Was Confirmed**:
  - **Weak Separability Confirmed**: Retained parent exhibited weak ROC AUC (0.5903 policy development / 0.6173 calibration gate) at ~12–13% base rate, proving mathematically that threshold calibration alone cannot achieve safe policy automation.
  - **Audit Protocol Locked Confirmed**: 150 items fully exported to blinded CSV, ground-truth key SHA-256 locked in `AUDIT_CONTRACT.json`, and final splits kept strictly unopened.
* **What Was Not Confirmed / Blocked**:
  - **Upstream Representation Retraining Blocked**: Phase 4E-B (upstream token-level attention adaptation) remains strictly blocked until human review quantifies paper-level annotation ambiguity and establishes verified target bounds.
* **Supporting Directory**:
  - [`21_phase4e_qasper_audit_results/`](./21_phase4e_qasper_audit_results/)
  - Key files: [`AUDIT_GUIDE.md`](./21_phase4e_qasper_audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_GUIDE.md), [`AUDIT_REVIEW_BLINDED.csv`](./21_phase4e_qasper_audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_REVIEW_BLINDED.csv), [`AUDIT_ANALYSIS.json`](./21_phase4e_qasper_audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_ANALYSIS.json), [`AUDIT_RESULT_LOCK.json`](./21_phase4e_qasper_audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_RESULT_LOCK.json).

---

### 22. Phase 4E-A2: Repaired QASPER Error & Evidence Audit
* **File**: [`22_phase4e_a2_qasper_audit_repair.ipynb`](./22_phase4e_a2_qasper_audit_repair.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4e.a2.0`, run label `qasper_error_audit_repair_s17`)
* **Target HW**: CPU (zero GPU compute; analytical audit)
* **What it Measured**:
  - Repaired the earlier broken audit packet where missing state text inflated ambiguity to 52.7%. Restored full state paper text (median 24,812 chars, up to 98,130 chars) with verified SHA-256 hashes across all 150 rows.
  - Completed blinded primary review across 150 QASPER episodes (70 policy-development errors, 60 calibration-gate errors, 20 policy-development blinded controls) without access to gold evidence, model predictions, or locked targets.
  - Evaluated agreement between primary human review (`answerable`, `semantic_none`, `ambiguous`) and locked benchmark targets.
  - Generated frozen adjudication packet (`AUDIT_ADJUDICATION_PACKET.csv`) for all 51 disagreed or ambiguous rows.
* **What Was Confirmed**:
  - **Audit Integrity & Pipeline Validation Confirmed**: Hash chain, joins, row counts, and adjudication selection all validate. No NaN values, duplication, or corruption issues found. Ambiguity collapsed from 52.7% (broken audit) to 0.67% (1 of 150) once full text context was supplied. Final splits remain closed (`final_opened: false`).
  - **Decided Agreement Baseline**: 99 of 149 decided cases agreed (66.4% overall agreement). High agreement on locked answerable targets (75/81 = 92.6%), but very low agreement on locked semantic-none targets (24/68 = 35.3%).
  - **Asymmetry in Missing-Answer Annotations**: Disagreement is highly asymmetric: 44 cases had reviewer answerable vs locked `semantic_none`, 6 cases had reviewer `semantic_none` vs locked answerable, and 1 case was genuinely ambiguous. Crucially, among the 44 challenged semantic-none labels, 38 were judged explicitly answerable and 6 implicitly answerable; 28 already had non-empty gold evidence attached in the source dataset, and 16 had clear answers in the paper text despite missing evidence annotations. This confirms that missing-option errors are heavily driven by incomplete or erroneous dataset labels rather than model incapacity alone.
* **What Was Not Confirmed / Disproved**:
  - **Representation Sufficiency Disproved**: In all 6 cases where the locked target was answerable but the reviewer found `semantic_none`, critical information was omitted during state serialization (numeric results only in omitted tables, table captions without table values, bibliography placeholders instead of model names, and code-mixed text without second-language identification). Passing state-text length and character hash checks did not guarantee answer-bearing representation coverage.
  - **Immediate Phase 4E-B Sweep Disproved**: Automatically authorizing Phase 4E-B representation retraining was rejected (`phase4e_b_automatically_authorized: false`). Model fine-tuning cannot recover information never serialized into state representations.
* **Decision Semantics Note**:
  - `model_prediction_label` is the raw 0.5 decision, while `operating_outcome` uses the frozen 0.210894 threshold. Seven adjudication rows differ between those two decisions. The calculations are verified correct; future outputs should rename this field to `argmax_prediction_label` and add an explicit `operating_prediction_label`.
* **Recommended Next Step & Defect Taxonomy**:
  - Do not run another parameter sweep or authorize Phase 4E-B yet.
  - The 51 rows require independent adjudication under a structured defect taxonomy:
    1. Incorrect semantic-none label
    2. Missing or incomplete gold evidence
    3. Table/float serialization loss
    4. Bibliography/reference-resolution loss
    5. Genuine underspecification
    6. Primary-review error
  - Final adjudication must be performed by a fresh reviewer or isolated adjudication run to preserve independence. A mixed repair (label/evidence cleanup and table/reference serialization) is required before any preregistered seed-17 model arm.
* **Supporting Directory & Artifacts**:
  - Local Directory: [`22_phase4e_a2_qasper_audit_repair_results/20260921T013558Z/qasper_error_audit_repair_s17/`](./22_phase4e_a2_qasper_audit_repair_results/20260921T013558Z/qasper_error_audit_repair_s17/)
  - Adjudication Packet: [`AUDIT_ADJUDICATION_PACKET.csv`](https://drive.google.com/file/d/1eu6evYz9JzRQBSr9eJa5g_XpK6szIKE1/view) (SHA-256 `2d6b3be4dfa532de7c0c320d4e84b67d2be571ff4ddcc7f886b22f437e88edeb`)
  - Analysis JSON: [`AUDIT_ANALYSIS.json`](https://drive.google.com/file/d/1-UTO4L8gRy5NrbAracI7o2jGxJoXyGu4/view) (SHA-256 `858d1b7b08d2e3e9969f7de4bdc1780cbaf83a831c379b9529d4acfe2a84b25f`)
  - Result Lock: [`AUDIT_RESULT_LOCK.json`](https://drive.google.com/file/d/1n7Xb9Ye_ek9kyHt1t6yRqKi1GMgJXY_c/view) (SHA-256 `89d0e283a2b743efcb95de3f73189c6fb0157620c5f6252beea149a44d1ecefd`)

---

### 23. Phase 4E-A3: QASPER Review Follow-up & Adjudication V2
* **File**: [`23_phase4e_a3_qasper_followup.ipynb`](./23_phase4e_a3_qasper_followup.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4e.a3.0`, run label `qasper_followup_cpu_s17_v1`)
* **Target HW**: CPU (zero GPU compute; analytical audit & repair recovery)
* **What it Measured**:
  - Repaired the CSV representation of 87 source numeric cells that were truncated or reformatted during export from the Phase 4E-A2 packet.
  - Processed 51 rows sent to independent adjudication under the six-part defect taxonomy, generating V2 adjudication artifacts (`AUDIT_ADJUDICATION_REPORT_V2.json`).
  - Extracted 13 exact, uniquely located character spans from the frozen state text across 8 questions (`EVIDENCE_SPAN_CANDIDATES.csv`) and analyzed three questions with explicit QA gaps (`EVIDENCE_SPAN_QA.json`).
  - Documented 5 representation defects from omitted tables, figures, and bibliography entries (`REPRESENTATION_SOURCE_CANDIDATES.csv`).
  - Scoped 6 same-assistant follow-up reviews as explicitly non-independent.
* **What Was Confirmed / Proved**:
  - **Audit Integrity & Hash Chain Confirmed**: Input checksums validated across packet (`2d6b3be4...edeb`), primary review, repaired audit key, and immutable Phase 4A corpus. 87 numeric source cells round-tripped cleanly.
  - **Adjudication Distribution Confirmed**: 51-row V2 distribution locked at 33 answerable, 13 semantic-none, 5 ambiguous. Confirmed defect breakdown: 22 annotation label errors, 11 gold evidence incomplete, 8 insufficient state evidence, 4 missing table values, 5 question underspecified, 1 unresolved reference placeholder.
  - **Asymmetric Annotation Disagreement Confirmed**: Reaffirmed that 38/44 challenged semantic-none labels were judged explicitly answerable and 6 implicitly answerable; 28 already had non-empty gold evidence in the source dataset, and 16 had clear answers in state text.
  - **No Benchmark Mutations Confirmed**: Zero mutations to original gold evidence, labels, states, or final test splits (`final_opened: false`).
* **What Was Not Confirmed / Disproved**:
  - **External Representation Integration Disproved**: External source leads (e.g. NarrativeQA Table 5 Masque scores, OpenTapioca Figure 2 F1, LCF-ATEPC Tables 3–4, User embeddings Table 3, BLI paper §3) cannot be directly inserted into state text without verified PDF version alignment. Marked as `external_candidate_not_integrated_version_unverified`.
  - **Immediate Phase 4E-B Retraining Blocked**: Authorizing Phase 4E-B representation retraining was explicitly rejected (`automatic_phase4e_b_authorization: false`). Version-aligned source verification and benchmark repair must precede any training.
* **Supporting Directory & Key Artifacts**:
  - Local Directory: [`23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/)
  - Key files: [`PHASE4E_A3_DECISION_RECORD.md`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/PHASE4E_A3_DECISION_RECORD.md), [`AUDIT_ADJUDICATION_REPORT_V2.json`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/AUDIT_ADJUDICATION_REPORT_V2.json), [`PHASE4E_A3_NONFINAL_REPORT.json`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/PHASE4E_A3_NONFINAL_REPORT.json), [`AUDIT_REPAIR_LEDGER.csv`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/AUDIT_REPAIR_LEDGER.csv), [`EVIDENCE_SPAN_CANDIDATES.csv`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/EVIDENCE_SPAN_CANDIDATES.csv), [`REPRESENTATION_SOURCE_CANDIDATES.csv`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/REPRESENTATION_SOURCE_CANDIDATES.csv), [`PHASE4E_A3_LOCK.json`](./23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/PHASE4E_A3_LOCK.json).

---

### 24. Phase 4E-A4: QASPER Source Alignment & Repair Preflight
* **File**: [`24_phase4e_a4_qasper_source_alignment.ipynb`](./24_phase4e_a4_qasper_source_alignment.ipynb)
* **Run ID**: `20260921T013558Z` (Workbench `4e.a4.0`, run label `qasper_source_alignment_cpu_s17_v1`)
* **Target HW**: CPU (zero GPU compute; data verification & preflight)
* **What it Measured**:
  - Rechecking candidate span partition membership: verifying which of the 13 candidate spans belong to policy development versus calibration gate.
  - Quarantining 3 unresolved evidence questions where paper text provides partial or ambiguous support.
  - Emitting `AUDIT_REPAIR_LEDGER_V2.csv` across all 150 audited rows.
  - Checking upstream alignment with pinned `allenai/qasper` parquet commit `06806e4608976fc2fac0a090ac425d5b2b29caf4` for 4 policy-development papers covering missing-table and unresolved-evidence cases.
* **What Was Confirmed / Proved**:
  - **Partition Verification Confirmed**: Rechecking offsets against `states.parquet` verified that exactly **12 spans across 7 questions** belong to policy development; the 1 remaining span/question belongs to calibration gate and is kept diagnostic only.
  - **Quarantine Policy Confirmed**: 3 unresolved cases (`ok4e2_2e059c32a14d29c3`, `ok4e2_63fa0f4da9616df9`, `ok4e2_1f5356f7d2f80896`) were quarantined as `quarantine_unresolved_evidence`.
  - **Policy-Development Disposition Lock**: For the 90 policy-development rows: 17 candidate label/evidence repairs, 3 unresolved-evidence quarantines, 2 ambiguous quarantines, 3 low-confidence follow-up quarantines, 2 representation-source requirements, 63 no-label-change rows. All 60 calibration-gate rows remain diagnostic only.
  - **Upstream Hygiene Confirmed**: Downloaded strictly upstream train and validation parquet files; upstream test parquet containing closed final data was never downloaded (`final_opened: false`).
* **What Was Not Confirmed / Disproved**:
  - **Upstream Equivalence Disproved**: Exact state spans and dataset parquet alignment do not establish published PDF version equivalence, recover missing table float values, or provide an end-to-end table/reference serializer.
  - **Phase 4E-B Retraining Blocked**: Kept `automatic_training_authorization: false`. A future arm requires a source-aligned representation provenance and reviewed label/evidence contract before any training.
* **Supporting Directory & Key Artifacts**:
  - Local Directory: [`24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/`](./24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/)
  - Key files: [`PHASE4E_A4_PRELIMINARY_RECORD.md`](./24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/PHASE4E_A4_PRELIMINARY_RECORD.md), [`PHASE4E_A4_PREFLIGHT.json`](./24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/PHASE4E_A4_PREFLIGHT.json), [`AUDIT_REPAIR_LEDGER_V2.csv`](./24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/AUDIT_REPAIR_LEDGER_V2.csv), [`EVIDENCE_SPAN_POLICY_CANDIDATES.csv`](./24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/EVIDENCE_SPAN_POLICY_CANDIDATES.csv), [`QASPER_SOURCE_ROW_ALIGNMENT.csv`](./24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/QASPER_SOURCE_ROW_ALIGNMENT.csv), [`QASPER_SOURCE_CAPTION_CANDIDATES.csv`](./24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/QASPER_SOURCE_CAPTION_CANDIDATES.csv).

---

### 25. Phase 4E-B.1: Candidate-Conditioned Option-Logit Audit
* **File**: [`25_phase4e_b1_candidate_option_logit_audit.ipynb`](./25_phase4e_b1_candidate_option_logit_audit.ipynb)
* **Run ID**: `candidate_option_logit_gate16_s17_v2` (Version `4e.b.1`, pinned `Qwen/Qwen3.5-4B-Base` FP32, revision `1001bb4d826a52d1f399e183466143f4da7b741b`)
* **Target HW**: GPU / CPU (Qwen3.5-4B FP32 text graph, no TF32, zero generation)
* **What it Measured**:
  - Constrained next-token option-letter readout (listing candidate options in the prompt and evaluating next-token logits over option letters plus `Z` for semantic none, JevK5 style) directly on the frozen Base backbone.
  - Matched-row method comparison against the frozen Phase 4A StateQuery reference on 16 calibration-gate states per source (N=325 questions: 272 ContractNLI, 53 QASPER).
  - Cache parity between cached hybrid continuation and fresh full-prompt forward passes.
  - Positional sensitivity to candidate presentation order (reversing candidate order).
* **What Was Confirmed / Proved**:
  - **Strong Conditional Candidate Ranking**: On answerable ContractNLI rows (N=148, two candidates), conditional option-logit accuracy reached **87.8% (130/148)** and NLL 0.4217, outperforming the frozen Phase 4A StateQuery reference (31.1%, 46/148; NLL 1.6763). Balanced accuracy reached 73.3% with `contradicted` recall 12/23 (majority `entailed` baseline was 84.5%, 125/148).
  - **Cache Parity Confirmed**: Hybrid cached continuation agreed with full-prompt evaluation to within FP32 machine precision (max logit diff $\le 1.91 \times 10^{-5}$, probability diff $\le 3.31 \times 10^{-6}$, identical actions).
  - **Execution Latency Breakdown**: Amortized state root prefill took ~65 ms (ContractNLI) and ~346 ms (QASPER); primary suffix evaluation took ~165 ms and ~142 ms.
* **What Was Not Confirmed / Disproved**:
  - **Semantic-None Rejection Collapse**: The uncalibrated `Z` token logit completely failed to detect semantic none on ContractNLI. `Z` won on **0/124** semantic-none questions (0.0% recall, 0.0% false-none), reducing full decision accuracy to 47.8% (virtually identical to the 46.0% majority-class baseline).
  - **Positional Order Sensitivity**: Reversing candidate order in ContractNLI changed the winning candidate on 18 of 272 questions (6.6%) and shifted option probabilities by up to 0.2865. While reversed-order raw accuracy was 89.2%, balanced accuracy dropped to 67.0%.
  - **QASPER Answerability Deficit**: On QASPER (1 candidate option + `Z`), accuracy was 77.4% (41/53), falling short of the simple answerable-majority baseline (84.9%, 45/53); `Z` caught only 2/8 semantic-none cases (25.0% recall, 13.3% false-none).
  - **Model Promotion Rejected**: Option logits alone cannot replace the decision engine without calibrated semantic-none handling. No model, prompt, or threshold was promoted; final splits remained unopened (`final_opened: false`).
* **Supporting Directory & Key Artifacts**:
  - Local Directory: [`25_phase4e_b1_candidate_option_logit_results/candidate_option_logit_gate16_s17_v2/`](./25_phase4e_b1_candidate_option_logit_results/candidate_option_logit_gate16_s17_v2/)
  - Key files: [`NONFINAL_EVALUATION.json`](./25_phase4e_b1_candidate_option_logit_results/candidate_option_logit_gate16_s17_v2/NONFINAL_EVALUATION.json), [`NONFINAL_RESULT_LOCK.json`](./25_phase4e_b1_candidate_option_logit_results/candidate_option_logit_gate16_s17_v2/NONFINAL_RESULT_LOCK.json), [`EXPERIMENT_CONTRACT.json`](./25_phase4e_b1_candidate_option_logit_results/candidate_option_logit_gate16_s17_v2/EXPERIMENT_CONTRACT.json), `NONFINAL_ROWS.parquet`.

---

### 26. Phase 4E-B.2: Candidate Ranking & Semantic-None Detection Sweep
* **File**: [`26_phase4e_b2_candidate_ranking_sweep.ipynb`](./26_phase4e_b2_candidate_ranking_sweep.ipynb)
* **Run ID**: `candidate_detection_ranking_sweep_s17_n12_v1` (Workbench `4e.b.2`, pinned `Qwen/Qwen3.5-4B-Base` FP32)
* **Target HW**: GPU / CPU (FP32)
* **What it Measured**:
  - Factorial sweep of 3 candidate rankers (`joint_original`, `joint_order_average`, `separate_support`) and 3 semantic-none detectors (`joint_z`, `max_support`, `any_supported`).
  - One-variable logistic detector calibration fitted on `calibration_fit`.
  - Component selection across a 5-value threshold grid (`0.25`, `0.35`, `0.50`, `0.65`, `0.75`) on `policy_development`.
  - Non-final gate evaluation across 12 states per source per split (741 questions total; gate sample N=250: 204 ContractNLI, 46 QASPER; no state-ID overlap with 4E-B.1).
* **What Was Confirmed / Proved**:
  - **Order Bias Neutralization**: Averaging original and reversed order (`joint_order_average`) successfully neutralized order sensitivity, achieving **89.3%** conditional ranking accuracy on development and **87.0% (94/108)** on gate on ContractNLI (beating the majority position baseline of 77.8%).
  - **Detector AUROC Separability**: Evaluated detectors demonstrated strong intrinsic ROC curves: on ContractNLI, `joint_z` AUROC was **0.7412** (AP 0.7353); on QASPER, `any_supported` AUROC was **0.8293**, `max_support` **0.8098**, and `joint_z` **0.8000**.
  - **High ContractNLI None Recall**: At selected threshold 0.35 with calibrated `joint_z`, semantic-none recall reached **75.0% (72/96)** on the gate, yielding overall full decision accuracy of 61.8% (balanced accuracy 62.5%), outperforming the full majority baseline (41.2%).
  - **Rigorous Hybrid Parity**: All 6 cached versus full-prompt checks across all 3 prompt families strictly passed numerical tolerances ($\le 3.31 \times 10^{-6}$) with zero action changes.
* **What Was Not Confirmed / Disproved**:
  - **Severe False-None Rejection**: Despite 75% none recall on ContractNLI, the false-none rate reached **42.6% (46/108)** on the gate, more than double the project's $\le 0.20$ safety ceiling. Conditional ranking captured 84/84 entailed but only 10/24 contradicted; after rejection, full-decision contradicted recall was gutted to **3/24 (12.5%)**.
  - **Operational QASPER None Rejection Failure**: For QASPER, the development-selected arm (`max_support` at threshold 0.25) predicted **0/5 semantic-none** on the gate (0.0% recall, 0.0% false-none; full accuracy 89.1%, identical to majority baseline). All 46 calibrated gate none probabilities were $< 0.25$ (max 0.2348), rendering the pre-registered threshold grid incapable of rejecting a single case.
  - **Cross-Source Calibration Generalization Failure**: ContractNLI and QASPER selected completely discordant detectors (`joint_z` vs `max_support`) and operating thresholds (0.35 vs 0.25), proving that a unified calibrated decision head cannot transfer across diverse document structures without source-specific calibration.
  - **Post-Hoc Threshold Probes Disproved**: An unlocked, post-hoc lower-threshold test ($Z$ at 0.175) found 2/5 QASPER none cases but falsely rejected 9/41 answerable (22.0% false-none). Post-hoc diagnostics cannot substitute for pre-registered selection.
  - **Release Promotion Denied**: The sweep failed both the ContractNLI false-none guardrail and the QASPER rejection threshold. Final splits remain unopened (`final_opened: false`); benchmark source repair and independent adjudication remain the mandatory prerequisites.
* **Supporting Directory & Key Artifacts**:
  - Local Directory: [`26_phase4e_b2_candidate_ranking_sweep_results/candidate_detection_ranking_sweep_s17_n12_v1/`](./26_phase4e_b2_candidate_ranking_sweep_results/candidate_detection_ranking_sweep_s17_n12_v1/)
  - Key files: [`DEVELOPMENT_SELECTION.json`](./26_phase4e_b2_candidate_ranking_sweep_results/candidate_detection_ranking_sweep_s17_n12_v1/DEVELOPMENT_SELECTION.json), [`NONFINAL_EVALUATION.json`](./26_phase4e_b2_candidate_ranking_sweep_results/candidate_detection_ranking_sweep_s17_n12_v1/NONFINAL_EVALUATION.json), [`NONFINAL_RESULT_LOCK.json`](./26_phase4e_b2_candidate_ranking_sweep_results/candidate_detection_ranking_sweep_s17_n12_v1/NONFINAL_RESULT_LOCK.json), [`EXPERIMENT_CONTRACT.json`](./26_phase4e_b2_candidate_ranking_sweep_results/candidate_detection_ranking_sweep_s17_n12_v1/EXPERIMENT_CONTRACT.json), `NONFINAL_ROWS.parquet`.

---

## Key Scientific Insights & Architectural Invariants

1. **Strict FP32 Reference Boundary**:
   Shared-prefix execution and batched continuation require strict IEEE FP32 precision to guarantee deterministic mathematical parity with sequential execution ($\Delta p \le 1.1 \times 10^{-5}$). Lower precisions (BF16, TF32) introduce non-negligible numerical drift and flip policy decisions.
2. **Tripartite Hybrid Branch State Isolation**:
   `Qwen/Qwen3.5-4B-Base` uses a hybrid architecture (24 DeltaNet linear-attention layers + 8 full attention layers). KV-cache cloning alone does **not** isolate branches. Continuation state must isolate:
   - Full-attention KV tensors
   - DeltaNet recurrent state matrices
   - 1D depthwise convolution buffers
3. **State-First Segmented Tokenization**:
   Structuring input sequences as $[\text{State Prefix}] + [\text{Question Suffix}] + [\text{Candidate Suffix}]$ enables prefilling the state document once to create an immutable root prefix. Question and candidate evaluations branch cleanly from this state without redundant computation.
4. **Failure of Uncalibrated Low-Bit Cache Compression**:
   Low-bit quantization of continuation state (e.g. TurboQuant 2-4 bit) produces unacceptable probability error and flips decisions. Lossless FP32 and FP16-KV caching are the only configurations that passed all parity gates.
5. **Rejection Discipline & Semantic None**:
   Standard softmax classification and naive pooling overconfidentially assign probability to provided candidates when the true answer is absent. Robust decision inference requires explicit semantic-none modeling (`__none__`), calibrated rejection thresholds, and domain-specific verification.
6. **Pre-Registration & Anti-Leakage Governance**:
   Development winners are not replaced post-hoc after inspecting final evaluation splits. Test partitions must remain strictly locked until all pre-registered acceptance criteria are satisfied.
7. **Saturation of Scalar Rejection Loss Weights**:
   Scaling a single scalar loss multiplier for missing-answer options saturates at the dataset's empirical class imbalance ratio (e.g. 8.5602 in Phase 4B.2). Forcing higher scalar penalties creates destructive cross-source interference—raising none-recall on one dataset while driving catastrophic false-none rejection (39.66%) on another.
8. **Generalization Collapse in Pairwise Applicability**:
   Adding within-state pairwise ranking losses can satisfy development recall thresholds in isolation, but fails to transport to untouched calibration gates. Models easily memorize within-state ranking artifacts that do not translate into globally calibrated confidence thresholds across new documents.
9. **Inadequacy of Isolated Head and Shallow Residual Tuning**:
   When pooled backbone representations lack the necessary resolution to distinguish unanswerable questions from answerable ones, neither fine-tuning the classification MLP head in isolation nor attaching shallow handcrafted diagnostic residuals can resolve the boundary. Both interventions degrade proper scoring rules (NLL and Brier) while failing to recover required true-positive recall.
10. **Blinded Adjudication Preceding Representation Learning**:
    Before committing compute to upstream token-level representation retraining, systematic errors must be evaluated under double-blind protocols. If dataset annotations contain inherent ambiguity or unresolvable answerability boundaries, training larger models against noisy labels guarantees failure. Rigorous error audits provide the only sound prerequisite for new representation modeling.
11. **Answer-Bearing Representation Coverage vs Raw State Coverage**:
    Passing string-length and character-hash validation on state documents does not guarantee that the serialized text contains the answer-bearing representation. When tables, float values, figure contents, or bibliography citations are dropped or truncated during serialization, answerable questions become unanswerable in the representation space. Models cannot learn representations for data that was never serialized; benchmark serialization repair and data adjudication must precede model retraining.
12. **Independent Review Governance vs Same-Assistant Disposition**:
    Blinded human error audits require strict separation between independent review and subsequent assistant workflow maintenance. Same-assistant follow-ups (e.g. Phase 4E-A3 6-row review or 27-case worklist) must be explicitly recorded as non-independent and quarantined from model-release governance. Benchmark gold evidence and test splits must remain intact until formally reviewed, versioned repair overlays are authorized.
13. **Candidate Option-Logit Expressivity vs Rejection Blindness**:
    Direct next-token candidate letter logits on frozen `Qwen/Qwen3.5-4B-Base` exhibit strong discriminative ranking capability on answerable candidate sets (e.g. 87.8% on ContractNLI), significantly outperforming unadapted pooling heads. However, next-token generation mechanics over-allocate probability mass to explicit candidate letters, rendering raw uncalibrated rejection tokens (like `Z`) completely blind to unanswerability (0.0% semantic-none recall on ContractNLI).
14. **Positional Bias in Next-Token Option Letter Ranking**:
    When candidate options are rendered sequentially in a prompt prefix, autoregressive attention introduces measurable positional bias (winner changes in 6.6% of cases under reversed ordering, with probability shifts up to 0.2865). Order averaging (`joint_order_average`) across canonical and reversed permutations successfully neutralizes presentation artifacts, yielding robust conditional ranking accuracy.
15. **Source Calibration Asymmetry and Grid Floor Dynamics**:
    Calibration curves and optimal rejection architectures differ fundamentally across document genres. While dense-none legal datasets (ContractNLI) achieve high none-recall with calibrated `Z` detectors at moderate thresholds (0.35) at the cost of high false-none penalties, sparse-none scientific QA (QASPER) places all calibrated probabilities below standard threshold floors ($<0.25$). A single universal rejection threshold across disparate document types collapses either into catastrophic over-rejection or total inaction.
