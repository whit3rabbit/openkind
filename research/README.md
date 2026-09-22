# OpenKind Research & Empirical Notebooks

This directory contains the experimental notebooks, frozen reference artifacts, benchmark datasets, and model selection studies that form the scientific foundation of **OpenKind**.

OpenKind evaluates typed decisions (`Choice`, `Score`, `Noul`) directly from neural representations without an autoregressive token-generation loop. The research journey documented here progresses from initial feasibility probes on `Qwen/Qwen3.5-4B-Base`, through precision contracts, prefix caching, cache compression, multi-domain criteria transfer, and multi-model screening, to the native Rust backbone handoff and multi-question natural-document architectures.

Detailed analysis, theoretical foundations, and mathematical formulations are documented in the [OpenKind Whitepaper v0.8.2](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md) and [Research Dossier](../docs/RESEARCH.md).

---

## Chronological Experiment Index

| # | Notebook | Phase / Run ID | Target HW | Core Research Focus | Status & Outcome | Supporting Artifacts |
|---|---|---|---|---|---|---|
| **1** | [1 - OpenKind_Phase2_Qwen3_5_4B_Probe.ipynb](./1%20-%20OpenKind_Phase2_Qwen3_5_4B_Probe.ipynb) | Phase 2 Probe | GPU (L4/A100) | Frozen Qwen3.5 feature extraction & linear decision head | Feasibility proven; identified text vs vision parameter count | [Whitepaper §3.1](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#31-research-chronology) |
| **2** | [2 - OpenKind_Phase2B_Qwen3_5_4B_Benchmark.ipynb](./2%20-%20OpenKind_Phase2B_Qwen3_5_4B_Benchmark.ipynb) | Phase 2B (`20260917T205849Z`) | NVIDIA L4 (BF16) | MultiNLI classification, pooling strategies, calibration, generation vs decision timing | Last-token + linear won (87.7% matched); 44.6× speedup over generation; temperature scaling rejected | [openkind_phase2b_20260917T205849Z/](./openkind_phase2b_20260917T205849Z) |
| **3** | [3 - OpenKind_Phase2C_Stability_DynamicChoice_LoRA.ipynb](./3%20-%20OpenKind_Phase2C_Stability_DynamicChoice_LoRA.ipynb) | Phase 2C (`20260917T222948Z`) | NVIDIA L4 (BF16) | Multi-seed stability, dynamic candidate scoring (Banking77), global `__none__` logit | Stable across seeds (87.0% / 88.8%); 80.8% seen / 64.6% unseen accuracy; 0.817 AUROC for none | [openkind_phase2c_20260917T222948Z/](./openkind_phase2c_20260917T222948Z) |
| **4** | [4 - OpenKind_Phase2D_Numerics_NoneHandling_RequestBench.ipynb](./4%20-%20OpenKind_Phase2D_Numerics_NoneHandling_RequestBench.ipynb) | Phase 2D (`20260917T234417Z`) | NVIDIA L4 | Precision diagnostics (BF16 vs FP32 vs TF32), none-head alternatives, request latency vs K | Isolated layer divergence in DeltaNet/conv; linear latency scaling without prefix caching | [openkind_phase2d_20260917T234417Z/](./openkind_phase2d_20260917T234417Z) |
| **5** | [5 - OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb](./5%20-%20OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb) | Phase 2E initial (`20260918T032049180933Z`, v2e.1.1) | GPU (L4/A100) | Module-level FP32 promotion, shared-prefix KV branching, application policies | Full FP32 stage halted by memory guard; isolated hybrid state isolation requirement | [openkind_phase2e_20260918T032049180933Z/](./openkind_phase2e_20260918T032049180933Z) |
| **6** | [6 - OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb](./6%20-%20OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb) | Phase 2E expanded (`20260918T114914072764Z`, v2e.2.0) | GPU (isolated processes) | Strict FP32 reference, batched prefix reuse parity vs BF16, component breakdown | Strict FP32 prefix reuse passed ($Δp \le 1.1 \times 10^{-5}$); BF16 failed tolerance ($>0.005$) and flipped actions | [openkind_phase2e_expanded_20260918T114914072764Z/](./openkind_phase2e_expanded_20260918T114914072764Z) |
| **7** | [7 - OpenKind_Phase2F_CacheCompression_TurboQuant_PrefixReuse.ipynb](./7%20-%20OpenKind_Phase2F_CacheCompression_TurboQuant_PrefixReuse.ipynb) | Phase 2F (`20260918T224427722898Z`, v2f.1.0) | NVIDIA L4 / A100 | TurboQuant low-bit KV compression (2/3/4-bit) vs FP16/lossless, GPU LRU prefix cache | Lossless & FP16-KV passed all gates; all 4 low-bit TurboQuant variants failed and flipped actions | [openkind_phase2f_20260918T224427722898Z/](./openkind_phase2f_20260918T224427722898Z) |
| **8** | [8 - OpenKind_Phase2G_FreshEvidence_TF32_CacheLifecycle.ipynb](./8%20-%20OpenKind_Phase2G_FreshEvidence_TF32_CacheLifecycle.ipynb) | Phase 2G (`20260919T005142584348Z`, v2g.1.0) | NVIDIA L4 / A100 | Generalization to fresh Banking/CLINC/OOS data, TF32 execution, cache TTL & eviction | TF32 cut latency ~2× but flipped 3 argmax decisions; revealed candidate head narrowness on OOS | [openkind_phase2g_20260919T005142584348Z/](./openkind_phase2g_20260919T005142584348Z) |
| **9** | [9 - OpenKind_Phase2H_Criteria_Rejection_Multidomain.ipynb](./9%20-%20OpenKind_Phase2H_Criteria_Rejection_Multidomain.ipynb) | Phase 2H (`20260919T040612625670Z`, v2h.1.2) | NVIDIA L4 / A100 | Multi-domain candidate fitting, criteria augmentation (support examples), rejection transfer | Original criteria transferred better (83.8% acc) than support examples (78.9% acc); held to pre-registration | [openkind_phase2h_20260919T040612625670Z/](./openkind_phase2h_20260919T040612625670Z) |
| **10** | [10 - OpenKind_Phase2IJ_Gated_Workbench.ipynb](./10%20-%20OpenKind_Phase2IJ_Gated_Workbench.ipynb) | Phase 2I/2J Workbench (`2ij.1.0`) | NVIDIA L4 / A100 | Multi-question reviewed study gate, state-first vs instruction-first GPU mechanics probe | Strict independent-review gate blocked unverified training; probe validated state-first prefill mechanics | [openkind_2ij_2ij_reviewed_multiquestion_v1/](./openkind_2ij_2ij_reviewed_multiquestion_v1) |
| **11** | [11 - OpenKind_Phase2IJ_Gated_Workbench (update & model selection).ipynb](./11%20-%20OpenKind_Phase2IJ_Gated_Workbench%20(update%20%26%20model%20selection).ipynb) | Phase 2I/2J Screen (`2ij.2.0`) | NVIDIA L4 / A100 | 13 fit jobs, 31 evaluation profiles across Qwen 3.5 4B, Qwen 2.5 1.5B/2B, ModernBERT | Selected & locked profile `a047d6802c3f06f085b8` (Qwen3.5-4B state-first score-summary head) | [openkind_2ij_2ij_model_selection_screen_v2/](./openkind_2ij_2ij_model_selection_screen_v2) |
| **12** | [12 - OpenKind_Phase2IJ_A100.ipynb](./12%20-%20OpenKind_Phase2IJ_A100.ipynb) | Phase 2I/2J Continuation | NVIDIA A100 | Clean selected-model bundle export and checksum verification on high-memory GPU | Produced golden bundle `4d9ffdee...3332` with zero OOM risk; validated reference repo | Informs profile `a047d6802c3f06f085b8` |
| **13** | [13 - OpenKind_Phase3A_BranchableState_BatchedQ.ipynb](./13%20-%20OpenKind_Phase3A_BranchableState_BatchedQ.ipynb) | Phase 3A (`20260920T024056Z`) | NVIDIA L4 / A100 | Hybrid state branching (KV + DeltaNet + conv), batched-Q/K topologies, crossover behavior | Zero-leakage branch isolation confirmed; batched-Q faster for long state/high Q; informed Rust scheduler | [Whitepaper §15](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#15-phase-3a-branchable-hybrid-state-and-batched-q-execution) |
| **14** | [14 - OpenKind_Phase3B_Qwen35_Backbone_Parity.ipynb](./14%20-%20OpenKind_Phase3B_Qwen35_Backbone_Parity.ipynb) | Phase 3B (`20260920T152206Z`) | GPU (FP32) | Layer-by-layer backbone traces, golden token fixtures, Rust handoff contract | Exported 47 FP32 vectors across 34 stages; maximum cached continuation drift $\le 1.91 \times 10^{-5}$ | [OpenKind_Phase3B_BackboneParity_20260920T152206Z/](./OpenKind_Phase3B_BackboneParity_20260920T152206Z) |
| **15** | [15 - OpenKind_Phase4A0_4A1_StateQuery_Workbench.ipynb](./15%20-%20OpenKind_Phase4A0_4A1_StateQuery_Workbench.ipynb) | Phase 4A.0 / 4A.1 (`20260921T013558Z`, v4a.0.2) | GPU (L4/A100) | Natural-document corpus lock (ContractNLI + QASPER) & StateQuery B1 representation probe | Locked 2,192 states / 15,368 Qs; B1 scored 0.7796 macro-acc but collapsed QASPER none-recall to 0.0 | [Whitepaper §18](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#18-phase-4a-natural-document-multi-question-benchmark-and-statequery-readout) |
| **16** | [16 - OpenKind_Phase4A2_StateQuery_Model_Comparison.ipynb](./16%20-%20OpenKind_Phase4A2_StateQuery_Model_Comparison.ipynb) | Phase 4A.2 (`4a.2.0`) | GPU (L4/A100) | Matched B0 control, source-balanced B1, factorized B2 without reopening final splits | B0 matched collapse (0.0 none-recall); balanced B1 raised none-recall to 0.168 (gate threshold: 0.30) | [Whitepaper §18.4](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#184-completed-non-final-comparison-aggregate-improvement-hides-a-qasper-collapse) |
| **17** | [17 - OpenKind_Phase4B2_StateQuery_Model_Comparison.ipynb](./17%20-%20OpenKind_Phase4B2_StateQuery_Model_Comparison.ipynb) | Phase 4B.2 (`4b.2.0` / `4b.2.1`) | GPU (L4/A100) | Scalar-weight saturation (cap 12) & source/class-stratified applicability | Realized weight saturated at 8.5602; stratified objective raised QASPER recall to 0.3111 gate, but gate policy cost failed (0.11675) | [OpenKind_Phase4B2_StateQuery_Model_Comparison_results/](./OpenKind_Phase4B2_StateQuery_Model_Comparison_results) |
| **18** | [18 - OpenKind_Phase4B3_Applicability_Ranking_Sweep.ipynb](./18%20-%20OpenKind_Phase4B3_Applicability_Ranking_Sweep.ipynb) | Phase 4B.3 (`4b.3.0`) | GPU (L4/A100) | Within-state pairwise applicability ranking loss ($w \in \{0.25, 0.50, 1.00\}$) | Cleared development recall (0.3451), but failed gate false-none (>0.21) and policy cost (0.10221); rejected | [Whitepaper §18.10](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#1810-phase-4b3-pairwise-applicability-passes-development-but-not-transfer) |
| **19** | [19 - OpenKind_Phase4C_Decoupled_Applicability_Head.ipynb](./19%20-%20OpenKind_Phase4C_Decoupled_Applicability_Head.ipynb) | Phase 4C (`4c.0.0`) | GPU (L4/A100) | Isolated applicability head fine-tuning over frozen B2 representations | Child epochs regressed NLL (+2.77%) and Brier (+2.10%) for only +2 true positives; parent epoch 0 retained | [OpenKind_Phase4C_Decoupled_Applicability_results/](./OpenKind_Phase4C_Decoupled_Applicability_results) |
| **20** | [20 - OpenKind_Phase4D_EvidenceAware_Applicability.ipynb](./20%20-%20OpenKind_Phase4D_EvidenceAware_Applicability.ipynb) | Phase 4D (`4d.0.0`) | GPU (L4/A100) | 19-dim handcrafted evidence/uncertainty diagnostic residual over frozen parent | Strict JSON enforced; child gained only +1 true positive while NLL worsened +6.08%; child rejected | [OpenKind_Phase4D_EvidenceAware_Applicability_results/](./OpenKind_Phase4D_EvidenceAware_Applicability_results) |
| **21** | [21 - OpenKind_Phase4E_QASPER_Error_Audit.ipynb](./21%20-%20OpenKind_Phase4E_QASPER_Error_Audit.ipynb) | Phase 4E (`4e.0.0`) | CPU | Blinded human audit of 150 QASPER false-negative, false-positive, and control cases | Prepares double-blind adjudication pack to separate representation failure from label ambiguity before 4E-B | [OpenKind_Phase4E_QASPER_Audit_results/](./OpenKind_Phase4E_QASPER_Audit_results) |

---

## Detailed Notebook Breakdown

### 1. Phase 2 Initial Feasibility Probe
* **File**: [`1 - OpenKind_Phase2_Qwen3_5_4B_Probe.ipynb`](./1%20-%20OpenKind_Phase2_Qwen3_5_4B_Probe.ipynb)
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
* **File**: [`2 - OpenKind_Phase2B_Qwen3_5_4B_Benchmark.ipynb`](./2%20-%20OpenKind_Phase2B_Qwen3_5_4B_Benchmark.ipynb)
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
  - [`openkind_phase2b_20260917T205849Z/`](./openkind_phase2b_20260917T205849Z)
  - Key files: [`openkind_phase2b_summary.md`](./openkind_phase2b_20260917T205849Z/openkind_phase2b_summary.md), `decision_benchmark.json`, `generation_benchmark.json`, `metrics.json`.

---

### 3. Phase 2C: Stability, Dynamic Choices, and Serving Gates
* **File**: [`3 - OpenKind_Phase2C_Stability_DynamicChoice_LoRA.ipynb`](./3%20-%20OpenKind_Phase2C_Stability_DynamicChoice_LoRA.ipynb)
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
  - [`openkind_phase2c_20260917T222948Z/`](./openkind_phase2c_20260917T222948Z)
  - Key files: [`openkind_phase2c_summary.md`](./openkind_phase2c_summary.md), `stability.json`, `dynamic_predictions_test_seen.json`, `dynamic_predictions_test_unseen.json`.

---

### 4. Phase 2D: Numerical Reference, None-Handling, and Request Benchmarks
* **File**: [`4 - OpenKind_Phase2D_Numerics_NoneHandling_RequestBench.ipynb`](./4%20-%20OpenKind_Phase2D_Numerics_NoneHandling_RequestBench.ipynb)
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
  - [`openkind_phase2d_20260917T234417Z/`](./openkind_phase2d_20260917T234417Z)
  - Key files: [`openkind_phase2d_summary.md`](./openkind_phase2d_summary.md), `layer_divergence.json`, `none_head_comparison.json`, `request_benchmarks.json`.

---

### 5. Phase 2E (Initial): Selective Precision & Shared-Prefix Policy
* **File**: [`5 - OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb`](./5%20-%20OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb)
* **Run ID**: `20260918T032049180933Z` (Version `2e.1.1`)
* **What it Measured**:
  - Feasibility of module-level selective precision (promoting specific linear-attention or normalization modules to FP32 while leaving others in BF16).
  - Implementation of state-prefill KV caching where shared message context is prefilled once and candidate suffixes are evaluated by cloning cache state.
  - Operational acceptance/abstention policies based on minimum winning margin and entropy thresholds.
* **Results**:
  - An in-process memory guard halted the full FP32 stage due to running the notebook coordinator and GPU tensors in the same process space.
  - Demonstrated that cloning Qwen attention KV tensors alone is insufficient for state branching: hybrid recurrent DeltaNet states must also be isolated.
* **Supporting Directory**:
  - [`openkind_phase2e_20260918T032049180933Z/`](./openkind_phase2e_20260918T032049180933Z)
  - Key files: [`README_results.md`](./openkind_phase2e_20260918T032049180933Z/README_results.md).

---

### 6. Phase 2E (Expanded): Isolated FP32 Reference & Batched Prefix Parity
* **File**: [`6 - OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb`](./6%20-%20OpenKind_Phase2E_SelectivePrecision_SharedPrefix_Policy.ipynb)
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
  - [`openkind_phase2e_expanded_20260918T114914072764Z/`](./openkind_phase2e_expanded_20260918T114914072764Z)
  - Key files: [`README_results.md`](./openkind_phase2e_expanded_20260918T114914072764Z/README_results.md), `fp32_strict_math/parity_rows.json`, `bf16_default/parity_rows.json`, `component_profiles.json`.

---

### 7. Phase 2F: Cache Compression, TurboQuant Codecs, and Serving Trade-Offs
* **File**: [`7 - OpenKind_Phase2F_CacheCompression_TurboQuant_PrefixReuse.ipynb`](./7%20-%20OpenKind_Phase2F_CacheCompression_TurboQuant_PrefixReuse.ipynb)
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
  - [`openkind_phase2f_20260918T224427722898Z/`](./openkind_phase2f_20260918T224427722898Z)
  - Key files: [`README_results.md`](./openkind_phase2f_20260918T224427722898Z/README_results.md), `codec_regression.json`, `lru_trace_results.json`, `storage_benchmarks.json`.

---

### 8. Phase 2G: Fresh Evidence, TF32 Arithmetic, and Cache Lifecycle
* **File**: [`8 - OpenKind_Phase2G_FreshEvidence_TF32_CacheLifecycle.ipynb`](./8%20-%20OpenKind_Phase2G_FreshEvidence_TF32_CacheLifecycle.ipynb)
* **Run ID**: `20260919T005142584348Z` (Version `2g.1.0`)
* **What it Measured**:
  - Generalization to fresh labeled messages: Banking77 fresh partition (416 episodes, 112 messages), CLINC150 non-financial domain transfer, and author-written out-of-scope (OOS) queries.
  - TF32-permitted FP32 execution on Ampere/Ada Tensor Cores versus strict FP32 IEEE math.
  - Persistent cache lifecycle: TTL expirations, eviction policies, and cache size bounds.
* **Results**:
  - **Rejection Limitation Identified**: On the CLINC panel, answerable accuracy was 93.75%, but omitted-intent recall was only 39.06% (author OOS recall was 46.88%). Revealed that candidate-conditioned scoring trained on in-domain tasks does not automatically generalize to out-of-domain rejection.
  - **TF32 Trade-off**: TF32 halved latency ($K=4$: 265 ms $\to$ 132 ms; $K=16$: 1,118 ms $\to$ 518 ms), but caused 3 argmax flips across 416 episodes. TF32 is valuable for high-throughput serving but cannot be used as the reference parity authority.
* **Supporting Directory**:
  - [`openkind_phase2g_20260919T005142584348Z/`](./openkind_phase2g_20260919T005142584348Z)
  - Key files: [`README_results.md`](./openkind_phase2g_20260919T005142584348Z/README_results.md), `fresh_quality_summary.json`, `tf32_comparison.json`, `cache_lifecycle_trace.json`.

---

### 9. Phase 2H: Criteria Augmentation, Rejection, and Multi-Domain Transfer
* **File**: [`9 - OpenKind_Phase2H_Criteria_Rejection_Multidomain.ipynb`](./9%20-%20OpenKind_Phase2H_Criteria_Rejection_Multidomain.ipynb)
* **Run ID**: `20260919T040612625670Z` (Version `2h.1.2`, continuation)
* **What it Measured**:
  - Multi-domain candidate head fitting and criteria transfer across Banking77 and CLINC150.
  - Comparing original criteria descriptions vs criteria augmented with few-shot training support examples.
  - Rigorous pre-registration discipline: development selection rule vs held-out final evaluation.
* **Results**:
  - The support-example joint head achieved 78.91% raw pooled accuracy (1,152 episodes), 85.55% on Banking fitting labels, and 89.84% on CLINC fitting domains, but dropped to 63.67% on held-out Banking labels and 66.02% on held-out CLINC domains.
  - **Methodological Victory**: The original-criteria control head transferred significantly better on held-out data (83.77% accuracy / 0.541 NLL) than the development-selected support-augmented head. Following pre-registration rules, the project did *not* retroactively swap winners, using this as motivation for architectural model selection in Phase 2I/2J.
* **Supporting Directory**:
  - [`openkind_phase2h_20260919T040612625670Z/`](./openkind_phase2h_20260919T040612625670Z)
  - Key files: [`README_results.md`](./openkind_phase2h_20260919T040612625670Z/README_results.md), `final_quality_report.json`, `comparison_arms.json`.

---

### 10. Phase 2I/2J: Gated Multi-Question Workbench
* **File**: [`10 - OpenKind_Phase2IJ_Gated_Workbench.ipynb`](./10%20-%20OpenKind_Phase2IJ_Gated_Workbench.ipynb)
* **Version**: `2ij.1.0` (`openkind_2ij_2ij_reviewed_multiquestion_v1`)
* **What it Measured**:
  - Setup of a formal review gate requiring independent human audit of task criteria and split manifests before launching expensive training.
  - Small synthetic GPU probe validating **state-first segmented tokenization**:
    $$\text{Tokens} = [\text{State Prefix}] + [\text{Question Suffix}] + [\text{Candidate Suffix}]$$
    contrasted with standard instruction-first rendering.
* **Results**:
  - The review gate successfully blocked unreviewed multi-question training, preventing unverified claims.
  - Synthetic GPU probe validated state-first mechanics: prefilling the state document once and branching questions/candidates reduced compute without token leakage.
* **Supporting Directory**:
  - [`openkind_2ij_2ij_reviewed_multiquestion_v1/`](./openkind_2ij_2ij_reviewed_multiquestion_v1)
  - Key files: [`REPORT.md`](./openkind_2ij_2ij_reviewed_multiquestion_v1/REPORT.md), `gate_report.json`, `contracts/SERVICE_HANDOFF.md`.

---

### 11. Phase 2I/2J: Model Selection Screen & Architectural Decision
* **File**: [`11 - OpenKind_Phase2IJ_Gated_Workbench (update & model selection).ipynb`](./11%20-%20OpenKind_Phase2IJ_Gated_Workbench%20(update%20%26%20model%20selection).ipynb)
* **Version**: `2ij.2.0` (`openkind_2ij_2ij_model_selection_screen_v2`)
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
  - [`openkind_2ij_2ij_model_selection_screen_v2/`](./openkind_2ij_2ij_model_selection_screen_v2)
  - Key files: [`REPORT.md`](./openkind_2ij_2ij_model_selection_screen_v2/REPORT.md), [`MODEL_DECISION.md`](./openkind_2ij_2ij_model_selection_screen_v2/MODEL_DECISION.md), `contracts/SERVICE_HANDOFF.md`.

---

### 12. Phase 2I/2J: A100 Selected-Model Bundle Export
* **File**: [`12 - OpenKind_Phase2IJ_A100.ipynb`](./12%20-%20OpenKind_Phase2IJ_A100.ipynb)
* **Goal**: High-memory export continuation on an NVIDIA A100 GPU to export the full model bundle for profile `a047d6802c3f06f085b8` without risk of VRAM fragmentation or host memory limits.
* **Outputs**:
  - Exported complete self-contained reference bundle:
    - SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`.
    - Published public reference repository: `cowWhySo/OpenKind-Qwen3.5-4B-StateFirst`.
  - Exported token fixtures, configuration JSONs, and model weights.

---

### 13. Phase 3A: Branchable Hybrid State & Batched-Q Execution
* **File**: [`13 - OpenKind_Phase3A_BranchableState_BatchedQ.ipynb`](./13%20-%20OpenKind_Phase3A_BranchableState_BatchedQ.ipynb)
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
  - [OpenKind Whitepaper §15](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#15-phase-3a-branchable-hybrid-state-and-batched-q-execution).

---

### 14. Phase 3B: Qwen3.5 Backbone Parity Reference for Rust
* **File**: [`14 - OpenKind_Phase3B_Qwen35_Backbone_Parity.ipynb`](./14%20-%20OpenKind_Phase3B_Qwen35_Backbone_Parity.ipynb)
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
  - [`OpenKind_Phase3B_BackboneParity_20260920T152206Z/`](./OpenKind_Phase3B_BackboneParity_20260920T152206Z)
  - Key files: [`RUST_BACKBONE_HANDOFF.md`](./OpenKind_Phase3B_BackboneParity_20260920T152206Z/RUST_BACKBONE_HANDOFF.md), `TOKEN_FIXTURES.json`, `PROBABILITY_REFERENCE.json`, `CONTINUATION_TRACE.json`, `QWEN35_BACKBONE_GOLDEN.safetensors`.

---

### 15. Phase 4A.0 / 4A.1: Natural-Document Corpus & StateQuery Workbench
* **File**: [`15 - OpenKind_Phase4A0_4A1_StateQuery_Workbench.ipynb`](./15%20-%20OpenKind_Phase4A0_4A1_StateQuery_Workbench.ipynb)
* **Run ID**: `20260921T013558Z` (Version `4a.0.2`)
* **What it Measured**:
  - **Phase 4A.0**: Assembled and locked a natural multi-question document benchmark combining **ContractNLI** (legal non-disclosure contracts) and **QASPER** (NLP research papers): 2,192 states, 15,368 questions, 25,687 options, 21,894 evidence rows. Partitions split strictly by document group (zero test leakage).
  - **Phase 4A.1**: Investigated whether a small learned query network (**StateQuery B1**) can evaluate questions directly over a **frozen, state-only Qwen representation** without re-encoding candidate descriptions.
* **Results**:
  - Corpus locked with SHA-256 table manifests before evaluation.
  - StateQuery B1 improved source-macro accuracy (0.7796 vs 0.3766 for historical candidate reference) and reduced NLL (0.5427 vs 1.6734).
  - **Critical Failure Mode Discovered**: StateQuery B1 collapsed on QASPER semantic-none recall to **0.0** (defaulted to predicting answerable options). Proved that aggregate macro accuracy can obscure catastrophic rejection collapse. Model promotion was halted.
* **Supporting Documentation**:
  - [OpenKind Whitepaper §18](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#18-phase-4a-natural-document-multi-question-benchmark-and-statequery-readout).

---

### 16. Phase 4A.2: StateQuery Model Comparison & Balance Optimization
* **File**: [`16 - OpenKind_Phase4A2_StateQuery_Model_Comparison.ipynb`](./16%20-%20OpenKind_Phase4A2_StateQuery_Model_Comparison.ipynb)
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
  - [OpenKind Whitepaper §18.4](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#184-completed-non-final-comparison-aggregate-improvement-hides-a-qasper-collapse).

---

### 17. Phase 4B.2: StateQuery Model Comparison & Scalar-Weight Saturation
* **File**: [`17 - OpenKind_Phase4B2_StateQuery_Model_Comparison.ipynb`](./17%20-%20OpenKind_Phase4B2_StateQuery_Model_Comparison.ipynb)
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
  - [`OpenKind_Phase4B2_StateQuery_Model_Comparison_results/`](./OpenKind_Phase4B2_StateQuery_Model_Comparison_results/)
  - Key files: [`b2q_weight12_s17/TRAINING_REPORT.json`](./OpenKind_Phase4B2_StateQuery_Model_Comparison_results/20260921T013558Z/b2q_weight12_s17/TRAINING_REPORT.json), [`b2_applicability_stratified_s17/NONFINAL_RESULT_LOCK.json`](./OpenKind_Phase4B2_StateQuery_Model_Comparison_results/20260921T013558Z/b2_applicability_stratified_s17/NONFINAL_RESULT_LOCK.json), `comparison_snapshot_20260922T115817Z.json`.

---

### 18. Phase 4B.3: Applicability Ranking & Pairwise Loss Sweep
* **File**: [`18 - OpenKind_Phase4B3_Applicability_Ranking_Sweep.ipynb`](./18%20-%20OpenKind_Phase4B3_Applicability_Ranking_Sweep.ipynb)
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
  - [OpenKind Whitepaper §18.10](../docs/whitepaper/OpenKind_Whitepaper_v0.8.2.md#1810-phase-4b3-pairwise-applicability-passes-development-but-not-transfer).

---

### 19. Phase 4C: Decoupled Applicability Head Recovery
* **File**: [`19 - OpenKind_Phase4C_Decoupled_Applicability_Head.ipynb`](./19%20-%20OpenKind_Phase4C_Decoupled_Applicability_Head.ipynb)
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
  - [`OpenKind_Phase4C_Decoupled_Applicability_results/`](./OpenKind_Phase4C_Decoupled_Applicability_results/)
  - Key files: [`b2_app_headonly_guard18_s17/TRAINING_REPORT.json`](./OpenKind_Phase4C_Decoupled_Applicability_results/20260921T013558Z/b2_app_headonly_guard18_s17/TRAINING_REPORT.json), [`b2_app_headonly_guard18_s17/NONFINAL_RESULT_LOCK.json`](./OpenKind_Phase4C_Decoupled_Applicability_results/20260921T013558Z/b2_app_headonly_guard18_s17/NONFINAL_RESULT_LOCK.json), `comparison_snapshot_20260922T172543Z.json`.

---

### 20. Phase 4D: Evidence-Aware Applicability Residual
* **File**: [`20 - OpenKind_Phase4D_EvidenceAware_Applicability.ipynb`](./20%20-%20OpenKind_Phase4D_EvidenceAware_Applicability.ipynb)
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
  - [`OpenKind_Phase4D_EvidenceAware_Applicability_results/`](./OpenKind_Phase4D_EvidenceAware_Applicability_results/)
  - Key files: [`b2_evidence_residual_s17/TRAINING_REPORT.json`](./OpenKind_Phase4D_EvidenceAware_Applicability_results/20260921T013558Z/b2_evidence_residual_s17/TRAINING_REPORT.json), [`b2_evidence_residual_s17/NONFINAL_RESULT_LOCK.json`](./OpenKind_Phase4D_EvidenceAware_Applicability_results/20260921T013558Z/b2_evidence_residual_s17/NONFINAL_RESULT_LOCK.json), `comparison_snapshot_20260922T200741Z.json`.

---

### 21. Phase 4E: QASPER Error & Evidence Audit
* **File**: [`21 - OpenKind_Phase4E_QASPER_Error_Audit.ipynb`](./21%20-%20OpenKind_Phase4E_QASPER_Error_Audit.ipynb)
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
  - [`OpenKind_Phase4E_QASPER_Audit_results/`](./OpenKind_Phase4E_QASPER_Audit_results/)
  - Key files: [`AUDIT_GUIDE.md`](./OpenKind_Phase4E_QASPER_Audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_GUIDE.md), [`AUDIT_REVIEW_BLINDED.csv`](./OpenKind_Phase4E_QASPER_Audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_REVIEW_BLINDED.csv), [`AUDIT_ANALYSIS.json`](./OpenKind_Phase4E_QASPER_Audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_ANALYSIS.json), [`AUDIT_RESULT_LOCK.json`](./OpenKind_Phase4E_QASPER_Audit_results/20260921T013558Z/qasper_error_audit_s17/AUDIT_RESULT_LOCK.json).

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
