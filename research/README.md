# OpenKind Research & Empirical Notebooks

This directory contains the experimental notebooks, frozen reference artifacts, benchmark datasets, and model selection studies that form the scientific foundation of **OpenKind**.

OpenKind evaluates typed decisions (`Choice`, `Score`, `Noul`) directly from neural representations without an autoregressive token-generation loop. The research journey documented here progresses from initial feasibility probes on `Qwen/Qwen3.5-4B-Base`, through precision contracts, prefix caching, cache compression, multi-domain criteria transfer, and multi-model screening, to the native Rust backbone handoff and multi-question natural-document architectures.

Detailed analysis, theoretical foundations, and mathematical formulations are documented in the [OpenKind Whitepaper](../docs/whitepaper/WHITEPAPER.md) and [Research Dossier](../docs/RESEARCH.md).

> [!NOTE]
> **Repository Size & Evidence Archival Policy**:
> Heavy raw evidence directories, checkpoint weights, and evaluation result folders have been pruned from git tracking to prevent repository bloat (freeing ~1.5 GB). All empirical findings remain 100% reproducible directly from the standalone numbered Jupyter notebooks (`01_*.ipynb` through `39_*.ipynb`). Complete raw run bundles, telemetry logs, receipts, and artifact tarballs are archived and documented on Google Drive.

## New training recipe

[35_local_decision_training.ipynb](./35_local_decision_training.ipynb) is a
self-contained Colab trainer for a mixed-task Qwen3.5-4B decision LoRA, targeting
A100 with an L4 NF4 path. It combines public labeled tasks, precomputed Qwen-teacher
decisions and exact rule examples, with separate development, calibration, gate
and reserved evaluation groups. Version 4 uses plain CE, atomic counterfactual
sampling, and 400 updates at 2,048 tokens. Reasoning data, occurrence-based schema
augmentation and the six-arm loss sweep are separate opt-in comparisons. E43/E44
motivate family, class and whole-request diagnostics with group uncertainty.
Frozen evaluation verifies and loads the exported implementation and artifacts.
See the [dataset rationale and run guide](./local_decision_training/README.md)
and [evidence-backed decisions](./local_decision_training/DECISIONS.md).
An optional [HelpSteer2 data ablation](./local_decision_training/README.md#strands-decider-training-lessons)
adds answer-adequacy supervision from human ratings, with request-group isolation.
Status: authored and locally checked with tiny models; full 4B CUDA training and
Mac qualification are unrun. It does not reopen historical final splits.

## CLEF and related decision-model research

[Cloudflare's CLEF announcement](https://blog.cloudflare.com/clef-decision-models/)
(1 October 2026) describes Qwen post-training with a learned schema head,
rank-256 LoRA, supervised probability losses, and calibration-oriented RL.

The [source review and whitepaper crosswalk](../docs/RESEARCH.md#cloudflare-clef-and-linked-decision-models-reviewed-2026-10-01)
cover the released CLEF/CLEF-flash code, Jev, DiffusionGemma, Kev, Laya,
the linked evaluation protocols, and the Hume and Raschka articles already
used by our whitepaper.

CLEF shares our finite-decision goal, but its fields attend to one another.
It is a separate model hypothesis from OpenKind's isolated question branches.
The review distinguishes disclosed training from verified inference code and
author-reported scores. No CLEF inference, training, or native qualification
was performed.

## Strands Decider training research

[Strands Decider's announcement](https://strandsagents.com/blog/introducing-strands-decider/)
and [Hobson v19 release](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
provide training scripts, synthetic rows, corpus hashes, adapter/head artifacts
and retention experiments for a Qwen3.5-2B-Base option-pointer model.

The [source review](../docs/RESEARCH.md#strands-decider-2b-hobson-v19-release-and-agent-interventions-reviewed-2026-10-01)
separates released v19 from later v20 experiments and checks how each aligns with
our 4B design and local failures. Experiment 35 adds an opt-in upstream HelpSteer2
proxy, preserves the baseline loss sweep, and records a
[tokenizer/data audit](./local_decision_training/STRANDS_SOURCE_AUDIT.json).
It imports no Strands benchmark labels or full training mixture. No Strands
neural inference or 4B training gain is established by this review.

## Orthrus adoption decision

**Decision:** Do not adopt Orthrus in the native decision engine.

The [Orthrus paper](https://arxiv.org/html/2605.12825v2) describes trained
diffusion modules that propose future token blocks and verify them through the
frozen autoregressive model, with both paths reusing the historical KV cache.
OpenKind's candidates are already known. Its native Qwen3.5 path processes their
suffixes through a score-summary readout, without an output-token generation loop.

Source review on 29 September 2026, against
[upstream commit `4dceab6`](https://github.com/chiennv2000/orthrus/tree/4dceab65156b3dfb5dadbb11181a0e65d0ad314d):

- No matching public Rust port was found in upstream, the default branches of
  its 23 public forks, GitHub Rust repository searches, or crates.io. The
  similarly named Rust crates are unrelated. Upstream provides PyTorch and
  Python MLX implementations; this search does not establish that no port exists.
- The [published model zoo](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/README.md#model-zoo)
  contains Orthrus-Qwen3-1.7B, Orthrus-Qwen3-4B, and Orthrus-Qwen3-8B.
  No released Orthrus-Qwen3.5 checkpoint was found in the author's model listing.
- The newer [Qwen3.5 implementation](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/models/modeling_orthrus_qwen3_5.py#L663)
  adds training support. Its block-generation helper recomputes the full prefix;
  it does not supply a cached hybrid continuation path for OpenKind.
- Lossless generation preserves the base model's generated-token distribution.
  It does not establish better Jev decision quality or parity for OpenKind's
  calibrated probabilities. Reported generation speedups are not OpenKind
  request-latency measurements.

This is a source assessment, with no new runtime measurements, decision-quality
evaluation, or backend qualification. The
[architecture boundary](../docs/ARCHITECTURE.md#orthrus-block-token-generation)
explains the execution decision. The
[benchmark guide](../docs/BENCHMARKS.md#candidate-pooling-diagnostic) owns native
latency evidence and optimization promotion gates.

---

## Chronological Experiment Index

| # | Notebook | Phase / Run ID | Target HW | Core Research Focus | Status & Outcome | Supporting Artifacts |
|---|---|---|---|---|---|---|
| **1** | [`01_phase2_qwen35_probe.ipynb`](./01_phase2_qwen35_probe.ipynb) | Phase 2 Probe | GPU (L4/A100) | Frozen Qwen3.5 feature extraction & linear decision head | Feasibility proven; identified text vs vision parameter count | [Whitepaper §3.1](../docs/whitepaper/WHITEPAPER.md#31-research-chronology) |
| **2** | [`02_phase2b_benchmark.ipynb`](./02_phase2b_benchmark.ipynb) | Phase 2B (`20260917T205849Z`) | NVIDIA L4 (BF16) | MultiNLI classification, pooling strategies, calibration, generation vs decision timing | Last-token + linear won (87.7% matched); 44.6× speedup over generation; temperature scaling rejected | Documented on Google Drive / Notebook outputs |
| **3** | [`03_phase2c_stability_dynamic_choice.ipynb`](./03_phase2c_stability_dynamic_choice.ipynb) | Phase 2C (`20260917T222948Z`) | NVIDIA L4 (BF16) | Multi-seed stability, dynamic candidate scoring (Banking77), global `__none__` logit | Stable across seeds (87.0% / 88.8%); 80.8% seen / 64.6% unseen accuracy; 0.817 AUROC for none | Documented on Google Drive / Notebook outputs |
| **4** | [`04_phase2d_numerics_none_handling.ipynb`](./04_phase2d_numerics_none_handling.ipynb) | Phase 2D (`20260917T234417Z`) | NVIDIA L4 | Precision diagnostics (BF16 vs FP32 vs TF32), none-head alternatives, request latency vs K | Isolated layer divergence in DeltaNet/conv; linear latency scaling without prefix caching | Documented on Google Drive / Notebook outputs |
| **5** | [`05_phase2e_selective_precision_initial.ipynb`](./05_phase2e_selective_precision_initial.ipynb) | Phase 2E initial (`20260918T032049180933Z`, v2e.1.1) | GPU (L4/A100) | Module-level FP32 promotion, shared-prefix KV branching, application policies | Full FP32 stage halted by memory guard; isolated hybrid state isolation requirement | Documented on Google Drive / Notebook outputs |
| **6** | [`06_phase2e_expanded_batched_prefix_parity.ipynb`](./06_phase2e_expanded_batched_prefix_parity.ipynb) | Phase 2E expanded (`20260918T114914072764Z`, v2e.2.0) | GPU (isolated processes) | Strict FP32 reference, batched prefix reuse parity vs BF16, component breakdown | Strict FP32 prefix reuse passed ($Δp \le 1.1 \times 10^{-5}$); BF16 failed tolerance ($>0.005$) and flipped actions | Documented on Google Drive / Notebook outputs |
| **7** | [`07_phase2f_cache_compression_prefix_reuse.ipynb`](./07_phase2f_cache_compression_prefix_reuse.ipynb) | Phase 2F (`20260918T224427722898Z`, v2f.1.0) | NVIDIA L4 / A100 | TurboQuant low-bit KV compression (2/3/4-bit) vs FP16/lossless, GPU LRU prefix cache | Lossless & FP16-KV passed all gates; all 4 low-bit TurboQuant variants failed and flipped actions | Documented on Google Drive / Notebook outputs |
| **8** | [`08_phase2g_fresh_evidence_tf32_cache.ipynb`](./08_phase2g_fresh_evidence_tf32_cache.ipynb) | Phase 2G (`20260919T005142584348Z`, v2g.1.0) | NVIDIA L4 / A100 | Generalization to fresh Banking/CLINC/OOS data, TF32 execution, cache TTL & eviction | TF32 cut latency ~2× but flipped 3 argmax decisions; revealed candidate head narrowness on OOS | Documented on Google Drive / Notebook outputs |
| **9** | [`09_phase2h_criteria_rejection_multidomain.ipynb`](./09_phase2h_criteria_rejection_multidomain.ipynb) | Phase 2H (`20260919T040612625670Z`, v2h.1.2) | NVIDIA L4 / A100 | Multi-domain candidate fitting, criteria augmentation (support examples), rejection transfer | Original criteria transferred better (83.8% acc) than support examples (78.9% acc); held to pre-registration | Documented on Google Drive / Notebook outputs |
| **10** | [`10_phase2ij_gated_workbench.ipynb`](./10_phase2ij_gated_workbench.ipynb) | Phase 2I/2J Workbench (`2ij.1.0`) | NVIDIA L4 / A100 | Multi-question reviewed study gate, state-first vs instruction-first GPU mechanics probe | Strict independent-review gate blocked unverified training; probe validated state-first prefill mechanics | Documented on Google Drive / Notebook outputs |
| **11** | [`11_phase2ij_model_selection_screen.ipynb`](./11_phase2ij_model_selection_screen.ipynb) | Phase 2I/2J Screen (`2ij.2.0`) | NVIDIA L4 / A100 | 13 fit jobs, 31 evaluation profiles across Qwen 3.5 4B, Qwen 2.5 1.5B/2B, ModernBERT | Selected & locked profile `a047d6802c3f06f085b8` (Qwen3.5-4B state-first score-summary head) | Documented on Google Drive / Notebook outputs |
| **12** | [`12_phase2ij_a100_bundle_export.ipynb`](./12_phase2ij_a100_bundle_export.ipynb) | Phase 2I/2J Continuation | NVIDIA A100 | Clean selected-model bundle export and checksum verification on high-memory GPU | Produced golden bundle `4d9ffdee...3332` with zero OOM risk; validated reference repo | Informs profile `a047d6802c3f06f085b8` |
| **13** | [`13_phase3a_branchable_state_batched_q.ipynb`](./13_phase3a_branchable_state_batched_q.ipynb) | Phase 3A (`20260920T024056Z`) | NVIDIA L4 / A100 | Hybrid state branching (KV + DeltaNet + conv), batched-Q/K topologies, crossover behavior | Zero-leakage branch isolation confirmed; batched-Q faster for long state/high Q; informed Rust scheduler | [Whitepaper §16](../docs/whitepaper/WHITEPAPER.md#16-phase-3a-full-hybrid-branchablestate-batched-qk-execution-and-the-rust-handoff) |
| **14** | [`14_phase3b_backbone_parity.ipynb`](./14_phase3b_backbone_parity.ipynb) | Phase 3B (`20260920T152206Z`) | GPU (FP32) | Layer-by-layer backbone traces, golden token fixtures, Rust handoff contract | Exported 47 FP32 vectors across 34 stages; maximum cached continuation drift $\le 1.91 \times 10^{-5}$ | Documented on Google Drive / Notebook outputs |
| **15** | [`15_phase4a_statequery_workbench.ipynb`](./15_phase4a_statequery_workbench.ipynb) | Phase 4A.0 / 4A.1 (`20260921T013558Z`, v4a.0.2) | GPU (L4/A100) | Natural-document corpus lock (ContractNLI + QASPER) & StateQuery B1 representation probe | Locked 2,192 states / 15,368 Qs; B1 scored 0.7796 macro-acc but collapsed QASPER none-recall to 0.0 | [Whitepaper §18](../docs/whitepaper/WHITEPAPER.md#18-phase-4a4e-locked-benchmark-applicability-experiments-and-audit-gate) |
| **16** | [`16_phase4a2_statequery_model_comparison.ipynb`](./16_phase4a2_statequery_model_comparison.ipynb) | Phase 4A.2 (`4a.2.0`) | GPU (L4/A100) | Matched B0 control, source-balanced B1, factorized B2 without reopening final splits | B0 matched collapse (0.0 none-recall); balanced B1 raised none-recall to 0.168 (gate threshold: 0.30) | [Whitepaper §18.4](../docs/whitepaper/WHITEPAPER.md#184-completed-non-final-comparison-aggregate-improvement-hides-a-qasper-collapse) |
| **17** | [`17_phase4b2_statequery_model_comparison.ipynb`](./17_phase4b2_statequery_model_comparison.ipynb) | Phase 4B.2 (`4b.2.0` / `4b.2.1`) | GPU (L4/A100) | Scalar-weight saturation (cap 12) & source/class-stratified applicability | Realized weight saturated at 8.5602; stratified objective raised QASPER recall to 0.3111 gate, but gate policy cost failed (0.11675) | Documented on Google Drive / Notebook outputs |
| **18** | [`18_phase4b3_applicability_ranking_sweep.ipynb`](./18_phase4b3_applicability_ranking_sweep.ipynb) | Phase 4B.3 (`4b.3.0`) | GPU (L4/A100) | Within-state pairwise applicability ranking loss ($w \in \{0.25, 0.50, 1.00\}$) | Cleared development recall (0.3451), but failed gate false-none (>0.21) and policy cost (0.10221); rejected | [Whitepaper §18.10](../docs/whitepaper/WHITEPAPER.md#1810-phase-4b3-pairwise-applicability-passes-development-but-not-transfer) |
| **19** | [`19_phase4c_decoupled_applicability_head.ipynb`](./19_phase4c_decoupled_applicability_head.ipynb) | Phase 4C (`4c.0.0`) | GPU (L4/A100) | Isolated applicability head fine-tuning over frozen B2 representations | Child epochs regressed NLL (+2.77%) and Brier (+2.10%) for only +2 true positives; parent epoch 0 retained | Documented on Google Drive / Notebook outputs |
| **20** | [`20_phase4d_evidence_aware_applicability.ipynb`](./20_phase4d_evidence_aware_applicability.ipynb) | Phase 4D (`4d.0.0`) | GPU (L4/A100) | 19-dim handcrafted evidence/uncertainty diagnostic residual over frozen parent | Strict JSON enforced; child gained only +1 true positive while NLL worsened +6.08%; child rejected | Documented on Google Drive / Notebook outputs |
| **21** | [`21_phase4e_qasper_error_audit.ipynb`](./21_phase4e_qasper_error_audit.ipynb) | Phase 4E (`4e.0.0`) | CPU | Blinded human audit of 150 QASPER false-negative, false-positive, and control cases | Prepares double-blind adjudication pack to separate representation failure from label ambiguity before 4E-B | Documented on Google Drive / Notebook outputs |
| **22** | [`22_phase4e_a2_qasper_audit_repair.ipynb`](./22_phase4e_a2_qasper_audit_repair.ipynb) | Phase 4E-A2 (`20260921T013558Z`, `qasper_error_audit_repair_s17`) | CPU | Repaired QASPER error/evidence audit with complete paper text; primary blinded review & adjudication packet generation | Validated hash chain, 66.4% decided agreement, 51 adjudication rows, asymmetric disagreement (44 challenged semantic_none, 6 representation/serialization defects); Phase 4E-B blocked pending benchmark/representation repair | Documented on Google Drive / Notebook outputs |
| **23** | [`23_phase4e_a3_qasper_followup.ipynb`](./23_phase4e_a3_qasper_followup.ipynb) | Phase 4E-A3 (`20260921T013558Z`, `qasper_followup_cpu_s17_v1`) | CPU | QASPER review follow-up, 87 numeric cell restorations, frozen-state evidence candidates, and external representation leads | Restored 87 numeric cells; 51-row V2 adjudication (33 answerable, 13 semantic-none, 5 ambiguous); 13 exact state candidate spans; 6 non-independent assistant follow-ups scoped; no benchmark mutations; Phase 4E-B training unauthorized | Documented on Google Drive / Notebook outputs |
| **24** | [`24_phase4e_a4_qasper_source_alignment.ipynb`](./24_phase4e_a4_qasper_source_alignment.ipynb) | Phase 4E-A4 (`20260921T013558Z`, `qasper_source_alignment_cpu_s17_v1`) | CPU | QASPER upstream source alignment preflight, span partition verification, and audit ledger V2 | Verified 12 development candidate spans (7 Qs) and 1 gate diagnostic span; quarantined 3 unresolved evidence cases; aligned with upstream allenai/qasper (train/val only); final unopened; training remains unauthorized | Documented on Google Drive / Notebook outputs |
| **25** | [`25_phase4e_b1_candidate_option_logit_audit.ipynb`](./25_phase4e_b1_candidate_option_logit_audit.ipynb) | Phase 4E-B.1 (`candidate_option_logit_gate16_s17_v2`) | GPU / CPU (FP32) | Constrained next-token candidate option readout and uncalibrated `Z` rejection over 16-state gate sample (N=325) | Answerable ContractNLI ranking improved (87.8% vs 31.1% StateQuery ref); uncalibrated `Z` failed semantic none (0/124 none recall); order reversal flipped winners on 6.6% of questions; QASPER accuracy 77.4% (below 84.9% majority baseline); model promotion rejected | Documented on Google Drive / Notebook outputs |
| **26** | [`26_phase4e_b2_candidate_ranking_sweep.ipynb`](./26_phase4e_b2_candidate_ranking_sweep.ipynb) | Phase 4E-B.2 (`candidate_detection_ranking_sweep_s17_n12_v1`) | GPU / CPU (FP32) | Candidate ranking (3 rankers) and logistic-calibrated semantic-none detection (3 detectors) sweep across 12-state splits (N=741) | Cache parity verified ($\le 3.3 \times 10^{-6}$); order-averaged ranking hit 87.0% gate accuracy on ContractNLI; ContractNLI none recall hit 75.0% but gate false-none was 42.6% (violating $\le 0.20$ guardrail); QASPER selected grid collapsed to 0/5 none recall; cross-source calibration diverged; final closed | Documented on Google Drive / Notebook outputs |
| **27** | [`27_m22_matched_decision_lora.ipynb`](./27_m22_matched_decision_lora.ipynb) | M2.2 (`openkind-m22-contract-only-lora/v1`, v0.4.1) | GPU (A100 BF16 / FP32 adapters) | Upstream matched decision-LoRA pilot across 4 arms (J0/J1 controls, J2/J3 rank-16 LoRA) under full-document ContractNLI supervision and QASPER transfer | Stable backward recomputation pinned to `SDPBackend.MATH`; gated on development NLL ($\ge$5-pt contradiction gain, $\le$5-pt other loss, $\le$3-pt QASPER loss); final remains closed | Milestone M2 (Qwen 4B decision model) |
| **28** | [`28_source_label_replay.ipynb`](./28_source_label_replay.ipynb) | v0.6.0 (`source_label_v060_s17_bf16_dbb5e724b5452f23`) | NVIDIA A100 (BF16 / FP32 adapters) | Source-label cross-entropy replay vs parent-KL consistency on SNLI to test joint ContractNLI/QASPER preservation | SNLI accuracy & probability scores improved (+17.7pp J6 vs J4, +18.2pp J7 vs J5); ContractNLI entailment and QASPER false-none failed preservation bounds; frozen parents retained | Milestone M2 (Qwen 4B decision model) |
| **29** | [`29_qwen_moe_decision_lab.ipynb`](./29_qwen_moe_decision_lab.ipynb) | MoE Lab v0.2 (`20260926T224132_613995Z`) | NVIDIA L4 (NF4 / BF16 compute) | Qwen MoE decision inference, expert routing sparsity, top-k truncation, expert allowlisting, late MoE bypass, exact prefix state sharing, and physical weight residency | Prefill touches 98.7%–99.6% of experts (active params $\ne$ resident VRAM); top-1 routing cuts latency by 1.59x but drops accuracy by 15.6pp; late MoE skip preserves 53.1% acc (vs 56.3% native) at 1.32x speedup; direct selected readout achieves exact zero-delta parity; shared prefix state fails strict parity on MoE without full router isolation; physical pruning frees memory only when non-routed modules are deleted | Documented on Google Drive / Notebook outputs |
| **30** | [`30_qwen_moe_quality_and_cache_followup.ipynb`](./30_qwen_moe_quality_and_cache_followup.ipynb) | MoE Follow-up v0.1 (`20260927T003918_481825Z`) | NVIDIA L4 (NF4 / BF16 compute) | Multi-split decision quality (252 Qs, 84 states), prompt selection, mass-matched top-k, late-block skip, FP32 linear reference cache numerics, option order diagnostics | Explicit three-way prompt won dev; `skip_last_6` selected on dev and evaluated on 96 fresh test cases (41.7% vs 37.5% native, 1.32x speedup, 8.3% coverage vs 14.6% native); native Unknown recall 0/32; mass-matched half-k beats raw half-k in NLL (2.038 vs 2.122); cache parity failed on both native ($\Delta p=0.120$) and FP32 linear reference ($\Delta p=0.155$); option order flips 25.0% of decisions ($\max \Delta p = 0.169$); research gate failed, promotion rejected | [Drive run](https://drive.google.com/drive/folders/1gXwnF1sXhR5Yh4-7R2izDQ9TJXDl_Lf1) / Notebook outputs |
| **31** | [`31_qwen_prefill_speed_accuracy_lab.ipynb`](./31_qwen_prefill_speed_accuracy_lab.ipynb) | MoE Prefill Speed & Accuracy Lab v0.2 (`e15e1e9f7a64e464a38354c59b0c79805d59bc13d517f7e4fca66873e5d5ff2e`) | NVIDIA A100-SXM4-40GB (vLLM 0.30.0, BF16 / GPTQ INT4) | Dense Qwen3.5-4B vs Qwen3.5-35B-A3B MoE INT4 prefill speed, exact-prefix caching, repeat/concurrency drift, and PrivateMode-style decision readout | Cache qualification failed (prefixes 59–105 tokens < 528/1,056 runtime blocks; 0 reused tokens); probability drift observed without cache reuse (MoE sequential repeat max $\Delta p = 17.60$ pp, concurrent vs seq $\max \Delta p = 11.92$ pp; 4B concurrent $\max \Delta p = 3.28$ pp); research gate failed, promotion rejected | [Drive run](https://drive.google.com/drive/folders/1LIKE7JSmEcO2fvrhf8Qx4_ZJfv-heGwD) / Notebook outputs |
| **32** | [`32_qwen_cache_and_native_decisions_lab.ipynb`](./32_qwen_cache_and_native_decisions_lab.ipynb) | Qwen Cache & Native Decisions Lab (`7047c6b31436f8e9b5aa85a5dad9ea4378d16eaa0912ebba288fae273c7e12ae`) | NVIDIA A100-SXM4-40GB (vLLM 0.30.0 & llama.cpp `parallel-decision`) | Controlled prefix boundary sweep (527–2,113 tokens), vLLM repeatability (serial vs concurrent), batch-invariance launch, and native tree branching | vLLM repeatability passed 1/4 rows (4B serial passed with $\Delta p = 0.0$; concurrent and MoE serial/concurrent failed with drift up to 29.81 pp); cache boundary confirmed (0 hits below 528/1,056; 528/1,056/2,112 tokens reused when exceeding block boundaries); 6/40 cache rows qualified; batch-invariance and full llama GPU offload threw CapabilityError; no model/cache promoted | Documented on Google Drive / Notebook outputs |
| **33** | [`33_qwen_readout_rules_history_lab.ipynb`](./33_qwen_readout_rules_history_lab.ipynb) | Readout, Rules & History (`20260927T192845_426758Z`, run key `ff6fd499...308a`) | NVIDIA A100-SXM4-40GB (Q4_K_M vs BF16) | Single-token integer codes vs natural labels, deterministic host action derivation (5-field + rule vs 6-field), exact-prefix caching, sequence reservation (24 vs 3), batch shapes (1 vs 4 contexts), request history / state leakage | 5-field + rule eliminated eligibility/action contradictions and lowered NLL; all 8 cache conditions passed exact parity ($\Delta p = 0.0$, 1.24–2.75x speedup); isolated native history passed, but JSON interleaving and sequence reservation interactions caused drift; model/cache not promoted | [Drive run](https://drive.google.com/drive/folders/1N8fq_wSct874PWi-VPwuGWAKiYL39xUM) / Notebook outputs |
| **34** | [`34_qwen35_9b_t4_l4_open_questions_lab.ipynb`](./34_qwen35_9b_t4_l4_open_questions_lab.ipynb) | 9B T4/L4 Open Questions (`20260928T220142_110595Z_1192c8`) | NVIDIA Tesla T4 16GB (Qwen3.5-9B Q4_K_M, 34/34 offloaded) | Qwen3.5-9B Q4 feasibility on T4; 5 arms (A: 6 fields recomputed, B: 5 fields + rule recomputed, C: 6 fields warm prefix, D: 5 fields + rule warm prefix, E: JSON); exact prefix reuse; genuine first-use cold trace; JSON interleaving history drift; sequence reservation (8 vs 3); risk coverage; offline deterministic route replay | T4 feasibility demonstrated (6.30 GiB peak VRAM); Arm D delivered 2.29–2.35x speedup over A (339.38 ms median) with 56% lower mean latency from prefix reuse and 0 eligibility/action contradictions; prefix cache passed 330/330 pairs ($\Delta p = 0$); JSON interleaving failed history gate ($\max \Delta p = 0.095$, 4 flips); 3 reserved sequences caused drift ($\Delta p = 0.059$); D showed two systematic errors (plain closed $\to$ duplicate; priority suffix appended); route replay healed route errors (93.8% field acc, 62.5% all-six); complete service not qualified | Review writeup `OpenKind_9B_T4_Results_Review_20260928.md` / [Drive run](https://drive.google.com/file/d/1Q-t2VhbH7qYUbkDodlg4IPgpNK8ySBa3/view) |
| **35** | [`35_local_decision_training.ipynb`](./35_local_decision_training.ipynb) | Local Decision Training | GPU (A100 / L4 NF4) | Self-contained Colab trainer for mixed-task Qwen3.5-4B decision LoRA across public tasks, teacher decisions, and rule examples | Authored & locally validated with tiny models; full 4B CUDA training unrun | [`local_decision_training/`](./local_decision_training) |
| **36** | [`36_openkind_unified_decision_validity_lab.ipynb`](./36_openkind_unified_decision_validity_lab.ipynb) | Unified Validity Lab (`20260929T194506_694066Z_dc1f74`, protocol `openkind-unified-decisions/v3.0.0`) | NVIDIA Tesla T4 16GB & CPU (Go 1.27.1) | Selective indexing (X), route composition (R/GR), grouped readouts (G/GR), joint formulation (J), resident process isolation vs restarts, schema sensitivity, scaling (native catalogue vs indecis open-option), CPU SIMD/assembly, calibration & confidence policies | Selective indexing (Arm X: 92.19% field acc, 60.42% all-six, 222.22 ms) and grouped routing (Arm GR: 95.49% field acc, 83.33% all-six, 1419.57 ms) advance the design; resident process isolation passes exact parity ($\Delta p = 0.0$) with ~4–6% trace overhead; Indecis CPU is fast (46 ms) but fails transfer (69.10% fresh policy, 39.58% SNLI); no promotion | [Drive run](https://drive.google.com/drive/folders/1gLVnuSIHUDAqyQlyxFyKuc2LWcOvklgx) / [Archive](https://drive.google.com/file/d/1yXOks6ms6-aaEhj-jZ1WvuWtBQsIIzjA/view) |
| **37** | [`37_openkind_t4_integrated_research_lab.ipynb`](./37_openkind_t4_integrated_research_lab.ipynb) | OpenKind T4 Integrated Research (`20261001T105345_828556Z_6e6d92`, protocol `openkind-t4-integrated-research/v4.0.0`) | NVIDIA Tesla T4 16GB (Qwen3.5-9B Q4_K_M, Qwen3.5-4B Q4_K_M) & CPU | Integrated readouts (XR, NJ), multi-catalogue host cache (G/GR), dedicated-process history isolation (X, XR, NJ), 4B vs 9B model size, batching diagnostics, schema & dynamic choice probes, calibration & action/review policies, encoder setup logging diagnostic | PARTIAL (no model or service promoted; completion marker absent); XR achieves 95.40% field acc / 84.38% all-six at 242.98 ms; NJ achieves 94.01% field acc / 79.69% all-six at 228.31 ms; multi-catalogue host cache cuts grouped request latency by ~54% (2.19x ratio) with exact output parity; dedicated processes pass history checks ($\Delta p = 0$); batching larger contexts does not improve throughput; schema sensitivity persists; encoder comparisons blocked by SameFileError in log helper | Review writeup (1 Oct 2026) / Run `20261001T105345_828556Z_6e6d92` |
| **38** | [`38_openkind_t4_recovery_reliability_lab.ipynb`](./38_openkind_t4_recovery_reliability_lab.ipynb) | T4 Recovery & Targeted Reliability (`20261001T202134_352688Z_4a8cbc`, protocol `openkind-t4-recovery-reliability/v4.1.0`) | NVIDIA Tesla T4 16GB & CPU | SameFileError log repair; safetensors 0.8.0 pin; blocked classifier recovery (BERT, ModernBERT, DeBERTa, GLiClass, Indecis); native prompt failure-first eligibility (NF) and joint factual readout (PF); dependency-aware policies across 378 balanced cases | PARTIAL (23 executed, 2 failed in ensurepip, 15 blocked); Native Qwen NF repairs targeted partial-information eligibility (48/48 vs 34/48 NJ) but retry regression drops all-six correctness (28.70% vs 33.33%); PF joint categories fail extraction (190/216); indecis input reuse delivers 4.80x speedup (117.13 ms $\to$ 24.38 ms) with identical outputs; dependency-aware policies lose to review-all; dedicated processes pass history checks ($\Delta p = 0$) | [Drive run](https://drive.google.com/file/d/1uAUvISSExacRD13iWpLIsNAYhENApY-S/view) / Archive `4a8cbc` |
| **39** | [`39_openkind_t4_recovery_reliability_v4_1_1.ipynb`](./39_openkind_t4_recovery_reliability_v4_1_1.ipynb) | T4 Classifier Environment Recovery (`20261001T223435_041738Z_f25616`, protocol `openkind-t4-recovery-reliability/v4.1.1`) | NVIDIA Tesla T4 16GB & CPU | Venv `ensurepip` failure repair via `--without-pip` and hash-pinned pip 26.2.1 target interpreter seeding; 17 classifier recovery stages (DeBERTa NLI, ModernBERT, BERT-base, GLiClass); retains unchanged U4 splits and r41_* targets | EXPLORATORY_COMPLETE (All 17 classifier stages executed); venv seeding fixed; DeBERTa NLI specialist reaches 90.10% acc (173/192) at 20.92 ms (vs Qwen 86.98% at 149.13 ms) with calibrated NLL 0.2972, but dev threshold fails review-all cost gate; BERT full fine-tuning (73.46% field / 21.30% all-six) and ModernBERT (70.45% / 19.44%) beat frozen heads (~51-52%) but collapse to majority classes; GLiClass yields 45.45% field acc / 3.24% all-six with long-input budget rejections | [Drive run](https://drive.google.com/file/d/1fpZNrhNLue6q55FYNH1AKrf7Pv1u4iHR/view) / Archive `f25616` |



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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, receipts, and benchmark logs are archived on Google Drive.
  - **Reproducibility**: All benchmark metrics, pooling comparisons, and calibration curves can be regenerated by running [`02_phase2b_benchmark.ipynb`](./02_phase2b_benchmark.ipynb).
  - **Key Output Files**: `openkind_phase2b_summary.md`, `decision_benchmark.json`, `generation_benchmark.json`, `metrics.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, stability logs, and predictions are archived on Google Drive.
  - **Reproducibility**: Multi-seed stability metrics and dynamic candidate predictions can be regenerated by running [`03_phase2c_stability_dynamic_choice.ipynb`](./03_phase2c_stability_dynamic_choice.ipynb).
  - **Key Output Files**: `openkind_phase2c_summary.md`, `stability.json`, `dynamic_predictions_test_seen.json`, `dynamic_predictions_test_unseen.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, layer drift traces, and precision evaluations are archived on Google Drive.
  - **Reproducibility**: Layer drift data and numerical precision comparisons can be regenerated by running [`04_phase2d_numerics_none_handling.ipynb`](./04_phase2d_numerics_none_handling.ipynb).
  - **Key Output Files**: `openkind_phase2d_summary.md`, `numerics/layer_drift.csv`, `numerics/single_precision_drift.json`, `none_models.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete run logs and execution reports are archived on Google Drive.
  - **Reproducibility**: Selective precision probes and state-prefill tests can be regenerated by running [`05_phase2e_selective_precision_initial.ipynb`](./05_phase2e_selective_precision_initial.ipynb).
  - **Key Output Files**: `README_results.md`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, parity tables, and component profiles are archived on Google Drive.
  - **Reproducibility**: Prefix branching and numerical parity evaluations across FP32/BF16 can be regenerated by running [`06_phase2e_expanded_batched_prefix_parity.ipynb`](./06_phase2e_expanded_batched_prefix_parity.ipynb).
  - **Key Output Files**: `README_results.md`, `fp32_strict_math/parity_rows.json`, `bf16_default/parity_rows.json`, `fp32_strict_math/component_profiles.json`, `bf16_default/component_profiles.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, compression parity metrics, and cross-request summaries are archived on Google Drive.
  - **Reproducibility**: TurboQuant quantization and prefix caching experiments can be regenerated by running [`07_phase2f_cache_compression_prefix_reuse.ipynb`](./07_phase2f_cache_compression_prefix_reuse.ipynb).
  - **Key Output Files**: `README_results.md`, `fp32_strict_math/compression_parity.json`, `fp32_strict_math/compression_quality.json`, `fp32_strict_math/cross_request_summary.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, fresh generalization data, and TF32 comparison files are archived on Google Drive.
  - **Reproducibility**: Cross-precision evaluations and cache lifecycle benchmarks can be regenerated by running [`08_phase2g_fresh_evidence_tf32_cache.ipynb`](./08_phase2g_fresh_evidence_tf32_cache.ipynb).
  - **Key Output Files**: `README_results.md`, `cross_precision_fp32_tf32_allowed.json`, `fp32_strict_math/fresh_rows.json`, `fp32_strict_math/traffic_summary.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, final contrast metrics, and evaluation summaries are archived on Google Drive.
  - **Reproducibility**: Multi-domain criteria transfer and rejection experiments can be regenerated by running [`09_phase2h_criteria_rejection_multidomain.ipynb`](./09_phase2h_criteria_rejection_multidomain.ipynb).
  - **Key Output Files**: `README_results.md`, `eval_qwen4b_strict/final_metrics.json`, `eval_qwen4b_strict/paired_final_contrasts.json`.

---

### 10. Phase 2I/2J: Gated Multi-Question Workbench
* **File**: [`10_phase2ij_gated_workbench.ipynb`](./10_phase2ij_gated_workbench.ipynb)
* **Version**: `2ij.1.0`
* **What it Measured**:
  - Setup of a formal review gate requiring independent human audit of task criteria and split manifests before launching expensive training.
  - Small synthetic GPU probe validating **state-first segmented tokenization**:
    $$\text{Tokens} = [\text{State Prefix}] + [\text{Question Suffix}] + [\text{Candidate Suffix}]$$
    contrasted with standard instruction-first rendering.
* **Results**:
  - The review gate successfully blocked unreviewed multi-question training, preventing unverified claims.
  - Synthetic GPU probe validated state-first mechanics: prefilling the state document once and branching questions/candidates reduced compute without token leakage.
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, gate report, and handoff contracts are archived on Google Drive.
  - **Reproducibility**: Review gate and state-first segmented tokenization probes can be regenerated by running [`10_phase2ij_gated_workbench.ipynb`](./10_phase2ij_gated_workbench.ipynb).
  - **Key Output Files**: `REPORT.md`, `gate_report.json`, `contracts/SERVICE_HANDOFF.md`.

---

### 11. Phase 2I/2J: Model Selection Screen & Architectural Decision
* **File**: [`11_phase2ij_model_selection_screen.ipynb`](./11_phase2ij_model_selection_screen.ipynb)
* **Version**: `2ij.2.0`
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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, model selection reports, and profile decision records are archived on Google Drive.
  - **Reproducibility**: Model screening across 13 fit jobs and 31 evaluation profiles can be regenerated by running [`11_phase2ij_model_selection_screen.ipynb`](./11_phase2ij_model_selection_screen.ipynb).
  - **Key Output Files**: `REPORT.md`, `MODEL_DECISION.md`, `contracts/SERVICE_HANDOFF.md`.

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
    - Directly informed the implementation of the capability-aware `choose_strategy` scheduler in `openkind-backends` (`qwen35/backbone/strategy.rs`).
* **Supporting Documentation**:
  - [OpenKind Whitepaper §16](../docs/whitepaper/WHITEPAPER.md#16-phase-3a-full-hybrid-branchablestate-batched-qk-execution-and-the-rust-handoff).

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, golden token fixtures, activation traces, and safetensors weights are archived on Google Drive.
  - **Reproducibility**: Golden token fixtures, layer traces, and parity contracts can be regenerated by running [`14_phase3b_backbone_parity.ipynb`](./14_phase3b_backbone_parity.ipynb).
  - **Key Output Files**: `RUST_BACKBONE_HANDOFF.md`, `TOKEN_FIXTURES.json`, `PROBABILITY_REFERENCE.json`, `CONTINUATION_TRACE.json`, `QWEN35_BACKBONE_GOLDEN.safetensors`.

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
  - [OpenKind Whitepaper §18](../docs/whitepaper/WHITEPAPER.md#18-phase-4a4e-locked-benchmark-applicability-experiments-and-audit-gate).

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, training reports, result locks, and comparison snapshots are archived on Google Drive.
  - **Reproducibility**: Scalar-weight saturation and source/class-stratified applicability training can be regenerated by running [`17_phase4b2_statequery_model_comparison.ipynb`](./17_phase4b2_statequery_model_comparison.ipynb).
  - **Key Output Files**: `b2q_weight12_s17/TRAINING_REPORT.json`, `b2_applicability_stratified_s17/NONFINAL_RESULT_LOCK.json`, `comparison_snapshot_20260922T115817Z.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, head-only training reports, and evaluation snapshots are archived on Google Drive.
  - **Reproducibility**: Decoupled applicability head fine-tuning can be regenerated by running [`19_phase4c_decoupled_applicability_head.ipynb`](./19_phase4c_decoupled_applicability_head.ipynb).
  - **Key Output Files**: `b2_app_headonly_guard18_s17/TRAINING_REPORT.json`, `b2_app_headonly_guard18_s17/NONFINAL_RESULT_LOCK.json`, `comparison_snapshot_20260922T172543Z.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, residual training reports, and lock files are archived on Google Drive.
  - **Reproducibility**: Evidence-aware diagnostic residual evaluations can be regenerated by running [`20_phase4d_evidence_aware_applicability.ipynb`](./20_phase4d_evidence_aware_applicability.ipynb).
  - **Key Output Files**: `b2_evidence_residual_s17/TRAINING_REPORT.json`, `b2_evidence_residual_s17/NONFINAL_RESULT_LOCK.json`, `comparison_snapshot_20260922T200741Z.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw audit directory, blinded CSVs, analysis JSONs, and lock files are archived on Google Drive.
  - **Reproducibility**: QASPER blinded error audit generation and statistical analysis can be regenerated by running [`21_phase4e_qasper_error_audit.ipynb`](./21_phase4e_qasper_error_audit.ipynb).
  - **Key Output Files**: `AUDIT_GUIDE.md`, `AUDIT_REVIEW_BLINDED.csv`, `AUDIT_ANALYSIS.json`, `AUDIT_RESULT_LOCK.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory and audit files are archived on Google Drive:
    - Adjudication Packet: [`AUDIT_ADJUDICATION_PACKET.csv`](https://drive.google.com/file/d/1eu6evYz9JzRQBSr9eJa5g_XpK6szIKE1/view) (SHA-256 `2d6b3be4dfa532de7c0c320d4e84b67d2be571ff4ddcc7f886b22f437e88edeb`)
    - Analysis JSON: [`AUDIT_ANALYSIS.json`](https://drive.google.com/file/d/1-UTO4L8gRy5NrbAracI7o2jGxJoXyGu4/view) (SHA-256 `858d1b7b08d2e3e9969f7de4bdc1780cbaf83a831c379b9529d4acfe2a84b25f`)
    - Result Lock: [`AUDIT_RESULT_LOCK.json`](https://drive.google.com/file/d/1n7Xb9Ye_ek9kyHt1t6yRqKi1GMgJXY_c/view) (SHA-256 `89d0e283a2b743efcb95de3f73189c6fb0157620c5f6252beea149a44d1ecefd`)
  - **Reproducibility**: The repaired audit packet and adjudication tables can be regenerated by running [`22_phase4e_a2_qasper_audit_repair.ipynb`](./22_phase4e_a2_qasper_audit_repair.ipynb).

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory and adjudication report files are archived on Google Drive.
  - **Reproducibility**: The V2 adjudication analysis, span candidates, and decision records can be regenerated by running [`23_phase4e_a3_qasper_followup.ipynb`](./23_phase4e_a3_qasper_followup.ipynb).
  - **Key Output Files**: `PHASE4E_A3_DECISION_RECORD.md`, `AUDIT_ADJUDICATION_REPORT_V2.json`, `PHASE4E_A3_NONFINAL_REPORT.json`, `AUDIT_REPAIR_LEDGER.csv`, `EVIDENCE_SPAN_CANDIDATES.csv`, `REPRESENTATION_SOURCE_CANDIDATES.csv`, `PHASE4E_A3_LOCK.json`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, alignment tables, and preflight records are archived on Google Drive.
  - **Reproducibility**: Source alignment preflight and span candidate partitions can be regenerated by running [`24_phase4e_a4_qasper_source_alignment.ipynb`](./24_phase4e_a4_qasper_source_alignment.ipynb).
  - **Key Output Files**: `PHASE4E_A4_PRELIMINARY_RECORD.md`, `PHASE4E_A4_PREFLIGHT.json`, `AUDIT_REPAIR_LEDGER_V2.csv`, `EVIDENCE_SPAN_POLICY_CANDIDATES.csv`, `QASPER_SOURCE_ROW_ALIGNMENT.csv`, `QASPER_SOURCE_CAPTION_CANDIDATES.csv`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, non-final evaluations, and row parquets are archived on Google Drive.
  - **Reproducibility**: Next-token candidate option readout evaluations and position checks can be regenerated by running [`25_phase4e_b1_candidate_option_logit_audit.ipynb`](./25_phase4e_b1_candidate_option_logit_audit.ipynb).
  - **Key Output Files**: `NONFINAL_EVALUATION.json`, `NONFINAL_RESULT_LOCK.json`, `EXPERIMENT_CONTRACT.json`, `NONFINAL_ROWS.parquet`.

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
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, sweep logs, and selection manifests are archived on Google Drive.
  - **Reproducibility**: Candidate ranking and semantic-none detection grid sweeps can be regenerated by running [`26_phase4e_b2_candidate_ranking_sweep.ipynb`](./26_phase4e_b2_candidate_ranking_sweep.ipynb).
  - **Key Output Files**: `DEVELOPMENT_SELECTION.json`, `NONFINAL_EVALUATION.json`, `NONFINAL_RESULT_LOCK.json`, `EXPERIMENT_CONTRACT.json`, `NONFINAL_ROWS.parquet`.

---

### 27. M2.2: Matched Decision-LoRA Pilot
* **File**: [`27_m22_matched_decision_lora.ipynb`](./27_m22_matched_decision_lora.ipynb)
* **Run ID / Schema**: `openkind-m22-contract-only-lora/v1` (`m22_v041_...`, Version `0.4.1`, 25 September 2026; seed 17)
* **Target HW**: GPU (NVIDIA A100 Colab runtime, BF16 arithmetic with FP32 adapter parameters and loss; SDPBackend.MATH; fallback/CPU autograd preflight)
* **What it Measured / Goal**:
  - Tests whether full-document ContractNLI decision supervision improves contradiction and unsupported/“not mentioned” decisions without damaging cross-task retention on QASPER.
  - Implements the Roadmap Milestone M2 adaptation hypothesis ("Establish useful decisions"): testing one limited upstream adaptation hypothesis with an attributable head-only/frozen control.
  - Deliberately narrower than joint QASPER/ContractNLI training: no QASPER supervision, no repaired-label guessing, no teacher calls, no new evidence selector, and no broad hyperparameter sweep.
* **Four Matched Arms**:
  - **J0**: Pinned Base (`Qwen/Qwen3.5-4B-Base`, revision `1001bb4d826a52d1f399e183466143f4da7b741b`), frozen control.
  - **J1**: Pinned post-trained model, frozen control.
  - **J2**: Pinned Base + rank-16 decision LoRA ($\alpha=32$, learning rate $2 \times 10^{-5}$, max 120 updates, gradient accumulation 8).
  - **J3**: Pinned post-trained + matched rank-16 decision LoRA (identical data, exposure schedule, and optimizer budget).
* **Key Methodology & Scientific Controls**:
  - **SDPA Backward Stability (v0.4.1 Repair)**: Forward pass and activation-checkpoint backward recomputation are both strictly pinned to `SDPBackend.MATH`, preventing non-reentrant checkpoint backend switching and ensuring stable backward gradients across the complete microbatch.
  - **Exposure & Data Schedule**: Training mixture is 75% class-balanced natural anchors + 25% ordinary natural-document replay within 4,096 state tokens and 8,192 total tokens. Longer documents are excluded, never relabeled as unsupported.
  - **Selection Gate & Decision Criteria**: Development selection is strictly governed by ContractNLI development NLL evaluated at updates `[40, 80, 120]`—not training loss. Step-zero unadapted checkpoint is eligible to win.
  - **Pre-Registered Screen**: Requires $\ge 5$-point contradiction-recall gain, $\le 5$-point loss in other ContractNLI classes, and $\le 3$-point QASPER accuracy loss versus the matched frozen control; retains false-none ceiling $\le 0.20$ and proper-score non-regression rule ($\le 0.01$).
  - **Anti-Leakage Governance**: Protected final splits remain unopened (`final_opened: false`). No automatic model promotion or seed/sweep expansion.
* **Supporting Artifacts & Implementation References**:
  - Embedded scripts: `m22_train.py`, `m22_data.py`, `m22_eval.py`, `test_checkpoint_backend.py`.
  - Contracts & Registries: `EXPERIMENT_CONTRACT.json`, `SOURCE_REGISTRY.json`, `MODEL_PINS.json`, `ADAPTER_COVERAGE.json`.
  - Milestone Anchor: Milestone M2 (Select a useful Qwen 4B decision model).

---

### 28. OpenKind v0.6.0: Source-Label Replay vs Parent Consistency
* **File**: [`28_source_label_replay.ipynb`](./28_source_label_replay.ipynb)
* **Run ID / Session**: `source_label_v060_s17_bf16_dbb5e724b5452f23` (Session `20260926T170047_182185Z`, Review Date 26 September 2026; seed 17)
* **Target HW**: GPU (NVIDIA A100-SXM4-80GB Colab runtime, PyTorch 2.11.0+cu128, Transformers 5.17.0, BF16 backbone / FP32 adapters and loss; SDPBackend.MATH)
* **What it Measured / Goal**:
  - Tests whether replacing parent-KL replay consistency with **source-label cross-entropy** on auxiliary short-premise NLI (SNLI) resolves cross-task regression without sacrificing ContractNLI improvements.
  - Compares two fresh fits:
    - **J6**: Starts fresh from pinned frozen J0 Base, compared against J4 fixed update 80 (parent-KL replay).
    - **J7**: Starts fresh from pinned frozen J1 post-trained, compared against J5 fixed update 80 (parent-KL replay).
  - Main schedule retains 8 ContractNLI examples per optimizer update (960 complete-fit exposures / 640 at fixed 80) and 2 SNLI replay examples (coefficient 1.0; 240 complete-fit / 160 at fixed 80) across 120 updates.
  - Maintains rank 16, alpha 32, lr $2 \times 10^{-5}$, BF16 backbone / FP32 adapters, source-preserving renderer, and math SDPA checkpointing policy.
* **Key Findings & Outcomes**:
  - **Local Source-Label Signal Is Strongly Positive**:
    - On the 192-example SNLI regression panel, J6 reached **82.81% accuracy** (159/192, NLL 0.43881, Brier 0.24465), gaining **+17.71 pp (+34 net correct: 46 repairs vs 12 new errors)** over J4 (65.10%).
    - J7 reached **86.46% accuracy** (166/192, NLL 0.40708, Brier 0.21611), gaining **+18.23 pp (+35 net correct: 40 repairs vs 5 new errors)** over J5 (68.23%).
    - Proves that true source labels provide a substantially more effective learning signal for short-premise NLI than imitating imperfect frozen parent predictions.
  - **Cross-Task Joint Preservation Failed**:
    - **ContractNLI Entailment Loss**: J6 correct entailments dropped to 70/84 (-11 cases / -13.10 pp) vs J0 (81/84); J7 dropped to 69/84 (-12 cases / -14.29 pp) vs J1 (81/84). Both severely violated the $\le 5$-point class preservation tolerance.
    - **QASPER Deterioration**: J6 lost 7 decisions vs J4 (accuracy 63.04% vs 78.26%, NLL 0.76052 vs 0.48680) with a catastrophic false-none rate of **41.46% (17/41)**. J7 had unchanged accuracy (76.09%) but masked worsened probability quality (NLL 0.67368 vs 0.49796) and excessive false-none rate of **24.39% (10/41)**, exceeding the $\le 20\%$ safety ceiling.
    - **Bootstrap Uncertainty Favors KL**: Paired bootstrap comparisons (1,000 draws) show NLL degradation on the two-source gate for both J6 vs J4 ($+0.15517$ NLL) and J7 vs J5 ($+0.09904$ NLL).
  - **Guarded Selector Behavior**:
    - Neither evaluated non-zero candidate passed joint development constraints across updates 40, 80, or 120.
    - Both selection locks correctly rejected adapted candidates and retained step-zero frozen parents: $J6_{\text{selected}} = J0$ and $J7_{\text{selected}} = J1$.
  - **Research Conclusion**:
    - Closed this specific recipe. Supervised replay improves short-premise NLI accuracy and calibration, but does not provide a general cross-task preservation solution for long-document understanding and answerability.
    - Subsequent work must target the recurring supported-entailment regression directly on complete-document records rather than relying on short sentence-pair replay.
* **Supporting Evidence & Verification**:
  - Run ID: `source_label_v060_s17_bf16_dbb5e724b5452f23`
  - Result path: `Colab Notebooks/OpenKind_Source_Label_Replay_results/source_label_v060_s17_bf16_dbb5e724b5452f23/results/`
  - Reconciled and verified via standalone `review.py` passing 5,168 granular consistency assertions across 7,410 primary rows, 1,920 SNLI rows, and all manifest checksums.

---

### 29. Qwen MoE Decision Lab: Expert Routing, Selective Computation & Weight Residency
* **File**: [`29_qwen_moe_decision_lab.ipynb`](./29_qwen_moe_decision_lab.ipynb)
* **Run ID / Session**: `20260926T224132_613995Z` (Workbench v0.2, 26 September 2026; seed 17; profile `qwen15_moe`, model `Qwen/Qwen1.5-MoE-A2.7B-Chat` at revision `ec052fda178e241c7c443468d2fa1db6618996be`, NF4 precision with BF16 compute, eager attention)
* **Target HW**: GPU (NVIDIA L4, Linux 6.6.122+, Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.57.1, CUDA 12.8; allocated 7.78 GiB / peak 7.82 GiB / reserved 10.55 GiB)
* **Evidence & Provenance**:
  - Primary inspected run: [Drive run `20260926T224132_613995Z`](https://drive.google.com/drive/folders/1344_xljzgYaHEWJTUVEe7k3_2DXoM3Fi) (19 raw source files audited; dataset manifest verified against SHA-256 `84780a9e936a9857d797ab33a683a4ecb300fb737a68034751aed9c340e91fa2`).
  - Separately attributed completed run: `20260926T232551_755243Z` (user-reported physical-removal results; raw removal/cache artifacts held separately).
  - Population: 80 authored teaching questions across 20 source states.
  - Split sizes: Routing fit 16; calibration 16; policy development 16; test 32.
  - Test population: 8 states; 8 questions each across policy, temporal, arithmetic, and composition families.
  - Gold classes: Yes 12; No 12; Unknown 8.
* **What it Measured / Goal**:
  - Investigates whether Mixture-of-Experts (MoE) architectures provide latency or resident memory advantages for non-autoregressive decision inference without text generation.
  - Tests 7 experimental arms across the 32 test questions:
    1. `native`: Full top-k=4 routing across 60 routed experts + shared expert (5,632 width).
    2. `half_topk`: Truncated routing to top-k=2 active experts per token.
    3. `top1`: Aggressive routing to single top-1 active expert per token.
    4. `allowlist_50pct`: Static expert pruning retaining only top-50% most frequent experts (30/60 per layer) learned from `routing_fit`.
    5. `skip_late_moe`: Bypassing MoE MLP layers in late transformer blocks (layers 18–23), retaining attention and early routing.
    6. `shared_only`: Total bypass of all routed experts, routing solely through the dense shared expert.
    7. `native_repeat`: Timing and numerical verification repeat of the native baseline.
  - Evaluates direct decision readout projection (selected token IDs vs full vocabulary 151,936 tokens).
  - Evaluates expert utilization footprint across prefill context lengths (117, 245, 629 tokens).
  - Evaluates exact prefix state caching parity across independent questions.
  - Evaluates physical module deletion vs logical router masking on resident GPU memory.
* **Confirmed Measurements from Drive Run**:

  | Arm | Correct / 32 | Accuracy | Calibrated NLL | Raw NLL | Brier | p50 (ms) | Speedup | Unknown Recall | Automatic Coverage | Disagree vs Native | Peak VRAM (GiB) |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | `native` | 18 | 56.25% | 0.9926 | 2.2206 | 0.5870 | 1,749.10 | 1.00x | 0/8 (0%) | 0.00% | 0.0% | 7.82 |
  | `native_repeat` | 18 | 56.25% | 0.9926 | 2.2206 | 0.5870 | 1,747.70 | 1.00x | 0/8 (0%) | 0.00% | 0.0% | 7.82 |
  | `skip_late_moe` | 17 | 53.13% | 1.0571 | 1.8984 | 0.6342 | 1,326.38 | 1.32x | 1/8 (12.5%) | 0.00% | 12.5% | 7.82 |
  | `half_topk` | 13 | 40.63% | 1.0353 | 2.0026 | 0.6229 | 1,537.28 | 1.14x | 0/8 (0%) | 0.00% | 15.6% | 7.82 |
  | `top1` | 13 | 40.63% | 1.2205 | 3.4385 | 0.7548 | 1,099.32 | 1.59x | 0/8 (0%) | 21.88% | 18.8% | 7.82 |
  | `allowlist_50pct` | 12 | 37.50% | 1.2342 | 2.9754 | 0.7412 | 935.77 | 1.87x | 0/8 (0%) | 0.00% | 28.1% | 7.82 |
  | `shared_only` | 11 | 34.38% | 1.0993 | 1.1401 | 0.6671 | 77.18 | 22.66x | 3/8 (37.5%) | 0.00% | 68.8% | 7.82 |

  *Timing includes prompt rendering, tokenization, inference, synchronization, and CPU logit transfer (64 test timing observations per arm, from 2 measurements per question).*

* **Key Findings, Audit Details & Engineering Diagnostics**:
  - **Decision Quality and Abstention Failure**:
    - The native model predicted Yes 23 times, No 9 times, and Unknown 0 times. Zero false-Unknown coexists with total failure of Unknown recall (0/8). On the 24 answerable questions, 18 were correct; all 8 unanswerable questions failed.
    - The 50% allowlist drops to the 37.5% majority-class baseline, predicting Yes 30 times and No twice—losing 6 previously correct decisions with zero gains.
    - Skipping late MoE blocks changes 4 predictions: 2 correct policy answers become wrong, 1 arithmetic Unknown becomes correct, and 1 wrong policy becomes another wrong answer. The net loss of 1 correct hides that internal mixture.
    - The fitted review policy rejects every test answer in all arms except `top1`. In `top1`, it accepts 7 questions and gets 5 of those 7 wrong (`accepted_error = NaN` for other arms reflects zero accepted decisions, not zero error).
  - **Boundary-Limited Calibration Ceiling**:
    - All 7 arms selected $T=5$, the exact upper bound of the original search grid, establishing a boundary-limited fit rather than an optimal temperature.
    - Post-hoc diagnostic refits on the calibration split allowing uniform probability selected $T \approx 11.54$ for native (calibration NLL fell from 1.1168 to 1.0705, but test NLL worsened from 0.9926 to 1.0093). Skip-late favored $T \approx 18.08$ (test NLL worsened from 1.0571 to 1.0705). Allowlist and shared-only calibration fits favored uniform distributions.
  - **Per-Request Working Set Touches Nearly All Experts**:
    - Across 384 layer-request observations from `routing_fit`, the mean unique experts touched was **59.33 / 60 (98.9%)**, ranging from 56 to 60, with ~8 token rows per touched expert.
    - Length scaling confirms full working-set saturation: 59.21 experts at 117 tokens, 59.71 at 245 tokens, and 59.75 at 629 tokens.
    - Per-token sparsity does NOT imply a sparse per-request weight working set, significantly weakening the viability of naive streaming expert caches during prefill.
  - **Top-k Confound Confirmed (`norm_topk_prob=false`)**:
    - `Qwen1.5-MoE-A2.7B-Chat` sets `norm_topk_prob=false`. Mean native selected probability mass in routing traces is ~0.319.
    - Lowering $k$ simultaneously drops active experts and reduces total routed weight magnitude. Accuracy degradation cannot be attributed solely to missing expert identities.
  - **Prefix-Cache Parity Failure & bitsandbytes Shape Numerics**:
    - Prefix reuse failed 4 of 8 parity checks ($\max \Delta p = 0.119915$ vs $0.005$ tolerance; 1 question flipped its decision).
    - `bitsandbytes` 0.48.1 dispatches single-row inputs to a dedicated 4-bit matrix-vector kernel and multi-row inputs to a general path. Splitting a prefill alters batch shapes dispatched to individual experts, presenting a strong shape-dependent numerical divergence hypothesis.
  - **Physical Weight Removal (User-Reported Run `20260926T232551_755243Z`)**:
    - Physically deleting unrouted modules from `ModuleList` reduced peak GPU allocation from 7.8194 to 4.8247 GiB (a reduction of 2.9947 GiB / **-38.3%**).
    - Accuracy and NLL matched the masked allowlist arm; physical removal provided no observed execution speedup.
* **Next Experiment Protocol (Quality, Scale Controls & Cache Isolation)**:
  - **Scale & Splitting**: 252 fresh authored questions across 84 states (24 prompt-dev, 36 intervention-dev, 48 calibration, 48 policy-dev, 96 final). Each state contributes balanced 1 Yes, 1 No, 1 Unknown target.
  - **Interventions**: Skip last 1, 2, 4, or 6 MoE blocks; compare raw half-$k$ against probability-mass-rescaled half-$k$ (rescaling retained weights to match the native selected sum).
  - **Cache Diagnostics**: Independent branch storage, explicit position/attention masks, root state SHA-256 hashes, batch shape logging, and comparison against an on-demand FP32 linear reference over NF4 weights.
  - **Persistent Evidence**: Automated Google Drive sync with manifests, locks, predictions, diagnostics, and SHA-256 checksums.
  - **Execution & Follow-up**: Implemented and executed in [`30_qwen_moe_quality_and_cache_followup.ipynb`](./30_qwen_moe_quality_and_cache_followup.ipynb) yielding follow-up run `20260927T003918_481825Z`, detailed in Section 30 below.
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, comparison CSVs, and routing logs are archived on Google Drive.
  - **Reproducibility**: Qwen MoE inference, routing heatmaps, and late MoE bypass experiments can be regenerated by running [`29_qwen_moe_decision_lab.ipynb`](./29_qwen_moe_decision_lab.ipynb).
  - **Key Output Files**: `manifest.json`, `comparison.csv`, `arm_results.json`, `routing_counts.json`, `observed_expert_work.csv`, `readout_comparison.json`, `shared_prefix_parity.json`, `length_study.json`, `suffix_policy.json`, `routing_heatmap.png`, `quality_latency.png`, `length_study.png`.

---

### 30. Qwen MoE Quality and Cache Follow-Up: Controlled Interventions, Cache Numerics & Option-Order Diagnostics
* **File**: [`30_qwen_moe_quality_and_cache_followup.ipynb`](./30_qwen_moe_quality_and_cache_followup.ipynb)
* **Run ID / Session**: `20260927T003918_481825Z` (Workbench v0.1, 27 September 2026; seed 73129; profile `qwen15_moe`, model `Qwen/Qwen1.5-MoE-A2.7B-Chat` at revision `ec052fda178e241c7c443468d2fa1db6618996be`, NF4 precision with BF16 compute, eager attention)
* **Target HW**: GPU (NVIDIA L4, Linux 6.6.122+, Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.57.1, bitsandbytes 0.48.1, CUDA 12.8; allocated 7.78 GiB / peak 7.82 GiB / reserved 10.55 GiB)
* **Evidence & Provenance**:
  - Archival Location: Primary inspected run archived on Google Drive: [Drive run `20260927T003918_481825Z`](https://drive.google.com/drive/folders/1gXwnF1sXhR5Yh4-7R2izDQ9TJXDl_Lf1) (22 raw source file hashes verified against SHA-256; final metrics recomputed directly from saved predictions).
  - Prior Evidence Audit: `prior_evidence/` (SHA-256 `7168962d24ea477dc4ed078757d6339e910118fe01cc8a4ad3285e1d205dbbc5`)
  - Fresh Dataset: `fresh_decisions.jsonl` (SHA-256 `59fabd2b2ae55553e136b2d636b645466591ac5ad5fc422f06ecdac7fcb4818d`)
  - Population: 252 authored teaching questions across 84 source states (balanced 1 Yes, 1 No, 1 Unknown per state).
  - Split sizes:
    - Prompt dev: 24 questions (8 states)
    - Selection dev: 36 questions (12 states)
    - Calibration: 48 questions (16 states)
    - Policy dev: 48 questions (16 states)
    - Final test: 96 questions (32 states: 32 Yes, 32 No, 32 Unknown)
* **What it Measured / Scope**:
  - Investigates the root causes behind the latency-quality trade-offs and cache divergences discovered in Notebook 29:
    1. **Prompt & Boundary Diagnostics**: Evaluates legacy prompt vs explicit three-way prompt and verifies answer-boundary probability concentration on allowed tokens.
    2. **Router Weight Scaling Confound**: Compares raw half-$k$ ($k=2$) against probability-mass-rescaled half-$k$ (rescaling retained weights to match native selected mass) to isolate structural expert loss from weight scaling magnitude in unnormalized MoEs (`norm_topk_prob=false`).
    3. **Controlled Layer Skips**: Sweeps late-block bypasses (skipping 1, 2, 4, or 6 late MoE MLP layers).
    4. **Cache Numerics Isolation**: Evaluates prefix-cache continuation with exact attention masks, position IDs, and isolated continuation buffers across two independent backends: native bitsandbytes NF4 GEMM and an on-demand FP32 dequantized linear reference over NF4 weights.
    5. **Option Presentation Order Bias**: Measures decision stability under candidate option letter permutations.
* **Stage-by-Stage Findings & Audit Results**:
  - **Prompt Development & Answer-Boundary Audit**:
    - Evaluated legacy prompt vs `explicit_three_way` prompt on 24 prompt-dev questions.
    - `explicit_three_way` won: identical accuracy (45.83%, 11/24), but reduced NLL (2.3000 vs 3.0039) and lower Brier score (0.9079 vs 0.9733). Locked via `prompt_lock.json`.
    - Direct readout projection achieved exact zero-delta parity ($\max \Delta p = 0.0$) against full-vocabulary softmax.
    - Allowed token mass on $\{A, B, C\}$ was 98.36%–99.77% across audited examples, confirming clean token concentration at the decision boundary.
  - **Intervention Selection on Development Split**:
    - Evaluated 7 candidate configurations on 36 selection-dev questions:
      - `native`: 36.11% acc, 0.0% none-recall, NLL 2.0983, p50 2,023.9 ms
      - `skip_last_1`: 36.11% acc, 0.0% none-recall, NLL 3.0794, p50 1,948.1 ms
      - `skip_last_2`: 38.89% acc, 0.0% none-recall, NLL 3.4700, p50 1,868.4 ms
      - `skip_last_4`: 41.67% acc, 8.33% none-recall, NLL 2.6293, p50 1,700.9 ms
      - `skip_last_6`: 38.89% acc, 0.0% none-recall, NLL 2.4208, p50 1,532.6 ms (1.32x speedup)
      - `half_topk_raw`: 38.89% acc, 0.0% none-recall, NLL 2.1217, p50 1,869.3 ms
      - `half_topk_mass_matched`: 38.89% acc, 0.0% none-recall, NLL 2.0377, p50 1,871.9 ms
    - **Mass Matching Confirmed**: Rescaling retained expert weights to match native selected mass improved NLL from 2.1217 to 2.0377, confirming that weight-scaling magnitude is an active confound in unnormalized MoE top-k reduction.
    - **Candidate Lock**: `skip_last_6` selected as the fastest arm clearing the development threshold ($\le 2$ pp accuracy loss, $\ge 1.1\times$ speedup). Locked via `selection_lock.json`.
  - **Calibration & Policy Threshold Optimization**:
    - Temperature calibration fitted on 48 calibration questions:
      - `native`: inverse temperature $\beta = 0.0987$ ($T \approx 10.13$)
      - `skip_last_6`: inverse temperature $\beta = 0.0739$ ($T \approx 13.53$)
    - Review policy thresholds fitted on 48 policy-dev questions (cost ratio 0.10):
      - `native`: threshold = 0.5071
      - `skip_last_6`: threshold = 0.4720
    - Both calibration fits softened overconfidence. Zero development error on small accepted sets (5 and 4 cases) did not generalize to final test cases.
  - **Final Locked Evaluation on Fresh Test Cases (N=96, 32 States)**:

    | Arm | Correct / 96 | Accuracy | Calibrated NLL | Raw NLL | Brier | p50 (ms) | Speedup | Unknown Recall | Automatic Coverage | Accepted Error | Peak VRAM (GiB) |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | `native` | 36 | 37.50% | 1.0836 | 2.3083 | 0.6564 | 2,022.49 | 1.00x | 0/32 (0.0%) | 14.58% (14/96) | 64.29% (9/14) | 7.823 |
    | `skip_last_6` | 40 | 41.67% | 1.0909 | 2.3765 | 0.6612 | 1,535.78 | 1.32x | 2/32 (6.25%) | 8.33% (8/96) | 37.50% (3/8) | 7.823 |

    - **Severe Baseline "Yes" Bias**: The native model predicted "Yes" on **89 of 96 questions (92.7%)**, completely failing on unknown detection (0/32). Explicit three-way prompts did not restore abstention.
    - **Substantial Behavioral Shift in `skip_last_6`**: Skipping the last 6 MoE layers gained 9 correct answers and lost 5; its net accuracy gain (+4 decisions) came entirely from the policy question family.
    - **Paired Nonparametric Bootstrap** (2,000 cluster draws over 32 states):
      - Accuracy difference: **+4.17 pp** in favor of `skip_last_6` (95% CI: [0.00%, +10.42%]).
      - The candidate cleared the speed requirement (1.32× vs 1.10× floor), but failed the accuracy ($\ge 80\%$), unknown-recall ($\ge 80\%$), coverage ($\ge 20\%$), and accepted-error ($\le 10\%$) gates.
  - **Resident Memory & Expert Utilization**:
    - Peak allocated VRAM remained **7.823 GiB**; skipping layers in software leaves inactive weights resident in memory unless physically deleted.
    - In the single profiled request, **59.875 of 60 experts per layer** were touched (~99.8% active footprint). This saturating working set provides no empirical support for prefill expert streaming architectures.
  - **Cache Numerics Diagnostics (Isolated from Quality Scoring)**:
    - Repeated execution, branch reordering, and root-cache integrity all passed successfully.
    - Input continuation splitting failed:
      - `native` (bitsandbytes NF4 GEMM): 4 passed, 4 failed ($\max \Delta p = 0.119915$, 1 decision flip).
      - `nf4_fp32_linear_reference` (dequantized weights in FP32): 2 passed, 6 failed ($\max \Delta p = 0.155268$, 1 decision flip).
    - Replacing bitsandbytes 4-bit GEMM with an explicit FP32 dequantized linear reference worsened drift ($\Delta p \approx 0.155$), demonstrating that prefix cache divergence in MoE architectures involves attention position/mask state interactions and dynamic router expert assignments during prefill slicing rather than just GEMM batching differences.
  - **Option Presentation Order Bias**:
    - Tested 24 paired permutations reversing candidate letter presentations.
    - `skip_last_6` flipped semantic answers on **5 of 12 option-order checks (41.7%)**, versus **1 of 12 (8.3%)** for native (6/24 flips total / 25.0%; $\max \Delta p = 0.1686$), demonstrating that late-layer bypass amplifies presentation bias.
  - **Routing Diagnostic Implementation Limitation Identified**:
    - The notebook's routing diagnostic reconstructs routes via `topk(k+1)` sliced to `k`. Because PyTorch's `torch.topk` does not guarantee stable index ordering when router logits tie ([PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)), route reconstructions during score ties require correction before supporting root-cause attribution. The independently measured cache parity failures and final quality metrics remain valid and unimpacted.
* **Research Governance, Acceptance Gate & Recommendations**:
  - Pre-registered gate criteria: accuracy $\ge 80\%$, unknown recall $\ge 80\%$, coverage $\ge 20\%$, accepted error $\le 10\%$.
  - Realized outcome: `research_gate_passed = false`, `promotion_authorized = false`, `model_promoted = false`.
  - Confirms exploratory completion (`EXPLORATORY_COMPLETE`). No MoE checkpoint promoted to production or Jev runtime serving.
  - **Next Priorities**: Prioritize establishing a trustworthy three-way decision baseline and actual-route capture next. Keep late-layer skipping as an experimental candidate only; current findings do not support further pruning or making streaming the primary serving architecture.
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, receipts, and figures are archived on Google Drive ([Drive run `20260927T003918_481825Z`](https://drive.google.com/drive/folders/1gXwnF1sXhR5Yh4-7R2izDQ9TJXDl_Lf1)).
  - **Reproducibility**: Intervention selections, calibration thresholds, and option-order diagnostics can be regenerated by running [`30_qwen_moe_quality_and_cache_followup.ipynb`](./30_qwen_moe_quality_and_cache_followup.ipynb).
  - **Key Output Files**: `RUN_SUMMARY.md`, `manifest.json`, `summary.json`, `final_results.csv`, `final_results.json`, `development_results.json`, `prompt_development.json`, `prompt_lock.json`, `selection_lock.json`, `cache_mode_summary.json`, `cache_numerics.json`, `option_order_diagnostic.json`, `expert_batch_shapes.csv`, `fresh_quality_and_coverage.png`, `cache_route_changes.png`.

---

### 31. Phase 4 Prefill Speed and Accuracy Lab: Dense vs MoE Serving & Cache Qualification
* **File**: [`31_qwen_prefill_speed_accuracy_lab.ipynb`](./31_qwen_prefill_speed_accuracy_lab.ipynb)
* **Run Key**: `e15e1e9f7a64e464a38354c59b0c79805d59bc13d517f7e4fca66873e5d5ff2e`
* **Target Hardware & Environment**: NVIDIA A100-SXM4-40GB (CUDA 13.0, PyTorch 2.13.0+cu130, vLLM 0.30.0, Transformers 5.17.0)
* **Evaluated Models**:
  - `qwen35_4b`: `Qwen/Qwen3.5-4B` (dense BF16, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`)
  - `qwen35_moe_int4`: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (MoE INT4 GPTQ/Marlin, revision `3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b`)
* **What it Measured / Scope**:
  - Tested modern Qwen MoE against dense Qwen control using the same vLLM serving engine (v0.30.0).
  - Adapted PrivateMode's constrained option-logprob readout method (`number_prefill`).
  - Evaluated decision quality across SNLI (288 questions) and synthetic policy datasets (288 questions).
  - Evaluated exact shared-state caching, sequential repeat stability, concurrent request invariance, and presentation order stability.
* **Stage-by-Stage Findings & Audit Results**:
  - **Evidence Inspection & Governance Outcome**:
    - Inspected the [completed run](https://drive.google.com/drive/folders/1LIKE7JSmEcO2fvrhf8Qx4_ZJfv-heGwD). The saved evidence explains the failures.
    - Model and cache promotion: **`false`** (`research_gate_passed = false`). A focused follow-up is needed to validate fixes.
  - **Cache Tests Were Too Short to Exercise Caching**:

    | Measurement | Qwen3.5 4B | MoE INT4 |
    |---|---|---|
    | Runtime cache-block size | 528 tokens | 1,056 tokens |
    | Tested shared prefixes | 59–105 tokens | 59–105 tokens |
    | Reported reused tokens | 0 throughout | 0 throughout |

    - Both engines fell back from Mamba cache mode `all` to `align` (`WARNING: Hybrid or mamba-based model detected without support for prefix caching with Mamba cache 'all' mode: falling back to 'align' mode`). That mode resumes cached computation at block boundaries; these prefixes never reached one. This matches vLLM's documented behavior ([vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.30.0/features/automatic_prefix_caching/?utm_source=chatgpt.com)).
    - That exposes a flaw in test design: failure on these short examples disabled cached modes in the later, longer-input benchmark. We therefore haven't established whether caching works correctly on sufficiently long inputs.
  - **Probability Drift Even When Requests Report Zero Cache Reuse**:
    - The allowed maximum difference was 0.5 percentage points ($\le 0.005$):

    | Check | 4B maximum difference | MoE maximum difference |
    |---|---|---|
    | Repeated sequential requests | 0.00 pp | 17.60 pp |
    | Concurrent requests versus sequential | 3.28 pp | 11.92 pp |

    - The 4B concurrent test changed 1 of 24 answers.
    - MoE sequential-repeat checks failed in 6 of 8 groups, despite unchanged winning answers.
    - MoE staged execution also changed an answer and an acceptance decision.
    - The server logs identify the MoE backend as GPTQ/Marlin (`Using 'MARLIN' WNA16 MoE backend`, `MarlinExperts`), but they don’t isolate whether the instability comes from that backend, another kernel, or execution scheduling. Calling this “cache corruption” would be premature.
  - **Next Experiment Recommendations**:
    - Establish repeatability with caching disabled, comparing compiled and eager execution.
    - Test batching separately.
    - Test cache reuse with prefixes exceeding the observed block boundaries (528 and 1,056 tokens), confirming actual hits before measuring speed.
    - **Correction to Earlier Setup**: vLLM 0.30 documents batch-invariance support for compute capability 8.0+, including A100. The earlier $\ge 9.0$ guard was too restrictive. That setting deserves a controlled test, though this exact quantized model/backend still needs verification ([vLLM Batch Invariance](https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/?utm_source=chatgpt.com)).
    - The completed accuracy results remain recorded observations. We can retain them and run a smaller cache/repeatability experiment rather than repeat the entire quality study.
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory and server logs are archived on Google Drive: [Completed Run `1LIKE7JSmEcO2fvrhf8Qx4_ZJfv-heGwD`](https://drive.google.com/drive/folders/1LIKE7JSmEcO2fvrhf8Qx4_ZJfv-heGwD).
  - **Reproducibility**: Prefill speed benchmarks, cache qualification sweeps, and repeatability diagnostics can be regenerated by running [`31_qwen_prefill_speed_accuracy_lab.ipynb`](./31_qwen_prefill_speed_accuracy_lab.ipynb).
  - **Key Output Files**: `RUN_SUMMARY.md`, `manifest.json`, `summary.json`, `quality.csv`, `performance.csv`, `accuracy.png`, `quality_latency.png`, `qwen35_4b_cache_qualification.json`, `qwen35_4b_server.log`, `qwen35_moe_int4_cache_qualification.json`, `qwen35_moe_int4_server.log`, `dataset.json`, `SHA256SUMS`.

---

### 32. Phase 4 Qwen Cache & Native Decisions Lab: Microbatching Repeatability, Boundary Sweep & Native Trees
* **File**: [`32_qwen_cache_and_native_decisions_lab.ipynb`](./32_qwen_cache_and_native_decisions_lab.ipynb)
* **Run Key**: `7047c6b31436f8e9b5aa85a5dad9ea4378d16eaa0912ebba288fae273c7e12ae`
* **Status**: **`PARTIAL`** (A completed study may contain failed scientific gates. No model, probability policy or cache path is promoted.)
* **Target Hardware & Environment**: NVIDIA A100-SXM4-40GB (CUDA 13.0, PyTorch 2.13.0+cu130, vLLM 0.30.0, Transformers 5.17.0, Ninja, custom llama.cpp `parallel-decision` build commit `ad129b08d9f134cd298d1f8a85efc52b1b66e18e`)
* **Evaluated Models & Profiles**:
  - `qwen35_4b`: `Qwen/Qwen3.5-4B` (dense BF16, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`)
  - `qwen35_moe_int4`: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (MoE INT4 GPTQ/Marlin, revision `3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b`)
  - `native llama.cpp`: `bartowski/Qwen_Qwen3.5-4B-GGUF` (Q4_K_M, revision `4168f45a16a1290d65a4ec0fa312ae917a4c15d6`)
* **What it Measured / Scope**:
  - Focused follow-up to Notebook 31 separating cache reuse from numerical repeatability.
  - **Repeatability Evaluation**: Measured serial vs concurrent (4-worker) request stability on cold namespaces with caching flags enabled across both models.
  - **Controlled Prefix Boundary Sweep**: Systematically tested prefixes spanning block boundaries:
    - 4B (block size 528): 527, 528, 529, 1,056, 1,057 tokens.
    - MoE INT4 (block size 1,056): 1,055, 1,056, 1,057, 2,112, 2,113 tokens.
    - Evaluated across 4 modes per length: `cold_serial`, `cold_concurrent`, `cold_staged`, `warm_prefix`.
  - **Batch-Invariance Mode Probe**: Evaluated vLLM 0.30's `--batch-invariant` flag on Compute Capability 8.0 (A100).
  - **Native llama.cpp Parallel Decision Branch**: Built and evaluated native branching with reserved sequence slots (1 vs 24 sequences) and padded vs unpadded inputs.
* **Stage-by-Stage Findings & Audit Results**:
  - **vLLM Repeatability (1/4 Rows Passed)**:

    | Arm | Mode | N | Repeats | Passed | Max $\Delta p$ | Decision Flips | Action Flips | Cohort p50 (ms) |
    |---|---|---|---|---|---|---|---|---|
    | `qwen35_4b__standard` | serial | 24 | 5 | **True** | **0.00 pp** | 0 | 0 | 1,518.6 ms |
    | `qwen35_4b__standard` | concurrent4 | 24 | 5 | **False** | **3.69 pp** | 4 | 5 | 775.0 ms |
    | `qwen35_moe_int4__standard` | serial | 24 | 5 | **False** | **13.60 pp** | 5 | 4 | 3,099.2 ms |
    | `qwen35_moe_int4__standard` | concurrent4 | 24 | 5 | **False** | **29.81 pp** | 3 | 4 | 1,533.2 ms |

    - Dense 4B achieved bitwise identical repeatability under serial execution ($\max \Delta p = 0.0$).
    - Client concurrency in 4B broke repeatability ($\max \Delta p = 3.69$ pp, 4 decision flips) because client concurrency alters GPU dynamic microbatching. Client concurrency does not fix the GPU microbatch.
    - MoE INT4 failed repeatability even in serial mode ($\max \Delta p = 13.60$ pp, 5 decision flips), worsening to 29.81 pp under concurrency. This confirms that MoE INT4 instability is intrinsic to kernel execution/scheduling rather than client concurrency alone.
  - **Empirical Verification of Mamba/Hybrid Block Boundaries (6/40 Rows Qualified)**:
    - Below block boundaries (527 tokens on 4B, 1,055 tokens on MoE INT4): **0 cached tokens reported**. Negative controls confirmed.
    - Exactly at block boundaries (528 tokens on 4B, 1,056 tokens on MoE INT4): **0 cached tokens reported** on warm prefix.
    - Exceeding block boundary by 1 token:
      - 4B at 529 tokens: `warm_prefix` reported **528 cached tokens**! `cold_staged` achieved **`QUALIFIED_MEASUREMENT`** status with $\max \Delta p = 4.37 \times 10^{-8}$ and a **1.92x speedup** (248.95 ms vs 478.44 ms baseline).
      - 4B at 1,057 tokens: `warm_prefix` reported **1,056 cached tokens** (2 blocks).
      - MoE INT4 at 1,057 tokens: `warm_prefix` reported **1,056 cached tokens** (1 block).
      - MoE INT4 at 2,113 tokens: `warm_prefix` reported **2,112 cached tokens** (2 blocks).
    - MoE INT4 failed all cache qualifications because its baseline sequential execution is unstable (`BASELINE_UNSTABLE_OR_NOT_COLD`, $\Delta p > 0.005$).
  - **Capability Errors & Engine Guardrails**:
    - `batch_invariant` engine launch: Both 4B and MoE INT4 failed to start with `CapabilityError: Engine exited` during `AsyncLLM.from_vllm_config` initialization on this vLLM 0.30 container.
    - Native `llama.cpp` parallel decision: Built successfully on CPU/A100, but failed runtime qualification guard with `CapabilityError: Full GPU layer offload not verified in native engine log`.
  - **Research Governance Outcome**:
    - `model_promoted = false`, `cache_promoted = false`.
    - No model, probability policy, or cache path is promoted.
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, boundary logs, and repeatability CSVs are archived on Google Drive.
  - **Reproducibility**: Microbatching repeatability, prefix boundary sweeps, and native llama.cpp builds can be regenerated by running [`32_qwen_cache_and_native_decisions_lab.ipynb`](./32_qwen_cache_and_native_decisions_lab.ipynb).
  - **Key Output Files**: `RUN_SUMMARY.md`, `manifest.json`, `summary.json`, `repeatability.csv`, `cache_boundaries.csv`, `vllm_replay_quality.csv`, `vllm_vs_standard.csv`, `runtime_memory.csv`, `errors.json`, `llama_build_identity.json`, `llama_build.log`, `repeatability.png`, `cache_latency.png`, `dataset.json`, `SHA256SUMS`.

---

### 33. Phase 4 Qwen Readout Codes, Deterministic Rule Composition, and Request History Lab
* **File**: [`33_qwen_readout_rules_history_lab.ipynb`](./33_qwen_readout_rules_history_lab.ipynb)
* **Run Key**: `ff6fd4996c079f53dee448296fb55a529d18418d8d849413a2e1ed0e42f6308a` (`20260927T192845_426758Z`)
* **Status**: **`EXPLORATORY_COMPLETE`** (All primary panels complete; `model_promoted = false`, `cache_promoted = false`.)
* **Target Hardware & Environment**: NVIDIA A100-SXM4-40GB (CUDA 12.8, Driver 580.82.07, Python 3.13.15, llama.cpp `parallel-decision` commit `ad129b08d9f134cd298d1f8a85efc52b1b66e18e`)
* **Evaluated Models & Profiles**:
  - `Q4_K_M`: `bartowski/Qwen_Qwen3.5-4B-GGUF` (revision `4168f45a`, SHA256 `13c16f42...a983`, 3.01 GB)
  - `BF16`: `bartowski/Qwen_Qwen3.5-4B-GGUF` (revision `4168f45a`, SHA256 `714270d4...ee5`, 8.67 GB)
* **What it Measured / Scope**:
  - Controlled comparison of single-token integer codes vs natural label readouts across 96 authored decision cases (2 technical repeats, 192 observations).
  - Deterministic host action composition (deriving action from eligibility in code vs independent model prediction) across 6-field and 5-field paths.
  - Exact prefix cache reuse across sequence reservation sizes (24 vs 3 sequences) and batch contexts (1 vs 4 contexts).
  - Request history and process state leakage: evaluating whether prior requests (JSON generation, unrelated queries, noop) induce probability drift in subsequent native decisions.
* **Stage-by-Stage Findings & Audit Results**:
  - **Readout & Rule Composition**:
    - Deriving action deterministically (5-field + rule) completely eliminated the 28.6% (Q4) / 31.2% (BF16) inconsistency rate between eligibility and action.
    - Field accuracy improved from 78.47% to 82.55% (Q4 natural) and 78.56% to 83.51% (BF16 natural), with all-fields accuracy rising from 15.10% to 23.44% (Q4) and 16.67% to 23.96% (BF16).
    - Aliases / integer codes reduced NLL (0.435 vs 0.934 on Q4) and improved field accuracy (83.25% vs 78.47%).
    - Compact JSON achieved higher all-fields accuracy (64.58% Q4, 56.25% BF16) but required ~4.5–7x higher latency (495 ms vs 108 ms).
  - **Cache Parity**:
    - All 8 full-panel cache conditions passed parity with **$\max \Delta p = 0.0$** and 0 field flips across 960 requests.
    - Qualified speedups: 1-context Q4 showed 2.75x speedup (47.3 ms warm vs 130.0 ms cold); 4-context Q4 showed 1.54x speedup (153.9 ms vs 237.2 ms).
    - BF16 showed 2.29x speedup on 1-context (40.5 ms warm vs 92.8 ms cold) and 1.40x on 4-context (130.1 ms vs 182.4 ms).
  - **Request History & State Leakage**:
    - Isolated native calls and noop interventions passed with $\Delta p = 0.0$.
    - However, interleaved JSON generation and tight sequence reservations (3 sequences with 4 contexts) introduced minor drift ($\Delta p \le 0.0040$), signaling that process state and memory layout can subtly interact with continuation state.
* **Evidence & Artifacts**:
  - **Archival Location**: Complete raw run directory, receipts, and charts are archived on Google Drive: [Drive run `20260927T192845_426758Z`](https://drive.google.com/drive/folders/1N8fq_wSct874PWi-VPwuGWAKiYL39xUM).
  - **Reproducibility**: Single-token readout codes, deterministic rule composition, and request history evaluations can be regenerated by running [`33_qwen_readout_rules_history_lab.ipynb`](./33_qwen_readout_rules_history_lab.ipynb).
  - **Key Output Files**: `REPORT.md`, `manifest.json`, `setup.json`, `quality_summary.csv`, `cache_summary.csv`, `history_summary.csv`, `measurement_gates.json`, `quality_latency.png`, `cache_timings.png`, `history_drift.png`.

---

### 34. Phase 4 Qwen 9B T4/L4 Open Questions Lab: Rule Composition, Prefix Cache & Serving History
* **File**: [`34_qwen35_9b_t4_l4_open_questions_lab.ipynb`](./34_qwen35_9b_t4_l4_open_questions_lab.ipynb)
* **Run ID**: `20260928T220142_110595Z_1192c8`
* **Status**: **`EXPLORATORY_EVALUATED`** (Reconciled across 39 planned blocks + 3 replay blocks; research gates distinguish outcomes; complete service not qualified due to JSON history leakage and semantic readout errors; `model_promoted = false`, `cache_promoted = false`.)
* **Target Hardware & Environment**: NVIDIA Tesla T4 (15.0 GiB reported VRAM, CUDA runtime, native engine commit `ad129b08d9f134cd298d1f8a85efc52b1b66e18e`)
* **Evaluated Model**: Qwen3.5-9B Q4 (Q4_K_M, 33 stored blocks, 1 auxiliary NextN/MTP block disabled, 32 backbone blocks, hidden size 4096, 34/34 layers offloaded to GPU)
* **What it Measured / Scope**:
  - Evaluation of Qwen3.5-9B Q4 feasibility on commodity 16GB GPU (Tesla T4).
  - Controlled 5-arm comparison across two panels: previously exposed continuity panel (96 cases, 2 repeats) and new synthetic diagnostic panel (36 cases, 2 repeats):
    - **Arm A**: Six native model-predicted fields; prefix lookup disabled (recomputed prefix).
    - **Arm B**: Five native fields + host-derived action; prefix lookup disabled.
    - **Arm C**: Six native model-predicted fields; verified warm prefix.
    - **Arm D**: Five native fields + host-derived action; verified warm prefix.
    - **Arm E**: Generated compact JSON (separate quality sessions).
  - Genuine first-use cold-start traces measuring cumulative time from process startup.
  - Dedicated cache telemetry and mathematical parity across 330 paired requests.
  - Process history stability: native vs JSON-interleaved execution, reset efficacy, and mitigation attempts (padding, CUDA graphs).
  - Sequence reservation sensitivity (8 vs 3 sequences) and confidence / risk-coverage gating.
  - Diagnostic separate-field requests and offline deterministic route replay.
* **Stage-by-Stage Findings & Audit Results**:
  - **T4 Feasibility Confirmed**:
    - Qwen3.5-9B Q4 fits comfortably within T4 VRAM: 6.27 GiB whole-device baseline usage, 6.30 GiB sampled peak usage under 4-context Arm D workload, with 34/34 layers offloaded.
  - **Combined Rule & Cache Path (Arm D)**:
    - **Speedup**: Arm D achieves **339.38 ms median request latency** on 96 cases (2.35x faster than Arm A at 798.68 ms, and 11.5x faster than Arm E JSON at 3,910.68 ms). Prefix reuse drives a 56% latency reduction over Arm B (763.67 ms). On the 36-case panel, Arm D hits 352.47 ms (2.29x over A, 55% over B).
    - **Accuracy**: Arm D improves field accuracy over Arm A by +5.38 pp (86.11% vs 80.73%, 95% CI: [3.82, 7.12]) on continuity, and +2.78 pp (86.11% vs 83.33%, 95% CI: [0.93, 5.09]) on synthetic. All-six accuracy increases from 22.92% to 32.29% (continuity) and 30.56% to 33.33% (synthetic).
  - **Deterministic Action Composition**:
    - Solves a severe model contradiction: In Arm A, eligibility is correct in 94/96 cases, but the independently predicted action is wrong in 27/96 cases.
    - Arms B and D derive action deterministically (`eligible -> grant access`, `ineligible -> deny access`, `undetermined -> request missing information`), completely eliminating contradictions (0/96) and reducing mean NLL from 0.900 to 0.655 (continuity) and 0.731 to 0.624 (synthetic). Action probability distribution is carried forward directly from eligibility.
  - **Concentrated Semantic Failure Modes in Arm D**:
    - Arm D achieves 100% accuracy on eligibility, action, retries, and urgency. Every single remaining error belongs to only two systematic failure modes:
      1. *Plain "closed" becomes "closed duplicate"*: All 36 plain-closed cases on continuity and all 12 on synthetic are predicted as duplicate (0% plain-closed recall).
      2. *Routing always appends "priority" suffix*: All 44 nonurgent continuity and 18 nonurgent synthetic cases receive the priority suffix despite the separate urgency field being correctly predicted as false.
    - *Deterministic Routing Replay*: Post-hoc CPU replay deriving route priority from predicted urgency (`route = base_route + (" priority" if predicted_urgent else "")`) eliminated all routing errors, boosting D field accuracy to **93.75%** (continuity) / **94.44%** (synthetic) and all-six accuracy to **62.50%** / **66.67%**, surpassing generated JSON.
    - *Readout Diagnostics*: Separate field-only requests on 24 calibration cases raised route accuracy from 50.0% to 87.5% and case-state from 66.7% to 100.0%, proving that the underlying model has the representational capability if prompted/formatted without catalogue interference.
  - **Exact Prefix Cache Parity & First-Use Accounting**:
    - Reconstructed 330 dedicated paired requests: **$\max \Delta p = 0.0$** and 0 flips across 5/6 fields and 1/4 contexts. Dedicated cache ratio: 2.27x for 1-context (772.70 ms -> 340.69 ms), 1.31x for 4-context (1,780.29 ms -> 1,358.72 ms).
    - Fresh-process cold trace (Arm D): First request takes 881 ms (~4.98 s including process startup/warmup); cumulative mean per request drops to 611 ms (2 reqs), 468 ms (4 reqs), 395 ms (8 reqs), and 360 ms (16 reqs; 5.75 s execution, ~9.85 s total). Warm savings withstand true cold-start accounting.
  - **Serving Instability: JSON Interleaving**:
    - Native-only controls pass history checks ($\Delta p = 0.0$).
    - Interleaving JSON generation in the same process causes significant probability drift ($\max \Delta p = 0.080$ to $0.095$) and flips up to 4 action decisions in 4-context runs. Session end vs fresh reset fails on 64/90 checks.
    - Disabling padding ($\max \Delta p = 0.067$) or requesting CUDA graphs disabled ($\max \Delta p = 0.080$) failed to remediate the drift. Native and JSON execution require process separation.
  - **Sequence Reservation Constraint**:
    - 8 reserved sequences preserve probabilities exactly ($\Delta p = 0.0$) when expanding from 1 to 4 contexts.
    - Dropping reservation to 3 sequences causes numerical drift ($\Delta p = 0.0589$) and fails numerical gates. The 8-sequence profile must be retained.
  - **Confidence & Risk-Coverage Limitations**:
    - Filtering by minimum predicted probability threshold (0.90, 0.95) does not filter out errors because the model is overconfident on its systematic semantic traps (at 0.95 threshold, 30.0% of accepted continuity and 77.78% of synthetic cases remain wrong). Rejection thresholds cannot substitute for architectural repair.
* **Supporting Review Artifacts**:
  - Review Document: `OpenKind_9B_T4_Results_Review_20260928.md`
  - Reconciliation Archive: `OpenKind_9B_T4_Review_and_Reconciliation_20260928.zip` (Run: `20260928T220142_110595Z_1192c8`)
  - Recorded Evidence Gates: [measurement_gates](https://drive.google.com/file/d/1CIvInA1CTO83ST5BnOWn8XCm1rTr8jm7/view), [COMPLETE](https://drive.google.com/file/d/1Q-t2VhbH7qYUbkDodlg4IPgpNK8ySBa3/view), [quality_summary](https://drive.google.com/file/d/1qyA-UtigDfyThVsjT4SUqjP85lfJyNhb/view), [history_summary](https://drive.google.com/file/d/1U5oYFS1cr4UMgwLqPp7NzUauOb0-Te8l/view)

---

### 35. Local Decision Training: Mixed-Task Qwen3.5-4B Decision LoRA Trainer
* **File**: [`35_local_decision_training.ipynb`](./35_local_decision_training.ipynb)
* **Goal**: Self-contained Colab trainer for a mixed-task Qwen3.5-4B decision LoRA targeting A100 with an L4 NF4 fallback path.
* **Scope & Methodology**:
  - Combines MultiNLI, BoolQ, Banking77, MultiRC, optional SST-5, precomputed Qwen-teacher decisions, and exact rule examples. Historical final splits and QASPER stay out of this trainer.
  - Enforces separate development, calibration-fit, calibration-gate, and reserved evaluation groups without reopening historical final splits.
  - Implements PEFT rank-16 LoRA adapters with `SDPBackend.MATH` attention consistency.
  - Version 4 defaults to plain CE and atomic sampling. Optional reasoning data, occurrence-based presentation augmentation and the six-arm loss sweep isolate interventions. E43/E44 motivate group-aware family/class and complete-request diagnostics. Frozen evaluation verifies the exported implementation, adapter, tokenizer and calibration. These mechanisms establish no measured 4B quality gain.
  - TypeSafe data is benchmark-only: a separate frozen-export cell reads five pinned test snapshots, compares the parent, and records coverage and question-level agreement. It does not train or select on those labels or claim official workflow action scores. CLEF-flash's released settings guide the architecture comparison, not the loss coefficients.
* **Status**: Authored and locally checked with tiny models; full 4B CUDA training and Mac qualification remain unrun.
* **Supporting Directory**: [`local_decision_training/`](./local_decision_training)

---

### 36. OpenKind Unified Decision Validity Lab: Selective Readouts, Process Isolation, Scaling, and Compact-Model Transfer
* **File**: [`36_openkind_unified_decision_validity_lab.ipynb`](./36_openkind_unified_decision_validity_lab.ipynb)
* **Run ID**: `20260929T194506_694066Z_dc1f74`
* **Protocol**: `openkind-unified-decisions/v3.0.0`
* **Run Date**: 29 September 2026
* **Target Hardware & Environment**:
  - NVIDIA Tesla T4 16GB (15.0 GiB reported VRAM, CUDA runtime, native engine commit with Qwen3.5-9B publisher Q4_K_M, 34/34 layers offloaded).
  - CPU Comparator: Go 1.27.1 built with SIMD and active assembly on host AVX2/FMA, 1 configured worker, backbone `hotchpotch/bekko-embedding-v1-a8m` (`c721113...`, ~212 MB weights).
* **Evaluated Models**: Qwen3.5-9B Q4_K_M (dense decoder); Indecis / Bekko Embedding v1 a8m (compact Go CPU encoder).
* **Status & Disposition**: **`RETAIN_BOUNDED_POSITIVE_RESULTS`** (Reconciled across 80 completed block receipts plus 1 explicitly unexecuted INT8 capability record; 0 execution failures; arithmetic and artifact verification confirmed; no model or service promotion).
* **What it Measured / Scope**:
  - Comprehensive resolution of open questions from the 9B/T4 exploratory study: routing dependency rules, selective and grouped readouts, joint output probabilities, resident process isolation vs restarts, actual Q/K token scaling, and compact CPU encoder (indecis) alternatives.
  - Primary evaluation comprises 14,280 technical quality requests over 660 unique evaluated inputs across 106 arm-by-panel rows, repeated twice. Dataset includes 1,860 records spanning fitting, development, calibration-fit, calibration-gate, and fresh confirmation partitions.
  - Evaluated seven primary arms across fresh-template confirmation (96 cases × 2 repeats):
    - **Arm D**: Five natural fields; action derived from eligibility; static-prefix reuse (baseline).
    - **Arm R**: D plus deterministic route-priority composition from predicted urgency.
    - **Arm X**: Selective indexing of route and case state; natural labels retained for eligibility, next_action, retries, urgent; derived action rule.
    - **Arm G**: Grouped requests: (1) eligibility/retries/urgency, (2) route separately, (3) case state separately; derived action rule.
    - **Arm GR**: Grouped requests (G) plus deterministic route-priority composition.
    - **Arm J**: Indexed formulation with 4-way joint region/urgency outcome; route and urgency pushed forward from joint distribution.
    - **Arm E**: Compact JSON generated in an isolated quality session.
  - Serving stability & process isolation: mixed-process vs native-only control vs two resident processes vs restarting native after each JSON request across two trace shapes (six-field 1-context, D-style 4-context).
  - Schema sensitivity: 4 diagnostic transformations (natural-option list reordering, semantic code remapping, adding an unrelated question, renaming visible field keys) across 12 development cases (48 comparisons).
  - Dedicated prefix caching & genuine first-use traces: 132 paired cache requests and 16-request cold traces.
  - Multi-question scaling: Native Qwen catalogue scaling ($Q \in \{1, 4, 8, 16\}$, $K \in \{2, 4, 8, 16\}$, state tokens up to 4,106) vs Indecis open-option scaling ($Q \in \{1, 4, 8, 16\}$ with candidate embedding cache).
  - Indecis (Go on CPU): Fixed head vs open-mode fitting (1,152 examples, 3 epochs) vs replay continuation (384 exposures), CPU AVX2 assembly parity (1,320 pairs) vs NOASM, memory footprint (RSS).
  - Calibration & confidence policies: temperature scaling gate on held-out confirmation across tasks; empirical decision policies with cost trade-offs.

* **Stage-by-Stage Findings & Audit Results**:

  #### 1. Fresh-Template Confirmation (96 cases × 2 technical repeats)
  | Arm | Complete Request Computation | Probability Contract | Field Accuracy | All Six Correct | Request p50 (ms) | Request p95 (ms) | Exact Marginal Dists |
  |---|---|---|---|---|---|---|---|
  | **D** | 5 natural fields; derived action; static prefix | 6 marginals available | 82.64% | 23.96% | 306.92 | 314.91 | 6/6 |
  | **R** | D + route-priority composition from predicted urgency | 5 marginals (route withheld) | 91.67% | 54.17% | 306.72 | 315.66 | 5/6 |
  | **X** | Index route and case state; retain others; derived action | 6 marginals available | 92.19% | 60.42% | 222.22 | 229.75 | 6/6 |
  | **G** | Grouped: (1) elig/retries/urgent, (2) route, (3) state | 6 marginals available | 93.23% | 73.96% | 1,422.24 | 1,437.59 | 6/6 |
  | **GR** | G + route-priority composition | 5 marginals (route withheld) | 95.49% | 83.33% | 1,419.57 | 1,438.18 | 5/6 |
  | **J** | Joint region/urgency source $\to$ pushforward route/urgent | 6 marginals available | 90.28% | 57.29% | 209.54 | 216.76 | 6/6 |
  | **E** | Compact JSON in isolated quality session | Point answers only | 93.75% | 69.79% | 2,930.54 | 3,012.36 | 0/6 |

  - **Selective Indexing (Arm X)**: Surgically indexing route and case state while keeping natural labels for the other fields delivers **+9.55 pp field accuracy** (92.19% vs 82.64%, 95% CI: [6.08, 12.85]) and **+36.46 pp all-six correctness** (60.42% vs 23.96%, 95% CI: [20.83, 51.07]). Median request time drops **27.60%** (306.92 ms $\to$ 222.22 ms) and full-probability NLL plummets from 0.7512 to 0.2270. Telemetry confirms X requires only 1 scoring decode call per request versus 2 for D.
  - **Routing Composition (Arm R & GR)**: Deriving route priority deterministically from predicted urgency raises fresh field accuracy by **+9.03 pp** (82.64% $\to$ 91.67%) with virtually identical latency (306.72 ms vs 306.92 ms). Grouped routing (GR) achieves **83.33% all-six correctness** at 1,419.57 ms (2.06x faster than JSON at 2,930.54 ms / 69.79%).
  - **Grouped Requests (Arm G & GR)**: Grouping fields eliminates catalogue crosstalk and boosts quality (73.96% all-six on G, 83.33% on GR), but executes 3 sequential subrequests costing ~1.42 s. Alternating catalogues currently yield 0 prefix cache hits; catalogue-resident caching is required for future speedup.
  - **Joint Outputs (Arm J)**: Pushforward from a 4-way joint region/urgency distribution guarantees consistent marginals at 209.54 ms p50, but retries accuracy drops from 100% to 72.92%.

  #### 2. Per-Field Error Distribution on Fresh Confirmation
  | Arm | `case_state` | `eligibility` | `next_action` | `retries` | `route` | `urgent` |
  |---|---|---|---|---|---|---|
  | **D** | 58.33% | 96.88% | 96.88% | 100.00% | 43.75% | 100.00% |
  | **R** | 58.33% | 96.88% | 96.88% | 100.00% | 97.92% | 100.00% |
  | **X** | 100.00% | 95.83% | 95.83% | 100.00% | 61.46% | 100.00% |
  | **G** | 95.83% | 92.71% | 92.71% | 100.00% | 78.12% | 100.00% |
  | **GR** | 95.83% | 92.71% | 92.71% | 100.00% | 91.67% | 100.00% |
  | **J** | 89.58% | 90.62% | 90.62% | 72.92% | 98.96% | 98.96% |
  | **E** | 100.00% | 95.83% | 95.83% | 100.00% | 70.83% | 100.00% |

  #### 3. Mixed-Request Serving Stability & Process Isolation
  | Treatment | 6-Field / 1-Context Trace | D-Style / 4-Context Trace | Probability & Answer Gate Verdict |
  |---|---|---|---|
  | **Mixed process** | 84.72 s | 104.45 s | **Fails**: $\max \Delta p = 0.080189 / 0.094815$; 0 / 3 field flips |
  | **Native-only control** | 50.48 s | 70.83 s | **Passes**: exact recorded probabilities & answers ($\Delta p = 0$) |
  | **Separate resident processes** | 89.64 s (+5.80%) | 108.53 s (+3.91%) | **Passes**: exact recorded probabilities & answers ($\Delta p = 0$) |
  | **Restart after JSON** | 305.11 s (3.60x) | 410.28 s (3.93x) | **Passes**: exact recorded probabilities & answers ($\Delta p = 0$) |

  - **Resident Process Isolation Validated**: Running native decisions and JSON generation in separate resident processes completely eliminates history contamination with only a **3.9%–5.8% trace time overhead**.
  - **Memory Footprint on T4**: Both processes reside simultaneously on Tesla T4 with full 34/34 GPU offload, consuming **12.10 GiB total device VRAM** (2.90 GiB free). In contrast, in-process restarting incurs a 3.6x–3.9x slowdown due to repeated process initialization (mean recovery cost: 13.67–19.06 s).

  #### 4. Schema Sensitivity Beyond Process Isolation
  | Transformation (12 dev cases) | Numerical Gate Failures | Maximum $\Delta p$ | Field Flips | Finding |
  |---|---|---|---|---|
  | **Natural-option list reorder** | 12 / 12 | 0.075006 | 0 | Probability drift without discrete decision flips |
  | **Semantic code remapping** | 12 / 12 | 0.912310 | 13 | Extreme probability disruption; some accuracy improvements |
  | **Add unrelated question** | 12 / 12 | 0.251380 | 2 | Flips 2 fields while aggregate score remains identical |
  | **Rename visible field keys** | 12 / 12 | 0.444171 | 4 | Significant sensitivity to visible prompt identifiers |

  - All 48 comparisons fail the strict numerical invariance gate. Process isolation eliminates runtime execution leakage, but model conditioning remains sensitive to prompt schema syntax, demonstrating that arbitrary-question independence cannot be assumed.

  #### 5. Cache Parity and First-Use Telemetry
  - **132 Dedicated D-Style Cache Pairs**: Exact numerical parity verified ($\Delta p = 0.0$). Prefix caching delivers a **2.34x speedup** on 1-context requests (686.09 ms cold $\to$ 292.83 ms warm).
  - **Fresh-Process First-Use Trace**: First useful request requires 782.86 ms (4.82 s total milestone including startup/bookkeeping). Subsequent requests average 309.05 ms, proving genuine amortization.

  #### 6. Multi-Question Scaling: Native Catalogue vs Indecis Open-Option
  - **Qwen Native Catalogue Scaling** ($K=4$, ~256 state tokens): $Q=1$ is 447.32 ms, $Q=4$ is 459.61 ms, $Q=8$ is 549.59 ms, and $Q=16$ is 701.66 ms. Scaling 16 questions costs only **1.57x** a single question, confirming strong native sub-linear amortization.
  - **Context-Length Dynamic State Boundary**: At $Q=4, K=4$ with ~4,106 state tokens, fresh prefix takes 5,694.38 ms vs 5,395.76 ms warm: static catalogue caching does not amortize dynamic state prefill.
  - **Indecis Open-Option Scaling** ($K=4$ with option embedding cache): $Q=1$ is 17.60 ms, $Q=4$ is 75.16 ms, $Q=8$ is 138.76 ms, and $Q=16$ is 281.78 ms. Scaling is strictly linear (**16.01x ratio**), as open mode re-encodes the input per question.

  #### 7. Indecis (Go on CPU) Confirmation & Transfer Breakdown
  | Arm | Policy Field Acc | Policy All-Six | Policy p50 (ms) | SNLI Accuracy | SNLI p50 (ms) |
  |---|---|---|---|---|---|
  | **`I_OPEN_FROZEN`** | 37.85% | 1.04% | 137.16 | 27.08% | 23.41 |
  | **`I_OPEN_FIT`** | 58.85% | 16.67% | 142.57 | 30.21% | 17.68 |
  | **`I_FIXED`** | 69.10% | 22.92% | 46.28 | 39.58% | 13.62 |
  | **`I_POLICY_ONLY`**| 70.14% | 30.21% | 43.52 | 39.58% | 13.72 |
  | **`I_REPLAY`** | 70.31% | 26.04% | 31.54 | 44.79% | 18.58 |
  | *(Qwen Reference)* | *92.19% (X)* | *60.42% (X)* | *222.22 (X)* | *92.71% (NLI_CODE)* | *152.88* |

  - **Severe Transfer Degradation**: `I_FIXED` drops from 91.15% field / 73.44% all-six on dev to 69.10% / 22.92% on fresh templates. SNLI accuracy is only 39.58% (vs Qwen's 92.71%).
  - **Replay Continuation Weakness**: Replay adds +5.21 pp on SNLI, with 95% CI spanning zero ([-2.08, +13.54]), while reducing fresh policy all-six from 30.21% to 26.04%.
  - **CPU Assembly Optimization**: Host AVX2/FMA assembly passes exact bitwise parity across 1,320 pairs ($\Delta p = 0.0$), delivering a **3.12x speedup** on policy (46.28 ms vs 144.39 ms) and **5.15x** on SNLI (13.62 ms vs 70.14 ms). INT8 was unexecuted due to missing AVX-VNNI hardware instructions.

  #### 8. Calibration & Application Policies
  - **Calibration Transfer Regression**: Temperature fitting accepted on calibration-gate worsens confirmation NLL on `I_FIXED` (0.674 $\to$ 1.328) and `G` (0.190 $\to$ 0.202). In contrast, D (0.751 $\to$ 0.345) and X (0.227 $\to$ 0.210) improve.
  - **Application Cost Analysis**: Filtering by minimum probability fails to beat review-all (mean cost 0.10): Arm D achieves mean cost 0.342, Arm G achieves 0.117. Arm X records zero errors but achieves only 5.21% coverage (cost 0.095).
  - **Qwen Frozen NLI Baseline**: `NLI_CODE` achieves **92.71% accuracy** (89/96), 63.54% coverage at threshold 0.8 with 0 errors, yielding mean cost **0.0365** (substantially beating review-all).

* **Supporting Review Artifacts**:
  - Run Key: `baa2f1ebf0ab02e9f3e4ad795be3330e9cd8c04bcd530cc7992d9f35f0f723bc`
  - Archive SHA: `2f9b758c8a4c0af86c725eb760e96d88250d209e50118b9bf2b6ede71c54cce1`
  - Summary SHA: `b4149b1226a3d20ca05e3f63b33d366bb000bf29daec0d0ffc7b8d2879187672`
  - Dataset SHA: `573585bd877ce15b0a111138a256db419b4dceb5a0cf857aa0f738035db19041`
  - Drive Folder: [Google Drive Run 20260929T194506](https://drive.google.com/drive/folders/1gLVnuSIHUDAqyQlyxFyKuc2LWcOvklgx)
  - Archive Download: [Google Drive Archive](https://drive.google.com/file/d/1yXOks6ms6-aaEhj-jZ1WvuWtBQsIIzjA/view)
  - Read-Only Reconciliation Scripts: `reconcile_unified.py`, `reconcile_execution.py`, `reconcile_calibration.py`.

---

### 37. OpenKind T4 Integrated Research: Useful Integrations, Exact Cache Savings, and an Unfinished Classifier Comparison
* **File**: [`37_openkind_t4_integrated_research_lab.ipynb`](./37_openkind_t4_integrated_research_lab.ipynb)
* **Run ID**: `20261001T105345_828556Z_6e6d92`
* **Protocol**: `openkind-t4-integrated-research/v4.0.0`
* **Run Date**: 1 October 2026
* **Target Hardware & Environment**:
  - NVIDIA Tesla T4 16GB (15.0 GiB reported VRAM, CUDA runtime, native engine with Qwen3.5-9B publisher Q4_K_M and Qwen3.5-4B publisher Q4_K_M, 34/34 layers offloaded).
  - CPU: Local Python runtime / scikit-learn / TF-IDF logistic regression baseline.
* **Evaluated Models**: Qwen3.5-9B Q4_K_M (primary native decoder); Qwen3.5-4B Q4_K_M (matched deployment profile comparator); TF-IDF + Logistic Regression (cheap baseline). *Note: BERT, ModernBERT, DeBERTa, GLiClass, and Indecis shared-build remained unmeasured due to an identity-unsafe log-copy failure in setup.*
* **Status & Disposition**: **`PARTIAL`** (Completion marker absent; 45 executed stages, 12 blocked, 2 failed; 41 experiment blocks with verified receipts; no model or service promoted).
* **What it Measured / Scope**:
  - Timed execution and whole-request correctness of integrated readouts: selective indexing + deterministic routing-priority composition (Arm XR) and a narrowly joint route/urgency readout (Arm NJ).
  - Multi-catalogue host snapshot caching for grouped requests (Arm G and GR) evaluating steady reuse, cold startup, and host RAM retention.
  - Dedicated-process history isolation extended directly to integrated readouts (X, XR, NJ) under 1-context and 4-context traces.
  - Deployment profile model size comparison: Qwen3.5-4B Q4_K_M vs Qwen3.5-9B Q4_K_M on fresh policy confirmation and 192-case SNLI.
  - Batching diagnostics on Arm X across batch sizes 1, 2, and 4.
  - Prompt schema sensitivity (4 transformations across 12 development cases) and dynamic Choice behavioral probes (runtime policy changes, ordinal rubrics, candidate omission/capacity traps, option interaction, sibling visibility, evidence position).
  - Calibration transfer and empirical action/review policies under asymmetric loss costs.
  - Bookkeeping failure root-cause analysis for blocked encoder / classifier stages.

* **Stage-by-Stage Findings & Audit Results**:

  #### 1. Complete 9B Request Comparison on Fresh Confirmation (192 fact groups × 2 technical repeats)
  | Profile | Field Accuracy (%) | All Six Correct (%) | p50 Request (ms) | p95 Request (ms) | Available Field Distributions / 6 |
  |---|---|---|---|---|---|
  | **9B:NJ:control** | 94.010% | 79.688% | 228.309 | 240.683 | 6 / 6 |
  | **9B:D:control** | 79.688% | 22.396% | 325.922 | 349.839 | 6 / 6 |
  | **9B:XR:control** | 95.399% | 84.375% | 242.983 | 256.648 | 5 / 6 |
  | **9B:GR:multicache** | 94.010% | 79.167% | 690.162 | 727.937 | 5 / 6 |
  | **9B:X:control** | 87.500% | 44.271% | 243.291 | 257.087 | 6 / 6 |
  | **9B:E:control** | 87.240% | 46.354% | 3,634.013 | 3,689.855 | 0 / 6 |
  | **9B:G:multicache** | 88.542% | 53.125% | 691.222 | 728.557 | 6 / 6 |

  *Timings reflect the full implemented request including subcalls, host rules, and serialization (excluding server startup and evidence writing). Missing NLL/Brier in XR/GR is intentional due to withheld route distribution; Arm E outputs point JSON only.*

  - **Selective Indexing + Routing Composition (Arm XR)**: Relative to Arm X, XR increases field accuracy by **+7.90 pp** (95% CI: [6.77, 9.11]) and whole-request correctness by **+40.10 pp** (95% CI: [33.33, 47.40]), reaching 162/192 all-six-correct requests (vs X's 85/192 and JSON's 89/192). Median latency remains virtually identical to X (242.98 ms vs 243.29 ms) and is **~14.96x faster than JSON** (3,634.01 ms). Saved-output analysis confirms 91 routing corrections and 0 routing harms, with all other fields completely preserved.
  - **Narrowly Joint Readout (Arm NJ)**: Replaces route and urgency with an explicit joint categorical source, calibrating the joint distribution before deriving marginal probabilities. Achieves **94.01% field accuracy** and **79.69% all-six correctness** (153/192) at **228.31 ms median** (~6.2% faster than X). All six marginal distributions are mathematically valid, though individual marginals do not yield a joint whole-request correctness probability.
  - **XR vs NJ Trade-off**: XR provides superior discrete accuracy (84.38% vs 79.69% all-six) but withholds route distribution; NJ provides full exact marginal distributions across all 6 fields with competitive accuracy.

  #### 2. Actionable Semantic Error Analysis
  - On new policy confirmation, XR achieves 100% accuracy on route and urgency across all 192 cases.
  - The 30 failing requests comprise three disjoint clusters:
    1. **23 Eligibility Errors** (and corresponding derived action errors): In all 23 cases, the model predicts `undetermined` instead of `ineligible` (leading to action `request missing information` instead of `deny access`). Every single case features unrecorded certification (`certified: null`) alongside a known failing points score ($< 70$). Under the contract, eligibility requires `certified == true AND points >= 70`; a known failing condition conclusively establishes ineligibility regardless of missing fields.
    2. **4 Case-State Errors**: The model selects `closed` instead of `open`.
    3. **3 Retry-Cap Errors**: The model outputs `1` instead of the capped value `3`.
  - NJ replicates the identical 23 eligibility errors, isolating partial-information logic, event recency, and numerical capping as targeted focal points for subsequent prompt/logic refinement.

  #### 3. Multi-Catalogue Snapshot Caching & Batching Diagnostics
  - **Dedicated Cache Experiment (96 Paired Inferences)**: Tested 24 cases twice across Arm G (48 requests) and Arm GR (48 requests). Control and multicache arms exhibited **zero observed probability drift ($\Delta p = 0.0$)** and identical discrete decisions across all 96 pairs.
  - **Cache Hit Dynamics & Amortization**:
    - First grouped request misses all 3 catalogues; subsequent 47 requests achieve **141 hit subrequests** (control records 0 hits).
    - Maximum retained snapshot payload in host RAM: **178,865,396 bytes (~170.6 MiB)**.

    | Dedicated Trace | Control First Request | Multicache First Request | Control Steady Median | Multicache Steady Median | Steady Speedup |
    |---|---|---|---|---|---|
    | **Arm G** | 1,480.12 ms | 1,512.98 ms | 1,440.76 ms | 586.27 ms | **2.46x** |
    | **Arm GR** | 1,482.45 ms | 1,561.74 ms | 1,432.09 ms | 605.52 ms | **2.36x** |

    - On full confirmation, G drops from 1,511.22 ms to 691.22 ms and GR drops from 1,511.75 ms to 690.16 ms (**~54% latency reduction / ~2.19x speedup**).
  - **Batching Diagnostic on Arm X**: Comparing batches of 1, 2, and 4 contexts on 12 cases (36 total items) over 8 reserved sequence slots:
    - Batch 1: 36 requests, median 212.04 ms, **4.749 items/sec**.
    - Batch 2: 18 requests, median 440.83 ms, **4.528 items/sec**.
    - Batch 4: 9 requests, median 891.13 ms, **4.477 items/sec**.
    - All 108 comparisons passed with $\Delta p = 0.0$, but larger batches showed slightly lower throughput on this short-context workload.

  #### 4. Model Size Deployment Comparison (4B vs 9B Q4_K_M on T4)
  | Readout / Task | 4B Accuracy (Field / All-Six) | 9B Accuracy (Field / All-Six) | 4B Median (ms) | 9B Median (ms) |
  |---|---|---|---|---|
  | **Policy: Arm X** | 79.60% / 25.52% | 87.50% / 44.27% | 150.09 ms | 243.29 ms |
  | **Policy: Arm XR** | 86.20% / 49.48% | 95.40% / 84.38% | 153.68 ms | 242.98 ms |
  | **Policy: Arm NJ** | 83.94% / 52.60% | 94.01% / 79.69% | 145.88 ms | 228.31 ms |
  | **SNLI Confirmation** | 90.63% (174/192) | 86.98% (167/192) | 98.47 ms | 168.21 ms |

  - On policy formulations, 9B buys a massive gain in whole-request correctness (+34.90 pp on XR, +27.09 pp on NJ).
  - On SNLI sentence-pair NLI, 4B is faster and scores higher (+3.65 pp, 95% CI: [-0.52, +7.81]). Model suitability is strictly task-dependent.

  #### 5. Dedicated-Process Serving Isolation for Integrated Readouts
  - All 6 mixed-process controls failed history checks; all 6 native-only controls and 6 dedicated-process treatments **passed exact parity ($\Delta p = 0.0$)** for X, XR, and NJ across 1-context and 4-context traces.
  - Mixed-process 4-context failures exhibited significant drift:
    - Arm X: $\max \Delta p = 0.113938$ (6 fixed diagnostic-policy flips).
    - Arm XR: $\max \Delta p = 0.113938$ (route distribution withheld).
    - Arm NJ: $\max \Delta p = 0.073574$ (4 discrete field flips, 1 policy flip).
  - Dedicated-process trace execution times ranged from 78 to 91 s, comparable to mixed-process times (73 to 99 s).

  #### 6. Schema Sensitivity and Dynamic Choice Probes
  - **Schema Sensitivity**: All 48 schema comparisons failed strict invariance gates. Maximum observed probability shifts:
    - Code remapping: $\max \Delta p = 0.871611$ (9 field flips).
    - Visible key renaming: $\max \Delta p = 0.513453$ (3 field flips).
    - Natural option reordering: $\max \Delta p = 0.235829$ (4 field flips).
    - Unrelated question addition: $\max \Delta p = 0.168059$ (0 field flips, 1 policy threshold flip).
  - **Dynamic Choice Probes**:
    - Runtime policy thresholds: 48/48 correct.
    - Ordinal rubric: 24/24 correct (4 unique payloads).
    - Candidate omission: 37/40 correct (1 error at K=16, 2 errors at K=32 where plan numeric IDs resemble required capacity despite differing textual descriptions).
    - Option interaction: 24/24 correct (4 unique payloads).
    - Sibling visibility: 36/36 correct (5 unique payloads).
    - Evidence position: 32/32 correct (8 unique payloads).

  #### 7. Probability Scores and Decision Automation Policies
  - **Calibration Impact**:
    - 9B D: raw NLL 0.819 $\to$ cal 0.438; raw Brier 0.356 $\to$ cal 0.265 (improved).
    - 9B X: raw NLL 0.353 $\to$ cal 0.294; raw Brier 0.198 $\to$ cal 0.168 (improved).
    - 9B NJ: raw NLL 0.207 $\to$ cal 0.170; raw Brier 0.107 $\to$ cal 0.087 (improved).
    - 9B G: raw NLL 0.270 $\to$ cal 0.291; raw Brier 0.164 $\to$ cal 0.178 (worsened).
    - 9B NLI: raw NLL 0.410 $\to$ cal 0.411; raw Brier 0.199 $\to$ cal 0.198.
    - 4B NLI: raw NLL 0.362 $\to$ cal 0.405; raw Brier 0.179 $\to$ cal 0.167.
  - **Empirical Action/Review Policies** (stipulated losses: wrong=5, review=0.1, correct=0; baseline review-all costs 0.1000):
    - Arms D, X, 4B X, lexical: 0% coverage (mean cost 0.1000).
    - Arm G: 44/192 accepted (4 wrong), 22.92% coverage, mean cost 0.18125 (worse than review-all).
    - 9B NLI: 80/192 accepted (0 wrong), 41.67% coverage, mean cost **0.05833** (beats review-all).
    - 4B NLI: 103/192 accepted (2 wrong), 53.65% coverage, mean cost 0.09844.
    - NJ recorded `NOT_MEASURED_JOINT_POLICY_NOT_IMPLEMENTED`; XR/GR lack the required route distribution.

  #### 8. Classifier Comparison Diagnostic & Bookkeeping Failure
  - The encoder evaluation suite was blocked due to a Python exception in the runner: `run_logged.publish()` unconditionally called `shutil.copyfile` to copy its execution log into the run directory even when the log already resided at the destination path, raising `SameFileError`.
  - The owned subprocesses terminated with return code -15 after ~0.018–0.037 s with empty logs. BERT, ModernBERT, DeBERTa, GLiClass, and Indecis shared-build were not evaluated; this was an execution harness bookkeeping defect rather than a GPU memory or model quality failure.
  - **TF-IDF + Logistic Regression Baseline**: Reached 95.05% field acc / 76.56% all-six on development, but dropped to 74.83% field acc / 31.77% all-six on fresh confirmation at 2.41 ms median (SNLI acc 37.5%), illustrating sharp out-of-distribution transfer degradation for simple bag-of-words classifiers.

  #### 9. Status of Outstanding Questions
  - **OQ-01 (X + Routing Composition)**: XR timed on fresh confirmation; confirms 84.38% all-six correctness (+40.1 pp over X) at 242.98 ms; route probability distribution remains withheld.
  - **OQ-02 (Narrowly Joint Source)**: NJ provides exact marginals and 79.69% all-six correctness at 228.31 ms; joint action/review policy not yet implemented.
  - **OQ-03 (Multi-Catalogue Caching)**: Grouped latency cut by ~54% (2.19x speedup) with bitwise output parity ($\Delta p = 0.0$); eviction, TTL, and concurrent readers remain unmeasured.
  - **OQ-05 (Readout History Isolation)**: Dedicated processes pass exact parity for X, XR, and NJ; mixed process drift confirmed across all readouts.
  - **OQ-14 (Matched Size Comparison)**: 9B is decisively superior on structured policy decisions; 4B is faster and slightly more accurate on short-premise NLI.
  - **OQ-09–11, OQ-12 (Stronger Classifiers & Indecis Reuse)**: Blocked by setup logging exception; requires harness repair and separate execution.
  - **OQ-17–23 (Semantic Scope & Probes)**: High accuracy on small probes; candidate omission reveals capacity/ID confusion; prompt schema sensitivity persists.
  - **OQ-33–39 (Probability & Policy Utility)**: Profile-specific calibration outcomes; completed joint policy implementation remains outstanding.

* **Supporting Review Artifacts**:
  - Run Key: `930195a6d0a3ff83a617822fde813eae5e4a83f80a64eda9d20d8eb830b3e658`
  - Archive SHA256: `f0cfde45b65efd1716220d5860695cb1a1122c03863ce6155c914c569569eac3`
  - Completion Marker: Absent (`PARTIAL`)
  - Primary References: R1 (`REPORT.md`, `RUN_STATUS.json`, `summary.json`, `manifest.json`), R2 (Quality/NLI block results), R3 (`paired_comparisons.json`), R4 (`cache_gate_*`, `multicache_summary.csv`), R5 (`history_*`, `isolation_summary.csv`), R6 (`dynamic_9B`, `schema_9B`, `shape_X`), R7 (`u4_native.py`, `u4_data.py`, `u4_models.py`), R8 (`errors_this_session.json`, command receipts), R9 (`calibration_results.json`, `policies.json`).
  - Read-Only Verification: `python reconcile_u4.py <evidence> <review>`, `python analyze_u4_details.py <evidence> <review>`, `python analyze_u4_error_clusters.py <evidence> <review>`.

---

### 38. OpenKind T4 Recovery & Targeted Reliability
* **File**: [`38_openkind_t4_recovery_reliability_lab.ipynb`](./38_openkind_t4_recovery_reliability_lab.ipynb)
* **Protocol**: `openkind-t4-recovery-reliability/v4.1.0`
* **Run ID**: `20261001T202134_352688Z_4a8cbc` (Run Key `4a8cbc`, Archive SHA-256 `d366adc6...15eac`)
* **Execution Status**: `PARTIAL` (23 executed, 2 failed in `ensurepip`, 15 blocked; completion marker absent).
* **Target Hardware & Environment**: NVIDIA Tesla T4 16GB & CPU.
* **Scope & What was Measured**:
  - Followed partial v4.0.0 run `20261001T105345_828556Z_6e6d92` with immutable parent checks.
  - Repaired subprocess logger self-copy (`SameFileError`), pinned `safetensors >= 0.8.0`.
  - Tested failure-first eligibility instructions (Arm NF: known failing condition takes precedence over unrecorded info) and joint factual readout (Arm PF: 9-way joint certification/points evidence categories $\to$ derived eligibility $\to$ action).
  - Measured within-request input encoding reuse in Indecis CPU Go worker across 1,980 paired technical requests (990 unique cases).
  - Evaluated dependency-aware whole-request policies (evaluating 4 source events covering the 6 outputs without multiplying independent field marginals) across new 216-case R41 confirmation panel (balanced across 9 certification/points strata).
  - Dedicated-process history isolation across 1-context and 4-context workloads.
* **Key Findings**:
  - **NF Targeted Logic Repair**: On the 48 new cases combining missing information with a known failing condition, NF achieved **48/48 correct eligibility** (versus NJ's 34/48 and XR's 34/48), proving the intended logical correction succeeded. Overall eligibility improved from 201/216 to 213/216.
  - **Collateral Regression in NF**: NF introduced severe retry regressions (retry correctness dropped from 89/216 under NJ to 76/216 under NF; 13 previously correct retry decisions became wrong with none repaired). Consequently, complete-request all-six correctness fell from 33.33% (72/216) to **28.70% (62/216)**.
  - **PF Primitive Joint Evidence Extraction**: PF correctly aggregated source-to-derived probability mass, but primitive evidence extraction failed (eligibility correctness fell to 190/216; known-failure with missing was only 26/48). Exact host rules cannot compensate for an inaccurate evidence distribution.
  - **Indecis Input Reuse**: Exact within-request input reuse delivered a **4.80× steady-state latency reduction** on R41 (117.13 ms $\to$ 24.38 ms median CPU time; U4 confirmation 172.30 ms $\to$ 36.00 ms, 4.79×) with **zero probability delta and zero decision flips** across all 1,980 paired requests. First-use latency showed modest savings (927.52 ms $\to$ 867.58 ms). Transfer accuracy remained zero (0/216 all-six correct).
  - **Error Clustering in Native Qwen**:
    - *Numeric Capping*: 145 cases had retry counts from 10 to 26 (correct capped answer is 3). XR got 53/145, NJ 21/145, NF 9/145, PF 36/145. **100% of errors equaled the first decimal digit** of the stated retry count (e.g. count 21 produced 2 instead of 3).
    - *Chronological Recency vs Display Order*: All four native arms made the exact same 36 case-state errors, predicting older "closed duplicate" instead of newer "open" on newest-first renderings despite explicit instructions that the first displayed event is newest. Indecis predicted the older event on 100% of cases (216/216 wrong).
  - **Dependency-Aware Policy Failure**: All frozen policies lost to review-all (cost 0.1000) under declared losses (wrong=5, review=0.1, correct=0): NJ accepted 9 (1 wrong, cost 0.11898), NF accepted 52 (17 wrong, cost 0.46944), PF accepted 108 (61 wrong, cost 1.46204). Accepted calibration temperatures degraded NLL on new renderings (e.g. NJ NLL worsened 0.4666 $\to$ 0.7621).
  - **Process Isolation**: Dedicated resident processes passed all history checks ($\Delta p = 0.0$ across 8 conditions), while mixed processes failed with probability drift up to 0.079 and field flips.

---

### 39. OpenKind T4 Classifier Environment Recovery
* **File**: [`39_openkind_t4_recovery_reliability_v4_1_1.ipynb`](./39_openkind_t4_recovery_reliability_v4_1_1.ipynb)
* **Protocol**: `openkind-t4-recovery-reliability/v4.1.1`
* **Run ID**: `20261001T223435_041738Z_f25616` (Run Key `f25616`, Archive SHA-256 `76c663fc...f1ea`)
* **Execution Status**: `EXPLORATORY_COMPLETE` (All 17 classifier stages executed successfully with verified completion marker).
* **Target Hardware & Environment**: NVIDIA Tesla T4 16GB & CPU.
* **Scope & What was Measured**:
  - Repaired `venv` environment creation via `--without-pip` followed by target interpreter seeding with SHA-256 hash-pinned pip 26.2.1 wheels.
  - Executed 17 classifier recovery stages: released DeBERTa-v3-base NLI specialist, ModernBERT-base (frozen backbone + fitted heads vs full fine-tuning), BERT-base controls (frozen vs full fine-tuning), and GLiClass dynamic-label models.
  - Evaluated on shared 192-case SNLI confirmation panel and 216-case R41 policy confirmation panel.
* **Key Findings**:
  - **Released DeBERTa NLI Specialist**: Strongest compact finding. Reached **90.10% accuracy (173/192)** at **20.92 ms median request time** on SNLI, outperforming Qwen 9B (167/192 = 86.98% at 149.13 ms; ~7.13× latency reduction). Temperature calibration ($T \approx 1.9319$) improved NLL to 0.2972 and Brier to 0.1559.
  - *DeBERTa Acceptance Caveat*: At the development-selected acceptance threshold, DeBERTa accepted 74 cases with 2 wrong, yielding mean cost **0.11354**—failing to beat review-all (0.1000). Qwen NLI accepted 80 with 0 wrong (mean cost 0.05833). DeBERTa is a strong specialist candidate, but its acceptance policy requires separate qualification.
  - **BERT-Family Policy Fine-Tuning**: Full fine-tuning produced genuine learning over frozen features (+18.06 pp field acc for ModernBERT, +22.15 pp for BERT):
    - *BERT full fine-tuning*: 73.46% field acc, 21.30% all-six at 15.78 ms.
    - *ModernBERT full fine-tuning*: 70.45% field acc, 19.44% all-six at 20.92 ms.
    - *Frozen backbones*: ~51–52% field acc, 0–5% all-six at 16–18 ms.
  - *Majority-Class Collapse in Small Encoders*: Both BERT arms and frozen ModernBERT predicted "ineligible" on all 216 new cases (matching 120-case majority). Every BERT/ModernBERT arm predicted `retries=3` on all 216 cases (yielding 87.96% retry accuracy without learning capping). The small fixed-schema recipe (1,152 training cases) failed to produce robust transfer.
  - **GLiClass Dynamic-Label Limitations**: Reached 45.45% field accuracy and 3.24% all-six correctness on R41 policy (65.13 ms across 5 serial forwards); SNLI accuracy was 35.42%. All 32 evidence-position cases were rejected due to sequence length exceeding input token budgets.
  - **FP16 Autocast Probes**: All 60 classifier FP16 probes passed numerical tolerance ($\max \Delta p < 0.003$).

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
16. **SDPA Recomputation Backend Consistency in Activation Checkpointing**:
    When training adapters or fine-tuning under activation checkpointing with forced SDPA kernels (e.g. `SDPBackend.MATH`), context managers must encompass both the initial forward evaluation and the activation recomputation phase during `.backward()`. Mismatched ambient attention kernels between forward execution and backward recomputation can trigger undefined operator replay, NaN gradients, or kernel dispatch failures.
17. **Task-Specific Replay vs Long-Document Joint Preservation**:
    Supervised replay with source-label cross-entropy on auxiliary short-premise NLI (SNLI) substantially repairs narrow regression benchmarks (+18 pp accuracy) and outperforms parent-KL consistency locally, but fails to prevent catastrophic drift on primary long-document tasks (ContractNLI entailment loss $>13$ pp, QASPER false-none rate $>24\%$). Replay on simple sentence pairs does not provide regularizing signal for long-context cross-document answerability; retention panels must directly evaluate the long-document reasoning and answerability boundaries being preserved.
18. **MoE Active Parameter Sparsity vs Prefill Weight Residency**:
    An MoE's active parameter count reflects only token-level floating-point operations, not memory residency. Even when per-token active routing selects only top-4 of 60 experts, document prefills of modest length (117 to 629 tokens) touch 98.7% to 99.6% of all resident experts across layers (mean 59.33 / 60 experts per layer). Consequently, runtime memory consumption remains governed by total model weights (7.82 GiB VRAM) rather than active parameter fractions unless unrouted experts are physically pruned from memory (-38.3% VRAM via structural module deletion).
19. **Top-k Routed Weight Normalization Confound & Quantized Dispatch Numerics**:
    In MoE architectures where router probabilities are unnormalized across selected experts (`norm_topk_prob=false`), truncating $k$ (e.g. top-4 to top-2 or top-1) reduces both active expert capacity and total routed weight magnitude ($\sum w_i < 1.0$), confounding routing selectivity with activation scaling. Furthermore, low-bit quantized backends (e.g. `bitsandbytes` NF4) often dispatch single-row and multi-row inputs through differing numerical kernel paths; prefill state splitting interacts with dynamic expert batch sizes, creating shape-dependent numerical drift ($\Delta p$ up to 0.12) that breaks prefix cache equivalence.
20. **MoE Sliced Continuation Divergence Across Linear References & Option Presentation Order Bias**:
    Replacing quantized matrix-vector kernels with an explicit IEEE FP32 dequantized linear reference over NF4 weights fails to restore prefix-cache mathematical equivalence on MoE models ($\max \Delta p$ increased from 0.1199 to 0.1553, with 1 decision flip), proving that prefix-sliced divergence is not solely a low-bit GEMM batching artifact but stems from subtle attention position/mask dynamics and dynamic router assignments across split sequences. Furthermore, unprompted option letter presentation introduces severe ordering bias in MoE next-token scoring, flipping the winner in 25.0% of cases ($\max \Delta p = 0.1686$) unless neutralized by explicit permutation averaging.
21. **Hybrid/Mamba Prefix Caching Block Boundaries & Quantized MoE Repeatability**:
    In hybrid DeltaNet/Mamba architectures (such as Qwen 3.5), vLLM automatic prefix caching falls back from `all` to `align` mode, resuming cached computation strictly at block boundaries (528 tokens for 4B, 1,056 tokens for MoE INT4). Prefixes shorter than a block boundary yield zero cached token reuse throughout. Furthermore, quantized MoE backends (e.g. GPTQ/Marlin) can exhibit severe probability drift (up to 17.60 pp sequential, 11.92 pp concurrent) even when zero cache reuse occurs. Repeatability with caching disabled must be verified across eager and compiled execution before evaluating prefix-cache speedups.
22. **Microbatching Perturbation of Repeatability & Staged Prefix Qualification**:
    While dense hybrid models (Qwen 3.5 4B) achieve bitwise deterministic outputs under serial execution ($\Delta p = 0.0$), concurrent client requests alter GPU dynamic microbatching, inducing measurable probability drift ($\Delta p = 0.0369$) and flipping decision winners (4/24 flips). Conversely, quantized MoE models (Qwen 3.5 MoE INT4) exhibit significant drift under serial execution ($\Delta p = 0.1360$), indicating kernel-level non-determinism independent of request concurrency. When shared prefixes exceed vLLM Mamba block boundaries (e.g. 529 tokens for a 528-token block), cold staged prefix reuse achieves strict numerical parity ($\Delta p = 4.37 \times 10^{-8}$) and a 1.92x speedup on dense 4B, proving that prefix caching is mathematically sound once block alignment thresholds are cleared.
23. **Deterministic Output Composition vs Unconstrained Neural Prediction**:
    When domain dependencies between fields are logically rigid (e.g. eligibility strictly dictating action: eligible $\to$ grant access, ineligible $\to$ deny access, undetermined $\to$ request missing info; or urgency determining route priority), asking the model to predict dependent fields independently causes severe cross-field inconsistency (up to 28%–31% contradiction rate in 4B and 27/96 contradictions in 9B). Deriving dependent outputs in host code deterministically eliminates cross-field contradictions, cuts unnecessary prediction latency, and significantly improves overall request-level accuracy (+5.38 pp field accuracy on 9B).
24. **Process-Level State Pollution from Interleaved Autoregressive Generation**:
    While consecutive native decision requests and exact prefix reuse achieve bitwise mathematical parity ($\Delta p = 0.0$), deliberately interleaving autoregressive text or JSON generation within the same process alters subsequent native forward passes ($\max \Delta p$ up to 0.095, flipping selected decisions). Neither prompt padding nor disabling CUDA graphs resolves the drift, and session-end states fail reset parity against fresh runs. Recomputing a prompt prefix is not equivalent to resetting dirty process state; complete process isolation is required when mixing native decision paths with freeform text generation.
25. **Systematic Semantic Traps and the Failure of Raw Confidence Gating**:
    Catalogue formatting and compound label naming can induce extreme, systematic semantic confusion in high-capacity models (e.g. Qwen3.5-9B Q4 collapsing plain "closed" into "closed duplicate" with 0% recall, and appending the "priority" suffix to every single routing label regardless of urgency). Because the model is highly confident in these erroneous classifications, filtering decisions by minimum probability thresholds ($p \ge 0.90$ or $0.95$) fails to remove them (30% to 78% of accepted cases remain wrong). Confidence thresholding cannot substitute for addressing catalogue interference, decomposing readouts, or applying deterministic post-hoc dependency rules.
26. **Sequence Slot Reservation Guardrails under Batched Execution**:
    Reducing runtime sequence slot reservations below the batch workload requirement (e.g. configuring 3 sequence slots for a 4-context batched request) introduces substantial numerical drift ($\Delta p \approx 0.059$) and fails numerical parity gates, even when discrete argmax winners happen to match. Runtime sequence reservations must be provisioned to cover maximum batch concurrency rather than trimmed for superficial VRAM savings.
27. **T4 Viability of 9B Quantized Decision Inference**:
    Large dense models (Qwen 3.5 9B) quantized to 4 bits (Q4_K_M) fit comfortably within commodity 16GB GPUs (Tesla T4), utilizing only 6.30 GiB peak VRAM under 4-context batched workloads with full 34/34 layer GPU offload. Combining 5-field prediction, deterministic action derivation, and verified prefix cache reuse delivers a 2.35x latency speedup (339 ms median vs 798 ms baseline) and an 11.5x speedup over generated JSON (3,910 ms), demonstrating that production-speed structured decision serving is feasible on low-cost hardware once readout traps and process-history gates are resolved.
28. **Selective Field Indexing (Arm X) Over Blanket Indexing**:
    Indexing only route and case state while retaining natural labels for other fields resolves the multi-field decoding bottleneck (reducing scoring decode calls from 2 to 1), cutting latency by 27.6% (222.22 ms vs 306.92 ms) and improving fresh field accuracy (+9.55 pp to 92.19%) and all-six accuracy (+36.46 pp to 60.42%). While blanket indexing of every field previously degraded performance, surgical indexing of high-cardinality or branch-heavy fields provides a superior speed/accuracy frontier.
29. **Deterministic Route-Priority Composition Confirmed on Timed Request Path (Arm R / GR)**:
    Composing the route priority rule deterministically from predicted base route and urgency delivers +9.03 pp field accuracy on fresh templates (raising D from 82.64% to 91.67% and G from 93.23% to 95.49%) with zero latency penalty on the measured request path (306.72 ms vs 306.92 ms). Grouped requests with route composition (Arm GR) achieve 83.33% all-six correctness. When fields possess rigid logical hierarchy, rule composition strictly outperforms end-to-end multi-label neural prediction.
30. **Resident Process Isolation Resolves Mixed-Traffic State Contamination**:
    Separating native decision serving and autoregressive JSON into dedicated resident processes completely eliminates the history drift observed in mixed processes ($\Delta p = 0.0$, passing all probability and answer gates) with only a ~4%–6% whole-trace execution penalty. Both resident processes fit simultaneously within a 15-GiB Tesla T4 (12.10 GiB total VRAM with full 34/34 offload). In-process restarts also preserve state but incur a severe 3.60x–3.93x latency penalty (~14–19s recovery per restart).
31. **Prompt Schema Sensitivity Persists Beyond Process Isolation**:
    While process separation purges accumulated runtime history, native decision probabilities remain sensitive to prompt formatting: reordering options, adding unrelated questions, remapping codes, or renaming visible keys alters probability vectors (up to $\Delta p = 0.912$, failing 48/48 numerical gates). An efficient prefix-cached runtime does not inherently guarantee arbitrary-question statistical independence; API design must treat prompt schema structure as part of the conditioning state.
32. **Catalogue Caching Sub-linear Multi-Question Amortization vs Dynamic State Limits**:
    Native Qwen catalogue caching amortizes multi-question execution sub-linearly (scaling from 1 to 16 questions costs only 1.57x the latency for ~256 state tokens: 447 ms to 702 ms). However, static catalogue caching cannot cache dynamic state: for long contexts (~4,106 state tokens), prefill dominates (5,396 ms warm vs 5,694 ms cold). In contrast, open-option architectures that re-encode input per question scale strictly linearly (Q=16 is 16.01x Q=1).
33. **Compact-Model Fixed Head & Replay Transfer Limitations**:
    Fast CPU fixed heads (Indecis) learn fixed schemas with low latency (~46 ms) but suffer severe transfer degradation on fresh rendering templates (dropping from 91.15% dev to 69.10% fresh field accuracy; SNLI at 39.58% vs Qwen's 92.71%). Replay continuations fail to preserve or transfer capabilities (confidence intervals spanning zero). High speed cannot substitute for representational capacity on out-of-distribution prompts.
34. **Calibration Gate Acceptance Does Not Guarantee Transfer Generalization**:
    Post-hoc temperature scaling accepted on held-out calibration-gate data can severely degrade proper scoring rules on fresh distribution transfer (e.g. Indecis fixed-head fresh-policy NLL doubling from 0.674 to 1.328). Calibration must be evaluated separately by task and rendering family rather than pooled into a single scalar gate.
35. **Timed Selective Indexing with Routing Composition (Arm XR)**:
    Surgically indexing high-cardinality fields (route and case state) combined with deterministic host-side routing-priority composition (Arm XR) delivers **95.40% field accuracy** and **84.38% all-six correctness** on fresh policy confirmation (gaining +40.10 pp whole-request correctness over raw selective indexing Arm X) with zero latency penalty (242.98 ms median). XR is ~14.96x faster than generated JSON (3,634.01 ms) while achieving 91 routing corrections and 0 routing harms.
36. **Narrowly Joint Categorical Readout (Arm NJ) and Marginal Probability Consistency**:
    Predicting a joint categorical source over interdependent fields (e.g. 4-way base route and urgency) and pushing forward to derived marginals yields exact field probability distributions (all 6 fields available) while avoiding naive independent-field assumptions. Arm NJ achieves 94.01% field accuracy and 79.69% all-six correctness at 228.31 ms median (~6.2% faster than X), offering an attractive operating point when downstream consumers demand calibrated probability distributions.
37. **Multi-Catalogue Host-RAM Snapshot Caching for Grouped Readouts**:
    Retaining compiled sequence snapshot payloads in host RAM across disparate catalogues (~170.6 MiB for 3 catalogues) eliminates cold-state recomputation in alternating grouped requests. Steady-state grouped request latency drops by ~54% (from 1,511 ms to 690 ms, a ~2.19x speedup) with bitwise identical probability vectors and zero discrete decision flips across 96 paired evaluations.
38. **Throughput Inefficiency of Dynamic Batching on Commodity GPUs for Short Sequences**:
    For short-context decision requests on commodity GPUs (Tesla T4), increasing batch size from 1 to 2 and 4 contexts maintains numerical parity ($\Delta p = 0.0$) but slightly degrades overall throughput (falling from 4.75 items/sec at batch 1 to 4.48 items/sec at batch 4). Without large batch queues or long shared prefill amortization, serial execution achieves superior latency and equivalent aggregate throughput.
39. **Dedicated-Process Isolation Across Diverse Readout Topologies**:
    The necessity and effectiveness of separate resident processes for native decision serving and autoregressive text/JSON generation extends to all evaluated readout topologies (X, XR, and NJ). While mixed-process execution introduces severe probability drift (up to $\Delta p = 0.114$, flipping discrete decisions and policy thresholds), dedicated resident processes achieve bitwise exact history invariance ($\Delta p = 0.0$) across both 1-context and 4-context workloads with negligible trace overhead (~4%–6%).
40. **Partial-Information Semantic Logic Failures in Natural Decision Boundaries**:
    Models can exhibit highly localized, systematic semantic failures when evaluating partial-information rules: across 192 fresh policy cases, 100% of eligibility errors (23/23) occurred when a required certification field was unrecorded while a separate numeric points score was conclusively failing ($< 70$). Rather than correctly concluding ineligibility under a conjunctive contract ($A \land B$), the model retreated to `undetermined` whenever any field was missing. Diagnostic error clustering localizes the intervention to partial-information rule framing rather than general model capacity tournaments.
41. **Exact Within-Request Input Reuse for Sequence-Scoring Encoders**:
    Reusing a single input sequence encoding across multiple predicted classification heads eliminates redundant encoder passes, cutting CPU request latency by ~4.80x (117.13 ms $\to$ 24.38 ms on Indecis) with bitwise identical probabilities ($\Delta p = 0.0$) and zero decision flips. However, faster execution does not repair deficient underlying semantic representations (0.0% all-six accuracy on transfer).
42. **Local Instruction Repairs vs Collateral Semantic Regressions (Arm NF)**:
    Targeted instruction changes can completely repair specific partial-information logical flaws (NF raising failure-with-missing eligibility from 34/48 to 48/48 correct, and 201 to 213 of 216 overall), yet introduce collateral regressions in unrelated fields (retries falling from 89 to 76 of 216). Consequently, whole-request all-six correctness drops from 33.33% to 28.70%. Instruction patches must be qualified at the complete-request level rather than accepted on isolated field gains.
43. **First-Digit Copying Traps in Multi-Digit Numeric Capping**:
    When prompted to map two-digit integers (10–26) into capped categories ($\le 3$), high-capacity causal models exhibit a severe lexical-copying failure: 100% of errors across 145 cases equal the first decimal digit of the input count (e.g. 21 yields 2 instead of 3). Host-side deterministic capping of extracted numbers or explicitly described distinct category symbols are necessary to bypass next-token digit copying.
44. **Chronological Recency Traps Under Inverted Display Order**:
    Presenting event histories in reverse-chronological ("newest-first") order causes severe systematic errors across both dense causal models and compact encoders: all four native Qwen arms and Indecis erroneously select the older event displayed second (36/36 case-state errors in Qwen, 216/216 in Indecis), despite explicit system text stating that the first displayed event is newest. Models exhibit strong presentation order bias that overrides explicit chronological instructions.
45. **Task-Specialized Pretrained Classifiers vs Small Multi-Task Schema Tuning**:
    Pretrained, task-specialized bidirectional models (released DeBERTa NLI) decisively outperform newly fitted multi-head encoders (90.10% accuracy at 20.92 ms median vs ~55% for BERT/ModernBERT and 86.98% for Qwen 9B). However, full fine-tuning on a small multi-task dataset (1,152 cases) produces severe majority-class collapse (predicting 100% ineligible and 100% retry=3), demonstrating that small fixed-schema fine-tuning is vulnerable to label imbalance and transfer failure.
46. **Dependency-Aware Policy Accounting vs Source Calibration Generalization**:
    Evaluating whole-request automation policies via source events (minimum-source or union-lower bounds) corrects naive probability-product errors, but cannot guarantee real-world safety when source probabilities fail calibration transfer. Gate-accepted temperatures from earlier rendering templates worsen NLL on fresh held-out layouts (e.g. NJ NLL worsening from 0.4666 to 0.7621), causing frozen acceptance policies to suffer unacceptably high error rates among accepted cases (losing to review-all).
