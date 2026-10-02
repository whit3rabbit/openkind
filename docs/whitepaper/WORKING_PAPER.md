# OpenKind: What the Evidence Supports for a Jev-Style Model and Inference Engine

**Working paper · Revision 0.9.3 · 1 October 2026 · America/Chicago**

**Research question:** Which ideas in [RESEARCH.md][R] should we actually pursue when building an accurate, low-latency, resource-conscious typed-decision model and runtime?

**Local evidence cutoff:** the completed encoder recovery run `20261001T223435_041738Z_f25616`, protocol `openkind-t4-recovery-reliability/v4.1.1` ([ER]), supplements the completed T4 integrated research study `20261001T105345_828556Z_6e6d92`, protocol `openkind-t4-integrated-research/v4.0.0` (E42), and the earlier results consolidated in [WHITEPAPER.md, version 0.8.11][W]. ER is an encoder-only recovery run; it does not replace E42's Qwen, caching or process-isolation evidence. External comparisons retain their previously reviewed, dated source scope. This document update performs no new model inference, training or benchmark timing.

## Abstract

The evidence supports building a Jev-style system by combining a capable pretrained model, a task-conditioned finite-answer readout, deterministic application logic and workload-dependent reuse of model computation. It does **not** yet support a universally superior custom architecture, a generally calibrated decision model, or an interchangeable fast execution mode.

The strongest integrated local results now provide measured answers to primary systems and readout questions: on 192 new authored policy cases, selective indexing combined with route-priority composition (Arm XR) runs as one timed path achieving **95.40% field accuracy** and **84.38% all-six correctness** at **242.98 ms median** (adding +40.1 pp whole-request correctness over X at 222.22 ms). A narrowly joint route/urgency readout (Arm NJ) avoids earlier collateral errors, supplying all six field distributions and obtaining **94.01% field accuracy** and **79.69% all-six correctness** at **228.31 ms**. Furthermore, multi-catalogue host snapshot caching cuts grouped request latency from ~1,511 ms to ~690 ms (**~54% latency reduction, 2.19x speedup ratio**) with zero probability drift ($\Delta p = 0.0$ across 96 paired runs) using ~170.6 MiB of host RAM. Separate resident native and JSON processes completely resolve mixed-request history drift ($\Delta p = 0.0$) across X, XR, and NJ with only a **3.9%–5.8% trace overhead** while fitting simultaneously on a 15-GiB T4 (12.10 GiB total VRAM). [W: §§21–23; E40–E42]

However, clear boundaries remain. All 23 E42 XR eligibility errors stem from a single structural pattern: unrecorded certification (`certified: null`) where a known failing points score ($< 70$) yields `undetermined` instead of `ineligible`. Dedicated process isolation does not eliminate prompt schema sensitivity (reordering, remapping, adding questions, or renaming visible keys alters probability vectors up to $\Delta p = 0.912$). The compact Go CPU model (Indecis) suffers severe transfer collapse on fresh policy templates (69.10% confirmation vs 91.15% dev). The later ER run removes E42's encoder execution blocker: all 17 scheduled stages executed, and fine-tuned BERT and ModernBERT reach **73.46%/70.45% field accuracy** but only **21.30%/19.44% all-six correctness** on 216 new authored policy cases, at **15.78/20.92 ms median**. This establishes a fast but still incomplete policy treatment, not a general rejection of encoders or a matched speedup over Qwen. [W: §§22–23; E41–E42; ER]

The practical direction is therefore **a pretrained, evidence-sensitive decision model with surgical selective indexing, deterministic route composition, explicit finite probabilities, multi-catalogue host snapshot caching, neural prediction only for genuinely semantic judgments, and resident process isolation for mixed traffic workloads**.

---

## 1. The answer: retain successful methods, not an assumed architecture

`RESEARCH.md` proposes a broad reproduction program and surveys leading systems. This paper answers that program through the experiments rather than restating the proposals as conclusions. The priorities below distinguish **retaining a demonstrated component** from **testing a promising next model**.

| Priority | Method to pursue | Positive evidence | Boundary on the recommendation |
|---|---|---|---|
| **Retain now** | Predict independent facts and derive declared dependent fields in code (Arm XR). | E39, E40, E41, and E42 improve field accuracy and eliminate contradictions; E42 measures selective indexing + route-priority composition (XR: 95.40% field, 84.38% all-six at 242.98 ms). | A correct rule can still receive a wrong predicted input. R and XR withhold an exact route distribution. |
| **Retain now** | Selective field indexing over blanket indexing. | E41 Arm X and E42 Arm XR index only route and case state, cutting latency to 222–243 ms and raising fresh accuracy to 92.19%–95.40%. | Blanket indexing of all fields fails; surgical indexing requires profiling field cardinality and branching. |
| **Retain now** | Multi-catalogue host snapshot caching for grouped requests. | E42 multi-catalogue host snapshot cache cuts grouped request time by ~54% (from ~1,511 to ~690 ms, 2.19x speedup) with zero drift ($\Delta p = 0.0$ on 96 pairs). | Requires ~170.6 MiB host RAM and explicit catalogue identity keys. |
| **Retain now** | Exact-prefix reuse with isolated, complete execution state. | FP32 candidate/state branching passes bounded parity tests; E40/E41 static-prefix pairs are exact ($\Delta p = 0.0$) and 2.34x faster. | Static catalogue caching and state-first question sharing are different mechanisms. Dynamic state prefill dominates long contexts. |
| **Retain now** | Resident process isolation for mixed native/JSON traffic. | E41 and E42 separate resident processes pass exact parity ($\Delta p = 0.0$) across D, X, XR, and NJ with ~4–6% trace overhead, fitting simultaneously in 12.10 GiB on T4. | Operational isolation does not diagnose or repair the underlying in-process mixed-state mechanism. Restarts incur a 3.6x–3.9x penalty. |
| **Retain as model baselines** | Task-conditioned learned heads and finite option-logit scoring, without free-form answer generation. | Strong frozen-feature NLI results; useful dynamic-candidate transfer; direct-logit ranking evidence; Qwen NLI_CODE achieves 92.71% accuracy and 63.5% automated coverage at cost 0.036; NJ reaches 94.01% field accuracy at 228 ms. | Neither readout dominates every task. A distribution over allowed codes does not automatically guarantee calibration or arbitrary-question independence. |
| **Discount under current recipe** | Compact fixed-head CPU models as general instruction followers. | Indecis fixed heads run fast on CPU with AVX2 assembly (46 ms median), but policy field accuracy drops from 91.15% to 69.10% on fresh templates; replay continuations cross zero. | High CPU throughput cannot compensate for representation collapse on unseen prompt structures. |
| **Retain as runtime discipline** | Version precision, kernels, rendering, batching, cache and request-history behavior together. | CPU/MLX parity assets identify reproducible modes; prompt schema sensitivity persists across all 4 transformations despite process isolation. | An efficient prefix-cached runtime does not inherently guarantee arbitrary-question statistical independence. |
| **Retain as scoped challengers** | Task-fitted BERT/ModernBERT and released task-appropriate encoders. | ER completed the recovery: fine-tuning improves new-authored policy field accuracy over frozen features by +22.15 pp for BERT and +18.06 pp for ModernBERT; released DeBERTa reaches 90.10% on exposed three-way SNLI. | Policy all-six correctness remains 21.30%/19.44%; GLiClass reaches 3.24%. Qwen was not rerun in ER, and exposed NLI is not unseen-policy evidence. |

**Source:** [W: §§4–10, 15–18, 20–23; E1–E7, E11–E12, E29–E32, E39–E42, RUSTM2; ER]. The recommendations synthesize those results; they are not claims that every component has already been integrated into one runtime.

The main architectural lesson is to **keep expensive question-conditioned interpretation where it is needed and remove redundant computation around it**. The record supports this more strongly than replacing the full decision computation with a cheap pooled state vector, indiscriminately compressing the cache, or assuming sparsity supplies a faster model. [W: §§9, 16–19, 23]

## 2. What counts as evidence in this paper

**Local measured** means the supplied whitepaper reports an executed OpenKind comparison. Its original sample, hardware, precision, prompt, selection rule and limitations still apply. This revision reviews those reported results; it does not independently reproduce the models or inherit earlier revisions' claims of having rerun their artifact checkers.

**External benchmark** means an evaluator measured a named system under a named protocol. **Author-reported** means the project's own model card, code or experiment supplies the claim. Both can motivate a comparison, but neither becomes an OpenKind measurement. **Post-hoc replay** means arithmetic applied to saved predictions, without a newly measured model request. **Proposed** means a design or experiment inferred from these findings, not a demonstrated improvement.

The disposition terms are correspondingly narrow: **retain** a demonstrated component; **conditional** when quality or execution trade-offs remain; **discount the tested form** when its intended objective failed; **unresolved** when the necessary comparison has not run. “Failed” does not mean every related architecture is impossible. “Passed” does not mean generalization, calibration and service reliability all passed.

Repeated requests and transformed options are not additional independent cases. In particular, E40's two technical repeats do not double its 96 continuity or 36 synthetic cases; the synthetic panel also shares three templates, and E41's two repeats likewise do not double its 96-case fresh policy or SNLI confirmation panels. E29–E32 reuse an exposed natural-document panel, including only 12 contracts and 12 papers at the calibration gate. These are useful development and diagnostic results, not an untouched deployment population. [W: §§3.3, 18.20–18.25, 21.1, 22]

ER separately labels old `u4_*` and regression panels as **exposed recovery/regression evidence**, and `r41_*` panels as **new authored confirmation/calibration evidence**. The policy confirmation contains 216 unique cases and 432 technical requests per profile. Its new fact groups and two held-out authored renderings do not constitute independent human confirmation or establish transfer to genuinely new policy families. Input coverage of 1.0 reports admitted cases, not a guarantee that every model retained every input token. The report, summary, manifest and completion record were read for this update; agreement among these generated outputs is not a new raw-prediction audit. [ER; ER-M; ER-S; ER-C]

## 3. What the benchmark leaders actually contribute

### 3.1 Preserve the dated comparison; do not make a permanent recipe from rank

The top-25 roster in `RESEARCH.md` is the **JevBench v1.4.2.2** snapshot scored on 27 September 2026. Its leading entries include Imajev, Plumb, Mapika decider, Jev, JevK5 and Cygnet. That composite combines intelligence, calibration, speed and cost; it is not a pure accuracy ranking or an ablation of model architecture. Preserve its public/sealed distinctions and configuration-specific notes. [R: “JevBench v1.4.2.2 top 25”; X0]

**External refresh:** the live board inspected for this revision displays **v1.5.0**, with 904 open and 720 sealed decisions and changed scoring. Cygnet and Winnow-12B Q8 lead its frozen official order with a reported statistical tie. Imajev and Plumb appear as separately reported addenda, outside that order. These are different benchmark conditions, not evidence that a model improved or deteriorated between versions. The older snapshot remains the basis for the roster crosswalk below. [X1]

This matters for the research conclusion: a frozen model with an effective readout is a serious comparator to a fine-tuned decision model. A leaderboard cannot by itself prove that decision LoRA, reinforcement learning, a dedicated head or a custom engine is necessary.

### 3.2 Technique-to-evidence crosswalk across the research roster

The rows group all 25 systems in `RESEARCH.md` by the questions they help answer. They are **method families, not a new ranking**. The historical roster supplies the coverage; the named primary-source checks refine selected examples.

| Systems from the historical top 25 | Technique worth investigating | What OpenKind's evidence says to do with it |
|---|---|---|
| **JevK5, Plumb, Hopper, reflex, metask, Open Spark Jev, Malkuth** | Qwen decision adaptation, finite option logits, varied supervision/replay and calibration. Plumb inherits JevK5 weights; this is not seven independent architecture replications. | Retain a frozen direct-logit control and test supervised adaptation against it. E30 establishes useful specialization, while E31/E32 require explicit retention checks. Do not treat the board's older reflex adapter as its newer frozen/order-averaged configuration. [R; X3–X5, X8] |
| **Imajev-4B, Mapika decider-4b v2, Jev-Omni** | Trained code/decision readouts or supervised option scoring, combined with backbone adaptation and richer task mixtures. These are not identical head architectures. | Compare a learned code head with allowed-vocabulary logits under a declared training contract. E1 supports learned readouts; E27/E30 support task-conditioned option scoring but expose rejection and retention limits. [R; X2, X6] |
| **Cygnet, SemIf, Jobe, local-jev, raw Qwen3-4B** | Frozen pretrained models used as finite-answer scorers, with differing prompts, calibration and runtimes. | Keep strong zero-training baselines. Cygnet demonstrates an external recipe using stock vLLM and one output token, not a newly trained decision network. SemIf supplies relevant hybrid-cache implementation ideas. [R; X7; W: §11.7] |
| **Winnow-12B Q8, jqv** | Shared-state prefill and independent question branches; Winnow also changes weights and quantization, while jqv is a stock-model inference design. | Borrow branch isolation and measure actual shared work. E12 supports long-state amortization, not uniformly flat question scaling. Do not attribute Winnow's complete result solely to its forked runtime. [R; X9; W: §16] |
| **OpenSourceJev Q4** | Quantized deployment plus constrained readout/calibration. | Measure memory and latency separately. E40 establishes one practical Q4/T4 profile; E39 shows that lower-bit weights need not be the faster operating point. [R; W: §§20–21] |
| **system-one-open** | A smaller Gemma backbone with decision LoRA. | Keep a smaller-model challenger, but select on useful quality at the target resource budget. E11 gives a positive 2B trade-off, not evidence for a universal sub-1.5B optimum. [R; W: §15] |
| **Mapika decider-35b-a3b** | A larger hybrid MoE with a decision-oriented training/readout contract. | Retain a quality/capacity challenger. E36's modern MoE is more accurate but approximately twice as slow as its dense comparator; it is not an expert-streaming result. [R; W: §19.2] |
| **djev** | Read constrained answer slots from a diffusion language model rather than a causal answer position. | Useful alternate mechanism, but OpenKind has no matched diffusion-versus-Qwen decision comparison. Do not infer a denoising advantage from “non-generative” or “parallel” terminology. [R; W: §2] |
| **Qwen3-Reranker-4B, ZeroEntropy zerank-2** | Repurpose relevance-trained cross-encoders for candidate decisions. | Test as separate task-transfer baselines, not as proof that relevance scores are calibrated decision probabilities. There is no local matched result establishing either as a replacement. [R] |
| **Jev, decision-machine-1** | Closed typed-decision services with externally observable behavior. | Compare quality, probabilities, latency and cost as systems. Their undisclosed internals do not supply a reproducible architecture or training recipe. [R; W: §2] |

**Identity correction:** `RESEARCH.md`'s top-25 row calls Imajev's starting checkpoint `Qwen3.5-4B-Base`, while its detailed review points to the J1 lineage. The current primary model card explicitly names **`Qwen/Qwen3.5-4B`, revision `851bf6e8`**, the post-trained checkpoint. This paper uses that identity and does not silently equate it with OpenKind's pinned Base integration reference. Its phase-3 adapter uses rank-64 LoRA and a learned 256-code readout, not merely a slice of unchanged language-model vocabulary logits. [R; X2]

### 3.3 The useful lessons from the leading recipes

**Hard cases and teacher distributions are credible experimental ingredients, not established independent causes of rank.** Plumb reports additional training on teacher-written and checked difficult decisions, including greater weight on uncertain or incorrect cases. Its public hard-set comparison is 89/111 versus JevK5's 82/111, but its disclosure says public aggregate results informed later recipe decisions. That is useful author-run evidence, not an untouched confirmation of a general hard-mining effect. [X3]

Winnow's author describes a refinement objective combining gold-label cross-entropy with teacher-distribution cross-entropy **only when the teacher agrees with gold**, alongside replay. That is a useful supervision-control idea. Its private training mixture and combined interventions prevent independently attributing the result to soft labels, LoRA, replay or runtime changes alone. [X9]

**Calibration can help in one setting and hurt in another.** Cygnet reports a held-out calibration-half ECE reduction from 0.140 to 0.101. That does not contradict OpenKind's temperature-scaling failures: the populations and probability engines differ. The right common method is a separate acceptance test, not always applying or always rejecting temperature scaling. Cygnet also aggregates multiple token IDs representing the same option letter, a concrete readout correctness detail worth testing in any tokenizer-dependent implementation. [X7]

**Training and retention recipes are workload-specific.** Mapika's later v2.1 card reports replacing hard-label replay with parent-distribution KL and changing temperature handling, recovering some sampled-play behavior while retaining most hard-decision gains. That is a different, author-reported recipe from the roster's v2 and from OpenKind's failed short-premise replay. It supports testing preservation on intended operations, not a universal claim that KL replay works or fails. [X6; W: §§18.23–18.26]

The result is a shortlist of experimental ingredients, not a mandate to copy every feature of the highest-ranked model.

## 4. Model design: what learned useful decisions, and what did not

### 4.1 Keep the decision-native path; do not assume one readout wins everywhere

A frozen Qwen3.5-4B-Base backbone with a small last-token linear head reached **87.67% matched and 87.33% mismatched MultiNLI accuracy** in Phase 2B. Phase 2C selected the same head family and reached **87.00% and 88.80%** on new sampled partitions. This is strong evidence for a simple learned readout over a capable representation, within NLI. It is not arbitrary-question competence. [W: §4; E1–E2]

In the initial comparison, mean-linear pooling reached 74.67% matched accuracy and max-linear 84.83%; the unadapted finite-code control reached 78.83%. A more complex head was not consistently necessary. However, the finite-code and learned-head paths differ in training and rendering, and these results do not establish that a learned head always beats a properly trained code readout. [W: §4.4]

Dynamic candidate scoring adds a different result. A shared scorer trained with 57 Banking77 labels reached 88.54% on seen-label and 82.29% on held-out-label sampled-choice episodes. Because it interprets candidate descriptions, it can score alternatives not represented by a fixed task-specific output neuron. The original cost, however, grows with candidate-conditioned continuations. [W: §5; E2]

Direct option logits deserve their own baseline. E27 improved conditional ContractNLI candidate ranking to **130/148**, versus 46/148 for its historical reference. Yet it found **0/124 semantic-none cases**, and its full-decision accuracy was only 130/272. QASPER answerability also remained below the answerable-majority baseline. **Better ranking is not a complete decision model.** [W: §18.17; E27]

**Three execution graphs must not be conflated.** A code-logit or code-head readout can score all offered codes at one answer position. The learned candidate scorer instead evaluates candidate-conditioned continuations. The native reader used in the E38–E40 line traverses permitted token-tree branches for finite field values; avoiding free-form JSON does not make that tree a one-forward classifier. Its distributions describe the locally constrained token policy, not unrestricted label-string likelihoods or calibrated correctness. Consequently, E40's speedup does not by itself validate the proposed one-position learned head. [W: §§11.2, 19.4, 20–21]

**Positive direction:** retain both a learned readout and a finite-code comparator. For a new fast model, the especially useful experiment is a joint-option request—state, question and all semantic alternatives—followed by either allowed-vocabulary logits or a learned code head. This can avoid separate candidate continuations, but its quality, probability space and execution cost must be measured; it is not already proven to dominate the immutable candidate-feature reference.

### 4.2 Decision LoRA worked as specialization, not as a general upgrade

E30 directly answers whether supervised decision adaptation can teach useful new behavior. Both Base and post-trained Qwen3.5-4B controls were evaluated with their adapted counterparts under the same BF16 execution profile and common joint-option renderer.

| E30 arm | ContractNLI accuracy, 204 questions | QASPER accuracy, 46 questions | ContractNLI entailment, correct / 84 |
|---|---:|---:|---:|
| J0: frozen Base | 56.86% | 86.96% | 81 |
| J2: Base + decision LoRA | **81.37%** | 76.09% | 73 |
| J1: frozen post-trained | 69.12% | 80.43% | 81 |
| J3: post-trained + decision LoRA | **82.35%** | 73.91% | 70 |

**Source:** [W: §18.21; E30]. These are matched within-run BF16 comparisons, not comparisons against E29's FP32 controls. The gate is exposed, document-clustered and small; both fits use one seed and select update 80 under the declared development rule.

J3 adds 27 correct ContractNLI decisions: 27 additional none decisions and 11 additional contradiction decisions, offset by 11 lost entailment decisions. Its identity-policy ContractNLI cost falls from 0.10539 to 0.08137 under the study's specified costs, with 38 accepted decisions and no observed errors. That is a bounded in-domain utility gain. QASPER policy cost instead becomes 0.11304, worse than review-all at 0.1, and both adapters fail the complete retention screen. [W: §18.21.3–§18.21.5]

The lesson is **not to abandon adaptation**. It is to stop selecting adaptations solely for aggregate task improvement. The training program should preserve class-sensitive and evidence-sensitive capabilities explicitly, include the frozen parent as a selectable outcome and evaluate the same complete operations expected in deployment.

No result here establishes rank 16, rank 64, a particular data mixture, or a specific loss coefficient as generally optimal. The much larger training programs in leading external models are recipes to test, not explanations for why an OpenKind pilot failed.

### 4.3 Replay taught its own task without preserving the intended scope

The parent-KL follow-on partially recovered probability quality relative to the specialists, but neither nonzero adapter met the joint preservation requirements. Replacing parent KL with source-label cross-entropy on the **same replay inputs** then improved SNLI from **125/192 to 159/192** and **131/192 to 166/192**. ContractNLI/QASPER probability scores worsened relative to the KL treatment, and supported-entailment/QASPER preservation still failed. Both guarded source-label selections retained the frozen parents. [W: §§18.23–18.26; E31–E32]

This separates three objectives: agreement with a parent, correctness on replay examples, and retention of the product's supported operations. None implies the others. A short-premise NLI replay panel was not sufficient evidence for preserving long-document answerability and entailment.

**Positive direction:** use reviewed hard examples and teacher distributions as controlled supervision additions only after checking the teacher on the target task. Keep genuine labels authoritative, measure teacher disagreement, and use replay that exercises the supported document and question operations. A new task-aligned preservation recipe remains proposed; the completed generic replay variants should not be described as untried fixes.

### 4.4 Evidence access deserves priority over another blind architecture sweep

The effective input—not the nominal source document—is what the model can use. In E29, QASPER questions retaining all recorded annotation strings increased from **12/119 to 79/119** when the prefix budget rose from 1,024 to 4,096 tokens. These are annotation-visibility counts, not a proof of semantic sufficiency. [W: §18.20.1]

On a fixed 32-question ContractNLI subset whose annotations became fully visible, post-trained J1 improved from **17/32 to 30/32**. At the same time, correct none decisions fell from **65/96 to 53/96**, so total ContractNLI correctness improved by only one. Aggregate accuracy almost hid the gain on newly visible evidence. The extra budget also adds other text, so the experiment does not isolate individual evidence spans as the sole cause. [W: §18.20.3]

E30 then lost five of the 30 correct decisions on that same evidence-sensitive subset after adaptation. Thus “use a larger context window” and “fine-tune on complete documents” are not sufficient retention strategies. [W: §18.21.4]

**Positive direction:** verify the final serialized/tokenized state, retain evidence-location diagnostics and train on admissible examples that require the intended evidence use. Preserve long-document and unsupported-answer slices during model selection. Retrieval, oracle windows, evidence-aware truncation and a learned evidence reader are separate possible interventions—not measured fixes supplied by the visibility audit.

### 4.5 A cheap state summary did not replace question-conditioned interpretation

`RESEARCH.md` proposes shared state memory queried by lightweight heads. The experiments distinguish three substantially different claims:

| Claim | Evidence-based answer |
|---|---|
| Reuse the complete state-prefill computation. | **Yes, within tested execution contracts.** Full hybrid-state branching preserves the chosen function and can amortize substantial work. |
| Use a frozen final-token/mean-pooled state vector plus cheap pooled question/candidate embeddings. | **The tested shortcut failed.** All three E26 arms fail the non-final gate and accept no decisions; the first two also lack a meaningful candidate ranker. |
| Train a richer token-level query/readout module over shared state. | **Partly tested, not successful end to end.** B1/B2 improve some aggregate discrimination but fail complete applicability/policy criteria. E26 does not test every richer reader. |

**Source:** [W: §§16, 18.3–18.8, 18.16; E12, E14–E20, E26].

The supported near-term architecture therefore retains full question/option interaction. A new compact state-query reader needs a stated difference in evidence representation, supervision or computation relative to the tested failures. “Prefill can be cached” is not evidence that its pooled final vector is a sufficient universal decision representation.

## 5. Output semantics: codes, rejection and calibration

### 5.1 Single-token codes are a serious design option, not a universal repair

PrivateMode and several leading systems motivate indexing semantic options and reading a finite answer distribution. This removes dependence on generating full natural-language labels, but the prompt must still convey their meanings and the code mapping must remain correct. PrivateMode's result is external evidence, not an OpenKind result. [X10; R]

E39 performed the relevant local readout comparison. It held checkpoint family and backend fixed within each precision, but changed output encoding and therefore rendering/token work.

| E39 readout | Q4 field accuracy / median ms | BF16 field accuracy / median ms |
|---|---:|---:|
| Natural labels | 78.47% / 111.13 | 78.56% / 79.97 |
| Integer indexes | 78.04% / 131.18 | 81.68% / 94.89 |
| Nonoverlapping aliases | 83.25% / 119.95 | 82.90% / 88.97 |
| Natural labels, five fields + action rule | **82.55% / 107.73** | **83.51% / 77.09** |
| Generated compact JSON | 90.45% / 494.99 | 88.89% / 578.76 |

**Source:** [W: §§20.2–20.3; E39]. Ninety-six exposed cases, two technical repeats; the study's interleaved process history remains a limitation on these comparisons.

Indexes improved some overlapping-label decisions but regressed integer/Boolean fields. They were **18.0%/18.7% slower uncached**, with a 508-token rather than 315-token static prefix. Their aggregate accuracy differences had intervals crossing zero. Aliases improved this panel, but semantic remapping still changed at least one answer on every tested remapping case in each precision/readout cell. Neither code choice nor randomized mapping established general invariance. [W: §20.2]

E40 narrowed the remaining issue: on 24 exposed diagnostic cases, separate field-only requests improved route accuracy from 50.00% to 87.50% and case-state accuracy from 66.67% to 100.00%. Selective indexes reached 54.17% and 87.50%. This supports investigating field formulation and catalogue interference, but changing the catalogue, prompt and workload together does not identify a tokenizer cause or supply a timed complete replacement. [W: §21.8.1]

**Order averaging is conditional too.** E28's order-averaged ranker gets 94/108 answerable ContractNLI choices correct, versus 93/108 in the original order, while the complete selected decision still fails rejection requirements. The extra evaluation work therefore needs a measured benefit, rather than being adopted as free robustness. A four-order author result must also not be substituted for a one-order leaderboard configuration. [W: §18.18; R: Imajev and reflex reviews]

**Positive direction:** prefer an explicit, validated codebook for a newly trained finite-answer model, while keeping natural labels and the existing candidate scorer as controls. Verify actual token IDs at the answer boundary, test code reassignment and option permutation separately, and measure whole-request cost. Selective readouts or field grouping deserve a bounded comparison; blanket conversion to indexes is not an established optimization.

### 5.2 Semantic none is learned behavior, not a spare code

Adding a `none` token ensures the output space can express no valid offered answer. It does not teach when that answer is correct. E27's zero ContractNLI none recall demonstrates this directly. E28's calibrated/order-averaged selection recovered none recall but falsely rejected **46/108 answerable ContractNLI questions**; its selected QASPER arm found **0/5 none cases**. [W: §§18.17–18.18]

The earlier seven-parameter set-aware none model is a useful positive ablation. It raised held-out Banking accuracy from 54.56% to 74.56%, but refitting a single constant already reached 72.56%; the set-aware increment over that stronger control was only 2.00 points. Held-out false-none rose from 3.50% to 18.00%. Retain the cheap controls and the trade-off, rather than crediting all improvement to a sophisticated rejection architecture. [W: §5.3; E3]

For training and evaluation, keep four notions distinct: an omitted correct option, insufficient evidence, unfamiliar/out-of-scope input and an application choice to request review. `Noul` is a binary primitive, not the semantic-none class. The mixture percentages proposed in `RESEARCH.md` remain unvalidated design settings, not a successful general recipe. [R; W: §§2, 5, 10]

### 5.3 Probability quality must survive the actual use case

The record contains several rejected temperature fits. Phase 2B matched NLL worsened from 0.3345 to 0.3392, and a native eligibility calibration changed test NLL from **0.27056 to 0.41902**. Positive scalar temperature does not change argmax, repair a logical contradiction or restore missing evidence. [W: §§4.4, 19.7]

Use NLL/cross-entropy as a supervised baseline; consider Brier or ranked-probability objectives in controlled ablations. The specific composite-loss coefficient, per-cardinality temperature and CADO/RL program proposed in `RESEARCH.md` have **not** been established as successful OpenKind recipes. A utility term can change the distribution-learning objective, so action optimization and probability estimation should not be treated as interchangeable. [R; W: §§6, 11–13; inference from the stated objectives]

Calibration acceptance needs held-out probability metrics and a separate policy assessment. At a minimum-field probability threshold of 0.95, E40 D still had **3 wrong among 10 accepted continuity requests** and **7 among 9 synthetic requests**. Minimum field probability is not the probability that the whole request is correct. Normalization, confidence concentration and zero observed errors on a small selected subset are not a deployment calibration guarantee. [W: §21.9]

## 6. Inference engine: the clearest quality-and-speed win

### 6.1 Deterministic composition and static-prefix reuse work together

E40 compares five complete paths on the same Qwen3.5-9B Q4/T4 profile. A predicts six fields with prefix lookup disabled; B predicts five and derives action; C caches A's static prefix; D caches B's static prefix. E generates compact JSON in a separate session.

| Arm | Continuity field accuracy | Continuity all-six correct | Continuity median ms | Synthetic field accuracy | Synthetic all-six correct | Synthetic median ms |
|---|---:|---:|---:|---:|---:|---:|
| A: six fields, recompute | 80.73% | 22.92% | 798.68 | 83.33% | 30.56% | 807.34 |
| B: five fields + action rule, recompute | 86.11% | 32.29% | 763.67 | 86.11% | 33.33% | 784.77 |
| C: six fields, warm prefix | 80.73% | 22.92% | 357.94 | 83.33% | 30.56% | 374.98 |
| **D: five fields + action rule, warm prefix** | **86.11%** | **32.29%** | **339.38** | **86.11%** | **33.33%** | **352.47** |
| E: generated JSON | 92.01% | 56.25% | 3,910.68 | 92.13% | 55.56% | 3,986.14 |

**Source:** [W: §21.3; E40]. Continuity has 96 cases; synthetic has 36. Times are per request, exclude startup and must not be divided into a claimed independent-field latency. Technical repeats do not increase the independent sample count.

The comparison answers two different questions. **B versus A** measures a reformulated five-field request plus the declared action rule: accuracy improves and median time falls modestly. **D versus B** measures warm static-prefix reuse of that five-field path: accuracy is unchanged and reported median time falls by 55.56%. The combined **D versus A** comparison improves field accuracy by 5.38 points and lowers median time by 57.51%. On the synthetic panel, the corresponding gain is 2.78 points and time reduction 56.34%. Ratios and reductions here are calculated from the displayed source medians, not new timing measurements. [W: §§21.3–21.4]

The recorded case-bootstrap field-accuracy intervals for D−A are **[3.82, 7.12] points** on continuity and **[0.93, 5.09]** on synthetic. The source's paired **mean-latency** ratios D/B, 0.440 and 0.449, are different estimands from ratios of medians. Keep them separate. [W: §21.3]

This is the best-supported direction because it removes a redundant learned decision and repeated prefix work rather than asking the model to become cheaper by losing information.

### 6.2 Derive probabilities as well as labels

The action rule preserves the eligibility distribution through the declared deterministic mapping. It does not assign action confidence 1. For a deterministic mapping `a = g(z)`, the probability calculation is:

```text
P(action = a) = sum of P(z) over values z for which g(z) = a
```

This is ordinary probability aggregation, not an additional learned prediction. A wrong eligibility estimate can still produce a wrong action. If a rule depends on several uncertain inputs, their marginal distributions alone do not determine the exact joint output distribution; an independence assumption must not be introduced silently.

Removing the action field also changes the model input. E40's continuity gain includes 29 corrected action fields and two corrected eligibility fields, rather than postprocessing completely unchanged A predictions. E39 had already shown timed composition gains, but with slightly worse NLL. The evidence supports this particular method and rule contract, not an unconditional probability-quality improvement from every dependency reduction. [W: §§20.3, 21.4]

The related route-priority idea has since moved from replay to measurement. The earlier figures — applying the deterministic rule to **saved D predictions** for 93.75%/94.44% field accuracy and 62.50%/66.67% all-six correctness — remain **post-hoc replay results only**. E41 then timed the rule as a real arm: R raises fresh-template field accuracy to 91.67% (+9.03 points) at unchanged median latency, and grouped GR reaches 95.49% field accuracy and 83.33% all-six correctness at 1,419.57 ms. Because base-route and urgency marginals do not determine their joint distribution, R and GR intentionally withhold an exact route vector rather than multiplying marginals; their full six-field NLL/Brier scores are unavailable, and that is the honest contract. E42 subsequently unified selective indexing and route composition into Arm XR: on 192 new authored policy cases, XR achieves **95.40% field accuracy** and **84.38% all-six correctness** at **242.98 ms median**, retaining both latency and accuracy gains. [W: §§21.8.2, 22.1, 23.1; V9–V11; E41–E42]

Joint outputs are the complementary lesson. In E41, Arm J derived route and urgency from one explicit four-way joint source, reconciling probabilities exactly with no independence assumption but costing retries accuracy (100% → 72.92%). E42 corrected this with Arm NJ (narrowly joint route/urgency), which keeps all six field distributions intact, repairs the retries regression, and achieves **94.01% field accuracy** and **79.69% all-six correctness** at **228.31 ms median**. Deriving dependent probabilities requires an appropriate source distribution, and a carefully targeted joint readout can supply valid distributions without collateral accuracy damage. [W: §§22.1, 23.1; E41–E42]

### 6.3 The faster path is not an equal-quality substitute for JSON

D is about 11.52× faster than E by continuity-panel median ratio, but JSON has substantially higher field and all-six correctness. The comparison is a speed/quality trade-off, not equivalent output at a lower cost. D's remaining errors are concentrated in priority-route suffixes and plain-closed versus closed-duplicate decisions; eligibility, action, retries and urgency are correct on both primary panels. [W: §§21.3, 21.8–21.9]

E41 narrowed this gap from the grouped side: GR reached **83.33% fresh all-six correctness at 1,419.57 ms — 2.06× faster than JSON (2,930.54 ms)**, but with zero prefix-cache hits. E42 directly solved this efficiency bottleneck by introducing a multi-catalogue host snapshot cache (~170.6 MiB host RAM): full-request latency dropped from **1,511.02 ms to 690.16 ms (a ~54% latency cut, 2.19× speedup ratio)** with **$\Delta p = 0.0$** across 96 paired runs. Combined with Arm XR (242.98 ms, 84.38% all-six), the native pipeline now substantially out-executes generated JSON (3,634.01 ms, 46.35% all-six in E42) in both speed and correctness. [W: §§22.1, 23.1, 23.3; E41–E42]

This identifies a better next step than immediately replacing the backbone: confirm a focused dependency/readout intervention on new cases, time the resulting complete request, and retain JSON as the quality control. A JSON fallback may be useful, but a reliable routing policy and safe mixed-mode execution are not established by this comparison.

## 7. Shared computation: when it was faster and when it was not

### 7.1 Three kinds of reuse must remain distinct

**Static-prefix caching** reuses instructions and a field catalogue. This is E39/E40's measured native serving optimization. **State-first sharing** reuses a particular dynamic state before isolated question continuations. This is the E11/E12 and immutable Rust/MLX reference design. **Persistent reuse** retains an exact prefix across requests and introduces admission, expiry, eviction and identity requirements. A win for one mechanism does not qualify the other two. [W: §§8–10, 16–17, 20–21]

E40 supplies **330 exact dedicated cache pairs** and **528 exact isolated quality-arm comparisons**. Its four fresh-process traces have zero-hit first requests. For the combined path, first request time is 878.19 ms, startup plus first trace point is 4.98 s, and the 16-request trace average is 359.62 ms. This is genuine first-use accounting, unlike E39's already-warm prime calls, but remains a repeated-anchor workload rather than a universal production break-even estimate. [W: §21.5]

### 7.2 The within-study comparisons support an adaptive scheduler

Each row below is a comparison **inside its named study**, not a cross-hardware speed ranking. “Cold” includes the prefix computation specified by that study; it does not mean cold model loading.

| Study and workload | Comparator | Alternative | Measured result | Disposition |
|---|---:|---:|---|---|
| E12, short semantic Q=4, three states | Repeated full: 869.3 ms | Warm nested batch: 962.4 ms | Sharing is **10.7% slower**. | Prefer the measured faster plan for comparable short shapes. |
| E12, synthetic L=256/Q=4/K=4 | Repeated full: 2,934.3 ms | Cold nested batch: 677.0 ms | Substantial saving even including prefill. | Positive mechanics evidence, not document accuracy. |
| E12, synthetic L=1,024/Q=4/K=4 | Repeated full: 9,448.1 ms | Cold: 1,101.2 ms; warm: 516.5 ms | **8.58× cold / 18.29× warm** median ratios. | Strong long-state amortization; warm excludes population. |
| E6, FP32 short request, K=2 | Full batch four: 147.2 ms | Shared batch four: 271.5 ms | Ordinary batching is faster. | Do not force cache branching for every request. |
| E6, FP32 short request, K=16 | Full batch four: 1,180.4 ms | Shared batch eight: 686.4 ms | Shared plan is **1.72× faster**. | Faster with a larger transient-memory requirement. |
| E7, strict-FP32 48-request expiry trace | No persistence: 22.0001 s | Lossless GPU LRU: 20.6376 s | **6.19% lower total service time**. | A locality-dependent persistence benefit. |
| E37, dense vLLM, 529-token prefix/Q=4 | 478.44 ms | Cold-staged: 248.95 ms | **1.92×**, passing this cell's gate. | Neighboring shapes and other profiles are not qualified by it. |
| E41, native catalogue mechanics, K=4, ~256 state tokens | Q=1: 447.32 ms | Q=16: 701.66 ms | Sixteen questions cost **1.57×** one question. | Sub-linear question amortization for catalogue work; the cache is schema-conditioned and cannot reuse dynamic state prefill. |
| E41, Indecis open options, K=4, cached option embeddings | Q=1: 17.60 ms | Q=16: 281.78 ms | **16.01×**, strictly linear; option caching still cuts Q=4 from 286.02 to 75.16 ms. | The open API re-encodes the input per question; it does not inherit the fixed-head or native "one-pass" claim. |

**Source:** [W: §§9.6, 10.8, 16.4–16.6, 19.3, 22.4; E6–E7, E12, E37, E41]. E12 is a Python/CUDA FP32 reference study; E6/E7 use the stated L4 stack; E37 is the separate modern-vLLM study; E41 scaling cells are the unified study's native-catalogue and Indecis mechanics fixtures. Synthetic token grids, natural semantic smoke cases and traffic traces answer different questions.

A scheduler should consider exact prefix length, Q, per-question K, suffix lengths and padding, cold/warm status, branch-state bytes, temporary expansion and the approved execution profile. The evidence supports **multiple plans and measured crossover**, not a universal “cache on,” “batch eight,” or fixed-K rule.

### 7.3 Fewer forwards and custom kernels were not automatically faster

The matched Mac flat-field diagnostic preserved the selected Qwen3.5-4B-Base FP32 readout and changed the traversal. It halved forward calls from 4 to 2 at Q2/K2 and 10 to 5 at Q8/K4. It nevertheless ran **6.0–12.1% slower** on the smaller shape and **64.3–64.5% slower** on the larger one. It repeated question tokens per candidate and increased padded work. These were prefill/continuation/readout timings on an M4 Max, excluding rendering, tokenization, transport and queueing. [W: §17.4; RUSTM2]

The separate cross-question candidate-pooling stage diagnostic was also slower, by 1.5% and 7.8% on its two shapes. A packed FP32 custom-Metal candidate passed parity but was **14–28% slower** than the reference operations in its same-host smoke sweep. Neither result justifies discarding vectorization or Metal; both reject the tested optimization as a faster replacement. [W: §§16.7, 17.4]

**External positive counterpoint:** JevK5 v0.2.0 reports about 13 ms with a CUDA graph per padded input length versus roughly 70 ms eager on an H100, with the same answers in that comparison. It also identifies optimized linear-attention kernels as a long-input acceleration. This is an author-reported implementation opportunity, not OpenKind timing or complete probability/history qualification. The local failure of a requested graph-disable intervention in E40 does not refute CUDA graphs as a speed technique; it means that intervention did not fix the observed history failure. [X4; W: §21.6]

**Positive direction:** optimize measured full-path cost, including repeated tokens, padding and branch materialization. Inspect optimized upstream kernels as candidates, but benchmark actual dispatch and renew parity checks. Do not copy an external forward-count reduction and assume its speedup transfers to a different backbone, readout and runtime.

## 8. Precision and memory: retain the narrow wins

### 8.1 Distinguish faster arithmetic from an interchangeable implementation

Strict-FP32 batching and lossless reuse repeatedly passed sampled equivalence tests for the older candidate scorer. Expanded E5's BF16 full-batch path instead changed a head's selected outcome in 14 of 128 episodes and a policy output in seven. Switching attention settings or selectively promoting modules did not solve the recorded BF16 shape drift. FP32 is therefore a useful reproducible reference, **not a generally proven semantic-accuracy winner**. [W: §7]

TF32-permitted full batching was approximately twice as fast in E7: 265.1→131.7 ms at K=4 and 1,118.3→518.5 ms at K=16. Yet the fresh comparison changed three head argmax outcomes across 416 episodes, and other context comparisons exceeded probability tolerance. It retains FP32 weight storage. Keep this as a separately versioned performance candidate, not a transparent speed or memory optimization. [W: §§10.3–10.4]

Similarly, E39's BF16 native path was faster than Q4 on its measured requests, while using more device memory and still exhibiting history-related answer drift. Neither “lower bits are always faster” nor “higher precision fixes the service” survives the local record. [W: §20]

### 8.2 Cache storage savings are not total-model savings

FP16 storage of full-attention KV reduced a measured 1,024-token FP32 hybrid root from **115 to 83 MiB**, a 27.83% reduction, with bounded parity passes. It did not reduce model weights, compress recurrent/convolution state, or establish faster serving. On the expiry trace it remained slower than lossless storage and did not add a cache hit. [W: §§9.5, 10.8]

All four tested low-bit TurboQuant snapshot configurations failed complete decision-equivalence gates. At short prefixes, recurrent state dominated storage, and codec tables could outweigh KV savings. These results discount those snapshot codecs and operating points, not every low-bit cache or fused low-bit attention implementation. [W: §§9.2–9.9]

Removing generation also does not necessarily remove vocabulary weights. The measured 4B reference ties its output projection to its input embedding, so omitting that output operation removes **zero marginal parameters**. This accounting is specific to that checkpoint; it must not be generalized to every model or to E40's separately inspected output-head configuration. [W: §§4.2, 21.1]

### 8.3 MoE is a capacity option; expert streaming remains untested

E36's Qwen3.5-35B-A3B GPTQ INT4 profile outperformed dense Qwen3.5-4B BF16 on the same two panels: **84.72% versus 75.35%** on SNLI and **92.71% versus 75.69%** on authored questions. Median request latency was about **126 ms versus 63 ms**. This is positive evidence for a stronger system profile at greater cost, not an isolated MoE effect: size, weights and quantization differ. [W: §19.2]

The older late-block-skipping probe reduced latency 2,022.49→1,535.78 ms but left peak allocation effectively unchanged at 7.823 GiB and failed decision utility. Another legacy probe touched an average **59.875 of 60 experts per layer** across a full input. Few active experts per token did not imply few experts for a request. No recorded experiment streams expert weights; a separate unrecovered physical-removal claim remains outside confirmed memory evidence. [W: §19.1]

**Positive direction:** keep modern MoE as a quality/teacher comparator, and measure request-wide expert locality before investing in a streaming engine. Expert residency, transfers, recurrent state and branch memory need actual accounting. Weight streaming is an open experiment—not a disproved idea, but not an evidence-backed optimization already available to this design.

E40 separately demonstrates that its publisher Q4 9B profile runs fully offloaded on a T4. A four-context workload samples 6.30 GiB whole-device peak on the reported 15.0-GiB device. This is useful feasibility evidence, not an exact allocation maximum, long-context/concurrency guarantee or Apple-Silicon memory estimate. [W: §21.10]

## 9. Runtime correctness: keep local wins without hiding service failures

Immediate replay and local cache equality are insufficient service tests. E40's native-only and no-op history controls pass, but **every planned JSON-interleaving condition fails** the unchanged probability/answer gate. Across planned blocks, end/reset comparisons fail 46/72; conditional replays bring the total to 64/90. Padding-off and requested CUDA-graph-disable do not repair the matched failures. Actual control graph capture was not measured, and no specific root cause is proven. [W: §21.6]

Eight-sequence one/four-context shape comparisons are exact in the tested configuration, but three sequences with four contexts produce a maximum probability difference of 0.058907. Unchanged answers do not override the probability gate. Conversely, E39 includes a BF16 answer flip below the 0.005 probability tolerance, so a probability threshold alone is also insufficient. [W: §§20.5, 21.7]

The existing native CPU service and pinned MLX FP32 parity results remain valid within their own frozen 4B integration profile. They do not qualify the 9B CUDA reader, its mixed-mode state or an adapted model. CPU load/soak evidence and MLX numerical qualification are not interchangeable service claims. [W: §17.3]

**Positive direction:** retain the demonstrated composition and cache components, but qualify their enclosing process. E41 established dedicated-process isolation as a measured remedy for the tested D-style shapes (passing reproduced failing traces at $\Delta p = 0.0$). E42 subsequently evaluated dedicated-process isolation across the newer readouts (X, XR, NJ): all three passed history checks with zero probability drift ($\Delta p = 0.0$), whereas mixed-process execution exhibited drift up to $\Delta p = 0.114$. What remains open is the underlying in-process state cause, whether same-process reset can be achieved without separate processes, and behavior under cancellation, concurrency, and long-running load. Acceptance must still cover probabilities, selected answers, policy outputs, request order, cancellation/recovery and the advertised concurrency/memory envelope—not only one repeated anchor. [W: §§21.6, 22.2, 23.4; E40–E42]

## 10. The architecture worth pursuing next

### 10.1 Preserve the working reference; investigate a cheaper joint-option path

The current immutable integration reference remains:

```text
Qwen3.5-4B-Base, profile a047d6802c3f06f085b8
state-first full hybrid root
    → isolated question continuation
    → isolated candidate continuations
    → learned candidate scores + semantic-none model
    → distribution + separate policy
```

Its CPU and pinned MLX FP32 evidence makes it a valuable differential reference. It need not be the final deployment model. [W: §§15–17]

The recommended **new-model/runtime integration hypothesis** is:

```text
Validated, versioned state and task contract
    → exact state tokens and immutable complete hybrid root, when reuse is eligible
    → isolated question + all described options
    → final answer-position representation
    → either selected vocabulary logits OR a trained finite-code head
    → explicit semantic distribution, including none where the task requires it
    → accepted calibration transform, if independently beneficial
    → deterministic dependency calculations and typed serialization
    → separately evaluated action/review policy
```

This combines individually supported ideas but is **not a claim that the full combination has run**. E29/E30's joint-option quality experiments use full forwards; E40's successful reuse is static catalogue caching; the immutable native reference uses candidate branches. A selected adapted joint-option model needs its own branching, precision, history and service qualification.

The reusable Qwen state must include attention KV, DeltaNet recurrence, convolution state and position information. Cache identity must bind the model/adapter, tokenizer, finalized token prefix, renderer, precision and backend. A block attention mask alone does not isolate a recurrent stream. Mutable branch state cannot be shared between independent questions. [W: §§8.1, 11.2–11.4, 16–17]

Keep question descriptions out of the supposedly question-independent state root. Where a static catalogue is intentionally part of the prefix, changing it changes cache identity and may change model behavior. Do not advertise that efficient schema-conditioned path as equivalent to arbitrary-question isolation. [W: §§11.7.1, 20–21]

### 10.2 Choose model size after choosing useful quality

The record does not establish the original 0.3B–1.5B proposal as optimal. E11's exploratory state-first 2B LoRA reaches **91.56%**, versus 95.00% for the selected 4B profile, with reported peak allocation **7.10 versus 15.80 GiB** and nonfinal Q≤4 p95 **1,523 versus 3,840 ms**. This is a credible smaller-model trade-off, but the exploratory set contains easy constructed families; its natural MultiRC component is less favorable than the pooled headline. [R; W: §15]

The same study's frozen ModernBERT arm is far faster and smaller but reaches 51.56%; the full-update arm reaches 42.19%. That rejects those tested treatments as quality replacements, not every encoder or released Laya/Von checkpoint. An encoder's one batched call may also repeat state encoding for every question. [W: §15; R: Laya and Von reviews]

### 10.2.1 Encoder recovery now supplies measurements, with a limited claim

E42's `SameFileError` remains part of that run's historical record. The later ER recovery executed all 17 scheduled encoder/setup/smoke stages with zero failed, blocked or pending stages. Its manifest enables encoders and GLiClass but disables native Qwen, cache and history experiments. `EXPLORATORY_COMPLETE` therefore closes the setup blocker for these treatments; it is not an accuracy target, general reliability result or promotion. Neither the model nor the service was promoted. [ER; ER-M; ER-C]

The following is a **within-run T4 policy comparison** on `r41_policy_confirmation`: 216 unique authored cases, each requested twice, with all requested cases admitted. The report retains the original U4 fitting examples and hyperparameters. Latencies are reported complete-request p50/p95; they are not cold-start, queueing or production-service measurements. [ER; ER-M]

| Profile | Field accuracy | All-six correct | Request p50 / p95 | NLL / Brier |
|---|---:|---:|---:|---:|
| BERT frozen features | 51.31% | 5.09% | 16.51 / 18.05 ms | 0.950 / 0.557 |
| BERT fine-tuned | 73.46% | 21.30% | 15.78 / 16.77 ms | 0.755 / 0.423 |
| ModernBERT frozen features | 52.39% | 0.00% | 18.65 / 27.42 ms | 0.949 / 0.555 |
| ModernBERT fine-tuned | 70.45% | 19.44% | 20.92 / 27.45 ms | 0.777 / 0.457 |
| GLiClass released dynamic-label treatment | 45.45% | 3.24% | 65.13 / 96.90 ms | 1.045 / 0.619 |

**What worked:** the tested fine-tuning recipes improve field accuracy over their frozen-feature controls by **22.15 pp (BERT)** and **18.06 pp (ModernBERT)**, with better all-six correctness and proper scores. BERT fine-tuning has the strongest descriptive point result of these five treatments. **What remains inadequate:** even it answers all six fields correctly on only 46/216 cases; ModernBERT achieves 42/216 and GLiClass 7/216. Complete probability availability and action consistency do not turn those answers into reliable whole requests. [ER; ER-S]

**Faster/slower:** within this panel, fine-tuned ModernBERT's median is 1.33× BERT's and GLiClass's is 4.13× BERT's, calculated from the reported medians. BERT's small measured latency difference from its frozen control is descriptive, not evidence that fine-tuning inherently accelerates an encoder. These are task/recipe comparisons, not architecture-only ablations. E42's XR/NJ numbers remain useful historical references, but its 192-case panel differs from ER's 216-case panel and Qwen did not run in ER; dividing those latencies would not establish a matched encoder-versus-Qwen speedup. [ER; ER-M; W §23.1]

**A separate positive NLI result:** the released three-way DeBERTa treatment achieves **90.10% (173/192)** on `u4_snli_confirmation`, with **20.92 ms p50 / 33.13 ms p95**, NLL **0.368** and Brier **0.166**. This is explicitly exposed recovery/regression evidence, with 384 technical requests. It justifies retaining a task-appropriate released NLI comparator; it does not establish caller-defined policy following, fresh NLI generalization or independent deployment-risk certification. GLiClass's original dynamic fixtures likewise remain exposed diagnostics. [ER]

For the current project, keep the 4B Base profile as the implementation reference and the measured 9B Q4 profile as a separate serving candidate. A matched 4B/9B comparison has not established a size winner. Bring a 2B student or alternative backbone forward only under a declared quality floor and measured target-machine resource benefit. External Cygnet/Winnow results justify a strong non-Qwen control when resources permit, not an automatic migration.

### 10.3 What to borrow from the general-model survey—and what not to borrow

`RESEARCH.md` also surveys general-purpose frontier and open models. Those model rankings do not answer the finite-decision question. Their useful ideas must be translated into this graph:

| General-model idea | Disposition for OpenKind |
|---|---|
| Post-training, difficult-example supervision and distillation | **Relevant.** Local adaptation learns a target task; external recipes motivate controlled teacher/replay additions. Retained correctness remains the condition. |
| Long-context and hybrid attention efficiency | **Relevant to execution and evidence access.** Preserve complete hybrid state and measure actual input coverage and branch cost. No local sparse-attention replacement has been validated. |
| Sparse experts and weight streaming | **Separate capacity/memory experiments.** Modern MoE can improve quality at greater latency; streaming has not been measured. |
| Long reasoning traces or additional inference effort | **Potential teacher or fallback tools.** They are not evidence that the main atomic decision path should regain a long autoregressive loop. No local benefit is established for that change. |
| Multi-token prediction/speculative decoding | **Not applicable to accelerating the existing zero-output decision readout.** It has no answer-token sequence to speculate through. E40 stores an auxiliary MTP block but does not enable MTP speculation. |
| Diffusion answer canvases or multimodal decisions | **Unresolved alternate workloads/architectures.** Their external existence does not provide a matched local quality/speed win or validate text-only results for images. |
| Reproducible data, model and runtime identities | **Retain.** The existing evidence repeatedly shows that configuration differences change the function being measured. |

**Source:** [R: “Twenty-five leading general-purpose models”; W: §§9.9, 16–19, 21.1]. These are implications for OpenKind, not a fresh technical audit of every general model named in the research survey.

## 11. The next experiments the evidence actually justifies

The next stage should resolve the few obstacles between the measured components and a useful integrated system, not restart an unrestricted architecture tournament. The experiments below are recommendations; no new run or implementation is claimed.

| Order | Proposed experiment | Why this is justified | Evidence required to advance |
|---|---|---|---|
| **1. Multi-catalogue grouped caching & selective composition** | E42 executed this priority: Arm XR combines selective indexing with route-priority composition (95.40% field, 84.38% all-six at 242.98 ms), and multi-catalogue host snapshot caching cuts grouped latency by ~54% (from ~1,511 to ~690 ms, 2.19x speedup, $\Delta p = 0.0$ across 96 pairs). The next step is evaluating host-to-device eviction and cold-start amortization under fluctuating traffic. | Grouped readouts previously suffered zero prefix hits in E41; E42 proves host RAM snapshot caching eliminates the penalty. XR establishes low-latency composition. | Formal lifecycle and cache residency tests under variable traffic and memory budgets; confirm host-to-device reloading latency. |
| **2. Test whether a task-fitted encoder meets a useful policy floor** | ER removes the setup blocker and shows useful fine-tuning gains, but only 21.30% BERT / 19.44% ModernBERT all-six correctness. Compare the recovered frozen/fine-tuned controls with fixed Qwen XR/NJ on the same admissible cases; keep released DeBERTa in a separate NLI lane. | The next discriminator is quality and selective usefulness under matched inputs, not another availability rerun or an unrestricted model search. | Locked development/calibration/confirmation splits, new task families distinct from authored wording changes, paired whole-request correctness and risk/coverage, actual input-token coverage, complete latency/memory and repeated-seed evidence. |
| **3. Decision learning with task-aligned retention** | Use admissible training/development records to diagnose supported-entailment errors, then compare one bounded preservation treatment against frozen and specialist controls. E41 strengthens the motivation: the Indecis compact recipe's fresh-policy NLL nearly doubles after accepted calibration, and replay's NLI gain is uncertain. | LoRA learns useful contradiction/none behavior; generic short-premise replay fails to retain document operations in both the Qwen and compact lines. | Class-, source-, evidence-location- and question-family results; separate probability and policy metrics; frozen parent remains selectable. New confirmation must not reuse inspected errors as a fresh holdout. |
| **4. Match readout to runtime** | Compare selected vocabulary logits and a learned code head under explicitly controlled training/rendering; port only a useful selected profile into shared-state execution. | Leading systems support both methods; local evidence does not settle their matched quality/resource trade-off. | Correct code/option mappings, meaningful none behavior, Q/K scaling and paired full-request measurements; adapter-specific branching and history qualification. |
| **5. Reduce deployment cost** | Compare a smaller Qwen/student or a frozen alternative after a useful task point exists; evaluate approved batching, exact reuse and storage formats on the target machine. | E11 supports a real smaller-model trade-off; E12/RUSTM2 show that cost depends on the actual graph and hardware. E41's native (1.57× at Q=16) versus open-mode (16×) scaling contrast shows the plan must come from the actual mechanism. | Retained useful quality, measured memory/latency benefit, and complete execution gates. Distillation additionally needs validated teacher quality. |

**Bounded next model experiment (proposed):**

1. Freeze a fixed-schema lane containing the recovered BERT and ModernBERT frozen/fine-tuned controls plus the selected Qwen XR and NJ controls. Reuse the already inspected ER panels only as exposed regression diagnostics. Keep DeBERTa's released NLI treatment in its own lane; do not interpret an NLI score as policy competence.
2. Define genuinely new policy/question families before fitting, separately from new facts or paraphrases of the old policy. Independently review labels and decisive evidence where feasible, and otherwise continue to call the panel authored. Split by source/fact group and keep related renderings together. Freeze the family mix, training budget, seeds, tuning rule and confirmation set before inspecting outcomes.
3. Admit common decisive content and candidate descriptions across models, record each tokenizer's actual retained evidence and exclusions, and lock checkpoint, pooling, label IDs, precision and renderer. Use multiple declared training seeds for fitted heads/backbones; a single-seed point ordering is not a stable model ranking.
4. Report per-field and all-six correctness, class-sensitive errors, NLL/Brier, rejection, and a development-selected review policy's confirmation risk/coverage. Fit calibration only on calibration-fit data, accept it on the gate, and never choose a treatment or threshold using confirmation. Complete field marginals are not a joint posterior; do not multiply dependent-field confidences or describe model-implied bounds as certified risk.
5. Advance an encoder only if it satisfies a **predeclared application quality/risk floor** and has a measured complete-request cost or memory benefit on shared cases. Report paired uncertainty using independent source groups, with technical repeats averaged first. If no treatment qualifies, record the bounded negative result and target its demonstrated errors; do not repeat the startup repair or declare the encoder family rejected. Cache eviction, process-history root cause, Rust/MLX and service qualification remain separate work.

[ER; ER-M; question register OQ-09–11, OQ-17, OQ-25–26, OQ-33–38, OQ-51]

These work items can proceed as separate model-quality and systems tracks. A model does not need to reproduce an older model's mistakes; an engine claiming to implement an unchanged model must reproduce its declared behavior. Neither distinction licenses post-hoc reselection or relaxation of historical acceptance rules. [W: §13.2]

## 12. Direct answers to the original research hypotheses

| Idea or assumption in `RESEARCH.md` | Answer supported by the record | Method disposition |
|---|---|---|
| Useful typed decisions require a newly invented neural architecture. | Existing pretrained models with learned or finite-token readouts already make useful bounded decisions. | **Do not require architectural novelty.** |
| A small learned head is sufficient. | Sufficient for the tested NLI and some dynamic-choice work, not general document applicability. | **Retain baseline; scope its claim.** |
| A shared-state/query architecture is the main path. | Complete-state reuse works; pooled shortcuts fail; richer readers have not passed complete quality gates. | **Retain reuse, not the unproven representation shortcut.** |
| One-token option codes are always better than natural labels. | Local effects vary by field and precision; uncached indexing was slower in E39. | **Controlled readout comparison.** |
| Independent field prediction is enough. | Derived action improves quality; remaining priority-route contradictions show the limit. | **Compute declared dependencies outside the model.** |
| More questions should add almost no latency. | Strong long-state mechanics gains coexist with a slower short-state shared path. | **Measure marginal Q cost; use adaptive plans.** |
| Fewer forwards or custom kernels guarantee speed. | Flat-field and the tested custom-Metal path are slower despite their mechanical changes. | **Reject proxy-only optimization.** |
| BF16 is a stable default and lower bits are faster. | Multiple precision/shape/history tests fail; BF16 can be faster than Q4 on a given stack. | **Version and test each operating point.** |
| Compressing KV solves memory use. | FP16-KV has bounded storage benefits; recurrent state and weights dominate other workloads. | **Retain measured accounting; discount failed low-bit codecs.** |
| Few active experts imply a small resident model. | Request-wide expert use can be broad; skipping did not unload weights; streaming is untested. | **Separate compute, residency and transfer experiments.** |
| More context solves missing evidence. | Visibility-sensitive decisions improve, but rejection can worsen and adaptation can lose those gains. | **Preserve evidence and test its use.** |
| Teacher/parent replay guarantees retention. | Parent agreement, replay-task learning and retained document correctness diverge. | **Use task-aligned retention, not proxy success.** |
| Explicit unknown plus temperature scaling yields calibrated decisions. | None recall, false-none and high-confidence errors remain; temperature fits can worsen held-out scores. | **Train semantics; accept calibration separately.** |
| RLCD/CADO is necessary. | No local result isolates an incremental benefit from decision-utility RL over supervised controls. | **Unresolved; not the next default investment.** |
| A 0.3B–1.5B encoder should be the initial winner. | ER measures fast, improved task-fitted encoders, but policy all-six correctness remains low; released DeBERTa is strong on exposed NLI. No matched Qwen replacement or general family verdict follows. | **Select by declared task quality and measured complete cost.** |
| MTP will accelerate the no-generation path. | No output sequence exists for that mechanism to accelerate; no local MTP result was measured. | **Do not prioritize for this graph.** |
| Passing a leaderboard or cache test qualifies a Jev replacement. | Benchmark quality, semantic scope, execution parity and service behavior have distinct failure modes. | **Require their separate evidence.** |

**Evidence map:** readout and heads—W §§4–5, 18.17–18.21, 20, 23; representations—§§16, 18.3–18.16; training/retention—§§18.20–18.26; systems/memory—§§7–10, 16–17, 19–23; hypothesis framing—R's architecture, training and evaluation sections. This table supersedes the *recommendation status* of earlier proposals in this working paper, not the original experiment records in R or W.

## Conclusion

The positive evidence is substantial enough to choose a direction. **Use a capable pretrained model to interpret the state, question and alternatives; expose a finite decision distribution; move deterministic dependencies into code; and reuse exact computation when the measured workload makes reuse worthwhile.** Preserve evidence-sensitive and class-sensitive behavior during adaptation, and treat the runtime configuration as part of the model's observable function.

The most persuasive local integrated results are now action composition plus static-prefix reuse, E42's selective indexing plus route composition (Arm XR: 95.40% field accuracy, 84.38% all-six correctness at 242.98 ms median), multi-catalogue host snapshot caching (~54% latency cut, 2.19× speedup, $\Delta p = 0.0$), and dedicated-process isolation of the reproduced history failure ($\Delta p = 0.0$ across X, XR, and NJ)—not a novel latent architecture, universal one-token encoding, aggressive cache compression or expert streaming. The strongest learning result is supervised task specialization—not a demonstrated general retention solution, and E41's compact comparator shows that fast CPU execution does not substitute for transfer. The strongest systems result is workload-dependent exact reuse—not universally flat question scaling: native catalogue work amortizes to 1.57× at Q=16 while the compact open-mode path scales linearly, and multi-catalogue host caching eliminates grouped catalogue stalls. [W: §§16–23]

For a new Jev-style model and engine, pursue these demonstrated components first, then test the missing joins: a useful joint-option model, faithful shared-state execution, a matched quality/risk test of the recovered encoders, an isolation qualification of the selected fast readout, and a schema-independent question interface. That is a positive engineering program grounded in what worked, while keeping the failed approaches and untested claims visible.

---

## Sources and provenance

### Local result authority

**[W]** `WHITEPAPER.md`, version 0.8.11. It contains the full methods, historical evidence register, reported audit scopes and frozen identities. Bracketed references such as `[W: §23.1; E42]` identify its section and experiment ID, not a new experiment in this revision.

| Evidence IDs | Relevant whitepaper sections | What this paper uses them for |
|---|---|---|
| E1–E3 | §§4–5 | Frozen heads, dynamic candidates and none-head comparisons. |
| E4–E7 | §§6–10 | Policy, arithmetic, exact reuse, storage and persistence trade-offs. |
| E9–E12 | §§13.1, 15–16 | Transfer/selection limits, smaller models and multi-question sharing. |
| RUST1–RUST11, RUSTM1–RUSTM2 | §§16.7–17.4 | Native/MLX qualification and slower flat-field/custom-kernel alternatives. |
| E14–E28 | §§18.1–18.19 | Natural-document applicability, audit limits, pooled-state and option-logit tests. |
| E29–E32 | §§18.20–18.26 | Evidence visibility, matched LoRA and replay/retention comparisons. |
| E33–E38 | §19 | Legacy/modern MoE, vLLM cache boundaries and native readout/calibration. |
| E39 | §20 | Matched readout, measured action composition, caching and history. |
| E40; V9 | §21 | Actual 9B/T4 combined path, first use, field diagnostics and separate offline routing replay. |
| E41; V10 | §22 | Unified decision validity: selective readouts (X, R, GR), resident process isolation, multi-question scaling, and compact CPU transfer. |
| E42; V11 | §23 | OpenKind T4 integrated research: useful integrations (XR, NJ), multi-catalogue host cache, dedicated process history, and classifier comparison diagnostic. |

**[ER]** [Encoder recovery report][ER], run `20261001T223435_041738Z_f25616`, protocol `openkind-t4-recovery-reliability/v4.1.1`, completed 1 October 2026. New local, source-reported encoder measurements; evidence roles and sample sizes remain those in the report. Supporting [manifest][ER-M], [summary][ER-S] and [completion record][ER-C] establish the executed scope. The downloaded summary's SHA-256 matches the completion record; this is a file-integrity check, not a re-audit of predictions, labels, model weights or the result archive. ER is a source label for this follow-up, not a newly assigned whitepaper experiment ID.

**[R]** `RESEARCH.md`, supplied research synthesis, hypothesis program and dated external-model roster. Earlier proposals remain proposals unless local result evidence changes their disposition. Its general-purpose-model survey is not treated as a ranked decision-model benchmark.

**Editing base:** repository `docs/whitepaper/WORKING_PAPER.md`, revision 0.9.2, Git blob `d6d853f9285e93e984aa841c8ad56212b971471c`. This update preserves its E42 and earlier evidence and changes the encoder status, interpretation and next experiment using ER. Older supplied-input hashes below retain their historical role; they are not identities for this revision or the later recovery run.

**Review boundary:** published-table ratios were recalculated where explicitly labeled. Model inference, training, archive reconciliation, source-label adjudication, checkpoint hashing, protected-final access and new latency measurement were not performed. Historical checker counts and hash-verification claims remain those of their original records.

### External comparison sources

External benchmark outcomes are evaluator-reported; training recipes and local project results are author-reported. None has been rerun by this revision.

| ID | Source | Use and limit |
|---|---|---|
| X0 | [JevBench v1.4.2.2 snapshot][X0] | Dated board underlying R's top-25 roster; not the current protocol. |
| X1 | [Live JevBench page][X1], displaying v1.5.0; [aggregate artifact][X1a] | Separately labeled external refresh; changed suite/method and addendum status. Not a longitudinal model-quality comparison. |
| X2 | [Imajev-4B model card][X2]; [pinned phase-3 summary][X2a] | Correct starting-checkpoint identity and distinguish learned code readout from vocabulary slicing. Recipe is compound, not an ingredient ablation. |
| X3 | [Plumb-4B model card][X3] | Teacher-generated hard decisions, JevK5 weight lineage, temperature and public-development disclosure. |
| X4 | [JevK5 v0.2.0 README][X4] | Historical versioned finite-answer baseline. Newer server/model behavior must not be assigned to this board row. |
| X5 | [Hopper model card][X5] | Decision LoRA and answer-type calibration; source/weight licensing scope remains a separate deployment check. |
| X6 | [Mapika decider-4b model card][X6] | Distinguishes roster v2 from later v2.1 parent-distribution replay and calibration changes. |
| X7 | [Cygnet recipe][X7] | Frozen Gemma, stock vLLM, one-token readout, token-mass aggregation and author-held-out calibration example. |
| X8 | [reflex repository][X8] | Distinguishes historical LoRA row from later frozen/order-averaged configuration. |
| X9 | [Winnow-12B model card][X9] | LoRA/refinement and gold-agreeing teacher supervision; shared-state runtime. Private data limits independent reproduction. |
| X10 | [PrivateMode method article][X10]; [implementation][X10a] | Motivation for indexed option scoring; not a local speed, quality or calibration result. |

**Input identities retained for this revision:**

```text
WHITEPAPER(8).md
  sha256 2c432102ecf31e56e74cc92377577660d41cfd68695559c9905a9394ff460278
WORKING_PAPER(4).md
  sha256 094d46c1d3045f1e20102f3d083f53a9c1797478cf43fcd4bcf4e42d00c4b999
RESEARCH.md
  sha256 eb2377a8a222424dbb0c8f25ce47919aab16092cc4934fca2572379dd30616cb
```

These are hashes of the three supplied Markdown inputs, not hashes or renewed audits of their referenced models and result archives.

[W]: WHITEPAPER.md
[R]: RESEARCH.md
[X0]: https://benchmarkheaven.com/jev-models/v1.4.2.2
[X1]: https://benchmarkheaven.com/jev-models
[X1a]: https://benchmarkheaven.com/api/jevbench/v1.5.0
[X2]: https://huggingface.co/mohit67890/imajev-4b
[X2a]: https://github.com/mohit67890/imajev/blob/6ee8a2c555ca6a3d1de9eceb33f1bd1cfeb268a2/model-cards/imajev-4b.md
[X3]: https://huggingface.co/crh225/plumb-4b
[X4]: https://github.com/allebee/jevk5/tree/v0.2.0
[X5]: https://huggingface.co/HopitAI/hopper
[X6]: https://huggingface.co/Mapika/decider-4b
[X7]: https://github.com/blockbrain-ai/cygnet-recipe
[X8]: https://github.com/kshetrajna12/reflex
[X9]: https://huggingface.co/EldanRing/Winnow-12B
[X10]: https://www.privatemode.ai/blog/system-one-from-glm-flash
[X10a]: https://github.com/edgelesssys/privatemode-decisions

[ER]: https://drive.google.com/file/d/1fpZNrhNLue6q55FYNH1AKrf7Pv1u4iHR/view
[ER-M]: https://drive.google.com/file/d/1X448XWfV7bnSvUAk5f0Sna-zS441tiiX/view
[ER-S]: https://drive.google.com/file/d/1iVivey4fVpG9SlWTmTel3vuUX2gZ4EH6/view
[ER-C]: https://drive.google.com/file/d/1qxUWUtqE2Zvr6TTLj_AwDB4E7pASy68w/view
