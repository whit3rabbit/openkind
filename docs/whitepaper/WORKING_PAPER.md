# OpenKind: What the Evidence Supports for a Jev-Style Model and Inference Engine

**Working paper · Revision 1.0.0 · 9 October 2026 · America/Chicago**

**Research question:** Which ideas in [RESEARCH.md][R] should we actually pursue when building an accurate, low-latency, resource-conscious typed-decision model and runtime?

**Local evidence cutoff:** completed **local-decision V13A**, `studies_v13a_checked_teacher/study_3e9da0121f40f75e/execution_c33f8407e52d09f4`, together with the V4–V12B notebook results and the earlier E1–E44/native CPU/MLX evidence carried forward from revision 0.9.3. **V13B is implemented; its results are not reviewed.** No V13B result is inferred from V13A or from its code. ([LD-V13A][]; [LD-V13-PLAN][])

**Scope and naming.** `LD-V…` denotes the newer local-decision training notebooks. These labels are **not** the older WHITEPAPER `V9–V12` aliases for E40–E44. The current architecture decision supersedes earlier future-work recommendations; it does not rewrite historical measurements or promote a model. The older native 4B-Base candidate scorer, 9B Q4 catalogue engine, and new post-trained 2B/4B joint-option LoRAs remain different systems.

This paper draws on already reviewed result/replay bundles and saved summaries/identity records. It adds **no model inference, training, new runtime timing, source-label adjudication, current-leaderboard refresh, or reserved-panel evaluation**. Summary values are reconciled with saved result files where available; that document check is not another raw-result re-audit. External material below remains dated to its original review unless explicitly identified as the later local Strands comparison.

## Abstract

The evidence now supports a specific **research mainline**: a post-trained Qwen text decoder, ordinary LoRA, one complete state–question–options forward, and a compact matrix copied from pretrained answer-code output rows. The mainline does not generate text, add a recurrent reader or require a learned pointer. **Adapted 2B is the resource-efficient development target; adapted 4B remains the stronger overall quality reference. No model through LD-V13A is qualified for deployment.** ([LD-V12B][]; [LD-V13A][])

LD-V12B supplies the controlled readout/size comparison previously missing from this paper. Fixed-code adaptation raises 2B source-macro accuracy from **47.28% to 69.89%**; its pointer alternative reaches **68.21%**. At 4B, fixed code reaches **76.41%**, versus **72.25%** for the replacement pointer. The pointer is genuinely trained jointly with the torso’s LoRA, but receives 800 extra initialization presentations, so the result concerns declared recipes rather than identical total optimization. On the same warm T4 workload, trained fixed-code 2B/4B medians are **168.07/291.68 ms**, with about **44.5% less recorded allocated CUDA memory** for 2B (1.688 versus 3.044 GiB). These are not Mac measurements. [LD-V12B]

The preceding mechanism tests constrain complexity. LD-V9’s first head iteration supplies **98.3%** of the raw-NLL gain observed at four iterations without a subsequent accuracy benefit. LD-V10 detects no useful gain from the extra token-memory attention operation under its frozen-feature recipe; a simpler pointer is better than those heads but below the adapted Qwen reference. LD-V12A finds exact observed compact/headless parity, yet physically removing an already bypassed LM-head module adds only about **0.11–0.23%** paired complete-request improvement and no meaningful live weight-memory saving. The expensive work remains in the decoder. ([LD-V9][]; [LD-V10]; [LD-V12A][])

Training improvements are real but scope-limited. SST-5 equal-class replay reduces extreme-class collapse; light original-parent preservation restores supported sentiment recall without solving unprotected none behavior. LD-V11’s verified curriculum improves transfer **82/144 → 94/144** versus its replay control but does not demonstrate a composition or retention improvement. LD-V13A continuation raises the 2B model to **73.15%** without new teacher guidance. Adding checked 4B guidance yields **73.28%**, slightly better cross-fitted NLL but exactly the same **672/928 correct**, with local MNLI/MultiRC gains and a new SST-5 positive-class regression. Four checked Plumb-none training opportunities exist, yet development none recall remains **1/10**. The added teacher term is therefore **not a validated default**. ([LD-V5]; [LD-V7]; [LD-V11][]; [LD-V13A][])

The older engine evidence remains useful in its own profiles: deterministic dependency composition and selective indexing, exact catalogue/state or input-encoding reuse when the workload supports it, and resident-process isolation of tested history failures. These components are **not already integrated and qualified for the new adapters**. The immediate next decision is to review LD-V13B’s independent identical-augmentation versus consistency comparison, then confirm a promising locked recipe on another seed and genuinely fresh evidence. Do not combine interventions, reopen resolved audits, relax failed thresholds or claim RLCD/publication readiness from a higher aggregate score.

## 1. The answer: retain successful methods, not an assumed architecture

`RESEARCH.md` proposes a broad reproduction program and surveys leading systems. This paper answers that program through the experiments rather than restating the proposals as conclusions. The priorities below distinguish **retaining a demonstrated component** from **testing a promising next model**.

### 1.1 Current local-decision choices: frozen for research, not certified for deployment

| Disposition | Choice | Positive or negative evidence | Decision boundary |
|---|---|---|---|
| **Default research architecture** | Joint-option Qwen + ordinary LoRA + compact frozen code rows | LD-V12B favors fixed code clearly at 4B and on point estimates only at 2B; no generated answer loop is needed. | Keep pointers as documented alternatives, not an active grid without a new hypothesis. |
| **Default size roles** | 2B resource candidate; 4B quality reference | +22.61 macro points after 2B adaptation; measured 2B resource advantage survives LoRA. | All absolute quality and presentation requirements still apply. |
| **Retain in the matched mixture** | Equal-class SST-5 replay | LD-V5 very-positive recall 7/32 → 23/32 against natural-sampling control. | A scoped, single-seed effect; not general sentiment or order invariance. |
| **Retain as a preservation control** | Original-parent KL 0.25 on trusted parent-correct training examples | LD-V7 restores very-positive recall to 25/32 and passes all five sentiment-class checks. | No direct coverage on excluded Banking77/Plumb-none examples; no complete-model qualification. |
| **Retain as a runtime simplification** | Compact fixed code matrix; optional stripped wrapper | LD-V12A output parity passes for these paths. | No material end-to-end speed or live-memory win from deleting the unused head. |
| **Mixed, not an adopted default** | Checked trained-4B guidance | LD-V13A: lower cross-fit NLL, no net extra correct primary answers, positive-sentiment regression. | Do not promote by counting fewer rejection messages. |
| **Mixed, not an adopted default** | 18% verified-curriculum replacement | LD-V11 improves transfer but not composition/retention as a whole. | Preserve verified examples; do not assume the allocation is optimal or universally helpful. |
| **Stop this tested form** | More recurrent passes / additional token-memory head | LD-V9/V10 add cost without a useful incremental capability result. | Not a rejection of all recurrent/backbone-latent architectures. |
| **Implemented; result unresolved** | Aligned JS consistency on identical augmented views | LD-V13B is independent of A and starts from the same V12B checkpoint. | No result reviewed; JS 0.5 remains a hypothesis. |
| **Not established** | Combination, RLCD, qualified publisher or Mac port | No necessary combined/acceptance/platform evidence follows from A. | Require distinct experiments and lineage-aware loaders. |

These choices freeze the next comparison. Rank 16, alpha 32, learning rate `2e-5`, NF4/FP16, a 1,024-total-token cap and 150-update stages are **working experimental settings**, not proven universal optima. Exact entry hashes are in [whitepaper §25.3][W25-3]; the configuration is in [LOCAL_DECISION_DESIGN.md](LOCAL_DECISION_DESIGN.md). [LD-V13A-PLAN]

### 1.2 Historical engine and specialist components retained within scope

The following earlier priorities are retained from revision 0.9.3. They are not promises that the new 2B/4B notebook artifact already includes these runtime components.

| Priority | Method to pursue | Positive evidence | Boundary on the recommendation |
|---|---|---|---|
| **Retain now** | Predict independent facts and derive declared dependent fields in code (Arm XR). | E39, E40, E41, and E42 improve field accuracy and eliminate contradictions; E42 measures selective indexing + route-priority composition (XR: 95.40% field, 84.38% all-six at 242.98 ms). | A correct rule can still receive a wrong predicted input. R and XR withhold an exact route distribution. |
| **Retain now** | Selective field indexing over blanket indexing. | E41 Arm X and E42 Arm XR index only route and case state, cutting latency to 222–243 ms and raising fresh accuracy to 92.19%–95.40%. | Blanket indexing of all fields fails; surgical indexing requires profiling field cardinality and branching. |
| **Retain now** | Multi-catalogue host snapshot caching for grouped requests. | E42 multi-catalogue host snapshot cache cuts grouped request time by ~54% (from ~1,511 to ~690 ms, 2.19x speedup) with zero drift ($\Delta p = 0.0$ on 96 pairs). | Requires ~170.6 MiB host RAM and explicit catalogue identity keys. |
| **Retain now** | Exact-prefix reuse with isolated, complete execution state. | FP32 candidate/state branching passes bounded parity tests; E40/E41 static-prefix pairs are exact ($\Delta p = 0.0$) and 2.34x faster. | Static catalogue caching and state-first question sharing are different mechanisms. Dynamic state prefill dominates long contexts. |
| **Retain now** | Exact within-request input encoding reuse in sequence encoders. | E43: Indecis input reuse delivers 4.80× steady-request speedup (117.13 ms $\to$ 24.38 ms on R41 confirmation) with bitwise zero drift ($\Delta p = 0.0$) across 1,980 paired requests. | Faster execution does not repair deficient semantic representations; transfer accuracy remains 0.0% all-six. |
| **Retain now** | Resident process isolation for mixed native/JSON traffic. | E41, E42, and E43 separate resident processes pass exact parity ($\Delta p = 0.0$) across D, X, XR, and NJ with ~4–6% trace overhead, fitting simultaneously in 12.10 GiB on T4. | Operational isolation does not diagnose or repair the underlying in-process mixed-state mechanism. Restarts incur a 3.6x–3.9x penalty. |
| **Retain as specialist baseline** | Released DeBERTa NLI specialist. | E44: 90.10% accuracy (173/192) at 20.92 ms median on SNLI confirmation (vs Qwen 9B's 86.98% at 149.13 ms; ~7.13× latency ratio); calibrated NLL improves to 0.2972. | Dev-selected threshold accepts 74 with 2 wrong, failing to beat review-all at cost 0.1000 (mean cost 0.11354); requires task-specific acceptance qualification. |
| **Retain as model baselines** | Task-conditioned learned heads and finite option-logit scoring, without free-form answer generation. | Strong frozen-feature NLI results; useful dynamic-candidate transfer; direct-logit ranking evidence; Qwen NLI_CODE achieves 92.71% accuracy and 63.5% automated coverage at cost 0.036; NJ reaches 94.01% field accuracy at 228 ms. | Neither readout dominates every task. A distribution over allowed codes does not automatically guarantee calibration or arbitrary-question independence. |
| **Discount under that historical recipe** | Small fixed-schema encoder fine-tuning and GLiClass dynamic-label interface. | E44: Full fine-tuning of ModernBERT (70.45% field / 19.44% all-six) and BERT (73.46% field / 21.30% all-six) beats frozen backbones (~51–52%) but collapses into majority classes (predicting 100% ineligible and 100% retry=3). GLiClass yields 45.45% field acc at 65.13 ms and rejects long inputs. | Recipe of 1,152 cases suffers severe label imbalance; dynamic-label formulation fails policy reasoning. |
| **Discount under that historical recipe** | Compact fixed-head CPU models as general instruction followers. | Indecis fixed heads run fast on CPU with AVX2 assembly (24–46 ms median), but policy field accuracy drops from 91.15% to 69.10% on fresh templates and 0.0% all-six on R41; replay continuations cross zero. | High CPU throughput cannot compensate for representation collapse on unseen prompt structures. |
| **Decompose / investigate** | Failure-first instruction patch (Arm NF). | E43: NF repairs partial-information eligibility (48/48 correct on failure-with-missing vs NJ's 34/48), raising overall eligibility from 201 to 213 of 216. | Collateral retry regression drops retry correctness from 89 to 76 of 216; all-six correctness drops from 33.33% to 28.70%. Cannot be deployed as complete request. |
| **Retain as runtime discipline** | Version precision, kernels, rendering, batching, cache and request-history behavior together. | CPU/MLX parity assets identify reproducible modes; prompt schema sensitivity persists across all 4 transformations despite process isolation. | An efficient prefix-cached runtime does not inherently guarantee arbitrary-question statistical independence. |

**Source:** [W: §§4–10, 15–18, 20–24; E1–E7, E11–E12, E29–E32, E39–E44, RUSTM2; ER]. The recommendations synthesize those results; they are not claims that every component has already been integrated into one runtime.

The main architectural lesson is to **keep expensive question-conditioned interpretation where it is needed and remove redundant computation around it**. The record supports this more strongly than replacing the full decision computation with a cheap pooled state vector, indiscriminately compressing the cache, or assuming sparsity supplies a faster model. [W: §§9, 16–19, 23]

## 2. What counts as evidence in this paper

**Local measured** means an executed OpenKind comparison is reported in its named source. E1–E44/native results below are carried from revision 0.9.3’s whitepaper synthesis; LD results are supported by the saved notebook records and already completed reviews. Original sample, hardware, precision, prompt, selection rule and limitations still apply. This paper does not rerun the models or inherit another review’s claim to have executed its artifact checker.

**External benchmark** means an evaluator measured a named system under a named protocol. **Author-reported** means the project's own model card, code or experiment supplies the claim. Both can motivate a comparison, but neither becomes an OpenKind measurement. **Post-hoc replay** means arithmetic applied to saved predictions, without a newly measured model request. **Proposed** means a design or experiment inferred from these findings, not a demonstrated improvement.

The disposition terms are correspondingly narrow: **retain** a demonstrated component; **conditional** when quality or execution trade-offs remain; **discount the tested form** when its intended objective failed; **unresolved** when the necessary comparison has not run. “Failed” does not mean every related architecture is impossible. “Passed” does not mean generalization, calibration and service reliability all passed.

Repeated requests and transformed options are not additional independent cases. In particular, E40's two technical repeats do not double its 96 continuity or 36 synthetic cases; the synthetic panel also shares three templates, and E41's two repeats likewise do not double its 96-case fresh policy or SNLI confirmation panels. E29–E32 reuse an exposed natural-document panel, including only 12 contracts and 12 papers at the calibration gate. These are useful development and diagnostic results, not an untouched deployment population. [W: §§3.3, 18.20–18.25, 21.1, 22]

ER separately labels old `u4_*` and regression panels as **exposed recovery/regression evidence**, and `r41_*` panels as **new authored confirmation/calibration evidence**. The policy confirmation contains 216 unique cases and 432 technical requests per profile. Its new fact groups and two held-out authored renderings do not constitute independent human confirmation or establish transfer to genuinely new policy families. Input coverage of 1.0 reports admitted cases, not a guarantee that every model retained every input token. The r0.9.3 review reported reading the report, summary, manifest and completion record; agreement among these generated outputs is not a new raw-prediction audit. [ER; ER-M; ER-S; ER-C]


**Newer panel and lineage boundaries.** The typical LD-V5-and-later historical payload contains 960 rows, of which **928 are primary and 32 additional diagnostics**. Primary accuracy/NLL is equal-source macro, whereas many paired bootstrap deltas are row-weighted; do not substitute one estimand for the other. SST-5 contributes 160 primary examples, other sources 128 each. V11 transfer/composition panels have 144 questions across 24 groups; V10 probes have 96 across 12 groups. Technical repetitions, additional stages over the same 1,200 training examples and transformed views do not create new independent groups. ([LD-V12B][]; [LD-V13A][])

**Task conversion and labels.** LD Banking77 uses sampled supplied alternatives plus none, not the standard 77-way classification benchmark. MNLI neutral is mapped to `__none__`; this does not make neutral entailment equivalent to missing a valid supplied answer or lacking evidence. SST-5 has a deliberately balanced development class panel, and Plumb retains its teacher/source-label provenance. None of these development percentages is a JevBench score.

**Freeze is not promotion.** A frozen entry is a content-bound starting checkpoint, not an accepted model and not necessarily frozen LoRA parameters during continuation. A completed fit, a passed serialization check, nonzero teacher opportunity, a passed relative slice, or a cached prefix that matches its uncached reference answers different questions. V13A’s controlled incremental comparison is teacher versus replay, not teacher versus the older entry alone. V13B’s control is its identical augmentation arm, not A’s replay arm.

## 3. What the benchmark leaders actually contribute

### 3.1 Preserve the dated comparison; do not make a permanent recipe from rank

The top-25 roster in `RESEARCH.md` is the **JevBench v1.4.2.2** snapshot scored on 27 September 2026. Its leading entries include Imajev, Plumb, Mapika decider, Jev, JevK5 and Cygnet. That composite combines intelligence, calibration, speed and cost; it is not a pure accuracy ranking or an ablation of model architecture. Preserve its public/sealed distinctions and configuration-specific notes. [R: “JevBench v1.4.2.2 top 25”; X0]

**Historical external snapshot (recorded 1 October 2026, not refreshed since):** the board then inspected displayed **v1.5.0**, with 904 open and 720 sealed decisions and changed scoring. Cygnet and Winnow-12B Q8 lead its frozen official order with a reported statistical tie. Imajev and Plumb appear as separately reported addenda, outside that order. These are different benchmark conditions, not evidence that a model improved or deteriorated between versions. The older snapshot remains the basis for the roster crosswalk below. [X1]

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

**Identity correction:** `RESEARCH.md`'s top-25 row calls Imajev's starting checkpoint `Qwen3.5-4B-Base`, while its detailed review points to the J1 lineage. The primary model card recorded in that October 1 review explicitly names **`Qwen/Qwen3.5-4B`, revision `851bf6e8`**, the post-trained checkpoint. This paper uses that identity and does not silently equate it with OpenKind's pinned Base integration reference. Its phase-3 adapter uses rank-64 LoRA and a learned 256-code readout, not merely a slice of unchanged language-model vocabulary logits. [R; X2]

### 3.3 The useful lessons from the leading recipes

**Hard cases and teacher distributions are credible experimental ingredients, not established independent causes of rank.** Plumb reports additional training on teacher-written and checked difficult decisions, including greater weight on uncertain or incorrect cases. Its public hard-set comparison is 89/111 versus JevK5's 82/111, but its disclosure says public aggregate results informed later recipe decisions. That is useful author-run evidence, not an untouched confirmation of a general hard-mining effect. [X3]

Winnow's author describes a refinement objective combining gold-label cross-entropy with teacher-distribution cross-entropy **only when the teacher agrees with gold**, alongside replay. That is a useful supervision-control idea. Its private training mixture and combined interventions prevent independently attributing the result to soft labels, LoRA, replay or runtime changes alone. [X9]

**Calibration can help in one setting and hurt in another.** Cygnet reports a held-out calibration-half ECE reduction from 0.140 to 0.101. That does not contradict OpenKind's temperature-scaling failures: the populations and probability engines differ. The right common method is a separate acceptance test, not always applying or always rejecting temperature scaling. Cygnet also aggregates multiple token IDs representing the same option letter, a concrete readout correctness detail worth testing in any tokenizer-dependent implementation. [X7]

**Training and retention recipes are workload-specific.** Mapika's later v2.1 card reports replacing hard-label replay with parent-distribution KL and changing temperature handling, recovering some sampled-play behavior while retaining most hard-decision gains. That is a different, author-reported recipe from the roster's v2 and from OpenKind's failed short-premise replay. It supports testing preservation on intended operations, not a universal claim that KL replay works or fails. [X6; W: §§18.23–18.26]

The result is a shortlist of experimental ingredients, not a mandate to copy every feature of the highest-ranked model.

### 3.4 Strands: related design, useful comparator, not an imported winning recipe

The later Strands comparison adds a distinct pointer system to the research roster. The reviewed article introduced Hobson **v19**, whereas the locally executed reference in LD-V10 was **`StrandsAgents/strands-decider-2B-hobson-v21`**. Preserve that version difference. Its 2B-Base torso, separate learned pointer, native renderer and training history differ from the 4B OpenKind adapter; the local T4 evaluation used a documented FP16 adaptation of the released BF16 model. Potential public-source training overlap limits independence. ([LD-V10]; [LD-V10-GUIDE])

In LD-V10’s common development comparison, Strands reaches **71.90% macro accuracy**, versus **76.47% for the v7 4B LoRA**. Neither percentage isolates head architecture. Strands shows good Banking77 behavior but weak composed-policy and ordinal-probe behavior; selected constant outputs make some aggregate correct counts misleading. It is not an infallible teacher or a validated arbitrary-question engine. Its shared-prefix implementation supplies concrete engineering reference code, but only three of six local multi-question fixtures pass the declared complete probability/answer parity gate; see §7.4.

The learning lesson became testable in LD-V12B: **jointly train** a replacement pointer and the same-size torso LoRA against the fixed-code control. That actual local comparison, not an architectural diagram or 2B-versus-4B system rank, supports the current fixed-code default. Released Strands’ data, distillation and trained-checkpoint replay remain separate methodological precedents. V13A tests one checked-teacher mechanism and finds a mixed effect; it does not reproduce the full Strands recipe.

## 4. Model design: what learned useful decisions, and what did not

### 4.1 Keep the decision-native path; do not assume one readout wins everywhere

A frozen Qwen3.5-4B-Base backbone with a small last-token linear head reached **87.67% matched and 87.33% mismatched MultiNLI accuracy** in Phase 2B. Phase 2C selected the same head family and reached **87.00% and 88.80%** on new sampled partitions. This is strong evidence for a simple learned readout over a capable representation, within NLI. It is not arbitrary-question competence. [W: §4; E1–E2]

In the initial comparison, mean-linear pooling reached 74.67% matched accuracy and max-linear 84.83%; the unadapted finite-code control reached 78.83%. A more complex head was not consistently necessary. However, the finite-code and learned-head paths differ in training and rendering, and these results do not establish that a learned head always beats a properly trained code readout. [W: §4.4]

Dynamic candidate scoring adds a different result. A shared scorer trained with 57 Banking77 labels reached 88.54% on seen-label and 82.29% on held-out-label sampled-choice episodes. Because it interprets candidate descriptions, it can score alternatives not represented by a fixed task-specific output neuron. The original cost, however, grows with candidate-conditioned continuations. [W: §5; E2]

Direct option logits deserve their own baseline. E27 improved conditional ContractNLI candidate ranking to **130/148**, versus 46/148 for its historical reference. Yet it found **0/124 semantic-none cases**, and its full-decision accuracy was only 130/272. QASPER answerability also remained below the answerable-majority baseline. **Better ranking is not a complete decision model.** [W: §18.17; E27]

**Three execution graphs must not be conflated.** A code-logit or code-head readout can score all offered codes at one answer position. The learned candidate scorer instead evaluates candidate-conditioned continuations. The native reader used in the E38–E40 line traverses permitted token-tree branches for finite field values; avoiding free-form JSON does not make that tree a one-forward classifier. Its distributions describe the locally constrained token policy, not unrestricted label-string likelihoods or calibrated correctness. Consequently, E40's speedup does not by itself validate the proposed one-position learned head. [W: §§11.2, 19.4, 20–21]

**Updated disposition:** retain the older learned/candidate scorers as historical and task-specific controls. The joint-option readout experiment has now run in LD-V9–V12B; compact fixed codes are the current mainline within that newer recipe. This does not establish dominance over the older immutable candidate scorer under matched conditions, nor erase its CPU/MLX reference role. See §§4.6–4.9.

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

**Positive direction:** use reviewed hard examples and teacher distributions as controlled supervision additions only after checking the teacher on the target task. Keep genuine labels authoritative, measure teacher disagreement, and use replay that exercises the supported document and question operations. That historical recommendation motivated the later scoped experiments in §§4.6–4.11. LD-V7/V13A now supply measured preservation and checked-teacher results; further task-aligned supervision is still a proposal, and neither the old replay failures nor the new mixed results should be called untried fixes.

### 4.4 Evidence access deserves priority over another blind architecture sweep

The effective input, not the nominal source document, is what the model can use. In E29, QASPER questions retaining all recorded annotation strings increased from **12/119 to 79/119** when the prefix budget rose from 1,024 to 4,096 tokens. These are annotation-visibility counts, not a proof of semantic sufficiency. [W: §18.20.1]

On a fixed 32-question ContractNLI subset whose annotations became fully visible, post-trained J1 improved from **17/32 to 30/32**. At the same time, correct none decisions fell from **65/96 to 53/96**, so total ContractNLI correctness improved by only one. Aggregate accuracy almost hid the gain on newly visible evidence. The extra budget also adds other text, so the experiment does not isolate individual evidence spans as the sole cause. [W: §18.20.3]

E30 then lost five of the 30 correct decisions on that same evidence-sensitive subset after adaptation. Thus “use a larger context window” and “fine-tune on complete documents” are not sufficient retention strategies. [W: §18.21.4]

**Positive direction:** verify the final serialized/tokenized state, retain evidence-location diagnostics and train on admissible examples that require the intended evidence use. Preserve long-document and unsupported-answer slices during model selection. Retrieval, oracle windows, evidence-aware truncation and a learned evidence reader are separate possible interventions, not measured fixes supplied by the visibility audit.

### 4.5 A cheap state summary did not replace question-conditioned interpretation

`RESEARCH.md` proposes shared state memory queried by lightweight heads. The experiments distinguish three substantially different claims:

| Claim | Evidence-based answer |
|---|---|
| Reuse the complete state-prefill computation. | **Yes, within tested execution contracts.** Full hybrid-state branching preserves the chosen function and can amortize substantial work. |
| Use a frozen final-token/mean-pooled state vector plus cheap pooled question/candidate embeddings. | **The tested shortcut failed.** All three E26 arms fail the non-final gate and accept no decisions; the first two also lack a meaningful candidate ranker. |
| Train a richer token-level query/readout module over shared state. | **Partly tested, not successful end to end.** B1/B2 improve some aggregate discrimination but fail complete applicability/policy criteria. E26 does not test every richer reader. |

**Source:** [W: §§16, 18.3–18.8, 18.16; E12, E14–E20, E26].

The supported near-term architecture therefore retains full question/option interaction. A new compact state-query reader needs a stated difference in evidence representation, supervision or computation relative to the tested failures. “Prefill can be cached” is not evidence that its pooled final vector is a sufficient universal decision representation.

### 4.6 What the bounded local-decision pilots added: balancing and targeted preservation

LD-V4 trained adapters and improved development aggregates, but each evaluated candidate failed retention. Step-zero selection was a fallback, not a qualified newly trained model; the parent’s separate absolute false-none gate could fail too. Later protocols enlarge panels, save per-example predictions, align support with independent groups, and separate best-observed from best-eligible/exported checkpoints. Those repairs improve interpretability; they do not turn old failures into accepted models.

LD-V5 compares natural versus equal-class SST-5 sampling and canonical versus paired alternative presentations. At the fixed final step, the unbalanced control reaches **75.40% macro accuracy and 7/32 very-positive recall**; SST-balanced reaches **76.00% and 23/32**. Very-negative recall is **17/32** in the balanced arm, matching its parent. Natural replay supplies only 14 unique very-positive examples, while balanced replay supplies 30, within the same overall SST budget. This directly supports retaining equal-class exposure in the next matched recipe, not assuming “more updates” alone repairs retention. [LD-V5]

LD-V6 tests smaller updates and a limited none-pair intervention. The balanced `1e-5` arm reaches **74.73%**, below LD-V5’s `2e-5` reference at its matched 150-update budget, and still fails multiple checks. None pairs improve Banking77 true-none from **18/28 to 20/28** but add false-none errors **11/100 → 16/100**; Plumb true-none remains **1/10**. These results discount those interventions as new defaults without claiming that no longer schedule or better negative supervision could help. [LD-V6]

LD-V7 fixes learning rate and sampling and varies original-parent KL strength. Its λ=0.25 arm reaches **76.47% macro accuracy, 0.6365 raw NLL and 0.6318 cross-fitted NLL**, restores **25/32 very-positive recall** and passes all five SST-5 class checks. The λ=1 arm worsens the trade-off and adds MNLI false-none failure. The λ=0 CE arm has the best cross-fitted NLL (**0.6221**) but loses class retention. Consequently “best probability objective,” “best retention trade-off” and “accepted checkpoint” are distinct labels; **none is accepted**. [LD-V7]

The preservation mask matters: of 1,200 unique scheduled rows, **630 are eligible** under the v7 trust rule. All 37 Banking77-none and all 23 Plumb-none training examples receive no added preservation loss, though they remain in ordinary CE. They are precisely the slices that still fail. Stronger λ would not test preservation on absent mask coverage. These observations do not authorize anchoring uncertain or parent-wrong labels; §5.2 describes the subsequent audit.

### 4.7 Recurrent reading and token-memory attention: the tested additions did not earn their cost

LD-V9 uses frozen Qwen features and trained residual heads. `shared_1` reaches **69.15% accuracy, 0.767381 raw NLL and 0.747196 cross-fit NLL**; `shared_4` reaches **69.04%, 0.766166 and 0.746608**. The first pass accounts for **98.3% of the four-pass raw-NLL reduction** relative to the parent. Moving from one to four passes corrects three cases and introduces four across the full 960-row panel; the primary paired NLL interval includes zero. Warm request medians rise **288.0 → 344.8 ms**. All heads fail retention/schema checks. [LD-V9]

LD-V10 then supplies the previously missing simpler controls. Latent interaction **without** token-memory attention yields **69.24% accuracy / 0.7467 cross-fit NLL**, compared with **69.15% / 0.7472** for the one-pass memory head. The paired NLL interval crosses zero. A mean-option pointer instead reaches **71.38% / 0.7039** and outperforms the full memory head on the paired panel. These are frozen-feature residual readouts, **not** replacement pointers jointly trained with Qwen. The adapted v7 reference is still stronger at **76.47% / 0.6318** in the common comparison. [LD-V10]

**Disposition:** remove the tested recurrence and added memory attention from the mainline. Retain the evidence boundary: no result here rules out joint backbone latent training, a different state-only encoder, stronger intermediate supervision or another recurrent architecture. A pooled-feature control is not state-free; its Qwen representations already depend on the state and options. A simpler head’s quality/complexity advantage also does not establish a complete-request speedup when decoder cost dominates.

### 4.8 Verified curriculum: a transfer gain, not a validated blanket replacement

LD-V11 continues the v7 adapter and compares replay with a fixed **18% presentation replacement**: 432 of 2,400 presentations become 216 verified curriculum examples, each repeated twice. It protects existing SST-5, Plumb, rules and semantic-none exposure, but changes non-none exposure in other sources. Equal updates/presentation count is not equal source mixture or token budget. [LD-V11]

The replay control reaches **77.05% macro accuracy, 0.6421 raw NLL and 0.6188 cross-fit NLL**; the curriculum reaches **76.00%, 0.6376 and 0.6164**. Both fail retention. Curriculum transfer improves **82/144 → 94/144**, with a descriptive group-bootstrap interval for accuracy gain of about **+2.78 to +13.19 points**. Composition changes **80/144 → 76/144**, with an interval including zero. The panels contain 24 underlying groups each, not 144 independent reasoning families.

**Disposition:** keep the verified examples and targeted failure understanding, but do not adopt the 18% allocation as a demonstrated general upgrade or expand it blindly. V12B/V13 deliberately return to the frozen balanced replay to isolate new losses/readouts. The original V10 probe IDs are diagnostic-only, not training examples.

### 4.9 The trained 2B/4B × readout comparison now favors fixed code

LD-V12B’s four final joint-step-150 arms are tabulated in [whitepaper §25.1][W25-1]: fixed code reaches **69.89%** (2B) and **76.41%** (4B) macro accuracy, versus 68.21% and 72.25% for the replacement pointer.

Source: [LD-V12B]. The 928 primary historical rows exclude 32 additional diagnostics. Each size starts from a pinned **post-trained**, not Base, Qwen; within a size the two arms start with identical fresh LoRA tensors. Both have 2,400 joint presentations, while the random pointer receives another **800 head-only cached-feature presentations**. No parent-code residual remains in pointer scoring, and both pointer and LoRA update. It is genuine replacement-pointer training, but not a matched total-initialization-cost comparison or reproduction of Strands’ training corpus.

At 4B the pointer loses **40 correct primary answers** against fixed code, a row-weighted difference of **−4.31 points**, descriptive interval **[−6.69, −1.85]**. Raw NLL is also worse. At 2B, the pointer’s accuracy interval includes zero, but no overall quality/cost gain justifies replacement. The mean-head versus endpoint-head result from V10 is not a basis for ignoring this later, differently trained comparison.

The strongest positive result is **2B adaptation**: **47.28% → 69.89% macro accuracy**, 270 corrected parent errors versus 61 introduced errors, net **209 more correct primary decisions**. Trained fixed-code 2B uses about **1.688 GiB allocated CUDA memory**, compared with **3.044 GiB** for trained fixed-code 4B, and the 4B/2B median-time ratio is **1.74×** (291.68/168.07 ms) on this 28-request/five-repeat T4 workload. No Mac or production-service result follows. [LD-V12B]

None of the four candidates passes quality and schema. 2B fixed-code has MultiRC false recall **59/69 versus parent 64/69**, rules false-none **14/58**, and a Plumb cross-fit-NLL regression. Plumb true-none is only **1/10**, matching a weak 2B parent and therefore not producing a relative recall-regression flag. The 4B fixed-code arm still fails Banking77 and Plumb none recall. More aggregate quality does not supply an absolute reliability floor.

Transfer favors fixed codes as well: 2B fixed/pointer get **71/55 of 144 transfer** cases, 4B **88/58**; the corresponding composition counts are **72/74** and **89/61**. The 4B pointer predicts ordinal level `2` on every counting question in both balanced 24-case counting slices. Full end-to-end option-order sensitivity remains: fixed-code SST-5 flips **19/32 at 2B and 18/32 at 4B**. An option-equivariant readout is not proof of a presentation-invariant causal encoder.

### 4.10 V13A separates continuation from the incremental teacher effect

Both arms continue the same frozen V12B 2B fixed-code checkpoint. They keep the data/order, supervised loss and original-parent preservation fixed; only one adds **0.25 × 2² × checked trained-teacher KL at temperature 2**. The 4B teacher is evaluated on training rows, cached and unloaded before fitting. It is not consulted at inference. [LD-V13A-PLAN]

The entry, replay and checked-teacher results are tabulated in [whitepaper §25.2][W25-2]: replay reaches **73.15%** and the teacher arm 73.28% macro accuracy, both **672/928** correct versus 641/928 at entry.

Source: [LD-V13A]. Teacher versus control fixes **14** mistakes, introduces **14**, and changes four other wrong answers. The **+0.134 macro percentage-point** difference comes from equal-source weights and unequal row counts, not an extra correct answer. The row-weighted accuracy interval is about **[−1.11, +1.06] points**. Both continuations net 31 more correct decisions than entry; this supports that bounded continuation, not an indefinitely beneficial training extension.

The teacher improves MNLI **94/128 → 99/128** and MultiRC false recall **59/69 → 61/69**, but both arms retain only **37/44 entailment** cases against entry’s **40/44**. Teacher guidance loses positive SST-5 recall **15/32 → 9/32**; four of the six lost positives shift to very positive and two to neutral. Very-positive improvement does not compensate for adjacent-class damage. Both arms fix rules false-none **14/58 → 0/58** with true-none **45/45**; the teacher is not necessary for that decision gain.

The preparation audit found **775 eligible unique examples, with 113 teacher-correct/entry-student-wrong opportunities**. Only **four Plumb-none groups** qualify, giving eight guided presentations. Entry, control and teacher still achieve **1/10** true-none development recall. “Intervention supported” means the training mask is nonempty; it does not establish a generalized policy/none operation, and final in-sample success on those four cases was not established by the frozen-entry audit. [LD-V13A-AUDIT]

Transfer is **73/144 in both arms**; composition **65/144 versus 67/144**; V10 probes **73/96 versus 72/96**. Both have zero completely correct composition groups. Option-order changes fall **55/224 → 51/224** and code-assignment changes **39/224 → 29/224**, yet SST-5 order flips worsen **18/32 → 20/32** and neither passes absolute stability. Warm medians are **163.48/167.15 ms** with identical allocated memory; the small timing difference does not show architectural inference overhead from offline guidance.

**Disposition:** do not lock the new teacher term as a validated improvement, do not choose it merely because its rejection list is shorter, and do not increase its coefficient to compensate for missing semantic coverage. The simpler replay arm is the correct control. Both outputs are research-only, and the frozen entry for V13B is unchanged.

### 4.11 V13B is an implemented outstanding test, not a result inferred from A

The independent experiment starts from the original V12B student, never a V13A output. It changes each canonical duplicate into an equivalent actual re-rendered view: **600 examples with option-order changes and 600 with code reassignment**. Both arms use identical views and supervision; the experimental arm adds **0.5 × Jensen–Shannon divergence** after canonical-option alignment, with gradients through both fresh forwards. Per-view original-parent logits are recomputed. [LD-V13-PLAN]

The next required evidence is lower actual flip/probability variation without losing class/none capability or learning constant outputs. Its results are not reviewed. A teacher-plus-consistency interaction remains untested and is not justified merely because the two separate notebooks exist. Keep instruction/fact changes and removed-valid-option contrasts out of the invariance loss: their correct decisions can differ.

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

**The completed none audit resolves review status, not model capability.** LD-V8’s 60 training-only cases finish as **50 confirmed none and ten uncertain**, with zero pending. It is an AI-performed source review, not independent human adjudication. The original v8 preservation experiment stops at three Plumb groups against its predeclared four-group floor; no training outcome should be invented from that stop. Later V13A has four **different eligibility outcomes under the trained teacher**, passes its nonzero-opportunity preparation and completes, but still shows 1/10 Plumb-none recall. The masks/teacher differ; there is no contradiction in the counts. ([LD-V8][]; [LD-V13A-AUDIT][])

None judgments must establish that all retained alternatives are invalid, not merely that the annotated best answer was removed. Reviewed parent-wrong examples are appropriate for supervised correction, not preservation of the wrong parent prediction. The specific v6 omission-pair treatment changes willingness to select none more than it demonstrates improved discrimination; do not adopt a larger global omission rate as an evidence-backed remedy. [LD-V6]

### 5.3 Probability quality must survive the actual use case

The record contains several rejected temperature fits. Phase 2B matched NLL worsened from 0.3345 to 0.3392, and a native eligibility calibration changed test NLL from **0.27056 to 0.41902**. Positive scalar temperature does not change argmax, repair a logical contradiction or restore missing evidence. [W: §§4.4, 19.7]

Use NLL/cross-entropy as a supervised baseline; consider Brier or ranked-probability objectives in controlled ablations. The specific composite-loss coefficient, per-cardinality temperature and CADO/RL program proposed in `RESEARCH.md` have **not** been established as successful OpenKind recipes. A utility term can change the distribution-learning objective, so action optimization and probability estimation should not be treated as interchangeable. [R; W: §§6, 11–13; inference from the stated objectives]

Calibration acceptance needs held-out probability metrics and a separate policy assessment. At a minimum-field probability threshold of 0.95, E40 D still had **3 wrong among 10 accepted continuity requests** and **7 among 9 synthetic requests**. Minimum field probability is not the probability that the whole request is correct. Normalization, confidence concentration and zero observed errors on a small selected subset are not a deployment calibration guarantee. [W: §21.9]

**Later calibration evidence changes the development profile, not the old verdicts.** LD-V6’s saved-logit, group-cross-fitted replay shows that BoolQ/MultiRC NLL regressions can be largely probability-sharpness problems even when answer accuracy improves. LD-V7 therefore prospectively separates raw decision retention from request-kind cross-fitted NLL/Brier, applying the same procedure to parent and candidate. Five-fold group separation prevents fitting a scored group’s temperature on that group; repeated development selection still prevents an independent-acceptance claim. The older raw results and rejection decisions stay unchanged. ([LD-V6]; [LD-V7])

LD-V13A demonstrates why proper-score and decision evidence must both remain visible. Teacher raw macro NLL is worse than replay (**0.7034 versus 0.6970**) while cross-fitted macro NLL is better (**0.6852 versus 0.6907**); both have **672/928 correct**. MultiRC’s wrong-answer NLL tail worsens despite recovery of two false labels. A slightly better calibrated aggregate does not erase a lost sentiment class or establish a reliable application policy. Neither entropy nor max probability is automatically a probability of correctness; finite ordinal uncertainty also merits an explicitly tested spread-sensitive report. [LD-V13A]

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

The related route-priority idea has since moved from replay to measurement. The earlier figures (applying the deterministic rule to **saved D predictions** for 93.75%/94.44% field accuracy and 62.50%/66.67% all-six correctness) remain **post-hoc replay results only**. E41 then timed the rule as a real arm: R raises fresh-template field accuracy to 91.67% (+9.03 points) at unchanged median latency, and grouped GR reaches 95.49% field accuracy and 83.33% all-six correctness at 1,419.57 ms. Because base-route and urgency marginals do not determine their joint distribution, R and GR intentionally withhold an exact route vector rather than multiplying marginals; their full six-field NLL/Brier scores are unavailable, and that is the honest contract. E42 subsequently unified selective indexing and route composition into Arm XR: on 192 new authored policy cases, XR achieves **95.40% field accuracy** and **84.38% all-six correctness** at **242.98 ms median**, retaining both latency and accuracy gains. [W: §§21.8.2, 22.1, 23.1; V9–V11; E41–E42]

Joint outputs are the complementary lesson. In E41, Arm J derived route and urgency from one explicit four-way joint source, reconciling probabilities exactly with no independence assumption but costing retries accuracy (100% → 72.92%). E42 corrected this with Arm NJ (narrowly joint route/urgency), which keeps all six field distributions intact, repairs the retries regression, and achieves **94.01% field accuracy** and **79.69% all-six correctness** at **228.31 ms median**. Deriving dependent probabilities requires an appropriate source distribution, and a carefully targeted joint readout can supply valid distributions without collateral accuracy damage. [W: §§22.1, 23.1; E41–E42]

### 6.3 The faster path is not an equal-quality substitute for JSON

D is about 11.52× faster than E by continuity-panel median ratio, but JSON has substantially higher field and all-six correctness. The comparison is a speed/quality trade-off, not equivalent output at a lower cost. D's remaining errors are concentrated in priority-route suffixes and plain-closed versus closed-duplicate decisions; eligibility, action, retries and urgency are correct on both primary panels. [W: §§21.3, 21.8–21.9]

E41 narrowed this gap from the grouped side: GR reached **83.33% fresh all-six correctness at 1,419.57 ms, 2.06× faster than JSON (2,930.54 ms)**, but with zero prefix-cache hits. E42 directly solved this efficiency bottleneck by introducing a multi-catalogue host snapshot cache (~170.6 MiB host RAM): full-request latency dropped from **1,511.02 ms to 690.16 ms (a ~54% latency cut, 2.19× speedup ratio)** with **$\Delta p = 0.0$** across 96 paired runs. Combined with Arm XR (242.98 ms, 84.38% all-six), the native pipeline now substantially out-executes generated JSON (3,634.01 ms, 46.35% all-six in E42) in both speed and correctness. [W: §§22.1, 23.1, 23.3; E41–E42]

For that older catalogue-engine track, the implication remains to confirm focused dependency/readout interventions on new cases, time the complete request and retain JSON as a quality control; it is not a reason to restart the now-completed LD pointer/size comparison. A JSON fallback may be useful, but a reliable routing policy and safe mixed-mode execution are not established by this comparison.

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

### 7.4 The later Strands cache comparison is promising but only partially parity-qualified

LD-V10 evaluates native shared-prefix behavior separately from quality. Of six actual multi-question fixtures, **three pass** the declared probability/answer parity rule; three single-question fixtures do not invoke split-prefill sharing. On passing medium/long-prefix fixtures, paired timing ratios versus full batching are **2.01×, 2.35× and 3.36×**. All recorded argmax comparisons pass, but maximum probability drift reaches **0.001456**, exceeding the **0.001** tolerance. That `0.001` is the stricter research replay tolerance; the repository acceptance tolerance is `0.005` ([documentation standards](../AGENTS.md#evidence--parity-standards)), so this drift fails the first and passes the second. Some ordinary serial/batched comparisons also exceed `0.001`. Numerical-path variation is a plausible interpretation, not a demonstrated root cause. [LD-V10]

This supports further precision/path and target-platform qualification, not “cache universally works” or “cache is broken.” Preserve sibling/order independence tests, near-tie and downstream-confidence tests, complete hybrid-state ownership and the original tolerance. These measurements do not qualify a shared-state engine for the new 2B/4B OpenKind LoRAs.

## 8. Precision and memory: retain the narrow wins

### 8.1 Distinguish faster arithmetic from an interchangeable implementation

Strict-FP32 batching and lossless reuse repeatedly passed sampled equivalence tests for the older candidate scorer. Expanded E5's BF16 full-batch path instead changed a head's selected outcome in 14 of 128 episodes and a policy output in seven. Switching attention settings or selectively promoting modules did not solve the recorded BF16 shape drift. FP32 is therefore a useful reproducible reference, **not a generally proven semantic-accuracy winner**. [W: §7]

TF32-permitted full batching was approximately twice as fast in E7: 265.1→131.7 ms at K=4 and 1,118.3→518.5 ms at K=16. Yet the fresh comparison changed three head argmax outcomes across 416 episodes, and other context comparisons exceeded probability tolerance. It retains FP32 weight storage. Keep this as a separately versioned performance candidate, not a transparent speed or memory optimization. [W: §§10.3–10.4]

Similarly, E39's BF16 native path was faster than Q4 on its measured requests, while using more device memory and still exhibiting history-related answer drift. Neither “lower bits are always faster” nor “higher precision fixes the service” survives the local record. [W: §20]

### 8.2 Cache storage savings are not total-model savings

FP16 storage of full-attention KV reduced a measured 1,024-token FP32 hybrid root from **115 to 83 MiB**, a 27.83% reduction, with bounded parity passes. It did not reduce model weights, compress recurrent/convolution state, or establish faster serving. On the expiry trace it remained slower than lossless storage and did not add a cache hit. [W: §§9.5, 10.8]

All four tested low-bit TurboQuant snapshot configurations failed complete decision-equivalence gates. At short prefixes, recurrent state dominated storage, and codec tables could outweigh KV savings. These results discount those snapshot codecs and operating points, not every low-bit cache or fused low-bit attention implementation. [W: §§9.2–9.9]

Removing generation also does not necessarily remove vocabulary weights. The measured 4B reference ties its output projection to its input embedding, so omitting that output operation removes **zero marginal parameters**. This accounting is specific to that checkpoint; it must not be generalized to every model or to E40's separately inspected output-head configuration. [W: §§4.2, 21.1]

### 8.2.1 Physical output-head removal has now been measured

LD-V12A compares final-position full vocabulary, per-request selected rows, a copied compact code table, and a genuinely stripped decoder-plus-table wrapper. Compact and stripped paths show zero observed probability difference and identical selected answers. The output module really is removed; the same decoder remains. [LD-V12A]

| System, same V12A timing grid | Restricted-row complete-request median | Stripped median | Paired restricted/stripped ratio |
|---|---:|---:|---:|
| Post-trained 2B | 181.96 ms | 181.34 ms | 1.0023× |
| Post-trained 4B | 433.95 ms | 433.25 ms | 1.0018× |
| 4B + v7 adapter | 527.74 ms | 527.11 ms | 1.0011× |

No complete-request comparison meets the predeclared 5% material-speed criterion. For adapted 4B, readout-only medians are **6.996 ms full vocabulary, 0.104 ms restricted rows and 0.0645 ms stripped compact**. Avoiding the vocabulary projection matters, but the current restricted implementation already avoids it. Speeding a roughly 0.1-ms readout does not materially accelerate a roughly 528-ms request.

Tied storage explains memory: adapted-4B allocated CUDA memory changes by only **512 bytes**, and unique registered storage by **128 bytes**. The larger allocator-**reserved** decrease is confounded by explicit cleanup in the stripped path but not equivalent cleanup in the intact path. It is not evidence of freed vocabulary-sized live weights. Future cleanup comparisons must be symmetric. Both forms load original weights first; direct headless disk loading/startup is unmeasured.

One 2B **full-vocabulary** case fails probability parity (**0.0018727** difference) with unchanged argmax; preferred compact/headless paths pass. Do not relax the tolerance. The grid has **33 admitted distinct requests**, nine repetitions and two process/order blocks on one T4. This is not a Mac result, and its timing denominators differ from V12B’s 28-request panel. **Close the repeat head-removal grid; retain compact scoring as a clean deployment function, not a new speed breakthrough.** [LD-V12A]

### 8.3 MoE is a capacity option; expert streaming remains untested

E36's Qwen3.5-35B-A3B GPTQ INT4 profile outperformed dense Qwen3.5-4B BF16 on the same two panels: **84.72% versus 75.35%** on SNLI and **92.71% versus 75.69%** on authored questions. Median request latency was about **126 ms versus 63 ms**. This is positive evidence for a stronger system profile at greater cost, not an isolated MoE effect: size, weights and quantization differ. [W: §19.2]

The older late-block-skipping probe reduced latency 2,022.49→1,535.78 ms but left peak allocation effectively unchanged at 7.823 GiB and failed decision utility. Another legacy probe touched an average **59.875 of 60 experts per layer** across a full input. Few active experts per token did not imply few experts for a request. No recorded experiment streams expert weights; a separate unrecovered physical-removal claim remains outside confirmed memory evidence. [W: §19.1]

**Positive direction:** keep modern MoE as a quality/teacher comparator, and measure request-wide expert locality before investing in a streaming engine. Expert residency, transfers, recurrent state and branch memory need actual accounting. Weight streaming is an open experiment: not a disproved idea, but not an evidence-backed optimization already available to this design.

E40 separately demonstrates that its publisher Q4 9B profile runs fully offloaded on a T4. A four-context workload samples 6.30 GiB whole-device peak on the reported 15.0-GiB device. This is useful feasibility evidence, not an exact allocation maximum, long-context/concurrency guarantee or Apple-Silicon memory estimate. [W: §21.10]

## 9. Runtime correctness: keep local wins without hiding service failures

Immediate replay and local cache equality are insufficient service tests. E40's native-only and no-op history controls pass, but **every planned JSON-interleaving condition fails** the unchanged probability/answer gate. Across planned blocks, end/reset comparisons fail 46/72; conditional replays bring the total to 64/90. Padding-off and requested CUDA-graph-disable do not repair the matched failures. Actual control graph capture was not measured, and no specific root cause is proven. [W: §21.6]

Eight-sequence one/four-context shape comparisons are exact in the tested configuration, but three sequences with four contexts produce a maximum probability difference of 0.058907. Unchanged answers do not override the probability gate. Conversely, E39 includes a BF16 answer flip below the 0.005 probability tolerance, so a probability threshold alone is also insufficient. [W: §§20.5, 21.7]

The existing native CPU service and pinned MLX FP32 parity results remain valid within their own frozen 4B integration profile. They do not qualify the 9B CUDA reader, its mixed-mode state or an adapted model. CPU load/soak evidence and MLX numerical qualification are not interchangeable service claims. [W: §17.3]

**Positive direction:** retain the demonstrated composition and cache components, but qualify their enclosing process. E41 established dedicated-process isolation as a measured remedy for the tested D-style shapes (passing reproduced failing traces at $\Delta p = 0.0$). E42 subsequently evaluated dedicated-process isolation across the newer readouts (X, XR, NJ): all three passed history checks with zero probability drift ($\Delta p = 0.0$), whereas mixed-process execution exhibited drift up to $\Delta p = 0.114$. What remains open is the underlying in-process state cause, whether same-process reset can be achieved without separate processes, and behavior under cancellation, concurrency, and long-running load. Acceptance must still cover probabilities, selected answers, policy outputs, request order, cancellation/recovery and the advertised concurrency/memory envelope, not only one repeated anchor. [W: §§21.6, 22.2, 23.4; E40–E42]

## 10. The architecture worth pursuing next

### 10.1 Keep separate the historical integration reference and current research mainline

The older immutable integration reference remains a different, useful differential function:

```text
Qwen3.5-4B-Base, profile a047d6802c3f06f085b8
state-first complete hybrid root
    → isolated question continuation
    → isolated candidate continuations
    → learned candidate scores + semantic-none model
    → distribution + separate policy
```

Its historical CPU and pinned MLX FP32 evidence is not replaced by the new T4 experiments. Conversely, it does not qualify the new post-trained LoRAs or their compact code contracts. [W: §§15–17]

The **executed local-decision research mainline** is now:

```text
Validated state, question and all described typed alternatives
    → complete admitted prompt (1,024 total tokens in current studies)
    → one post-trained Qwen3.5 decoder forward with ordinary LoRA
    → final answer-position hidden state
    → compact frozen pretrained A–P code vectors
    → probability distribution over the supplied outcomes
    → typed response / separately evaluated calibration and action policy
```

There is no free-form answer generation, replacement pointer, recurrent refinement block or token-memory head. The current notebook reference executes isolated question prefills serially; it is **not already a shared-state Rust/MLX engine**. Deterministic application dependencies and exact reuse remain integration targets whose historical component evidence is retained. ([LD-V12B][]; [LD-V13-PLAN][])

For future shared-state integration, retain full attention KV, DeltaNet recurrence, convolution state and positions. Cache identity binds model/adapter, tokenizer, exact finalized prefix, renderer, precision and backend. Keep question descriptions out of a question-independent root; catalogue-conditioned prefixes have a different contract. Mutable continuation state must be isolated, and a block attention mask alone does not isolate a recurrent stream. Adapter changes invalidate parent-feature/state caches unless separately proven compatible.

**Frozen V13 entries, not promoted checkpoints:** the student in A and B is LD-V12B `2b/fixed_code` and the teacher in A only is LD-V12B `4b/fixed_code`, both at joint step 150. Exact paths, base revisions and adapter/readout file hashes are in [whitepaper §25.3][W25-3] and [`FROZEN_RESEARCH_PROFILE.json`](FROZEN_RESEARCH_PROFILE.json). Full `FROZEN_ENTRIES.json`/study-plan manifests in the private external store bind readout, tokenizer, base revision, contract and reference predictions; adapter-file and tensor hashes are different concepts. Recorded identities are not a reload of the model weights. [LD-V13A-PLAN]

Freeze the current **controlled training settings**: rank 16/alpha 32/dropout zero, NF4/FP16 and FP32 adapters, learning rate `2e-5`, original-parent KL 0.25 on its trust mask, original CE with teacher-source CE weight 0.5, 150-update stages with accumulation 16, seven warmup updates then cosine, evaluation/checkpoints every 25 updates, and final-step selection. The 2,400 presentations cover 1,200 unique examples, not all 8,000 admitted rows. Exact module names and every value are in [LOCAL_DECISION_DESIGN.md](LOCAL_DECISION_DESIGN.md).

The **new trained-teacher KL** is not the original-parent preservation term, and is not adopted as default after V13A. The **JS consistency term** remains an unresolved experiment. Freezing these experimental identities is not a claim that each rank, loss weight, sampling ratio or schedule is globally optimal. New models and new LoRAs require fresh tokenization/admission, baseline, teacher/preservation targets, execution identities and qualification.

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

For historical integration, keep the 4B Base profile and measured 9B Q4 catalogue profile separate; no new 4B/9B size winner is asserted. **For the current local-decision training program, 2B has now earned an active resource-efficient research role** through LD-V12B/V13A, while post-trained adapted 4B remains the quality reference. Its target-Mac resource benefit and absolute reliability still need qualification. Older encoder/E11 results are not substituted for these newer joint-option measurements.

### 10.3 What to borrow from the general-model survey, and what not to borrow

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

## 11. The next decisions the evidence actually justifies

The October 1 unrestricted “next readout/size/teacher comparison” list is superseded by completed LD-V9–V13A results. Do not rerun answered grids or mistake the latest checkpoint for a qualified result.

| Question | Disposition at this cutoff | Next evidence / stop rule |
|---|---|---|
| Does wrapper-level LM-head removal materially accelerate the already restricted scorer? | **Answered negatively for LD-V12A’s workload; compact parity passed.** | No repeat grid. Direct packaging/startup, backend changes and Mac remain separate. |
| Which tested readout remains mainline? | **Fixed code**, favored by the real joint-LoRA comparison clearly at 4B and on point estimates only at 2B. | Preserve pointer failures and extra-initialization accounting; require a new hypothesis to revisit. |
| Is adapted 2B worth improving? | **Yes as a research candidate**, not a deployment replacement. | Retain latency/memory advantage while improving absolute class/none/whole-request behavior. |
| Does checked trained-4B guidance help more than continuation? | **Mixed completed V13A result**, not a default improvement. | No coefficient grid merely to turn a failed screen green; identify a better supported mechanism before another teacher intervention. |
| Does the teacher know the missing Plumb-none operation? | Four checked training opportunities; no measured development none-recall gain. | Broader verified operation supervision, not blind imitation or another old-anchor quota. |
| Does explicit consistency beat identical augmented exposure? | **V13B implemented; results not reviewed.** | Finish/review the fixed experiment unchanged; compare its own two arms, not B against A. |
| Can teacher and consistency effects be combined? | **Untested**, and A is not unambiguously positive. | Only an independently justified interaction study with a common entry and matched control. |
| Will gains survive new seeds and workloads? | **Unestablished.** | Lock a promising recipe; replicate and confirm on fresh disjoint groups/families before final acceptance. |
| Is the current confidence a correctness probability? | **No such claim.** | Keep concentration, probability calibration, risk/coverage and ordinal spread distinct. |
| Does it work on Mac/MLX/Rust, merged/quantized or prefix-cached? | **Not qualified for these adapters.** | Actual platform parity, representative request timing, memory and history/cancellation/lifecycle tests. |
| Is RLCD the next default? | **No local incremental RL evidence or tested reward/rationale loop.** | Reliable reward definition and a matched supervised control. This is not a requirement to prove perfect SFT before any RL research. |
| Can a candidate be published as qualified? | **No model through V13A meets all requirements.** | Lineage-aware loader, locked candidate/calibrator, independent acceptance and post-conversion/reload checks. |

### 11.1 Immediate model work

Keep V13B’s frozen V12B student, augmentation schedule, coefficients and final-step rule unchanged. On completion, inspect class/none retention, raw/cross-fitted probability quality, actual re-rendered stability, constant-output baselines, complete-group correctness and resource cost. An improvement on one transformation does not establish invariance; a lower aggregate NLL does not repair a lost class.

If B gives a useful effect without new quality damage, confirm the locked recipe with another seed and genuinely fresh evidence. If not, prioritize the repeated semantic/class-boundary failures and verified supervision coverage rather than more head complexity or arbitrary coefficients. V11’s transfer result is useful evidence for targeted supervision, but not justification to silently add its whole allocation to B.

Do not automatically chain either V13A output into B. Both A and B were designed as independent descendants of the same entry. Neither the 113 teacher opportunities nor the same-source macro improvement supports a teacher-plus-JS recipe before an interaction study.

### 11.2 Reusable training and release pipeline

Keep one three-stage conceptual workflow rather than many unrelated model definitions:

| Stage | Required contract |
|---|---|
| **Step 1: initial training/import** | Pin base/tokenizer/readout/module/precision configuration; establish matched baseline and immutable adapter/readout/renderer handoff. For a new model do not reuse old token IDs, features or probabilities. |
| **Step 2: targeted continuation** | Import exact entry weights; create a new optimizer stage for a new experiment; change one declared mechanism, retain a matched control and compare against both pretrained and entering models. Resume only identical committed executions. |
| **Step 3: packaging/qualification/publication** | Explicitly support the new lineage and readout contract; preserve training footprints and held-out boundaries; qualify candidate, calibration, artifact reload and any conversion. Private research packaging is not qualified release. |

The old canonical publisher must not infer that a new V11/V12B/V13 artifact is compatible because it contains a LoRA file. A compact fixed-code or custom-head bundle has a bound readout and tokenizer. Eight-case reload parity is useful serialization evidence, not model/service qualification. No training/diagnostic examples or credentials belong in a published artifact by default. [LD-V13-PLAN]

### 11.3 Separate systems track

Continue the historically supported deterministic-composition and exact-reuse work under named model/runtime identities. For the current adapters, independently qualify a target-Mac loader, numeric path, realistic prompt lengths, active lanes, cache ownership and mixed-history behavior. Symmetric allocator cleanup is required for a fair memory comparison; fewer forwards and nominal parameter counts are not latency/memory evidence.

The older proposed one-point noninferiority, 20% cascade-latency gain and Mac service budgets are **product design proposals**, not the current studies’ acceptance thresholds. Preserve the actual pilot profile: max source accuracy drop 0.03, supported recall drop 0.05 with at least eight groups, false-none ceiling 0.20, cross-fit NLL/Brier increase 0.01, relative schema flip/TV increases 0.05/0.02 and separate absolute flip target 0.05. Set application costs and stricter prospective limits before future acceptance, never after observing failures. [LD-V13A-PLAN]

These are two different obligations: a model-learning intervention should improve useful decisions, not reproduce every parental mistake; an engine claiming the same model function must preserve its declared probabilities and answers. Neither distinction authorizes post-hoc reselection, hidden evidence truncation, reuse of inspected tests as fresh acceptance, or relabeling AI review as human confirmation.

## 12. Direct answers to the original research hypotheses

| Idea or assumption in `RESEARCH.md` | Answer supported by the record | Method disposition |
|---|---|---|
| Useful typed decisions require a newly invented neural architecture. | Existing pretrained models with learned or finite-token readouts already make useful bounded decisions. | **Do not require architectural novelty.** |
| A small learned head is sufficient. | Earlier NLI successes remain; LD-V9/V10 heads do not replace adapted Qwen, and LD-V12B replacement pointers trail fixed code under their recipes (clearly at 4B, on point estimates only at 2B). | **Compact fixed-code mainline; scope historical head wins.** |
| A shared-state/query architecture is the main path. | Complete-state reuse works; pooled shortcuts fail; richer readers have not passed complete quality gates. | **Retain reuse, not the unproven representation shortcut.** |
| One-token codes are always better than every alternative. | E39 effects vary by field and precision. LD-V12B favors fixed-code joint LoRA over the tested replacement pointer clearly at 4B and on point estimates only at 2B, with declared initialization differences. | **Default within the measured newer recipe, not a universal ranking.** |
| Independent field prediction is enough. | Derived action improves quality; remaining priority-route contradictions show the limit. | **Compute declared dependencies outside the model.** |
| More questions should add almost no latency. | Strong long-state mechanics gains coexist with a slower short-state shared path. | **Measure marginal Q cost; use adaptive plans.** |
| Fewer forwards or custom kernels guarantee speed. | Flat-field and the tested custom-Metal path are slower despite their mechanical changes. | **Reject proxy-only optimization.** |
| BF16 is a stable default and lower bits are faster. | Multiple precision/shape/history tests fail; BF16 can be faster than Q4 on a given stack. | **Version and test each operating point.** |
| Compressing KV solves memory use. | FP16-KV has bounded storage benefits; recurrent state and weights dominate other workloads. | **Retain measured accounting; discount failed low-bit codecs.** |
| Few active experts imply a small resident model. | Request-wide expert use can be broad; skipping did not unload weights; streaming is untested. | **Separate compute, residency and transfer experiments.** |
| More context solves missing evidence. | Visibility-sensitive decisions improve, but rejection can worsen and adaptation can lose those gains. | **Preserve evidence and test its use.** |
| Teacher/parent replay guarantees retention. | LD-V7 protects covered sentiment classes; LD-V13A adds mixed local effects and no net extra correct primary answers, while Plumb-none remains 1/10. | **Retain scoped preservation; do not adopt the new teacher term as a validated default.** |
| Explicit unknown plus temperature scaling yields calibrated decisions. | None recall, false-none and high-confidence errors remain; temperature fits can worsen held-out scores. | **Train semantics; accept calibration separately.** |
| RLCD/CADO is necessary. | No local result isolates an incremental benefit from decision-utility RL over supervised controls. | **Unresolved; not the next default investment.** |
| A sub-1.5B encoder should be the initial winner. | Historical specialists are fast but not general policy replacements. LD-V12B adapts 2B to 69.89%; LD-V13A control reaches 73.15% while remaining unqualified. | **2B resource-efficient research role, 4B quality reference; specialists conditional.** |
| Removing the LM-head module supplies the main speed/memory gain. | LD-V12A compact/stripped scoring preserves outputs but adds no material complete-request or live-weight-memory saving beyond restricted scoring. | **Implementation simplification; no repeat grid.** |
| More recurrent head iterations improve decisions. | LD-V9 one pass supplies 98.3% of the four-pass NLL gain; LD-V10 finds no useful extra token-memory-attention effect. | **Stop the tested recurrence/memory recipe; retain narrow scope.** |
| Gold-checked teacher opportunities prove a skill was taught. | LD-V13A has four eligible Plumb-none opportunities but unchanged 1/10 development recall. Coverage is not generalization. | **Verify semantic operation coverage and outcomes separately.** |
| Consistency or teacher-plus-consistency is validated. | V13B is implemented; its results are not reviewed. A/B interaction is untested. | **Unresolved; preserve independent entries and controls.** |
| MTP will accelerate the no-generation path. | No output sequence exists for that mechanism to accelerate; no local MTP result was measured. | **Do not prioritize for this graph.** |
| Passing a leaderboard or cache test qualifies a Jev replacement. | Benchmark quality, semantic scope, execution parity and service behavior have distinct failure modes. | **Require their separate evidence.** |

**Evidence map:** newer LD model/results: §§4.6–4.11 and W §25; newer audits/calibration: §5; headless timing/storage: §8.2.1; exact current entries: §10.1 and W §25.3; pending decisions and release boundaries: §11. Historical readout and heads: W §§4–5, 18.17–18.21, 20, 23, 24; representations: §§16, 18.3–18.16; training/retention: §§18.20–18.26; systems/memory: §§7–10, 16–17, 19–24; hypothesis framing: R's architecture, training and evaluation sections. This table supersedes the *recommendation status* of earlier proposals in this working paper, not the original experiment records in R or W.

## Conclusion

**The mainline is now specific: one fully conditioned Qwen forward, ordinary LoRA, compact frozen answer-code scoring, explicit typed probabilities and separately qualified deterministic application logic. Develop 2B for resource efficiency and retain 4B for higher-quality comparison.** This is supported more directly than adding another recurrent reader, replacing the readout with the tested pointer, or streaming a larger MoE. It is not a production qualification.

The most useful new positive results are equal-class sentiment replay, scoped light parent preservation, recovery of **22.61 macro accuracy points** by 2B adaptation, and another **31 correct primary answers** from each bounded V13A continuation. The important negative results are equally actionable: pointer replacement did not win the actual joint-training comparison; head-module removal did not materially accelerate an already restricted scorer; teacher guidance gave **no net extra correct answers** beyond replay and introduced a sentiment regression; neither proportional confidence nor a small count of failed checks establishes reliable semantic none. ([LD-V5]; [LD-V7]; [LD-V12A][]; [LD-V12B][]; [LD-V13A][])

Historical composition, selective indexing, exact catalogue/state/input reuse and process isolation remain evidence-backed engine components only within their original profiles. Integrating them with the newer adapters requires renewed numerical, workload and target-platform qualification. Do not read their earlier Mac/native wins as measured serving performance for the new 2B research model.

**The next action is to review V13B’s independent consistency result, not automatically combine it with A.** Confirm any useful locked recipe on another seed and genuinely fresh evidence. Where the missing operation is not supported by trustworthy examples or a reliable teacher, improve verified supervision rather than strengthen imitation. Keep unanswered RLCD, cascade, long-context, Mac/MLX/Rust and publisher questions explicit. This is the evidence-based direction: retain demonstrated components, stop answered grids and make each next run answer a remaining question.

## Sources and provenance

### Local result authority

**[W]** `WHITEPAPER.md`, version 0.8.13. Version 0.8.12 integrated E43/E44; version 0.8.13 adds [§25][W25], which owns the LD-V12B and LD-V13A measured results. Historical section/experiment anchors and measurements are carried as source-reported material; bracketed references such as `[W: §23.1; E42]` do not identify new experiments or new raw audits.

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
| LD-V12B; LD-V13A | §25 | Local-decision readout/size and checked-teacher results, frozen V13 entries and their evidence boundary. |

**[ER]** [Encoder recovery report][ER], run `20261001T223435_041738Z_f25616`, protocol `openkind-t4-recovery-reliability/v4.1.1`, completed 1 October 2026. New local, source-reported encoder measurements; evidence roles and sample sizes remain those in the report. Supporting [manifest][ER-M], [summary][ER-S] and [completion record][ER-C] establish the executed scope. The prior r0.9.3 review reported that the downloaded summary’s SHA-256 matched the completion record; that historical file-integrity check was not a re-audit of predictions, labels, model weights or the result archive and has not been repeated. ER is a source label for this follow-up, not a newly assigned whitepaper experiment ID.

**[R]** `RESEARCH.md`, the research synthesis, hypothesis program and dated external-model roster. Earlier proposals remain proposals unless local result evidence changes their disposition. Its general-purpose-model survey is not treated as a ranked decision-model benchmark.

**Lineage:** revision 1.0.0 succeeds working-paper revision 0.9.3 (1 October 2026) and draws on the October 3 `LOCAL_DECISION_DESIGN.md` proposal. Revision 0.9.3 attributed its own base to repository revision 0.9.2 / Git blob `d6d853f9285e93e984aa841c8ad56212b971471c`, retained as historical lineage. Input/output hashes for revision 1.0.0 are recorded in `UPDATE_PROVENANCE.json` (external, not checked into the repository).

**Review boundary for r1.0.0:** saved summary values, study entries and document claims are reconciled against saved result artifacts. Revision 1.0.0 does not rerun model inference, training, full raw-prediction replays, weight loading, source-label adjudication, reserved-final evaluation, current benchmark pages or V13B results. Earlier review/checker counts remain attributed to those reviews. Historical E-series records are not independently re-audited or renumbered.

### Newer local-decision evidence register

These are **LD notebook** labels, not newly assigned E-series experiment IDs. This evidence lives in a private external store (the project Google Drive) and is not checked into the repository. Drive paths below are relative to `OpenKind_Local_Decision_Training_T4/`, and linked records require access to that store. JSON result links are primary output records; prose review links/packaged summaries carry clearly labeled replay or audit interpretation. LD-V4, LD-V5, LD-V6, LD-V7, LD-V10 and LD-V10-GUIDE (the LD-V10 run guide) are cited by name only.

| ID | Record and study | Use / boundary |
|---|---|---|
| LD-V4 | Initial `sweep_decaf3f2a6d13369`; preserved initial review | Early learning-rate improvements, rejected retention, and parent fallback/gate distinction; smaller earlier panel. |
| LD-V5 (external) | `studies_v5/study_c5f1a0f106115251`, summary and saved comparisons | Class balancing and presentation pilot; all candidates rejected. |
| LD-V6 (external) | `studies_v6_retention_repair/study_67df9575d6dd9d19` | Lower-rate and none-pair comparison; saved-logit calibration diagnostic does not reclassify the original result. |
| LD-V7 (external) | `studies_v7_parent_preservation/study_8b0844ccd3ca0554`, `SELECTION_V7.json` | Scoped original-parent preservation and prospective calibration-aware profile; no winner. |
| [LD-V8][] | `audits_v8_none/bbecaeffe6ca6110`, completed AI source review | 60 reviewed cases, 50 confirmed/10 uncertain; original preservation-coverage stop, not training failure. |
| [LD-V9][] | `study_1c79a2359c64ec05`, saved recurrent-head results | Frozen-feature recurrence, one-pass contribution, cost and limitations. |
| LD-V10 (external) | `studies_v10_readout_strands/study_575b0a8bba29ab9f` | Readout/memory ablation, external Strands system comparison and local cache diagnostics. |
| [LD-V11][] | `studies_v11_simple_decisions/study_9f2931f8c94a4e5b/execution_d6dc6371ddacc28e` | Replay versus verified-curriculum replacement; transfer win and unresolved composition/retention. |
| [LD-V12A][] | `studies_v12_head_removal_size/study_880ad0d19c175e72/execution_3f51011cf751adf1` | Equivalent fixed-readout timing/parity, tied-storage/cleanup caveats and unadapted size comparison. |
| [LD-V12B][] | `studies_v12b_joint_readout/study_6ea138781306a1b0/execution_c33f8407e52d09f4` | Four joint-LoRA/readout recipes, extra pointer initialization, size/quality/resource trade-off. |
| [LD-V13A][] | `studies_v13a_checked_teacher/study_3e9da0121f40f75e/execution_c33f8407e52d09f4` | Completed two-arm teacher increment versus ordinary continuation; both rejected. |
| [LD-V13A-AUDIT][] | Same execution, `TEACHER_AUDIT.json` | Eligibility/opportunities, not proof of skill acquisition or independent human audit. |
| [LD-V13-PLAN][] | Frozen-entry guide and [executed A plan][LD-V13A-PLAN] | Exact entries, losses and independent A/B design; no B outcome included. |

New evidence metrics remain scoped: larger-model and small-model quality/timing comparisons are within their specified study; source-macro and row-weighted estimates differ; technical repeats do not increase independent group count; current native integration and model promotion do not follow from saved research artifacts.

### External comparison sources

External benchmark outcomes are evaluator-reported; training recipes and local project results are author-reported. None has been rerun for this paper.

| ID | Source | Use and limit |
|---|---|---|
| X0 | [JevBench v1.4.2.2 snapshot][X0] | Dated board underlying R's top-25 roster; not the current protocol. |
| X1 | [JevBench page][X1] and [v1.5.0 aggregate artifact][X1a], as recorded in the October 1 review | Dated external snapshot, not the current board. Changed suite/method and addendum status prevent longitudinal model-quality inference. |
| X2 | [Imajev-4B model card][X2]; [pinned phase-3 summary][X2a] | Correct starting-checkpoint identity and distinguish learned code readout from vocabulary slicing. Recipe is compound, not an ingredient ablation. |
| X3 | [Plumb-4B model card][X3] | Teacher-generated hard decisions, JevK5 weight lineage, temperature and public-development disclosure. |
| X4 | [JevK5 v0.2.0 README][X4] | Historical versioned finite-answer baseline. Newer server/model behavior must not be assigned to this board row. |
| X5 | [Hopper model card][X5] | Decision LoRA and answer-type calibration; source/weight licensing scope remains a separate deployment check. |
| X6 | [Mapika decider-4b model card][X6] | Distinguishes roster v2 from later v2.1 parent-distribution replay and calibration changes. |
| X7 | [Cygnet recipe][X7] | Frozen Gemma, stock vLLM, one-token readout, token-mass aggregation and author-held-out calibration example. |
| X8 | [reflex repository][X8] | Distinguishes historical LoRA row from later frozen/order-averaged configuration. |
| X9 | [Winnow-12B model card][X9] | LoRA/refinement and gold-agreeing teacher supervision; shared-state runtime. Private data limits independent reproduction. |
| X10 | [PrivateMode method article][X10]; [implementation][X10a] | Motivation for indexed option scoring; not a local speed, quality or calibration result. |

**Older source-input identities retained as historical lineage only:**

```text
WHITEPAPER(8).md
  sha256 2c432102ecf31e56e74cc92377577660d41cfd68695559c9905a9394ff460278
WORKING_PAPER(4).md
  sha256 094d46c1d3045f1e20102f3d083f53a9c1797478cf43fcd4bcf4e42d00c4b999
RESEARCH.md
  sha256 eb2377a8a222424dbb0c8f25ce47919aab16092cc4934fca2572379dd30616cb
```

These are hashes of the three Markdown source inputs, not hashes or renewed audits of their referenced models and result archives.

[W]: WHITEPAPER.md
[W25]: WHITEPAPER.md#25-local-decision-training-results-ld-v12b-and-ld-v13a
[W25-1]: WHITEPAPER.md#251-ld-v12b-trained-readout-and-model-size
[W25-2]: WHITEPAPER.md#252-ld-v13a-continuation-versus-checked-teacher-guidance
[W25-3]: WHITEPAPER.md#253-frozen-entries-for-ld-v13a-and-ld-v13b
[R]: ../RESEARCH.md
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


[LD-V8]: https://drive.google.com/file/d/1I_6nW2BGwFFymwiZC2Jt5ePK9HZ0HCGC/view
[LD-V9]: https://drive.google.com/file/d/160XJFZ6Zo5_HBG7zQ5C9p9HoGcR7L-qP/view
[LD-V11]: https://drive.google.com/file/d/1yO6bqcDo-EDnJFpcJHagorq4aeaOX8Ju/view
[LD-V12A]: https://drive.google.com/file/d/1JED9qnzcH_rgDIoKxEjayWoSlciQIMAT/view
[LD-V12B]: https://drive.google.com/file/d/17UDnx0xR6mO7Au09bsrBfeT30cUT0NLc/view
[LD-V13A]: https://drive.google.com/file/d/1eUTSTZVisA4Ar-EMgMDQvYPHgiLjYk_V/view
[LD-V13A-AUDIT]: https://drive.google.com/file/d/1sIxApSB3egf6shA4uQxqzdt9FUfJ3ulf/view
[LD-V13A-PLAN]: https://drive.google.com/file/d/1uEH6PPaqV2ENRL-b0jKp3VJsV8q5ZqNH/view
[LD-V13-PLAN]: https://drive.google.com/file/d/14AZr6eyj4vE3Hom6gF0HWOXwAiWzLANi/view
