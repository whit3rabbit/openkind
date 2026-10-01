# OpenKind: Outstanding Questions for a Fast, Accurate Jev Alternative

**Research and engineering question register · Version 1.1 · 1 October 2026**

## Executive assessment

**Yes—BERT-family classifiers are still worth testing. The existing results reject particular compact-model recipes, not the entire encoder family.** The useful comparison is no longer “a small classifier versus generated JSON.” It is a properly trained or pretrained task-aware encoder versus OpenKind’s strongest finite-answer Qwen paths, under the same evidence, task and complete-request measurement contract. [W §15; U3 §§2,7; U4 §2; BERT; MB; DNLI]

The larger outstanding question is whether one integrated system can combine **useful semantic accuracy, complete and trustworthy probabilities, independent question behavior, and low end-to-end cost**. The project has demonstrated parts of that system, but their best results come from different profiles and experiments. They cannot be combined on paper into a tested deployment. [W §§16–17,21,23; U3 §§2–4; U4 §2]

The next work should concentrate on three opportunities. First, combine the strongest measured readout and dependency ideas without introducing hidden probability assumptions. Second, eliminate identified repeated work: grouped catalogue cache misses and repeated open-encoder text encoding. Third, run a small, fair classifier comparison with stronger BERT-family baselines and genuinely held-out question families. Natural-evidence review, uncertainty transfer and service qualification remain requirements across all three tracks—not a final cosmetic check.

This document contains **60 outstanding questions**, grouped into eight workstreams. It is a deduplicated register of the gaps identifiable in the supplied evidence and research, not a claim to enumerate every possible future research idea. Each question states what is already known, the unresolved comparison, and what evidence would close it. **There are no new model results in this document.** The existing papers, experiment evidence, runtime defaults and protected-final boundaries are unchanged.

## 1. Which evidence owns the current status?

The supplied sources have different cutoffs. Their future-work sections must not be concatenated into a backlog without reconciliation.

| Authority | Evidence boundary | How it is used here |
|---|---|---|
| W: whitepaper v0.8.11 | Historical studies through the 1 October T4 integrated run, E42 | Authority for older model, evidence, rejection, cache, CPU, MLX, and recent integrated findings |
| WP: working paper revision 0.9.2 | Synthesis through E42, with external research comparisons | Research-question framing, incorporating the latest T4 integrated results |
| U3: Unified results review | Run `20260929T194506_694066Z_dc1f74`, protocol v3.0.0 | Supersedes earlier untried labels for routing composition, grouped readouts, indecis and process-isolation treatments |
| U4: T4 integrated research review | Run `20261001T105345_828556Z_6e6d92`, protocol v4.0.0 (E42) | Measured selective indexing + route composition (XR), narrowly joint readout (NJ), multi-catalogue host cache, dedicated process history for X/XR/NJ, and diagnosed classifier setup exception |
| R: newly supplied research | Includes the reported 29 September Rust joint-choice follow-up, called R29 here | Additional local **source-reported** results; their linked benchmark files were not re-audited in this document |
| H, S and primary external sources | Public pages reviewed for this document | Architectural hypotheses, comparison designs and released baselines—not OpenKind measurements |

The current review reads the supplied result synthesis and research; it does not repeat the earlier archive-reconciliation scripts, training, GPU measurements or Rust tests. The source manifest records hashes of the input documents only. Earlier raw-result verification remains attributed to the reviews that performed it. R29 is a convenient label for the dated section in R, **not a newly assigned whitepaper experiment ID**.

### Work that must not be described as wholly untested

| Question already addressed | What was actually learned | What is still open |
|---|---|---|
| Can dependent outputs be derived? | Action derivation and routing rules improve bounded quality; U4 measures XR as a timed path: 95.40% field accuracy, 84.38% all-six correctness at 242.98 ms median. | Complete joint probability semantics; new policies and tasks; root cause of missing cert on failing points |
| Can a different field formulation help? | X reaches 92.19% field accuracy at 222 ms; U4 Arm NJ (narrowly joint route/urgency) supplies all 6 field distributions at 94.01% field accuracy and 79.69% all-six correctness at 228.31 ms. | Integration, generalization, and calibrated policy usefulness |
| Can multi-catalogue caching help? | U4 multi-catalogue host snapshot cache cuts grouped request latency by ~54% (from ~1,511 to ~690 ms, 2.19x speedup) with $\Delta p = 0.0$ across 96 pairs. | Host-to-device eviction and cold-start amortization under variable load |
| Can mixed-history interference be avoided? | Dedicated processes pass history checks with $\Delta p = 0.0$ across D, X, XR, and NJ (vs mixed drift up to $\Delta p = 0.114$). | Cheaper same-process root-cause repair, cancellation and concurrency |
| Have compact encoders been tried? | E11 ModernBERT and U3 indecis missed quality targets; U4 attempted matched encoder tests (ModernBERT, DeBERTa, GLiClass) but was blocked by a `SameFileError` logging exception during setup. | Fix file-copy logging defect, rerun encoder training/evaluation pipelines, and compare against XR and NJ |
| Has the Rust catalogue idea run? | R29 reports an improved fitted-head catalogue renderer, factorized order/code probes and a gated temperature result. | Natural-task confirmation and complete target-host request cost |
| Have Rust and MLX been validated at all? | The pinned 4B reference has native CPU and bounded MLX FP32 parity; CPU service/load evidence exists. | New useful profiles, full MLX service cost and expanded release qualification |
| Have Noul/Score and high K been exercised? | Bounded BoolQ/SST-5 and high-cardinality mechanics already exist; U4 tested high-K omission capacity and ID confusion at K=16, 32. | Unseen binary criteria, unfamiliar rubrics, semantic high-K accuracy and rejection |

Sources: W §§12–17,20–23; WP §§4–10; U3 §§2–11; U4 §§2–10; R, “Outcomes of the 2026-09-29 runs.” These entries retain their original sample and hardware limits.

## 2. What the two articles add—and what they do not

### Hume: test information flow, not a guessed proprietary topology

Hume’s API probes motivate direct readouts, shared computation, separation between sibling questions, and interaction among options within a question. His causal-backbone and MoE explanations remain hypotheses; the probes do not uniquely identify a dedicated head versus selected vocabulary rows. Server timings and output-token accounting are not isolated neural throughput measurements. [H]

The concrete OpenKind consequence is to distinguish **question isolation** from **candidate independence**. A question should not learn a secret available only in another question. However, its alternatives may legitimately explain one another or include “none of the above.” Test both boundaries directly. An irrelevant-option test is a diagnostic, not a universal requirement that all pairwise odds remain unchanged. [R: Hume evidence review; OQ-18, OQ-20]

### Raschka: fixed-task classifiers are a necessary control, not the entire product

Raschka’s IMDb comparison makes specialized text classifiers a serious baseline: a fine-tuned ModernBERT can perform strongly on a fixed classification task. His Jev API results and local classifier timings use different execution environments, and potential pretraining exposure remains unresolved. A movie-review classifier does not establish arbitrary caller-defined question or rubric following. [S]

The experimental response is to maintain separate lanes for **fixed-schema specialists**, **dynamic-label/question-aware classifiers**, and **general typed-decision systems**. A specialist may be the right product for a stable workload even when it is not a general Jev replacement. Conversely, broad interface support cannot excuse poor specialist performance on a task the product claims to cover.

## 3. BERT and other classifiers: the comparison that is still missing

### 3.1 A bounded shortlist

Do not launch every model below simultaneously. Start with ModernBERT-base, one correct NLI encoder and one dynamic-label model; retain the existing Qwen and indecis controls. Add the historical and cheap baselines where their task contract applies.

| Candidate | Role in the experiment | Important boundary |
|---|---|---|
| Rules plus TF-IDF/logistic regression | A cheap fixed-schema floor, especially on highly structured or lexical tasks | It is not an arbitrary-question reasoner. Rules receive observable inputs, never gold labels. |
| BERT-base | Historical, bidirectional encoder control with a trained task head | Use proper task fitting and preprocessing; an unfitted head is not a meaningful quality comparison. [BERT] |
| `answerdotai/ModernBERT-base` | Main new trained encoder challenger; pretrained encoder with a task-aware head | The model card describes 149M parameters and an 8,192-token native context, not a ready-made universal classifier. Capacity does not prove long-input quality or fit for every training batch. [MB] |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Strong released three-way NLI control | Preserve entailment, neutral and contradiction and verify label IDs. It is not the same model or task as binary zero-shot classification. [DNLI] |
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c` | Separate binary entailment/not-entailment candidate-hypothesis control | A hypothesis template is part of the model contract; neutral and contradiction are not separately available. Its input budget must be audited. [DZ] |
| One pinned GLiClass uni-encoder checkpoint | Dynamic-label comparison that reads text and label descriptions together | The default approach is not a fixed-label BERT head, but a single schema pass does not prove multi-question state reuse or sibling isolation. [GL; GLM] |
| SetFit over a declared sentence encoder | Optional small-labeled-data fixed-task control | Contrastive adaptation and a classifier are a different supervision contract, not evidence of new-instruction following. [SF] |
| One released Laya or Von checkpoint | Conditional typed option-marker comparison | Verify exact checkpoint, mask, effective text budget, calibration and license; earlier home-built ModernBERT failures do not evaluate these releases. [R: Laya and Von] |
| Existing indecis fixed/open profiles | Retained compact CPU baselines and an input-reuse optimization target | Do not rebrand its failed fresh-template transfer as a successful general classifier because its development score or CPU speed is attractive. [U3 §§6–8] |

### 3.2 Three experiments that must remain separate

**Released-system comparison:** freeze the published model, its declared rendering and any required calibration procedure, then compare the complete system on common cases. Different pretraining and supervision are acknowledged; this identifies a useful product option, not a causal architecture effect.

**Matched-learning comparison:** give a limited set of backbones the same admissible labeled examples, split rules and explicit tuning budget. Distinguish frozen-feature heads, parameter-efficient adaptation and full fine-tuning. Report training exposures, compute and seeds, not just epochs. A failed preprocessing bridge, wrong pooling index or truncated evidence is not evidence that the model family is inadequate.

**Same-model execution comparison:** hold the learned function, tokens and renderer fixed while changing batching, output projection, quantization, caching or backend. This comparison uses the numerical/answer/policy preservation gates. A new model or prompt may improve old answers and must instead be judged by its own quality gates.

### 3.3 Why ordinary BERT caching is different

In a standard bidirectional joint input such as `[state; question; options]`, state-token representations can attend to the later question and options. Changing that suffix can therefore change the state representation itself. **Putting the state first does not make a causal-style immutable state cache exact.** This is a computational consequence of bidirectional conditioning, not a new measured OpenKind result. [BERT]

A fixed-schema multi-head encoder can encode the text once because its learned questions live in the heads. A bi-encoder can also encode text once and compare many separately encoded options. Both may be inexpensive, but they provide different interactions from joint state/question/option encoding. A separate state encoder with query cross-attention, restricted masks, or a T5-style encoder-decoder reader is another architecture to train and validate—not a semantics-preserving cache patch. [R: architecture proposal; T5; OQ-06, OQ-12, OQ-59]

### 3.4 Fair quality and resource reporting

Report a **common-content comparison** first: every model sees the same decisive evidence and candidate descriptions. Separately report each model’s maximum-supported input configuration, including truncation and unsupported cases. The same nominal token count is not the same text across tokenizers.

Keep fixed-schema policy classification, three-way NLI, runtime candidate choice, binary criteria and ordinal rubrics in separate tables. Do not compare independent multilabel scores, relevance scores, conditional Choice softmaxes and semantic-none distributions as though their normalization made them equivalent. [W §§2,5; OQ-19, OQ-23, OQ-38]

Use the actual Colab CPU and T4/L4 measurements. Start with short/medium inputs and modest batches, then perform admission checks before long-context fitting. Neither the card’s context capacity nor a parameter-count estimate guarantees training fit. Mac CPU/MLX testing remains a separate target-host stage.

## 4. Recommended sequence: a few bounded experiment packs

P0 means a direct integration or evidence blocker; P1 means the next discriminating comparison; P2 is conditional research. OQ-60 is a release requirement, not a demand to complete every speculative extension before a scoped product can exist. The register has 17 P0 questions, 33 P1 questions, nine P2 questions and one P0-release question. Related-question links are not a rigid acyclic build plan; several contracts must be designed together.

| Pack | Smallest useful comparison | Decision it should settle |
|---|---|---|
| **A. Integrate the current leaders** | D/X/R versus X-plus-rule; a narrowly joint source only if complete probabilities are required; selected readout under dedicated-process isolation | Can a single measured profile retain the separate accuracy, probability and stability gains? OQ-01,02,05 |
| **B. Remove identified repeated work** | Existing G/GR versus exact multi-catalogue cache; existing open encoder versus encode-once input reuse | Can the same function become cheaper without a new model? OQ-03,12,44 |
| **C. Challenge Qwen fairly** | ModernBERT-base, the appropriate pretrained DeBERTa and one GLiClass configuration versus fixed Qwen controls | Does an encoder meet a declared quality/risk floor on new task families at lower complete cost? OQ-09–11,14,17 |
| **D. Confirm the state-first catalogue path** | Recover R29’s pinned renderer; test admissible natural tasks and the real scheduler path | Do its local quality gains survive evidence-sensitive tasks and their 2.3× scorer-time cost? OQ-04,06,18,25–27 |
| **E. Qualify reliability and deployment** | Locked calibration and review policy, realistic traffic, cancellation, memory pressure, API conformance and the intended Rust/MLX profile | Is the selected system useful and stable in its advertised operating envelope? OQ-33–39,48–55,60 |

Data review and confirmation design precede model selection in every pack. The original Qwen document-retention program has its own admissible training contract and belongs in OQ-28; it should not be smuggled into a compact-classifier experiment. Matched learned-head training, distillation, RL and alternate modalities follow only when their prerequisites and decision value are clear.

One future notebook can orchestrate selected Colab packs and record unavailable stages honestly. It cannot turn CUDA results into Mac measurements, supply independent human adjudication, or validate every optional architecture simply by containing a question ledger.

## 5. How to close a question

A question is **not closed positively** merely because a stage ran, a graph looks faster, or a confidence column is populated. A result can close a specific treatment negatively while the broader problem remains open.

**Semantic closure:** report all-fields and per-field correctness, class-sensitive recall, semantic-none/false-none, ordinal error where applicable, and independently defined evaluation groups. More complete probabilities are not automatically better decisions.

**Execution closure:** for a claimed unchanged-model implementation, retain the existing maximum probability difference of 0.005 together with zero selected-answer and policy changes over its specified qualification panel. These are project regression gates, not a population-risk bound or proof that the reference is correct. Preserve the exact model, rendering, head and arithmetic identities.

**Efficiency closure:** measure the complete request or realistic trace, including every required subcall, cache population, lookup, transfer, recovery and serialization. Separate first-use, warm and queue-inclusive timings. Report actual input lengths, Q/K, memory high-water observations, throughput and tail latency with their sampling scope. Never infer speed from serialized output-token accounting, one framework call, cache-hit count or active parameters alone.

**Probability and policy closure:** identify the probability space, retain raw scores, fit calibration on a separate partition, accept or reject it on a separate gate, and freeze the application policy before confirmation. Derive dependent distributions from a valid source joint distribution or state explicitly that only point answers/bounds are available. The minimum field probability is not the probability that all fields are right.

**Independent confirmation:** previously inspected U3 and R29 cases are now regression/diagnostic material for changes inspired by their errors. Protect old final partitions. Hold out source and task/rubric families, account for shared-state and repeated-seed correlations, and report ambiguity or label disputes without retrospective relabeling.

**Release targets are still an explicit product decision.** The supplied record does not specify one universal acceptable error rate, minimum useful coverage, p95/p99 limit or deployment memory ceiling for every task. A prospective contract must fill these in before release comparison; this document does not silently create new production thresholds or weaken old gates. A schema for those unset requirements is included in the JSON register.

### A useful sample-size reality check

Under an idealized independent Bernoulli model with a frozen acceptance policy, zero errors among n accepted examples gives a one-sided 95% upper bound `1 - 0.05^(1/n)`. With n=61, the bound is about 4.8%, not zero. Approximately 299 independent accepted examples with zero errors would be required to push that bound below 1%, and approximately 2,995 for 0.1%. These are algebraic planning examples, not deployment guarantees: correlated examples, adaptive selection, multiple subgroups and distribution shift invalidate the simple interpretation. They explain why a small zero-error accepted subset cannot certify rare-event reliability. [Derivation; OQ-37]

## 6. The outstanding-question register

The source key after each entry separates existing observations from the proposed test. Status labels such as PARTIAL or SOURCE_REPORTED_PARTIAL do not assert that all treatments are unrun. Questions are closed only within the named task and execution scope; negative results remain part of the record.

| Workstream | Question IDs | Main decision |
|---|---|---|
| A. Integrate the strongest readouts without losing their contracts | OQ-01–OQ-08 | One useful, stable integrated readout |
| B. Determine whether a smaller classifier can replace or complement Qwen | OQ-09–OQ-16 | Whether a compact classifier can replace or complement Qwen |
| C. Establish the actual semantic scope of a Jev alternative | OQ-17–OQ-24 | What the product can legitimately claim to understand |
| D. Improve learning while preserving evidence and retained tasks | OQ-25–OQ-32 | What data, evidence and adaptation actually improve retained quality |
| E. Make uncertainty and downstream policy useful | OQ-33–OQ-39 | Whether probabilities support useful decisions |
| F. Reduce complete-request cost while preserving the selected function | OQ-40–OQ-49 | Which optimizations preserve behavior and lower actual cost |
| G. Close conformance, evidence and deployment boundaries | OQ-50–OQ-55 | Whether the implementation and consumer meet their declared contract |
| H. Conditional extensions—not prerequisites for the next useful result | OQ-56–OQ-60 | Which broader research is conditional rather than an immediate prerequisite |

## A. Integrate the strongest readouts without losing their contracts

### OQ-01. Does selective indexing plus routing composition retain both gains?

**Priority:** P0 · **Status:** closed on tested policy panel (U4 / E42)

**Known and still missing.** U3 separately measured X (selective indexing) and R (route rule). U4 / E42 evaluated the unified timed path (Arm XR) across 192 new authored policy cases: XR achieved **95.40% field accuracy** and **84.38% all-six correctness** at **242.98 ms median** (+40.1 pp all-six correctness over X alone at 222.22 ms). Route marginals are computed from the composed rule with exact route vector withheld. 100% of remaining XR eligibility errors (23/23) were traced to missing certification data on failing score cases (`certified: null`, score < 70).

**Test.** Extend XR to natural document tasks, long-context states, and unseen schemas. Validate whether the unrecorded certification failure pattern generalizes or responds to explicit schema prompt clarification.

**Evidence to close.** Whole-request accuracy on natural tasks; explicit handling of missing certification/eligibility rules without regression; complete multi-task calibration.

**Sources:** U3 §§2,12; U4 §2; W §23.1. **Related / prerequisites:** OQ-17, OQ-25. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-02. Can a narrowly joint readout provide exact derived probabilities without J’s collateral errors?

**Priority:** P0 · **Status:** demonstrated on tested policy panel (U4 / E42)

**Known and still missing.** U3 J demonstrated a joint source but collapsed retries accuracy (100% → 72.92%). U4 / E42 evaluated Arm NJ (narrowly joint route and urgency readout), which models the four joint route/urgency outcomes explicitly while keeping other fields isolated. NJ supplied all six field distributions and achieved **94.01% field accuracy** and **79.69% all-six correctness** at **228.31 ms median**, completely avoiding retries regression.

**Test.** Test NJ across diverse domain distributions, multi-label candidate sets, and under calibrated policy thresholds. Compare empirical policy costs between NJ and XR.

**Evidence to close.** Normalized source-to-output probabilities across heterogeneous policy families; demonstrated calibration transfer; competitive decision utility against XR.

**Sources:** U3 §§2,9; U4 §2; WP §§6.2,10.1; W §23.1. **Related / prerequisites:** OQ-01, OQ-33. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-03. Can multi-catalogue prefix caching make grouped requests substantially cheaper?

**Priority:** P0 · **Status:** demonstrated on tested policy panel (U4 / E42)

**Known and still missing.** U3 G/GR incurred zero prefix hits due to single-entry cache alternation (1.42s latency). U4 / E42 implemented an identity-bound multi-catalogue host snapshot cache in host RAM (~170.6 MiB), cutting full-request latency from **1,511.02 ms to 690.16 ms (a ~54% latency cut, 2.19x speedup ratio)** with zero probability drift (**$\Delta p = 0.0$**) across 96 paired runs.

**Test.** Evaluate multi-catalogue host cache eviction policies (LRU/LFU), memory scaling across 10+ distinct schema catalogues, and host-to-device transfer latency under concurrent requests.

**Evidence to close.** Cache stability under high concurrency and memory pressure; automated catalogue eviction without cache pollution; sub-millisecond host-to-device reloading.

**Sources:** U3 §2; U4 §4; W §§9–10,21,23.3. **Related / prerequisites:** OQ-44, OQ-49. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-04. Does catalogue_state_first/v1 transfer to natural tasks at a useful full-request cost?

**Priority:** P0 · **Status:** source reported partial

**Known and still missing.** R29 reports 93/96 versus 71/96 on authored rules and 18/24 versus 0/24 on reference cards while retaining the fitted head, but approximately 2.3× warm scorer time. These are not new U3 results.

**Test.** Recover the pinned R29 experiment and compare the unchanged candidate scorer, catalogue renderer and fixed joint readout on admissible natural tasks, including missing-answer cases. Then time the surviving full scheduler path on its target host.

**Evidence to close.** Natural-task correctness, none recall/false-none and proper scores survive, and complete cost is acceptable. Validate the reported lineage first; four-case CPU/MLX parity is not service qualification.

**Sources:** R: Outcomes of the 2026-09-29 runs; W §§17–18. **Related / prerequisites:** OQ-25, OQ-27. **Execution scope:** Pinned 4B reference on its supported CPU/MLX host; CUDA only as a separately identified comparison.

### OQ-05. Does the chosen useful readout remain stable inside the intended service?

**Priority:** P0 · **Status:** demonstrated on tested readouts (U4 / E42)

**Known and still missing.** U3 demonstrated that dedicated processes eliminate mixed JSON/native history drift for baseline D. U4 / E42 extended this evaluation to the newer readouts (X, XR, and NJ): across all three readouts, dedicated-process execution achieved zero history drift (**$\Delta p = 0.0$**), whereas mixed-process execution showed drift up to $\Delta p = 0.114$.

**Test.** Test dedicated worker stability under sustained concurrent load, client cancellation, variable payload sizes, and process crash recovery. Investigate in-process state reset to remove the need for dual resident processes.

**Evidence to close.** Long-running soak tests with zero drift without process restarts; bounded peak RSS under burst traffic; exact parity between persistent workers and fresh processes under arbitrary request sequences.

**Sources:** U3 §3; U4 §5; W §§21.6, 22.2, 23.4. **Related / prerequisites:** OQ-01 or OQ-02 or OQ-04; OQ-49. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-06. Can the quality-leading formulation use a truly question-independent state root?

**Priority:** P0 · **Status:** partial

**Known and still missing.** Older 4B state-first branching works mechanically; U3’s fast path caches a schema catalogue and remains sensitive to sibling questions. Those are different reuse contracts.

**Test.** Compare full independent calls with state-first shared roots and isolated question-plus-entire-option-list branches. Keep every Qwen recurrent, convolution, attention and position component isolated; evaluate real heterogeneous questions.

**Evidence to close.** Individual-versus-bundled distributions and policies agree for unchanged questions, useful semantic quality survives the changed renderer, and actual state work is amortized. A fast schema-conditioned cache is not a substitute.

**Sources:** W §§11.2,16–17; U3 §§4,6; R: Question Isolation. **Related / prerequisites:** OQ-20, OQ-42. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-07. Which readout wins under a matched learning and rendering contract?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Frozen-head NLI, finite codes, token trees and R29 catalogue scoring have all been tested, but not an attribution-clean comparison of trained vocabulary rows, a finite code head and a pointer head.

**Test.** Fix checkpoint, task splits, input information, answer slots and compute budget. Separate frozen-backbone readout fitting from matched backbone adaptation; compare learned candidate, vocabulary, code-head and pointer variants in a staged screen.

**Evidence to close.** Held-out quality, semantic-none, calibration and full-path cost identify a useful operating point. Declare any unavoidable prompt/training differences instead of calling them a head-only ablation.

**Sources:** W §§4–5,18.17–18.21; WP §4.1; R: Hume reconstruction. **Related / prerequisites:** OQ-17, OQ-25. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-08. Which codebook and rendering choices remain robust outside the inspected fixtures?

**Priority:** P1 · **Status:** partial

**Known and still missing.** U3 finds code/order sensitivity. R29 already separates text positions from letter assignments and finds a four-render ensemble worse; those specific tests are completed, not missing.

**Test.** Lock a small set of codebooks before new evaluation. Test exact boundary tokenization, whitespace/token aliases, multi-token labels, option order, independently assigned codes and duplicate descriptions across models and K.

**Evidence to close.** Correct semantic mapping and measured robustness without sacrificing useful cost. Any ensemble must improve a new gate after paying all forwards; do not reuse the failed four-render recipe as an assumed fix.

**Sources:** R: Outcomes of the 2026-09-29 runs; U3 §4; WP §5.1. **Related / prerequisites:** OQ-21, OQ-40. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.


## B. Determine whether a smaller classifier can replace or complement Qwen

### OQ-09. Can a well-trained BERT-family specialist meet the fixed-schema quality target?

**Priority:** P1 · **Status:** diagnostic blocked at setup (U4 / E42)

**Known and still missing.** E11’s ModernBERT treatments and U3’s tiny bekko/indecis recipe are bounded negative comparisons, not a verdict on BERT, DeBERTa or ModernBERT as families. U4 / E42 attempted a matched ModernBERT-base fixed-schema comparison, but the harness was blocked during initialization by a `SameFileError` exception in `run_logged.publish()` when logging utilities attempted to overwrite in-place. The encoder comparison was terminated before model quality or latency could be evaluated.

**Test.** Fix the logging publisher file-copy defect. Compare BERT-base and ModernBERT-base with frozen features plus a trained head against full or parameter-efficient fitting on the same admissible policy examples.

**Evidence to close.** Useful fresh-family accuracy, whole-request correctness and selective risk at lower measured cost against Qwen XR. Verify tokenizer, pooling, masks, truncation and reference-framework outputs.

**Sources:** W §§15,23.6; U3 §7; U4 §8; BERT; MB. **Related / prerequisites:** OQ-17, OQ-25, OQ-51. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-10. Can a strong pretrained NLI encoder answer dynamic questions accurately enough?

**Priority:** P1 · **Status:** diagnostic blocked at setup (U4 / E42)

**Known and still missing.** The small U3 classifier’s weak NLI does not test a mature NLI-tuned DeBERTa. U4 / E42 scheduled a three-way DeBERTa-v3 NLI arm alongside Qwen, but subprocess execution was blocked by the same `SameFileError` logging exception during setup.

**Test.** Evaluate the named three-way DeBERTa checkpoint directly on NLI once the harness is unblocked. Separately test a binary zero-shot checkpoint on candidate hypotheses, with explicit hypothesis templates and complete state/question information.

**Evidence to close.** Task-appropriate labels and probability semantics, strong unseen-task performance and measured Q×K cost. Preserve neutral where required; candidate softmax must not silently erase no-valid-option cases.

**Sources:** U3 §7; U4 §8; W §23.6; DNLI; DZ. **Related / prerequisites:** OQ-17, OQ-19, OQ-38. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-11. Can a dynamic-label encoder or option-marker model outperform the current trade-off?

**Priority:** P1 · **Status:** diagnostic blocked at setup (U4 / E42)

**Known and still missing.** GLiClass, released Laya and Von are not equivalent to the specific encoder arms already evaluated. U4 / E42 included a GLiClass dynamic-label model in the classifier comparison suite, which was blocked by the logging helper exception.

**Test.** Run the pinned GLiClass uni-encoder checkpoint on common policy cases under the repaired logging harness. Add at most one correctly identified typed-decision marker checkpoint after compatibility checks.

**Evidence to close.** Accuracy, rejection, full distributions, schema sensitivity and whole-request latency on shared cases. Record all text/label truncation and do not convert independent multilabel scores into an unjustified Choice posterior.

**Sources:** R: Laya, Von, Indecis reviews; GL; GLM; U3 §7; U4 §8; W §23.6. **Related / prerequisites:** OQ-21, OQ-26, OQ-38. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-12. Can open-encoder inference share input encoding across questions without changing its function?

**Priority:** P1 · **Status:** untested optimization

**Known and still missing.** U3 open-option caching saves candidate encoding, but Q=16 still costs approximately 16× Q=1 because the input is re-encoded per open question.

**Test.** For a scorer whose input embedding is genuinely independent of the question, compute that embedding once and compare with the unchanged Go path across distinct question sets. Reuse candidates only under exact model/preprocessing identities.

**Evidence to close.** All distributions and selections match and complete request cost falls. This only optimizes the existing bi-encoder function; it does not add listwise reasoning or prove a pooled vector is sufficient for all tasks.

**Sources:** U3 §6; R: Indecis review. **Related / prerequisites:** OQ-50. **Execution scope:** CPU Go implementation first; target-machine comparisons separately.

### OQ-13. Do cross-encoders, late-interaction models or relevance rerankers offer a useful middle ground?

**Priority:** P2 · **Status:** untested matched baseline

**Known and still missing.** The research lists relevance-trained Qwen/zerank comparators, but relevance ranking has not been shown locally to supply typed semantic decisions or calibrated rejection.

**Test.** Compare a single pretrained cross-encoder with a bi-encoder or token-level late-interaction scorer on the same described alternatives. Test retrieval shortlist generation separately from final decision scoring and include hard distractors.

**Evidence to close.** Useful full-decision quality, candidate-recall losses and actual K-scaling are measured. Similarity/relevance scores remain distinct from correctness probabilities until the declared calibration and none tests pass.

**Sources:** R: top-25 reranker rows; WP §3.2; DNLI. **Related / prerequisites:** OQ-10, OQ-19, OQ-21. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-14. Is 9B worth its extra cost over a matched smaller Qwen?

**Priority:** P1 · **Status:** evaluated on publisher profiles (U4 / E42)

**Known and still missing.** Prior runs used different stacks or readouts. U4 / E42 directly evaluated matched publisher profiles for Qwen 4B and 9B under identical conditions: 9B demonstrated clear quality superiority on complex policy tasks (XR: 95.40% field accuracy, 84.38% all-six correctness), while 4B showed a latency advantage on simple classification/NLI tasks (30.15 ms median for 4B on SNLI vs 42.14 ms for 9B).

**Test.** Evaluate whether a fine-tuned or post-trained 4B can match 9B's policy accuracy under XR, and establish the exact throughput/quality frontier on target 16–32 GB deployment hosts.

**Evidence to close.** A quality/resource frontier on the same host and workload, including rejection and class floors, comparing post-trained 4B against 9B.

**Sources:** WP §10.2; U3 §11; U4 §2; W §§15,23.1. **Related / prerequisites:** OQ-01 or OQ-02; OQ-17. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-15. Does a non-Qwen pretrained backbone change the best operating point?

**Priority:** P2 · **Status:** untested matched comparison

**Known and still missing.** The research contains frozen Gemma-style finite-logit systems and encoder-decoder alternatives; their external results are not local matched evidence.

**Test.** After locking a useful Qwen baseline, select one resource-admissible frozen alternative. A T5-style encoder-decoder is a separate architectural test; changing where questions enter requires its own rendering/training contract.

**Evidence to close.** A useful gain on the same task and cost envelope after native-template and probability audits. Avoid a many-model leaderboard and do not attribute checkpoint/pretraining differences solely to architecture.

**Sources:** R: Cygnet, Winnow and general-model survey; T5; WP §10.2. **Related / prerequisites:** OQ-14, OQ-17. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-16. Can a compact-first cascade reduce cost without hiding its rejected or wrong cases?

**Priority:** P1 · **Status:** untested integration

**Known and still missing.** U3 compact fixed heads are quick but transfer poorly; most selected compact policies accept nothing. A cascade is not already useful merely because its first model is cheap.

**Test.** Compare always-Qwen, compact-only and a development-locked compact→Qwen→review cascade. Include both inference costs, routing overhead, fallback cold starts, correlated errors and all denied/accepted cases.

**Evidence to close.** Lower measured expected cost and useful coverage at the same preregistered risk/quality requirements. A confidence router must beat trivial always-Qwen and review-all controls on fresh confirmation.

**Sources:** U3 §§7,10; WP §§5.3,10.3. **Related / prerequisites:** OQ-09 or OQ-10; OQ-36, OQ-37. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.


## C. Establish the actual semantic scope of a Jev alternative

### OQ-17. Does the model follow genuinely new questions and policies, not just new wording?

**Priority:** P0 · **Status:** partial

**Known and still missing.** U3 confirmation changes authored rendering families, while fixed indecis heads learn a fixed schema. Neither establishes arbitrary caller-defined question/rubric generalization.

**Test.** Hold out complete policy, instruction, task and source families. Include the same state paired with questions that require different answers, new candidate criteria and explicit counterfactual policy changes.

**Evidence to close.** Useful per-family quality and question sensitivity on independent confirmation without retraining per new question. Distinguish new text, new labels, new policies and pretraining-unknown knowledge.

**Sources:** U3 §1; W §§2.4,11.6; R: evaluation framework. **Related / prerequisites:** OQ-25. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-18. What level of option interaction is necessary, and which sensitivities are defects?

**Priority:** P1 · **Status:** partial

**Known and still missing.** R29 catalogue and Hume reference-card probes show why independent candidate scores can miss information elsewhere in the same option list. Option-set changes need not preserve pairwise odds.

**Test.** Compare independent, catalogue-conditioned and joint scorers using value-switch cards, equal-K description interventions, duplication and clearly irrelevant options. Factor text order, output codes and list-dependent calibration separately.

**Evidence to close.** The reader uses task-relevant rival information without unacceptable order or decoy effects. Do not impose independence of irrelevant alternatives as a universal semantic requirement or infer a unique mask from API behavior.

**Sources:** R: Hume reconstruction and R29 outcomes; H. **Related / prerequisites:** OQ-04, OQ-08. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-19. Can one declared probability contract distinguish omitted answers, insufficient evidence and out-of-scope input?

**Priority:** P0 · **Status:** partial

**Known and still missing.** None-head fitting improved Banking omission cases, but natural-document and transfer failures remain. A none code or low confidence is not evidence for every form of uncertainty.

**Test.** Build separately labeled present/omitted, contradictory/incomplete evidence and author-OOS strata. Compare the retained scalar/set-aware controls with one evidence-aware candidate, using fixed prevalence assumptions and false-none limits.

**Evidence to close.** Useful recall and false-positive behavior in every declared stratum, complete offered-option-plus-none mass and favorable transferred policy cost. Preserve review as an application action, not automatically a semantic class.

**Sources:** W §§5,10,18; WP §5.2; U3 §10. **Related / prerequisites:** OQ-25, OQ-33, OQ-38. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-20. Are independent questions informationally isolated, including opaque identifiers?

**Priority:** P0 · **Status:** partial

**Known and still missing.** U3 unrelated-question additions change answers. Older state-first fixtures pass narrow isolation tests; they do not qualify the new schema-conditioned quality path.

**Test.** Place a unique fact in a sibling question, then in state, with literal matched wording and positive controls. Test individually versus bundled, duplicate/reordered questions, hidden-ID renames and visible semantic-key renames as distinct transformations.

**Evidence to close.** A question reads allowed state and its own instructions/options but not sibling-only facts; hidden IDs do not alter model input. Check recurrent state as well as attention masks and apply the declared numerical/answer contract.

**Sources:** U3 §4; W §§11.2,17.3; R: Question Isolation; API. **Related / prerequisites:** OQ-06. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-21. Does high-cardinality Choice retain semantic accuracy and useful rejection?

**Priority:** P1 · **Status:** partial

**Known and still missing.** K up to 255 has prior native admission/continuation mechanics evidence; U3’s K grid uses simple item facts. Neither settles close semantic alternatives at high K.

**Test.** Increase K in stages with real, distinct descriptions, hard negatives and answer-absent pairs. Test K=1 where supported, overlapping meanings, description budgets and candidate shortlisting; inspect exactly which option text remains visible.

**Evidence to close.** Accuracy, proper scores, none recall, false-none and full latency/memory at each K. Mechanical support or unchanged argmax on a copied-answer probe is not high-K generalization.

**Sources:** W §§17.3,11.7.1; U3 §6; R: cardinality sweep. **Related / prerequisites:** OQ-19, OQ-26. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-22. Can Noul handle unseen binary criteria without conflating false with unknown?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Bounded BoolQ and other binary probes already ran. They do not establish general yes/no judgment on arbitrary criteria or incomplete evidence.

**Test.** Test new positive/negative criteria, negation, nested conditions, counterexamples and missing evidence. Compare native binary readouts with explicit two- or three-way decisions only under separately declared task mappings.

**Evidence to close.** Per-criterion discrimination, probability quality and appropriate uncertainty behavior. Do not reinterpret an unanswerable fact as false merely to fit a binary interface.

**Sources:** W §§2,13.1; WP §5.2; API. **Related / prerequisites:** OQ-17, OQ-19. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-23. Can Score produce meaningful ordinal distributions for unfamiliar rubrics?

**Priority:** P1 · **Status:** partial

**Known and still missing.** SST-5 and bounded ordinal probes are completed, not absent. They do not establish transfer to new level descriptions, irregular spacing or operational severity scales.

**Test.** Compare categorical NLL, ordinal cumulative losses and fixed rubric verbalizations on held-out rubrics. Preserve full level distributions; evaluate expected score, modal level, threshold events and declared cost separately.

**Evidence to close.** Ordinal error and probability quality improve without collapsing to a point estimate. Distinguish the mean from the selected class and define unsupported/non-consecutive level behavior in the API.

**Sources:** W §§6.3,13.1; R: ordinal scoring; API. **Related / prerequisites:** OQ-17, OQ-30, OQ-33. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-24. Does strong atomic accuracy translate into useful closed-loop decisions?

**Priority:** P2 · **Status:** untested target scope

**Known and still missing.** Raschka’s examples motivate broad decision use, but neither sentiment accuracy nor U3’s independent policy cases measures a multi-step controller.

**Test.** Use a bounded simulator or workflow state machine with reproducible seeds, fully specified visible observations and legal actions. Compare frozen policies on episode success, cumulative cost, invalid actions and latency; separate full from partial observability.

**Evidence to close.** Better complete-episode outcomes under the same observation and action contract. No future-state leakage, gold trajectory initialization or inference from one attractive game demo.

**Sources:** S; R: SalesRL leakage audit and generality discussion; W §11.6. **Related / prerequisites:** OQ-17, OQ-36, OQ-55. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.


## D. Improve learning while preserving evidence and retained tasks

### OQ-25. Are the labels, criteria and model-visible evidence independently sufficient?

**Priority:** P0 · **Status:** partial

**Known and still missing.** The natural-document audit found alignment and evidence issues; recorded source consistency and quarantine counts are not independent semantic adjudication.

**Test.** Complete a blinded criteria/evidence review on admissible non-final material, preserving disagreements and quarantines. Record source spans, document groups, answerability meanings and adjudication versions before constructing new confirmation.

**Evidence to close.** An auditable task contract with reviewed evidence and independent split/group checks. Do not silently relabel old errors or open protected final data to make a model pass.

**Sources:** W §§3.3,18.14–18.26; WP §§4.4,5.2. **Execution scope:** CPU/document review; human adjudication remains a distinct requirement.

### OQ-26. How much accuracy is lost through token budgets and evidence placement?

**Priority:** P0 · **Status:** partial

**Known and still missing.** Longer input budgets recover some evidence-sensitive decisions but can worsen rejection. Standard encoder and decoder token limits are not comparable by nominal token counts alone.

**Test.** Measure effective text/option coverage after each tokenizer. Pair evidence-first/middle/last, distractor and overlength variants; compare a common-content subset separately from each model’s maximum-capacity configuration.

**Evidence to close.** Coverage and quality are reported together, with truncation/unsupported inputs explicit. Faster results obtained by deleting required evidence do not qualify as equal-quality acceleration.

**Sources:** W §18.20; WP §4.4; U3 §6; MB; DZ. **Related / prerequisites:** OQ-25. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-27. Can evidence selection reduce input cost without losing decisive information?

**Priority:** P1 · **Status:** untested controlled treatment

**Known and still missing.** Visibility diagnostics identify a problem; they do not validate retrieval, hierarchical reading, chunk aggregation or a learned evidence selector as its solution.

**Test.** Compare complete supported input, gold-evidence windows as an explicitly nondeployable upper bound, and automatic retrieval/windowing under the same downstream scorer. Include distributed evidence, negation and answer-absent documents.

**Evidence to close.** End-to-end quality and retained-none behavior after retrieval misses, with selector/index/aggregation costs included. Oracle excerpts must not be reported as achievable deployed accuracy.

**Sources:** WP §4.4; W §18.20; R: long-context evaluation. **Related / prerequisites:** OQ-25, OQ-26. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-28. Can task-aligned adaptation preserve supported entailment and QASPER competence?

**Priority:** P1 · **Status:** failed treatments open problem

**Known and still missing.** The contract-only, parent-KL and source-label replay recipes already ran and failed the joint retention screen. U3 compact SNLI replay is a different experiment.

**Test.** Diagnose supported-to-none versus supported-to-contradiction errors on admissible training/development data. Choose one bounded intervention exercising complete-document operations; compare frozen parent, old specialist and matched treatment with class-aware selection.

**Evidence to close.** Prespecified in-domain gains and all retained-source/class/probability/policy gates pass on new groups. Keep frozen parent selectable and leave historical selected-zero/update-80 identities unchanged.

**Sources:** W §§18.21–18.26; WP §§4.2–4.4; U3 §7. **Related / prerequisites:** OQ-25, OQ-26. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-29. Which data diversity and augmentation actually improve task-family transfer?

**Priority:** P1 · **Status:** partial

**Known and still missing.** U3’s compact fit learns development templates much better than new renderings. External hard-case recipes combine many changes, so they do not isolate the responsible ingredient.

**Test.** Hold backbone and exposure budget fixed while varying one of task-family diversity, hard-negative coverage, question paraphrases, label permutations or evidence-position augmentation. Reserve whole sources and rubric families, not only random rows.

**Evidence to close.** Consistent new-family gains across seeds without class/none regressions. Do not let template variants leak across splits or select an augmentation based on the confirmation set it is meant to test.

**Sources:** U3 §7; WP §§3.3,4.3; R: training mixture. **Related / prerequisites:** OQ-17, OQ-25. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-30. Do proper-score training objectives improve reliability beyond ordinary NLL?

**Priority:** P1 · **Status:** untested matched objectives

**Known and still missing.** NLL-trained heads work in bounded settings; a specific NLL+Brier/RPS coefficient is not an established OpenKind recipe.

**Test.** Compare NLL, NLL plus Brier and an ordinal RPS variant only where the labels are ordered. Keep data, model, updates and selection budget matched; retain raw and independently gated post-hoc calibration controls.

**Evidence to close.** Fresh probability and decision-utility gains without unacceptable accuracy/retention cost. Never optimize binned ECE alone or describe expected ordinal distance as a proper distribution loss.

**Sources:** R: Phase 2 supervised training; WP §5.3; CAL. **Related / prerequisites:** OQ-25, OQ-28. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-31. Can a validated teacher produce a smaller accurate student?

**Priority:** P1 · **Status:** untested matched supervision

**Known and still missing.** The record motivates distillation but does not show a broadly useful student or establish teacher predictions as ground truth.

**Test.** Validate the teacher per target family, then compare gold-only fitting, reviewed teacher-written hard cases and soft-distribution distillation under the same student/data budget. Track teacher-gold disagreement and retain an independent human/source-labeled confirmation.

**Evidence to close.** Student useful-quality floors, uncertainty and resource gains all survive; agreement with the teacher alone is insufficient. Report supervision provenance, license scope and teacher generation cost.

**Sources:** WP §§3.3,4.3; W §11.7; R: external training recipes. **Related / prerequisites:** OQ-09 or OQ-14; OQ-25, OQ-29. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-32. Does decision-utility RL add anything beyond supervised and direct utility training?

**Priority:** P2 · **Status:** unresolved incremental value

**Known and still missing.** No local result isolates a benefit from TypeSafe-style RLCD, whose algorithm is undisclosed. RLCR is a different public method, not a reproduction of RLCD.

**Test.** Only after a useful supervised baseline, compare matched NLL/proper-score/calibration controls with a bounded utility-trained treatment. When labels determine every action’s reward, include a direct expected-utility objective rather than giving RL an artificially weak control.

**Evidence to close.** An incremental fresh-family selective-risk or utility gain at accounted training/inference cost, with retained probability semantics and task quality. Name the local method independently; do not claim proprietary recipe recovery.

**Sources:** R: CADO ablation; WP §5.3; RLCR. **Related / prerequisites:** OQ-28, OQ-30, OQ-36. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.


## E. Make uncertainty and downstream policy useful

### OQ-33. Which calibration transforms transfer across renderer, domain and cardinality?

**Priority:** P0 · **Status:** partial

**Known and still missing.** U3 accepted temperatures help D/X but hurt compact policy transfer. R29’s successful temperature fails on reversed rendering. Calibration has both positive and negative local evidence.

**Test.** Fit on calibration-fit, accept on a separate gate and evaluate once on new domains/renderings/K. Compare identity with scalar, carefully regularized conditional and joint-source calibration; preserve raw outputs and transform identity.

**Evidence to close.** Proper-score improvements and useful policy behavior in the intended scope, not merely a lower pooled ECE. No transform is silently reused across changed prompts, kernels or candidate sets.

**Sources:** U3 §9; R: R29 outcomes; WP §5.3; CAL. **Related / prerequisites:** OQ-17, OQ-25. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-34. What exactly does each exposed confidence or score estimate?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Class probability, constrained token-tree probability, similarity, concentration and a generated confidence string are different quantities. The API shape cannot establish their statistical meaning.

**Test.** Inventory every field’s source, normalization, offered-option conditioning, rounding and calibration. Evaluate reliability for the actual event claimed, including any transformations and exclusion of none mass.

**Evidence to close.** Each number has a versioned probability/event definition and empirical support appropriate to its use. Keep concentration summaries distinct from P(correct) and do not relabel similarity as calibration.

**Sources:** W §2; WP §§4.1,5.3; R: confidence boundary; CONF. **Related / prerequisites:** OQ-38, OQ-52. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-35. Can whole-request correctness be estimated without pretending the fields are independent?

**Priority:** P1 · **Status:** untested reliability model

**Known and still missing.** U3 confirms that minimum selected-field probability is not P(all six correct). Deterministic dependencies also make naive products and duplicated evidence problematic.

**Test.** Compare conservative bounds, explicitly modeled small joint events and a held-out whole-request error predictor. Calibrate dependencies at their source; keep point-only arms out of exact-vector claims.

**Evidence to close.** Better risk/coverage on fresh request groups, with well-defined event probabilities and no test-set fitting. More confident marginal estimates are not automatically a better estimate of joint success.

**Sources:** U3 §§2,9–10; WP §6.2. **Related / prerequisites:** OQ-02, OQ-33, OQ-37. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-36. Does a fixed action/review policy lower real expected cost at useful coverage?

**Priority:** P0 · **Status:** partial

**Known and still missing.** Under U3’s illustrative 5/0.1 losses, D and G fail to beat review-all, X has little coverage and NLI_CODE has a promising bounded result. Those losses are not measured business costs.

**Test.** Specify application losses, review capacity/quality and fallback latency before fitting policy thresholds. Compare frozen policies with review-all, always-answer and any cascade on new task groups across plausible prevalence shifts.

**Evidence to close.** Lower total expected cost with a declared error bound and useful coverage, including delay and fallback costs. Zero accepted cases have undefined conditional error; do not select a new threshold on confirmation.

**Sources:** U3 §10; W §6; WP §5.3. **Related / prerequisites:** OQ-17, OQ-33, OQ-37. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-37. Is there enough independent evidence to support the required error bound?

**Priority:** P0 · **Status:** partial

**Known and still missing.** Two repeats, many fields, option rotations and multiple checkpoints do not create independent examples. Zero errors among 61 selected SNLI cases does not certify rare deployment errors.

**Test.** Preregister the unit of independence, minimum effect/risk requirements, sample size and selection rule. Use source/document-clustered paired uncertainty, multiple seeds where fitting matters, and a separately reserved release set.

**Evidence to close.** The interval is narrow enough for the actual decision and the selection/evaluation boundary remains intact. Any conformal or selective-risk guarantee must state its assumptions and not claim robustness to arbitrary distribution shift.

**Sources:** U3 §§1,10; W §3.3; WP §2. **Related / prerequisites:** OQ-17, OQ-25. **Execution scope:** CPU analysis plus newly collected/adjudicated evidence; not solved by more duplicate GPU calls.

### OQ-38. Can different classifier probability spaces be compared without changing the task?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Exclusive Choice, independent multilabel membership, NLI entailment, relevance scores and offered-options-plus-none are not interchangeable softmax outputs.

**Test.** Define a semantic adapter for each challenger before evaluation. Test absent correct labels, multiple valid labels, duplicated/synonymous choices, changing K and one-candidate inputs. Preserve raw logits and any withheld mass.

**Evidence to close.** Normalization and decisions implement the declared event space. Unsupported multi-valid/unknown cases are explicit; no post-hoc renormalization turns a ranking improvement into full-decision correctness.

**Sources:** W §§2,5; R: probability contract; GL; DNLI; DZ. **Related / prerequisites:** OQ-19, OQ-21. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-39. How will uncertainty and policy degrade as the workload changes?

**Priority:** P1 · **Status:** untested deployment scope

**Known and still missing.** The experiments already show calibration and rejection transfer failures, but there is no qualified online drift/recalibration process for the new service.

**Test.** Replay controlled task/prevalence/language shifts with delayed labels and frozen policies. Test alarms, abstention or fallback behavior and a versioned recalibration workflow that uses only permitted feedback data.

**Evidence to close.** Measured detection/response costs and maintained task-specific risk under the declared shifts. Do not assume detecting an embedding shift proves error growth, or that silently updating temperatures preserves the operating contract.

**Sources:** U3 §§7,9–10; WP §§5.2–5.3. **Related / prerequisites:** OQ-33, OQ-36, OQ-54. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.


## F. Reduce complete-request cost while preserving the selected function

### OQ-40. How much useful work can be removed from the output projection and host path?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Avoiding decoding works; the tied 4B embedding/output matrix has no marginal removable weights. That does not settle selected-row projection cost or another checkpoint’s storage layout.

**Test.** Profile and compare complete vocabulary projection with selected-row or learned-head projection under identical hidden states and label mappings. Measure tokenization, schema preparation, IPC/HTTP and serialization separately and end to end.

**Evidence to close.** Equivalent declared distributions plus actual full-path savings. Verify tied versus untied weights and physical allocations; reduced logical output dimensions are not automatically reduced model memory.

**Sources:** W §§4.2,8; U3 §2; WP §8.2. **Related / prerequisites:** OQ-07, OQ-08. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-41. Which optimized kernels or captured graphs are genuinely faster and correct on this profile?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Earlier reference fallbacks and slower custom-Metal/flat-field trials leave room for improvement. Requesting graphs disabled did not repair E40 history, but that is not a graph-speed ablation.

**Test.** Measure actual dispatched DeltaNet/convolution/attention/matmul kernels, compile/warmup cost and captured-graph reuse. Compare one implementation change at a time against the numerical reference under representative input lengths and shapes.

**Evidence to close.** Complete latency/throughput gains with unchanged-probability/answer/policy gates. A configuration flag, fewer forward calls or an upstream benchmark cannot substitute for local dispatch and timing.

**Sources:** W §§3.1,17.4,21.6; WP §7.3; HYD; DEFT. **Related / prerequisites:** OQ-05, OQ-42. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-42. Which execution plan wins across real Q, K, state length and load?

**Priority:** P1 · **Status:** partial

**Known and still missing.** State sharing already helps some long workloads and loses on some short ones. U3 simple-value Q scaling does not identify a production scheduler.

**Test.** Compare repeated-full, within-question candidate batches, state-first question branches and approved grouped plans across real workloads. Include unequal suffixes, padding, batch occupancy, queue delay and cold/warm states.

**Evidence to close.** A preregistered scheduler or crossover rule improves held-out workload cost while retaining quality and execution gates. Measure service p50/p95/p99 and throughput under a latency target, not only synthetic mean slope.

**Sources:** W §§8–10,16–17; U3 §6; WP §7. **Related / prerequisites:** OQ-06, OQ-21, OQ-49. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-43. Which weight, activation and cache precision profiles offer the best valid trade-off?

**Priority:** P1 · **Status:** partial

**Known and still missing.** BF16/TF32 and several cache codecs have specific failed-equivalence results; Q4 runs on T4; U3 Go INT8 did not execute. None settles matched Q4/Q8 or another encoder runtime.

**Test.** Compare pinned quantizations and arithmetic modes sequentially under actual hardware support. Separate weight error, execution-shape error and cache-storage error; audit exported encoder kernels and retained recurrence precision.

**Evidence to close.** Both semantic quality and resource gains are reported. An intentionally changed numerical model needs new quality/calibration validation; a claimed transparent implementation must also pass unchanged-output gates. Quantization calibration data must remain separate from probability calibration.

**Sources:** W §§7,9–10,20–21; U3 §8; ORT. **Related / prerequisites:** OQ-14 or OQ-09; OQ-33. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-44. Does cache identity and lifecycle remain correct for new renderers and multi-entry execution?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Older native cache identity/lifecycle checks exist. They do not automatically validate new model heads, catalogue variants or the proposed grouped-cache integration.

**Test.** Bind complete finalized token prefixes, model/adapter/head, tokenizer, renderer, positions, precision, backend and tenant boundaries. Exercise reuse, concurrent readers, mutation attempts, exact expiry, oversize bypass and adapter/schema changes.

**Evidence to close.** No false hit, root mutation or cross-identity reuse; output and lifecycle tests pass through interruption/recovery. Bound real retained and temporary bytes rather than only nominal cache payload.

**Sources:** W §§9–10,11.4,17.3; U3 §2. **Related / prerequisites:** OQ-03, OQ-49. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-45. Can the mixed-process defect be fixed without keeping two resident models?

**Priority:** P1 · **Status:** failed control with workaround

**Known and still missing.** Dedicated processes and full restart are demonstrated workarounds: U3 and U4 confirm dedicated resident processes pass history tests with $\Delta p = 0.0$ across D, X, XR, and NJ, whereas mixed-process execution exhibits drift up to $\Delta p = 0.114$. However, dual resident processes require ~12.10 GiB VRAM on a 15-GiB T4. The underlying in-process history contamination mechanism is still unidentified; same-process reset without model reload remains unresolved.

**Test.** Minimize the preserved failing trace, compare full execution-context replacement with narrower resets, and localize the earliest divergent hidden/state tensors. Keep a failing matched control and independent fresh-reset oracle.

**Evidence to close.** A causally attributable repair passes broader histories and costs less than the dedicated-process reference without stale recurrence or shared mutable buffers. No root-cause claim from one final-logit comparison.

**Sources:** U3 §3; U4 §5; W §§21.6, 22.2, 23.4. **Related / prerequisites:** OQ-05. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-46. When is persistent CPU/GPU storage worth its transfer, eviction and startup cost?

**Priority:** P1 · **Status:** demonstrated for host RAM multi-catalogue snapshots (U4 / E42)

**Known and still missing.** Lossless persistence already has bounded trace gains. U4 / E42 demonstrated that host RAM snapshot caching (~170.6 MiB) is highly profitable for multi-catalogue grouped requests, cutting full-request latency by ~54% (2.19x speedup ratio) with $\Delta p = 0.0$ across 96 paired runs. However, disk persistence and larger state transfer economics under fluctuating concurrent traffic remain unmeasured.

**Test.** Replay a declared realistic state reuse distribution through no-persistence, GPU and host-backed caches, adding disk only after transfer profiling. Include cold load, misses, expiry, scans, fragmentation and resource admission.

**Evidence to close.** Actual trace/service cost and memory improve while exact identities and selected behavior survive. Distinguish state storage from model-weight streaming and do not infer traffic hit rates from a repeated-anchor test.

**Sources:** W §§9–10,17.3,23.3; WP §§7.1,8.2; U4 §4. **Related / prerequisites:** OQ-44, OQ-49. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-47. Can incrementally updated state be reused safely and profitably?

**Priority:** P2 · **Status:** untested target scope

**Known and still missing.** Exact prefix reuse is established in narrower settings. No supplied result qualifies arbitrary document edits, rolling observations or newly appended facts for the selected fast profile.

**Test.** Compare full recomputation with reuse of the verified longest unchanged token prefix for append, edit and delete operations. Test changed evidence, corrected facts, tokenizer boundary merges and explicit history summarization as a different model input.

**Evidence to close.** Identical finalized-input execution is preserved for exact reuse; semantic state changes produce appropriate new answers. Never reuse stale state because texts are merely similar or treat a summary as exact cache equivalence.

**Sources:** W §§8,11.4; R: shared-state program. **Related / prerequisites:** OQ-06, OQ-44. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-48. Does a useful new profile retain its advantage in the Rust/MLX deployment path?

**Priority:** P1 · **Status:** partial

**Known and still missing.** The immutable 4B CPU service and pinned MLX FP32 parity already have bounded evidence. They do not qualify the newer 9B token reader or a changed catalogue/code head.

**Test.** Port one selected useful profile through token, full-forward, head, branch, batched and service parity stages. Measure complete requests, process startup, unified-memory pressure and throughput on the actual Apple Silicon host.

**Evidence to close.** Target-host quality, probability/policy agreement and end-to-end latency/memory meet the declared scope. No CUDA-to-Mac timing transfer and no restart of already completed immutable-reference bring-up.

**Sources:** W §§17.3–17.4; WP §9; R: R29 outcomes; U3 §3. **Related / prerequisites:** OQ-04 or OQ-05; OQ-49. **Execution scope:** Actual Apple Silicon/Rust/MLX host; cannot be closed by a T4/L4 Colab result.

### OQ-49. Does the integrated service remain correct under load, cancellation and memory pressure?

**Priority:** P0 · **Status:** partial

**Known and still missing.** Older named-machine CPU service tests exist; U3 dual residency is not simultaneous-load or long-running qualification of its new CUDA readout.

**Test.** Run queue-inclusive concurrent traffic with deadlines, cancellation, worker failure, overload, long inputs and repeated schema/state changes. Observe true or clearly sampled high-water memory, live references and cleanup after failures.

**Evidence to close.** Bounded queue/process memory, no incorrect response association or poisoned state, explicit overload/unsupported responses and acceptable tail latency. Preserve permits until actual work ends; completion alone is not service acceptance.

**Sources:** W §17.3; U3 §3. **Related / prerequisites:** OQ-05, OQ-44. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.


## G. Close conformance, evidence and deployment boundaries

### OQ-50. Can every claimed result be reproduced from an immutable execution identity?

**Priority:** P0 · **Status:** partial

**Known and still missing.** The project has extensive hashes and reconciliation assets, but new R29 claims and future alternative checkpoints need their own raw artifacts and active-code provenance.

**Test.** Maintain one manifest linking source, model weights, tokenizers, data groups, compiled binaries, kernels, renderer, heads and policies. Verify real pretrained bridge outputs, not just interface stubs; record unsupported and failed treatments separately.

**Evidence to close.** A clean host can reproduce the specified scope and answer the question ledger without silently refitting, substituting models or confusing completed blocks with scientific passes. Source synthesis is labeled when raw re-audit was not done.

**Sources:** W Appendix A; U3 §1; R: R29 outcomes. **Execution scope:** CPU evidence checks plus each actual backend; separate build feasibility from model execution.

### OQ-51. Are candidate rankings stable across seeds, machines and evaluation choices?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Some earlier head studies have multiple seeds, but U3’s short authored confirmation and many comparisons do not establish a universal architecture ranking.

**Test.** Predeclare comparisons and practical effect sizes; repeat only surviving candidates across training seeds and separate evaluation groups. Balance arm order and thermal/cache conditions, and compare kernels on a single host before cross-host aggregation.

**Evidence to close.** Uncertainty respects source groups, seeds and task families. Report all preregistered candidates, negative results and resource failures; do not tune on the confirmation set or reinterpret nonsignificance as equivalence.

**Sources:** W §3.3; U3 §§1–2; WP §2. **Related / prerequisites:** OQ-17, OQ-37. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-52. Does the implementation match the chosen typed API and confidence semantics?

**Priority:** P0 · **Status:** partial with documented mismatch

**Known and still missing.** The whitepaper documents a legacy normalized-entropy confidence, whereas the inspected official adapter uses another Choice formula. This flags a compatibility question, not a freshly verified defect in current source.

**Test.** Freeze a reference API/adapter version and differential fixtures for Choice/Noul/Score, none handling, hidden IDs, ties, rounding, invalid input and unsupported shapes. Audit active implementation rather than assuming the old documentation is current.

**Evidence to close.** Either conformance passes or intentional differences are versioned and documented. Numerical confidence, field names and probability spaces must agree; no extra class or renormalized-away none mass is silently inserted.

**Sources:** W §§2,11.8,17.3; R: confidence boundary; API; CONF. **Related / prerequisites:** OQ-34, OQ-38. **Execution scope:** CPU/API conformance suite first; actual-model fixtures where identifiers or rendering can affect inference.

### OQ-53. Can untrusted state or option text corrupt the intended decision semantics?

**Priority:** P1 · **Status:** unqualified security scope

**Known and still missing.** Typed serialization prevents malformed output, not prompt injection, false evidence or fake option boundaries. Existing task benchmarks do not qualify an adversarial boundary.

**Test.** Use authorized defensive cases with instruction-like state text, forged markers, Unicode variants, distracting reference options and conflicting priorities. Keep trusted policy separate and compare perturbed cases with independently reviewed expected outcomes.

**Evidence to close.** Explicit semantic robustness and failure handling at the accepted risk scope. Treat evidence authenticity separately from model agreement; no claim that choosing only allowed options prevents harmful decisions.

**Sources:** W §11.4; R: structural validity and authority boundaries. **Related / prerequisites:** OQ-20, OQ-25, OQ-52. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-54. Are tenant, privacy and observability boundaries preserved by the new fast paths?

**Priority:** P1 · **Status:** partial

**Known and still missing.** Legacy tenant-key/lifecycle checks are useful assets, not a complete privacy audit of multi-worker caches, traces and newly exported models.

**Test.** Exercise cross-tenant key collisions, restart/eviction/deletion, log redaction and bounded input retention. Audit authentication, exposed metrics and permissions separately from content hashes; test deployment defaults without sending private evidence to public comparison services.

**Evidence to close.** No unauthorized reuse/disclosure and a documented retention/authorization policy, with explicit limits on timing side-channel assurance. A digest or numerical parity pass is not access control.

**Sources:** W §§11.4,11.8,17.3; R: scoped authority. **Related / prerequisites:** OQ-44, OQ-49. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-55. Can downstream actions remain correct and authorized when the classifier is wrong?

**Priority:** P1 · **Status:** unqualified consumer scope

**Known and still missing.** The research separates semantic decisions from tool authority. No classifier accuracy figure proves that an offered action is lawful, permitted or successfully executed.

**Test.** Test a consumer with deterministic eligible-action filtering, independent authorization, dependency checks and verified postconditions. Include classifier errors, stale state and ambiguous criteria; replay outcomes without performing irreversible real-world actions.

**Evidence to close.** Fail-closed authority and task-specific verified effects, with model error accounted for in policy cost. Keep permissions and irreversible execution outside the probability model.

**Sources:** R: Agent consumers and scoped authority; W §§2,11. **Related / prerequisites:** OQ-24, OQ-36, OQ-53. **Execution scope:** CPU workflow/simulator and authorized service integration tests.


## H. Conditional extensions—not prerequisites for the next useful result

### OQ-56. Does a bounded reasoning or deterministic-tool budget pay for itself?

**Priority:** P2 · **Status:** untested matched budget

**Known and still missing.** Jeeves-style reasoning and Von-style bounded calculations are external leads. OpenKind has not measured their incremental quality/cost on matched untouched tasks.

**Test.** Compare no-reasoning decisions with capped reasoning, evidence-grounded deterministic calculations and a development-locked fallback. Include the actual generated tokens, tool calls, memory and uncertainty calibration.

**Evidence to close.** Better complete-task utility at a stated extra cost. MTP/speculation can accelerate an actually generated reasoning sequence, not the existing zero-output readout; fixed tool confidence is not calibrated probability.

**Sources:** R: Jeeves and Von reviews; WP §10.3. **Related / prerequisites:** OQ-16, OQ-36. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-57. Can MoE capacity or expert streaming improve this workload’s quality/resource frontier?

**Priority:** P2 · **Status:** partial capacity untested streaming

**Known and still missing.** A modern MoE profile improved bounded quality at greater latency; earlier skipping did not unload weights. Few active experts per token did not imply few request-wide experts.

**Test.** First measure request-wide expert locality, resident bytes and transfer volume for decision prefills. Only then compare fixed cache/residency/streaming policies against the complete resident model, including warm and cold traffic.

**Evidence to close.** Measured memory or latency gains with retained quality and explicit arithmetic differences. Hume’s MoE hypothesis is not evidence that streaming will help on a T4 or Mac.

**Sources:** W §19; WP §8.3; R: sparse capacity. **Related / prerequisites:** OQ-42, OQ-46. **Execution scope:** A host that fits the baseline or a separately declared capacity study; not silently substituted into the current Colab.

### OQ-58. Are multimodal and diffusion decision paths useful for a separately defined need?

**Priority:** P2 · **Status:** untested target scope

**Known and still missing.** The research surveys image/audio/video and diffusion-based systems, but text-only results do not establish those modalities or their costs.

**Test.** Only with a defined modality requirement, use matched source-backed image/table/audio cases and typed decisions. Compare direct readout, text extraction where admissible and diffusion/iterative alternatives with full preprocessing and iteration accounting.

**Evidence to close.** A modality-specific quality and cost benefit on clean held-out evidence. Model-family support or API shape does not prove task capability; denoising acceleration is a separate experiment.

**Sources:** R: djev, Jev-Omni, Jeeves and general-model survey; WP §10.3. **Related / prerequisites:** OQ-25, OQ-60. **Execution scope:** Separate capability/admission check; no claim that current text-only T4 measurements transfer.

### OQ-59. Can a richer shared-state reader remove expensive question-conditioned work?

**Priority:** P2 · **Status:** failed shortcuts open architecture

**Known and still missing.** Frozen pooled-state shortcuts failed and token-level query modules were partly tested without clearing the full gates. State cache reuse is not proof that a tiny summary is sufficient.

**Test.** Specify a materially different token-level representation, question/candidate cross-attention or encoder-decoder reader, with evidence-sensitive supervision. Compare against full question-conditioned computation and the existing failed shortcuts rather than removing them from the baseline set.

**Evidence to close.** Matched useful quality, rejection and real complete-request savings with a trained, explicit information-flow contract. A bidirectional state cache cannot be called exact reuse of a jointly encoded input after queries change.

**Sources:** W §§11.2,18.3–18.16; WP §4.5; BERT; T5. **Related / prerequisites:** OQ-17, OQ-25, OQ-28. **Execution scope:** T4/L4; sequential model loading unless the test explicitly requires otherwise.

### OQ-60. What would justify calling the finished system a fast, accurate Jev alternative?

**Priority:** P0 RELEASE · **Status:** open release gate

**Known and still missing.** Neither the older papers, U3 nor the new R29 summary establishes independent release-grade generalization or exact proprietary architecture reproduction.

**Test.** Freeze one complete product profile and independently reviewed release tasks, including unseen questions, none/rubric cases and realistic traffic. Compare strong local classifiers and, where access and data authorization permit, a pinned Jev API with matched inputs and clearly separate client/service timing.

**Evidence to close.** All declared semantic, probability, policy, isolation, resource and operational requirements pass without post-hoc gate changes. Report conditional superiority and unsupported workloads; do not infer hidden topology or training from API similarity.

**Sources:** R: reproduction boundaries; W §§2,13,17–18; U3 §12; H; S. **Related / prerequisites:** All requirements relevant to the explicitly declared release scope; OQ-17, OQ-25, OQ-37, OQ-49, OQ-52. **Execution scope:** Independent review and target deployment hardware; external API calls require separate credentials, cost and data authorization.


## 7. Crosswalk to the original research program

The register covers unresolved extensions of the original research ideas, not just the five unmeasured rows in the Unified ledger.

| Original research theme | Relevant outstanding questions | Already addressed boundary |
|---|---|---|
| Shared-state/set-query architecture | OQ-04,06,18,20,42,59 | Complete hybrid-state reuse and earlier query modules were tested; the remaining target joins useful semantics and isolated efficient execution. |
| Fixed heads, finite codes and dynamic candidates | OQ-07–11,13,21,38,40 | Learned NLI/candidate heads and finite readouts are established components; their fully matched trade-off remains open. |
| Choice/Noul/Score and deterministic serialization | OQ-19,22,23,34,35,52 | Host types, bounded primitive probes and parts of the native wire path exist; general semantics and precise current conformance are not implied. |
| Supervised proper losses, calibration and CADO | OQ-28–39 | Adaptation and multiple calibration/replay variants have run; no general retention recipe or necessary incremental RL benefit is established. |
| Diverse data, explicit unknown and high K | OQ-17–29,37 | Earlier label transfer, OOS/omission tests and mechanics grids are not an independently reviewed generalist evaluation. |
| Prefix branching and efficient serving | OQ-03,05,06,12,40–49 | Existing reuse and CPU/MLX evidence is retained; new profiles, catalogue caching and full-service economics need their own tests. |
| Resource reduction and alternative backbones | OQ-09–16,31,43,56–59 | Small Qwen, ModernBERT and MoE already have scoped comparisons. Better-trained encoders, matched 4B/9B and explicit streaming remain different questions. |
| Benchmark validity, authority and release | OQ-24–26,37,50–55,60 | A leaderboard, execution-complete marker or valid schema cannot close these independent contracts. |

### The Unified Q01–Q20 ledger, interpreted as remaining work

| Earlier ledger items | Status after the completed run | Remaining register entries |
|---|---|---|
| Q01–Q03: dependencies/readouts/joint probabilities | Several measured positive results, with missing combinations and collateral errors | OQ-01,02,07,08,18,35 |
| Q04–Q06: process history | Bounded isolation treatments pass; selected new readouts and real service load remain unqualified | OQ-05,06,20,45,49 |
| Q07–Q09: cache, scaling, transformations | Exact reference cache passes; simple scaling succeeds; semantic sensitivity remains | OQ-03,08,20,21,26,42,44,46 |
| Q10–Q15: compact models, calibration, NLI, replay and CPU execution | Actual Go results, weak transfer, mixed calibration, assembly parity; INT8 unavailable | OQ-09–13,16,28–39,43 |
| Q16: matched Qwen sizes | No matched current 4B/9B result; older 2B comparisons still count as prior evidence | OQ-14 |
| Q17: vocabulary versus learned Qwen head | Earlier head/readout results and R29 are partial, not the matched learning ablation | OQ-07 |
| Q18: original document retention | Still unresolved despite completed failed treatments | OQ-25–30 |
| Q19: Rust/MLX, MoE, diffusion, RLCD | New-profile/extension work remains; older Rust/MLX parity and CPU service are not erased | OQ-32,41,43,48,56–59 |
| Q20: independent release confirmation | Still open for the complete declared product | OQ-17,25,37,49–55,60 |

## 8. Work not justified as an automatic next step

Do not repeat the exact failed generic short-premise replay recipes and label them a new document-retention hypothesis. Do not assume that a four-order ensemble, blanket indexing, or a pooled frozen state vector will help after the recorded negative results. A new experiment must identify what differs and why that difference could address the failure. [W §§18,20; R29; U3 §§2,7]

Do not treat failed TurboQuant snapshot configurations as proof that all quantization is bad, or an assembly speedup as proof that INT8 is available. Conversely, do not treat nominal lower bit width, fewer active experts, fewer forwards or one batched framework call as measured end-to-end savings. [W §§9–10,17.4,19; U3 §§6,8]

Do not prioritize MTP/speculative decoding for the existing no-generated-output path. It becomes a meaningful question only for a genuinely generated reasoning or fallback sequence. Diffusion, multimodal expansion and large MoE streaming remain conditional workload decisions, not requirements for fixing the current text classifier. [WP §10.3; R: Jeeves and diffusion reviews]

Finally, recovering Jev’s proprietary topology is not the release criterion. A useful open alternative can be evaluated against its declared interface, quality, uncertainty and resource envelope without claiming identical hidden architecture or RLCD training. [R: reproduction boundaries; H]

## 9. Sources, provenance and interpretation notes

Bracketed source IDs refer to the materials below. Local evidence IDs such as E40 and RUSTM2 retain their original definitions. External descriptions are leads and primary-source facts, not newly verified model outcomes. Candidate checkpoint and runtime revisions must be locked before a future experiment; this document does not claim to have downloaded or run their weights.

**Explicit citation correction:** R’s additional-reference list associates GLiClass with arXiv 2501.12345. This document uses the verified paper title at **arXiv 2508.07662** and its author repository instead. The original research file is unchanged. [GL]

**Explicit conformance question:** the older whitepaper describes entropy-derived confidence in the native adapter, whereas the inspected official pinned adapter gives a different Choice statistic. OQ-52 treats this as an active-code and versioned-contract check, not a freshly demonstrated bug in today’s repository. [W §11.8; CONF]

**No new empirical claim:** no new model training, inference, timing, calibration fit, protected-final access, Drive mutation, or raw-result re-audit was performed for this question register. The supplied U3 review’s prior verification counts are not repeated as work performed here. R29’s source-reported local results remain distinguished from the separately reconciled U3 run.

### [W] OpenKind whitepaper, version 0.8.8

**Evidence type:** supplied consolidated local evidence. Sections and E/RUST evidence IDs cited in the question entries; evidence through E40. Supplied file: `WHITEPAPER.md`. Its input digest is in the companion JSON manifest.

### [WP] OpenKind working paper, revision 0.8

**Evidence type:** supplied evidence synthesis. Research-question synthesis; local cutoff E40. Supplied file: `WORKING_PAPER(6).md`. Its input digest is in the companion JSON manifest.

### [U3] OpenKind Unified Decisions Results Review, 29 September 2026

**Evidence type:** supplied latest completed-run review. Run 20260929T194506_694066Z_dc1f74; prior saved-output reconciliation, not rerun for this document. Supplied file: `OpenKind_Unified_Decisions_Results_20260929.md`. Its input digest is in the companion JSON manifest.

[Primary source](https://drive.google.com/drive/folders/1gLVnuSIHUDAqyQlyxFyKuc2LWcOvklgx)

### [R] Jev, Reverse-Engineered: supplied RESEARCH(1).md

**Evidence type:** supplied research synthesis, with source-reported local experiments. Includes Outcomes of the 2026-09-29 runs. R29 means this section; its linked local raw files were not fetched or re-audited here. Supplied file: `RESEARCH(1).md`. Its input digest is in the companion JSON manifest.

### [H] Archer Hume, Jev’s Architecture Unmasked

**Evidence type:** external author API probes and architectural hypotheses. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://archerhume.com/posts/jevs-architecture-unmasked)

### [S] Sebastian Raschka, Language Models for Text Classification: From Bag-of-Words to Jev

**Evidence type:** external author comparison and survey. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://magazine.sebastianraschka.com/p/classifier-history-and-jev)

### [BERT] Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Evidence type:** primary research. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://aclanthology.org/N19-1423/)

### [MB] Answer.AI ModernBERT-base model card

**Evidence type:** primary model documentation. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://huggingface.co/answerdotai/ModernBERT-base) · [Companion paper or implementation](https://arxiv.org/abs/2412.13663)

### [DNLI] DeBERTa-v3-base-mnli-fever-anli model card

**Evidence type:** primary released three-way NLI model documentation. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)

### [DZ] deberta-v3-base-zeroshot-v2.0-c model card

**Evidence type:** primary released binary zero-shot model documentation. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c)

### [GL] GLiClass: Generalist Lightweight Model for Sequence Classification Tasks

**Evidence type:** primary research and implementation. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://arxiv.org/abs/2508.07662) · [Companion paper or implementation](https://github.com/Knowledgator/GLiClass)

### [GLM] GLiClass small v1.0 model card

**Evidence type:** primary candidate checkpoint documentation. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://huggingface.co/knowledgator/gliclass-small-v1.0)

### [SF] SetFit: Efficient Few-Shot Learning Without Prompts

**Evidence type:** primary research and official documentation. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://arxiv.org/abs/2209.11055) · [Companion paper or implementation](https://huggingface.co/docs/setfit/index)

### [API] TypeSafe API documentation

**Evidence type:** official external interface specification. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://docs.typesafe.ai/api)

### [CONF] TypeSafe Python adapter confidence_metrics.py, pinned fb52b103

**Evidence type:** primary implementation at an explicit revision. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://github.com/typesafe-ai/system-one-adapter-python/blob/fb52b1030b7fc1f4f1cf39910afa5da54f9835e3/src/system_one_adapter/_utils/confidence_metrics.py)

### [ORT] ONNX Runtime quantization documentation

**Evidence type:** official execution documentation. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

### [CAL] Guo et al., On Calibration of Modern Neural Networks

**Evidence type:** primary research. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://proceedings.mlr.press/v70/guo17a.html)

### [RLCR] Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty

**Evidence type:** primary calibration-aware RL research, not TypeSafe RLCD. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://arxiv.org/abs/2507.16806)

### [T5] Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

**Evidence type:** primary encoder-decoder research. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://arxiv.org/abs/1910.10683)

### [HYD] Hydragen: High-Throughput LLM Inference with Shared Prefixes

**Evidence type:** primary shared-prefix systems research. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://arxiv.org/abs/2402.05099)

### [DEFT] DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference

**Evidence type:** primary tree-attention systems research. Public source accessed 30 September 2026; no local model result follows.

[Primary source](https://arxiv.org/abs/2404.00242)


## Closing decision

The next objective is not “try every architecture.” It is to establish the first **fully specified useful operating point**, then make controlled challenges against it. The highest-leverage work is the missing integration of selective readouts, declared dependencies, exact caching and behavioral isolation; the highest-value new model comparison is a properly scoped BERT-family/dynamic-label challenge. Evidence quality and policy reliability determine whether either result is worth deploying.
