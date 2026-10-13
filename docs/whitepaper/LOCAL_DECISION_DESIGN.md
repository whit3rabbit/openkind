# A local decision model and engine for 16–32 GB Macs

**Design revision 1.0 · 9 October 2026 · evidence cutoff: completed local-decision V13A.**

Target: text decisions through OpenKind’s **Choice, Noul and finite ordinal Score** contract on an everyday Apple Silicon Mac with 16–32 GB RAM. Workload prevalence, application error costs and required latency remain unspecified. Mac budgets in this document are proposed targets, not measurements of the new trained models.

**Recommendation.** Keep a **single fully conditioned Qwen forward, ordinary LoRA and a compact frozen answer-code readout**. Develop **Qwen3.5-2B as the resource-efficient research model** and retain **Qwen3.5-4B as the higher-quality research reference**. Do not add a pointer, recurrent reader or generated rationale to the mainline on the present evidence. Preserve deterministic application logic and exact-computation reuse as separately qualified engine components. No adapted model discussed here has passed the complete current quality and presentation requirements. ([LD-V12B][]; [LD-V13A][])

**What is locked is the experimental architecture and provenance, not a production model or universal hyperparameter optimum.** V13A’s checked-teacher term is **not** a validated default. V13B’s consistency term is **implemented but unresolved**: its results are not reviewed. Both V13 experiments retain the same immutable V12B student entry; V13B must not silently continue V13A. [LD-V13-PLAN]

This design supersedes the October 3 proposal’s active priorities, including “first compare finite codes/pointers” and the recurrent-reader priority. The T4 local-decision runs described here use a 1,024-token cap and 150-update stages. The 2,048-token/400-update default belongs to the separate notebook-35 (A100/L4) run identity ([run guide](../../research/local_decision_training/README.md#t4-fp16-pilot-differences)). It preserves the original historical engine findings and unqualified Mac targets in their own scope. **Local-decision notebook labels use `LD-V…` here. They are not the older WHITEPAPER aliases `V9–V12` attached to E40–E44.** See the companion [working paper](WORKING_PAPER.md) for the evidence synthesis and source boundaries.

## 1. Decisions retained, conditional and unresolved

| Disposition | Decision | Evidence and reason | Limit |
|---|---|---|---|
| **Retain as the research default** | One joint-option Qwen forward + LoRA + compact fixed codes | LD-V12B fixed code clearly beats its replacement-pointer recipe at 4B (pointer accuracy interval [−6.69, −1.85] points) and leads at 2B on point estimates only (69.89% versus 68.21%, interval includes zero). Both LoRA and pointer actually trained in the alternatives. | One seed and bounded recipes; not a theorem against pointers. |
| **Retain as the small-model path** | Adapted 2B, with 4B retained as a quality reference | LD-V12B lifts 2B macro accuracy from 47.28% to 69.89%; trained fixed-code 2B has 168.07 ms versus 291.68 ms warm median for trained 4B on the same timing panel. LD-V13A control reaches 73.15%. | The 2B and 4B weights/pretraining differ; all candidates still fail requirements. No Mac extrapolation. |
| **Retain in matched training** | SST-5 equal-class replay | LD-V5 improves very-positive recall from 7/32 in control to 23/32 at matched exposure. | Class balancing did not itself make every class or presentation stable. |
| **Retain as a scoped preservation control** | Original-parent KL coefficient 0.25 on eligible source-correct training rows | LD-V7 light preservation restores very-positive recall to 25/32 and passes all five SST-5 class-recall checks. | It did not protect excluded Banking77/Plumb none slices or produce a qualified model; coefficient is not globally optimal. |
| **Retain as an implementation simplification** | Compact answer vectors; optionally remove the general LM-head wrapper | LD-V12A records exact compact/stripped output parity in its checks. | Complete-request improvement was only about 0.11–0.23%; no meaningful live weight-memory saving. |
| **Retain the measured historical engine components** | Declared dependencies, selective field readouts, exact reuse and resident process isolation | E39–E44 contain bounded positive quality, latency and parity results. | Those older profiles are not an already integrated engine for the new adapters. |
| **Do not add to the default** | Checked trained-4B guidance at weight 0.25, temperature 2 | LD-V13A has mixed effects: 672/928 correct in both arms; teacher fixes 14 control errors and introduces 14. | Local MNLI/MultiRC benefits and lower cross-fitted NLL coexist with new SST-5 positive-class damage. |
| **Do not add to the default** | LD-V11’s 18% verified curriculum replacement | Transfer improves 82/144 → 94/144 versus its control, but composition changes 80/144 → 76/144 and retention fails. | Preserve useful labels and the result; the allocation is not an established general improvement. |
| **Stop the current grids** | More recurrent passes, the tested token-memory head and the tested pointer/size grid | LD-V9 attributes 98.3% of the four-pass NLL gain to one pass; LD-V10 finds no useful extra memory-attention benefit; LD-V12B favors fixed codes. | Revisit only with a materially new, stated hypothesis and matched control. |
| **Await result review** | Explicit aligned presentation consistency | LD-V13B compares identical augmented views with JS weight 0 versus 0.5. | No outcome is incorporated here; 0.5 is a hypothesis, not a locked benefit. |

“Retain” identifies a useful component or control. It never means that the full model, calibration, service, router or target platform is qualified.

## 2. The current request path, and the separate engine target

### 2.1 Executed model path

```text
Validate state, question, described options and typed outcome space
    → render the complete state + this question + all its options
    → verify tokenizer/code mapping and admit the whole prompt (≤1,024 tokens)
    → Qwen text decoder with the selected ordinary LoRA
    → final hidden state
    → compact frozen vectors for the supplied A–P answer codes
    → softmax over supplied outcomes only
    → typed decision and probability vector
    → separately declared calibration / deterministic application logic
    → consumer action or review policy
```

The current research implementation runs isolated questions serially. **No output text, rationale or autoregressive answer sequence is generated.** It does not compute a full vocabulary and discard most scores afterward. Copying the frozen code rows once gives the same intended score function:

\[
z_i = w_{\mathrm{code}(i)}^{\mathsf T} h_{\mathrm{final}},\qquad
p_i = \operatorname{softmax}(z)_i.
\]

The pretrained torso and code matrix are frozen during these LoRA fits; the adapter changes the contextual representations. The matrix is **not a randomly initialized classifier** and **not a Strands pointer**. V12A’s physical wrapper removal does not imply that tied input embeddings can be removed. ([LD-V12A][]; [LD-V12B][])

### 2.2 Engine integration target, not the current notebook execution path

```text
Validated exact state + model/adapter/renderer/runtime identity
    → complete immutable hybrid root, only for a qualified reuse plan
        ├─ isolated question 1 + all of its options → distribution 1
        ├─ isolated question 2 + all of its options → distribution 2
        └─ isolated question Q + all of its options → distribution Q
    → declared dependency/computation plan
    → request-bound validation and typed serialization
```

Question-conditioned interpretation remains in the decoder. Shared prefill is not a claim that a pooled state vector suffices. A reusable hybrid root must include attention KV, DeltaNet recurrence, convolution state and positions. Question branches must not read another question’s continuation. Keep question catalogues out of a supposedly question-independent root; intentional catalogue-prefix caching is a separate schema-conditioned profile. ([W][] §§8–10, 16–17; historical design)

Adapters that change the torso change its state. Select the adapter before prefill and bind its identity into cache keys. Do not share roots across different adapters, model sizes, tokenizers, renderers, precisions or approximate semantic matches without a separately defined and tested method. The existing Base/candidate-scoring CPU/MLX integration remains a differential reference, not the same function as the new post-trained joint-option model.

## 3. Frozen entries and checkpoint lineage

**Active immutable entries for V13A and V13B:**

| Role | Base and readout | Exact entry |
|---|---|---|
| Student | `Qwen/Qwen3.5-2B`, post-trained; compact fixed code | LD-V12B `2b/fixed_code`, joint update 150 |
| Teacher in V13A only | `Qwen/Qwen3.5-4B`, post-trained; compact fixed code | LD-V12B `4b/fixed_code`, joint update 150 |
| Historical 4B quality reference | Same pinned 4B base; light-preservation LoRA | LD-V7 `parent_kl_025`, update 150; do not confuse with the fresh V12B fit |
| Legacy integration reference | Qwen3.5-4B-Base candidate-scoring profile | Preserve its original CPU/MLX identity and evidence separately |

Exact entry paths (under the private Drive output root `OpenKind_Local_Decision_Training_T4/`), base/tokenizer revisions and adapter/readout file SHA-256 values are recorded in [whitepaper §25.3](WHITEPAPER.md#253-frozen-entries-for-ld-v13a-and-ld-v13b) and machine-readably in [`FROZEN_RESEARCH_PROFILE.json`](FROZEN_RESEARCH_PROFILE.json). They are file hashes recorded by the executed [V13A study plan][LD-V13A-PLAN] and [frozen-entry guide][LD-V13-PLAN], not interchangeable tensor hashes or a reload of the weights. Full artifact inventories, contracts, tokenizer files and reference predictions remain authoritative.

**Frozen entry means immutable starting bytes.** Each arm loads that entry and trains a new 2B LoRA continuation with a fresh optimizer stage. An unchanged interrupted arm can resume its own committed state. A completed run is not extended by silently changing `max_steps`. The V13A replay and teacher outputs are additional research records, not automatic replacements for the V13B entry. Checkpoints do not become qualified because they have a newer date or a higher aggregate score.

## 4. The fixed training profile

| Item | Current declared value | Why it is kept / qualification |
|---|---|---|
| Backbone family | Pinned post-trained Qwen3.5 text decoder, 2B or 4B by named role | Local joint-option results; Base and post-trained identities are not interchangeable. |
| Readout | Compact frozen A–P matrix; only supplied outcomes scored | Best tested joint-readout recipe; no generated answer sequence. Verify actual IDs with each tokenizer. |
| LoRA | Rank 16, alpha 32, dropout 0 | Working matched configuration, not a proven capacity optimum. |
| Adapted modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` | Actual Qwen profile; require explicit compatibility for a different family. |
| Numerical path | NF4/FP16 torso; FP32 LoRA; math SDPA and nonreentrant checkpoint recomputation | Executed T4 reference; not an MLX/GGUF equivalence claim. |
| Optimizer stage | Learning rate `2e-5`; seven warmup updates then cosine; clipping 1.0; weight decay 0 | Current pilot profile. LD-V6 did not favor `1e-5` at its matched budget. |
| Per-arm budget | 150 additional updates × accumulation 16 | 2,400 presentations; one fresh declared stage, not a universal stopping rule. |
| Data footprint | 1,200 unique examples, each twice; 8,000 admitted rows are not all presented | Actual exposure, not a full-dataset epoch. Reusing the stage adds exposure, not unique training groups. |
| Source presentations | Banking77/BoolQ/MNLI/MultiRC/rules/SST-5: 300 each; Plumb: 600 | Frozen balanced schedule; all five SST-5 classes receive 60 presentations. |
| Base supervised loss | Original CE, no smoothing and no added Brier loss; Plumb/teacher-source CE weight 0.5 | Retained control. Preserve original soft targets rather than silently converting them to hard labels. |
| Original-parent preservation | KL coefficient 0.25, source-correct trusted eligibility | Scoped retention result in LD-V7. Not the extra trained-teacher term tested in V13A. |
| Additional trained-teacher loss | **Not a default** | V13A tested 0.25 × 2² × masked KL at temperature 2; incremental effect was mixed. |
| Consistency loss | **Not yet a default** | V13B tests JS coefficient 0.5 versus zero on identical transformed inputs; results not reviewed. |
| Input admission | 1,024 **total prompt** tokens; at most 16 supplied outcomes, counting `__none__` on research Choice rows (Score panels such as SST-5 have no none option) | Evidence is rejected, not truncated. The 1,024 cap applies to these T4 runs; notebook 35’s 2,048-token default is a separate run identity. Native serving requires a non-empty `__none__` on every Choice. Multi-question limits remain contract/profile-specific. |
| Checkpoint policy | Evaluate/save every 25 updates; final update 150 eligible in current studies | Earlier checkpoints are diagnostic, not a post-hoc route around final failure. |

Source: ([LD-V12B-PLAN][]; [LD-V13A-PLAN][]). These settings freeze a reproducible comparison. They do not establish that a different Qwen, LoRA rank, language, context length or use case should inherit the same optimum.

The executed V13A stack records Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.17.0, PEFT 0.21.1, bitsandbytes 0.50.2, Accelerate 1.15.0, tokenizers 0.23.2 and safetensors 0.8.0 on one Tesla T4. The earlier incompatible optional TorchAO install was a setup failure, subsequently repaired; it is not a model-quality result. Preserve the environment identity and native preflight rather than treating these version strings as a permanently current compatibility guarantee. [LD-V13A-ENV]

## 5. What the training experiments actually established

### 5.1 Initial adaptation, balancing and parent preservation

LD-V4’s learning-rate arms improved development aggregates but fell back to step zero under retention. “Retain parent” meant no accepted adaptation, not that the parent passed an absolute none-error gate. Later panels, calibration and checkpoint-selection profiles changed; historical results must not be retroactively relabeled.

LD-V5 isolated balancing and paired presentation. Equal SST-5 sampling improved very-positive recall **7/32 → 23/32** compared with natural sampling at the matched final budget. LD-V6’s smaller learning rate did not improve the overall 150-update result. Its extra gold-removal pairs recovered two Banking77 true-none decisions but added five false-none decisions; Plumb true-none remained **1/10**. Merely making none more likely did not demonstrate better applicability reasoning. ([LD-V5]; [LD-V6])

LD-V7’s light original-parent KL restored very-positive recall **25/32**, but the remaining Banking77/Plumb none slices had **zero direct preservation coverage**. The 60-case V8 source review completed with **50 confirmed-none, ten uncertain and zero pending**. It was a user-authorized AI source review, not human or blinded adjudication. Original v8 coverage remained **27 Banking77 groups versus eight required, and three Plumb groups versus four**. That was a completed audit and a pretraining stop, not another failed fine-tune. Do not restart that quota search or alter uncertain judgments to pass it. ([LD-V7]; [LD-V8][])

### 5.2 Readout and model size are no longer an unrun first experiment

LD-V12B’s four final arms are tabulated in [whitepaper §25.1](WHITEPAPER.md#251-ld-v12b-trained-readout-and-model-size): fixed code reaches **69.89%** (2B) and **76.41%** (4B) source-macro accuracy, versus 68.21% and 72.25% for the replacement pointer.

All four failed quality and presentation requirements. Metrics use **928 primary rows** from a 960-row development payload; timing uses **28 distinct requests × five repeats** per arm on one T4. Pointer initialization adds **800 cached-feature supervised presentations** before the matched 2,400 joint presentations. This is a declared recipe comparison with unequal total initialization cost, not universal proof of code-head superiority. [LD-V12B]

The trained fixed-code 2B model improves from **47.28% to 69.89%** macro accuracy. Against trained fixed-code 4B in this study, the 4B/2B warm median-time ratio is about **1.74×** (291.68/168.07 ms) and 2B allocates about **44.5% less CUDA memory** (1.688 versus 3.044 GiB). It remains weaker overall. The 4B teacher is also fallible: on twelve policy-precedence probes 2B gets seven right and 4B gets two. Do not blindly distill every larger-model output.

### 5.3 V13A: continuation helped; checked guidance is not a locked benefit

The entry, replay and checked-teacher results are tabulated in [whitepaper §25.2](WHITEPAPER.md#252-ld-v13a-continuation-versus-checked-teacher-guidance): replay reaches **73.15%** and the teacher arm 73.28% macro accuracy, both **672/928** correct versus 641/928 at entry.

The teacher fixes 14 control errors and introduces 14. The **+0.134 macro percentage-point** difference comes from equal-source weighting with unequal source row counts, not more correct decisions. No new training seed or fresh acceptance population was evaluated. Both continuations remain rejected. [LD-V13A]

The teacher improves MNLI correct count **94/128 → 99/128** and MultiRC false recall **59/69 → 61/69**, but both arms lose entailment retention relative to stage entry (**40/44 → 37/44**). Teacher guidance drops SST-5 positive recall **15/32 → 9/32** versus control. Both arms eliminate rules false-none errors (**14/58 at entry → 0/58**) while preserving **45/45 true-none**; that decision gain does not require the added teacher term. Plumb true-none remains **1/10** for entry and both continuations.

Teacher coverage is **775/1,200 eligible unique training examples**, including **113 teacher-correct/student-wrong opportunities**. Only four eligible Plumb-none groups exist, providing eight guided presentations; the missing Plumb-none operation does not demonstrably generalize. `intervention_supported=true` means nonzero training opportunity, not learned capability. Passing against an originally weak parent also does not waive retention against the entering adapter or absolute quality requirements. [LD-V13A-AUDIT]

### 5.4 V13B remains independent and unresolved

The implemented comparison is `augmentation_control` versus `aligned_consistency`. Both use the same frozen V12B 2B entry and the same pairs: **600 unique examples with option-order changes and 600 with answer-code reassignment**, two presentations per example. Both views receive fresh decoder forwards and their own original-parent logits. The only extra term is **0.5 × JS** after canonical-key alignment. Its results are not reviewed. [LD-V13-PLAN]

Do not compare A’s replay arm with B’s consistency arm as if only one loss changed. Do not combine A and B automatically: A is mixed, and the interaction has not been tested. Different facts, changed instructions and removed-correct-option examples must not be forced to have equal predictions.

## 6. Semantic targets, evidence access and calibration

**Choice** returns one distribution over the offered described alternatives and explicit semantic none. Native serving rejects a Choice without a non-empty `__none__` option and reports its semantic-none mass ([root invariant 7](../../AGENTS.md#cross-workspace-invariants)). Score panels such as SST-5 and binary Noul rows carry no none option. None is not a spare code, low-confidence flag or permission to request review. **Noul** is binary and has no separate confidence field. **Score** is a finite ordered-level distribution with an expected level; continuous-score wire integration is a separate contract. Preserve the distinction between a missing valid option, insufficient evidence, an unfamiliar task and an application’s review decision.

**Task conversion matters.** Banking77 here uses sampled dynamic alternatives plus semantic none, not ordinary 77-way benchmark accuracy. MNLI’s neutral class is rendered as `__none__`, but its entailment relation is not the same operation as an absent valid supplied option in Banking77 or Plumb. SST-5 uses the balanced 32-per-class development panel; the Plumb source includes synthetic teacher labels and must not inherit a blanket human-audit claim. These rates are not JevBench leaderboard scores.

An omitted annotated best answer does not prove every retained alternative is invalid. Review must establish semantic validity and evidence sufficiency. Parent agreement is an eligibility test, not an independent audit. Known errors require a new, common supervision manifest for both arms; uncertain records must never be silently promoted to verified truth. Keeping original CE on uncertain records is a declared limitation, not a statement that their labels were certified.

The current 1,024-token limit is a total serialized-prompt cap. Longer documents, retrieval and a larger context window need their own admitted-evidence and quality measurements. Retrieval misses do not prove full-document absence. Changed tokenizer or template means new admission checks; historical token IDs and retained evidence do not transfer by assumption. ([LD-V13-PLAN][]; [W][] §§18.20–18.21)

**Calibration is separate from answer retention.** Since LD-V7, the prospective development profile uses five-fold, group-separated temperature fitting by request kind, applied to both baseline and candidate. It keeps raw metrics and class/none checks visible. Cross-fitting excludes the scored groups from temperature fitting, but does not undo repeated use of development data for model selection. These maps are not deployment calibrators. Positive temperature preserves argmax; it can still change a Noul probability, expected Score or thresholded downstream action.

Choice/Score entropy confidence remains a **concentration measure**, not the probability that the answer or complete request is correct. Evaluate maximum probability, entropy, margin, calibration and risk/coverage separately. Adjacent-level and opposite-extreme Score uncertainty can have equal entropy; ordinal spread is a useful **proposed additional report**, not an adopted confidence replacement.

## 7. Acceptance, diagnosis and frozen-rule discipline

The current pilot checks must remain distinct from product requirements:

| Check | Current study value | Meaning |
|---|---:|---|
| Source accuracy drop | At most 0.03 | Relative development screen, not deployment noninferiority certification |
| Supported-class recall drop | At most 0.05 | Applied with at least eight independent supporting groups |
| False-none rate | At most 0.20 | Absolute pilot ceiling; may be too permissive for an application |
| Cross-fitted NLL/Brier increase | At most 0.01 | Matched development comparison, alongside unchanged raw reports |
| Schema flip increase / mean-TV increase | At most 0.05 / 0.02 | Relative to the specified baseline(s) |
| Absolute schema flip target | At most 0.05 | Separate from a weak parent’s relative screen |
| Same-function probability parity | `atol=0.001` **and identical argmax** in the recent profile | Engineering equivalence check, not accuracy validation |

Source: [LD-V13A-PLAN]. The `0.001` probability tolerance is these studies’ stricter research replay tolerance; the repository acceptance tolerance is `0.005` ([documentation standards](../AGENTS.md#evidence--parity-standards)). LD-V10’s maximum drift of 0.001456 (§8.2) fails the first and passes the second. Earlier proposals of one-point noninferiority, 20% cascade latency savings, or a four-group Plumb minimum are not these studies’ unchanged acceptance rules. Application costs must eventually set their own prospective requirements; do not relax an already observed run to get a winner.

Record both pretrained-parent and stage-entry comparisons for continuations, source/class results, true-none denominators, false-none denominators, raw/cross-fitted scores, error tails, majority/constant-output baselines, complete groups and actual re-rendered schema decisions. Repeated technical requests are not new independent groups. The 928 primary historical rows, 32 extra historical diagnostics, 144-example transfer/composition panels with 24 groups each, and 96 V10 probes with 12 groups are different evidence roles.

Fix the final checkpoint rule before training. A quality nominee receives one schema qualification decision; diagnostic evaluation of all arms is not permission to choose a runner-up after failure. `retain_parent`, `research_artifact` and nonzero teacher coverage are not synonyms for accepted quality. No local-decision adapter through V13A has passed all current requirements. Reserved calibration, gate and test payloads remained closed in V12B/V13A; older v4 parent-gate use remains a separate historical record. ([LD-V12B][]; [LD-V13A][])

## 8. Runtime choices: keep the actual wins in their actual scope

### 8.1 Head removal and input processing

LD-V12A’s complete-request paired ratios for stripped versus already-restricted scoring were about **1.0023× (2B), 1.0018× (4B) and 1.0011× (4B+v7)**. No comparison met its 5% material-speed rule. The adapted-4B readout alone changed **0.104 ms → 0.0645 ms**, whereas the complete request was about **528 ms**. A full-vocabulary final-position projection cost about **7.0 ms**; the existing restricted path already avoided that cost. [LD-V12A]

Compact and stripped paths showed zero recorded probability delta and identical selected answers. One 2B **full-vocabulary** comparison exceeded `0.001` probability tolerance without changing argmax; it was not a headless-path failure. The tied input embeddings remain resident: adapted-4B live allocation differed by only **512 bytes** and unique registered storage by **128 bytes**. The much larger reduction in allocator-reserved memory was confounded by unequal cleanup/empty-cache treatment. A later memory experiment must use the same cleanup for both forms. Neither a direct headless disk loader nor Mac startup was measured.

Do not compare V12A’s 33-request timing grid directly with V12B/V13A’s 28-request panel as a chronological speedup. For example, V12B’s trained fixed-code 2B/4B medians are **168.07/291.68 ms**; V13A’s control/teacher medians are **163.48/167.15 ms** with the same 2B architecture and allocation. Small within-architecture differences are not evidence that distillation accelerates inference. ([LD-V12B][]; [LD-V13A][])

### 8.2 Historical native mechanisms still worth retaining

| Earlier evidence, carried from the whitepaper and working paper | Design consequence | Boundary |
|---|---|---|
| E42 XR: 95.40% field / 84.38% all-six at 242.98 ms; NJ: 94.01% / 79.69% at 228.31 ms | Keep selective facts and declared dependency composition; use an explicit joint source where necessary. | 9B Q4/T4 authored policy study, not the new 2B adapter or Mac. XR withholds an exact route vector. |
| E42 multi-catalogue snapshots: about 54% grouped-latency reduction with zero paired drift | Keep exact catalogue reuse with host-state accounting. | About 170.6 MiB host storage; catalogue-conditioned reuse, not arbitrary-question state-root qualification. |
| E41–E43 isolated resident native/JSON processes pass bounded history traces | Explicit process/profile ownership remains valuable. | Does not diagnose same-process state defects or qualify cancellation/concurrency. |
| E43 input-encoding reuse: 117.13 → 24.38 ms, bitwise probability parity | Reuse exact computation in sequence encoders when appropriate. | Faster execution did not solve policy transfer; R41 all-six remained 0%. |
| Native MLX flat-field traversal used fewer calls but was slower on tested shapes | Choose plans on measured complete-path work. | Historical M4 Max reference, not a result for these LoRAs. |

These are source-reported historical measurements, preserved without re-audit. ([WP][] §§6–9)

LD-V10’s local Strands shared-prefix test adds a useful but narrower result: **three of six multi-question fixtures** passed the declared probability/answer parity rule, with **2.01×–3.36×** timing ratios over full batching on those passing fixtures. Single-question fixtures did not use the split path. All recorded argmax checks passed, but maximum probability difference reached about **0.001456**, above `0.001`; some serial-versus-batch comparisons also exceeded tolerance. Treat numerical-path variation as a hypothesis, not a diagnosed cache bug or a universally qualified cache. No direct transfer to OpenKind’s 2B/4B LoRAs is established. [LD-V10]

## 9. Deterministic logic, specialists and deferred architectures

For tasks with explicit calculations or dependencies, the model may extract facts and identify the applicable rule while Rust executes the declared arithmetic or logical mapping. Keep source positions and missing/uncertain extraction states. Deterministic arithmetic cannot repair a wrong premise.

For `a=g(z)`, aggregate probability mass over source outcomes that map to each action. If `g` depends on several uncertain facts, exact derived probabilities need their joint distribution or an explicitly justified factorization. Independent execution branches do not imply statistical independence. Do not silently multiply marginals, assign certainty to a derived answer or let a model score authorize an unsafe action.

A specialist cascade remains **conditional**, not the current request path. Historical DeBERTa evidence supports a scoped NLI reference; recovered encoder fine-tunes still exhibited majority collapse and weak whole-request policy correctness. Laya/CLEF/Strands remain implementation or system comparators, not drop-in sources of OpenKind probability semantics. A routed system must preserve original evidence for fallback, qualify its routing scope, account for both models’ memory and measure request-level fallback frequency, queueing and tail latency. ([WP][] §§3–4, 10)

The October 3 document proposed recurrent token-memory reading with ModernBERT-scale encoders. LD-V9/V10 now discount the tested frozen-Qwen-feature recurrent/memory recipes; they **do not test every small-encoder recurrent architecture, full-backbone latent loop or verified intermediate-supervision method**. Such directions are deferred until they pose a distinct question and beat a strong one-pass control. No continuous hidden-state transfer between live models or rationale decoder is part of the current implementation.

MoE and expert streaming remain secondary on the 16–32 GB target. Historical E36 compares a stronger, larger MoE profile at greater measured cost with confounded size/precision; it is not an expert-streaming result. Sparse activity does not imply small resident storage. Dense-to-MoE upcycling, cross-model latent bridges, diffusion readouts, multimodal expansion and RLCD/CADO are not measured improvements in this current local-decision mainline. Preserve their historical references in the working paper without converting them into active requirements.

## 10. Apple Silicon budgets and integration responsibilities

The original target remains 16–32 GB unified memory. **No new candidate’s T4 allocation is a prediction of Mac RSS, Metal/MLX allocation, system memory pressure or sustained swap behavior.** Existing native Base-profile Mac measurements remain valid only under their original identities.

| Deployment target | Current candidates to qualify | Retained proposed service peak budget | Status |
|---|---|---:|---|
| 16 GB Mac | 2B resource profile; 4B quality alternative | 7 GB including live model(s), state and scratch | Design target only; do not assume a resident specialist is beneficial |
| 32 GB Mac | Same 2B/4B pair for comparable workloads; separately test a larger profile if needed | 10 GB | Design target only; not a selected 9B replacement |

The earlier latency goal (resident model, cold dynamic state, 1,024 **state** tokens, four questions and eight short options, p50 <1 s and p95 <2 s) remains a **proposed engine workload**. It does **not** fit automatically under the current 1,024-**total-prompt** training cap. Either measure an admitted shorter state or preregister a larger-window profile; never truncate evidence to claim compliance. Specialist sub-100 ms targets, 4,096-token workloads and concurrency envelopes remain separate unmeasured targets.

| Existing repository seam | Responsibility / current boundary |
|---|---|
| [DecisionEngine](../../crates/openkind-engine/src/engine.rs) | Bounded request validation, execution and typed response. A new adapter/model contract needs explicit integration. |
| [ModelExecutionProfile](../../crates/openkind-engine/src/profile.rs) | Bind weights, readout, rendering, outcome space, quantization, calibrator and allowed plans. |
| [BranchableState](../../crates/openkind-runtime/src/branch/state.rs) | Complete immutable roots; independently mutable lanes; positions and tensor-byte accounting. |
| [MlxRuntime](../../crates/openkind-backends/src/qwen35/mlx/runtime.rs) | Preserve process-wide ownership; benchmark permitted batching inside it. |
| [Native finite-logit family](../../crates/openkind-backends/src/families/decoder_logit_qwen35/model.rs) | Mechanism for selected rows, not proof that the latest unmerged/custom research artifacts already load. |
| [Benchmark methodology](../BENCHMARKS.md) | Complete request cost, stage attribution, parity, quality and memory. |

These seams exist in the repository; none of them yet loads the new research adapters or readout contracts. Qualify quantized kernels without wholesale per-request FP32 expansion, full hybrid-state copies, lane/scratch bounds, exact tenant-scoped cache keys, request history, cancellations, unequal lengths, eviction and recovery. Do not change several execution mechanisms in an experiment that is meant to isolate model learning.

## 11. One reusable three-step workflow

| Step | Responsibility now | What must not happen |
|---|---|---|
| **1: Initial training / import** | Pin a new backbone/tokenizer/readout or import an existing ordinary LoRA with complete lineage. Establish a matched pretrained baseline and a frozen handoff. | Treat `latest` as a reproducible checkpoint; reuse another model’s code IDs, cached probabilities or hidden features. |
| **2: Targeted continuation** | Import the exact declared adapter; initialize a new optimizer stage; run one controlled intervention and its matched control. V13A/B are independent branches from V12B. | Continue a rejected newer adapter implicitly; mix data/head/loss changes; train on diagnostic failures; carry optimizer state across changed numerical identities. |
| **3: Package / qualify / publish** | Preflight lineage, runtime/renderer/readout compatibility, source terms, selected candidate, frozen calibration and independent acceptance. Distinguish private research from qualified release. | Let an old LoRA-only loader accept a new readout contract by inference; publish an unqualified model as accepted; disclose raw datasets or tokens. |

The original Step 1/2/3 notebooks are reusable entry points, not an assertion that every later research artifact is already supported by their old release code. V11 and V12B/V13 introduce new training/readout lineage contracts. Upgrade the publisher’s loader and leakage checks explicitly before qualification. Even fixed-code research bundles include a bound readout and tokenizer; do not load an adapter alone and guess the rest.

For another Qwen or another ordinary LoRA, re-establish module compatibility, embedding/output ties, maximum supplied outcomes, code-token validity, renderer and tokenizer hashes, exact admitted evidence, base/entry predictions, cache identities and fresh acceptance boundaries. Snapshot immutability does not imply generality of the current hyperparameters.

## 12. Outstanding questions and next actions

| Question | Evidence status through V13A | Required next evidence |
|---|---|---|
| Does deleting the already bypassed head materially accelerate requests? | LD-V12A: no material complete-request benefit; compact parity passed. | No repeat grid. Disk packaging/startup and Mac behavior are separate. |
| Which tested readout stays default? | LD-V12B favors compact fixed code in its declared recipes. | Stop the current pointer/size grid; maintain bounded alternatives as records. |
| Is adapted 2B useful to develop? | +22.61 macro points in LD-V12B; another 31 correct primary answers in each V13A continuation. | Absolute capability, class retention and stability must improve without losing the resource benefit. |
| Does checked trained-4B guidance add value? | Mixed completed result: no net extra primary correct answers, small cross-fit NLL gain, source trade-offs and new positive-sentiment failure. | Do not adopt it as default or sweep weights without a new mechanism hypothesis. |
| Does explicit consistency beat identical augmentation? | V13B implemented; **results not reviewed**. | Read B’s own fixed-final paired result, class/none checks and actual re-rendered stability. |
| Does the teacher supply general Plumb-none competence? | Four checked training opportunities; entry/control/teacher remain 1/10 on development true-none. | Broader verified operation coverage, not stronger blind imitation or a revived old quota. |
| Can A and B be combined? | Untested; A is not unambiguously beneficial. | A separately justified interaction experiment, not automatic chaining. |
| Will a result survive a new seed or workload? | Not established by these single-seed reused panels. | Lock a promising recipe; replicate and evaluate fresh disjoint families/groups before final acceptance. |
| Do gains survive MLX/Rust/merge/quantization/cache changes? | Unqualified for these new artifacts. | Actual target-platform request tests, lineage, probability/answer parity and memory. |
| Is RLCD the next default? | No tested rationale/reward-verifier loop or RL-vs-supervised increment. | A reliable reward definition and matched supervised control; no requirement to claim perfect SFT, but no unsupported algorithm switch. |
| Can the model be published as qualified? | No candidate through V13A meets all requirements. | Lineage-aware loader, locked candidate/calibrator, independent acceptance and post-conversion/reload tests. |

**Immediate action:** preserve V13B’s entry and configuration, review its completed result when available, and decide from the two separate effects. Do not resolve failures by weakening thresholds, relabeling AI review as human adjudication, fitting on failed development cases or selecting a runner-up after observing schema failure. A bounded negative result is not an impossibility theorem; a useful method can still fail the requirements.

## Appendix A. Dated external design precedents retained from the October 3 proposal

This appendix preserves the October 3 proposal’s external framing and implementation details. **It is not a fresh source review or an assertion that these are the latest releases.** The local outcomes in §§1–8 determine current priorities; external architectures and their claimed results do not substitute for OpenKind measurements.

| Dated source / idea | What the October 3 proposal recorded | Current use and limit |
|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) and the [earlier MLX campaign](../benchmarks/2026-09-28-laya-mlx-campaign/README.md) | M4 Max shape777 campaign: 24.34 English and 56.37 multilingual decisions/s, 2.12/2.94 GB peak RSS. Authors report 0.362 typed-decision accuracy for a base English checkpoint versus 0.766 for a specialist. | A scoped encoder/reference implementation. Throughput records are not base-Mac latency or quality; its shipped conditional probability space does not become learned semantic-none mass by relabeling. |
| [CLEF](https://blog.cloudflare.com/clef-decision-models/) / [historical research dossier](../RESEARCH.md#cloudflare-clef-and-linked-decision-models-reviewed-2026-10-01) | The proposal records a Qwen3.5-9B flash torso and a larger declared Qwen3.8-27B model, one no-cache prefill retaining token representations, option/evidence routing and cross-field mixing. The pinned flash head is width 1,024, with two routing layers, four decoder layers, 16 heads and FF width 4,096; its backbone width 4,096 differs from the selected 4B width 2,560. | Not dimensionally interchangeable head weights or a recovered training recipe. Cross-field mixing changes the information boundary; a head mask cannot undo a torso’s full-schema conditioning. Do not copy silent truncation, opaque-ID fallback or winning-probability confidence into OpenKind. The LD head experiments do not reproduce the entire released system. |
| [Strands article](https://strandsagents.com/blog/introducing-strands-decider/) | Article introduces v19 with a 2B-Base torso, rank-16 LoRA and learned option-endpoint/answer pointer. Later LD-V10 evaluates the separately identified v21 system; LD-V12B performs a local joint-LoRA replacement-pointer comparison. | Do not conflate article version, executed checkpoint, external system score and local controlled readout result. Native shared-prefix and MLX code are engineering references, not qualification of this new adapter. |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) / [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) | Teacher answers, rationales and distributions can supervise smaller models in the cited tasks. R1-derived students still generate reasoning. | Motivates verified offline supervision, not a claim of equivalent one-pass reasoning. LD-V13A now measures one checked-distribution intervention and finds a mixed increment. |
| [CALM](https://arxiv.org/abs/2401.02412) | Trained projections/cross-attention combine capabilities of live models. | Both models still execute; this is not a demonstrated Qwen-plus-Laya memory or latency saving, nor part of the current model. |
| [Coconut](https://arxiv.org/html/2412.06769v4) / [CODI](https://arxiv.org/html/2502.21074v3) / [recurrent depth](https://arxiv.org/abs/2502.05171) | The October 3 proposal records Coconut GSM8k 34.1% versus 42.9% textual CoT, CODI GPT-2-scale 3.1× compression with joint shared-weight teacher/student training and a weak separate-teacher ablation, and a 3.5B recurrent-depth proof of concept trained on 800B tokens. | Dated task-specific rationale/latent precedents, not proven hidden-state transplantation. LD-V9’s small frozen-feature recurrent-head result does not reproduce or refute these architectures. Sequential latent work still needs measured cost. |
| [Qwen MoE card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) / [Sparse Upcycling](https://arxiv.org/abs/2212.05055) | The source proposal distinguishes 35B stored from 3B active parameters; the ideal four-bit payload alone is about 17.5 GB decimal, before scales, higher-precision tensors, state and the operating system. Upcycling initializes experts but still requires specialization/training. | A capacity/teacher direction, not a selected 16 GB deployment or a measured streaming engine. E36’s stronger-but-costlier profile and broad request-wide expert use do not justify assuming inactive weights disappear. |

The legacy optional HelpSteer2/adequacy, smoothing/Brier and intermediate-relation/rationale objectives remain **separate unadopted ablations**. The current V13 objective retains unsmoothed CE and scoped parent preservation; neither external popularity nor a parameter suggestion establishes a useful local increment. A family-specific adapter, specialist router or deterministic decomposition still needs the complete routed/derived probability and request-quality checks described above.

## Sources and update provenance

This design builds on the October 3 `LOCAL_DECISION_DESIGN.md` proposal, working-paper revision 0.9.3, the reviewed local result records through V13A and the frozen V13 plan. Summary values and identities are reconciled against saved result files. The design adds **no new model inference, training, source-label adjudication, live benchmark refresh, V13B-result review or repository/runtime integration**. Earlier raw-replay validation counts belong to their original reviews. Input/output hashes for design revision 1.0 are recorded in `UPDATE_PROVENANCE.json` (external, not checked into the repository).

The working paper retains the older external literature and repository links as **dated source material**, not claims that a model card, leaderboard or software version was newly checked. The canonical [whitepaper][W] is version 0.8.13; its [§25](WHITEPAPER.md#25-local-decision-training-results-ld-v12b-and-ld-v13a) owns the LD-V12B and LD-V13A measured results.


### Current source lookup

| Record | Link / scope |
|---|---|
| Initial balancing and preservation | LD-V5, LD-V6, LD-V7 (external, cited by name only); older within-study results, not current qualification |
| AI source audit | [LD-V8][]; completed judgments and provenance, not a new human audit |
| Readout/recurrence and external comparator | [LD-V9][] and LD-V10 (external, cited by name only); do not conflate the frozen-feature heads with replacement joint training |
| Curriculum | [LD-V11][]; source allocation intervention and its actual rejections |
| Head-removal and model-size inference | [LD-V12A][]; equivalent scoring/parity and scoped timing/memory |
| Joint model-size/readout training | [LD-V12B][] and [plan][LD-V12B-PLAN] |
| Latest completed teacher experiment | [LD-V13A][], [training audit][LD-V13A-AUDIT], [study plan][LD-V13A-PLAN], [environment][LD-V13A-ENV] |
| V13 entry/independent-study definition | [LD-V13-PLAN][]; plan, not a V13B result |

The LD evidence records live in a private external store (the project Google Drive) and are not checked into the repository. Linked records are Drive citations that require access to that store; LD-V5, LD-V6, LD-V7 and LD-V10 are cited by name only. Their older next-step recommendations are superseded by this document, not reissued.

[W]: WHITEPAPER.md
[WP]: WORKING_PAPER.md
[LD-V8]: https://drive.google.com/file/d/1I_6nW2BGwFFymwiZC2Jt5ePK9HZ0HCGC/view
[LD-V9]: https://drive.google.com/file/d/160XJFZ6Zo5_HBG7zQ5C9p9HoGcR7L-qP/view
[LD-V11]: https://drive.google.com/file/d/1yO6bqcDo-EDnJFpcJHagorq4aeaOX8Ju/view
[LD-V12A]: https://drive.google.com/file/d/1JED9qnzcH_rgDIoKxEjayWoSlciQIMAT/view
[LD-V12B]: https://drive.google.com/file/d/17UDnx0xR6mO7Au09bsrBfeT30cUT0NLc/view
[LD-V12B-PLAN]: https://drive.google.com/file/d/1Y0n5YM27KcDBtRyJEGtKzXUr53Ubys51/view
[LD-V13A]: https://drive.google.com/file/d/1eUTSTZVisA4Ar-EMgMDQvYPHgiLjYk_V/view
[LD-V13A-AUDIT]: https://drive.google.com/file/d/1sIxApSB3egf6shA4uQxqzdt9FUfJ3ulf/view
[LD-V13A-PLAN]: https://drive.google.com/file/d/1uEH6PPaqV2ENRL-b0jKp3VJsV8q5ZqNH/view
[LD-V13A-ENV]: https://drive.google.com/file/d/1Wp4WwenIx_zLcqwn0EnV6cyC_NPBNj8V/view
[LD-V13-PLAN]: https://drive.google.com/file/d/14AZr6eyj4vE3Hom6gF0HWOXwAiWzLANi/view
