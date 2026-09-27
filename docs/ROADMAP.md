# OpenKind roadmap

**Revision 0.12.0 — Qwen focus, 26 September 2026.**

Build a local, auditable decision engine around **dense Qwen3.5-4B, a direct
option-logit readout, calibrated probabilities, and shared-state Rust/MLX
execution**. Start with document-evidence judgments. Earn generalization and
supported-answer retention before optimizing the accepted operating point.

The next model comparison is frozen Qwen controls plus one published Qwen 4B
specialist. Test whether that specialist already solves the workload before
funding another custom fine-tune. Keep a larger Qwen teacher, a 2B student, and
a 9B capacity challenger conditional on a measured need. Broad family support
moves out of the active program.

**Order:** workload and evidence → Qwen 4B quality → complete request cost →
fresh confirmation. One frozen-profile MLX systems study can run alongside
data preparation. This is an experiment plan; external results motivate it
but do not become confirmed OpenKind findings.

[WORKING_PAPER.md](WORKING_PAPER.md) owns confirmed lessons; the
[whitepaper](whitepaper/WHITEPAPER.md) retains full research history.
[ARCHITECTURE.md](ARCHITECTURE.md), [MLX.md](MLX.md), and
[BENCHMARKS.md](BENCHMARKS.md) own implemented contracts, backend operations,
and measurement. Historical task IDs remain in the crosswalk below and
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md).

## Why this focus

The supplied report describes JevBench **v1.4.1, 23 September 2026**. The publisher
now identifies v1.4.2; ranks are moving. Preserve benchmark, model, adapter,
hardware, and date together. Its composite blends quality with speed and cost;
self-hosted cost estimates and timing adjustments are not local measurements. [X1]

| External evidence | Roadmap implication |
|---|---|
| The supplied v1.4.1 snapshot reports JevK5 v0.2 at 62.0 overall, 85.3% public accuracy and 33.1% sealed accuracy: a 52.2 percentage-point gap. The author confirms the v0.2 sealed result. [X2] | A Qwen 4B specialist is a credible candidate. Fresh source and task-family generalization are acceptance criteria; leaderboard rank is not the objective. |
| JevK5 uses a Qwen3.5-4B LoRA and option-letter logits. Its newer v0.3 has 17,408 teacher questions plus 30,052 public replay items, but author-reported accuracy on a separate 4,723-question mix is 66.3%, versus 66.5% for v0.2. [X2] | Test released weights first. More data or a newer version does not establish preservation or transfer. Keep the v0.2 benchmark row separate from v0.3 evaluation. |
| The author's correction identifies 940 MMLU-Pro test items in the v0.2/2B training recipe. v0.3 changes the data recipe. [X3] | Audit source splits, overlap and rights before importing a checkpoint or corpus. This disclosure is not evidence of JevBench item leakage. |
| JevK5 reports about 13 ms on an H100 short-input path; SemIf documents a Qwen3.5 MLX implementation with direct readout and prefix reuse. [X4, X5] | Use these as implementation references. Measure our full Mac request path, including realistic documents, multiple questions, synchronization and service overhead. |

The model choice is Qwen for deployed inference. JevK5 v0.3's teacher data
includes both Qwen and GPT outputs; preserve that provenance. Its card also
reports retained phrase overlap with ContractNLI/SGD development or test text.
Evaluate overlap against our actual partitions before making an independence
claim. [X2]

## Confirmed starting point

| Area | Recorded result | Next consequence |
|---|---|---|
| CPU reference | Bounded frozen-profile parity, hybrid-state branching, high-K completion, persistence and named-machine lifecycle/load/soak pass. | Maintain the oracle; completed bring-up is not the next milestone. |
| Apple Silicon | Pinned-base MLX FP32 full, nested and unequal-length vectorized parity pass. Forced execution and bounded memory recovery are recorded. BF16 fails the frozen equivalence gate; the packed kernel is slower and opt-in. | Measure useful vectorization. New checkpoint/readout profiles require their own reference and accelerated-service qualification. |
| Natural documents | E29 frozen comparison and E30–E32 adaptations are complete; no complete quality/retention pass. | Investigate supported-answer loss and test one attributable remedy if released specialists also fail. |
| Evidence | A3 has 51 reviewed rows plus six non-independent follow-ups; A4 aligns four papers and 40 caption candidates. The separate 27-case disposition leaves gold unchanged. | Complete required source/PDF, table/reference and independent correction review. Existing visibility work is bounded, not full repair. |
| Cheap readers | Pooled-root, learned StateQuery and factorized applicability studies miss complete quality/policy gates. | Keep causal execution available. A new encoder or bottleneck explanation is not established. |

The critical retention result remains **J3 ContractNLI 141/204 → 168/204,
entailed 81/84 → 70/84, QASPER 37/46 → 34/46**, using matched BF16 controls.
Parent-KL J4/J5 and source-label J6/J7 selectors retain **frozen0**. Source-label
replay improves SNLI from 125/192 to 159/192 and 131/192 to 166/192 at fixed80,
but fails document preservation. These are exposed non-final panels; QASPER has
only 46 questions from 12 papers. No proxy gain closes the task-quality gate.

Sources: [E29 frozen comparison](https://drive.google.com/file/d/14QNUaQbTQtlnZctqDFneDilsjYyrA5Zb/view),
[E30 decision LoRA](https://drive.google.com/file/d/1Ov6794Pey5bkXrW8W73acGy2EylFhEQQ/view),
[E31 parent-KL](https://drive.google.com/file/d/1YHm5PSVb6GTC3kNwLKboooJq-TQKlDMZ/view),
[E32 source-label replay](https://drive.google.com/file/d/1YX6xJU9X2mQS9c8KfQ45JnwaQD7xOzIq/view),
[CPU service evidence](verification/native-service-gate/2026-09-22-rerun2/README.md),
and [MLX follow-up](verification/phase3m-2026-09-22/README.md).

## Model and engine choices

| Role | Choice and boundary |
|---|---|
| Immutable integration reference | Existing `encoder-state-first` profile `a047d6802c3f06f085b8`: Qwen Base with candidate branches and the learned score-summary head. Preserve its fixtures; it is not the selected deployment winner. |
| Frozen controls | J0 `Qwen/Qwen3.5-4B-Base`, revision `1001bb4d826a52d1f399e183466143f4da7b741b`; J1 `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. E29 found no universal winner. |
| First external candidate | **JevK5 4B v0.3**, subject to artifact, provenance and input-contract checks. Resolve immutable weight/runtime revisions before evaluation; never load moving `main` as an experiment identity. v0.2 is a historical reproduction option. |
| Own adaptation | Start from J1 when a specific residual failure justifies training. Preserve J0 and the relevant completed adaptation controls. A base-model swap is a separate hypothesis. |
| Cost challenger | Qwen 2B only after the 4B quality gate, with task-verified teacher supervision if needed. A released 2B model does not inherit its 4B sibling's results. |
| Capacity / teacher | One 9B Qwen challenger if 4B quality is insufficient and memory permits; a larger Qwen teacher is an offline option after a task-quality check. No mandatory larger-model runtime dependency. |

The primary readout offers **all outcome descriptions in one question prompt**,
then reads the allowed single-token answer-code logits and normalizes them.
There is no generated explanation or JSON decoding in this path. It is one
forward path per question, with an opportunity to share the exact state prefix
across questions. It is not one forward pass for every question together;
option text and question suffixes still cost compute.

The initial proposed limit is **16 declared outcomes including semantic none**.
Verify every code is one token at the actual answer boundary, use the last real
position for padded inputs, and map opaque consumer keys outside the model.
Keep yes/no, semantic none, and application review distinct. Larger option sets
must be explicitly unsupported or dispatched to a separately qualified profile;
preserve existing high-K fixtures. A multi-pass tournament needs its own
quality and calibration evidence.

Move the joint-option research path into its own Rust/MLX candidate profile.
Checkpoint, tokenizer, renderer, code mapping, readout, dtype, calibration and
runtime revisions belong in its identity. Existing candidate-branch parity does
not qualify this new path. Establish its differential reference before
optimizing it.

## Active milestones

| Milestone | Next deliverable | Exit |
|---|---|---|
| **M0 — Workload** | Text document judgments, supported option/context ranges, target hardware, selection limits | Versioned scope and finite quality/resource bounds |
| **M1 — Evidence** | Repaired input ledger, partitioned data and independent review record | Defects resolved or quarantined without model-driven eligibility |
| **M2 — Qwen quality** | Frozen controls versus one released 4B specialist; one diagnosis-led adaptation only if necessary | Complete non-final quality, retention and policy pass |
| **M3 — Cost** | Joint-option Rust/MLX profile, exact-prefix reuse, matched request measurements | Accepted quality at lower complete cost, within resource limits |
| **M4 — Preview** | Fresh confirmation, accelerated service qualification and reproducible manifest | All locked release limits pass |

### M0: Fix the task and success measure

- [ ] Specify document sources, question/rubric families, option descriptions,
  state and total-input token budgets, unsupported inputs, and rejection or
  truncation rules. Four distinct questions per document is a proposed demo,
  not established semantic capability. Noul and ordinal Score need their own
  quality definitions before broad support claims.
- [ ] Specify missing evidence, contradiction, semantic none, out-of-domain
  input and review behavior. Preserve wire semantics and consumer authority.
- [ ] Name deployment hardware. The development M4 Max and exploratory L4 are
  not deployment requirements by implication.
- [ ] Optimize **correct accepted decisions per second**, subject to accepted
  error, minimum coverage, request p95 and memory bounds. Count failures and
  reviews; accepted error is undefined at zero acceptance.

Retain the unresolved fields `max_family_macro_nll`,
`max_panel_accepted_error`, `min_policy_coverage`, `max_request_p95_ms`, and
`max_peak_allocated_gib`. Fix population, aggregation, queueing, and memory
semantics before selection. This revision supplies no invented latency target,
risk tolerance or independent approval. Bounded preparation can continue while
selection limits and review responsibility are resolved.

### M1: Repair inputs and protect evaluation

- [ ] Extend A4 alignment to required source versions, table cells, captions
  and references. Trace evidence into finalized input tokens; distinguish
  source answerability from visible-state answerability. Negative or ambiguous
  labels need documented grounds, not fabricated positive spans.
- [x] Retain E29 visibility preparation: 741 questions, 72 source
  states/components, four FP32 comparison cells at 1,024/4,096 state-token caps.
  This is completed, previously exposed non-final evidence.
- [ ] Preserve original corpora, locks, partitions, tie rules and thresholds.
  Put corrections in a versioned overlay with serializer/input hashes. Keep
  77 original-annotation ties diagnostic until reviewed rules change them.
  Independently review proposed corrections; do not repeat completed A3 work.
- [ ] Assign disjoint **training, model-selection, calibration-fit,
  policy-selection and final-confirmation** roles. Group by source document
  and generated/paraphrased relatives; hold out domains and question/rubric
  families. Keep exposed gates and QASPER evaluation records out of fitting.
- [ ] Audit external checkpoint/data licenses and exact/near overlap with all
  evaluation partitions. Public JevBench is report-only. A zero exact-match
  scan alone does not establish independent generalization.

A prediction-blind, text-complete prototype scope may be registered separately.
Report excluded populations and quarantine unresolved cases. Re-evaluate all
comparators on the same repaired inputs. A narrower scope does not pass the
original two-source gate, and annotations stay out of inference inputs unless
legitimately available at deployment. Final remains closed.

### M2: Select a useful Qwen 4B decision model

**First compare, then train if needed.** Prepare one pinned specialist candidate
and the existing frozen controls. Run each native pipeline as a deployable
system comparison, including its renderer and calibration. For claims about
LoRA, data or readout effects, use separate matched ablations; cross-repository
score differences do not isolate a cause.

- [ ] Include stock J1 under the specialist's same renderer/readout, with
  identity-temperature and separately fitted calibration controls. Distinguish
  the effect of trained weights from prompt and probability processing.
- [ ] Evaluate document-level and source/family slices: full semantic accuracy,
  per-class recall, false-none, NLL/Brier, calibration and review risk/coverage.
  Add option-order/code-permutation and unsupported-input checks. QASPER's
  one-candidate task has no conditional-ranking metric.
- [ ] Retain the registered research screen: source-specific recall floors,
  false-none ≤0.20, proper-score non-regression, and policy cost below review-all
  at 0.10 with useful coverage. Copy exact populations and rules into the new
  contract; these do not substitute for production risk limits.
- [ ] If the specialist passes, qualify that operating point in M3. If it
  fails, diagnose supported→none versus supported→contradiction on admissible
  train/development records, by evidence location, document length, hypothesis
  family and source. Reviewed oracle windows are diagnostic contrasts, not
  validated deployable retrieval.
- [ ] Register **one preservation hypothesis**. The preferred candidate, if
  diagnosis supports it, is a document-grounded mixture balancing supported,
  contradicted and missing-evidence cases across realistic input lengths.
  Include policy exceptions, temporal/numeric reasoning, conflicting evidence,
  distractors and shared-document questions within M0 scope. Fix mixture,
  dose and selectors before fitting; generic short-premise replay already
  failed the needed preservation test.
- [ ] Retain frozen parents and relevant E30–E32 controls. Match inputs,
  supervision and execution precision; add a head-only control when attributing
  gains specifically to upstream adaptation. Protect supported-answer recall
  and proper scores, not only aggregate accuracy. A selector retaining frozen0
  rejects the treatment correctly.

If teacher data is required, first test the proposed Qwen teacher on the target
strata. Validate labels independently; agreement between its own answers is
not a correctness proof. Keep provenance, quarantine unresolved labels, and
train soft targets only where justified distributions exist. Do not turn a
teacher's verbal confidence into gold probabilities. Reuse the established
bounded LoRA machinery; change one hypothesis rather than launching another
size, seed and recipe sweep.

Fit calibration on its assigned partition with an identity-temperature control.
Evaluate proper scores and the frozen review policy on separate data. A
normalized distribution over supplied options is not automatically a reliable
probability of correctness or detection of out-of-domain input. Changing model,
precision, renderer, option scheme or task distribution reopens calibration
qualification. Preserve a frozen fallback throughout.

**Exit:** every declared non-final quality, retention and policy requirement
passes. No favorable aggregate, public benchmark or proxy score alone promotes
a model. Do not reselect historical checkpoints or expand seeds 42/123 on that
basis. E30–E32 remain completed failed-preservation studies.

### M3: Make the accepted path fast on Rust/MLX

**Available now:** measure the existing pinned FP32 `ReferenceOps` per-lane and
forced vectorized implementations on identical inputs. Compatible forced lanes
currently span 2–8; automatic scheduling remains per-lane until measurements
justify a useful range. This frozen systems study is separate from model selection.

**Candidate profile:** implement the selected joint-option readout, its
reference fixtures and direct projection onto allowed vocabulary rows. This
can avoid unnecessary output projection work; tied embedding weights remain
needed for input lookup, so it does not remove their resident memory.

- [ ] Cache only a longest exact prefix of finalized token IDs. Identity must
  include the model/rendering profile and complete hybrid state: attention KV,
  recurrent/convolution state and positions. Verify branch isolation, nesting,
  unequal lengths and eviction. Moving rubric/options to create more reuse
  changes the model input and requires quality evaluation.
- [ ] Start with a declared grid of state caps 128/1,024/4,096, Q=1/4/8 and
  K=2/4/8/16 **total outcomes**, plus real traces and unequal suffixes. Include
  full prompt lengths. These are proposed measurement points, not supported
  performance claims.
- [ ] Compare repeated-full, cold-shared and warm-shared requests. Measure
  p50/p95, `T(Q)/T(1)`, marginal question cost, correct accepted throughput,
  forward calls and cache bytes. Include rendering, tokenization, actual
  device synchronization, probability transfer and policy work. Report model
  load separately and include queueing in service latency.
- [ ] Report weights, branch/cache state, scratch/allocator peak and process
  memory separately, with hardware/OS/runtime/commit identity. Use SemIf's MLX
  work as a porting reference; pin and verify dependencies rather than silently
  upgrading to its environment. [X5]
- [ ] Optimize one measured bottleneck: compatible length buckets, suffix
  batching, selected-row projection or a justified kernel. Keep complete
  probability/argmax/policy checks. CUDA graph numbers are not MLX guarantees.

For same-model implementation equivalence, retain the frozen gate of maximum
probability difference ≤0.005 with zero decision/policy changes on its declared
fixtures. A new model or precision needs its own semantic gate, not an exemption
from quality evaluation. The existing BF16 failure and slower packed kernel
remain recorded.

After 4B quality passes, select **one** cost treatment from profiling: an 8-bit
weight candidate first, a 4-bit candidate if memory requires it, or a Qwen 2B
student. Do not run a blanket ladder. Each changed numerical model needs its
own calibration, quality, memory and latency record. Weight quantization is
separate from the failed low-bit KV snapshots. K=255 latency work stays
conditional on workload demand.

**Exit:** a useful operating point meets resource limits with measured complete
cost improvement. Sharing or batching that loses on short inputs remains
conditional; dispatch must follow measured ranges. Parity alone is insufficient.

### M4: Confirm and release a scoped preview

- [ ] Lock model, renderer, data/input contract, calibration, review policy,
  baseline settings and selection rules before fresh final outcomes. Previously
  inspected non-final panels and the exposed `2ij.2.0` final are not fresh.
- [ ] Confirm on protected source-state groups and unseen question/rubric
  families. Estimate uncertainty by source group; expanded questions and
  permutations are not independent samples.
- [ ] Qualify the actual accelerated service: overload, deadlines, cancellation,
  recovery, memory stability, queue-inclusive load/soak and clean-commit replay.
  CPU qualification does not automatically qualify MLX.
- [ ] Publish supported scope, exclusions, class failures, risk/coverage,
  hardware/resource figures, artifact hashes and reproducible release manifest.

**Exit:** all locked requirements pass without post-final tuning or reselection.
Failed confirmation requires fresh data for any new selection. Wire compatibility
alone does not establish a general Jev replacement.

## Deferred work and stopping rules

Keep one modeling hypothesis and one frozen systems experiment active at a time.
The Qwen direct-logit profile is now the family-integration priority. The surveys
in [families/README.md](families/README.md) and integration checklist in
[families/NEW_FAMILY.md](families/NEW_FAMILY.md) remain reference material; the
previous broad F1–F5 rollout is deferred, with no automatic post-M4 commitment.

Defer non-Qwen encoders, diffusion, multimodal expansion, multi-engine routing,
private RLCD reconstruction, and MTP for the no-decoding path. Reopen shallow-head,
pooled-root or replay variants only with a new diagnosis. Streaming-MoE remains
untested: consider it only for a demonstrated capacity need that resident dense
Qwen cannot meet, measuring expert I/O, cold/warm latency, memory and quality.
Weight streaming and KV compression address different costs.

## Historical task crosswalk

| Previous IDs | Current disposition |
|---|---|
| 0–1, 2A–2H, 2H-C1–C5 | Completed contracts/research; retain results and failures. |
| 2I.1–2I.2, 4E-ADJ, A3/A4 | M0/M1 retain scope, evidence repair and independent-review work. |
| 2I.3–2I.6, 3A/3B, 3.1–3.10 | Preserve rendering, branching, native parity, high-K and persistence evidence. M2 tests usefulness; M3 measures relevant cost. |
| 2J.1–2J.4, 4A–4D, 4E-B.0/B.1/B.2 | Exploratory comparisons complete; no Phase 4 model accepted. |
| E29–E32 | Frozen comparison, LoRA, KL and label replay complete. Preserve FP32/BF16 identities, selections, failed retention and fixed80 diagnostics. |
| 2J.5–2J.6 | M2 comparator selection; M4 fresh release confirmation. |
| S.1–S.5, 3.11 | CPU contract/service evidence complete within scope; M4 owns accelerated-service qualification. |
| 3M.0–3M.8 | Retain runtime/FP32 parity and bounded daemon/memory evidence. M3 measures cost; M4 owns promotion. |
| P2.1–P2.3, 4A.3 | Conditional cost/distillation work; teacher quality remains a prerequisite. |
| F0–F5 | Protect F0 reference; prioritize the Qwen joint-logit profile from F2. Other family expansion is deferred. |

## External source register

Primary sources checked **26 September 2026**. External measurements are
publisher/author reports, not OpenKind replications. Preserve cited release
versions in future experiment manifests; live pages can change.

- **X1:** [Benchmark Heaven alternatives and methodology](https://benchmarkheaven.com/jev-models/alternatives).
  The user's v1.4.1 report is a dated snapshot, not the current rank list.
- **X2:** [JevK5 4B model card](https://huggingface.co/alibiserikbay/JevK5).
  Distinguishes v0.2 benchmark evidence, v0.3 author evaluations and data provenance.
- **X3:** [JevK5 changelog](https://github.com/allebee/jevk5/blob/main/CHANGELOG.md).
  Release changes and the 24 September training-split correction.
- **X4:** [JevK5 implementation](https://github.com/allebee/jevk5) and
  [prompt/readout contract](https://github.com/allebee/jevk5/blob/main/jevk5/prompt.py).
- **X5:** [SemIf MLX implementation notes](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/docs/MLX.md).

For exact historical checkboxes and run identities, use
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md). This revision changes priorities and
planned comparisons. It runs no inference or training, opens no final outcomes,
and promotes no model.
