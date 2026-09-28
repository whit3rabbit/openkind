# OpenKind roadmap

**Revision 0.14.0 — Qwen quality, calibrated automation and qualified prefill reuse.**  
**26 September 2026, America/Chicago (27 September UTC).** The prefill study is complete: all three arms reviewed and all 26 checksum entries verified. Dense Qwen is the next research control; acceptance and cache qualification remain open.

Build a local, auditable decision engine around **a quality-qualified Qwen 4B
profile, direct option logits, independently checked calibration and acceptance,
and exact-prefix Rust/MLX execution**. Preserve Qwen3.5 document controls and
add the measured Qwen3-4B-Instruct-2507 control; no deployment winner is selected. Start with document-evidence judgments. Earn generalization and
supported-answer retention before optimizing the accepted operating point.

The current frozen-Qwen comparison is complete. Spend the next Colab budget on
dense numerical and policy controls; archive this slow, low-quality legacy MoE
profile as diagnostic evidence. Then compare the frozen document controls with one published
Qwen 4B specialist before funding another custom fine-tune. Keep a larger Qwen teacher, a 2B student, and
a 9B capacity challenger conditional on a measured need. Broad family support
moves out of the active program.

**Order:** workload and evidence → Qwen 4B quality → complete request cost →
fresh confirmation. One frozen-profile MLX systems study can run alongside
data preparation. This is an experiment plan; external results motivate it
but do not become confirmed OpenKind findings.

[WORKING_PAPER.md](whitepaper/WORKING_PAPER.md) owns confirmed lessons; the
[whitepaper](whitepaper/WHITEPAPER.md) retains full research history.
[ARCHITECTURE.md](ARCHITECTURE.md), [MLX.md](MLX.md), and
[BENCHMARKS.md](BENCHMARKS.md) own implemented contracts, backend operations,
and measurement. Historical task IDs remain in the crosswalk below and
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md).

## Why this focus

The supplied report describes JevBench **v1.4.1, 23 September 2026**. At the 26 September source review, the publisher
identified v1.4.2; the 27 September [v1.4.2.2 board](https://benchmarkheaven.com/jev-models)
now ranks imajev-4b first. Preserve benchmark, model, adapter,
hardware, and date together. Its composite blends quality with speed and cost;
self-hosted cost estimates and timing adjustments are not local measurements. [X1, X11]

| External evidence | Roadmap implication |
|---|---|
| The supplied v1.4.1 snapshot reports JevK5 v0.2 at 62.0 overall, 85.3% public accuracy and 33.1% sealed accuracy: a 52.2 percentage-point gap. The author confirms the v0.2 sealed result. [X2] | A Qwen 4B specialist is a credible candidate. Fresh source and task-family generalization are acceptance criteria; leaderboard rank is not the objective. |
| JevK5 uses a Qwen3.5-4B LoRA and option-letter logits. Its newer v0.3 has 17,408 teacher questions plus 30,052 public replay items, but author-reported accuracy on a separate 4,723-question mix is 66.3%, versus 66.5% for v0.2. [X2] | Test released weights first. More data or a newer version does not establish preservation or transfer. Keep the v0.2 benchmark row separate from v0.3 evaluation. |
| The author's correction identifies 940 MMLU-Pro test items in the v0.2/2B training recipe. v0.3 changes the data recipe. [X3] | Audit source splits, overlap and rights before importing a checkpoint or corpus. This disclosure is not evidence of JevBench item leakage. |
| JevK5 reports about 13 ms on an H100 short-input path; SemIf documents a Qwen3.5 MLX implementation with direct readout and prefix reuse. [X4, X5] | Use these as implementation references. Measure our full Mac request path, including realistic documents, multiple questions, synchronization and service overhead. |
| Imajev-4B phase 3 leads the official v1.4.2.2 composite at 67.37 with one option order and fitted calibration, but its Intelligence axis is below Jev's and both reach about 37% sealed accuracy. The selected checkpoint failed its own unknown-case and calibration gates under an explicit owner override. [X11–X13] | Screen this same-base J1 specialist before choosing the M2 external comparison. Preserve one-order/four-order, abstention, calibration, full-input, and source-transfer evidence as separate results. |

The model choice is Qwen for deployed inference. JevK5 v0.3's teacher data
includes both Qwen and GPT outputs; preserve that provenance. Its card also
reports retained phrase overlap with ContractNLI/SGD development or test text.
Evaluate overlap against our actual partitions before making an independence
claim. [X2]

### What PrivateMode changes

PrivateMode supplies an external constrained-logit baseline: a frozen model can
make useful typed decisions without a custom head or task fine-tuning. [X6, X7]
Keep this as **J_base** for every candidate: frozen weights, all options in one
question prompt, verified answer tokens, no generated reasoning, with raw and
separately calibrated distributions reported. E29 and the new dense arm already
provide instances; they do not pass the complete deployment contract.

The comparison supports the interface, not equivalence of models or backends.
Its reported p=0.64 does not prove statistical equivalence. [X8] Its repeatability
and option-granularity results motivate execution and K controls, not relaxed
parity or a monotonic option-count law. [X6, X10] Treat proposed label corrections
as independently reviewed evidence work; agreement among models is insufficient. [X9]

Answer prefilling chooses where to read logits. State-prefix caching reuses
computation. A pooled-state decision head changes the representation. Keep all
three claims separate. The article supplies no streamed-expert benchmark.

## Confirmed starting point

| Area | Recorded result | Next consequence |
|---|---|---|
| CPU reference | Bounded frozen-profile parity, hybrid-state branching, high-K completion, persistence and named-machine lifecycle/load/soak pass. | Maintain the oracle; completed bring-up is not the next milestone. |
| Apple Silicon | Pinned-base MLX FP32 full, nested and unequal-length vectorized parity pass. Forced execution and bounded memory recovery are recorded. Paired native compute diagnostics reject candidate pooling and flat-field batching on Q2/K2 and Q8/K4. BF16 fails the frozen equivalence gate; the packed kernel is slower and opt-in. | Retain current scheduling. Measure complete request and service cost for any next optimization. New checkpoint/readout profiles require their own reference and accelerated-service qualification. |
| Natural documents | E29 frozen comparison and E30–E32 adaptations are complete; no complete quality/retention pass. | Investigate supported-answer loss and test one attributable remedy if released specialists also fail. |
| Evidence | A3 has 51 reviewed rows plus six non-independent follow-ups; A4 aligns four papers and 40 caption candidates. The separate 27-case disposition leaves gold unchanged. | Complete required source/PDF, table/reference and independent correction review. Existing visibility work is bounded, not full repair. |
| Cheap readers | Pooled-root, learned StateQuery and factorized applicability studies miss complete quality/policy gates. | Keep causal execution available. A new encoder or bottleneck explanation is not established. |
| MoE follow-up | Native 36/96 correct; late-six skip 40/96 and 1.3169× faster, with unknown recall 0/32 and 2/32. Peak allocation stays about 7.823 GiB. Both cache modes fail. [R1] | No pruning, cache or automation promotion; baseline quality is inadequate. |
| Completed profile comparison | Dense: 155/192 SNLI and 84/96 synthetic, p50 49.28/52.74 ms. Native MoE: 110/192 and 32/96, p50 1,781.62/1,802.03 ms. FP32-router MoE: 109/192 and 33/96. All have zero qualified coverage. [R2] | Continue with the dense control for this workload. Generation, quantization and backend differ, so this does not establish an architecture-only advantage. |
| Completed execution checks | Dense full batch / sequential prefix / batched prefix reach Δp 0.032107/0.010099/0.019341. Both MoE arms pass 0/8 groups in every mode, with answer changes in several variants. [R2] | All speed sweeps and final/LRU cache comparisons were skipped. Diagnose dense numerical paths before collecting a cache speed claim. |
| Actual routing | Observer parity passes. FP32 router removes observed ties on one probe, but full/split expert-set differences persist. Each MoE arm changes 9/24 option-order decisions, versus dense 0/24. [R2] | Router precision alone did not repair quality or cache equivalence. Do not repeat this intervention as the next main experiment. |

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
| New frozen execution control | `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`: BF16/SDPA dense profile measured in R2. This is a different model from J1, not a replacement result on ContractNLI/QASPER. |
| Sparse research control | `Qwen/Qwen1.5-MoE-A2.7B-Chat`, revision `ec052fda178e241c7c443468d2fa1db6618996be`: NF4/eager profile in R1/R2. Retain native routing; the FP32-router arm is diagnostic. |
| External candidate screen | Compare **imajev-4b phase 3** and **JevK5 4B v0.3** on artifact identity, provenance and the document input contract, then select one for the M2 system comparison. Imajev shares J1's base revision but requires at least two declared Choice options; settle the one-candidate QASPER mapping before selection. Resolve immutable weight/runtime revisions; never load moving `main` as an experiment identity. JevK5 v0.2 is a historical reproduction option. |
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
| **M2 — Qwen quality** | Current controls complete; qualify dense acceptance and document retention; then one released 4B specialist or one justified adaptation | Complete non-final quality, retention and policy pass |
| **M3 — Cost** | Dense numerical/cache diagnosis, qualified cold/warm comparisons, then joint-option Rust/MLX port | Accepted quality at lower complete cost, within resource limits |
| **M4 — Preview** | Fresh confirmation, accelerated service qualification and reproducible manifest | All locked release limits pass |

The benchmark harness now supports an unwarmed exact A/B/A run and a separate
non-final Choice quality/history report (see [BENCHMARKS.md](BENCHMARKS.md#non-final-choice-qualification)).
A synthetic checkpoint probe is only a bounded history observation. Document
quality, specialist comparison, broader history and production limits remain
open. The selected profile and daemon defaults are unchanged.

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

**Current evidence:** the new dense arm has useful raw accuracy but no policy
qualification. The ≥30 accepted-state / ≤10% upper-bound screen is a declared
research rule, not a production risk specification. Inspect per-state error
clusters and risk/coverage on development data; if power is insufficient,
register more independent source states. Do not relax the rule after reading
final results or count three correlated questions as three independent states.
Any revised selector needs fresh confirmation. R2 does not close the older
ContractNLI/QASPER retention gate.

The completed policies all fell back to review-all at development selection;
this is not six measured zero-error automation policies. A retrospective
development-only scan finds dense zero-error threshold regions containing
11 questions / 10 states on SNLI and 31 questions / 23 states on synthetic,
short of the 30-state requirement. Register enough independent policy-development
and policy-audit groups for the intended coverage/risk before another test.
Keep thresholds fixed during audit and retain a fresh final partition;
do not lower the rule or promote those diagnostic thresholds after exposure.

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
  Add option-order/code-permutation, independently reviewed meaning-preserving
  option-description variants, and unsupported-input checks. Renaming labels is
  not a clean memorization test if it changes their meaning. QASPER's
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

**Completed Colab study and next numerical experiment:**

- [x] Deliver the [prefill speed/accuracy notebook](https://colab.research.google.com/drive/1a3_k4ZbV459Hyr2OhXGjQpKrHHfZoe9L), with 864 questions across 288 states, budget/resume support and independent baseline quality when cache qualification fails.
- [x] Close all three arms and verify all 26 checksum entries. No execution errors are recorded. Every alternative execution mode fails qualification; latency sweeps and final/LRU cache comparisons are skipped. [R2]
- [x] Compare actual native and FP32-router MoE behavior. The precision change does not repair task quality or prefix equivalence. Retain this negative result; do not rerun the completed notebook merely to resume it.
- [ ] On fixed development probes, compare dense BF16/SDPA with matched BF16/eager and, if memory permits, FP32. Pin TF32 and attention settings; change one factor at a time. Record full, batched and split outputs, raw probability drift, calibrated drift, decisions and policy actions. This isolates candidates for the mechanism; it does not assume SDPA, BF16 or routing is the cause.
- [x] Capture actual selected expert IDs and verify observer parity. The new observation supersedes the old top-(k+1) reconstruction. On one 115-token probe, FP32 routing removes exact boundary ties but leaves expert-set disagreement. The FP32-router control remains distinct from R1's NF4/FP32-linear reference.
- [ ] If a mode passes, measure cold and warm cache requests against the **fastest qualified uncached method**, including full-input batching. Include prefill, copies, tokenization, synchronization, policy and LRU hits/misses/evictions. If no mode passes, stop its sweep and retain scalar full input.

The completed notebook planned prefix targets 256/768/1,536, Q=1/4/8 and
three outcomes, but no timing-grid rows were collected because qualification
failed. Do not claim measured behavior at these planned shapes. The broader K
and natural-document study below is also still open. A code/precision change creates a new run identity;
do not merge it into the current checkpoints. Use the 90-minute budget and
checkpoints for useful measurements, not repeated unqualified sweeps.


**Available now:** the pinned FP32 `ReferenceOps` native compute study has
compared nested batching, same-position candidate pooling, and a Rust port of
flat shared-root field batching on matched Q2/K2 and Q8/K4 token workloads.
The [flat-field](benchmarks/2026-09-27-python-flat-field/) and
[candidate-pooling](benchmarks/2026-09-27-candidate-pooling/) records reject
those two diagnostic alternatives. Compatible forced vectorized lanes span
2–8; automatic scheduling remains per-lane. Full request-path and
queue-inclusive comparisons for a new candidate optimization still need
their own paired runs. This systems study is separate from model selection.

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
  full prompt lengths. Hold state, question and correct outcome fixed where
  possible while adding reviewed distractors; keep taxonomy-granularity tests
  separate. Measure accuracy, NLL/Brier, rejection, token count and memory by K. These are proposed measurement points, not supported
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
One bounded exception landed 2026-09-27: the `laya` decision-encoder family
(three rust-loadable profiles with reference-parity readout, benchmarked and
installable through the registry) was implemented per the registry guide's
expansion order; it does not reopen the broad rollout.

Defer non-Qwen encoders, diffusion, multimodal expansion, multi-engine routing,
private RLCD reconstruction, and MTP for the no-decoding path. Reopen shallow-head,
pooled-root or replay variants only with a new diagnosis. Actual expert streaming remains untested. The next bounded study, only after a
quality-qualified model demonstrates a memory need, is **resident versus
streamed execution of identical experts**, not another arbitrary allowlist.
Keep attention, routers and shared experts resident; record expert bytes
read, cache hits, transfer stalls/overlap, disk and OS-cache conditions,
process memory, and complete cold/warm latency with output parity.

R1 touched 59.875 of 60 experts per layer on one profiled request; broader
prefill can erase the expected I/O benefit of per-token sparsity. Start with
a trace-based transfer lower bound and compare against the accepted resident
dense baseline before implementing streaming. Stop if memory savings do not
justify the added request cost. Q-wide expert batching is a hypothesis: it
may amortize reads while enlarging the working set and activation memory.
Joint-option inference has one suffix per question; K changes that suffix,
not an automatic Q×K lane multiplier. Expert-major execution and prediction
must preserve native routing, with miss fallback rather than dropped experts.
CUDA results do not qualify MLX/SSD behavior. Weight streaming, static
pruning and KV compression address different costs.

## Historical task crosswalk

| Previous IDs | Current disposition |
|---|---|
| 0–1, 2A–2H, 2H-C1–C5 | Completed contracts/research; retain results and failures. |
| 2I.1–2I.2, 4E-ADJ, A3/A4 | M0/M1 retain scope, evidence repair and independent-review work. |
| 2I.3–2I.6, 3A/3B, 3.1–3.10 | Preserve rendering, branching, native parity, high-K and persistence evidence. M2 tests usefulness; M3 measures relevant cost. |
| 2J.1–2J.4, 4A–4D, 4E-B.0/B.1/B.2 | Exploratory comparisons complete; no Phase 4 model accepted. |
| E29–E32 | Frozen comparison, LoRA, KL and label replay complete. Preserve FP32/BF16 identities, selections, failed retention and fixed80 diagnostics. |
| R1: MoE follow-up | Complete exploratory failure: speed gain, inadequate quality/acceptance, unqualified cache. No model promotion. |
| R2: prefill accuracy lab | Complete, three arms, 26 verified hashes. No acceptance or cache qualification. Dense is the next research control; M2 owns policy/retention and M3 owns numerical diagnosis. |
| 2J.5–2J.6 | M2 comparator selection; M4 fresh release confirmation. |
| S.1–S.5, 3.11 | CPU contract/service evidence complete within scope; M4 owns accelerated-service qualification. |
| 3M.0–3M.8 | Retain runtime/FP32 parity and bounded daemon/memory evidence. M3 measures cost; M4 owns promotion. |
| P2.1–P2.3, 4A.3 | Conditional cost/distillation work; teacher quality remains a prerequisite. |
| F0–F5 | Protect F0 reference; prioritize the Qwen joint-logit profile from F2. Other family expansion is deferred. |

## External source register

X1–X5 retain the **26 September 2026** source review. X6–X10 and R1–R2
were checked **27 September UTC / 26 September America/Chicago**. External measurements are
publisher/author reports, not OpenKind replications. Preserve cited release
versions in future experiment manifests; live pages can change. X11–X13 are
the 27 September imajev/board review.

- **X1:** [Benchmark Heaven alternatives and methodology](https://benchmarkheaven.com/jev-models/alternatives).
  The user's v1.4.1 report is a dated snapshot, not the current rank list.
- **X2:** [JevK5 4B model card](https://huggingface.co/alibiserikbay/JevK5).
  Distinguishes v0.2 benchmark evidence, v0.3 author evaluations and data provenance.
- **X3:** [JevK5 changelog](https://github.com/allebee/jevk5/blob/main/CHANGELOG.md).
  Release changes and the 24 September training-split correction.
- **X4:** [JevK5 implementation](https://github.com/allebee/jevk5) and
  [prompt/readout contract](https://github.com/allebee/jevk5/blob/main/jevk5/prompt.py).
- **X5:** [SemIf MLX implementation notes](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/docs/MLX.md).
- **X6:** Hötter and Rosenmüller, [PrivateMode: Turn GLM-5.3-Flash into a Jev-like System One model](https://www.privatemode.ai/blog/system-one-from-glm-flash), 24 September 2026. External vendor report; not rerun here.
- **X7:** [PrivateMode Decisions README](https://github.com/edgelesssys/privatemode-decisions). For a future vLLM adapter, request every allowed code's log probability explicitly rather than relying on a top-N list; verify tokenizer IDs at the answer boundary and include semantic none when required. The local selected-row path already avoids missing-option top-N truncation.
- **X8:** [PrivateMode benchmark README](https://github.com/edgelesssys/privatemode-decisions-benchmark). Paired comparisons, separate geographic latency probes and dated billing evidence; no local performance equivalence inferred.
- **X9:** [PrivateMode methodology](https://github.com/edgelesssys/privatemode-decisions-benchmark/blob/main/METHODOLOGY.md). Useful audit controls; model consensus and altered label semantics are not ground truth or clean causal tests.
- **X10:** [PrivateMode suite results](https://github.com/edgelesssys/privatemode-decisions-benchmark/blob/main/results/suite.md). Completed results control quantitative claims when planning prose differs.
- **X11:** [JevBench v1.4.2.2 official board](https://benchmarkheaven.com/jev-models),
  scored 27 September 2026. Composite and axes are board results, not OpenKind replications.
- **X12:** [imajev phase-3 model card](https://github.com/mohit67890/imajev/blob/6ee8a2c555ca6a3d1de9eceb33f1bd1cfeb268a2/model-cards/imajev-4b.md)
  and [technical report](https://mohit67890.github.io/imajev/report/#top). The report's section 6 retains the older rank-16/255-code recipe.
- **X13:** [imajev phase-3 selection and failed gates](https://github.com/mohit67890/imajev/blob/6ee8a2c555ca6a3d1de9eceb33f1bd1cfeb268a2/results/phase3/final-comparison.md)
  and [merged DecisionBench result](https://github.com/Hanno-Labs/decision-bench-results/pull/68).
- **R1:** [MoE follow-up `20260927T003918_481825Z`](https://drive.google.com/drive/folders/1gXwnF1sXhR5Yh4-7R2izDQ9TJXDl_Lf1). 22 artifact hashes checked; 96 authored final questions / 32 states. No streamed execution. See WORKING_PAPER §§4.3, references 7–8.
- **R2:** [Completed prefill study `20260927T015527_929864Z`](https://drive.google.com/drive/folders/14FsOosgKt9oJKK-Y8yPuKtkuaDClBgC3), [final summary](https://drive.google.com/file/d/1x6WPopdMqgma4PLUVNPYLP3S819oHmpo/view), [native MoE](https://drive.google.com/file/d/1RnEU6QnAgrTPHwDFQSA2GOUfHTKOay5Z/view), [FP32-router MoE](https://drive.google.com/file/d/1IdQLpiukvtV0FuCSJTPVyyJVnLv4pDg7/view). Completed at `2026-09-27T03:06:39.794435Z`; all 26 checksum entries verified. See WORKING_PAPER §4.4 and reference 9 for raw counts, probability drift, routing scope and reproducibility.

For exact historical checkboxes and run identities, use
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md). This documentation review runs no
inference or training, opens no final outcomes, and promotes no model.
