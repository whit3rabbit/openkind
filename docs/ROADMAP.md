# OpenKind roadmap

**Revision 0.9.0, 24 September 2026.** Current priorities and exit conditions.

Build a local, auditable decision engine that answers several independent,
well-scoped questions over shared evidence. Keep state-first computation,
typed probability distributions, and an independently evaluated review policy.
The next milestone is useful evidence-grounded decisions. Model size and a
custom neural architecture are choices to test after that milestone.

**Active sequence: define the workload, repair the evidence contract, prove
useful decisions, reduce cost, then confirm a scoped preview.** One frozen-profile
MLX performance study can proceed alongside the first two steps. New model
fitting waits for the repair and review gate.

This file owns current work. The [whitepaper](whitepaper/WHITEPAPER.md)
owns scientific interpretation, [ARCHITECTURE.md](ARCHITECTURE.md) owns implemented
contracts, [MLX.md](MLX.md) owns backend operations, and
[BENCHMARKS.md](BENCHMARKS.md) owns measurement methodology. The previous roadmap,
including its evidence register and latest evidence addenda, is preserved
in [ROADMAP_HISTORY.md](ROADMAP_HISTORY.md). Historical task IDs remain traceable
through the crosswalk below.

## Current position

| Area | Recorded result | Remaining decision |
|---|---|---|
| Native CPU reference | Frozen-profile parity, complete hybrid-state branching, bounded high-K completion, full persistence replay, and named-machine lifecycle/load/soak pass within their declared fixtures. | Maintain the differential oracle. Completed bring-up is not the next product milestone. |
| Apple Silicon execution | Pinned-base MLX FP32 full, nested, and variable-length vectorized parity pass. Forced daemon execution and bounded memory recovery are recorded. | Measure vectorized request cost. MLX load/soak and a clean-commit promotion record remain open. BF16 fails its frozen gate. The slower packed kernel remains opt-in. |
| Natural-document decisions | Phase 4A–4D and exploratory 4E-B.0/B.1/B.2 have produced no promotable model. Some candidate-ranking signal is present, but applicability and policy transfer remain inadequate. | Establish useful decisions on reviewed, model-visible evidence. |
| Evidence repair | A3 preserves 51 independently adjudicated rows, with later non-independent follow-ups on six of them. A4 records bounded dataset alignment for four papers and 40 caption candidates. A separate 27-case disposition leaves gold unchanged. | Resolve PDF/source equivalence, missing table/reference content, exact-input visibility, and independent review of proposed corrections. |
| Cheap shared queries | The pooled-root probe fails. Earlier learned StateQuery and factorized applicability arms also miss complete quality/policy gates. | A cheaper reader remains conditional. Neither a terminal-token bottleneck nor a superior bidirectional replacement has been established as the cause or remedy. |

Evidence: whitepaper [§17.3](whitepaper/WHITEPAPER.md#173-current-rust-boundary)
and [§§18.7–18.19](whitepaper/WHITEPAPER.md#187-completed-phase-4a2-architecture-and-weight-8-sweep),
the [CPU service report](verification/native-service-gate/2026-09-22-rerun2/README.md),
the [MLX follow-up](verification/phase3m-2026-09-22/README.md), and
[A4 alignment QA](../research/24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/QASPER_SOURCE_ALIGNMENT_QA.json).
Execution completion, numerical equivalence, model acceptance, and release
promotion are separate statuses.

## Active milestones

| Milestone | Status and dependency | Deliverable and exit |
|---|---|---|
| **M0: Supported workload** | Active, scope and limits open | A task/input/policy contract with explicit semantics, deployment hardware, finite quality/resource limits, selection rules, and review responsibility. |
| **M1: Evidence repair** | Active, building on A3/A4 | A versioned source/annotation overlay, serializer, exact-input visibility ledger, and independent review. Defects are repaired or quarantined under prediction-blind rules. |
| **M2: Useful decisions** | Gated on M0/M1 | Retained reference methods on common repaired inputs, then at most one justified adaptation hypothesis and matched control. All declared non-final quality and policy screens pass. |
| **M3: Lower cost** | Frozen-profile systems study can start with a declared workload. New learned challengers wait for M2. | A matched MLX execution comparison, then one smaller-model or cheaper-query challenger. Retain the useful operating point while reducing complete cost. |
| **M4: Scoped preview** | Gated on a locked M2/M3 operating point | Fresh held-out confirmation, accelerated-service evidence, model card, and release manifest. All declared quality, coverage, risk, latency, memory, and lifecycle requirements pass. |

### M0: Define the first useful workload

Start with document-evidence judgments: several atomic questions over the same
supplied document, modest option sets, explicit descriptions, and declared
unsupported inputs. Four distinct questions is a proposed demonstration target,
not a validated semantic capability or a new release requirement. Broader Noul
and Score quality claims need their own task definitions and evidence.

- [ ] Define document sources, question/rubric families, option limits, input
  token budgets, truncation/rejection behavior, and prediction-blind eligibility.
- [ ] Distinguish semantic none, missing observations, contradiction,
  out-of-domain input, and application review where the task requires them.
  Preserve the current wire semantics and consumer authorization boundary.
- [ ] Approve deployment hardware and finite selection bounds. The named M4 Max
  is the existing development host. Neither it nor the exploratory L4 is an
  approved deployment requirement by implication.
- [ ] Define the main outcome: **correct accepted decisions per second subject
  to accepted-error, minimum-coverage, latency, and memory limits**. Record the
  decision population and timing scope, including queueing for service claims.
  Accepted-error is undefined at zero acceptance.

Carry forward the unresolved release fields from the
[reviewed-study contract](whitepaper/WHITEPAPER.md#144-why-the-next-study-remains-blocked):
`max_family_macro_nll`, `max_panel_accepted_error`, `min_policy_coverage`,
`max_request_p95_ms`, and `max_peak_allocated_gib`. Define their population,
aggregation, and memory/timing semantics before selection. No numerical limit
or reviewer approval is supplied by this roadmap revision.

**Exit:** a versioned contract fixes the supported scope and selection rules.
Missing limits or independent-review responsibility block model selection and
promotion, while bounded contract and data preparation can continue.

### M1: Repair the source-to-input evidence contract

The next modeling workbench is data and visibility work. Preserve original
corpora, result locks, thresholds, partitions, and the final boundary. Put
corrections in a new versioned overlay, with an attributable serializer and
regenerated input/feature identities where inputs change.

- [ ] Extend the bounded A4 dataset alignment to the required document versions,
  table cells, captions, and references. Its four aligned papers and 40 caption
  candidates do not establish PDF-version equivalence or recover table values.
- [ ] Record source revision, original annotations and aggregation/tie rule,
  serialized state, finalized model input, adjudicated outcome, and visibility.
  Trace positive evidence into actual input tokens. Record grounds for negative
  or ambiguous judgments without inventing positive spans.
- [ ] Separate source answerability from answerability using the visible state.
  The full-paper audit does not establish evidence coverage inside the Phase 4
  feature cache's 1,024-state-token limit.
- [ ] Preserve the A3 51-row population, six same-assistant follow-ups, and
  separate 27-case disposition. Keep the 77 original-annotation ties diagnostic
  until a new reviewed contract changes their treatment. Review proposed
  corrections independently of implementation.
- [ ] Export the repair/review pack with partition and identity checks. Keep
  annotation evidence and labels out of inference inputs unless the same
  information is legitimately available under the deployment input contract.

A bounded text-complete prototype scope is allowed as a separately registered
benchmark. Fix eligibility without consulting model predictions, report
exclusions by reason/source, and quarantine unresolved cases. Do not discard
hard examples because of their errors or call a narrowed scope a pass of the
original two-source Phase 4 gate.

**Exit:** the review record resolves each proposed correction or quarantine,
and the effective-input ledger distinguishes missing evidence from model
failure. A3 adjudication need not be repeated. Its proposals must not silently
become benchmark gold. Final remains closed.

### M2: Establish useful decisions

First evaluate full question/candidate-conditioned Qwen on repaired inputs,
including a fixed direct-logit method. Compare it with the retained Phase 4D
StateQuery diagnostic parent and majority/prior controls on those same inputs.
These are new results. They do not replace historical tables or imply that the
reference profile already has adequate answerability.

- [ ] Freeze a bounded non-final comparison, including calibration and
  training/development-only threshold selection, deterministic ties,
  all-answer/all-none controls, and minimum positive-class support. Preflight
  the selection procedure for reachable operating points before reading gate
  outcomes. The 4E-B.2 grid failure does not authorize post-hoc reselection.
- [ ] If visible evidence is sufficient and full question/candidate-conditioned
  Qwen still fails, test **one limited upstream adaptation hypothesis** with an
  attributable head-only control. Explain the difference from B0/B1/B2 and the
  earlier 2B LoRA pilot. Token-level access and applicability factorization
  have already been tested. Match input, supervision, and optimization budgets
  where causal attribution is intended.
- [ ] Report conditional ranking, full semantic accuracy, per-class recall,
  semantic-none recall, false-none, NLL/Brier, and review-policy cost/coverage
  by source. QASPER's one-candidate task has no conditional ranking metric.
- [ ] Retain the applicable Phase 4 research screen: source-specific recall
  floors, false-none at most 0.20, the registered proper-score non-regression
  rule, and policy cost below review-all at 0.10 with useful coverage. Copy the
  exact rules and comparison population into the new contract before fitting.
  These are research screens, not production risk guarantees.

Use reviewed oracle evidence windows only to diagnose causes. If a window
works while the full document fails, investigate evidence access. If both
fail, revisit semantics, supervision, and adaptation. Oracle windows are not
deployable retrieval evidence.

**Exit:** a complete non-final pass under the registered screen. Failure pauses
that hypothesis. Do not expand seeds 42/123, fit another nearby head, or open
final on the strength of one favorable metric. A changed model receives a new
profile and execution reference. The frozen integration profile stays intact.

### M3: Reduce complete request cost

**Systems study, available alongside M0/M1:** compare pinned FP32 MLX
`ReferenceOps` per-lane and forced vectorized execution under the same finalized
inputs, checkpoint, renderer, head, and policy. Declare realistic state lengths,
Q/K, and unequal suffix lengths before timing. The current forced vectorized
path supports 2–8 compatible lanes. Automatic scheduling remains per-lane until
performance evidence supports a useful range.

- [ ] Record cold/warm state, repeated-full and shared execution, complete
  request latency, `T(Q)/T(1)`, marginal question cost, forward calls, branch
  bytes, allocator use, and process memory with host/commit/runtime identity.
- [ ] Keep probability, argmax, policy, and state-isolation checks in the same
  evidence bundle. Parity alone cannot promote automatic batching or the
  currently slower packed kernel.
- [ ] After M2, select one learned cost challenger. Prefer limited Qwen2B
  adaptation on the strength of its exploratory evidence, unless profiling
  identifies question continuation as the reason to test a richer shared-state
  query reader. Do not launch both a size ladder and an architecture sweep.

**Exit:** a same-model implementation passes parity and lowers complete cost,
or a new model passes its own quality/resource gate at the required operating
point. A failed cheap reader leaves causal branching available. Keep high-K
correctness and admission fixtures, but make K=255 latency optimization
conditional on workload demand. Weight quantization remains a separate,
newly qualified operating point from the failed low-bit KV snapshots.

### M4: Independently confirm a scoped preview

- [ ] Lock model, renderer, input contract, calibration, review policy, fair
  baseline configurations, and selection rules before final outcomes.
  Previously inspected non-final gates remain diagnostic. The exposed
  `2ij.2.0` final panel is not fresh confirmation.
- [ ] Evaluate protected source-state groups and held-out question/rubric
  families for transfer claims. Estimate uncertainty by source-state group,
  not by treating expanded questions or variants as independent observations.
- [ ] Qualify the actual accelerated service for overload, deadlines,
  cancellation/recovery, memory stability, and queue-inclusive latency/load/soak
  under the approved resource envelope. CPU service evidence does not qualify
  MLX automatically.
- [ ] Publish supported tasks, exclusions, risk/coverage, hardware, model/runtime
  identities, failure strata, model card, and reproducible release manifest.

**Exit:** all predeclared limits pass without post-final tuning or reselection.
A failed confirmation stays failed and requires fresh data for a new selection.
A scoped developer preview is not a general Jev replacement. Wire compatibility
does not establish semantic competence.

## Paused and conditional work

Keep one modeling hypothesis and one frozen-profile systems experiment active
at a time. Pause additional scalar-none-weight, shallow-head, pooled-root, and
handcrafted-residual sweeps without a new diagnosis. Distillation requires
demonstrated teacher quality on the supported task. A numerical oracle is not
automatically a reliable teacher.

RLCD reconstruction, a diffusion rewrite, MTP for the no-output-decoding graph,
and repeated low-bit KV snapshots on the same workload are outside the critical
path. A custom small encoder, bidirectionality, or resemblance to Jev's private
architecture is not a success criterion. Do not reopen completed CPU bring-up
or persistence gates as the main milestone.

## Historical task crosswalk

| Previous IDs | Current disposition |
|---|---|
| 0–1, 2A–2H, 2H-C1–C5 | Completed historical contracts/research. Retain results and failures in the history and whitepaper. |
| 2I.1–2I.2, 4E-ADJ, A3/A4 | M0/M1 carry the remaining scope, evidence repair, and independent-review work. A3 review and bounded A4 alignment are recorded assets. |
| 2I.3–2I.6, 3A/3B, 3.1–3.10 | Preserve completed rendering, branching, native parity, high-K correctness, and replay evidence. M2 tests semantic usefulness, M3 measures relevant cost. |
| 2J.1–2J.4, 4A–4D, 4E-B.0/B.1/B.2 | Completed exploratory comparisons, no Phase 4 model accepted. New modeling belongs under M2 after M1. |
| 2J.5–2J.6 | M2 freezes useful comparators, M4 owns reviewed release confirmation and fair baseline evidence. |
| S.1–S.5, 3.11 | Native CPU contract/service evidence complete within scope. M4 retains actual accelerated-service qualification. |
| 3M.0–3M.8 | Recorded runtime/FP32 parity and bounded daemon/memory evidence retained. M3 owns matched performance, M4 owns service promotion. BF16 and community-profile qualification remain unpromoted. |
| P2.1–P2.3, 4A.3 | Conditional M3 work after useful quality and an identified cost problem. Teacher quality is a separate prerequisite. |

For exact old checkboxes, run identities, and source links, use
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md). This refocus changes the work order
and documentation. It registers no experiment, runs no training or inference,
opens no final outcomes, supplies no independent approval, and promotes no model.
