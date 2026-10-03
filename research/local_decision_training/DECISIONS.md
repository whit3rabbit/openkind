# Local decision training v4 decisions

Reviewed 3 October 2026. These decisions define an experiment system, not a
measured 4B improvement. The [run guide](README.md) owns commands and parameters;
the [whitepaper](../../docs/whitepaper/WHITEPAPER.md) owns local results. No
historical evaluation cases or model assets are imported by these changes.

## D1. Preserve a simple, bounded control

**Evidence.** E30–E32 show task specialization without complete retention
([whitepaper §§18.21–18.26](../../docs/whitepaper/WHITEPAPER.md)). The
[CLEF source review](../../docs/RESEARCH.md#cloudflare-clef-and-linked-decision-models-reviewed-2026-10-01)
discloses smoothed CE/Brier but no recoverable coefficient recipe.

**Limit.** Neither establishes the best 4B loss, training budget or Mac profile.

**Decision.** Keep the pinned post-trained Qwen3.5-4B, rank-16 LoRA and finite
answer-code readout. Start with unsmoothed CE, zero Brier weight, 400 updates,
effective batch 16 and 2,048 total prompt tokens. The frozen parent remains
selectable. Retain the existing six-arm loss sweep as an explicit option.

**Alternative.** A combined-loss default or rank-256 adapter would change the
control without local evidence for that choice.

**Experiment and acceptance.** Compare losses on one admitted dataset with
matched seeds, execution and budgets. Select by source-macro development NLL
subject to existing retention screens. Calibrate one winner and evaluate the
gate once. A failed gate exports the frozen parent, without trying another arm.

## D2. Separate split groups from atomic units

**Evidence.** Previous row-wise cap sampling could keep counterfactual siblings
in one role but include only some siblings in training. Shared document/request
identity still prevents leakage across the five roles.

**Limit.** Exact normalized-state grouping is not semantic decontamination.

**Decision.** Preserve the 75/8/7/5/5 training, development, calibration, gate and
reserved-test roles. Explicit pairs and triplets are atomic after deduplication
and token admission. Reject incomplete units; skip units that cannot fit a row
cap and report unused capacity. Ordinary document rows remain individually
sampleable within their assigned role.

**Alternative.** Whole-document atomic sampling can discard large natural
corpora; row-wise sampling breaks the intended counterfactual comparison.

**Experiment and acceptance.** Offline tests must prove role isolation, complete
atomic units, cap compliance, deterministic sampling and reported rejections.
Compare actual admitted counts and source exposure before interpreting quality.

## D3. Isolate new reasoning data

**Evidence.** [E43/E44, whitepaper §24](../../docs/whitepaper/WHITEPAPER.md#24-openkind-t4-recovery--classifier-verification-input-reuse-nli-specialization-and-general-policy-limits)
identify capping, chronology and partial-information errors. An eligibility
repair damages retry decisions in the same requests. The
[pinned Strands v20 outcome](https://raw.githubusercontent.com/strands-labs/strands-decider/f91487ab8f7e4b4967ae57e46b8d90e91e67d616/research/preregistrations/PREREGISTRATION-v20.md)
improves its generated instruction-flip pairs but fails required retention guards.

**Limit.** These model, renderer and task settings differ from this 4B trainer.
Generated examples can teach their own wording without improving transfer.

**Decision.** Keep `data_intervention="control"` as the existing mixture. The
`reasoning` intervention replaces half the capped rules allocation with exact
multi-digit capping, chronological-order, decisive-known-fact and answer-changing
instruction families. Use independently identified facts, split-isolated groups
and held-out rendering families. Keep broad-source exposure, total training
presentations and evaluation manifests fixed. Preserve family, template, request,
intervention, supervision and semantic-none provenance in predictions.

**Alternative.** Importing historical cases, adding rows without an exposure
control or copying Strands' combined treatment prevents an attributable comparison.
HelpSteer2 remains a separate opt-in data experiment.

**Experiment and acceptance.** Test labels against exact rules, pair completeness,
held-out templates and cross-arm evaluation identity. Report both-siblings-correct
and complete-request correctness alongside retained source/class scores. A skill
gain alone cannot satisfy the existing candidate screen.

## D4. Vary presentation by reproducible occurrence

**Evidence.** [E42, whitepaper §23.6](../../docs/whitepaper/WHITEPAPER.md#236-prompt-schema-sensitivity-and-dynamic-choice-probes)
finds large probability shifts under code, key and option transformations.

**Limit.** Process stability and a correctly implemented transform do not prove
presentation invariance or better quality.

**Decision.** Default `presentation_augmentation="none"` fixes one presentation
per admitted row. Optional `option_order`, `code_assignment`, `opaque_keys`, and
`all` modes derive each training occurrence from `augmentation_seed`. Preserve
canonical hard/soft targets and semantic none exactly. Keep `split_seed`,
`initialization_seed`, `sampling_seed`, and `augmentation_seed` separate; unset
values inherit `seed=17`.

**Alternative.** Changing every presentation dimension and the semantic dataset
together obscures the cause. Fresh unrecorded randomness breaks checkpoint replay.

**Experiment and acceptance.** Test each transform independently, target/code
alignment, deterministic occurrence replay and resumed training. Keep evaluation
rows fixed across arms. Report probability and winner changes as diagnostics,
without converting them into new native parity tolerances.

## D5. Measure retention beyond average accuracy

**Evidence.** [E43, §24.2](../../docs/whitepaper/WHITEPAPER.md#242-qwen-the-failure-first-eligibility-repair-nf-and-collateral-retry-regressions)
shows a repaired field with worse complete requests. [E44, §24.6](../../docs/whitepaper/WHITEPAPER.md#246-bert-family-policy-classification-fine-tuning-gains-transfer-limits-and-majority-collapse)
shows high field accuracy from majority-class predictions. The recovered encoder
experiments are complete within their bounded recipe; the old harness failure
is no longer the current status.

**Limit.** Default panels are small. Repeated rows and paired renderings are not
independent samples; sparse slices cannot confirm retention.

**Decision.** Preserve existing source/class, NLL/Brier and false-none screens.
Add family/kind slices, predicted-class distributions, majority controls, paired
correctness and complete-request correctness where fixtures declare requests.
Report independent group counts and paired group-bootstrap intervals. Label
numerical thresholds as pilot screening rules, not confirmed guarantees.

**Alternative.** Aggregate accuracy can reward class collapse. Row bootstraps
would overstate precision when siblings share facts.

**Experiment and acceptance.** Exact toy panels must detect majority collapse,
field-versus-request disagreement, incomplete requests and paired group changes.
Keep all new descriptive slices out of checkpoint and loss-arm selection.

## D6. Keep confidence semantics explicit

**Evidence.** The [reference inference contract](README.md#outputs-and-verification)
uses normalized entropy for Choice/Score confidence. [Whitepaper §§24.7–24.9](../../docs/whitepaper/WHITEPAPER.md)
separates useful classification from accepted automation policies.

**Limit.** High probability or low entropy is not an error-rate guarantee.

**Decision.** Report maximum-probability and exported entropy-confidence
risk/coverage separately. Noul retains no confidence field. Keep calibration-only
temperature fitting, one gate evaluation and frozen-parent fallback.

**Alternative.** A single unnamed confidence curve can silently measure a
different quantity from the exported helper or imply application authorization.

**Experiment and acceptance.** Tests distinguish the two score definitions,
verify small-denominator counts and fallback temperature, and preserve `decide`
and Jev wire interfaces. Neither curve selects a consumer action policy.

## D7. Bind frozen evaluation to exported artifacts

**Evidence.** Earlier bundle checks hashed exported files but did not fully bind
the executed trainer, tokenizer and adapter to that verified export.

**Limit.** Integrity checks establish artifact identity, not model quality,
trust in arbitrary code, native numerical parity or deployment readiness.

**Decision.** Version run artifacts as v4 and include implementation dependencies
in run identity. Preserve older artifacts and reject incompatible resumes.
Final evaluation and benchmarking share a verifier/loader for the exported
implementation, tokenizer, adapter, accepted calibration and inference contract.
Export a small synthetic replay pack with exact rendered inputs, token/code
mappings, logits, temperature, probability vectors and typed answers.

**Alternative.** Loading the current working-tree trainer while checking a
bundle's hashes leaves a different implementation responsible for its scores.

**Experiment and acceptance.** Offline tests reject changed implementation,
tokenizer, adapter, calibration and contract files before frozen inference. Check
replay semantics and frozen-parent restoration. TypeSafe/reference labels remain
outside fitting and selection, and failed-question denominators remain intact.
The notebook must embed the reviewed runtime and test sources exactly.

## Completion boundary

Run the offline suite, validate notebook structure and embedded source identity,
and run repository-required checks. Full Colab 4B training, L4 NF4 execution,
Drive resume, adapter merge, quantization and native Mac qualification require
separate measured runs. A 4,096-token study and learned pointer/evidence reader
also remain separate experiments. None is implied by completion of this code.
