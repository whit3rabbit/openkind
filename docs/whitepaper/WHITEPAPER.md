# OpenKind
## Shared-state decision inference: evidence, execution, and useful decisions

**Document version:** 0.8.5 model-research record with 27 September 2026 Rust/MLX systems addendum (completed v0.6.0 source-label replay results saved approximately 19:29 UTC on 26 September)

### Abstract

OpenKind investigates local decision inference over shared evidence. It returns
typed answers and probability distributions without an autoregressive text
generation loop. Its state-first Qwen reference reuses complete hybrid execution
state across isolated question and candidate branches. Native CPU parity,
fresh-process decision replay, and named-machine CPU service gates pass within
their declared scope. Pinned-base MLX FP32 full, nested, and variable-length
vectorized parity pass separately. Paired native compute tests reject the
Python-style flat-field and cross-question candidate-pooling diagnostics on
the tested shapes. Complete request-path performance and MLX service
promotion remain open. [E11–E13; RUST1–RUST11; RUSTM2; §§17.3–17.4]

Natural-document learning now has a bounded positive result, but no promotable
multi-source model. The completed frozen J0/J1 two-budget comparison separates
evidence visibility from semantic-none behavior. A matched rank-16 decision-LoRA
pilot then improves ContractNLI accuracy from **56.86% to 81.37%** for Base and
from **69.12% to 82.35%** for the post-trained checkpoint under common BF16
execution. Contradiction and none decisions improve, but entailment recall and
QASPER retention regress; both adapted arms fail the complete research screen.
The result is useful task specialization, not a general decision-model upgrade.
A completed follow-on adds short-premise SNLI replay with frozen-parent consistency.
It improves probability scores relative to the specialists but does not meet joint
preservation requirements; both guarded selectors retain the frozen parents.
The completed source-label replay comparison then raises SNLI regression accuracy
to **82.81% / 86.46%**, but worsens ContractNLI/QASPER probability scores relative
to parent KL and still fails supported-entailment and QASPER preservation. Both
new guarded selections also retain the frozen parents. [E29–E32; §§18.20–18.25]

The evidence audit separates source correctness, model-visible evidence, and
model use. New finalized-input diagnostics show substantial gains on fixed
question subsets whose annotated evidence becomes visible at a larger budget.
The LoRA pilot uses complete, source-record-checked ContractNLI training documents;
it excludes rather than truncates over-cap documents. Stored-record consistency
and annotation visibility do not establish independent semantic adjudication or
complete source-document equivalence. Existing audit quarantines and benchmark
labels remain unchanged. [E14; E22–E25; E29; E30]

Behavior preservation, replay-task learning, and retained correctness have
separate measured boundaries. Source-label replay substantially outperforms
parent consistency on the exposed SNLI panel while worsening the original
two-source probability scores. Its false-none behavior improves on SNLI but
worsens on QASPER, so a uniform rejection-shift explanation is insufficient.
The next proposed work targets the recurring supported-entailment regression
using admissible complete-document training/development records and a retention
scope that tests the intended operations. No corrective treatment is demonstrated
by this update. Lower cost remains conditional on useful scoped quality. The
contribution is an auditable execution contract and measured distinctions among
reuse, evidence access, targeted learning, retention, and automation utility—not
an inference about Jev's private architecture or a universally superior neural
topology. [E29–E32; V6; §18.26; synthesis]

**Current work:** the frozen comparison, contract-only pilot, parent-KL follow-on,
and source-label replay comparison have completed. None of the three tested
adaptation recipes satisfies the full quality/retention exit. Preserve J0/J1
controls, J2/J3 locked update-80 specialists, and J4–J7 diagnostic snapshots with
their selected-zero outcomes. Diagnose supported-entailment and long-document
retention before another bounded treatment; this is proposed work, not a new
fit or authorization. Protected final and promotion remain closed.
[ROADMAP.md](../ROADMAP.md) owns M0–M4 work items and is not edited by this revision.
The historical roadmap remains in [ROADMAP_HISTORY.md](../ROADMAP_HISTORY.md), and
the historical opening is retained in [Appendix E](#appendix-e-historical-opening-before-the-september-refocus).
[E30–E32; V6]

**Reading guide:** current interpretation in [§1](#1-executive-assessment),
acceptance tracks and work order in [§13](#13-refocused-research-program-and-next-milestone),
native evidence in [§17.3](#173-current-rust-boundary), and audit/model evidence
in [§18](#18-phase-4a4e-locked-benchmark-applicability-experiments-and-audit-gate).
The [frozen comparison](#1820-m1-visibility-preparation-and-completed-m21-frozen-comparison),
[matched LoRA pilot](#1821-m22-matched-decision-lora-pilot-in-domain-gains-failed-retention),
[retention follow-on](#1823-retention-aware-replay-pilot-partial-recovery-no-eligible-adaptation),
[source-label replay results](#1825-source-label-replay-pilot-stronger-snli-learning-failed-joint-preservation),
and [current synthesis](#1826-current-synthesis-after-the-replay-target-comparison)
are in §§18.20–18.26. Sections 4–10, the dated interpretations in
§§18.19/18.22/18.24, and prior checkpoint/revision records retain their historical
scope. Their original future-work language is not the current queue.

---

# 1. Executive assessment

**The execution foundation is established within bounded contracts. Targeted
natural-document learning is demonstrated; preservation across classes and tasks
is the next unresolved model question.** The frozen two-budget comparison shows
that evidence access and none-class behavior can move in opposite directions.
The subsequent matched 4B joint-option LoRA pilot improves ContractNLI strongly,
but fails entailment preservation, QASPER retention, and the complete research
screen. The completed parent-KL follow-on partly recovers probability quality;
source-label replay then improves SNLI strongly but worsens the main two-source
probability scores relative to KL. Both follow-ons return frozen checkpoints
under their guarded selectors. None of these results establishes arbitrary-domain
competence, validates a cheap state summary, or qualifies a changed model for the
existing native service. [E1–E7; E11–E32; §17.3; §§18.20–18.26]

A frozen text-only Qwen3.5-4B-Base backbone with last-token features and a linear head reached 87.67% matched and 87.33% mismatched accuracy in Phase 2B’s three-class MultiNLI experiment. Phase 2C selected the same head family across three training seeds and obtained 87.0% and 88.8% on new matched and mismatched test samples. These are sampled NLI results, not general decision accuracy, and separation from earlier experiments does not establish separation from Qwen’s pretraining corpus. [E1; E2]

The variable-candidate experiment is a second, different result. A shared candidate scorer trained on 57 Banking77 labels transferred to 20 labels withheld from head fitting. It achieved 88.54% accuracy on seen-label sampled-choice episodes and 82.29% on held-out-label episodes. However, missing-option recall was substantially weaker than accuracy on answerable cases. Later experiments improved this behavior with a seven-parameter set-aware “none” model, at the cost of more unnecessary rejection on answerable inputs. [E2; E3]

The systems experiments make precision part of the decision contract. Expanded Phase 2E completed both FP32 and BF16 workers, but completion did not establish equivalence. Relative to FP32 full-sequential execution, FP32 batching and cache reuse stayed within roughly 0.000011 maximum probability difference on the 128-episode panel, with no recorded class or policy-output changes. BF16 variants failed the declared 0.005 probability tolerance and changed decisions.

Because the panel contains only eight distinct messages, these counts are diagnostic. They are not estimates of deployment failure rates. [E5]

Phase 2F supplies the previously missing compression and persistent-reuse evidence. All stages completed with frozen weights, heads, and policies. In FP32, lossless and FP16-KV snapshots passed all 32 compression-panel episodes; the four low-bit TurboQuant variants failed the declared gates. Lossless GPU caching reduced total time across the controlled 32-request traces without recorded policy changes, whereas the compressed LRU path altered outputs. The practical result is a bounded benefit from exact-prefix reuse, not a general endorsement of low-bit cache compression. [E6]

Phase 2G adds fresh labeled decisions instead of extending only the old replay panel. Strict-FP32 lossless and FP16-KV cached paths pass both the 56-episode fresh parity subset and the 72-episode controlled-context panel. TF32-permitted full batching reduces measured K = 4/16 request latency from approximately 265/1,118 ms to 132/518 ms, yet it is not a fully interchangeable execution mode. Its fresh full-sequential comparison changes three head argmax outcomes across 416 episodes, despite no change under the nine frozen application policies. [E7]

The new task evidence shifts the immediate bottleneck toward criteria and rejection. The set-linear head's fresh accuracy is 69.53% on Banking head-training labels, 75.78% on Banking held-out labels, and 66.41% on sampled CLINC non-financial intents. On that CLINC in-scope panel, answerable accuracy is 93.75% but omitted-intent recall is only 39.06%; separate author-OOS recall is 46.88%.

These are different test constructions from earlier phases, not a measured deterioration in unchanged weights. A high-confidence four-episode error cluster derives from one ambiguously described physical-card request. The recorded errors remain errors while the criteria warrant review. [E7; R2]

**Phase 2H is now complete within its declared scope.** Both evaluation workers completed using the saved fits, original reserved final inputs and a checked artifact lock. The selected support/seed43 model records 85.55% Banking fitting-label accuracy, 89.84% CLINC fitting-domain accuracy, 63.67% held-out Banking accuracy, 66.02% held-out CLINC accuracy and 100% sampled author-OOS accuracy. Its raw pooled result is 78.91% / 0.721931 NLL. All 14 comparisons remain visible: original-criteria joint seed29 reaches 83.77% / 0.541440 on the same final panel, despite losing the original development selection. This is evidence to improve future selection and transfer tests, not permission to substitute a new winner after inspecting final results. [E9; V2]

**Closure is not promotion.** The selected model still has weak held-out omission recall, context-sensitive false-none behavior and residual wrong acceptances under the frozen policies. TF32 changes no final argmax or selected-model policy output, but two episodes exceed the 0.005 probability tolerance; additional saved-row context comparisons expose more drift. BoolQ and SST-5 now have bounded final readout scores, and complete in-process request timings are available. None establishes a general primitive, optimized-kernel dispatch, Q-level sharing, Rust/Metal parity or a real service. Section 13.1 preserves the full positive and negative readout. [E9; V2]

**Phase 4A converts the multi-question goal into a locked natural-document benchmark, but neither Phase 4A.2 nor the Phase 4B.2 scalar-weight follow-on produces a promotable model.** The corpus joins ContractNLI and QASPER under state/component-safe partitions and preserves final labels and predictions unopened. On the non-final calibration gate, the completed 4A.2 sweep shows that balanced B1 and factorized B2 improve source-macro discrimination, while QASPER semantic-none recall remains only 0.1000 and 0.1111. A QASPER-weight-8 B2 arm raises that recall to 0.1667 and reaches the best observed source-macro balanced accuracy/F1, 0.5872/0.5905. Phase 4B.2 then requests a cap of 12, but the empirical class ratio saturates the effective weight at 8.5602; QASPER none recall is still 0.1556 on the gate, while ContractNLI false-none rises to 0.3966. The scalar semantic-none-weight sweep is therefore closed as a negative result. [E14–E16]

**M2.1 and the bounded M2.2 pilot are now completed, not future work.** The FP32
frozen comparison supplies a conditional J1/4,096 quality challenger rather than
a universal winner. The BF16 pilot then trains J2 (Base + LoRA) and J3
(post-trained + LoRA), each for 120 attempted updates and both selected at update
80 on ContractNLI development NLL. On the exposed calibration gate, J3 improves
contradiction recall from 8/24 to 19/24 and none recall from 52/96 to 79/96,
while entailment falls from 81/84 to 70/84 and QASPER accuracy from 37/46 to 34/46.
J2 exhibits the same in-domain gain/retention-failure pattern. Preserve these as
useful research specialists, not promoted multi-source models. [E29; E30]

**The retention follow-on is also completed.** J4/J5 each train 120 updates, but
both guarded selections choose zero. At fixed80, replay improves NLL/Brier over
J2/J3 while each loses two ContractNLI decisions and gains one QASPER decision.
Both still fail entailment and QASPER preservation. The old specialists outperform
replay on the new SNLI diagnostic, limiting claims of general forgetting and of
parent agreement as a correctness proxy. The later target-source comparison is
now completed separately as E32; E31's outcomes and selected-zero identities remain
unchanged. [E31; V5]

**The source-label replay comparison is now completed as well.** J6/J7 each
complete 120 updates and again select frozen0. At fixed80, replacing replay KL
with source-label cross-entropy on the same inputs improves SNLI from 125/192 to
159/192 and from 131/192 to 166/192. It does not preserve the original task scope:
ContractNLI entailment is 70/84 and 69/84 against 81/84 for either frozen parent;
QASPER is 29/46 and 35/46 against 36/46 and 35/46 for J4/J5, with worse NLL/Brier
on both benchmark sources. Replay-domain learning and cross-task retention must
not be collapsed into one success claim. [E32; V6; §18.25]

| Question | Current evidence | Practical conclusion |
|---|---|---|
| Can Qwen make decisions without generating text? | Yes: trained NLI and candidate heads over frozen features. | Keep the decision-only path as a valid baseline. |
| Does dropping the LM head make the model small? | No removable parameters from the tied output projection in the measured text model. | Separate output-compute savings from weight-memory savings. |
| Can candidates be supplied dynamically? | Yes in Banking77 and a new sampled CLINC transfer test; matching offered intents is stronger than rejecting missing ones. | Do not label this arbitrary-domain calibrated competence. |
| Is “none” a solved safety mechanism? | No. Omitted-intent and author-OOS tests now both show rejection limits. | Keep semantic none, author-OOS detection, and application review separate. |
| Does shared-prefix execution work? | Strict-FP32 parity survives new-input and controlled-context panels. | Retain it as a sampled execution reference, not proof of semantic correctness. |
| Is BF16 interchangeable with that reference? | No, under the tested kernels and execution shapes. | Treat precision and batching as versioned model behavior. |
| Is TF32 permission a free speedup? | Full batching is about 2× faster, but some selected outcomes and longer-context equivalence checks change. | Keep FP32 storage and behavioral trade-offs explicit; do not automatically promote it. |
| Is cache compression already beneficial? | FP16 KV storage passed FP32 sample gates; all four tested low-bit codecs failed. | Retain lossless as the baseline; evaluate FP16 storage by prefix length and full cost. |
| Does persistent prefix reuse help? | Lossless FP32 savings were 13.91%/11.88% in F and 6.19% on G’s different expiry trace. | Benefits depend on workload and cache budget; no production or cross-question guarantee. |
| Has H completed and selected a universally better model? | Required final/robustness/primitive/request workers completed; the development-selected support arm has weaker held-out transfer than a retained original-criteria control. | Close recovery; keep model selection, numerical acceptance and broader generality open. [E9] |
| Have Phase 4A/4B solved multi-source question answering and semantic none? | No. The completed architecture and scalar-weight sweeps improve aggregate discrimination, but every arm misses the declared 0.30 QASPER none-recall floor. The cap-12 request saturates at effective weight 8.5602 and trades limited QASPER recovery for a 0.3966 ContractNLI false-none rate. | Keep final closed. Subsequent stratified, pairwise, head-only, and residual tests also failed the complete gate. Repair source/input evidence before another model intervention. [E14–E25] |
| Has the matched frozen J0/J1 comparison completed? | Yes, in FP32 at 1,024/4,096-token state-prefix caps; effects differ by source and evidence stratum. | Keep both controls; common-renderer results are not a universal checkpoint ranking. [E29] |
| Can bounded decision-LoRA improve natural-document decisions without damaging retention? | ContractNLI accuracy and contradiction/none recall improve substantially, but both adapters lose entailment and QASPER behavior. | Close the exact pilot as executed with failed full retention; test preservation, not automatic scale-up. [E30; V4] |
| Did parent-consistency replay preserve the useful gains without regressions? | It partially recovers probability scores relative to specialists, but no nonzero checkpoint meets the joint preservation rules; both selectors choose frozen. | Keep the completed KL recipe and historical choices; E32 supplies the separate source-label comparison. [E31; E32] |
| Did replacing parent KL with source-label replay repair retention? | SNLI accuracy/probability quality improve substantially, but supported entailment and QASPER retention still fail; both guarded selectors retain frozen0. | Close the exact source-label recipe; diagnose the recurring complete-document failure rather than equating a better replay proxy with retained scope. [E32; V6] |

**Current direction after E32:** retain the immutable integration reference,
its CPU/MLX FP32 qualifications, J0/J1 controls, and all historical adapter and
selection identities. The SNLI target-source comparison is completed, not an
untried retention fix. Pause generic short-premise replay variants as the main
response to long-document failure. First distinguish supported-to-none and
supported-to-contradiction errors, hypothesis-family concentration, and evidence
location using admissible training/development records; then specify one bounded
preservation treatment and a task-appropriate retention panel. This is a proposed
diagnosis, not proof of a cause or permission to train on inspected gate errors.
Do not reselect historical checkpoints or relax constraints after reading these
results. An immutable-profile MLX performance study remains separate systems work.
[E17–E32; V6; §17.3; §18.26; proposed program]

The exploratory 2I/2J screen already compared frozen heads, limited LoRA,
Qwen3.5-2B, and ModernBERT. Its 4B selection is an integration reference, not a
mandatory deployment model. After useful quality is established, 2B adaptation
is the first smaller-model challenger unless measured question-continuation
cost justifies a shared-query reader instead. Avoid a new architecture tournament
before the task and evidence contract can distinguish causes. [E11; §§15, 18.19]

Preserve strict FP32, lossless reuse, and bounded FP16-KV storage as execution references; keep TF32 as a separately versioned performance candidate and the tested low-bit configurations outside the accepted-equivalence set. Retain inspected G examples as regression data, not an untouched final test after tuning. When attributing an effect, do not change prompts, kernels, precision, heads, policies, and caching simultaneously. None of the revised priorities retroactively changes the F/G verdicts. [E3–E7; recommendation]

**Workbench status now has two preserved stages.** The first `2ij.1.0` invocation remains a blocked review-gated checkpoint with no semantic training/final results; §14 preserves its mechanics evidence and blockers. The later `2ij.2.0` exploratory screen completed 13 fit jobs, evaluated 31 locked final profiles, measured bounded state-first sharing and exported the selected reference bundle. It does not erase the earlier review gate or convert the pilot into release-quality evidence. [E10; E11; I0; V3]

# 2. What “Jev-style” should mean

## 2.1 A software-facing contract, not a claim about hidden internals

TypeSafe documents Jev as a model that accepts a shared state and named typed questions, then returns structured decisions and probability distributions without free-text generation. Its published interface has three primitives: Choice, Score, and Noul. The documentation describes questions as evaluated in parallel and in isolation against the same state. These are the relevant behavioral targets for OpenKind. They are not a public specification of Jev’s complete neural topology. [P1]

Choice returns a selected member of a supplied option set, probabilities, and a confidence statistic. The documentation reviewed for version 0.1 permits up to 255 options. Score returns a distribution over described levels and a probability-weighted mean of their level numbers; equal means can conceal different distributions. Noul returns the probability of a yes answer, without a separate confidence value. A Noul is not the project’s “none” class. [P2–P4]

TypeSafe describes confidence as a statistic derived from the returned distribution. It should not be silently equated with the maximum class probability or with a calibrated probability that an entire workflow is correct. OpenKind should expose the full distribution and explicitly version any additional concentration statistic. Matching field names while changing the statistic’s meaning would create a misleading compatibility claim. [P5; design implication]

TypeSafe publicly identifies a new architecture, a parallel sampler, and Reinforcement Learning for Calibrated Decisions, or RLCD. The primary pages reviewed describe the objective and behavior, not enough algorithmic detail to establish that OpenKind reproduces the training procedure. The appropriate claim is an independently designed, open-weight implementation of a similar decision interface. [P6; P7]

## 2.2 Five properties that must be tested separately

**Structural validity** means every successful response obeys the requested types and option membership. Host-side construction can provide this without letting a model invent keys or serialize JSON. It says nothing about whether the selected answer is true.

**Decision quality** measures whether the predictions resolve the intended task. This requires labeled examples that represent the target use, not only a demonstration with a few appealing outputs.

**Calibration** asks whether stated probabilities correspond to observed frequencies within a defined evaluation population. A model can be accurate but overconfident, or have a low aggregate calibration error while being nearly useless at discrimination. Phase 2B’s training-prior baseline shows the latter: near-chance accuracy accompanied a low ECE. [E1]

**Isolation and consistency** cover several contracts. Renaming opaque keys should not change tokenized meaning. Candidate permutation should permute results appropriately. Unrelated questions should not affect another question’s answer. Approved execution strategies should stay within defined numerical and policy tolerances. A change in candidate-set composition can legitimately change a Choice distribution because the alternatives have changed.

**Efficiency** must include complete requests, not only the final classification layer. Tokenization, prefill, candidate suffixes, cache copies, transfers, probability computation, and serialization all contribute. Existing measurements cover selected single-process GPU workloads; they do not establish network, concurrency, queueing, or cold-model service latency. [E3–E5]

## 2.3 Relation to existing open work

Public projects provide useful components but do not settle all five properties. AlexWortega/openjev documents a Qwen3.5-4B NLI checkpoint with three classes and plain cross-entropy, and now also describes larger-backbone latent-head experiments. Its existence supports the practicality of the NLI approach, not proprietary Jev equivalence. The LFM2.5-2.6B-RLCD model card explicitly calls its constrained-decoding implementation inference-only and uncalibrated, and says it does not reproduce TypeSafe’s RLCD. [P12; P13]

The closest research references answer different questions. SALSA supplies a strong finite-token classification baseline using relevant output-token logits and parameter-efficient fine-tuning. GLiClass investigates efficient classification with dynamically specified labels. Perceiver IO supplies a general architectural precedent for querying shared representations to produce structured outputs. None of these papers identifies Jev’s internal implementation. They are comparison points and design options, not interchangeable solutions. [P9–P11]

Prior project discussions also considered community novelty and attribution claims. This paper does not convert discussion-level accusations into findings. Establishing conceptual overlap, implementing a similar interface, reproducing an experiment, and establishing provenance misconduct are separate tasks. The present evidence supports engineering comparisons, not an attribution verdict.

## 2.4 Generality is about question meaning, not only option count

The intended scope is many well-defined judgments, not unrestricted long-form reasoning. TypeSafe’s public guidance favors atomic questions composed in application code. OpenKind should therefore test whether changing the instruction changes the distinction being evaluated, while unrelated questions remain isolated. Supplying unseen intent names is useful evidence, but is not the same as following an unseen question or rubric. [P1; R3]

Use Q for independent questions and K for the alternatives within one question. Larger K does not show larger Q. Likewise, a batch of Q complete inputs may be one framework call while still encoding the state Q times; true state sharing requires a separately measured computation graph. The new research target joins three properties that the existing evidence establishes only separately or incompletely: useful question-conditioned judgments, controlled uncertainty/rejection, and efficient multi-question execution. “Generalist” remains a hypothesis to test on held-out task and rubric families, not a new claim about the existing checkpoint. [E2–E7; R3; proposed definition]

# 3. Evidence base and experimental method

## 3.1 Research chronology

| Stage | Main question | Evidence status |
|---|---|---|
| Initial Phase 2 probe | Can Qwen features feed a small decision head? | Executed feasibility demonstration; two training examples, not a benchmark. |
| Phase 2B | Which frozen-feature pooling/head baseline works, and what does it cost? | Completed NLI benchmark, timing, export, and architectural audit. |
| Phase 2C | Does the result survive new samples and seeds; can labels be dynamic? | Completed NLI, stability, and Banking77 candidate experiments; LoRA disabled. |
| Phase 2D | What causes execution drift; how should missing options be modeled? | Completed precision diagnostics, none-head comparison, and request timing. |
| Phase 2E | Can selective precision, prefix reuse, and acceptance policies help? | Partial systems run; policy evaluation completed. Full FP32 stage skipped by memory guard. |
| Expanded Phase 2E | Can isolated workers validate full FP32 and batched prefix reuse? | Completed FP32 and BF16 execution; strategy acceptance differs by precision. |
| Phase 2F | Do compression and cross-request reuse improve real serving trade-offs? | Completed both workers, six codecs, cold requests, persistent-cache traces, and long-prefix mechanics. Acceptance remains strategy-specific. |
| Phase 2G | Do the reference paths transfer to fresh labeled inputs; can TF32 and cache expiry help? | Completed strict-FP32 and TF32-permitted workers, fresh Banking/CLINC/OOS panels, context controls, and expiry traces. Semantic quality and equivalence remain separate. |
| Phase 2H, original attempt | Criteria/rejection and primitive fitting | Original `20260919T040612625670Z`, v2h.1.1: saved fits/development, then reporting failure; preserved unchanged. [E8] |
| Phase 2H, completed continuation | Finish the locked original evaluation without retraining | `20260919T040612625670Z__finish_2h_1_2`, v2h.1.2: both evaluation workers completed; final quality, policy, parity, robustness, primitive and request results saved. Optional state-first/2B/LoRA remain disabled. [E9] |
| Phase 2I/2J workbench checkpoint | Prepare reviewed multi-question comparisons and test state-first mechanics | `2ij_reviewed_multiquestion_v1`, version `2ij.1.0`: preparation and synthetic GPU probe completed; reviewed-study gate blocked; no new semantic training/final results. [E10; I0] |
| Phase 2I/2J exploratory screen | Compare state-first/instruction-first Qwen, smaller/adapted Qwen and compact controls; export an integration target | `2ij_model_selection_screen_v2`, version `2ij.2.0`: completed exploratory scope; 13 fit jobs, 31 final profiles, selected/exported profile `a047d6802c3f06f085b8`. [E11] |
| Phase 3A Python systems reference | Validate complete hybrid-state branching, breadth-first Q/K batching, high-K mechanics, repeatability and crossover behavior without changing the model | Run `20260920T024056Z`: completed notebook scope; semantic/high-K parity passed; workload-dependent latency and memory measured; no Rust/Metal claim. [E12] |
| Phase 4A.0 corpus and feature lock | Can natural documents support several questions per shared state under an auditable split and cache contract? | Run `20260921T013558Z`: 2,192 states and 15,368 questions from ContractNLI/QASPER locked before model predictions; group-safe partitions and BF16 state features saved; final labels/predictions unopened. [E14] |
| Phase 4A.1 original StateQuery B1 | Does a trained question-conditioned readout improve the historical candidate-conditioned reference? | Completed non-final training/evaluation and model lock. Aggregate metrics improve strongly, but QASPER collapses to majority-answerable behavior with 0.0 semantic-none recall; promotion gates are unset and final remains closed. [E14] |
| Phase 4A.2 matched comparison | Do source balancing, semantic-none weighting, and matched B0/B1/B2/B2R contracts isolate the architecture and repair rejection? | Completed B0, balanced B1, factorized B2, refined B2R, and B2 QASPER-weight-8 arms. B2 weight 8 has the strongest source-macro balanced accuracy/F1 (0.5872/0.5905), but QASPER none recall is only 0.1667; every arm misses the 0.30 development/gate objective. [E15] |
| Phase 4B.2 scalar-weight boundary | Does increasing the QASPER semantic-none cap beyond 8 repair applicability without unacceptable source transfer? | Completed seed-17 cap-12 request. The effective weight saturates at 8.5602, QASPER gate none recall is 0.1556, and ContractNLI false-none is 0.3966. Post-hoc threshold analysis confirms that meeting the recall floor would require excessive false-none. Scalar tuning is closed; final remains unopened. [E16] |
| M1 / M2.1 frozen joint-option diagnostic | Separate source/input visibility from Base/post-trained and input-budget effects | `v0.3.2`, session `20260925T174531_575346Z`: 2,964 FP32 decisions, two checkpoints × two budgets, previously exposed non-final panel; no training or promotion. [E29] |
| M2.2 matched decision-LoRA pilot | Improve ContractNLI contradiction/unsupported decisions while retaining QASPER | `v0.4.1`, session `20260925T232651_880064Z`: both fits finish 120 updates and select 80; all 3,964 BF16 model decisions complete; targeted in-domain gains, failed class/cross-task retention; no final opening or promotion. [E30] |
| Retention-aware replay follow-on | Can fixed train-only replay with parent KL preserve targeted gains and other capabilities? | v0.5.0 completed; J4/J5 each fit 120 updates, both select frozen0; fixed80 probabilities improve versus specialists but joint retention fails. [E31] |
| Source-label replay target comparison | Does source-label CE on the same replay inputs improve retained correctness relative to parent-only KL? | v0.6.0 completed, session `20260926T170047_182185Z`: both fits complete 120 updates and select frozen0; fixed80 SNLI gains do not preserve ContractNLI entailment or QASPER; no promotion. [E32; N1] |

The historical E1–E7 measured sequence uses Qwen/Qwen3.5-4B-Base at revision `1001bb4d826a52d1f399e183466143f4da7b741b`. The text backbone has 4,205,751,296 parameters, hidden width 2,560, and 32 blocks. Its layer list contains 24 linear-attention and eight full-attention blocks. The core results were obtained on an NVIDIA L4. The saved environment includes Transformers 5.17.0; the expanded workers record PyTorch 2.11.0+cu128. Environment details should travel with results because kernel and precision behavior matter. [E1; E2; E5]

The notebook execution logs also report missing optimized causal-convolution and linear-attention kernels, with reference implementations used instead. Transformers documents these optimized versus reference paths. The recorded timings should therefore be treated as measurements of this particular stack, not the speed limit of Qwen on an L4. Installing faster kernels is a future experiment requiring both new timing and renewed probability/policy parity checks. [E5; E6; P16]

## 3.2 What was reviewed and what was not rerun

**Version 0.8.5 boundary:** this revision uses the supplied v0.8.4 paper and E32
result/contract/selection snapshots, and reruns the supplied v0.6.0 saved-output
review on copies. Its 5,168 granular bookkeeping assertions pass, including all
12 result-manifest hashes/sizes, 7,410 primary rows, 120 semantic-count rows,
300 class rows, identity-policy arithmetic, both declared selections and both
120-update logs. SNLI softmax, argmax, confidence, NLL/Brier and conditional ranking
are reconstructed from 1,920 exported offered-outcome logit rows; the main benchmark
CSV lacks full logits, so its NLL/Brier and all bootstrap intervals remain
source-reported. Sixteen bundled source files match the recorded runtime hashes;
that is not a separate runtime-source download. No model execution, calibration
refit, threshold reselection, tensor-binary audit, label adjudication, original
source rerender, or protected-final access occurs. Earlier numerical/native audits
are not rerun. The document diff and preservation checks are separate from the
saved-output assertions; neither count measures scientific replications. [E32; V6]

**Version 0.8.4 boundary:** this revision reads E31's completed result, selection,
data-contract, and review snapshots and reruns its supplied saved-output checker
on copies (452 checks). It does not rerun E29/E30 numerical audits or E31 model
computation. NLL/Brier, calibration, and interval values remain source-reported
where raw logits were not reconstructed. New notebook tests are separately
identified under N1 and do not supply new pretrained-model results. [V5]

Earlier versions synthesized executed notebook snapshots, full result JSONs for Phases 2B–2F, summary documents, and relevant embedded source. Version 0.2 added the completed Phase 2F archive and matched its uploaded paste-back summary to Drive. That earlier audit reconstructed 1,664 Phase 2F probability/action comparisons, checked 384 storage records and 922 timing medians, and replayed 12 LRU traces under zero-expiry conditions; the expanded 2E outputs had also been re-aggregated. These historical audit scopes are retained rather than presented as new inference. [E5; E6]

Version 0.3 also read the completed Phase 2G archive and the two prior review companions. Its independent saved-output check reconstructed 6,960 distributions and 20,880 application-policy actions from retained candidate scores and coefficients, with maximum distribution reconstruction residual approximately 1.22 × 10⁻¹⁵. That v0.3 audit rechecked 24 full fresh-quality head summaries, 768 within-mode comparisons, 416 fresh and 72 controlled-context cross-mode comparisons, 16 request-time aggregates, and 12 trace totals. The prior G review separately checked broader summary records and replayed eight scalar LRU traces. No Qwen inference, training, GPU timing, annotation adjudication, or full external-reference re-review was performed for v0.3. Arithmetic agreement does not independently reproduce the backbone or validate author labels. [E7; R1; R2; v0.3 verification companion]

The two supplied ChatGPT share links resolved to their conversation titles but did not expose readable conversation bodies through the available web reader. Earlier project reports in the Library supplied additional framing; executable notebook code and saved result artifacts take priority over those reports. The source register records this limitation rather than implying that the shared transcripts were fully inspected. [S1; S2]

For v0.4, the supplied v0.3 Markdown was the authority for retained numerical results. The saved architecture-review note R3 was read in full, and the H0 notebook's methods, configuration defaults, and treatment descriptions were inspected. That Library snapshot had no executed code cells; it did not contradict the owner's report of a separate in-progress run. No H archive or live run was inspected for v0.4. S3 exposed its title only; R3 recovered the main review rather than every subsequent turn. Available conversation history supplied the later Laya/R4T recommendations, with narrow primary-source checks supporting Section 11.7. Those recommendations remain proposals, not OpenKind measurements. [R3; H0; U1; S3; P18–P20; historical v0.4 scope]

The v0.4 integrity check compared retained tables and measured sections against its supplied source; it did not recompute E1–E7 experiment outputs. Earlier archive hashes and audit reports remain historical provenance, not work repeated for v0.5. [v0.4 revision manifest]

For v0.5, the attached v0.4 is the editing authority. The new H paste matches the archived compact summary after normalizing Markdown escapes and automatic URL wrapping. The audit reconciles 14 profile records, the restricted six-profile joint selection, 12 main-profile temperature-gate decisions, ten none-model selections, 126 policy-selection records and six primitive-seed records. It checks 37 archived implementation hashes and all 15 pairwise exact-group intersections among the six main split manifests. Two saved development fixtures are independently reconstructed using the exported head and none coefficients: maximum candidate-score residual is approximately 5.12 × 10⁻⁷ and probability residual approximately 4.70 × 10⁻⁹ under the audit's float64 arithmetic. These are limited saved-output/head-algebra checks, not Qwen inference or reproduction of all H development losses. The fitting feature/logit cache is absent from the ZIP. [E8; v0.5 audit companion]

For the original E8 snapshot, the traceback and archived worker source located the exception after fitting artifacts were written, during the final memory snapshot. That attempt contained no global final lock, evaluation-worker directory or final prediction file. The v0.5 audit did not inspect reserved final stimuli, repair or rerun the notebook, modify reservations or re-audit B–G. Those historical findings remain correct for E8; the separately completed E9 continuation now supplies the formerly missing outputs. [E8; E9]

For v0.5.1, the Library v0.5 paper and the uploaded roadmap are the editing bases. The saved Colab finish-cell summary is equal to the archive's compact summary; the separately retrieved full summary is byte-identical to the archive copy. A read-only checker reconciles all 14 profile development values and the six-profile restricted selection, verifies matching worker/profile copies, six primitive weight files and 126 nonfinal policy records, checks all 37 archived implementation hashes, and checks the retained inventory for final-lock/evaluation outputs. All 22 source-consistency checks pass; **the original attempt's status remained partial at that revision**. This is not an independent reconstruction of development loss or a repeat of the prior v0.5 fixture/overlap audit. No reserved final stimuli or unavailable live Colab runtime were inspected. [E8; R4; R5; V1]

Version 0.1 used a Phase 2F notebook snapshot retrieved around 22:59 UTC on 18 September, while the run was still executing. Version 0.2 supersedes that status with the completed run `20260918T224427722898Z`, whose result files were saved to Drive at approximately 23:56 UTC. Both precision workers and all requested stages completed. Completion does not imply that every codec passed its numerical or policy gates. The old notebook remains historical provenance; the completed archive is the Phase 2F result authority. [E6]

For v0.6, E9's standalone full and compact summaries match the corresponding archive members byte-for-byte. The original ZIP, locked fitting files and payloads retain their identities. V2 independently re-aggregates **32,256 final probability rows into 168 final quality records and 1,512 final policy records**, reproduces 26 paired final contrasts and the accuracy-bootstrap calculations, checks 216 within-mode parity rows and 14 cross-mode profile comparisons, and checks 1,024 reliability rows, 20 reliability quality records, 180 reliability policy records and 24 context-position records. It recomputes 12 primitive-seed metric records from 1,536 saved distributions and checks 80 benchmark-row medians from 240 saved timings. Policy interval values remain source-reported. The largest checked metric residual is about 2.22 × 10⁻¹⁶. These counts include repeated profiles, arithmetic modes and variants of shared source messages; they are not independent sample counts. [E9; V2]

V2 also verifies 39 implementation hashes, 37 locked fitting/runtime-source files, three locked payload hashes and 46 evaluation-output inventory hashes. The 24 locked fitting-directory files match the nested original attempt exactly. The completed final payload is read only to verify the already-evaluated IDs/labels and aggregate primitive results; source annotations are not re-adjudicated. Additional K/condition and reliability cross-mode tables are labeled saved-row aggregations. No Qwen inference, training, new timing, native-backend execution, complete security audit or historical B–G recomputation occurs. These checks support the reported arithmetic and lineage, not universal calibration or operational safety. [E9; V2]

For v0.6.1, E10's standalone compact summary matches its report-archive member byte-for-byte. V3 records **34 saved-artifact consistency checks**: status/gate/job agreement, recorded feature-threshold arithmetic, runtime-source identity, cache-family and exclusion inventories, three recorded model revision strings, unsigned/stale review metadata and the canonical protocol-body hash. It verifies 18 report files against the separately retrieved snapshot manifest; the SQLite backup and CPU-test XML are not materialized. The source reports 48 passed CPU tests, but this revision does not rerun them. The raw hidden vectors are absent from the compact archive, so the feature deltas remain source-reported. [E10; I0; V3]

The intake is reviewed as metadata only: protocol, unsigned review and refreshed manifest. The 92 case IDs are counted from their hash inventory; `cases.jsonl`, final questions and their annotations are not opened. The duplicate-question coverage finding comes from static inspection of the archived generator/call site. No model code is imported, no new GPU timing or H numerical reconstruction is performed, and no approval or persistent file is changed. The distinction between a prepared draft and a reviewed dataset remains intact. [I0; V3]

For v0.8.1, the supplied v0.8.0 Markdown (SHA-256 `6ddb149e7c3229188e60a97399533e0cff1cc5a2e94ecff669c6cca49a0e6201`) is the editing authority. The Phase 4A record was reconstructed directly from the Drive-resident corpus manifest/lock, feature manifest, training reports, non-final evaluations, model/result locks, experiment contracts, comparison snapshot, and B1 progress pointers. Reported metrics are read from those JSON artifacts; raw predictions were not re-scored and Qwen was not run. No source label was adjudicated, no notebook or Drive result was modified, and no final label or prediction was opened. The balanced B1 entry is explicitly a live checkpoint observed at `2026-09-21T21:39:24Z`, not a completed training or evaluation result. [E14; E15]

For v0.8.2, the supplied v0.8.1 Markdown (SHA-256 `801ac783c4ef07d828d703f2658d7c3576e50e5b4d9adb73ef94a9a4467dd590`) is the editing authority. Later completed artifacts supersede only v0.8.1's progress statuses: balanced B1 and every Phase 4A–4D follow-on now have their recorded terminal reports/locks. This revision reads and reconciles those saved artifacts, but does not rerun training, modify result directories, adjudicate labels, or open final. [E15 amendment; E17–E20]

For v0.8.3, the uploaded `WHITEPAPER.md` (SHA-256 `9579909e53dff0d0fd404632f763b416df78c05285ab600cabdc539c05b4656f`) is the editing authority. E29 and E30 are the completed frozen and matched-adaptation result authorities. This update reruns the included CPU-only readout scripts on local copies: six E29 result hashes and nine E30 result hashes match; seven E30 runtime-source hashes and the canonical data-lock identity match. Count, class, identity-policy, component separation, schedule, and selection checks are recorded in V4. The additional document-bootstrap accuracy intervals and exact policy-cost interpretation are review-derived, distinguished from source-reported probability scores and original intervals. No Qwen inference, fitting, raw-logit recalibration, new source adjudication, checkpoint-binary audit, repository/Drive mutation, or protected-final access is performed. Historical E0–E28 and native result records are retained rather than remeasured. [E29; E30; V4]

## 3.3 Evaluation units and leakage boundaries

MultiNLI splits were separated using normalized premise groups, not only individual rows. Phase 2C excluded 4,021 prior groups from its newly selected splits and used 2,400 training rows, 300 development rows, 1,000 calibration-fit rows, 500 calibration-gate rows, and 1,000 rows in each new test. Training examples remained the saved Phase 2B training set rather than a larger corpus. [E2]

Banking77 results require a different denominator. Several candidate-set episodes can come from one underlying message. Phase 2C’s 480 episodes per test split came from 160 messages; Phase 2E’s 1,024 episodes per split came from 64 messages. Expanded 2E and Phase 2F replay the same eight messages in a 128-episode factorial panel. Phase 2F selects 32 episodes for codec regression and 16 for short-request benchmarking; these subsets still span only eight messages. Its repeated-request traces do not add independent examples. Confidence intervals and uncertainty discussions must respect message clustering. [E2; E4–E6]

The 57/20 Banking label split withholds labels from candidate-head training, development selection, and calibration. It does not withhold them from Qwen’s pretraining. Those two label groups remain within banking; Phase 2G separately adds CLINC non-financial and author-OOS transfer panels. “Fresh” means new relative to the named recorded project-message manifests, not necessarily novel knowledge to the backbone. Repeating the same seeds reuses the same selected examples. [E2–E4; E7]

Phase 2G excludes 2,912 normalized prior Banking message hashes and constructs 416 fresh episodes from 112 messages. Its 56 parity episodes come from 16 messages; 72 controlled-context episodes transform six selected messages; eight episodes are timed independently; repeated 48-request traces supply no new semantic observations. Those units are kept distinct throughout Section 10. Once these G examples have been inspected to choose the next modeling intervention, they belong to historical/regression evidence rather than another untouched final evaluation. [E7; R2; methodological implication]

E29/E30 use the same previously exposed 741-question, 72-component natural-document panel. Its calibration gate has 204 ContractNLI questions from 12 contracts and 46 QASPER questions from 12 papers. E30's 128-component training pool, 12-component development pool, and evaluation components are recorded as disjoint. The full 960-exposure schedule visits 734 unique training questions; the selected update-80 checkpoint has seen 640 exposures and 538 unique questions. Repeated training exposures and code/order variants are not independent evidence. QASPER is held out of LoRA and checkpoint selection, not out of its existing calibration-fit/policy-development roles. Source `test` identifiers already assigned to the OpenKind exposed calibration gate are not the protected OpenKind final partition. [E29; E30; V4]

# 4. Qwen as a decision backbone

## 4.1 The initial probe and its corrected interpretation

The first probe established basic plumbing: hidden states could be extracted, last/mean/max pooling produced 2,560-dimensional vectors, a small head could fit two security-flavored examples, and a decision path could run without generating an answer. Its approximately 7.48-second generation observation versus 0.262-second head-path observation was exploratory, not an equal-quality benchmark. Two fitted examples cannot show classification generalization. [E0]

The probe also exposed a misleading model comparison. `AutoModel` loaded a multimodal wrapper with a vision tower, producing a count of 4.539 billion parameters, while the text causal-LM load reported 4.206 billion. That does not show a causal LM becoming larger or smaller because of its output head; the loaded module sets were different. Phase 2B corrected this by reporting the exact text backbone, whether vision was loaded, and whether the vocabulary matrix was tied. [E0; E1]

## 4.2 Removing generation is not removing the backbone

The logical vocabulary projection contains 248,320 × 2,560 = 635,699,200 weights. In the measured checkpoint, it shares storage with the input embedding. Omitting its use on the decision path therefore has zero marginal removable parameters unless the embedding scheme is also changed. The important saving is avoiding vocabulary projection and repeated answer-token decoding, not deleting most of the model. [E1]

This distinction prevents a misleading memory claim. Expanded 2E recorded approximately 8,021.9 MiB allocated after directly loading BF16 weights and 16,043.7 MiB after directly loading FP32 weights. These are process tensor-allocation observations, not total service memory or a device-independent requirement. Cache storage, transient activations, allocator reservations, runtime overhead, and concurrency remain additional costs. [E5]

## 4.3 Fixed NLI head

For the fixed three-class experiment, the decision computation is conceptually:

```text
h = final hidden vector at the last non-padding input token
u = (h − training_mean) / training_std
z = W u + b
p = softmax(z / T)
```

The export preserves normalization, label ordering, tokenization and pooling rules, and selected temperature. A different pooling index, label permutation, or normalization convention changes the function even if the matrix multiplication is otherwise correct. Last-token extraction must follow the mask contract; “last array position” is not a portable substitute when padding is present. [E2]

The strongest conclusion is specific: last-token features with a small head worked well under the tested causal prompt representation. This is not a theorem that last-token pooling always dominates. Mean pooling mixes positions that have seen different amounts of context, and the comparison also depends on the prompt and head capacity. The causal explanation is plausible; the experiment establishes the empirical ranking, not that explanation as the sole cause. [E1; interpretation]

## 4.4 Fixed-head results and calibration

| Evaluation | Accuracy | NLL | ECE, 15 bins | Temperature |
|---|---:|---:|---:|---:|
| 2B matched, raw | 87.67% | 0.3345 | 0.0298 | 1.0000 |
| 2B matched, temperature-scaled | 87.67% | 0.3392 | 0.0380 | 1.1441 |
| 2B mismatched, temperature-scaled | 87.33% | 0.3233 | 0.0398 | 1.1441 |
| 2C fresh matched, gate-selected | 87.00% | 0.3400 | 0.0194 | 1.0000 |
| 2C fresh mismatched, gate-selected | 88.80% | 0.3246 | 0.0247 | 1.0000 |

Source: selected-head records in E1 and E2. The tests differ between phases; the rows are not a controlled estimate of improvement from 2B to 2C. Phase 2B’s matched accuracy interval was 85.19–90.22% under premise-cluster bootstrap.

In 2B, last-linear accuracy was 87.67% matched versus 74.67% for mean-linear and 84.83% for max-linear. An MLP or attention-pooling alternative was not automatically better under the development selection rule. The untuned finite-code baseline achieved 78.83% matched and 81.50% mismatched, making it a more relevant comparator than an arbitrary long generated response. Phase 2C’s mean development NLL was 0.35297 for linear and 0.35472 for MLP across its configured seeds; the small difference does not justify a sweeping architecture claim. [E1; E2]

Temperature scaling preserved the winning class but worsened Phase 2B’s matched test NLL and ECE. Phase 2C added separate calibration-fit and calibration-gate samples; its NLI gate kept T = 1 because the fitted temperature did not improve the gate criterion. Phase 2D similarly rejected a fitted temperature for the none-head experiment. Calibration methods need held-out acceptance tests, not automatic application. The broader calibration literature motivates temperature scaling as a useful baseline, not a guarantee under every shift. [E1–E3; P8]

# 5. Dynamic Choice and the missing-option problem

## 5.1 A shared scorer instead of a fixed output vocabulary

Phase 2C scores each supplied candidate description against the message and instruction. The backbone is frozen; the final candidate-conditioned token is standardized, then mapped to a scalar using a shared affine scorer. A learned global none logit is appended before normalization. The candidate component has 2,562 trained parameters including its bias and the none scalar. Opaque candidate IDs do not enter model input. [E2]

```text
s_j = wᵀ standardize(backbone(message, instruction, candidate_j)) + b
p_j = exp(s_j / T) / [exp(s_none / T) + Σ_k exp(s_k / T)]
```

This permits unseen candidate labels without adding a new output neuron for every possible label. It does not make one forward pass independent of candidate count: the original implementation re-encoded the message for each candidate. Later cache work amortized common prefix computation while retaining candidate-conditioned suffix processing. [E2; E5]

## 5.2 What the Banking77 transfer test established

The candidate head trained on 600 message episodes with K ∈ {2, 4, 8}, using uniformly sampled distractors and an approximately 25% missing-intent construction. Head-fitting labels numbered 57; 20 labels were reserved. Evaluation used 480 episodes over 160 messages in each of the seen and held-out label groups. These are sampled-choice tests, not the standard 77-way Banking77 classification benchmark. [E2]

| Phase 2C metric | Seen labels | Held-out labels |
|---|---:|---:|
| Overall choice/none accuracy | 88.54% | 82.29% |
| Accuracy on answerable episodes | 95.16% | 93.01% |
| Recall when the annotated intent is absent | 65.74% | 45.37% |
| False-none rate on answerable episodes | 1.34% | 3.23% |
| Selected-temperature NLL | 0.4000 | 0.5522 |

Source: E2, dynamic evaluations. The saved metric name `false_abstention_rate` refers here to predicting semantic none on an answerable episode, not a separately selected human-review policy.

Candidate count matters. Seen-label none recall fell from 91.67% at K = 2 to 44.44% at K = 8. With fixed existing logits and positive temperature, adding a candidate lowers the probability assigned to a constant none logit through the softmax denominator. However, selecting none by argmax depends on whether its logit exceeds every candidate logit, subject to the implementation’s tie rule. It does not depend on whether the logit exceeds the sum of their exponentials. [E2; R1; algebraic distinction]

```text
P(none) = exp(b / T) / [exp(b / T) + Σ_j exp(s_j / T)]
none wins by argmax when b > max_j s_j, apart from the declared tie rule
```

Adding a weak alternative can dilute none probability without changing the selected outcome. Adding more alternatives also creates more opportunities for a distractor to outrank none. Probability dilution and argmax errors are distinct effects; these experiments do not establish either as the sole cause of the recall decline. [R1; algebraic interpretation]

The small reliability panel found exact unbatched order agreement but a maximum batched probability change of 0.01867 on 24 episodes. A paraphrased instruction changed one prediction out of 24 and moved the panel’s accuracy from 95.83% to 91.67%. These checks show where invariance needs testing; the samples are too small to establish general reliability rates. [E2]

## 5.3 Phase 2D: learn the evidence for none from the set

Phase 2D froze the Qwen backbone and candidate scorer and compared the original global none value, a refitted global value, a count-adjusted model, and a set-aware linear model. The selected set-aware model uses six symmetric summaries plus an intercept: maximum candidate score, top-two gap, mean, population standard deviation, log-mean-exp, and log K. It therefore has seven fitted parameters. Its inputs depend on the set rather than the arbitrary order of candidates. [E3; embedded decision-head source]

The experiment added lexical hard distractors, evaluated candidate counts through 16, and paired present/absent episodes. Development selection minimized message-weighted NLL under an explicitly assumed 25% absent prior. The raw paired test is 50% absent. These population choices must not be conflated. [E3]

| Phase 2D paired stress result | Seen: original none | Seen: set-linear | Held-out: original none | Held-out: set-linear |
|---|---:|---:|---:|---:|
| Accuracy | 64.44% | 83.94% | 54.56% | 74.56% |
| None recall | 35.63% | 79.38% | 26.00% | 73.13% |
| False-none rate when answerable | 1.25% | 7.50% | 3.50% | 18.00% |
| Raw NLL, 50% absent stress set | 1.2349 | 0.5471 | 1.3627 | 0.6952 |
| Weighted NLL, assumed 25% absent | 0.7420 | 0.4551 | 0.9214 | 0.6836 |

Source: E3, evaluation.test_seen/test_unseen. Each split contains 1,600 correlated episodes from 100 messages. This is a within-2D comparison, not a claim that 2D accuracy can be directly compared with the easier 2C setup.

The one-parameter refitted-constant ablation must also remain visible:

| Phase 2D none model | Coefficients newly fitted in 2D | Seen-label accuracy | Held-out-label accuracy |
|---|---:|---:|---:|
| Original frozen global value | 0 | 64.44% | 54.56% |
| Refitted global value | 1 | 83.19% | 72.56% |
| Set-conditioned linear | 7 | 83.94% | 74.56% |

Source: E3, complete evaluation records; R1. Most of the accuracy gain over the original none model is already obtained by refitting one constant on the new training mixture. The set-conditioned model adds 0.75 percentage points on seen labels and 2.00 points on held-out labels relative to that refitted constant. It was selected on development NLL, not retrospectively on these test accuracies. Do not attribute the entire original-versus-set-linear improvement solely to candidate-set conditioning.

The gain is real within the experiment, but it is not free. The held-out-label false-none rate rises to 18%, so the system becomes much better at detecting omitted intents while turning away more cases it could answer. Which trade-off is useful depends on costs and deployment prevalence. A single “accuracy improved” headline would conceal that decision. [E3]

## 5.4 Three different meanings of not answering

**Semantic none:** the supplied alternatives omit the annotated correct intent. This is what the C/D experiments train and evaluate.

**Review or abstention policy:** the application declines to automate because the expected cost or uncertainty exceeds its threshold. It may do this even when one listed candidate is actually correct.

**Unknown-domain or insufficient-evidence detection:** the system encounters a task outside its coverage, contradictory information, or information insufficient to decide. The Banking77 omission construction does not establish this capability. Phase 2G adds a bounded author-labeled OOS test, reported separately, with only 46.88% none recall for the set-linear reference; it is not a general validation of OOS or insufficient-evidence handling. [E7]

Keep these concepts distinct in datasets, probabilities, telemetry, and API schemas. A review action should not be inserted into a model’s semantic probability vector as though it were another fact about the message. Conversely, an explicit none label should not automatically count as a successful safety refusal. [E2–E4; design recommendation]

# 6. Calibration becomes useful through decision policy

## 6.1 Probability quality and automation utility differ

Phase 2E froze the probability models and selected an acceptance policy on fresh development messages. It varied assumed missing-intent prevalence over 5%, 25%, and 50%, and wrong-answer cost over 1, 5, and 20, with review cost fixed at 0.1 and correct-answer cost zero. The “prior” scenarios reweight evaluation and selection; they are not measurements of deployment prevalence or proof that the probabilities are recalibrated for each population. [E4]

For an ideal calibrated probability p that the proposed action is correct, a simple one-step loss comparison gives:

```text
expected loss of answering = wrong_cost × (1 − p)
expected loss of review    = review_cost
answer only when p > 1 − review_cost / wrong_cost
```

This is a derivation under stated assumptions: review has a fixed total cost, correct automation has zero cost, and a wrong answer has one common cost. It is not the exact empirical selection algorithm and is not a production policy prescription. Real review may be delayed, capacity-limited, imperfect, or more costly for some cases.

## 6.2 Measured policy trade-offs

Each new 2E development/test split has 64 messages expanded into 1,024 episodes. No neural or none-head coefficients were refitted in this phase. The chosen head and threshold depend on the declared costs rather than one universally best model. [E4]

| Assumed absent prior / wrong cost | Selected head / threshold | Seen test: answer rate / cost | Held-out test: answer rate / cost |
|---|---|---|---|
| 5% / 1 | Original global / 0.94 | 66.62% / 0.0465 | 60.20% / 0.0517 |
| 5% / 5 | Set-linear / 0.98 | 38.81% / 0.0812 | 33.78% / 0.0667 |
| 5% / 20 | Set-linear / 0.98 | 38.81% / 0.1413 | 33.78% / 0.0682 |
| 25% / 1 | Refitted global / 0.95 | 43.31% / 0.0650 | 36.28% / 0.0647 |
| 25% / 5 | Refitted global / 0.99 | 26.22% / 0.0738 | 22.71% / 0.0773 |
| 25% / 20 | Original global / 1.00 | 0% / 0.1000 | 0% / 0.1000 |

Source: E4, policies.test_results. Rates and costs are scenario-weighted. Always reviewing costs 0.1 under the experiment’s assumptions. At 50% absent prevalence, the cost-5 and cost-20 scenarios also selected all-review behavior.

The 5%-absent, cost-20 example is a useful negative result. The selected policy has a seen-test point cost of 0.1413, worse than reviewing everything at 0.1, even though development selection favored it. Its message-bootstrap interval is wide, approximately 0.0562–0.3687. This does not prove a stable inferiority on all future data; it shows why a small development panel cannot certify rare, costly errors. [E4]

A low measured error among accepted answers also needs a coverage denominator. An all-review policy produces no wrong automated answers because it produces no automated answers. Expanded 2E checks nine policies, including conservative ones; zero changes for a policy that always reviews provide little evidence about fine-grained probability fidelity. Report distribution drift, class changes, answer/review changes, and accepted-candidate changes separately. [E4; E5]

## 6.3 Training implications

Continue using log loss and other proper probability metrics alongside accuracy, Brier score, reliability plots, and cost/coverage curves. A single ECE binning can hide subgroup problems and should not be the sole training or deployment target. Fit calibration only on reserved data and select it on another reserved gate; retain a fresh final evaluation after any training or threshold change. Phase 2C’s gated temperature procedure is a useful pattern. [E1–E4; P8]

Within completed E1–E7, there is no successful LoRA comparison or proprietary-style RLCD training. H0 describes optional matched online-head/LoRA controls, not completed outcomes. The broader matched adaptation comparison is now an early priority rather than something deferred behind another cache-tuning cycle. Use direct supervised probability training first; compare frozen features, limited adaptation, smaller models, and a trained finite-token baseline under common task splits and declared optimization budgets. A model name containing RLCD is not evidence about how it was trained. [E1–E7; H0; R3; P9; P13]

For Score, do not describe an expected ordinal-distance penalty as a proper probability scoring rule. An action-distance objective can favor a median decision rather than recover the full distribution. Start with NLL as the distribution baseline; an ordinal cumulative-probability Brier/ranked-probability objective is a separate candidate, while expected action cost should also be evaluated separately. Any mixture of losses still requires held-out calibration testing. H’s five-level sentiment probe, even when completed, will be evidence for that rubric rather than arbitrary Score semantics. [R3; H0; training recommendation]

# 7. Numerical stability is part of correctness

## 7.1 The behavior is execution-shape dependent

Phase 2B’s two-example batch check was too small to settle stability. Phase 2C expanded it to 200 examples per scenario. Repeating the same isolated call produced no differences, while batching or padding changed probabilities and sometimes crossed decision thresholds. Thus the issue is not simply nondeterministic sampling: the model uses no generated answer sampling on this path. [E1; E2]

Phase 2D used a 32-example diagnostic panel that deliberately included earlier high-drift cases. The maximum observed probability differences across its shape scenarios were:

| Mode | Maximum probability difference | Verdict at tolerance 0.005 |
|---|---:|---|
| BF16 default | 0.067509 | Needs review |
| BF16, math attention | 0.069881 | Needs review |
| BF16, strict math settings | 0.069097 | Needs review |
| FP32, strict math settings | 0.00000727 | Within sampled tolerance |

Source: E3, numerics.shape_summary. This panel is enriched for debugging; it is not a random sample for estimating how often users will see drift.

PyTorch documents that batched and unbatched floating-point computations need not be bitwise identical, and that results can differ across releases and platforms. That explains why exact equality is an inappropriate default expectation; it does not automatically excuse a probability shift large enough to alter an application action. The contract must specify acceptable numerical error and acceptable decision behavior. [P14; E3]

The evidence does not isolate one universal root cause. Switching attention backend or stricter reduction settings did not resolve the BF16 diagnostic. The backbone combines matrix operations, full attention, recurrent linear attention, convolution state, normalization, and nonlinearities. Layer traces and dtype audits are useful localization tools, but no single-kernel repair is proven here. [E3; E4]

## 7.2 Selective FP32 was tested, not merely proposed

The original 2E run tried promoting the first four blocks, DeltaNet modules, and MLP modules to FP32 while retaining a lower-precision model elsewhere. None passed the shape tolerance. Maximum probability differences were approximately 0.07482, 0.06336, and 0.04500 respectively. Promoting MLPs reduced that maximum relative to the BF16-default 2E row but increased measured single-call time from about 83.1 to 110.2 ms. This is not a validated production compromise. [E4]

The full FP32 stage in that run was skipped by a conservative memory guard: 9.40 GiB free versus 12.20 GiB estimated additional requirement. The result is a skipped stage, not a failed FP32 correctness test. Expanded 2E addressed the execution problem by loading one target-precision model directly in each fresh worker rather than converting a live model through an in-process sweep. [E4; E5]

## 7.3 Expanded 2E separates completion from acceptance

The expanded run replayed eight messages across K = 2, 4, 8, and 16; uniform and lexical-hard distractors; and present/absent conditions, producing 128 episodes. Every comparison uses the same frozen heads and nine frozen policies. Within each precision, the reference is full-sequential evaluation. [E5]

| Mode and alternative strategy | Max probability delta | Episodes over 0.005 | Episodes with class change | Episodes with any policy-output change |
|---|---:|---:|---:|---:|
| FP32, full batch four | 0.00000928 | 0 | 0 | 0 |
| FP32, shared-prefix chunk | 0.00000776 | 0 | 0 | 0 |
| FP32, equal-length suffix batch four | 0.00001072 | 0 | 0 | 0 |
| BF16, full batch four | 0.110223 | 78 | 14 | 7 |
| BF16, shared-prefix chunk | 0.085090 | 78 | 10 | 7 |
| BF16, equal-length suffix batch four | 0.085090 | 80 | 14 | 11 |

Source: E5, workers.*.parity_summary. “Class change” records an episode where at least one evaluated probability head changed argmax; policy changes similarly aggregate across the fixed policy set. These are not counts of independent messages or independently adjudicated errors.

A separate cross-precision comparison of FP32 full-sequential and BF16 full-sequential found a maximum probability difference of 0.059015, 18 episodes with a class change, and eight with a policy-output change. This establishes disagreement, not that FP32 is semantically more accurate. FP32 is useful here because its alternative execution strategies closely reproduce its own reference. [E5]

**Engineering consequence:** version precision, kernel implementation, batch strategy, cache algorithm, prompt rendering, and calibration together. A transition that preserves top-1 accuracy on average can still change a rare expensive action at a threshold. Conversely, a tiny probability difference away from every threshold may have no policy consequence. Both numerical and application-level checks are necessary. [E3–E5; recommendation]

# 8. Shared-prefix execution: what works and when

## 8.1 The implemented graph

The current candidate scorer conceptually performs repeated full-prompt work:

```text
instruction + message + candidate A → Qwen → scalar A
instruction + message + candidate B → Qwen → scalar B
instruction + message + candidate C → Qwen → scalar C
                                      ↓
                           none model + softmax + policy
```

The cached path first finalizes each candidate sequence using the original segmented encoder: instruction, state, delimiters, and candidate segments are encoded separately with `add_special_tokens=False`, then their token IDs are concatenated. The cache planner finds the longest exact common token prefix of those finalized sequences and branches from a reusable hybrid cache. Full and cached execution consume identical finalized token IDs, with the correct position offsets and an isolated copy of mutable state for every branch. [E5; E6 embedded encoder/cache source; R1]

Replacing the segmented encoder with whole-string tokenization is a separate prompt-contract change. It must not be introduced silently as a caching fix, even when the visible concatenated text looks identical. This correction describes the implementation already used in the comparisons; it does not invalidate their same-token premise. [R1]

```text
exact common token prefix → prefill → immutable root snapshot
                                           ├─ fork → suffix A → scalar A
                                           ├─ fork → suffix B → scalar B
                                           └─ fork → suffix C → scalar C
                                                        ↓
                                             none + softmax + policy
```

Qwen’s hybrid cache is not only keys and values. The inspected implementation carries full-attention KV tensors, recurrent linear-attention state, and convolution state. Copying only KV, or sharing writable expanded views, would violate the branch-isolation design. The harness checks root integrity and cache storage separation, and expanded 2E tests equal-length suffix batches without suffix padding. [E4–E6]

## 8.2 Candidate reuse is not yet arbitrary-question reuse

A important limitation appears in the saved prompt contract: the instruction precedes the message. The shared prefix therefore belongs to the same instruction-plus-message context, not simply to an abstract state. The completed experiment amortizes candidates within a question. It does not show prefill-once execution across unrelated questions with different instructions. [E2, dynamic.export.prompt_segments; E6, cache planner]

A future state-first rendering could place an identical state prefix before question-specific suffixes. That would create more opportunities for reuse, but it changes the causal representation: the state tokens would no longer be conditioned on the preceding question instruction. It must be treated as a new prompt/model contract, evaluated and potentially trained accordingly. It is not a semantics-preserving cache patch. A shared-encoder/query architecture is another option, also requiring a separate learning and evaluation program. [Design inference; P11]

Similarly, K in these benchmarks is the number of candidate alternatives, not the number of independent Jev questions. The distinction matters when projecting latency, memory, and isolation behavior to a multi-question API.

Phase 2F’s persistent cache extends reuse across requests only when their exact token prefixes match. It does not change the instruction-first prompt contract or show an instruction-independent state representation. The request traces therefore add evidence for persistent prefix reuse, not for arbitrary-question sharing. [E6; implementation interpretation]

**Later mechanics update (§14):** the completed historical candidate-sharing results above are unchanged. E10 separately implements state-first nested branching for one synthetic Q = 2, K = 2 GPU fixture and compares hidden features. That is an initial question-branch mechanics observation, not a trained decision-quality, probability/policy-equivalence or general Q-scaling result. [E10; V3]

## 8.3 Complete short-request timings

The following values are medians across per-episode synchronized median complete-request times. Each K row draws from eight timed episodes and four messages; 32 timed episodes cover eight messages across all K values. They include cold prefix preparation within the request and should not be described as production p50 latency. [E5]

| Precision / candidates | Full sequential | Full batch four | Shared prefix, sequential suffixes | Shared prefix, equal-length suffix batch four |
|---|---:|---:|---:|---:|
| FP32 / K = 2 | 199.7 ms | 148.2 ms | 277.3 ms | 195.0 ms |
| FP32 / K = 4 | 399.1 ms | 278.7 ms | 460.5 ms | 377.8 ms |
| FP32 / K = 8 | 789.9 ms | 550.8 ms | 822.4 ms | 493.4 ms |
| FP32 / K = 16 | 1,585.8 ms | 1,113.1 ms | 1,550.5 ms | 763.5 ms |
| BF16 / K = 2 | 155.5 ms | 80.0 ms | 255.7 ms | 170.5 ms |
| BF16 / K = 4 | 309.7 ms | 91.9 ms | 428.9 ms | 344.4 ms |
| BF16 / K = 8 | 617.9 ms | 182.6 ms | 774.6 ms | 447.6 ms |
| BF16 / K = 16 | 1,238.5 ms | 363.7 ms | 1,465.5 ms | 654.9 ms |

Source: E5, benchmark_summary. BF16 rows are performance observations, not equivalent replacements for the accepted FP32 reference or for BF16 full-sequential decisions.

There is no universal “cache wins” result. For short FP32 requests, ordinary full batching was faster at K = 2 and 4; equal-length cached batching became faster at K = 8 and 16. Sequential cached suffixes often lost because saved prefix work did not offset extra calls and copies. Under BF16 the full-batch path was fastest on these short workloads, but its decision-equivalence gate failed. [E5]

For contrast, 2B’s fixed three-class batch-one decision path had an approximately 80.75 ms median. Generating 1, 8, or 32 tokens took about 91.32, 533.54, or 2,035.35 ms in the corresponding exploratory timing comparison. The often-attractive 25.2× ratio at 32 tokens is a difference in workloads. It does not establish an equal-quality speedup or an expected advantage over a tuned one-token classifier. [E1]

## 8.4 Longer prefixes reveal the amortization opportunity

Expanded 2E also used synthetic, pretokenized prefixes at K = 8. These measure execution mechanics, not semantic accuracy on long documents. FP32 results were:

| Shared prefix length | Full sequential | Full batch four | Cached sequential suffixes | Cached suffix batch four |
|---|---:|---:|---:|---:|
| 64 tokens | 1,173.6 ms | 718.4 ms | 821.0 ms | 324.1 ms |
| 256 tokens | 2,658.9 ms | 2,238.6 ms | 1,009.9 ms | 508.7 ms |
| 1,024 tokens | 8,854.7 ms | 8,908.4 ms | 1,757.8 ms | 1,269.6 ms |

Source: E5, fp32_strict_math.long_prefix. At 1,024 tokens, the cached batch-four path is approximately 6.97× faster than full sequential in this synthetic comparison. It also uses more transient memory: approximately 774.2 MiB peak extra allocation versus 311.6 MiB for full sequential. The result is a measured speed/memory trade-off, not a free reduction in both. [E5; ratio derived]

The natural operating principle is to select execution strategy by measured prefix length, candidate count, suffix-length distribution, available memory, and the required numerical contract. A cost model should include prefill, suffix work, cache copying/expansion, host overhead, and any compression/restoration. Counts of logically evaluated tokens alone do not predict the fastest strategy. Intrusive component profiles help locate costs, but should not be added to or substituted for independently timed complete requests. [E5]

# 9. Phase 2F: cache compression and prefix reuse

## 9.1 Completed experiment and evidence boundary

Phase 2F version 2f.1.0 completed run `20260918T224427722898Z` on the NVIDIA L4, with separate `fp32_strict_math` and `bf16_default` workers. It retained the Qwen checkpoint, candidate and none heads, and nine application policies from expanded 2E. No weights, thresholds, or calibration parameters were fitted. The completed results supersede the incomplete Phase 2F status in version 0.1 of this paper. [E6]

The main panel has 128 episodes from eight archived messages. Compression uses 32 episodes over those same eight messages; cold-request timing uses 16 episodes, four per K value, with two messages represented at each K. Each precision also runs two 32-request locality traces and synthetic prefixes of 64, 256, and 1,024 tokens. These denominators establish execution-regression coverage, not new cross-domain or long-context semantic validity. [E6]

The six storage configurations are `lossless`, `kv_fp16`, `tq_k3_v4_r0`, `tq_k3_v4_r32`, `tq_k4_v4_r32`, and `tq_k3_v2_r32`. In the TurboQuant names, k and v identify configured key/value bit settings, and r identifies the number of recent prefix tokens retained exactly. The pinned 0xSero implementation is revision `31660314b229b8d1c29bfddf8b9f5026b5121095`. The 384 MiB LRU budget and 600-second TTL are experiment settings, not established production optima. [E6]

## 9.2 What compression preserved and changed

Only full-attention keys and values are quantized. Recurrent and convolution state remain exact, and model weights remain unquantized. The adapter stores really packed tensors but restores floating-point KV before suffix attention. It therefore evaluates snapshot storage, restoration, and reuse; it does not evaluate fused low-bit attention or the upstream vLLM backend. CPU snapshot storage was tested separately from model offloading: both workers kept the model on the GPU. [E6]

TurboQuant’s paper motivates online vector quantization with rotations and scalar quantization; its published results do not guarantee that this particular snapshot adapter preserves decision probabilities. The notebook explicitly distinguishes the pinned community implementation from an official paper-author implementation or a paper replication. Its recorded GPL-3.0 license remains a dependency-provenance flag, not a licensing opinion. [P15; E6]

Storage labels also need care. The upstream nominal four-bit key configuration uses five bits per coordinate for its packed index-plus-sign representation before other metadata. The active codec bank reports 8,389,448 bytes of shared tables across the tested configurations; a representative individual TurboQuant configuration accounts for approximately 4 MiB. Tables are initialized before steady-state request timing. Actual measured bytes, not nominal bit labels, govern the storage comparison. [E6]

## 9.3 Probability and policy preservation: two distinct comparisons

Within each precision, full-sequential execution is the reference. FP32 full batching and equal-length shared suffix batches of four or eight all passed the 128-episode baseline, with maximum probability differences no larger than 0.00001072 and no selected-outcome or policy-output changes. BF16 did not reproduce its own full-sequential reference: full batch four changed selected outcomes in 14 episodes and policy outputs in seven; shared batches of four or eight changed selected outcomes in 14 and policy outputs in 11. [E6]

Codec error must then be separated from execution-shape error. Phase 2F compares each restored snapshot both with full-sequential output and with uncompressed cached output under the same suffix chunking. The audit independently re-aggregated the saved probability vectors and reconstructed policy actions from the frozen thresholds. The acceptance rule requires maximum absolute probability difference at most 0.005, no head argmax change, and no policy-output change. “Accepted” is an engineering regression gate, not adjudicated semantic correctness. [E6]

| FP32 snapshot | Max probability delta | Outcome changes | Policy changes | Accepted / 32 |
|---|---|---|---|---|
| `lossless` | 0.00000880 | 0 | 0 | 32 |
| `kv_fp16` | 0.00042450 | 0 | 0 | 32 |
| `tq_k3_v4_r0` | 0.21882282 | 8 | 2 | 5 |
| `tq_k3_v4_r32` | 0.18655649 | 8 | 5 | 6 |
| `tq_k4_v4_r32` | 0.12008530 | 2 | 1 | 12 |
| `tq_k3_v2_r32` | 0.47489768 | 13 | 6 | 5 |

Source: E6, compression results versus FP32 full sequential, 32 episodes from eight messages per codec. A selected-outcome change means at least one of three probability heads changed argmax; a policy change means at least one of nine frozen policies changed its returned action or accepted candidate. Counts are not independent message counts.

Lossless restoration reproduced the same-chunking control exactly. FP16 KV storage’s maximum codec-only difference was 0.00042351, with no class or policy changes. All four TurboQuant configurations failed even against the same-chunking control; their respective maximum codec-only differences were approximately 0.218824, 0.186557, 0.120082, and 0.474896. Retaining a 32-token exact tail or increasing the nominal key precision did not make the tested configurations pass. [E6]

| BF16 snapshot | Max delta vs full | Max delta vs same chunking | Accepted: full / same | Policy changes: full / same |
|---|---|---|---|---|
| `lossless` | 0.074689 | 0.000000 | 13 / 32 | 1 / 0 |
| `kv_fp16` | 0.074689 | 0.001884 | 13 / 32 | 1 / 0 |
| `tq_k3_v4_r0` | 0.337395 | 0.326159 | 7 / 8 | 2 / 3 |
| `tq_k3_v4_r32` | 0.186124 | 0.188542 | 11 / 12 | 3 / 2 |
| `tq_k4_v4_r32` | 0.159492 | 0.163646 | 8 / 9 | 4 / 3 |
| `tq_k3_v2_r32` | 0.410984 | 0.402534 | 4 / 4 | 4 / 5 |

Source: E6, BF16 codec regression. The left and right comparisons use the same 32-episode subset. Same-chunking lossless and FP16 storage introduce no selected-outcome or policy changes, but neither resolves the difference between cached execution and full-sequential execution. All four TurboQuant configurations introduce additional codec-only policy changes.

The BF16 lossless result is particularly informative: a zero-error snapshot round trip can still sit inside an execution strategy that changes decisions. Conversely, an unchanged top class on one fixture does not establish probability preservation. Some low-bit variants improved small-panel accuracy or NLL for individual heads; those frozen-panel observations cannot establish a quality gain or override the declared equivalence failure. This is a negative result for these specific adapter/configuration operating points, not a rejection of every TurboQuant implementation or quantization method. [E6; interpretation]

## 9.4 Measured storage: short-prefix overhead dominates

For the representative 42-token prefix, the FP32 root contains 48 MiB of recurrent state, 3 MiB of convolution state, and only 2.625 MiB of attention KV: 53.625 MiB total. The BF16 root still contains 48 MiB of recurrent state, with 1.5 MiB of convolution state and 1.3125 MiB of KV: 50.8125 MiB total. Thus lowering KV storage precision targets a small minority of the short-prefix cache. [E6]

| Storage configuration | FP32 snapshot | FP32 + tables | BF16 snapshot | BF16 + tables |
|---|---|---|---|---|
| `lossless` | 53.625 | 53.625 | 50.812 | 50.812 |
| `kv_fp16` | 52.312 | 52.312 | 50.812 | 50.812 |
| `tq_k3_v4_r0` | 51.379 | 55.380 | 49.879 | 53.880 |
| `tq_k3_v4_r32` | 53.090 | 57.091 | 50.590 | 54.591 |
| `tq_k4_v4_r32` | 53.110 | 57.110 | 50.610 | 54.610 |
| `tq_k3_v2_r32` | 53.071 | 57.071 | 50.571 | 54.571 |

Source: E6, `example_storage`, same representative prefix. Values are MiB of tensor storage. “+ tables” attributes the configuration’s shared tables to one entry; a deployment sharing one table across many entries should amortize that cost once. Python objects, allocator fragmentation, model weights, and restored working copies are excluded.

FP16 storage saves 1.3125 MiB, or 2.45%, of this FP32 root. In BF16 it saves no bytes: this is a format conversion, not a storage reduction. All four TurboQuant configurations make the single short entry larger after their tables are included. For `tq_k3_v4_r32`, the reported FP32 ratio is only 1.00065 at eight identical-size entries and 1.00770 at 32; BF16 remains slightly larger at eight. These are accounting examples, not measured cache-capacity or serving-quality gains. [E6; percentages derived]

In the actual grouped FP32 LRU trace, lossless entries occupy 394,199,040 bytes. The compressed entries occupy 389,769,984 bytes, but adding 4,194,608 table bytes leaves only 234,448 bytes less storage, approximately 0.22 MiB. Both caches hold seven entries and have the same hit/miss pattern. In BF16, the compressed entries plus tables exceed the corresponding lossless total. Compression therefore did not improve cache capacity in these traces, while its output-parity gate failed. [E6; arithmetic derived]

## 9.5 The hybrid-state memory ceiling, now checked against measurements

The earlier tensor-shape calculation remains useful. The 24 recurrent states contain 32 × 128 × 128 FP32 values each, totaling 48 MiB regardless of the two tested model dtypes. Convolution and full-attention KV account for the remaining dtype-dependent storage. For a single branch root, the measured tensor shapes give: [E4; E6; derivation]

```text
BF16 hybrid-cache storage(L) = 49.5 + 0.03125 × L MiB
FP32 hybrid-cache storage(L) = 51.0 + 0.06250 × L MiB
```

| Prefix tokens | Exact recurrent + convolution | BF16 KV | Total hybrid cache | Ideal total with 4-bit KV only | Ideal total reduction |
|---|---:|---:|---:|---:|---:|
| 64 | 49.5 MiB | 2.0 MiB | 51.5 MiB | 50.0 MiB | 2.9% |
| 256 | 49.5 MiB | 8.0 MiB | 57.5 MiB | 51.5 MiB | 10.4% |
| 1,024 | 49.5 MiB | 32.0 MiB | 81.5 MiB | 57.5 MiB | 29.4% |

This table preserves the idealized BF16 four-bit-KV calculation from version 0.1. It is not the measured behavior of the asymmetric Phase 2F codecs. It omits scales, norms, codebooks, exact tails, restoration buffers, and extra working copies. A large reduction in the KV component is not the same reduction in hybrid-cache storage, still less in total model memory.

At the measured 1,024-token FP32 synthetic prefix, lossless storage is 115 MiB and FP16 KV storage is 83 MiB: a 32 MiB, or 27.83%, root-storage reduction, with the sampled probability and policy checks passing. `tq_k3_v4_r0` instead occupies approximately 64.25 MiB including its tables, but fails the probability tolerance on that fixture. The storage benefit at longer prefixes does not erase the codec’s failed decision-preservation result. [E6; arithmetic derived]

## 9.6 Complete cold requests: reuse depends on candidate structure

| K | Full sequential | Full batch 4 | Shared batch 4 | Shared batch 8 | Lossless snapshot | FP16-KV snapshot |
|---|---|---|---|---|---|---|
| 2 | 195.4 | 147.2 | 271.5 | 271.7 | 274.6 | 287.3 |
| 4 | 392.1 | 281.4 | 373.7 | 373.3 | 376.7 | 389.2 |
| 8 | 793.2 | 582.5 | 576.6 | 576.9 | 578.8 | 590.8 |
| 16 | 1,595.3 | 1,180.4 | 724.5 | 686.4 | 727.3 | 738.8 |

Source: E6, FP32 medians of per-episode median complete-request times, in milliseconds. There are four episodes and two messages per K, with five measured repeats and two warmups per episode/strategy. Prefix computation is included in every cold request. Codec-table setup and cold model loading are excluded. These are not production latency percentiles.

Ordinary full batching is faster for K = 2 and 4. Shared batching is approximately tied with full batching at K = 8 and clearly faster at K = 16. At K = 16, suffix batch eight takes 686.4 ms: 2.32× faster than full sequential and 1.72× faster than full batch four in this panel. Its peak extra allocation is approximately 576.9 MiB versus 350.6 MiB for suffix batch four and 110.8 MiB for full batch four. Increasing batch capacity trades memory for latency; equal-length buckets need not actually fill that capacity. [E6; ratios derived]

Snapshot packing/restoration is not free. FP16 storage is slower than lossless snapshot storage at each tested K; the TurboQuant snapshot configurations take approximately 753–754 ms at K = 16 and fail the numerical gate. The BF16 full-batch path records 78.4, 89.4, 179.5, and 363.3 ms at K = 2, 4, 8, and 16 respectively, but fails equivalence in its own precision. Those lower timings are not interchangeable implementations of the accepted FP32 decision function. [E6]

Phase 2F’s benchmark subset and measured timings differ from expanded 2E’s. The tables should remain separate rather than replacing the earlier measurements or implying a controlled cross-phase performance gain. Intrusive component profiles retain tokenization, prefill, packing, restoration, cloning/expansion, suffix evaluation, and head/policy costs; those synchronized profiles should not be substituted for the independent complete-request measurements. [E5; E6]

## 9.7 Persistent exact-prefix reuse: a bounded positive result

| Trace | Strategy | Total seconds | Hits / misses | Evictions | Accepted |
|---|---|---|---|---|---|
| Grouped | No persistent cache | 15.740 | 0 / 32 | — | 32 / 32 |
| Grouped | GPU lossless LRU | 13.551 | 24 / 8 | 1 | 32 / 32 |
| Grouped | CPU lossless LRU | 14.413 | 24 / 8 | 1 | 32 / 32 |
| Grouped | GPU TQ k3/v4/r32 | 13.987 | 24 / 8 | 1 | 6 / 32 |
| Shuffled | No persistent cache | 15.701 | 0 / 32 | — | 32 / 32 |
| Shuffled | GPU lossless LRU | 13.836 | 21 / 11 | 4 | 32 / 32 |
| Shuffled | CPU lossless LRU | 14.605 | 21 / 11 | 4 | 32 / 32 |
| Shuffled | GPU TQ k3/v4/r32 | 14.333 | 21 / 11 | 4 | 6 / 32 |

Source: E6, FP32 single-worker traces. Each workload contains the same 32 requests over eight messages; the second changes their order. Total time includes initial misses and evictions, with one measured timing per request. The no-persistent-cache comparator still uses within-request prefix sharing and lossless snapshot restoration. This isolates persistence; it is not a comparison with full-sequential candidate evaluation.

Relative to that comparator, lossless GPU caching reduces total time by 13.91% for grouped locality and 11.88% for shuffled requests. Lossless CPU storage reduces total time by 8.43% and 6.98%, respectively, but is slower than GPU storage on these traces. Lossless GPU and CPU variants preserve the FP32 reference in all measured requests. The compressed GPU cache is slower than lossless GPU caching, has no additional hits, and changes policy outputs in four requests in each workload. [E6; percentages derived]

The configured 384 MiB budget covers cache-entry tensors only. Codec tables, the model, restored roots, temporary branches, and allocator reservations remain outside it. No TTL expiry or oversize bypass occurred. The audit reproduces the reported hit, miss, eviction, and retained-byte counts from the saved key/size trace; it does not validate expiration behavior or concurrent access. Comparing hit-only and miss-only medians is also not a paired speed test because those populations can have different candidate counts and suffix work. [E6]

BF16 GPU lossless caching also reduces trace time, by 14.98% and 13.27%, but every lossless trace strategy retains the same four selected-outcome and two policy-output disagreements with BF16 full sequential; only 14 of 32 requests pass. Reuse has not cured the underlying execution-shape issue. The trace’s cache keys preserve the current exact instruction/message prefix identity, not an abstract state shared across arbitrary new questions. [E6; interpretation]

## 9.8 Long-prefix mechanics: report cold and warm separately

| Prefix tokens | Full sequential | Full batch 4 | Cold lossless | Warm lossless | Warm FP16-KV |
|---|---|---|---|---|---|
| 64 | 1,235.5 | 760.5 | 329.0 | 232.4 | 235.9 |
| 256 | 2,827.1 | 2,329.5 | 525.2 | 238.1 | 239.5 |
| 1024 | 9,217.4 | 9,250.7 | 1,295.9 | 272.9 | 274.7 |

Source: E6, FP32 synthetic K = 8 mechanics, times in milliseconds. Each cell summarizes three repeats. Every listed path passes the sampled probability/policy gate; a warm snapshot excludes prefix population. The long-prefix harness reports population-request time separately. These repeated synthetic tokens do not establish decision accuracy on a real long document.

At 1,024 tokens, cold shared-prefix execution is approximately 1,296 ms versus 9,217 ms full sequential; warm lossless reuse is approximately 273 ms after an initial population request of approximately 1,295 ms. These answer different latency questions and must not be presented as a single unconditional speedup. FP16 storage’s warm time is approximately 275 ms at this length; its principal measured benefit is smaller retained storage, not faster suffix evaluation. [E6]

All four low-bit codecs fail the 0.005 probability gate at all three synthetic prefix lengths in FP32, even though those particular fixtures record no argmax or policy-output changes. BF16’s faster warm lossless times, approximately 178, 179, and 184 ms, also fail comparison with BF16 full sequential, with maximum probability differences approximately 0.01683, 0.02084, and 0.03232. Warm execution, compressed storage, and accepted decision equivalence remain separate axes. [E6]

## 9.9 What Phase 2F changes and leaves open

Phase 2F moves exact-prefix persistence from a proposal to a measured single-worker result. It adds FP16 storage of full-attention KV as a candidate FP32 optimization. Its sampled parity passes, and its storage savings grow with prefix length. Phase 2F also supplies a specific negative result: none of the four tested TurboQuant snapshot configurations preserves the current frozen decision contract. Codec execution and CPU packing selftests passed; downstream equivalence did not. [E6]

The engineering interpretation is to keep lossless storage as the reference, evaluate FP16 KV storage on fresh semantic and realistic traffic panels, and retain low-bit variants as experiments rather than accepted replacements. Any refitting of heads or policies to tolerate a new codec would define a new model operating point and require fresh final evaluation. No new Score/Noul training, arbitrary-question shared-state architecture, Rust/Metal backend, HTTP service, concurrent serving, or deployment safety result follows from this run. [E6; recommendation]

Multi-token prediction remains not applicable: no output tokens are generated, and every candidate suffix token is already supplied. The completed report correctly records a missing generation benchmark as not applicable rather than inventing an MTP gain. Further acceleration should target the measured decision graph and its complete request costs, not an output-decoding loop it does not contain. [E6]

Phase 2G subsequently tests lossless and FP16-KV storage on fresh inputs and an expiry-bearing traffic trace, while adding TF32 and semantic transfer diagnostics. Its findings in Section 10 extend this evidence without replacing the Phase 2F workload, timings, or failed-codec outcomes. [E7]

# 10. Phase 2G: fresh decisions, TF32, and cache lifecycle

## 10.1 Completed run and evaluation scope

Phase 2G version `2g.1.0`, run `20260919T005142584348Z`, completed both `fp32_strict_math` and `fp32_tf32_allowed` workers on an NVIDIA L4. The workers loaded the same text-only Qwen checkpoint directly into separate processes; neural weights, feature normalization, all three evaluated none heads, and the nine previously selected application policies remained frozen. No calibration, threshold fitting, LoRA, or additional neural training occurred. The experiment therefore tests transfer and execution changes, not a newly improved trained model. [E7]

The main evaluation contains 416 sampled-choice episodes from 112 distinct messages. Banking77 selections preserve the original author test labels and the prior 57/20 head-training label split. CLINC adds non-financial-domain requests and a separately reported author-labeled out-of-scope (OOS) panel. The eight selected non-financial domains each contribute four messages. In-scope messages receive paired correct-intent-present and correct-intent-omitted episodes at K = 4 and 16, with one preassigned distractor sampler per message. OOS messages receive no fabricated positive-answer counterpart. [E7, data and experiment_inputs.json]

| Fresh family | Distinct messages | Episodes | Meaning of the absent-answer target |
|---|---:|---:|---|
| Banking77, head-training labels | 32 | 128 | The annotated correct banking intent is deliberately omitted |
| Banking77, held-out labels | 32 | 128 | The annotated correct banking intent is deliberately omitted |
| CLINC, non-financial domains | 32 | 128 | The annotated correct in-scope intent is deliberately omitted |
| CLINC, author-labeled OOS | 16 | 32 | The author marks the request out of scope; no offered in-scope intent is the target |
| Total | 112 | 416 | Do not pool omission and author-OOS rates without stating the mixture |

Source: E7. The first three families are 50% absent by construction. These are sampled alternatives, not standard full 77-way or 150-way benchmark accuracies. The 2,912 excluded normalized Banking message hashes come from the recorded C/D/E manifests; expanded E and F replayed those earlier messages. Cross-family exact duplicates are also excluded. Neither paraphrase deduplication nor Qwen-pretraining decontamination is established.

The optimized-path parity panel is a predefined subset of 56 episodes from 16 messages. The context experiment creates 72 variants from six selected in-scope messages. Complete-request timing uses eight episodes, and traffic repeats a 48-request event sequence twice per strategy. None of those repeated or constructed observations adds an independent source message. Matched reference quality is reported for each optimization subset rather than comparing its accuracy with the larger 112-message population. [E7]

## 10.2 Strict-FP32 cache agreement survives fresh inputs

Full-prompt batching, lossless shared-prefix snapshots, and FP16 attention-KV snapshots all passed the declared numerical, selected-outcome, and policy-output gate on the fresh parity panel. Each is compared with the same episode under strict-FP32 full-sequential execution. [E7]

| Panel | Alternative | Episodes / source messages | Max probability delta | Outcome changes | Policy changes | Accepted |
|---|---|---|---|---|---|---|
| Fresh inputs | Full batch four | 56 / 16 | 0.00000731 | 0 | 0 | 56/56 |
| Fresh inputs | Shared lossless | 56 / 16 | 0.00000774 | 0 | 0 | 56/56 |
| Fresh inputs | Shared FP16-KV | 56 / 16 | 0.00045509 | 0 | 0 | 56/56 |
| Controlled context | Full batch four | 72 / 6 | 0.00003582 | 0 | 0 | 72/72 |
| Controlled context | Shared lossless | 72 / 6 | 0.00001442 | 0 | 0 | 72/72 |
| Controlled context | Shared FP16-KV | 72 / 6 | 0.00054408 | 0 | 0 | 72/72 |

Source: E7, fresh and controlled-context prediction rows; independently re-aggregated for v0.3; not rerun in v0.5. Probability differences are absolute probability units, not percentages. The tolerance is 0.005, equivalent to 0.5 percentage points. Selected-outcome and policy counts are episode-level any-change indicators across three heads and nine policies, not counts of independent messages.

This strengthens the positive result from E/F: the tested strict-FP32 cached graph reproduces its full-prompt reference on inputs beyond the old eight-message replay panel. FP16-KV remains a storage conversion applied only to attention keys and values; recurrent and convolution state stay exact, model weights remain FP32, and restored attention computation uses the original floating-point dtype. It is not a reduced-precision model-weight result. [E6; E7]

The claim remains sampled agreement. The fresh panel has 16 messages; the context panel has six selected source messages and constructed backgrounds. Zero observed failures do not establish a universal failure bound or deployment safety. The notebook's message-level zero-event bounds require independent/exchangeable-message assumptions and are not a certification. In particular, faithfully reproducing the reference says nothing by itself about the correctness of that reference's semantic answer. [E7]

## 10.3 TF32 permission improves speed without reducing FP32 storage

The TF32-permitted worker records `float32_matmul_precision = high` and `matmul.allow_tf32 = true`; the strict worker records `highest` and `false`. Both retain FP32 parameters. Convolution TF32 is disabled, and the comparison retains the prescribed math-attention setting. These flags authorize an arithmetic path; the experiment does not trace every dispatched matrix kernel or establish which individual operations use tensor cores. Both workers allocate approximately 16,043.7 MiB immediately after loading. This is not a model-weight-memory saving. [E7, flags and memory_snapshots]

| Execution strategy | K | Strict FP32, ms | TF32 permitted, ms | Strict / TF32 time |
|---|---|---|---|---|
| Full sequential | 4 | 395.1 | 345.9 | 1.14× |
| Full batch four | 4 | 265.1 | 131.7 | 2.01× |
| Shared lossless | 4 | 336.4 | 316.0 | 1.06× |
| Shared FP16-KV | 4 | 349.1 | 323.2 | 1.08× |
| Full sequential | 16 | 1572.7 | 1372.9 | 1.15× |
| Full batch four | 16 | 1118.3 | 518.5 | 2.16× |
| Shared lossless | 16 | 679.1 | 620.2 | 1.09× |
| Shared FP16-KV | 16 | 691.2 | 627.9 | 1.10× |

Source: E7, complete cold-request benchmark. Values are medians of per-episode median wall times; each strategy/K cell contains four episodes, with three timed repetitions and one warmup per episode. Tokenization, transfers, prefix creation where applicable, snapshot restoration, scoring, policies, and serialization are included. Model loading, offline evaluation-score caches, HTTP/network transport, and concurrent serving are excluded. Ratios are derived from the unrounded recorded medians, not production latency percentiles.

The largest gain is full-prompt batching: about 2.01× at K = 4 and 2.16× at K = 16. Sequential full prompts improve by roughly 1.14–1.15×, while lossless cached execution improves by only about 1.06–1.09×. On these short requests, TF32-permitted full batching becomes faster than shared-prefix execution at both tested candidate counts. Under strict FP32, full batching wins at K = 4 and lossless sharing wins at K = 16. [E7; arithmetic derived]

This is a scheduler-design observation, not a universal candidate-count threshold. The preferred path depends on arithmetic, prefix length, suffix grouping, available memory, and the required behavioral contract. Phase-to-phase timing tables remain separate: the F and G panels differ, so these results do not establish a controlled speed change from F to G. The substantial synthetic long-prefix benefits in E/F also do not imply that every short request should be cached. [E5–E7; engineering interpretation]

## 10.4 The complete TF32 equivalence gate does not pass

A probability-difference tolerance alone would miss the main fresh-panel discrepancy. Across 416 identical full-sequential inputs, TF32 versus strict FP32 stays below 0.005 on every episode, yet three episodes change a head's selected outcome. Two underlying messages are affected. None of the nine frozen policy outputs changes, so the complete unchanged gate accepts 413/416 episodes rather than all 416. [E7; R2]

| Comparison | Episodes / source messages | Maximum probability difference | Over 0.005 | Selected-outcome changes | Policy-output changes | Accepted |
|---|---:|---:|---:|---:|---:|---:|
| TF32 full sequential versus strict-FP32 full sequential, fresh panel | 416 / 112 | 0.00383692 | 0 | 3 | 0 | 413/416 |
| TF32 full sequential versus strict-FP32 full sequential, controlled context | 72 / 6 | 0.00675709 | 2 | 0 | 0 | 70/72 |

Source: E7, fresh cross-precision summary and raw controlled-context rows; R2. The second row is an additional saved-output audit, not part of the notebook's fresh-only cross-precision summary. It was rechecked for v0.3, not rerun for v0.5. These two panels have different constructions and should not be pooled into one independent error-rate estimate.

Inspection of the three fresh changes found two refitted-global predictions moving from correct none to an incorrect offered intent, and one change between two already-incorrect offered candidates. The previously selected set-linear head's fresh-panel argmax predictions did not change. This localizes the recorded outcome difference but does not establish that set-linear would remain invariant on a broader population. Near a tie, small numeric changes can still select a different answer. [E7, fresh_rows.json; R2]

Within the TF32 mode, all three optimized strategies passed all 56 fresh parity episodes against TF32 full sequential. The longer context variants did not all pass. Each optimized strategy accepted 70/72 context episodes: one exceeded the probability tolerance and a different episode changed argmax, with no policy-output changes. [E7]

| TF32 alternative versus TF32 full sequential | Max probability delta | Over 0.005 | Outcome changes | Policy changes | Accepted / 72 |
|---|---|---|---|---|---|
| Full batch four | 0.00670605 | 1 | 1 | 0 | 70 |
| Shared lossless | 0.00580375 | 1 | 1 | 0 | 70 |
| Shared FP16-KV | 0.00795940 | 1 | 1 | 0 | 70 |

Source: E7, within-mode controlled-context comparisons. The maximum FP16-KV difference is 0.00795940, or approximately 0.796 percentage points. These strategy comparisons do not isolate storage error alone: each also includes the cached/batched execution difference from full sequential. The single changed argmax and the single numeric-threshold failure are distinct episodes for each row.

TF32 is therefore a separately versioned performance candidate, not an accepted exact replacement. An unchanged application policy can coexist with a changed semantic outcome, especially when a policy reviews both alternatives. The paper does not relax the gate after seeing the results, claim that TF32 is universally less accurate, or promote its speed advantage into a universal deployment recommendation. [E7; interpretation]

## 10.5 Fresh semantic quality exposes a rejection-transfer limit

The fresh, strict-FP32 full-sequential results preserve all three none-head comparisons. None of these heads was selected or refitted on Phase 2G. The set-linear head shown as the reference remains the model selected in Phase 2D; the other rows are retained ablations rather than test-set-driven replacement selections. [E3; E7]

| Family | Frozen none head | Accuracy | NLL | Answerable accuracy | None recall |
|---|---|---|---|---|---|
| Banking, head-training labels | Original global | 45.31% | 1.8243 | 81.25% | 9.38% |
| Banking, head-training labels | Refitted global | 64.84% | 1.0459 | 67.19% | 62.50% |
| Banking, head-training labels | Set-linear | 69.53% | 0.8821 | 70.31% | 68.75% |
| Banking, held-out labels | Original global | 51.56% | 1.4743 | 90.62% | 12.50% |
| Banking, held-out labels | Refitted global | 77.34% | 0.7043 | 85.94% | 68.75% |
| Banking, held-out labels | Set-linear | 75.78% | 0.6471 | 82.81% | 68.75% |
| CLINC, non-financial domains | Original global | 49.22% | 2.6243 | 96.88% | 1.56% |
| CLINC, non-financial domains | Refitted global | 62.50% | 1.5126 | 96.88% | 28.12% |
| CLINC, non-financial domains | Set-linear | 66.41% | 1.0366 | 93.75% | 39.06% |
| CLINC, author OOS | Original global | 0.00% | 4.2253 | Not applicable | 0.00% |
| CLINC, author OOS | Refitted global | 25.00% | 2.0479 | Not applicable | 25.00% |
| CLINC, author OOS | Set-linear | 46.88% | 1.3385 | Not applicable | 46.88% |

Source: E7, fresh_summary and stored prediction vectors. Accuracy combines correct offered answers and semantic none under each panel's declared mixture. Answerable accuracy counts all present-target episodes, including false-none predictions as errors; it is not accuracy conditional on the model choosing to answer. NLL uses the full offered-candidate-plus-none distribution. OOS has no present-target condition, so answerable accuracy is not applicable rather than zero.

For the set-linear reference, the fresh Banking seen-label accuracy is 69.53%, with a recorded 95% message-bootstrap interval of approximately 57.8–80.1%; held-out-label accuracy is 75.78%, with an interval of approximately 66.4–84.4%. Those small-sample results cannot establish that held-out labels are generally easier. They also cannot be treated as a measured deterioration from C or D: the weights did not change, and the messages, candidate-count mix, and evaluation construction differ. [E7]

CLINC provides the clearest separation between matching an available answer and detecting that no answer is available. Set-linear is correct on 60/64 episodes where the annotated intent is offered, but predicts none on only 25/64 episodes where it is omitted. On the separate author-OOS panel, it predicts none on 15/32 episodes from 16 messages; 17 episodes instead receive an offered intent under unthresholded argmax. These figures are evidence of incomplete rejection transfer, not a general OOS detection rate for all domains. [E7; counts derived]

The corresponding set-linear ECE values are approximately 0.1812 for Banking seen labels, 0.0911 for Banking held-out labels, 0.1605 for CLINC non-financial requests, and 0.3187 for author-OOS. ECE is descriptive and sample-dependent; neither those values nor the NLL rows certify calibration in deployment. The model can use descriptions beyond banking, but the frozen banking-derived none mechanism and policies are not established as universally calibrated. [E7]

## 10.6 High-confidence errors reveal a criteria-definition problem to investigate

The previously selected policy with assumed missing-answer prevalence 5%, wrong-answer cost 20, and review cost 0.1 accepted 20/128 Banking-seen episodes. Four accepted episodes were incorrect under the unchanged author annotations: a 20% error rate among accepted episodes and a scenario-weighted cost of 0.698125, compared with the stipulated always-review cost of 0.1. The raw 20/128 coverage and the scenario-weighted cost use different weighting conventions and should not be conflated. [E7]

All four errors derive from one source message expanded across K = 4/16 and present/omitted conditions:

> How can I get a physical card

The author label is `order_physical_card`. When offered, its description is “order physical card.” The model instead selects the alternative described as “get physical card,” with set-linear probabilities approximately 0.990888, 0.998946, 0.988551, and 0.996572 across the four episodes. The episode IDs start `g_banking_seen_2501_label_lexical_hard_`; the raw candidate mapping and predictions remain in the archive. [E7, experiment_inputs.json and fp32_strict_math/fresh_rows.json; R2]

The distinction is not clear from the terse descriptions alone. That is an interpretation warranting a documented domain-specific criteria and annotation review, not an adjudication that the dataset is wrong or that the prediction should be accepted. The four episodes remain errors under the declared benchmark. They must not be removed, relabeled, or retroactively counted as synonyms to improve the current result. [R2; research recommendation]

The same 0.98-threshold set-linear policy is used in the 5%-absent, wrong-cost-5 scenario; it accepts the same episodes and has scenario-weighted cost 0.229375. Changing the cost assumption alone does not fix an overconfident confusion, and changing the threshold after seeing these test cases would consume the test set. More explicit candidate criteria, training coverage, rejection fitting, and calibration are distinct interventions requiring separate evaluation. [E7]

This case explains why a dynamic-choice API needs both stable opaque IDs and clear semantic descriptions. IDs should remain machine-facing; descriptions must carry the task distinction. Before broadening the system, audit confusing candidate pairs and document the intended boundary. There is no evidence yet that a larger model, LoRA, or stricter threshold alone is the best remedy. [E7; interpretation and proposed work]

## 10.7 Context length and evidence position affect semantic decisions

The controlled-context test wraps six labeled requests in generated administrative notes. The instruction says to classify the current request rather than the background. A minimal-wrapper control uses the same instruction, while longer variants place the request first or last with at least 256 or 1,024 state tokens. The source label is retained, and cases that would truncate the request are not silently treated as valid comparisons. These are controlled transformations, not natural long operational documents. [E7]

| Minimum state-token target | Request position | Correct / episodes | Source messages | Accuracy | NLL |
|---|---|---|---|---|---|
| 0 | first | 11/12 | 6 | 91.67% | 0.3454 |
| 0 | last | 11/12 | 6 | 91.67% | 0.2839 |
| 256 | first | 9/12 | 6 | 75.00% | 0.5187 |
| 256 | last | 10/12 | 6 | 83.33% | 0.4363 |
| 1024 | first | 8/12 | 6 | 66.67% | 0.5932 |
| 1024 | last | 10/12 | 6 | 83.33% | 0.3342 |

Source: E7, strict-FP32 full-sequential context rows; R2 pooled aggregation rechecked for v0.3. Every row has 12 paired episodes from the same six messages. A minimum-token target of zero denotes the minimal wrapper, not an empty input; the two minimal layouts are not independent replications. NLL is averaged over the listed episodes.

Strict-FP32 cached strategies reproduced the full-sequential results throughout this panel, but the full-sequential task performance still changed with added background and request position. Thus execution fidelity and long-context semantic validity diverge in a concrete experiment: an optimized path can faithfully reproduce a decision that is wrong under the source label. The sample is too small to establish a general evidence-position law, but it identifies a useful next target for controlled reliability training and fresh evaluation. [E7; interpretation]

## 10.8 Persistent reuse with actual expiry decisions

Phase 2G reduces the cache-entry budget to 192 MiB and uses a 20-second TTL. The 48-request trace includes hot reuse, scans, two tenant namespaces, and arrival gaps that cross expiry boundaries. The same trace is run twice per strategy under each numerical mode. Arrival time advances on a controlled virtual clock, while request service time is measured. The test does not sleep through idle gaps or measure queueing, concurrent readers, an HTTP server, or deployment traffic. [E7]

| Arithmetic | Strategy | Mean total service time, s | Reduction versus no persistence | Same-mode parity |
|---|---|---|---|---|
| Strict FP32 | No persistence | 22.0001 | Reference | 48/48 in each repetition |
| Strict FP32 | GPU LRU, lossless | 20.6376 | 6.19% | 48/48 in each repetition |
| Strict FP32 | GPU LRU, FP16-KV | 21.1306 | 3.95% | 48/48 in each repetition |
| TF32 permitted | No persistence | 20.3693 | Reference | 48/48 in each repetition |
| TF32 permitted | GPU LRU, lossless | 19.1290 | 6.09% | 48/48 in each repetition |
| TF32 permitted | GPU LRU, FP16-KV | 19.4215 | 4.65% | 48/48 in each repetition |

Source: E7, traffic_rows.json and traffic_summary; mean of two trace totals, each covering the same 48 events. “No persistence” still shares the prefix within a request and therefore isolates cross-request reuse rather than comparing against full-sequential candidate re-encoding. All measured trace strategies pass their own mode's full-sequential reference gate. This does not override TF32 failures on other panels.

Per repetition, both stored-cache variants record 15 hits, 33 misses, 21 capacity evictions, and nine expired entries. Three entries remain. Lossless storage retains 161 MiB, versus 157 MiB for FP16-KV, but neither variant gains another slot or hit in this trace. Lossless caching reduces total service time by approximately 6.19% in strict FP32 and 6.09% with TF32 permission. FP16-KV also reduces time versus no persistence, but remains slower than lossless caching; smaller retained storage is not automatically lower request latency. [E7; R2; arithmetic derived]

Seven small-tensor CPU lifecycle checks cover exact-boundary expiry, byte-bounded eviction, oversized-entry bypass, tenant/execution-key separation, independently mutable restored state, exact recurrent-state preservation, and abandonment of a reader without root mutation. The GPU trace adds measured expiry and capacity events; it does not validate an authorization boundary, in-flight cancellation, or concurrent mutation. Phase 2F's no-expiry traces and Phase 2G's expiry trace should remain separate evidence rather than being described as a controlled change in cache efficiency. [E6; E7]

## 10.9 Updated interpretation after Phase 2G

The execution foundation is stronger: strict-FP32 lossless reuse and FP16-KV storage pass broader sampled tests, including controlled context. TF32-permitted full batching supplies a substantial measured speed opportunity but does not pass every semantic-outcome/numerical equivalence check. Neither inference completion nor policy stability alone constitutes full acceptance. [E7]

The task-quality conclusion after G was more demanding: supplied-candidate transfer did not settle missing-answer/OOS rejection, terse criteria could cause high-probability errors, and irrelevant context could change labeled decisions. Those findings motivated the controlled H study. H now has completed final evidence in §13.1.8–§13.1.16; it extends the quality record without changing G’s frozen-head results or failed equivalence outcomes. Its results still support separate criteria, rejection, selection and execution experiments rather than changing prompts, precision, heads and caching together. [E7; E9; historical-to-current handoff]

The subsequent architecture review broadens the response: retain that study, but prioritize adaptation to different questions and rubrics, early smaller-model comparisons, and genuine multi-question execution. The execution suite remains binding for implementations claiming to preserve the same model, not a requirement that every newly trained model reproduce G’s errors. Cache tuning, rendering, arithmetic, heads, and LoRA still require controlled comparisons for attribution. [R3; v0.4 recommendation]

# 11. Architecture and implementation recommendations

## 11.1 Preserve three distinct layers

**Decision semantics:** define state, question instructions, candidate descriptions, label/level meaning, missing-option semantics, and the desired probability contract. This layer determines what the model is being asked.

**Probability engine:** bind the model revision, tokenizer, prompt rendering, normalization, head parameters, precision, kernels, and cache strategy. Its output is a versioned numerical function, not merely a checkpoint filename.

**Application policy:** choose accept, review, gather more information, or route elsewhere using an explicit cost model and authorization rules. This layer must not be mistaken for probability calibration or a semantic class. The E policy experiments and G transfer failures show why separating it is useful. [E2–E7; recommendation]

A service should return model/engine and policy version identifiers with telemetry, and report truncation and unsupported inputs explicitly. In the historical E2–E7 studies, most untouched task-quality inputs use a maximum length of 256. Phase 2G adds controlled labeled contexts through a minimum 1,024 state tokens, but their six source messages and generated administrative notes do not establish reliable decisions on natural long operational documents. Keep the synthetic mechanics and labeled context controls distinct. [E2–E7]

## 11.2 State-first Qwen and conditional model alternatives

**State-first Qwen path.** E11 selected state-first rendering, and §17.3 records
native full-hybrid branching parity. Retain the following execution structure
for the immutable candidate-branch integration reference. The newer joint-option
quality graph is distinguished below. Any upstream adaptation or input repair changes
the model/execution identity and requires its own readout, normalization,
calibration, and quality record. [E11–E13; §17.3]

```text
stable format + state → immutable shared root
    ├─ isolated question A → question-A snapshot
    │      ├─ isolated candidate A1 → score
    │      └─ isolated candidate A2 → score
    └─ isolated question B → question-B snapshot
           ├─ isolated candidate B1 → score
           └─ isolated candidate B2 → score
```

Each question receives its own none/applicability calculation and distribution. Full-attention KV, recurrent state, and convolution state must all remain isolated at both fork levels. A block attention mask alone does not isolate independent questions concatenated into Qwen’s recurrent stream. Verify branch positions, finalized token IDs, root immutability, and equivalence to full execution of the **new** input contract before measuring reuse. [E4–E7; R3; proposed controls]

**Joint-option quality path.** E29/E30 place the question and all its offered
semantic outcomes in one request, and score only the allowed answer-code rows at
the final real input position. E30 trains that decision objective through the
hybrid backbone with bounded LoRA; it does not train a head over a pooled state
prefill. The measured graph is:

```text
state + question + all offered outcomes → full Qwen forward
    → final answer-position representation → selected vocabulary-row logits
    → semantic distribution → separate review policy
```

Both studies use full forwards per question, not newly qualified shared-cache
execution. Sharing an immutable root across these question suffixes remains a
systems extension that needs adapter-specific identity, branch isolation, and
numerical/service checks. In-domain training gains do not qualify that extension
or replace the older candidate-branch reference. [E29; E30]

E31/E32 retain this full-forward quality path for their replay comparisons.
Source-label replay's positive SNLI result and failed long-document retention do
not test a new shared-state topology, decoder, evidence reader, or accelerated
service. Preserve the distinction between a changed learning objective and a
changed execution graph. [E31; E32]

**Compact bidirectional dynamic-candidate path.** E11 completed the exploratory
ModernBERT comparison. It did not establish a release-quality smaller encoder.
A further compact-model comparison is conditional on useful quality and an
identified cost problem. A released external checkpoint remains a separate
baseline with unmatched training history. [E11; §15.3]

**Shared-representation/query-module path.** B1/B2 already tested learned access
to frozen token-level state representations, including factorized applicability.
They improved some aggregate metrics but missed complete quality/policy gates.
A new reader must specify what input information or training changes relative
to those arms. The later pooled-root failure does not rule out every learned
query architecture. Perceiver IO remains a conceptual precedent. [P11; E14–E20;
E26]

No path should claim nearly flat question scaling from a candidate-only benchmark or a single batched call. Measure both Q and K, state length, complete latency, memory, and semantic isolation. [R3; proposed evaluation]

**Implementation boundary.** E10's duplicate-added-question probe remains
historical. E11 corrected the distinct-question mechanics, E12 established the
Python branching reference, and native CPU plus separately gated MLX FP32
execution now pass their stated fixtures. Preserve those assets while M1/M2
establish reviewed evidence and useful decisions. [E10–E13; §17.3]

## 11.3 Rust and Apple Silicon: retain the parity ladder

The parity ladder remains the qualification method for new implementations.
The selected profile now passes native CPU full/nested execution and fresh-process
replay, plus separately qualified MLX FP32 full/nested/vectorized fixtures.
BF16 remains outside its unchanged gate. Saved head fixtures alone would not
establish these results. Section 17.3 and [MLX.md](../MLX.md) record their scope.

The recommended ladder starts with deterministic head algebra: normalization, affine score, stable softmax, label masking, none features, and temperature. Next compare token IDs, masks, truncation, and position IDs. Then validate full-backbone hidden vectors and decisions. Only after that should the port introduce hybrid-cache branching, batched suffixes, alternative precision, and persistent cache storage.

For each step retain both probability error and policy-output differences. Do not import L4 latency numbers as Mac predictions, and do not assume CUDA’s accepted precision strategy has the same behavior under Metal or a different kernel implementation. A backend should advertise the exact tested operating mode and return a clear unsupported-path error rather than quietly falling back to a semantically different pipeline. [P14; E5; recommendation]

For a linear fixed head, the training normalization can be folded algebraically into an effective matrix and bias, but any such transformation should first pass exported fixtures at the chosen precision. The general identity is W′ = W / σ and b′ = b − W(μ / σ), with column-wise division; it does not remove the requirement to reproduce tokenizer and backbone behavior. [Algebraic implementation proposal]

## 11.4 Cache identity, isolation, and safety controls

Use exact token-prefix identity together with model, tokenizer/rendering, position convention, adapter, and execution-mode versions. A cache entry for one instruction/state pair must never be reused for a different pair merely because their text is similar. Establish tenant/authorization boundaries independently of content hashing, and treat cached state as potentially sensitive user data. [E6 cache identity checks; recommended extension]

Bound cache capacity by actual bytes, including auxiliary states and shared codec buffers. Define cancellation, TTL, eviction, and active-reader behavior; prohibit mutation of a root still available to another branch. Capacity and admission should account for temporary expansion into multiple suffix rows, not only the resident snapshot. Measure cache-hit rate on a specified workload rather than assuming a warm cache is typical. [E5; E6; recommendation]

Phase 2F shows why a cache-entry budget is not a process-memory ceiling: its 384 MiB limit excludes shared codec tables and temporary restored/expanded state. Its traces exercise misses and capacity eviction without expiry. Phase 2G adds 192 MiB traces with nine expiration events and 21 capacity evictions per repetition under a virtual arrival clock, plus seven small-tensor lifecycle contracts. Neither run validates concurrent GPU readers, authorization enforcement, in-flight cancellation, HTTP behavior, or a memory-leak soak test. Keep those properties separate from passed output-regression gates. [E6; E7; implementation recommendation]

Typed construction narrows what a model can output, but it does not make malicious state content harmless. An adversarial message can still influence a wrong candidate score. Before cybersecurity or other high-consequence deployment, add adversarial instruction-in-state cases, missing evidence, contradictory state, malformed Unicode, duplicate or ambiguous descriptions, and cross-question consistency tests. Keep authorization and irreversible actions outside the probabilistic model. These are proposed safeguards; the completed Banking77/CLINC and controlled-context panels do not validate an adversarial-security boundary.

## 11.5 Make candidate criteria an explicit versioned contract

Preserve the separation between opaque candidate IDs and their descriptions, but do not assume label names alone supply adequate semantic criteria. The Phase 2G physical-card case warrants review of confusing pairs under documented annotation rules. Revised descriptions, merged categories, or allowed multi-label/synonym judgments would change the task contract and should receive new versions, training/evaluation manifests, and regression fixtures. Existing gold labels and recorded errors remain unchanged. [E7; R2; recommendation]

Evaluate repairs separately: independently reviewed criteria changes, additional support examples, none/calibration refitting, multi-domain readout fitting and backbone adaptation. H's support-example variant adds labeled information without independent review. Its final results confirm that the support/readout interaction is not a uniformly beneficial transfer intervention: the preselected support model improves fitting-family accuracy but has weaker held-out results than a retained original-criteria joint control. Treatment-specific policies were selected before final evaluation and remain fixed. Carry these attribution boundaries into the next study rather than treating H as a completed independent criteria audit. [E8; E9; interpretation]

Retain strict FP32 as the historical execution reference and TF32 as a performance candidate; do not mix an input-contract change with automatic arithmetic or policy promotion. Evaluate accepted-answer correctness as well as errors withheld for review. Replaying G after using its errors to design a repair is regression testing, not another fresh final test. [E7; R3; proposed work]

## 11.6 Train question semantics and evidence-aware rejection

The next reviewed dataset should contain different questions over the same state, including questions that legitimately require different answers. Include explicit inclusion/exclusion criteria, negation, conflicting evidence, insufficient evidence, unrelated background, and instruction-like text inside the state. Hold out whole question or rubric families in addition to source-state groups; held-out intent names alone do not test this target. Begin with a modest, carefully reviewed set rather than scaling unverified synthetic labels. Keep annotation disagreements and missing evidence visible instead of forcing every item into a confident single-answer target. [R3; proposed dataset]

Add a separate **state/evidence sufficiency** axis. Deliberately omitted candidates, author-OOS, insufficient evidence, an observation that simply does not expose the required fact, and an application decision to escalate are not the same event. The DGX Doom study is useful motivation: changing the textual observation/history representation materially changed controller outcomes even though the underlying game task was unchanged. OpenKind should include reviewed cases where the requested answer is not recoverable from the visible state, and should measure how richer but still non-leaking state summaries affect both correctness and confidence. This is a dataset/evaluation requirement, not evidence that the current profile already solves partial observability. [P22; proposed dataset extension]

For example, one security-event state could support a question about the affected asset, another about whether external communication is evidenced, and a third about which documented handling category applies. This is an illustrative training design, not a claim that the current model handles security workflows. Deterministic calculations and authorization remain in application code. The same-state examples make instruction sensitivity testable without confusing it with a change in underlying facts. [R3; proposed task construction]

The seven-parameter none model remains a useful ablation. Symmetric score
summaries cannot distinguish inputs with identical summaries, and changing a
none logit cannot repair candidate ordering. Feature-aware and factorized
alternatives have now been tested in Phase 4. Their incomplete transfer means
that adding semantic features or separating losses is not by itself a solution.
Retain the controls and investigate evidence/supervision before another head
sweep. [E3; E14–E20]

The factorization tested by the B2 family separates applicability from ranking:

```text
a = P(at least one offered option is valid | state, question, candidates)
r_j = P(candidate j is the correct choice | an offered option is valid,
        state, question, candidates)
P(none) = 1 − a
P(candidate j) = a × r_j, with sum_j r_j = 1
```

This factorization does not manufacture better evidence or calibration. Its value must come from supervision and representations that distinguish applicable candidates, omitted correct options, author-OOS inputs, and insufficient evidence. Keep those evaluation strata separate; application review remains a policy, not a semantic class. Specify how ambiguous or multiply valid alternatives are annotated before fitting a single-choice distribution. [R3; proposed model and annotation contract]

E30 supplies measured in-domain gains under its complete-document contract but
fails entailment/QASPER preservation. E31 executes parent-KL replay; E32 replaces
that replay loss with original source-label CE on identical inputs. Source-label
supervision improves SNLI substantially but still yields no nonzero candidate
meeting all per-class development constraints and worsens benchmark probability
scores relative to KL. Neither a frozen parent nor a better score on the replay
task establishes preservation of the intended scope. [E30–E32]

The proposed next work is a bounded complete-document training/development
failure analysis: distinguish supported-to-none from supported-to-contradiction
changes, question/hypothesis families, and evidence location before choosing one
preservation intervention. A retention panel supporting answerability or
long-document claims must exercise those operations. These are research
requirements, not a measured remedy or a newly authorized fit. Preserve all
historical selections and keep exposed QASPER/SNLI regression rows out of corrective
training and checkpoint selection. [E32; V6; §18.26]

## 11.7 Lessons from compact encoders and teacher-to-student research

**Laya: borrow a testable design, not its claimed generality.** The author model card describes a fully fine-tuned 395M ModernBERT-large backbone plus a decision head, approximately 421M parameters in total. Options are scored at marker positions within a per-question input, with a 512-token question/options/state budget. It reports multi-question batching and limits its calibration claims to the evaluated distributions. These are author descriptions, not OpenKind results or an independent checkpoint audit. [P19]

E11 already includes a bounded ModernBERT-style joint-candidate comparison.
It supplies exploratory evidence about that treatment, not a verdict on all
compact encoders or a matched evaluation of released Laya. Retain released
checkpoints as fair, explicitly unmatched comparators when relevant to the
supported workload. [E11; P19]

**R4T: the transferable idea is offline supervision for a cheap student.** Retrieve-for-Train uses an RL-trained fan-out policy to produce objective-aligned supervision for a lightweight retrieval model; the paper’s application is set-valued retrieval, not typed decision calibration. Its diffusion retriever and reported retrieval speedups are not evidence that OpenKind should switch to diffusion or will obtain the same benefit. [P20]

Only after establishing a useful supervised baseline and demonstrating teacher quality on the supported task, compare the same student trained on reviewed labels alone, labels plus reviewed teacher-generated examples, and an additional distribution-distillation treatment. Keep criteria, data splits, candidate ordering, none semantics, and budgets explicit; never replace independent final labels with teacher judgments. The 4B numerical oracle is only a possible teacher. Its inadequate answerability and policy transfer do not justify treating its probabilities as semantic truth. Evaluate student correctness, calibration, rejection, policy cost/coverage, and total resource use; agreement with an overconfident teacher is not enough. Direct supervised training remains first, and a large RL program needs a specific failure that simpler losses do not address. These are proposed OpenKind adaptations of the discussion, not results established by R4T. [R3; S3; proposal]

**Backlog traceability:** the exploratory compact comparison under 2J.1–2J.2
has run. The released-Laya comparator under 2J.5 and the conditional teacher/student
study P2.3 remain distinct. The [roadmap crosswalk](../ROADMAP.md#historical-task-crosswalk)
maps these IDs to M2–M4. Teacher quality and a measured cost problem must precede
distillation. [E11; RC; RP2]

### 11.7.1 Parallel constrained decoding and external Q-scaling evidence

Two new public implementations/benchmarks help separate the useful execution ideas from stronger architectural claims.

**Community Qwen Parallel Constrained Decoding (PCD).** The public `harshatheg/Qwen-2.5-1B-RLCD` page describes an inference engine that prefills a context plus semantic schema catalog once, broadcasts the decoder cache across fields, slices logits to valid answer tokens, performs limited continuation for multi-token candidate trees and assembles JSON in host code. Its model card reports Apple Silicon M4 Max timings of roughly 68–75 ms for four-field examples, 270 ms for 28 fields and 89 ms for one 255-choice example, with 5.6–7.0× latency reductions relative to its own autoregressive JSON baseline. Those are author-reported measurements on Qwen2.5-1.5B/MLX configurations, not OpenKind results. The public artifact does **not** document a TypeSafe-style reinforcement-learning procedure; its useful contribution here is an execution pattern. Labeling a softmax over candidate logits “calibrated” does not establish empirical calibration. [P21]

The public PCD rendering also exposes an important architectural distinction. Its shared prefix includes the semantic schema/question catalog before the context. Therefore changing, adding or reordering fields can change the cached representation that precedes every branch. That is efficient but weaker than OpenKind's intended isolation contract. The selected OpenKind profile is **state-first**: the shared state root should be independent of the set of questions submitted with it, then isolated question branches should be created from that root. The design lesson is therefore **borrow breadth-first branch batching, not schema-conditioned state representation**.

```text
Schema-first PCD reference:
    schema/question catalog + state → shared decoder cache → field branches

OpenKind selected direction:
    stable format + state → immutable hybrid root
        ├─ question batch Q1..Qn → isolated question states
        │      ├─ candidate branches → scores
        │      └─ ...
        └─ per-question rejection/distribution/policy
```

For Qwen3.5, the root and every fork must include the full hybrid continuation state, not only attention keys/values: full-attention KV, DeltaNet recurrent state, convolution state, positions and profile identity. A Rust abstraction should therefore model **branchable execution state**, not assume a generic `KVCache` is sufficient. This also keeps the runtime interface usable by future architectures whose reusable state is not represented as ordinary decoder KV. [E4–E7; E10; E11; architectural recommendation]

**DGX Spark decision-reader benchmark.** Reddy's September 2026 comparison reports whole-request p50/p95 and question throughput for one versus four questions. In that campaign, Jev 1.13 moved from **105.1 ms p50 at Q=1 to 109.2 ms at Q=4** (~1.04×), Laya moved from **16.4 to 29.1 ms** (~1.77×), while the tuned Qwen3.5 wrapper moved from **167.0 to 665.1 ms** (~3.98×). The absolute values are not apples-to-apples: local readers are warm on a DGX Spark while Jev includes a hosted HTTP round trip. The useful signal is the *within-system slope*: a Jev-style engine should make additional independent questions much cheaper than Q repeated full evaluations when shared-state work dominates. [P22]

That benchmark also supports three methodological conclusions already present in OpenKind. First, architecture rankings are task-dependent: the tuned compact encoder leads the reported WANLI slice while Jev leads the reported BoolQ slice, so a fast encoder should not be promoted from one narrow benchmark. Second, probability quality must be evaluated independently: the study reports separate Brier and ECE values rather than treating typed output or maximum probability as calibration. Third, repeatability is an observable serving property: identical Jev API requests changed some choices and many WANLI probability vectors between campaigns. This does not identify the cause, but it motivates a pinned local repeatability campaign for OpenKind rather than assuming version labels imply identical numerical behavior. [P22]

**Implication for the selected OpenKind profile.** Native CPU and pinned MLX
FP32 parity are recorded in §17.3. Matched stage replays and a Rust port of
shared-root flat-field batching now test two possible vectorized execution
graphs; neither beats the current nested batched graph on its measured shapes
(§17.4). The remaining systems comparison is complete request cost, `T(Q)/T(1)`,
marginal question latency, forward calls, prefill share, and branch/process
memory on a useful workload while retaining parity and isolation checks.
External Jev ratios and the Python Qwen2.5 benchmark do not transfer to this
Qwen3.5 result. [E11; P21; P22; RUSTM2]

High-cardinality Choice should likewise be split into systems and semantic questions. P21 shows that a constrained-token implementation can mechanically handle 255 declared choices at useful latency on its own workload; it does not show 255-way dynamic semantic accuracy. OpenKind should first stress K=32/64/128/255 for memory, batching and scheduler behavior, then add a smaller reviewed high-K semantic panel with candidate descriptions and rejection cases. [P21; recommendation]

**SemIf MLX backend (independent convergence).** The SemIf project's Apple-Silicon MLX backend independently converges on decision-native scoring with shared-state reuse, and its implementation (reviewed 2026-09-20 at pinned commit `ca3ba65f…`; see `docs/RESEARCH.md`) demonstrates the mechanism a future vectorized forward needs: native hybrid-state caches, deep-copied or merged prefix branches, right-padded suffix lanes with true lengths passed to the recurrent caches, and last-real-token readout in one batched call. SemIf executes `Qwen/Qwen3.5-4B` at revision `851bf6e…` in BF16 — not OpenKind's `Qwen3.5-4B-Base` at `1001bb4…` in FP32 — so its reported drift (5–6 changed argmaxes out of 777 between BF16 shared reuse and fresh scoring) and throughput are external implementation evidence, never OpenKind results. Its precision drift independently reinforces this paper's rule that a changed device, precision, or kernel path is a new arithmetic identity. [prior-art implementation evidence]

## 11.8 Resolved native wire mapping and direct service path

The native adapter resolves semantic none explicitly. Choice requests must include caller-supplied `__none__` with a non-empty description. That entry is not sent as a real candidate; after head evaluation, the native semantic-none mass is returned under the same key. The adapter neither appends an unrequested class nor discards and renormalizes none mass. [E2–E7; P2; R3; RUST8]

Question IDs remain hidden from inference. Real Choice options are sorted by stable option key; a null description falls back to that key as semantic text. Noul keeps its confidence-free Jev shape. Choice and Score confidence use normalized distribution entropy, not maximum probability. These are explicit adapter semantics, not inferred equivalence to an undocumented vendor statistic. [P2; R3; RUST8]

The implemented service path is:

```text
Rust HTTP/gRPC boundary → EngineRegistry → bounded Qwen35DecisionEngine
                       → native tokenizer/backbone/head → validated typed response
```

This direct adapter loads only explicit offline artifacts, defaults the current per-lane CPU backend to sequential sharing, bounds running plus queued requests, maps overload separately from invalid and backend failures, and fails unsupported shapes explicitly. The Python implementation remains a differential oracle. Cancellation/recovery and queue-inclusive CPU load/soak now have bounded named-machine evidence. MLX service qualification remains separate. [RUST8; RUST11; §17.3]

Protected metrics, redaction of sensitive inputs, bounded request work, and explicit opt-in before unauthenticated non-loopback serving remain deployment requirements. Direct registration is not load/soak evidence, an audit, release promotion, or Metal parity. [R3; recommendation]

## 11.9 One current contract and status authority

The recovered review reported drift across agent instructions, roadmap phases, probability terminology and test-count snapshots. Version 0.5.1 synchronizes **this paper and the attached roadmap**, but does not edit or audit `AGENTS.md`, architecture files, Rust source or CI. The roadmap is the current task/status authority; this paper retains experiment interpretation and provenance. Repository instructions must adopt the same task IDs and supported contracts under Track S.1. Use commit-stamped CI results for current test counts rather than duplicated “at HEAD” prose. The supplied roadmap's 195-test snapshot has no exact commit and was not rerun here. [R3; R4; V1]

H is completed through its separately identified continuation. The native
CPU adapter, persistence replay, and named-machine service gates are also
recorded. [ROADMAP.md](../ROADMAP.md) now owns M0–M4, while
[ROADMAP_HISTORY.md](../ROADMAP_HISTORY.md) preserves prior phase IDs and their
evidence register. Current test claims require commit-stamped verification.
Candidate probability remains distinct from confidence. This v0.8.5 update records
completed model evidence through E32 and the proposed evidence-sensitive
preservation diagnosis. It creates no new notebook and does not claim that the
repository roadmap, agent instructions or architecture files were edited or
synchronized. [E8; E9; E29–E32; V6; RUST11]

# 12. Research questions answered and still open

| Research question | Answer supported so far |
|---|---|
| Must useful decisions be generated as text? | No. Frozen features plus small heads work on the measured tasks. |
| Does a base model suffice? | It suffices for bounded probes. E29 now compares Base/post-trained checkpoints under a common renderer; E30 compares matched frozen/adapted BF16 controls. Effects depend on task and neither adapted model passes full retention. This is not an independently optimized native-prompt comparison. [E29; E30] |
| Is a linear head enough? | It is a strong selected NLI baseline. More complex heads were not consistently necessary in this setting. |
| Do unseen candidate labels work? | Yes within bounded tests, but H’s preselected support arm has substantially weaker held-out than fitting-family results. Held-out omission remains distinct from the all-correct sampled author-OOS panel. [E9] |
| Is calibration automatic after supervised fitting? | No. Post-hoc scaling can worsen held-out metrics, and risk changes under new costs and prevalence. |
| Has 4B joint-option LoRA learned the targeted natural-document task? | Yes within E30's pilot: ContractNLI accuracy rises to 81.37%/82.35%, with better contradiction/none decisions. Entailment and QASPER retention fail, so the result is task specialization rather than a complete M2.2 pass. [E30] |
| Do larger inputs and lower development NLL ensure preservation? | No. E29's fixed evidence-visible subsets improve with budget, but E30 loses some of that behavior; NLL-selected update 80 already loses development entailment recall. Preservation needs an explicit test. [E29; E30; V4] |
| Can a tiny none model help? | Yes in controlled comparisons. H now adds final refitting/readout evidence, but score-summary rejection does not settle applicability, missing evidence or policy risk. [E3; E9] |
| Does an isolated repeat prove deterministic deployment? | No. Padding, batching, precision, and cache chunking changed results. |
| Is FP32 the most accurate model? | Not established generally. It is the most internally consistent tested reference; numerical and semantic quality are evaluated separately. |
| Does TF32 permission preserve the decision contract? | Not fully. H’s selected profile has no final argmax/policy changes but two numeric-tolerance failures; additional controlled-context cross-mode failures remain. Historical G failures are unchanged. [E7; E9; V2] |
| Can shared prefixes amortize work? | Yes, especially with longer prefixes and batched suffixes in the completed mechanical tests. |
| Has state-once, arbitrary-question-many execution been demonstrated? | **Mechanically, in bounded Python and native Rust systems tests:** E11/E12 validate the reference graph; RUST5/RUST6 reproduce sequential and lane-topology Q/K execution exactly for the frozen fixtures. The CPU evidence establishes lane topology. Separate MLX FP32 fixtures now cover vectorized forward. Neither establishes arbitrary-question semantic competence. [E11; E12; RUST5; RUST6; §17.3] |
| Is TurboQuant already a win for this model? | Not under the frozen equivalence gate: all four tested snapshot codecs failed; short-prefix overhead also limited storage benefit. |
| Does FP16 KV storage help? | It passed strict-FP32 fresh/context gates in G and earlier storage tests; it saved bytes but did not add a cache hit or beat lossless trace time in G. |
| Does persistent prefix reuse help? | Yes on controlled single-worker traces: F and G show workload-dependent gains, with G exercising expiry; no production concurrency claim follows. |
| Are clear criteria and calibrated rejection solved? | No. H’s completed support/refitting study still has held-out omission, context and policy limitations; independent criteria review was not performed. [E9] |
| Does MTP speed up this decision path? | Not in its present no-output-decoding graph. |
| Are Score, Noul, and API parity finished? | H’s bounded BoolQ and SST-5 final probes are complete. The native service and wire mapping are implemented. General binary/rubric semantics and insufficient-evidence competence remain open. [E9; RUST8; RUST11] |
| Has Rust/Metal or a smaller Qwen been validated? | Rust CPU full-backbone and cached-continuation parity pass for the selected 4B profile. Pinned MLX FP32 full/nested/vectorized parity also passes. MLX service promotion and reviewed smaller-model quality remain open. H disabled smaller Qwen, but E11 later tested it exploratorily. [E11; §17.3] |
| Has Jev’s RLCD been reproduced? | No. The project has a distinct, inspectable research path toward a similar software interface. |
| Is the frozen 4B scorer the required deployment model? | No. It is the measured reference; evidence repair and useful decision quality precede another cost challenger. |
| Must a newly trained model reproduce all old probabilities? | No. Same-model implementation equivalence and fresh new-model quality have separate gates. |
| Does a batched multi-question call prove state-once computation? | No. Q-scaling, actual shared work, isolation, and semantic quality must be measured separately. |
| Is there now a locked natural-document multi-question benchmark? | Yes, within the declared Phase 4A scope: ContractNLI and QASPER contribute 2,192 states and 15,368 questions under state/component-safe partitions, with hashes and model-feature identities locked before predictions. This is a benchmark asset, not proof of label quality or deployment representativeness. [E14] |
| Does the original StateQuery B1 solve multi-source applicability? | No. It materially improves the historical reference on aggregate non-final metrics, but QASPER balanced accuracy is 0.5 and semantic-none recall is 0.0. [E14] |
| Does the matched B0/B1 comparison support a query-module architecture gain? | Yes, narrowly. Balanced B1 improves source-macro balanced accuracy from 0.5025 to 0.5679 and macro F1 from 0.4952 to 0.5711 versus B0, while QASPER none recall rises from 0.0 to 0.10. It still does not satisfy the QASPER applicability gate. [E15] |
| Did source/class-stratified applicability solve QASPER none discrimination? | Partly, not sufficiently. At its frozen operating threshold it reaches 0.2564 recall on policy development and 0.3111 on the gate, but the development floor is 0.30 and the transferred gate policy costs 0.11675 versus 0.10 for review-all. [E17] |
| Did pairwise applicability transfer after passing development? | No. It reaches 0.3451 QASPER development recall, but gate false-none rates are 0.2111 for ContractNLI and 0.2110 for QASPER, both above the 0.20 ceiling; gate policy cost is 0.10221. [E18] |
| Can an applicability-head-only or small evidence-residual continuation rescue the frozen representation? | Not in the tested forms. The best head-only child worsens development NLL/Brier by 2.77%/2.10%. The best evidence-residual child gains one QASPER none true positive, 34 to 35 of 113, while worsening NLL/Brier by 6.08%/6.60%. [E19; E20] |
| May the Phase 4 final split be opened now? | No. Every Phase 4 follow-on is non-final, comparison-only or failed; no checkpoint clears the complete applicability, false-none, proper-score, and transferred-policy gates. Final remains unavailable and unopened. [E14–E20] |
| Does the public Qwen `RLCD` repository reproduce TypeSafe RLCD training? | Not from the reviewed public artifact. Its useful evidence is parallel constrained-decoding/cache-broadcast execution; a softmax over candidate logits is not by itself a calibration study. [P21] |
| What new systems property should OpenKind target after nested sharing? | **Adaptive question amortization.** E12 shows sharing is workload-dependent: short semantic requests can lose to repeated-full, while long shared-state mechanics gain strongly. Preserve multiple execution strategies and learn the target-machine crossover instead of assuming one universally fast path. [E12; P22] |
| Should OpenKind switch to schema-first PCD? | No. Keep the selected state-first root for stronger question-set isolation; borrow breadth-first branch batching and constrained-token baselines where useful. [P21; E11] |
| Does Laya establish a better replacement, or R4T establish a diffusion decision architecture? | No. They motivate bounded comparisons and a later distillation study, not completed OpenKind findings. |
| Has Phase 2H completed? | Yes: both required `2h.1.2` continuation workers completed and 2H-C1–C5 are closed. The original `2h.1.1` failed attempt is unchanged; completion does not promote a model or arithmetic mode. [E8; E9] |
| Does preservation of the frozen parent imply preservation of useful correctness? | No. E31 partly recovers benchmark probabilities but fails joint retention; E32 then exceeds parent-KL SNLI performance without preserving supported entailment or QASPER. Parent agreement, replay-task learning and retained scope are separate. [E31; E32] |
| Did the source-label replay fit run, and can its fixed80 adapters be selected as upgrades? | Both fits complete 120 updates, but no evaluated nonzero candidate meets all development conditions. J6_selected = J0 and J7_selected = J1; the fixed80 results are diagnostics, not alternative selected winners. [E32; V6] |

**Current 2I/2J status:** the original `2ij.1.0` reviewed-study path remains blocked by unsigned review, null promotion bounds and stale review metadata; separately, `2ij.2.0` completed an **exploratory** model-selection screen and exported a provisional integration profile. The pilot is sufficient to start native parity work but does not satisfy the independent-review/natural-data release gate. [E10; E11; I0]

**Current Phase 4/model status:** bounded frozen and adaptation diagnostics now
include E29–E32. All three tested adaptation recipes miss the full retention exit.
E32 demonstrates better short-premise SNLI learning but not a preserved complete
scope; both source-label selectors retain frozen0. The complete-document training
contract does not imply independent semantic correction or source-PDF equivalence.
Existing labels, quarantines, thresholds, and result locks remain unchanged.
The proposed next diagnosis targets supported-entailment and long-document
retention, not another automatically authorized fit. [E14–E32; §§18.20–18.26]

# 13. Refocused research program and next milestone

The objective remains useful, auditable decisions over shared evidence. Native
CPU and pinned-base MLX FP32 qualifications are bounded systems results. M2.1,
the contract-only M2.2 pilot, parent-KL replay, and source-label replay are now
completed model experiments; none of the adaptation recipes meets the full
quality/retention exit. The source-label comparison establishes stronger SNLI
learning, not a general retention solution. [E11–E32; §17.3; §18.25]

The next proposed work is a bounded diagnosis of the recurring supported-entailment
regression on admissible complete-document training/development records, followed
by one attributable preservation treatment and an appropriately scoped retention
panel. No particular corrective loss or new dataset is validated here. Evidence
coverage, semantic use, behavioral agreement, replay-task correctness, and retained
scope remain separate. The historical interpretations in §§18.19/18.22/18.24 are
retained; §§18.25–18.26 supply the completed result and current synthesis. The
roadmap owns task management and is not modified by this documentation-only update.
[E32; V6; proposed work]

## 13.1 Phase 2H: completed continuation and retained development history

**Current status: required evaluation completed through E9.** Sections 13.1.1–13.1.7 retain the original E8 fitting/development record and the failed-attempt boundary that was correct through v0.5.2. Sections 13.1.8–13.1.16 add the completed `2h.1.2` continuation, final results, acceptance limits and closeout. Do not read the historical “not evaluated” labels below as the current status; they identify what existed in the unchanged original attempt. [E8; E9]

### 13.1.1 Original attempt: what ran, what was saved, and what did not run

The recorded traceback points to `phase2h_worker.py:247`:

```python
report['memory'].append(memory_snapshot('after_stage')); report['status'] = 'completed'
# NameError: name 'memory_snapshot' is not defined
```

The archived source calls this line after `fit_all(...)` returns and after the sampled backbone-integrity check. `profiles.json`, `selection.json`, eight candidate-scorer files (six trained and two frozen controls), six primitive-head files, and selected-head development fixtures are present. The missing name is a reporting/finalization defect in this notebook, not evidence that the fitted model failed its task. Nevertheless, the worker's status remains `failed` and the coordinator's status remains `partial`; neither is silently promoted to completed. The recorded error is nonfatal to CUDA (`fatal_cuda: false`), unlike the earlier device-side-assert failure. [E8; archived traceback, worker source and artifact inventory]

| Component | Recorded outcome in this attempt | Evidence boundary |
|---|---|---|
| Data preparation and criteria freeze | Completed; split and support manifests saved | Reservations and exact-group checks, not semantic validation |
| Text-backbone load and feature extraction | L4, strict FP32; finite forward smoke check; fitting outputs saved | No new H serving-performance result |
| Main candidate/rejection study | 14 development profiles; six joint candidate-head fits across two criteria variants and three seeds | Development/selection evidence, not final quality |
| Calibration and application policies | 12 main-profile temperature gates and 126 selected policy records saved | Calibration-gate and policy-development measurements only |
| BoolQ and SST-5 probes | Three fitted seeds per task and selected seeds saved | No final primitive accuracy, calibration, or ordinal-error result |
| Final, TF32, robustness and request tests | Enabled in configuration, but downstream of the failed fit-worker completion gate | Not executed in the archived attempt |
| State-first, Qwen3.5-2B and LoRA | Disabled in this run | No result follows from their implementation or default availability |
| Independent criteria review and natural documents | Not performed / none supplied | Support examples are not reviewed definitions; generated contexts are not natural-document evidence |

The worker records the pinned text model `Qwen/Qwen3.5-4B-Base` at revision `1001bb4d826a52d1f399e183466143f4da7b741b`, 4,205,751,296 parameters and 2,560 hidden dimensions, without generation or CPU offload. Its post-load allocation is approximately 16,043.7 MiB. That snapshot is not the peak for fitting, and the missing end-of-stage snapshot cannot be replaced with an assumed value. The sampled base-weight check reports unchanged values; it is not a full hash of every weight. [E8]

### 13.1.2 Original reserved study partitions and decision contracts

The executed standard configuration uses data seed `20260923`, head seeds 17/29/43, fitting candidate counts K = 4/8, and reserved final counts K = 4/16. Main input length is capped at 512 tokens; the configured longer-context cap is 1,792, but those reliability tests did not run. Prior C/D/E/G exact normalized message groups are excluded. This is neither paraphrase removal nor pretraining decontamination. [E8]

| Main partition | Banking fitting labels | CLINC fitting domains | Author OOS | Other held-out groups | Total messages | Episodes |
|---|---:|---:|---:|---:|---:|---:|
| Training | 256 | 256 | 40 | 0 | 552 | 1,064 |
| Development | 64 | 64 | 12 | 0 | 140 | 268 |
| Calibration fit | 64 | 64 | 12 | 0 | 140 | 268 |
| Calibration gate | 48 | 48 | 12 | 0 | 108 | 204 |
| Policy development | 64 | 64 | 12 | 0 | 140 | 268 |
| Reserved final — not evaluated | 64 | 64 | 64 | 128 | 320 | 1,152 |

Source: E8 split manifest and data audit. The final row's other held-out groups are 64 Banking held-out-label messages and 64 CLINC held-out-domain messages. Message counts, repeated episodes and stage completion must not be conflated. Final counts describe reserved inputs, not tested predictions.

There are 294 separately reserved criteria-support messages. The preparation report excludes 3,024 prior project groups and records 3,102 new reserved groups across the main study, support and primitive reservations; 3,102 is not the size of the dynamic-choice training set. The reviewed main split manifest has zero pairwise exact-group overlap. The original 57/20 Banking label partition is retained. Six CLINC domains supply fitting labels: auto/commute, home, kitchen/dining, meta, small talk and utility. Travel and work are withheld from fitting candidates and support examples. The final held-out transfer tests remain unexecuted. [E8]

The objective assigns total mass 60% to answer-present episodes, 25% to deliberately omitted in-scope intents, and 15% to author-OOS cases, with equal message weight within each origin and equal weight across that message's variants. These are declared training/selection assumptions. For example, the raw development data contain 128 present, 128 omitted and 12 OOS episodes; the reported weighted NLL is not their unweighted average. All main fitting/calibration/policy partitions use source-training data, except the documented CLINC author-OOS auxiliary partitions drawn from `oos_val`; final data are reserved from source test/`oos_test`. [E8; archived `weights` and `objective` definitions]

Two criteria variants actually ran: the original terse descriptions and descriptions with reserved author-training support examples. The latter adds labeled information; it is not a pure wording repair. No independent criteria review was performed, and held-out labels received no support examples. The historical source labels, G error cases and earlier results are unchanged. [E8]

### 13.1.3 Development comparison: adaptation helps this selection set; examples alone do not

The table retains every main development profile. Values are **pre-temperature, message/origin-weighted development NLL**, not accuracy and not final-test NLL. Lower values are favorable only within this declared comparison. All models use instruction-first rendering and a frozen Qwen backbone. [E8]

| Treatment | Original criteria | Support-example criteria |
|---|---:|---:|
| Historical candidate scorer + inherited set-linear none | 0.705098 | 0.761534 |
| Historical candidate scorer + newly selected none | 0.546988 | 0.610291 |
| Joint candidate-head fit, seed 17 + newly selected none | 0.429585 | 0.312515 |
| Joint candidate-head fit, seed 29 + newly selected none | 0.421695 | 0.298156 |
| Joint candidate-head fit, seed 43 + newly selected none | 0.426548 | 0.297705 |
| Lexical description scorer + newly selected none | 1.033746 | 0.794191 |
| One-step finite-code vocabulary baseline | 0.595000 | 0.764155 |

Source: E8, `profiles.json`, `selection.json`, and compact `profile_dev_nll`; values reconciled across saved copies for v0.5. The inherited none control is the saved **set-linear** model, not the original C global scalar. The affine fits update 2,562 trainable readout parameters, including a scalar none term during joint training, and subsequently select a separate refitted-none model. Their initialization is the historical 4B scorer; they are not fresh random-head or LoRA treatments. Backbone feature extraction is reused across head optimization. [E8]

The selected joint profile is `support_examples__instruction_first__joint_seed43__refit_none`, with development NLL 0.297704818. The selector is restricted to the six fitted joint-head profiles and uses development NLL with a deterministic name tie-break; it is not a final-test search or a universal selection across every possible model family. Seed 29 under the same criteria is close at 0.298156370. That small gap is not evidence that seed 43 will generally perform better. Three seeds on one selected development set do not supply three independent task-population replications. [E8]

The pattern supports three limited observations. Refitting rejection reduces original-criteria development NLL from 0.705098 to 0.546988. Updating the candidate head as well reduces it further, to 0.421695 for the best original-criteria joint profile. Support examples worsen both frozen affine controls, yet every support-example joint fit has lower development NLL than its original-criteria seed counterpart. Thus the apparent benefit of support depends on adapting the readout in this experiment; the evidence does not establish extra examples as an automatic improvement for the frozen model. The finite-code control also worsens under support, while the lexical control improves. These observations motivate the reserved final comparison, not a claim that H has resolved rejection transfer or G's physical-card ambiguity. [E8; within-development interpretation]

The one-parameter none alternatives remain important for attribution:

| Candidate scorer / criteria | Refitted constant none: development NLL | Set-conditioned none: development NLL |
|---|---:|---:|
| Historical scorer / original | 0.668111 | 0.546988 |
| Historical scorer / support examples | 0.646117 | 0.610291 |
| Joint seed 29 / original | 0.457600 | 0.421695 |
| Joint seed 43 / support examples | 0.349783 | 0.297705 |

Source: E8 saved `none_selection.alternatives`; the two joint rows are the development-selected seed within each criteria variant, not newly selected on final results. Both none alternatives use the same candidate scores within a row. They have one and seven fitted coefficients, respectively. These are H development comparisons and must not replace D's separate final-test ablation table.

No raw full-development logits or feature cache is retained in this ZIP. This revision reconciles the saved NLL values and selection arithmetic but does not independently reconstruct all 14 losses from predictions or rerun feature extraction. The two retained development-head fixtures are sufficient for a limited algebra check, not for reproducing the complete development benchmark. [E8; v0.5 audit boundary]

### 13.1.4 Original calibration and policy selection, before final evaluation

All eight newly calibrated affine profiles retain temperature 1.0 under their separate calibration gate; the two unchanged affine controls also keep their inherited temperature of 1.0 without a new fit. For the selected support/seed43 profile, fitting proposes T = 1.07802455, but gate NLL changes from 0.29473706 to 0.29719335. The transform is rejected under the required 0.002 improvement. The two lexical and two finite-code profiles instead select non-unit temperatures under their own gates. This is evidence of selection behavior, not proof of calibration on reserved final messages. Main development-profile ranking uses pre-temperature NLL. [E8]

Nine application policies are saved for each of the 14 profiles: 126 policy-selection records. The selected joint profile's records are:

| Assumed absent prior | Wrong-answer cost | Selected threshold | Scenario-weighted policy-development coverage | Scenario-weighted policy-development cost | Wrong / accepted episodes, raw |
|---|---:|---:|---:|---:|---:|
| 5% | 1 | 0.80 | 78.01% | 0.022773 | 4 / 109 |
| 5% | 5 | 0.80 | 78.01% | 0.025898 | 4 / 109 |
| 5% | 20 | 0.90 | 72.01% | 0.031895 | 1 / 98 |
| 25% | 1 | 0.80 | 61.91% | 0.041992 | 4 / 109 |
| 25% | 5 | 0.90 | 56.93% | 0.047949 | 1 / 98 |
| 25% | 20 | 0.95 | 50.39% | 0.049609 | 0 / 86 |
| 50% | 1 | 0.90 | 38.09% | 0.063867 | 1 / 98 |
| 50% | 5 | 0.95 | 33.59% | 0.066406 | 0 / 86 |
| 50% | 20 | 0.95 | 33.59% | 0.066406 | 0 / 86 |

Source: E8 selected profile, `policies`, on the 140-message/268-episode policy-development partition. Review cost is 0.1, and author-OOS is assumed to supply half of the absent mass. Weighted coverage/cost and raw accepted counts use different denominators. These thresholds were chosen on the displayed policy-development population; the rows are not held-out policy performance or measured deployment savings. Zero wrong answers among 86 selected-development acceptances is not a rare-error guarantee. G's policy-transfer failures remain relevant and unchanged. [E7; E8]

Each treatment chooses its own thresholds. Consequently, the frozen-control arm freezes its probability model but does not freeze the entire historical application policy. Any future comparison should distinguish changed model probabilities from changed policy selection, preserve omission versus author-OOS strata, and retain accepted-error and coverage denominators. H's pooled development NLL cannot establish separate held-out-domain answerability or OOS success. [E8; interpretation]

### 13.1.5 Primitive heads fitted before the failure

The complete worker report retains results omitted from the compact paste-back profile table. BoolQ and SST-5 each use 384 training examples, 64 development examples, 64 calibration-fit examples and 64 calibration-gate examples in the fitting payload. Three linear readout seeds were fitted for each task, with the lowest development NLL selecting seed 29 in both. [E8]

| Primitive probe | Seed 17 development NLL | Seed 29 development NLL | Seed 43 development NLL | Selected seed | Selected temperature |
|---|---:|---:|---:|---:|---:|
| BoolQ yes/no readout | 0.324775 | 0.260388 | 0.328890 | 29 | 1.000000 |
| SST-5 five-level sentiment readout | 1.231912 | 1.177823 | 1.196803 | 29 | 1.139008 |

Source: E8 `selection.json` and `primitive_profiles`; losses are the unscaled development cross-entropies recorded for the selected checkpoint of each seed. They are not the origin-weighted dynamic-choice objective and should not be numerically ranked against the preceding intent tables. The dataset revisions are recorded in Appendix A. Six fitted weight files and their selection histories exist; final primitive metrics do not.

This moves H's primitive work from design-only to **fitted, not finally evaluated**. BoolQ supplies a bounded false/true distribution and P(yes), without dedicated insufficient-evidence labels. SST-5 supplies a five-level sentiment distribution and expected level, not arbitrary rubric semantics or a target soft-label distribution. No final accuracy, Brier score, calibration verdict, ordinal error or primitive latency follows from the development losses alone. Do not label the Noul/Score API generally validated. [E8]

### 13.1.6 Historical recovery requirements and their disposition

The partial study makes the head-adaptation comparison concrete without changing the strategic destination in Sections 11–13. It supplies a development-selected artifact worth evaluating and shows that the support, scorer and rejection interventions interact. It does not establish a production model, a head-family optimum, generalized question following, a smaller-model winner, a LoRA gain, or a correction of historical semantic errors. No selected H model has a new TF32/equivalence verdict or a measured complete-request quality/resource trade-off yet. [E8; bounded interpretation]

The recorded recovery requirement was to repair the undefined memory-reporting name and validate reuse against the saved artifacts, original run, reservations and selection rules. E9 now supplies that continuation without marking the original failed worker completed, deleting reservations or selecting a new split. The source repair and recovery validation are versioned separately; this documentation update did not execute or modify them. [E8; E9; HR]

The originally required final-lock and final-evaluation workflow is now complete in E9. All retained treatments, family and rejection strata, policy/calibration metrics, reliability, primitive results, arithmetic comparisons and request records are available. The absence of these artifacts in E8 remains historical; it is no longer the current study status. The final outcomes must not be reused as fresh selection data for the next intervention. [E8; E9]

The one-step vocabulary baseline remains unadapted and uses a different rendering and gradient budget from the learned candidate readout. H's completed continuation retains disabled LoRA, smaller-Qwen and state-first arms. Future LoRA treatment still needs its matched online-head BCE control, and future generalization/service work retains its separate task contract. [H0; E8; E9; R3]


### 13.1.7 Historical `study.finish()` output versus the completed continuation

The earlier E8 Colab cell printed `Overall execution status: partial`, correctly describing its failed fitting-worker finalization and missing final outputs. The E9 Finish notebook now records `OVERALL EXECUTION STATUS: COMPLETED` with both required evaluation workers complete, accompanied by the final lock and prediction files. The two statements describe different preserved attempts; neither generic open-question text nor ZIP export alone determines the status. Use E9 for current completion and E8 for historical failure/fitting provenance. [E8; E9; V2]

Consequently, preserve the fitted/development evidence, recover or explicitly close the original attempt, and keep enabled-but-unexecuted evaluation distinct from disabled optional arms and independently unimplemented service/model work. The source recheck confirms the v0.5 interpretation rather than adding a new completed experiment. The five operational closeout tasks are **2H-C1–C5** in the roadmap; their completion requires new evidence and is not claimed here. [E8; R4; R5; V1]

### 13.1.8 Completed continuation: lineage, lock and evaluated scope

**Execution is complete.** The `2h.1.2` continuation `20260919T040612625670Z__finish_2h_1_2` completed both `eval_qwen4b_strict` and `eval_qwen4b_tf32`, each with exit code zero. Its final reports and archive were saved to Drive around **14:45 UTC on 19 September 2026**. The original `2h.1.1` attempt remains partial and its failed fitting-worker report remains unchanged. Recovery is a separate recorded status, `recovered_artifacts_validated`; it is not a retroactive assertion that the failed worker succeeded. [E8; E9]

The continuation reused the original reservation, criteria, fitted weights, normalizers, rejection models, temperatures, policies and selection. Its recovery report records 14 profiles, eight affine weight files, six primitive weight files, 126 saved policies, and two development-head fixtures, with no retraining or calibration recomputation. The final lock binds the final and fitting payloads, frozen criteria, fitted files, runtime-source files, execution code and configuration. The saved lock precedes the evaluation outputs. The v0.6 audit confirms that all 24 locked fitting-directory files match the original attempt byte-for-byte, that the original ZIP retains its recorded SHA-256, and that the final payload, split manifest and criteria are unchanged. These checks establish artifact lineage, not an independently enforceable guarantee about all human access to the earlier final data. [E9; V2]

The evaluated model remains `Qwen/Qwen3.5-4B-Base` at revision `1001bb4d826a52d1f399e183466143f4da7b741b`, with a frozen text backbone and instruction-first candidate rendering. Both workers record **NVIDIA L4, PyTorch 2.11.0+cu128, Transformers 5.17.0 and Python 3.13.15**. Strict FP32 and TF32-permitted execution retain FP32 weights; they are not a model-weight-quantization comparison. The recorded sampled weight-integrity checks pass, but are not full-backbone hashes. [E9]

The final Choice study contains **320 distinct source-message groups and 1,152 episodes**. Four in-scope families each contribute 64 messages expanded into 256 episodes: answer-present and omitted-intent variants at K = 4 and 16. The 64 author-OOS messages contribute 128 episodes, with no fabricated positive-answer counterpart. Every one of the 14 retained profiles is evaluated in both modes, yielding 32,256 saved probability rows over the **same** 1,152 episodes. These are not 32,256 independent examples. The 57/20 Banking label partition and CLINC travel/work holdouts remain as documented in §13.1.2. All final outcomes retain author labels; this is still sampled dynamic Choice, not a standard full-label Banking77 or CLINC leaderboard result. [E9]

The following required work now has completed outputs: final profile comparisons, frozen-policy evaluation, within-mode parity, controlled reliability, in-process request benchmarks, and all six bounded primitive readouts in each mode. State-first, smaller-Qwen and LoRA were disabled; independent criteria review was not performed; no natural-document input set was supplied. Those omissions are explicit scope boundaries, not missing runs to silently add to the original experiment. [E9]

### 13.1.9 Selected model: final quality, rejection and population weighting

The model selected **before final evaluation** remains:

```text
support_examples__instruction_first__joint_seed43__refit_none
```

Its temperature remains 1.0. Both modes have the same selected-model argmax on every final episode, hence identical accuracy in the following table; their probability values and NLLs are not identical. [E9]

| Final family | Messages | Episodes | Accuracy, both modes | Strict-FP32 NLL | TF32-permitted NLL |
|---|---|---|---|---|---|
| Banking, fitting labels | 64 | 256 | 85.55% | 0.472064 | 0.472205 |
| Banking, held-out labels | 64 | 256 | 63.67% | 1.225458 | 1.225508 |
| CLINC, fitting domains | 64 | 256 | 89.84% | 0.354001 | 0.354366 |
| CLINC, held-out travel/work | 64 | 256 | 66.02% | 1.182342 | 1.182467 |
| CLINC, author OOS | 64 | 128 | 100.00% | 0.029644 | 0.029703 |
| Pooled raw panel | 320 | 1152 | 78.91% | 0.721931 | 0.722088 |

Source: E9 `worker_results.<worker>.final_metrics.<selected_profile>.<family>.quality`; reproduced from saved vectors by V2. The pooled line is **909 correct out of 1,152 episodes**. Its source key is `pooled_declared_mixture`, but its `quality.accuracy`, `quality.nll`, Brier and ECE fields are **raw episode averages**, not 60/25/15 population-weighted metrics. The actual raw panel has 512 answer-present, 512 omitted-intent and 128 author-OOS episodes. The separately named `assumption_weighted_nll` applies the declared 60/25/15 origin weights and equals **0.631972** for strict FP32, rather than the raw NLL **0.721931**. This distinction follows the archived `quality()` and `weights()` definitions; it does not rename or change the source result. [E9; V2]

The selected model is stronger on the sampled fitting-label/domain families than on the really withheld label/domain families. It correctly rejects all 128 author-OOS episodes from 64 messages, but this does not settle deliberately omitted in-scope choices or insufficient evidence. Separate conditional metrics expose that limit. Each row below contains 128 answer-present and 128 omitted-intent episodes from the same 64 source messages. [E9]

| In-scope family | Answerable accuracy | Omitted-intent none recall | False-none on answerable | Wrong offered answer / all answerable |
|---|---|---|---|---|
| Banking, fitting labels | 85.94% | 85.16% | 12.50% | 1.56% |
| Banking, held-out labels | 78.91% | 48.44% | 13.28% | 7.81% |
| CLINC, fitting domains | 92.19% | 87.50% | 7.03% | 0.78% |
| CLINC, held-out travel/work | 69.53% | 62.50% | 22.66% | 7.81% |

Source: E9 `quality.conditional`, strict FP32. In the last column, the denominator is all answer-present episodes, not only accepted answers. In particular, Banking held-out labels produce **62/128 correct none outcomes and 66/128 wrong offered outcomes when the correct intent is omitted**. CLINC held-out domains produce 80/128 correct none outcomes and 48/128 wrong offered outcomes in that stratum. The 100% author-OOS result therefore cannot be described as a universal answerability or rejection result. [E9; V2]

Probability diagnostics and the saved message-cluster accuracy intervals are:

| Family | Brier, sum over classes | Top-label ECE, 15 bins | Assumption-weighted NLL | Recorded 95% accuracy interval |
|---|---|---|---|---|
| Banking, fitting labels | 0.219365 | 0.036262 | 0.475318 | 80.26–89.84% |
| Banking, held-out labels | 0.534282 | 0.100286 | 1.019521 | 57.81–69.53% |
| CLINC, fitting domains | 0.173418 | 0.038471 | 0.320725 | 85.55–93.96% |
| CLINC, held-out travel/work | 0.495822 | 0.115266 | 1.137495 | 60.34–71.48% |
| CLINC, author OOS | 0.005957 | 0.027233 | 0.029644 | 100.00–100.00% |
| Pooled raw panel | 0.316859 | 0.047653 | 0.631972 | 76.18–81.54% |

Source: E9; V2 recomputes point metrics and the specified accuracy-bootstrap calculation. The intervals condition on the sampled data and grouping; they do not cover pretraining contamination or all training/model-selection uncertainty. The all-correct OOS bootstrap degenerates to 100–100% because every resampled observed group is correct. That interval is **not** a zero-risk guarantee for future OOS inputs. The lower pooled ECE also does not erase the larger held-out-family ECEs or the application-policy errors below. [E9; interpretation]

For candidate-count sensitivity, an additional saved-row aggregation separates K without changing the selected model:

| Family | K=4 accuracy | K=4 NLL | K=16 accuracy | K=16 NLL |
|---|---|---|---|---|
| Banking, fitting labels | 84.38% | 0.446761 | 86.72% | 0.497367 |
| Banking, held-out labels | 50.78% | 1.313991 | 76.56% | 1.136926 |
| CLINC, fitting domains | 91.41% | 0.330826 | 88.28% | 0.377177 |
| CLINC, held-out travel/work | 60.94% | 1.165090 | 71.09% | 1.199593 |
| CLINC, author OOS | 100.00% | 0.043940 | 100.00% | 0.015348 |
| Pooled raw panel | 75.00% | 0.728586 | 82.81% | 0.715275 |

Source: V2 aggregation of E9 final rows; each K contains 576 episodes from the same 320 messages, including 128 episodes per in-scope family and 64 OOS episodes. These are the recorded candidate-set constructions, not proof that a larger K universally improves decisions. Rejection behavior, distractor composition and candidate-set conditioning can change together. No independent-question Q experiment was enabled. [E9; V2; interpretation]

### 13.1.10 All retained comparisons: the development winner is not the best final-transfer arm

The full final readout must accompany the selected-model table. These are paired comparisons on the same final episodes, with original versus support-example criteria and the predefined retained baselines. They are **post-calibration raw final metrics**, unlike the pre-temperature, origin-weighted development losses in §13.1.3. [E8; E9]

| Treatment | Original accuracy | Original NLL | Support accuracy | Support NLL |
|---|---|---|---|---|
| Historical scorer + inherited set-linear none | 68.32% | 0.937535 | 65.19% | 0.987117 |
| Historical scorer + refitted none | 78.65% | 0.619760 | 77.52% | 0.652057 |
| Joint seed 17 + refitted none | 83.59% | 0.547076 | 80.64% | 0.672946 |
| Joint seed 29 + refitted none | 83.77% | 0.541440 | 80.47% | 0.664284 |
| Joint seed 43 + refitted none (support arm preselected) | 83.59% | 0.546304 | 78.91% | 0.721931 |
| Lexical control | 69.10% | 1.072799 | 71.18% | 1.027965 |
| Unadapted finite-code control | 72.22% | 0.872400 | 76.39% | 0.803906 |

Source: E9 strict-FP32 final metrics for all 14 profiles. The selected support/seed43 profile improves raw pooled accuracy over the original frozen inherited-none control by **10.59 percentage points** (68.32% to 78.91%) and reduces NLL from 0.937535 to 0.721931. The saved paired message-bootstrap contrast is +0.105903 accuracy, with interval **+0.070982 to +0.133960**, and −0.215604 NLL, with interval **−0.298437 to −0.114922**. V2 reproduces these contrasts; they are within-H comparisons, not changes from G's different final population. [E9; V2]

However, the **original-criteria joint seed29** profile, already the development-selected seed within the original-criteria variant, records **83.77% pooled accuracy and 0.541440 NLL**, versus the global preselected support/seed43 model's 78.91% and 0.721931. All three original-criteria joint fits outperform their same-seed support-example counterparts on pooled final accuracy and NLL, reversing the corresponding development-loss pattern. None of this permits selecting a new default on the same exposed final data and calling that selection independently validated. The original global selection is retained, and the original-criteria arm becomes a control for the next separately specified study. [E8; E9; interpretation]

The trade-off is clearer by family:

| Family | Original joint seed29 accuracy | Selected support seed43 accuracy | Original joint none/OOS recall | Selected support none/OOS recall |
|---|---|---|---|---|
| Banking, fitting labels | 82.42% | 85.55% | 83.59% | 85.16% |
| Banking, held-out labels | 78.52% | 63.67% | 83.59% | 48.44% |
| CLINC, fitting domains | 84.38% | 89.84% | 82.81% | 87.50% |
| CLINC, held-out travel/work | 82.42% | 66.02% | 89.06% | 62.50% |
| CLINC, author OOS | 98.44% | 100.00% | 98.44% | 100.00% |

Source: E9 strict-FP32 metrics. None recall uses omitted-intent episodes in the first four rows and author-OOS episodes in the last row. Relative to this original-criteria arm, the selected support arm is better on fitting-label/domain accuracy but worse on held-out Banking and CLINC accuracy, with particularly weaker omitted-intent recall. The original and support frozen inherited-none controls give identical held-out-family probability vectors where no support example is supplied, whereas the fitted readouts differ. Support was available only for fitting labels. The pattern is **consistent with a mismatch between support-conditioned fitting/selection and unsupported held-out inference**, but the experiment does not isolate that explanation from the other effects of readout fitting and selection. It does not prove a general cause or warrant relabeling final examples. [E8; E9; V2; interpretation]

This is a model-selection and transfer finding, not a reason to abandon Qwen or a demonstration that a larger backbone solves the problem. It supports carrying both criteria variants and a strong original-criteria joint control into the next multi-task study, with selection diagnostics representing the intended transfer conditions and a new untouched final set. Support examples are extra supervised input, not independently reviewed criteria. [E9; proposed consequence for 2I.1–2I.2 and 2J.1–2J.2]

### 13.1.11 Frozen policies: useful coverage, residual omission errors and cost sensitivity

All 126 profile policies were selected before final evaluation and reused unchanged. The following are the selected model's **nine pooled final policies** under the declared absent-prior/error-cost scenarios. Author-OOS supplies half of the assumed absent mass, review cost is 0.1, and correct acceptance costs zero. These are scenario assumptions rather than measured deployment prevalence or monetary prices. [E8; E9]

| Assumed absent prior | Wrong cost | Threshold | Scenario coverage | Scenario cost | Recorded 95% cost interval | Raw wrong/accepted |
|---|---|---|---|---|---|---|
| 5.00% | 1 | 0.80 | 60.64% | 0.040874 | 0.036469–0.045688 | 31/357 |
| 5.00% | 5 | 0.80 | 60.64% | 0.046929 | 0.041835–0.052410 | 31/357 |
| 5.00% | 20 | 0.90 | 54.26% | 0.062339 | 0.052863–0.073408 | 17/309 |
| 25.00% | 1 | 0.80 | 48.51% | 0.059058 | 0.054879–0.063564 | 31/357 |
| 25.00% | 5 | 0.90 | 43.19% | 0.077563 | 0.067319–0.090481 | 17/309 |
| 25.00% | 20 | 0.95 | 37.35% | 0.091943 | 0.068144–0.122705 | 6/260 |
| 50.00% | 1 | 0.90 | 29.35% | 0.078955 | 0.074291–0.084277 | 17/309 |
| 50.00% | 5 | 0.95 | 25.10% | 0.089551 | 0.077601–0.104980 | 6/260 |
| 50.00% | 20 | 0.95 | 25.10% | 0.133496 | 0.085791–0.192871 | 6/260 |

Source: E9 selected-profile `pooled_declared_mixture.policies`. V2 recomputes point costs, coverage, raw counts and effective origin masses; the policy-bootstrap intervals are retained as source-reported rather than re-estimated by this revision. Raw accepted counts and error fractions do not use the scenario-weighted coverage denominator. At threshold 0.80, 31/357 accepted episodes are wrong (8.68%); at 0.90, 17/309 are wrong (5.50%); at 0.95, 6/260 are wrong (2.31%). The repetitions of a threshold across scenario rows are not independent validations. [E9; V2]

The six wrong acceptances at threshold 0.95 come from **five source messages**, all in the deliberately omitted-intent stratum: three episodes from CLINC fitting domains and three from held-out domains. Thus excellent author-OOS behavior coexists with confident wrong selections when an in-scope answer is missing. Under 50% absent prevalence and wrong cost 20, the pooled point cost is **0.133496**, above always-review cost 0.1, while its recorded interval 0.085791–0.192871 includes 0.1. This is a negative point-estimate trade-off, not a statistically established universal disadvantage. At 25%/20, the point cost is below 0.1 but its interval also crosses the all-review reference. The model is not certified for rare expensive errors. [E9; V2; interpretation]

The error dossier at top probability at least 0.98 contains **five selected-model error episodes from four messages**: four wrong offered answers under omission and one false-none result on an answerable episode. Top probability is not the separately defined API confidence statistic. The underlying labels and criteria remain unchanged; the dossier is post-evaluation diagnostic material, not permission to repair this final benchmark after inspection. [E9 `high_confidence_errors.json`; V2]

Per-family policy costs condition on the origins available in that family. For example, an in-scope-only family has no author-OOS rows; the prescribed masses are renormalized over present and omitted strata. The source reports `complete_population_mixture_available=false` and `effective_origin_masses` for these cases. The pooled policy table above includes all three origins. Do not average family costs as though all used the same population denominator. A correctly emitted semantic none is also not an automatically accepted offered answer: an OOS-only family can have 100% semantic accuracy and zero application coverage. [E9; archived policy definitions]

### 13.1.12 Numerical acceptance: completion and unchanged accuracy do not imply equivalence

On the selected model's predefined **36-episode, ten-message** parity subset, all three optimized strategies pass against their own mode's full-sequential reference. The acceptance rule remains maximum absolute probability difference at most 0.005, no argmax change and no output change under any of the nine frozen policies. [E9]

| Alternative vs same-mode full sequential | Strict max delta | Strict accepted | TF32 max delta | TF32 accepted |
|---|---|---|---|---|
| full_batch4 | 0.00000479 | 36/36 | 0.00373623 | 36/36 |
| shared_lossless | 0.00000938 | 36/36 | 0.00191846 | 36/36 |
| shared_kv_fp16 | 0.00059041 | 36/36 | 0.00304892 | 36/36 |

Source: E9 `parity_summary` and `parity_rows`; V2 reconstructs all 216 within-mode comparisons across both workers. Each strategy's 36 episodes use the same ten source messages, including separate present/omitted K variants and OOS episodes. These are probability units, not percentages. H's parity sample is not a replacement for the earlier E/F/G panels, and it does not establish long-context optimized-path equivalence in every case. [E9; V2]

Across **all 1,152 final episodes**, TF32 versus strict FP32 is more limited:

| Profile | Max probability delta | Over 0.005 | Argmax changes | Any-policy output changes | Accepted |
|---|---|---|---|---|---|
| Original / Historical scorer + inherited set-linear none | 0.00376903 | 0 | 2 | 0 | 1150/1152 |
| Original / Historical scorer + refitted none | 0.00362419 | 0 | 0 | 2 | 1150/1152 |
| Original / Joint seed 17 + refitted none | 0.00621837 | 2 | 1 | 2 | 1148/1152 |
| Original / Joint seed 29 + refitted none | 0.00647778 | 3 | 1 | 1 | 1148/1152 |
| Original / Joint seed 43 + refitted none | 0.00625595 | 2 | 1 | 3 | 1146/1152 |
| Original / Lexical control | 0.00000000 | 0 | 0 | 0 | 1152/1152 |
| Original / Unadapted finite-code control | 0.00256981 | 0 | 0 | 0 | 1152/1152 |
| Support / Historical scorer + inherited set-linear none | 0.00520183 | 1 | 0 | 1 | 1150/1152 |
| Support / Historical scorer + refitted none | 0.00369966 | 0 | 0 | 3 | 1149/1152 |
| Support / Joint seed 17 + refitted none | 0.00789060 | 5 | 1 | 1 | 1145/1152 |
| Support / Joint seed 29 + refitted none | 0.00724399 | 4 | 1 | 0 | 1147/1152 |
| Support / Joint seed 43 + refitted none | 0.00693844 | 2 | 0 | 0 | 1150/1152 |
| Support / Lexical control | 0.00000000 | 0 | 0 | 0 | 1152/1152 |
| Support / Unadapted finite-code control | 0.00653808 | 1 | 0 | 0 | 1151/1152 |

Source: E9 `cross_mode`, independently reconstructed by V2. The selected support/seed43 model has **zero argmax and policy changes**, but **two numeric failures**, with maximum difference **0.00693844** (about 0.694 percentage points); only **1,150/1,152** episodes pass the combined gate. Other profiles can change argmax or application output. The equality of the headline accuracy rows is therefore not sufficient to promote TF32 as an interchangeable implementation. CPU lexical equality is not GPU arithmetic validation. Existing F/G failures remain historical results and are not relaxed. [E9; V2]

An **additional saved-vector comparison**, not a newly run GPU experiment or a new notebook gate, applies the same rule to strict/TF32 controlled-robustness rows. For the selected model it finds **256 variants from 16 messages**, maximum difference **0.02578599**, six episodes over 0.005, zero argmax changes and one episode with a policy-output change; **249/256** pass. The matched frozen support control has maximum 0.00512470, one numeric failure, one argmax change, two policy-change episodes and 252/256 accepted. The selected model's one policy-change episode is distinct from its six numeric failures. These are full-sequential cross-mode comparisons; they do not add an unperformed batched/cache parity test on the same long contexts. [V2 aggregation of E9 `robustness_rows.json`]

Retain strict FP32 as the selected profile's measured numerical reference. TF32 remains a measured performance option whose behavior needs its own declared operating contract, not a silently accepted optimization. This is a bounded numerical verdict, not a claim that TF32 is always semantically worse. [E9; V2; implementation recommendation]

### 13.1.13 Controlled context and instruction tests: answerability remains sensitive

The completed reliability study uses **16 in-scope source messages**, with answer-present and omitted-intent K=4 variants and eight controlled transformations, producing 256 episodes per profile. It evaluates the selected joint model and its **matched support-example frozen inherited-none control**, not only the most favorable arm. Each table row contains 32 episodes from the same 16 messages. No natural-document dataset was supplied. [E9]

| Minimum state tokens | Layout / instruction treatment | Selected accuracy | Selected NLL | Selected answerable accuracy | Frozen-control accuracy | Frozen-control NLL |
|---|---|---|---|---|---|---|
| 0 | first | 81.25% | 0.474177 | 81.25% | 71.88% | 0.603085 |
| 0 | last | 81.25% | 0.539942 | 81.25% | 75.00% | 0.611836 |
| 256 | first | 84.38% | 0.543801 | 75.00% | 78.12% | 0.507538 |
| 256 | last | 75.00% | 1.158690 | 56.25% | 81.25% | 0.406060 |
| 1024 | first | 81.25% | 0.639294 | 75.00% | 78.12% | 0.555287 |
| 1024 | last | 75.00% | 1.298829 | 56.25% | 84.38% | 0.445854 |
| 256 | instruction_paraphrase | 71.88% | 1.317874 | 43.75% | 68.75% | 0.577990 |
| 256 | instruction_in_notes | 81.25% | 0.838045 | 62.50% | 84.38% | 0.389003 |

Source: the first six rows are E9 `context_by_position`; the two instruction-treatment rows are additional V2 aggregations of E9 saved reliability inputs/predictions using the archived metric definition. Minimum-token zero is a minimal wrapper, not an empty state. The instruction treatments use their specific constructed contexts; comparisons among these rows do not isolate all wording, background and layout factors. [E9; V2]

At the 1,024-token last-evidence condition, the selected model has 75.00% combined accuracy and **56.25% answerable accuracy**, with **43.75% false-none** on answerable episodes. Under the recorded instruction-paraphrase treatment, answerable accuracy is 43.75% and false-none is 56.25%, even though combined accuracy is 71.88% because omission rejection contributes correct outputs. The matched frozen control is better on several background conditions, including 84.38% combined accuracy and 0.445854 NLL at 1,024 tokens with evidence last. This is not uniform reliability improvement from head adaptation. [E9; V2]

The instruction-in-notes probe was executed, but one constructed probe is not an adversarial security validation. Neither passing execution parity nor acceptable pooled accuracy establishes that the relevant evidence is interpreted correctly in long documents. Keep independent natural documents, explicit evidence-insufficiency annotation and broader instruction/evidence-position tests open under 2I.2/2I.6 and 2J.4. [E9; interpretation and roadmap consequence]

### 13.1.14 Completed bounded primitive probes

BoolQ and SST-5 now have final predictions, not just fitted readouts. Each task evaluates **128 saved final examples/groups** for three seeds in each arithmetic mode. Seed **29** was selected before final evaluation for both tasks; the other seeds remain reported comparisons. The following accuracy is identical across strict/TF32 within each seed, but that is not a separate distribution-equivalence certification. [E9]

| Probe | Seed | Final examples | Accuracy, both modes | Strict NLL | TF32 NLL | Strict Brier, sum over classes |
|---|---|---|---|---|---|---|
| noul_boolq | 17 | 128 | 85.16% | 0.319406 | 0.319205 | 0.197133 |
| noul_boolq | 29 (preselected) | 128 | 86.72% | 0.319617 | 0.319439 | 0.199444 |
| noul_boolq | 43 | 128 | 83.59% | 0.342303 | 0.342113 | 0.205603 |
| score_sst5 | 17 | 128 | 47.66% | 1.167932 | 1.167860 | 0.642698 |
| score_sst5 | 29 (preselected) | 128 | 54.69% | 1.137379 | 1.137232 | 0.621491 |
| score_sst5 | 43 | 128 | 51.56% | 1.189320 | 1.189382 | 0.640849 |

Source: E9 `primitive_metrics` and per-seed predictions; V2 recomputes all 12 final metric records from the saved distributions and archived labels. The preselected BoolQ readout is correct on **111/128** examples, with NLL 0.319617 and binary yes-probability Brier **0.099722**. Its recorded accuracy interval is 80.47–91.41%. The selected SST-5 readout is correct on **70/128**, with NLL 1.137379, expected-level mean absolute error **0.575645**, argmax-level mean absolute error **0.531250**, and ranked probability score **0.095034**. Its recorded accuracy interval is 46.09–62.91%. The ranked probability score follows the source's mean squared cumulative-probability error across four cutpoints, not a sum with a different scale. [E9; V2]

These results support **bounded yes/no-on-passage and five-level sentiment readouts**. BoolQ has no dedicated insufficient-evidence output. SST-5 is not an arbitrary user-defined scoring rubric, and its expected level does not validate every ordinal action rule. The saved primitive final metrics do not include a dedicated primitive latency benchmark or a full API-conformance result. Completing these probes closes H's primitive evaluation, not the broader Noul/Score work in 2J.4 or the service contract in Track S. [E9; scope boundary]

### 13.1.15 Complete-request timing and memory: a measured, bounded trade-off

The request panel contains **eight episodes from eight source messages**, four episodes at each K, with one warmup and three timed repetitions per episode/strategy. Each worker retains 40 benchmark rows: selected affine full sequential, full batch four, shared lossless, plus the matching support lexical and one-step finite-code controls. The table reports the **median of the four per-episode median times** for each cell; it is not a service latency percentile. [E9; V2]

| K | Method / execution | Strict FP32 ms | TF32 permitted ms | Strict/TF32 time | Maximum extra allocated MiB |
|---|---|---|---|---|---|
| 4 | Selected head / full sequential | 595.16 | 377.90 | 1.57× | 43.87 |
| 4 | Selected head / full batch four | 443.17 | 198.42 | 2.23× | 177.96 |
| 4 | Selected head / shared lossless | 463.30 | 409.74 | 1.13× | 206.41 |
| 4 | Support finite-code control | 221.47 | 115.17 | 1.92× | 91.61 |
| 4 | Support lexical control | 4.03 | 4.15 | 0.97× | 0.00 |
| 16 | Selected head / full sequential | 2025.60 | 1449.00 | 1.40× | 61.39 |
| 16 | Selected head / full batch four | 1501.11 | 664.21 | 2.26× | 241.25 |
| 16 | Selected head / shared lossless | 1191.17 | 1027.30 | 1.16× | 351.82 |
| 16 | Support finite-code control | 553.79 | 230.97 | 2.40× | 278.44 |
| 16 | Support lexical control | 5.27 | 5.38 | 0.98× | 0.00 |

Source: V2 aggregation of the 80 E9 benchmark rows and 240 timed values. Extra memory is the maximum per-request increment across the four episodes and is identical between modes for each displayed method/strategy in this run. The timed scope includes tokenization, GPU work where used, transfers, head calculation, policy outputs and JSON assembly. It excludes model loading, HTTP, queueing, concurrent serving and prior offline feature-cache hits; timed model requests recompute their work. The lexical control runs in the same model-resident worker, so its approximately 4–5 ms timing is not a claim that the entire process has lexical-only memory usage. [E9; archived `time_request` and `Predictor.request`]

TF32 full batching is approximately **2.23×** faster at K=4 and **2.26×** at K=16 in this panel. Strict FP32 prefers full batching at K=4 and shared lossless at K=16; TF32 full batching is faster than shared lossless at both K values. The selected model's shared execution still shares candidates within one instruction/state pair; Q-level sharing was disabled. No universal scheduler cutoff follows. Support-example inputs and the H-selected head differ from G's workload, so the timing tables are not a controlled before/after systems-speed comparison. [E9; V2; interpretation]

The finite-code control is substantially faster in these requests and has its own complete final-quality rows in §13.1.10, but it is unadapted vocabulary scoring with different input rendering and training budget. These timings do not settle the still-open trained finite-token/SALSA-style comparison or equal-quality useful-automation throughput. No additional FP16-KV request-time benchmark was run in H, despite its inclusion in the parity subset. [E9; scope boundary]

Both workers record approximately **16,043.69 MiB allocated immediately after model loading**, and **16,051.81 MiB allocated / 17,528 MiB reserved** at the final snapshot. The saved `after_stage.peak_allocated_mib` is approximately **16,330.25 MiB**, but the timing code calls `reset_peak_memory_stats()` before each request benchmark. It is therefore a peak **since the last reset**, not an established whole-study or service high-water mark. The per-request extra-memory measurements above retain their narrower meaning. A process-memory ceiling, memory-leak soak result, deployment concurrency limit or Apple-Silicon estimate is not established here. [E9; source-code interpretation]

### 13.1.16 Closeout and consequences for the next study

**2H-C1–C5 are closed for the required original study:** preserved source attempt and verified inputs; repaired/recovered execution; locked final evaluation; published final/reliability/primitive/arithmetic/request readouts; and a source-linked whitepaper/roadmap handoff. Negative numerical or quality findings do not prevent honest experiment completion. The original failed attempt remains an unchanged historical artifact, while E9 is the completed-continuation authority. This closes the recovery queue, not all questions raised by H. [E9; V2]

The main learning is more specific than “training helped.” Rejection refitting and small readout adaptation improve paired final results relative to the historical control, but support-conditioned development selection did not choose the strongest final-transfer arm. Perfect author-OOS classification on this sample did not eliminate omitted-intent errors, policy risk or context sensitivity. TF32 offers a substantial batching speed benefit but still fails the unchanged combined equivalence gate. Bounded BoolQ/SST-5 evidence is now available without validating general primitives. [E9; V2; synthesis]

The required H study and exploratory E11 comparison are complete. Their
exposed final sets remain historical/regression evidence. Native parity and
bounded CPU service work subsequently completed within scope. Independent
review, repaired natural inputs, declared limits, and fresh held-out confirmation
remain necessary for model promotion under M0–M4. [E9; E11; §17.3]

At the E9 checkpoint, state-first execution and the native service remained
open. Later records close the bounded execution, replay, and CPU service gates.
That history does not establish semantic generality, MLX service qualification,
or teacher quality. The 4B profile remains an immutable numerical reference
while the supported task determines which model should ship. [E9; RUST1–RUST11;
§17.3]


## 13.2 Two acceptance tracks, not one universal parity rule

| Track | Question being tested | Required comparison | What does not count as success |
|---|---|---|---|
| Implementation equivalence | Does an optimization or backend preserve the specified model? | Same finalized input, weights, rendering, heads, calibration and policy; compare full probabilities, argmax, and directed policy outputs | A faster path or unchanged aggregate accuracy that fails the declared same-model gate |
| New-model quality | Does a new representation, head, size, input contract, or trained operating point improve useful decisions? | Fresh labeled task/rubric families; predeclared quality, risk/coverage and resource requirements; its own versioned execution reference | Mere agreement with the old model, a favorable inspected test, or throughput obtained by reviewing everything |

The historical 0.005 probability tolerance and no-outcome/no-policy-change requirements retain their meaning for E/F/G. No failed codec or TF32 comparison is retroactively promoted. A new trained model may legitimately change old answers, including correcting old errors; it should not be rejected solely for doing so. Once selected, its implementations must still meet their declared stability and equivalence requirements. A new-model label is not permission to ignore batching drift or deployment risk. [E5–E7; R3; proposed acceptance policy]

## 13.3 Priority order and synchronized roadmap

[ROADMAP.md](../ROADMAP.md) owns M0–M4 and all active exit conditions. This paper
records the following evidence update without claiming a roadmap-file edit.
Completed execution is not the same as satisfying a milestone's quality exit.
[E29–E32]

| Work item | Evidence status at this update | Consequence |
|---|---|---|
| M0/M1 scope, source and effective-input contract | Bounded source-record/visibility diagnostics and a complete-document ContractNLI training scope are available; independent semantic correction and general release scope remain open. | Preserve label/audit boundaries; do not treat the pilot as blanket benchmark repair. |
| M2.1 frozen baseline | J0/J1 × 1,024/4,096 diagnostic completed in FP32. | Keep both checkpoint controls and evidence strata. [E29] |
| M2.2 decision adaptation | Both matched BF16 fits and four-arm evaluation completed; both adapted arms fail full retention. | Preserve update-80 specialists and the failed complete-screen verdict; no automatic seed expansion. [E30] |
| Retention-aware follow-on | Parent-KL replay and per-class selection completed; no evaluated nonzero candidate qualifies. | Keep J4/J5 fixed80 as diagnostics and selected-zero identities unchanged. [E31] |
| Source-label replay follow-on | Completed J6/J7 fits; SNLI improves, but entailment/QASPER and complete screens fail; both select frozen0. | Close this exact recipe and preserve fixed80 diagnostics and selected-zero identities. [E32] |
| Next bounded diagnosis | Proposed train/development-only investigation of supported-entailment failures and task-appropriate retention; no corrective treatment tested yet. | Separate supported-to-none/contradiction changes, hypothesis families and evidence-location effects before selecting one intervention; no gate-error training or automatic sweep. [V6; §18.26] |
| M3 cost reduction | Immutable-profile MLX performance work remains separate; no adapter-specific native/cache/service qualification follows from fitting. | Defer learned compression or replacement until a useful scoped operating point survives quality/retention checks. |
| M4 independent confirmation | No protected-final opening or promotion. | New exposed-panel success alone cannot supply independent confirmation. |

Full hybrid-state branching remains valid reusable computation within its tested
contract. The failed pooled reader and incomplete StateQuery studies are not
erased; E30 adds a different positive learning result with a different failure
boundary. Do not infer that all shared-query architectures fail or that
post-trained initialization is universally superior after adaptation.
[E12–E20; E26; E29; E30]

The historical crosswalk below explains earlier phase-name changes and remains
provenance, not evidence of a new current roadmap edit.

### 13.3.1 Historical phase-name collision and task migration

The roadmap reviewed at the v0.5.1 checkpoint called Phase 2H “Contract Hardening, Criteria Review, Feature Rejection & Prototype Bridge” and marks it NEXT. The actual H notebook instead performed criteria/rejection-transfer fitting with bounded primitive probes. Both refer to useful work, but they are not the same milestone. The synchronized roadmap restores the experiment's name and relocates the open proposals explicitly. [R4; E8]

| Previous roadmap task | Current location | Why it remains separate from H's saved results |
|---|---|---|
| Old 2H.1 — current contracts and frozen regression suite | **S.1; 2H-C1** | Existing artifacts are evidence, not proof that repository instructions and recovery are synchronized |
| Old 2H.2 — independent criteria/annotation review | **2I.2** | H supplied training examples without independent review; keep its frozen criteria unchanged for its original evaluation |
| Old 2H.3 — feature-conditioned rejection | **2J.3** | H refitted score-summary none models; it did not validate a hidden-feature applicability head |
| Old 2H.4 — resident Python worker bridge | **Retired; S.3–S.5 use direct native registration** | Python remains a differential oracle; load/soak is still separate from Colab fitting |
| Old 2H.5 — none/key/API contract | **S.2** | A probability vector and a compatible wire response require an explicit mapping |
| Old 2J.3 — multi-task supervision | **2I.1–2I.2** | The common data/evaluation contract must precede the matched comparison, not arrive afterward |

These are task migrations, not erased requirements or executed improvements. The roadmap also retains a crosswalk for renumbered 2I rendering/sharing tasks. Its reported **195-test** code snapshot lacks an exact commit in the attachment; that historical total is not promoted to a verified current-HEAD result. No repository build or test was run for this update. [R4; V1]

### 13.3.2 Stage exits and limits on promotion

**H closeout is satisfied for the declared required scope.** E9 provides traceable recovery, unchanged fitting artifacts and final payload, a checked lock, both completed evaluation workers, full saved outputs and this evidence handoff. E8 remains failed/partial as history. Numerical failures and mixed quality do not reopen execution bookkeeping; independent review, new models, Q-level sharing and deployment claims remain separate open work. [E8; E9; V2]

**The 2I/2J model milestone** requires reviewed task definitions, independent state/question/rubric holdouts, per-stratum rejection and distribution metrics, useful coverage and measured resource limits. Feature-aware rejection is an ablation against the constant and score-summary controls, not a preselected replacement. Actual parameter/memory accounting must replace the old roadmap's approximate halving of 4B memory for a 2B candidate. [R4; proposed comparison gate]

**The Track S prototype** now has a directly registered native profile returning probabilities with bounded work and truthful capability errors. Its named-machine CPU service gates pass within their declared scope (§17.3). It may use the historical reference for integration without calling that model production-ready. The native Phase 3 gate applies to the selected model and backend's real capabilities; it neither requires every proposed backend nor imposes Qwen-specific hybrid-cache machinery on a different architecture. Historical fixture tolerances and prefix-storage observations are not universal new-model acceptance thresholds. [R4; proposed integration/native gates]


## 13.4 A concrete multi-question milestone

Show several independently defined questions over one state through a real, versioned engine. Begin with supported Choice judgments and clearly bounded binary/rubric tasks rather than claiming all primitive semantics are solved. Use held-out question/rubric families to test transfer, and distinguish answering Q questions correctly from encoding their shared state once. Both properties matter; neither proves the other. [R3; proposed milestone]

M0 now selects a bounded document workload before timing or model selection. The following earlier design grid remains illustrative and does not require a full sweep.

A proposed initial scaling grid is **Q = 1, 4, 16** and **K = 2, 4, 8, 16**, over proposed state-length targets **64, 256 and 1,024 tokens** and supported question types. Actual finalized token counts include question/criteria/formatting overhead; these are initial protocol targets, not established limits. It is a design grid, not a completed benchmark or required maximum API size. Hold other dimensions fixed for causal comparisons and use representative subsets before an expensive full sweep. Test question-order changes, opaque-ID renaming, candidate permutation, and adding/removing unrelated questions. Separate expected changes from changing a candidate set from unintended cross-question influence. [R3; proposed protocol]

Report correctness, NLL/Brier and reliability diagnostics, conditional rejection rates, accepted-answer error, coverage, and scenario-defined application cost. Report independent source-state counts alongside all expanded episode/question counts and use source-group-aware uncertainty. For performance, include complete latency, decisions/questions per second, peak and resident memory, model invocations, branch-state bytes, state-prefill share, **`T(Q)/T(1)` and marginal latency per added question**. Compare repeated-full, sequential-nested and batched-nested execution on the same profile. Separate startup, cold-state, warm-state/cache-hit, and queueing costs; do not mix L4 GPU measurements, DGX Spark measurements or hosted Jev timings with unmeasured Mac performance. [R3; P22; proposed measurement contract]

The preferred selection criterion is **correct accepted decisions per second subject to predeclared accepted-error, coverage, latency, and memory requirements**. Set the numerical requirements and target hardware before looking at the final comparison; this paper does not invent an evidence-free safety threshold or resource winner. Always reviewing is a cost baseline, not a useful high-throughput success. At zero accepted answers, conditional accepted-error is undefined. A model should advance only with an explicit task scope and enough evidence to support its stated operating region. [R3; proposed selection rule]

## 13.5 Work to pause or keep conditional

Keep one modeling hypothesis and one frozen-profile systems experiment active
at a time. Pause nearby scalar-weight, shallow-head, pooled-root, and handcrafted
residual sweeps without a materially different diagnosis. Distillation waits
for demonstrated teacher quality. E30's contract-only pilot is executed but not a
full quality pass; neither more updates nor seed expansion is automatically
justified. Parent-KL and source-label replay have now both completed and failed
the full preservation contract. Pause generic short-premise replay variants as
the main remedy while the repeated supported-entailment and long-document failures
are diagnosed. Do not loosen class guards or retune inspected gates to obtain a
pass. Preserve high-K correctness and admission fixtures, but keep K=255 latency
work conditional on supported-workload demand. Completed CPU bring-up and replay
remain maintained assets. [E14–E32; §17.3; §18.26]

Pause further low-bit KV snapshot sweeps on the same short-prefix workloads. Keep lossless and bounded FP16-KV paths available as reference infrastructure, with the unsuccessful codecs preserved as negative results. Reopen compression only when a materially different workload, codec, or memory profile supplies a specific hypothesis. Failed snapshot quantization is not a verdict on model-weight quantization; they affect different tensors and require separate tests. [E6; E7; R3]

Do not prioritize MTP for a graph that generates no output tokens, a large RLCD-inspired program before supervised/proper-scoring baselines, or a diffusion rewrite based on a retrieval paper. The new PCD reference strengthens the case for optimizing the existing bounded-decision graph before inventing a new training model; it does not establish that RL is unnecessary forever. A verified optimized backend can be an earlier controlled experiment, but package installation alone does not establish kernel use or support for the hidden-state/cache operations the engine needs. [E6; R3; P20; proposed limits]

# 14. Phase 2I/2J workbench: preparation, mechanical evidence and the review gate

## 14.1 What this checkpoint establishes

Study `2ij_reviewed_multiquestion_v1`, workbench version `2ij.1.0`, exported its report around **16:10 UTC on 19 September 2026**. Its overall status is **`blocked`**, not `failed` or `completed`. The reviewed-study gate stopped the new semantic training/evaluation program, while the permitted historical checks, draft preparation and a small real-GPU mechanics probe completed. The word “reviewed” in the study ID is a name, not evidence that independent review occurred. The summary has `registered_scope: null`, `model_profiles: []` and `final_results: []`. [E10]

| Component | Saved outcome | Evidentiary meaning |
|---|---|---|
| Historical H lineage and fixtures | Completed | Read-only reference checking and exclusions; no H retraining or new H model selection |
| Review package | Completed as a draft | Constructed cases and unsigned review/protocol templates, not an approved dataset |
| Native probability contract | Draft exported | A native representation and worker handoff, not approved Jev wire compatibility or Rust integration |
| Revision-resolution stage | Returned `resolved` | Inspect the protocol for model identities; its summary `models` field is a change list, not a model inventory |
| Reviewed-study gate | Blocked | Independent-review attestations, target/quality limits and matching review manifest are missing |
| GPU mechanics | `completed_mechanics_only` | Frozen pretrained Qwen features on synthetic inputs; no new readout training or semantic quality result |
| New model comparisons and final selection | No profiles or final results | No smaller-model, LoRA, ModernBERT, feature-rejection, calibration or new task-quality conclusion |
| Released external checkpoints, Rust/Metal service and P2 studies | Deferred/blocked | Their own adapter, integration, selected-profile and hypothesis requirements remain open |

The recorded H audit checks 39 historical implementation hashes and 24 preserved fitted files, replays saved head algebra, and builds **7,578 unique exclusion hashes** from available H/earlier manifests. These are historical-lineage and exact-group exclusions, not new samples or paraphrase/pretraining decontamination. The archived CPU test log records **48 tests passed in 72.61 seconds**. This revision checks those saved records; it does not rerun that test suite or the wider completed-H audit. [E10; V3]

The native contract keeps option probabilities unconditional, represents Choice none mass separately, uses a null selected ID when native none wins, and labels maximum probability `top_probability`, not vendor confidence. It specifies binary/ordinal outputs only for registered task definitions and rejects unsupported or overlength inputs. Its explicit `not_jev_conformance: true` flag matters: producing this draft advances contract preparation but does not close S.2 or establish an HTTP service. [E10, `contracts/native_probability_contract.json`]

## 14.2 The small state-first GPU probe

The permitted mechanics worker loaded the pinned **Qwen3.5-4B-Base** checkpoint in strict FP32 on an **NVIDIA L4**, using Torch `2.11.0+cu128`, CUDA 12.8 and Transformers `5.17.0`. Its arithmetic flags request highest matmul precision, disable matmul/convolution TF32 and restrict SDPA to math attention. The worker log reports reference paths for the optional linear-attention/convolution implementation; no optimized-kernel dispatch result is established. [E10, runtime and worker log]

The fixture uses **one fictional state, two synthetic question rows and two candidates per question**, under the new `state_first segmented v1` contract. It compares candidate-conditioned hidden features from full execution with nested state → question → candidate branches. The frozen pretrained backbone is used without fitting a new decision head. Dummy fixture targets are not scored for semantic accuracy. [E10, `odij_scaling.py` and `mechanics/DONE.json`]

For each question's feature array X and alternative Y, the source computes maximum absolute error and a scaled error. The reported scaled quantity is the maximum over question rows of `max(abs(X − Y)) / max(1, max(abs(X)))`. It is an array-scale normalization, not coordinate-wise relative error and not a probability difference. [E10, `compare` implementation]

| Saved comparison | Maximum absolute feature delta | Maximum scaled feature delta | Scaled tolerance | Recorded gate |
|---|---:|---:|---:|---|
| Nested versus full sequential | 4.57763672e-5 | 1.12739406e-6 | 1e-4 | Pass |
| Reversed question order versus original nested order | 0 | 0 | 1e-5 | Pass |
| Source field `unrelated_question_addition` — actually duplicate-question append; see §14.3 | 0 | 0 | 1e-5 | Pass for the implemented fixture only |
| Exact-length full batching, capacity four, versus sequential | 2.67028809e-5 | 6.60674232e-7 | 1e-4 | Pass |

Source: E10 saved mechanics output and implementation. V3 reconciles the duplicated records and saved threshold arithmetic. Raw full/nested hidden arrays are not retained in the compact archive, so the feature differences are **source-reported, not independently recomputed**. These feature tolerances do not replace the historical 0.005 probability/argmax/policy gate. No trained probability vector, rejection metric, calibration result or application-policy preservation test is reported for this probe. [E10; V3]

The state root's recorded **55,312,384 bytes = 52.75 MiB** span 24 recurrent-state entries, 24 convolution-state entries, eight attention-key entries and eight attention-value entries. `root_unchanged` is true. This supplies a bounded mechanical check of the all-state branching implementation; one fixture is not universal branch isolation, tenant security or concurrency validation. [E10; V3 inventory arithmetic]

The runtime records **33.229 seconds model load time**, **15.667663 GiB allocated** and **15.6875 GiB reserved** immediately after load. Those are startup and post-load observations. They are not complete-request latency, whole-probe peak memory, a Q/K scaling benchmark, or a comparison with the smaller and encoder models. No synthetic-probe speedup is inferred from them. [E10]

## 14.3 Coverage correction: the added question was a duplicate

The result key `unrelated_question_addition` overstates what its fixture tests. The archived generator restarts its numbering at zero on every call. The probe first creates `synthetic_rows(2, 2)`, then appends:

```python
synthetic_rows(1, 2, rows[0]['state'])
```

With the same state and candidate count, that call recreates the first question: the same `q0` ID, synthetic instruction, candidate descriptions and row content. The existing output shows invariance when **appending a duplicate of the first question**, not when adding a distinct unrelated judgment. This is a source-code-derived coverage finding; the original field name, zeros and pass flags remain unchanged in E10. [E10, `synthetic_rows` and `mechanics_probe`; V3]

Under **2I.6**, add a really different extra question with a unique ID and distinct instruction/candidates; assert that it is not a duplicate before testing unaffected original outputs. Retain duplicate insertion as a separately named regression case. The new test should still distinguish mechanical feature agreement from semantic independence under trained judgments. This document does not repair the notebook or assert an outcome for the corrected test. The nested/full and question-reordering observations remain valid within their own recorded fixture scope. [V3; proposed follow-up]

## 14.4 Why the next study remains blocked

The 21 gate messages describe three practical requirements, not 21 independent model failures. [E10]

**Independent review is unsigned.** Protocol owner, distinct independent reviewer and review date are blank; approval and the label/criteria/final-selection attestations are false. The separate review manifest is also unsigned. An attestation is a workflow record, not cryptographic identity verification or a substitute for actually reviewing the cases. Neither the notebook nor this revision supplies that review. [I0]

**Selection limits are undefined.** `target_device` and all five `quality_requirements` values remain null: `max_family_macro_nll`, `max_panel_accepted_error`, `min_policy_coverage`, `max_request_p95_ms`, and `max_peak_allocated_gib`. The L4 used for the mechanics probe is an observed device, not an automatically approved deployment target or a basis for inventing thresholds. These limits must be chosen before model selection and retained with their population, timing and memory definitions. [I0]

**The old review points to a stale protocol.** The review manifest records protocol-body hash `454d3b6e843b96203489b723227dc8a0cf1715f0a2af48413fda6731698ad4ad`. The retrieved protocol, excluding its `approval` section under the source's canonical serialization rule, hashes to `30239d9a2699b9ebf1bde047a9c9b453b64c01244079801d56512364c74c4791`. The separate refreshed review template has that current hash and remains unsigned. Refreshing the hash addresses consistency only; it does not satisfy independent review or the missing limits. [I0; V3]

The review metadata lists **92 case hashes**: 32 training cases and 12 each for development, calibration fit, calibration gate, policy development and final. These counts describe a draft inventory, not approved independent samples or completed predictions. V3 counts IDs/hashes without opening `cases.jsonl` or final annotations; it does not verify the annotations or recompute the recorded full-corpus hash. The intake's `constructed_pilot` scope and review instructions explicitly warn that its `author_oos` tag denotes a constructed out-of-task case, not an original CLINC OOS annotation or natural operational evidence. Natural documents are not required by this draft pilot configuration; that does not close the broader roadmap's natural-document work. [I0; V3]

The intake protocol does contain three explicit 40-character checkpoint revisions:

| Configured checkpoint | Revision recorded in intake | Outcome in this invocation |
|---|---|---|
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` | Used only for the synthetic mechanics worker |
| `Qwen/Qwen3.5-2B-Base` | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | Configured; no trained or evaluated profile |
| `answerdotai/ModernBERT-large` | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | Configured; no trained or evaluated profile |

The resolver's summary `models: []` reports revisions **changed during that invocation**, because already populated revisions are skipped. It is not evidence of an empty protocol model list. This revision checks stored identities and the resolver's source; it does not query the model Hub or independently validate the two unexecuted checkpoints. [E10; I0; V3]

Nine arms are proposed in the protocol: frozen 4B and 2B; matched 4B online-head and LoRA; frozen and fully fine-tuned ModernBERT; 2B finite-token frozen and LoRA controls; and lexical scoring. Their presence in a configuration does not make them results. There is no registered semantic study, trained profile, final selection or quality/resource comparison in this checkpoint. [E10; I0]

## 14.5 Roadmap disposition and continuation procedure

**Keep 2H closed.** E9's completed original study and its errors remain historical. The new invocation did not retrain it, reverse its selection or turn its final set into fresh 2I/2J evaluation data. [E9; E10]

**Credit preparation without closing the broader milestones.** The review package advances 2I.1–2I.2 preparation; the GPU probe provides narrow 2I.4/2I.6 mechanics evidence; the native contract advances S.1–S.2 drafting. Full reviewed-data approval, trained rendering comparisons, Q/K performance, genuine unrelated-question isolation, model adaptation/selection and Rust/HTTP integration remain open. No 2J or conditional P2 experiment receives a completion mark. [E10; I0; roadmap disposition]

To continue the intended study, use the existing `OpenKind_Phase2IJ_review` intake. Finalize the cases and task/holdout definitions, specify the device and five quality/resource limits, then run the notebook's revision-resolution/review-template refresh step. Have a distinct reviewer inspect and sign the matching case/protocol manifest and protocol attestations. Only then rerun the same study ID, provided it is still unregistered; after registration, changed approved inputs require a new study identity. Preserve the blocked report as a dated snapshot rather than overwriting it with a claim of completion. These are the existing workflow requirements, not authorization to bypass them. [I0; E10 source workflow]

The compact report is **not a full resume archive**: it explicitly excludes H source archives, weights, optimizer state and SQLite feature caches. The `LATEST.json` pointer identifies separate complete Drive snapshot generations for recovery. V3 verifies the current snapshot-manifest identity and the 18 files shared with this report; it does not validate the unmaterialized SQLite backup or test a fresh-runtime restore. Do not infer future resume success merely from a report ZIP. No notebook, intake approval, reservation, source result or persistent Drive file was changed by this documentation update. [E10; V3]

#
# 15. Completed exploratory 2I/2J model selection and Rust handoff

## 15.1 Completion and evidence boundary

Study `2ij_model_selection_screen_v2`, workbench `2ij.2.0`, completed its requested exploratory scope after an A100 export-only continuation. The run preserved Phase 2H as read-only historical evidence, registered a separate exploratory pilot, completed all configured fitting jobs, locked model selection before final evaluation, evaluated every locked final profile, completed the bounded multi-question scaling study, and exported the selected-model reference bundle. The final workflow status is `completed_requested_scope`.

This completion has three important limits. First, the study scope is `exploratory_pilot`; the cases were not independently reviewed as an operational benchmark. Second, the five promotion bounds for family-macro NLL, accepted error, minimum coverage, request p95 and peak allocated memory were intentionally unset, so no release gate can be inferred from the point estimates. Third, native Rust/Metal parity and target-machine deployment measurements remain separate. The correct result is therefore **a measured provisional integration candidate with an exported executable reference contract**, not a production model.

## 15.2 Comparison design and pre-final selection

The screen completed 13 configured fit jobs and retained 31 final profiles after expanding rejection variants. It compared:

- Qwen3.5-2B frozen, online-head and limited-LoRA treatments under instruction-first and state-first rendering;
- Qwen3.5-4B frozen controls under both renderings;
- frozen and fully fine-tuned ModernBERT-large joint-option scorers;
- frozen and LoRA-trained finite-token controls on Qwen3.5-2B; and
- a lexical control.

The comparison is not an equal-compute tournament. The 2B online-head/LoRA pair is the specifically matched adaptation comparison; full ModernBERT training and finite-token adaptation have different optimization graphs and trainable-parameter budgets. All listed results are one-seed exploratory measurements unless otherwise stated.

Model selection did not inspect final labels. The locked rule first applied any declared nonfinal constraints, then kept neural candidates within 0.02 family-macro development NLL of the best eligible profile, and then used the measured nonfinal Q<=4 resident-request latency as the tie-breaker. Because promotion limits were unset, the resulting choice is explicitly provisional.

The best nonfinal family-macro NLL in the selection band was the Qwen4B state-first semantic-feature rejection variant at 0.049045. The score-summary variant was close at 0.049276 and had the lower measured Q<=4 p95 within the selection band, so profile `a047d6802c3f06f085b8` was frozen before final evaluation. No post-final reselection occurred.

## 15.3 Final comparison: state-first Qwen is the strongest tested family in this pilot

The table below reports the best final-NLL profile within each major fitted arm. It is descriptive of this pilot's data, renderer, optimization schedule and hardware; it is not an architecture ranking beyond that scope.

| Arm | Rendering / treatment | Final accuracy | Final NLL | Family-macro NLL | Nonfinal p95 | Peak allocated |
|---|---|---:|---:|---:|---:|---:|
| Qwen4B frozen | **state-first**, semantic-feature rejection | **95.00%** | **0.12944** | **0.06104** | 3,877 ms | 15.80 GiB |
| **Selected Qwen4B frozen** | **state-first**, score-summary rejection | **95.00%** | **0.13006** | **0.06168** | **3,840 ms** | **15.80 GiB** |
| Qwen4B frozen | instruction-first, semantic-feature rejection | 86.25% | 0.23308 | 0.11326 | 3,890 ms | 15.80 GiB |
| Qwen2B LoRA | state-first, score-summary rejection | 91.56% | 0.17834 | 0.10182 | 1,523 ms | 7.10 GiB |
| Qwen2B frozen | state-first, semantic-feature rejection | 89.69% | 0.18936 | 0.12118 | 1,496 ms | 7.08 GiB |
| Qwen2B online-head | state-first, score-summary rejection | 89.38% | 0.20791 | 0.11888 | 1,530 ms | 7.09 GiB |
| Qwen2B LoRA | instruction-first, semantic-feature rejection | 85.31% | 0.42661 | 0.18712 | 1,500 ms | 7.10 GiB |
| Qwen2B frozen | instruction-first, semantic-feature rejection | 84.38% | 0.46262 | 0.23046 | 1,497 ms | 7.08 GiB |
| Qwen2B finite-token + LoRA | finite-token | 77.50% | 0.45772 | 0.38869 | 769 ms | 7.10 GiB |
| ModernBERT-large frozen | joint option markers | 51.56% | 0.78912 | 0.80354 | 222 ms | 1.52 GiB |
| ModernBERT-large full | full encoder update | 42.19% | 0.88529 | 0.93364 | 221 ms | 1.53 GiB |
| Lexical control | native | 55.00% | 0.83242 | 0.86094 | 4.7 ms | ~0 GiB |

Three conclusions are supported within the experiment. **First, state-first rendering is not merely an execution optimization:** the state-first Qwen profiles substantially outperform their instruction-first counterparts on this pilot, so rendering belongs to the learned model contract. **Second, the 2B model is a credible deployment alternative but did not match the 4B state-first quality point:** the strongest 2B result reaches 91.56% rather than 95.0%, while using roughly half the measured GPU allocation and less than half the Q<=4 latency. **Third, the compact ModernBERT arm is much faster and smaller but is not competitive on this task setup.** The result argues against selecting the compact encoder merely from architectural efficiency; it does not reject Laya or every bidirectional scorer, because the released Laya checkpoint was not evaluated and this training treatment is specific.

The 2B LoRA result is also useful: limited adaptation improves the strongest 2B state-first operating point relative to the frozen and online-head variants in final accuracy/NLL, but does not close the observed quality gap to the selected 4B profile. The finite-token LoRA control improves materially over its unadapted finite-token counterpart but remains below the candidate-conditioned Qwen profiles.

## 15.4 Selected candidate final quality

The selected profile's final exploratory panel has **320 question episodes from 56 source messages**. Its aggregate metrics are:

| Metric | Selected profile |
|---|---:|
| Accuracy | **95.00%** |
| Source-group bootstrap 95% accuracy interval | **91.16%–97.64%** |
| NLL | **0.13006** |
| Family-macro NLL | **0.06168** |
| Brier | **0.07319** |
| ECE, 15 bins | **0.01661** |
| Ordinal argmax MAE | **0.0000** |
| Ordinal expected-value MAE | **0.00031** |

The constructed asset, external, compromise, routing, urgency, impact-rubric and evidence-sufficiency families are all 100% accurate on their 32-episode slices. That is useful evidence that the model can learn the pilot contract, but it also indicates that these generated families are too easy to support a broad competence claim.

The independently sourced MultiRC answer-correctness component is the informative counterweight: **80/96 correct = 83.33% accuracy**, with NLL 0.40361, Brier 0.23607 and ECE 0.07863 across 24 source messages. The overall 95% headline should therefore never be read without this natural-data slice.

All sampled semantic-none strata in this small final panel are correct: 8/8 omitted-option episodes, 8/8 constructed author-OOS episodes and 5/5 insufficient-evidence episodes. These counts are too small and too constructed to supersede the harder rejection findings from G/H or certify operational OOS/insufficient-evidence handling.

The selected application policy uses threshold 0.98. It accepts 214/320 episodes (66.875% raw coverage) and records **one wrong accepted decision**, or 0.467% error among accepted episodes. Family/group-weighted coverage is 72.92%; the corresponding weighted cost is 0.03359 under the pilot cost model versus 0.1 for always-review. These are observed pilot point estimates; because the promotion limits were left unset and rare-error uncertainty is not certified, they are not release acceptance results.

## 15.5 Multi-question state-first execution is now measured, but only boundedly

The sharing study intentionally selected a state-first Qwen4B semantic-feature profile from development data only (`5f993e269e342aeec63c`) so the execution experiment did not depend on the final model-selection outcome. Three policy-development source states supplied genuine same-state Q=1 and Q=4 semantic comparisons. Q=16 semantic evaluation was not applicable because those states did not contain 16 distinct reviewed questions.

Across the Q=1/Q=4 semantic rows, nested state→question→candidate execution stayed within approximately **7.2e-7 maximum probability difference** from full sequential execution, with zero argmax changes and zero policy changes. Reordering questions, adding a really distinct available question, renaming opaque IDs and permuting candidates also produced no recorded policy changes in these bounded cases. This closes the earlier duplicate-question fixture defect as an implementation test, while leaving broad semantic independence open.

For the three Q=4 semantic states, full-sequential medians were approximately 1.65–1.75 s. Nested-cold medians were approximately 1.38–1.41 s and nested-warm medians approximately 1.23–1.29 s. Thus genuine state-first reuse reduced the measured Q=4 model/head work on these cases, but the benchmark excludes server queueing, network, request parsing and startup.

The separate 12-cell mechanics grid spans actual state lengths 64, 256 and 1,024 tokens with Q in {1,4,16} and K values chosen from {2,4,8,16}. Every cell passed the declared scaled-feature tolerance. The largest recorded scaled feature delta is about 2.06e-6. The performance effect grows with shared state length: for example, at L=1024, Q=4, K=4, full sequential measured about 18.29 s, nested cold 3.16 s and nested warm 2.07 s; at L=1024, Q=16, K=2, the corresponding medians were about 36.48 s, 5.89 s and 4.81 s. These synthetic rows establish execution mechanics, not semantic quality at Q=16 or long natural-document competence.

## 15.6 The selected-model bundle closes the research-to-Rust handoff

The first bundle attempt on an L4 was skipped by its FP32 free-memory planning guard after the scientific study itself had completed. A dedicated A100 continuation restored the immutable study and retried only the selected-model bundle stage. The export completed without retraining, reselection or new final evaluation.

The reference bundle has SHA-256:

`4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`

Base Qwen weights are **not** duplicated in the approximately 4.3 MB bundle; they remain an immutable external checkpoint dependency. The bundle contains the selected fitted artifacts, tokenizer assets/identity, model and probability contracts, exact nonfinal golden fixtures, tensor inventory, implementation snapshot and Rust handoff notes.

A fresh Python reload of the exported bundle passes its reference parity check on four nonfinal questions:

- maximum probability delta: **3.6673555e-6**;
- selected-ID changes: **0**;
- independent NumPy/f64 head-algebra check: **passed**;
- largest saved head-algebra logit delta: approximately **2.07e-6**.

This validates artifact integrity and reference reload behavior. It is not Rust, Metal, Candle, GGUF or ONNX parity. The bundle deliberately exposes that boundary so native implementation work can proceed against exact fixtures instead of reverse-engineering the notebook.

## 15.7 Roadmap consequence

The project can now stop treating model identity as the main blocker for Rust engineering. **Profile `a047d6802c3f06f085b8` is the provisional integration target.** The appropriate next implementation sequence is exported head/probability algebra → exact tokenizer/state-first renderer → full-backbone hidden-state parity on the intended local runtime → supported nested sharing → service integration and target-device measurements.

The remaining model-research gates are promotion gates, not prerequisites for beginning the port: independent review/natural-data confirmation, explicit release-quality/resource bounds, released Laya/GLiClass external baselines, and any conditional P2 quantization or teacher/student study. A later reviewed confirmation can reject the provisional candidate for release without invalidating its value as the fixed implementation target used to build and test the Rust/native path.


# 16. Phase 3A: full-hybrid BranchableState, batched Q/K execution, and the Rust handoff

## 16.1 Scope, lineage, and non-goals

Phase 3A run `20260920T024056Z` is an implementation/systems study over the already selected E11 profile:

```text
profile_id: a047d6802c3f06f085b8
backbone:   Qwen/Qwen3.5-4B-Base
weights:    frozen
renderer:   state-first
rejection:  score-summary
bundle:     4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332
```

The saved summary explicitly records `model_changed=false`, `training_performed=false`, and `selection_performed=false`. Phase 3A therefore cannot be interpreted as a new model-quality result or post-final reselection. It asks a narrower question: **can the selected numerical function be executed through a reusable, fully isolated hybrid continuation state and vectorized across independent questions/candidates without changing the declared decisions?** [E11; E12]

The notebook reports `completed_notebook_scope`, three semantic smoke cases, `semantic_batched_parity_all_pass=true`, `high_k_parity_all_pass=true`, same-process repeatability maximum probability delta `0.0`, and zero repeatability argmax changes. It also retains `release_quality_claim=false` and `rust_metal_parity_claim=false`. These flags are evidence boundaries rather than deficiencies to be silently removed. [E12]

The public reference repository for this integration line is:

<https://huggingface.co/cowWhySo/OpenKind-Qwen3.5-4B-StateFirst>

This publication improves inspectability and handoff. It does not imply that the frozen Qwen base weights were retrained, that TypeSafe RLCD was reproduced, or that OpenKind has passed a release-quality/natural-data or Rust/Metal gate. [PUB1]

## 16.2 Why generic KV-cache batching was insufficient

The first Phase 3A draft exposed a concrete Qwen3.5 runtime mismatch. Transformers 5.17.0 exposes a top-level cache `batch_repeat_interleave()` helper, but Qwen3.5's hybrid cache contains `LinearAttentionLayer` entries that do not implement that generic repeat operation. Treating the cache as ordinary attention KV therefore fails before the semantic benchmark.

The corrected reference deliberately **does not** monkey-patch only the attention portion. It treats the continuation state as a heterogeneous object:

```text
Qwen3.5 continuation state
├── 8 full-attention layers
│   ├── key cache
│   └── value cache
└── 24 DeltaNet / linear-attention layers
    ├── recurrent state
    └── convolution state
```

Logical token position and execution/profile identity are also part of the reusable-state contract even though they are not simply tensor payloads.

The corrected fan-out is conceptually:

```python
root = deep_copy(full_hybrid_cache)       # root remains reusable
lanes = deep_copy(root)
lanes.reorder_cache([0, 0, 0, 0])        # one root -> four complete-state lanes

selected = deep_copy(lanes)
selected.reorder_cache([2])               # gather one lane
```

`reorder_cache()` dispatches the index operation through each supported layer representation; repeated indices provide fan-out and arbitrary indices provide gather/select. The Phase 3A self-test performs fan-out then selection and requires the selected full-state fingerprint to match the original while confirming the source root was not mutated. The important contract is **complete-state isolation**, not the Python method name itself. A Rust backend can implement a more efficient representation, but cloning/gathering only attention KV would not be equivalent. [E12]

This finding strengthens the architectural reason for a backend-neutral `BranchableState` abstraction. For the selected Qwen profile, at minimum it must bind:

- attention KV;
- recurrent DeltaNet state;
- convolution state;
- logical position;
- profile/model/tokenizer/renderer/execution identity;
- storage-byte accounting;
- immutable-root fork semantics;
- batched fork and gather/select semantics.

## 16.3 Batched execution and numerical gates

Phase 3A keeps three execution strategies separate:

```text
repeated_full
    encode each question/candidate through its complete prompt independently

nested_sequential
    prefill state once
    fork question states one at a time
    fork candidate states one at a time/bucket

nested_batched
    prefill state once
    fan out compatible question lanes
    advance questions breadth-first
    fan out compatible candidate lanes
    advance candidates breadth-first
```

Question and candidate suffixes are grouped by exact or explicitly declared compatible length rather than padding recurrent streams merely to make a batch. This matters because the selected backbone is not a pure attention transformer; inserting padding or advancing recurrent state on dummy positions would define a different execution graph.

The semantic smoke gate passes for all three tested states, and the high-cardinality systems gate also passes. The recorded same-process repeatability comparison has maximum probability delta zero and zero argmax changes. These are implementation-equivalence observations for the selected nonfinal fixtures; they are not estimates of semantic error rate, high-K calibrated accuracy, or future backend failure probability. [E12]

## 16.4 Short semantic Q scaling: parity passes, speed does not yet look Jev-like

The three semantic smoke states each contain Q=1 and Q=4 measurements. Taking the median across the three per-case medians gives:

| Mode | Q=1 median | Q=4 median | `T(4)/T(1)` |
|---|---:|---:|---:|
| `repeated_full` | **265.0 ms** | **869.3 ms** | **3.28×** |
| `nested_sequential_cold` | 456.0 ms | 1,260.9 ms | 2.77× |
| `nested_batched_cold` | 380.8 ms | 1,058.4 ms | 2.78× |
| `nested_batched_warm` | 288.4 ms | 962.4 ms | 3.34× |

The negative result matters: on these short real semantic requests, warm nested batching is approximately **10.7% slower** than repeated-full execution at Q=4. The tested selected profile therefore does **not** yet reproduce the nearly flat Q=1→4 latency slope reported for hosted Jev in P22, and it would be misleading to present Phase 3A as a universal multi-question speedup. [E12; P22]

The likely systems interpretation is that, at these short state lengths, the fixed costs of creating/reindexing branch state, arranging exact-length buckets, and running the question/candidate layers are large enough that avoided state re-encoding does not dominate. That interpretation should be confirmed by component profiling rather than assumed as a proven causal decomposition.

## 16.5 Shared-state length changes the operating point

The mechanical grid provides the complementary result. These inputs are systems fixtures rather than independently labeled natural semantic cases, but they isolate the cost structure:

| State / shape | Repeated full | Nested sequential cold | Nested batched cold | Nested batched warm | Warm speedup vs repeated |
|---|---:|---:|---:|---:|---:|
| L=64, Q=4, K=4 | 1,375.4 ms | 1,918.5 ms | 597.9 ms | 514.9 ms | **2.67×** |
| L=256, Q=4, K=4 | 2,934.3 ms | 1,962.4 ms | 677.0 ms | 507.6 ms | **5.78×** |
| L=1024, Q=4, K=4 | 9,448.1 ms | 2,348.7 ms | 1,101.2 ms | 516.5 ms | **18.29×** |
| L=1024, Q=16, K=2 | 18,928.6 ms | 4,884.9 ms | 2,506.3 ms | 1,922.0 ms | **9.85×** |

The qualitative transition is clear. At L=64, sequential sharing loses to repeated execution while batched sharing already wins. At L=256, even sequential sharing becomes useful. At L=1024, repeated state computation dominates so strongly that both cold and warm shared-state execution are substantially faster. Phase 3A therefore supports the **state-first sharing architecture** but rejects a single unconditional execution strategy. [E12]

Memory moves in the opposite direction. At L=1024/Q=16/K=2, recorded peak allocated memory is approximately:

- repeated full: **16.42 GiB**;
- nested sequential cold: **16.63 GiB**;
- nested batched cold: **18.67 GiB**;
- nested batched warm: **18.55 GiB**.

The roughly 2.25 GiB peak increase from repeated-full to batched-cold is a real scheduler/admission cost. A runtime selecting the lower-latency path without checking branch expansion and available memory would be incomplete.

## 16.6 The scheduler is now an architectural component

The measured result implies at least three first-class execution plans:

```rust
enum ExecutionPlan {
    RepeatedFull,
    NestedSequential,
    NestedBatched {
        question_batch: usize,
        candidate_batch: usize,
    },
}
```

The exact Rust API may differ, but the choice cannot be hidden inside a generic “fast mode.” A planner needs observable features such as:

```text
state token length
Q independent questions
per-question K
question suffix lengths
candidate suffix lengths
exact-length bucket occupancy
cold vs warm reusable state
BranchableState bytes per lane
temporary branch expansion
available device memory
backend / precision / kernel identity
```

The next target-machine work should separately profile:

```text
render/tokenize
state prefill
root clone/snapshot
fan-out/reindex
question forward
question-state materialization
candidate fan-out
candidate forward
head + rejection + calibration + policy
device synchronization
serialization
```

This decomposition is the purpose of a possible **Phase 3A.1 scheduler/crossover notebook**. It is conditional, not a prerequisite for starting Rust parity. If native profiling already gives stable component costs and crossover rules, another Colab adds little. If the Rust backend obscures the cause of a crossover or differs materially from the Python reference, Phase 3A.1 should freeze the model/profile and study only execution cost. It should not reopen model search.

## 16.7 Rust handoff and acceptance boundary

Phase 3A changes the Rust handoff in one important respect. Native engineering no longer needs to infer what “branch the Qwen cache” means. The reference behavior is now explicit:

1. reproduce head/probability algebra;
2. reproduce exact tokenizer and state-first rendering;
3. reproduce full-backbone hidden features/distributions;
4. construct a complete profile-bound state root;
5. prove the root is immutable under fork/fan-out;
6. support one-lane fork and batched fan-out/gather;
7. reproduce sequential nested execution;
8. reproduce batched Q/K parity;
9. measure strategy crossover on the Mac rather than importing A100 thresholds.

A successful Rust `BranchableState` implementation may use different tensor layouts, copy-on-write storage, preallocated slabs, gather kernels, MLX/Metal-native buffers, or another backend representation. Equivalence is judged by finalized input identity, full probabilities, argmax/policy behavior, branch isolation, and declared numeric tolerances. It is not judged by whether Rust mimics Python data structures. The first backend to exercise this clause is the Phase 3M MLX/Metal parity backend (feature-gated `mlx`): MLX immutable refcounted arrays execute under one process-wide mutex on one explicit cross-thread GPU stream, against the same frozen Phase 3B fixtures. The pinned-base FP32 `ReferenceOps` path passes full, nested, and variable-length vectorized batch decision gates; vectorized execution is available only when explicitly forced while matched performance evidence is collected. The separately gated native-BF16 reference profile fails full and nested probability tolerance (`0.00610` and `0.02658`) and remains unpromoted. The first packed FP32 custom-Metal candidate also passes full and nested parity, but it remains opt-in because its same-host smoke sweep was 14–28% slower than `ReferenceOps`; numerical qualification is necessary but not sufficient for production promotion. The forced daemon request path and bounded memory admission/recovery results are recorded in the 22 September Phase 3M follow-up. The explicit adapter for the downloaded `mlx-community/Qwen3.5-4B-MLX-bf16` export loads and executes, but fails the frozen model-parity gates because its source model and conversion differ from the pinned `Qwen/Qwen3.5-4B-Base` target. Candle CPU remains the oracle.

The public Hugging Face repository should be treated as the public identity of this reference line, while the immutable profile ID and bundle hash remain the stronger experiment identifiers. A future LoRA, weight-quantized, different-renderer, different-rejection, or otherwise behavior-changing artifact must receive a distinct model/execution profile and its own quality/equivalence record. [E11; E12; PUB1]

At the RUST7 checkpoint, all nine handoff steps passed for the frozen fixtures on the correctness-first CPU path: the native path verified both checkpoint shards, executed embedding, all 32 decoder blocks, final RMSNorm, candidate features, and Qwen-specific cached continuation; the backend-neutral branch contract held root immutability, fork, fan-out, and gather; sequential nested execution matched the full-sequence oracle exactly; batched Q/K execution was exactly equal to the sequential baseline; and the strategy crossover had been measured on the named Mac with a recorded threshold rather than imported A100 numbers. RUST8 later implements high-cardinality admission, cache/snapshot persistence contracts, and direct service registration. RUST9 reruns the complete ladder and benchmark from a clean subject commit. RUST10 adds the named-machine native follow-up campaign: bounded model-backed K=32/64/128/255 completion and memory stability, structural fresh-process persistence replay, native service lifecycle smoke, and the Rust 1.88 workspace floor. Restored candidate-feature/decision replay, practical high-K latency, Metal, load/soak, and release promotion remain open. [E13; RUST1–RUST10]

---

# 17. Phase 3B: layer-localized backbone reference and Rust parity checkpoint

## 17.1 Frozen identity and exported evidence

Phase 3B run `20260920T152206Z` uses the same selected identity as E11 and E12:

```text
profile_id:   a047d6802c3f06f085b8
bundle:       4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332
base_model:   Qwen/Qwen3.5-4B-Base
base_revision: 1001bb4d826a52d1f399e183466143f4da7b741b
renderer:     state_first
rejection:    score-summary
```

The saved summary records `training_performed=false`, `model_selection_performed=false`, and `model_modified=false`. Phase 3B therefore adds implementation evidence without redefining the learned model or reopening selection. [E13]

The export has four token-fixture records and 47 FP32 vectors:

| Reference group | Count | Purpose |
|---|---:|---|
| Embedding, 32 decoder layers, final RMSNorm | 34 | Find the first divergent full-sequence stage |
| Full-sequence candidate features | 10 | Reproduce the four published decision fixtures |
| Root, question, and candidate continuation vectors | 3 | Check cached continuation after full-sequence parity |

The architecture contract fixes a 2,560-wide, 32-layer hybrid decoder with 24 DeltaNet/linear-attention layers and 8 full-attention layers. It also records the exact 4,205,751,296-parameter inventory and base revision. A native implementation must treat recurrent DeltaNet state, convolution state, and full-attention KV as separate parts of the continuation contract. [E13]

## 17.2 Hidden-state diagnostics do not replace decision gates

Fresh Phase 3B full-sequence candidate features differ from the earlier bundle features by at most `4.9591064453125e-05`. The saved cached candidate continuation differs from the fresh full-sequence candidate by `1.9073486328125e-05`. Both values pass the notebook's `1e-4` internal self-consistency guard. They measure same-reference reconstruction and help localize errors; they do not create an arbitrary native hidden-state tolerance. [E13]

The selected implementation acceptance contract remains:

- maximum probability delta `0.005`;
- ordering tolerance `1e-5`;
- zero selected-ID changes;
- zero directed-policy changes.

Rust bring-up should report max-absolute error, RMS error, and cosine similarity at every exported stage. It should stop at the first material divergence. Final acceptance still depends on the probability and decision gates above. [E13]

## 17.3 Current Rust boundary

The current Rust implementation establishes eleven bounded results:

1. The immutable `ModelExecutionProfile` records the selected identity, calibration, policy, and numerical tolerances. It does not change `DecisionEngine`.
2. The selected score-summary head reproduces the exported golden probabilities, selection, semantic-none behavior, and review policy with f64 host-side algebra.
3. The digest-locked offline tokenizer reproduces all four exported token records exactly. The Phase 3B loader validates the architecture and 426-entry parameter inventory. It also validates layer order, 47 tensor records and hashes, continuation positions, and candidate-feature probability replay.
4. The native embedding path verifies the immutable checkpoint config and safetensors index, the 5.3 GB embedding shard's size and SHA-256, the BF16 `[248320, 2560]` tensor layout, token bounds, finite values, and BF16-to-FP32 widening. The diagnostic sequence's last-token row matches `diagnostic.embedding` exactly: maximum absolute error 0, RMS error 0, cosine similarity 1. [RUST2]
5. The Candle 0.8.0 CPU path verifies the second checkpoint shard and executes the frozen sequence through 24 DeltaNet blocks, 8 full-attention blocks, and final RMSNorm in FP32. Across the 34-stage trace, embedding is exact and final RMSNorm has maximum absolute error `5.8174e-05`, RMS `1.1531e-05`, and cosine `0.999999999993`. Across 10 candidate sequences, maximum feature delta is `1.0300e-04`, maximum probability delta is `4.5869e-06`, and there are zero argmax or policy changes. Hidden-state deltas remain localization diagnostics, not a new acceptance tolerance. [RUST3]
6. The Qwen-specific continuation state contains full-attention KV, DeltaNet recurrent state, convolution state, and absolute position. The exported branch preserves positions `98 → 109 → 121`, accounts for exactly `59,899,904` root-state bytes, leaves the root unchanged, and produces a candidate vector exactly equal to this native implementation's independent full-sequence path. Its maximum absolute delta from the saved A100 cached candidate is `6.8665e-05`. [RUST3]

7. The backend-neutral `BranchableState`/`BranchBatch` contract is owned by `openkind-runtime`, while the Qwen backend supplies its concrete implementation. `SchedulingFingerprint` and `ContentFingerprint` are distinct types. The exact `59,899,904`-byte Phase 3B root is tensor payload only (`6,422,528` attention KV, `50,331,648` recurrent, `3,145,728` convolution); it does not claim metadata, allocator, mapped weights, or scratch. Process admission adds those concerns separately. Root isolation, reorder/duplicate gather, and cached-versus-full content identity still pass exactly (`0.0`). [RUST4; RUST8]

8. Sequential nested `state → question → candidate` execution prefills each fixture case's shared state exactly once, then evaluates every question and candidate through immutable `BranchableState` forks with fail-closed position and immutability verification. The checkpoint-gated native nested stage executes all four Phase 3B questions and all 10 candidates: maximum probability delta against the golden head fixtures is `4.5869e-06` under the unchanged `0.005` contract, with zero argmax and zero policy changes. The retained root equals an independent fresh prefill exactly after all fork work, question and candidate replay from the retained fork states is exact, reversed sibling-order advancement is exact, and every position matches the exported token fixtures. The `repeated_full` oracle agrees exactly: cached-versus-full candidate features are `0.0` for all 10 candidates and every cached continuation state's strict fingerprint equals its independent full-sequence state. Feature deltas against the frozen Phase 3B vectors reach `1.0300e-04`, and the traced continuation vectors replay at root `1.1444e-04`, question `7.8201e-05`, and candidate `6.8665e-05` — the same localization diagnostics recorded by results 5–7, not new tolerances. [RUST5]

9. Breadth-first Q/K execution proves the state topology: immutable `fork_batch` question lanes and per-question candidate lanes remain isolated and exactly equal to the sequential baseline for all frozen fixtures. Per-lane executor calls remain the current CPU primitive, so this is **state-semantic batching, not compute-vectorized model forward**. `BackendCapabilities` encodes that distinction and explicit lane limits. The current CPU scheduler therefore prefers `NestedSequential`; `NestedBatched` becomes eligible only for a backend with vectorized question and candidate forward. [RUST6; RUST8]

10. The adaptive scheduler promotes all three strategies behind `run_strategy`/`run_repeated_full` with call and token accounting. The commit-stamped named M4 Max rerun measured five warm CPU workloads with feature parity asserted in every repetition. The CPU-default `NestedSequential` path beat `repeated_full` in every cell by 1.294×–2.018×; the per-lane `NestedBatched` topology stayed within 2.9% of sequential. `T(3)/T(1)` was 2.815 repeated versus 2.170 sequential and 2.233 batched, and post-benchmark peak resident memory was 10.968 GiB. The lowest measured token-work ratio remains `2.52`; the policy uses `LOWEST_MEASURED_SHARED_SAVINGS_RATIO = 2.52` instead of presenting 2.0 as measured. Admission separately exposes exact tensor payload, observed process peak, forward scratch, allocator headroom, hard process limits, and vectorized lane ceilings. [RUST7–RUST9]

11. Phase 3.9a exercises K=32/64/128/255 and mixed Q/K through estimator/admission gates without allocating a naïve K-wide Qwen state fan-out. Phase 3.9b completes model-backed candidate continuations for K=32/64/128/255 on the named M4 Max with invariant root and constant memory footprint. `BranchStateCache` enforces tenant isolation, TTL, tensor-byte LRU eviction, and strict content keys. The Qwen snapshot format is atomic, versioned, envelope-digested, identity/layout checked, and strict-content verified; two separate processes pass structural fresh-process replay. `Qwen35DecisionEngine` loads the pinned tokenizer/backbone/head directly behind `EngineRegistry`, bounds concurrent plus queued requests, retains permits when a caller cancels already-running blocking work, returns overload distinctly, requires caller-visible `__none__`, preserves none mass, and derives confidence from normalized entropy. A real native request, overload rejection (HTTP 529), cancellation/recovery, clean shutdown, and 20 health probes pass in native service smoke. RUST10 records this bounded smoke checkpoint. The later full replay and named-machine CPU load/soak results are recorded below. Practical high-K latency remains conditional systems work. [RUST8; RUST10; RUST11]

The `1e-5` absolute-logit tolerance remains the Phase 3.1 fixed-feature algebra gate. Phase 3B's own freshly exported candidate vectors replay against the earlier fixed-feature logit fixture with maximum absolute delta `9.6905e-05`; the native features reach `2.5652e-04` against that older fixture while preserving probabilities and discrete decisions. Therefore the backbone result is judged by the documented Phase 3B hidden diagnostics plus probability, argmax, and policy gates; no tolerance is widened or retroactively redefined. [E13; RUST1; RUST3]

The full restored candidate-feature and decision replay in Phase 3.10 now passes. Phase 3.11's queue-inclusive native CPU service load, deadline, recovery, memory, and 30-minute soak gates also pass on the named M4 Max, with a release-mode daemon hash recorded in the [RUST11 evidence report](../verification/native-service-gate/2026-09-22-rerun2/README.md). The [22 September Phase 3M follow-up](../verification/phase3m-2026-09-22/README.md) records pinned-base FP32 variable-length vectorized parity, a forced daemon vectorized request, and bounded unified-memory admission/recovery evidence; native BF16 full/nested Gate B fails the frozen probability tolerance. The subsequent flat-field and candidate-pooling diagnostics reject those two native compute graphs (§17.4). Practical high-K latency, full-request batch/kernel performance, reviewed task quality, MLX service load/soak, and official release promotion remain open. **CPU native parity and the separately gated pinned-base MLX FP32 parity path now pass; accelerated production promotion does not follow from either result.** Direct backend registration, wire-level semantic-none mapping, bounded high-K completion, full fresh-process persistence replay, and native CPU service gates are recorded as passed. [E12; E13; RUST4–RUST11; RUSTM1; RUSTM2; 3M follow-up]

## 17.4 Ported flat-field execution and paired MLX comparison

The public Python Qwen2.5 constrained-decoding implementation evaluates field
suffixes from one shared prefix and reads answer-token logits. OpenKind ports
that **execution traversal** to the selected Qwen3.5 FP32 Rust backend as an
opt-in diagnostic: prefill one state-first root, concatenate each question and
candidate suffix, fork a complete lane from the root, and evaluate right-padded
groups of at most eight. The candidate-feature score-summary readout remains
unchanged. The existing `nested_batched` path instead evaluates each question
once and branches its candidates from the resulting question state. This is a
comparison of execution schedules on one model and readout, not a Qwen2.5
loader, a port of the Python token-logit decision rule, or a new Jev model.
[P21; RUSTM2]

The paired test used pinned `Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b`, profile
`a047d6802c3f06f085b8`, MLX 0.32.2 FP32 `ReferenceOps`, and the 36-GiB
`Mac16,5` M4 Max. Each strategy ran in a fresh process in two reversed orders.
The first sample per process was excluded; Q2/K2 retained three timed samples,
Q8/K4 two. The timed region includes prefill, continuation, and readout but
excludes model load, rendering, tokenization, validation, transport, and
queueing. Q8/K4 repeats frozen token suffixes as a load shape. These are not
natural-document quality or full-request service measurements. [RUSTM2]

| Shape | Existing nested batched median, pairs 1 / 2 | Flat median, pairs 1 / 2 | Flat latency change |
|---|---:|---:|---:|
| Q2/K2 | 1.342 / 1.268 s | 1.422 / 1.421 s | +6.0% / +12.1% |
| Q8/K4 | 5.005 / 5.005 s | 8.222 / 8.232 s | +64.3% / +64.5% |

Flat execution reduces physical forward calls from **4 to 2** at Q2/K2 and
**10 to 5** at Q8/K4. It repeats question tokens per candidate, while padded
token slots rise from **0 to 2** and **14 to 104**, respectively. The flat
suffix stage alone takes 868–875 ms at Q2/K2 and 7,594–7,598 ms at Q8/K4;
the corresponding nested question plus candidate stages take 737–795 ms and
4,434–4,436 ms. Fewer forwards therefore do not establish lower latency.
Peak active MLX allocation is about 14.0 versus 14.38 GiB at Q2/K2 and
16.49 versus 15.43 GiB at Q8/K4 (nested versus flat); peak process RSS is
mixed across pairs. These few samples do not establish tail latency or a
stable memory ratio. [RUSTM2]

Selections match. The largest paired probability differences are `2.71e-6`
at Q2/K2 and `8.08e-7` at Q8/K4, below the fixed `0.005` tolerance. Every
top probability is at least `0.01099` from the `0.98` policy threshold, so
the unchanged selections imply unchanged policy actions under the fixed rule.
Formal pinned full-sequence and nested MLX gates independently pass, with
maximum probability errors `1.514845e-6` and `1.378739e-5`, zero selection
and policy changes, and complete nested isolation/position/unequal-length
checks. The formal gates validate the existing backbone; the paired flat
comparison supplies the direct flat-path parity evidence. [RUSTM2]

The separate same-position cross-question candidate-pooling diagnostic is
also slower than ordinary nested batching in its stage replay: **1.5%** at
Q2/K2 and **7.8%** at Q8/K4. Its fresh-process full-request baseline drifts
at Q8/K4 and does not establish a stable speedup. Neither diagnostic meets
the promotion threshold. The service and automatic scheduler retain the
current graph. [RUSTM2]

---

# 18. Phase 4A–4E: locked benchmark, applicability experiments, and audit gate

## 18.1 Scope, status, and evidence boundary

Phase 4A is the first OpenKind study in this record that combines many questions over the same natural document with a trained question-conditioned readout. It uses the selected state-first Qwen3.5-4B-Base identity from E11–E13, but it asks a new modeling question: whether one frozen state representation can support ContractNLI entailment judgments and QASPER answerability decisions without re-encoding each state for every question. [E14]

The original v0.8.2 evidence cutoff is 22 September 2026; its 24 September addenda preserve later audit/probe evidence. Version 0.8.3 adds the 25–26 September frozen and matched-adaptation results in §§18.20–18.22. Version 0.8.4 adds the completed retention follow-on and the then-prospective source-label comparison in §§18.23–18.24. Version 0.8.5 records that comparison's completed result and current synthesis in §§18.25–18.26 without changing historical measurements or authorizations. These statuses remain distinct:

| Stage | Recorded artifact status | Permitted interpretation |
|---|---|---|
| Phase 4A.0 corpus/cache | Completed and hash-locked before model predictions | Reproducible benchmark and feature-cache identity |
| Phase 4A.1 original B1 StateQuery | Training, non-final evaluation, checkpoint, and exploratory model lock completed | Completed non-final model result; no promotion |
| Phase 4A.2 B0 matched control | Training, non-final evaluation, and comparison-only result lock completed | Completed control result; final unavailable |
| Phase 4A.2 B1/B2/B2R and weight-8 follow-on | Training, selected checkpoints, non-final evaluations, and comparison-only result locks completed | Completed architecture/weight sweep; no eligible checkpoint |
| Phase 4B.2 B2 weight-cap-12 | Training, selected checkpoint, non-final evaluation, prediction tables, and comparison-only result lock completed | Completed negative dose-response result; scalar sweep closed |
| Phase 4B.2.1 stratified applicability | Training and non-final threshold transfer completed; selected epoch 8 | Best surviving parent; development and policy gates still fail |
| Phase 4B.3 pairwise applicability | Training and non-final threshold transfer completed; selected epoch 9 | Development pass does not transfer; child rejected |
| Phase 4C head-only continuation | Three child epochs evaluated against the frozen parent; epoch 0 retained | Isolated head refit rejected on proper scores |
| Phase 4D evidence residual | Eight child epochs evaluated; zero-init epoch 0 retained | Small residual rejected; strict-JSON/NaN repair validated |
| Phase 4E-A2/A3/A4 | Repaired audit, 51-row adjudication, and source-alignment preflight recorded | Review proposals and source candidates; no committed gold or serializer repair |
| Later 4E-A 27-case disposition | Assistant-reviewed case and tie-policy lock frozen | Audit decision only; no independent release review or training authorization |
| Exploratory 4E-B.0 prefill probe | Three seed-17 non-final arms evaluated; all gates fail; separate result lock absent | Reject this cheap readout only; final closed |
| Exploratory 4E-B.1 option-logit audit | Fixed 325-question non-final sample, result lock and row metrics verified | Candidate-ranking signal, failed semantic-none behavior; no model promotion |
| Exploratory 4E-B.2 candidate sweep | Completed locked ranking/rejection comparison | Ranking signal with false-none/grid failures; no promotion. [E28] |
| M1 / M2.1 finalized-input and frozen-checkpoint diagnostic | Completed v0.3.2 FP32 J0/J1 × two-budget comparison | Evidence access and none behavior remain separable; no broad checkpoint winner. [E29] |
| M2.2 matched decision-LoRA pilot | Completed v0.4.1 BF16 J0/J1/J2/J3 experiment; both fits select update 80 | ContractNLI specialization improves; entailment/QASPER retention and complete screens fail; no promotion. [E30] |
| Parent-KL retention follow-on | Completed v0.5.0 BF16 J4/J5 experiment; both fit120 and select frozen0 | Fixed80 probability recovery versus specialists does not satisfy joint preservation. [E31] |
| Source-label target comparison | Completed v0.6.0 BF16 J6/J7 experiment; both fit120 and select frozen0 | Stronger exposed SNLI learning, failed entailment/QASPER preservation and complete screens; no promotion. [E32] |

No Phase 4 final label or prediction was opened for this documentation update. The original B1 lock has six unset promotion gates, later contracts declare the final split unavailable, and the Phase 4D result is `nonfinal_failed_final_unavailable`. Therefore this chapter records completed non-final comparisons and failure modes; it does not select or promote a Phase 4 model. Legacy Drive folders retain their historical `OpenDecision_...` names as immutable provenance even though the project and new artifacts now use OpenKind. [E14–E27]

## 18.2 Phase 4A.0: corpus, partitions, and frozen feature identity

The corpus manifest joins two natural-document sources under one state/question/option/evidence schema:

| Corpus object | Count |
|---|---:|
| States | 2,192 |
| Questions | 15,368 |
| Options | 25,687 |
| Evidence rows | 21,894 |
| Criteria | 18 |

ContractNLI contributes 10,319 `contract_entailment` questions. QASPER contributes 5,049 `paper_answerability` questions. The source revisions are pinned in the manifest: ContractNLI commit `eced6528dd3c1d14d73f9a87df8f7bdbc03126f9`; QASPER repository revision `fdc9d8214fbab5dd782958601db4d678e6934a54` with parquet/data revision `06806e4608976fc2fac0a090ac425d5b2b29caf4`. Both sources are recorded as CC-BY-4.0 and training-allowed. [E14]

State counts preserve state/component grouping across six partitions:

| Source | Train | Development | Calibration fit | Calibration gate | Policy development | Final | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| ContractNLI | 341 | 70 | 34 | 71 | 39 | 52 | 607 |
| QASPER | 707 | 242 | 119 | 199 | 101 | 217 | 1,585 |
| **Total** | **1,048** | **312** | **153** | **270** | **140** | **269** | **2,192** |

The manifest SHA-256 is `1aad04ca7d37ea8be640a51878857c6d2c501d635e913eca12234cb548f8399d`. The lock binds the exact tables and records `locked_before_model_predictions=true`:

| Locked table | SHA-256 |
|---|---|
| States | `7e6333202ad934e66148724a565b27176fcaad4133a566d2c8f8b57044ef3120` |
| Questions | `549b446f609a9407eceb5df7ef4c0f14779bc73d72396ae749424626d8777434` |
| Options | `cbd40c9162909c6c637296e9c8f408e0ba1bece0aca1786cef98f73d242f6cf2` |
| Evidence | `566e0e69e8f57ed0ce626021647a3b770a89ae5a3eb318e1d4f8bea284f2c6be` |
| Criteria | `1e3e62c93506fc7add11c88dc7232dba9f6c0b9ee8db2ee4b465fb283a91430b` |

This is stronger provenance than a mutable list of dataset names, although it does not independently validate every source label or establish deployment representativeness. [E14]

The frozen feature cache binds the benchmark to profile `a047d6802c3f06f085b8`, bundle `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`, and `Qwen/Qwen3.5-4B-Base` revision `1001bb4d826a52d1f399e183466143f4da7b741b`. It contains 2,192 BF16 state-feature files capped at 1,024 state tokens and 13,725 non-final candidate-conditioned control files. The frozen input-embedding artifact has SHA-256 `19175cf27eb1cb2cb49e1ed3afde552b9588a1c5ae6a004bd3d7efd2e384d093`. Historical reference predictions explicitly exclude final. [E14]

Several safeguards match the conservative benchmark discipline described by SemIf: freeze IDs, labels, semantics, revisions, and metrics before full output inspection; report unlike sources separately rather than collapsing them into one accuracy; preserve source-grouped uncertainty; and keep systems mechanics distinct from semantic quality. Phase 4A implements those principles through its corpus lock, per-source metrics, state bootstrap, and closed final boundary. P23 is methodological context, not an independent audit of Phase 4A or evidence that the two projects use the same benchmark. [P23]

## 18.3 Phase 4A.1: original B1 training and selection

The original B1 StateQuery model trains 26,384,647 parameters over frozen state features and a frozen 635,699,200-parameter input embedding. Development loss selects epoch 1; the subsequent decrease in training loss does not transfer consistently to development:

| Epoch | Train state loss | Development state loss |
|---:|---:|---:|
| **1 (selected)** | **0.616901** | **0.587660** |
| 2 | 0.570560 | 0.640708 |
| 3 | 0.536795 | 0.592321 |
| 4 | 0.524565 | 0.606392 |

This is early overfitting evidence, not merely incomplete optimization. The selected checkpoint SHA-256 is `9b19138268b5de2bb48b461337e88b9de6492a7bc52eff6489525e09eb63d940`. The model lock preserves the corpus lock and fitted temperatures, but its status is `exploratory_locked_gates_unset`; all six promotion thresholds are null. [E14]

## 18.4 Completed non-final comparison: aggregate improvement hides a QASPER collapse

The saved non-final by-source comparison evaluates the historical candidate-conditioned reference, the original B1 StateQuery model, and the completed Phase 4A.2 B0 matched control on the same 1,207 ContractNLI and 692 QASPER decisions. Source-macro rows are unweighted means of the two source metrics. Higher is favorable for accuracy, balanced accuracy, macro F1, and semantic-none recall; lower is favorable for NLL, Brier, and ECE.

| Model | Source | n | Accuracy | Balanced acc. | Macro F1 | NLL | Brier | ECE15 | Semantic-none recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical reference | ContractNLI | 1,207 | 0.2908 | 0.4018 | 0.2849 | 1.8788 | 0.9887 | 0.4211 | 0.3538 |
| Original B1 | ContractNLI | 1,207 | 0.6893 | 0.6089 | 0.6340 | 0.6913 | 0.4220 | 0.0454 | 0.6482 |
| B0 matched control | ContractNLI | 1,207 | 0.5866 | 0.5050 | 0.5252 | 0.8536 | 0.5221 | 0.0511 | 0.7194 |
| B1 balanced | ContractNLI | 1,207 | 0.6943 | 0.6323 | 0.6399 | 0.6633 | 0.4085 | 0.0559 | 0.7391 |
| B2 factorized | ContractNLI | 1,207 | 0.6926 | 0.6281 | 0.6321 | 0.6694 | 0.4104 | 0.0362 | 0.6621 |
| B2R refined | ContractNLI | 1,207 | 0.6703 | 0.6171 | 0.6174 | 0.7202 | 0.4380 | 0.0352 | 0.7787 |
| B2 QASPER weight 8 | ContractNLI | 1,207 | 0.7017 | 0.6426 | 0.6475 | 0.6526 | 0.3993 | 0.0390 | 0.7115 |
| B2 QASPER cap 12 | ContractNLI | 1,207 | 0.6852 | 0.6115 | 0.6315 | 0.7382 | 0.4334 | 0.0571 | 0.8399 |
| Historical reference | QASPER | 692 | 0.4624 | 0.5209 | 0.4067 | 1.4680 | 0.8174 | 0.3750 | 0.6000 |
| Original B1 | QASPER | 692 | 0.8699 | 0.5000 | 0.4652 | 0.3940 | 0.2291 | 0.0432 | 0.0000 |
| B0 matched control | QASPER | 692 | 0.8699 | 0.5000 | 0.4652 | 0.3887 | 0.2269 | 0.0332 | 0.0000 |
| B1 balanced | QASPER | 692 | 0.8020 | 0.5035 | 0.5023 | 0.4808 | 0.2966 | 0.0895 | 0.1000 |
| B2 factorized | QASPER | 692 | 0.8353 | 0.5273 | 0.5290 | 0.4296 | 0.2608 | 0.0456 | 0.1111 |
| B2R refined | QASPER | 692 | 0.8671 | 0.5078 | 0.4851 | 0.4120 | 0.2410 | 0.0611 | 0.0222 |
| B2 QASPER weight 8 | QASPER | 692 | 0.8020 | 0.5318 | 0.5335 | 0.4782 | 0.3014 | 0.0689 | 0.1667 |
| B2 QASPER cap 12 | QASPER | 692 | 0.8439 | 0.5512 | 0.5597 | 0.4229 | 0.2564 | 0.0394 | 0.1556 |

| Model | Accuracy | Balanced acc. | Macro F1 | NLL | Brier | ECE15 | Semantic-none recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical reference | 0.3766 | 0.4613 | 0.3458 | 1.6734 | 0.9031 | 0.3980 | 0.4769 |
| Original B1 | **0.7796** | 0.5545 | 0.5496 | **0.5427** | **0.3256** | 0.0443 | 0.3241 |
| B0 matched control | 0.7283 | 0.5025 | 0.4952 | 0.6211 | 0.3745 | 0.0421 | 0.3597 |
| B1 balanced | 0.7482 | 0.5679 | 0.5711 | 0.5721 | 0.3526 | 0.0727 | 0.4196 |
| B2 factorized | 0.7639 | 0.5777 | 0.5806 | 0.5495 | 0.3356 | **0.0409** | 0.3866 |
| B2R refined | 0.7687 | 0.5624 | 0.5513 | 0.5661 | 0.3395 | 0.0482 | 0.4004 |
| B2 QASPER weight 8 | 0.7519 | **0.5872** | 0.5905 | 0.5654 | 0.3504 | 0.0540 | 0.4391 |
| B2 QASPER cap 12 | 0.7646 | 0.5814 | **0.5956** | 0.5805 | 0.3449 | 0.0483 | **0.4977** |

The original B1 remains the source-macro accuracy and proper-scoring leader, while B2 and the two weight-follow-ons provide the strongest balanced-accuracy, macro-F1, or semantic-none-recall values among the matched follow-ons. None of those aggregate winners resolves applicability. On QASPER, every trained model remains below the predeclared 0.30 semantic-none-recall floor; the best observed gate recall is only 0.1667. The cap-12 arm improves QASPER accuracy, balanced accuracy, macro F1, NLL, Brier, and ECE relative to weight 8, but its none recall falls from 0.1667 to 0.1556. Meanwhile its high ContractNLI none recall is accompanied by 278 false-none decisions among 701 answerable questions. Source-macro recall therefore conceals a source-specific failure and cannot be the promotion statistic. [E14–E16]

The original B1 state-bootstrap accuracy interval is 0.7330–0.7824. B0's corresponding interval is 0.6616–0.7220, and the cap-12 arm records 0.7174–0.7724. These intervals summarize the saved non-final population under state resampling; they do not repair the source-specific failure or establish performance on the closed final split.

## 18.5 Calibration and policy behavior remain non-operational

The original B1 calibration-gate diagnostic pools 1,899 decisions and records 0.7551 accuracy, 0.58299 NLL, 0.04462 ECE15, and 0.5503 semantic-none recall. Policy development records 0.7508 accuracy, 0.59148 NLL, 0.04344 ECE15, and 0.5877 semantic-none recall. Those pooled values again average across a strong majority-class QASPER result and a more discriminating ContractNLI result, so the per-source table remains the controlling diagnostic. [E14]

The declared policy charges 5.0 for a wrong accepted decision and 0.1 for review. Reviewing every case therefore costs 0.1 per decision. The selected thresholds deliver little or no useful automation:

| Model | Threshold | Policy-development coverage | Accepted / wrong | Policy-development cost | Calibration-gate coverage | Accepted / wrong | Gate cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original B1 | 0.608636 | 0.305% | 3 / 0 | 0.09969 | 1.264% | 24 / 1 | 0.10137 |
| B0 matched control | 0.829407 | 1.017% | 10 / 0 | 0.09898 | 0.263% | 5 / 1 | 0.10237 |
| B2 QASPER cap 12 | 0.730910 | 2.747% | 27 / 0 | 0.09725 | 1.422% | 27 / 2 | 0.10384 |

All three policies accept too few cases to show meaningful automation. On the calibration gate, each is more expensive than reviewing everything under the declared cost model. The cap-12 policy's 1.422% coverage and 0.10384 gate cost confirm that modestly broader development coverage does not transfer into useful automation. Calibration and low ECE therefore do not imply an operationally useful acceptance policy. [E14–E16]

## 18.6 Phase 4A.2 B0: matched-control result

Phase 4A.2 introduces a shared experiment contract for the B0 and balanced-B1 arms: source undersampling to the smaller source each epoch, QASPER semantic-none weighting capped at 5.0, development none-recall floors of 0.35 for ContractNLI and 0.30 for QASPER, seed 17, and a selection rule that first requires every floor and then minimizes source-macro NLL. Before eligibility, it minimizes total recall deficit and then NLL. This contract is pinned to the original corpus, feature cache, historical predictions, and selected profile. [E15]

B0 completes six epochs before early stopping from a maximum of 12 and selects epoch 2. Its selected development result has source-macro accuracy 0.7131, NLL 0.6538, and semantic-none recall 0.4336. ContractNLI none recall is 0.8671, but QASPER none recall is 0.0, so the checkpoint is ineligible with a 0.30 total deficit. The later calibrated non-final comparison is reported in §18.4.

The result lock binds selected checkpoint `checkpoints/epoch_002_model.safetensors` with SHA-256 `82f50370b789941b0cfdee55b6cfc786ed5edff55bf743a93fc329e19950f714`. Its status is `comparison_only_final_unavailable`, and `final_opened=false`. B0 therefore answers a diagnostic question: candidate-conditioned frozen features under the balanced recipe do not repair QASPER applicability. It does not yet isolate the value of B1 because the original B1 used a different training recipe. [E15]

## 18.7 Completed Phase 4A.2 architecture and weight-8 sweep

The balanced B1, factorized B2, refined B2R, and B2 QASPER-weight-8 arms all completed under immutable experiment contracts with selected checkpoints, training reports, calibrated non-final evaluations, prediction tables, and `comparison_only_final_unavailable` result locks. The completed gate results are included in §18.4.

The matched architecture comparison separates several effects:

- Balanced B1 improves QASPER gate none recall from 0.0 to 0.1000 while reaching 0.5711 source-macro F1.
- Factorized B2 improves QASPER gate recall to 0.1111 and provides the strongest Phase 4A.2 architecture-arm proper scores: 0.5495 source-macro NLL and 0.3356 Brier.
- B2R raises ContractNLI none recall to 0.7787 but reduces QASPER recall to 0.0222; an additional state-refiner layer does not repair applicability.
- B2 with a QASPER none weight of 8 reaches the best Phase 4A.2 source-macro balanced accuracy (0.5872), macro F1 (0.5905), and QASPER gate none recall (0.1667), but still misses the 0.30 floor.

The weight-8 checkpoint selects epoch 11. Its QASPER development none recall is 0.1681, so it remains ineligible despite useful movement from the original collapse. These completed results justify retaining B2 as the base architecture for one bounded objective study, not opening final or replicating seeds. [E15]

## 18.8 Phase 4B.2 cap 12 closes the scalar-weight sweep

Phase 4B.2 holds B2, seed 17, the optimizer, source-balanced state sampling, partitions, and selection rule fixed while raising only the configured QASPER semantic-none weight cap from 8 to 12. The realized training weight is 8.560185, the uncapped answerable-to-none ratio. The configured cap is therefore no longer binding: higher caps would not increase this weight under the current rule. [E16]

Training runs all 12 epochs and selects epoch 11. The selected development checkpoint records ContractNLI none recall 0.8604 and QASPER none recall 0.1239 (14/113). QASPER misses its floor by 0.1761 and `eligible=false`; 0.1239 is also the largest QASPER development recall observed during the run. The stronger scalar weight therefore fails its primary development criterion before the calibration gate is considered.

On the untouched calibration gate, QASPER none recall is 0.1556 (14/90), precision is 0.3043, and false-none rate is 0.0532. QASPER's ordinary classification improves relative to weight 8—accuracy rises from 0.8020 to 0.8439 and NLL falls from 0.4782 to 0.4229—but none recall falls by 0.0111. ContractNLI moves in the opposite operational direction: none recall rises to 0.8399, but 278 of 701 answerable questions are rejected as none, for a 0.3966 false-none rate. The 0.4977 source-macro none recall is therefore not a balanced success; it averages two incompatible source behaviors.

A read-only diagnostic over the saved prediction tables tests whether thresholding alone could recover the floor. Selecting the highest global none-probability threshold that meets both recall floors on `policy_development` gives 0.3077 QASPER recall but a 0.3345 QASPER false-none rate. Applied once to `calibration_gate`, it gives 0.4778 recall and a 0.3688 false-none rate. QASPER none-probability ROC AUC is 0.5194 on policy development and 0.5729 on the gate. This is weak separability, not merely a poorly selected operating threshold. The threshold diagnostic is not a replacement selection rule and does not alter the locked report. [E16]

The negative result is technically credible. The experiment hash agrees across the contract, training report, evaluation, lock, best pointer, and latest pointer; selected epoch and checkpoint references agree; the evaluation and all three prediction-file hashes independently reproduce; the comparison snapshot exactly matches the report; and 3,837 prediction rows contain no duplicate question IDs, null cells, cross-partition state/question overlap, invalid targets, probability-sum errors, or recomputed metric differences. Final remains unavailable and unopened. [E16]

## 18.9 Phase 4B.2.1 stratified applicability improves recall but not policy transfer

Phase 4B.2.1 keeps factorized B2, seed 17, the locked corpus/features, answer distribution, and non-final partitions fixed. It removes the scalar none multiplier from the joint choice loss, computes candidate ranking only for answerable questions, and trains a binary applicability term with equal source/class mass. The model has 35,843,335 trainable parameters and selects epoch 8. Its experiment SHA-256 is `4313087778356312ac8646486ac8c45731885595b88e7236edcefa5c985d703e`; selected-checkpoint SHA-256 is `87c3b3a4069b64e4a150ad6e0812ce2d953346eb17999865770df508269ed871`. [E17]

The objective improves applicability separation, but the result depends on which frozen threshold and partition are inspected. Selected-epoch raw development QASPER none recall is 0.1770 (20/113), so the training gate never passes. At the later frozen operating threshold, policy-development QASPER recall is 0.2564 (10/39) with a 0.1459 false-none rate; the calibration gate reaches 0.3111 (28/90) with a 0.1545 false-none rate. ContractNLI gate recall/false-none is 0.5751/0.1626. The threshold therefore transfers directionally, but the development QASPER floor remains missed.

The calibration-gate source-macro result is 0.7613 accuracy, 0.5810 balanced accuracy, 0.5952 macro F1, 0.57085 NLL, 0.34969 Brier, and 0.04255 ECE15. QASPER's raw-choice gate recall is still only 0.1556. More importantly, the cost-sensitive policy accepts 14 development decisions with zero errors for cost 0.09858, then accepts 32 gate decisions with seven errors for cost 0.11675. Because review-all costs 0.10, the transferred policy fails even though the gate applicability threshold clears its QASPER recall and false-none diagnostics. The result remains `comparison_only_final_unavailable`. [E17]

## 18.10 Phase 4B.3 pairwise applicability passes development but not transfer

Phase 4B.3 adds a 0.25-weight pairwise applicability term with zero margin and otherwise retains the stratified experiment boundary. The 35,843,335-parameter run selects epoch 9. At selection, both development operating points pass their declared recall/false-none constraints: ContractNLI records 0.5811 recall and 0.1971 false-none; QASPER records 0.3451 recall (39/113) and 0.1936 false-none. Source-macro development NLL/Brier is 0.56555/0.33557. This is a real development improvement, but not a transferable one. [E18]

On policy development, the frozen QASPER threshold yields 0.2821 recall, 0.1744 false-none, 0.5536 ROC AUC, and 0.1565 average precision. On the calibration gate, ContractNLI false-none rises to 0.2111; QASPER reaches 0.3333 recall but a 0.2110 false-none rate. Both sources therefore miss the 0.20 false-none ceiling. QASPER gate ROC AUC/AP is 0.5925/0.1758, and raw-choice QASPER none recall is 0.0. The policy accepts 6/0 development decisions at cost 0.09939, but 8/1 gate decisions at cost 0.10221. A development pass is therefore insufficient: pairwise applicability is rejected on threshold transfer, proper-score context, and policy cost. Its lock remains `comparison_only_final_unavailable`. [E18]

## 18.11 Phase 4C head-only continuation is rejected

Phase 4C asks whether the existing stratified checkpoint merely needs a better applicability head. It freezes the parent representation and fits only the 1,771,009-parameter applicability MLP. The parent starts at source-macro development NLL 0.56457 and Brier 0.33934. All three child epochs are worse on NLL. The best child by that metric, epoch 3, records NLL 0.58022 and Brier 0.34645—regressions of 0.01565 (2.77%) and 0.00712 (2.10%). QASPER operating recall moves from 34/113 (0.3009) to 36/113 (0.3186), but the two added true positives do not justify the proper-score loss. Epoch 0, the unchanged parent, is retained; no child checkpoint is promoted. [E19]

This result narrows the diagnosis. The weakness is not plausibly fixed by continuing to optimize the existing applicability MLP in isolation. It does not prove that every alternative head must fail, but it closes this exact head-only continuation and argues against another local head hyperparameter sweep.

## 18.12 Phase 4D evidence residual gains one true positive while proper scores regress

Phase 4D adds a zero-initialized 19→64→1 evidence-aware residual to the frozen parent applicability logit. Only 1,345 parameters are trainable. The implementation also repairs the earlier non-standard `NaN` artifact issue: unavailable legacy values serialize as JSON `null`, every output is written with `allow_nan=False`, and the saved contract/report/lock parse as strict JSON. [E20]

All eight trained epochs are ineligible. The strongest child for QASPER operating recall is epoch 1: recall moves from 34/113 (0.3009) to 35/113 (0.3097), while source-macro development NLL worsens from 0.56457 to 0.59888 (+6.08%) and Brier from 0.33934 to 0.36172 (+6.60%). The 0.35 target requires 40 true positives, so even that child remains five short. Meanwhile the training proxy falls from 0.75076 at epoch 1 to 0.65405 at epoch 8 while evaluation NLL/Brier remain worse. This divergence is direct objective-mismatch evidence, not a reason to continue training. Epoch 0—the unchanged, residual-augmented parent—is retained, and the result status is `nonfinal_failed_final_unavailable`. [E20]

The retained parent's terminal diagnostics are unchanged: QASPER policy-development recall is 0.2564 (10/39), QASPER gate recall/false-none is 0.3111/0.1545, and gate policy cost is 0.11675 with 7 wrong among 32 accepted. Proper-score preservation is recorded only because the zero-initialized epoch-0 parent is selected; no trained residual passes.

## 18.13 Lessons learned and Phase 4E gate

The completed Phase 4 program now supports these bounded conclusions:

| Tested idea | Best relevant number | Decision |
|---|---:|---|
| Scalar none weighting | Cap 12 realizes only 8.5602; QASPER gate recall 0.1556; ContractNLI false-none 0.3966 | Saturated and rejected |
| Source/class-stratified BCE | QASPER operating recall 0.2564 development / 0.3111 gate; gate policy cost 0.11675 | Retain as diagnostic parent, not promotion candidate |
| Pairwise applicability | Development QASPER recall 0.3451, but gate false-none 0.2110 and policy cost 0.10221 | Development gain does not transfer |
| Head-only continuation | +2 QASPER true positives, but NLL/Brier worsen 2.77%/2.10% | Reject child |
| Evidence residual | +1 QASPER true positive, but NLL/Brier worsen 6.08%/6.60% | Reject child |

QASPER separability remains weak. For the retained stratified parent, policy-development ROC AUC/AP is 0.5903/0.1663 at prevalence 39/320 = 0.1219; calibration-gate ROC AUC/AP is 0.6173/0.1841 at prevalence 90/692 = 0.1301. These are above chance but too weak to support a safe thresholded policy. Repeating nearby weights, margins, shallow heads, or handcrafted residuals is unlikely to resolve the underlying representation/label boundary.

These results led to **4E-A, a blinded QASPER error audit**. The original audit requirements were:

1. reconstruct all 70 frozen-threshold policy-development errors (29 false negatives and 41 false positives);
2. add a deterministic state-grouped calibration-gate sample of 30 false negatives and 30 false positives, plus 10 true-positive and 10 true-negative controls;
3. hide labels, model scores, predictions, and error classes from reviewers while preserving state, question, options, and available evidence;
4. collect `answerable`, `semantic_none`, or `ambiguous` adjudications with reason, confidence, notes, and reviewer identity;
5. write a strict-JSON manifest and result lock, preserving the immutable key separately; and
6. keep final closed.

Sections 18.14–18.15 record the subsequent audit and adjudication. Source/input repair and independent review of proposed corrections remain the prerequisites for a new, separately contracted **4E-B** representation-learning arm. That arm must reuse the non-final partitions, retain per-source recall and false-none gates, preserve proper-score and policy-cost constraints, run seed 17 first, and authorize seeds 42/123 only after a complete non-final pass. If the audit instead finds material ambiguity or label/evidence defects, data adjudication precedes new training. [E14–E20; P23]

## 18.14 Phase 4E-A2: Repaired QASPER error audit reveals asymmetric annotation and serialization defects

Phase 4E-A2 executed the repaired blinded audit protocol across 150 QASPER episodes using full restored paper text (median 24,812 characters, up to 98,130 characters) with verified SHA-256 state hashes. The run validates end-to-end: the hash chain, joins, row counts, and adjudication selection all pass validation without NaN values, duplicate rows, or artifact corruption. Final splits remain strictly closed (`final_opened: false`).

### Verdict
Phase 4E-A2 succeeded technically, but its results direct the next step toward a **benchmark and representation repair** rather than an immediate model retraining sweep. Automatically authorizing Phase 4E-B was explicitly rejected (`phase4e_b_automatically_authorized: false`).

### Main results and agreement

The primary blinded review was conducted by an evaluator without access to gold evidence, model predictions, or locked targets:

| Comparison metric | Primary review vs. Locked target |
|---|---|
| Primary review distribution | 119 answerable, 30 semantic-none, 1 ambiguous |
| Locked target distribution | 81 answerable, 69 semantic-none |
| Decided agreement rate | **99 / 149 = 66.4%** |
| Emitted adjudication rows | **51** |
| Locked answerable agreement | **75 / 81 = 92.6%** |
| Locked semantic-none agreement | **24 / 68 = 35.3%** |

Disagreement is highly asymmetric across the 51 adjudication rows:
- **44 cases**: reviewer judged `answerable`, locked target was `semantic_none`.
- **6 cases**: reviewer judged `semantic_none`, locked target was `answerable`.
- **1 case**: genuinely ambiguous versus locked `semantic_none`.

### Detailed breakdown of challenged semantic-none labels

Among the 44 cases where the primary reviewer challenged the locked `semantic_none` target:
- **38 cases** were judged explicitly answerable from the paper text.
- **6 cases** were judged implicitly answerable from surrounding context.
- **28 cases** already had non-empty gold evidence attached in the source dataset annotations, despite being targeted as missing answers.
- **16 cases** lacked annotated evidence despite having clear answers within the full serialized paper.

This heavy concentration of disagreement confirms that missing-option error rates are substantially driven by incomplete or contradictory benchmark annotations rather than simple model classification failure.

### Serialization and representation failure analysis

All six cases where the locked target was answerable but the reviewer judged `semantic_none` were inspected. They reveal a fundamental state serialization limitation:
1. **Omitted tables**: Numeric results existed only in omitted tables.
2. **Missing table contents**: Table captions were serialized without corresponding table values.
3. **Reference placeholders**: Bibliography placeholders appeared instead of model or method names.
4. **Unidentified languages**: Code-mixed datasets lacked identification of the second language.

Passing raw state-text character-length and string-hash checks does not ensure answer-bearing representation coverage. A representation-aware model cannot learn or retrieve information that was never serialized into text.

### Important caveats
1. **Error-enriched sample**: The 150-row audit intentionally samples model failure cases (70 policy-development errors, 60 calibration-gate errors, 20 controls). The 66.4% agreement rate reflects this error-enriched subset and must not be cited as population-wide QASPER label accuracy.
2. **Blinded control sample**: The 20-row blinded-control subset exhibited 50% agreement, though the sample size is too small for a precise population estimate.
3. **Superseding broken audit**: The prior broken audit’s 81.7% agreement is superseded. Its 52.7% ambiguity rate dropped to 0.67% (1 of 150) once full paper text was provided, demonstrating that missing context distorted the initial audit.
4. **Decision threshold semantics**: In the output records, `model_prediction_label` reflects the raw 0.5 decision, while `operating_outcome` uses the frozen 0.210894 threshold. Seven adjudication rows differ between these decisions. Future pipelines will formally label these as `argmax_prediction_label` and `operating_prediction_label`.

### Recommended next step: defect taxonomy and repair

This was the required next step at the original 22 September cutoff. Section 18.15 records the later A3 adjudication and the source/repair work that remains open.

At that cutoff, Phase 4E-B model training was blocked pending independent adjudication of 51 rows using this six-defect taxonomy:
1. `incorrect_semantic_none`: ground-truth label should be answerable.
2. `missing_incomplete_evidence`: answer is present but annotated gold spans are missing or partial.
3. `table_serialization_loss`: answer was lost due to omitted table cells or figures.
4. `reference_resolution_loss`: answer was lost due to unexpanded bibliography citations.
5. `genuine_underspecification`: question is unanswerable or contradictory even in full paper text.
6. `primary_review_error`: primary reviewer erred in finding an answer.

A3 subsequently preserved the independent adjudication (§18.15). The remaining gate is source/input repair and review of proposed corrections independently of implementation before a new seed-17 representation-learning arm.

### Evidence artifacts
- Adjudication Packet: [`AUDIT_ADJUDICATION_PACKET.csv`](https://drive.google.com/file/d/1eu6evYz9JzRQBSr9eJa5g_XpK6szIKE1/view) (SHA-256 `2d6b3be4dfa532de7c0c320d4e84b67d2be571ff4ddcc7f886b22f437e88edeb`)
- Analysis JSON: [`AUDIT_ANALYSIS.json`](https://drive.google.com/file/d/1-UTO4L8gRy5NrbAracI7o2jGxJoXyGu4/view) (SHA-256 `858d1b7b08d2e3e9969f7de4bdc1780cbaf83a831c379b9529d4acfe2a84b25f`)
- Result Lock: [`AUDIT_RESULT_LOCK.json`](https://drive.google.com/file/d/1n7Xb9Ye_ek9kyHt1t6yRqKi1GMgJXY_c/view) (Experiment SHA-256 `89d0e283a2b743efcb95de3f73189c6fb0157620c5f6252beea149a44d1ecefd`)
- Local Directory: [`22_phase4e_a2_qasper_audit_repair_results/20260921T013558Z/qasper_error_audit_repair_s17/`](../../research/22_phase4e_a2_qasper_audit_repair_results/20260921T013558Z/qasper_error_audit_repair_s17/)

## 18.15 Audit continuation and frozen 27-case disposition

The A2 packet's 51 disagreement rows received an independent adjudication preserved by A3. A CSV float round-trip had changed the string representation of 87 source numeric cells; A3 reconstructed the first 20 columns of all 51 rows exactly from the frozen packet and changed only adjudication columns. Six low-confidence rows then received a **same-assistant** follow-up. These six are not additional independent votes. The corrected 51-row review distribution is 33 answerable, 13 semantic-none, and five ambiguous. These judgments are audit proposals, not QASPER gold changes. The 150-row packet is enriched for errors and cannot estimate population label quality. [E23]

A3 identifies short, exact frozen-state span candidates and source leads for
omitted table values and references. A4 preflight verifies 12 spans across seven
policy-development questions and keeps one calibration-gate span diagnostic.
Three unresolved evidence cases are quarantined. The later saved
[alignment QA](../../research/24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/QASPER_SOURCE_ALIGNMENT_QA.json)
and four-row alignment table record exact dataset-title, abstract, and paragraph
coverage for four policy papers at QASPER revision
`06806e4608976fc2fac0a090ac425d5b2b29caf4`, plus 40 caption candidates. The
preliminary prose saying that source alignment has not run is stale relative
to those outputs. PDF-version equivalence is unproven, recovered table values
remain zero, and general table/reference serialization repair is incomplete.
This checks saved output consistency, not a fresh upstream reproduction.
Source leads are not benchmark gold or permission to paste answers into state.
[E23; E24]

A later, **separate** 27-case policy worklist received user-authorized assistant review and a frozen disposition lock. Its categories are 10 incomplete gold/evidence quarantines, 10 locked-label tie diagnostic quarantines, three question/answer-alignment quarantines, three insufficient-state-evidence quarantines, and one source-grounded representation candidate only. The 77 one-to-one original-annotation ties across 3,598 non-final questions remain a diagnostic stratum; their locked labels do not change. The NarrativeQA lead is a source-grounded representation candidate, not permission to insert the four original-answer scores into state. The LCF comparator mismatch remains quarantined. The lock records no change to labels, gold evidence, states, the A4 ledger, thresholds, or final data, and `phase4e_b_authorized` is false. This assistant-conducted disposition does not replace independent review of proposed benchmark corrections or the separate A3 51-row population. [E25]

The next data gate is versioned source alignment and benchmark repair: verify the pinned source and document versions, recover exact table/caption/reference provenance, resolve or quarantine uncertain label/evidence proposals, and obtain review independent of implementation. The frozen benchmark stays intact until a new contract permits an overlay. [E23–E25]

## 18.16 Exploratory prefill representation probe

The seed-17 4E-B.0 probe asks a narrower question than Phase 3A/3B execution parity: can a frozen state-prefill **summary** support decisions through cheap text probes without candidate-conditioned Qwen continuation? It uses final-token and mean-pooled root descriptors, pooled frozen question/candidate token embeddings, and a read-only Phase 4D comparator. The fixed arms are `root_only`, `root_question`, and `root_question_candidate`; the first two cannot rank candidates and use uniform candidate probabilities. The contract selects thresholds on policy development, fits temperature on calibration fit, and evaluates on the non-final calibration gate. Final is forbidden and recorded unopened. This probe does not test a token-level cross-attention query module, a learned state encoder, or a router backed by full Qwen. [E26]

| Arm | Non-final calibration-gate finding | Decision |
|---|---|---|
| `root_only` | Source-macro semantic-none recall 0; zero accepted requests | Fails both-source recall and useful coverage |
| `root_question` | Source-macro raw semantic-none recall 0; zero accepted requests | Cheap question embedding does not recover a passing policy |
| `root_question_candidate` | ContractNLI false-none 1.0; QASPER semantic-none recall 0; source-macro false-none 0.5; zero accepted requests | Candidate text pooling still fails both-source behavior and proper-score non-regression |

All three `calibration_gate_pass` flags are false. The candidate-aware arm has no eligible operating point and worsens NLL/Brier against the frozen Phase 4D reference on both sources. The reported QASPER accuracy near 0.87 for root-only/root-question is an answerable-majority artifact: balanced accuracy is 0.50 and semantic-none recall is zero. The tested root summary plus pooled text embeddings therefore does not replace candidate-conditioned interaction. The result leaves prefill reuse and full hybrid-state branching intact as execution mechanisms. It does not test richer token-level interaction within this probe. Earlier B1/B2 studies did test learned state-query access, without passing their complete quality/policy gates. The Drive folder contains `NONFINAL_EVALUATION.json` and predictions, but no separate `NONFINAL_RESULT_LOCK.json`; record this as an exploratory non-final failure, not a formally locked release result or authority for another seed, sweep, final evaluation, or distillation study. [E26]

## 18.17 Exploratory option-logit audit

The separate 4E-B.1 audit asks whether one constrained next-token readout can use **full candidate-conditioned Qwen continuation** to score the frozen natural-document questions. It holds the Phase 4A Base checkpoint, FP32 precision, corpus, and question IDs fixed, but changes prompt serialization and readout relative to the historical Phase 4A head. The result is a matched-row method comparison, not a head-only ablation. One state prefill is reused across its questions. Sixteen calibration-gate states per source were selected by state-ID hash before scoring, yielding 272 ContractNLI and 53 QASPER questions. No training, temperature fit, prompt selection, threshold fit, or final evaluation occurred. The comparator is the historical Phase 4A reference, not the Phase 4D parent or a JevK5 checkpoint. [E27]

| Scored task | Option logits | Frozen Phase 4A reference | Constraint |
|---|---:|---:|---|
| ContractNLI, answerable-only two-option accuracy | 130/148, 87.8% | 46/148, 31.1% | The `entailed` majority baseline is 125/148, 84.5%; option-logit balanced accuracy is 73.3% and `contradicted` recall is 12/23. |
| ContractNLI, full decision accuracy | 130/272, 47.8% | 90/272, 33.1% | Explicit `Z` selects semantic none on 0/124 cases; majority-class full accuracy is 125/272, 46.0%. |
| QASPER, one-option `answerable` versus `Z` | 41/53, 77.4% | 23/53, 43.4% | The answerable-majority baseline is 45/53, 84.9%; semantic-none recall is 2/8 versus 6/8 for the reference. No candidate-ranking task exists here. |

Reversing ContractNLI's two candidates changes the candidate winner on 18/272 questions and shifts an option probability by up to 0.2865. Among answerable questions, reversed-order raw accuracy is 89.2% but balanced accuracy falls to 67.0%. The run therefore supports a narrow candidate-ranking signal on this sample while failing as a complete semantic-none decision path. Better full-decision NLL/Brier on these rows does not compensate for zero ContractNLI none recall. The previously exposed calibration-gate sample, small number of source states, and unresolved QASPER annotation/evidence defects prevent confirmation or promotion. No gate threshold was selected from these data. [E27]

The result lock hashes match its contract, 325-row file, and report; independent row recomputation reproduces the stated counts and metrics. Cached and full-prompt logits agree on one question per source, with maximum absolute differences `7.63e-6` and `1.91e-5` and unchanged selected actions. This is a bounded execution check. The Colab run used slower reference PyTorch convolution/DeltaNet fallbacks and did not record a complete host identity, so its timings are not a reproducible throughput claim. The next model gate remains source-aligned benchmark repair and independent review. [E27]

## 18.18 Exploratory candidate ranking and semantic-none sweep

The 4E-B.2 notebook fixes three candidate rankers, three semantic-none detectors, and a five-value threshold grid before scoring. It fits a one-variable detector calibration on `calibration_fit`, selects the component pair and threshold on `policy_development`, and reports a 12-state-per-source sample of `calibration_gate`. The 741 non-final questions span 72 states. The selected gate states do not overlap 4E-B.1's gate sample, but the gate partition was already exposed by prior work, so this is a bounded diagnostic rather than untouched confirmation. The same pinned Qwen3.5-4B-Base FP32 text model and effective state input are used; the prompts and readouts still differ from the trained Phase 4D parent. Final remains unopened. [E28]

| Source and development-selected arm | Sampled gate result | Boundary |
|---|---:|---|
| ContractNLI: order-averaged option logits for ranking, calibrated `Z` for none, threshold `0.35` | Candidate ranking 94/108 answerable (87.0%) versus 84/108 (77.8%) majority-position baseline; full decision 126/204 (61.8%); none recall 72/96 (75.0%); false-none 46/108 (42.6%) | False-none exceeds the Phase 4E ceiling of 0.20. Conditional ranking gets 84/84 entailed and 10/24 contradicted; after rejection, full-decision contradicted recall is only 3/24. |
| QASPER: one candidate, calibrated strongest-candidate support for none, threshold `0.25` | Full decision 41/46 (89.1%), equal to the answerable-majority baseline; none recall 0/5 | All calibrated none probabilities are below `0.25` (maximum `0.2348`), so the declared grid cannot reject any development or gate question under this selected detector. One supplied candidate means there is no ranking comparison. |

Order averaging changes the ContractNLI candidate winner on 13/204 sampled gate questions. Its answerable-only count is one higher than original-order logits (94 versus 93 of 108), while separate support scores 86/108. This is a narrow ranking signal. The selected ContractNLI full-action balanced accuracy is 0.6250, but its answerable rejection burden and contradicted collapse fail the complete decision task. QASPER's selected development arm also finds 0/5 none cases. The gate `Z` detector has AUROC 0.80 with only five positives; that small sample is insufficient to claim robust separation. An unlocked lower-threshold diagnostic selected `Z` at `0.175` on development (3/5 none, 6/38 false none) and, on the already exposed gate, finds 2/5 none with 9/41 false none. This post-hoc check explains the original grid's floor and is not a replacement locked selection. [E28]

The contract, development selection, 741-row Parquet file, and report match the saved result-lock hashes. Independent row checks confirm unique questions, exact selected-state membership, no overlap with 4E-B.1 gate states, finite score fields, and the selected-arm counts. Six cached/full-prompt checks cover three prompt families on one question per source and stay within the declared tolerances. The lock records 72 state-part hashes; those individual parts were not independently downloaded in this review. This sweep satisfies neither the QASPER semantic-none objective nor the ContractNLI false-none guardrail, does not evaluate policy cost or a matched Phase 4D parent on these rows, and promotes no model. Source-aligned benchmark repair and independent review remain the next model gate. [E28]

---

## 18.19 Scientific interpretation after the evidence review

**Historical interpretation as of 24 September 2026.** The text below retains the
pre-M2.1/M2.2 work order. Later bounded authorizations and completed experiments
are recorded in §§18.20–18.21; the current interpretation is §18.22. Historical
future-work language here is not an assertion that those later fits remain unrun.

**Reusable computation and a reusable decision representation are different
results.** Complete hybrid-state branching preserves the tested numerical
function. A root final-token or mean-pooled vector, formed before the question
is known, is a different representation from the last token after full
state/question/candidate interaction. The cheap-prefill failure rejects its
specific shortcut. B1/B2's learned token-level access also failed complete
quality/policy gates, but neither result establishes that all shared-query
architectures fail. [E12–E20; E26]

The severe terminal-token bottleneck and superior bidirectional-encoder claims
in the historical roadmap remain hypotheses. Existing comparisons do not
isolate those causes from missing/truncated evidence, annotation ambiguity,
supervision, or readout training. A custom small encoder is therefore a
conditional cost hypothesis rather than the project's predetermined destination.
[E14–E28; interpretation]

**Source correctness, input coverage, and model use need separate evidence.**
The repaired audit's median full-paper length is 24,812 characters, while the
Phase 4 cache caps state features at 1,024 tokens. A reviewer finding support in
the paper does not establish that the model received it. The visibility ledger
must trace source spans or table cells through serialization and truncation to
finalized model input. Negative and ambiguous cases require explicit grounds,
not fabricated positive spans. Report source-answerability and visible-input
answerability separately. [E14; E22–E24]

The 44 challenged semantic-none labels were audit disagreements, not 44
independently confirmed gold errors. A3's 51-row adjudication, its six
same-assistant follow-ups, and the separate 27-case disposition are different
records. Their error-enriched populations cannot estimate overall label error.
Retain original gold and use independently reviewed, versioned repairs or
prediction-blind quarantine. A text-complete prototype subset is a new scope,
not a retrospective pass of the original two-source gate. [E22–E25]

**Ranking, applicability, and application review also need separate tests.**
In 4E-B.2, ContractNLI candidate ranking reaches 94/108, yet full decisions reach
126/204 and rejection removes 46/108 answerable cases. QASPER's one-candidate
task has no ranking comparison. Its selected detector's maximum gate none
probability, 0.2348, falls below the grid floor of 0.25. That procedure cannot
reject those cases. A future threshold-selection procedure needs reachable
operating points, deterministic ties, class-support checks, and all-answer/
all-none controls fixed before gate evaluation. Post-hoc thresholds remain
diagnostic. [E28]

First evaluate full question/candidate-conditioned Qwen on repaired inputs,
including a fixed direct-logit method. Compare it with the retained Phase 4D
StateQuery parent and majority/prior controls on those same inputs. The pinned
Base-model logit audits changed both prompts and readout.
They are neither clean head-only ablations nor tests of JevK5's trained adapter.
If visible evidence is sufficient and Qwen still fails, the preferred next
modeling intervention is one bounded upstream-adaptation treatment with an
attributable head-only control. Its contract must explain what changes relative
to B0/B1/B2 and the earlier 2B LoRA pilot. This is a hypothesis, not a promised
solution. [E11; E15; E20; E27–E28; proposed experiment]

Reviewed evidence-window diagnostics can distinguish evidence access from
supervision/model-use failures. A window that succeeds where the full input
fails motivates representation/access work. Failure on both motivates revisiting
the task and adaptation. Oracle windows do not measure deployable retrieval.
Review policy is evaluated after the semantic distribution, with per-source
proper scores, recall, false-none, cost, and coverage. Zero acceptance does not
establish low accepted-error. [§13.4; proposed diagnostic]

**Lower cost follows a useful operating point.** The exploratory screen's 4B
95.0% aggregate includes saturated constructed families, while natural MultiRC
is 83.33% across 24 source messages. The 2B state-first LoRA result of 91.56%
makes it a credible first smaller-model challenger, not an accepted replacement.
H's development/final transfer reversal argues for representative held-out
question/rubric families and source-group uncertainty. [E9; E11]

One matched MLX FP32 per-lane/vectorized study can proceed on the frozen
integration profile while evidence repair continues. New learned cost
challengers wait for useful quality. If a cheap reader cannot preserve the
required trade-off, retain causal branching. The 4B numerical oracle has not
earned semantic teacher status, so distillation also waits for demonstrated
teacher quality. [§17.3; E17–E28; proposed sequence]

---

## 18.20 M1 visibility preparation and completed M2.1 frozen comparison

### 18.20.1 Scope and source/input boundary

The completed `OpenKind_Source_Visibility_J0_J1_v0_3_2.ipynb` run closes the bounded frozen **Base versus post-trained checkpoint × state-prefix budget** diagnostic. The session is `20260925T174531_575346Z`, engine `da6e86349af2df1f`, under `OpenKind_M1_M21_Closeout_results/m1_m21_joint_option_v032_s17_fp32`. Its result manifest binds six result files. Execution completion does not close the wider M0/M1 release-review requirements or authorize promotion. [E29]

The common panel contains **741 questions across 72 source states/components**: 244 calibration-fit questions, 247 policy-development questions, and 250 calibration-gate questions. The last group contains **204 ContractNLI questions from 12 contracts and 46 QASPER questions from 12 papers**. These are previously exposed non-final examples. Four comparison cells produce **2,964 unique `(arm, condition, question_id)` decisions**, not 2,964 independent observations. [E29; V4]

J0 is the pinned Base checkpoint and J1 the pinned post-trained checkpoint. Both use the common state-first, joint-option, no-thinking answer-code renderer. The readout scores allowed vocabulary rows at the final answer position; it does not generate a text answer. This is a comparison under a common renderer, not a comparison of each checkpoint's independently optimized native prompt. The run uses FP32, a full forward per request, and state-prefix caps of **1,024 and 4,096 tokens**; the caps do not include all question, option, and formatting overhead. It is not a shared-cache, MLX, or service benchmark. [E29]

The preceding M1 preparation separates stored-source integrity, annotated-string visibility, and semantic sufficiency. For example, its previously exposed full-panel QASPER accounting finds all known annotated strings retained on **12/119** annotated questions at 1,024 tokens and **79/119** at 4,096. The denominator excludes ten questions without usable annotated strings. At the larger cap, 24 annotated questions retain only some strings and 16 retain none. For ContractNLI, all strings are retained on **248/356** annotated questions at 1,024 and **356/356** at 4,096; another 256 questions have no usable annotated strings. These checks concern exact recorded annotations, not independently established answerability. [E29; V4]

A hash match does not adjudicate labels. Retaining all annotated strings does not prove complete evidence sufficiency, and missing an annotated string does not prove semantic none. The separate 27-case disposition and annotation-tie policy retain their original labels and quarantine boundaries. Source-document/PDF equivalence, independent correction review, and reviewed oracle windows are not established by the new tokenization or stored-record checks. The later training pilot deliberately uses a narrower complete-document, source-record-checked ContractNLI scope rather than declaring the whole repaired benchmark independently approved. [E25; E29; E30]

The historical threshold question also has a bounded closeout. Recomputing E28's selected QASPER detector on its **43 policy-development questions** gives a minimum score of **0.0615064535** and maximum **0.2030772089**, below every threshold in the old `0.25/0.35/0.50/0.65/0.75` grid. The grid represents one distinct decision policy; exact breakpoints and endpoints represent 44. This confirms grid non-reachability on those saved development scores. It does not reselect the detector or threshold, prove that an alternative passes transfer, or replace the distinct **0.2348 gate maximum** discussed in §18.19. [E28; E29]

### 18.20.2 Frozen checkpoint and budget results

Identity-calibration metrics on the **previously exposed calibration gate** follow. Lower NLL is better. Equal-source accuracy is the mean of the two source-level accuracies, not the pooled 250-question accuracy. [E29; V4]

| Checkpoint / state-prefix cap | ContractNLI correct / 204 | QASPER correct / 46 | Equal-source accuracy | Equal-source NLL |
|---|---:|---:|---:|---:|
| J0 Base / 1,024 | 122 — 59.80% | 37 — 80.43% | 70.12% | 0.70238 |
| J1 post-trained / 1,024 | 141 — 69.12% | 29 — 63.04% | 66.08% | 0.71484 |
| J0 Base / 4,096 | 117 — 57.35% | 40 — 86.96% | 72.15% | 0.65009 |
| J1 post-trained / 4,096 | 142 — 69.61% | 36 — 78.26% | 73.93% | 0.57138 |

At 4,096 tokens, J1 gains **25** correct ContractNLI decisions but loses **four** QASPER decisions relative to J0. Its saved equal-source accuracy difference is **+1.78 percentage points**, with a descriptive component-bootstrap 95% interval of **−5.88 to +7.42 points**. The raw NLL difference is **−0.07870**, interval **[−0.14440, −0.00420]**. This supports retaining J1 as a research challenger, not a confirmed universal accuracy winner. These intervals are source-reported descriptive diagnostics on an exposed panel. [E29]

### 18.20.3 Evidence visibility and aggregate cancellation

The fixed `improved_to_all` stratum contains the same questions at both budgets: known annotated strings are not all retained at 1,024 but are all retained at 4,096. Membership is not reselected from each model's successes. [E29; V4]

| Fixed question subset | J0: 1,024 → 4,096 correct | J1: 1,024 → 4,096 correct |
|---|---:|---:|
| ContractNLI, 32 questions | 23 → 30 | **17 → 30** |
| QASPER, 25 questions | 19 → 22 | **14 → 21** |

For J1 ContractNLI, the 76 questions with all annotations visible at both budgets remain at **59 correct**. Correct semantic-none decisions fall from **65/96 to 53/96**. Consequently, **+13** correct decisions on newly fully visible questions and **−12** on none-labeled questions produce only **+1** correct decision overall. Aggregate accuracy nearly conceals the evidence-budget gain. The `unestablished` ContractNLI visibility stratum here is the 96 none-labeled questions, not 96 newly demonstrated corrupt documents. [E29; V4]

The larger prefix also adds other text, so the comparison does not isolate the causal contribution of an individual evidence span. It nevertheless demonstrates why evidence access and judgments of non-support require separate analysis. At 4,096 tokens, J1's conditional real-option ranking is **99/108**, while only **89/108** real-option targets receive a correct full semantic decision; eleven are assigned none. J1's ContractNLI class counts are 81/84 entailed, 8/24 contradicted, and 53/96 none. Those errors motivated the later targeted contradiction/unsupported training pilot. [E29; interpretation]

The frozen J1/4,096 identity policy accepts **61/204 ContractNLI decisions, seven wrong**, with row-mean cost **0.10441**, above review-all at 0.1. On QASPER it accepts **11/46, zero wrong observed**, cost **0.07609**. Thus even the leading frozen quality challenger does not establish useful automation on both sources. Neither larger inputs nor post-training alone closes the complete decision contract. [E29]

## 18.21 M2.2 matched decision-LoRA pilot: in-domain gains, failed retention

### 18.21.1 Completed run and attributable comparison

The `OpenKind_M22_Matched_Decision_LoRA_v0_4_1.ipynb` experiment completed as **`COMPLETED_MODEL_RESULTS`**. Its run is `m22_v041_s17_bf16_6b295d46b7436a52`, session `20260925T232651_880064Z`; the saved result artifacts are dated **26 September 2026, approximately 01:21 UTC**. This result answers a new, narrower hypothesis:

> Can whole-document ContractNLI decision training improve contradiction and unsupported decisions while retaining QASPER behavior?

The intervention and controls are explicit. [E30]

| Arm | Starting checkpoint | Treatment |
|---|---|---|
| J0 | `Qwen/Qwen3.5-4B-Base`, revision `1001bb4d826a52d1f399e183466143f4da7b741b` | Frozen joint-option control |
| J1 | `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Frozen joint-option control |
| J2 | Same J0 Base identity | Decision-specific LoRA |
| J3 | Same J1 post-trained identity | Matched decision-specific LoRA |

All four arms are scored under the **same BF16 profile in this run**. The earlier FP32 values in §18.20 are not substituted as adaptation controls. The recorded hardware is an NVIDIA A100-SXM4-80GB with PyTorch `2.11.0+cu128` and Transformers `5.17.0`. Full forwards use no reusable state cache. SDPA math is pinned for original forward, activation-checkpoint recomputation, and backward; non-reentrant checkpointing retains RNG and metadata checks. The v0.4.0 backend-context failure and v0.4.1 repair are execution history, not failed semantic fits. This successful run does not qualify BF16 as interchangeable with the older strict-FP32 Rust/MLX reference. [E30]

The common renderer applies J1's no-thinking template to both checkpoints, followed by an `Answer:` prefill. Training uses cross-entropy over the offered semantic outcomes, with gradients through the complete request and selected vocabulary-row projection. The **rank-16, alpha-32** adapters cover full-attention and Gated DeltaNet linear projections. Embeddings, vocabulary weights, MLPs, and normalization weights remain frozen. There is no separate shallow-head treatment in this pilot; it is not a clean LoRA-versus-head-only experiment or a controlled comparison with historical StateQuery B0/B1/B2. [E30]

### 18.21.2 Training population, schedule, and selection

The admitted training pool contains **128 complete ContractNLI document components and 2,176 questions**: 1,041 entailed, 274 contradicted, and 861 semantic-none targets. Development contains **12 components and 204 questions**: 112 entailed, 19 contradicted, and 73 none. Original source-record checks cover text, hypothesis, label, and annotated evidence; over-cap documents are excluded rather than truncated or relabeled. Source-record equivalence is not independent semantic adjudication. Recorded train, development, and evaluation components are disjoint. [E30; V4]

The matched schedule contains **960 exposures per fit**, eight unpadded microbatches per optimizer update, for **120 attempted updates**. It combines **720 class-balanced exposures and 240 ordinary ContractNLI replay exposures** and visits 734 unique questions. This is in-domain replay, not broad instruction replay. Learning rate is `2e-5`, warmup 12 updates, gradient clipping 1.0, seed 17. Training randomizes answer-code assignments and option layout. The state cap is 4,096 and the total-input cap 8,192 tokens. [E30]

QASPER enters neither LoRA supervision nor checkpoint selection. It remains a source/task-held-out retention diagnostic; its separately assigned calibration-fit and policy-development roles are retained for later probability/policy analysis. Raw identity-calibration results are the primary retention comparison. The fixed ContractNLI hypotheses across documents mean that even successful document transfer would not establish unseen-rubric competence. [E30]

Checkpoint selection minimizes **ContractNLI full-document development group-mean NLL**, with earlier updates preferred on ties and frozen step zero eligible. Both fits select **update 80**, corresponding to **640 exposures, 538 unique questions, and 125 visited training groups**, although both runs continue through update 120. These counts distinguish eligible pool size, visited groups, unique questions, and repeated exposures; none is an epoch count. [E30; V4]

| Update | J2 development group-mean NLL | J3 development group-mean NLL |
|---|---:|---:|
| 0 | 0.82290 | 0.71223 |
| 40 | 0.59599 | 0.58424 |
| **80 — selected** | **0.53375** | **0.52622** |
| 120 | 0.57592 | 0.62944 |

Selection follows the declared rule and does not use gate or QASPER labels. However, at update 80, development entailment recall is already **9.82 points lower for J2** and **7.14 points lower for J3** than at step zero. The NLL-only rule did not protect every class. Update 120 has worse NLL for both fits, although some class recalls recover. No QASPER comparison of updates 40/120 is established by this report; do not retroactively select another checkpoint using exposed retention outcomes. [E30; V4]

### 18.21.3 Matched raw gate results and class trade-offs

The evaluation executes **3,964 model decisions**: four arms each score 741 primary questions plus 250 code/order variants. The primary result table has **2,964 unique `(arm, question_id)` rows**, over the same 72-component non-final panel described above. The extra variants do not add independent examples. The optional 1,024-token gate stress evaluation is not run. [E30; V4]

The following identity-calibration gate values use the common **4,096-token state-prefix condition**. Unlike admitted training documents, evaluation inputs can remain truncated diagnostic prefixes with locked original labels. NLL and Brier are saved evaluator values; Brier is the sum over all declared semantic outcomes, not divided by the number of classes. [E30]

| Arm | Source | Correct / n | Accuracy | Balanced accuracy | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| J0 | ContractNLI | 116/204 | 56.86% | 54.71% | 0.97272 | 0.56700 |
| J1 | ContractNLI | 141/204 | 69.12% | 61.31% | 0.74811 | 0.42123 |
| J2 | ContractNLI | 166/204 | **81.37%** | 77.93% | 0.55262 | 0.29263 |
| J3 | ContractNLI | 168/204 | **82.35%** | 81.60% | 0.55134 | 0.29436 |
| J0 | QASPER | 40/46 | 86.96% | 66.34% | 0.33376 | 0.20364 |
| J1 | QASPER | 37/46 | 80.43% | 71.46% | 0.39921 | 0.26105 |
| J2 | QASPER | 35/46 | **76.09%** | 86.59% | 0.66467 | 0.39704 |
| J3 | QASPER | 34/46 | **73.91%** | 67.80% | 0.57011 | 0.34273 |

J2 adds **50** correct ContractNLI decisions relative to J0, **+24.51 percentage points**: 59 repaired errors minus nine introduced errors. J3 adds **27**, **+13.24 points**: 40 repaired minus 13 introduced. On QASPER, J2 repairs three but introduces eight errors; J3 repairs none and introduces three. J3 is only two ContractNLI decisions better and one QASPER decision worse than J2; these results do not establish a universal adapted-checkpoint winner. [E30; V4]

| ContractNLI gold class | J0 correct / total | J1 correct / total | J2 correct / total | J3 correct / total |
|---|---:|---:|---:|---:|
| Entailed | 81/84 | 81/84 | 73/84 | 70/84 |
| Contradicted | 10/24 | 8/24 | 16/24 | 19/24 |
| Semantic none | 25/96 | 52/96 | 77/96 | 79/96 |

For J3 versus J1, **+27 none +11 contradiction −11 entailment = +27 correct decisions**. Contradiction recall improves from **33.33% to 79.17%**, while entailment recall falls from **96.43% to 83.33%**. J2's entailment loss is **9.52 points**. Both exceed the declared **five-point other-class non-regression allowance**. J3 reduces contradiction-to-entailment errors from seven to one, so the result is not adequately described as only predicting more none. Nevertheless, this single recipe does not isolate supervision, changed class exposure, representation adaptation, and code/option learning as causal mechanisms. [E30; V4; interpretation]

Across all 108 real-option ContractNLI targets, J1 and J3 each make **89 correct full decisions**, and each has **100/108 correct conditional real-option rankings** in this BF16 run. Identical aggregate conditional ranking can therefore coexist with a substantial redistribution between entailment and contradiction. These counts must not be substituted for E29's FP32 conditional-ranking count of 99/108. [E29; E30]

### 18.21.4 Retention and evidence-visibility regression

| QASPER gate quantity | J0 | J1 | J2 | J3 |
|---|---:|---:|---:|---:|
| Correct none decisions / 5 | 2 | 3 | 5 | 3 |
| False none on answerable targets / 41 | 3 | 7 | 11 | 10 |
| False-none rate | 7.32% | 17.07% | **26.83%** | **24.39%** |

Both adapters breach the **20% false-none cap**. Relative to their own frozen controls, QASPER accuracy falls **10.87 points for J2** and **6.52 points for J3**, exceeding the declared **three-point retention allowance**. J2's 100% none recall is based on five targets and does not override the eleven answerable cases assigned none. These are decisions against unchanged benchmark labels, not new adjudications of ambiguous or evidence-incomplete examples. [E30]

The same direction appears on policy-development: J2/J0 QASPER correct counts are **25/43 versus 34/43**; J3/J1 counts are **30/43 versus 36/43**. The adapted false-none rates there are **39.47% and 31.58%**. Positive scalar temperature leaves semantic argmax unchanged. It improves some proper scores but does not eliminate QASPER non-regression failures: calibrated gate NLL is **0.58925 versus 0.34273** for J2/J0 and **0.50853 versus 0.40647** for J3/J1. QASPER differs in domain and decision structure; this is measured negative transfer on one held-out task, not proof of broad catastrophic forgetting or a unique forgetting mechanism. [E30; interpretation]

The previously recovered evidence-sensitive subset also needs preservation. On the **same 32 ContractNLI questions** in E29's `improved_to_all` stratum, scored here at 4,096 tokens:

| Arm | Correct / 32 | False-none decisions / 32 |
|---|---:|---:|
| J0 | 30 | 1 |
| J1 | 30 | 0 |
| J2 | 25 | 5 |
| J3 | 25 | 5 |

J3 improves the 76-question `both_all` stratum from J1's **59 to 64** correct, yet loses five correct decisions on the 32-question stratum whose annotations require the larger prefix. This is a fixed-stratum analysis of one budget, not a new budget intervention. Annotated coverage remains distinct from semantic sufficiency. The finding is that aggregate in-domain improvement does not guarantee preservation of previously useful evidence-sensitive behavior. [E29; E30; V4]

### 18.21.5 Automation utility, weighting, and exact break-even

Identity policies are selected on policy-development and transferred unchanged. Correct automation costs zero, a wrong accepted decision costs one, and review costs 0.1. The semantic distribution remains separate from the review action. [E30]

| Arm / source | Accepted / n | Wrong accepted | Accepted error | Row-mean cost |
|---|---:|---:|---:|---:|
| J0 / ContractNLI | 34/204 | 5 | 14.71% | 0.10784 |
| J1 / ContractNLI | 59/204 | 7 | 11.86% | 0.10539 |
| J2 / ContractNLI | 46/204 | 2 | 4.35% | **0.08725** |
| J3 / ContractNLI | 38/204 | 0 | 0% observed | **0.08137** |
| J0 / QASPER | 14/46 | 0 | 0% observed | 0.06957 |
| J1 / QASPER | 11/46 | 0 | 0% observed | 0.07609 |
| J2 / QASPER | 10/46 | 1 | 10.00% | **0.10000 — exact break-even** |
| J3 / QASPER | 4/46 | 1 | 25.00% | **0.11304** |

J3's ContractNLI operating point is an observed in-domain benefit, with **18.63% coverage**. Zero wrong answers among 38 accepted, clustered cases does not establish zero future error. Its QASPER policy is worse than review-all and has only four accepted decisions. Global-temperature calibration does not rescue the full research screen. [E30]

**Reporting correction, not a changed experiment.** E30 stores J2's QASPER row cost as `0.09999999999999995`. The evaluator's strict floating-point `< 0.1` test therefore passes one `policy_cost_below_review_all` subcheck. Exact count arithmetic gives:

```text
n = 46; accepted = 10; wrong = 1; reviewed = 36
cost = (1 + 0.1 × 36) / 46 = 0.1
```

The correct interpretation is **break-even, not strictly below review-all**. For these costs, strict benefit is equivalently `wrong < 0.1 × accepted`, evaluated without floating-point tie error. V4 includes a proposed exact-fraction correction and focused tests. The original evaluator, saved flag, thresholds, selection, and result lock remain unchanged. Both adapted arms already fail the complete screen; correcting this subcheck adds a J2 failure but reverses no overall verdict. [E30; V4]

**Weighting is a separate issue.** V4 recomputes the following gate summaries for the same frozen identity policies:

| Arm | Pooled-question row cost | Equal-source mean row cost | Equal-source, equal-component cost |
|---|---:|---:|---:|
| J0 | 0.10080 | 0.08870 | 0.09055 |
| J1 | 0.10000 | 0.09074 | 0.09280 |
| J2 | 0.08960 | 0.09363 | **0.12193** |
| J3 | 0.08720 | 0.09721 | **0.12610** |

Pooled cost weights every question equally. The middle column averages source-level question means. The last column first averages question costs within each source component, then averages components within each source, then weights the two sources equally. This last weighting matches the policy-search objective. Adaptation looks beneficial in the row-pooled aggregate while both adapted component-weighted aggregates are worse than review-all. This is not another arithmetic defect: the summaries describe different populations. Preserve the declared per-source row-cost screen and report the selection-weight-aligned aggregate alongside it; do not substitute a new pass rule retrospectively. [E30; V4]

### 18.21.6 Code/order sensitivity and uncertainty

The 250-question gate transformation jointly changes code assignment and reverses option order. Semantic agreement is measured after remapping. It is not a pure order ablation, no inference ensemble is evaluated, and agreement alone is not correctness. [E30; V4]

| Arm | ContractNLI agreement / 204 | QASPER agreement / 46 |
|---|---:|---:|
| J0 | 174 — 85.29% | 42 — 91.30% |
| J1 | 196 — 96.08% | 42 — 91.30% |
| J2 | 188 — 92.16% | 36 — 78.26% |
| J3 | 193 — 94.61% | 41 — 89.13% |

Randomized training does not establish invariance. J2 improves in-domain agreement but damages the held-out task; J3 remains below its own frozen control on both source agreement counts. Separate code-only and order-only tests would be new diagnostics, not explanations already supplied by this run. [E30; interpretation]

V4 additionally computes paired gate accuracy differences using **10,000 document-cluster bootstrap draws, seed 17**, resampling the 12 documents per source. These are review-derived descriptive intervals, not the notebook's original source-macro intervals; calibration and selection are not refitted and there is no multiple-comparison adjustment. [V4]

| Contrast | Source | Accuracy difference | Descriptive 95% interval |
|---|---|---:|---:|
| J2 − J0 | ContractNLI | +24.51 pp | +15.20 to +33.82 pp |
| J3 − J1 | ContractNLI | +13.24 pp | +7.84 to +18.63 pp |
| J2 − J0 | QASPER | −10.87 pp | −27.50 to 0.00 pp |
| J3 − J1 | QASPER | −6.52 pp | −14.71 to 0.00 pp |

There is one training seed, a small repeatedly exposed gate, only five QASPER none targets, and no independent general-purpose retention panel. Failing a declared observed non-regression screen and proving a population-level decline with statistical certainty are different claims. The substantial in-domain effects support this recipe's task-learning benefit within scope, but neither the small J3-versus-J2 difference nor the retention point estimates establish a universal model ranking. [E30; V4]

### 18.21.7 Verdict, artifact status, and unrun work

**Execution completed; the full improve-and-retain hypothesis did not pass.** Both arms fail QASPER false-none and proper-score non-regression checks on policy-development and gate, the gate entailment-preservation allowance, and the gate QASPER raw-accuracy retention allowance. J3 also fails gate QASPER policy cost; J2 is exactly break-even there and should not pass a strict-benefit subcheck. The contradiction-gain criterion does pass for both. Do not reduce this to either “LoRA failed to learn” or “the model is now ready.” [E30; V4]

Retain J2/J3 at their locked **`step_000080`** identities as research specialists and J0/J1 as frozen controls. Their native OpenKind linear-LoRA safetensors and resumable optimizer snapshots are not a PEFT export, merged deployment model, quantized model, or MLX-qualified artifact. No protected-final opening, general retention certification, teacher calls, seed expansion, automatic sweep, or model promotion is recorded. Training completion does not qualify cache reuse for changed weights or transfer the old native-service evidence to the adapters. [E30]

V4 checks nine result-file hashes, seven runtime-source hashes, canonical data-lock identity, 2,964 primary decision rows, 48 count-based metric records, 120 class records, 24 identity-policy records, eight visibility aggregate groups, 20 tie-stratum count/accuracy rows, and 1,000 code/order records. It reconciles recorded component separation, the shared exposure schedule, both continuous 120-update logs, and minimum-development-NLL selection. It does not inspect adapter/optimizer tensor binaries, rerun the backbone or gradients, re-adjudicate source documents, reconstruct NLL/Brier/calibration from raw logits, or independently recover every visibility-stratum membership. These limits apply equally to this documentation update. [V4]

## 18.22 Current synthesis after M2.1 and the M2.2 pilot

**Historical v0.8.3 synthesis, retained unchanged below.** The proposed
parent-consistency replay and guarded selection were subsequently executed in
E31 (§18.23). E32 (§18.25) now completes the subsequent source-label comparison;
§18.26 provides the current synthesis. The future-work wording below remains
historical, not the current status.

The completed sequence now separates three bottlenecks that a headline accuracy could conflate:

| Question | Measured answer | Remaining boundary |
|---|---|---|
| Does giving the frozen model more recorded evidence help? | Yes on the fixed newly fully visible subsets; aggregate none-class changes can nearly cancel the gain. [E29] | A larger prefix is not a gold-evidence-only intervention or proof of complete source sufficiency. |
| Can the joint-option 4B model learn the targeted natural-document distinctions? | This matched LoRA recipe substantially improves ContractNLI contradiction and none decisions. [E30] | Class exposure, adaptation, and code learning are not isolated, and the hypotheses are fixed across documents. |
| Does that improvement preserve the capabilities needed for a broader model? | Not in the tested pilot: entailment, QASPER, and parts of evidence-sensitive/code-order behavior regress. [E30] | One held-out task is not a general forgetting assessment; no universal negative result follows. |
| Do better aggregate probabilities guarantee useful automation? | No. In-domain policy cost improves while held-out and component-weighted costs worsen. [E30; V4] | Metric weights, semantic outcomes, and the review action must remain explicit. |
| Does a successful CUDA fit qualify the shared-state service? | No. The training/evaluation uses full forwards and a separately versioned BF16 profile. [E30] | Adapter-specific cache, native/MLX parity, load, and service qualification are unrun. |

The current research conclusion is therefore **in-domain adaptation benefit with failed entailment/cross-task retention**, not a complete M2.2 exit or a general architectural victory. The smaller/faster-model question remains downstream of a useful, scoped quality operating point. The immutable CPU/MLX reference remains a maintained systems asset; the new adapters do not inherit its qualifications. [E29; E30; §17.3]

**Proposed next hypothesis, not an executed intervention:** test whether a bounded retention-aware treatment can retain the observed contradiction/none gains without sacrificing supported entailment and a separate task. The preceding results review proposes fixed **train-only out-of-domain replay with a frozen-parent consistency objective** as one candidate treatment. The parent supplies a preservation target, not automatically correct semantic ground truth. No such replay/consistency treatment was executed in E30. Its supervision, resource budget, and attribution must be frozen in a new contract. [V4; proposal]

A future selection rule should predeclare per-class preservation and a separate retention-development criterion while allowing the frozen checkpoint to win. This is a new rule, not a retroactive change to E30's NLL-only selection. Keep QASPER gate/audit errors out of training and checkpoint selection. Using QASPER training data in a later mixture would change its source-held-out role and require a separately defined retention task; current held-out results cannot silently become proof of retention after tuning on them. Independent correction review and genuinely held-out source/question/rubric families remain necessary for broader claims. [E25; E30; V4; proposed safeguards]

Do not automatically continue the same fit, increase rank, expand seeds, replace the backbone, add a teacher, introduce an evidence reader, and alter calibration together. The evidence supports narrowing the next causal question, not launching a tournament. Correct the exact-break-even reporting test under a new evaluator identity and expose row-, source-, and component-weighted policy costs without changing the original result lock. The completed pilot, earlier negative studies, audit dispositions, and protected-final reservation all remain intact. [E14–E30; V4; proposed sequence]

---

## 18.23 Retention-aware replay pilot: partial recovery, no eligible adaptation

**Completed evidence, not a proposed follow-on.** E31 records the v0.5.0 run
`retention_v050_s17_bf16_a69ba45a76e74b63`, session
`20260926T023951_020889Z`. Both new fits complete 120 optimizer updates. Neither
has an eligible nonzero development checkpoint, so **J4_selected = J0** and
**J5_selected = J1**. The fixed-update-80 adapters remain visible as diagnostics;
they are not the selected model. The correct closeout is **partial probability-score
recovery at a matched update count; joint entailment/cross-task preservation not
achieved; frozen parents retained by the declared selector**. No protected final
or deployment promotion follows. [E31; V5]

### 18.23.1 Treatment, populations, and attribution

J4 starts fresh from the pinned Base J0 and J5 from the pinned post-trained J1.
Neither initializes from J2/J3 nor restores a historical optimizer. Rank 16,
alpha 32, learning rate 2e-5, 12 warmup updates, 120 attempted updates, clipping
1.0, BF16 backbone computation, FP32 adapters/loss, and the working math-SDPA
activation-checkpoint policy remain matched to E30. The actual run records an
A100-SXM4-40GB. This is a model-quality experiment using full forwards, not a
shared-cache, MLX, or service-speed qualification. [E30; E31, experiment contract]

The original eight ContractNLI inputs per update are unchanged. The added
objective is:

\[
L_t = \operatorname{mean}_{x\in C_t}\mathrm{CE}(y_x,p_\theta(x))
+ 1.0\operatorname{mean}_{x\in R_t}\mathrm{KL}(p_{\mathrm{parent}}(x)\Vert p_\theta(x)),
\qquad |C_t|=8,\ |R_t|=2.
\]

The consistency temperature is 1.0, separate from downstream calibration.
Frozen-parent distributions over offered outcomes are numerical preservation
targets, not generated rationales or calibrated ground truth. SNLI source labels
stratify the sample and score development/diagnostic predictions; they do **not**
enter this replay loss. Each complete fit has 960 ContractNLI exposures and 240
SNLI replay exposures. At fixed80 the new treatment has 640 ContractNLI and 160
SNLI exposures; the old specialist has the same 640 ContractNLI exposures without
SNLI. Thus the fixed-dose contrast tests replay and consistency **together**, with
extra computation, rather than isolating replay from its objective or matching
compute budgets. [E31, contract and data lock]

Only the pinned SNLI upstream train shard is used. Prediction-blind selection
constructs 240 replay, 192 retention-development, and 192 retention-diagnostic
examples, balanced across the three labels, with one representative per normalized
exact-premise group. These 624 groups are disjoint across roles. This is not an
image-family, paraphrase, or pretraining-contamination guarantee; the export does
not supply the required image-family qualification. Neutral maps to the declared
NLI neither-supported-nor-contradicted outcome, not a universal abstention label.
QASPER remains outside gradients and checkpoint selection. The new SNLI panel is
short-premise NLI, not broad instruction-following or long-document QA retention.
[E31, retained source/data contract]

The primary benchmark remains 741 questions across 72 source components. Its
calibration gate has 204 ContractNLI questions from 12 contracts and 46 QASPER
questions from 12 papers; it was exposed before this run. There are eight named
reporting views but only six numerical checkpoint identities because the two
selected views alias frozen parents. The 5,928 primary rows, 2,000 code/order
variant summaries, and eight 192-example SNLI diagnostic views represent 9,464
question views, **not 9,464 independent examples or necessarily that many distinct
forward computations**. SNLI was a held-out diagnostic within E31; after this
readout it is exposed regression evidence for subsequent work. [E31; V5]

### 18.23.2 Matched fixed80 quality and the selected-fallback distinction

The following results use identity calibration and the common BF16 4,096-token
**state-prefix** condition. NLL is averaged within each source; lower is better.
Historical FP32 E29 values are not substituted for these matched controls. [E31]

| Numerical checkpoint | ContractNLI correct / accuracy | ContractNLI NLL | QASPER correct / accuracy | QASPER NLL |
|---|---|---:|---|---:|
| J0 frozen Base | 116/204 (56.86%) | 0.97272 | 40/46 (86.96%) | 0.33376 |
| J1 frozen post-trained | 141/204 (69.12%) | 0.74811 | 37/46 (80.43%) | 0.39921 |
| J2 locked80, contract-only | 166/204 (81.37%) | 0.55262 | 35/46 (76.09%) | 0.66467 |
| J3 locked80, contract-only | 168/204 (82.35%) | 0.55134 | 34/46 (73.91%) | 0.57011 |
| J4 fixed80, parent-KL replay | 164/204 (80.39%) | 0.51614 | 36/46 (78.26%) | 0.48680 |
| J5 fixed80, parent-KL replay | 166/204 (81.37%) | 0.52210 | 35/46 (76.09%) | 0.49796 |

Both new fixed80 models improve NLL and Brier relative to their corresponding
contract-only specialists on both benchmark sources. Both lose two correct
ContractNLI decisions and gain one correct QASPER decision. The saved-row
reconstruction finds that J4 repairs two and breaks four ContractNLI decisions;
J5 repairs none and breaks two. On QASPER, J4 repairs two and breaks one, whereas
J5 repairs one and breaks none. These changes increase equal-source mean accuracy
by about **0.597 percentage points**, while reducing pooled accuracy by one out
of 250 decisions. The weighting must accompany either claim. [E31; V5]

The source-reported paired, source-stratified component-bootstrap analysis uses
1,000 draws. J4 minus J2 has an equal-source NLL difference of approximately
−0.10717, with a descriptive 95% interval [−0.18708, −0.06254]; J5 minus J3 is
−0.05070 [−0.10636, −0.02300]. The corresponding accuracy intervals include zero:
[−1.471, +3.922] and [−0.980, +3.413] percentage points. These intervals were not
reconstructed from raw logits in this documentation revision. They describe this
one-seed, already-exposed panel, not independent confirmation or population-wide
noninferiority. The stronger positive finding is probability-score recovery
relative to specialists, not an established accuracy advantage. [E31; V5]

The selected views instead reproduce J0/J1 exactly. That fallback satisfies the
selection rule but does not prove successful learned retention or imply that the
frozen models pass all semantic/policy screens. Fixed-dose and selected-pipeline
results answer different questions and must not be pooled. [E31]

### 18.23.3 Targeted gains persist; supported entailment remains the trade-off

| Numerical checkpoint | Entailed correct / 84 | Contradicted correct / 24 | Semantic-none correct / 96 |
|---|---:|---:|---:|
| J0 | 81 | 10 | 25 |
| J1 | 81 | 8 | 52 |
| J2 locked80 | 73 | 16 | 77 |
| J3 locked80 | 70 | 19 | 79 |
| J4 fixed80 | 74 | 15 | 75 |
| J5 fixed80 | 69 | 19 | 78 |

These are complete semantic decisions, not real-option rankings conditional on
removing none. J4 retains most of J2's contradiction/none gains; J5 retains all 19
correct contradiction decisions and 78 of J3's 79 correct none decisions. Yet
entailment recall is 8.33 percentage points below J0 for J4 and 14.29 points below
J1 for J5, exceeding the five-point preservation allowance. J5 is also one
entailed case worse than J3. Out-of-domain parent consistency therefore did not
resolve the supported-entailment loss. [E31, class results]

For the historical 32-question ContractNLI cohort whose annotated evidence became
fully visible only with the larger prefix, both frozen controls score 30/32 at
the current larger budget. J2/J3 score 25/32, J4 26/32, and J5 25/32. This is a
stratification of the new 4,096-token results, not a new 1,024-versus-4,096 run.
Annotation visibility remains distinct from semantic sufficiency. [E31, visibility
strata; E29]

### 18.23.4 Why both preservation selectors chose zero

The prospective selector evaluates updates 0, 40, 80, and 120 on unchanged
ContractNLI development data and a separate SNLI retention-development panel.
Every ContractNLI and SNLI class must remain within five percentage points of
its frozen recall; SNLI accuracy must remain within three points and NLL/Brier
within 0.01, while ContractNLI group-mean NLL must improve by more than 0.01.
Minimum ContractNLI NLL among eligible nonzero candidates wins, with earlier
updates resolving ties; otherwise zero wins. No QASPER or retention-diagnostic
outcome is eligible for selection. [E31, selections and contract]

Both frozen ContractNLI development baselines correctly entail 106/112 cases.
At the five-point allowance, at least 101 correct entails are required. [E31; V5]

| Fit | Update | ContractNLI development NLL | Correct entailments / 112 | Failed selection checks |
|---|---:|---:|---:|---|
| J4 | 40 | 0.60769 | 90 | ContractNLI entailment |
| J4 | 80 | 0.52247 | 96 | ContractNLI entailment |
| J4 | 120 | 0.54869 | 97 | ContractNLI entailment |
| J5 | 40 | 0.55499 | 90 | ContractNLI entailment |
| J5 | 80 | 0.50181 | 99 | ContractNLI entailment; SNLI NLL/Brier |
| J5 | 120 | 0.53541 | 102 | SNLI NLL/Brier |

J4's SNLI-development checks pass at all evaluated nonzero updates. J5 at 120
preserves the ContractNLI classes but has SNLI NLL about 0.76521 versus 0.73011
frozen and Brier about 0.43258 versus 0.41594, beyond the 0.01 allowances. Thus
no evaluated checkpoint satisfies all conditions simultaneously. This is not a
failed training loop or grounds to relax the criteria retrospectively. It does
not establish what untested intermediate checkpoints or different objectives
would do. [E31; V5]

### 18.23.5 Short-premise retention exposes task-dependent transfer

The 192-example SNLI diagnostic is balanced at 64 per class and was not used for
fitting or selection in E31. Results are uncalibrated. [E31]

| Numerical checkpoint | Correct / 192 | Accuracy | NLL | Brier |
|---|---:|---:|---:|---:|
| J0 | 117 | 60.94% | 0.92107 | 0.54200 |
| J1 | 132 | 68.75% | 0.72758 | 0.43185 |
| J2 locked80 | 136 | 70.83% | 0.81815 | 0.45812 |
| J3 locked80 | 141 | 73.44% | 0.69906 | 0.40497 |
| J4 fixed80 | 125 | 65.10% | 0.89585 | 0.52519 |
| J5 fixed80 | 131 | 68.23% | 0.80251 | 0.47444 |

The contract-only specialists perform better here than both their frozen parents
and their replay-trained counterparts. J4/J5 have 11/10 fewer correct predictions
than J2/J3, respectively, with worse NLL and Brier. This directly limits a blanket
interpretation that ContractNLI adaptation destroyed all out-of-domain capability:
it damages QASPER while improving this tested SNLI population. Domain, question
semantics, input length, and decision structure were not isolated. [E31]

**Interpretation rather than demonstrated mechanism:** parent KL penalizes a
departure from the parent's distribution whether that departure helps or harms
source-label correctness. These results are consistent with anchoring limiting
some useful changes. They do not demonstrate that copied parent errors caused
all losses or that source-label replay will succeed. J4 meets the SNLI diagnostic
preservation bounds versus J0 while still failing entailment and QASPER retention;
J5 retains near-frozen SNLI accuracy but fails its probability-score bounds.
Behavioral agreement, accuracy, proper scores, and cross-task utility are separate
requirements. Short SNLI is not a sufficient sole proxy for broad retention.
[E31; V5; inference]

### 18.23.6 QASPER and policy limits remain binding

Both new fixed80 models assign none to 9/41 answerable QASPER questions, or
21.95%, above the 20% cap. J4 detects 4/5 none cases and scores 36/46 overall,
four correct below J0; J5 detects 3/5 and scores 35/46, two below J1. These
8.70- and 4.35-point losses exceed the three-point accuracy allowance. Probability
scores still regress versus the matching frozen parent, including on policy
development. These statements preserve locked benchmark labels and the existing
annotation/visibility qualifications; they are not new effective-input adjudications.
[E31]

At the identity-policy thresholds selected on policy-development, J4 accepts
38 ContractNLI decisions with one wrong and 12 QASPER decisions with one wrong.
Its source row costs are 0.08627 and 0.09565. J5 accepts 59 ContractNLI decisions
with one wrong and seven QASPER decisions with one wrong, for costs 0.07598 and
0.10652. Review-all costs 0.1. Those acceptance counts are small and clustered;
one observed low-error operating point is not a deployment guarantee. [E31]

| Numerical checkpoint | Pooled-row gate cost | Equal-source/equal-component gate cost |
|---|---:|---:|
| J0 | 0.10080 | 0.09055 |
| J1 | 0.10000 | 0.09280 |
| J2 locked80 | 0.08960 | 0.12193 |
| J3 locked80 | 0.08720 | 0.12610 |
| J4 fixed80 | 0.08800 | 0.11970 |
| J5 fixed80 | 0.08160 | 0.12046 |

The weighting-aligned aggregate improves somewhat versus specialists but remains
above review-all for both replay models. J4's two source row costs both beat
review-all while its component-weighted aggregate does not. Reporting the weights
is therefore part of the result, not optional presentation. E31 introduces the
previously proposed exact-rational policy comparison: break-even is not a strict
pass. It correctly records J2's QASPER cost and J1's pooled gate cost as exactly
0.1 without changing E30's evaluator or lock. [E31; V5]

The combined code-reassignment/option-reversal diagnostic improves QASPER
semantic agreement from 36/46 for J2 to 38/46 for J4 and from 41/46 for J3 to
43/46 for J5. ContractNLI agreement is 188/204 versus 188/204 and 193/204 versus
194/204, respectively. Agreement is not correctness; the joint perturbation does
not isolate pure order invariance. No inference ensemble is introduced. [E31; V5]

### 18.23.7 Verification and scope of the documentation update

**Historical v0.8.4 documentation-check scope, preserved below.** The v0.8.5
source-label result and its separate saved-output recheck are E32/V6; no older
model computation or audit is silently re-attributed to the present edit.

The supplied CPU-only saved-output checker was rerun on a copy for this revision:
452 checks pass, including all ten result-manifest hashes/sizes, 5,928 unique
primary view rows, 48 identity semantic-summary rows, 240 class rows, 48
identity-policy source rows, 24 identity-policy aggregate rows, selected/frozen
identity reuse, and both development selections. Both training logs contain
120 unique completed updates with finite recorded losses and gradients. Twelve
supplied authoring-source files match hashes recorded by the runtime contract;
this is not a fresh fetch/audit of every runtime source. [V5]

No Qwen forward/backward, adapter/optimizer-binary audit, independent source
adjudication, calibration refit, threshold reselection, raw-logit NLL/Brier
reconstruction, or bootstrap rerun is performed by this saved-output checker.
SNLI counts are reconciled with stored confusion summaries, not independently
retrieved per-example predictions. Separate local tests of the next notebook
are software evidence under N1, not new E31 model evidence. One seed, repeated
exposure of the benchmark gate, small source groups, and incomplete source/semantic
review remain material limits. [E31; V5; N1]

---

## 18.24 Current synthesis and the next source-label replay test

**Historical interpretation at the v0.8.4 cutoff, before the v0.6.0 pretrained
run.** The following proposal and unrun-status statements are retained as the
prospective record. E32 and §18.25 supersede only its execution status; §18.26
owns the current interpretation. N1 remains the preparation artifact, not the
completed-result authority.

The completed model sequence now distinguishes **access**, **targeted learning**,
**behavior preservation**, and **retained correctness**. E29 shows that larger
inputs can improve supported decisions while none-class behavior changes in the
opposite direction. E30 shows substantial ContractNLI learning with failed
entailment and QASPER retention. E31 recovers some probability quality relative
to those specialists but finds no adaptation meeting the joint preservation
contract. Its short-premise results show task-dependent transfer rather than a
general forgetting verdict. [E29–E31]

The active hypothesis is narrower: **does replacing parent-only KL replay with
source-label supervision on the same train-only inputs improve retained
correctness without losing the targeted gains?** The accompanying v0.6.0 notebook
implements fresh J6/J7 fits from pinned J0/J1. It keeps the same 240 SNLI replay
examples and order, ContractNLI exposures, rank, learning rate, update budget,
renderer, arithmetic, and per-class development selector. The replacement loss is
mean eight original ContractNLI CEs plus 1.0 times mean two SNLI source-label CEs;
parent KL is absent. Source labels are traced through stored source rows and each
input's semantic/code mapping before training. This changes supervision and the
objective, not merely a coefficient. It is a prospective experiment with no new
pretrained-model result in this whitepaper. [N1; proposed comparison]

Primary fixed-dose comparisons are **J6_fixed80 versus J4_locked80** and
**J7_fixed80 versus J5_locked80**. J2/J3 remain contract-only controls; selected
J6/J7 views are evaluated against J0/J1. The old J4/J5 selected views remain zero
and must not replace their actual fixed80 adapters in the objective comparison.
Equal coefficient and matched examples do not imply equal gradient magnitude or
wall-clock computation. Fixed-dose and selected-pipeline estimands remain distinct;
source-label CE is not claimed to identify every mechanism behind the old KL
trade-off. [N1]

The same ContractNLI and SNLI-development preservation rules remain in force,
including eligibility for the frozen fallback. The existing 192-example SNLI
diagnostic is now explicitly **exposed regression evidence**, as is the reused
QASPER gate; neither enters the optimizer or checkpoint selection. Reusing those
panels allows paired regression comparisons, not fresh independent confirmation.
The narrow experiment deliberately adds no new backbone, evidence reader,
long-document dataset, hard-example mining, seed sweep, longer schedule, or
teacher-generated targets. It therefore does not claim to solve the insufficiency
of short SNLI as a broad retention panel. [E31; N1]

A favorable source-label result would justify another bounded research decision,
not release or a retroactive pass of E30/E31. Failure would further narrow the
supervision/transfer hypothesis without proving that Qwen, LoRA, or replay is
inherently unsuitable. Preserve historical selections and checkpoints; do not
relax constraints after inspecting the gate or use its errors as training examples.
A useful release scope still needs independently reviewed evidence, appropriate
unseen source/question/rubric families, and actual adapter-specific accelerated
execution qualification. The maintained native CPU/MLX FP32 reference does not
confer those properties on new fits. [E25; E29–E31; N1; §17.3]

---

## 18.25 Source-label replay pilot: stronger SNLI learning, failed joint preservation

**Completed experiment, not a proposed fit.** E32 is notebook v0.6.0, run
`source_label_v060_s17_bf16_dbb5e724b5452f23`, session
`20260926T170047_182185Z`, with result artifacts saved about 19:29 UTC on
26 September 2026. Execution is `COMPLETED_MODEL_RESULTS`; neither the fixed80
nor the selected view passes the complete research screen. Both new fits finish
120 updates, and both guarded selectors retain frozen0. [E32; V6]

The result resolves N1's execution question but gives a split scientific answer:
**source-label replay learns the short-premise SNLI task substantially better
than parent consistency, while failing to preserve supported entailment or
QASPER and worsening the main two-source probability scores relative to KL.**
A better replay-task score is not a retained-capability upgrade. [E32]

### 18.25.1 Attributable treatment and unchanged evaluation boundaries

J6 starts fresh from pinned J0 Base; J7 starts fresh from pinned J1 post-trained.
Neither loads an old adapter as initialization or resumes a historical optimizer.
The fixed80 primary comparisons are J6 versus actual J4 fixed80 and J7 versus
actual J5 fixed80. The `J4_locked80` and `J5_locked80` names in E32 identify
immutable diagnostic comparison snapshots, not changes to E31's selected-zero
outcomes. J2/J3 retain their original selected80 specialist identities. [E32]

Source-label cross-entropy **replaces**, rather than supplements, parent KL:

```text
E31: mean(8 original ContractNLI CE losses)
     + 1.0 × mean(2 SNLI KL(frozen parent || student) losses)
E32: mean(8 original ContractNLI CE losses)
     + 1.0 × mean(2 SNLI source-label CE losses)
```

The same 240 SNLI replay examples, exposure order, original ContractNLI schedule,
renderer and code mappings are retained. Rank 16, alpha 32, learning rate 2e-5,
120 updates, the BF16-backbone/FP32-adapter-and-loss profile and math-attention
checkpoint policy are unchanged. Each full fit has 960 ContractNLI exposures and
240 replay exposures; update 80 has 640 and 160 respectively. Parent scores are
diagnostic only in the new replay objective. Equal examples, dose and coefficient
do not imply equal gradient magnitude, difficulty or wall-clock compute. This
is a target-source comparison, not a gradient-matched or runtime-matched trial.
[E32; N1]

The source records an A100-SXM4-80GB with PyTorch 2.11.0+cu128 and Transformers
5.17.0. These are execution provenance, not new speed or numerical-equivalence
claims. Both training logs contain exactly 120 ordered updates with finite
loss/gradient records. V6 checks the saved logs, not the underlying CUDA gradients.
[E32; V6]

The original 741 primary questions across 72 components remain non-final,
previously exposed inputs. The gate has 204 ContractNLI questions from 12 contracts
and 46 QASPER questions from 12 papers. QASPER stays outside the optimizer and
checkpoint selector; its existing calibration-fit and policy-development roles
remain downstream. The 192-example SNLI regression panel is disjoint from the
240-example replay and 192-example retention-development roles within the recorded
normalized-exact-premise grouping. It is already exposed, not fresh confirmation;
image/near-duplicate-family and pretraining separation are not established.
[E32; E31; V6]

There are ten named reporting views but eight numerical checkpoint identities:
7,410 primary rows, 2,500 code/order pair records and 1,920 SNLI rows. The declared
11,830 question views include aliases and variants, not independent examples.
`J6_selected = J0` and `J7_selected = J1` match exactly in primary and SNLI output
rows. Zero selected-versus-parent differences are identity reuse, not a trained
adapter achieving perfect preservation. [E32; V6]

### 18.25.2 Positive replay-domain learning: SNLI

All values below use raw offered-outcome probabilities on the 192-example,
64-per-class **exposed regression panel**, not on replay training examples and
not on the retention-development selector. The Brier convention is the sum over
three declared semantic outcomes, not divided by class count. [E32]

| Checkpoint | Correct / 192 | Accuracy | NLL ↓ | Brier ↓ |
| --- | --- | --- | --- | --- |
| J0 — frozen Base | 117 | 60.94% | 0.92107 | 0.54200 |
| J1 — frozen post-trained | 132 | 68.75% | 0.72758 | 0.43185 |
| J2 — contract-only, locked80 | 136 | 70.83% | 0.81815 | 0.45812 |
| J3 — contract-only, locked80 | 141 | 73.44% | 0.69906 | 0.40497 |
| J4 — parent KL, fixed80 | 125 | 65.10% | 0.89585 | 0.52519 |
| J5 — parent KL, fixed80 | 131 | 68.23% | 0.80251 | 0.47444 |
| J6 — source labels, fixed80 | 159 | 82.81% | 0.43881 | 0.24465 |
| J7 — source labels, fixed80 | 166 | 86.46% | 0.40708 | 0.21611 |

V6 reconstructs softmax, semantic argmax, confidence, NLL/Brier and conditional
ranking from the exported per-example SNLI logits. All ten stored aggregate views,
including the selected/frozen aliases, reconcile. [V6]

Relative to J4, J6 repairs 46 SNLI decisions and introduces 12 errors, gaining
34 correct decisions (**17.71 percentage points**). Relative to J5, J7 repairs
40 and introduces five, gaining 35 (**18.23 points**). Relative to the older
contract-only specialists, the gains are 23 and 25 correct decisions. These
paired counts are saved-row reconstructions, not new inference. [E32; V6]

This supports the narrow hypothesis that original source labels provide useful
learning information beyond the tested frozen-parent distribution on this task.
It does not prove that individual parent errors caused all earlier regressions,
that source-label CE generally dominates consistency, or that the resulting
adapters preserve other tasks. [E32; interpretation]

### 18.25.3 Main benchmark quality: the replay gain does not transfer

These are raw identity-calibration gate results, all under common BF16 execution
and the same **4,096-state-token prefix cap**, not a total request length. The
source-label comparison does not replace the old FP32 evidence with BF16 controls.
The table retains original benchmark labels and every frozen/specialist/KL control.
[E29; E32]

| Checkpoint view | Source | Correct / n | Accuracy | NLL ↓ | Brier ↓ |
| --- | --- | --- | --- | --- | --- |
| J0 | contractnli | 116/204 | 56.86% | 0.97272 | 0.56700 |
| J0 | qasper | 40/46 | 86.96% | 0.33376 | 0.20364 |
| J1 | contractnli | 141/204 | 69.12% | 0.74811 | 0.42123 |
| J1 | qasper | 37/46 | 80.43% | 0.39921 | 0.26105 |
| J2_locked80 | contractnli | 166/204 | 81.37% | 0.55262 | 0.29263 |
| J2_locked80 | qasper | 35/46 | 76.09% | 0.66467 | 0.39704 |
| J3_locked80 | contractnli | 168/204 | 82.35% | 0.55134 | 0.29436 |
| J3_locked80 | qasper | 34/46 | 73.91% | 0.57011 | 0.34273 |
| J4_locked80 | contractnli | 164/204 | 80.39% | 0.51614 | 0.27979 |
| J4_locked80 | qasper | 36/46 | 78.26% | 0.48680 | 0.29875 |
| J5_locked80 | contractnli | 166/204 | 81.37% | 0.52210 | 0.28655 |
| J5_locked80 | qasper | 35/46 | 76.09% | 0.49796 | 0.31847 |
| J6_fixed80 | contractnli | 162/204 | 79.41% | 0.55277 | 0.30208 |
| J6_fixed80 | qasper | 29/46 | 63.04% | 0.76052 | 0.48762 |
| J7_fixed80 | contractnli | 165/204 | 80.88% | 0.54445 | 0.29752 |
| J7_fixed80 | qasper | 35/46 | 76.09% | 0.67368 | 0.38878 |

J6 versus J4 repairs four ContractNLI errors and introduces six, losing two correct
decisions. On QASPER it repairs one and introduces eight, losing **seven of 46**.
J7 versus J5 repairs three and introduces four ContractNLI errors; QASPER exchanges
one repair for one new error, leaving accuracy unchanged. Both new models have
**worse NLL and Brier on both sources** than their corresponding KL comparator.
Main-benchmark probability scores remain source-reported because the primary
CSV does not include full logits. [E32; V6]

This comparison favors source-label replay for the tested SNLI task, but not for
the original two-source retention objective. J7's unchanged QASPER accuracy must
not conceal NLL increasing from 0.49796 to 0.67368 or the changed error mix. The
ContractNLI point accuracies remain well above frozen, but that positive fact
cannot substitute for per-class and cross-task requirements. [E32]

### 18.25.4 Task-dependent semantic-none behavior

| Checkpoint view | Correct none / 5 | False-none / 41 | False-none rate |
| --- | --- | --- | --- |
| J4_locked80 | 4/5 | 9/41 | 21.95% |
| J6_fixed80 | 5/5 | 17/41 | 41.46% |
| J5_locked80 | 3/5 | 9/41 | 21.95% |
| J7_fixed80 | 4/5 | 10/41 | 24.39% |

J6's 100% QASPER none recall is five correctly detected none cases accompanied by
17 false-none errors. Both new false-none rates exceed the declared 20% ceiling.
Policy-development shows the same limitation: J6 assigns none to 18/38 answerable
questions and J7 to 15/38. These are semantic argmax outcomes, not errors repaired
by choosing a different automation threshold. [E32; V6]

On SNLI, false-none moves in the opposite direction: **26.56% to 7.03%** for
J6 versus J4, and **20.31% to 7.81%** for J7 versus J5. The observed behavior is
therefore not one uniform tendency to reject more everywhere. Domain, context
length, question meaning, option structure and their interactions remain
unisolated; this run does not identify a latent cause. Missing annotation spans
are still not automatically negative evidence or permission to change a label.
[E32; interpretation]

### 18.25.5 Supported-entailment regression and the guarded selections

Full ContractNLI semantic decisions, not conditional ranking with none removed:

| Checkpoint view | Entailed correct / 84 | Contradicted correct / 24 | Semantic-none correct / 96 |
| --- | --- | --- | --- |
| J0 | 81 | 10 | 25 |
| J1 | 81 | 8 | 52 |
| J2_locked80 | 73 | 16 | 77 |
| J3_locked80 | 70 | 19 | 79 |
| J4_locked80 | 74 | 15 | 75 |
| J5_locked80 | 69 | 19 | 78 |
| J6_fixed80 | 70 | 15 | 77 |
| J7_fixed80 | 69 | 17 | 79 |

J6 loses 11 correct entailed cases relative to J0 (**13.10 percentage points**)
and J7 loses 12 relative to J1 (**14.29 points**), exceeding the five-point
preservation allowance. J7's 17/24 contradiction recall also falls two cases below
J3's 19/24; the **8.33-point** loss misses the separate specialist-gain preservation
bound. Strong none/contradiction gains over frozen do not make those losses vanish.
[E32; V6]

The selector rejects candidates using only declared development evidence, not
these gate outcomes. Both frozen development baselines correctly entail 106/112
ContractNLI cases and 62/64 SNLI cases. The five-point loss allowances require
at least **101/112** and **59/64** correct entailments. [E32]

| Fit | Update | ContractNLI entailments / 112 | SNLI entailments / 64 | Contract development NLL | Failed preservation constraints |
| --- | --- | --- | --- | --- | --- |
| J6 | 40 | 82/112 | 58/64 | 0.71836 | ContractNLI entailment; SNLI entailment |
| J6 | 80 | 90/112 | 58/64 | 0.58564 | ContractNLI entailment; SNLI entailment |
| J6 | 120 | 96/112 | 59/64 | 0.58138 | ContractNLI entailment |
| J7 | 40 | 83/112 | 51/64 | 0.65127 | ContractNLI entailment; SNLI entailment |
| J7 | 80 | 93/112 | 55/64 | 0.55550 | ContractNLI entailment; SNLI entailment |
| J7 | 120 | 101/112 | 56/64 | 0.59305 | SNLI entailment |

SNLI overall accuracy and proper-score preservation pass, but individual
entailment preservation can still fail. At update 120, J7 meets the ContractNLI
class constraints yet remains three correct SNLI entailments short of the minimum.
No evaluated nonzero candidate passes all constraints at once; both selections
are correctly frozen0. The regression panel's different class outcomes cannot
be used to replace the development selection. This result says nothing about
untested intermediate candidates and authorizes no retrospective reselection.
[E32; V6]

On the historical 32-question ContractNLI cohort whose annotated evidence became
fully visible only with the larger prefix in E29, the current larger-prefix
scores are 30/32 for either frozen control, 26/32 for J4, 25/32 for J5, **23/32 for
J6 and 26/32 for J7**. These are small, fixed-membership diagnostic strata—not
new two-budget inference, independent semantic-sufficiency checks, or proof that
context length caused the regression. The recurring supported-entailment loss
persists without a uniquely identified mechanism. [E29; E32]

### 18.25.6 Uncertainty, policy utility and code/order diagnostics

The primary intervals below are **source-reported**, using 1,000 paired,
source-stratified component-bootstrap draws and 12 gate components per source.
Accuracy is the equal-source mean, not pooled-question accuracy. Negative
accuracy and positive NLL/Brier deltas favor the KL comparator. [E32]

| Primary contrast | Equal-source accuracy delta [95% interval] | Equal-source NLL delta [95% interval] | Equal-source Brier delta [95% interval] |
| --- | --- | --- | --- |
| J6_fixed80 minus J4_locked80 | -8.10 pp [-14.08, -2.85] | +0.15517 [+0.11390, +0.22108] | +0.10558 [+0.07482, +0.14038] |
| J7_fixed80 minus J5_locked80 | -0.25 pp [-3.81, +2.70] | +0.09904 [+0.04815, +0.20024] | +0.04064 [+0.01915, +0.07144] |

The recorded NLL/Brier intervals favor KL on the main benchmark; Base-side accuracy
is descriptively worse, while the post-trained accuracy interval includes zero.
A single seed and repeated exposure prevent an independent-confirmation claim.
No new interval or bootstrap is fitted for this documentation update. [E32; V6]

At identity-policy thresholds fixed on policy-development, J6 accepts **33/204
ContractNLI decisions with zero observed errors** and **8/46 QASPER decisions with
one error**. The row costs are 0.08382 and 0.10435. J7 accepts **21/204 with zero
observed errors** and **4/46 with one error**, costing 0.08971 and 0.11304.
Review-all costs 0.1. A zero observed error count in a small, clustered acceptance
sample does not establish zero future risk. [E32; V6]

| Checkpoint view | Pooled-row cost | Equal-source / equal-component cost | Weighted status |
| --- | --- | --- | --- |
| J4_locked80 | 0.08800 | 0.11970 | ABOVE_REVIEW_ALL |
| J5_locked80 | 0.08160 | 0.12046 | ABOVE_REVIEW_ALL |
| J6_fixed80 | 0.08760 | 0.12177 | ABOVE_REVIEW_ALL |
| J7_fixed80 | 0.09400 | 0.12940 | ABOVE_REVIEW_ALL |

Both new weighted costs exceed review-all even when the pooled-row costs are
lower. J7 accepts **zero QASPER policy-development questions** at its selected
identity threshold; conditional accepted error is undefined, not zero, and the
coverage requirement fails. V6 reconstructs exact identity-policy counts and
row/component fractions without refitting or moving any threshold. Break-even
is correctly treated as break-even, not a strict benefit. [E32; V6]

Code reassignment plus option reversal produces mixed agreement: QASPER J6 is
42/46 versus J4's 38/46, and J7 is 41/46 versus J5's 43/46. ContractNLI agreement
is 188/204 versus 188/204 and 191/204 versus 194/204, respectively. This is combined
code/order sensitivity, not pure permutation invariance or correctness, and no
inference ensemble is applied. [E32; V6]

### 18.25.7 Closeout and verification boundary

**Close this exact recipe:** source-label replay substantially improves SNLI
correctness and probability quality, but fails joint supported-entailment and
QASPER preservation; frozen parents retained. Neither fixed80 research screen
passes, and selected-zero aliases do not satisfy the adaptation-gain screens.
Keep J6/J7 snapshots as diagnostic artifacts, not promoted models. Preserve J2/J3
selected80 and J4/J5 selected0 history. [E32]

The supplied read-only reviewer passes **5,168 granular bookkeeping assertions**
when rerun on copies for v0.8.5. It verifies 12 manifest files, primary/shared-input
identities and counts, 120 semantic-count metric rows, 300 class rows,
identity-policy arithmetic, SNLI logits and aggregate scores, both selections
and ordered training logs. Sixteen authored-bundle source files match hashes
recorded by the runtime contract. These are assertions over saved data, not
5,168 independent tests, replications or model executions. [V6]

Not repeated: pretrained forward/backward execution, adapter/optimizer tensor
inspection, original-source adjudication, complete effective-input rerender,
main-benchmark NLL/Brier reconstruction from raw prediction caches, calibration
or policy fitting, bootstrap computation, or native/accelerated-service
qualification. Historical-preservation and source-read claims keep the run's
reported scope except where the listed saved-file checks directly apply. [V6]

## 18.26 Current synthesis after the replay-target comparison

The completed sequence separates several questions. **E29** measures evidence access
and full decisions; **E30** demonstrates targeted ContractNLI learning with failed
preservation; **E31** tests behavioral anchoring and recovers some probability
quality; **E32** shows that more useful supervision for the replay task can
improve that task while worsening the original multi-source objective. Semantic
accuracy, probability scores, per-class preservation, parent agreement and
policy utility are not interchangeable success criteria. [E29–E32]

The source-label proposal receives narrow support as an SNLI learning treatment,
not as a general preservation solution. Nor does its failure imply that all
source-label replay, all consistency training, Qwen or the joint-option graph
is unsuitable. The studies do not isolate sampling pressure, source supervision
coverage, domain, context length, rubric semantics or gradient interference.
Changing the next model size, reader, replay weights and training schedule at
once would not explain these observations. [E32; interpretation]

**Proposed next work, not a completed intervention:** make the recurring
supported-entailment regression the direct research target. First use admissible
training/development records to separate supported-to-none from
supported-to-contradiction changes, hypothesis-family concentration and
evidence-location effects. Then define one evidence-sensitive preservation
treatment on source-checked complete-document inputs. A retention panel meant to
support long-document or answerability claims must test those operations; short
SNLI alone has not protected them under either replay objective. No new loss,
dataset, fit, checkpoint choice or success claim is introduced by this revision.
[V6; proposed program]

Pause generic SNLI replay variants as the main remedy. Do not merely increase a
replay coefficient, extend an optimizer, add seeds or relax a class guard until a
candidate passes. Such changes require an explicit prospective hypothesis and
contract, not reinterpretation of E30–E32. The current QASPER and SNLI regression
rows remain out of corrective training and checkpoint selection; future reuse is
regression evidence, not untouched confirmation. A changed utility or application
scope is also a prospective decision. [E30–E32; V6]

The execution reference remains separate. Preserve the bounded CPU/MLX FP32
qualifications and immutable hybrid-state assets; new adapters need their own
cache identities, numerical parity, latency/memory and service tests. No retained
fit inherits those qualifications or opens protected final. Existing independent
source-review limitations, audit quarantines, label semantics and historical
selection locks remain binding. [E25; E29–E32; §17.3]

---

# Conclusion

OpenKind has a bounded execution foundation: typed decisions, complete
hybrid-state reuse, native CPU parity and persistence replay, named-machine CPU
service evidence, and separately qualified pinned MLX FP32 parity. Its strongest
contribution is the joint contract for rendering, state isolation, numerical
behavior, probabilities, and policy outputs, together with retained evidence
of where optimizations and model shortcuts fail. [E11–E13; RUST1–RUST11; §17.3]

The natural-document record now includes a substantial, bounded positive
learning result. M2.1 shows that improved evidence visibility can help while
none-class behavior moves in the opposite direction. The matched M2.2 LoRA pilot
improves ContractNLI accuracy, contradiction, and semantic-none decisions, but
reduces entailment recall and QASPER retention; both adapted arms fail the
complete research screen. The result is useful task specialization, not a
promotable multi-source model, independent semantic adjudication, or a universal
checkpoint/architecture winner. [E29; E30; V4; §§18.20–18.22]

The completed parent-KL replay follow-on partly recovers probability quality
relative to the contract-only specialists but produces no adaptation satisfying
joint preservation. Both development selectors retain frozen parents. The
contract-only specialists' stronger SNLI scores also show why task-dependent
transfer should not be renamed general forgetting and why parent agreement is
not the same as useful correctness. [E31; V5; §18.23]

The source-label replay comparison is now completed. J6/J7 improve SNLI regression
accuracy to 82.81%/86.46%, yet worsen ContractNLI/QASPER probability scores relative
to parent KL and fail supported-entailment and QASPER preservation. Both guarded
selectors again retain frozen0. Better replay-domain learning does not establish
retention of the intended decision scope; opposite false-none changes on SNLI and
QASPER make the task dependence explicit. Historical J2/J3 selected80 and J4–J7
selected0 identities remain unchanged. [E32; V6; §18.25]

The next proposed work directly diagnoses supported-entailment regressions on
admissible complete-document training/development records before choosing one
preservation intervention and a retention panel that exercises the required
operations. Neither another generic SNLI replay variant nor a larger architecture
is justified as the automatic next step. The causes remain unisolated, and no new
corrective treatment or prospective contract is supplied by this paper update.
Inspected gate/regression errors do not become training or selection examples.
[E32; V6; §18.26; proposed program]

The M0–M4 distinction between evidence, useful decisions, lower cost and
independent confirmation remains. One immutable-profile MLX performance study
can proceed as separate systems work. New adapters need their own numerical,
cache, backend and service qualifications; no CUDA fitting result transfers
those properties automatically. Independent correction review, held-out task/rubric
families, protected final and accelerated-service evidence remain open. This is
a documentation-only update, not a change to the roadmap, notebooks, historical
selections, source labels, policy thresholds, authorizations or experiment
artifacts. [§§13.2–13.5; §17.3; E25; E29–E32]

---

# Appendix A. Source and reproducibility register

The source IDs below identify the evidence behind the numbered sections. In the accompanying evidence manifest, local snapshot SHA-256 hashes distinguish the exact files reviewed from later Drive edits. Result paths are under `Google Drive / Colab Notebooks`. Timestamps embedded in run IDs are UTC.

**27 September 2026 systems addendum boundary.** RUSTM2 uses checked-in Rust
source, raw paired-process JSON, checksums, and formal pinned-model parity
reports. The Qwen3.5 benchmark measures native compute and readout on frozen
token workloads, not model quality, full HTTP requests, or queue-inclusive
latency. It does not rerun or revise E29–E32 model experiments, open protected
final, change the Jev API, or promote the automatic scheduler.

**Version 0.8.5 documentation boundary.** Editing base: the delivered v0.8.4
`WHITEPAPER.md`, SHA-256
`893fcbdcc49f2511d951e1f4764a8ed533b183521381ad8f7841b3afa34e277a`.
This revision adds E32/V6 and §§18.25–18.26; it dates §18.24's former proposal,
updates current status and synthesis, and preserves historical numeric sections,
prior audit scopes, selections and Appendix E. N1 remains historical preparation
provenance; E32 is the completed-run authority. The supplied v0.6.0 review archive
is the evidence snapshot; its named Drive URLs are source locators, not a claim
that every live file was newly fetched for this edit. Only new versioned
whitepaper deliverables may be saved; existing experiment files are not modified.

**E32: Completed source-label replay target comparison (v0.6.0).** Run
`OpenKind_Source_Label_Replay_results/source_label_v060_s17_bf16_dbb5e724b5452f23`,
session `20260926T170047_182185Z`; results saved about `2026-09-26T19:29Z`.
[Run folder](https://drive.google.com/drive/folders/1EMrI0aaw3gqk2chHZOgtJ4fMFEtosOol),
[report](https://drive.google.com/file/d/1YX6xJU9X2mQS9c8KfQ45JnwaQD7xOzIq/view),
[full results](https://drive.google.com/file/d/19KCQ4m4QHPSvggrdl5a0WgBNO5kUn70i/view),
[contract](https://drive.google.com/file/d/1hYmgYR3IxhToXYru-uTcXZyDAGCvv7oJ/view),
[result manifest](https://drive.google.com/file/d/1MkDx45s9bI6DNOkAnXsDhXEWzf_wR4y0/view),
[J6 selection](https://drive.google.com/file/d/1JeG68fgi82j0IYgJemcSkKiH8H2unWp9/view),
[J7 selection](https://drive.google.com/file/d/1sgCVBjYnW8sfP_NlGDZl5fe7kk0A7q3s/view),
and [SNLI per-example logits](https://drive.google.com/file/d/1ljIAm7bQ_iA-pjLUK-_GOj3f-lUX48wY/view).
Report SHA-256
`1584373f7969b23c7f4e4a8f26f9e5f82e8376a904f9faecf33433847d41695c`;
full-results SHA-256
`63f08096f307f8ac7d7c0a09f88a42ad6ea6d95399b2e0ffb8aedf2e639dcbd0`;
primary-row SHA-256
`cb39f56b3a2efcf630d8462b8ff311d92bfdd87312d44e689e6a986736cabc40`;
SNLI-row SHA-256
`60d924069f8ef48301b80be7bece638156918fab5227e7c5f4943eae296576fb`.
Contract file SHA-256
`4f799c7615e9aaefaacbb204c4e087df54de80d21428f566def1546eea2141c1`;
J6/J7 selection-file SHA-256 values are
`378188f94226764cddd25c7cd74589fe032c59d048ff0687bc04098a0d341c96`
and `b5f730d0cfed2478f05eb0101701ca27c9bb3bd75ab6bc9350b84d5e74d19e0e`.
The twelve-file result manifest binds the reports and tables. Source labels enter
only the disjoint replay role's new loss; QASPER and SNLI regression do not enter
the optimizer or selector. Both fit120 and select0. Fixed80 snapshots remain
distinct diagnostic views; no historical selection is changed.

**V6: v0.6.0 saved-output review and v0.8.5 documentation checks.** The supplied
`OpenKind_Source_Label_v060_Completed_Run_Review.zip` contains `review.py`, exact
source snapshots, `SOURCE_INDEX.json`, derived counts, selection reconstruction
and explicit audit limits. Rerunning the standard-library script on copies passes
5,168 granular assertions. These cover 12 result files, 7,410 primary rows,
120 semantic-count metric rows, 300 class rows, exact identity-policy counts and
fractions, 1,920 SNLI logit rows, both selectors and ordered 120-update logs.
Sixteen bundled implementation files match hashes recorded in the execution
contract; no independent runtime-source download is claimed. Main-benchmark
NLL/Brier and all paired bootstrap intervals remain source-reported. The new
paper's generated tables, retained historical sections, headings, source hashes
and Appendix E are checked separately. No model execution, original-source
adjudication, tensor-binary inspection, recalibration, threshold fitting,
bootstrap rerun, protected-final access or systems qualification occurs.

**Historical documentation records below.** Each earlier version's validation
and preparation claims retain its original cutoff and scope. In particular,
N1's unrun statements describe the v0.8.4 authoring stage, not the completed E32
run recorded above.

**Version 0.8.4 documentation boundary.** Editing base: the delivered v0.8.3
`WHITEPAPER.md`, SHA-256 `407df614b4fd5a7a2350c515129413142ced3eed7f74e552b77dd471983ae95f`. This revision preserves its completed historical
measurements, prior audit scopes, and Appendix E. It adds E31/V5, dates the former
§18.22 synthesis, and updates active status/next-work language. N1 is a separately
prepared experiment, not a new trained model. The accompanying diff and validation
manifest identify every edit. No historical Drive experiment file is changed.

**E31: Completed parent-KL retention follow-on (v0.5.0).** Run
`OpenKind_Retention_Adaptation_results/retention_v050_s17_bf16_a69ba45a76e74b63`,
session `20260926T023951_020889Z`; results saved about `2026-09-26T04:56Z`.
[Report](https://drive.google.com/file/d/1YHm5PSVb6GTC3kNwLKboooJq-TQKlDMZ/view),
[contract](https://drive.google.com/file/d/1KiQbNr5AO8iRzHZsxDMLLYRY_fdFdkPa/view),
[result manifest](https://drive.google.com/file/d/1nxqd3lapZ1f8yZOCn66JDcBkPe1s1gfv/view),
[J4 selection](https://drive.google.com/file/d/1_Bi9VbZiRD0XG4vY1u7t8hbIu3Wg1AbY/view),
[J5 selection](https://drive.google.com/file/d/1oAHfkB5dcV7Z2MpDdVsxeDu1SSsM0KLw/view),
and [retention data lock](https://drive.google.com/file/d/1epw_LpAV_Zw-N5f_oAFdrCDsckU-dD4v/view).
The ten-file result manifest binds report SHA-256
`63825573e576b30cde350e9cd25e71a056da6db6ebc01eca7d62d9831df8ed8b`,
full-results `d324131026e9f1bf5f07fd6ac5aa6d0e6afcc3a0715c22535978fa5584374133`,
and primary-decision rows `86bab451b5fb10580e718f8ddfd59e1e2be592dc886de2fc5de52aa044c71293`.
Both fits attempt 120 updates, both selected views resolve to frozen0; fixed80
remains a distinct diagnostic view. Stored source labels/roles are unchanged.

**V5: v0.5.0 saved-output review and v0.8.4 documentation checks.** The supplied
`OpenKind_Retention_v050_Completed_Run_Review.zip` contains source snapshots and
`review_saved_outputs.py`. Rerunning it on a copy passes 452 checks with the scope
in §18.23.7. The companion includes the executed report, source hash manifest,
exact paper diff, and checks that historical result sections/tables and Appendix E
remain unchanged. This does not revalidate CUDA gradients, reconstruct raw-logit
proper scores, inspect saved tensor binaries, or re-adjudicate source evidence.
Any older audit not rerun here keeps its historical attribution.

**N1: Source-label replay comparison workbench (v0.6.0), not completed model
results.** `OpenKind_Source_Label_Replay_v0_6_0.ipynb`, preregistration, source,
tests, and release hashes are delivered separately. J6/J7 start fresh from J0/J1;
the sole planned learning change versus J4/J5 is source-label CE instead of
parent KL on the same SNLI replay schedule. The earlier 192-example diagnostic
is explicitly regression evidence. Local checks comprise 99 unit/regression tests,
nine tiny-CPU integration checks, and source/target/role reconciliation of 624
retained source records. The authoring environment lacks CUDA, Transformers, and
PyArrow; actual pinned-shard validation, tokenizer rerendering, Qwen CUDA preflight,
and pretrained fitting/evaluation remain runtime checks. Neither these software
checks nor the frozen controls establish a successful retention outcome.

**N1 completion cross-reference (v0.8.5).** The historical preparation record
above is unchanged. E32 now supplies completed pretrained fitting/evaluation,
with both selected-zero outcomes and failed full retention; its result is not
inferred from N1's local software checks.

**Version 0.8.3 documentation boundary.** Editing base: supplied `WHITEPAPER.md`, SHA-256 `9579909e53dff0d0fd404632f763b416df78c05285ab600cabdc539c05b4656f`. Result timestamps determine the 25–26 September cutoff. The update adds E29/E30 and the V4 companion, updates current interpretation/status text, and preserves original numerical histories and prior approvals. All 15 newly used result-manifest entries and seven E30 source hashes are checked from supplied local snapshots. The scripts operate on copies; no live Drive notebook, result, checkpoint, source label, policy threshold, or protected-final artifact is changed. Probability scores remain source-reported where no raw-logit reconstruction is performed. The README, exact diff, revision manifest, and document checks accompany this Markdown.

**E29: M1 preparation and completed M2.1 frozen two-budget diagnostic.** Run `OpenKind_M1_M21_Closeout_results/m1_m21_joint_option_v032_s17_fp32`, session `20260925T174531_575346Z`, engine `da6e86349af2df1f`. [Result folder](https://drive.google.com/drive/folders/1XpQFA256aAdc5S075zKUO2-XJoN00TVt), [report](https://drive.google.com/file/d/14QNUaQbTQtlnZctqDFneDilsjYyrA5Zb/view), [full results](https://drive.google.com/file/d/1C7Q15Qamds3bwW6RhXzmV5ETc4TRER6C/view), [decision rows](https://drive.google.com/file/d/1WYPE9l6rBVPSBIOmcK8gxrf75LHi_Jd2/view), and [result manifest](https://drive.google.com/file/d/1r6wtSUWujZhO84w79_D0VDdeZU4yTwle/view). Completed report SHA-256 `1f4b976f45686b0e2e30a7ef6b8e95f5b1653c99753070bd422612bfa2502a97`; full-results SHA-256 `343b4644046b3193f324f909810c5d129fe2b98a26512ddd410c73778a5052dd`; decision-rows SHA-256 `6f1f4251a53ddbe5158b27faada62a120ab11231fdbf9ef3fb4c7073d19e516d`. Six result files are bound by the manifest. The earlier v0.2 session `20260925T140444_525763Z` supplies the separately identified `CLOSEOUT.json` historical-score reconstruction and `EVIDENCE_VISIBILITY.csv` preparation diagnostics; its no-inference status is not the later v0.3.2 status. The v0.2 protocol that selected only 1,024-token inference is not presented as the completed v0.3.2 two-budget protocol. Local sources are retained in the revision companion.

**E30: Completed M2.2 matched decision-LoRA pilot.** Run `OpenKind_M22_Decision_LoRA_results/m22_v041_s17_bf16_6b295d46b7436a52`, session `20260925T232651_880064Z`; results saved approximately `2026-09-26T01:21Z`. [Run folder](https://drive.google.com/drive/folders/1cpL0X1qIfJJCBxsD93PL3eJx5ui3q2pM), [report](https://drive.google.com/file/d/1Ov6794Pey5bkXrW8W73acGy2EylFhEQQ/view), [full results](https://drive.google.com/file/d/14DbznQQlYORTspE48hjS3dl5YmEosRgb/view), [contract](https://drive.google.com/file/d/1eWYvPEhcyNagNGt54pYBk1AWsTDYYroX/view), [data lock](https://drive.google.com/file/d/1c-qzDH9vcTJQ6KzPjf9SKWpTPQx33J1o/view), [J2 selection](https://drive.google.com/file/d/1-01oGgenUE14b8EDPpXLa-pKuqAKY6-y/view), [J3 selection](https://drive.google.com/file/d/1ELb_t9ulfQdkhKbya1YV75DOMQVKodp7/view), and [result manifest](https://drive.google.com/file/d/13Qk6Rpn1X93iU99rZzE5F8dPuyRLf_QF/view). Report SHA-256 `995d0807201cfec8217694b092d3f74374eff0392f0644d9a1af85a5f21d2aca`; full-results SHA-256 `2b15b867679c0c9b4c171d5bcd17d61a172ca0da262f1398a744a5d9b7b14bba`; primary-decision-rows SHA-256 `3f34d8ca83c270607c6a23927af3c21102b542ebcea4dc50f53a11a757122614`. Canonical data-lock SHA-256 `4d193807b9515ab24661d36ebb3a1bffe4a1b79b892a811fde631a6397cbd0d2`. J2 training identity `c5f70caa14fcafc69f0c71c457263b49aa956713b36747018110a957d0984128`; J3 identity `6449d8f47628f98f3b634977bd47c7e339f122e90a851299ce02bbbd28ba84b6`; both select `step_000080`. Nine result files and seven runtime sources are checked by V4. This is a separately authorized bounded pilot, not an edit of E25's historical no-training approval. Original source-record checks are not independent semantic adjudication. The metadata string `TOKENIZERS_READY_NO_WEIGHTS_LOADED` inside the contract is a tokenizer-stage status, not a claim that the later completed training/evaluation loaded no weights.

**V4: M2.1/M2.2 saved-output reviews and v0.8.3 documentation checks.** Supplied companions `OpenKind_v032_Completed_Model_Readout.zip` and `OpenKind_M22_v041_Completed_Run_Review.zip` contain acquired source snapshots and CPU-only readout scripts. This update reruns those scripts on copies, checking the counts, arithmetic, selection, and explicit limits in §§3.2 and 18.21.7. E29 reconciles 2,964 unique decisions, 48 semantic-count rows, 24 identity-policy rows, and 64 fixed-visibility count/accuracy records. E30 reconciles 2,964 primary rows plus 1,000 code/order records, 120 class records, schedule/selection, and policy weighting. Its additional 10,000-draw document-bootstrap intervals are review-derived, not the original source-macro intervals. The exact-break-even policy patch remains **proposed/unapplied** to the original evaluator. This revision does not execute experiment modules, refit or reselect anything, inspect adapter/optimizer binaries, read protected final, or renew historical native/backend qualifications. The revision package supplies `WHITEPAPER_v0.8.2_to_v0.8.3.diff`, source/output hashes, and preservation checks; historical documentation claims retain their original audit scope.

**Version 0.8.2 documentation boundary.** The editing base is `OpenKind_Whitepaper_v0.8.1.md`, SHA-256 `801ac783c4ef07d828d703f2658d7c3576e50e5b4d9adb73ef94a9a4467dd590`. E14–E20 were read directly from the named Drive artifacts. This revision preserves the v0.8.1 Phase 4A/4B tables and adds the completed stratified, pairwise, head-only, and evidence-residual contracts, reports, evaluations, and result locks. It reconciles their selected epochs, thresholds, reported counts, proper-score changes, policy costs, and artifact identities. It does not rerun training or model inference, adjudicate labels, or inspect final labels/predictions. The completed non-final locks remain the result authority; legacy `OpenDecision_...` folder names are retained as provenance rather than silently renamed.

**Historical version 0.2 audit boundary.** The completed Phase 2F archive SHA-256 is `e1d3fea878d35b9e929e83424d4f3aa8b5de0385bb100ace001110f7e5f83c37`. The version 0.2 companion retained version 0.1 evidence and added the Phase 2F manifest, compact results, and an independent standard-library aggregation audit. Probability and policy comparisons are recomputed where raw vectors/actions are retained. Request/long-prefix rows retain comparison diagnostics rather than all output vectors, so their numerical gates remain source-reported while timing aggregates are independently checked. No source Drive files were modified. [E6]

**Historical version 0.3 audit boundary.** The attached v0.2 source SHA-256 is `1adad381998535be157cc8c3806427ff7db1e8e21b7cd05afad5129969942cc2`. The completed G archive SHA-256 is `24ff15fa1fec3c8eeb17ef073da3d852cabe40683678b247d4e1d917e847414b`. The v0.3 verification companion includes a standalone saved-output checking script, its aggregate JSON, this revision’s source/hash manifest, the earlier correction and G review notes, a change log, and a text diff. The script does not download or run Qwen. Section 3.2 retains the checks rerun for v0.3; the earlier wider audit is attributed to R2. All v0.3 numerical additions are from the supplied/completed artifacts, not new external research. No original attachment, notebook, source result, or Drive file was modified.

**Historical version 0.4 revision boundary.** The supplied v0.3 Markdown SHA-256 was `6d73d8ea4e36249b98629997768fe5ec35c32de744dfd09670efd1cf701b0e08`. That revision preserved completed-result tables and checked textual integrity, source hashes, and status consistency. It did not rerun the v0.3 saved-output checker or independently reopen the B–G archives. Its companion contained a change log, unified diff and source/integrity manifest, not a new experiment audit. H0 and R3 were read-only working copies; persistent sources were unchanged.

**Historical version 0.5 revision and audit boundary.** The supplied v0.4 Markdown SHA-256 is `122da05cc4c4668b03a0cdc8fabfc65903fd7a40f72a8a2e4c030dfdbd756ec4`. The H archive SHA-256 is `51c772ab248d8949fe73bbbd91056826975eebb3594d03ca693438355d9a4cda`. Version 0.5 preserved the strategic framing and completed B–G numeric tables, added the partial H readout, and updated its status throughout. The companion contains the complete whitepaper, change log, unified diff, source hashes, a runnable NumPy/standard-library saved-artifact checker, its output, and selected metadata/source snapshots. The checker verifies the bounded scope stated in Section 3.2; it does not run Qwen, reconstruct unavailable development logits, open final stimuli, or validate held-out H results. No source notebook, run status, reservation registry, original attachment or persistent Drive file is changed. [E8]

**Version 0.5.1 synchronization boundary.** The editing bases are R4 (uploaded roadmap) and R5 (newer Library whitepaper v0.5). The same E8 run/archive is rechecked directly against the saved Colab finish output; this does not supersede its partial status or add a final result. The V1 checker records 22 source-consistency checks and leaves model computation, final-data inspection, previous numerical audits, source repair and repository tests outside scope. The revision package includes exact diffs, hashes, a source-reconciliation report and a document-integrity check. No source notebook, failed-attempt status, reservation registry, original attachment or persistent Drive file was changed.

**R4: Uploaded roadmap and current task migration.** `ROADMAP(20260919-123538).md`, conversation file `file_00000000657881f5b537b1e3162383be`. Supplies code-milestone descriptions, historical test counts and the conflicting proposed Phase 2H name. It is not independently verified repository state. The historical revision 0.5.1 of the companion roadmap restored the measured H experiment, adds its development/closeout boundary, and maps uncompleted proposals to 2I/2J/Track S/Phase 3. Source and output hashes are retained in the companion manifest.

**R5: Prior whitepaper v0.5 and its evidence package.** Library files `file_00000000fe4c81f5b521d6540f042f85` (`OpenKind_Whitepaper_v0.5.md`) and `file_00000000d21881f59c548cdabcd28071` (`OpenKind_Whitepaper_v0.5_Evidence.zip`). The Library Markdown is byte-identical to the paper in that evidence package. Its completed B–G and partial H numerical tables are retained; its earlier audit results remain attributed to v0.5 rather than represented as work rerun for v0.5.1.

**V1: Saved-source and document reconciliation for v0.5.1.** `check_saved_sources.py`, `evidence/source_reconciliation.json`, `evidence/document_integrity.json`, source/output hash manifest and exact diffs in the accompanying revision package. The check reads named metadata/report/code artifacts without executing experiment modules or opening `final_payload_reserved.json`. It verifies consistency of retained records, not the generalization validity of development-selected models or the existence of any unsaved live Colab continuation.

**H0: Phase 2H design snapshot, not completed evidence.** `OpenKind_Phase2H_Criteria_Rejection_Multidomain.ipynb`, version `2h.1.0`, Library file `file_00000000044481f585f71c2a526ebdc9`, snapshot version 1; SHA-256 `05906ff8aafc1add2fdd1f818cc077cdf0ea9b2582254b1cb9c2de67f2e698d6`. Reviewed notebook methods, configuration cell, treatment descriptions, final-data controls, and stated boundaries. The snapshot has 30 cells and no executed code cells. The default configuration enables the main/TF32/primitive/finite-token studies and disables state-first, smaller-model, and LoRA extensions. These defaults do not establish the configuration or progress of the owner’s active run; embedded code was not executed or comprehensively audited.

H0 remains a historical design source. E8 records the original `2h.1.1` configuration, fitting outputs and failure; E9 supplies the completed `2h.1.2` continuation. Default flags, notebook availability and earlier owner status do not substitute for these versioned outcomes.

**U1: Project-owner status and revision request.** In the request accompanying v0.3, Chris states that Phase 2H is in progress and asks to retain it while refocusing the whitepaper on feedback in S3. This supplies status and editing intent, not run completion, enabled-arm details, or measurements.

**R3: Recovered architecture-review note.** `Pasted markdown(20260919-032946).md`, Library file `file_00000000c58881f5997fb51ee11dbef0`, snapshot version 1; SHA-256 `27b1ab204729db4d490ad8acd72aa563cd2aa1c330ccf8596acc33680405e129`. Read in full. Supplies the central refocus, separate equivalence/new-model tracks, multi-question and smaller-Qwen priorities, evidence-aware rejection proposal, nested-sharing design, API concerns, prototype service path, and documentation-drift observations. It reviews documentation rather than confirming current Rust implementation or test results. It is strategy feedback, not a new experimental artifact.

**E0: Initial feasibility probe.** `OpenKind_Phase2_Qwen3_5_4B_Probe.ipynb`. Executed exploratory notebook; model-loading comparison, pooling shapes, two-example head fit, and rough generation timing. [Open notebook](https://colab.research.google.com/drive/1bd1FNFL7FP0fRxBLg17XW9yz2kEXHNh8).

**E1: Phase 2B.** Run `20260917T205849Z`; `OpenKind_Phase2B_results / <run> / openkind_phase2b_summary.json`. Principal fields: `architecture`, `data`, `all_metrics`, `matched_test`, `matched_uncalibrated`, `performance`, `generation_comparison`, `batch_invariance`, `frozen_export`. [Full result](https://drive.google.com/file/d/1Kj6Ph12DYU8VwRHnQ4-vTgA4Q2f2RmmS/view).

**E2: Phase 2C.** Run `20260917T222948Z`; `OpenKind_Phase2C_results / <run> / openkind_phase2c_summary.json`. Principal fields: `nli_data`, `nli`, `stability`, `dynamic.data`, `dynamic.training`, `dynamic.evaluations`, `dynamic.robustness`, `dynamic.export`, `frozen_export`. [Full result](https://drive.google.com/file/d/1lXy-mCtjkPkgb3KZR8fsn0mRKEZNB4E9/view). [Executed notebook](https://colab.research.google.com/drive/1iYTfdtn1-G26zpNyK98iodp_IGBi1YAz).

**E3: Phase 2D.** Run `20260917T234417Z`; `OpenKind_Phase2D_results / <run> / openkind_phase2d_summary.json`. Principal fields: `numerics`, `dynamic_data`, `selection`, `temperature`, `evaluation`, `request_benchmark`, `export`. [Full result](https://drive.google.com/file/d/12EDUXCzU3psTCDN-qTkkKTiRpxvnzkMX/view). [Executed notebook](https://colab.research.google.com/drive/1N_V3MnrZUbddO5j-ZQqDew_dEjXig6p1).

**E4: Phase 2E, original systems/policy run.** Run `20260918T032049180933Z`; `OpenKind_Phase2E_results / <run> / openkind_phase2e_summary.json`. Principal fields: `data`, `precision.modes`, `shared_prefix`, `policies`, `policy_scoring`, `run_status`. Overall status is partial; completed policies remain valid evidence within their scope. [Full result](https://drive.google.com/file/d/1JjBbFdsFHjjXqdzBkFpigwZ9QhDWWgzq/view).

**E5: Expanded Phase 2E.** Run `20260918T114914072764Z`; `OpenKind_Phase2E_expanded_results / <run> / openkind_phase2e_expanded_summary.json`. Principal fields: `sampling`, `workers.<mode>.parity_rows`, `parity_summary`, `benchmark_summary`, `long_prefix`, `memory_summary`, `cross_precision`. Completed execution; BF16 strategy-equivalence gates fail. [Full result](https://drive.google.com/file/d/1SxOG4VY4TlqK0e4eqBgZ_KwfDYjbXeFX/view). [Notebook containing the expanded run](https://colab.research.google.com/drive/17fuU04ZwOc2RJdahFiIyFaJ88YBgcvk8).

**E6: Completed Phase 2F.** Run `20260918T224427722898Z`, version 2f.1.0; results saved at approximately 23:56 UTC on 18 September 2026. Source folder: `OpenKind_Phase2F_results / <run>`. Both precision workers completed. Principal fields: `baseline_parity`, `compression_rows`, `benchmark_rows`, `component_profiles`, `cross_request_summary`, `long_prefix_rows`, `cross_precision`. The archive also retains raw `cross_request_rows.json`, inputs, source modules, and frozen head exports. [Full result](https://drive.google.com/file/d/1jKdR5xozQ7CQfzaTT82OQ7bszBEfJGPe/view). [Paste-back summary](https://drive.google.com/file/d/1UJMv80N6vQ_bgdQH4ejZfIztj8xHZ8TS/view). [Complete archive](https://drive.google.com/file/d/1m9PbuQqFlO4e3ilEpaSJ4Qvf8VcW56_-/view). [Notebook](https://colab.research.google.com/drive/1DmkUfMAUwgnn4vWg_kWJe2cK75QVr60G). The version 0.1 partial notebook snapshot remains historical evidence, not the current run status.

**E7: Completed Phase 2G.** Run `20260919T005142584348Z`, version `2g.1.0`; results saved around 01:45 UTC on 19 September 2026. Source folder: `OpenKind_Phase2G_results / <run>`. Both `fp32_strict_math` and `fp32_tf32_allowed` workers completed. Principal fields: `data`, `workers.<mode>.fresh_summary`, `context_summary`, `context_by_length_position`, `benchmark_rows`, `traffic_summary`, `flags`, `memory_snapshots`, and `cross_precision`. The archive retains `experiment_inputs.json`, `fresh_message_manifest.json`, `prior_exclusions.json`, worker `fresh_rows.json`, `context_rows.json`, `traffic_rows.json`, `traffic_events.json`, `benchmark_rows.json`, implementation snapshots, and frozen coefficients. [Full result](https://drive.google.com/file/d/1liuu456rvPFIQJXonRjxh83DTdJjKUGE/view). [Paste-back summary](https://drive.google.com/file/d/1F5vaHSj2Grix-RNChoTn2mhOWhet7pb2/view). [Complete archive](https://drive.google.com/file/d/1CAm4ooAuQ8gsxJAZNHn7jsnztnkF9IzP/view). [Notebook](https://colab.research.google.com/drive/1KVEB2apgM_8GIFdjLsX0x9LPvc94Pnzg). Freshness is relative to recorded project manifests; completion and strategy acceptance remain distinct.

**E8: Partial Phase 2H, saved fitting/development evidence.** Run `20260919T040612625670Z`, version `2h.1.1`; outputs exported around 05:03 UTC on 19 September 2026. Source folder: `OpenKind_Phase2H_results / <run>`. Overall status `partial`; `fit_qwen4b` status `failed`, exit code 2, nonfatal `NameError: name 'memory_snapshot' is not defined`. The archive saves 14 main development profiles, 126 policy-selection records, six primitive fits, sampled weight-integrity status and two development-head fixtures. Principal artifacts: `paste_back_summary.json`, `openkind_phase2h_summary.json`, `fit_qwen4b/worker_report.json`, `profiles.json`, `selection.json`, `traceback.txt`, `split_manifest.json`, fitting payload, source snapshots and fitted `.safetensors` files. No final-evaluation worker or final lock is present. [Full result](https://drive.google.com/file/d/1IT4cJN74vgOW0td7haE_bl2KfviHiaP1/view). [Compact summary](https://drive.google.com/file/d/1525L3h-0hVKtgm50c_IAX1dCAKHNZNUe/view). [Attempt archive](https://drive.google.com/file/d/1X8JP-8hhb3lu_PmMCWNLovMMdXPK6eDO/view). [Notebook](https://colab.research.google.com/drive/1fhRJTek7Ura3aSdItwBjJTJubXUIee4a). These references identify the inspected attempt, not a later repaired run.

**E9: Completed Phase 2H continuation.** Run `20260919T040612625670Z__finish_2h_1_2`, version `2h.1.2`, originating from E8. Both `eval_qwen4b_strict` and `eval_qwen4b_tf32` completed; outputs were saved around 14:45 UTC on 19 September 2026. Source path: `Google Drive / Colab Notebooks / OpenKind_Phase2H_Finish_results / 20260919T040612625670Z__finish_2h_1_2`. [Full summary](https://drive.google.com/file/d/1rhO2B5ro3rQxEGJ3ZSt6JhflluV7-NPz/view). [Compact summary](https://drive.google.com/file/d/1072Mee3vu6JFjMXADPBe-GWm_BhCbGGE/view). [Completed continuation archive](https://drive.google.com/file/d/1kSnKjnOinF6fvyc-ZO9UF3rbfNT5eMCR/view). [Fitted-artifact recovery](https://drive.google.com/file/d/139rnY0E9EK6lIXrXDE0PhmSESZrPNiIu/view). [Final lock](https://drive.google.com/file/d/15nnlgaO__RDAzVQOKh96LzF9XvIBWkeG/view). [Finish notebook](https://colab.research.google.com/drive/1FAhX21Es0LsDYLS_CotbwsR1OsxrXAUS). The archive SHA-256 is `7df9de857e4683386f4b687244e0bbc28a4377855d5d696d1699cb973e76fa8f`; its preserved original-attempt ZIP hash is `51c772ab248d8949fe73bbbd91056826975eebb3594d03ca693438355d9a4cda`; final-lock SHA-256 is `3b30ced9af995812aeb23db12fdedfb7ba3247324c184e56871dd9e4108c3d14`. The source contains per-worker reports, final/reliability prediction rows, parity rows, benchmark rows, primitive predictions, fitted artifacts, final lock, input manifests and source snapshots. Its compact archive excludes the evaluation feature databases, which the source says belong to worker checkpoints. Later Drive modifications are not implicitly part of this snapshot.

**V2: Version 0.6 saved-output and lineage audit.** `audit_phase2h_finish.py`, `completed_h_audit.json`, `check_additional_lineage.py`, `additional_lineage.json`, source/output hash manifest and document-integrity report in the v0.6 companion. The checker uses NumPy and the standard library and does not import or execute experiment modules. It reproduces the metric/point-policy/accuracy-bootstrap/paired-contrast and parity arithmetic stated in §3.2, preserves source-reported policy intervals, checks hashes and unchanged recovered fits, and labels additional K/condition/reliability cross-mode aggregations separately. No training, Qwen inference, new GPU timing, annotation adjudication, original notebook alteration or native-service test was performed. Final labels are used only as the archived targets for already-completed evaluation. E9 and V2 support required-study closure, not universal scientific or deployment acceptance.

**Version 0.6 editing lineage.** Base whitepaper v0.5.2 SHA-256 `38353636f6c100fb2a4d133ba6e663af9a84723bbccfd27a9594a34e53654dd4`; base roadmap SHA-256 `6823105da8d38d548936c888f5a58dde156a3f6fdb50775da6d10d3032558ed5`. Historical B–G and H-development numeric tables are retained in the whitepaper. The roadmap intentionally replaces its detailed H-development tables with a concise completed-study summary and links to §13.1. That is a documentation consolidation, not deletion or alteration of the recorded experiment. The 16-group review-to-work/test map and all open 2I/2J/Track S/P2 requirements remain.

**H primitive dataset identities.** The saved preparation resolves BoolQ `google/boolq` to `35b264d03638db9f4ce671b711558bf7ff0f80d5` and `SetFit/sst5` to `e51bdcd8cd3a30da231967c1a249ba59361279a3`. BoolQ's final source partition is validation; SST-5's is test. One BoolQ validation row is removed by the explicit length rule (3,270 before, 3,269 after), while the recorded SST-5 counts are unchanged by that filter. These are dataset/preparation identities and eligibility counts, not final prediction results or an independent review of the annotations. The H criteria SHA-256 is `311a0e1db256686c246f4efb9ce352e8e8b94864dc0ca54b9077793a10e07646`; independent review remains not performed. [E8]

**R1: Prior whitepaper correction memo.** `OpenKind_Whitepaper_v0.2_Review.md`. Identifies the probability-versus-argmax correction, segmented-tokenization correction, and refitted-constant attribution clarification applied in Sections 5.2, 8.1, and 5.3. The file is retained in the v0.3 evidence bundle.

**R2: Prior Phase 2G saved-result review.** `OpenKind_Phase2G_Result_Review.md` and `OpenKind_Phase2G_Review_Audit.json`. Preserve the independent distribution/policy reconstruction, source-label/omission checks, context aggregation, ambiguity case, and scalar cache replay scope. The v0.3 checker independently rechecks its stated arithmetic subset; neither review claims new model inference or label adjudication.

**Pinned datasets.** MultiNLI revision `da70db2af9d09693783c3320c4249840212ee221`; Banking77 resolved revision `90d4e2ee5521c04fc1488f065b8b083658768c57`. Banking CSV hashes retained in the results are train `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` and test `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`. These identify downloaded data, not the contents of Qwen’s pretraining corpus.

**Phase 2G dataset extension.** The pinned CLINC `oos-eval` revision is `828f8093932c8fe6ca7936c3d2e52903b1c523de`. Recorded SHA-256 hashes are `data/data_full.json`: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`; `data/domains.json`: `b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a`; and `LICENSE`: `e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76`. G’s prior-exclusion manifest SHA-256 is `f3549213c79dc4530dd8d04ecf61e5a6f08b8ba3e470a3a00d8296bc7bc5b42a`; its fresh panel hash is `367d31772bdf2b9630766272f3b7252131b7caf5f7e2f692da8d4ce4d3a5ad7d`. These are recorded download/selection identities, not a new corpus or annotation audit. [E7]


**S1: User-supplied discussion.** “Open Source Inference Plan,” [shared conversation](https://chatgpt.com/share/6aadc0b1-db54-83ea-856a-efc569000843). Title resolved; conversation body was not available to the web reader.

**S2: User-supplied foundation discussion.** “Jev Architecture Overview,” [shared conversation](https://chatgpt.com/share/6aadc0ec-8ee8-83ea-9343-b16fb73dbdc3). Title resolved; conversation body was not available to the web reader. Earlier Library reports supplied secondary context, not replacement evidence for measured results.

**S3: Current refocus discussion.** “Review Qwen Architecture,” [shared conversation](https://chatgpt.com/share/6aae15d6-c970-83ea-abbf-156c6dc734dc). The live page exposed its title but not readable conversation text. R3 recovers the main architecture-review response. Later R4T/Laya recommendations were available through retrieved conversation history, not a complete independently readable transcript; v0.4 used them as proposed direction and checked the narrow external descriptions against P18–P20; those historical descriptions are retained here without a new external review. It does not claim verbatim access to the entire shared conversation or repeat unverified third-party benchmark/provenance allegations.

**RC: Version 0.5.2 review-to-work mapping.** `OpenKind_Review_Followup_Traceability.md` and the preserved `sources/Recovered_Architecture_Review.md` capture the recovered main review and the follow-up recommendations retained in §11.7. The main review SHA-256 is `27b1ab204729db4d490ad8acd72aa563cd2aa1c330ccf8596acc33680405e129`. The matrix is a source-based future-work/test cross-reference, not a complete transcript, a new experiment or an outside-fact audit.

**RP2: Restored conditional backlog.** The previous roadmap v0.4 revision package retains explicit P2.1–P2.3 tasks; v0.5.1 summarized those topics without their standalone checkboxes. The synchronized roadmap restores the original conditional section and maps it to §§11.7 and 13.3–13.5. The exact restored excerpt is included in the revision package.

**HR: Historical recovery-notebook deliverable.** `OpenKind_Phase2H_Recovery_Notes.md` records the previously delivered `2h.1.2` Finish notebook, original-notebook repair and local/saved-artifact checks. It remains a repair-provenance source, not the final evaluation authority. E9 supplies the completed continuation and V2 its read-only saved-output audit. This documentation update does not run or modify either notebook.

**E10: First Phase 2I/2J workbench checkpoint, blocked overall.** Study `2ij_reviewed_multiquestion_v1`, version `2ij.1.0`, saved around 16:10 UTC on 19 September 2026. [Compact summary](https://drive.google.com/file/d/11X6HKs18bjjgGTDKbQwTe8PWK4nLXwfh/view), [report](https://drive.google.com/file/d/15rtftQ3WwGY_1WU1smLfAM0kXVfk5QAX/view), [report archive](https://drive.google.com/file/d/1bIyJfim0n-XsphCIEZu11cs8NEb90kO8/view), [mechanics result](https://drive.google.com/file/d/178mr-qS7CARw6qgfxGqZpghOvfjohmf_/view), and [workbench notebook](https://colab.research.google.com/drive/1BeeurQlJf0BJ_fV0lA-_7aZbXDhcgLQn). Archive SHA-256: `a12c0e4c00f24bfab3e1fbdcd99991cafad5f478753e995e44be58640a162143`. The archive has 29 files (1,180,633 uncompressed bytes), including saved stage/gate reports, source snapshots, CPU-test log, mechanics job/runtime/results and historical-reference audit. It contains no registered study, new fitted profile or semantic final result. Its quoted “unrelated” test retains the original name; §14.3 records the actual duplicate fixture.

**E10 snapshot boundary.** [LATEST pointer](https://drive.google.com/file/d/1392d9KjhWkxd-nCFww1-9q1dr6yeLVcS/view) identifies `snapshot-1789834208243222126`; its [completion manifest](https://drive.google.com/file/d/10oN9cUW3P0dafKi9qQNAPoOfJum8FyGp/view) SHA-256 is `328b99c2d09f506a959b83b86dbfa85dc948f6c5fd09fbde8575c2715391f860`. V3 checks the 18 manifest entries overlapping the compact report. The two additional entries, `mechanics/features.sqlite.backup` and `validation/cpu_tests.xml`, were not materialized or validated. Snapshot/report export does not imply approved semantic study completion or demonstrate future recovery.

**I0: Unsigned 2IJ intake metadata, separate from the report archive.** Under `Google Drive / Colab Notebooks / OpenKind_Phase2IJ_review`: [protocol](https://drive.google.com/file/d/1zDdBv1JtGFTzMKYZshx-T8olFj5dVM9c/view), [unsigned review](https://drive.google.com/file/d/1_eIzpj9722xYDLBOmBb5lG1j0fng-rqO/view), [refreshed review template](https://drive.google.com/file/d/1rk2jLwBnI_nsgJ9xXwqylOfTmC5GJsvk/view), and [review instructions](https://drive.google.com/file/d/1En_2qNQBx1Jo21a8OrAFmwHv0bH2KcZn/view). The metadata snapshots have blank identities/false approval, a null target and five null quality/resource bounds. V3 checks the current canonical protocol-body digest `30239d9a2699b9ebf1bde047a9c9b453b64c01244079801d56512364c74c4791` against the refreshed template and records the stale unsigned review. Cases are counted only from the hash manifest; this revision does not open or adjudicate `cases.jsonl` or final annotations. Later edits are not implicitly included in this snapshot.

**V3: Version 0.6.1 saved-record and source-coverage checks.** `audit_2ij_checkpoint.py`, `evidence/saved_artifact_audit.json`, source hashes, exact document diffs and integrity checks in the companion. All 34 defined record checks pass, while the experiment remains blocked. The checks cover status agreement, source identities, saved feature-threshold arithmetic, inventory counts, 18 snapshot-file hashes, review-manifest consistency and static identification of the duplicate-question fixture. They do not reproduce hidden vectors, run the 48 source-reported tests, rerun H, train/evaluate models, resolve public checkpoint revisions, review labels, repair notebooks or change persistent sources. The historical `OpenKind_Phase2IJ_Checkpoint_Readout.md` includes source excerpts and the practical next steps.

**Version 0.6.1 editing lineage.** The editing authorities are the latest delivered v0.6 whitepaper and roadmap, whose hashes are recorded in the companion manifest. The older v0.3 paper and pre-H roadmap surfaced in the conversation are historical, not replacement authorities. All prior whitepaper numerical tables are retained verbatim; only current-status/priority text and the explicitly new checkpoint evidence are added or updated. The v0.6 completed-H closure and all 16 review-to-work groups remain intact. No source Drive file is modified.



**E11: Completed exploratory 2I/2J model-selection screen and selected-model export.** Study `2ij_model_selection_screen_v2`, workbench `2ij.2.0`; final workflow `completed_requested_scope`. The saved result set contains 13 completed fit jobs, 31 completed final profiles, bounded multi-question scaling, `MODEL_DECISION.json`, and an A100-completed reference model bundle. Selected profile: `a047d6802c3f06f085b8`, Qwen/Qwen3.5-4B-Base, state-first, score-summary rejection. The selected final panel contains 320 episodes from 56 messages and records 95.0% accuracy / 0.13006 NLL. Bundle SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`; fresh reload parity passed with max probability delta `3.6673555e-6` and zero selected-ID changes. Scope remains exploratory; independent review, production acceptance limits and Rust/Metal parity are not established.

**E12: Phase 3A Python BranchableState and batched-Q systems reference.** Run `20260920T024056Z`, schema `openkind-phase3a-summary/v1`, source/result path `Google Drive / Colab Notebooks / OpenKind_Phase3A_results / 20260920T024056Z`. The run uses selected E11 profile `a047d6802c3f06f085b8` and bundle SHA-256 `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`; it records no model change, training or reselection. Saved headline gates: three semantic smoke cases, semantic batched parity all pass, high-K systems parity all pass, same-process repeatability maximum probability delta 0.0 and zero argmax changes. The benchmark rows retain repeated-full, nested-sequential, batched-cold and batched-warm timings and peak allocations over semantic Q=1/Q=4 cases and a mechanics grid through L=1024/Q=16. The corrected notebook uses complete-cache reindexing for hybrid fan-out/select after the generic Transformers repeat helper proved unsupported for Qwen3.5 linear-attention cache layers. `release_quality_claim` and `rust_metal_parity_claim` remain false. This whitepaper derives ratios only from the saved timing rows; it does not rerun Qwen.

**E13: Phase 3B Python Qwen3.5 backbone-parity reference.** Run `20260920T152206Z` uses schema `openkind-phase3b-backbone-parity-summary/v1`. Its checked-in path is `research/14_phase3b_backbone_parity_results`. The run uses E11 profile `a047d6802c3f06f085b8`, bundle SHA-256 `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`, and base revision `1001bb4d826a52d1f399e183466143f4da7b741b`. It records no training, model selection, or model modification.

The E13 export contains 4 exact token records and 47 FP32 vectors. Those vectors cover 34 ordered trace stages, 10 full-sequence candidate features, and 3 continuation points. Maximum fresh-feature delta versus the earlier bundle is `4.9591064453125e-05`. Cached continuation versus fresh full sequence is `1.9073486328125e-05`. Both are Python self-consistency diagnostics under the notebook's `1e-4` guard. The saved summary explicitly sets `rust_parity_claim=false` and `metal_parity_claim=false`.

**E14: Phase 4A.0 corpus/cache lock and Phase 4A.1 original B1 non-final result.** Run `20260921T013558Z`; [run folder](https://drive.google.com/drive/folders/1Sh7NQdO68AP6bcdY3fU4iIrP4B4c_7f0). Principal artifacts: [declared contract](https://drive.google.com/file/d/1bbrz2MSLz3Ryf4Qqd2XUZCLOO4hSnphE/view), [corpus manifest](https://drive.google.com/file/d/1N79DfdZPuUsdGzuuRckwtdg6_LdSHfst/view), [corpus lock](https://drive.google.com/file/d/1gCzZWaNgdCVrRAx4UCRzeghchhV80OSP/view), [feature manifest](https://drive.google.com/file/d/1LoqWYzvFEu-BSxXpPmgcGOFxur_etmdy/view), [training report](https://drive.google.com/file/d/1teZKYQjeKzpSxV2okdPVvHPRXWEyddUk/view), [non-final evaluation](https://drive.google.com/file/d/1yXpkhMmGzc1yFOu__3MzJPNIWWUFqK0H/view), and [model-selection lock](https://drive.google.com/file/d/1kROx4vzWWECcR9bR0kgt50YFVLuBkBv0/view). The corpus manifest records 2,192 states, 15,368 questions, 25,687 options, 21,894 evidence rows, and 18 criteria; corpus-manifest SHA-256 `1aad04ca7d37ea8be640a51878857c6d2c501d635e913eca12234cb548f8399d`. The feature cache binds profile `a047d6802c3f06f085b8`, bundle `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`, and base revision `1001bb4d826a52d1f399e183466143f4da7b741b`. The original B1 selected checkpoint SHA-256 is `9b19138268b5de2bb48b461337e88b9de6492a7bc52eff6489525e09eb63d940`; non-final report SHA-256 `a003c4648a1c28aa4a900b680b6c9ca3318c039a900cd3d106701b9be03bbd57`. Status is `exploratory_locked_gates_unset`; final remains unopened.

**E15: Phase 4A.2 matched StateQuery comparison.** Workbench `4a.2.0`, base run `20260921T013558Z`; [comparison root](https://drive.google.com/drive/folders/1LMeEV_5_trzVBsPFYcZKhM0cN1XnWyDl) and [run folder](https://drive.google.com/drive/folders/1CZWhbgcu3H0RwEFQgPYBFyIX0MoRew3Y). The completed [B0 folder](https://drive.google.com/drive/folders/1Cx8eNZbz6djc7xkUa3g2jMmkBatC-S4e) contains its experiment contract, six-epoch early-stopped training report, non-final evaluation, selected checkpoint, and `comparison_only_final_unavailable` result lock. Selected epoch 2 checkpoint SHA-256 is `82f50370b789941b0cfdee55b6cfc786ed5edff55bf743a93fc329e19950f714`. The [comparison snapshot](https://drive.google.com/file/d/15q90M8aLlKRDfj8Z7O9Z1YCvILwvMvN7/view) was created `2026-09-21T20:14:09Z` and contains the historical reference, original B1, and B0 non-final tables reproduced in §18.4. The [balanced-B1 folder](https://drive.google.com/drive/folders/1Oz-tSCBXDJZn2Ssatqhqwhq21oF9rt6X) contained only its contract, checkpoints, [LATEST](https://drive.google.com/file/d/19XxGI4Dk2q5O13m2H6O8ISx9RJT8Fqyi/view), and [BEST](https://drive.google.com/file/d/1G2jlNOYujMNlDvnQmFSgert3a3hoG-jP/view) at the documentation cutoff. Both pointers identified epoch 11 and were last modified `2026-09-21T21:39:24Z`; no completed training report, non-final evaluation, or result lock was present. Its values are provisional development metrics, not part of the completed comparison snapshot.

**E15 amendment: completed Phase 4A.2 comparison sweep.** The Phase 4A.2 root now contains completed, locked B0, balanced-B1, factorized-B2, refined-B2R, and `b2q_weight8_s17` arms. The latest retained comparison snapshot before Phase 4B.2 reports all five Phase 4A.2 arms alongside the historical reference and original B1. Each completed arm has an experiment contract, selected checkpoint, training report, non-final evaluation, stored predictions, and a `comparison_only_final_unavailable` lock with `final_opened=false`. This amendment supersedes only E15's earlier progress status; it does not rewrite the timestamped checkpoint history.

**E16: Phase 4B.2 QASPER cap-12 result and prediction validation.** Workbench `4b.2.0`, base run `20260921T013558Z`, run label `b2q_weight12_s17`; [run folder](https://drive.google.com/drive/folders/1IAC0FrXyJGJieonUQZX0_pFNA8D3Og8P) and [comparison snapshot](https://drive.google.com/file/d/1TTS_E65KWNR_wZZy5auk2IZyxOosQfSx/view). Experiment SHA-256 is `ac1c90aa2592b5536d4351686fb559e1c6733885d0f902230a8b57155cc6bfdb`. Selected epoch 11 checkpoint SHA-256 is `722bc8c4e24654c71c6e8d9be7e421b506d76fd1c5be50081a237b0e9551709d`; non-final evaluation SHA-256 is `980c4401a1b76738cc8ebaf873cc7188c41b6e9cd824c5a688a8e18826d4f44c`. The report records an effective QASPER semantic-none weight of `8.560185185185185`, `eligible=false`, `stop_reason=max_epochs`, `final_split_available=false`, and `final_opened=false`. Read-only validation reproduces the evaluation hash and all three prediction-file hashes; recomputes every reported calibration-gate metric exactly; and checks 3,837 stored prediction rows for duplicates, nulls, target/label mismatches, probability errors, and cross-partition state/question overlap, finding none. The additional threshold/AUC diagnostic in §18.8 is derived from these saved non-final rows and does not modify the locked selection or result.

**E17: Phase 4B.2.1 source/class-stratified applicability.** Workbench `4b.2.1`, base run `20260921T013558Z`, run `b2_applicability_stratified_s17`; [run folder](https://drive.google.com/drive/folders/1Pwx_yNhld4ELrRbJYkmuPCyXUt1ZGCUg). Experiment SHA-256 is `4313087778356312ac8646486ac8c45731885595b88e7236edcefa5c985d703e`; selected epoch 8 checkpoint SHA-256 is `87c3b3a4069b64e4a150ad6e0812ce2d953346eb17999865770df508269ed871`; non-final evaluation SHA-256 is `ec4e62daa2508bbdec4d413625b53560290b36c74975df238fe2d1dfdd648f58`. The run trains 35,843,335 parameters with answerable-only ranking and source/class-equalized applicability BCE. It records QASPER operating recall of 0.2564 on policy development and 0.3111 on the calibration gate; gate policy cost is 0.11675 with 7 wrong among 32 accepted. Status is `comparison_only_final_unavailable`; final is false.

**E18: Phase 4B.3 pairwise applicability.** Workbench `4b.3.0`, base run `20260921T013558Z`, run `b2_app_pairwise_w025_s17`; [run folder](https://drive.google.com/drive/folders/1jYwC5VoYgbutYsM6C1F-6OeOHEFR06xY). Experiment SHA-256 is `38637f8b95ef0f432e489b2f01d5c4544dc01f3834293e520a1569709dbe9093`; selected epoch 9 checkpoint SHA-256 is `dba87d60892ea5a2380b2c9f9d4e49addbd5895b8a50a2d9e44d2b2c66c1e077`; non-final evaluation SHA-256 is `5e8794fdc736a3fe3114d9f6dc84b6203bd96af50b0f0254c1cef932a927f0fc`. The 0.25-weight, zero-margin pairwise term clears development applicability gates, but calibration-gate false-none is 0.2111 for ContractNLI and 0.2110 for QASPER, and gate policy cost is 0.10221. Status is `comparison_only_final_unavailable`; final is false.

**E19: Phase 4C applicability-head-only continuation.** Workbench `4c.0.0`, base run `20260921T013558Z`, run `b2_app_headonly_guard18_s17`; [run folder](https://drive.google.com/drive/folders/1Tqc_nwC1aWd1-Br6fdGJM9Gs6NHWVGjW). Experiment SHA-256 is `396748e3f495403dd5cdccf1a6708ddc3de815e2f54fcf75e0b92525a27037ef`; the retained epoch-0 parent uses checkpoint SHA-256 `87c3b3a4069b64e4a150ad6e0812ce2d953346eb17999865770df508269ed871`; non-final evaluation SHA-256 is `f8b49ec9de2482706188f1721171d3ce4574f82649803858581bf9cbf4ab4dff`. The 1,771,009-parameter head-only continuation runs three child epochs. Best-child NLL/Brier worsens by 2.77%/2.10%, so epoch 0 is retained. Status is `comparison_only_final_unavailable`; final is false.

**E20: Phase 4D evidence-aware residual.** Workbench `4d.0.0`, base run `20260921T013558Z`, run `b2_evidence_residual_s17`; [run folder](https://drive.google.com/drive/folders/1O-FLsm4zgfMvwBV4mE73JAhVQ4k8oa_W). Experiment SHA-256 is `70ac547c7a257d978cbc241f049fb2dbbc912b0d5801ca28080820d1e183d4c5`; retained epoch-0 checkpoint SHA-256 is `314d3a45009a51159bb74bebed9806af76d1df103a14eca9a38160d0dab0d972`; non-final evaluation SHA-256 is `9cc328dd3deeafb175aabaf18ba2daaf9914863b95879368a61416582c96680b`. A zero-initialized 19→64→1 residual adds 1,345 trainable parameters. The best trained child gains one QASPER true positive but worsens source-macro NLL/Brier by 6.08%/6.60%; all eight epochs are ineligible. The run validates strict JSON with unavailable values represented as `null`. Status is `nonfinal_failed_final_unavailable`; final is false.

**E21: Phase 4E QASPER blinded error audit.** Workbench `4e.0.0`, base run `20260921T013558Z`, run `qasper_error_audit_s17`; [run folder](../../research/21_phase4e_qasper_audit_results/20260921T013558Z/qasper_error_audit_s17/). Audit contract SHA-256 `40c6cacea0f197a1ab5ac76b7f5946ae9ad13a76a1a3c281e305e71ce8117b75`. Exported 150 blinded review rows across policy-development and calibration-gate errors and controls with cryptographically locked ground-truth key (`AUDIT_KEY.parquet`). Initial primary review revealed a high ambiguity rate (52.7%) due to missing full paper state text, requiring a repaired audit packet before adjudication. Status is `superseded_by_phase4e_a2`; final is false.

**E22: Phase 4E-A2 QASPER repaired audit and adjudication packet.** Workbench `4e.a2.0`, base run `20260921T013558Z`, run `qasper_error_audit_repair_s17`; [run folder](../../research/22_phase4e_a2_qasper_audit_repair_results/20260921T013558Z/qasper_error_audit_repair_s17/). Audit experiment SHA-256 `89d0e283a2b743efcb95de3f73189c6fb0157620c5f6252beea149a44d1ecefd`; completed review SHA-256 `ca5e1ccbd31e575fce4ec97b07f93bc7c5e215ed6a796902f8eb2a1821eed4d8`; analysis SHA-256 `858d1b7b08d2e3e9969f7de4bdc1780cbaf83a831c379b9529d4acfe2a84b25f`; adjudication packet SHA-256 `2d6b3be4dfa532de7c0c320d4e84b67d2be571ff4ddcc7f886b22f437e88edeb`. Restored complete state text (median 24,812 chars, 100% hash match). Blinded primary review across 150 rows yielded 99/149 (66.4%) decided agreement and emitted 51 adjudication rows. Disagreement is highly asymmetric (44 challenged semantic-none, 6 representation/serialization losses). Result lock establishes `repaired_audit_primary_review_complete_adjudication_required`; Phase 4E-B is not automatically authorized (`phase4e_b_automatically_authorized: false`); final is false.

**E23: Phase 4E-A3 51-row adjudication follow-up.** [Decision record](../../research/23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/PHASE4E_A3_DECISION_RECORD.md), [corrected report](../../research/23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/AUDIT_ADJUDICATION_REPORT_V2.json), and [lock](../../research/23_phase4e_a3_qasper_followup_results/20260921T013558Z/qasper_followup_cpu_s17_v1/PHASE4E_A3_LOCK.json). The corrected report references independent-adjudication SHA-256 `aaec70a8ae64deebc45c33945748480eafd472c0b00b467422f15a36e5278701` and identifies six same-assistant follow-ups as non-independent. The resulting 51-row distribution is 33 answerable, 13 semantic-none, five ambiguous. No source data or final split changed.

**E24: Phase 4E-A4 preflight and bounded dataset alignment.** The
[preliminary record](../../research/24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/PHASE4E_A4_PRELIMINARY_RECORD.md)
and [preflight](../../research/24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/PHASE4E_A4_PREFLIGHT.json)
confirm 12 policy-development spans across seven questions, one diagnostic gate
span, and three unresolved quarantines. The later
[alignment QA](../../research/24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/QASPER_SOURCE_ALIGNMENT_QA.json)
and [row alignment](../../research/24_phase4e_a4_qasper_source_alignment_results/20260921T013558Z/qasper_source_alignment_cpu_s17_v1/QASPER_SOURCE_ROW_ALIGNMENT.csv)
record four aligned policy papers and 40 caption candidates. QA SHA-256:
`cf0706e5ee8d7b773727ad3a6f10520c3c0b5613087d494dd4dc8550a654a74b`.
Row-table SHA-256:
`2c1c2076039b4250b49f5bae2c01b4fb5c2597aed8e4c017c3ccf2f192b89311`.
The four row counts and 40-caption inventory agree with the QA. PDF equivalence,
table-value recovery, serializer repair, and review of corrections remain open.
The saved flags record no test download, gold mutation, final opening, or
automatic training authorization. No upstream dataset was downloaded again
for this documentation check.

**E25: Later Phase 4E-A 27-case disposition.** [Disposition lock](https://drive.google.com/file/d/1vbrJDGBxPx969eWznaaIXKGU9J4YgLYM/view?usp=drivesdk), [summary](https://drive.google.com/file/d/1OWV1UE6lOkFgxrtsu5u-MfABQxZMEsoR/view?usp=drivesdk), and [NarrativeQA representation candidate](https://drive.google.com/file/d/1Yj2e6iYx2lspPqEco19fw46XC-UpwndC/view?usp=drivesdk), created 24 September 2026. Assistant-conducted, user-authorized evidence review freezes 27 dispositions and the 77-tie diagnostic policy. Case CSV SHA-256 `eea9b77154f4a94eed356cbfd21b2d613006a8fb2c88c376b382bc14dab02d5d`; tie-policy SHA-256 `8aea66a1e80cc4b1cb8708aab2099f053c4c376f3b85d3fd368be56b3c0ae4bd`. The lock records `phase4e_b_authorized: false`, `final_opened: false`, and no mutation of gold, labels, states, A4 ledger, or thresholds. This is distinct from E23's 51-row population.

**E26: Exploratory 4E-B.0 prefill representation probe.** [Experiment contract](https://drive.google.com/file/d/1wrzxdmV7Lxdfc6vpSle4qevv-cFLmaNa/view?usp=drivesdk) and [non-final evaluation](https://drive.google.com/file/d/1VX6BaBBJ8g9u4w5vcAya0a_ZVX1VuQYe/view?usp=drivesdk), run `prefill_representation_probe_s17_v1`, seed 17. Fixed arms: `root_only`, `root_question`, `root_question_candidate`, and the read-only Phase 4D reference. All three probe arms fail `calibration_gate_pass`, accept zero requests, and leave final unopened. The result folder lacks a separate `NONFINAL_RESULT_LOCK.json`; no model promotion or new-arm authorization follows.

**E27: Exploratory 4E-B.1 candidate-conditioned option-logit audit.** [Notebook](https://drive.google.com/file/d/1vFKFXeSAZvkkYbIve0OS1P3zBxBNLXtY/view), [contract](https://drive.google.com/file/d/1mQqZ2K63MBP294S3H5zI6xmXmlUbyey6/view), [325 non-final rows](https://drive.google.com/file/d/1_hQY3B2twDgRnLvakQ-pmvQk54fDmkJy/view), [evaluation](https://drive.google.com/file/d/1DApZhZtxhgNV6b0JDd1S38QNLC84GAf1/view), and [result lock](https://drive.google.com/file/d/1jafTJgEUeN4iGh5f7oKpMmcERcnfd7Lw/view), run `candidate_option_logit_gate16_s17_v2`. Contract SHA-256 `01899f640d7692d639387d779eb2e8221b7d27e0bb71cb62acf62b655d597251`, row SHA-256 `3fff196e3e6507f536135abf5106bf1ab0c8ed36e882e1f8caf6308a40ea2447`, and evaluation SHA-256 `53cff1590988e66d43de7d779d47a6c62cf271a27ad187f8fe8367c91a6e30a6` were independently verified with row-level metric recomputation. The pinned Base model scores two ContractNLI candidates and one QASPER candidate plus a `Z` semantic-none action. ContractNLI answerable-only candidate accuracy is 130/148 but none recall is 0/124; QASPER none recall is 2/8. One cached/full-prompt check per source passes. This is an exploratory fixed-sample comparison, not a release-quality or throughput gate; final is false.

**E28: Exploratory 4E-B.2 candidate-ranking and rejection sweep.** [Notebook](https://colab.research.google.com/drive/1LKpPR5zYFZHDEnWYX6898Hhvwkoqh0I3), [contract](https://drive.google.com/file/d/1Foj3pXRBnk7Im0daheP8RTCc9k3nEnwV/view), [development selection](https://drive.google.com/file/d/1uSWSPHxd1Xp6sP1oCOdb3H1UhxoFj8eF/view), [741 non-final rows](https://drive.google.com/file/d/1oZDuYud8WsySg8k4evEwSaBXzeNmSkH3/view), [evaluation](https://drive.google.com/file/d/1uuo-JxYYCPgvupqdJzPvv3ByXZxH72KL/view), and [result lock](https://drive.google.com/file/d/1hIwuvAFm2b5Kqep-1UYpNUwCxVa48obK/view), run `candidate_detection_ranking_sweep_s17_n12_v1`. Contract SHA-256 `897291b40209dbaa92e671f15119d7c9ce1f1a9f9469383f2ad706263559f8ae`, selection SHA-256 `d451329afc008befea44ff3d67e63ab4ec5a0dbd43a469db1936b410e491402a`, row SHA-256 `48755e082e4442d8f099434bbdc28a4bf814ba93860db2c51777e989960e5b83`, and evaluation SHA-256 `495bb051c99ef3213c03f817a541455ae4360c19701a85873e7821f80e59191a` were independently verified along with row coverage and selected-arm counts. ContractNLI selected ranking is 94/108 answerable, but false-none is 46/108; QASPER selected none recall is 0/5 because the threshold grid starts above every selected detector probability on development and gate. Six cached/full checks across prompt families pass. The gate partition was already exposed, the 72 state parts were not individually downloaded here, and final remains closed.

**RUST1: Rust Phase 3.1/3.2 implementation checkpoint.** The repository adds `ModelExecutionProfile` in `openkind-engine`. `openkind-backends` adds the selected `qwen35` profile, head, tokenizer, and Phase 3B reference loader. Offline tests validate bundle and head hashes. They replay four original golden-feature cases and reproduce all four Phase 3B token records exactly.

The tests also validate the 47-vector contract and replay 10 Phase 3B candidate features through the selected head. The historical implementation checkpoint used Rust 1.75, but the current workspace requires Rust 1.88 because the tonic 0.14.6 service stack declares that MSRV. The current CI floor is verified with locked, all-features workspace tests. The recorded working session also passes formatting, strict workspace Clippy, workspace tests, schema regeneration/no-diff, and `git diff --check`. No native Qwen weights, Candle/Metal execution, `BranchableState`, server registration, or Jev semantic-none mapping are claimed.

**RUST2: Rust Phase 3.3 input-embedding checkpoint.** `Qwen35Embedding` verifies the immutable base checkpoint's config and safetensors index, then checks the embedding shard size and SHA-256 before reading token rows. It validates the pinned BF16 `[248320, 2560]` tensor layout, widens BF16 values exactly to FP32, rejects invalid IDs and non-finite values, and preserves token order. An offline row fixture derived from token ID 25 at base revision `1001bb4d826a52d1f399e183466143f4da7b741b` has SHA-256 `84ce40703c960b305ad72adbdc0f6fc5b8df6fd279e5b18d5dd194e4764377c7`; its Rust FP32 output matches E13 `diagnostic.embedding` exactly. A diagnostic probe applies the same comparison to the complete locally downloaded 5.3 GB embedding shard. This is input-embedding parity only, not decoder, full-backbone, Metal, or service parity.

**RUST3: Rust Phase 3.3 CPU backbone and continuation checkpoint.** `Qwen35Backbone` pins Candle `0.8.0`, verifies both immutable BF16 shards from base revision `1001bb4d826a52d1f399e183466143f4da7b741b`, and performs FP32 CPU execution for embedding, 24 DeltaNet blocks, 8 full-attention blocks, and final RMSNorm. On an Apple M4 Max (`Mac16,5`, 36 GiB, macOS 26.6.2), the complete 34-stage diagnostic has exact embedding and the final-norm values reported in §17. All 10 candidate sequences preserve four exported distributions with maximum probability delta `4.5869e-06`, zero argmax changes, and zero policy changes. The correctness run took `399.75 s`; macOS `/usr/bin/time -l` reported `7.77 GB` maximum resident accounting including mapped model shards and a `680 MB` peak memory footprint. The internal `BackboneState` carries attention KV, recurrent, convolution, and position state; its exported root byte count is exact, its source remains immutable, and cached candidate output equals native full-sequence output exactly. This is CPU fixture parity, not Metal, backend-neutral branching, production throughput, service integration, or release promotion.

**RUST4: Rust Phase 3.4 branch-state checkpoint.** The runtime-owned `BranchableState` and `BranchBatch` contracts bind the complete Qwen hybrid continuation state, including attention KV, DeltaNet recurrent state, convolution state, and position. The named-Mac branch probe verifies pinned identity, exact tensor-payload accounting, immutable-root fork, independent lane advancement, gather identity, and exact cached-state versus full-state equality.

**RUST5: Rust Phase 3.5 sequential nested checkpoint.** The checkpoint-gated nested probe executes all four Phase 3B questions and 10 candidates from one immutable prefill per fixture case. Decision, isolation, oracle, position, replay, and root-immutability gates pass; nested and repeated-full probabilities agree exactly while the frozen reference probability delta remains `4.5869e-06`.

**RUST6: Rust Phase 3.6/3.7 batched Q/K checkpoint.** Breadth-first question and candidate fan-out over `fork_batch` is state-identical to the sequential baseline for every frozen fixture. This proves state topology and lane isolation, not compute-vectorized model forward.

**RUST7: Rust Phase 3.8 adaptive-scheduler checkpoint.** The initial named-Mac benchmark measures five warm CPU workloads, establishes feature parity across strategies, and supplies the `2.52` lowest measured token-work ratio used by the scheduler. Cold-start, cache-warmth, memory-pressure, and vectorized suffix packing remain open.

**RUST8: Rust Phase 3.9a and direct-service contract checkpoint.** Offline K=32/64/128/255 estimator/admission stress, typed scheduling/content fingerprints, tenant/TTL/byte-bounded cache behavior, atomic pinned-state snapshots, cancellation-safe capacity ownership, direct native engine registration, explicit semantic none, and entropy-derived confidence are implemented. Model-backed high-K, fresh-process replay, and service load/soak remain open.

**RUST9: Commit-stamped v0.8.0 verification.** [`../verification/2026-09-20-v0.8.0-35c481a.md`](../verification/2026-09-20-v0.8.0-35c481a.md) records the clean named-Mac run for subject commit `35c481a6e95a`. Workspace gates, zero-diff schema regeneration, scheduler stress, both checkpoint-shard digests, full/branch/nested/batched native parity, commit hygiene, and the canonical warm-process benchmark pass. The five-workload rerun measures `NestedSequential` at 1.294×–2.018× faster than repeated-full, `NestedBatched` within 2.9% of sequential, Q-amortization at 2.815 repeated versus 2.170/2.233 shared, and post-benchmark peak resident memory at 10.968 GiB. This does not close the RUST8 open gates or establish release promotion.

**RUST10: Named-Mac native follow-up campaign.** [`../verification/2026-09-20-v0.8.0-native-follow-up-working-tree.md`](../verification/2026-09-20-v0.8.0-native-follow-up-working-tree.md) records the dirty-tree run anchored at `9d086107bb017bf721d815bb5bcb8ba516ce0e6e`. Rust 1.98.1 workspace tests, all-features tests, formatting, strict Clippy, and diff hygiene pass. Rust 1.88.0 is verified as the workspace MSRV floor and added to CI. Model-backed K=32/64/128/255 stress passes bounded completion, root immutability, and memory stability; two fresh processes pass structural persistence replay; and a real native daemon request, overload admission, cancellation/recovery, clean shutdown, and 20 health probes pass. A separate 22 September follow-up ([verification record](../verification/phase3.10-2026-09-22/README.md)) closes full restored candidate-feature/head/probability/argmax/policy replay on the named CPU host: 8 candidate features and complete probability vectors match the independent full path exactly, with zero argmax or policy changes. Practical high-K latency, production load/soak, Metal, and release promotion remain open.

**RUST11: Named-machine native CPU service gate.** The
[provenance and results](../verification/native-service-gate/2026-09-22-rerun2/README.md)
record a release-mode daemon on the 36-GiB M4 Max, anchored at commit
`a5a752ab50efccba2eff0345fc5435c01248d41e` with dirty-tree source fingerprints
and executable identity. Lifecycle, queue-inclusive load/deadline/recovery,
memory, and 30-minute soak checks pass for that artifact. The fixture has no
reviewed labels. This is neither a clean-commit build nor model-quality,
accelerated-service, or release-promotion evidence.

**RUSTM1: Pinned-base MLX FP32 qualification and remaining promotion gates.**
[MLX.md](../MLX.md) owns the operating contract. The
[22 September follow-up](../verification/phase3m-2026-09-22/README.md)
records full/nested and unequal-length vectorized parity, forced daemon
execution, and bounded unified-memory admission/recovery on the named M4 Max
with MLX 0.32.2, Xcode 27.0, and Metal 32023.921. Its dirty-tree base revision
is `a5a752ab50efccba2eff0345fc5435c01248d41e`. BF16 fails the unchanged
probability gate. The slower packed kernel stays opt-in. Matched complete-request
vectorized/per-lane performance, accelerated-service load/soak, and clean-commit
promotion remain open. These observations are independent of CPU parity.

**RUSTM2: Rust Qwen3.5 flat-field and candidate-pooling diagnostics.** The
[flat-field record](../benchmarks/2026-09-27-python-flat-field/README.md)
contains paired fresh-process FP32 MLX Q2/K2 and Q8/K4 runs, source and
commands, raw timing/probability/memory JSON, checksums, and clean-commit
formal full/nested parity reports for `78b9e0cc4ed8e66fe04407b626eac29a358bfadd`.
The [candidate-pooling record](../benchmarks/2026-09-27-candidate-pooling/README.md)
contains the earlier stage replay and distinct full-request baseline. The
flat path is a Qwen3.5 Rust execution diagnostic with unchanged readout; the
external Qwen2.5 Python result is a separate model and baseline. Neither
diagnostic enters the service or automatic scheduler.

**PUB1: Public OpenKind state-first reference repository.** <https://huggingface.co/cowWhySo/OpenKind-Qwen3.5-4B-StateFirst>. Project-owned publication of the selected state-first integration line. The repository URL is a public identity/reference surface; experiment identity remains pinned by profile ID, base-model revision, renderer/head/rejection contracts and bundle hash. Publication does not establish TypeSafe RLCD reproduction, release-quality promotion, or Rust/Metal parity.

# Appendix B. Primary external references

Version 0.8.2 adds no new external-reference review. It applies the previously reviewed P23 method principles to the Phase 4E audit design and does not treat SemIf results as OpenKind measurements. Version 0.8.1 added the focused P23 review for Phase 4A benchmark-reporting context.

Version 0.6.1 adds no external-reference review. New checkpoint claims use E10/I0 and V3; the following public references retain their prior review dates.

Versions 0.5, 0.5.1 and 0.5.2 add no external-reference review. The following descriptions and their review dates are retained from v0.4; new H claims use E8 rather than outside literature.

Primary pages P1–P16 were checked for version 0.1 on 18 September 2026; their historical uses remain attributed to that review. P17 records the CLINC dataset identity used in G. For v0.4, focused checks revisit the TypeSafe introduction/Choice contract (P1/P2) and add Qwen3.5-2B, Laya, and R4T (P18–P20). This is not a full re-review of earlier references, a code/checkpoint audit, or an independent validation of vendor performance. Public model cards are unpinned descriptive sources here; an experimental comparison must resolve and record exact revisions before fitting.

**P1.** TypeSafe AI. *Introduction.* Shared state, typed questions, and documented parallel/isolation behavior. [Documentation](https://docs.typesafe.ai/introduction).

**P2.** TypeSafe AI. *Choice.* Option-set semantics and current cardinality limit. [Documentation](https://docs.typesafe.ai/primitives/choice).

**P3.** TypeSafe AI. *Score.* Described levels, distributions, and expected level number. [Documentation](https://docs.typesafe.ai/primitives/score).

**P4.** TypeSafe AI. *Noul.* Binary truth/yes probability and response semantics. [Documentation](https://docs.typesafe.ai/primitives/noul).

**P5.** TypeSafe AI. *Confidence.* Distribution-derived confidence and application thresholds. [Documentation](https://docs.typesafe.ai/confidence).

**P6.** Almeida, D. (15 September 2026). *Introducing System One Models & Jev.* Public architecture/sampler/RLCD claims and benchmark caveats. [Announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

**P7.** TypeSafe AI. *AI primer.* The documented goal of calibrated decision training. [Documentation](https://docs.typesafe.ai/introduction/machine-learning-primer).

**P8.** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* Proceedings of ICML, PMLR 70, 1321–1330. [Paper record](https://proceedings.mlr.press/v70/guo17a.html).

**P9.** Berdichevsky, R., Nahum-Gefen, S., & Ben Zaken, E. (2025). *SALSA: Single-pass Autoregressive LLM Structured Classification.* arXiv:2510.22691. [Paper record](https://arxiv.org/abs/2510.22691).

**P10.** Stepanov, I., et al. (2025). *GLiClass: Generalist Lightweight Model for Sequence Classification Tasks.* arXiv:2508.07662. [Paper record](https://arxiv.org/abs/2508.07662).

**P11.** Jaegle, A., et al. (2021; revised 2022). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* arXiv:2107.14795. [Paper record](https://arxiv.org/abs/2107.14795).

**P12.** AlexWortega. *openjev model card.* Community Qwen NLI and latent-head implementation; not independent confirmation of Jev equivalence. [Model card](https://huggingface.co/AlexWortega/openjev).

**P13.** monotykamary. *LFM2.5-2.6B-RLCD model card.* Explicitly inference-only parallel constrained decoding, with disclosed calibration and quality limits. [Model card](https://huggingface.co/monotykamary/LFM2.5-2.6B-RLCD).

**P14.** PyTorch. *Numerical accuracy.* Floating-point, batching, platform, and reduced-precision caveats. Current documentation reviewed; project results separately pin PyTorch 2.11.0. [Documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

**P15.** Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874. [Paper record](https://arxiv.org/abs/2504.19874).

**P16.** Hugging Face Transformers. *Qwen3.5, version 5.17.0 documentation.* Hybrid architecture and optimized/reference-kernel distinctions. The architecture counts in this paper come from the actual 4B project artifact, not generic configuration defaults. [Documentation](https://huggingface.co/docs/transformers/v5.17.0/model_doc/qwen3_5).

**P17.** CLINC dataset authors. *oos-eval*, pinned revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`. The G artifacts record author labels in `data/data_full.json`, domain membership in `data/domains.json`, and the license file, all with content hashes. This paper uses those recorded source identities and separates author-OOS from deliberately omitted in-scope labels. [Pinned repository](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de).

**P18.** Qwen. *Qwen3.5-2B-Base model card.* Official same-family smaller-checkpoint source, checked for v0.4. Its published text-model specification is not an OpenKind quality, memory, or latency result. [Model card](https://huggingface.co/Qwen/Qwen3.5-2B-Base).

**P19.** Convai Innovations. *Laya model card.* Author-described compact bidirectional decision architecture, option-marker scoring, per-question input budget, batching, and limitations, checked for v0.4. No author timing, calibration, or Jev-superiority claim is adopted as an independently reproduced result. [Model card](https://huggingface.co/convaiinnovations/laya). The linked project is a candidate for a future code audit, not an audited dependency of this revision.

**P20.** Jiang, P., et al. (2026). *Efficient, Property-Aligned Fan-Out Retrieval via RL-Compiled Diffusion.* arXiv:2603.06397v1, 6 March 2026. The R4T workflow is cited for its separation of expensive teacher-side optimization from lightweight deployment; the proposed decision-model distillation study is an extrapolation, not a paper result. [Paper](https://arxiv.org/html/2603.06397v1).


**P21.** Gundala, H. *Qwen-2.5-1B-RLCD / Parallel Constrained Decoding for Apple Silicon.* Hugging Face model/repository documentation, reviewed 20 September 2026. Describes a shared-prefix Qwen/MLX inference path with cache broadcasting across fields, constrained candidate-token logit slicing, token-tree continuation and host-side JSON assembly. The page reports M4 Max benchmark values including 68–75 ms four-field cases, 270 ms for 28 fields and 89 ms for one 255-choice case. OpenKind cites these as author-reported inference mechanics/performance, not evidence that TypeSafe RLCD training has been reproduced or that the resulting probabilities are empirically calibrated. [Model card](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).

**P22.** Reddy, N. (19 September 2026). *Jev-style models on DGX Spark.* More Than a Machine. External comparison of Jev 1.13, Laya and local decision readers on WANLI, BoolQ, serving Q-scaling and ViZDoom controllers. The reported serving table has Jev p50 105.1→109.2 ms from one to four questions, Laya 16.4→29.1 ms, and tuned Qwen3.5 167.0→665.1 ms; Jev includes hosted HTTP while local readers are warm/in-process, so absolute latencies are not directly comparable. The post also reports separate Brier/ECE metrics and campaign-to-campaign Jev API variation. OpenKind uses it as external benchmark context, not evidence of Jev's private architecture. [Article](https://morethanamachine.com/posts/jev-style-decisions-dgx-spark/).

**P23.** Lee, T. *SemIf — Method.* Repository method note, reviewed 21 September 2026. It freezes prompts, IDs, labels, task semantics, revisions, and metrics before full evaluation; reports unlike task families separately; uses source-group bootstrap intervals; aligns perturbation probabilities by semantic option ID; and distinguishes shape-matched systems measurements from semantic or Jev head-to-head claims. OpenKind uses these as methodological checks for the Phase 4A record, not as Phase 4A measurements or evidence that the benchmarks are identical. [Method](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/METHOD.md).

# Appendix C. Reading the metrics

**NLL:** negative log-likelihood; lower is better. It penalizes confident wrong predictions and is sensitive to probability quality rather than only the winning class.

**Brier score:** the recorded multiclass sum of squared probability errors; comparisons should retain the same convention and task.

**ECE:** expected calibration error; these reports use 15-bin top-label ECE. It is descriptive and sensitive to binning and sample composition.

**None recall:** fraction of target-none episodes correctly assigned none, with the source of that target stated. In C–F and G’s in-scope panels, it means deliberate omission of an annotated intent, not an OOS detection rate. In G’s separate author-OOS panel, it refers to that specific labeled OOS sample. Do not merge those meanings without an explicit population definition. [E7]

**False-none rate:** fraction of answerable episodes assigned none. Some source tables call this false abstention; the whitepaper uses the more specific semantic name.

**Answer rate / coverage:** proportion receiving an automated candidate answer under a specified definition. Unthresholded candidate coverage, raw panel policy coverage, and scenario-weighted policy answer rate are not interchangeable.

**Answerable accuracy:** correctness over all episodes where the annotated correct option is supplied; predicting none is an error in this denominator. **Error among accepted answers:** incorrect accepted candidate outputs divided by accepted candidate outputs under the specified policy; undefined when the policy accepts nothing. Report independent-message counts as well as episode counts.

**Policy-output change:** an execution alternative changes answer versus review or changes the accepted candidate under at least one fixed policy. It is a disagreement measure unless separately adjudicated against ground truth.

**MiB / GiB:** binary units. Tensor allocated memory, allocator reserved memory, driver free memory, persistent cache storage, and peak temporary allocation describe different quantities.

**Codec-only versus full-reference delta:** the former compares restored compressed KV with uncompressed cached execution under the same chunking; the latter also includes any chunking/batching difference from full sequential. **One-entry storage ratio:** original hybrid-cache tensor bytes divided by stored bytes plus the configuration’s shared codec tables. A ratio below one means the attributed stored form is larger. [E6]

**Fresh test:** new relative to named project experiments. It does not imply pretraining decontamination. **Regression panel:** reused inputs with frozen expected behavior; useful for implementation validation, not a fresh generalization estimate.


**Independent-question count Q versus candidate count K:** Q counts distinct questions against a state; K counts alternatives within one Choice question. More K, more batch rows, and one framework call do not establish that state computation was shared across Q.

**New-model quality versus implementation equivalence:** quality compares separately versioned learned/input-contract profiles on fresh task evidence; equivalence checks implementations claiming to preserve one profile. Neither is a substitute for the other.

**In-progress study versus design snapshot:** owner-reported execution status does not prove which optional arms ran or that results passed; an unexecuted Library notebook describes a design, not the current runtime.

**Development selection versus final result:** H's `profile_dev_nll` remains pre-temperature, message/origin-weighted development loss under 60/25/15 weights. E9 adds actual final predictions: its ordinary `quality` fields are raw episode averages, and `assumption_weighted_nll` is separate. The original selected profile and thresholds remain fixed despite different final rankings. Newly inspected final outcomes are not fresh data for later tuning. [E8; E9]

**Partial run with retained fits:** a worker can save valid intermediate fitting artifacts and then fail during reporting. Retain both facts. A ZIP, a selected profile, an enabled `run_final` flag or a complete-looking summary is not proof of final evaluation. [E8]

**Whole-document training versus prefix evaluation:** E30 admits only complete training documents within its cap; the diagnostic gate still uses prefix-capped inputs with locked original labels. Neither positive span retention nor missing annotations automatically determine semantic sufficiency. [E29; E30]

**Training pool, exposure, and selection:** eligible questions are not the number actually visited. E30 has 2,176 eligible training questions, 960 exposures per full fit, 734 unique questions visited, and a selected update-80 snapshot after 640 exposures/538 unique questions. The selected checkpoint is not necessarily the last optimizer snapshot. [E30; V4]

**Retention versus calibration:** holding QASPER out of LoRA and checkpoint selection is distinct from its continuing role in calibration-fit and policy-development. Raw semantic retention, proper-score calibration, and policy transfer must be reported separately. [E30]

**Policy-cost weighting and strict benefit:** pooled-question, equal-source row-mean, and equal-source/equal-component costs are different estimands. With unit wrong cost and review cost 0.1, exact break-even is not a strict improvement, even if binary floating-point represents it slightly below 0.1. A documentation correction does not rewrite a frozen evaluator or selection. [E30; V4]

**Fixed-dose versus selected view:** E31/E32 report fixed80 learning contrasts
separately from guarded development selection. A selected-zero view aliases the
frozen parent; repeated alias rows are not independent examples or a trained
model with perfect retention. [E31; E32]

**Replay-task learning versus retention:** a held-out-from-fit regression score on
the replay domain can improve while another task or individual class regresses.
E32 improves SNLI substantially without preserving QASPER or supported ContractNLI
entailment. An already-exposed regression panel is not fresh confirmation. [E32]

# Appendix D. Revision history

## Version 0.3: historical record

| Area | Change from version 0.2 |
|---|---|
| Front matter, abstract, executive assessment, chronology | Include completed Phase 2G and distinguish stronger execution evidence from unresolved semantic/rejection quality |
| Section 5.2 | Separate softmax probability dilution from argmax-based none selection, including the explicit maximum-logit rule |
| Section 5.3 | Restore the refitted-constant ablation and its share of the original-to-set-linear improvement |
| Section 8.1 | Describe the existing segmented tokenizer followed by exact common-prefix planning; do not imply whole-string retokenization |
| New Section 10 | Add G’s scope, strict-FP32 parity, TF32 behavior and timing, fresh Banking/CLINC/OOS quality, criteria-confusion case, context controls, and TTL traffic |
| Sections 11–13 | Renumber previous recommendation chapters and prioritize criteria/rejection transfer; keep Rust, state-first, Score/Noul, and smaller-model work explicitly uncompleted |
| Appendices | Add E7/R1/R2 provenance, G hashes and dataset identity, refined metric definitions, and this revision record |

Historical B–F numerical tables are retained. New G figures are not substituted for prior workloads or presented as a controlled cross-phase accuracy/latency improvement. The original v0.2 attachment and all source results remain unchanged. The separate change log and unified diff record the edits; the verification companion records what was recomputed from saved artifacts rather than newly inferred.


## Version 0.4: research refocus, with Phase 2H retained

| Area | Change from version 0.3 |
|---|---|
| Title, abstract, executive assessment | Reframe the destination as a compact, broadly useful multi-question model, with the current 4B scorer retained as a reference |
| Chronology and Section 13.1 | Record Phase 2H as in progress; preserve its design, optional/default distinctions, attribution limits, and unknown live-run outcomes |
| Sections 2.4 and 11.2 | Distinguish question semantics and Q-scaling from dynamic labels and K-scaling; prioritize matched state-first nested sharing |
| Sections 6.3 and 11.6 | Bring supervised adaptation, smaller-model comparisons, feature-aware applicability, and honest ordinal/probability objectives forward |
| Section 11.7 | Add bounded compact-encoder/Laya and later teacher/distillation lessons, with external descriptions separated from proposed experiments |
| Sections 11.8–11.9 | Add explicit none/key semantics, a bounded Rust-to-resident-reference-worker prototype, and one current contract/status authority |
| Section 13.2 | Operationalize separate implementation-equivalence and new-model-quality gates; preserve all historical failures and tolerances |
| Sections 13.3–13.5 | Replace the old serial roadmap with early matched model/data/Q-scaling work and parallel thin-service integration; pause repetitive low-bit cache sweeps and defer speculative rewrites |
| Source register and companion | Record the recovered review, H notebook design, owner status, share-page limitation, focused primary checks, and textual-integrity—not experiment-reproduction—scope |

Completed B–G numerical tables are unchanged. No Phase 2H outcomes, smaller-model winner, LoRA gain, compact-encoder superiority, service performance, or Rust/Metal parity is asserted. The supplied whitepaper and all persistent source files remain unchanged. New prose distinguishes historical measurements, in-progress work, external author descriptions, and proposed next experiments.


## Version 0.5: partial Phase 2H fitting/development update

| Area | Change from version 0.4 |
|---|---|
| Front matter, abstract and executive assessment | Add the actual H attempt and its preliminary development findings; distinguish the reporting failure from a completed final benchmark |
| Sections 3 and 12 | Replace design-only/current in-progress status with versioned partial-run evidence; document the bounded archive audit and missing final outputs |
| Section 13.1 | Add actual splits, objective weights, all 14 main development profiles, constant-versus-set none controls, selected calibration/policies, and six primitive fits |
| Strategic recommendations and conclusion | Preserve compact Qwen-led multi-question aims, early matched adaptation/model-size comparisons, and separate new-model/equivalence tracks; require recovery of H without changing its registered study |
| Sources and definitions | Add E8, dataset/run hashes, failure location, development-versus-final terminology, and separate historical audit dates from current checks |
| Companion files | Add full diff, revision manifest, runnable saved-artifact checker and bounded audit output; leave original files and notebook unchanged |

Completed B–G numerical tables are retained unchanged. The H additions are fitting, development, calibration-gate or policy-selection records, not new final-test accuracy, generalization, reliability, TF32 acceptance, primitive validation or serving-performance evidence. The source v0.4, the original failed attempt and its reservations remain unchanged. No recovery or new model computation was performed while preparing this revision.


## Version 0.5.1: direct saved-result confirmation and roadmap synchronization

| Area | Change from version 0.5 |
|---|---|
| Source authority | Build on the newer v0.5 rather than revert to v0.4; directly reconcile saved Colab finish output, Drive summary and same-run archive |
| Phase 2H status | Keep partial/fits-saved/final-blocked status; distinguish generic open-question strings, export success, worker completion and final results |
| Task mapping | Restore actual H criteria/rejection study; use 2H-C1–C5 for closeout; relocate old H engineering proposals to 2I/2J/Track S |
| Dependencies | Place reviewed data and selection contracts before matched model comparison; allow independent service/contract work alongside H recovery and modeling |
| Acceptance | Keep same-model equivalence separate from new-model quality; require profile-specific native/capability gates and avoid invented universal memory/precision thresholds |
| Test/status authority | Preserve supplied 195-test history but remove an unsupported assertion of verified current HEAD; no repository tests performed |
| Integrity | Preserve all historical B–G and H numeric tables; retain old errors and failed gates; supply diffs, hashes and bounded source/document checks |

At the v0.5.1 cutoff, no new model inference, final evaluation, notebook repair, source mutation or native-backend test was claimed. That was a coordinated documentation revision, not a new experimental run. The original failed H attempt was then the inspected result authority; E9 now supplies its traceable continuation. [E8; R4; R5; V1]


## Version 0.5.2: recovered discussion mapped to future work and tests

The main recommendations were already present in v0.5.1. This revision adds a stable review-to-task/test matrix, explicitly names the ModernBERT/Laya early comparison in the roadmap, and restores the earlier P2.1–P2.3 conditional checkboxes. It retains the distinction between matched compact training and a released external checkpoint, and between an offline teacher/student experiment and a diffusion or RLCD architecture commitment.

All historical metric tables and the H saved-result section remain unchanged. The source boundary remains explicit: the recovered main review is archived, while a complete readable shared-chat transcript is unavailable. Recovery-notebook delivery is noted separately from the archived H run’s status. No new execution or test result is asserted.

## Version 0.6: completed Phase 2H continuation and bounded study closeout

| Area | Change from version 0.5.2 |
|---|---|
| Current status | Record both required continuation workers completed; preserve the failed original attempt and no-retraining lineage |
| Detailed H readout | Retain original development history and add final family/rejection/calibration metrics, all 14 profiles, policy risk, numerical gates, controlled robustness, primitive results and request/memory measurements |
| Interpretation | Keep pre-final selection fixed; expose support-conditioned development/final transfer reversal, remaining omission errors and non-equivalent TF32 behavior |
| Roadmap | Close 2H-C1–C5; replace H's detailed roadmap tables with a summary and source links; retain open 2I/2J/Track S/P2 work |
| Evidence | Add E9 and V2, source hashes, read-only saved-output checker, exact diffs and document-integrity report |

No historical measurements or gold labels are changed. Additional comparisons are explicitly identified as saved-row aggregations. H completion does not mean broad primitive, arbitrary-question, smaller-model, LoRA, compact-encoder, native-backend or service acceptance. The sources and original result folders are unchanged by this revision.

## Version 0.6.1: review-gated workbench checkpoint

| Area | Change from v0.6 |
|---|---|
| Current status | Keep H completed; record 2I/2J preparation and GPU mechanics with an overall blocked review gate |
| New §14 | Add source-reported synthetic feature deltas, root/runtime observations, unsigned intake and the exact gate requirements |
| Test coverage | Preserve original result names/values while identifying duplicate-question append under the misleading unrelated-question key |
| Roadmap | Credit bounded preparation without closing 2I/2J; prioritize actual review, explicit device/limits, matching manifest and a distinct extra-question test |
| Provenance | Add E10/I0/V3 and a 34-check saved-artifact audit; keep raw feature reproduction and annotation review outside scope |

Historical measurement tables, earlier failures, H closeout, ModernBERT/Laya and R4T mappings and the existing acceptance framework remain intact. No model result or human approval is inferred from archive export.

## Version 0.7: exploratory model selection and Rust handoff

The `2ij.2.0` exploratory screen completed the bounded model-selection program, selected `a047d6802c3f06f085b8` before final evaluation, measured state-first multi-question mechanics and exported a reload-verified reference bundle. State-first Qwen4B was the strongest tested family in this pilot; Qwen2B remained a credible deployment challenger, while the tested ModernBERT treatments were much faster/smaller but substantially weaker on this task construction. The result unblocked native parity work without constituting release promotion.

## Version 0.7.1: external PCD/DGX benchmark refocus

This revision adds P21/P22 and changes no OpenKind model result, selection or historical acceptance decision. It clarifies that the public `Qwen-2.5-1B-RLCD` artifact is primarily evidence for parallel constrained-decoding mechanics rather than an established TypeSafe-RLCD training procedure; contrasts schema-first PCD with OpenKind's selected state-first isolation contract; adds explicit state/evidence-sufficiency and high-cardinality follow-ups; and refocuses Phase 3 on native parity → branchable hybrid state → sequential nested parity → batched question execution → candidate batching → Q-amortization/high-K/repeatability measurements. External Jev/Laya/Spark numbers are benchmark context, not OpenKind release thresholds.

## Version 0.7.2: Phase 3A/3B references and initial Rust parity gates

This revision adds E12, E13, RUST1, RUST2, RUST3, and PUB1 without changing the selected profile or prior quality results. It records the completed Phase 3A Python systems run. That run covers corrected hybrid-state reindexing, semantic/high-K parity, repeatability, and the measured execution crossover. The revision also records the completed Phase 3B Python backbone export with exact tokens and 47 vectors. Rust head/probability, exact-token, CPU full-sequence backbone, and Qwen-specific cached-continuation parity now pass. The fail-closed Phase 3B loader and two-shard checkpoint path are implemented.

Backend-neutral `BranchableState`, nested/batched Q/K execution, Metal, service integration, and Mac scheduling remain open. The public Hugging Face repository is a reference publication, not release certification. Phase 3A.1 remains conditional scheduler work if native profiling cannot derive stable crossover rules.

## Version 0.7.2 amendment: backend-neutral BranchableState (RUST4)

This implementation amendment adds the Phase 3.4 Rust branch-state checkpoint without changing the selected profile, bundle, base revision, model results, or tolerances. The contract now resides in `crates/openkind-runtime/src/branch`, and `BackboneState` binds it with complete hybrid state, immutable-root fork, batched fork, and gather/select. The checkpoint replays the Phase 3B root/question/candidate fixtures with exact tensor-payload accounting and exact cached-versus-full content identity. RUST8 later split the scheduling and content fingerprint types and qualified process-memory accounting. CPU native parity now passes; Metal/native accelerated parity does not follow from that result.

## Version 0.7.2 amendment: sequential nested execution (RUST5)

This implementation amendment adds the Phase 3.5 Rust sequential nested checkpoint without changing the selected profile, bundle, base revision, model, tolerances, or fixtures. `crates/openkind-backends/src/qwen35/backbone/nested.rs` implements the nested graph behind a `SequentialNestedExecutor` trait: prefill the shared state once, fork the immutable root per question, advance it, fork the question state per candidate, and advance each candidate fork, with fail-closed position and immutability verification after every stage. The checkpoint-gated `qwen35_nested_parity` stage executes all four Phase 3B questions and all 10 candidates through this path with maximum probability delta `4.5869e-06`, zero argmax changes, zero policy changes, root content identity against an independent prefill, exact replay and sibling-order determinism, and exact cached-versus-full feature and state equality (`0.0`). Hidden-vector deltas remain localization diagnostics. CPU native parity now passes; Metal/native accelerated parity does not yet follow from that result. Batched question execution is the next gate, measured against this sequential baseline. Phase 3A.1 stays a conditional cost-model study.

## Version 0.7.2 amendment: breadth-first batched Q/K execution (RUST6)

This implementation amendment adds the Phase 3.6/3.7 Rust batched execution checkpoint without changing the selected profile, bundle, base revision, model, tolerances, or fixtures. `crates/openkind-backends/src/qwen35/backbone/batched.rs` implements the breadth-first graph (`run_batched_questions`, `run_batched_candidates`, `run_batched_nested`) over `BranchableState::fork_batch`: one immutable prefill, `Q` isolated question lanes advanced breadth-first, then a `K` candidate fan-out per question state, with fail-closed root, sibling-lane, and position verification after every stage. The checkpoint-gated `qwen35_batched_parity` stage proves exact parity with the sequential baseline — every question and candidate feature delta `0.0` with identical strict state fingerprints across all four Phase 3B questions and 10 candidates — while the frozen head reaches the same `4.5869e-06` maximum probability delta with zero argmax, zero policy, and zero cross-strategy decision changes, and fan-out byte accounting is exact. Offline tests pin gather/reorder equivalence. Per-lane executor calls remain the primitive, so this does not claim vectorized suffix kernels, Metal, or adaptive scheduling; CPU native parity does not imply Metal or accelerated parity. Target-Mac scheduler measurement is the next gate. Phase 3A.1 stays a conditional cost-model study.

## Version 0.7.2 amendment: measured adaptive scheduler (RUST7)

This implementation amendment adds the Phase 3.8 Rust adaptive-scheduler checkpoint without changing the selected profile, bundle, base revision, model, tolerances, or fixtures. The named M4 Max harness measured five warm CPU workloads with feature parity asserted per repetition: sharing beat `repeated_full` in every measured cell (1.25×–1.98×), the two per-lane shared topologies were equal within noise, and `T(3)/T(1)` was 2.87 repeated versus 2.11–2.18 shared. The lowest measured token-work ratio was `2.52`. The original 2.0 policy was below the measured envelope and therefore heuristic, not a measured crossover. RUST8 corrects the default to 2.52, distinguishes state fan-out from vectorized forward, and adds process-peak admission. Cold-start/cache-warmth and vectorized suffix packing remain open.

## Version 0.8.0 contract and service corrections (RUST8)

RUST8 moves generic branch/execution contracts into `openkind-runtime`, replaces the interchangeable fingerprint value with `SchedulingFingerprint` and `ContentFingerprint`, and names the 59,899,904-byte Qwen root correctly as tensor payload. Scheduler admission can combine tensor payload with observed process peak, forward scratch, allocator headroom, hard process limits, and vectorized lane ceilings. The current CPU backend advertises per-lane forward and therefore selects `NestedSequential`; the breadth-first `NestedBatched` plan requires explicit vectorized question and candidate capability. Offline K=32/64/128/255 stress covers estimator/admission behavior without allocating a naïve full fan-out, while `qwen35_model_stress` defines the remaining checkpoint-gated streaming run. `BranchStateCache` adds tenant/TTL/byte-bounded in-process reuse keyed only by strict content identity. `Qwen35DecisionEngine` registers the pinned native path directly with bounded queueing. Choice requires explicit `__none__` and preserves its probability mass; Choice/Score confidence is normalized entropy, not top probability. These are implementation and contract gates, not model-promotion or load/soak evidence.

## Version 0.8.0: Native CPU reference engine through adaptive scheduling

This milestone unifies the Phase 3.1–3.8 implementation amendments into a clean provenance boundary rather than accumulating incremental RUST amendments under v0.7.2:

| Area | Version 0.8.0 Consolidation |
|---|---|
| Native 4B Qwen backbone | Offline manifest/bundle verification, digest-locked tokenizer, exact token-fixture rendering, and 32-layer Candle CPU decoder in FP32 (`crates/openkind-backends`) |
| Layer & candidate parity | Exact embedding widening, all 34 diagnostic stages verified, 10 candidate sequences evaluated within `4.5869e-06` probability delta and zero argmax/policy changes |
| Cached continuation | Immutable Qwen hybrid continuation state isolating attention KV, DeltaNet recurrent state, convolution state, and logical position |
| Backend-neutral branching | Runtime-owned `BranchableState` and `BranchBatch`, distinct scheduling/content fingerprint types, exact tensor-payload accounting, and fork/gather traits |
| Sequential nested execution | `SequentialNestedExecutor` performing one shared prefill, per-question forks, and per-candidate forks with fail-closed immutability and position checks |
| Breadth-first batched Q/K | Fan-out over `fork_batch` with per-lane executor advancing, achieving exact `0.0` feature deltas against the sequential baseline and verified byte accounting |
| Native adaptive scheduler | Capability-aware strategy policy using the lowest measured ratio (`2.52`), tensor/process-peak admission, and sequential CPU default until vectorized forward exists |
| M4 Max measurements | Commit-stamped hardware rerun across 5 workloads: CPU-default state sharing is 1.29×–2.02× faster than repeated-full, batched topology stays within 2.9% of sequential, feature parity holds across strategies, and no measured claim is made below 2.52 |
| High-K and persistence contracts | K=32/64/128/255 estimator/admission stress, tenant/TTL/byte-bounded strict-content cache, and versioned/digest-checked pinned-state snapshot |
| Direct service adapter | Bounded native `DecisionEngine`, explicit `__none__` preservation, entropy confidence, and offline artifact registration |

CPU native parity does not imply Metal or accelerated parity. Bounded model-backed high-K completion, fresh-process persistence replay, and named-machine native CPU service load/soak are recorded; practical high-K latency, Metal or accelerated parity, reviewed model quality, and product-release promotion remain open.

## Version 0.8.0 native follow-up: bounded high-K, structural persistence, and service smoke (RUST10)

The named-machine native follow-up campaign ([`../verification/2026-09-20-v0.8.0-native-follow-up-working-tree.md`](../verification/2026-09-20-v0.8.0-native-follow-up-working-tree.md)) completes bounded execution for the first tier of post-scheduler gates without claiming release promotion:

1. **Model-backed high-K completion (3.9b).** Candidate continuations for $K \in \{32, 64, 128, 255\}$ streamed from one question branch without retaining fan-out states. Wall times were 60.16 s, 90.40 s, 145.06 s, and 246.77 s; peak RSS remained tightly bounded between 7,982,170,112 and 7,991,951,360 bytes. The state root remained byte-identical and immutable throughout.
2. **Structural fresh-process persistence (3.10).** Process A persisted the pinned root state to disk; process B restored it and executed question and candidate continuations against the restored state. The restored `ContentFingerprint` matched exactly, positions advanced identically ($98 \to 109 \to 121$), and cached versus independent full hidden output delta was `0.0`. The 20 September run still had candidate-feature and probability replay open; the 22 September follow-up below closes that comparison.
3. **Native service lifecycle smoke (3.11).** The `openkindd` binary booted with `--models qwen35=qwen35-native-cpu` and explicit offline artifact paths. Real `/v1/systemone` evaluation returned HTTP 200 with typed answers; concurrency 1 with queue 1 admitted two requests and returned HTTP 529 on the third in 0.65 ms; client cancellation during blocking model execution released permits properly; queued cancellation recovered; 20 health probes passed at stable resident memory; and SIGINT produced a clean exit.
4. **Workspace MSRV alignment.** Rust 1.88 is verified as the locked workspace floor and enforced in CI due to the tonic 0.14.6 gRPC stack requirement.
5. **Standardized evidence packaging.** Execution identities bind role-typed token digests (`ExecutionInputDigest`, `SemanticSetDigest`), physical compute modes (`BatchForwardMode`), and sanitized invocation records into the backend-neutral `openkind-native-run/v1` schema.

The 22 September persistence follow-up ([verification record](../verification/phase3.10-2026-09-22/README.md)) closes the full 3.10 replay gate on the named M4 Max. Across 8 candidates and 3 questions under the restored root, all candidate features and calibrated probability vectors matched independent full-sequence execution exactly; argmax and policy actions had zero changes. The restored root matched an independent prefill and remained immutable. This is CPU FP32 implementation-equivalence evidence, not model-quality, Metal, production-load, or release-promotion evidence.

## Native service-gate follow-up (RUST11)

The release-mode `openkindd` campaign in
[`verification/native-service-gate/2026-09-22-rerun2/`](../verification/native-service-gate/2026-09-22-rerun2/README.md)
completed on a Mac16,5 Apple M4 Max (36 GiB, macOS 26.6.2; Rust 1.98.1).
The working tree was dirty. The pre-build, post-build, and daemon-start source
fingerprints match on the commit, tracked diff, untracked crate sources, and
harness. Shared source files changed after daemon startup and the report records
the final workspace fingerprints separately. The release executable hash
identifies the artifact exercised by the campaign and stayed unchanged through
the run.

At client concurrency 4, nine requests returned HTTP 200 and three returned
HTTP 529 at the configured admission limit. Accepted queue-inclusive p50/p95/p99
latencies were 31.07/47.02/47.02 seconds at 0.06420 requests per second. The
30-minute single-client soak completed 115/115 requests with HTTP 200 and 345
answers; p50/p95/p99 were 15.78/15.88/15.94 seconds at 0.06339 requests per
second. Sampled RSS peaked at 10,280,976,384 bytes, while the daemon peak-RSS
metric reached 10,283,155,456 bytes. A disconnected request incremented the cancellation
counter, and the follow-up request recovered with HTTP 200. A separate 1 ms
queue-inclusive timeout returned HTTP 504 in 2.12 ms. Validation returned
HTTP 422 for an invalid wire request and HTTP 400 for malformed JSON; request
IDs were echoed, logs passed the body/ID redaction audit, and both daemons shut
down cleanly.

The service fixture has no reviewed labels, and its bundle manifest declares
`data_scope: exploratory_pilot` and `production_ready: false`. RUST11 therefore
closes native CPU service lifecycle and release-mode artifact evidence only.
It does not establish semantic accuracy, accepted-error/coverage, Metal
behavior, signed packaging, or product-release promotion.

## Version 0.8.1: Phase 4A benchmark and non-final model record

| Area | Version 0.8.1 update |
|---|---|
| Version identity | Marks the document as v0.8.1 while preserving v0.8.0 as the editing base with recorded SHA-256 |
| Phase 4A.0 | Adds the locked ContractNLI/QASPER corpus, state/component-safe split counts, table hashes, frozen model/cache identity, and closed-final boundary |
| Phase 4A.1 | Adds the original B1 training curve, selected checkpoint, non-final calibration/source metrics, policy behavior, bootstrap interval, and unset-promotion-gate status |
| Phase 4A.2 B0 | Adds the matched-control contract, completed early-stopped result, by-source/source-macro comparison, policy result, checkpoint lock, and final-unavailable status |
| Phase 4A.2 completed sweep | Adds completed balanced-B1, factorized-B2, refined-B2R, and B2 weight-8 reports, checkpoints, locks, and common calibration-gate comparisons while preserving the earlier epoch-11 progress record |
| Phase 4B.2 | Adds the completed cap-12/effective-8.5602 B2 result, selected epoch 11, per-source false-none diagnostics, policy result, prediction integrity checks, and final-unavailable status |
| Interpretation | Separates raw accuracy from balanced discrimination/applicability, identifies the repeated QASPER failure and cross-source false-none tradeoff, and closes the scalar-weight sweep |
| Method | Adds SemIf's public benchmark method as contextual guidance on frozen matrices, per-source reporting, grouped uncertainty, perturbations, and systems/quality separation |
| Program | Makes a source/class-normalized applicability objective with both recall and false-none gates the next seed-17 test; replication and final opening remain conditional |
| Provenance | Adds E14–E16/P23 with direct source links and states exactly what was read, recomputed, not rerun, and not opened |

No prior metric, failure, selection decision, native-parity result, or revision history is removed. This revision runs no training or model inference, opens no Phase 4A/4B final label or prediction, and promotes no model. It updates the runnable non-final workbench only for a separately labeled future objective experiment; all completed result directories and locks remain immutable.

## Version 0.8.2: OpenKind name confirmation, Phase 4B–4D closeout, and Phase 4E audit gate

| Area | Version 0.8.2 update |
|---|---|
| Identity | Confirms **OpenKind** as the working project name; preserves legacy `OpenDecision_...` Drive paths as immutable provenance |
| Phase 4B.2.1 | Adds the stratified applicability result: QASPER operating recall 0.2564 development / 0.3111 gate, but gate policy cost 0.11675 |
| Phase 4B.3 | Records the development pass and failed transfer: QASPER development recall 0.3451, gate false-none 0.2110, gate policy cost 0.10221 |
| Phase 4C | Rejects the head-only child: best-child NLL/Brier regress 2.77%/2.10%; epoch-0 parent retained |
| Phase 4D | Rejects the 1,345-parameter evidence residual: one extra QASPER true positive with 6.08%/6.60% NLL/Brier regressions; strict JSON/`null` serialization validated |
| Interpretation | Closes further scalar-weight, pairwise, head-only, and shallow-residual sweeps under the frozen representation; identifies weak QASPER separability and policy transfer as the controlling failure |
| Phase 4E | Completes Phase 4E-A2 repaired audit; 66.4% decided agreement, 51 adjudication rows emitted; reveals 44 challenged semantic-none labels and 6 serialization/table defects; blocks Phase 4E-B pending independent adjudication and benchmark/serialization repair |
| Provenance | Adds E17–E22 with direct run links and exact experiment/checkpoint/evaluation hashes; final remains unavailable and unopened |

No prior metric, failure, selection decision, native-parity result, or revision history is removed. This revision runs no training or model inference, opens no Phase 4 final label or prediction, and promotes no model. It documents completed immutable results and creates a separate Phase 4E audit workbench; all prior result directories and locks remain unchanged.

## 24 September 2026 evidence addendum to version 0.8.2

Adds E23–E26 and §§18.15–18.16. A3 preserves the 51-row independent adjudication while identifying six later same-assistant follow-ups; A4 is a source-alignment preflight. A separate assistant-reviewed 27-case disposition freezes quarantine and the 77-tie diagnostic policy without modifying benchmark inputs. The seed-17 4E-B.0 cheap-prefill probe fails every non-final gate and has no separate result lock. Source-aligned repair, independent review of proposed corrections, conditional new model work, and final evaluation remain open. This addendum edits documentation only; it runs no model or notebook and promotes no checkpoint.

## Later 24 September 2026 option-logit evidence addendum

Adds E27 and §18.17 from the completed, separately locked 4E-B.1 Colab run. The direct-logit method improves candidate ranking on the sampled answerable ContractNLI rows but misses every ContractNLI semantic-none case; QASPER's one-option answerability result also falls below its majority baseline. The result does not satisfy the Phase 4E source-repair and independent-review gate, reopen final, or promote a checkpoint.

## Later 24 September 2026 candidate-sweep evidence addendum

Adds E28 and §18.18 from the locked 4E-B.2 Colab sweep. Development selection favors order-averaged ContractNLI ranking with calibrated `Z`, but the gate falsely rejects 46/108 answerable questions and retains only 3/24 contradicted full decisions. QASPER's selected threshold grid never predicts semantic none on development or gate. A lower-threshold check is explicitly post-hoc. The result neither closes the source-review gate nor opens final or promotes a model.

## 24 September 2026 strategy and evidence-consistency addendum

Refocuses the abstract, executive assessment, recommendations, and priority
order around reviewed evidence and useful decisions before learned cost
reduction. Section 18.19 distinguishes reuse from cheap representation, ranking
from applicability, and full-source evidence from finalized-input visibility.
Completed factorization/StateQuery and native CPU/MLX FP32 work are no longer
presented as untried or uniformly pending. Teacher status remains conditional.

The documentation review inspected local A3/A4 and runtime evidence, including
the four-row A4 alignment table and 40-caption inventory. It also reread the
Drive E28 report and result lock. The report hash matches the lock and its
aggregate counts support the ranking/rejection account. This review did not
repeat the earlier row-level recomputation or download the 72 state parts,
rerun notebooks, train a model, or access final outcomes. Prior experiments and
locks remain unchanged. The earlier opening is retained below as history.

## Version 0.8.3: completed frozen comparison and matched decision-LoRA results

This 25–26 September result update uses the uploaded `WHITEPAPER.md` as the
editing authority. It adds E29/E30/V4 and §§18.20–18.22, and updates the abstract,
executive assessment, chronology, audit/evaluation scope, architecture distinction,
research questions, current work order, conclusion, and metric definitions.
Historical experiment tables, prior revision entries, and Appendix E are retained.

| Updated area | Recorded change |
|---|---|
| M1 / M2.1 | Finalized-input/visibility and raw historical-score diagnostics are distinguished from completed FP32 J0/J1 × two-budget inference; the frozen comparison is no longer future work. |
| M2.2 execution | Both BF16 LoRA fits complete 120 updates and select update 80; all four matched arms and the code/order diagnostic complete. |
| Main finding | ContractNLI accuracy and contradiction/none decisions improve substantially; entailment preservation and QASPER retention fail. No promotable multi-source model is selected. |
| Policy accounting | The exact J2/QASPER break-even subcheck is documented as a floating-point reporting defect; weighting-aligned and row-pooled costs are reported separately. Original results remain immutable. |
| Next hypothesis | Retention-aware training/selection is proposed, not executed; no automatic longer training, seed expansion, architecture sweep, compression, or final opening follows. |
| Verification boundary | CPU-only supplied-snapshot readouts and document checks are rerun; no model, optimizer, source adjudication, historical native test, or protected-final evaluation is run. |

The active roadmap, architecture files, source notebook, Drive results, labels,
thresholds, checkpoint selections, and approvals are not changed by this update.
The retained §18.19 interpretation is explicitly dated so its earlier future-work
language cannot be confused with the completed later experiments.

## 26 September 2026 — version 0.8.4 retention-result update

Adds E31, V5, and §§18.23–18.24 for the completed parent-KL replay experiment.
Updates abstract, executive conclusions, chronology, research questions, active
model status, priority table, and conclusion. Records both selected-zero outcomes,
fixed80 probability recovery, entailment and QASPER failures, stronger SNLI results
for contract-only specialists, exact policy weighting, and the now-exposed
retention-diagnostic boundary. Dates rather than erases the earlier §18.22 proposal.

The companion N1 notebook prepares the source-label CE versus parent-KL comparison
on unchanged inputs and with unchanged preservation checks. Its local software
tests are not new pretrained results. Historical numeric tables and selections,
E29 FP32 versus E30/E31 BF16 boundaries, original source labels/locks, and Appendix E
are retained. No roadmap, old experiment notebook, historical optimizer, or
protected-final artifact is edited or opened.

## 26 September 2026 — version 0.8.5 source-label replay result update

Adds E32/V6 and §§18.25–18.26 for the completed v0.6.0 J6/J7 experiment. Updates
the abstract, executive assessment, chronology, research questions, current
model status, proposed work, chapter status table and conclusion. Records strong
SNLI source-label learning, worse original two-source probability scores versus
parent KL, continued entailment/QASPER failures, both guarded-zero selections,
policy weighting, code/order and visibility limits, and source-reported uncertainty.
Dates rather than erases §18.24's pre-run proposal and preserves N1 as authoring
provenance with an E32 completion cross-reference.

The saved-output reviewer is rerun on copies, including SNLI reconstruction from
exported logits; main-benchmark proper scores and bootstrap intervals remain
source-reported. Historical numeric tables, E29 FP32 versus E30–E32 BF16 boundaries,
J2/J3 selected80 and J4–J7 selected0 identities, prior audit/authorization scopes
and Appendix E remain intact. No new notebook, model fit, data/label edit,
threshold change, checkpoint reselection, roadmap edit, protected-final opening
or model promotion is performed. The next evidence-sensitive preservation work
remains proposed and causally unestablished.

## 27 September 2026 systems addendum to version 0.8.5

Adds the Rust flat-field and candidate-pooling negative results to §§11.7.1,
17.3–17.4 and source RUSTM2. The method, numerical parity, latency, memory,
and request-path limits are tied to checked-in records. Historical model
training, selection, quality and audit results retain their earlier cutoff.

# Appendix E. Historical opening before the September refocus

The following opening is preserved from the pre-refocus version 0.8.2 working
copy. Its revision-scoped results remain historical evidence. Its priority,
"next", "open", and proposed-architecture language is superseded by the current
abstract, §§13.3 and 18.19, and the active roadmap. In particular, later A4
outputs record bounded dataset alignment, and native CPU service plus pinned
MLX FP32 parity are no longer universally pending.

## Previous abstract and opening
OpenKind treats decision inference as a scoring protocol rather than a text-generation task. A frozen Qwen3.5-4B-Base backbone produces features; small trained heads convert those features into typed answers and probability distributions. Completed Phases 2B–2G establish useful NLI and dynamic-candidate behavior, important rejection limits, and a strict-FP32 execution reference. Lossless prefix reuse and bounded FP16 attention-KV storage pass the reported sampled gates; four tested low-bit snapshot codecs do not. TF32-permitted batching offers a measured speed opportunity without full equivalence.

On Phase 2G’s sampled non-financial CLINC panel, the set-linear model reaches 93.75% answerable accuracy but only 39.06% omitted-intent recall; separate author-OOS recall is 46.88%. The evidence therefore describes a narrowly adapted scorer, not the limit of a Qwen model trained for broader decisions. [E1–E7; R3]

The research target is a compact model that answers several independent questions over one state. Phase 2H tests that direction without retraining: it validates and reuses saved fitting artifacts, then runs strict-FP32 and TF32-permitted final workers. The preselected support-example joint head reaches 78.91% raw pooled accuracy over 1,152 episodes from 320 messages, but only 63.67% on held-out Banking labels and 66.02% on held-out CLINC domains. An original-criteria joint-head control transfers better than the development-selected support arm, so the paper does not promote a new model after observing the final set.

The selected model rejects all sampled author-OOS episodes, but it still makes omission and policy errors. TF32 preserves final argmax outcomes but fails the 0.005 numerical gate on two final episodes, with further drift in controlled contexts. These results call for broader supervision, representative transfer-aware selection, richer rejection, and genuine question sharing. They do not support more tuning against the exposed final set. The early smaller-Qwen/LoRA/compact-encoder comparisons, separate equivalence and quality tracks, and thin-service path remain priorities. [E8; E9; V2; R3; proposed program]

The subsequent 2I/2J workbench tests the state-sharing mechanism on a real GPU. For synthetic Q = 2 and K = 2, hidden features pass their declared tolerances and the root remains unchanged. This result does not establish trained multi-question quality or compare models.

The reviewed-study gate remains blocked by unsigned review, unspecified hardware and quality limits, and a stale review-protocol hash. Because the added-question fixture duplicates the first question, it cannot count as a distinct unrelated-question test. The next stage is reviewed protocol and data approval, followed by the planned comparisons. [E10; I0; V3]

Version 0.7 applies a pre-final selection rule to the completed `2ij.2.0` exploratory screen. Thirteen configured fit jobs completed, and 31 locked final profiles were evaluated. The rule selects Qwen3.5-4B-Base with a frozen backbone, state-first rendering, and the score-summary rejection head (profile `a047d6802c3f06f085b8`) as the provisional integration candidate.

The held-out exploratory panel contains 320 question episodes from 56 source messages. It records 95.0% accuracy, 0.13006 NLL, 0.07319 Brier, and 0.01661 15-bin ECE. The natural MultiRC answer-correctness slice is harder at 83.33% accuracy and 0.40361 NLL, while the constructed families are near-saturated. This is a bounded pilot result, not a release-quality claim.

A 0.98 policy accepted 214/320 episodes with one wrong accepted decision. The model-selection bundle later passed a clean A100 reload check with maximum probability delta 3.67e-6, zero selected-ID changes, and an independent NumPy/f64 head-algebra check. The exported reference bundle is now the implementation contract for Rust/native parity work. [E11]

**Historical version 0.6 scope.** That revision built on v0.5.2, including its recovered-discussion traceability and conditional P2 tasks. It preserved the historical B–G and H-development numerical tables, added the completed `2h.1.2` continuation, and closed the bounded 2H-C1–C5 tasks. Detailed methods, all retained comparisons, rejection/policy behavior, numerical limits, reliability, primitive scores and resource measurements are in §13.1; the roadmap now keeps a concise closeout and next-work summary. V2 independently checks saved-output arithmetic and artifact lineage, not new model computation. No Qwen inference, retraining, new GPU benchmark, source-label adjudication, repository test or external-literature review was performed during this documentation revision. [E9; V2]

**Version 0.7 scope.** This revision preserves all prior H and 2I/2J checkpoint evidence, adds the completed exploratory model comparison, final held-out pilot readout, multi-question execution measurements, and the A100-verified selected-model bundle. It does not convert the exploratory pilot into an independently reviewed release study, establish production acceptance limits, or claim Rust/Metal parity. The selected integration candidate remains provisional. [E11]

**Version 0.7.1 external-benchmark update.** Two new public sources sharpen the execution roadmap without changing the selected integration profile. A community Qwen Parallel Constrained Decoding implementation shows one-prefill/batched-field mechanics and author-reported Apple-Silicon latency, but does not establish TypeSafe-style RLCD training or calibrated probabilities. A DGX Spark comparison reports substantially flatter Jev Q=1→4 latency than sequential local Qwen wrappers, while also showing task-dependent quality rankings, independent probability-quality variation and service repeatability differences. These results are treated as external benchmarks and architectural prompts, not OpenKind measurements or evidence of Jev internals. The immediate implementation emphasis therefore becomes native parity followed by branchable hybrid state, breadth-first batched question execution and explicit Q-amortization measurement. [P21; P22; recommendation]

**Version 0.7.2 Phase 3A systems update.** Phase 3A run `20260920T024056Z` keeps selected profile `a047d6802c3f06f085b8` and its bundle unchanged and performs no training or reselection. It validates a corrected full-hybrid-state branching reference for Qwen3.5, passes the notebook's semantic batched-parity and high-K systems-parity gates, and records exact same-process repeatability in the saved run. Batched state sharing is slightly slower on the three short semantic Q=4 cases. As shared state length grows, it becomes much faster and consumes more memory.

The result supports Rust `BranchableState` parity plus an adaptive scheduler. It does not support a model redesign or an “always share” optimization rule. The selected integration line is now publicly available at <https://huggingface.co/cowWhySo/OpenKind-Qwen3.5-4B-StateFirst>. Publication does not convert the exploratory profile into a release-quality model or establish Rust/Metal parity. [E12; PUB1]

**Version 0.7.2 Phase 3B and Rust checkpoint.** Run `20260920T152206Z` keeps the same profile, bundle, and base revision. It records no training, model selection, or model modification. It exports four exact token-fixture records. Its 47 FP32 vectors comprise 34 trace stages, 10 full-sequence candidate features, and 3 continuation vectors. The largest fresh-feature difference is `4.9591064453125e-05`. Cached continuation differs from fresh full-sequence execution by at most `1.9073486328125e-05`.

These values are Python self-consistency diagnostics under the notebook's `1e-4` guard. They are not new Rust acceptance tolerances. Rust now passes the selected head/probability, exact-token, full-sequence CPU backbone, and Qwen-specific cached-continuation gates. Across the native 34-stage trace, embedding is exact and final RMSNorm has maximum absolute error `5.8174e-05`. Across all 10 candidate sequences, maximum probability delta is `4.5869e-06`, with zero argmax or policy changes. The native cached candidate exactly matches the native full-sequence result while preserving the source root. [E13; RUST1–RUST3]

**Version 0.8.0 scope: Native CPU reference engine through safe adaptive scheduling and direct registration.** This architectural milestone consolidates Phases 3.1–3.8 (RUST1–RUST7), then closes the review issues recorded in RUST8. RUST9 adds a clean, commit-stamped rerun of the complete workspace, native parity ladder, scheduler stress, and warm-process benchmark. RUST10 adds the named-Mac native follow-up campaign: bounded model-backed K=32/64/128/255 completion and memory stability, structural fresh-process persistence replay, native service lifecycle smoke, and the Rust 1.88 workspace floor. Full restored candidate-feature/decision replay, practical high-K latency, vectorized kernels, Metal, and production load/soak remain open. [E11–E13; RUST1–RUST10]

**Version 0.8.1 scope: Phase 4A natural-document benchmark and completed non-final StateQuery sweep.** Phase 4A.0 locks a two-source corpus with 2,192 states, 15,368 questions, 25,687 options, 21,894 evidence rows, and 18 criteria before model predictions. Phase 4A.1's original B1 StateQuery checkpoint improves the historical candidate-conditioned reference on source-macro accuracy and proper-scoring metrics, but collapses QASPER semantic-none recall to 0.0 and yields negligible policy coverage. Phase 4A.2 now has completed, locked non-final results for the matched B0 control, balanced B1, factorized B2, refined B2R, and the B2 QASPER-weight-8 follow-on. B2 produces the strongest Phase 4A.2 source-macro proper scores among the architecture arms, while the weight-8 arm reaches the best source-macro balanced accuracy, macro F1, and semantic-none recall in that sweep; neither reaches the QASPER development recall floor. Phase 4B.2 then raises the configured weight cap to 12, which saturates at an effective weight of 8.5602. The selected epoch remains ineligible, QASPER gate none recall is 0.1556, and ContractNLI incurs a 0.3966 false-none rate. No Phase 4A/4B final label or prediction was opened, no promotion gate is satisfied, and no model is promoted in this revision. [E14–E16]

**Version 0.8.2 scope: Phase 4B.2.1 through Phase 4D and the Phase 4E audit gate.** The source/class-stratified B2 objective improves QASPER operating-threshold recall to 0.2564 on policy development and 0.3111 on the calibration gate, but misses the 0.30 development floor and transfers to a gate policy cost of 0.11675 with 7 wrong among 32 accepted decisions. Pairwise applicability reaches 0.3451 QASPER development recall, yet its calibration-gate false-none rates rise to 0.2111 for ContractNLI and 0.2110 for QASPER, and gate policy cost remains above review-all at 0.10221. A head-only continuation worsens source-macro development NLL by 2.77% and Brier by 2.10%; an evidence-residual continuation gains only one QASPER none true positive (34 to 35 of 113) while worsening NLL by 6.08% and Brier by 6.60%. Every child is rejected, the stratified epoch-8 parent remains the diagnostic reference, and final stays unavailable and unopened. Phase 4E-A2 completes the repaired blinded QASPER error audit with full state text, validating all hashes, joins, and row counts (final unopened). Decided agreement is 66.4% (99/149) with 51 adjudication rows emitted. Disagreement is highly asymmetric: 44 cases challenge locked semantic-none labels (38 explicitly and 6 implicitly answerable; 28 with nonempty gold evidence attached), while 6 cases reflect state serialization deficits (omitted tables, captions without values, bibliography placeholders). The verdict establishes that raw state coverage does not guarantee answer-bearing representation coverage, and blocks Phase 4E-B training sweeps pending independent adjudication under a structured defect taxonomy and benchmark/representation repair. [E17–E22]

**24 September 2026 evidence addendum.** A3 preserves the independent 51-row adjudication and separates six later same-assistant follow-ups; A4 verifies frozen-state span candidates but has not completed source alignment or serializer repair. A separate user-authorized assistant review freezes a 27-case disposition and keeps 77 one-to-one annotation ties as locked-label diagnostics. No gold, state, evidence, A4 ledger, threshold, or final data changed. The exploratory seed-17 4E-B.0 probe tests a frozen root final-token/mean-pool descriptor with cheap pooled question and candidate embeddings. All three arms fail their non-final gates, with no accepted policy decisions and no separate result lock. This rejects that specific cheap-prefill readout, not prefill reuse as computation or every possible learned state-query architecture. Source-aligned repair and independent review of proposed corrections remain open before a new representation arm. [E23–E26; §18.15–18.16]

**Later 24 September option-logit audit.** A separate, locked 4E-B.1 diagnostic tests a JevK5-style next-token answer-letter readout on the pinned Base checkpoint and 325 non-final questions. It improves sampled two-option ContractNLI ranking over the historical Phase 4A reference, but the explicit `Z` action misses all 124 ContractNLI semantic-none cases. QASPER has one offered candidate and therefore tests answerability, where `Z` catches only 2/8 none cases. This does not select a new model, prompt, or threshold, and it does not change the Phase 4E-A repair gate. [E27; §18.17]

**Version 0.6.1 scope.** This update uses the latest delivered v0.6 paper and roadmap as its editing bases, not the older v0.3 paper or pre-H roadmap also present in the conversation. It adds the first `2ij.1.0` workbench report, whose overall status is **blocked**: preparation and a Qwen4B/L4 synthetic feature-equivalence probe completed, but independent review and the selection contract are not approved. All historical B–G/H measurement tables and 2H closeout remain unchanged.

Section 14 separates the probe from semantic quality, records a duplicate-question fixture mislabeled as unrelated-question addition, and identifies the exact review/target/manifest blockers. V3 checks saved records and source code without running the notebook or reading final-case annotations. No review is signed, no model is retrained and no Drive source is changed. [E10; I0; V3]

**Evidence boundary.** E8 remains the unchanged failed `2h.1.1` attempt and fitting/development source. E9 is the completed continuation with final predictions and an artifact lock; it supersedes the old final-pending status without rewriting the original failure. Completion is separate from numerical acceptance, task generality and deployment readiness. The earlier revision scopes remain in the source and revision registers. [E8; E9]

**Operational plan:** the synchronized [roadmap](../ROADMAP.md) keeps 2H-C1–C5 closed and preserves the blocked `2ij.1.0` review-gated checkpoint. E11–E13 remain the model-selection and Python/native handoff authorities; RUST1–RUST11 remain the native CPU execution and service record. E14–E20 provide the locked Phase 4A corpus and completed architecture, objective, head-only, and evidence-residual studies. E21–E25 record the audit sequence and its separate 51-row and 27-case populations. E26 rejects the tested cheap-prefill readout on non-final data. E27 records a bounded direct-logit comparison with unresolved semantic-none behavior. E28 records the calibrated ranking/rejection sweep: ContractNLI ranks well but over-rejects, and QASPER remains at the answerable-majority decision. Finish source-aligned table/caption/reference repair and review of proposed label/evidence corrections under a new versioned contract before any new representation-learning arm. Do not run seeds 42/123, open final, or promote a checkpoint. Practical high-K latency and separate MLX qualification remain systems work. CPU native parity does not imply Metal or accelerated parity. [E9–E28; RUST1–RUST11; P21–P23; recommendation]

**Phase 2H reading guide:** [Completed continuation](#1318-completed-continuation-lineage-lock-and-evaluated-scope) · [Final quality and rejection](#1319-selected-model-final-quality-rejection-and-population-weighting) · [All comparison arms](#13110-all-retained-comparisons-the-development-winner-is-not-the-best-final-transfer-arm) · [Source archive and audit](#appendix-a-source-and-reproducibility-register).

**New checkpoint reading guide:** [Workbench outcome](#14-phase-2i2j-workbench-preparation-mechanical-evidence-and-the-review-gate) · [Synthetic mechanics](#142-the-small-state-first-gpu-probe) · [Probe coverage correction](#143-coverage-correction-the-added-question-was-a-duplicate) · [Gate and continuation](#144-why-the-next-study-remains-blocked).

**Phase 4A–4E reading guide:** [Scope and status](#181-scope-status-and-evidence-boundary) · [Corpus and cache lock](#182-phase-4a0-corpus-partitions-and-frozen-feature-identity) · [Completed comparison](#184-completed-non-final-comparison-aggregate-improvement-hides-a-qasper-collapse) · [Scalar-weight closeout](#188-phase-4b2-cap-12-closes-the-scalar-weight-sweep) · [Stratified objective](#189-phase-4b21-stratified-applicability-improves-recall-but-not-policy-transfer) · [Pairwise objective](#1810-phase-4b3-pairwise-applicability-passes-development-but-not-transfer) · [Head-only and residual continuations](#1811-phase-4c-head-only-continuation-is-rejected) · [Phase 4E gate](#1813-lessons-learned-and-phase-4e-gate) · [Phase 4E-A2 repaired audit](#1814-phase-4e-a2-repaired-qasper-error-audit-reveals-asymmetric-annotation-and-serialization-defects) · [A3/A4 and 27-case disposition](#1815-audit-continuation-and-frozen-27-case-disposition) · [4E-B.0 prefill probe](#1816-exploratory-prefill-representation-probe) · [4E-B.1 option-logit audit](#1817-exploratory-option-logit-audit) · [4E-B.2 ranking/rejection sweep](#1818-exploratory-candidate-ranking-and-semantic-none-sweep).

**Next milestone:** close two independent tracks without mixing their evidence. On the model track, preserve the A3 51-row independent adjudication and separate six-row follow-up, complete A4 source alignment and versioned benchmark/serialization repair, and review proposed corrections independently of implementation before authorizing a new representation arm. The separate 27-case disposition, failed 4E-B.0 probe, and diagnostic 4E-B.1/4E-B.2 direct-logit runs do not authorize retraining, seed expansion, or final evaluation. On the systems track, practical high-K latency and separate MLX qualification remain open after the named CPU persistence and service gates. [E23–E28; RUST10–RUST11; proposed milestone]

---
