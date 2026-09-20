# OpenDecision
## A Qwen-led path to compact, multi-question decision models

### Abstract
OpenDecision treats decision inference as a scoring protocol rather than a text-generation task. A frozen Qwen3.5-4B-Base backbone produces features; small trained heads convert those features into typed answers and probability distributions. Completed Phases 2B–2G establish useful NLI and dynamic-candidate behavior, important rejection limits, and a strict-FP32 execution reference. Lossless prefix reuse and bounded FP16 attention-KV storage pass the reported sampled gates; four tested low-bit snapshot codecs do not. TF32-permitted batching offers a measured speed opportunity without full equivalence.

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

**Version 0.7.1 external-benchmark update.** Two new public sources sharpen the execution roadmap without changing the selected integration profile. A community Qwen Parallel Constrained Decoding implementation shows one-prefill/batched-field mechanics and author-reported Apple-Silicon latency, but does not establish TypeSafe-style RLCD training or calibrated probabilities. A DGX Spark comparison reports substantially flatter Jev Q=1→4 latency than sequential local Qwen wrappers, while also showing task-dependent quality rankings, independent probability-quality variation and service repeatability differences. These results are treated as external benchmarks and architectural prompts, not OpenDecision measurements or evidence of Jev internals. The immediate implementation emphasis therefore becomes native parity followed by branchable hybrid state, breadth-first batched question execution and explicit Q-amortization measurement. [P21; P22; recommendation]

**Version 0.7.2 Phase 3A systems update.** Phase 3A run `20260920T024056Z` keeps selected profile `a047d6802c3f06f085b8` and its bundle unchanged and performs no training or reselection. It validates a corrected full-hybrid-state branching reference for Qwen3.5, passes the notebook's semantic batched-parity and high-K systems-parity gates, and records exact same-process repeatability in the saved run. Batched state sharing is slightly slower on the three short semantic Q=4 cases. As shared state length grows, it becomes much faster and consumes more memory.

The result supports Rust `BranchableState` parity plus an adaptive scheduler. It does not support a model redesign or an “always share” optimization rule. The selected integration line is now publicly available at <https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>. Publication does not convert the exploratory profile into a release-quality model or establish Rust/Metal parity. [E12; PUB1]

**Version 0.7.2 Phase 3B and Rust checkpoint.** Run `20260920T152206Z` keeps the same profile, bundle, and base revision. It records no training, model selection, or model modification. It exports four exact token-fixture records. Its 47 FP32 vectors comprise 34 trace stages, 10 full-sequence candidate features, and 3 continuation vectors. The largest fresh-feature difference is `4.9591064453125e-05`. Cached continuation differs from fresh full-sequence execution by at most `1.9073486328125e-05`.

These values are Python self-consistency diagnostics under the notebook's `1e-4` guard. They are not new Rust acceptance tolerances. Rust now passes the selected head/probability, exact-token, full-sequence CPU backbone, and Qwen-specific cached-continuation gates. Across the native 34-stage trace, embedding is exact and final RMSNorm has maximum absolute error `5.8174e-05`. Across all 10 candidate sequences, maximum probability delta is `4.5869e-06`, with zero argmax or policy changes. The native cached candidate exactly matches the native full-sequence result while preserving the source root. [E13; RUST1–RUST3]

**Version 0.6.1 scope.** This update uses the latest delivered v0.6 paper and roadmap as its editing bases, not the older v0.3 paper or pre-H roadmap also present in the conversation. It adds the first `2ij.1.0` workbench report, whose overall status is **blocked**: preparation and a Qwen4B/L4 synthetic feature-equivalence probe completed, but independent review and the selection contract are not approved. All historical B–G/H measurement tables and 2H closeout remain unchanged.

Section 14 separates the probe from semantic quality, records a duplicate-question fixture mislabeled as unrelated-question addition, and identifies the exact review/target/manifest blockers. V3 checks saved records and source code without running the notebook or reading final-case annotations. No review is signed, no model is retrained and no Drive source is changed. [E10; I0; V3]

**Evidence boundary.** E8 remains the unchanged failed `2h.1.1` attempt and fitting/development source. E9 is the completed continuation with final predictions and an artifact lock; it supersedes the old final-pending status without rewriting the original failure. Completion is separate from numerical acceptance, task generality and deployment readiness. The earlier revision scopes remain in the source and revision registers. [E8; E9]

**Operational plan:** the synchronized [roadmap](../ROADMAP.md) keeps 2H-C1–C5 closed and preserves the blocked `2ij.1.0` review-gated checkpoint. E11 supplies the exploratory model-selection authority, E12 supplies the Python branch/batch reference, and E13 supplies the layer-localized backbone handoff. RUST1–RUST6 close the deterministic head, exact-token, correctness-first CPU backbone, Qwen-specific continuation, backend-neutral `BranchableState`, sequential nested execution, and batched Q/K gates against `a047d6802c3f06f085b8`. Adaptive scheduler measurement on the named Mac is next; CPU native parity does not imply Metal or accelerated parity. Independent review, natural-data confirmation, and explicit release-quality/resource limits remain promotion gates rather than prerequisites for the port. If later native Mac profiling cannot determine stable execution-strategy thresholds, Phase 3A.1 should create a dedicated scheduler/crossover notebook. [E9–E13; RUST1–RUST6; P21; P22; recommendation]

**Phase 2H reading guide:** [Completed continuation](#1318-completed-continuation-lineage-lock-and-evaluated-scope) · [Final quality and rejection](#1319-selected-model-final-quality-rejection-and-population-weighting) · [All comparison arms](#13110-all-retained-comparisons-the-development-winner-is-not-the-best-final-transfer-arm) · [Source archive and audit](#appendix-a-source-and-reproducibility-register).

**New checkpoint reading guide:** [Workbench outcome](#14-phase-2i2j-workbench-preparation-mechanical-evidence-and-the-review-gate) · [Synthetic mechanics](#142-the-small-state-first-gpu-probe) · [Probe coverage correction](#143-coverage-correction-the-added-question-was-a-duplicate) · [Gate and continuation](#144-why-the-next-study-remains-blocked).

**Next milestone:** measure Q-amortization, rejection, calibration, useful automation coverage, complete-request latency, and peak memory on the named Mac with the adaptive scheduler over the landed sequential and batched paths. The current 4B FP32 scorer is an integration reference, not a required shipping configuration. [E11–E13; RUST6; P21; P22; proposed milestone]

---

# 1. Executive assessment

**The central feasibility question has a positive, bounded answer:** a frozen Qwen backbone can support useful, non-generative decisions through small trained heads. The mechanism is simple: encode the input once, read a task-specific feature, and score typed alternatives without decoding answer text. The research does not establish an arbitrary-domain, fully Jev-compatible decision model. It establishes an experimental foundation and locates the main deployment risks and performance costs. [E1–E7]

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

**Revised direction after E11–E13 and RUST1–RUST3:** keep reviewed multi-question data and release confirmation open. E11 fixes a provisional state-first Qwen4B integration profile; E12 supplies the Python full-hybrid branching/batching reference and shows that execution strategy must be workload-adaptive; E13 and RUST1–RUST6 close the correctness-first CPU parity ladder through cached continuation, the backend-neutral branch-state contract, sequential nested execution, and batched Q/K execution. Target-Mac scheduler measurement is now the immediate engineering bottleneck, while independent review/natural-data confirmation, efficient external baselines and Track S service work continue in parallel. [E9–E13; RUST1–RUST6; R3; RC; recommendation]

Bring matched frozen-head versus LoRA versus Qwen3.5-2B comparisons forward, with a compact bidirectional dynamic-candidate arm in the same early comparison program. Keep Qwen as the leading research path without assuming the 4B model must ship or that a smaller encoder is automatically adequate. Measure the useful quality/resource trade-off, not agreement with every historical answer. Model selection and implementation equivalence are different decisions; Section 13 makes their gates explicit. [R3; S3; proposed program]

Preserve strict FP32, lossless reuse, and bounded FP16-KV storage as execution references; keep TF32 as a separately versioned performance candidate and the tested low-bit configurations outside the accepted-equivalence set. Retain inspected G examples as regression data, not an untouched final test after tuning. When attributing an effect, do not change prompts, kernels, precision, heads, policies, and caching simultaneously. None of the revised priorities retroactively changes the F/G verdicts. [E3–E7; recommendation]

**Workbench status now has two preserved stages.** The first `2ij.1.0` invocation remains a blocked review-gated checkpoint with no semantic training/final results; §14 preserves its mechanics evidence and blockers. The later `2ij.2.0` exploratory screen completed 13 fit jobs, evaluated 31 locked final profiles, measured bounded state-first sharing and exported the selected reference bundle. It does not erase the earlier review gate or convert the pilot into release-quality evidence. [E10; E11; I0; V3]

# 2. What “Jev-style” should mean

## 2.1 A software-facing contract, not a claim about hidden internals

TypeSafe documents Jev as a model that accepts a shared state and named typed questions, then returns structured decisions and probability distributions without free-text generation. Its published interface has three primitives: Choice, Score, and Noul. The documentation describes questions as evaluated in parallel and in isolation against the same state. These are the relevant behavioral targets for OpenDecision. They are not a public specification of Jev’s complete neural topology. [P1]

Choice returns a selected member of a supplied option set, probabilities, and a confidence statistic. The documentation reviewed for version 0.1 permits up to 255 options. Score returns a distribution over described levels and a probability-weighted mean of their level numbers; equal means can conceal different distributions. Noul returns the probability of a yes answer, without a separate confidence value. A Noul is not the project’s “none” class. [P2–P4]

TypeSafe describes confidence as a statistic derived from the returned distribution. It should not be silently equated with the maximum class probability or with a calibrated probability that an entire workflow is correct. OpenDecision should expose the full distribution and explicitly version any additional concentration statistic. Matching field names while changing the statistic’s meaning would create a misleading compatibility claim. [P5; design implication]

TypeSafe publicly identifies a new architecture, a parallel sampler, and Reinforcement Learning for Calibrated Decisions, or RLCD. The primary pages reviewed describe the objective and behavior, not enough algorithmic detail to establish that OpenDecision reproduces the training procedure. The appropriate claim is an independently designed, open-weight implementation of a similar decision interface. [P6; P7]

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

The intended scope is many well-defined judgments, not unrestricted long-form reasoning. TypeSafe’s public guidance favors atomic questions composed in application code. OpenDecision should therefore test whether changing the instruction changes the distinction being evaluated, while unrelated questions remain isolated. Supplying unseen intent names is useful evidence, but is not the same as following an unseen question or rubric. [P1; R3]

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

The completed measured sequence uses Qwen/Qwen3.5-4B-Base at revision `1001bb4d826a52d1f399e183466143f4da7b741b`. The text backbone has 4,205,751,296 parameters, hidden width 2,560, and 32 blocks. Its layer list contains 24 linear-attention and eight full-attention blocks. The core results were obtained on an NVIDIA L4. The saved environment includes Transformers 5.17.0; the expanded workers record PyTorch 2.11.0+cu128. Environment details should travel with results because kernel and precision behavior matter. [E1; E2; E5]

The notebook execution logs also report missing optimized causal-convolution and linear-attention kernels, with reference implementations used instead. Transformers documents these optimized versus reference paths. The recorded timings should therefore be treated as measurements of this particular stack, not the speed limit of Qwen on an L4. Installing faster kernels is a future experiment requiring both new timing and renewed probability/policy parity checks. [E5; E6; P16]

## 3.2 What was reviewed and what was not rerun

Earlier versions synthesized executed notebook snapshots, full result JSONs for Phases 2B–2F, summary documents, and relevant embedded source. Version 0.2 added the completed Phase 2F archive and matched its uploaded paste-back summary to Drive. That earlier audit reconstructed 1,664 Phase 2F probability/action comparisons, checked 384 storage records and 922 timing medians, and replayed 12 LRU traces under zero-expiry conditions; the expanded 2E outputs had also been re-aggregated. These historical audit scopes are retained rather than presented as new inference. [E5; E6]

Version 0.3 also read the completed Phase 2G archive and the two prior review companions. Its independent saved-output check reconstructed 6,960 distributions and 20,880 application-policy actions from retained candidate scores and coefficients, with maximum distribution reconstruction residual approximately 1.22 × 10⁻¹⁵. That v0.3 audit rechecked 24 full fresh-quality head summaries, 768 within-mode comparisons, 416 fresh and 72 controlled-context cross-mode comparisons, 16 request-time aggregates, and 12 trace totals. The prior G review separately checked broader summary records and replayed eight scalar LRU traces. No Qwen inference, training, GPU timing, annotation adjudication, or full external-reference re-review was performed for v0.3. Arithmetic agreement does not independently reproduce the backbone or validate author labels. [E7; R1; R2; v0.3 verification companion]

The two supplied ChatGPT share links resolved to their conversation titles but did not expose readable conversation bodies through the available web reader. Earlier project reports in the Library supplied additional framing; executable notebook code and saved result artifacts take priority over those reports. The source register records this limitation rather than implying that the shared transcripts were fully inspected. [S1; S2]

For v0.4, the supplied v0.3 Markdown was the authority for retained numerical results. The saved architecture-review note R3 was read in full, and the H0 notebook's methods, configuration defaults, and treatment descriptions were inspected. That Library snapshot had no executed code cells; it did not contradict the owner's report of a separate in-progress run. No H archive or live run was inspected for v0.4. S3 exposed its title only; R3 recovered the main review rather than every subsequent turn. Available conversation history supplied the later Laya/R4T recommendations, with narrow primary-source checks supporting Section 11.7. Those recommendations remain proposals, not OpenDecision measurements. [R3; H0; U1; S3; P18–P20; historical v0.4 scope]

The v0.4 integrity check compared retained tables and measured sections against its supplied source; it did not recompute E1–E7 experiment outputs. Earlier archive hashes and audit reports remain historical provenance, not work repeated for v0.5. [v0.4 revision manifest]

For v0.5, the attached v0.4 is the editing authority. The new H paste matches the archived compact summary after normalizing Markdown escapes and automatic URL wrapping. The audit reconciles 14 profile records, the restricted six-profile joint selection, 12 main-profile temperature-gate decisions, ten none-model selections, 126 policy-selection records and six primitive-seed records. It checks 37 archived implementation hashes and all 15 pairwise exact-group intersections among the six main split manifests. Two saved development fixtures are independently reconstructed using the exported head and none coefficients: maximum candidate-score residual is approximately 5.12 × 10⁻⁷ and probability residual approximately 4.70 × 10⁻⁹ under the audit's float64 arithmetic. These are limited saved-output/head-algebra checks, not Qwen inference or reproduction of all H development losses. The fitting feature/logit cache is absent from the ZIP. [E8; v0.5 audit companion]

For the original E8 snapshot, the traceback and archived worker source located the exception after fitting artifacts were written, during the final memory snapshot. That attempt contained no global final lock, evaluation-worker directory or final prediction file. The v0.5 audit did not inspect reserved final stimuli, repair or rerun the notebook, modify reservations or re-audit B–G. Those historical findings remain correct for E8; the separately completed E9 continuation now supplies the formerly missing outputs. [E8; E9]

For v0.5.1, the Library v0.5 paper and the uploaded roadmap are the editing bases. The saved Colab finish-cell summary is equal to the archive's compact summary; the separately retrieved full summary is byte-identical to the archive copy. A read-only checker reconciles all 14 profile development values and the six-profile restricted selection, verifies matching worker/profile copies, six primitive weight files and 126 nonfinal policy records, checks all 37 archived implementation hashes, and checks the retained inventory for final-lock/evaluation outputs. All 22 source-consistency checks pass; **the original attempt's status remained partial at that revision**. This is not an independent reconstruction of development loss or a repeat of the prior v0.5 fixture/overlap audit. No reserved final stimuli or unavailable live Colab runtime were inspected. [E8; R4; R5; V1]

Version 0.1 used a Phase 2F notebook snapshot retrieved around 22:59 UTC on 18 September, while the run was still executing. Version 0.2 supersedes that status with the completed run `20260918T224427722898Z`, whose result files were saved to Drive at approximately 23:56 UTC. Both precision workers and all requested stages completed. Completion does not imply that every codec passed its numerical or policy gates. The old notebook remains historical provenance; the completed archive is the Phase 2F result authority. [E6]

For v0.6, E9's standalone full and compact summaries match the corresponding archive members byte-for-byte. The original ZIP, locked fitting files and payloads retain their identities. V2 independently re-aggregates **32,256 final probability rows into 168 final quality records and 1,512 final policy records**, reproduces 26 paired final contrasts and the accuracy-bootstrap calculations, checks 216 within-mode parity rows and 14 cross-mode profile comparisons, and checks 1,024 reliability rows, 20 reliability quality records, 180 reliability policy records and 24 context-position records. It recomputes 12 primitive-seed metric records from 1,536 saved distributions and checks 80 benchmark-row medians from 240 saved timings. Policy interval values remain source-reported. The largest checked metric residual is about 2.22 × 10⁻¹⁶. These counts include repeated profiles, arithmetic modes and variants of shared source messages; they are not independent sample counts. [E9; V2]

V2 also verifies 39 implementation hashes, 37 locked fitting/runtime-source files, three locked payload hashes and 46 evaluation-output inventory hashes. The 24 locked fitting-directory files match the nested original attempt exactly. The completed final payload is read only to verify the already-evaluated IDs/labels and aggregate primitive results; source annotations are not re-adjudicated. Additional K/condition and reliability cross-mode tables are labeled saved-row aggregations. No Qwen inference, training, new timing, native-backend execution, complete security audit or historical B–G recomputation occurs. These checks support the reported arithmetic and lineage, not universal calibration or operational safety. [E9; V2]

For v0.6.1, E10's standalone compact summary matches its report-archive member byte-for-byte. V3 records **34 saved-artifact consistency checks**: status/gate/job agreement, recorded feature-threshold arithmetic, runtime-source identity, cache-family and exclusion inventories, three recorded model revision strings, unsigned/stale review metadata and the canonical protocol-body hash. It verifies 18 report files against the separately retrieved snapshot manifest; the SQLite backup and CPU-test XML are not materialized. The source reports 48 passed CPU tests, but this revision does not rerun them. The raw hidden vectors are absent from the compact archive, so the feature deltas remain source-reported. [E10; I0; V3]

The intake is reviewed as metadata only: protocol, unsigned review and refreshed manifest. The 92 case IDs are counted from their hash inventory; `cases.jsonl`, final questions and their annotations are not opened. The duplicate-question coverage finding comes from static inspection of the archived generator/call site. No model code is imported, no new GPU timing or H numerical reconstruction is performed, and no approval or persistent file is changed. The distinction between a prepared draft and a reviewed dataset remains intact. [I0; V3]

## 3.3 Evaluation units and leakage boundaries

MultiNLI splits were separated using normalized premise groups, not only individual rows. Phase 2C excluded 4,021 prior groups from its newly selected splits and used 2,400 training rows, 300 development rows, 1,000 calibration-fit rows, 500 calibration-gate rows, and 1,000 rows in each new test. Training examples remained the saved Phase 2B training set rather than a larger corpus. [E2]

Banking77 results require a different denominator. Several candidate-set episodes can come from one underlying message. Phase 2C’s 480 episodes per test split came from 160 messages; Phase 2E’s 1,024 episodes per split came from 64 messages. Expanded 2E and Phase 2F replay the same eight messages in a 128-episode factorial panel. Phase 2F selects 32 episodes for codec regression and 16 for short-request benchmarking; these subsets still span only eight messages. Its repeated-request traces do not add independent examples. Confidence intervals and uncertainty discussions must respect message clustering. [E2; E4–E6]

The 57/20 Banking label split withholds labels from candidate-head training, development selection, and calibration. It does not withhold them from Qwen’s pretraining. Those two label groups remain within banking; Phase 2G separately adds CLINC non-financial and author-OOS transfer panels. “Fresh” means new relative to the named recorded project-message manifests, not necessarily novel knowledge to the backbone. Repeating the same seeds reuses the same selected examples. [E2–E4; E7]

Phase 2G excludes 2,912 normalized prior Banking message hashes and constructs 416 fresh episodes from 112 messages. Its 56 parity episodes come from 16 messages; 72 controlled-context episodes transform six selected messages; eight episodes are timed independently; repeated 48-request traces supply no new semantic observations. Those units are kept distinct throughout Section 10. Once these G examples have been inspected to choose the next modeling intervention, they belong to historical/regression evidence rather than another untouched final evaluation. [E7; R2; methodological implication]

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

A service should return model/engine and policy version identifiers with telemetry, and report truncation and unsupported inputs explicitly. Most untouched task-quality inputs use a maximum length of 256. Phase 2G adds controlled labeled contexts through a minimum 1,024 state tokens, but their six source messages and generated administrative notes do not establish reliable decisions on natural long operational documents. Keep the synthetic mechanics and labeled context controls distinct. [E2–E7]

## 11.2 A Qwen-first extension, with a real compact-model comparator

**Incremental Qwen path.** Move from candidate-only sharing toward nested state → question → candidate execution. Compare state-first rendering with the current instruction-first contract under matched training. Refit the affected heads and normalization rather than assuming the old readout is optimal after token reordering. This is an early research priority, not a semantics-preserving cache patch. [E2–E7; R3; proposal]

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

**Compact bidirectional dynamic-candidate path.** Run a bounded joint-candidate encoder comparison in the same early model study as smaller Qwen and LoRA. It tests whether the task can be solved with much less model work, rather than committing to a second product architecture. It must pass the same held-out question/rubric and rejection tests; speed on a narrow classifier is not enough. Section 11.7 distinguishes a controlled model-family comparison from evaluating Laya’s released checkpoint as an external system. [R3; S3; proposal]

**Shared-representation/query-module path.** A purpose-trained shared encoder with isolated query modules remains a later architectural fork if the incremental paths miss the quality/resource target. Perceiver IO is a conceptual precedent, not an implementation of Jev or an automatic substitute for question-conditioned Qwen features. Do not start a large architectural rewrite before the simpler comparisons identify a need. [P11; R3; proposal]

No path should claim nearly flat question scaling from a candidate-only benchmark or a single batched call. Measure both Q and K, state length, complete latency, memory, and semantic isolation. [R3; proposed evaluation]

**Implementation progress after Phase 3A.** The original E10 workbench supplied only an early Q=2/K=2 mechanics probe and contained the duplicate-added-question fixture documented in §14.3. E11 later corrected the distinct-question test and measured bounded semantic sharing. E12 now goes further: it validates the selected profile's complete Python hybrid-state fan-out/select path, breadth-first question/candidate batching, high-K systems parity and same-process repeatability. That closes the Python execution-reference question sufficiently for native engineering; independently reviewed semantic generality and natural-document quality remain separate promotion gates. [E10–E12; V3]

## 11.3 Rust and Apple Silicon: define the parity ladder

Saved head fixtures make a Rust port more tractable, but fixture export is not execution parity. The earlier exports and E/F/G head replays do not validate a Rust backbone. H contributes selected-head development fixtures and now completed CUDA final predictions and bounded same-mode parity results. They provide source material for the per-profile parity ladder, not a Rust/Metal implementation result. [E2; E3; E5–E9]

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

Add a separate **state/evidence sufficiency** axis. Deliberately omitted candidates, author-OOS, insufficient evidence, an observation that simply does not expose the required fact, and an application decision to escalate are not the same event. The DGX Doom study is useful motivation: changing the textual observation/history representation materially changed controller outcomes even though the underlying game task was unchanged. OpenDecision should include reviewed cases where the requested answer is not recoverable from the visible state, and should measure how richer but still non-leaking state summaries affect both correctness and confidence. This is a dataset/evaluation requirement, not evidence that the current profile already solves partial observability. [P22; proposed dataset extension]

For example, one security-event state could support a question about the affected asset, another about whether external communication is evidenced, and a third about which documented handling category applies. This is an illustrative training design, not a claim that the current model handles security workflows. Deterministic calculations and authorization remain in application code. The same-state examples make instruction sensitivity testable without confusing it with a change in underlying facts. [R3; proposed task construction]

The seven-parameter none model is a useful ablation, not a universal answerability mechanism. It sees symmetric score summaries rather than all the semantic information in the already-computed hidden representations. Identical summaries necessarily yield identical none outputs; changing a none logit cannot repair the ordering of two offered candidates. Compare it with a small, permutation-aware applicability/rejection head that can consume candidate-conditioned features, without presuming that another backbone pass is necessary. Keep the refitted-constant control. [E3; R3; architectural inference and proposal]

A candidate experimental factorization for a single-choice task is:

```text
a = P(at least one offered option is valid | state, question, candidates)
r_j = P(candidate j is the correct choice | an offered option is valid,
        state, question, candidates)
P(none) = 1 − a
P(candidate j) = a × r_j, with sum_j r_j = 1
```

This factorization does not manufacture better evidence or calibration. Its value must come from supervision and representations that distinguish applicable candidates, omitted correct options, author-OOS inputs, and insufficient evidence. Keep those evaluation strata separate; application review remains a policy, not a semantic class. Specify how ambiguous or multiply valid alternatives are annotated before fitting a single-choice distribution. [R3; proposed model and annotation contract]

In the matched study, fit frozen multi-task heads, adapt the same backbone with limited LoRA, and apply the same recipe to a smaller Qwen. Refit normalization and calibration when representations change. Match losses, pairs, update budgets, and selection criteria within causal comparisons; report unmatched pretrained systems separately. Use multiple seeds for leading configurations and reserve a new final evaluation after using G or completed H findings to design the next intervention. [R3; H0; proposal]

## 11.7 Lessons from compact encoders and teacher-to-student research

**Laya: borrow a testable design, not its claimed generality.** The author model card describes a fully fine-tuned 395M ModernBERT-large backbone plus a decision head, approximately 421M parameters in total. Options are scored at marker positions within a per-question input, with a 512-token question/options/state budget. It reports multi-question batching and limits its calibration claims to the evaluated distributions. These are author descriptions, not OpenDecision results or an independent checkpoint audit. [P19]

The useful comparison is joint candidate scoring with a compact bidirectional model and broader task supervision. Put a freshly trained compact arm beside the current Qwen control, adapted Qwen, and smaller Qwen using common data and task definitions. Evaluate released Laya separately as an external baseline: its training history and budget are not matched. Do not treat batching as proof of state-once execution, silently truncate away evidence to meet its input budget, equate a confidence statistic with correctness, or import its training/calibration pipeline without evaluation. This revision adopts neither Jev-superiority claims nor a presumption that encoder size determines generality. [S3; P19; proposed comparison and interpretation]

**R4T: the transferable idea is offline supervision for a cheap student.** Retrieve-for-Train uses an RL-trained fan-out policy to produce objective-aligned supervision for a lightweight retrieval model; the paper’s application is set-valued retrieval, not typed decision calibration. Its diffusion retriever and reported retrieval speedups are not evidence that OpenDecision should switch to diffusion or will obtain the same benefit. [P20]

After establishing the supervised baseline, compare the same student trained on reviewed labels alone, labels plus reviewed teacher-generated examples, and an additional distribution-distillation treatment. Keep criteria, data splits, candidate ordering, none semantics, and budgets explicit; never replace independent final labels with teacher judgments. A 4B model is only a possible teacher, not automatically a better one. Evaluate student correctness, calibration, rejection, policy cost/coverage, and total resource use; agreement with an overconfident teacher is not enough. Direct supervised training remains first, and a large RL program needs a specific failure that simpler losses do not address. These are proposed OpenDecision adaptations of the discussion, not results established by R4T. [R3; S3; proposal]

**Backlog traceability:** the compact comparison is an early **2J.1–2J.2** treatment, with released Laya evaluated separately under **2J.5**; the teacher/student comparison is the restored conditional **P2.3** study. The historical `OpenDecision_Review_Followup_Traceability.md` review matrix retains their distinct tests and non-goals under RQ-06 and RQ-13. These are experimental commitments to compare, not commitments to replace Qwen or to deploy diffusion. [RC; RP2]

### 11.7.1 Parallel constrained decoding and external Q-scaling evidence

Two new public implementations/benchmarks help separate the useful execution ideas from stronger architectural claims.

**Community Qwen Parallel Constrained Decoding (PCD).** The public `harshatheg/Qwen-2.5-1B-RLCD` page describes an inference engine that prefills a context plus semantic schema catalog once, broadcasts the decoder cache across fields, slices logits to valid answer tokens, performs limited continuation for multi-token candidate trees and assembles JSON in host code. Its model card reports Apple Silicon M4 Max timings of roughly 68–75 ms for four-field examples, 270 ms for 28 fields and 89 ms for one 255-choice example, with 5.6–7.0× latency reductions relative to its own autoregressive JSON baseline. Those are author-reported measurements on Qwen2.5-1.5B/MLX configurations, not OpenDecision results. The public artifact does **not** document a TypeSafe-style reinforcement-learning procedure; its useful contribution here is an execution pattern. Labeling a softmax over candidate logits “calibrated” does not establish empirical calibration. [P21]

The public PCD rendering also exposes an important architectural distinction. Its shared prefix includes the semantic schema/question catalog before the context. Therefore changing, adding or reordering fields can change the cached representation that precedes every branch. That is efficient but weaker than OpenDecision's intended isolation contract. The selected OpenDecision profile is **state-first**: the shared state root should be independent of the set of questions submitted with it, then isolated question branches should be created from that root. The design lesson is therefore **borrow breadth-first branch batching, not schema-conditioned state representation**.

```text
Schema-first PCD reference:
    schema/question catalog + state → shared decoder cache → field branches

OpenDecision selected direction:
    stable format + state → immutable hybrid root
        ├─ question batch Q1..Qn → isolated question states
        │      ├─ candidate branches → scores
        │      └─ ...
        └─ per-question rejection/distribution/policy
```

For Qwen3.5, the root and every fork must include the full hybrid continuation state, not only attention keys/values: full-attention KV, DeltaNet recurrent state, convolution state, positions and profile identity. A Rust abstraction should therefore model **branchable execution state**, not assume a generic `KVCache` is sufficient. This also keeps the runtime interface usable by future architectures whose reusable state is not represented as ordinary decoder KV. [E4–E7; E10; E11; architectural recommendation]

**DGX Spark decision-reader benchmark.** Reddy's September 2026 comparison reports whole-request p50/p95 and question throughput for one versus four questions. In that campaign, Jev 1.13 moved from **105.1 ms p50 at Q=1 to 109.2 ms at Q=4** (~1.04×), Laya moved from **16.4 to 29.1 ms** (~1.77×), while the tuned Qwen3.5 wrapper moved from **167.0 to 665.1 ms** (~3.98×). The absolute values are not apples-to-apples: local readers are warm on a DGX Spark while Jev includes a hosted HTTP round trip. The useful signal is the *within-system slope*: a Jev-style engine should make additional independent questions much cheaper than Q repeated full evaluations when shared-state work dominates. [P22]

That benchmark also supports three methodological conclusions already present in OpenDecision. First, architecture rankings are task-dependent: the tuned compact encoder leads the reported WANLI slice while Jev leads the reported BoolQ slice, so a fast encoder should not be promoted from one narrow benchmark. Second, probability quality must be evaluated independently: the study reports separate Brier and ECE values rather than treating typed output or maximum probability as calibration. Third, repeatability is an observable serving property: identical Jev API requests changed some choices and many WANLI probability vectors between campaigns. This does not identify the cause, but it motivates a pinned local repeatability campaign for OpenDecision rather than assuming version labels imply identical numerical behavior. [P22]

**Implication for the selected OpenDecision profile.** Do not restart model selection. E11 already provides a fixed integration target and a verified bundle. The immediate systems experiment is now sharper: reproduce the profile natively, establish sequential nested parity, then evaluate a **batched question fork** where all compatible question suffixes advance breadth-first from one immutable state root. Candidate branches can be batched at the next level. Compare repeated-full, sequential-nested and batched-nested paths under the same model/profile and report `T(Q)/T(1)`, marginal latency per added question, questions/second, forward-call count, state-prefill share and peak branch-state bytes. External Jev ratios are reference context, not a predeclared OpenDecision release threshold. [E11; P21; P22; recommendation]

High-cardinality Choice should likewise be split into systems and semantic questions. P21 shows that a constrained-token implementation can mechanically handle 255 declared choices at useful latency on its own workload; it does not show 255-way dynamic semantic accuracy. OpenDecision should first stress K=32/64/128/255 for memory, batching and scheduler behavior, then add a smaller reviewed high-K semantic panel with candidate descriptions and rejection cases. [P21; recommendation]

## 11.8 Resolve the wire contract and build a thin real-service path

Two semantic issues should be settled before advertising Jev compatibility. First, the research distribution includes internal semantic none. A wire format restricted to caller-defined options must specify where that mass goes. TypeSafe’s Choice guide recommends an explicit caller-supplied other/none alternative. Do not append an unrequested key or discard none and renormalize while presenting the resulting probabilities as unchanged unconditional probabilities. Choose and version the native result and compatibility-adapter behavior. [E2–E7; P2; R3; recommendation]

Second, TypeSafe documents option names and descriptions as semantic input, while question IDs are hidden. The current OpenDecision scorer deliberately hides candidate IDs. Separate machine IDs from semantic labels/descriptions internally; a compatibility adapter must supply meaning when a caller uses a semantic option name with a null description. This is a documented contract difference, not a defect proven in an inspected Rust implementation. Pin accepted schemas and conformance fixtures rather than assuming identical field names imply identical behavior. [P2; R3]

A proposed integration bridge is:

```text
Rust HTTP boundary → bounded worker queue → resident Python reference worker
                  → real decision probabilities → validated typed response
```

This is **Track S** in the synchronized roadmap and should arrive before a complete native-backbone port, not replace that port. Start with one resident model and one bounded queue; declare supported primitives, input limits, engine/policy versions, deadlines, cancellation behavior, and token accounting. Unsupported questions must fail explicitly rather than return mock judgments. Separate cold-model startup from warm requests, and cache misses from hits. Measure queueing in service latency rather than reporting only GPU work. [R3; proposed prototype]

Require protected or private metrics, redaction of sensitive inputs, bounded request work, and explicit opt-in before unauthenticated non-loopback serving. These are prototype requirements, not claims that the current service has been audited or has particular vulnerabilities. Preserve the native parity ladder in Section 11.3; a working bridge exposes interface and scheduling problems without claiming Rust/Metal inference parity or production readiness. [R3; recommendation]

## 11.9 One current contract and status authority

The recovered review reported drift across agent instructions, roadmap phases, probability terminology and test-count snapshots. Version 0.5.1 synchronizes **this paper and the attached roadmap**, but does not edit or audit `AGENTS.md`, architecture files, Rust source or CI. The roadmap is the current task/status authority; this paper retains experiment interpretation and provenance. Repository instructions must adopt the same task IDs and supported contracts under Track S.1. Use commit-stamped CI results for current test counts rather than duplicated “at HEAD” prose. The supplied roadmap's 195-test snapshot has no exact commit and was not rerun here. [R3; R4; V1]

Record H as **completed through a separately identified continuation**, with the original failed attempt preserved. The roadmap closes 2H-C1–C5 and keeps independent criteria review, richer applicability, multi-question work and the service bridge under their own task IDs. This does not mark `AGENTS.md`, repository CI, HTTP integration or native inference complete. Candidate probability remains distinct from the separate confidence statistic; all historical memory formulas retain their model-specific layer factors. [E8; E9; V2; roadmap coordination]

# 12. Research questions answered and still open

| Research question | Answer supported so far |
|---|---|
| Must useful decisions be generated as text? | No. Frozen features plus small heads work on the measured tasks. |
| Does a base model suffice? | It suffices for these probes; Base versus Instruct was not controlled here. |
| Is a linear head enough? | It is a strong selected NLI baseline. More complex heads were not consistently necessary in this setting. |
| Do unseen candidate labels work? | Yes within bounded tests, but H’s preselected support arm has substantially weaker held-out than fitting-family results. Held-out omission remains distinct from the all-correct sampled author-OOS panel. [E9] |
| Is calibration automatic after supervised fitting? | No. Post-hoc scaling can worsen held-out metrics, and risk changes under new costs and prevalence. |
| Can a tiny none model help? | Yes in controlled comparisons. H now adds final refitting/readout evidence, but score-summary rejection does not settle applicability, missing evidence or policy risk. [E3; E9] |
| Does an isolated repeat prove deterministic deployment? | No. Padding, batching, precision, and cache chunking changed results. |
| Is FP32 the most accurate model? | Not established generally. It is the most internally consistent tested reference; numerical and semantic quality are evaluated separately. |
| Does TF32 permission preserve the decision contract? | Not fully. H’s selected profile has no final argmax/policy changes but two numeric-tolerance failures; additional controlled-context cross-mode failures remain. Historical G failures are unchanged. [E7; E9; V2] |
| Can shared prefixes amortize work? | Yes, especially with longer prefixes and batched suffixes in the completed mechanical tests. |
| Has state-once, arbitrary-question-many execution been demonstrated? | **Mechanically, in bounded Python systems tests:** E11/E12 validate nested and batched state-first execution for the selected profile, including semantic smoke parity and a Q=16 mechanics grid. This is not independently reviewed arbitrary-question semantic competence or native/Rust evidence. [E11; E12] |
| Is TurboQuant already a win for this model? | Not under the frozen equivalence gate: all four tested snapshot codecs failed; short-prefix overhead also limited storage benefit. |
| Does FP16 KV storage help? | It passed strict-FP32 fresh/context gates in G and earlier storage tests; it saved bytes but did not add a cache hit or beat lossless trace time in G. |
| Does persistent prefix reuse help? | Yes on controlled single-worker traces: F and G show workload-dependent gains, with G exercising expiry; no production concurrency claim follows. |
| Are clear criteria and calibrated rejection solved? | No. H’s completed support/refitting study still has held-out omission, context and policy limitations; independent criteria review was not performed. [E9] |
| Does MTP speed up this decision path? | Not in its present no-output-decoding graph. |
| Are Score, Noul, and API parity finished? | H’s bounded BoolQ and SST-5 final probes are complete. General binary/rubric semantics, insufficient-evidence handling and real-service/API parity remain open. [E9] |
| Has Rust/Metal or a smaller Qwen been validated? | No completed full-backbone comparison in the reviewed evidence. H's smaller-Qwen, LoRA and state-first arms were disabled in the actual run. |
| Has Jev’s RLCD been reproduced? | No. The project has a distinct, inspectable research path toward a similar software interface. |
| Is the frozen 4B scorer the required deployment model? | No. It is the measured reference; matched adaptation and compact-model selection are now early priorities. |
| Must a newly trained model reproduce all old probabilities? | No. Same-model implementation equivalence and fresh new-model quality have separate gates. |
| Does a batched multi-question call prove state-once computation? | No. Q-scaling, actual shared work, isolation, and semantic quality must be measured separately. |
| Does the public Qwen `RLCD` repository reproduce TypeSafe RLCD training? | Not from the reviewed public artifact. Its useful evidence is parallel constrained-decoding/cache-broadcast execution; a softmax over candidate logits is not by itself a calibration study. [P21] |
| What new systems property should OpenDecision target after nested sharing? | **Adaptive question amortization.** E12 shows sharing is workload-dependent: short semantic requests can lose to repeated-full, while long shared-state mechanics gain strongly. Preserve multiple execution strategies and learn the target-machine crossover instead of assuming one universally fast path. [E12; P22] |
| Should OpenDecision switch to schema-first PCD? | No. Keep the selected state-first root for stronger question-set isolation; borrow breadth-first branch batching and constrained-token baselines where useful. [P21; E11] |
| Does Laya establish a better replacement, or R4T establish a diffusion decision architecture? | No. They motivate bounded comparisons and a later distillation study, not completed OpenDecision findings. |
| Has Phase 2H completed? | Yes: both required `2h.1.2` continuation workers completed and 2H-C1–C5 are closed. The original `2h.1.1` failed attempt is unchanged; completion does not promote a model or arithmetic mode. [E8; E9] |

**Current 2I/2J status:** the original `2ij.1.0` reviewed-study path remains blocked by unsigned review, null promotion bounds and stale review metadata; separately, `2ij.2.0` completed an **exploratory** model-selection screen and exported a provisional integration profile. The pilot is sufficient to start native parity work but does not satisfy the independent-review/natural-data release gate. [E10; E11; I0]

# 13. Refocused research program and next milestone

The revised objective remains a compact, instruction-sensitive, multi-question decision model rather than a permanently frozen 4B scorer. E11 and E12 change the sequencing, however: a provisional profile and Python execution reference now exist, so the next engineering budget moves to native/Rust parity and target-machine execution while reviewed model-promotion work proceeds independently. The accumulated Python numerical/branching evidence becomes the contract the native implementation must reproduce, not a reason to defer the port for another model-search cycle. [E11; E12; R3; proposed program]

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

The required H study is closed and the exploratory E11 comparison has already supplied the provisional implementation profile. The **immediate engineering priority is Phase 3 native parity** against that immutable profile and the E12 execution fixtures. In parallel, **2I.1–2I.2 and 2J.5–2J.6** remain the release-quality path: independent review, natural/held-out task evidence, explicit promotion bounds and fair external baselines. New model interventions still require fresh development/final data; H and E11 final sets are historical/regression evidence once they influence design. [E9; E11; E12; R3; RC]

State-first/nested Q sharing, independently reviewed natural documents and criteria, broad Noul/Score evidence, and the real Rust/resident-worker service remain open. The Laya comparator and later R4T-inspired teacher/student work remain captured in §11.7, P2.1–P2.3 and the traceability matrix. The refocus toward a compact, instruction-sensitive, multi-question engine is preserved; the 4B FP32 scorer is still a reference rather than a mandatory shipping model. [R3; RC; E9; proposed program]


## 13.2 Two acceptance tracks, not one universal parity rule

| Track | Question being tested | Required comparison | What does not count as success |
|---|---|---|---|
| Implementation equivalence | Does an optimization or backend preserve the specified model? | Same finalized input, weights, rendering, heads, calibration and policy; compare full probabilities, argmax, and directed policy outputs | A faster path or unchanged aggregate accuracy that fails the declared same-model gate |
| New-model quality | Does a new representation, head, size, input contract, or trained operating point improve useful decisions? | Fresh labeled task/rubric families; predeclared quality, risk/coverage and resource requirements; its own versioned execution reference | Mere agreement with the old model, a favorable inspected test, or throughput obtained by reviewing everything |

The historical 0.005 probability tolerance and no-outcome/no-policy-change requirements retain their meaning for E/F/G. No failed codec or TF32 comparison is retroactively promoted. A new trained model may legitimately change old answers, including correcting old errors; it should not be rejected solely for doing so. Once selected, its implementations must still meet their declared stability and equivalence requirements. A new-model label is not permission to ignore batching drift or deployment risk. [E5–E7; R3; proposed acceptance policy]

## 13.3 Priority order and synchronized roadmap

The companion [ROADMAP.md](../ROADMAP.md) is the task/status authority; this paper is the experimental-evidence and interpretation authority. The same package IDs are used below. They separate the **actual H study and its closeout** from **new modeling and service proposals**. A notebook run cannot mark a service or a different model experiment complete. [R4; E8; proposed coordination contract]

| Priority | Roadmap package | Decision it enables | Evidence required before promotion |
|---|---|---|---|
| Closed — required H scope | **2H-C1–C5: preservation, recovery, locked final evaluation and evidence handoff** | Adds complete final evidence without retraining or erasing the failed attempt | E9/V2; no automatic model, TF32 or deployment promotion |
| P0 — now | **Phase 3.8: adaptive scheduler and Mac Q-amortization; 3.1–3.7 complete** | Converts the landed execution paths and E12 reference into measured scheduling decisions | Measured `repeated_full`/`nested_sequential`/`nested_batched` crossover, Q-amortization, and branch-memory evidence [E12; E13; RUST6] |
| P0 — parallel contract work | **S.1–S.2: current status and probability/API contracts** | Prevents different components from implementing different meanings | Pinned decision/execution/capability contracts; explicit none/key semantics; commit-stamped test evidence |
| P1 — native optimization | **Phase 3.8–3.10: adaptive scheduler, high-K/repeatability** | Establishes whether the selected profile amortizes work on the target Mac | Measured crossover/Q-amortization, branch memory and repeatability on the named Mac against E12 reference [RUST6] |
| P1 — release confirmation in parallel | **2I.1–2I.2 + 2J.5–2J.6** | Determines whether the provisional profile may be promoted beyond exploratory scope | Independent review, natural/held-out cases, explicit release limits, fair external baselines and fresh final evidence [E10; I0; E11] |
| P1 — service path in parallel | **S.3–S.5: thin Rust service plus resident reference worker** | Exposes API, queueing, capability and resource limits during native work | Real probabilities, bounded queue/work, unsupported-input errors, cancellation/failure tests, queue-inclusive latency/memory |
| P2 — conditional | **P2.1–P2.3: optimized-kernel, weight-precision and teacher/distillation studies** | Tests a specific remaining bottleneck after the native baseline | Verified runtime capabilities; fresh quality when behavior changes; same-model parity when equivalence is claimed |

**Dependency order after RUST3.** H closeout and the exploratory 2I/2J screen remain complete enough for implementation. Head/tokenizer/backbone and Qwen-specific continuation parity now pass against the frozen exported profile **without waiting for release promotion**, while 2I.1–2I.2 independent review/natural-data work continues as the release-quality gate. Establish backend-neutral branch isolation and sequential nested sharing next, then batch the question layer and candidate layer separately so numerical or semantic regressions can be localized. Use fresh reviewed data for model promotion; use the exported E11/E13 fixtures for implementation equivalence. [E9; E11–E13; RUST1–RUST3; P21; P22; proposed sequencing]

**Observed gate:** E10 has generated the review package but has not satisfied that dependency. Resolve the exact intake requirements in §14.4–§14.5 before starting new semantic model selection. Correct the duplicate/unrelated fixture under 2I.6 as separately versioned test work; this checkpoint does not justify closing the full isolation task. [E10; I0; V3]

The compact encoder belongs in the early model-selection round. A purpose-built shared-encoder/query-module redesign remains conditional on the simpler approaches missing the target. The actual H run disabled state-first, smaller-Qwen and LoRA arms; future use of their code must have an explicit new treatment/configuration rather than retroactively expanding H's scope. [E8; H0; S3]

**Recovered-review audit trail:** the historical `OpenDecision_Review_Followup_Traceability.md` file maps the recovered discussion to this priority table and to concrete roadmap tasks. Restored **P2.1** verifies optimized execution, **P2.2** evaluates weight precision for a selected profile, and **P2.3** tests offline teacher/student supervision after a supervised baseline. This restores earlier task detail without changing the priority order.

### 13.3.1 Correct the phase-name collision without claiming work was done

The uploaded roadmap calls Phase 2H “Contract Hardening, Criteria Review, Feature Rejection & Prototype Bridge” and marks it NEXT. The actual H notebook instead performed criteria/rejection-transfer fitting with bounded primitive probes. Both refer to useful work, but they are not the same milestone. The synchronized roadmap restores the experiment's name and relocates the open proposals explicitly. [R4; E8]

| Previous roadmap task | Current location | Why it remains separate from H's saved results |
|---|---|---|
| Old 2H.1 — current contracts and frozen regression suite | **S.1; 2H-C1** | Existing artifacts are evidence, not proof that repository instructions and recovery are synchronized |
| Old 2H.2 — independent criteria/annotation review | **2I.2** | H supplied training examples without independent review; keep its frozen criteria unchanged for its original evaluation |
| Old 2H.3 — feature-conditioned rejection | **2J.3** | H refitted score-summary none models; it did not validate a hidden-feature applicability head |
| Old 2H.4 — resident Python worker bridge | **S.3–S.5** | Colab fitting is not HTTP integration or service-lifecycle validation |
| Old 2H.5 — none/key/API contract | **S.2** | A probability vector and a compatible wire response require an explicit mapping |
| Old 2J.3 — multi-task supervision | **2I.1–2I.2** | The common data/evaluation contract must precede the matched comparison, not arrive afterward |

These are task migrations, not erased requirements or executed improvements. The roadmap also retains a crosswalk for renumbered 2I rendering/sharing tasks. Its reported **195-test** code snapshot lacks an exact commit in the attachment; that historical total is not promoted to a verified current-HEAD result. No repository build or test was run for this update. [R4; V1]

### 13.3.2 Stage exits and limits on promotion

**H closeout is satisfied for the declared required scope.** E9 provides traceable recovery, unchanged fitting artifacts and final payload, a checked lock, both completed evaluation workers, full saved outputs and this evidence handoff. E8 remains failed/partial as history. Numerical failures and mixed quality do not reopen execution bookkeeping; independent review, new models, Q-level sharing and deployment claims remain separate open work. [E8; E9; V2]

**The 2I/2J model milestone** requires reviewed task definitions, independent state/question/rubric holdouts, per-stratum rejection and distribution metrics, useful coverage and measured resource limits. Feature-aware rejection is an ablation against the constant and score-summary controls, not a preselected replacement. Actual parameter/memory accounting must replace the old roadmap's approximate halving of 4B memory for a 2B candidate. [R4; proposed comparison gate]

**The Track S prototype** requires a real supported profile returning probabilities behind the existing service boundary, with bounded work and truthful capability errors. It may use the historical reference for integration without calling that model production-ready. The native Phase 3 gate applies to the selected model and backend's real capabilities; it neither requires every proposed backend nor imposes Qwen-specific hybrid-cache machinery on a different architecture. Historical fixture tolerances and prefix-storage observations are not universal new-model acceptance thresholds. [R4; proposed integration/native gates]


## 13.4 A concrete multi-question milestone

Show several independently defined questions over one state through a real, versioned engine. Begin with supported Choice judgments and clearly bounded binary/rubric tasks rather than claiming all primitive semantics are solved. Use held-out question/rubric families to test transfer, and distinguish answering Q questions correctly from encoding their shared state once. Both properties matter; neither proves the other. [R3; proposed milestone]

A proposed initial scaling grid is **Q = 1, 4, 16** and **K = 2, 4, 8, 16**, over proposed state-length targets **64, 256 and 1,024 tokens** and supported question types. Actual finalized token counts include question/criteria/formatting overhead; these are initial protocol targets, not established limits. It is a design grid, not a completed benchmark or required maximum API size. Hold other dimensions fixed for causal comparisons and use representative subsets before an expensive full sweep. Test question-order changes, opaque-ID renaming, candidate permutation, and adding/removing unrelated questions. Separate expected changes from changing a candidate set from unintended cross-question influence. [R3; proposed protocol]

Report correctness, NLL/Brier and reliability diagnostics, conditional rejection rates, accepted-answer error, coverage, and scenario-defined application cost. Report independent source-state counts alongside all expanded episode/question counts and use source-group-aware uncertainty. For performance, include complete latency, decisions/questions per second, peak and resident memory, model invocations, branch-state bytes, state-prefill share, **`T(Q)/T(1)` and marginal latency per added question**. Compare repeated-full, sequential-nested and batched-nested execution on the same profile. Separate startup, cold-state, warm-state/cache-hit, and queueing costs; do not mix L4 GPU measurements, DGX Spark measurements or hosted Jev timings with unmeasured Mac performance. [R3; P22; proposed measurement contract]

The preferred selection criterion is **correct automated decisions per second subject to predeclared accepted-error, coverage, latency, and memory requirements**. Set the numerical requirements and target hardware before looking at the final comparison; this paper does not invent an evidence-free safety threshold or resource winner. Always reviewing is a cost baseline, not a useful high-throughput success. At zero accepted answers, conditional accepted-error is undefined. A model should advance only with an explicit task scope and enough evidence to support its stated operating region. [R3; proposed selection rule]

## 13.5 Work to pause or keep conditional

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

To continue the intended study, use the existing `OpenDecision_Phase2IJ_review` intake. Finalize the cases and task/holdout definitions, specify the device and five quality/resource limits, then run the notebook's revision-resolution/review-template refresh step. Have a distinct reviewer inspect and sign the matching case/protocol manifest and protocol attestations. Only then rerun the same study ID, provided it is still unregistered; after registration, changed approved inputs require a new study identity. Preserve the blocked report as a dated snapshot rather than overwriting it with a claim of completion. These are the existing workflow requirements, not authorization to bypass them. [I0; E10 source workflow]

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

<https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>

This publication improves inspectability and handoff. It does not imply that the frozen Qwen base weights were retrained, that TypeSafe RLCD was reproduced, or that OpenDecision has passed a release-quality/natural-data or Rust/Metal gate. [PUB1]

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

A successful Rust `BranchableState` implementation may use different tensor layouts, copy-on-write storage, preallocated slabs, gather kernels, MLX/Metal-native buffers, or another backend representation. Equivalence is judged by finalized input identity, full probabilities, argmax/policy behavior, branch isolation, and declared numeric tolerances. It is not judged by whether Rust mimics Python data structures.

The public Hugging Face repository should be treated as the public identity of this reference line, while the immutable profile ID and bundle hash remain the stronger experiment identifiers. A future LoRA, weight-quantized, different-renderer, different-rejection, or otherwise behavior-changing artifact must receive a distinct model/execution profile and its own quality/equivalence record. [E11; E12; PUB1]

At the current Rust checkpoint, steps 1 through 8 pass for the frozen fixtures on the correctness-first CPU path: the native path verifies both checkpoint shards, executes embedding, all 32 decoder blocks, final RMSNorm, candidate features, and Qwen-specific cached continuation; the backend-neutral branch contract holds root immutability, fork, fan-out, and gather; sequential nested execution matches the full-sequence oracle exactly, and batched Q/K execution is exactly equal to the sequential baseline. Step 9 — measured strategy crossover on the named Mac — remains open, as do Metal, service integration, and release promotion. [E13; RUST1–RUST6]

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

The current Rust implementation establishes nine bounded results:

1. The immutable `ModelExecutionProfile` records the selected identity, calibration, policy, and numerical tolerances. It does not change `DecisionEngine`.
2. The selected score-summary head reproduces the exported golden probabilities, selection, semantic-none behavior, and review policy with f64 host-side algebra.
3. The digest-locked offline tokenizer reproduces all four exported token records exactly. The Phase 3B loader validates the architecture and 426-entry parameter inventory. It also validates layer order, 47 tensor records and hashes, continuation positions, and candidate-feature probability replay.
4. The native embedding path verifies the immutable checkpoint config and safetensors index, the 5.3 GB embedding shard's size and SHA-256, the BF16 `[248320, 2560]` tensor layout, token bounds, finite values, and BF16-to-FP32 widening. The diagnostic sequence's last-token row matches `diagnostic.embedding` exactly: maximum absolute error 0, RMS error 0, cosine similarity 1. [RUST2]
5. The Candle 0.8.0 CPU path verifies the second checkpoint shard and executes the frozen sequence through 24 DeltaNet blocks, 8 full-attention blocks, and final RMSNorm in FP32. Across the 34-stage trace, embedding is exact and final RMSNorm has maximum absolute error `5.8174e-05`, RMS `1.1531e-05`, and cosine `0.999999999993`. Across 10 candidate sequences, maximum feature delta is `1.0300e-04`, maximum probability delta is `4.5869e-06`, and there are zero argmax or policy changes. Hidden-state deltas remain localization diagnostics, not a new acceptance tolerance. [RUST3]
6. The Qwen-specific continuation state contains full-attention KV, DeltaNet recurrent state, convolution state, and absolute position. The exported branch preserves positions `98 → 109 → 121`, accounts for exactly `59,899,904` root-state bytes, leaves the root unchanged, and produces a candidate vector exactly equal to this native implementation's independent full-sequence path. Its maximum absolute delta from the saved A100 cached candidate is `6.8665e-05`. [RUST3]

7. The backend-neutral `BranchableState`/`BranchBatch` contract lifts the complete Qwen continuation state behind profile/model/tokenizer/renderer/arithmetic identity, process-local lineage, a structural scheduling fingerprint plus a strict content fingerprint, exact attention-KV/recurrent/convolution/metadata byte accounting, immutable-root fork, batched fan-out, and gather/select. The checkpoint-gated native branch stage replays the Phase 3B `root → question → candidate` fixtures through the contract at positions `98 → 109 → 121` with the exact `59,899,904`-byte root accounting (`6,422,528` attention-KV, `50,331,648` recurrent, `3,145,728` convolution tensor bytes), root immutability under fork/fan-out/gather, batch lanes that advance independently and exactly match single-fork replay, preserved lane identity under reorder/duplicate gather, and a cached continuation state exactly equal to the independent full-sequence state (maximum absolute delta `0.0`). [RUST4]

8. Sequential nested `state → question → candidate` execution prefills each fixture case's shared state exactly once, then evaluates every question and candidate through immutable `BranchableState` forks with fail-closed position and immutability verification. The checkpoint-gated native nested stage executes all four Phase 3B questions and all 10 candidates: maximum probability delta against the golden head fixtures is `4.5869e-06` under the unchanged `0.005` contract, with zero argmax and zero policy changes. The retained root equals an independent fresh prefill exactly after all fork work, question and candidate replay from the retained fork states is exact, reversed sibling-order advancement is exact, and every position matches the exported token fixtures. The `repeated_full` oracle agrees exactly: cached-versus-full candidate features are `0.0` for all 10 candidates and every cached continuation state's strict fingerprint equals its independent full-sequence state. Feature deltas against the frozen Phase 3B vectors reach `1.0300e-04`, and the traced continuation vectors replay at root `1.1444e-04`, question `7.8201e-05`, and candidate `6.8665e-05` — the same localization diagnostics recorded by results 5–7, not new tolerances. [RUST5]

9. Breadth-first batched Q/K execution fans one immutable prefilled root into `Q` question lanes with `BranchableState::fork_batch`, advances the question suffixes breadth-first, and fans each question state into `K` candidate lanes with fail-closed root, sibling-lane, and position verification after every stage. The checkpoint-gated native batched stage executes the same four Phase 3B questions and 10 candidates — case 0 fans 3 question lanes holding exactly `3 × 59,899,904 = 179,699,712` root bytes and case 1 fans 1 lane at the `76,087,296`-byte 345-token root — and every batched question and candidate feature equals the sequential baseline of result 8 exactly (`0.0`) with identical strict state fingerprints. The frozen head reaches the same `4.5869e-06` maximum probability delta with zero argmax changes, zero policy changes, and zero decision changes between the sequential and batched strategies. The retained root stays exactly identical to an independent fresh prefill, and offline tests pin gather/reorder equivalence. Per-lane executor calls remain the execution primitive; length-bucketed vectorized suffix kernels and the adaptive scheduler remain open. [RUST6]

The `1e-5` absolute-logit tolerance remains the Phase 3.1 fixed-feature algebra gate. Phase 3B's own freshly exported candidate vectors replay against the earlier fixed-feature logit fixture with maximum absolute delta `9.6905e-05`; the native features reach `2.5652e-04` against that older fixture while preserving probabilities and discrete decisions. Therefore the backbone result is judged by the documented Phase 3B hidden diagnostics plus probability, argmax, and policy gates; no tolerance is widened or retroactively redefined. [E13; RUST1; RUST3]

The next gate is the adaptive scheduler and target-Mac measurement (Phase 3.8): `repeated_full`, `nested_sequential`, and `nested_batched` are now three parity-proven native strategies over the landed branch contract, and the remaining question is which strategy wins for which measured state length, Q, K, and cache warmth on the named Mac. **CPU native parity now passes; Metal/native accelerated parity does not yet follow from that result.** Vectorized suffix kernels, Metal, backend registration, and wire-level semantic-none mapping remain open. [E12; E13; RUST4; RUST5; RUST6]

---

# Conclusion

OpenDecision’s completed evidence supports a bounded mechanism: encode with Qwen, score typed decisions with small heads, and treat execution precision as part of the model contract. The evidence includes frozen-feature task results, dynamic candidate transfer, measured rejection trade-offs, cost-sensitive policies, strict-FP32 shared-prefix agreement, and bounded cache-compression and persistence findings. Phase 2G adds stronger execution evidence and clearer limits in rejection, criteria interpretation, and background reliability. These limits do not invalidate Qwen. They show that the narrowly fitted frozen scorer is not the finished model. [E1–E7; R3]

**The revised direction keeps Qwen as the lead while preserving model comparison.** It requires better-defined judgments, richer answerability evidence, a credible smaller-Qwen challenger, and genuine independent questions over shared state. Completed H contributes transfer and rejection warnings. E11 adds the bounded matched model screen and fixes a state-first Qwen4B integration target.

The next engineering step is to reproduce that profile natively, then vectorize the validated nested graph at the question layer. Independent reviewed and natural-data confirmation, efficient external baselines, and later teacher/student work remain promotion and research gates. They are not reasons to postpone the port. [E9; E11; P21; P22; recommendation]

The next defensible claim is neither “an open Jev clone” nor “the old FP32 scorer reproduced more precisely.” It is an open, auditable, versioned decision model. The model should answer several well-scoped questions over one state with measured quality, uncertainty, useful coverage, and resource cost. Preserve historical equivalence gates and evaluate new models on fresh quality gates. Those comparisons should determine what ships, not an assumed architecture or model size. [R3; proposed milestone]

The first 2I/2J workbench invocation remains useful historical evidence of a blocked review gate and an early branching probe. The later exploratory screen supplies the trained comparison that the first invocation lacked, but does not retroactively approve the unsigned review package. Preserve both stages: use E11 for the provisional implementation target and E10/I0 for the still-open independent-review requirements. [E10; E11; I0; V3]

The later `2ij.2.0` continuation completed the bounded comparison and fixed the provisional integration candidate. Phase 3A adds the Python execution reference. Its corrected cache-wide path supports complete hybrid-state fan-out and selection. Semantic and high-K parity gates pass, and the saved run repeats exactly in its recorded same-process check.

Phase 3B adds exact tokens and stage-localized backbone vectors without changing the model. Rust now passes the head/probability, exact-token, CPU full-sequence backbone, Qwen-specific cached-continuation, backend-neutral branch-state, sequential nested execution, and batched Q/K gates for the frozen fixtures.

The performance result remains mixed. Short semantic Q=4 requests favor repeated-full execution. Longer shared state produces large batched-sharing gains with higher peak memory.

The next implementation sequence is **adaptive Mac scheduler → high-K/repeatability → production lifecycle**. Each gate depends on the preceding gate. Another model-architecture search is not the next step. A fresh independently reviewed confirmation remains required before a release-quality model claim. [E11–E13; RUST1–RUST6; PUB1]

---

# Appendix A. Source and reproducibility register

The source IDs below identify the evidence behind the numbered sections. In the accompanying evidence manifest, local snapshot SHA-256 hashes distinguish the exact files reviewed from later Drive edits. Result paths are under `Google Drive / Colab Notebooks`. Timestamps embedded in run IDs are UTC.

**Historical version 0.2 audit boundary.** The completed Phase 2F archive SHA-256 is `e1d3fea878d35b9e929e83424d4f3aa8b5de0385bb100ace001110f7e5f83c37`. The version 0.2 companion retained version 0.1 evidence and added the Phase 2F manifest, compact results, and an independent standard-library aggregation audit. Probability and policy comparisons are recomputed where raw vectors/actions are retained. Request/long-prefix rows retain comparison diagnostics rather than all output vectors, so their numerical gates remain source-reported while timing aggregates are independently checked. No source Drive files were modified. [E6]

**Historical version 0.3 audit boundary.** The attached v0.2 source SHA-256 is `1adad381998535be157cc8c3806427ff7db1e8e21b7cd05afad5129969942cc2`. The completed G archive SHA-256 is `24ff15fa1fec3c8eeb17ef073da3d852cabe40683678b247d4e1d917e847414b`. The v0.3 verification companion includes a standalone saved-output checking script, its aggregate JSON, this revision’s source/hash manifest, the earlier correction and G review notes, a change log, and a text diff. The script does not download or run Qwen. Section 3.2 retains the checks rerun for v0.3; the earlier wider audit is attributed to R2. All v0.3 numerical additions are from the supplied/completed artifacts, not new external research. No original attachment, notebook, source result, or Drive file was modified.

**Historical version 0.4 revision boundary.** The supplied v0.3 Markdown SHA-256 was `6d73d8ea4e36249b98629997768fe5ec35c32de744dfd09670efd1cf701b0e08`. That revision preserved completed-result tables and checked textual integrity, source hashes, and status consistency. It did not rerun the v0.3 saved-output checker or independently reopen the B–G archives. Its companion contained a change log, unified diff and source/integrity manifest, not a new experiment audit. H0 and R3 were read-only working copies; persistent sources were unchanged.

**Historical version 0.5 revision and audit boundary.** The supplied v0.4 Markdown SHA-256 is `122da05cc4c4668b03a0cdc8fabfc65903fd7a40f72a8a2e4c030dfdbd756ec4`. The H archive SHA-256 is `51c772ab248d8949fe73bbbd91056826975eebb3594d03ca693438355d9a4cda`. Version 0.5 preserved the strategic framing and completed B–G numeric tables, added the partial H readout, and updated its status throughout. The companion contains the complete whitepaper, change log, unified diff, source hashes, a runnable NumPy/standard-library saved-artifact checker, its output, and selected metadata/source snapshots. The checker verifies the bounded scope stated in Section 3.2; it does not run Qwen, reconstruct unavailable development logits, open final stimuli, or validate held-out H results. No source notebook, run status, reservation registry, original attachment or persistent Drive file is changed. [E8]

**Version 0.5.1 synchronization boundary.** The editing bases are R4 (uploaded roadmap) and R5 (newer Library whitepaper v0.5). The same E8 run/archive is rechecked directly against the saved Colab finish output; this does not supersede its partial status or add a final result. The V1 checker records 22 source-consistency checks and leaves model computation, final-data inspection, previous numerical audits, source repair and repository tests outside scope. The revision package includes exact diffs, hashes, a source-reconciliation report and a document-integrity check. No source notebook, failed-attempt status, reservation registry, original attachment or persistent Drive file was changed.

**R4: Uploaded roadmap and current task migration.** `ROADMAP(20260919-123538).md`, conversation file `file_00000000657881f5b537b1e3162383be`. Supplies code-milestone descriptions, historical test counts and the conflicting proposed Phase 2H name. It is not independently verified repository state. The historical revision 0.5.1 of the companion roadmap restored the measured H experiment, adds its development/closeout boundary, and maps uncompleted proposals to 2I/2J/Track S/Phase 3. Source and output hashes are retained in the companion manifest.

**R5: Prior whitepaper v0.5 and its evidence package.** Library files `file_00000000fe4c81f5b521d6540f042f85` (`OpenDecision_Whitepaper_v0.5.md`) and `file_00000000d21881f59c548cdabcd28071` (`OpenDecision_Whitepaper_v0.5_Evidence.zip`). The Library Markdown is byte-identical to the paper in that evidence package. Its completed B–G and partial H numerical tables are retained; its earlier audit results remain attributed to v0.5 rather than represented as work rerun for v0.5.1.

**V1: Saved-source and document reconciliation for v0.5.1.** `check_saved_sources.py`, `evidence/source_reconciliation.json`, `evidence/document_integrity.json`, source/output hash manifest and exact diffs in the accompanying revision package. The check reads named metadata/report/code artifacts without executing experiment modules or opening `final_payload_reserved.json`. It verifies consistency of retained records, not the generalization validity of development-selected models or the existence of any unsaved live Colab continuation.

**H0: Phase 2H design snapshot, not completed evidence.** `OpenDecision_Phase2H_Criteria_Rejection_Multidomain.ipynb`, version `2h.1.0`, Library file `file_00000000044481f585f71c2a526ebdc9`, snapshot version 1; SHA-256 `05906ff8aafc1add2fdd1f818cc077cdf0ea9b2582254b1cb9c2de67f2e698d6`. Reviewed notebook methods, configuration cell, treatment descriptions, final-data controls, and stated boundaries. The snapshot has 30 cells and no executed code cells. The default configuration enables the main/TF32/primitive/finite-token studies and disables state-first, smaller-model, and LoRA extensions. These defaults do not establish the configuration or progress of the owner’s active run; embedded code was not executed or comprehensively audited.

H0 remains a historical design source. E8 records the original `2h.1.1` configuration, fitting outputs and failure; E9 supplies the completed `2h.1.2` continuation. Default flags, notebook availability and earlier owner status do not substitute for these versioned outcomes.

**U1: Project-owner status and revision request.** In the request accompanying v0.3, Chris states that Phase 2H is in progress and asks to retain it while refocusing the whitepaper on feedback in S3. This supplies status and editing intent, not run completion, enabled-arm details, or measurements.

**R3: Recovered architecture-review note.** `Pasted markdown(20260919-032946).md`, Library file `file_00000000c58881f5997fb51ee11dbef0`, snapshot version 1; SHA-256 `27b1ab204729db4d490ad8acd72aa563cd2aa1c330ccf8596acc33680405e129`. Read in full. Supplies the central refocus, separate equivalence/new-model tracks, multi-question and smaller-Qwen priorities, evidence-aware rejection proposal, nested-sharing design, API concerns, prototype service path, and documentation-drift observations. It reviews documentation rather than confirming current Rust implementation or test results. It is strategy feedback, not a new experimental artifact.

**E0: Initial feasibility probe.** `OpenDecision_Phase2_Qwen3_5_4B_Probe.ipynb`. Executed exploratory notebook; model-loading comparison, pooling shapes, two-example head fit, and rough generation timing. [Open notebook](https://colab.research.google.com/drive/1bd1FNFL7FP0fRxBLg17XW9yz2kEXHNh8).

**E1: Phase 2B.** Run `20260917T205849Z`; `OpenDecision_Phase2B_results / <run> / opendecision_phase2b_summary.json`. Principal fields: `architecture`, `data`, `all_metrics`, `matched_test`, `matched_uncalibrated`, `performance`, `generation_comparison`, `batch_invariance`, `frozen_export`. [Full result](https://drive.google.com/file/d/1Kj6Ph12DYU8VwRHnQ4-vTgA4Q2f2RmmS/view).

**E2: Phase 2C.** Run `20260917T222948Z`; `OpenDecision_Phase2C_results / <run> / opendecision_phase2c_summary.json`. Principal fields: `nli_data`, `nli`, `stability`, `dynamic.data`, `dynamic.training`, `dynamic.evaluations`, `dynamic.robustness`, `dynamic.export`, `frozen_export`. [Full result](https://drive.google.com/file/d/1lXy-mCtjkPkgb3KZR8fsn0mRKEZNB4E9/view). [Executed notebook](https://colab.research.google.com/drive/1iYTfdtn1-G26zpNyK98iodp_IGBi1YAz).

**E3: Phase 2D.** Run `20260917T234417Z`; `OpenDecision_Phase2D_results / <run> / opendecision_phase2d_summary.json`. Principal fields: `numerics`, `dynamic_data`, `selection`, `temperature`, `evaluation`, `request_benchmark`, `export`. [Full result](https://drive.google.com/file/d/12EDUXCzU3psTCDN-qTkkKTiRpxvnzkMX/view). [Executed notebook](https://colab.research.google.com/drive/1N_V3MnrZUbddO5j-ZQqDew_dEjXig6p1).

**E4: Phase 2E, original systems/policy run.** Run `20260918T032049180933Z`; `OpenDecision_Phase2E_results / <run> / opendecision_phase2e_summary.json`. Principal fields: `data`, `precision.modes`, `shared_prefix`, `policies`, `policy_scoring`, `run_status`. Overall status is partial; completed policies remain valid evidence within their scope. [Full result](https://drive.google.com/file/d/1JjBbFdsFHjjXqdzBkFpigwZ9QhDWWgzq/view).

**E5: Expanded Phase 2E.** Run `20260918T114914072764Z`; `OpenDecision_Phase2E_expanded_results / <run> / opendecision_phase2e_expanded_summary.json`. Principal fields: `sampling`, `workers.<mode>.parity_rows`, `parity_summary`, `benchmark_summary`, `long_prefix`, `memory_summary`, `cross_precision`. Completed execution; BF16 strategy-equivalence gates fail. [Full result](https://drive.google.com/file/d/1SxOG4VY4TlqK0e4eqBgZ_KwfDYjbXeFX/view). [Notebook containing the expanded run](https://colab.research.google.com/drive/17fuU04ZwOc2RJdahFiIyFaJ88YBgcvk8).

**E6: Completed Phase 2F.** Run `20260918T224427722898Z`, version 2f.1.0; results saved at approximately 23:56 UTC on 18 September 2026. Source folder: `OpenDecision_Phase2F_results / <run>`. Both precision workers completed. Principal fields: `baseline_parity`, `compression_rows`, `benchmark_rows`, `component_profiles`, `cross_request_summary`, `long_prefix_rows`, `cross_precision`. The archive also retains raw `cross_request_rows.json`, inputs, source modules, and frozen head exports. [Full result](https://drive.google.com/file/d/1jKdR5xozQ7CQfzaTT82OQ7bszBEfJGPe/view). [Paste-back summary](https://drive.google.com/file/d/1UJMv80N6vQ_bgdQH4ejZfIztj8xHZ8TS/view). [Complete archive](https://drive.google.com/file/d/1m9PbuQqFlO4e3ilEpaSJ4Qvf8VcW56_-/view). [Notebook](https://colab.research.google.com/drive/1DmkUfMAUwgnn4vWg_kWJe2cK75QVr60G). The version 0.1 partial notebook snapshot remains historical evidence, not the current run status.

**E7: Completed Phase 2G.** Run `20260919T005142584348Z`, version `2g.1.0`; results saved around 01:45 UTC on 19 September 2026. Source folder: `OpenDecision_Phase2G_results / <run>`. Both `fp32_strict_math` and `fp32_tf32_allowed` workers completed. Principal fields: `data`, `workers.<mode>.fresh_summary`, `context_summary`, `context_by_length_position`, `benchmark_rows`, `traffic_summary`, `flags`, `memory_snapshots`, and `cross_precision`. The archive retains `experiment_inputs.json`, `fresh_message_manifest.json`, `prior_exclusions.json`, worker `fresh_rows.json`, `context_rows.json`, `traffic_rows.json`, `traffic_events.json`, `benchmark_rows.json`, implementation snapshots, and frozen coefficients. [Full result](https://drive.google.com/file/d/1liuu456rvPFIQJXonRjxh83DTdJjKUGE/view). [Paste-back summary](https://drive.google.com/file/d/1F5vaHSj2Grix-RNChoTn2mhOWhet7pb2/view). [Complete archive](https://drive.google.com/file/d/1CAm4ooAuQ8gsxJAZNHn7jsnztnkF9IzP/view). [Notebook](https://colab.research.google.com/drive/1KVEB2apgM_8GIFdjLsX0x9LPvc94Pnzg). Freshness is relative to recorded project manifests; completion and strategy acceptance remain distinct.

**E8: Partial Phase 2H, saved fitting/development evidence.** Run `20260919T040612625670Z`, version `2h.1.1`; outputs exported around 05:03 UTC on 19 September 2026. Source folder: `OpenDecision_Phase2H_results / <run>`. Overall status `partial`; `fit_qwen4b` status `failed`, exit code 2, nonfatal `NameError: name 'memory_snapshot' is not defined`. The archive saves 14 main development profiles, 126 policy-selection records, six primitive fits, sampled weight-integrity status and two development-head fixtures. Principal artifacts: `paste_back_summary.json`, `opendecision_phase2h_summary.json`, `fit_qwen4b/worker_report.json`, `profiles.json`, `selection.json`, `traceback.txt`, `split_manifest.json`, fitting payload, source snapshots and fitted `.safetensors` files. No final-evaluation worker or final lock is present. [Full result](https://drive.google.com/file/d/1IT4cJN74vgOW0td7haE_bl2KfviHiaP1/view). [Compact summary](https://drive.google.com/file/d/1525L3h-0hVKtgm50c_IAX1dCAKHNZNUe/view). [Attempt archive](https://drive.google.com/file/d/1X8JP-8hhb3lu_PmMCWNLovMMdXPK6eDO/view). [Notebook](https://colab.research.google.com/drive/1fhRJTek7Ura3aSdItwBjJTJubXUIee4a). These references identify the inspected attempt, not a later repaired run.

**E9: Completed Phase 2H continuation.** Run `20260919T040612625670Z__finish_2h_1_2`, version `2h.1.2`, originating from E8. Both `eval_qwen4b_strict` and `eval_qwen4b_tf32` completed; outputs were saved around 14:45 UTC on 19 September 2026. Source path: `Google Drive / Colab Notebooks / OpenDecision_Phase2H_Finish_results / 20260919T040612625670Z__finish_2h_1_2`. [Full summary](https://drive.google.com/file/d/1rhO2B5ro3rQxEGJ3ZSt6JhflluV7-NPz/view). [Compact summary](https://drive.google.com/file/d/1072Mee3vu6JFjMXADPBe-GWm_BhCbGGE/view). [Completed continuation archive](https://drive.google.com/file/d/1kSnKjnOinF6fvyc-ZO9UF3rbfNT5eMCR/view). [Fitted-artifact recovery](https://drive.google.com/file/d/139rnY0E9EK6lIXrXDE0PhmSESZrPNiIu/view). [Final lock](https://drive.google.com/file/d/15nnlgaO__RDAzVQOKh96LzF9XvIBWkeG/view). [Finish notebook](https://colab.research.google.com/drive/1FAhX21Es0LsDYLS_CotbwsR1OsxrXAUS). The archive SHA-256 is `7df9de857e4683386f4b687244e0bbc28a4377855d5d696d1699cb973e76fa8f`; its preserved original-attempt ZIP hash is `51c772ab248d8949fe73bbbd91056826975eebb3594d03ca693438355d9a4cda`; final-lock SHA-256 is `3b30ced9af995812aeb23db12fdedfb7ba3247324c184e56871dd9e4108c3d14`. The source contains per-worker reports, final/reliability prediction rows, parity rows, benchmark rows, primitive predictions, fitted artifacts, final lock, input manifests and source snapshots. Its compact archive excludes the evaluation feature databases, which the source says belong to worker checkpoints. Later Drive modifications are not implicitly part of this snapshot.

**V2: Version 0.6 saved-output and lineage audit.** `audit_phase2h_finish.py`, `completed_h_audit.json`, `check_additional_lineage.py`, `additional_lineage.json`, source/output hash manifest and document-integrity report in the v0.6 companion. The checker uses NumPy and the standard library and does not import or execute experiment modules. It reproduces the metric/point-policy/accuracy-bootstrap/paired-contrast and parity arithmetic stated in §3.2, preserves source-reported policy intervals, checks hashes and unchanged recovered fits, and labels additional K/condition/reliability cross-mode aggregations separately. No training, Qwen inference, new GPU timing, annotation adjudication, original notebook alteration or native-service test was performed. Final labels are used only as the archived targets for already-completed evaluation. E9 and V2 support required-study closure, not universal scientific or deployment acceptance.

**Version 0.6 editing lineage.** Base whitepaper v0.5.2 SHA-256 `38353636f6c100fb2a4d133ba6e663af9a84723bbccfd27a9594a34e53654dd4`; base roadmap SHA-256 `6823105da8d38d548936c888f5a58dde156a3f6fdb50775da6d10d3032558ed5`. Historical B–G and H-development numeric tables are retained in the whitepaper. The roadmap intentionally replaces its detailed H-development tables with a concise completed-study summary and links to §13.1. That is a documentation consolidation, not deletion or alteration of the recorded experiment. The 16-group review-to-work/test map and all open 2I/2J/Track S/P2 requirements remain.

**H primitive dataset identities.** The saved preparation resolves BoolQ `google/boolq` to `35b264d03638db9f4ce671b711558bf7ff0f80d5` and `SetFit/sst5` to `e51bdcd8cd3a30da231967c1a249ba59361279a3`. BoolQ's final source partition is validation; SST-5's is test. One BoolQ validation row is removed by the explicit length rule (3,270 before, 3,269 after), while the recorded SST-5 counts are unchanged by that filter. These are dataset/preparation identities and eligibility counts, not final prediction results or an independent review of the annotations. The H criteria SHA-256 is `311a0e1db256686c246f4efb9ce352e8e8b94864dc0ca54b9077793a10e07646`; independent review remains not performed. [E8]

**R1: Prior whitepaper correction memo.** `OpenDecision_Whitepaper_v0.2_Review.md`. Identifies the probability-versus-argmax correction, segmented-tokenization correction, and refitted-constant attribution clarification applied in Sections 5.2, 8.1, and 5.3. The file is retained in the v0.3 evidence bundle.

**R2: Prior Phase 2G saved-result review.** `OpenDecision_Phase2G_Result_Review.md` and `OpenDecision_Phase2G_Review_Audit.json`. Preserve the independent distribution/policy reconstruction, source-label/omission checks, context aggregation, ambiguity case, and scalar cache replay scope. The v0.3 checker independently rechecks its stated arithmetic subset; neither review claims new model inference or label adjudication.

**Pinned datasets.** MultiNLI revision `da70db2af9d09693783c3320c4249840212ee221`; Banking77 resolved revision `90d4e2ee5521c04fc1488f065b8b083658768c57`. Banking CSV hashes retained in the results are train `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` and test `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`. These identify downloaded data, not the contents of Qwen’s pretraining corpus.

**Phase 2G dataset extension.** The pinned CLINC `oos-eval` revision is `828f8093932c8fe6ca7936c3d2e52903b1c523de`. Recorded SHA-256 hashes are `data/data_full.json`: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`; `data/domains.json`: `b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a`; and `LICENSE`: `e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76`. G’s prior-exclusion manifest SHA-256 is `f3549213c79dc4530dd8d04ecf61e5a6f08b8ba3e470a3a00d8296bc7bc5b42a`; its fresh panel hash is `367d31772bdf2b9630766272f3b7252131b7caf5f7e2f692da8d4ce4d3a5ad7d`. These are recorded download/selection identities, not a new corpus or annotation audit. [E7]


**S1: User-supplied discussion.** “Open Source Inference Plan,” [shared conversation](https://chatgpt.com/share/6aadc0b1-db54-83ea-856a-efc569000843). Title resolved; conversation body was not available to the web reader.

**S2: User-supplied foundation discussion.** “Jev Architecture Overview,” [shared conversation](https://chatgpt.com/share/6aadc0ec-8ee8-83ea-9343-b16fb73dbdc3). Title resolved; conversation body was not available to the web reader. Earlier Library reports supplied secondary context, not replacement evidence for measured results.

**S3: Current refocus discussion.** “Review Qwen Architecture,” [shared conversation](https://chatgpt.com/share/6aae15d6-c970-83ea-abbf-156c6dc734dc). The live page exposed its title but not readable conversation text. R3 recovers the main architecture-review response. Later R4T/Laya recommendations were available through retrieved conversation history, not a complete independently readable transcript; v0.4 used them as proposed direction and checked the narrow external descriptions against P18–P20; those historical descriptions are retained here without a new external review. It does not claim verbatim access to the entire shared conversation or repeat unverified third-party benchmark/provenance allegations.

**RC: Version 0.5.2 review-to-work mapping.** `OpenDecision_Review_Followup_Traceability.md` and the preserved `sources/Recovered_Architecture_Review.md` capture the recovered main review and the follow-up recommendations retained in §11.7. The main review SHA-256 is `27b1ab204729db4d490ad8acd72aa563cd2aa1c330ccf8596acc33680405e129`. The matrix is a source-based future-work/test cross-reference, not a complete transcript, a new experiment or an outside-fact audit.

**RP2: Restored conditional backlog.** The previous roadmap v0.4 revision package retains explicit P2.1–P2.3 tasks; v0.5.1 summarized those topics without their standalone checkboxes. The synchronized roadmap restores the original conditional section and maps it to §§11.7 and 13.3–13.5. The exact restored excerpt is included in the revision package.

**HR: Historical recovery-notebook deliverable.** `OpenDecision_Phase2H_Recovery_Notes.md` records the previously delivered `2h.1.2` Finish notebook, original-notebook repair and local/saved-artifact checks. It remains a repair-provenance source, not the final evaluation authority. E9 supplies the completed continuation and V2 its read-only saved-output audit. This documentation update does not run or modify either notebook.

**E10: First Phase 2I/2J workbench checkpoint, blocked overall.** Study `2ij_reviewed_multiquestion_v1`, version `2ij.1.0`, saved around 16:10 UTC on 19 September 2026. [Compact summary](https://drive.google.com/file/d/11X6HKs18bjjgGTDKbQwTe8PWK4nLXwfh/view), [report](https://drive.google.com/file/d/15rtftQ3WwGY_1WU1smLfAM0kXVfk5QAX/view), [report archive](https://drive.google.com/file/d/1bIyJfim0n-XsphCIEZu11cs8NEb90kO8/view), [mechanics result](https://drive.google.com/file/d/178mr-qS7CARw6qgfxGqZpghOvfjohmf_/view), and [workbench notebook](https://colab.research.google.com/drive/1BeeurQlJf0BJ_fV0lA-_7aZbXDhcgLQn). Archive SHA-256: `a12c0e4c00f24bfab3e1fbdcd99991cafad5f478753e995e44be58640a162143`. The archive has 29 files (1,180,633 uncompressed bytes), including saved stage/gate reports, source snapshots, CPU-test log, mechanics job/runtime/results and historical-reference audit. It contains no registered study, new fitted profile or semantic final result. Its quoted “unrelated” test retains the original name; §14.3 records the actual duplicate fixture.

**E10 snapshot boundary.** [LATEST pointer](https://drive.google.com/file/d/1392d9KjhWkxd-nCFww1-9q1dr6yeLVcS/view) identifies `snapshot-1789834208243222126`; its [completion manifest](https://drive.google.com/file/d/10oN9cUW3P0dafKi9qQNAPoOfJum8FyGp/view) SHA-256 is `328b99c2d09f506a959b83b86dbfa85dc948f6c5fd09fbde8575c2715391f860`. V3 checks the 18 manifest entries overlapping the compact report. The two additional entries, `mechanics/features.sqlite.backup` and `validation/cpu_tests.xml`, were not materialized or validated. Snapshot/report export does not imply approved semantic study completion or demonstrate future recovery.

**I0: Unsigned 2IJ intake metadata, separate from the report archive.** Under `Google Drive / Colab Notebooks / OpenDecision_Phase2IJ_review`: [protocol](https://drive.google.com/file/d/1zDdBv1JtGFTzMKYZshx-T8olFj5dVM9c/view), [unsigned review](https://drive.google.com/file/d/1_eIzpj9722xYDLBOmBb5lG1j0fng-rqO/view), [refreshed review template](https://drive.google.com/file/d/1rk2jLwBnI_nsgJ9xXwqylOfTmC5GJsvk/view), and [review instructions](https://drive.google.com/file/d/1En_2qNQBx1Jo21a8OrAFmwHv0bH2KcZn/view). The metadata snapshots have blank identities/false approval, a null target and five null quality/resource bounds. V3 checks the current canonical protocol-body digest `30239d9a2699b9ebf1bde047a9c9b453b64c01244079801d56512364c74c4791` against the refreshed template and records the stale unsigned review. Cases are counted only from the hash manifest; this revision does not open or adjudicate `cases.jsonl` or final annotations. Later edits are not implicitly included in this snapshot.

**V3: Version 0.6.1 saved-record and source-coverage checks.** `audit_2ij_checkpoint.py`, `evidence/saved_artifact_audit.json`, source hashes, exact document diffs and integrity checks in the companion. All 34 defined record checks pass, while the experiment remains blocked. The checks cover status agreement, source identities, saved feature-threshold arithmetic, inventory counts, 18 snapshot-file hashes, review-manifest consistency and static identification of the duplicate-question fixture. They do not reproduce hidden vectors, run the 48 source-reported tests, rerun H, train/evaluate models, resolve public checkpoint revisions, review labels, repair notebooks or change persistent sources. The historical `OpenDecision_Phase2IJ_Checkpoint_Readout.md` includes source excerpts and the practical next steps.

**Version 0.6.1 editing lineage.** The editing authorities are the latest delivered v0.6 whitepaper and roadmap, whose hashes are recorded in the companion manifest. The older v0.3 paper and pre-H roadmap surfaced in the conversation are historical, not replacement authorities. All prior whitepaper numerical tables are retained verbatim; only current-status/priority text and the explicitly new checkpoint evidence are added or updated. The v0.6 completed-H closure and all 16 review-to-work groups remain intact. No source Drive file is modified.



**E11: Completed exploratory 2I/2J model-selection screen and selected-model export.** Study `2ij_model_selection_screen_v2`, workbench `2ij.2.0`; final workflow `completed_requested_scope`. The saved result set contains 13 completed fit jobs, 31 completed final profiles, bounded multi-question scaling, `MODEL_DECISION.json`, and an A100-completed reference model bundle. Selected profile: `a047d6802c3f06f085b8`, Qwen/Qwen3.5-4B-Base, state-first, score-summary rejection. The selected final panel contains 320 episodes from 56 messages and records 95.0% accuracy / 0.13006 NLL. Bundle SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`; fresh reload parity passed with max probability delta `3.6673555e-6` and zero selected-ID changes. Scope remains exploratory; independent review, production acceptance limits and Rust/Metal parity are not established.

**E12: Phase 3A Python BranchableState and batched-Q systems reference.** Run `20260920T024056Z`, schema `opendecision-phase3a-summary/v1`, source/result path `Google Drive / Colab Notebooks / OpenDecision_Phase3A_results / 20260920T024056Z`. The run uses selected E11 profile `a047d6802c3f06f085b8` and bundle SHA-256 `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`; it records no model change, training or reselection. Saved headline gates: three semantic smoke cases, semantic batched parity all pass, high-K systems parity all pass, same-process repeatability maximum probability delta 0.0 and zero argmax changes. The benchmark rows retain repeated-full, nested-sequential, batched-cold and batched-warm timings and peak allocations over semantic Q=1/Q=4 cases and a mechanics grid through L=1024/Q=16. The corrected notebook uses complete-cache reindexing for hybrid fan-out/select after the generic Transformers repeat helper proved unsupported for Qwen3.5 linear-attention cache layers. `release_quality_claim` and `rust_metal_parity_claim` remain false. This whitepaper derives ratios only from the saved timing rows; it does not rerun Qwen.

**E13: Phase 3B Python Qwen3.5 backbone-parity reference.** Run `20260920T152206Z` uses schema `opendecision-phase3b-backbone-parity-summary/v1`. Its checked-in path is `research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z`. The run uses E11 profile `a047d6802c3f06f085b8`, bundle SHA-256 `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`, and base revision `1001bb4d826a52d1f399e183466143f4da7b741b`. It records no training, model selection, or model modification.

The E13 export contains 4 exact token records and 47 FP32 vectors. Those vectors cover 34 ordered trace stages, 10 full-sequence candidate features, and 3 continuation points. Maximum fresh-feature delta versus the earlier bundle is `4.9591064453125e-05`. Cached continuation versus fresh full sequence is `1.9073486328125e-05`. Both are Python self-consistency diagnostics under the notebook's `1e-4` guard. The saved summary explicitly sets `rust_parity_claim=false` and `metal_parity_claim=false`.

**RUST1: Rust Phase 3.1/3.2 implementation checkpoint.** The repository adds `ModelExecutionProfile` in `opendecision-engine`. `opendecision-backends` adds the selected `qwen35` profile, head, tokenizer, and Phase 3B reference loader. Offline tests validate bundle and head hashes. They replay four original golden-feature cases and reproduce all four Phase 3B token records exactly.

The tests also validate the 47-vector contract and replay 10 Phase 3B candidate features through the selected head. The backend suite passes under Rust 1.75. The recorded working session also passes formatting, strict workspace Clippy, workspace tests, schema regeneration/no-diff, and `git diff --check`. No native Qwen weights, Candle/Metal execution, `BranchableState`, server registration, or Jev semantic-none mapping are claimed.

**RUST2: Rust Phase 3.3 input-embedding checkpoint.** `Qwen35Embedding` verifies the immutable base checkpoint's config and safetensors index, then checks the embedding shard size and SHA-256 before reading token rows. It validates the pinned BF16 `[248320, 2560]` tensor layout, widens BF16 values exactly to FP32, rejects invalid IDs and non-finite values, and preserves token order. An offline row fixture derived from token ID 25 at base revision `1001bb4d826a52d1f399e183466143f4da7b741b` has SHA-256 `84ce40703c960b305ad72adbdc0f6fc5b8df6fd279e5b18d5dd194e4764377c7`; its Rust FP32 output matches E13 `diagnostic.embedding` exactly. A diagnostic probe applies the same comparison to the complete locally downloaded 5.3 GB embedding shard. This is input-embedding parity only, not decoder, full-backbone, Metal, or service parity.

**RUST3: Rust Phase 3.3 CPU backbone and continuation checkpoint.** `Qwen35Backbone` pins Candle `0.8.0`, verifies both immutable BF16 shards from base revision `1001bb4d826a52d1f399e183466143f4da7b741b`, and performs FP32 CPU execution for embedding, 24 DeltaNet blocks, 8 full-attention blocks, and final RMSNorm. On an Apple M4 Max (`Mac16,5`, 36 GiB, macOS 26.6.2), the complete 34-stage diagnostic has exact embedding and the final-norm values reported in §17. All 10 candidate sequences preserve four exported distributions with maximum probability delta `4.5869e-06`, zero argmax changes, and zero policy changes. The correctness run took `399.75 s`; macOS `/usr/bin/time -l` reported `7.77 GB` maximum resident accounting including mapped model shards and a `680 MB` peak memory footprint. The internal `BackboneState` carries attention KV, recurrent, convolution, and position state; its exported root byte count is exact, its source remains immutable, and cached candidate output equals native full-sequence output exactly. This is CPU fixture parity, not Metal, backend-neutral branching, production throughput, service integration, or release promotion.

**PUB1: Public OpenDecision state-first reference repository.** <https://huggingface.co/cowWhySo/OpenDecision-Qwen3.5-4B-StateFirst>. Project-owned publication of the selected state-first integration line. The repository URL is a public identity/reference surface; experiment identity remains pinned by profile ID, base-model revision, renderer/head/rejection contracts and bundle hash. Publication does not establish TypeSafe RLCD reproduction, release-quality promotion, or Rust/Metal parity.

# Appendix B. Primary external references

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

**P18.** Qwen. *Qwen3.5-2B-Base model card.* Official same-family smaller-checkpoint source, checked for v0.4. Its published text-model specification is not an OpenDecision quality, memory, or latency result. [Model card](https://huggingface.co/Qwen/Qwen3.5-2B-Base).

**P19.** Convai Innovations. *Laya model card.* Author-described compact bidirectional decision architecture, option-marker scoring, per-question input budget, batching, and limitations, checked for v0.4. No author timing, calibration, or Jev-superiority claim is adopted as an independently reproduced result. [Model card](https://huggingface.co/convaiinnovations/laya). The linked project is a candidate for a future code audit, not an audited dependency of this revision.

**P20.** Jiang, P., et al. (2026). *Efficient, Property-Aligned Fan-Out Retrieval via RL-Compiled Diffusion.* arXiv:2603.06397v1, 6 March 2026. The R4T workflow is cited for its separation of expensive teacher-side optimization from lightweight deployment; the proposed decision-model distillation study is an extrapolation, not a paper result. [Paper](https://arxiv.org/html/2603.06397v1).


**P21.** Gundala, H. *Qwen-2.5-1B-RLCD / Parallel Constrained Decoding for Apple Silicon.* Hugging Face model/repository documentation, reviewed 20 September 2026. Describes a shared-prefix Qwen/MLX inference path with cache broadcasting across fields, constrained candidate-token logit slicing, token-tree continuation and host-side JSON assembly. The page reports M4 Max benchmark values including 68–75 ms four-field cases, 270 ms for 28 fields and 89 ms for one 255-choice case. OpenDecision cites these as author-reported inference mechanics/performance, not evidence that TypeSafe RLCD training has been reproduced or that the resulting probabilities are empirically calibrated. [Model card](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).

**P22.** Reddy, N. (19 September 2026). *Jev-style models on DGX Spark.* More Than a Machine. External comparison of Jev 1.13, Laya and local decision readers on WANLI, BoolQ, serving Q-scaling and ViZDoom controllers. The reported serving table has Jev p50 105.1→109.2 ms from one to four questions, Laya 16.4→29.1 ms, and tuned Qwen3.5 167.0→665.1 ms; Jev includes hosted HTTP while local readers are warm/in-process, so absolute latencies are not directly comparable. The post also reports separate Brier/ECE metrics and campaign-to-campaign Jev API variation. OpenDecision uses it as external benchmark context, not evidence of Jev's private architecture. [Article](https://morethanamachine.com/posts/jev-style-decisions-dgx-spark/).

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

This revision adds P21/P22 and changes no OpenDecision model result, selection or historical acceptance decision. It clarifies that the public `Qwen-2.5-1B-RLCD` artifact is primarily evidence for parallel constrained-decoding mechanics rather than an established TypeSafe-RLCD training procedure; contrasts schema-first PCD with OpenDecision's selected state-first isolation contract; adds explicit state/evidence-sufficiency and high-cardinality follow-ups; and refocuses Phase 3 on native parity → branchable hybrid state → sequential nested parity → batched question execution → candidate batching → Q-amortization/high-K/repeatability measurements. External Jev/Laya/Spark numbers are benchmark context, not OpenDecision release thresholds.

## Version 0.7.2: Phase 3A/3B references and initial Rust parity gates

This revision adds E12, E13, RUST1, RUST2, RUST3, and PUB1 without changing the selected profile or prior quality results. It records the completed Phase 3A Python systems run. That run covers corrected hybrid-state reindexing, semantic/high-K parity, repeatability, and the measured execution crossover. The revision also records the completed Phase 3B Python backbone export with exact tokens and 47 vectors. Rust head/probability, exact-token, CPU full-sequence backbone, and Qwen-specific cached-continuation parity now pass. The fail-closed Phase 3B loader and two-shard checkpoint path are implemented.

Backend-neutral `BranchableState`, nested/batched Q/K execution, Metal, service integration, and Mac scheduling remain open. The public Hugging Face repository is a reference publication, not release certification. Phase 3A.1 remains conditional scheduler work if native profiling cannot derive stable crossover rules.

## Version 0.7.2 amendment: backend-neutral BranchableState (RUST4)

This implementation amendment adds the Phase 3.4 Rust branch-state checkpoint without changing the selected profile, bundle, base revision, model results, or tolerances. `crates/opendecision-backends/src/branch` defines the backend-neutral `BranchableState`/`BranchBatch` contract, and `BackboneState` binds it: profile/model/tokenizer/renderer/arithmetic identity, process-local lineage, a structural scheduling fingerprint and a strict little-endian content fingerprint, exact attention-KV/recurrent/convolution/metadata byte accounting, immutable-root fork, batched fork, and gather/select. The checkpoint-gated native branch stage replays the Phase 3B root/question/candidate continuation fixtures with exact root bytes and exact cached-versus-full state equality. CPU native parity now passes; Metal/native accelerated parity does not yet follow from that result. Sequential nested parity remains the next gate. Phase 3A.1 stays a conditional cost-model study.

## Version 0.7.2 amendment: sequential nested execution (RUST5)

This implementation amendment adds the Phase 3.5 Rust sequential nested checkpoint without changing the selected profile, bundle, base revision, model, tolerances, or fixtures. `crates/opendecision-backends/src/qwen35/backbone/nested.rs` implements the nested graph behind a `SequentialNestedExecutor` trait: prefill the shared state once, fork the immutable root per question, advance it, fork the question state per candidate, and advance each candidate fork, with fail-closed position and immutability verification after every stage. The checkpoint-gated `qwen35_nested_parity` stage executes all four Phase 3B questions and all 10 candidates through this path with maximum probability delta `4.5869e-06`, zero argmax changes, zero policy changes, root content identity against an independent prefill, exact replay and sibling-order determinism, and exact cached-versus-full feature and state equality (`0.0`). Hidden-vector deltas remain localization diagnostics. CPU native parity now passes; Metal/native accelerated parity does not yet follow from that result. Batched question execution is the next gate, measured against this sequential baseline. Phase 3A.1 stays a conditional cost-model study.

## Version 0.7.2 amendment: breadth-first batched Q/K execution (RUST6)

This implementation amendment adds the Phase 3.6/3.7 Rust batched execution checkpoint without changing the selected profile, bundle, base revision, model, tolerances, or fixtures. `crates/opendecision-backends/src/qwen35/backbone/batched.rs` implements the breadth-first graph (`run_batched_questions`, `run_batched_candidates`, `run_batched_nested`) over `BranchableState::fork_batch`: one immutable prefill, `Q` isolated question lanes advanced breadth-first, then a `K` candidate fan-out per question state, with fail-closed root, sibling-lane, and position verification after every stage. The checkpoint-gated `qwen35_batched_parity` stage proves exact parity with the sequential baseline — every question and candidate feature delta `0.0` with identical strict state fingerprints across all four Phase 3B questions and 10 candidates — while the frozen head reaches the same `4.5869e-06` maximum probability delta with zero argmax, zero policy, and zero cross-strategy decision changes, and fan-out byte accounting is exact. Offline tests pin gather/reorder equivalence. Per-lane executor calls remain the primitive, so this does not claim vectorized suffix kernels, Metal, or adaptive scheduling; CPU native parity does not imply Metal or accelerated parity. Target-Mac scheduler measurement is the next gate. Phase 3A.1 stays a conditional cost-model study.
