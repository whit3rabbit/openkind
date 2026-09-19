# OpenDecision
## Building a Qwen-based, Jev-style decision engine

**Research whitepaper · Version 0.3 · 18 September 2026 (America/Chicago) · Completed Phase 2G update**

Prepared for Chris from the OpenDecision Colab notebooks, saved experimental artifacts, earlier project research, and primary technical sources. This is a research synthesis, not a claim of production readiness or a reproduction of TypeSafe’s proprietary training method.

### Abstract
OpenDecision investigates whether an open-weight language-model backbone can act as a typed decision engine without generating answer text. A frozen Qwen3.5-4B-Base backbone with small trained heads remains the measured foundation. The research now spans fixed NLI, dynamic candidate scoring, missing-option modeling, cost-sensitive policies, numerical diagnostics, shared-prefix execution, completed cache-compression experiments, and fresh cross-domain and cache-lifecycle tests. Phase 2F supports lossless prefix persistence and bounded FP16 attention-KV storage while rejecting four tested low-bit TurboQuant snapshot configurations under the unchanged decision-preservation gate. Phase 2G strengthens strict-FP32 cache agreement on 56 fresh episodes from 16 messages and 72 controlled-context variants from six messages. TF32-permitted full batching is approximately twice as fast on the measured short requests, but three of 416 fresh full-sequential episodes change a head's selected outcome and longer context produces additional equivalence failures. Fresh task evidence is more limiting: the set-aware none head reaches 93.75% answerable accuracy on sampled non-financial CLINC intents but only 39.06% recall when the correct intent is omitted; author-OOS recall is 46.88%. High-confidence criteria confusions and policy-transfer errors show why numerical agreement, semantic accuracy, calibration, and automation utility must remain separate. The next priority is clearer decision criteria and independently evaluated rejection/calibration transfer, while preserving the execution reference. Removing a tied vocabulary head still saves output computation, not shared parameters. [E1–E7]

**Version 0.3 scope.** This revision applies the previously identified none-probability, segmented-tokenization, and refitted-constant attribution corrections; retains the historical Phase 2B–2F results; and adds the completed Phase 2G evidence and revised research priorities. Run identifiers use UTC: the Phase 2G results were saved around 01:45 UTC on 19 September, which is 20:45 on 18 September in America/Chicago. No model inference or training was performed while updating this paper.

---

# 1. Executive assessment

**The central feasibility question now has a positive, bounded answer:** Qwen can support useful, non-generative decisions through small trained heads. The research has not yet established an arbitrary-domain, fully Jev-compatible decision model. It has established a credible experimental foundation and identified where deployment risk and performance costs actually arise. [E1–E7]

A frozen text-only Qwen3.5-4B-Base backbone with last-token features and a linear head reached 87.67% matched and 87.33% mismatched accuracy in Phase 2B’s three-class MultiNLI experiment. Phase 2C selected the same head family across three training seeds and obtained 87.0% and 88.8% on new matched and mismatched test samples. These are sampled NLI results, not general decision accuracy, and separation from earlier experiments does not establish separation from Qwen’s pretraining corpus. [E1; E2]

The variable-candidate experiment is a second, different result. A shared candidate scorer trained on 57 Banking77 labels transferred to 20 labels withheld from head fitting. It achieved 88.54% accuracy on seen-label sampled-choice episodes and 82.29% on held-out-label episodes. However, missing-option recall was substantially weaker than accuracy on answerable cases. Later experiments improved this behavior with a seven-parameter set-aware “none” model, at the cost of more unnecessary rejection on answerable inputs. [E2; E3]

The systems experiments make precision part of the decision contract. Expanded Phase 2E completed both FP32 and BF16 workers, but completion was not equivalence. Relative to FP32 full-sequential execution, FP32 batching and cache reuse stayed within roughly 0.000011 maximum probability difference on the 128-episode panel, with no recorded class or policy-output changes. BF16 variants failed the declared 0.005 probability tolerance and changed decisions. The panel contains only eight distinct messages, so these counts are diagnostic—not estimates of deployment failure rates. [E5]

Phase 2F now supplies the previously missing compression and persistent-reuse evidence. All stages completed with frozen weights, heads, and policies. In FP32, lossless and FP16-KV snapshots passed all 32 compression-panel episodes; the four low-bit TurboQuant variants failed the declared gates. Lossless GPU caching reduced total time across the controlled 32-request traces without recorded policy changes, whereas the compressed LRU path altered outputs. The actionable result is a bounded benefit from exact-prefix reuse, not a general endorsement of low-bit cache compression. [E6]

Phase 2G adds fresh labeled decisions instead of extending only the old replay panel. Strict-FP32 lossless and FP16-KV cached paths pass both the 56-episode fresh parity subset and the 72-episode controlled-context panel. TF32-permitted full batching reduces measured K = 4/16 request latency from approximately 265/1,118 ms to 132/518 ms, yet it is not a fully interchangeable execution mode. Its fresh full-sequential comparison changes three head argmax outcomes across 416 episodes, despite no change under the nine frozen application policies. [E7]

The new task evidence shifts the immediate bottleneck toward criteria and rejection. The set-linear head's fresh accuracy is 69.53% on Banking head-training labels, 75.78% on Banking held-out labels, and 66.41% on sampled CLINC non-financial intents. On that CLINC in-scope panel, answerable accuracy is 93.75% but omitted-intent recall is only 39.06%; separate author-OOS recall is 46.88%. These are different test constructions from earlier phases, not a measured deterioration in unchanged weights. A high-confidence four-episode error cluster derives from one ambiguously described physical-card request. The recorded errors remain errors while the criteria warrant review. [E7; R2]

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

**Recommended immediate direction:** preserve strict FP32, lossless reuse, and the bounded FP16-KV result as execution references; keep TF32 as an explicitly versioned performance candidate and the tested low-bit configurations outside the accepted-equivalence set. Make clearer candidate criteria and independently evaluated missing-answer/OOS rejection the next quality study. Retain old and newly inspected G examples as regression data, not an untouched final test after tuning. Avoid changing prompts, kernels, precision, heads, policies, and caching simultaneously. A faster engine with altered probabilities is a new operating point, not an automatically equivalent implementation. [E3–E7; engineering recommendation]

# 2. What “Jev-style” should mean

## 2.1 A software-facing contract, not a claim about hidden internals

TypeSafe documents Jev as a model that accepts a shared state and named typed questions, then returns structured decisions and probability distributions without free-text generation. Its published interface has three primitives: Choice, Score, and Noul. The documentation describes questions as evaluated in parallel and in isolation against the same state. These are the relevant behavioral targets for OpenDecision. They are not a public specification of Jev’s complete neural topology. [P1]

Choice returns a selected member of a supplied option set, probabilities, and a confidence statistic. The documentation reviewed for version 0.1 permits up to 255 options. Score returns a distribution over described levels and a probability-weighted mean of their level numbers; equal means can conceal different distributions. Noul returns the probability of a yes answer, without a separate confidence value. A Noul is not the project’s “none” class. [P2–P4]

TypeSafe describes confidence as a statistic derived from the returned distribution. It should not be silently equated with the maximum class probability or with a calibrated probability that an entire workflow is correct. OpenDecision should expose the full distribution and explicitly version any additional concentration statistic. Matching field names while changing the statistic’s meaning would create a misleading compatibility claim. [P5; design implication]

TypeSafe publicly identifies a new architecture, a parallel sampler, and Reinforcement Learning for Calibrated Decisions, or RLCD. The primary pages reviewed describe the objective and behavior, not enough algorithmic detail to establish that OpenDecision reproduces the training procedure. The appropriate claim is an independently designed, open-weight implementation of a similar decision interface. [P6; P7]

## 2.2 Five properties that must be tested separately

**Structural validity** means every successful response obeys the requested types and option membership. Host-side construction can provide this without letting a model invent keys or serialize JSON. It says nothing about whether the selected answer is true.

**Decision quality** measures whether the predictions resolve the intended task. This requires labeled examples that represent the target use, not only a demonstration with a few appealing outputs.

**Calibration** asks whether stated probabilities correspond to observed frequencies within a defined evaluation population. A model can be accurate but overconfident, or have a low aggregate calibration error while being nearly useless at discrimination. Phase 2B’s training-prior baseline illustrates the latter: near-chance accuracy accompanied a low ECE. [E1]

**Isolation and consistency** cover multiple different contracts: renaming opaque keys should not change tokenized meaning; candidate permutation should permute results appropriately; unrelated questions should not affect another question’s answer; and approved execution strategies should stay within defined numerical and policy tolerances. A change in candidate-set composition can legitimately change a Choice distribution because the alternatives have changed.

**Efficiency** must include complete requests, not only the final classification layer. Tokenization, prefill, candidate suffixes, cache copies, transfers, probability computation, and serialization all contribute. Existing measurements cover selected single-process GPU workloads; they do not establish network, concurrency, queueing, or cold-model service latency. [E3–E5]

## 2.3 Relation to existing open work

Public projects provide useful components but do not settle all five properties. AlexWortega/openjev documents a Qwen3.5-4B NLI checkpoint with three classes and plain cross-entropy, and now also describes larger-backbone latent-head experiments. Its existence supports the practicality of the NLI approach, not proprietary Jev equivalence. The LFM2.5-2.6B-RLCD model card explicitly calls its constrained-decoding implementation inference-only and uncalibrated, and says it does not reproduce TypeSafe’s RLCD. [P12; P13]

The closest research references answer different questions. SALSA supplies a strong finite-token classification baseline using relevant output-token logits and parameter-efficient fine-tuning. GLiClass investigates efficient classification with dynamically specified labels. Perceiver IO supplies a general architectural precedent for querying shared representations to produce structured outputs. None of these papers identifies Jev’s internal implementation. They are comparison points and design options, not interchangeable solutions. [P9–P11]

Prior project discussions also considered community novelty and attribution claims. This paper does not convert discussion-level accusations into findings. Establishing conceptual overlap, implementing a similar interface, reproducing an experiment, and establishing provenance misconduct are separate tasks. The present evidence supports engineering comparisons, not an attribution verdict.

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

The completed measured sequence uses Qwen/Qwen3.5-4B-Base at revision `1001bb4d826a52d1f399e183466143f4da7b741b`. The text backbone has 4,205,751,296 parameters, hidden width 2,560, and 32 blocks. Its layer list contains 24 linear-attention and eight full-attention blocks. The core results were obtained on an NVIDIA L4. The saved environment includes Transformers 5.17.0; the expanded workers record PyTorch 2.11.0+cu128. Environment details should travel with results because kernel and precision behavior matter. [E1; E2; E5]

The notebook execution logs also report missing optimized causal-convolution and linear-attention kernels, with reference implementations used instead. Transformers documents these optimized versus reference paths. The recorded timings should therefore be treated as measurements of this particular stack, not the speed limit of Qwen on an L4. Installing faster kernels is a future experiment requiring both new timing and renewed probability/policy parity checks. [E5; E6; P16]

## 3.2 What was reviewed and what was not rerun

Earlier versions synthesized executed notebook snapshots, full result JSONs for Phases 2B–2F, summary documents, and relevant embedded source. Version 0.2 added the completed Phase 2F archive and matched its uploaded paste-back summary to Drive. That earlier audit reconstructed 1,664 Phase 2F probability/action comparisons, checked 384 storage records and 922 timing medians, and replayed 12 LRU traces under zero-expiry conditions; the expanded 2E outputs had also been re-aggregated. These historical audit scopes are retained rather than presented as new inference. [E5; E6]

Version 0.3 additionally reads the completed Phase 2G archive and the two prior review companions. Its independent saved-output check reconstructs 6,960 distributions and 20,880 application-policy actions from retained candidate scores and coefficients, with maximum distribution reconstruction residual approximately 1.22 × 10⁻¹⁵. It rechecks 24 full fresh-quality head summaries, 768 within-mode comparisons, 416 fresh and 72 controlled-context cross-mode comparisons, 16 request-time aggregates, and 12 trace totals. The prior G review separately checked broader summary records and replayed eight scalar LRU traces. No Qwen inference, training, GPU timing, annotation adjudication, or full external-reference re-review was performed for this revision. Arithmetic agreement does not independently reproduce the backbone or validate author labels. [E7; R1; R2; v0.3 verification companion]

The two supplied ChatGPT share links resolved to their conversation titles but did not expose readable conversation bodies through the available web reader. Earlier project reports in the Library supplied additional framing; executable notebook code and saved result artifacts take priority over those reports. The source register records this limitation rather than implying that the shared transcripts were fully inspected. [S1; S2]

Version 0.1 used a Phase 2F notebook snapshot retrieved around 22:59 UTC on 18 September, while the run was still executing. Version 0.2 supersedes that status with the completed run `20260918T224427722898Z`, whose result files were saved to Drive at approximately 23:56 UTC. Both precision workers and all requested stages completed. Completion does not imply that every codec passed its numerical or policy gates. The old notebook remains historical provenance; the completed archive is the Phase 2F result authority. [E6]

## 3.3 Evaluation units and leakage boundaries

MultiNLI splits were separated using normalized premise groups, not only individual rows. Phase 2C excluded 4,021 prior groups from its newly selected splits and used 2,400 training rows, 300 development rows, 1,000 calibration-fit rows, 500 calibration-gate rows, and 1,000 rows in each new test. Training examples remained the saved Phase 2B training set rather than a larger corpus. [E2]

Banking77 results require a different denominator. Several candidate-set episodes can come from one underlying message. Phase 2C’s 480 episodes per test split came from 160 messages; Phase 2E’s 1,024 episodes per split came from 64 messages. Expanded 2E and Phase 2F replay the same eight messages in a 128-episode factorial panel. Phase 2F selects 32 episodes for codec regression and 16 for short-request benchmarking; these subsets still span only eight messages. Its repeated-request traces do not add independent examples. Confidence intervals and uncertainty discussions must respect message clustering. [E2; E4–E6]

The 57/20 Banking label split withholds labels from candidate-head training, development selection, and calibration. It does not withhold them from Qwen’s pretraining. Those two label groups remain within banking; Phase 2G separately adds CLINC non-financial and author-OOS transfer panels. “Fresh” means new relative to the named recorded project-message manifests, not necessarily novel knowledge to the backbone. Repeating the same seeds reuses the same selected examples. [E2–E4; E7]

Phase 2G excludes 2,912 normalized prior Banking message hashes and constructs 416 fresh episodes from 112 messages. Its 56 parity episodes come from 16 messages; 72 controlled-context episodes transform six selected messages; eight episodes are timed independently; repeated 48-request traces supply no new semantic observations. Those units are kept distinct throughout Section 10. Once these G examples have been inspected to choose the next modeling intervention, they belong to historical/regression evidence rather than another untouched final evaluation. [E7; R2; methodological implication]

# 4. Qwen as a decision backbone

## 4.1 The initial probe and its corrected interpretation

The first probe established basic plumbing: hidden states could be extracted, last/mean/max pooling produced 2,560-dimensional vectors, a small head could fit two security-flavored examples, and a decision path could run without generating an answer. Its approximately 7.48-second generation observation versus 0.262-second head-path observation was exploratory, not an equal-quality benchmark. Two fitted examples cannot demonstrate classification generalization. [E0]

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

Candidate count matters. Seen-label none recall fell from 91.67% at K = 2 to 44.44% at K = 8. With fixed existing logits and positive temperature, adding a candidate lowers the probability assigned to a constant none logit through the softmax denominator. However, selecting none by argmax depends on whether its logit exceeds every candidate logit, subject to the implementation’s tie rule—not whether it exceeds the sum of their exponentials. [E2; R1; algebraic distinction]

```text
P(none) = exp(b / T) / [exp(b / T) + Σ_j exp(s_j / T)]
none wins by argmax when b > max_j s_j, apart from the declared tie rule
```

Adding a weak alternative can dilute none probability without changing the selected outcome. Adding more alternatives also creates more opportunities for a distractor to outrank none. Probability dilution and argmax errors are distinct effects; these experiments do not establish either as the sole cause of the recall decline. [R1; algebraic interpretation]

The small robustness panel found exact unbatched order agreement but a maximum batched probability change of 0.01867 on 24 episodes. A paraphrased instruction changed one prediction out of 24 and moved the panel’s accuracy from 95.83% to 91.67%. These checks show where invariance needs testing; the samples are too small to establish general robustness rates. [E2]

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

The gain is real within the experiment, but it is not free. The held-out-label false-none rate rises to 18%, so the system becomes much better at detecting omitted intents while turning away more cases it could potentially answer. Which trade-off is useful depends on costs and deployment prevalence. A single “accuracy improved” headline would conceal that decision. [E3]

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

The 5%-absent, cost-20 example is a useful negative result. The selected policy has a seen-test point cost of 0.1413—worse than reviewing everything at 0.1—even though development selection favored it. Its message-bootstrap interval is wide, approximately 0.0562–0.3687. This does not prove a stable inferiority on all future data; it shows why a small development panel cannot certify rare, costly errors. [E4]

A low measured error among accepted answers also needs a coverage denominator. An all-review policy produces no wrong automated answers because it produces no automated answers. Expanded 2E checks nine policies, including conservative ones; zero changes for a policy that always reviews provide little evidence about fine-grained probability fidelity. Report distribution drift, class changes, answer/review changes, and accepted-candidate changes separately. [E4; E5]

## 6.3 Training implications

Continue using log loss and other proper probability metrics alongside accuracy, Brier score, reliability plots, and cost/coverage curves. A single ECE binning can hide subgroup problems and should not be the sole training or deployment target. Fit calibration only on reserved data and select it on another reserved gate; retain a fresh final evaluation after any training or threshold change. Phase 2C’s gated temperature procedure is a useful pattern. [E1–E4; P8]

LoRA and RL-based post-training remain open experiments in this project: the completed reports do not include a successful LoRA comparison or proprietary-style RLCD training. The next training study should compare frozen features, limited adaptation, and a matched finite-token baseline under equal task splits and declared optimization budgets. A model name containing RLCD is not evidence about how it was trained. [E1; E2; P9; P13]

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

A separate cross-precision comparison—FP32 full-sequential versus BF16 full-sequential—found a maximum probability difference of 0.059015, 18 episodes with a class change, and eight with a policy-output change. This establishes disagreement, not that FP32 is semantically more accurate. FP32 is useful here because its alternative execution strategies closely reproduce its own reference. [E5]

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

A crucial limitation appears in the saved prompt contract: the instruction precedes the message. The shared prefix therefore belongs to the same instruction-plus-message context, not simply to an abstract state. The completed experiment amortizes candidates within a question. It does not demonstrate prefill-once execution across unrelated questions with different instructions. [E2, dynamic.export.prompt_segments; E6, cache planner]

A future state-first rendering could place an identical state prefix before question-specific suffixes. That would create more opportunities for reuse, but it changes the causal representation: the state tokens would no longer be conditioned on the preceding question instruction. It must be treated as a new prompt/model contract, evaluated and potentially trained accordingly—not as a semantics-preserving cache patch. A shared-encoder/query architecture is another option, also requiring a separate learning and evaluation program. [Design inference; P11]

Similarly, K in these benchmarks is the number of candidate alternatives, not the number of independent Jev questions. The distinction matters when projecting latency, memory, and isolation behavior to a multi-question API.

Phase 2F’s persistent cache extends reuse across requests only when their exact token prefixes match. It does not change the instruction-first prompt contract or demonstrate an instruction-independent state representation. The request traces therefore add evidence for persistent prefix reuse, not for arbitrary-question sharing. [E6; implementation interpretation]

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

For contrast, 2B’s fixed three-class batch-one decision path had an approximately 80.75 ms median. Generating 1, 8, or 32 tokens took about 91.32, 533.54, or 2,035.35 ms in the corresponding exploratory timing comparison. The often-attractive 25.2× ratio at 32 tokens is a difference in workloads—not a demonstrated equal-quality speedup or an expected advantage over a tuned one-token classifier. [E1]

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

The six storage configurations are `lossless`, `kv_fp16`, `tq_k3_v4_r0`, `tq_k3_v4_r32`, `tq_k4_v4_r32`, and `tq_k3_v2_r32`. In the TurboQuant names, k and v identify configured key/value bit settings, and r identifies the number of recent prefix tokens retained exactly. The pinned 0xSero implementation is revision `31660314b229b8d1c29bfddf8b9f5026b5121095`. The 384 MiB LRU budget and 600-second TTL are experiment settings, not demonstrated production optima. [E6]

## 9.2 What was—and was not—compressed

Only full-attention keys and values are quantized. Recurrent and convolution state remain exact, and model weights remain unquantized. The adapter stores genuinely packed tensors but restores floating-point KV before suffix attention. It therefore evaluates snapshot storage, restoration, and reuse; it does not evaluate fused low-bit attention or the upstream vLLM backend. CPU snapshot storage was tested separately from model offloading: both workers kept the model on the GPU. [E6]

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

In the actual grouped FP32 LRU trace, lossless entries occupy 394,199,040 bytes. The compressed entries occupy 389,769,984 bytes, but adding 4,194,608 table bytes leaves only 234,448 bytes less storage—approximately 0.22 MiB. Both caches hold seven entries and have the same hit/miss pattern. In BF16, the compressed entries plus tables exceed the corresponding lossless total. Compression therefore did not improve cache capacity in these traces, while its output-parity gate failed. [E6; arithmetic derived]

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

At 1,024 tokens, cold shared-prefix execution is approximately 1,296 ms versus 9,217 ms full sequential; warm lossless reuse is approximately 273 ms after an initial population request of approximately 1,295 ms. These answer different latency questions and must not be presented as a single unconditional speedup. FP16 storage’s warm time is approximately 275 ms at this length; its principal demonstrated benefit is smaller retained storage, not faster suffix evaluation. [E6]

All four low-bit codecs fail the 0.005 probability gate at all three synthetic prefix lengths in FP32, even though those particular fixtures record no argmax or policy-output changes. BF16’s faster warm lossless times—approximately 178, 179, and 184 ms—also fail comparison with BF16 full sequential, with maximum probability differences approximately 0.01683, 0.02084, and 0.03232. Warm execution, compressed storage, and accepted decision equivalence remain separate axes. [E6]

## 9.9 What Phase 2F changes—and leaves open

Phase 2F moves exact-prefix persistence from a proposal to a measured single-worker result. It adds a candidate FP32 optimization—FP16 storage of full-attention KV—whose sampled parity passes and whose storage savings grow with prefix length. It also supplies a specific negative result: none of the four tested TurboQuant snapshot configurations preserves the current frozen decision contract. Codec execution and CPU packing selftests passed; downstream equivalence did not. [E6]

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

Source: E7, fresh and controlled-context prediction rows; independently re-aggregated for this revision. Probability differences are absolute probability units, not percentages. The tolerance is 0.005, equivalent to 0.5 percentage points. Selected-outcome and policy counts are episode-level any-change indicators across three heads and nine policies, not counts of independent messages.

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

Source: E7, fresh cross-precision summary and raw controlled-context rows; R2. The second row is an additional saved-output audit, not part of the notebook's fresh-only cross-precision summary. It was rechecked during this revision. These two panels have different constructions and should not be pooled into one independent error-rate estimate.

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

Strict-FP32 cached strategies reproduced the full-sequential results throughout this panel, but the full-sequential task performance still changed with added background and request position. Thus execution fidelity and long-context semantic validity diverge in a concrete experiment: an optimized path can faithfully reproduce a decision that is wrong under the source label. The sample is too small to establish a general evidence-position law, but it identifies a useful next target for controlled robustness training and fresh evaluation. [E7; interpretation]

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

The task-quality conclusion is more demanding. Frozen candidate scoring can identify offered intents outside banking, yet missing-answer and author-OOS detection remain weak, high-confidence criteria confusions can defeat saved policies, and irrelevant context can change labeled decisions. The next research priority is therefore clearer decision criteria and independently evaluated rejection/calibration transfer, with the established execution suite retained as a regression constraint. Cache tuning, state-first redesign, lower precision, new heads, and LoRA should not all be changed in the same experiment. Phase 2H remains a proposed controlled study, not a completed result. [E7; research recommendation]

# 11. Architecture and implementation recommendations

## 11.1 Preserve three distinct layers

**Decision semantics:** define state, question instructions, candidate descriptions, label/level meaning, missing-option semantics, and the desired probability contract. This layer determines what the model is being asked.

**Probability engine:** bind the model revision, tokenizer, prompt rendering, normalization, head parameters, precision, kernels, and cache strategy. Its output is a versioned numerical function, not merely a checkpoint filename.

**Application policy:** choose accept, review, gather more information, or route elsewhere using an explicit cost model and authorization rules. This layer must not be mistaken for probability calibration or a semantic class. The E policy experiments and G transfer failures demonstrate why separating it is useful. [E2–E7; recommendation]

A service should return model/engine and policy version identifiers with telemetry, and report truncation and unsupported inputs explicitly. Most untouched task-quality inputs use a maximum length of 256. Phase 2G adds controlled labeled contexts through a minimum 1,024 state tokens, but their six source messages and generated administrative notes do not establish reliable decisions on natural long operational documents. Keep the synthetic mechanics and labeled context controls distinct. [E2–E7]

## 11.2 Two plausible paths toward broader Jev behavior

**Incremental Qwen path.** Retain the existing candidate-conditioned scorer, improve coverage and calibration, and add a separately evaluated state-first prompt contract for multi-question reuse. This preserves the most project knowledge but requires new data and tests when the question/state ordering changes. A learned set-aware none model and a separately trained binary decision head can coexist with the candidate scorer; Score requires its own rubric/ordinal evaluation.

**Shared-representation/query path.** Encode a state once and let isolated learned query modules produce decision representations for each question or candidate. Perceiver IO offers a conceptual precedent for query-driven structured outputs; GLiClass is a relevant efficient dynamic-label comparator. This is a research fork, not a drop-in optimization of the current frozen head. It may reduce repeated work, but could sacrifice question-conditioned feature quality unless trained appropriately. [P10; P11; proposal]

The incremental path is the lower-disruption architectural option, but the immediate Phase 2H priority is a criteria/rejection study under the existing rendering, not an untracked switch to state-first inputs. Neither architecture should claim Jev’s flat latency scaling or broad task competence without controlled measurements as both question count and candidate count increase. [E7; proposed sequence]

## 11.3 Rust and Apple Silicon: define the parity ladder

Saved head fixtures make a Rust port more tractable, but fixture export is not execution parity. The Phase 2D export explicitly states that its eight fixtures come from Python and do not validate a Rust runtime; Phase 2C excludes base weights from the head package. Phases 2F and 2G replay saved head algebra but still do not benchmark the full decision graph on Rust, Metal, or Apple Silicon. [E2; E3; E5–E7]

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

Evaluate proposed repairs separately: criteria clarification with the frozen model, small none/calibration refitting, a multi-domain candidate-head update, and limited LoRA as a controlled additional treatment. Retain strict FP32 as the execution reference and TF32 as a performance candidate; do not mix an input-contract change with automatic arithmetic or policy promotion. Evaluate both accepted-answer correctness and errors withheld for review. Replaying G after using its errors to design the repair is regression testing, not another fresh final test. [E7; proposed work]

# 12. Research questions answered—and still open

| Research question | Answer supported so far |
|---|---|
| Must useful decisions be generated as text? | No. Frozen features plus small heads work on the measured tasks. |
| Does a base model suffice? | It suffices for these probes; Base versus Instruct was not controlled here. |
| Is a linear head enough? | It is a strong selected NLI baseline. More complex heads were not consistently necessary in this setting. |
| Do unseen candidate labels work? | Held-out banking labels and fresh sampled CLINC intents show offered-answer transfer, but none/OOS behavior remains weak. |
| Is calibration automatic after supervised fitting? | No. Post-hoc scaling can worsen held-out metrics, and risk changes under new costs and prevalence. |
| Can a tiny none model help? | Yes. A refitted constant captures most of the 2D accuracy gain; seven set-conditioned coefficients add value, but G exposes transfer limits. |
| Does an isolated repeat prove deterministic deployment? | No. Padding, batching, precision, and cache chunking changed results. |
| Is FP32 the most accurate model? | Not established generally. It is the most internally consistent tested reference; numerical and semantic quality are evaluated separately. |
| Does TF32 permission preserve the decision contract? | Not fully. Full batching is about 2× faster in G, but fresh head argmax and controlled-context equivalence failures remain. |
| Can shared prefixes amortize work? | Yes, especially with longer prefixes and batched suffixes in the completed mechanical tests. |
| Has state-once, arbitrary-question-many execution been demonstrated? | No. Current reuse is across candidates sharing an instruction/message prefix. |
| Is TurboQuant already a win for this model? | Not under the frozen equivalence gate: all four tested snapshot codecs failed; short-prefix overhead also limited storage benefit. |
| Does FP16 KV storage help? | It passed strict-FP32 fresh/context gates in G and earlier storage tests; it saved bytes but did not add a cache hit or beat lossless trace time in G. |
| Does persistent prefix reuse help? | Yes on controlled single-worker traces: F and G show workload-dependent gains, with G exercising expiry; no production concurrency claim follows. |
| Are clear criteria and calibrated rejection solved? | No. G exposes confident errors on terse overlapping descriptions, weak missing-answer/OOS transfer, and context-sensitive semantics. |
| Does MTP speed up this decision path? | Not in its present no-output-decoding graph. |
| Are Score, Noul, and API parity finished? | No dedicated general Score/Noul training and evaluation or full Jev HTTP compatibility suite is evidenced. |
| Has Rust/Metal or a smaller Qwen been validated? | No completed full-backbone benchmark in the reviewed artifacts. |
| Has Jev’s RLCD been reproduced? | No. The project has a distinct, inspectable research path toward a similar software interface. |

# 13. Prioritized next experiments

Phase 2G completes the requested fresh-input, TF32, and bounded cache-lifecycle study. The next cycle should not repeat the same cache panel until an operating point happens to pass. Preserve the strict-FP32 cache reference, the FP16-KV storage result, the failed low-bit configurations, and the measured TF32 trade-off. Shift the immediate quality work toward decision criteria and rejection, using independently reserved data after any changes. [E6; E7; research recommendation]

**Proposed Phase 2H: criteria and rejection transfer.** First document and review confusing label descriptions without altering historical gold labels. Then compare criteria-only changes under the frozen model, a small none/calibration update on new fitting data, and a multi-domain candidate-head update. Limited LoRA is an additional matched treatment only after the smaller interventions have a baseline. Keep omitted in-scope intents and author-OOS cases separate, and report policy cost/coverage under predeclared population assumptions. These are proposals, not executed improvements. [E7; R2]

| Priority | Experiment | What it resolves | Acceptance evidence to retain |
|---|---|---|---|
| P0 | Criteria and annotation review, including confusing candidate pairs | Whether supplied descriptions express the intended decision boundary | Written criteria, unchanged historical labels/errors, independent domain review, versioned description variants, untouched final tests |
| P0 | Rejection and calibration transfer: fixed scorer, small none update, then multi-domain head | Whether missing-answer and author-OOS errors improve without unacceptable false rejection | Separate fitting/development/calibration/final sets; both omission and author-OOS panels; answerable accuracy, NLL, accepted-answer error, coverage, clustered uncertainty |
| P0 | Irrelevant-context and instruction robustness under the chosen criteria | Whether semantic decisions survive realistic background and evidence placement | More independent labeled messages; controlled and natural documents reported separately; no silent truncation or post-hoc relabeling |
| P1 | TF32 or another serving arithmetic path with frozen task/policy contracts | Whether the speed advantage is useful under explicit quality and behavioral requirements | Matched same-input strict-FP32 reference, cross-mode and within-mode gates, tie/threshold failures, fresh labels, complete latency and memory |
| P1 | State-first versus current rendering, after establishing quality baselines | Whether reuse can span different questions rather than only candidates | Adapted heads where required, versioned prompt IDs, matched task quality, cross-question isolation, scaling in both questions and candidates |
| P1 | Matched finite-token and dynamic-label baselines | Whether the custom graph wins on useful decisions rather than long-output workloads | Same splits, costs and compute budgets; short direct outputs; equal-quality latency/coverage comparison |
| P1 | Frozen head versus LoRA and smaller backbone | Whether adaptation or model-size changes improve the measured trade-off | Multiple seeds, separate fits, calibration gates, fresh final tests, full memory/latency accounting; no assumed 2B/9B winner |
| P1 | Dedicated Noul and Score tasks | Whether the remaining API primitives have semantic evidence | Binary calibration, described rubric levels, ordered errors, ambiguous/insufficient-evidence cases, complete distributions |
| P2 | Rust/Metal parity and service load | Whether the measured function survives another backend and real serving | Head → tokenizer → backbone → hybrid-cache parity, cancellation/concurrency/isolation, real request latency, no CUDA-to-Mac speed extrapolation |

The old and newly inspected G messages should be frozen as historical regression cases. A description fix or model selected using them must face a fresh final evaluation; reporting improved scores on G afterward is a diagnostic result, not an untouched test. Preserve label and criteria provenance, and do not collapse several variants of one confusing message into several independent failures. [E7; methodology]

Do not turn the current 0.005 probability tolerance into a universal safety threshold. For regression it is a declared engineering gate. For deployment, retain directed action changes, errors among accepted answers, coverage, plausible prevalence/cost scenarios, and uncertainty on expensive events. G shows that a probability difference below 0.005 can change argmax, and that zero policy-output changes can occur while semantic labels differ. Tolerances and selection rules must be set before evaluating a new candidate implementation. [E7]

The 4B backbone remains the measured baseline, not the established optimum. A smaller deployment model, larger teacher, or distribution-preserving distillation remains an experimental option; no reviewed run establishes a smaller-model winner or proves that model size alone would resolve G’s ambiguous criteria and rejection errors. Distillation should evaluate probability and review behavior, not merely hard-label agreement. [E1–E7; proposal]

## Conclusion

OpenDecision has moved beyond a concept demo into an inspectable decision-only research system: frozen-feature task experiments, dynamic candidate scoring, measured missing-option trade-offs, cost-sensitive policies, a numerically consistent strict-FP32 shared-prefix reference, and completed cache-compression and persistence studies. Phase 2F supports lossless persistence and bounded FP16-KV storage while rejecting four tested low-bit snapshot configurations under the existing gate. Phase 2G strengthens that reference on fresh inputs and demonstrates a substantial TF32 batching speed opportunity without proving universal equivalence. [E1–E7]

The fresh semantic evidence also narrows what can be claimed. The frozen scorer can identify supplied non-banking intents, but missing-answer and author-OOS rejection are incomplete, a terse-criteria confusion produces high-confidence policy errors, and added background can change labeled decisions even when the cached implementation is faithful. Efficiency, numerical consistency, semantic accuracy, calibration, and application utility cannot substitute for one another. [E7]

The next credible claim is still narrower and stronger than “an open Jev clone”: **an open, auditable Qwen decision engine whose task coverage, probabilities, execution consistency, and automation policies are measured separately.** The immediate next study should improve criteria and rejection on independently held-out tasks while preserving the systems reference. Broader question sharing, dedicated Score/Noul evaluation, optimized kernels, smaller models, and Rust/Metal deployment remain explicitly uncompleted parts of the program. [E7; proposed direction]

---

# Appendix A. Source and reproducibility register

The source IDs below identify the evidence behind the numbered sections. In the accompanying evidence manifest, local snapshot SHA-256 hashes distinguish the exact files reviewed from later Drive edits. Result paths are under `Google Drive / Colab Notebooks`. Timestamps embedded in run IDs are UTC.

**Historical version 0.2 audit boundary.** The completed Phase 2F archive SHA-256 is `e1d3fea878d35b9e929e83424d4f3aa8b5de0385bb100ace001110f7e5f83c37`. The version 0.2 companion retained version 0.1 evidence and added the Phase 2F manifest, compact results, and an independent standard-library aggregation audit. Probability and policy comparisons are recomputed where raw vectors/actions are retained. Request/long-prefix rows retain comparison diagnostics rather than all output vectors, so their numerical gates remain source-reported while timing aggregates are independently checked. No source Drive files were modified. [E6]

**Version 0.3 audit boundary.** The attached v0.2 source SHA-256 is `1adad381998535be157cc8c3806427ff7db1e8e21b7cd05afad5129969942cc2`. The completed G archive SHA-256 is `24ff15fa1fec3c8eeb17ef073da3d852cabe40683678b247d4e1d917e847414b`. The v0.3 verification companion includes a standalone saved-output checking script, its aggregate JSON, this revision’s source/hash manifest, the earlier correction and G review notes, a change log, and a text diff. The script does not download or run Qwen. Section 3.2 states the checks rerun for this revision; the earlier wider audit is attributed to R2. All numerical additions are from the supplied/completed artifacts, not new external research. No original attachment, notebook, source result, or Drive file was modified.

**E0 — Initial feasibility probe.** `OpenDecision_Phase2_Qwen3_5_4B_Probe.ipynb`. Executed exploratory notebook; model-loading comparison, pooling shapes, two-example head fit, and rough generation timing. [Open notebook](https://colab.research.google.com/drive/1bd1FNFL7FP0fRxBLg17XW9yz2kEXHNh8).

**E1 — Phase 2B.** Run `20260917T205849Z`; `OpenDecision_Phase2B_results / <run> / opendecision_phase2b_summary.json`. Principal fields: `architecture`, `data`, `all_metrics`, `matched_test`, `matched_uncalibrated`, `performance`, `generation_comparison`, `batch_invariance`, `frozen_export`. [Full result](https://drive.google.com/file/d/1Kj6Ph12DYU8VwRHnQ4-vTgA4Q2f2RmmS/view).

**E2 — Phase 2C.** Run `20260917T222948Z`; `OpenDecision_Phase2C_results / <run> / opendecision_phase2c_summary.json`. Principal fields: `nli_data`, `nli`, `stability`, `dynamic.data`, `dynamic.training`, `dynamic.evaluations`, `dynamic.robustness`, `dynamic.export`, `frozen_export`. [Full result](https://drive.google.com/file/d/1lXy-mCtjkPkgb3KZR8fsn0mRKEZNB4E9/view). [Executed notebook](https://colab.research.google.com/drive/1iYTfdtn1-G26zpNyK98iodp_IGBi1YAz).

**E3 — Phase 2D.** Run `20260917T234417Z`; `OpenDecision_Phase2D_results / <run> / opendecision_phase2d_summary.json`. Principal fields: `numerics`, `dynamic_data`, `selection`, `temperature`, `evaluation`, `request_benchmark`, `export`. [Full result](https://drive.google.com/file/d/12EDUXCzU3psTCDN-qTkkKTiRpxvnzkMX/view). [Executed notebook](https://colab.research.google.com/drive/1N_V3MnrZUbddO5j-ZQqDew_dEjXig6p1).

**E4 — Phase 2E, original systems/policy run.** Run `20260918T032049180933Z`; `OpenDecision_Phase2E_results / <run> / opendecision_phase2e_summary.json`. Principal fields: `data`, `precision.modes`, `shared_prefix`, `policies`, `policy_scoring`, `run_status`. Overall status is partial; completed policies remain valid evidence within their scope. [Full result](https://drive.google.com/file/d/1JjBbFdsFHjjXqdzBkFpigwZ9QhDWWgzq/view).

**E5 — Expanded Phase 2E.** Run `20260918T114914072764Z`; `OpenDecision_Phase2E_expanded_results / <run> / opendecision_phase2e_expanded_summary.json`. Principal fields: `sampling`, `workers.<mode>.parity_rows`, `parity_summary`, `benchmark_summary`, `long_prefix`, `memory_summary`, `cross_precision`. Completed execution; BF16 strategy-equivalence gates fail. [Full result](https://drive.google.com/file/d/1SxOG4VY4TlqK0e4eqBgZ_KwfDYjbXeFX/view). [Notebook containing the expanded run](https://colab.research.google.com/drive/17fuU04ZwOc2RJdahFiIyFaJ88YBgcvk8).

**E6 — Completed Phase 2F.** Run `20260918T224427722898Z`, version 2f.1.0; results saved at approximately 23:56 UTC on 18 September 2026. Source folder: `OpenDecision_Phase2F_results / <run>`. Both precision workers completed. Principal fields: `baseline_parity`, `compression_rows`, `benchmark_rows`, `component_profiles`, `cross_request_summary`, `long_prefix_rows`, `cross_precision`. The archive also retains raw `cross_request_rows.json`, inputs, source modules, and frozen head exports. [Full result](https://drive.google.com/file/d/1jKdR5xozQ7CQfzaTT82OQ7bszBEfJGPe/view). [Paste-back summary](https://drive.google.com/file/d/1UJMv80N6vQ_bgdQH4ejZfIztj8xHZ8TS/view). [Complete archive](https://drive.google.com/file/d/1m9PbuQqFlO4e3ilEpaSJ4Qvf8VcW56_-/view). [Notebook](https://colab.research.google.com/drive/1DmkUfMAUwgnn4vWg_kWJe2cK75QVr60G). The version 0.1 partial notebook snapshot remains historical evidence, not the current run status.

**E7 — Completed Phase 2G.** Run `20260919T005142584348Z`, version `2g.1.0`; results saved around 01:45 UTC on 19 September 2026. Source folder: `OpenDecision_Phase2G_results / <run>`. Both `fp32_strict_math` and `fp32_tf32_allowed` workers completed. Principal fields: `data`, `workers.<mode>.fresh_summary`, `context_summary`, `context_by_length_position`, `benchmark_rows`, `traffic_summary`, `flags`, `memory_snapshots`, and `cross_precision`. The archive retains `experiment_inputs.json`, `fresh_message_manifest.json`, `prior_exclusions.json`, worker `fresh_rows.json`, `context_rows.json`, `traffic_rows.json`, `traffic_events.json`, `benchmark_rows.json`, implementation snapshots, and frozen coefficients. [Full result](https://drive.google.com/file/d/1liuu456rvPFIQJXonRjxh83DTdJjKUGE/view). [Paste-back summary](https://drive.google.com/file/d/1F5vaHSj2Grix-RNChoTn2mhOWhet7pb2/view). [Complete archive](https://drive.google.com/file/d/1CAm4ooAuQ8gsxJAZNHn7jsnztnkF9IzP/view). [Notebook](https://colab.research.google.com/drive/1KVEB2apgM_8GIFdjLsX0x9LPvc94Pnzg). Freshness is relative to recorded project manifests; completion and strategy acceptance remain distinct.

**R1 — Prior whitepaper correction memo.** `OpenDecision_Whitepaper_v0.2_Review.md`. Identifies the probability-versus-argmax correction, segmented-tokenization correction, and refitted-constant attribution clarification applied in Sections 5.2, 8.1, and 5.3. The file is retained in the v0.3 evidence bundle.

**R2 — Prior Phase 2G saved-result review.** `OpenDecision_Phase2G_Result_Review.md` and `OpenDecision_Phase2G_Review_Audit.json`. Preserve the independent distribution/policy reconstruction, source-label/omission checks, context aggregation, ambiguity case, and scalar cache replay scope. The v0.3 checker independently rechecks its stated arithmetic subset; neither review claims new model inference or label adjudication.

**Pinned datasets.** MultiNLI revision `da70db2af9d09693783c3320c4249840212ee221`; Banking77 resolved revision `90d4e2ee5521c04fc1488f065b8b083658768c57`. Banking CSV hashes retained in the results are train `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` and test `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`. These identify downloaded data, not the contents of Qwen’s pretraining corpus.

**Phase 2G dataset extension.** The pinned CLINC `oos-eval` revision is `828f8093932c8fe6ca7936c3d2e52903b1c523de`. Recorded SHA-256 hashes are `data/data_full.json`: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`; `data/domains.json`: `b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a`; and `LICENSE`: `e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76`. G’s prior-exclusion manifest SHA-256 is `f3549213c79dc4530dd8d04ecf61e5a6f08b8ba3e470a3a00d8296bc7bc5b42a`; its fresh panel hash is `367d31772bdf2b9630766272f3b7252131b7caf5f7e2f692da8d4ce4d3a5ad7d`. These are recorded download/selection identities, not a new corpus or annotation audit. [E7]


**S1 — User-supplied discussion.** “Open Source Inference Plan,” [shared conversation](https://chatgpt.com/share/6aadc0b1-db54-83ea-856a-efc569000843). Title resolved; conversation body was not available to the web reader.

**S2 — User-supplied foundation discussion.** “Jev Architecture Overview,” [shared conversation](https://chatgpt.com/share/6aadc0ec-8ee8-83ea-9343-b16fb73dbdc3). Title resolved; conversation body was not available to the web reader. Earlier Library reports supplied secondary context, not replacement evidence for measured results.

# Appendix B. Primary external references

Primary pages P1–P16 were checked for version 0.1 on 18 September 2026. Their references and external-context sections are retained in this source-based v0.3 update; no new external-reference review is implied. The additional CLINC entry records the dataset source used in the completed G artifacts, not a newly conducted literature review. Vendor descriptions are cited as descriptions, not independent validation of performance or proprietary implementation.

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


# Appendix D. Version 0.3 revision record

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
