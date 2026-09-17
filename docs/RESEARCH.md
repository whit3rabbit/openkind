# Jev, Reverse-Engineered: How Reproducible Is TypeSafe’s Non-Autoregressive Decision Model?

## Executive finding

As of **September 17, 2026**, Jev is reproducible in the sense that matters most for an open-source project, but **not reproducible exactly**. TypeSafe has published enough of the external contract to reconstruct a model with essentially the same programming model—shared state in; `Choice`, `Score`, and binary `Noul` questions out; probabilities returned directly; many independent questions evaluated together; no free-form generation—but it has **not** published Jev's weights, parameter count, architecture, pretraining recipe, training corpus, sampler implementation, or the mathematical definition of Reinforcement Learning for Calibrated Decisions, or RLCD. Founder Diogo Almeida explicitly said the architecture is being kept “close to the chest” for now and that the company has discussed publishing a paper; he also suggested the data may be more important than the architecture. citeturn24search0turn24search3

That means there are really three reproduction targets:

| Target | Reproducible today? | Assessment |
|---|---:|---|
| Jev's **API/output behavior** | **Yes** | Straightforward |
| Jev's **no-generation, parallel-decision execution model** | **Yes, approximately** | Strong public precedents and an existing community proof of concept |
| Jev's **calibration/accuracy/latency Pareto frontier** | **Unknown** | Requires substantial training and direct Jev comparison |
| Jev's **actual proprietary architecture and RLCD algorithm** | **No** | Critical details undisclosed |
| A useful open-source **“OpenJev” that behaves like Jev** | **Yes** | Technically credible project |

TypeSafe's public documentation is unusually revealing about the behavioral constraints. Jev receives one shared state and a set of typed questions. `Choice` operates over caller-supplied alternatives, `Score` over an ordered caller-supplied rubric, and `Noul` returns a binary probability. TypeSafe says questions in a request are evaluated **independently against the same state and in parallel**, rather than conditioning on one another. Its own batching cookbook tests thirteen mixed questions against a roughly 54,000-character document and reports a 0.27-second batched call versus 2.71 seconds for thirteen serial calls; crucially, the company says the answers are effectively unchanged whether questions are submitted alone or together. citeturn17view1

There is also now some independent evidence for the product behavior. Every's Mike Taylor submitted 21 questions across 37 documents—777 judgments—and reports completion in under 0.7 seconds at an estimated cost of roughly a quarter-cent. In a much smaller accuracy comparison, Jev caught six of seven deliberately planted writing defects while Claude Fable 5.1 caught all seven; Jev's reported median latency was 0.35 seconds per passage versus 8.83 seconds for Fable. That supports the claim that the latency advantage is real, while also showing why speed must not be confused with frontier accuracy. citeturn20view0

The most important conclusion from reverse-engineering the interface is this:

> **Jev probably should not be thought of as a “non-autoregressive LLM” in the same sense as non-autoregressive text-generation research. It is better understood as a dynamically programmable, zero-shot structured predictor whose output space is supplied at inference time.**

That distinction changes how I would reproduce it. I would **not** begin by building a parallel text generator. I would build a pretrained semantic model whose expensive representation of the shared state is computed once, followed by a parallel bank of lightweight question/candidate evaluations, and train the resulting probabilities explicitly for decision quality and calibration. This interpretation is consistent with TypeSafe's API, its batching behavior, Almeida's agreement that a “zero-shot” classifier is a reasonable characterization, and the existing structured-classification literature. citeturn24search19turn21academia0

The supplied analysis was therefore directionally correct in separating the community `openjev` implementation from the real Jev substrate. The current `openjev` model card confirms that it is a Qwen3.5-4B NLI cross-encoder with three fixed labels—contradiction, entailment, neutral—last-token pooling, and ordinary cross-entropy training. That is a useful baseline, but it does not reproduce TypeSafe's mixed typed-question architecture, shared-state computation, or undisclosed calibration training. fileciteturn0file0 citeturn26search0

My overall assessment is therefore:

**An open-source Jev analogue is highly feasible. An exact Jev clone is currently impossible. The difficult research problem is no longer “how do we eliminate autoregressive generation?”—that part is relatively straightforward. The hard problems are obtaining broad zero-shot judgment ability, retaining that ability in a much cheaper decision-only architecture, and producing probabilities that remain calibrated across changing user-defined schemas and distribution shift.**

## What Jev actually reveals about its architecture

TypeSafe describes Jev as its first “System One” model and says it combines a **new architecture**, a **parallel sampler**, and **RLCD**. The model deliberately does not generate strings: the declared output space is typed in advance, so the application receives values and probability distributions rather than a token sequence that subsequently needs to be parsed. TypeSafe explicitly distinguishes this structural guarantee from correctness: the “zero hallucination” language refers to inability to violate the requested output type, not inability to choose the wrong answer. citeturn1view0

That seemingly simple constraint tells us a great deal.

### The output is almost certainly classification-like rather than language decoding

For a normal decoder LLM, an answer such as:

```json
{
  "route": "security",
  "severity": 3,
  "malicious": 0.91
}
```

is a sequence of tokens. Even with grammar-constrained generation, the model must repeatedly project hidden states into its vocabulary, sample or select tokens, and advance the sequence.

Jev's natural internal representation can instead be:

\[
p(y=k\mid x,q,C)
\]

where:

- \(x\) is the shared state,
- \(q\) is a natural-language decision instruction,
- \(C=\{c_1,\ldots,c_K\}\) is the dynamically supplied candidate/rubric set,
- and the output is a categorical distribution over those candidates.

`Noul` is simply the \(K=2\) special case. `Choice` is ordinary categorical decision-making with variable \(K\). `Score` can likewise be represented as a probability distribution over ordered rubric levels, from which a point estimate can be derived. TypeSafe's own API documentation exposes precisely these probability distributions. citeturn17view1turn18view0

The JSON object need never come from the neural network. Ordinary host code can serialize the selected labels and floating-point probabilities. **Schema validity can therefore be literally deterministic.**

### The model cannot have a conventional fixed classification head

Choice alternatives are supplied by the caller at inference time, including natural-language descriptions, and TypeSafe supports as many as 255 alternatives. A traditional `Linear(hidden_size, num_classes)` classifier trained for fixed intents such as `{billing, security, sales}` cannot do this. The model must somehow **semantically encode the question and candidate definitions themselves**. citeturn1view0turn17view1

That narrows the plausible design space to mechanisms such as:

1. a cross-encoder that separately evaluates each state/question/candidate combination;
2. a shared state encoder plus dynamically encoded candidate/query representations;
3. a causal-model prefill whose state cache is forked into many candidate-scoring branches;
4. a query-decoder architecture in which learned or textual query representations attend to a common state representation.

The first works but unnecessarily recomputes the state. The last three naturally explain Jev's batching economics.

### Independent questions strongly imply a separable attention/computation boundary

TypeSafe does not merely say questions execute simultaneously; its documentation says each one is evaluated **in isolation against the same state**. Its GDPR cookbook specifically tests whether adding twelve other questions perturbs an individual question and says it does not beyond ordinary run-to-run noise. citeturn17view1

That property is architecturally significant.

A naïve Transformer input like:

```text
[state]
[question 1]
[question 2]
[question 3]
...
```

with unrestricted bidirectional self-attention would allow Question 1 to affect Question 2, violating the advertised isolation property.

A Jev-like mask is more naturally:

```text
                 ┌──────── question A / candidates A ────────► answer A
                 │
state ─► shared ─┼──────── question B / candidates B ────────► answer B
representation   │
                 ├──────── question C / candidates C ────────► answer C
                 │
                 └──────── question D / candidates D ────────► answer D
```

Each branch can attend to the state. Branches cannot attend to one another.

This is my inference, not a disclosed TypeSafe design, but it fits the observable contract exceptionally well. Architectures such as **Perceiver IO** establish the general precedent: encode input into a shared latent representation and use arbitrary output queries to request outputs with different semantics. Perceiver IO was explicitly designed around flexible queries and structured outputs, although there is no evidence TypeSafe copied that architecture. citeturn21academia1

### A shared prefill plus branch sampler is another especially plausible implementation

There is an even less exotic explanation. Start with a pretrained causal model:

```text
state tokens → expensive shared prefill/KV state
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         question A     question B     question C
         candidates     candidates     candidates
              │              │              │
           logits         logits         logits
```

The state is processed exactly once. Its cached representation is forked. All short question/candidate suffixes run as a batch. Instead of allowing the model to emit arbitrary vocabulary tokens, a classification/scoring head computes only the requested decisions.

This is sufficiently obvious that a community implementation has already demonstrated it. `LFM2.5-2.6B-RLCD`—despite the misleading name—uses **unchanged** LFM2.5 weights, prefills shared context once, branches the attention/convolution state, evaluates allowed answers in parallel, and constructs the typed JSON in Python. Its author explicitly says it is inference-only, uncalibrated, and **not** a reproduction of TypeSafe's RLCD. citeturn25search9

That project is highly informative because it proves that a major part of the Jev execution pattern requires **no mysterious new training algorithm at all**. Shared-context prefill plus parallel finite-choice scoring already eliminates sequential answer generation.

### “Parallel” does not mean constant compute

This also explains an important marketing nuance. TypeSafe says adding questions “barely changes” response time, but the company's own cookbook acknowledges that cost depends strongly on the shared document and that concurrent independent requests narrow the apparent latency advantage relative to serial requests. citeturn17view1

The real computational relationship is closer to:

\[
C_{\text{Jev-like}}
\approx
C_{\text{state}}
+
\sum_{q=1}^{Q}C_{\text{short-query},q}
+
\sum_{q=1}^{Q}C_{\text{candidate-head},q}
\]

instead of the wasteful:

\[
C_{\text{naive cross encoder}}
\approx
\sum_{q=1}^{Q}\sum_{k=1}^{K_q}
C_{\text{state+question+candidate}}.
\]

It is therefore reasonable for latency to be nearly flat when **the state dominates computation** and the GPU has enough unused parallel capacity. It would not remain flat indefinitely as the number and length of questions or candidates grow.

### Jev “confidence” is not the same thing as calibrated correctness probability

There is one important correction to the initial discussion. TypeSafe's `confidence` field on `Choice` and `Score` is **derived from the shape of the probability distribution**, with concentrated distributions receiving greater confidence; `Noul` does not even have a separate confidence field. TypeSafe explicitly calls confidence a statistic calculated from the underlying probabilities. citeturn18view0

So the statement:

> “a Jev confidence of 0.7 should be correct 70% of the time”

is not supported.

The calibration claim properly applies to **probabilities**: predictions assigned probability 0.8 should correspond to outcomes occurring roughly 80% of the time over a suitable population. TypeSafe states this directly in its RLCD primer. citeturn17view4

That distinction needs to survive into any open implementation.

## What the research literature tells us

The surprising conclusion from the literature is that very little of Jev's **conceptual** design requires unknown mathematics. Almost every component has a strong predecessor. What TypeSafe may have invented is a particularly successful integration, training distribution, sampler, or scale recipe.

### The nearest architecture paper is probably GLiClass, not non-autoregressive translation

**GLiClass**, published in 2025, attacks almost exactly the inefficiency that appears in Jev's problem statement. The authors note that generative LLMs are inefficient for zero-shot classification, while conventional NLI/reranking cross-encoders repeatedly process text-label pairs. GLiClass instead supports dynamic class labels and is designed to process multiple labels in a single forward pass; the authors also experiment with PPO for classification from sparse data or feedback. citeturn21academia0

That makes GLiClass a substantially better architectural starting point than most “non-autoregressive LLM” work.

Conceptually:

```text
                       label/query representation
                                  │
                                  ▼
state/document ───────► semantic interaction ──────► score(label)
                                  ▲
                                  │
                       label/query representation
```

Scale that from “document + class labels” to:

```text
state + question + arbitrary criteria → probability distribution
```

and you are already remarkably close to Jev's external semantics.

### Perceiver IO provides the cleanest model of arbitrary parallel output queries

Perceiver IO is another valuable architectural ancestor because its output layer is explicitly driven by **queries specifying the semantics of requested outputs**. It scales input processing through a latent representation and supports output structures with very different sizes and meanings. citeturn21academia1

A purpose-built OpenJev could reinterpret each `(question, candidate)` pair as an output query:

\[
z = E_{\text{state}}(x)
\]

\[
r_{q,k} = E_{\text{query}}(q,c_k)
\]

\[
h_{q,k} = \operatorname{CrossAttend}(r_{q,k}, z)
\]

\[
s_{q,k}=w^\top h_{q,k}
\]

\[
p_{q,k} =
\frac{\exp(s_{q,k}/T)}
{\sum_j \exp(s_{q,j}/T)}
\]

All candidates are scored simultaneously. Different questions are mask-isolated.

That is the architecture I would eventually aim for.

### Traditional non-autoregressive text generation is related, but less directly than the launch discussion suggests

Gu et al.'s **Non-Autoregressive Neural Machine Translation** showed in 2017 that parallel output generation can produce order-of-magnitude inference latency improvements, albeit initially at a quality cost. Ghazvininejad et al.'s **Mask-Predict** later used parallel masked prediction followed by iterative refinement. citeturn21academia3turn21academia2

Those papers prove that sequential token dependence is not inevitable.

But Jev's easier problem is more fundamental: **it does not need to generate a target sequence at all**. There is no reason to solve the hard non-autoregressive-language-generation problem when the legitimate outputs form a finite set.

That is why calling Jev a “parallel decoder” can obscure what is happening. A better open design is a **semantic structured-prediction network**.

### Calibration is a mature field, not an RLCD invention

Calibration predates modern LLMs by decades. Guo et al. famously showed in 2017 that high-performing neural classifiers can nevertheless be poorly calibrated and found simple temperature scaling surprisingly effective in many settings. citeturn22search1

Kuleshov and Liang's 2015 **Calibrated Structured Prediction** is particularly relevant to Jev because it addresses systems with large structured output spaces where users may ask different probability queries over an output. Their work shows how structured predictions can be recalibrated rather than treating raw model scores as trustworthy probabilities. citeturn23search0

So the statement that ordinary softmax plus cross-entropy makes calibration “automatic” needs qualification. Negative log likelihood is a **proper scoring rule** whose population optimum corresponds to the correct conditional distribution under ideal conditions; real finite, misspecified, overparameterized neural networks can still be badly calibrated. That empirical gap is exactly why temperature scaling and the calibration literature exist. citeturn22search1

### Reinforcement learning for calibration is also established prior art

The clearest pre-Jev example is **Rewarding Doubt**. Stangel and colleagues fine-tune LLMs using reinforcement learning with a reward based on the logarithmic scoring rule so that models are punished for both overconfidence and underconfidence. They report improved calibration and transfer to unseen tasks. citeturn22academia6

Two 2026 papers make the case even stronger. **Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning** reports that conventional RL with verifiable rewards can make decision models overconfident and proposes an RL formulation that explicitly modifies decision-token probabilities, reducing reported ECE while retaining accuracy. citeturn23academia6

**Decoupling Reasoning and Confidence** independently argues that RLVR can produce severe calibration degeneration and identifies gradient conflict between maximizing answer correctness and minimizing calibration error, motivating its DCPO method for separating the objectives. citeturn23academia7

An additional recent study on sycophancy-inducing GRPO finds a directionally worse calibration result, although its reported ECE degradation was small and not statistically significant at the training budget studied. That is useful context because it shows that statements like “RLHF/RL always destroys calibration” are too strong; the effect depends on training regime and evidence. citeturn23academia5

Therefore, **RLCD as a high-level idea is not by itself a research novelty**. The novelty, if any, would have to lie in TypeSafe's particular reward, training procedure, data mixture, architecture interaction, or empirical scale.

### ECE should not be the main training target

Expected Calibration Error is convenient, but modern work gives good reasons not to treat ordinary binned ECE as the gold standard.

Hu and Wu's **Calibration Error for Decision Making** introduces Calibration Decision Loss, or CDL, which asks a more operational question: how much decision payoff could a downstream decision-maker gain by replacing the model's probabilities with calibrated ones? It separates CDL theoretically from standard ECE. citeturn22academia4

Hartline, Hu, and Wu's 2026 COLT paper goes further. They show that standard finite-sample calibration measures can create incentives to misreport probabilities and introduce **Averaged Two-Bin Calibration Error**, a perfectly and strictly truthful batch calibration measure. citeturn22search0

That matters immensely for an OpenJev. If the objective is trustworthy machine-to-machine probabilities, we should not optimize only a visually attractive ECE number.

### Existing “open Jev” projects prove different pieces, not the entire product

There are currently two especially useful community experiments.

`AlexWortega/openjev` is a **Qwen3.5-4B NLI cross-encoder**. It is trained with plain three-way cross-entropy and returns probabilities for entailment, contradiction, and neutral. Its author demonstrates reranking, grading, guard-like applications, and game control. citeturn26search0

`monotykamary/LFM2.5-2.6B-RLCD`, by contrast, leaves its LFM weights unchanged and explores the **inference architecture**: one shared prefill, branched model state, parallel finite-choice evaluation, Python-side typed serialization. The author explicitly labels calibration and RL training as missing. citeturn25search9

They are complementary:

| Capability | `openjev` | LFM PCD | TypeSafe Jev |
|---|---:|---:|---:|
| Semantic decision model | Yes | Existing LM | Yes |
| Dynamic finite decisions | Via NLI reformulation | Yes | Yes |
| Shared-state compute reuse | Not the central design | **Yes** | **Yes** |
| Mixed `Choice` / `Score` / binary API | No native equivalent | Partial | **Yes** |
| Host-side schema guarantee | Possible | **Yes** | **Yes** |
| No free-form output generation | **Yes** | **Yes** | **Yes** |
| Explicit calibration training | No | No | Claimed |
| Reproduces RLCD | No | No | Proprietary |
| Architecture published | Yes | Yes | **No** |

The existing projects therefore already answer an important research question: **the interface and much of the speed story are reproducible without access to Jev's internals.**

## A plausible open-source Jev architecture

I would build the reproduction in two generations rather than attempting a novel foundation model immediately.

### Start with a shared-prefill decision model

The first serious prototype should use an existing pretrained base model but never ask it to produce prose.

For each request:

```text
Request
│
├── state: shared arbitrary text / JSON / document
│
└── questions
      ├── Choice(instruction, candidates[])
      ├── Noul(instruction)
      └── Score(instruction, ordered_levels[])
```

Internally:

```text
                      ┌──────────────────────────────┐
state tokens ────────►│ shared pretrained backbone │
                      └──────────────┬───────────────┘
                                     │
                                state cache H
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
       Question A              Question B              Question C
       + candidates            + candidates            + candidates
             │                       │                       │
             ▼                       ▼                       ▼
       parallel scorer         parallel scorer         parallel scorer
             │                       │                       │
      categorical p(A)          Bernoulli p(B)         ordinal p(C)
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     ▼
                              deterministic serializer
                                     ▼
                               typed response object
```

An implementation based on Qwen-class or LFM-class open weights could reuse a shared causal prefill cache almost immediately. Hugging Face already exposes Qwen3.5 sequence-classification support, and the existing `openjev` model confirms that a Qwen3.5 base can be converted into an NLI classifier rather than a text generator. citeturn25search1turn26search0

The crucial optimization is to **avoid projecting every decision hidden state through a 200,000-plus-token language vocabulary** when the caller has supplied only two, five, or fifty-five legal decisions. The decision model should have a compact scalar/candidate scoring head.

### Then move to a native dynamic-query architecture

The more ambitious version should stop treating the pretrained causal model's generation machinery as fundamental.

A good research architecture would contain:

\[
H_x = E_x(x)
\]

for one state representation, and:

\[
R_{q,k}=E_q(q,c_k)
\]

for every question/candidate description.

Then:

\[
Z_{q,k}=D(R_{q,k},H_x)
\]

where \(D\) is a lightweight cross-attention/query stack.

Finally:

\[
s_{q,k}=f(Z_{q,k})
\]

and:

\[
p_{q,k}=\operatorname{softmax}_k(s_{q,k}).
\]

For a Noul:

\[
p_{\text{yes}}=\sigma(s_q).
\]

For an ordered Score rubric, I would initially retain the full categorical distribution rather than collapse the problem prematurely:

\[
p(l_0),p(l_1),...,p(l_m)
\]

and compute the displayed score from the expected level or another explicitly documented transformation.

This combines the most relevant features of Perceiver IO's query-driven output design and GLiClass's dynamic-label classification while preserving TypeSafe's stated question independence. citeturn21academia1turn21academia0

### Use block masks to guarantee question isolation

The attention graph should be explicit:

```text
STATE TOKENS
   ▲ ▲ ▲ ▲
   │ │ │ │
   │ │ │ └──────────── Question D / candidates
   │ │ └────────────── Question C / candidates
   │ └──────────────── Question B / candidates
   └────────────────── Question A / candidates

A cannot attend B/C/D
B cannot attend A/C/D
C cannot attend A/B/D
D cannot attend A/B/C
```

This would give the open implementation a meaningful semantic guarantee:

\[
P(y_A \mid x,q_A)
\]

must not become:

\[
P(y_A \mid x,q_A,q_B,q_C,q_D).
\]

That is much more valuable than merely saying the questions happened to run concurrently. It makes batching a serving optimization rather than a change in model semantics, matching the behavior TypeSafe demonstrates in its cookbook. citeturn17view1

### Treat types as host-language objects, not model output tokens

The core API might internally normalize everything into one structure:

```python
DecisionQuestion(
    kind="categorical",
    instruction="...",
    candidates=[
        Candidate(key="...", description="..."),
        ...
    ],
    ordered=False,
)
```

Then:

- `Noul` becomes a two-outcome or single-logit binary decision.
- `Choice` becomes an unordered categorical decision.
- `Score` becomes an ordered categorical decision.

The model returns tensors. Application code turns those tensors into the API structure.

This makes the schema guarantee completely independent of model intelligence. It also means “zero malformed JSON” is not something the neural network needs to learn.

### Do not make strings part of the answer space

This is one of Jev's strongest ideas.

For a classifier serving:

```text
["refund", "technical", "security"]
```

the actual network output should be:

```text
[0.08, 0.17, 0.75]
```

plus the selected integer:

```text
2
```

Host code maps `2 → "security"`.

No tokenizer ever needs to generate the characters `s e c u r i t y`.

The reduction in unnecessary work can be enormous when compared with long reasoning traces or verbose structured output, although TypeSafe's exact 40–200× claims are task-, comparison-, and serving-dependent rather than universal constants. The company's own launch material acknowledges that its comparisons are particularly favorable for System-One-shaped workloads. citeturn1view0turn20view1

## A step-by-step reproduction program

### Establish the contract before training anything

The first artifact should be an open API-compatible decision schema, not a model.

Implement:

```text
State
Question[]
 ├── Noul
 ├── Choice
 └── Score

→ Answer[]
```

with deterministic validation and serialization.

TypeSafe itself publishes a **System One Adapter**, a drop-in version of its evaluation interface backed by ordinary LLM APIs. It supports the same conceptual primitives and can request either probability distributions or discrete answers. That repository is practically a ready-made evaluation harness for an open reproduction because it lets the same workload be run against LLMs and Jev-shaped systems. citeturn25search0

The project's first invariant should be:

```text
valid input schema  →  valid output schema
```

with schema errors impossible after validation.

### Build a no-training logit-scoring baseline

Before fine-tuning, determine how much of Jev can be explained purely by changing inference.

Take an open pretrained model and implement:

1. tokenize/prefill state once;
2. cache its hidden/KV representation;
3. fork the cache for all questions;
4. evaluate the question text in parallel;
5. evaluate allowable outcomes rather than generate text;
6. normalize scores into probabilities;
7. serialize outside the model.

For a binary question, a crude initial implementation could score textual hypotheses such as:

```text
YES: proposition is true
NO: proposition is false
```

For `Choice`, score each candidate description.

For `Score`, score each rubric description.

This is essentially the experiment already being explored by the LFM parallel-constrained-decoding project, and it gives the project an essential baseline: **how much speed can we get before changing a single weight?** citeturn25search9

### Establish a strong NLI cross-encoder baseline

Then reproduce and extend what `openjev` does.

Natural-language inference is an excellent transformation for arbitrary decisions because many queries can be rewritten as:

```text
premise:    state + question context
hypothesis: candidate proposition
```

and evaluated as entailment/non-entailment.

The existing `openjev` checkpoint demonstrates exactly this with Qwen3.5-4B, three NLI classes, and plain cross-entropy. citeturn26search0

But benchmark it honestly. An ordinary cross-encoder evaluates:

\[
(x,q,c_1),
(x,q,c_2),
...
(x,q,c_K)
\]

which repeatedly consumes \(x\).

That should be your **quality baseline, not your final serving architecture**.

### Train a general dynamic-label decision model

The training representation should look approximately like:

```json
{
  "state": "...",
  "question": {
    "type": "choice",
    "instructions": "...",
    "criteria": [
      {"key": "A", "description": "..."},
      {"key": "B", "description": "..."},
      {"key": "C", "description": "..."}
    ]
  },
  "target": "B"
}
```

The same semantic task should be rendered many different ways.

For multiclass source data:

```text
class 0 = sports
class 1 = finance
class 2 = politics
```

randomize:

- order;
- symbolic keys;
- label descriptions;
- wording of the instructions;
- number and composition of distractors.

Otherwise the model learns a conventional fixed classification problem rather than **zero-shot schema following**.

This zero-shot dynamic-class behavior is important enough that Almeida specifically preferred “zero-shot” over “instruction-tuned” when discussing how to characterize Jev. citeturn24search6turn24search19

### Build the training mixture around judgments, not chatbot conversations

A realistic corpus should contain a broad range of bounded decisions:

| Family | Training transformation |
|---|---|
| NLI / factual entailment | `Noul` and `Choice` |
| sentiment / toxicity / moderation | `Choice`, `Noul`, ordinal `Score` |
| intent and ticket routing | dynamic `Choice` |
| topic classification | dynamic `Choice` |
| retrieval / reranking | candidate `Choice` |
| factual verification | `Noul` |
| document-policy matching | `Noul` / `Choice` |
| quality grading | ordinal `Score` |
| risk / urgency | ordinal `Score` |
| preference data | pairwise or multiway `Choice` |
| tool/action selection | dynamic `Choice` |
| agent trace checking | multiple independent `Noul`s |
| anomaly / compliance checks | `Noul` plus rubric `Score` |

The important training trick is **task diversity under one common decision language**. GLiClass follows a similar generalist direction rather than training one classifier per task, which is one reason it is such relevant prior work. citeturn21academia0

Every training batch should also contain cases where many questions share one state. Apply the same isolation mask at training and inference time.

### Explicitly train “none,” “unknown,” and insufficient-evidence behavior

A closed candidate set creates a nasty failure mode:

```text
Which country is the user in?
A: France
B: Germany
C: Italy
```

when the evidence actually indicates Canada.

Softmax still satisfies:

\[
p_A+p_B+p_C=1.
\]

The model is mathematically forced to allocate all probability to wrong alternatives.

Anthony Maio correctly identifies this as an important Jev deployment problem: bounded output schemas need explicit paths such as “unknown,” “none of the above,” or “insufficient evidence” when those states are possible. citeturn20view1

During training, deliberately remove the correct candidate from a fraction of examples and require selection of an abstention candidate.

This is essential for meaningful calibration.

### Distill semantic ability from stronger models, but do not confuse teacher probabilities with ground truth

This is likely where the real cost of a competitive OpenJev lies.

A compact decision model can be trained from:

- objective labeled datasets;
- human-labeled judgments;
- multiple independent human annotators;
- consensus labels;
- synthetic schemas generated from existing classification data;
- larger teacher models.

Teacher distillation is particularly attractive because an expensive frontier model can generate millions of varied state/question/schema examples offline.

However, teacher probability labels do **not** automatically give the student true calibration. A student matching:

\[
p_{\text{student}} \approx p_{\text{teacher}}
\]

inherits the teacher's probability errors as well as its semantic competence.

For calibration-sensitive examples, true observed outcomes, repeated annotation, or carefully curated objective labels are considerably more valuable.

Almeida's comment that the **data may be more interesting than the architecture** is therefore credible: the difficult asset may be the giant collection of heterogeneous decisions and outcome labels on which the model learned what 0.2, 0.5, and 0.9 should mean across domains. citeturn24search0

### Begin calibration with proper supervised scoring rules

I would **not** begin by inventing RLCD.

Start with ordinary probabilistic training.

For Choice:

\[
L_{\text{NLL}}=-\log p(y).
\]

For binary Noul:

\[
L_{\text{BCE}}
=
-y\log p
-(1-y)\log(1-p).
\]

Add Brier loss:

\[
L_{\text{Brier}}
=
\sum_k(p_k-y_k)^2.
\]

For ordered Score tasks, add an ordinal component that penalizes predictions more severely as they move farther from the true rubric level.

A practical initial mixture might be conceptually:

\[
L =
L_{\text{NLL}}
+
\lambda_B L_{\text{Brier}}
+
\lambda_O L_{\text{ordinal}}.
\]

NLL and Brier-style losses are proper scoring approaches to probabilistic prediction; after training, held-out temperature scaling is an extremely strong baseline for calibration. Guo et al.'s classic experiments are the obvious reference point. citeturn22search1

### Add RL only after establishing that supervised calibration is insufficient

There is a key conceptual point here:

**A native classifier does not need reinforcement learning merely to learn calibrated probabilities.**

Unlike a generative LLM that has to learn to *express* “I am 70% confident” through language tokens, our architecture directly exposes \(p(y\mid x)\). The gradient of the proper scoring objective is available directly.

RL becomes interesting when the training signal is instead:

- delayed task success;
- human feedback;
- downstream payoff;
- abstention utility;
- workflow outcomes;
- bandit feedback;
- or another non-differentiable decision criterion.

That means an open project can test a very revealing ablation:

```text
CE/NLL
vs.
CE + Brier
vs.
CE + post-hoc calibration
vs.
calibration-aware RL
vs.
decision-utility RL
```

If calibration-aware RL adds nothing after strong supervised probabilistic training, TypeSafe's RLCD may be primarily a useful training/product framing. If it materially improves zero-shot calibration under domain shift without sacrificing accuracy, then it becomes genuinely interesting.

**Rewarding Doubt** gives a published blueprint for a log-scoring-rule reward, while the 2026 calibration-aware RL and DCPO papers provide more recent approaches and warnings concerning conflict between accuracy-oriented RL and calibration. citeturn22academia6turn23academia6turn23academia7

### Add a decision-theoretic RLCD-like objective

For the more experimental version, I would go beyond “minimize ECE.”

Suppose the model predicts a probability \(p\), and an application chooses action \(a\) based on an action-dependent utility:

\[
U(a,y).
\]

Train on random downstream decision problems where the rational action changes with \(p\).

For example:

```text
correct approval      +1
incorrect approval   -10
manual review         -0.1
```

versus:

```text
correct routing       +1
incorrect routing     -0.2
manual review         -1
```

The same probability distribution must support different rational decisions under different loss matrices.

An RLCD-like reward could combine:

\[
R =
R_{\text{proper-score}}
+
\alpha R_{\text{decision-utility}}
-
\beta R_{\text{overconfidence}}
\]

rather than directly rewarding a low binned ECE estimate.

That direction has strong intellectual support from Calibration Decision Loss, whose entire motivation is that calibration matters because downstream actors consume predicted probabilities to make decisions. citeturn22academia4

I would call this something neutral such as **Decision-Calibrated Post-Training** rather than copying TypeSafe's RLCD name until the actual Jev algorithm is published.

### Build the native parallel query network

Once the training pipeline works with a standard backbone, replace repeated candidate evaluation with the native architecture discussed above.

The key computational objective should be:

\[
O(\text{state encoding})
+
O(\text{all query tokens})
+
O(\text{all candidate interactions}),
\]

not:

\[
O(\text{state length}
\times
\text{question count}
\times
\text{candidate count}).
\]

Candidate representations and question representations can be ragged tensors. A block-diagonal attention mask gives isolation. A fused kernel or FlashAttention-style implementation can process many independent branches efficiently in the same launch.

At this stage, GLiClass is particularly worth studying in source because it is explicitly designed to avoid the traditional cross-encoder's label-by-label inference penalty. citeturn21academia0

### Remove the language-model vocabulary head from production inference

A causal LLM may carry a huge language head because it must answer:

\[
P(\text{next token}\mid h)
\]

across its vocabulary.

OpenJev does not need that production objective.

Once distillation/training is complete, the serving path should terminate at compact decision projections. This provides another large opportunity for lower memory bandwidth and inference cost, especially if the base architecture otherwise contains a very large vocabulary.

### Quantize and optimize the serving runtime only after correctness

The final runtime should support:

- shared-state prefill caching;
- ragged dynamic candidate sets;
- batched independent question branches;
- block-sparse or block-diagonal attention;
- low-precision weights/activations where accuracy permits;
- compile-time/fused decision heads;
- no textual decoder loop;
- no output-token KV-cache growth;
- deterministic host-side response assembly.

This is where the difference between “an NLI model that resembles Jev” and “a Jev-like serving system” becomes substantial.

## How to determine whether we actually reproduced Jev

An open implementation should not declare success because it can play Doom or return valid JSON. The benchmark must independently test **quality, calibration, scaling, and software behavior**.

### Quality must be measured separately from calibration

Report ordinary task performance:

\[
\text{accuracy},\quad
\text{macro-F1},\quad
\text{AUROC},
\]

as appropriate.

For ranking:

\[
\text{MRR},\quad
\text{NDCG},\quad
\text{Recall@K}.
\]

For ordered Score tasks, include ordinal error.

A perfectly calibrated useless predictor can simply output class base rates. Calibration therefore never replaces discrimination or accuracy, a point also emphasized in independent analysis of Jev. citeturn20view1

### Calibration requires several metrics, not one ECE number

I would report at least:

\[
\text{NLL}
\]

\[
\text{Brier Score}
\]

\[
\text{ECE}
\]

plus reliability diagrams, classwise calibration, and a modern truthful calibration metric such as ATB where applicable. For workflows in which probabilities actually drive consequential actions, add a decision-theoretic evaluation inspired by CDL. citeturn22search0turn22academia4

The most convincing plot is still simple:

```text
Predicted probability      Actual frequency
0.1                        ≈ 0.1
0.2                        ≈ 0.2
0.3                        ≈ 0.3
...
0.9                        ≈ 0.9
```

but it needs adequate sample sizes and uncertainty intervals.

### Measure calibration under distribution shift

This is where the strongest Jev claim should be tested.

Train on one set of domains and measure:

- unseen topics;
- unseen task formulations;
- unseen candidate vocabularies;
- longer documents;
- domain terminology;
- adversarially misleading evidence;
- shifted class frequencies;
- schema paraphrases.

A model that is beautifully calibrated in-distribution but turns every OOD example into `0.99` is unsuitable as an automated control primitive.

TypeSafe's own docs sensibly advise users to choose thresholds based on performance on their use case rather than treating the returned confidence as a universal guarantee. citeturn18view0

### Test the Jev-specific batching invariants

For every test question, evaluate:

```text
A alone
A + B
A + B + C + ...
same questions shuffled
same questions in different request positions
```

Then compare probability vectors.

For a correctly isolated architecture:

\[
P_A^{\text{alone}}
\approx
P_A^{\text{batch}}.
\]

The stricter goal for deterministic inference is equality up to floating-point/kernel nondeterminism.

This directly reproduces TypeSafe's own batching experiment rather than merely measuring throughput. citeturn17view1

### Test label-order invariance

Given:

```text
A = phishing
B = malware
C = benign
```

then reorder:

```text
C = benign
A = phishing
B = malware
```

The semantic probabilities should reorder correspondingly.

The model should not develop “first option” or “last option” biases.

### Test high cardinality directly

Sweep:

\[
K=\{2,4,8,16,32,64,128,255\}.
\]

Measure both:

\[
\text{latency}(K)
\]

and:

\[
\text{accuracy}(K).
\]

A genuinely useful architecture should show graceful scaling rather than requiring a full state recomputation per option.

TypeSafe's 255-choice ceiling provides a concrete target for parity. citeturn1view0

### Measure latency against number of questions, not merely token throughput

Run fixed state lengths such as:

```text
short state
medium state
long document
```

and sweep:

\[
Q=\{1,2,4,8,16,32,64,128\}.
\]

Report:

- p50;
- p95;
- p99;
- requests/second;
- decisions/second;
- GPU memory;
- GPU milliseconds per request;
- cost per million decisions.

The critical graph is:

```text
latency
  │
  │                        normal repeated cross-encoder
  │                    /
  │                /
  │            /
  │        /
  │   ____ OpenJev shared-state architecture
  │__/
  └──────────────────────────────────
            number of questions
```

That is the experiment that establishes whether the parallel architecture really captures Jev's central economic advantage.

### Compare against four baselines, not one

A serious benchmark should compare:

| System | Why it matters |
|---|---|
| Frontier LLM + JSON/structured output | conventional solution |
| Same LLM + constrained/logit decision scoring | isolates generation overhead |
| `openjev`-style NLI cross-encoder | strong ordinary classifier |
| OpenJev shared-state decision model | proposed architecture |
| TypeSafe Jev | target, when API access is available |

TypeSafe's own System One Adapter was explicitly built to run LLMs behind its decision API, so it can provide much of the common benchmarking harness. citeturn25search0

### Use TypeSafe's public Jev result as a target, not as ground truth

TypeSafe's internal workflows compare security incident response, agent-trace observability, invoice processing, and customer-service workflows. Independent analysis of the published dashboard notes aggregate Jev agreement around 67.8%, while some stronger comparator configurations reached roughly 73–74%; importantly, TypeSafe's references were themselves generated from frontier-model consensus rather than independent real-world ground truth. citeturn20view1

That benchmark therefore demonstrates an interesting cost/latency/quality trade-off, but it should **not** be the ultimate OpenJev objective.

Every's independent mini-test reaches a similarly useful conclusion: the speed benefit is tangible, while a frontier model caught one defect Jev missed. citeturn20view0

The reproduction should aim for an entire **Pareto frontier**, not one leaderboard number:

```text
                         higher accuracy
                               ▲
                               │        frontier LLM
                               │           ●
                               │
                               │     ● Jev?
                               │
                               │   ● OpenJev-native
                               │
                               │ ● NLI baseline
                               │
                               └────────────────────► lower latency / cost
```

### The decisive experiments can reveal what TypeSafe's secret sauce actually is

Several outcomes would be especially informative.

**If shared-prefill constrained scoring with unchanged open weights approaches Jev's accuracy, speed, and calibration**, then much of Jev is an inference-system insight: stop generating strings, reuse state computation, and expose distributions directly.

**If `openjev`-style NLI training matches Jev's judgment quality but is much slower**, then the major contribution is likely the sampler/architecture.

**If a native GLiClass/Perceiver-like decision model matches the latency but remains materially less capable**, then TypeSafe's data, scale, or teacher-distillation process is likely crucial.

**If NLL/Brier plus temperature calibration matches Jev's probability quality**, then RLCD may contribute less than the branding suggests.

**If an RLCD-like stage produces dramatically better out-of-domain ATB/CDL and risk-coverage behavior without harming accuracy**, then TypeSafe's focus on calibration-aware post-training points toward a genuine important advance, even if the exact proprietary algorithm differs.

That is the research program I would use to reverse-engineer the product empirically rather than trying to guess one hidden architecture from marketing terminology.

## Bottom line on reproducibility

The evidence now supports a much sharper conclusion than “Jev is a mysterious new non-autoregressive LLM.”

**Jev appears to combine four ideas:**

| Layer | Novelty assessment |
|---|---|
| Typed finite decision interface | Known idea, unusually well productized |
| No autoregressive output generation | Straightforward for finite-output tasks |
| Shared-state parallel dynamic classification | Established ingredients; integration may be novel |
| Calibration-focused post-training | Strong prior art; Jev's actual recipe unknown |

The public literature already gives us almost every ingredient required for an open reconstruction. Non-autoregressive generation shows why sequential decoding is avoidable, although Jev need not generate a sequence at all. Perceiver IO gives a blueprint for shared representations queried by arbitrary outputs. GLiClass gives a particularly close blueprint for generalist dynamic-label classification in one forward pass. Classical calibration work gives NLL/Brier/recalibration foundations. Rewarding Doubt and 2026 calibration-aware RL work demonstrate that reinforcement-learning objectives can explicitly shape probability calibration. CDL and ATB provide stronger ways to evaluate that calibration than merely reporting ordinary ECE. citeturn21academia3turn21academia1turn21academia0turn22search1turn22academia6turn22academia4turn22search0

Meanwhile, community experiments have already covered two major halves of the engineering problem: `openjev` shows that a modern pretrained model can be converted into a non-generative NLI decision engine with a standard classification head, while the LFM parallel-constrained-decoding project shows that shared context can be prefetched once and branched into parallel finite-choice evaluations without retraining the model at all. citeturn26search0turn25search9

What is **not** public is precisely what one would need to claim an exact reproduction: TypeSafe's architecture, scale, weights, training mixture, RLCD reward/objective, calibration methodology, and serving implementation. Almeida's statement that the architecture is still being held back and that a paper is only being discussed is the strongest primary-source reason not to overstate what can presently be reverse-engineered. citeturn24search0

I would therefore define an open-source reproduction project around this architecture:

```text
                         OPENJEV

       ┌──────────────────────────────────────┐
       │        Pretrained semantic trunk     │
       │  decoder-prefill or native encoder  │
       └──────────────────┬───────────────────┘
                          │
                    shared state
                    representation
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      query bank      query bank       query bank
       Choice           Noul             Score
          │               │                │
    candidates[]       true/false       levels[]
          │               │                │
          └───── parallel cross-attention ─┘
                          │
                    scalar logits
                          │
              calibrated distributions
                          │
           deterministic typed serializer
                          │
       ┌──────────────────┼─────────────────┐
       ▼                  ▼                 ▼
    Choice             Noul              Score
 + probabilities    probability       distribution
 + concentration                       + value
```

Train it first with **proper supervised probabilistic losses**, broad zero-shot dynamic-schema data, label permutation, explicit abstention examples, and teacher distillation. Add calibration-aware RL only after a supervised baseline exists. Optimize inference around **one state computation plus many isolated query branches**, not parallel text generation. Evaluate calibration with NLL, Brier, reliability curves, ATB, distribution-shift tests, selective risk, and decision utility—not ECE alone. Use TypeSafe's adapter to run identical workloads against conventional LLMs, the open NLI baseline, the new architecture, and Jev itself. citeturn25search0turn22search0turn22academia4

The most technically interesting result may ultimately be that **“Jev architecture” is less exotic than it sounds**. For bounded software decisions, autoregressive language generation is unnecessary in the first place. The genuinely difficult part is building a broad semantic model that can understand arbitrary caller-defined decisions, produce useful probability distributions in one inexpensive pass, and remain calibrated well enough that software can safely act on those probabilities. TypeSafe has demonstrated a compelling product built around that premise; it has **not yet published enough evidence to establish that its particular architecture or RLCD algorithm is indispensable to achieving it**. citeturn17view4turn20view0turn20view1turn24search0

---

## Phase 2B Empirical Benchmark: Qwen3.5-4B Decision Head & Baseline Readout (Run 20260917T205849Z)

An empirical benchmark run was conducted on **September 17, 2026** (run ID: `20260917T205849Z`, archived in `research/opendecision_phase2b_20260917T205849Z`) probing `Qwen/Qwen3.5-4B-Base` (commit `1001bb4d826a52d1f399e183466143f4da7b741b`) on an **NVIDIA L4 GPU** with native BF16 (`torch.bfloat16`). LoRA was disabled.

The primary finding is that a **frozen Qwen3.5-4B backbone coupled with a lightweight linear classification head** achieves **87.67% matched** and **87.33% mismatched** test accuracy on MultiNLI, trained on 2,400 examples. Recomputation of accuracy, negative log-likelihood (NLL), and Brier scores from the archived prediction tensors confirms the reported metrics.

This run establishes an immutable baseline for the decision engine, clarifies the computational boundary between classification and text generation, and exposes concrete engineering constraints for porting the model to Rust.

### 1. The Frozen Backbone is Viable Without Generation

Seven trained-head architectures (spanning last-token, mean, max, and learned attention pooling paired with linear and MLP heads) were trained on frozen backbone representations. In the MultiNLI evaluation:

| Pooling & Head | Matched Accuracy | Mismatched Accuracy | Matched NLL | Mismatched NLL | Matched Brier | Matched ECE |
|---|---|---|---|---|---|---|
| **Last token + linear** (`winner`) | **87.67%** | **87.33%** | **0.3345** | **0.3175** | **0.1834** | **0.0298** |
| Last token + MLP | 87.83% | 88.67% | 0.3361 | 0.3071 | 0.1835 | 0.0472 |
| Mean + linear | 74.67% | 76.33% | 0.6048 | 0.5779 | 0.3469 | 0.0430 |
| Mean + MLP | 75.33% | 76.17% | 0.5952 | 0.5643 | 0.3470 | 0.0585 |
| Max + linear | 84.83% | 85.33% | 0.3920 | 0.3644 | 0.2171 | 0.0228 |
| Max + MLP | 86.00% | 86.50% | 0.3745 | 0.3591 | 0.2097 | 0.0366 |
| Learned attention + MLP | 86.00% | 84.83% | 0.3591 | 0.3565 | 0.2041 | 0.0474 |
| *Untuned finite-code baseline* | *78.83%* | *81.50%* | *0.5122* | *0.4988* | *0.3021* | *0.0319* |
| *Class-prior baseline* | *33.17%* | *33.00%* | *1.0995* | *1.0992* | *0.6673* | *0.0102* |

#### Key Analytical Takeaways

1. **Supervised Head Training Yields a Real Decision Boundary**: The untuned finite-code baseline (restricting the original language model vocabulary projection to label tokens `A`, `B`, `C`) achieves 78.83% matched and 81.50% mismatched accuracy. Training the linear head improves performance by **+8.83** and **+5.83 percentage points**, respectively. This demonstrates that supervised head training is genuinely extracting a superior decision boundary from the frozen representation, not merely reformatting tokens.
2. **Pooling Hierarchy**: Contrary to the prior hypothesis that learned attention pooling would dominate, **last-token pooling performed best**. Mean pooling performed markedly worse (~74.7% matched), and learned attention pooling (86.00%) lagged behind last-token variants while adding architectural complexity. For the initial Rust engine design, last-token pooling is the obvious, high-performing choice.
3. **Linear vs. MLP Selection Rationale**: The notebook selection policy was strictly declared as **lowest development-set NLL**, not peak test accuracy. Under this rule, `last_linear_seed17` was chosen (`dev NLL = 0.3459`). While `last_mlp_seed17` scored marginally higher on held-out test accuracy (+1 matched example, +8 mismatched examples on a 600-example test split), selecting models post-hoc based on test-set accuracy would degrade the test set into a development partition. The reported 95% cluster-bootstrap confidence interval for the linear head spans roughly **85.2% to 90.2%**, meaning the difference between linear and MLP is within expected sampling noise. Both variants should be retained for multi-seed validation.

### 2. Parameter Audit: Removing Generation Saves Compute, Not Weights

A rigorous parameter audit of `Qwen/Qwen3.5-4B-Base` in this configuration yielded unambiguous accounting:

| Measurement | Recorded Result |
|---|---|
| Text backbone parameters (`Qwen3_5TextModel`) | 4,205,751,296 |
| Text causal model parameters (`Qwen3_5ForCausalLM`) | 4,205,751,296 |
| Logical vocabulary projection (`lm_head`) | 635,699,200 coefficients |
| Additional parameters removed by omitting LM head | **0** |
| Vision tower loaded | False |
| Backbone trainable parameters after freezing | 0 |
| Trainable parameters in selected linear head | **7,683** ($2560 \times 3 + 3$) |

Because `embed_tokens` and `lm_head` share weights (input/output weight tying), the matrix weights remain strictly necessary to ingest input tokens, even when the output projection is bypassed. Dropping the generation loop eliminates vocabulary matrix multiplications and autoregressive decoding, but does **not** shrink model storage on disk.

For the selected `last_linear_seed17` head, the classifier consists solely of:
\[
2560 \times 3 + 3 = 7{,}683 \text{ parameters}
\]
The head checkpoint (`head.safetensors`) is approximately **51.5 KB**. The associated feature-mean and feature-standard-deviation vectors are fixed buffers computed over the training set, not additional transformer layers.

#### Memory Profile
Under PyTorch BF16 on an NVIDIA L4, the resident model allocation was **8,039 MiB (~7.85 GiB)**. Single-example inference added **~37 MiB peak** dynamic allocation. Consequently, substantial VRAM reductions cannot come from stripping the generation head; they must come from quantization (e.g. GGUF / AWQ) or smaller backbone variants.

### 3. Execution Workload and Latency Scaling

Single-example decision latency was measured at **80.75 ms median** (Python end-to-end path including tokenization, device transfer, and serialization was **81.05 ms median**).

Comparing the decision head to generative execution over the same backbone clarifies the latency dynamics:

| Workload | Median Latency | Generative $\div$ Decision Ratio |
|---|---|---|
| **Decision Head** (1 forward pass) | **80.75 ms** | — |
| Generate 1 token | 91.32 ms | 1.13× |
| Generate 8 tokens | 533.54 ms | 6.61× |
| Generate 32 tokens | 2,035.35 ms | 25.21× |

> [!IMPORTANT]
> The latency advantage is heavily dictated by how much text the alternative generates. Avoiding a 32-token generated paragraph yields a **~25× speedup**. Avoiding a single categorical token yields only a modest **~1.13× speedup** (~10.5 ms savings). The principal benefit over a 1-token generative baseline is not raw latency, but **decision quality and calibration** (+8.83 pp accuracy over untuned token projection).

#### Independent Batching Throughput
Throughput measurements on independent examples showed:
- **Batch size 1**: 12.38 decisions/second (80.75 ms)
- **Batch size 4**: 25.79 decisions/second (155.11 ms)
- **Batch size 8**: 24.24 decisions/second (329.98 ms)

Because the padded input lengths varied across batches (94 tokens at $B=1$, 104 at $B=4$, 128 at $B=8$), batch size 4 cannot be declared universally optimal. Crucially, this measured independent batching, **not** shared-prefill branching.

### 4. Calibration & Temperature Scaling Observations

Temperature scaling was fitted on a separate 300-example calibration set, finding $T = 1.1441$. However, applying temperature scaling to the held-out test sets resulted in slightly worse probability metrics:

| Metric | Matched (Raw) | Matched (Temp-Scaled) | Mismatched (Raw) | Mismatched (Temp-Scaled) |
|---|---|---|---|---|
| **NLL** $\downarrow$ | **0.3345** | 0.3392 | **0.3175** | 0.3233 |
| **Brier score** $\downarrow$ | **0.1834** | 0.1854 | **0.1801** | 0.1818 |
| **ECE** (15 bins) $\downarrow$ | **0.0298** | 0.0380 | **0.0311** | 0.0398 |
| Accuracy | 87.67% | 87.67% | 87.33% | 87.33% |

Test accuracy was unchanged, but log-likelihood, Brier score, and ECE all degraded marginally under the fitted temperature. This indicates that on a small calibration partition (300 examples), post-hoc temperature scaling can overfit or encounter distribution shift.

In addition, the class-prior baseline achieved an ECE near **0.010** while producing a useless **33% accuracy**. This underscores the vital principle: **low ECE in isolation does not validate a decision model**. In the API and engine, raw probabilities should be preserved, and calibration transforms should be explicitly identified as `temperature_scaled` rather than unconditionally stamped as "calibrated".

### 5. Batching Invariance Analysis

The verification check between a sample evaluated individually versus in a padded batch revealed:
- **Maximum probability difference**: `0.012738` (~1.27 percentage points) over 2 tested examples.

In PyTorch, batched and non-batched floating-point kernels are not bitwise identical due to non-associative floating-point summation and padding masks. However, before certifying deterministic batch invariance for production Jev workloads, three effects must be empirically isolated:
1. **Duplicate in same-length batch**: isolates kernel summation order without padding.
2. **Identical sample with variable padding**: isolates attention mask and positional encoding handling.
3. **Heterogeneous production batches**: isolates real-world serving batch interactions.

### 6. Reference Specification for Rust Port

The verified Phase 2B pipeline is now fully specified for engine implementation:

```text
Prompt ("nli-described-abc-v1")
  │
  ▼
Tokenize (right-padded, pad_token_id: 248044)
  │
  ▼
Frozen Qwen3.5-4B Backbone (32 layers: 24 linear attention + 8 full attention)
  │
  ▼
Last Non-Padding Token Extraction (2560-dim vector)
  │
  ▼
Fixed Feature Standardization (subtract train mean, divide by train std buffer)
  │
  ▼
Linear Projection (2560 -> 3 logits, 7,683 params)
  │
  ▼
Softmax (in strict label order: [entailment, neutral, contradiction])
```

Exported reference artifacts in `research/opendecision_phase2b_20260917T205849Z/frozen_export/`:
- `manifest.json`: Architecture, tokenization, prompt segments, and label specifications.
- `head.safetensors`: Serialized linear head weights and normalization buffers (~51.5 KB).
- `golden_head_inputs.npz`: Reference input hidden states and expected output logits for Rust parity testing.

#### Experimental Limitations
1. **Fixed 3-Class NLI**: This benchmark tests fixed 3-class NLI, not arbitrary-candidate Jev `Choice`, `Noul`, or ordinal `Score`.
2. **LoRA Disabled**: This run does not determine whether LoRA fine-tuning provides material gains.
3. **No KV-Cache Branching**: Prefill cache sharing across questions was not exercised in this run.

### 7. Strategic Next Steps

The next work is organized into bounded phases:

#### Phase 2C: Advanced Qwen Model Research (Python / Colab)
1. **Stabilize & Validate Baseline**: Run multi-seed evaluations of last-token linear and MLP heads; execute the 3-tier batching invariance diagnostic (same-length duplicates vs variable padding vs heterogeneous batches).
2. **Dynamic Choice & Candidate Scoring**: Extend beyond 3-class NLI to arbitrary caller-defined decision schemas ($K \le 255$ candidates with descriptions, label permutation invariance, explicit unanswerable/abstention cases, and ordinal `Score`).
3. **Controlled LoRA Comparison**: Execute an identical-split comparison with LoRA adapters enabled on Q/K/V to quantify accuracy and calibration deltas against this frozen baseline.
4. **Model Scaling & Memory Optimization**: Evaluate `Qwen3.5-2B-Base` and `0.8B-Base` along with quantization (AWQ / INT8 / GGUF) to compress the ~7.85 GiB VRAM footprint below 4 GiB while preserving decision competence.
5. **Synthetic Domain Benchmarks**: Generate domain-specific evaluation sets (security triage, routing, code review) with controlled ambiguity using teacher LLMs.

#### Phase 2D: Shared-State Prefill & KV-Cache Branching (Python / PyTorch)
- Validate prefilling shared `state` once into the KV / GatedDeltaNet cache and branching across $N$ parallel independent question evaluations.
- Measure latency/throughput scaling and verify inter-question zero-interference isolation.

#### Phase 3: Production Rust Engine & Backends (Rust)
- Port the validated Qwen architecture, dynamic candidate head, and shared-state branching to Rust (`openpick-engine`, `openpick-backends` with Candle/GGUF, `openpick-runtime`).
- Execute parity verification against exported golden vectors (`golden_head_inputs.npz`).