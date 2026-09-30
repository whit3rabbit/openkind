# Family survey: encoder multitask heads

> A small bidirectional encoder with one trained head per named question. One
> state encoding supplies `Noul`, `Choice`, and ordinal `Score` readouts.

## Status in OpenKind

**Surveyed only.** [Indecis](https://github.com/Bornholm/indecis/tree/9930c7db1913818db7cee68c1b42a384f59592e3)
is the reference implementation. OpenKind has no pinned Indecis-trained
profile, Rust head loader, or daemon registration. The existing
[research review](../RESEARCH.md#indecis-small-trained-encoder-with-fixed-and-open-decisions-reviewed-2026-09-29)
records its training method, author measurements, and evidence limits.

This is a separate family from [laya](./laya.md) and
[encoder-instruct-label](./encoder-instruct-label.md): those paths score
request-time criteria with shared readouts. Indecis trains a distinct head
for each question ID and answers the learned schema from one encoder pass.

## Inference and training pattern

| Aspect | Indecis pattern |
|---|---|
| Backbone | Fully fine-tuned ModernBERT encoder. The tested defaults are the multilingual `bekko-embedding-v1-a8m` (4 layers) and `a25m` (13 layers). |
| Fixed readout | One head per named question: binary `Noul`, categorical `Choice`, or ordered-threshold `Score`. Heads share the same pooled state vector and have no text-generation step. |
| Calibration | One temperature per learned question, fitted on held-out labels. |
| Open questions | A separate embedding-similarity path compares state and request-time option descriptions. Its softmax probabilities are uncalibrated. |
| Training | Indecis fully fine-tunes the encoder and heads in Go. Sparse Adam updates the large embedding table; training and calibration remain offline from OpenKind serving. |

The fixed and open paths have different contracts. In the upstream
[decision adapter](https://github.com/Bornholm/indecis/blob/9930c7db1913818db7cee68c1b42a384f59592e3/decision/decision.go),
a learned question ID selects its head and ignores request instructions.
`Choice` can request a subset of learned options, with probabilities
renormalized over that subset. `Score` uses the learned level order. An
unrecognized ID uses [open mode](https://github.com/Bornholm/indecis/blob/9930c7db1913818db7cee68c1b42a384f59592e3/docs/open-categories.md),
which permits changing options but does not provide calibrated probabilities
or a learned semantic-none class.

The author reports 1.5 ms for 15 tokens and 23 ms for 256 tokens using an
int8 prompt-injection model on one Core Ultra 7 265U core. Those are
[Indecis inference measurements](https://github.com/Bornholm/indecis/blob/9930c7db1913818db7cee68c1b42a384f59592e3/docs/inference.md),
not OpenKind request-path or Apple Silicon measurements.

## TypeSafe and Rust fit

Keep the existing `SystemRequest` to `DecisionEngine` to `SystemResponse`
path and `/v1/systemone` route. Indecis's server also exposes that route,
but [its `Noul` response](https://github.com/Bornholm/indecis/blob/9930c7db1913818db7cee68c1b42a384f59592e3/decision/server.go)
includes `confidence`, which OpenKind's
[`NoulAnswer`](../../crates/openkind-core/src/answer.rs) does not. Route
availability alone does not establish TypeSafe wire parity.

OpenKind's [ModernBERT forward](../../crates/openkind-backends/src/families/modernbert.rs)
is a starting point for a native CPU loader. A pinned export would require
parity for Indecis tokenization, paired-input truncation, pooled state vectors,
the three head readouts, and per-question temperatures. Its
[saved format](https://github.com/Bornholm/indecis/blob/9930c7db1913818db7cee68c1b42a384f59592e3/model.go)
contains `config.json`, `tokenizer.json`, `model.safetensors`, and
`indecis.json`; the loader must account for exact fine-tuned embedding rows
or the optional int8 embedding table. Go remains the offline training and
reference implementation; a Rust adapter would load verified local
artifacts and register under an OpenKind alias.

A loadable profile needs an explicit request policy. Match learned question
IDs, types, options, and ordered Score levels to the trained schema, and
reject incompatible requests instead of silently substituting a head. For
Choice, train `__none__` as a real, non-empty option and report its
semantic-none probability mass; do not invent that mass from a similarity
threshold.
Treat open mode as a separately qualified path until its probabilities and
none behavior meet the request contract. Preserve the existing TypeSafe
HTTP and gRPC surfaces without new wire fields.

## Gate for a loadable profile

1. Select a task-specific trained export, pin its revision and artifact
   digests, and establish model and training-data rights. No checkpoint is
   selected by this survey.
2. Compare token IDs, pooled vectors, all three head outputs, calibrated
   probabilities, and final typed answers against the pinned Go reference.
   Add small offline parity fixtures and TypeSafe SDK conformance cases.
3. Test unsupported IDs and option sets, Score order, `__none__`, malformed
   artifacts, and admission behavior. Register the Rust engine only after
   the request-bound response contract passes.
4. Measure matched full requests on the target host, including load time,
   latency, throughput, and peak memory. Run labeled task-quality and
   calibration checks separately; the upstream short-text timing does not
   establish either result.

Until then, this family has no runnable profile or model-registry entry.
