# Proxy cache: a distilling cache in front of a remote Jev API

`openkindd --proxy-cache-upstream <url>` turns the daemon into a transparent
proxy that sits in front of a Jev-compatible API (TypeSafe System One). It
forwards every request it cannot answer confidently and records the teacher's
answers as training rows. Per task — one (tenant, model, instructions,
criteria) tuple — a small linear student learns the teacher's full
probability distributions over frozen request embeddings. Once the student
passes a statistical qualification, confident requests are answered locally
at a bounded disagreement rate.

The feature is experimental and off by default. It changes nothing unless
`--proxy-cache-upstream` is set.

## How it works

1. **Group.** Each request's choice questions are grouped by their
   (instructions, criteria) pair; duplicates share one routing. Noul and
   score questions are always forwarded, and a request containing any of
   them is forwarded wholesale (all-or-nothing per request).
2. **Embed.** The request state becomes one text (canonical JSON for
   objects, verbatim for strings) and one embedding from the configured
   encoder. Requests that are only forwarded are never embedded.
3. **Route.** Each group asks its task engine. The engine answers locally
   only when the student's confidence passes a calibrated threshold, the
   embedding passes a kNN out-of-distribution gate, and the predicted label
   is not deferred. Otherwise the whole request goes upstream.
4. **Forward.** Upstream calls carry the caller's own bearer key (or the
   configured `--proxy-cache-upstream-key` when set). When caller keys are
   forwarded, a key is trusted only after the upstream answered a request made
   with it; raw keys are never stored (only salted hashes, in memory).
5. **Record.** Teacher answers become training rows: state text (optional),
   embedding, full probability distribution, routing decision, and the
   resolved teacher model. A seeded per-request draw reserves a fraction of
   the traffic for calibration (IID channels only; co-deferred rows always
   train).
6. **Train.** A background worker fits a multinomial logistic-regression
   student (full-batch Adam on weighted soft-target cross-entropy with
   early stopping) plus the kNN gate reference.
7. **Calibrate.** A fixed threshold grid is tested strictest-first; the
   loosest threshold whose Clopper–Pearson upper bound on the rate of
   (answered and disagreed) fits the task budget is chosen, fitted at 85%
   of the budget.
8. **Shadow and promote.** The candidate runs alongside production without
   serving. Once enough shadow observations accumulate, the candidate is
   judged at the full budget pooled with its calibration counts, and must
   not lose more than 5% coverage versus production. Passing promotes it;
   failing rejects it and backs off.
9. **Monitor.** A 2% audit slice always goes upstream with the student's
   would-be answer recorded. Drift monitoring scores those rows as served:
   a suspicious agreement drops raises the audit rate and requests a
   retrain; a statistically broken agreement forces fallback to
   teacher-only and restarts training data at the fallback row. A
   `jev-latest` version change (20 consecutive answers from a different
   resolved model) switches the training lineage and drops the old
   lineage's shadow.

Responses carry `x-openkind-cache: local|upstream` and
`x-openkind-cache-detail` (JSON: question id → `student-vN` version or the
forward reason). Local responses keep the Jev wire contract exactly,
including the resolved upstream model name, and report zero token usage.
Any proxy-internal failure fails open to the upstream.

## CLI

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--proxy-cache-upstream <url>` | `OPENKIND_PROXY_CACHE_UPSTREAM` | off | Upstream Jev-compatible base URL. Setting it enables proxy mode. |
| `--proxy-cache-models <list>` | `OPENKIND_PROXY_CACHE_MODELS` | `jev-latest` | Model aliases the proxy intercepts. |
| `--proxy-cache-encoder <name>` | `OPENKIND_PROXY_CACHE_ENCODER` | `hash` | Embedder: `hash`, or an `openkind pull` name such as `encoder-embedding:8d9498269ef05d95d93c`. |
| `--proxy-cache-encoder-backend` | `OPENKIND_PROXY_CACHE_ENCODER_BACKEND` | `cpu` | `cpu` (candle FP32) or `mlx-fp32` (macOS arm64, `mlx` feature). |
| `--proxy-cache-data-dir <dir>` | `OPENKIND_PROXY_CACHE_DATA_DIR` | platform data dir | Task stores, student versions, key salt. |
| `--proxy-cache-upstream-key <key>` | `OPENKIND_PROXY_CACHE_UPSTREAM_KEY` | caller's key | Bearer key used for upstream calls instead of the caller's. |
| `--proxy-cache-upstream-timeout-ms` | `OPENKIND_PROXY_CACHE_UPSTREAM_TIMEOUT_MS` | `9000` | Per-attempt upstream timeout. |
| `--proxy-cache-target-agreement` | `OPENKIND_PROXY_CACHE_TARGET_AGREEMENT` | `0.98` | Per-task budget: at most `1 - agreement` probability mass of answered-and-disagreed. |
| `--proxy-cache-store-text` | `OPENKIND_PROXY_CACHE_STORE_TEXT` | `true` | Store request text in training rows (`false` keeps salted hashes + embeddings only). |
| `--proxy-cache-admission-min` | `OPENKIND_PROXY_CACHE_ADMISSION_MIN` | `50` | Requests a new task must observe before an engine is created. |
| `--proxy-cache-min-train-samples` | `OPENKIND_PROXY_CACHE_MIN_TRAIN_SAMPLES` | `1000` | Teacher-labelled train rows required for the first fit. |
| `--proxy-cache-min-calib-samples` | `OPENKIND_PROXY_CACHE_MIN_CALIB_SAMPLES` | `500` | Teacher-labelled calibration rows required for the first fit. |
| `--proxy-cache-shadow-min-samples` | `OPENKIND_PROXY_CACHE_SHADOW_MIN_SAMPLES` | `1000` | Shadow observations required before a candidate is judged. |
| `--proxy-cache-calib-fraction` | `OPENKIND_PROXY_CACHE_CALIB_FRACTION` | `0.2` | Fraction of IID teacher-answered requests reserved for calibration. |
| `--proxy-cache-min-new-samples` | `OPENKIND_PROXY_CACHE_MIN_NEW_SAMPLES` | `2000` | New teacher answers that trigger a retrain. |

## Encoders

`hash` (the default) is a keyed hashed bag of words + bigrams. It needs no
weights, no download, and makes the whole system runnable in minutes;
quality is correspondingly crude. For production-grade embeddings install
the pinned sentence encoder:

```bash
openkind pull encoder-embedding:8d9498269ef05d95d93c
openkindd --proxy-cache-upstream https://api.typesafe.ai \
  --proxy-cache-encoder encoder-embedding:8d9498269ef05d95d93c
```

If the named model is not installed, startup fails closed with the download
instruction (the daemon never downloads on its own):

```
Error: proxy cache encoder `...` is not installed (...). Download the model
first with `openkind pull ...` (regular or MLX profile), then restart
openkindd.
```

The same profile serves both backends: the artifacts are digest-verified
once, and `--proxy-cache-encoder-backend mlx-fp32` selects the MLX path on
macOS arm64 (build the daemon with `--features mlx`). See
[`families/encoder-embedding.md`](families/encoder-embedding.md) for the
pinned profile.

## On-disk layout

```
<data-dir>/
  key-salt                      # 32 random bytes, created on first start
  tasks/<20-hex task key>/
    task.json                   # task identity and settings
    samples.sqlite              # training rows and lifecycle events (WAL)
    versions/
      registry.json             # production pointer + version states
      student-v1/               # head.safetensors, ood.safetensors,
                                # policy.json, meta.json
```

Task identity hashes the canonical (instructions, criteria) JSON with the
tenant and requested model. Any wording change is a new task that trains
from scratch; class or JSON key order is irrelevant.

## Guarantees and limits

- **Bounded disagreement.** With probability at least the configured
  confidence (0.95), the rate of (answered and disagreed) stays within the
  task budget, as long as traffic stays exchangeable with the calibration
  window. The math follows the fixed-sequence + Clopper–Pearson design of
  the reference proxy-cache studies; the audit channel is scored as served
  and forces fallback when the bound breaks.
- **Wire transparency.** Requests openkindd does not route are forwarded
  byte-for-byte through the client stack; responses relay unchanged except
  for the added `x-openkind-cache*` headers. Errors from the upstream pass
  through (401, 429 with retry-after, 5xx).
- **Not qualified for release.** The proxy cache is a research feature.
  The disagreement budget is a statistical property of the collected
  traffic, not a model-quality certificate. The standard
  loadable ≠ task-qualified ≠ release-promoted ladder applies. Benchmarks
  live in [`BENCHMARKS.md`](BENCHMARKS.md) when measured.

## Tests

- Unit: task identity, canonical text, student fit, OOD gate, calibration
  (including the closed-form zero-disagreement bound), store, version
  registry, and admission (`crates/openkind-backends`).
- Loop: bootstrap → fit → shadow → promote → local serve → OOD forward →
  restart persistence → teacher-change fallback
  (`crates/openkind-backends/tests/proxy_cache_loop.rs`).
- E2E: the real daemon against a fake Jev upstream, exercising caller-key
  passthrough, promotion, `x-openkind-cache` headers, OOD forwarding, and
  unverified-key behavior
  (`crates/openkind-server/tests/proxy_cache_e2e.rs`).
