# Local model benchmark suite, 2026-10-01

## Status

The supplemental runner is being stopped after the in-flight Banking77
evaluation for clef-flash-gguf reaches a terminal result. It must not proceed
to another dataset or profile. This is a deliberately partial campaign, not a
claim that every registered profile is fully benchmarked.

At the time this note was prepared, suite-status.json recorded 210 completed
dataset evaluations, 23 request-path score runs, 27 skipped combinations, and
18 failed evaluations. The Banking77 result is not yet included in those
counts. The runner status and logs are the detailed record.

## Run definition

- Host: Apple M4 Max, 14 cores, 36 GiB RAM, macOS arm64.
- The catalog snapshot contains 25 profiles. The main pass uses commit
  e3cb136f431e817ac4aca475ea381b0840ef2cd0; supplemental records use
  4f49048714c056cf97757de1f5fb2ea126fc87f6. The working tree was dirty, so
  these supplied commit labels do not identify a clean source checkout.
- Each dataset evaluation uses at most 50 rows from the pinned dataset
  registry and the profile engine selected by the campaign runner. Outputs
  include a materialized workload, predictions, and a summary with provenance.
- Request-path score runs use the 12-row smoke workload except for the
  Qwen3 0.6B CPU control, Plumb-4B MLX, and router-script, which have 777-row
  records.
- These measurements are local evidence for the checked-in engine profiles.
  They do not establish parity, release readiness, or quality promotion by
  themselves.

## Remaining work

The current Banking77 evaluation is the only run in progress. After it reaches
a terminal result, stop the supplemental runner before it starts anything
else. Twenty-one primitive-compatible dataset evaluations and two 12-row
score runs remain after this evaluation.

| Profile | Dataset evaluations still to attempt | Request-path score still to run |
|---|---|---|
| clef-flash-gguf | CLINC150, ARC, HellaSwag, WinoGrande, CommonsenseQA, PAWS, BoolQ, STS-B, SST-5, after Banking77 | 12-row smoke |
| clef-27b-gguf | SST-2, AG News, Banking77, CLINC150, ARC, HellaSwag, WinoGrande, CommonsenseQA, PAWS, BoolQ, STS-B, SST-5 | 12-row smoke |

The supplemental runner had completed SST-2 and AG News for clef-flash-gguf
before the Banking77 run. Dataset rows that exceed a profile's declared
candidate, token, or fixed-label limits are recorded as capability failures or
skips, not as quality results. The current status file contains 18 failures;
the logs show three Laya CLINC150 requests with 152 options against a 100
option limit, seven requests over frozen input-length limits, and eight
Qwen3Guard requests whose dataset labels do not match its fixed
safe/unsafe/controversial schema. These combinations need an input/profile
change before they can produce comparable dataset results.

Other scope boundaries:

- The Qwen3 0.6B control already has a 777-row CPU record. The 4B and 17B
  Qwen3 controls remain at the documented 12-row smoke scale. Plumb-4B has a
  777-row MLX score record; Decider-4B has only its 12-row CPU smoke record.
  A Decider-4B MLX adapter and its 777-row run remain separate implementation
  and benchmark work.
- The encoder-embedding profile has separate CPU and MLX component reports,
  not labeled dataset-evaluation results. Those reports measure 30 calls after
  five warmup calls: CPU p50 22.108 ms at 47.09 embeddings/s, MLX p50 3.361 ms
  at 289.98 embeddings/s.
- Winnow's score run measures routing cost with mock siblings. It is not model
  quality evidence, and no dataset-quality result is claimed for it.

## Completed evidence

Detailed results are indexed in [suite-status.json](suite-status.json).
Per-profile logs are under logs/, dataset outputs are under quality/, and
score reports are under performance/. The completed 777-row score profiles
are decoder-logit-qwen3-06b on CPU, plumb-4b on MLX, and router-script. Most
other profile scores are 12-row smoke records.

The supplemental clef-flash-gguf run produced two labeled dataset results
before Banking77: SST-2 accuracy 0.96 and macro-F1 0.958; AG News accuracy
0.90 and macro-F1 0.886, 50 rows each. Both use model revision
d7f376ea88c05e7bb1014dd5351a93df9dd8029e. See the [SST-2
report](quality/clef-flash-gguf/sst2/dataset-eval-sst2-eval.json) and [AG News
report](quality/clef-flash-gguf/ag_news/dataset-eval-ag_news-eval.json).
These are descriptive 50-row dataset results, not a release or parity gate.
Separate Clef joint-fixture parity records for the three profiles are in the
[Clef family campaign](../2026-10-02-clef/README.md).

The supplemental models that already existed in the local cache were reused
after manifest size and SHA-256 checks. The campaign did not remove those
pre-existing caches. Campaign-owned temporary installs are removed by the
runner after their profile completes.

## Resume

After reviewing the recorded terminal status, run:

```bash
python3 benchmarks/2026-10-01-local-model-suite/run_supplemental_profiles.py
```

The runner verifies existing reports before skipping completed evaluations.
