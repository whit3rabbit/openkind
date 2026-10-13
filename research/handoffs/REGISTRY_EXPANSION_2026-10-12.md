# Passdown: extend the model registry with the Decision Index 0.3.1 leaders

Written 2026-10-12 for the next agent. Read this file, then the sources in
section 1, then do the work in section 9. Nothing here was downloaded or run:
every fact is Hub API metadata, a model card, a shipped config, or the
leaderboard JSON bundle, fetched on 2026-10-12. Re-resolve every revision before
pinning (section 10 has the script).

## 0. Goal and non-goals

Goal: record immutable, verifiable pins (repository, full commit SHA, file sizes,
SHA-256, license) for the open-weight models that top the
[Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
0.3.1, plus the best small ones and their base models, so later family work
starts from fixed sources. Record every MLX conversion found and what each one
does and does not establish.

Non-goals: no loader, no catalog profile, no `installable: true`, no weight
downloads in tests or CI, no quality claims. The existing file
`registry/v1/jev-decision-index.json` states `installable: false` and the test
`crates/openkind-model-store/tests/jev_decision_index.rs` enforces that, so
keep it true for anything you add there.

## 1. Read first

1. [`AGENTS.md`](../../AGENTS.md) (invariants 6, 10, 12, 13) and [`docs/AGENTS.md`](../../docs/AGENTS.md) ("Source Identity Before Access Status", "Evidence & Parity Standards").
2. [`docs/MODEL_REGISTRY.md`](../../docs/MODEL_REGISTRY.md): assets belong in the public mirror before their commit, sizes, and digests are pinned here.
3. [`docs/families/NEW_FAMILY.md`](../../docs/families/NEW_FAMILY.md): "surveyed", "Rust-loadable", "task-qualified", and "release-promoted" stay separate.
4. [`registry/v1/jev-decision-index.json`](../../registry/v1/jev-decision-index.json) (schema `openkind-jev-decision-index/v1`, 0.2.1 snapshot, ten entries) and its test. Decide whether to bump the schema for a 0.3.1 snapshot or add a sibling file; the test pins structure, so read it before editing.
5. [`registry/v1/mlx-alternatives.json`](../../registry/v1/mlx-alternatives.json) (covers catalog profiles only).
6. [`docs/RESEARCH.md`](../../docs/RESEARCH.md), section "Why the Decision Index 0.3.1 Leaders Score Well (Surveyed 2026-10-12)", for the mechanisms and loader needs per entry.
7. [`docs/MODELS.md`](../../docs/MODELS.md) must be updated if `registry/v1` changes or a loadable profile is added.

## 2. Source snapshot to pin

| Item | Value |
|---|---|
| Leaderboard Space | https://huggingface.co/spaces/multimodalart/jev-decision-index |
| Space revision at survey | `f6571b3212a95c770d6c101ec5bb2157218d8cb1` (lastModified 2026-10-12T07:04:13Z) |
| Bundle `data/v03.json` generated | 2026-10-12T04:03:19Z, edition "Decision Index 0.3.1", SHA-256 `482604710eaaeaac424f798c898eac6ccaea161defbd049df629a79215017bb8` |
| Bundle `data/index.json` | SHA-256 `c03e52d0f7d789e24039f4f0b8b53c4195926d965eae07c890ebf629103c7b4d` |
| Bundle `data/methodology.json` | SHA-256 `2e81c8327ceef87f1df9537873f4d1c8dfb76579c8d0de6090efb179453eaa53` |
| Ranking field | `v` in `v03.json` (Full = 0.20 public + 0.50 private same-skills + 0.30 private new-domains); tie band 0.9 |

The hashes were computed over files fetched from `raw/main` while the Space was
at or after `f6571b3`. The Space changes several times a day. Fetch the bundle
at one fixed Space revision (`resolve/<sha>/data/v03.json`), hash it, and record
both. If the hash differs from the one above, the board moved; say so in the file.

## 3. Tier 1: include first

Open weights, permissive license, top of the Full score or the best practical
size. Board columns are rank / Full / public / new-domain from `v03.json`. The
d3 sibling rows have no board entry in the bundle; their scores are the
author-run public numbers on the cards.

| Repository | Board rank / Full / Public / New-domain | License | Params | HEAD at survey | Proposed pin | Pin reason | Notes |
|---|---|---|---|---|---|---|---|
| [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) | 1 / 65.65 / 64.25 / 58.64 | apache-2.0 | 26.09B | `093cffbaa7be6f5dfe14252ae7e80c396c06dbd9` | `cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75` | commit titled "d3 v3.1.0"; the semantic-router pins this release (`cbff3a3c`). Tag `v3.1.0` points elsewhere, see section 5 | Qwen3.8-27B, noncausal, 255-code readout, image and video input |
| [Mapika/decider-31b](https://huggingface.co/Mapika/decider-31b) | 2 / 65.56 / 64.27 / 58.32 | apache-2.0 | 20.87B | `e12ddf20fbc8aa975622414c6b3568804717c8ae` | `cf50c0ba2e3e3516141dcf3bb0434b9f1c116c13` | the commit the card says produced its 64.27 public run | Gemma-4-31B-it, NVFP4 (Blackwell only), second-reading rule in `decider_config.json` |
| [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | 4 / 62.75 / 62.25 / 55.57 | apache-2.0 | 26.09B | `14c9d90137962d300f3de95633f7b6535e80c5ce` | `db124e5056059be396ec5ba69b97a33dd6adb815` | "Release evaluated step-02446 checkpoint". Later commit titles show card, tag, and image-cap code edits; confirm the weight LFS oids are unchanged before choosing HEAD instead | Qwen3.8-27B, noncausal, full training source and data manifest shipped |
| [FINAL-Bench/Darwin-27B-ZTC](https://huggingface.co/FINAL-Bench/Darwin-27B-ZTC) | 3 / 63.21 / 62.63 / 57.66 | apache-2.0 | 25.62B | `2b4bb9e22687b89e3c5703696b575ea2d2774da8` | `2b4bb9e22687b89e3c5703696b575ea2d2774da8` | HEAD at survey time; re-resolve before pinning | Weights upload is `3ccdb11a9f6ad6a1d6adab8bd16f0477a97a1d56`; HEAD adds a `decision_config.json` cleanup and the server files, so pin HEAD after diffing. Ships a copy of the `autojev` package; backbone Darwin-27B-RSI (provenance unclear) |
| [sthanika-ai/Sieve-27B](https://huggingface.co/sthanika-ai/Sieve-27B) | 6 / 61.25 / 60.04 / 54.8 | apache-2.0 | - | `361baad17cb70d2bd5b0ec3dc30f1d01508ce7aa` | `361baad17cb70d2bd5b0ec3dc30f1d01508ce7aa` | HEAD at survey time; re-resolve before pinning | LoRA adapter plus `head.pt` on Qwen3.8-27B, no merged weights |
| [jaredpalmer/kev-27b](https://huggingface.co/jaredpalmer/kev-27b) | 17 / 58.78 / 56.69 / 55.49 | apache-2.0 | 25.62B | `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf` | `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf` | HEAD at survey time; re-resolve before pinning | Full fine-tune with `head.pt`; kev is already a catalog family (`kev`) at small sizes |
| [shisa-ai/shisa-de-2](https://huggingface.co/shisa-ai/shisa-de-2) | 7 / 60.92 / 57.57 / 57.02 | apache-2.0 | 25.81B | `fd9f487763b4d5b7dc4c719f2704290330a65d62` | `fd9f487763b4d5b7dc4c719f2704290330a65d62` | HEAD at survey time; re-resolve before pinning | Gemma-4-26B-A4B-it MoE LoRA; board row is a "repeat" run using `chat_template_repeat.jinja` |
| [vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash) | not on board | apache-2.0 | 8.39B | `f6075bff3c0f7d5704e353d0fb2ebf94888d1691` | `521525effa79469d899a570726547dd7f00da3af` | commit titled "d3-flash v3.1.0"; semantic-router pin `521525ef` | 9B, text public 57.79 (card, author-run) |
| [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini) | not on board | apache-2.0 | 4.54B | `0e78bbdbe3cf480ed1461c185c9e0c4fa9258011` | `a475367de0d8133a3c1a2a0dc12303d89cfa2ab4` | commit titled "d3-mini v3.1.0"; semantic-router pin `a475367d` | 4B, text public 54.90 (card); best 4B-class lead |
| [vllm-sr/d3-nano](https://huggingface.co/vllm-sr/d3-nano) | not on board | apache-2.0 | 2.21B | `9d9b937f5f117f78c030ea7ba677c0b3d328ec58` | `3aeb1c22a5632aac72353c36b3d0111c40e7dbf4` | commit titled "d3-nano v3.1.0"; semantic-router pin `3aeb1c22` | 2B, text public 42.22 (card) |
| [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite) | not on board | apache-2.0 | 0.85B | `3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61` | `a8d25a91ef75552803f7009d25b7bd435991abd9` | commit titled "d3-lite v3.1.0"; semantic-router pin `a8d25a91` | 0.8B, text public 36.93 (card) |
| [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge) | not on board | apache-2.0 | 0.59B | `20c745653400d191ecacd99c4eafc0c5900dc15e` | `18d61df78684675463af9ae732872478845c5a79` | commit titled "d3-edge v3.1.0"; semantic-router pin `18d61df7` (a second commit `391af2ac` has the same title) | 0.59B, backbone pruned to 12 layers, public 28.82 (card) |

## 4. Tier 2: include after tier 1, or when the family exists

Same columns. Licenses differ: Torchcast, StartLux, and Nimble are CC BY-NC 4.0,
which may rule out catalog distribution. Decide the license policy before adding
them (the existing registry pins only what it can legally mirror).

| Repository | Board rank / Full / Public / New-domain | License | Params | HEAD at survey | Proposed pin | Pin reason | Notes |
|---|---|---|---|---|---|---|---|
| [Maincode/matilda-jev-v1.5](https://huggingface.co/Maincode/matilda-jev-v1.5) | 10 / 60.06 / 63.47 / 48.51 | apache-2.0 | 26.09B | `931662e8f54244d3713fcef8df113346c9e79fc6` | `931662e8f54244d3713fcef8df113346c9e79fc6` | HEAD at survey time; re-resolve before pinning | Full fine-tune, custom `configuration_matilda_jev.py`; public 63.47 but new-domain 48.51 |
| [shgao/rsi-jev-v6.1-vl-27b](https://huggingface.co/shgao/rsi-jev-v6.1-vl-27b) | 11 / 59.97 / 65.64 / 47.4 | apache-2.0 | - | `d0dccc052ee844ab7481ddfacad1b7335ab394e0` | `d0dccc052ee844ab7481ddfacad1b7335ab394e0` | HEAD at survey time; re-resolve before pinning | LoRA on Qwen3.8-27B; highest public score (65.64), weak new-domain (47.4) |
| [shgao/rsi-jev-v6.1-vl-4b](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b) | 48 / 47.48 / 50.98 / 36.91 | apache-2.0 | - | `7f93b1926dc23c62635538f5d40f1dabbddee5b6` | `7f93b1926dc23c62635538f5d40f1dabbddee5b6` | HEAD at survey time; re-resolve before pinning | Full fine-tune on Qwen3.5-4B-Base (same base as our selected profile); has first-party MLX 4/8-bit |
| [torchcast-ai/torchcast-decision-27b](https://huggingface.co/torchcast-ai/torchcast-decision-27b) | 12 / 59.91 / 65.1 / 50.83 | cc-by-nc-4.0 | 27.36B | `08645834899cb8e00d370b77a3f4e0baa3241dd1` | `08645834899cb8e00d370b77a3f4e0baa3241dd1` | HEAD at survey time; re-resolve before pinning | CC BY-NC 4.0; causal, StartLux letter format, shared-prefix serving |
| [torchcast-ai/torchcast-decision-12b](https://huggingface.co/torchcast-ai/torchcast-decision-12b) | not on board | cc-by-nc-4.0 | 11.96B | `49107b5589bf2dc4ec3f5acd54316971dd5978b1` | `49107b5589bf2dc4ec3f5acd54316971dd5978b1` | HEAD at survey time; re-resolve before pinning | CC BY-NC 4.0; Gemma-4-12B-it LoRA; no Decision Index entry |
| [torchcast-ai/torchcast-decision-4b](https://huggingface.co/torchcast-ai/torchcast-decision-4b) | not on board | cc-by-nc-4.0 | 5.17B | `f4a34e7aedbf22917f6c85b8c4e03a3da49c1baf` | `f4a34e7aedbf22917f6c85b8c4e03a3da49c1baf` | HEAD at survey time; re-resolve before pinning | CC BY-NC 4.0; Qwen3.5-4B; `yes/no commit floor` 0.801; no Decision Index entry |
| [startlux-models/StartLux-Decision-v1-27B](https://huggingface.co/startlux-models/StartLux-Decision-v1-27B) | not on board | cc-by-nc-4.0 | 27.78B | `5de3f54a7e0300b986d7cccbd3bf1cd59085dbac` | `5de3f54a7e0300b986d7cccbd3bf1cd59085dbac` | HEAD at survey time; re-resolve before pinning | CC BY-NC 4.0; base of Torchcast 27B; 63.88 on 0.2.1 per its card; not on the 0.3.1 board |
| [startlux-models/StartLux-Decision-v1-9B](https://huggingface.co/startlux-models/StartLux-Decision-v1-9B) | not on board | cc-by-nc-4.0 | 9.65B | `c030630a94f8146c4d08e60f4da1b168e17f2e06` | `c030630a94f8146c4d08e60f4da1b168e17f2e06` | HEAD at survey time; re-resolve before pinning | CC BY-NC 4.0; 58.63 on 0.2.1 per its card |
| [startlux-models/StartLux-Decision-v1-4B](https://huggingface.co/startlux-models/StartLux-Decision-v1-4B) | not on board | cc-by-nc-4.0 | 4.66B | `9c0a32e9809d6a3f612434dd0bce150295dcc274` | `9c0a32e9809d6a3f612434dd0bce150295dcc274` | HEAD at survey time; re-resolve before pinning | CC BY-NC 4.0; 52.75 on 0.2.1 per its card |
| [jsaurabh/qwen-decision-27b-noncausal-lora](https://huggingface.co/jsaurabh/qwen-decision-27b-noncausal-lora) | 13 / 59.78 / 57.7 / 51.58 | apache-2.0 | - | `b2eefb929a883136b3bd56ce70999fdac656ee8a` | `b2eefb929a883136b3bd56ce70999fdac656ee8a` | HEAD at survey time; re-resolve before pinning | LoRA plus head with noncausal attention: a second noncausal data point for the mask ablation |
| [MISA-core/linne-fern-1-0](https://huggingface.co/MISA-core/linne-fern-1-0) | 18 / 58.75 / 58.92 / 50.71 | apache-2.0 | 27.78B | `5b73573802016a45d10b702c647063f28854d339` | `5b73573802016a45d10b702c647063f28854d339` | HEAD at survey time; re-resolve before pinning | Full fine-tune on Qwen3.8-27B, `linne_engine.py` |
| [chinhnc/Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) | 19 / 58.71 / 60.79 / 49.12 | apache-2.0 | 31.27B | `9a0f051112ba4301f6e4c8b87fc088ce00b11aae` | `9a0f051112ba4301f6e4c8b87fc088ce00b11aae` | HEAD at survey time; re-resolve before pinning | Gemma-4-31B-it LoRA merged, `quyet_config.json` |
| [juspay/Xor-26B-A4B](https://huggingface.co/juspay/Xor-26B-A4B) | 20 / 57.88 / 57.62 / 49.06 | apache-2.0 | 25.81B | `8259b45a58286fd58ed3b9dacc462360eba66d5b` | `8259b45a58286fd58ed3b9dacc462360eba66d5b` | HEAD at survey time; re-resolve before pinning | Gemma-4-26B-A4B MoE; ships serving tarball; 103 GB used storage (history) |
| [PixilabAI/Blink-v0.3-26B-A4B-NVFP4](https://huggingface.co/PixilabAI/Blink-v0.3-26B-A4B-NVFP4) | 21 / 57.76 / 58.21 / 50.08 | apache-2.0 | 14.39B | `08caad20c1574fb76341ca809ff38fdb081f7d4a` | `08caad20c1574fb76341ca809ff38fdb081f7d4a` | HEAD at survey time; re-resolve before pinning | MoE NVFP4, Blackwell-only |
| [plantwaterer/garas](https://huggingface.co/plantwaterer/garas) | 15 / 59.19 / 56.59 / 53.05 | apache-2.0 | - | `531f3633c1feef79744e6ccce35ee2265cc94561` | `531f3633c1feef79744e6ccce35ee2265cc94561` | HEAD at survey time; re-resolve before pinning | Gemma-4-31B-it adapter plus `garas_engine.py` |
| [vllm-sr/Decision-2.0-Vega-27B](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) | 26 / 55.88 / 56.97 / 47.91 | apache-2.0 | - | `71d6ae1db4a786d98775d65c68c41be0e39bfe7c` | `71d6ae1db4a786d98775d65c68c41be0e39bfe7c` | HEAD at survey time; re-resolve before pinning | Predecessor of d3 27B; adapter plus head, vendored `decision2` runtime; has a first-party GGUF |
| [vllm-sr/Decision-2.0-Lux-9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B) | 44 / 48.21 / 46.76 / 44.45 | apache-2.0 | - | `1dcd1e7f240eab37ac09a6d0b0f77654b59411f9` | `1dcd1e7f240eab37ac09a6d0b0f77654b59411f9` | HEAD at survey time; re-resolve before pinning | Predecessor 9B |
| [vllm-sr/Decision-2.0-Nox-4B](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B) | 59 / 44.95 / 44.21 / 42.19 | apache-2.0 | - | `ae269ecf34a0b3a89026bc1afd9c995279212fad` | `ae269ecf34a0b3a89026bc1afd9c995279212fad` | HEAD at survey time; re-resolve before pinning | Predecessor 4B on Qwen3.5-4B-Base |
| [sthanika-ai/Sieve-9B-Plus](https://huggingface.co/sthanika-ai/Sieve-9B-Plus) | 33 / 52.06 / 53.35 / 45.38 | apache-2.0 | - | `02d213693cc79d0fc02095cc1b07c959b8b4486d` | `02d213693cc79d0fc02095cc1b07c959b8b4486d` | HEAD at survey time; re-resolve before pinning | LoRA plus head on Qwen3.5-9B |
| [sthanika-ai/Sieve-4B](https://huggingface.co/sthanika-ai/Sieve-4B) | 50 / 47.4 / 50.06 / 40.35 | apache-2.0 | - | `ea7cd1edee29590a82ed9fbfcb644227a8eed5f1` | `ea7cd1edee29590a82ed9fbfcb644227a8eed5f1` | HEAD at survey time; re-resolve before pinning | LoRA plus `head.safetensors` on Qwen3.5-4B; ships a small `sieve4b` Python engine |
| [bespokelabs/Bespoke-Nimble-9B-v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3) | 31 / 54.67 / 57.19 / 49.05 | cc-by-nc-4.0 | - | `8e927b9b4afdbb14479fac10a7364d1a695be208` | `8e927b9b4afdbb14479fac10a7364d1a695be208` | HEAD at survey time; re-resolve before pinning | CC BY-NC 4.0 adapter on Qwen3.5-9B; Nimble also supplies pplx training converters |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) | 32 / 53.08 / 61.71 / 41.88 | apache-2.0 | 27.36B | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` | HEAD at survey time; re-resolve before pinning | clef family already exists in the catalog (`clef-27b-gguf`); re-pin only if the HEAD differs from manifests |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 47 / 47.61 / 56.15 / 42.5 | apache-2.0 | 9.41B | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | HEAD at survey time; re-resolve before pinning | Already catalog-installable; board rank 47 |

## 5. Pinning caveats that will bite

- **d3 tags versus release commits.** The semantic-router parity record pins
  `cbff3a3c` (d3), `521525ef` (flash), `a475367d` (mini), `3aeb1c22` (nano),
  `a8d25a91` (lite), `18d61df7` (edge) as the "v3.1.0" release. The Hub tags
  named `v3.1.0` point at different commits (d3 `78a4c3242336499bf9acf5510d71fac91bf058a5`,
  flash `d0c3fcb65aca55634a0cbdb051c90aec2983a9ce`, mini `b5598a4b003bd6f4d9d5704d6e6de3430c79aded`,
  nano `be647da0ddc798d8c5339d77b409ce750713a6d4`, lite `dfa7ac9c1ae71dd3a3cca0f3d5feabd54594e2b2`,
  edge `3a3653c5c8fd6a1255ebda7560c22cccde80aeee`), and tags `v3.1.1` and `v3.1.2`
  exist. HEAD on d3 moved four times on 2026-10-12 (runtime server edits). Compare
  `MODEL_MANIFEST.json` (`files_sha256`, `model_sha256`) across the candidate
  commits. If the weights and readout hashes match, pin the commit the board
  and semantic-router evaluated and record the others as equivalent. If they
  differ, stop and report.
- **Evaluated commit versus HEAD.** pplx v1.1, Decider 31B, and Darwin all
  received commits after their evaluated release. Choose by weight and config
  blob equality, not by recency.
- **Source identity before access status** (`docs/AGENTS.md`). The pplx v1.1
  card says "authenticated Hugging Face access to this private repository", but
  the Hub API reports `private: false`, `gated: false`, and unauthenticated
  fetches succeed. Record the observed status and date. `surogate/rune-26b-a4b-GGUF`
  in the old file was gated with an open mirror; check whether any entry here
  has a similar mirror instead of assuming.
- **Merged versus adapter.** Sieve, garas, Nimble, Vega, and the noncausal LoRA
  ship adapters plus a head, not merged weights. Pin the base revision (section
  6) and the adapter revision; a loader needs both.
- **Quantized-only releases.** Decider 31B (NVFP4 plus FP8 KV) and Blink
  (NVFP4) have no unquantized checkpoint here. OpenKind has no NVFP4 execution
  path.
- **License on the MLX repos.** Several third-party MLX repos have `license: null`.

## 6. Base models to pin

| Repository | Full SHA | Used by |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | d3, pplx, Sieve-27B, Kev, Matilda, RSI 27B, linne-fern, noncausal LoRA, clef (same revision is already pinned in the 0.2.1 registry file) |
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | `842da3794eaa0b77d5f08bae87a17459d91ff475` | Decider 31B, Quyet, garas (same revision is already pinned in the 0.2.1 registry file) |
| [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | `1001bb4d826a52d1f399e183466143f4da7b741b` | RSI 4B, Decision 2.0 Nox (and OpenKind's selected profile) |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | d3-flash, Sieve 9B Plus, Nimble v3, clef-flash |

Already pinned in the 0.2.1 file and not re-resolved here: Gemma-4-26B-A4B-it
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (Shisa, Xor, Blink) and Gemma-4-12B-it
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (Torchcast 12B). Not resolved at all:
Qwen3.5-2B, Qwen3.5-2B-Base, and Qwen3.5-0.8B (d3-nano, d3-lite, d3-edge,
Sieve 2B Plus). Check each checkpoint's own `config.json` and card for which
exact base it used before pinning.

## 7. Excluded, with reasons

| Entry | Reason |
|---|---|
| Jev (TypeSafe) | Closed. |
| Fastino GLiDE no-thinking (28B), board rank 8 | Board `weights_note`: "Open weights coming soon". No repository. (`fastino/GLiNER2.5-Decide*` are different, smaller models; do not substitute.) |
| AnyJev L0, decisio, deck31b, reflex, Decider chat, simple-jev, Jevfire, JoshuaSP diffgemma | `kind: inference technique`: no new weights. Prompting or decoding over stock checkpoints; outside the single-forward readout contract. Context only. |
| RSI-Jev v6.1 27B effort variants | Same repo as the `auto` row; one entry. |
| Darwin-27B-ZTC-v2 | Exists (`ed78457802e7...`), not on the 0.3.1 board. Re-check the board before adding. |

## 8. MLX conversions found

Method: Hub `filter=base_model:quantized|finetune|adapter:<repo>` plus a name
search, keeping repos tagged `mlx` or with `mlx` in the id, then reading each
candidate's file list. This is a survey of metadata. **None was downloaded,
loaded, or compared against its source**, so none is parity evidence. Per the
registry rule, an MLX conversion of a backbone OpenKind cannot load is not an
executable equivalent.

### 8.1 Fine-tune conversions (carry the decision readout)

| Source | MLX repo | SHA | Notes |
|---|---|---|---|
| [pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | [mlx-community/pplx-decider-v1.1-27b-4bit](https://huggingface.co/mlx-community/pplx-decider-v1.1-27b-4bit) | `66383f28b37637658b02b8e82b1029598b138642` | Apache-2.0, 3 shards (about 15.4 GB), `decider_mlx.py`, `server.py`, `decision_config.json`; custom code. Uploaded 2026-10-07, two days after the evaluated release; check it carries the noncausal mask and the 255-row readout. No `readout.safetensors` listed, so the readout may be folded into the weights; verify. |
| [d3](https://huggingface.co/vllm-sr/d3) | [ahmedandaloes/d3-mlx](https://huggingface.co/ahmedandaloes/d3-mlx) | `cbc374ff4bc8e2ee17cb78df4725231f97dc1e17` | `d3_mlx.py`, `readout.safetensors`, `decision_config.json`; no license field; 2 shards. Unverified third party. |
| [d3-flash](https://huggingface.co/vllm-sr/d3-flash) | [ahmedandaloes/d3-flash-mlx](https://huggingface.co/ahmedandaloes/d3-flash-mlx) | `77adb3bea9abde92d1f32158743a0cbbb0bf7aa3` | same layout |
| d3-flash | [thoddnn/d3-flash-q4](https://huggingface.co/thoddnn/d3-flash-q4) | `5b9c0de2e6640380b50491ee2f9eef8eece100e3` | Apache-2.0, 4-bit, `fenn_d3.json`, single shard |
| [d3-mini](https://huggingface.co/vllm-sr/d3-mini) | [ahmedandaloes/d3-mini-mlx](https://huggingface.co/ahmedandaloes/d3-mini-mlx) | `c645173f5729051d473916283f9e37208b5113bb` | same layout as above |
| d3-mini | [thoddnn/d3-mini-q4](https://huggingface.co/thoddnn/d3-mini-q4) | `bb492fe52df011c04ad328b63d1887e30302651d` | Apache-2.0, 4-bit |
| d3-mini | [Jakevin/d3-mini-ternary-mlx](https://huggingface.co/Jakevin/d3-mini-ternary-mlx) | `0823df087abede3c49596150f52b034332e4f0a3` | Apache-2.0, ternary quantization, `allocation.json`; accuracy impact unknown |
| [d3-nano](https://huggingface.co/vllm-sr/d3-nano) | [ahmedandaloes/d3-nano-mlx](https://huggingface.co/ahmedandaloes/d3-nano-mlx) | `b503613cd2bdbbaef1cfbd09c6e348a5758b255b` | same layout as above |
| d3-nano | [thoddnn/d3-nano-q4](https://huggingface.co/thoddnn/d3-nano-q4) | `2d9493e2fba59fb05676d3967273ef36eedba8fe` | Apache-2.0, 4-bit |
| d3-nano | [Jakevin/d3-nano-ternary-mlx](https://huggingface.co/Jakevin/d3-nano-ternary-mlx) | `cda2549ca1dc6ec10aa2f2e2ee305a9d5671fc61` | ternary |
| d3-nano | [Jakevin/d3-nano-ternary-3bit-mlx](https://huggingface.co/Jakevin/d3-nano-ternary-3bit-mlx) | `863da7fd2e646fad18cb83d2c4eb6935ab7cdfc9` | ternary 3-bit, ships `VERIFY.json` (read it) |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) | [ahmedandaloes/d3-lite-mlx](https://huggingface.co/ahmedandaloes/d3-lite-mlx) | `235b1fb6ed8d20ab911e10801a4f4cec9cadc24a` | same layout |
| [d3-edge](https://huggingface.co/vllm-sr/d3-edge) | [ahmedandaloes/d3-edge-mlx](https://huggingface.co/ahmedandaloes/d3-edge-mlx) | `33e8fdb0adb076d7af3d5d1a0049b60cdc1d7939` | same layout |
| [RSI-Jev v6.1 VL 4B](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b) | [shgao/rsi-jev-v6.1-vl-4b-mlx-8bit](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b-mlx-8bit) | `ec7d5e05f833126b082e9fd9f0c5346a12157b81` | First-party (same author), Apache-2.0, `calibration.json`, `mlx.json`, 5 shards. Same base as OpenKind's selected profile, so the most useful MLX lead for a comparison. |
| RSI-Jev v6.1 VL 4B | [shgao/rsi-jev-v6.1-vl-4b-mlx-4bit](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b-mlx-4bit) | `73a6578d68c19d60031da7c9fd111201be950f64` | 4-bit |
| [Decision-2.0-Lux-9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B) | [moritalous/Decision-2.0-Lux-9B-MLX-4bit](https://huggingface.co/moritalous/Decision-2.0-Lux-9B-MLX-4bit) | `50671d85a390f14a06e3372f5dd9583ca4649310` | Apache-2.0, vendored `decision2` runtime and `decision_head.safetensors` |
| [Decision-2.0-Nox-4B](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B) | [moritalous/Decision-2.0-Nox-4B-MLX-4bit](https://huggingface.co/moritalous/Decision-2.0-Nox-4B-MLX-4bit) | `b316e603d16bc90c32f2fe582715567d056e9dc9` | same |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) | [mlx-community/clef-4bit](https://huggingface.co/mlx-community/clef-4bit) | `b973b6f405d02f8e9b83208c7e82ddf43a742eb6` | `joint_head.safetensors`, `clef_mlx.py`; also `clef-8bit` `1c8887c20827d91cb1b0c6f186e24f5bc5b0e0ce`. Many further clef MLX repos exist (Jundot oQ, nativ-community MXFP, AutomatosX, aufklarer, TrevorJS); the list is in the search output, not pinned here. |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | [mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit) | `b5adbe2c3a95200d4bf04a0727c8ec457157d5e1` | also `clef-flash-8bit` `5a195dc147926b63b1d139691ed4480ca9f92c0e` |

### 8.2 Entries with no MLX conversion found

Decider 31B, Darwin-27B-ZTC, Sieve (all sizes), Shisa DE-2, Matilda, RSI 27B,
Torchcast (27B, 12B, 4B), the noncausal LoRA, Kev 27B, linne-fern, Quyet, Xor,
Blink, garas, Vega, Nimble v3. StartLux ships its own `mlx_model.py` and
`mlx_int8.py` inside the model repo (text only) rather than a separate MLX repo;
the Torchcast 27B repo carries the same two files. Those are author-supplied
runtimes, not converted weights.

### 8.3 Backbone-only MLX (not equivalents)

`mlx-community` converts the bases generously: Qwen3.8-27B (4/8-bit, bf16, MTP
variants), gemma-4-31b-it (4/5/6/8-bit, mxfp4/8, nvfp4, bf16), Qwen3.5-9B, and a
few Qwen3.5-4B-Base conversions (`pa5haw/Qwen3.5-4B-Base-mlx-fp16`). Pins for the
Qwen3.8-27B and gemma 26B/31B grids are already in the 0.2.1 registry file.
Backbones lack the readout, so a decision model cannot be recovered from them.

### 8.4 GGUF (useful for the existing GGUF runner, not MLX)

First-party: StartLux 27B, 9B, 4B (Q8_0, Q4_K_M, BF16 repos under
`startlux-models/`), `vllm-sr/Decision-2.0-Vega-27B-GGUF`,
`vllm-sr/Decision-2.0-Lux-9B-GGUF`, Darwin-27B-RSI-GGUF. Third party:
`mradermacher/*` for Torchcast 27B and 12B, Shisa DE-2, Xor, linne-fern, Quyet,
`kdercksen/Kev-27B-GGUF`, `ggml-org/Bespoke-Nimble-9B-v3-GGUF`. A GGUF of a
noncausal or custom-readout checkpoint may drop the readout or the mask; the
existing `clef-*-gguf` profiles show how that gap was handled.

## 9. Suggested work order

1. Re-resolve every row in sections 3, 4, and 6 (script in section 10). Record `sha`, `lastModified`, `private`, `gated`, license, and the LFS file list with sizes and SHA-256 at the chosen revision.
2. Decide the file shape: a 0.3.1 refresh of `jev-decision-index.json` (schema bump, test update) or a sibling file. Keep `installable: false`.
3. Add tier 1 first, d3 family together (identical runtime files, same `d3-code-readout-v1` format), then pplx (same format family, noncausal), then Decider 31B and Darwin.
4. For each entry copy the fields the 0.2.1 entries use: `base_model` (repository, revision, architecture), `weights` (repository, revision, format, license, gated), `openkind_loadable: false`, `blocker`, `nearest_openkind_path`, `mlx_leads` with a `relationship` of `exact-conversion-unverified` or `base-model-only`.
5. Write blockers from `docs/RESEARCH.md` (new section): noncausal full attention means `repeated_full` only; separate 255-row readout; NVFP4 has no path; MoE has no routing in the gemma4 family; adapters need base plus adapter merge.
6. Update `docs/MODELS.md` and `docs/MODEL_REGISTRY.md` only where `registry/v1` changed. Do not duplicate numbers across docs; link to the registry file.
7. Run the root verification battery from `AGENTS.md`; the model-store tests include `jev_decision_index.rs`.

## 10. Re-resolving pins

```python
# Run with: python3 -I resolve.py <repo> [<repo> ...]
import json, sys, urllib.request
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "openkind-registry"}), timeout=60) as r:
        return json.load(r)
for repo in sys.argv[1:]:
    d = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
    sha = d["sha"]
    files = [(s["rfilename"], s.get("size"), (s.get("lfs") or {}).get("sha256")) for s in d["siblings"]]
    print(repo, sha, d["lastModified"], d.get("private"), d.get("gated"), (d.get("cardData") or {}).get("license"))
    for f in files:
        print("  ", *f)
# Commits: /api/models/<repo>/commits/main   Tags: /api/models/<repo>/refs
# Derivatives: /api/models?filter=base_model:quantized:<repo>&limit=100
```

## 11. Open questions for the owner

- License policy: may CC BY-NC 4.0 entries (Torchcast, StartLux, Nimble) appear in the registry file as research metadata, or only Apache-2.0 and MIT?
- Should the 0.2.1 file stay as history or be replaced by the 0.3.1 snapshot?
- The d3 tag versus release-commit mismatch (section 5): which one is canonical?
- Are third-party MLX repos acceptable as `mlx_leads`, or only first-party and `mlx-community`?
