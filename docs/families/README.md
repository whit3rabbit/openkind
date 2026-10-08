# Model families and model registry

This directory has two jobs: it lists the model profiles that the Rust code
can load, and it records architecture families that have only been surveyed.
The supported-model table is a static catalogue. Runtime model aliases are
registered separately by `openkind-engine`.

To add a model or family, use the [contributor guide](./NEW_FAMILY.md).
Most pinned profiles are also catalog-installable through `openkind pull`;
[`../MODELS.md`](../MODELS.md) owns the operator-facing pull names.

## Runnable model profiles

| Family and profile | Base model | Profile bundle | Rust loader |
|---|---|---|---|
| [`encoder-state-first`](./encoder-state-first.md), `a047d6802c3f06f085b8` | `Qwen/Qwen3.5-4B-Base` at `1001bb4d826a52d1f399e183466143f4da7b741b` | `cowWhySo/OpenKind-Qwen3.5-4B-StateFirst` at `20974648aa087369645494e898351253248627a0` | [`Qwen35DecisionEngine::load`](../../crates/openkind-backends/src/qwen35/engine/mod.rs) |
| [`decoder-logit-letter`](./decoder-logit-letter.md), `5492c97dfcdaf3fe9439` | `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775` | checkpoint direct (digest-verified in place) | [`DecoderLetterEngine::load`](../../crates/openkind-backends/src/families/decoder_logit_letter/mod.rs) |
| [`encoder-nli`](./encoder-nli.md), `1041a4c362338a61b820` | `typeform/distilbert-base-uncased-mnli` at `cfa538a0fddbbd978fefe8966c1aeff7ad409c90` | checkpoint direct (digest-verified in place) | [`EncoderNliEngine::load`](../../crates/openkind-backends/src/families/encoder_nli/mod.rs) |
| [`encoder-instruct-label`](./encoder-instruct-label.md), `9fd68313a5606eca42f2` | `knowledgator/gliclass-modern-base-v3.0` at `ac369222ca4375ca66ebaf7fb5220f223514c035` | checkpoint direct (digest-verified in place) | [`EncoderInstructLabelEngine::load`](../../crates/openkind-backends/src/families/encoder_instruct_label/mod.rs) |
| [`decoder-logit-llm`](./decoder-logit-llm.md), `465963d705b6f35d6208` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (q8_0) at `9217f5db79a29953eb74d5343926648285ec7e67` | checkpoint direct (digest-verified in place) | [`DecoderLlmEngine::load`](../../crates/openkind-backends/src/families/decoder_logit_llm/mod.rs) |
| [`schema-scorer`](./schema-scorer.md), `5a7350af556f0ee66566` | `cross-encoder/ms-marco-MiniLM-L-6-v2` at `233902d25c440f23af6f7d6e94d2946bac0bee0a` | checkpoint direct (digest-verified in place) | [`SchemaScorerEngine::load`](../../crates/openkind-backends/src/families/schema_scorer/mod.rs) |
| [`qwen3guard`](./qwen3guard.md) (Stream), `0fcf416cab16d94f933d` | `Qwen/Qwen3Guard-Stream-0.6B` at `419364a715de9840d47b1457982f64ff37f90ed4` | checkpoint direct (digest-verified in place) | [`Qwen3GuardEngine::load`](../../crates/openkind-backends/src/families/qwen3guard/mod.rs) |
| [`kev`](./kev.md), `39d88c11faeb4ac165fa` | `jaredpalmer/kev-0.6b` at `dece6dba8d43f0f7ded45e9f5b9df12474d90843` over `Qwen/Qwen3-0.6B-Base` at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | adapter checkpoint direct; base checkpoint direct (digest-verified in place) | [`KevEngine::load`](../../crates/openkind-backends/src/families/kev/mod.rs) |
| [`decoder-logit-qwen35`](./decoder-logit-qwen35.md), `415bcf4a064e6dadcf85` | `alibiserikbay/JevK5` at `c4f7fdb3aeab5582336406e78d3bef11bf98833d` (merged Qwen3.5-4B weights) | checkpoint direct (digest-verified in place) | [`DecoderLogitQwen35Engine::load`](../../crates/openkind-backends/src/families/decoder_logit_qwen35/mod.rs) |
| [`decoder-logit-qwen35`](./decoder-logit-qwen35.md) `plumb-4b` `c1f080794d38e94a0bc2` | `crh225/plumb-4b` at `24f7bf77e7ee258a2d158c61ea2dce2b60321010` (merged Qwen3.5-4B weights, JevK5 v0.2 fine-tune) | checkpoint direct (digest-verified in place) | [`DecoderLogitQwen35Engine::load`](../../crates/openkind-backends/src/families/decoder_logit_qwen35/mod.rs) |
| [`decider`](./decider.md) `decider-4b` `0529bf6f2bed84641701` | `Mapika/decider-4b` at `eb5fbdfc9448473ec25e399882912863afbdb70e` (merged Qwen3.5-4B weights) | checkpoint direct (digest-verified in place) | [`DeciderEngine::load`](../../crates/openkind-backends/src/families/decider/mod.rs) |
| [`strands-decider`](./strands-decider.md) `strands-decider-2b` `6a02bb0d1c6b25cae74b` | `StrandsAgents/strands-decider-2B-hobson-v19` at `bb282d786bc251fd4e3068de3ada9ddbb38127cd` over `Qwen/Qwen3.5-2B-Base` at `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | adapter checkpoint direct; base checkpoint direct (digest-verified in place) | [`StrandsDeciderEngine::load`](../../crates/openkind-backends/src/families/strands_decider/mod.rs) |
| [`strands-decider`](./strands-decider.md) `strands-decider-2b` `f7156bf28400a79ea1b8` | `StrandsAgents/strands-decider-2B-hobson-v21` at `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5` over `Qwen/Qwen3.5-2B-Base` at `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | adapter checkpoint direct; base checkpoint direct (digest-verified in place) | [`StrandsDeciderEngine::load`](../../crates/openkind-backends/src/families/strands_decider/mod.rs) |
| [`decoder-logit-qwen3`](./decoder-logit-qwen3.md) `decoder-logit-qwen3-06b` `d900f4af57509fe02e62` | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` (raw control) | checkpoint direct (digest-verified in place) | [`DecoderLogitQwen3Engine::load`](../../crates/openkind-backends/src/families/decoder_logit_qwen3/mod.rs) |
| [`decoder-logit-qwen3`](./decoder-logit-qwen3.md) `decoder-logit-qwen3-17b` `8119b9271f8d011e7d03` | `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (raw control) | checkpoint direct (digest-verified in place) | [`DecoderLogitQwen3Engine::load`](../../crates/openkind-backends/src/families/decoder_logit_qwen3/mod.rs) |
| [`decoder-logit-qwen3`](./decoder-logit-qwen3.md) `decoder-logit-qwen3-4b` `9dfaf11792a8d061b6b8` | `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554` (raw control) | checkpoint direct (digest-verified in place) | [`DecoderLogitQwen3Engine::load`](../../crates/openkind-backends/src/families/decoder_logit_qwen3/mod.rs) |
| [`laya`](./laya.md) `laya-english` `c8ea29bf1e33a343c4b7` | `convaiinnovations/laya` at `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | checkpoint direct (digest-verified in place) | [`LayaEngine::load`](../../crates/openkind-backends/src/families/laya/mod.rs) |
| [`laya`](./laya.md) `laya-multilingual` `f4064eb56fb7f7d325e1` | `convaiinnovations/laya-multilingual` at `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` | checkpoint direct (digest-verified in place) | [`LayaEngine::load`](../../crates/openkind-backends/src/families/laya/mod.rs) |
| [`laya`](./laya.md) `laya-typed-decisions` `9d28cfa9567902801ed1` | `convaiinnovations/laya-typed-decisions` at `1a793eb568e6718f15941d08f85432581df534e3` | checkpoint direct (digest-verified in place) | [`LayaEngine::load`](../../crates/openkind-backends/src/families/laya/mod.rs) |
| [`von`](./von.md), `69219703407bd39cca0c` | `wfzyx/von` at `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd` (author's `option_marker.pt` pickle) | checkpoint direct (digest-verified in place) | [`VonEngine::load`](../../crates/openkind-backends/src/families/von/mod.rs) |
| [`router-script`](./router-script.md) | none — Unicode script detector over registered siblings | no artifacts (rule table) | [`RouterScriptEngine::new`](../../crates/openkind-backends/src/families/router_script/mod.rs) |
| [`winnow`](./winnow.md), `4dff8c5b03cfbf680db6` | `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775` + in-house LoRA (vendored) | adapter vendored, base checkpoint direct | [`WinnowEngine::load`](../../crates/openkind-backends/src/families/winnow/mod.rs) |
| [`encoder-embedding`](./encoder-embedding.md), `8d9498269ef05d95d93c` | `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | checkpoint direct (digest-verified in place) | [`BertEmbedder::load`](../../crates/openkind-backends/src/proxy_cache/bert_encoder.rs); MLX: [`MlxBertEmbedder::load`](../../crates/openkind-backends/src/proxy_cache/mlx_bert_encoder.rs) |
| [`clef`](./clef.md) `clef-flash` `dfe12a21a5c9dd5b2fb1` | `Cloudflare/clef-flash` at `17f0b0ad64efb65d273590632833508766b2aae6` | checkpoint direct (digest-verified in place) | [`ClefEngine::load`](../../crates/openkind-backends/src/families/clef/mod.rs) |
| [`clef`](./clef.md) `clef-flash-gguf` `c330d9ee7e9cc658ad45` | `bartowski/Cloudflare_clef-flash-GGUF` (Q4_K_M) at `d7f376ea88c05e7bb1014dd5351a93df9dd8029e` + official `joint_head.safetensors` | checkpoint direct (digest-verified in place) | [`ClefEngine::load`](../../crates/openkind-backends/src/families/clef/mod.rs) |
| [`clef`](./clef.md) `clef-27b-gguf` `48cb5634b4a258de5a6b` | `bartowski/Cloudflare_clef-GGUF` (Q4_K_M) at `e306f00c6c85da175dfb8de952ebb872087426a7` + official `joint_head.safetensors` | checkpoint direct (digest-verified in place) | [`ClefEngine::load`](../../crates/openkind-backends/src/families/clef/mod.rs) |

The loader constants and bundle validation are in
[`qwen35/mod.rs`](../../crates/openkind-backends/src/qwen35/mod.rs) and
[`qwen35/profile.rs`](../../crates/openkind-backends/src/qwen35/profile.rs).
The CPU backend is the default. The optional MLX backend and its parity limits
are documented in [`../MLX.md`](../MLX.md). A loader being present does not
mean the model has release promotion or reviewed model-quality evidence.

## Load the profile from Rust

Loading is offline. Prepare three local paths before calling Rust:

- `bundle_root`: the profile bundle at the pinned revision above. It must
  contain `BUNDLE_MANIFEST.json` and the manifest-covered profile and head
  files.
- `checkpoint_root`: the pinned base checkpoint directory with both
  safetensors shards.
- `tokenizer_path`: the pinned checkpoint's `tokenizer.json`.

The loader checks the pinned bundle, model checkpoint, and tokenizer. It does
not download missing files. For model acquisition and the checkpoint revision,
see the [backend acquisition instructions](../../crates/openkind-backends/AGENTS.md#model-acquisition-explicit-opt-in).

The selected profile bundle is also available at the pinned Hugging Face
revision. Download both repositories as an explicit operator step:

```bash
MODEL_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/openkind"

hf download Qwen/Qwen3.5-4B-Base \
  --revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --local-dir "$MODEL_ROOT/qwen35-base"

hf download cowWhySo/OpenKind-Qwen3.5-4B-StateFirst \
  --revision 20974648aa087369645494e898351253248627a0 \
  --local-dir "$MODEL_ROOT/qwen35-statefirst"
```

Use the first directory as `checkpoint_root`, its `tokenizer.json` as
`tokenizer_path`, and the second directory as `bundle_root`. The commands do
not belong in builds or tests.

This helper loads the CPU engine and registers it under an application-chosen
alias. Supply a continuation-tensor limit chosen for the serving host.

```rust
use std::{path::PathBuf, sync::Arc, time::Duration};

use openkind_backends::qwen35::{
    Qwen35Backend, Qwen35DecisionEngine, Qwen35EngineConfig, Qwen35Error,
    SchedulerConfig,
};
use openkind_engine::EngineRegistry;

fn register_pinned_qwen35(
    registry: &mut EngineRegistry,
    alias: impl Into<String>,
    bundle_root: PathBuf,
    checkpoint_root: PathBuf,
    tokenizer_path: PathBuf,
    max_tensor_storage_bytes: usize,
) -> Result<(), Qwen35Error> {
    let scheduler = SchedulerConfig::for_pinned_profile(
        SchedulerConfig::LOWEST_MEASURED_SHARED_SAVINGS_RATIO,
        Some(max_tensor_storage_bytes),
    );
    let engine = Qwen35DecisionEngine::load(Qwen35EngineConfig {
        bundle_root,
        checkpoint_root,
        tokenizer_path,
        backend: Qwen35Backend::NativeCpu,
        scheduler,
        max_concurrent_requests: 1,
        max_queued_requests: 2,
        retry_after_ms: 1_000,
        evaluation_timeout: Some(Duration::from_secs(600)),
    })?;

    registry.register(alias, Arc::new(engine));
    Ok(())
}

fn register_models(
    bundle_root: PathBuf,
    checkpoint_root: PathBuf,
    tokenizer_path: PathBuf,
    max_tensor_storage_bytes: usize,
) -> Result<EngineRegistry, Qwen35Error> {
    let mut registry = EngineRegistry::new();
    register_pinned_qwen35(
        &mut registry,
        "qwen35-native",
        bundle_root,
        checkpoint_root,
        tokenizer_path,
        max_tensor_storage_bytes,
    )?;
    Ok(registry)
}
```

`max_tensor_storage_bytes` limits retained continuation tensors only. It does
not cover mapped model weights, forward scratch, or allocator overhead. Use a
process-memory envelope as well when the service needs a process-wide memory
ceiling. The scheduler crossover is a measurement for this profile and host;
see [`../BENCHMARKS.md`](../BENCHMARKS.md), and do not treat it as a throughput
guarantee for other hosts.

## Runtime alias registry

[`EngineRegistry`](../../crates/openkind-engine/src/registry.rs) maps a model
alias to an already-loaded `Arc<dyn DecisionEngine>`. Its `models()` and
`list_models()` methods expose registered aliases and metadata, including the
server's `GET /v1/models` response. It is not a lazy model loader or a mapping
from a Hugging Face model ID to a backend.

The daemon uses an alias for the native Qwen engine only when that alias is in
both `--models` and `--qwen35-aliases`; other configured aliases use the mock
engine. The default `--models` list does not include `qwen35-native`, so add it
explicitly when serving the native profile. The generic `BackendType` enum in
`openkind-backends` also does not provide loaders for its Candle, GGUF, or ONNX
variants.

## Surveyed families

Each page records the architecture and, where a pinned profile exists, its
Rust loader and registration status. A "Rust-loadable" entry is a prototype
profile: it loads local artifacts offline and serves through the daemon, and
it carries no model-quality claim.

| Family | Backbone pattern | Rust loading status |
|---|---|---|
| [encoder-nli](./encoder-nli.md) | Encoder, one premise-hypothesis pass per candidate, entailment probabilities | Rust-loadable (prototype profile) |
| [encoder-instruct-label](./encoder-instruct-label.md) | Instruction-tuned encoder with pooled label markers | Rust-loadable (prototype profile, hand-implemented ModernBERT backbone) |
| [encoder-multitask-heads](./encoder-multitask-heads.md) | Small encoder with one trained head per named question, plus a separate open-option embedding path | Surveyed only (Indecis reference; no Rust loader or pinned trained profile) |
| [decoder-logit-letter](./decoder-logit-letter.md) | Decoder with next-token logits restricted to option-letter tokens | Rust-loadable (prototype profile) |
| [decoder-logit-llm](./decoder-logit-llm.md) | Decoder with a label-logit readout | Rust-loadable (prototype profile) |
| [router-script](./router-script.md) | Lightweight script detector that selects a sibling family | Rust-loadable (composite over registered siblings) |
| [winnow](./winnow.md) | Decoder plus LoRA and a script-aware router | Rust-loadable (prototype profile, in-house trained) |
| [kev](./kev.md) | Qwen base with a LoRA adapter and pointer head | Rust-loadable (prototype profile, published open checkpoint) |
| [jeeves](./jeeves.md) | Qwen3.5-9B with a pointer head and optional generated reasoning | Surveyed; pinned external CUDA download/launch helper, no Rust loader or catalog entry |
| [decoder-logit-qwen35](./decoder-logit-qwen35.md) | Qwen3.5 hybrid decoder with a letter next-token-logit readout and knockout combination | Rust-loadable (prototype profile, published open checkpoint) |
| [laya](./laya.md) | ModernBERT-family encoder with a typed-decision marker head and shipped temperature calibration | Rust-loadable (three prototype profiles, reference-parity readout) |
| [von](./von.md) | ModernBERT option-marker encoder scoring all options jointly in one pass | Rust-loadable (prototype profile, published open checkpoint) |
| [schema-scorer](./schema-scorer.md) | Single-logit cross-encoder for the Jev question schema | Rust-loadable (prototype profile, open-weights realization) |
| [qwen3guard](./qwen3guard.md) | Decoder fine-tune for fixed-preset safety verdicts | Rust-loadable (Stream variant, prototype profile) |
| [encoder-embedding](./encoder-embedding.md) | Frozen BERT sentence encoder (CLS pooling, L2-normalized) feeding the proxy-cache distilling student | Rust-loadable (prototype profile; candle CPU + optional MLX) |
| [parallel-constrained-qwen2](./parallel-constrained-qwen2.md) | Shared-prefix Qwen2.5 decoder with batched field suffixes and token-logit readout | Surveyed only (no Rust MLX loader or Jev adapter) |
| [jev-style](./jev-style.md) | Qwen3.5-2B with block-causal attention and a yes/no logit-difference readout | Surveyed only; upstream MLX runtime verified locally, no OpenKind Rust loader or installable catalog entry |
| [jev-gev](./jev-gev.md) | JEV-27B-VL (Qwen3.5 hybrid, vocabulary-logit readout) and GEV-26B-Decide (Gemma 4 MoE, softcapped 24-slot head) on the JEV protocol | Surveyed and pinned; hardware-neutral readout implemented and tested against the mlx-vlm reference; no backbone, `DecisionEngine`, or catalog entry |
| [gemma4-decision](./gemma4-decision.md) | Gemma 4 decoders with trained decision readouts (JevBench ranks 6, 8, 11, 16) | Rust-loadable prototype (`winnow-e4b` letter readout over the in-tree Gemma 4 backbone; 12B and gated checkpoints unpinned) |
| [qwen35-slot-readout](./qwen35-slot-readout.md) | Qwen3.5-4B fine-tunes with trained hidden-state readout heads (JevBench ranks 1, 3, 7, 9, 12, 18, 21) | Surveyed only (backbone reusable from the native path; readouts not implemented) |
| [reranker-logit](./reranker-logit.md) | Dense Qwen3 4B yes/no or letter next-token-logit scorers (JevBench ranks 20, 22, 24) | Surveyed only (0.6B-scale Qwen3 exists in kev; 4B not implemented) |
| [diffusion-decision](./diffusion-decision.md) | Diffusion-decoder decision scorers (JevBench rank 10, djev) | Blocked — ranked checkpoint unpublished; base DiffusionGemma surveyed |
| [clef](./clef.md) | Qwen3.5 hybrid decoder with a joint schema head scoring every option of every question in one pass | Rust-loadable (three prototype profiles: BF16 CPU oracle, Q4_K_M GGUF flash and 27B; MLX 4-bit path in tree, parity open) |

## Ownership and updates

This index owns the catalogue of family-to-loader availability. It does not
duplicate benchmark results, roadmap status, or architecture contracts. Their
canonical owners are:

| Subject | Canonical owner |
|---|---|
| Loadable-model index for operators: types, pull names, backends, sizes, measured runs | [`../MODELS.md`](../MODELS.md) |
| Landed crate boundaries and data flow | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) |
| Benchmark methods and recorded results | [`../BENCHMARKS.md`](../BENCHMARKS.md) |
| MLX runtime contract and limitations | [`../MLX.md`](../MLX.md) |
| Research and prior-art evidence | [`../RESEARCH.md`](../RESEARCH.md) |
| Scientific rationale and measured results | [`../whitepaper/WHITEPAPER.md`](../whitepaper/WHITEPAPER.md) |
| Jev wire compatibility | [`../JEV_COMPATIBILITY.md`](../JEV_COMPATIBILITY.md) |

When adding a family, add its architecture page and a row in the surveyed
table. Move it into the runnable-profile table only when a Rust loader and
profile are implemented. Link to the relevant code and canonical evidence;
do not copy benchmark results or roadmap milestones into these pages.
