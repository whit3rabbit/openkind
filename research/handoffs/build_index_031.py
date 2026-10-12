#!/usr/bin/env python3
"""Build registry/v1/jev-decision-index-0.3.1.json from the 2026-10-12 survey.

Every revision below was re-resolved against the Hub API on 2026-10-12
(research/handoffs/resolve_registry_expansion.py) and every pin-vs-HEAD
difference was checked at file level (pin-vs-head-diff.json): where the
handoff's evaluated release commit differs from HEAD, only README, runtime
server, or processor files moved -- no weight or readout blob changed.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "registry/v1/jev-decision-index-0.3.1.json"

# ---------------------------------------------------------------------------
# Shared architecture descriptions (reused from the 0.2.1 file where the base
# is the same; new sizes described from each checkpoint's shipped config).
# ---------------------------------------------------------------------------

ARCH_QWEN38_27B = (
    "Qwen3_5ForConditionalGeneration hybrid: hidden 5120, 64 layers in a 3:1 linear:full "
    "pattern, DeltaNet 16 key / 48 value heads x 128 with conv kernel 4, GQA 24 query / "
    "4 KV heads x 256, full_attention_interval 4, partial rotary 0.25 theta 1e7, vocab "
    "248320 - the same architecture openkind implements, at exactly the clef-27B dims"
)
ARCH_QWEN35_9B = (
    "Qwen3_5 hybrid at hidden 4096, 32 layers, full_attention_interval 4, vocab 248320 "
    "(the d3-flash checkpoint ships Qwen3_5Model, the backbone class without an LM head)"
)
ARCH_QWEN35_4B = (
    "Qwen3_5 hybrid at hidden 2560, 32 layers, full_attention_interval 4, vocab 248320"
)
ARCH_QWEN35_2B = "Qwen3_5 hybrid at hidden 2048, 24 layers, full_attention_interval 4, vocab 248320"
ARCH_QWEN35_08B = "Qwen3_5 hybrid at hidden 1024, 24 layers, full_attention_interval 4, vocab 248320"
ARCH_GEMMA31B = (
    "Gemma4ForConditionalGeneration dense text stack: hidden 5376, 60 layers in a 5:1 "
    "sliding:full pattern, sliding window 1024, 32 query / 16 KV heads plus 4 global KV "
    "heads, head_dim 256 / global 512, attention_k_eq_v true, no per-layer-input "
    "embeddings and no KV sharing, dual rope, final_logit_softcapping 30.0"
)
ARCH_GEMMA26B = (
    "Gemma4ForConditionalGeneration MoE text stack: hidden 2816, 30 layers in a 5:1 "
    "sliding:full pattern, sliding window 1024, head_dim 256 / global 512, 128 experts "
    "with top 8 active, moe_intermediate 704, dual rope, final_logit_softcapping 30.0"
)
ARCH_GEMMA12B = (
    "Gemma4UnifiedForConditionalGeneration (gemma4_unified) text stack: hidden 3840, 48 "
    "layers in a 5:1 sliding:full pattern, sliding window 1024, 16 query / 8 KV heads "
    "plus 1 global KV head, head_dim 256 / global 512, attention_k_eq_v true - the same "
    "class the gemma4 family loader rejects today"
)
ARCH_DARWIN_BASE = (
    "FINAL-Bench/Darwin-27B-RSI is a 53.79 GB Qwen3.8-27B-class hybrid checkpoint "
    "(bf16 safetensors); its provenance is the author's own RSI run and the Darwin "
    "decision_config records it only as a local-sha256 revision, so the lineage chain "
    "to Qwen/Qwen3.8-27B is disclosed by the author, not independently verified"
)

QWEN38_27B_4BIT_LEAD = {
    "repository": "mlx-community/Qwen3.8-27B-4bit",
    "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    "quantization": "4-bit",
    "base_model": "Qwen/Qwen3.8-27B",
    "relationship": "base-model-only",
    "notes": "Backbone-only conversion already pinned by the 0.2.1 file; carries no decision readout.",
}
GEMMA31B_4BIT_LEAD = {
    "repository": "mlx-community/gemma-4-31b-it-4bit",
    "revision": "696d436c404745a59f30e4939a658162b0a9e57f",
    "quantization": "4-bit",
    "base_model": "google/gemma-4-31B-it",
    "relationship": "base-model-only",
    "notes": "Backbone-only conversion already pinned by the 0.2.1 file; carries no decision readout.",
}
GEMMA26B_4BIT_LEAD = {
    "repository": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "revision": "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
    "quantization": "4-bit MoE",
    "base_model": "google/gemma-4-26B-A4B-it",
    "relationship": "base-model-only",
    "notes": "Backbone-only MoE conversion already pinned by the 0.2.1 file; carries no decision readout.",
}
GEMMA12B_4BIT_LEAD = {
    "repository": "mlx-community/gemma-4-12B-it-4bit",
    "revision": "73bcf09092aa277861d5a191b989b666f7f32e8f",
    "quantization": "4-bit",
    "base_model": "google/gemma-4-12B-it",
    "relationship": "base-model-only",
    "notes": "Backbone-only conversion already pinned by the 0.2.1 file; carries no decision readout.",
}


def lead(repo, rev, quant, base, notes):
    return {
        "repository": repo,
        "revision": rev,
        "quantization": quant,
        "base_model": base,
        "relationship": "exact-conversion-unverified",
        "notes": notes,
    }


def board(rank, full, public, same_skills, new_domains, tied):
    return {
        "rank": rank,
        "full": full,
        "public": public,
        "same_skills": same_skills,
        "new_domains": new_domains,
        "tied": tied,
    }


def entry(engine, leaderboard_name, tier, kind, board_row, base_model, weights,
          blocker, nearest_openkind_path, mlx_leads, off_board_scores=None,
          already_catalog=False):
    data = {
        "engine": engine,
        "leaderboard_name": leaderboard_name,
        "tier": tier,
        "board": board_row,
        "off_board_scores": off_board_scores,
        "already_catalog": already_catalog,
        "kind": kind,
        "base_model": base_model,
        "weights": weights,
        "openkind_loadable": False,
        "blocker": blocker,
        "nearest_openkind_path": nearest_openkind_path,
        "mlx_leads": mlx_leads,
    }
    return data


def weights(repo, rev, fmt, license_, head_at_survey, equivalents=(), notes=None):
    data = {
        "repository": repo,
        "revision": rev,
        "format": fmt,
        "license": license_,
        "gated": False,
        "head_at_survey": head_at_survey,
        "equivalent_revisions": list(equivalents),
    }
    if notes:
        data["notes"] = notes
    return data


def base(repo, rev, architecture, used_by, notes=None):
    data = {
        "repository": repo,
        "revision": rev,
        "architecture": architecture,
        "used_by": used_by,
    }
    if notes:
        data["notes"] = notes
    return data


D3_RUNTIME_NOTE = (
    "Ships d3_format.py, d3_runtime.py, and d3_server.py implementing the "
    "d3-code-readout-v1 contract: tokenizer.encode the rendered prompt, one "
    "noncausal forward, softmax the 255 code logits at the last position at "
    "temperature 1.0, and map codes to answer tokens"
)

# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

TIER1 = [
    entry(
        "vllm-sr-d3",
        "vLLM-SR d3 (27B)",
        1,
        "full fine-tune (merged, vision-capable)",
        board(1, 65.65, 64.25, 63.59, 58.64, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B,
             "declared by decision_config.json base_model/revision"),
        weights("vllm-sr/d3", "cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75",
                "merged BF16 safetensors, 12 shards, 52,176,025,120 bytes, plus a separate "
                "5,222,480-byte readout.safetensors ([255, 5120] float32 rows); decision_config.json "
                "declares format d3-code-readout-v1 version 1, attention_mode "
                "noncausal_full_attention, pooling last, temperature 1.0, and 255 answer codes "
                "mapped to token ids; " + D3_RUNTIME_NOTE,
                "apache-2.0", "093cffbaa7be6f5dfe14252ae7e80c396c06dbd9",
                ["093cffbaa7be6f5dfe14252ae7e80c396c06dbd9"],
                "The pin is the commit the Decision Index board run and the vLLM semantic-router "
                "parity record evaluated (the 'd3 v3.1.0' release commit; the Hub tag v3.1.0 points "
                "elsewhere). HEAD moved four times on 2026-10-12 over runtime server files only; "
                "every weight, readout, and config blob is byte-identical (same sizes and LFS "
                "sha256) between the pin and HEAD, so HEAD is recorded as equivalent. The Hub tag "
                "v3.1.1 and v3.1.2 exist and were not evaluated by the board."),
        "The qwen3_5 hybrid backbone kernels exist at these dims (the clef family), but the "
        "checkpoint needs what no openkind family has yet: the noncausal full-attention mask "
        "in the 16 full-attention layers (repeated_full execution only, no state forking), "
        "the separate 255-row float32 readout, the d3-code-readout-v1 prompt bytes, and the "
        "vision tower for image and video input.",
        "A d3 family adapter (noncausal mask + d3-code-readout-v1 readout) over the existing "
        "qwen3_5 27B-class backbone execution, with the vision path as a later increment; "
        "text-only entries of the same family are the practical first targets.",
        [lead("ahmedandaloes/d3-mlx", "cbc374ff4bc8e2ee17cb78df4725231f97dc1e17", "MLX (bit depth unstated)",
              "vllm-sr/d3",
              "Unverified third-party conversion (2 shards, d3_mlx.py, readout.safetensors, "
              "decision_config.json; no license field). Never downloaded or compared against "
              "the source; not parity evidence."),
         QWEN38_27B_4BIT_LEAD],
    ),
    entry(
        "decider-31b",
        "Decider 31B (NVFP4)",
        1,
        "full fine-tune, merged, NVFP4-quantized release",
        board(2, 65.56, 64.27, 63.65, 58.32, True),
        base("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475",
             ARCH_GEMMA31B, "declared by the model card"),
        weights("Mapika/decider-31b", "cf50c0ba2e3e3516141dcf3bb0434b9f1c116c13",
                "4 NVFP4 safetensors shards, 32,633,477,808 bytes (about 31 GB with FP8 KV cache "
                "as served); no unquantized checkpoint exists in this repository; "
                "decider_config.json carries the two-reading rule (first reading at T1(n) = "
                "2.042 + 0.003 ln n, second reversed-options reading below 0.7 confidence, "
                "re-normalized at T2(n) = 1.953 + 0.010 ln n)",
                "apache-2.0", "e12ddf20fbc8aa975622414c6b3568804717c8ae",
                ["e12ddf20fbc8aa975622414c6b3568804717c8ae"],
                "The pin is the commit the card says produced its 64.27 public run. HEAD "
                "(e12ddf20) differs only in README.md; all four weight shards are byte-identical "
                "between pin and HEAD, so HEAD is recorded as equivalent."),
        "Three stacked blockers: NVFP4 weights and openkind has no NVFP4 execution path; the "
        "gemma4 family loader is fail-closed pinned to E4B geometry and rejects the dense 31B "
        "stack; and the two-reading rule is a per-question re-read with reversed options that "
        "changes the serving contract for Choice questions. The decider letter-slot readout "
        "contract itself is the one openkind already implements at 4B.",
        "Decider-family letter readout over a new gemma4 dense-31B backbone config once an "
        "NVFP4-capable or dequantized execution path exists; the two-reading rule would be a "
        "profile-level policy.",
        [GEMMA31B_4BIT_LEAD],
    ),
    entry(
        "darwin-27b-ztc",
        "Darwin-27B-ZTC",
        1,
        "full fine-tune (merged)",
        board(3, 63.21, 62.63, 60.18, 57.66, True),
        base("FINAL-Bench/Darwin-27B-RSI", "08d44a01bbe4762fbc95171c952f0536bd5d46c6",
             ARCH_DARWIN_BASE,
             "declared by decision_config.json as a local-sha256 revision; the Hub repository "
             "was resolved separately on 2026-10-12"),
        weights("FINAL-Bench/Darwin-27B-ZTC", "2b4bb9e22687b89e3c5703696b575ea2d2774da8",
                "merged BF16 safetensors, 12 shards, 51,251,905,248 bytes, plus a "
                "2,611,288-byte readout.safetensors ([255, 5120] half-precision rows by size); "
                "config declares Qwen3_5TextModel (text-only 5120/64 hybrid without a vision "
                "tower); decision_config.json records training provenance digests (step 675, "
                "temperature 1.0692) and no attention_mode field",
                "apache-2.0", "2b4bb9e22687b89e3c5703696b575ea2d2774da8",
                [],
                "HEAD at survey time and unchanged on re-resolution; the weights upload commit "
                "is 3ccdb11a9f6ad6a1d6adab8bd16f0477a97a1d56 and HEAD adds a decision_config "
                "cleanup plus the server files."),
        "Same noncausal-format questions as d3 minus vision, plus an unresolved base lineage: "
        "the declared base Darwin-27B-RSI is itself an author-published checkpoint whose "
        "training provenance is a local digest, so a loader would pin a two-step chain with "
        "one unverified link. The pplx-shaped 255-row readout and any mask policy need a new "
        "family adapter either way.",
        "The qwen3_5 text-only backbone path (once a text-only variant config is parameterized) "
        "plus the pplx-style 255-row readout adapter; treat the base lineage as unverified "
        "until the Darwin-27B-RSI config is compared against Qwen/Qwen3.8-27B.",
        [],
    ),
    entry(
        "pplx-decider-v1.1-27b",
        "Perplexity Decider v1.1 (27B)",
        1,
        "full fine-tune (merged)",
        board(4, 62.75, 62.25, 61.1, 55.57, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B,
             "declared by decision_config.json base_model/revision"),
        weights("perplexity-ai/pplx-decider-v1.1-27b", "db124e5056059be396ec5ba69b97a33dd6adb815",
                "merged BF16 safetensors, 12 shards, 52,173,413,928 bytes (about 49 GiB), plus a "
                "2,611,288-byte readout.safetensors ([255, 5120] half-precision rows by size); "
                "config declares Qwen3_5Model (backbone without an LM head); decision_config.json "
                "declares format_version 1, attention_mode noncausal_full_attention, pooling "
                "last, step 2446, temperature 1.0087; the release ships the full training source "
                "(source/src/autojev) and data manifest",
                "apache-2.0", "14c9d90137962d300f3de95633f7b6535e80c5ce",
                ["14c9d90137962d300f3de95633f7b6535e80c5ce"],
                "The pin is the 'Release evaluated step-02446 checkpoint' commit. HEAD (14c9d901) "
                "edits README.md, processor_config.json, and source/src/autojev/model.py; the 12 "
                "weight shards and readout.safetensors are byte-identical between pin and HEAD, "
                "so HEAD is recorded as equivalent. The card calls the repository private, but "
                "the Hub API reported private: false and gated: false on 2026-10-12 and "
                "unauthenticated fetches succeeded (source identity before access status)."),
        "The enable_noncausal_full_attention hook lifts the causal mask in the 16 "
        "full-attention layers, which forces the repeated_full execution strategy and gives up "
        "state forking; the separate 255-row half-precision readout, the autojev record-style "
        "prompt, the 8,192-token cap, and the optional image path all need a new family "
        "adapter. The BF16 checkpoint is also about 49 GiB, above this project's 36 GB "
        "reference host.",
        "A pplx family adapter (noncausal mask + 255-row readout) over the existing qwen3_5 "
        "27B-class backbone execution; MLX 4-bit conversion exists but is unverified, so "
        "parity would start from the BF16 checkpoint on larger hardware.",
        [lead("mlx-community/pplx-decider-v1.1-27b-4bit", "66383f28b37637658b02b8e82b1029598b138642",
              "4-bit", "perplexity-ai/pplx-decider-v1.1-27b",
              "Apache-2.0, 3 shards (about 15.4 GB), decider_mlx.py, server.py, "
              "decision_config.json; uploaded two days after the evaluated release. No "
              "readout.safetensors is listed, so the readout may be folded into the weights; "
              "verify it carries the noncausal mask and the 255-row readout before treating it "
              "as an executable equivalent. Never downloaded or compared; not parity evidence.")],
    ),
    entry(
        "sieve-27b",
        "Sieve 27B",
        1,
        "LoRA adapter + separate head",
        board(6, 61.25, 60.04, 59.61, 54.8, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by adapter_config.json base_model_name_or_path"),
        weights("sthanika-ai/Sieve-27B", "361baad17cb70d2bd5b0ec3dc30f1d01508ce7aa",
                "no merged weights: a 933,953,200-byte LoRA adapter (adapter_model.safetensors, "
                "r=64, alpha=64, dropout 0.05, targets include q/k/v/o/gate/up/down plus the "
                "DeltaNet in_proj/out_proj projections) and a 10,490,175-byte torch head.pt; "
                "a loader needs Qwen/Qwen3.8-27B at the pinned base revision plus the adapter "
                "merge plus a torch-head conversion",
                "apache-2.0", "361baad17cb70d2bd5b0ec3dc30f1d01508ce7aa", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "Adapter-plus-torch-head release: openkind would need the 27B base download, an adapter "
        "merge at load time or offline, and a head.pt parser (the openkind contract never "
        "executes model-supplied code, so the torch pickle needs a safe re-expression as "
        "safetensors or an in-tree equivalent). The readout shape and any mask policy are "
        "undocumented outside the pickle.",
        "The decider/pplx pattern: re-express head.pt as a safetensors readout, then an adapter "
        "merge over the existing qwen3_5 27B-class backbone path.",
        [QWEN38_27B_4BIT_LEAD],
    ),
    entry(
        "kev-27b",
        "Kev 27B",
        1,
        "full fine-tune + separate head",
        board(17, 58.78, 56.69, 55.51, 55.49, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by the model card"),
        weights("jaredpalmer/kev-27b", "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
                "merged BF16 safetensors, 11 shards, 51,249,293,960 bytes, plus a "
                "10,494,335-byte torch head.pt",
                "apache-2.0", "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "Structurally the decoder-logit pattern openkind serves, but the kev family is pinned "
        "to small backbones and this is a 51.25 GB 27B checkpoint above the 36 GB reference "
        "host; the head is a torch pickle needing the same safe re-expression as Sieve's.",
        "Extend the kev family with a 27B qwen3_5 backbone config and a converted head; the "
        "kev catalog family (kev:39d88c11faeb4ac165fa) is the existing executable relative.",
        [QWEN38_27B_4BIT_LEAD],
    ),
    entry(
        "shisa-de-2-repeat",
        "Shisa DE-2 (repeat)",
        1,
        "LoRA merged into MoE base",
        board(7, 60.92, 57.57, 58.17, 57.02, True),
        base("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
             ARCH_GEMMA26B, "declared by the model card"),
        weights("shisa-ai/shisa-de-2", "fd9f487763b4d5b7dc4c719f2704290330a65d62",
                "merged BF16 safetensors, 3 shards, 51,612,009,636 bytes; ships both "
                "chat_template.jinja and chat_template_repeat.jinja - the board row is the "
                "'repeat' run using the repeat template",
                "apache-2.0", "fd9f487763b4d5b7dc4c719f2704290330a65d62", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "No MoE path exists in the gemma4 family: its loader is fail-closed pinned to Gemma 4 "
        "E4B geometry and rejects the 26B-A4B stack (128 experts, top 8). The readout lives in "
        "the repeated chat template rather than a separate head, so a loader also needs the "
        "template-exact prompt bytes.",
        "A rune-style new-family effort per docs/families/NEW_FAMILY.md: gemma4 MoE routing "
        "plus the repeat-template readout.",
        [GEMMA26B_4BIT_LEAD],
    ),
    entry(
        "d3-flash",
        "vLLM-SR d3-flash (9B)",
        1,
        "full fine-tune (merged, text)",
        None,
        base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             ARCH_QWEN35_9B, "declared by decision_config.json base_model/revision"),
        weights("vllm-sr/d3-flash", "521525effa79469d899a570726547dd7f00da3af",
                "merged BF16 safetensors, 5 shards, 16,789,656,976 bytes, plus a "
                "4,178,000-byte readout.safetensors ([255, 4096] float32 rows); the same "
                "d3-code-readout-v1 decision_config contract as vllm-sr/d3; "
                + D3_RUNTIME_NOTE,
                "apache-2.0", "f6075bff3c0f7d5704e353d0fb2ebf94888d1691",
                ["f6075bff3c0f7d5704e353d0fb2ebf94888d1691"],
                "The pin is the 'd3-flash v3.1.0' release commit pinned by the semantic-router "
                "parity record; the Hub tag v3.1.0 points elsewhere. HEAD differs only in "
                "runtime server files; every weight and readout blob is byte-identical between "
                "pin and HEAD, so HEAD is recorded as equivalent."),
        "Same noncausal-plus-255-row-readout family adapter need as vllm-sr/d3 at a 16.79 GB "
        "size; the only d3 sibling openkind's reference host could run at BF16.",
        "The d3 family adapter over a 4096/32 qwen3_5 backbone config; the semantic-router "
        "parity record gives a CPU-served reference to diff against.",
        [lead("ahmedandaloes/d3-flash-mlx", "77adb3bea9abde92d1f32158743a0cbbb0bf7aa3",
              "MLX (bit depth unstated)", "vllm-sr/d3-flash",
              "Unverified third-party conversion, same layout as ahmedandaloes/d3-mlx; no "
              "license field. Not parity evidence."),
         lead("thoddnn/d3-flash-q4", "5b9c0de2e6640380b50491ee2f9eef8eece100e3", "4-bit",
              "vllm-sr/d3-flash",
              "Apache-2.0 4-bit conversion (fenn_d3.json, single shard). Not parity evidence."),
         {
             "repository": "mlx-community/Qwen3.5-9B-4bit",
             "revision": "8b2b98c00a6b4d291155e4890773ca8f769aee53",
             "quantization": "4-bit",
             "base_model": "Qwen/Qwen3.5-9B",
             "relationship": "base-model-only",
             "notes": "Backbone-only conversion of the declared base, resolved on 2026-10-12; "
                      "carries no decision readout.",
         }],
        off_board_scores={"source": "author-run public suite 0.3 score from the model card", "public": 57.79, "note": "d3 siblings are not ranked on the 0.3.1 board; cards give author-run public numbers."},
    ),
    entry(
        "d3-mini",
        "vLLM-SR d3-mini (4B)",
        1,
        "full fine-tune (merged, text)",
        None,
        base("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             ARCH_QWEN35_4B, "declared by decision_config.json base_model/revision"),
        weights("vllm-sr/d3-mini", "a475367de0d8133a3c1a2a0dc12303d89cfa2ab4",
                "merged BF16 safetensors, 3 shards, 9,081,227,360 bytes, plus a "
                "2,611,280-byte readout.safetensors ([255, 2560] float32 rows); the same "
                "d3-code-readout-v1 decision_config contract as vllm-sr/d3; "
                + D3_RUNTIME_NOTE,
                "apache-2.0", "0e78bbdbe3cf480ed1461c185c9e0c4fa9258011",
                ["0e78bbdbe3cf480ed1461c185c9e0c4fa9258011"],
                "The pin is the 'd3-mini v3.1.0' release commit pinned by the semantic-router "
                "parity record; the Hub tag v3.1.0 points elsewhere. HEAD differs only in "
                "runtime server files; every weight and readout blob is byte-identical between "
                "pin and HEAD, so HEAD is recorded as equivalent."),
        "The d3 family adapter at the most practical size: the backbone dims (2560/32) match "
        "the decoder-logit-qwen35 family's pinned geometry, but the noncausal mask, the 255-row "
        "float32 readout, and the d3 prompt bytes are all new family surface, and the base is "
        "Qwen3.5-4B (instruct), not the -Base checkpoint openkind's selected profile pins.",
        "The d3 family adapter over a 2560/32 backbone config; best next loader target among "
        "the tier 1 entries (54.90 author-run public at 9.08 GB).",
        [lead("ahmedandaloes/d3-mini-mlx", "c645173f5729051d473916283f9e37208b5113bb",
              "MLX (bit depth unstated)", "vllm-sr/d3-mini",
              "Unverified third-party conversion, same layout as ahmedandaloes/d3-mlx; no "
              "license field. Not parity evidence."),
         lead("thoddnn/d3-mini-q4", "bb492fe52df011c04ad328b63d1887e30302651d", "4-bit",
              "vllm-sr/d3-mini", "Apache-2.0 4-bit conversion. Not parity evidence."),
         lead("Jakevin/d3-mini-ternary-mlx", "0823df087abede3c49596150f52b034332e4f0a3",
              "ternary", "vllm-sr/d3-mini",
              "Apache-2.0 ternary quantization with allocation.json; accuracy impact unknown. "
              "Not parity evidence.")],
        off_board_scores={"source": "author-run public suite 0.3 score from the model card", "public": 54.9, "note": "d3 siblings are not ranked on the 0.3.1 board; cards give author-run public numbers."},
    ),
    entry(
        "d3-nano",
        "vLLM-SR d3-nano (2B)",
        1,
        "full fine-tune (merged, text)",
        None,
        base("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc",
             ARCH_QWEN35_2B, "declared by decision_config.json base_model/revision"),
        weights("vllm-sr/d3-nano", "3aeb1c22a5632aac72353c36b3d0111c40e7dbf4",
                "merged BF16 safetensors, 2 shards, 4,428,644,200 bytes, plus a "
                "2,089,040-byte readout.safetensors ([255, 2048] float32 rows); the same "
                "d3-code-readout-v1 decision_config contract as vllm-sr/d3; "
                + D3_RUNTIME_NOTE,
                "apache-2.0", "9d9b937f5f117f78c030ea7ba677c0b3d328ec58",
                ["9d9b937f5f117f78c030ea7ba677c0b3d328ec58"],
                "The pin is the 'd3-nano v3.1.0' release commit pinned by the semantic-router "
                "parity record; the Hub tag v3.1.0 points elsewhere. HEAD differs only in "
                "runtime server files; every weight and readout blob is byte-identical between "
                "pin and HEAD, so HEAD is recorded as equivalent."),
        "The d3 family adapter at a 4.43 GB size on a backbone openkind has no pinned profile "
        "for (Qwen3.5-2B); quality drops steeply in the small d3 siblings (42.22 author-run "
        "public versus 54.90 for d3-mini).",
        "The d3 family adapter over a 2048/24 backbone config.",
        [lead("ahmedandaloes/d3-nano-mlx", "b503613cd2bdbbaef1cfbd09c6e348a5758b255b",
              "MLX (bit depth unstated)", "vllm-sr/d3-nano",
              "Unverified third-party conversion, same layout as ahmedandaloes/d3-mlx; no "
              "license field. Not parity evidence."),
         lead("thoddnn/d3-nano-q4", "2d9493e2fba59fb05676d3967273ef36eedba8fe", "4-bit",
              "vllm-sr/d3-nano", "Apache-2.0 4-bit conversion. Not parity evidence."),
         lead("Jakevin/d3-nano-ternary-mlx", "cda2549ca1dc6ec10aa2f2e2ee305a9d5671fc61",
              "ternary", "vllm-sr/d3-nano", "Ternary quantization. Not parity evidence."),
         lead("Jakevin/d3-nano-ternary-3bit-mlx", "863da7fd2e646fad18cb83d2c4eb6935ab7cdfc9",
              "ternary 3-bit", "vllm-sr/d3-nano",
              "Ternary 3-bit quantization; ships a VERIFY.json that has not been read here. "
              "Not parity evidence.")],
        off_board_scores={"source": "author-run public suite 0.3 score from the model card", "public": 42.22, "note": "d3 siblings are not ranked on the 0.3.1 board; cards give author-run public numbers."},
    ),
    entry(
        "d3-lite",
        "vLLM-SR d3-lite (0.8B)",
        1,
        "full fine-tune (merged, text)",
        None,
        base("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17",
             ARCH_QWEN35_08B, "declared by decision_config.json base_model/revision"),
        weights("vllm-sr/d3-lite", "a8d25a91ef75552803f7009d25b7bd435991abd9",
                "merged BF16 safetensors, 2 shards, 1,707,072,248 bytes, plus a "
                "1,044,560-byte readout.safetensors ([255, 1024] float32 rows); the same "
                "d3-code-readout-v1 decision_config contract as vllm-sr/d3; "
                + D3_RUNTIME_NOTE,
                "apache-2.0", "3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61",
                ["3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61"],
                "The pin is the 'd3-lite v3.1.0' release commit pinned by the semantic-router "
                "parity record; the Hub tag v3.1.0 points elsewhere. HEAD differs only in "
                "runtime server files; every weight and readout blob is byte-identical between "
                "pin and HEAD, so HEAD is recorded as equivalent."),
        "The d3 family adapter at a 1.71 GB size; 36.93 author-run public makes it a "
        "convenience target for adapter bring-up rather than a quality lead.",
        "The d3 family adapter over a 1024/24 backbone config.",
        [lead("ahmedandaloes/d3-lite-mlx", "235b1fb6ed8d20ab911e10801a4f4cec9cadc24a",
              "MLX (bit depth unstated)", "vllm-sr/d3-lite",
              "Unverified third-party conversion, same layout as ahmedandaloes/d3-mlx; no "
              "license field. Not parity evidence.")],
        off_board_scores={"source": "author-run public suite 0.3 score from the model card", "public": 36.93, "note": "d3 siblings are not ranked on the 0.3.1 board; cards give author-run public numbers."},
    ),
    entry(
        "d3-edge",
        "vLLM-SR d3-edge (0.59B)",
        1,
        "pruned full fine-tune (merged, text)",
        None,
        base("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17",
             ARCH_QWEN35_08B + "; the fine-tune prunes the backbone to 12 of the 24 layers",
             "declared by decision_config.json base_model/revision"),
        weights("vllm-sr/d3-edge", "18d61df78684675463af9ae732872478845c5a79",
                "merged BF16 safetensors, 2 shards, 1,189,722,528 bytes, plus a "
                "1,044,560-byte readout.safetensors ([255, 1024] float32 rows); the same "
                "d3-code-readout-v1 decision_config contract as vllm-sr/d3; the backbone is "
                "pruned to 12 layers, so a loader needs a per-checkpoint pruned layer list; "
                + D3_RUNTIME_NOTE,
                "apache-2.0", "20c745653400d191ecacd99c4eafc0c5900dc15e",
                ["20c745653400d191ecacd99c4eafc0c5900dc15e"],
                "The pin is the 'd3-edge v3.1.0' release commit pinned by the semantic-router "
                "parity record (a second commit 391af2ac has the same title); the Hub tag "
                "v3.1.0 points elsewhere. HEAD differs only in runtime server files; every "
                "weight and readout blob is byte-identical between pin and HEAD, so HEAD is "
                "recorded as equivalent."),
        "The d3 family adapter plus support for a pruned (12-layer) layer list; smallest and "
        "weakest of the family at 28.82 author-run public, but also the cheapest to execute. "
        "This entry is the one tier-1 checkpoint with a loadable OpenKind profile: the d3 "
        "family adapter (noncausal mask, per-layer mixer table, 255-row float32 readout) "
        "landed as d3-edge:664f9ead734372935286 with prompt byte-parity against the shipped "
        "d3_format.py; see docs/families/d3.md.",
        "Landed as the d3-edge catalog profile over the pruned 1024/12 noncausal backbone.",
        [lead("ahmedandaloes/d3-edge-mlx", "33e8fdb0adb076d7af3d5d1a0049b60cdc1d7939",
              "MLX (bit depth unstated)", "vllm-sr/d3-edge",
              "Unverified third-party conversion, same layout as ahmedandaloes/d3-mlx; no "
              "license field. Not parity evidence.")],
        off_board_scores={"source": "author-run public suite 0.3 score from the model card", "public": 28.82, "note": "d3 siblings are not ranked on the 0.3.1 board; cards give author-run public numbers."},
        already_catalog=True,
    ),
]

# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------

TIER2 = [
    entry(
        "matilda-jev-v1.5",
        "Matilda Jev v1.5",
        2,
        "full fine-tune (merged)",
        board(10, 60.06, 63.47, 60.94, 48.51, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B,
             "per the model card and board metadata; the shipped decision_config.json instead "
             "declares base_model Maincode/MATILDA at revision 'unknown', and that repository "
             "returned HTTP 401 on 2026-10-12 (moved, renamed, or private)"),
        weights("Maincode/matilda-jev-v1.5", "931662e8f54244d3713fcef8df113346c9e79fc6",
                "merged BF16 safetensors, 12 shards, 52,173,414,024 bytes, plus a "
                "2,611,288-byte readout.safetensors ([255, 5120] half-precision rows by size) "
                "and a custom configuration_matilda_jev.py; decision_config.json records "
                "temperature 1.5354",
                "apache-2.0", "931662e8f54244d3713fcef8df113346c9e79fc6", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The pplx-shaped 255-row readout over a declared base whose repository is "
        "unresolvable (Maincode/MATILDA, HTTP 401), plus a custom config class; the strong "
        "public score (63.47) with weak new-domain transfer (48.51) repeats the board's "
        "public-overfit pattern.",
        "The pplx-style readout adapter over the qwen3_5 27B-class backbone; pin the true "
        "training base only if Maincode/MATILDA reappears or the config class discloses it.",
        [],
    ),
    entry(
        "rsi-jev-v6.1-vl-27b",
        "RSI-Jev v6.1 VL 27B (effort auto)",
        2,
        "LoRA on the 27B hybrid, self-contained release",
        board(11, 59.97, 65.64, 60.77, 47.4, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by meta.json base_model/base_revision"),
        weights("shgao/rsi-jev-v6.1-vl-27b", "d0dccc052ee844ab7481ddfacad1b7335ab394e0",
                "5 safetensors shards, 54,468,026,840 bytes (self-contained per meta.json, "
                "which also records tapped_layer 64, bf16 tower dtype, and 2,000 training "
                "steps); the highest public score on the board (65.64) with the weakest "
                "new-domain transfer among the leaders (47.4)",
                "apache-2.0", "d0dccc052ee844ab7481ddfacad1b7335ab394e0", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "A self-contained 54.47 GB release whose readout is the tapped-layer scheme described "
        "by meta.json rather than a separate safetensors head; openkind would need to "
        "re-express that readout before any loader work, and the 27B BF16 execution exceeds "
        "the 36 GB reference host.",
        "The decoder-logit pattern at 27B once a tapped-layer readout adapter exists; the "
        "4B sibling below is the practical size.",
        [],
    ),
    entry(
        "rsi-jev-v6.1-vl-4b",
        "RSI-Jev v6.1 VL 4B",
        2,
        "full fine-tune (merged)",
        board(48, 47.48, 50.98, 50.26, 36.91, True),
        base("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b",
             ARCH_QWEN35_4B,
             "declared by the model card; the same base revision openkind's selected profile pins"),
        weights("shgao/rsi-jev-v6.1-vl-4b", "7f93b1926dc23c62635538f5d40f1dabbddee5b6",
                "merged BF16 safetensors, 5 shards, 9,676,732,504 bytes; calibration.json and "
                "mlx.json ship in the repository",
                "apache-2.0", "7f93b1926dc23c62635538f5d40f1dabbddee5b6", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "Same base as openkind's selected profile, but the readout and calibration contract "
        "differ from every openkind family and need their own adapter plus parity fixtures; "
        "no Decision Index loader exists for the tapped-layer scheme at 4B either.",
        "The most useful MLX comparison target against openkind's selected profile (the "
        "first-party MLX conversions below share the base), pending a readout adapter.",
        [lead("shgao/rsi-jev-v6.1-vl-4b-mlx-8bit", "ec7d5e05f833126b082e9fd9f0c5346a12157b81",
              "8-bit", "shgao/rsi-jev-v6.1-vl-4b",
              "First-party (same author) Apache-2.0 conversion, 5 shards, with calibration.json; "
              "never downloaded or compared; not parity evidence."),
         lead("shgao/rsi-jev-v6.1-vl-4b-mlx-4bit", "73a6578d68c19d60031da7c9fd111201be950f64",
              "4-bit", "shgao/rsi-jev-v6.1-vl-4b",
              "First-party 4-bit conversion. Not parity evidence.")],
    ),
    entry(
        "torchcast-decision-27b",
        "Torchcast Decision 27B",
        2,
        "LoRA merged into a StartLux fine-tune",
        board(12, 59.91, 65.1, 58.13, 50.83, True),
        base("startlux-models/StartLux-Decision-v1-27B", "5de3f54a7e0300b986d7cccbd3bf1cd59085dbac",
             ARCH_QWEN38_27B + "; the StartLux base is itself a causal letter-readout decision "
             "model on the same hybrid",
             "declared by the model card"),
        weights("torchcast-ai/torchcast-decision-27b", "08645834899cb8e00d370b77a3f4e0baa3241dd1",
                "merged BF16 safetensors, 12 shards, 54,713,606,808 bytes; decision_config.json "
                "declares format StartLux-Decision-v1 with max_options_per_pass 26; causal "
                "attention with shared-prefix serving; the repository also ships author "
                "MLX runtimes (torchcast_decision/mlx_model.py, mlx_int8.py) that are "
                "code, not converted weights",
                "cc-by-nc-4.0", "08645834899cb8e00d370b77a3f4e0baa3241dd1", [],
                "HEAD at survey time and unchanged on re-resolution. CC BY-NC 4.0 may rule out "
                "catalog distribution; recorded here as research metadata only."),
        "The letter-logit readout reads 26 lm_head rows (grouped rounds above 26 options), "
        "per-type temperatures (noul 1.421, choice 1.2098, score 1.2294), and the jev_render_v3 "
        "prompt; none of that surface exists in an openkind family, and the license is "
        "non-commercial.",
        "The StartLux-Decision-v1 readout adapter over the qwen3_5 27B-class backbone; the "
        "license keeps it out of any distributable profile.",
        [],
    ),
    entry(
        "torchcast-decision-12b",
        "Torchcast Decision 12B",
        2,
        "LoRA (distillation-trained)",
        None,
        base("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
             ARCH_GEMMA12B, "declared by the shipped config (gemma4_unified, hidden 3840, 48 layers)"),
        weights("torchcast-ai/torchcast-decision-12b", "49107b5589bf2dc4ec3f5acd54316971dd5978b1",
                "merged BF16 safetensors, 1 shard, 23,919,549,408 bytes; readout_config.json "
                "ships with the checkpoint; a 0.2 yes/no temperature selected from a calibration "
                "grid",
                "cc-by-nc-4.0", "49107b5589bf2dc4ec3f5acd54316971dd5978b1", [],
                "HEAD at survey time and unchanged on re-resolution. No Decision Index entry; "
                "the card's 203/231 (87.88%) is an author-run JevBench public figure."),
        "The gemma4 family loader rejects the gemma4_unified 12B stack (same class as "
        "Winnow-12B), and the StartLux letter readout plus non-commercial license apply here "
        "too.",
        "A second pinned config in a future gemma4 unified family plus the StartLux readout "
        "adapter; mradermacher publishes third-party GGUFs.",
        [],
        off_board_scores={"source": "author-run JevBench public figure from the model card", "public": 87.88,
            "note": "No Decision Index entry; 203/231 (87.88%) is an author-run JevBench public number that influenced checkpoint selection, on a different scale from Decision Index scores."},
    ),
    entry(
        "torchcast-decision-4b",
        "Torchcast Decision 4B",
        2,
        "fine-tune (merged)",
        None,
        base("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             ARCH_QWEN35_4B, "declared by the shipped config (qwen3_5, hidden 2560, 32 layers); "
             "the same base revision d3-mini declares"),
        weights("torchcast-ai/torchcast-decision-4b", "f4a34e7aedbf22917f6c85b8c4e03a3da49c1baf",
                "merged BF16 safetensors, 3 shards, 10,350,018,536 bytes; decision_config.json "
                "declares format StartLux-Decision-v1; calibration temperature 0.9 and a "
                "yes/no commit floor of 0.801 that overstates reported probability for "
                "uncertain items - a metric-tuned policy layer, not a calibrated probability",
                "cc-by-nc-4.0", "f4a34e7aedbf22917f6c85b8c4e03a3da49c1baf", [],
                "HEAD at survey time and unchanged on re-resolution. No Decision Index entry."),
        "The StartLux letter readout, the commit-floor policy, and the non-commercial license; "
        "the base dims match openkind's pinned geometry, but nothing else in the serving "
        "contract does.",
        "The StartLux readout adapter at 4B if the license policy ever allows it.",
        [],
        off_board_scores={"source": "model card", "public": None,
            "note": "No Decision Index entry and no published leaderboard run; the card documents a yes/no commit floor of 0.801."},
    ),
    entry(
        "startlux-decision-v1-27b",
        "StartLux Decision v1 27B",
        2,
        "fine-tune (merged)",
        None,
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by the model card"),
        weights("startlux-models/StartLux-Decision-v1-27B", "5de3f54a7e0300b986d7cccbd3bf1cd59085dbac",
                "merged BF16 safetensors, 11 shards, 55,563,006,888 bytes; decision_config.json "
                "declares format StartLux-Decision-v1 with max_options_per_pass 26; ships "
                "author MLX runtimes (startlux_decision/mlx_model.py, mlx_int8.py) that are "
                "code, not converted weights; first-party GGUF builds (Q8_0/Q4_K_M/BF16) live "
                "in sibling startlux-models repositories",
                "cc-by-nc-4.0", "5de3f54a7e0300b986d7cccbd3bf1cd59085dbac", [],
                "HEAD at survey time and unchanged on re-resolution. The card reports 63.88 on "
                "Decision Index 0.2.1; the entry is not on the 0.3.1 board (it is the base of "
                "torchcast-decision-27b)."),
        "The StartLux-Decision-v1 letter readout and per-type calibration, plus the "
        "non-commercial license; the causal shared-prefix serving is the part closest to "
        "openkind's nested execution.",
        "The StartLux readout adapter over the qwen3_5 27B-class backbone.",
        [],
        off_board_scores={"source": "model card, Decision Index 0.2.1 author report", "full": 63.88,
            "note": "Not on the 0.3.1 board; the card reports 63.88 on 0.2.1. This entry is the base of torchcast-decision-27b."},
    ),
    entry(
        "startlux-decision-v1-9b",
        "StartLux Decision v1 9B",
        2,
        "fine-tune (merged)",
        None,
        base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             ARCH_QWEN35_9B, "declared by the model card"),
        weights("startlux-models/StartLux-Decision-v1-9B", "c030630a94f8146c4d08e60f4da1b168e17f2e06",
                "merged BF16 safetensors, 4 shards, 19,306,311,104 bytes; the same "
                "StartLux-Decision-v1 contract and author MLX runtimes as the 27B",
                "cc-by-nc-4.0", "c030630a94f8146c4d08e60f4da1b168e17f2e06", [],
                "HEAD at survey time and unchanged on re-resolution. The card reports 58.63 on "
                "Decision Index 0.2.1; not on the 0.3.1 board."),
        "The StartLux letter readout plus the non-commercial license at a 19.31 GB size.",
        "The StartLux readout adapter over a 4096/32 backbone config.",
        [],
        off_board_scores={"source": "model card, Decision Index 0.2.1 author report", "full": 58.63,
            "note": "Not on the 0.3.1 board."},
    ),
    entry(
        "startlux-decision-v1-4b",
        "StartLux Decision v1 4B",
        2,
        "fine-tune (merged)",
        None,
        base("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             ARCH_QWEN35_4B, "declared by the model card; the same base revision d3-mini declares"),
        weights("startlux-models/StartLux-Decision-v1-4B", "9c0a32e9809d6a3f612434dd0bce150295dcc274",
                "merged BF16 safetensors, 2 shards, 9,319,828,240 bytes; the same "
                "StartLux-Decision-v1 contract and author MLX runtimes as the 27B",
                "cc-by-nc-4.0", "9c0a32e9809d6a3f612434dd0bce150295dcc274", [],
                "HEAD at survey time and unchanged on re-resolution. The card reports 52.75 on "
                "Decision Index 0.2.1; not on the 0.3.1 board."),
        "The StartLux letter readout plus the non-commercial license at openkind's pinned "
        "geometry.",
        "The StartLux readout adapter over a 2560/32 backbone config.",
        [],
        off_board_scores={"source": "model card, Decision Index 0.2.1 author report", "full": 52.75,
            "note": "Not on the 0.3.1 board."},
    ),
    entry(
        "qwen-decision-27b-noncausal",
        "Qwen Decision 27B noncausal LoRA",
        2,
        "LoRA adapter + head (noncausal)",
        board(13, 59.78, 57.7, 60.21, 51.58, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by decision_config.json base_model/revision"),
        weights("jsaurabh/qwen-decision-27b-noncausal-lora",
                "b2eefb929a883136b3bd56ce70999fdac656ee8a",
                "no merged weights: an 80,997,376-parameter LoRA (rank 16, alpha 32, adapter/ "
                "directory) plus a 2,611,288-byte readout.safetensors ([255, 5120] "
                "half-precision rows by size) and a processor chat template; decision_config.json "
                "declares architecture autojev-readout+lora, attention_mode "
                "noncausal_full_attention, pooling last, temperature 1.0863, and a full "
                "training manifest (1,533 steps, 48,847 rows, max_length 512)",
                "apache-2.0", "b2eefb929a883136b3bd56ce70999fdac656ee8a", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The pplx-style autojev release in adapter form: noncausal mask (repeated_full only), "
        "the 255-row readout, and an adapter merge over the 27B base; a second noncausal data "
        "point for the mask ablation alongside pplx v1.1.",
        "The pplx family adapter with an adapter-merge front end; the published training "
        "manifest makes it a useful reproduction reference for the mask ablation.",
        [],
    ),
    entry(
        "linne-fern-1-0",
        "linne-fern 1.0 (27B)",
        2,
        "full fine-tune (merged)",
        board(18, 58.75, 58.92, 58.38, 50.71, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by the model card"),
        weights("MISA-core/linne-fern-1-0", "5b73573802016a45d10b702c647063f28854d339",
                "merged BF16 safetensors, 18 shards, 55,563,006,776 bytes; ships "
                "linne_engine.py custom serving code and a preserved legacy chat template "
                "(chat_template.jinja.goc-20260820)",
                "apache-2.0", "5b73573802016a45d10b702c647063f28854d339", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "A 55.56 GB merged checkpoint whose readout lives in custom serving code "
        "(linne_engine.py) rather than a pinned config artifact; openkind would have to "
        "re-express that engine before any loader work.",
        "The decoder-logit pattern at 27B once the linne engine's readout is re-expressed.",
        [],
    ),
    entry(
        "quyet-1-0-large",
        "Quyet-1.0-Large",
        2,
        "LoRA merged into the base",
        board(19, 58.71, 60.79, 58.83, 49.12, True),
        base("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475",
             ARCH_GEMMA31B, "declared by the model card"),
        weights("chinhnc/Quyet-1.0-Large", "9a0f051112ba4301f6e4c8b87fc088ce00b11aae",
                "merged BF16 safetensors, 2 shards, 62,546,338,248 bytes; quyet_config.json "
                "carries the readout contract",
                "apache-2.0", "9a0f051112ba4301f6e4c8b87fc088ce00b11aae", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The gemma4 dense-31B backbone blocker plus a quyet_config.json readout contract that "
        "would need its own adapter.",
        "The decider-family letter readout pattern over a future gemma4 dense-31B config.",
        [GEMMA31B_4BIT_LEAD],
    ),
    entry(
        "xor-26b-a4b",
        "Xor 26B-A4B v1.1",
        2,
        "MoE fine-tune (merged)",
        board(20, 57.88, 57.62, 58.5, 49.06, True),
        base("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
             ARCH_GEMMA26B, "declared by the shipped config (gemma4 MoE, hidden 2816, 30 layers)"),
        weights("juspay/Xor-26B-A4B", "8259b45a58286fd58ed3b9dacc462360eba66d5b",
                "merged BF16 safetensors, 2 shards, 51,612,009,332 bytes; ships a serving "
                "tarball; repository history reports 103 GB used storage",
                "apache-2.0", "8259b45a58286fd58ed3b9dacc462360eba66d5b", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "No MoE path exists in the gemma4 family (fail-closed pinned to E4B geometry); the "
        "readout contract ships inside the serving tarball rather than a pinned config.",
        "A rune-style gemma4 MoE new-family effort per docs/families/NEW_FAMILY.md.",
        [GEMMA26B_4BIT_LEAD],
    ),
    entry(
        "blink-v0-3-26b-a4b-nvfp4",
        "Blink v0.3 26B-A4B (NVFP4)",
        2,
        "MoE fine-tune, NVFP4-quantized release",
        board(21, 57.76, 58.21, 57.21, 50.08, True),
        base("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
             ARCH_GEMMA26B, "declared by the shipped config (gemma4 MoE, hidden 2816, 30 layers)"),
        weights("PixilabAI/Blink-v0.3-26B-A4B-NVFP4", "08caad20c1574fb76341ca809ff38fdb081f7d4a",
                "NVFP4 safetensors, 2 shards, 18,788,485,588 bytes (14.39B served params); no "
                "unquantized checkpoint in the repository",
                "apache-2.0", "08caad20c1574fb76341ca809ff38fdb081f7d4a", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "NVFP4 weights with no openkind execution path, stacked on the missing gemma4 MoE "
        "router.",
        "Blocked until both an NVFP4 path and a gemma4 MoE family exist.",
        [GEMMA26B_4BIT_LEAD],
    ),
    entry(
        "garas",
        "garas (Gemma 4 31B)",
        2,
        "LoRA adapter + custom engine",
        board(15, 59.19, 56.59, 58.3, 53.05, True),
        base("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475",
             ARCH_GEMMA31B, "declared by adapter_config.json base_model_name_or_path"),
        weights("plantwaterer/garas", "531f3633c1feef79744e6ccce35ee2265cc94561",
                "no merged weights: a 489,840,816-byte LoRA adapter (rank 16, alpha 32, "
                "MLP plus attention targets) and garas_engine.py custom serving code",
                "apache-2.0", "531f3633c1feef79744e6ccce35ee2265cc94561", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "Adapter plus a custom engine whose readout is expressed in Python code, not a pinned "
        "config artifact, over the gemma4 dense-31B backbone the family loader rejects.",
        "Re-express the garas engine readout, then the gemma4 dense-31B backbone config.",
        [GEMMA31B_4BIT_LEAD],
    ),
    entry(
        "decision-2-0-vega-27b",
        "vLLM-SR Decision 2.0 Vega 27B",
        2,
        "LoRA adapter + separate head",
        board(26, 55.88, 56.97, 55.71, 47.91, True),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by the model card; adapter_config leaves "
             "base_model_name_or_path null and the vendored decision2 runtime owns the merge"),
        weights("vllm-sr/Decision-2.0-Vega-27B", "71d6ae1db4a786d98775d65c68c41be0e39bfe7c",
                "no merged backbone: a 14,941,293,880-byte LoRA adapter (rank 512, alpha 1024 - "
                "nearly full-rank) in adapter/, a 21,056,344-byte decision_head.safetensors, "
                "and the vendored decision2 Python runtime; a first-party GGUF lives in "
                "vllm-sr/Decision-2.0-Vega-27B-GGUF",
                "apache-2.0", "71d6ae1db4a786d98775d65c68c41be0e39bfe7c", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The rank-512 adapter is nearly a full-parameter delta, so the adapter-merge cost "
        "approaches a full 27B download; the d3 family supersedes this predecessor and the "
        "decision2 runtime is Python code openkind would have to re-express.",
        "The d3 family adapter is the forward path; this entry is kept as the predecessor "
        "baseline and for its first-party GGUF.",
        [lead("moritalous/Decision-2.0-Lux-9B-MLX-4bit", "50671d85a390f14a06e3372f5dd9583ca4649310",
              "4-bit", "vllm-sr/Decision-2.0-Lux-9B",
              "Recorded on the 9B sibling entry; no Vega MLX conversion was found. Not parity "
              "evidence.")],
    ),
    entry(
        "decision-2-0-lux-9b",
        "vLLM-SR Decision 2.0 Lux 9B",
        2,
        "fine-tune with separate head",
        board(44, 48.21, 46.76, 47.35, 44.45, True),
        base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             ARCH_QWEN35_9B, "declared by the model card"),
        weights("vllm-sr/Decision-2.0-Lux-9B", "1dcd1e7f240eab37ac09a6d0b0f77654b59411f9",
                "backbone/ directory with the merged fine-tuned backbone (10 shards, "
                "17,926,620,224 bytes) plus a 16,845,656-byte decision_head.safetensors and the "
                "vendored decision2 runtime",
                "apache-2.0", "1dcd1e7f240eab37ac09a6d0b0f77654b59411f9", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The decision2 head contract plus the predecessor runtime; d3-flash supersedes it at "
        "the same size with a documented contract.",
        "The d3 family adapter is the forward path; kept as the predecessor baseline.",
        [lead("moritalous/Decision-2.0-Lux-9B-MLX-4bit", "50671d85a390f14a06e3372f5dd9583ca4649310",
              "4-bit", "vllm-sr/Decision-2.0-Lux-9B",
              "Apache-2.0 third-party conversion carrying decision_head.safetensors and the "
              "vendored runtime; never downloaded or compared; not parity evidence.")],
    ),
    entry(
        "decision-2-0-nox-4b",
        "vLLM-SR Decision 2.0 Nox 4B",
        2,
        "fine-tune with separate head",
        board(59, 44.95, 44.21, 43.72, 42.19, True),
        base("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b",
             ARCH_QWEN35_4B,
             "declared by the model card; the same base revision openkind's selected profile pins"),
        weights("vllm-sr/Decision-2.0-Nox-4B", "ae269ecf34a0b3a89026bc1afd9c995279212fad",
                "backbone/ directory with the merged fine-tuned backbone (6 shards, "
                "9,695,398,936 bytes) plus a 10,529,624-byte decision_head.safetensors and the "
                "vendored decision2 runtime",
                "apache-2.0", "ae269ecf34a0b3a89026bc1afd9c995279212fad", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The decision2 head contract at openkind's pinned base geometry; d3-mini supersedes it.",
        "The d3 family adapter is the forward path; kept as the predecessor baseline sharing "
        "openkind's base revision.",
        [lead("moritalous/Decision-2.0-Nox-4B-MLX-4bit", "b316e603d16bc90c32f2fe582715567d056e9dc9",
              "4-bit", "vllm-sr/Decision-2.0-Nox-4B",
              "Apache-2.0 third-party conversion carrying decision_head.safetensors and the "
              "vendored runtime; never downloaded or compared; not parity evidence.")],
    ),
    entry(
        "sieve-9b-plus",
        "Sieve 9B Plus",
        2,
        "LoRA adapter + separate head",
        board(33, 52.06, 53.35, 51.61, 45.38, True),
        base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             ARCH_QWEN35_9B, "declared by adapter_config.json base_model_name_or_path"),
        weights("sthanika-ai/Sieve-9B-Plus", "02d213693cc79d0fc02095cc1b07c959b8b4486d",
                "no merged weights: a 346,291,760-byte LoRA adapter (rank 32, alpha 64) and an "
                "8,394,303-byte torch head.pt",
                "apache-2.0", "02d213693cc79d0fc02095cc1b07c959b8b4486d", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The Sieve adapter-plus-torch-head pattern at 9B; same head.pt re-expression problem "
        "as Sieve-27B.",
        "The Sieve head re-expression followed by the 9B backbone path.",
        [],
    ),
    entry(
        "sieve-4b",
        "Sieve 4B",
        2,
        "LoRA adapter + separate head",
        board(50, 47.4, 50.06, 47.72, 40.35, True),
        base("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             ARCH_QWEN35_4B, "declared by adapter_config.json base_model_name_or_path"),
        weights("sthanika-ai/Sieve-4B", "ea7cd1edee29590a82ed9fbfcb644227a8eed5f1",
                "no merged weights: a 259,784,032-byte LoRA adapter (rank 32, alpha 64), a "
                "2,611,288-byte head.safetensors ([255, 2560] float32 rows by size), and a "
                "small sieve4b Python engine",
                "apache-2.0", "ea7cd1edee29590a82ed9fbfcb644227a8eed5f1", [],
                "HEAD at survey time and unchanged on re-resolution."),
        "The same adapter-plus-head pattern at openkind's pinned geometry with the head "
        "already in safetensors; the sieve4b engine's prompt bytes still need re-expression.",
        "The closest Sieve to a loader: adapter merge over the decoder-logit-qwen35 geometry "
        "plus the 255-row head.",
        [],
    ),
    entry(
        "nimble-v3",
        "Bespoke Nimble 9B v3",
        2,
        "LoRA adapter",
        board(31, 54.67, 57.19, 52.34, 49.05, False),
        base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             ARCH_QWEN35_9B, "declared by adapter_config.json base_model_name_or_path"),
        weights("bespokelabs/Bespoke-Nimble-9B-v3", "8e927b9b4afdbb14479fac10a7364d1a695be208",
                "no merged weights: a 692,529,024-byte LoRA adapter (rank 64, alpha 128)",
                "cc-by-nc-4.0", "8e927b9b4afdbb14479fac10a7364d1a695be208", [],
                "HEAD at survey time and unchanged on re-resolution. CC BY-NC 4.0 may rule out "
                "catalog distribution; Nimble also supplies pplx training converters."),
        "Adapter-only release whose readout is defined by the Nimble serving stack, plus the "
        "non-commercial license.",
        "ggml-org publishes a GGUF; any loader still needs the Nimble readout contract "
        "re-expressed first.",
        [],
    ),
    entry(
        "clef",
        "Cloudflare clef",
        2,
        "fine-tune with joint head (family already in the catalog)",
        board(32, 53.08, 61.71, 53.11, 41.88, False),
        base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
             ARCH_QWEN38_27B, "declared by the shipped config"),
        weights("Cloudflare/clef", "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
                "merged BF16 safetensors, 13 shards, 54,969,731,792 bytes, plus "
                "joint_head.safetensors and joint_head_config.json",
                "apache-2.0", "0b331204bb13fbd2ca93a64df1956f5b55478ce5", [],
                "HEAD at survey time and unchanged on re-resolution. The clef family is already "
                "curated (clef-27b-gguf:48cb5634b4a258de5a6b and clef-flash profiles); this "
                "entry re-pins the upstream BF16 source at its surveyed HEAD and defers to the "
                "catalog profiles as the executable path."),
        "The upstream BF16 checkpoint is 54.97 GB, above the 36 GB reference host; the catalog's "
        "GGUF profile is the executable relative. No new blocker beyond scale.",
        "clef-27b-gguf remains the executable path; re-pinning here records the upstream source "
        "revision the catalog derives from.",
        [lead("mlx-community/clef-4bit", "b973b6f405d02f8e9b83208c7e82ddf43a742eb6", "4-bit",
              "Cloudflare/clef",
              "joint_head.safetensors and clef_mlx.py; also an 8-bit build at "
              "1c8887c20827d91cb1b0c6f186e24f5bc5b0e0ce. Many further third-party clef MLX "
              "repositories exist (Jundot oQ, nativ-community MXFP, AutomatosX, aufklarer, "
              "TrevorJS); surveyed but not pinned individually. Not parity evidence."),
         lead("mlx-community/clef-8bit", "1c8887c20827d91cb1b0c6f186e24f5bc5b0e0ce", "8-bit",
              "Cloudflare/clef", "The 8-bit companion to clef-4bit. Not parity evidence.")],
        already_catalog=True,
    ),
    entry(
        "clef-flash",
        "Cloudflare clef-flash",
        2,
        "fine-tune with joint head (family already in the catalog)",
        board(47, 47.61, 56.15, 44.01, 42.5, True),
        base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             ARCH_QWEN35_9B, "declared by the shipped config"),
        weights("Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
                "merged BF16 safetensors, 5 shards, 19,063,259,136 bytes, plus "
                "joint_head.safetensors and joint_head_config.json",
                "apache-2.0", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", [],
                "HEAD at survey time and unchanged on re-resolution. Already catalog-installable "
                "as clef-flash:dfe12a21a5c9dd5b2fb1 (BF16) and clef-flash-gguf:c330d9ee7e9cc658ad45."),
        "No new blocker: this entry records the upstream source revision behind the existing "
        "installable profiles.",
        "clef-flash:dfe12a21a5c9dd5b2fb1 remains the executable path.",
        [lead("mlx-community/clef-flash-4bit", "b5adbe2c3a95200d4bf04a0727c8ec457157d5e1",
              "4-bit", "Cloudflare/clef-flash",
              "Also an 8-bit build at 5a195dc147926b63b1d139691ed4480ca9f92c0e. Not parity "
              "evidence."),
         lead("mlx-community/clef-flash-8bit", "5a195dc147926b63b1d139691ed4480ca9f92c0e",
              "8-bit", "Cloudflare/clef-flash", "The 8-bit companion. Not parity evidence.")],
        already_catalog=True,
    ),
]

# ---------------------------------------------------------------------------
# Base models, excluded rows
# ---------------------------------------------------------------------------

BASE_MODELS = [
    base("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", ARCH_QWEN38_27B,
         ["vllm-sr-d3", "pplx-decider-v1.1-27b", "sieve-27b", "kev-27b", "matilda-jev-v1.5",
          "rsi-jev-v6.1-vl-27b", "qwen-decision-27b-noncausal", "linne-fern-1-0",
          "startlux-decision-v1-27b"],
         "The same revision is pinned by the 0.2.1 file and unchanged on 2026-10-12 "
         "(lastModified 2026-08-14)."),
    base("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", ARCH_GEMMA31B,
         ["decider-31b", "quyet-1-0-large", "garas"],
         "The same revision is pinned by the 0.2.1 file."),
    base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", ARCH_QWEN35_9B,
         ["d3-flash", "sieve-9b-plus", "nimble-v3", "clef-flash", "startlux-decision-v1-9b",
          "decision-2-0-lux-9b"]),
    base("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b", ARCH_QWEN35_4B,
         ["rsi-jev-v6.1-vl-4b", "decision-2-0-nox-4b"],
         "The same revision openkind's selected profile pins."),
    base("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", ARCH_QWEN35_4B,
         ["d3-mini", "sieve-4b", "torchcast-decision-4b", "startlux-decision-v1-4b"],
         "The instruct (post-trained) 4B, not the -Base checkpoint; declared by the d3-mini "
         "decision_config and the Sieve-4B adapter_config."),
    base("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", ARCH_QWEN35_2B,
         ["d3-nano"], "Resolved on 2026-10-12; the handoff left this base unresolved."),
    base("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", ARCH_QWEN35_08B,
         ["d3-lite", "d3-edge"],
         "Resolved on 2026-10-12; both small d3 siblings declare it."),
    base("Qwen/Qwen3.5-2B-Base", "b1485b2fa6df2c0e83c08e14eb4e91d3906ce874",
         ARCH_QWEN35_2B, [],
         "Resolved for completeness per the handoff; no pinned entry declares it. Stated as "
         "the likely base of Sieve 2B Plus, which is not a pinned entry here."),
    base("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", ARCH_GEMMA26B,
         ["shisa-de-2-repeat", "xor-26b-a4b", "blink-v0-3-26b-a4b-nvfp4"],
         "The same revision is pinned by the 0.2.1 file."),
    base("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", ARCH_GEMMA12B,
         ["torchcast-decision-12b"], "The same revision is pinned by the 0.2.1 file."),
    base("FINAL-Bench/Darwin-27B-RSI", "08d44a01bbe4762fbc95171c952f0536bd5d46c6",
         ARCH_DARWIN_BASE, ["darwin-27b-ztc"],
         "The Darwin-27B-ZTC decision_config records the base only as a local-sha256 revision; "
         "the Hub repository was resolved separately and its lineage to Qwen/Qwen3.8-27B is "
         "author-disclosed, not verified."),
]

EXCLUDED = [
    {"entry": "Jev (TypeSafe)", "reason": "Closed weights; never pinnable."},
    {"entry": "Fastino GLiDE no-thinking (28B), board rank 8",
     "reason": "Board weights_note says 'Open weights coming soon' and no repository exists. "
               "The fastino/GLiNER2.5-Decide models are different, smaller models; not substitutes."},
    {"entry": "AnyJev L0, decisio, deck31b, reflex, Decider chat, simple-jev, Jevfire, "
              "JoshuaSP diffgemma",
     "reason": "kind: inference technique - prompting or decoding over stock checkpoints with "
               "no new weights, outside openkind's single-forward readout contract."},
    {"entry": "RSI-Jev v6.1 27B effort variants",
     "reason": "Same repository as the effort-auto row; one entry."},
    {"entry": "Darwin-27B-ZTC-v2",
     "reason": "Exists upstream but is not on the 0.3.1 board; re-check the board before adding."},
]

# ---------------------------------------------------------------------------

DOCUMENT = {
    "schema": "openkind-jev-decision-index/v2",
    "checked_at": "2026-10-12",
    "installable": False,
    "scope": (
        "The open-weight Decision Index 0.3.1 leaders plus the best small open entries and "
        "their base models: tier 1 (twelve repositories the next family effort should start "
        "from, including the whole d3 family) and tier 2 (the remaining board entries surveyed, "
        "several under non-commercial licenses). Supplemental research metadata only: none of "
        "these entries is a catalog profile, and no OpenKind loader accepts these checkpoints "
        "today. Each entry pins the re-resolved revisions of the upstream weights and any MLX "
        "conversion leads so later qualification starts from immutable sources. Supersedes "
        "registry/v1/jev-decision-index.json (Decision Index 0.2.1), which is retained as "
        "history."
    ),
    "source": {
        "leaderboard": "https://huggingface.co/spaces/multimodalart/jev-decision-index",
        "bundle": "https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/f6571b3212a95c770d6c101ec5bb2157218d8cb1/data/v03.json",
        "bundle_sha256": "482604710eaaeaac424f798c898eac6ccaea161defbd049df629a79215017bb8",
        "index_bundle_sha256": "c03e52d0f7d789e24039f4f0b8b53c4195926d965eae07c890ebf629103c7b4d",
        "methodology_bundle_sha256": "2e81c8327ceef87f1df9537873f4d1c8dfb76579c8d0de6090efb179453eaa53",
        "space_revision": "f6571b3212a95c770d6c101ec5bb2157218d8cb1",
        "bundle_generated_utc": "2026-10-12T04:03:19+00:00",
        "edition": "Decision Index 0.3.1",
        "ranking_field": "v",
        "ranking_field_note": (
            "The Space's 0.3.1 Full score v = 0.20 public + 0.50 private same-skills + 0.30 "
            "private new-domains, equated onto the public scale; ties are declared inside a "
            "0.9-point band, so the d3 and Decider 31B tie group is not separable. The 0.2.1 "
            "edition ranked on scores.balanced_skill instead. All three bundle files were "
            "fetched at the pinned Space revision on 2026-10-12 and hash-match the recorded "
            "digests."
        ),
        "supersedes": "registry/v1/jev-decision-index.json",
    },
    "pin_policy": (
        "Every revision was re-resolved against the Hub API on 2026-10-12; head_at_survey is "
        "the commit observed at the earlier survey and equivalent_revisions lists observed "
        "HEADs whose weight, readout, and config blobs are byte-identical (same sizes and LFS "
        "sha256) to the pin - only README, runtime server, or processor files moved. Where a "
        "board run or parity record evaluated a specific release commit, that commit is the "
        "pin even when tags point elsewhere (the d3 family's Hub tags v3.1.0/v3.1.1/v3.1.2 "
        "disagree with the semantic-router release pins). No weights were downloaded; all "
        "observations are Hub API metadata and shipped configs."
    ),
    "entries": TIER1 + TIER2,
    "base_models": BASE_MODELS,
    "excluded": EXCLUDED,
}


def check(document):
    import re
    hex40 = re.compile(r"^[0-9a-f]{40}$")
    repos = set()
    engines = set()
    for entry in document["entries"]:
        assert entry["engine"] not in engines, entry["engine"]
        engines.add(entry["engine"])
        assert entry["openkind_loadable"] is False
        assert entry["tier"] in (1, 2)
        for field in ("blocker", "nearest_openkind_path"):
            assert entry[field].strip(), entry["engine"]
        if entry["board"] is None:
            assert entry["off_board_scores"] or entry["already_catalog"], entry["engine"]
        for revision in [entry["base_model"]["revision"], entry["weights"]["revision"],
                         entry["weights"]["head_at_survey"], *entry["weights"]["equivalent_revisions"]]:
            assert revision and hex40.match(revision), (entry["engine"], revision)
        assert entry["weights"]["license"], entry["engine"]
        assert entry["weights"]["repository"] not in repos, entry["weights"]["repository"]
        repos.add(entry["weights"]["repository"])
        for model in entry["mlx_leads"]:
            assert hex40.match(model["revision"]), (entry["engine"], model)
            assert model["relationship"] in ("exact-conversion-unverified", "base-model-only")
    for model in document["base_models"]:
        assert hex40.match(model["revision"]), model["repository"]


def main():
    check(DOCUMENT)
    OUT.write_text(json.dumps(DOCUMENT, indent=2) + "\n")
    print(f"wrote {OUT} with {len(TIER1)} tier-1 and {len(TIER2)} tier-2 entries, "
          f"{len(BASE_MODELS)} base models")


if __name__ == "__main__":
    main()
