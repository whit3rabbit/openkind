# Qwen3.5 embedding parity fixture

`embedding_token_25.bf16.hex` is the 5,120-byte row for token ID 25 from
`model.language_model.embed_tokens.weight` in
`Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b`.

The source tensor is BF16 with shape `[248320, 2560]` in
`model.safetensors-00001-of-00002.safetensors`. The source shard has SHA-256
`df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712`.
The extracted row has SHA-256
`84ce40703c960b305ad72adbdc0f6fc5b8df6fd279e5b18d5dd194e4764377c7`.

The fixture is stored as lowercase hexadecimal so it remains reviewable and
portable without adding a binary editing path. Tests decode it locally and do
not access the network.

## Offline Phase 3B reference subset

The contract, architecture, traces, token IDs, probability reference and 47
golden vectors are recovered byte-for-byte from Git commit `d5baa48c95a90f19b8d313fe19e023d1ec220a37`,
under `research/OpenKind_Phase3B_BackboneParity_20260920T152206Z`. `REFERENCE_SHA256.json` records their digests;
the reference loader also verifies the tensor index and each golden vector.

These files total under 650 KiB and contain no checkpoint weights. They preserve
the embedding comparison and frozen-head replay tests on a clean checkout.
The 20 MB pretrained tokenizer is intentionally external: its dedicated
qualification tests require `OPENKIND_QWEN35_TOKENIZER` and `--ignored`.
Experimental renderer unit tests use a synthetic byte tokenizer and make no
pretrained-tokenizer parity claim.
