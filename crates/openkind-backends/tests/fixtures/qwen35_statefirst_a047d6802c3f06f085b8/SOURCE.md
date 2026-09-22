# Phase 3.1 reference fixtures

These files are the model-independent Rust parity subset of the public
`cowWhySo/OpenKind-Qwen3.5-4B-StateFirst` reference bundle.

- Hugging Face revision: `20974648aa087369645494e898351253248627a0`
- Profile: `a047d6802c3f06f085b8`
- Exported bundle SHA-256: `4d9ffdee0aea5c71c666d0feae372cffe79a05934aedee2245012e3a53c23332`
- Manifest SHA-256: `dd42289e525d82a1ab8d55efd3843970e6c31a23059512a2c7e4ee7ca6459f78`

`BUNDLE_MANIFEST.json` authenticates the vendored profile, protocol, head
graph, model contract, golden fixtures, and safetensors head. The published
manifest predates the two reload receipts, so their source-revision hashes are:

- `HEAD_ALGEBRA_CHECK.json`: `0ea4c52a86cb46662444ef978a312940cf01b2ea01acc50094cb42927bfd8937`
- `RELOAD_CHECK.json`: `c465a65559476571f0fd502fc5d6646a51c4e6588008fccc9f2958c586f3b314`

The fixture set deliberately excludes Qwen base weights, tokenizer execution,
training data, final labels, and the Python implementation. Tests must remain
offline and validate these files against their recorded hashes before using
their expected values.
