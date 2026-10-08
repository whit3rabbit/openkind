//! Pinned artifacts of the JEV-27B-VL 8-bit MLX profile.
//!
//! Generated from `registry/v1/jev-gev-mlx-models.json`; a test keeps the two
//! equal. Each tuple is `(file name, size in bytes, SHA-256)`.

// The digests are consumed by the MLX loader and the tests; default builds
// compile neither.
#![allow(dead_code)]

/// Hub repository of the converted checkpoint.
pub const REPOSITORY: &str = "nativ-community/JEV-27B-VL-MLX-8bit";
/// Immutable Hub revision the digests below were taken from.
pub const REVISION: &str = "a871d5f8787b3d8ce9d618e7260e959393030c7b";
/// Source model the conversion was made from.
pub const SOURCE_REPOSITORY: &str = "autotrust/JEV-27B-VL";
/// Revision of the source model.
pub const SOURCE_REVISION: &str = "4000d2393be6718e604f8e7dca563a780ab78e78";
/// Root `config.json`.
pub const CONFIG: (&str, u64, &str) = (
    "config.json",
    6741,
    "9b588e1ba7933712469bbf3830c655ab450cd53da163fc63b1a01381ac99872c",
);
/// Safetensors weight index.
pub const INDEX: (&str, u64, &str) = (
    "model.safetensors.index.json",
    218127,
    "b62bce6b2acbc06dfa48c5602cda610350f3379310fb255f1d96e523440d649b",
);
/// Tokenizer definition.
pub const TOKENIZER: (&str, u64, &str) = (
    "tokenizer.json",
    19989325,
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
);
/// Weight shards, in index order.
pub const SHARDS: [(&str, u64, &str); 6] = [
    (
        "model-00001-of-00006.safetensors",
        5317707581,
        "45cfd0a4a0326065426dd3a6b25ad5baa40380d213d7663f0174938d933c0361",
    ),
    (
        "model-00002-of-00006.safetensors",
        5354102610,
        "fd2e136ac54e9b4a236d5ff270437ba85c2e989f51bd53c936b862b2f0d83a31",
    ),
    (
        "model-00003-of-00006.safetensors",
        5354184694,
        "8374a6c7d6e1f380e93c0bcd7043317ff77c61d0ca46a9cc5f55ae172d11d370",
    ),
    (
        "model-00004-of-00006.safetensors",
        5337309653,
        "4748aa1f9608f72798acd071716ce927ef6f5cb06d320f44d53750554993379f",
    ),
    (
        "model-00005-of-00006.safetensors",
        5292848464,
        "b00f38282bf9b14b92e4f58e096a5afa5e149938801113796bedabf84cc16817",
    ),
    (
        "model-00006-of-00006.safetensors",
        4037001250,
        "d2c4bd8b312fe5bcb7ccdc6c422e010929c11d5e60afb5496d2abc2e4617a1e0",
    ),
];
