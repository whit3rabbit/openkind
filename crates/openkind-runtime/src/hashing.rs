//! Plain SHA-256 helpers shared by the backends, bench, server, and evidence
//! writer.
//!
//! These hash raw bytes with no domain separation. Domain-separated token
//! identities live in [`crate::digest`]. Keeping one lowercase-hex formatter
//! here avoids the per-byte `format!` loops that used to be scattered across
//! the workspace.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// Streaming read size for [`sha256_file`]. Large enough to keep multi-GB
/// shard verification I/O-bound, small enough to stay out of the way.
const FILE_CHUNK_BYTES: usize = 1 << 20;

/// Lowercase hex of a digest. sha2 0.11 digests no longer implement
/// `LowerHex`, and a direct dependency on a hex crate is not worth one loop.
pub fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

/// Lowercase hex SHA-256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

/// Lowercase hex SHA-256 over the concatenation of `parts`, without
/// allocating the joined buffer.
#[must_use]
pub fn sha256_hex_parts(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hex_lower(&hasher.finalize())
}

/// Stream a file through SHA-256 in place. Large checkpoint shards are never
/// copied or staged: verification reads them where they live.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0u8; FILE_CHUNK_BYTES];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_lower_covers_every_nibble_edge() {
        assert_eq!(
            hex_lower(&[0x00, 0x09, 0x0a, 0x0f, 0xf0, 0xa5, 0xff]),
            "00090a0ff0a5ff"
        );
        assert_eq!(hex_lower(&[]), "");
    }

    #[test]
    fn hex_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn parts_equal_concatenation() {
        assert_eq!(sha256_hex_parts(&[b"a", b"b", b"c"]), sha256_hex(b"abc"));
    }

    #[test]
    fn file_hash_matches_in_memory_hash() {
        let dir = std::env::temp_dir().join(format!("openkind-hashing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob");
        let body = vec![7u8; (FILE_CHUNK_BYTES * 2) + 13];
        std::fs::write(&path, &body).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_hex(&body));
        std::fs::remove_dir_all(&dir).ok();
    }
}
