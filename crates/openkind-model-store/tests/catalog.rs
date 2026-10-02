use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use openkind_model_store::{Catalog, Manifest, CATALOG_SHA256};
use sha2::{Digest, Sha256};

#[test]
fn curated_qwen_manifest_matches_the_pinned_local_bundle() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog_bytes = std::fs::read(repo.join("registry/v1/catalog.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&catalog_bytes)),
        CATALOG_SHA256,
        "the production catalog digest must pin the checked-in catalog"
    );
    let catalog: Catalog = serde_json::from_slice(&catalog_bytes).unwrap();
    catalog.validate().unwrap();
    let entry = catalog
        .models
        .iter()
        .find(|entry| entry.name == "qwen35-state-first:a047d6802c3f06f085b8")
        .expect("curated catalog carries the qwen35-state-first profile");
    let manifest_bytes =
        std::fs::read(repo.join("registry/v1").join(&entry.manifest_path)).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest_bytes)),
        entry.manifest_sha256
    );
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.name, entry.name);
    assert_eq!(manifest.profile_id, "a047d6802c3f06f085b8");
    for artifact in &manifest.artifacts {
        if artifact.source.kind != "github" {
            continue;
        }
        assert_eq!(
            artifact.source.repository,
            "whit3rabbit/openkind-model-registry"
        );
        assert_eq!(
            artifact.source.path,
            format!(
                "assets/qwen35-state-first/a047d6802c3f06f085b8/{}",
                artifact.path
            )
        );
        let source = if let Some(path) = artifact.path.strip_prefix("bundle/") {
            repo.join(
                "crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8",
            )
            .join(path)
        } else {
            assert_eq!(artifact.path, "checkpoint/tokenizer.json");
            // Pin the descriptor offline; full tokenizer bytes have an explicit qualification below.
            assert_eq!(artifact.size, 19_989_325);
            assert_eq!(
                artifact.sha256,
                "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
            );
            continue;
        };
        let mut input = BufReader::new(File::open(&source).unwrap());
        let mut digest = Sha256::new();
        let mut size = 0_u64;
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let count = input.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            size += count as u64;
            digest.update(&buffer[..count]);
        }
        assert_eq!(size, artifact.size, "{}", artifact.path);
        assert_eq!(
            format!("{:x}", digest.finalize()),
            artifact.sha256,
            "{}",
            artifact.path
        );
    }
}

#[test]
#[ignore = "requires the external pinned tokenizer; set OPENKIND_QWEN35_TOKENIZER"]
fn curated_qwen_tokenizer_bytes_match_the_pinned_manifest() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(
            repo.join("registry/v1/manifests/qwen35-state-first-a047d6802c3f06f085b8.json"),
        )
        .unwrap(),
    )
    .unwrap();
    manifest.validate().unwrap();
    let artifact = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.path == "checkpoint/tokenizer.json")
        .unwrap();
    let source =
        std::env::var_os("OPENKIND_QWEN35_TOKENIZER").expect("set the external tokenizer path");
    let bytes = std::fs::read(source).expect("read pinned tokenizer");
    assert_eq!(bytes.len() as u64, artifact.size);
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), artifact.sha256);
}
