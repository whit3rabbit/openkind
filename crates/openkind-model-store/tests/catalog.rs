use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use openkind_model_store::{Catalog, Manifest};
use sha2::{Digest, Sha256};

#[test]
fn curated_qwen_manifest_matches_the_pinned_local_bundle() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog_bytes = std::fs::read(repo.join("registry/v1/catalog.json")).unwrap();
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
            repo.join("research/11_phase2ij_model_selection_results/final/434894dccea25608c3c8/runtime/tokenizer/tokenizer.json")
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
