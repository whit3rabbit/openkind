use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use futures::StreamExt;
use reqwest::header::{CONTENT_RANGE, RANGE};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::manifest::{sha256, valid_name, valid_relative_path, valid_sha256};
use crate::{
    Catalog, CatalogEntry, Error, Manifest, Result, CATALOG_URL, ENCODER_EMBEDDING_MODEL_NAME,
    LAYA_ENGLISH_MODEL_NAME, LAYA_MULTILINGUAL_MODEL_NAME, LAYA_TYPED_DECISIONS_MODEL_NAME,
    QWEN35_STATE_FIRST_MODEL_NAME,
};

const MAX_METADATA_BYTES: u64 = 4 * 1024 * 1024;
const PINNED_QWEN_PROFILE: &str = "a047d6802c3f06f085b8";
const PINNED_LAYA_ENGLISH_PROFILE: &str = "c8ea29bf1e33a343c4b7";
const PINNED_LAYA_MULTILINGUAL_PROFILE: &str = "f4064eb56fb7f7d325e1";
const PINNED_LAYA_TYPED_DECISIONS_PROFILE: &str = "9d28cfa9567902801ed1";
const PINNED_ENCODER_EMBEDDING_PROFILE: &str = "8d9498269ef05d95d93c";

fn supported_profile(manifest: &Manifest) -> bool {
    (manifest.name == QWEN35_STATE_FIRST_MODEL_NAME
        && manifest.loader_id == "qwen35-state-first"
        && manifest.profile_id == PINNED_QWEN_PROFILE)
        || (manifest.name == LAYA_ENGLISH_MODEL_NAME
            && manifest.loader_id == "laya-english"
            && manifest.profile_id == PINNED_LAYA_ENGLISH_PROFILE)
        || (manifest.name == LAYA_MULTILINGUAL_MODEL_NAME
            && manifest.loader_id == "laya-multilingual"
            && manifest.profile_id == PINNED_LAYA_MULTILINGUAL_PROFILE)
        || (manifest.name == LAYA_TYPED_DECISIONS_MODEL_NAME
            && manifest.loader_id == "laya-typed-decisions"
            && manifest.profile_id == PINNED_LAYA_TYPED_DECISIONS_PROFILE)
        || (manifest.name == ENCODER_EMBEDDING_MODEL_NAME
            && manifest.loader_id == "encoder-embedding"
            && manifest.profile_id == PINNED_ENCODER_EMBEDDING_PROFILE)
        || (cfg!(test)
            && manifest.loader_id == "test-loader"
            && manifest.profile_id == "test-profile")
}

#[derive(Debug)]
pub struct ModelStore {
    root: PathBuf,
    catalog_url: String,
    client: reqwest::Client,
}

/// A verified local installation. The shared file lock protects it from `rm`
/// for as long as the daemon holds this value.
#[derive(Debug)]
pub struct InstalledModel {
    pub manifest: Manifest,
    pub root: PathBuf,
    _lock: File,
}

pub fn default_models_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("OPENKIND_MODELS_DIR") {
        if path.is_empty() {
            return Err(Error::Invalid("OPENKIND_MODELS_DIR is empty".into()));
        }
        return Ok(PathBuf::from(path));
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| Error::Invalid("HOME is unavailable; set OPENKIND_MODELS_DIR".into()))?;
        Ok(PathBuf::from(home).join("Library/Application Support/openkind/models"))
    }
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("APPDATA")
            .or_else(|| {
                std::env::var_os("USERPROFILE")
                    .map(|p| PathBuf::from(p).join("AppData/Roaming").into_os_string())
            })
            .ok_or_else(|| {
                Error::Invalid("APPDATA is unavailable; set OPENKIND_MODELS_DIR".into())
            })?;
        Ok(PathBuf::from(base).join("openkind/models"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let base = match std::env::var_os("XDG_DATA_HOME") {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(std::env::var_os("HOME").ok_or_else(|| {
                Error::Invalid("HOME is unavailable; set OPENKIND_MODELS_DIR".into())
            })?)
            .join(".local/share"),
        };
        Ok(base.join("openkind/models"))
    }
}

impl ModelStore {
    pub fn new(root: PathBuf) -> Result<Self> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() >= 8 || attempt.url().scheme() != "https" {
                    attempt.stop()
                } else {
                    attempt.follow()
                }
            }))
            .build()?;
        Ok(Self {
            root,
            catalog_url: CATALOG_URL.to_owned(),
            client,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub async fn catalog(&self) -> Result<Catalog> {
        let bytes = self.fetch_small(&self.catalog_url).await?;
        let catalog: Catalog = serde_json::from_slice(&bytes)?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn list(&self) -> Result<Vec<Manifest>> {
        let mut models = Vec::new();
        let dir = self.root.join("models");
        if !dir.exists() {
            return Ok(models);
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || !valid_name(&name) {
                continue;
            }
            models.push(self.read_installed_manifest(&name)?);
        }
        models.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(models)
    }

    pub fn show(&self, name: &str) -> Result<Manifest> {
        self.read_installed_manifest(name)
    }

    pub fn acquire_serving(&self, name: &str) -> Result<InstalledModel> {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid model name".into()));
        }
        let lock = self.model_lock(name, false)?;
        let manifest = self.read_installed_manifest(name)?;
        let root = self.model_dir(name);
        for artifact in &manifest.artifacts {
            verify_file(&root.join(&artifact.path), artifact.size, &artifact.sha256)?;
        }
        Ok(InstalledModel {
            manifest,
            root,
            _lock: lock,
        })
    }

    pub fn rm(&self, name: &str) -> Result<()> {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid model name".into()));
        }
        let _global = self.global_lock()?;
        let _model = self.model_lock(name, true)?;
        let manifest = self.read_installed_manifest(name)?;
        let used: HashSet<String> = self
            .list()?
            .into_iter()
            .filter(|model| model.name != name)
            .flat_map(|model| model.artifacts.into_iter().map(|a| a.sha256))
            .collect();
        fs::remove_dir_all(self.model_dir(name))?;
        for artifact in manifest.artifacts {
            if !used.contains(&artifact.sha256) {
                let blob = self.blob_path(&artifact.sha256);
                if blob.exists() {
                    fs::remove_file(blob)?;
                }
            }
        }
        Ok(())
    }

    /// Pull only a model named in the repository-controlled catalog. The
    /// callback receives the artifact path, present bytes, and expected bytes.
    /// For artifacts being pulled, it runs before remote requests and local
    /// blob verification, then as downloaded bytes arrive, so callers can
    /// report stalled transfers and verification work as well as byte
    /// movement. The count can return to zero when partial data is discarded.
    pub async fn pull<F>(&self, name: &str, mut progress: F) -> Result<Manifest>
    where
        F: FnMut(&str, u64, u64),
    {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid model name".into()));
        }
        let _global = self.global_lock()?;
        let _model = self.model_lock(name, true)?;
        let catalog = self.catalog().await?;
        let entry = catalog
            .models
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| Error::NotCurated(name.into()))?;
        let (manifest, manifest_bytes) = self.fetch_manifest(entry).await?;
        if !supported_profile(&manifest) {
            return Err(Error::Invalid(format!(
                "this OpenKind build cannot load profile {}",
                manifest.profile_id
            )));
        }
        if self.model_dir(name).exists() {
            let existing = self.read_installed_manifest(name)?;
            if sha256(&fs::read(self.model_dir(name).join("manifest.json"))?)
                == sha256(&manifest_bytes)
            {
                for artifact in &existing.artifacts {
                    verify_file(
                        &self.model_dir(name).join(&artifact.path),
                        artifact.size,
                        &artifact.sha256,
                    )?;
                }
                return Ok(existing);
            }
            return Err(Error::Invalid(format!(
                "installed model {name} differs from the curated manifest"
            )));
        }
        fs::create_dir_all(self.root.join("blobs/sha256"))?;
        for artifact in &manifest.artifacts {
            let blob = self.blob_path(&artifact.sha256);
            if blob.exists() {
                progress(&artifact.path, artifact.size, artifact.size);
                verify_file(&blob, artifact.size, &artifact.sha256)?;
            } else {
                self.download_artifact(artifact, &blob, &mut progress)
                    .await?;
            }
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::Invalid("system clock before Unix epoch".into()))?
            .as_nanos();
        let stage = self.root.join("models").join(format!(
            ".stage-{}-{}-{nonce}",
            name,
            std::process::id()
        ));
        fs::create_dir_all(&stage)?;
        let install = (|| -> Result<()> {
            for artifact in &manifest.artifacts {
                let target = stage.join(&artifact.path);
                fs::create_dir_all(target.parent().expect("artifact has a parent"))?;
                fs::hard_link(self.blob_path(&artifact.sha256), target)?;
            }
            fs::write(stage.join("manifest.json"), &manifest_bytes)?;
            fs::rename(&stage, self.model_dir(name))?;
            Ok(())
        })();
        if install.is_err() {
            let _ = fs::remove_dir_all(&stage);
        }
        install?;
        Ok(manifest)
    }

    async fn fetch_manifest(&self, entry: &CatalogEntry) -> Result<(Manifest, Vec<u8>)> {
        let base = reqwest::Url::parse(&self.catalog_url)
            .map_err(|e| Error::Invalid(format!("catalog URL: {e}")))?;
        let url = base
            .join(&entry.manifest_path)
            .map_err(|e| Error::Invalid(format!("manifest URL: {e}")))?;
        if url.origin() != base.origin() {
            return Err(Error::Invalid(
                "manifest URL escapes the catalog host".into(),
            ));
        }
        let bytes = self.fetch_small(url.as_str()).await?;
        if sha256(&bytes) != entry.manifest_sha256 {
            return Err(Error::DigestMismatch(entry.manifest_path.clone()));
        }
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
        if !entry.matches(&manifest) {
            return Err(Error::Invalid(
                "catalog and manifest identities differ".into(),
            ));
        }
        Ok((manifest, bytes))
    }

    async fn fetch_small(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.client.get(url).send().await?.error_for_status()?;
        if response
            .content_length()
            .is_some_and(|size| size > MAX_METADATA_BYTES)
        {
            return Err(Error::Invalid("remote metadata exceeds 4 MiB".into()));
        }
        let bytes = response.bytes().await?;
        if bytes.len() as u64 > MAX_METADATA_BYTES {
            return Err(Error::Invalid("remote metadata exceeds 4 MiB".into()));
        }
        Ok(bytes.to_vec())
    }

    async fn download_artifact<F>(
        &self,
        artifact: &crate::Artifact,
        blob: &Path,
        progress: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&str, u64, u64),
    {
        self.download_from_url(&artifact.source.url(), artifact, blob, progress)
            .await
    }

    async fn download_from_url<F>(
        &self,
        url: &str,
        artifact: &crate::Artifact,
        blob: &Path,
        progress: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&str, u64, u64),
    {
        let part = blob.with_extension("part");
        let mut present = part.metadata().map(|m| m.len()).unwrap_or(0);
        if present == artifact.size {
            progress(&artifact.path, artifact.size, artifact.size);
            match verify_file(&part, artifact.size, &artifact.sha256) {
                Ok(()) => {
                    fs::rename(part, blob)?;
                    return Ok(());
                }
                Err(Error::DigestMismatch(_)) => {
                    fs::remove_file(&part)?;
                    present = 0;
                    progress(&artifact.path, present, artifact.size);
                }
                Err(error) => return Err(error),
            }
        }
        if present > artifact.size {
            fs::remove_file(&part)?;
            present = 0;
        }
        progress(&artifact.path, present, artifact.size);
        let mut request = self.client.get(url);
        if present > 0 {
            request = request.header(RANGE, format!("bytes={present}-"));
        }
        let response = request.send().await?.error_for_status()?;
        let status = response.status();
        if present > 0 && status == reqwest::StatusCode::OK {
            present = 0;
            fs::remove_file(&part)?;
        } else if present > 0 {
            let range = response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|h| h.to_str().ok())
                .unwrap_or("");
            if status != reqwest::StatusCode::PARTIAL_CONTENT
                || !range.starts_with(&format!("bytes {present}-"))
            {
                return Err(Error::Invalid(
                    "resumed response has wrong byte range".into(),
                ));
            }
        } else if status != reqwest::StatusCode::OK {
            return Err(Error::Invalid(
                "artifact response is not a full file".into(),
            ));
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part)
            .await?;
        progress(&artifact.path, present, artifact.size);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            present = present
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::Invalid("artifact size overflow".into()))?;
            if present > artifact.size {
                return Err(Error::Invalid(format!(
                    "artifact {} exceeds declared size",
                    artifact.path
                )));
            }
            file.write_all(&chunk).await?;
            progress(&artifact.path, present, artifact.size);
        }
        file.sync_all().await?;
        drop(file);
        if let Err(error) = verify_file(&part, artifact.size, &artifact.sha256) {
            if matches!(error, Error::DigestMismatch(_)) {
                let _ = fs::remove_file(&part);
            }
            return Err(error);
        }
        fs::rename(part, blob)?;
        Ok(())
    }

    fn read_installed_manifest(&self, name: &str) -> Result<Manifest> {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid model name".into()));
        }
        let path = self.model_dir(name).join("manifest.json");
        let metadata = path
            .metadata()
            .map_err(|_| Error::NotInstalled(name.to_owned()))?;
        if metadata.len() > MAX_METADATA_BYTES {
            return Err(Error::Invalid("installed manifest exceeds 4 MiB".into()));
        }
        let manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
        manifest.validate()?;
        if manifest.name != name {
            return Err(Error::Invalid("installed manifest name mismatch".into()));
        }
        Ok(manifest)
    }

    fn model_dir(&self, name: &str) -> PathBuf {
        self.root.join("models").join(name)
    }

    fn blob_path(&self, sha: &str) -> PathBuf {
        self.root.join("blobs/sha256").join(sha)
    }

    fn global_lock(&self) -> Result<File> {
        fs::create_dir_all(self.root.join("locks"))?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join("locks/.global"))?;
        file.try_lock_exclusive()
            .map_err(|_| Error::Busy("model store".into()))?;
        Ok(file)
    }

    fn model_lock(&self, name: &str, exclusive: bool) -> Result<File> {
        fs::create_dir_all(self.root.join("locks"))?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join("locks").join(format!("{name}.lock")))?;
        let result = if exclusive {
            FileExt::try_lock_exclusive(&file)
        } else {
            FileExt::try_lock_shared(&file)
        };
        result.map_err(|_| Error::Busy(name.into()))?;
        Ok(file)
    }
}

fn verify_file(path: &Path, size: u64, sha: &str) -> Result<()> {
    if !valid_sha256(sha)
        || !valid_relative_path(path.file_name().and_then(|s| s.to_str()).unwrap_or(""))
    {
        return Err(Error::Invalid("invalid artifact identity".into()));
    }
    let metadata = path.symlink_metadata()?;
    if !metadata.file_type().is_file() || metadata.len() != size {
        return Err(Error::DigestMismatch(path.display().to_string()));
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let len = reader.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        hasher.update(&buffer[..len]);
    }
    if format!("{:x}", hasher.finalize()) != sha {
        return Err(Error::DigestMismatch(path.display().to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Artifact, Source};
    use axum::http::{header, HeaderMap, StatusCode};
    use axum::routing::get;
    use axum::Router;
    use tempfile::tempdir;

    const NAME: &str = "fixture:abc123";

    fn fixture_manifest(name: &str, bytes: &[u8]) -> Manifest {
        Manifest {
            schema: "openkind-model/v1".into(),
            name: name.into(),
            profile_id: "test-profile".into(),
            loader_id: "test-loader".into(),
            description: "Offline fixture".into(),
            release_date: "2026-09-25".into(),
            support_status: "rust-loadable".into(),
            question_types: vec!["choice".into()],
            artifacts: vec![Artifact {
                path: "bundle/head.bin".into(),
                size: bytes.len() as u64,
                sha256: sha256(bytes),
                source: Source {
                    kind: "github".into(),
                    repository: "example/models".into(),
                    revision: "a".repeat(40),
                    path: "head.bin".into(),
                },
            }],
        }
    }

    async fn test_store(manifests: Vec<Manifest>) -> (tempfile::TempDir, ModelStore) {
        let entries: Vec<_> = manifests
            .iter()
            .map(|manifest| {
                let bytes = serde_json::to_vec(manifest).unwrap();
                (manifest.name.clone(), bytes)
            })
            .collect();
        let catalog = Catalog {
            schema: "openkind-catalog/v1".into(),
            models: entries
                .iter()
                .map(|(name, bytes)| CatalogEntry {
                    name: name.clone(),
                    profile_id: "test-profile".into(),
                    loader_id: "test-loader".into(),
                    description: "Offline fixture".into(),
                    support_status: "rust-loadable".into(),
                    manifest_path: format!("manifests/{}.json", name.replace(':', "-")),
                    manifest_sha256: sha256(bytes),
                })
                .collect(),
        };
        let catalog_bytes = serde_json::to_vec(&catalog).unwrap();
        let mut app = Router::new().route(
            "/catalog.json",
            get(move || {
                let bytes = catalog_bytes.clone();
                async move { bytes }
            }),
        );
        for (name, bytes) in entries {
            let path = format!("/manifests/{}.json", name.replace(':', "-"));
            app = app.route(
                &path,
                get(move || {
                    let bytes = bytes.clone();
                    async move { bytes }
                }),
            );
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = tempdir().unwrap();
        let mut store = ModelStore::new(dir.path().to_path_buf()).unwrap();
        store.catalog_url = format!("http://{address}/catalog.json");
        (dir, store)
    }

    fn seed_blob(store: &ModelStore, bytes: &[u8]) {
        let blob = store.blob_path(&sha256(bytes));
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(blob, bytes).unwrap();
    }

    #[tokio::test]
    async fn pull_install_lock_and_remove_are_atomic() {
        let bytes = b"verified offline fixture";
        let (_dir, store) = test_store(vec![fixture_manifest(NAME, bytes)]).await;
        seed_blob(&store, bytes);
        let model = store.pull(NAME, |_, _, _| {}).await.unwrap();
        assert_eq!(model.name, NAME);
        assert_eq!(store.list().unwrap().len(), 1);
        let guard = store.acquire_serving(NAME).unwrap();
        assert!(matches!(store.rm(NAME), Err(Error::Busy(_))));
        drop(guard);
        store.rm(NAME).unwrap();
        assert!(store.list().unwrap().is_empty());
        assert!(!store.blob_path(&sha256(bytes)).exists());
    }

    #[tokio::test]
    async fn shared_blob_survives_until_last_model_is_removed() {
        let bytes = b"shared artifact";
        let second = "fixture:def456";
        let (_dir, store) = test_store(vec![
            fixture_manifest(NAME, bytes),
            fixture_manifest(second, bytes),
        ])
        .await;
        seed_blob(&store, bytes);
        store.pull(NAME, |_, _, _| {}).await.unwrap();
        store.pull(second, |_, _, _| {}).await.unwrap();
        store.rm(NAME).unwrap();
        assert!(store.blob_path(&sha256(bytes)).exists());
        store.rm(second).unwrap();
        assert!(!store.blob_path(&sha256(bytes)).exists());
    }

    #[tokio::test]
    async fn concurrent_pulls_leave_one_complete_installation() {
        let bytes = b"concurrent fixture";
        let (_dir, store) = test_store(vec![fixture_manifest(NAME, bytes)]).await;
        seed_blob(&store, bytes);
        let (first, second) = tokio::join!(
            store.pull(NAME, |_, _, _| {}),
            store.pull(NAME, |_, _, _| {})
        );
        assert!(first.is_ok() ^ second.is_ok());
        assert!(matches!(
            first.as_ref().err().or(second.as_ref().err()),
            Some(Error::Busy(_))
        ));
        assert_eq!(store.list().unwrap().len(), 1);
        store.acquire_serving(NAME).unwrap();
        // A later pull observes the already verified installation.
        store.pull(NAME, |_, _, _| {}).await.unwrap();
    }

    #[tokio::test]
    async fn corrupt_blob_never_becomes_installed() {
        let bytes = b"correct";
        let (_dir, store) = test_store(vec![fixture_manifest(NAME, bytes)]).await;
        seed_blob(&store, b"incorrect");
        // Put wrong bytes under the expected digest, as could happen after a disk error.
        fs::write(store.blob_path(&sha256(bytes)), b"incorrect").unwrap();
        assert!(matches!(
            store.pull(NAME, |_, _, _| {}).await,
            Err(Error::DigestMismatch(_))
        ));
        assert!(store.list().unwrap().is_empty());
    }

    #[tokio::test]
    async fn resumes_from_the_reported_byte_range() {
        let bytes = b"abcdefghij";
        let artifact = fixture_manifest(NAME, bytes).artifacts.remove(0);
        let mut app = Router::new();
        app = app.route(
            "/artifact",
            get(move |headers: HeaderMap| async move {
                let range = headers.get(header::RANGE).and_then(|h| h.to_str().ok());
                if range == Some("bytes=4-") {
                    (
                        StatusCode::PARTIAL_CONTENT,
                        [(header::CONTENT_RANGE, "bytes 4-9/10")],
                        bytes[4..].to_vec(),
                    )
                } else {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_RANGE, "")],
                        bytes.to_vec(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
        let blob = store.blob_path(&artifact.sha256);
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(blob.with_extension("part"), &bytes[..4]).unwrap();
        store
            .download_from_url(
                &format!("http://{address}/artifact"),
                &artifact,
                &blob,
                &mut |_, _, _| {},
            )
            .await
            .unwrap();
        assert_eq!(fs::read(blob).unwrap(), bytes);
    }
}
