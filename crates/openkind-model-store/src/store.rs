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
    Catalog, CatalogEntry, Error, Manifest, Result, CATALOG_SHA256, CATALOG_URL,
    CLEF_27B_GGUF_MODEL_NAME, CLEF_FLASH_GGUF_MODEL_NAME, CLEF_FLASH_MODEL_NAME,
    DECIDER_4B_MODEL_NAME, DECODER_LOGIT_LETTER_MODEL_NAME, DECODER_LOGIT_LLM_MODEL_NAME,
    DECODER_LOGIT_QWEN35_MODEL_NAME, DECODER_LOGIT_QWEN3_06B_MODEL_NAME,
    DECODER_LOGIT_QWEN3_17B_MODEL_NAME, DECODER_LOGIT_QWEN3_4B_MODEL_NAME,
    ENCODER_EMBEDDING_MODEL_NAME, ENCODER_INSTRUCT_LABEL_MODEL_NAME, ENCODER_NLI_MODEL_NAME,
    KEV_MODEL_NAME, LAYA_ENGLISH_MODEL_NAME, LAYA_MULTILINGUAL_MODEL_NAME,
    LAYA_TYPED_DECISIONS_MODEL_NAME, PLUMB_4B_MODEL_NAME, QWEN35_STATE_FIRST_MODEL_NAME,
    QWEN3GUARD_MODEL_NAME, SCHEMA_SCORER_MODEL_NAME, STRANDS_DECIDER_2B_MODEL_NAME, VON_MODEL_NAME,
    WINNOW_E4B_MODEL_NAME, WINNOW_MODEL_NAME,
};

const MAX_METADATA_BYTES: u64 = 4 * 1024 * 1024;

/// The pinned `(model name, loader id, profile id)` triples this build can
/// pull and serve. The profile ids mirror the `PROFILE_ID` constants in
/// `openkind-backends` family loaders; the server dispatch re-checks them
/// against those constants when loading an installation.
const SUPPORTED_PROFILES: &[(&str, &str, &str)] = &[
    (
        ENCODER_EMBEDDING_MODEL_NAME,
        "encoder-embedding",
        "8d9498269ef05d95d93c",
    ),
    (
        QWEN35_STATE_FIRST_MODEL_NAME,
        "qwen35-state-first",
        "a047d6802c3f06f085b8",
    ),
    (
        LAYA_ENGLISH_MODEL_NAME,
        "laya-english",
        "c8ea29bf1e33a343c4b7",
    ),
    (
        LAYA_MULTILINGUAL_MODEL_NAME,
        "laya-multilingual",
        "f4064eb56fb7f7d325e1",
    ),
    (
        LAYA_TYPED_DECISIONS_MODEL_NAME,
        "laya-typed-decisions",
        "9d28cfa9567902801ed1",
    ),
    (
        DECODER_LOGIT_LETTER_MODEL_NAME,
        "decoder-logit-letter",
        "5492c97dfcdaf3fe9439",
    ),
    (CLEF_FLASH_MODEL_NAME, "clef-flash", "dfe12a21a5c9dd5b2fb1"),
    (
        CLEF_FLASH_GGUF_MODEL_NAME,
        "clef-flash-gguf",
        "c330d9ee7e9cc658ad45",
    ),
    (
        CLEF_27B_GGUF_MODEL_NAME,
        "clef-27b-gguf",
        "48cb5634b4a258de5a6b",
    ),
    (
        ENCODER_NLI_MODEL_NAME,
        "encoder-nli",
        "1041a4c362338a61b820",
    ),
    (
        ENCODER_INSTRUCT_LABEL_MODEL_NAME,
        "encoder-instruct-label",
        "9fd68313a5606eca42f2",
    ),
    (
        DECODER_LOGIT_LLM_MODEL_NAME,
        "decoder-logit-llm",
        "465963d705b6f35d6208",
    ),
    (
        SCHEMA_SCORER_MODEL_NAME,
        "schema-scorer",
        "5a7350af556f0ee66566",
    ),
    (QWEN3GUARD_MODEL_NAME, "qwen3guard", "0fcf416cab16d94f933d"),
    (KEV_MODEL_NAME, "kev", "39d88c11faeb4ac165fa"),
    (
        DECODER_LOGIT_QWEN35_MODEL_NAME,
        "decoder-logit-qwen35",
        "415bcf4a064e6dadcf85",
    ),
    (
        DECODER_LOGIT_QWEN3_06B_MODEL_NAME,
        "decoder-logit-qwen3-06b",
        "d900f4af57509fe02e62",
    ),
    (
        DECODER_LOGIT_QWEN3_17B_MODEL_NAME,
        "decoder-logit-qwen3-17b",
        "8119b9271f8d011e7d03",
    ),
    (
        DECODER_LOGIT_QWEN3_4B_MODEL_NAME,
        "decoder-logit-qwen3-4b",
        "9dfaf11792a8d061b6b8",
    ),
    (PLUMB_4B_MODEL_NAME, "plumb-4b", "c1f080794d38e94a0bc2"),
    (DECIDER_4B_MODEL_NAME, "decider-4b", "0529bf6f2bed84641701"),
    (VON_MODEL_NAME, "von", "69219703407bd39cca0c"),
    (WINNOW_MODEL_NAME, "winnow", "4dff8c5b03cfbf680db6"),
    (WINNOW_E4B_MODEL_NAME, "winnow-e4b", "656ac636ce450cf79c7d"),
    (
        STRANDS_DECIDER_2B_MODEL_NAME,
        "strands-decider-2b",
        "6a02bb0d1c6b25cae74b",
    ),
];

fn supported_profile(manifest: &Manifest) -> bool {
    SUPPORTED_PROFILES
        .iter()
        .any(|(name, loader_id, profile_id)| {
            manifest.name == *name
                && manifest.loader_id == *loader_id
                && manifest.profile_id == *profile_id
        })
        || (cfg!(test)
            && manifest.loader_id == "test-loader"
            && manifest.profile_id == "test-profile")
}

#[derive(Debug)]
pub struct ModelStore {
    root: PathBuf,
    catalog_url: String,
    catalog_sha256: String,
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
            catalog_sha256: CATALOG_SHA256.to_owned(),
            client,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub async fn catalog(&self) -> Result<Catalog> {
        let bytes = self.fetch_small(&self.catalog_url).await?;
        if sha256(&bytes) != self.catalog_sha256 {
            return Err(Error::DigestMismatch("catalog.json".into()));
        }
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
            let raw = entry.file_name().to_string_lossy().into_owned();
            if raw.starts_with('.') {
                continue;
            }
            let Some(name) = from_disk_name(&raw) else {
                continue;
            };
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
    /// name may be a curated pull name or one of its catalog aliases; the
    /// installation is always keyed by the canonical pull name. The
    /// callback receives the artifact path, present bytes, and expected
    /// bytes. For artifacts being pulled, it runs before remote requests
    /// and local blob verification, then as downloaded bytes arrive, so
    /// callers can report stalled transfers and verification work as well
    /// as byte movement. The count can return to zero when partial data is
    /// discarded.
    pub async fn pull<F>(&self, name: &str, mut progress: F) -> Result<Manifest>
    where
        F: FnMut(&str, u64, u64),
    {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid model name".into()));
        }
        let _global = self.global_lock()?;
        let catalog = self.catalog().await?;
        let entry = catalog
            .models
            .iter()
            .find(|entry| entry.name == name || entry.aliases.iter().any(|alias| alias == name))
            .ok_or_else(|| Error::NotCurated(name.into()))?;
        let canonical = entry.name.clone();
        let _model = self.model_lock(&canonical, true)?;
        let (manifest, manifest_bytes) = self.fetch_manifest(entry).await?;
        if !supported_profile(&manifest) {
            return Err(Error::Invalid(format!(
                "this OpenKind build cannot load profile {}",
                manifest.profile_id
            )));
        }
        if self.model_dir(&canonical).exists() {
            let existing = self.read_installed_manifest(&canonical)?;
            if sha256(&fs::read(self.model_dir(&canonical).join("manifest.json"))?)
                == sha256(&manifest_bytes)
            {
                for artifact in &existing.artifacts {
                    verify_file(
                        &self.model_dir(&canonical).join(&artifact.path),
                        artifact.size,
                        &artifact.sha256,
                    )?;
                }
                return Ok(existing);
            }
            return Err(Error::Invalid(format!(
                "installed model {canonical} differs from the curated manifest"
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
            on_disk_name(&canonical),
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
            fs::rename(&stage, self.model_dir(&canonical))?;
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
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            // Enforce the cap while streaming, including bodies without Content-Length.
            if chunk.len() > MAX_METADATA_BYTES as usize - bytes.len() {
                return Err(Error::Invalid("remote metadata exceeds 4 MiB".into()));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
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
        let mut present = match part.symlink_metadata() {
            Ok(metadata) if metadata.file_type().is_file() => metadata.len(),
            Ok(_) => {
                return Err(Error::Invalid(
                    "partial artifact is not a regular file".into(),
                ))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        if present == artifact.size {
            progress(&artifact.path, artifact.size, artifact.size);
            match verify_file(&part, artifact.size, &artifact.sha256) {
                Ok(()) => {
                    replace_part(&part, blob)?;
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
        let response_end = if present > 0 && status == reqwest::StatusCode::OK {
            present = 0;
            fs::remove_file(&part)?;
            artifact.size
        } else if present > 0 {
            let range = response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|h| h.to_str().ok())
                .unwrap_or("");
            if status != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(Error::Invalid(
                    "resumed response has wrong byte range".into(),
                ));
            }
            resumed_response_end(range, present, artifact.size)
                .ok_or_else(|| Error::Invalid("resumed response has wrong byte range".into()))?
        } else if status != reqwest::StatusCode::OK {
            return Err(Error::Invalid(
                "artifact response is not a full file".into(),
            ));
        } else {
            artifact.size
        };
        let mut options = tokio::fs::OpenOptions::new();
        options.create(true).append(true);
        // Do not follow a replaced symlink or block on a replaced FIFO between check and open.
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let mut file = options.open(&part).await?;
        let metadata = file.metadata().await?;
        if !metadata.is_file() || metadata.len() != present {
            return Err(Error::Invalid(
                "partial artifact changed before append".into(),
            ));
        }
        progress(&artifact.path, present, artifact.size);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            present = present
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::Invalid("artifact size overflow".into()))?;
            if present > response_end {
                return Err(Error::Invalid(format!(
                    "artifact {} exceeds declared response range",
                    artifact.path
                )));
            }
            file.write_all(&chunk).await?;
            progress(&artifact.path, present, artifact.size);
        }
        file.sync_all().await?;
        drop(file);
        // An incomplete transfer has no full-file digest yet; retain it for the next range request.
        if present < artifact.size {
            return Err(Error::Invalid(format!(
                "artifact {} is incomplete: received {present} of {} bytes",
                artifact.path, artifact.size
            )));
        }
        if let Err(error) = verify_file(&part, artifact.size, &artifact.sha256) {
            if matches!(error, Error::DigestMismatch(_)) {
                let _ = fs::remove_file(&part);
            }
            return Err(error);
        }
        replace_part(&part, blob)?;
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
        self.root.join("models").join(on_disk_name(name))
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
            .open(
                self.root
                    .join("locks")
                    .join(format!("{}.lock", on_disk_name(name))),
            )?;
        let result = if exclusive {
            FileExt::try_lock_exclusive(&file)
        } else {
            FileExt::try_lock_shared(&file)
        };
        result.map_err(|_| Error::Busy(name.into()))?;
        Ok(file)
    }
}

/// Publish a verified `.part` file as its final blob. Windows `rename` does
/// not replace an existing destination, so clear any stale blob first; the
/// exclusive model lock keeps this window invisible to readers.
fn replace_part(part: &Path, blob: &Path) -> Result<()> {
    if blob.exists() {
        fs::remove_file(blob)?;
    }
    fs::rename(part, blob)?;
    Ok(())
}

/// Map a validated model name to its on-disk spelling. Windows forbids `:`
/// in filenames, so the family/version separator is replaced with `@`, a
/// character the name grammar never emits. Unix keeps the raw spelling for
/// compatibility with stores created before Windows support.
fn on_disk_name(name: &str) -> String {
    if cfg!(windows) {
        name.replace(':', "@")
    } else {
        name.to_owned()
    }
}

/// Inverse of [`on_disk_name`] for a `models/` directory entry; returns
/// `None` when the entry is not a spelled-out installed model.
fn from_disk_name(entry: &str) -> Option<String> {
    if cfg!(windows) {
        let name = entry.replace('@', ":");
        valid_name(&name).then_some(name)
    } else {
        valid_name(entry).then(|| entry.to_owned())
    }
}

fn resumed_response_end(range: &str, present: u64, size: u64) -> Option<u64> {
    let (interval, total) = range.strip_prefix("bytes ")?.split_once('/')?;
    let (start, end) = interval.split_once('-')?;
    let start = start.parse::<u64>().ok()?;
    let end = end.parse::<u64>().ok()?;
    let total = total.parse::<u64>().ok()?;
    (start == present && end >= start && end < size && total == size).then(|| end + 1)
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

    #[test]
    fn pinned_catalog_profiles_are_supported_by_identity() {
        for (name, loader_id, profile_id) in SUPPORTED_PROFILES {
            let manifest = Manifest {
                schema: "openkind-model/v1".into(),
                name: (*name).into(),
                profile_id: (*profile_id).into(),
                loader_id: (*loader_id).into(),
                description: "allowlist probe".into(),
                release_date: "2026-09-29".into(),
                support_status: "rust-loadable".into(),
                question_types: vec!["choice".into()],
                artifacts: vec![],
            };
            assert!(supported_profile(&manifest), "{name}");
            let mut wrong = manifest.clone();
            wrong.profile_id = "0".repeat(20);
            assert!(
                !supported_profile(&wrong),
                "{name} accepted a foreign profile id"
            );
        }
    }

    #[test]
    fn checked_in_catalog_and_manifests_match_compiled_profiles() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let registry = repo.join("registry/v1");
        let catalog_bytes = fs::read(registry.join("catalog.json")).unwrap();
        assert_eq!(sha256(&catalog_bytes), CATALOG_SHA256);
        let catalog: Catalog = serde_json::from_slice(&catalog_bytes).unwrap();
        catalog.validate().unwrap();

        let expected: HashSet<_> = SUPPORTED_PROFILES.iter().copied().collect();
        let actual: HashSet<_> = catalog
            .models
            .iter()
            .map(|entry| {
                (
                    entry.name.as_str(),
                    entry.loader_id.as_str(),
                    entry.profile_id.as_str(),
                )
            })
            .collect();
        assert_eq!(
            actual, expected,
            "catalog entries must have compiled loaders"
        );

        for entry in &catalog.models {
            let manifest_bytes = fs::read(registry.join(&entry.manifest_path)).unwrap();
            assert_eq!(
                sha256(&manifest_bytes),
                entry.manifest_sha256,
                "{}",
                entry.name
            );
            let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();
            manifest.validate().unwrap();
            assert_eq!(manifest.name, entry.name);
            assert_eq!(manifest.loader_id, entry.loader_id);
            assert_eq!(manifest.profile_id, entry.profile_id);
            assert!(supported_profile(&manifest), "{}", entry.name);
        }
    }

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
        test_store_with_aliases(manifests, Vec::new()).await
    }

    async fn test_store_with_aliases(
        manifests: Vec<Manifest>,
        aliases: Vec<(String, Vec<String>)>,
    ) -> (tempfile::TempDir, ModelStore) {
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
                    aliases: aliases
                        .iter()
                        .filter(|(owner, _)| owner == name)
                        .flat_map(|(_, aliases)| aliases.clone())
                        .collect(),
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
        let catalog_sha256 = sha256(&catalog_bytes);
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
        store.catalog_sha256 = catalog_sha256;
        (dir, store)
    }

    fn seed_blob(store: &ModelStore, bytes: &[u8]) {
        let blob = store.blob_path(&sha256(bytes));
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(blob, bytes).unwrap();
    }

    #[tokio::test]
    async fn rejects_catalog_that_does_not_match_the_release_digest() {
        let bytes = b"verified offline fixture";
        let (_dir, mut store) = test_store(vec![fixture_manifest(NAME, bytes)]).await;
        store.catalog_sha256 = "0".repeat(64);
        assert!(matches!(
            store.catalog().await,
            Err(Error::DigestMismatch(path)) if path == "catalog.json"
        ));
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
    async fn pull_resolves_a_catalog_alias_to_the_canonical_installation() {
        let bytes = b"verified offline fixture";
        let alias = "fixture-alias:tag";
        let (_dir, store) = test_store_with_aliases(
            vec![fixture_manifest(NAME, bytes)],
            vec![(NAME.to_owned(), vec![alias.to_owned()])],
        )
        .await;
        seed_blob(&store, bytes);
        let model = store.pull(alias, |_, _, _| {}).await.unwrap();
        assert_eq!(model.name, NAME, "the manifest stays canonical");
        // The installation is keyed by the canonical pull name only.
        assert_eq!(store.list().unwrap().len(), 1);
        assert_eq!(store.list().unwrap()[0].name, NAME);
        store.acquire_serving(NAME).unwrap();
        assert!(store.show(alias).is_err());
        // Pulling the canonical name afterwards observes the same verified
        // installation instead of downloading again.
        store.pull(NAME, |_, _, _| {}).await.unwrap();
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn pull_by_alias_then_by_name_share_one_installation() {
        let bytes = b"verified offline fixture";
        let alias = "fixture-alias:tag";
        let (_dir, store) = test_store_with_aliases(
            vec![fixture_manifest(NAME, bytes)],
            vec![(NAME.to_owned(), vec![alias.to_owned()])],
        )
        .await;
        seed_blob(&store, bytes);
        store.pull(alias, |_, _, _| {}).await.unwrap();
        store.rm(NAME).unwrap();
        assert!(
            !store.blob_path(&sha256(bytes)).exists(),
            "the alias pull must own the same single installation"
        );
        assert!(store.list().unwrap().is_empty());
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
    async fn metadata_alias_cannot_install_or_overwrite_a_verified_blob() {
        let bytes = b"verified artifact";
        let mut manifest = fixture_manifest(NAME, bytes);
        manifest.artifacts[0].path = "MANIFEST.JSON".into();
        let (_dir, store) = test_store(vec![manifest]).await;
        seed_blob(&store, bytes);
        let result = store.pull(NAME, |_, _, _| {}).await;
        assert_eq!(
            fs::read(store.blob_path(&sha256(bytes))).unwrap(),
            bytes,
            "metadata aliases must not mutate a content-addressed blob"
        );
        assert!(matches!(result, Err(Error::Invalid(_))));
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

    async fn mock_url(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}/artifact")
    }

    #[tokio::test]
    async fn chunked_metadata_is_bounded_before_the_response_finishes() {
        let app = Router::new().route(
            "/artifact",
            get(|| async {
                let chunks = futures::stream::once(async {
                    Ok::<_, std::io::Error>(vec![b'x'; MAX_METADATA_BYTES as usize + 1])
                })
                .chain(futures::stream::pending());
                axum::body::Body::from_stream(chunks)
            }),
        );
        let url = mock_url(app).await;
        let dir = tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
        let result =
            tokio::time::timeout(std::time::Duration::from_secs(2), store.fetch_small(&url))
                .await
                .expect("reject oversized metadata without waiting for EOF");
        assert!(matches!(result, Err(Error::Invalid(_))));
    }

    #[tokio::test]
    async fn resumed_download_rejects_inconsistent_range_headers() {
        let bytes = b"abcdefghij";
        let artifact = fixture_manifest(NAME, bytes).artifacts.remove(0);
        for range in [
            "bytes 4-9/999",
            "bytes 4-8/10",
            "bytes 4-3/10",
            "bytes 4-10/10",
            "bytes 4-/10",
            "bytes 4-9/*",
            "bytes 4-18446744073709551615/10",
        ] {
            let app = Router::new().route(
                "/artifact",
                get(move || async move {
                    (
                        StatusCode::PARTIAL_CONTENT,
                        [(header::CONTENT_RANGE, range)],
                        bytes[4..].to_vec(),
                    )
                }),
            );
            let url = mock_url(app).await;
            let dir = tempdir().unwrap();
            let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
            let blob = store.blob_path(&artifact.sha256);
            let part = blob.with_extension("part");
            fs::create_dir_all(blob.parent().unwrap()).unwrap();
            fs::write(&part, &bytes[..4]).unwrap();
            let result = store
                .download_from_url(&url, &artifact, &blob, &mut |_, _, _| {})
                .await;
            assert!(
                matches!(result, Err(Error::Invalid(_))),
                "{range}: {result:?}"
            );
            assert!(!blob.exists(), "{range} must not publish a blob");
            assert_eq!(fs::read(part).unwrap(), &bytes[..4]);
        }
    }

    #[tokio::test]
    async fn short_artifact_response_preserves_bytes_for_the_next_resume() {
        let bytes = b"abcdefghij";
        let app = Router::new().route(
            "/artifact",
            get(move |headers: HeaderMap| async move {
                if headers.get(header::RANGE).and_then(|h| h.to_str().ok()) == Some("bytes=4-") {
                    (
                        StatusCode::PARTIAL_CONTENT,
                        [(header::CONTENT_RANGE, "bytes 4-9/10")],
                        bytes[4..].to_vec(),
                    )
                } else {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_RANGE, "")],
                        bytes[..4].to_vec(),
                    )
                }
            }),
        );
        let url = mock_url(app).await;
        let dir = tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
        let artifact = fixture_manifest(NAME, bytes).artifacts.remove(0);
        let blob = store.blob_path(&artifact.sha256);
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        assert!(matches!(
            store
                .download_from_url(&url, &artifact, &blob, &mut |_, _, _| {})
                .await,
            Err(Error::Invalid(_))
        ));
        assert_eq!(fs::read(blob.with_extension("part")).unwrap(), &bytes[..4]);
        assert!(!blob.exists());
        store
            .download_from_url(&url, &artifact, &blob, &mut |_, _, _| {})
            .await
            .unwrap();
        assert_eq!(fs::read(blob).unwrap(), bytes);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn partial_symlink_cannot_append_outside_the_store() {
        let bytes = b"abcdefghij";
        let app = Router::new().route(
            "/artifact",
            get(move || async move {
                (
                    StatusCode::PARTIAL_CONTENT,
                    [(header::CONTENT_RANGE, "bytes 4-9/10")],
                    bytes[4..].to_vec(),
                )
            }),
        );
        let url = mock_url(app).await;
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let target = outside.path().join("target");
        fs::write(&target, &bytes[..4]).unwrap();
        let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
        let artifact = fixture_manifest(NAME, bytes).artifacts.remove(0);
        let blob = store.blob_path(&artifact.sha256);
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&target, blob.with_extension("part")).unwrap();
        let result = store
            .download_from_url(&url, &artifact, &blob, &mut |_, _, _| {})
            .await;
        assert_eq!(
            fs::read(target).unwrap(),
            &bytes[..4],
            "outside file must be unchanged"
        );
        assert!(matches!(result, Err(Error::Invalid(_))));
        assert!(!blob.exists());
    }
}
