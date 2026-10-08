use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    QWEN3GUARD_MODEL_NAME, SCHEMA_SCORER_MODEL_NAME, STRANDS_DECIDER_2B_MODEL_NAME,
    STRANDS_DECIDER_2B_V21_MODEL_NAME, VON_MODEL_NAME, WINNOW_E4B_MODEL_NAME, WINNOW_MODEL_NAME,
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
    (
        STRANDS_DECIDER_2B_V21_MODEL_NAME,
        "strands-decider-2b",
        "f7156bf28400a79ea1b8",
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
    #[cfg(test)]
    artifact_base_url: Option<String>,
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
    default_models_dir_from(&|key| std::env::var_os(key))
}

fn default_models_dir_from(env: &dyn Fn(&str) -> Option<std::ffi::OsString>) -> Result<PathBuf> {
    if let Some(path) = env("OPENKIND_MODELS_DIR") {
        if path.is_empty() {
            return Err(Error::Invalid("OPENKIND_MODELS_DIR is empty".into()));
        }
        return Ok(PathBuf::from(path));
    }
    #[cfg(target_os = "macos")]
    {
        let home = env("HOME")
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Error::Invalid("HOME is unavailable; set OPENKIND_MODELS_DIR".into()))?;
        Ok(PathBuf::from(home).join("Library/Application Support/openkind/models"))
    }
    #[cfg(target_os = "windows")]
    {
        let base = env("APPDATA")
            .filter(|value| !value.is_empty())
            .or_else(|| {
                env("USERPROFILE")
                    .filter(|value| !value.is_empty())
                    .map(|p| PathBuf::from(p).join("AppData/Roaming").into_os_string())
            })
            .ok_or_else(|| {
                Error::Invalid("APPDATA is unavailable; set OPENKIND_MODELS_DIR".into())
            })?;
        Ok(PathBuf::from(base).join("openkind/models"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let base = match env("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(env("HOME").filter(|value| !value.is_empty()).ok_or_else(
                || Error::Invalid("HOME is unavailable; set OPENKIND_MODELS_DIR".into()),
            )?)
            .join(".local/share"),
        };
        Ok(base.join("openkind/models"))
    }
}

impl ModelStore {
    pub fn new(root: PathBuf) -> Result<Self> {
        if root.as_os_str().is_empty() {
            return Err(Error::Invalid("model directory is empty".into()));
        }
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .read_timeout(Duration::from_secs(120))
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
            #[cfg(test)]
            artifact_base_url: None,
        })
    }

    fn cleanup_stages(&self) -> Result<()> {
        // Every stage is created while the exclusive store lock is held.
        // Once that lock is acquired again, surviving stages are abandoned.
        let entries = match fs::read_dir(self.root.join("models")) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            if !entry.file_name().to_string_lossy().starts_with(".stage-") {
                continue;
            }
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                fs::remove_dir_all(entry.path())?;
            } else if file_type.is_symlink() {
                // Unlink the stage name without following an external target.
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
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
        self.scan_installed(None, true)
    }

    fn scan_installed(
        &self,
        excluded_name: Option<&str>,
        skip_invalid: bool,
    ) -> Result<Vec<Manifest>> {
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
            if excluded_name == Some(name.as_str()) {
                continue;
            }
            match self.read_installed_manifest(&name) {
                Ok(manifest) => models.push(manifest),
                Err(_) if skip_invalid => continue,
                Err(error) => return Err(error),
            }
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
        self.cleanup_stages()?;
        let _model = self.model_lock(name, true)?;
        let manifest = match self.read_installed_manifest(name) {
            Ok(manifest) => Some(manifest),
            Err(Error::NotInstalled(_)) => return Err(Error::NotInstalled(name.to_owned())),
            // Removing a damaged install is the recovery path. Its digests
            // cannot be trusted, so retain blobs rather than risk deleting a
            // blob referenced by another install.
            Err(_) => None,
        };
        let used = self.scan_installed(Some(name), false).ok().map(|models| {
            models
                .into_iter()
                .flat_map(|model| model.artifacts.into_iter().map(|artifact| artifact.sha256))
                .collect::<HashSet<_>>()
        });
        fs::remove_dir_all(self.model_dir(name))?;
        if let (Some(manifest), Some(used)) = (manifest, used) {
            for artifact in manifest.artifacts {
                if !used.contains(&artifact.sha256) {
                    let blob = self.blob_path(&artifact.sha256);
                    if blob.exists() {
                        fs::remove_file(blob)?;
                    }
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
        self.cleanup_stages()?;
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
            if reusable_blob(&blob, artifact.size, &artifact.sha256)? {
                progress(&artifact.path, artifact.size, artifact.size);
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
            let manifest_path = stage.join("manifest.json");
            let mut manifest_file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(manifest_path)?;
            manifest_file.write_all(&manifest_bytes)?;
            manifest_file.sync_all()?;
            // Windows cannot rename a staging directory while its manifest is open.
            drop(manifest_file);
            sync_directory(&stage)?;
            fs::rename(&stage, self.model_dir(&canonical))?;
            sync_directory(&self.root.join("models"))?;
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
        let url = artifact.source.url();
        #[cfg(test)]
        let url = self
            .artifact_base_url
            .as_ref()
            .map(|base| format!("{base}/{}", artifact.source.path))
            .unwrap_or(url);
        self.download_from_url(&url, artifact, blob, progress).await
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

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
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

fn reusable_blob(path: &Path, size: u64, sha: &str) -> Result<bool> {
    match path.symlink_metadata() {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                match parent.metadata() {
                    Ok(metadata) if !metadata.is_dir() => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::NotADirectory,
                            "blob parent path is not a directory",
                        )
                        .into());
                    }
                    Err(parent_error) if parent_error.kind() != std::io::ErrorKind::NotFound => {
                        return Err(parent_error.into());
                    }
                    _ => {}
                }
            }
            return Ok(false);
        }
        Err(error) => return Err(error.into()),
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(Error::Invalid("blob is not a regular file".into()));
        }
        Ok(_) => {}
    }
    match verify_file(path, size, sha) {
        Ok(()) => Ok(true),
        Err(Error::DigestMismatch(_)) => {
            // Unlink the name, rather than rewriting a possibly shared inode.
            fs::remove_file(path)?;
            Ok(false)
        }
        Err(error) => Err(error),
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
                    context_limit: 8192,
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
    async fn pull_replaces_corrupt_blobs_and_cleans_abandoned_stages() {
        let bytes = b"correct";
        let (_dir, mut store) = test_store(vec![fixture_manifest(NAME, bytes)]).await;
        store.artifact_base_url = Some(
            mock_url(Router::new().route("/artifact/head.bin", get(|| async { "correct" }))).await,
        );
        seed_blob(&store, bytes);
        let blob = store.blob_path(&sha256(bytes));
        fs::write(&blob, b"corrupt").unwrap();
        let stage = store.root.join("models/.stage-interrupted");
        fs::create_dir_all(&stage).unwrap();
        fs::hard_link(&blob, stage.join("head.bin")).unwrap();
        store.pull(NAME, |_, _, _| {}).await.unwrap();
        assert!(!stage.exists());
        assert_eq!(fs::read(&blob).unwrap(), bytes);
        let serving = store.acquire_serving(NAME).unwrap();
        assert_eq!(
            fs::read(serving.root.join("bundle/head.bin")).unwrap(),
            bytes
        );
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

    #[cfg(test)]
    mod serving_integrity_tests {
        use super::*;

        fn installed(store: &ModelStore, name: &str, bytes: &[u8]) -> Manifest {
            let manifest = Manifest {
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
            };
            manifest.validate().unwrap();
            let root = store.model_dir(name);
            fs::create_dir_all(root.join("bundle")).unwrap();
            fs::write(root.join("bundle/head.bin"), bytes).unwrap();
            fs::write(
                root.join("manifest.json"),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
            manifest
        }

        #[test]
        fn acquire_serving_detects_a_corrupted_installation() {
            let dir = tempdir().unwrap();
            let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
            let bytes = b"verified bytes";
            installed(&store, NAME, bytes);

            store.acquire_serving(NAME).expect("intact install serves");

            let root = store.model_dir(NAME);
            fs::write(root.join("bundle/head.bin"), b"corrupted bytes").unwrap();
            assert!(matches!(
                store.acquire_serving(NAME),
                Err(Error::DigestMismatch(_))
            ));

            // A truncated artifact fails on size before the digest streams.
            fs::write(root.join("bundle/head.bin"), &bytes[..3]).unwrap();
            assert!(matches!(
                store.acquire_serving(NAME),
                Err(Error::DigestMismatch(_))
            ));
        }

        #[cfg(unix)]
        #[test]
        fn verify_file_rejects_symlinked_artifacts_and_bad_identities() {
            let dir = tempdir().unwrap();
            let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
            let bytes = b"verified bytes";
            let manifest = installed(&store, NAME, bytes);
            let artifact = &manifest.artifacts[0];
            let root = store.model_dir(NAME);

            // A symlink pointing at byte-identical content must still be
            // rejected: artifacts are verified in place, not followed.
            let outside = dir.path().join("outside.bin");
            fs::write(&outside, bytes).unwrap();
            fs::remove_file(root.join("bundle/head.bin")).unwrap();
            std::os::unix::fs::symlink(&outside, root.join("bundle/head.bin")).unwrap();
            assert!(matches!(
                verify_file(
                    &root.join("bundle/head.bin"),
                    artifact.size,
                    &artifact.sha256
                ),
                Err(Error::DigestMismatch(_))
            ));

            fs::remove_file(root.join("bundle/head.bin")).unwrap();
            fs::write(root.join("bundle/head.bin"), bytes).unwrap();
            assert!(matches!(
                verify_file(
                    &root.join("bundle/head.bin"),
                    artifact.size,
                    "A".repeat(64).as_str()
                ),
                Err(Error::Invalid(_))
            ));
            assert!(verify_file(
                &root.join("bundle/head.bin"),
                artifact.size,
                &artifact.sha256
            )
            .is_ok());
        }

        #[tokio::test]
        async fn name_guards_cover_pull_show_acquire_and_remove() {
            let dir = tempdir().unwrap();
            let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
            assert!(matches!(
                store.pull("bad name", |_, _, _| {}).await,
                Err(Error::Invalid(_))
            ));
            assert!(matches!(
                store.acquire_serving("no spaces"),
                Err(Error::Invalid(_))
            ));
            assert!(matches!(store.rm("no spaces"), Err(Error::Invalid(_))));
            assert!(matches!(store.show(NAME), Err(Error::NotInstalled(_))));
            assert!(matches!(
                store.acquire_serving(NAME),
                Err(Error::NotInstalled(_))
            ));

            // An uncurated name fails closed against the offline harness
            // catalog, before any artifact request.
            let bytes = b"verified offline fixture";
            let (_server, store) =
                super::test_store(vec![super::fixture_manifest(NAME, bytes)]).await;
            assert!(matches!(
                store.pull("unknown:v1", |_, _, _| {}).await,
                Err(Error::NotCurated(_))
            ));
            assert!(store.list().unwrap().is_empty());
        }

        #[tokio::test]
        async fn repull_rejects_an_installation_that_differs_from_the_manifest() {
            let bytes = b"verified offline fixture";
            let (_server, store) =
                super::test_store(vec![super::fixture_manifest(NAME, bytes)]).await;
            super::seed_blob(&store, bytes);
            store.pull(NAME, |_, _, _| {}).await.unwrap();

            // Serve the same catalog with different manifest bytes: the
            // installation must survive untouched and the pull must fail.
            let mut drifted = super::fixture_manifest(NAME, bytes);
            drifted.description = "Drifted fixture".into();
            let drifted_bytes = serde_json::to_vec(&drifted).unwrap();
            let served_manifest = drifted_bytes.clone();
            let entries = [(NAME.to_string(), drifted_bytes.clone())];
            let catalog = Catalog {
                schema: "openkind-catalog/v1".into(),
                models: entries
                    .iter()
                    .map(|(name, bytes)| CatalogEntry {
                        name: name.clone(),
                        aliases: vec![],
                        profile_id: "test-profile".into(),
                        loader_id: "test-loader".into(),
                        description: drifted.description.clone(),
                        context_limit: 8192,
                        support_status: "rust-loadable".into(),
                        manifest_path: format!("manifests/{}.json", name.replace(':', "-")),
                        manifest_sha256: sha256(bytes),
                    })
                    .collect(),
            };
            let catalog_bytes = serde_json::to_vec(&catalog).unwrap();
            let served_catalog = catalog_bytes.clone();
            let app = axum::Router::new()
                .route(
                    "/catalog.json",
                    axum::routing::get(move || {
                        let bytes = served_catalog.clone();
                        async move { bytes }
                    }),
                )
                .route(
                    "/manifests/fixture-abc123.json",
                    axum::routing::get(move || {
                        let bytes = served_manifest.clone();
                        async move { bytes }
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let mut drifted_store = ModelStore::new(store.root().to_path_buf()).unwrap();
            drifted_store.catalog_url = format!("http://{address}/catalog.json");
            drifted_store.catalog_sha256 = sha256(&catalog_bytes);

            let result = drifted_store.pull(NAME, |_, _, _| {}).await;
            assert!(
                matches!(&result, Err(Error::Invalid(message))
                if message.contains("differs from the curated manifest")),
                "unexpected error: {result:?}"
            );
            assert_eq!(store.list().unwrap().len(), 1);
            store.acquire_serving(NAME).expect("installation survives");
        }

        #[tokio::test]
        async fn pull_reports_progress_for_present_and_installed_artifacts() {
            let bytes = b"verified offline fixture";
            let (_server, store) =
                super::test_store(vec![super::fixture_manifest(NAME, bytes)]).await;
            super::seed_blob(&store, bytes);
            let mut calls: Vec<(String, u64, u64)> = Vec::new();
            store
                .pull(NAME, |path, present, total| {
                    calls.push((path.to_owned(), present, total));
                })
                .await
                .unwrap();
            assert_eq!(
                calls,
                vec![(
                    "bundle/head.bin".to_owned(),
                    bytes.len() as u64,
                    bytes.len() as u64
                )],
                "a present blob reports its full size before verification"
            );
        }

        #[test]
        fn list_ignores_files_stages_and_unspellable_entries() {
            let dir = tempdir().unwrap();
            let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
            let bytes = b"verified bytes";
            installed(&store, NAME, bytes);

            let models = store.root().join("models");
            // A regular file, a leftover staging directory, and a misspelled
            // directory must all be skipped without failing the listing.
            fs::write(models.join("regular-file"), b"").unwrap();
            fs::create_dir_all(models.join(".stage-1-2-3")).unwrap();
            fs::create_dir_all(models.join("bogus")).unwrap();
            #[cfg(unix)]
            fs::create_dir_all(models.join("fixture@abc123")).unwrap();

            let listed = store.list().unwrap();
            assert_eq!(
                listed.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
                vec![NAME],
                "only the valid installation is listed"
            );
            assert!(
                store
                    .list()
                    .unwrap()
                    .windows(2)
                    .all(|w| w[0].name <= w[1].name),
                "listings are sorted by name"
            );
        }

        #[test]
        fn tampered_installed_manifests_fail_closed() {
            let dir = tempdir().unwrap();
            let store = ModelStore::new(dir.path().to_path_buf()).unwrap();
            let bytes = b"verified bytes";
            installed(&store, NAME, bytes);
            let manifest_path = store.model_dir(NAME).join("manifest.json");

            // Junk bytes fail to decode.
            fs::write(&manifest_path, b"not json").unwrap();
            assert!(matches!(store.show(NAME), Err(Error::Json(_))));

            // A manifest naming a different model is a directory/manifest
            // mismatch.
            let swapped = super::fixture_manifest("fixture:other", bytes);
            fs::write(&manifest_path, serde_json::to_vec(&swapped).unwrap()).unwrap();
            assert!(matches!(store.show(NAME), Err(Error::Invalid(_))));

            // A manifest failing validation (zero-size artifact) is rejected.
            let mut invalid = super::fixture_manifest(NAME, bytes);
            invalid.artifacts[0].size = 0;
            fs::write(&manifest_path, serde_json::to_vec(&invalid).unwrap()).unwrap();
            assert!(matches!(store.show(NAME), Err(Error::Invalid(_))));
        }

        #[test]
        fn disk_names_round_trip_and_reject_invalid_spellings() {
            if cfg!(windows) {
                assert_eq!(on_disk_name("fixture:abc123"), "fixture@abc123");
                assert_eq!(
                    from_disk_name("fixture@abc123").as_deref(),
                    Some("fixture:abc123")
                );
            } else {
                assert_eq!(on_disk_name("fixture:abc123"), "fixture:abc123");
                assert_eq!(
                    from_disk_name("fixture:abc123").as_deref(),
                    Some("fixture:abc123")
                );
                assert_eq!(from_disk_name("fixture@abc123"), None);
            }
            assert_eq!(from_disk_name("junk"), None);
            assert_eq!(from_disk_name(""), None);
        }

        #[test]
        fn default_models_dir_honors_the_override_and_rejects_an_empty_one() {
            static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var_os("OPENKIND_MODELS_DIR");

            let override_dir = tempdir().unwrap();
            std::env::set_var("OPENKIND_MODELS_DIR", override_dir.path());
            assert_eq!(default_models_dir().unwrap(), override_dir.path());

            std::env::set_var("OPENKIND_MODELS_DIR", "");
            assert!(matches!(
                default_models_dir(),
                Err(Error::Invalid(message)) if message.contains("OPENKIND_MODELS_DIR is empty")
            ));

            match previous {
                Some(value) => std::env::set_var("OPENKIND_MODELS_DIR", value),
                None => std::env::remove_var("OPENKIND_MODELS_DIR"),
            }
        }
    }
    #[test]
    fn empty_model_root_is_rejected_before_io() {
        assert!(matches!(
            ModelStore::new(PathBuf::new()),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn reusable_blob_propagates_nonregular_file_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("blob");
        fs::create_dir(&path).unwrap();
        assert!(matches!(
            reusable_blob(&path, 0, &sha256(b"")),
            Err(Error::Invalid(_))
        ));
        assert!(path.is_dir());
    }
    #[test]
    fn empty_platform_fallbacks_require_a_usable_home() {
        let blank = |key: &str| match key {
            "HOME" | "USERPROFILE" | "APPDATA" | "XDG_DATA_HOME" => Some(std::ffi::OsString::new()),
            _ => None,
        };
        assert!(default_models_dir_from(&blank).is_err());
        let home = |key: &str| match key {
            "HOME" | "USERPROFILE" => Some(std::ffi::OsString::from("/usable-home")),
            "APPDATA" | "XDG_DATA_HOME" => Some(std::ffi::OsString::new()),
            _ => None,
        };
        assert!(default_models_dir_from(&home)
            .unwrap()
            .starts_with("/usable-home"));
    }

    #[cfg(unix)]
    #[test]
    fn stage_cleanup_preserves_snapshots_partial_downloads_and_external_targets() {
        let root = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        fs::write(external.path().join("keep"), b"external").unwrap();
        let store = ModelStore::new(root.path().to_owned()).unwrap();
        let installs = root.path().join("models");
        fs::create_dir_all(installs.join("installed-snapshot")).unwrap();
        fs::create_dir_all(root.path().join("blobs/sha256")).unwrap();
        let part = root.path().join("blobs/sha256/resumable.part");
        fs::write(&part, b"partial").unwrap();
        std::os::unix::fs::symlink(external.path(), installs.join(".stage-link")).unwrap();
        let _lock = store.global_lock().unwrap();
        store.cleanup_stages().unwrap();
        assert!(!installs.join(".stage-link").exists());
        assert!(installs.join("installed-snapshot").is_dir());
        assert_eq!(fs::read(&part).unwrap(), b"partial");
        assert_eq!(fs::read(external.path().join("keep")).unwrap(), b"external");
    }
    #[test]
    fn reusable_blob_does_not_hide_unrelated_io_failures() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-directory");
        fs::write(&blocker, b"keep").unwrap();
        assert!(reusable_blob(&blocker.join("blob"), 0, &sha256(b"")).is_err());
        assert_eq!(fs::read(blocker).unwrap(), b"keep");
    }
}
