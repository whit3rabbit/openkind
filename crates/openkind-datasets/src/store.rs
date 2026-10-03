//! Verified dataset installation: resumable pinned downloads, digest checks,
//! content-addressed blobs, locked installs, and revision pinning.
//!
//! The download machinery mirrors `openkind-model-store` (byte-range resume
//! with strict `Content-Range` validation, partial files guarded against
//! symlink following via `O_NOFOLLOW` on Unix and a `symlink_metadata`
//! pre-check elsewhere, size caps, streaming SHA-256, content-addressed blobs
//! hard-linked into staged installs) adapted for Hugging Face dataset
//! repositories and bearer-token authentication. Nothing here runs during
//! builds or tests; only the explicit `dataset pull` and `dataset pin`
//! commands drive the network.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use futures::StreamExt;
use reqwest::header::{CONTENT_RANGE, LINK, RANGE};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::auth::{resolve_token, TokenSource};
use crate::definitions::{definition_splits, find_definition, template};
use crate::registry::{
    valid_name, valid_relative_path, valid_sha256, DatasetEntry, DatasetFile, DatasetRegistry,
    MAX_DATASET_BYTES,
};
use crate::{Error, Result};

const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TREE_PAGES: usize = 1_000;
const HF_ENDPOINT: &str = "https://huggingface.co";

pub struct DatasetStore {
    root: PathBuf,
    endpoint: Cow<'static, str>,
    registry_bytes: Cow<'static, str>,
    token: Option<String>,
    token_source: TokenSource,
    client: reqwest::Client,
}

impl std::fmt::Debug for DatasetStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DatasetStore")
            .field("root", &self.root)
            .field("endpoint", &self.endpoint)
            .field("token_source", &self.token_source)
            .field("has_token", &self.token.is_some())
            .finish_non_exhaustive()
    }
}

/// A verified local dataset installation. The shared file lock protects it
/// from `rm` for as long as the caller holds this value.
#[derive(Debug)]
pub struct InstalledDataset {
    pub entry: DatasetEntry,
    /// Root of the installed files; shard paths are relative to this.
    pub root: PathBuf,
    _lock: File,
}

impl InstalledDataset {
    pub fn file_path(&self, file: &DatasetFile) -> PathBuf {
        self.root.join(&file.path)
    }
}

pub fn default_datasets_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("OPENKIND_DATASETS_DIR") {
        if path.is_empty() {
            return Err(Error::Invalid("OPENKIND_DATASETS_DIR is empty".into()));
        }
        return Ok(PathBuf::from(path));
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME").ok_or_else(|| {
            Error::Invalid("HOME is unavailable; set OPENKIND_DATASETS_DIR".into())
        })?;
        Ok(PathBuf::from(home).join("Library/Application Support/openkind/datasets"))
    }
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("APPDATA")
            .or_else(|| {
                std::env::var_os("USERPROFILE")
                    .map(|p| PathBuf::from(p).join("AppData/Roaming").into_os_string())
            })
            .ok_or_else(|| {
                Error::Invalid("APPDATA is unavailable; set OPENKIND_DATASETS_DIR".into())
            })?;
        Ok(PathBuf::from(base).join("openkind/datasets"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let base = match std::env::var_os("XDG_DATA_HOME") {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(std::env::var_os("HOME").ok_or_else(|| {
                Error::Invalid("HOME is unavailable; set OPENKIND_DATASETS_DIR".into())
            })?)
            .join(".local/share"),
        };
        Ok(base.join("openkind/datasets"))
    }
}

impl DatasetStore {
    pub fn new(root: PathBuf) -> Result<Self> {
        let (token, token_source) = resolve_token()?;
        Self::build(
            root,
            Cow::Borrowed(HF_ENDPOINT),
            Cow::Borrowed(crate::DATASETS_JSON),
            token,
            token_source,
        )
    }

    fn build(
        root: PathBuf,
        endpoint: Cow<'static, str>,
        registry_bytes: Cow<'static, str>,
        token: Option<String>,
        token_source: TokenSource,
    ) -> Result<Self> {
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
            endpoint,
            registry_bytes,
            token,
            token_source,
            client,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn token_source(&self) -> TokenSource {
        self.token_source
    }

    /// The registry compiled into this build.
    pub fn registry(&self) -> Result<DatasetRegistry> {
        DatasetRegistry::parse(self.registry_bytes.as_bytes())
    }

    /// Installed datasets with locally verified identity records.
    pub fn list(&self) -> Result<Vec<DatasetEntry>> {
        self.scan_installed(None, true)
    }

    fn scan_installed(
        &self,
        excluded_name: Option<&str>,
        skip_invalid: bool,
    ) -> Result<Vec<DatasetEntry>> {
        let mut datasets = Vec::new();
        let dir = self.root.join("datasets");
        if !dir.exists() {
            return Ok(datasets);
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
            if excluded_name == Some(name.as_str()) {
                continue;
            }
            match self.read_installed_entry(&name) {
                Ok(entry) => datasets.push(entry),
                Err(_) if skip_invalid => continue,
                Err(error) => return Err(error),
            }
        }
        datasets.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(datasets)
    }

    /// Open an installed dataset without re-hashing its files. Use
    /// [`DatasetStore::verify`] to re-check digests on demand.
    pub fn installed(&self, name: &str) -> Result<InstalledDataset> {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid dataset name".into()));
        }
        let lock = self.dataset_lock(name, false)?;
        let entry = self.read_installed_entry(name)?;
        let root = self.dataset_dir(name).join("files");
        Ok(InstalledDataset {
            entry,
            root,
            _lock: lock,
        })
    }

    /// Re-verify every installed file digest against its entry.
    pub fn verify(&self, name: &str) -> Result<DatasetEntry> {
        let installed = self.installed(name)?;
        for file in &installed.entry.files {
            verify_file(&installed.file_path(file), file.size, &file.sha256)?;
        }
        Ok(installed.entry)
    }

    pub fn rm(&self, name: &str) -> Result<()> {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid dataset name".into()));
        }
        let _global = self.global_lock()?;
        let _dataset = self.dataset_lock(name, true)?;
        let entry = match self.read_installed_entry(name) {
            Ok(entry) => Some(entry),
            Err(Error::NotInstalled(_)) => return Err(Error::NotInstalled(name.to_owned())),
            // Removing a damaged install is the recovery path. Retain blobs
            // because its manifest can no longer establish their ownership.
            Err(_) => None,
        };
        let used = self.scan_installed(Some(name), false).ok().map(|datasets| {
            datasets
                .into_iter()
                .flat_map(|installed| installed.files.into_iter().map(|file| file.sha256))
                .collect::<HashSet<_>>()
        });
        fs::remove_dir_all(self.dataset_dir(name))?;
        if let (Some(entry), Some(used)) = (entry, used) {
            for file in entry.files {
                if !used.contains(&file.sha256) {
                    let blob = self.blob_path(&file.sha256);
                    if blob.exists() {
                        fs::remove_file(blob)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Pull a pinned dataset from the compiled-in registry. The callback
    /// receives the shard path, present bytes, and expected bytes.
    pub async fn pull<F>(&self, name: &str, mut progress: F) -> Result<DatasetEntry>
    where
        F: FnMut(&str, u64, u64),
    {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid dataset name".into()));
        }
        let entry = self
            .registry()?
            .find(name)
            .cloned()
            .ok_or_else(|| Error::NotCurated(name.into()))?;
        if entry.files.is_empty() {
            return Err(Error::Invalid(format!(
                "dataset {name} is defined but not pinned yet; run `dataset pin {name}` first"
            )));
        }
        let _global = self.global_lock()?;
        self.install(&entry, &mut progress).await
    }

    /// Resolve the current upstream revisions for a curated dataset
    /// definition, download and hash the referenced shards, install the
    /// result, and return the complete registry entry for review. The
    /// returned entry still needs to be committed to
    /// `registry/v1/datasets.json` before `dataset pull` will use it.
    pub async fn pin<F>(&self, name: &str, mut progress: F) -> Result<DatasetEntry>
    where
        F: FnMut(&str, u64, u64),
    {
        let definition = find_definition(name).ok_or_else(|| Error::NotCurated(name.into()))?;
        let _global = self.global_lock()?;
        if self.dataset_dir(name).exists() {
            return Err(Error::Invalid(format!(
                "dataset {name} is already installed; `dataset rm {name}` before re-pinning"
            )));
        }
        let hf_revision = self.resolve_revision(definition.hf_repo, "main").await?;
        let convert_revision = self
            .resolve_revision(definition.hf_repo, "refs%2Fconvert%2Fparquet")
            .await?;
        let used_splits: Vec<&str> = definition.splits.iter().map(|(_, split)| *split).collect();
        let mut files = Vec::new();
        let mut total_bytes = 0_u64;
        for (path, size) in self
            .list_tree(definition.hf_repo, &convert_revision)
            .await?
        {
            let mut segments = path.splitn(3, '/');
            let config = segments.next().unwrap_or_default();
            let split = segments.next().unwrap_or_default();
            if config.is_empty()
                || split.is_empty()
                || !path.ends_with(".parquet")
                || !definition.configs.contains(&config)
                || !used_splits.contains(&split)
            {
                continue;
            }
            total_bytes = total_bytes
                .checked_add(size)
                .filter(|total| *total <= MAX_DATASET_BYTES)
                .ok_or_else(|| Error::Invalid("dataset size exceeds 2 GiB".into()))?;
            if size == 0 {
                return Err(Error::Invalid(format!("dataset shard {path} has no size")));
            }
            let url = self.convert_url(definition.hf_repo, &convert_revision, &path);
            let sha256 = self
                .fetch_to_blob(&url, &path, size, None, &mut progress)
                .await?;
            files.push(DatasetFile { path, size, sha256 });
        }
        if files.is_empty() {
            return Err(Error::Invalid(format!(
                "no converted parquet shards found for {name} (configs {}, splits {})",
                definition.configs.join(", "),
                used_splits.join(", ")
            )));
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let entry = DatasetEntry {
            name: definition.name.to_owned(),
            description: definition.description.to_owned(),
            hf_repo: definition.hf_repo.to_owned(),
            hf_revision,
            convert_revision,
            files,
            splits: definition_splits(definition),
            configs: definition
                .configs
                .iter()
                .map(|config| (*config).to_owned())
                .collect(),
            license: definition.license.to_owned(),
            gated: definition.gated,
            access_note: definition.access_note.to_owned(),
            task_family: definition.task_family.to_owned(),
            primitives: definition
                .primitives
                .iter()
                .map(|p| (*p).to_owned())
                .collect(),
            template: template(),
        };
        entry.validate()?;
        self.install(&entry, &mut progress).await?;
        Ok(entry)
    }

    async fn install<F>(&self, entry: &DatasetEntry, progress: &mut F) -> Result<DatasetEntry>
    where
        F: FnMut(&str, u64, u64),
    {
        let _dataset = self.dataset_lock(&entry.name, true)?;
        let entry_bytes = serde_json::to_vec(entry)?;
        if self.dataset_dir(&entry.name).exists() {
            let installed = self.read_installed_entry(&entry.name)?;
            if sha256(&fs::read(self.dataset_dir(&entry.name).join("entry.json"))?)
                == sha256(&entry_bytes)
            {
                for file in &installed.files {
                    verify_file(
                        &self.dataset_dir(&entry.name).join("files").join(&file.path),
                        file.size,
                        &file.sha256,
                    )?;
                }
                return Ok(installed);
            }
            return Err(Error::Invalid(format!(
                "installed dataset {} differs from the registry entry; `dataset rm {}` first",
                entry.name, entry.name
            )));
        }
        fs::create_dir_all(self.root.join("blobs/sha256"))?;
        for file in &entry.files {
            let blob = self.blob_path(&file.sha256);
            if blob.exists() {
                progress(&file.path, file.size, file.size);
                verify_file(&blob, file.size, &file.sha256)?;
            } else {
                let url = self.convert_url(&entry.hf_repo, &entry.convert_revision, &file.path);
                self.fetch_to_blob(&url, &file.path, file.size, Some(&file.sha256), progress)
                    .await?;
            }
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::Invalid("system clock before Unix epoch".into()))?
            .as_nanos();
        let stage = self.root.join("datasets").join(format!(
            ".stage-{}-{}-{nonce}",
            entry.name,
            std::process::id()
        ));
        fs::create_dir_all(&stage)?;
        let install = (|| -> Result<()> {
            for file in &entry.files {
                let target = stage.join("files").join(&file.path);
                fs::create_dir_all(target.parent().expect("shard has a parent"))?;
                fs::hard_link(self.blob_path(&file.sha256), target)?;
            }
            let entry_path = stage.join("entry.json");
            let mut entry_file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(entry_path)?;
            entry_file.write_all(&entry_bytes)?;
            entry_file.sync_all()?;
            sync_directory(&stage)?;
            fs::rename(&stage, self.dataset_dir(&entry.name))?;
            sync_directory(&self.root.join("datasets"))?;
            Ok(())
        })();
        if install.is_err() {
            let _ = fs::remove_dir_all(&stage);
        }
        install?;
        Ok(entry.clone())
    }

    fn read_installed_entry(&self, name: &str) -> Result<DatasetEntry> {
        if !valid_name(name) {
            return Err(Error::Invalid("invalid dataset name".into()));
        }
        let path = self.dataset_dir(name).join("entry.json");
        let metadata = path
            .metadata()
            .map_err(|_| Error::NotInstalled(name.to_owned()))?;
        if metadata.len() > 4 * 1024 * 1024 {
            return Err(Error::Invalid("installed entry exceeds 4 MiB".into()));
        }
        let entry: DatasetEntry = serde_json::from_slice(&fs::read(path)?)?;
        entry.validate()?;
        if entry.name != name {
            return Err(Error::Invalid("installed entry name mismatch".into()));
        }
        Ok(entry)
    }

    fn dataset_dir(&self, name: &str) -> PathBuf {
        self.root.join("datasets").join(name)
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
            .map_err(|_| Error::Busy("dataset store".into()))?;
        Ok(file)
    }

    fn dataset_lock(&self, name: &str, exclusive: bool) -> Result<File> {
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

    fn convert_url(&self, repo: &str, convert_revision: &str, path: &str) -> String {
        format!(
            "{endpoint}/datasets/{repo}/resolve/{convert_revision}/{path}",
            endpoint = self.endpoint
        )
    }

    fn authed(&self, url: &str) -> reqwest::RequestBuilder {
        let request = self.client.get(url);
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn resolve_revision(&self, repo: &str, reference: &str) -> Result<String> {
        let url = format!(
            "{endpoint}/api/datasets/{repo}/revision/{reference}",
            endpoint = self.endpoint
        );
        let bytes = self.fetch_small(&url).await?;
        #[derive(serde::Deserialize)]
        struct Revision {
            sha: String,
        }
        let revision: Revision = serde_json::from_slice(&bytes)?;
        if !crate::registry::valid_revision(&revision.sha) {
            return Err(Error::Invalid(format!(
                "upstream returned a non-commit revision for {repo}"
            )));
        }
        Ok(revision.sha)
    }

    async fn list_tree(&self, repo: &str, revision: &str) -> Result<Vec<(String, u64)>> {
        let mut page_url = reqwest::Url::parse(&format!(
            "{endpoint}/api/datasets/{repo}/tree/{revision}",
            endpoint = self.endpoint
        ))
        .map_err(|error| Error::Invalid(format!("dataset tree URL: {error}")))?;
        page_url
            .query_pairs_mut()
            .append_pair("recursive", "true")
            .append_pair("limit", "1000");
        let trusted_origin = page_url.clone();
        let mut seen_pages = HashSet::new();
        let mut total_metadata_bytes = 0_u64;
        let mut page_count = 0;
        let mut files = Vec::new();
        #[derive(serde::Deserialize)]
        struct TreeEntry {
            #[serde(rename = "type")]
            kind: String,
            path: String,
            #[serde(default)]
            size: u64,
        }
        loop {
            page_count += 1;
            if page_count > MAX_TREE_PAGES {
                return Err(Error::Invalid("dataset tree has too many pages".into()));
            }
            let page_key = page_url.as_str().to_owned();
            if !seen_pages.insert(page_key) {
                return Err(Error::Invalid(
                    "dataset tree pagination repeated a page".into(),
                ));
            }
            let response = self
                .authed(page_url.as_str())
                .send()
                .await
                .map_err(denied)?
                .error_for_status()
                .map_err(denied)?;
            let next_link: Option<String> = response
                .headers()
                .get(LINK)
                .map(|header| -> Result<Option<String>> {
                    let header = header.to_str().map_err(|_| {
                        Error::Invalid("dataset tree returned an invalid Link header".into())
                    })?;
                    next_page_link(header).map(|opt| opt.map(str::to_owned))
                })
                .transpose()?
                .flatten();
            let remaining = MAX_METADATA_BYTES - total_metadata_bytes;
            if response
                .content_length()
                .is_some_and(|size| size > remaining)
            {
                return Err(Error::Invalid("remote tree metadata exceeds 64 MiB".into()));
            }
            let mut bytes = Vec::new();
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if chunk.len() as u64 > remaining - bytes.len() as u64 {
                    return Err(Error::Invalid("remote tree metadata exceeds 64 MiB".into()));
                }
                bytes.extend_from_slice(&chunk);
            }
            total_metadata_bytes += bytes.len() as u64;
            let tree: Vec<TreeEntry> = serde_json::from_slice(&bytes)?;
            files.extend(
                tree.into_iter()
                    .filter(|entry| entry.kind == "file")
                    .map(|entry| (entry.path, entry.size)),
            );

            let Some(next_link) = next_link else {
                break;
            };
            let next_url = page_url.join(&next_link).map_err(|error| {
                Error::Invalid(format!("invalid dataset tree page URL: {error}"))
            })?;
            if !same_origin(&trusted_origin, &next_url) {
                return Err(Error::Invalid(
                    "dataset tree pagination escaped the configured endpoint".into(),
                ));
            }
            page_url = next_url;
        }
        Ok(files)
    }

    async fn fetch_small(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.authed(url).send().await.map_err(denied)?;
        let response = response.error_for_status().map_err(denied)?;
        if response
            .content_length()
            .is_some_and(|size| size > MAX_METADATA_BYTES)
        {
            return Err(Error::Invalid("remote metadata exceeds 64 MiB".into()));
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if chunk.len() > MAX_METADATA_BYTES as usize - bytes.len() {
                return Err(Error::Invalid("remote metadata exceeds 64 MiB".into()));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    /// Download `url` into the content-addressed blob store, resuming partial
    /// transfers and verifying size and (when known) digest. Returns the
    /// observed SHA-256 of the stored bytes.
    async fn fetch_to_blob<F>(
        &self,
        url: &str,
        label: &str,
        size: u64,
        expected_sha: Option<&str>,
        progress: &mut F,
    ) -> Result<String>
    where
        F: FnMut(&str, u64, u64),
    {
        let blob = match expected_sha {
            Some(sha) => self.blob_path(sha),
            None => self
                .root
                .join("blobs")
                .join(format!(".pin-{}", sha256(url.as_bytes()))),
        };
        if let Some(parent) = blob.parent() {
            fs::create_dir_all(parent)?;
        }
        if expected_sha.is_none() && blob.exists() {
            let metadata = fs::symlink_metadata(&blob)?;
            if !metadata.file_type().is_file() {
                return Err(Error::Invalid(
                    "staged dataset shard is not a regular file".into(),
                ));
            }
            if metadata.len() == size {
                let observed = sha256_file(&blob)?;
                let content_addressed = self.blob_path(&observed);
                promote_blob(&blob, &content_addressed, size, &observed)?;
                return Ok(observed);
            }
            fs::remove_file(&blob)?;
        }
        let part = blob.with_extension("part");
        let mut present = match part.symlink_metadata() {
            Ok(metadata) if metadata.file_type().is_file() => metadata.len(),
            Ok(_) => {
                return Err(Error::Invalid(
                    "partial dataset shard is not a regular file".into(),
                ))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        if present == size {
            progress(label, size, size);
            match sha256_file(&part) {
                Ok(observed) => {
                    let matches_expected = expected_sha.is_none_or(|expected| observed == expected);
                    if matches_expected {
                        let content_addressed = self.blob_path(&observed);
                        if let Some(parent) = content_addressed.parent() {
                            fs::create_dir_all(parent)?;
                        }
                        promote_blob(&part, &content_addressed, size, &observed)?;
                        return Ok(observed);
                    }
                    fs::remove_file(&part)?;
                    present = 0;
                    progress(label, present, size);
                }
                Err(error) => return Err(error),
            }
        }
        if present > size {
            fs::remove_file(&part)?;
            present = 0;
        }
        progress(label, present, size);
        let mut request = self.authed(url);
        if present > 0 {
            request = request.header(RANGE, format!("bytes={present}-"));
        }
        let response = request.send().await.map_err(denied)?;
        let response = response.error_for_status().map_err(denied)?;
        let status = response.status();
        let response_end = if present > 0 && status == reqwest::StatusCode::OK {
            present = 0;
            fs::remove_file(&part)?;
            size
        } else if present > 0 {
            let range = response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|header| header.to_str().ok())
                .unwrap_or("");
            if status != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(Error::Invalid(
                    "resumed response has wrong byte range".into(),
                ));
            }
            resumed_response_end(range, present, size)
                .ok_or_else(|| Error::Invalid("resumed response has wrong byte range".into()))?
        } else if status != reqwest::StatusCode::OK {
            return Err(Error::Invalid(
                "dataset shard response is not a full file".into(),
            ));
        } else {
            size
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
                "partial dataset shard changed before append".into(),
            ));
        }
        progress(label, present, size);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            present = present
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| Error::Invalid("dataset shard size overflow".into()))?;
            if present > response_end {
                return Err(Error::Invalid(format!(
                    "dataset shard {label} exceeds declared response range"
                )));
            }
            file.write_all(&chunk).await?;
            progress(label, present, size);
        }
        file.sync_all().await?;
        drop(file);
        // An incomplete transfer has no full-file digest yet; retain it for the next range request.
        if present < size {
            return Err(Error::Invalid(format!(
                "dataset shard {label} is incomplete: received {present} of {size} bytes"
            )));
        }
        let observed = sha256_file(&part)?;
        if let Some(expected) = expected_sha {
            if observed != expected {
                let _ = fs::remove_file(&part);
                return Err(Error::DigestMismatch(label.to_owned()));
            }
        }
        let content_addressed = self.blob_path(&observed);
        if let Some(parent) = content_addressed.parent() {
            fs::create_dir_all(parent)?;
        }
        promote_blob(&part, &content_addressed, size, &observed)?;
        Ok(observed)
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

fn next_page_link(header: &str) -> Result<Option<&str>> {
    for link in header.split(">,") {
        let link = link.trim();
        let Some((target, attributes)) = link.split_once('>') else {
            continue;
        };
        let is_next = attributes.split(';').any(|attribute| {
            attribute
                .trim()
                .split_once('=')
                .is_some_and(|(key, value)| {
                    key.trim() == "rel"
                        && value
                            .trim()
                            .trim_matches('"')
                            .split_ascii_whitespace()
                            .any(|relation| relation == "next")
                })
        });
        if is_next {
            return target
                .strip_prefix('<')
                .filter(|target| !target.is_empty())
                .map(Some)
                .ok_or_else(|| {
                    Error::Invalid("dataset tree returned an invalid next link".into())
                });
        }
    }
    Ok(None)
}

fn same_origin(left: &reqwest::Url, right: &reqwest::Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn promote_blob(source: &Path, destination: &Path, size: u64, digest: &str) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    if destination.exists() {
        verify_file(destination, size, digest)?;
        fs::remove_file(source)?;
    } else {
        fs::rename(source, destination)?;
    }
    Ok(())
}

/// Attach an actionable hint to authentication failures against gated repos.
fn denied(error: reqwest::Error) -> Error {
    if error
        .status()
        .is_some_and(|status| status.as_u16() == 401 || status.as_u16() == 403)
    {
        return Error::Invalid(format!(
            "Hugging Face denied the request ({error}); for gated datasets run `hf auth login`, accept the repository terms, and retry"
        ));
    }
    error.into()
}

fn resumed_response_end(range: &str, present: u64, size: u64) -> Option<u64> {
    let (interval, total) = range.strip_prefix("bytes ")?.split_once('/')?;
    let (start, end) = interval.split_once('-')?;
    let start = start.parse::<u64>().ok()?;
    let end = end.parse::<u64>().ok()?;
    let total = total.parse::<u64>().ok()?;
    (start == present && end >= start && end < size && total == size).then(|| end + 1)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
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
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_file(path: &Path, size: u64, sha: &str) -> Result<()> {
    if !valid_sha256(sha)
        || !valid_relative_path(path.file_name().and_then(|s| s.to_str()).unwrap_or(""))
    {
        return Err(Error::Invalid("invalid dataset shard identity".into()));
    }
    let metadata = path.symlink_metadata()?;
    if !metadata.file_type().is_file() || metadata.len() != size {
        return Err(Error::DigestMismatch(path.display().to_string()));
    }
    if sha256_file(path)? != sha {
        return Err(Error::DigestMismatch(path.display().to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::Path as AxumPath;
    use axum::http::{header, HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::get;

    fn test_registry(sha: &str) -> String {
        format!(
            r#"{{
                "schema": "openkind-dataset-registry/v1",
                "datasets": [{{
                    "name": "demo",
                    "description": "demo dataset",
                    "hf_repo": "owner/data",
                    "hf_revision": "{}",
                    "convert_revision": "{}",
                    "files": [
                        {{"path": "default/test/0000.parquet", "size": 8, "sha256": "{sha}"}},
                        {{"path": "default/train/0000.parquet", "size": 8, "sha256": "{sha}"}}
                    ],
                    "splits": {{"dev": "train", "eval": "test"}},
                    "configs": ["default"],
                    "license": "mit",
                    "gated": "none",
                    "task_family": "classification",
                    "primitives": ["choice"],
                    "template": {{"id": "t", "version": 1, "source": "src"}}
                }}]
            }}"#,
            "a".repeat(40),
            "b".repeat(40),
        )
    }

    const TEST_BYTES: &[u8; 8] = b"12345678";

    fn test_store(root: &Path, endpoint: &str, sha: &str) -> DatasetStore {
        DatasetStore::build(
            root.to_owned(),
            Cow::Owned(endpoint.to_owned()),
            Cow::Owned(test_registry(sha)),
            None,
            TokenSource::Anonymous,
        )
        .unwrap()
    }

    /// Serve the test shard; `resume` makes Range requests answer 206 with the
    /// remainder, exercising the resume path.
    fn file_routes(resume: bool) -> axum::Router {
        async fn shard(
            AxumPath(_rest): AxumPath<String>,
            headers: HeaderMap,
            resume: bool,
        ) -> axum::response::Response {
            let bytes = TEST_BYTES.to_vec();
            let range = headers
                .get(header::RANGE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            if !resume || range.is_none() {
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/octet-stream")],
                    bytes,
                )
                    .into_response();
            }
            let present: u64 = range
                .unwrap()
                .trim_start_matches("bytes=")
                .trim_end_matches('-')
                .parse()
                .unwrap_or(0);
            let partial = TEST_BYTES[present as usize..].to_vec();
            (
                StatusCode::PARTIAL_CONTENT,
                [
                    (header::CONTENT_TYPE, "application/octet-stream".to_owned()),
                    (header::CONTENT_RANGE, format!("bytes {present}-7/8")),
                ],
                partial,
            )
                .into_response()
        }
        axum::Router::new().route(
            "/datasets/owner/data/resolve/{*rest}",
            get(
                move |path: AxumPath<String>, headers: HeaderMap| async move {
                    shard(path, headers, resume).await
                },
            ),
        )
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    async fn spawn_server(routes: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, routes).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[test]
    fn pull_installs_and_reverifies() {
        let dir = tempfile::tempdir().unwrap();
        let rt = runtime();
        let store = rt.block_on(async {
            let endpoint = spawn_server(file_routes(true)).await;
            test_store(dir.path(), &endpoint, &sha256(TEST_BYTES))
        });
        rt.block_on(store.pull("demo", |_, _, _| {})).unwrap();
        assert!(store
            .dataset_dir("demo")
            .join("files/default/test/0000.parquet")
            .exists());
        assert!(store.dataset_dir("demo").join("entry.json").exists());
        store.verify("demo").unwrap();
        {
            let installed = store.installed("demo").unwrap();
            assert_eq!(installed.entry.name, "demo");
        }
        // A second pull is idempotent and re-verifies.
        rt.block_on(store.pull("demo", |_, _, _| {})).unwrap();
    }

    #[test]
    fn pull_resumes_from_a_partial_blob() {
        let dir = tempfile::tempdir().unwrap();
        let sha = sha256(TEST_BYTES);
        let rt = runtime();
        let store = rt.block_on(async {
            let endpoint = spawn_server(file_routes(true)).await;
            test_store(dir.path(), &endpoint, &sha)
        });
        let part = store.blob_path(&sha).with_extension("part");
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, &TEST_BYTES[..4]).unwrap();
        rt.block_on(store.pull("demo", |_, _, _| {})).unwrap();
        assert!(store.blob_path(&sha).exists());
        assert!(!part.exists());
    }

    #[test]
    fn pull_rejects_digest_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let rt = runtime();
        let store = rt.block_on(async {
            let endpoint = spawn_server(file_routes(false)).await;
            test_store(dir.path(), &endpoint, &"0".repeat(64))
        });
        let result = rt.block_on(store.pull("demo", |_, _, _| {}));
        assert!(matches!(result, Err(Error::DigestMismatch(_))));
        assert!(!store.dataset_dir("demo").exists());
    }

    #[test]
    fn rm_drops_unshared_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let rt = runtime();
        let store = rt.block_on(async {
            let endpoint = spawn_server(file_routes(false)).await;
            test_store(dir.path(), &endpoint, &sha256(TEST_BYTES))
        });
        rt.block_on(store.pull("demo", |_, _, _| {})).unwrap();
        let blob = store.blob_path(&sha256(TEST_BYTES));
        assert!(blob.exists());
        store.rm("demo").unwrap();
        assert!(!blob.exists());
        assert!(!store.dataset_dir("demo").exists());
        assert!(matches!(
            store.installed("demo"),
            Err(Error::NotInstalled(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_partial_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = {
            let endpoint = {
                let rt = runtime();
                rt.block_on(async { spawn_server(file_routes(false)).await })
            };
            test_store(dir.path(), &endpoint, &sha256(TEST_BYTES))
        };
        let blob = store.blob_path(&sha256(TEST_BYTES));
        let part = blob.with_extension("part");
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        let target = dir.path().join("target.txt");
        std::fs::write(&target, b"x").unwrap();
        std::os::unix::fs::symlink(&target, &part).unwrap();
        let result = runtime().block_on(store.pull("demo", |_, _, _| {}));
        assert!(result.is_err());
        assert!(!store.dataset_dir("demo").exists());
    }

    #[test]
    fn registry_entries_and_definitions_stay_in_sync() {
        let registry = DatasetRegistry::embedded().unwrap();
        for definition in crate::definitions::DATASET_DEFINITIONS {
            let entry = registry.find(definition.name).unwrap_or_else(|| {
                panic!("definition {} missing from the registry", definition.name)
            });
            assert_eq!(entry.hf_repo, definition.hf_repo);
            assert_eq!(entry.template, template());
            assert!(
                !entry.files.is_empty(),
                "{} has no pinned files",
                definition.name
            );
        }
        for entry in &registry.datasets {
            assert!(
                crate::definitions::find_definition(&entry.name).is_some(),
                "registry entry {} has no compiled-in definition",
                entry.name
            );
        }
    }

    #[test]
    fn denied_upstream_repositories_name_the_hf_login_remedy() {
        // A gated-repo 403 must surface the `hf auth login` hint instead of
        // a bare HTTP error; other statuses keep their original error.
        let dir = tempfile::tempdir().unwrap();
        let rt = runtime();
        rt.block_on(async {
            let routes = axum::Router::new().route(
                "/datasets/owner/data/resolve/{*rest}",
                get(|| async { (StatusCode::FORBIDDEN, "gated").into_response() }),
            );
            let endpoint = spawn_server(routes).await;
            let store = test_store(dir.path(), &endpoint, &sha256(TEST_BYTES));
            let error = store
                .pull("demo", |_, _, _| {})
                .await
                .expect_err("a denied dataset must fail");
            let message = error.to_string();
            assert!(
                message.contains("hf auth login") || message.contains("denied the request"),
                "unexpected error: {message}"
            );
        });
    }

    #[test]
    fn verify_detects_tampered_truncated_and_swapped_shards() {
        let dir = tempfile::tempdir().unwrap();
        let sha = sha256(TEST_BYTES);
        let rt = runtime();
        let store = rt.block_on(async {
            let endpoint = spawn_server(file_routes(true)).await;
            test_store(dir.path(), &endpoint, &sha)
        });
        rt.block_on(store.pull("demo", |_, _, _| {})).unwrap();
        store.verify("demo").expect("intact installs verify");

        // A flipped byte fails the digest.
        let shard = store
            .dataset_dir("demo")
            .join("files/default/test/0000.parquet");
        std::fs::write(&shard, b"12345670").unwrap();
        assert!(matches!(
            store.verify("demo"),
            Err(Error::DigestMismatch(_))
        ));

        // A truncated shard fails on size before the digest.
        std::fs::write(&shard, b"1234").unwrap();
        assert!(matches!(
            store.verify("demo"),
            Err(Error::DigestMismatch(_))
        ));

        // Restore and corrupt the other split to cover the multi-file walk.
        std::fs::write(&shard, TEST_BYTES).unwrap();
        let other = store
            .dataset_dir("demo")
            .join("files/default/train/0000.parquet");
        std::fs::write(&other, b"87654321").unwrap();
        assert!(matches!(
            store.verify("demo"),
            Err(Error::DigestMismatch(_))
        ));
    }

    #[test]
    fn resume_ranges_are_enforced_by_the_header_arithmetic() {
        // `resumed_response_end` is the guard behind the resume path; its
        // full matrix lives in `resumed_response_end_accepts_only_…`.
        assert_eq!(resumed_response_end("bytes 0-7/8", 0, 8), Some(8));
        assert_eq!(resumed_response_end("bytes 0-7/9", 0, 8), None);
    }

    #[test]
    fn resumed_response_end_accepts_only_consistent_ranges() {
        // A valid range starting exactly at the resume offset.
        assert_eq!(resumed_response_end("bytes 4-9/10", 4, 10), Some(10));
        // Wrong prefix, missing separators, and non-numeric fields.
        for range in [
            "bytes4-9/10",
            "",
            "bytes 4_9/10",
            "bytes 4/10",
            "bytes -9/10",
            "bytes 4-x/10",
            "bytes 4-9/x",
        ] {
            assert_eq!(resumed_response_end(range, 4, 10), None, "{range}");
        }
        // Start must equal the bytes already present.
        assert_eq!(resumed_response_end("bytes 3-9/10", 4, 10), None);
        // End may not precede start.
        assert_eq!(resumed_response_end("bytes 4-3/10", 4, 10), None);
        // End is exclusive of the total size.
        assert_eq!(resumed_response_end("bytes 4-10/10", 4, 10), None);
        // The declared total must match the artifact size.
        assert_eq!(resumed_response_end("bytes 4-9/999", 4, 10), None);
    }
}
