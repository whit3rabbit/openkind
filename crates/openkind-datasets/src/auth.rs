//! Hugging Face token resolution for gated-repository downloads.
//!
//! Resolution order mirrors huggingface_hub: the `HF_TOKEN` environment
//! variable, then `HF_TOKEN_PATH`, then the token file written by
//! `hf auth login` (under `HF_HOME`, default `~/.cache/huggingface/token`).
//! Anonymous access is used when no token is available; the token value
//! itself is never logged.

use std::path::PathBuf;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenSource {
    EnvHfToken,
    EnvHfTokenPath,
    DefaultTokenFile,
    Anonymous,
}

impl TokenSource {
    pub fn describe(self) -> &'static str {
        match self {
            Self::EnvHfToken => "HF_TOKEN environment variable",
            Self::EnvHfTokenPath => "HF_TOKEN_PATH token file",
            Self::DefaultTokenFile => "huggingface_hub token file (hf auth login)",
            Self::Anonymous => "anonymous access (no token found)",
        }
    }
}

/// Resolve the Hugging Face token from the environment or the CLI's token
/// file. Returns `None` with [`TokenSource::Anonymous`] when no token is
/// configured.
///
/// # Errors
/// Returns an error when an explicitly configured token source (`HF_TOKEN`
/// set to a non-empty value, or `HF_TOKEN_PATH`) cannot be read, so a broken
/// half-configuration never silently degrades to anonymous requests against
/// gated repositories.
pub fn resolve_token() -> Result<(Option<String>, TokenSource)> {
    resolve_token_from(&|key: &str| std::env::var_os(key), &default_token_file)
}

fn default_token_file() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = match std::env::var_os("HF_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(home).join(".cache").join("huggingface"),
    };
    Some(base.join("token"))
}

fn resolve_token_from(
    env: &dyn Fn(&str) -> Option<std::ffi::OsString>,
    default_file: &dyn Fn() -> Option<PathBuf>,
) -> Result<(Option<String>, TokenSource)> {
    if let Some(value) = env("HF_TOKEN") {
        let value = value.to_string_lossy().trim().to_owned();
        if !value.is_empty() {
            return Ok((Some(value), TokenSource::EnvHfToken));
        }
    }
    if let Some(path) = env("HF_TOKEN_PATH") {
        if !path.is_empty() {
            let token = read_token_file(PathBuf::from(path))?;
            return Ok((Some(token), TokenSource::EnvHfTokenPath));
        }
    }
    if let Some(path) = default_file() {
        if path.is_file() {
            let token = read_token_file(path)?;
            return Ok((Some(token), TokenSource::DefaultTokenFile));
        }
    }
    Ok((None, TokenSource::Anonymous))
}

fn read_token_file(path: PathBuf) -> Result<String> {
    let bytes = std::fs::read(&path).map_err(|error| {
        Error::Invalid(format!(
            "cannot read the Hugging Face token file at {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() > 4096 {
        return Err(Error::Invalid(
            "the Hugging Face token file is unexpectedly large".into(),
        ));
    }
    let token = String::from_utf8_lossy(&bytes).trim().to_owned();
    if token.is_empty() {
        return Err(Error::Invalid(format!(
            "the Hugging Face token file at {} is empty",
            path.display()
        )));
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<std::ffi::OsString> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        move |key: &str| map.get(key).map(|value| value.into())
    }

    #[test]
    fn prefers_env_token_then_path_then_file() {
        let (token, source) =
            resolve_token_from(&env(&[("HF_TOKEN", "hf_env")]), &|| None).unwrap();
        assert_eq!(token.as_deref(), Some("hf_env"));
        assert_eq!(source, TokenSource::EnvHfToken);

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("token");
        std::fs::write(&file, "hf_file\n").unwrap();
        let (token, source) =
            resolve_token_from(&env(&[("HF_TOKEN_PATH", file.to_str().unwrap())]), &|| None)
                .unwrap();
        assert_eq!(token.as_deref(), Some("hf_file"));
        assert_eq!(source, TokenSource::EnvHfTokenPath);

        let (token, source) = resolve_token_from(&env(&[]), &|| Some(file.clone())).unwrap();
        assert_eq!(token.as_deref(), Some("hf_file"));
        assert_eq!(source, TokenSource::DefaultTokenFile);

        let (token, source) = resolve_token_from(&env(&[]), &|| None).unwrap();
        assert!(token.is_none());
        assert_eq!(source, TokenSource::Anonymous);
    }

    #[test]
    fn broken_explicit_paths_are_errors() {
        let missing = tempfile::tempdir().unwrap().path().join("missing");
        let result = resolve_token_from(
            &env(&[("HF_TOKEN_PATH", missing.to_str().unwrap())]),
            &|| None,
        );
        assert!(result.is_err());
    }
}
