use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::inspect::MAX_CLI_INPUT_BYTES;

/// Asynchronously evaluates a decision request against a remote opendecision server.
pub async fn cmd_evaluate_async(
    file: PathBuf,
    server: String,
    api_key: Option<String>,
    pretty: bool,
) -> Result<()> {
    let raw = if file.as_os_str() == "-" {
        tokio::task::spawn_blocking(|| {
            use std::io::Read;
            let mut buffer = String::new();
            std::io::stdin()
                .take(MAX_CLI_INPUT_BYTES)
                .read_to_string(&mut buffer)
                .context("read request from stdin")?;
            Ok::<_, anyhow::Error>(buffer)
        })
        .await
        .context("stdin read task")??
    } else {
        let meta = tokio::fs::metadata(&file)
            .await
            .with_context(|| format!("stat {}", file.display()))?;
        if meta.len() > MAX_CLI_INPUT_BYTES {
            anyhow::bail!(
                "file {} exceeds maximum allowed size ({} bytes, limit {} bytes)",
                file.display(),
                meta.len(),
                MAX_CLI_INPUT_BYTES
            );
        }
        tokio::fs::read_to_string(&file)
            .await
            .with_context(|| format!("read {}", file.display()))?
    };

    let url = format!("{}/v1/systemone", server.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none()) // Prevent leaking Authorization header across redirects
        .build()?;
    let mut req_builder = client
        .post(&url)
        .header("content-type", "application/json")
        .body(raw);

    let resolved_key = api_key
        .or_else(|| std::env::var("OPENDECISION_API_KEY").ok())
        .or_else(|| std::env::var("TYPESAFE_API_KEY").ok())
        .filter(|s| !s.is_empty());
    if let Some(key) = resolved_key {
        req_builder = req_builder.header("authorization", format!("Bearer {key}"));
    }

    let resp = req_builder
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    let status = resp.status();
    let body = resp.text().await.context("read response body")?;
    if pretty {
        let v: serde_json::Value = serde_json::from_str(&body)
            .unwrap_or_else(|_| serde_json::Value::String(format!("(invalid JSON: {body})")));
        println!("HTTP {status}");
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!("HTTP {status}");
        println!("{body}");
    }
    if !status.is_success() {
        std::process::exit(1);
    }
    Ok(())
}

/// Synchronous entrypoint for the `evaluate` CLI subcommand, executing within a fresh Tokio runtime.
pub fn cmd_evaluate(
    file: PathBuf,
    server: String,
    api_key: Option<String>,
    pretty: bool,
) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(cmd_evaluate_async(file, server, api_key, pretty))
}
