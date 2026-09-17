//! `openpick` — command-line client.
//!
//! Phase 1 subcommands:
//! - `openpick serve`     — short-hand to launch openpickd
//! - `openpick evaluate`  — POST a request file to a running server
//! - `openpick inspect`   — validate a request file against the schema
//! - `openpick version`   — print the API version

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use openpick_core::{validate_request, SystemRequest};

#[derive(Parser, Debug)]
#[command(name = "openpick", about = "openpick — Jev-compatible decision inference CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Validate a request JSON file against the openpick schema.
    Inspect {
        /// Path to the request JSON file.
        file: PathBuf,
    },

    /// Send a request to a running openpick server.
    Evaluate {
        /// Path to the request JSON file.
        file: PathBuf,
        /// Server URL (e.g. http://127.0.0.1:8080).
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        server: String,
        /// Print the response as pretty JSON.
        #[arg(long)]
        pretty: bool,
    },

    /// Print the openpick wire API version.
    Version,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { file } => cmd_inspect(file),
        Commands::Evaluate { file, server, pretty } => cmd_evaluate(file, server, pretty),
        Commands::Version => {
            println!("openpick {}", openpick_core::api_version());
            Ok(())
        }
    }
}

fn cmd_inspect(file: PathBuf) -> Result<()> {
    let raw = std::fs::read_to_string(&file)
        .with_context(|| format!("read {}", file.display()))?;
    let req: SystemRequest =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", file.display()))?;
    validate_request(&req).with_context(|| format!("validate {}", file.display()))?;
    println!(
        "{} ✓ (model={}, {} questions)",
        file.display(),
        req.model,
        req.questions.len()
    );
    Ok(())
}

async fn cmd_evaluate_async(file: PathBuf, server: String, pretty: bool) -> Result<()> {
    let raw = tokio::fs::read_to_string(&file)
        .await
        .with_context(|| format!("read {}", file.display()))?;
    let url = format!("{}/v1/systemone", server.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .body(raw)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    let status = resp.status();
    let body = resp.text().await.context("read response body")?;
    if pretty {
        let v: serde_json::Value = serde_json::from_str(&body).unwrap_or_else(|_| {
            serde_json::Value::String(format!("(invalid JSON: {body})"))
        });
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

fn cmd_evaluate(file: PathBuf, server: String, pretty: bool) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(cmd_evaluate_async(file, server, pretty))
}