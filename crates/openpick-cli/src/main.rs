//! `openpick`: Operator command-line utility for Jev-compatible decision inference.
//!
//! # Subcommands
//! - `openpick inspect <file>`: Local schema and semantics validator for Jev request JSON files.
//! - `openpick evaluate <file>`: Submits a request payload (or stdin via `-`) to a running `openpickd` daemon.
//! - `openpick serve`: Launches the `openpickd` daemon process with configured flags.
//! - `openpick version`: Outputs the current wire API version constant.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use openpick_core::{validate_request, SystemRequest};

#[derive(Parser, Debug)]
#[command(
    name = "openpick",
    about = "openpick — Jev-compatible decision inference CLI"
)]
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
        /// Path to the request JSON file (use '-' for stdin).
        file: PathBuf,
        /// Server URL (e.g. http://127.0.0.1:8080).
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        server: String,
        /// Optional API key for bearer authentication.
        #[arg(long, env = "OPENPICK_API_KEY")]
        api_key: Option<String>,
        /// Print the response as pretty JSON.
        #[arg(long)]
        pretty: bool,
    },

    /// Launch the openpick inference daemon (executes openpickd).
    Serve {
        /// Address to bind the HTTP server on.
        #[arg(long, env = "OPENPICK_HTTP_ADDR", default_value = "0.0.0.0:8080")]
        http_addr: String,
        /// Address to bind the gRPC server on.
        #[arg(long, env = "OPENPICK_GRPC_ADDR", default_value = "0.0.0.0:9090")]
        grpc_addr: String,
        /// Comma-separated model aliases to expose.
        #[arg(long, env = "OPENPICK_MODELS", default_value = "mock,jev-latest")]
        models: String,
        /// Optional bearer token required for /v1/*.
        #[arg(long, env = "OPENPICK_API_KEY")]
        api_key: Option<String>,
    },

    /// Print the openpick wire API version.
    Version,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { file } => cmd_inspect(file),
        Commands::Evaluate {
            file,
            server,
            api_key,
            pretty,
        } => cmd_evaluate(file, server, api_key, pretty),
        Commands::Serve {
            http_addr,
            grpc_addr,
            models,
            api_key,
        } => cmd_serve(http_addr, grpc_addr, models, api_key),
        Commands::Version => {
            println!("openpick {}", openpick_core::api_version());
            Ok(())
        }
    }
}

/// Maximum allowed input file size (32 MB) to prevent local memory exhaustion.
pub const MAX_CLI_INPUT_BYTES: u64 = 32 * 1024 * 1024;

fn cmd_inspect(file: PathBuf) -> Result<()> {
    let meta = std::fs::metadata(&file).with_context(|| format!("stat {}", file.display()))?;
    if meta.len() > MAX_CLI_INPUT_BYTES {
        anyhow::bail!(
            "file {} exceeds maximum allowed size ({} bytes, limit {} bytes)",
            file.display(),
            meta.len(),
            MAX_CLI_INPUT_BYTES
        );
    }
    let raw = std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
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

fn cmd_serve(
    http_addr: String,
    grpc_addr: String,
    models: String,
    api_key: Option<String>,
) -> Result<()> {
    let mut cmd = std::process::Command::new("openpickd");
    cmd.arg("--http-addr").arg(http_addr);
    cmd.arg("--grpc-addr").arg(grpc_addr);
    cmd.arg("--models").arg(models);
    if let Some(key) = api_key {
        // Pass via environment variable to avoid leaking the secret in
        // process table listings (e.g. ps aux / /proc/*/cmdline).
        cmd.env("OPENPICK_API_KEY", key);
    }
    let status = cmd
        .status()
        .context("execute openpickd (is openpickd built and on PATH?)")?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

async fn cmd_evaluate_async(
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
        .or_else(|| std::env::var("OPENPICK_API_KEY").ok())
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

fn cmd_evaluate(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parse_version() {
        let cli = Cli::try_parse_from(["openpick", "version"]).unwrap();
        assert!(matches!(cli.command, Commands::Version));
    }

    #[test]
    fn cli_parse_inspect() {
        let cli = Cli::try_parse_from(["openpick", "inspect", "my_request.json"]).unwrap();
        match cli.command {
            Commands::Inspect { file } => assert_eq!(file, PathBuf::from("my_request.json")),
            _ => panic!("expected Inspect"),
        }
    }

    #[test]
    fn cli_parse_evaluate_defaults_and_flags() {
        let cli = Cli::try_parse_from(["openpick", "evaluate", "my_request.json"]).unwrap();
        match cli.command {
            Commands::Evaluate {
                file,
                server,
                api_key,
                pretty,
            } => {
                assert_eq!(file, PathBuf::from("my_request.json"));
                assert_eq!(server, "http://127.0.0.1:8080");
                assert_eq!(api_key, None);
                assert!(!pretty);
            }
            _ => panic!("expected Evaluate"),
        }

        let cli_custom = Cli::try_parse_from([
            "openpick",
            "evaluate",
            "req.json",
            "--server",
            "http://10.0.0.1:9090",
            "--pretty",
        ])
        .unwrap();
        match cli_custom.command {
            Commands::Evaluate {
                file,
                server,
                api_key,
                pretty,
            } => {
                assert_eq!(file, PathBuf::from("req.json"));
                assert_eq!(server, "http://10.0.0.1:9090");
                assert_eq!(api_key, None);
                assert!(pretty);
            }
            _ => panic!("expected Evaluate"),
        }
    }

    #[test]
    fn cmd_inspect_valid_request() {
        let dir = std::env::temp_dir();
        let file = dir.join("openpick_test_valid_req.json");
        std::fs::write(
            &file,
            r#"{
                "state": "test content",
                "model": "mock",
                "questions": {
                    "is_ok": { "type": "noul", "instructions": "Is it ok?" }
                }
            }"#,
        )
        .unwrap();

        let res = cmd_inspect(file.clone());
        let _ = std::fs::remove_file(file);
        assert!(res.is_ok());
    }

    #[test]
    fn cmd_inspect_nonexistent_file() {
        let res = cmd_inspect(PathBuf::from("/non/existent/file.json"));
        assert!(res.is_err());
    }

    #[test]
    fn cmd_inspect_invalid_json() {
        let dir = std::env::temp_dir();
        let file = dir.join("openpick_test_invalid_json.json");
        std::fs::write(&file, "not valid json").unwrap();

        let res = cmd_inspect(file.clone());
        let _ = std::fs::remove_file(file);
        assert!(res.is_err());
    }

    #[test]
    fn cmd_inspect_invalid_schema_empty_questions() {
        let dir = std::env::temp_dir();
        let file = dir.join("openpick_test_empty_q.json");
        std::fs::write(
            &file,
            r#"{
                "state": "test",
                "model": "mock",
                "questions": {}
            }"#,
        )
        .unwrap();

        let res = cmd_inspect(file.clone());
        let _ = std::fs::remove_file(file);
        assert!(res.is_err());
    }

    #[test]
    fn cli_parse_serve_defaults_and_custom() {
        let cli = Cli::try_parse_from(["openpick", "serve"]).unwrap();
        match cli.command {
            Commands::Serve {
                http_addr,
                grpc_addr,
                models,
                api_key,
            } => {
                assert_eq!(http_addr, "0.0.0.0:8080");
                assert_eq!(grpc_addr, "0.0.0.0:9090");
                assert_eq!(models, "mock,jev-latest");
                assert_eq!(api_key, None);
            }
            _ => panic!("expected Serve"),
        }

        let cli_custom = Cli::try_parse_from([
            "openpick",
            "serve",
            "--http-addr",
            "127.0.0.1:18080",
            "--grpc-addr",
            "127.0.0.1:19090",
            "--models",
            "mock",
            "--api-key",
            "secret123",
        ])
        .unwrap();
        match cli_custom.command {
            Commands::Serve {
                http_addr,
                grpc_addr,
                models,
                api_key,
            } => {
                assert_eq!(http_addr, "127.0.0.1:18080");
                assert_eq!(grpc_addr, "127.0.0.1:19090");
                assert_eq!(models, "mock");
                assert_eq!(api_key, Some("secret123".into()));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn cli_parse_evaluate_with_api_key() {
        let cli = Cli::try_parse_from(["openpick", "evaluate", "req.json", "--api-key", "my-key"])
            .unwrap();
        match cli.command {
            Commands::Evaluate {
                file,
                server,
                api_key,
                pretty,
            } => {
                assert_eq!(file, PathBuf::from("req.json"));
                assert_eq!(server, "http://127.0.0.1:8080");
                assert_eq!(api_key, Some("my-key".into()));
                assert!(!pretty);
            }
            _ => panic!("expected Evaluate"),
        }
    }
}
