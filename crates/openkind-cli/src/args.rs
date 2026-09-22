use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Top-level command-line argument parser for the `openkind` CLI.
#[derive(Parser, Debug)]
#[command(
    name = "openkind",
    about = "openkind — Jev-compatible decision inference CLI"
)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Subcommands supported by the `openkind` CLI.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Validate a request JSON file against the openkind schema.
    Inspect {
        /// Path to the request JSON file.
        file: PathBuf,
    },

    /// Send a request to a running openkind server.
    Evaluate {
        /// Path to the request JSON file (use '-' for stdin).
        file: PathBuf,
        /// Server URL (e.g. http://127.0.0.1:8080).
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        server: String,
        /// Optional API key for bearer authentication.
        #[arg(long, env = "OPENKIND_API_KEY")]
        api_key: Option<String>,
        /// Print the response as pretty JSON.
        #[arg(long)]
        pretty: bool,
    },

    /// Launch the openkind inference daemon (executes openkindd).
    Serve {
        /// Address to bind the HTTP server on.
        #[arg(long, env = "OPENKIND_HTTP_ADDR", default_value = "0.0.0.0:8080")]
        http_addr: String,
        /// Address to bind the gRPC server on.
        #[arg(long, env = "OPENKIND_GRPC_ADDR", default_value = "0.0.0.0:9090")]
        grpc_addr: String,
        /// Comma-separated model aliases to expose.
        #[arg(long, env = "OPENKIND_MODELS", default_value = "mock,jev-latest")]
        models: String,
        /// Optional bearer token required for /v1/*.
        #[arg(long, env = "OPENKIND_API_KEY")]
        api_key: Option<String>,
    },

    /// Print the openkind wire API version.
    Version,
}
