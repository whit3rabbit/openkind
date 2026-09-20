use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "opendecision",
    about = "opendecision — Jev-compatible decision inference CLI"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Validate a request JSON file against the opendecision schema.
    Inspect {
        /// Path to the request JSON file.
        file: PathBuf,
    },

    /// Send a request to a running opendecision server.
    Evaluate {
        /// Path to the request JSON file (use '-' for stdin).
        file: PathBuf,
        /// Server URL (e.g. http://127.0.0.1:8080).
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        server: String,
        /// Optional API key for bearer authentication.
        #[arg(long, env = "OPENDECISION_API_KEY")]
        api_key: Option<String>,
        /// Print the response as pretty JSON.
        #[arg(long)]
        pretty: bool,
    },

    /// Launch the opendecision inference daemon (executes opendecisiond).
    Serve {
        /// Address to bind the HTTP server on.
        #[arg(long, env = "OPENDECISION_HTTP_ADDR", default_value = "0.0.0.0:8080")]
        http_addr: String,
        /// Address to bind the gRPC server on.
        #[arg(long, env = "OPENDECISION_GRPC_ADDR", default_value = "0.0.0.0:9090")]
        grpc_addr: String,
        /// Comma-separated model aliases to expose.
        #[arg(long, env = "OPENDECISION_MODELS", default_value = "mock,jev-latest")]
        models: String,
        /// Optional bearer token required for /v1/*.
        #[arg(long, env = "OPENDECISION_API_KEY")]
        api_key: Option<String>,
    },

    /// Print the opendecision wire API version.
    Version,
}
