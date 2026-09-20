//! `opendecision`: Operator command-line utility for Jev-compatible decision inference.
//!
//! # Subcommands
//! - `opendecision inspect <file>`: Local schema and semantics validator for Jev request JSON files.
//! - `opendecision evaluate <file>`: Submits a request payload (or stdin via `-`) to a running `opendecisiond` daemon.
//! - `opendecision serve`: Launches the `opendecisiond` daemon process with configured flags.
//! - `opendecision version`: Outputs the current wire API version constant.

mod args;
mod evaluate;
mod inspect;
mod serve;
#[cfg(test)]
mod tests;

use anyhow::Result;
use clap::Parser;

use crate::args::{Cli, Commands};
use crate::evaluate::cmd_evaluate;
use crate::inspect::cmd_inspect;
use crate::serve::cmd_serve;

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
            println!("opendecision {}", opendecision_core::api_version());
            Ok(())
        }
    }
}
