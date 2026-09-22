//! `openkind`: Operator command-line utility for Jev-compatible decision inference.
//!
//! The library surface keeps command dispatch and request inspection available
//! to tests and benchmarks without spawning the `openkind` process.

mod args;
mod evaluate;
mod inspect;
mod serve;

#[cfg(test)]
mod tests;

use anyhow::Result;

pub use args::{Cli, Commands};
pub use inspect::{parse_and_validate_request, MAX_CLI_INPUT_BYTES};

use evaluate::cmd_evaluate;
use inspect::cmd_inspect;
use serve::cmd_serve;

/// Dispatch a parsed command using the same path as the `openkind` binary.
pub fn run(cli: Cli) -> Result<()> {
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
            println!("openkind {}", openkind_core::api_version());
            Ok(())
        }
    }
}
