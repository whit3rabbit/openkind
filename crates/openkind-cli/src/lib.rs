//! `openkind`: Operator command-line utility for Jev-compatible decision inference.
//!
//! The library surface keeps command dispatch and request inspection available
//! to tests and benchmarks without spawning the `openkind` process.

mod args;
mod daemon;
mod evaluate;
mod inspect;
mod models;
mod output;
mod playground;
mod serve;
mod status;

#[cfg(test)]
mod tests;

use anyhow::Result;

pub use args::{Cli, Commands, EvaluateFormat};
pub use inspect::{parse_and_validate_request, MAX_CLI_INPUT_BYTES};

use evaluate::cmd_evaluate;
use inspect::cmd_inspect;
use serve::cmd_serve;

/// Dispatch a parsed command using the same path as the `openkind` binary.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Doctor {
            json,
            cuda_device,
            rocm_device,
            onnx_runtime,
        } => daemon::doctor(json, cuda_device, rocm_device, onnx_runtime),
        Commands::Inspect { file } => cmd_inspect(file),
        Commands::Evaluate {
            file,
            server,
            api_key,
            pretty,
            format,
            verbose,
        } => cmd_evaluate(file, server, api_key, pretty, format, verbose),
        Commands::Serve {
            http_addr,
            grpc_addr,
            models,
            installed_models,
            models_dir,
            api_key,
        } => cmd_serve(
            http_addr,
            grpc_addr,
            models,
            installed_models,
            models_dir,
            api_key,
        ),
        Commands::Playground {
            http_addr,
            models,
            installed_models,
            models_dir,
            api_key,
            no_open,
        } => playground::cmd_playground(
            http_addr,
            models,
            installed_models,
            models_dir,
            api_key,
            no_open,
        ),
        Commands::Catalog { json } => models::catalog(json),
        Commands::Pull { name, models_dir } => models::pull(&name, models_dir),
        Commands::List { models_dir, json } => models::list(models_dir, json),
        Commands::Show {
            name,
            models_dir,
            json,
        } => models::show(&name, models_dir, json),
        Commands::Rm { name, models_dir } => models::rm(&name, models_dir),
        Commands::Status {
            server,
            api_key,
            watch,
        } => status::run(server, api_key, watch),
        Commands::Version => {
            println!("openkind {}", openkind_core::api_version());
            Ok(())
        }
    }
}
