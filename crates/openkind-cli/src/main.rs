use anyhow::Result;
use clap::Parser;
use openkind_cli::Cli;

fn main() -> Result<()> {
    openkind_cli::run(Cli::parse())
}
