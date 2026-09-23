use anyhow::Result;
use openkind_cli::Cli;

fn main() -> Result<()> {
    let cli = Cli::try_parse_from(std::env::args_os()).unwrap_or_else(|err| err.exit());
    openkind_cli::run(cli)
}
