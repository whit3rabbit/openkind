use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use smallvec::SmallVec;

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluateFormat {
    Json,
    Text,
}

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
        /// Output format for successful responses.
        #[arg(long, value_enum, default_value = "json")]
        format: EvaluateFormat,
        /// Include probabilities and request timing in text output.
        #[arg(long)]
        verbose: bool,
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
        /// Comma-separated installed model names to load at daemon startup.
        #[arg(long, env = "OPENKIND_INSTALLED_MODELS", default_value = "")]
        installed_models: String,
        /// Directory shared by model commands and the daemon.
        #[arg(long, env = "OPENKIND_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// Optional bearer token required for /v1/*.
        #[arg(long, env = "OPENKIND_API_KEY")]
        api_key: Option<String>,
    },

    /// Launch the local web playground. Connects to a running daemon when
    /// one is healthy at the target address; otherwise spawns a loopback-only
    /// `openkindd` with the playground route enabled.
    Playground {
        /// Address for the playground daemon (loopback by default).
        #[arg(long, env = "OPENKIND_HTTP_ADDR", default_value = "127.0.0.1:8080")]
        http_addr: String,
        /// Comma-separated model aliases to expose.
        #[arg(long, env = "OPENKIND_MODELS", default_value = "mock,jev-latest")]
        models: String,
        /// Comma-separated installed model names to load at daemon startup.
        #[arg(long, env = "OPENKIND_INSTALLED_MODELS", default_value = "")]
        installed_models: String,
        /// Directory shared by model commands and the daemon.
        #[arg(long, env = "OPENKIND_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// Optional API key for bearer authentication (spawned daemon only).
        #[arg(long, env = "OPENKIND_API_KEY")]
        api_key: Option<String>,
        /// Print the playground URL instead of opening a browser.
        #[arg(long)]
        no_open: bool,
    },

    /// List curated models available to pull.
    Catalog {
        /// Print the catalog as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Download and verify a curated model.
    Pull {
        name: String,
        #[arg(long, env = "OPENKIND_MODELS_DIR")]
        models_dir: Option<PathBuf>,
    },
    /// List installed models without network access.
    List {
        #[arg(long, env = "OPENKIND_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// Print installed profiles as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show a local model's pinned identity and artifacts.
    Show {
        name: String,
        #[arg(long, env = "OPENKIND_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// Print the profile manifest as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove a local model that is not being served.
    Rm {
        name: String,
        #[arg(long, env = "OPENKIND_MODELS_DIR")]
        models_dir: Option<PathBuf>,
    },

    /// Show daemon health and the model aliases registered by that daemon.
    Status {
        /// Server base URL.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        server: String,
        /// Optional API key for bearer authentication.
        #[arg(long, env = "OPENKIND_API_KEY")]
        api_key: Option<String>,
        /// Refresh daemon health and registered model aliases until quit.
        #[arg(long)]
        watch: bool,
    },

    /// Print the openkind wire API version.
    Version,
}

impl Cli {
    /// Parse an argv iterator into a [`Cli`], with identical semantics to
    /// the generated clap parser.
    ///
    /// Well-formed, unambiguous argv shapes are matched by a streaming fast
    /// path that skips clap's generic machinery (the dominant CLI-startup
    /// cost). Every other shape — help requests, diagnostics, unknown or
    /// repeated flags, `--` separators, non-UTF-8 input — falls back to the
    /// full clap parser, so behavior and error messages are byte-for-byte
    /// identical to `<Self as Parser>::try_parse_from`.
    pub fn try_parse_from<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        let argv: SmallVec<[OsString; 12]> = itr.into_iter().map(Into::into).collect();
        if let Some(cli) = fast_parse(&argv) {
            return Ok(cli);
        }
        <Self as Parser>::try_parse_from(argv)
    }
}

/// One `--long` option accepted by a subcommand's fast path.
struct FlagSpec {
    /// Long flag name without the leading dashes.
    long: &'static str,
    /// Environment variable consulted when the flag is absent.
    env: Option<&'static str>,
    /// Value used when neither the flag nor the environment supplies one.
    default: Option<&'static str>,
}

type MatchedFlags<'a> = (SmallVec<[Option<String>; 4]>, SmallVec<[&'a str; 1]>);

/// Match `rest` (the tokens after the subcommand) against the flag grammar.
///
/// Returns the resolved value per spec (explicit flag value, then
/// environment, then default — clap's precedence) plus the collected
/// positional arguments. Returns `None` — routing the whole invocation to
/// the clap fallback — on anything not proven well-formed: unknown flags,
/// repeated flags, boolean flags carrying `=value`, values that look like
/// flags, more positionals than allowed, or non-UTF-8 environment values.
fn match_flags<'a>(
    rest: &[&'a str],
    specs: &[FlagSpec],
    bool_flags: &[&'static str],
    max_positionals: usize,
) -> Option<MatchedFlags<'a>> {
    let mut values: SmallVec<[Option<String>; 4]> = specs.iter().map(|_| None).collect();
    let mut seen_bools: SmallVec<[&'a str; 2]> = SmallVec::new();
    let mut positionals: SmallVec<[&'a str; 1]> = SmallVec::new();

    let mut index = 0;
    while index < rest.len() {
        let token = rest[index];
        if let Some((name, inline_value)) = token.strip_prefix("--").and_then(|t| t.split_once('='))
        {
            // `--name=value` form; only value-taking flags accept it.
            if bool_flags.contains(&name) {
                return None;
            }
            let position = specs.iter().position(|s| s.long == name)?;
            if values[position].is_some() {
                return None; // repeated flag: clap errors, so defer to clap
            }
            values[position] = Some(inline_value.to_owned());
        } else if let Some(name) = token.strip_prefix("--") {
            if bool_flags.contains(&name) {
                if seen_bools.contains(&name) {
                    return None; // repeated flag: clap errors, so defer to clap
                }
                seen_bools.push(name);
            } else {
                let position = specs.iter().position(|s| s.long == name)?;
                if values[position].is_some() {
                    return None; // repeated flag: clap errors, so defer to clap
                }
                let value = rest.get(index + 1)?;
                if value.is_empty() || value.starts_with('-') {
                    return None; // clap rejects flag-shaped values
                }
                values[position] = Some((*value).to_owned());
                index += 1;
            }
        } else if !token.is_empty() && !token.starts_with('-') {
            if positionals.len() == max_positionals {
                return None; // extra positional: clap errors
            }
            positionals.push(token);
        } else {
            return None; // short flags, `--`, or empty positional: defer to clap
        }
        index += 1;
    }

    let mut resolved = SmallVec::new();
    for (seen, spec) in values.into_iter().zip(specs) {
        let value = match seen {
            Some(value) => Some(value),
            None => match spec.env.and_then(std::env::var_os) {
                Some(raw) => {
                    // clap consumes present environment values as-is, even
                    // empty ones; non-UTF-8 values must defer to clap's error.
                    Some(raw.into_string().ok()?)
                }
                None => spec.default.map(str::to_owned),
            },
        };
        resolved.push(value);
    }
    Some((resolved, positionals))
}

/// Streaming fast path over a UTF-8 argv. See [`Cli::try_parse_from`].
fn fast_parse(argv: &[OsString]) -> Option<Cli> {
    let args: SmallVec<[&str; 12]> = argv.iter().map(|a| a.to_str()).collect::<Option<_>>()?;
    let [_, subcommand, rest @ ..] = args.as_slice() else {
        return None;
    };
    match *subcommand {
        "version" if rest.is_empty() => Some(Cli {
            command: Commands::Version,
        }),
        "inspect" if rest.len() == 1 && !rest[0].is_empty() && !rest[0].starts_with('-') => {
            Some(Cli {
                command: Commands::Inspect {
                    file: PathBuf::from(rest[0]),
                },
            })
        }
        "evaluate" => {
            let specs = [
                FlagSpec {
                    long: "server",
                    env: None,
                    default: Some("http://127.0.0.1:8080"),
                },
                FlagSpec {
                    long: "api-key",
                    env: Some("OPENKIND_API_KEY"),
                    default: None,
                },
                FlagSpec {
                    long: "format",
                    env: None,
                    default: Some("json"),
                },
            ];
            let (values, positionals) = match_flags(rest, &specs, &["pretty", "verbose"], 1)?;
            let [file] = positionals.as_slice() else {
                return None; // the file positional is required
            };
            let mut resolved = values.into_iter();
            let server = resolved
                .next()
                .flatten()
                .expect("default guarantees a server value");
            let api_key = resolved.next().flatten();
            let format = match resolved.next().flatten().as_deref() {
                Some("json") => EvaluateFormat::Json,
                Some("text") => EvaluateFormat::Text,
                _ => return None,
            };
            let pretty = rest.contains(&"--pretty");
            let verbose = rest.contains(&"--verbose");
            Some(Cli {
                command: Commands::Evaluate {
                    file: PathBuf::from(*file),
                    server,
                    api_key,
                    pretty,
                    format,
                    verbose,
                },
            })
        }
        "serve" => {
            let specs = [
                FlagSpec {
                    long: "http-addr",
                    env: Some("OPENKIND_HTTP_ADDR"),
                    default: Some("0.0.0.0:8080"),
                },
                FlagSpec {
                    long: "grpc-addr",
                    env: Some("OPENKIND_GRPC_ADDR"),
                    default: Some("0.0.0.0:9090"),
                },
                FlagSpec {
                    long: "models",
                    env: Some("OPENKIND_MODELS"),
                    default: Some("mock,jev-latest"),
                },
                FlagSpec {
                    long: "installed-models",
                    env: Some("OPENKIND_INSTALLED_MODELS"),
                    default: Some(""),
                },
                FlagSpec {
                    long: "models-dir",
                    env: Some("OPENKIND_MODELS_DIR"),
                    default: None,
                },
                FlagSpec {
                    long: "api-key",
                    env: Some("OPENKIND_API_KEY"),
                    default: None,
                },
            ];
            let (values, positionals) = match_flags(rest, &specs, &[], 0)?;
            if !positionals.is_empty() {
                return None;
            }
            let mut resolved = values.into_iter();
            let mut take = || resolved.next().flatten();
            Some(Cli {
                command: Commands::Serve {
                    http_addr: take().expect("default guarantees an http-addr value"),
                    grpc_addr: take().expect("default guarantees a grpc-addr value"),
                    models: take().expect("default guarantees a models value"),
                    installed_models: take().unwrap_or_default(),
                    models_dir: take().map(PathBuf::from),
                    api_key: take(),
                },
            })
        }
        _ => None,
    }
}
