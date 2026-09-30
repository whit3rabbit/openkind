//! `openkind-gen-schemas`: One-shot generator for Jev JSON Schemas (Draft 2020-12).
//!
//! # Overview
//! Extracts schemars schema definitions from `openkind_core::SystemRequest` and
//! `openkind_core::SystemResponse` and serializes them into canonical schema files:
//! - `crates/openkind-core/schemas/jev-v1-request.json`
//! - `crates/openkind-core/schemas/jev-v1-response.json`
//!
//! Run with `--write` to overwrite schema files in-place:
//! ```bash
//! cargo run -p openkind-gen-schemas -- --write
//! ```

use openkind_core::{SystemRequest, SystemResponse};
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    Print,
    Write,
    Help,
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> io::Result<Mode> {
    let mut write = false;
    let mut help = false;
    for arg in args {
        if arg == "--write" || arg == "-w" {
            write = true;
        } else if arg == "--help" || arg == "-h" {
            help = true;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown argument {arg:?}; use --help for usage"),
            ));
        }
    }
    Ok(if help {
        Mode::Help
    } else if write {
        Mode::Write
    } else {
        Mode::Print
    })
}

fn schema_dir() -> PathBuf {
    // Resolve against this build's source tree, independent of the caller's cwd or environment.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../openkind-core/schemas")
}

fn atomic_write(path: &Path, contents: &[u8]) -> io::Result<()> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "schema path has no parent"))?;
    let temp_path = parent.join(format!(
        ".openkind-schema-{}-{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    // A same-directory rename prevents an interrupted write from truncating the committed file.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)?;
    let result = (|| {
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp_path, path)
    })();
    if result.is_err() {
        // Only remove the temporary file successfully created by this invocation.
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

fn write_schemas(dir: &Path, request: &str, response: &str) -> io::Result<()> {
    for (name, json) in [
        ("jev-v1-request.json", request),
        ("jev-v1-response.json", response),
    ] {
        let path = dir.join(name);
        atomic_write(&path, format!("{json}\n").as_bytes()).map_err(|error| {
            io::Error::new(error.kind(), format!("{}: {error}", path.display()))
        })?;
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mode = parse_args(std::env::args_os().skip(1))?;
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    if mode == Mode::Help {
        writeln!(output, "Usage: openkind-gen-schemas [--write|-w]")?;
        writeln!(
            output,
            "Print Jev schemas; --write also updates the source-tree schema files."
        )?;
        output.flush()?;
        return Ok(());
    }
    let req = schemars::schema_for!(SystemRequest);
    let resp = schemars::schema_for!(SystemResponse);
    let req_json = serde_json::to_string_pretty(&req)?;
    let resp_json = serde_json::to_string_pretty(&resp)?;

    if mode == Mode::Write {
        let dir = schema_dir();
        write_schemas(&dir, &req_json, &resp_json)?;
        eprintln!("Successfully updated JSON schemas in {}", dir.display());
    }

    writeln!(output, "REQUEST:{req_json}")?;
    writeln!(output, "RESPONSE:{resp_json}")?;
    output.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "openkind-schema-test-{}-{}",
                std::process::id(),
                NEXT_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn arguments_are_explicit_and_validated_before_writing() {
        assert_eq!(parse_args([]).unwrap(), Mode::Print);
        for flag in ["--write", "-w"] {
            assert_eq!(parse_args([flag.into()]).unwrap(), Mode::Write);
        }
        for flag in ["--help", "-h"] {
            assert_eq!(parse_args([flag.into()]).unwrap(), Mode::Help);
        }
        for invalid in ["--writ", "request.json"] {
            assert_eq!(
                parse_args(["--write".into(), invalid.into()])
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_argument_is_rejected_without_panicking() {
        use std::os::unix::ffi::OsStringExt;

        assert_eq!(
            parse_args([OsString::from_vec(vec![0xff])])
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn successful_writes_replace_files_and_leave_no_temporary_files() {
        let dir = TestDir::new();
        let request_path = dir.0.join("jev-v1-request.json");
        std::fs::write(&request_path, "previous schema").unwrap();

        write_schemas(&dir.0, "request", "response").unwrap();

        assert_eq!(std::fs::read_to_string(request_path).unwrap(), "request\n");
        assert_eq!(
            std::fs::read_to_string(dir.0.join("jev-v1-response.json")).unwrap(),
            "response\n"
        );
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 2);
    }

    #[test]
    fn failed_replacement_preserves_destination_and_cleans_temporary_file() {
        let dir = TestDir::new();
        let path = dir.0.join("jev-v1-request.json");
        std::fs::create_dir(&path).unwrap();
        let marker = path.join("keep");
        std::fs::write(&marker, "existing contents").unwrap();

        let error = write_schemas(&dir.0, "request", "response").unwrap_err();

        assert!(error.to_string().contains("jev-v1-request.json"));
        assert_eq!(
            std::fs::read_to_string(marker).unwrap(),
            "existing contents"
        );
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
    }

    #[test]
    fn missing_schema_directory_is_an_error() {
        let dir = TestDir::new();
        let missing = dir.0.join("missing");

        assert_eq!(
            write_schemas(&missing, "request", "response")
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        assert!(!missing.exists());
    }
}
