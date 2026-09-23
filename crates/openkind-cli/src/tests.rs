use std::path::PathBuf;

use clap::Parser;

use crate::args::{Cli, Commands};
use crate::inspect::{cmd_inspect, parse_and_validate_request, MAX_CLI_INPUT_BYTES};

/// Serializes tests that read or mutate the CLI environment variables.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn cli_parse_version() {
    let cli = Cli::try_parse_from(["openkind", "version"]).unwrap();
    assert!(matches!(cli.command, Commands::Version));
}

#[test]
fn cli_parse_inspect() {
    let cli = Cli::try_parse_from(["openkind", "inspect", "my_request.json"]).unwrap();
    match cli.command {
        Commands::Inspect { file } => assert_eq!(file, PathBuf::from("my_request.json")),
        _ => panic!("expected Inspect"),
    }
}

#[test]
fn cli_parse_evaluate_defaults_and_flags() {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let cli = Cli::try_parse_from(["openkind", "evaluate", "my_request.json"]).unwrap();
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
        "openkind",
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
    let file = dir.join("openkind_test_valid_req.json");
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
fn parse_and_validate_request_accepts_valid_input() {
    let request = parse_and_validate_request(
        r#"{
            "state": "test content",
            "model": "mock",
            "questions": {
                "is_ok": { "type": "noul", "instructions": "Is it ok?" }
            }
        }"#,
    )
    .expect("valid request");
    assert_eq!(request.model, "mock");
    assert_eq!(request.questions.len(), 1);
}

#[test]
fn parse_and_validate_request_preserves_parse_and_validation_failures() {
    let parse_error = parse_and_validate_request("not valid json").expect_err("invalid JSON");
    assert!(parse_error.to_string().contains("parse request JSON"));

    let validation_error =
        parse_and_validate_request(r#"{"state":"test","model":"mock","questions":{}}"#)
            .expect_err("empty questions");
    assert!(validation_error.to_string().contains("validate request"));
}

#[test]
fn cmd_inspect_nonexistent_file() {
    let res = cmd_inspect(PathBuf::from("/non/existent/file.json"));
    assert!(res.is_err());
}

#[test]
fn cmd_inspect_invalid_json() {
    let dir = std::env::temp_dir();
    let file = dir.join("openkind_test_invalid_json.json");
    std::fs::write(&file, "not valid json").unwrap();

    let error = cmd_inspect(file.clone()).expect_err("invalid JSON");
    let _ = std::fs::remove_file(&file);
    assert_eq!(error.to_string(), format!("parse {}", file.display()));
}

#[test]
fn cmd_inspect_invalid_schema_empty_questions() {
    let dir = std::env::temp_dir();
    let file = dir.join("openkind_test_empty_q.json");
    std::fs::write(
        &file,
        r#"{
            "state": "test",
            "model": "mock",
            "questions": {}
        }"#,
    )
    .unwrap();

    let error = cmd_inspect(file.clone()).expect_err("empty questions");
    let _ = std::fs::remove_file(&file);
    assert_eq!(error.to_string(), format!("validate {}", file.display()));
}

#[test]
fn cmd_inspect_rejects_oversized_file() {
    // Sparse file: reserves MAX + 1 bytes on disk metadata without
    // actually writing that much data.
    let dir = std::env::temp_dir();
    let file = dir.join("openkind_test_oversized.json");
    let f = std::fs::File::create(&file).unwrap();
    f.set_len(MAX_CLI_INPUT_BYTES + 1).unwrap();

    let res = cmd_inspect(file.clone());
    let _ = std::fs::remove_file(file);
    let err = res.expect_err("oversized input must be rejected");
    assert!(
        err.to_string().contains("exceeds maximum allowed size"),
        "unexpected error: {err}"
    );
}

#[test]
fn cli_parse_serve_defaults_and_custom() {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let cli = Cli::try_parse_from(["openkind", "serve"]).unwrap();
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
        "openkind",
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
    let cli =
        Cli::try_parse_from(["openkind", "evaluate", "req.json", "--api-key", "my-key"]).unwrap();
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

/// The streaming fast path must agree with the clap parser on well-formed
/// argv shapes, `--` separators, and every diagnostic shape.
#[test]
fn fast_path_parity_with_clap() {
    // Several shapes resolve `api_key` from the environment, so this must
    // not interleave with the env-mutating test below.
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let shapes: &[&[&str]] = &[
        // Well-formed fast-path shapes.
        &["openkind", "version"],
        &["openkind", "inspect", "request.json"],
        &["openkind", "inspect", "a b/c.json"],
        &["openkind", "evaluate", "request.json"],
        &["openkind", "evaluate", "request.json", "--pretty"],
        &[
            "openkind",
            "evaluate",
            "request.json",
            "--server",
            "http://10.0.0.1:9090",
            "--api-key",
            "bench-key",
            "--pretty",
        ],
        &[
            "openkind",
            "evaluate",
            "--pretty",
            "request.json",
            "--server",
            "http://x",
        ],
        &["openkind", "evaluate", "request.json", "--server=http://x"],
        &["openkind", "evaluate", "request.json", "--server="],
        &[
            "openkind",
            "evaluate",
            "--api-key=k",
            "request.json",
            "--pretty",
        ],
        &["openkind", "serve"],
        &[
            "openkind",
            "serve",
            "--http-addr",
            "127.0.0.1:8080",
            "--grpc-addr",
            "127.0.0.1:9090",
            "--models",
            "mock",
            "--api-key",
            "bench-key",
        ],
        &[
            "openkind",
            "serve",
            "--models=m1,m2",
            "--http-addr=1.2.3.4:1",
        ],
        // `--` separator and hyphen-leading path: fast path defers to clap,
        // which accepts them, so both must agree on the success value.
        &["openkind", "inspect", "--", "-weird.json"],
        &["openkind", "inspect", "--", "--pretty"],
        // Diagnostic shapes: fast path defers to clap, errors must be identical.
        &["openkind"],
        &["openkind", "bogus"],
        &["openkind", "version", "extra"],
        &["openkind", "inspect"],
        &["openkind", "inspect", ""],
        &["openkind", "inspect", "-looks-like-flag.json"],
        &["openkind", "evaluate"],
        &["openkind", "evaluate", "--pretty"],
        &["openkind", "evaluate", "a.json", "b.json"],
        &["openkind", "evaluate", "f.json", "--nope"],
        &["openkind", "evaluate", "f.json", "--server"],
        &["openkind", "evaluate", "f.json", "--server", "--pretty"],
        &[
            "openkind", "evaluate", "f.json", "--server", "a", "--server", "b",
        ],
        &["openkind", "evaluate", "f.json", "--pretty", "--pretty"],
        &["openkind", "evaluate", "f.json", "--pretty=true"],
        &["openkind", "serve", "--nope"],
        &["openkind", "serve", "positional"],
        &["openkind", "--help"],
        &["openkind", "evaluate", "--help"],
        &["openkind", "-V"],
    ];
    for argv in shapes {
        let fast = Cli::try_parse_from(argv.to_vec());
        let reference = <Cli as Parser>::try_parse_from(argv.to_vec());
        match (fast, reference) {
            (Ok(a), Ok(b)) => {
                assert_eq!(format!("{a:?}"), format!("{b:?}"), "argv {argv:?}");
            }
            (Err(a), Err(b)) => {
                assert_eq!(a.kind(), b.kind(), "kind mismatch for {argv:?}");
                assert_eq!(
                    a.to_string(),
                    b.to_string(),
                    "message mismatch for {argv:?}"
                );
            }
            (fast, reference) => {
                panic!("outcomes diverge for {argv:?}: fast={fast:?} clap={reference:?}");
            }
        }
    }
}

#[test]
fn fast_path_environment_resolution_matches_clap() {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::env::set_var("OPENKIND_API_KEY", "env-key");
    std::env::set_var("OPENKIND_HTTP_ADDR", "1.2.3.4:5");
    std::env::set_var("OPENKIND_GRPC_ADDR", "5.6.7.8:9");
    std::env::set_var("OPENKIND_MODELS", "env-model");
    let evaluate = ["openkind", "evaluate", "f.json"];
    assert_eq!(
        format!("{:?}", Cli::try_parse_from(evaluate).unwrap()),
        format!("{:?}", <Cli as Parser>::try_parse_from(evaluate).unwrap())
    );
    let serve = ["openkind", "serve", "--models", "m"];
    assert_eq!(
        format!("{:?}", Cli::try_parse_from(serve).unwrap()),
        format!("{:?}", <Cli as Parser>::try_parse_from(serve).unwrap())
    );
    // No flags given: every `serve` env spec (http, grpc, models) resolves
    // through the environment on both paths.
    let serve_defaults = ["openkind", "serve"];
    assert_eq!(
        format!("{:?}", Cli::try_parse_from(serve_defaults).unwrap()),
        format!(
            "{:?}",
            <Cli as Parser>::try_parse_from(serve_defaults).unwrap()
        )
    );

    // Present-but-empty environment values are consumed by clap as-is.
    std::env::set_var("OPENKIND_API_KEY", "");
    assert_eq!(
        format!("{:?}", Cli::try_parse_from(evaluate).unwrap()),
        format!("{:?}", <Cli as Parser>::try_parse_from(evaluate).unwrap())
    );

    std::env::remove_var("OPENKIND_API_KEY");
    std::env::remove_var("OPENKIND_HTTP_ADDR");
    std::env::remove_var("OPENKIND_GRPC_ADDR");
    std::env::remove_var("OPENKIND_MODELS");
    assert_eq!(
        format!("{:?}", Cli::try_parse_from(evaluate).unwrap()),
        format!("{:?}", <Cli as Parser>::try_parse_from(evaluate).unwrap())
    );
}

/// Non-UTF-8 argv must fall back to clap (which accepts arbitrary bytes for
/// `PathBuf` positionals) rather than panic in the streaming fast path.
#[cfg(unix)]
#[test]
fn fast_path_defers_non_utf8_argv_to_clap() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let argv: Vec<std::ffi::OsString> = vec![
        "openkind".into(),
        "evaluate".into(),
        OsStr::from_bytes(b"req\xff.json").to_owned(),
    ];
    let fast = Cli::try_parse_from(argv.clone());
    let reference = <Cli as Parser>::try_parse_from(argv);
    match (fast, reference) {
        (Ok(a), Ok(b)) => {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
        (Err(a), Err(b)) => {
            assert_eq!(a.kind(), b.kind());
            assert_eq!(a.to_string(), b.to_string());
        }
        (fast, reference) => {
            panic!("outcomes diverge: fast={fast:?} clap={reference:?}");
        }
    }
}
