use std::path::PathBuf;

use clap::Parser;

use crate::args::{Cli, Commands};
use crate::inspect::{cmd_inspect, MAX_CLI_INPUT_BYTES};

#[test]
fn cli_parse_version() {
    let cli = Cli::try_parse_from(["opendecision", "version"]).unwrap();
    assert!(matches!(cli.command, Commands::Version));
}

#[test]
fn cli_parse_inspect() {
    let cli = Cli::try_parse_from(["opendecision", "inspect", "my_request.json"]).unwrap();
    match cli.command {
        Commands::Inspect { file } => assert_eq!(file, PathBuf::from("my_request.json")),
        _ => panic!("expected Inspect"),
    }
}

#[test]
fn cli_parse_evaluate_defaults_and_flags() {
    let cli = Cli::try_parse_from(["opendecision", "evaluate", "my_request.json"]).unwrap();
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
        "opendecision",
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
    let file = dir.join("opendecision_test_valid_req.json");
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
fn cmd_inspect_nonexistent_file() {
    let res = cmd_inspect(PathBuf::from("/non/existent/file.json"));
    assert!(res.is_err());
}

#[test]
fn cmd_inspect_invalid_json() {
    let dir = std::env::temp_dir();
    let file = dir.join("opendecision_test_invalid_json.json");
    std::fs::write(&file, "not valid json").unwrap();

    let res = cmd_inspect(file.clone());
    let _ = std::fs::remove_file(file);
    assert!(res.is_err());
}

#[test]
fn cmd_inspect_invalid_schema_empty_questions() {
    let dir = std::env::temp_dir();
    let file = dir.join("opendecision_test_empty_q.json");
    std::fs::write(
        &file,
        r#"{
            "state": "test",
            "model": "mock",
            "questions": {}
        }"#,
    )
    .unwrap();

    let res = cmd_inspect(file.clone());
    let _ = std::fs::remove_file(file);
    assert!(res.is_err());
}

#[test]
fn cmd_inspect_rejects_oversized_file() {
    // Sparse file: reserves MAX + 1 bytes on disk metadata without
    // actually writing that much data.
    let dir = std::env::temp_dir();
    let file = dir.join("opendecision_test_oversized.json");
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
    let cli = Cli::try_parse_from(["opendecision", "serve"]).unwrap();
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
        "opendecision",
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
    let cli = Cli::try_parse_from([
        "opendecision",
        "evaluate",
        "req.json",
        "--api-key",
        "my-key",
    ])
    .unwrap();
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
