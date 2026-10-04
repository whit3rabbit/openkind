//! Environment-derived roots are checked without mutating other tests' process environment.

#[test]
fn empty_store_environment_is_rejected_before_cli_or_daemon_io() {
    for command in ["serve", "list", "pull"] {
        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_openkind"));
        child.arg(command);
        if command == "pull" {
            child.arg("fixture:v1");
        }
        let output = child.env("OPENKIND_MODELS_DIR", "").output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{command}");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains("store directory must be nonempty"),
            "{error}"
        );
    }
}
