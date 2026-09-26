use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn startup_error(args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_openkindd"))
        .args(args)
        .arg("--models-dir")
        .arg(dir.path())
        .arg("--grpc-addr")
        .arg("0")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(!status.success());
            let output = child.wait_with_output().unwrap();
            return String::from_utf8(output.stderr).unwrap();
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            panic!("daemon did not fail startup for invalid installed model");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn missing_installed_model_fails_before_listening() {
    let stderr = startup_error(&["--installed-models", "fixture:v1"]);
    assert!(stderr.contains("not installed"), "{stderr}");
}

#[test]
fn installed_alias_cannot_shadow_a_mock_alias() {
    let stderr = startup_error(&["--models", "fixture:v1", "--installed-models", "fixture:v1"]);
    assert!(stderr.contains("collides with --models"), "{stderr}");
}

#[test]
fn duplicate_installed_alias_fails_before_loading() {
    let stderr = startup_error(&["--installed-models", "fixture:v1,fixture:v1"]);
    assert!(stderr.contains("duplicate --installed-models"), "{stderr}");
}
