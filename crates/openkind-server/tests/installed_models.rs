use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn startup_error(args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_openkindd"));
    command.args(args).arg("--models-dir").arg(dir.path());
    if !args.contains(&"--grpc-addr") {
        command.args(["--grpc-addr", "0"]);
    }
    let mut child = command
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
            panic!("daemon did not fail startup for invalid configuration");
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

#[test]
fn router_script_self_reference_fails_instead_of_binding_a_mock() {
    let stderr = startup_error(&[
        "--models",
        "router-script",
        "--router-script-rules",
        "default=router-script",
    ]);
    assert!(stderr.contains("unregistered sibling"), "{stderr}");
}

#[test]
fn unresolved_router_script_dependency_fails_instead_of_binding_a_mock() {
    let stderr = startup_error(&[
        "--models",
        "first,second",
        "--router-script-aliases",
        "first,second",
        "--router-script-rules",
        "default=second;default=first",
    ]);
    assert!(stderr.contains("unregistered sibling"), "{stderr}");
}

#[test]
fn native_family_collision_fails_before_loading_artifacts() {
    let stderr = startup_error(&[
        "--models",
        "decoder-letter-native",
        "--qwen35-aliases",
        "decoder-letter-native",
    ]);
    assert!(
        stderr.contains("assigned to more than one family engine"),
        "{stderr}"
    );
}

#[test]
fn listener_bind_failures_stop_the_other_protocol() {
    let occupied = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = occupied.local_addr().unwrap().to_string();
    let http_error = startup_error(&["--http-addr", &address, "--grpc-addr", "127.0.0.1:0"]);
    assert!(http_error.contains("bind http"), "{http_error}");
    let grpc_error = startup_error(&["--http-addr", "127.0.0.1:0", "--grpc-addr", &address]);
    assert!(grpc_error.contains("grpc serve"), "{grpc_error}");
}

#[test]
fn missing_qwen3_root_fails_before_listening() {
    let stderr = startup_error(&[
        "--models",
        "victim17",
        "--decoder-logit-qwen3-aliases",
        "17b=victim17",
    ]);
    assert!(
        stderr.contains("decoder-logit-qwen3-17b") && stderr.contains("no root for 17b"),
        "{stderr}"
    );
}

#[test]
fn documented_qwen3_multiprofile_syntax_fails_on_missing_artifact_instead_of_mock() {
    let stderr = startup_error(&[
        "--models",
        "victim17",
        "--decoder-logit-qwen3-aliases",
        "06b=victim06;17b=victim17;4b=victim4",
        "--decoder-logit-qwen3-model-roots",
        "06b=/nonexistent/06;17b=/nonexistent/17;4b=/nonexistent/4",
    ]);
    assert!(
        stderr.contains("load decoder-logit-qwen3-17b engine"),
        "{stderr}"
    );
}
