use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const REQUEST: &str = r#"{"state":"fixture","model":"mock","questions":{"q":{"type":"noul","instructions":"Check fixture"}}}"#;
const RESPONSE: &str = r#"{"model":"mock","answers":{"q":{"type":"noul","noul":0.75,"provider_note":"extra answer metadata"}},"usage":{"input_tokens":1,"output_tokens":1},"provider_note":"extra response metadata"}"#;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "openkind-cli-{}-{unique}-{count}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cli() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_openkind"));
    for variable in [
        "OPENKIND_API_KEY",
        "OPENDECISION_API_KEY",
        "OPENPICK_API_KEY",
        "TYPESAFE_API_KEY",
        "OPENKIND_HTTP_ADDR",
        "OPENKIND_GRPC_ADDR",
        "OPENKIND_MODELS",
        "OPENKIND_INSTALLED_MODELS",
        "OPENKIND_MODELS_DIR",
    ] {
        command.env_remove(variable);
    }
    command
}

#[test]
fn keygen_prints_one_random_key_without_a_daemon_or_model_store() {
    let dir = TempDir::new();
    let mut keys = Vec::new();
    for _ in 0..2 {
        let output = cli()
            .arg("keygen")
            .env("OPENKINDD_BINARY", dir.0.join("missing-daemon"))
            .env("OPENKIND_MODELS_DIR", dir.0.join("missing-models"))
            .current_dir(&dir.0)
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(stdout.lines().count(), 1);
        let key = stdout
            .strip_suffix('\n')
            .expect("newline for shell capture");
        assert_eq!(key.len(), 67);
        let token = key.strip_prefix("ok_").expect("OpenKind key prefix");
        assert!(token.bytes().all(|byte| byte.is_ascii_hexdigit()));
        keys.push(key.to_owned());
    }
    assert_ne!(keys[0], keys[1]);
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
}

#[test]
fn help_does_not_print_configured_api_keys() {
    for command in ["serve", "playground", "evaluate", "status"] {
        let output = cli()
            .args([command, "--help"])
            .env("OPENKIND_API_KEY", "help-output-secret")
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("OPENKIND_API_KEY"));
        assert!(!stdout.contains("help-output-secret"));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("help-output-secret"));
    }
}

#[test]
fn invalid_server_keys_fail_before_startup_without_echoing_secrets() {
    for command in ["serve", "playground"] {
        for key in [
            "",
            " ",
            "secret with spaces",
            "-secret with spaces",
            "secret\tvalue",
            "secret\nvalue",
            "sëcret",
        ] {
            for from_env in [false, true] {
                let mut child = cli();
                child.arg(command).env("OPENKINDD_BINARY", "missing-daemon");
                if from_env {
                    child.env("OPENKIND_API_KEY", key);
                } else {
                    child.args(["--api-key", key]);
                }
                let output = child.output().unwrap();
                assert_eq!(output.status.code(), Some(1), "{command}: {from_env}");
                assert!(output.stdout.is_empty());
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(stderr.contains("API key must be non-empty visible ASCII"));
                if key.len() > 1 {
                    assert!(!stderr.contains(key));
                }
            }
        }
    }
}

#[test]
fn serve_forwards_optional_keys_via_environment_only() {
    let dir = TempDir::new();
    let daemon = fake_daemon(&dir.0);
    for (env_key, flag_key, expected) in [
        (None, None, "unset"),
        (Some("env-token"), None, "env-token"),
        (None, Some("flag-token"), "flag-token"),
        (Some("env-token"), Some("flag-token"), "flag-token"),
    ] {
        let mut child = cli();
        child
            .arg("serve")
            .env("OPENKINDD_BINARY", &daemon)
            .env("OPENKIND_CLI_TEST_EXPECTED_KEY", expected);
        if let Some(key) = env_key {
            child.env("OPENKIND_API_KEY", key);
        }
        if let Some(key) = flag_key {
            child.args(["--api-key", key]);
        }
        let output = child.output().unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("DAEMON_ARG:--models"));
        assert!(!stdout.contains("--api-key"));
        assert!(!stdout.contains("env-token"));
        assert!(!stdout.contains("flag-token"));
        assert!(output.stderr.is_empty());
    }
}

fn evaluate(request: &str, status: &str, body: &str, flags: &[&str]) -> Output {
    let dir = TempDir::new();
    let path = dir.0.join("request.json");
    std::fs::write(&path, request).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let server = format!("http://{}", listener.local_addr().unwrap());
    let status = status.to_owned();
    let body = body.to_owned();
    let worker = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "CLI did not send an evaluation");
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept evaluation: {error}"),
            }
        };
        socket.set_nonblocking(false).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 2048];
        loop {
            let size = socket.read(&mut buffer).unwrap();
            assert_ne!(size, 0);
            request.extend_from_slice(&buffer[..size]);
            if let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                assert!(headers.starts_with("POST /v1/systemone "));
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().unwrap())
                    })
                    .unwrap();
                if request.len() >= header_end + 4 + length {
                    break;
                }
            }
        }
        write!(socket, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nx-typesafe-request-id: fixture-id\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    });
    let output = cli()
        .arg("evaluate")
        .arg(path)
        .arg("--server")
        .arg(server)
        .args(flags)
        .output()
        .unwrap();
    worker.join().unwrap();
    output
}

#[test]
fn evaluate_preserves_unknown_metadata_after_validating_answers() {
    for flags in [vec![], vec!["--pretty"], vec!["--verbose"]] {
        let output = evaluate(REQUEST, "200 OK", RESPONSE, &flags);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            actual,
            serde_json::from_str::<serde_json::Value>(RESPONSE).unwrap()
        );
    }
}

#[test]
fn every_output_mode_rejects_invalid_successful_answers() {
    let choice_request = r#"{"state":"fixture","model":"mock","questions":{"q":{"type":"choice","instructions":"Choose","criteria":{"a":"A","b":"B"}}}}"#;
    let score_request = r#"{"state":"fixture","model":"mock","questions":{"q":{"type":"score","instructions":"Score","criteria":["low","high"]}}}"#;
    for (request, body) in [
        (REQUEST, "{}"),
        (
            REQUEST,
            r#"{"model":"mock","answers":{"wrong":{"type":"noul","noul":0.75}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ),
        (
            REQUEST,
            r#"{"model":"mock","answers":{"q":{"type":"noul","noul":2.0}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ),
        (
            choice_request,
            r#"{"model":"mock","answers":{"q":{"type":"choice","choice":"other","confidence":0.5,"probabilities":{"a":0.5,"b":0.5}}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ),
        (
            score_request,
            r#"{"model":"mock","answers":{"q":{"type":"score","score":0.5,"confidence":0.5,"probabilities":{"0":0.5,"1":0.5},"legend":{"0":"forged","1":"high"}}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ),
        (
            score_request,
            r#"{"model":"mock","answers":{"q":{"type":"score","score":0.9,"confidence":0.5,"probabilities":{"0":0.5,"1":0.5},"legend":{"0":"low","1":"high"}}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ),
    ] {
        for format in ["json", "text"] {
            let output = evaluate(request, "200 OK", body, &["--format", format]);
            assert_eq!(output.status.code(), Some(1), "{format}: {body}");
            assert!(
                output.stdout.is_empty(),
                "invalid answers must not reach stdout"
            );
            assert!(String::from_utf8_lossy(&output.stderr).contains("evaluation response"));
        }
    }
}

#[test]
fn evaluate_http_errors_keep_status_and_request_id_on_stderr() {
    let output = evaluate(
        REQUEST,
        "503 Service Unavailable",
        r#"{"error":"overloaded"}"#,
        &[],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("503"));
    assert!(stderr.contains("fixture-id"));
    assert!(stderr.contains("overloaded"));
}

#[test]
fn oversized_stdin_is_rejected_instead_of_sending_a_valid_prefix() {
    let mut child = cli()
        .args(["evaluate", "-", "--server", "http://127.0.0.1:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input.write_all(REQUEST.as_bytes()).unwrap();
    let mut remaining = openkind_cli::MAX_CLI_INPUT_BYTES + 1 - REQUEST.len() as u64;
    let whitespace = [b' '; 8192];
    while remaining > 0 {
        let size = remaining.min(whitespace.len() as u64) as usize;
        input.write_all(&whitespace[..size]).unwrap();
        remaining -= size as u64;
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("exceeds maximum allowed size"));
}

#[cfg(unix)]
#[test]
fn non_utf8_string_environment_values_use_clap_diagnostics() {
    use std::os::unix::ffi::OsStringExt;
    for (variable, arguments) in [
        ("OPENKIND_API_KEY", vec!["evaluate", "missing.json"]),
        ("OPENKIND_HTTP_ADDR", vec!["serve"]),
        ("OPENKIND_INSTALLED_MODELS", vec!["serve"]),
    ] {
        let output = cli()
            .args(arguments)
            .env(variable, std::ffi::OsString::from_vec(vec![0xff]))
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{variable}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("invalid UTF-8"));
    }
}

fn fake_daemon(directory: &Path) -> PathBuf {
    // Compile a native fixture so subprocess contracts run on Windows without a shell.
    let path = directory.join(format!("openkindd{}", std::env::consts::EXE_SUFFIX));
    let output = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .arg("--edition=2021")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/daemon.rs"))
        .arg("-o")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "compile daemon fixture: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    path
}

#[test]
fn empty_installed_model_selection_clears_inherited_models_for_both_daemon_commands() {
    let dir = TempDir::new();
    let daemon = fake_daemon(&dir.0);
    let socket = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap().to_string();
    drop(socket);
    for command in ["serve", "playground"] {
        let mut child = cli();
        child
            .args([command, "--installed-models", "", "--http-addr", &address])
            .env("OPENKIND_INSTALLED_MODELS", "inherited:profile")
            .env("OPENKINDD_BINARY", &daemon);
        if command == "playground" {
            child.arg("--no-open");
        }
        let output = child.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("DAEMON_INSTALLED_MODELS:unset\n")
                && !stdout.contains("DAEMON_ARG:--installed-models\n"),
            "{command}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(unix)]
#[test]
fn serve_preserves_the_pid_targeted_by_process_supervisors() {
    use std::os::unix::{fs::PermissionsExt, process::CommandExt};
    let dir = TempDir::new();
    let daemon = dir.0.join("openkindd");
    std::fs::write(&daemon, "#!/bin/sh\ntrap 'exit 0' TERM\nprintf '%s' \"$$\" > \"$OPENKIND_CLI_TEST_PID\"\nwhile :; do /bin/sleep 0.05; done\n").unwrap();
    std::fs::set_permissions(&daemon, std::fs::Permissions::from_mode(0o755)).unwrap();
    let record = dir.0.join("pid");
    let mut child = cli()
        .arg("serve")
        .process_group(0)
        .env("PATH", &dir.0)
        .env("OPENKINDD_BINARY", &daemon)
        .env("OPENKIND_CLI_TEST_PID", &record)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let daemon_pid = loop {
        if let Ok(pid) = std::fs::read_to_string(&record) {
            if let Ok(pid) = pid.parse::<u32>() {
                break pid;
            }
        }
        if Instant::now() >= deadline {
            unsafe {
                libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
            }
            child.wait().unwrap();
            panic!("fixture daemon did not start");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    // The recorded PID belongs to the unreaped fixture process.
    assert_eq!(
        unsafe { libc::kill(daemon_pid as libc::pid_t, libc::SIGTERM) },
        0
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            unsafe {
                libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
            }
            child.wait().unwrap();
            panic!("fixture daemon did not shut down");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(
        daemon_pid,
        child.id(),
        "daemon must receive signals sent to the CLI PID"
    );
    assert!(status.success());
}

#[cfg(unix)]
#[test]
fn playground_supervises_startup_and_ready_signals_and_preserves_exit_codes() {
    use std::io::BufRead;
    use std::os::unix::{fs::PermissionsExt, process::CommandExt};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    for (healthy, signal, expected_code) in [
        (false, Some(libc::SIGTERM), 0),
        (true, Some(libc::SIGINT), 0),
        (true, None, 7),
    ] {
        let dir = TempDir::new();
        let pid_file = dir.0.join("pid");
        let stop_file = dir.0.join("stop");
        let signal_file = dir.0.join("signal");
        let daemon = dir.0.join("openkindd");
        std::fs::write(&daemon, "#!/bin/sh\ntrap 'printf stopped > \"$OPENKIND_CLI_TEST_SIGNAL\"; exit 0' TERM INT\nprintf '%s' \"$$\" > \"$OPENKIND_CLI_TEST_PID\"\nwhile [ ! -f \"$OPENKIND_CLI_TEST_STOP\" ]; do /bin/sleep 0.05; done\nexit 7\n").unwrap();
        std::fs::set_permissions(&daemon, std::fs::Permissions::from_mode(0o755)).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let finished = Arc::new(AtomicBool::new(false));
        let server_finished = finished.clone();
        let server_pid_file = pid_file.clone();
        let server = std::thread::spawn(move || {
            while !server_finished.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        socket.set_nonblocking(false).unwrap();
                        socket
                            .set_read_timeout(Some(Duration::from_secs(1)))
                            .unwrap();
                        let mut buffer = [0; 2048];
                        let _ = socket.read(&mut buffer);
                        let status = if healthy && server_pid_file.exists() {
                            "200 OK"
                        } else {
                            "503 Service Unavailable"
                        };
                        let body = r#"{"status":"ok"}"#;
                        let _ = write!(socket, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    Err(error) => panic!("fixture health listener: {error}"),
                }
            }
        });
        let mut child = cli()
            .args(["playground", "--http-addr", &address, "--no-open"])
            .process_group(0)
            .env("PATH", &dir.0)
            .env("OPENKINDD_BINARY", &daemon)
            .env("OPENKIND_CLI_TEST_PID", &pid_file)
            .env("OPENKIND_CLI_TEST_STOP", &stop_file)
            .env("OPENKIND_CLI_TEST_SIGNAL", &signal_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let ready = Arc::new(AtomicBool::new(false));
        let output_ready = ready.clone();
        let stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                if line.unwrap().contains("Ctrl-C stops the daemon.") {
                    output_ready.store(true, Ordering::Release);
                }
            }
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut daemon_pid = None;
        while Instant::now() < deadline {
            daemon_pid = std::fs::read_to_string(&pid_file)
                .ok()
                .and_then(|pid| pid.parse::<libc::pid_t>().ok());
            if daemon_pid.is_some() && (!healthy || ready.load(Ordering::Acquire)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let started = daemon_pid.is_some() && (!healthy || ready.load(Ordering::Acquire));
        if let Some(signal) = signal {
            // Only the wrapper receives this signal; it must forward to its child.
            assert_eq!(unsafe { libc::kill(child.id() as libc::pid_t, signal) }, 0);
        } else {
            std::fs::write(&stop_file, "stop").unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut status = None;
        while Instant::now() < deadline {
            status = child.try_wait().unwrap();
            if status.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if status.is_none() {
            unsafe {
                libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
            }
        }
        let _ = child.wait();
        let child_reaped = daemon_pid.is_some_and(|pid| {
            // Signal zero checks process existence without delivering a signal.
            let result = unsafe { libc::kill(pid, 0) };
            result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
        });
        if !child_reaped {
            if let Some(pid) = daemon_pid {
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }
        // A late-starting child can inherit the stdout pipe after the wrapper
        // is killed. Its dedicated group must close before joining the reader.
        if !child_reaped {
            unsafe {
                libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
            }
        }
        finished.store(true, Ordering::Release);
        server.join().unwrap();
        reader.join().unwrap();
        assert!(started, "fixture did not reach requested startup phase");
        assert_eq!(status.and_then(|status| status.code()), Some(expected_code));
        assert!(
            child_reaped,
            "playground left its daemon running or unreaped"
        );
        if signal.is_some() {
            assert!(
                signal_file.exists(),
                "daemon never received shutdown signal"
            );
        }
    }
}
