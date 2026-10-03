use std::io::{Read, Write};
use std::net::TcpListener;
#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;
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

#[cfg(unix)]
fn fake_daemon(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, "#!/bin/sh\nprintf 'DAEMON_ARG:%s\\n' \"$@\"\n").unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
#[test]
fn empty_installed_model_selection_is_forwarded_to_both_daemon_commands() {
    let dir = TempDir::new();
    fake_daemon(&dir.0.join("openkindd"));
    let mut paths = vec![dir.0.clone()];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    let path = std::env::join_paths(paths).unwrap();
    let socket = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap().to_string();
    drop(socket);
    for command in ["serve", "playground"] {
        let mut child = cli();
        child
            .args([command, "--installed-models", "", "--http-addr", &address])
            .env("OPENKIND_INSTALLED_MODELS", "inherited:profile")
            .env("PATH", &path)
            .env("OPENKINDD_BINARY", dir.0.join("openkindd"));
        if command == "playground" {
            child.arg("--no-open");
        }
        let output = child.output().unwrap();
        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("DAEMON_ARG:--installed-models\nDAEMON_ARG:\n"),
            "{command}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(unix)]
#[test]
fn serve_preserves_the_pid_targeted_by_process_supervisors() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new();
    let daemon = dir.0.join("openkindd");
    std::fs::write(&daemon, "#!/bin/sh\ntrap 'exit 0' TERM\nprintf '%s' \"$$\" > \"$OPENKIND_CLI_TEST_PID\"\nwhile :; do /bin/sleep 0.05; done\n").unwrap();
    std::fs::set_permissions(&daemon, std::fs::Permissions::from_mode(0o755)).unwrap();
    let record = dir.0.join("pid");
    let mut child = cli()
        .arg("serve")
        .env("PATH", &dir.0)
        .env("OPENKINDD_BINARY", &daemon)
        .env("OPENKIND_CLI_TEST_PID", &record)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    let daemon_pid = loop {
        if let Ok(pid) = std::fs::read_to_string(&record) {
            if let Ok(pid) = pid.parse::<u32>() {
                break pid;
            }
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
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
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
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
    use std::os::unix::fs::PermissionsExt;
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
        let deadline = Instant::now() + Duration::from_secs(5);
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
        let mut status = None;
        while Instant::now() < deadline {
            status = child.try_wait().unwrap();
            if status.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if status.is_none() {
            let _ = child.kill();
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
