use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::Connection;
use serde_json::Value;

const RESPONSE: &str = r#"{"model":"mock","answers":{"q":{"type":"noul","noul":0.75}},"usage":{"input_tokens":1,"output_tokens":1}}"#;

fn request(state: &str) -> String {
    serde_json::json!({"state":state,"model":"mock","questions":{"q":{"type":"noul","instructions":"Check fixture"}}}).to_string()
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "openkind-batch-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn input(&self, text: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join("input.jsonl");
        std::fs::write(&path, text).unwrap();
        path
    }
    fn job(&self) -> PathBuf {
        self.0.join("job")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cli() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_openkind"));
    for name in [
        "OPENKIND_API_KEY",
        "OPENDECISION_API_KEY",
        "TYPESAFE_API_KEY",
        "OPENKIND_BASE_URL",
        "TYPESAFE_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        command.env_remove(name);
    }
    command.arg("batch");
    command
}

struct Process(Option<Child>);
impl Deref for Process {
    type Target = Child;
    fn deref(&self) -> &Child {
        self.0.as_ref().unwrap()
    }
}
impl DerefMut for Process {
    fn deref_mut(&mut self) -> &mut Child {
        self.0.as_mut().unwrap()
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
impl Process {
    fn finish(mut self) -> Output {
        wait_for(|| self.try_wait().unwrap().is_some());
        self.0.take().unwrap().wait_with_output().unwrap()
    }
}

fn spawn(input: &Path, job: &Path, server: &str, interval: u64) -> Process {
    spawn_with_args(input, job, server, interval, &[])
}

fn spawn_with_args(
    input: &Path,
    job: &Path,
    server: &str,
    interval: u64,
    args: &[&str],
) -> Process {
    Process(Some(
        cli()
            .arg("run")
            .arg(input)
            .arg("--job-dir")
            .arg(job)
            .args(["--server", server, "--interval-ms", &interval.to_string()])
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ))
}

fn wait_for(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(8);
    while !predicate() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for batch process state"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn saved(job: &Path, sql: &str) -> Option<i64> {
    Connection::open_with_flags(
        job.join("job.sqlite3"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .ok()?
    .query_row(sql, [], |row| row.get(0))
    .ok()
}

fn json_lines(output: &Output) -> Vec<Value> {
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn status(job: &Path) -> Value {
    let output = cli().arg("status").arg(job).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[derive(Clone)]
struct Reply {
    status: u16,
    body: String,
    headers: String,
    delay: Duration,
}
impl Reply {
    fn ok() -> Self {
        Self {
            status: 200,
            body: RESPONSE.into(),
            headers: String::new(),
            delay: Duration::ZERO,
        }
    }
    fn delayed(ms: u64) -> Self {
        Self {
            delay: Duration::from_millis(ms),
            ..Self::ok()
        }
    }
    fn error(status: u16) -> Self {
        Self {
            status,
            body: r#"{"error":{"message":"fixture failure"}}"#.into(),
            ..Self::ok()
        }
    }
}

#[derive(Clone)]
struct Event {
    state: String,
    headers: String,
    started: Instant,
}

struct Server {
    url: String,
    events: Arc<Mutex<Vec<Event>>>,
    maximum: Arc<AtomicUsize>,
    stopped: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    fn new(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let events = Arc::new(Mutex::new(Vec::new()));
        let maximum = Arc::new(AtomicUsize::new(0));
        let stopped = Arc::new(AtomicBool::new(false));
        let (thread_events, thread_maximum, thread_stopped) =
            (events.clone(), maximum.clone(), stopped.clone());
        let thread = std::thread::spawn(move || {
            let active = Arc::new(AtomicUsize::new(0));
            let replies = Arc::new(Mutex::new(VecDeque::from(replies)));
            let mut connections = Vec::new();
            while !thread_stopped.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let (events, maximum, active, replies) = (
                            thread_events.clone(),
                            thread_maximum.clone(),
                            active.clone(),
                            replies.clone(),
                        );
                        connections.push(std::thread::spawn(move || {
                            serve(stream, events, maximum, active, replies)
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("{error}"),
                }
            }
            for connection in connections {
                connection.join().unwrap();
            }
        });
        Self {
            url,
            events,
            maximum,
            stopped,
            thread: Some(thread),
        }
    }
    fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }
    fn wait_events(&self, count: usize) {
        wait_for(|| self.events().len() >= count);
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap();
    }
}

fn serve(
    mut stream: TcpStream,
    events: Arc<Mutex<Vec<Event>>>,
    maximum: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    replies: Arc<Mutex<VecDeque<Reply>>>,
) {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0; 8192];
    let header_end = loop {
        let Ok(count) = stream.read(&mut buffer) else {
            return;
        };
        if count == 0 {
            return;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8(bytes[..header_end].to_vec())
        .unwrap()
        .to_lowercase();
    let length = headers
        .lines()
        .find_map(|line| {
            line.strip_prefix("content-length:")
                .map(|value| value.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    while bytes.len() < header_end + length {
        let Ok(count) = stream.read(&mut buffer) else {
            return;
        };
        if count == 0 {
            return;
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let body: Value = serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
    let reply = replies
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or_else(Reply::ok);
    let running = active.fetch_add(1, Ordering::SeqCst) + 1;
    maximum.fetch_max(running, Ordering::SeqCst);
    events.lock().unwrap().push(Event {
        state: body["state"].as_str().unwrap().into(),
        headers,
        started: Instant::now(),
    });
    std::thread::sleep(reply.delay);
    active.fetch_sub(1, Ordering::SeqCst);
    let wire = format!("HTTP/1.1 {} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n{}",reply.status,reply.body.len(),reply.headers,reply.body);
    let _ = stream.write_all(wire.as_bytes());
}

#[test]
fn jsonl_framing_retries_and_pacing_remain_sequential() {
    let dir = TempDir::new();
    let input = dir.input(format!(
        "\n{}\r\n{}\n{}",
        request("1"),
        request("2"),
        request("3")
    ));
    let rate_limit = Reply {
        headers: "retry-after-ms: 120\r\nRetry-After: 10\r\n".into(),
        ..Reply::error(429)
    };
    let server = Server::new(vec![
        rate_limit,
        Reply::delayed(50),
        Reply::delayed(50),
        Reply::delayed(50),
    ]);
    let output = spawn(&input, &dir.job(), &server.url, 45).finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = json_lines(&output);
    assert_eq!(lines.len(), 3);
    for (index, line) in lines.iter().enumerate() {
        assert_eq!(line["record"], index + 1);
        assert_eq!(line["line"], index + 2);
        assert_eq!(line["response"]["answers"]["q"]["noul"], 0.75);
    }
    let events = server.events();
    assert_eq!(
        events
            .iter()
            .map(|event| event.state.as_str())
            .collect::<Vec<_>>(),
        ["1", "1", "2", "3"]
    );
    assert!(!events[0].headers.contains("x-typesafe-retry-count"));
    assert!(events[1].headers.contains("x-typesafe-retry-count: 1"));
    assert!(!events[0].headers.contains("authorization:"));
    assert!(events[1].started.duration_since(events[0].started) >= Duration::from_millis(100));
    assert!(events[2].started.duration_since(events[1].started) >= Duration::from_millis(75));
    assert_eq!(server.maximum.load(Ordering::SeqCst), 1);
    assert_eq!(status(&dir.job())["state"], "completed");
    assert_eq!(
        json_lines(&cli().arg("export").arg(dir.job()).output().unwrap()),
        lines
    );
}

#[test]
fn row_range_is_inclusive_counts_blank_lines_and_survives_resume() {
    let dir = TempDir::new();
    let input = dir.input(format!(
        "not JSON\n{}\r\n\n{}\nnot JSON\n",
        request("2"),
        request("4")
    ));
    let server = Server::new(vec![Reply::ok(), Reply::error(401)]);
    let output = spawn_with_args(
        &input,
        &dir.job(),
        &server.url,
        0,
        &["--start-row", "2", "--end-row", "4"],
    )
    .finish();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(json_lines(&output)[0]["line"], 2);
    assert_eq!(status(&dir.job())["last_successful_row"], 2);
    assert_eq!(saved(&dir.job(), "SELECT next_line FROM job"), Some(5));

    // The cursor is past the end, but its captured failed record still needs
    // replay. Malformed input beyond the range must never pause the job.
    let output = cli().arg("resume").arg(dir.job()).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(json_lines(&output)[0]["line"], 4);
    assert_eq!(
        server
            .events()
            .iter()
            .map(|event| event.state.as_str())
            .collect::<Vec<_>>(),
        ["2", "4", "4"]
    );
    let progress = status(&dir.job());
    assert_eq!(progress["state"], "completed");
    assert_eq!(progress["captured"], 2);
    assert_eq!(progress["last_successful_row"], 4);
    let export = json_lines(&cli().arg("export").arg(dir.job()).output().unwrap());
    assert_eq!(
        export
            .iter()
            .map(|result| result["line"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [2, 4]
    );
    assert_eq!(export[0]["record"], 1);
    assert_eq!(export[1]["record"], 2);
    let output = cli().arg("resume").arg(dir.job()).output().unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(server.events().len(), 3);
}

#[test]
fn row_bounds_can_be_used_independently_and_allow_an_empty_selection() {
    for (args, expected) in [
        (vec!["--start-row", "3"], vec![3]),
        (vec!["--end-row", "2"], vec![1]),
        (vec!["--start-row", "3", "--end-row", "3"], vec![3]),
        (vec!["--start-row", "4"], vec![]),
        (vec!["--start-row", "2", "--end-row", "2"], vec![]),
        (vec!["--end-row", "100"], vec![1, 3]),
    ] {
        let dir = TempDir::new();
        let input = dir.input(format!("{}\n\n{}", request("1"), request("3")));
        let server = Server::new(vec![]);
        let output = spawn_with_args(&input, &dir.job(), &server.url, 0, &args).finish();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let lines = json_lines(&output);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["line"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(server.events().len(), expected.len());
        assert_eq!(
            status(&dir.job())["last_successful_row"],
            serde_json::json!(expected.last())
        );
    }
}

#[test]
fn pipe_range_finishes_without_eof_and_ignores_oversized_unselected_rows() {
    let dir = TempDir::new();
    let server = Server::new(vec![]);
    let mut process = spawn_with_args(
        Path::new("-"),
        &dir.job(),
        &server.url,
        0,
        &["--start-row", "2", "--end-row", "2"],
    );
    let mut stdin = process.stdin.take().unwrap();
    stdin.write_all(&vec![b'x'; 32 * 1024 * 1024 + 1]).unwrap();
    writeln!(stdin, "\n{}", request("2")).unwrap();
    let output = process.finish();
    // An open producer must not prevent completion of a finite range.
    drop(stdin);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(json_lines(&output)[0]["line"], 2);
    assert_eq!(server.events().len(), 1);
    assert_eq!(status(&dir.job())["captured"], 1);
    assert_eq!(status(&dir.job())["source_complete"], false);
    let output = cli().arg("resume").arg(dir.job()).output().unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(server.events().len(), 1);
}

#[test]
fn pipe_resume_keeps_row_numbers_after_consuming_unselected_rows() {
    let dir = TempDir::new();
    let server = Server::new(vec![]);
    let mut process = spawn_with_args(
        Path::new("-"),
        &dir.job(),
        &server.url,
        0,
        &["--start-row", "3", "--end-row", "4"],
    );
    let mut stdin = process.stdin.take().unwrap();
    stdin.write_all(b"not JSON\n\n").unwrap();
    wait_for(|| saved(&dir.job(), "SELECT next_line FROM job") == Some(3));
    assert!(cli()
        .arg("stop")
        .arg(dir.job())
        .output()
        .unwrap()
        .status
        .success());
    let output = process.finish();
    drop(stdin);
    assert_eq!(output.status.code(), Some(1));
    assert!(server.events().is_empty());

    let mut continuation = Process(Some(
        cli()
            .arg("resume")
            .arg(dir.job())
            .args(["--input", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));
    let mut stdin = continuation.stdin.take().unwrap();
    writeln!(stdin, "{}\n{}\nnot JSON", request("3"), request("4")).unwrap();
    let output = continuation.finish();
    drop(stdin);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        json_lines(&output)
            .iter()
            .map(|line| line["line"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [3, 4]
    );
    assert_eq!(status(&dir.job())["last_successful_row"], 4);
    assert_eq!(server.events().len(), 2);
}

#[test]
fn skipping_a_failure_at_the_end_preserves_the_last_successful_row() {
    let dir = TempDir::new();
    let input = dir.input(format!("{}\nnot JSON\nnot JSON\n", request("1")));
    let server = Server::new(vec![]);
    assert_eq!(
        spawn_with_args(&input, &dir.job(), &server.url, 0, &["--end-row", "2"])
            .finish()
            .status
            .code(),
        Some(1)
    );
    let output = cli()
        .arg("resume")
        .arg(dir.job())
        .arg("--skip-failed")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let progress = status(&dir.job());
    assert_eq!(progress["state"], "completed");
    assert_eq!(progress["skipped"], 1);
    assert_eq!(progress["last_successful_row"], 1);
    let output = cli().arg("stop").arg(dir.job()).output().unwrap();
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("last successful row so far: 1"));
    let output = cli().arg("status").arg(dir.job()).output().unwrap();
    assert!(String::from_utf8_lossy(&output.stdout).contains("Last successful row: 1"));
}

#[test]
fn failure_pauses_and_resume_uses_overrides_without_replaying_successes_or_saving_keys() {
    let dir = TempDir::new();
    let input = dir.input(format!(
        "{}\n{}\n{}\n",
        request("1"),
        request("2"),
        request("3")
    ));
    let server = Server::new(vec![Reply::ok(), Reply::error(401)]);
    let output = spawn(&input, &dir.job(), &server.url, 0).finish();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(json_lines(&output).len(), 1);
    assert!(String::from_utf8_lossy(&output.stderr).contains("record 2"));
    assert_eq!(server.events().len(), 2);
    assert_eq!(status(&dir.job())["failed"], 1);
    let replacement = Server::new(vec![]);
    let output = cli()
        .arg("resume")
        .arg(dir.job())
        .args([
            "--server",
            &replacement.url,
            "--api-key",
            "resume-secret-key",
            "--interval-ms",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        replacement
            .events()
            .iter()
            .map(|event| event.state.as_str())
            .collect::<Vec<_>>(),
        ["2", "3"]
    );
    assert!(replacement.events()[0]
        .headers
        .contains("authorization: bearer resume-secret-key"));
    assert_eq!(json_lines(&output)[0]["record"], 2);
    assert_eq!(
        json_lines(&cli().arg("export").arg(dir.job()).output().unwrap()).len(),
        3
    );
    for entry in std::fs::read_dir(dir.job()).unwrap() {
        let bytes = std::fs::read(entry.unwrap().path()).unwrap();
        assert!(!bytes
            .windows(b"resume-secret-key".len())
            .any(|window| window == b"resume-secret-key"));
    }
}

#[test]
fn invalid_responses_are_not_retried_and_explicit_skip_retains_a_failure_exit() {
    for body in [
        "not json",
        r#"{"model":"mock","answers":{"wrong":{"type":"noul","noul":0.75}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
    ] {
        let dir = TempDir::new();
        let input = dir.input(format!("{}\n{}\n", request("1"), request("2")));
        let server = Server::new(vec![Reply {
            body: body.into(),
            ..Reply::ok()
        }]);
        let output = spawn(&input, &dir.job(), &server.url, 0).finish();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_eq!(server.events().len(), 1);
        let output = cli()
            .arg("resume")
            .arg(dir.job())
            .arg("--skip-failed")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(json_lines(&output)[0]["record"], 2);
        let status = status(&dir.job());
        assert_eq!(status["state"], "completed");
        assert_eq!(status["skipped"], 1);
        assert_eq!(status["succeeded"], 1);
    }
}

#[test]
fn retry_exhaustion_never_advances_to_the_next_record() {
    let dir = TempDir::new();
    let input = dir.input(format!("{}\n{}\n", request("1"), request("2")));
    let reply = Reply {
        headers: "retry-after-ms: 1\r\n".into(),
        ..Reply::error(529)
    };
    let server = Server::new(vec![reply; 3]);
    let output = spawn(&input, &dir.job(), &server.url, 0).finish();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(server.events().len(), 3);
    assert!(server.events().iter().all(|event| event.state == "1"));
    assert!(server.events()[2]
        .headers
        .contains("x-typesafe-retry-count: 2"));
    assert_eq!(status(&dir.job())["state"], "paused");
}

#[test]
fn malformed_utf8_and_oversized_input_pause_before_http() {
    let mut oversized = vec![b'x'; 32 * 1024 * 1024 + 1];
    oversized.push(b'\n');
    for bytes in [b"{}\n".to_vec(), vec![0xff, b'\n'], oversized] {
        let dir = TempDir::new();
        let input = dir.input(bytes);
        let server = Server::new(vec![]);
        let output = spawn(&input, &dir.job(), &server.url, 0).finish();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(server.events().is_empty());
        assert_eq!(status(&dir.job())["failed"], 1);
    }
}

#[test]
fn changed_input_and_existing_job_are_rejected_without_replay() {
    let dir = TempDir::new();
    let input = dir.input(request("1"));
    let server = Server::new(vec![]);
    assert!(spawn(&input, &dir.job(), &server.url, 0)
        .finish()
        .status
        .success());
    assert_eq!(
        spawn(&input, &dir.job(), &server.url, 0)
            .finish()
            .status
            .code(),
        Some(1)
    );
    dir.input(request("changed"));
    let output = cli().arg("resume").arg(dir.job()).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("input has changed"));
    assert_eq!(server.events().len(), 1);
    assert_eq!(status(&dir.job())["succeeded"], 1);
}

#[test]
fn idle_pipe_stops_promptly_and_can_accept_remaining_producer_output() {
    let dir = TempDir::new();
    let server = Server::new(vec![]);
    let mut process = spawn(Path::new("-"), &dir.job(), &server.url, 0);
    wait_for(|| saved(&dir.job(), "SELECT COUNT(*) FROM job WHERE state='running'") == Some(1));
    let started = Instant::now();
    assert!(cli()
        .arg("stop")
        .arg(dir.job())
        .output()
        .unwrap()
        .status
        .success());
    // Keep the producer's stdin handle open while the client shuts down.
    let stdin = process.stdin.take().unwrap();
    let output = process.finish();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("last successful row: none"));
    assert!(started.elapsed() < Duration::from_secs(2));
    drop(stdin);
    assert_eq!(status(&dir.job())["state"], "stopped");
    assert_eq!(status(&dir.job())["last_successful_row"], Value::Null);
    assert!(server.events().is_empty());
    let output = cli().arg("resume").arg(dir.job()).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--input -"));
    let mut continuation = Process(Some(
        cli()
            .arg("resume")
            .arg(dir.job())
            .args(["--input", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));
    writeln!(
        continuation.stdin.take().unwrap(),
        "{}",
        request("remaining")
    )
    .unwrap();
    let output = continuation.finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(json_lines(&output).len(), 1);
    assert_eq!(status(&dir.job())["source_complete"], true);
}

#[test]
fn stop_during_pacing_preserves_captured_record_and_lock_prevents_a_second_runner() {
    let dir = TempDir::new();
    let input = dir.input(format!("ignored\n{}\n{}\n", request("1"), request("2")));
    let server = Server::new(vec![]);
    let process = spawn_with_args(
        &input,
        &dir.job(),
        &server.url,
        60000,
        &["--start-row", "2", "--end-row", "3"],
    );
    wait_for(|| {
        saved(
            &dir.job(),
            "SELECT COUNT(*) FROM records WHERE status='succeeded'",
        ) == Some(1)
    });
    assert_eq!(status(&dir.job())["runner_active"], true);
    let competing = cli().arg("resume").arg(dir.job()).output().unwrap();
    assert_eq!(competing.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&competing.stderr).contains("another runner"));
    assert!(cli()
        .arg("stop")
        .arg(dir.job())
        .output()
        .unwrap()
        .status
        .success());
    let output = process.finish();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("last successful row: 2"));
    assert_eq!(server.events().len(), 1);
    let output = cli()
        .arg("resume")
        .arg(dir.job())
        .args(["--interval-ms", "0"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(json_lines(&output)[0]["record"], 2);
    assert_eq!(json_lines(&output)[0]["line"], 3);
    assert_eq!(server.events().len(), 2);
}

#[cfg(unix)]
fn signal(process: &Process, signal: i32) {
    assert_eq!(unsafe { libc::kill(process.id() as i32, signal) }, 0);
}

#[cfg(unix)]
#[test]
fn signals_and_remote_stop_drain_the_active_request_then_stop() {
    for interrupt in [Some(libc::SIGINT), Some(libc::SIGTERM), None] {
        let dir = TempDir::new();
        let input = dir.input(format!("{}\n{}\n", request("1"), request("2")));
        let server = Server::new(vec![Reply::delayed(500)]);
        let process = spawn(&input, &dir.job(), &server.url, 0);
        server.wait_events(1);
        if let Some(interrupt) = interrupt {
            signal(&process, interrupt);
        } else {
            assert!(cli()
                .arg("stop")
                .arg(dir.job())
                .output()
                .unwrap()
                .status
                .success());
        }
        let output = process.finish();
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(server.events().len(), 1);
        let status = status(&dir.job());
        assert_eq!(status["state"], "stopped");
        assert_eq!(status["succeeded"], 1);
        assert_eq!(status["last_successful_row"], 1);
        assert!(String::from_utf8_lossy(&output.stderr).contains("last successful row: 1"));
        assert_eq!(json_lines(&output).len(), 1);
        assert_eq!(
            json_lines(&cli().arg("export").arg(dir.job()).output().unwrap()).len(),
            1
        );
    }
}

#[test]
fn echoed_authentication_keys_are_redacted_before_saving_failure_diagnostics() {
    let dir = TempDir::new();
    let input = dir.input(request("fixture"));
    let secret = "fixture-credential-never-save";
    let reply = Reply {
        body: serde_json::json!({"error":{"message":format!("rejected credential {secret}")}})
            .to_string(),
        ..Reply::error(401)
    };
    let server = Server::new(vec![reply]);
    let output = cli()
        .arg("run")
        .arg(input)
        .arg("--job-dir")
        .arg(dir.job())
        .args(["--server", &server.url, "--api-key", secret])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&output.stderr).contains(secret));
    assert!(String::from_utf8_lossy(&output.stderr).contains("[redacted]"));
    let connection = Connection::open(dir.job().join("job.sqlite3")).unwrap();
    let diagnostic: String = connection
        .query_row("SELECT error FROM records", [], |row| row.get(0))
        .unwrap();
    assert!(!diagnostic.contains(secret));
    assert!(diagnostic.contains("[redacted]"));
    drop(connection);
    for entry in std::fs::read_dir(dir.job()).unwrap() {
        let bytes = std::fs::read(entry.unwrap().path()).unwrap();
        assert!(!bytes
            .windows(secret.len())
            .any(|window| window == secret.as_bytes()));
    }
}

#[test]
fn echoed_keys_are_redacted_before_message_truncation() {
    let secret = "fixture-credential-never-save";
    for body in [
        serde_json::json!({"error":{"message":format!("{}{secret}", "x".repeat(185))}}).to_string(),
        format!("{}{secret}", "x".repeat(185)),
        serde_json::json!({"unknown":format!("{}{secret}", "x".repeat(173))}).to_string(),
    ] {
        let dir = TempDir::new();
        let input = dir.input(request("fixture"));
        let server = Server::new(vec![Reply {
            body,
            ..Reply::error(401)
        }]);
        let output = cli()
            .arg("run")
            .arg(input)
            .arg("--job-dir")
            .arg(dir.job())
            .args(["--server", &server.url, "--api-key", secret])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains(&secret[..15]), "{stderr}");
        let connection = Connection::open(dir.job().join("job.sqlite3")).unwrap();
        let diagnostic: String = connection
            .query_row("SELECT error FROM records", [], |row| row.get(0))
            .unwrap();
        assert!(!diagnostic.contains(&secret[..15]), "{diagnostic}");
    }
}

#[test]
fn overlapping_authentication_keys_do_not_leave_secret_suffixes() {
    let dir = TempDir::new();
    let input = dir.input(request("fixture"));
    let explicit = "fixture-credential";
    let inherited = format!("{explicit}-private-suffix");
    let server = Server::new(vec![Reply {
        body: serde_json::json!({"error":{"message":format!("{explicit} {inherited}")}})
            .to_string(),
        ..Reply::error(401)
    }]);
    let output = cli()
        .arg("run")
        .arg(input)
        .arg("--job-dir")
        .arg(dir.job())
        .args(["--server", &server.url, "--api-key", explicit])
        .env("TYPESAFE_API_KEY", inherited)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("private-suffix"), "{stderr}");
    let connection = Connection::open(dir.job().join("job.sqlite3")).unwrap();
    let diagnostic: String = connection
        .query_row("SELECT error FROM records", [], |row| row.get(0))
        .unwrap();
    assert!(!diagnostic.contains("private-suffix"), "{diagnostic}");
}

#[cfg(unix)]
#[test]
fn second_interrupt_exits_immediately_and_resume_reports_uncertainty() {
    let dir = TempDir::new();
    let input = dir.input(format!("{}\n{}\n", request("1"), request("2")));
    let server = Server::new(vec![Reply::ok(), Reply::delayed(1200)]);
    let process = spawn(&input, &dir.job(), &server.url, 0);
    server.wait_events(2);
    signal(&process, libc::SIGINT);
    wait_for(|| saved(&dir.job(), "SELECT stop_requested FROM job") == Some(1));
    let started = Instant::now();
    signal(&process, libc::SIGINT);
    let output = process.finish();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("last successful row: 1"));
    assert!(started.elapsed() < Duration::from_millis(700));
    assert_eq!(status(&dir.job())["state"], "interrupted");
    let replacement = Server::new(vec![]);
    let output = cli()
        .arg("resume")
        .arg(dir.job())
        .args(["--server", &replacement.url])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("uncertain outcome"));
    assert_eq!(json_lines(&output).len(), 1);
    assert_eq!(json_lines(&output)[0]["line"], 2);
}

#[test]
fn process_crash_preserves_the_unfinished_request_for_recovery() {
    let dir = TempDir::new();
    let input = dir.input(format!("{}\n{}\n", request("1"), request("2")));
    let server = Server::new(vec![Reply::ok(), Reply::delayed(800)]);
    let mut process = spawn(&input, &dir.job(), &server.url, 0);
    server.wait_events(2);
    process.kill().unwrap();
    let _ = process.finish();
    let status = status(&dir.job());
    assert_eq!(status["state"], "interrupted");
    assert_eq!(status["succeeded"], 1);
    assert_eq!(status["unfinished"], 1);
    let replacement = Server::new(vec![]);
    let output = cli()
        .arg("resume")
        .arg(dir.job())
        .args(["--server", &replacement.url])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("uncertain outcome"));
    assert_eq!(json_lines(&output)[0]["record"], 2);
    assert_eq!(replacement.events()[0].state, "2");
    assert_eq!(
        json_lines(&cli().arg("export").arg(dir.job()).output().unwrap()).len(),
        2
    );
}

#[test]
fn usage_errors_remain_code_two_and_help_does_not_expose_credentials() {
    for args in [
        vec!["--start-row", "0"],
        vec!["--end-row", "0"],
        vec!["--start-row", "5", "--end-row", "4"],
        vec!["--end-row", "-1"],
    ] {
        let dir = TempDir::new();
        let output = cli()
            .args(["run", "-", "--job-dir"])
            .arg(dir.job())
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(!dir.job().exists());
    }
    assert_eq!(
        cli().args(["run", "-"]).output().unwrap().status.code(),
        Some(2)
    );
    assert_eq!(
        cli()
            .args(["resume", "job", "--input", "other-file"])
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    for command in ["run", "resume"] {
        let output = cli()
            .args([command, "--help"])
            .env("OPENKIND_API_KEY", "help-secret")
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("help-secret"));
    }
}
