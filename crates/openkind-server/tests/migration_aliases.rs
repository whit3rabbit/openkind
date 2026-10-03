use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use openkind_proto::openkind::question::Kind as PbQKind;
use openkind_proto::openkind::state::Value as PbStateValue;
use openkind_proto::openkind::system_one_client::SystemOneClient;
use openkind_proto::openkind::NoulQuestion as PbNoul;
use openkind_proto::openkind::Question as PbQuestion;
use openkind_proto::openkind::State as PbState;
use openkind_proto::openkind::SystemOneRequest as PbRequest;

struct TestServer {
    child: std::process::Child,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn noul_q() -> PbQuestion {
    PbQuestion {
        kind: Some(PbQKind::Noul(PbNoul {
            instructions_json: serde_json::to_vec(&serde_json::json!("?")).unwrap().into(),
            criteria: None,
        })),
    }
}

fn clean_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_openkindd"));
    for (key, _) in std::env::vars() {
        if key.starts_with("OPENKIND_")
            || key.starts_with("OPENDECISION_")
            || key.starts_with("OPENPICK_")
            || key == "TYPESAFE_API_KEY"
        {
            command.env_remove(&key);
        }
    }
    command
}

async fn wait_for_health(client: &reqwest::Client, base_url: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok(resp) = client.get(format!("{base_url}/health")).send().await {
            if resp.status().is_success() {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server at {base_url} did not become ready within deadline");
}

fn startup_error_with_env(envs: &[(&str, &str)], extra_args: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let mut command = clean_command();
    for (k, v) in envs {
        command.env(k, v);
    }
    command.args(extra_args).arg("--models-dir").arg(dir.path());
    let output = command.output().expect("execute openkindd binary");
    assert!(
        !output.status.success(),
        "daemon should fail startup for conflicting/invalid configuration"
    );
    String::from_utf8(output.stderr).unwrap()
}

#[tokio::test]
async fn opendecision_migration_aliases_enable_auth_and_bind_listeners() {
    let http_port = free_port();
    let http_addr = format!("127.0.0.1:{http_port}");
    let secret = "previously-secret-key-12345";
    let dir = tempfile::tempdir().unwrap();

    let mut command = clean_command();
    command.env("OPENDECISION_API_KEY", secret);
    command.env("OPENDECISION_HTTP_ADDR", &http_addr);
    command.env("OPENDECISION_GRPC_ADDR", "0");
    command.arg("--models-dir").arg(dir.path());

    let child = command
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let _guard = TestServer { child };

    let client = reqwest::Client::new();
    let base_url = format!("http://{http_addr}");
    wait_for_health(&client, &base_url).await;

    let payload = serde_json::json!({
        "state": "test state",
        "model": "mock",
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": "test question"
            }
        }
    });

    // 1. Unauthenticated request must return 401 Unauthorized
    let unauth_resp = client
        .post(format!("{base_url}/v1/systemone"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(unauth_resp.status(), reqwest::StatusCode::UNAUTHORIZED);

    // 2. Request with invalid token must return 401 Unauthorized
    let wrong_resp = client
        .post(format!("{base_url}/v1/systemone"))
        .header("Authorization", "Bearer invalid-token")
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(wrong_resp.status(), reqwest::StatusCode::UNAUTHORIZED);

    // 3. Request with valid token must return 200 OK
    let ok_resp = client
        .post(format!("{base_url}/v1/systemone"))
        .header("Authorization", format!("Bearer {secret}"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(ok_resp.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn opendecision_grpc_migration_alias_enforces_auth() {
    let http_port = free_port();
    let grpc_port = free_port();
    let http_addr = format!("127.0.0.1:{http_port}");
    let grpc_addr = format!("127.0.0.1:{grpc_port}");
    let secret = "opendecision-grpc-secret-token";
    let dir = tempfile::tempdir().unwrap();

    let mut command = clean_command();
    command.env("OPENDECISION_API_KEY", secret);
    command.env("OPENDECISION_HTTP_ADDR", &http_addr);
    command.env("OPENDECISION_GRPC_ADDR", &grpc_addr);
    command.arg("--models-dir").arg(dir.path());

    let child = command
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let _guard = TestServer { child };

    let client = reqwest::Client::new();
    wait_for_health(&client, &format!("http://{http_addr}")).await;

    let grpc_url = format!("http://{grpc_addr}");
    let mut grpc_client = SystemOneClient::connect(grpc_url).await.unwrap();

    let mut questions = HashMap::default();
    questions.insert("q".to_string(), noul_q());
    let pb_req = PbRequest {
        state: Some(PbState {
            value: Some(PbStateValue::Text("test".into())),
        }),
        model: "mock".into(),
        questions,
    };

    // Unauthenticated gRPC call fails with Code::Unauthenticated
    let err = grpc_client.evaluate(pb_req.clone()).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::Unauthenticated);

    // Authenticated gRPC call succeeds
    let mut authed_req = tonic::Request::new(pb_req);
    authed_req
        .metadata_mut()
        .insert("authorization", format!("Bearer {secret}").parse().unwrap());
    let resp = grpc_client.evaluate(authed_req).await.unwrap();
    assert!(resp.into_inner().answers.contains_key("q"));
}

#[test]
fn conflicting_api_keys_fail_startup_without_leaking_secrets() {
    let s1 = "top-secret-current-token";
    let s2 = "top-secret-legacy-token";
    let stderr = startup_error_with_env(
        &[("OPENKIND_API_KEY", s1), ("OPENDECISION_API_KEY", s2)],
        &["--grpc-addr", "0"],
    );
    assert!(
        stderr.contains(
            "conflicting values for OPENKIND_API_KEY and deprecated OPENDECISION_API_KEY"
        ),
        "{stderr}"
    );
    assert!(!stderr.contains(s1), "stderr leaked current token");
    assert!(!stderr.contains(s2), "stderr leaked legacy token");
}

#[test]
fn conflicting_opendecision_and_openpick_api_keys_fail_startup() {
    let s1 = "opendecision-secret";
    let s2 = "openpick-secret";
    let stderr = startup_error_with_env(
        &[("OPENDECISION_API_KEY", s1), ("OPENPICK_API_KEY", s2)],
        &["--grpc-addr", "0"],
    );
    assert!(
        stderr.contains(
            "conflicting values for OPENDECISION_API_KEY and deprecated OPENPICK_API_KEY"
        ),
        "{stderr}"
    );
    assert!(!stderr.contains(s1));
    assert!(!stderr.contains(s2));
}

#[test]
fn conflicting_cli_and_opendecision_api_key_fail_startup() {
    let s1 = "cli-token";
    let s2 = "env-token";
    let stderr = startup_error_with_env(
        &[("OPENDECISION_API_KEY", s2)],
        &["--api-key", s1, "--grpc-addr", "0"],
    );
    assert!(
        stderr.contains(
            "conflicting values for OPENKIND_API_KEY and deprecated OPENDECISION_API_KEY"
        ),
        "{stderr}"
    );
    assert!(!stderr.contains(s1));
    assert!(!stderr.contains(s2));
}

#[test]
fn conflicting_http_addrs_fail_startup() {
    let stderr = startup_error_with_env(
        &[
            ("OPENKIND_HTTP_ADDR", "127.0.0.1:18080"),
            ("OPENDECISION_HTTP_ADDR", "127.0.0.1:18081"),
        ],
        &["--grpc-addr", "0"],
    );
    assert!(
        stderr.contains(
            "conflicting values for OPENKIND_HTTP_ADDR and deprecated OPENDECISION_HTTP_ADDR"
        ),
        "{stderr}"
    );
}

#[test]
fn conflicting_grpc_addrs_fail_startup() {
    let stderr = startup_error_with_env(
        &[
            ("OPENKIND_GRPC_ADDR", "0"),
            ("OPENDECISION_GRPC_ADDR", "127.0.0.1:19090"),
        ],
        &[],
    );
    assert!(
        stderr.contains(
            "conflicting values for OPENKIND_GRPC_ADDR and deprecated OPENDECISION_GRPC_ADDR"
        ),
        "{stderr}"
    );
}

#[test]
fn conflicting_typesafe_and_opendecision_api_keys_fail_startup() {
    let s1 = "opendecision-secret";
    let s2 = "typesafe-secret";
    let stderr = startup_error_with_env(
        &[("OPENDECISION_API_KEY", s1), ("TYPESAFE_API_KEY", s2)],
        &["--grpc-addr", "0"],
    );
    assert!(
        stderr.contains("conflicting values for OPENDECISION_API_KEY and TYPESAFE_API_KEY"),
        "{stderr}"
    );
    assert!(!stderr.contains(s1));
    assert!(!stderr.contains(s2));
}

#[tokio::test]
async fn openpick_migration_aliases_still_work() {
    let http_port = free_port();
    let http_addr = format!("127.0.0.1:{http_port}");
    let secret = "openpick-secret-token-xyz";
    let dir = tempfile::tempdir().unwrap();

    let mut command = clean_command();
    command.env("OPENPICK_API_KEY", secret);
    command.env("OPENPICK_HTTP_ADDR", &http_addr);
    command.env("OPENPICK_GRPC_ADDR", "0");
    command.arg("--models-dir").arg(dir.path());

    let child = command
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let _guard = TestServer { child };

    let client = reqwest::Client::new();
    let base_url = format!("http://{http_addr}");
    wait_for_health(&client, &base_url).await;

    let payload = serde_json::json!({
        "state": "test state",
        "model": "mock",
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": "test question"
            }
        }
    });

    let unauth_resp = client
        .post(format!("{base_url}/v1/systemone"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(unauth_resp.status(), reqwest::StatusCode::UNAUTHORIZED);

    let ok_resp = client
        .post(format!("{base_url}/v1/systemone"))
        .header("Authorization", format!("Bearer {secret}"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(ok_resp.status(), reqwest::StatusCode::OK);
}
