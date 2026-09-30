use super::*;

const REQUEST: &[u8] = b"GET /health HTTP/1.1\r\nhost: localhost\r\n\r\n";

async fn reply(wire: Vec<u8>, limit: usize) -> (H1Pool, Result<FastResponse, FastError>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        while find_double_crlf(&request).is_none() {
            assert!(stream.read_buf(&mut request).await.unwrap() > 0);
        }
        let _ = stream.write_all(&wire).await;
        std::future::pending::<()>().await;
    });
    let pool = H1Pool::new("127.0.0.1", address.port(), None);
    let result = exchange(&pool, REQUEST, limit, Duration::from_secs(1))
        .await
        .expect("framed responses must complete while the socket remains open");
    (pool, result)
}

#[tokio::test]
async fn informational_responses_are_skipped_until_the_final_response() {
    let (_, result) = reply(
        b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 103 Early Hints\r\nlink: </help>\r\n\r\nHTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}".to_vec(),
        100,
    ).await;
    let Ok(response) = result else {
        panic!("expected the final response")
    };
    assert_eq!(response.status, 200);
    assert_eq!(response.body, b"{}");
}

#[tokio::test]
async fn no_body_statuses_complete_without_waiting_for_connection_close() {
    for status in [204, 304] {
        let (_, result) = reply(format!("HTTP/1.1 {status} Empty\r\n\r\n").into_bytes(), 100).await;
        let Ok(response) = result else {
            panic!("expected a bodyless response")
        };
        assert_eq!(response.status, status);
        assert!(response.body.is_empty());
    }
}

#[tokio::test]
async fn oversized_chunk_arithmetic_fails_without_overflow() {
    let wire = format!(
        "HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n1\r\nx\r\n{:x}\r\n",
        usize::MAX
    );
    let (_, result) = reply(wire.into_bytes(), 100).await;
    assert!(matches!(result, Err(FastError::TooLarge)));
}

#[tokio::test]
async fn complete_oversized_heads_and_chunk_lines_are_rejected() {
    let pad = "x".repeat(MAX_HEAD_BYTES + 1);
    for wire in [
        format!("HTTP/1.1 200 OK\r\nx-pad: {pad}\r\ncontent-length: 0\r\n\r\n"),
        format!("HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n1;{pad}\r\nx\r\n0\r\n\r\n"),
        format!("HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n0\r\nx-pad: {pad}\r\n\r\n"),
    ] {
        let (_, result) = reply(wire.into_bytes(), 100).await;
        assert!(
            matches!(result, Err(FastError::Io(error)) if error.kind() == io::ErrorKind::InvalidData)
        );
    }
}

#[tokio::test]
async fn ambiguous_content_lengths_are_rejected() {
    let (_, result) = reply(
        b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\ncontent-length: 0\r\n\r\n{}".to_vec(),
        100,
    )
    .await;
    assert!(
        matches!(result, Err(FastError::Io(error)) if error.kind() == io::ErrorKind::InvalidData)
    );
}

#[tokio::test]
async fn connections_with_unconsumed_bytes_are_not_reused() {
    let (pool, result) = reply(
        b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}unexpected".to_vec(),
        100,
    )
    .await;
    assert!(result.is_ok());
    assert!(pool.idle.lock().await.is_empty());
}

#[tokio::test]
async fn http10_requires_explicit_keep_alive_before_reuse() {
    let (pool, result) = reply(
        b"HTTP/1.0 200 OK\r\ncontent-length: 2\r\n\r\n{}".to_vec(),
        100,
    )
    .await;
    assert!(result.is_ok());
    assert!(pool.idle.lock().await.is_empty());
}

#[test]
fn request_encoding_emits_one_owned_host_and_json_content_type() {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("host", "wrong.test".parse().unwrap());
    headers.insert("content-type", "text/plain".parse().unwrap());
    let wire = encode_request(
        "POST",
        "/v1/systemone",
        "localhost",
        &headers,
        Some(b"{}"),
        None,
    );
    let text = String::from_utf8(wire).unwrap();
    assert_eq!(text.matches("host:").count(), 1);
    assert_eq!(text.matches("content-type:").count(), 1);
    assert!(!text.contains("wrong.test"));
    assert!(!text.contains("text/plain"));
}

#[tokio::test]
async fn ipv6_urls_use_a_socket_host_without_brackets_and_a_bracketed_authority() {
    let Ok(listener) = tokio::net::TcpListener::bind("[::1]:0").await else {
        return;
    };
    let address = listener.local_addr().unwrap();
    let url = reqwest::Url::parse(&format!("http://{address}")).unwrap();
    let pool = H1Pool::new(url.host_str().unwrap(), address.port(), None);
    assert_eq!(pool.authority(), address.to_string());
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        while find_double_crlf(&request).is_none() {
            assert!(stream.read_buf(&mut request).await.unwrap() > 0);
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
            .await
            .unwrap();
    });
    let wire = encode_request(
        "GET",
        "/health",
        pool.authority(),
        &reqwest::header::HeaderMap::new(),
        None,
        None,
    );
    let result = exchange(&pool, &wire, 100, Duration::from_secs(1))
        .await
        .unwrap();
    assert!(
        result.is_ok(),
        "an IPv6 URL must establish a loopback connection"
    );
    server.await.unwrap();
}
