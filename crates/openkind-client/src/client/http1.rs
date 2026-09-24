//! Lean HTTP/1.1 keep-alive transport for plain-`http` URLs.
//!
//! reqwest mediates every request through a per-connection dispatch task,
//! which adds a task handoff (and its scheduler wakeup) to each direction
//! of every round trip. On loopback daemons that overhead is a large share
//! of request latency. This module speaks HTTP/1.1 directly on pooled
//! `TcpStream`s from the caller's own task: the same wire bytes and
//! keep-alive semantics with one fewer cross-thread wakeup per direction.
//!
//! Scope and fallbacks, all preserving reqwest behavior where it differs:
//! - plain `http://` URLs only; `https://` always uses the reqwest client.
//! - disabled when the caller supplied their own `reqwest::Client` or when
//!   proxy environment variables are set (reqwest honors those; this
//!   transport must not bypass a configured proxy).
//! - a 3xx response with a `Location` header is reported back to the caller
//!   so the full logical request re-runs through reqwest and follows the
//!   redirect chain exactly as before.

use std::collections::VecDeque;
use std::io;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::error::Error;

/// Cap on the response head (status line + headers) this transport will
/// buffer before giving up, so a hostile server cannot grow memory without
/// bound.
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// Largest single buffer growth step while scanning for the head end.
const READ_CHUNK: usize = 8 * 1024;

/// Upper bound on idle keep-alive connections retained per authority.
const MAX_IDLE_CONNECTIONS: usize = 8;

/// A parsed minimal response: everything the SDK layer needs after a call.
pub(crate) struct FastResponse {
    /// HTTP status code.
    pub(crate) status: u16,
    /// Header (name, value) pairs with names lowercased.
    headers: Vec<(String, String)>,
    /// Decoded body bytes.
    pub(crate) body: Vec<u8>,
}

impl FastResponse {
    /// First header value for a lowercase name, trimmed.
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.trim())
    }
}

/// Internal transport failure, before HTTP-level error mapping.
pub(crate) enum FastError {
    /// Network or protocol failure; the connection must not be reused.
    Io(io::Error),
    /// A body exceeded the client's fixed memory-safety limit.
    TooLarge,
    /// Redirect response — re-run the logical request through reqwest.
    Redirect,
}

/// One reusable connection with its read-ahead buffer.
struct Connection {
    stream: TcpStream,
    /// Buffered bytes; `start` is the first unconsumed byte.
    buf: Vec<u8>,
    start: usize,
}

impl Connection {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            buf: Vec::with_capacity(READ_CHUNK),
            start: 0,
        }
    }

    fn buffered(&self) -> &[u8] {
        &self.buf[self.start..]
    }

    fn compact(&mut self) {
        if self.start > 0 {
            self.buf.drain(..self.start);
            self.start = 0;
        }
    }

    /// Read more bytes into the buffer; returns how many arrived. Zero
    /// means the peer closed the stream.
    async fn fill(&mut self) -> io::Result<usize> {
        self.compact();
        if self.buf.len() == self.buf.capacity() {
            self.buf.reserve(READ_CHUNK);
        }
        self.stream.read_buf(&mut self.buf).await
    }

    /// Ensure a full response head (`\r\n\r\n` terminated) is buffered;
    /// returns the head length excluding the terminator.
    async fn read_head(&mut self) -> io::Result<usize> {
        loop {
            if let Some(pos) = find_double_crlf(self.buffered()) {
                return Ok(pos);
            }
            if self.buf.len() - self.start > MAX_HEAD_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "HTTP response head exceeded 64 KiB",
                ));
            }
            if self.fill().await? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed before response head was complete",
                ));
            }
        }
    }
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Pool of keep-alive connections to one authority.
///
/// Only the idle list is lock-protected, and never across an exchange:
/// concurrent callers each take a connection (opening a fresh one when the
/// idle list is empty), so parallel requests get parallel connections just
/// like the reqwest pool they replace.
pub(crate) struct H1Pool {
    authority: String,
    host: String,
    port: u16,
    connect_timeout: Option<Duration>,
    idle: tokio::sync::Mutex<VecDeque<Connection>>,
}

impl H1Pool {
    /// Build a pool for `host:port`. Connections are created lazily.
    pub(crate) fn new(host: &str, port: u16, connect_timeout: Option<Duration>) -> Self {
        let authority = if port == 80 {
            host.to_string()
        } else {
            format!("{host}:{port}")
        };
        Self {
            authority,
            host: host.to_string(),
            port,
            connect_timeout,
            idle: tokio::sync::Mutex::new(VecDeque::new()),
        }
    }

    /// Authority string for the Host header (`host` or `host:port`).
    pub(crate) fn authority(&self) -> &str {
        &self.authority
    }

    /// Take a live idle connection, or connect a fresh one.
    ///
    /// Idle entries are probed non-blockingly first: a connection the
    /// server has since closed (or that carries unexpected bytes) is
    /// discarded, so a stale keep-alive surfaces as a fresh connect instead
    /// of a mid-request failure.
    async fn acquire(&self) -> io::Result<Connection> {
        loop {
            let candidate = { self.idle.lock().await.pop_back() };
            let Some(mut conn) = candidate else { break };
            let mut probe = [0u8; 16];
            match conn.stream.try_read(&mut probe) {
                Ok(0) => continue,  // closed by peer
                Ok(_n) => continue, // unexpected pre-response bytes: stale
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    conn.buf.clear();
                    conn.start = 0;
                    return Ok(conn);
                }
                Err(_) => continue,
            }
        }
        let connect = TcpStream::connect((self.host.as_str(), self.port));
        let stream = match self.connect_timeout {
            Some(timeout) => tokio::time::timeout(timeout, connect).await.map_err(|_| {
                io::Error::new(io::ErrorKind::TimedOut, "connect timeout elapsed")
            })??,
            None => connect.await?,
        };
        stream.set_nodelay(true)?;
        Ok(Connection::new(stream))
    }

    async fn release(&self, conn: Connection) {
        let mut idle = self.idle.lock().await;
        if idle.len() < MAX_IDLE_CONNECTIONS {
            idle.push_back(conn);
        }
    }
}

/// Whether any proxy environment variable is configured; the fast path must
/// not silently bypass a proxy reqwest would otherwise use.
pub(crate) fn proxy_env_present() -> bool {
    [
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    .iter()
    .any(|name| std::env::var_os(name).is_some())
}

/// Serialize one request into wire bytes.
pub(crate) fn encode_request(
    method: &str,
    target: &str,
    authority: &str,
    headers: &reqwest::header::HeaderMap,
    body: Option<&[u8]>,
    retry_count: Option<u32>,
) -> Vec<u8> {
    // Request line, Host, framing headers, then the merged header set.
    let mut head = format!("{method} {target} HTTP/1.1\r\nhost: {authority}\r\n");
    if let Some(body) = body {
        head.push_str("content-type: application/json\r\n");
        head.push_str("content-length: ");
        head.push_str(&body.len().to_string());
        head.push_str("\r\n");
    }
    if let Some(retries) = retry_count {
        head.push_str("x-typesafe-retry-count: ");
        head.push_str(&retries.to_string());
        head.push_str("\r\n");
    }
    let mut bytes = head.into_bytes();
    for (name, value) in headers {
        // Content-Length is computed above; skip any caller-supplied copy.
        if name.as_str() == "content-length" {
            continue;
        }
        bytes.extend_from_slice(name.as_str().as_bytes());
        bytes.extend_from_slice(b": ");
        bytes.extend_from_slice(value.as_bytes());
        bytes.extend_from_slice(b"\r\n");
    }
    bytes.extend_from_slice(b"\r\n");
    if let Some(body) = body {
        bytes.extend_from_slice(body);
    }
    bytes
}

/// Execute one request/response exchange on a pooled connection.
///
/// The outer `Err` is a whole-attempt timeout (already mapped to
/// [`Error::Timeout`]); the inner result is the exchange outcome.
pub(crate) async fn exchange(
    pool: &H1Pool,
    request: &[u8],
    max_body: usize,
    timeout: Duration,
) -> Result<Result<FastResponse, FastError>, Error> {
    match tokio::time::timeout(timeout, exchange_inner(pool, request, max_body)).await {
        Ok(inner) => Ok(inner),
        Err(_) => Err(Error::Timeout { timeout }),
    }
}

async fn exchange_inner(
    pool: &H1Pool,
    request: &[u8],
    max_body: usize,
) -> Result<FastResponse, FastError> {
    // A connect timeout surfaces with kind `TimedOut` and is classified
    // into `Error::Timeout` by `classify_fast_error`.
    let mut conn = pool.acquire().await.map_err(FastError::Io)?;
    let result = exchange_on(&mut conn, request, max_body).await;
    if result.as_ref().is_ok_and(response_keep_alive) {
        pool.release(conn).await;
    }
    result
}

async fn exchange_on(
    conn: &mut Connection,
    request: &[u8],
    max_body: usize,
) -> Result<FastResponse, FastError> {
    if let Err(e) = conn.stream.write_all(request).await {
        return Err(FastError::Io(e));
    }
    if let Err(e) = conn.stream.flush().await {
        return Err(FastError::Io(e));
    }

    let head_len = conn.read_head().await.map_err(FastError::Io)?;
    let head = conn.buffered()[..head_len].to_vec();
    conn.start += head_len + 4;

    let (status, headers) = parse_head(&head).map_err(FastError::Io)?;

    if (300..400).contains(&status) && header_lookup(&headers, "location").is_some() {
        // The body is not consumed; the connection is dropped rather than
        // pooled, and the caller re-runs the request through reqwest.
        return Err(FastError::Redirect);
    }

    let chunked = header_lookup(&headers, "transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
    let content_length =
        header_lookup(&headers, "content-length").and_then(|v| v.trim().parse::<usize>().ok());

    let body = if chunked {
        read_chunked_body(conn, max_body).await?
    } else if let Some(length) = content_length {
        if length > max_body {
            // The connection is abandoned, not drained.
            return Err(FastError::TooLarge);
        }
        read_exact_body(conn, length).await?
    } else {
        // No framing headers: the body runs to connection close, and the
        // connection cannot be reused.
        read_until_eof(conn, max_body).await?
    };

    Ok(FastResponse {
        status,
        headers,
        body,
    })
}

fn response_keep_alive(response: &FastResponse) -> bool {
    !response
        .header("connection")
        .is_some_and(|v| v.to_ascii_lowercase().contains("close"))
}

fn header_lookup<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

/// Parse a status line plus header block.
fn parse_head(head: &[u8]) -> io::Result<(u16, Vec<(String, String)>)> {
    let text = std::str::from_utf8(head)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 response head"))?;
    let mut lines = text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty response head"))?;
    let mut parts = status_line.splitn(3, ' ');
    let version = parts.next().unwrap_or_default();
    if !version.starts_with("HTTP/1.") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported HTTP version in `{status_line}`"),
        ));
    }
    let status: u16 = parts.next().unwrap_or_default().parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("malformed status line `{status_line}`"),
        )
    })?;
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    Ok((status, headers))
}

async fn read_exact_body(conn: &mut Connection, length: usize) -> Result<Vec<u8>, FastError> {
    let mut body = Vec::with_capacity(length.min(1 << 20));
    while body.len() < length {
        if conn.buffered().is_empty() {
            match conn.fill().await {
                Ok(0) => {
                    return Err(FastError::Io(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "connection closed inside response body",
                    )))
                }
                Ok(_) => continue,
                Err(e) => return Err(FastError::Io(e)),
            }
        }
        let take = (length - body.len()).min(conn.buffered().len());
        body.extend_from_slice(&conn.buffered()[..take]);
        conn.start += take;
    }
    Ok(body)
}

async fn read_until_eof(conn: &mut Connection, max_body: usize) -> Result<Vec<u8>, FastError> {
    let mut body = Vec::new();
    loop {
        let available = conn.buffered().len();
        if available > 0 {
            if body.len() + available > max_body {
                return Err(FastError::TooLarge);
            }
            body.extend_from_slice(conn.buffered());
            conn.start += available;
        }
        match conn.fill().await {
            Ok(0) => return Ok(body),
            Ok(_) => {}
            Err(e) => return Err(FastError::Io(e)),
        }
    }
}

/// Decode a `transfer-encoding: chunked` body.
async fn read_chunked_body(conn: &mut Connection, max_body: usize) -> Result<Vec<u8>, FastError> {
    let mut body = Vec::new();
    loop {
        // Chunk-size line, e.g. `1a;name=val`.
        let line = match read_line(conn).await? {
            Some(line) => line,
            None => {
                return Err(FastError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed before chunk size",
                )))
            }
        };
        let size_text = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16).map_err(|_| {
            FastError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed chunk size `{size_text}`"),
            ))
        })?;
        if size == 0 {
            // Trailer section: consume lines until the empty line.
            loop {
                match read_line(conn).await? {
                    Some(trailer) if trailer.is_empty() => return Ok(body),
                    Some(_) => continue,
                    None => {
                        return Err(FastError::Io(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "connection closed inside chunked trailers",
                        )))
                    }
                }
            }
        }
        if body.len() + size > max_body {
            return Err(FastError::TooLarge);
        }
        let chunk = read_exact_body(conn, size).await?;
        body.extend_from_slice(&chunk);
        // Trailing CRLF after each chunk.
        let crlf = read_exact_body(conn, 2).await?;
        if crlf != b"\r\n" {
            return Err(FastError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing CRLF after chunk",
            )));
        }
    }
}

/// Read one CRLF-terminated line (returned without the terminator), or
/// `None` on a clean EOF at a line boundary.
async fn read_line(conn: &mut Connection) -> Result<Option<String>, FastError> {
    loop {
        if let Some(pos) = conn.buffered().iter().position(|&b| b == b'\n') {
            let mut line_end = pos;
            if line_end > 0 && conn.buffered()[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            let line = String::from_utf8_lossy(&conn.buffered()[..line_end]).into_owned();
            conn.start += pos + 1;
            return Ok(Some(line));
        }
        match conn.fill().await {
            Ok(0) => return Ok(None),
            Ok(_) => {}
            Err(e) => return Err(FastError::Io(e)),
        }
    }
}

/// Map a fast-transport failure into the SDK error taxonomy.
pub(crate) fn classify_fast_error(error: FastError, timeout: Duration, max_body: usize) -> Error {
    match error {
        FastError::Redirect => {
            Error::Config("redirect handling requires the reqwest transport".into())
        }
        FastError::TooLarge => Error::ResponseTooLarge { limit: max_body },
        FastError::Io(e) if e.kind() == io::ErrorKind::TimedOut => Error::Timeout { timeout },
        FastError::Io(e) => Error::Connection(Box::new(e)),
    }
}
