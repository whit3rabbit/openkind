//! Helpers shared by the daemon integration tests.

/// Reserve an ephemeral loopback port and release it for the daemon to bind.
///
/// Another process can claim the port in the gap; the callers tolerate that
/// by waiting for the daemon's health check rather than assuming success.
pub fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}
