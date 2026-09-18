//! Build script for `openpick-proto`.
//!
//! Invokes `tonic-prost-build` to compile `proto/openpick.proto` into Rust structs
//! and gRPC client/server stubs with byte field mappings.

use std::io::Result;

fn main() -> Result<()> {
    let proto = "proto/openpick.proto";
    println!("cargo:rerun-if-changed={proto}");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let mut config = prost_build::Config::new();
    config.bytes(["."]);

    // tonic-prost-build generates the gRPC server/client stubs in
    // addition to the message types.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_with_config(config, &[proto], &["proto"])?;
    Ok(())
}
