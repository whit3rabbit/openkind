use std::io::Write;

use anyhow::{bail, Context, Result};
use ring::rand::{SecureRandom, SystemRandom};

pub(crate) fn cmd_keygen() -> Result<()> {
    let key = generate()?;
    writeln!(std::io::stdout().lock(), "{key}").context("write generated API key")
}

fn generate() -> Result<String> {
    let mut bytes = [0u8; 32];
    // API keys need OS entropy, never the deterministic benchmark RNG.
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("OS random source unavailable for API key generation"))?;
    let mut key = String::with_capacity(67);
    key.push_str("ok_");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        key.push(HEX[(byte >> 4) as usize] as char);
        key.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(key)
}

pub(crate) fn validate(key: Option<&str>) -> Result<()> {
    if let Some(key) = key {
        if key.is_empty() || !key.bytes().all(|byte| byte.is_ascii_graphic()) {
            // Do not put the supplied secret in diagnostic output.
            bail!("API key must be non-empty visible ASCII without whitespace; omit it to disable authentication");
        }
    }
    Ok(())
}
