//! Demand-driven input: the reader never queues another logical record.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

use crate::inspect::MAX_CLI_INPUT_BYTES;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileSource {
    pub path: PathBuf,
    pub identity: String,
    pub digest: String,
}

fn identity(file: &File) -> Result<String> {
    let metadata = file.metadata()?;
    anyhow::ensure!(
        metadata.is_file(),
        "batch input must be a regular file or '-' for stdin"
    );
    let modified = metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?;
    let value = format!("{}:{}", metadata.len(), modified.as_nanos());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(format!("{}:{}:{value}", metadata.dev(), metadata.ino()))
    }
    #[cfg(not(unix))]
    Ok(value)
}

fn fingerprint_file(path: &Path) -> Result<(File, FileSource)> {
    let path = path.canonicalize().context("resolve batch input path")?;
    let mut file = File::open(&path).context("open batch input")?;
    let before = identity(&file)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    anyhow::ensure!(
        before == identity(&file)?,
        "batch input changed during fingerprinting"
    );
    file.seek(SeekFrom::Start(0))?;
    Ok((
        file,
        FileSource {
            path,
            identity: before,
            digest: format!("{:x}", digest.finalize()),
        },
    ))
}

pub async fn open_file(path: PathBuf) -> Result<(File, FileSource)> {
    file_worker(move || fingerprint_file(&path)).await
}

pub async fn reopen_file(source: FileSource, offset: u64) -> Result<File> {
    file_worker(move || {
        let (mut file, current) = fingerprint_file(&source.path)?;
        anyhow::ensure!(
            current.identity == source.identity && current.digest == source.digest,
            "batch input has changed; start a new job with the changed file"
        );
        file.seek(SeekFrom::Start(offset))?;
        Ok(file)
    })
    .await
}

async fn file_worker<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let (reply, done) = oneshot::channel();
    // Hashing a large source must not block signal handling or runtime exit.
    std::thread::Builder::new()
        .name("batch-fingerprint".into())
        .spawn(move || {
            let _ = reply.send(work());
        })?;
    done.await.context("batch fingerprint worker stopped")?
}

pub struct Line {
    pub bytes: Vec<u8>,
    pub oversized: bool,
    pub offset: u64,
    pub number: u64,
}

pub struct Input {
    demand: mpsc::Sender<oneshot::Sender<Result<Option<Line>>>>,
}

impl Input {
    pub fn file(file: File, source: FileSource, offset: u64, number: u64) -> Result<Self> {
        Self::spawn(move |demands| {
            read_loop(BufReader::new(file), offset, number, demands, |reader| {
                anyhow::ensure!(
                    identity(reader.get_ref())? == source.identity,
                    "batch input changed while the job was running"
                );
                Ok(())
            });
        })
    }

    pub fn stdin(offset: u64, number: u64) -> Result<Self> {
        Self::spawn(move |demands| {
            read_loop(std::io::stdin().lock(), offset, number, demands, |_| Ok(()));
        })
    }

    fn spawn(
        worker: impl FnOnce(mpsc::Receiver<oneshot::Sender<Result<Option<Line>>>>) + Send + 'static,
    ) -> Result<Self> {
        let (demand, demands) = mpsc::channel();
        // An OS thread can stay blocked on an idle producer without holding
        // the Tokio runtime open on exit. Only requested complete lines are
        // captured; buffered/partial stdin is not a durable producer queue.
        std::thread::Builder::new()
            .name("batch-input".into())
            .spawn(move || worker(demands))?;
        Ok(Self { demand })
    }

    pub async fn next(&self) -> Result<Option<Line>> {
        let (sender, receiver) = oneshot::channel();
        self.demand
            .send(sender)
            .context("batch input reader stopped")?;
        receiver.await.context("batch input reader stopped")?
    }
}

fn read_loop<R: BufRead>(
    mut reader: R,
    mut offset: u64,
    mut number: u64,
    demands: mpsc::Receiver<oneshot::Sender<Result<Option<Line>>>>,
    check: impl Fn(&R) -> Result<()>,
) {
    for reply in demands {
        let result = check(&reader).and_then(|()| read_line(&mut reader, &mut offset, number));
        let done = !matches!(result, Ok(Some(_)));
        if reply.send(result).is_err() || done {
            break;
        }
        number = number.saturating_add(1);
    }
}

fn read_line(reader: &mut impl BufRead, offset: &mut u64, number: u64) -> Result<Option<Line>> {
    let mut bytes = Vec::new();
    let mut oversized = false;
    let mut consumed = false;
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        let newline = buffer.iter().position(|&byte| byte == b'\n');
        let count = newline.map_or(buffer.len(), |index| index + 1);
        let content = newline.unwrap_or(count);
        // Keep one extra byte so a terminal CR does not consume the payload
        // limit, including when CRLF straddles reader buffers.
        let remaining = (MAX_CLI_INPUT_BYTES as usize + 1).saturating_sub(bytes.len());
        bytes.extend_from_slice(&buffer[..content.min(remaining)]);
        oversized |= content > remaining;
        *offset = offset
            .checked_add(count as u64)
            .context("batch input offset overflow")?;
        reader.consume(count);
        consumed = true;
        if newline.is_some() {
            break;
        }
    }
    if !consumed {
        return Ok(None);
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    oversized |= bytes.len() > MAX_CLI_INPUT_BYTES as usize;
    bytes.truncate(MAX_CLI_INPUT_BYTES as usize);
    Ok(Some(Line {
        bytes,
        oversized,
        offset: *offset,
        number,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_lines_preserve_framing_and_resume_offsets() {
        let mut reader = BufReader::with_capacity(2, std::io::Cursor::new(b"a\r\n\nlast"));
        let mut offset = 0;
        let a = read_line(&mut reader, &mut offset, 1).unwrap().unwrap();
        assert_eq!(a.bytes, b"a");
        assert_eq!(a.offset, 3);
        assert!(read_line(&mut reader, &mut offset, 2)
            .unwrap()
            .unwrap()
            .bytes
            .is_empty());
        assert_eq!(
            read_line(&mut reader, &mut offset, 3)
                .unwrap()
                .unwrap()
                .bytes,
            b"last"
        );
        assert_eq!(offset, 8);
        assert!(read_line(&mut reader, &mut offset, 4).unwrap().is_none());
    }

    #[test]
    fn oversized_line_is_drained_without_retaining_an_unbounded_buffer() {
        let data = vec![b'x'; MAX_CLI_INPUT_BYTES as usize + 1];
        let mut reader = std::io::Cursor::new(data);
        let line = read_line(&mut reader, &mut 0, 1).unwrap().unwrap();
        assert!(line.oversized);
        assert_eq!(line.bytes.len(), MAX_CLI_INPUT_BYTES as usize);
        assert_eq!(line.offset, MAX_CLI_INPUT_BYTES + 1);
    }

    #[test]
    fn exact_payload_limit_accepts_lf_crlf_and_unterminated_lines() {
        for ending in [b"\n".as_slice(), b"\r\n".as_slice(), b"".as_slice()] {
            let mut data = vec![b'x'; MAX_CLI_INPUT_BYTES as usize];
            data.extend_from_slice(ending);
            let mut reader = BufReader::with_capacity(8191, std::io::Cursor::new(data));
            let line = read_line(&mut reader, &mut 0, 1).unwrap().unwrap();
            assert!(!line.oversized);
            assert_eq!(line.bytes.len(), MAX_CLI_INPUT_BYTES as usize);
        }
    }
}
