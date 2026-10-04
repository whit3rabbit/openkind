//! Sequential evaluation jobs with a durable local recovery boundary.

mod input;
mod store;

use std::future::Future;
use std::io::Write;
use std::time::Duration;

use anyhow::{Context, Result};
use openkind_client::Client;

use crate::args::BatchCommands;
use crate::inspect::parse_and_validate_request;
use crate::output::terminal_safe;
use input::Input;
use store::{RunnerLock, Settings, Store};

pub fn run(command: BatchCommands) -> Result<()> {
    match command {
        BatchCommands::Status { job_dir, json } => {
            let status = Store::open(&job_dir)?.status(&job_dir)?;
            if json {
                println!("{}", serde_json::to_string(&status)?);
            } else {
                println!("Job: {}\nState: {}\nRunner active: {}\nCaptured: {}\nSucceeded: {}\nFailed: {}\nSkipped: {}\nUnfinished: {}\nLast successful row: {}",
                    terminal_safe(&job_dir.display().to_string()), status.state, status.runner_active,
                    status.captured, status.succeeded, status.failed, status.skipped, status.unfinished,
                    row_label(status.last_successful_row));
                if let Some(error) = status.error {
                    eprintln!("{}", terminal_safe(&error));
                }
            }
            Ok(())
        }
        BatchCommands::Stop { job_dir } => {
            let store = Store::open(&job_dir)?;
            store.request_stop()?;
            eprintln!(
                "Stop requested for {}; last successful row so far: {}",
                terminal_safe(&job_dir.display().to_string()),
                row_label(store.last_successful_row()?)
            );
            Ok(())
        }
        BatchCommands::Export { job_dir } => {
            Store::open(&job_dir)?.export(&mut std::io::stdout().lock())
        }
        command => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(execute(command)),
    }
}

fn client(server: &str, key: Option<String>) -> Result<Client> {
    let mut builder = Client::builder()
        .allow_unauthenticated()
        .base_url(server)
        .timeout(Duration::from_secs(30));
    if let Some(key) = key {
        builder = builder.api_key(key);
    }
    Ok(builder.build()?)
}

struct Redactor(Vec<String>);

impl Redactor {
    fn new(explicit: &Option<String>) -> Self {
        let mut keys = Vec::new();
        if let Some(key) = explicit {
            keys.push(key.trim().to_owned());
        }
        for variable in [
            "OPENKIND_API_KEY",
            "OPENDECISION_API_KEY",
            "TYPESAFE_API_KEY",
        ] {
            if let Ok(key) = std::env::var(variable) {
                keys.push(key.trim().to_owned());
            }
        }
        keys.retain(|key| !key.is_empty());
        Self(keys)
    }

    fn error(&self, error: &anyhow::Error) -> String {
        error
            .chain()
            .map(|cause| {
                let api = match cause.downcast_ref::<openkind_client::Error>() {
                    Some(openkind_client::Error::Api(api)) => Some(api.as_ref()),
                    _ => cause
                        .downcast_ref::<openkind_client::ApiError>()
                        .or_else(|| {
                            cause
                                .downcast_ref::<Box<openkind_client::ApiError>>()
                                .map(Box::as_ref)
                        }),
                };
                if let Some(api) = api {
                    // Redact the original fields before Display can cut a key
                    // in half. Fallback excerpts may already be truncated.
                    let safe = openkind_client::ApiError {
                        status: api.status,
                        code: api.code.as_deref().map(|text| self.text(text)),
                        message: api.message.as_deref().and_then(|text| {
                            if text.ends_with('…') {
                                None
                            } else {
                                Some(self.text(text))
                            }
                        }),
                        body: None,
                        request_id: api.request_id.as_deref().map(|text| self.text(text)),
                        retry_after: api.retry_after,
                        endpoint: api.endpoint.as_deref().map(|text| self.text(text)),
                    };
                    safe.to_string()
                } else {
                    self.text(&cause.to_string())
                }
            })
            .collect::<Vec<_>>()
            .join(": ")
    }

    fn text(&self, mut remaining: &str) -> String {
        let mut result = String::new();
        while let Some((offset, key)) = self
            .0
            .iter()
            .filter_map(|key| remaining.find(key).map(|offset| (offset, key)))
            .min_by_key(|(offset, key)| (*offset, std::cmp::Reverse(key.len())))
        {
            // Match against the original text so overlapping keys neither
            // expose suffixes nor rewrite the replacement marker.
            result.push_str(&remaining[..offset]);
            result.push_str("[redacted]");
            remaining = &remaining[offset + key.len()..];
        }
        result.push_str(remaining);
        result
    }
}

async fn execute(command: BatchCommands) -> Result<()> {
    let mut control = Control::new()?;
    let (mut store, settings, reader, client, _lock, redactor) = match command {
        BatchCommands::Run {
            input,
            job_dir,
            server,
            api_key,
            interval_ms,
            start_row,
            end_row,
        } => {
            anyhow::ensure!(
                start_row > 0 && end_row.is_none_or(|end| end >= start_row),
                "invalid batch row range"
            );
            let redactor = Redactor::new(&api_key);
            let client = client(&server, api_key)?;
            validate_interval(interval_ms)?;
            std::fs::create_dir_all(&job_dir).context("create batch job directory")?;
            let lock = RunnerLock::acquire(&job_dir)?;
            let (file, source) = if input.as_os_str() == "-" {
                (None, None)
            } else {
                let (file, source) = control.prepare(input::open_file(input)).await??;
                (Some(file), Some(source))
            };
            let settings = Settings {
                server: client.base_url().to_owned(),
                interval_ms,
                file: source.clone(),
                start_row,
                end_row,
            };
            let store = Store::create(&job_dir, &settings)?;
            // Initial start preserves stop requests arriving after creation.
            store.set_state("running", None)?;
            let reader = match (file, source) {
                (Some(file), Some(source)) => Input::file(file, source, 0, 1)?,
                _ => Input::stdin(0, 1)?,
            };
            eprintln!(
                "Job saved in {}",
                terminal_safe(&job_dir.display().to_string())
            );
            (store, settings, Some(reader), client, lock, redactor)
        }
        BatchCommands::Resume {
            job_dir,
            server,
            api_key,
            interval_ms,
            skip_failed,
            input,
        } => {
            let lock = RunnerLock::acquire(&job_dir)?;
            let mut store = Store::open(&job_dir)?;
            let mut settings = store.settings()?;
            if let Some(server) = server {
                settings.server = server;
            }
            if let Some(interval_ms) = interval_ms {
                settings.interval_ms = interval_ms;
            }
            let redactor = Redactor::new(&api_key);
            let client = client(&settings.server, api_key)?;
            settings.server = client.base_url().to_owned();
            validate_interval(settings.interval_ms)?;
            let (offset, line, source_done) = store.position()?;
            anyhow::ensure!(
                settings.file.is_none() || input.is_none(),
                "--input - applies only to stdin jobs"
            );
            if skip_failed {
                anyhow::ensure!(
                    store
                        .pending()?
                        .is_some_and(|record| record.status == "failed"),
                    "only a recorded failure can be skipped"
                );
            }
            store.begin(&settings)?;
            let reader = if let Some(source) = &settings.file {
                // Verification precedes replay and skip, even at saved EOF.
                let verification = idle(
                    input::reopen_file(source.clone(), offset),
                    &store,
                    &mut control,
                )
                .await?;
                let file = match verification {
                    Some(Ok(file)) => file,
                    Some(Err(error)) => {
                        store.set_state("paused", Some(&redactor.error(&error)))?;
                        return Err(error);
                    }
                    None => return stopped(&store),
                };
                if source_done {
                    None
                } else {
                    Some(Input::file(file, source.clone(), offset, line)?)
                }
            } else if input.is_some() {
                Some(Input::stdin(offset, line)?)
            } else {
                None
            };
            if skip_failed {
                store.skip_failed()?;
            }
            if let Some(record) = store.pending()? {
                if record.status == "inflight" {
                    eprintln!("Record {} has an uncertain outcome; retrying may repeat server work or billing", record.number);
                }
            }
            if input.is_some() {
                store.append_stdin()?;
            }
            eprintln!("Resuming {}", terminal_safe(&job_dir.display().to_string()));
            (store, settings, reader, client, lock, redactor)
        }
        _ => unreachable!("read commands are dispatched synchronously"),
    };
    let result = drive(
        &mut store,
        &settings,
        reader,
        &client,
        &mut control,
        &redactor,
    )
    .await;
    if let Err(error) = result {
        let diagnostic = redactor.error(&error);
        // A disk failure may prevent this write. The runner lock still
        // releases, allowing status/recovery to detect an interrupted job.
        if matches!(store.saved_state().as_deref(), Ok("running")) {
            let _ = store.set_state("paused", Some(&diagnostic));
        }
        anyhow::bail!("{}", terminal_safe(&diagnostic));
    }
    Ok(())
}

async fn drive(
    store: &mut Store,
    settings: &Settings,
    reader: Option<Input>,
    client: &Client,
    control: &mut Control,
    redactor: &Redactor,
) -> Result<()> {
    let output = Output::new()?;
    let mut next_dispatch = None;
    loop {
        if control.stopping(store)? {
            return stopped(store);
        }
        let record = if let Some(record) = store.pending()? {
            record
        } else {
            let (_, next_line, source_done) = store.position()?;
            if source_done || settings.end_row.is_some_and(|end| next_line > end) {
                store.set_state("completed", None)?;
                anyhow::ensure!(!store.has_skipped()?, "job completed with skipped records");
                eprintln!("Job completed");
                return Ok(());
            }
            let reader = reader.as_ref().context("stdin source is incomplete; resume with --input - and the producer's remaining output")?;
            let next = match idle(reader.next(), store, control).await? {
                Some(result) => result?,
                None => return stopped(store),
            };
            let Some(line) = next else {
                store.source_done()?;
                continue;
            };
            if line.number < settings.start_row
                || (!line.oversized && line.bytes.iter().all(u8::is_ascii_whitespace))
            {
                // Unselected rows advance the source cursor without becoming
                // requests or failure records, including malformed rows.
                store.advance_line(&line)?;
                continue;
            }
            store.capture(line)?
        };
        if control.stopping(store)? {
            return stopped(store);
        }
        let request = (|| -> Result<_> {
            if let Some(error) = &record.input_error {
                anyhow::bail!("{error}");
            }
            let raw = std::str::from_utf8(&record.input).context("record must be UTF-8")?;
            parse_and_validate_request(raw)
        })();
        let request = match request {
            Ok(request) => request,
            Err(error) => {
                let diagnostic = redactor.error(&error);
                store.fail(&record, &diagnostic)?;
                anyhow::bail!(
                    "record {} (line {}): {diagnostic}; last successful row: {}",
                    record.number,
                    record.line,
                    row_label(store.last_successful_row()?)
                );
            }
        };
        if let Some(deadline) = next_dispatch {
            if idle(tokio::time::sleep_until(deadline), store, control)
                .await?
                .is_none()
            {
                return stopped(store);
            }
        }
        if control.stopping(store)? {
            return stopped(store);
        }
        store.start_record(&record)?;
        let evaluation = client.evaluate(request);
        tokio::pin!(evaluation);
        let result = loop {
            tokio::select! {
                result = &mut evaluation => break result,
                event = control.event(store) => { event?; }
            }
        };
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                let diagnostic = redactor.error(&error.into());
                store.fail(&record, &diagnostic)?;
                anyhow::bail!(
                    "record {} (line {}): {diagnostic}; last successful row: {}",
                    record.number,
                    record.line,
                    row_label(store.last_successful_row()?)
                );
            }
        };
        store.succeed(&record, &serde_json::to_string(&response)?)?;
        next_dispatch = Some(
            tokio::time::Instant::now()
                .checked_add(Duration::from_millis(settings.interval_ms))
                .context("interval exceeds the supported clock range")?,
        );
        let mut line = serde_json::to_vec(
            &serde_json::json!({"record":record.number,"line":record.line,"response":response}),
        )?;
        line.push(b'\n');
        // The response is already durable if delivery is interrupted or the
        // downstream consumer closes its pipe. Export recovers all results.
        match deliver(output.write(line), store, control).await? {
            Some(result) => result?,
            None => return stopped(store),
        }
        eprintln!("Saved record {} (line {})", record.number, record.line);
    }
}

fn validate_interval(interval_ms: u64) -> Result<()> {
    anyhow::ensure!(
        tokio::time::Instant::now()
            .checked_add(Duration::from_millis(interval_ms))
            .is_some(),
        "interval exceeds the supported clock range"
    );
    Ok(())
}

async fn deliver<F: Future>(
    future: F,
    store: &Store,
    control: &mut Control,
) -> Result<Option<F::Output>> {
    tokio::pin!(future);
    let mut deadline = None;
    loop {
        if control.stopping(store)? && deadline.is_none() {
            // Give the saved final response time to reach stdout, while a
            // blocked downstream consumer cannot prevent graceful exit.
            deadline = Some(tokio::time::Instant::now() + Duration::from_millis(500));
        }
        tokio::select! {
            result = &mut future => return Ok(Some(result)),
            event = control.event(store) => { event?; }
            _ = async { match deadline { Some(deadline) => tokio::time::sleep_until(deadline).await, None => std::future::pending().await } } => return Ok(None),
        }
    }
}

fn stopped(store: &Store) -> Result<()> {
    store.set_state("stopped", None)?;
    anyhow::bail!(
        "job stopped; last successful row: {}; saved results are available through batch export",
        row_label(store.last_successful_row()?)
    );
}

fn row_label(row: Option<u64>) -> String {
    row.map_or_else(|| "none".to_owned(), |row| row.to_string())
}

async fn idle<F: Future>(
    future: F,
    store: &Store,
    control: &mut Control,
) -> Result<Option<F::Output>> {
    tokio::pin!(future);
    loop {
        if control.stopping(store)? {
            return Ok(None);
        }
        tokio::select! {
            event = control.event(store) => { event?; }
            result = &mut future => return Ok(Some(result)),
        }
    }
}

struct Control {
    stopping: bool,
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
    #[cfg(not(unix))]
    interrupt: std::pin::Pin<Box<dyn Future<Output = std::io::Result<()>> + Send>>,
}

impl Control {
    fn new() -> Result<Self> {
        Ok(Self {
            stopping: false,
            #[cfg(unix)]
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
            #[cfg(unix)]
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
            #[cfg(not(unix))]
            interrupt: Box::pin(tokio::signal::ctrl_c()),
        })
    }

    fn stopping(&mut self, store: &Store) -> Result<bool> {
        if !self.stopping && store.stop_requested()? {
            self.stopping = true;
            eprintln!("Stopping after the active evaluation is saved");
        }
        Ok(self.stopping)
    }

    async fn prepare<F: Future>(&mut self, future: F) -> Result<F::Output> {
        tokio::pin!(future);
        loop {
            tokio::select! {
                result = &mut future => return Ok(result),
                signal = self.signal() => {
                    anyhow::ensure!(signal.is_none(), "job stopped before input fingerprinting completed");
                }
            }
        }
    }

    async fn signal(&mut self) -> Option<bool> {
        #[cfg(unix)]
        let interrupt = tokio::select! {
            _ = self.interrupt.recv() => Some(true),
            _ = self.terminate.recv() => Some(false),
            _ = tokio::time::sleep(Duration::from_millis(100)) => None,
        };
        #[cfg(not(unix))]
        let interrupt = tokio::select! {
            _ = &mut self.interrupt => { self.interrupt = Box::pin(tokio::signal::ctrl_c()); Some(true) },
            _ = tokio::time::sleep(Duration::from_millis(100)) => None,
        };
        interrupt
    }

    async fn event(&mut self, store: &Store) -> Result<()> {
        if let Some(interrupt) = self.signal().await {
            if self.stopping && interrupt {
                store.set_state(
                    "interrupted",
                    Some("immediate interrupt; active request outcome may be uncertain"),
                )?;
                anyhow::bail!("immediate interrupt; last successful row: {}; active request outcome may be uncertain", row_label(store.last_successful_row()?));
            }
            store.request_stop()?;
        }
        self.stopping(store)?;
        Ok(())
    }
}

type Delivery = (Vec<u8>, tokio::sync::oneshot::Sender<Result<()>>);

struct Output {
    sender: std::sync::mpsc::Sender<Delivery>,
}

impl Output {
    fn new() -> Result<Self> {
        let (sender, receiver) = std::sync::mpsc::channel::<Delivery>();
        std::thread::Builder::new()
            .name("batch-output".into())
            .spawn(move || {
                let mut output = std::io::stdout().lock();
                for (line, reply) in receiver {
                    let result = output
                        .write_all(&line)
                        .and_then(|()| output.flush())
                        .map_err(anyhow::Error::from);
                    if reply.send(result).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self { sender })
    }

    async fn write(&self, line: Vec<u8>) -> Result<()> {
        let (reply, done) = tokio::sync::oneshot::channel();
        self.sender
            .send((line, reply))
            .context("batch output worker stopped")?;
        done.await.context("batch output worker stopped")?
    }
}
