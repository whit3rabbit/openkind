//! The local journal owns recovery; stdout is only a live delivery channel.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::input::{FileSource, Line};

const DATABASE: &str = "job.sqlite3";
const LOCK: &str = "runner.lock";

pub struct RunnerLock {
    _file: File,
}

impl RunnerLock {
    pub fn acquire(directory: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.join(LOCK))
            .context("open batch runner lock")?;
        fs2::FileExt::try_lock_exclusive(&file).context("another runner already owns this job")?;
        Ok(Self { _file: file })
    }

    fn active(directory: &Path) -> Result<bool> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory.join(LOCK))?;
        match fs2::FileExt::try_lock_exclusive(&file) {
            Ok(()) => Ok(false),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(true),
            Err(error) => Err(error.into()),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub server: String,
    pub interval_ms: u64,
    pub file: Option<FileSource>,
    #[serde(default = "first_row")]
    pub start_row: u64,
    #[serde(default)]
    pub end_row: Option<u64>,
}

fn first_row() -> u64 {
    1
}

pub struct Record {
    pub number: i64,
    pub line: u64,
    pub input: Vec<u8>,
    pub input_error: Option<String>,
    pub status: String,
}

#[derive(Serialize)]
pub struct Status {
    pub state: String,
    pub runner_active: bool,
    pub captured: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub skipped: u64,
    pub unfinished: u64,
    pub current_record: Option<i64>,
    pub last_successful_row: Option<u64>,
    pub source_complete: bool,
    pub stop_requested: bool,
    pub error: Option<String>,
}

pub struct Store {
    connection: Connection,
}

impl Store {
    pub fn create(directory: &Path, settings: &Settings) -> Result<Self> {
        // Reserve the database without ever replacing an existing job.
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options.open(directory.join(DATABASE)).context(
            "job already exists or cannot be created; use batch resume for an existing job",
        )?;
        let connection = Connection::open(directory.join(DATABASE))?;
        Self::configure(&connection)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA user_version=1;
            CREATE TABLE job (
                id INTEGER PRIMARY KEY CHECK(id=1), settings TEXT NOT NULL,
                state TEXT NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
                next_line INTEGER NOT NULL DEFAULT 1, source_done INTEGER NOT NULL DEFAULT 0,
                stop_requested INTEGER NOT NULL DEFAULT 0, error TEXT,
                last_finished_record INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE records (
                number INTEGER PRIMARY KEY, line INTEGER NOT NULL, input BLOB NOT NULL,
                input_error TEXT, status TEXT NOT NULL, response TEXT, error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0
            );",
        )?;
        connection.execute(
            "INSERT INTO job(id,settings,state) VALUES(1,?1,'created')",
            [serde_json::to_string(settings)?],
        )?;
        Ok(Self { connection })
    }

    pub fn open(directory: &Path) -> Result<Self> {
        let connection = Connection::open_with_flags(
            directory.join(DATABASE),
            OpenFlags::SQLITE_OPEN_READ_WRITE,
        )
        .context("open batch job database")?;
        Self::configure(&connection)?;
        let version: u32 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        anyhow::ensure!(version == 1, "unsupported batch database version {version}");
        Ok(Self { connection })
    }

    fn configure(connection: &Connection) -> Result<()> {
        connection.busy_timeout(Duration::from_secs(2))?;
        connection.execute_batch("PRAGMA synchronous=FULL;")?;
        Ok(())
    }

    pub fn settings(&self) -> Result<Settings> {
        let value: String =
            self.connection
                .query_row("SELECT settings FROM job WHERE id=1", [], |row| row.get(0))?;
        Ok(serde_json::from_str(&value)?)
    }

    pub fn saved_state(&self) -> Result<String> {
        Ok(self
            .connection
            .query_row("SELECT state FROM job WHERE id=1", [], |row| row.get(0))?)
    }

    pub fn has_skipped(&self) -> Result<bool> {
        Ok(self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM records WHERE status='skipped')",
            [],
            |row| row.get(0),
        )?)
    }

    pub fn last_successful_row(&self) -> Result<Option<u64>> {
        Ok(self
            .connection
            .query_row(
                "SELECT line FROM records WHERE status='succeeded' ORDER BY number DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?)
    }

    pub fn begin(&self, settings: &Settings) -> Result<()> {
        self.connection.execute(
            "UPDATE job SET settings=?1,state='running',stop_requested=0,error=NULL WHERE id=1",
            [serde_json::to_string(settings)?],
        )?;
        Ok(())
    }

    pub fn position(&self) -> Result<(u64, u64, bool)> {
        Ok(self.connection.query_row(
            "SELECT offset,next_line,source_done FROM job WHERE id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?)
    }

    pub fn pending(&self) -> Result<Option<Record>> {
        Ok(self
            .connection
            .query_row(
                "SELECT number,line,input,input_error,status FROM records
            WHERE status NOT IN ('succeeded','skipped') ORDER BY number LIMIT 1",
                [],
                |row| {
                    Ok(Record {
                        number: row.get(0)?,
                        line: row.get(1)?,
                        input: row.get(2)?,
                        input_error: row.get(3)?,
                        status: row.get(4)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn capture(&mut self, line: Line) -> Result<Record> {
        let input_error = line
            .oversized
            .then(|| "record exceeds the 32 MiB input limit".to_owned());
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO records(line,input,input_error,status) VALUES(?1,?2,?3,'pending')",
            params![line.number, line.bytes, input_error],
        )?;
        let number = transaction.last_insert_rowid();
        transaction.execute(
            "UPDATE job SET offset=?1,next_line=?2 WHERE id=1",
            params![line.offset, line.number + 1],
        )?;
        transaction.commit()?;
        Ok(Record {
            number,
            line: line.number,
            input: line.bytes,
            input_error,
            status: "pending".into(),
        })
    }

    pub fn advance_line(&self, line: &Line) -> Result<()> {
        self.connection.execute(
            "UPDATE job SET offset=?1,next_line=?2 WHERE id=1",
            params![line.offset, line.number + 1],
        )?;
        Ok(())
    }

    pub fn source_done(&self) -> Result<()> {
        self.connection
            .execute("UPDATE job SET source_done=1 WHERE id=1", [])?;
        Ok(())
    }

    pub fn append_stdin(&self) -> Result<()> {
        self.connection
            .execute("UPDATE job SET source_done=0 WHERE id=1", [])?;
        Ok(())
    }

    pub fn start_record(&self, record: &Record) -> Result<()> {
        self.connection.execute(
            "UPDATE records SET status='inflight',attempts=attempts+1 WHERE number=?1",
            [record.number],
        )?;
        Ok(())
    }

    pub fn succeed(&mut self, record: &Record, response: &str) -> Result<()> {
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "UPDATE records SET status='succeeded',response=?1,error=NULL WHERE number=?2",
            params![response, record.number],
        )?;
        transaction.execute(
            "UPDATE job SET last_finished_record=?1 WHERE id=1",
            [record.number],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn fail(&mut self, record: &Record, error: &str) -> Result<()> {
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "UPDATE records SET status='failed',error=?1 WHERE number=?2",
            params![error, record.number],
        )?;
        transaction.execute("UPDATE job SET state='paused',error=?1 WHERE id=1", [error])?;
        transaction.commit()?;
        Ok(())
    }

    pub fn skip_failed(&mut self) -> Result<()> {
        let record = self.pending()?.context("no failed record to skip")?;
        anyhow::ensure!(
            record.status == "failed",
            "only a recorded failure can be skipped"
        );
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "UPDATE records SET status='skipped' WHERE number=?1",
            [record.number],
        )?;
        transaction.execute(
            "UPDATE job SET last_finished_record=?1 WHERE id=1",
            [record.number],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn set_state(&self, state: &str, error: Option<&str>) -> Result<()> {
        self.connection.execute(
            "UPDATE job SET state=?1,error=?2 WHERE id=1",
            params![state, error],
        )?;
        Ok(())
    }

    pub fn request_stop(&self) -> Result<()> {
        self.connection
            .execute("UPDATE job SET stop_requested=1 WHERE id=1", [])?;
        Ok(())
    }

    pub fn stop_requested(&self) -> Result<bool> {
        Ok(self
            .connection
            .query_row("SELECT stop_requested FROM job WHERE id=1", [], |row| {
                row.get(0)
            })?)
    }

    pub fn status(&self, directory: &Path) -> Result<Status> {
        let runner_active = RunnerLock::active(directory)?;
        let (mut state, source_complete, stop_requested, error): (
            String,
            bool,
            bool,
            Option<String>,
        ) = self.connection.query_row(
            "SELECT state,source_done,stop_requested,error FROM job WHERE id=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        if state == "running" && !runner_active {
            state = "interrupted".into();
        }
        let mut status = Status {
            state,
            runner_active,
            captured: 0,
            succeeded: 0,
            failed: 0,
            skipped: 0,
            unfinished: 0,
            current_record: self.connection.query_row("SELECT number FROM records WHERE status NOT IN ('succeeded','skipped') ORDER BY number LIMIT 1", [], |row| row.get(0)).optional()?,
            last_successful_row: self.last_successful_row()?,
            source_complete,
            stop_requested,
            error,
        };
        let mut statement = self
            .connection
            .prepare("SELECT status,COUNT(*) FROM records GROUP BY status")?;
        for row in statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
        })? {
            let (state, count) = row?;
            status.captured += count;
            match state.as_str() {
                "succeeded" => status.succeeded = count,
                "failed" => status.failed = count,
                "skipped" => status.skipped = count,
                _ => status.unfinished += count,
            }
        }
        Ok(status)
    }

    pub fn export(&self, output: &mut impl std::io::Write) -> Result<()> {
        let mut statement = self.connection.prepare(
            "SELECT number,line,response FROM records WHERE status='succeeded' ORDER BY number",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let response: String = row.get(2)?;
            let envelope = serde_json::json!({"record":row.get::<_, i64>(0)?,"line":row.get::<_, u64>(1)?,"response":serde_json::from_str::<serde_json::Value>(&response)?});
            serde_json::to_writer(&mut *output, &envelope)?;
            writeln!(output)?;
        }
        output.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new() -> Self {
            static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "openkind-batch-store-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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

    fn settings() -> Settings {
        Settings {
            server: "http://127.0.0.1:8080".into(),
            interval_ms: 0,
            file: None,
            start_row: 1,
            end_row: None,
        }
    }
    fn line() -> Line {
        Line {
            bytes: b"fixture".to_vec(),
            number: 7,
            offset: 100,
            oversized: false,
        }
    }

    #[test]
    fn legacy_settings_default_to_the_whole_source() {
        let directory = TempDir::new();
        let store = Store::create(&directory.0, &settings()).unwrap();
        store
            .connection
            .execute(
                "UPDATE job SET settings=?1",
                [r#"{"server":"http://127.0.0.1:8080","interval_ms":0,"file":null}"#],
            )
            .unwrap();
        let saved = store.settings().unwrap();
        assert_eq!(saved.start_row, 1);
        assert_eq!(saved.end_row, None);
    }

    #[test]
    fn recovery_across_capture_dispatch_and_result_commit_boundaries() {
        let directory = TempDir::new();
        let mut store = Store::create(&directory.0, &settings()).unwrap();
        let record = store.capture(line()).unwrap();
        assert_eq!(store.last_successful_row().unwrap(), None);
        drop(store);
        let store = Store::open(&directory.0).unwrap();
        assert_eq!(store.position().unwrap(), (100, 8, false));
        assert_eq!(store.pending().unwrap().unwrap().status, "pending");
        assert_eq!(store.last_successful_row().unwrap(), None);
        store.start_record(&record).unwrap();
        drop(store);
        let mut store = Store::open(&directory.0).unwrap();
        let record = store.pending().unwrap().unwrap();
        assert_eq!(record.status, "inflight");
        assert_eq!(store.last_successful_row().unwrap(), None);
        store.succeed(&record, r#"{"answers":{}}"#).unwrap();
        drop(store);
        let store = Store::open(&directory.0).unwrap();
        assert!(store.pending().unwrap().is_none());
        assert_eq!(store.last_successful_row().unwrap(), Some(7));
        let finished: i64 = store
            .connection
            .query_row("SELECT last_finished_record FROM job", [], |row| row.get(0))
            .unwrap();
        assert_eq!(finished, record.number);
        let mut results = Vec::new();
        store.export(&mut results).unwrap();
        let result: serde_json::Value = serde_json::from_slice(&results).unwrap();
        assert_eq!(result["record"], record.number);
        assert_eq!(result["line"], 7);
    }

    #[test]
    fn disk_full_rolls_back_capture_and_result_progress_together() {
        let directory = TempDir::new();
        let mut store = Store::create(&directory.0, &settings()).unwrap();
        let pages: u64 = store
            .connection
            .query_row("PRAGMA page_count", [], |row| row.get(0))
            .unwrap();
        store
            .connection
            .execute_batch(&format!("PRAGMA max_page_count={pages};"))
            .unwrap();
        let oversized = Line {
            bytes: vec![b'x'; 1024 * 1024],
            ..line()
        };
        assert!(store.capture(oversized).is_err());
        assert_eq!(store.position().unwrap(), (0, 1, false));
        assert!(store.pending().unwrap().is_none());
        let record = store.capture(line()).unwrap();
        store.start_record(&record).unwrap();
        assert!(store.succeed(&record, &"x".repeat(1024 * 1024)).is_err());
        assert_eq!(store.pending().unwrap().unwrap().status, "inflight");
        let finished: i64 = store
            .connection
            .query_row("SELECT last_finished_record FROM job", [], |row| row.get(0))
            .unwrap();
        assert_eq!(finished, 0);
        assert!(store.export(&mut Vec::new()).is_ok());
    }

    #[test]
    fn future_database_versions_are_rejected_without_modification() {
        let directory = TempDir::new();
        let store = Store::create(&directory.0, &settings()).unwrap();
        store
            .connection
            .execute_batch("PRAGMA user_version=2")
            .unwrap();
        assert!(Store::open(&directory.0).is_err());
    }
}
