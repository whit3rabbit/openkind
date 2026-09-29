//! Per-task sample store: SQLite, WAL mode, one file per task.
//!
//! Rows record what was served and what the teacher said, so training,
//! calibration, shadow judgement, and drift monitoring all read the same
//! history. Splits are assigned per request (a seeded draw), and only
//! IID channels (`bootstrap`, `audit`, `fallback`) participate in the
//! calibration split.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};

use super::task::Channel;
use super::ProxyCacheError;

/// One collected sample row.
#[derive(Debug, Clone, Default)]
pub struct SampleRow {
    /// Store row id (assigned on insert; zero on unsaved rows).
    pub id: i64,
    /// Task version (short fingerprint) this row belongs to.
    pub task_version: String,
    /// Train/calib split assigned per request.
    pub split: String,
    /// Serving channel (`bootstrap`, `audit`, `fallback`, `co_deferred`, `student`).
    pub channel: String,
    /// Why the item was routed.
    pub routing_reason: String,
    /// `teacher` or `student`.
    pub served_by: String,
    /// Request text (empty when text storage is disabled).
    pub text: String,
    /// Salted text hash (always stored; identifies text without storing it).
    pub text_hash: String,
    /// Encoder lineage identity.
    pub encoder_id: String,
    /// L2-normalized embedding (f32 little-endian blob).
    pub embedding: Vec<f32>,
    /// `"text"` or `"json"` state.
    pub state_type: String,
    /// Importance weight for training.
    pub weight: f64,
    /// Teacher's argmax label.
    pub teacher_label: Option<String>,
    /// Teacher's full distribution (JSON object).
    pub teacher_probs: Option<String>,
    /// Teacher's reported confidence.
    pub teacher_confidence: Option<f64>,
    /// Resolved teacher model (`jev-1.13.0`).
    pub teacher_model: Option<String>,
    /// Teacher round-trip latency in milliseconds.
    pub teacher_latency_ms: Option<f64>,
    /// Student version that would have served (`student-v3`).
    pub student_version: Option<String>,
    /// Student's argmax label at routing time.
    pub student_label: Option<String>,
    /// Student's distribution at routing time (JSON object).
    pub student_probs: Option<String>,
    /// Student's routing confidence at routing time.
    pub student_confidence: Option<f64>,
    /// kNN gate score at routing time.
    pub ood_score: Option<f64>,
    /// Shadow candidate version that scored the row.
    pub shadow_version: Option<String>,
    /// Shadow candidate's label.
    pub shadow_label: Option<String>,
    /// Shadow candidate's confidence.
    pub shadow_confidence: Option<f64>,
}

impl SampleRow {
    /// Whether this row carries a usable teacher label.
    pub fn teacher_labelled(&self) -> bool {
        self.teacher_label.is_some()
    }

    /// Parse the channel token.
    pub fn channel_kind(&self) -> Option<Channel> {
        Channel::parse(&self.channel)
    }
}

/// Row counts used for readiness checks and status logging.
#[derive(Debug, Clone, Copy, Default)]
pub struct StoreCounts {
    /// Teacher-labelled rows in the train split.
    pub train_labelled: i64,
    /// Teacher-labelled rows in the calibration split (IID channels only).
    pub calib_labelled: i64,
    /// All teacher-labelled rows.
    pub teacher_labelled: i64,
    /// Rows answered locally by the student.
    pub student_served: i64,
    /// Newest row id.
    pub max_id: i64,
}

/// Per-class teacher-labelled train counts: `label -> n`.
pub type LabelCounts = Vec<(String, i64)>;

/// A SQLite-backed sample store for one task.
pub struct SampleStore {
    path: PathBuf,
    connection: Mutex<Connection>,
}

const SCHEMA_VERSION: i64 = 1;

impl SampleStore {
    /// Open (or create) the store at `path`, creating parent directories.
    pub fn open(path: &Path) -> Result<Self, ProxyCacheError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                ProxyCacheError::Store(format!("create {}: {error}", path.display()))
            })?;
        }
        let connection = Connection::open(path)
            .map_err(|error| ProxyCacheError::Store(format!("open {}: {error}", path.display())))?;
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|error| ProxyCacheError::Store(format!("enable WAL: {error}")))?;
        connection
            .pragma_update(None, "synchronous", "NORMAL")
            .map_err(|error| ProxyCacheError::Store(format!("set synchronous: {error}")))?;
        connection
            .execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY,
                    ts REAL NOT NULL,
                    task_version TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    text_hash TEXT NOT NULL DEFAULT '',
                    encoder_id TEXT NOT NULL DEFAULT '',
                    embedding BLOB NOT NULL,
                    channel TEXT NOT NULL,
                    split TEXT NOT NULL,
                    served_by TEXT NOT NULL,
                    routing_reason TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    state_type TEXT NOT NULL DEFAULT 'text',
                    teacher_label TEXT,
                    teacher_probs TEXT,
                    teacher_confidence REAL,
                    teacher_model TEXT,
                    teacher_latency_ms REAL,
                    student_version TEXT,
                    student_label TEXT,
                    student_probs TEXT,
                    student_confidence REAL,
                    ood_score REAL,
                    shadow_version TEXT,
                    shadow_label TEXT,
                    shadow_confidence REAL
                );
                CREATE INDEX IF NOT EXISTS idx_samples_split
                    ON samples (task_version, split, channel);
                CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples (ts);
                CREATE INDEX IF NOT EXISTS idx_samples_lineage
                    ON samples (task_version, teacher_model, split);
                CREATE INDEX IF NOT EXISTS idx_samples_shadow
                    ON samples (shadow_version);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    detail TEXT
                );
                "#,
            )
            .map_err(|error| ProxyCacheError::Store(format!("create schema: {error}")))?;
        connection
            .pragma_update(None, "user_version", SCHEMA_VERSION)
            .map_err(|error| ProxyCacheError::Store(format!("set user_version: {error}")))?;
        Ok(Self {
            path: path.to_path_buf(),
            connection: Mutex::new(connection),
        })
    }

    /// Store file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Insert one row, returning its id.
    pub fn insert(&self, row: &SampleRow) -> Result<i64, ProxyCacheError> {
        let embedding_bytes = embedding_to_bytes(&row.embedding);
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        connection
            .execute(
                r#"
                INSERT INTO samples (
                    ts, task_version, text, text_hash, encoder_id, embedding,
                    channel, split, served_by, routing_reason, weight, state_type,
                    teacher_label, teacher_probs, teacher_confidence, teacher_model,
                    teacher_latency_ms, student_version, student_label, student_probs,
                    student_confidence, ood_score, shadow_version, shadow_label,
                    shadow_confidence
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                    ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25
                )
                "#,
                params![
                    now_unix(),
                    row.task_version,
                    row.text,
                    row.text_hash,
                    row.encoder_id,
                    embedding_bytes,
                    row.channel,
                    row.split,
                    row.served_by,
                    row.routing_reason,
                    row.weight,
                    row.state_type,
                    row.teacher_label,
                    row.teacher_probs,
                    row.teacher_confidence,
                    row.teacher_model,
                    row.teacher_latency_ms,
                    row.student_version,
                    row.student_label,
                    row.student_probs,
                    row.student_confidence,
                    row.ood_score,
                    row.shadow_version,
                    row.shadow_label,
                    row.shadow_confidence,
                ],
            )
            .map_err(|error| ProxyCacheError::Store(format!("insert sample: {error}")))?;
        Ok(connection.last_insert_rowid())
    }

    /// Insert a lifecycle event.
    pub fn insert_event(
        &self,
        kind: &str,
        detail: &serde_json::Value,
    ) -> Result<(), ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        connection
            .execute(
                "INSERT INTO events (ts, kind, detail) VALUES (?1, ?2, ?3)",
                params![now_unix(), kind, detail.to_string()],
            )
            .map_err(|error| ProxyCacheError::Store(format!("insert event: {error}")))?;
        Ok(())
    }

    /// Readiness counters for the should-train checks.
    pub fn counts(&self, task_version: &str) -> Result<StoreCounts, ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        let scalar = |sql: &str, bound: &str| -> Result<i64, ProxyCacheError> {
            connection
                .query_row(sql, params![task_version, bound], |row| row.get(0))
                .map_err(|error| ProxyCacheError::Store(format!("count query: {error}")))
        };
        let scalar_unbound = |sql: &str| -> Result<i64, ProxyCacheError> {
            connection
                .query_row(sql, params![task_version], |row| row.get(0))
                .map_err(|error| ProxyCacheError::Store(format!("count query: {error}")))
        };
        Ok(StoreCounts {
            train_labelled: scalar(
                "SELECT COUNT(*) FROM samples WHERE task_version = ?1 AND split = ?2 \
                 AND teacher_label IS NOT NULL",
                "train",
            )?,
            calib_labelled: scalar(
                "SELECT COUNT(*) FROM samples WHERE task_version = ?1 AND split = ?2 \
                 AND teacher_label IS NOT NULL AND channel IN ('bootstrap','audit','fallback')",
                "calib",
            )?,
            teacher_labelled: scalar_unbound(
                "SELECT COUNT(*) FROM samples WHERE task_version = ?1 AND teacher_label IS NOT NULL",
            )?,
            student_served: scalar_unbound(
                "SELECT COUNT(*) FROM samples WHERE task_version = ?1 AND served_by = 'student'",
            )?,
            max_id: scalar_unbound("SELECT COALESCE(MAX(id), 0) FROM samples WHERE task_version = ?1")?,
        })
    }

    /// Newest teacher-labelled rows per split, returning `(train, calib)`.
    pub fn labelled_rows(
        &self,
        task_version: &str,
        max_train: usize,
        max_calib: usize,
    ) -> Result<(Vec<SampleRow>, Vec<SampleRow>), ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        let query_train = "
            SELECT * FROM samples
            WHERE task_version = ?1 AND split = 'train' AND teacher_label IS NOT NULL
            ORDER BY id DESC LIMIT ?2";
        let query_calib = "
            SELECT * FROM samples
            WHERE task_version = ?1 AND split = 'calib' AND teacher_label IS NOT NULL
              AND channel IN ('bootstrap','audit','fallback')
            ORDER BY id DESC LIMIT ?2";
        let train = read_rows(
            &connection,
            query_train,
            params![task_version, max_train as i64],
        )?;
        let calib = read_rows(
            &connection,
            query_calib,
            params![task_version, max_calib as i64],
        )?;
        Ok((train, calib))
    }

    /// Per-class teacher-labelled train counts (for rare-class deferral).
    pub fn train_label_counts(&self, task_version: &str) -> Result<LabelCounts, ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        let mut statement = connection
            .prepare(
                "SELECT teacher_label, COUNT(*) FROM samples \
                 WHERE task_version = ?1 AND split = 'train' AND teacher_label IS NOT NULL \
                 GROUP BY teacher_label",
            )
            .map_err(|error| ProxyCacheError::Store(format!("label counts: {error}")))?;
        let rows = statement
            .query_map(params![task_version], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|error| ProxyCacheError::Store(format!("label counts: {error}")))?;
        let mut counts: Vec<(String, i64)> = rows.filter_map(|row| row.ok()).collect();
        counts.sort();
        Ok(counts)
    }

    /// Newest rows carrying a shadow score for the given candidate version.
    pub fn shadow_rows(
        &self,
        task_version: &str,
        shadow_version: &str,
        limit: usize,
    ) -> Result<Vec<SampleRow>, ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        read_rows(
            &connection,
            "SELECT * FROM samples \
             WHERE task_version = ?1 AND shadow_version = ?2 AND shadow_label IS NOT NULL \
               AND teacher_label IS NOT NULL
             ORDER BY id DESC LIMIT ?3",
            params![task_version, shadow_version, limit as i64],
        )
    }

    /// Teacher-labelled rows with id ≥ `since_id`, oldest first (recovery
    /// training after drift fallback restarts at `since_id`).
    pub fn rows_since(
        &self,
        task_version: &str,
        since_id: i64,
        limit: usize,
    ) -> Result<Vec<SampleRow>, ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        read_rows(
            &connection,
            "SELECT * FROM samples \
             WHERE task_version = ?1 AND id >= ?2 AND teacher_label IS NOT NULL \
             ORDER BY id ASC LIMIT ?3",
            params![task_version, since_id, limit as i64],
        )
    }

    /// `teacher_model` of the newest teacher-labelled row (lineage across
    /// restarts).
    pub fn latest_teacher_model(
        &self,
        task_version: &str,
    ) -> Result<Option<String>, ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        connection
            .query_row(
                "SELECT teacher_model FROM samples \
                 WHERE task_version = ?1 AND teacher_model IS NOT NULL \
                 ORDER BY id DESC LIMIT 1",
                params![task_version],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| ProxyCacheError::Store(format!("latest teacher model: {error}")))
    }

    /// Newest audit rows (teacher-labelled, channel `audit`) for drift
    /// monitoring, newest first.
    pub fn recent_audit_rows(
        &self,
        task_version: &str,
        limit: usize,
    ) -> Result<Vec<SampleRow>, ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        read_rows(
            &connection,
            "SELECT * FROM samples \
             WHERE task_version = ?1 AND channel = 'audit' AND teacher_label IS NOT NULL \
             ORDER BY id DESC LIMIT ?2",
            params![task_version, limit as i64],
        )
    }

    /// Prune old rows: keep the newest `keep_train`/`keep_calib`
    /// teacher-labelled rows per split and the newest `keep_local`
    /// student-served rows. Returns deleted row count.
    pub fn prune(
        &self,
        task_version: &str,
        keep_train: usize,
        keep_calib: usize,
        keep_local: usize,
    ) -> Result<usize, ProxyCacheError> {
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        let transaction = connection
            .transaction()
            .map_err(|error| ProxyCacheError::Store(format!("prune begin: {error}")))?;
        let mut deleted = 0;
        // Teacher-labelled history per split (student-served rows carry no
        // teacher label and are pruned separately, regardless of split).
        let buckets = [
            (
                "DELETE FROM samples WHERE id IN (\
                 SELECT id FROM samples WHERE task_version = ?1 AND split = 'train' \
                   AND channel IN ('bootstrap','audit','fallback','co_deferred') \
                 ORDER BY id DESC LIMIT -1 OFFSET ?2)",
                keep_train,
            ),
            (
                "DELETE FROM samples WHERE id IN (\
                 SELECT id FROM samples WHERE task_version = ?1 AND split = 'calib' \
                   AND channel IN ('bootstrap','audit','fallback') \
                 ORDER BY id DESC LIMIT -1 OFFSET ?2)",
                keep_calib,
            ),
            (
                "DELETE FROM samples WHERE id IN (\
                 SELECT id FROM samples WHERE task_version = ?1 AND served_by = 'student' \
                 ORDER BY id DESC LIMIT -1 OFFSET ?2)",
                keep_local,
            ),
        ];
        for (sql, keep) in buckets {
            deleted += transaction
                .execute(sql, params![task_version, keep as i64])
                .map_err(|error| ProxyCacheError::Store(format!("prune: {error}")))?;
        }
        transaction
            .commit()
            .map_err(|error| ProxyCacheError::Store(format!("prune commit: {error}")))?;
        Ok(deleted)
    }

    /// List lifecycle event kinds (newest first) for status recovery.
    pub fn recent_events(&self, limit: usize) -> Result<Vec<(String, String)>, ProxyCacheError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| ProxyCacheError::Store("store mutex poisoned".into()))?;
        let mut statement = connection
            .prepare("SELECT kind, detail FROM events ORDER BY id DESC LIMIT ?1")
            .map_err(|error| ProxyCacheError::Store(format!("events: {error}")))?;
        let rows = statement
            .query_map(params![limit as i64], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(|error| ProxyCacheError::Store(format!("events: {error}")))?;
        Ok(rows.filter_map(|row| row.ok()).collect())
    }
}

fn read_rows<P: rusqlite::Params>(
    connection: &Connection,
    sql: &str,
    parameters: P,
) -> Result<Vec<SampleRow>, ProxyCacheError> {
    let mut statement = connection
        .prepare(sql)
        .map_err(|error| ProxyCacheError::Store(format!("prepare row query: {error}")))?;
    let column_names: Vec<String> = statement
        .column_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let rows = statement
        .query_map(parameters, |row| {
            let get_string = |name: &str| -> rusqlite::Result<Option<String>> {
                if column_names.iter().any(|column| column == name) {
                    row.get(name)
                } else {
                    Ok(None)
                }
            };
            let get_f64 = |name: &str| -> rusqlite::Result<Option<f64>> {
                if column_names.iter().any(|column| column == name) {
                    row.get(name)
                } else {
                    Ok(None)
                }
            };
            let embedding_blob: Vec<u8> = row.get("embedding")?;
            Ok(SampleRow {
                id: row.get("id")?,
                task_version: row.get("task_version")?,
                split: row.get("split")?,
                channel: row.get("channel")?,
                routing_reason: get_string("routing_reason")?.unwrap_or_default(),
                served_by: row.get("served_by")?,
                text: get_string("text")?.unwrap_or_default(),
                text_hash: get_string("text_hash")?.unwrap_or_default(),
                encoder_id: get_string("encoder_id")?.unwrap_or_default(),
                embedding: bytes_to_embedding(&embedding_blob),
                state_type: get_string("state_type")?.unwrap_or_else(|| "text".into()),
                weight: row.get("weight").unwrap_or(1.0),
                teacher_label: get_string("teacher_label")?,
                teacher_probs: get_string("teacher_probs")?,
                teacher_confidence: get_f64("teacher_confidence")?,
                teacher_model: get_string("teacher_model")?,
                teacher_latency_ms: get_f64("teacher_latency_ms")?,
                student_version: get_string("student_version")?,
                student_label: get_string("student_label")?,
                student_probs: get_string("student_probs")?,
                student_confidence: get_f64("student_confidence")?,
                ood_score: get_f64("ood_score")?,
                shadow_version: get_string("shadow_version")?,
                shadow_label: get_string("shadow_label")?,
                shadow_confidence: get_f64("shadow_confidence")?,
            })
        })
        .map_err(|error| ProxyCacheError::Store(format!("row query: {error}")))?;
    Ok(rows.filter_map(|row| row.ok()).collect())
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> (tempfile::TempDir, SampleStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = SampleStore::open(&dir.path().join(name)).unwrap();
        (dir, store)
    }

    fn row(version: &str, channel: &str, split: &str, label: Option<&str>) -> SampleRow {
        SampleRow {
            task_version: version.into(),
            split: split.into(),
            channel: channel.into(),
            routing_reason: "test".into(),
            served_by: if label.is_some() {
                "teacher"
            } else {
                "student"
            }
            .into(),
            text: "hello".into(),
            text_hash: "deadbeef".into(),
            encoder_id: "hash-512-0-bi".into(),
            embedding: vec![0.5, 0.5, 0.25],
            state_type: "text".into(),
            weight: 1.0,
            teacher_label: label.map(str::to_owned),
            teacher_probs: label.map(|_| r#"{"a":0.9,"b":0.1}"#.to_owned()),
            teacher_confidence: label.map(|_| 0.9),
            teacher_model: label.map(|_| "jev-1.0.0".to_owned()),
            ..Default::default()
        }
    }

    #[test]
    fn insert_and_counts_roundtrip() {
        let (_dir, store) = temp_store("samples.sqlite");
        for i in 0..30 {
            let mut sample = row("v1", "bootstrap", "train", Some("a"));
            if i % 5 == 0 {
                sample.split = "calib".into();
            }
            store.insert(&sample).unwrap();
        }
        store.insert(&row("v1", "student", "train", None)).unwrap();
        let counts = store.counts("v1").unwrap();
        // 24 train + 6 calib labelled rows; the student row has no label.
        assert_eq!(counts.train_labelled, 24);
        assert_eq!(counts.calib_labelled, 6);
        assert_eq!(counts.teacher_labelled, 30);
        assert_eq!(counts.student_served, 1);
        assert_eq!(counts.max_id, 31);

        // Another task version does not see them.
        let other = store.counts("v2").unwrap();
        assert_eq!(other.teacher_labelled, 0);
    }

    #[test]
    fn labelled_rows_respect_limit_and_channel_filter() {
        let (_dir, store) = temp_store("samples.sqlite");
        for i in 0..40 {
            let split = if i % 2 == 0 { "train" } else { "calib" };
            let mut sample = row("v1", "bootstrap", split, Some("a"));
            if split == "calib" && i % 4 == 3 {
                // Co-deferred rows must never enter the calibration split.
                sample.channel = "co_deferred".into();
            }
            store.insert(&sample).unwrap();
        }
        let (train, calib) = store.labelled_rows("v1", 50, 50).unwrap();
        assert_eq!(train.len(), 20);
        assert_eq!(calib.len(), 10, "co_deferred calib rows are excluded");
        assert!(train
            .windows(2)
            .all(|w| w[0].task_version == w[1].task_version));
        // Newest first.
        assert!(
            train[0].text_hash >= train[1].text_hash || train[0].embedding == train[1].embedding
        );

        let (train_limited, calib_limited) = store.labelled_rows("v1", 5, 3).unwrap();
        assert_eq!(train_limited.len(), 5);
        assert_eq!(calib_limited.len(), 3);
    }

    #[test]
    fn prune_keeps_the_newest_rows() {
        let (_dir, store) = temp_store("samples.sqlite");
        for i in 0..20 {
            let mut sample = row("v1", "bootstrap", "train", Some("a"));
            sample.text = format!("row {i}");
            store.insert(&sample).unwrap();
        }
        for i in 0..8 {
            let mut sample = row("v1", "student", "train", None);
            sample.text = format!("local {i}");
            store.insert(&sample).unwrap();
        }
        let deleted = store.prune("v1", 5, 5, 3).unwrap();
        // 15 labelled train rows + 5 local rows deleted.
        assert_eq!(deleted, 20);
        let (train, _calib) = store.labelled_rows("v1", 50, 50).unwrap();
        assert_eq!(train.len(), 5);
        assert_eq!(train[0].text, "row 19");
        assert_eq!(train[4].text, "row 15");
        let counts = store.counts("v1").unwrap();
        assert_eq!(counts.student_served, 3);
    }

    #[test]
    fn shadow_rows_filter_by_version() {
        let (_dir, store) = temp_store("samples.sqlite");
        let mut sample = row("v1", "audit", "calib", Some("a"));
        sample.shadow_version = Some("student-v2".into());
        sample.shadow_label = Some("a".into());
        sample.shadow_confidence = Some(0.8);
        store.insert(&sample).unwrap();
        let mut other = row("v1", "audit", "calib", Some("a"));
        other.shadow_version = Some("student-v3".into());
        other.shadow_label = Some("b".into());
        store.insert(&other).unwrap();
        let rows = store.shadow_rows("v1", "student-v2", 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].shadow_label.as_deref(), Some("a"));
    }

    #[test]
    fn lineage_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lineage.sqlite");
        {
            let store = SampleStore::open(&path).unwrap();
            let mut sample = row("v1", "bootstrap", "train", Some("a"));
            sample.teacher_model = Some("jev-1.0.0".into());
            store.insert(&sample).unwrap();
            let mut newer = row("v1", "bootstrap", "train", Some("a"));
            newer.teacher_model = Some("jev-1.1.0".into());
            store.insert(&newer).unwrap();
        }
        let reopened = SampleStore::open(&path).unwrap();
        assert_eq!(
            reopened.latest_teacher_model("v1").unwrap().as_deref(),
            Some("jev-1.1.0")
        );
    }

    #[test]
    fn events_roundtrip() {
        let (_dir, store) = temp_store("events.sqlite");
        store
            .insert_event("promoted", &serde_json::json!({"version": "student-v1"}))
            .unwrap();
        store
            .insert_event("rejected", &serde_json::json!({"reason": "budget"}))
            .unwrap();
        let events = store.recent_events(10).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "rejected");
        assert_eq!(events[1].0, "promoted");
    }

    #[test]
    fn embedding_blob_roundtrip() {
        let bytes = embedding_to_bytes(&[0.25, -0.5, 3.75]);
        assert_eq!(bytes_to_embedding(&bytes), vec![0.25, -0.5, 3.75]);
    }

    #[test]
    fn recent_audit_rows_returns_newest_first() {
        let (_dir, store) = temp_store("audit.sqlite");
        for i in 0..10 {
            let mut sample = row("v1", "audit", "train", Some("a"));
            sample.text = format!("audit {i}");
            store.insert(&sample).unwrap();
        }
        store
            .insert(&row("v1", "bootstrap", "train", Some("a")))
            .unwrap();
        let rows = store.recent_audit_rows("v1", 3).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].text, "audit 9");
    }
}
