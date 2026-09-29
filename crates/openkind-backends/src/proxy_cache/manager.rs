//! Task registry: admission, LRU load/unload, and the background training
//! worker.
//!
//! Tasks are keyed by `sha256(canonical(tenant, type, fingerprint, model))`;
//! identical questions under different names share one task. A task is only
//! created after the same key was seen `admission_min_requests` times within
//! the admission window, so one-off questions never allocate engines or
//! stores. Restored tasks (present on disk) bypass admission.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::encoder::TextEmbedder;
use super::engine::{FitInput, TaskEngine};
use super::task::{TaskConfig, TaskSpec};
use super::ProxyCacheError;

/// Manager configuration.
#[derive(Debug, Clone)]
pub struct ProxyCacheManagerConfig {
    /// Root data directory (`tasks/` lives below it).
    pub data_dir: PathBuf,
    /// Task configuration defaults (operator overrides already applied).
    pub task_config: TaskConfig,
    /// Default disagreement budget input: target agreement (0.98).
    pub target_agreement: f64,
    /// Optional reported-confidence floor for every task.
    pub confidence_floor: Option<f64>,
    /// Requests a new task key must see within the window before a task is
    /// created.
    pub admission_min_requests: usize,
    /// Admission window.
    pub admission_window: Duration,
    /// Loaded engines kept in memory (LRU beyond this).
    pub max_loaded: usize,
    /// Total tasks on disk.
    pub max_tasks: usize,
    /// Tenant label for shared tenancy.
    pub default_tenant: String,
    /// New tasks each key may create (abuse guard).
    pub max_new_tasks_per_key: usize,
}

impl Default for ProxyCacheManagerConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("proxy-cache-data"),
            task_config: TaskConfig::default(),
            target_agreement: 0.98,
            confidence_floor: None,
            admission_min_requests: 50,
            admission_window: Duration::from_secs(86_400),
            max_loaded: 64,
            max_tasks: 10_000,
            default_tenant: "default".into(),
            max_new_tasks_per_key: 100,
        }
    }
}

struct AdmissionCounter {
    window_start: Instant,
    count: usize,
}

struct ManagerState {
    engines: HashMap<String, Arc<Mutex<TaskEngine>>>,
    lru: VecDeque<String>,
    admission: HashMap<String, AdmissionCounter>,
    tasks_on_disk: usize,
}

/// The proxy-cache manager: owns all task engines and the training worker.
pub struct ProxyCacheManager {
    config: ProxyCacheManagerConfig,
    embedder: Arc<dyn TextEmbedder>,
    state: Mutex<ManagerState>,
    salt: Vec<u8>,
    train_tx: std::sync::mpsc::Sender<String>,
}

impl ProxyCacheManager {
    /// Build a manager, spawning the training worker thread. Existing task
    /// directories are indexed (their engines load lazily on first use).
    pub fn new(
        config: ProxyCacheManagerConfig,
        embedder: Arc<dyn TextEmbedder>,
    ) -> Result<Arc<Self>, ProxyCacheError> {
        let tasks_dir = config.data_dir.join("tasks");
        std::fs::create_dir_all(&tasks_dir).map_err(|error| {
            ProxyCacheError::Store(format!("create {}: {error}", tasks_dir.display()))
        })?;
        let salt = load_or_create_salt(&config.data_dir)?;
        let (train_tx, train_rx) = std::sync::mpsc::channel::<String>();

        let tasks_on_disk = std::fs::read_dir(&tasks_dir)
            .map(|entries| entries.filter_map(|entry| entry.ok()).count())
            .unwrap_or(0);

        let manager = Arc::new(Self {
            config,
            embedder,
            state: Mutex::new(ManagerState {
                engines: HashMap::new(),
                lru: VecDeque::new(),
                admission: HashMap::new(),
                tasks_on_disk,
            }),
            salt,
            train_tx,
        });

        // Background training worker: one fit at a time, FIFO. The worker
        // owns the receiving end and exits when the manager is dropped.
        let weak = Arc::downgrade(&manager);
        std::thread::Builder::new()
            .name("proxy-cache-trainer".into())
            .spawn(move || {
                for key in train_rx {
                    let Some(manager) = weak.upgrade() else {
                        return;
                    };
                    manager.run_training(key);
                }
            })
            .map_err(|error| ProxyCacheError::Store(format!("spawn trainer: {error}")))?;

        Ok(manager)
    }

    /// The shared embedder.
    pub fn embedder(&self) -> &Arc<dyn TextEmbedder> {
        &self.embedder
    }

    /// Manager configuration.
    pub fn config(&self) -> &ProxyCacheManagerConfig {
        &self.config
    }

    /// The manager salt (key hashing, text hashing when text is not stored).
    pub fn salt(&self) -> &[u8] {
        &self.salt
    }

    /// Resolve (or admit) the task for one request item and return its engine.
    ///
    /// `admit` drives admission: pass `true` when the caller counted this
    /// request toward the key already (the proxy counts once per request).
    pub fn route_context(
        &self,
        spec: &TaskSpec,
        model: &str,
    ) -> Result<Option<Arc<Mutex<TaskEngine>>>, ProxyCacheError> {
        let key = spec.task_key(&self.config.default_tenant, model);
        let needs_creation = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| ProxyCacheError::Store("manager mutex poisoned".into()))?;
            if state.engines.contains_key(&key) {
                false
            } else {
                // On disk?
                let dir = self.task_dir(&key);
                if dir.join("task.json").exists() {
                    false
                } else {
                    // Admission gate.
                    let counter = state
                        .admission
                        .entry(key.clone())
                        .or_insert(AdmissionCounter {
                            window_start: Instant::now(),
                            count: 0,
                        });
                    if counter.window_start.elapsed() > self.config.admission_window {
                        counter.window_start = Instant::now();
                        counter.count = 0;
                    }
                    counter.count += 1;
                    counter.count < self.config.admission_min_requests.max(1)
                }
            }
        };
        if needs_creation {
            return Ok(None);
        }
        let engine = self.load_engine(&key, spec, model)?;
        Ok(Some(engine))
    }

    /// Load or insert the engine for `key` (LRU-capped).
    fn load_engine(
        &self,
        key: &str,
        spec: &TaskSpec,
        model: &str,
    ) -> Result<Arc<Mutex<TaskEngine>>, ProxyCacheError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ProxyCacheError::Store("manager mutex poisoned".into()))?;
        if let Some(engine) = state.engines.get(key).cloned() {
            touch_lru(&mut state.lru, key);
            return Ok(engine);
        }
        let dir = self.task_dir(key);
        let engine = if dir.join("task.json").exists() {
            TaskEngine::load(
                &dir,
                key.to_owned(),
                self.config.task_config.clone(),
                self.config.target_agreement,
                self.config.confidence_floor,
            )?
        } else {
            if state.tasks_on_disk >= self.config.max_tasks {
                return Err(ProxyCacheError::Contract(format!(
                    "proxy-cache task limit reached ({})",
                    self.config.max_tasks
                )));
            }
            let created = TaskEngine::create(
                &dir,
                key.to_owned(),
                self.config.default_tenant.clone(),
                model.to_owned(),
                spec.clone(),
                self.config.task_config.clone(),
                self.config.target_agreement,
                self.config.confidence_floor,
            )?;
            state.tasks_on_disk += 1;
            created
        };
        let engine = Arc::new(Mutex::new(engine));
        state.engines.insert(key.to_owned(), engine.clone());
        touch_lru(&mut state.lru, key);
        // Evict beyond the LRU cap (state lives in the store; engines reload
        // on demand).
        while state.lru.len() > self.config.max_loaded.max(1) {
            let Some(evict_key) = state.lru.pop_front() else {
                break;
            };
            state.engines.remove(&evict_key);
        }
        Ok(engine)
    }

    /// Count one request toward admission for every group key it touches.
    /// The proxy calls this when the request is forwarded (unverified key or
    /// not-yet-created tasks).
    pub fn count_toward_admission(&self, spec: &TaskSpec, model: &str) {
        let key = spec.task_key(&self.config.default_tenant, model);
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
        if state.engines.contains_key(&key) || self.task_dir(&key).join("task.json").exists() {
            return;
        }
        let counter = state.admission.entry(key).or_insert(AdmissionCounter {
            window_start: Instant::now(),
            count: 0,
        });
        if counter.window_start.elapsed() > self.config.admission_window {
            counter.window_start = Instant::now();
            counter.count = 0;
        }
        counter.count += 1;
    }

    /// Queue a fit for one task key on the training worker.
    pub fn request_training(&self, key: &str) -> bool {
        self.train_tx.send(key.to_owned()).is_ok()
    }

    /// Run the fit for one task key (worker thread).
    fn run_training(&self, key: String) {
        let engine = {
            let state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            state.engines.get(&key).cloned()
        };
        let Some(engine) = engine else {
            return;
        };
        let input = {
            let mut guard = match engine.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            match guard.prepare_fit() {
                Ok(Some(input)) => Some(input),
                Ok(None) => None,
                Err(error) => {
                    tracing::warn!(task = %key, "proxy-cache fit preparation failed: {error}");
                    None
                }
            }
        };
        let Some(input) = input else {
            // prepare_fit backed off (admission windows, recovery horizons):
            // the want remains, so keep the flag and retry on a later tick.
            if let Ok(mut guard) = engine.lock() {
                guard.clear_training_in_flight(true);
            }
            return;
        };
        let output = {
            let guard = match engine.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            guard.run_fit(&input)
        };
        // Drop the extracted rows before touching the engine again.
        drop_input(input);
        match output {
            Ok(output) => {
                if let Ok(mut guard) = engine.lock() {
                    match guard.apply_fit(output) {
                        Ok(events) => {
                            for (kind, detail) in events {
                                tracing::info!(task = %key, event = %kind, ?detail, "proxy-cache");
                            }
                        }
                        Err(error) => {
                            tracing::warn!(task = %key, "proxy-cache fit apply failed: {error}");
                        }
                    }
                }
            }
            Err(error) => {
                if let Ok(mut guard) = engine.lock() {
                    guard.clear_training_in_flight(false);
                }
                tracing::warn!(task = %key, "proxy-cache fit failed: {error}");
            }
        }
    }

    fn task_dir(&self, key: &str) -> PathBuf {
        self.config.data_dir.join("tasks").join(key)
    }

    /// Status of one task (None when unknown).
    pub fn task_status(&self, key: &str) -> Option<super::engine::TaskStatus> {
        let state = self.state.lock().ok()?;
        let engine = state.engines.get(key)?;
        let guard = engine.lock().ok()?;
        Some(guard.status())
    }

    /// All loaded task statuses.
    pub fn loaded_status(&self) -> Vec<super::engine::TaskStatus> {
        let state = match self.state.lock() {
            Ok(state) => state,
            Err(_) => return Vec::new(),
        };
        state
            .engines
            .values()
            .filter_map(|engine| engine.lock().ok().map(|guard| guard.status()))
            .collect()
    }
}

fn drop_input(_input: FitInput) {
    // Rows are consumed by value; nothing to do beyond dropping.
}

fn touch_lru(lru: &mut VecDeque<String>, key: &str) {
    if let Some(position) = lru.iter().position(|entry| entry == key) {
        lru.remove(position);
    }
    lru.push_back(key.to_owned());
}

fn load_or_create_salt(data_dir: &std::path::Path) -> Result<Vec<u8>, ProxyCacheError> {
    let path = data_dir.join("key-salt");
    if let Ok(bytes) = std::fs::read(&path) {
        if bytes.len() == 32 {
            return Ok(bytes);
        }
    }
    let mut salt = vec![0_u8; 32];
    // fastrand is not a CSPRNG; salt stability across restarts matters more
    // than secrecy here (key hashes never leave the process).
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x5eed);
    let mut rng = fastrand::Rng::with_seed(seed ^ fastrand::u64(..));
    for byte in salt.iter_mut() {
        *byte = rng.u8(..);
    }
    std::fs::write(&path, &salt)
        .map_err(|error| ProxyCacheError::Store(format!("write salt: {error}")))?;
    Ok(salt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy_cache::encoder::HashEmbedder;
    use serde_json::json;

    fn manager(dir: &std::path::Path, admission: usize) -> Arc<ProxyCacheManager> {
        let mut config = ProxyCacheManagerConfig {
            data_dir: dir.to_path_buf(),
            ..Default::default()
        };
        config.admission_min_requests = admission;
        config.default_tenant = "default".into();
        ProxyCacheManager::new(config, Arc::new(HashEmbedder::new(64, 0, true).unwrap())).unwrap()
    }

    fn spec() -> TaskSpec {
        TaskSpec {
            instructions: json!("pick one"),
            criteria: [
                ("a".to_string(), Some("A".to_string())),
                ("b".to_string(), Some("B".to_string())),
            ]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn admission_gates_new_tasks_then_persists() {
        let dir = tempfile::tempdir().unwrap();
        let built = manager(dir.path(), 3);
        let task = spec();
        assert!(built.route_context(&task, "jev-latest").unwrap().is_none());
        assert!(built.route_context(&task, "jev-latest").unwrap().is_none());
        let engine = built.route_context(&task, "jev-latest").unwrap();
        assert!(engine.is_some());

        // Restart: existing task bypasses admission.
        let key = task.task_key("default", "jev-latest");
        assert!(dir
            .path()
            .join("tasks")
            .join(&key)
            .join("task.json")
            .exists());
        let restarted = manager(dir.path(), 3);
        assert!(restarted
            .route_context(&task, "jev-latest")
            .unwrap()
            .is_some());
    }

    #[test]
    fn identical_specs_share_one_task() {
        let dir = tempfile::tempdir().unwrap();
        let built = manager(dir.path(), 1);
        let a = spec();
        let mut b = spec();
        b.criteria = a
            .criteria
            .iter()
            .rev()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let engine_a = built.route_context(&a, "jev-latest").unwrap().unwrap();
        let engine_b = built.route_context(&b, "jev-latest").unwrap().unwrap();
        assert!(Arc::ptr_eq(&engine_a, &engine_b));
        // Sequential locks: both Arcs alias one mutex, so the guards must
        // never overlap.
        let version_a = engine_a.lock().unwrap().status().task_version;
        let version_b = engine_b.lock().unwrap().status().task_version;
        assert_eq!(version_a, version_b);
    }

    #[test]
    fn salt_is_stable_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let first = load_or_create_salt(dir.path()).unwrap();
        let second = load_or_create_salt(dir.path()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 32);
    }
}
