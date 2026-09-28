//! Engine registry for model alias routing and metadata discovery.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use openkind_core::ModelInfo;

use crate::engine::DecisionEngine;

/// A registry mapping model alias → engine. Lets the server dispatch by
/// the `model` field in the request without the engine itself knowing.
#[derive(Default, Clone)]
pub struct EngineRegistry {
    engines: Arc<RwLock<HashMap<String, Arc<dyn DecisionEngine>>>>,
}

impl std::fmt::Debug for EngineRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineRegistry")
            .field("models", &self.models())
            .finish()
    }
}

impl EngineRegistry {
    /// Construct an empty `EngineRegistry`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a decision engine under the specified model alias (e.g. `"jev-latest"`).
    pub fn register(&mut self, alias: impl Into<String>, engine: Arc<dyn DecisionEngine>) {
        self.engines
            .write()
            .expect("registry lock poisoned")
            .insert(alias.into(), engine);
    }

    /// Look up a decision engine by its registered model alias.
    pub fn get(&self, model: &str) -> Option<Arc<dyn DecisionEngine>> {
        self.engines
            .read()
            .expect("registry lock poisoned")
            .get(model)
            .cloned()
    }

    /// Sorted list of registered alias names.
    pub fn models(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .engines
            .read()
            .expect("registry lock poisoned")
            .keys()
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// `GET /v1/models` payload — one entry per registered alias,
    /// pulling metadata from the engine itself.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        // Snapshot the handles before asking backends for metadata. No registry
        // lock is held during backend code or inference.
        let engines = self.engines.read().expect("registry lock poisoned").clone();
        let mut models: Vec<_> = engines
            .into_iter()
            .map(|(name, engine)| {
                let mut metadata = engine.model_metadata();
                metadata.name = name;
                metadata
            })
            .collect();
        models.sort_by(|a, b| a.name.cmp(&b.name));
        models
    }

    /// Publish a loaded engine without replacing a live alias. Cloned registries
    /// share updates so HTTP and gRPC observe the same model lifecycle.
    pub fn register_if_absent(&self, alias: String, engine: Arc<dyn DecisionEngine>) -> bool {
        use std::collections::hash_map::Entry;
        match self
            .engines
            .write()
            .expect("registry lock poisoned")
            .entry(alias)
        {
            Entry::Vacant(entry) => {
                entry.insert(engine);
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    /// Stop new dispatches. Existing requests retain their engine handle until
    /// completion, so unloading does not interrupt an accepted request.
    pub fn unregister(&self, alias: &str) -> Option<Arc<dyn DecisionEngine>> {
        self.engines
            .write()
            .expect("registry lock poisoned")
            .remove(alias)
    }
}
