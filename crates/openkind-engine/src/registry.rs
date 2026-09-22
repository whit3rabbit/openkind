//! Engine registry for model alias routing and metadata discovery.

use std::collections::HashMap;
use std::sync::Arc;

use openkind_core::ModelInfo;

use crate::engine::DecisionEngine;

/// A registry mapping model alias → engine. Lets the server dispatch by
/// the `model` field in the request without the engine itself knowing.
#[derive(Default, Clone)]
pub struct EngineRegistry {
    engines: HashMap<String, Arc<dyn DecisionEngine>>,
}

impl std::fmt::Debug for EngineRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineRegistry")
            .field("models", &self.engines.keys().collect::<Vec<_>>())
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
        self.engines.insert(alias.into(), engine);
    }

    /// Look up a decision engine by its registered model alias.
    pub fn get(&self, model: &str) -> Option<Arc<dyn DecisionEngine>> {
        self.engines.get(model).cloned()
    }

    /// Sorted list of registered alias names.
    pub fn models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.engines.keys().cloned().collect();
        v.sort();
        v
    }

    /// `GET /v1/models` payload — one entry per registered alias,
    /// pulling metadata from the engine itself.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let aliases = self.models();
        aliases
            .into_iter()
            .filter_map(|name| {
                self.engines
                    .get(&name)
                    .map(|engine| engine.model_metadata())
                    .map(|mut m| {
                        // The alias registered with the engine is the
                        // canonical name the SDK uses, not the engine's
                        // internal backend id. Override here.
                        m.name = name;
                        m
                    })
            })
            .collect()
    }
}
