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
        let alias = alias.into();
        let replaced = self
            .engines
            .write()
            .expect("registry lock poisoned")
            .insert(alias, engine);
        // Backend cleanup can be slow or consult the registry itself.
        drop(replaced);
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;
    use openkind_core::{SystemRequest, SystemResponse};

    use super::*;
    use crate::{EngineResult, MockEngine};

    struct CleanupBackend {
        registry: EngineRegistry,
        cleanup_had_access: Arc<AtomicBool>,
    }

    #[async_trait]
    impl DecisionEngine for CleanupBackend {
        fn backend_id(&self) -> &str {
            "cleanup"
        }

        async fn evaluate(&self, _req: SystemRequest) -> EngineResult<SystemResponse> {
            unreachable!("cleanup-only test")
        }
    }

    impl Drop for CleanupBackend {
        fn drop(&mut self) {
            self.cleanup_had_access
                .store(self.registry.engines.try_write().is_ok(), Ordering::SeqCst);
        }
    }

    #[test]
    fn replacement_releases_registry_lock_before_backend_cleanup() {
        let mut registry = EngineRegistry::new();
        let cleanup_had_access = Arc::new(AtomicBool::new(false));
        registry.register(
            "model",
            Arc::new(CleanupBackend {
                registry: registry.clone(),
                cleanup_had_access: cleanup_had_access.clone(),
            }),
        );

        registry.register("model", Arc::new(MockEngine::new()));

        assert!(cleanup_had_access.load(Ordering::SeqCst));
        assert_eq!(registry.get("model").unwrap().backend_id(), "mock");
    }

    #[test]
    fn rejected_registration_releases_lock_before_backend_cleanup() {
        let mut registry = EngineRegistry::new();
        registry.register("model", Arc::new(MockEngine::new()));
        let cleanup_had_access = Arc::new(AtomicBool::new(false));

        assert!(!registry.register_if_absent(
            "model".into(),
            Arc::new(CleanupBackend {
                registry: registry.clone(),
                cleanup_had_access: cleanup_had_access.clone(),
            }),
        ));

        assert!(cleanup_had_access.load(Ordering::SeqCst));
        assert_eq!(registry.get("model").unwrap().backend_id(), "mock");
    }
}
