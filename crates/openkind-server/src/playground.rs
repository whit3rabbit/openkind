//! Explicit local model lifecycle for the opt-in playground.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use openkind_api::playground::{PlaygroundModel, PlaygroundModels};
use openkind_api::ApiError;
use openkind_engine::{EngineRegistry, MockEngine};
use openkind_model_store::{default_models_dir, InstalledModel, ModelStore};
use tokio::sync::Mutex;

use crate::args::Args;
use crate::installed::{installed_kind, load_installed_engine};

pub(crate) struct LocalModels {
    args: Arc<Args>,
    registry: Arc<EngineRegistry>,
    store: Arc<ModelStore>,
    mocks: HashSet<String>,
    startup: HashSet<String>,
    // A timed-out native request can still own a blocking task. Retaining the
    // small disk leases until shutdown prevents `rm` racing those tasks, even
    // after unloading releases the registry's model weights.
    leases: Arc<Mutex<HashMap<String, InstalledModel>>>,
}

impl LocalModels {
    pub(crate) fn new(
        args: Arc<Args>,
        registry: Arc<EngineRegistry>,
        installed: Vec<InstalledModel>,
    ) -> anyhow::Result<Self> {
        let dir = args
            .models_dir
            .clone()
            .map(Ok)
            .unwrap_or_else(default_models_dir)?;
        let mocks = args
            .models
            .iter()
            .filter(|name| {
                registry
                    .get(name)
                    .is_some_and(|engine| engine.backend_id() == "mock")
            })
            .cloned()
            .collect();
        Ok(Self {
            startup: args.models.iter().cloned().collect(),
            args,
            registry,
            store: Arc::new(ModelStore::new(dir)?),
            mocks,
            leases: Arc::new(Mutex::new(
                installed
                    .into_iter()
                    .map(|model| (model.manifest.name.clone(), model))
                    .collect(),
            )),
        })
    }
}

fn invalid(error: impl std::fmt::Display) -> ApiError {
    ApiError::InvalidBody(error.to_string())
}

#[async_trait::async_trait]
impl PlaygroundModels for LocalModels {
    async fn list(&self) -> Result<Vec<PlaygroundModel>, ApiError> {
        let store = self.store.clone();
        let installed = tokio::task::spawn_blocking(move || store.list())
            .await
            .map_err(|e| ApiError::Internal(e.to_string()))?
            .map_err(invalid)?;
        let mut models = BTreeMap::new();
        for manifest in installed {
            let manageable = installed_kind(&manifest).is_some();
            let name = manifest.name;
            models.insert(
                name.clone(),
                PlaygroundModel {
                    loaded: self.registry.get(&name).is_some(),
                    name,
                    description: manifest.description,
                    source: "installed".into(),
                    manageable,
                },
            );
        }
        for name in &self.startup {
            let mock = self.mocks.contains(name);
            models.insert(
                name.clone(),
                PlaygroundModel {
                    name: name.clone(),
                    description: if mock {
                        "Deterministic demo engine. No model weights or quality claims.".into()
                    } else {
                        "Configured at startup. Restart the daemon to change this engine.".into()
                    },
                    source: if mock { "mock" } else { "startup" }.into(),
                    loaded: self.registry.get(name).is_some(),
                    manageable: mock,
                },
            );
        }
        Ok(models.into_values().collect())
    }

    async fn set_loaded(&self, name: String, loaded: bool) -> Result<(), ApiError> {
        if self.startup.contains(&name) && !self.mocks.contains(&name) {
            return Err(invalid("This startup engine requires a daemon restart"));
        }
        // Move the permit into the blocking task. Disconnecting the browser
        // must not release it while verification/loading is still running.
        let mut leases =
            self.leases
                .clone()
                .try_lock_owned()
                .map_err(|_| ApiError::Overloaded {
                    retry_after_ms: 1_000,
                })?;
        let store = self.store.clone();
        let registry = self.registry.clone();
        let args = self.args.clone();
        let mock = self.mocks.contains(&name);
        tokio::task::spawn_blocking(move || {
            if !loaded {
                registry
                    .unregister(&name)
                    .ok_or_else(|| invalid("Model is not loaded"))?;
                return Ok(());
            }
            if registry.get(&name).is_some() {
                return Err(invalid("Model is already loaded"));
            }
            let engine = if mock {
                Arc::new(MockEngine::new()) as Arc<dyn openkind_engine::DecisionEngine>
            } else {
                let installed = store.acquire_serving(&name).map_err(invalid)?;
                let manifest = &installed.manifest;
                let kind = installed_kind(manifest)
                    .ok_or_else(|| invalid("Unsupported installed model profile"))?;
                let root = installed.root.clone();
                let engine =
                    load_installed_engine(&args, kind, &root, &registry).map_err(invalid)?;
                leases.insert(name.clone(), installed);
                engine
            };
            if !registry.register_if_absent(name, engine) {
                return Err(invalid("Model alias is already in use"));
            }
            Ok(())
        })
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn manager(root: &std::path::Path) -> LocalModels {
        let args = Args::parse_from([
            "openkindd",
            "--models",
            "mock",
            "--models-dir",
            root.to_str().unwrap(),
        ]);
        let mut registry = EngineRegistry::new();
        registry.register("mock", Arc::new(MockEngine::new()));
        LocalModels::new(Arc::new(args), Arc::new(registry), vec![]).unwrap()
    }

    #[tokio::test]
    async fn explicit_load_unload_and_failed_load_leave_consistent_inventory() {
        let root = tempfile::tempdir().unwrap();
        let manager = manager(root.path());
        let peer = manager.registry.as_ref().clone();
        assert!(manager.list().await.unwrap()[0].loaded);
        manager.set_loaded("mock".into(), false).await.unwrap();
        assert!(peer.get("mock").is_none());
        assert!(!manager.list().await.unwrap()[0].loaded);
        manager.set_loaded("mock".into(), true).await.unwrap();
        assert!(peer.get("mock").is_some());
        assert!(manager.set_loaded("mock".into(), true).await.is_err());
        assert!(manager.set_loaded("missing".into(), true).await.is_err());
        assert!(manager.set_loaded("../escape".into(), true).await.is_err());
        assert_eq!(peer.models(), vec!["mock"]);
    }

    #[tokio::test]
    async fn concurrent_mutation_is_rejected_and_startup_engines_are_pinned() {
        let root = tempfile::tempdir().unwrap();
        let mut manager = manager(root.path());
        let guard = manager.leases.lock().await;
        assert!(matches!(
            manager.set_loaded("mock".into(), false).await,
            Err(ApiError::Overloaded { .. })
        ));
        drop(guard);
        manager.mocks.clear();
        assert!(manager.set_loaded("mock".into(), false).await.is_err());
        assert!(manager.registry.get("mock").is_some());
    }
}
