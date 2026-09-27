//! Composite `DecisionEngine` for the router-script family.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};
use openkind_engine::{DecisionEngine, EngineError, EngineResult};

use super::router::{dominant_script, ScriptRuleTable};

/// Deterministic script router delegating to registered sibling engines.
///
/// The router holds `Arc<dyn DecisionEngine>` handles to already-registered
/// siblings; it adds no admission of its own because the sibling engines
/// enforce theirs. A dropped HTTP/gRPC caller cancels the routed future
/// exactly as it would cancel the sibling directly.
pub struct RouterScriptEngine {
    siblings: HashMap<String, Arc<dyn DecisionEngine>>,
    rules: ScriptRuleTable,
}

impl std::fmt::Debug for RouterScriptEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterScriptEngine")
            .field("siblings", &self.siblings.keys().collect::<Vec<_>>())
            .field("rules", &self.rules)
            .finish()
    }
}

impl RouterScriptEngine {
    /// Construct the router from a rule table and the sibling engines it
    /// references. Every alias the table can produce must resolve to a
    /// registered sibling, otherwise construction fails closed.
    pub fn new(
        rules: ScriptRuleTable,
        siblings: &HashMap<String, Arc<dyn DecisionEngine>>,
    ) -> Result<Self, super::RouterScriptError> {
        for alias in rules.referenced_aliases() {
            if !siblings.contains_key(alias) {
                return Err(super::RouterScriptError::UnregisteredSibling(
                    alias.to_owned(),
                ));
            }
        }
        Ok(Self {
            siblings: siblings.clone(),
            rules,
        })
    }

    /// Route one request's text sample to a sibling engine handle.
    fn route(&self, text: &str) -> Result<(Arc<dyn DecisionEngine>, String), EngineError> {
        let script = dominant_script(text);
        let alias = self
            .rules
            .route(script)
            .ok_or_else(|| EngineError::Unsupported {
                backend: super::FAMILY_SLUG.to_owned(),
                message: format!(
                    "no route for script `{}` and no default route",
                    script.as_str()
                ),
            })?;
        let sibling = self
            .siblings
            .get(alias)
            .ok_or_else(|| EngineError::Unsupported {
                backend: super::FAMILY_SLUG.to_owned(),
                message: format!("routed sibling alias `{alias}` is not registered"),
            })?;
        Ok((Arc::clone(sibling), sibling.backend_id().to_owned()))
    }
}

#[async_trait]
impl DecisionEngine for RouterScriptEngine {
    fn backend_id(&self) -> &str {
        "router-script/detector"
    }

    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: String::new(),
            description: "Deterministic Unicode script router over registered sibling engines."
                .into(),
            release_date: "2026-09-26".into(),
        }
    }

    async fn evaluate(&self, request: SystemRequest) -> EngineResult<SystemResponse> {
        // The router inspects the state and question text, never the
        // candidate list, per the family contract.
        let sample = match &request.state {
            openkind_core::State::Text(text) => text.clone(),
            openkind_core::State::Object(map) => {
                serde_json::to_string(map).map_err(|error| EngineError::Unsupported {
                    backend: super::FAMILY_SLUG.to_owned(),
                    message: format!("state serialization failed: {error}"),
                })?
            }
            openkind_core::State::Array(items) => {
                serde_json::to_string(items).map_err(|error| EngineError::Unsupported {
                    backend: super::FAMILY_SLUG.to_owned(),
                    message: format!("state serialization failed: {error}"),
                })?
            }
        };
        let routed_started = std::time::Instant::now();
        let (sibling, sibling_backend) = self.route(&sample)?;
        metrics::histogram!("openkind_router_script_route_seconds")
            .record(routed_started.elapsed().as_secs_f64());
        metrics::counter!(
            "openkind_router_script_routes_total",
            "sibling" => sibling_backend,
        )
        .increment(1);
        sibling.evaluate(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkind_core::SystemRequest;
    use openkind_engine::MockEngine;
    use serde_json::json;

    fn router(rules: &str) -> RouterScriptEngine {
        let table = ScriptRuleTable::parse(rules).expect("parse rules");
        let mut siblings: HashMap<String, Arc<dyn DecisionEngine>> = HashMap::new();
        siblings.insert("latin-sibling".to_owned(), Arc::new(MockEngine::new()));
        siblings.insert("cyrillic-sibling".to_owned(), Arc::new(MockEngine::new()));
        RouterScriptEngine::new(table, &siblings).expect("compose router")
    }

    fn request(state: &str) -> SystemRequest {
        serde_json::from_value(json!({
            "state": state,
            "model": "router-script",
            "questions": {
                "q0": { "type": "noul", "instructions": "Does the record state the fact?" }
            }
        }))
        .expect("request")
    }

    #[tokio::test]
    async fn routes_latin_and_cyrillic_deterministically() {
        let engine = router("latin=latin-sibling,cyrillic=cyrillic-sibling,default=latin-sibling");
        let latin_response = engine
            .evaluate(request("Fictional incident record with plain text."))
            .await
            .expect("route latin");
        let cyrillic_response = engine
            .evaluate(request("Сводка инцидента на русском языке."))
            .await
            .expect("route cyrillic");
        assert_eq!(latin_response.model, "router-script");
        assert_eq!(cyrillic_response.model, "router-script");
        assert!(latin_response.answers.contains_key("q0"));
        assert!(cyrillic_response.answers.contains_key("q0"));
    }

    #[tokio::test]
    async fn default_route_covers_unrouted_scripts() {
        let engine = router("latin=latin-sibling,cyrillic=cyrillic-sibling,default=latin-sibling");
        let response = engine
            .evaluate(request("事故レポートのテキスト。"))
            .await
            .expect("default route");
        assert!(response.answers.contains_key("q0"));
    }

    #[test]
    fn construction_fails_closed_on_unregistered_siblings() {
        let table = ScriptRuleTable::parse("latin=missing-sibling").expect("parse");
        let siblings: HashMap<String, Arc<dyn DecisionEngine>> = HashMap::new();
        let error = RouterScriptEngine::new(table, &siblings)
            .expect_err("unregistered sibling must fail construction");
        assert!(error
            .to_string()
            .contains("unregistered sibling alias `missing-sibling`"));
    }

    #[test]
    fn no_default_route_fails_closed_on_unknown_script() {
        let table = ScriptRuleTable::parse("latin=latin-sibling").expect("parse");
        let mut siblings: HashMap<String, Arc<dyn DecisionEngine>> = HashMap::new();
        siblings.insert("latin-sibling".to_owned(), Arc::new(MockEngine::new()));
        let engine = RouterScriptEngine::new(table, &siblings).expect("compose router");
        let error = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime")
            .block_on(engine.evaluate(request("事故レポートのテキスト。")))
            .expect_err("kana input without a route must fail");
        assert!(matches!(error, EngineError::Unsupported { .. }));
    }
}
