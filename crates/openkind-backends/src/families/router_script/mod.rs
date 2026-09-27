//! Family: `router-script`.
//!
//! Lightweight language/script detection with no model forward pass. The
//! router inspects the request's state and question text, picks a sibling
//! family alias from a fixed rule table, and delegates the request to that
//! sibling engine. Routing is deterministic and sub-millisecond; the
//! response semantics are entirely the sibling family's.
//!
//! Profile semantics:
//! - Backbone class: none — Unicode script ranges plus a fixed rule table.
//! - Probability space: inherited from the routed sibling.
//! - Text generation: none of its own; delegation preserves the Jev wire
//!   contract (`request.model` is untouched).
//! - `default` route: every script without an explicit rule, including
//!   unassigned and symbol-only text.

mod engine;
mod router;

pub use self::engine::RouterScriptEngine;
pub use self::router::{detect_script, dominant_script, Script, ScriptRuleTable};

use thiserror::Error;

/// Family slug used for identity derivation and telemetry.
pub const FAMILY_SLUG: &str = "router-script";

/// Errors raised while configuring or evaluating the router-script family.
#[derive(Debug, Error)]
pub enum RouterScriptError {
    /// The rule table referenced a sibling alias that is not registered.
    #[error("router-script rule references unregistered sibling alias `{0}`")]
    UnregisteredSibling(String),

    /// The rule table had no `default` route and the input matched no rule.
    #[error("router-script has no route for the detected script `{0}` and no default route")]
    NoRoute(String),

    /// The rule table itself was malformed.
    #[error("invalid router-script rule table: {0}")]
    InvalidRules(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_stable() {
        assert_eq!(FAMILY_SLUG, "router-script");
    }
}
