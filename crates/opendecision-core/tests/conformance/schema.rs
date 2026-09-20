use opendecision_core::*;
use serde_json::json;

// ------------------------------------------------------------------
// JSON Schema generation (used by MCP adapters, OpenAPI clients)
// ------------------------------------------------------------------

#[test]
fn json_schema_generates_for_request_and_response() {
    let req_schema = schemars::schema_for!(SystemRequest);
    let resp_schema = schemars::schema_for!(SystemResponse);

    let req_json = serde_json::to_value(&req_schema).unwrap();
    let resp_json = serde_json::to_value(&resp_schema).unwrap();

    // Sanity: schemas are non-empty objects.
    assert!(req_json.is_object());
    assert!(resp_json.is_object());

    // Spot-check: SystemRequest has a "questions" property of type "object".
    let props = req_json
        .get("properties")
        .and_then(|p| p.get("questions"))
        .unwrap();
    assert_eq!(props.get("type").and_then(|t| t.as_str()), Some("object"));
}

// ------------------------------------------------------------------
// Spec section: "model" field — exact string equality preserved
// ------------------------------------------------------------------

#[test]
fn model_string_is_preserved_exactly() {
    // Per the literal-preservation rule: don't normalize tokens.
    let raw = json!({
        "state": "x",
        "model": "custom-alias/with-slashes",
        "questions": { "q": { "type": "noul", "instructions": "?" } }
    });
    let req: SystemRequest = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!(req.model, "custom-alias/with-slashes");
    assert_eq!(serde_json::to_value(&req).unwrap(), raw);
}
