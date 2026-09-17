use openpick_core::{SystemRequest, SystemResponse};
use std::path::Path;

fn main() {
    let req = schemars::schema_for!(SystemRequest);
    let resp = schemars::schema_for!(SystemResponse);
    let req_json = serde_json::to_string_pretty(&req).expect("serialize request schema");
    let resp_json = serde_json::to_string_pretty(&resp).expect("serialize response schema");

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--write" || a == "-w") {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let schema_dir = Path::new(&manifest_dir).join("../openpick-core/schemas");
        if schema_dir.exists() {
            let req_path = schema_dir.join("jev-v1-request.json");
            let resp_path = schema_dir.join("jev-v1-response.json");
            std::fs::write(&req_path, format!("{}\n", req_json)).expect("write request schema");
            std::fs::write(&resp_path, format!("{}\n", resp_json)).expect("write response schema");
            eprintln!(
                "Successfully updated JSON schemas in {}",
                schema_dir.display()
            );
        } else {
            eprintln!("Schema directory not found at {}", schema_dir.display());
        }
    }

    println!("REQUEST:{}", req_json);
    println!("RESPONSE:{}", resp_json);
}
