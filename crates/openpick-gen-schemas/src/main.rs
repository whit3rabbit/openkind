use openpick_core::{SystemRequest, SystemResponse};

fn main() {
    let req = schemars::schema_for!(SystemRequest);
    let resp = schemars::schema_for!(SystemResponse);
    println!("REQUEST:{}", serde_json::to_string_pretty(&req).unwrap());
    println!("RESPONSE:{}", serde_json::to_string_pretty(&resp).unwrap());
}
