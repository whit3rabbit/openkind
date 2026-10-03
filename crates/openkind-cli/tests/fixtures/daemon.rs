use std::env;

fn main() {
    if let Ok(expected) = env::var("OPENKIND_CLI_TEST_EXPECTED_KEY") {
        let actual = env::var("OPENKIND_API_KEY").unwrap_or_else(|_| "unset".into());
        if actual != expected {
            std::process::exit(7);
        }
    }
    for argument in env::args().skip(1) {
        println!("DAEMON_ARG:{argument}");
    }
    let installed = env::var("OPENKIND_INSTALLED_MODELS").unwrap_or_else(|_| "unset".into());
    println!("DAEMON_INSTALLED_MODELS:{installed}");
}
