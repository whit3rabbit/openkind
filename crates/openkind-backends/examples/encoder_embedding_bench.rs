//! Opt-in BGE sentence-embedding latency benchmark.
//!
//! Reads an installed local checkpoint. Timed calls include tokenization and
//! one single-text encode, matching the proxy-cache request path; model loading
//! and warmup are excluded.

use std::{
    error::Error,
    path::PathBuf,
    time::{Duration, Instant},
};

use openkind_backends::proxy_cache::{
    bert_encoder::{BertEmbedder, BertEmbedderArtifacts},
    TextEmbedder,
};
use sha2::{Digest, Sha256};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use openkind_backends::proxy_cache::mlx_bert_encoder::MlxBertEmbedder;

const MODEL_REPOSITORY: &str = "BAAI/bge-small-en-v1.5";
const MODEL_REVISION: &str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";
const STATES: [&str; 3] = [
    r#"{"customer":"Acme","issue":"Invoice total differs from the purchase order","ticket_id":"T-1042"}"#,
    r#"{"customer":"Northwind","issue":"The account owner cannot sign in after enabling the new identity provider. The error appears after the redirect, and the previous password reset did not change the result.","ticket_id":"T-2088"}"#,
    r#"{"customer":"Contoso","issue":"A billing administrator needs a copy of the tax invoice for the April renewal. They have confirmed the workspace, renewal date, and invoice number, but the billing page shows only the payment receipt.","ticket_id":"T-3317"}"#,
];

#[derive(Clone, Copy)]
enum Backend {
    Cpu,
    Mlx,
}

struct Args {
    model_root: PathBuf,
    backend: Backend,
    host: String,
    commit: String,
    working_tree_dirty: bool,
    warmup_calls: usize,
    sample_calls: usize,
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut model_root =
        std::env::var_os("OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT").map(PathBuf::from);
    let mut backend = Backend::Cpu;
    let mut host = None;
    let mut commit = None;
    let mut working_tree_dirty = None;
    let mut warmup_calls = 5;
    let mut sample_calls = 30;
    let mut args = std::env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "Usage: encoder_embedding_bench --host LABEL --commit HASH \
                 --working-tree-dirty true|false [--model-root DIR] \
                 [--backend cpu|mlx] [--warmup-calls N] [--sample-calls N]\n\
                 Default model root: OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT"
            );
            std::process::exit(0);
        }
        let value = args.next().ok_or_else(|| {
            invalid_input(format!(
                "missing value for {flag}; run with --help for usage"
            ))
        })?;
        match flag.as_str() {
            "--model-root" => model_root = Some(PathBuf::from(value)),
            "--backend" => {
                backend = match value.as_str() {
                    "cpu" => Backend::Cpu,
                    "mlx" => Backend::Mlx,
                    _ => return Err(invalid_input("--backend must be cpu or mlx")),
                }
            }
            "--host" => host = Some(value),
            "--commit" => commit = Some(value),
            "--working-tree-dirty" => {
                working_tree_dirty = Some(
                    value
                        .parse::<bool>()
                        .map_err(|_| invalid_input("--working-tree-dirty must be true or false"))?,
                )
            }
            "--warmup-calls" => warmup_calls = value.parse()?,
            "--sample-calls" => sample_calls = value.parse()?,
            _ => return Err(invalid_input(format!("unknown option: {flag}"))),
        }
    }

    let model_root = model_root.ok_or_else(|| {
        invalid_input(
            "pass --model-root or set OPENKIND_ENCODER_EMBEDDING_MODEL_ROOT to a verified local model directory",
        )
    })?;
    let host = host.ok_or_else(|| invalid_input("pass --host with the hardware label"))?;
    let commit = commit.ok_or_else(|| invalid_input("pass --commit with the source revision"))?;
    let working_tree_dirty = working_tree_dirty.ok_or_else(|| {
        invalid_input("pass --working-tree-dirty true|false to record source state")
    })?;
    if sample_calls < 2 {
        return Err(invalid_input("--sample-calls must be at least 2"));
    }
    Ok(Args {
        model_root,
        backend,
        host,
        commit,
        working_tree_dirty,
        warmup_calls,
        sample_calls,
    })
}

fn percentile(sorted: &[Duration], percentile: f64) -> Duration {
    let index = ((percentile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn benchmark(
    embedder: &dyn TextEmbedder,
    args: &Args,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let states: Vec<String> = STATES.iter().map(|state| (*state).to_owned()).collect();
    let mut digest = Sha256::new();
    for state in &states {
        digest.update(state.as_bytes());
        digest.update(b"\n");
    }
    let input_sha256 = format!("{:x}", digest.finalize());

    for index in 0..args.warmup_calls {
        let state = &states[index % states.len()];
        let vectors = embedder.encode(std::slice::from_ref(state))?;
        if vectors.len() != 1 || vectors[0].len() != embedder.dim() {
            return Err(invalid_input("encoder returned an unexpected output shape"));
        }
    }

    let mut samples = Vec::with_capacity(args.sample_calls);
    for index in 0..args.sample_calls {
        let state = &states[index % states.len()];
        let start = Instant::now();
        let vectors = embedder.encode(std::slice::from_ref(state))?;
        samples.push(start.elapsed());
        if vectors.len() != 1 || vectors[0].len() != embedder.dim() {
            return Err(invalid_input("encoder returned an unexpected output shape"));
        }
    }

    let elapsed_total: Duration = samples.iter().copied().sum();
    samples.sort_unstable();
    let median_ms = if samples.len() % 2 == 0 {
        (samples[samples.len() / 2 - 1].as_secs_f64() + samples[samples.len() / 2].as_secs_f64())
            * 500.0
    } else {
        samples[samples.len() / 2].as_secs_f64() * 1_000.0
    };
    let p95_ms = percentile(&samples, 0.95).as_secs_f64() * 1_000.0;
    let embeddings_per_second = args.sample_calls as f64 / elapsed_total.as_secs_f64();
    let average_input_bytes =
        STATES.iter().map(|state| state.len()).sum::<usize>() as f64 / STATES.len() as f64;

    Ok(serde_json::json!({
        "schema": "openkind-encoder-embedding-bench/v1",
        "model": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "encoder_id": embedder.id(),
        "backend": embedder.backend_id(),
        "embedding_dimensions": embedder.dim(),
        "input_sha256": input_sha256,
        "input_count": STATES.len(),
        "average_input_bytes": average_input_bytes,
        "warmup_calls": args.warmup_calls,
        "sample_calls": args.sample_calls,
        "host": args.host,
        "commit": args.commit,
        "working_tree_dirty": args.working_tree_dirty,
        "p50_ms": median_ms,
        "p95_ms": p95_ms,
        "embeddings_per_second": embeddings_per_second,
        "host_os": std::env::consts::OS,
        "host_arch": std::env::consts::ARCH,
    }))
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let artifacts = BertEmbedderArtifacts::from_model_root(&args.model_root, MODEL_REPOSITORY);
    let report = match args.backend {
        Backend::Cpu => {
            let embedder = BertEmbedder::load(&artifacts)?;
            benchmark(&embedder, &args)?
        }
        Backend::Mlx => {
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            {
                let embedder = MlxBertEmbedder::load(&artifacts)?;
                benchmark(&embedder, &args)?
            }
            #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
            {
                return Err(invalid_input("MLX requires --features mlx on macOS arm64"));
            }
        }
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("encoder_embedding_bench: {error}");
        std::process::exit(2);
    }
}
