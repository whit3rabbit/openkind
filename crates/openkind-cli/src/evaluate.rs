use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use openkind_core::{validate_response_for_request, Answer, SystemResponse};

use crate::args::EvaluateFormat;
use crate::inspect::{
    parse_and_validate_request, read_bounded_input, read_request_file, MAX_CLI_INPUT_BYTES,
};
use crate::output;

/// Asynchronously evaluates a decision request against a remote openkind server.
pub async fn cmd_evaluate_async(
    file: PathBuf,
    server: String,
    api_key: Option<String>,
    pretty: bool,
    format: EvaluateFormat,
    verbose: bool,
) -> Result<()> {
    let raw = tokio::task::spawn_blocking(move || {
        if file.as_os_str() == "-" {
            read_bounded_input(std::io::stdin().lock(), MAX_CLI_INPUT_BYTES)
                .context("read request from stdin")
        } else {
            read_request_file(&file)
        }
    })
    .await
    .context("request read task")??;
    let request = parse_and_validate_request(&raw)?;

    let url = format!("{}/v1/systemone", server.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none()) // Prevent leaking Authorization header across redirects
        .build()?;
    let mut req_builder = client
        .post(&url)
        .header("content-type", "application/json")
        .body(raw);

    let resolved_key = api_key
        .or_else(|| std::env::var("OPENKIND_API_KEY").ok())
        .or_else(|| std::env::var("TYPESAFE_API_KEY").ok())
        .filter(|s| !s.is_empty());
    if let Some(key) = resolved_key {
        req_builder = req_builder.header("authorization", format!("Bearer {key}"));
    }

    let started = Instant::now();
    let resp = req_builder
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    let status = resp.status();
    let request_id = resp
        .headers()
        .get("x-typesafe-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = resp.text().await.context("read response body")?;
    let elapsed = started.elapsed();
    if !status.is_success() {
        eprintln!("HTTP {status}");
        if let Some(request_id) = request_id {
            eprintln!("Request ID: {}", output::terminal_safe(&request_id));
        }
        if !body.is_empty() {
            eprintln!("{}", output::terminal_safe(&body));
        }
        anyhow::bail!("evaluation request failed with HTTP {status}");
    }

    // Every output mode crosses the same request-bound validation boundary
    // before emitting answers that a caller may use to make a decision.
    let response: SystemResponse =
        serde_json::from_str(&body).context("parse evaluation response JSON")?;
    validate_response_for_request(&response, &request).context("validate evaluation response")?;

    match format {
        EvaluateFormat::Json => {
            if pretty {
                let value: serde_json::Value = serde_json::from_str(&body)?;
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                println!("{body}");
            }
            if verbose {
                eprintln!(
                    "Evaluation completed in {:.1} ms ({} input tokens, {} output tokens)",
                    elapsed.as_secs_f64() * 1000.0,
                    response.usage.input_tokens,
                    response.usage.output_tokens
                );
            }
        }
        EvaluateFormat::Text => {
            render_text_response(&response, elapsed, verbose);
        }
    }
    Ok(())
}

fn render_text_response(response: &SystemResponse, elapsed: std::time::Duration, verbose: bool) {
    output::print_heading("Evaluation");
    output::print_key_value("Model", &response.model);
    let mut answer_ids: Vec<&String> = response.answers.keys().collect();
    answer_ids.sort();
    let rows: Vec<Vec<String>> = answer_ids
        .iter()
        .map(|question_id| {
            let answer = &response.answers[*question_id];
            let (kind, value, confidence, details) = match answer {
                Answer::Noul(answer) => (
                    "Noul",
                    format!("noul={:.3}", answer.noul),
                    "not provided".to_owned(),
                    if verbose {
                        format!("p(noul)={:.3}", answer.noul)
                    } else {
                        String::new()
                    },
                ),
                Answer::Choice(answer) => (
                    "Choice",
                    answer.choice.clone(),
                    format!("{:.3}", answer.confidence),
                    if verbose {
                        format_probabilities(&answer.probabilities, None)
                    } else {
                        String::new()
                    },
                ),
                Answer::Score(answer) => (
                    "Score",
                    format!("{:.3}", answer.score),
                    format!("{:.3}", answer.confidence),
                    if verbose {
                        format_probabilities(&answer.probabilities, Some(&answer.legend))
                    } else {
                        String::new()
                    },
                ),
            };
            let mut row = vec![(*question_id).clone(), kind.to_owned(), value, confidence];
            if verbose {
                row.push(details);
            }
            row
        })
        .collect();
    if verbose {
        output::print_table(
            "Answers",
            &["QUESTION", "TYPE", "ANSWER", "CONFIDENCE", "PROBABILITIES"],
            &rows,
        );
    } else {
        output::print_table(
            "Answers",
            &["QUESTION", "TYPE", "ANSWER", "CONFIDENCE"],
            &rows,
        );
    }
    if verbose {
        println!(
            "Completed in {:.1} ms, {} input tokens, {} output tokens",
            elapsed.as_secs_f64() * 1000.0,
            response.usage.input_tokens,
            response.usage.output_tokens
        );
    }
}

fn format_probabilities(
    probabilities: &HashMap<String, f64>,
    legend: Option<&HashMap<String, String>>,
) -> String {
    let mut entries: Vec<_> = probabilities.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
        .into_iter()
        .map(|(key, probability)| {
            if let Some(description) = legend.and_then(|legend| legend.get(key)) {
                format!("{key} ({description})={probability:.3}")
            } else {
                format!("{key}={probability:.3}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Synchronous entrypoint for the `evaluate` CLI subcommand, executing within a fresh Tokio runtime.
pub fn cmd_evaluate(
    file: PathBuf,
    server: String,
    api_key: Option<String>,
    pretty: bool,
    format: EvaluateFormat,
    verbose: bool,
) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(cmd_evaluate_async(
        file, server, api_key, pretty, format, verbose,
    ))
}
