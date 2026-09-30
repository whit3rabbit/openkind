use std::io::IsTerminal;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bubbletea_rs::{quit, Cmd, KeyMsg, Model, Msg, Program};
use crossterm::event::{KeyCode, KeyModifiers};
use openkind_core::{ModelInfo, ModelsResponse};
use reqwest::header::AUTHORIZATION;

use crate::output;

const REFRESH_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Clone)]
struct WatchConfig {
    client: reqwest::Client,
    server: String,
    api_key: Option<String>,
}

struct StartWatchMsg;

struct ConfigureWatchMsg(WatchConfig);

struct RefreshStatusMsg(std::result::Result<Vec<ModelInfo>, String>);

struct StatusWatch {
    config: Option<WatchConfig>,
    status: Option<std::result::Result<Vec<ModelInfo>, String>>,
    checked_at: Option<Instant>,
}

impl StatusWatch {
    fn refresh(config: WatchConfig, delay: Duration) -> Cmd {
        Box::pin(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let result = fetch_status(&config.client, &config.server, config.api_key.as_deref())
                .await
                .map(|response| response.models)
                .map_err(|error| format!("{error:#}"));
            Some(Box::new(RefreshStatusMsg(result)) as Msg)
        })
    }

    fn view_status(&self, lines: &mut Vec<String>) {
        match &self.status {
            None => {
                lines.push(format!("Health: {}", output::style_muted("checking...")));
                lines.push("Registered model aliases: checking...".to_owned());
            }
            Some(Ok(models)) => {
                lines.push(format!("Health: {}", output::style_success("healthy")));
                lines.push(String::new());
                lines.push(output::style_heading("Registered model aliases"));
                if models.is_empty() {
                    lines.push("No model aliases are registered with this daemon.".to_owned());
                } else {
                    let rows: Vec<Vec<String>> = models
                        .iter()
                        .map(|model| vec![model.name.clone(), model.release_date.clone()])
                        .collect();
                    lines.push(output::render_table(&["ALIAS", "RELEASE DATE"], &rows));
                }
            }
            Some(Err(error)) => {
                lines.push(format!(
                    "Daemon status: {}",
                    output::style_error("check failed")
                ));
                lines.push(format!("Error: {}", output::style_error(error)));
            }
        }
    }
}

impl Model for StatusWatch {
    fn init() -> (Self, Option<Cmd>) {
        let start = Box::pin(async { Some(Box::new(StartWatchMsg) as Msg) });
        (
            Self {
                config: None,
                status: None,
                checked_at: None,
            },
            Some(start),
        )
    }

    fn update(&mut self, msg: Msg) -> Option<Cmd> {
        if let Some(key) = msg.downcast_ref::<KeyMsg>() {
            match key.key {
                KeyCode::Char('q') | KeyCode::Esc => return Some(quit()),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Some(quit());
                }
                _ => {}
            }
        }

        if let Some(config) = msg.downcast_ref::<ConfigureWatchMsg>() {
            self.config = Some(config.0.clone());
            return Some(Self::refresh(config.0.clone(), Duration::ZERO));
        }

        if let Some(status) = msg.downcast_ref::<RefreshStatusMsg>() {
            self.status = Some(status.0.clone());
            self.checked_at = Some(Instant::now());
            return self
                .config
                .clone()
                .map(|config| Self::refresh(config, REFRESH_INTERVAL));
        }

        None
    }

    fn view(&self) -> String {
        let mut lines = vec![
            output::style_heading("OpenKind daemon status"),
            output::style_muted("Live view, refreshes every 5 seconds"),
            String::new(),
        ];
        if let Some(config) = &self.config {
            lines.push(format!("Server: {}", output::terminal_safe(&config.server)));
        } else {
            lines.push("Server: connecting...".to_owned());
        }
        if let Some(checked_at) = self.checked_at {
            lines.push(format!(
                "Last checked: {}s ago",
                checked_at.elapsed().as_secs()
            ));
        }
        lines.push(String::new());
        self.view_status(&mut lines);
        lines.push(String::new());
        lines.push(output::style_muted("Press q, Esc, or Ctrl-C to quit"));
        lines.join("\n")
    }
}

pub fn run(server: String, api_key: Option<String>, watch: bool) -> Result<()> {
    let server = server.trim_end_matches('/').to_owned();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    if watch {
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            anyhow::bail!(
                "status --watch requires an interactive terminal; run `openkind status` for a one-shot report"
            );
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        return runtime.block_on(run_watch(WatchConfig {
            client,
            server,
            api_key,
        }));
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let models = runtime.block_on(fetch_status(&client, &server, api_key.as_deref()))?;
    print_status(&server, models);
    Ok(())
}

async fn run_watch(config: WatchConfig) -> Result<()> {
    let filter_config = config.clone();
    let program = Program::<StatusWatch>::builder()
        .alt_screen(true)
        .with_fps(10)
        // Bubble Tea initializes models through a static `init`; pass CLI
        // configuration into the model on its first startup message.
        .filter(move |_model, msg| {
            if msg.downcast_ref::<StartWatchMsg>().is_some() {
                Some(Box::new(ConfigureWatchMsg(filter_config.clone())) as Msg)
            } else {
                Some(msg)
            }
        })
        .build()?;
    program.run().await?;
    Ok(())
}

async fn fetch_status(
    client: &reqwest::Client,
    server: &str,
    api_key: Option<&str>,
) -> Result<ModelsResponse> {
    let health_url = format!("{server}/health");
    let health = client
        .get(&health_url)
        .send()
        .await
        .with_context(|| format!("GET {health_url}"))?;
    let health_status = health.status();
    let health_body = health.text().await.context("read health response")?;
    if !health_status.is_success() {
        anyhow::bail!(
            "GET {health_url} returned HTTP {health_status}: {}",
            output::terminal_safe(&health_body)
        );
    }
    let health_json: serde_json::Value =
        serde_json::from_str(&health_body).context("parse health response JSON")?;
    if health_json.get("status").and_then(|value| value.as_str()) != Some("ok") {
        anyhow::bail!(
            "GET {health_url} returned an unexpected health response: {}",
            output::terminal_safe(&health_body)
        );
    }

    let models_url = format!("{server}/v1/models");
    let mut request = client.get(&models_url);
    let resolved_key = api_key
        .map(str::to_owned)
        .or_else(|| std::env::var("OPENKIND_API_KEY").ok())
        .or_else(|| std::env::var("TYPESAFE_API_KEY").ok())
        .filter(|value| !value.is_empty());
    if let Some(key) = resolved_key {
        request = request.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("GET {models_url}"))?;
    let status = response.status();
    let request_id = response
        .headers()
        .get("x-typesafe-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = response.text().await.context("read model-list response")?;
    if !status.is_success() {
        let request_id = request_id
            .map(|id| format!(" (request id {})", output::terminal_safe(&id)))
            .unwrap_or_default();
        anyhow::bail!(
            "GET {models_url} returned HTTP {status}{request_id}: {}",
            output::terminal_safe(&body)
        );
    }
    serde_json::from_str(&body).context("parse model-list response")
}

fn print_status(server: &str, models: ModelsResponse) {
    output::print_heading("Daemon status");
    output::print_key_value("Server", server);
    output::print_key_value("Health", "healthy");
    let rows: Vec<Vec<String>> = models
        .models
        .into_iter()
        .map(|model| vec![model.name, model.release_date, model.description])
        .collect();
    if rows.is_empty() {
        println!("No model aliases are registered with this daemon.");
    } else {
        output::print_table(
            "Registered model aliases",
            &["ALIAS", "RELEASE DATE", "DESCRIPTION"],
            &rows,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{Model, ModelInfo, RefreshStatusMsg, StatusWatch};
    use bubbletea_rs::Msg;

    #[test]
    fn watch_view_lists_registered_aliases() {
        let (mut model, _) = StatusWatch::init();
        let message = RefreshStatusMsg(Ok(vec![ModelInfo {
            name: "mock".to_owned(),
            description: "Mock model".to_owned(),
            release_date: "2026-09-27".to_owned(),
        }]));
        model.update(Box::new(message) as Msg);

        let view = model.view();
        assert!(view.contains("Health:"));
        assert!(view.contains("healthy"));
        assert!(view.contains("Registered model aliases"));
        assert!(view.contains("mock"));
        assert!(view.contains("2026-09-27"));
    }

    #[test]
    fn watch_view_reports_failed_checks_without_claiming_daemon_health() {
        let (mut model, _) = StatusWatch::init();
        model.update(Box::new(RefreshStatusMsg(Err("connection refused".to_owned()))) as Msg);

        let view = model.view();
        assert!(view.contains("Daemon status:"));
        assert!(view.contains("check failed"));
        assert!(view.contains("connection refused"));
        assert!(!view.contains("Health: unavailable"));
    }
}
