//! Terminal output formatting, color styling, and pull progress reporting.

use std::io::IsTerminal;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use lipgloss::{Color, Style};
use unicode_width::UnicodeWidthStr;

fn stdout_color_enabled() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

/// Escape control characters before including untrusted text in terminal output.
///
/// JSON output deliberately retains the original values. Human-readable output uses this
/// representation so C0/C1 controls cannot be interpreted as ANSI, OSC, or cursor commands.
pub fn terminal_safe(value: &str) -> String {
    value.chars().fold(String::new(), |mut safe, character| {
        if character.is_control() {
            safe.extend(character.escape_default());
        } else {
            safe.push(character);
        }
        safe
    })
}

/// Render text in bold if terminal styling is supported on stdout.
pub fn style_bold(value: &str) -> String {
    let value = terminal_safe(value);
    if stdout_color_enabled() {
        Style::new().bold(true).render(&value)
    } else {
        value
    }
}

/// Render a section heading in bold color if terminal styling is supported on stdout.
pub fn style_heading(value: &str) -> String {
    let value = terminal_safe(value);
    if stdout_color_enabled() {
        Style::new()
            .bold(true)
            .foreground(Color::from("6"))
            .render(&value)
    } else {
        value
    }
}

fn stderr_color_enabled() -> bool {
    std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

/// Render muted secondary text if terminal styling is supported on stdout.
pub fn style_muted(value: &str) -> String {
    let value = terminal_safe(value);
    if stdout_color_enabled() {
        Style::new().foreground(Color::from("8")).render(&value)
    } else {
        value
    }
}

/// Render success indicator text in bold green if terminal styling is supported on stdout.
pub fn style_success(value: &str) -> String {
    let value = terminal_safe(value);
    if stdout_color_enabled() {
        Style::new()
            .bold(true)
            .foreground(Color::from("10"))
            .render(&value)
    } else {
        value
    }
}

/// Render error text in bold red if terminal styling is supported on stdout.
pub fn style_error(value: &str) -> String {
    let value = terminal_safe(value);
    if stdout_color_enabled() {
        Style::new()
            .bold(true)
            .foreground(Color::from("9"))
            .render(&value)
    } else {
        value
    }
}

/// Print a styled heading followed by a newline.
pub fn print_heading(title: &str) {
    println!("{}", style_heading(title));
}

/// Print a tabular dataset with a section heading and aligned columns.
pub fn print_table(title: &str, headers: &[&str], rows: &[Vec<String>]) {
    print_heading(title);
    if rows.is_empty() {
        return;
    }

    println!("{}", render_table(headers, rows));
}

/// Render a table with header separator and column widths aligned to Unicode character boundaries.
pub fn render_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    if headers.is_empty() || rows.is_empty() {
        return String::new();
    }

    let widths: Vec<usize> = (0..headers.len())
        .map(|column| {
            rows.iter()
                .filter_map(|row| row.get(column))
                .map(|value| terminal_safe(value))
                .map(|value| UnicodeWidthStr::width(value.as_str()))
                .fold(UnicodeWidthStr::width(headers[column]), usize::max)
        })
        .collect();
    let header = render_row(
        &headers
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>(),
        &widths,
    );
    let mut lines = vec![style_heading(&header)];
    lines.push(
        widths
            .iter()
            .map(|width| "-".repeat(*width))
            .collect::<Vec<_>>()
            .join("  "),
    );
    for row in rows {
        lines.push(render_row(row, &widths));
    }
    lines.join("\n")
}

fn render_row(values: &[String], widths: &[usize]) -> String {
    let mut line = String::new();
    for (index, width) in widths.iter().enumerate() {
        if index > 0 {
            line.push_str("  ");
        }
        let value = terminal_safe(values.get(index).map(String::as_str).unwrap_or(""));
        line.push_str(&value);
        if index + 1 < widths.len() {
            line.push_str(
                &" ".repeat(width.saturating_sub(UnicodeWidthStr::width(value.as_str()))),
            );
        }
    }
    line
}

/// Print a key-value label pair with bold styling on the key.
pub fn print_key_value(label: &str, value: &str) {
    println!("{}: {}", style_bold(label), terminal_safe(value));
}

/// Format a byte count into a human-readable string with binary prefixes (e.g. KiB, MiB, GiB).
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

struct ArtifactProgress {
    path: String,
    total: u64,
    initial: u64,
    last_done: u64,
    downloaded: u64,
    started: Instant,
    already_present: bool,
    bar: Option<ProgressBar>,
}

/// Progress monitor for model pulls, displaying dynamic progress bars in terminals or line logs in pipes.
pub struct PullProgress {
    terminal: bool,
    current: Option<ArtifactProgress>,
    downloaded: u64,
    callback_count: usize,
}

impl PullProgress {
    /// Create a new progress tracker detecting whether stderr is an interactive terminal.
    pub fn new() -> Self {
        Self {
            terminal: std::io::stderr().is_terminal(),
            current: None,
            downloaded: 0,
            callback_count: 0,
        }
    }

    /// Update progress for a specific artifact, advancing bytes transferred or verified.
    pub fn update(&mut self, path: &str, done: u64, total: u64) {
        if self
            .current
            .as_ref()
            .is_some_and(|current| current.path != path)
        {
            self.finish_artifact(true);
        }
        if self.current.is_none() {
            self.start_artifact(path, done, total);
        }

        let current = self.current.as_mut().expect("artifact progress started");
        if done < current.last_done {
            if !self.terminal {
                eprintln!("Partial data for {path} was discarded; restarting from byte zero");
            }
            current.initial = 0;
            current.last_done = 0;
            current.downloaded = 0;
            current.started = Instant::now();
            current.already_present = false;
            if let Some(bar) = &current.bar {
                bar.set_prefix(path.to_owned());
            }
        }
        let delta = done.saturating_sub(current.last_done);
        current.downloaded = current.downloaded.saturating_add(delta);
        self.downloaded = self.downloaded.saturating_add(delta);
        current.last_done = done;
        self.callback_count += 1;

        let message = if current.already_present {
            "checking SHA-256, bytes already present".to_owned()
        } else if done >= total {
            "verifying SHA-256".to_owned()
        } else {
            let elapsed = current.started.elapsed().as_secs_f64().max(0.001);
            let rate = (current.downloaded as f64 / elapsed) as u64;
            let resumed = if current.initial > 0 {
                format!("; resumed {}", format_bytes(current.initial))
            } else {
                String::new()
            };
            format!("{} per second{resumed}", format_bytes(rate))
        };
        if let Some(bar) = &current.bar {
            bar.set_position(done.min(total));
            bar.set_message(message);
        } else if !self.terminal && done >= total && !current.already_present {
            eprintln!("Verifying {path} (SHA-256)");
        }
    }

    fn start_artifact(&mut self, path: &str, done: u64, total: u64) {
        let already_present = done >= total;
        let bar = if self.terminal {
            let draw_target = ProgressDrawTarget::stderr_with_hz(10);
            let bar = ProgressBar::with_draw_target(Some(total), draw_target);
            let template = if stderr_color_enabled() {
                "{prefix:.bold} [{bar:32.cyan/blue}] {bytes}/{total_bytes} {msg}"
            } else {
                "{prefix} [{bar:32}] {bytes}/{total_bytes} {msg}"
            };
            let style = ProgressStyle::with_template(template)
                .expect("valid model download progress template")
                .progress_chars("=>-");
            bar.set_style(style);
            let prefix = if done > 0 && !already_present {
                format!(
                    "{path} (resume {} of {})",
                    format_bytes(done),
                    format_bytes(total)
                )
            } else {
                path.to_owned()
            };
            bar.set_prefix(prefix);
            bar.set_position(done.min(total));
            Some(bar)
        } else {
            if already_present {
                eprintln!("Verifying existing artifact {path} (SHA-256)");
            } else if done > 0 {
                eprintln!(
                    "Downloading {path}, resuming at {} of {}",
                    format_bytes(done),
                    format_bytes(total)
                );
            } else {
                eprintln!("Downloading {path} ({})", format_bytes(total));
            }
            None
        };
        self.current = Some(ArtifactProgress {
            path: path.to_owned(),
            total,
            initial: done,
            last_done: done,
            downloaded: 0,
            started: Instant::now(),
            already_present,
            bar,
        });
    }

    fn finish_artifact(&mut self, success: bool) {
        let Some(current) = self.current.take() else {
            return;
        };
        if let Some(bar) = current.bar {
            if success {
                bar.finish_with_message(if current.already_present {
                    "verified, reused"
                } else {
                    "verified"
                });
            } else {
                bar.abandon_with_message("failed");
            }
        } else if success {
            if current.already_present {
                eprintln!("Verified {} (reused)", current.path);
            } else {
                eprintln!("Verified {}", current.path);
            }
        } else if !success {
            eprintln!("Failed {}", current.path);
        }
    }

    /// Return the cumulative number of newly downloaded bytes across all artifacts.
    pub fn downloaded_bytes(&self) -> u64 {
        self.downloaded
    }

    /// Return whether any progress callback has fired during the pull operation.
    pub fn has_progress(&self) -> bool {
        self.callback_count > 0
    }

    /// Complete the current active artifact progress tracking with success status.
    pub fn finish_success(&mut self) {
        self.finish_artifact(true);
    }

    /// Complete the current active artifact progress tracking with failure status.
    pub fn finish_failure(&mut self, artifact_error: bool) {
        let failed_artifact = self
            .current
            .as_ref()
            .is_some_and(|current| current.last_done < current.total || artifact_error);
        self.finish_artifact(!failed_artifact);
    }
}

#[cfg(test)]
mod tests {
    use super::{render_table, terminal_safe};
    use unicode_width::UnicodeWidthStr;

    #[test]
    fn table_columns_align_wide_unicode_values() {
        let rows = vec![
            vec!["模型".to_owned(), "2026-09-27".to_owned()],
            vec!["x".to_owned(), "2026-09-27".to_owned()],
        ];
        let rendered = render_table(&["ALIAS", "RELEASE DATE"], &rows);
        let lines: Vec<&str> = rendered.lines().collect();
        let first_value_start = lines[2].find("2026-09-27").unwrap();
        let second_value_start = lines[3].find("2026-09-27").unwrap();

        assert_eq!(
            UnicodeWidthStr::width(&lines[2][..first_value_start]),
            UnicodeWidthStr::width(&lines[3][..second_value_start])
        );
    }

    #[test]
    fn terminal_text_escapes_control_sequences() {
        let malicious = "model\u{1b}]52;c;T1BFTktJTkQ=\u{7}\nnext\u{85}line";
        let safe = terminal_safe(malicious);

        assert_eq!(
            safe,
            "model\\u{1b}]52;c;T1BFTktJTkQ=\\u{7}\\nnext\\u{85}line"
        );
        assert!(!safe.chars().any(char::is_control));
    }

    #[test]
    fn table_escapes_controls_before_measuring_columns() {
        let rows = vec![
            vec!["bad\u{1b}[2J".to_owned(), "first".to_owned()],
            vec!["ok".to_owned(), "second".to_owned()],
        ];

        let rendered = render_table(&["ALIAS", "VALUE"], &rows);

        assert!(rendered.contains("bad\\u{1b}[2J"));
        let lines: Vec<_> = rendered.lines().collect();
        assert!(!lines[2..].iter().any(|line| line.contains('\u{1b}')));
        assert_eq!(lines[2].find("first"), lines[3].find("second"));
    }
}
