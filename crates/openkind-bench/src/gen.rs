//! Seeded deterministic workload generation (`gen-workload`).
//!
//! Produces an owned shape-matched binary-criterion grid: `--states` support
//! tickets, each scored against `--criteria` fixed binary rubric questions,
//! mirroring the state × criterion decision shape used by prior-art systems
//! benchmarks (see `docs/BENCHMARKS.md`). Identical `--seed` inputs produce
//! byte-identical output, so the run summary's fixture digest pins the
//! workload without vendoring large files.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::workload::{QuestionSpec, WorkloadRow};

/// The fixed binary rubric. Kept deliberately owned and domain-neutral: each
/// entry becomes one yes/no decision per state.
const CRITERIA: [&str; 21] = [
    "the customer reports a billing or charge dispute",
    "the customer requests a refund or credit",
    "the customer threatens to cancel or churn",
    "the customer reports data loss or corruption",
    "the customer cannot sign in or is locked out",
    "the issue blocks a production or live workflow",
    "the customer reports a security or privacy concern",
    "the customer asks about an integration with another system",
    "the customer reports degraded performance or latency",
    "the customer already attempted the documented workaround",
    "the transcript contains an explicit escalation request",
    "the issue involves a third-party provider outage",
    "the customer reports the mobile client specifically",
    "the customer provides reproduction steps",
    "the issue is a duplicate of an earlier ticket",
    "the customer reports incorrect or misleading output",
    "the transcript includes compliance or audit constraints",
    "the customer reports a billing-plan upgrade interest",
    "the issue can be resolved without engineering involvement",
    "the customer reports the problem as intermittent",
    "the ticket mentions contractual or SLA obligations",
];

const PRODUCTS: [&str; 6] = ["ledger", "sync", "insights", "gateway", "vault", "relay"];

const TIERS: [&str; 3] = ["free", "team", "enterprise"];

const SENTIMENTS: [&str; 4] = ["neutral", "frustrated", "escalated", "satisfied"];

const ACCOUNT_HEALTH: [&str; 3] = ["healthy", "at_risk", "delinquent"];

const OPENINGS: [&str; 8] = [
    "We started seeing this after the last release.",
    "This has been happening on and off for about a week.",
    "Since this morning nothing works for our team.",
    "A user reported something odd in the audit log.",
    "Our nightly job failed again and we are not sure why.",
    "We tried the steps in your docs but the error came back.",
    "Two of our agents hit the same wall yesterday.",
    "This started right after we rotated our API keys.",
];

const DETAILS: [&str; 10] = [
    "The dashboard shows the numbers but exports come back empty.",
    "Requests intermittently return a 500 with no error id attached.",
    "Webhook deliveries stop for a few minutes and then catch up in bulk.",
    "The mobile client logs the user out after every background sync.",
    "Search returns stale results until the workspace is reopened.",
    "The invoice PDF renders the wrong tax summary for our region.",
    "Latency on the report endpoint jumped from 200 ms to about 9 s.",
    "The bulk import rejects our CSV even though the preview accepts it.",
    "Permissions reset for two members after the SSO group sync.",
    "The usage counters on the billing page disagree with our export.",
];

const CLOSINGS: [&str; 8] = [
    "Please treat this as time-sensitive for us.",
    "Happy to jump on a call if that speeds things up.",
    "We have a workaround but it is manual and error-prone.",
    "This is the third time we filed something like this.",
    "No rush, but we would like to understand the cause.",
    "Our compliance review depends on an answer this week.",
    "Attaching the trace id and the request payload.",
    "Other teams on our plan have not reported this.",
];

/// Result of [`generate_workload`].
pub struct GeneratedWorkload {
    /// The generated rows in emission order.
    pub rows: Vec<WorkloadRow>,
    /// SHA-256 of the emitted JSONL bytes.
    pub sha256: String,
}

/// Deterministic `states × criteria` JSONL workload.
///
/// # Errors
/// Returns an error for invalid or unrepresentable dimensions, failed
/// allocations, or an output path that cannot be written.
pub fn generate_workload(
    states: usize,
    criteria: usize,
    seed: u64,
    output: &Path,
) -> Result<GeneratedWorkload> {
    anyhow::ensure!(states >= 1, "states must be at least 1");
    anyhow::ensure!(
        criteria >= 1 && criteria <= CRITERIA.len(),
        "criteria must be between 1 and {}",
        CRITERIA.len()
    );
    let row_count = states
        .checked_mul(criteria)
        .context("states times criteria exceeds the workload size range")?;
    // Validate and reserve before generation or file writes, so oversized
    // operator input returns an error instead of wrapping or panicking.
    states
        .checked_add(9_999)
        .context("state count exceeds the generated ticket id range")?;
    let body_capacity = row_count
        .checked_mul(512)
        .context("generated JSONL capacity exceeds the workload size range")?;
    let mut rows = Vec::new();
    rows.try_reserve(row_count)
        .context("reserve workload rows")?;
    let mut state_values: Vec<(usize, openkind_core::State)> = Vec::new();
    state_values
        .try_reserve(states)
        .context("reserve workload states")?;
    let mut rng = StdRng::seed_from_u64(seed);
    for state_index in 0..states {
        let state_value = ticket_state(&mut rng, state_index);
        let state: openkind_core::State =
            serde_json::from_value(state_value).context("convert generated state")?;
        state_values.push((state_index, state));
    }
    for (state_index, state) in &state_values {
        for (criterion_index, criterion_text) in CRITERIA.iter().enumerate().take(criteria) {
            rows.push(WorkloadRow {
                id: format!("s{state_index:02}.c{criterion_index:02}"),
                state: state.clone(),
                question: QuestionSpec::Noul {
                    text: format!("Answer yes if {criterion_text}. Otherwise answer no."),
                    criteria: None,
                },
            });
        }
    }

    let mut body = String::new();
    body.try_reserve(body_capacity)
        .context("reserve generated JSONL")?;
    for row in &rows {
        let line = serde_json::to_string(row).context("serialize generated row")?;
        body.push_str(&line);
        body.push('\n');
    }
    let sha256 = {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(body.as_bytes());
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    };
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    fs::write(output, &body).with_context(|| format!("write {}", output.display()))?;
    Ok(GeneratedWorkload { rows, sha256 })
}

fn ticket_state(rng: &mut StdRng, index: usize) -> serde_json::Value {
    let transcript: Vec<String> = (0..3)
        .map(|_| {
            let opening = OPENINGS[rng.random_range(0..OPENINGS.len())];
            let detail = DETAILS[rng.random_range(0..DETAILS.len())];
            let closing = CLOSINGS[rng.random_range(0..CLOSINGS.len())];
            format!("{opening} {detail} {closing}")
        })
        .collect();
    serde_json::json!({
        "ticket_id": format!("TCK-{:05}", 10_000 + index),
        "product": PRODUCTS[rng.random_range(0..PRODUCTS.len())],
        "customer": {
            "tier": TIERS[rng.random_range(0..TIERS.len())],
            "account_health": ACCOUNT_HEALTH[rng.random_range(0..ACCOUNT_HEALTH.len())],
        },
        "sentiment": SENTIMENTS[rng.random_range(0..SENTIMENTS.len())],
        "transcript": transcript,
        "attachments": rng.random_range(0..4),
    })
}
