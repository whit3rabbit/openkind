//! The per-task lifecycle loop: route → record → train → calibrate → shadow
//! → promote → monitor.
//!
//! One [`TaskEngine`] owns one task's store, production student, shadow
//! candidate, audit bookkeeping, teacher lineage, and drift state. The
//! engine never performs I/O on the upstream API and never embeds text —
//! the caller supplies embeddings and teacher answers; the engine decides
//! what may be answered locally and when to train.
//!
//! The guarantee: the probability that the student answers and disagrees
//! with the teacher stays at or below the task's budget at confidence
//! `1 - delta`, maintained by (a) calibration rows drawn only from IID
//! channels, (b) a fixed threshold grid tested strictest-first with a
//! Clopper–Pearson bound over all calibration rows, (c) shadow judgement at
//! the full budget pooled with the candidate's own calibration counts, and
//! (d) an always-on audit slice scored as served.

mod fit;
mod init;
mod lifecycle;
mod observe;
mod readiness;
mod route;
mod types;

pub use types::{
    FitInput, FitOutput, LocalPrediction, RouteDecision, TaskEngine, TaskStatus, TeacherAnswer,
    TickOutcome,
};
