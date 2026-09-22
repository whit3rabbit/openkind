//! Decision engine trait definition and token estimation helpers.

use async_trait::async_trait;
use openkind_core::{ModelInfo, SystemRequest, SystemResponse};

use crate::error::EngineResult;

/// The trait every backend implements. The engine is **runtime-agnostic**:
/// it doesn't know it's being served over HTTP or gRPC — that's deliberate.
/// Callers pass `SystemRequest` in and get `SystemResponse` back.
#[async_trait]
pub trait DecisionEngine: Send + Sync {
    /// Identifier of this backend, e.g. `"mock"`, `"qwen-3b-candle"`.
    fn backend_id(&self) -> &str;

    /// Public metadata shown by `GET /v1/models`. The default is a
    /// minimal mock entry; real backends should override with a real
    /// description and release date.
    fn model_metadata(&self) -> ModelInfo {
        ModelInfo {
            name: self.backend_id().to_string(),
            description: format!("{} backend", self.backend_id()),
            release_date: "1970-01-01".to_string(),
        }
    }

    /// Evaluate a single request. Returns one answer per question id.
    async fn evaluate(&self, req: SystemRequest) -> EngineResult<SystemResponse>;

    /// Estimate input tokens (cheap, before evaluation). The mock uses
    /// a rough `chars / 4` heuristic; real backends will use a tokenizer.
    fn estimate_input_tokens(&self, req: &SystemRequest) -> u32 {
        let state_chars = match &req.state {
            openkind_core::State::Text(s) => s.len(),
            openkind_core::State::Object(m) => count_json_bytes(m),
            openkind_core::State::Array(a) => count_json_bytes(a),
        };
        let instr_chars: usize = req
            .questions
            .values()
            .map(|q| match q {
                openkind_core::Question::Noul(n) => count_json_bytes(&n.instructions),
                openkind_core::Question::Choice(c) => count_json_bytes(&c.instructions),
                openkind_core::Question::Score(s) => count_json_bytes(&s.instructions),
            })
            .sum();
        let total_chars = state_chars.saturating_add(instr_chars);
        if total_chars == 0 {
            0
        } else {
            u32::try_from(total_chars.div_ceil(4)).unwrap_or(u32::MAX)
        }
    }
}

/// Zero-allocation byte counter implementing std::io::Write.
struct ByteCounter(usize);

impl std::io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn count_json_bytes<T: serde::Serialize + ?Sized>(val: &T) -> usize {
    let mut counter = ByteCounter(0);
    serde_json::to_writer(&mut counter, val)
        .map(|()| counter.0)
        .unwrap_or(0)
}
