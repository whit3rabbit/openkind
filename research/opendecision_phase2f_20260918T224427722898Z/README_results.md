# OpenDecision Phase 2F

Run 20260918T224427722898Z; status completed

See paste_back_summary.json first. Raw worker rows retain paired scores, frozen policy outputs, actual cache bytes and timing samples.

TurboQuant is a pinned external GPL-3.0 dependency: standalone codecs adapted to HF prefix snapshots. Native vLLM integration and fused low-bit attention are NOT benchmarked.

Stored cache compression does not shrink model weights or imply smaller transient inference memory. Warm-cache traces include population misses. MTP is not applicable because no output tokens are generated.

Prior examples are reused for regression; these are not new independent generalization results.
