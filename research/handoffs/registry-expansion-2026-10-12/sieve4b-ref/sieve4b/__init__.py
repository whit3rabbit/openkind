"""Sieve-4B: a typed-decision model. A state and typed questions go in; a calibrated probability for every option
of every question comes out. It never generates text.

    from sieve4b import load, decide
    engine = load("sthanika-ai/Sieve-4B")    # a Hub id or a local copy; fetches Qwen/Qwen3.5-4B on first use
    decide(engine, state, questions)              # the /v1/systemone response body: {"answers": {...}, "usage": {...}}
"""
from pathlib import Path


def load(repo_or_path: str = "sthanika-ai/Sieve-4B", device: str = "cuda:0", name: str = "sieve-4b",
         prefix_cache_gb: float = 1.0, **engine_kw):
    from sieve4b.engine import Engine
    from sieve4b.model import DecisionModel, PrefixCache
    path = Path(repo_or_path)
    if not path.exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(repo_or_path)))
    model = DecisionModel.load(path, device=device).eval()
    model._prefix_cache = PrefixCache(max_bytes=int(prefix_cache_gb * 2**30))
    return Engine(model, model_name=name, **engine_kw)


def decide(engine, state, questions: dict, model: str = "sieve-4b") -> dict:
    from sieve4b.contract import SystemOneRequest
    body, _ = engine.evaluate(SystemOneRequest(state=state, model=model, questions=questions))
    return body
