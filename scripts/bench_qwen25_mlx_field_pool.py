#!/usr/bin/env python3
"""Offline MLX reference timing for the Qwen2.5 field-batching pattern.

This measures the external Python implementation's forward shape, not an
OpenKind engine or a Jev request. It never downloads model files.
"""

import argparse
import copy
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--fields", type=int, required=True)
    parser.add_argument("--mode", choices=("pooled", "per_field"), required=True)
    parser.add_argument("--reps", type=int, default=5)
    return parser.parse_args()


def clone_cache(cache):
    return [copy.copy(layer) for layer in cache]


def forward(model, tokenizer, schema, context, mode):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    meta = schema.compile_parallel_metadata(tokenizer)
    items = meta["field_items"]
    suffix_lengths = meta["suffix_lengths"]
    suffixes = meta["suffixes_batch"]
    schema_str = schema.to_parallel_schema_str()
    prompt = (
        f"<|im_start|>system\nClassify JSON attributes:\n{schema_str}<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\n"
    )
    prompt_tokens = tokenizer.encode(prompt)
    started = time.perf_counter()
    cache = make_prompt_cache(model)
    prefix_out = model(mx.array(prompt_tokens)[None], cache=cache)
    mx.eval(prefix_out)
    prefill_ms = (time.perf_counter() - started) * 1000

    branch_started = time.perf_counter()
    if mode == "pooled":
        lanes = clone_cache(cache)
        width = len(items)
        for layer in lanes:
            if getattr(layer, "keys", None) is not None:
                layer.keys = mx.repeat(layer.keys, width, axis=0)
                layer.values = mx.repeat(layer.values, width, axis=0)
        arrays = [array for layer in lanes for array in
                  (getattr(layer, "keys", None), getattr(layer, "values", None))
                  if array is not None]
        if arrays:
            mx.eval(*arrays)
        branch_ms = (time.perf_counter() - branch_started) * 1000
        suffix_started = time.perf_counter()
        outputs = model(suffixes, cache=lanes)
        mx.eval(outputs)
        suffix_ms = (time.perf_counter() - suffix_started) * 1000
        logits = [outputs[index, length - 1, :]
                  for index, length in enumerate(suffix_lengths)]
        forward_calls = 2
        padded_slots = sum(suffixes.shape[1] - length for length in suffix_lengths)
    else:
        branch_ms = 0.0
        suffix_ms = 0.0
        logits = []
        for index, length in enumerate(suffix_lengths):
            lane_started = time.perf_counter()
            lane_cache = clone_cache(cache)
            branch_ms += (time.perf_counter() - lane_started) * 1000
            suffix_started = time.perf_counter()
            output = model(suffixes[index:index + 1, :length], cache=lane_cache)
            mx.eval(output)
            suffix_ms += (time.perf_counter() - suffix_started) * 1000
            logits.append(output[0, -1, :])
        forward_calls = 1 + len(items)
        padded_slots = 0

    readout_started = time.perf_counter()
    scores = [[float(logits[index][token]) for token in tokens]
              for index, tokens in enumerate(meta["cands_per_field"])]
    winners = [max(range(len(row)), key=row.__getitem__) for row in scores]
    readout_ms = (time.perf_counter() - readout_started) * 1000
    return {
        "total_ms": (time.perf_counter() - started) * 1000,
        "prefill_ms": prefill_ms,
        "branch_ms": branch_ms,
        "suffix_ms": suffix_ms,
        "readout_ms": readout_ms,
        "forward_calls": forward_calls,
        "padded_slots": padded_slots,
        "suffix_lengths": suffix_lengths,
        "collisions": sum(meta["has_collisions"]),
        "winners": winners,
        "scores": scores,
    }


def main():
    args = parse_args()
    if args.reps < 1 or args.fields < 1:
        raise SystemExit("reps and fields must be positive")
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        if not (args.checkpoint / name).is_file():
            raise SystemExit(f"missing local checkpoint file: {name}")
    if not (args.source_root / "core" / "schema.py").is_file():
        raise SystemExit("missing local upstream schema.py")
    sys.path.insert(0, str(args.source_root))
    from core.schema import StructuredSchema
    from mlx_lm import load

    preset = json.loads(args.preset.read_text())
    fields = list(preset["schema"].items())[:args.fields]
    if len(fields) != args.fields:
        raise SystemExit("preset has fewer fields than requested")
    schema = StructuredSchema(dict(fields))
    load_started = time.perf_counter()
    model, tokenizer = load(str(args.checkpoint))
    load_ms = (time.perf_counter() - load_started) * 1000
    forward(model, tokenizer, schema, preset["context"], args.mode)
    samples = [forward(model, tokenizer, schema, preset["context"], args.mode)
               for _ in range(args.reps)]
    print(json.dumps({
        "kind": "external-python-mlx-field-reference/v1",
        "checkpoint_dir": args.checkpoint.name,
        "source_dir": args.source_root.name,
        "preset": args.preset.name,
        "host": platform.platform(),
        "fields": args.fields,
        "mode": args.mode,
        "warmups": 1,
        "reps": args.reps,
        "load_ms": load_ms,
        "median_total_ms": statistics.median(s["total_ms"] for s in samples),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "samples": samples,
    }))


if __name__ == "__main__":
    main()
