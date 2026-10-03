"""Evaluation-only TypeSafe snapshots. No training, calibration or workflow execution."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import time

# The bundle loader supplies the already verified module when executing saved source.
if "recipe" not in globals():
    import train as recipe

TYPESAFE_SOURCES = {
    "onet": dict(repo="typesafe/evalsafe-onet", revision="bda14bdd85be4d93140a842332359f526543b314", license="Apache-2.0"),
    "invoice_processing": dict(repo="typesafe/evalsafe-invoice-processing", revision="6beeb2d2acd65c086c835022f5f4d7434114cafc", license="Apache-2.0"),
    "customer_service": dict(repo="typesafe/evalsafe-customer-service", revision="b1342f5a704587dbc465867c38c2694348ff86e4", license="unspecified in the dataset card"),
    "security_incidents": dict(repo="typesafe/evalsafe-security-incidents", revision="fbe1ea5c69cf494157fd23f2002a0d9d9a418443", license="unspecified in the dataset card"),
    "agent_trace_observability": dict(repo="typesafe/evalsafe-agent-trace-observability", revision="8635540973910a92465fe2bc53e195375aa6e1a8", license="unspecified in the dataset card"),
}
QUESTION_FILE = "data/questions.parquet"


def load_source(spec):
    """Read only pinned published test references, never candidate run_results."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    paths = {name: hf_hub_download(spec["repo"], filename=name, revision=spec["revision"], repo_type="dataset")
             for name in (QUESTION_FILE, "dataset.json", "README.md")}
    return pq.read_table(paths[QUESTION_FILE]).to_pylist(), {
        **spec, "upstream_split": "test", "file": QUESTION_FILE,
        "hashes": {name: recipe.file_digest(path) for name, path in paths.items()},
    }


def convert(source, raw):
    """Keep descriptions and soft reference mass; modal ties need no arbitrary gold label."""
    question = json.loads(raw["question_json"])
    kind = raw["kind"]
    assert kind in {"noul", "choice", "score"}, "Unsupported question kind"
    assert question.get("type", kind) == kind, "Question kind disagrees with schema"
    instruction = question.get("instructions", question.get("instruction"))
    assert isinstance(instruction, str) and instruction.strip(), "Missing instructions"
    criteria = question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, dict):
            options = [(str(k), v) for k, v in criteria.items()]
        else:
            options = [(c["name"], c["description"]) for c in question["choices"]]
    elif kind == "score":
        if isinstance(criteria, list):
            options = [(str(i), v) for i, v in enumerate(criteria)]
        else:
            levels = question["levels"]
            assert [x["value"] for x in levels] == list(range(len(levels))), "Noncontiguous ordinal levels"
            options = [(str(x["value"]), x["description"]) for x in levels]
        assert 2 <= len(options) <= 10, "Score exceeds the finite-level inference contract"
    else:
        if isinstance(criteria, dict):
            assert set(criteria) == {"false", "true"}, "Invalid binary criteria"
            options = [(k, criteria[k]) for k in ("false", "true")]
        elif question.get("levels"):
            levels = question["levels"]
            assert [x["value"] for x in levels] == [0, 1], "Invalid binary levels"
            options = [(str(bool(x["value"])).lower(), x["description"]) for x in levels]
        else:
            options = [("false", "The statement is false."), ("true", "The statement is true.")]
    assert all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in options), "Undescribed option"
    keys = [k for k, _ in options]
    assert len(set(keys)) == len(keys), "Duplicate options"
    label = raw["consensus"]
    assert label["status"] in {"ok", "answered"}, "Reference not answered"
    entries = label["probabilities"]
    probs = {p["option"]: float(p["probability"]) for p in entries}
    assert len(probs) == len(entries) and set(probs) == set(keys), "Reference option space differs"
    assert all(math.isfinite(p) and p >= 0 for p in probs.values()), "Invalid reference probability"
    mass = sum(probs.values())
    assert abs(mass - 1) <= 0.001, "Reference mass is not normalized"
    target = [probs[k] / mass for k in keys]
    added_none = kind == "choice" and "__none__" not in keys
    if added_none:
        # OpenKind requires this extra outcome. Retain its mass in scoring rather than conditioning it away.
        options.append(("__none__", "None of the supplied options is a valid answer under the stated criteria."))
        target.append(0.0)
    assert 2 <= len(options) <= len(recipe.CODES), "Option count exceeds answer-code contract"
    state = json.loads(raw["state_json"])
    state = state if isinstance(state, str) else recipe.canonical(state)
    row = recipe.record(TYPESAFE_SOURCES[source]["repo"], raw["question_instance_id"], state,
                        instruction, options, options[max(range(len(target)), key=target.__getitem__)][0],
                        kind, target=target)
    row.update(benchmark_only=True, supervision="synthetic_model_consensus", case_id=raw["case_id"],
               reference_mass=mass, semantic_none_added=added_none)
    return row


def prepare_source(source, raw_rows, tokenizer, seed, max_length, cases_per_source, excluded_groups=()):
    """Sample complete cases before admission so short prompts cannot select an easier denominator."""
    assert cases_per_source is None or (type(cases_per_source) is int and cases_per_source > 0)
    cases = sorted({r["case_id"] for r in raw_rows}, key=lambda c: recipe.digest([seed, source, c]))
    chosen = set(cases if cases_per_source is None else cases[:cases_per_source])
    ledger = []
    for raw in raw_rows:
        if raw["case_id"] not in chosen:
            continue
        entry = dict(id=raw["question_instance_id"], case_id=raw["case_id"], kind=raw["kind"])
        try:
            row = convert(source, raw)
            if row["group"] in excluded_groups:
                entry["failure"] = "overlap_with_prepared_mixture"
            else:
                encoded = recipe.encode(row, tokenizer, seed)
                entry["prompt_tokens"] = len(encoded["input_ids"])
                entry["semantic_none_added"] = row["semantic_none_added"]
                if len(encoded["input_ids"]) > max_length:
                    entry["failure"] = "overlength"
                else:
                    entry["row"] = encoded
        except (AssertionError, KeyError, ValueError, TypeError, IndexError) as exc:
            entry["failure"] = "unsupported_reference: " + str(exc)
        ledger.append(entry)
    assert ledger and len({x["id"] for x in ledger}) == len(ledger), "Empty or duplicate question instances"
    return ledger, dict(raw_questions=len(raw_rows), raw_cases=len(cases), selected_cases=sorted(chosen),
                        selected_questions=len(ledger), full_snapshot=len(chosen) == len(cases))


def agreement(kind, keys, target, prediction):
    assert len(keys) == len(target) == len(prediction)
    assert all(math.isfinite(p) and p >= 0 for p in prediction)
    assert abs(sum(prediction) - 1) < 1e-5
    if kind == "score":
        values = [float(k) for k in keys]
        width = max(values) - min(values)
        assert width > 0
        return max(0.0, 1 - abs(sum(v * (p - t) for v, p, t in zip(values, prediction, target))) / width)
    # The published metric is divergence, not its square root (Jensen-Shannon distance).
    middle = [(t + p) / 2 for t, p in zip(target, prediction)]
    divergence = 0.5 * sum(t * math.log2(t / m) if t else 0 for t, m in zip(target, middle))
    divergence += 0.5 * sum(p * math.log2(p / m) if p else 0 for p, m in zip(prediction, middle))
    return max(0.0, min(1.0, 1 - divergence))


def evaluate(model, ledger, temperature):
    import torch
    model.eval()
    predictions = []
    with torch.no_grad():
        for entry in ledger:
            output = {k: v for k, v in entry.items() if k != "row"}
            row = entry.get("row")
            if row is None:
                output["agreement"] = 0.0
            else:
                try:
                    p = recipe.probabilities(recipe.logits(model, row).cpu().tolist(), temperature)
                    output.update(keys=row["keys"], reference=row["target"], probabilities=p,
                                  agreement=agreement(row["kind"], row["keys"], row["target"], p))
                except (RuntimeError, AssertionError, ValueError) as exc:
                    # Count a failed prediction as zero. Stop on OOM instead of repeatedly exhausting the device.
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                    output.update(failure="inference_failed: " + str(exc), agreement=0.0)
            predictions.append(output)
    by_kind = {}
    for kind in sorted({x["kind"] for x in predictions}):
        rows = [x for x in predictions if x["kind"] == kind]
        by_kind[kind] = dict(n=len(rows), answered=sum("failure" not in x for x in rows),
                             agreement=sum(x["agreement"] for x in rows) / len(rows))
    return dict(by_kind=by_kind,
                equal_type_agreement=sum(x["agreement"] for x in by_kind.values()) / len(by_kind),
                failures=dict(collections.Counter(x["failure"] for x in predictions if "failure" in x)),
                predictions=predictions)


def run_benchmark(model, tokenizer, config, run, identity, excluded_groups=(), cases_per_source=10, max_length=None,
                  *, tokenizer_loader=None, source_loader=None):
    """Dispatch into the verified export; optional loaders are for marked offline fixtures."""
    bundle = recipe.load_frozen_bundle(model, config, run, identity,
                                      tokenizer_loader=tokenizer_loader, benchmark_path=__file__)
    assert source_loader is None or bundle.contract["offline_fixture"], "Source overrides are restricted to offline test bundles"
    return bundle.benchmark._run_frozen_benchmark(bundle, model, run, identity, excluded_groups,
                                                 cases_per_source, max_length, source_loader)


def _run_frozen_benchmark(bundle, model, run, identity, excluded_groups, cases_per_source, max_length, source_loader):
    run = Path(run)
    export, contract, config = bundle.export, bundle.contract, bundle.config
    tokenizer = bundle.tokenizer
    seed = recipe.experiment_seed(config, "sampling")
    max_length = contract["max_length"] if max_length is None else max_length
    assert type(max_length) is int and 0 < max_length <= 16384, "Benchmark length must be within the bounded 16K panel"
    excluded_groups = set(excluded_groups)
    plan = dict(schema="typesafe-frozen-questions-v1", identity=identity, export_sha256=recipe.file_digest(export / "EXPORT.json"),
                sources=TYPESAFE_SOURCES, seed=seed, cases_per_source=cases_per_source,
                max_length=max_length, export_max_length=contract["max_length"],
                extended_context_unqualified=max_length > contract["max_length"], temperature=contract["temperature"],
                excluded_groups_sha256=recipe.digest(sorted(excluded_groups)),
                used_for_training_or_selection=False, workflow_execution=False,
                rule="frozen descriptive benchmark; any subsequent tuning spends these references")
    destination = run / ("typesafe_benchmark_" + recipe.digest(plan)[:16])
    recipe.immutable_json(destination / "BENCHMARK_OPENED.json", plan)
    result_path = destination / "BENCHMARK_RESULT.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    prepared, inventory = {}, {}
    for source, spec in TYPESAFE_SOURCES.items():
        raw, inventory[source] = (load_source if source_loader is None else source_loader)(spec)
        prepared[source], sampling = prepare_source(source, raw, tokenizer, seed,
                                                    max_length, cases_per_source, excluded_groups)
        inventory[source]["sampling"] = sampling
    result = dict(plan=plan, sources=inventory, candidate={}, parent={}, offline_fixture=contract["offline_fixture"],
                  reference="synthetic Astra/Fable consensus, not human ground truth",
                  comparison="O*NET metric definitions; workflow question diagnostics, not exact-action scores",
                  direct_leaderboard_comparison=False,
                  schema_adaptation="append described __none__ to Choice if absent; reference mass zero; retain predicted none mass",
                  timings="serial local evaluation wall time including Python overhead, not service or full-workflow latency",
                  model_promoted=False)
    try:
        for label, parent, temperature in (("candidate", False, contract["temperature"]), ("parent", True, 1.0)):
            recipe.load_bundle_adapter(bundle, model, parent=parent)
            for source, ledger in prepared.items():
                started = time.perf_counter()
                report = evaluate(model, ledger, temperature)
                result[label][source] = {k: v for k, v in report.items() if k != "predictions"}
                result[label][source]["wall_seconds"] = time.perf_counter() - started
                recipe.immutable_json(destination / f"{label}_{source}_predictions.json", report["predictions"])
    finally:
        recipe.load_bundle_adapter(bundle, model)
    recipe.immutable_json(result_path, result)
    return result
