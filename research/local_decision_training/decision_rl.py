"""Optional one-step decision RL; an open experiment, not a reproduction of Clef RLCD."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import random
import time

import train as recipe

VERSION = "calibration-aware-decision-optimization-v1"
DEFAULT_RL_CONFIG = dict(method="reinforce", samples=8, seed=29, policy_weight=0.1,
                         ce_weight=1.0, brier_weight=0.1, kl_weight=0.05,
                         ordinal_credit=0.5, exact_record_weight=0.25)


def validate_config(config):
    assert set(config) == set(DEFAULT_RL_CONFIG), "Use a complete decision-RL configuration"
    assert config["method"] in ("reinforce", "expected_utility", "supervised")
    assert type(config["samples"]) is int and 2 <= config["samples"] <= 64
    assert type(config["seed"]) is int and 0 <= config["seed"] < 2**31
    for name in ("policy_weight", "ce_weight", "brier_weight", "kl_weight", "ordinal_credit", "exact_record_weight"):
        value = config[name]
        assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, name
    assert config["ordinal_credit"] <= 1
    assert config["ce_weight"] > 0, "Retain a proper-score anchor in every arm"
    assert config["kl_weight"] > 0, "Use a declared frozen-reference constraint"


def training_units(rows):
    """Complete requests receive record rewards; counterfactual pairs are not records."""
    recipe.reject_benchmark_data(rows=rows)
    grouped = collections.defaultdict(list)
    for row in rows:
        assert not row.get("diagnostic", False) and row.get("role", "train") == "train", "Only training rows may receive rewards"
        key = ((row["source"], "request", row["request_id"]) if row.get("request_size", 1) > 1 else
               (row["source"], "atomic", row["atomic_unit"]) if row.get("atomic_unit") else
               (row["source"], "row", row["id"]))
        grouped[key].append(row)
    units = []
    for key, fields in sorted(grouped.items()):
        is_record = key[1] == "request"
        size_key = "request_size" if is_record else "atomic_size"
        expected = fields[0].get(size_key, 1)
        assert len(fields) == expected and all(r.get(size_key, 1) == expected for r in fields), "Incomplete training request or atomic unit"
        assert len({r["id"] for r in fields}) == len(fields)
        if is_record:
            assert len({r["group"] for r in fields}) == 1 and len({r["state"] for r in fields}) == 1
            assert len({r["question_id"] for r in fields}) == len(fields), "Duplicate field in training request"
            assert all(max(r["target"]) == 1.0 for r in fields), "Exact-record rewards require exact field targets"
        units.append(dict(id=recipe.digest(key), source=key[0], fields=fields, is_record=is_record))
    assert units, "No admitted training units"
    return units


def action_utilities(row, config, *, device=None):
    """Credit belongs to semantic ordinal levels, never answer-code positions."""
    import torch
    target = torch.tensor(row["target"], dtype=torch.float32, device=device)
    utilities = torch.eye(len(target), device=device)
    if row["kind"] == "score" and config["ordinal_credit"]:
        levels = sorted((float(key), index) for index, key in enumerate(row["keys"]) if key != "__none__")
        assert all(math.isfinite(level) for level, _ in levels)
        assert len({level for level, _ in levels}) == len(levels), "Ordinal levels must be distinct"
        for (_, left), (_, right) in zip(levels, levels[1:]):
            utilities[left, right] = utilities[right, left] = config["ordinal_credit"]
    return utilities @ target


def objective(predictions, fields, reference_logits, config):
    """RLOO REINFORCE on discrete actions plus exact differentiable CE, Brier and KL.

    Probability-dependent proper scores use their direct gradients: placing the
    same Brier reward on every action would cancel under a group baseline.
    """
    import torch
    validate_config(config)
    assert len(predictions) == len(fields) == len(reference_logits) and predictions
    logs = [prediction.log_softmax(-1) for prediction in predictions]
    probabilities = [log.exp() for log in logs]
    targets = [torch.tensor(row["target"], device=prediction.device, dtype=torch.float32)
               for prediction, row in zip(predictions, fields)]
    references = [torch.tensor(values, device=prediction.device, dtype=torch.float32).log_softmax(-1)
                  for prediction, values in zip(predictions, reference_logits)]
    assert all(prediction.shape == target.shape == reference.shape for prediction, target, reference in zip(predictions, targets, references))
    ce = torch.stack([-(target * log).sum() for target, log in zip(targets, logs)]).mean()
    brier = torch.stack([(probability - target).square().sum() for probability, target in zip(probabilities, targets)]).mean()
    kl = torch.stack([(probability * (log - reference)).sum() for probability, log, reference in zip(probabilities, logs, references)]).mean()
    entropy = torch.stack([-(probability * log).sum() for probability, log in zip(probabilities, logs)]).mean()
    utilities = [action_utilities(row, config, device=prediction.device) for row, prediction in zip(fields, predictions)]
    is_record = len(fields) > 1 and any(row.get("request_size", 1) > 1 for row in fields)
    if is_record:
        assert len({r["request_id"] for r in fields}) == 1 and all(r["request_size"] == len(fields) for r in fields)
        assert all(float(target.max()) == 1.0 for target in targets)
    expected_reward = torch.stack([(p * u).sum() for p, u in zip(probabilities, utilities)]).mean()
    expected_record = torch.stack([(p * t).sum() for p, t in zip(probabilities, targets)]).prod() if is_record else ce.new_zeros(())
    policy_loss = ce.new_zeros(())
    sampled_reward = expected_reward.detach()
    sampled_record = expected_record.detach()
    if config["method"] == "reinforce":
        actions = [torch.multinomial(p.detach(), config["samples"], replacement=True) for p in probabilities]
        sampled_reward = torch.stack([utility[action] for utility, action in zip(utilities, actions)]).mean(0)
        sampled_record = torch.stack([target[action] for target, action in zip(targets, actions)]).prod(0) if is_record else torch.zeros_like(sampled_reward)
        # Each field has its own utility term; the record term differentiates the joint action.
        policy_terms = []
        for log, utility, action in zip(logs, utilities, actions):
            reward = utility[action]
            advantage = reward - (reward.sum() - reward) / (config["samples"] - 1)
            policy_terms.append(-(advantage.detach() * log[action]).mean())
        policy_loss = torch.stack(policy_terms).mean()
        if is_record:
            advantage = sampled_record - (sampled_record.sum() - sampled_record) / (config["samples"] - 1)
            joint_log = torch.stack([log[action] for log, action in zip(logs, actions)]).sum(0)
            policy_loss -= config["exact_record_weight"] * (advantage.detach() * joint_log).mean()
        sampled_reward, sampled_record = sampled_reward.mean(), sampled_record.mean()
    elif config["method"] == "expected_utility":
        # Finite actions permit an exact, lower-variance comparator without rollouts.
        policy_loss = -expected_reward - config["exact_record_weight"] * expected_record
    loss = config["ce_weight"] * ce + config["brier_weight"] * brier + config["kl_weight"] * kl + config["policy_weight"] * policy_loss
    assert torch.isfinite(loss), "Nonfinite decision-RL objective"
    metrics = dict(ce=ce, brier=brier, reference_kl=kl, entropy=entropy,
                   policy_loss=policy_loss, action_reward=sampled_reward, exact_record_reward=sampled_record,
                   expected_action_reward=expected_reward, expected_record_reward=expected_record)
    return loss, {key: value.detach() for key, value in metrics.items()}


def source_checkpoint(run, step=None):
    """Verify a trained warm start and bind its adapter bytes before any model load."""
    run = Path(run)
    saved = json.loads((run / "CONFIG.json").read_text())
    config = saved["config"]
    assert set(config) == set(recipe.DEFAULT_CONFIG), "Use a supervised v4 run"
    assert not (run / "POSTTRAINING_PLAN.json").exists(), "Chain from a supervised run, not another RL search"
    identity = saved["identity"]
    selection = json.loads((run / "SELECTION.json").read_text())
    assert selection["identity"] == identity
    step = selection["selected_step"] if step is None else step
    assert type(step) is int and step > 0, "No trained checkpoint selected. Choose an explicit experimental supervised step or obtain a retained supervised candidate."
    checkpoint = run / "checkpoints" / f"step_{step:06d}"
    commit = recipe.verify_checkpoint(checkpoint, identity)
    assert commit["step"] == step
    adapter = json.loads((checkpoint / "adapter/adapter_config.json").read_text())
    assert adapter["r"] == config["rank"] and adapter["lora_alpha"] == config["alpha"]
    assert adapter.get("base_model_name_or_path") == recipe.MODEL_ID, "Unexpected base model"
    assert adapter.get("revision") in (None, recipe.MODEL_REVISION), "Unexpected base revision"
    source = dict(identity=identity, supervised_step=step, adapter_sha256=commit["files"]["adapter/adapter_model.safetensors"],
                  commit_sha256=recipe.file_digest(checkpoint / "COMMIT.json"),
                  config_sha256=recipe.digest(config), explicit_checkpoint=step != selection["selected_step"],
                  prior_gate_opened=(run / "GATE_RESULT.json").exists(), prior_test_opened=(run / "TEST_OPENED.json").exists())
    from safetensors.torch import load_file
    source["adapter_state_sha256"] = _adapter_fingerprint(load_file(str(checkpoint / "adapter/adapter_model.safetensors")))
    if (run / "export").exists():
        metadata, contract, _ = recipe.verify_export(run / "export", identity)
        assert (contract["model_id"], contract["model_revision"]) == (recipe.MODEL_ID, recipe.MODEL_REVISION)
        source["export_manifest_sha256"] = recipe.file_digest(run / "export/EXPORT.json")
    return config, checkpoint, source


def _adapter_fingerprint(state):
    import hashlib
    import torch
    return recipe.digest({name: dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                         sha256=hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                         for name, tensor in sorted(state.items())})


def read_source_data(directory, source_run):
    directory, source_run = Path(directory), Path(source_run)
    manifest = json.loads((directory / "DATA_MANIFEST.json").read_text())
    assert manifest["schema"] == recipe.VERSION
    rows = {role: json.loads((directory / f"{role}.json").read_text()) for role in recipe.ROLES}
    for role, values in rows.items():
        assert recipe.digest(values) == manifest["hashes"][role] and len(values) == manifest["counts"][role], "Source data changed: " + role
    # Older runs bind data in the locked export; new runs also bind it before fitting.
    if (source_run / "export/DATA_MANIFEST.json").exists():
        assert manifest == json.loads((source_run / "export/DATA_MANIFEST.json").read_text()), "Data does not belong to the supervised run"
    else:
        saved = json.loads((source_run / "CONFIG.json").read_text())
        assert saved.get("data_manifest_sha256") == recipe.digest(manifest), "Use the original data directory and its run binding"
    return rows, manifest


def fresh_evidence_groups(rows):
    """Exclude every earlier acceptance group, without reassigning split roles."""
    return sorted({row["group"] for role in ("calibration", "gate", "test") for row in rows[role]})


def stage_schedule(data, config, rl_config, tokenizer=None):
    units = training_units(data["train"])
    if rl_config["exact_record_weight"] and rl_config["method"] != "supervised":
        assert any(unit["is_record"] for unit in units), "Exact-record rewards need multi-field training; enable the reasoning data intervention"
    # Tickets count source units, so larger records do not silently replace other sources.
    tickets = [{"id": unit["id"], "source": unit["source"]} for unit in units]
    order = recipe.build_training_schedule(tickets, config)
    schedule = []
    for occurrence, index in enumerate(order):
        unit = units[index]
        fields = [recipe.training_presentation(row, tokenizer, config, occurrence) for row in unit["fields"]]
        assert all(len(row["input_ids"]) <= config["max_length"] for row in fields), "Overlength post-training presentation"
        schedule.append(dict(id=unit["id"], source=unit["source"], fields=fields))
    return schedule


def _reference_cache(model, schedule, run, identity):
    path, lock = run / "REFERENCE_LOGITS.json", run / "REFERENCE_LOCK.json"
    unique = {}
    for unit in schedule:
        for row in unit["fields"]:
            key = recipe.digest([row["input_ids"], row["code_ids"], row["keys"]])
            unique[key] = row
    if lock.exists():
        saved = json.loads(lock.read_text())
        assert saved["identity"] == identity and saved["sha256"] == recipe.file_digest(path), "Frozen reference cache changed"
        values = json.loads(path.read_text())
        assert set(values) == set(unique), "Reference presentations changed"
        return values
    import torch
    from tqdm.auto import tqdm
    model.eval()
    with torch.no_grad():
        values = {key: recipe.logits(model, row).detach().cpu().tolist()
                  for key, row in tqdm(sorted(unique.items()), desc="Frozen supervised reference")}
    recipe.atomic_json(path, values)
    recipe.immutable_json(lock, dict(identity=identity, sha256=recipe.file_digest(path)))
    return values


def development_utility(predictions, rl_config):
    """Development diagnostics use the same decoded actions as the accuracy report."""
    import torch
    by_source, requests = collections.defaultdict(list), collections.defaultdict(list)
    for row in predictions:
        index = max(range(len(row["logits"])), key=lambda i: row["logits"][i])
        by_source[row["source"]].append(float(action_utilities(row, rl_config)[index]))
        if row.get("request_size", 1) > 1:
            requests[(row["source"], row["request_id"])].append((row, row["keys"][index] == row["answer_key"]))
    complete = [float(all(correct for _, correct in fields)) for fields in requests.values()
                if len(fields) == fields[0][0]["request_size"]]
    return dict(by_source={source: sum(values) / len(values) for source, values in sorted(by_source.items())},
                complete_records=len(complete), exact_record_accuracy=sum(complete) / len(complete) if complete else None)


def fit(model, data, config, rl_config, run, identity, source, *, tokenizer=None, on_event=None):
    """Fresh RL optimizer, immutable reference, complete-record updates, exact resume."""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from tqdm.auto import tqdm
    validate_config(rl_config)
    recipe.reject_benchmark_data(rows=data["development"])
    for name in ("max_steps", "accumulation", "evaluate_every", "checkpoint_every"):
        assert type(config[name]) is int and config[name] > 0, name
    assert math.isfinite(config["learning_rate"]) and config["learning_rate"] > 0
    assert source["supervised_step"] > 0
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    checkpoints = run / "checkpoints"
    plan = dict(schema=VERSION, identity=identity, config=config, rl_config=rl_config, source=source,
                trainer_sha256=recipe.file_digest(__file__), data_hashes={role: recipe.digest(data[role]) for role in ("train", "development")},
                reference="cached logits of the uncalibrated supervised checkpoint; no second backbone",
                proper_score_gradient="direct CE and Brier gradients; never detached into a shared action reward",
                selection="source-macro development NLL subject to retention against the supervised checkpoint")
    recipe.immutable_json(run / "POSTTRAINING_PLAN.json", plan)
    started = time.perf_counter()
    schedule = stage_schedule(data, config, rl_config, tokenizer)
    schedule_hash = recipe.digest(schedule)
    recipe.immutable_json(run / "TRAINING_SCHEDULE.json", dict(schema=VERSION, schedule_sha256=schedule_hash,
        source_unit_counts=dict(collections.Counter(unit["source"] for unit in schedule)),
        fields_presented=sum(len(unit["fields"]) for unit in schedule),
        presentations=[dict(unit=unit["id"], fields=[row["id"] for row in unit["fields"]]) for unit in schedule]))
    if (checkpoints / "step_000000/COMMIT.json").exists():
        recipe.restore(checkpoints / "step_000000", model, identity)
    from peft import get_peft_model_state_dict
    assert _adapter_fingerprint(get_peft_model_state_dict(model)) == source["adapter_state_sha256"], "Model is not the declared supervised warm start"
    reference_started = time.perf_counter()
    references = _reference_cache(model, schedule, run, identity)
    reference_seconds = time.perf_counter() - reference_started
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, config["max_steps"] // 20), config["max_steps"])
    completed = sorted(checkpoints.glob("step_*/COMMIT.json"))
    performance = dict(schema="decision-rl-timing-v1", optimizer_updates=0, presented_tokens=0,
                       training_seconds=0., development_seconds=0., checkpoint_seconds=0., reference_seconds=reference_seconds)

    def evaluate(step, history):
        phase = time.perf_counter()
        predictions = recipe.predict(model, data["development"])
        metrics = recipe.report(predictions)
        performance["development_seconds"] += time.perf_counter() - phase
        return dict(step=step, metrics=metrics, retention=recipe.retention(metrics, history[0]["metrics"], config) if history else dict(passed=True, reasons=[]),
                    utility=development_utility(predictions, rl_config))

    if completed:
        meta = recipe.restore(completed[-1].parent, model, identity, optimizer, scheduler)
        step, history = meta["step"], meta["history"]
    else:
        torch.manual_seed(rl_config["seed"])
        random.seed(rl_config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rl_config["seed"])
        step, history = 0, [evaluate(0, [])]
        phase = time.perf_counter()
        recipe.checkpoint(checkpoints, model, optimizer, scheduler, 0, history, identity)
        performance["checkpoint_seconds"] += time.perf_counter() - phase
    performance["resumed_from"] = step
    if on_event:
        for entry in history:
            on_event("development", entry)
    try:
        for step in tqdm(range(step + 1, config["max_steps"] + 1), desc="Decision-RL updates"):
            phase = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            model.train()
            total, diagnostics, tokens = None, {}, 0
            for micro in range(config["accumulation"]):
                fields = schedule[(step - 1) * config["accumulation"] + micro]["fields"]
                predictions = [recipe.logits(model, row) for row in fields]
                reference = [references[recipe.digest([row["input_ids"], row["code_ids"], row["keys"]])] for row in fields]
                loss, metrics = objective(predictions, fields, reference, rl_config)
                if fields[0]["supervision"] == "teacher":
                    loss = loss * config["teacher_weight"]
                (loss / config["accumulation"]).backward()
                total = loss.detach() if total is None else total + loss.detach()
                for name, value in metrics.items():
                    diagnostics[name] = diagnostics.get(name, 0) + value / config["accumulation"]
                tokens += sum(len(row["input_ids"]) for row in fields)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_value, grad_norm = float(total) / config["accumulation"], float(norm)
            seconds = time.perf_counter() - phase
            performance["optimizer_updates"] += 1
            performance["presented_tokens"] += tokens
            performance["training_seconds"] += seconds
            entry = dict(step=step, train_loss=loss_value, grad_norm=grad_norm,
                         learning_rate=optimizer.param_groups[0]["lr"], seconds=seconds, tokens=tokens,
                         rl={name: float(value) for name, value in diagnostics.items()})
            recipe.atomic_json(run / "LAST_RL_UPDATE.json", entry)
            recipe.atomic_json(run / "updates" / f"step_{step:06d}.json", entry)
            if on_event:
                on_event("update", entry)
            if step % config["evaluate_every"] == 0 or step == config["max_steps"]:
                history.append(evaluate(step, history))
                history[-1].update(rl=entry["rl"], train_loss=entry["train_loss"])
                if on_event:
                    on_event("development", history[-1])
                print(dict(step=step, macro_nll=history[-1]["metrics"]["macro_nll"], retention=history[-1]["retention"]), flush=True)
            if step % config["checkpoint_every"] == 0 or step % config["evaluate_every"] == 0 or step == config["max_steps"]:
                phase = time.perf_counter()
                recipe.checkpoint(checkpoints, model, optimizer, scheduler, step, history, identity)
                performance["checkpoint_seconds"] += time.perf_counter() - phase
    except BaseException as error:
        recipe.atomic_json(run / "INTERRUPTED.json", dict(error=type(error).__name__, attempted_step=step, recovery="Resume the same plan from its last committed checkpoint"))
        raise
    finally:
        performance.update(completed_step=performance["resumed_from"] + performance["optimizer_updates"], elapsed_seconds=time.perf_counter() - started)
        recipe.atomic_json(run / "PERFORMANCE.json", performance)
    selected = min((entry for entry in history if entry["retention"]["passed"]), key=lambda entry: (entry["metrics"]["macro_nll"], entry["step"]))
    selection = dict(identity=identity, selected_step=selected["step"], history=history, rule=plan["selection"],
                     calibration_used=False, gate_used=False, test_opened=False)
    recipe.immutable_json(run / "SELECTION.json", selection)
    recipe.restore(checkpoints / f"step_{selected['step']:06d}", model, identity)
    return selection
