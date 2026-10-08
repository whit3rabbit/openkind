"""Optional W&B observation of one complete, locally governed sweep."""
from contextlib import contextmanager
import json
from pathlib import Path

import train as recipe


def run_sweep_with_wandb(model_factory, data, config, output_root, context, dimension, arms=None, *,
                         tokenizer=None, project="openkind-local-decisions", entity=None, mode="offline",
                         wandb_module=None, log_every=10):
    """Observe live updates and development reports without changing the locally governed study."""
    assert mode in ("offline", "online", "disabled"), mode
    assert dimension in ("learning_rate", "rank", "loss"), dimension
    assert type(log_every) is int and log_every > 0

    def track_arm(arm, run_dir, identity, sweep_identity):
        return track_run(arm, run_dir, identity, "openkind_sweep_" + sweep_identity[:16],
                         project=project, entity=entity, mode=mode, wandb_module=wandb_module, log_every=log_every)

    # Arm validation, common-parent checks, resume and winner selection belong to the trainer.
    if dimension == "loss":
        return recipe.run_loss_sweep(model_factory, data, config, output_root, context, arms,
                                     tokenizer=tokenizer, arm_context=track_arm)
    return recipe.run_parameter_sweep(model_factory, data, config, output_root, context, dimension, arms,
                                      tokenizer=tokenizer, arm_context=track_arm)


@contextmanager
def track_run(arm, run_dir, identity, group, *, project="openkind-local-decisions", entity=None,
              mode="offline", wandb_module=None, log_every=10):
    assert mode in ("offline", "online", "disabled"), mode
    assert type(log_every) is int and log_every > 0
    run_dir = Path(run_dir)
    # Import only after study validation, keeping W&B optional for ordinary/offline tests.
    if wandb_module is None:
        import wandb as wandb_module
    resume = dict(id=identity[:24], resume="allow") if mode == "online" else {}
    run = wandb_module.init(project=project, entity=entity, mode=mode,
                            group=group, name=arm["name"],
                            config=dict(arm["config"]), dir=str(run_dir), reinit="finish_previous", **resume)
    try:
        run.define_metric("optimizer_step")
        run.define_metric("*", step_metric="optimizer_step")
    except BaseException:
        run.finish(exit_code=1)
        raise
    last_development_step = run.summary.get("last_development_step", -1)
    last_update_step = run.summary.get("last_update_step", -1)

    def on_event(kind, entry):
        nonlocal last_development_step, last_update_step
        if kind == "development":
            if entry["step"] <= last_development_step:
                return
            last_development_step = entry["step"]
            run.summary["last_development_step"] = last_development_step
            metrics = entry["metrics"]
            values = {"optimizer_step": entry["step"], "development/macro_nll": metrics["macro_nll"],
                      "development/macro_accuracy": metrics["macro_accuracy"],
                      "development/passes_retention": entry["retention"]["passed"]}
            for source, report in metrics["by_source"].items():
                for metric in ("nll", "accuracy", "brier"):
                    values[f"development/{source}/{metric}"] = report[metric]
            utility = entry.get("utility", {})
            for source, value in utility.get("by_source", {}).items():
                values[f"development/{source}/decision_utility"] = value
            if utility.get("exact_record_accuracy") is not None:
                values["development/exact_record_accuracy"] = utility["exact_record_accuracy"]
            run.log(values)
        elif kind == "update" and (entry["step"] % log_every == 0 or entry["step"] == arm["config"]["max_steps"]):
            if entry["step"] <= last_update_step:
                return
            values = {"optimizer_step": entry["step"], "train/loss": entry["train_loss"],
                     "train/grad_norm": entry["grad_norm"], "train/learning_rate": entry["learning_rate"],
                     "speed/update_seconds": entry["seconds"],
                     "speed/tokens_per_second": entry["tokens"] / max(entry["seconds"], 1e-9)}
            values.update({"decision_rl/" + name: value for name, value in entry.get("rl", {}).items()})
            run.log(values)
            last_update_step = entry["step"]
            run.summary["last_update_step"] = last_update_step

    def log_arm(selection, result):
        for entry in selection["history"]:
            on_event("development", entry)
        run.summary.update(dict(identity=identity, selected_step=result["selected_step"],
                                selected_macro_nll=result["metrics"]["macro_nll"],
                                selected_macro_accuracy=result["metrics"]["macro_accuracy"],
                                retained_parent=result["selected_step"] == 0))
        if (run_dir / "PERFORMANCE.json").exists():
            run.summary["performance"] = json.loads((run_dir / "PERFORMANCE.json").read_text())
    log_arm.on_event = on_event
    log_arm.run = run

    try:
        yield log_arm
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)



def run_finetune_with_wandb(model, data, config, run_dir, identity, *, tokenizer=None, **tracking):
    """Track a single fixed recipe with the same development-only observer as sweep arms."""
    arm = dict(name="control", config=config)
    with track_run(arm, run_dir, identity, "openkind_finetune_" + identity[:16], **tracking) as logger:
        selection = recipe.fit(model, data, config, run_dir, identity, tokenizer=tokenizer, on_event=logger.on_event)
        selected = next(entry for entry in selection["history"] if entry["step"] == selection["selected_step"])
        logger(selection, dict(selected_step=selected["step"], metrics=selected["metrics"]))
    return selection


def log_qualification(run_dir, identity, *, log_reports=False, **tracking):
    """Record the actual frozen outcome after export, including parent fallback and calibration."""
    run_dir = Path(run_dir)
    _, contract, _ = recipe.verify_export(run_dir / "export", identity)
    saved = json.loads((run_dir / "CONFIG.json").read_text())
    gate = json.loads((run_dir / "GATE_RESULT.json").read_text())
    posttraining = (run_dir / "POSTTRAINING_PLAN.json").exists()
    group = ("openkind_decision_rl_" + saved.get("posttraining_study_identity", identity)[:16] if posttraining
             else "openkind_sweep_" + saved["sweep_identity"][:16] if "sweep_identity" in saved
             else "openkind_finetune_" + identity[:16])
    arm = dict(name=run_dir.name if "sweep_identity" in saved else "control", config=saved["config"])
    if posttraining:
        plan = json.loads((run_dir / "POSTTRAINING_PLAN.json").read_text())
        arm = dict(name=plan["rl_config"]["method"], config={**saved["config"], "decision_rl": plan["rl_config"], "supervised_parent": plan["source"]})
    with track_run(arm, run_dir, identity, group, **tracking) as logger:
        logger.run.summary.update(dict(gate_decision=gate["decision"], exported_step=contract["exported_step"],
            deployed_temperature=contract["temperature"], experimental_candidate_exported=contract["exported_step"] > 0,
            retention_evidence_status=gate.get("evidence_status", "pilot screen"),
            confirmed_retention=gate.get("confirmed_retention", False), native_qualified=False, model_promoted=False))
        if log_reports:
            module = tracking.get("wandb_module")
            if module is None:
                import wandb as module
            artifact = module.Artifact("openkind-reports-" + identity[:16], type="decision-evidence")
            for name in ("BEST_MODEL_SUMMARY.json", "GATE_RESULT.json", "PERFORMANCE.json", "EXPORT_LOCK.json"):
                path = run_dir / name
                if path.is_file():
                    artifact.add_file(str(path), name=name)
            logger.run.log_artifact(artifact)
