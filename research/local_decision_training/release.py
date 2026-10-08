"""Stage a frozen decision adapter for a separate, explicit private Hub upload."""
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

import train as recipe


def unpack_export(archive_path, destination):
    destination = Path(destination)
    source_hash = recipe.file_digest(archive_path)
    if destination.exists():
        marker = destination / "EXTRACTED_FROM.json"
        assert marker.is_file() and json.loads(marker.read_text()) == dict(archive_sha256=source_hash), "Extraction source changed or is incomplete"
        identity = json.loads((destination / "EXPORT_LOCK.json").read_text())["identity"]
        recipe.verify_export(destination / "export", identity)
        return destination / "export"
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        assert len(entries) <= 4096 and sum(entry.file_size for entry in entries) <= 2 * 1024**3, "Not a bounded adapter bundle"
        names = [entry.filename for entry in entries]
        assert len(set(names)) == len(names), "Duplicate archive members"
        for entry in entries:
            path = Path(entry.filename)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in entry.filename, "Invalid archive path"
            assert entry.filename == "EXPORT_LOCK.json" or entry.filename.startswith("export/"), "Unexpected archive member"
            assert stat.S_IFMT(entry.external_attr >> 16) != stat.S_IFLNK, "Archive symlink"
        archive.extractall(destination)
    export = destination / "export"
    identity = json.loads((destination / "EXPORT_LOCK.json").read_text())["identity"]
    recipe.verify_export(export, identity)
    recipe.atomic_json(destination / "EXTRACTED_FROM.json", dict(archive_sha256=source_hash))
    return export


def prepare_release(export, destination, model_name="OpenKind Decision 4B", license_id=None):
    export, destination = Path(export), Path(destination)
    identity = json.loads((export.parent / "EXPORT_LOCK.json").read_text())["identity"]
    metadata, contract, config = recipe.verify_export(export, identity)
    assert contract["exported_step"] > 0, "The frozen parent was retained; it is not a new trained model"
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,100}", model_name), "Invalid model name"
    assert license_id is None or re.fullmatch(r"[a-z0-9-]+", license_id), "Use a reviewed Hub license identifier"
    if destination.exists():
        manifest = verify_release(destination)
        assert (manifest["identity"], manifest["model_name"], manifest["license"]) == (identity, model_name, license_id), "Release request changed"
        assert manifest["source_export_sha256"] == recipe.file_digest(export / "EXPORT.json")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openkind-release-", dir=destination.parent) as root:
        stage = Path(root) / "stage"
        stage.mkdir()
        # Keep the locked research bundle intact; conventional PEFT entry files sit beside it.
        shutil.copytree(export, stage / "bundle/export")
        shutil.copy2(export.parent / "EXPORT_LOCK.json", stage / "bundle/EXPORT_LOCK.json")
        shutil.copytree(export / "tokenizer", stage / "tokenizer")
        shutil.copy2(export / "adapter/adapter_model.safetensors", stage / "adapter_model.safetensors")
        adapter_config = json.loads((export / "adapter/adapter_config.json").read_text())
        adapter_config.update(base_model_name_or_path=contract["model_id"], revision=contract["model_revision"])
        recipe.atomic_json(stage / "adapter_config.json", adapter_config)
        shutil.copy2(export / "DECISION_CONTRACT.json", stage / "DECISION_CONTRACT.json")
        gate = json.loads((export / "GATE_RESULT.json").read_text())
        card_metadata = dict(language="en", library_name="peft", base_model=contract["model_id"],
                             base_model_relation="adapter", tags=["openkind", "structured-decisions", "lora"])
        if license_id is not None:
            card_metadata["license"] = license_id
        header = "\n".join(f"{key}: {json.dumps(value)}" for key, value in card_metadata.items())
        card = f"""---
{header}
---

# {model_name}

Experimental decision LoRA for the pinned `{contract['model_id']}` revision
`{contract['model_revision']}`. It returns typed Noul, Choice and finite ordinal Score
answers from selected answer-code logits. It has no text-generation inference contract.

## Evidence

Exported update: {contract['exported_step']}. Gate outcome: `{gate['decision']}`.
Retention evidence: `{gate.get('evidence_status', 'pilot screen')}`.
Confirmed retention: {gate.get('confirmed_retention', False)}.
Training stage: `{contract.get('training_stage', 'supervised decision fine-tuning')}`.
Parent comparison: {contract.get('parent_semantics', 'the original pinned Qwen checkpoint')}.
This pilot screen does not establish broad task quality, merged-model parity, Mac latency,
or native OpenKind qualification. See `bundle/export/GATE_RESULT.json` for denominators,
sparse slices and the matched-parent comparison. No leaderboard result is claimed.

## Use

Load the pinned Qwen text base and this PEFT adapter, then load `tokenizer/` and call
`decide` from `bundle/export/train.py` with `DECISION_CONTRACT.json`. Supply dynamic
options, including a described `__none__` for Choice. The frozen contract owns rendering,
answer-code mapping, temperature and probability semantics. A generic generation pipeline
does not implement this readout. The base weights are not redistributed here.

The adapter was trained with precision `{config['precision']}` (resolved execution is in
`bundle/export/ENVIRONMENT.json`) and a {config['max_length']}-token admission cap.
Changing the substrate, merging or quantizing requires fresh probability and quality checks.

## Provenance and release status

The locked research bundle is in `bundle/`. Source pins and declared source terms are in
`bundle/export/DATA_MANIFEST.json` and `bundle/export/DECISION_CONTRACT.json`.
License metadata: {license_id or 'not declared; review base and source terms before public distribution'}.
This package is prepared for private review. Public release and registry installation are
separate steps after licensing, held-out quality, native parity, memory and latency checks.
"""
        (stage / "README.md").write_text(card, encoding="utf-8")
        manifest = dict(schema="openkind-adapter-release-v1", identity=identity, model_name=model_name,
                        model_id=contract["model_id"], model_revision=contract["model_revision"],
                        exported_step=contract["exported_step"], offline_fixture=contract["offline_fixture"],
                        license=license_id, source_export_sha256=recipe.file_digest(export / "EXPORT.json"),
                        public_release_qualified=False, native_qualified=False,
                        files={str(path.relative_to(stage)): recipe.file_digest(path)
                               for path in sorted(stage.rglob("*")) if path.is_file()})
        recipe.atomic_json(stage / "RELEASE_MANIFEST.json", manifest)
        verify_release(stage)
        stage.rename(destination)
    return destination


def verify_release(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "RELEASE_MANIFEST.json").read_text())
    assert manifest["schema"] == "openkind-adapter-release-v1"
    files = manifest["files"]
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in files)
    assert not any(path.is_symlink() for path in folder.rglob("*")), "Release symlink"
    assert set(files) == {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()} - {"RELEASE_MANIFEST.json"}, "Release inventory changed"
    for name, sha in files.items():
        assert recipe.file_digest(folder / name) == sha, "Release asset changed: " + name
    _, contract, _ = recipe.verify_export(folder / "bundle/export", manifest["identity"])
    assert contract["exported_step"] == manifest["exported_step"] > 0
    assert manifest["offline_fixture"] == contract["offline_fixture"]
    assert manifest["source_export_sha256"] == recipe.file_digest(folder / "bundle/export/EXPORT.json")
    assert manifest["model_id"] == contract["model_id"] and manifest["model_revision"] == contract["model_revision"]
    assert files["adapter_model.safetensors"] == recipe.file_digest(folder / "bundle/export/adapter/adapter_model.safetensors")
    adapter = json.loads((folder / "adapter_config.json").read_text())
    original_adapter = json.loads((folder / "bundle/export/adapter/adapter_config.json").read_text())
    assert adapter == {**original_adapter, "base_model_name_or_path": contract["model_id"], "revision": contract["model_revision"]}
    assert json.loads((folder / "DECISION_CONTRACT.json").read_text()) == contract
    assert manifest["public_release_qualified"] is False and manifest["native_qualified"] is False
    return manifest


def upload_private_release(folder, repo_id, token, *, api=None):
    manifest = verify_release(folder)
    assert not manifest["offline_fixture"], "Offline test fixtures cannot be uploaded"
    assert isinstance(repo_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id), "Specify namespace/model"
    assert token, "Provide an explicit write token"
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    # A new private repository avoids mixing this inventory with existing model files.
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False)
    assert api.repo_info(repo_id=repo_id, repo_type="model").private, "Destination must be private"
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
                               commit_message="Frozen OpenKind adapter " + manifest["identity"][:16])
    expected = set(manifest["files"]) | {"RELEASE_MANIFEST.json"}
    uploaded = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=commit.oid))
    assert expected <= uploaded <= expected | {".gitattributes"}, "Uploaded inventory differs"
    return commit
