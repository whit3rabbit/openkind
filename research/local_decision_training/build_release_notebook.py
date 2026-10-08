"""Generate a CPU-only, self-contained frozen-adapter release preparation notebook."""
from pathlib import Path
import textwrap
import nbformat as nbf

HERE = Path(__file__).resolve().parent
TARGET = HERE / "local_decision_release.ipynb"
cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

def code(source, hidden=False):
    cell = nbf.v4.new_code_cell(textwrap.dedent(source).strip())
    if hidden:
        cell.metadata["cellView"] = "form"
    cells.append(cell)

md("""
# Prepare a frozen OpenKind decision adapter for release

Run this with an export from the experiment notebook or the
[fixed-recipe fine-tuning notebook](local_decision_finetuning.ipynb). An existing
suitable export does not need another training run.
It verifies an existing export, generates a PEFT adapter package and an evidence-backed
model card, and optionally uploads to a **new private** Hugging Face model repository.
No model weights are trained or loaded, and no GPU is required.

`UPLOAD_TO_HUB=False` prepares local files only. A retained frozen parent is rejected as
a new model. Public distribution, merged weights, quantized builds and native registry
promotion require their own evidence and license review. The Hub package remains a
custom decision adapter with its renderer/readout, not a generic text-generation model.
""")
code("""
from pathlib import Path
import importlib
import json
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub==1.33.0"])
WORK = Path("/content/openkind_release_tools")
WORK.mkdir(parents=True, exist_ok=True)
""")
embedded_names = ("train.py", "benchmark.py", "release.py")
code("# @title Embedded export verification and release implementation\n"
     + f"EMBEDDED_SOURCE_FILES = {embedded_names!r}\n"
     + "\n".join(f"(WORK / {name!r}).write_text({(HERE / name).read_text()!r}, encoding='utf-8')" for name in embedded_names), True)
md("""
## 1. Select the frozen export

Use an existing Drive export directory, or upload `openkind_local_decision_export.zip`
to the Colab Files pane and set its path below. ZIP extraction uses a fresh directory.
Keep `EXPORT_LOCK.json` beside `export/`; both are in the generated archive.
Set `LICENSE_ID` only after reviewing the base and source terms. Leaving it empty makes
the missing license declaration explicit in the card, and the upload stays private.
""")
code("""
EXPORT_DIRECTORY = ""  # Existing path ending in export, with EXPORT_LOCK.json in its parent.
EXPORT_ZIP_PATH = "/content/openkind_local_decision_export.zip"
EXTRACT_TO = Path("/content/openkind_release_input")
STAGE_DIRECTORY = Path("/content/openkind_adapter_release")
MODEL_NAME = "OpenKind Decision 4B"
LICENSE_ID = None
UPLOAD_TO_HUB = False
HUB_REPO_ID = ""  # namespace/new-model-repo

sys.path.insert(0, str(WORK))
import train
import release
importlib.reload(train)
importlib.reload(release)
if EXPORT_DIRECTORY:
    export = Path(EXPORT_DIRECTORY)
else:
    assert Path(EXPORT_ZIP_PATH).is_file(), "Upload the frozen export archive or set EXPORT_DIRECTORY"
    export = release.unpack_export(EXPORT_ZIP_PATH, EXTRACT_TO)
identity = json.loads((export.parent / "EXPORT_LOCK.json").read_text())["identity"]
metadata, contract, config = train.verify_export(export, identity)
print("Export identity:", identity, "exported update:", contract["exported_step"])
""")
md("""
## 2. Prepare and inspect the package

The package includes adapter weights, a pinned PEFT base revision, tokenizer, decision
contract, model card and the original locked research bundle. Optimizers and base
checkpoint tensors are excluded. Re-running unchanged preparation verifies the existing
package. Changing metadata requires a new staging directory.
""")
code("""
stage = release.prepare_release(export, STAGE_DIRECTORY, MODEL_NAME, LICENSE_ID)
manifest = release.verify_release(stage)
print((stage / "README.md").read_text())
print("Staged files:", len(manifest["files"]), "directory:", stage)
print("Public release qualified:", manifest["public_release_qualified"])
""")
md("""
## 3. Optional private upload

Inspect the package before enabling `UPLOAD_TO_HUB`. Provide `HF_TOKEN` through the
environment or Colab Secrets with notebook access. Choose a new repository; existing
repositories are rejected. The returned commit pins the uploaded file inventory.
This upload does not install a profile in OpenKind or establish native qualification.
""")
code("""
if UPLOAD_TO_HUB:
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    commit = release.upload_private_release(stage, HUB_REPO_ID, token)
    print("Private release commit:", commit.commit_url)
else:
    print("Local package verified. No Hub repository was created or uploaded.")
""")
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": TARGET.name, "provenance": []},
})
for index, cell in enumerate(notebook.cells):
    cell.id = f"local-decisions-release-{index:02d}"
nbf.validate(notebook)
nbf.write(notebook, TARGET)
print(TARGET)
