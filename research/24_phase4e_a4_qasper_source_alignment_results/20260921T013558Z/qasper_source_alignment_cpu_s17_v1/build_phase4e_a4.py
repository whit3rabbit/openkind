"""Build self-contained Phase 4E-A4 Colab notebook from the checked CPU module."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "phase4e_a4_pipeline.py"
NOTEBOOK = ROOT / "OpenKind_Phase4E_A4_QASPER_Source_Alignment.ipynb"


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "colab": {"provenance": [], "gpuType": "", "name": NOTEBOOK.name},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# OpenKind Phase 4E-A4 — QASPER source alignment and repair preflight\n\n"
            "**CPU only; no training; no final access.** This notebook checks immutable A3 inputs, "
            "corrects one disposition in a *new* ledger, and compares four audited policy-paper "
            "rows against the pinned original QASPER parquet export. No benchmark gold, labels, "
            "states or predictions are modified.\n\n"
            "A3 returned 13 exact candidate spans for eight questions, but one span/question "
            "belongs to the calibration gate. Twelve spans across seven policy-development "
            "questions are available for a future reviewed evidence overlay. Three policy "
            "questions still lack reliable answer evidence. A3's ledger mistakenly classified "
            "those three as label/evidence repair candidates; this notebook quarantines them, "
            "changing the policy count from 20 to **17 candidates plus three unresolved**. "
            "The 150 reviewed rows were selected for audit, many because of errors; these "
            "counts are not estimates of overall QASPER prevalence."
        ),
        nbf.v4.new_markdown_cell(
            "## Run settings\n\n"
            "Use a **CPU** Colab runtime and Run all. The only setting you may need to "
            "change is `DOWNLOAD_SOURCE`, if pinned source files already exist in the "
            "local runtime cache. Reusing the same `RUN_LABEL` is safe only when the output "
            "bytes match; changing scientific inputs requires a fresh name. Source downloads "
            "are about the **upstream train and validation** parquet files only. Upstream "
            "test contains the project's closed final partition and is never downloaded.\n\n"
            "The pinned QASPER data commit and frozen state hash come from the original corpus "
            "manifest. This check establishes *dataset-source* alignment, not equivalence "
            "with a later PDF revision. Original QASPER figure/table **captions** can be "
            "inspected; missing numeric table cells are not recovered by this step."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, os\n"
            "RUN_PREFLIGHT = True\n"
            "RUN_SOURCE_ALIGNMENT = True\n"
            "DOWNLOAD_SOURCE = True\n"
            "TRAIN = False\n"
            "OPEN_FINAL = False\n"
            "BASE_RUN_ID = '20260921T013558Z'\n"
            "RUN_LABEL = 'qasper_source_alignment_cpu_s17_v1'\n"
            "if TRAIN or OPEN_FINAL:\n"
            "    raise PermissionError('Phase 4E-A4 is a CPU audit. Training and final are closed.')\n"
            "COLAB_ROOT = Path(os.environ.get('OPENKIND_COLAB_ROOT', '/content/drive/MyDrive/Colab Notebooks'))\n"
            "if 'OPENKIND_A4_LOCAL_A3' in os.environ:\n"
            "    A3_DIR = Path(os.environ['OPENKIND_A4_LOCAL_A3'])\n"
            "    CORPUS_DIR = Path(os.environ['OPENKIND_A4_LOCAL_CORPUS'])\n"
            "    OUTPUT = Path(os.environ.get('OPENKIND_A4_LOCAL_OUTPUT', '/content/openkind_a4_test'))\n"
            "else:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    A3_DIR = COLAB_ROOT / 'OpenKind_Phase4E_A3_QASPER_Followup_results' / BASE_RUN_ID / 'qasper_followup_cpu_s17_v1'\n"
            "    CORPUS_DIR = COLAB_ROOT / 'OpenDecision_Phase4A_StateQuery_results' / BASE_RUN_ID / 'opendecision-mq-v1'\n"
            "    OUTPUT = COLAB_ROOT / 'OpenKind_Phase4E_A4_QASPER_Source_Alignment_results' / BASE_RUN_ID / RUN_LABEL\n"
            "CACHE = Path(os.environ.get('OPENKIND_A4_CACHE', '/content/openkind_a4_source_cache')) / BASE_RUN_ID\n"
            "OUTPUT.mkdir(parents=True, exist_ok=True)\n"
            "print('A3:', A3_DIR, '\\nFrozen corpus:', CORPUS_DIR, '\\nA4 results:', OUTPUT)"
        ),
        nbf.v4.new_markdown_cell("## Validated, self-contained CPU implementation\n\nThe exact implementation used in local preflight is embedded below. It verifies every frozen input SHA-256 before any output."),
        nbf.v4.new_code_cell(MODULE.read_text(encoding="utf-8")),
        nbf.v4.new_code_cell(
            "if RUN_PREFLIGHT:\n"
            "    INPUTS = input_paths(A3_DIR, CORPUS_DIR)\n"
            "    source_leads, ledger_v2, frozen_states, preflight = prepare(INPUTS, OUTPUT)\n"
            "    print('Policy action counts:', json.dumps(preflight['revised_policy_actions'], indent=2))\n"
            "    print('Exact policy spans:', preflight['policy_development_exact_candidate_spans'])\n"
            "    print('No original gold edited:', not preflight['original_gold_evidence_mutated'])"
        ),
        nbf.v4.new_markdown_cell(
            "## Align pinned QASPER source rows\n\n"
            "Only the original dataset commit's **train and validation** parquet exports "
            "are fetched. Four policy-development papers are matched by split, ID, title, "
            "abstract and paragraphs to the frozen state. Figure/table captions are exported "
            "as *candidates* with data-file SHA-256 and source indices. Captions and values "
            "must still be checked against the correct PDF revision before a representation "
            "repair can be defined. The A3 calibration-gate source leads are not mined here."
        ),
        nbf.v4.new_code_cell(
            "if RUN_SOURCE_ALIGNMENT:\n"
            "    if not RUN_PREFLIGHT:\n"
            "        raise RuntimeError('Preflight must run before source alignment.')\n"
            "    alignment = align_pinned_qasper(source_leads, INPUTS, OUTPUT, CACHE, allow_download=DOWNLOAD_SOURCE)\n"
            "    print(json.dumps(alignment, indent=2))\n"
            "    if alignment['source_dataset_aligned'] != alignment['audited_policy_papers']:\n"
            "        print('Inspect QASPER_SOURCE_ROW_ALIGNMENT.csv: some source rows differ from frozen states.')\n"
            "    print('Results folder:', OUTPUT)"
        ),
        nbf.v4.new_markdown_cell(
            "## Read the result before designing an arm\n\n"
            "`AUDIT_REPAIR_LEDGER_V2.csv` retains all 150 original source rows and changes "
            "only the three unresolved action values. `EVIDENCE_SPAN_POLICY_CANDIDATES.csv` "
            "has the 12 policy-only, exact spans; none is committed gold. "
            "`QASPER_SOURCE_ROW_ALIGNMENT.csv` and `QASPER_SOURCE_CAPTION_CANDIDATES.csv` "
            "describe pinned data-source rows and caption leads. **No PDF version is proven** "
            "equivalent here, no missing table values are recovered, no Phase 4E-B "
            "training is authorized, and final remains closed.**"),
    ]
    nbf.validate(nb)
    nbf.write(nb, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
