"""Generate the fixed-recipe notebook using the shared, tested training pipeline."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("build_t4_notebook.py")),
              init_globals={"FIXED_RECIPE_NOTEBOOK": True})
