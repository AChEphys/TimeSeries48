"""Execute a notebook's code cells headlessly, from the notebook's folder.

Usage:
    python run_notebook.py RUN_DIR/analysis.ipynb

Prints every cell's printed output and displayed tables to stdout, and lets
the cells save their figures to plots/. It does not write outputs back into the
notebook (the user re-runs it in VS Code to see them inline). Stops at the
first failing cell and reports which one.
"""

import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def main() -> int:
    nb_path = Path(sys.argv[1]).resolve()
    os.chdir(nb_path.parent)
    ns: dict = {"display": print}
    for cell in json.loads(nb_path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        print(f"\n===== cell {cell.get('id')} =====")
        try:
            exec(compile(src, str(cell.get("id")), "exec"), ns)
        except Exception:
            traceback.print_exc()
            print(f"FAILED in cell {cell.get('id')}")
            return 1
    print("\nALL CELLS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
