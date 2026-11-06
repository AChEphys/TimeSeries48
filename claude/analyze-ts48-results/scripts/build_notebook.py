"""Assemble <run>/analysis.ipynb from the skill's cell files.

Usage:
    python build_notebook.py RUN_DIR [--baseline BASELINE_DIR]
        [--leaderboard RUN_DIR ...] [--force]

Cells come from ../cells/ in file-name order (NN_name.md / NN_name.py).
Comparison cells (09_* changes, 10_*/11_* comparison, 14_* hypotheses) are
included only when --baseline is given. The leaderboard cells (15_*: runs side by
side, best values in bold) are included only when --leaderboard lists run folders
(the analyzed run is added if missing).
Placeholders: __RUN_NAME__, __BASE_NAME__, __BASELINE_DIR__, __COMPARE_LINE__,
__LEADERBOARD_DIRS__.
Refuses to overwrite an existing analysis.ipynb unless --force (it may hold
edits or written findings); use set_text_cells.py to update text only.
"""

import argparse
import json
import sys
from pathlib import Path

CELLS = Path(__file__).resolve().parents[1] / "cells"
COMPARE_PREFIXES = ("09_", "10_", "11_", "14_")
LEADERBOARD_PREFIXES = ("15_",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--leaderboard", type=Path, nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run = args.run_dir.resolve()
    if not (run / "results_lgbm.json").exists():
        sys.exit(f"Not a run folder (no results_lgbm.json): {run}")
    base = args.baseline.resolve() if args.baseline else None
    if base is not None and not (base / "results_lgbm.json").exists():
        sys.exit(f"Not a run folder (no results_lgbm.json): {base}")
    if base is not None and base == run:
        base = None
    board = [p.resolve() for p in args.leaderboard] if args.leaderboard else None
    for p in board or []:
        if not (p / "results_lgbm.json").exists():
            sys.exit(f"Not a run folder (no results_lgbm.json): {p}")
    out = run / "analysis.ipynb"
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to rebuild (text cells will be reset).")

    subs = {
        "__RUN_NAME__": run.name,
        "__BASE_NAME__": base.name if base else "(none)",
        "__BASELINE_DIR__": f'Path("{base}")' if base else "None",
        "__COMPARE_LINE__": (f"Compared against the baseline run `{base.name}`."
                             if base else "No baseline comparison."),
        "__LEADERBOARD_DIRS__": "[" + ", ".join(f'"{p}"' for p in board or []) + "]",
    }
    cells = []
    for f in sorted(CELLS.iterdir()):
        if f.suffix not in {".md", ".py"}:
            continue
        if base is None and f.name.startswith(COMPARE_PREFIXES):
            continue
        if board is None and f.name.startswith(LEADERBOARD_PREFIXES):
            continue
        src = f.read_text(encoding="utf-8")
        for k, v in subs.items():
            src = src.replace(k, v)
        cell_id = f.stem.replace("_", "-") + ("" if f.suffix == ".md" else "-code")  # unique ids
        cell = {"id": cell_id, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if f.suffix == ".md":
            cell["cell_type"] = "markdown"
        else:
            cell.update(cell_type="code", execution_count=None, outputs=[])
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells; baseline: {base.name if base else 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
