"""Replace the text (markdown) cells that Claude writes, in place.

Usage:
    python set_text_cells.py RUN_DIR/analysis.ipynb
        [--changes C.md] [--findings F.md] [--summary S.md] [--hypotheses H.md]

Cells are matched by id (09-changes, 12-findings, 13-summary, 14-hypotheses),
falling back to their heading ("## 6. What changed", "## Findings",
"## Summary", "## Hypotheses") for notebooks built by older versions. Only
those cells change; code cells and user edits elsewhere are left untouched.
"""

import argparse
import json
import sys
from pathlib import Path

TARGETS = {  # flag: (cell id, heading prefix)
    "changes": ("09-changes", "## 6. What changed"),
    "findings": ("12-findings", "## Findings"),
    "summary": ("13-summary", "## Summary"),
    "hypotheses": ("14-hypotheses", "## Hypotheses"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    for flag in TARGETS:
        ap.add_argument(f"--{flag}", type=Path)
    args = ap.parse_args()
    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    done = []
    for flag, (cell_id, heading) in TARGETS.items():
        path = getattr(args, flag)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        if not text.lstrip().startswith(heading):
            sys.exit(f"{path} must start with '{heading}'")
        cell = next((c for c in nb["cells"] if c.get("id") == cell_id), None) or next(
            (c for c in nb["cells"] if c["cell_type"] == "markdown"
             and "".join(c["source"]).lstrip().startswith(heading)), None)
        if cell is None:
            sys.exit(f"cell for --{flag} not found (id {cell_id} / heading '{heading}'); "
                     "was the notebook built with --baseline?")
        cell["source"] = text.splitlines(keepends=True)
        done.append(flag)
    args.notebook.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"updated {', '.join(done) or 'nothing'} in {args.notebook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
