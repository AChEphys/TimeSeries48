"""
Update multi-day_performance.ipynb: the multi-day family vs Run-2, plus Run-5.

Usage:
    python update_multiday_table.py [--force]

Columns: Run-2 (the author's baseline, leftmost), Run-5 (demeaned) (the reference:
48 raw features, seed variants 5.1/5.2 folded in), then every finished run of the
multi-day family, i.e. every run whose archived script is a copy of
train_multi-day-feature-sets.py. Each column is compared with Run-2 paired on common
rows, using the compare-ts48-baseline skill's code (which runs the analyze-ts48-results
section-7 cells), so the numbers match the other tables and notebooks.

Every column except Run-2 is **demeaned per day** (prediction − that day's mean
prediction, the standard post-processing since N-24). Run-2 stays raw, as the author
trained it (user decision, 2026-10-26). Ranking metrics are unchanged by demeaning;
magnitudes (slopes, R², RMSE) are those of the demeaned predictions.

After the table, a code cell plots test Rank IC and test R² per window for every
column (demeaned, Run-2 raw). Its data are written into the cell, and its output is
stored, so the plot shows without running anything.

Results are cached per run (cache/<run>.json), keyed on the run's files; --force
recomputes. Generated cells (ids intro, runs, grand-table, definitions, plots) are
replaced in place (the intro only once); any other cells are kept.
The grand table shows Run-2, Run-5 and the TABLE_RUNS (plus runs not yet reviewed);
the full-table cell holds every column, collapsed.
"""

# ruff: noqa: RUF001, RUF002, T201, INP001
# RUF001/RUF002: the notebook text uses Greek letters and minus signs on purpose.
# T201: the CLI reports by print. INP001: scripts/ is not a package (as in
# compare-ts48-baseline).
import argparse
import base64
import importlib.util
import io
import json
import os
import pprint
import sys
import time
from pathlib import Path
from types import ModuleType

os.environ.setdefault("MPLBACKEND", "Agg")  # The plot cell runs headless here.

SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[1]
BASELINE_SCRIPT = ROOT / "claude/compare-ts48-baseline/scripts/update_baseline_table.py"
NOTEBOOK = ROOT / "multi-day_performance.ipynb"
CACHE = SKILL / "cache"
REFERENCE_RUN = 5  # Run-5: the 48 raw features (Run-5 demeaned is the reference).
FAMILY_MARK = 'Feature family "multi-day"'  # In the family script's docstring.
MAGNITUDE_NOTE = ", demeaned per day except Run-2"
REFERENCE_NAME = "Run-5 (demeaned)"
# The grand table shows Run-2, Run-5 and only these family runs, plus every run numbered
# above REVIEWED_UP_TO (new, not yet judged); the full table is collapsed below it
# (user decision, 2026-10-27: too many columns to read or scroll).
TABLE_RUNS = (11, 17, 21, 22, 23, 25)  # Lags; labels; lags + labels x3; ff 0.5.
REVIEWED_UP_TO = 30
MULTIDAY_REFERENCE = 22  # The family's reference run (user decision, 2026-10-27).


def load_baseline_module() -> ModuleType:
    """Import the compare-ts48-baseline script (its folder isn't a package)."""
    spec = importlib.util.spec_from_file_location("update_baseline_table",
                                                  BASELINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = load_baseline_module()


def in_family(run: Path) -> bool:
    """Tell whether the run's archived script is a copy of the multi-day script."""
    script = bl.archived_script(run)
    return script is not None and FAMILY_MARK in script.read_text(encoding="utf-8")


def compare_cached(run: Path, n_ref: int, *, force: bool, log: dict) -> dict:
    """Paired comparison with Run-2 (raw), the run demeaned per day; cached per run."""
    cache_file = CACHE / f"{run.name}.json"
    stamp = [*bl.fingerprint(run), "demeaned"]
    if cache_file.exists() and not force:
        stored = json.loads(cache_file.read_text())
        if stored.get("fingerprint") == stamp:
            log["cached"].append(run.name)
            return stored["data"]
    started = time.time()
    data = bl.compare(run, n_ref, demean=True)
    cache_file.write_text(json.dumps({"fingerprint": stamp, "data": data}, indent=1))
    log["computed"].append(f"{run.name} ({time.time() - started:.0f} s)")
    return data


def column_name(run: Path) -> str:
    """Short column name; the reference column says it is demeaned."""
    is_reference = bl.run_key(run)[0] == REFERENCE_RUN
    return REFERENCE_NAME if is_reference else bl.short_name(run)


def build_columns(
    runs: list[Path], data: dict[Path, dict], skipped: list
) -> tuple[list, dict]:
    """One Column per configuration; run-N.k folds into run-N only if seed-only."""
    primaries = {bl.run_key(r)[0]: r for r in runs if bl.run_key(r)[1] == 0}
    replicates: dict[Path, list[Path]] = {}
    for run in runs:
        number, sub = bl.run_key(run)
        primary = primaries.get(number)
        if sub and primary is not None and (bl.same_but_seed(primary, run)
                                            or number in bl.CODE_ONLY_DIFFS):
            replicates.setdefault(primary, []).append(run)
        else:
            if sub and primary is not None:
                why = "differs from its run-N in more than the seed; own column"
                skipped.append((run.name, why))
            replicates.setdefault(run, [])
    columns = [bl.Column(column_name(run), data[run],
                         [data[run], *[data[r] for r in reps]])
               for run, reps in replicates.items()]
    columns.sort(key=lambda c: bl.run_key(Path(c.main["run"])))
    return columns, replicates


def table_columns(columns: list) -> list:
    """Return Run-5, the TABLE_RUNS columns and every run above REVIEWED_UP_TO."""
    def kept(c: "bl.Column") -> bool:
        number = bl.run_key(Path(c.main["run"]))[0]
        return number in {REFERENCE_RUN, *TABLE_RUNS} or number > REVIEWED_UP_TO
    return [c for c in columns if kept(c)]


def table_cells(columns: list, ref_windows: list[tuple[str, str]]) -> tuple[str, str]:
    """Markdown of the selected grand table and of the collapsed full table."""
    shown = table_columns(columns)
    family = [c.name for c in shown if c.name != REFERENCE_NAME]
    new = [c.name for c in shown
           if bl.run_key(Path(c.main["run"]))[0] > REVIEWED_UP_TO]
    hidden = len(columns) - len(shown)
    selected = (
        "## Grand table (selected columns)\n\n"
        f"Run-2, Run-5 (demeaned) and the most promising family runs: "
        f"{', '.join(family)}. **Run-{MULTIDAY_REFERENCE} is the multi-day reference**"
        + (f" (not yet reviewed: {', '.join(new)})" if new else "")
        + f". {hidden} more columns are in **All columns** below. Best and seed-noise "
        "highlights compare the columns shown here.\n\n"
        + bl.grand_table(shown, ref_windows, MAGNITUDE_NOTE))
    full = (
        f"## All columns ({len(columns) + 1})\n\n"
        "<details>\n<summary>Show every column (scroll sideways inside the box)"
        "</summary>\n\n"
        '<div style="overflow-x: auto; max-width: 100%;">\n'
        + bl.grand_table(columns, ref_windows, MAGNITUDE_NOTE)
        + "\n</div>\n\n</details>\n")
    return selected, full


# ---------------------------------------------------------------- notebook text
INTRO = """# Multi-day feature family: every run vs Run-2

Runs of the **multi-day** feature family (`train_multi-day-feature-sets.py`: the 48
raw features plus each stock's own history, e.g. realized labels, deviations, lags;
`claude/project-notes.md` N-32, N-33), each compared with **Run-2**, the author's
baseline, **paired** on the same test dates and stocks. **Run-5 (demeaned)** (48 raw
features, our best baseline and the reference) is the second column.

**Every column except Run-2 is demeaned per day** (the standard post-processing,
N-24). Run-2 stays raw, as the author trained it. Run-5 (demeaned) matches the column
of the same name in `baseline_performance.ipynb`.

Generated by the `compare-ts48-multiday` skill (`/compare-ts48-multiday`), which
detects new runs of the family and refreshes the cells below. Don't edit the
generated cells by hand; add your own cells anywhere, and they will be kept.
"""

DEFINITIONS = """## How to read the table

- **Columns are configurations.** Run-2 is the comparison partner of every column;
  Run-5 (demeaned) (48 raw features) is the reference the family has to beat. Full
  run names are in the *Runs* cell.
  - `run-N.k` folds into Run-N's column only when its archived script differs from
    run-N's in the seed alone. Otherwise (e.g. 8.1 / 8.2, different feature sets) it
    gets its own column.
  - A column with seed variants shows its **primary seed** (comparable with the
    single-seed columns) plus a muted line *"low ~ high"* with the range over its
    seeds; the header shows the seed count.
- **Each cell:** first line = the raw value; second line = % change vs Run-2
  (relative to |Run-2's value|; Run-2 shows +0.0%).
- **Demeaned predictions, except Run-2:** `demeaned = prediction − mean of all
  stocks' predictions that day` (no labels used). Rank IC, Pearson IC, Q5−Q1, Q1/Q5,
  agreement and the days, windows and months won are unchanged by construction; the
  magnitude rows (slopes, R², RMSE) are the demeaned ones. Run-2's are raw, as the
  author trained it, so part of every column's magnitude gain over Run-2 is the
  demeaning itself (Run-5: R² 0.00879 raw → 0.00942 demeaned).
- **Seed noise ± (last column):** the 95% band for the difference between two
  single-seed runs of the *same* configuration, 1.96·√2·σ_seed, with σ pooled over
  the seed groups in this table (every column with "N seeds" in its header, demeaned).
  **Differences between columns smaller than this can be seed luck.** It assumes
  every configuration is about as seed-sensitive as those; with 3 seeds each σ is rough.
- **Dark green** (bold) = the best value in the row. **Light green** = within the
  seed noise of the best.
  - No highlight: the CI and t of the difference, prediction agreement, tree count.
- **Best** means: highest for Rank IC, Pearson IC, Q5−Q1, Q5, R², days, windows and
  months won; lowest for Q1 (sharper loser detection), RMSE, mean |slope − 1| and
  runtime; **closest to 1** for calibration slopes.
- **ΔRank IC rows** (CI, t, days, windows, months) describe the paired daily
  difference run − Run-2, with Newey–West (4 lags) for the overlapping labels.
- **Comparing two columns other than Run-2** (e.g. Run-11 vs Run-5) needs their own
  paired test; reading across columns only compares levels. The paired results vs
  Run-5 are in each run's `analysis.ipynb` and in N-33.
- **Calibration slope** = slope of label on prediction (1 = right size, < 1 =
  overconfident, > 1 = underconfident). Read the per-window slopes and their mean
  distance from 1: the pooled slope can look ideal when windows cancel.
- **Regimes** are ex-ante (trailing stock-specific dispersion vs the previous year);
  the turbulent regime has only ~3 episodes.
- **Caveat:** many variants have been compared on the same 6 windows, which inflates
  the best one's edge (selection).
"""

SHOWN = 2  # Best and worst family columns in the plot (user decision, 2026-10-26).

PLOTS_MD = """## Per window: test Rank IC and test R² (demeaned; Run-2 raw)

Only the references and the extremes are drawn, so the plot stays legible: Run-2
(raw) grey and dashed, Run-5 (demeaned) black, the **{shown} best** family columns
(solid) and the **{shown} worst** (dashed), out of {total}.
- **Score:** each family column's **weaker** rank of two: its rank on Rank IC and its
  rank on R² (demeaned), over the family columns (primary seeds). Doing badly on either
  makes a column bad, since the predictions feed a portfolio optimizer that uses both.
  Ties go to the better rank sum, then the higher Rank IC.
- **Shown:** best {best}; worst {worst}. Every column is in the table above.

The data are written into the cell by the skill; re-running the cell redraws the same
figure.
"""

PLOT_CODE = """\
import matplotlib.pyplot as plt

DATA = __DATA__  # Written by the compare-ts48-multiday skill.

STYLES = {  # Fixed per role; colours from the validated family palette.
    "reference": {"color": "#8c8c8c", "ls": "--", "marker": "s", "lw": 2},
    "base": {"color": "#222222", "ls": "-", "marker": "o", "lw": 2.6},
    "best 1": {"color": "#2a6fdb", "ls": "-", "marker": "o", "lw": 2},
    "best 2": {"color": "#1a9e77", "ls": "-", "marker": "D", "lw": 2},
    "worst 1": {"color": "#c23b3b", "ls": "--", "marker": "v", "lw": 2},
    "worst 2": {"color": "#e07b1f", "ls": "--", "marker": "^", "lw": 2},
}

ticks = range(1, len(DATA["windows"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), layout="constrained")
for s in DATA["series"]:
    st = STYLES[s["role"]]
    label = s["name"]
    if s["role"] not in {"reference", "base"}:
        label += f" ({s['role']})"
    for ax, key in zip(axes, ["rank_ic", "r2"], strict=True):
        ax.plot(ticks, s[key], color=st["color"], ls=st["ls"], marker=st["marker"],
                lw=st["lw"], ms=7, label=label)
axes[0].set_title("Test Rank IC per window (mean daily)")
axes[1].set_title("Test R² per window (demeaned per day; Run-2 raw)")
for ax in axes:
    ax.set_xticks(ticks, [f"W{k}\\n{w}" for k, w in enumerate(DATA["windows"], 1)],
                  fontsize=8)
    ax.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside right center", frameon=False, fontsize=9)
plt.show()
"""


def series_of(c: "bl.Column", role: str) -> dict:
    """One plotted line: the column's per-window Rank IC and R² (its own, demeaned)."""
    return {"name": c.name, "role": role,
            "rank_ic": [round(w[0], 6) for w in c.main["window_rank_ic"]],
            "r2": [round(w[0], 6) for w in c.main["window_r2"]]}


def plot_data(columns: list, ref_windows: list[tuple[str, str]]) -> dict:
    """Return the references plus the SHOWN best and worst family columns."""
    first = columns[0].main
    series = [{"name": "Run-2", "role": "reference",
               "rank_ic": [round(w[1], 6) for w in first["window_rank_ic"]],
               "r2": [round(w[1], 6) for w in first["window_r2"]]}]
    family = []
    for c in columns:
        if bl.run_key(Path(c.main["run"]))[0] == REFERENCE_RUN:
            series.append(series_of(c, "base"))
        else:
            family.append(c)
    by_ic = sorted(family, key=lambda c: -c.main["rank_ic"][0])
    by_r2 = sorted(family, key=lambda c: -c.main["r2"][0])
    ranks = {c.name: (by_ic.index(c), by_r2.index(c)) for c in family}
    # The weaker of the two ranks decides: bad at either ranking or R² means bad.
    ranked = sorted(family, key=lambda c: (max(ranks[c.name]), sum(ranks[c.name]),
                                           -c.main["rank_ic"][0]))
    best = ranked[:SHOWN]
    worst = [c for c in ranked[::-1] if c not in best][:SHOWN]
    series += [series_of(c, f"best {k}") for k, c in enumerate(best, 1)]
    series += [series_of(c, f"worst {k}") for k, c in enumerate(worst, 1)]
    windows = [f"{start[2:7]}→{end[2:7]}" for start, end in ref_windows]
    return {"windows": windows, "total": len(family), "series": series}


def plots_md(data: dict) -> str:
    """Return the plot's explanation, naming the columns shown."""

    def names(role: str) -> str:
        shown = [s["name"] for s in data["series"] if s["role"].startswith(role)]
        return ", ".join(shown)

    return PLOTS_MD.format(shown=SHOWN, total=data["total"], best=names("best"),
                           worst=names("worst"))


def plot_cell(data: dict) -> dict:
    """Build the plot code cell, with its figure stored as output (run headless)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    literal = pprint.pformat(data, width=88, sort_dicts=False, compact=True)
    source = PLOT_CODE.replace("__DATA__", literal)
    plt.close("all")
    exec(compile(source, "plot_cell", "exec"), {"__name__": "__main__"})  # noqa: S102
    buffer = io.BytesIO()
    plt.gcf().savefig(buffer, format="png", dpi=100)
    plt.close("all")
    png = base64.b64encode(buffer.getvalue()).decode()
    return {"id": "plots", "cell_type": "code", "execution_count": None, "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": [{"output_type": "display_data", "metadata": {},
                         "data": {"image/png": png, "text/plain": ["<Figure>"]}}]}


def write_notebook(cells: dict[str, str | dict]) -> None:
    """Replace the generated cells in place (intro once); keep any other cells."""
    if NOTEBOOK.exists():
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    else:
        kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb = {"cells": [], "metadata": {"kernelspec": kernel,
                                        "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
    by_id = {c.get("id"): c for c in nb["cells"]}
    position = 0
    for cell_id, content in cells.items():
        cell = content if isinstance(content, dict) else {
            "id": cell_id, "cell_type": "markdown", "metadata": {},
            "source": content.splitlines(keepends=True)}
        if cell_id in by_id:
            index = nb["cells"].index(by_id[cell_id])
            if cell_id != "intro":  # The intro is written once; the user may edit it.
                nb["cells"][index] = cell
            position = index + 1
        else:
            nb["cells"].insert(position, cell)
            position += 1
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> int:
    """Compare every multi-day run (and Run-5) with Run-2; rewrite the notebook."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute every run")
    args = ap.parse_args()

    ref_windows = bl.test_windows(bl.REFERENCE)
    CACHE.mkdir(exist_ok=True)
    log: dict[str, list[str]] = {"computed": [], "cached": []}
    skipped: list[tuple[str, str]] = []
    runs: list[Path] = []
    for run in sorted(bl.RUNS.glob("run-*"), key=bl.run_key):
        wanted = bl.run_key(run)[0] == REFERENCE_RUN or in_family(run)
        if not run.is_dir() or not wanted:
            continue
        reason = bl.skip_reason(run, ref_windows)
        if reason:
            skipped.append((run.name, reason))
            continue
        runs.append(run)
    if not any(in_family(r) for r in runs):
        sys.exit("No finished multi-day runs to compare with Run-2.")
    data = {run: compare_cached(run, len(ref_windows), force=args.force, log=log)
            for run in runs}
    columns, replicates = build_columns(runs, data, skipped)

    lines = [f"## Runs (updated {time.strftime('%Y-%m-%d %H:%M')})", "",
             (f"- **Run-2**: `{bl.REFERENCE.name}` (the author's baseline, seed "
              f"{bl.seed_of(bl.REFERENCE)})")]
    for c in columns:
        run = bl.RUNS / c.main["run"]
        reps = replicates.get(run, [])
        variants = ", ".join(f"`{r.name}` (seed {bl.seed_of(r)})" for r in reps)
        extra = f"; seed variants: {variants}" if reps else ""
        is_reference = bl.run_key(run)[0] == REFERENCE_RUN
        role = (" — reference: 48 raw features, predictions demeaned per day"
                if is_reference else " — demeaned per day")
        lines.append(f"- **{c.name}**: `{run.name}` (primary seed {bl.seed_of(run)}"
                     f"{extra}){role}")
    low = [f"{c.name}: {100 * c.main['coverage']:.1f}%" for c in columns
           if c.main["coverage"] < 0.999]  # noqa: PLR2004
    notes = [f"Not a column: `{n}` ({why})." for n, why in skipped]
    if low:
        notes.append(f"Coverage below 100% of Run-2's rows: {', '.join(low)}.")
    runs_md = "\n".join([*lines, *(["", *notes] if notes else [])])

    plot = plot_data(columns, ref_windows)
    selected, full = table_cells(columns, ref_windows)
    write_notebook({
        "intro": INTRO,
        "runs": runs_md,
        "grand-table": selected,
        "full-table": full,
        "definitions": DEFINITIONS,
        "plots-md": plots_md(plot),
        "plots": plot_cell(plot),
    })

    print(f"updated {NOTEBOOK}")
    print(f"  computed: {', '.join(log['computed']) or 'none'}")
    print(f"  from cache: {', '.join(log['cached']) or 'none'}")
    for name, why in skipped:
        print(f"  not a column: {name}: {why}")
    for run, reps in replicates.items():
        if reps:
            print(f"  seed group {bl.short_name(run)}: {1 + len(reps)} seeds")
    first = columns[0].main
    levels = [f"Run-2 {first['rank_ic'][1]:.4f} / {first['r2'][1]:.5f}",
              *[f"{c.name} {c.main['rank_ic'][0]:.4f} / {c.main['r2'][0]:.5f}"
                for c in columns]]
    print("  Rank IC / R² per column (Run-2 raw, others demeaned): "
          + "; ".join(levels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
