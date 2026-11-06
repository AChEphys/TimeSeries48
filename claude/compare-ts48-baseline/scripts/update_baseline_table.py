"""Update baseline_performance.ipynb: every configuration vs Run-2, paired on common rows.

Usage:
    python update_baseline_table.py [--force]

Scans runs/run-*/ for finished runs, skips Run-2 itself and runs that don't cover all
of Run-2's test windows, and compares each with Run-2 using the analyze-ts48-results
skill's own section-7 code (cells 01, 02, 03, 07, 10), so the numbers match the
per-run analysis notebooks. Results are cached per run (cache/<run>.json) and
recomputed only when the run's files change, or with --force.

Seed variants: `run-N.k_*` is a seed replicate of `run-N_*` when their archived
scripts differ only in RUN_DIR, ARXIV_FILE and RANDOM_SEED (comments ignored).
Replicates don't get columns: Run-N's column shows its primary seed (run-N itself,
comparable with the single-seed columns) plus the range over all its seeds, and the
"Seed noise ±" column gives the 95% band for a difference between two single-seed
runs, 1.96·√2·σ_seed, with σ pooled over all seed groups.

The notebook's generated cells (ids runs, grand-table, definitions) are replaced in
place; the intro is written once; any other cells are left alone.
"""

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # Cells import pyplot; no display here.

SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[1]
RUNS = ROOT / "runs"
REFERENCE = RUNS / "run-2_original-baseline-six-windows"
CELLS = ROOT / "claude/analyze-ts48-results/cells"
NOTEBOOK = ROOT / "baseline_performance.ipynb"
CACHE = SKILL / "cache"
SECTION_CELLS = ["01_setup.py", "02_load.py", "03_daily.py", "07_regimes.py", "10_compare.py"]
VERSION = 1  # Bump to invalidate the cache when the metrics change.
REGIMES = ["calm", "normal", "turbulent"]
SEED_LINES = re.compile(r"^\s*(RUN_DIR|ARXIV_FILE|RANDOM_SEED)\s*=")
# Run numbers whose seed variants also differ by verified code-only changes.
CODE_ONLY_DIFFS = {
    5: "run 5 was re-run on 2026-10-23 with out-of-fold logging added (code-only; its "
       "predictions matched the original within 1.4e-17), so 5.1/5.2's scripts lack it",
}
FAMILY_MARK = 'Feature family "'  # Docstring of every feature-family script (run 7 on).
DEMEANED_RUNS = (5, 6)  # Run numbers that also get a "Run-N (demeaned)" column (user, 2026-10-23).


# ---------------------------------------------------------------- run discovery
def run_key(run: Path) -> tuple[int, int]:
    """'run-5.2_...' -> (5, 2); 'run-5_...' -> (5, 0)."""
    m = re.match(r"run-(\d+)(?:\.(\d+))?_", run.name + "_")
    return (int(m[1]), int(m[2] or 0)) if m else (10**9, 0)


def short_name(run: Path) -> str:
    """'run-5_baseline-...' -> 'Run-5'."""
    return run.name.split("_")[0].capitalize()


def prediction_files(run: Path) -> list[Path]:
    """predictions/predictions_*.parquet, or the early runs' flat layout."""
    return (sorted(run.glob("predictions/predictions_*.parquet"))
            or sorted(run.glob("predictions_*.parquet")))


def fingerprint(run: Path) -> list:
    files = [run / "results_lgbm.json", *prediction_files(run)]
    return [VERSION, *[[f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files]]


def test_windows(run: Path) -> list[tuple[str, str]]:
    results = json.loads((run / "results_lgbm.json").read_text())
    return [(r["test_start"], r["test_end"]) for r in results]


def skip_reason(run: Path, ref_windows: list[tuple[str, str]]) -> str | None:
    if not (run / "results_lgbm.json").exists():
        return "not finished (no results_lgbm.json)"
    windows = test_windows(run)
    if windows[: len(ref_windows)] != ref_windows:
        covered = len(set(windows) & set(ref_windows))
        return f"covers {covered} of Run-2's {len(ref_windows)} test windows"
    if not prediction_files(run):
        return "no prediction files"
    return None


def archived_script(run: Path) -> Path | None:
    scripts = sorted(run.glob("*.py"))
    return scripts[0] if scripts else None


def is_family_run(run: Path) -> bool:
    """
    Tell whether the run belongs to a feature family, i.e. is not a baseline run.

    "Baseline" = the original 48 raw features (user decision, 2026-10-24); the
    feature-family scripts (train_single-day-..., train_multi-day-...) say
    'Feature family "<name>"' in their docstring, and their runs are archived with it.
    """
    script = archived_script(run)
    return script is not None and FAMILY_MARK in script.read_text(encoding="utf-8")


def seed_of(run: Path) -> str:
    script = archived_script(run)
    m = re.search(r"^RANDOM_SEED\s*=\s*(\d+)", script.read_text(), re.M) if script else None
    return m[1] if m else "?"


def same_but_seed(a: Path, b: Path) -> bool:
    """True if the archived scripts differ only in RUN_DIR, ARXIV_FILE, RANDOM_SEED."""
    def code(run: Path) -> list[str] | None:
        script = archived_script(run)
        if script is None:
            return None
        lines = [line.split("#")[0].rstrip() for line in script.read_text().splitlines()]
        return [line for line in lines if line.strip() and not SEED_LINES.match(line)]
    ca, cb = code(a), code(b)
    return ca is not None and ca == cb


# ---------------------------------------------------------------- one comparison
def subtract_day_mean(ns: dict) -> list[dict]:
    """
    Demean the run's test predictions per day, in the cells' namespace.

    demeaned = prediction − mean of all stocks' predictions on that day (no labels used).
    Returns, per window, the subtracted daily means: their average, SD over days and
    range, plus the average daily label mean for comparison (label units).
    """
    import polars as pl  # noqa: PLC0415

    preds = ns["preds"]
    day = preds.group_by("window", "Date").agg(
        pl.col("Prediction").mean().alias("m"), pl.col("label").mean().alias("y"))
    stats = day.group_by("window").agg(
        pl.col("m").mean().alias("mean"), pl.col("m").std().alias("sd"),
        pl.col("m").min().alias("min"), pl.col("m").max().alias("max"),
        pl.col("y").mean().alias("label_mean")).sort("window")
    ns["preds"] = preds.with_columns(
        pl.col("Prediction") - pl.col("Prediction").mean().over("Date"))
    return stats.to_dicts()


def compare(run: Path, n_ref: int, *, demean: bool = False) -> dict:
    """Run the analysis skill's section-7 cells for run vs Run-2; return the numbers."""
    ns: dict = {"display": lambda *a, **k: None, "__name__": "__main__"}
    demean_stats: list[dict] = []
    cwd = Path.cwd()
    os.chdir(run)  # The cells take RUN_DIR from the working directory.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in SECTION_CELLS:
                src = (CELLS / cell).read_text(encoding="utf-8")
                src = src.replace("__BASELINE_DIR__", f'Path("{REFERENCE}")')
                exec(compile(src, str(CELLS / cell), "exec"), ns)  # noqa: S102
                if demean and cell == "02_load.py":  # Right after the predictions load.
                    demean_stats = subtract_day_mean(ns)
    finally:
        os.chdir(cwd)

    paired, overall = ns["paired"], ns["overall"]
    by_window = ns["by_window_delta"].sort("scope").to_dicts()[:n_ref]
    by_regime = {r["scope"]: r for r in ns["by_regime_delta"].to_dicts()}
    calib = ns["calib"].sort("window").to_dicts()[:n_ref]
    mag = ns["magnitude_all"].row(0, named=True)
    results, b_results = ns["results"][:n_ref], ns["b_results"][:n_ref]

    def both(run_value: float, base_value: float) -> list[float]:
        return [float(run_value), float(base_value)]

    def slope_dev(key: str) -> float:
        return sum(abs(c[key] - 1) for c in calib) / len(calib)

    return {
        "run": run.name,
        "demeaned": demean,
        "demean_stats": demean_stats[:n_ref],
        "coverage": ns["pair"].height / ns["b_preds"].height,
        "rank_ic": both(overall["RankIC run"], overall["RankIC base"]),
        "ci": overall["95% CI ±"],
        "t": overall["t(NW)"],
        "days_better": overall["run better %"],
        "windows_pos": sum(w["dRankIC"] > 0 for w in by_window),
        "months_pos": int((ns["by_month_delta"]["dRankIC"] > 0).sum()),
        "months": ns["by_month_delta"].height,
        "ic": both(paired["ic_run"].mean(), paired["ic_base"].mean()),
        "spread": both(paired["spread_run"].mean(), paired["spread_base"].mean()),
        "q1": both(overall["Q1 run"], overall["Q1 base"]),
        "q5": both(overall["Q5 run"], overall["Q5 base"]),
        "agreement": overall["agreement"],
        "window_rank_ic": [both(w["RankIC run"], w["RankIC base"]) for w in by_window],
        "regime_rank_ic": [both(by_regime[r]["RankIC run"], by_regime[r]["RankIC base"])
                           for r in REGIMES],
        "slope": both(mag["slope_run"], mag["slope_base"]),
        "slope_dev": both(slope_dev("slope_run"), slope_dev("slope_base")),
        "r2": both(mag["R2_run"], mag["R2_base"]),
        "rmse": both(mag["RMSE_run"], mag["RMSE_base"]),
        "window_r2": [both(c["R2_run"], c["R2_base"]) for c in calib],
        "window_slope": [both(c["slope_run"], c["slope_base"]) for c in calib],
        "runtime_min": both(sum(r["runtime"] for r in results) / 60,
                            sum(r["runtime"] for r in b_results) / 60),
        "trees": both(sum(r["best_iteration"] for r in results) / len(results),
                      sum(r["best_iteration"] for r in b_results) / len(b_results)),
    }



# ---------------------------------------------------------------- table
@dataclass
class Column:
    """One configuration: its primary seed's numbers, and all seed variants'."""

    name: str
    main: dict
    seeds: list[dict]  # Includes `main`; length 1 without replicates.
    derived: bool = False  # Post-processed predictions of another column's models.


STRONG = "background-color: rgba(46, 160, 67, 0.38); font-weight: bold"  # Best.
LIGHT = "background-color: rgba(46, 160, 67, 0.13)"  # Within seed noise of the best.
SECTION = "background-color: rgba(127, 127, 127, 0.15); text-align: left"
MUTED = "font-size: 85%; opacity: 0.7"
NOISE = "text-align: right; font-size: 90%; opacity: 0.8; border-left: 2px solid rgba(127,127,127,0.4)"
Z95 = 1.96


def seed_ceiling(groups: list[list[float]]) -> float | None:
    """95% band of a difference between two single-seed runs: 1.96·√2·σ (pooled σ)."""
    ss, df = 0.0, 0
    for g in groups:
        if len(g) >= 2:
            mean = sum(g) / len(g)
            ss += sum((v - mean) ** 2 for v in g)
            df += len(g) - 1
    return Z95 * math.sqrt(2 * ss / df) if df else None


def grand_table(columns: list[Column], ref_windows: list[tuple[str, str]],
                magnitude_note: str = "") -> str:
    """
    HTML table: Run-2 then one column per configuration, then the seed-noise band.

    magnitude_note is appended to the magnitude section titles (e.g. "demeaned").
    """
    first = columns[0].main
    names = ["Run-2", *[c.name for c in columns]]
    seed_counts = [0, *[len(c.seeds) for c in columns]]
    heads = [n + (f'<br><span style="{MUTED}">{k} seeds</span>' if k > 1 else "")
             for n, k in zip(names, seed_counts, strict=True)]
    html = ["<table>", "<thead><tr><th></th>" + "".join(
        f'<th style="text-align: right">{h}</th>' for h in heads)
        + f'<th style="{NOISE}">Seed noise ±</th></tr></thead>', "<tbody>"]

    def section(title: str) -> None:
        html.append(f'<tr><td colspan="{len(names) + 2}" style="{SECTION}"><b>{title}</b></td></tr>')

    def row(label: str, get, fmt: str, best: str | None, *, pct: bool = True,
            versus: bool = False, prefix: str = "", suffix: str = "",
            run2_text: str = "—", noise: bool = True) -> None:
        """One metric row. get(data, side) -> value; side 0 = the run, 1 = Run-2."""
        base = None if versus else get(first, 1)
        values = [base, *[get(c.main, 0) for c in columns]]
        seed_sets = [[get(s, 0) for s in c.seeds] for c in columns]
        independent = [ss for ss, c in zip(seed_sets, columns, strict=True) if not c.derived]
        ceiling = seed_ceiling(independent) if noise else None  # Derived = same models.
        shown = [None if v is None else float(format(v, fmt)) for v in values]
        known = [(i, v) for i, v in enumerate(shown) if v is not None]
        strong, light = set(), set()
        if best and known:
            score = {"max": lambda v: -v, "min": lambda v: v, "one": lambda v: abs(v - 1)}[best]
            top = min(score(v) for _, v in known)
            strong = {i for i, v in known if score(v) == top}
            if ceiling:
                light = {i for i, v in known if i not in strong and score(v) - top <= ceiling}
        cells = [f'<td style="text-align: left">{label}</td>']
        for i, v in enumerate(values):
            if v is None:
                cells.append(f'<td style="text-align: right">{run2_text}</td>')
                continue
            text = f"{prefix}{format(v, fmt)}{suffix}"
            text = f"<b>{text}</b>" if i in strong else text
            if pct:
                change = 100 * (v - base) / abs(base) if base else 0.0
                text += f'<br><span style="{MUTED}">({change:+.1f}%)</span>'
            seeds = seed_sets[i - 1] if i else []
            if len(seeds) > 1:
                lo, hi = format(min(seeds), fmt), format(max(seeds), fmt)
                spread = (f"all {prefix}{lo}{suffix}" if lo == hi
                          else f"{prefix}{lo}{suffix} ~ {prefix}{hi}{suffix}")
                text += f'<br><span style="{MUTED}">{spread}</span>'
            style = "text-align: right" + (f"; {STRONG}" if i in strong
                                           else f"; {LIGHT}" if i in light else "")
            cells.append(f'<td style="{style}">{text}</td>')
        unit = suffix if suffix == "%" else ""  # Not "/6" for counts.
        if ceiling is None:
            band = "—"
        elif float(format(ceiling, fmt)) == 0:
            band = "≈0"
        else:
            band = f"±{format(ceiling, fmt)}{unit}"
        cells.append(f'<td style="{NOISE}">{band}</td>')
        html.append("<tr>" + "".join(cells) + "</tr>")

    def pair(key: str, k: int | None = None):
        return lambda c, side: (c[key] if k is None else c[key][k])[side]

    def scalar(key: str):
        return lambda c, side: c[key] if side == 0 else None

    n_months = first["months"]
    section("Ranking (paired with Run-2 on common rows)")
    row("Rank IC (mean daily)", pair("rank_ic"), ".4f", "max")
    row("ΔRank IC vs Run-2: 95% CI (NW, day sampling)", scalar("ci"), ".4f", None,
        pct=False, versus=True, prefix="±", noise=False)
    row("ΔRank IC vs Run-2: t (NW)", scalar("t"), ".1f", None, pct=False, versus=True,
        noise=False)
    row("Days better than Run-2", scalar("days_better"), ".0f", "max", pct=False,
        versus=True, suffix="%")
    row("Windows with ΔRank IC > 0", scalar("windows_pos"), ".0f", "max", pct=False,
        versus=True, suffix=f"/{len(ref_windows)}")
    row("Months with ΔRank IC > 0", scalar("months_pos"), ".0f", "max", pct=False,
        versus=True, suffix=f"/{n_months}")
    row("Pearson IC (mean daily)", pair("ic"), ".4f", "max")
    row("Q5−Q1 (mean daily)", pair("spread"), ".4f", "max")
    row("Q1 mean label (lower = better)", pair("q1"), ".4f", "min")
    row("Q5 mean label", pair("q5"), ".4f", "max")
    row("Prediction agreement with Run-2", scalar("agreement"), ".2f", None, pct=False,
        run2_text="1.00")

    section("Rank IC per window")
    for k, (start, end) in enumerate(ref_windows):
        row(f"W{k + 1} ({start} → {end})", pair("window_rank_ic", k), ".4f", "max")

    section("Rank IC per ex-ante regime")
    for k, regime in enumerate(REGIMES):
        row(regime, pair("regime_rank_ic", k), ".4f", "max")

    section(f"Magnitudes (common rows{magnitude_note})")
    row("Calibration slope, pooled (best = closest to 1)", pair("slope"), ".3f", "one",
        pct=False)
    row("Mean |slope − 1| per window (lower = better)", pair("slope_dev"), ".3f", "min",
        pct=False)
    row("R², pooled", pair("r2"), ".5f", "max")
    row("RMSE, pooled", pair("rmse"), ".5f", "min")

    per_window = f" ({magnitude_note.removeprefix(', ')})" if magnitude_note else ""
    section(f"R² per window{per_window}")
    for k, (start, end) in enumerate(ref_windows):
        row(f"W{k + 1} ({start} → {end})", pair("window_r2", k), ".5f", "max")

    section(f"Calibration slope per window{per_window} (best = closest to 1)")
    for k, (start, end) in enumerate(ref_windows):
        row(f"W{k + 1} ({start} → {end})", pair("window_slope", k), ".3f", "one", pct=False)

    section("Cost")
    row("Runtime, all windows (minutes)", pair("runtime_min"), ".1f", "min")
    row("Mean trees per window", pair("trees"), ".0f", None)
    html += ["</tbody>", "</table>"]
    return "\n".join(html)


# ---------------------------------------------------------------- notebook
INTRO = """# Baseline performance: every baseline run vs Run-2

**Run-2** (`runs/run-2_original-baseline-six-windows`) is the author's baseline,
trained exactly as specified. Every later **baseline** run (the original 48 raw
features; runs 3 to 6.x) is compared with it **paired**: on the
same test dates and stocks (common rows), using the same code as section 7 of the
per-run `analysis.ipynb` notebooks. The final reserved window is never included.

Generated by the `compare-ts48-baseline` skill (`/compare-ts48-baseline`), which
detects new runs and refreshes the cells below. Don't edit the generated cells by
hand; add your own cells anywhere, and they will be kept.
"""

DEFINITIONS = """## How to read the table

- **Columns are configurations.** Seed variants (`run-N.k`, identical to `run-N`
  except `RANDOM_SEED`) get no column of their own. Run-N's column shows its
  **primary seed** (run-N itself, so it stays comparable with the single-seed
  columns), plus a muted line *"low ~ high"* (or *"all x"*) with the range over all
  its seeds; the header shows the seed count (e.g. *"3 seeds"*).
- **Each cell:** first line = the raw value; second line = % change vs Run-2
  (relative to |Run-2's value|; Run-2 shows +0.0%).
- **Seed noise ± (last column):** the 95% band for the difference between two
  single-seed runs of the *same* configuration, 1.96·√2·σ_seed, with σ pooled over
  all seed groups. **Differences between columns smaller than this can be seed
  luck.**
  - It assumes every configuration is about as seed-sensitive as the measured ones.
  - With few seeds (3 so far) σ is rough; the true value could plausibly be 2–3×
    larger.
  - It is a different noise source from day sampling, which the "95% CI (NW, day
    sampling)" row covers. Day noise is ~3× larger for overall Rank IC; seed noise
    matters more per window (e.g. W6).
- **Dark green** (bold) = the best value in the row. **Light green** = within the
  seed noise of the best, i.e. not distinguishable from it by seed alone.
  - Rows without a well-defined "best" have no highlight: the CI and t of the
    difference, prediction agreement, and the tree count.
- **Best** means: highest for Rank IC, Pearson IC, Q5−Q1, Q5, R², days, windows and
  months won; lowest for Q1 (sharper loser detection), RMSE, mean |slope − 1| and
  runtime; **closest to 1** for calibration slopes.
- **ΔRank IC rows** (CI, t, days, windows, months) describe the paired daily
  difference run − Run-2, with Newey–West (4 lags) for the overlapping labels.
- **Calibration slope** = slope of label on prediction (1 = right size, < 1 =
  overconfident, > 1 = underconfident). Read the per-window slopes and their mean
  distance from 1: the pooled slope can look ideal when over- and underconfident
  windows cancel.
- **Regimes** are ex-ante (trailing stock-specific dispersion vs the previous year);
  the turbulent regime has only ~3 episodes.
- **Caveat:** with many variants compared on the same 6 windows, the best one's
  edge is inflated (selection). The per-run notebooks hold the details and the
  interpretation.
"""


def definitions(demeaned: list[Column]) -> str:
    """The "How to read" text, plus the demeaning bullet and its per-window table."""
    if not demeaned:
        return DEFINITIONS
    names = ", ".join(c.name for c in demeaned)
    text = DEFINITIONS + f"""- **Demeaned columns ({names}):** the same models' test predictions with
  each day's mean prediction subtracted:
  `demeaned = prediction − mean of all stocks' predictions that day`.
  - No labels are used; the day's mean is known when predicting.
  - Rank IC, Pearson IC, Q5−Q1, agreement, and the windows, days and months won are
    unchanged by construction. Only the magnitudes (slopes, R², RMSE) can differ.
    Runtime and trees are the raw run's, since demeaning costs nothing.
  - Why: the label is nearly demeaned per day, and the model's daily mean prediction
    doesn't track it (corr ≈ −0.16), so the day-level part is pure error
    (`claude/project-notes.md` N-24).
  - Their seed variants are demeaned too (the muted range line). They don't enter
    the seed-noise column, because they are the same models as the raw column.
  - **Means subtracted per window** (label units, primary seed): the average over
    test days of the daily mean prediction, its SD over days, and for comparison the
    average daily label mean.

"""
    head = "| Window |"
    rule = "|---|"
    for c in demeaned:
        short = c.name.replace(" (demeaned)", "")
        head += f" {short}: mean subtracted | {short}: SD over days |"
        rule += "---|---|"
    head += " Daily label mean |"
    rule += "---|"
    rows = []
    for k, stat in enumerate(demeaned[0].main["demean_stats"]):
        line = f"| W{stat['window']} |"
        for c in demeaned:
            st = c.main["demean_stats"][k]
            line += f" {st['mean']:+.5f} | {st['sd']:.5f} |"
        rows.append(line + f" {stat['label_mean']:+.5f} |")
    return text + "\n".join([head, rule, *rows]) + "\n"


def write_notebook(runs_md: str, table_html: str, definitions_md: str) -> None:
    generated = {
        "intro": INTRO,
        "runs": runs_md,
        "grand-table": "## Grand table\n\n" + table_html,
        "definitions": definitions_md,
    }
    if NOTEBOOK.exists():
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    else:
        nb = {"cells": [], "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    by_id = {c.get("id"): c for c in nb["cells"]}
    position = 0
    for cell_id, text in generated.items():
        source = text.splitlines(keepends=True)
        if cell_id in by_id:
            if cell_id != "intro":  # The intro is written once; the user may edit it.
                by_id[cell_id]["source"] = source
            position = nb["cells"].index(by_id[cell_id]) + 1
        else:
            nb["cells"].insert(position, {"id": cell_id, "cell_type": "markdown",
                                          "metadata": {}, "source": source})
            position += 1
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute every run")
    args = ap.parse_args()

    ref_windows = test_windows(REFERENCE)
    CACHE.mkdir(exist_ok=True)
    data: dict[Path, dict] = {}
    new, cached, skipped, family = [], [], [], []
    for run in sorted(RUNS.glob("run-*"), key=run_key):
        if not run.is_dir() or run.resolve() == REFERENCE.resolve():
            continue
        if is_family_run(run):  # Not a baseline run (user decision, 2026-10-26).
            family.append(run.name)
            continue
        reason = skip_reason(run, ref_windows)
        if reason:
            skipped.append((run.name, reason))
            continue
        cache_file = CACHE / f"{run.name}.json"
        stamp = fingerprint(run)
        if cache_file.exists() and not args.force:
            stored = json.loads(cache_file.read_text())
            if stored.get("fingerprint") == stamp:
                data[run] = stored["data"]
                cached.append(run.name)
                continue
        started = time.time()
        data[run] = compare(run, len(ref_windows))
        cache_file.write_text(json.dumps({"fingerprint": stamp, "data": data[run]}, indent=1))
        new.append(f"{run.name} ({time.time() - started:.0f} s)")
    if not data:
        sys.exit("No runs to compare with Run-2.")

    # Demeaned variants (same predictions, minus each day's mean) of selected runs.
    data_dm: dict[Path, dict] = {}
    for run in [r for r in data if run_key(r)[0] in DEMEANED_RUNS]:
        cache_file = CACHE / f"{run.name}.demeaned.json"
        stamp = [*fingerprint(run), "demeaned"]
        if cache_file.exists() and not args.force:
            stored = json.loads(cache_file.read_text())
            if stored.get("fingerprint") == stamp:
                data_dm[run] = stored["data"]
                cached.append(f"{run.name} (demeaned)")
                continue
        started = time.time()
        data_dm[run] = compare(run, len(ref_windows), demean=True)
        cache_file.write_text(json.dumps({"fingerprint": stamp, "data": data_dm[run]}, indent=1))
        new.append(f"{run.name} (demeaned, {time.time() - started:.0f} s)")

    # Group seed replicates (run-N.k) under their primary run (run-N).
    primaries = {run_key(r)[0]: r for r in data if run_key(r)[1] == 0}
    columns: list[Column] = []
    replicates: dict[Path, list[Path]] = {}
    for run in data:
        number, sub = run_key(run)
        primary = primaries.get(number)
        if sub and primary is not None and (same_but_seed(primary, run)
                                            or number in CODE_ONLY_DIFFS):
            replicates.setdefault(primary, []).append(run)
        elif sub:
            skipped.append((run.name, "not a pure seed variant of its run-N; shown as its own column"))
            replicates.setdefault(run, [])
        else:
            replicates.setdefault(run, [])
    for run, reps in replicates.items():
        columns.append(Column(short_name(run), data[run], [data[run], *[data[r] for r in reps]]))
        if run in data_dm:
            columns.append(Column(f"{short_name(run)} (demeaned)", data_dm[run],
                                  [data_dm[run], *[data_dm[r] for r in reps if r in data_dm]],
                                  derived=True))
    columns.sort(key=lambda c: (run_key(Path(c.main["run"])), c.derived))

    low = [f"{c.name}: {100 * c.main['coverage']:.1f}%" for c in columns
           if c.main["coverage"] < 0.999]
    lines = [f"## Runs (updated {time.strftime('%Y-%m-%d %H:%M')})", "",
             f"- **Run-2**: `{REFERENCE.name}` (reference, seed {seed_of(REFERENCE)})"]
    for c in columns:
        run = RUNS / c.main["run"]
        if c.derived:
            lines.append(f"- **{c.name}**: {short_name(run)}'s predictions minus each day's "
                         "mean prediction (all its seeds; see *How to read*)")
            continue
        reps = replicates.get(run, [])
        extra = ("; seed variants: " + ", ".join(f"`{r.name}` (seed {seed_of(r)})" for r in reps)
                 if reps else "")
        lines.append(f"- **{c.name}**: `{run.name}` (primary seed {seed_of(run)}{extra})")
        if reps and run_key(run)[0] in CODE_ONLY_DIFFS:
            lines.append(f"  - Seed variants accepted although {CODE_ONLY_DIFFS[run_key(run)[0]]}.")
    notes = [f"Not included as a column: `{n}` ({why})." for n, why in skipped]
    if family:
        notes.append(f"Feature-family runs ({len(family)}, from run 7 on) are not "
                     "baseline runs and are left out; the multi-day family has its own "
                     "table, `multi-day_performance.ipynb`.")
    if low:
        notes.append(f"Coverage below 100% of Run-2's rows: {', '.join(low)}.")
    runs_md = "\n".join(lines + ([""] + notes if notes else []))
    write_notebook(runs_md, grand_table(columns, ref_windows),
                   definitions([c for c in columns if c.derived]))

    print(f"updated {NOTEBOOK}")
    print(f"  computed: {', '.join(new) or 'none'}")
    print(f"  from cache: {', '.join(cached) or 'none'}")
    for name, why in skipped:
        print(f"  not a column: {name}: {why}")
    print(f"  feature-family runs left out: {len(family)}")
    for run, reps in replicates.items():
        if reps:
            print(f"  seed group {short_name(run)}: {1 + len(reps)} seeds")
    print("  Rank IC / R² per column: " + "; ".join(
        f"{c.name} {c.main['rank_ic'][0]:.4f} / {c.main['r2'][0]:.5f}" for c in columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
