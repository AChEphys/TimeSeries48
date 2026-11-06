#!/usr/bin/env python3
"""
Test single-day feature sets with the purged k-fold only: no retrain, no test.

Sweeper of the "single-day" feature family (train_single-day-feature-sets.py, T-12 in
claude/project-notes.md). For every rolling window and every configuration in GRID,
this builds the configuration's feature set (`"features"`, a FEATURE_SETS key of that
script) and runs train_baseline's purged k-fold with `lgbm_params()` plus any other
overrides. The protocol settings are the feature script's (`share_head()`), so a
sweep validates exactly what a run of that script would train. It records the
fold-mean validation metrics at the chosen tree count:
- R² of the raw and of the per-day demeaned predictions;
- Rank IC;
- calibration;
- trees and runtime.

The test dates are never touched, so picking a feature set here doesn't leak the test
sets.

Output, under SWEEP_DIR:
    <config>/curves/curves_<window>.json            learning curves (folds + mean)
    <config>/validation/validation_<window>.parquet  out-of-fold predictions
    summary.csv                                      one row per (config, window)
    sweep.json                                       grid and shared settings
    archive/                                         the imported scripts, as run
    curves.ipynb                                     live monitor (opens in VS Code)
    plots/sweep_summary.png                          the monitor's final summary plot
    <archive copy of this script>

Interrupted sweeps resume: a (config, window) whose files are complete is not re-run.

BY: Claude (migrated from optimize_leaf_baseline.py on 2026-10-24).
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import inspect
import json
import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import polars as pl

import optimize_leaf_baseline as ol
import train_baseline as tb

sdf = importlib.import_module("train_single-day-feature-sets")  # NOTE: Hyphenated name.

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "pre-runs/feat-1_sweep-feature-sets_goss-5-fold-cv"
ARXIV_FILE = SWEEP_DIR / "test_single-day-feature-sets_goss-5-fold-cv.py"  # NOTE: Archive copy.
QUIET = {"verbosity": -1}  # NOTE: Silences LightGBM's messages; doesn't change the model.

GRID: dict[str, dict[str, object]] = {  # NOTE: Name -> overrides of tb.lgbm_params(),
    "raw": {},  # NOTE: plus "features" (an sdf.FEATURE_SETS key). The reference (run 5).
    "rank": {"features": "rank"},
    "z": {"features": "z"},
    "raw+rank": {"features": "raw+rank"},
    "rank+daystats": {"features": "rank+daystats"},
    "raw+daystats": {"features": "raw+daystats"},  # NOTE: Added 2026-10-24 (round 2).
    "raw+daymean": {"features": "raw+daymean"},
    "raw+daystd": {"features": "raw+daystd"},
    "raw+rank+daymean": {"features": "raw+rank+daymean"},
    "raw+rank+daystd": {"features": "raw+rank+daystd"},
    "raw+rank+daystats": {"features": "raw+rank+daystats"},
}


PREVIOUS_RUNTIME: dict[tuple[str, str], float] = {}  # NOTE: (config, window) -> s, from summary.csv.


def sweep_window(
    days: pl.DataFrame, history_dates: list, seed: int, tag: str
) -> list[dict]:
    """
    Run every GRID configuration on one window's history; one summary row each.

    `days` holds the history dates' rows before any filtering: per-day features need
    whole days, then finite_rows() drops rows with non-finite inputs or labels.
    """
    rows = []
    for name, config in GRID.items():
        kind = str(config.get("features", "raw"))
        overrides = {k: v for k, v in config.items() if k != "features"}
        config_dir = SWEEP_DIR / name
        tb.CURVES_DIR = config_dir / "curves"  # NOTE: purged_cv() writes its curves here,
        tb.PREDICTIONS_DIR = config_dir / "validation"  # NOTE: and its out-of-fold files here.
        tb.CURVES_DIR.mkdir(parents=True, exist_ok=True)
        tb.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        curves_file = tb.CURVES_DIR / f"curves_{tag}.json"
        validation_file = tb.PREDICTIONS_DIR / f"validation_{tag}.parquet"

        started = time.time()
        runtime = None
        saved = ol.finished(curves_file, validation_file)
        if saved is None:
            print(f"--- {name} | {tag}")  # noqa: T201
            history, features = sdf.build_features(days, kind)
            history = tb.finite_rows(history, [*features, "label"])
            params = {**tb.lgbm_params(seed), **QUIET, **overrides}
            tb.purged_cv(history, history_dates, features, params, tag)
            saved = json.loads(curves_file.read_text())
            del history
            gc.collect()
        else:
            print(f"--- {name} | {tag}: done before, metrics from its files")  # noqa: T201
            runtime = PREVIOUS_RUNTIME.get((name, tag))  # NOTE: Keep the real training time.
        best = saved["best_iteration"]
        curves = saved["curves"]
        rows.append({
            "config": name,
            "window": tag,
            "features": kind,
            "n_features": len(sdf.build_features(days.head(0), kind)[1]),
            "trees": best,
            "train_r2": curves["train"]["r2"][best - 1],
            "val_r2": curves["val"]["r2"][best - 1],
            "val_rank_ic": curves["val"]["rank_ic"][best - 1],
            **ol.oof_metrics(validation_file),
            "runtime_s": time.time() - started if runtime is None else runtime,
        })
    return rows


MONITOR_CELL = textwrap.dedent(  # NOTE: The monitor notebook's code; __X__ filled in below.
'''\
    import json
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    from IPython.display import clear_output

    SWEEP_DIR = Path("__SWEEP_DIR__")
    CONFIGS = __CONFIGS__        # Sweep settings in GRID order (colours C0, C1, ...).
    N_WINDOWS = __N_WINDOWS__
    SWEEP_PID = __PID__          # Process ID of the sweep.
    REFRESH_SECONDS = __REFRESH__
    FOLD_COLORS = ["C3", "C4", "C9", "C6", "C8", "C5", "C7"]
    _r2_cache: dict = {}


    def read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None


    def demeaned_r2(path: Path) -> float:
        """Fold-mean R² of the per-day demeaned out-of-fold predictions (cached)."""
        key = (path, path.stat().st_mtime)
        if key not in _r2_cache:
            df = pl.read_parquet(path).with_columns(
                pl.col("Prediction") - pl.col("Prediction").mean().over("Date"))
            r2 = [1 - ((f["label"] - f["Prediction"]) ** 2).mean()
                  / f["label"].var(ddof=0) for f in df.partition_by("fold")]
            _r2_cache[key] = sum(r2) / len(r2)
        return _r2_cache[key]


    def scan() -> tuple[list[dict], tuple | None]:
        """Finished (config, window) results, and the newest unfinished curves file."""
        done, running = [], None
        tags = sorted({f.stem.removeprefix("curves_") for c in CONFIGS
                       for f in (SWEEP_DIR / c / "curves").glob("curves_*.json")})
        for c in CONFIGS:
            for f in (SWEEP_DIR / c / "curves").glob("curves_*.json"):
                d = read_json(f)
                if d is None:
                    continue
                tag = f.stem.removeprefix("curves_")
                val = SWEEP_DIR / c / "validation" / f"validation_{tag}.parquet"
                best = d["best_iteration"]
                if best is not None and val.exists():
                    done.append({
                        "config": c, "window": tags.index(tag) + 1, "trees": best,
                        "val R² (demeaned)": demeaned_r2(val),
                        "val Rank IC": d["curves"]["val"]["rank_ic"][best - 1],
                    })
                elif running is None or f.stat().st_mtime > running[3]:
                    running = (c, tags.index(tag) + 1, d, f.stat().st_mtime)
        return done, running


    def draw_summary(done: list[dict]) -> plt.Figure:
        """Per window: demeaned validation R² and Rank IC of every finished setting."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), layout="constrained")
        table = pl.DataFrame(done) if done else None
        for k, c in enumerate(CONFIGS):
            rows = table.filter(pl.col("config") == c).sort("window") if done else None
            if rows is None or rows.is_empty():
                continue
            metrics = ["val R² (demeaned)", "val Rank IC"]
            for ax, metric in zip(axes, metrics, strict=True):
                ax.plot(rows["window"], rows[metric], "o-", c=f"C{k}", lw=2, ms=6,
                        label=c)
        for ax, title in zip(axes, ["Validation R² (demeaned), fold mean",
                                    "Validation Rank IC, fold mean"], strict=True):
            ax.set_title(title)
            ticks = range(1, N_WINDOWS + 1)
            ax.set_xticks(ticks, [f"W{w}" for w in ticks])
            ax.set_xlim(0.7, N_WINDOWS + 0.3)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"{SWEEP_DIR.name}: finished {len(done)} of "
                     f"{len(CONFIGS) * N_WINDOWS} (setting, window) pairs", fontsize=11)
        return fig


    def draw_running(config: str, window: int, d: dict) -> plt.Figure:
        """The (setting, window) in progress: fold-mean and per-fold curves."""
        c, folds = d["curves"], d.get("folds", {})
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout="constrained")
        axes[0].plot(c["train"]["r2"], c="C0", label="train (fold mean)")
        axes[0].plot(c["val"]["r2"], c="C1", label="val (fold mean)")
        axes[0].set_title("R²")
        for k, v in enumerate(folds.get("val_r2", [])):
            axes[1].plot(v, c=FOLD_COLORS[k % 7], lw=1.2, label=f"fold {k + 1}")
        axes[1].plot(c["val"]["r2"], c="black", lw=2, label="mean")
        axes[1].set_title("Validation R² per fold")
        for k, v in enumerate(folds.get("val_rank_ic", [])):
            axes[2].plot(v, c=FOLD_COLORS[k % 7], lw=1.2)
        axes[2].plot(c["val"]["rank_ic"], c="black", lw=2, label="mean")
        axes[2].set_title("Validation Rank IC per fold")
        for ax in axes:
            ax.set_xlabel("iteration")
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"Running: {config}, W{window}, {len(c['val']['r2'])} iterations",
                     fontsize=11)
        return fig


    # Refresh until the sweep process exits; then save the summary plot and stop.
    # Interrupt the cell to stop watching; the sweep is unaffected.
    while True:
        alive = Path(f"/proc/{SWEEP_PID}").exists()
        done, running = scan()
        clear_output(wait=True)
        fig = draw_summary(done)
        if not alive:
            (SWEEP_DIR / "plots").mkdir(exist_ok=True)
            fig.savefig(SWEEP_DIR / "plots" / "sweep_summary.png", dpi=120)
        plt.show()
        plt.close(fig)
        if done:
            print(pl.DataFrame(done).pivot(on="config", index="window",
                                           values="val R² (demeaned)").sort("window"))
        if running is not None and alive:
            fig = draw_running(*running[:3])
            plt.show()
            plt.close(fig)
        stamp = time.strftime("%H:%M:%S")
        if not alive:
            print(f"{stamp}: the sweep has exited. Final summary saved to plots/.")
            break
        print(f"{stamp}: sweep running (PID {SWEEP_PID})...")
        time.sleep(REFRESH_SECONDS)
''')


def launch_monitor(n_windows: int) -> Path:
    """Write SWEEP_DIR/curves.ipynb (a live sweep monitor) and open it in VS Code."""
    source = (MONITOR_CELL.replace("__SWEEP_DIR__", str(SWEEP_DIR))
              .replace("__CONFIGS__", json.dumps(list(GRID)))
              .replace("__N_WINDOWS__", str(n_windows))
              .replace("__PID__", str(os.getpid()))
              .replace("__REFRESH__", str(sdf.PLOT_REFRESH_SECONDS)))
    intro = (f"# Sweep monitor: `{SWEEP_DIR.name}`\n\nRun the cell below once; it "
             "refreshes until the sweep ends. Top: every finished (setting, window), "
             "per window. Bottom: the curves of the one in progress.")
    notebook = {
        "cells": [
            {"id": "monitor-intro", "cell_type": "markdown", "metadata": {},
             "source": intro.splitlines(keepends=True)},
            {"id": "monitor-live", "cell_type": "code", "execution_count": None,
             "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = SWEEP_DIR / "curves.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    code_cli = shutil.which("code")  # NOTE: VS Code's CLI, as in train_baseline.
    if code_cli is None:
        print(f"Open {path} in Jupyter to watch the sweep.")  # noqa: T201
        return path
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run([code_cli, str(path)], check=False, timeout=30)  # noqa: S603
    return path


def summarize(summary: pl.DataFrame) -> None:
    """Print each configuration's means over windows, and the windows it wins."""
    reference = next(iter(GRID))
    ref = summary.filter(pl.col("config") == reference).select(
        "window", pl.col("val_r2_demeaned").alias("ref_r2"))
    table = (
        summary.join(ref, on="window")
        .group_by("config", maintain_order=True)
        .agg(
            pl.len().alias("windows"),
            pl.col("trees").mean().round(0),
            pl.col("train_r2").mean(),
            pl.col("val_r2").mean(),
            pl.col("val_r2_demeaned").mean(),
            pl.col("val_rank_ic").mean(),
            pl.col("fold_slope_dev_demeaned").mean(),
            (pl.col("val_r2_demeaned") > pl.col("ref_r2")).sum()
            .alias("windows_beat_ref"),
            (pl.col("runtime_s").sum() / 60).round(1).alias("minutes"),
        )
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=5):
        print(f"\nMeans over windows (reference: {reference}):")  # noqa: T201
        print(table)  # noqa: T201
        print("\nDemeaned validation R² per window:")  # noqa: T201
        print(summary.pivot(on="config", index="window", values="val_r2_demeaned"))  # noqa: T201


def main() -> None:
    """Sweep GRID over all rolling windows (the last one skipped, as in training)."""
    started = time.time()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), ARXIV_FILE)
    (SWEEP_DIR / "archive").mkdir(exist_ok=True)
    for module in (sdf, tb, ol):  # NOTE: The imported scripts, as run.
        shutil.copy2(Path(inspect.getfile(module)).resolve(), SWEEP_DIR / "archive")
    sdf.share_head()  # NOTE: The feature script's protocol settings, into train_baseline.
    if (SWEEP_DIR / "summary.csv").exists():  # NOTE: Runtimes of pairs done before.
        old = pl.read_csv(SWEEP_DIR / "summary.csv")
        PREVIOUS_RUNTIME.update({(r["config"], r["window"]): r["runtime_s"]
                                 for r in old.iter_rows(named=True)})
    (SWEEP_DIR / "sweep.json").write_text(json.dumps({
        "grid": GRID,
        "feature_sets": sdf.FEATURE_SETS,
        "quiet": QUIET,
        "base_params": {k: v for k, v in tb.lgbm_params(0).items() if k != "seed"},
        "seed": "RANDOM_SEED + window index",
        "RANDOM_SEED": sdf.RANDOM_SEED, "LOOKBACK": sdf.LOOKBACK, "GAP": sdf.GAP,
        "TEST_DAYS": sdf.TEST_DAYS, "VALIDATION_FOLD": sdf.VALIDATION_FOLD,
        "PURGE": sdf.PURGE, "SELECT_FEATURES": sdf.SELECT_FEATURES,
        "MAX_ROUNDS": sdf.MAX_ROUNDS,
        "EARLY_STOPPING_ROUNDS": sdf.EARLY_STOPPING_ROUNDS,
    }, indent=2, default=str))

    data = pl.read_parquet(sdf.DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    windows = [(i, min(i + sdf.TEST_DAYS, len(dates)))  # NOTE: Same windows as train().
               for i in range(sdf.LOOKBACK, len(dates), sdf.TEST_DAYS)]
    if sdf.SKIP_LAST_WINDOW and windows[-1][1] == len(dates):
        windows = windows[:-1]
    if sdf.SMOKE_TEST:
        windows = windows[:1]
    launch_monitor(len(windows))
    rows: list[dict] = []
    for i, end in windows:
        history_dates = dates[i - sdf.LOOKBACK : i - sdf.GAP]
        days = data.filter(pl.col("Date").is_in(history_dates))
        tag = (f"train_{dates[i - sdf.LOOKBACK]}_{dates[i - sdf.GAP - 1]}_"
               f"test_{dates[i]}_{dates[end - 1]}")
        rows += sweep_window(days, history_dates, sdf.RANDOM_SEED + i, tag)
        pl.DataFrame(rows).write_csv(SWEEP_DIR / "summary.csv")  # NOTE: After every window.
        del days
        gc.collect()

    summarize(pl.DataFrame(rows))
    minutes = (time.time() - started) / 60
    print(f"\nSweep finished in {minutes:.1f} minutes: {SWEEP_DIR}")  # noqa: T201


if __name__ == "__main__":
    main()
