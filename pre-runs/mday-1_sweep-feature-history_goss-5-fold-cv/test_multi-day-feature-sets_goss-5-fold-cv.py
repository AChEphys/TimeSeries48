#!/usr/bin/env python3
"""
Test multi-day feature sets with the purged k-fold only: no retrain, no test.

Sweeper of the "multi-day" feature family (train_multi-day-feature-sets.py, N-32 ...
N-34 in claude/project-notes.md). For every rolling window and every configuration in
GRID, this builds the configuration's feature set (`"features"`, a FEATURE_SETS key of
that script) over the window's history plus the dates its history columns need, and
runs train_baseline's purged k-fold with `lgbm_params()` plus any other overrides. The
protocol settings are the feature script's (`share_head()`), so a sweep validates
exactly what a run of that script would train. Rows are the baseline's (raw features
and label finite); history columns may be missing, as in training.

Use it only where validation has been shown to track test (N-33): feature history
(lags, means, deviations). Label history needs full runs.

Recorded per (config, window), at the chosen tree count, fold means of:
- R² of the raw and of the per-day demeaned predictions;
- Rank IC;
- calibration;
- trees and runtime.
The test dates are never touched.

Configs "raw" and "raw+lags" are runs 5 and 11: their curves and out-of-fold files are
copied in (REUSE) and not trained again, if the seed matches (RANDOM_SEED in
train_multi-day-feature-sets.py's head, 70563 for both runs).

Output, under SWEEP_DIR: the same layout as test_single-day-feature-sets.py
(<config>/curves, <config>/validation, summary.csv, sweep.json, archive/, the live
monitor curves.ipynb, plots/sweep_summary.png, an archive copy of this script). Its
monitor and summary are reused from that script. Interrupted sweeps resume.

BY: Claude.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import re
import shutil
import time
from pathlib import Path

import polars as pl

import optimize_leaf_baseline as ol
import train_baseline as tb

mdf = importlib.import_module("train_multi-day-feature-sets")  # NOTE: Hyphenated name.
tsd = importlib.import_module("test_single-day-feature-sets")  # NOTE: Monitor, summary.

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "pre-runs/mday-1_sweep-feature-history_goss-5-fold-cv"
ARXIV_FILE = SWEEP_DIR / "test_multi-day-feature-sets_goss-5-fold-cv.py"  # NOTE: Archive copy.
QUIET = {"verbosity": -1}  # NOTE: Silences LightGBM's messages; doesn't change the model.

GRID: dict[str, dict[str, object]] = {  # NOTE: Name -> overrides of tb.lgbm_params(),
    "raw": {},  # NOTE: plus "features" (an mdf.FEATURE_SETS key). Run 5's validation.
    "raw+lags": {"features": "raw+lags"},  # NOTE: Lags 1, 5: run 11's validation.
    "raw+lags-1": {"features": "raw+lags-1"},
    "raw+lags-1-2": {"features": "raw+lags-1-2"},
    "raw+lags-1-2-3": {"features": "raw+lags-1-2-3"},
    "raw+lags-1-2-3-4-5": {"features": "raw+lags-1-2-3-4-5"},
    "raw+lags-1-5-10": {"features": "raw+lags-1-5-10"},
    "raw+lags-1-5-20": {"features": "raw+lags-1-5-20"},
    "raw+lags-1-2-5-10-20": {"features": "raw+lags-1-2-5-10-20"},
    "raw+fmean-1-5+fmean-6-20": {"features": "raw+fmean-1-5+fmean-6-20"},
    "raw+lags-1+fmean-2-5+fmean-6-20": {"features": "raw+lags-1+fmean-2-5+fmean-6-20"},
}

REUSE = {  # NOTE: Config -> finished run whose purged k-fold is exactly that config's.
    "raw": ROOT / "runs/run-5_baseline-all-feat-purged-5-fold-val-retrain",
    "raw+lags": ROOT / "runs/run-11_multi-day-rawfeat+lag1feat+lag5feat-p5fcv",
}
SEED_LINE = re.compile(r"^RANDOM_SEED\s*=\s*(\d+)", re.MULTILINE)

PREVIOUS_RUNTIME: dict[tuple[str, str], float] = {}  # NOTE: (config, window) -> s, from summary.csv.


def reuse_runs() -> None:
    """
    Copy the REUSE runs' curves and out-of-fold files into SWEEP_DIR, if seeds match.

    A run's purged k-fold is the sweep's for the same feature set and seed (checked on
    synthetic data), so its windows count as done and aren't trained again. Skipped
    when the run's archived RANDOM_SEED differs from the feature script's.
    """
    for name, run in REUSE.items():
        if name not in GRID or not run.exists():
            continue
        script = next(run.glob("*.py"), None)
        found = SEED_LINE.search(script.read_text()) if script else None
        if found is None or int(found[1]) != mdf.RANDOM_SEED:
            print(f"{name}: seed differs from {run.name}'s; trained, not reused")  # noqa: T201
            continue
        copied = 0
        for curves in sorted((run / "curves").glob("curves_*.json")):
            tag = curves.stem.removeprefix("curves_")
            validation = run / "predictions" / f"validation_{tag}.parquet"
            target = SWEEP_DIR / name
            if not validation.exists() or (target / "curves" / curves.name).exists():
                continue
            (target / "curves").mkdir(parents=True, exist_ok=True)
            (target / "validation").mkdir(parents=True, exist_ok=True)
            shutil.copy2(curves, target / "curves" / curves.name)
            shutil.copy2(validation, target / "validation" / validation.name)
            copied += 1
        print(f"{name}: {copied} window(s) reused from {run.name}")  # noqa: T201


def sweep_window(
    span: pl.DataFrame, history_dates: list, seed: int, tag: str
) -> list[dict]:
    """
    Run every GRID configuration on one window's history; one summary row each.

    `span` holds whole days: the history dates plus the dates before them that the
    largest feature set's history columns need (mdf.warmup). Features are built on
    it, then only the history dates are kept, filtered as in training.
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
        _, features = mdf.build_features(span.head(0), kind)
        if saved is None:
            print(f"--- {name} | {tag}")  # noqa: T201
            frame, features = mdf.build_features(span, kind)
            raw = [f for f in features if "_" not in f]
            history = tb.finite_rows(
                frame.filter(pl.col("Date").is_in(history_dates)), [*raw, "label"]
            )
            del frame
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
            "n_features": len(features),
            "trees": best,
            "train_r2": curves["train"]["r2"][best - 1],
            "val_r2": curves["val"]["r2"][best - 1],
            "val_rank_ic": curves["val"]["rank_ic"][best - 1],
            **ol.oof_metrics(validation_file),
            "runtime_s": time.time() - started if runtime is None else runtime,
        })
    return rows


def main() -> None:
    """Sweep GRID over all rolling windows (the last one skipped, as in training)."""
    started = time.time()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), ARXIV_FILE)
    (SWEEP_DIR / "archive").mkdir(exist_ok=True)
    for module in (mdf, tsd, tb, ol):  # NOTE: The imported scripts, as run.
        shutil.copy2(Path(inspect.getfile(module)).resolve(), SWEEP_DIR / "archive")
    mdf.share_head()  # NOTE: The feature script's protocol settings, into train_baseline.
    vars(tsd).update(SWEEP_DIR=SWEEP_DIR, GRID=GRID)  # NOTE: For its monitor, summary.
    reuse_runs()
    if (SWEEP_DIR / "summary.csv").exists():  # NOTE: Runtimes of pairs done before.
        old = pl.read_csv(SWEEP_DIR / "summary.csv")
        PREVIOUS_RUNTIME.update({(r["config"], r["window"]): r["runtime_s"]
                                 for r in old.iter_rows(named=True)})
    kinds = [str(c.get("features", "raw")) for c in GRID.values()]
    (SWEEP_DIR / "sweep.json").write_text(json.dumps({
        "grid": GRID,
        "feature_sets": {k: mdf.FEATURE_SETS[k] for k in kinds},
        "quiet": QUIET,
        "base_params": {k: v for k, v in tb.lgbm_params(0).items() if k != "seed"},
        "seed": "RANDOM_SEED + window index",
        "RANDOM_SEED": mdf.RANDOM_SEED, "LOOKBACK": mdf.LOOKBACK, "GAP": mdf.GAP,
        "TEST_DAYS": mdf.TEST_DAYS, "VALIDATION_FOLD": mdf.VALIDATION_FOLD,
        "PURGE": mdf.PURGE, "SELECT_FEATURES": mdf.SELECT_FEATURES,
        "MAX_ROUNDS": mdf.MAX_ROUNDS,
        "EARLY_STOPPING_ROUNDS": mdf.EARLY_STOPPING_ROUNDS,
        "LABEL_LAG": mdf.LABEL_LAG,
    }, indent=2, default=str))

    data = pl.read_parquet(mdf.DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    data = data.with_columns(pl.col("Date").rank("dense").cast(pl.Int64).alias("d"))
    windows = [(i, min(i + mdf.TEST_DAYS, len(dates)))  # NOTE: Same windows as train().
               for i in range(mdf.LOOKBACK, len(dates), mdf.TEST_DAYS)]
    if mdf.SKIP_LAST_WINDOW and windows[-1][1] == len(dates):
        windows = windows[:-1]
    if mdf.SMOKE_TEST:
        windows = windows[:1]
    warmup = max(mdf.warmup(k) for k in kinds)
    tsd.launch_monitor(len(windows))
    rows: list[dict] = []
    for i, end in windows:
        begin = i - mdf.LOOKBACK
        history_dates = dates[begin : i - mdf.GAP]
        span_dates = dates[max(begin - warmup, 0) : i - mdf.GAP]
        span = data.filter(pl.col("Date").is_in(span_dates))
        tag = (f"train_{dates[begin]}_{dates[i - mdf.GAP - 1]}_"
               f"test_{dates[i]}_{dates[end - 1]}")
        rows += sweep_window(span, history_dates, mdf.RANDOM_SEED + i, tag)
        pl.DataFrame(rows).write_csv(SWEEP_DIR / "summary.csv")  # NOTE: After every window.
        del span
        gc.collect()

    tsd.summarize(pl.DataFrame(rows))
    minutes = (time.time() - started) / 60
    print(f"\nSweep finished in {minutes:.1f} minutes: {SWEEP_DIR}")  # noqa: T201


if __name__ == "__main__":
    main()
