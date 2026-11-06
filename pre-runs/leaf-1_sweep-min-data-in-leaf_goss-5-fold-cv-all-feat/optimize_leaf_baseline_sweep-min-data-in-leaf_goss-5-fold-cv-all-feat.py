#!/usr/bin/env python3
"""
Sweep leaf-regularization settings with the purged k-fold only: no retrain, no test.

For every rolling window and every configuration in GRID, this runs train_baseline's
purged k-fold (`purged_cv`) with `lgbm_params()` plus the configuration's overrides.
It records the fold-mean validation metrics at the chosen tree count:
- R² of the raw and of the per-day demeaned predictions;
- Rank IC;
- calibration;
- trees and runtime.

The test dates are never touched, so picking a configuration here doesn't leak the
test sets (T-6, N-20 in claude/project-notes.md).

Output, under SWEEP_DIR:
    <config>/curves/curves_<window>.json            learning curves (folds + mean)
    <config>/validation/validation_<window>.parquet  out-of-fold predictions
    summary.csv                                      one row per (config, window)
    sweep.json                                       grid and shared settings
    <archive copy of this script>

Interrupted sweeps resume: a (config, window) whose files are complete is not re-run.

BY: Claude.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

import train_baseline as tb

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "pre-runs/leaf-1_sweep-min-data-in-leaf_goss-5-fold-cv-all-feat"
ARXIV_FILE = SWEEP_DIR / "optimize_leaf_baseline_sweep-min-data-in-leaf_goss-5-fold-cv-all-feat.py"  # NOTE: Archive copy.  # noqa: E501

GRID: dict[str, dict[str, object]] = {  # NOTE: Name -> overrides of tb.lgbm_params().
    "min-data-20": {},  # NOTE: The reference (run 5); 20 is LightGBM's default.
    "min-data-100": {"min_data_in_leaf": 100},
    "min-data-200": {"min_data_in_leaf": 200},
    "min-data-1000": {"min_data_in_leaf": 1000},
    "min-data-4000": {"min_data_in_leaf": 4000},
}


def ols_slope(p: np.ndarray, y: np.ndarray) -> float:
    """Slope b of the least-squares line y ≈ a + b·p (calibration: 1 = right size)."""
    p, y = p - p.mean(), y - y.mean()
    return float(p @ y / (p @ p))


def oof_metrics(path: Path) -> dict[str, float]:
    """
    Fold-mean validation metrics from the saved out-of-fold predictions.

    R² per fold is 1 - mean squared error / variance of that fold's labels, as in
    purged_cv's curves. "demeaned" = prediction - mean of all stocks' predictions on
    that day (the reference's standard output, N-24).
    """
    df = pl.read_parquet(path).with_columns(
        (pl.col("Prediction") - pl.col("Prediction").mean().over("Date"))
        .alias("demeaned")
    )
    r2_raw, r2_dm, slope_dev = [], [], []
    for fold in df.partition_by("fold", maintain_order=True):
        y = fold["label"].to_numpy()
        for column, out in (("Prediction", r2_raw), ("demeaned", r2_dm)):
            p = fold[column].to_numpy()
            out.append(1 - float(((y - p) ** 2).mean()) / float(y.var()))
        slope_dev.append(abs(ols_slope(fold["demeaned"].to_numpy(), y) - 1))
    return {
        "val_r2_oof": float(np.mean(r2_raw)),
        "val_r2_demeaned": float(np.mean(r2_dm)),
        "fold_slope_dev_demeaned": float(np.mean(slope_dev)),
    }


def finished(curves_file: Path, validation_file: Path) -> dict | None:
    """Return the saved curves of a completed (config, window), or None."""
    if not (curves_file.exists() and validation_file.exists()):
        return None
    saved = json.loads(curves_file.read_text())
    return saved if saved.get("best_iteration") else None


def sweep_window(
    history: pl.DataFrame, history_dates: list, features: list[str], seed: int, tag: str
) -> list[dict]:
    """Run every GRID configuration on one window's history; one summary row each."""
    rows = []
    for name, overrides in GRID.items():
        config_dir = SWEEP_DIR / name
        tb.CURVES_DIR = config_dir / "curves"  # NOTE: purged_cv() writes its curves here,
        tb.PREDICTIONS_DIR = config_dir / "validation"  # NOTE: and its out-of-fold files here.
        tb.CURVES_DIR.mkdir(parents=True, exist_ok=True)
        tb.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        curves_file = tb.CURVES_DIR / f"curves_{tag}.json"
        validation_file = tb.PREDICTIONS_DIR / f"validation_{tag}.parquet"

        started = time.time()
        saved = finished(curves_file, validation_file)
        if saved is None:
            print(f"--- {name} | {tag}")  # noqa: T201
            params = {**tb.lgbm_params(seed), **overrides}
            tb.purged_cv(history, history_dates, features, params, tag)
            saved = json.loads(curves_file.read_text())
            gc.collect()
        else:
            print(f"--- {name} | {tag}: done before, metrics from its files")  # noqa: T201
        best = saved["best_iteration"]
        curves = saved["curves"]
        rows.append({
            "config": name,
            "window": tag,
            "trees": best,
            "train_r2": curves["train"]["r2"][best - 1],
            "val_r2": curves["val"]["r2"][best - 1],
            "val_rank_ic": curves["val"]["rank_ic"][best - 1],
            **oof_metrics(validation_file),
            "runtime_s": time.time() - started,
        })
    return rows


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
    (SWEEP_DIR / "sweep.json").write_text(json.dumps({
        "grid": GRID,
        "base_params": {k: v for k, v in tb.lgbm_params(0).items() if k != "seed"},
        "seed": "RANDOM_SEED + window index",
        "RANDOM_SEED": tb.RANDOM_SEED, "LOOKBACK": tb.LOOKBACK, "GAP": tb.GAP,
        "TEST_DAYS": tb.TEST_DAYS, "VALIDATION_FOLD": tb.VALIDATION_FOLD,
        "PURGE": tb.PURGE, "SELECT_FEATURES": tb.SELECT_FEATURES,
        "MAX_ROUNDS": tb.MAX_ROUNDS, "EARLY_STOPPING_ROUNDS": tb.EARLY_STOPPING_ROUNDS,
    }, indent=2, default=str))

    data = pl.read_parquet(tb.DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    features = [c for c in data.columns if c.startswith("feature")]
    rows: list[dict] = []
    for i in range(tb.LOOKBACK, len(dates), tb.TEST_DAYS):  # NOTE: Same windows as train().
        end = min(i + tb.TEST_DAYS, len(dates))
        if tb.SKIP_LAST_WINDOW and end == len(dates):
            break
        history_dates = dates[i - tb.LOOKBACK : i - tb.GAP]
        history = tb.finite_rows(
            data.filter(pl.col("Date").is_in(history_dates)), [*features, "label"]
        )
        tag = (f"train_{dates[i - tb.LOOKBACK]}_{dates[i - tb.GAP - 1]}_"
               f"test_{dates[i]}_{dates[end - 1]}")
        rows += sweep_window(history, history_dates, features, tb.RANDOM_SEED + i, tag)
        pl.DataFrame(rows).write_csv(SWEEP_DIR / "summary.csv")  # NOTE: After every window.
        del history
        gc.collect()
        if tb.SMOKE_TEST:
            break

    summarize(pl.DataFrame(rows))
    minutes = (time.time() - started) / 60
    print(f"\nSweep finished in {minutes:.1f} minutes: {SWEEP_DIR}")  # noqa: T201


if __name__ == "__main__":
    main()
