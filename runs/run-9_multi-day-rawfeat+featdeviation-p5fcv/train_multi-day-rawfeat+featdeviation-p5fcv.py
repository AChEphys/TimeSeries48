#!/usr/bin/env python3
"""
Train rolling LightGBM models on multi-day feature sets.

Feature family "multi-day" (stage 5, T-12 (b)/(c) in claude/project-notes.md): on top
of the 48 raw features, each stock gets columns from its own past, using only what is
known on the prediction date:
- labelhist-max / labelhist-min: the stock's own realized labels (dates <= t -
  LABEL_LAG), with the trailing max or min label (two versions, runs 8.1 / 8.2);
- dev:       each feature's deviation from its own trailing week / month mean;
- lags:      each feature's value 1 and 5 trading dates back (naive control).
Everything else is the baseline's: train_baseline.py is imported as a library for
the validation protocol (purged k-fold, retrain), LightGBM parameters, curve monitor
and helpers. This script only adds its head, the feature sets (`FEATURE_SETS`,
`build_features`) and a `train()` that builds them; otherwise `train()` mirrors
train_baseline.train().

Rows are the baseline's: only the raw features and the label must be finite. History
columns may be missing (new listings, suspensions, the first dates of the data);
LightGBM treats them as missing values.

"Baseline" is reserved for the original 48 features (user decision, 2026-10-24).

Archive: this script goes to ARXIV_FILE, and the imported train_baseline.py to
RUN_DIR/archive/train_baseline.py, so a run can be reproduced from its folder.

BY: Claude.
"""

from __future__ import annotations

import gc
import inspect
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import r2_score

import train_baseline as tb

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data.parquet"
RUN_DIR = ROOT / "runs/run-9_multi-day-rawfeat+featdeviation-p5fcv"  # NOTE: Set per run.
RESULT_FILE = RUN_DIR / "results_lgbm.json"
MODEL_DIR = RUN_DIR / "models"
PREDICTIONS_DIR = RUN_DIR / "predictions"
CURVES_DIR = RUN_DIR / "curves"
ARXIV_FILE = RUN_DIR / "train_multi-day-rawfeat+featdeviation-p5fcv.py"  # NOTE: Archive copy.
CURVE_FLUSH_EVERY = 10  # NOTE: Write learning curves to disk every N iterations.
PLOT_REFRESH_SECONDS = 3  # NOTE: Refresh the learning-curve plot every N seconds.

RANDOM_SEED = 70563
SMOKE_TEST = False  # NOTE: If True, only the first rolling window is trained.
SKIP_LAST_WINDOW = True  # NOTE: The last window is reserved for the final evaluation.
FEATURE_SET = "raw+dev"  # NOTE: A FEATURE_SETS key.

CORR_BAR = 0.02  # NOTE: Protocol settings as in train_baseline (run 5); share_head()
LOOKBACK = 240  # NOTE: copies the ones its helpers use into train_baseline.
GAP = 10
TEST_DAYS = 120
VALIDATION_FOLD = 5
PURGE = GAP
SELECT_FEATURES = False # NOTE: If True, filter features based on correlation. False = use all features.
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 200

LABEL_LAG = 6  # NOTE: On date t, labels of dates <= t - 6 are realized (user, 2026-10-25).
LABEL_MEANS = (20, 60)  # NOTE: Trailing label means over these many dates (month, quarter).
LABEL_SPREAD = 20  # NOTE: Trailing label SD and max / min over these many dates.
DEV_WINDOWS = (5, 20)  # NOTE: Feature deviations from the mean of dates t-n ... t-1.
LAGS = (1, 5)  # NOTE: Feature values n dates back.
STALE_LIMIT = 20  # NOTE: Label history older than this many dates (suspension) is dropped.
WARMUP = max(LABEL_LAG + max(LABEL_MEANS), *DEV_WINDOWS, *LAGS)  # NOTE: Dates of history needed before a window.

SHARED = (  # NOTE: Head settings that train_baseline's helpers read from their module.
    "RUN_DIR", "MODEL_DIR", "PREDICTIONS_DIR", "CURVES_DIR", "CURVE_FLUSH_EVERY",
    "CORR_BAR", "VALIDATION_FOLD", "PURGE", "SELECT_FEATURES", "MAX_ROUNDS",
    "EARLY_STOPPING_ROUNDS",
)

FEATURE_SETS = {  # NOTE: Feature set -> its column groups (see build_features).
    "raw": ("raw",),
    "raw+labelhist-max": ("raw", "labelhist-max"),  # NOTE: Lottery (MAX) effect.
    "raw+labelhist-min": ("raw", "labelhist-min"),  # NOTE: Same with the downside extreme.
    "raw+dev": ("raw", "dev"),
    "raw+labelhist-max+dev": ("raw", "labelhist-max", "dev"),
    "raw+labelhist-min+dev": ("raw", "labelhist-min", "dev"),
    "raw+lags": ("raw", "lags"),
}


def finite(column: str) -> pl.Expr:
    """Return the column with non-finite values as null (missing)."""
    return pl.when(pl.col(column).is_finite()).then(pl.col(column))


def label_history(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Per (Code, date index d): the stock's realized label statistics known on date d.

    Statistics over the stock's labels of dates s - n + 1 ... s, counted in trading
    dates (not rows), are known from date s + LABEL_LAG on. The as-of join gives each
    row the latest statistics known by then, so a stock that didn't trade on
    d - LABEL_LAG still gets its earlier ones (up to STALE_LIMIT dates old).
    - labelhist_last:     the label of date s (the latest realized one);
    - labelhist_meanN:    mean over N dates (LABEL_MEANS);
    - labelhist_stdN / labelhist_maxN / labelhist_minN: SD, max and min over
      N = LABEL_SPREAD dates (signed: max is the best one, min the worst).
    """
    label = finite("label")
    n = LABEL_SPREAD

    def rolling(stat: str, size: int) -> pl.Expr:
        window = {"window_size": f"{size}i", "min_samples": size // 2}
        return getattr(label, f"rolling_{stat}_by")("d", **window).over("Code")

    known = (
        frame.select("Code", "d", "label")
        .sort("Code", "d")
        .select(
            "Code",
            (pl.col("d") + LABEL_LAG).alias("d"),
            label.alias("labelhist_last"),
            *[rolling("mean", m).alias(f"labelhist_mean{m}") for m in LABEL_MEANS],
            rolling("std", n).alias(f"labelhist_std{n}"),
            rolling("max", n).alias(f"labelhist_max{n}"),
            rolling("min", n).alias(f"labelhist_min{n}"),
        )
        .sort("d")
    )
    return frame.select("Code", "d").sort("d").join_asof(
        known, on="d", by="Code", strategy="backward",
        tolerance=STALE_LIMIT - LABEL_LAG,
        check_sortedness=False,  # NOTE: Both sides are sorted by d just above.
    )


def build_features(frame: pl.DataFrame, kind: str) -> tuple[pl.DataFrame, list[str]]:
    """
    Add feature set `kind` to whole days of data; return the model input names.

    `frame` holds every row of a span of consecutive trading dates, with the
    trading-date index `d` (0, 1, 2, ... over all dates of the data). Statistics are
    computed per stock over trading dates, so missing rows (suspensions) shorten a
    window instead of reaching further back. Column groups:
    - raw:       featureN as it is.
    - labelhist-max / labelhist-min: see label_history(); labels of dates <= t -
      LABEL_LAG only. Both take latest, means and SD; then max or min.
    - dev:       featureN_devK = featureN - mean of featureN over dates t-K ... t-1
                 (K in DEV_WINDOWS; at least half of those dates present).
    - lags:      featureN_lagK = featureN on date t-K (K in LAGS).
    Nothing uses a later date's features or a label that isn't realized on date t.
    The first WARMUP dates of `frame` lack (part of) their history: drop them.
    """
    if kind not in FEATURE_SETS:
        msg = f"Unknown feature set {kind!r}; choose from {list(FEATURE_SETS)}"
        raise ValueError(msg)
    raw = [c for c in frame.columns if c.startswith("feature") and "_" not in c]
    groups = FEATURE_SETS[kind]
    out = frame.sort("Code", "d")
    names = list(raw)
    if "dev" in groups:
        exprs = []
        for k in DEV_WINDOWS:
            for f in raw:
                past = finite(f).rolling_mean_by(
                    "d", window_size=f"{k}i", closed="left", min_samples=(k + 1) // 2
                ).over("Code")
                exprs.append((finite(f) - past).cast(pl.Float32).alias(f"{f}_dev{k}"))
        out = out.with_columns(exprs)
        names += [e.meta.output_name() for e in exprs]
    if "lags" in groups:
        for k in LAGS:
            lagged = out.select(
                "Code", (pl.col("d") + k).alias("d"),
                *[finite(f).alias(f"{f}_lag{k}") for f in raw],
            )
            out = out.join(lagged, on=["Code", "d"], how="left")
            names += [f"{f}_lag{k}" for f in raw]
    extreme = {"labelhist-max": "max", "labelhist-min": "min"}
    for group, keep in extreme.items():
        if group in groups:
            drop = "min" if keep == "max" else "max"
            history = label_history(out).drop(f"labelhist_{drop}{LABEL_SPREAD}")
            out = out.join(history, on=["Code", "d"], how="left")
            names += [c for c in history.columns if c.startswith("labelhist_")]
    return out.sort("Date", "Code"), names


def share_head() -> None:
    """Point train_baseline's helpers (purged_cv, curve monitor, ...) at this head."""
    for name in SHARED:
        setattr(tb, name, globals()[name])


def r2_metric(prediction: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """LightGBM feval: R² on a dataset (the retrain's training data)."""
    return "r2", float(r2_score(np.asarray(dataset.get_label()), prediction)), True


def train() -> list[dict]:
    """
    As train_baseline.train(), on feature set FEATURE_SET.

    The only differences: per window, the features are built once over the window's
    dates plus the WARMUP dates before it (history needs earlier dates); rows are
    filtered as in the baseline (raw features and label finite), so history columns
    may be missing; and the results record the feature set.
    """
    data = pl.read_parquet(DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    data = data.with_columns(pl.col("Date").rank("dense").cast(pl.Int64).alias("d"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for i in range(LOOKBACK, len(dates), TEST_DAYS):
        runtime_start = time.time()
        tb.reset_peak_memory()
        begin = i - LOOKBACK
        end = min(i + TEST_DAYS, len(dates))
        history_dates = dates[begin : i - GAP]
        test_dates = dates[i:end]
        if SKIP_LAST_WINDOW and end == len(dates):
            print(f"Skipping the last window from {dates[begin]} to {dates[end - 1]}.")  # noqa: T201
            break

        span = data.filter(pl.col("Date").is_in(dates[max(begin - WARMUP, 0) : end]))
        window, features = build_features(span, FEATURE_SET)
        del span
        raw = [f for f in features if "_" not in f]
        history = tb.finite_rows(
            window.filter(pl.col("Date").is_in(history_dates)), [*raw, "label"]
        )
        params = tb.lgbm_params(RANDOM_SEED + i)
        window_tag = (f"train_{dates[begin]}_{dates[i - GAP - 1]}_"
                      f"test_{dates[i]}_{dates[end - 1]}")
        cv = tb.purged_cv(history, history_dates, features, params, window_tag)
        best_iteration = cv.best_iteration

        selected = tb.select_features(history, features)
        test_df = tb.finite_rows(
            window.filter(pl.col("Date").is_in(test_dates)), [*raw, "label"]
        )
        del window

        rng = np.random.default_rng(RANDOM_SEED)
        full_order = rng.permutation(len(history))
        full_matrix = lgb.Dataset(
            history.select(selected).to_numpy()[full_order],
            label=history["label"].to_numpy()[full_order],
            feature_name=selected,
        )
        retrain_curves: dict[str, dict[str, list[float]]] = {}
        retrain_file = CURVES_DIR / f"retrain_{window_tag}.json"
        retrain_status: dict[str, object] = {"done": False, "rounds": best_iteration}
        tb.write_json_atomic(retrain_file, {**retrain_status, "curves": {}})
        model = lgb.train(
            params,
            full_matrix,
            num_boost_round=best_iteration,
            valid_sets=[full_matrix],
            valid_names=["retrain"],
            feval=r2_metric,
            callbacks=[
                lgb.log_evaluation(100),
                tb.record_curves(
                    retrain_curves, retrain_file, CURVE_FLUSH_EVERY, retrain_status
                ),
            ],
        )
        tb.write_json_atomic(
            retrain_file, {**retrain_status, "done": True, "curves": retrain_curves}
        )

        model_name = "lgbm_" + window_tag
        model.save_model(MODEL_DIR / f"{model_name}.txt")
        (MODEL_DIR / f"{model_name}.features.json").write_text(
            json.dumps({"tot": len(selected), "features": selected}, indent=2),
            encoding="utf-8",
        )

        test_x = test_df.select(selected).to_numpy()
        prediction = np.asarray(model.predict(test_x))
        scored = test_df.select("Date", "Code", "label").with_columns(
            pl.Series("Prediction", prediction)
        )
        scored.write_parquet(PREDICTIONS_DIR / f"predictions_{window_tag}.parquet")
        ic = tb.daily_ic(scored)
        valid_ic = ic[np.isfinite(ic)]
        real = scored["label"].to_numpy()
        missing = {  # NOTE: Share of history rows without each history column.
            f: round(history[f].null_count() / len(history), 4)
            for f in features
            if "_" in f
        }

        result = {  # NOTE: The baseline's keys (the analysis skills read them), plus feature_set.
            "train_start": str(dates[begin]),
            "retrain_end": str(dates[i - GAP - 1]),
            "folds": VALIDATION_FOLD,
            "fold_validation": cv.fold_validation,
            "test_start": str(dates[i]),
            "test_end": str(dates[end - 1]),
            "train_rows": round(np.mean(cv.fold_train_rows)),
            "validation_rows": round(np.mean(cv.fold_validation_rows)),
            "fold_train_rows": cv.fold_train_rows,
            "fold_validation_rows": cv.fold_validation_rows,
            "fold_features": cv.fold_features,
            "retrain_rows": len(history),
            "test_rows": len(test_df),
            "feature_set": FEATURE_SET,
            "features": len(selected),
            "history_missing_share": missing,
            "best_iteration": best_iteration,
            "rank_ic_best_iteration": cv.rank_ic_best,
            "cv_val_r2": cv.val_r2,
            "cv_val_rank_ic": cv.val_rank_ic,
            "r2": float(r2_score(real, prediction)),
            "kt": float(stats.kendalltau(prediction, real).statistic),
            "ic": float(valid_ic.mean()),
            "ir": float(valid_ic.mean() / valid_ic.std() * np.sqrt(len(test_dates))),
            "runtime": time.time() - runtime_start,
            "peak_memory_GB": tb.peak_memory_gb(),
        }
        results.append(result)
        RESULT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))  # noqa: T201

        del test_df, test_x, model, scored, prediction, real, ic, valid_ic
        del history, full_matrix, full_order, retrain_curves, cv
        gc.collect()
        if SMOKE_TEST:
            print("Smoke test: only the first rolling window is trained and evaluated.")  # noqa: T201
            break
    return results


def main() -> None:
    """Entry point: archive the scripts, share the head, open the monitor, train."""
    started = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), ARXIV_FILE)
    (RUN_DIR / "archive").mkdir(exist_ok=True)
    baseline_file = Path(inspect.getfile(tb)).resolve()  # NOTE: str, unlike __file__.
    shutil.copy2(baseline_file, RUN_DIR / "archive" / "train_baseline.py")
    share_head()
    tb.launch_curve_notebook(PLOT_REFRESH_SECONDS)
    results = train()
    minutes = (time.time() - started) / 60
    print(f"Completed {len(results)} windows in {minutes:.1f} minutes")  # noqa: T201


if __name__ == "__main__":
    main()
