#!/usr/bin/env python3
"""
Train rolling LightGBM models on single-day feature sets.

Feature family "single-day" (stage 5, T-12 in claude/project-notes.md): every model
input is computed from one day's cross-section of the 48 raw features (ranks,
z-scores, daily statistics), so nothing looks across dates. Everything else is the
baseline's: train_baseline.py is imported as a library for the validation protocol
(purged k-fold, retrain), LightGBM parameters, curve monitor and helpers. This script
only adds its head, the feature sets (`FEATURE_SETS`, `build_features`) and a
`train()` that builds them; otherwise `train()` mirrors train_baseline.train().

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
RUN_DIR = ROOT / "runs/run-7_single-day-rawfeat+rank+std-p5fcv"  # NOTE: Set per run.
RESULT_FILE = RUN_DIR / "results_lgbm.json"
MODEL_DIR = RUN_DIR / "models"
PREDICTIONS_DIR = RUN_DIR / "predictions"
CURVES_DIR = RUN_DIR / "curves"
ARXIV_FILE = RUN_DIR / "train_single-day-rawfeat+rank+std-p5fcv.py"  # NOTE: Archive copy.
CURVE_FLUSH_EVERY = 10  # NOTE: Write learning curves to disk every N iterations.
PLOT_REFRESH_SECONDS = 3  # NOTE: Refresh the learning-curve plot every N seconds.

RANDOM_SEED = 70563
SMOKE_TEST = False  # NOTE: If True, only the first rolling window is trained.
SKIP_LAST_WINDOW = True  # NOTE: The last window is reserved for the final evaluation.
FEATURE_SET = "raw+rank+daystd"  # NOTE: A FEATURE_SETS key; pick it with test_single-day-feature-sets.py.

CORR_BAR = 0.02  # NOTE: Protocol settings as in train_baseline (run 5); share_head()
LOOKBACK = 240  # NOTE: copies the ones its helpers use into train_baseline.
GAP = 10
TEST_DAYS = 120
VALIDATION_FOLD = 5
PURGE = GAP
SELECT_FEATURES = False # NOTE: If True, filter features based on correlation. False = use all features.
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 200

SHARED = (  # NOTE: Head settings that train_baseline's helpers read from their module.
    "RUN_DIR", "MODEL_DIR", "PREDICTIONS_DIR", "CURVES_DIR", "CURVE_FLUSH_EVERY",
    "CORR_BAR", "VALIDATION_FOLD", "PURGE", "SELECT_FEATURES", "MAX_ROUNDS",
    "EARLY_STOPPING_ROUNDS",
)

FEATURE_SETS = {  # NOTE: Feature set -> its column groups (see build_features).
    "raw": ("raw",),
    "rank": ("rank",),
    "z": ("z",),
    "raw+rank": ("raw", "rank"),
    "rank+daystats": ("rank", "daymean", "daystd"),
    "raw+daystats": ("raw", "daymean", "daystd"),
    "raw+daymean": ("raw", "daymean"),
    "raw+daystd": ("raw", "daystd"),
    "raw+rank+daymean": ("raw", "rank", "daymean"),
    "raw+rank+daystd": ("raw", "rank", "daystd"),
    "raw+rank+daystats": ("raw", "rank", "daymean", "daystd"),
}


def build_features(frame: pl.DataFrame, kind: str) -> tuple[pl.DataFrame, list[str]]:
    """
    Add feature set `kind` to whole days of data; return the model input names.

    Column groups, each over the 48 raw features `featureN`:
    - raw:     featureN as it is.
    - rank:    featureN_rank, its percentile among that day's stocks, in (0, 1].
    - z:       featureN_z = (featureN - day mean) / day SD.
    - daymean: featureN_daymean, the day's mean (same for every stock that day).
    - daystd:  featureN_daystd, the day's SD.
    Per-day statistics use only that day's features (no labels, no other dates), so
    nothing looks ahead. Non-finite raw values are left out of them and give null,
    which finite_rows() drops, as it drops the raw value.
    """
    if kind not in FEATURE_SETS:
        msg = f"Unknown feature set {kind!r}; choose from {list(FEATURE_SETS)}"
        raise ValueError(msg)
    raw = [c for c in frame.columns if c.startswith("feature") and "_" not in c]
    groups = FEATURE_SETS[kind]
    names = list(raw) if "raw" in groups else []
    exprs = []
    for f in raw:
        x = pl.when(pl.col(f).is_finite()).then(pl.col(f))  # NOTE: Null if non-finite.
        derived = {
            "rank": x.rank().over("Date") / x.count().over("Date"),
            "z": (x - x.mean().over("Date")) / x.std().over("Date"),
            "daymean": x.mean().over("Date"),
            "daystd": x.std().over("Date"),
        }
        for group in groups:
            if group != "raw":
                exprs.append(derived[group].cast(pl.Float32).alias(f"{f}_{group}"))
                names.append(f"{f}_{group}")
    names.sort(key=lambda c: groups.index(c.split("_")[1]) if "_" in c else 0)
    return (frame.with_columns(exprs) if exprs else frame), names


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

    The only differences: per window, the history and test rows get
    build_features(FEATURE_SET) before finite_rows() (per-day features need whole
    days), and the results record the feature set.
    """
    data = pl.read_parquet(DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
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

        history, features = build_features(
            data.filter(pl.col("Date").is_in(history_dates)), FEATURE_SET
        )
        history = tb.finite_rows(history, [*features, "label"])
        params = tb.lgbm_params(RANDOM_SEED + i)
        window_tag = (f"train_{dates[begin]}_{dates[i - GAP - 1]}_"
                      f"test_{dates[i]}_{dates[end - 1]}")
        cv = tb.purged_cv(history, history_dates, features, params, window_tag)
        best_iteration = cv.best_iteration

        selected = tb.select_features(history, features)
        test_rows, _ = build_features(
            data.filter(pl.col("Date").is_in(test_dates)), FEATURE_SET
        )
        test_df = tb.finite_rows(test_rows, [*selected, "label"])

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

        del test_rows, test_df, test_x, model, scored, prediction, real, ic, valid_ic
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
