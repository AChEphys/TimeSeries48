#!/usr/bin/env python3
"""Train rolling LightGBM models from an existing data.parquet."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as stats
import lightgbm as lgb
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data.parquet"
RESULT_FILE = ROOT / "run0_results_lgbm.json"
MODEL_DIR = ROOT / "models"

CORR_BAR = 0.02
LOOKBACK = 240
GAP = 10
TEST_DAYS = 120


def finite_rows(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Remove rows with null, NaN, or infinite model inputs."""
    return df.filter(pl.all_horizontal([pl.col(c).is_finite() for c in columns]))


def daily_ic(frame: pl.DataFrame) -> np.ndarray:
    """Calculate one Pearson IC for each test date."""
    values = []
    for day in frame.partition_by("Date", maintain_order=True):
        prediction = day["Prediction"].to_numpy()
        real = day["label"].to_numpy()
        values.append(
            np.corrcoef(prediction, real)[0, 1] if len(day) > 1 else np.nan
        )
    return np.asarray(values, dtype=np.float64)


def train() -> list[dict]:
    data = pl.read_parquet(DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    features = [column for column in data.columns if column.startswith("feature")]

    if not features:
        raise ValueError(f"No feature columns found in {DATA_FILE}")

    MODEL_DIR.mkdir(exist_ok=True)
    results: list[dict] = []

    for i in range(LOOKBACK, len(dates), TEST_DAYS):
        begin = i - LOOKBACK
        end = min(i + TEST_DAYS, len(dates))
        history_dates = dates[begin : i - GAP]
        test_dates = dates[i:end]

        history = finite_rows(
            data.filter(pl.col("Date").is_in(history_dates)),
            features + ["label"],
        )
        correlations = history.select(
            [pl.corr(feature, "label").alias(feature) for feature in features]
        ).row(0, named=True)
        selected = [
            feature
            for feature in features
            if correlations[feature] is not None
            and abs(correlations[feature]) > CORR_BAR
        ]
        if not selected:
            raise RuntimeError(f"No feature passed the threshold in window {i}")

        # Match hf2DayTrain.py: every third historical date is validation.
        train_dates = [date for j, date in enumerate(history_dates) if j % 3]
        validation_dates = [
            date for j, date in enumerate(history_dates) if not j % 3
        ]
        train_df = history.filter(pl.col("Date").is_in(train_dates))
        validation_df = history.filter(pl.col("Date").is_in(validation_dates))
        test_df = finite_rows(
            data.filter(pl.col("Date").is_in(test_dates)), selected + ["label"]
        )

        rng = np.random.default_rng(int(time.time()))
        train_order = rng.permutation(len(train_df))
        validation_order = rng.permutation(len(validation_df))
        train_x = train_df.select(selected).to_numpy()[train_order]
        train_y = train_df["label"].to_numpy()[train_order]
        validation_x = validation_df.select(selected).to_numpy()[validation_order]
        validation_y = validation_df["label"].to_numpy()[validation_order]

        params = {
            "objective": "mse",
            "metric": "rmse",
            "device": "cpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "num_threads": os.cpu_count(),
            "boosting_type": "gbdt",
            "data_sample_strategy": "goss",
            "num_leaves": 20,
            "max_depth": 7,
            "subsample": 0.8,
            "reg_alpha": 0.98,
            "reg_lambda": 0.98,
            "min_child_weight": 0.7544,
            "seed": int(time.time()),
            "learning_rate": 1e-2,
            "verbosity": 0,
        }

        def r2_metric(
            prediction: np.ndarray, dataset: lgb.Dataset
        ) -> tuple[str, float, bool]:
            r2 = r2_score(dataset.get_label(), prediction)
            return "r2", float(r2), True

        train_matrix = lgb.Dataset(
            train_x, label=train_y, feature_name=selected, free_raw_data=False
        )
        validation_matrix = lgb.Dataset(
            validation_x,
            label=validation_y,
            feature_name=selected,
            reference=train_matrix,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            train_matrix,
            num_boost_round=2000,
            valid_sets=[train_matrix, validation_matrix],
            valid_names=["train", "val"],
            feval=r2_metric,
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
        )

        model_name = (
            f"lgbm_train_{dates[begin]}_{dates[i-GAP]}"
            f"_test_{dates[i]}_{dates[end-1]}"
        )
        model.save_model(MODEL_DIR / f"{model_name}.txt")
        (MODEL_DIR / f"{model_name}.features.json").write_text(
            json.dumps({"tot": len(selected), "features": selected}, indent=2),
            encoding="utf-8",
        )

        test_x = test_df.select(selected).to_numpy()
        prediction = model.predict(test_x, num_iteration=model.best_iteration)
        scored = test_df.select("Date", "Code", "label").with_columns(
            pl.Series("Prediction", prediction)
        )
        ic = daily_ic(scored)
        valid_ic = ic[np.isfinite(ic)]
        real = scored["label"].to_numpy()

        result = {
            "train_start": str(dates[begin]),
            "train_end": str(dates[i-GAP]),
            "test_start": str(dates[i]),
            "test_end": str(dates[end-1]),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "features": len(selected),
            "best_iteration": int(model.best_iteration),
            "r2": float(r2_score(real, prediction)),
            "kt": float(stats.kendalltau(prediction, real).statistic),
            "ic": float(valid_ic.mean()),
            "ir": float(
                valid_ic.mean() / valid_ic.std() * np.sqrt(len(test_dates))
            ),
        }
        results.append(result)
        RESULT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))

        del train_df, validation_df, test_df, train_x, train_y
        del validation_x, validation_y, test_x, model, train_matrix
        del validation_matrix, scored, prediction, real, ic
        gc.collect()

    return results


def main() -> None:
    started = time.time()
    results = train()
    print(
        f"completed {len(results)} windows "
        f"in {(time.time() - started) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
