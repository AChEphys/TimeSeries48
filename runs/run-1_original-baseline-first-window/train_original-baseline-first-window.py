#!/usr/bin/env python3
"""Train rolling LightGBM models from an existing data.parquet."""

from __future__ import annotations

import contextlib  # BY: Claude
import gc
import json
import os
import shutil  # BY: Claude
import subprocess  # BY: Claude
import textwrap  # BY: Claude
import time
from pathlib import Path
from typing import TYPE_CHECKING  # BY: Claude

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import r2_score

if TYPE_CHECKING:  # BY: Claude
    from collections.abc import Callable  # BY: Claude

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data.parquet"
RUN_DIR = ROOT / "runs/run-1_original-baseline-first-window"
RESULT_FILE = RUN_DIR / "results_lgbm.json"
MODEL_DIR = RUN_DIR / "models"
ARXIV_FILE = RUN_DIR / "train_original-baseline-first-window.py"  # NOTE: Archive copy of this script, for reproducibility.
CURVE_FLUSH_EVERY = 50  # BY: Claude  # NOTE: Write learning curves to disk every N iterations.

RANDOM_SEED = 70563
SMOKE_TEST = True       # NOTE: If True, only the first rolling window is trained and evaluated. Useful for debugging.
SKIP_LAST_WINDOW = True # NOTE: If True, skip the last rolling window. The last window is reserved for future evaluation on unseen data.

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
        values.append(np.corrcoef(prediction, real)[0, 1] if len(day) > 1 else np.nan)
    return np.asarray(values, dtype=np.float64)


def reset_peak_memory() -> None:  # BY: Claude
    """Reset this process's peak resident memory to its current value (Linux 4.0+)."""  # BY: Claude
    with contextlib.suppress(OSError):  # BY: Claude
        Path("/proc/self/clear_refs").write_text("5")  # BY: Claude


def peak_memory_gb() -> float:  # BY: Claude
    """Peak resident memory (VmHWM) since the last reset, in GiB; NaN if unavailable."""  # BY: Claude
    with contextlib.suppress(OSError):  # BY: Claude
        for line in Path("/proc/self/status").read_text().splitlines():  # BY: Claude
            if line.startswith("VmHWM:"):  # BY: Claude
                return int(line.split()[1]) / 1024**2  # BY: Claude  # NOTE: kB -> GiB.
    return float("nan")  # BY: Claude


def write_json_atomic(path: Path, payload: object) -> None:  # BY: Claude
    """Write JSON via a temp file, so a reader never sees a half-written file."""  # BY: Claude
    tmp = path.with_suffix(path.suffix + ".tmp")  # BY: Claude
    tmp.write_text(json.dumps(payload), encoding="utf-8")  # BY: Claude
    tmp.replace(path)  # BY: Claude


def record_curves(  # BY: Claude
    history: dict[str, dict[str, list[float]]], path: Path, every: int  # BY: Claude
) -> Callable[[lgb.callback.CallbackEnv], None]:  # BY: Claude
    """Build a callback that records each iteration's metrics and flushes to JSON."""  # BY: Claude
    def _callback(env: lgb.callback.CallbackEnv) -> None:  # BY: Claude
        for item in env.evaluation_result_list or []:  # BY: Claude
            history.setdefault(
                item.dataset_name, {}
                ).setdefault(
                item.metric_name, []
                ).append(item.metric_value)  # BY: Claude
        if (env.iteration + 1) % every == 0:  # BY: Claude
            write_json_atomic(path, {"best_iteration": None, "curves": history})  # BY: Claude  # NOTE: None = still training.
    return _callback  # BY: Claude


CURVE_CELL = textwrap.dedent(  # BY: Claude  # NOTE: Lines inside this string are the notebook cell's code, all BY: Claude.
'''\
    import json
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    RUN_DIR = Path("__RUN_DIR__")
    TRAIN_PID = __TRAIN_PID__      # Process ID of the training script.
    REFRESH_SECONDS = __REFRESH__


    def training_alive() -> bool:
        """True while the training process still exists."""
        return Path(f"/proc/{TRAIN_PID}").exists()


    def plot_curves() -> None:
        """Plot train/val RMSE and R² for every window's curves file."""
        files = sorted(RUN_DIR.glob("curves_*.json"))
        if not files:
            print("No curves yet: training just started...")
        for f in files:
            try:
                d = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            best, c = d["best_iteration"], d["curves"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
            axes[0].plot(c["train"]["rmse"])
            axes[0].set_title("train RMSE")
            axes[1].plot(c["val"]["rmse"])
            axes[1].set_title("val RMSE")
            axes[2].plot(c["train"]["r2"], label="train")
            axes[2].plot(c["val"]["r2"], label="val")
            axes[2].set_title("R²")
            axes[2].legend()
            for ax in axes:
                ax.set_xlabel("iteration")
                if best is not None:
                    ax.axvline(best, ls="--", c="gray")  # Best iteration.
            status = f"best_iteration={best}" if best is not None else "training..."
            fig.suptitle(f"{f.stem}  ({len(c['val']['rmse'])} iterations, {status})")
            plt.show()


    # Refresh until the training process exits, then draw a final plot and stop.
    # Interrupt the cell (stop button) to end it early; training is unaffected.
    while True:
        alive = training_alive()
        clear_output(wait=True)
        plot_curves()
        stamp = time.strftime("%H:%M:%S")
        if not alive:
            print(f"{stamp}: training process has exited. Final plot.")
            break
        print(f"{stamp}: training running (PID {TRAIN_PID})...")
        time.sleep(REFRESH_SECONDS)
''')  # BY: Claude  # NOTE: Placeholders __X__ are filled in by launch_curve_notebook().


def launch_curve_notebook(refresh_seconds: int = 60) -> Path:  # BY: Claude
    """Write a live learning-curve notebook into RUN_DIR and open it in VS Code."""  # BY: Claude
    source = (  # BY: Claude
        CURVE_CELL.replace("__RUN_DIR__", str(RUN_DIR))  # BY: Claude
        .replace("__TRAIN_PID__", str(os.getpid()))  # BY: Claude
        .replace("__REFRESH__", str(refresh_seconds))  # BY: Claude
    )  # BY: Claude
    notebook = {  # BY: Claude
        "cells": [  # BY: Claude
            {  # BY: Claude
                "id": "curves-intro",  # BY: Claude
                "cell_type": "markdown",  # BY: Claude
                "metadata": {},  # BY: Claude
                "source": [
                    f"# Learning curves: `{RUN_DIR.name}`\n",
                    "Run the cell below once; it refreshes itself until training ends."
                    ],  # BY: Claude
            },  # BY: Claude
            {  # BY: Claude
                "id": "curves-live",  # BY: Claude
                "cell_type": "code",  # BY: Claude
                "execution_count": None,  # BY: Claude
                "metadata": {},  # BY: Claude
                "outputs": [],  # BY: Claude
                "source": source.splitlines(keepends=True),  # BY: Claude
            },  # BY: Claude
        ],  # BY: Claude
        "metadata": {  # BY: Claude
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
                },  # BY: Claude
            "language_info": {"name": "python"},  # BY: Claude
        },  # BY: Claude
        "nbformat": 4,  # BY: Claude
        "nbformat_minor": 5,  # BY: Claude
    }  # BY: Claude
    path = RUN_DIR / "curves.ipynb"  # BY: Claude
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")  # BY: Claude
    code_cli = shutil.which("code")  # BY: Claude  # NOTE: VS Code's CLI; present in its integrated terminal.
    if code_cli is None:  # BY: Claude
        print(f"Open {path} in Jupyter to watch the learning curves.")  # BY: Claude  # noqa: T201
        return path  # BY: Claude
    with contextlib.suppress(OSError, subprocess.SubprocessError):  # BY: Claude
        subprocess.run([code_cli, str(path)], check=False, timeout=30)  # BY: Claude  # noqa: S603  # NOTE: Opens a tab; returns immediately.
    return path  # BY: Claude


def train() -> list[dict]:
    """
    Train, evaluate, and save GBDT models on rolling windows from existing parquet data.

    For each rolling window, the function:
    1. Divides the window into training, validation, and test sets.
    2. Validation dates are every third date in the training window.
    3. A purge gap is left out between training and test sets to avoid leakage.
    4. Features are selected based on correlation with the label on the training AND validation sets.
    5. A GBDT model is trained to find: selected features of a SINGLE stock on a SINGLE day -> label of that stock on that day.
    6. The model along with its selected features are saved to disk.
    7. The model is evaluated on the test set, and metrics are recorded. Results are saved to disk.
    """  # noqa: E501
    data = pl.read_parquet(DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    features = [column for column in data.columns if column.startswith("feature")]

    if not features:
        msg = f"No feature columns found in {DATA_FILE}"
        raise ValueError(msg)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # BY: Claude  # NOTE: parents=True, since runs/<name>/ doesn't exist yet.
    results: list[dict] = []

    for i in range(LOOKBACK, len(dates), TEST_DAYS):
        runtime_start = time.time()
        reset_peak_memory()  # BY: Claude
        begin = i - LOOKBACK                    # NOTE: Start of training window (inclusive), 0-based index.
        end = min(i + TEST_DAYS, len(dates))    # NOTE: Exclusive, 0-based index.
        history_dates = dates[begin : i - GAP]  # NOTE: LOOKBACK - GAP days. i - GAP is exclusive.
        test_dates = dates[i:end]               # NOTE: TEST_DAYS days (or fewer), end exclusive.

        if SKIP_LAST_WINDOW and end == len(dates):
            print(f"Skipping the last window from {dates[begin]} to {dates[end - 1]}.")  # noqa: T201
            break

        history = finite_rows(
            data.filter(pl.col("Date").is_in(history_dates)),
            [*features, "label"],
        )   # NOTE: Select data within history_dates whose features and label are finite.
        correlations = history.select(
            [pl.corr(feature, "label").alias(feature) for feature in features]
        ).row(0, named=True)
        selected = [
            feature
            for feature in features
            if correlations[feature] is not None
            and abs(correlations[feature]) > CORR_BAR
        ]  # NOTE: Only features of sufficient correlation with the label are considered.
        if not selected:
            msg = f"No feature passed the threshold in window {i}"
            raise RuntimeError(msg)

        # TUTOR: Match hf2DayTrain.py: every third historical date is validation.
        # ISSUE: Interleaving 1-day validation may result in leakage and overfitting.
        train_dates = [
            date for j, date in enumerate(history_dates) if j % 3
        ]  # NOTE: Indices 1,2,4,5,7,8,...
        validation_dates = [
            date for j, date in enumerate(history_dates) if not j % 3
        ]  # NOTE: Indices 0,3,6,9,...
        train_df = history.filter(pl.col("Date").is_in(train_dates))
        validation_df = history.filter(pl.col("Date").is_in(validation_dates))
        test_df = finite_rows(
            data.filter(pl.col("Date").is_in(test_dates)), [*selected, "label"]
        )

        rng = np.random.default_rng(RANDOM_SEED)   # ISSUE(ADDRESSED): Seeding with time may hinder reproducibility.
        train_order = rng.permutation(len(train_df))    # NOTE: Shuffle all rows. Seems to be a common practice in ML training.
        validation_order = rng.permutation(len(validation_df))
        train_x = train_df.select(selected).to_numpy()[train_order]
        train_y = train_df["label"].to_numpy()[train_order]
        validation_x = validation_df.select(selected).to_numpy()[validation_order]
        validation_y = validation_df["label"].to_numpy()[validation_order]

        params = {
            "objective": "mse",             # NOTE: Training objective.
            "metric": "rmse",               # NOTE: Validation metric.
            "device": "cpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "num_threads": os.cpu_count(),  # NOTE: Recommended by LightGBM.
            "boosting_type": "gbdt",        # NOTE: As opposed to random forest, etc.
            "data_sample_strategy": "goss", # NOTE: Gradient-based One-Side Sampling (GOSS) prioritizes samples with larger gradients. Small-gradient samples are not used for optimization.
            "num_leaves": 20,               # NOTE: Max num leaves. Affects complexity and overfitting.
            "max_depth": 7,                 # NOTE: 2^max_depth >= num_leaves. Avoids lop-sided trees.
            "subsample": 0.8,               # NOTE: Fraction of data used for each iteration. ISSUE: No use here with GOSS.
            "reg_alpha": 0.98,              # NOTE: L1 penalty. Avoids overfitting. Favors leaves with large residuals. Has the same dimension as the label.
            "reg_lambda": 0.98,             # NOTE: L2 penalty. Avoids overfitting. Renormalizes the residuals to smaller values. Dimensionless (~extra sample counts). ISSUE: 0.98 probably too small to mean anything.
            "min_child_weight": 0.7544,     # NOTE: Min sum of hessian in leaf. ISSUE: Probably useless with min_data_in_leaf=20.
            # "min_data_in_leaf": 20,       # NOTE: Larger reduces overfitting. 20 is the default.  # noqa: E501, ERA001
            "seed": RANDOM_SEED + i,        # ISSUE(ADDRESSED): Could be fixed for reproducibility.
            "learning_rate": 1e-2,          # NOTE: Small for slower, smoother descent.
            "verbosity": 1,                 # NOTE: 0: error/warning, 1: info, 2: debug.
        }

        def r2_metric(
            prediction: np.ndarray, dataset: lgb.Dataset
        ) -> tuple[str, float, bool]:
            r2 = r2_score(np.asarray(dataset.get_label()), prediction)
            return "r2", float(r2), True

        train_matrix = lgb.Dataset(
            train_x, label=train_y, feature_name=selected, free_raw_data=False
        )   # NOTE: train_x contains only selected features, no information about time or stock code.
            # NOTE: The model finds the relationship between selected features of each stock on a single day and the label.
            # NOTE: It doesn't care which stock or which day.
            # NOTE: The stock's characteristics are already baked into the features.
            # NOTE: The features-to-label relationship is assumed to be stable over time within the rolling window.
        validation_matrix = lgb.Dataset(
            validation_x,
            label=validation_y,
            feature_name=selected,
            reference=train_matrix,
            free_raw_data=False,
        )
        window_tag = (  # BY: Claude
            f"train_{dates[begin]}_{dates[i - GAP - 1]}_"
            f"test_{dates[i]}_{dates[end - 1]}"  # BY: Claude
        )  # BY: Claude
        curves: dict[str, dict[str, list[float]]] = {}  # BY: Claude
        curves_file = RUN_DIR / f"curves_{window_tag}.json"  # BY: Claude
        model = lgb.train(
            params,
            train_matrix,
            num_boost_round=2000,
            valid_sets=[train_matrix, validation_matrix],
            valid_names=["train", "val"],
            feval=r2_metric,    # ISSUE: Probably redundant with metric="rmse".
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(100),
                record_curves(curves, curves_file, CURVE_FLUSH_EVERY),  # BY: Claude
            ],
            # NOTE: Each callback is called after each iteration.
            # NOTE: Early stopping if not improving for 200 iterations.
            # NOTE: Log evaluation every 100 iterations.
            # NOTE: Record evaluation metrics to curves and save to curves_file every CURVE_FLUSH_EVERY iterations.
            # NOTE: Plot the curves in Jupyter.
        )   # NOTE: This function blocks the main thread until training is complete.
        write_json_atomic(  # BY: Claude
            curves_file, {"best_iteration": int(model.best_iteration), "curves": curves}  # BY: Claude
        )  # BY: Claude  # NOTE: Final flush, now with best_iteration.

        model_name = "lgbm_" + window_tag
        model.save_model(MODEL_DIR / f"{model_name}.txt")
        (MODEL_DIR / f"{model_name}.features.json").write_text(
            json.dumps({"tot": len(selected), "features": selected}, indent=2),
            encoding="utf-8",
        )

        test_x = test_df.select(selected).to_numpy()
        prediction = np.asarray(
            model.predict(test_x, num_iteration=model.best_iteration)
        )
        scored = test_df.select("Date", "Code", "label").with_columns(
            pl.Series("Prediction", prediction)
        )
        scored.write_parquet(RUN_DIR / f"predictions_{window_tag}.parquet")  # BY: Claude
        ic = daily_ic(scored)
        valid_ic = ic[np.isfinite(ic)]
        real = scored["label"].to_numpy()

        result = {
            "train_start": str(dates[begin]),
            "train_end": str(dates[i - GAP - 1]),           # ISSUE(ADDRESSED): dates[i - GAP] not included. Better to use dates[i - GAP - 1] for clarity.
            "test_start": str(dates[i]),
            "test_end": str(dates[end - 1]),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "features": len(selected),
            "best_iteration": int(model.best_iteration),
            "r2": float(r2_score(real, prediction)),    # NOTE: R^2 on test set only.
            "kt": float(stats.kendalltau(prediction, real).statistic),
            # NOTE: Measures pairwise ordinal association. 1: perfect agreement, -1: perfect disagreement.
            # ISSUE: Kendall's tau computed on all test rows, not daily. Maybe compute daily and average for consistency with IC.
            "ic": float(valid_ic.mean()),               # NOTE: Mean daily IC on test set only.
            "ir": float(valid_ic.mean() / valid_ic.std() * np.sqrt(len(test_dates))),
            # NOTE: Prediction ability devided by volatility, scaled by the breadth where the ability is tested.
            # NOTE: Higher IR means consistently better prediction tested across sufficiently long time.
            # ISSUE: valid_ic excludes ill values, but IR is scaled by the total number of test dates.
            "runtime": time.time() - runtime_start,
            "peak_memory_GB": peak_memory_gb(),  # BY: Claude  # NOTE: Whole process, incl. the loaded dataset.
        }
        results.append(result)
        RESULT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))  # noqa: T201

        del train_df, validation_df, test_df, train_x, train_y
        del validation_x, validation_y, test_x, model, train_matrix
        del validation_matrix, scored, prediction, real, ic, valid_ic
        del history, correlations, curves
        gc.collect()

        if SMOKE_TEST:
            print("Smoke test: only the first rolling window is trained and evaluated.")  # noqa: T201
            break

    return results


def main() -> None:
    """Entry point. Timed."""
    started = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)  # BY: Claude
    script = Path(__file__).resolve()  # BY: Claude
    shutil.copy2(script, ARXIV_FILE)  # BY: Claude  # NOTE: Archive copy only; run the one under ROOT.
    launch_curve_notebook()  # BY: Claude
    results = train()
    print(  # noqa: T201
        f"Completed {len(results)} windows "
        f"in {(time.time() - started) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
