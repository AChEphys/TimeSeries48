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
RUN_DIR = ROOT / "runs/run-3_baseline-chrono-val-plus-retrain"
RESULT_FILE = RUN_DIR / "results_lgbm.json"
MODEL_DIR = RUN_DIR / "models"
PREDICTIONS_DIR = RUN_DIR / "predictions"
CURVES_DIR = RUN_DIR / "curves"
ARXIV_FILE = RUN_DIR / "train_baseline-chrono-val-plus-retrain.py"  # NOTE: Archive copy of this script, for reproducibility.
CURVE_FLUSH_EVERY = 50      # BY: Claude  # NOTE: Write learning curves to disk every N iterations.
PLOT_REFRESH_SECONDS = 5    # NOTE: Refresh the learning-curve plot every N seconds.

RANDOM_SEED = 70563
SMOKE_TEST = False      # NOTE: If True, only the first rolling window is trained and evaluated. Useful for debugging.
SKIP_LAST_WINDOW = True # NOTE: If True, skip the last rolling window. The last window is reserved for future evaluation on unseen data.

CORR_BAR = 0.02
LOOKBACK = 240
GAP = 10
TEST_DAYS = 120
VALIDATION_FRACTION = 1 / 3  # BY: Claude  # NOTE: Newest third of the history dates validates (chronological split, I-1).
PURGE = GAP  # BY: Claude  # NOTE: History dates dropped between train and validation; labels look ~6 dates ahead.


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
    history: dict[str, dict[str, list[float]]],  # BY: Claude
    path: Path,  # BY: Claude
    every: int,  # BY: Claude
    status: dict[str, object] | None = None,  # BY: Claude  # NOTE: Extra fields written with each flush.
) -> Callable[[lgb.callback.CallbackEnv], None]:  # BY: Claude
    """Build a callback that records each iteration's metrics and flushes to JSON."""  # BY: Claude
    fields = status or {"best_iteration": None}  # BY: Claude  # NOTE: Default: early-stopping fit, None = still training.

    def _callback(env: lgb.callback.CallbackEnv) -> None:  # BY: Claude
        for item in env.evaluation_result_list or []:  # BY: Claude
            history.setdefault(
                item.dataset_name, {}
                ).setdefault(
                item.metric_name, []
                ).append(item.metric_value)  # BY: Claude
        if (env.iteration + 1) % every == 0:  # BY: Claude
            write_json_atomic(path, {**fields, "curves": history})  # BY: Claude
    return _callback  # BY: Claude


CURVE_CELL = textwrap.dedent(  # BY: Claude  # NOTE: Lines inside this string are the notebook cell's code, all BY: Claude.
'''\
    import json
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    from IPython.display import Image, clear_output, display

    CURVES_DIR = Path("__CURVES_DIR__")
    PLOTS_DIR = Path("__PLOTS_DIR__")   # Finished windows' plots are saved here as PNG.
    TRAIN_PID = __TRAIN_PID__      # Process ID of the training script.
    REFRESH_SECONDS = __REFRESH__
    COLORS = {"train": "C0", "val": "C1", "retrain": "C2"}


    def training_alive() -> bool:
        """True while the training process still exists."""
        return Path(f"/proc/{TRAIN_PID}").exists()


    def read_json(path: Path) -> dict | None:
        """Read a curves file; None if it doesn't exist yet."""
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None


    def window_title(stem: str) -> str:
        """'curves_train_A_B_test_C_D' -> 'train A → B  ·  test C → D'."""
        p = stem.removeprefix("curves_").split("_")
        if len(p) == 6:
            return f"train {p[1]} → {p[2]}     ·     test {p[4]} → {p[5]}"
        return stem


    def status_line(d: dict, r: dict | None) -> str:
        """Progress of the early-stopping fit and of the retrain."""
        n, best = len(d["curves"]["val"]["rmse"]), d["best_iteration"]
        if best is None:
            return f"early-stopping fit: {n} iterations so far..."
        fit = f"early-stopping fit: {n} iterations, best {best}"
        if r is None:
            return f"{fit}     ·     retrain: starting..."
        if r["done"]:
            return f"{fit}     ·     retrain: done ({r['rounds']} trees)"
        k = len(r["curves"].get("retrain", {}).get("rmse", []))
        return f"{fit}     ·     retrain: {k} / {r['rounds']} iterations..."


    def draw(f: Path, d: dict, r: dict | None) -> plt.Figure:
        """One window: early-stopping fit (train/val), retrain overlaid."""
        best, c = d["best_iteration"], d["curves"]
        rc = (r or {}).get("curves", {}).get("retrain", {})
        n = len(c["val"]["rmse"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
        ax_rmse, ax_val, ax_r2 = axes
        ax_rmse.plot(c["train"]["rmse"], c=COLORS["train"], label="train")
        if rc.get("rmse"):
            ax_rmse.plot(rc["rmse"], c=COLORS["retrain"], label="retrain (all history)")
        ax_rmse.set_title("Training RMSE")
        ax_val.plot(c["val"]["rmse"], c=COLORS["val"], label="val")
        ax_val.set_title("Validation RMSE")
        ax_r2.plot(c["train"]["r2"], c=COLORS["train"], label="train")
        ax_r2.plot(c["val"]["r2"], c=COLORS["val"], label="val")
        if rc.get("r2"):
            ax_r2.plot(rc["r2"], c=COLORS["retrain"], label="retrain (all history)")
        ax_r2.set_title("R²")
        for ax in axes:
            if best is not None:
                label = f"best iteration ({best})"
                ax.axvline(best, ls="--", lw=1, c="gray", label=label)
            ax.set_xlim(0, max(n - 1, 1))  # Set by the early-stopping fit only.
            ax.set_xlabel("iteration")
            ax.ticklabel_format(axis="y", useOffset=False)  # No offset text on top.
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, frameon=False)
        fig.suptitle(f"{window_title(f.stem)}\\n{status_line(d, r)}", fontsize=11)
        return fig


    def plot_curves() -> None:
        """Show every window; finished ones (retrain done) as their saved PNG."""
        files = sorted(CURVES_DIR.glob("curves_*.json"))
        if not files:
            print("No curves yet: training just started...")
        for f in files:
            d = read_json(f)
            if d is None:
                continue
            rf = f.with_name(f.name.replace("curves_", "retrain_", 1))
            r = read_json(rf)
            png = PLOTS_DIR / f"{f.stem}.png"
            done = r is not None and r["done"]
            # A finished window is drawn and saved once, then shown as a static PNG,
            # unless its curves are newer than the PNG (e.g. after a rerun).
            if done and png.exists() and png.stat().st_mtime >= max(
                f.stat().st_mtime, rf.stat().st_mtime
            ):
                display(Image(filename=str(png)))
                continue
            fig = draw(f, d, r)
            if done:
                PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                fig.savefig(png, dpi=120)  # Before show().
            plt.show()
            plt.close(fig)


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
        CURVE_CELL.replace("__CURVES_DIR__", str(CURVES_DIR))  # BY: Claude
        .replace("__PLOTS_DIR__", str(RUN_DIR / "plots"))  # BY: Claude
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
                    (  # BY: Claude
                        "Run the cell below once; it refreshes itself until training "  # BY: Claude
                        "ends. A window is final (saved to `plots/`, then shown as a "  # BY: Claude
                        "static PNG) once its retrain is done."  # BY: Claude
                    ),  # BY: Claude
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
    2. Validation dates are the newest third of the history dates (chronological).
    3. Purge gaps are left out between training and validation, and between history
       and test, to avoid leakage through overlapping forward labels.
    4. Features are selected based on correlation with the label on the training dates only.
    5. A GBDT model is trained to find: selected features of a SINGLE stock on a SINGLE day -> label of that stock on that day.
       Early stopping on the validation set picks the number of trees.
    6. The model is retrained on ALL history dates with that number of trees.
    7. The model along with its selected features are saved to disk.
    8. The model is evaluated on the test set, and metrics are recorded. Results are saved to disk.
    """  # noqa: E501  # BY: Claude  # NOTE: Items 2-6 rewritten for the chronological split (run 3).
    data = pl.read_parquet(DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    features = [column for column in data.columns if column.startswith("feature")]

    if not features:
        msg = f"No feature columns found in {DATA_FILE}"
        raise ValueError(msg)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # BY: Claude  # NOTE: parents=True, since runs/<name>/ doesn't exist yet.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
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

        # ISSUE(ADDRESSED): Interleaved 1-day validation leaked through overlapping labels (I-1).  # BY: Claude
        n_validation = round(len(history_dates) * VALIDATION_FRACTION)  # BY: Claude
        n_train = len(history_dates) - n_validation - PURGE  # BY: Claude
        train_dates = history_dates[:n_train]  # BY: Claude  # NOTE: Oldest dates.
        validation_dates = history_dates[n_train + PURGE :]  # BY: Claude  # NOTE: Newest dates, after the purge gap.
        train_df = history.filter(pl.col("Date").is_in(train_dates))
        validation_df = history.filter(pl.col("Date").is_in(validation_dates))

        correlations = train_df.select(  # BY: Claude  # NOTE: Training dates only, so validation labels stay unseen (I-1).
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
        curves_file = CURVES_DIR / f"curves_{window_tag}.json"  # BY: Claude
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

        best_iteration = int(model.best_iteration)  # BY: Claude
        full_order = rng.permutation(len(history))  # BY: Claude
        full_matrix = lgb.Dataset(  # BY: Claude
            history.select(selected).to_numpy()[full_order],  # BY: Claude
            label=history["label"].to_numpy()[full_order],  # BY: Claude
            feature_name=selected,  # BY: Claude
        )  # BY: Claude  # NOTE: All history dates, incl. validation and purge dates.
        retrain_curves: dict[str, dict[str, list[float]]] = {}  # BY: Claude
        retrain_file = CURVES_DIR / f"retrain_{window_tag}.json"  # BY: Claude
        retrain_status: dict[str, object] = {"done": False, "rounds": best_iteration}  # BY: Claude
        write_json_atomic(retrain_file, {**retrain_status, "curves": {}})  # BY: Claude  # NOTE: Signals "retrain started" to the curve notebook.
        model = lgb.train(  # BY: Claude
            params,  # BY: Claude
            full_matrix,  # BY: Claude
            num_boost_round=best_iteration,  # BY: Claude
            valid_sets=[full_matrix],  # BY: Claude  # NOTE: Training data only; there is no validation set in the retrain.
            valid_names=["retrain"],  # BY: Claude
            feval=r2_metric,  # BY: Claude
            callbacks=[  # BY: Claude
                lgb.log_evaluation(100),  # BY: Claude
                record_curves(  # BY: Claude
                    retrain_curves, retrain_file, CURVE_FLUSH_EVERY, retrain_status  # BY: Claude
                ),  # BY: Claude
            ],  # BY: Claude
        )  # BY: Claude  # NOTE: Retrain with the chosen number of trees, so the newest dates are learned too. No early stopping.
        write_json_atomic(  # BY: Claude
            retrain_file, {**retrain_status, "done": True, "curves": retrain_curves}  # BY: Claude
        )  # BY: Claude  # NOTE: Final flush; the notebook then saves this window's PNG.

        model_name = "lgbm_" + window_tag
        model.save_model(MODEL_DIR / f"{model_name}.txt")
        (MODEL_DIR / f"{model_name}.features.json").write_text(
            json.dumps({"tot": len(selected), "features": selected}, indent=2),
            encoding="utf-8",
        )

        test_x = test_df.select(selected).to_numpy()
        prediction = np.asarray(model.predict(test_x))  # BY: Claude  # NOTE: All trees of the retrained model (= best_iteration).
        scored = test_df.select("Date", "Code", "label").with_columns(
            pl.Series("Prediction", prediction)
        )
        scored.write_parquet(PREDICTIONS_DIR / f"predictions_{window_tag}.parquet")  # BY: Claude
        ic = daily_ic(scored)
        valid_ic = ic[np.isfinite(ic)]
        real = scored["label"].to_numpy()

        result = {
            "train_start": str(dates[begin]),
            "train_end": str(train_dates[-1]),  # BY: Claude  # NOTE: Last training date of the early-stopping fit.
            "retrain_end": str(dates[i - GAP - 1]),           # ISSUE(ADDRESSED): dates[i - GAP] not included. Better to use dates[i - GAP - 1] for clarity.
            "validation_start": str(validation_dates[0]),  # BY: Claude
            "test_start": str(dates[i]),
            "test_end": str(dates[end - 1]),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "retrain_rows": len(history),  # BY: Claude  # NOTE: Rows of the retrained model.
            "test_rows": len(test_df),
            "features": len(selected),
            "best_iteration": best_iteration,  # BY: Claude
            "r2": float(r2_score(real, prediction)),    # NOTE: R^2 on test set only.
            "kt": float(stats.kendalltau(prediction, real).statistic),
            # NOTE: Measures pairwise ordinal association. 1: perfect agreement, -1: perfect disagreement.
            # ISSUE: Kendall's tau computed on all test rows, not daily. Maybe compute daily and average for consistency with IC.
            "ic": float(valid_ic.mean()),               # NOTE: Mean daily IC on test set only.
            "ir": float(valid_ic.mean() / valid_ic.std() * np.sqrt(len(test_dates))),
            # NOTE: Prediction ability devided by volatility, scaled by the breadth where the ability is tested.
            # NOTE: Higher IR means consistently better prediction tested across sufficiently long time.
            # ISSUE: Effective count of independent test dates is less than len(test_dates) due to autocorrelation. IR overestimated.
            # ISSUE: valid_ic excludes ill values, but IR is scaled by the total number of test dates.
            "runtime": time.time() - runtime_start,     # NOTE: Window runtime in seconds.
            "peak_memory_GB": peak_memory_gb(),  # BY: Claude  # NOTE: Whole process, incl. the loaded dataset.
        }
        results.append(result)
        RESULT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))  # noqa: T201

        del train_df, validation_df, test_df, train_x, train_y
        del validation_x, validation_y, test_x, model, train_matrix
        del validation_matrix, scored, prediction, real, ic, valid_ic
        del history, correlations, curves
        del full_matrix, full_order, retrain_curves  # BY: Claude
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
    launch_curve_notebook(PLOT_REFRESH_SECONDS)  # BY: Claude
    results = train()
    print(  # noqa: T201
        f"Completed {len(results)} windows "
        f"in {(time.time() - started) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
