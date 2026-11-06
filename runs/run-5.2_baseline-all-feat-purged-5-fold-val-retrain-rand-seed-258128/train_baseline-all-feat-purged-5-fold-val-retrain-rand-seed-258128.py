#!/usr/bin/env python3
"""Train rolling LightGBM models from an existing data.parquet."""

from __future__ import annotations

import contextlib  # BY: Claude
import gc
import itertools  # BY: Claude
import json
import os
import shutil  # BY: Claude
import subprocess  # BY: Claude
import textwrap  # BY: Claude
import time
from dataclasses import dataclass  # BY: Claude
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
RUN_DIR = ROOT / "runs/run-5.2_baseline-all-feat-purged-5-fold-val-retrain-rand-seed-258128"  # noqa: E501
RESULT_FILE = RUN_DIR / "results_lgbm.json"
MODEL_DIR = RUN_DIR / "models"
PREDICTIONS_DIR = RUN_DIR / "predictions"
CURVES_DIR = RUN_DIR / "curves"
ARXIV_FILE = RUN_DIR / "train_baseline-all-feat-purged-5-fold-val-retrain-rand-seed-258128.py"  # NOTE: Archive copy of this script, for reproducibility.  # noqa: E501
CURVE_FLUSH_EVERY = 10      # BY: Claude  # NOTE: Write learning curves to disk every N iterations.
PLOT_REFRESH_SECONDS = 3    # NOTE: Refresh the learning-curve plot every N seconds.

RANDOM_SEED = 258128
SMOKE_TEST = False      # NOTE: If True, only the first rolling window is trained and evaluated. Useful for debugging.
SKIP_LAST_WINDOW = True # NOTE: If True, skip the last rolling window. The last window is reserved for future evaluation on unseen data.

CORR_BAR = 0.02
LOOKBACK = 240
GAP = 10
TEST_DAYS = 120
VALIDATION_FOLD = 5  # NOTE: Purged k-fold validation (k=5).
PURGE = GAP  # BY: Claude  # NOTE: History dates dropped between train and validation; labels look ~6 dates ahead.
SELECT_FEATURES = False  # BY: Claude  # NOTE: Correlation filter (CORR_BAR) on/off. Off = all features.
MAX_ROUNDS = 2000  # BY: Claude  # NOTE: Cap on boosting rounds in the k-fold.
EARLY_STOPPING_ROUNDS = 200  # BY: Claude  # NOTE: Stop when the fold-mean validation R² hasn't improved for this many rounds.


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


def select_features(df: pl.DataFrame, features: list[str]) -> list[str]:  # BY: Claude
    """Features with |corr(feature, label)| > CORR_BAR on `df` (all if filter off)."""  # BY: Claude
    if not SELECT_FEATURES:  # BY: Claude
        return features  # BY: Claude
    correlations = df.select(  # BY: Claude
        [pl.corr(feature, "label").alias(feature) for feature in features]  # BY: Claude
    ).row(0, named=True)  # BY: Claude
    selected = [  # BY: Claude
        feature  # BY: Claude
        for feature in features  # BY: Claude
        if correlations[feature] is not None  # BY: Claude
        and abs(correlations[feature]) > CORR_BAR  # BY: Claude
    ]  # BY: Claude  # NOTE: Only features of sufficient correlation with the label are considered.
    if not selected:  # BY: Claude
        msg = "No feature passed the correlation threshold"  # BY: Claude
        raise RuntimeError(msg)  # BY: Claude
    return selected  # BY: Claude


def purged_folds(  # BY: Claude
    n_dates: int, k: int, purge: int  # BY: Claude
) -> list[tuple[list[int], list[int]]]:  # BY: Claude
    """
    Purged k-fold on date indices: (training, validation) index lists per fold.

    The validation blocks are k contiguous, equal-size (±1) blocks. A fold trains on
    every other date except `purge` dates on each side of its block, so no training
    label overlaps a validation label. Edge folds lose one purge gap, middle folds two.
    """  # BY: Claude
    folds = []  # BY: Claude
    for block in np.array_split(np.arange(n_dates), k):  # BY: Claude
        lo, hi = int(block[0]) - purge, int(block[-1]) + purge  # BY: Claude
        train_idx = [j for j in range(n_dates) if j < lo or j > hi]  # BY: Claude
        folds.append((train_idx, block.tolist()))  # BY: Claude
    return folds  # BY: Claude


def day_starts(days: np.ndarray) -> np.ndarray:  # BY: Claude
    """Row offsets where each day begins (rows sorted by day), plus the total length."""  # BY: Claude
    return np.flatnonzero(np.r_[True, days[1:] != days[:-1], True])  # BY: Claude


def day_ranks(values: np.ndarray, starts: np.ndarray) -> np.ndarray:  # BY: Claude
    """Average ranks of `values` within each day (ties share their mean rank)."""  # BY: Claude
    ranks = np.empty(len(values), dtype=np.float64)  # BY: Claude
    for a, b in itertools.pairwise(starts):  # BY: Claude
        ranks[a:b] = stats.rankdata(values[a:b])  # BY: Claude
    return ranks  # BY: Claude


def daily_rank_ic(  # BY: Claude
    prediction: np.ndarray, label_rank: np.ndarray, starts: np.ndarray  # BY: Claude
) -> float:  # BY: Claude
    """Mean over days of the within-day Spearman correlation (Rank IC)."""  # BY: Claude
    counts = np.diff(starts)  # BY: Claude
    rank = day_ranks(prediction, starts)  # BY: Claude
    rank_sum = np.add.reduceat(rank, starts[:-1])  # BY: Claude
    label_sum = np.add.reduceat(label_rank, starts[:-1])  # BY: Claude
    dp = rank - np.repeat(rank_sum / counts, counts)  # BY: Claude  # NOTE: Demeaned within each day.
    dl = label_rank - np.repeat(label_sum / counts, counts)  # BY: Claude
    with np.errstate(invalid="ignore", divide="ignore"):  # BY: Claude  # NOTE: A day of equal predictions gives NaN.
        spread = np.add.reduceat(dp * dp, starts[:-1]) * np.add.reduceat(  # BY: Claude
            dl * dl, starts[:-1]  # BY: Claude
        )  # BY: Claude
        ic = np.add.reduceat(dp * dl, starts[:-1]) / np.sqrt(spread)  # BY: Claude
    return float(np.nanmean(ic)) if np.isfinite(ic).any() else float("nan")  # BY: Claude


@dataclass  # BY: Claude
class Fold:  # BY: Claude
    """One purged fold: its booster, plus what its metrics need."""  # BY: Claude

    booster: lgb.Booster  # BY: Claude
    train_var: float  # BY: Claude  # NOTE: Label variance, to turn RMSE into R² = 1 - RMSE² / var.
    val_var: float  # BY: Claude
    starts: np.ndarray  # BY: Claude  # NOTE: Day offsets of the validation rows (sorted by date).
    label_rank: np.ndarray  # BY: Claude  # NOTE: Within-day ranks of the validation labels.

    def rank_ic(  # BY: Claude
        self, prediction: np.ndarray, _data: lgb.Dataset  # BY: Claude
    ) -> tuple[str, float, bool]:  # BY: Claude
        """LightGBM `feval`: daily Rank IC on this fold's validation rows."""  # BY: Claude
        return "rank_ic", daily_rank_ic(prediction, self.label_rank, self.starts), True  # BY: Claude


@dataclass  # BY: Claude
class CVResult:  # BY: Claude
    """What the purged k-fold decided, and fold facts for the results file."""  # BY: Claude

    best_iteration: int  # BY: Claude  # NOTE: Argmax of the fold-mean validation R².
    rank_ic_best: int  # BY: Claude  # NOTE: Argmax of the fold-mean validation Rank IC (information only).
    val_r2: float  # BY: Claude  # NOTE: Fold-mean validation R² at best_iteration.
    val_rank_ic: float  # BY: Claude  # NOTE: Fold-mean validation Rank IC at best_iteration.
    fold_validation: list[list[str]]  # BY: Claude  # NOTE: [first, last] validation date per fold.
    fold_train_rows: list[int]  # BY: Claude
    fold_validation_rows: list[int]  # BY: Claude
    fold_features: list[int]  # BY: Claude


def build_folds(  # BY: Claude
    history: pl.DataFrame, history_dates: list, features: list[str], params: dict  # BY: Claude
) -> tuple[list[Fold], CVResult]:  # BY: Claude
    """Split the history into purged folds; select features, make a booster per fold."""  # BY: Claude
    folds: list[Fold] = []  # BY: Claude
    info = CVResult(0, 0, float("nan"), float("nan"), [], [], [], [])  # BY: Claude
    for train_idx, val_idx in purged_folds(len(history_dates), VALIDATION_FOLD, PURGE):  # BY: Claude
        train_df = history.filter(  # BY: Claude
            pl.col("Date").is_in([history_dates[j] for j in train_idx])  # BY: Claude
        )  # BY: Claude
        val_df = history.filter(  # BY: Claude
            pl.col("Date").is_in([history_dates[j] for j in val_idx])  # BY: Claude
        )  # BY: Claude  # NOTE: Stays sorted by date, like `history`.
        cols = select_features(train_df, features)  # BY: Claude  # NOTE: This fold's training dates only.
        train_y = train_df["label"].to_numpy()  # BY: Claude
        val_y = val_df["label"].to_numpy()  # BY: Claude
        train_set = lgb.Dataset(  # BY: Claude
            train_df.select(cols).to_numpy(), label=train_y, feature_name=cols  # BY: Claude
        )  # BY: Claude
        val_set = lgb.Dataset(  # BY: Claude
            val_df.select(cols).to_numpy(),  # BY: Claude
            label=val_y,  # BY: Claude
            feature_name=cols,  # BY: Claude
            reference=train_set,  # BY: Claude
        )  # BY: Claude
        booster = lgb.Booster(params=params, train_set=train_set)  # BY: Claude
        booster.add_valid(val_set, "val")  # BY: Claude
        starts = day_starts(val_df["Date"].to_numpy())  # BY: Claude
        folds.append(  # BY: Claude
            Fold(  # BY: Claude
                booster,  # BY: Claude
                float(np.var(train_y)),  # BY: Claude
                float(np.var(val_y)),  # BY: Claude
                starts,  # BY: Claude
                day_ranks(val_y, starts),  # BY: Claude
            )  # BY: Claude
        )  # BY: Claude
        info.fold_validation.append(  # BY: Claude
            [str(history_dates[val_idx[0]]), str(history_dates[val_idx[-1]])]  # BY: Claude
        )  # BY: Claude
        info.fold_train_rows.append(len(train_df))  # BY: Claude
        info.fold_validation_rows.append(len(val_df))  # BY: Claude
        info.fold_features.append(len(cols))  # BY: Claude
    return folds, info  # BY: Claude


def purged_cv(  # BY: Claude
    history: pl.DataFrame,  # BY: Claude
    history_dates: list,  # BY: Claude
    features: list[str],  # BY: Claude
    params: dict,  # BY: Claude
    curves_file: Path,  # BY: Claude
) -> CVResult:  # BY: Claude
    """
    Boost all folds in lockstep and pick the tree count from the fold-mean curves.

    Early stopping uses the fold-mean validation R² (each fold, i.e. each market
    period, weighs the same). The fold-mean daily Rank IC is recorded alongside.
    """  # BY: Claude
    folds, info = build_folds(history, history_dates, features, params)  # BY: Claude
    curves: dict[str, dict[str, list[float]]] = {  # BY: Claude
        "train": {"rmse": [], "r2": []},  # BY: Claude  # NOTE: Fold means, as the analysis skill expects.
        "val": {"rmse": [], "r2": [], "rank_ic": []},  # BY: Claude
    }  # BY: Claude
    per_fold: dict[str, list[list[float]]] = {  # BY: Claude
        "val_r2": [[] for _ in folds],  # BY: Claude
        "val_rank_ic": [[] for _ in folds],  # BY: Claude
    }  # BY: Claude
    payload: dict[str, object] = {  # BY: Claude
        "best_iteration": None,  # BY: Claude  # NOTE: None = still training.
        "rank_ic_best": None,  # BY: Claude
        "fold_validation": info.fold_validation,  # BY: Claude
        "curves": curves,  # BY: Claude
        "folds": per_fold,  # BY: Claude
    }  # BY: Claude
    best_r2, best_iteration = -np.inf, 0  # BY: Claude
    for iteration in range(1, MAX_ROUNDS + 1):  # BY: Claude
        rows = []  # BY: Claude  # NOTE: (train RMSE, train R², val RMSE, val R², val Rank IC) per fold.
        for k, fold in enumerate(folds):  # BY: Claude
            fold.booster.update()  # BY: Claude  # NOTE: One more tree in this fold.
            train_rmse = fold.booster.eval_train()[0][2]  # BY: Claude
            val_eval = fold.booster.eval_valid(feval=fold.rank_ic)  # BY: Claude  # NOTE: [rmse, rank_ic] on the validation block.
            val_rmse, rank_ic = val_eval[0][2], val_eval[1][2]  # BY: Claude
            train_r2 = 1 - train_rmse**2 / fold.train_var  # BY: Claude
            val_r2 = 1 - val_rmse**2 / fold.val_var  # BY: Claude
            rows.append((train_rmse, train_r2, val_rmse, val_r2, rank_ic))  # BY: Claude
            per_fold["val_r2"][k].append(val_r2)  # BY: Claude
            per_fold["val_rank_ic"][k].append(rank_ic)  # BY: Claude
        mean = np.mean(rows, axis=0)  # BY: Claude
        keys = [("train", "rmse"), ("train", "r2"), ("val", "rmse"), ("val", "r2"),
                ("val", "rank_ic")]  # BY: Claude
        for (split, metric), value in zip(  # BY: Claude
            keys, mean, strict=True  # BY: Claude
        ):  # BY: Claude
            curves[split][metric].append(float(value))  # BY: Claude
        if mean[3] > best_r2:  # BY: Claude
            best_r2, best_iteration = float(mean[3]), iteration  # BY: Claude
        if iteration % 100 == 0:  # BY: Claude
            print(  # BY: Claude  # noqa: T201
                f"[{iteration}] fold-mean val R² {mean[3]:.5f}, "  # BY: Claude
                f"Rank IC {mean[4]:.4f} (best R² at {best_iteration})"  # BY: Claude
            )  # BY: Claude
        if iteration % CURVE_FLUSH_EVERY == 0:  # BY: Claude
            write_json_atomic(curves_file, payload)  # BY: Claude
        if iteration - best_iteration >= EARLY_STOPPING_ROUNDS:  # BY: Claude
            break  # BY: Claude
    info.best_iteration = best_iteration  # BY: Claude
    info.rank_ic_best = int(np.nanargmax(curves["val"]["rank_ic"])) + 1  # BY: Claude
    info.val_r2 = curves["val"]["r2"][best_iteration - 1]  # BY: Claude
    info.val_rank_ic = curves["val"]["rank_ic"][best_iteration - 1]  # BY: Claude
    payload["best_iteration"] = info.best_iteration  # BY: Claude
    payload["rank_ic_best"] = info.rank_ic_best  # BY: Claude
    write_json_atomic(curves_file, payload)  # BY: Claude  # NOTE: Final flush, now with best_iteration.
    return info  # BY: Claude


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
    FOLD_COLORS = ["C3", "C4", "C9", "C6", "C8", "C5", "C7"]  # Fold k: FOLD_COLORS[k].


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
        """Progress of the k-fold and of the retrain."""
        n, best = len(d["curves"]["val"]["r2"]), d["best_iteration"]
        k = len(d.get("folds", {}).get("val_r2", []))
        if best is None:
            return f"{k}-fold CV: {n} iterations so far..."
        cv = (
            f"{k}-fold CV: {n} iterations, best {best} (mean val R²; "
            f"mean Rank IC peaks at {d.get('rank_ic_best')})"
        )
        if r is None:
            return f"{cv}     ·     retrain: starting..."
        if r["done"]:
            return f"{cv}     ·     retrain: done ({r['rounds']} trees)"
        m = len(r["curves"].get("retrain", {}).get("rmse", []))
        return f"{cv}     ·     retrain: {m} / {r['rounds']} iterations..."


    def fold_label(k: int, ranges: list) -> str:
        """'fold 2 (val 2022-03 → 2022-05)'."""
        if k < len(ranges):
            return f"fold {k + 1} (val {ranges[k][0][:7]} → {ranges[k][1][:7]})"
        return f"fold {k + 1}"


    def draw(f: Path, d: dict, r: dict | None) -> plt.Figure:
        """One window: fold means and per-fold validation curves, retrain overlaid."""
        best, rbest, c = d["best_iteration"], d.get("rank_ic_best"), d["curves"]
        folds, ranges = d.get("folds", {}), d.get("fold_validation", [])
        rc = (r or {}).get("curves", {}).get("retrain", {})
        n = len(c["val"]["r2"])
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
        (ax_rmse, ax_r2), (ax_fold, ax_ric) = axes
        ax_rmse.plot(c["train"]["rmse"], c=COLORS["train"], label="train (fold mean)")
        if rc.get("rmse"):
            ax_rmse.plot(rc["rmse"], c=COLORS["retrain"], label="retrain (all history)")
        ax_rmse.set_title("Training RMSE")
        ax_r2.plot(c["train"]["r2"], c=COLORS["train"], label="train (fold mean)")
        ax_r2.plot(c["val"]["r2"], c=COLORS["val"], label="val (fold mean)")
        if rc.get("r2"):
            ax_r2.plot(rc["r2"], c=COLORS["retrain"], label="retrain (all history)")
        ax_r2.set_title("R²")
        for k, values in enumerate(folds.get("val_r2", [])):
            color = FOLD_COLORS[k % len(FOLD_COLORS)]
            ax_fold.plot(values, c=color, lw=1.3, label=fold_label(k, ranges))
        ax_fold.plot(c["val"]["r2"], c="black", lw=2, label="mean (sets the trees)")
        ax_fold.set_title("Validation R² per fold")
        for k, values in enumerate(folds.get("val_rank_ic", [])):
            ax_ric.plot(values, c=FOLD_COLORS[k % len(FOLD_COLORS)], lw=1.3)
        ax_ric.plot(c["val"]["rank_ic"], c="black", lw=2, label="mean")
        if rbest is not None:
            label = f"mean Rank IC peak ({rbest})"
            ax_ric.axvline(rbest, ls=":", lw=1.5, c="black", label=label)
        ax_ric.set_title("Validation daily Rank IC per fold (colours as left)")
        for ax in axes.flat:
            if best is not None:
                label = f"best iteration ({best})"
                ax.axvline(best, ls="--", lw=1, c="gray", label=label)
            ax.set_xlim(0, max(n - 1, 1))  # Set by the k-fold only.
            ax.set_xlabel("iteration")
            ax.ticklabel_format(axis="y", useOffset=False)  # No offset text on top.
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, frameon=False)
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
                        "ends. Top: fold means and the retrain; bottom: each fold's "  # BY: Claude
                        "validation curves. A window is final (saved to `plots/`, "  # BY: Claude
                        "then shown as a static PNG) once its retrain is done."  # BY: Claude
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
    1. Divides the window into history and test dates, separated by a purge gap (GAP).
    2. Purged k-fold on the history dates: k contiguous validation blocks. Each fold trains
       on the other history dates, minus PURGE dates on each side of its block, so no
       training label overlaps a validation label.
    3. In each fold, features are selected on that fold's training dates only (if SELECT_FEATURES).
    4. The k folds are boosted in lockstep. The number of trees is the argmax of the
       fold-mean validation R² (early stopping on that mean).
    5. Features are re-selected on ALL history dates (same rule), and a GBDT model is retrained
       on ALL history dates with that number of trees. It maps the selected features of a
       SINGLE stock on a SINGLE day -> label of that stock on that day.
    6. The model along with its selected features are saved to disk.
    7. The model is evaluated on the test set, and metrics are recorded. Results are saved to disk.
    """  # noqa: E501  # BY: Claude  # NOTE: Rewritten for purged k-fold (run 4).
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

        window_tag = (  # BY: Claude
            f"train_{dates[begin]}_{dates[i - GAP - 1]}_"
            f"test_{dates[i]}_{dates[end - 1]}"  # BY: Claude
        )  # BY: Claude
        curves_file = CURVES_DIR / f"curves_{window_tag}.json"  # BY: Claude
        # ISSUE(ADDRESSED): One chronological validation block = one market period; tree counts were unstable (N-14).  # BY: Claude
        cv = purged_cv(history, history_dates, features, params, curves_file)  # BY: Claude  # NOTE: Blocks until all folds stop.
        best_iteration = cv.best_iteration  # BY: Claude

        selected = select_features(history, features)  # BY: Claude  # NOTE: Same rule as in the folds, now on all history dates.
        test_df = finite_rows(
            data.filter(pl.col("Date").is_in(test_dates)), [*selected, "label"]
        )

        rng = np.random.default_rng(RANDOM_SEED)   # ISSUE(ADDRESSED): Seeding with time may hinder reproducibility.
        full_order = rng.permutation(len(history))  # BY: Claude  # NOTE: Shuffle all rows. Seems to be a common practice in ML training.
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
        )  # NOTE: Retrain with the chosen number of trees, so all history dates are learned. No early stopping.
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
            "retrain_end": str(dates[i - GAP - 1]),           # ISSUE(ADDRESSED): dates[i - GAP] not included. Better to use dates[i - GAP - 1] for clarity.
            "folds": VALIDATION_FOLD,  # BY: Claude
            "fold_validation": cv.fold_validation,  # BY: Claude  # NOTE: [first, last] validation date per fold.
            "test_start": str(dates[i]),
            "test_end": str(dates[end - 1]),
            "train_rows": round(np.mean(cv.fold_train_rows)),  # BY: Claude  # NOTE: Mean over folds.
            "validation_rows": round(np.mean(cv.fold_validation_rows)),  # BY: Claude  # NOTE: Mean over folds.
            "fold_train_rows": cv.fold_train_rows,  # BY: Claude
            "fold_validation_rows": cv.fold_validation_rows,  # BY: Claude
            "fold_features": cv.fold_features,  # BY: Claude  # NOTE: Features selected per fold.
            "retrain_rows": len(history),  # BY: Claude  # NOTE: Rows of the retrained model.
            "test_rows": len(test_df),
            "features": len(selected),  # NOTE: Features of the retrained model.
            "best_iteration": best_iteration,  # BY: Claude  # NOTE: Argmax of the fold-mean validation R².
            "rank_ic_best_iteration": cv.rank_ic_best,  # BY: Claude  # NOTE: Argmax of the fold-mean validation Rank IC (not used).
            "cv_val_r2": cv.val_r2,  # BY: Claude  # NOTE: Fold-mean validation R² at best_iteration.
            "cv_val_rank_ic": cv.val_rank_ic,  # BY: Claude # NOTE: Fold-mean validation Rank IC at best_iteration.
            "r2": float(r2_score(real, prediction)),    # NOTE: R^2 on test set only.
            "kt": float(stats.kendalltau(prediction, real).statistic),
            # NOTE: Measures pairwise ordinal association. 1: perfect agreement, -1: perfect disagreement.
            # ISSUE: Kendall's tau computed on all test rows, not daily. Maybe compute daily and average for consistency with IC.
            "ic": float(valid_ic.mean()),               # NOTE: Mean daily Pearson IC on test set only.
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

        del test_df, test_x, model, scored, prediction, real, ic, valid_ic  # BY: Claude
        del history, full_matrix, full_order, retrain_curves, cv  # BY: Claude
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
