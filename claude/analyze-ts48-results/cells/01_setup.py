import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from scipy import stats

# The notebook lives in the run folder; VS Code runs it from there.
RUN_DIR = Path.cwd()
BASELINE_DIR = __BASELINE_DIR__  # Path to the baseline run folder, or None (set by the skill).
assert (RUN_DIR / "results_lgbm.json").exists(), f"Not a run folder: {RUN_DIR}"
COMPARE = BASELINE_DIR is not None and BASELINE_DIR.resolve() != RUN_DIR.resolve()
DATA_FILE = RUN_DIR.parents[1] / "data.parquet"
PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Fixed categorical order, never cycled: one colour per window (7th = final window).
WINDOW_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SPLIT_COLORS = {"train": "#52514e", "val": "#6250d6", "test": "#e34948"}
MODEL_COLORS = {"baseline": "#8f8e88", "run": "#2a78d6"}
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2,
    "legend.frameon": False,
})
warnings.filterwarnings("ignore", message=".*non-interactive.*")


def save(fig, name: str) -> None:
    """Save a figure to plots/analysis_<name>.png, then show it."""
    fig.savefig(PLOTS_DIR / f"analysis_{name}.png", dpi=120, bbox_inches="tight")
    plt.show()


def newey_west_t(x: np.ndarray, lags: int = 4) -> float:
    """t-statistic of mean(x) with a Newey–West (Bartlett-weighted) standard error."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    e = x - x.mean()
    n = len(x)
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(var / n))


def nw_ci(x: np.ndarray) -> float:
    """Half-width of a 95% confidence interval for mean(x), Newey–West (4 lags)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = newey_west_t(x)
    return float(1.96 * abs(x.mean() / t)) if t else float("nan")


def find_files(run_dir: Path, kind: str, pattern: str) -> list[Path]:
    """Files in run_dir/<kind>/ (current layout) or run_dir/ (early runs' flat layout)."""
    sub = run_dir / kind
    files = sorted(sub.glob(pattern)) if sub.is_dir() else []
    return files or sorted(run_dir.glob(pattern))


def load_run(run_dir: Path) -> tuple[list[dict], pl.DataFrame, dict[int, dict]]:
    """Results, test predictions (with a 1-based `window` column) and learning curves."""
    results = json.loads((run_dir / "results_lgbm.json").read_text())
    start_to_window = {r["test_start"]: k + 1 for k, r in enumerate(results)}

    def window_of(path: Path) -> int:
        return start_to_window[re.search(r"_test_(\d{4}-\d{2}-\d{2})_", path.name)[1]]

    preds = pl.concat([
        pl.read_parquet(f).with_columns(pl.lit(window_of(f)).cast(pl.Int32).alias("window"))
        for f in find_files(run_dir, "predictions", "predictions_*.parquet")
    ]).sort(["window", "Date", "Code"])
    curves = {window_of(f): json.loads(f.read_text())
              for f in find_files(run_dir, "curves", "curves_*.json")}
    return results, preds, curves


def daily_metrics(preds: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Per-date IC, Rank IC, Kendall and Q5−Q1; plus quintile means and preds with quintiles."""
    daily = (
        preds.group_by(["window", "Date"], maintain_order=True)
        .agg(
            pl.corr("Prediction", "label").alias("ic"),
            pl.corr("Prediction", "label", method="spearman").alias("rank_ic"),
            pl.len().alias("n"),
        )
        .sort(["window", "Date"])
    )
    kendall = [
        stats.kendalltau(d["Prediction"].to_numpy(), d["label"].to_numpy()).statistic
        for d in preds.sort(["window", "Date"]).partition_by(["window", "Date"], maintain_order=True)
    ]
    daily = daily.with_columns(pl.Series("kendall", kendall))
    key = ["window", "Date"]
    preds = preds.with_columns(
        ((pl.col("Prediction").rank("ordinal").over(key).cast(pl.Int64) - 1) * 5
         // pl.len().over(key) + 1).alias("quintile")
    )
    quint_daily = preds.group_by(["window", "Date", "quintile"]).agg(
        pl.col("label").mean().alias("label_mean")
    )
    spread = (
        quint_daily.pivot(on="quintile", index=["window", "Date"], values="label_mean")
        .select("window", "Date", (pl.col("5") - pl.col("1")).alias("q5_minus_q1"))
    )
    daily = daily.join(spread, on=["window", "Date"]).sort(["window", "Date"])
    return daily, quint_daily, preds
