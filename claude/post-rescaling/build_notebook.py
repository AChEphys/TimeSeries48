"""Build RUN_DIR/post_rescaling.ipynb: calibrate test predictions with out-of-fold slopes.

Usage:
    python build_notebook.py RUN_DIR [--force]
    python build_notebook.py RUN_DIR --findings F.md --summary S.md   # text cells only

The run needs predictions/validation_<window>.parquet (out-of-fold predictions, saved
by train_baseline.py since run 6) next to predictions/predictions_<window>.parquet.
Seed variants of the run (run-N.k_* whose archived scripts differ only in the seed
lines) that also have out-of-fold predictions are listed in the notebook's robustness
cell. Refuses to overwrite an existing notebook without --force (it resets the text
cells). Execute it headless with claude/analyze-ts48-results/scripts/run_notebook.py.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude/compare-ts48-baseline/scripts"))
from update_baseline_table import run_key, same_but_seed  # noqa: E402

NOTEBOOK = "post_rescaling.ipynb"

INTRO = """\
# Post-rescaling: calibrating the model's output with out-of-fold slopes

**Why.** The predictions feed a portfolio optimizer, so their *size* matters, not only
their order. The calibration slope (label regressed on prediction) should be 1. Per
window it ranges from about 0.7 to 1.3, partly because the tree count sets the
prediction scale (N-15) and bagging shrinks it a little (N-23).

**Idea (T-3).** The purged k-fold already gives honest predictions for every history
row: each one comes from the fold model that did not train on it
(`predictions/validation_*.parquet`). Fit the slope there, before the test period,
and multiply the test predictions by it.

**What can't change.** Multiplying all of a day's predictions by the same positive
number, or subtracting the day's mean, keeps every day's ordering and correlations.
So Rank IC, Pearson IC and the quintile spread must stay identical; section 3
checks this. Only magnitudes (R², RMSE, calibration slope) can change.

**Sections:** 1. setup · 2. the transformation (fit and apply) · 3. corrected vs raw
on the test sets · 4. robustness over seeds · 5. findings · 6. summary.
"""

TRANSFORM_MD = """\
## 2. The transformation

For each window, using **only that window's history** (no test labels):

- **(A) rescaled:** `corrected = b × raw`.
  - `b` is the least-squares slope of label on prediction over the out-of-fold rows:
    `b = Σ (p − p̄)(y − ȳ) / Σ (p − p̄)²`.
  - `b > 1`: the model's predictions are too small, so stretch them; `b < 1`:
    too large, so shrink them.
- **(C) demeaned only:** `corrected = raw − raw's mean on that day`. No slope; this
  isolates what removing day-level shifts does on its own.
- **(B) rescaled + demeaned:** `corrected = b_day × (raw − raw's mean on that day)`.
  - The label is nearly demeaned per day, and the model can't predict day-level
    shifts (N-8), so (B) removes them from the predictions.
  - `b_day` is the same slope fitted after removing each day's mean from both
    predictions and labels, i.e. comparing stocks within a day.
  - The day's mean uses only that day's predictions, which are known when
    predicting, not labels.

**Assumption:** the fold models (trained on ~4/5 of the history) and the retrained
model (all history, same tree count) produce predictions of similar scale. The test
results in section 3 show whether this holds well enough.
"""


def code_setup(run_dir: Path, seed_dirs: list[Path]) -> str:
    seeds = ",\n    ".join(f'Path("{p}")' for p in seed_dirs)
    return f'''\
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_width_chars(250)

RUN_DIR = Path("{run_dir}")
PRED_DIR = RUN_DIR / "predictions"
PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_FILE = RUN_DIR.parents[1] / "data.parquet"
SEED_DIRS = [{seeds}]  # Seed variants of this configuration with out-of-fold files.


def window_tags(run_dir: Path) -> list[str]:
    """Window names, e.g. 'train_2022-01-04_2022-12-14_test_2022-12-29_2023-06-30'."""
    files = sorted((run_dir / "predictions").glob("validation_*.parquet"))
    return [f.name.removeprefix("validation_").removesuffix(".parquet") for f in files]


tags = window_tags(RUN_DIR)
results = json.loads((RUN_DIR / "results_lgbm.json").read_text())
trees = [r["best_iteration"] for r in results]
print(f"{{RUN_DIR.name}}: {{len(tags)}} windows with out-of-fold predictions")
print("seed variants:", [p.name for p in SEED_DIRS] or "none")
'''


CODE_TRANSFORM = '''\
def ols_slope(p: np.ndarray, y: np.ndarray) -> float:
    """Slope b of the least-squares line y ≈ a + b·p."""
    p, y = p - p.mean(), y - y.mean()
    return float(p @ y / (p @ p))


def minus_day_mean(column: str) -> pl.Expr:
    """The column minus its mean over all stocks on the same day."""
    return pl.col(column) - pl.col(column).mean().over("Date")


def rescale_window(oof: pl.DataFrame, test: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    """
    Fit the slopes on one window's out-of-fold rows; apply them to its test rows.

    oof:  history rows (Date, Code, label, Prediction, fold), each predicted by the
          fold model that did not train on it.
    test: test rows (Date, Code, label, Prediction), predicted by the retrained model.
    """
    # (A) one scale factor for the window.
    b = ols_slope(oof["Prediction"].to_numpy(), oof["label"].to_numpy())

    # (B) the same slope, comparing stocks within each day.
    oof_day = oof.select(minus_day_mean("Prediction").alias("p"),
                         minus_day_mean("label").alias("y"))
    b_day = ols_slope(oof_day["p"].to_numpy(), oof_day["y"].to_numpy())

    corrected = test.with_columns(
        (b * pl.col("Prediction")).alias("rescaled"),                      # (A)
        minus_day_mean("Prediction").alias("demeaned"),                     # (C)
        (b_day * minus_day_mean("Prediction")).alias("rescaled_demeaned"),  # (B)
    )
    fold_slopes = [ols_slope(f["Prediction"].to_numpy(), f["label"].to_numpy())
                   for f in oof.partition_by("fold", maintain_order=True)]
    return {"b": b, "b_day": b_day, "fold_slopes": fold_slopes}, corrected


def rescale_run(run_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply rescale_window to every window of a run: (fitted slopes, test rows)."""
    fits, tests = [], []
    for k, tag in enumerate(window_tags(run_dir), start=1):
        oof = pl.read_parquet(run_dir / "predictions" / f"validation_{tag}.parquet")
        test = pl.read_parquet(run_dir / "predictions" / f"predictions_{tag}.parquet")
        fit, corrected = rescale_window(oof, test)
        tests.append(corrected.with_columns(pl.lit(k).alias("window")))
        fits.append({"window": k, "b (A)": fit["b"], "b_day (B)": fit["b_day"],
                     "fold slope min": min(fit["fold_slopes"]),
                     "fold slope max": max(fit["fold_slopes"])})
    return pl.DataFrame(fits), pl.concat(tests)


fits, test = rescale_run(RUN_DIR)
fits = fits.with_columns(pl.Series("trees", trees[: fits.height]))
print("Fitted on out-of-fold history rows (b > 1: stretch; b < 1: shrink):")
print(fits.with_columns(pl.selectors.float().round(3)))
'''

COMPARE_MD = """\
## 3. Corrected vs raw on the test sets

- **Must not change:** Rank IC, Pearson IC and Q5−Q1, which are per-day orderings
  and correlations.
- **Should improve:** calibration slope (closer to 1), R² and RMSE.
- R² here is pooled over all test rows, against the overall label mean, as in the
  analysis notebooks.
- The calm and volatile rows split stocks by ex-ante volatility (the stock's own
  label SD over rows 7–26 back): the lowest and highest 20% each day (N-20).
"""

CODE_COMPARE = '''\
VARIANTS = {"raw": "Prediction", "rescaled (A)": "rescaled", "demeaned (C)": "demeaned",
            "rescaled + demeaned (B)": "rescaled_demeaned"}


def daily(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Per-day Rank IC, Pearson IC and Q5−Q1 (top minus bottom prediction quintile)."""
    quintile = ((pl.col(col).rank() * 5 / pl.len()).ceil().clip(1, 5)).over("Date")
    return (df.with_columns(quintile.alias("q"))
            .group_by("Date").agg(
                pl.corr(pl.col(col).rank(), pl.col("label").rank()).alias("rank_ic"),
                pl.corr(col, "label").alias("ic"),
                (pl.col("label").filter(pl.col("q") == 5).mean()
                 - pl.col("label").filter(pl.col("q") == 1).mean()).alias("q5_q1"))
            .sort("Date"))


def magnitude(df: pl.DataFrame, col: str) -> dict[str, float]:
    y, p = df["label"].to_numpy(), df[col].to_numpy()
    return {"R²": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "slope": ols_slope(p, y), "prediction SD": float(p.std())}


def evaluate(test: pl.DataFrame, sigma: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Overall table (rows = metrics) and per-window table, for each variant."""
    df = test.join(sigma, on=["Date", "Code"], how="left")
    overall, per_window = {}, []
    for name, col in VARIANTS.items():
        d = daily(df, col)
        windows = [magnitude(df.filter(pl.col("window") == k), col)
                   for k in range(1, df["window"].max() + 1)]
        overall[name] = {
            "Rank IC": d["rank_ic"].mean(), "Pearson IC": d["ic"].mean(),
            "Q5−Q1": d["q5_q1"].mean(), **magnitude(df, col),
            "mean |slope − 1| per window": np.mean([abs(w["slope"] - 1) for w in windows]),
            "slope, calm stocks": magnitude(df.filter(pl.col("decile") <= 2), col)["slope"],
            "slope, volatile stocks": magnitude(df.filter(pl.col("decile") >= 9), col)["slope"],
            "R², calm stocks": magnitude(df.filter(pl.col("decile") <= 2), col)["R²"],
            "R², volatile stocks": magnitude(df.filter(pl.col("decile") >= 9), col)["R²"],
        }
        per_window += [{"window": k, "variant": name, "slope": w["slope"], "R²": w["R²"]}
                       for k, w in enumerate(windows, start=1)]
    table = pl.DataFrame([{"metric": m, **{v: overall[v][m] for v in VARIANTS}}
                          for m in overall["raw"]])
    return table, pl.DataFrame(per_window)


# Ex-ante volatility deciles (no test labels used: rows 7–26 back are realized).
sigma = (pl.read_parquet(DATA_FILE, columns=["Date", "Code", "label"])
         .sort("Code", "Date")
         .with_columns(pl.col("label").shift(7).rolling_std(20, min_samples=10)
                       .over("Code").alias("sigma"))
         .select("Date", "Code", "sigma")
         .with_columns(((pl.col("sigma").rank() / pl.col("sigma").count()).over("Date")
                        * 10).ceil().clip(1, 10).alias("decile")))

table, per_window = evaluate(test, sigma)
print("All test windows:")
print(table.with_columns(pl.selectors.float().round(5)))

wide = per_window.pivot(on="variant", index="window", values=["slope", "R²"])
print("\\nPer window (slope: 1 = calibrated):")
print(wide.join(fits.select("window", "b (A)", "b_day (B)"), on="window")
      .with_columns(pl.selectors.float().round(4)))

# Intact check: per-day ordering metrics must be identical.
raw_daily = daily(test, "Prediction")
for name, col in list(VARIANTS.items())[1:]:
    d = daily(test, col)
    print(f"\\n{name}: max |Δ daily Rank IC| {np.abs(d['rank_ic'] - raw_daily['rank_ic']).max():.1e}, "
          f"max |Δ daily IC| {np.abs(d['ic'] - raw_daily['ic']).max():.1e}, "
          f"max |Δ daily Q5−Q1| {np.abs(d['q5_q1'] - raw_daily['q5_q1']).max():.1e}")

# Does the out-of-fold slope track the test slope across windows?
test_slope = per_window.filter(pl.col("variant") == "raw")["slope"].to_numpy()
print(f"\\ncorr(b, raw test slope) over windows: "
      f"{np.corrcoef(fits['b (A)'].to_numpy(), test_slope)[0, 1]:.2f}")

COLORS = {"raw": "#8c8c8c", "rescaled (A)": "#2a6fdb", "demeaned (C)": "#1a9e77",
          "rescaled + demeaned (B)": "#e07b1f"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name in VARIANTS:
    w = per_window.filter(pl.col("variant") == name)
    for ax, metric in zip(axes, ["slope", "R²"], strict=True):
        ax.plot(w["window"], w[metric], "o-", color=COLORS[name], lw=2, ms=7, label=name)
axes[0].axhline(1, color="0.4", ls="--", lw=1, label="calibrated (slope 1)")
axes[0].set_title("Calibration slope per window (1 = calibrated)")
axes[1].set_title("Test R² per window")
for ax in axes:
    ax.set_xticks(range(1, len(tags) + 1), [f"W{k}" for k in range(1, len(tags) + 1)])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
fig.suptitle(f"{RUN_DIR.name}: raw vs rescaled predictions on the test sets", fontsize=10)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "post_rescaling.png", dpi=110)
plt.show()
'''

ROBUST_MD = """\
## 4. Robustness over seeds

The same fit-and-apply steps for every seed variant of this configuration that saved
out-of-fold predictions. A correction worth keeping should help for every seed.
"""

CODE_ROBUST = '''\
rows = []
for run_dir in [RUN_DIR, *SEED_DIRS]:
    _, t = rescale_run(run_dir)
    tab, _ = evaluate(t, sigma)
    get = {r["metric"]: r for r in tab.iter_rows(named=True)}
    for name in VARIANTS:
        rows.append({"run": run_dir.name.split("_")[0], "variant": name,
                     "R²": get["R²"][name], "RMSE": get["RMSE"][name],
                     "slope": get["slope"][name],
                     "mean |slope − 1|": get["mean |slope − 1| per window"][name],
                     "Rank IC": get["Rank IC"][name]})
print(pl.DataFrame(rows).with_columns(pl.selectors.float().round(5)))
'''

FINDINGS = "## 5. Findings\n\n_To be written after the notebook has been run._\n"
SUMMARY = "## 6. Summary\n\n_To be written after the notebook has been run._\n"


def cell(kind: str, source: str, cell_id: str) -> dict:
    base = {"cell_type": kind, "id": cell_id, "metadata": {},
            "source": source.splitlines(keepends=True)}
    return {**base, "execution_count": None, "outputs": []} if kind == "code" else base


def seed_variants(run_dir: Path) -> list[Path]:
    n, k = run_key(run_dir)
    found = []
    for other in sorted(run_dir.parent.glob(f"run-{n}*"), key=run_key):
        on, ok = run_key(other)
        if (other != run_dir and on == n and ok != k and same_but_seed(run_dir, other)
                and any((other / "predictions").glob("validation_*.parquet"))):
            found.append(other.resolve())
    return found


def build(run_dir: Path) -> dict:
    cells = [
        cell("markdown", INTRO, "intro"),
        cell("markdown", "## 1. Setup\n", "setup-md"),
        cell("code", code_setup(run_dir.resolve(), seed_variants(run_dir)), "setup"),
        cell("markdown", TRANSFORM_MD, "transform-md"),
        cell("code", CODE_TRANSFORM, "transform"),
        cell("markdown", COMPARE_MD, "compare-md"),
        cell("code", CODE_COMPARE, "compare"),
        cell("markdown", ROBUST_MD, "robust-md"),
        cell("code", CODE_ROBUST, "robust"),
        cell("markdown", FINDINGS, "findings"),
        cell("markdown", SUMMARY, "summary"),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--findings", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    path = args.run_dir / NOTEBOOK
    if args.findings or args.summary:
        nb = json.loads(path.read_text())
        for cell_id, src in (("findings", args.findings), ("summary", args.summary)):
            if src:
                target = next(c for c in nb["cells"] if c.get("id") == cell_id)
                target["source"] = src.read_text().splitlines(keepends=True)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"updated text cells in {path}")
        return 0
    if not any((args.run_dir / "predictions").glob("validation_*.parquet")):
        print(f"{args.run_dir}: no predictions/validation_*.parquet (out-of-fold) files")
        return 1
    if path.exists() and not args.force:
        print(f"{path} exists; use --force to rebuild (resets the text cells)")
        return 1
    path.write_text(json.dumps(build(args.run_dir), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
