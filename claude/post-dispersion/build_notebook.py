"""Build RUN_DIR/post_dispersion-via-HAR.ipynb: size predictions by forecast dispersion.

Usage:
    python build_notebook.py RUN_DIR [--force]
    python build_notebook.py RUN_DIR --findings F.md --summary S.md   # text cells only

final = c x sigma_hat(day) x z, where z is the run's prediction standardized per day,
sigma_hat is a forecast of that day's cross-sectional label SD (A0: trailing mean;
A1-A3: HAR regressions fitted per window on its history days), and c is one constant
per window, fitted on the window's out-of-fold validation predictions. Needs
predictions/predictions_*.parquet and predictions/validation_*.parquet. Execute it
headless with claude/analyze-ts48-results/scripts/run_notebook.py.
"""

import argparse
import json
import sys
from pathlib import Path

NOTEBOOK = "post_dispersion-via-HAR.ipynb"

INTRO = """\
# Post-dispersion: sizing predictions by a forecast of the day's dispersion (HAR)

**Why (T-10, N-24, N-29).**
- The predictions' size matters for the optimizer, but the calibration slope swings
  by period (0.7–1.3 per window).
- A static scale factor fixes only the average (N-24). Day-level features inside
  the main model learn day-specific sizes that don't carry over (N-28, N-29).

**Idea:** split the jobs ("alpha = IC × volatility × score", Grinold & Kahn).
- **Score:** the main model's ordering and relative distances,
  `z = (prediction − day mean) / day SD`.
- **Volatility:** a small auxiliary model forecasts how widely labels will spread
  across stocks on that day (`σ̂_day`). Dispersion is persistent (volatility
  clustering): last month's realized spread explains ~20% of the next one.
- **IC:** one constant `c` per window, fitted on the window's validation
  predictions.
- **final = c × σ̂_day × z.**

**What can't change:** on each day the predictions are only shifted and multiplied
by a positive number, so the daily Rank IC, Pearson IC and Q5−Q1 are identical to
the raw model's. Only magnitudes can change.

**Variants:**

| Variant | σ̂_day | Fitted parameters (per window) |
|---|---|---|
| Reference | none: the run's demeaned output (Run-N demeaned, N-24) | 0 |
| Z | a constant (per-day standardized, one scale) | c |
| A0 | trailing mean of realized dispersion, dates t−26 … t−7 | c |
| A1 | HAR on log dispersion: day (t−7), week (t−11 … t−7), month (t−26 … t−7) | 4 + c |
| A2 | A1 + the main model's own prediction spread that day | 5 + c |
| A3 | A2 + one summary of the 48 features' daily SDs | 6 + c |
| Oracle | the realized dispersion of that day (hindsight; ceiling only) | c |

Labels look ~5 days ahead, so on date t only labels of dates ≤ t−7 are realized (the
same lag as the regime measure).
"""

CODE_SETUP = '''\
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(250)

RUN_DIR = Path("__RUN_DIR__")
PRED_DIR = RUN_DIR / "predictions"
PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_FILE = RUN_DIR.parents[1] / "data.parquet"
LAG = 7            # On date t, labels of dates <= t - LAG are realized.
MIN_STOCKS = 1000  # Days with fewer stocks (two near-empty dates, I-9) give no dispersion.

tags = [f.name.removeprefix("validation_").removesuffix(".parquet")
        for f in sorted(PRED_DIR.glob("validation_*.parquet"))]
oof = {k: pl.read_parquet(PRED_DIR / f"validation_{t}.parquet") for k, t in enumerate(tags, 1)}
test = {k: pl.read_parquet(PRED_DIR / f"predictions_{t}.parquet") for k, t in enumerate(tags, 1)}
print(f"{RUN_DIR.name}: {len(tags)} windows")
'''

DAY_MD = """\
## 2. The day-level table

One row per date, from the whole dataset:
- **sigma:** the label's SD across stocks that day, i.e. the dispersion to
  forecast.
- **D / W / M:** realized dispersion known on that date. That's the value 7
  dates back (D), and its means over dates t−11 … t−7 (W) and t−26 … t−7 (M).
- **featsd:** each of the 48 features' SD across stocks that day, used for A3 only.
"""

CODE_DAY = '''\
features = [c for c in pl.read_parquet_schema(DATA_FILE) if c.startswith("feature")]
raw = pl.read_parquet(DATA_FILE, columns=["Date", "label", *features])
finite = lambda c: pl.col(c).filter(pl.col(c).is_finite())  # noqa: E731
day = (raw.group_by("Date")
       .agg(finite("label").std().alias("sigma"), pl.len().alias("n"),
            *[finite(f).std().alias(f"sd_{f}") for f in features])
       .sort("Date")
       .with_columns(pl.when(pl.col("n") >= MIN_STOCKS).then(pl.col("sigma"))
                     .alias("sigma")))
known = pl.col("sigma").shift(LAG)  # Realized by date t.
day = day.with_columns(
    known.rolling_mean(5, min_samples=3).alias("W"),
    known.rolling_mean(20, min_samples=10).alias("M"),
).with_columns(pl.coalesce(known, pl.col("W")).alias("D"))
del raw
print(day.select("Date", "n", "sigma", "D", "W", "M").describe())
'''

MODEL_MD = """\
## 3. The auxiliary models, fitted per window

For each window, using **only its history dates**:
- **A1:** `log σ = a + b₁ log D + b₂ log W + b₃ log M`, the HAR model (Corsi, 2009).
- **A2:** A1 + `log spread`, where spread is the SD across stocks of the main model's
  predictions that day.
  - History days use the validation predictions (fold models); test days use the
    retrained model.
- **A3:** A2 + `featsd`, the 48 features' log daily SDs, each standardized over
  the window's history days, then averaged into one number.
- The forecasts are `exp(fitted log σ)`. A constant bias from the log doesn't
  matter, because `c` absorbs it.

On test days the inputs D/W/M keep updating with labels realized by then, as they
would in live use. The fitted coefficients stay fixed until the next window.
"""

CODE_MODEL = '''\
def spread(df: pl.DataFrame) -> pl.DataFrame:
    """Per date: log SD across stocks of the main model's predictions."""
    return df.group_by("Date").agg(pl.col("Prediction").std().log().alias("log_spread"))


def design(rows: pl.DataFrame, variant: str) -> np.ndarray:
    """Columns of the regression for one variant (with an intercept)."""
    cols = [np.ones(rows.height), *[np.log(rows[c].to_numpy()) for c in ("D", "W", "M")]]
    if variant in ("A2", "A3"):
        cols.append(rows["log_spread"].to_numpy())
    if variant == "A3":
        cols.append(rows["featsd"].to_numpy())
    return np.column_stack(cols)


sigma_hat, coefs, quality = {}, [], []
for k in range(1, len(tags) + 1):
    hist_dates, test_dates = oof[k]["Date"].unique(), test[k]["Date"].unique()
    sd_cols = [f"sd_{f}" for f in features]
    hist_sd = day.filter(pl.col("Date").is_in(hist_dates)).select(
        [pl.col(c).log() for c in sd_cols])
    mu, sd = hist_sd.mean().row(0), hist_sd.std().row(0)
    featsd = pl.mean_horizontal([(pl.col(c).log() - m) / s
                                 for c, m, s in zip(sd_cols, mu, sd, strict=True)])
    base = day.select("Date", "sigma", "D", "W", "M", featsd.alias("featsd"))
    hist = (base.filter(pl.col("Date").is_in(hist_dates))
            .join(spread(oof[k]), on="Date").drop_nulls())
    fut = base.filter(pl.col("Date").is_in(test_dates)).join(spread(test[k]), on="Date")
    out = {"hist": hist.select("Date"), "test": fut.select("Date")}
    for v in ("A1", "A2", "A3"):
        beta, *_ = np.linalg.lstsq(design(hist, v), np.log(hist["sigma"].to_numpy()),
                                   rcond=None)
        out["hist"] = out["hist"].with_columns(pl.Series(v, np.exp(design(hist, v) @ beta)))
        out["test"] = out["test"].with_columns(pl.Series(v, np.exp(design(fut, v) @ beta)))
        coefs.append({"window": k, "variant": v, "days": hist.height,
                      **{n: float(b) for n, b in zip(
                          ["a", "b_day", "b_week", "b_month", "b_spread", "b_featsd"],
                          beta, strict=False)}})
    for part, rows in (("hist", hist), ("test", fut)):
        out[part] = out[part].with_columns(
            pl.Series("A0", rows["M"].to_numpy()),
            pl.Series("Z", np.ones(rows.height)),
            pl.Series("Oracle", rows["sigma"].to_numpy()))
    sigma_hat[k] = out
    real = np.log(fut["sigma"].to_numpy())
    for v in ("A0", "A1", "A2", "A3"):
        f = np.log(out["test"][v].to_numpy())
        ok = np.isfinite(real) & np.isfinite(f)
        quality.append({"window": k, "variant": v,
                        "corr(log σ̂, log σ) on test days": np.corrcoef(f[ok], real[ok])[0, 1]})

print("Fitted coefficients (log scale):")
print(pl.DataFrame(coefs).with_columns(pl.selectors.float().round(3)))
q = pl.DataFrame(quality)
print("\\nForecast quality on test days (correlation with the realized log dispersion):")
print(q.pivot(on="variant", index="window", values=q.columns[-1])
      .with_columns(pl.selectors.float().round(3)))
print(q.group_by("variant").agg(pl.col(q.columns[-1]).mean().round(3)).sort("variant"))
'''

RESCALE_MD = """\
## 4. The rescaling, and the comparison on the test sets

For each window and variant:
1. `z = (prediction − day mean) / day SD`, on the validation and the test
   predictions.
2. `x = σ̂_day × z`.
3. `c` = the least-squares slope of label on `x` over the window's validation rows,
   fitted through the origin. `z` sums to zero each day, so the labels' daily mean
   doesn't matter.
4. **final = c × x** on the test rows.

The reference is the run's demeaned output (`prediction − day mean`).

**Metrics on the test sets:**
- R², RMSE and pooled slope;
- each window's slope and the mean |slope − 1| over windows;
- each month's slope and the mean |slope − 1| over months, since dispersion
  changes faster than windows do;
- daily Rank IC and Pearson IC, which must not change.
"""

CODE_RESCALE = '''\
VARIANTS = ["Z", "A0", "A1", "A2", "A3", "Oracle"]


def standardized(df: pl.DataFrame) -> pl.DataFrame:
    p = pl.col("Prediction")
    return df.with_columns(((p - p.mean().over("Date")) / p.std().over("Date")).alias("z"))


def slope(p: np.ndarray, y: np.ndarray) -> float:
    p, y = p - p.mean(), y - y.mean()
    return float(p @ y / (p @ p))


frames, scales = [], []
for k in range(1, len(tags) + 1):
    v_rows = standardized(oof[k]).join(sigma_hat[k]["hist"], on="Date")
    t_rows = standardized(test[k]).join(sigma_hat[k]["test"], on="Date", how="left")
    p = pl.col("Prediction")
    t_rows = t_rows.with_columns((p - p.mean().over("Date")).alias("Reference"),
                                 pl.lit(k).alias("window"))
    for v in VARIANTS:
        x = (v_rows["z"] * v_rows[v]).to_numpy()
        y = v_rows["label"].to_numpy()
        ok = np.isfinite(x)
        c = float(x[ok] @ y[ok] / (x[ok] @ x[ok]))
        scales.append({"window": k, "variant": v, "c": c})
        t_rows = t_rows.with_columns((c * pl.col("z") * pl.col(v)).alias(v))
    frames.append(t_rows)
scored = pl.concat(frames).with_columns(pl.col("Date").dt.strftime("%Y-%m").alias("month"))
missing = scored.select([pl.col(v).is_null().sum() for v in VARIANTS]).row(0)
print("test rows without a forecast (near-empty days):", dict(zip(VARIANTS, missing)))
print("c per window:")
print(pl.DataFrame(scales).pivot(on="variant", index="window", values="c")
      .with_columns(pl.selectors.float().round(4)))

y_all = scored["label"].to_numpy()
rows, per_window = [], []
for name in ["Reference", *VARIANTS]:
    df = scored.filter(pl.col(name).is_not_null())
    y, pred = df["label"].to_numpy(), df[name].to_numpy()
    d = df.group_by("Date").agg(
        pl.corr(pl.col(name).rank(), pl.col("label").rank()).alias("ric"),
        pl.corr(name, "label").alias("ic"))
    w_slopes = [slope(g[name].to_numpy(), g["label"].to_numpy())
                for g in df.partition_by("window", maintain_order=True)]
    m_slopes = [slope(g[name].to_numpy(), g["label"].to_numpy())
                for g in df.partition_by("month", maintain_order=True) if g.height > 50_000]
    rows.append({
        "variant": name, "rows": df.height,
        "R²": 1 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum()),
        "RMSE": float(np.sqrt(((y - pred) ** 2).mean())),
        "pooled slope": slope(pred, y),
        "mean |slope − 1|, windows": float(np.mean(np.abs(np.array(w_slopes) - 1))),
        "mean |slope − 1|, months": float(np.mean(np.abs(np.array(m_slopes) - 1))),
        "Rank IC": d["ric"].mean(), "Pearson IC": d["ic"].mean(),
    })
    for g in df.partition_by("window", maintain_order=True):
        yy, pp = g["label"].to_numpy(), g[name].to_numpy()
        per_window.append({"variant": name, "window": g["window"][0],
                           "slope": slope(pp, yy),
                           "R²": 1 - float(((yy - pp) ** 2).sum())
                           / float(((yy - yy.mean()) ** 2).sum())})
results = pl.DataFrame(rows)
print("\\nAll test windows:")
print(results.with_columns(pl.selectors.float().round(5)))
pw = pl.DataFrame(per_window)
print("\\nCalibration slope per window (1 = calibrated):")
print(pw.pivot(on="variant", index="window", values="slope")
      .with_columns(pl.selectors.float().round(3)))
print("\\nR² per window:")
print(pw.pivot(on="variant", index="window", values="R²")
      .with_columns(pl.selectors.float().round(5)))
'''

PLOT_MD = "## 5. Plots\n"

CODE_PLOT = '''\
COLORS = {"Reference": "#8c8c8c", "Z": "#b07aa1", "A0": "#2a6fdb", "A1": "#1a9e77",
          "A2": "#e07b1f", "A3": "#c23b3b", "Oracle": "#000000"}
fig, ax = plt.subplots(figsize=(14, 4), layout="constrained")
real = day.filter(pl.col("Date").is_in(scored["Date"].unique()))
ax.plot(real["Date"], real["sigma"], c="0.6", lw=1, label="realized dispersion (label SD)")
for v in ("A0", "A2"):
    s = pl.concat([sigma_hat[k]["test"].select("Date", v) for k in sigma_hat]).sort("Date")
    scale = real["sigma"].mean() / s[v].mean()  # Forecasts on the same level, to compare shapes.
    ax.plot(s["Date"], s[v] * scale, c=COLORS[v], lw=1.6, label=f"{v} forecast (rescaled)")
ax.set_title("Daily cross-sectional label SD on the test dates: realized vs forecast")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, frameon=False)
fig.savefig(PLOTS_DIR / "post_dispersion_forecast.png", dpi=110)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
for name in ["Reference", *VARIANTS]:
    g = pw.filter(pl.col("variant") == name).sort("window")
    style = dict(c=COLORS[name], lw=1.4 if name == "Oracle" else 2, ms=6,
                 ls=":" if name == "Oracle" else "-")
    axes[0].plot(g["window"], g["slope"], "o", label=name, **style)
    axes[1].plot(g["window"], g["R²"], "o", label=name, **style)
axes[0].axhline(1, c="0.4", ls="--", lw=1)
axes[0].set_title("Calibration slope per window (1 = calibrated)")
axes[1].set_title("Test R² per window")
for ax in axes:
    ax.set_xticks(range(1, len(tags) + 1), [f"W{k}" for k in range(1, len(tags) + 1)])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(PLOTS_DIR / "post_dispersion_windows.png", dpi=110)
plt.show()
'''

COMBO_MD = """\
## 6. Combining the model's own spread with the forecast (user idea, N-31 follow-up)

Sections 4–5 *replace* the model's own daily spread by `σ̂_day`. Here `σ̂_day` only
*adjusts* it, and the data decide how much:

`final = k₁ · (prediction − day mean) + (k₀ + k₂ · σ̂_day) · z`

- `k₁` weights the model's own spread (the reference is `k₁ = 1`, `k₀ = k₂ = 0`).
- `k₀` adds a constant per-day size, `k₂` one that grows with the forecast.
- The weights are fitted per window on the validation rows by least squares
  through the origin, as `c` was.
- Equivalently, each day's demeaned prediction is multiplied by
  `k₁ + (k₀ + k₂ σ̂_day) / day SD`. If that multiplier turns negative on a day, the
  day's ranking flips, so Rank IC is reported too.

| Variant | Columns fitted |
|---|---|
| own, refit | `k₁` only (a static rescale of the reference) |
| own + const | `k₁`, `k₀` (no dispersion forecast) |
| own + A0 / A1 / A2 | `k₁`, `k₀`, `k₂` with that forecast |
| own + Oracle | `k₁`, `k₀`, `k₂` with the realized dispersion (hindsight ceiling) |
"""

CODE_COMBO = '''\
COMBOS = {  # Columns of the least-squares fit; final = columns @ weights.
    "own, refit": ["own"],
    "own + const": ["own", "z"],
    "own + A0": ["own", "z", "A0z"],
    "own + A1": ["own", "z", "A1z"],
    "own + A2": ["own", "z", "A2z"],
    "own + Oracle": ["own", "z", "Oraclez"],
}
WEIGHT_NAMES = ["k1 (own)", "k0 (const)", "k2 (σ̂)"]


def with_terms(df: pl.DataFrame, forecasts: pl.DataFrame, how: str) -> pl.DataFrame:
    p = pl.col("Prediction")
    df = standardized(df).with_columns((p - p.mean().over("Date")).alias("own"))
    return df.join(forecasts, on="Date", how=how).with_columns(
        [(pl.col(v) * pl.col("z")).alias(f"{v}z")
         for v in ("A0", "A1", "A2", "Oracle")])


frames, weights = [], []
for k in range(1, len(tags) + 1):
    v_rows = with_terms(oof[k], sigma_hat[k]["hist"], "inner")
    t_rows = with_terms(test[k], sigma_hat[k]["test"], "left").with_columns(
        pl.col("own").alias("Reference"), pl.lit(k).alias("window"))
    for name, cols in COMBOS.items():
        x, y = v_rows.select(cols).to_numpy(), v_rows["label"].to_numpy()
        ok = np.isfinite(x).all(axis=1)
        w, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
        weights.append({"window": k, "variant": name,
                        **dict(zip(WEIGHT_NAMES, map(float, w), strict=False))})
        t_rows = t_rows.with_columns(
            pl.Series(name, t_rows.select(cols).to_numpy() @ w).fill_nan(None))
    frames.append(t_rows)
combined = pl.concat(frames).with_columns(
    pl.col("Date").dt.strftime("%Y-%m").alias("month"))

wt = pl.DataFrame(weights)
print("Fitted weights per window:")
print(wt.with_columns(pl.selectors.float().round(3)))

rows, per_window = [], []
for name in ["Reference", *COMBOS]:
    df = combined.filter(pl.col(name).is_not_null())
    y, pred = df["label"].to_numpy(), df[name].to_numpy()
    d = df.group_by("Date").agg(
        pl.corr(pl.col(name).rank(), pl.col("label").rank()).alias("ric"))
    groups = df.partition_by("window", maintain_order=True)
    w_slopes = [slope(g[name].to_numpy(), g["label"].to_numpy()) for g in groups]
    m_slopes = [slope(g[name].to_numpy(), g["label"].to_numpy())
                for g in df.partition_by("month", maintain_order=True)
                if g.height > 50_000]
    rows.append({
        "variant": name,
        "R²": 1 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum()),
        "RMSE": float(np.sqrt(((y - pred) ** 2).mean())),
        "mean |slope − 1|, windows": float(np.mean(np.abs(np.array(w_slopes) - 1))),
        "mean |slope − 1|, months": float(np.mean(np.abs(np.array(m_slopes) - 1))),
        "Rank IC": d["ric"].mean(),
    })
    for g in groups:
        yy, pp = g["label"].to_numpy(), g[name].to_numpy()
        per_window.append({"variant": name, "window": g["window"][0],
                           "R²": 1 - float(((yy - pp) ** 2).sum())
                           / float(((yy - yy.mean()) ** 2).sum())})
print("\\nAll test windows:")
print(pl.DataFrame(rows).with_columns(pl.selectors.float().round(5)))
cw = pl.DataFrame(per_window)
print("\\nR² per window:")
print(cw.pivot(on="variant", index="window", values="R²")
      .with_columns(pl.selectors.float().round(5)))

own_const = wt.filter(pl.col("variant") == "own + const")
print("\\nown + const: implied daily multiplier of (prediction − day mean)")
for k, k1, k0 in own_const.select("window", "k1 (own)", "k0 (const)").iter_rows():
    day_sd = combined.filter(pl.col("window") == k).group_by("Date").agg(
        pl.col("Prediction").std())["Prediction"].to_numpy()
    m = k1 + k0 / day_sd
    print(f"  W{k}: median {np.median(m):.2f}, 10–90% {np.percentile(m, 10):.2f}"
          f" … {np.percentile(m, 90):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
COMBO_COLORS = {"Reference": "#8c8c8c", "own, refit": "#b07aa1",
                "own + const": "#2a6fdb", "own + A0": "#1a9e77",
                "own + A1": "#e07b1f", "own + A2": "#c23b3b", "own + Oracle": "#000000"}
for name in COMBOS:
    g = wt.filter(pl.col("variant") == name).sort("window")
    axes[0].plot(g["window"], g["k1 (own)"], "o-", c=COMBO_COLORS[name], lw=2, ms=6,
                 label=name)
for name in ["Reference", *COMBOS]:
    g = cw.filter(pl.col("variant") == name).sort("window")
    axes[1].plot(g["window"], g["R²"], "o-", c=COMBO_COLORS[name], lw=2, ms=6,
                 label=name)
axes[0].axhline(1, c="0.4", ls="--", lw=1)
axes[0].set_title("Weight on the model's own spread, k₁ (reference = 1)")
axes[1].set_title("Test R² per window")
for ax in axes:
    ax.set_xticks(range(1, len(tags) + 1), [f"W{k}" for k in range(1, len(tags) + 1)])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(PLOTS_DIR / "post_dispersion_combination.png", dpi=110)
plt.show()
'''

FINDINGS = "## 7. Findings\n\n_To be written after the notebook has been run._\n"
SUMMARY = "## 8. Summary\n\n_To be written after the notebook has been run._\n"


def cell(kind: str, source: str, cell_id: str) -> dict:
    base = {"cell_type": kind, "id": cell_id, "metadata": {},
            "source": source.splitlines(keepends=True)}
    return {**base, "execution_count": None, "outputs": []} if kind == "code" else base


def build(run_dir: Path) -> dict:
    cells = [
        cell("markdown", INTRO, "intro"),
        cell("markdown", "## 1. Setup\n", "setup-md"),
        cell("code", CODE_SETUP.replace("__RUN_DIR__", str(run_dir.resolve())), "setup"),
        cell("markdown", DAY_MD, "day-md"),
        cell("code", CODE_DAY, "day"),
        cell("markdown", MODEL_MD, "model-md"),
        cell("code", CODE_MODEL, "model"),
        cell("markdown", RESCALE_MD, "rescale-md"),
        cell("code", CODE_RESCALE, "rescale"),
        cell("markdown", PLOT_MD, "plot-md"),
        cell("code", CODE_PLOT, "plot"),
        cell("markdown", COMBO_MD, "combo-md"),
        cell("code", CODE_COMBO, "combo"),
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
        print(f"{args.run_dir}: no out-of-fold validation predictions")
        return 1
    if path.exists() and not args.force:
        print(f"{path} exists; use --force to rebuild (resets the text cells)")
        return 1
    path.write_text(json.dumps(build(args.run_dir), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
