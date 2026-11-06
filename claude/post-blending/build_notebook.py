"""Build RUN_DIR/post_blending.ipynb: blend a run's predictions with a partner run's.

Usage:
    python build_notebook.py RUN_DIR --partner PARTNER_DIR [--force]
    python build_notebook.py RUN_DIR --findings F.md --summary S.md   # text cells only

Both runs need the same windows, test predictions (predictions/predictions_*.parquet)
and out-of-fold validation predictions (predictions/validation_*.parquet). The blend is
w x (run, demeaned per day) + (1 - w) x (partner, demeaned per day). The weight is chosen
per window on that window's validation predictions only, then scored on its test set;
the full test curve over w is shown as hindsight. Refuses to overwrite an existing
notebook without --force (it resets the text cells). Execute it headless with
claude/analyze-ts48-results/scripts/run_notebook.py.
"""

import argparse
import json
import sys
from pathlib import Path

NOTEBOOK = "post_blending.ipynb"

INTRO = """\
# Post-blending: __RUN__ ⊕ __PARTNER__

**Why.** The two models agree only moderately (daily rank correlation ≈ 0.79,
N-29), so they carry partly different signals. Averaging two different, similarly
good predictions usually keeps the strengths of both and dilutes their separate
errors.

**The transformation** (section 2):
`blend = w × (this run − its day mean) + (1 − w) × (partner − its day mean)`.
- Both inputs are demeaned per day first, the standard output since N-24.
- `w = 0` is the partner alone (demeaned); `w = 1` is this run alone.

**Choosing w without peeking** (section 3):
- Each window's weight is chosen on **that window's validation predictions**: the
  purged k-fold predictions of its history, made before its test period.
- Two rules: the weight with the best validation R² (magnitudes), and the one with
  the best validation Rank IC (ranking).
- One weight fitted on all windows' validation data would use history *after* the
  earlier test windows, so it isn't used for the verdict.

**What can't change:** nothing about either model. Only how the two outputs are
combined.

**Sections:** 1. setup · 2. the transformation · 3. choosing w on validation ·
4. test results · 5. hindsight scan on test · 6. findings · 7. summary.
"""

CODE_SETUP = '''\
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(250)

RUN_DIR = Path("__RUN_DIR__")
PARTNER_DIR = Path("__PARTNER_DIR__")
PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)  # w on this run; 1 - w on the partner.
NW_LAGS = 4  # Newey-West lags for daily differences, as in the analysis notebooks.


def window_tags(run_dir: Path) -> list[str]:
    files = sorted((run_dir / "predictions").glob("predictions_*.parquet"))
    return [f.name.removeprefix("predictions_").removesuffix(".parquet") for f in files]


tags = window_tags(RUN_DIR)
assert tags == window_tags(PARTNER_DIR), "the two runs have different windows"


def load(kind: str) -> pl.DataFrame:
    """Rows of both runs side by side: p_run, p_partner, label, window (and fold)."""
    keys = ["Date", "Code", "fold"] if kind == "validation" else ["Date", "Code"]
    frames = []
    for k, tag in enumerate(tags, start=1):
        a = pl.read_parquet(RUN_DIR / "predictions" / f"{kind}_{tag}.parquet")
        b = pl.read_parquet(PARTNER_DIR / "predictions" / f"{kind}_{tag}.parquet")
        both = a.rename({"Prediction": "p_run"}).join(
            b.select(*keys, pl.col("Prediction").alias("p_partner"),
                     pl.col("label").alias("label_partner")), on=keys)
        assert (both["label"] == both["label_partner"]).all(), "labels differ"
        print(f"W{k} {kind}: run {a.height:,} rows, partner {b.height:,}, "
              f"common {both.height:,}")
        frames.append(both.drop("label_partner").with_columns(pl.lit(k).alias("window")))
    return pl.concat(frames)


val = load("validation")
test = load("predictions")
'''

TRANSFORM_MD = """\
## 2. The transformation

For a weight `w` between 0 and 1:

1. Demean each model's predictions per day: `p − mean of all stocks' p that day`.
   This uses only that day's predictions, not labels.
2. `blend = w × run_demeaned + (1 − w) × partner_demeaned`.

The blend stays demeaned. Because both inputs are in label units, so is the blend.
"""

CODE_TRANSFORM = '''\
def demeaned(column: str) -> pl.Expr:
    """The column minus its mean over all stocks on the same day."""
    return pl.col(column) - pl.col(column).mean().over("Date")


def add_blend(df: pl.DataFrame, w: float, name: str = "blend") -> pl.DataFrame:
    """blend = w x run (demeaned) + (1 - w) x partner (demeaned)."""
    return df.with_columns(
        (w * demeaned("p_run") + (1 - w) * demeaned("p_partner")).alias(name))


def ols_slope(p: np.ndarray, y: np.ndarray) -> float:
    p, y = p - p.mean(), y - y.mean()
    return float(p @ y / (p @ p))


def daily(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Per day: Rank IC, Pearson IC, Q5-Q1."""
    q = ((pl.col(col).rank() * 5 / pl.len()).ceil().clip(1, 5)).over("Date")
    return (df.with_columns(q.alias("q")).group_by("window", "Date").agg(
        pl.corr(pl.col(col).rank(), pl.col("label").rank()).alias("rank_ic"),
        pl.corr(col, "label").alias("ic"),
        (pl.col("label").filter(pl.col("q") == 5).mean()
         - pl.col("label").filter(pl.col("q") == 1).mean()).alias("q5_q1"),
    ).sort("window", "Date"))


def fold_r2(df: pl.DataFrame, col: str) -> float:
    """Validation R²: mean over folds of 1 - MSE / label variance (as the sweeps)."""
    r2 = [1 - float(((f["label"] - f[col]) ** 2).mean()) / float(f["label"].var(ddof=0))
          for f in df.partition_by("fold")]
    return float(np.mean(r2))


def pooled_r2(df: pl.DataFrame, col: str) -> float:
    """Test R²: pooled over rows, against the overall label mean (as the analyses)."""
    y, p = df["label"].to_numpy(), df[col].to_numpy()
    return 1 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def newey_west_se(x: np.ndarray, lags: int = NW_LAGS) -> float:
    x = x - x.mean()
    n, v = len(x), float(x @ x) / len(x)
    for lag in range(1, lags + 1):
        v += 2 * (1 - lag / (lags + 1)) * float(x[lag:] @ x[:-lag]) / n
    return float(np.sqrt(v / n))
'''

CHOOSE_MD = """\
## 3. Choosing w on validation, per window

For each window and each `w` in 0, 0.05, …, 1: the fold-mean R² and the mean daily
Rank IC of the blend on that window's validation predictions. Each window keeps:
- **w_R²:** the weight with the best validation R² (magnitudes);
- **w_RankIC:** the weight with the best validation Rank IC (ranking).
"""

CODE_CHOOSE = '''\
rows = []
for k in range(1, len(tags) + 1):
    v = val.filter(pl.col("window") == k)
    for w in WEIGHTS:
        b = add_blend(v, w)
        rows.append({"window": k, "w": w, "val_r2": fold_r2(b, "blend"),
                     "val_rank_ic": daily(b, "blend")["rank_ic"].mean()})
val_scan = pl.DataFrame(rows)
chosen = val_scan.group_by("window").agg(
    pl.col("w").sort_by("val_r2").last().alias("w_R2"),
    pl.col("w").sort_by("val_rank_ic").last().alias("w_RankIC"),
).sort("window")
print("Weight on this run, chosen per window on its validation predictions:")
print(chosen)
print("\\nValidation curves (rows: w; columns: window), R²:")
print(val_scan.pivot(on="window", index="w", values="val_r2")
      .with_columns(pl.selectors.float().round(5)))
'''

TEST_MD = """\
## 4. Test results with the chosen weights

Each window's test predictions are blended with its own chosen weight. Compared
with the partner alone (w = 0) and this run alone (w = 1), all demeaned.
- **Ranking:** Rank IC, Pearson IC, Q5−Q1.
- **Magnitudes:** pooled R², mean |slope − 1| over windows.
- **ΔRank IC** vs the partner: the paired daily difference, with a Newey–West 95% CI.
"""

CODE_TEST = '''\
def blend_with(df: pl.DataFrame, weights: dict[int, float]) -> pl.DataFrame:
    """Each window's rows blended with its own weight."""
    return pl.concat([add_blend(df.filter(pl.col("window") == k), w)
                      for k, w in weights.items()])


rules = {
    "partner alone (w = 0)": dict.fromkeys(range(1, len(tags) + 1), 0.0),
    "this run alone (w = 1)": dict.fromkeys(range(1, len(tags) + 1), 1.0),
    "w_R² (per window)": dict(zip(chosen["window"], chosen["w_R2"], strict=True)),
    "w_RankIC (per window)": dict(zip(chosen["window"], chosen["w_RankIC"], strict=True)),
}
base_daily = daily(blend_with(test, rules["partner alone (w = 0)"]), "blend")
summary, per_window = [], []
for name, weights in rules.items():
    b = blend_with(test, weights)
    d = daily(b, "blend")
    diff = (d["rank_ic"] - base_daily["rank_ic"]).to_numpy()
    se = newey_west_se(diff)
    slopes = [ols_slope(b.filter(pl.col("window") == k)["blend"].to_numpy(),
                        b.filter(pl.col("window") == k)["label"].to_numpy())
              for k in weights]
    summary.append({
        "rule": name, "Rank IC": d["rank_ic"].mean(), "ΔRank IC vs partner": diff.mean(),
        "± 95% (NW)": 1.96 * se, "t": diff.mean() / se if se else 0.0,
        "Pearson IC": d["ic"].mean(), "Q5-Q1": d["q5_q1"].mean(),
        "R²": pooled_r2(b, "blend"),
        "mean |slope - 1|": float(np.mean([abs(s - 1) for s in slopes])),
    })
    for k, w in weights.items():
        bw = b.filter(pl.col("window") == k)
        per_window.append({"rule": name, "window": k, "w": w,
                           "Rank IC": d.filter(pl.col("window") == k)["rank_ic"].mean(),
                           "R²": pooled_r2(bw, "blend"), "slope": slopes[k - 1]})
print(pl.DataFrame(summary).with_columns(pl.selectors.float().round(5)))
pw = pl.DataFrame(per_window)
print("\\nPer window:")
print(pw.pivot(on="rule", index="window", values=["Rank IC", "R²"])
      .with_columns(pl.selectors.float().round(5)))
'''

HINDSIGHT_MD = """\
## 5. Hindsight scan on test (descriptive only)

The same blend with **one weight for all windows**, scanned over the test sets.
Picking the best point of this curve would use the test labels, so it's shown only
to see how the test responds to w, and how well validation predicted it.
"""

CODE_HINDSIGHT = '''\
rows = []
for w in WEIGHTS:
    b = add_blend(test, w)
    d = daily(b, "blend")
    rows.append({"w": w, "Rank IC": d["rank_ic"].mean(), "Pearson IC": d["ic"].mean(),
                 "R²": pooled_r2(b, "blend")})
test_scan = pl.DataFrame(rows)
print(test_scan.with_columns(pl.selectors.float().round(5)))
val_mean = val_scan.group_by("w").agg(pl.col("val_r2").mean(),
                                      pl.col("val_rank_ic").mean()).sort("w")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
for ax, (t_col, v_col, title) in zip(axes, [
        ("Rank IC", "val_rank_ic", "Rank IC vs blend weight"),
        ("R²", "val_r2", "R² (demeaned) vs blend weight")], strict=True):
    ax.plot(test_scan["w"], test_scan[t_col], "o-", c="#2a6fdb", lw=2, ms=5,
            label="test (one w for all windows)")
    ax.plot(val_mean["w"], val_mean[v_col], "s--", c="#8c8c8c", lw=1.5, ms=4,
            label="validation (mean over windows)")
    ax.set_title(title)
    ax.set_xlabel(f"w = weight on {RUN_DIR.name.split('_')[0]} "
                  f"(1 - w on {PARTNER_DIR.name.split('_')[0]})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
fig.suptitle(f"Blend of {RUN_DIR.name} and {PARTNER_DIR.name}", fontsize=10)
fig.savefig(PLOTS_DIR / "post_blending.png", dpi=110)
plt.show()
'''

FINDINGS = "## 6. Findings\n\n_To be written after the notebook has been run._\n"
SUMMARY = "## 7. Summary\n\n_To be written after the notebook has been run._\n"


def cell(kind: str, source: str, cell_id: str) -> dict:
    base = {"cell_type": kind, "id": cell_id, "metadata": {},
            "source": source.splitlines(keepends=True)}
    return {**base, "execution_count": None, "outputs": []} if kind == "code" else base


def build(run_dir: Path, partner: Path) -> dict:
    short = run_dir.name.split("_")[0], partner.name.split("_")[0]
    cells = [
        cell("markdown", INTRO.replace("__RUN__", short[0])
             .replace("__PARTNER__", short[1]), "intro"),
        cell("markdown", "## 1. Setup\n", "setup-md"),
        cell("code", CODE_SETUP.replace("__RUN_DIR__", str(run_dir.resolve()))
             .replace("__PARTNER_DIR__", str(partner.resolve())), "setup"),
        cell("markdown", TRANSFORM_MD, "transform-md"),
        cell("code", CODE_TRANSFORM, "transform"),
        cell("markdown", CHOOSE_MD, "choose-md"),
        cell("code", CODE_CHOOSE, "choose"),
        cell("markdown", TEST_MD, "test-md"),
        cell("code", CODE_TEST, "test"),
        cell("markdown", HINDSIGHT_MD, "hindsight-md"),
        cell("code", CODE_HINDSIGHT, "hindsight"),
        cell("markdown", FINDINGS, "findings"),
        cell("markdown", SUMMARY, "summary"),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--partner", type=Path)
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
    if args.partner is None:
        parser.error("--partner is required to build the notebook")
    for d in (args.run_dir, args.partner):
        if not any((d / "predictions").glob("validation_*.parquet")):
            print(f"{d}: no out-of-fold validation predictions")
            return 1
    if path.exists() and not args.force:
        print(f"{path} exists; use --force to rebuild (resets the text cells)")
        return 1
    path.write_text(json.dumps(build(args.run_dir, args.partner), indent=1,
                               ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
