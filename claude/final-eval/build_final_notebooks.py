"""
Build the final evaluation's two notebooks (Stage 6, T-20).

- runs/run-final/post-processing.ipynb: setups' raw final-window predictions ->
  each candidate's final-predictions-<candidate>.parquet (and post-processing.json).
- final-evaluation_performances.ipynb (repo root): candidates, metrics, grand table,
  breakdowns, verdict.

Usage: python claude/final-eval/build_final_notebooks.py [--force]

The cell sources are notebook text kept in strings: long lines and the notebook's
Unicode symbols (R², −, …) are exempt from line length and RUF001 for that reason.
It refuses to overwrite an existing notebook without --force (the verdict cell is
written by hand after the run; --force would reset it).
"""
# ruff: noqa: E501, RUF001, RUF002, INP001

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POST_NB = ROOT / "runs/run-final/post-processing.ipynb"
EVAL_NB = ROOT / "final-evaluation_performances.ipynb"
METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

# ---------------------------------------------------------------- shared text
CANDIDATE_SPECS = """CANDIDATES = {  # name -> (members, demeaned per day, weighting); order = table order
    "run-2": (["run-2"], False, "single"),
    "run-5": (["run-5"], True, "single"),
    "run-22": (["run-22"], True, "single"),
    "run-23": (["run-23"], True, "single"),
    "run-11": (["run-11"], True, "single"),
    "run-17": (["run-17"], True, "single"),
    "mix-11-17-equal": (["run-11", "run-17"], True, "equal"),
    "mix-11-17-fitted": (["run-11", "run-17"], True, "fitted"),
    "lagmix-equal": (["run-11", "run-13", "run-14", "run-15", "run-16"], True, "equal"),
    "lagmix-fitted": (["run-11", "run-13", "run-14", "run-15", "run-16"], True, "fitted"),
}"""

# ------------------------------------------------------- post-processing notebook
POST_CELLS = [
    ("md", "intro", """# Post-processing: the candidates' final predictions

This notebook turns the setups' test predictions on the **final window** (trained by
`train_final-window.py`, one subfolder per setup in this folder) into each candidate's
final predictions, `final-predictions-<candidate>.parquet` (columns Date, Code, label,
Prediction), plus `post-processing.json` (weights, rows, dates).

The rules were fixed on 2026-11-05, before any final-window result existed
(`claude/project-notes.md`, T-20):

1. **Per-day demeaning** (each day's mean prediction across stocks is subtracted) for
   every candidate except **Run-2**, the author's baseline, which is scored raw as
   intended.
2. **Mixes** average their members' demeaned predictions on the rows all members share.
   - *Equal weights*: 1/k each.
   - *Fitted weights*: non-negative, summing to 1, minimizing the squared error of the
     mix on the final window's **out-of-fold validation predictions**
     (`predictions/validation_*.parquet`): each history row predicted by the fold model
     that did not train on it, at the tree count the final model uses. History only, so
     no test information enters the weights. Demeaned per day like the test predictions.
3. Nothing else: no rescaling, clipping or shrinkage.

Run it after `train_final-window.py` has finished every setup."""),
    ("code", "setup", """import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

# The notebook lives in runs/run-final/; TS48_FINAL_DIR overrides it (tests).
FINAL_DIR = Path(os.environ.get("TS48_FINAL_DIR", Path.cwd())).resolve()
SETUPS = ["run-2", "run-5", "run-11", "run-13", "run-14", "run-15", "run-16", "run-17",
          "run-22", "run-23"]
""" + CANDIDATE_SPECS + '''

missing = [s for s in SETUPS if not (FINAL_DIR / s / "done.json").exists()]
assert not missing, f"Setups not finished (no done.json): {missing}"


def final_window(setup: str) -> tuple[str, str]:
    """Test start and end of the setup's last (final) window."""
    results = json.loads((FINAL_DIR / setup / "results_lgbm.json").read_text())
    last = max(results, key=lambda r: r["test_start"])
    return last["test_start"], last["test_end"]


windows = {s: final_window(s) for s in SETUPS}
assert len(set(windows.values())) == 1, f"Setups disagree on the final window: {windows}"
TEST_START, TEST_END = windows["run-22"]
print(f"Final window: {TEST_START} -> {TEST_END}; folder {FINAL_DIR}")


def window_file(setup: str, kind: str) -> Path:
    """Return the setup's final-window file of a kind ("predictions" or "validation")."""
    files = [p for p in (FINAL_DIR / setup).rglob(f"{kind}_*.parquet")
             if f"_test_{TEST_START}_" in p.name]
    assert len(files) == 1, (setup, kind, files)
    return files[0]


def demean(frame: pl.DataFrame) -> pl.DataFrame:
    """Subtract each day's mean prediction across stocks."""
    return frame.with_columns(pl.col("Prediction") - pl.col("Prediction").mean().over("Date"))


def combine(frames: list[pl.DataFrame], weights: np.ndarray) -> pl.DataFrame:
    """Weighted sum of the members' predictions on the rows they all share."""
    out = frames[0].select("Date", "Code", "label",
                           (pl.col("Prediction") * weights[0]).alias("Prediction"))
    for frame, w in zip(frames[1:], weights[1:], strict=True):
        out = (out.join(frame.select("Date", "Code", pl.col("Prediction").alias("_p")),
                        on=["Date", "Code"])
               .with_columns(pl.col("Prediction") + w * pl.col("_p")).drop("_p"))
    return out.sort("Date", "Code")'''),
    ("md", "test-md", """## 1. The setups' test predictions on the final window

Rows per setup. The family setups share one universe (raw features and label finite);
Run-2's own feature filter can keep a slightly different set of rows. The evaluation
notebook scores every candidate on the rows they all share."""),
    ("code", "test", """test = {}
for s in SETUPS:
    test[s] = pl.read_parquet(window_file(s, "predictions")).select(
        "Date", "Code", "label", "Prediction")
    df = test[s]
    print(f"{s:7} {df.height:>9,} rows, {df['Date'].n_unique()} dates, "
          f"prediction mean {df['Prediction'].mean():+.5f}, SD {df['Prediction'].std():.5f}")"""),
    ("md", "fit-md", """## 2. Fitted weights (out-of-fold validation predictions, history only)

For each mix: the weights that minimize the squared error of the demeaned mix on the
final window's out-of-fold predictions, under w ≥ 0 and Σw = 1. Shown with each member's
and each mix's out-of-fold Rank IC and R², for reference: these are in-sample for the
fitted weights, so they flatter them.

Caveats noted when the candidates were chosen (T-17): the out-of-fold predictions come
from fold models trained on 80% of the history, and validation favoured the members
closest to lag 1 (N-37)."""),
    ("code", "fit", '''def fit_weights(frames: list[pl.DataFrame]) -> tuple[np.ndarray, int]:
    """Weights >= 0 summing to 1 that minimize the mix's squared error; rows used."""
    joined = frames[0].select("Date", "Code", "label", pl.col("Prediction").alias("p0"))
    for k, frame in enumerate(frames[1:], start=1):
        joined = joined.join(frame.select("Date", "Code", pl.col("Prediction").alias(f"p{k}")),
                             on=["Date", "Code"])
    p = joined.select([f"p{k}" for k in range(len(frames))]).to_numpy().astype(float)
    y = joined["label"].to_numpy().astype(float)
    a, b, k = p.T @ p / len(y), p.T @ y / len(y), len(frames)
    scale = np.trace(a) / k  # objective of order 1, so the optimizer's tolerance bites
    result = minimize(
        lambda w: (w @ a @ w - 2 * b @ w) / scale, np.full(k, 1 / k),
        jac=lambda w: (2 * a @ w - 2 * b) / scale, method="SLSQP", bounds=[(0, 1)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda _w: np.ones(k)}],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert result.success, result.message
    w = np.clip(result.x, 0, None)
    return w / w.sum(), joined.height


def scores(frame: pl.DataFrame) -> tuple[float, float]:
    """Mean daily Rank IC and pooled R² of a (demeaned) prediction frame."""
    ric = frame.group_by("Date").agg(pl.corr(pl.col("Prediction").rank(), pl.col("label").rank()))
    y, p = frame["label"].to_numpy(), frame["Prediction"].to_numpy()
    return float(ric["Prediction"].mean()), float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


oof = {}
fitted = {}
for name, (members, _, how) in CANDIDATES.items():
    if how != "fitted":
        continue
    for m in members:
        oof.setdefault(m, demean(pl.read_parquet(window_file(m, "validation")).select(
            "Date", "Code", "label", "Prediction")))
    frames = [oof[m] for m in members]
    w, rows = fit_weights(frames)
    equal = np.full(len(members), 1 / len(members))
    fitted[name] = {"weights": w, "rows": rows,
                    "dates": [str(oof[members[0]]["Date"].min()), str(oof[members[0]]["Date"].max())]}
    print(f"{name}: fitted on {rows:,} out-of-fold rows "
          f"({fitted[name]['dates'][0]} -> {fitted[name]['dates'][1]})")
    for m, wm in zip(members, w, strict=True):
        ic, r2 = scores(oof[m])
        print(f"   {m:7} weight {wm:.3f} | out-of-fold Rank IC {ic:.4f}, R² {r2:.5f}")
    for label, ww in (("equal", equal), ("fitted", w)):
        ic, r2 = scores(combine(frames, ww))
        print(f"   mix, {label:6} weights: out-of-fold Rank IC {ic:.4f}, R² {r2:.5f}")'''),
    ("md", "save-md", """## 3. The candidates' final predictions

Each candidate's final predictions on the test rows of the final window, saved as
`final-predictions-<candidate>.parquet`."""),
    ("code", "save", """summary = {"test_start": TEST_START, "test_end": TEST_END, "candidates": {}}
for name, (members, demeaned, how) in CANDIDATES.items():
    frames = [demean(test[m]) if demeaned else test[m] for m in members]
    if how == "single":
        w = np.array([1.0])
    elif how == "equal":
        w = np.full(len(members), 1 / len(members))
    else:
        w = fitted[name]["weights"]
    final = combine(frames, w)
    path = FINAL_DIR / f"final-predictions-{name}.parquet"
    final.write_parquet(path)
    summary["candidates"][name] = {
        "members": members, "demeaned": demeaned, "weighting": how,
        "weights": [round(float(x), 6) for x in w], "rows": final.height,
        **({"fitted_on": {k: v for k, v in fitted[name].items() if k != "weights"}}
           if how == "fitted" else {}),
    }
    print(f"{name:17} {final.height:>9,} rows  weights "
          + " ".join(f"{m}={x:.3f}" for m, x in zip(members, w, strict=True)) + f"  -> {path.name}")
(FINAL_DIR / "post-processing.json").write_text(json.dumps(summary, indent=2))
print("wrote post-processing.json")"""),
]

# ---------------------------------------------------------- evaluation notebook
EVAL_INTRO = """# Final evaluation: the candidates on the reserved final window

The one-time test of the project's candidates on the **final window**: the last 84 test
dates (2025-12-19 → 2026-04-28), held back since the start (issue I-6). No model, feature
set or post-processing step was chosen with these dates. This notebook only reads
`runs/run-final/` (trained by `train_final-window.py`, post-processed by
`runs/run-final/post-processing.ipynb`).

## Pre-registration (fixed 2026-11-05, before any final-window result)

- **Primary candidate: run 22** (raw + lags {1, 5} + labels T−6 … T−10 scaled per day),
  the multi-day reference and the best single model on the six test windows.
- **Primary question:** does run 22 beat the baselines here, on ranking without worse
  magnitudes? Compared with **Run-5 (demeaned)** (our best baseline, same protocol) and
  **Run-2** (the author's original baseline, scored raw).
  - *Confirmed*: ΔRank IC > 0 with the 95% Newey–West CI excluding 0, and R² and
    calibration not worse.
  - *Consistent*: ΔRank IC > 0, CI includes 0.
  - *Not confirmed*: ΔRank IC ≤ 0.
- **Power:** on the six windows run 22 beat Run-5 by +0.0104 Rank IC (t 4.4 over 720
  days; t > 2 in 5 of 6 single windows) and Run-2 by +0.0154 (t 6.6). On 84 days the
  expected t is about 1.5 vs Run-5 and 2.2 vs Run-2, so "consistent" is a likely outcome
  even if the gain is real.
- **Secondary candidates** (run 23, the two mixes with equal and fitted weights; runs 11
  and 17 on their own, added later as the 11 + 17 mix's members) are compared with run 22
  descriptively. With eight comparisons on 84 days, none replaces run 22 unless it wins
  clearly on ranking *and* magnitudes, and even then it is flagged as a multiple-comparison
  result.
- **Post-processing** as in the post-processing notebook: per-day demeaning (not Run-2),
  equal or fitted mixing weights; nothing else.

**Statistics.** Daily cross-sectional Rank IC (Spearman) and Pearson IC; Newey–West
standard errors with 4 lags (the 5-day labels overlap); all candidates scored on the rows
they share. R² is pooled over all rows of the window; for demeaned candidates it is the
"R² dm" of the earlier tables, for Run-2 the raw R²."""

EVAL_CANDIDATES_MD = """## The candidates: setup and post-processing

All models: LightGBM, GOSS, 20 leaves, learning rate 0.01, seed 70563 + window offset;
rolling windows of 240 dates (230 history dates, a 10-date gap, then the test dates);
the final window's models were trained on the history up to 2025-12-04. Every setup was
trained with its archived run's exact settings (checked by `train_final-window.py`
before training; window 6 retrained and compared with the archive, section 1).

| Candidate | Model(s) | Features | Validation / tree count | Post-processing | Six test windows (Rank IC / R²) |
|---|---|---|---|---|---|
| **Run-2** (author's baseline) | run 2's original script, untouched | 48 raw features, kept if \\|corr\\| with the label > 0.02 | every 3rd history date validates (leaks through overlapping labels, I-1); early stopping on validation RMSE | **none** (raw) | 0.0581 / 0.00762 (raw) |
| **Run-5** (best baseline) | `train_baseline.py` (run 5) | all 48 raw features | purged 5-fold CV (purge 10), tree count from the fold-mean validation R², retrain on all history | per-day demeaning | 0.0631 / 0.00942 |
| **Run 22** (primary) | family script, `raw+lags+labelraw-6-10-scaled` | 48 raw + the same at T−1 and T−5 (96) + the stock's realized labels of T−6 … T−10 divided by that day's cross-sectional label SD (5): 149 | as run 5 | per-day demeaning | 0.0735 / 0.01064 |
| **Run 23** | family script, `raw+lags+labelraw-6-10-rank` | as run 22, labels as per-day percentile ranks: 149 | as run 5 | per-day demeaning | 0.0733 / 0.01062 |
| **Run 11** (mix member) | family script, `raw+lags` | 48 raw + the same at T−1 and T−5: 144 | as run 5 | per-day demeaning | 0.0674 / 0.00997 |
| **Run 17** (mix member) | family script, `raw+labelraw-6-10` | 48 raw + the stock's raw realized labels of T−6 … T−10: 53 | as run 5 | per-day demeaning | 0.0700 / 0.00978 |
| **Mix 11 + 17, equal** | run 11 (`raw+lags`: raw + T−1, T−5; 144) and run 17 (`raw+labelraw-6-10`: raw + raw labels T−6 … T−10; 53) | — | as run 5 | demean each, average 50/50 | 0.0716 / 0.01065 |
| **Mix 11 + 17, fitted** | runs 11, 17 | — | as run 5 | demean each, weights fitted on the final window's out-of-fold validation predictions (≥ 0, sum 1) | — (new) |
| **Lag mix, equal** | runs 11, 13, 14, 15, 16: raw + lags {1, 5}, {1, 2}, {1, 3}, {1, 4}, {1, 6} (144 each) | — | as run 5 | demean each, average 1/5 | 0.0679 / 0.01028 |
| **Lag mix, fitted** | the same five | — | as run 5 | demean each, fitted weights as above | — (new) |

Label lag: label-based inputs on date t use labels of dates ≤ t − 6 only (realized).

Runs 11 and 17 on their own were added on 2026-11-05 at the user's request, after the
final models were trained, so the mix 11 + 17 can be read against its two members. They
are descriptive, like the other secondary candidates."""

EVAL_CELLS = [
    ("md", "intro", EVAL_INTRO),
    ("md", "candidates", EVAL_CANDIDATES_MD),
    ("code", "setup", """import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import HTML, display

ROOT = Path(os.environ.get("TS48_ROOT", Path.cwd())).resolve()  # the repo root; TS48_* override paths (tests)
FINAL_DIR = Path(os.environ.get("TS48_FINAL_DIR", ROOT / "runs/run-final")).resolve()
DATA_FILE = Path(os.environ.get("TS48_DATA_FILE", ROOT / "data.parquet")).resolve()
PLOTS_DIR = FINAL_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
NW_LAGS = 4
PRIMARY = "run-22"
""" + CANDIDATE_SPECS + '''
LABELS = {
    "run-2": "Run-2 (raw)", "run-5": "Run-5 (dm)", "run-22": "Run 22 (primary)",
    "run-23": "Run 23", "run-11": "Run 11", "run-17": "Run 17", "mix-11-17-equal": "Mix 11+17 equal",
    "mix-11-17-fitted": "Mix 11+17 fitted", "lagmix-equal": "Lag mix equal",
    "lagmix-fitted": "Lag mix fitted",
}
STYLE = {  # fixed per candidate: references grey/dashed, primary blue and thick
    "run-2": ("#8f8e88", "--", 1.5), "run-5": ("#52514e", "--", 1.5),
    "run-22": ("#2a78d6", "-", 3.0), "run-23": ("#1baf7a", "-", 1.5),
    "run-11": ("#e8a33d", "-.", 1.2), "run-17": ("#c4467a", "-.", 1.2),
    "mix-11-17-equal": ("#eb6834", "-", 1.5), "mix-11-17-fitted": ("#eb6834", ":", 1.5),
    "lagmix-equal": ("#6250d6", "-", 1.5), "lagmix-fitted": ("#6250d6", ":", 1.5),
}
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
warnings.filterwarnings("ignore", message=".*non-interactive.*")


def newey_west_se(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """Newey–West (Bartlett) standard error of mean(x)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    e, n = x - x.mean(), len(x)
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(np.sqrt(var / n))


pp = json.loads((FINAL_DIR / "post-processing.json").read_text())
TEST_START, TEST_END = pp["test_start"], pp["test_end"]
preds = {c: pl.read_parquet(FINAL_DIR / f"final-predictions-{c}.parquet") for c in CANDIDATES}
board = preds[PRIMARY].select("Date", "Code", "label")
for c in CANDIDATES:
    board = board.join(preds[c].select("Date", "Code", pl.col("Prediction").alias(c)),
                       on=["Date", "Code"])
board = board.sort("Date", "Code")
print(f"Final window {TEST_START} -> {TEST_END}: {board['Date'].n_unique()} dates, "
      f"{board.height:,} common rows")
for c in CANDIDATES:
    print(f"  {LABELS[c]:18} {preds[c].height:>9,} rows ({board.height / preds[c].height:.1%} common)"
          + "  weights " + ", ".join(f"{m} {w:.3f}" for m, w in
                                     zip(pp["candidates"][c]["members"], pp["candidates"][c]["weights"], strict=True)))'''),
    ("md", "repro-md", """## 1. Checks: the final-window models are the evaluated models

`train_final-window.py` checked every setup's settings against its archived run before
training and retrained window 6 (Run-2: windows 1–6) to compare with the archived
predictions. A correlation ≈ 1 and an unchanged Rank IC mean the code still reproduces
the models that were evaluated on the six windows."""),
    ("code", "repro", """rows = []
for s in sorted({m for c in CANDIDATES.values() for m in c[0]}, key=lambda s: int(s.split("-")[1])):
    done = json.loads((FINAL_DIR / s / "done.json").read_text())
    for r in json.loads((FINAL_DIR / s / "verify.json").read_text()):
        rows.append({"setup": s, "window": r["window"], "rows matched": r["rows_matched"],
                     "corr": r.get("corr"), "max |diff|": r.get("max_abs_diff"),
                     "Rank IC archived": r.get("rank_ic_archived"), "Rank IC now": r.get("rank_ic_new"),
                     "minutes": done["minutes"]})
with pl.Config(tbl_rows=-1, float_precision=5):
    display(pl.DataFrame(rows))"""),
    ("md", "metrics-md", """## 2. Metrics on the final window

Per candidate, on the common rows: daily Rank IC and Pearson IC (mean, ICIR, NW t, share of
positive days), pooled R² and RMSE, calibration slope (regression of the label on the
prediction: 1 = right size, < 1 = too large), daily quintiles of the prediction (mean label
in Q1 / Q5), agreement with run 22, and paired daily ΔRank IC against Run-2, Run-5 and
run 22."""),
    ("code", "metrics", '''def quintile_label(c: str, *, top: bool) -> pl.Expr:
    """Mean label of the day's top (Q5) or bottom (Q1) fifth by prediction."""
    share = pl.col(c).rank() / pl.len()
    return pl.col("label").filter(share > 0.8 if top else share <= 0.2).mean()


daily = board.group_by("Date").agg(
    *[pl.corr(pl.col(c).rank(), pl.col("label").rank()).alias(f"ric|{c}") for c in CANDIDATES],
    *[pl.corr(c, "label").alias(f"ic|{c}") for c in CANDIDATES],
    *[quintile_label(c, top=False).alias(f"q1|{c}") for c in CANDIDATES],
    *[quintile_label(c, top=True).alias(f"q5|{c}") for c in CANDIDATES],
).sort("Date")

y = board["label"].to_numpy().astype(float)
sst = ((y - y.mean()) ** 2).sum()
metrics = {}
for c in CANDIDATES:
    p = board[c].to_numpy().astype(float)
    ric, ic = daily[f"ric|{c}"].to_numpy(), daily[f"ic|{c}"].to_numpy()
    slope = np.cov(p, y)[0, 1] / p.var(ddof=1)
    m = {
        "rank_ic": ric.mean(), "rank_ic_t": ric.mean() / newey_west_se(ric),
        "rank_ic_pos": (ric > 0).mean(), "icir": ric.mean() / ric.std(ddof=1),
        "ic": ic.mean(), "r2": 1 - ((y - p) ** 2).sum() / sst,
        "rmse": np.sqrt(((y - p) ** 2).mean()), "slope": slope, "slope_dev": abs(slope - 1),
        "q1": daily[f"q1|{c}"].mean(), "q5": daily[f"q5|{c}"].mean(),
        "agree": np.corrcoef(p, board[PRIMARY].to_numpy())[0, 1],
    }
    m["q5_q1"] = m["q5"] - m["q1"]
    for ref in ("run-2", "run-5", PRIMARY):
        d = ric - daily[f"ric|{ref}"].to_numpy()
        se = newey_west_se(d)
        m[f"d_{ref}"], m[f"ci_{ref}"] = d.mean(), 1.96 * se
        m[f"t_{ref}"] = d.mean() / se if se > 0 else np.nan
        m[f"pos_{ref}"] = (d > 0).mean()
    metrics[c] = m

table = pl.DataFrame([{"candidate": LABELS[c], **{k: float(v) for k, v in m.items()}}
                      for c, m in metrics.items()])
with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=4, tbl_width_chars=250):
    display(table.select("candidate", "rank_ic", "rank_ic_t", "icir", "rank_ic_pos", "ic",
                         "r2", "rmse", "slope", "q1", "q5", "q5_q1", "agree"))
    display(table.select("candidate", *[f"{k}_{r}" for r in ("run-2", "run-5", PRIMARY)
                                        for k in ("d", "ci", "t", "pos")]))'''),
    ("md", "noise-md", """## 3. Noise yardstick: how much one window moves with the seed alone

From the seed groups of the six test windows (runs 5, 11 and 22 with three seeds each): the
seed-to-seed SD of a single window's Rank IC and R², pooled over groups and windows. The
band 1.96·√2·σ is the 95% range of the difference between two single-seed runs on one
window. Differences inside it can't be told apart from the seed."""),
    ("code", "noise", """SEED_GROUPS = {"5": ["5", "5.1", "5.2"], "11": ["11", "11.1", "11.2"], "22": ["22", "22.1", "22.2"]}
var_ic, var_r2 = [], []
for group in SEED_GROUPS.values():
    per_seed = []
    for tag in group:
        run = next((ROOT / "runs").glob(f"run-{tag}_*"))
        files = sorted((run / "predictions").glob("predictions_*.parquet"))[:6]
        stats = []
        for f in files:
            df = pl.read_parquet(f).with_columns(
                pl.col("Prediction") - pl.col("Prediction").mean().over("Date"))
            ric = df.group_by("Date").agg(pl.corr(pl.col("Prediction").rank(), pl.col("label").rank()))
            yy, pp_ = df["label"].to_numpy(), df["Prediction"].to_numpy()
            stats.append((ric["Prediction"].mean(), 1 - ((yy - pp_) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()))
        per_seed.append(stats)
    arr = np.array(per_seed)  # seeds x windows x (ic, r2)
    var_ic += list(arr[:, :, 0].var(axis=0, ddof=1))
    var_r2 += list(arr[:, :, 1].var(axis=0, ddof=1))
BAND_IC = 1.96 * np.sqrt(2) * np.sqrt(np.mean(var_ic))
BAND_R2 = 1.96 * np.sqrt(2) * np.sqrt(np.mean(var_r2))
print(f"One window, two single-seed runs: Rank IC ±{BAND_IC:.4f}, R² ±{BAND_R2:.5f} "
      f"(pooled over {len(var_ic)} group-windows)")"""),
    ("md", "grand-md", """## 4. Grand table: final window only

Common rows, final window. Dark green: best in the row (at the shown precision); light
green: within the one-window seed band of the best (Rank IC, Pearson IC, R²). Run-2 is raw;
the others are demeaned per day. Paired rows: the candidate minus the reference, per day."""),
    ("code", "grand", '''ROWS = [  # (section, label, key, format, best: max / min / one / None, band)
    ("Ranking", "Rank IC", "rank_ic", "{:.4f}", "max", "ic"),
    ("Ranking", "Rank IC, NW t", "rank_ic_t", "{:.1f}", None, None),
    ("Ranking", "ICIR (daily mean / SD)", "icir", "{:.2f}", "max", None),
    ("Ranking", "Days with Rank IC > 0", "rank_ic_pos", "{:.0%}", "max", None),
    ("Ranking", "Pearson IC", "ic", "{:.4f}", "max", "ic"),
    ("Paired vs Run-2", "ΔRank IC", "d_run-2", "{:+.4f}", "max", None),
    ("Paired vs Run-2", "95% CI ±", "ci_run-2", "{:.4f}", None, None),
    ("Paired vs Run-2", "t (NW)", "t_run-2", "{:+.1f}", None, None),
    ("Paired vs Run-2", "Days better", "pos_run-2", "{:.0%}", "max", None),
    ("Paired vs Run-5", "ΔRank IC", "d_run-5", "{:+.4f}", "max", None),
    ("Paired vs Run-5", "95% CI ±", "ci_run-5", "{:.4f}", None, None),
    ("Paired vs Run-5", "t (NW)", "t_run-5", "{:+.1f}", None, None),
    ("Paired vs Run-5", "Days better", "pos_run-5", "{:.0%}", "max", None),
    ("Paired vs run 22", "ΔRank IC", f"d_{PRIMARY}", "{:+.4f}", None, None),
    ("Paired vs run 22", "t (NW)", f"t_{PRIMARY}", "{:+.1f}", None, None),
    ("Paired vs run 22", "Agreement (corr of predictions)", "agree", "{:.3f}", None, None),
    ("Magnitudes", "R²", "r2", "{:.5f}", "max", "r2"),
    ("Magnitudes", "RMSE", "rmse", "{:.5f}", "min", None),
    ("Magnitudes", "Calibration slope", "slope", "{:.2f}", "one", None),
    ("Magnitudes", "Q1 mean label", "q1", "{:+.5f}", "min", None),
    ("Magnitudes", "Q5 mean label", "q5", "{:+.5f}", "max", None),
    ("Magnitudes", "Q5 − Q1", "q5_q1", "{:.5f}", "max", None),
]
DARK, LIGHT = "rgba(46,160,67,0.28)", "rgba(46,160,67,0.12)"
BANDS = {"ic": BAND_IC, "r2": BAND_R2}
SCORE = {"max": lambda v: v, "min": lambda v: -v, "one": lambda v: -abs(v - 1)}


def shown(value: float, fmt: str) -> float:
    """Return the value as displayed (ties at the shown precision count as ties)."""
    text = fmt.format(value)
    return float(text.rstrip("%")) / 100 if text.endswith("%") else float(text)


th = "text-align:right;padding:2px 8px"
html = ['<table style="border-collapse:collapse;font-size:0.92em"><tr><th style="text-align:left">Metric</th>']
html += [f'<th style="{th}">{LABELS[c]}</th>' for c in CANDIDATES]
html.append("</tr>")
section = None
for sec, label, key, fmt, best, band in ROWS:
    if sec != section:
        html.append(f'<tr><td colspan="{len(CANDIDATES) + 1}" style="font-weight:bold;padding-top:8px">{sec}</td></tr>')
        section = sec
    own_ref = key.split("_", 1)[1] if key.startswith(("d_", "ci_", "t_", "pos_")) else None
    values = {c: metrics[c][key] for c in CANDIDATES if c != own_ref}
    winners, top_value = set(), None
    if best:
        scores = {c: SCORE[best](shown(v, fmt)) for c, v in values.items()}
        top = max(scores.values())
        winners = {c for c, sc in scores.items() if sc == top}
        top_value = values[next(iter(winners))]
    html.append(f'<tr><td style="padding:2px 8px">{label}</td>')
    for c in CANDIDATES:
        style = th
        if c in winners:
            style += f";background:{DARK};font-weight:bold"
        elif band and c in values and abs(values[c] - top_value) <= BANDS[band]:
            style += f";background:{LIGHT}"
        text = "—" if c == own_ref else fmt.format(values[c])
        html.append(f'<td style="{style}">{text}</td>')
    html.append("</tr>")
html.append("</table>")
display(HTML("".join(html)))
print(f"Seed band, one window: Rank IC ±{BAND_IC:.4f}, R² ±{BAND_R2:.5f}")'''),
    ("md", "months-md", """## 5. Per month

Rank IC, R² and calibration slope per calendar month (December 2025 holds only the window's
first days). Months are short (about 20 dates, a handful of independent 5-day labels), so
read them as patterns, not tests."""),
    ("code", "months", """board_m = board.with_columns(pl.col("Date").dt.strftime("%Y-%m").alias("month"))
daily_m = daily.with_columns(pl.col("Date").dt.strftime("%Y-%m").alias("month"))
ric_month = daily_m.group_by("month").agg(
    pl.len().alias("dates"), *[pl.col(f"ric|{c}").mean().alias(LABELS[c]) for c in CANDIDATES]
).sort("month")
mag = []
for (month,), g in board_m.group_by("month"):
    yy = g["label"].to_numpy().astype(float)
    row = {"month": month}
    for c in CANDIDATES:
        p = g[c].to_numpy().astype(float)
        row[f"R² {LABELS[c]}"] = 1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        row[f"slope {LABELS[c]}"] = np.cov(p, yy)[0, 1] / p.var(ddof=1)
    mag.append(row)
mag = pl.DataFrame(mag).sort("month")
with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=4, tbl_width_chars=250):
    print("Rank IC per month")
    display(ric_month)
    print("R² per month")
    display(mag.select("month", *[f"R² {LABELS[c]}" for c in CANDIDATES]))
    print("Calibration slope per month")
    display(mag.select("month", *[f"slope {LABELS[c]}" for c in CANDIDATES]))"""),
    ("md", "regimes-md", """## 6. Ex-ante regimes

The analysis notebooks' regime rule: each day's trailing cross-sectional label dispersion
(dates T−26 … T−7, known on the day) ranked against the previous ~250 days: calm (lowest
third), normal, turbulent (highest third). With 84 dates, count the episodes before
reading anything into a regime."""),
    ("code", "regimes", """LAG, SPAN, HISTORY, MIN_HISTORY = 7, 20, 250, 120
per_day = (pl.read_parquet(DATA_FILE, columns=["Date", "label"]).drop_nulls()
           .group_by("Date").agg(pl.col("label").std().alias("std")).sort("Date")
           .with_columns(pl.col("std").shift(LAG).rolling_mean(SPAN).alias("trail")))
td = per_day["trail"].fill_null(np.nan).to_numpy()
pct = np.full(len(td), np.nan)
for i in range(len(td)):
    past = td[max(0, i - HISTORY):i]
    past = past[~np.isnan(past)]
    if len(past) >= MIN_HISTORY and not np.isnan(td[i]):
        pct[i] = (past < td[i]).mean()
per_day = per_day.with_columns(pl.Series("pct", pct).fill_nan(None)).with_columns(
    pl.when(pl.col("pct").is_null()).then(None)
    .when(pl.col("pct") < 1 / 3).then(pl.lit("calm"))
    .when(pl.col("pct") < 2 / 3).then(pl.lit("normal"))
    .otherwise(pl.lit("turbulent")).alias("regime"))
daily_r = daily.join(per_day.select("Date", "regime"), on="Date", how="left")
runs_ = (daily_r.with_columns((pl.col("regime") != pl.col("regime").shift()).fill_null(True)
                              .cum_sum().alias("run"))
         .group_by("run", maintain_order=True).agg(pl.col("regime").first(), pl.len().alias("days")))
rows = []
for r in ["calm", "normal", "turbulent"]:
    g = daily_r.filter(pl.col("regime") == r)
    row = {"regime": r, "dates": g.height,
           "episodes (>= 5 dates)": runs_.filter((pl.col("regime") == r) & (pl.col("days") >= 5)).height}
    for c in CANDIDATES:
        row[LABELS[c]] = g[f"ric|{c}"].mean() if g.height else None
    if g.height > 5:
        for ref in ("run-5", "run-2"):
            d = (g[f"ric|{PRIMARY}"] - g[f"ric|{ref}"]).to_numpy()
            row[f"run 22 − {ref} (t)"] = f"{d.mean():+.4f} ({d.mean() / newey_west_se(d):+.1f})"
    rows.append(row)
with pl.Config(tbl_cols=-1, float_precision=4, tbl_width_chars=250):
    display(pl.DataFrame(rows))
print("dates without a regime:", daily_r["regime"].null_count())"""),
    ("md", "plots-md", """## 7. Over time

Top: cumulative daily Rank IC (the slope is the IC level). Bottom: cumulative daily
difference of run 22 against the two baselines (rising = run 22 ahead)."""),
    ("code", "plots", """dates = daily["Date"].to_numpy()
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for c in CANDIDATES:
    color, ls, lw = STYLE[c]
    axes[0].plot(dates, daily[f"ric|{c}"].cum_sum(), color=color, ls=ls, lw=lw, label=LABELS[c])
axes[0].set_ylabel("cumulative Rank IC")
axes[0].legend(ncol=4, fontsize=9, loc="upper left")
for ref in ("run-2", "run-5"):
    color, ls, _ = STYLE[ref]
    axes[1].plot(dates, (daily[f"ric|{PRIMARY}"] - daily[f"ric|{ref}"]).cum_sum(), color=color,
                 ls=ls, lw=2, label=f"run 22 − {LABELS[ref]}")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_ylabel("cumulative ΔRank IC")
axes[1].legend(fontsize=9, loc="upper left")
fig.suptitle(f"Final window {TEST_START} → {TEST_END}")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "final_cumulative.png", dpi=120, bbox_inches="tight")
plt.show()"""),
    ("md", "context-md", """## 8. Context: the final window against the six test windows

Each candidate's Rank IC and R² on the final window next to its level on the six test
windows (all 720 dates) and its range over the six single windows: does the final window
fall inside the range seen before? Fitted mixes have no six-window value (their weights
were only fitted for the final window)."""),
    ("code", "context", '''def six_window_frames(setup: str, *, demeaned: bool) -> list[pl.DataFrame]:
    """Load the archived run's six test windows' predictions (demeaned if asked)."""
    run = next((ROOT / "runs").glob(f"{setup}_*"))
    files = sorted((run / "predictions").glob("predictions_*.parquet")) or sorted(run.glob("predictions_*.parquet"))
    out = []
    for f in files[:6]:
        df = pl.read_parquet(f).select("Date", "Code", "label", "Prediction")
        out.append(df.with_columns(pl.col("Prediction") - pl.col("Prediction").mean().over("Date"))
                   if demeaned else df)
    return out


def level(frames: list[pl.DataFrame]) -> tuple[float, float, list[float]]:
    """Rank IC over all dates, pooled R², per-window Rank IC."""
    per_window, all_ric, sse, ys = [], [], 0.0, []
    for df in frames:
        ric = df.group_by("Date").agg(pl.corr(pl.col("Prediction").rank(), pl.col("label").rank()))["Prediction"]
        per_window.append(ric.mean())
        all_ric.append(ric)
        ys.append(df["label"].to_numpy())
        sse += ((df["label"].to_numpy() - df["Prediction"].to_numpy()) ** 2).sum()
    yy = np.concatenate(ys)
    return pl.concat(all_ric).mean(), 1 - sse / ((yy - yy.mean()) ** 2).sum(), per_window


rows = []
cache = {}
for c, (members, demeaned, how) in CANDIDATES.items():
    row = {"candidate": LABELS[c], "final Rank IC": metrics[c]["rank_ic"], "final R²": metrics[c]["r2"]}
    if how != "fitted":
        frames_by_member = [cache.setdefault((m, demeaned), six_window_frames(m, demeaned=demeaned)) for m in members]
        mixed = []
        for w in range(6):
            df = frames_by_member[0][w].select("Date", "Code", "label",
                                               (pl.col("Prediction") / len(members)).alias("Prediction"))
            for fm in frames_by_member[1:]:
                df = (df.join(fm[w].select("Date", "Code", pl.col("Prediction").alias("_p")), on=["Date", "Code"])
                      .with_columns(pl.col("Prediction") + pl.col("_p") / len(members)).drop("_p"))
            mixed.append(df)
        ic6, r26, per_w = level(mixed)
        row |= {"six-window Rank IC": ic6, "six-window R²": r26,
                "single-window Rank IC range": f"{min(per_w):.4f} … {max(per_w):.4f}",
                "final inside range": "yes" if min(per_w) <= metrics[c]["rank_ic"] <= max(per_w) else "no"}
    rows.append(row)
with pl.Config(tbl_cols=-1, float_precision=4, tbl_width_chars=250):
    display(pl.DataFrame(rows))'''),
    ("md", "verdict", """## Verdict

*To be written after the final run, against the pre-registered criteria at the top.*"""),
]


def cell(kind: str, cid: str, text: str) -> dict:
    """One notebook cell (outputs empty; the user runs the notebook in VS Code)."""
    lines = text.splitlines(keepends=True)
    base = {"cell_type": "markdown" if kind == "md" else "code", "id": cid,
            "metadata": {}, "source": lines}
    if kind == "code":
        base |= {"execution_count": None, "outputs": []}
    return base


def write(path: Path, cells: list[tuple[str, str, str]], *, force: bool) -> None:
    """Write a notebook unless it exists (then only with force)."""
    if path.exists() and not force:
        print(f"exists, not overwritten (use --force): {path}")  # noqa: T201
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [cell(*c) for c in cells], "metadata": METADATA,
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")  # noqa: T201


def main() -> int:
    """Build both notebooks (paths overridable for tests)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--post", type=Path, default=POST_NB)
    parser.add_argument("--eval", type=Path, default=EVAL_NB)
    args = parser.parse_args()
    write(args.post, POST_CELLS, force=args.force)
    write(args.eval, EVAL_CELLS, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
