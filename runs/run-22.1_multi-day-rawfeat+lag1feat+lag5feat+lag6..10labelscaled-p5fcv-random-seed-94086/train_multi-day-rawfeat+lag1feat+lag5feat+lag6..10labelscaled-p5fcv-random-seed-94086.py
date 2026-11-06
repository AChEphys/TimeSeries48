#!/usr/bin/env python3
"""
Train rolling LightGBM models on multi-day feature sets.

Feature family "multi-day" (stage 5, T-12 (b)/(c) in claude/project-notes.md): on top
of the 48 raw features, each stock gets columns from its own past, using only what is
known on the prediction date. Column groups (see build_features), in the four
categories of N-34:
- raw historical features:     lags (t-1, t-5), lags-K1-K2-... (any lags);
- derived historical features: dev (deviation from the trailing week / month mean),
                               fmean-A-B (mean over dates t-B ... t-A);
- raw historical labels:       labelraw-A-B (the realized labels of t-A ... t-B);
- derived historical labels:   labelstats-max / -min (trailing means, SD, max or min);
- both label kinds:            labelhist-max / -min (= labelstats plus the latest
                               label; runs 8.1 / 8.2).
Label groups take a suffix -rank (each column as its percentile among the day's
stocks) or -scaled (labels divided by their day's cross-sectional SD first), which
remove the market-wide level shifts after shocks (N-33).
Everything else is the baseline's: train_baseline.py is imported as a library for
the validation protocol (purged k-fold, retrain), LightGBM parameters, curve monitor
and helpers. This script only adds its head, the feature sets (`FEATURE_SETS`,
`build_features`) and a `train()` that builds them; otherwise `train()` mirrors
train_baseline.train().

Rows are the baseline's: only the raw features and the label must be finite. History
columns may be missing (new listings, suspensions, the first dates of the data);
LightGBM treats them as missing values.

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
RUN_DIR = ROOT / "runs/run-22.1_multi-day-rawfeat+lag1feat+lag5feat+lag6..10labelscaled-p5fcv-random-seed-94086"  # NOTE: Set per run.  # noqa: E501
RESULT_FILE = RUN_DIR / "results_lgbm.json"
MODEL_DIR = RUN_DIR / "models"
PREDICTIONS_DIR = RUN_DIR / "predictions"
CURVES_DIR = RUN_DIR / "curves"
ARXIV_FILE = RUN_DIR / "train_multi-day-rawfeat+lag1feat+lag5feat+lag6..10labelscaled-p5fcv-random-seed-94086.py"  # NOTE: Archive copy.  # noqa: E501
CURVE_FLUSH_EVERY = 10  # NOTE: Write learning curves to disk every N iterations.
PLOT_REFRESH_SECONDS = 3  # NOTE: Refresh the learning-curve plot every N seconds.

RANDOM_SEED = 94086
SMOKE_TEST = False  # NOTE: If True, only the first rolling window is trained.
SKIP_LAST_WINDOW = True  # NOTE: The last window is reserved for the final evaluation.
FEATURE_SET = "raw+lags+labelraw-6-10-scaled"  # NOTE: A FEATURE_SETS key.

CORR_BAR = 0.02  # NOTE: Protocol settings as in train_baseline (run 5); share_head()
LOOKBACK = 240  # NOTE: copies the ones its helpers use into train_baseline.
GAP = 10
TEST_DAYS = 120
VALIDATION_FOLD = 5
PURGE = GAP
SELECT_FEATURES = False # NOTE: If True, filter features based on correlation. False = use all features.
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 200

LABEL_LAG = 6  # NOTE: On date t, labels of dates <= t - 6 are realized (user, 2026-10-25).
LABEL_MEANS = (20, 60)  # NOTE: Trailing label means over these many dates (month, quarter).
LABEL_SPREAD = 20  # NOTE: Trailing label SD and max / min over these many dates.
DEV_WINDOWS = (5, 20)  # NOTE: Feature deviations from the mean of dates t-n ... t-1.
LAGS = (1, 5)  # NOTE: Feature values n dates back (group "lags"; run 11).
STALE_LIMIT = 20  # NOTE: Label history older than this many dates (suspension) is dropped.

SHARED = (  # NOTE: Head settings that train_baseline's helpers read from their module.
    "RUN_DIR", "MODEL_DIR", "PREDICTIONS_DIR", "CURVES_DIR", "CURVE_FLUSH_EVERY",
    "CORR_BAR", "VALIDATION_FOLD", "PURGE", "SELECT_FEATURES", "MAX_ROUNDS",
    "EARLY_STOPPING_ROUNDS",
)

FEATURE_SETS = {  # NOTE: Feature set -> its column groups (see build_features).
    "raw": ("raw",),
    "raw+labelhist-max": ("raw", "labelhist-max"),  # NOTE: Lottery (MAX) effect.
    "raw+labelhist-min": ("raw", "labelhist-min"),  # NOTE: Same with the downside extreme.
    "raw+dev": ("raw", "dev"),
    "raw+labelhist-max+dev": ("raw", "labelhist-max", "dev"),
    "raw+labelhist-min+dev": ("raw", "labelhist-min", "dev"),
    "raw+lags": ("raw", "lags"),  # NOTE: Lags 1 and 5 (run 11).
    # NOTE: Raw / level-keeping feature history (N-34 H1/H2; validation trusted, sweep).
    "raw+lags-1": ("raw", "lags-1"),
    "raw+lags-1-2": ("raw", "lags-1-2"),
    "raw+lags-1-2-3": ("raw", "lags-1-2-3"),
    "raw+lags-1-3": ("raw", "lags-1-3"),  # NOTE: {1, x}: does lag 5 (≈ label horizon)
    "raw+lags-1-4": ("raw", "lags-1-4"),  # NOTE: beat its neighbours? (N-37)
    "raw+lags-1-6": ("raw", "lags-1-6"),
    "raw+lags-1-2-3-4-5": ("raw", "lags-1-2-3-4-5"),
    "raw+lags-1-5-10": ("raw", "lags-1-5-10"),
    "raw+lags-1-5-20": ("raw", "lags-1-5-20"),
    "raw+lags-1-2-5-10-20": ("raw", "lags-1-2-5-10-20"),
    "raw+fmean-1-5+fmean-6-20": ("raw", "fmean-1-5", "fmean-6-20"),
    "raw+lags-1+fmean-2-5+fmean-6-20": ("raw", "lags-1", "fmean-2-5", "fmean-6-20"),
    # NOTE: Label history split and normalized (N-34 H3/H4; full runs).
    "raw+labelraw-6-10": ("raw", "labelraw-6-10"),
    "raw+labelstats-max": ("raw", "labelstats-max"),
    "raw+labelhist-max-rank": ("raw", "labelhist-max-rank"),
    "raw+labelhist-max-scaled": ("raw", "labelhist-max-scaled"),
    # NOTE: Lags {1, 5} (run 11) + raw labels (run 17), T-16 step 3 (N-39).
    "raw+lags+labelraw-6-10": ("raw", "lags", "labelraw-6-10"),
    "raw+lags+labelraw-6-10-scaled": ("raw", "lags", "labelraw-6-10-scaled"),
    "raw+lags+labelraw-6-10-rank": ("raw", "lags", "labelraw-6-10-rank"),
}


def finite(column: str) -> pl.Expr:
    """Return the column with non-finite values as null (missing)."""
    return pl.when(pl.col(column).is_finite()).then(pl.col(column))


def label_values(*, scaled: bool) -> pl.Expr:
    """Return the realized label (non-finite: null); / its day's SD if scaled."""
    label = finite("label")
    return label / label.std().over("Date") if scaled else label


def label_history(frame: pl.DataFrame, *, scaled: bool = False) -> pl.DataFrame:
    """
    Per (Code, date index d): the stock's realized label statistics known on date d.

    Statistics over the stock's labels of dates s - n + 1 ... s, counted in trading
    dates (not rows), are known from date s + LABEL_LAG on. The as-of join gives each
    row the latest statistics known by then, so a stock that didn't trade on
    d - LABEL_LAG still gets its earlier ones (up to STALE_LIMIT dates old).
    - labelhist_last:     the label of date s (the latest realized one);
    - labelhist_meanN:    mean over N dates (LABEL_MEANS);
    - labelhist_stdN / labelhist_maxN / labelhist_minN: SD, max and min over
      N = LABEL_SPREAD dates (signed: max is the best one, min the worst).
    With scaled=True, each label is first divided by the SD of all stocks' labels on
    its date (known with the label), and the names end in "_scaled".
    """
    n = LABEL_SPREAD
    end = "_scaled" if scaled else ""
    known = frame.select("Code", "d", label_values(scaled=scaled).alias("y"))
    label = pl.col("y")

    def rolling(stat: str, size: int) -> pl.Expr:
        window = {"window_size": f"{size}i", "min_samples": size // 2}
        return getattr(label, f"rolling_{stat}_by")("d", **window).over("Code")

    known = (
        known.sort("Code", "d")
        .select(
            "Code",
            (pl.col("d") + LABEL_LAG).alias("d"),
            label.alias(f"labelhist_last{end}"),
            *[rolling("mean", m).alias(f"labelhist_mean{m}{end}") for m in LABEL_MEANS],
            rolling("std", n).alias(f"labelhist_std{n}{end}"),
            rolling("max", n).alias(f"labelhist_max{n}{end}"),
            rolling("min", n).alias(f"labelhist_min{n}{end}"),
        )
        .sort("d")
    )
    return frame.select("Code", "d").sort("d").join_asof(
        known, on="d", by="Code", strategy="backward",
        tolerance=STALE_LIMIT - LABEL_LAG,
        check_sortedness=False,  # NOTE: Both sides are sorted by d just above.
    )


def numbers(group: str) -> list[int]:
    """'lags-1-2-5' -> [1, 2, 5]; 'fmean-6-20' -> [6, 20]."""
    return [int(x) for x in group.split("-")[1:]]


def split_modifier(group: str) -> tuple[str, str]:
    """'labelhist-max-rank' -> ('labelhist-max', 'rank'); no suffix -> ''."""
    for modifier in ("rank", "scaled"):
        if group.endswith(f"-{modifier}"):
            return group.removesuffix(f"-{modifier}"), modifier
    return group, ""


def warmup(kind: str) -> int:
    """Dates of history the feature set needs before a window's first date."""
    need = [0]
    for group in FEATURE_SETS[kind]:
        base, _ = split_modifier(group)
        if base == "dev":
            need.append(max(DEV_WINDOWS))
        elif base == "lags":
            need.append(max(LAGS))
        elif base.startswith(("lags-", "fmean-", "labelraw-")):
            need.append(max(numbers(base)))
        elif base.startswith(("labelhist-", "labelstats-")):
            need.append(LABEL_LAG + max(LABEL_MEANS))
    return max(need)


def per_day_rank(columns: list[str]) -> list[pl.Expr]:
    """Each column as its percentile among the day's stocks, in (0, 1] (nulls stay)."""
    return [(pl.col(c).rank() / pl.col(c).count()).over("Date").cast(pl.Float32)
            .alias(f"{c}_rank") for c in columns]


def label_group(
    out: pl.DataFrame, base: str, modifier: str
) -> tuple[pl.DataFrame, list[str]]:
    """Add one label group (labelhist-*, labelstats-*, labelraw-A-B) to `out`."""
    scaled = modifier == "scaled"
    if base.startswith("labelraw-"):
        first, last = numbers(base)
        if first < LABEL_LAG:  # NOTE: Labels of later dates aren't realized on date t.
            msg = f"{base}: labels before t - {LABEL_LAG} are not realized yet"
            raise ValueError(msg)
        end = "_scaled" if scaled else ""
        values = out.select("Code", "d", label_values(scaled=scaled).alias("y"))
        added = []
        for k in range(first, last + 1):
            name = f"labelraw_lag{k}{end}"
            lagged = values.select("Code", (pl.col("d") + k).alias("d"),
                                   pl.col("y").cast(pl.Float32).alias(name))
            out = out.join(lagged, on=["Code", "d"], how="left")
            added.append(name)
    else:
        kind, extreme = base.split("-")
        drop = "min" if extreme == "max" else "max"
        history = label_history(out, scaled=scaled)
        history = history.drop([c for c in history.columns
                                if c.startswith(f"labelhist_{drop}")
                                or (kind == "labelstats"
                                    and c.startswith("labelhist_last"))])
        added = [c for c in history.columns if c.startswith("labelhist_")]
        history = history.with_columns([pl.col(c).cast(pl.Float32) for c in added])
        out = out.join(history, on=["Code", "d"], how="left")
    if modifier == "rank":
        out = out.with_columns(per_day_rank(added)).drop(added)
        added = [f"{c}_rank" for c in added]
    return out, added


def group_order(group: str) -> int:
    """
    Column order of the groups: dev, lags, fmean, labels (stable within a kind).

    The order of the earlier script (runs 8-11), so their columns, and with them
    LightGBM's split tie-breaking, are reproduced exactly.
    """
    prefixes = ("raw", "dev", "lags", "fmean-", "label")
    for rank, prefix in enumerate(prefixes):
        if group.startswith(prefix):
            return rank
    return len(prefixes)


def feature_group(
    out: pl.DataFrame, base: str, raw: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    """Add one feature-history group (dev, lags, lags-K..., fmean-A-B) to `out`."""
    if base == "dev":
        exprs = []
        for k in DEV_WINDOWS:
            for f in raw:
                past = finite(f).rolling_mean_by(
                    "d", window_size=f"{k}i", closed="left", min_samples=(k + 1) // 2
                ).over("Code")
                exprs.append((finite(f) - past).cast(pl.Float32).alias(f"{f}_dev{k}"))
        return out.with_columns(exprs), [e.meta.output_name() for e in exprs]
    added: list[str] = []
    if base.startswith("fmean-"):
        first, last = numbers(base)
        size = last - first + 1
        means = out.rolling(
            index_column="d", period=f"{size}i", offset=f"-{last + 1}i",
            closed="right", group_by="Code",
        ).agg([
            pl.when(finite(f).count() >= (size + 1) // 2).then(finite(f).mean())
            .cast(pl.Float32).alias(f"{f}_mean{first}to{last}")
            for f in raw
        ])
        return (out.join(means, on=["Code", "d"], how="left"),
                [f"{f}_mean{first}to{last}" for f in raw])
    for k in LAGS if base == "lags" else numbers(base):
        lagged = out.select(
            "Code", (pl.col("d") + k).alias("d"),
            *[finite(f).alias(f"{f}_lag{k}") for f in raw],
        )
        out = out.join(lagged, on=["Code", "d"], how="left")
        added += [f"{f}_lag{k}" for f in raw]
    return out, added


def build_features(frame: pl.DataFrame, kind: str) -> tuple[pl.DataFrame, list[str]]:
    """
    Add feature set `kind` to whole days of data; return the model input names.

    `frame` holds every row of a span of consecutive trading dates, with the
    trading-date index `d` (0, 1, 2, ... over all dates of the data). Statistics are
    computed per stock over trading dates, so missing rows (suspensions) shorten a
    window instead of reaching further back. Column groups:
    - raw:          featureN as it is.
    - dev:          featureN_devK = featureN - mean of featureN over dates t-K ... t-1
                    (K in DEV_WINDOWS; at least half of those dates present).
    - lags:         featureN_lagK = featureN on date t-K (K in LAGS);
      lags-K1-K2-…: the same for the listed K.
    - fmean-A-B:    featureN_meanAtoB = mean of featureN over dates t-B ... t-A (at
                    least half of them present).
    - labelhist-max / labelhist-min: see label_history(): latest label, means, SD,
                    and max or min; labels of dates <= t - LABEL_LAG only.
    - labelstats-max / labelstats-min: the same without the latest label.
    - labelraw-A-B: labelraw_lagK = the stock's label of date t-K, K = A ... B
                    (A >= LABEL_LAG).
    - label groups + "-rank": each column replaced by its percentile among the day's
                    stocks; + "-scaled": labels divided by their day's SD first.
    Nothing uses a later date's features or a label that isn't realized on date t.
    The first warmup(kind) dates of `frame` lack (part of) their history: drop them.
    """
    if kind not in FEATURE_SETS:
        msg = f"Unknown feature set {kind!r}; choose from {list(FEATURE_SETS)}"
        raise ValueError(msg)
    raw = [c for c in frame.columns if c.startswith("feature") and "_" not in c]
    out = frame.sort("Code", "d")
    names = list(raw)
    for group in sorted(FEATURE_SETS[kind], key=group_order):  # NOTE: Fixed column order.
        base, modifier = split_modifier(group)
        if base == "raw":
            continue
        if base.startswith(("labelhist-", "labelstats-", "labelraw-")):
            out, added = label_group(out, base, modifier)
        elif base in ("dev", "lags") or base.startswith(("lags-", "fmean-")):
            out, added = feature_group(out, base, raw)
        else:
            msg = f"Unknown column group {group!r} in feature set {kind!r}"
            raise ValueError(msg)
        names += added
    return out.sort("Date", "Code"), names


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

    The only differences: per window, the features are built once over the window's
    dates plus the warmup(FEATURE_SET) dates before it (history needs earlier
    dates); rows are filtered as in the baseline (raw features and label finite), so
    history columns may be missing; and the results record the feature set.
    """
    data = pl.read_parquet(DATA_FILE).sort(["Date", "Code"])
    dates = data["Date"].unique().sort().to_list()
    data = data.with_columns(pl.col("Date").rank("dense").cast(pl.Int64).alias("d"))
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

        first = max(begin - warmup(FEATURE_SET), 0)
        span = data.filter(pl.col("Date").is_in(dates[first:end]))
        window, features = build_features(span, FEATURE_SET)
        del span
        raw = [f for f in features if "_" not in f]
        history = tb.finite_rows(
            window.filter(pl.col("Date").is_in(history_dates)), [*raw, "label"]
        )
        params = tb.lgbm_params(RANDOM_SEED + i)
        window_tag = (f"train_{dates[begin]}_{dates[i - GAP - 1]}_"
                      f"test_{dates[i]}_{dates[end - 1]}")
        cv = tb.purged_cv(history, history_dates, features, params, window_tag)
        best_iteration = cv.best_iteration

        selected = tb.select_features(history, features)
        test_df = tb.finite_rows(
            window.filter(pl.col("Date").is_in(test_dates)), [*raw, "label"]
        )
        del window

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
        missing = {  # NOTE: Share of history rows without each history column.
            f: round(history[f].null_count() / len(history), 4)
            for f in features
            if "_" in f
        }

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
            "history_missing_share": missing,
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

        del test_df, test_x, model, scored, prediction, real, ic, valid_ic
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
