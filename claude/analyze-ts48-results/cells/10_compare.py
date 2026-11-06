assert COMPARE, "No baseline set: skip this section."

# ---- Align the two runs on common (Date, Code) rows -----------------------------
pair = (
    preds.select("window", "Date", "Code", "label", pl.col("Prediction").alias("p_run"))
    .join(b_preds.select("Date", "Code", pl.col("label").alias("label_base"),
                         pl.col("Prediction").alias("p_base")),
          on=["Date", "Code"], how="inner")
)
label_mismatch = pair.filter((pl.col("label") - pl.col("label_base")).abs() > 1e-6).height
print(f"rows: run {preds.height:,} | baseline {b_preds.height:,} | common {pair.height:,} "
      f"({100 * pair.height / preds.height:.1f}% of run) on {pair['Date'].n_unique()} dates; "
      f"label mismatches: {label_mismatch}")
assert pair.height > 0, "no common rows"
if label_mismatch:
    print("  WARNING: labels differ between runs on some rows (transformed labels?). "
          "The comparison uses the run's labels.")

# ---- Paired daily metrics ------------------------------------------------------------
quint_expr = [
    ((pl.col(p).rank("ordinal").over("Date").cast(pl.Int64) - 1) * 5 // pl.len().over("Date") + 1)
    .alias(f"q_{p[2:]}") for p in ("p_run", "p_base")
]
pair = pair.with_columns(quint_expr)


def q_mean(q_col: str, q: int) -> pl.Expr:
    return pl.col("label").filter(pl.col(q_col) == q).mean()


paired = (
    pair.group_by(["window", "Date"])
    .agg(
        pl.corr("p_run", "label", method="spearman").alias("ric_run"),
        pl.corr("p_base", "label", method="spearman").alias("ric_base"),
        pl.corr("p_run", "label").alias("ic_run"),
        pl.corr("p_base", "label").alias("ic_base"),
        (q_mean("q_run", 5) - q_mean("q_run", 1)).alias("spread_run"),
        (q_mean("q_base", 5) - q_mean("q_base", 1)).alias("spread_base"),
        q_mean("q_run", 1).alias("q1_run"), q_mean("q_base", 1).alias("q1_base"),
        q_mean("q_run", 5).alias("q5_run"), q_mean("q_base", 5).alias("q5_base"),
        pl.corr("p_run", "p_base", method="spearman").alias("pred_agreement"),
    )
    .with_columns(
        (pl.col("ric_run") - pl.col("ric_base")).alias("d_ric"),
        (pl.col("ic_run") - pl.col("ic_base")).alias("d_ic"),
        (pl.col("spread_run") - pl.col("spread_base")).alias("d_spread"),
        pl.col("Date").dt.strftime("%Y-%m").alias("month"),
    )
    .join(day_regime, on="Date", how="left")
    .sort("Date")
)


def delta_row(d: pl.DataFrame, **label) -> dict:
    """Summary of paired daily deltas for one slice of dates."""
    x = d.sort("Date")["d_ric"].drop_nans().to_numpy()
    return {
        **label,
        "days": d.height,
        "RankIC base": d["ric_base"].mean(),
        "RankIC run": d["ric_run"].mean(),
        "dRankIC": x.mean(),
        "95% CI ±": nw_ci(x) if len(x) > 10 else float("nan"),
        "t(NW)": newey_west_t(x) if len(x) > 10 else float("nan"),
        "run better %": 100 * (x > 0).mean(),
        "dIC": d["d_ic"].mean(),
        "dQ5-Q1": d["d_spread"].mean(),
        "Q1 base": d["q1_base"].mean(), "Q1 run": d["q1_run"].mean(),
        "Q5 base": d["q5_base"].mean(), "Q5 run": d["q5_run"].mean(),
        "agreement": d["pred_agreement"].mean(),
    }


overall = delta_row(paired, scope="all")
by_window_delta = pl.DataFrame(
    [delta_row(paired.filter(pl.col("window") == w), scope=f"W{w}")
     for w in sorted(paired["window"].unique())]
)
by_month_delta = (
    paired.group_by("month").agg(
        pl.len().alias("days"), pl.col("d_ric").mean().alias("dRankIC"),
        pl.col("ric_base").mean().alias("RankIC base"), pl.col("ric_run").mean().alias("RankIC run"),
        (100 * (pl.col("d_ric") > 0).mean()).alias("run better %"),
    ).sort("month")
)
by_regime_delta = pl.DataFrame(
    [delta_row(paired.filter(pl.col("regime") == r), scope=r) for r in REGIMES
     if paired.filter(pl.col("regime") == r).height]
)

# ---- Calibration and per-window facts for both models ---------------------------
def slope(frame: pl.DataFrame, p: str) -> pl.Expr:
    return (pl.cov(p, "label") / pl.col(p).var()).alias(f"slope_{p[2:]}")


def r2(p: str) -> pl.Expr:
    return (1 - ((pl.col("label") - pl.col(p)) ** 2).sum()
            / ((pl.col("label") - pl.col("label").mean()) ** 2).sum()).alias(f"R2_{p[2:]}")


def rmse(p: str) -> pl.Expr:
    return ((pl.col("label") - pl.col(p)) ** 2).mean().sqrt().alias(f"RMSE_{p[2:]}")


magnitude_exprs = [slope(pair, "p_base"), slope(pair, "p_run"), r2("p_base"), r2("p_run"),
                   rmse("p_base"), rmse("p_run")]
calib = pair.group_by("window").agg(magnitude_exprs).sort("window")
magnitude_all = pair.select(magnitude_exprs)
b_by_start = {r["test_start"]: r for r in b_results}
facts = pl.DataFrame([{
    "window": k + 1,
    "best_iter base": b_by_start.get(r["test_start"], {}).get("best_iteration"),
    "best_iter run": r.get("best_iteration"),
    "features base": b_by_start.get(r["test_start"], {}).get("features"),
    "features run": r.get("features"),
    "test R2 base": b_by_start.get(r["test_start"], {}).get("r2"),
    "test R2 run": r.get("r2"),
    "runtime base": b_by_start.get(r["test_start"], {}).get("runtime"),
    "runtime run": r.get("runtime"),
} for k, r in enumerate(results)]).join(calib, on="window", how="left")

print(f"\nOVERALL: Rank IC {overall['RankIC base']:.4f} -> {overall['RankIC run']:.4f}  "
      f"dRankIC {overall['dRankIC']:+.4f} ± {overall['95% CI ±']:.4f} (t NW {overall['t(NW)']:.1f}), "
      f"run better on {overall['run better %']:.0f}% of days; dIC {overall['dIC']:+.4f}; "
      f"dQ5-Q1 {overall['dQ5-Q1']:+.4f}; prediction agreement {overall['agreement']:.2f}")
m = magnitude_all.row(0, named=True)
print(f"MAGNITUDES (common rows): calibration slope {m['slope_base']:.3f} -> {m['slope_run']:.3f}; "
      f"R2 {m['R2_base']:.4f} -> {m['R2_run']:.4f}; RMSE {m['RMSE_base']:.5f} -> {m['RMSE_run']:.5f}")
n_pos = (by_window_delta["dRankIC"] > 0).sum()
print(f"windows with dRankIC > 0: {n_pos} of {by_window_delta.height}")
with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=250):
    display(by_window_delta)  # noqa: F821
    display(by_regime_delta)  # noqa: F821
    display(by_month_delta)  # noqa: F821
    display(facts)  # noqa: F821
