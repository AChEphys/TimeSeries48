LAG, SPAN, HISTORY, MIN_HISTORY = 7, 20, 250, 120
per_day = (
    pl.read_parquet(DATA_FILE, columns=["Date", "label"]).drop_nulls()
    .group_by("Date").agg(pl.col("label").std().alias("std"))
    .sort("Date")
)
REGIMES = ["calm", "normal", "turbulent"]
REGIME_COLORS = {"calm": "#cde2fb", "normal": "#6da7ec", "turbulent": "#184f95"}

per_day = per_day.sort("Date").with_columns(
    pl.col("std").shift(LAG).rolling_mean(SPAN).alias("trail_disp")   # dates T-26 ... T-7
)
td = per_day["trail_disp"].fill_null(np.nan).to_numpy()
pct = np.full(len(td), np.nan)
for i in range(len(td)):
    past = td[max(0, i - HISTORY):i]
    past = past[~np.isnan(past)]
    if len(past) >= MIN_HISTORY and not np.isnan(td[i]):
        pct[i] = (past < td[i]).mean()      # percentile vs the previous ~1 year
per_day = per_day.with_columns(pl.Series("disp_pct", pct).fill_nan(None)).with_columns(
    pl.when(pl.col("disp_pct").is_null()).then(None)
    .when(pl.col("disp_pct") < 1 / 3).then(pl.lit("calm"))
    .when(pl.col("disp_pct") < 2 / 3).then(pl.lit("normal"))
    .otherwise(pl.lit("turbulent")).alias("regime")
)
day_regime = per_day.select("Date", "regime")
daily = daily.drop("regime", strict=False).join(day_regime, on="Date", how="left")
preds = preds.drop("regime", strict=False).join(day_regime, on="Date", how="left")
test_days = (
    daily.group_by("Date").agg(pl.col("rank_ic").mean(), pl.col("regime").first())
    .sort("Date")
)
print("test days without a regime (too little history):", test_days["regime"].null_count())


def episodes(frame: pl.DataFrame, min_len: int = 5) -> pl.DataFrame:
    """Contiguous runs of the same regime over consecutive test days."""
    runs = (
        frame.with_columns((pl.col("regime") != pl.col("regime").shift()).fill_null(True)
                           .cum_sum().alias("run"))
        .group_by("run", maintain_order=True)
        .agg(pl.col("regime").first(), pl.col("Date").first().alias("start"),
             pl.col("Date").last().alias("end"), pl.len().alias("days"))
    )
    return runs.filter(pl.col("days") >= min_len)


rows = []
eps = episodes(test_days.drop_nulls("regime"))
for r in REGIMES:
    d = daily.filter(pl.col("regime") == r)
    ric = d.sort("Date")["rank_ic"].drop_nans().to_numpy()
    rows.append({
        "regime": r, "days": d["Date"].n_unique(),
        "episodes (>=5 days)": eps.filter(pl.col("regime") == r).height,
        "RankIC": ric.mean(), "RankIC 95% CI ±": nw_ci(ric),
        "IC": d["ic"].mean(), "RankIC>0 %": 100 * (ric > 0).mean(),
        "Q5-Q1": d["q5_minus_q1"].mean(),
    })
by_regime = pl.DataFrame(rows)
by_window = (
    daily.drop_nulls("regime").group_by(["window", "regime"])
    .agg(pl.len().alias("days"), pl.col("rank_ic").mean().alias("RankIC"))
    .sort(["window", "regime"])
)
with pl.Config(float_precision=4, tbl_rows=-1):
    display(by_regime)  # noqa: F821
    display(by_window.pivot(on="regime", index="window", values=["days", "RankIC"]))  # noqa: F821
