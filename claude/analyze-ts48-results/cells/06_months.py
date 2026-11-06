def month_stats(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Pooled R² and calibration slope per group of rows."""
    return frame.group_by(by).agg(
        (1 - ((pl.col("label") - pl.col("Prediction")) ** 2).sum()
         / ((pl.col("label") - pl.col("label").mean()) ** 2).sum()).alias("R2"),
        (pl.cov("Prediction", "label") / pl.col("Prediction").var()).alias("slope"),
    )


preds = preds.with_columns(pl.col("Date").dt.strftime("%Y-%m").alias("month"))
daily = daily.with_columns(pl.col("Date").dt.strftime("%Y-%m").alias("month"))
monthly = (
    daily.group_by(["window", "month"]).agg(
        pl.len().alias("days"),
        pl.col("ic").mean().alias("IC"),
        pl.col("rank_ic").mean().alias("RankIC"),
        (100 * (pl.col("ic") > 0).mean()).alias("IC>0 %"),
    )
    .join(month_stats(preds, ["window", "month"]), on=["window", "month"])
    .sort(["month", "window"])
)
with pl.Config(tbl_rows=-1, float_precision=3):
    display(monthly)  # noqa: F821

x = np.arange(monthly.height)
colors = [WINDOW_COLORS[w - 1] for w in monthly["window"]]
labels_x = [m if d >= 5 else f"{m}*" for m, d in zip(monthly["month"], monthly["days"])]
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].bar(x, monthly["RankIC"], color=colors, width=0.8)
axes[0].plot(x, monthly["IC"], "o", color="#0b0b0b", ms=4, label="Pearson IC")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].axhline(daily["rank_ic"].mean(), color="gray", ls="--", lw=1,
                label=f"overall mean Rank IC ({daily['rank_ic'].mean():.3f})")
axes[0].set_ylabel("correlation")
axes[0].set_title("Monthly mean daily Rank IC (bars, colour = window) and Pearson IC (dots)", pad=20)
axes[0].legend(loc="upper left", fontsize=8)
axes[1].bar(x, monthly["slope"], color=colors, width=0.8)
axes[1].axhline(1, color="gray", ls="--", lw=1, label="calibrated (slope 1)")
axes[1].axhline(0.5, color="gray", ls=":", lw=1, label="strongly overconfident (< 0.5)")
axes[1].set_ylabel("slope of label on prediction")
axes[1].set_title("Monthly calibration slope (< 1 = predictions too spread out)")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_xticks(x, labels_x, rotation=60, fontsize=8)
# Window labels above the top panel.
for w in range(1, n_windows + 1):
    idx = np.flatnonzero(monthly["window"].to_numpy() == w)
    axes[0].annotate(f"W{w}", (idx.mean(), 1.0), xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=9, color=WINDOW_COLORS[w - 1])
fig.text(0.01, 0.0, "* fewer than 5 test days in that month for that window", fontsize=8)
save(fig, "monthly")

robust = daily.group_by("window").agg(
    pl.col("ic").mean().alias("IC mean"),
    pl.col("ic").median().alias("IC median"),
    pl.col("ic").filter(
        pl.col("ic").is_between(pl.col("ic").quantile(0.1), pl.col("ic").quantile(0.9))
    ).mean().alias("IC trimmed 10%"),
    pl.col("rank_ic").median().alias("RankIC median"),
    (100 * (pl.col("rank_ic") > 0).mean()).alias("RankIC>0 %"),
).sort("window")
with pl.Config(float_precision=4):
    display(robust)  # noqa: F821
