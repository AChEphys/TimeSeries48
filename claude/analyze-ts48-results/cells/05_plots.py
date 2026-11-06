fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, sharey=True)
for ax, col, name in zip(axes, ["ic", "rank_ic"], ["Pearson IC", "Rank IC (Spearman)"]):
    for w in range(1, n_windows + 1):
        d = daily.filter(pl.col("window") == w)
        x = d["Date"].to_numpy()
        ax.plot(x, d[col], color="#b5b4ad", lw=0.7)
        ax.plot(x, d[col].rolling_mean(20, min_samples=5),
                color=WINDOW_COLORS[w - 1], label=f"W{w} (20-day mean)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Daily {name}: grey = daily, colour = 20-day rolling mean per window")
axes[0].legend(ncol=n_windows, loc="upper left", fontsize=8)
save(fig, "ic_over_time")

fig, ax = plt.subplots(figsize=(14, 3.8))
x = daily["Date"].to_numpy()
ax.plot(x, daily["ic"].fill_nan(0).cum_sum(), color=WINDOW_COLORS[0], label="Pearson IC")
ax.plot(x, daily["rank_ic"].fill_nan(0).cum_sum(), color=WINDOW_COLORS[1], label="Rank IC")
for w in range(2, n_windows + 1):
    ax.axvline(daily.filter(pl.col("window") == w)["Date"][0], color="gray", ls="--", lw=0.8)
ax.set_title("Cumulative daily IC (slope = IC level; not a profit curve). Dashed: new window/model")
ax.legend()
save(fig, "cumulative_ic")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
wins = summary["window"].to_list()
for ax, col in zip(axes, ["IC", "RankIC", "Kendall(daily)"]):
    ax.bar(wins, summary[col], color=WINDOW_COLORS[:n_windows], width=0.7)
    ax.set_title(f"mean daily {col.replace('(daily)', '')}")
    ax.set_xticks(wins, [f"W{w}" for w in wins])
save(fig, "metrics_by_window")

fig, ax = plt.subplots(figsize=(8, 3.8))
for split, col in [("train", "train R2"), ("val", "val R2"), ("test", "test R2")]:
    ax.plot(wins, summary[col], "o-", color=SPLIT_COLORS[split], label=split, ms=8)
ax.set_xticks(wins, [f"W{w}" for w in wins])
ax.set_title("R² by window (train/val at best_iteration)")
ax.legend()
save(fig, "r2_by_window")

quint = (
    quint_daily.group_by(["window", "quintile"]).agg(pl.col("label_mean").mean())
    .sort(["window", "quintile"])
)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for w in range(1, n_windows + 1):
    q = quint.filter(pl.col("window") == w)
    axes[0].plot(q["quintile"], q["label_mean"], "o-", color=WINDOW_COLORS[w - 1], label=f"W{w}", ms=6)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_xlabel("prediction quintile (1 = lowest predicted)")
axes[0].set_title("Mean label by prediction quintile")
axes[0].legend(ncol=2)
binned = (
    preds.with_columns(
        ((pl.col("Prediction").rank("ordinal").over("window").cast(pl.Int64) - 1) * 20
         // pl.len().over("window")).alias("bin")
    )
    .group_by(["window", "bin"]).agg(pl.col("Prediction").mean(), pl.col("label").mean())
    .sort(["window", "bin"])
)
for w in range(1, n_windows + 1):
    b = binned.filter(pl.col("window") == w)
    axes[1].plot(b["Prediction"], b["label"], "o-", color=WINDOW_COLORS[w - 1], label=f"W{w}", ms=4, lw=1.5)
lim = [binned["Prediction"].min(), binned["Prediction"].max()]
axes[1].plot(lim, lim, "--", color="gray", lw=1, label="y = x")
axes[1].set_xlabel("mean prediction in bin")
axes[1].set_ylabel("mean label in bin")
axes[1].set_title("Calibration (20 prediction bins per window)")
axes[1].legend(ncol=2, fontsize=8)
save(fig, "quintiles_calibration")
print(quint.pivot(on="quintile", index="window", values="label_mean"))
