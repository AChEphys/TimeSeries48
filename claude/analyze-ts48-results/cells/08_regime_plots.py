def shade(ax) -> None:
    """Shade the background by regime (all runs, including short ones)."""
    runs = episodes(test_days.drop_nulls("regime"), min_len=1)
    starts = runs["start"].to_list()
    ends = starts[1:] + [test_days["Date"].max()]
    for regime_name, s, e in zip(runs["regime"], starts, ends):
        ax.axvspan(s, e, color=REGIME_COLORS[regime_name], alpha=0.45, lw=0)


in_test = per_day.filter(pl.col("Date").is_between(test_days["Date"].min(), test_days["Date"].max()))
# Largest dispersion spikes in the test period, at least 60 days apart.
spikes = []
for row in in_test.sort("std", descending=True).iter_rows(named=True):
    if all(abs((row["Date"] - s).days) > 60 for s in spikes):
        spikes.append(row["Date"])
    if len(spikes) == 3:
        break

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [1.2, 0.8, 1.2]})
dates = in_test["Date"].to_numpy()
axes[0].plot(dates, in_test["std"], color="#8f8e88", lw=0.8, label="daily dispersion (label std)")
axes[0].plot(dates, in_test["trail_disp"], color="#0b0b0b", lw=2, label="trailing measure (T-26 ... T-7)")
axes[0].set_title("1. Stock-specific dispersion: the trailing measure reacts only after a spike")
axes[1].plot(dates, in_test["disp_pct"], color="#0b0b0b", lw=1.5)
for level in (1 / 3, 2 / 3):
    axes[1].axhline(level, color="gray", ls="--", lw=1)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("percentile")
axes[1].set_title("2. Trailing dispersion vs its previous year: < 1/3 calm, > 2/3 turbulent")
t = test_days["Date"].to_numpy()
axes[2].plot(t, test_days["rank_ic"], color="#8f8e88", lw=0.6)
axes[2].plot(t, test_days["rank_ic"].rolling_mean(20, min_samples=5), color=WINDOW_COLORS[0],
             lw=2, label="Rank IC, 20-day mean")
axes[2].axhline(0, color="black", lw=0.8)
axes[2].axhline(test_days["rank_ic"].mean(), color="gray", ls="--", lw=1, label="overall mean")
axes[2].set_title("3. Daily Rank IC against the regime shading")
for ax in axes:
    shade(ax)
    for s in spikes:
        ax.axvline(s, color="#d03b3b", ls=":", lw=1.2)
for s in spikes:
    axes[0].annotate(f"spike {s}", (s, 1.0), xycoords=("data", "axes fraction"),
                     ha="left", va="top", fontsize=8, color="#d03b3b", xytext=(3, -3),
                     textcoords="offset points")
from matplotlib.patches import Patch  # noqa: E402
regime_handles = [Patch(color=REGIME_COLORS[r], alpha=0.45, label=r) for r in REGIMES]
axes[0].legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 0.9))
axes[2].legend(handles=axes[2].get_legend_handles_labels()[0] + regime_handles, loc="upper left",
               fontsize=8, ncol=5)
save(fig, "regime_timeline")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
xs = np.arange(len(REGIMES))
axes[0].bar(xs, by_regime["RankIC"], color=[REGIME_COLORS[r] for r in REGIMES], width=0.6,
            yerr=by_regime["RankIC 95% CI ±"], capsize=6, ecolor="#52514e")
axes[0].axhline(daily["rank_ic"].mean(), color="gray", ls="--", lw=1, label="all days")
for xi, row in zip(xs, by_regime.iter_rows(named=True)):
    axes[0].annotate(f"{row['days']} days\n{row['episodes (>=5 days)']} episodes", (xi, 0),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8,
                     color="#0b0b0b" if row["regime"] != "turbulent" else "white")
axes[0].set_xticks(xs, REGIMES)
axes[0].set_title("Mean daily Rank IC by regime (95% Newey–West CI)")
axes[0].legend(loc="upper left", fontsize=8)
for w in range(1, n_windows + 1):
    bw = by_window.filter((pl.col("window") == w) & (pl.col("days") >= 10))
    xw = [REGIMES.index(r) + (w - (n_windows + 1) / 2) * 0.06 for r in bw["regime"]]
    axes[1].plot(xw, bw["RankIC"], "-", color=WINDOW_COLORS[w - 1], lw=1, alpha=0.6)
    axes[1].scatter(xw, bw["RankIC"], s=bw["days"] * 3, color=WINDOW_COLORS[w - 1],
                    edgecolor="white", linewidth=1, label=f"W{w}", zorder=3)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xticks(xs, REGIMES)
axes[1].set_title("Per window (dot size = days; regimes with < 10 days omitted)")
axes[1].legend(ncol=3, fontsize=8, loc="upper left")
save(fig, "regimes")
