assert COMPARE, "No baseline set: skip this section."
run_name, base_name = RUN_DIR.name, BASELINE_DIR.name
win_starts = [paired.filter(pl.col("window") == w)["Date"].min() for w in sorted(paired["window"].unique())]

# ---- 1. Where in time does the run gain or lose? ------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
t = paired["Date"].to_numpy()
axes[0].plot(t, paired["d_ric"].fill_nan(0).cum_sum(), color=MODEL_COLORS["run"], label="cumulative ΔRank IC")
axes[0].plot(t, paired["d_ic"].fill_nan(0).cum_sum(), color="#eb6834", lw=1.5, label="cumulative ΔPearson IC")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title(f"Cumulative daily difference, {run_name} − {base_name} (rising = run ahead)")
axes[0].legend(loc="upper left", fontsize=8)
axes[1].plot(t, paired["d_ric"], color="#b5b4ad", lw=0.6)
axes[1].plot(t, paired["d_ric"].rolling_mean(20, min_samples=5), color=MODEL_COLORS["run"],
             label="ΔRank IC, 20-day mean")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Daily ΔRank IC against the ex-ante regime shading")
for ax in axes:
    shade(ax)
    for s in win_starts[1:]:
        ax.axvline(s, color="gray", ls="--", lw=0.8)
axes[1].legend(handles=axes[1].get_legend_handles_labels()[0]
               + [Patch(color=REGIME_COLORS[r], alpha=0.45, label=r) for r in REGIMES],
               loc="upper left", fontsize=8, ncol=4)
save(fig, "compare_timeline")

# ---- 2. Is the gain consistent across windows, regimes, months? ---------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 0.7, 1.6]})
wl = by_window_delta["scope"].to_list()
wcol = [WINDOW_COLORS[int(s[1:]) - 1] for s in wl]
axes[0].bar(wl, by_window_delta["dRankIC"], color=wcol, width=0.6,
            yerr=by_window_delta["95% CI ±"], capsize=5, ecolor="#52514e")
axes[0].set_title("ΔRank IC per window (95% NW CI)")
rl = by_regime_delta["scope"].to_list()
axes[1].bar(rl, by_regime_delta["dRankIC"], color=[REGIME_COLORS[r] for r in rl], width=0.6,
            yerr=by_regime_delta["95% CI ±"], capsize=5, ecolor="#52514e")
axes[1].set_title("ΔRank IC per ex-ante regime")
month_window = paired.group_by("month").agg(pl.col("window").mode().first()).sort("month")
mcol = [WINDOW_COLORS[w - 1] for w in month_window["window"]]
xm = np.arange(by_month_delta.height)
axes[2].bar(xm, by_month_delta["dRankIC"], color=mcol, width=0.8)
axes[2].set_xticks(xm, by_month_delta["month"], rotation=70, fontsize=7)
axes[2].set_title("ΔRank IC per month (colour = window)")
for ax in axes:
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(overall["dRankIC"], color=MODEL_COLORS["run"], ls="--", lw=1)
save(fig, "compare_breakdown")

# ---- 3. Levels and magnitudes for both models ----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
xw = np.arange(len(wl))
axes[0].plot(xw, by_window_delta["RankIC base"], "o-", color=MODEL_COLORS["baseline"], label=base_name, ms=7)
axes[0].plot(xw, by_window_delta["RankIC run"], "o-", color=MODEL_COLORS["run"], label=run_name, ms=7)
axes[0].set_xticks(xw, wl)
axes[0].set_title("Mean daily Rank IC per window")
qcols = [f"Q{q}" for q in (1, 5)]
qb = [overall["Q1 base"], overall["Q5 base"]]
qr = [overall["Q1 run"], overall["Q5 run"]]
xq = np.arange(2)
axes[1].bar(xq - 0.18, qb, width=0.35, color=MODEL_COLORS["baseline"], label=base_name)
axes[1].bar(xq + 0.18, qr, width=0.35, color=MODEL_COLORS["run"], label=run_name)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xticks(xq, ["bottom quintile (Q1)", "top quintile (Q5)"])
axes[1].set_title("Mean label of the extreme quintiles")
axes[2].plot(xw, facts["slope_base"], "o-", color=MODEL_COLORS["baseline"], label=base_name, ms=7)
axes[2].plot(xw, facts["slope_run"], "o-", color=MODEL_COLORS["run"], label=run_name, ms=7)
axes[2].axhline(1, color="gray", ls="--", lw=1)
axes[2].set_xticks(xw, [f"W{w}" for w in facts["window"]])
axes[2].set_title("Calibration slope per window (1 = calibrated)")
for ax in axes:
    ax.legend(fontsize=8)
save(fig, "compare_levels")
print(f"Mean daily prediction agreement (rank corr run vs baseline): {overall['agreement']:.3f}")
