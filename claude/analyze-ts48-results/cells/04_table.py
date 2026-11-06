def curve_at(c: dict | None, split: str, metric: str) -> float:
    """A learning-curve metric at best_iteration; NaN if it wasn't recorded."""
    if not c or not c.get("best_iteration"):
        return float("nan")
    values = c["curves"].get(split, {}).get(metric)
    best = c["best_iteration"]
    return values[best - 1] if values and best <= len(values) else float("nan")


rows = []
for w in range(1, n_windows + 1):
    d = daily.filter(pl.col("window") == w)
    ic = d["ic"].drop_nans().to_numpy()
    r = results[w - 1]
    rows.append({
        "window": w,
        "test": f"{r['test_start']} -> {r['test_end']}",
        "days": len(ic),
        "features": r.get("features"),
        "best_iter": r.get("best_iteration"),
        "IC": ic.mean(),
        "RankIC": d["rank_ic"].mean(),
        "Kendall(daily)": d["kendall"].mean(),
        "ICIR": ic.mean() / ic.std(ddof=1),
        "IC>0 %": 100 * (ic > 0).mean(),
        "t(naive)": ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)),
        "t(NW)": newey_west_t(ic),
        "IC ac1": np.corrcoef(ic[1:], ic[:-1])[0, 1],
        "train R2": curve_at(curves.get(w), "train", "r2"),
        "val R2": curve_at(curves.get(w), "val", "r2"),
        "test R2": r.get("r2", float("nan")),
        "Q5-Q1": d["q5_minus_q1"].mean(),
        "runtime s": r.get("runtime", float("nan")),
        "peak GB": r.get("peak_memory_GB", r.get("peak_memory_gb", float("nan"))),
    })
summary = pl.DataFrame(rows)
all_ic = daily["ic"].drop_nans().to_numpy()
all_ric = daily["rank_ic"].drop_nans().to_numpy()
print(f"All windows: RankIC={all_ric.mean():.4f} (t NW {newey_west_t(all_ric):.1f})  "
      f"IC={all_ic.mean():.4f}  ICIR={all_ic.mean() / all_ic.std(ddof=1):.2f}  "
      f"t(NW)={newey_west_t(all_ic):.1f}  IC>0 on {100 * (all_ic > 0).mean():.0f}% of days")
with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=250):
    display(summary)  # noqa: F821  (IPython built-in)
