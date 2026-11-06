from IPython.display import Markdown, display

LEADERBOARD_DIRS = __LEADERBOARD_DIRS__  # Run folders to compare (set by the skill).


def short_name(path: Path) -> str:
    """'run-5_baseline-all-feat-...' -> 'run-5'."""
    return path.name.split("_")[0]


def r2_expr() -> pl.Expr:
    """Pooled R² of Prediction for label."""
    resid = ((pl.col("label") - pl.col("Prediction")) ** 2).sum()
    total = ((pl.col("label") - pl.col("label").mean()) ** 2).sum()
    return (1 - resid / total).alias("r2")


def run_summary(run_dir: Path) -> tuple[dict[str, float], pl.DataFrame]:
    """Headline test metrics over all dates, and per-window Rank IC / R² / ICIR."""
    _, p, _ = load_run(run_dir)
    d, _, _ = daily_metrics(p)
    per_window = (
        p.group_by("window").agg(r2_expr())
        .join(d.group_by("window").agg(
            pl.col("rank_ic").mean(),
            (pl.col("ic").mean() / pl.col("ic").std()).alias("icir"),
        ), on="window")
        .sort("window")
    )
    head = {
        "Rank IC": d["rank_ic"].mean(),
        "Pearson IC": d["ic"].mean(),
        "Kendall": d["kendall"].mean(),
        "R² (all rows)": p.select(r2_expr()).item(),
        "ICIR (window mean)": per_window["icir"].mean(),
    }
    return head, per_window


def bold_table(row_names: list[str], col_names: list[str], values: np.ndarray,
               fmts: list[str], best_per: str) -> str:
    """Markdown table; the highest shown value per 'row' or 'column' is bold."""
    shown = [[format(values[i, j], fmts[j if best_per == "column" else 0])
              for j in range(len(col_names))] for i in range(len(row_names))]
    num = np.array([[float(s) for s in row] for row in shown])
    best = num.max(axis=0 if best_per == "column" else 1)
    lines = ["| | " + " | ".join(col_names) + " |",
             "|---|" + "---|" * len(col_names)]
    for i, name in enumerate(row_names):
        cells = [f"**{s}**" if num[i, j] == (best[j] if best_per == "column" else best[i])
                 else s for j, s in enumerate(shown[i])]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


dirs = [Path(p) for p in LEADERBOARD_DIRS]
if RUN_DIR.resolve() not in [p.resolve() for p in dirs]:
    dirs.append(RUN_DIR)
names = [short_name(p) for p in dirs]
summaries = {name: run_summary(p) for name, p in zip(names, dirs, strict=True)}

metrics = list(next(iter(summaries.values()))[0])
head_values = np.array([[summaries[n][0][m] for m in metrics] for n in names])
head_md = bold_table(names, metrics, head_values,
                     ["0.4f", "0.4f", "0.4f", "0.5f", "0.3f"], best_per="column")

window_rows = [f"W{k + 1} ({r['test_start']} → {r['test_end']})"
               for k, r in enumerate(results)] + ["all test dates"]
rank_ic = np.array([[*summaries[n][1]["rank_ic"].to_list(), summaries[n][0]["Rank IC"]]
                    for n in names]).T
r2 = np.array([[*summaries[n][1]["r2"].to_list(), summaries[n][0]["R² (all rows)"]]
               for n in names]).T
rank_md = bold_table(window_rows, names, rank_ic, ["0.4f"], best_per="row")
r2_md = bold_table(window_rows, names, r2, ["0.5f"], best_per="row")

legend = "\n".join(f"- **{n}**: `{p.name}`" for n, p in zip(names, dirs, strict=True))
text = (f"**Runs**\n\n{legend}\n\n**Headline test metrics** (best per column in bold)\n\n"
        f"{head_md}\n\n**Rank IC per window** (best per row in bold)\n\n{rank_md}\n\n"
        f"**R² per window** (best per row in bold)\n\n{r2_md}")
display(Markdown(text))
print(text)  # Plain text too, for headless runs.
