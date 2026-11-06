---
name: compare-ts48-baseline
description: Compare every finished TimeSeries48 baseline run (the original 48 raw features; runs 3 to 6.x, no feature-family runs) under runs/ with Run-2 (the author's original baseline), paired on common test rows, and refresh the grand table in baseline_performance.ipynb at the repo root. Detects new runs automatically. Use when the user asks to update the baseline table or overview, compare all runs with the original baseline or Run-2, or add a newly finished baseline run to the overview. Multi-day family runs go to compare-ts48-multiday instead.
---

# compare-ts48-baseline

Maintains `baseline_performance.ipynb` (repo root): one grand table with **Run-2**
(`runs/run-2_original-baseline-six-windows`, trained exactly as the author's
baseline) and every later **baseline** run side by side.
- **Baseline runs only** (user decision, 2026-10-26): "baseline" = the original 48
  raw features (runs 3 to 6.x). Runs of a feature family (run 7 on; their archived
  script says `Feature family "..."`) are left out. The multi-day family has its own
  table and skill (`compare-ts48-multiday`).
- **Comparison:** each run is compared with Run-2 **paired** on the same test
  dates and stocks.
- **Code:** it uses the `analyze-ts48-results` skill's own section-7 code (cells
  01, 02, 03, 07 and 10), so the numbers match the per-run `analysis.ipynb`
  notebooks.

## Procedure

1. Run `.venv/bin/python claude/compare-ts48-baseline/scripts/update_baseline_table.py`
   from the repo root. It:
   - scans `runs/run-*/` and leaves out feature-family runs (`is_family_run()`);
   - **skips** Run-2 itself, unfinished runs, and runs that don't cover all six of
     Run-2's test windows (e.g. Run-1, a single window);
   - folds **seed variants** into their configuration's column. `run-N.k_*`
     counts as a seed variant of `run-N_*` if the archived scripts differ only in
     `RUN_DIR`, `ARXIV_FILE` and `RANDOM_SEED`; otherwise it gets its own column
     and a note;
   - computes **new or changed** runs and takes the others from
     `claude/compare-ts48-baseline/cache/<run>.json` (keyed on the run's
     results and prediction files);
   - rewrites only the generated cells (`runs`, `grand-table`, `definitions`; the
     `intro` cell is written once), keeping any cells the user added.

   `--force` recomputes every run. Bump `VERSION` in the script when you change
   the metrics, so stale cache entries are dropped.
2. **Check the output:** which runs were computed, cached or skipped (and why),
   and any coverage warning (fewer common rows than Run-2's). A run that includes
   the reserved final window is compared on Run-2's windows only. Never present
   final-window numbers here (I-6).
3. **Read the grand-table cell** and tell the user about the new run(s):
   - Rank IC and R² (raw value and % vs Run-2);
   - the paired ΔRank IC t-stat and windows won;
   - calibration (per-window slopes and mean |slope − 1|);
   - runtime;
   - which rows it is now best in.

   Keep it short; the interpretation lives in the per-run analysis notebook
   (`analyze-ts48-results`).
4. If `baseline_performance.ipynb` is open in VS Code, the user must use "File:
   Revert File" (not save) to see the update.

## Table rules (keep them when editing the script)

- **Columns:** Run-2 first, then runs by number, labelled `Run-N`. Full names are
  in the `runs` cell.
- **Cells:** line 1 = the raw value; line 2 = % change vs Run-2 (relative to
  |Run-2's value|; Run-2 shows +0.0%).
  - Calibration slopes and mean |slope − 1| show raw values only, because their
    differences are confusing.
  - Rows that only exist relative to Run-2 (the CI and t of ΔRank IC, days,
    windows and months won) show "—" for Run-2.
- **Best value per row:** shown as a **green cell** with bold text, compared at
  the displayed precision; ties are all highlighted.
  - The grand table is an inline-HTML `<table>` inside a markdown cell, because
    markdown tables can't colour cells.
  - The green is translucent (`rgba(46,160,67,0.28)`), so it works in light and
    dark themes. Bold stays as a fallback if a renderer strips styles.
  - The % line is smaller and muted.
  - Highest: Rank IC, Pearson IC, Q5−Q1, Q5, R², days, windows and months won.
  - Lowest: Q1, RMSE, mean |slope − 1|, runtime.
  - Closest to 1: calibration slopes.
  - No highlight: the CI and t (significance, not quality), prediction agreement, the
    tree count.
- **Seed variants** (user decision, 2026-10-22): no column of their own, and no
  averaging.
  - Run-N's column shows its **primary seed** (run-N), so it stays comparable
    with the single-seed columns, plus a muted line "low ~ high" (or "all x"). The
    column header shows the seed count ("3 seeds").
  - The last column, **Seed noise ±**, is 1.96·√2·σ_seed, with σ pooled over all
    seed groups: the 95% band for a difference between two single-seed runs.
    It reads "—" until some configuration has ≥ 2 seeds, and "≈0" when it
    rounds to zero.
  - **Light green** marks values within that band of the best (dark green): not
    distinguishable from the best by seed alone.
- **Demeaned columns** (user decision, 2026-10-23): `DEMEANED_RUNS` (now 5 and
  6) also get a column `Run-N (demeaned)`, right after Run-N.
  - It holds the same predictions minus each day's mean prediction. `compare(...,
    demean=True)` applies this right after the analysis cells load the predictions.
  - Its seed variants are demeaned too. It is marked `derived`, so it does not
    enter the seed-noise pooling, because it holds the same models.
  - Cached as `cache/<run>.demeaned.json`.
  - The *How to read* cell gets a bullet on demeaning, plus a per-window table of
    the subtracted means: the average and SD over days, and the daily label mean,
    in label units.
- **Code-only differences:** `CODE_ONLY_DIFFS` lists run numbers whose seed
  variants differ from run-N by a verified code-only change. Run 5 is listed: it
  was re-run with out-of-fold logging, and its predictions matched within 1e-17.
  The runs cell states the exception.
- **Sections:**
  - ranking (paired);
  - Rank IC per window;
  - Rank IC per ex-ante regime;
  - magnitudes;
  - R² per window;
  - calibration slope per window;
  - cost (runtime in minutes, mean trees).

## Pitfalls

- Run folders must sit at `runs/<run>/`. The cells resolve the data file as
  `RUN_DIR.parents[1] / "data.parquet"`.
- If the analysis skill's cells change how a metric is computed, bump `VERSION`
  (or use `--force`) so cached runs are recomputed.
- This table shows levels and paired differences against Run-2 only. Whether one
  later run beats another needs their own paired comparison (`analyze-ts48-results`
  with `--baseline`), not a reading across columns.
- `compare(..., demean=True)` and `grand_table(..., magnitude_note)` are also used by
  `compare-ts48-multiday` (it imports this script); keep them when editing.
