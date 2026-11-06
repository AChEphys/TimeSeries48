---
name: compare-ts48-multiday
description: Compare every finished run of the TimeSeries48 multi-day feature family (runs trained with train_multi-day-feature-sets.py) with Run-2 (the author's original baseline), paired on common test rows, with Run-5 (demeaned) as the reference column, and refresh the grand table and the per-window Rank IC / R² plot in multi-day_performance.ipynb at the repo root. Detects new family runs automatically. Use when the user asks to update the multi-day table or overview, or to add a newly finished multi-day run to it.
---

# compare-ts48-multiday

Maintains `multi-day_performance.ipynb` (repo root), the multi-day family's
counterpart of `baseline_performance.ipynb` (skill `compare-ts48-baseline`):
- **Columns:** Run-2 (`runs/run-2_original-baseline-six-windows`, the author's
  baseline, raw) leftmost, then **Run-5 (demeaned)** (48 raw features, our best
  baseline and the reference; seed variants 5.1/5.2 folded in), then every finished
  **multi-day** run. A run belongs to the
  family when its archived script is a copy of `train_multi-day-feature-sets.py`
  (its docstring says `Feature family "multi-day"`).
- **Comparison:** every column vs Run-2, **paired** on the same test dates and
  stocks, with the analyze-ts48-results section-7 cells, run through
  `compare-ts48-baseline`'s `compare()`. The numbers match the other tables and
  notebooks.
- **Every column except Run-2 is demeaned per day** (N-24). **Run-2 stays raw**:
  demeaning wasn't part of the author's intent (user decision, 2026-10-26). Ranking
  rows are unchanged by demeaning; the magnitude rows of all other columns are the
  demeaned ones. Run-5 (demeaned) matches the column of that name in
  `baseline_performance.ipynb`.
- **Plot:** after the table, a code cell draws test Rank IC and test R² per window
  for Run-2 (raw), Run-5 (demeaned) and the 2 best and 2 worst family columns
  (demeaned). Its data are written into
  the cell and its figure is stored as output, so it shows without running.

## Procedure

1. Run `.venv/bin/python claude/compare-ts48-multiday/scripts/update_multiday_table.py`
   from the repo root. It:
   - scans `runs/run-*/` for Run-5 and its seed variants and for finished family runs;
     it skips runs that don't cover all six of Run-2's test windows;
   - folds `run-N.k` into Run-N's column only when the archived scripts differ in
     `RUN_DIR`, `ARXIV_FILE` and `RANDOM_SEED` alone. Runs like 8.1 / 8.2 (different
     `FEATURE_SET`) get their own columns. A `run-N.k` without a `run-N` is simply
     its own column;
   - computes new or changed runs and takes the others from
     `claude/compare-ts48-multiday/cache/<run>.json` (keyed on the run's results and
     prediction files; a few seconds per new run);
   - rewrites only the generated cells (`runs`, `grand-table`, `full-table`, `definitions`,
     `plots-md`, `plots`; the `intro` is written once) and keeps any other cells.

   `--force` recomputes every run.
2. **Check the output:** which runs were computed, cached or skipped (and why),
   and any coverage warning. Never present final-window numbers (I-6); the runs
   skip that window.
3. **Look at the figure**, e.g. by extracting the `plots` cell's PNG output, and
   check that the labels are legible.
4. **Read the grand table** and tell the user about the new run(s), briefly:
   - Rank IC and R² (demeaned), raw value and % vs Run-2, and how they compare with
     Run-5 (demeaned). Part of every gain in R² over Run-2 is the demeaning itself
     (Run-5: 0.00879 raw → 0.00942 demeaned);
   - the paired ΔRank IC t-stat vs Run-2 and windows won;
   - calibration (per-window slopes, mean |slope − 1|);
   - runtime;
   - which rows it is now best in.

   Whether a run beats **Run-5** needs their own paired test: point to the run's
   `analysis.ipynb` (skill `analyze-ts48-results`, baseline run 5).
5. If `multi-day_performance.ipynb` is open in VS Code, the user must use "File:
   Revert File" (not save) to see the update.

## Table and plot rules

Column selection (user decision, 2026-10-27: too many columns, and VS Code wouldn't
scroll the table sideways):
- the **Grand table** cell shows Run-2, Run-5 (demeaned), the family runs in
  `TABLE_RUNS` (now 11, 17, 21, 22, 23, 25) and every run numbered above `REVIEWED_UP_TO` (now 30),
  i.e. new runs not yet judged. Highlights compare only the columns shown;
- the **All columns** cell holds the full table inside a collapsed `<details>` block
  and an `overflow-x: auto` box (scrolls sideways where the renderer allows it);
- `MULTIDAY_REFERENCE` (now 22, user decision 2026-10-27) is named in the table's heading;
- after reviewing new runs, **ask the user** which to keep, then update `TABLE_RUNS`
  and raise `REVIEWED_UP_TO` to the last reviewed run.


The table rules are `compare-ts48-baseline`'s (shared code, `grand_table()`):
- cells: raw value, then % vs Run-2;
- dark green = best per row; light green = within seed noise of the best;
- the seed-noise column pools σ over the seed groups in *this* table (now Run-5,
  Run-11 and Run-22, 3 seeds each);
- the magnitude section titles carry "demeaned per day except Run-2".

Plot (user decision, 2026-10-26: plotting every run became illegible):
- only the references and the extremes: Run-2 (raw) grey dashed, Run-5 (demeaned)
  black, the `SHOWN` = 2 **best** family columns (solid; blue, green) and the 2
  **worst** (dashed; red, orange): six lines at most;
- **score:** the **weaker** of a column's two ranks, on Rank IC and on R² (demeaned),
  over the family columns' primary seeds: bad at either one means bad (user decision,
  2026-10-26). Ties go to the better rank sum, then the higher Rank IC. The markdown cell above the plot names the
  runs shown and is rewritten on every update;
- left panel test Rank IC, right panel test R² (demeaned; Run-2 raw), x = W1–W6;
- colours are fixed per role (not per run) from the validated family palette, each
  with its own marker; orange is low-contrast on white, hence markers and legend.

## Pitfalls

- The skill imports `claude/compare-ts48-baseline/scripts/update_baseline_table.py`
  and uses its `compare(..., demean=True)`. The `grand_table(..., magnitude_note)`
  option exists for this skill; keep it when editing that script.
- Feature-family runs are deliberately absent from `baseline_performance.ipynb`
  (baseline runs only, up to 6.x); this table is where multi-day runs are compared.
- If the analysis cells change how a metric is computed, bump `VERSION` in the
  baseline script (it is part of both skills' cache keys) or use `--force`.
- A new feature family gets its own table and skill, the way this one copies the
  baseline's. Don't widen `FAMILY_MARK`.
- The `intro` cell is written only once, so the user's edits survive refreshes. If
  you change `INTRO` in the script, replace the notebook's intro cell yourself (only
  if the user hasn't edited it), or the old text stays.
