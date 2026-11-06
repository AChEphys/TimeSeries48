---
name: analyze-ts48-results
description: Analyze a TimeSeries48 training run (rolling LightGBM windows under runs/<run>/) and compare it with a baseline run. Builds and executes runs/<run>/analysis.ipynb (daily IC / Rank IC, Newey–West stats, per-window table, quintiles, calibration, monthly and ex-ante-regime breakdowns, paired comparison vs baseline), reads the plots, then writes the What-changed, Findings, Summary and Hypotheses cells and logs results in claude/project-notes.md. Use when the user asks to analyze, evaluate, review or compare a training run's results in this project.
---

# analyze-ts48-results

Turns a finished training run into an analysis notebook with interpreted
findings and a summary that **compares the run with the baseline**. The
method distils the baseline study (run 2, Oct 2026). The reasoning behind
each rule is in `claude/project-notes.md` (N-6 … N-12, I-1 … I-8).

## Inputs

- **Run folder:** `runs/<run>/`, produced by `train_baseline.py` or a
  variant. Needs `results_lgbm.json`, `predictions/predictions_<window>.parquet`
  (Date, Code, label, Prediction) and `curves/curves_<window>.json`. The early
  flat layout (files directly in the run folder) also works.
- **Baseline folder:** the reference to compare against. Default:
  `runs/run-2_original-baseline-six-windows`, unless `claude/project-notes.md`
  names a newer reference (e.g. a revised-protocol baseline after the I-1
  fix). If unclear, ask the user which baseline to use. Skip the comparison
  only when analyzing the baseline itself.

## Procedure

Scripts live in `claude/analyze-ts48-results/scripts/`. Run them with the
project's `.venv/bin/python`.

1. **Sanity check the run.** Read `results_lgbm.json`: window count, test
   dates, `best_iteration` (≈ 2000 means capped), runtime, peak memory.
   - **Final-window rule:** if the run includes the reserved final window
     (test dates after 2025-12-18, issue I-6), confirm with the user that
     this is the one-time final evaluation. Never use it to choose between
     variants.
   - Check the run's settings (archived script copy in the run folder) for
     what changed against the baseline. Note label transforms (clip / rank),
     which make R² and calibration non-comparable.
2. **Build the notebook:**
   `python scripts/build_notebook.py runs/<run> --baseline runs/<baseline>`.
   It refuses to overwrite an existing `analysis.ipynb`. Rebuild with
   `--force` only if the user agrees, since it resets the text cells.
   Optional: `--leaderboard RUN_DIR ...` appends a leaderboard (section 15).
   It shows the listed runs side by side on the same test dates, with the
   best value in bold: headline metrics (Rank IC, Pearson IC, daily Kendall,
   pooled R², window-mean ICIR), per-window Rank IC and per-window R². The
   analyzed run is added if missing. Use it when the user wants several runs
   compared at a glance; remind them that the paired tests decide
   significance.
3. **Execute it headless:**
   `python scripts/run_notebook.py runs/<run>/analysis.ipynb > /tmp/out.txt`.
   Takes seconds. Fix any failing cell in `cells/` (not in the notebook),
   then rebuild.
4. **Read the outputs:** the printed tables, and **look at every
   `runs/<run>/plots/analysis_*.png`** with the Read tool. Check the figures
   for overlapping labels; fix the cell code if needed.
5. **Establish what changed** before reading any comparison numbers (see
   "What changed" below). Use the setup-diff cell output (a comment-free
   diff of the archived scripts, row and feature counts, selected-feature
   differences), the user's description and the project notes. Ask the user
   about anything the diff can't explain, e.g. changed data files or
   external settings.
6. **Interpret** with the checklist below. Back every claim with a number.
   Run extra ad-hoc queries where the notebook isn't enough (e.g. the
   cross-model test).
7. **Write** the text cells as markdown files, then insert them:
   `python scripts/set_text_cells.py runs/<run>/analysis.ipynb --changes C.md --findings F.md --summary S.md --hypotheses H.md`.
   Required headings: `## 6. What changed vs the baseline: …`, `## Findings…`,
   `## Summary…`, `## Hypotheses…`. Without a baseline, only findings and
   summary.
8. **Log** a new N-entry (and any new I-entries) in `claude/project-notes.md`,
   with a row in its index tables. Include what changed, the per-window
   comparison table and the hypotheses.
9. **Tell the user:** headline numbers, the verdict versus the baseline, and
   the link to the notebook. If it is open in VS Code, they must run "File:
   Revert File" (not save) to see the new version.

## Notebook sections (cells/)

| File | Content |
|---|---|
| 00 | Intro and ground rules |
| 01–02 | Setup (helpers: `newey_west_t`, `nw_ci`, `load_run`, `daily_metrics`), load run and baseline |
| 03–04 | Daily metrics; per-window table (ICIR, t naive / NW, IC autocorrelation, train / val / test R², Q5−Q1, cost) |
| 05 | IC over time, cumulative IC, per-window bars, R², quintiles, calibration |
| 06 | Monthly Rank IC (bars) + Pearson IC (dots), monthly calibration slope, robust summaries |
| 07–08 | Ex-ante regimes (trailing dispersion T−26 … T−7, percentile vs the previous year), timeline and per-regime Rank IC with CIs and episode counts |
| 09 | **What changed** (only with `--baseline`): text cell written by Claude, plus the automatic setup diff (comment-free script diff, per-window rows and features, selected-feature differences) |
| 10–11 | **Comparison** (only with `--baseline`): paired daily ΔRank IC / ΔIC / ΔQ5−Q1 on common rows, per window / regime / month, prediction agreement, calibration and quintiles for both, cumulative Δ timeline |
| 12–13 | Findings and Summary (text cells written by Claude) |
| 14 | **Hypotheses** (only with `--baseline`): text cell written by Claude |
| 15 | **Leaderboard** (only with `--leaderboard`): unpaired levels of several runs, best in bold |

Edit the cell files, not generated notebooks. Notebook-only changes are lost
on the next build.

## Interpretation checklist (lessons from the baseline)

**Headline and significance**
- **Ranking and magnitude are both primary.** The predictions feed the
  author's portfolio optimizer, so both the ordering and the size of
  predictions matter. For ranking: Rank IC is robust; Pearson IC is
  inflated by fat-tailed labels (baseline: Pearson ≈ 1.6× Rank IC), so
  report both and note the gap. For magnitude: calibration slope, R², RMSE
  (see Magnitudes).
- **Use Newey–West t, never naive t or the script's `ir`.** Daily ICs are
  autocorrelated (lag-1 ≈ 0.5–0.7) through the overlapping 5-day labels, so
  naive t is ~1.7× too high. ICIR (unscaled) is fine as a description.
- Check that medians and trimmed means ≈ means (no window carried by a few
  days), and the share of positive days.

**Magnitudes**
- Magnitudes matter as much as ranking (the optimizer uses them). R² is
  tiny by nature (≈ IC²), so judge it relative to the baseline, together
  with the **calibration slope** (1 = right, < 1 = overconfident, < 0.5 =
  strongly overconfident), RMSE and the calibration plot. Ranking can be fine
  while magnitudes are off. Cheap post-processing (per-day demeaning,
  shrinkage fitted on validation data) can fix magnitudes without touching
  ranking: mention it when relevant.
- **val R² ≫ test R²** in a window means either the I-1 leak or a hard test
  period. Look at that window's months first. In the baseline it was the
  test period (W4/W5), not the leak.

**Where it works**
- **Quintiles:** the baseline's edge is in **Q1** (volatile, lottery-like
  stocks drift down under A-share shorting limits); big winners are
  news-driven and unpredictable, so Q2–Q5 are flat. Check whether a variant
  changes this, e.g. improves Q5.
- **Months:** ~20 days ≈ only ~5 independent days. Look for patterns:
  months bad across models, or runs of weak months. Never exclude months from
  headline numbers.
- **Attributing a bad stretch:** if needed, score older saved models on the
  same test period (all trained before it, so no leak). If they fail in the
  same months, the **period** is hard, not the model.
- **Regimes are ex-ante only** and don't foresee shocks: spike onsets look
  calm. Read them with the CI and **episode count**; 2–3 episodes is a hint,
  not a finding. Calm ≈ normal in the baseline; post-spike was higher.
- The dispersion measure is stock-specific (labels are factor-adjusted), not
  market volatility.

**Comparison with the baseline**
- Paired on common (Date, Code) rows only. Report the coverage. If the
  universe changed, say so: the run's own metrics and the paired ones differ.
- **Verdict criteria:**
  - **Better:** ΔRank IC > 0 with a 95% NW CI excluding 0, positive in
    most windows (e.g. ≥ 5 of 6), not carried by one regime or a few
    months; Pearson IC and Q5−Q1 not materially worse; **and magnitudes not
    worse** (calibration slope, R², RMSE on common rows). A pure magnitude
    improvement (better calibration / R², same Rank IC) also counts as
    better. Say which kind it is.
  - **No detectable difference:** the CI includes 0, or the windows are
    split.
  - **Worse:** ΔRank IC < 0 with a CI excluding 0.
  - About +0.005 Rank IC is a meaningful gain if it is consistent.
- Report **where** it helps (windows, regimes, months, Q1 vs Q5) and the
  **cost** (runtime, memory, `best_iteration`).
- **Prediction agreement** near 1 means the change barely altered the
  model; low agreement plus a similar IC means a different but equally good
  signal (a candidate for blending).
- If labels were transformed (clipping, rank-Gauss), map predictions back
  to label units before judging magnitudes. Rank-Gauss discards magnitude
  information, so it is a poor fit for an optimizer that uses magnitudes.
- Watch for selection: many variants tried on the same windows inflate the
  best one's apparent gain. Mention how many variants have been compared so
  far.

**Label meaning: don't over-interpret.** The label is roughly a 5-day
forward residual return (Barra-type factor model;
scale details unresolved and deliberately not pursued, I-8). Don't attach
further financial meaning to label or prediction values; report them in
**label units**. How the author interprets and scales them for their
portfolio optimizer is their business. Our job is to improve ranking *and*
magnitude.

**Caveats to state when relevant:** Q5−Q1 is gross (overlapping horizons,
costs, limited shorting, untradeable limit-locked stocks). Features are
opaque, so economic explanations are hypotheses.

## "What changed" cell

Written **before** interpreting the comparison, so the verdict is read
against the intended changes. List every difference, grouped:

| Category | Examples |
|---|---|
| Rows / universe | row selection or filtering (e.g. dropping new listings or limit-locked stocks), date range, windows |
| Features | added history features (lags, rolling summaries), feature-selection rule (`CORR_BAR`, dates used) |
| Labels | clipping, rank-Gauss, Huber target |
| Validation protocol | interleaved vs chronological split, purge gap, early-stopping metric and patience |
| Model parameters | `num_leaves`, `min_data_in_leaf`, GOSS vs bagging, regularization, learning rate |
| Training schedule | `LOOKBACK`, `TEST_DAYS`, retraining frequency, seeds |
| Evaluation | metric changes, universe used for scoring |
| Code-only | refactors with no intended effect (say so) |

For each: **baseline → run**, why it was changed, and the **expected effect**.
Also give the effective consequences from the setup facts (e.g. "training
rows −12%", "features 34 → 170"). State whether a change applies to training
only or also to the test universe: that decides whether the paired
comparison covers all rows. If more than one thing changed, say so
explicitly; it limits attribution in the hypotheses.

## Findings cell (detailed)

Numbered points, each with numbers: overall performance; consistency
(windows, medians, positive days); Pearson vs Rank; quintiles; calibration;
val vs test (leak or period); monthly patterns; regimes (with episode
counts); then **comparison**: overall Δ with CI and t, windows positive,
regimes and months, quintile changes, agreement, cost; the verdict, read
against the expected effects from the "What changed" cell; open questions.

## Summary cell (brief): template

```
## Summary: <run> vs <baseline>

**Setup.** One line on what changed vs the baseline (details in section 6);
windows / test dates; final window held back (or this is the final
evaluation).

**Overall performance.** Ranking: Rank IC (Pearson IC), ICIR, % positive
days, NW t, Q5−Q1. Magnitude: calibration slope, R², RMSE. Cost.

**Comparison with the baseline.** ΔRank IC ± CI (t), run better on X% of
days and in N of M windows; where it helps or hurts (windows, regimes,
quintiles); agreement; cost change. **Verdict:** better / no detectable
difference / worse.

**Where it excels.** 2–3 bullets.

**Where it stumbles.** 2–3 bullets.

**Probable reasons.** Ranked, with evidence; separate period effects from
model effects.

**What this run achieved and what to try next.** 2–4 bullets.

**Hypotheses in brief.** One line each: what most likely helped, what
most likely hurt, what didn't matter (details in the Hypotheses cell).
```

## Hypotheses cell (last)

Explain the differences causally, as **testable hypotheses**, not
conclusions:

```
## Hypotheses: what drove the difference vs <baseline>

**Likely improvements.** For each: change → mechanism → evidence (which
numbers, where: windows / regimes / months / Q1 vs Q5 / calibration) →
confidence (high / medium / low).

**Likely degradations.** Same format. Include costs (runtime, memory,
fewer rows, more overfitting: train–val gap, best_iteration).

**What matters most / what matters little.** Rank the changes by estimated
contribution, and name the ones with no detectable effect (CI includes 0,
windows split). Also note what *didn't* change but is the bigger driver
(e.g. the test period / market regime, which moves every model's IC more
than most setup changes).

**Confounding.** If several things changed at once, say which effects can't
be separated, and why (e.g. both changes act on the same quintile).

**Experiments to test them.** For each main hypothesis, the cheapest
ablation or follow-up run (change one thing, same windows, same seed), and
what result would confirm or refute it.
```

Evidence patterns to use:
- **Gain concentrated in Q1 vs Q5:** the change sharpened loser detection
  vs added upside information.
- **Gain mainly in turbulent regimes or around shocks:** the change helps
  adaptation (e.g. history features); in calm days: a better static mapping.
- **Δ Pearson IC ≫ Δ Rank IC:** the gain comes from extreme stocks (tails),
  and is less robust.
- **Better calibration but the same Rank IC:** a magnitude effect (label
  clipping, regularization), not ranking skill.
- **Low prediction agreement with a similar IC:** a different signal (blend
  candidate). High agreement with a small Δ: an incremental tweak.
- **Higher `best_iteration` or train–val gap with no test gain:** more
  overfitting.
- **Δ positive in most windows but the CI includes 0:** too small to detect
  at ~720 days. Say how many more windows or seeds would be needed, or that
  seed noise should be measured (run 2–3 seeds of each).

## Pitfalls

- Never overwrite a notebook the user may have edited. Use
  `set_text_cells.py` for text; `--force` only with consent.
- Notebooks open in VS Code: tell the user to Revert, not Save.
- The headless runner doesn't store outputs in the notebook. The user
  re-runs it in VS Code to see them inline; the plots are saved as PNGs.
- `DATA_FILE` is resolved as `RUN_DIR.parents[1] / "data.parquet"`, so run
  folders must sit at `<repo>/runs/<run>/`.
