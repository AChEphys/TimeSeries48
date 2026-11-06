# Project notes: a learning log

These are the working notes of the TimeSeries48 project. Claude (Claude Code) kept them
while the user, who owns this repository, worked through the project with Claude as an
analysis partner. Codex wrote the first project plan (`codex/project-plan.md`). The user
ran every training job and made the decisions, and Claude built most of the analysis
tooling. They are arranged as
chapters in roughly chronological order. Each chapter opens with the question, what
happened and the takeaways, followed by the detailed entries.

**How to read the entries.**
- **I-n (issue):** a problem in the project or scripts; its status says whether and how
  it was fixed.
- **N-n (note):** an insight, a concept worth remembering, or the analysis of a run.
- **T-n (task):** a planned experiment; the queue is in Appendix A.
- **Runs** are numbered in the order they were trained (`runs/run-<n>_<description>/`);
  `run-5.1`, `run-5.2` are the same setup with other random seeds. **Pre-runs** are
  validation-only sweeps (`pre-runs/`).
- **Windows** W1–W6 are the rolling test periods used for development (each ~120
  trading dates, 2022-12-29 → 2025-12-18). W7 (84 dates, to 2026-04-28) was reserved
  for the final evaluation.
- Numbers are in label units. Entries keep their original wording and dates; later
  corrections are marked in place rather than rewritten.

**Glossary of recurring terms.** *Rank IC*: the Spearman correlation between predictions
and labels across stocks on one date, averaged over dates. *Pearson IC*: the same with
values instead of ranks. *ICIR*: mean daily IC ÷ its standard deviation. *NW t*: a
t-statistic with Newey–West standard errors, which allow for the overlap between
neighbouring dates' labels. *Q1 … Q5*: quintiles of each day's predictions (Q1 lowest).
*Calibration slope*: the slope of labels on predictions (1 = sizes right, < 1 =
overconfident). *Demeaned*: each day's mean prediction subtracted. *Purged k-fold*:
cross-validation in contiguous date blocks with gaps around each validation block.
*Seed noise*: how much a result moves when only the random seed changes.

## Contents

- [Chapter 1. Understanding the task and the starter code](#chapter-1-understanding-the-task-and-the-starter-code)
- [Chapter 2. Measuring the baseline](#chapter-2-measuring-the-baseline)
- [Chapter 3. Making validation honest](#chapter-3-making-validation-honest)
- [Chapter 4. Tuning the trees and the outputs](#chapter-4-tuning-the-trees-and-the-outputs)
- [Chapter 5. Single-day feature engineering](#chapter-5-single-day-feature-engineering)
- [Chapter 6. Adding each stock's history](#chapter-6-adding-each-stocks-history)
- [Chapter 7. Past labels, and one model for both](#chapter-7-past-labels-and-one-model-for-both)
- [Chapter 8. Ideas that didn't help](#chapter-8-ideas-that-didnt-help)
- [Chapter 9. The final evaluation](#chapter-9-the-final-evaluation)
- [Appendix A. Experiment queue and decisions](#appendix-a-experiment-queue-and-decisions)
- [Appendix B. Index of issues and notes](#appendix-b-index-of-issues-and-notes)
- [Appendix C. Tooling](#appendix-c-tooling)

---

## Chapter 1. Understanding the task and the starter code

*2026-10-15 → 10-18*

Before training anything, the user went through the author's starter script line by
line with Claude, and Codex helped the user write the first project plan
(`codex/project-plan.md`).
The questions were basic but necessary: what is a boosted tree, what does each LightGBM
parameter do, how is one model per window shared by ~2,800 stocks, and what exactly is
the label?

**Takeaways**

- Some starter parameters did nothing (`subsample` without bagging, a `min_child_weight` worth one row), and the seeds came from the clock. The user fixed the seeds first, so any later difference between two runs could be trusted (I-2, I-3).
- Every run now saves its own folder: per-row predictions, learning curves, runtime and memory, plus a copy of the script that produced it (I-7). All later analysis rests on this.
- One model pooled over all stocks is reasonable here, because the label already has the market, industry and style parts removed (N-1, N-4).
- The label covers about five future trading days. That single fact drives the leakage questions of the next chapters (I-8).

### I-2. Ineffective or questionable LightGBM parameters

**Status:** Open (recorded 2026-10-16). Address before training the baseline,
or at least record the settings as they are.

**Where:** the `params` dict in [train_baseline.py](../train_baseline.py).

| Setting | Problem |
|---|---|
| `subsample: 0.8` | Has no effect. It is an alias of `bagging_fraction`, which needs `bagging_freq > 0` (default 0), and GOSS sampling replaces bagging anyway. Misleading, because it looks like 20% row sampling is happening. |
| `min_child_weight: 0.7544` | Has practically no effect. For the squared-error objective, each row contributes 1 to the hessian sum, so this means "at least ~1 row per leaf". |
| `min_data_in_leaf` (not set) | Defaults to 20 rows. With ~430k noisy training rows, leaves this small can fit noise. Probably the most important regularizer to explore (e.g. 500–5,000). |
| `reg_lambda: 0.98` (L2) | Likely negligible. It is added to the leaf's row count (often thousands), so it barely shrinks anything. |
| `reg_alpha: 0.98` (L1) | Possibly **strong**, not weak (corrected 2026-10-16). It is subtracted from the leaf's *summed residual*, which is in label units, and labels are small (~0.01). A leaf of a few hundred rows with a small mean residual can be zeroed out entirely. Its effect depends on the label scale; measure it (try 0 vs 0.98) rather than assume. Both odd values suggest a copy from a search on another dataset. See N-3. |
| `data_sample_strategy: "goss"` | GOSS keeps the rows with the largest errors. On noisy financial labels with fat tails, those are often outliers, so the model may focus on noise. Compare against plain `"bagging"`. |
| `gpu_platform_id`, `gpu_device_id` | Ignored with `device: "cpu"`. Harmless clutter. |
| `feval=r2_metric` | Redundant with RMSE on the same set (R² = 1 − MSE / variance of the labels), so it picks the same best iteration. It is also computed on the training set every round, which slows training slightly. |

### I-3. Time-based seeds make runs non-reproducible

**Status:** Resolved 2026-10-18: the user set `RANDOM_SEED` (row shuffle) and `RANDOM_SEED + i` (LightGBM seed per window). Recorded 2026-10-16.

`np.random.default_rng(int(time.time()))` (row shuffle) and
`"seed": int(time.time())` (LightGBM sampling) change on every run, so results
cannot be reproduced exactly, and differences between two experiments mix
real effects with random luck. Fix: use a constant seed (e.g. 42). To measure
how much randomness matters, run a few fixed seeds and report the spread.

### I-7. Run outputs and analysis records

**Status:** Resolved 2026-10-18.
- The user added the run folder `runs/<name>/`, `RANDOM_SEED`, the `SMOKE_TEST`
  flag, per-window `runtime`, and fixed the `train_end` label.
- Claude added (lines tagged `# BY: Claude`):
  - `peak_memory_gb` per window: the process's peak resident memory (Linux
    `VmHWM`), reset at the start of each window. It includes the loaded
    dataset.
  - `predictions_<window>.parquet` (Date, Code, label, Prediction).
  - `curves_<window>.json` (per-iteration train/val metrics). It is flushed
    every 50 iterations during training, so a notebook can plot progress
    live; the final write adds `best_iteration`.
  - `parents=True` on `MODEL_DIR.mkdir`, which previously would have crashed
    because `runs/` didn't exist.
  - 2026-10-19: `main()` archives a copy of the script into the run folder
    (a record only; always run the copy under ROOT), and writes and opens
    `runs/<name>/curves.ipynb`. Its cell redraws every 60 s while the
    training process (by PID) is alive, then draws a final plot and stops.

### I-8. What exactly is the label? (parked)

**Status:** Parked 2026-10-21 (user decision). Recorded 2026-10-18.

The task describes the label as a dimensionless, factor-adjusted 5-day forward return:
the part of a stock's return over roughly the next five trading days that a Barra-type
risk-factor model doesn't explain. The author's exact construction is private and is not
part of this public copy.

**What we checked against the data.**
- **The horizon.** A stock's label, rank-correlated with its own label k trading days
  later, gives 0.78, 0.55, 0.34, 0.14, −0.04, −0.05 for k = 1 … 6. Five overlapping daily
  parts would give 0.8, 0.6, 0.4, 0.2, 0, 0. So the label covers about five days starting
  the next day: the label of date T is fully known only on date T + 6. This fact matters
  for the whole project: it causes the validation leak in I-1, and it sets the 6-date lag
  for any input built from past labels (N-32).
- **The daily mean** is ≈ 0 (std over days 0.0027): the factor adjustment nearly
  demeans it, so the model only has to rank and size stocks against each other.
- **The scale and the tails.** Some details of the scale didn't match our reading of the
  description, and the tails are heavy (a few new listings and speculative price-limit
  streaks). We sent questions to the author.

**Decision.** The open details change neither training nor any ranking metric (IC and
Rank IC don't depend on the scale). They matter only for reading values as returns. So we
stopped there, and from then on:
- treat the label as *something like a 5-day forward residual return*;
- report predictions and errors in **label units**, never as percentages;
- attach no further financial meaning to label values.

**Lesson.** Check a data description against the data early, write down which facts the
project actually depends on (here the horizon, because of leakage), and park the
questions whose answers wouldn't change any decision.

### N-1. Same-day correlation between stocks

**Recorded:** 2026-10-16.

#### How the baseline treats stocks

Each row (one stock on one date) is an independent example: that stock's
selected features → one prediction for that stock. A tree never sees other
stocks' features. `model.predict(test_x)` processes all rows at once only for
speed; each output depends on its own row alone.

#### Why same-day co-movement matters less than it seems

1. **The label has had the shared part removed.** Most same-day co-movement
   comes from the market, industries and styles (size, value, momentum), and
   the Barra adjustment subtracts exactly those. The residual is meant to be
   roughly stock-specific, which justifies per-stock modelling.
2. **The features may already be relative to the other stocks (unverified).**
   Values like 0.81 and -0.88 suggest per-day normalization across stocks
   (rank or z-score). If so, each feature already encodes "how this stock
   compares with the others today". To check: compute each feature's mean and
   standard deviation per date and see whether they are roughly constant
   across dates.
3. **IC only looks within a day.** Daily IC compares stocks on the same date, so
   only the within-day *ordering* counts. Shifting all of a day's predictions
   by the same amount leaves IC unchanged.

#### A consequence that does matter

The 2,800 stocks on one day are **not** 2,800 independent pieces of evidence:
they share that day's market conditions. The effective sample size is closer
to the number of days. This is another reason to split train, validation and
test **by date, never by row** (see I-1).

#### Ideas for using cross-stock information

| Difficulty | Idea |
|---|---|
| Easy | Add day-level context to each row: the day's cross-stock mean/std of features, or industry averages (needs an industry mapping, not in the data) |
| Easy | Rank or z-score features per day yourself, which makes them comparable across dates |
| Easy | Train on a per-day ranked or demeaned label, which matches what IC measures |
| Advanced | Models that take a whole day's stocks at once: transformers with attention across stocks, graph neural networks. Postpone until the simpler experiments are understood |

### N-2. LightGBM parameter guide and what to explore first

**Recorded:** 2026-10-16. Applies to the `params` dict in
[train_baseline.py](../train_baseline.py). See I-2 for the settings that are
broken or ineffective.

#### What each parameter controls

| Parameter | Value | Controls | Other options |
|---|---|---|---|
| `objective` | `"mse"` | Loss the trees minimize: squared error (alias of `regression`) | `huber` / `fair` / `regression_l1` (robust to outliers); `quantile`; `lambdarank` / `rank_xendcg` with one group per date (optimizes within-day ranking, closer to IC) |
| `metric` | `"rmse"` | Score tracked on the validation set; drives early stopping | `l2`, `l1`, `huber`, `ndcg` (for ranking objectives) |
| `device` | `"cpu"` | Hardware | `"gpu"` (OpenCL), `"cuda"` (Nvidia) |
| `num_threads` | `os.cpu_count()` | Parallel threads | LightGBM recommends the number of *physical* cores |
| `boosting_type` | `"gbdt"` | Boosting algorithm | `"dart"` (randomly drops trees; slower, awkward with early stopping), `"rf"` (random forest) |
| `data_sample_strategy` | `"goss"` | Which rows each tree trains on. GOSS keeps the ~20% of rows with the largest errors, plus a random ~10% of the rest (upweighted) | `"bagging"` (default; random rows, enabled by `bagging_fraction` < 1 and `bagging_freq` > 0) |
| `num_leaves` | `20` | Maximum leaves per tree, the main complexity control (default 31) | ~8–64 for noisy data |
| `max_depth` | `7` | Depth cap (default −1, unlimited). 2⁷ = 128 > 20 leaves, so it only blocks very lopsided trees | 4–8 or −1 |
| `subsample` | `0.8` | Alias of `bagging_fraction`. Unused here (see I-2) | 0.5–1.0, with `bagging_freq` ≥ 1 |
| `reg_alpha` | `0.98` | L1 penalty on leaf values; can zero out weak leaves (default 0). Possibly strong given the small label scale (see N-3) | 0 to large; scale depends on the labels |
| `reg_lambda` | `0.98` | L2 penalty; shrinks leaf values (default 0) | 0 to large (e.g. 10–1000 given big leaves) |
| `min_child_weight` | `0.7544` | Minimum hessian sum per leaf (≈ row count for MSE). Effectively unused | Use `min_data_in_leaf` instead |
| `seed` | `time.time()` | Randomness of the sampling. Non-reproducible (I-3) | Any fixed integer |
| `learning_rate` | `0.01` | Shrinkage: each tree's contribution is multiplied by this. Smaller means smoother learning but more trees | 0.005–0.1; check whether `best_iteration` hits the 2,000-round cap |
| `verbosity` | `0` | Log detail: −1 fatal only, 0 warnings, 1 info, 2+ debug | Use 1 while learning, to see warnings and info |

Not set, but worth knowing:
- `min_data_in_leaf`: minimum rows per leaf.
- `feature_fraction`: random subset of features per tree (default 1.0).
- `bagging_freq`: how often to resample rows when bagging.

#### Suggested exploration order (one change at a time, fixed seed)

1. `min_data_in_leaf`: 20 → 500, 2,000, 5,000.
2. `num_leaves`: 8, 20, 64.
3. `data_sample_strategy`: GOSS vs bagging (`bagging_fraction` 0.7, `bagging_freq` 1).
4. `feature_fraction`: 1.0 vs 0.7.
5. `learning_rate`: 0.01 vs 0.05 (faster experiments; check that IC holds up).
6. Later: `objective` `huber`, or a per-date ranking objective.

Judge by validation **and** test IC, `best_iteration`, and the train/validation
gap, never by training scores.

### N-3. Leaves, "weight", and L1/L2 penalties

**Recorded:** 2026-10-16.

#### Leaves

A tree routes each row through threshold questions (e.g. `feature12 < 0.3?`)
until it reaches a **leaf**, an end branch. Every training row that reaches the
same leaf gets the same output: the **leaf value**. A leaf's **size** is how
many training rows land in it. The leaf value is roughly the average of
those rows' residuals, so:

- **Small leaf (e.g. 20 rows):** an average of 20 noisy labels. Likely a fluke,
  so it fits noise and overfits.
- **Large leaf (e.g. 5,000 rows):** a stable average that is more likely to be a
  real pattern.

`min_data_in_leaf` (alias `min_child_samples`; default 20; **not in the
script's dict**, add it to `params` to set it) makes LightGBM reject any split
that would leave a child leaf with fewer rows than this.

#### "Weight" = hessian sum

Boosting uses each row's first and second derivatives of the loss with
respect to the current prediction p:

- gradient g = ∂loss/∂p
- hessian h = ∂²loss/∂p²

For squared error, ½(y − p)²: g = p − y (minus the residual), and **h = 1 for
every row**. Sums over the rows in a leaf: G = Σg, H = Σh. For squared error,
**H = number of rows**.

`min_child_weight` is the minimum H in a leaf. With h = 1, any leaf with at
least 1 row has H ≥ 1 > 0.7544, so the setting rules nothing out. (It's not a
conversion: a single row already satisfies it.) The concept exists because
for other losses h varies per row. For log-loss, h = p(1 − p) ≤ 0.25, so
"weight" there is not a row count.

#### Leaf value with penalties

```
no penalty:  w = −G / H                          = Σresidual / n       (mean residual)
L2 (λ):      w = −G / (H + λ)                    = Σresidual / (n + λ)
L1 (α):      w = −sign(G) · max(|G| − α, 0) / (H + λ)
```

`reg_lambda` = λ and `reg_alpha` = α are the coefficients of the penalty terms
½λw² and α|w| that are added to the loss for each leaf value w. The tree is then
scaled by `learning_rate` before being added to the model. The same penalties
also lower each candidate split's gain, so marginal splits get skipped.

- **L2 (ridge-style):** acts like λ extra "phantom" rows with residual 0,
  pulling the leaf value toward 0. Matters only when λ is comparable to n.
  With λ = 0.98 and n in the thousands, the effect is negligible.
- **L1 (lasso-style), "soft-thresholding":** subtracts α from the *summed*
  residual; if |Σresidual| < α, the leaf value becomes exactly 0. This is in
  label units, so its strength depends on the label scale.

Toy leaf: 4 rows with residuals 0.02, 0.01, 0.03, −0.02 (Σ = 0.04).

| Setting | Leaf value |
|---|---|
| No penalty | 0.04 / 4 = 0.0100 |
| L2, λ = 0.98 | 0.04 / 4.98 = 0.0080 |
| L1, α = 0.98 | \|0.04\| < 0.98, so **0** |

#### Why this fights overfitting

Each penalty says "don't trust a correction unless the evidence is strong":
L2 shrinks values from small leaves; L1 cancels corrections whose total
evidence is below a threshold; `min_data_in_leaf` forbids tiny leaves outright.
Noisy financial labels make all three relevant. `min_data_in_leaf` is the
easiest to reason about, because it is measured in rows.

Background: the same idea as ridge (L2) and lasso (L1) regression, where
penalties on the coefficients shrink them, or set them to zero, to reduce
overfitting.

#### Splits

A **split** is one threshold question at a node, "`feature_k` ≤ t?", that sends
rows left or right. To grow a tree, LightGBM groups each feature's values into
up to 255 bins (`max_bin`), tries every bin boundary on every feature, and keeps
the split with the largest **gain** (loss reduction). It grows leaf-wise: next
it splits whichever existing leaf offers the largest gain, until it reaches
`num_leaves`, `max_depth`, or no allowed split has positive gain.

#### Why the hessian sum, and why "weight"

Second-order Taylor expansion of the loss around the current prediction,
summed over a leaf's rows, for a leaf value w:

```
Σ loss(p_i + w)  ≈  const + G·w + ½·H·w²      →  minimized at  w = −G / H
```

This is a Newton step. H is the **curvature**: how sharply the loss pins
down w, i.e. how much evidence the leaf holds. For squared error every row has
curvature 1, so every row is equal evidence and H = n. For log-loss, rows
already predicted confidently have tiny h and count for little. "Weight"
(XGBoost's name) comes from two facts: the Newton step equals a weighted
least-squares fit with per-row weights h_i, and user-supplied sample weights
multiply h_i. For squared error, H is literally the total sample weight.

#### Are λ and α Lagrange multipliers?

Related, but not the same. Minimizing `loss + λ·w²` gives the same solution as
minimizing `loss` subject to `w² ≤ c` for some budget c, and λ is that
constraint's Lagrange multiplier (duality / KKT). The difference: a Lagrange
multiplier is *solved for* from a given budget c; here you *set* λ directly and
the implied budget follows. A Bayesian view also works: L2 is a Gaussian
prior on leaf values centred at 0, and L1 a Laplace prior.

#### Is α really in absolute label units?

Yes, based on LightGBM's C++ source (`feature_histogram.hpp`), recalled from
knowledge and not re-checked in this repo. LightGBM computes the leaf output as
`−ThresholdL1(sum_gradients, lambda_l1) / (sum_hessians + lambda_l2)`, where
`ThresholdL1(s, α) = sign(s) · max(|s| − α, 0)`. For the L2 objective,
gradient = prediction − label, with no automatic label scaling. So α is
compared with the **sum** (not the mean) of the leaf's residuals: a leaf is
zeroed when n · |mean residual| < α.

**Update 2026-10-18:** run 1 implies a label std of about **0.058** (RMSE ≈ std when R² ≈ 0), about 5× larger than the ~0.01 assumed below. So α = 0.98 is correspondingly **milder** than this table suggests: a leaf with mean residual 0.005 needs only ~200 rows to clear it.

Rough scale (original estimate, labels assumed ~0.01):

| Leaf rows n | Mean residual | n·\|r\| | Effect of α = 0.98 |
|---|---|---|---|
| 200 | 0.001 | 0.2 | Zeroed |
| 2,000 | 0.001 | 2 | Shrunk ~50% |
| 20,000 | 0.0005 | 10 | Shrunk ~10% |

With ~430k training rows and 20 leaves (~20k rows per leaf on average), the
model can still learn. Mainly, L1 acts like a label-scale-dependent minimum
leaf size: it cancels small, weak leaves and makes splits that create them
unattractive. As boosting removes signal, the remaining within-leaf residual
means shrink, so L1 zeroes more leaves over time and acts as a brake. It may
help trigger early stopping. To verify: measure the label std, and compare
`best_iteration` and IC with `reg_alpha` at 0 vs 0.98.

#### Do the brakes overlap? Which matters most?

- **L1 does not replace `min_data_in_leaf`.** L1 zeroes leaves with a small
  *summed* residual. A small leaf of **extreme** rows (e.g. 20 rows × residual
  0.1 = 2 > 0.98) passes L1 easily, and those outlier leaves are the typical
  overfitting risk with fat-tailed labels. GOSS makes this worse by keeping
  the large-error rows. Only `min_data_in_leaf` blocks them regardless of
  their values.
- **Leaf sizes are not ≈ n / num_leaves.** That is only the average. Leaf-wise
  growth makes lopsided trees: splits often carve out small regions in the
  tails of a feature, next to one huge leaf. Also, under GOSS each tree sees
  only ~30% of the rows, so the counts are smaller still.
- **`min_child_weight` = 0.7544 is fully redundant:** any leaf with
  ≥ 20 rows (the `min_data_in_leaf` default) has H ≥ 20.
- **Priority for controlling overfitting in this project** (most to least
  important):
  1. Honest early stopping, which needs leak-free validation (I-1).
  2. `min_data_in_leaf`: scale-free, interpretable, catches outlier leaves.
  3. `num_leaves` / `max_depth`: complexity per tree.
  4. Randomization: `feature_fraction`, and bagging instead of GOSS.
  5. Robustness to outliers: `huber` objective, or clipping extreme labels in
     training only.
  6. `reg_alpha` / `reg_lambda`: secondary. α depends on the label scale, so its
     meaning silently changes if the label is ever rescaled. λ would need values
     in the hundreds or more to matter.

  Whether the current brakes are already enough is an empirical question: look
  at the train-vs-validation gap and at `best_iteration`.

### N-4. One pooled model per window: stocks and time

**Recorded:** 2026-10-16.

Each rolling window trains **one model shared by all stocks and all dates** in
its ~230 history dates. The inputs carry no date or stock code. This is a
*pooled cross-sectional* (panel) model, the standard setup in quant equity.
Row order is not information, so shuffling changes nothing here.

#### Differences between stocks

- **Stocks differ through their feature values, not their names.** The model
  learns "stocks that *look like* this tend to do that", which transfers to any
  stock, including new listings.
- **Per-stock models are infeasible:** ~150 training rows per stock per window,
  for a very noisy target. Pooling shares evidence across ~2,800 stocks.
- **A stock-code feature would mostly memorize** each stock's average residual
  return, which tends not to persist, and is undefined for new codes. Market,
  industry and style differences are already removed from the label (Barra).
- Possible extension: stock-level *attributes* (industry, size), to allow
  interactions such as a feature working differently in small caps.
  Industry/size data are not in the dataset.

#### Changes over time

- **No date feature, deliberately.** Test dates always lie after the training
  range, and trees cannot extrapolate: every test date would fall into the
  "latest dates" branch. It would be useless or harmful.
- **The model treats feature → label behaviour as roughly constant within its
  window** (stationarity) and averages over ~230 days.
- **Rolling retraining is how the baseline adapts:** a new model every 120
  dates on the latest history. Window length trades off stability (longer)
  against adaptivity (shorter).
- **A single row has no memory** of the stock's own past beyond what the
  features already encode. Adding history is the project's main experiment
  (README, Stage 5).
- Ideas: time-decay sample weights (`lgb.Dataset(weight=...)`) that favour
  recent dates; day-level market-state features so the model can condition on
  the regime (see N-1); compare `LOOKBACK` lengths.

### N-5. `lgb.train` monitoring: valid_sets, feval, callbacks

**Recorded:** 2026-10-16.

#### `valid_sets=[train_matrix, validation_matrix]`

- Every listed set is **evaluated** (metrics computed) every iteration.
- **Early stopping ignores the training set.** LightGBM recognizes it as the
  same object as the training data and skips it. Only `"val"` decides when to
  stop.
- The training scores are included for **diagnosis**: training error keeps
  falling, so a growing gap between train and val curves signals overfitting.
- Cost: both metrics are computed on ~430k training rows every round, and the
  custom R² goes through scikit-learn, so this slows training a little.

#### `feval` vs `params["metric"]`

- `feval` **adds** to `metric`; it does not replace it. Both `rmse` and `r2`
  are computed on both sets. To use only the custom metric, set
  `"metric": "None"`.
- `r2_metric` returns `(name, value, is_higher_better)`. `True` tells early
  stopping to *maximize* R².
- With several metrics, early stopping checks all of them (on val) and stops
  when **any** one has gone `stopping_rounds` without improving
  (`first_metric_only=False`). On the same set, R² = 1 − MSE / variance of the
  labels, so R² and RMSE move together: same best iteration, redundant work.

#### Callbacks

A **callback** is a function you hand to a library for it to call at set
points. `lgb.train` calls each callback after every boosting iteration, passing
an environment object (iteration number, current evaluation results, model).

- `lgb.early_stopping(200)` remembers the best val score and iteration for each
  metric. After 200 rounds without improvement it raises an exception, which
  `lgb.train` catches to stop and set `model.best_iteration`. The model still
  *contains* the 200 extra trees, so `predict` passes
  `num_iteration=model.best_iteration`. (`save_model` saves up to the best
  iteration by default.)
- `lgb.log_evaluation(100)` prints the metrics every 100 iterations. Printing
  only; it does not affect training.
- **`lgb.record_evaluation(history_dict)`** stores every iteration's metrics in
  a dict. **Add it in Stage 3** to plot train vs val learning curves (the
  script currently keeps no history).
- `lgb.reset_parameter(learning_rate=...)` changes parameters during training,
  e.g. a learning-rate schedule. You can also write your own:
  `def cb(env): ...`.

#### Live training curves despite a blocking `lgb.train`

Callbacks run in the main Python thread between iterations, so a callback can
draw the plot itself (redraw every ~25 iterations, not every one):

- **Jupyter notebook (simplest; `ipykernel` is already a dev dependency):** the
  callback collects `env.evaluation_result_list` entries
  `(data_name, metric, value, higher_better)` and redraws with
  `IPython.display.clear_output(wait=True)` plus a matplotlib plot.
- **Plain script:** the callback appends each iteration's metrics to a file
  (e.g. JSON lines); a separate notebook or process re-reads and plots it. The
  devcontainer has no display, so GUI plot windows (`plt.ion()`) won't work.
- Heavier options: TensorBoard, MLflow, Weights & Biases.
- `matplotlib` is not in `pyproject.toml` yet: `uv add matplotlib`.
- Train and val RMSE sit at different levels, and val changes very little, so
  use two panels (or plot change from iteration 0) to see the val curve's
  shape.

---

## Chapter 2. Measuring the baseline

*2026-10-17 → 10-21*

Run 1 (one window) and run 2 (six windows) reproduce the author's baseline. This
chapter is about *how to measure* a cross-sectional return forecast: which metrics, how to
test them, and where the signal lives.

**Takeaways**

- Judge a forecast day by day: the daily IC compares stocks on the same date. Rank IC is the robust headline; Pearson IC was 1.6× higher because it rewards a few extreme stocks (N-6, N-8).
- Daily ICs are autocorrelated, because neighbouring labels share four of their five days. Naive t-statistics were ~1.7× too large, so every test from here on uses Newey–West standard errors (I-5, N-8).
- The script's `ir` was a t-statistic that grows with the window length, and its `kt` mixed days together (I-4, I-5).
- Keep the last window untouched until the very end, so the final verdict isn't tuned on (I-6).
- The edge is one-sided: the model spots losers (volatile, lottery-like stocks; short selling is limited in A-shares) but not winners, whose jumps are news-driven (N-12).
- Don't remove turbulent periods from the metrics; break the results down by month and by regimes known in advance instead (N-11).
- The author feeds the predictions into a portfolio optimizer, so **magnitudes** (R², calibration) count as much as ranking (N-13).

### I-4. `kt` is pooled over all test days; IC is per day

**Status:** Open (recorded 2026-10-17).

**Where:** `"kt": stats.kendalltau(prediction, real)` in the results dict of
[train_baseline.py](../train_baseline.py).

IC is computed **within each day** and then averaged, so it only judges how
stocks are ordered against each other on the same day. `kt` is computed once
over **all ~330k test rows from ~120 days pooled**, so it also compares stock A
on Monday with stock B on Thursday. Day-to-day level shifts in labels or
predictions then add to or subtract from `kt`, but cannot affect IC. The two
numbers measure different things.

Fix: compute Kendall (or Spearman, i.e. "Rank IC") **per day** and average, in
the same way as `daily_ic`. Keep the pooled version only if day-level
prediction is deliberately of interest.

### I-5. `ir` is a t-statistic, not a standard ICIR

**Status:** Open (recorded 2026-10-17).

**Where:** `"ir": valid_ic.mean() / valid_ic.std() * np.sqrt(len(test_dates))`.

**What it measures.** The **information ratio** in finance is mean active
return ÷ its volatility: reward per unit of inconsistency. Applied to the
daily IC series, mean(IC) / std(IC) is the **ICIR**: how *consistently* the
signal works. For example, mean IC 0.05 with std 0.05 is far more dependable
than mean IC 0.05 with std 0.20, which often flips sign.

**Problems with the script's version:**

1. Multiplying by √N turns it into the **t-statistic** for "mean IC ≠ 0"
   (mean / (std / √N)). That is a fine quantity, but it grows with the number of
   test days, so the final 84-day window scores about √(84/120) ≈ 0.84× lower
   for identical quality. Windows of different lengths can't be compared.
2. Daily ICs are **not independent**: labels overlap across ~5 days, so
   consecutive ICs are correlated. The t-statistic is therefore overstated,
   possibly by up to about √5 ≈ 2.2×.
3. Minor: it uses `len(test_dates)` rather than the number of finite ICs
   (`len(valid_ic)`), and `np.std` defaults to the population formula
   (`ddof=0`).

Fix: report the unscaled ICIR, mean(IC) / std(IC), which is comparable across
windows. If significance is wanted, use a t-statistic with an overlap
correction, e.g. Newey–West standard errors.

### N-6. IC (Pearson) vs rank metrics

**Recorded:** 2026-10-17.

| Metric | Uses | Computation |
|---|---|---|
| Pearson IC (the script's `ic`) | Magnitudes | Linear correlation of the values |
| Spearman / **Rank IC** | Ordering only | Pearson correlation of the *ranks* |
| Kendall's tau (the script's `kt`) | Ordering only | (concordant pairs − discordant pairs) / all pairs. A pair is concordant if prediction and label order the two stocks the same way |

**Outlier sensitivity.** Predictions [1, 2, 3, 4, 10] vs labels
[4, 3, 2, 1, 10]: the ordering of the first four stocks is completely wrong,
but one extreme stock is right.

- Pearson = **0.8**, driven by the single outlier.
- Kendall = (4 concordant − 6 discordant) / 10 = **−0.2**.

The labels here have fat tails (minimum ≈ −1.29), so Pearson IC can be
dominated by a few extreme stocks per day. Rank metrics are robust to that,
and quant reports usually show **IC and Rank IC side by side**.

**Scale.** For the same underlying association, Kendall's tau is
numerically smaller than Pearson or Spearman. For roughly normal data,
τ = (2/π)·arcsin(ρ) ≈ 0.64·ρ when ρ is small. Compare each metric only with
itself across models, never one metric's value with another's.

**Which to use.** Pearson IC rewards getting the size of the moves right as
well as the order; rank metrics reward only the order. A long-short portfolio
that weights stocks by predicted *rank* cares mostly about ordering. Report
both mean daily IC and mean daily Rank IC.

### I-6. Reserve the final window

**Status:** Open (recorded 2026-10-18). Add before the first full run.
`SMOKE_TEST` only trains window 1, but a full run would also train and score
window 7 (test dates 961–1044).

**Why.** Every time you look at test scores and then change something
(parameters, features, history design), those test dates become part of your
*selection* process. After many such rounds, the best-looking variant is partly
lucky on those dates, so its scores are optimistic (overfitting to the test
set through you, not through the model). Walk-forward training already keeps
each *model* from seeing its test data; the reserve keeps *you* from tuning on
the final dates.

**How.** Windows 1–6 are the development set: compare and tune there. In
Stage 6, freeze the baseline and the chosen multi-day variant(s), then run
window 7 **once** for each, using the same protocol. Report those scores next
to the development scores. Do not tune after seeing them. Implementation: a
flag (e.g. `FINAL_EVAL = False`) that stops the loop before the last window
unless set.

### N-7. Reading run 1 (reference baseline, window 1)

**Recorded:** 2026-10-18. Run folder: `runs/run-1_original-baseline-first-window`.
Train 2022-01-04 → 2022-12-14, test 2022-12-29 → 2023-06-30. 34 of 48 features
selected. 27 s, 2.24 GB peak.

| Metric | Value | Reading |
|---|---|---|
| best_iteration | 762 (stopped at 962) | Not capped by 2,000 |
| train / val / test R² | 0.020 / 0.0078 / 0.0074 | Tiny, as is normal for returns; R² ≈ IC² (0.084² ≈ 0.007) |
| mean daily IC | 0.084 | Solid; not suspicious for a composite ML signal on A-shares |
| `ir` | 14.6 | ICIR = 14.6 / √120 ≈ **1.33** → daily IC std ≈ 0.063; IC positive on ~90% of days (if roughly normal). Even dividing the t-stat by ~2.2 for overlap leaves ~6.6: clearly significant |
| `kt` (pooled) | 0.031 | Below the ~0.054 implied by IC; pooled across days and/or IC driven by outliers. Compute daily Rank IC to separate them (I-4, N-6) |

**Curves.** Val RMSE falls quickly until ~200 iterations, then is essentially
flat; the gains from 200 → 762 are ~0.00002, which is noise. Train keeps
improving, so the train/val R² gap widens (0.020 vs 0.0078). That is the
overfitting signature, but **benign** here: val never gets worse, so the
extra trees neither help nor hurt. A model at ~300 iterations would likely
score the same, faster.

**Leakage (I-1).** Val R² ≈ test R². The interleaved-validation optimism
predicted in I-1 is **small here**. Likely reason: a heavily regularized model
(20 leaves, learning rate 0.01, a few hundred trees) can't memorize individual
stock-days, so it can't exploit near-duplicate neighbouring labels. Leakage
hurts high-capacity models most. Still fix I-1 (principled, and the effect may
differ in other windows or with history features, where neighbouring rows
become even more alike), and check across all 6 windows.

**Runtime.** ~27 s per window, against the plan's 3–20 min estimate. A full
6-window run takes ~3 min, which makes seed-variance runs, purged k-fold and
many ablations affordable. Memory headroom is fine, even with 5× the
features for history.

### N-8. Run 2: reference baseline, windows 1–6

**Recorded:** 2026-10-19. Run folder:
`runs/run-2_original-baseline-six-windows`. Full analysis, plots and a findings
cell are in `analysis.ipynb`, with plots in `plots/analysis_*.png`.

| W | Test period | IC | Rank IC | ICIR | IC>0 % | t(NW) | val R² | test R² | Q5−Q1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2022-12-29 → 2023-06-30 | 0.084 | 0.045 | 1.33 | 96 | 8.9 | 0.0078 | 0.0074 | 0.0046 |
| 2 | 2023-07-03 → 2023-12-25 | 0.103 | 0.059 | 1.55 | 93 | 10.1 | 0.0102 | 0.0108 | 0.0074 |
| 3 | 2023-12-26 → 2024-06-27 | 0.133 | 0.090 | 1.64 | 96 | 10.1 | 0.0142 | 0.0169 | 0.0131 |
| 4 | 2024-06-28 → 2024-12-23 | 0.094 | 0.056 | 1.30 | 90 | 8.4 | 0.0146 | 0.0036 | 0.0071 |
| 5 | 2024-12-24 → 2025-06-25 | 0.072 | 0.052 | 1.15 | 83 | 7.8 | 0.0126 | 0.0023 | 0.0077 |
| 6 | 2025-06-26 → 2025-12-18 | 0.080 | 0.047 | 1.40 | 93 | 10.1 | 0.0100 | 0.0064 | 0.0082 |

Key lessons:
- The signal is positive and significant in every window. Daily IC
  autocorrelation 0.5–0.74 makes naive t-stats ~1.7× too high (Newey–West
  needed).
- Pearson IC ≈ 1.6× Rank IC: the predictive power sits in the extreme stocks.
  Report Rank IC as the conservative headline.
- The edge sits in the **bottom quintile** (spotting losers); quintiles 2–5 are
  nearly flat.
- Calibration is fine in the bulk; extreme negative predictions overshoot.
- val R² ≈ test R² in W1–3 but ≫ in W4–5, while IC holds. The miss is in
  magnitudes: regime (label-spread spikes in early and autumn 2024) or I-1? A
  chronological-validation run will separate them.
- The label's daily mean is ≈ 0 (std 0.0027), so pooled vs daily metrics barely
  differ (I-4 is low impact). Label: std 0.057, robust std 0.031, excess
  kurtosis 22.7 (I-8).
- Ideas prompted by the heavy tails: Huber objective, or clipping labels in
  training only; compare on Rank IC.

#### How good is run 2, by quant standards? (added 2026-10-20)

- Rough rules of thumb: a *single* factor with |Rank IC| ≈ 0.02–0.03 is usable,
  and > 0.05 is good. An ML *composite* of many features is expected to do
  better. Qlib's public LightGBM / Alpha158 benchmark on CSI 300 large caps
  reports Rank IC ≈ 0.05 (recalled from memory; different universe and
  horizon, so only a loose reference).
- Run 2: Rank IC 0.058 (Pearson 0.0945), positive every window, ICIR ≈ 1.3.
  That is a **respectable, plausible composite signal; not exceptional, not
  suspicious**.
- What a quant would ask next:
  1. Feature timing: are all proprietary features known by day T's close?
     Any look-ahead in them would inflate everything.
  2. Rank IC, not Pearson, as the headline.
  3. Long-only usefulness: the edge is mostly in the bottom quintile, and A-share
     shorting is limited.
  4. Turnover and costs.
  5. Tradability (limit-locked small caps) and capacity.
- For this project, the baseline is a solid benchmark. Judge history features
  by **paired daily Rank IC differences** against it.

#### Daily mean prediction vs daily mean label (run 2, added 2026-10-21)

Plot: `runs/run-2_original-baseline-six-windows/plots/analysis_daily_means.png`.

| W | pred mean | label mean | bias |
|---|---|---|---|
| 1 | 0.00054 | 0.00020 | +0.00034 |
| 2 | 0.00088 | 0.00074 | +0.00013 |
| 3 | 0.00002 | −0.00002 | +0.00003 |
| 4 | −0.00086 | 0.00079 | **−0.00165** |
| 5 | 0.00014 | 0.00006 | +0.00008 |
| 6 | 0.00040 | −0.00069 | +0.00108 |

- The daily label mean is ≈ 0 and small (std over days 0.0027, vs a
  cross-sectional std of ~0.055): the factor adjustment nearly demeans it.
- The daily *prediction* mean varies less (std 0.0011) and is uncorrelated with
  the daily label mean (−0.16). The model can't predict day-level shifts, and
  doesn't need to for IC.
- But predictions **shift en bloc** in shocks: early Feb 2024, **Oct 2024
  (≈ −0.012)** and early Apr 2025. When the whole cross-section's features
  look "loser-like" (e.g. volatile), the pooled model predicts broad losses
  that the demeaned label can't deliver. This explains W4's −0.0017 bias and
  part of its calibration failure; ranking (IC) is unaffected.
- Demeaning predictions per day (free, IC-neutral) raises test R²: W4
  0.0036 → 0.0054, W3 0.0169 → 0.0180, W5 0.0023 → 0.0029; others unchanged.
  It is only a partial fix: most of W4/W5's gap is overconfident spread, not
  the level. Candidate standard post-processing: per-day demeaning (plus
  shrinkage from N-8).

### N-10. Handling extreme labels

**Recorded:** 2026-10-20.

#### Three separate decisions

1. **Evaluation (test set): never filter by label value.** The label is the
   future; at date T you can't know which stocks will become extreme.
   Dropping them makes test scores optimistic (look-ahead). Test filters may
   only use information known at T.
2. **Universe (who is eligible at T):** exclusions knowable at T are
   legitimate in both training and test: each stock's first N trading days
   after listing, stocks halted or locked at their price limit (untradeable),
   ST stocks (not in the data). Compare variants on the same universe.
3. **Training labels:** past data, so they may be transformed freely, but judge
   every choice on the unfiltered test set.

#### Training options

| Option | Effect | Trade-off |
|---|---|---|
| Keep (MSE, current) | Tails dominate gradients; Pearson IC flattered | Noisy; chases mostly unpredictable events |
| **Clip (winsorize) per day**, e.g. at the 1st/99th percentile | Keeps the sign ("strong loser"), caps influence | Standard practitioner choice; one parameter |
| Huber objective (`objective="huber"`, `alpha` ≈ 1–2× robust std ≈ 0.03–0.06) | Squared error near 0, linear in the tails | Similar to clipping; also check `metric` |
| Per-day rank → normal ("rank-Gauss") labels | Model learns ordering only; matches Rank IC | Loses magnitude (calibration, R² no longer meaningful) |
| Drop rows by label | Discards real information | Not recommended; clipping keeps the direction |

#### Should we pursue the tails?

Mostly no. These tails are largely rumour- and theme-driven limit-up streaks
and new listings: hard to predict from features and often untradeable. The downside tail is partly predictable (the bottom-quintile edge), and
clipping keeps that direction while reducing noise.

#### Suggested experiment (after run 3)

One change at a time, scored on full test sets with Rank IC, Pearson IC,
Q5−Q1, Q1 and Q5 separately, and RMSE on raw labels:
- B: per-day clip
- C: Huber
- D: rank-Gauss
- E: exclude each stock's first N days (training and test; compare on the
  common universe)

Expect Rank IC up and Pearson IC possibly down. Choose by the intended use
(ranking → Rank IC). Keep early stopping consistent: either compute val loss
on the same transformed labels, or use a daily Rank IC `feval`. Apply the
chosen treatment to both baseline and history models so their comparison
stays fair.

#### Calibration, and W4 (added 2026-10-20)

- **Calibration** asks whether prediction *magnitudes* are right: among stocks
  predicted −0.02, is the average actual label −0.02? Ranking (IC) can be
  perfect while calibration is bad (predictions 10× too large still give IC 1).
- **Reading the plot:** per window, test rows are sorted by prediction into 20
  equal-count bins (5% each); dot = (mean prediction, mean label). On y = x
  means calibrated. Rising dots mean ranking works. Flatter than y = x means
  overconfident (too spread). Above the line on the left means predicted
  losses are bigger than the realized ones.
- **W4** (test 2024-07 → 2024-12): worst bin predicted −0.029, realized −0.018;
  near-zero bins realized ≈ +0.004. Overconfident on losers. This explains
  test R² 0.0036 while IC stayed 0.094: R² penalizes magnitude errors, IC
  doesn't. Plausible cause (unproven): the model learned big loser magnitudes
  from the early-2024 small-cap sell-off in its training data, then met the
  autumn-2024 rally.
- Calibration matters if predictions size positions or feed an optimizer, not
  for pure ranking. Fixes: shrink predictions by a slope fitted on validation,
  isotonic regression, or robust training (N-10).

#### Monthly breakdown: what really hurt W4/W5 (added 2026-10-20)

Per-month test R², pooled IC and calibration slope (slope of label on
prediction; 1 = calibrated, < 1 = overconfident) for W3–W6:

| Window | Bad test months (slope, R²) | Same window's good months |
|---|---|---|
| W3 | 2024-02 (0.35, −0.007) | 2024-04/05: slope 1.2–1.4, R² 0.02–0.05 |
| W4 | 2024-06 (0.27), **2024-09 (0.32), 2024-10 (0.18, −0.012)** | 2024-07/08, 12: slope 0.87–0.97 |
| W5 | 2025-02 (0.38), 2025-04 (0.40), 2025-05 (0.29) | 2025-01: slope 1.04 |
| W6 | none below 0.73 | (its training includes the Oct-2024 spike) |

**Correction** to the earlier guess ("W4 learned big losses from the early-2024
crash in its training data"): the failures sit in turbulent **test** months
(the Feb-2024 small-cap sell-off falls in W3's *test*; the Sep–Oct-2024
stimulus rally in W4's test). IC also drops in those months (0.02–0.04), so
ranking weakens too, and R² amplifies it. Training-period spikes are not
required: W6 trained through the Oct-2024 spike and was fine. W3 survived
its bad month thanks to exceptional Apr–May 2024.

**I-1:** the leak would inflate val in every window. Val R² is similar across
W3–W6 (0.010–0.015); the W4/W5 gap comes from the test side. So this pattern
does **not** point to the leak; W1–W3 (val ≈ test) suggest the leak is small.

**Implications:** clipping or per-day standardization of training labels
tames overconfidence (smaller, steadier prediction magnitudes) but can't
remove regime shifts in the test period. Options for those: market-state
features (N-1), scaling predictions by recent market volatility, or accepting
some bad months and judging by ICIR and the share of positive days.

#### Cross-model test: is W5's test period hard, or W5's model bad? (added 2026-10-20)

The saved W3, W4 and W5 models were all scored on **W5's test period**
(2024-12-24 → 2025-06-25; all three were trained before it, so no leak):

| Model (training period) | IC | Rank IC | Bad months (slope < 0.45) |
|---|---|---|---|
| W3 (2022-12 → 2023-12, no spikes) | 0.075 | 0.061 | Feb, Apr, May 2025 |
| W4 (2023-07 → 2024-06, incl. Feb-2024) | 0.077 | 0.056 | Feb, Apr, May 2025 |
| W5 (2023-12 → 2024-12, both spikes) | 0.072 | 0.052 | Feb, Apr, May 2025 |

All three models stumble in the **same months** by nearly the same amount, so
the difficulty lies in the **test period**, not in W5's training. A fresher
model didn't help either (the oldest was marginally best).

Lessons:
- **The label-std plot is not market volatility.** Labels are factor-adjusted,
  so market-wide moves (e.g. the April-2025 tariff shock) are removed. What
  remains is stock-specific dispersion. Hard months are ones where the
  feature → label relationship weakens (e.g. theme rotations), which needn't
  raise dispersion.
- **W3** survived its bad February because Mar–Jun 2024 were exceptional
  (monthly IC 0.12–0.23, Rank IC ≈ 0.13).
- **Differences between windows mostly reflect the market, not the model.**
  This is a strong argument for comparing variants **paired on the same
  days** (history experiments), never window against window.
- Fresher retraining didn't help here, so the relationship is fairly stable.
  `LOOKBACK` and retraining-frequency experiments are low priority.

### N-11. "Normal-condition" learning and metrics that ignore turbulence

**Recorded:** 2026-10-20.

**What the evidence supports.** Models trained on calm (W3) vs turbulent (W5)
periods behave almost identically on the same test period. So the learned
mapping is **stable** and **not dominated** by the training-period anomalies.
That is good evidence of robustness, but "stable" ≠ "true": all models share
the same features and the same blind spots.

**What it doesn't support.** "Failing in turbulent months proves the model
ignored noise" reasons backwards. A model that *had* fitted noise would fail
there too. Evidence against noise-fitting comes from cross-model agreement
and val ≈ test, not from the failures. And those months aren't "noise" to an
investor: they are real, unavoidable losses.

**Don't remove anomalous periods from the metrics.**
1. Look-ahead: "anomalous" is known only afterwards, so excluding such
   months cherry-picks the favourable ones.
2. Real use spans all periods.
3. For comparing models (history vs baseline), turbulence hits both on the
   **same days**, and paired daily differences already cancel most of it.
4. History features might help precisely in regime months; excluding them
   could hide the effect we are looking for.

**Do instead.**
- Keep the full-period numbers as the headline.
- Add **robust** summaries: Rank IC (less driven by tail stocks), median
  daily IC, % positive days, ICIR, and a trimmed mean as a sensitivity check.
- **Conditional** diagnostics, with regimes defined using only information
  available at T: e.g. split days by trailing 20-day market volatility or
  trailing cross-sectional dispersion, and report IC in each regime.
- **Paired** comparison: daily ΔRank IC (variant − baseline), with a
  Newey–West t-stat, and also per window and per regime.

#### Finer-grain analysis added to run 2 (2026-10-20)

`analysis.ipynb` section 5 adds a monthly table and plot
(`plots/analysis_monthly.png`), robust per-window summaries, and ex-ante
regimes (`plots/analysis_regimes.png`). Findings:
- Medians and trimmed means ≈ means: no window is driven by a few days.
- Overconfident months (slope < 0.5) occur in every window, 1–3 times each;
  W4/W5 had more of them and lacked strong months.
- Pearson and Rank IC can diverge within a month (2024-01: 0.088 vs −0.029),
  when the signal works only through extreme stocks.
- Trailing-dispersion regimes: "turbulent" (after a dispersion spike) has
  the *highest* Rank IC (0.086 vs 0.052). Shock onsets look calm ex-ante, and
  aftermaths are the best periods. Caveat: thresholds come from 2022 only.

**Why ex-ante regimes at all? (added 2026-10-20)** They don't predict shocks.
They test whether performance depends on conditions *observable at T*,
because only those can be acted on: trust or size the signal more in some
conditions, add regime features, or check whether a variant (e.g. history)
helps mainly in certain conditions. Ex-post labels ("that was a crash month")
explain the past but can't be used, and they invite cherry-picking. In run 2
the split gave one negative result (trailing dispersion doesn't flag shock
onsets) and one weak positive (the aftermath of dispersion spikes has a
stronger signal; only ~2–3 episodes, so far from proven). Calm days score
≈ the overall average; they are not worse. The per-regime calibration slope
panel turned out uninformative (all 0.72–0.86, within noise). Replace it with
a timeline of trailing dispersion against the thresholds, plus monthly Rank IC.

**Regimes reworked (2026-10-20).** `analysis.ipynb` now has its own section 6:
an explanation of the split, rolling one-year percentile thresholds, a
3-row timeline (`plots/analysis_regime_timeline.png`), and per-regime Rank IC
with Newey–West CIs, episode counts and a per-window breakdown
(`plots/analysis_regimes.png`). The uninformative per-regime calibration panel
was dropped. Run-2 result: calm 0.042 ± 0.012 (8 episodes), normal
0.053 ± 0.011 (11), turbulent 0.088 ± 0.018 (only 3 episodes; mainly the
post-Feb-2024 and post-Sep-2024 aftermaths; W1's turbulent stretch was
≈ 0). Calm ≈ normal. Spike onsets (2024-02-07, 2024-09-23, 2025-03-31) are
not flagged. This supersedes the earlier 2022-tercile result.

**Run-2 summary cell (2026-10-20).** `analysis.ipynb` ends with a summary:
overall performance, strengths and weaknesses, label tails and their effect
on training and evaluation, probable failure causes, achievements, and the
improvement list. That cell is the go-to recap of the baseline.

### N-12. Why the model spots losers but not winners

**Recorded:** 2026-10-20. Run 2's test rows, grouped by daily prediction
quintile (Q1 = lowest predicted). "Past" stats come from each stock's labels
over T−26 … T−7, which are known at T.

| Q | mean | median | % up | std | % > +0.2 | % < −0.2 | mean of \|label\| < 0.1 | past vol pct | past return pct |
|---|---|---|---|---|---|---|---|---|---|
| 1 | −0.0058 | −0.0099 | 39.4 | 0.074 | **1.73** | **0.68** | −0.010 | 0.62 | 0.58 |
| 2 | 0.0012 | −0.0038 | 44.1 | 0.048 | 0.70 | 0.06 | −0.003 | 0.45 | 0.48 |
| 3 | 0.0014 | −0.0037 | 44.2 | 0.047 | 0.67 | 0.05 | −0.003 | 0.44 | 0.47 |
| 4 | 0.0018 | −0.0038 | 44.2 | 0.049 | 0.76 | 0.06 | −0.003 | 0.46 | 0.48 |
| 5 | 0.0023 | −0.0043 | 44.1 | 0.057 | 1.09 | 0.12 | −0.004 | 0.52 | 0.50 |

All rows: mean 0.0002, median −0.0048, 43% up.

- **The data is right-skewed, not "losers everywhere":** the typical stock
  slightly underperforms (median < 0, 43% up), and a few big winners pull the
  mean back to ≈ 0. The winners are the fat right tail.
- **The winners aren't identifiable from the features:** big winners are
  spread across all quintiles, and Q1 has the *most* of them (1.7%). They are
  mostly news- or theme-driven jumps. So Q5 can't concentrate
  them, and Q2–Q5 look alike.
- **Q1 = volatile, speculative ("lottery-like") stocks:** the highest
  volatility (std 0.074, past-volatility percentile 0.62), slightly recent
  winners, *both* tails far fatter (11× more large losses), yet a negative
  centre. It is not only tail events: even within |label| < 0.1, Q1's mean is
  −1.0% vs −0.3%.
- **The model is neutral; the predictability is one-sided.** The well-known
  economic explanation: with short-selling constrained (as in A-shares),
  pessimists can't trade, so overpriced, attention-grabbing or lottery-like
  stocks stay overpriced and drift down, while underpriced stocks get bought
  up quickly (Miller 1977). Anomaly returns concentrate on the short side,
  more so where shorting is hard (Stambaugh, Yu & Yuan 2012). Investors also
  overpay for lottery-like payoffs (the MAX effect, Bali, Cakici & Whitelaw
  2011). Our proxies (past label mean and std) are crude and the features are
  opaque, so this is a consistent interpretation, not a proof.
- **Implications:** long-only use means "avoid Q1". Better upside needs other
  information (news, attention, flows); history features might capture some
  of it. Label clipping would also cut the right-tail noise that Q1 stocks
  contribute to training.

### N-13. What the predictions are used for: ranking and magnitude both matter

**Recorded:** 2026-10-21 (from the author, via the user).

The author uses the predictions to **optimize a portfolio**, so both the
**ordering** and the **size** of predictions matter.

- Evaluation: Rank IC (robust ranking) and Pearson IC, **plus** calibration
  slope, R² and RMSE (magnitudes) as co-primary. Earlier notes that call
  Rank IC "the headline" now read: the headline for *ranking*.
- Variants that only improve magnitudes (better calibration or R², same Rank
  IC) count as improvements. Variants that improve Rank IC but worsen
  calibration need a magnitude fix (or must be flagged).
- Rank-Gauss labels (N-10) discard magnitudes, so they are a poor fit. Prefer
  per-day clipping or Huber loss.
- Cheap magnitude fixes worth standardizing: per-day demeaning of
  predictions (N-8: raised W4 R² 0.0036 → 0.0054) and shrinkage fitted on
  validation data.
- The analyze-ts48-results skill was updated accordingly: the ground rules,
  verdict criteria, and magnitude deltas (slope, R², RMSE on common rows) in
  the comparison.
- The run-2 notebook's Findings and Summary wording was updated: no
  financial over-interpretation of the label, and magnitudes stressed.

---

## Chapter 3. Making validation honest

*2026-10-15 → 10-22*

The starter script validated on every third date. Because labels overlap, those
dates are near-copies of their training neighbours. This chapter fixes the protocol in
two steps (runs 3, 4, 5) and measures how much randomness alone moves the results (runs
5.1, 5.2).

**Takeaways**

- Interleaved validation leaks through overlapping labels. A chronological split with a purge gap, followed by a retrain on all history, removes the leak (I-1, run 3).
- With one validation block, the chosen tree count jumped around, and the tree count sets the *size* of the predictions (N-14, N-15).
- Purged 5-fold cross-validation (contiguous blocks, gaps on both sides, López de Prado) stabilized the tree count and fixed the magnitudes. Using all 48 features beat the correlation filter (N-16 … N-18).
- Three seeds of the same setup set a noise floor. Without it, a +0.002 Rank IC 'gain' can't be read (N-19).
- Result: **Run-5** (purged 5-fold, all features) became the reference.

### I-1. Interleaved validation dates leak through overlapping forward labels

**Status:** Open (recorded 2026-10-15). Address before/while training the baseline.

**Fix implemented 2026-10-21 (run 3, `runs/run-3_baseline-chrono-val-plus-retrain`).**
Oldest dates train, then a `PURGE` (= `GAP` = 10) date gap, then the newest
third (`VALIDATION_FRACTION`) validates: 143 / 10 / 77 of the 230 history
dates. Features are selected on training dates only. After early stopping,
the model is retrained on all 230 history dates with `best_iteration` trees
(no early stopping), and that model is saved and scored. Note for analysis:
the curves and val R² belong to the early-stopping fit; the test metrics belong
to the retrained model. Everything else (GOSS, MSE, params) is unchanged.
The retrain is also recorded: `curves/retrain_<window>.json` (keys `done`,
`rounds`, `curves.retrain.{rmse,r2}` on the full history), overlaid in green
in `curves.ipynb`.

**Where:** [train_baseline.py](../train_baseline.py), the train/validation split
(`j % 3 != 0` → train, `j % 3 == 0` → validation, around lines 79–85).

#### Problem

Every third history date is used for validation, so each validation date sits
between two training dates. The label for date T is a forward return over roughly
T+1 … T+6, so adjacent dates' labels share most of
the same future returns:

```
label(T)   : returns T+1 … T+6
label(T+1) : returns T+2 … T+7   ← shares 4 of the 5 daily returns
```

Features are also likely autocorrelated from day to day. The model is therefore
validated on near-copies of what it trained on (interpolation, not forecasting).

#### Consequences

- Validation RMSE / R² are optimistic.
- Early stopping relies on that signal, so `best_iteration` tends to be too high,
  which leans toward overfitting the window's market conditions.
- Test scores remain honest: the 10-date `GAP` separates history from test. The
  leak affects model *selection* only.
- Related, smaller leak: correlation-based feature selection (`pl.corr` over
  `history`) uses labels from the validation dates too. Select features on
  training dates only.

#### Options

| Option | Pros | Cons |
|---|---|---|
| Interleaved single dates (current) | Validation covers the same market conditions as training; training includes the most recent dates | Leaks; optimistic validation; inflated `best_iteration` |
| Chronological: first ~2/3 train, purge gap of at least 6 dates (use 10), last ~1/3 validate | Leak-free; realistic (past → future); simple | One validation period, so the score is noisy; most recent dates are not trained on, unless you retrain on all of them |
| Purged k-fold: contiguous blocks (~20 dates), each block validated in turn, with gaps dropped on both sides | Leak-free; averages over several market conditions, so the score is more stable | Some folds validate the past using a model trained partly on the future; more compute |

Key insight: "use recent data" vs "avoid leakage" is **not** a real trade-off.
Choose the number of trees with a clean chronological split, then **retrain on
all history dates** with that count. The genuine trade-off is between a
*realistic but noisy* validation score (one chronological block) and a *stable
but less realistic* one (purged k-fold). Interleaved single dates are purged
k-fold with one-date blocks and no purge: the extreme case that leaks.

Reference: López de Prado, *Advances in Financial Machine Learning*, Chapter 7
(purged k-fold cross-validation, embargo).

#### Suggested plan

1. Default: chronological split with a 10-date purge gap, then a full retrain
   on all history dates using the chosen `best_iteration`.
2. Diagnostic exercise: on one window, run interleaved vs chronological
   validation and compare `best_iteration` and the validation-vs-test score gap.
   This shows the leakage directly.
3. Move to purged k-fold only if choices swing a lot between windows.

### N-14. Run 3 curves: tree counts, feature selection, retrain R²

**Recorded:** 2026-10-21. Run 3 = chronological validation (oldest 143 dates
train, 10 purge, newest 77 validate), features selected on the training dates,
retrain on all 230 history dates (I-1 fix).

**Tree counts.** Run 2: 762, 612, 837, 770, 641, 1367. Run 3: 209, **1487**,
278, 708, **59**, 116.
- Iteration range where val R² is within 5% of its peak:
  - Run 2: ~200–400 → ~800–1,500 in every window, and all six ranges overlap.
  - Run 3: 131–408, 724–1686, 166–477, 318–907, 40–73, 89–181. These don't
    overlap.
- Cause: a single chronological validation block is a single market period.
  The tree count measures how well the training period transfers to that
  period.
  - W5 validates Aug–Dec 2024, which includes the Sep–Oct stimulus rally. Val
    R² peaks at 0.002 and val RMSE then rises, so it stops at 59 trees.
  - W6 validates Feb–Jun 2025, the hard months of N-8, and stops at 116.
  - W2's validation curve is a long, flat plateau, so the argmax lands on
    noise (1487).
- Run 2's interleaved validation shared its market conditions with training,
  so the curve shapes were alike in every window. This stability was partly
  flattering: run-2 validation R² was 0.008–0.015, versus 0.002–0.014 in run 3.
- This is the "realistic but noisy" side of the trade-off in I-1.
- First look at the test metrics (results JSON): test IC/R² ≈ run 2 in every
  window despite 59 vs 641 trees (W5), so the test score seems insensitive to
  tree count over a wide range. The test analysis will confirm or refute this.

**Feature selection.** The windows don't select the same features, in either
run.
- Run 2: counts 34–40; 28 in all six windows, 42 in at least one; Jaccard
  between consecutive windows 0.76–0.95.
- Run 3: counts 33–42; 26 in all six, 47 in at least one; Jaccard 0.59–0.94.
  It falls in W4→W6.
- Same window, run 2 vs run 3: 2–9 features differ. This confounds the
  comparison of the two runs.
- Cause: the |corr| values are tiny (median of the always-selected ones 0.046,
  max 0.082) and vary over time (e.g. feature40 ranges 0.002–0.063 across
  windows). 21 features cross the 0.02 bar in some windows; 1–8 sit within
  ±0.005 of the bar per window. Run 3 selects on 143 dates instead of 230, so
  its selection is noisier. No feature changed sign.
- The filter is linear and marginal. Trees can use weakly correlated features
  (non-linearly, or in interactions) and can ignore useless ones anyway.

**Retrain R² < first-fit training R² (every window).** This is expected and
harmless.
- It is in-sample R². The same number of trees spread over ~1.6× more rows
  memorizes less per row, so in-sample R² falls toward the out-of-sample
  value. The retrain curve lies between the train and val curves.
- The retrain set also includes the validation period, which may be harder or
  differently dispersed. Training-RMSE levels differ between periods for the
  same reason: RMSE ≈ that period's label std. E.g. W4's training period
  (Jul 2023 – Jan 2024) has RMSE 0.046, while adding the Feb–Jun 2024 period
  raises the retrain RMSE to 0.052.
- Not a quality measure; judge by test metrics.

**Options:**
- Tree count:
  - fix one count for all windows during experiments, which removes a noise
    source from paired comparisons;
  - or average validation curves over several purged blocks (purged k-fold)
    per window.
  - Check first, without training: score run 2's saved models on their test
    sets at truncated iterations (`num_iteration` = 50, 100, 200, 400, …, all)
    to see how flat test IC/R² is in the tree count.
- Features:
  - use all 48 (ablation: run 3 with no selection);
  - or select once, or with a hysteresis/stability rule.

### N-15. Run 3 vs run 2: protocol fix, tree count and prediction scale

**Recorded:** 2026-10-21. Notebook:
`runs/run-3_baseline-chrono-val-plus-retrain/analysis.ipynb` (What changed,
Findings, Summary, Hypotheses).

| W | trees r2 → r3 | Rank IC r2 → r3 | ΔRank IC ± CI | test R² r2 → r3 | slope r2 → r3 | agreement |
|---|---|---|---|---|---|---|
| 1 | 762 → 209 | 0.045 → 0.053 | +0.0077 ± 0.0035 | 0.0074 → 0.0078 | 1.05 → 1.47 | 0.87 |
| 2 | 612 → 1487 | 0.059 → 0.060 | +0.0012 ± 0.0028 | 0.0108 → 0.0106 | 1.13 → 1.01 | 0.88 |
| 3 | 837 → 278 | 0.090 → 0.093 | +0.0034 ± 0.0030 | 0.0169 → 0.0174 | 0.94 → 1.08 | 0.91 |
| 4 | 770 → 708 | 0.056 → 0.055 | −0.0004 ± 0.0019 | 0.0036 → 0.0050 | 0.63 → 0.69 | 0.92 |
| 5 | 641 → 59 | 0.052 → 0.061 | +0.0090 ± 0.0040 | 0.0023 → 0.0034 | 0.60 → 1.53 | 0.73 |
| 6 | 1367 → 116 | 0.047 → 0.052 | +0.0051 ± 0.0036 | 0.0064 → 0.0052 | 0.95 → 1.81 | 0.76 |

- **Overall:**
  - Rank IC 0.0581 → 0.0624, ΔRank IC **+0.0043 ± 0.0014** (NW t 6.0), 5 of
    6 windows; the gains are in calm/normal regimes.
  - ΔPearson IC +0.0016, ΔQ5−Q1 +0.0003.
  - Pooled R² 0.0076 → 0.0080; RMSE unchanged.
  - The pooled slope 0.81 → 0.98 is a cancellation: mean |slope − 1| per
    window got worse, 0.18 → 0.37.
- **Verdict:** better for ranking, not for magnitudes. Seed noise unmeasured.
- **Tree-count check (T-2):** saved models scored at truncated
  `num_iteration` on their test sets. Diagnostic only; it uses test data.
  - The **tree count sets the prediction scale** (lr 0.01 shrinkage): the
    per-day prediction std grows with trees. The calibration slope is ≈ 5 at
    25 trees, ≈ 1.4 at 200 and ≈ 1 at 500–800 in calm-ish windows. In W4/W5,
    ≈ 1 is reached already at ~100 trees.
  - The **ranking optimum differs by window:** run 2 peaks at ~200 (W1),
    ~300–500 (W2/W3) and 25–50 (W4–W6; W5 0.0905 at 50 vs 0.052 at 641).
    More trees raise Pearson IC but lower Rank IC in W4–W6: later trees fit
    the tails.
  - **At matched tree counts,** run 3 still leads by +0.002 … +0.005 in W1,
    W2, W3 and W6. It is equal in W4, and much worse in W5 at 50 trees (0.061
    vs 0.0905).
- **Interpretation:**
  - The gain comes from accidental tree-count reductions (W5/W6) plus a
    modest protocol/retrain effect.
  - The leak itself was small: with similar trees (W4), nothing changed.
  - The test period still dominates the levels.
- **Consequences for the plan:**
  - Purged k-fold should stabilize the tree count.
  - Decouple magnitude from the tree count with an out-of-fold scale
    correction (T-3).
  - In the folds, track daily Rank IC per iteration as well as RMSE/R²,
    since the ranking-optimal and RMSE-optimal counts differ.

### N-16. Purged k-fold design (run 4)

**Recorded:** 2026-10-22. Implemented in `train_baseline.py`: `purged_folds`,
`build_folds`, `purged_cv`, and the flags `VALIDATION_FOLD`, `SELECT_FEATURES`,
`MAX_ROUNDS`, `EARLY_STOPPING_ROUNDS`.

- **Folds.** The 230 history dates are split into 5 contiguous validation
  blocks of 46 dates. Each fold trains on all other dates except
  `PURGE` = 10 dates on each side of its block.
  - Edge folds (blocks 1 and 5) lose one purge gap: 174 training dates.
  - Middle folds lose two: 164 training dates.
- **Fairness.**
  - The validation blocks are equal: each scores the same amount of market.
  - The training sizes differ by ~6% (174 vs 164 dates). That shifts each
    fold's curve level and optimum only slightly, compared with the
    differences between market periods (e.g. run 3's val R² 0.002–0.014).
  - The final retrain uses 230 dates anyway, more than any fold.
  - Equalizing would mean dropping 10 more training dates from the edge
    folds, which throws away data for no real gain.
- **Feature selection.** This follows the standard rule for preprocessing in
  cross-validation: fit it inside each fold, then refit it on all data for the
  final model.
  - Each fold applies the `CORR_BAR` filter to its own training dates, so its
    validation labels stay unseen and the validation curves include the
    selection's own instability.
  - The final model re-applies the same rule to all 230 history dates.
  - Neither a union nor an intersection: those would build a set that no fold
    validated, and union/intersection have no counterpart at retrain time.
  - "Filter once on all history" was rejected: it is cheap, but it leaks
    validation labels into every fold.
  - `SELECT_FEATURES = False` uses all features everywhere (the T-1 ablation).
- **Tree count.** All folds are boosted in lockstep (`Booster.update()`),
  rather than with `lgb.cv`, to keep per-fold curves and a custom metric.
  - Early stopping uses the **fold-mean validation R²** (patience 200, cap
    2000). R² is scale-free, so each market period weighs the same; a mean
    RMSE would weigh volatile folds more.
  - R² = 1 − RMSE² / var(labels): exact, so no sklearn per iteration.
- **Rank IC.** The fold-mean daily Rank IC is computed every iteration, via a
  `feval` on the cached validation scores (≈ 7 ms per fold per iteration). Its
  argmax is reported as `rank_ic_best_iteration` but not used (N-15: the
  ranking-optimal and magnitude-optimal counts differ).
- **Outputs.**
  - `curves_<window>.json`: `curves.train` / `curves.val` hold fold means, so
    the analysis skill works unchanged. `folds.val_r2` and
    `folds.val_rank_ic` hold per-fold curves; `fold_validation` holds the
    date ranges.
  - `results_lgbm.json` adds `fold_*` lists, `rank_ic_best_iteration`,
    `cv_val_r2` and `cv_val_rank_ic`. `train_rows` / `validation_rows` are
    fold means.
- **Monitor.** The figure is 2×2:
  - training RMSE (fold mean + retrain);
  - R² (train / val fold means + retrain);
  - per-fold validation R² (thick mean sets the trees);
  - per-fold validation Rank IC (with its peak marked).
- **Cost:** about 5× the fitting time of a single fit, plus Rank IC. Estimated
  at 1–3 min per window.
- **Caveat:** middle folds validate the past with a model that has seen
  later market conditions. That is acceptable for choosing a tree count;
  performance is still judged only on the test windows.

### N-17. Runs 4 and 5: what the k-fold curves show (before test analysis)

**Recorded:** 2026-10-22.
- Run 4 = purged 5-fold with the per-fold `CORR_BAR` filter
  (`runs/run-4_baseline-sel-feat-purged-5-fold-val-retrain`).
- Run 5 = the same with all 48 features
  (`runs/run-5_baseline-all-feat-purged-5-fold-val-retrain`).

| W | trees r4 / r5 | fold-mean val R² r4 → r5 | fold-mean val Rank IC at best r4 → r5 | features per fold (r4) |
|---|---|---|---|---|
| 1 | 411 / 374 | 0.00554 → 0.00575 | 0.0710 → 0.0706 | 30–35 |
| 2 | 453 / 512 | 0.00572 → 0.00614 | 0.0507 → 0.0504 | 35–42 |
| 3 | 334 / 408 | 0.00927 → 0.00928 | 0.0577 → 0.0579 | 38–43 |
| 4 | 402 / 563 | 0.01022 → 0.01074 | 0.0521 → 0.0519 | 35–42 |
| 5 | 273 / 286 | 0.00688 → 0.00790 | 0.0474 → 0.0485 | 27–41 |
| 6 | 350 / 371 | 0.00543 → 0.00576 | 0.0720 → 0.0746 | 24–37 |

1. **Stable tree counts:**
   - Run 4: 273–453. Run 5: 286–563 (range ×2).
   - Compare run 3 (59–1487, ×25) and run 2 (612–1367).
   - The fold-mean val R² is flat: within 5% of its peak from ~170 to ~600
     iterations in every window. The remaining variation is noise on a
     plateau.
2. **Fold optima depend on the market period; the mean is a compromise.**
   - Per-fold argmax ranges from 64 to 754 within a window. Fold peak R²
     ranges from −0.001 to 0.024, which dwarfs any model difference.
   - Two kinds of folds:
     - steady folds keep improving to 400–750 trees (e.g. W4 fold 5,
       Apr–Jun 2024, R² 0.024);
     - **early-peak folds** peak at 60–150 trees and then decline. These are
       the known hard periods: Jan–Mar and Oct–Dec 2022, Sep–Dec 2024 (W5 fold
       5 ends below 0), Sep–Nov 2024, Jan–Jun 2025.
   - This matches N-15 (hard test periods prefer few trees). The k-fold
     averages over it but cannot remove it, since the period type isn't known
     ex ante.
   - No edge-fold effect: edge folds (174 training dates) are among both the
     best and the worst. The 174 vs 164 date difference doesn't show.
3. **Early peaks cluster near iteration 100:** 17 of the 23 fold optima below
   200 fall at 60–140.
   - LightGBM's GOSS only starts sampling after 1/learning_rate = 100
     iterations (all rows before that; from the C++ source, recalled, not
     re-checked).
   - Hypothesis: once GOSS focuses on large-residual rows, hard-period folds
     start to degrade.
   - The fold-mean training R² shows no kink at 100 (gain per iteration
     ~12% lower after than before, a smooth deceleration). So this is
     unproven.
   - Test: GOSS → bagging (T-5). If true, those folds should decline less
     after ~100.
4. **Ranking is learned early; later trees mostly add magnitude.**
   - The fold-mean val Rank IC is already 0.046–0.076 after one tree and
     nearly flat after ~100–200. Val R² keeps rising to ~300–500.
   - The R²-chosen count costs at most 0.0007 of mean val Rank IC versus its
     peak. So stopping on R² is fine for ranking too. The Rank IC peak
     iteration (9 … 730) is noise on a flat curve; don't use it.
   - Rank IC is jumpy in the first ~30 iterations: predictions take few
     distinct values then, so many stocks tie.
5. **All features ≥ filter in validation:**
   - Fold-mean val R² is higher in 5 of 6 windows (+0.0002 … +0.0010, i.e.
     +4% … +15%), equal in W3, and positive at 100/300/500 iterations alike.
   - Val Rank IC: about equal in W1–W4, better in W5 (+0.0011) and W6
     (+0.0026).
   - The largest gains are where run 4's per-fold selection was least stable
     (W5: 27–41 features, W6: 24–37). The filter's instability costs, and
     weakly correlated features still help the trees (non-linearly or in
     interactions).
   - Training R² is also higher (more capacity), but validation rose too: no
     sign of extra overfitting.
6. **Runtime:** 85–140 s per window, about 10–11 min per run, as estimated
   (N-16).
7. **Expectations for the test analysis:**
   - Run 5 ≥ run 4 (mostly via magnitudes).
   - Calibration more consistent than run 3, since all tree counts are in the
     270–560 range: slopes near ~1–1.3 in calm windows, and still < 1 in
     W4/W5 (hard test periods; N-15).

### N-18. Runs 4 and 5 on the test sets

**Recorded:** 2026-10-22. Notebooks: `runs/run-4_…/analysis.ipynb` (vs run 3)
and `runs/run-5_…/analysis.ipynb` (vs run 4, plus a leaderboard of runs 2–5
in section 15).

Leaderboard (each run's own test metrics, 720 dates):

| | Rank IC | Pearson IC | Kendall (daily) | R² (all rows) | ICIR (window mean) | mean \|slope − 1\| |
|---|---|---|---|---|---|---|
| run 2 | 0.0581 | 0.0945 | 0.0389 | 0.00762 | 1.395 | **0.18** |
| run 3 | 0.0624 | 0.0961 | 0.0420 | 0.00803 | 1.379 | 0.37 |
| run 4 | 0.0617 | 0.0972 | 0.0414 | 0.00850 | 1.392 | 0.19 |
| run 5 | **0.0631** | **0.0985** | **0.0424** | **0.00879** | **1.417** | **0.18** |

Per-window Rank IC (runs 2/3/4/5):

| W | run 2 | run 3 | run 4 | run 5 |
|---|---|---|---|---|
| 1 | 0.0450 | 0.0527 | 0.0506 | 0.0485 |
| 2 | 0.0590 | 0.0603 | 0.0639 | 0.0655 |
| 3 | 0.0900 | 0.0934 | 0.0938 | 0.0930 |
| 4 | 0.0556 | 0.0552 | 0.0581 | 0.0571 |
| 5 | 0.0520 | 0.0610 | 0.0527 | 0.0574 |
| 6 | 0.0469 | 0.0519 | 0.0512 | 0.0572 |

- **Run 4 vs run 3** (purged 5-fold):
  - ΔRank IC −0.0007 ± 0.0013 (t −1.1), 3 of 6 windows. W5 −0.0083 is run
    3's lucky 59 trees being lost.
  - R² 0.0080 → 0.0085, better in 5 of 6 windows. Mean |slope − 1| 0.37 →
    0.19.
  - Verdict: **better, magnitude kind**; ranking unchanged. Runtime ×3.2.
- **Run 5 vs run 4** (all 48 features):
  - ΔRank IC +0.0014 ± 0.0011 (t 2.5), but only 3 of 6 windows. Gains in W5
    (+0.0047) and W6 (+0.0060), i.e. a 2025 effect; W1 −0.0021, W4 −0.0011.
  - R² +3% (5 of 6 windows), calibration unchanged. Agreement 0.94.
  - Not a tree-count effect: at run 4's counts, run 5 is within 0.0006.
  - The dropped features (8–15 per window) have test daily |Rank IC|
    averaging 0.015–0.028, up to 0.069.
  - Verdict: **slightly better, not consistent**. Adopted as the new
    reference for simplicity.
- **Cumulative over the project** (run 2 → run 5): Rank IC +0.005, Pearson
  +0.004, R² +15%, with calibration error back to run 2's level.
- **Caveats:** 4 variants compared on the same 6 windows; seed noise
  unmeasured (T-4 is now the priority before reading small deltas).

### N-19. Seed noise: runs 5, 5.1, 5.2 (seeds 70563, 94086, 258128)

**Recorded:** 2026-10-22.
- The runs are identical except `RANDOM_SEED`. The seed drives GOSS's row
  sampling (`seed = RANDOM_SEED + i`) and the row shuffle before the retrain.
  Folds, features and data are fixed.
- Analysis: scratchpad scripts reusing the analysis skill's cells (each seed's
  own metrics, plus paired seed-vs-seed comparisons).

| Metric | 70563 | 94086 | 258128 | SD | SD / mean |
|---|---|---|---|---|---|
| Rank IC | 0.0631 | 0.0634 | 0.0633 | 0.0001 | 0.2% |
| Pearson IC | 0.0985 | 0.0980 | 0.0979 | 0.0003 | 0.3% |
| Kendall (daily) | 0.0424 | 0.0426 | 0.0425 | 0.0001 | 0.2% |
| Q5−Q1 | 0.0085 | 0.0085 | 0.0084 | 0.0001 | 0.6% |
| R² (pooled) | 0.00879 | 0.00875 | 0.00872 | 0.00004 | 0.4% |
| ICIR (window mean) | 1.417 | 1.416 | 1.410 | 0.004 | 0.3% |
| mean \|slope − 1\| | 0.183 | 0.190 | 0.185 | 0.003 | 1.8% |

- **Paired seed vs seed:**
  - Overall ΔRank IC is −0.0001 … +0.0003 (|t| ≤ 1.7), each seed better on
    ~50% of days.
  - Prediction agreement **0.987** (run 4 vs run 5: 0.94; run 2 vs run 5:
    0.88).
- **Per window,** the seed can matter: **W6 differs by up to 0.0019 Rank IC
  with paired t ≈ ±4.2** between seeds, and W4 reaches t = 2.1.
  - The paired NW test covers day sampling only; it treats the model as fixed.
  - So **per-window paired t-stats are not trustworthy on their own.**
- **Tree counts** vary 10–20% with the seed (e.g. W4: 563 / 446 / 466).
  Fold-mean validation R² is nearly identical (0.01074 / 0.01074 / 0.01072):
  the plateau is flat. Seed 70563 happened to give the most trees in every
  window.
- **Noise ceilings,** the 95% band for a difference between two single-seed
  runs of the same configuration, 1.96·√2·σ_seed:

| Metric | Seed ceiling | Compare: day-sampling 95% CI of paired ΔRank IC |
|---|---|---|
| Rank IC overall | ±0.0004 | ±0.0011 |
| Pearson IC overall | ±0.0009 | |
| R² overall | ±0.00011 (~1.3%) | |
| ICIR | ±0.011 | |
| mean \|slope − 1\| | ±0.009 | |
| Rank IC per window | ±0.0003 … ±0.0008; **±0.0029 in W6** | ±0.002 … ±0.005 |
| R² per window | ±0.00014 … ±0.00047 | |
| Slope per window | ±0.017 … ±0.046 | |

- **Conclusions:**
  1. Overall metrics barely depend on the seed. Day-sampling noise is ~3×
     larger, and combining the two widens the overall CIs by only ~10%.
  2. Earlier verdicts hold:
     - run 5 − run 4: +0.0014 Rank IC and +0.0003 R² are 3–4× the ceiling;
     - W5/W6 gains (+0.005 / +0.006) exceed even W6's ±0.003;
     - run 4 − run 3: −0.0007 stays "no difference";
     - small per-window differences (|Δ| < 0.001) are within seed noise.
  3. Per-window significance needs a higher bar, especially in W6.
  4. Calibration is systematic, not luck: slopes vary only ±0.02–0.05 with the
     seed.
- **Caveats:**
  - 3 seeds give 2 degrees of freedom, so σ is rough (the true value could be
    2–3× larger).
  - Only run 5's configuration was tested. More random configurations (bagging,
    `feature_fraction` < 1) may need their own seed check.
- **Presentation (user decision):** seed variants get no column of their own
  in `baseline_performance.ipynb`, and are not averaged.
  - The configuration's column shows its primary seed plus "k seeds: low to
    high".
  - A "Seed noise ±" column gives the ceiling per row.
  - Light green marks values within the seed noise of the best.

---

## Chapter 4. Tuning the trees and the outputs

*2026-10-23*

With an honest protocol in place, the user tried the obvious model-side levers: row
sampling (GOSS vs bagging), leaf size, tree size, and cheap fixes applied to the
predictions after training.

**Takeaways**

- GOSS vs bagging was a tie once seed noise was measured; a single-seed 'win' was corrected (N-22, N-23).
- Subtracting each day's mean prediction is free, leaves ranking unchanged and raised R² by ~7%. It became standard post-processing (**Run-5 (demeaned)**). Rescaling by an out-of-fold slope failed (N-24).
- The leaf-size and leaf-count sweeps found the author's tree settings already at an optimum. Tree tuning was exhausted, so the next gains had to come from the inputs (N-25, N-26).

### N-20. Bulk vs tails: what to balance, and how (run 6 onwards)

**Recorded:** 2026-10-23. Discussed with the user. The aim is to balance the
normal-looking bulk and the fat tails without overweighting the extremes or
cutting them off. All stocks matter to the portfolio optimizer.

**What the optimizer needs.**
- It needs each stock's **expected** 5-day return: the average outcome,
  including the small chance of a jump.
- MSE is the loss whose best answer is exactly that average. Clipping, Huber
  or MAE move the answer toward the median.
- So the MSE target is right. The problem is **estimation noise**: with fat
  tails, a few rows can set a leaf's value.

**Data check** (all 2.85 M rows with a stock history; labels demeaned per day;
ex-ante σ = the stock's own label SD over rows 7–26 back):

| Check | Result |
|---|---|
| Top 1% \|label\| rows: which stocks? | 2,548 of 2,988 stocks; half of these rows come from 588 stocks |
| Direction of the top 1% | **86% up**; overall skew 2.4 |
| Mean vs median by ex-ante σ decile | Median < mean in every decile: calmest −0.0033 vs +0.0003, most volatile −0.0115 vs −0.0034 |
| Ex-ante σ vs realized \|label\| (daily rank corr) | 0.26 (weak) |
| Squared-label mass by σ decile | 4% (calmest) … 22% (most volatile): concentrated, but far from all in a few stocks |
| Kurtosis: raw / label ÷ σ / label ÷ max(σ, day median σ) | 25 / 96 / 40 (normal = 3) |

- **Conclusions:**
  - The tails are **rare, mostly upward jumps that hit almost every stock**
    now and then, and past volatility barely predicts them. They are not a
    fixed set of extreme stocks.
  - **Per-day clipping (old T-6) mostly cuts upside.** It would bias
    predictions down, about twice as much for volatile stocks (mean − median
    gap 0.008 vs 0.004), exactly where the optimizer takes its biggest
    positions. Dropped.
  - **Inverse-variance weighting** (the textbook fix for uneven noise: weight
    rows by 1/σ², or divide labels by σ) doesn't work with trailing σ: the
    tails get worse. Not pursued.
- **Strategy:** keep MSE, and limit how much any single row can move the
  model:
  1. **Remove the amplifier:** GOSS → bagging (T-5, run 6). GOSS keeps the
     largest-gradient rows, which are the jumps.
  2. **Leaf regularization (new T-6):**
     - A leaf's value is ≈ sum(residuals) / (rows + `reg_lambda`).
     - The current minimum is 20 rows per leaf, out of ~500k training rows.
       One jump in such a leaf can set its value.
     - Raise `min_data_in_leaf` (~500–2000) and `reg_lambda` (similar order).
       Jumps still count in proportion to how often they happen, but they
       can't own a leaf.
  3. **Huber with a high threshold** (~99th percentile of \|residual\|) only as
     a measured ablation, with magnitudes checked per σ decile (T-8).
  4. **Later, if the tails are still the problem:** a two-part model: a bulk
     model plus P(jump | x) × typical jump size.
- **To judge the balance (evaluation to-do):** split calibration slope and
  R² into bulk (99% of rows) vs tails (top 1% \|label\|), and by ex-ante σ
  decile, so we see whether a change helps one group at the other's expense.
- Script: scratchpad `tails.py` (not kept). Recompute if needed: it's 30
  lines of polars.

#### Bagging parameters (run 6)

| Parameter | Value | Meaning |
|---|---|---|
| `data_sample_strategy` | `"bagging"` (was `"goss"`) | How rows are sampled per tree |
| `bagging_fraction` (alias `subsample`) | 0.8 | Share of rows drawn per tree, uniformly, without replacement |
| `bagging_freq` (alias `subsample_freq`) | 1 | Redraw every k trees; **0 (the default) disables bagging** |
| `bagging_seed` | derived from `seed` | Not set explicitly: `seed = RANDOM_SEED + i` generates it |

- **The author's `subsample: 0.8` never took effect:** GOSS ignores it, and
  even without GOSS, `bagging_freq` was 0.
- **Dropped GOSS settings** (defaults, never set): `top_rate` 0.2 (keep the 20%
  largest gradients), `other_rate` 0.1 (a random 10% of the rest, upweighted
  ×8). GOSS starts only after 1/learning_rate = 100 trees, so trees 1–100 used
  all rows.
- **How the values were chosen:**
  - `bagging_freq` = 1: a fresh sample per tree gives the most decorrelated
    trees, at no cost; the standard choice.
  - `bagging_fraction` = 0.8: the author's intended value, so run 6 changes
    only *how* rows are picked, not the intended amount.
  - The useful range is ~0.5–0.9 (Friedman's stochastic boosting found
    ~0.5–0.8 best). Lower adds more regularization and speed, but noisier trees.
  - With ~500k rows per fold, even 0.5 leaves ~250k per tree, so the value is
    probably not critical.
  - **If tuned:** pick it on the k-fold fold-mean validation R² / Rank IC
    (e.g. 0.5 / 0.7 / 0.8 / 1.0), never on test.
- **Seed:** bagging makes every tree random, while GOSS was deterministic for
  the first 100 trees. So seed noise may differ from N-19; a seed check may be
  needed for close calls.

#### Out-of-fold predictions (from run 6)

- `predictions/validation_<window>.parquet` (Date, Code, label, Prediction,
  fold): each history row scored by the fold model that left it out, at
  `best_iteration` trees. They cover every history date once.
- **Uses:**
  - T-3 (scale correction): fit the calibration slope here, apply it to test;
  - bulk-vs-tail diagnostics without touching test;
  - checking fold effects.
- **Caveat:** fold models train on ~4/5 of the history, so their scale is
  close to, but not exactly, the retrain's.
- The skills read only `predictions_*`, so these files don't enter the test
  metrics.

### N-21. Run 6 curves (GOSS → bagging), before the test analysis

**Recorded:** 2026-10-23. Compared with run 5 and its seed variants (5.1, 5.2),
whose spread is the yardstick.

| W | Trees 6 (5 seeds) | Max fold-mean val R², 6 (5 seeds) | Val Rank IC at best, 6 (5 seeds) |
|---|---|---|---|
| 1 | 331 (313–374) | **0.00587** (0.00575–0.00576) | 0.0709 (0.0705–0.0707) |
| 2 | 429 (463–512) | **0.00628** (0.00607–0.00614) | 0.0485 (0.0501–0.0504) |
| 3 | 421 (339–408) | **0.00944** (0.00928–0.00938) | 0.0569 (0.0579–0.0581) |
| 4 | 531 (446–563) | **0.01103** (0.01072–0.01074) | 0.0507 (0.0513–0.0519) |
| 5 | 277 (256–286) | **0.00806** (0.00790–0.00792) | 0.0491 (0.0485–0.0486) |
| 6 | 328 (336–371) | **0.00583** (0.00576–0.00578) | 0.0753 (0.0744–0.0746) |

- **Validation R² is up in all 6 windows,** by +1% to +3%, each above the
  3-seed range of run 5. A small but consistent magnitude gain.
- **Validation Rank IC is mixed:** 3 windows down (W2–W4, −0.001 to −0.002), 3
  up (W1, W5, W6). The 6-window mean is −0.0004. Ranking is roughly unchanged.
- **Tree counts, train–val gap, the per-fold optima pattern and runtime are
  unchanged** (within seed range). Bagging at 0.8 regularizes little. It
  changes which rows each tree sees, not how far the model fits.
- **Confirms that GOSS starts at 1/lr = 100 trees:** run 5's curves are
  identical across seeds up to tree 100 (e.g. Rank IC at trees 1, 5 and 50),
  so GOSS made no random draws before that. Run 6's early trees differ.
- **Early Rank IC spikes:** the fold-mean Rank IC "peaks" at tree 4 (W2) and 5
  (W6), e.g. W6 fold 3 hits ~0.107 at tree ~4. These are flukes of a
  few-tree model, whose predictions take only a few distinct values, plus one
  lucky random sample. They have no effect on the tree count, which follows R².
  The `rank_ic_best_iteration` field and the monitor's dotted line become
  misleading. A burn-in (e.g. ignore trees < 50) would fix the display.
- **Expectation for the test:** small R² / calibration gains, Rank IC about
  the same. The hypothesis: GOSS's ×8 upweighting of sampled small-gradient
  rows, plus its keeping of the large-gradient (tail) rows, added noise to the
  leaf values.
- **Caveat:** bagging's seed noise is unmeasured (N-20). Run 5's seed spread
  is the best available yardstick.

### N-22. Run 6 vs run 5 on the test sets: GOSS → bagging

**Recorded:** 2026-10-23. Notebook:
`runs/run-6_baseline-all-feat-bagging-rows-purged-5-fold-val/analysis.ipynb`.
- **What changed:** only the row sampling. `"goss"` became `"bagging"`, with
  `bagging_fraction` 0.8 and `bagging_freq` 1 (N-20). Out-of-fold predictions
  are now saved. Same seed, data and protocol; 100% common test rows.

| W | ΔRank IC (t NW) | vs 5.1 / 5.2 | Trees 5 → 6 | Slope 5 → 6 | Test R² 5 → 6 |
|---|---|---|---|---|---|
| 1 | 0.0000 (0.0) | −0.0001 / −0.0002 | 374 → 331 | 1.23 → 1.29 | 0.0090 → 0.0089 |
| 2 | −0.0025 (−1.9) | −0.0019 / −0.0021 | 512 → 429 | 1.11 → 1.14 | 0.0111 → 0.0111 |
| 3 | −0.0005 (−0.8) | −0.0007 / −0.0006 | 408 → 421 | 1.02 → 1.01 | 0.0179 → 0.0180 |
| 4 | +0.0009 (2.3) | +0.0005 / +0.0003 | 563 → 531 | 0.70 → 0.71 | 0.0050 → 0.0053 |
| 5 | +0.0027 (5.7) | +0.0031 / +0.0027 | 286 → 277 | 0.73 → 0.74 | 0.0040 → 0.0042 |
| 6 | +0.0045 (6.5) | +0.0026 / +0.0041 | 371 → 328 | 1.17 → 1.20 | 0.0071 → 0.0070 |
| All | **+0.0009 ± 0.0007 (2.5)** | +0.0006 / +0.0007 | −7% | pooled 0.910 → 0.919; mean \|slope − 1\| 0.183 → 0.198 | 0.00879 → 0.00885 |

- **Ranking:** Rank IC 0.0631 → 0.0640, beyond the seed ceiling (±0.0004)
  and ahead of every run-5 seed.
  - The gain comes from mid-2024 on: all 12 months of 2025 are positive, and
    Jul–Aug 2023 is −0.006 / −0.007.
  - It is in normal regimes (+0.0021, t 4.8).
  - ΔPearson IC +0.0001, ΔQ5−Q1 0; agreement 0.980 (seed vs seed: 0.987).
- **Magnitudes:** overall unchanged.
  - R² +0.7% is within seed noise.
  - Per-window calibration is slightly worse, through fewer trees (W1, W2, W6
    underconfident).
- **Bulk vs tails** (first use of the N-20 evaluation):
  - Calm stocks (ex-ante σ deciles 1–2): R² 0.00225 → 0.00245 (+9%), slope
    0.79 → 0.84, both outside the seed range.
  - Volatile stocks (deciles 9–10): unchanged (R² 0.0143, slope 0.95).
  - Bagging helped the bulk without hurting the tails.
  - Don't use slope or R² on rows *selected by their label* (top 1% |label|):
    selecting on the outcome biases both (slope ~3.4, R² −0.9 for every run).
    Use ex-ante σ deciles.
- **Training-tail hypothesis:**
  - Per window, ΔRank IC correlates 0.81 with the top-1% share of the training
    sum of squares (0.317 / 0.315 / 0.351 / 0.365 / 0.370 / 0.371).
  - GOSS's tail focus may hurt more when the training history holds shocks.
    This is confounded with time: 6 points, and the share rises
    monotonically.
- **Correction (N-23, 2026-10-23):** with 4 bagging seeds, the gain shrinks to
  +0.0005 ± 0.0008 (a tie). The calibration loss is a bagging scale effect,
  not fewer trees.
- **Verdict:** **mildly better,** uneven over time (4/6 windows, W2 worse).
  - It misses the ≥ 5/6 bar for "better".
  - Bagging is conceptually cleaner (N-20), so it's a candidate new reference,
    pending the user's decision and a bagging seed check (T-9).
  - Configurations compared on these windows so far: 4 (runs 3–6).
- **Hypotheses and tests:**
  1. The bulk regains weight (medium confidence). Test with the out-of-fold
     calm-stock calibration, or a GOSS re-run that saves out-of-fold
     predictions.
  2. Heavier-tailed training sets gain more (low confidence). Test by scoring
     W6's GOSS and bagging models on W5's test dates.
  3. The W2 loss is unexplained, possibly seed noise (T-9).

### N-23. Bagging seed noise (runs 6, 6.1, 6.2, 6.3): bagging vs GOSS is a tie

**Recorded:** 2026-10-23.
- Bagging seeds: 70563, 94086, 258128, plus 6.3 with `int(time.time())`, which
  was not recorded, so run 6.3 isn't reproducible. The scripts differ only in
  the seed lines (checked).
- GOSS reference: runs 5, 5.1 and 5.2.
- Script: scratchpad `seeds6.py`.

| Metric | GOSS seeds (5) | Bagging seeds (6) | Ceiling GOSS → bagging | Bagging − GOSS (seed means) | Seed-only t |
|---|---|---|---|---|---|
| Rank IC | 0.0631–0.0634 | 0.0634–0.0640 | ±0.0004 → **±0.0007** | +0.0005 | 3.3 (ranges separate) |
| Pearson IC | 0.0979–0.0985 | 0.0984–0.0986 | ±0.0009 → ±0.0004 | +0.0004 | 1.8 |
| R² | 0.00872–0.00879 | 0.00877–0.00885 | ±0.00011 → ±0.00011 | +0.00006 | 2.0 |
| Pooled slope | 0.910–0.925 | 0.909–0.927 | ±0.021 → ±0.022 | +0.002 | 0.4 |
| Mean \|slope − 1\| | 0.183–0.190 | 0.198–0.206 | ±0.009 → ±0.011 | **+0.015 (worse)** | 5.6 (ranges separate) |
| Trees (mean per window) | 359–419 | 365–404 | ±86 → ±50 | −4 | −0.2 |

- **Per-window Rank IC ceilings, GOSS → bagging:**
  - W1 ±0.0003 → ±0.0011;
  - W2 ±0.0008 → ±0.0012;
  - W3 ±0.0004 → ±0.0011;
  - W4 ±0.0008 → ±0.0005;
  - W5 ±0.0006 → **±0.0025**;
  - W6 ±0.0029 → ±0.0020.

  As expected, bagging's ranking is noisier, because every tree is random.
  Magnitude noise is about the same.
- **ΔRank IC per window, seed means (12 bagging × GOSS seed pairs):**
  - W1 −0.0003 (4/12 positive);
  - **W2 −0.0020 (0/12)**;
  - **W3 −0.0011 (0/12)**;
  - W4 +0.0004 (11/12);
  - **W5 +0.0015 (12/12)**;
  - **W6 +0.0044 (12/12)**.

  The split over time is systematic, not seed noise.
- **Overall ranking:**
  - All 12 pairs are ≥ 0 (min +0.0000). On these test days, the bagging edge
    is not seed luck.
  - Adding day sampling (NW on the seed-averaged daily Δ, plus the seed SE):
    **+0.0005 ± 0.0008, t 1.3.** Only 23 of 37 months are positive.
- **Magnitudes:**
  - R² is unchanged, except W4: +6%, ranges separate.
  - Per-window calibration is worse beyond noise: W1 1.24 → 1.29, W6 1.18 →
    1.21; both were already > 1, i.e. predictions too small.
  - At equal tree counts, bagging's predictions are ~3% smaller in W1 and W6
    (prediction SD). Plausible mechanism: without GOSS's weight on large
    gradients, leaf values come out smaller.
- **Calm stocks** (σ deciles 1–2): R² +7% and slope +0.04 on average, but
  seed 6.2 shows no gain. Suggestive only.
- **Verdict: no detectable difference** (tie). The combined CI includes 0, and
  the windows split 3–3 with stable signs. Bagging trades a small,
  period-dependent ranking edge for a small calibration loss.
- **Decision proposed:**
  - Keep bagging as the base. Uniform sampling suits T-6, which limits
    single-row influence, whereas GOSS boosts large-gradient rows. T-3 should
    remove the scale loss.
  - Record bagging as a tie, not a gain.
  - From now on, compare against seed-averaged results or the ±0.0007
    ceiling. Use fixed seeds only.
- **Corrects N-22:**
  1. The calibration loss isn't caused by fewer trees; seed-averaged tree
     counts are equal. It is a bagging scale effect.
  2. Run 6's +0.0009 Rank IC and +9% calm-stock R² were inflated by a lucky
     seed. Seed-averaged: +0.0005 and +7%.

### N-24. Post-rescaling (T-3) on run 6: the out-of-fold slope fails, per-day demeaning works

**Recorded:** 2026-10-23. Notebook:
`runs/run-6_baseline-all-feat-bagging-rows-purged-5-fold-val/post_rescaling.ipynb`
(built by `claude/post-rescaling/build_notebook.py`).
- The run-5 version follows after the user re-runs run 5 with out-of-fold
  logging. `train_baseline.py` has been reverted to run 5's GOSS setup.
  The original run-5 outputs are backed up in the scratchpad, to check
  reproduction.
- **Variants,** each fitted only on its window's out-of-fold history rows:
  - **(A)** `b × raw`, where b is the OLS slope of label on prediction;
  - **(C)** `raw − day mean`;
  - **(B)** `b_day × (raw − day mean)`.

| Metric (run 6) | Raw | (A) rescaled | (C) demeaned | (B) both |
|---|---|---|---|---|
| R² | 0.00885 | 0.00898 | **0.00947** | 0.00951 |
| Pooled slope | 0.919 | 1.024 | 0.963 | 1.035 |
| Mean \|slope − 1\| per window | 0.198 | 0.230 | **0.193** | 0.213 |
| R² calm / volatile stocks | 0.0024 / 0.0143 | 0.0027 / 0.0143 | 0.0026 / 0.0149 | 0.0028 / 0.0150 |
| Rank IC, IC, Q5−Q1 | identical (max per-day Δ 2·10⁻¹⁶) | | | |

- **Out-of-fold b** is 0.84–0.98 (shrink) in every window, but the raw test
  slopes are 0.71–1.29 (corr with b 0.57).
  - Within each window's history, the five folds' slopes range 0.53–1.95.
  - **The slope is period-dependent;** a history-fitted factor only fixes
    the average.
  - (A) helps W4/W5 and hurts W2/W3/W6.
- **(C) per-day demeaning:**
  - R² +7%, with calibration slightly better.
  - Strongest in W4 (0.0053 → 0.0068, the Oct 2024 shock, N-8).
  - Same pattern in all 4 seeds (R² +0.0006 each).
  - This confirms the 2026-10-21 run-2 observation (section "Daily mean
    prediction vs daily mean label").
- **Recommendation (pending the user):** adopt per-day demeaning as the
  standard output; don't adopt the out-of-fold slope.
- **Remaining per-window miscalibration:** a period effect. Idea: a
  regime-aware slope, i.e. predict the period's slope from ex-ante
  dispersion (new T-10).
- **Assumption check:** after (A), the pooled test slope is ≈ 1, so the
  out-of-fold → retrain scale transfer works on average.
- **Run 5 (GOSS), 2026-10-23:**
  - Re-run with out-of-fold logging; it reproduces the original exactly
    (same trees and metrics; predictions within 1.4·10⁻¹⁷).
  - Same result as run 6:
    - (A) pooled slope 0.910 → 1.033, mean |slope − 1| 0.183 → 0.234 (worse);
    - **(C) R² 0.00879 → 0.00942 (+7.2%), mean |slope − 1| 0.183 → 0.178**;
    - (B) 0.00944 / 0.217.
  - b is 0.83–0.94, corr with the test slope 0.41; fold slopes 0.53–1.84.
  - (C) on seeds 5.1/5.2: R² +0.0006 and mean |slope − 1| −0.005 each.
  - **After demeaning, GOSS vs bagging:**
    - R² ties (0.00934–0.00942 vs 0.00940–0.00947);
    - per-window calibration still favours GOSS (0.178–0.184 vs 0.193–0.201).
  - Notebook: `runs/run-5_baseline-all-feat-purged-5-fold-val-retrain/post_rescaling.ipynb`.

### N-25. Sweep leaf-1 (`min_data_in_leaf`, GOSS): bigger leaves hurt; small leaves carry real tail signal

**Recorded:** 2026-10-23.
- Sweep: `pre-runs/leaf-1_sweep-min-data-in-leaf_goss-5-fold-cv-all-feat`, purged
  5-fold validation only.
- 6 windows, seed 70563; reference = 20 (run 5).

| min_data_in_leaf | Trees | Train R² | Val R² (demeaned) | R² windows beating 20 | Val Rank IC | Rank IC windows beating 20 | Minutes |
|---|---|---|---|---|---|---|---|
| **20** | 419 | 0.0216 | **0.00800** | — | 0.0590 | — | 9.4 |
| 100 | 393 | 0.0193 | 0.00786 | 1/6 | 0.0585 | 1/6 | 9.1 |
| 200 | 397 | 0.0183 | 0.00762 | 0/6 | 0.0585 | 1/6 | 9.2 |
| 1000 | 598 | 0.0171 | 0.00684 | 0/6 | 0.0582 | 2/6 | 12.7 |
| 4000 | 351 | 0.0111 | 0.00591 | 0/6 | **0.0602** | 5/6 | 8.4 |

- **Where the R² is lost** (out-of-fold, day-demeaned, all windows pooled; the
  share of variance explained, split by rows):

| min_data | Total | From the bulk 99% | From the tail 1% (top \|label\| per day) | From calm stocks (σ deciles 1–2) | From volatile stocks (deciles 9–10) |
|---|---|---|---|---|---|
| 20 | 0.00791 | 0.00876 | −0.00084 | 0.00004 | 0.00530 |
| 200 | 0.00752 | 0.00878 | −0.00126 | −0.00003 | 0.00514 |
| 1000 | 0.00669 | 0.00855 | −0.00186 | −0.00014 | 0.00482 |
| 4000 | 0.00578 | 0.00825 | −0.00247 | −0.00009 | 0.00410 |

- **Mechanism:** about 75% of the loss from 20 → 4000 (−0.00163 of −0.00213)
  comes from the **tail rows**, where predictions add error (negative
  contribution), and that error triples.
  - The bulk barely changes (−6%).
  - Small leaves let the model isolate jump-prone pockets (mostly volatile
    stocks; 86% of tail moves are up). Their predictions stay less negative
    there. Forcing big leaves merges those pockets into the general "volatile →
    loser" group (N-12), so the jumps cost more.
  - This is out-of-fold validated, so it's **real signal, not noise**. It
    **refutes the T-6 premise** (N-20) that small leaves mostly fit noise from
    jump rows.
- **Ranking vs magnitude trade-off:**
  - Coarse trees (4000) rank slightly better (Rank IC +0.0012, 5/6 windows),
    but lose 26% of R².
  - The bulk ordering prefers smooth trees; magnitudes need the fine,
    tail-aware leaves.
- **Where magnitude skill lives:**
  - Volatile stocks (deciles 9–10) supply ~2/3 of the explained variance.
  - Calm stocks supply ≈ 0 at every setting (the model can't size calm stocks
    at all).
  - The tail rows are net negative even at 20.
- Side effects at 1000: twice the trees in W4/W5 (1051 / 985), slope 0.84,
  12.7 min.
- **Decision:**
  - Keep `min_data_in_leaf` 20.
  - Drop T-6 stage 2 (`reg_lambda`): it shrinks exactly the small leaves that
    carry the tail signal (λ = 100 halves a 100-row leaf), so the same
    degradation is expected.
  - Next: test the opposite direction, **more capacity for fine partitions**
    (`num_leaves`, T-11). Then history features, which could target the tail
    penalty directly (jump-proneness).

### N-26. Sweep leaf-2 (`num_leaves`, GOSS): more leaves don't help either; the author's tree settings sit at an optimum

**Recorded:** 2026-10-23.
- Sweep: `pre-runs/leaf-2_sweep-num-leaves_goss-5-fold-cv-all-feat`, validation
  only, seed 70563.
- The 20-leaf reference was copied from leaf-1 (`min-data-20`).

| num_leaves | Trees | Train R² | Val R² (demeaned) | R² windows beating 20 | Val Rank IC | Rank IC windows beating 20 | Minutes |
|---|---|---|---|---|---|---|---|
| **20** | 419 | 0.0216 | **0.00800** | — | **0.0590** | — | 9.4 |
| 40 | 259 | 0.0230 | 0.00794 | 2/6 | 0.0576 | 0/6 | 10.7 |
| 80 | 230 | 0.0272 | 0.00776 | 0/6 | 0.0565 | 0/6 | 15.5 |
| 120 | 216 | 0.0279 | 0.00777 | 0/6 | 0.0563 | 0/6 | 17.0 |

- **Ranking:** more leaves fit the training data better, but lower validation
  Rank IC in every window (−0.0014 at 40, −0.0027 at 120).
- **Magnitude:** R² is flat to slightly lower.
- **80 ≈ 120:** the "no further splits" warnings suggest `max_depth` 7 caps the
  lopsided leaf-wise trees, so 120 builds about the same trees as 80.
- **Both sweeps together (N-25, N-26):**
  - Validation R² peaks at the author's settings (20 leaves, min 20 rows per
    leaf), and both coarser and finer trees lose R².
  - **Rank IC prefers simpler trees in both sweeps:** larger leaves +0.0012,
    more leaves −0.0027.
  - Ranking and magnitude pull in opposite directions, and the current setting
    is the best compromise for R².
  - Tree-shape tuning moves validation metrics by only a few percent at most.
- **Decision (proposed):** stop tuning the tree shape. The remaining knobs
  (`feature_fraction`, `learning_rate`, `reg_alpha`) are expected to move
  things by ~1–2%. Move to **information:** feature engineering (stage 5,
  T-12).

---

## Chapter 5. Single-day feature engineering

*2026-10-24 → 10-25*

Before adding history, the user asked whether the same day's data could be used better:
cross-sectional ranks, z-scores and day-level statistics of the 48 features, a blend of
models, and sizing the predictions by a forecast of each day's dispersion.

**Takeaways**

- Raw values carry the magnitudes; ranks and day statistics helped validation Rank IC but cost R² (N-27, N-28).
- Run 7's validation gains didn't carry over to the test windows: validation over-rewards features that describe the day's regime (N-29).
- Neither the blend nor the dispersion sizing beat the demeaned reference (N-30, N-31). Negative results, recorded with their reasons, and the family closed.

### N-27. Sweep feat-1, round 1 (single-day feature sets): raw values carry the magnitudes; day statistics help ranking

**Recorded:** 2026-10-24.
- Sweep: `pre-runs/feat-1_sweep-feature-sets_goss-5-fold-cv`
  (`test_single-day-feature-sets.py`), validation only.
- Seed 70563, run-5 tree settings. `raw` was copied from leaf-1.

| Feature set | Features | Trees | Val R² (demeaned) | R² windows beating raw | Val Rank IC | Rank IC windows beating raw | Minutes |
|---|---|---|---|---|---|---|---|
| **raw** | 48 | 419 | **0.00800** | — | 0.0590 | — | 9.4 |
| rank | 48 | 324 | 0.00742 | 0/6 | 0.0590 | 4/6 | 8.2 |
| z | 48 | 289 | 0.00660 | 0/6 | 0.0586 | 3/6 | 7.6 |
| raw+rank | 96 | 280 | 0.00767 | **4/6** | 0.0597 | 3/6 | 8.1 |
| rank+daystats | 144 | 288 | 0.00617 | 0/6 | **0.0607** | 3/6 | 9.6 |

- **Raw values carry the magnitude skill.** Every set without them loses R²
  (rank −7%, z −18%).
  - Plausible reason: the absolute level of a feature (e.g. a volatility-like
    feature) tells how *big* a stock's move tends to be. A rank keeps only the
    order within the day.
- **raw+rank:** R² beats raw in 4/6 windows (W2–W5, by +0.00003–0.00007), but
  collapses in W6 (0.00446 vs 0.00616). The mean is therefore lower.
- **Day statistics help ranking but hurt magnitudes:** rank+daystats has the
  best Rank IC (+0.0017; W3 +0.0041, W4 +0.0130) but the worst R² (−23%).
  - The day context lets the cross-sectional ordering change with the market
    state. Splits on day-level columns also move whole days, which is noise for
    a demeaned label.
- Single seed and validation only: differences below ~0.0001 R² per window can
  be seed noise.
- **Round 2 (user, 2026-10-24):** raw+daystats (and its halves raw+daymean,
  raw+daystd), raw+rank+daystats, and its
  halves raw+rank+daymean and raw+rank+daystd. The goal is to test whether
  raw values plus day context get both the ranking gain and the magnitudes.

### N-28. Sweep feat-1, complete (11 single-day feature sets): component effects, and a blend that gets both

**Recorded:** 2026-10-24.
- Sweep: `pre-runs/feat-1_sweep-feature-sets_goss-5-fold-cv`, validation only,
  seed 70563. Round 1 is in N-27.

| Feature set | Features | Train R² | Val R² (demeaned) | Val Rank IC | Minutes |
|---|---|---|---|---|---|
| **raw** | 48 | 0.0216 | **0.00800** | 0.0590 | 9.4 |
| rank | 48 | 0.0200 | 0.00742 | 0.0590 | 8.2 |
| z | 48 | 0.0190 | 0.00660 | 0.0586 | 7.6 |
| raw+rank | 96 | 0.0208 | 0.00767 | 0.0597 | 8.1 |
| raw+daymean | 96 | 0.0224 | 0.00673 | 0.0585 | 8.0 |
| raw+daystd | 96 | 0.0252 | 0.00671 | 0.0604 | 9.2 |
| raw+daystats | 144 | 0.0247 | 0.00664 | 0.0600 | 9.3 |
| rank+daystats | 144 | 0.0241 | 0.00617 | 0.0607 | 9.6 |
| raw+rank+daymean | 144 | 0.0239 | 0.00674 | 0.0591 | 9.5 |
| raw+rank+daystd | 144 | 0.0249 | 0.00678 | **0.0613** | 9.9 |
| raw+rank+daystats | 192 | 0.0242 | 0.00674 | 0.0605 | 10.0 |

- **Component effects,** each the same addition compared across contexts:
  - **+ raw:** R² +3% (to rank) and +9% (to rank+daystats); Rank IC ±0.
    Raw levels carry the magnitude skill.
  - **+ rank:** Rank IC +0.0005 … +0.0009 in all four contexts. R² ≈ 0, except
    raw → raw+rank at −4%, which is W6 alone.
    Small, consistent ranking help.
  - **+ daystd:** Rank IC +0.0014 (raw), +0.0016 (raw+rank). R² −15% … −16%.
    It carries the whole ranking gain of the day statistics: W3 +0.005, W4
    +0.012, i.e. 2024.
  - **+ daymean:** R² −16% for no ranking gain (Rank IC −0.0005 … +0.0001).
    Useless.
  - **z vs rank:** z is worse on both metrics.
- **Why day statistics cost R² but not Rank IC,** checked on the validation
  predictions:
  - With them, the daily spread of predictions varies more over days
    (coefficient of variation 0.23 → 0.31–0.34) but tracks the day's real
    return spread worse (corr 0.31 → 0.09–0.16). The per-day slope varies more
    (SD 0.73 → 0.93–0.97).
  - Rank IC ignores per-day scale; R² doesn't.
  - Training R² rises (0.0216 → 0.025) while validation R² falls: the 48 daily
    columns act as a fingerprint of each training day. The model can learn
    day-specific sizes that don't carry over to unseen days.
  - Raw alone sizes days best: raw feature levels show which days will be
    volatile.
- **Blend test** (validation predictions, demeaned, no training):
  `w × raw+rank+daystd + (1 − w) × raw`.

| w | 0 (raw) | 0.25 | **0.5** | 0.75 | 1 |
|---|---|---|---|---|---|
| Val R² (demeaned) | 0.00800 | **0.00816** | **0.00800** | 0.00754 | 0.00678 |
| Val Rank IC | 0.0590 | 0.0604 | **0.0614** | 0.0619 | 0.0613 |

  - Agreement between the two models: 0.787, i.e. two different signals.
  - **The 50/50 blend keeps raw's R² and adds Rank IC +0.0024 (+4%),**
    better than either alone.
  - Caveats:
    - w was picked on the same validation data, but only 5 values were tried;
    - single seed;
    - the per-window gains are concentrated in 2024 (W3/W4).
- **Next (T-13):**
  - Train raw+rank+daystd for real (run 7, `train_single-day-feature-sets.py`).
  - Blend its test predictions with run 5's, which already exist, so only run 7
    needs training.
  - Judge after demeaning against Run-5 (demeaned).
  - Keep daymean and z out.

### N-29. Run 7 (raw + rank + daystd, 144 single-day features) vs run 5 and run 2: validation gains didn't carry over

**Recorded:** 2026-10-24. Notebook:
`runs/run-7_single-day-rawfeat+rank+std-p5fcv/analysis.ipynb` (baseline: run 5,
leaderboard: runs 2/5/7).
- **What changed:** inputs only (48 → 144: raw + per-day rank + per-day SD of
  each feature), via `train_single-day-feature-sets.py`. Same seed, protocol,
  parameters and rows.
- The validation scores reproduced the sweep exactly.

| W | Rank IC 5 → 7 | ΔRank IC (t NW) | R² 5 → 7 | Slope 5 → 7 | Trees 5 → 7 |
|---|---|---|---|---|---|
| 1 | 0.0485 → 0.0527 | +0.0042 (2.0) | 0.0090 → 0.0056 | 1.23 → 1.36 | 374 → 259 |
| 2 | 0.0655 → 0.0626 | −0.0029 (−1.1) | 0.0111 → 0.0096 | 1.11 → 1.06 | 512 → 275 |
| 3 | 0.0930 → 0.0929 | −0.0001 (−0.1) | 0.0179 → 0.0180 | 1.02 → 1.20 | 408 → 319 |
| 4 | 0.0571 → **0.0333** | **−0.0238 (−3.5)** | 0.0050 → **0.0009** | 0.70 → 0.53 | 563 → 556 |
| 5 | 0.0574 → 0.0659 | +0.0085 (2.0) | 0.0040 → 0.0045 | 0.73 → 0.75 | 286 → 233 |
| 6 | 0.0572 → 0.0699 | +0.0126 (1.6) | 0.0071 → 0.0039 | 1.17 → 2.06 | 371 → 102 |
| All | 0.0631 → 0.0629 | −0.0003 ± 0.0044 (−0.1) | 0.00879 → 0.00679 (demeaned 0.00942 → 0.00709, −25%) | mean \|slope − 1\| 0.18 → 0.40 | |

- **Ranking:** a tie overall.
  - The W4 collapse is concentrated in Oct–Nov 2024 (−0.059 / −0.043), the
    Sep 2024 shock. Turbulent days −0.0087.
  - Gains in 2025 (W5/W6 are the best Rank IC of any run) and in calm/normal
    days.
  - Pearson IC −0.0087 (5/6 windows worse); agreement 0.79.
- **Magnitudes:** clearly worse.
  - W6 stopped at 102 trees (slope 2.06).
  - W4's top predictions are far too large.
  - Peak memory 3.4 GB (vs 1.6).
- **Not extrapolation:** only 8% of W4's daystd values in Oct 2024 were outside
  the training range (W3's Feb 2024: 34%, and W3 was fine).
  - Familiar inputs, wrong mapping: the day context learned from one shock (Feb
    2024) misread the next one.
  - Validation had promised the biggest gain exactly in W4 (+0.0115).
- **Vs run 2:**
  - Rank IC +0.0048 ± 0.0044 (t 2.1), 5/6 windows.
  - Pearson IC −0.0048, R² −11% (demeaned −7%), calibration 0.18 → 0.40.
  - Run 5 dominates run 7 as an improvement on run 2.
- **Verdict vs run 5: worse** (ranking a tie, magnitudes worse).
- **New caveat (protocol):** purged k-fold within a one-year history
  over-rewards regime-identifying features. A shock and its aftermath sit in
  neighbouring folds; the test brings a new regime. This applies to every
  day-context or history feature; validate those with held-out shock periods
  too.
- **Hypotheses / next:**
  - run 8 = raw+rank (no day statistics);
  - seed check of run 7;
  - T-13 blend with run 5;
  - a validation design that holds out a whole shock period.
- Configurations compared on these windows so far: 6 runs plus 11 feature sets
  on validation.

### N-30. Blend run 7 ⊕ run 5 (T-13): no gain, even in hindsight

**Recorded:** 2026-10-24. Notebook:
`runs/run-7_single-day-rawfeat+rank+std-p5fcv/post_blending.ipynb` (built by
`claude/post-blending/build_notebook.py`).
- **Blend:** `w × run7_demeaned + (1 − w) × run5_demeaned`, w = 0 … 1 in steps
  of 0.05.
- **Weight per window,** chosen on that window's validation predictions
  (purged k-fold history), then scored on test. Two rules: best validation R²
  or best validation Rank IC.

| Rule | Rank IC | ΔRank IC vs run 5 | R² (demeaned) | Mean \|slope − 1\| |
|---|---|---|---|---|
| run 5 alone | **0.0631** | — | **0.00942** | **0.178** |
| run 7 alone | 0.0629 | −0.0003 ± 0.0044 | 0.00709 | 0.408 |
| w_R² (0.05 / 0.25 / 0.05 / 0.45 / 0.35 / 0.15) | 0.0625 | −0.0007 ± 0.0012 | 0.00925 | 0.202 |
| w_RankIC (0.35 / 0.9 / 1 / 1 / 0.25 / 0.65) | 0.0605 | −0.0026 ± 0.0031 | 0.00781 | 0.356 |

- **Hindsight on test** (one w for all windows): the best Rank IC is +0.0003 at
  w = 0.35, within seed noise, with R² −2%. R² declines from w = 0.1.
- Validation had promised Rank IC +0.0029 near w = 0.75 with flat R².
- **Why it fails:** run 7's gains (2025, calm days) and its W4 collapse cancel.
  Validation overrates run 7 exactly where it fails on test (W3/W4), so weights
  chosen there backfire (the N-29 caveat again).
- **Decision:** Run-5 (demeaned) stays the reference. Run 7 is not adopted,
  alone or blended. T-13 is closed.
- Tooling: `claude/post-blending/build_notebook.py RUN --partner RUN2` builds
  this notebook for any pair with validation predictions.
- **Validation is anti-predictive for day-context features** (added 2026-10-24,
  from the sweep's saved validation predictions):

| W | ΔRank IC (raw+rank+daystd − raw), all folds | Last (most recent) fold only | Test |
|---|---|---|---|
| 1 | −0.0024 | +0.0014 | +0.0042 |
| 2 | +0.0029 | +0.0108 | −0.0029 |
| 3 | +0.0051 | +0.0179 | −0.0001 |
| 4 | +0.0115 | +0.0188 | −0.0238 |
| 5 | −0.0018 | −0.0452 | +0.0085 |
| 6 | −0.0012 | +0.0033 | +0.0127 |

  - Correlation with the test Δ over windows: **−0.93** (all folds, 0/6 signs
    right) and −0.53 (last fold, 2/6).
  - With 6 points, but the sign is wrong in every window.
  - Interpretation: what day-context features learn about the history's
    regimes tends to reverse in the next period.
  - **Protocol consequence:** purged k-fold validation can rank tree settings
    and similar knobs (N-25, N-26 behaved), but **not regime-identifying
    features**. Using the most recent fold doesn't fix it.
  - Feature families like these need out-of-period evidence, e.g. development
    windows (W1–W3) vs confirmation windows (W4–W6) on the test sets, stated
    honestly as multiple testing.

### N-31. Day-level context: what it can and can't do; sizing by forecast dispersion (T-10) fails

**Recorded:** 2026-10-24. Discussed with the user.
- **Three jobs for day-level context:**
  1. **Predict the day's level:** useless for a factor-adjusted label (daily
     label mean SD 0.0027 vs cross-section 0.055); demeaning handles it.
  2. **Change the cross-sectional mapping by regime:** real in finance
     (momentum crashes after rebounds; reversal stronger in volatile, illiquid
     markets; lottery and overpricing effects stronger after high sentiment).
     But it is identified by the number of regimes (1–3 per training year), not
     rows, and regimes differ (Feb 2024 crash vs Sep 2024 stimulus rally). Run 7
     failed here (N-29).
  3. **Sizing:** dispersion is persistent (lagged realized dispersion predicts the
     next day's at corr 0.45–0.51).
- **T-10 test:** `runs/run-5_baseline-all-feat-purged-5-fold-val-retrain/post_dispersion-via-HAR.ipynb`
  (built by `claude/post-dispersion/build_notebook.py`).
  - `final = c × σ̂_day × z`, z the run's per-day standardized prediction.
  - c is fitted per window on validation; σ̂ from A0 (trailing mean), A1 (HAR:
    day/week/month), A2 (+ model spread), A3 (+ feature-SD summary), plus the
    Oracle (realized σ).

| Variant | R² | Mean \|slope − 1\|, windows | Mean \|slope − 1\|, months |
|---|---|---|---|
| **Reference (demeaned)** | **0.00942** | 0.178 | **0.275** |
| Z (constant) | 0.00879 | 0.176 | 0.357 |
| A0 / A1 / A2 / A3 | 0.00896 / 0.00884 / 0.00901 / 0.00891 | 0.19–0.22 | 0.33–0.34 |
| Oracle (hindsight) | 0.00856 | 0.211 | 0.357 |

- **Forecast quality on test:** corr 0.13 (A0) to 0.28 (A3); none in 2025.
- **Even the Oracle loses:** the best daily scale is IC_day × σ_day, and IC drops on
  the most dispersed days.
  - By σ quintile: best scale 0.0043 / 0.0052 / 0.0058 / 0.0059 / **0.0053**;
    daily IC 0.100 / 0.104 / 0.107 / 0.101 / **0.080**.
  - corr(best daily scale, σ) = −0.01; corr(best daily scale, model's own
    spread) = **+0.32**.
- **The model already sizes days better than a dispersion forecast.** Removing its
  own spread (Z) costs 7% of R².
- Per window: the variants help in W4/W5 (after the Sep 2024 shock) and lose more
  in W3 (after Feb 2024).
- **Decision:** keep the demeaned output; T-10 closed (negative). Day-level sizing
  beyond the model's own would need IC timing by regime, which failed already.
- **Follow-up (user idea, 2026-10-24): combine the model's own spread with HAR**
  instead of replacing it (scratchpad test, run 5).
  - `final = k₁·(p − day mean) + (k₀ + k₂·σ̂)·z`, with weights fitted per window
    on validation by least squares.
  - R²: own + HAR A1 0.00928, own + A2 0.00932, own + A0 0.00921, own + Oracle
    0.00942, vs the reference 0.00942.
  - HAR adds nothing beyond the model's own spread, even with perfect hindsight.
  - The weights are unstable across windows: k₁ from −2.2 to +1.6. HAR and the
    model's spread are collinear, with only ~50 effective daily observations per
    window. A "dynamic weight" would be even less stable, and it amounts to IC
    timing.
  - Without HAR, own spread + constant (`(k₁ + k₀/s_day)·(p − day mean)`):
    - R² 0.00946 (+0.4%), mean |slope − 1| per window 0.178 → 0.120, per month
      0.275 → 0.262;
    - but R² is lower in 4/6 windows, and the weights are unstable (k₁ 0 … 1.5).
    - Noise-level; not adopted.
  - Added 2026-10-25 as section 6 of `post_dispersion-via-HAR.ipynb` (table,
    weights per window, plot `plots/post_dispersion_combination.png`). Rank IC is
    unchanged for every variant (no day got a negative multiplier).

---

## Chapter 6. Adding each stock's history

*2026-10-25 → 10-26*

This is the project's main question: does a stock's recent past improve the
forecast? Lags of the features, deviations from recent means, rolling means, and summaries
of past labels, first as a grid of full runs (8–11), then as a validation-only sweep and
runs 12–16.

**Takeaways**

- History must be aligned by stock and trading date. Stocks are suspended, listed and delisted almost every day (I-9).
- A label is known only six dates later, so label-based inputs use dates ≤ T − 6 (N-32).
- Lags beat deviations and long means; recent history helps most and older history hurts. Run 11 (features of T−1 and T−5) beat Run-5 in all six windows on both ranking and magnitude, and held over three seeds (N-33, N-35 … N-37).
- Which second lag to use (2, 3, 4, 5 or 6) didn't matter within seed noise (N-38).

### I-9. Universe stability and gaps (matters for history features)

**Status:** Open (recorded 2026-10-21). Address when building history
features (Stage 5).

- **Slow drift:** ~2,560 stocks/day (2022-01) → ~2,880 (2026-04), about
  +75 per year; growth flattened from late 2023.
- **Small daily churn:** |Δ count| median 1, 88% of days ≤ 2, max 36
  (excluding outliers). But the stock *set* is identical to the previous
  day on only 18% of days: entries (median 1/day) and exits (median 1/day)
  happen almost daily. Runs of unchanged counts are short (mean 1.45 days,
  longest 9).
- **Yearly dips around late April / early May** (2022–2025, and again at
  the end of the data): the count drops by ~20–40 for a few days, and
  missing labels spike (up to ~50/day). Plausibly annual-report-season
  suspensions or delisting-risk designations (deadline 30 April); unverified.
- **Two near-empty dates:** 2022-06-20 (709 rows) and 2022-06-28 (126).
  Cause unknown.
- **Missing labels:** 4,652 in total, on 978 of 1,044 dates; median 3 per
  affected date.
- Plot: `runs/run-2_original-baseline-six-windows/plots/analysis_stock_counts.png`.
- **Implications:** "the previous row of this stock" ≠ "the previous trading
  date". Build lags by (Code, Date) against the market calendar, and decide
  how to treat missing days: NaN (LightGBM handles it), forward-fill, or
  require full history. Count how many rows lose history (new listings,
  suspensions, the two June-2022 dates), and keep evaluation rows identical
  between variants (the paired comparison handles that).

### N-32. Multi-day family: design, evaluation protocol, label lag

**Recorded:** 2026-10-25. Discussed with the user. The single-day family is closed
(T-12 (a)): Run-5 (demeaned) is the best single-day setup found. One optional knob,
the Huber loss (T-8), is still untested.
- **Label lag (user, 2026-10-25):** on date t, labels of dates <= **t − 6** are
  realized, not only <= t − 7. The regime measure (T−26 … T−7) and the HAR
  notebook used 7, which is one date conservative, not wrong.
  - T − 5 may also be realized (its label uses information up to today's close),
    but only if the prediction may be made after the close; unknown, so we keep
    t − 6 to be safe. Switching is `LABEL_LAG = 5` plus one run.
- **Purged k-fold with history features: purge unchanged (10), no training days
  lost.**
  - The purge guards against overlapping *targets* (labels look ~5 dates ahead);
    history features don't change the targets.
  - Overlapping *inputs* are harmless. Validation rows looking back into training
    dates use the realized past, as test rows do. Training rows after a validation
    block take its labels as inputs paired with their own later targets; no
    validation row's inputs contain its own label.
  - The look-back comes from `WARMUP` (66) dates before each window, which supply
    history only; history dates, folds and rows are the same as run 5's. W1 starts
    at the first date of the data, so its early rows have missing history columns.
  - If something leaked anyway, validation would beat test for these sets: the
    agreement check would show it.
- **Evaluation protocol for feature families:**
  - Full runs instead of the validation-only sweeper. A full run costs only ~25%
    more than one sweep configuration (run 5, W1: 102 s vs 80 s), and it also
    saves the validation predictions.
  - Decide on the test windows W1–W6: genuinely out of period, paired against
    Run-5 (demeaned), a second seed only for promising sets. Declare the variants
    up front and count them (multiple testing).
  - **Agreement check:** for each family's first runs, compare validation Δ with
    test Δ per window (as in N-30). If the signs agree in ≥ 5 of 6 windows, the
    sweeper may tune within that family (horizons etc.) without spending test
    windows. Otherwise every choice needs a full run.
  - The reserved final window (I-6) stays untouched until the end.
  - Dropped: a W1–W3 development / W4–W6 confirmation split. It halves the power
    (95% band on ΔRank IC ±0.0044 → about ±0.006) at the gain size we hope for.
- **Design choices:**
  - Stacking raw past days: only as a control (`raw+lags`). Trees split on one
    column at a time, so they handle a difference of two columns poorly; lags are
    noisy near-copies; the column count explodes.
  - Trailing summaries instead: each feature's deviation from its own trailing
    week / month mean (`dev`), the same day / week / month idea as HAR, at stock
    level.
  - The stock's own realized labels (`labelhist`): latest, 20- and 60-date means
    (reversal, momentum), 20-date SD and max (volatility, lottery).
  - Market-level history last, if at all (regime identification, N-29, N-31).
  - One model first. A correction model (LightGBM `init_score` = run 5's
    out-of-fold predictions on the history, its test predictions on test) only if
    the big model weakens today's signal, as run 7 did.
- **Memory:** the machine has 11 GB, about 3 GB free with VS Code open. Run 7
  (144 columns) peaked at 3.3 GB. Keep sets to about 150 columns.
- **First grid (T-14):** `raw+labelhist-max` (run 8.1), `raw+labelhist-min`
  (run 8.2), `raw+dev`, `raw+labelhist-…+dev`, `raw+lags`, seed 70563, via
  `train_multi-day-feature-sets.py`.
- **Max vs min (user, 2026-10-25):** two label-history versions that differ only in
  the trailing 20-date extreme (signed max = lottery / MAX effect, Bali, Cakici &
  Whitelaw 2011; or signed min). Data-only prior (no model), daily Rank IC
  against the label, dates through 2025-12-18 (final window excluded), 924 days:

| Column | Rank IC (NW t) | 2022 / 2023 / 2024 / 2025 |
|---|---|---|
| max20 | −0.079 (−15.9) | −0.080 / −0.069 / −0.079 / −0.086 |
| min20 | +0.023 (4.4) | +0.007 / +0.025 / +0.020 / +0.036 |
| std20 | −0.071 (−17.4) | |
| mean20 / mean60 / last | −0.057 / −0.039 / −0.051 | |
| max20 after std, means, last | −0.004 (−2.0) | |
| min20 after std, means, last | −0.006 (−3.0) | |

  - The lottery pattern is strong and stable on its own, but mostly it is "volatile
    stocks underperform": corr(max20, std20) = 0.85. Beyond SD and means, both
    extremes add little.
  - Prediction: runs 8.1 and 8.2 differ little, because both contain std20. The
    real question is what label history adds beyond the 48 raw features.

### N-33. Multi-day grid (T-14): runs 8.1, 8.2, 9, 10.1, 10.2, 11 vs run 5 and run 2

**Recorded:** 2026-10-25. Notebooks: `runs/run-{8.1,8.2,9,10.1,10.2,11}_*/analysis.ipynb`
(baseline run 5, leaderboard runs 2, 5 and these six; text cells written per run).
Only the input columns changed (N-32); same seed, protocol, rows and windows.
Paired on all 2,028,817 test rows, NW with 10 lags (the notebooks' cell 10 uses 4
lags, so their CIs are ~10% narrower). R² dm = after per-day demeaning.

| Run | Set | Rank IC | ΔRank IC vs 5 (t) | Windows | Pearson IC | R² dm | ΔRank IC vs 2 (t) |
|---|---|---|---|---|---|---|---|
| 5 | raw | 0.0631 | — | — | 0.0985 | 0.00942 | +0.0051 (7.2) |
| 8.1 | raw+labelhist-max | 0.0698 | +0.0067 ± 0.0072 (1.8) | 5/6 | 0.0962 | 0.00858 | +0.0117 (3.3) |
| 8.2 | raw+labelhist-min | 0.0673 | +0.0042 ± 0.0075 (1.1) | 3/6 | 0.0962 | 0.00881 | +0.0093 (2.5) |
| 9 | raw+dev | 0.0651 | +0.0020 ± 0.0026 (1.5) | 3/6 | 0.0962 | 0.00911 | +0.0070 (4.8) |
| 10.1 | raw+labelhist-max+dev | **0.0717** | +0.0085 ± 0.0073 (2.3) | 5/6 | 0.0957 | 0.00872 | +0.0136 (3.8) |
| 10.2 | raw+labelhist-min+dev | 0.0691 | +0.0060 ± 0.0076 (1.5) | 4/6 | 0.0957 | 0.00857 | +0.0110 (3.0) |
| 11 | raw+lags | 0.0674 | **+0.0043 ± 0.0016 (5.3)** | **6/6** | **0.0998** | **0.00997** | +0.0093 (8.1) |

- **Verdicts vs run 5:**
  - Run 11 is the only **better** run, on both ranking and magnitude: R² dm higher
    in 6/6 windows, Q1 and Q5 both sharper, all 17 months from 2024-08 positive.
  - 10.1 has the best ranking but worse magnitudes (R² dm −7%, Pearson −0.0028), so
    it is not better under the criteria.
  - 8.1, 8.2, 9 and 10.2 show no detectable ranking difference, and their
    magnitudes are worse or mixed.
  - Every set beats run 2 on ranking.
- **Max vs min (the user's lottery test):**
  - 8.1 − 8.2 = +0.0025 ± 0.0016 (t 3.0, 5/6). It replicates in 10.1 − 10.2
    (+0.0026, t 3.1); the per-window patterns correlate 0.96, and W5/W6 carry it.
  - Where it comes from: 8.1 ranks the whole top-max20 quintile lower (its mean label
    is −0.0031). Within quintiles of max20, min20 or std20, 8.2 ranks as well or
    better.
  - The labels show a MAX effect beyond SD: in the top std20 quintile, high-max20
    stocks score −0.0045 lower (t −8.7). But 8.1 doesn't exploit that better.
  - So the evidence supports a high-max discount, and the specific lottery effect
    only weakly. One seed each.
- **Lags beat deviations** (9 − 11: −0.0023 ± 0.0020, t −2.3), refuting the design
  argument in N-32.
  - What predicts is a persistent level. Yesterday's value is a second reading of it
    (partly independent, e.g. corr(x_t, lag1) 0.42 for feature28).
  - A deviation subtracts that level out (daily Rank IC: feature28 x_t 0.057, lag1
    0.048, dev5 0.015).
  - Gain importance in run 11: raw 58%, lag1 24%, lag5 18%. Every top lag column is a
    lag1 (feature28_lag1 ranks 4–9; feature47_lag1 rises to 5–6% in W4/W5).
  - Run 9 is underconfident (35% fewer trees, slopes 1.40/1.34 in W1/W2).
- **Dev on top of label history:** 10.1 − 8.1 = +0.0019 (t 2.1), 10.2 − 8.2 = +0.0017
  (t 2.6), paying only from Sep 2024 on, like run 9 (which loses in W1–W3 and wins in
  W4–W6). The effects add up roughly: +0.0067 + 0.0019 ≈ +0.0085.
- **Label history: gains are concentrated, and magnitudes break after shocks:**
  - W1 and W5 give ~80% of the gain. 10.1 shows no gain from Aug 2023 to Aug 2024
    (−0.0048, t −0.8, 265 days) and +0.0217 (t 4.3) from Sep 2024 to Jun 2025.
  - W5's gain is spread over 6 of 7 months.
  - The gain sits mid-cross-section: Q1 is less sharp (−0.0061 → −0.0057) and
    Pearson/Rank falls from 1.56 to 1.38. The R² loss comes from the lower Pearson IC,
    not from scale (pooled slope 0.98), so rescaling won't fix it.
  - **W4 (run 8.1):** October 2024 gives 72% of the W4 R² loss, and 8–11 Oct alone
    give 37%.
    - The Sep-2024 spike's labels entered most stocks' history columns at once:
      median labelhist_std20 0.026 → 0.040; rows above the training 95th
      percentile 4.8% → 11.2%.
    - The model predicted a reversal of the last label (its extra prediction
      correlates −0.61 with labelhist_last), and labels didn't reverse.
    - October prediction spread was +27%, slope 0.09 vs 0.32.
    - Correction to N-32's check: the between-day share of variance is small
      on average (0.2–3%), but **after shocks the columns do shift market-wide**.
- **Validation vs test (the N-32 agreement check):**

| Run | corr(val Δ, test Δ) Rank IC | Signs agree | Mean val Δ | Mean test Δ |
|---|---|---|---|---|
| 8.1 / 8.2 | −0.72 / −0.42 | 5 / 3 | +0.0075 / +0.0064 | +0.0067 / +0.0042 |
| 9 | **+0.86** | 5 | −0.0004 | +0.0020 |
| 10.1 / 10.2 | −0.66 / −0.46 | 4 / 4 | +0.0060 / +0.0061 | +0.0085 / +0.0060 |
| 11 | **+0.77** | 5 | +0.0011 | +0.0043 |

  - **Feature dynamics (lags, dev): validation can be trusted** per window. The
    sweeper may tune lag sets and horizons. It understates the test gain.
  - **Label history: only the average direction is right.** Per window it is
    anti-correlated, like run 7 (−0.93). Use full runs.
- **Cost:** 8.8–12.5 min (run 5 11.1); peak memory 2.3 GB (labelhist) to 4.0 GB
  (149 columns). Trees: run 11 more in W2/W3 (673/504), the others 24–38% fewer.
- **Selection:** 12 configurations compared on these windows. 10.1's t 2.3 is the
  best of 6, below a Bonferroni bar of ~2.6; run 11's t 5.3 clears it.
- **Next (T-15):**
  - (a) a second seed for run 11 (11.1);
  - (b) sweep lag sets on validation (trusted for this family), e.g. {1}, {1,2},
    {1,2,3}, {1,2,3,5}, {1,5,10};
  - (c) label history ranked per day (removes the post-shock level jump), alone and
    with the best lags, as full runs;
  - (d) lags + labelhist-max combined;
  - (e) maybe a blend of run 11 with 8.1 (agreement 0.84).

### N-34. Four categories of history information: hypotheses, and what to try next

**Recorded:** 2026-10-26. The user's taxonomy: raw historical features (lags),
derived historical features (deviations, ...), raw historical labels (realized
values), derived historical labels (statistics).
- **Mapping:** run 11 = raw features; run 9 = derived features; runs 8.x = raw label
  (`labelhist_last`) **plus** derived labels (means, SD, max/min), so the two label
  categories are confounded; 10.x = derived features + both label categories.
- **Gain importance** (training data, descriptive; share of total gain, mean over
  windows):
  - 8.1: raw 48 features 65%; `labelhist_last` 13.0% alone, ranked 1–2 in 5/6 windows
    (W4/W5: 18% / 21%); max20 6.0% (rank 2–3 in W4/W5); mean20 6.7%; mean60 4.6%;
    std20 4.6%.
  - 11: raw 58%, lag1 24%, lag5 18%. 9: raw 63%, dev5 18%, dev20 19%.
- **Hypotheses:**
  - H1, raw historical features: a second, noisy reading of each stock's persistent
    feature level (denoising) plus the 1-day change. Stock-level and on the same
    scale in every regime, so steady ranking gains (6/6), better magnitudes (Pearson
    up, both tails sharper), validation trustworthy.
  - H2, derived historical features (deviations): they remove the level, which is
    the predictive part, and keep only the change. Weak, period-dependent (lose
    W1–W3, win W4–W6), underconfident.
    - Prediction: **level-keeping** derivations (trailing means, EWMA) behave like
      lags, or better.
  - H3, raw historical label (latest realized label): a high-leverage short-term
    reversal signal whose sign depends on the regime.
    - Big ranking gains when reversal works (W5); damage after shocks, when it
      doesn't (Oct 2024: predicted reversal, labels didn't reverse).
    - Its scale jumps with market dispersion.
    - Suspected main source of both the label family's gains and its magnitude
      breaks.
  - H4, derived historical labels: persistent stock characteristics (volatility,
    lottery / max, medium-term drift). Moderate, steadier ranking information.
    Their scale also shifts after shocks (median std20 0.026 → 0.040).
  - **Unifying:** what matters is less raw vs derived than two properties:
    - (a) does the column keep the stock's persistent level?
    - (b) is its scale stable across regimes?
    Feature-based columns pass (b), label-based ones fail it. That explains the
    label family's magnitude damage and its untrustworthy validation.
- **Shots besides more lags** (to decide with the user):
  - (A) label history normalized per day: ranks, or labels divided by that day's
    dispersion before computing statistics;
  - (B) split raw vs derived labels, two runs;
  - (C) level-keeping derived features (trailing means / EWMA), screened in the
    sweeper;
  - (D) best lags + best label history;
  - long shots: a correction model for label history (init_score = run 11's
    out-of-fold predictions), a stock's own recent model errors, trailing SD of
    each feature.

### N-35. Run 11 seed check (T-15 a); feature-history sweep and label-split sets set up

**Recorded:** 2026-10-26.
- **Seeds 11.1 (94086) and 11.2 (258128),** vs Run-2 (`multi-day_performance.ipynb`;
  Run-5 and Run-11 demeaned):

| Metric | Run-5, 3 seeds | Run-11, 3 seeds | Seed noise ± |
|---|---|---|---|
| Rank IC | 0.0631 ~ 0.0634 | **0.0665 ~ 0.0674** | 0.0009 |
| Pearson IC | 0.0979 ~ 0.0985 | 0.0997 ~ 0.1002 | 0.0008 |
| R² (demeaned) | 0.00934 ~ 0.00942 | **0.00997 ~ 0.01005** | 0.00011 |
| Mean \|slope − 1\| | 0.178 ~ 0.184 | 0.174 ~ 0.184 | 0.012 |

  - Every run-11 seed beats every run-5 seed on Rank IC (smallest gap +0.0031) and
    R². Run 11's gain is not seed luck. t vs Run-2: 8.6 ~ 9.2, 6/6 windows.
  - Seed noise for Rank IC rises from ±0.0004 (run 5 only) to ±0.0009 when pooled with
    run 11's seeds. Lags seem to make the model a little more seed-sensitive.
- **User note:** no hard column cap; ~300 columns are feasible. The question is
  whether columns help, not whether they fit.
- **Set up (T-16):**
  - `train_multi-day-feature-sets.py`: new groups `lags-K1-K2-...`, `fmean-A-B`,
    `labelraw-A-B`, `labelstats-max/min`, and the label-group suffixes `-rank` /
    `-scaled`.
  - `test_multi-day-feature-sets.py`: the family's sweeper, for feature history only
    (validation trusted, N-33).
  - First sweep (`pre-runs/mday-1_...`), with lag sets chosen to separate two
    explanations:
    - **dense recent readings** (denoising, H1): {1}, {1,2}, {1,2,3}, {1..5};
    - **longer memory**: {1,5,10}, {1,5,20}, {1,2,5,10,20} (288 columns);
    - **level-keeping means** instead of many lags: fmean 1–5 + 6–20, and lag1 +
      fmean 2–5 + 6–20;
    - plus raw and raw+lags = runs 5 / 11 on validation (reproduction checks).
  - Label shots for full runs afterwards: `raw+labelraw-6-10`, `raw+labelstats-max`,
    `raw+labelhist-max-rank`, `raw+labelhist-max-scaled`.

### N-36. Sweep mday-1 (lag sets, feature means): recency helps, memory and smoothing hurt

**Recorded:** 2026-10-26. `pre-runs/mday-1_sweep-feature-history_goss-5-fold-cv`, seed
70563, validation only (`raw` and `raw+lags` reused from runs 5 / 11). Fold-mean
validation, mean over 6 windows:

| Config | Columns | Trees | Rank IC | Δ vs {1,5} (windows >) | R² dm | Δ vs {1,5} (windows >) | Train/val R² |
|---|---|---|---|---|---|---|---|
| raw (run 5) | 48 | 419 | 0.0590 | −0.0011 | 0.00800 | −0.00053 | 2.9 |
| lags {1,5} (run 11) | 144 | 393 | 0.0601 | — | 0.00853 | — | 3.3 |
| **lags {1}** | **96** | 440 | 0.0609 | +0.0008 (4/6) | **0.00861** | +0.00009 (3/6) | 3.2 |
| lags {1,2} | 144 | 437 | **0.0613** | +0.0012 (4/6) | 0.00846 | −0.00006 (2/6) | 3.5 |
| lags {1,2,3} | 192 | 408 | 0.0608 | +0.0007 (5/6) | 0.00833 | −0.00020 (3/6) | 3.6 |
| lags {1..5} | 288 | 390 | 0.0604 | +0.0003 (2/6) | 0.00822 | −0.00031 (0/6) | — |
| lags {1,5,10} | 192 | 360 | 0.0602 | +0.0001 (3/6) | 0.00837 | −0.00016 (2/6) | 3.5 |
| lags {1,5,20} | 192 | 338 | 0.0598 | −0.0002 (3/6) | 0.00772 | −0.00081 (0/6) | 3.8 |
| lags {1,2,5,10,20} | 288 | 325 | 0.0597 | −0.0004 (1/6) | 0.00763 | −0.00089 (0/6) | 4.0 |
| fmean 1–5 + 6–20 | 144 | 195 | 0.0576 | −0.0024 (1/6) | 0.00627 | −0.00226 (0/6) | 4.0 |
| lag1 + fmean 2–5 + 6–20 | 192 | 226 | 0.0591 | −0.0010 (2/6) | 0.00654 | −0.00198 (0/6) | 4.3 |

- **Recency is what helps.** Lag 1 alone gives the best R² with the fewest columns,
  and {1,2} the best Rank IC. Every further or older lag lowers R², and the long-reach
  sets (10, 20 dates) and more columns cost trees and R².
- **Feature means are clearly worse**, worse than no history at all on R², with half the
  trees.
  - This refutes N-34 H2's prediction (level-keeping derivations ≈ lags) and the
    "denoising by averaging" reading of H1.
  - Revised H1: lag 1 helps through **fresh information from the latest day** (the
    1-day change, yesterday's level), not by averaging out noise.
- **Overfitting grows with slow-moving stock-level columns.** Train/val R² goes 2.9
  (raw) → 3.2–3.3 (lag 1) → 4.0+ (20-date reach, means).
  - The same holds in the full runs: run 9 (dev) 3.6, 8.2 3.9, 8.1 4.1, 10.1 4.5.
  - Hypothesis H5: columns that barely change over weeks for a stock act as a stock
    fingerprint. Trees memorize stock-specific label levels from the training
    period; these don't carry into another period. Fast-changing columns (lag 1)
    can't be used that way.
  - This complements N-34's scale argument for label history, and predicts that
    per-day ranking of label statistics (still slow-moving) fixes the scale shifts
    but not the overfitting.
- **Resolution:** validation tracks test for this family (N-33) but understated run
  11's test gain (+0.0011 vs +0.0043). The sweep clearly ranks recent lags > far lags
  > means, but {1}, {1,2}, {1,5} and {1,2,3} are within ~0.0012 Rank IC and ~0.0003
  R² of each other. That is likely within seed noise, which has no validation
  estimate yet.
- **Next:** a full run of `raw+lags-1` (parsimonious; paired vs run 11 on test), then
  the label shots on `raw`, then the combination on lags 1. For label history, try
  stronger regularization or the correction model (H5).

### N-37. Run 12 (raw + lag 1) vs run 11 (lags {1, 5}), run 5, run 2

**Recorded:** 2026-10-26. Notebook `runs/run-12_multi-day-rawfeat+lag1feat-p5fcv/analysis.ipynb`
(paired baseline run 11; leaderboard runs 2, 5, 11, 12). Only change vs run 11: lag-5
columns removed (144 → 96). Same seed (70563), rows, protocol. The setup diff also shows
the family script's refactor (code-only; old sets verified identical).

| W | Rank IC 11 → 12 | Δ vs 11 / 11.1 / 11.2 | R² dm 11 → 12 | Slope 11 → 12 | Trees 11 → 12 |
|---|---|---|---|---|---|
| 1 | 0.0514 → 0.0510 | −0.0004 / +0.0007 / −0.0004 | 0.0097 → 0.0086 | 1.38 → 1.33 | 373 → 310 |
| 2 | 0.0668 → 0.0671 | +0.0004 / +0.0003 / 0.0000 | 0.0120 → 0.0111 | 0.98 → 1.00 | 673 → 801 |
| 3 | 0.0947 → 0.0943 | −0.0004 / +0.0005 / −0.0004 | 0.0188 → 0.0189 | 1.00 → 1.01 | 504 → 470 |
| 4 | 0.0610 → 0.0617 | +0.0007 / +0.0011 / +0.0021 | 0.0055 → 0.0053 | 0.74 → 0.70 | 293 → 458 |
| 5 | 0.0673 → 0.0637 | **−0.0036** / −0.0019 / −0.0034 | 0.0045 → 0.0044 | 0.80 → 0.76 | 218 → 279 |
| 6 | 0.0632 → 0.0604 | **−0.0028** / −0.0016 / −0.0010 | 0.0074 → 0.0078 | 1.30 → 1.29 | 295 → 324 |
| All | 0.0674 → 0.0664 | −0.0010 ± 0.0012 (t −1.7) / −0.0001 / −0.0005 | 0.00997 → 0.00981 (seeds of 11: 0.00997–0.01005) | | |

(R² per window here is raw, from the notebook; the pooled R² is demeaned.)
- **Verdict vs run 11: no detectable ranking difference** (within seed noise; run 11's
  primary seed is its best), **magnitudes slightly worse** (R² dm −0.00016 … −0.00024,
  beyond the seed noise ±0.00011).
  - The losses are only in 2025 (W5/W6), with the same sign against all three run-11
    seeds. The cumulative Δ turns down around Oct 2024. Calm regime −0.0019 (t −2.2).
- **Vs run 5: better** on ranking and magnitude: +0.0033 ± 0.0010 (t 6.7), 6/6,
  ΔPearson +0.0017, R² dm 0.00981 vs 0.00942. Lag 1 carries ~90% of the lag gain.
- **Vs run 2:** +0.0083 ± 0.0017 (t 9.4), 6/6.
- **Validation bias (protocol):** the sweep's lag 1 − {1, 5} was +0.0015 / +0.0026 /
  +0.0007 / +0.0024 / −0.0010 / −0.0012, against test −0.0004 / +0.0004 / −0.0004 /
  +0.0007 / −0.0036 / −0.0028.
  - The per-window pattern matches (corr 0.95), but validation is **+0.0018 too
    favourable to lag 1 in every window**.
  - Hypothesis: weaker extra columns need more rows; the folds train on 80% of the
    history.
  - Consequence: the sweep ranks clearly different sets; full runs decide close calls.
- **Cost:** 2.8 vs 3.8 GB; runtime ≈ equal; train/val R² 3.2 (same as in the sweep).
- **Decision:** lags {1, 5} (run 11) stays the multi-day reference and the base for
  combinations; run 12 is the lighter fallback.
- **Hypotheses / tests:**
  - lag 5 matches the label horizon, and its information mattered from late 2024 on.
    Tests: `raw+lags-5` alone (gain in W5/W6?); `raw+lags-1-4` / `-1-6` vs {1, 5}.
  - Validation bias: `VALIDATION_FOLD` 10.
  - A second seed of run 12.
- Configurations compared on these test windows: 13.

### N-38. Lag series {1, x} (runs 13–16) vs run 11 {1, 5}, run 12 {1}, run 5

**Recorded:** 2026-10-26. Notebooks `runs/run-{13,14,15,16}_…/analysis.ipynb` (paired
baseline run 11; leaderboard runs 2, 5, 11–16). Each run is run 11 with lag 5 replaced by
lag 2 / 3 / 4 / 6 (144 columns); same seed (70563), rows, protocol. Question (N-37): is lag 5
special because it matches the ~5-day label horizon?
Run 11's folders were renamed to the series' pattern (`run-11*_multi-day-rawfeat+lag1feat+lag5feat-p5fcv…`).

| Set | Run | Rank IC | R² dm | Δ vs 11 (t) | Δ vs 11 seed mean | Δ vs 12 (t) | Late Δ vs 12 (W5–W6) | \|slope − 1\| |
|---|---|---|---|---|---|---|---|---|
| {1} | 12 | 0.0664 | 0.00981 | −0.0010 (−1.7) | −0.0006 | — | — | 0.190 |
| {1,2} | 13 | 0.0671 | 0.01000 | −0.0004 (−0.6) | +0.0001 | +0.0007 (1.8) | +0.0014 (1.9) | 0.191 |
| {1,3} | 14 | 0.0671 | 0.00997 | −0.0003 (−0.7) | +0.0001 | +0.0007 (1.5) | +0.0014 (1.9) | 0.206 |
| {1,4} | 15 | 0.0670 | 0.01003 | −0.0004 (−1.0) | +0.0001 | +0.0006 (1.3) | +0.0015 (2.1) | 0.201 |
| {1,5} | 11 / 11.1 / 11.2 | 0.0674 / 0.0665 / 0.0669 | 0.00997 / 0.01005 / 0.01002 | — | — | +0.0010 / +0.0001 / +0.0005 | +0.0032 / +0.0017 / +0.0022 | 0.174–0.184 |
| {1,6} | 16 | 0.0670 | 0.00996 | −0.0004 (−1.3) | +0.0000 | +0.0006 (1.0) | +0.0020 (2.1) | 0.194 |

(NW 10 lags; seed noise ±0.0009 Rank IC, ±0.00011 R² dm.)
- **Verdict: every {1, x} ties run 11** on ranking and magnitude. **Lag 5 is not special**:
  the label-horizon hypothesis of N-37 is refuted. Run 11's primary seed is its luckiest.
- **Any second lag beats lag 1 alone slightly**: +0.0006 … +0.0010, R² dm +0.0002, all of
  it in 2025 (W1–W4 ≈ 0). Each run alone t 1.0–1.8, but all five agree. Run 12 is a single
  seed, so it might just be unlucky (T-17 c).
- **Mechanism (hypothesis):** lag 1 + lag k let the trees see a recent change; changes over
  2–6 days carry much the same information (persistent features). Gain shares are identical
  (lag 1 ~24%, second lag ~18%, raw ~58%; run 12: lag 1 34%).
- **Agreement varies smoothly with k** (demeaned correlation with run 12: 0.974 / 0.969 /
  0.962 / 0.957 for k = 2 / 3 / 4 / 6; with run 11: 0.952 / 0.965 / 0.974 / 0.974): different
  models, equal scores.
- **Blend (ad hoc, equal weights, no fitting):** the five {1, x} runs give Rank IC 0.0679,
  R² dm 0.01028. vs run 11's 3-seed blend (0.0671, 0.01007): +0.0008 ± 0.0007 (t 2.3); vs
  run 12 +0.0016 (t 2.8). Seed averaging alone adds almost nothing; lag diversity does.
  Calibration slightly more underconfident (|slope − 1| 0.218 vs 0.180). Caveat: tested
  after seeing the runs; 5× the cost.
- **Validation** can't separate the series either (fold-mean Rank IC 0.0601–0.0613; {1,2}
  highest), consistent with N-37's bias.
- **Overfitting / cost:** train/val R² 3.2–3.4 (runs 11/12: 3.3 / 3.2); 12.1–13.5 min,
  3.5–3.6 GB.
- **Decision:** keep run 11 ({1, 5}, 3 seeds) as the multi-day reference and the base for
  combinations. The {1, x} lag-mix ensemble is a **candidate for the final evaluation**
  (user, 2026-10-26); weights may be fitted on out-of-fold validation predictions (T-17).
- Configurations compared on these test windows: 17.

---

## Chapter 7. Past labels, and one model for both

*2026-10-26 → 11-02*

The other kind of history is the stock's own past outcomes. Runs 17–20 tried several
label summaries; runs 21–23 combined the best label inputs with the best feature lags in
one model.

**Takeaways**

- The raw labels of T−6 … T−10 worked best. Dividing them by the day's spread (or ranking them) fixed the calibration after market shocks (N-39).
- The two kinds of history help in different places: lags react to what changed since yesterday, and past labels to last week's outcome (N-40).
- **Run 22** (features of T, T−1, T−5 plus scaled labels of T−6 … T−10, 149 inputs) beat the average of two separate models on ranking at half the cost, and held over three seeds. It became the primary candidate (N-41, N-42).

### N-39. Label series (runs 17–20) vs run 8.1 (labelhist-max), run 5, run 11

**Recorded:** 2026-10-27. Notebooks `runs/run-{17,18,19,20}_…/analysis.ipynb` (paired
baseline run 8.1; leaderboard runs 2, 5, 8.1, 11, 17–20). T-16 step 2: four label-history
variants on raw, one change each vs 8.1, seed 70563.

| Set | Run | Rank IC | Pearson | R² dm | \|slope − 1\| | W4 R² dm / slope | Train/val | Δ vs 8.1 (t) | Δ vs run 5 seed mean (t) | Δ vs 11 (t) |
|---|---|---|---|---|---|---|---|---|---|---|
| labelhist-max | 8.1 | 0.0698 | 0.0962 | 0.00858 | 0.157 | 0.0026 / 0.58 | 4.1 | — | +0.0065 (1.8) | +0.0024 (0.7) |
| **labelraw-6-10** | 17 | 0.0700 | **0.1008** | **0.00978** | 0.156 | 0.0049 / 0.65 | **3.2** | +0.0002 (0.1) | +0.0067 (2.3) | +0.0026 (0.9) |
| labelstats-max | 18 | 0.0670 | 0.0944 | 0.00846 | 0.170 | 0.0046 / 0.67 | 3.6 | −0.0028 (−1.6) | +0.0037 (1.1) | −0.0004 |
| labelhist-max-rank | 19 | 0.0696 | 0.0957 | 0.00908 | **0.085** | **0.0064 / 0.92** | 3.3 | −0.0002 (−0.2) | +0.0063 (1.8) | +0.0022 |
| labelhist-max-scaled | 20 | 0.0699 | 0.0963 | 0.00922 | 0.111 | 0.0055 / 0.80 | 3.4 | +0.0001 (0.1) | +0.0066 (1.8) | +0.0024 |
| raw (ref.) | 5 | 0.0631 | 0.0985 | 0.00942 | 0.178 | 0.0065 / 0.75 | 2.9 | | | |
| lags {1, 5} (ref.) | 11 | 0.0674 | 0.0997 | 0.00997 | 0.184 | 0.0069 / 0.80 | 3.3 | | | |

(NW 10 lags; R² demeaned.)
- **Verdicts vs 8.1:** 17 **better, magnitude kind** (same Rank IC, R² +0.0012, Pearson
  +0.0047; agreement only 0.81: per-window pattern differs, W2 +0.0162, W1/W3/W5 −0.006 …
  −0.008). 19 and 20 **better, magnitude kind** (same ranking, agreement 0.93/0.94,
  calibration and W4 fixed). 18 **not better, most likely worse** (1/6 windows, calm −0.0043,
  t −3.2).
- **What we learned (taxonomy, N-34):**
  - **The latest realized label is the core** of label history (top label column in 8.1 and
    17; dropping it, run 18, loses).
  - **Raw label history beats derived** on magnitudes: without the slow statistics, overfitting
    drops to the lag runs' level (4.1 → 3.2) and R² recovers (H5 confirmed). In run 17, lag 6
    (11.8% of gain) and lag 10 (5.4%) dominate, lags 7–9 2–3% each. Hypothesis: 6 and 10 are
    adjacent non-overlapping ~5-date labels.
  - **Scale hypothesis confirmed:** per-day ranking or scaling removes the post-shock
    miscalibration (W4 slope 0.58 → 0.92 / 0.80). Run 19's calibration (0.085) is the best of
    any run.
  - Every label set's ranking gain over run 5 sits mostly in W1 and W5 (+0.008 … +0.024);
    CIs ±0.006–0.007. Large, uneven gains.
- **Validation:** again right on average (17: val +0.0072 vs test +0.0069 over run 5), anti-
  correlated per window (−0.66 … −0.75). Validation R² did flag 17 as the only label set
  better than run 5 on R² (+0.00086).
- **Blends with run 11 (ad hoc, equal weights, demeaned):** 11 + 17: Rank IC **0.0716**, R² dm
  **0.01065** (best R² of anything so far), +0.0041 ± 0.0033 vs 11 (t 2.5), 5/6 windows.
  11 + 20: 0.0715 / 0.01036 (t 2.1); 11 + 19: 0.0712 / 0.01034; 11 + 8.1: 0.0716 / 0.01015; 11 + 18:
  0.0694 / 0.00996. Label runs agree only 0.83–0.86 with run 11: a different signal.
- **Next (T-16 step 3):** one model with lags {1, 5} + labelraw-6-10 (149 columns); optionally
  the same with `labelraw-6-10-scaled` (or `-rank`) to add the calibration fix.
- Configurations compared on these test windows: 21.

### N-40. What lags and latest labels each contribute (run 11 − run 5, run 17 − run 5)

**Recorded:** 2026-10-27 (ad hoc, `scratchpad/components.py`; not a notebook). Test rows split
each day into quintiles by ex-ante variables; within-quintile Rank IC and R² gains vs run 5
(seed 5.1 as the noise yardstick).
- **Lags: what changed since yesterday.** Gain grows with how much the stock's features
  moved since T−1 (by quintile +0.0008 / +0.0027 / +0.0034 / +0.0063 / +0.0088, t 6.3);
  |Δprediction| tracks the move (corr +0.053, t 12.8; labels' −0.003). Broad: positive in
  every group, R² gain spread across outcome deciles (43% from the worst decile), all regimes
  (calm +0.0037, normal +0.0059, turbulent +0.0031). Strongest on the most volatile stocks
  (+0.0060, 41% of its R² gain) and in W3/W4 (incremental IC +0.027 / +0.041). Steadier (t 11
  vs 6 for labels).
- **Labels: last week's outcome.** The latest realized label predicts the next one with the
  opposite sign (daily Rank IC −0.052; run 5 already −0.10 via features); the label part
  pushes against it (corr −0.52).
  - Helps after extreme labels (|label| top quintile +0.0139, t 3.5), mostly after large
    **positive** ones (+0.0131, t 5.0); **hurts after large negative ones** (−0.0126, t −4.1;
    R² −0.00136). Lags help both ends.
  - Calm +0.0100, normal +0.0093, turbulent −0.0006; mid-to-high label-SD stocks (+0.0088,
    +0.0152), not the most volatile.
  - R² gain entirely from the worst-outcome decile (sharper loser detection), losses on the
    best decile; enlarges predictions (R² up only where run 5 is confident).
- **Complementary:** components correlate +0.39. The 11 + 17 mix cuts the labels' damage
  after negative labels to −0.0026 and spreads the R² gain over almost every group.
- **Open:** both components gain most R² on stocks whose features barely moved (+0.0019 /
  +0.0027); unexplained (stale rows, suspensions, limit locks?). The follow-up was killed by
  memory during training.

### N-41. Step 3: lags {1, 5} + raw labels (runs 21–23) vs run 11, run 17, the 11 + 17 mix

**Recorded:** 2026-10-27. Notebooks `runs/run-{21,22,23}_…/analysis.ipynb` (paired baseline
run 11; leaderboard 2, 5, 11, 17, 21–23). Run 11 + run 17's five label columns (149), plain /
divided by the day's SD / ranked per day. Seed 70563.

| Model | Rank IC | Pearson | R² dm | \|slope − 1\| | W4 R² dm / slope | Train/val | Δ vs 11 (t) | Δ vs mix (t) | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| run 11 | 0.0674 | 0.0997 | 0.00997 | 0.184 | 0.00693 / 0.80 | 3.3 | — | −0.0041 (−2.5) | 12.5 |
| run 17 | 0.0700 | 0.1008 | 0.00978 | 0.156 | 0.00493 / 0.65 | 3.2 | +0.0026 (0.9) | −0.0015 (−1.1) | 10.7 |
| mix 11 + 17 | 0.0716 | 0.1034 | 0.01065 | 0.187 | 0.00738 / 0.79 | — | +0.0041 (2.5) | — | 23.2 |
| run 21 (plain) | 0.0725 | 0.1023 | 0.01028 | 0.178 | 0.00593 / 0.68 | 3.6 | +0.0051 (2.0) | +0.0009 (1.0) | 13.3 |
| **run 22 (scaled)** | **0.0735** | 0.1026 | 0.01064 | 0.146 | 0.00720 / 0.77 | 3.4 | **+0.0061 (2.5)** | **+0.0019 (2.1)** | 13.1 |
| **run 23 (ranked)** | 0.0733 | 0.1024 | 0.01062 | **0.134** | 0.00724 / 0.80 | 3.3 | +0.0059 (2.5) | +0.0018 (1.8) | 12.6 |

Per window vs run 11 (run 22): +0.0076 / +0.0128 / −0.0057 / +0.0058 / +0.0107 / +0.0053.
- **Verdicts vs run 11:** 21 better (ranking kind; W4 magnitudes worse); **22 and 23 better on
  both ranking and magnitude** (vs run 11's seed mean +0.0065 / +0.0064, t 2.8 / 2.7; calm and
  normal regimes; 28/37 months).
- **One model beats the average:** 22/23 vs the mix +0.0019 / +0.0018 Rank IC, same R²,
  better calibration, half the cost.
- **Normalizing the labels is needed in the combination:** plain labels bring back the W4
  level problem (slope 0.68, R² below run 5); scaled / ranked fix it. 22 vs 21 +0.0010 (t 2.0).
  Scaled vs ranked: tie (agreement 0.989); averaging 21–23 adds nothing.
- **Labels' weak spot (N-40):** after the most negative latest labels, the combined models
  repair the sizes (R² vs run 5 +0.00054 / +0.00071 for 22 / 23) but not the ranking (0.041–
  0.043 vs raw 0.047, lags 0.052).
- **W3 (Dec 2023 – Jun 2024)** is the one losing window for every label model (−0.0054 …
  −0.0079 vs run 11).
- **Validation:** per-window anti-correlated again (−0.40 … −0.53), level roughly right; its R²
  ordering (22 > 23 > 21 > 17) matched test.
- Gain shares: raw 48–49%, lag 1 20%, lag 5 15–16%, labels 15–16%. Train/val 3.3–3.6.
- **Decision (user, 2026-10-27): run 22 is the new multi-day reference.** Earlier note: seeds of both would settle the tie
  and the margin over the mix. Configurations compared: 24.

### N-42. Run 22 seed check (22, 22.1, 22.2)

**Recorded:** 2026-11-02. Seeds 70563 / 94086 / 258128, otherwise identical (folded into
Run-22's column of `multi-day_performance.ipynb`). Ad hoc: `scratchpad/seeds22.py`.

| Metric | Run 5 (3 seeds) | Run 11 (3 seeds) | **Run 22 (3 seeds)** | Band, run 22 alone | Pooled band (5 + 11 + 22) |
|---|---|---|---|---|---|
| Rank IC | 0.0631 ~ 0.0634 | 0.0665 ~ 0.0674 | **0.0732 ~ 0.0735** | ±0.0005 | ±0.0008 (was ±0.0009) |
| Pearson IC | 0.0979 ~ 0.0985 | 0.0997 ~ 0.1002 | **0.1026 ~ 0.1028** | ±0.0003 | ±0.0007 |
| R² dm | 0.00934 ~ 0.00942 | 0.00997 ~ 0.01005 | **0.01064 ~ 0.01072** | ±0.00012 | ±0.00011 |
| Mean \|slope − 1\| | 0.178 ~ 0.184 | 0.174 ~ 0.184 | **0.146 ~ 0.148** | ±0.002 | ±0.010 |

- **Label models are not noisier:** run 22's seed spread is between run 5's and run 11's;
  agreement between seeds 0.993 (runs 5 / 11: 0.991–0.993). W6 is the noisiest window for
  every group (run 22: per-window SD 0.0015).
- **Run 22's gain is not seed luck:** every 22 seed beats every 11 seed by +0.0057 … +0.0070
  (t 2.4–2.9, 5/6 windows each); the R² ranges don't overlap. vs the 11 + 17 mix: +0.0016 …
  +0.0019 (t 1.7–2.1); vs run 21: +0.0007 … +0.0010; vs run 23: −0.0002 … +0.0002 (tie).
- The primary seed is the best on Rank IC (+0.0003 vs the others, t 1.3–1.5) and the worst on
  R²: within noise either way.
- **Averaging the three seeds** adds nothing to ranking (0.0734) and +0.00009 R² (0.01073):
  not worth 3× the cost. (Contrast: averaging different lag sets did help, N-38.)
- Configurations compared: 24 (seeds don't count as new configurations).

---

## Chapter 8. Ideas that didn't help

*2026-11-04 → 11-05*

Three last ideas on top of run 22: column sampling, weighting recent training dates
more, and a second "correction" model trained on the first one's errors.

**Takeaways**

- Column sampling behaved like a new seed: no gain (N-43).
- Recency weighting got worse the stronger it was, although validation said it would help. Validation folds with neighbours on both sides can mislead (N-44).
- Correction models stopped after a few trees and landed back at run 11; joint training wins (N-45). Time to stop exploring and run the final test.

### N-43. T-18: column sampling and max20 on top of run 22 (runs 24–26)

**Recorded:** 2026-11-04. Baseline run 22 (3 seeds). One change per run, seed 70563.
Ad hoc: `scratchpad/ffx.py`; notebooks `runs/run-2{4,5,6}_*/analysis.ipynb`.

| Model | Rank IC | Pearson | R² dm | \|slope − 1\| | W4 R² dm / slope | Δ vs 22 (t) | Δ vs 22 seed mean (t) | Agreement with 22 | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| run 22 | 0.0735 | 0.1026 | 0.01064 | 0.146 | 0.00720 / 0.77 | — | +0.0001 | — | 13.1 min |
| run 24 (ff 0.8) | 0.0735 | 0.1028 | 0.01072 | 0.143 | 0.00763 / 0.77 | −0.0000 (−0.0) | +0.0001 (0.4) | 0.994 | 13.2 min |
| run 25 (ff 0.5) | 0.0737 | 0.1028 | 0.01072 | 0.155 | 0.00741 / 0.79 | +0.0002 (1.0) | +0.0003 (1.5) | 0.990 | 11.7 min |
| run 26 (+ max20) | 0.0734 | 0.1022 | 0.01058 | 0.157 | 0.00713 / 0.78 | −0.0001 (−0.2) | −0.0000 (−0.0) | 0.986 | 12.2 min |
| mix 22 + 25 | 0.0738 | 0.1030 | 0.01073 | 0.152 | 0.00736 / 0.79 | +0.0003 (2.1) | +0.0004 (2.4) | — | 24.8 min |

Per window ΔRank IC vs 22: 24 −0.0005 / −0.0001 / +0.0011 / +0.0004 / −0.0001 / −0.0008;
25 −0.0000 / +0.0005 / +0.0013 / +0.0009 / −0.0008 / −0.0004; 26 +0.0014 / −0.0000 /
+0.0013 / −0.0020 / −0.0005 / −0.0006. Months better 18 / 17 / 18 of 37 (seed 22.1: 17).

- **Column sampling = another seed.** Agreement with run 22 0.994 / 0.990 vs 0.993 between
  seeds; Rank IC, R² and calibration move inside the seed band. It spreads the gain (top-10
  share 42% → 40% → 37%, Herfindahl 0.031 → 0.028 → 0.024) without finding new signal: the
  near-copies (T, T−1, T−5) carry the same information. 0.5 is 11% faster. No regime effect
  (24: normal +0.0007, t 2.2, but a seed moves normal days by −0.0007, t −2.3).
- **max20 adds variance, not signal.** 6.0% of gain (4.7–6.7% per window), taken from labels
  (15.0 → 13.0%), lags and raw; Rank IC tie; R² −0.00006, Pearson −0.0004, |slope − 1| +0.011;
  monthly Δ twice as spread as a seed's (SD 0.0026 vs 0.0017); ahead until mid-2024, then
  loses it (W4 −0.0020). Fold-mean validation R² is lower in **all six windows** (mean
  −0.0008; W4 0.00875 vs 0.01116), so validation would have rejected it too. Recent big
  winners (top max20 quintile) unchanged: run 22 already ranks them well (within IC 0.153),
  the five scaled labels already carry the max effect.
- **Averaging 22 + 25** (+0.0003, t 2.1) is the size of seed averaging: not worth 2× cost.
- **Decision support:** run 22 stays the reference. ff 0.5 is a free option (same quality,
  faster); max20 can be dropped. Remaining levers: magnitudes (W4 slope 0.77) and T-17.
- Configurations compared: 27 (24, 25, 26 added).

### N-44. T-19: recency-weighted training (runs 27, 28)

**Recorded:** 2026-11-04. Run 22 with `RECENCY_HALF_LIFE` 120 (run 27) / 60 (run 28); fold-local
weights (tooling note, 2026-11-04). Ad hoc: `scratchpad/rwx.py`; notebooks
`runs/run-2{7,8}_*/analysis.ipynb`.

| Model | Rank IC | Pearson | R² dm | \|slope − 1\| | W4 R² dm / slope | Δ vs 22 (t) | Windows won | Agreement |
|---|---|---|---|---|---|---|---|---|
| run 22 | 0.0735 | 0.1026 | 0.01064 | 0.146 | 0.00720 / 0.77 | — | — | — |
| run 27 (hl 120) | 0.0684 | 0.0987 | 0.00956 | 0.211 | 0.00429 / 0.66 | −0.0051 (−3.7) | 1/6 | 0.974 |
| run 28 (hl 60) | 0.0655 | 0.0939 | 0.00819 | 0.282 | 0.00194 / 0.57 | −0.0080 (−3.6) | 0/6 | 0.930 |
| mix 22 + 27 | 0.0713 | 0.1014 | 0.01024 | 0.180 | 0.00589 / 0.73 | −0.0022 (−3.0) | 3/6 | — |

Per window ΔRank IC (27 / 28): W1 −0.0000 / −0.0000, W2 −0.0109 / −0.0199, W3 −0.0011 /
−0.0003, W4 −0.0129 / −0.0212, W5 +0.0001 / −0.0004, W6 −0.0056 / −0.0060.

- **Worse, with dose-response** on ranking and magnitudes; outside the band against each of
  run 22's seeds; losses in calm, normal and turbulent regimes alike; Q5 − Q1 0.0087 → 0.0083
  → 0.0080. Run 28 is no better than run 11 (lags only).
- **W4 is worse from July 2024**, months before the October shock (July slope 0.86 → 0.71 →
  0.58; October 0.32 → 0.04 → −0.06). Prediction SD unchanged (0.0075 / 0.0075 / 0.0080): the
  slope falls from lost signal, not larger predictions.
- **Why:** weights only re-weight the past; W4's last training date is 2024-06-13, four months
  before the shift. Upweighting the newest months (for W4: W3's period, already hard for label
  history, N-41) amplifies a small, idiosyncratic sample; the effective sample shrinks (87% /
  65%).
- **Validation was misleading:** fold-mean Rank IC +0.0018 / +0.0037, test −0.0051 / −0.0080
  (per-window correlation −0.68 / −0.58). Fold-local weights let each block lean on neighbours
  on both sides; deployment only has the past. Lesson: don't trust k-fold validation for
  time-local changes (weights, short-memory features) without a test check.
- **Consequence:** the W4 miscalibration is not fixable by re-weighting history (T-19) or by
  dispersion sizing (T-10, N-31). Equal weights stay; `RECENCY_HALF_LIFE = None`.
- Configurations compared: 29.

### N-45. T-21: correction models on top of run 11 (runs 29, 30)

**Recorded:** 2026-11-05. `CORRECTION_OF` = run 11: its out-of-fold (history) and test
predictions are the starting score; a correction model learns the rest, from the 5 scaled label
columns (run 29) or the raw features plus those labels (run 30, 53 columns). Ad hoc:
`scratchpad/corrx.py`; notebooks `runs/run-{29,30}_*/analysis.ipynb` (baseline run 22,
leaderboard with run 11).

| Model | Rank IC | Pearson | R² dm | \|slope − 1\| | Correction trees | Δ vs 22 (t) | Δ vs 11 (t) | Windows won vs 22 | Agreement with 11 |
|---|---|---|---|---|---|---|---|---|---|
| run 11 | 0.0674 | 0.0997 | 0.00997 | 0.184 | — | −0.0061 | — | — | 1 |
| run 22 | 0.0735 | 0.1026 | 0.01064 | 0.146 | — | — | — | — | 0.932 |
| mix 11 + 17 (equal) | 0.0716 | 0.1034 | 0.01065 | 0.187 | — | — | — | — | — |
| run 29 (labels) | 0.0691 | 0.1004 | 0.01005 | 0.186 | 9 / 9 / 1 / 30 / 20 / 1 | −0.0044 (−2.1) | +0.0016 (+2.7) | 1/6 | 0.998 |
| run 30 (raw + labels) | 0.0675 | 0.0998 | 0.00998 | 0.184 | 1 everywhere | −0.0060 (−2.5) | +0.0001 (+4.0) | 1/6 | 1.000 |

Per window ΔRank IC vs run 22 (29 / 30): W1 −0.0057 / −0.0075, W2 −0.0109 / −0.0126, W3 +0.0055 /
+0.0056, W4 −0.0027 / −0.0057, W5 −0.0077 / −0.0104, W6 −0.0052 / −0.0052.

- **Worse than run 22** (t −1.9 / −2.4 against each of its other seeds too), in calm and
  normal regimes (turbulent: tie); better in 9 of 37 months, mostly W3's, where run 11 already
  beat run 22 (N-41). Also below the equal-weight mix 11 + 17 (−0.0025 / −0.0040).
- **The correction barely moves run 11:** its spread is 0.2–11% (run 29) and 0.5–0.8% (run 30)
  of run 11's. It points the right way (Rank IC vs run 11's residual +0.023, t 4.6 / +0.020,
  t 4.9) but is far too small to matter.
- **Why so few trees:** on validation, correction trees raise Rank IC but lower R² (run 29 W2:
  Rank IC 0.0465 → 0.0538 over 209 trees, R² 0.00783 → 0.00678, mostly fold 1, Jul – Sep
  2022). The R² stopping rule (kept for magnitudes) stops them after 1–30 trees.
- **Unconstrained it still loses:** run 29's best validation Rank IC at any tree count is below
  run 22's in 5 of 6 windows (W5 the exception, 0.0606 vs 0.0574); run 30's in all 6. The
  stacking leak would, if anything, flatter the correction there. So stopping on Rank IC would
  cost R² and still not reach run 22.
- **Reading:** label history helps inside one model, where it combines with the current
  features in every tree; as a fit to run 11's leftovers it adds little. Runtime: 3.4 / 3.5 min
  on top of run 11's 12.5 min.
- **Consequence:** T-21 closed; run 22 stays the primary; the final evaluation is unchanged.
  `CORRECTION_OF = None`.
- Configurations compared: 31.

---

## Chapter 9. The final evaluation

*2026-11-05*

The last window (84 test dates, 2025-12-19 → 2026-04-28) was kept unseen through
all of the above. Before running it, we wrote down the primary candidate, the question
and the decision rule. Then all candidates were trained once on it and scored.

**Takeaways**

- Pre-registration keeps you honest: the primary candidate (run 22) and the criteria were fixed before any final-window number existed.
- Outcome: run 22 vs Run-5 (demeaned) **consistent** (+0.0020 Rank IC, t 1.0, confidence interval includes 0); vs the author's Run-2 a **tie** on ranking (+0.0000), with slightly better R² and Pearson IC. The baselines had an unusually strong window.
- Over all seven windows (804 dates), run 22 raised Rank IC from 0.0592 (Run-2) to 0.0730 (+0.0138, t 6.4; better in 6 of 7 windows and on 65% of days) and R² from 0.0074 to 0.0101 (`all-windows_performances.ipynb`, report `ts48.pdf`).
- The gain depends on the period. More recent data are needed to know whether it holds in other market conditions. Averaging several models was the most robust choice on the new window, which is a hint, not a result.

### The plan, fixed before the final run

**Pre-registration of the final evaluation (T-20, fixed 2026-11-05 before any final-window
result; user agreed to name the primary first):**
- **Primary candidate: run 22.** Primary question: does it beat Run-5 (demeaned) and Run-2
  (raw) on the final window? *Confirmed*: ΔRank IC > 0 with the 95% NW CI (4 lags) excluding
  0, and R² / calibration not worse; *consistent*: ΔRank IC > 0, CI includes 0; *not
  confirmed*: ≤ 0. Power: six-window gains +0.0104 vs Run-5 (t 4.4 / 720 days) and +0.0154 vs
  Run-2 (t 6.6) imply t ≈ 1.5 and 2.2 on 84 days.
- **Secondary** (descriptive vs run 22): run 23; mix 11 + 17 and the lag mix (11, 13–16),
  each with equal and fitted weights. None replaces run 22 unless it wins clearly on ranking
  and magnitudes; even then flagged as a multiple-comparison result. Runs 11 and 17 on their
  own (demeaned) added 2026-11-05 at the user's request, after the final models were
  trained, as descriptive references for the 11 + 17 mix.
- **Post-processing:** per-day demeaning for all but Run-2; mixes average demeaned
  predictions on common rows; fitted weights ≥ 0, sum 1, least squares on the final window's
  out-of-fold validation predictions (demeaned; history only); nothing else.
- **Reproduction:** `train_final-window.py` checks every setup's head settings and LightGBM
  parameters against its archived script, and retrains window 6 (Run-2: windows 1–6, its
  script can't pick windows) to compare with the archived predictions (`verify.json`).
- **Multi-day reference: run 22** (raw + lags {1, 5} + labels T−6 … T−10 scaled per day) from
  2026-10-27 (user decision, N-41); before that run 11. Compare new family runs with run 22
  (`analyze-ts48-results --baseline runs/run-22_…`).

**Candidates for the final evaluation (Stage 6, reserved window, I-6):**
- **Lag-mix ensemble** of the {1, x} runs (11, 13, 14, 15, 16; T-17, N-38): equal weights,
  or weights fitted on out-of-fold validation predictions. Added 2026-10-26 (user). Caveats
  for the weight fit: the out-of-fold predictions come from the fold models (80% of the
  history), and validation is biased +0.0018 toward fewer lags (N-37), so fitted weights may
  lean toward the members closest to lag 1; with ~5 weights on noisy daily ICs, shrink toward
  equal weights.
- **Average of run 11 and run 17** (lags {1, 5} ⊕ raw labels T−6 … T−10; N-39): equal
  weights on the demeaned predictions, a post-processing step with no extra training. Ad hoc
  on test: Rank IC 0.0716, R² dm 0.01065, +0.0041 vs run 11 (t 2.5), 5/6 windows. Added
  2026-10-27 (user). Compare it in the final evaluation with the single combined model
  (`raw+lags+labelraw-6-10`, T-16 step 3) and, if useful, weights fitted on out-of-fold
  validation predictions (same caveats as above).
- **Run 22 / run 23** (lags {1, 5} + raw labels scaled / ranked per day; N-41): the best single
  models so far (Rank IC 0.0735 / 0.0733, R² dm 0.01064 / 0.01062), and they beat the 11 + 17
  average on ranking at equal R² and half the cost. Added 2026-10-27.

**Main + auxiliary model ideas, status (2026-11-05, user question):**
- Auxiliary dispersion model for sizing (HAR, T-10): tested, failed (N-31). Dropped.
- Blending separately trained models: tested as run 7 ⊕ run 5 (N-30, no gain) and the
  {1, x} lag mix / 11 + 17 average (gains, final-eval secondaries).
- Correction model (LightGBM `init_score` = the main model's out-of-fold predictions; a
  second model fits what is left, N-34): tested in T-21 (runs 29, 30, 2026-11-05, before
  the final run): worse than run 22 and practically equal to run 11 (N-45). Dropped.
  Joint training (run 22) beats both the mix and the correction route.

### N-46. Final evaluation (T-20): results on the reserved window

**Recorded:** 2026-11-05. Window 7 (history 2024-12-24 → 2025-12-04; test 84 dates, 2025-12-19 →
2026-04-28; 241,061 common rows). All 10 setups reproduced window 6 exactly. Full verdict:
`final-evaluation_performances.ipynb` (last cell; never rebuild that notebook with --force).

| Candidate | Rank IC | Δ vs run 22 (t) | Δ vs Run-5 (t) | Δ vs Run-2 (t) | R² | Slope |
|---|---|---|---|---|---|---|
| Run-2 (raw) | 0.0684 | −0.0000 | +0.0020 (+0.9) | — | 0.0057 | 1.03 |
| Run-5 (dm) | 0.0665 | −0.0020 (−1.0) | — | −0.0020 (−0.9) | 0.0056 | 1.19 |
| **Run 22 (primary)** | 0.0685 | — | +0.0020 (+1.0) | +0.0000 (+0.0) | 0.0061 | 1.20 |
| Run 23 | 0.0691 | +0.0006 (+2.3) | +0.0026 (+1.3) | +0.0007 (+0.2) | 0.0061 | 1.22 |
| Run 11 | 0.0694 | +0.0010 (+1.1) | +0.0029 (+1.8) | +0.0010 (+0.3) | 0.0058 | 1.19 |
| Run 17 | 0.0669 | −0.0016 (−0.9) | +0.0004 (+0.3) | −0.0016 (−0.6) | 0.0061 | 1.13 |
| Mix 11+17 equal | 0.0707 | +0.0022 (+2.7) | +0.0042 (+2.9) | +0.0023 (+0.9) | 0.0063 | 1.25 |
| Mix 11+17 fitted | 0.0702 | +0.0017 (+1.7) | +0.0037 (+2.6) | +0.0018 (+0.7) | 0.0063 | 1.23 |
| Lag mix equal | 0.0702 | +0.0018 (+2.3) | +0.0037 (+2.1) | +0.0018 (+0.6) | 0.0058 | 1.19 |
| Lag mix fitted | 0.0705 | +0.0021 (+2.6) | +0.0040 (+2.2) | +0.0021 (+0.7) | 0.0058 | 1.20 |

- **Pre-registered outcome:** vs Run-5 *consistent* (Δ > 0, CI includes 0; R² +0.0005, slope
  equal); vs Run-2 formally *consistent* (+0.00002) but a tie on ranking; R² and Pearson IC
  higher, calibration worse (1.20 vs 1.03).
- **Per window, run 22 − Run-5:** +0.0105, +0.0140, −0.0040, +0.0097, +0.0207, +0.0114, final
  +0.0020; **− Run-2:** +0.0140, +0.0205, −0.0009, +0.0112, +0.0260, +0.0217, final +0.0000.
  The second-weakest of 7 windows, after W3. As in W3, run 11 (no label history) beat run 17
  and run 22: label history is the period-sensitive part.
- **Baselines unusually strong:** Run-2 0.0684 (six-window 0.0581), Run-5 0.0665 (0.0631); run
  22 0.0685 (0.0735), inside its single-window range.
- **Timing:** Run-2 led December 2025 (0.0709 vs 0.0569) to early January; run 22 caught up and
  led April. Turbulent dates favour run 22 (vs Run-2 +0.0086, t 3.7), calm / normal favour
  Run-2 (−0.005, t −1.0); 1–2 episodes per regime.
- **Mixes best**, reversing the six-window order; gains just outside the single-window seed
  band (±0.0018), eight comparisons: descriptive, no replacement. Fitted weights no better
  than equal. Hint: averaging models (variance reduction) was the most robust on new data;
  testing it needs data after 2026-04-28.
- **Magnitudes:** monthly calibration slopes 0.24–0.48 (December) to 1.44–1.78 (April) for all
  models alike: a period effect.

---

## Appendix A. Experiment queue and decisions

The to-do list as it evolved: each planned experiment (T-n), its status and its source. The reference model used for comparisons is described first.

Reference (**decided with the user 2026-10-23**): **Run-5 (demeaned)**.
- Configuration: purged 5-fold, all 48 features, **GOSS**, retrain on all history,
  and **per-day demeaned output** (`prediction − day mean`).
- Seeds 70563 / 94086 / 258128 (runs 5, 5.1, 5.2).
- Before: run 5 raw (from 2026-10-22), run 3.
- **Why GOSS over bagging (a tie on ranking and R², N-23/N-24):**
  - per-window calibration after demeaning is better beyond seed noise (0.178–0.184
    vs 0.193–0.201);
  - Rank IC seed noise is lower (±0.0004 vs ±0.0007; bagging needs ~2.8× the
    seeds for the same power);
  - it is the author's choice.
- If T-6 helps, check it once with bagging too: bagging suits T-6's aim of
  limiting single-row influence.
- **Rules:**
  - One change at a time.
  - Judge after demeaning.
  - Use ≥ 2 fixed seeds per new configuration (70563, 94086).
  - Paired comparison against the reference with the analysis skill.

Order agreed with the user on 2026-10-21: purged k-fold first, then GOSS and
the loss. Revised 2026-10-23 (N-20): keep MSE; bagging, then leaf
regularization, then both.

| # | Status | Experiment | Why / source |
|---|---|---|---|
| T-1 | **Done 2026-10-22** (run 4 = filter, run 5 = all features; N-17, N-18; all features adopted) | Purged k-fold (e.g. 5 contiguous blocks, 10-date purge each side) to choose the tree count from averaged out-of-fold curves; retrain on all history. Run it **with and without** the `CORR_BAR` filter (two runs) | One validation period gives unstable tree counts (N-14, N-15). Feature selection is unstable (N-14) |
| T-2 | **Done 2026-10-21** (no training) | Tree-count sensitivity: saved models scored at truncated `num_iteration` | Result in N-15: tree count sets the prediction scale; the ranking optimum varies by window |
| T-3 | **Done 2026-10-23** (runs 5 and 6; N-24: the slope fails, per-day demeaning wins) | Out-of-fold scale correction: multiply predictions by the calibration slope fitted on k-fold out-of-fold predictions. Analysis only, no training | Decouple magnitudes from the tree count (N-15); Rank IC unchanged by construction |
| T-4 | **Done 2026-10-22** (runs 5.1, 5.2; N-19) | Seed-noise floor: reference config with 2 more seeds | Needed to judge deltas of ~0.004 |
| T-5 | **Done 2026-10-23** (run 6; N-21–N-23: a tie with GOSS after the seed check; GOSS kept as the reference, see above) | GOSS → bagging (`bagging_fraction` 0.8, `bagging_freq` 1; `subsample` renamed). Also saves out-of-fold predictions | GOSS focuses on large-residual rows (I-2); later trees seem to chase tails (N-15); N-20 |
| T-6 | **Done 2026-10-23, premise refuted** (N-25: keep `min_data_in_leaf` 20; stage 2 `reg_lambda` dropped) | Leaf regularization, picked on validation only: `min_data_in_leaf` 20/100/200/1000/4000 | Stop single jump rows from setting leaf values (N-20) |
| T-7 | **Dropped** (T-6 didn't help) | T-6 with bagging instead of GOSS | Bagging suits limiting single-row influence; checks the interaction |
| T-8 | Optional | Huber loss, threshold ≈ 99th percentile of \|residual\|; check magnitudes per ex-ante σ decile | Robust alternative; biased toward the median under positive skew (N-20) |
| T-9 | **Done 2026-10-23** (runs 6.1–6.3; N-23) | Bagging seed check: run 6 with 1–2 more seeds | Bagging randomizes every tree; the W2 loss and the W5/W6 gains need a noise yardstick (N-22) |
| T-10 | **Done 2026-10-24, negative** (N-31: dispersion sizing via HAR A0–A3 and even the Oracle lose to the demeaned output) | Regime-aware scaling: size predictions by a forecast of each day's dispersion (auxiliary HAR model) | The slope is period-dependent (N-24) |
| T-11 | **Done 2026-10-23** (N-26: keep 20; more leaves lower Rank IC in 6/6 windows) | Sweep `num_leaves` 20 / 40 / 80 / 120 (`max_depth` 7 allows ≤ 128), validation only | Small leaves carry tail signal (N-25): test more capacity for fine partitions |
| T-12 | **(a) done 2026-10-24, single-day family closed 2026-10-25** (N-27–N-31: no single feature set, blend or day-level sizing beats Run-5 demeaned); (b)/(c) next, evaluation protocol to settle first (N-30) | Feature engineering (stage 5): (a) single-day sets; (b) the stock's own trailing label statistics, lagged ≥ 7 dates; (c) lags / rolling means of the 48 features | Tree tuning is exhausted (N-25, N-26) |
| T-13 | **Done 2026-10-24** (N-29, N-30: run 7 worse alone; blend no gain; not adopted) | Run 7 = raw+rank+daystd (`train_single-day-feature-sets.py`, seed 70563); blend its test predictions 50/50 with run 5's (demeaned), compare with Run-5 (demeaned) | Validation blend: R² kept, Rank IC +0.0024 (N-28) |
| T-14 | **Done 2026-10-25** (N-33: run 11 raw+lags better on both; label history best Rank IC but worse R²; validation trusted for lags/dev only) | Multi-day grid, full runs, seed 70563: `raw+labelhist-max` (8.1), `raw+labelhist-min` (8.2), `raw+dev`, `raw+labelhist-…+dev`, `raw+lags`; compare each with Run-5 (demeaned), then the validation-agreement check | T-12 (b)/(c); protocol N-32 |
| T-15 | (a) **done 2026-10-26** (N-35: run 11 holds over 3 seeds); (b)–(e) see T-16 | (a) seed check run 11; (b) lag-set sweep on validation; (c) per-day-ranked label history; (d) lags + labelhist-max; (e) blend 11 ⊕ 8.1 | N-33 |
| T-16 | 1) **done** (N-36 sweep, N-37 run 12, N-38 runs 13–16: keep lags {1, 5}); 2) **done** (N-39 runs 17–20: raw labels best); 3) **done** (N-41: runs 22/23 best; candidate new reference): sets `raw+lags+labelraw-6-10`, `raw+lags+labelraw-6-10-scaled` (run 20's fix) and `raw+lags+labelraw-6-10-rank` (run 19's fix, best calibration) added 2026-10-27 (149 columns, verified identical to runs 11 / 17's columns) | 1) sweep `test_multi-day-feature-sets.py` (lag sets, fmean); 2) full runs of the label shots (labelraw-6-10, labelstats-max, labelhist-max-rank / -scaled); 3) best lags + best label history | N-34, N-35 |
| T-17 | **Candidate for the final evaluation** (user decision, 2026-10-26); follow-ups optional | Lag-mix ensemble (N-38): average the {1, x} runs' test predictions (no extra training). (a) **weights fitted on the out-of-fold validation predictions** (`predictions/validation_*.parquet`, per window, history only: no leak), non-negative and summing to 1, shrunk toward equal weights; compare with equal weights; (b) one model with lags 1–6 (full run); (c) 5 seeds of one {1, x} set blended, to separate lag diversity from model count; (d) a second seed of run 12 | The equal-weight {1, x} blend beat run 11's seed blend (+0.0008, t 2.3) and keeps lag 1 in every member |
| T-18 | **Done 2026-11-04** (runs 24–26, N-43): no gain; run 22 stays the reference | On top of run 22: (a) `FEATURE_FRACTION` < 1 (per-tree column sampling, e.g. 0.8 and 0.5) with `FEATURE_SET = raw+lags+labelraw-6-10-scaled`; (b) `raw+lags+labelraw-6-10-scaled+labelmax-scaled` (150 columns: + the stock's max scaled label over 20 dates). Full runs, baseline run 22 | (a) in-model version of the {1, x} lag-mix diversity (N-38); (b) 8.1's W1/W5 gains came partly from max20, which run 17 dropped (N-39) |
| T-19 | **Done 2026-11-04, negative** (runs 27, 28; N-44): equal weights stay; next: final evaluation | Recency weighting on top of run 22 (`RECENCY_HALF_LIFE` in the family script): training dates weighted 0.5 ** (distance / half-life); retrain: distance to the test period; each purged fold: distance to its own validation block (so validation sees the effect and the tree count isn't distorted); weights mean 1; validation unweighted. Runs: half-life 120 (newest/oldest 3.8×, effective sample 87%) and 60 (14×, 65%). Baseline run 22; look at W4 (post-shock slope 0.77) first | User's choice of the options listed after N-43: adapt faster after regime shifts (Oct 2024: daily label SD 0.051 → 0.078). Afterwards: set up the final evaluation |
| T-20 | **Done 2026-11-05** (N-46; verdict cell in final-evaluation_performances.ipynb): primary vs Run-5 consistent, vs Run-2 a tie on ranking; final window spent | Final evaluation (Stage 6, I-6) on the reserved final window (84 dates, 2025-12-19 → 2026-04-28; models trained on history to 2025-12-04): `train_final-window.py` trains the 10 setups (runs 2, 5, 11, 13–17, 22, 23) into `runs/run-final/<setup>/`; `runs/run-final/post-processing.ipynb` writes `final-predictions-<candidate>.parquet`; `final-evaluation_performances.ipynb` (root) evaluates. Plan below ("Pre-registration of the final evaluation") | The candidates list; user request 2026-11-05 |
| T-21 | **Done 2026-11-05, negative** (runs 29, 30; N-45): run 22 stays the primary; nothing added to the final evaluation; next: final evaluation | Correction models on top of run 11 (`CORRECTION_OF` = run 11's folder: starting score = its out-of-fold predictions on the history, its test predictions on test; LightGBM init_score): run 29 `labelraw-6-10-scaled` (5 label columns), run 30 `raw+labelraw-6-10-scaled` (53). Baseline run 22 (same information, one model). If one beats run 22 beyond the seed band, it joins the final evaluation as a secondary candidate **before** the final run | N-34's long shot; user question 2026-11-05 ("main + auxiliary models") |

---

## Appendix B. Index of issues and notes

### Issues

| # | Status | Stage to address | Issue |
|---|---|---|---|
| I-1 | **Resolved 2026-10-21** (run 3; leak effect small, see N-15) | Before training the baseline | Interleaved validation dates leak through overlapping forward labels |
| I-2 | Partly addressed 2026-10-23 (run 6: GOSS → bagging, `subsample` made effective; leaf settings in T-6) | Before training the baseline | Several LightGBM parameters are ineffective or questionable (`subsample` unused, `min_child_weight` unused, no `min_data_in_leaf`, L1 strength unknown, GOSS on noisy labels) |
| I-3 | **Resolved 2026-10-18** | Before training the baseline | Time-based seeds make runs non-reproducible (fixed: `RANDOM_SEED`, per-window `RANDOM_SEED + i`) |
| I-4 | Low impact (2026-10-19) | Stage 2–3 (evaluation) | Reported `kt` is pooled over all test days, unlike the per-day IC. Run 2: pooled ≈ daily Kendall, because the label's daily mean ≈ 0 |
| I-5 | Open | Stage 2–3 (evaluation) | Reported `ir` is really a t-statistic: it grows with window length (last window not comparable) and is inflated by overlapping labels |
| I-6 | Open | Before the first full (multi-window) run | Final window (last 84 test dates) must be reserved for the Stage 6 final evaluation; no flag yet excludes it |
| I-7 | **Resolved 2026-10-18** | Before the first run | Runs overwrote each other; no per-row predictions, learning curves, runtime or memory saved |
| I-8 | **Parked 2026-10-21** (not pursued further) | — | What the label is: a 5-day forward residual return from a Barra-type factor model; scale details unresolved. Report label units, attach no further financial meaning |
| I-9 | Open | Stage 5 (history features) | Universe gaps: two near-empty dates (2022-06-20: 709 rows, 2022-06-28: 126), yearly late-April/early-May dips, suspensions and new listings, so history features must be aligned by stock and date, with explicit missing handling |

### Notes

| # | Topic | Possible stage |
|---|---|---|
| N-1 | Same-day correlation between stocks: why per-stock modelling is acceptable, and how to use cross-stock information | Stage 5+ (after history experiments) |
| N-2 | LightGBM parameter guide, and which settings to explore first | Stage 4 (hyperparameter exploration) |
| N-3 | Leaves, hessian "weight", `min_data_in_leaf`, and how L1/L2 penalties shrink leaf values | Stage 1 (understanding) / Stage 4 |
| N-4 | One pooled model per window: how it handles differences between stocks and changes over time | Stage 1 / Stage 4–5 |
| N-5 | `lgb.train` monitoring: `valid_sets`, `feval` vs `metric`, callbacks and early stopping | Stage 1 / Stage 3 (learning curves) |
| N-6 | IC (Pearson) vs rank metrics (Kendall, Spearman / Rank IC); outlier sensitivity | Stage 3 (evaluation) |
| N-7 | Reading run 1 (reference baseline, window 1): curves, metrics, what they imply | Stage 3 |
| N-8 | Run 2 (reference baseline, windows 1–6): what the six windows show | Stage 3 |
| N-9 | *Not in the public copy:* a single-stock case study about the label's construction | — |
| N-10 | Handling extreme labels: separate training, evaluation and universe; clip/Huber/rank vs drop | Stage 4 (after run 3 fixes I-1) |
| N-11 | Do models learn the "normal" relation? Why not to remove turbulent periods from metrics; robust, conditional and paired metrics instead | Stage 3–5 (evaluation) |
| N-12 | Why the edge is one-sided (losers): Q1 = volatile, lottery-like stocks; winners are unpredictable | Stage 3 / interpretation |
| N-13 | Predictions feed the author's portfolio optimizer: ranking **and** magnitude both matter | All stages (evaluation and model choices) |
| N-14 | Run 3 curves: unstable tree counts, unstable feature selection, why retrain R² is lower | Stage 3–4 (protocol) |
| N-15 | Run 3 vs run 2: ranking +0.0043, magnitudes mixed; tree count sets the prediction scale | Stage 3–4 |
| N-16 | Purged k-fold design: fold sizes, feature selection per fold, stopping on fold-mean R² | Stage 3–4 (protocol) |
| N-17 | Runs 4–5 curves: stable tree counts, period-dependent fold optima, all features ≥ filter in validation | Stage 3–4 |
| N-18 | Runs 4–5 test results: k-fold fixes magnitudes; all features slightly better (2025); run 5 = new reference | Stage 3–4 |
| N-19 | Seed noise (runs 5, 5.1, 5.2): tiny overall, notable per window (W6); noise ceilings for single-seed comparisons | Stage 3–4 (evaluation) |
| N-20 | Bulk vs tails: tails are rare upward jumps across nearly all stocks; keep MSE, curb single-row influence (bagging, leaf regularization); bagging parameters; out-of-fold predictions | Stage 4 |
| N-21 | Run 6 curves: validation R² up in 6/6 windows (+1–3%), Rank IC mixed; trees unchanged; early Rank IC spikes are flukes | Stage 4 |
| N-22 | Run 6 vs run 5 (test): Rank IC +0.0009 (t 2.5), gain from 2025; bulk magnitudes better, tails unchanged; calibration slightly worse via tree count | Stage 4 |
| N-23 | Bagging seed noise (runs 6–6.3): Rank IC ceiling ±0.0007; bagging vs GOSS is a tie (+0.0005 ± 0.0008); calibration slightly worse via smaller scale; corrects N-22 | Stage 4 |
| N-24 | Post-rescaling (T-3), runs 5 and 6: the out-of-fold slope fixes the pooled slope but worsens per-window calibration; per-day demeaning gives R² +7%, robust over seeds | Stage 4 (post-processing) |
| N-25 | Sweep leaf-1: larger `min_data_in_leaf` loses R² (−26% at 4000) mostly on tail rows; small leaves carry validated tail signal; coarse trees rank slightly better | Stage 4 |
| N-26 | Sweep leaf-2: more leaves lower Rank IC (0/6 windows) and R²; the author's tree settings are an optimum for R²; Rank IC prefers simpler trees | Stage 4 |
| N-27 | Sweep feat-1 round 1: raw values carry magnitudes (all raw-free sets lose R²); day stats raise Rank IC (+0.0017) but cut R² (−23%); round 2 queued | Stage 5 |
| N-28 | Sweep feat-1 complete: raw = magnitudes, rank = small Rank IC help, daystd = Rank IC +0.0015 but R² −15% (noisy per-day sizing), daymean useless; 50/50 blend raw ⊕ raw+rank+daystd keeps R², Rank IC +0.0024 | Stage 5 |
| N-29 | Run 7 (raw+rank+daystd) vs runs 5/2: Rank IC tie vs run 5 (W4 −0.024 in the Sep–Nov 2024 shock, 2025 +0.01), R² −25%, calibration 0.18 → 0.40; validation over-rewards regime features | Stage 5 |
| N-30 | Blend run 7 ⊕ run 5: validation-chosen weights slightly worse than run 5; hindsight best +0.0003 Rank IC (noise); run 7 not adopted | Stage 5 |
| N-31 | Day-level context (level vs regime mapping vs sizing); T-10 dispersion sizing (HAR, A0–A3, Oracle) loses to the demeaned output: IC drops on high-dispersion days; the model's own spread sizes days best | Stage 4–5 |
| N-32 | Multi-day family: full runs judged on W1–W6 plus a validation-agreement check (no sweeper first); labels realized by t − 6; trailing summaries and label history instead of stacked lags | Stage 5 |
| N-33 | Multi-day grid: run 11 (raw+lags) better than run 5 on ranking and magnitude (+0.0043, t 5.3, 6/6; R² dm +6%); label history best Rank IC (10.1 +0.0085) but R² worse, breaks after shocks; max > min (t 3.0); validation trusted for lags/dev, not label history | Stage 5 |
| N-34 | History taxonomy (raw/derived × features/labels): hypotheses; latest raw label = most important column in run 8.1 (13% gain, rank 1 in W4/W5), suspected source of gains and magnitude breaks; next shots | Stage 5 |
| N-35 | Run 11 seed check: all 3 seeds beat all 3 run-5 seeds (Rank IC 0.0665–0.0674 vs 0.0631–0.0634; R² dm 0.00997–0.01005 vs 0.00934–0.00942); feature-history sweep + label-split sets set up | Stage 5 |
| N-36 | Sweep mday-1: lag 1 alone best R² (96 cols), {1,2} best Rank IC; older lags and feature means hurt (means: R² −26%); overfitting (train/val) grows with slow-moving stock-level columns, also in label/dev runs (H5: stock fingerprints) | Stage 5 |
| N-37 | Run 12 (raw + lag 1): ties run 11 on ranking within seed noise (−0.0010, t −1.7; losses only W5/W6), magnitudes slightly worse; vs run 5 better (+0.0033, t 6.7, 6/6); sweep biased +0.0018 toward lag 1 in every window; keep lags {1, 5} | Stage 5 |
| N-38 | Lag series {1, x}, x = 2, 3, 4, 6 (runs 13–16): all tie run 11 {1, 5} within seed noise (vs its seed mean +0.0000 … +0.0001); lag 5 is not special; any second lag beats lag 1 alone slightly, late (W5/W6); equal-weight blend of the five {1, x} runs beats run 11's 3-seed blend (+0.0008, t 2.3) | Stage 5 |
| N-39 | Label series (runs 17–20) vs 8.1: raw labels T−6…T−10 (17) keep the ranking (0.0700, best Pearson 0.1008) and recover R² (0.00978, +14%; train/val 4.1 → 3.2); per-day rank / scaling (19/20) fix post-shock calibration (W4 slope 0.58 → 0.92/0.80; \|slope−1\| 0.085 best ever); stats without the latest label (18) worst; blend 11 + 17: Rank IC 0.0716, R² 0.01065 (t 2.5 vs 11) | Stage 5 |
| N-40 | Components (run 11 − run 5 vs run 17 − run 5): lags = what changed since yesterday (gain grows with the feature move, +0.0008 → +0.0088), broad and steady, strongest on volatile stocks and 2024; labels = last week's outcome (pushes against the latest label, corr −0.52), strong after extreme labels and on calm days, sharpens loser detection, hurts after very negative labels (−0.0126, t −4.1) | Stage 5 |
| N-41 | Step 3 (runs 21–23: lags {1,5} + raw labels plain / scaled / ranked) vs run 11: +0.0051 / +0.0061 / +0.0059 (t 2.0–2.5), R² dm 0.01028 / 0.01064 / 0.01062; normalized labels fix W4; runs 22/23 beat the 11 + 17 mix on ranking (+0.0019) at equal R² and half the cost; 22 ≈ 23 | Stage 5 |
| N-42 | Run 22 seed check (22, 22.1, 22.2): Rank IC 0.0732–0.0735, R² dm 0.01064–0.01072, \|slope−1\| 0.146–0.148; every 22 seed beats every 11 seed (+0.0057 … +0.0070, t 2.4–2.9); vs the 11 + 17 mix +0.0016 … +0.0019; ties run 23; label models not noisier; pooled band ±0.0008 Rank IC | Stage 5 |
| N-43 | T-18 on top of run 22: feature_fraction 0.8 / 0.5 (runs 24 / 25) and + max20 scaled (run 26): all ties within seed noise (Δ −0.0001 … +0.0002 vs ±0.0008); ff behaves like a new seed (agreement 0.990–0.994), ff 0.5 is 11% faster; max20 takes 6% of gain but adds no test gain, worse validation R² in all windows, slightly worse magnitudes; run 22 stays the reference | Stage 5 |
| N-44 | T-19 recency weighting (runs 27 / 28, half-life 120 / 60) vs run 22: worse, with dose-response: Rank IC −0.0051 / −0.0080 (t −3.7 / −3.6), R² dm 0.00956 / 0.00819, \|slope−1\| 0.211 / 0.282; W4 worse from its first month (slope 0.77 → 0.66 → 0.57); validation said better (+0.0018 / +0.0037): fold neighbours on both sides mislead; equal weights stay | Stage 5 |
| N-45 | T-21 correction models on top of run 11 (runs 29 / 30: 5 label columns / raw + labels) vs run 22: worse, Rank IC 0.0691 / 0.0675 (Δ −0.0044 / −0.0060, t −2.1 / −2.5), R² dm 0.01005 / 0.00998; ≈ run 11 (agreement 0.998 / 1.000): validation-R² stopping halts the correction after 1–30 trees; even unconstrained its best validation Rank IC stays below run 22's in 5 / 6 windows; joint training wins; nothing added to the final evaluation | Stage 5 |
| N-46 | **Final evaluation (T-20), window 7, 84 dates:** run 22 vs Run-5 +0.0020 ± 0.0040 (t +1.0) → *consistent*; vs Run-2 +0.00002 (t 0.0) → formally consistent, in substance a tie; run 22's R² (0.0061) and Pearson IC above both baselines; Run-2 had its second-best window (0.0684). Mixes best (11 + 17 equal 0.0707, +0.0022 vs 22, t 2.7), reversing the six-window order; descriptive only (8 comparisons). Verdict cell written | Stage 6 |

---

## Appendix C. Tooling

Scripts, skills and notebook builders written along the way (newest first).

- **`claude/analyze-ts48-results/`** (2026-10-20): skill for analyzing a run
  and comparing it with a baseline. `SKILL.md` holds the procedure, the
  interpretation checklist distilled from N-6 … N-12, the verdict criteria and
  the summary template. `cells/` holds the notebook cells; `scripts/` holds
  `build_notebook.py`, `run_notebook.py` and `set_text_cells.py`. Tested on a
  synthetic variant vs run 2, in no-baseline mode, and on run 1's flat
  layout. Linked as `.claude/skills/analyze-ts48-results` (symlink, 2026-10-21),
  so `/analyze-ts48-results` works.
- **Correction models (2026-11-05, T-21):** family script `CORRECTION_OF` (default None);
  `init_col` keyword in train_baseline's `build_folds` / `purged_cv` /
  `save_validation_predictions` (starting score in training and validation Datasets; added
  back to `Booster.predict()` output in the saved out-of-fold and test predictions). Sets
  without the raw group drop the raw inputs but rows are still filtered on them (row filter
  now uses every featureN column; unchanged for earlier sets). Known caveat: in the
  correction model's own folds, training rows' starting scores come from main fold models
  that saw the other folds' validation dates (standard stacking leak, second order).
  Tested: tiny synthetic run (12 stocks, 380 dates, 1.5 min): validation R² after one tree =
  the main model's out-of-fold R² (0.2781 → 0.2799), the planted label effect learned; real
  data: every history / test row of windows 1–6 finds its run-11 score.
- **All-windows table (2026-11-05, user request):** `all-windows_performances.ipynb` (repo
  root), built by `claude/final-eval/build_all_windows_notebook.py` (`--force` to rebuild;
  no hand-written cells). The ten final-evaluation candidates on W1–W7 (804 dates, 2,269,878
  common rows), paired vs Run-2 and Run-5 (NW 4 lags); setup rows plus footnotes defining
  every term. Reads the archives (W1–W6) and `runs/run-final` (W7); trains nothing. The
  weighted mixes' W1–W6 weights are fitted there with post-processing.ipynb's rule (W7's
  asserted equal to post-processing.json). Seed noise from runs 5 / 11 / 22 on W1–W6 (W7
  rows: pooled per-window noise; vs-Run-5 rows without Run-5's seeds). Checked: W7 values
  equal N-46's. Seven-window levels (Rank IC / R²): Run-2 0.0592 / 0.00740, Run-5 0.0635 /
  0.00896, run 22 0.0730 / 0.01009 (best Rank IC), mix 11 + 17 equal 0.0715 / 0.01013.
  Section 5 plots Rank IC, Pearson IC, R² and calibration slope per window (Run-2 black;
  each panel's winner over all windows bold: run 22, mix 11 + 17 equal, mix 11 + 17 equal,
  run 23), saved as `runs/run-final/plots/all_windows_per_window.png`.
  PDF for the author: `ts48.pdf` (repo root, 7 pages A4
  landscape: intro, grand table + footnotes as in the notebook, figure, mix weights,
  summary), built by `claude/final-eval/build_all_windows_report.py` in two steps (`parts`
  in .venv; `pdf` in an env with playwright + markdown-it-py; Chromium needed apt libs).
  The footnotes' internal references (paths, notes, issue I-1) are removed (CLEANUPS).
- **Final evaluation tooling (2026-11-05, T-20):** `train_final-window.py` (repo root; one
  subprocess per setup, resumable via `done.json`, `--dry-run` checks settings only,
  `--setup`, `--data-file` / `--final-dir` for tests); `WINDOWS` setting added to
  `train_baseline.py` and the family script (default None = all windows). Notebooks built by
  `claude/final-eval/build_final_notebooks.py` (`--force` overwrites; it would reset the
  hand-written verdict cell); `TS48_FINAL_DIR` / `TS48_ROOT` / `TS48_DATA_FILE` override
  paths for tests. Tested: dry run on the real archives (all 10 setups match); full batch on
  synthetic data (scratchpad); both notebooks on a stand-in built from the archived window-6
  files, reproducing the known W6 and six-window numbers (run 22 W6 0.0686, six-window
  0.0735 / 0.01064; mixes 0.0716 / 0.0679), fitted weights equal to a brute-force grid.
- **Recency weighting (2026-11-04, T-19):** `train_baseline.recency_weights()` and a
  keyword `half_life` in `build_folds()` / `purged_cv()` (default None: unweighted, every
  earlier script unchanged); the family script's `RECENCY_HALF_LIFE` uses them and records
  `recency_half_life` in the results. Fold-local design chosen after a synthetic test:
  with the retrain's weights in every fold, folds validating old blocks trained mostly on
  far dates, early stopping collapsed (1 tree vs 8) and validation penalized weighting by
  construction. Synthetic regime flip: the weighted retrain follows the recent regime
  (slope −0.022 vs +0.021), GOSS honors the weights. Caveat: middle folds have near dates
  on both sides (deployment only has the past), so validation may overstate the benefit;
  judge on test. The sweeper does not apply it.
- **Run-11 rename (2026-10-26, user request):** `run-11*_multi-day-rawfeat+lagfeat-p5fcv…`
  → `run-11*_multi-day-rawfeat+lag1feat+lag5feat-p5fcv…`, matching runs 12–16. Folders,
  archived scripts (file names, `RUN_DIR`, `ARXIV_FILE`), notebooks, the multi-day cache
  files and the sweeper's `REUSE` (also in the mday-1 archive) were updated; the caches
  stayed valid (their fingerprint uses file names, sizes and mtimes inside the folder).
- **Multi-day grand table (2026-10-27, user decision):** the table cell shows only Run-2,
  Run-5 and `TABLE_RUNS` (now 11, 17, 21, 22, 23, 25; runs 19, 24, 26, 27, 28 collapsed) plus runs above
  `REVIEWED_UP_TO` (28); the heading names the multi-day reference (run 22); every column
  sits in a collapsed, sideways-scrollable `full-table` cell below.
- **Multi-day plot (2026-10-26, user decision):** `multi-day_performance.ipynb` plots only
  Run-2, Run-5 (demeaned) and the 2 best and 2 worst family columns (score: the weaker of the
  two ranks on Rank IC and R² dm, so bad at either means bad). Plotting every run had become illegible.
- **`claude/compare-ts48-multiday/`** (2026-10-26): skill `/compare-ts48-multiday`
  keeping `multi-day_performance.ipynb` (repo root) up to date.
  - Columns: Run-2 (paired partner, **raw**: demeaning wasn't the author's intent),
    Run-5 (demeaned) (the reference, 3 seeds), then every run whose archived script is
    `train_multi-day-feature-sets.py`, **demeaned per day**. User decisions,
    2026-10-26.
  - After the table, a plot cell shows test Rank IC and test R² per window (demeaned;
    Run-2 raw). Its data are embedded and its figure is stored as output.
  - It imports `update_baseline_table.py` and uses `compare(demean=True)` and the new
    `grand_table(..., magnitude_note)` option.
- **Baseline table scope (user decision, 2026-10-26):** `baseline_performance.ipynb`
  lists **baseline runs only** (48 raw features: runs 3 to 6.x).
  - `is_family_run()` leaves out every run whose archived script says
    `Feature family "..."`, i.e. run 7 on. Its intro now says so.
  - Run 7 (single-day family) is no longer in any grand table; its numbers are in its
    `analysis.ipynb` and N-29.
- **`claude/compare-ts48-baseline/`** (2026-10-22): skill that keeps
  `baseline_performance.ipynb` (repo root) up to date. It holds one grand table of
  Run-2 (the author's baseline) and every later run, paired with Run-2 on common
  test rows, using the analysis skill's section-7 code. New runs are detected
  automatically and results cached per run. Linked as
  `.claude/skills/compare-ts48-baseline` (`/compare-ts48-baseline`).
- 2026-10-23: the grand table gained `Run-5 (demeaned)` and `Run-6 (demeaned)`
  columns (`DEMEANED_RUNS`), with a *How to read* bullet listing the subtracted
  per-window means. `CODE_ONLY_DIFFS` keeps 5.1/5.2 as run 5's seed variants
  after run 5's re-run with out-of-fold logging.
- **`optimize_leaf_baseline.py`** (repo root, 2026-10-23): a sweep of LightGBM
  overrides that runs **only the purged k-fold**: no retrain, no test.
  - It imports `train_baseline` and uses its `lgbm_params()`, `purged_cv()`,
    `finite_rows()` and the window settings, so the folds and parameters are
    identical.
  - It writes under `pre-runs/<sweep>/`: per configuration the curves and the
    validation predictions, plus `summary.csv` (one row per configuration and
    window) and `sweep.json`.
  - Metrics: fold-mean validation R² (raw and demeaned), Rank IC, calibration of
    the demeaned predictions, trees.
  - Interrupted sweeps resume. Resumed rows' runtime counts only the metric
    recomputation.
  - Code-only refactor of `train_baseline.py`: the parameter dict moved into
    `lgbm_params(seed)`, checked identical to run 5's archive. Run 7's setup diff
    will show it.
- **Script convention (user decision, 2026-10-24):**
  - **"Baseline" means the original 48 features per stock per day only.**
    `train_baseline.py` (run 5 + validation-prediction dump + `lgbm_params()`) and
    its sweeper `optimize_leaf_baseline.py` (as of leaf-2) stay baseline-only.
  - **Each feature family gets its own main script and sweeper** at the repo
    root. Both write into `runs/` and `pre-runs/` as before.
  - New scripts import `train_baseline` as a library. A run folder archives the
    family's script at the top level (so the analysis tools see one script) and
    the imported scripts under `archive/`.
- **`train_single-day-feature-sets.py`** (2026-10-24): the single-day family.
  - Feature sets `FEATURE_SETS` / `build_features(frame, kind)`:
    - `raw`;
    - `rank` (per-day percentile);
    - `z` (per-day z-score);
    - `raw+rank`;
    - `rank+daystats` (ranks plus each feature's daily mean and SD).
  - Transforms run on whole days before `finite_rows()`: same rows as raw,
    Float32, non-finite values give null and are dropped.
  - Own head (`FEATURE_SET`, run folder, seed, protocol). `share_head()` pushes
    the protocol settings into `train_baseline`'s helpers.
  - `train()` mirrors the baseline's. On synthetic data, `FEATURE_SET = "raw"`
    reproduces `train_baseline.train()` exactly (Δ prediction 0.0).
  - Results add `feature_set`.
- **`train_multi-day-feature-sets.py`** (2026-10-25): the multi-day family (N-32).
  - Built like the single-day script (own head, `share_head()`, archive), but the
    features are built once per window over its dates plus `WARMUP` (66) earlier
    dates, then split into history and test.
  - Groups: `labelhist-max` / `labelhist-min` (labels of dates <= t − `LABEL_LAG` =
    6: latest, means over 20 / 60 dates, SD, and max or min over 20; as-of join, at most `STALE_LIMIT`
    20 dates old), `dev` (featureN − mean over t−5 … t−1 and t−20 … t−1, at
    least half the dates present), `lags` (t−1, t−5).
  - Windows count trading dates (index `d`), not rows, so suspensions shorten a
    window instead of reaching further back.
  - Rows are the baseline's (raw features and label finite); history columns may
    be missing (LightGBM missing values). Results add `feature_set` and
    `history_missing_share`.
  - Checked: 39,000 history values match a brute-force computation on a
    real-data subset with suspended stocks; `FEATURE_SET = "raw"` reproduces
    `train_baseline.train()` on synthetic data (Δ prediction 0.0); other sets
    keep the same rows. Feature building takes < 1 s per window and adds about
    0.4 GB.
- **Multi-day groups and sweeper** (2026-10-26):
  - New groups in `train_multi-day-feature-sets.py`: `lags-K1-K2-...`, `fmean-A-B`
    (mean over t−B … t−A via `DataFrame.rolling` with an offset), `labelraw-A-B` (it
    refuses A < `LABEL_LAG`), `labelstats-max/min` (labelhist without the latest
    label), and label suffixes `-rank` (per-day percentile) / `-scaled` (label ÷ its
    day's SD before the statistics).
  - `warmup(kind)` replaces the constant `WARMUP` (only the dates a set needs).
  - `group_order()` fixes the column order as before (dev, lags, fmean, labels), so
    runs 8–11 are reproduced exactly; the old sets give identical columns.
  - 52,500 brute-force values match.
  - Code changed after runs 11.1/11.2, so further seed reruns of runs 8–11 won't
    fold into their columns automatically (the archived code differs).
  - `test_multi-day-feature-sets.py`: validation-only sweeper. It reuses
    `test_single-day-feature-sets.py`'s monitor and summary (`tsd.SWEEP_DIR` /
    `tsd.GRID` set at start). On synthetic data its out-of-fold predictions equal
    `train()`'s exactly.
  - `REUSE`: configs `raw` and `raw+lags` copy run 5's / run 11's curves and
    out-of-fold files instead of training, if the head's `RANDOM_SEED` equals the
    run's (70563); otherwise they are trained. Resume after an interruption tested.
- **`test_single-day-feature-sets.py`** (2026-10-24): the family's sweeper,
  validation only, moved from the leaf sweeper.
  - GRID entries name `"features"`; it uses the feature script's settings
    (`share_head()`).
  - `verbosity` −1 (logs only). It archives its imported scripts under
    `SWEEP_DIR/archive/`.
  - It writes a live monitor `SWEEP_DIR/curves.ipynb`, opened in VS Code:
    finished (setting, window) results per window, plus the curves in progress.
    The final plot goes to `plots/sweep_summary.png`.
  - It reuses `optimize_leaf_baseline.oof_metrics` / `finished`.
- **`claude/post-dispersion/build_notebook.py`** (2026-10-24): writes
  `RUN_DIR/post_dispersion-via-HAR.ipynb`.
  - Day table of realized dispersion and its lags; per-window auxiliary
    models (A0 trailing mean, A1 HAR, A2 + model spread, A3 + feature-SD
    summary), plus the Z and Oracle reference rows.
  - It rescales with c fitted on validation and compares test magnitudes.
  - `--findings` / `--summary` fill in the text cells.
- **`claude/post-blending/build_notebook.py`** (2026-10-24): writes
  `RUN_DIR/post_blending.ipynb`.
  - It blends a run with a partner run (both demeaned).
  - Each window's weight is chosen on its validation predictions (by best R²
    or best Rank IC) and then scored on test, plus a hindsight scan of one
    weight on test.
  - `--findings` / `--summary` fill in the text cells.
- **`claude/post-rescaling/build_notebook.py`** (2026-10-23): writes
  `RUN_DIR/post_rescaling.ipynb`.
  - It fits the out-of-fold slopes and applies them in an explicit cell, then
    compares raw, rescaled, demeaned and both on the test sets.
  - It runs the same steps for any seed variants that saved out-of-fold
    predictions.
  - `--findings` / `--summary` fill in the text cells.
  - Run it headless with the analysis skill's `run_notebook.py`.
- 2026-10-22: `build_notebook.py --leaderboard RUN ...` adds section 15, a
  side-by-side table of several runs with the best values in bold.
- 2026-10-21: comparisons now start with a **What changed** cell (setup
  differences grouped by category, with expected effects, backed by an
  automatic comment-free script diff, row and feature counts, and
  selected-feature differences) and end with a **Hypotheses** cell (likely
  improvements and degradations, what matters most and least, confounding,
  ablations to test them).
