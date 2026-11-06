# TimeSeries48: beginner project plan

This document summarizes the project inspection and our discussion. It is a learning roadmap, not a record of completed training. No model has been trained during this conversation.

## Goal

Understand the supplied single-day LightGBM baseline, reproduce and evaluate it, then test whether adding each stock's recent history improves prediction.

Success means a reproducible, fair experiment and a clear explanation of its results. A multi-day model does not have to outperform the baseline for the project to be a valid learning exercise.

## What is provided

| File | Purpose |
|---|---|
| [README.md](../README.md) | Exercise description: improve single-day predictions using historical features. |
| [train.py](../train.py) | Rolling LightGBM training and evaluation script. |
| [data.parquet](../data.parquet) | Table of stock/date observations, inputs, and targets (proprietary; not included in the public copy). |
| [pyproject.toml](../pyproject.toml) | Python dependencies and development settings. |

### Data inspected

| Property | Value |
|---|---|
| File size | Approximately 450 MB |
| Rows | 2,901,265 |
| Columns | 51: `Date`, `Code`, 48 features, and `label` |
| Recorded dates | 1,044, from 2022-01-04 through 2026-04-28 |
| Distinct stock codes | 2,993 A-share stocks |
| Rows per date | Mean about 2,779; median 2,827; range 126–2,882 |
| Duplicate stock/date combinations | None |
| Feature values | 32-bit floats; no null, NaN, or infinite values found |
| Missing labels | 4,652; 2,896,613 rows have usable labels |

Daily stock coverage varies; 2022-06-20 has only 709 rows and 2022-06-28 has only 126. Their cause is undocumented.

### Feature provenance and prediction target

The 48 features summarize each stock on each date using a company-owned algorithm. Their financial meanings are convoluted; treat them as opaque numerical inputs rather than assigning an economic interpretation to each column.

The project owner clarified that common-factor contributions are removed from returns when constructing the label. Our working interpretation is therefore a **five-day forward stock return with Barra common-factor contributions subtracted**, in dimensionless units. The precise construction is private and is omitted from this public copy.

Barra is a family of multifactor risk models that separates common influences, such as market, industry, size, value, and momentum, from a stock-specific residual. The model therefore learns to predict the part of the return that the common factors don't explain.

`GAP=10` covers the five-day forward window at the history/test boundary: the label of date T depends on prices up to about T+6, and ten excluded dates provide sufficient separation. Feature availability and training/validation separation still matter.

## Boosted decision trees and gradient descent

A decision tree asks threshold questions about features and outputs a number at the end of a branch. Boosting adds trees sequentially to correct the combined model's remaining errors; earlier trees normally stay fixed.

For squared-error loss, the negative gradient with respect to a prediction is the residual:

```text
residual = actual label − current combined prediction
next prediction function = current function + learning rate × new tree
final prediction = initial constant + learning rate × sum of tree outputs
```

This resembles gradient descent in function space: a tree approximates a useful correction direction. Splits are found by searching thresholds, rather than differentiating split positions. LightGBM adds regularization and other refinements. The sum is additive in tree outputs, but can be nonlinear in the original features.

## How the supplied script works

1. Read and sort the data by `Date` and `Code`.
2. Take a 240-date lookback, excluding the last 10 dates: 230 dates remain.
3. Remove rows with invalid features or labels.
4. Keep features whose absolute Pearson correlation with the label exceeds `0.02`.
5. Use every third historical date for validation: 153 fitting dates and 77 validation dates.
6. Fit a fresh CPU LightGBM model, allowing up to 2,000 boosting rounds.
7. Monitor validation RMSE and R², with early-stopping patience of 200 rounds.
8. Predict and score the next 120 dates using the selected best iteration.
9. Save the model, selected feature names, and aggregate scores; advance 120 dates and repeat.

Key settings: `gbdt`, squared-error objective, learning rate `0.01`, up to 20 leaves per tree, maximum depth 7, and regularization.

Using one-based recorded-date positions:

| Model | Training/validation history | Gap | Test |
|---|---|---|---|
| 1 | 1–230 | 231–240 | 241–360 |
| 2 | 121–350 | 351–360 | 361–480 |
| 3 | 241–470 | 471–480 | 481–600 |
| 7 | 721–950 | 951–960 | 961–1044 |

There are seven independent models: six full test blocks and a final 84-date block. They are not automatically merged. Each can predict compatible rows, but its assigned test block specifies where it is evaluated. Stitching those test predictions together evaluates the rolling procedure; it does not merge the models.

The baseline learns from many historical dates, but each prediction explicitly receives only one row's features. Adding historical rows as inputs is the challenge.

## Stages and checkpoints

| Stage | Actions | Checkpoint |
|---|---|---|
| 1. Understand | Follow one window through the script. Ask about unfamiliar calls as they arise. Track variable contents, array shapes, and date boundaries. | Explain how a stock/date row becomes a prediction and which labels each operation sees. |
| 2. Make experiments trustworthy | Review actual errors separately from style warnings. Fix reproducibility and reporting issues. Record feature provenance and the factor-adjusted target hypothesis. Choose and record the evaluation protocol. | Repeatable settings, explicit splits, and a documented list of unresolved assumptions. |
| 3. Establish the baseline | Run one full window first to check behavior, runtime, and memory; then run the full evaluation. Preserve scores, learning curves, and predictions. | A reproducible reference result with understood metrics and resource costs. |
| 4. Explore modestly | Try about four to six hyperparameter configurations on development periods. Change one setting at a time initially. | Explain which changes help, hurt, or make little difference. |
| 5. Add history | Keep LightGBM and compare two-date, five-date, and compact historical-summary inputs. | Fair comparisons on the same dates and eligible stock rows. |
| 6. Finish evaluation | Freeze the chosen approach, evaluate a reserved final period, and write a short report with plots. | Reproducible conclusions, limitations, and a clear account of whether history helped. |

### Stage 2: specific checks

- Use fixed seeds instead of the script's current time-based seeds.
- Correct the reported training end date: it currently names the first excluded gap date.
- Give experiments separate output locations; the existing results filename is overwritten on another run.
- For independent validation, select features using training labels only. The supplied script uses labels from both fitting and validation dates.
- Consider a later chronological validation block instead of interleaved dates, with separation that prevents forward label windows from crossing the fitting/validation boundary. The existing ten-date history/test gap covers the supplied target's lookahead.
- Keep the supplied result as a reference and record methodological changes. Compare new inputs under the same revised protocol.
- Reserve the last 84-date block before repeated tuning. Use earlier periods for development; do not choose configurations using the reserved block.

### Stage 3: monitor and visualize

During training, watch training versus validation RMSE, validation R², best iteration, stopping iteration, selected features, row counts, elapsed time, memory, and warnings. Training improving while validation worsens suggests overfitting. Reaching 2,000 rounds is not itself a success or failure.

Later, preserve per-row predictions with `Date`, `Code`, and `label`, plus learning histories. The current script saves aggregate scores and models, but not these analysis records.

Useful plots:

| Plot | Question |
|---|---|
| Training and validation RMSE versus iteration | When does additional learning stop helping? |
| Daily IC and a rolling average | Is usefulness persistent or concentrated in one period? |
| Scores by rolling window | Does performance vary across time? |
| Sampled/binned prediction versus actual label | Are numerical predictions calibrated? |
| Mean actual label in five predicted-score groups | Do higher predictions identify higher outcomes? |
| Feature importance across windows | Does the model's reliance on inputs change? |

Feature importance is diagnostic, not proof of causation or intrinsic usefulness. Correlated inputs can substitute for each other. Cumulative IC is not cumulative investment profit.

### Stage 4: bounded hyperparameter exploration

Start with tree size (`num_leaves`), minimum rows per leaf (`min_data_in_leaf`), and modest learning-rate changes with early stopping. Maintain a table of settings, scores, runtime, and interpretation. Avoid a large automated search initially.

Move to history once you can reproduce and explain the baseline and recognize overfitting. Full baseline optimization is not a prerequisite.

### Stage 5: recommended experiments

| Experiment | Inputs |
|---|---|
| A: reference | Current date's features |
| B: two dates | Current plus previous market date |
| C: five dates | Current plus four previous market dates; up to 240 inputs |
| D: summaries | Current features plus five-date averages and changes |

Align history by stock and date, decide how to handle missing observations, and use only information available at prediction time. Keep evaluation rows consistent between variants.

Let the trees discover useful lags first. Scaling an older feature column by a small positive constant generally does not reduce its importance to a tree: thresholds can rescale. Weighted averages genuinely change the historical information supplied.

Averaging current and past single-day predictions is a useful separate smoothing experiment; it may reduce noise while reacting more slowly. Every historical prediction must come from a model that was available at that time.

If history helps, remove individual lags or summaries to discover which additions matter—an **ablation experiment**. A regularized linear model is a reasonable later comparison. There is no guaranteed better replacement for BDT; postpone neural networks until these simpler experiments are understood.

## Evaluation: especially IC

| Score | Meaning |
|---|---|
| RMSE | Numerical prediction error in label units. |
| R² | Numerical fit relative to a constant-mean reference. |
| Kendall's tau | Agreement between predicted and actual ordering. |
| Daily IC | Pearson correlation between predictions and actual labels across stocks on one date. |
| Reported `ic` | Average of finite daily IC values. |
| Reported `ir` | Mean daily IC divided by its standard deviation, multiplied by the square root of the test-date count; this is the script's convention, not trading-return IR. |

Pearson IC uses magnitudes as well as ordering.

Toy example in arbitrary units:

| Stock | Actual label | Single-day prediction | Multi-day prediction |
|---|---:|---:|---:|
| A | 2 | 1 | 2 |
| B | 1 | 2 | 1 |
| C | 4 | 3 | 3 |
| D | 3 | 4 | 4 |

Single-day IC is `0.6`; multi-day IC is `0.8`; improvement that date is `+0.2`. Evaluate these paired daily differences across many dates and development periods. A few exceptional dates should not determine the conclusion.

Predictions `[20, 10, 40, 30]` against labels `[2, 1, 4, 3]` have IC `1` despite being ten times too large. IC therefore does not replace numerical-error metrics.

Use mean daily IC as the main comparison of whether higher predictions accompany higher realized factor-adjusted forward-return labels, with RMSE and R² assessing numerical accuracy alongside it. RMSE is in the same dimensionless units as the label. Freeze the chosen approach before evaluating the reserved final period.

## Practical scope on the MacBook M4 Pro

The 14-core CPU and 24 GB RAM should be sufficient for the baseline. LightGBM's documented GPU implementation does not support macOS, so the 20 GPU cores do not accelerate this script.

The earlier rough planning estimate was **3–20 minutes per full window, budgeting about 10 minutes**, and approximately **1–2 hours for seven windows**. These are unbenchmarked estimates, not guarantees. Time one window before scheduling a full experiment series.

All original feature columns occupy about 0.56 GB as raw 32-bit numbers. Materializing every row with five, ten, or twenty dates of features would require about 2.8, 5.6, or 11.1 GB respectively, before copies and training overhead. Start with five dates; build expanded data window by window and consider compact summaries.

## Completion checklist

- [ ] Explain the script and the gradient-boosting process.
- [ ] Explain the factor-adjusted forward-return target and document that the proprietary features are treated as opaque numerical inputs.
- [ ] Reproduce a single-day baseline with explicit settings and splits.
- [ ] Build historical inputs without crossing stocks or using future information.
- [ ] Compare two or three multi-day variants fairly.
- [ ] Evaluate across time and on a reserved final period.
- [ ] Save configurations, scores, models, and predictions needed to reproduce the work.
- [ ] Produce a short report with a few plots, resource costs, conclusions, and limitations.

A supported conclusion that history did not help is a valid outcome. Financial usefulness or profit cannot be established from prediction scores alone; the supplied script does not simulate trading or costs.

## Reference documentation

- [Gradient boosting and its mathematical formulation](https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation)
- [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)
- [LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)
- [Data leakage and reproducibility](https://scikit-learn.org/stable/common_pitfalls.html)
- [Time-series evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [LightGBM installation and macOS support](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)
- [Barra factor contributions and stock-specific returns (MSCI, Appendix H)](https://www.msci.com/documents/10199/f1af8a2f-70ad-4c7e-96b8-a6ed39f1ee8f)
