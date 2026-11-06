# TimeSeries48: Predict factor-adjusted stock returns

This is an exercise about quantitative research. I worked out the exercise from the task stated below to its final status with the assistance of Claude and Codex. Claude (Claude Code) went through the code with me, built most of the tools that I used to analyze the training runs, and kept the project notes. Codex inspected the project at the beginning and wrote a learning plan for me. I ran all the training jobs and decided what to try next.

This repository contains the code, the results and analyses of all the training runs, and the notes that record how my understanding of the problem developed. The data and the trained models are not included, because the data are proprietary. The scripts therefore cannot be run without the data file, but all the results, plots and notes are kept as they were produced.

## The task

Consider a stock market trading across an extensive calendar period. There are 48 features (real numbers) summarized for each stock per trading day. This project aims to train a model to predict the stocks' return in the near future (provided as a dimensionless label) based on the 48 features.

A baseline approach is provided. The baseline trains **boosted decision tree (BDT)** models that take in the 48 features of the stocks within a *single* day to predict the label (return in the imminent future) of those stocks. See [`train.py`](./example/train.py) for reference.

First, establish the baseline by training the described single-day models. Evaluate your baseline models. How well do they predict the label in general? Visualize the evaluations.

Next, try to improve the models' prediction ability by adding the stock's historical features to the input. For example, instead of using merely the features of a single day, *aggregate the feature history* of the past week. Can the additional information give a better prediction? Evaluate your multi-day model(s). Compare with the baseline on later dates (not used for training) under the same evaluation procedure.

## Data and prediction target

The data file `data.parquet` (not included in this repository) contains 2,901,265 stock/date observations of A-share stocks across 1,044 recorded trading dates, from 2022-01-04 through 2026-04-28.

| Columns | Contents |
|---|---|
| `Date`, `Code` | Observation date and stock identifier. |
| `feature0`–`feature47` | 48 numerical features per stock per date with inexplicit financial meanings. |
| `label` | The prediction target. Dimensionless, factor-adjusted 5-day forward return. |

The target represents a stock's return over the next five trading days, from which the contributions of the common risk factors in a Barra risk factor model have been subtracted. In other words, the label is the residual of the Barra model, and the aim is to predict this remaining stock-specific component. The precise construction of the label is private and is not part of this repository.

## Example training script

[`train.py`](./example/train.py) uses **Light Gradient Boosting Machine (LightGBM)** to train gradient-boosted decision trees. Each prediction uses a single date's selected features.

The script trains and tests model within a forward-rolling window of dates. Each rolling window uses `(LOOKBACK - GAP)` historical dates for fitting and validation against overfitting. After a purge gap, it evaluates the next `TEST_DAYS` dates. An individual BDT model is trained for each rolling window, resulting in several consecutive models representing different time ranges. By default, training allows up to 2,000 boosting rounds.

The script saves models and selected feature lists in `models/`, and evaluation summaries in `.json` format.

## Training requirements

- `Python 3.13` or later.
- `LightGBM`, `NumPy`, `Polars`, `SciPy`, and `scikit-learn`, as declared in [`pyproject.toml`](./pyproject.toml).
- By default, models are trained using `LightGBM` on CPU. For Nvidia GPU training, consider using `XGBoost`.
- The data file `data.parquet` weighs about 500 MB.

## Expected result

Deliver a reproducible single-day baseline and multi-day comparison(s), with parameter settings and scores across time. Visualize the training process: how fast does your model converge? Are the scores perfect? What signals overfitting? Pay attention to *target leakage*: use only information available at prediction time. Reserve a final period for evaluation.

Compare mean daily **information coefficient (IC)**, numerical prediction accuracy, and runtime for the same evaluation period. Report whether historical features improve prediction; a negative result provided with probable grounds is also valid.

## Result

I compared three models on the same test dates and stocks. Run-2 is the author's baseline, which I trained exactly as it was provided. Run-5 uses the same single-day inputs as Run-2, but it was trained with a validation procedure that I corrected during the project (see chapter 3 of the notes). Run-22 is my final model. In addition to the 48 features of the prediction date, it takes the features of the previous date and of the date five days earlier, as well as the stock's own labels from six to ten dates earlier. Those labels are already known on the prediction date, and I scaled them before using them as inputs.

The main measure is the mean daily Rank IC. On each test date, it is the rank correlation between the predictions and the labels across all stocks, and these daily values are then averaged over the test dates.

| | Run-2 (author's baseline) | Run-5 (corrected validation) | Run-22 (with history) |
|---|---|---|---|
| Inputs | 33 to 40 of the 48 features, chosen by their correlation with the label | all 48 features | 149 inputs: the 48 features of dates T, T−1 and T−5, and the scaled labels of T−6 to T−10 |
| Mean daily Rank IC, all 7 windows (804 dates) | 0.0592 | 0.0635 | **0.0730** |
| $R^2$, all 7 windows | 0.0074 | 0.0090 | **0.0101** |
| Rank IC of Run-22 minus this model (Newey–West t) | +0.0138 (t = 6.4) | +0.0095 (t = 4.5) | |

Adding the recent history of each stock improved the ranking in six of the seven test windows and on 65% of the test dates, and it also improved $R^2$. I had reserved the last window for a single final evaluation, and I did not look at it until I had chosen the candidates and written down the rule that would decide the verdict. In that window, the improvement over Run-5 was consistent with the earlier windows (+0.0020, t = 1.0), while Run-22 and Run-2 were tied. The final window happened to be unusually good for both baselines, so more recent data would be needed to tell whether the improvement holds in other market periods. The full tables are in [all-windows_performances.ipynb](all-windows_performances.ipynb), and [ts48.pdf](ts48.pdf) is the final report.

## How the project went

The full project history, with all the numbers, is in the project notes ([claude/project-notes.md](claude/project-notes.md)), which are arranged in nine chapters. In short:

1. **Understanding the task and the starter code.** I first went through the example script with Claude to understand what each part does. Some of the LightGBM parameters turned out to have no effect, and the random seeds were taken from the clock. I fixed the seeds and made every run save its predictions and learning curves in a folder of its own. I also found that the label covers about five trading days after the prediction date, which became important later.
2. **Measuring the baseline (runs 1 and 2).** I trained the author's baseline and learned how to assess it. The Rank IC is computed across the stocks of each date, and its significance has to be judged with Newey–West t-statistics, because the labels of neighboring dates overlap. Most of the baseline's skill came from identifying the stocks that would do worst. Since the predictions are to be fed into a portfolio optimizer, the magnitudes of the predictions matter as well as their ranking.
3. **Making validation honest (runs 3 to 5).** The example script uses every third date for validation. Because the label of one date overlaps with the labels of the next few dates, information leaked from the training dates into the validation dates. I replaced this split with a purged five-fold cross-validation, which leaves gaps around each validation block. The predicted magnitudes became more stable, and Run-5, which was trained this way, became the new reference.
4. **Tuning the trees and the outputs (run 6 and several sweeps).** Changing the settings of LightGBM did not help, because the original settings were already close to the best ones. However, subtracting the mean prediction of each day improved $R^2$ by about 7% without changing the ranking at all.
5. **Single-day feature engineering (run 7 and a sweep).** Before adding history, I tried to transform the features of the same date, for example by ranking them across stocks. These features looked better on the validation sets, but the gains did not carry over to the test windows. I recorded this as a negative result.
6. **Adding each stock's history (runs 8 to 16 and a sweep).** The features of earlier dates improved the forecast in all six development windows. The most recent dates helped the most, in particular the previous date and the date five days earlier, while older information did not add anything.
7. **Past labels, and one model for both (runs 17 to 23).** A stock's own past labels can be used as inputs, as long as they are already known on the prediction date. This means that they must be at least six dates old. The labels from six to ten dates earlier helped most, especially after scaling, and putting them together with the lagged features in a single model gave the best result (Run-22).
8. **Ideas that didn't help (runs 24 to 30).** Sampling the columns, giving more weight to recent dates, and adding models that correct the errors of the main model did not improve the forecast. In some of these cases, the validation sets even pointed in the wrong direction.
9. **The final evaluation.** I chose the candidates and wrote down the rule for the verdict before training them on the reserved final window. Run-22 was consistent with its earlier improvement over Run-5 and tied with Run-2.

Throughout the project, Claude and I kept a few working rules. Each run changed only one thing compared with its reference. The random seeds were fixed, and for the important setups I trained two or three seeds to see how much of a difference was only noise. Two models were always compared on the same dates and stocks. We judged both how well the predictions rank the stocks and whether their magnitudes are right. The last window was left untouched until the candidates and the decision rule had been fixed. The lines that Claude wrote in my scripts are marked with `# BY: Claude`.

## What I learned

**Evaluating a forecast of returns.** Because the model predicts all the stocks of a date at once, its forecast should be scored across the stocks of each date, and the daily scores are then averaged over the dates. I used the Rank IC as the main measure, since it is less sensitive to a few extreme stocks than the Pearson IC. The gap between the two shows how much the extreme stocks contribute.

The daily scores are autocorrelated, because the labels of neighboring dates overlap. A naive t-statistic therefore overstates how significant a result is. Newey–West standard errors correct for this, and comparing two models through their paired daily differences removes much of the noise that both models share.

Ranking the stocks well is not enough when the predictions go into a portfolio optimizer. The $R^2$, the calibration slope and the RMSE can be poor even when the IC looks fine. Part of this can be fixed cheaply after training, for example by subtracting each day's mean prediction. Before trusting a small improvement, I also need to know how much the result changes when only the random seed changes.

When assessing a forecast from our trained models, we should look at their performances within specific windows, months, or regimes that we could identify in advance. More importantly, every period should be equally considered in the assessment. We should not exclude a period simply because of an unusual movement of the market in that period.

**Avoiding leakage, and not fooling myself.** A label that looks into the future leaks into the validation when the validation dates are interleaved with the training dates. A chronological split or a purged k-fold cross-validation, which leaves gaps between the training and the validation dates, avoids this, and the model can be retrained on all of the history afterwards. In the same way, a label is known only after its horizon has ended, so any input that is built from past labels has to be lagged. In this project, the lag is six dates.

If many variants are tried on the same test windows, the best of them looks better than it really is. That is why I reserved a final window and wrote down the candidates and the rule for the verdict before I looked at it. I also learned that the validation sets can favor some kinds of changes, such as features that describe the market regime or weights that favor recent dates, so such changes have to be confirmed on the test windows.

**Modeling.** I learned what the main settings of LightGBM do on a noisy target: the number of leaves, the minimum number of rows in a leaf, GOSS sampling compared with bagging, and the L1 and L2 regularization. I learned how to build features for panel data, in which every date contains many stocks. Such features include ranks and statistics across the stocks of each date, lagged values, deviations from rolling means, and the history of past labels. Lagged values have to be aligned by stock and by trading date, because some stocks are suspended or newly listed. I also compared three ways of combining information: one model that takes all the inputs, an average of separate models, and models that correct the errors of a main model.

**Research practice and tools.** Every experiment has a run folder of its own, in which the script that produced it is archived. I wrote down my hypotheses before seeing the results, and I recorded the negative results together with their probable reasons. On the technical side, I used Python with Polars, NumPy and LightGBM, generated Jupyter notebooks from scripts, and managed the project with `uv` and `ruff`. I also learned to work with AI assistants, Claude Code and Codex, while keeping the decisions in my own hands.

## Repository map

| Path | Contents |
|---|---|
| [example/train.py](example/train.py) | The author's starter script, which produced the baseline (Run-2) |
| [train_baseline.py](train_baseline.py) | The baseline trainer with the corrected validation (purged five-fold cross-validation, retraining, and records for each run) |
| [train_single-day-feature-sets.py](train_single-day-feature-sets.py), [train_multi-day-feature-sets.py](train_multi-day-feature-sets.py) | The trainers for the two families of features: transformations of the same date's features, and each stock's history |
| [optimize_leaf_baseline.py](optimize_leaf_baseline.py), [test_single-day-feature-sets.py](test_single-day-feature-sets.py), [test_multi-day-feature-sets.py](test_multi-day-feature-sets.py) | Sweeps that use only the validation sets (results in [pre-runs/](pre-runs/)) |
| [train_final-window.py](train_final-window.py) | Trains the candidates once on the reserved final window |
| [runs/](runs/) | One folder for each run, with the archived script, `results_lgbm.json`, the learning curves, the plots, and `analysis.ipynb` with the findings |
| [baseline_performance.ipynb](baseline_performance.ipynb), [multi-day_performance.ipynb](multi-day_performance.ipynb) | Overview tables of all the runs, each compared with the baseline |
| [final-evaluation_performances.ipynb](final-evaluation_performances.ipynb) | The evaluation on the final window and its verdict |
| [all-windows_performances.ipynb](all-windows_performances.ipynb), [ts48.pdf](ts48.pdf) | The final report with all seven windows in one table |
| [claude/project-notes.md](claude/project-notes.md) | The project notes |
| [claude/](claude/) | The analysis skills and notebook builders that Claude wrote (linked in `.claude/skills/`) |
| [codex/project-plan.md](codex/project-plan.md) | The initial plan that Codex wrote |

## Running the code

The scripts expect the data file, which is not included, as `data.parquet` in the root of the repository. With the data in place, the following commands install the dependencies and train the baseline:

```bash
uv sync
uv run python train_baseline.py
```

The settings of each script, such as the run folder, the seed and the windows, are at the top of the script. Each run writes its output to `runs/<run name>/`. The analysis notebooks are built by the scripts in [claude/](claude/), which are described in the `SKILL.md` files.
