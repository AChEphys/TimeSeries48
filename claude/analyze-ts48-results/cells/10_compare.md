## 7. Comparison with the baseline: `__BASE_NAME__`

**Is the run better than the baseline, and where?** Market conditions move
every model's IC far more than model changes do: whole windows rise and fall
together. So the two models are compared **paired**: on the **same (Date,
Code) rows**, day by day.

- **Coverage:** only rows present in both runs are compared. If the run
  changes the universe (e.g. drops new listings), the comparison is
  restricted to the common rows; the run's own section 1–5 metrics still use
  all of its rows.
- **Daily delta** = run − baseline for Rank IC, IC and Q5−Q1 on each date.
  Its mean is the improvement, with a **Newey–West 95% CI** (deltas are
  autocorrelated too). **% days better** = share of dates where the run's
  Rank IC is higher.
- **Where:** the delta per window, per month and per ex-ante regime. A real
  improvement should be positive in **most windows**, not carried by one.
- **Agreement:** the daily rank correlation between the two models'
  predictions. Near 1 means the run barely changed what the model believes.
- **Magnitudes** (as important as ranking: the predictions feed a portfolio
  optimizer): calibration slope, R² and RMSE per window on the common rows,
  and quintile profiles, for both models.

Rules of thumb: an improvement of ~+0.005 Rank IC is meaningful **if** its CI
excludes 0 and it holds in most windows. Never choose a variant using the
final reserved window.
