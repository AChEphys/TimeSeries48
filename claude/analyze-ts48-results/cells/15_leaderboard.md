## Leaderboard: runs side by side

Each run's **own** test-set metrics on the same 720 test dates. In each
column (first table) or row (other tables), the best value is **bold**.

- **Rank IC, Pearson IC, Kendall:** means of the daily values over all test
  dates. Kendall is daily, unlike the training script's pooled `kt` (I-4).
- **R²:** pooled over all test rows (first table) or over each window's rows.
- **ICIR:** mean(daily Pearson IC) / std, per window, averaged over the
  windows (unscaled, comparable across windows).

Read with care:
- These are unpaired levels. The paired tests (section 7) are what decide
  significance: a gap of ~0.001 Rank IC is within noise.
- Seed noise is not measured yet.
- Every extra variant compared on the same windows inflates the best one's
  apparent edge.
