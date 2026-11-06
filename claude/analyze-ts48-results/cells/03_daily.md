## 1. Daily metrics

Every metric is computed **within each test date** (stocks vs stocks on the
same day), then summarised per window.

- **IC:** Pearson correlation of prediction and label. Uses magnitudes, so
  it is sensitive to extreme stocks.
- **Rank IC:** Spearman correlation, i.e. ordering only. The robust ranking
  metric (magnitudes are judged separately, by calibration and R²).
- **Kendall:** pairwise ordering agreement; numerically ≈ 0.68 × Rank IC.
- **Q5−Q1:** mean label of the top prediction quintile minus the bottom
  quintile, in label units (the label is roughly a 5-day forward residual
  return; no further financial meaning is attached). Gross, not a profit.
