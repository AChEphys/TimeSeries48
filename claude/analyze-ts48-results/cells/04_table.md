## 2. Per-window summary

- **ICIR** = mean(IC) / std(IC), unscaled, so it is comparable across windows.
  The training script's `ir` multiplies by √days and is really a naive
  t-statistic.
- **t (naive)** assumes independent days. **t (NW)** uses Newey–West standard
  errors with 4 lags, needed because the 5-day labels overlap. **IC ac1** is
  the lag-1 autocorrelation of daily IC.
- **train / val R²** at `best_iteration`, from the learning curves; **test
  R²** from the results. val ≫ test can mean the interleaved-validation leak
  (I-1) **or** a hard test period: check months (section 4) before blaming
  the leak.
- R² and calibration are meaningful only if predictions are in label units.
  They are not if the run trains on transformed labels (e.g. rank-Gauss).
