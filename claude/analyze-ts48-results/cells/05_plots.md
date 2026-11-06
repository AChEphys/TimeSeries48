## 3. Plots

- **IC over time:** daily values (grey) and a 20-day rolling mean per window
  (colour).
- **Cumulative IC:** the slope is the IC level. Flat stretches are periods
  where the signal stopped working. Not a profit curve.
- **Per-window metrics and R²:** train / val / test R² side by side.
- **Quintiles:** mean label per prediction quintile. In the baseline the edge
  sits in Q1: volatile, lottery-like stocks drift down, while big winners are
  news-driven and unpredictable.
- **Calibration:** 20 prediction bins per window, (mean prediction, mean
  label). On y = x means calibrated; flatter means overconfident.
