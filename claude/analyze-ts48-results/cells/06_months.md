## 4. Finer grain: months and robust summaries

A weak window can mean broad weakness, or a few bad stretches inside an
otherwise good period. This section separates the two.

**How to read it.** A single month (~20 test days, with strongly
autocorrelated daily ICs) is worth only ~5 independent days. One bad month
proves little; look for **patterns**: months that are bad across several
models, or runs of weak months. Headline metrics stay full-period. Months are
never excluded after the fact (that would be cherry-picking).

- **Top panel:** monthly mean daily **Rank IC** (bars, colour = window) and
  **Pearson IC** (black dots). Dots far above their bar mark months where the
  signal worked only through a few extreme stocks.
- **Bottom panel:** monthly **calibration slope**, the regression slope of
  label on prediction. 1 means magnitudes are right; below 1 means
  overconfident (predictions too spread out); below 0.5 (dotted) means
  strongly overconfident.
- **Robust table:** median daily IC and a 10%-trimmed mean. If they are close
  to the mean, the window's result isn't driven by a few extreme days.
