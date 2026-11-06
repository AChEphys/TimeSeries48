## 5. Ex-ante regimes: does performance depend on observable conditions?

**What a regime split is.** Each test day is put into a group ("regime")
according to the market conditions on that day, and performance is compared
across groups.

**Why ex-ante (known in advance).** A regime split is only useful if you could
have *acted* on it: trust or size the signal more in some conditions, add the
condition as a model feature, or check whether a model variant (e.g. history
features) helps mainly in certain conditions. So each day's regime is built
**only from information available at the close of day T**. Labelling months
after the fact ("that was a crash month") can explain the past, but nobody
could have traded on it, and it invites cherry-picking. A regime split does
**not** try to foresee shocks. Finding that the regime *can't* flag the bad
periods is itself a useful (negative) result.

**How the regime is built.**
1. *Dispersion* = the cross-sectional std of the label on a date: how
   differently stocks moved. Labels are factor-adjusted, so this is
   stock-specific dispersion, **not** market volatility.
2. *Trailing dispersion* at T = the mean dispersion over dates T−26 … T−7.
   Those labels are fully realized by day T, so there is no look-ahead.
3. *Percentile* = where today's trailing dispersion ranks among the previous
   ~250 trading days (one year, at least 120 needed). Below the 33rd
   percentile is **calm**, above the 67th is **turbulent**, and in between is
   **normal**. Relative to the past year, so the thresholds adapt over time.

**How to read the charts.**
- *Timeline* (three rows sharing the date axis):
  1. Daily dispersion (thin) and the trailing measure (bold). Dotted lines
     mark the largest dispersion spikes. Note how the trailing measure reacts
     only after a spike: shock onsets look calm ex-ante.
  2. The percentile and the 1/3 and 2/3 thresholds. Background shading =
     regime (light = calm, dark = turbulent), the same in all rows.
  3. Daily Rank IC (faint) and its 20-day mean (bold), against the regime
     shading.
- *Summary*: left, mean daily Rank IC per regime, with Newey–West 95%
  confidence intervals and the number of days and of distinct **episodes**
  (separate runs of ≥ 5 days). Few episodes means weak evidence, however many
  days they cover. Right, the same per window: if the effect is real it
  should appear in most windows, not just one.

**In the comparison (section 6)** the same regimes are used to check *where*
a variant helps: e.g. whether history features mainly help after dispersion
spikes.
