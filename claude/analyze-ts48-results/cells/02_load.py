results, preds, curves = load_run(RUN_DIR)
n_windows = len(results)
assert preds["window"].n_unique() == n_windows, "predictions missing for some windows"
print(f"RUN {RUN_DIR.name}: {n_windows} windows, {preds.height:,} test rows, "
      f"{preds['Date'].n_unique()} test dates ({preds['Date'].min()} -> {preds['Date'].max()})")
if len(curves) != n_windows:
    print(f"  note: learning curves found for {len(curves)} of {n_windows} windows")

if COMPARE:
    b_results, b_preds, b_curves = load_run(BASELINE_DIR)
    print(f"BASELINE {BASELINE_DIR.name}: {len(b_results)} windows, {b_preds.height:,} rows, "
          f"{b_preds['Date'].min()} -> {b_preds['Date'].max()}")
    run_windows = {(r["test_start"], r["test_end"]) for r in results}
    base_windows = {(r["test_start"], r["test_end"]) for r in b_results}
    if run_windows != base_windows:
        print("  WARNING: test windows differ between run and baseline; "
              "the comparison uses only their common (Date, Code) rows.")
