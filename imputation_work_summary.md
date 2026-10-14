# IV Imputation Work Summary

Last updated: 2026-06-05

## Goal

Predict/fill missing implied-volatility (IV) values in `dataset.csv` while minimizing MSE on the actual hidden missing cells.

## Data Understanding

- Dataset has 975 rows and 30 columns.
- Key non-IV columns: `datetime`, `underlying_price`.
- There are 28 option IV columns: CE and PE strikes for `NIFTY27JAN26`.
- Native missing IV cells: 5,460, about 20%.
- Expiry date parsed from option names: 2026-01-27.
- DTE split mattered diagnostically:
  - `DTE >= 1` day is much easier; best proxy MSE was around `0.000005`.
  - `DTE < 1` day is much harder; errors there dominate overall MSE.

## Important Fix

The original strike parser was wrong. It parsed names like `NIFTY27JAN2625200CE` as strike `2625200` instead of `25200`.

This was fixed in:

- `robust_imputation_search.py`
- `impute_analysis.py`

Correct strike parsing improved feature engineering and moneyness-based modeling.

## Validation Lessons

Uniform random 10% masking was too optimistic and did not match the hidden actual-missing score well.

Better proxy:

- Native-row-pattern masking: for each row, mask the same number of observed cells as are natively missing in that row.
- This matched actual hidden scores much better.

Avoided leakage:

- The earlier `Column stacked ensemble` was validation-leaky because it learned stack weights from the same masked cells used for scoring.
- That file should not be trusted for final scoring: `filled_dataset_refined_best.csv`.

## Best Actual Scores Reported

- `PowerMICE(p=0.27, alpha=0.007)` actual MSE: `0.0000913096`.
- `Power(p=1, alpha=0.03, rich features)` actual MSE: `0.0000675136`.

## Current Best Candidate

Current selected best filled CSV:

- `filled_best_model_current.csv`

This is a copy of:

- `filled_meta_column_alpha10.csv`

Reason:

- It had the best no-leak native-pattern proxy among the latest candidates.
- Proxy MSE was about `0.0000453`.
- It was trained using pseudo-masked observed cells, not actual hidden missing labels.

## Latest Candidate Files To Check

Recommended order:

1. `filled_meta_column_alpha10.csv`
2. `filled_meta_group_alpha0p1.csv`
3. `filled_meta_global_alpha1.csv`
4. `filled_adaptive_row_p1rich03.csv`

Previous strong candidates:

1. `filled_native_proxy_p1_rich_alpha003.csv`
2. `filled_native_proxy_blend_p1rich03_95_row_05.csv`
3. `filled_native_proxy_blend_p1rich03_90_p027base_10.csv`
4. `filled_native_proxy_p1_rich_alpha002.csv`
5. `filled_native_proxy_blend_p1rich02_55_p085rich02_45.csv`

## Key Scripts

- `native_meta_stack_search.py`: no-leak out-of-fold meta-stack evaluation.
- `fill_meta_stack_candidates.py`: creates latest meta-stack candidate CSVs.
- `native_model_search_v3.py`: native-pattern search with residual/surface/KNN ideas.
- `fill_native_pattern_candidates.py`: creates native-pattern proxy candidate CSVs.
- `no_leak_model_search_v2.py`: repeated no-leak masked search.
- `leakage_safe_imputation_eval.py`: leakage-safe evaluation scaffold.

## Metric Files

- `plots/native_meta_stack_oof_metrics.csv`
- `plots/native_v3_summary.csv`
- `plots/native_pattern_high_power_grid_summary.csv`
- `plots/native_pattern_blend_rich_summary.csv`
- `plots/native_row_pattern_summary.csv`

## Next Update Protocol

Whenever a candidate CSV is checked against actual hidden missing data:

1. Add the actual MSE here.
2. Mark whether it beat the current best.
3. If it wins, copy/rename it to `filled_best_model_current.csv`.
4. Continue searching around the winning family.
