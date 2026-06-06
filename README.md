**#read only final_model_p1rich03_surface2d10.ipynb
**
## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final model:

```bash
python run_final_model.py
```

This creates:

```text
outputs/filled_blend_p1rich03_90_surface2d10.csv
outputs/submission_blend_p1rich03_90_surface2d10.csv
```

The notebook version is:

```text
final_model_p1rich03_surface2d10.ipynb
```


# Expiry-Aware IV Imputation for NIFTY Options

This project fills missing implied-volatility (IV) values in a NIFTY options dataset. The final solution combines a stable multivariate iterative imputer with a small local time-strike surface correction.

Best recorded actual MSE for the selected submission:

```text
0.0000658973
```

Final model:

```text
0.90 * PowerMICE(power=1.0, alpha=0.03, rich_features)
+ 0.10 * Local 2D IV Surface Smoother
```

## Problem

The dataset contains timestamped NIFTY option IV values with missing cells across call and put strikes. The objective is to predict the missing IV values as accurately as possible.

The main file is:

```text
dataset.csv
```

It contains:

- `datetime`
- `underlying_price`
- 28 option IV columns for `NIFTY27JAN26`

## Key Findings

### 1. Random masking was too optimistic

Uniform random masking gave very low validation errors, but those scores did not transfer well to the actual missing cells. The better validation proxy was native-pattern masking: for each row, mask the same number of observed cells as were originally missing in that row.

### 2. Expiry day mattered

I split the data into:

- `DTE < 1`
- `DTE >= 1`

The IV versus log-moneyness curve showed that expiry-day rows behave very differently. The non-expiry region was much easier to impute, while expiry-day rows had steeper local IV behavior and dominated the harder errors.

### 3. PowerMICE was the best backbone

PowerMICE is a transformed MICE-style iterative ridge imputer:

- optionally transform IV as `IV^power`
- iteratively predict each option column from the other option columns
- include rich row-level features such as underlying price, days-to-expiry, intraday time, expiry flag, and underlying-price movement
- apply ridge regularization through `alpha`

The best actual-performing backbone used raw IV scale:

```text
power = 1.0
alpha = 0.03
features = rich
```

### 4. Local 2D smoothing helped as a small correction

The local smoother fits nearby values in both time and strike dimensions, separately for CE and PE. It uses tighter time locality on expiry day.

The smoother alone was unstable, but a small 10% contribution improved the final submission.

## Repository Structure

Recommended core files:

```text
dataset.csv
final_model_p1rich03_surface2d10.ipynb
run_final_model.py
robust_imputation_search.py
advanced_imputation_search.py
leakage_safe_imputation_eval.py
no_leak_model_search_v2.py
surface2d_search.py
requirements.txt
README.md
```


```


It includes:

- loading and preprocessing
- missingness EDA
- IV versus log-moneyness graph
- expiry-day analysis
- explanation of PowerMICE
- local 2D smoothing
- final blend
- filled CSV generation
- submission CSV conversion

## Final Output Format

The submission format uses one row per originally missing value:

```text
id,value
datetime||column_name,predicted_iv
```

Example ID:

```text
07-01-2026 09:15||NIFTY27JAN2625500CE
```




