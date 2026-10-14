import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_imputation_search import expiry_features, row_type_interp
from leakage_safe_imputation_eval import power_iterative_ridge, raw_iterative_ridge_no_cap
from no_leak_model_search_v2 import make_extra, mask_observed
from robust_imputation_search import evaluate, fill_column_mean, load_data, softimpute_numpy, time_linear


OUT_DIR = Path("plots")


def column_time_model(masked, window=4, power=1.0):
    x = np.where(np.isnan(masked), np.nan, np.power(np.clip(masked, 1e-8, None), power))
    out = x.copy()
    for col in range(x.shape[1]):
        s = pd.Series(x[:, col])
        interp = s.interpolate(limit_direction="both")
        roll = interp.rolling(window, min_periods=1, center=True).median()
        out[:, col] = roll.to_numpy()
        out[~np.isnan(x[:, col]), col] = x[~np.isnan(x[:, col]), col]
    return np.power(np.clip(out, 1e-8, None), 1.0 / power)


def local_row_slope_model(masked, option_cols, option_info):
    out = masked.copy()
    for opt_type in ["CE", "PE"]:
        idx = np.array([i for i, col in enumerate(option_cols) if option_info[col]["type"] == opt_type])
        strikes = np.array([option_info[option_cols[i]]["strike"] for i in idx], dtype=float)
        order = np.argsort(strikes)
        idx = idx[order]
        strikes = strikes[order]
        for row in range(masked.shape[0]):
            vals = out[row, idx].copy()
            obs = ~np.isnan(vals)
            if obs.sum() == 0:
                continue
            missing = np.where(~obs)[0]
            for pos in missing:
                left = np.where(obs & (np.arange(len(vals)) < pos))[0]
                right = np.where(obs & (np.arange(len(vals)) > pos))[0]
                if len(left) and len(right):
                    l = left[-1]
                    r = right[0]
                    vals[pos] = vals[l] + (vals[r] - vals[l]) * (strikes[pos] - strikes[l]) / (strikes[r] - strikes[l])
                elif len(left) >= 2:
                    l1, l2 = left[-2], left[-1]
                    slope = (vals[l2] - vals[l1]) / (strikes[l2] - strikes[l1])
                    vals[pos] = vals[l2] + slope * (strikes[pos] - strikes[l2])
                elif len(right) >= 2:
                    r1, r2 = right[0], right[1]
                    slope = (vals[r2] - vals[r1]) / (strikes[r2] - strikes[r1])
                    vals[pos] = vals[r1] + slope * (strikes[pos] - strikes[r1])
                else:
                    vals[pos] = vals[obs][0]
            out[row, idx] = vals
    return np.clip(fill_column_mean(out), 1e-8, None)


def group_blend(a, b, dte, weight_expiry, weight_non):
    w = np.where(dte[:, None] < 1.0, weight_expiry, weight_non)
    return w * a + (1.0 - w) * b


def fit_candidates(masked, df, option_cols, option_info, dte, intraday):
    extra_base = make_extra(df, dte, intraday, "base")
    extra_rich = make_extra(df, dte, intraday, "rich")
    models = {}
    models["p027_a0007"] = power_iterative_ridge(masked, extra_base, power=0.27, alpha=0.0007)
    models["p05_a003"] = power_iterative_ridge(masked, extra_base, power=0.5, alpha=0.003)
    models["p07_a007"] = power_iterative_ridge(masked, extra_base, power=0.7, alpha=0.007)
    models["p1_a03"] = power_iterative_ridge(masked, extra_base, power=1.0, alpha=0.03)
    models["raw_rich_a01"] = raw_iterative_ridge_no_cap(masked, extra_rich, alpha=0.01)
    models["row_linear"] = row_type_interp(masked, option_cols, option_info)
    models["row_local_slope"] = local_row_slope_model(masked, option_cols, option_info)
    models["time_linear"] = time_linear(masked, option_cols)
    models["time_median_w3_p027"] = column_time_model(masked, window=3, power=0.27)
    models["time_median_w5_p027"] = column_time_model(masked, window=5, power=0.27)
    models["soft_rank3"] = softimpute_numpy(masked, rank=3, shrink=1e-6)

    base_names = list(models)
    for a, b in itertools.combinations(["p027_a0007", "p05_a003", "p07_a007", "p1_a03", "row_linear", "time_median_w3_p027"], 2):
        for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
            models[f"blend {a} {w:g} + {b} {1-w:g}"] = w * models[a] + (1 - w) * models[b]

    # DTE-gated blends: let expiry day use one model and non-expiry another with fixed weights.
    for expiry_model in ["p05_a003", "p07_a007", "p1_a03", "row_linear", "time_median_w3_p027"]:
        for non_model in ["p027_a0007", "p05_a003", "p07_a007"]:
            if expiry_model == non_model:
                continue
            for we in [0.25, 0.5, 0.75, 1.0]:
                for wn in [0.0, 0.25, 0.5, 0.75]:
                    models[f"dte_blend exp={expiry_model} non={non_model} we={we:g} wn={wn:g}"] = group_blend(
                        models[expiry_model], models[non_model], dte, we, wn
                    )
    return models


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, option_cols, option_info = load_data("dataset.csv")
    values = df[option_cols].to_numpy(dtype=float)
    dte, intraday = expiry_features(df, option_cols)
    seeds = [42, 123, 202, 777, 2026, 7, 99, 314, 1001, 9001]
    rows = []
    group_rows = []
    for seed in seeds:
        masked, coords, y_true = mask_observed(values, 0.10, seed)
        models = fit_candidates(masked, df, option_cols, option_info, dte, intraday)
        exp_mask = dte[coords[:, 0]] < 1.0
        for name, matrix in models.items():
            row = evaluate(name, matrix, coords, y_true)
            row["seed"] = seed
            rows.append(row)
            pred = matrix[coords[:, 0], coords[:, 1]]
            for label, mask in [("expiry_lt_1", exp_mask), ("expiry_gte_1", ~exp_mask)]:
                if mask.sum():
                    err = y_true[mask] - pred[mask]
                    group_rows.append(
                        {
                            "seed": seed,
                            "model": name,
                            "group": label,
                            "mse": float(np.mean(err**2)),
                            "mae": float(np.mean(np.abs(err))),
                            "n": int(mask.sum()),
                        }
                    )
        print(f"finished seed {seed}")

    results = pd.DataFrame(rows)
    groups = pd.DataFrame(group_rows)
    summary = (
        results.groupby("model")
        .agg(
            mean_mse=("mse", "mean"),
            median_mse=("mse", "median"),
            std_mse=("mse", "std"),
            min_mse=("mse", "min"),
            max_mse=("mse", "max"),
            mean_mae=("mae", "mean"),
            n=("mse", "count"),
        )
        .sort_values("mean_mse")
        .reset_index()
    )
    group_summary = (
        groups.groupby(["model", "group"])
        .agg(mean_mse=("mse", "mean"), median_mse=("mse", "median"), mean_mae=("mae", "mean"), mean_n=("n", "mean"))
        .reset_index()
    )
    results.to_csv(OUT_DIR / "expiry_specialist_seed_results.csv", index=False)
    groups.to_csv(OUT_DIR / "expiry_specialist_group_results.csv", index=False)
    summary.to_csv(OUT_DIR / "expiry_specialist_summary.csv", index=False)
    group_summary.to_csv(OUT_DIR / "expiry_specialist_group_summary.csv", index=False)
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
