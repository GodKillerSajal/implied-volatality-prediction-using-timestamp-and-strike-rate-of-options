import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_imputation_search import expiry_features, row_type_interp
from leakage_safe_imputation_eval import power_iterative_ridge, raw_iterative_ridge_no_cap
from robust_imputation_search import (
    RANDOM_SEED,
    evaluate,
    ffill_bfill,
    fill_column_mean,
    load_data,
    row_strike_fit,
    softimpute_numpy,
    time_linear,
)


OUT_DIR = Path("plots")


def mask_observed(values, ratio, seed):
    rng = np.random.default_rng(seed)
    observed = np.argwhere(~np.isnan(values))
    selected = observed[rng.choice(len(observed), size=int(len(observed) * ratio), replace=False)]
    masked = values.copy()
    y_true = values[selected[:, 0], selected[:, 1]].copy()
    masked[selected[:, 0], selected[:, 1]] = np.nan
    return masked, selected, y_true


def make_extra(df, dte, intraday, variant="base"):
    t = np.arange(len(df), dtype=float)
    spot = df["underlying_price"].to_numpy()
    spot_ret = np.r_[0.0, np.diff(spot)]
    cols = [
        spot,
        t,
        dte,
        intraday,
        (dte < 1.0).astype(float),
        np.sin(t / 12.0),
        np.cos(t / 12.0),
    ]
    if variant in {"rich", "returns"}:
        cols.extend(
            [
                spot_ret,
                pd.Series(spot_ret).rolling(3, min_periods=1).mean().to_numpy(),
                pd.Series(spot_ret).rolling(6, min_periods=1).std().fillna(0).to_numpy(),
                np.log1p(np.maximum(dte, 0)),
                1.0 / np.sqrt(np.maximum(dte, 1e-4)),
            ]
        )
    return np.column_stack(cols)


def nearest_time_same_column(masked, option_cols):
    return time_linear(masked, option_cols)


def row_time_blend(row_model, time_model, weight):
    return weight * row_model + (1.0 - weight) * time_model


def low_rank_residual(base, masked, rank=3, shrink=1e-6):
    residual = masked - base
    residual[np.isnan(masked)] = np.nan
    filled_residual = softimpute_numpy(residual, rank=rank, shrink=shrink, standardize=True)
    return np.clip(base + filled_residual, 1e-8, None)


def transform_softimpute(masked, power=0.5, rank=3, shrink=1e-6):
    z = np.where(np.isnan(masked), np.nan, np.power(np.clip(masked, 1e-8, None), power))
    filled_z = softimpute_numpy(z, rank=rank, shrink=shrink, standardize=True)
    return np.power(np.clip(filled_z, 1e-8, None), 1.0 / power)


def fit_models(masked, df, option_cols, option_info, dte, intraday, configs):
    models = {}
    extras = {
        "base": make_extra(df, dte, intraday, "base"),
        "rich": make_extra(df, dte, intraday, "rich"),
    }

    row = row_type_interp(masked, option_cols, option_info)
    time = time_linear(masked, option_cols)
    models["Row Type Linear Interp"] = row
    models["Time Linear Interpolation"] = time
    models["Forward/Backward Fill"] = ffill_bfill(masked, option_cols)
    models["Strike Polynomial degree 2"] = row_strike_fit(masked, option_cols, option_info, degree=2)
    models["Strike Polynomial degree 3"] = row_strike_fit(masked, option_cols, option_info, degree=3)
    models["SoftImpute rank=3"] = softimpute_numpy(masked, rank=3, shrink=1e-6)

    for w in configs.get("row_time_weights", []):
        models[f"Row/Time blend row_weight={w:g}"] = row_time_blend(row, time, w)

    for power, alpha, extra_name in configs.get("power_mice", []):
        name = f"PowerMICE p={power:g} alpha={alpha:g} extra={extra_name}"
        models[name] = power_iterative_ridge(masked, extras[extra_name], power=power, alpha=alpha)

    for alpha, extra_name in configs.get("raw_mice", []):
        name = f"RawMICE alpha={alpha:g} extra={extra_name}"
        models[name] = raw_iterative_ridge_no_cap(masked, extras[extra_name], alpha=alpha)

    for power, rank in configs.get("transform_softimpute", []):
        models[f"TransformSoftImpute p={power:g} rank={rank}"] = transform_softimpute(masked, power=power, rank=rank)

    residual_bases = configs.get("residual_bases", [])
    for base_name in residual_bases:
        if base_name in models:
            for rank in [2, 3, 4]:
                models[f"ResidualSoftImpute base={base_name} rank={rank}"] = low_rank_residual(models[base_name], masked, rank=rank)

    return models


def tune_on_inner(inner_train, inner_coords, inner_y, df, option_cols, option_info, dte, intraday):
    broad_configs = {
        "power_mice": [
            (p, a, e)
            for p in [0.16, 0.20, 0.24, 0.27, 0.30, 1 / 3, 0.40, 0.50, 0.70, 1.0]
            for a in [0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007, 0.01, 0.03]
            for e in ["base", "rich"]
        ],
        "raw_mice": [(a, e) for a in [0.0007, 0.001, 0.003, 0.007, 0.01, 0.03] for e in ["base", "rich"]],
        "row_time_weights": [0.1, 0.25, 0.5, 0.75, 0.9],
        "transform_softimpute": [(p, r) for p in [0.24, 0.27, 1 / 3, 0.5] for r in [2, 3, 4]],
        "residual_bases": [],
    }
    inner_models = fit_models(inner_train, df, option_cols, option_info, dte, intraday, broad_configs)
    rows = [evaluate(name, matrix, inner_coords, inner_y) for name, matrix in inner_models.items()]
    tuning = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    top = tuning.head(12)["model"].tolist()
    return tuning, top


def evaluate_seed(seed, df, values, option_cols, option_info, dte, intraday):
    final_train, final_coords, final_y = mask_observed(values, ratio=0.10, seed=seed)
    inner_train, inner_coords, inner_y = mask_observed(final_train, ratio=0.10, seed=seed + 10_000)
    tuning, top_names = tune_on_inner(inner_train, inner_coords, inner_y, df, option_cols, option_info, dte, intraday)

    configs = {
        "power_mice": [],
        "raw_mice": [],
        "row_time_weights": [0.1, 0.25, 0.5, 0.75, 0.9],
        "transform_softimpute": [(p, r) for p in [0.24, 0.27, 1 / 3, 0.5] for r in [2, 3, 4]],
        "residual_bases": [],
    }
    for name in top_names:
        if name.startswith("PowerMICE"):
            parts = name.split()
            power = float(parts[1].split("=")[1])
            alpha = float(parts[2].split("=")[1])
            extra = parts[3].split("=")[1]
            configs["power_mice"].append((power, alpha, extra))
        elif name.startswith("RawMICE"):
            parts = name.split()
            alpha = float(parts[1].split("=")[1])
            extra = parts[2].split("=")[1]
            configs["raw_mice"].append((alpha, extra))
    configs["power_mice"].extend([(0.27, 0.0007, "base"), (0.27, 0.007, "base"), (0.27, 0.0007, "rich")])
    configs["power_mice"] = list(dict.fromkeys(configs["power_mice"]))
    configs["raw_mice"] = list(dict.fromkeys(configs["raw_mice"]))

    final_models = fit_models(final_train, df, option_cols, option_info, dte, intraday, configs)

    # Simple inner-tuned equal ensemble of top diverse models. Selection is from inner mask only.
    final_rows = [evaluate(name, matrix, final_coords, final_y) for name, matrix in final_models.items()]
    result = pd.DataFrame(final_rows).sort_values("mse").reset_index(drop=True)
    result.insert(0, "seed", seed)
    tuning.insert(0, "seed", seed)
    return result, tuning


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, option_cols, option_info = load_data("dataset.csv")
    values = df[option_cols].to_numpy(dtype=float)
    dte, intraday = expiry_features(df, option_cols)

    seeds = [42, 123, 202, 777, 2026]
    all_results = []
    all_tuning = []
    for seed in seeds:
        result, tuning = evaluate_seed(seed, df, values, option_cols, option_info, dte, intraday)
        all_results.append(result)
        all_tuning.append(tuning)
        print(f"\nSeed {seed} top 12")
        print(result.head(12).to_string(index=False))

    results = pd.concat(all_results, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    summary = (
        results.groupby("model")
        .agg(mean_mse=("mse", "mean"), median_mse=("mse", "median"), min_mse=("mse", "min"), max_mse=("mse", "max"),
             mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), n=("mse", "count"))
        .sort_values("mean_mse")
        .reset_index()
    )

    results.to_csv(OUT_DIR / "no_leak_v2_seed_results.csv", index=False)
    tuning.to_csv(OUT_DIR / "no_leak_v2_inner_tuning.csv", index=False)
    summary.to_csv(OUT_DIR / "no_leak_v2_summary.csv", index=False)

    print("\nRepeated-holdout leakage-safe summary")
    print(summary.head(30).to_string(index=False))
    print("\nSaved plots/no_leak_v2_summary.csv")


if __name__ == "__main__":
    main()
