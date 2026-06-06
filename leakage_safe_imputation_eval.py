import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from robust_imputation_search import (
    RANDOM_SEED,
    evaluate,
    ffill_bfill,
    fill_column_mean,
    load_data,
    make_mask,
    row_strike_fit,
    softimpute_numpy,
    time_linear,
)
from advanced_imputation_search import expiry_features, row_type_interp


OUT_DIR = Path("plots")
DATASET = Path("dataset.csv")


def mask_observed(values, ratio=0.10, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    observed = np.argwhere(~np.isnan(values))
    selected = observed[rng.choice(len(observed), size=int(len(observed) * ratio), replace=False)]
    masked = values.copy()
    y_true = values[selected[:, 0], selected[:, 1]].copy()
    masked[selected[:, 0], selected[:, 1]] = np.nan
    return masked, selected, y_true


def power_iterative_ridge(masked, extra, power=0.27, alpha=0.0007, max_iter=18):
    transformed = np.where(
        np.isnan(masked),
        np.nan,
        np.power(np.clip(masked, 1e-8, None), power),
    )
    out = fill_column_mean(transformed)
    missing_masks = [np.isnan(transformed[:, j]) for j in range(transformed.shape[1])]
    extra = np.asarray(extra, dtype=float)
    extra = (extra - extra.mean(axis=0)) / (extra.std(axis=0) + 1e-12)

    for _ in range(max_iter):
        previous = out.copy()
        for target in range(transformed.shape[1]):
            miss = missing_masks[target]
            obs = ~miss
            if miss.sum() == 0 or obs.sum() < 10:
                continue
            predictors = np.column_stack([extra, np.delete(out, target, axis=1)])
            train_x = np.column_stack([np.ones(obs.sum()), predictors[obs]])
            pred_x = np.column_stack([np.ones(miss.sum()), predictors[miss]])
            penalty = alpha * np.eye(train_x.shape[1])
            penalty[0, 0] = 0.0
            coef = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ transformed[obs, target])
            out[miss, target] = pred_x @ coef

        if np.nanmean((out - previous) ** 2) < 1e-13:
            break

    return np.power(np.clip(out, 1e-8, None), 1.0 / power)


def raw_iterative_ridge_no_cap(masked, extra, alpha=0.01, max_iter=18):
    out = fill_column_mean(masked)
    missing_masks = [np.isnan(masked[:, j]) for j in range(masked.shape[1])]
    extra = np.asarray(extra, dtype=float)
    extra = (extra - extra.mean(axis=0)) / (extra.std(axis=0) + 1e-12)

    for _ in range(max_iter):
        previous = out.copy()
        for target in range(masked.shape[1]):
            miss = missing_masks[target]
            obs = ~miss
            if miss.sum() == 0 or obs.sum() < 10:
                continue
            predictors = np.column_stack([extra, np.delete(out, target, axis=1)])
            train_x = np.column_stack([np.ones(obs.sum()), predictors[obs]])
            pred_x = np.column_stack([np.ones(miss.sum()), predictors[miss]])
            penalty = alpha * np.eye(train_x.shape[1])
            penalty[0, 0] = 0.0
            coef = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ out[obs, target])
            out[miss, target] = pred_x @ coef
        if np.nanmean((out - previous) ** 2) < 1e-13:
            break
    return np.clip(out, 1e-8, None)


def extra_features(df, dte, intraday):
    return np.column_stack(
        [
            df["underlying_price"].to_numpy(),
            np.arange(len(df), dtype=float),
            dte,
            intraday,
            (dte < 1.0).astype(float),
            np.sin(np.arange(len(df)) / 12.0),
            np.cos(np.arange(len(df)) / 12.0),
        ]
    )


def fit_candidate_models(masked, df, option_cols, option_info, dte, intraday, powers_alphas):
    extra = extra_features(df, dte, intraday)
    models = {
        "Forward/Backward Fill": ffill_bfill(masked, option_cols),
        "Time Linear Interpolation": time_linear(masked, option_cols),
        "Row Type Linear Interp": row_type_interp(masked, option_cols, option_info),
        "Strike Polynomial degree 2": row_strike_fit(masked, option_cols, option_info, degree=2),
        "Strike Polynomial degree 3": row_strike_fit(masked, option_cols, option_info, degree=3),
        "SoftImpute rank=3": softimpute_numpy(masked, rank=3, shrink=1e-6),
        "Raw Iterative Ridge alpha=0.01 no_cap": raw_iterative_ridge_no_cap(masked, extra, alpha=0.01),
    }
    for power, alpha in powers_alphas:
        models[f"PowerMICE p={power:.4g} alpha={alpha:g}"] = power_iterative_ridge(
            masked, extra, power=power, alpha=alpha
        )
    return models


def tune_power_mice(inner_train, inner_coords, inner_y, df, dte, intraday):
    extra = extra_features(df, dte, intraday)
    grid = list(itertools.product(
        [0.20, 0.24, 0.25, 0.27, 0.28, 0.30, 1 / 3, 0.40, 0.50],
        [0.0003, 0.0007, 0.001, 0.003, 0.01],
    ))
    rows = []
    for power, alpha in grid:
        matrix = power_iterative_ridge(inner_train, extra, power=power, alpha=alpha)
        row = evaluate(f"PowerMICE p={power:.4g} alpha={alpha:g}", matrix, inner_coords, inner_y)
        row["power"] = power
        row["alpha"] = alpha
        rows.append(row)
    tuned = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    return tuned, float(tuned.loc[0, "power"]), float(tuned.loc[0, "alpha"])


def tune_row_correction(inner_base, inner_row, inner_coords, inner_y, dte, option_cols, option_info):
    base_pred = inner_base[inner_coords[:, 0], inner_coords[:, 1]]
    row_pred = inner_row[inner_coords[:, 0], inner_coords[:, 1]]
    coord_types = np.array([option_info[option_cols[j]]["type"] for j in inner_coords[:, 1]])
    coord_expiry = dte[inner_coords[:, 0]] < 1.0
    groups = [
        ("expiry_CE", coord_expiry & (coord_types == "CE"), np.linspace(0.03, 0.20, 35)),
        ("expiry_PE", coord_expiry & (coord_types == "PE"), np.linspace(0.03, 0.20, 35)),
        ("non_expiry_CE", (~coord_expiry) & (coord_types == "CE"), np.linspace(0.002, 0.04, 20)),
        ("non_expiry_PE", (~coord_expiry) & (coord_types == "PE"), np.linspace(0.002, 0.04, 20)),
    ]
    params = []
    for name, mask, d_grid in groups:
        if mask.sum() == 0:
            params.append({"group": name, "cap": 0.0, "weight": 0.0, "inner_mse": np.nan})
            continue
        best = (np.inf, 0.0, 0.0)
        for cap in d_grid:
            delta = np.clip(row_pred[mask] - base_pred[mask], -cap, cap)
            denom = np.dot(delta, delta)
            weight = np.dot(inner_y[mask] - base_pred[mask], delta) / denom if denom > 0 else 0.0
            weight = float(np.clip(weight, -2.0, 2.0))
            pred = base_pred[mask] + weight * delta
            mse = float(np.mean((inner_y[mask] - pred) ** 2))
            if mse < best[0]:
                best = (mse, float(cap), weight)
        params.append({"group": name, "cap": best[1], "weight": best[2], "inner_mse": best[0]})
    return pd.DataFrame(params)


def apply_row_correction(base, row, dte, option_cols, option_info, params):
    out = base.copy()
    param_map = {record["group"]: record for record in params.to_dict("records")}
    for row_idx in range(out.shape[0]):
        is_expiry = dte[row_idx] < 1.0
        for col_idx, col in enumerate(option_cols):
            opt_type = option_info[col]["type"]
            if is_expiry and opt_type == "CE":
                param = param_map["expiry_CE"]
            elif is_expiry and opt_type == "PE":
                param = param_map["expiry_PE"]
            elif opt_type == "CE":
                param = param_map["non_expiry_CE"]
            else:
                param = param_map["non_expiry_PE"]
            delta = np.clip(row[row_idx, col_idx] - base[row_idx, col_idx], -param["cap"], param["cap"])
            out[row_idx, col_idx] = base[row_idx, col_idx] + param["weight"] * delta
    return np.clip(out, 1e-8, None)


def tune_global_stack(inner_models, inner_coords, inner_y, ridge=1e-4):
    names = list(inner_models)
    pred = np.column_stack([inner_models[name][inner_coords[:, 0], inner_coords[:, 1]] for name in names])
    x = np.column_stack([np.ones(len(inner_y)), pred])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ inner_y)
    return names, coef


def apply_global_stack(models, names, coef):
    out = np.full_like(next(iter(models.values())), coef[0])
    for weight, name in zip(coef[1:], names):
        out += weight * models[name]
    return np.clip(out, 1e-8, None)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, option_cols, option_info = load_data(DATASET)
    values = df[option_cols].to_numpy(dtype=float)
    dte, intraday = expiry_features(df, option_cols)

    final_train, final_coords, final_y = mask_observed(values, ratio=0.10, seed=RANDOM_SEED)
    inner_train, inner_coords, inner_y = mask_observed(final_train, ratio=0.10, seed=RANDOM_SEED + 1)

    inner_tuning, tuned_power, tuned_alpha = tune_power_mice(inner_train, inner_coords, inner_y, df, dte, intraday)
    inner_tuning.to_csv(OUT_DIR / "leakage_safe_inner_power_mice_tuning.csv", index=False)

    tuned_pair = [(tuned_power, tuned_alpha)]
    comparison_pairs = [
        (0.27, 0.0007),
        (0.27, 0.001),
        (1 / 3, 0.001),
        (0.50, 0.001),
        (1.00, 0.01),
    ]
    powers_alphas = list(dict.fromkeys(tuned_pair + comparison_pairs))

    inner_models = fit_candidate_models(inner_train, df, option_cols, option_info, dte, intraday, powers_alphas)
    inner_base_name = f"PowerMICE p={tuned_power:.4g} alpha={tuned_alpha:g}"
    row_params = tune_row_correction(
        inner_models[inner_base_name],
        inner_models["Row Type Linear Interp"],
        inner_coords,
        inner_y,
        dte,
        option_cols,
        option_info,
    )
    row_params.to_csv(OUT_DIR / "leakage_safe_inner_row_correction_params.csv", index=False)
    inner_models[f"{inner_base_name} + inner-tuned row correction"] = apply_row_correction(
        inner_models[inner_base_name],
        inner_models["Row Type Linear Interp"],
        dte,
        option_cols,
        option_info,
        row_params,
    )

    stack_names, stack_coef = tune_global_stack(inner_models, inner_coords, inner_y, ridge=1e-4)
    pd.DataFrame({"term": ["intercept"] + stack_names, "weight": stack_coef}).to_csv(
        OUT_DIR / "leakage_safe_inner_global_stack_weights.csv", index=False
    )

    final_models = fit_candidate_models(final_train, df, option_cols, option_info, dte, intraday, powers_alphas)
    final_base_name = f"PowerMICE p={tuned_power:.4g} alpha={tuned_alpha:g}"
    final_models[f"{final_base_name} + inner-tuned row correction"] = apply_row_correction(
        final_models[final_base_name],
        final_models["Row Type Linear Interp"],
        dte,
        option_cols,
        option_info,
        row_params,
    )
    final_models["Global stacked ensemble inner-tuned"] = apply_global_stack(final_models, stack_names, stack_coef)

    metrics = pd.DataFrame(
        [evaluate(name, matrix, final_coords, final_y) for name, matrix in final_models.items()]
    ).sort_values("mse").reset_index(drop=True)
    metrics.to_csv(OUT_DIR / "leakage_safe_imputation_metrics.csv", index=False)

    print("Leakage-safe final holdout MSEs")
    print(f"Final holdout cells: {len(final_y)}")
    print(f"Inner-tuned PowerMICE: p={tuned_power:.4g}, alpha={tuned_alpha:g}")
    print(metrics.to_string(index=False))
    print("\nSaved plots/leakage_safe_imputation_metrics.csv")
    print("No filled dataset was written.")


if __name__ == "__main__":
    main()
