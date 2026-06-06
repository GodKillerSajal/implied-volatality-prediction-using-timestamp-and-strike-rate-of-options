import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_imputation_search import expiry_features, row_type_interp
from leakage_safe_imputation_eval import power_iterative_ridge
from no_leak_model_search_v2 import make_extra
from robust_imputation_search import evaluate, fill_column_mean, load_data, time_linear


OUT_DIR = Path("plots")


def native_row_mask(values, native_missing, seed):
    rng = np.random.default_rng(seed)
    masked = values.copy()
    coords = []
    for row in range(values.shape[0]):
        k = int(native_missing[row].sum())
        observed = np.where(~np.isnan(values[row]))[0]
        if k and len(observed):
            chosen = rng.choice(observed, size=min(k, len(observed)), replace=False)
            masked[row, chosen] = np.nan
            coords.extend((row, int(col)) for col in chosen)
    coords = np.array(coords, dtype=int)
    return masked, coords, values[coords[:, 0], coords[:, 1]]


def type_indices(option_cols, option_info, opt_type):
    idx = [i for i, col in enumerate(option_cols) if option_info[col]["type"] == opt_type]
    return np.array(sorted(idx, key=lambda i: option_info[option_cols[i]]["strike"]), dtype=int)


def weighted_median(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return values[np.searchsorted(np.cumsum(weights), cutoff)]


def local_surface_2d(
    masked,
    option_cols,
    option_info,
    dte,
    power=1.0,
    method="plane",
    expiry_time_scale=1.5,
    non_time_scale=10.0,
    strike_scale=1.5,
    expiry_time_window=4,
    non_time_window=24,
    strike_window=3,
    ridge=1e-4,
    fallback=None,
):
    transformed = np.where(np.isnan(masked), np.nan, np.power(np.clip(masked, 1e-8, None), power))
    out = transformed.copy()
    if fallback is None:
        fallback_t = fill_column_mean(transformed)
    else:
        fallback_t = np.power(np.clip(fallback, 1e-8, None), power)

    for opt_type in ["CE", "PE"]:
        global_idx = type_indices(option_cols, option_info, opt_type)
        sub = transformed[:, global_idx]
        missing = np.argwhere(np.isnan(sub))
        for row, pos in missing:
            is_expiry = dte[row] < 1.0
            time_scale = expiry_time_scale if is_expiry else non_time_scale
            time_window = expiry_time_window if is_expiry else non_time_window
            r0 = max(0, row - time_window)
            r1 = min(sub.shape[0], row + time_window + 1)
            c0 = max(0, pos - strike_window)
            c1 = min(sub.shape[1], pos + strike_window + 1)
            block = sub[r0:r1, c0:c1]
            obs = np.argwhere(~np.isnan(block))
            if len(obs) < 3:
                out[row, global_idx[pos]] = fallback_t[row, global_idx[pos]]
                continue

            rr = obs[:, 0] + r0
            cc = obs[:, 1] + c0
            y = block[obs[:, 0], obs[:, 1]]
            dr = (rr - row).astype(float)
            ds = (cc - pos).astype(float)
            weights = np.exp(-np.abs(dr) / max(time_scale, 1e-6)) * np.exp(-np.abs(ds) / max(strike_scale, 1e-6))
            weights *= np.where(dr == 0, 2.0, 1.0)
            weights *= np.where(ds == 0, 1.5, 1.0)

            if method == "mean" or len(obs) < 5:
                pred = np.average(y, weights=weights)
            elif method == "median":
                pred = weighted_median(y, weights)
            else:
                x = np.column_stack(
                    [
                        np.ones(len(y)),
                        dr / max(time_scale, 1e-6),
                        ds / max(strike_scale, 1e-6),
                        (ds / max(strike_scale, 1e-6)) ** 2,
                    ]
                )
                wx = x * np.sqrt(weights[:, None])
                wy = y * np.sqrt(weights)
                penalty = ridge * np.eye(x.shape[1])
                penalty[0, 0] = 0.0
                try:
                    coef = np.linalg.solve(wx.T @ wx + penalty, wx.T @ wy)
                    pred = coef[0]
                except np.linalg.LinAlgError:
                    pred = np.average(y, weights=weights)
            out[row, global_idx[pos]] = pred

    result = np.power(np.clip(out, 1e-8, None), 1.0 / power)
    if np.isnan(result).any():
        result[np.isnan(result)] = np.power(fallback_t[np.isnan(result)], 1.0 / power)
    return np.clip(result, 1e-8, None)


def fit_core_models(masked, df, option_cols, option_info, dte, intraday):
    extra_rich = make_extra(df, dte, intraday, "rich")
    extra_base = make_extra(df, dte, intraday, "base")
    p1rich03 = power_iterative_ridge(masked, extra_rich, power=1.0, alpha=0.03)
    p1rich02 = power_iterative_ridge(masked, extra_rich, power=1.0, alpha=0.02)
    p085rich02 = power_iterative_ridge(masked, extra_rich, power=0.85, alpha=0.02)
    p027base = power_iterative_ridge(masked, extra_base, power=0.27, alpha=0.0007)
    row = row_type_interp(masked, option_cols, option_info)
    time = time_linear(masked, option_cols)
    return {
        "p1rich03": p1rich03,
        "p1rich02": p1rich02,
        "p085rich02": p085rich02,
        "p027base": p027base,
        "row": row,
        "time": time,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, option_cols, option_info = load_data("dataset.csv")
    values = df[option_cols].to_numpy(dtype=float)
    native_missing = np.isnan(values)
    dte, intraday = expiry_features(df, option_cols)
    seeds = [42, 123, 202, 777, 2026, 7, 99, 314, 1001, 9001]
    rows = []

    configs = []
    for method in ["mean", "plane"]:
        for power in [0.85, 1.0]:
            for ets in [0.75, 1.5]:
                for nts in [4.0, 8.0]:
                    configs.append((method, power, ets, nts))

    for seed in seeds:
        masked, coords, y_true = native_row_mask(values, native_missing, seed)
        core = fit_core_models(masked, df, option_cols, option_info, dte, intraday)
        models = {
            "p1rich03": core["p1rich03"],
            "blend_p1rich03_95_row_05": 0.95 * core["p1rich03"] + 0.05 * core["row"],
            "blend_p1rich02_55_p085_45": 0.55 * core["p1rich02"] + 0.45 * core["p085rich02"],
        }
        for method, power, ets, nts in configs:
            surface = local_surface_2d(
                masked,
                option_cols,
                option_info,
                dte,
                power=power,
                method=method,
                expiry_time_scale=ets,
                non_time_scale=nts,
                fallback=core["p1rich03"],
            )
            name = f"surf2d {method} p={power:g} ets={ets:g} nts={nts:g}"
            models[name] = surface
            for weight in [0.1, 0.2]:
                models[f"blend p1rich03 {1-weight:g} + {name} {weight:g}"] = (
                    (1 - weight) * core["p1rich03"] + weight * surface
                )

        for name, matrix in models.items():
            row = evaluate(name, matrix, coords, y_true)
            row["seed"] = seed
            rows.append(row)
        print(f"finished seed {seed}")

    results = pd.DataFrame(rows)
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
    results.to_csv(OUT_DIR / "surface2d_seed_results.csv", index=False)
    summary.to_csv(OUT_DIR / "surface2d_summary.csv", index=False)
    print(summary.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
