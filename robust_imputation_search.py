import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
DATASET = Path("dataset.csv")
OUT_DIR = Path("plots")


def parse_option_column(name):
    match = re.search(r"\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$", name)
    if not match:
        match = re.search(r"(\d+)(CE|PE)$", name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def load_data(path=DATASET):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)
    info = {col: parse_option_column(col) for col in df.columns}
    option_cols = [col for col, parsed in info.items() if parsed is not None]
    option_cols = sorted(option_cols, key=lambda c: (info[c][1], info[c][0]))
    info = {col: {"strike": info[col][0], "type": info[col][1]} for col in option_cols}
    return df, option_cols, info


def make_mask(values, mask_ratio=0.10, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    observed = np.argwhere(~np.isnan(values))
    n_mask = int(len(observed) * mask_ratio)
    selected = observed[rng.choice(len(observed), size=n_mask, replace=False)]
    masked = values.copy()
    y_true = values[selected[:, 0], selected[:, 1]].copy()
    masked[selected[:, 0], selected[:, 1]] = np.nan
    return masked, selected, y_true


def mse_mae(y_true, y_pred):
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    return mse, float(np.sqrt(mse)), float(np.mean(np.abs(err)))


def fill_column_mean(x):
    out = x.copy()
    means = np.nanmean(out, axis=0)
    means = np.nan_to_num(means, nan=np.nanmean(means))
    rows, cols = np.where(np.isnan(out))
    out[rows, cols] = means[cols]
    return out


def ffill_bfill(x, option_cols):
    return pd.DataFrame(x, columns=option_cols).ffill().bfill().fillna(0.12).to_numpy()


def time_linear(x, option_cols):
    df = pd.DataFrame(x, columns=option_cols)
    return df.interpolate(method="linear", axis=0, limit_direction="both").ffill().bfill().fillna(0.12).to_numpy()


def row_strike_fit(x, option_cols, info, degree=1):
    out = x.copy()
    for opt_type in ["CE", "PE"]:
        idx = [i for i, col in enumerate(option_cols) if info[col]["type"] == opt_type]
        strikes = np.array([info[option_cols[i]]["strike"] for i in idx], dtype=float)
        order = np.argsort(strikes)
        idx = np.array(idx)[order]
        strikes = strikes[order]
        for r in range(out.shape[0]):
            vals = out[r, idx]
            obs = ~np.isnan(vals)
            if obs.sum() == 0:
                continue
            if obs.sum() <= degree:
                pred = np.interp(strikes, strikes[obs], vals[obs])
            else:
                fitted_degree = min(degree, obs.sum() - 1)
                scaled = (strikes - strikes.mean()) / strikes.std()
                coef = np.polyfit(scaled[obs], vals[obs], fitted_degree)
                pred = np.polyval(coef, scaled)
            vals[~obs] = np.clip(pred[~obs], 0.01, 1.5)
            out[r, idx] = vals
    return fill_column_mean(out)


def blend(*matrices, weights=None):
    if weights is None:
        weights = np.ones(len(matrices)) / len(matrices)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return sum(w * m for w, m in zip(weights, matrices))


def softimpute_numpy(x, rank=4, shrink=0.002, max_iter=200, tol=1e-7, standardize=True):
    observed = ~np.isnan(x)
    if standardize:
        means = np.nanmean(x, axis=0)
        stds = np.nanstd(x, axis=0)
        stds[stds == 0] = 1.0
        work = (x - means) / stds
        shrink_work = shrink / np.nanmean(stds)
    else:
        means = np.zeros(x.shape[1])
        stds = np.ones(x.shape[1])
        work = x.copy()
        shrink_work = shrink
    filled = fill_column_mean(work)
    for _ in range(max_iter):
        old_missing = filled[~observed].copy()
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        s = np.maximum(s - shrink_work, 0)
        k = min(rank, np.count_nonzero(s))
        reconstructed = (u[:, :k] * s[:k]) @ vt[:k, :] if k else np.zeros_like(filled)
        filled[~observed] = reconstructed[~observed]
        if old_missing.size and np.linalg.norm(filled[~observed] - old_missing) / (np.linalg.norm(old_missing) + 1e-12) < tol:
            break
    return filled * stds + means


def iterative_ridge(x, extra_features, alpha=1e-4, max_iter=15):
    out = fill_column_mean(x)
    missing_masks = [np.isnan(x[:, j]) for j in range(x.shape[1])]
    extra = np.asarray(extra_features, dtype=float)
    extra = (extra - extra.mean(axis=0)) / (extra.std(axis=0) + 1e-12)
    for _ in range(max_iter):
        previous = out.copy()
        for target in range(x.shape[1]):
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
        if np.nanmean((out - previous) ** 2) < 1e-12:
            break
    return np.clip(out, 0.01, 1.5)


def evaluate(name, matrix, coords, y_true):
    pred = matrix[coords[:, 0], coords[:, 1]]
    mse, rmse, mae = mse_mae(y_true, pred)
    return {"model": name, "mse": mse, "rmse": rmse, "mae": mae}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, option_cols, info = load_data()
    values = df[option_cols].to_numpy(dtype=float)
    masked, coords, y_true = make_mask(values)

    eda = pd.DataFrame(
        {
            "rows": [len(df)],
            "iv_columns": [len(option_cols)],
            "native_missing_cells": [int(np.isnan(values).sum())],
            "native_missing_pct": [float(np.isnan(values).mean() * 100)],
            "masked_validation_cells": [len(coords)],
            "underlying_min": [df["underlying_price"].min()],
            "underlying_max": [df["underlying_price"].max()],
        }
    )
    eda.to_csv(OUT_DIR / "eda_summary.csv", index=False)

    extra = np.column_stack(
        [
            df["underlying_price"].to_numpy(),
            np.arange(len(df), dtype=float),
            np.sin(np.arange(len(df)) / 12.0),
            np.cos(np.arange(len(df)) / 12.0),
        ]
    )

    matrices = {}
    matrices["Forward/Backward Fill"] = ffill_bfill(masked, option_cols)
    matrices["Time Linear Interpolation"] = time_linear(masked, option_cols)
    for degree in [1, 2, 3]:
        matrices[f"Strike Polynomial degree {degree}"] = row_strike_fit(masked, option_cols, info, degree=degree)
    for rank, shrink in itertools.product([2, 3, 4, 6, 8, 12], [0.0001, 0.0005, 0.001, 0.002, 0.005]):
        matrices[f"SoftImpute numpy rank={rank} shrink={shrink}"] = softimpute_numpy(masked, rank=rank, shrink=shrink)
    for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        matrices[f"Iterative Ridge/MICE alpha={alpha:g}"] = iterative_ridge(masked, extra, alpha=alpha)

    base_names = list(matrices)
    rows = [evaluate(name, mat, coords, y_true) for name, mat in matrices.items()]
    best_names = [r["model"] for r in sorted(rows, key=lambda r: r["mse"])[:8]]
    for size in [2, 3, 4, 5]:
        for combo in itertools.combinations(best_names, size):
            name = "Ensemble: " + " + ".join(combo)
            matrices[name] = blend(*[matrices[c] for c in combo])
            rows.append(evaluate(name, matrices[name], coords, y_true))

    results = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    results.to_csv(OUT_DIR / "imputation_model_search_metrics.csv", index=False)

    best_name = results.loc[0, "model"]
    filled = df.copy()
    filled[option_cols] = matrices[best_name]
    filled.to_csv("filled_dataset_best.csv", index=False)

    print("EDA summary")
    print(eda.to_string(index=False))
    print("\nTop 20 models by masked validation MSE")
    print(results.head(20).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("Saved plots/imputation_model_search_metrics.csv and filled_dataset_best.csv")


if __name__ == "__main__":
    main()
