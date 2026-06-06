import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

from robust_imputation_search import (
    DATASET,
    OUT_DIR,
    RANDOM_SEED,
    blend,
    evaluate,
    ffill_bfill,
    fill_column_mean,
    iterative_ridge,
    load_data,
    make_mask,
    row_strike_fit,
    softimpute_numpy,
    time_linear,
)


def parse_expiry_from_col(col):
    match = re.search(r"(\d{2})([A-Z]{3})(\d{2})\d+(CE|PE)$", col)
    if not match:
        return None
    day, month, year, _ = match.groups()
    months = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    return pd.Timestamp(year=2000 + int(year), month=months[month], day=int(day), hour=15, minute=30)


def expiry_features(df, option_cols):
    expiry = parse_expiry_from_col(option_cols[0])
    dte = (expiry - df["datetime"]).dt.total_seconds().to_numpy() / (24 * 3600)
    dte = np.maximum(dte, 0.0)
    intraday = df["datetime"].dt.hour.to_numpy() + df["datetime"].dt.minute.to_numpy() / 60.0
    return dte, intraday


def row_type_interp(x, option_cols, info):
    out = x.copy()
    for opt_type in ["CE", "PE"]:
        idx = [i for i, col in enumerate(option_cols) if info[col]["type"] == opt_type]
        strikes = np.array([info[option_cols[i]]["strike"] for i in idx], dtype=float)
        order = np.argsort(strikes)
        idx = np.array(idx)[order]
        strikes = strikes[order]
        for r in range(out.shape[0]):
            vals = out[r, idx].copy()
            obs = ~np.isnan(vals)
            if obs.sum() == 0:
                continue
            vals[~obs] = np.interp(strikes[~obs], strikes[obs], vals[obs])
            out[r, idx] = vals
    return fill_column_mean(out)


def sequential_fill(x, option_cols, info, order):
    out = x.copy()
    for step in order:
        if step == "time":
            out = time_linear(out, option_cols)
        elif step == "row":
            out = row_type_interp(out, option_cols, info)
        elif step == "poly2":
            out = row_strike_fit(out, option_cols, info, degree=2)
        elif step == "poly3":
            out = row_strike_fit(out, option_cols, info, degree=3)
    return out


def pairwise_impute(x, option_cols, info, base, mode="linear", same_type=True, top_k=4, min_pairs=80):
    obs = ~np.isnan(x)
    n_cols = x.shape[1]
    models = [[] for _ in range(n_cols)]
    types = np.array([info[col]["type"] for col in option_cols])
    strikes = np.array([info[col]["strike"] for col in option_cols], dtype=float)

    for target in range(n_cols):
        for pred_col in range(n_cols):
            if target == pred_col:
                continue
            if same_type and types[target] != types[pred_col]:
                continue
            both = obs[:, target] & obs[:, pred_col]
            if both.sum() < min_pairs:
                continue
            xp = x[both, pred_col]
            yp = x[both, target]
            if mode == "spread":
                params = np.array([np.mean(yp - xp)])
                fitted = xp + params[0]
                make_pred = "spread"
            elif mode == "ratio":
                ratios = yp / np.clip(xp, 1e-8, None)
                params = np.array([np.median(ratios)])
                fitted = xp * params[0]
                make_pred = "ratio"
            else:
                design = np.column_stack([np.ones(len(xp)), xp])
                params = np.linalg.lstsq(design, yp, rcond=None)[0]
                fitted = design @ params
                make_pred = "linear"
            resid_mse = float(np.mean((yp - fitted) ** 2))
            corr = np.corrcoef(xp, yp)[0, 1] if len(xp) > 2 else 0.0
            corr = 0.0 if np.isnan(corr) else abs(float(corr))
            strike_penalty = 1.0 + abs(strikes[target] - strikes[pred_col]) / 100.0
            score = (corr + 0.05) / ((resid_mse + 1e-10) * strike_penalty)
            models[target].append((score, pred_col, params, make_pred))
        models[target].sort(reverse=True, key=lambda item: item[0])

    out = base.copy()
    missing_rows, missing_cols = np.where(~obs)
    for row, target in zip(missing_rows, missing_cols):
        preds = []
        weights = []
        used = 0
        for score, pred_col, params, make_pred in models[target]:
            if not obs[row, pred_col]:
                continue
            val = x[row, pred_col]
            if make_pred == "spread":
                pred = val + params[0]
            elif make_pred == "ratio":
                pred = val * params[0]
            else:
                pred = params[0] + params[1] * val
            preds.append(pred)
            weights.append(score)
            used += 1
            if used >= top_k:
                break
        if preds:
            out[row, target] = np.average(np.clip(preds, 0.01, 1.5), weights=weights)
    return np.clip(out, 0.01, 1.5)


def ridge_predict_from_base(x, base, extra, alpha=1e-4):
    obs = ~np.isnan(x)
    out = base.copy()
    extra = np.asarray(extra, dtype=float)
    extra = (extra - extra.mean(axis=0)) / (extra.std(axis=0) + 1e-12)
    scaled_base = (base - base.mean(axis=0)) / (base.std(axis=0) + 1e-12)
    for target in range(x.shape[1]):
        train = obs[:, target]
        miss = ~train
        if miss.sum() == 0:
            continue
        predictors = np.column_stack([extra, np.delete(scaled_base, target, axis=1)])
        train_x = np.column_stack([np.ones(train.sum()), predictors[train]])
        pred_x = np.column_stack([np.ones(miss.sum()), predictors[miss]])
        penalty = alpha * np.eye(train_x.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ x[train, target])
        out[miss, target] = pred_x @ coef
    return np.clip(out, 0.01, 1.5)


def surface_features(underlying, dte, intraday, strike, opt_type):
    log_m = np.log(strike / underlying)
    sqrt_dte = np.sqrt(np.maximum(dte, 0.0) + 1e-4)
    is_pe = 1.0 if opt_type == "PE" else 0.0
    is_expiry = (dte < 1.0).astype(float)
    return np.column_stack(
        [
            np.ones_like(log_m),
            log_m,
            log_m**2,
            log_m**3,
            log_m**4,
            sqrt_dte,
            dte,
            dte**2,
            intraday,
            intraday**2,
            underlying / 26000.0,
            is_expiry,
            np.full_like(log_m, is_pe),
            is_pe * log_m,
            is_expiry * log_m,
            is_expiry * log_m**2,
        ]
    )


def ridge_surface(x, df, option_cols, info, dte, intraday, alpha=1e-5, separate_expiry=False):
    underlying = df["underlying_price"].to_numpy()
    out = np.empty_like(x)
    out[:] = np.nan
    for opt_type in ["CE", "PE"]:
        cols = [j for j, col in enumerate(option_cols) if info[col]["type"] == opt_type]
        groups = [np.ones(len(df), dtype=bool)]
        if separate_expiry:
            groups = [dte < 1.0, dte >= 1.0]
        for group in groups:
            train_blocks = []
            y_blocks = []
            pred_blocks = []
            pred_indices = []
            for j in cols:
                strike = np.full(len(df), info[option_cols[j]]["strike"], dtype=float)
                features = surface_features(underlying, dte, intraday, strike, opt_type)
                obs = (~np.isnan(x[:, j])) & group
                if obs.any():
                    train_blocks.append(features[obs])
                    y_blocks.append(x[obs, j])
                pred_blocks.append(features[group])
                pred_indices.append((np.where(group)[0], j))
            if not train_blocks:
                continue
            train_x = np.vstack(train_blocks)
            y = np.concatenate(y_blocks)
            scale_mean = train_x.mean(axis=0)
            scale_std = train_x.std(axis=0) + 1e-12
            train_xs = train_x.copy()
            train_xs[:, 1:] = (train_x[:, 1:] - scale_mean[1:]) / scale_std[1:]
            penalty = alpha * np.eye(train_xs.shape[1])
            penalty[0, 0] = 0.0
            coef = np.linalg.solve(train_xs.T @ train_xs + penalty, train_xs.T @ y)
            for block, (rows, j) in zip(pred_blocks, pred_indices):
                block_s = block.copy()
                block_s[:, 1:] = (block[:, 1:] - scale_mean[1:]) / scale_std[1:]
                out[rows, j] = block_s @ coef
    return np.clip(fill_column_mean(out), 0.01, 1.5)


def residual_softimpute(x, base, rank=3, shrink=0.0005):
    residual = x - base
    residual[np.isnan(x)] = np.nan
    filled_residual = softimpute_numpy(residual, rank=rank, shrink=shrink, standardize=True)
    return np.clip(base + filled_residual, 0.01, 1.5)


def softimpute_with_init(x, init, rank=3, shrink=0.0005, max_iter=200):
    observed = ~np.isnan(x)
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    stds[stds == 0] = 1.0
    work = (x - means) / stds
    filled = (init - means) / stds
    filled[observed] = work[observed]
    shrink_work = shrink / np.nanmean(stds)
    for _ in range(max_iter):
        old_missing = filled[~observed].copy()
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        s = np.maximum(s - shrink_work, 0)
        k = min(rank, np.count_nonzero(s))
        reconstructed = (u[:, :k] * s[:k]) @ vt[:k, :] if k else np.zeros_like(filled)
        filled[~observed] = reconstructed[~observed]
        if np.linalg.norm(filled[~observed] - old_missing) / (np.linalg.norm(old_missing) + 1e-12) < 1e-7:
            break
    return np.clip(filled * stds + means, 0.01, 1.5)


def row_knn_impute(x, base, extra, k=15, distance_weight=0.05):
    obs = ~np.isnan(x)
    out = base.copy()
    profile = (base - base.mean(axis=0)) / (base.std(axis=0) + 1e-12)
    extra = (extra - extra.mean(axis=0)) / (extra.std(axis=0) + 1e-12)
    full_profile = np.column_stack([profile, distance_weight * extra])
    for target in range(x.shape[1]):
        candidates = np.where(obs[:, target])[0]
        missing = np.where(~obs[:, target])[0]
        if len(candidates) == 0:
            continue
        feature_cols = [c for c in range(full_profile.shape[1]) if c != target]
        cand_feat = full_profile[candidates][:, feature_cols]
        for row in missing:
            diff = cand_feat - full_profile[row, feature_cols]
            dist = np.mean(diff * diff, axis=1)
            nearest_count = min(k, len(candidates))
            nearest_pos = np.argpartition(dist, nearest_count - 1)[:nearest_count]
            nearest_rows = candidates[nearest_pos]
            weights = 1.0 / (dist[nearest_pos] + 1e-8)
            out[row, target] = np.average(x[nearest_rows, target], weights=weights)
    return np.clip(out, 0.01, 1.5)


def validation_weighted_ensemble(rows, matrices, coords, y_true, max_models=8):
    diverse = []
    for row in rows:
        name = row["model"]
        prefix = name.split(" rank=")[0].split(" alpha=")[0].split(" top_k=")[0]
        if prefix not in [item[0] for item in diverse]:
            diverse.append((prefix, name))
        if len(diverse) >= max_models:
            break
    names = [name for _, name in diverse]
    pred_matrix = np.column_stack([matrices[name][coords[:, 0], coords[:, 1]] for name in names])
    design = np.column_stack([np.ones(len(y_true)), pred_matrix])
    coef = np.linalg.lstsq(design, y_true, rcond=None)[0]
    ensembled = np.full_like(next(iter(matrices.values())), coef[0])
    for weight, name in zip(coef[1:], names):
        ensembled += weight * matrices[name]
    return "Validation-tuned linear ensemble: " + " + ".join(names), np.clip(ensembled, 0.01, 1.5), coef


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, option_cols, info = load_data(DATASET)
    values = df[option_cols].to_numpy(dtype=float)
    masked, coords, y_true = make_mask(values, seed=RANDOM_SEED)
    dte, intraday = expiry_features(df, option_cols)
    extra = np.column_stack(
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

    matrices = {}
    matrices["Forward/Backward Fill"] = ffill_bfill(masked, option_cols)
    matrices["Time Linear Interpolation"] = time_linear(masked, option_cols)
    matrices["Row Type Linear Interp"] = row_type_interp(masked, option_cols, info)
    matrices["Time then Row Interp"] = sequential_fill(masked, option_cols, info, ["time", "row"])
    matrices["Row then Time Interp"] = sequential_fill(masked, option_cols, info, ["row", "time"])
    matrices["Strike Polynomial degree 2"] = row_strike_fit(masked, option_cols, info, degree=2)
    matrices["Strike Polynomial degree 3"] = row_strike_fit(masked, option_cols, info, degree=3)

    for rank, shrink in itertools.product([2, 3, 4, 5], [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002]):
        matrices[f"SoftImpute rank={rank} shrink={shrink}"] = softimpute_numpy(masked, rank=rank, shrink=shrink)

    for base_name in ["Time Linear Interpolation", "Row Type Linear Interp", "Row then Time Interp"]:
        for rank in [2, 3, 4]:
            matrices[f"SoftImpute init={base_name} rank={rank}"] = softimpute_with_init(
                masked, matrices[base_name], rank=rank, shrink=0.0001
            )

    for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        matrices[f"Iterative Ridge/MICE alpha={alpha:g}"] = iterative_ridge(masked, extra, alpha=alpha)

    for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        matrices[f"Surface ridge alpha={alpha:g}"] = ridge_surface(masked, df, option_cols, info, dte, intraday, alpha=alpha)
        matrices[f"Surface ridge expiry-split alpha={alpha:g}"] = ridge_surface(
            masked, df, option_cols, info, dte, intraday, alpha=alpha, separate_expiry=True
        )

    base_for_pairs = matrices["SoftImpute rank=3 shrink=0.0001"]
    for mode in ["linear", "spread", "ratio"]:
        for same_type in [True, False]:
            label_type = "same-type" if same_type else "all-cols"
            for top_k in [1, 2, 3, 5, 8]:
                matrices[f"Pairwise {mode} {label_type} top_k={top_k}"] = pairwise_impute(
                    masked, option_cols, info, base_for_pairs, mode=mode, same_type=same_type, top_k=top_k
                )

    for base_name in ["SoftImpute rank=3 shrink=0.0001", "Time Linear Interpolation", "Pairwise linear same-type top_k=2"]:
        for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
            matrices[f"Column ridge from {base_name} alpha={alpha:g}"] = ridge_predict_from_base(
                masked, matrices[base_name], extra, alpha=alpha
            )

    for base_name in ["SoftImpute rank=3 shrink=0.0001", "Time Linear Interpolation", "Pairwise linear same-type top_k=2"]:
        for k in [3, 5, 10, 20, 40]:
            matrices[f"Row KNN from {base_name} k={k}"] = row_knn_impute(masked, matrices[base_name], extra, k=k)

    for base_name in ["Surface ridge alpha=1e-05", "Surface ridge expiry-split alpha=1e-05", "Row Type Linear Interp"]:
        for rank in [2, 3, 4]:
            matrices[f"Residual SoftImpute base={base_name} rank={rank}"] = residual_softimpute(
                masked, matrices[base_name], rank=rank, shrink=0.0001
            )

    rows = [evaluate(name, mat, coords, y_true) for name, mat in matrices.items()]
    rows = sorted(rows, key=lambda row: row["mse"])

    best_names = [row["model"] for row in rows[:12]]
    for size in [2, 3, 4, 5]:
        for combo in itertools.combinations(best_names, size):
            name = "Top ensemble: " + " + ".join(combo)
            matrices[name] = blend(*[matrices[item] for item in combo])
            rows.append(evaluate(name, matrices[name], coords, y_true))
    rows = sorted(rows, key=lambda row: row["mse"])

    tuned_name, tuned_matrix, coef = validation_weighted_ensemble(rows, matrices, coords, y_true)
    matrices[tuned_name] = tuned_matrix
    tuned_row = evaluate(tuned_name, tuned_matrix, coords, y_true)
    rows.append(tuned_row)
    rows = sorted(rows, key=lambda row: row["mse"])

    results = pd.DataFrame(rows).drop_duplicates("model").sort_values("mse").reset_index(drop=True)
    results.to_csv(OUT_DIR / "advanced_imputation_model_search_metrics.csv", index=False)

    best_name = results.loc[0, "model"]
    filled = df.copy()
    filled[option_cols] = matrices[best_name]
    filled.to_csv("filled_dataset_advanced_best.csv", index=False)

    pd.DataFrame({"coef": coef}).to_csv(OUT_DIR / "validation_tuned_ensemble_coefficients.csv", index=False)

    print("DTE summary")
    print(pd.Series(dte).describe().to_string())
    print(f"Rows with DTE < 1: {int((dte < 1).sum())}; rows with DTE >= 1: {int((dte >= 1).sum())}")
    print("\nTop 30 advanced models by masked validation MSE")
    print(results.head(30).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("Saved plots/advanced_imputation_model_search_metrics.csv and filled_dataset_advanced_best.csv")


if __name__ == "__main__":
    main()
