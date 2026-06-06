from pathlib import Path

import numpy as np
import pandas as pd

from advanced_imputation_search import expiry_features
from leakage_safe_imputation_eval import power_iterative_ridge
from no_leak_model_search_v2 import make_extra
from robust_imputation_search import load_data
from surface2d_search import local_surface_2d


DATASET_PATH = "dataset.csv"
OUTPUT_DIR = Path("outputs")
FILLED_OUTPUT = OUTPUT_DIR / "filled_blend_p1rich03_90_surface2d10.csv"
SUBMISSION_OUTPUT = OUTPUT_DIR / "submission_blend_p1rich03_90_surface2d10.csv"
SEPARATOR = "||"


def generate_solution(original_path, filled_path, output_path):
    original = pd.read_csv(original_path)
    filled = pd.read_csv(filled_path)
    feature_cols = [col for col in original.columns if col != "datetime"]

    rows = []
    for col in feature_cols:
        was_missing = original[col].isna()
        for idx in original.index[was_missing]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})

    solution = pd.DataFrame(rows, columns=["id", "value"])
    solution = solution.sort_values("id").reset_index(drop=True)
    solution.to_csv(output_path, index=False)
    return solution


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_model, option_cols, option_info = load_data(DATASET_PATH)
    df_original = pd.read_csv(DATASET_PATH)
    values = df_model[option_cols].to_numpy(dtype=float)
    native_missing = np.isnan(values)

    dte, intraday = expiry_features(df_model, option_cols)
    extra_rich = make_extra(df_model, dte, intraday, variant="rich")

    p1rich03 = power_iterative_ridge(values, extra_rich, power=1.0, alpha=0.03)
    surface2d = local_surface_2d(
        values,
        option_cols,
        option_info,
        dte,
        power=0.85,
        method="plane",
        expiry_time_scale=0.75,
        non_time_scale=4.0,
        fallback=p1rich03,
    )

    final_matrix = 0.90 * p1rich03 + 0.10 * surface2d
    filled_values = values.copy()
    filled_values[native_missing] = final_matrix[native_missing]
    filled_values = np.clip(filled_values, 1e-8, None)

    filled_df = df_original.copy()
    filled_df[option_cols] = filled_values
    filled_df.to_csv(FILLED_OUTPUT, index=False)

    solution = generate_solution(DATASET_PATH, FILLED_OUTPUT, SUBMISSION_OUTPUT)

    observed_delta = np.nanmax(
        np.abs((filled_df[option_cols].to_numpy(dtype=float) - values)[~native_missing])
    )
    print(f"Saved filled file: {FILLED_OUTPUT}")
    print(f"Saved submission file: {SUBMISSION_OUTPUT}")
    print(f"Submission rows: {len(solution)}")
    print(f"Remaining missing IV cells: {int(filled_df[option_cols].isna().sum().sum())}")
    print(f"Max observed-cell change: {observed_delta}")


if __name__ == "__main__":
    main()
