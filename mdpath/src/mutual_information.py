import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy


def NMI_calc(df_all_residues: pd.DataFrame, num_bins=35) -> pd.DataFrame:
    normalized_mutual_info = {}
    total_iterations = len(df_all_residues.columns) ** 2
    progress_bar = tqdm(
        total=total_iterations, desc="Calculating Normalized Mutual Information"
    )
    for col1 in df_all_residues.columns:
        for col2 in df_all_residues.columns:
            if col1 != col2:
                hist_col1, _ = np.histogram(df_all_residues[col1], bins=num_bins)
                hist_col2, _ = np.histogram(df_all_residues[col2], bins=num_bins)
                hist_joint, _, _ = np.histogram2d(
                    df_all_residues[col1], df_all_residues[col2], bins=num_bins
                )
                mi = mutual_info_score(hist_col1, hist_col2, contingency=hist_joint)
                entropy_col1 = entropy(hist_col1)
                entropy_col2 = entropy(hist_col2)
                nmi = mi / np.sqrt(entropy_col1 * entropy_col2)
                normalized_mutual_info[(col1, col2)] = nmi
                progress_bar.update(1)
    progress_bar.close()
    mi_diff_df = pd.DataFrame(
        normalized_mutual_info.items(), columns=["Residue Pair", "MI Difference"]
    )
    max_mi_diff = mi_diff_df["MI Difference"].max()
    mi_diff_df["MI Difference"] = (
        max_mi_diff - mi_diff_df["MI Difference"]
    )  # Calculate the the weights
    return mi_diff_df
