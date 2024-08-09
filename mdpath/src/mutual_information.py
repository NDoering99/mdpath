import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy


class NMICalculator:
    def __init__(self, df_all_residues, num_bins=35) -> None:
        self.df_all_residues = df_all_residues
        self.num_bins = num_bins
        self.mi_diff_df = self.NMI_calc()

    def NMI_calcs(self) -> pd.DataFrame:
        """Nornmalized Mutual Information calculation for all residue pairs.

        Args:

        Returns:
            mi_diff_df (pd.DataFrame): Pandas dataframe with residue pair and mutual information difference.
        """
        normalized_mutual_info = {}
        total_iterations = len(self.df_all_residues.columns) ** 2
        with tqdm(
            total=total_iterations,
            desc="\033[1mCalculating Normalized Mutual Information\033[0m",
        ) as progress_bar:
            for col1 in self.df_all_residues.columns:
                for col2 in self.df_all_residues.columns:
                    if col1 != col2:
                        hist_col1, _ = np.histogram(
                            self.df_all_residues[col1], bins=self.num_bins
                        )
                        hist_col2, _ = np.histogram(
                            self.df_all_residues[col2], bins=self.num_bins
                        )
                        hist_joint, _, _ = np.histogram2d(
                            self.df_all_residues[col1],
                            self.df_all_residues[col2],
                            bins=self.num_bins,
                        )
                        mi = mutual_info_score(
                            hist_col1, hist_col2, contingency=hist_joint
                        )
                        entropy_col1 = entropy(hist_col1)
                        entropy_col2 = entropy(hist_col2)
                        nmi = mi / np.sqrt(entropy_col1 * entropy_col2)
                        normalized_mutual_info[(col1, col2)] = nmi
                        progress_bar.update(1)
        mi_diff_df = pd.DataFrame(
            normalized_mutual_info.items(), columns=["Residue Pair", "MI Difference"]
        )
        max_mi_diff = mi_diff_df["MI Difference"].max()
        mi_diff_df["MI Difference"] = (
            max_mi_diff - mi_diff_df["MI Difference"]
        )  # Calculate the the weights
        return mi_diff_df
