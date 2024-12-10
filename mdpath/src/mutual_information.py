"""Mutual Information Calculation --- :mod:`mdpath.src.mutual_information`
===============================================================================

This module contains the class `NMICalculator` which calculates the Normalized Mutual Information (NMI)
for all residue pairs in a given dataset based on the dihedral angle movements over the course of the analysed MD trajectory.


Classes
--------

:class:`NMICalculator`
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy


class NMICalculator:
    """Calculate Normalized Mutual Information (NMI) between dihedral angle movements of residue pairs.

    Attributes:
        df_all_residues (pd.DataFrame): DataFrame containing all residues.

        num_bins (int): Number of bins to use for histogram calculations. Default is 35.

        GMM (optional): Option to switch between histogram method and Gaussian Mixture Model for binning before NMI calculation. Default is False.

        mi_diff_df (pd.DataFrame): DataFrame containing the mutual information differences. Is calculated using either GMM or histogram method.

        entropy_df (pd.DataFrame): Pandas dataframe with residue and entropy values. Is calculated using either GMM or histogram method.
    """

    def __init__(
        self, df_all_residues: pd.DataFrame, num_bins: int = 35, GMM=False
    ) -> None:
        self.df_all_residues = df_all_residues
        self.num_bins = num_bins
        self.GMM = GMM
        if GMM:
            self.mi_diff_df, self.entropy_df = self.NMI_calcs_with_GMM()
        else:
            self.mi_diff_df, self.entropy_df = self.NMI_calcs()

    def NMI_calcs(self) -> pd.DataFrame:
        """Nornmalized Mutual Information and Entropy calculation for all residue pairs.

        Returns:
            mi_diff_df (pd.DataFrame): Pandas dataframe with residue pair and mutual information difference.

            entropy_df (pd.DataFrame): Pandas dataframe with residue and entropy values.
        """
        entropys = {}
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
                        entropys[col1] = entropy_col1
                        entropys[col2] = entropy_col2
                        nmi = mi / np.sqrt(entropy_col1 * entropy_col2)
                        normalized_mutual_info[(col1, col2)] = nmi
                        progress_bar.update(1)
        entropy_df = pd.DataFrame(entropys.items(), columns=["Residue", "Entropy"])
        mi_diff_df = pd.DataFrame(
            normalized_mutual_info.items(), columns=["Residue Pair", "MI Difference"]
        )
        max_mi_diff = mi_diff_df["MI Difference"].max()
        mi_diff_df["MI Difference"] = (
            max_mi_diff - mi_diff_df["MI Difference"]
        )  # Calculate the the weights
        return mi_diff_df, entropy_df

    def select_n_components(data: pd.DataFrame, max_components: int = 10) -> int:
        """Select the optimal number of GMM components using BIC

        Args:
            data (pd.DataFrame): Data to fit the GMM model.

            max_components (int): Maximum number of components to test. Default is 10.

        Returns:
            best_n_components (int): Optimal number of components.
        """
        lowest_bic = np.inf
        best_n_components = 1
        bic_scores = []

        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n)
            gmm.fit(data)
            bic = gmm.bic(data)
            bic_scores.append(bic)
            if bic < lowest_bic:
                lowest_bic = bic
                best_n_components = n
        return best_n_components

    def NMI_calcs_with_GMM(self) -> pd.DataFrame:
        """Nornmalized Mutual Information and Entropy calculation for all residue pairs using Gaussian Mixture Models (GMM) for binning.

        Returns:
            mi_diff_df (pd.DataFrame): Pandas dataframe with residue pair and mutual information difference.

            entropy_df (pd.DataFrame): Pandas dataframe with residue and entropy values.
        """
        entropys = {}
        normalized_mutual_info = {}
        total_iterations = len(self.df_all_residues.columns) ** 2

        with tqdm(
            total=total_iterations,
            desc="\033[1mCalculating Normalized Mutual Information (GMM)\033[0m",
        ) as progress_bar:
            for col1 in self.df_all_residues.columns:
                for col2 in self.df_all_residues.columns:
                    if col1 != col2:
                        data_col1 = self.df_all_residues[[col1]]
                        data_col2 = self.df_all_residues[[col2]]
                        n_components_col1 = self.select_n_components(
                            data_col1, max_components=10
                        )
                        n_components_col2 = self.select_n_components(
                            data_col2, max_components=10
                        )

                        gmm_col1 = GaussianMixture(n_components=n_components_col1).fit(
                            data_col1
                        )
                        gmm_col2 = GaussianMixture(n_components=n_components_col2).fit(
                            data_col2
                        )
                        labels_col1 = gmm_col1.predict(data_col1)
                        labels_col2 = gmm_col2.predict(data_col2)

                        mi = mutual_info_score(labels_col1, labels_col2)

                        entropy_col1 = entropy(np.bincount(labels_col1))
                        entropy_col2 = entropy(np.bincount(labels_col2))
                        entropys[col1] = entropy_col1
                        entropys[col2] = entropy_col2
                        nmi = mi / np.sqrt(entropy_col1 * entropy_col2)
                        normalized_mutual_info[(col1, col2)] = nmi

                        progress_bar.update(1)

        entropy_df = pd.DataFrame(entropys.items(), columns=["Residue", "Entropy"])
        mi_diff_df = pd.DataFrame(
            normalized_mutual_info.items(), columns=["Residue Pair", "MI Difference"]
        )

        max_mi_diff = mi_diff_df["MI Difference"].max()
        mi_diff_df["MI Difference"] = (
            max_mi_diff - mi_diff_df["MI Difference"]
        )  # Calculate the weights

        return mi_diff_df, entropy_df
