import pandas as pd
import numpy as np
from mdpath.src.graph import (
    graph_assign_weights,
    collect_path_total_weights,
)
from mdpath.src.mutual_information import NMI_calc
from typing import Dict, Set, Tuple, List
import os

def create_bootstrap_sample(df: pd.DataFrame) -> tuple[int, set[tuple]]:
    """Creates a sample from the dataframe with replacement for bootstrap analysis.

    Args:
        df (pd.DataFrame):Pandas dataframe with residue dihedral angle movements.

    Returns:
        bootstrap_sample (pd.DataFrame): Pandas dataframe containing the frames for the bootstrap analysis.
    """
    bootstrap_sample = df.apply(
        lambda col: col.sample(n=len(df), replace=True).reset_index(drop=True)
    )
    return bootstrap_sample


def process_bootstrap_sample(
    df_all_residues: pd.DataFrame,
    residue_graph_empty: Dict,
    df_distant_residues: pd.DataFrame,
    pathways_set: Set[Tuple],
    numpath: int,
    sample_num: int,
    num_bins: int = 35,
) -> Tuple[int, List[List[int]]]:
    """Process a bootstrap sample to find common paths with the original sample.

    Args:
        df_all_residues (pd.DataFrame): Pandas dataframe with residue dihedral angle movements.
        residue_graph_empty (dict): Empty residue graph.
        df_distant_residues (pd.DataFrame): Pandas dataframe with distant residues.
        pathways_set (set[tuple]): Set of tuples with the pathways for bootstrapping.
        numpath (int): Amount of top paths to consider.
        num_bins (int, optional): Number of bins to group dihedral angle movements into for NMI calculation. Defaults to 35.

    Returns:
        common_count (int): Number of common paths between the bootstrap sample and the original sample.
        bootstrap_pathways (list[list[int]]): List of paths within the bootstrap sample.
    """
    bootstrap_sample = create_bootstrap_sample(df_all_residues)
    bootstrap_mi_diff = NMI_calc(bootstrap_sample, num_bins=num_bins)
    bootstrap_residue_graph = graph_assign_weights(
        residue_graph_empty, bootstrap_mi_diff
    )
    bootstrap_path_total_weights = collect_path_total_weights(
        bootstrap_residue_graph, df_distant_residues
    )
    bootstrap_sorted_paths = sorted(
        bootstrap_path_total_weights, key=lambda x: x[1], reverse=True
    )
    bootstrap_pathways = [path for path, _ in bootstrap_sorted_paths[:numpath]]
    file_name = f"bootstrap_sample_{sample_num}.txt"
    new_file_path = os.path.join("bootstrap", file_name)
    with open(new_file_path, 'w') as file:
        for pathway in bootstrap_pathways:
            file.write(f"{pathway}\n")

    bootstrap_set = set(tuple(path) for path in bootstrap_pathways)
    common_elements = bootstrap_set.intersection(pathways_set)
    common_count = len(common_elements)
    return common_count, bootstrap_pathways


def bootstrap_analysis(
    df_all_residues: pd.DataFrame,
    residue_graph_empty: Dict,
    df_distant_residues: pd.DataFrame,
    sorted_paths: List[Tuple],
    num_bootstrap_samples: int,
    numpath: int,
    num_bins: int = 35,
) -> Tuple[np.ndarray, Dict]:
    """Analyse the common paths between the original sample and bootstrap samples.

    Args:
        df_all_residues (pd.DataFrame): Pandas dataframe with residue dihedral angle movements.
        residue_graph_empty (dict): Empty residue graph.
        df_distant_residues (pd.DataFrame): Pandas dataframe with distant residues.
        sorted_paths (list[tuple]): List of tuples with the sorted paths.
        num_bootstrap_samples (int): Amount of samples to generate for bootstrap analysis.
        numpath (int): Number of top paths to consider.
        num_bins (int, optional): Number of bins to group dihedral angle movements into for NMI calculation. Defaults to 35.

    Returns:
        common_counts (np.array): Array with the counts of common paths between the original sample and bootstrap samples.
        path_confidence_intervals (dict): Dictionary with the confidence intervals for each path.
    """
    os.makedirs("bootstrap", exist_ok=True)
    pathways = [path for path, _ in sorted_paths[:numpath]]
    pathways_set = set(tuple(path) for path in pathways)
    results = []
    path_occurrences = {tuple(path): [] for path in pathways_set}

    for _ in range(num_bootstrap_samples):
        result, occurrences = process_bootstrap_sample(
            df_all_residues,
            residue_graph_empty,
            df_distant_residues,
            pathways_set,
            numpath,
            sample_num = _,
            num_bins=num_bins,
        )
        results.append(result)
        current_paths = set(tuple(path) for path in occurrences)
        for path in path_occurrences.keys():
            if path in current_paths:
                path_occurrences[path].append(1)
            else:
                path_occurrences[path].append(0)

    common_counts = np.array(results)
    standard_error = np.std(common_counts) / np.sqrt(num_bootstrap_samples)
    print("Standard error:", standard_error)

    path_confidence_intervals = {}
    for path, occurrences in path_occurrences.items():
        occurrences = np.array(occurrences, dtype=int)
        mean_occurrence = np.mean(occurrences)
        lower_bound = np.percentile(occurrences, 2.5)
        upper_bound = np.percentile(occurrences, 97.5)
        path_confidence_intervals[path] = (mean_occurrence, lower_bound, upper_bound)

    return common_counts, path_confidence_intervals
