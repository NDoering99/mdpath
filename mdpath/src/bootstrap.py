import pandas as pd
import numpy as np
from mdpath.src.graph import (
    graph_assign_weights,
    collect_path_total_weights,
)
from mdpath.src.mutual_information import NMI_calc


def create_bootstrap_sample(df):
    bootstrap_sample = pd.DataFrame()
    for col in df.columns:
        bootstrap_sample[col] = (
            df[col].sample(n=len(df), replace=True).reset_index(drop=True)
        )
    return bootstrap_sample


def process_bootstrap_sample(
    df_all_residues, residue_graph_empty, df_distant_residues, pathways_set, num_bins=35
):
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
    bootstrap_pathways = [path for path, _ in bootstrap_sorted_paths[:500]]
    bootstrap_set = set(tuple(path) for path in bootstrap_pathways)  # Convert to tuple
    common_elements = bootstrap_set.intersection(pathways_set)
    common_count = len(common_elements)
    return common_count


def bootstrap_analysis(
    df_all_residues,
    residue_graph_empty,
    df_distant_residues,
    pathways_set,
    num_bootstrap_samples,
    num_bins=35,
):
    results = []
    for _ in range(num_bootstrap_samples):
        result = process_bootstrap_sample(
            df_all_residues,
            residue_graph_empty,
            df_distant_residues,
            pathways_set,
            num_bins=num_bins,
        )
        results.append(result)
    common_counts = np.array(results)
    standard_error = np.std(common_counts) / np.sqrt(num_bootstrap_samples)
    print("Standard error:", standard_error)
    return common_counts
