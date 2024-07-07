import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_overlap(pathways: list[list[int]], df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the overlap between all pathways.

    Args:
        pathways (list[list[int]]): List of pathways.
        df (pd.DataFrame): Pandas dataframe with close residue pairs.

    Returns:
        overlap_df (pd.DataFrame): Pandas dataframe with the overlap between all given pathways.
    """
    overlap_df = pd.DataFrame(columns=["Pathway1", "Pathway2", "Overlap"])
    for i in tqdm(range(len(pathways))):
        path1 = pathways[i]
        for j in range(i + 1, len(pathways)):
            path2 = pathways[j]
            overlap_counter = 0
            for res1 in path1:
                for res2 in path2:
                    if ((df["Residue1"] == res1) & (df["Residue2"] == res2)).any() or (
                        (df["Residue1"] == res2) & (df["Residue2"] == res1)
                    ).any():
                        overlap_counter += 1
            overlap_df._append(
                {"Pathway1": i, "Pathway2": j, "Overlap": overlap_counter},
                ignore_index=True,
            )
            overlap_df._append(
                {"Pathway1": j, "Pathway2": i, "Overlap": overlap_counter},
                ignore_index=True,
            )

    return overlap_df


def calculate_overlap_for_pathway(
    args: tuple[int, list[int], list[list[int]], pd.DataFrame]
) -> list[dict]:
    """Calculates the overlap between a pathway and all other pathways.

    Args:
        args (tuple[int, list[int], list[list[int]], pd.DataFrame]): Argument wrapper conatining the pathway index, the pathway, all pathways and the dataframe with close residue pairs.

    Returns:
        result (list[dict]): List of dictionaries with the overlap between the given pathway and all other pathways.
    """
    i, path1, pathways, df = args
    result = []
    for j in range(i + 1, len(pathways)):
        if i != j:
            path2 = pathways[j]
            overlap_counter = 0
            for res1 in path1:
                for res2 in path2:
                    if ((df["Residue1"] == res1) & (df["Residue2"] == res2)).any() or (
                        (df["Residue1"] == res2) & (df["Residue2"] == res1)
                    ).any():
                        overlap_counter += 1
            result.append({"Pathway1": i, "Pathway2": j, "Overlap": overlap_counter})
            result.append({"Pathway1": j, "Pathway2": i, "Overlap": overlap_counter})
    return result


def calculate_overlap_parallel(
    pathways: list[list[int]], df: pd.DataFrame, num_processes: int
) -> pd.DataFrame:
    """Parallelization wrapper for the calculate_overlap_for_pathway function.

    Args:
        pathways (list[list[int]]): List of all pathways.
        df (pd.DataFrame): Pandas dataframe with close residue pairs.
        num_processes (int): Number of processes to use for parallelization.

    Returns:
        overlap_df (pd.DataFrame): Pandas dataframe with the overlap between all pathways wih all other pathways.
    """
    overlap_df = pd.DataFrame(columns=["Pathway1", "Pathway2", "Overlap"])
    with Pool(processes=num_processes) as pool:
        with tqdm(
            total=(len(pathways) ** 2 - len(pathways)),
            ascii=True,
            desc="Calculating pathway residue overlapp: ",
        ) as pbar:
            for result in pool.imap_unordered(
                calculate_overlap_for_pathway,
                [(i, path, pathways, df) for i, path in enumerate(pathways)],
            ):
                for row in result:
                    overlap_df = overlap_df._append(row, ignore_index=True)
                    pbar.update(1)
    return overlap_df


def pathways_cluster(
    overlap_df: pd.DataFrame, n_top_clust=0, save_path="clustered_paths.png"
) -> dict[int, list[int]]:
    """Clustering of pathways based on the overlap between them.

    Args:
        overlap_df (pd.DataFrame): Pandas dataframe with the overlap between all pathways.
        n_top_clust (int, optional): Number of clusters to output. Defaults to all.
        save_path (str, optional): Save path for cluster dendogram figure. Defaults to "clustered_paths.png".

    Returns:
        clusters (dict[int, list[int]]): Dictionary with the clusters and their pathways.
    """
    overlap_matrix = overlap_df.pivot(
        index="Pathway1", columns="Pathway2", values="Overlap"
    ).fillna(0)
    distance_matrix = 1 - overlap_matrix
    linkage_matrix = hierarchy.linkage(distance_matrix.values, method="complete")

    silhouette_scores = []
    for n_clusters in range(2, len(overlap_matrix) + 1):
        cluster_labels = hierarchy.fcluster(
            linkage_matrix, n_clusters, criterion="maxclust"
        )
        silhouette_scores.append(silhouette_score(distance_matrix, cluster_labels))

    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print("Optimal number of clusters:", optimal_num_clusters)

    cluster_labels = hierarchy.fcluster(
        linkage_matrix, optimal_num_clusters, criterion="maxclust"
    )
    cluster_pathways = {cluster: [] for cluster in range(1, optimal_num_clusters + 1)}
    for i, label in enumerate(cluster_labels):
        cluster_pathways[label].append(overlap_matrix.index[i])

    silhouette_avg = silhouette_score(distance_matrix, cluster_labels)
    print("Silhouette Score:", silhouette_avg)
    plt.figure(figsize=(10, 7))
    hierarchy.dendrogram(linkage_matrix, labels=overlap_matrix.index, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Pathways")
    plt.ylabel("Distance")
    plt.savefig(save_path)
    plt.close()

    sorted_clusters = sorted(
        cluster_pathways.items(), key=lambda x: len(x[1]), reverse=True
    )
    clusters = {}
    if n_top_clust == 0:
        for cluster, pathways in sorted_clusters:
            print(f"Cluster {cluster} (Size: {len(pathways)})")
            clusters[cluster] = pathways
    else: 
        for cluster, pathways in sorted_clusters[:n_top_clust]:
            print(f"Cluster {cluster} (Size: {len(pathways)})")
            clusters[cluster] = pathways
    return clusters
