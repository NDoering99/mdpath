import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def calculate_overlap(pathways, df):
    overlap_df = pd.DataFrame(columns=["Pathway1", "Pathway2", "Overlap"])
    for i in tqdm(range(len(pathways))):
        path1 = pathways[i]
        for j in range(i + 1, len(pathways)):
            path2 = pathways[j]
            overlap_counter = 0
            for res1 in path1:
                for res2 in path2:
                    if ((df["Residue1"] == res1) & (df["Residue2"] == res2)).any() or ((df["Residue1"] == res2) & (df["Residue2"] == res1)).any():
                        overlap_counter += 1
            overlap_df._append({"Pathway1": i, "Pathway2": j, "Overlap": overlap_counter}, ignore_index=True)
            overlap_df._append({"Pathway1": j, "Pathway2": i, "Overlap": overlap_counter}, ignore_index=True)
            
    return overlap_df


def calculate_overlap_for_pathway(args):
    i, path1, pathways, df = args
    result = []
    for j in range(i + 1, len(pathways)):
        if i != j:
            path2 = pathways[j]
            overlap_counter = 0
            for res1 in path1:
                for res2 in path2:
                    if ((df["Residue1"] == res1) & (df["Residue2"] == res2)).any() or ((df["Residue1"] == res2) & (df["Residue2"] == res1)).any():
                        overlap_counter += 1
            result.append({"Pathway1": i, "Pathway2": j, "Overlap": overlap_counter})
            result.append({"Pathway1": j, "Pathway2": i, "Overlap": overlap_counter})
    return result  


def calculate_overlap_parallel(pathways, df, num_processes):
    overlap_df = pd.DataFrame(columns=["Pathway1", "Pathway2", "Overlap"])
    with Pool(processes=num_processes) as pool:
        with tqdm(total=(len(pathways) ** 2 - len(pathways)), ascii=True, desc="Calculating pathway residue overlapp: ") as pbar:
            for result in pool.imap_unordered(calculate_overlap_for_pathway, [(i, path, pathways, df) for i, path in enumerate(pathways)]):
                for row in result:
                    overlap_df = overlap_df._append(row, ignore_index=True)
                    pbar.update(1)
    print(overlap_df.head())
    return overlap_df     


def pathways_cluster(overlap_df, n_top_clust=3):
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
    
    linked = hierarchy.linkage(distance_matrix.values, 'single')

    # Create dendrogram
    plt = plt.figure(figsize=(10, 7))
    dendro = hierarchy.dendrogram(linked, labels=overlap_matrix.index, orientation='top', color_threshold=0)

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Pathways")
    plt.ylabel("Distance")

    # Add line
    plt.axvline(x=optimal_num_clusters - 0.5, ymin=0, ymax=1, color='Red', linewidth=3)

    plt.save("dendrogram.png")
    
    sorted_clusters = sorted(
        cluster_pathways.items(), key=lambda x: len(x[1]), reverse=True
    )
    clusters = {}
    for cluster, pathways in sorted_clusters[:n_top_clust]:
        print(f"Cluster {cluster} (Size: {len(pathways)})")
        clusters[cluster] = pathways
    return clusters
