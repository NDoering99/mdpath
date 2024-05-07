import pandas as pd
from multiprocessing import Pool, Manager
from tqdm import tqdm
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score


def calculate_overlap(args):
    i, j, path1, path2, df, counter_queue = args
    count_true = 0
    for res1 in path1:
        for res2 in path2:
            if ((df["Residue1"] == res1) & (df["Residue2"] == res2)).any() or (
                (df["Residue1"] == res2) & (df["Residue2"] == res1)
            ).any():
                count_true += 1
    counter_queue.put(1)

    return i, j, count_true


def calculate_overlap_multiprocess(pathways, df, num_processes):
    overlap_df = pd.DataFrame(columns=["Pathway1", "Pathway2", "Overlap"])
    args_list = []
    manager = Manager()
    counter_queue = manager.Queue()

    for i, path1 in enumerate(pathways):
        for j, path2 in enumerate(pathways):
            args_list.append((i, j, path1, path2, df, counter_queue))

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap(calculate_overlap, args_list), total=len(args_list))
        )

    while not counter_queue.empty():
        counter_queue.get()

    for result in results:
        i, j, count_true = result
        overlap_df = overlap_df.append(
            {"Pathway1": i, "Pathway2": j, "Overlap": count_true}, ignore_index=True
        )
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
    fig = ff.create_dendrogram(
        distance_matrix.values, orientation="bottom", labels=overlap_matrix.index
    )
    fig.update_layout(
        title="Hierarchical Clustering Dendrogram",
        xaxis=dict(title="Pathways"),
        yaxis=dict(title="Distance"),
        xaxis_tickangle=-90,
    )
    fig.add_shape(
        type="line",
        x0=optimal_num_clusters - 0.5,
        y0=0,
        x1=optimal_num_clusters - 0.5,
        y1=silhouette_avg,
        line=dict(color="Red", width=3),
    )
    sorted_clusters = sorted(
        cluster_pathways.items(), key=lambda x: len(x[1]), reverse=True
    )
    clusters = {}
    for cluster, pathways in sorted_clusters[:n_top_clust]:
        print(f"Cluster {cluster} (Size: {len(pathways)})")
        clusters[cluster] = pathways
    return clusters
