import os
import argparse
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from sklearn.metrics import mutual_info_score
import numpy as np
from Bio import PDB
import networkx as nx
from scipy.stats import entropy
from tqdm import tqdm
from itertools import combinations
from multiprocessing import Pool, Manager
import pandas as pd
from scipy.cluster import hierarchy
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster



# Normalized distance between atoms
def calculate_distance(atom1, atom2):
    distance_vector = atom1 - atom2
    distance = np.linalg.norm(distance_vector)
    return distance


# Dihedral angle movements
def calc_dihedral_angle_movement(i, traj):
    res = traj.residues[i]
    ags = [res.phi_selection()]
    R = Dihedral(ags).run()
    dihedrals = R.results.angles
    dihedral_angle_movement = np.diff(dihedrals, axis=0)
    return i, dihedral_angle_movement


def calc_dihedral_angle_movement_wrapper(args):
    residue_id, traj = args
    return calc_dihedral_angle_movement(residue_id, traj)


def update_progress(res):
    res.update()
    return res


def NMI_calc(df_all_residues, num_bins=35):
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


def graph_building(pdb_file, end, dist=5.0):
    residue_graph = nx.Graph()
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    heavy_atoms = ["C", "N", "O", "S"]
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            for res1, res2 in tqdm(
                combinations(residues, 2),
                desc="Processing residues",
                total=len(residues) * (len(residues) - 1) // 2,
            ):
                res1_id = res1.get_id()[1]
                res2_id = res2.get_id()[1]
                if res1_id <= end and res2_id <= end:
                    for atom1 in res1:
                        if atom1.element in heavy_atoms:
                            for atom2 in res2:
                                if atom2.element in heavy_atoms:
                                    distance = calculate_distance(
                                        atom1.coord, atom2.coord
                                    )
                                    if distance <= dist:
                                        residue_graph.add_edge(
                                            res1.get_id()[1], res2.get_id()[1], weight=0
                                        )
    return residue_graph


def graph_assign_weights(residue_graph, mi_diff_df):
    for edge in residue_graph.edges():
        u, v = edge
        pair = ("Res " + str(u), "Res " + str(v))
        if pair in mi_diff_df["Residue Pair"].apply(tuple).tolist():
            weight = mi_diff_df.loc[
                mi_diff_df["Residue Pair"].apply(tuple) == pair, "MI Difference"
            ].values[0]
            residue_graph.edges[edge]["weight"] = weight
    return residue_graph


def faraway_residues(pdb_file, end, dist=12.0):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    heavy_atoms = ["C", "N", "O", "S"]
    distant_residues = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            for res1, res2 in tqdm(
                combinations(residues, 2),
                desc="Processing residues",
                total=len(residues) * (len(residues) - 1) // 2,
            ):
                res1_id = res1.get_id()[1]
                res2_id = res2.get_id()[1]
                if res1_id <= end and res2_id <= end:
                    are_distant = True
                    for atom1 in res1:
                        if atom1.element in heavy_atoms:
                            for atom2 in res2:
                                if atom2.element in heavy_atoms:
                                    distance = calculate_distance(
                                        atom1.coord, atom2.coord
                                    )
                                    if distance <= dist:
                                        are_distant = False
                                        break
                            if not are_distant:
                                break
                    if are_distant:
                        distant_residues.append((res1.get_id()[1], res2.get_id()[1]))
    return pd.DataFrame(distant_residues, columns=["Residue1", "Residue2"])


def max_weight_shortest_path(graph, source, target):
    shortest_path = nx.dijkstra_path(graph, source, target, weight="weight")
    total_weight = sum(
        graph[shortest_path[i]][shortest_path[i + 1]]["weight"]
        for i in range(len(shortest_path) - 1)
    )
    return shortest_path, total_weight


def collect_path_total_weights(residue_graph, df_distant_residues):
    path_total_weights = []
    for index, row in df_distant_residues.iterrows():
        try:
            shortest_path, total_weight = max_weight_shortest_path(
                residue_graph, row["Residue1"], row["Residue2"]
            )
            path_total_weights.append((shortest_path, total_weight))
        except nx.NetworkXNoPath:
            continue
    return path_total_weights


def close_residues(pdb_file, end, dist=10.0):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    heavy_atoms = ["C", "N", "O", "S"]
    close_residues = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            for res1, res2 in tqdm(
                combinations(residues, 2),
                desc="Processing residues",
                total=len(residues) * (len(residues) - 1) // 2,
            ):
                res1_id = res1.get_id()[1]
                res2_id = res2.get_id()[1]
                if res1_id <= end and res2_id <= end:
                    are_close = False
                    for atom1 in res1:
                        if atom1.element in heavy_atoms:
                            for atom2 in res2:
                                if atom2.element in heavy_atoms:
                                    distance = calculate_distance(
                                        atom1.coord, atom2.coord
                                    )
                                    if distance <= dist:
                                        are_close = True
                                        break
                            if are_close:
                                break
                    if are_close:
                        close_residues.append((res1_id, res2_id))
    return pd.DataFrame(close_residues, columns=["Residue1", "Residue2"])

def calculate_overlap(args):
    i, j, path1, path2, df, counter_queue = args
    count_true = 0
    for res1 in path1:
        for res2 in path2:
            if ((df["Residue1"] == res1) & (df["Residue2"] == res2)).any() or \
               ((df["Residue1"] == res2) & (df["Residue2"] == res1)).any():
                count_true += 1
    
    # Increment counter
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
        results = list(tqdm(pool.imap(calculate_overlap, args_list), total=len(args_list)))
        
    while not counter_queue.empty():
        counter_queue.get()
    
    for result in results:
        i, j, count_true = result
        overlap_df = overlap_df.append({"Pathway1": i, "Pathway2": j, "Overlap": count_true}, ignore_index=True)
    return overlap_df

def pathways_cluster(overlap_df, n_top_clust=3):
    overlap_matrix = overlap_df.pivot(index="Pathway1", columns="Pathway2", values="Overlap").fillna(0)
    distance_matrix = 1 - overlap_matrix
    linkage_matrix = hierarchy.linkage(distance_matrix.values, method="complete")
    silhouette_scores = []
    for n_clusters in range(2, len(overlap_matrix) + 1):
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        silhouette_scores.append(silhouette_score(distance_matrix, cluster_labels))
    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print("Optimal number of clusters:", optimal_num_clusters)
    cluster_labels = fcluster(linkage_matrix, optimal_num_clusters, criterion="maxclust")
    cluster_pathways = {cluster: [] for cluster in range(1, optimal_num_clusters + 1)}
    for i, label in enumerate(cluster_labels):
        cluster_pathways[label].append(overlap_matrix.index[i])
    silhouette_avg = silhouette_score(distance_matrix, cluster_labels)
    print("Silhouette Score:", silhouette_avg)
    fig = ff.create_dendrogram(distance_matrix.values, orientation="bottom", labels=overlap_matrix.index)
    fig.update_layout(title="Hierarchical Clustering Dendrogram", xaxis=dict(title="Pathways"), yaxis=dict(title="Distance"), xaxis_tickangle=-90,)
    fig.add_shape(type="line", x0=optimal_num_clusters - 0.5, y0=0, x1=optimal_num_clusters - 0.5, y1=silhouette_avg, line=dict(color="Red", width=3),)
    sorted_clusters = sorted(cluster_pathways.items(), key=lambda x: len(x[1]), reverse=True)
    clusters = {}
    for cluster, pathways in sorted_clusters[:n_top_clust]:
        print(f"Cluster {cluster} (Size: {len(pathways)})")
        clusters[cluster] = pathways
    return clusters

def residue_CA_coordinates(pdb_file, end):
    residue_coordinates_dict = {}
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            for res in tqdm(residues, desc="Processing residues"):
                res_id = res.get_id()[1]
                if res_id <= end:
                    for atom in res:
                        if atom.name == "CA":
                            if res_id not in residue_coordinates_dict:
                                residue_coordinates_dict[res_id] = []
                            residue_coordinates_dict[res_id].append(atom.coord)
    return residue_coordinates_dict


def cluster_prep_for_visualisaton(cluster, pdb_file):
        cluster = []
        for pathway in cluster:
            pathways = []
            for residue in pathway:
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure("pdb_structure", pdb_file)
                res_id = ('', residue, '')
                try:
                    res = structure[0][res_id]
                    atom = res["CA"]
                    coord = atom.get_coord()
                    pathways.append(coord)
                except KeyError:
                    print(res + " not found.")
                cluster.append(pathways)
        return preped_cluster
def apply_backtracking(original_dict, translation_dict):
    updated_dict = original_dict.copy() 
    
    for key, lists_of_lists in original_dict.items():
        for i, inner_list in enumerate(lists_of_lists):
            for j, item in enumerate(inner_list):
                if item in translation_dict:
                    updated_dict[key][i][j] = translation_dict[item]
    
    return updated_dict

def main():
    import pandas as pd

    parser = argparse.ArgumentParser(
        prog="mdpath",
        description="Calculate signal transduction paths in your MD trajectories",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-top",
        dest="topology",
        help="Topology file of your MD simulation",
        required=True,
    )
    parser.add_argument(
        "-traj",
        dest="trajectory",
        help="Trajectory file of your MD simulation",
        required=True,
    )
    parser.add_argument(
        "-cpu",
        dest="num_parallel_processes",
        help="Amount of cores used in multiprocessing",
        default=(os.cpu_count() // 2),
    )
    parser.add_argument(
        "-first",
        dest="first_res_num",
        help="ID of the residue start residue in your chain",
        required=True,
    )
    parser.add_argument(
        "-last",
        dest="last_res_num",
        help="ID of the residue last residue in your chain",
        required=True,
    )
    
    parser.add_argument(
        "-lig",
        dest="lig_interaction",
        help="Protein ligand interacting residues",
        default=False,
    )
    
    args = parser.parse_args()
    # Initial inputs
    num_parallel_processes = int(args.num_parallel_processes)
    topology = args.topology
    trajectory = args.trajectory
    traj = mda.Universe(topology, trajectory)
    first_res_num = int(args.first_res_num)
    last_res_num = int(args.last_res_num)
    num_residues = last_res_num - first_res_num
    lig_interaction = args.lig_interaction

    first_frame = traj.trajectory[-1]
    with mda.Writer("first_frame.pdb", multiframe=False) as pdb:
        pdb.write(traj.atoms)

    try:
        with Pool(processes=num_parallel_processes) as pool:
            residue_args = [(i, traj) for i in range(first_res_num, last_res_num + 1)]
            df_all_residues = pd.DataFrame()
            with tqdm(total=num_residues, ascii=True, desc="Processing residue dihedral movements: ",) as pbar:
                for res_id, result in pool.imap_unordered(calc_dihedral_angle_movement_wrapper, residue_args):
                    try:
                        df_residue = pd.DataFrame(result, columns=[f"Res {res_id}"])
                        df_all_residues = pd.concat([df_all_residues, df_residue], axis=1)
                        pbar.update(1)  # Update progress bar
                    except Exception as e:
                        print(f"Error processing residue {res_id}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    mi_diff_df = NMI_calc(df_all_residues, num_bins=35)
    print(mi_diff_df)

    residue_graph = graph_building("first_frame.pdb", last_res_num, dist=5.0)
    residue_graph = graph_assign_weights(residue_graph, mi_diff_df)

    for edge in residue_graph.edges():
        print(edge, residue_graph.edges[edge]["weight"])

    df_distant_residues = faraway_residues("first_frame.pdb", last_res_num, dist=12.0)
    if lig_interaction:
        with open(lig_interaction, "r") as file:
            content = file.read()
            numbers_as_strings = content.split(",")
            lig_interaction = [int(num.strip()) for num in numbers_as_strings]
        df_distant_residues = df_distant_residues[(df_distant_residues["Residue1"].isin(lig_interaction)) | (df_distant_residues["Residue2"].isin(lig_interaction))]

    print(df_distant_residues)

    import networkx as nx

    path_total_weights = collect_path_total_weights(residue_graph, df_distant_residues)

    # Sort paths based on the sum of their weights
    sorted_paths = sorted(path_total_weights, key=lambda x: x[1], reverse=True)

    # remove this later
    for path, total_weight in sorted_paths[:500]:
        print("Path:", path, "Total Weight:", total_weight)
    close_res = close_residues("first_frame.pdb", last_res_num, dist=12.0)
    # TODO multiprocess this
    # Computation of overlap by comparing every residue of every path with each other
    pathways = [path for path, _ in sorted_paths[:500]]
    overlap_df = calculate_overlap_multiprocess(pathways, close_res, num_parallel_processes)
    
    from scipy.cluster import hierarchy
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import silhouette_score
    import plotly.figure_factory as ff
    
    clusters=pathways_cluster(overlap_df)
    cluster_pathways_dict = {}
    for cluster_num, cluster_pathways in clusters.items():
        cluster_pathways_list = []
    for pathway_id in cluster_pathways:
        pathway = sorted_paths[pathway_id]
        cluster_pathways_list.append(pathway[0])
    cluster_pathways_dict[cluster_num] = cluster_pathways_list
    print(cluster_pathways_dict)
    # Coord dict for backtracking
  
    residue_coordinates_dict= residue_CA_coordinates("first_frame.pdb", last_res_num)
    updated_dict = apply_backtracking(cluster_pathways_dict, residue_coordinates_dict)
    print(updated_dict)
     
if __name__ == "__main__":
    main()
