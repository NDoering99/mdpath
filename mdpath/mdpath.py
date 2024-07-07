import os
import argparse
import pandas as pd
import numpy as np
import MDAnalysis as mda
import json
import multiprocessing
from tqdm import tqdm
import numpy as np
import pickle

from mdpath.src.structure import (
    calculate_dihedral_movement_parallel,
    faraway_residues,
    close_residues,
    res_num_from_pdb,
)
from mdpath.src.mutual_information import NMI_calc
from mdpath.src.graph import (
    graph_building,
    graph_assign_weights,
    collect_path_total_weights,
)
from mdpath.src.cluster import (
    calculate_overlap_parallel,
    pathways_cluster,
    calculate_overlap,
)
from mdpath.src.visualization import (
    residue_CA_coordinates,
    apply_backtracking,
    cluster_prep_for_visualisaton,
    format_dict,
    visualise_graph,
    precompute_path_properties,
    precompute_cluster_properties_quick,
)
from mdpath.src.bootstrap import bootstrap_analysis


def main():
    parser = argparse.ArgumentParser(
        prog="mdpath",
        description="Calculate signal transduction paths in your MD trajectories",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-top",
        dest="topology",
        help="Topology file of your MD simulation",
        required=False,
    )
    parser.add_argument(
        "-traj",
        dest="trajectory",
        help="Trajectory file of your MD simulation",
        required=False,
    )
    parser.add_argument(
        "-cpu",
        dest="num_parallel_processes",
        help="Amount of cores used in multiprocessing",
        default=(os.cpu_count() // 2),
    )
    parser.add_argument(
        "-lig",
        dest="lig_interaction",
        help="Protein ligand interacting residues",
        default=False,
    )
    parser.add_argument(
        "-bs",
        dest="bootstrap",
        help="How often bootstrapping should be performed.",
        default=False,
    )
    # Comp Flags
    parser.add_argument(
        "-comp",
        dest="comp",
        help="Morphes a cluster onto another pdb.",
        default=False,
    )
    parser.add_argument(
        "-atop",
        dest="atop",
        help="residue_coordinates that are used as a template.",
        default=False,
    )
    parser.add_argument(
        "-bcluster",
        dest="bcluster",
        help="Cluster that is morphed.",
        default=False,
    )

    args = parser.parse_args()
    # Initial inputs

    if args.comp:
        if args.atop and args.bcluster:
            with open(args.atop, "rb") as pkl_file:
                residue_coordinates_dict = pickle.load(pkl_file)
            with open(args.bcluster, "rb") as pkl_file:
                cluster_pathways_dict = pickle.load(pkl_file)
            updated_dict = apply_backtracking(
                cluster_pathways_dict, residue_coordinates_dict
            )
            formatted_dict = format_dict(updated_dict)
            with open("morphed_clusters_paths.json", "w") as json_file_2:
                json.dump(formatted_dict, json_file_2)
            exit()
        else:
            print(
                "Topology (residue_coordinates) and bcluster (cluster) are required and ajson needed."
            )
            exit()
    if not args.topology or not args.trajectory:
        print("Both trajectory and topology files are required!")
        exit()
    num_parallel_processes = int(args.num_parallel_processes)
    topology = args.topology
    trajectory = args.trajectory
    traj = mda.Universe(topology, trajectory)
    lig_interaction = args.lig_interaction
    bootstrap = args.bootstrap
    first_frame = traj.trajectory[-1]
    with mda.Writer("first_frame.pdb", multiframe=False) as pdb:
        pdb.write(traj.atoms)
    first_res_num, last_res_num = res_num_from_pdb("first_frame.pdb")
    num_residues = last_res_num - first_res_num
    df_all_residues = calculate_dihedral_movement_parallel(
        num_parallel_processes, first_res_num, last_res_num, num_residues, traj
    )

    mi_diff_df = NMI_calc(df_all_residues, num_bins=35)
    mi_diff_df.to_csv("mi_diff_df.csv", index=False)
    residue_graph_empty = graph_building("first_frame.pdb", last_res_num, dist=5.0)
    residue_graph = graph_assign_weights(residue_graph_empty, mi_diff_df)
    visualise_graph(residue_graph)

    df_distant_residues = faraway_residues("first_frame.pdb", last_res_num, dist=12.0)
    if lig_interaction:
        if os.path.exists(str(lig_interaction)):
            with open(str(lig_interaction), "r") as file:
                content = file.read()
                lig_interaction = [int(res.strip()) for res in content.split(",")]
        else:
            lig_interaction = [res for res in str(lig_interaction).split(",")]
            df_distant_residues = df_distant_residues[
                (df_distant_residues["Residue1"].isin(lig_interaction))
                | (df_distant_residues["Residue2"].isin(lig_interaction))
            ]


    path_total_weights = collect_path_total_weights(residue_graph, df_distant_residues)
    sorted_paths = sorted(path_total_weights, key=lambda x: x[1], reverse=True)
    sorted_paths_bs = sorted_paths

    with open("output.txt", "w") as file:
        for path, total_weight in sorted_paths[:500]:
            file.write(f"Path: {path}, Total Weight: {total_weight}\n")


    if bootstrap:
        num_bootstrap_samples = int(bootstrap)
        common_counts, path_confidence_intervals = bootstrap_analysis(
            df_all_residues,
            residue_graph_empty,
            df_distant_residues,
            sorted_paths_bs,
            num_bootstrap_samples,
        )
        for path, (mean, lower, upper) in path_confidence_intervals.items():
            path_str = " -> ".join(map(str, path))
            file_name = "path_confidence_intervals.txt"
            with open(file_name, "w") as file:
                for path, (mean, lower, upper) in path_confidence_intervals.items():
                    path_str = " -> ".join(map(str, path))
                    file.write(
                        f"{path_str}: Mean={mean}, 2.5%={lower}, 97.5%={upper}\n"
                    )
        print(f"Path confidence intervals have been saved to {file_name}")

    close_res = close_residues("first_frame.pdb", last_res_num, dist=12.0)
    pathways = [path for path, _ in sorted_paths[:500]]
    overlap_df = calculate_overlap_parallel(pathways, close_res, num_parallel_processes)

    clusters = pathways_cluster(overlap_df)
    cluster_pathways_dict = {}
    for cluster_num, cluster_pathways in clusters.items():
        cluster_pathways_list = []
        for pathway_id in cluster_pathways:
            pathway = sorted_paths[pathway_id]
            cluster_pathways_list.append(pathway[0])
        cluster_pathways_dict[cluster_num] = cluster_pathways_list

    residue_coordinates_dict = residue_CA_coordinates("first_frame.pdb", last_res_num)
    # For comp
    with open("residue_coordinates.pkl", "wb") as pkl_file:
        pickle.dump(residue_coordinates_dict, pkl_file)

    with open("cluster_pathways_dict.pkl", "wb") as pkl_file:
        pickle.dump(cluster_pathways_dict, pkl_file)

    updated_dict = apply_backtracking(cluster_pathways_dict, residue_coordinates_dict)

    formated_dict = format_dict(updated_dict)
    with open("clusters_paths.json", "w") as json_file:
        json.dump(formated_dict, json_file)

    with open("clusters_paths.json", "r") as json_file:
        json_data = json.load(json_file)

    path_properties = precompute_path_properties(json_data)
    with open("precomputed_clusters_paths.json", "w") as out_file:
        json.dump(path_properties, out_file, indent=4)
    quick_path_properties = precompute_cluster_properties_quick(json_data)
    with open("quick_precomputed_clusters_paths.json", "w") as out_file2:
        json.dump(quick_path_properties, out_file2, indent=4)


if __name__ == "__main__":
    main()
