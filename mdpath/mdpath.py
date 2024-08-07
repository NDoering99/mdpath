import os
import argparse
import pandas as pd
import numpy as np
import MDAnalysis as mda
import json
import multiprocessing
from tqdm import tqdm
import numpy as np
from collections import defaultdict
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
)
from mdpath.src.visualization import (
    residue_CA_coordinates,
    apply_backtracking,
    cluster_prep_for_visualisation,
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
        nargs='+'
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
    parser.add_argument(
        "-multitraj",
        dest="multitraj",
        help="List of multiple pathways from previous analysis.",
        default=False,
        nargs='+',
    )

    # Dist flags

    parser.add_argument(
        "-fardist",
        dest="fardist",
        help="Default distance for faraway residues.",
        required=False,
        default=12.0,
    )
    parser.add_argument(
        "-closedist",
        dest="closedist",
        help="Default distance for close residues.",
        required=False,
        default=12.0,
    )
    parser.add_argument(
        "-graphdist",
        dest="graphdist",
        help="Default distance for residues making up the graph.",
        required=False,
        default=5.0,
    )
    parser.add_argument(
        "-numpath",
        dest="numpath",
        help="Default number of top paths considered for clustering.",
        required=False,
        default=500,
    )
    parser.add_argument(
        "-scale",
        dest="scale",
        help="Scales the radius of the json.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-json",
        dest="json",
        help="Json to scale.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-flat",
        dest="flat",
        help="Sets every radius to the input value.",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-clusterscale",
        dest="clusterscale",
        help="Input the maximum radius for the cluster based scaling.",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-recolor",
        dest="color",
        help="Changes the coloring of paths.",
        required=False,
        default=False,
    )

    # Gather input arguments
    args = parser.parse_args()

    if args.color and not args.json:
        print("\033[1mRecoloring requires a valid -json to recolor.\033[0m")
    if args.color and args.json:
        json_file = args.json
        color_file_path = args.color
        with open(color_file_path, 'r') as color_file:
            colors = json.load(color_file)
        with open(json_file, 'r') as file:
            data = json.load(file)
        cluster_colors = {}
        num_colors = len(colors)
        cluster_ids = {entry['clusterid'] for entry in data if 'clusterid' in entry}
        for i, cluster_id in enumerate(cluster_ids):
            cluster_colors[cluster_id] = colors[i % num_colors]
        for entry in data:
            if 'clusterid' in entry:
                clusterid = entry['clusterid']
                if clusterid in cluster_colors:
                    entry['color'] = cluster_colors[clusterid]
        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_recolored{ext}"
        with open(new_file, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Saved the modified data with colors to {new_file}.")
        exit()

    if args.scale and args.json and not args.flat and not args.clusterscale:
        json_file = args.json
        factor = float(args.scale)
        with open(json_file, 'r') as file:
            data = json.load(file)
        for entry in data:
            if 'radius' in entry:
                entry['radius'] *= factor
        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_scaled_{factor}{ext}"
        with open(new_file, 'w') as file:
            json.dump(data, file, indent=4)
        print("\033[1mSaved the modified data.\033[0m")
        exit()
    
    if not args.scale and args.json and args.flat and not args.clusterscale:
        json_file = args.json
        flat = float(args.flat)
        with open(json_file, 'r') as file:
            data = json.load(file)
        for entry in data:
            if 'radius' in entry:
                entry['radius'] = flat
        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_flat_{flat}{ext}"
        with open(new_file, 'w') as file:
            json.dump(data, file, indent=4)
        print("\033[1mSaved the modified data.\033[0m")
        exit()

    if not args.scale and not args.flat and args.json and args.clusterscale:
        json_file = args.json
        max_radius = float(args.clusterscale)
        with open(json_file, 'r') as file:
            data = json.load(file)
        cluster_counts = defaultdict(int)
        for entry in data:
            cluster_counts[entry['clusterid']] += 1
        max_clusterid = max(cluster_counts.keys())
        scaling_factors = {cid: (max_clusterid / cid) * max_radius for cid in cluster_counts}
        for entry in data:
            if 'radius' in entry:
                clusterid = entry['clusterid']
                if clusterid in scaling_factors:
                    entry['radius'] = scaling_factors[clusterid]

        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_cluster_scaled_{max_radius}{ext}"
        with open(new_file, 'w') as file:
            json.dump(data, file, indent=4)
        print("\033[1mSaved the modified data.\033[0m")
        exit()

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
                "Topology (residue_coordinates) and bcluster (cluster) are required and a json needed for comparing two simulations."
            )
            exit()

    if args.multitraj and args.topology:
        merged_data = []
        topology = args.topology
        num_parallel_processes = int(args.num_parallel_processes)
        closedist = float(args.closedist)
        first_res_num, last_res_num = res_num_from_pdb(topology)
        num_residues = last_res_num - first_res_num
        for filepath in args.multitraj:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            merged_data.extend(data)
        df_close_res = close_residues(topology, last_res_num, dist=closedist)
        overlap_df = calculate_overlap_parallel(
        merged_data, df_close_res, num_parallel_processes
    )
        overlap_df.to_csv('overlap_df.csv', index=False)
        clusters = pathways_cluster(overlap_df)
        cluster_pathways_dict = {}
        for cluster_num, cluster_pathways in clusters.items():
            cluster_pathways_list = []
            for pathway_id in cluster_pathways:
                pathway = merged_data[pathway_id]
                cluster_pathways_list.append(pathway)
            cluster_pathways_dict[cluster_num] = cluster_pathways_list

        residue_coordinates_dict = residue_CA_coordinates(topology, last_res_num)
        updated_dict = apply_backtracking(cluster_pathways_dict, residue_coordinates_dict)
        formated_dict = format_dict(updated_dict)
        with open("multitraj_clusters_paths.json", "w") as json_file:
            json.dump(formated_dict, json_file)
        path_properties = precompute_path_properties(formated_dict)
        with open("multitraj_precomputed_clusters_paths.json", "w") as out_file:
            json.dump(path_properties, out_file, indent=4)
        quick_path_properties = precompute_cluster_properties_quick(formated_dict)
        with open("multitraj_quick_precomputed_clusters_paths.json", "w") as out_file2:
            json.dump(quick_path_properties, out_file2, indent=4)
        print("\033[1mAnalyzed multiple trajectories.\033[0m")
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
    fardist = float(args.fardist)
    closedist = float(args.closedist)
    graphdist = float(args.graphdist)
    numpath = int(args.numpath)

    # Prepare the trajectory for analysis
    with mda.Writer("first_frame.pdb", multiframe=False) as pdb:
        traj.trajectory[0]
        pdb.write(traj.atoms)
    first_res_num, last_res_num = res_num_from_pdb("first_frame.pdb")
    num_residues = last_res_num - first_res_num
    df_distant_residues = faraway_residues(
        "first_frame.pdb", last_res_num, dist=fardist
    )
    df_close_res = close_residues("first_frame.pdb", last_res_num, dist=closedist)
    df_all_residues = calculate_dihedral_movement_parallel(
        num_parallel_processes, first_res_num, last_res_num, num_residues, traj
    )
    print("\033[1mTrajectory is processed and ready for analysis.\033[0m")

    # Calculate the mutual information and build the graph
    mi_diff_df = NMI_calc(df_all_residues, num_bins=35)
    mi_diff_df.to_csv("mi_diff_df.csv", index=False)
    residue_graph_empty = graph_building(
        "first_frame.pdb", last_res_num, dist=graphdist
    )
    residue_graph = graph_assign_weights(residue_graph_empty, mi_diff_df)
    visualise_graph(residue_graph)  # Exports image of the Graph to PNG

    # Calculate paths
    path_total_weights = collect_path_total_weights(residue_graph, df_distant_residues)
    sorted_paths = sorted(path_total_weights, key=lambda x: x[1], reverse=True)
    sorted_paths_bs = sorted_paths
    with open("output.txt", "w") as file:
        for path, total_weight in sorted_paths[:numpath]:
            file.write(f"Path: {path}, Total Weight: {total_weight}\n")
    top_pathways = [
        path for path, _ in sorted_paths[:numpath]
    ]  

    # Calculate the paths including ligand interacting residues
    if lig_interaction:
        try:
            lig_interaction = [int(res) for res in lig_interaction]
        except ValueError:
            print("Error: All -lig inputs must be integers.") 
        sorted_paths = [
        path for path in sorted_paths
        if any(residue in lig_interaction for residue in path[0])  
    ]
        top_pathways = [
            path for path, _ in sorted_paths[:numpath]
    ]
        print("\033[1mLigand pathways gathered..\033[0m")

    # Bootstrap analysis
    if bootstrap:
        num_bootstrap_samples = int(bootstrap)
        common_counts, path_confidence_intervals = bootstrap_analysis(
            df_all_residues,
            residue_graph_empty,
            df_distant_residues,
            sorted_paths_bs,
            num_bootstrap_samples,
            numpath,
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

    # Cluster pathways to get signaltransduction paths
    overlap_df = calculate_overlap_parallel(
        top_pathways, df_close_res, num_parallel_processes
    )
    clusters = pathways_cluster(overlap_df)
    cluster_pathways_dict = {}
    for cluster_num, cluster_pathways in clusters.items():
        cluster_pathways_list = []
        for pathway_id in cluster_pathways:
            pathway = sorted_paths[pathway_id]
            cluster_pathways_list.append(pathway[0])
        cluster_pathways_dict[cluster_num] = cluster_pathways_list
    residue_coordinates_dict = residue_CA_coordinates("first_frame.pdb", last_res_num)

    # Export residue coordinates and pathways dict for comparisson functionality
    with open("residue_coordinates.pkl", "wb") as pkl_file:
        pickle.dump(residue_coordinates_dict, pkl_file)

    with open("cluster_pathways_dict.pkl", "wb") as pkl_file:
        pickle.dump(cluster_pathways_dict, pkl_file)
    
    with open("top_pathways.pkl", "wb") as pkl_file:
        pickle.dump(top_pathways, pkl_file)

    # Export the cluster pathways for visualization
    updated_dict = apply_backtracking(cluster_pathways_dict, residue_coordinates_dict)
    formated_dict = format_dict(updated_dict)
    with open("clusters_paths.json", "w") as json_file:
        json.dump(formated_dict, json_file)
    path_properties = precompute_path_properties(formated_dict)
    with open("precomputed_clusters_paths.json", "w") as out_file:
        json.dump(path_properties, out_file, indent=4)
    quick_path_properties = precompute_cluster_properties_quick(formated_dict)
    with open("quick_precomputed_clusters_paths.json", "w") as out_file2:
        json.dump(quick_path_properties, out_file2, indent=4)


if __name__ == "__main__":
    main()
