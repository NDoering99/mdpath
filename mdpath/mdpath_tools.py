import os
import argparse
import pandas as pd
import numpy as np
import MDAnalysis as mda
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pickle

from mdpath.src.structure import StructureCalculations
from mdpath.src.cluster import PatwayClustering
from mdpath.src.visualization import MDPathVisualize


def edit_3D_visualization_json():
    parser = argparse.ArgumentParser(
        prog="mdpath_json_edit",
        description="Edit the 3D visualization JSONS to your visualization needs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-json",
        dest="json",
        help="Json file to edit.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-scale",
        dest="scale",
        help="Factor to mutiply radius with",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-recolor",
        dest="color",
        help="Color file for 3D visualization colors.",
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
    args = parser.parse_args()

    # Recoloring
    if args.color and not args.json:
        print("\033[1mRecoloring requires a valid -json to recolor.\033[0m")
    if args.color and args.json:
        json_file = args.json
        color_file_path = args.color
        with open(color_file_path, "r") as color_file:
            colors = json.load(color_file)
        with open(json_file, "r") as file:
            data = json.load(file)
        cluster_colors = {}
        num_colors = len(colors)
        cluster_ids = {entry["clusterid"] for entry in data if "clusterid" in entry}
        for i, cluster_id in enumerate(cluster_ids):
            cluster_colors[cluster_id] = colors[i % num_colors]
        for entry in data:
            if "clusterid" in entry:
                clusterid = entry["clusterid"]
                if clusterid in cluster_colors:
                    entry["color"] = cluster_colors[clusterid]
        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_recolored{ext}"
        with open(new_file, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Saved the modified data with colors to {new_file}.")
        exit()
    # Scaling
    if args.scale and args.json and not args.flat and not args.clusterscale:
        json_file = args.json
        factor = float(args.scale)
        with open(json_file, "r") as file:
            data = json.load(file)
        for entry in data:
            if "radius" in entry:
                entry["radius"] *= factor
        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_scaled_{factor}{ext}"
        with open(new_file, "w") as file:
            json.dump(data, file, indent=4)
        print("\033[1mSaved the modified data.\033[0m")
        exit()

    if not args.scale and args.json and args.flat and not args.clusterscale:
        json_file = args.json
        flat = float(args.flat)
        with open(json_file, "r") as file:
            data = json.load(file)
        for entry in data:
            if "radius" in entry:
                entry["radius"] = flat
        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_flat_{flat}{ext}"
        with open(new_file, "w") as file:
            json.dump(data, file, indent=4)
        print("\033[1mSaved the modified data.\033[0m")
        exit()

    if not args.scale and not args.flat and args.json and args.clusterscale:
        json_file = args.json
        max_radius = float(args.clusterscale)
        with open(json_file, "r") as file:
            data = json.load(file)
        cluster_counts = defaultdict(int)
        for entry in data:
            cluster_counts[entry["clusterid"]] += 1
        max_clusterid = max(cluster_counts.keys())
        scaling_factors = {
            cid: (max_clusterid / cid) * max_radius for cid in cluster_counts
        }
        for entry in data:
            if "radius" in entry:
                clusterid = entry["clusterid"]
                if clusterid in scaling_factors:
                    entry["radius"] = scaling_factors[clusterid]

        base, ext = os.path.splitext(json_file)
        new_file = f"{base}_cluster_scaled_{max_radius}{ext}"
        with open(new_file, "w") as file:
            json.dump(data, file, indent=4)
        print("\033[1mSaved the modified data.\033[0m")
        exit()


def path_comparison():
    parser = argparse.ArgumentParser(
        prog="mdpath_compare",
        description="Compare Pathways of different simulations",
        formatter_class=argparse.RawTextHelpFormatter,
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
    if args.atop and args.bcluster:
        with open(args.atop, "rb") as pkl_file:
            residue_coordinates_dict = pickle.load(pkl_file)
        with open(args.bcluster, "rb") as pkl_file:
            cluster_pathways_dict = pickle.load(pkl_file)
        updated_dict = MDPathVisualize.apply_backtracking(
            cluster_pathways_dict, residue_coordinates_dict
        )
        formatted_dict = MDPathVisualize.format_dict(updated_dict)
        with open("morphed_clusters_paths.json", "w") as json_file_2:
            json.dump(formatted_dict, json_file_2)
        exit()
    else:
        print(
            "Topology (residue_coordinates) and bcluster (cluster) are required and a json needed for comparing two simulations."
        )
        exit()


def multitraj_analysis():
    parser = argparse.ArgumentParser(
        prog="mdpath_multitraj",
        description="Merge and analyze multiple trajectories.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-top",
        dest="topology",
        help="Topology file of your MD simulation",
        required=False,
    )
    parser.add_argument(
        "-multitraj",
        dest="multitraj",
        help="List of multiple pathways from previous analysis.",
        default=False,
        nargs="+",
    )
    args = parser.parse_args()
    if args.multitraj and args.topology:
        merged_data = []
        topology = args.topology
        num_parallel_processes = int(args.num_parallel_processes)
        closedist = float(args.closedist)
        structure_calc = StructureCalculations(topology)
        for filepath in args.multitraj:
            with open(filepath, "rb") as file:
                data = pickle.load(file)
            merged_data.extend(data)
        df_close_res = structure_calc.calculate_residue_suroundings(closedist, "close")
        clustering = PatwayClustering(df_close_res, merged_data, num_parallel_processes)
        clustering.overlap_df.to_csv("overlap_df.csv", index=False)
        clusters = clustering.pathways_cluster()
        cluster_pathways_dict = {}
        for cluster_num, cluster_pathways in clusters.items():
            cluster_pathways_list = []
            for pathway_id in cluster_pathways:
                pathway = merged_data[pathway_id]
                cluster_pathways_list.append(pathway)
            cluster_pathways_dict[cluster_num] = cluster_pathways_list

        residue_coordinates_dict = MDPathVisualize.residue_CA_coordinates(
            topology, structure_calc.last_res_num
        )
        updated_dict = MDPathVisualize.apply_backtracking(
            cluster_pathways_dict, residue_coordinates_dict
        )
        formated_dict = MDPathVisualize.format_dict(updated_dict)
        with open("multitraj_clusters_paths.json", "w") as json_file:
            json.dump(formated_dict, json_file)
        path_properties = MDPathVisualize.precompute_path_properties(formated_dict)
        with open("multitraj_precomputed_clusters_paths.json", "w") as out_file:
            json.dump(path_properties, out_file, indent=4)
        quick_path_properties = MDPathVisualize.precompute_cluster_properties_quick(
            formated_dict
        )
        with open("multitraj_quick_precomputed_clusters_paths.json", "w") as out_file2:
            json.dump(quick_path_properties, out_file2, indent=4)
        print("\033[1mAnalyzed multiple trajectories.\033[0m")
        exit()
    else:
        print(
            "Topology and pathways are required for merging and analyzing multiple trajectories."
        )
        exit()


def gpcr_2D_vis():
    parser = argparse.ArgumentParser(
        prog="mdpath_gpcr",
        description="Create a 2D Visualization of Paths throught a GPCR.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-top",
        dest="topology",
        help="Topology file of your MD simulation",
        required=False,
    )
    parser.add_argument(
        "-clust",
        dest="cluster_pathways",
        help="Picle file with cluster pathways dict.",
        required=True,
    )
    parser.add_argument(
        "-cut",
        dest="cutoff",
        help="Percentage of the top paths to visualize.",
        default=0,
        required=False,
    )
    parser.add_argument(
        "-num",
        dest="numberd_structure",
        help="Path to write the numbered structure file to.",
        default="./numbered_structure.pdb",
        required=False,
    )
    args = parser.parse_args()
    if args.topology and args.cluster_pathways:
        args = parser.parse_args()
        topology = args.topology
        cluster_pathways_file = args.cluster_pathways
        cutoff_percentage = float(args.cutoff)
        numberd_structure = args.numberd_structure

        MDPathVisualize.remove_non_protein(topology, "./protein.pdb")
        MDPathVisualize.assign_generic_numbers(
            "./protein.pdb", output_file_path=numberd_structure
        )
        os.remove("./protein.pdb")
        with open(cluster_pathways_file, "rb") as f:
            cluster_pathways = pickle.load(f)
        generic_number_dict = MDPathVisualize.parse_pdb_and_create_dictionary(
            numberd_structure
        )
        updated_cluster_residues, no_genetic_numbers_found = (
            MDPathVisualize.assign_generic_numbers_paths(
                cluster_pathways, generic_number_dict
            )
        )
        MDPathVisualize.create_gpcr_2d_path_vis(
            updated_cluster_residues, cutoff_percentage=cutoff_percentage
        )
        exit()
    else:
        print(
            "Topology and cluster pathways are required for creating a 2D visualization of GPCR paths."
        )
        exit()
