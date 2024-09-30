"""MDPath --- MD signal transduction calculation and visualization --- :mod:`mdpath.mdapth`
====================================================================

MDPath is a Python package for calculating signal transduction paths in molecular dynamics (MD) simulations. 
The package uses mutual information to identify connections between residue movements.
Using a graph shortest paths with the highest mutual information are calculated.
Paths are then clustered based on the overlap between them to identify a continuous network throught the analysed protein.
The package also includes functionalitys for the visualization of results.

Release under the MIT License. See LICENSE for details.

This is the main script of MDPath. It is used to run MDPath from the command line.
MDPath can be called from the comadline using 'mdapth' after instalation
Use the -h flag to see the available options.

Functions
----------

:func:`main` - Main function for running MDPath from the command line.
"""

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

from mdpath.src.structure import StructureCalculations, DihedralAngles
from mdpath.src.mutual_information import NMICalculator
from mdpath.src.graph import GraphBuilder
from mdpath.src.cluster import PatwayClustering
from mdpath.src.visualization import MDPathVisualize
from mdpath.src.bootstrap import BootstrapAnalysis


def main():
    """Main function for running MDPath from the command line.
    It can be called using 'mdpath' after installation.
    
    Command-line inputs:
    -top: Topology file of your MD simulation
    -traj: Trajectory file of your MD simulation
    -cpu: Amount of cores used in multiprocessing (default: half of available cores)
    -lig: Protein ligand interacting residues (default: False)
    -bs: How often bootstrapping should be performed (default: False)
    -fardist: Default distance for faraway residues (default: 12.0)
    -closedist: Default distance for close residues (default: 12.0)
    -graphdist: Default distance for residues making up the graph (default: 5.0)
    -numpath: Default number of top paths considered for clustering (default: 500).
    """
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
        nargs="+",
    )
    parser.add_argument(
        "-bs",
        dest="bootstrap",
        help="How often bootstrapping should be performed.",
        default=False,
    )
    # TODO maybe move settingsflags to a conffile that can be changed
    # Settings Flags
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
        "-GMM",
        dest="GMM",
        help="Histograms are created using GMM instead of binning.",
        required=False,
        default=False,
    )

    args = parser.parse_args()
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
    GMM = bool(args.GMM)

    # Prepare the trajectory for analysis
    with mda.Writer("first_frame.pdb", multiframe=False) as pdb:
        traj.trajectory[0]
        pdb.write(traj.atoms)
    structure_calc = StructureCalculations(topology)
    df_distant_residues = structure_calc.calculate_residue_suroundings(fardist, "far")
    df_close_res = structure_calc.calculate_residue_suroundings(closedist, "close")
    dihedral_calc = DihedralAngles(
        traj,
        structure_calc.first_res_num,
        structure_calc.last_res_num,
        structure_calc.last_res_num,
    )
    df_all_residues = dihedral_calc.calculate_dihedral_movement_parallel(
        num_parallel_processes
    )
    print("\033[1mTrajectory is processed and ready for analysis.\033[0m")

   # Calculate the mutual information and build the graph
    nmi_calc = NMICalculator(df_all_residues, GMM = GMM)
    nmi_calc.mi_diff_df.to_csv("mi_diff_df.csv", index=False)
    graph_builder = GraphBuilder(
        topology, structure_calc.last_res_num, nmi_calc.mi_diff_df, graphdist
    )
    MDPathVisualize.visualise_graph(
        graph_builder.graph
    )  # Exports image of the Graph to PNG

    # Calculate paths
    path_total_weights = graph_builder.collect_path_total_weights(df_distant_residues)
    sorted_paths = sorted(path_total_weights, key=lambda x: x[1], reverse=True)
    with open("output.txt", "w") as file:
        for path, total_weight in sorted_paths[:numpath]:
            file.write(f"Path: {path}, Total Weight: {total_weight}\n")
    top_pathways = [path for path, _ in sorted_paths[:numpath]]

    # Calculate the paths including ligand interacting residues
    if lig_interaction:
        try:
            lig_interaction = [int(res) for res in lig_interaction]
        except ValueError:
            print("Error: All -lig inputs must be integers.")
        sorted_paths = [
            path
            for path in sorted_paths
            if any(residue in lig_interaction for residue in path[0])
        ]
        top_pathways = [path for path, _ in sorted_paths[:numpath]]
        print("\033[1mLigand pathways gathered..\033[0m")

    # Bootstrap analysis
    if bootstrap:
        num_bootstrap_samples = int(bootstrap)
        bootstrap_analysis = BootstrapAnalysis(
            df_all_residues,
            df_distant_residues,
            sorted_paths,
            num_bootstrap_samples,
            numpath,
            topology,
            structure_calc.last_res_num,
            graphdist,
        )
        file_name = "path_confidence_intervals.txt"
        bootstrap_analysis.bootstrap_write(file_name)
        print(f"Path confidence intervals have been saved to {file_name}")

    # Cluster pathways to get signaltransduction paths
    clustering = PatwayClustering(df_close_res, top_pathways, num_parallel_processes)
    clusters = clustering.pathways_cluster()
    cluster_pathways_dict = clustering.pathway_clusters_dictionary(
        clusters, sorted_paths
    )
    residue_coordinates_dict = MDPathVisualize.residue_CA_coordinates(
        "first_frame.pdb", structure_calc.last_res_num
    )

    # Export residue coordinates and pathways dict for comparisson functionality
    with open("residue_coordinates.pkl", "wb") as pkl_file:
        pickle.dump(residue_coordinates_dict, pkl_file)

    with open("cluster_pathways_dict.pkl", "wb") as pkl_file:
        pickle.dump(cluster_pathways_dict, pkl_file)

    with open("top_pathways.pkl", "wb") as pkl_file:
        pickle.dump(top_pathways, pkl_file)

    # Export the cluster pathways for visualization
    updated_dict = MDPathVisualize.apply_backtracking(
        cluster_pathways_dict, residue_coordinates_dict
    )
    formated_dict = MDPathVisualize.format_dict(updated_dict)
    with open("clusters_paths.json", "w") as json_file:
        json.dump(formated_dict, json_file)
    path_properties = MDPathVisualize.precompute_path_properties(formated_dict)
    with open("precomputed_clusters_paths.json", "w") as out_file:
        json.dump(path_properties, out_file, indent=4)
    quick_path_properties = MDPathVisualize.precompute_cluster_properties_quick(
        formated_dict
    )
    with open("quick_precomputed_clusters_paths.json", "w") as out_file2:
        json.dump(quick_path_properties, out_file2, indent=4)


if __name__ == "__main__":
    main()
