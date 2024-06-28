from Bio import PDB
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
            res_id = ("", residue, "")
            try:
                res = structure[0][res_id]
                atom = res["CA"]
                coord = atom.get_coord()
                pathways.append(coord)
            except KeyError:
                print(res + " not found.")
            cluster.append(pathways)
    return cluster


def apply_backtracking(original_dict, translation_dict):
    updated_dict = original_dict.copy()

    for key, lists_of_lists in original_dict.items():
        for i, inner_list in enumerate(lists_of_lists):
            for j, item in enumerate(inner_list):
                if item in translation_dict:
                    updated_dict[key][i][j] = translation_dict[item]

    return updated_dict


def format_dict(updated_dict):
    def transform_list(nested_list):
        transformed = []
        for item in nested_list:
            if isinstance(item, np.ndarray):
                transformed.append(item.tolist())
            elif isinstance(item, list):
                transformed.append(transform_list(item))  # Append instead of extend
            else:
                transformed.append(item)
        return transformed

    transformed_dict = {
        key: transform_list(value) for key, value in updated_dict.items()
    }
    return transformed_dict

def visualise_graph(graph, k=0.1, node_size=200):
    labels = {i: str(i) for i in graph.nodes()}
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(graph, k=k)  
    nx.draw(graph, pos, node_size=node_size, with_labels=True, labels=labels, font_size=8, edge_color='gray', node_color='blue')
    plt.savefig('graph.png', dpi=300, bbox_inches='tight')



def precompute_path_properties(json_data, colors):
    cluster_colors = {}
    color_index = 0
    path_properties = []

    for clusterid, cluster in json_data.items():
        cluster_colors[clusterid] = colors[color_index % len(colors)]
        color_index += 1
        coord_pair_counts = {}
        path_number = 1

        for pathway_index, pathway in enumerate(cluster):
            for i in range(len(pathway) - 1):
                coord1 = pathway[i][0]
                coord2 = pathway[i + 1][0]
                if isinstance(coord1, list) and isinstance(coord2, list) and len(coord1) == 3 and len(coord2) == 3:
                    coord_pair = (tuple(coord1), tuple(coord2))
                    if coord_pair not in coord_pair_counts:
                        coord_pair_counts[coord_pair] = 0
                    coord_pair_counts[coord_pair] += 1
                    radius = 0.015 + 0.015 * (coord_pair_counts[coord_pair] - 1)
                    color = cluster_colors[clusterid]

                    path_properties.append({
                        "clusterid": clusterid,
                        "pathway_index": pathway_index,
                        "path_segment_index": i,
                        "coord1": coord1,
                        "coord2": coord2,
                        "color": color,
                        "radius": radius,
                        "path_number": path_number
                    })

                    path_number += 1
                else:
                    print(f"Ignoring pathway {pathway} as it does not fulfill the coordinate format.")
    return path_properties

def precompute_cluster_properties_quick(json_data, colors):
    cluster_colors = {}
    color_index = 0
    cluster_properties = []

    for clusterid, cluster in json_data.items():
        cluster_colors[clusterid] = colors[color_index % len(colors)]
        color_index += 1
        coord_pair_counts = {}

        for pathway_index, pathway in enumerate(cluster):
            for i in range(len(pathway) - 1):
                coord1 = pathway[i][0]
                coord2 = pathway[i + 1][0]
                if isinstance(coord1, list) and isinstance(coord2, list) and len(coord1) == 3 and len(coord2) == 3:
                    coord_pair = (tuple(coord1), tuple(coord2))
                    if coord_pair not in coord_pair_counts:
                        coord_pair_counts[coord_pair] = 0
                    coord_pair_counts[coord_pair] += 1
                    radius = 0.015 + 0.015 * (coord_pair_counts[coord_pair] - 1)
                    color = cluster_colors[clusterid]

                    cluster_properties.append({
                        "clusterid": clusterid,
                        "coord1": coord1,
                        "coord2": coord2,
                        "color": color,
                        "radius": radius
                    })
                else:
                    print(f"Ignoring pathway {pathway} as it does not fulfill the coordinate format.")
    return cluster_properties
