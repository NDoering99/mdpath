from Bio import PDB
from tqdm import tqdm
import pandas as pd
import numpy as np

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
    
    transformed_dict = {key: transform_list(value) for key, value in updated_dict.items()}
    return transformed_dict
