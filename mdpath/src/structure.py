from tqdm import tqdm
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from multiprocessing import Pool
from Bio import PDB
from itertools import combinations


def res_num_from_pdb(pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb)
    first_res_num = float('inf')
    last_res_num = float('-inf')
    for res in structure.get_residues():
        if PDB.Polypeptide.is_aa(res):
            res_num = res.id[1]
            if res_num < first_res_num:
                first_res_num = res_num
            if res_num > last_res_num:
                last_res_num = res_num
    return int(first_res_num), int(last_res_num)



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


def calculate_dihedral_movement_parallel(
    num_parallel_processes, first_res_num, last_res_num, num_residues, traj
):
    try:
        with Pool(processes=num_parallel_processes) as pool:
            residue_args = [(i, traj) for i in range(first_res_num, last_res_num + 1)]
            df_all_residues = pd.DataFrame()
            with tqdm(
                total=num_residues,
                ascii=True,
                desc="Processing residue dihedral movements: ",
            ) as pbar:
                for res_id, result in pool.imap_unordered(
                    calc_dihedral_angle_movement_wrapper, residue_args
                ):
                    try:
                        df_residue = pd.DataFrame(result, columns=[f"Res {res_id}"])
                        df_all_residues = pd.concat(
                            [df_all_residues, df_residue], axis=1
                        )
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing residue {res_id}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return df_all_residues


def calculate_distance(atom1, atom2):
    distance_vector = atom1 - atom2
    distance = np.linalg.norm(distance_vector)
    return distance


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
