from tqdm import tqdm
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from multiprocessing import Pool
from Bio import PDB
from itertools import combinations


    

def res_num_from_pdb(pdb: str) -> tuple[int, int]:
    """Gets first and last residue number from a PDB file.

    Args:
        pdb (str): Path to PDB file.

    Returns:
        first_res_num (int): First residue number.
        last_res_num (int): Last residue number.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb)
    first_res_num = float("inf")
    last_res_num = float("-inf")
    for res in structure.get_residues():
        if PDB.Polypeptide.is_aa(res):
            res_num = res.id[1]
            if res_num < first_res_num:
                first_res_num = res_num
            if res_num > last_res_num:
                last_res_num = res_num
    return int(first_res_num), int(last_res_num)


def calc_dihedral_angle_movement(i: int, traj: mda.Universe) -> tuple[int, np.array]:
    """Calculates dihedral angle movement for a residue over the cours of the MD trajectory.

    Args:
        i (int): Residue number.
        traj (mda.Universe): MDAnalysis Universe object containing the trajectory.

    Returns:
        i (int): Residue number.
        dihedral_angle_movement (np.array): Dihedral angle movement for the residue over the course of the trajectory.
    """
    res = traj.residues[i]
    ags = [res.phi_selection()]
    R = Dihedral(ags).run()
    dihedrals = R.results.angles
    dihedral_angle_movement = np.diff(dihedrals, axis=0)
    return i, dihedral_angle_movement


def calc_dihedral_angle_movement_wrapper(
    args: tuple[int, mda.Universe]
) -> tuple[int, np.array]:
    """Wrapper function for calculating dihedral angle movement for a residue over the course of the MD trajectory.

    Args:
        args (tuple[int, mda.Universe]): Tuple containing residue number and MDAnalysis Universe object.

    Returns:
        i (int): Residue number.
        dihedral_angle_movement (np.array): Dihedral angle movement for the residue over the course of the trajectory.
    """
    residue_id, traj = args
    return calc_dihedral_angle_movement(residue_id, traj)


def update_progress(res: tqdm) -> tqdm:
    """Update progress bar.

    Args:
        res (tqdm): TQDM progress bar object.

    Returns:
        res: TQDM progress bar object.
    """
    res.update()
    return res


def calculate_dihedral_movement_parallel(
    num_parallel_processes: int,
    first_res_num: int,
    last_res_num: int,
    num_residues: int,
    traj: mda.Universe,
) -> pd.DataFrame:
    """Parallel calculation of dihedral angle movement for all residues in the trajectory.

    Args:
        num_parallel_processes (int): Amount of parallel processes.
        first_res_num (int): First residue number.
        last_res_num (int): Last residue number.
        num_residues (int): Amount of residues.
        traj (mda.Universe): MDAnalysis Universe object containing the trajectory.

    Returns:
        df_all_residues (pd.DataFrame): Pandas dataframe with all residue dihedral angle movements.
    """
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


def calculate_distance(atom1: int, atom2: int) -> float:
    """Calculates the distance between two atoms.

    Args:
        atom1 (tuple[float]): Coordinates of the first atom.
        atom2 (tuple[float]): Coordinates of the second atom.

    Returns:
        distance (float): Normalized distance between the two atoms.
    """
    distance_vector = [atom1[i] - atom2[i] for i in range(min(len(atom1), len(atom2)))]
    distance = np.linalg.norm(distance_vector)
    return distance


def faraway_residues(pdb_file: str, end: int, dist=12.0) -> pd.DataFrame:
    """Calculates residues that are far away from each other in a PDB structure.

    Args:
        pdb_file (str): Path to PDB file.
        end (int): Last residue number.
        dist (float, optional): Distance cutoff for faraway residues. Defaults to 12.0.

    Returns:
        pd.DataFrame: Pandas dataframe with faraway residue pairs and their distance.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    heavy_atoms = ["C", "N", "O", "S"]
    distant_residues = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            for res1, res2 in tqdm(
                combinations(residues, 2),
                desc="Calculating distant residues",
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


def close_residues(pdb_file: str, end: int, dist=10.0) -> pd.DataFrame:
    """Calculates residues that are close to each other in a PDB structure.

    Args:
        pdb_file (str): Path to PDB file.
        end (int): Last residue number.
        dist (float, optional): Distance cutoff for close residues. Defaults to 10.0.

    Returns:
        pd.DataFrame: Pandas dataframe with close residue pairs and their distance.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    heavy_atoms = ["C", "N", "O", "S"]
    close_residues = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            for res1, res2 in tqdm(
                combinations(residues, 2),
                desc="Calculating close residues",
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
